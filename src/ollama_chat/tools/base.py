from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from functools import cached_property
import logging
from typing import Any

from pydantic import BaseModel, ValidationError

from .truncation import truncate_output

LOGGER = logging.getLogger(__name__)


@dataclass
class Attachment:
    type: str  # "file"
    mime: str
    url: str  # data:<mime>;base64,<b64> or https://...


@dataclass
class ToolResult:
    title: str
    output: str
    metadata: dict[str, Any]
    attachments: list[Attachment] = field(default_factory=list)


@dataclass
class ToolContext:
    session_id: str
    message_id: str
    agent: str
    abort: asyncio.Event  # set when the user aborts
    call_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    messages: list[Any] = field(default_factory=list)

    # Injected by the runtime:
    _metadata_cb: Callable[[dict], None] = field(default=lambda _: None, repr=False)
    _ask_cb: Callable[..., Awaitable[None]] | None = field(default=None, repr=False)

    def metadata(self, title: str | None = None, metadata: dict | None = None) -> None:
        """Update live streaming metadata visible in the UI."""
        self._metadata_cb({"title": title, "metadata": metadata or {}})

    async def ask(
        self,
        permission: str,
        patterns: list[str],
        always: list[str],
        metadata: dict,
    ) -> None:
        """
        Request user approval.
        Raises PermissionDeniedError / PermissionRejectedError on denial.
        """
        if self._ask_cb:
            await self._ask_cb(
                permission=permission,
                patterns=patterns,
                always=always,
                metadata=metadata,
                session_id=self.session_id,
            )

    def with_call_id(self, call_id: str) -> ToolContext:
        """Return a shallow-copied context with a different call_id."""
        return replace(self, call_id=call_id)

    @property
    def project_root(self) -> Path:
        """Get resolved project directory.

        Replaces 16+ duplicate Path(...).expanduser().resolve() calls.
        Added during Phase 1.2 refactoring.
        """
        from pathlib import Path

        root = self.extra.get("project_dir", ".")
        return Path(str(root)).expanduser().resolve()

    def resolve_path(self, path: str | Path) -> Path:
        """Resolve path relative to project root.

        Args:
            path: Absolute or relative path

        Returns:
            Resolved absolute path

        Added during Phase 1.2 refactoring to eliminate duplicate path resolution.
        """
        from pathlib import Path

        p = Path(path).expanduser()
        if not p.is_absolute():
            p = self.project_root / p
        return p.resolve()

    async def check_permission(
        self,
        operation: str,
        paths: list[Path],
        *,
        allow_workspace: bool = True,
    ) -> None:
        """Centralized permission checks for tool operations."""
        from .external_directory import assert_external_directory

        project_root = self.project_root
        for path in paths:
            resolved = path.resolve()
            if allow_workspace:
                try:
                    resolved.relative_to(project_root)
                    continue
                except Exception:
                    pass

            await assert_external_directory(self, str(resolved))
            await self.ask(
                permission=operation,
                patterns=[str(resolved)],
                always=["*"],
                metadata={"operation": operation, "path": str(resolved)},
            )


class ParamsSchema(BaseModel):
    """Base class for all tool parameter schemas."""

    model_config = {"extra": "ignore"}


class Tool(ABC):
    """
    Abstract base class for all tools.

    Subclasses set:
        id          – unique tool name (matches permission key)
        description – shown to the LLM
        params_schema – a ParamsSchema subclass

    The base run() helper performs:
      1. Validate params via the Pydantic schema
      2. Call the concrete execute()
      3. Apply output truncation (truncate_output()) unless the result
         already has metadata["truncated"] set
    """

    id: str
    description: str = ""
    params_schema: type[ParamsSchema]

    @abstractmethod
    async def execute(
        self, params: ParamsSchema, ctx: ToolContext
    ) -> ToolResult:  # pragma: no cover - interface only
        ...

    @cached_property
    def _schema_cache(self) -> dict[str, Any]:
        """Cached schema generation - computed once per tool instance."""
        raw_schema = self.params_schema.model_json_schema()
        return self._clean_pydantic_schema(raw_schema)

    @staticmethod
    def _clean_pydantic_schema(schema: dict[str, Any]) -> dict[str, Any]:
        """Remove Pydantic-specific fields that Ollama doesn't need."""
        # Remove top-level keys Ollama ignores
        cleaned = {
            k: v
            for k, v in schema.items()
            if k not in ("$defs", "title", "$schema", "definitions")
        }

        # Ensure required fields exist
        cleaned.setdefault("type", "object")
        cleaned.setdefault("properties", {})
        cleaned.setdefault("required", [])

        # Ollama expects additionalProperties to be present
        cleaned.setdefault("additionalProperties", True)

        return cleaned

    def schema(self) -> dict:
        """Return the inner function schema (legacy method - DEPRECATED).

        Use to_ollama_schema() instead for proper Ollama format.
        This method is kept for backward compatibility but will be removed
        in a future version.
        """
        return {
            "name": self.id,
            "description": self.description,
            "parameters": self._schema_cache,
        }

    def to_ollama_schema(self) -> dict[str, Any]:
        """Return Ollama-compatible tool schema with proper wrapping.

        Returns:
            {
                "type": "function",
                "function": {
                    "name": "tool_name",
                    "description": "...",
                    "parameters": {...}
                }
            }
        """
        return {
            "type": "function",
            "function": {
                "name": self.id,
                "description": self.description,
                "parameters": self._schema_cache,
            },
        }

    def format_validation_error(self, error: Exception) -> str | None:
        """Override to provide custom error messages for schema validation failures."""
        return None

    async def run(self, raw_params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        """Validate, execute, and apply truncation to the tool output."""
        try:
            params = self.params_schema.model_validate(raw_params)
        except ValidationError as exc:  # pragma: no cover - defensive
            # Log validation errors for debugging
            LOGGER.warning(
                f"Tool '{self.id}' validation error",
                extra={
                    "event": "tool.validation_error",
                    "tool_id": self.id,
                    "session_id": ctx.session_id,
                    "agent": ctx.agent,
                    "error": str(exc),
                    "raw_params": raw_params,
                },
            )
            msg = self.format_validation_error(exc) or str(exc)
            return ToolResult(
                title=f"{self.id}: invalid parameters",
                output=msg,
                metadata={"ok": False, "validation_error": True},
            )

        result = await self.execute(params, ctx)
        # Respect explicit truncation metadata from the tool implementation.
        if not result.metadata.get("truncated"):
            trunc = await truncate_output(result.output, agent=ctx.agent)
            result.output = trunc.content
            # Merge truncation metadata non-destructively.
            result.metadata = {**result.metadata, "truncated": trunc.truncated}
            if trunc.output_path:
                result.metadata.setdefault("output_path", trunc.output_path)
        return result
