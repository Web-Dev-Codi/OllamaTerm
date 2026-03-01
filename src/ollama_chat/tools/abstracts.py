"""Abstract base classes for common tool patterns.

These classes provide reusable patterns for tool implementation,
eliminating duplication and enforcing consistency.

"""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path

from .base import ParamsSchema, Tool, ToolContext, ToolResult
from .utils import check_file_safety, notify_file_change

try:
    from pydantic import Field
except ImportError:
    # Fallback if pydantic Field not available
    Field = lambda *args, **kwargs: kwargs.get("default")  # noqa: E731


class FileOperationParams(ParamsSchema):
    """Base parameters for file operations."""

    file_path: str


class FileOperationTool(Tool):
    """Base class for tools that operate on files.

    Provides:
    - Automatic path resolution
    - Safety checks (external directory, concurrent access)
    - File change notifications
    - Consistent error handling

    Subclasses only implement perform_operation() with core logic.
    """

    async def execute(
        self, params: FileOperationParams, ctx: ToolContext
    ) -> ToolResult:
        """Execute file operation with safety checks and notifications."""
        # Resolve path using ToolContext helper
        file_path = ctx.resolve_path(params.file_path)

        # Centralized permission check (external directory gating only).
        await ctx.check_permission(self.id, [file_path])

        # Safety checks (modified-since-read and other invariants)
        try:
            await check_file_safety(
                file_path,
                ctx,
                check_external=True,
                assert_not_modified=getattr(params, "require_unchanged", False),
            )
        except Exception as exc:
            return ToolResult(
                title=str(file_path),
                output=f"Safety check failed: {exc}",
                metadata={"ok": False, "error": "permission_denied"},
            )

        # Perform the actual operation
        try:
            result = await self.perform_operation(file_path, params, ctx)
        except Exception as exc:
            return ToolResult(
                title=str(file_path),
                output=f"Operation failed: {exc}",
                metadata={"ok": False, "error": str(exc)},
            )

        # Notify file change if successful
        if result.metadata.get("ok", False):
            event_type = result.metadata.get("event", "change")
            try:
                await notify_file_change(file_path, event_type, ctx)
            except Exception:
                pass  # Don't fail operation due to notification error

        return result

    @abstractmethod
    async def perform_operation(
        self,
        file_path: Path,
        params: FileOperationParams,
        ctx: ToolContext,
    ) -> ToolResult:
        """Perform the actual file operation.

        Subclasses implement this method with the core logic.
        Path resolution, safety checks, and notifications are handled by execute().

        Args:
            file_path: Resolved absolute path
            params: Tool parameters
            ctx: Tool execution context

        Returns:
            ToolResult with operation outcome
        """
        ...


class SearchParams(ParamsSchema):
    """Base parameters for search operations."""

    pattern: str
    path: str = "."


class SearchTool(Tool):
    """Base class for search operations (grep, glob, find, etc.).

    Provides:
    - Path resolution
    - Safety checks
    - Consistent result formatting
    - Error handling

    Example:
        class GrepTool(SearchTool):
            id = "grep"
            params_schema = GrepParams

            async def perform_search(self, path, params, ctx):
                # Execute ripgrep or grep
                results = await run_grep(path, params.pattern)
                return results
    """

    async def execute(self, params: SearchParams, ctx: ToolContext) -> ToolResult:
        """Execute search with common handling."""
        # Resolve search path (not all tools provide path; default to ".")
        search_path = ctx.resolve_path(getattr(params, "path", ".") or ".")

        # Centralized permission check
        await ctx.check_permission(self.id, [search_path])

        # Compute a helpful title detail: prefer pattern, fall back to path
        detail = getattr(params, "pattern", None)
        if detail is None:
            detail = getattr(params, "path", None)
        title = f"{self.id}: {detail}" if detail else self.id

        # Safety check
        try:
            await check_file_safety(
                search_path,
                ctx,
                check_external=True,
                assert_not_modified=False,
            )
        except Exception as exc:
            return ToolResult(
                title=title,
                output=f"Permission denied: {exc}",
                metadata={"ok": False, "count": 0},
            )

        # Perform search
        try:
            results = await self.perform_search(search_path, params, ctx)
        except Exception as exc:
            return ToolResult(
                title=title,
                output=f"Search failed: {exc}",
                metadata={"ok": False, "count": 0},
            )

        # Format result
        meta = {
            "ok": True,
            "count": len(results.split("\n")) if results else 0,
        }
        if getattr(params, "pattern", None) is not None:
            meta["pattern"] = params.pattern
        return ToolResult(
            title=title,
            output=results,
            metadata=meta,
        )

    @abstractmethod
    async def perform_search(
        self,
        path: Path,
        params: SearchParams,
        ctx: ToolContext,
    ) -> str:
        """Perform the actual search operation.

        Args:
            path: Resolved search directory
            params: Search parameters
            ctx: Tool execution context

        Returns:
            Search results as formatted string
        """
        ...


# TODO: Additional abstract classes for common patterns:
# - CommandExecutionTool (bash, shell commands)
# - DataTransformTool (format conversion, processing)
# - NetworkTool (web fetch, API calls)
# - DialogTool (user interaction)
