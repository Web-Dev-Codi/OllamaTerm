"""Async Ollama chat client wrapper with streaming, thinking, tools, and vision support."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
import inspect
import json
import logging
import re
import time
from typing import Any, Literal

from .capability_cache import CapabilityPersistence, ModelCapabilityCache
from .exceptions import (
    OllamaChatError,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaStreamingError,
    OllamaToolError,
)
from .message_store import MessageStore

try:
    import httpx
except ModuleNotFoundError:  # pragma: no cover - optional transport dependency.
    httpx = None  # type: ignore[assignment]

try:
    from ollama import AsyncClient as _AsyncClient
except (
    ModuleNotFoundError
):  # pragma: no cover - exercised only in missing dependency environments.
    _AsyncClient = None  # type: ignore[misc,assignment]

try:
    import ollama as _ollama_pkg
except ModuleNotFoundError:  # pragma: no cover - optional runtime detail for logging
    _ollama_pkg = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)

# Tool-use policy appended to the system message when tools are active.
# Reduces false-positive tool calls on conversational/creative prompts.
_TOOL_USE_POLICY = (
    "\n\nTOOL USE POLICY: Only invoke tools when the task explicitly requires "
    "file, code, or system operations. For conversational questions, creative "
    "writing, general knowledge, or anything the model can answer directly "
    "— respond without calling any tools."
)

# Question-asking policy for clarifying ambiguous requests.
# Calibrated for "liberal" usage (40-60% question rate) with models ≤14B params.
# Supports: Qwen 3, Llama 3.1 8B, Mistral Nemo 12B, DeepSeek Coder 6.7B, Phi-3.5 Mini.
_QUESTION_USE_POLICY = """

CLARIFYING QUESTIONS POLICY:
You have access to an 'ask_user_question' tool. Use it proactively when:

ALWAYS ASK WHEN:
• User request is ambiguous ("fix the bug", "add auth", "optimize code")
• Multiple valid implementation approaches exist
• Important technology/library choices need to be made
• Destructive operations without clear scope ("delete files", "remove code")
• Parameters are missing ("add caching" - which backend? "deploy" - where?)
• File/module targets are unclear ("refactor the handler" - which one?)
• Configuration values are needed (ports, URLs, API keys)
• User intent could mean different things

QUESTION QUALITY GUIDELINES:
• Provide 2-5 specific, actionable options
• Make options mutually exclusive when possible
• Enable 'custom' for unique user requirements
• Ask ONE question at a time for complex decisions
• Frame positively: "Which approach?" not "I don't understand"
• Questions should be answerable in <10 seconds

FORMAT REQUIREMENTS (CRITICAL):
• The tool call MUST include a 'question' string
• The tool call MUST include an 'options' list of 2-5 strings
• WRONG: {"options": [{"label": "Redis"}]} ❌
• CORRECT: {"options": ["Redis", "Memcached"]} ✓

WHEN NOT TO ASK (rare):
• Task is completely unambiguous ("write factorial function")
• User already provided all necessary details in their message
• The answer is obvious from recent conversation context

IMPORTANT: It's better to ask and get it right than to guess and waste user time.
"""

# Few-shot examples showing proper question-asking behavior.
# These concrete examples help smaller models (≤14B) understand when and how to ask.
_QUESTION_EXAMPLES = """

EXAMPLES OF PROPER QUESTION-ASKING:

Example 1 - Ambiguous Code Target:
User: "Refactor the database connection code"
You (thinking): "Multiple files handle DB connections. Need to ask which."
You (action): Call ask_user_question tool with:
  question: "Which database connection code should I refactor?"
  options: [
    "Main connection pool (db/pool.py)",
    "ORM models (models/base.py)",
    "Redis cache connection (cache/redis.py)",
    "All DB-related code"
  ]
User: "Main connection pool"
You: "I'll refactor the connection pool in db/pool.py focusing on connection reuse and error handling..."

Example 2 - Technology Choice:
User: "Add caching to the API endpoints"
You (thinking): "Many caching strategies exist. Should ask."
You (action): Call ask_user_question tool with:
  question: "Which caching backend should I use?"
  options: [
    "Redis (fast, external, scalable)",
    "Memcached (simple, in-memory)",
    "In-memory Python dict (no dependencies)",
    "Database query caching (SQLAlchemy)"
  ]
User: "Redis"
You: "I'll integrate Redis caching with redis-py. I'll add caching decorators to expensive endpoints..."

Example 3 - Clear Task (No Question Needed):
User: "Write a function to calculate the factorial of a number"
You (thinking): "This is unambiguous. Input: integer. Output: factorial. No question needed."
You (action): Write the code directly:
```python
def factorial(n: int) -> int:
    if n < 0:
        raise ValueError("Factorial undefined for negative numbers")
    if n <= 1:
        return 1
    return n * factorial(n - 1)
```
"""

# Tools that are I/O-bound and fast - don't need thread pool overhead
# These tools complete quickly (<10ms) and don't block the event loop.
# NOTE: Tools that may wait on user interaction (e.g. "question") MUST NOT be
# listed here, otherwise they will block the Textual event loop and freeze the UI.
FAST_SYNC_TOOLS = {
    "read",
    "write",
    "edit",
    "glob",
    "grep",
    "ls",
    "list",
    "invalid",
    "todo_read",
    "todo_write",
}


@dataclass
class ChatChunk:
    """A single typed chunk yielded during a streaming agent-loop response."""

    kind: Literal["thinking", "content", "tool_call", "tool_result"]
    text: str = ""
    tool_name: str = ""
    tool_args: dict[str, Any] = field(default_factory=dict)
    tool_result: str = ""
    # Index of the tool call within a parallel batch (from the API response).
    # None for non-tool chunks or when the API does not include an index.
    tool_index: int | None = None


@dataclass(frozen=True)
class CapabilityReport:
    """Capability metadata returned from Ollama's /api/show.

    known=False means the API call failed or the response did not include a
    capabilities field (old Ollama versions, missing metadata, etc.).

    known=True means the server explicitly returned capability metadata. In that
    case, an empty `caps` set is authoritative (the model reports no capabilities).
    """

    caps: frozenset[str]
    known: bool


@dataclass(frozen=True)
class ChatSendOptions:
    """Typed options for sending a chat message (≤2 param API).

    Group related optional arguments to avoid boolean flag params and
    long parameter lists in public APIs.
    """

    images: list[str | bytes] | None = None
    tool_registry: Any | None = None
    think: bool = False
    max_tool_iterations: int = 10


@dataclass(frozen=True)
class ModelReadyOptions:
    """Options for ensuring a model is ready."""

    pull_if_missing: bool = True


class OllamaChat:
    """Stateful chat wrapper that keeps bounded message history and streams replies."""

    def __init__(
        self,
        host: str,
        model: str,
        system_prompt: str,
        timeout: int = 120,
        retries: int = 2,
        retry_backoff_seconds: float = 0.5,
        max_history_messages: int = 200,
        max_context_tokens: int = 4096,
        client: Any | None = None,
        api_key: str = "",
    ) -> None:
        self.host = host
        self.model = model
        self.system_prompt = system_prompt
        self.timeout = timeout
        self.retries = retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.max_context_tokens = max_context_tokens

        if client is not None:
            self._client = client
        elif _AsyncClient is not None:
            client_kwargs: dict[str, Any] = {"host": host, "timeout": timeout}
            stripped_key = (api_key or "").strip()
            if stripped_key:
                client_kwargs["headers"] = {"authorization": f"Bearer {stripped_key}"}
            self._client = _AsyncClient(**client_kwargs)
        else:
            raise OllamaConnectionError(
                "The ollama package is not installed. Install dependencies with pip install -e ."
            )

        self.message_store = MessageStore(
            system_prompt=system_prompt,
            max_history_messages=max_history_messages,
            max_context_tokens=max_context_tokens,
        )

        try:
            self._chat_param_names = set(
                inspect.signature(self._client.chat).parameters.keys()
            )
        except Exception:
            self._chat_param_names = set()

        # Capability caching for auto-filtering based on model support
        self._capability_persistence = CapabilityPersistence()
        self._current_capability_cache: ModelCapabilityCache | None = None
        self._formatted_tools_cache: list[dict[str, Any]] | None = None

        try:
            sdk_version = (
                getattr(_ollama_pkg, "__version__", "unknown")
                if _ollama_pkg is not None
                else "not installed"
            )
        except Exception:
            sdk_version = "unknown"
        LOGGER.info(
            "chat.sdk.signature",
            extra={
                "event": "chat.sdk.signature",
                "sdk_version": sdk_version,
                "supported_params": sorted(self._chat_param_names),
            },
        )

    @property
    def messages(self) -> list[dict[str, Any]]:
        """Expose message history for UI and tests."""
        return self.message_store.messages

    def clear_history(self) -> None:
        """Clear the current conversation while keeping configured system prompts."""
        self.message_store.clear()

    def load_history(self, messages: list[dict[str, str]]) -> None:
        """Replace current history from a persisted conversation payload."""
        self.message_store.replace_messages(messages)

    def set_model(self, model_name: str) -> None:
        """Update the active model name and invalidate caches."""
        normalized = model_name.strip()
        if normalized and normalized != self.model:
            self.model = normalized
            # Invalidate capability and tools caches when model changes
            self._current_capability_cache = None
            self._formatted_tools_cache = None

    async def send(
        self, user_message: str, options: ChatSendOptions | None = None
    ) -> AsyncGenerator[ChatChunk, None]:
        """Two-parameter alternative to ``send_message`` using typed options.

        Implemented as an async generator that yields from ``send_message`` so
        callers can iterate with ``async for`` directly.
        """
        opts = options or ChatSendOptions()
        async for chunk in self.send_message(
            user_message,
            images=opts.images,
            tool_registry=opts.tool_registry,
            think=opts.think,
            max_tool_iterations=opts.max_tool_iterations,
        ):
            yield chunk

    @staticmethod
    def _model_name_matches(requested_model: str, available_model: str) -> bool:
        requested = requested_model.strip().lower()
        available = available_model.strip().lower()
        if requested == available:
            return True
        if ":" not in requested and available.startswith(f"{requested}:"):
            return True
        return False

    async def list_models(self) -> list[str]:
        """Return available model names from Ollama."""
        response = await self._client.list()
        names: list[str] = []
        models: Any = None
        if hasattr(response, "models"):
            models = response.models
        elif isinstance(response, dict):
            models = response.get("models")
        elif hasattr(response, "model_dump"):
            try:
                models = response.model_dump().get("models")
            except Exception:
                models = None

        if isinstance(models, list):
            for model in models:
                candidate_name: str | None = None
                if isinstance(model, dict):
                    for key in ("name", "model"):
                        value = model.get(key)
                        if isinstance(value, str) and value.strip():
                            candidate_name = value.strip()
                            break
                else:
                    for attr in ("name", "model"):
                        value = getattr(model, attr, None)
                        if isinstance(value, str) and value.strip():
                            candidate_name = value.strip()
                            break
                if candidate_name:
                    names.append(candidate_name)
        return names

    async def ensure_model_ready(self, pull_if_missing: bool = True) -> bool:
        """Ensure configured model is available; optionally pull it when missing."""
        try:
            available_models = await self.list_models()
        except Exception as exc:
            raise self._map_exception(exc) from exc

        if any(
            self._model_name_matches(self.model, available)
            for available in available_models
        ):
            LOGGER.info(
                "chat.model.ready",
                extra={"event": "chat.model.ready", "model": self.model},
            )
            return True

        if not pull_if_missing:
            raise OllamaModelNotFoundError(
                f"Configured model {self.model!r} is not available."
            )

        LOGGER.info(
            "chat.model.pull.start",
            extra={"event": "chat.model.pull.start", "model": self.model},
        )
        try:
            await self._client.pull(model=self.model, stream=False)
        except Exception as exc:
            raise self._map_exception(exc) from exc

        LOGGER.info(
            "chat.model.pull.complete",
            extra={"event": "chat.model.pull.complete", "model": self.model},
        )
        return True

    async def ensure_model_ready_with(self, options: ModelReadyOptions) -> bool:
        """Two-parameter variant that accepts typed options instead of a flag."""
        return await self.ensure_model_ready(pull_if_missing=options.pull_if_missing)

    async def ensure_model_ready_pull(self) -> bool:
        """Ensure model is ready, pulling if missing."""
        return await self.ensure_model_ready(True)

    async def ensure_model_ready_no_pull(self) -> bool:
        """Ensure model is ready without pulling when missing."""
        return await self.ensure_model_ready(False)

    async def check_connection(self) -> bool:
        """Return whether the Ollama host is reachable."""
        try:
            await self._client.list()
            return True
        except Exception:
            return False

    async def show_model_capabilities(
        self, model_name: str | None = None
    ) -> CapabilityReport:
        """Return the capability strings reported by Ollama's /api/show for a model.

        Capabilities are lowercase strings such as ``"tools"``, ``"vision"``,
        and ``"thinking"``.  An **empty** frozenset means the information is
        unavailable (old Ollama version, model not found, or the model predates
        capability metadata) — callers should treat an empty result as *unknown*
        and keep their config flags unchanged (permissive fallback).

        A **non-empty** frozenset is the authoritative list from Ollama and
        should be intersected with user config flags to derive effective behaviour.
        """
        name = (model_name or self.model).strip()
        try:
            response = await self._client.show(name)
            caps_raw: Any = None
            caps_known = False

            # SDK object path.
            if hasattr(response, "capabilities"):
                caps_raw = response.capabilities
                # Treat explicit None as unknown; anything else counts as present.
                if caps_raw is not None:
                    caps_known = True

            # Dict path.
            elif isinstance(response, dict):
                if "capabilities" in response:
                    caps_known = True
                caps_raw = response.get("capabilities")

            # Pydantic model path.
            elif hasattr(response, "model_dump"):
                try:
                    dumped = response.model_dump()
                    if "capabilities" in dumped:
                        caps_known = True
                    caps_raw = dumped.get("capabilities")
                except Exception:
                    caps_raw = None
                    caps_known = False

            parsed: set[str] = set()
            if isinstance(caps_raw, list):
                for item in caps_raw:
                    value = str(item).lower().strip()
                    if value:
                        parsed.add(value)
            elif isinstance(caps_raw, dict):
                # Some servers may return a mapping of capability -> bool.
                for key, value in caps_raw.items():
                    if value:
                        name_key = str(key).lower().strip()
                        if name_key:
                            parsed.add(name_key)
            elif isinstance(caps_raw, str):
                # Defensive: accept comma/whitespace-separated strings.
                parts = [p.strip().lower() for p in caps_raw.replace(",", " ").split()]
                parsed.update(p for p in parts if p)

            return CapabilityReport(caps=frozenset(parsed), known=caps_known)
        except Exception:
            LOGGER.debug(
                "chat.model.show.failed",
                extra={"event": "chat.model.show.failed", "model": name},
            )
        return CapabilityReport(caps=frozenset(), known=False)

    @property
    def estimated_context_tokens(self) -> int:
        """Return deterministic token estimate for current context."""
        # Pass None so MessageStore reads _messages directly (no copy).
        return self.message_store.estimated_tokens()

    @staticmethod
    def _extract_from_chunk(chunk: Any, field: str) -> Any:
        """Extract a named field from message.field in an Ollama chunk payload.

        Tries SDK object attribute access first, then falls back to dict paths
        produced by model_dump() / dict().  Returns None when the field is absent.
        """
        message_obj = getattr(chunk, "message", None)
        if message_obj is not None:
            value = getattr(message_obj, field, None)
            if value is not None:
                return value

        # Normalise Pydantic-model chunks to plain dicts once.
        if hasattr(chunk, "model_dump"):
            try:
                chunk = chunk.model_dump()
            except Exception:
                pass
        elif hasattr(chunk, "dict"):
            try:
                chunk = chunk.dict()
            except Exception:
                pass

        if isinstance(chunk, dict):
            message = chunk.get("message")
            if isinstance(message, dict):
                value = message.get(field)
                if value is not None:
                    return value
            # Top-level fallback (e.g. generate endpoint).
            return chunk.get(field)
        return None

    @classmethod
    def _extract_chunk_text(cls, chunk: Any) -> str:
        """Extract streamed token text from an Ollama chunk payload."""
        value = cls._extract_from_chunk(chunk, "content")
        if isinstance(value, str) and value:
            return value
        # Fallback for generate-style payloads (response instead of message.content).
        value = cls._extract_from_chunk(chunk, "response")
        return value if isinstance(value, str) else ""

    @classmethod
    def _extract_chunk_thinking(cls, chunk: Any) -> str:
        """Extract streamed thinking text from an Ollama chunk payload."""
        value = cls._extract_from_chunk(chunk, "thinking")
        return value if isinstance(value, str) else ""

    @classmethod
    def _extract_chunk_tool_calls(cls, chunk: Any) -> list[Any]:
        """Extract tool_calls from an Ollama chunk payload."""
        value = cls._extract_from_chunk(chunk, "tool_calls")
        return value if isinstance(value, list) else []

    @staticmethod
    def _parse_inline_tool_call_from_content(
        content: str, allowed_names: set[str]
    ) -> list[dict[str, Any]]:
        """Parse a tool call embedded as JSON in content code blocks.

        Some models emit a JSON object like {"name": "ls", "arguments": {}}
        instead of structured tool_calls. Convert it to a minimal tool_call
        dict only when the name is in allowed_names.
        """
        text = (content or "").strip()
        if not text:
            return []

        # Prefer ```json code blocks if present.
        match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
        candidate = match.group(1) if match else text

        try:
            parsed = json.loads(candidate)
        except Exception:
            return []

        def _as_call(obj: dict[str, Any]) -> list[dict[str, Any]]:
            if not isinstance(obj, dict):
                return []
            if isinstance(obj.get("function"), dict):
                fn = obj["function"]
                name = str(fn.get("name", ""))
                if name and name in allowed_names:
                    args = fn.get("arguments", {})
                    if not isinstance(args, dict):
                        args = {}
                    return [{"function": {"name": name, "arguments": args}}]
                return []
            name = str(obj.get("name", ""))
            if name and name in allowed_names:
                args = obj.get("arguments", {})
                if not isinstance(args, dict):
                    args = {}
                return [{"function": {"name": name, "arguments": args}}]
            return []

        if isinstance(parsed, list):
            for item in parsed:
                calls = _as_call(item)
                if calls:
                    return calls
            return []
        if isinstance(parsed, dict):
            return _as_call(parsed)
        return []

    def _map_exception(self, exc: Exception) -> OllamaChatError:
        if isinstance(exc, OllamaChatError):
            return exc

        lower_message = str(exc).lower()

        if httpx is not None and isinstance(
            exc,
            (
                httpx.ConnectError,
                httpx.ConnectTimeout,
                httpx.ReadTimeout,
                httpx.NetworkError,
            ),
        ):
            return OllamaConnectionError(
                f"Unable to connect to Ollama host {self.host}."
            )

        if "model" in lower_message and "not found" in lower_message:
            return OllamaModelNotFoundError(
                f"Model {self.model!r} was not found on {self.host}."
            )
        if "404" in lower_message and "model" in lower_message:
            return OllamaModelNotFoundError(
                f"Model {self.model!r} was not found on {self.host}."
            )

        return OllamaStreamingError(
            f"Failed to stream response from Ollama at {self.host}: {exc}"
        )

    async def _ensure_capability_cache(
        self, tool_registry: Any | None = None
    ) -> ModelCapabilityCache:
        """Ensure we have fresh capability metadata for current model.

        Returns cached data if available and fresh, otherwise fetches from /api/show.
        """
        # Check if we already have a fresh cache for this model
        if self._current_capability_cache is not None:
            if self._current_capability_cache.model_name == self.model:
                return self._current_capability_cache

        # Try persistent cache first
        cached = self._capability_persistence.get(self.model, max_age_seconds=86400)
        if cached is not None:
            self._current_capability_cache = cached
            LOGGER.info(
                "capability_cache.hit",
                extra={
                    "event": "capability_cache.hit",
                    "model": self.model,
                },
            )
            return cached

        # Fetch from Ollama
        LOGGER.info(
            "capability_cache.miss",
            extra={
                "event": "capability_cache.miss",
                "model": self.model,
            },
        )

        caps_report = await self.show_model_capabilities()

        # Build cache entry
        cache = ModelCapabilityCache(
            model_name=self.model,
            supports_tools="tools" in caps_report.caps if caps_report.known else True,
            supports_vision="vision" in caps_report.caps if caps_report.known else True,
            supports_thinking="thinking" in caps_report.caps
            if caps_report.known
            else True,
            raw_capabilities=list(caps_report.caps),
            timestamp=time.time(),
        )

        # Store in memory and persist
        self._current_capability_cache = cache
        self._capability_persistence.set(cache)

        # Invalidate formatted tools cache since capabilities changed
        self._formatted_tools_cache = None

        return cache

    def _format_tools_for_model(self, tool_registry: Any) -> list[dict[str, Any]]:
        """Format tools once and cache the result."""
        if self._formatted_tools_cache is not None:
            return self._formatted_tools_cache

        if tool_registry is None or tool_registry.is_empty:
            self._formatted_tools_cache = []
            return []

        # Get tools using build_tools_list() method
        formatted = tool_registry.build_tools_list()
        self._formatted_tools_cache = formatted

        return formatted

    async def _stream_once_with_capabilities(
        self,
        request_messages: list[dict[str, Any]],
        tool_registry: Any | None,
        think: bool,
    ) -> AsyncGenerator[ChatChunk, None]:
        """Stream a single chat turn with automatic capability filtering.

        Only includes tools, think, etc. if the model supports them based on
        the /api/show capabilities response.

        Args:
            request_messages: Message history to send
            tool_registry: ToolRegistry instance (not a formatted list anymore)
            think: Whether to request thinking traces

        Yields:
            ChatChunk objects for thinking, content, and tool_calls
        """
        # Ensure we have capability metadata
        caps = await self._ensure_capability_cache(tool_registry)

        # Build base kwargs
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": request_messages,
            "stream": True,
            # Align Ollama's context window with client-side trim budget
            "options": {"num_ctx": self.max_context_tokens},
        }

        # Only add 'think' if model supports it
        if think and caps.supports_thinking:
            if "gpt-oss" in self.model.lower():
                kwargs["think"] = "medium"
            else:
                kwargs["think"] = True

        # Only add 'tools' if model supports them
        formatted_tools: list[dict[str, Any]] = []
        if tool_registry and caps.supports_tools:
            formatted_tools = self._format_tools_for_model(tool_registry)
            if formatted_tools:
                kwargs["tools"] = formatted_tools

        # Inject tool-use policy into the system message when tools are active.
        # Operates on a shallow copy so MessageStore is never mutated.
        if formatted_tools:
            # Build policy text - start with base tool use policy
            policy_text = _TOOL_USE_POLICY

            # Check if ask_user_question tool is available in this request
            has_question_tool = any(
                t.get("function", {}).get("name") == "ask_user_question"
                for t in formatted_tools
            )

            # Add question-specific guidance if tool is present
            # This is injected dynamically per-request to support model switching
            if has_question_tool:
                policy_text += _QUESTION_USE_POLICY
                policy_text += _QUESTION_EXAMPLES

            patched_messages = list(request_messages)
            if patched_messages and patched_messages[0].get("role") == "system":
                first = dict(patched_messages[0])
                # Append after user's custom system prompt (Option A per user request)
                first["content"] = str(first.get("content", "")) + policy_text
                patched_messages[0] = first
            else:
                patched_messages.insert(
                    0,
                    {"role": "system", "content": policy_text.lstrip()},
                )
            kwargs["messages"] = patched_messages

        # Strip kwargs the SDK doesn't accept (for older ollama versions)
        if self._chat_param_names:
            if "think" not in self._chat_param_names:
                kwargs.pop("think", None)
            if "tools" not in self._chat_param_names:
                kwargs.pop("tools", None)

        stream = await self._client.chat(**kwargs)
        async for chunk in stream:
            thinking_text = self._extract_chunk_thinking(chunk)
            if thinking_text:
                yield ChatChunk(kind="thinking", text=thinking_text)

            content_text = self._extract_chunk_text(chunk)
            if content_text:
                yield ChatChunk(kind="content", text=content_text)

            chunk_tool_calls = self._extract_chunk_tool_calls(chunk)

            # Only parse inline tool calls if model supports tools and we sent them
            if (
                not chunk_tool_calls
                and caps.supports_tools
                and formatted_tools
                and content_text
            ):
                allowed: set[str] = set()
                for t in formatted_tools:
                    if isinstance(t, dict):
                        fn = t.get("function", {})
                        if isinstance(fn, dict):
                            n = fn.get("name")
                            if isinstance(n, str) and n:
                                allowed.add(n)
                for tc in self._parse_inline_tool_call_from_content(
                    content_text, allowed
                ):
                    chunk_tool_calls.append(tc)

            for tc in chunk_tool_calls:
                name, args, index = self._parse_tool_call(tc)
                if name:
                    yield ChatChunk(
                        kind="tool_call",
                        tool_name=name,
                        tool_args=args,
                        tool_index=index,
                    )

    @staticmethod
    def _parse_tool_call(tc: Any) -> tuple[str, dict[str, Any], int | None]:
        """Extract (name, arguments, index) from a tool call object or dict.

        The ``index`` field is present in parallel tool-call responses and is
        needed to correctly correlate tool results with their originating call
        when sending the follow-up request.  Returns ``None`` when absent.
        """
        # SDK object: tc.function.name / tc.function.arguments / tc.function.index
        fn = getattr(tc, "function", None)
        if fn is not None:
            name = getattr(fn, "name", None) or ""
            args = getattr(fn, "arguments", None) or {}
            if not isinstance(args, dict):
                args = {}
            raw_index = getattr(fn, "index", None)
            try:
                index: int | None = int(raw_index) if raw_index is not None else None
            except (TypeError, ValueError):
                index = None
            return str(name), args, index

        # Dict-based fallback
        if isinstance(tc, dict):
            fn_dict = tc.get("function", {})
            if isinstance(fn_dict, dict):
                name = str(fn_dict.get("name", ""))
                args = fn_dict.get("arguments", {})
                if not isinstance(args, dict):
                    args = {}
                raw_index = fn_dict.get("index")
                try:
                    index = int(raw_index) if raw_index is not None else None
                except (TypeError, ValueError):
                    index = None
                return name, args, index
        return "", {}, None

    async def send_message(
        self,
        user_message: str,
        images: list[str | bytes] | None = None,
        tool_registry: Any | None = None,
        think: bool = False,
        max_tool_iterations: int = 10,
    ) -> AsyncGenerator[ChatChunk, None]:
        """Send a user message and stream the assistant reply as typed ChatChunk objects.

        Supports thinking traces, tool calling with a full agent loop, and
        image attachments for vision-capable models.

        Images are passed to the API but are NOT stored in conversation history.

        If streaming fails after the user message has been appended to history,
        the user message is rolled back to prevent consecutive user messages from
        corrupting the API context on the next call.
        """
        normalized = user_message.strip()
        if not normalized and not images:
            return

        # Build the user message dict; include images only for the API call.
        user_msg: dict[str, Any] = {"role": "user", "content": normalized}
        if images:
            user_msg["images"] = list(images)

        # Persist only the text portion to history (images are ephemeral).
        self.message_store.append("user", normalized)

        # Build the initial API context; inject images into the last user message.
        request_messages: list[dict[str, Any]] = list(
            self.message_store.build_api_context()
        )
        if images and request_messages:
            request_messages[-1] = dict(request_messages[-1])
            request_messages[-1]["images"] = list(images)

        # Accumulated assistant message parts (for history persistence).
        # Use lists for efficient string building instead of += concatenation
        accumulated_thinking_parts: list[str] = []
        accumulated_content_parts: list[str] = []
        accumulated_tool_calls: list[dict[str, Any]] = []

        # Track the most-recent iteration's content so that if max_tool_iterations
        # is exhausted (loop ends without a clean break), we can still persist the
        # last streamed content rather than an empty string.
        _last_iteration_content = ""

        try:
            for iteration in range(max_tool_iterations):
                for attempt in range(self.retries + 1):
                    try:
                        async for chunk in self._stream_once_with_capabilities(
                            request_messages, tool_registry, think
                        ):
                            if chunk.kind == "thinking":
                                accumulated_thinking_parts.append(chunk.text)
                                yield chunk
                            elif chunk.kind == "content":
                                accumulated_content_parts.append(chunk.text)
                                yield chunk
                            elif chunk.kind == "tool_call":
                                accumulated_tool_calls.append(
                                    {
                                        "name": chunk.tool_name,
                                        "args": chunk.tool_args,
                                        "index": chunk.tool_index,
                                    }
                                )
                                yield chunk
                        break
                    except asyncio.CancelledError:
                        LOGGER.info(
                            "chat.request.cancelled",
                            extra={"event": "chat.request.cancelled"},
                        )
                        raise
                    except OllamaToolError:
                        raise
                    except Exception as exc:  # noqa: BLE001 - external API can fail in many ways.
                        mapped_exc = self._map_exception(exc)
                        LOGGER.warning(
                            "chat.request.retry",
                            extra={
                                "event": "chat.request.retry",
                                "attempt": attempt + 1,
                                "error_type": mapped_exc.__class__.__name__,
                            },
                        )
                        if attempt >= self.retries:
                            raise mapped_exc from exc
                        await asyncio.sleep(self.retry_backoff_seconds * (attempt + 1))

                # If no tool calls, the agent loop is complete.
                if not accumulated_tool_calls:
                    break

                # Append the assistant turn (with tool calls) to the request context.
                # accumulated_tool_calls is non-empty here (checked above).
                # The index field is preserved from the API response so that
                # parallel tool call batches are correctly correlated when
                # sending follow-up requests.
                tool_call_entries: list[dict[str, Any]] = []
                for seq, tc in enumerate(accumulated_tool_calls):
                    fn_entry: dict[str, Any] = {
                        "name": tc["name"],
                        "arguments": tc["args"],
                    }
                    # Use the API-provided index when available; fall back to
                    # sequential position so the field is always present.
                    fn_entry["index"] = tc["index"] if tc["index"] is not None else seq
                    tool_call_entries.append({"type": "function", "function": fn_entry})

                # Join accumulated parts for API message
                accumulated_content = "".join(accumulated_content_parts)
                accumulated_thinking = "".join(accumulated_thinking_parts)

                assistant_turn: dict[str, Any] = {
                    "role": "assistant",
                    "content": accumulated_content,
                    "tool_calls": tool_call_entries,
                }
                if accumulated_thinking:
                    assistant_turn["thinking"] = accumulated_thinking
                request_messages.append(assistant_turn)

                # Execute each tool call.
                # Fast I/O-bound tools run directly (no thread pool overhead).
                # Slow/CPU-intensive tools run in thread pool to avoid blocking.
                for tc in accumulated_tool_calls:
                    tool_name = tc["name"]
                    tool_args = tc["args"]
                    LOGGER.info(
                        "chat.tool.call",
                        extra={
                            "event": "chat.tool.call",
                            "tool": tool_name,
                            "iteration": iteration + 1,
                        },
                    )
                    try:
                        # Fast I/O-bound tools run directly
                        if tool_name in FAST_SYNC_TOOLS:
                            result = tool_registry.execute(
                                tool_name,
                                tool_args,  # type: ignore[union-attr]
                            )
                        else:
                            # Slow/CPU-intensive tools run in thread pool
                            result = await asyncio.to_thread(
                                tool_registry.execute,
                                tool_name,
                                tool_args,  # type: ignore[union-attr]
                            )
                    except OllamaToolError as exc:
                        result = f"[Tool error: {exc}]"
                        LOGGER.warning(
                            "chat.tool.error",
                            extra={
                                "event": "chat.tool.error",
                                "tool": tool_name,
                                "error": str(exc),
                            },
                        )
                    yield ChatChunk(
                        kind="tool_result",
                        tool_name=tool_name,
                        tool_args=tool_args,
                        tool_result=result,
                    )
                    request_messages.append(
                        {"role": "tool", "tool_name": tool_name, "content": result}
                    )

                # Save content before reset so it's available if the loop exhausts.
                _last_iteration_content = accumulated_content

                # Reset accumulators for the next agent-loop iteration.
                accumulated_thinking_parts.clear()
                accumulated_content_parts.clear()
                accumulated_tool_calls = []

        except BaseException:
            # On any failure (cancellation, network error, tool error) roll back
            # the user message that was already appended so that the history does
            # not end with an unanswered user turn.  The next call to send_message
            # would otherwise send two consecutive user messages to the API.
            self.message_store.rollback_last_user_append()
            raise

        # Persist the final assistant response (text only) to history.
        # If the loop was exhausted (all iterations had tool calls), fall back to
        # the last content that was streamed to the UI but then reset.
        final_content = "".join(accumulated_content_parts)
        final_response = (final_content or _last_iteration_content).strip()
        self.message_store.append("assistant", final_response)
