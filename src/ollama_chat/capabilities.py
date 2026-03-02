"""Capability, search, and attachment state containers for the chat application."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CapabilityContext:
    """Effective runtime capability snapshot for the active model.

    The three auto-detected fields (``think``, ``tools_enabled``,
    ``vision_enabled``) are **not** read from the config file — they are
    computed by intersecting Ollama's ``/api/show`` response with what the
    model actually supports.  When capability metadata is unavailable
    (``known=False``), all three default to ``True`` (permissive fallback).

    The remaining fields come from the ``[capabilities]`` config section and
    represent pure user / app preferences that have no model-side equivalent.
    """

    # Auto-detected from /api/show (not configurable).
    think: bool = True
    tools_enabled: bool = True
    vision_enabled: bool = True

    # User / app preferences from [capabilities] config.
    show_thinking: bool = True
    web_search_enabled: bool = False
    web_search_api_key: str = ""
    max_tool_iterations: int = 10

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> CapabilityContext:
        """Build a CapabilityContext from the [capabilities] config section.

        Only the user-preference fields are read from config.  The
        auto-detected fields (``think``, ``tools_enabled``, ``vision_enabled``)
        are left at their permissive defaults (``True``) and will be updated
        by ``_update_effective_caps()`` once ``/api/show`` returns.
        """
        cap_cfg = config.get("capabilities", {})
        return cls(
            # Auto-detected fields: always start permissive.
            think=True,
            tools_enabled=True,
            vision_enabled=True,
            # User / app preferences from config.
            show_thinking=bool(cap_cfg.get("show_thinking", True)),
            web_search_enabled=bool(cap_cfg.get("web_search_enabled", False)),
            web_search_api_key=str(cap_cfg.get("web_search_api_key", "")),
            max_tool_iterations=int(cap_cfg.get("max_tool_iterations", 20)),
        )


@dataclass
class SearchState:
    """Tracks in-conversation search position and results."""

    query: str = ""
    results: list[int] = field(default_factory=list)
    position: int = -1

    def reset(self) -> None:
        """Clear all search state."""
        self.query = ""
        self.results = []
        self.position = -1

    def advance(self) -> int:
        """Move to the next result, wrapping around. Returns the current message index."""
        if not self.results:
            return -1
        self.position = (self.position + 1) % len(self.results)
        return self.results[self.position]

    def has_results(self) -> bool:
        """Return True when there are search results to navigate."""
        return len(self.results) > 0


@dataclass
class AttachmentState:
    """Pending image and file attachments awaiting the next send."""

    images: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)

    def add_image(self, path: str) -> None:
        """Queue an image attachment."""
        self.images.append(path)

    def add_file(self, path: str) -> None:
        """Queue a file attachment."""
        self.files.append(path)

    def clear(self) -> None:
        """Discard all pending attachments."""
        self.images.clear()
        self.files.clear()

    def has_any(self) -> bool:
        """Return True when at least one attachment is pending."""
        return bool(self.images or self.files)
