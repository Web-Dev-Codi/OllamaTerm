"""Message bubble widget for conversation rendering."""

from __future__ import annotations

from typing import Any

from rich.markdown import Markdown
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Static

from .code_block import CodeBlock, split_message


class MessageBubble(Vertical):
    """Render a single chat message with role, optional timestamp, thinking, and tool traces."""

    DEFAULT_CSS = """
    MessageBubble {
        height: auto;
    }
    MessageBubble > #header-block {
        padding: 0;
    }
    MessageBubble > #thinking-block {
        color: $text-muted;
        padding: 0 1;
        border-left: solid $panel;
        margin-bottom: 1;
        max-height: 12;
        overflow-y: auto;
    }
    MessageBubble > #tool-trace {
        color: $text-muted;
        padding: 0 1;
        border-left: solid $warning;
        margin-bottom: 1;
    }
    MessageBubble > #content-block {
        height: auto;
    }
    MessageBubble > .prose-segment {
        height: auto;
        padding: 0;
    }
    MessageBubble #copy-row {
        height: auto;
        padding: 0 0 1 0;
        align-horizontal: right;
    }
    MessageBubble #copy-row Button {
        border: none;
        background: transparent;
        color: $text-muted;
    }
    MessageBubble #copy-row Button:hover {
        color: $text;
        background: $panel;
    }
    """

    def __init__(
        self,
        content: str,
        role: str,
        timestamp: str = "",
        show_thinking: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.message_content = content
        self.role = role
        self.timestamp = timestamp
        self.show_thinking = show_thinking
        self._thinking_buffer = ""
        self._tool_trace_lines: list[str] = []
        self.add_class(f"role-{role}")

        # Direct references to inner widgets, set in compose().
        self._header_widget: Static | None = None
        self._thinking_widget: Static | None = None
        self._tool_widget: Static | None = None
        self._content_widget: Static | None = None
        # Mounted content segment widgets (prose + code blocks).
        self._segment_widgets: list[Static | CodeBlock] = []
        self._copy_button: Button | None = None

    @property
    def role_prefix(self) -> str:
        """Return a human-friendly role label."""
        return "You" if self.role == "user" else "Assistant"

    def _compose_header(self) -> str:
        if self.timestamp:
            return f"**{self.role_prefix}**  _{self.timestamp}_"
        return f"**{self.role_prefix}**"

    def compose(self) -> ComposeResult:
        """Compose the bubble layout with separate header, thinking, tool, and content blocks."""
        # Header is rendered once and never updated during streaming.
        self._header_widget = Static(
            Markdown(self._compose_header()), id="header-block"
        )
        self._thinking_widget = Static("", id="thinking-block")
        self._tool_widget = Static("", id="tool-trace")
        # content-block is a plain Static used during streaming; replaced on finalise.
        self._content_widget = Static("", id="content-block")

        yield self._header_widget
        if self.show_thinking:
            yield self._thinking_widget
        yield self._tool_widget
        yield self._content_widget
        copy_button = Button("⎘ copy reply", id="copy-message-btn")
        self._copy_button = copy_button
        with Horizontal(id="copy-row"):
            yield copy_button

    def on_mount(self) -> None:
        """Perform initial content render after mount."""
        self._refresh_content()
        if self._copy_button is not None and self.role == "user":
            self._copy_button.display = False

    def _refresh_content(self) -> None:
        if self._content_widget is None:
            return
        # During streaming we update the plain Static for performance.
        text = self.message_content.rstrip()
        self._content_widget.update(Markdown(text) if text else "")

    async def _rebuild_content_segments(self) -> None:
        """Replace the plain content-block with per-segment prose/code widgets."""
        text = self.message_content.rstrip()
        segments = split_message(text)

        # Remove old segment widgets from a previous rebuild.
        for w in self._segment_widgets:
            await w.remove()
        self._segment_widgets = []

        # Hide the streaming Static; segments take over rendering.
        if self._content_widget is not None:
            self._content_widget.display = False

        for content, lang in segments:
            if lang is None:
                # Prose segment.
                widget: Static | CodeBlock = Static(
                    Markdown(content), classes="prose-segment"
                )
            else:
                widget = CodeBlock(code=content, lang=lang)
            self._segment_widgets.append(widget)
            await self.mount(widget)

    def on_code_block_copy_requested(self, event: CodeBlock.CopyRequested) -> None:
        """Forward copy request to the app clipboard."""
        event.stop()
        app = self.app
        if hasattr(app, "copy_to_clipboard"):
            app.copy_to_clipboard(event.code)  # type: ignore[attr-defined]
            app.sub_title = "Code copied to clipboard."
        else:
            app.sub_title = "Clipboard unavailable."

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Copy entire message content when the copy button is clicked."""
        if event.button.id != "copy-message-btn":
            return
        event.stop()
        app = self.app
        text = self.message_content.strip()
        if not text:
            app.sub_title = "Nothing to copy."
            return
        if hasattr(app, "copy_to_clipboard"):
            app.copy_to_clipboard(text)  # type: ignore[attr-defined]
            app.sub_title = "Reply copied to clipboard."
        else:
            app.sub_title = "Clipboard unavailable."

    def _refresh_thinking(self) -> None:
        if not self.show_thinking or self._thinking_widget is None:
            return
        if self._thinking_buffer:
            self._thinking_widget.update(
                Text(f"Thinking:\n{self._thinking_buffer}", style="dim italic")
            )
            self._thinking_widget.display = True
        else:
            self._thinking_widget.display = False

    def _refresh_tool_trace(self) -> None:
        if self._tool_widget is None:
            return
        if self._tool_trace_lines:
            self._tool_widget.update(
                Text("\n".join(self._tool_trace_lines), style="dim")
            )
            self._tool_widget.display = True
        else:
            self._tool_widget.display = False

    def set_content(self, content: str) -> None:
        """Update message content and rerender."""
        self.message_content = content
        self._refresh_content()

    def append_content(self, content_chunk: str) -> None:
        """Append streamed content and rerender once per batch."""
        self.message_content += content_chunk
        self._refresh_content()

    async def finalize_content(self) -> None:
        """Rebuild content into prose+code segments after streaming ends."""
        await self._rebuild_content_segments()

    def append_thinking(self, thinking_chunk: str) -> None:
        """Accumulate a streamed thinking token and rerender the thinking block."""
        self._thinking_buffer += thinking_chunk
        self._refresh_thinking()

    def finalize_thinking(self) -> None:
        """Seal the thinking block with a final label (called when content starts)."""
        if (
            not self.show_thinking
            or not self._thinking_buffer
            or self._thinking_widget is None
        ):
            return
        self._thinking_widget.update(
            Text(f"Thought:\n{self._thinking_buffer}", style="dim italic")
        )

    def append_tool_call(self, name: str, args: dict[str, Any]) -> None:
        """Add a tool-call line to the tool trace block."""
        args_repr = ", ".join(f"{k}={v!r}" for k, v in args.items())
        self._tool_trace_lines.append(f"> Calling: {name}({args_repr})")
        self._refresh_tool_trace()

    def append_tool_result(self, name: str, result: str) -> None:
        """Add a tool-result line to the tool trace block."""
        # Truncate very long results in the UI display.
        preview = result[:200] + "..." if len(result) > 200 else result
        self._tool_trace_lines.append(f"< {name}: {preview}")
        self._refresh_tool_trace()
