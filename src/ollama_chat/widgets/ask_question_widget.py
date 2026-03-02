from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.events import Key
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Input, OptionList, Static
from textual.widgets.option_list import Option


class AskQuestionWidget(Widget):
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("escape", "cancel", "Cancel question", show=False)
    ]

    DEFAULT_CSS: ClassVar[str] = """
    AskQuestionWidget {
        display: none;
        height: auto;
        max-height: 50%;
        background: $panel;
        border: heavy $accent;
        padding: 0;
    }

    AskQuestionWidget #aq-question {
        background: $primary-muted;
        color: $text;
        text-style: bold;
        padding: 1 2;
        width: 100%;
    }

    AskQuestionWidget OptionList {
        height: auto;
        max-height: 12;
        background: $surface;
        padding: 0 1;
    }

    AskQuestionWidget OptionList > .option-list--option-highlighted {
        background: $accent;
        color: $text;
        text-style: bold;
    }

    AskQuestionWidget OptionList > .option-list--option-hover {
        background: $boost;
    }

    AskQuestionWidget #aq-custom-input {
        margin: 1 1 0 1;
    }

    AskQuestionWidget:focus-within {
        border: heavy $accent;
    }
    """

    class Answered(Message):
        def __init__(self, value: str | None) -> None:
            self.value = value
            super().__init__()

    def __init__(self) -> None:
        super().__init__()
        self._question: str = ""
        self._options: list[str] = []
        self._custom_enabled: bool = True

    def compose(self) -> ComposeResult:
        yield Static("", id="aq-question")
        yield OptionList(id="aq-options")
        yield Input(
            placeholder="Type your answer and press Enter…",
            id="aq-custom-input",
        )

    def on_mount(self) -> None:
        self.border_title = "Assistant Question"
        self.border_subtitle = "↑↓ · 1-9 · Enter · Esc=cancel"

    def load_question(self, question: str, options: list[str], *, custom: bool = True) -> None:
        self._question = question
        self._options = list(options)
        self._custom_enabled = bool(custom)

        self.query_one("#aq-question", Static).update(question)

        option_list = self.query_one("#aq-options", OptionList)
        option_list.clear_options()
        for i, opt in enumerate(self._options):
            option_list.add_option(Option(f"[bold]{i + 1}.[/] {opt}", id=f"opt-{i}"))
        if self._options:
            option_list.highlighted = 0

        custom_input = self.query_one("#aq-custom-input", Input)
        custom_input.value = ""
        custom_input.display = self._custom_enabled

    def _post_answer(self, value: str | None) -> None:
        self.post_message(self.Answered(value))

    def action_cancel(self) -> None:
        self._post_answer(None)

    def _handle_number_key(self, key: str) -> bool:
        if not key.isdigit():
            return False
        idx = int(key) - 1
        if idx < 0 or idx >= len(self._options):
            return False

        option_list = self.query_one("#aq-options", OptionList)
        option_list.highlighted = idx
        if hasattr(option_list, "action_select"):
            option_list.action_select()  # type: ignore[attr-defined]
        else:
            self._post_answer(self._options[idx])
        return True

    def on_key(self, event: Key) -> None:
        if self._handle_number_key(event.key):
            event.stop()
            return

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        idx = getattr(event, "option_index", None)
        if idx is None:
            idx = getattr(event, "index", None)
        try:
            option_index = int(idx)
        except (TypeError, ValueError):
            return
        if 0 <= option_index < len(self._options):
            self._post_answer(self._options[option_index])

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        event.stop()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        if not self._custom_enabled:
            return
        value = (event.value or "").strip()
        if not value:
            return
        self._post_answer(value)
