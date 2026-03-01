"""Reusable modal screens for pickers and info dialogs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, OptionList, SelectionList, Static


class InfoScreen(ModalScreen[None]):
    """Modal that shows a block of text and closes on Escape/OK."""

    CSS = """
    InfoScreen {
        align: center middle;
    }

    #info-dialog {
        width: 80;
        max-width: 120;
        max-height: 28;
        padding: 1 2;
        border: round $panel;
        background: $surface;
    }

    #info-body {
        height: auto;
    }

    #info-actions {
        dock: bottom;
        height: 3;
        align: right middle;
    }
    """

    def __init__(self, text: str) -> None:
        super().__init__()
        self._text = text

    def compose(self) -> ComposeResult:
        with Container(id="info-dialog"):
            yield Static(self._text, id="info-body")
            with Container(id="info-actions"):
                yield Button("OK", id="info-ok", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "info-ok":
            event.stop()
            self.dismiss(None)

    def on_key(self, event: Any) -> None:  # noqa: ANN401
        key = str(getattr(event, "key", "")).lower()
        if key in {"escape", "enter"}:
            self.dismiss(None)


class SimplePickerScreen(ModalScreen[str | None]):
    """Modal picker for selecting from a list of strings."""

    CSS = """
    SimplePickerScreen {
        align: center middle;
    }

    #picker-dialog {
        width: 60;
        max-height: 24;
        padding: 1 2;
        border: round $panel;
        background: $surface;
    }

    #picker-title {
        padding-bottom: 1;
        text-style: bold;
    }

    #picker-help {
        padding-top: 1;
    }
    """

    def __init__(self, title: str, options: list[str]) -> None:
        super().__init__()
        self._title = title
        self._options = options

    def compose(self) -> ComposeResult:
        with Container(id="picker-dialog"):
            yield Static(self._title, id="picker-title")
            yield OptionList(*self._options, id="picker-options")
            yield Static("Enter/click to select | Esc to cancel", id="picker-help")

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        idx = getattr(event, "option_index", None)
        if idx is None:
            idx = getattr(event, "index", -1)
        try:
            selected = int(idx or -1)
        except (TypeError, ValueError):
            selected = -1
        if 0 <= selected < len(self._options):
            self.dismiss(self._options[selected])

    def on_key(self, event: Any) -> None:  # noqa: ANN401
        if str(getattr(event, "key", "")).lower() == "escape":
            self.dismiss(None)


class ImageAttachScreen(ModalScreen[str | None]):
    """Fallback modal for collecting an image path when native dialog is unavailable."""

    CSS = """
    ImageAttachScreen {
        align: center middle;
    }

    #image-attach-dialog {
        width: 60;
        padding: 1 3;
        border: round $panel;
        background: $surface;
    }

    #image-attach-title {
        padding-bottom: 1;
        text-style: bold;
    }

    #image-attach-input {
        width: 100%;
        margin: 1 0;
    }

    #image-attach-help {
        padding-top: 1;
        text-align: center;
    }
    """

    def compose(self) -> ComposeResult:
        with Container(id="image-attach-dialog"):
            yield Static("Attach image", id="image-attach-title")
            yield Input(
                placeholder="Enter absolute or relative image path...",
                id="image-attach-input",
            )
            yield Static("Enter to confirm  |  Esc to cancel", id="image-attach-help")

    def on_mount(self) -> None:
        self.query_one("#image-attach-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "image-attach-input":
            return
        value = event.value.strip()
        self.dismiss(value if value else None)

    def on_key(self, event: Any) -> None:  # noqa: ANN401
        if str(getattr(event, "key", "")).lower() == "escape":
            self.dismiss(None)


class TextPromptScreen(ModalScreen[str | None]):
    """Modal screen to prompt for a single line of text."""

    CSS = """
    TextPromptScreen {
        align: center middle;
    }

    #text-prompt-dialog {
        width: 60;
        height: auto;
        padding: 1 2;
        border: round $panel;
        background: $surface;
    }

    #text-prompt-title {
        padding-bottom: 1;
        text-style: bold;
    }

    #text-prompt-input {
        width: 100%;
        margin-bottom: 1;
    }
    """

    def __init__(self, title: str, placeholder: str = "") -> None:
        super().__init__()
        self._title = title
        self._placeholder = placeholder

    def compose(self) -> ComposeResult:
        with Container(id="text-prompt-dialog"):
            yield Static(self._title, id="text-prompt-title")
            yield Input(
                placeholder=self._placeholder,
                id="text-prompt-input",
            )
            yield Static("Enter to confirm | Esc to cancel", id="picker-help")

    def on_mount(self) -> None:
        self.query_one("#text-prompt-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "text-prompt-input":
            return
        value = event.value.strip()
        self.dismiss(value if value else "")

    def on_key(self, event: Any) -> None:  # noqa: ANN401
        if str(getattr(event, "key", "")).lower() == "escape":
            self.dismiss(None)


@dataclass(frozen=True)
class ConversationListItem:
    """Item shown in the conversation picker."""

    label: str
    path: str


class ConversationPickerScreen(ModalScreen[str | None]):
    """Picker for saved conversations from persistence index."""

    CSS = """
    ConversationPickerScreen {
        align: center middle;
    }

    #conv-dialog {
        width: 80;
        max-height: 26;
        padding: 1 2;
        border: round $panel;
        background: $surface;
    }

    #conv-title {
        padding-bottom: 1;
        text-style: bold;
    }

    #conv-help {
        padding-top: 1;
    }
    """

    def __init__(self, items: list[dict[str, str]]) -> None:
        super().__init__()
        self._items: list[ConversationListItem] = []
        for row in items:
            path = str(row.get("path", "")).strip()
            created_at = str(row.get("created_at", "")).strip()
            name = str(row.get("name", "")).strip()
            if not path:
                continue
            if name and created_at:
                label = f"{name}  —  {created_at}"
            elif name:
                label = name
            else:
                label = created_at if created_at else path
            self._items.append(ConversationListItem(label=label, path=path))

    def compose(self) -> ComposeResult:
        with Container(id="conv-dialog"):
            yield Static("Conversations", id="conv-title")
            yield OptionList(*(item.label for item in self._items), id="conv-options")
            yield Static("Enter/click to load | Esc to cancel", id="conv-help")

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        idx = getattr(event, "option_index", None)
        if idx is None:
            idx = getattr(event, "index", -1)
        try:
            selected = int(idx or -1)
        except (TypeError, ValueError):
            selected = -1
        if 0 <= selected < len(self._items):
            self.dismiss(self._items[selected].path)

    def on_key(self, event: Any) -> None:  # noqa: ANN401
        if str(getattr(event, "key", "")).lower() == "escape":
            self.dismiss(None)


class ThemePickerScreen(ModalScreen[str | None]):
    """Modal picker for selecting from available themes."""

    CSS = """
    ThemePickerScreen {
        align: center middle;
    }

    #theme-picker-dialog {
        width: 70;
        max-height: 28;
        padding: 1 2;
        border: round $panel;
        background: $surface;
    }

    #theme-picker-title {
        padding-bottom: 1;
        text-style: bold;
    }

    #theme-preview {
        height: 6;
        margin: 1 0;
        padding: 1;
        border: solid $panel;
        background: $background;
    }

    #color-swatch-primary,
    #color-swatch-secondary,
    #color-swatch-accent,
    #color-swatch-success,
    #color-swatch-warning,
    #color-swatch-error {
        width: 4;
        height: 1;
        margin: 0 1;
        border: solid red;
    }

    #theme-help {
        padding-top: 1;
    }
    """

    def __init__(self, themes: dict[str, Any], current_theme: str) -> None:
        super().__init__()
        self._themes = themes
        self._current_theme = current_theme
        self._theme_names = sorted(
            [name for name in themes.keys() if not name.endswith("-ansi")]
        )

    def compose(self) -> ComposeResult:
        with Container(id="theme-picker-dialog"):
            yield Static("Select a theme", id="theme-picker-title")
            yield OptionList(*self._theme_names, id="theme-options")

            # Theme preview area
            with Vertical(id="theme-preview"):
                yield Static("Theme preview will appear here", id="preview-text")
                with Horizontal(id="color-swatches"):
                    yield Static("Primary", id="color-swatch-primary")
                    yield Static("Secondary", id="color-swatch-secondary")
                    yield Static("Accent", id="color-swatch-accent")
                    yield Static("Success", id="color-swatch-success")
                    yield Static("Warning", id="color-swatch-warning")
                    yield Static("Error", id="color-swatch-error")

            yield Static("Enter/click to select | Esc to cancel", id="theme-help")

    def on_mount(self) -> None:
        options = self.query_one("#theme-options", OptionList)
        # Highlight current theme
        try:
            current_index = self._theme_names.index(self._current_theme)
            options.highlighted = current_index
        except ValueError:
            pass  # Current theme not in list
        options.focus()
        self._update_preview()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        option_index = getattr(event, "option_index", None)
        if option_index is None:
            option_index = getattr(event, "index", -1)
        try:
            selected_index = int(option_index or -1)
        except (TypeError, ValueError):
            selected_index = -1
        if 0 <= selected_index < len(self._theme_names):
            self.dismiss(self._theme_names[selected_index])

    def on_option_list_option_highlighted(
        self, _event: OptionList.OptionHighlighted
    ) -> None:
        self._update_preview()

    def _update_preview(self) -> None:
        """Update the theme preview with colors from the highlighted theme."""
        options = self.query_one("#theme-options", OptionList)
        preview_text = self.query_one("#preview-text", Static)

        try:
            highlighted_index = options.highlighted
            if highlighted_index is None or highlighted_index >= len(self._theme_names):
                return

            theme_name = self._theme_names[highlighted_index]
            theme = self._themes[theme_name]

            # Update preview text
            is_dark = "Dark" if getattr(theme, "dark", True) else "Light"
            preview_text.update(f"Theme: {theme_name} ({is_dark})")

            # Update color swatches
            swatches = self.query_one("#color-swatches", Horizontal)
            swatches.remove_children()

            colors = [
                ("Primary", getattr(theme, "primary", "#000000")),
                ("Secondary", getattr(theme, "secondary", "#000000")),
                ("Accent", getattr(theme, "accent", "#000000")),
                ("Success", getattr(theme, "success", "#000000")),
                ("Warning", getattr(theme, "warning", "#000000")),
                ("Error", getattr(theme, "error", "#000000")),
            ]

            for label, color in colors:
                swatch_id = f"color-swatch-{label.lower()}"
                swatch = Static(label, id=swatch_id)
                swatch.styles.background = color
                swatch.styles.color = "#ffffff"  # White text for contrast
                swatches.mount(swatch)

        except Exception:
            preview_text.update("Preview unavailable")

    def on_key(self, event: Any) -> None:  # noqa: ANN401
        if str(getattr(event, "key", "")).lower() == "escape":
            self.dismiss(None)


class QuestionScreen(ModalScreen["list[str] | None"]):
    """Modal that presents a structured question from the LLM and collects the answer.

    Dismisses with:
      list[str] — the selected/typed answer(s)
      None      — user cancelled (Escape or Cancel button)
    """

    CUSTOM_LABEL = "Type your own answer..."

    CSS = """
    QuestionScreen {
        align: center middle;
    }

    #question-dialog {
        width: 70;
        max-width: 110;
        max-height: 36;
        padding: 1 2;
        border: round $accent;
        background: $surface;
    }

    #question-header {
        text-style: bold;
        color: $accent;
        padding-bottom: 1;
    }

    #question-text {
        padding-bottom: 1;
    }

    #question-options {
        height: auto;
        max-height: 14;
        border: none;
    }

    #question-custom-input {
        margin-top: 1;
        display: none;
    }

    #question-custom-input.visible {
        display: block;
    }

    #question-actions {
        dock: bottom;
        height: 3;
        align: right middle;
        padding-top: 1;
    }

    #question-help {
        color: $text-muted;
        padding-top: 1;
    }
    """

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, question_info: dict) -> None:
        super().__init__()
        self._q = question_info
        self._multiple: bool = bool(question_info.get("multiple", False))
        self._custom: bool = bool(question_info.get("custom", False))
        self._options: list[dict] = list(question_info.get("options", []))
        self._custom_selected: bool = (
            False  # True when custom input is active in multi mode
        )
        self._pending_multi: list[str] = []  # Accumulates selections in multi mode

    def compose(self) -> ComposeResult:
        with Container(id="question-dialog"):
            yield Static(self._q.get("header", "Question"), id="question-header")
            yield Static(self._q.get("question", ""), id="question-text")

            if self._multiple:
                items = [(opt["label"], opt["label"], False) for opt in self._options]
                if self._custom:
                    items.append((self.CUSTOM_LABEL, self.CUSTOM_LABEL, False))
                yield SelectionList(*items, id="question-options")
                with Horizontal(id="question-actions"):
                    yield Button("Confirm", id="question-confirm", variant="primary")
                    yield Button("Cancel", id="question-cancel")
            else:
                option_labels = [opt["label"] for opt in self._options]
                if self._custom:
                    option_labels.append(self.CUSTOM_LABEL)
                yield OptionList(*option_labels, id="question-options")

            yield Input(
                placeholder="Type your answer and press Enter\u2026",
                id="question-custom-input",
            )

            if self._multiple:
                yield Static(
                    "Space=toggle  Enter/Confirm=submit  Esc=cancel",
                    id="question-help",
                )
            else:
                yield Static(
                    "Enter=select  Esc=cancel",
                    id="question-help",
                )

    def on_mount(self) -> None:
        self.query_one("#question-options").focus()

    def action_cancel(self) -> None:
        self.dismiss(None)

    # ── Single-select ────────────────────────────────────────────────────────

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if self._multiple:
            return  # handled by button
        label = str(event.option.prompt)
        if label == self.CUSTOM_LABEL:
            self._show_custom_input()
        else:
            self.dismiss([label])

    # ── Multi-select ─────────────────────────────────────────────────────────

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "question-cancel":
            self.dismiss(None)
        elif event.button.id == "question-confirm":
            self._submit_multi()

    def _submit_multi(self) -> None:
        sl: SelectionList = self.query_one("#question-options", SelectionList)
        selected_values: list[str] = [str(v) for v in sl.selected]
        if self.CUSTOM_LABEL in selected_values:
            # Switch to custom input; will merge on Input.Submitted
            selected_values.remove(self.CUSTOM_LABEL)
            self._custom_selected = True
            self._pending_multi = selected_values
            self._show_custom_input()
        else:
            self.dismiss(selected_values if selected_values else [])

    # ── Custom text input ─────────────────────────────────────────────────────

    def _show_custom_input(self) -> None:
        inp: Input = self.query_one("#question-custom-input", Input)
        inp.add_class("visible")
        inp.focus()
        # Hide the option/selection list so input is prominent
        self.query_one("#question-options").display = False

    def on_input_submitted(self, event: Input.Submitted) -> None:
        typed = event.value.strip()
        if self._multiple:
            answers = list(self._pending_multi)
            if typed:
                answers.append(typed)
            self.dismiss(answers if answers else [])
        else:
            self.dismiss([typed] if typed else [])
