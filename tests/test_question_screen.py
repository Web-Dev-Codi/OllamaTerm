"""Unit tests for AskQuestionWidget inline question UI."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_enter_selects_first_option() -> None:
    """Pressing Enter on the OptionList posts Answered() with the highlighted option."""
    from textual.app import App

    from ollama_chat.widgets.ask_question_widget import AskQuestionWidget

    results: list = []

    class _App(App):
        def compose(self):  # type: ignore[override]
            yield AskQuestionWidget()

        def on_mount(self) -> None:
            w = self.query_one(AskQuestionWidget)
            w.display = True
            w.load_question("Which format do you prefer?", ["JSON", "YAML"], custom=False)
            w.query_one("#aq-options").focus()

        async def on_ask_question_widget_answered(self, message: AskQuestionWidget.Answered) -> None:
            results.append(message.value)
            self.exit()

    async with _App().run_test(headless=True) as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

    assert results == ["JSON"]


@pytest.mark.asyncio
async def test_escape_posts_none() -> None:
    """Pressing Escape cancels and posts Answered(None)."""
    from textual.app import App

    from ollama_chat.widgets.ask_question_widget import AskQuestionWidget

    results: list = []

    class _App(App):
        def compose(self):  # type: ignore[override]
            yield AskQuestionWidget()

        def on_mount(self) -> None:
            w = self.query_one(AskQuestionWidget)
            w.display = True
            w.load_question("Pick one", ["A", "B"], custom=False)
            w.query_one("#aq-options").focus()

        async def on_ask_question_widget_answered(self, message: AskQuestionWidget.Answered) -> None:
            results.append(message.value)
            self.exit()

    async with _App().run_test(headless=True) as pilot:
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()

    assert results == [None]


@pytest.mark.asyncio
async def test_escape_does_not_trigger_app_escape_binding() -> None:
    """Escape should cancel the question locally, not trigger app-level escape actions."""
    from textual.app import App

    from ollama_chat.widgets.ask_question_widget import AskQuestionWidget

    results: list = []

    class _App(App):
        BINDINGS = [("escape", "interrupt_stream", "Interrupt")]

        def __init__(self) -> None:
            super().__init__()
            self.interrupt_triggered = False

        def compose(self):  # type: ignore[override]
            yield AskQuestionWidget()

        def on_mount(self) -> None:
            w = self.query_one(AskQuestionWidget)
            w.display = True
            w.load_question("Pick one", ["A", "B"], custom=False)
            w.query_one("#aq-options").focus()

        def action_interrupt_stream(self) -> None:
            self.interrupt_triggered = True

        async def on_ask_question_widget_answered(self, message: AskQuestionWidget.Answered) -> None:
            results.append(message.value)
            self.exit()

    app = _App()
    async with app.run_test(headless=True) as pilot:
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()

    assert results == [None]
    assert app.interrupt_triggered is False


@pytest.mark.asyncio
async def test_number_hotkey_selects_option() -> None:
    """Pressing number keys selects the corresponding option."""
    from textual.app import App

    from ollama_chat.widgets.ask_question_widget import AskQuestionWidget

    results: list = []

    class _App(App):
        def compose(self):  # type: ignore[override]
            yield AskQuestionWidget()

        def on_mount(self) -> None:
            w = self.query_one(AskQuestionWidget)
            w.display = True
            w.load_question("Pick one", ["A", "B"], custom=False)
            w.query_one("#aq-options").focus()

        async def on_ask_question_widget_answered(self, message: AskQuestionWidget.Answered) -> None:
            results.append(message.value)
            self.exit()

    async with _App().run_test(headless=True) as pilot:
        await pilot.pause()
        await pilot.press("2")
        await pilot.pause()

    assert results == ["B"]


@pytest.mark.asyncio
async def test_custom_input_submits_value() -> None:
    """Submitting the custom Input posts Answered(value) when non-empty."""
    from textual.app import App

    from ollama_chat.widgets.ask_question_widget import AskQuestionWidget

    results: list = []

    class _App(App):
        def compose(self):  # type: ignore[override]
            yield AskQuestionWidget()

        def on_mount(self) -> None:
            w = self.query_one(AskQuestionWidget)
            w.display = True
            w.load_question("Type a value", ["One", "Two"], custom=True)
            inp = w.query_one("#aq-custom-input")
            inp.value = "pytest"
            inp.focus()

        async def on_ask_question_widget_answered(self, message: AskQuestionWidget.Answered) -> None:
            results.append(message.value)
            self.exit()

    async with _App().run_test(headless=True) as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

    assert results == ["pytest"]
