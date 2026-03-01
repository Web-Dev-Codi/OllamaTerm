"""Unit tests for QuestionScreen modal."""

from __future__ import annotations

import pytest

SINGLE_Q = {
    "question": "Which format do you prefer?",
    "header": "Format",
    "options": [
        {"label": "JSON", "description": "Machine-readable"},
        {"label": "YAML", "description": "Human-readable"},
    ],
    "multiple": False,
    "custom": False,
}

MULTI_Q = {
    "question": "Pick all that apply",
    "header": "Multi",
    "options": [
        {"label": "A", "description": "Option A"},
        {"label": "B", "description": "Option B"},
    ],
    "multiple": True,
    "custom": False,
}

CUSTOM_Q = {
    "question": "Name the tool",
    "header": "Tool",
    "options": [
        {"label": "pytest", "description": "Python test runner"},
    ],
    "multiple": False,
    "custom": True,
}


@pytest.mark.asyncio
async def test_single_select_returns_label() -> None:
    """Selecting an option in single-select mode dismisses with [label]."""
    from textual.app import App

    from ollama_chat.screens import QuestionScreen

    results: list = []

    class _App(App):
        def on_mount(self) -> None:
            async def _show() -> None:
                result = await self.push_screen_wait(QuestionScreen(SINGLE_Q))
                results.append(result)
                self.exit()

            self.run_worker(_show())

    async with _App().run_test(headless=True) as pilot:
        await pilot.pause()
        # Highlight first option and press Enter
        await pilot.press("enter")
        await pilot.pause()

    assert results == [["JSON"]]


@pytest.mark.asyncio
async def test_escape_returns_none() -> None:
    """Pressing Escape dismisses with None."""
    from textual.app import App

    from ollama_chat.screens import QuestionScreen

    results: list = []

    class _App(App):
        def on_mount(self) -> None:
            async def _show() -> None:
                result = await self.push_screen_wait(QuestionScreen(SINGLE_Q))
                results.append(result)
                self.exit()

            self.run_worker(_show())

    async with _App().run_test(headless=True) as pilot:
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()

    assert results == [None]


@pytest.mark.asyncio
async def test_multi_select_confirm() -> None:
    """Multi-select: Space toggles, Confirm submits selected labels."""
    from textual.app import App

    from ollama_chat.screens import QuestionScreen

    results: list = []

    class _App(App):
        def on_mount(self) -> None:
            async def _show() -> None:
                result = await self.push_screen_wait(QuestionScreen(MULTI_Q))
                results.append(result)
                self.exit()

            self.run_worker(_show())

    async with _App().run_test(headless=True) as pilot:
        await pilot.pause()
        await pilot.press("space")  # toggle first item (A)
        await pilot.press("down")
        await pilot.press("space")  # toggle second item (B)
        await pilot.click("#question-confirm")
        await pilot.pause()

    assert results == [["A", "B"]]


@pytest.mark.asyncio
async def test_custom_entry_visible_when_enabled() -> None:
    """When custom=True, the last option in the list is 'Type your own answer...'."""
    from textual.app import App

    from ollama_chat.screens import QuestionScreen

    class _App(App):
        async def on_mount(self) -> None:
            await self.push_screen(QuestionScreen(CUSTOM_Q))

    async with _App().run_test(headless=True) as pilot:
        await pilot.pause()
        # The OptionList should have 2 options: the real one + custom
        option_list = pilot.app.screen.query_one("#question-options")
        assert option_list.option_count == 2
        await pilot.press("escape")
