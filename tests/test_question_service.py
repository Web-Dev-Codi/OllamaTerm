"""Tests for question_service ask/reply round-trip."""

from __future__ import annotations

import asyncio
import unittest


class TestQuestionServiceReply(unittest.IsolatedAsyncioTestCase):
    """question_service.reply() must resolve ask() before any timeout."""

    async def test_reply_resolves_ask(self) -> None:
        """ask() returns the answers passed to reply() without timing out."""
        from ollama_chat.support import question_service

        # Clear any stale state from other tests
        question_service._pending.clear()

        received: list = []

        async def _ask() -> None:
            result = await question_service.ask(
                session_id="test",
                questions=[{"question": "Pick one", "header": "Q", "options": []}],
            )
            received.extend(result)

        async def _reply_after_tick() -> None:
            await asyncio.sleep(0)  # yield to let ask() register the future
            qid = next(iter(question_service._pending))
            question_service.reply(qid, [["Option A"]])

        await asyncio.gather(_ask(), _reply_after_tick())
        assert received == [["Option A"]]

    async def test_empty_reply_resolves_ask(self) -> None:
        """reply() with empty answers resolves ask() cleanly."""
        from ollama_chat.support import question_service

        question_service._pending.clear()
        received: list = []

        async def _ask() -> None:
            result = await question_service.ask(
                session_id="test",
                questions=[{"question": "Q", "header": "H", "options": []}],
            )
            received.extend(result)

        async def _reply() -> None:
            await asyncio.sleep(0)
            qid = next(iter(question_service._pending))
            question_service.reply(qid, [[]])

        await asyncio.gather(_ask(), _reply())
        assert received == [[]]
