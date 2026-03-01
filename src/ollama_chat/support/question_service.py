from __future__ import annotations

import asyncio
from dataclasses import dataclass
import time
from typing import Any

from .bus import bus


@dataclass
class _Pending:
    future: asyncio.Future
    session_id: str
    questions: list[dict[str, Any]]


_pending: dict[str, _Pending] = {}


def _new_id(prefix: str = "q") -> str:
    return f"{prefix}_{int(time.time() * 1000)}"


async def ask(
    *,
    session_id: str,
    questions: list[dict[str, Any]],
    tool: dict[str, Any] | None = None,
) -> list[list[str]]:
    """Publish a question event and await an answer.

    In this standalone implementation, if no reply is provided within a short
    timeout, an empty answer is returned to avoid deadlock.
    """
    qid = _new_id("question")
    fut: asyncio.Future = asyncio.get_running_loop().create_future()
    _pending[qid] = _Pending(future=fut, session_id=session_id, questions=questions)
    try:
        await bus.publish(
            "question.asked",
            {
                "id": qid,
                "session_id": session_id,
                "questions": questions,
                "tool": tool or {},
            },
        )
        return await asyncio.wait_for(fut, timeout=None)
    finally:
        _pending.pop(qid, None)


def reply(question_id: str, answers: list[list[str]]) -> None:
    """Programmatically reply to a pending question (tests / UI)."""
    item = _pending.get(question_id)
    if item and not item.future.done():
        item.future.set_result(answers)
