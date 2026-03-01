from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
import uuid

from .bus import bus


@dataclass
class _Pending:
    future: asyncio.Future
    session_id: str
    questions: list[dict[str, Any]]


_pending: dict[str, _Pending] = {}


def _new_id(prefix: str = "q") -> str:
    """Generate a unique question ID using UUID to ensure uniqueness."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


async def ask(
    *,
    session_id: str,
    questions: list[dict[str, Any]],
    tool: dict[str, Any] | None = None,
) -> list[list[str]]:
    """Publish a question event and await an answer.

    This function waits indefinitely (timeout=None) for a reply. The UI must
    call reply() to resolve the future and unblock tool execution.

    Args:
        session_id: The session requesting the question
        questions: List of question dictionaries with header, question, options, etc.
        tool: Optional tool metadata (message_id, call_id) for tracing/debugging

    Returns:
        List of answers, one per question. Each answer is a list of selected strings.
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
