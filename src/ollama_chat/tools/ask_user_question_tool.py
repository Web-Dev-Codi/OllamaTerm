from __future__ import annotations

from pydantic import Field

from ..support import question_service
from .base import ParamsSchema, Tool, ToolContext, ToolResult


class AskUserQuestionParams(ParamsSchema):
    question: str = Field(description="The question to present to the user")
    options: list[str] = Field(
        description="List of 2–5 answer options to display",
        min_length=2,
        max_length=5,
    )


class AskUserQuestionTool(Tool):
    id = "ask_user_question"
    description = (
        "Ask the user a multiple-choice question when you need clarification before proceeding. "
        "Use this instead of guessing. Provide 2 to 5 clear, distinct options. "
        "The user may also type a custom answer."
    )
    params_schema = AskUserQuestionParams

    async def execute(
        self, params: AskUserQuestionParams, ctx: ToolContext
    ) -> ToolResult:
        question = (params.question or "").strip()
        options = [str(opt).strip() for opt in (params.options or []) if str(opt).strip()]

        if not question:
            return ToolResult(
                title="Question",
                output="User did not answer.",
                metadata={"answer": None},
            )

        if len(options) < 2:
            return ToolResult(
                title="Question",
                output="User did not answer.",
                metadata={"answer": None},
            )

        questions = [
            {
                "header": "Assistant Question",
                "question": question,
                "options": [{"label": opt, "description": ""} for opt in options],
                "multiple": False,
                "custom": True,
            }
        ]

        answers = await question_service.ask(session_id=ctx.session_id, questions=questions)
        chosen: str | None = None
        if answers and answers[0]:
            chosen = str(answers[0][0]).strip() if str(answers[0][0]).strip() else None

        if not chosen:
            return ToolResult(
                title="Question",
                output="User did not answer.",
                metadata={"answer": None},
            )

        return ToolResult(
            title="Question",
            output=chosen,
            metadata={"answer": chosen},
        )
