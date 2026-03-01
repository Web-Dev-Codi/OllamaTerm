from __future__ import annotations

import logging

from pydantic import BaseModel, ValidationError

from ..support import question_service
from .base import ParamsSchema, Tool, ToolContext, ToolResult

LOGGER = logging.getLogger(__name__)


class QuestionOption(BaseModel):
    label: str
    description: str


class QuestionInfo(BaseModel):
    header: str
    question: str
    options: list[QuestionOption]
    multiple: bool = False
    custom: bool = True


class QuestionParams(ParamsSchema):
    questions: list[QuestionInfo]


class QuestionTool(Tool):
    id = "question"
    params_schema = QuestionParams
    description = (
        "Ask the user clarifying questions when their request is ambiguous, requires "
        "important decisions, or has multiple valid interpretations. Use this tool "
        "PROACTIVELY - asking is better than guessing wrong. Provide 2-5 specific options "
        "for the user to choose from. Set 'custom=True' to allow free-text answers. "
        "Set 'multiple=True' to allow selecting multiple options. "
        "Ask ONE focused question per call. The user will see your question in a modal "
        "dialog and select their preferred option. This suspends execution until answered."
    )

    def format_validation_error(self, error: ValidationError) -> str:
        """Provide LLM-friendly error message for validation failures."""
        return (
            "Question tool validation failed. Please ensure:\n\n"
            "1. Each question has a 'header' field (short label, max 30 chars)\n"
            "2. Each question has a 'question' field (full question text)\n"
            "3. 'options' is a list of objects with 'label' and 'description' fields\n\n"
            "CORRECT FORMAT:\n"
            "{\n"
            '  "questions": [{\n'
            '    "header": "Database Choice",\n'
            '    "question": "Which database should I use?",\n'
            '    "options": [\n'
            '      {"label": "PostgreSQL", "description": "Relational, ACID compliant"},\n'
            '      {"label": "MongoDB", "description": "Document-based, flexible schema"}\n'
            "    ]\n"
            "  }]\n"
            "}\n\n"
            "WRONG FORMAT (options as strings):\n"
            '  "options": ["PostgreSQL", "MongoDB"]  ❌\n\n'
            f"Validation error details:\n{error}"
        )

    async def execute(self, params: QuestionParams, ctx: ToolContext) -> ToolResult:
        # Log question usage for tracking and analysis
        LOGGER.info(
            "question_tool.invoked",
            extra={
                "event": "question_tool.invoked",
                "num_questions": len(params.questions),
                "session_id": ctx.session_id,
                "agent": ctx.agent,
                "questions": [q.question for q in params.questions],
            },
        )

        answers = await question_service.ask(
            session_id=ctx.session_id,
            questions=[q.model_dump() for q in params.questions],
            tool={"message_id": ctx.message_id, "call_id": ctx.call_id},
        )
        pairs: list[str] = []
        for q, ans in zip(params.questions, answers, strict=True):
            pairs.append(f'"{q.question}"="{", ".join(ans)}"')
        output = (
            "User has answered your questions: "
            + "; ".join(pairs)
            + ". You can now continue with the user's answers in mind."
        )
        return ToolResult(
            title="Question Answered",
            output=output,
            metadata={"answers": answers},
        )
