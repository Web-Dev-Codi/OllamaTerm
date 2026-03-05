from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .apply_patch_tool import ApplyPatchTool
from .ask_user_question_tool import AskUserQuestionTool
from .base import Tool
from .bash_tool import BashTool
from .batch_tool import BatchTool
from .codesearch_tool import CodeSearchTool
from .edit_tool import EditTool
from .glob_tool import GlobTool
from .grep_tool import GrepTool
from .invalid_tool import InvalidTool
from .ls_tool import ListTool
from .lsp_tool import LspTool
from .multiedit_tool import MultiEditTool
from .plan_tool import PlanExitTool
from .read_tool import ReadTool
from .skill_tool import SkillTool
from .task_tool import TaskTool
from .todo_tool import TodoReadTool, TodoWriteTool
from .webfetch_tool import WebFetchTool
from .websearch_tool import WebSearchTool
from .write_tool import WriteTool


@dataclass
class ToolDefinition:
    name: str
    factory: type[Tool]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.id] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def tools_for_model(
        self,
        provider_id: str | None = None,
        model_id: str | None = None,
        agent: str | None = None,
    ) -> list[Tool]:  # noqa: D401 - simple
        # Minimal filtering: return all built-ins. Environment gating is not enforced here.
        return self.all()

    def build_ollama_tools(self) -> list[dict[str, Any]]:
        """Build Ollama-formatted tool list with proper schema wrapping.

        Returns a list of tools in Ollama's expected format:
        [
            {
                "type": "function",
                "function": {
                    "name": "tool_name",
                    "description": "...",
                    "parameters": {...}
                }
            },
            ...
        ]

        This method replaces the old pattern of mixing callable functions
        with schema dicts.
        """
        tools: list[dict[str, Any]] = []
        for tool in self.all():
            tools.append(tool.to_ollama_schema())
        return tools

    @classmethod
    def build_default(cls, include_web_tools: bool = False) -> ToolRegistry:
        reg = cls()
        # Built-in tool order roughly as specified
        reg.register(InvalidTool())
        reg.register(AskUserQuestionTool())
        reg.register(BashTool())
        reg.register(ReadTool())
        reg.register(GlobTool())
        reg.register(GrepTool())
        reg.register(EditTool())
        reg.register(WriteTool())
        reg.register(TaskTool())
        if include_web_tools:
            reg.register(WebFetchTool())
        reg.register(TodoWriteTool())
        reg.register(TodoReadTool())
        if include_web_tools:
            reg.register(WebSearchTool())
        reg.register(CodeSearchTool())
        reg.register(SkillTool())
        reg.register(ApplyPatchTool())
        reg.register(MultiEditTool())
        reg.register(ListTool())
        reg.register(LspTool())
        reg.register(BatchTool())
        reg.register(PlanExitTool())
        return reg


# Provide a simple singleton accessor
_default_registry: ToolRegistry | None = None


def get_registry() -> ToolRegistry:
    # NOTE: This singleton always uses include_web_tools=False.
    # Web tool gating is handled by ToolsPackageAdapter.to_specs() which calls
    # build_default(include_web_tools=...) directly. Do not use this singleton
    # for contexts where web tools may be needed.
    global _default_registry
    if _default_registry is None:
        _default_registry = ToolRegistry.build_default()
    return _default_registry
