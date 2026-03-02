"""Tests for the ToolRegistry."""

from __future__ import annotations

import unittest

from ollama_chat.exceptions import OllamaToolError
from ollama_chat.tooling import (
    ToolRegistry,
    ToolRegistryOptions,
    ToolRuntimeOptions,
    build_default_registry,
    build_registry,
)
from ollama_chat.tools.registry import ToolRegistry as BuiltinToolRegistry


def _add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: First integer.
        b: Second integer.

    Returns:
        The sum.
    """
    return a + b


def _failing_tool(x: str) -> str:
    """A tool that always raises.

    Args:
        x: Input string.

    Returns:
        Never returns.
    """
    raise ValueError(f"bad input: {x}")


class ToolRegistryTests(unittest.TestCase):
    """Unit tests for ToolRegistry."""

    def test_register_and_build_tools_list(self) -> None:
        registry = ToolRegistry()
        registry.register(_add)
        tools = registry.build_tools_list()
        self.assertEqual(len(tools), 1)
        self.assertIs(tools[0], _add)

    def test_execute_known_tool_returns_string(self) -> None:
        registry = ToolRegistry()
        registry.register(_add)
        result = registry.execute("_add", {"a": 3, "b": 4})
        self.assertEqual(result, "7")

    def test_execute_unknown_tool_raises(self) -> None:
        registry = ToolRegistry()
        with self.assertRaises(OllamaToolError) as ctx:
            registry.execute("nonexistent", {})
        self.assertIn("nonexistent", str(ctx.exception))

    def test_execute_tool_exception_wrapped_as_tool_error(self) -> None:
        registry = ToolRegistry()
        registry.register(_failing_tool)
        with self.assertRaises(OllamaToolError) as ctx:
            registry.execute("_failing_tool", {"x": "oops"})
        self.assertIn("_failing_tool", str(ctx.exception))

    def test_is_empty_true_when_no_tools(self) -> None:
        registry = ToolRegistry()
        self.assertTrue(registry.is_empty)

    def test_is_empty_false_after_register(self) -> None:
        registry = ToolRegistry()
        registry.register(_add)
        self.assertFalse(registry.is_empty)

    def test_build_default_registry_empty_when_web_search_disabled(self) -> None:
        registry = build_default_registry(web_search_enabled=False)
        self.assertTrue(registry.is_empty)

    def test_build_default_registry_registers_web_tools_when_enabled(self) -> None:
        registry = build_default_registry(
            web_search_enabled=True, web_search_api_key="test-key"
        )
        tools = registry.build_tools_list()
        tool_names = [fn.__name__ for fn in tools]
        # Inner factory functions are named _web_search_tool / _web_fetch_tool.
        self.assertTrue(any("web_search" in n for n in tool_names))
        self.assertTrue(any("web_fetch" in n for n in tool_names))

    def test_build_default_registry_raises_without_api_key(self) -> None:
        # Ensure OLLAMA_API_KEY is not set for this test.
        import os

        from ollama_chat.exceptions import OllamaToolError

        old = os.environ.pop("OLLAMA_API_KEY", None)
        try:
            with self.assertRaises(OllamaToolError):
                build_default_registry(web_search_enabled=True, web_search_api_key="")
        finally:
            if old is not None:
                os.environ["OLLAMA_API_KEY"] = old

    def test_multiple_registrations_do_not_duplicate(self) -> None:
        registry = ToolRegistry()
        registry.register(_add)
        registry.register(_add)
        # Second registration overwrites the first (same name key).
        self.assertEqual(len(registry.build_tools_list()), 1)

    def test_schema_tools_are_exported(self) -> None:
        registry = build_registry(ToolRegistryOptions())
        tools = registry.build_tools_list()
        schema_tools = [item for item in tools if isinstance(item, dict)]
        self.assertTrue(
            any(item["function"]["name"] == "read" for item in schema_tools)
        )
        self.assertTrue(
            any(item["function"]["name"] == "bash" for item in schema_tools)
        )

    def test_builtin_adapter_allowlist_and_execution(self) -> None:
        # Built-in tools package is enabled by default.
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry = build_registry(
                ToolRegistryOptions(
                    runtime_options=ToolRuntimeOptions(workspace_root=str(root)),
                )
            )
            names = set(registry.list_tool_names())
            # Built-in tools should be present
            for name in {"codesearch", "edit", "grep", "list", "read"}:
                self.assertIn(name, names)
            # Write tool should also be present now (no longer filtered)
            self.assertIn("write", names)

            # Verify read executes by creating a file and reading it
            target = root / "foo.txt"
            target.write_text("hello\nworld\n", encoding="utf-8")
            read_out = registry.execute("read", {"file_path": str(target), "limit": 1})
            self.assertIn("hello", read_out)

            # grep may use ripgrep if available; still should not crash on a simple pattern
            grep_out = registry.execute("grep", {"pattern": "hello", "path": str(root)})
            self.assertIn("Found", grep_out)

    def test_schema_tool_validation_rejects_missing_required_argument(self) -> None:
        registry = build_registry(ToolRegistryOptions())
        with self.assertRaises(OllamaToolError):
            registry.execute("read", {})

    def test_truncation_applies_to_schema_tool_outputs(self) -> None:
        registry = build_registry(
            ToolRegistryOptions(
                runtime_options=registry_runtime_options(
                    max_output_lines=2,
                    max_output_bytes=5000,
                ),
            )
        )
        registry.execute("todo", {"item": "line-1"})
        registry.execute("todo", {"item": "line-2"})
        registry.execute("todo", {"item": "line-3"})
        rendered = registry.execute("todoread", {})
        self.assertIn("truncated", rendered)


def registry_runtime_options(
    *,
    max_output_lines: int,
    max_output_bytes: int,
) -> ToolRuntimeOptions:
    return ToolRuntimeOptions(
        max_output_lines=max_output_lines,
        max_output_bytes=max_output_bytes,
    )


class TestWebToolGating(unittest.TestCase):
    """Verify WebSearchTool/WebFetchTool are only included when configured."""

    def test_web_tools_excluded_by_default(self) -> None:
        """WebSearchTool and WebFetchTool must NOT appear when include_web_tools=False."""
        reg = BuiltinToolRegistry.build_default(include_web_tools=False)
        tool_ids = {t.id for t in reg.all()}
        self.assertNotIn("websearch", tool_ids)
        self.assertNotIn("webfetch", tool_ids)

    def test_ask_user_question_is_included_and_question_is_deprecated(self) -> None:
        """ask_user_question must be present; legacy question tool must not be registered."""
        reg = BuiltinToolRegistry.build_default(include_web_tools=False)
        tool_ids = {t.id for t in reg.all()}
        self.assertIn("ask_user_question", tool_ids)
        self.assertNotIn("question", tool_ids)

        schema = reg.build_ollama_tools()
        ask_tool = next(
            tool
            for tool in schema
            if tool.get("function", {}).get("name") == "ask_user_question"
        )
        params = ask_tool["function"]["parameters"]
        self.assertEqual(params["type"], "object")
        self.assertIn("options", params["properties"])

        options_schema = params["properties"]["options"]
        self.assertEqual(options_schema["type"], "array")
        self.assertEqual(options_schema["items"]["type"], "string")

    def test_web_tools_included_when_requested(self) -> None:
        """WebSearchTool and WebFetchTool MUST appear when include_web_tools=True."""
        reg = BuiltinToolRegistry.build_default(include_web_tools=True)
        tool_ids = {t.id for t in reg.all()}
        self.assertIn("websearch", tool_ids)
        self.assertIn("webfetch", tool_ids)

    def test_build_registry_excludes_web_tools_when_disabled(self) -> None:
        """build_registry with default ToolRuntimeOptions excludes web tools."""
        opts = ToolRegistryOptions(
            enable_builtin_tools=True,
            runtime_options=ToolRuntimeOptions(include_web_tools=False),
        )
        reg = build_registry(opts)
        names = reg.list_tool_names()
        self.assertNotIn("websearch", names)
        self.assertNotIn("webfetch", names)

    def test_build_registry_includes_web_tools_when_enabled(self) -> None:
        """build_registry with include_web_tools=True includes web tools."""
        opts = ToolRegistryOptions(
            enable_builtin_tools=True,
            runtime_options=ToolRuntimeOptions(include_web_tools=True),
        )
        reg = build_registry(opts)
        names = reg.list_tool_names()
        self.assertIn("websearch", names)
        self.assertIn("webfetch", names)


if __name__ == "__main__":
    unittest.main()
