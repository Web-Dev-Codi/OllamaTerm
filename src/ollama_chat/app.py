"""Main Textual application for chatting with Ollama."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime
import inspect
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any
import urllib.parse

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.events import Key, Paste
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Header, Input, OptionList, Static

from .capabilities import AttachmentState, CapabilityContext, SearchState
from .chat import ChatSendOptions, OllamaChat
from .commands import parse_inline_directives
from .config import load_config
from .events.bus import event_bus as app_event_bus
from .exceptions import (
    OllamaChatError,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaStreamingError,
    OllamaToolError,
)
from .logging_utils import configure_logging
from .managers import (
    AttachmentManager,
    CapabilityManager,
    CommandManager,
    ConnectionManager,
    ConversationManager,
    MessageRenderer,
    StreamManager,
    ThemeManager,
)
from .persistence import ConversationPersistence
from .plugins.interface import PluginManager
from .screens import (
    ConversationPickerScreen,
    ImageAttachScreen,
    InfoScreen,
    SimplePickerScreen,
    TextPromptScreen,
    ThemePickerScreen,
)
from .state import ConnectionState, ConversationState, StateManager
from .task_manager import TaskManager
from .tooling import (
    ToolRegistry,
    ToolRegistryOptions,
    ToolRuntimeOptions,
    ToolSpec,
    _run_async_from_sync,
    build_registry,
)
from .tools.base import ToolContext
from .widgets.activity_bar import ActivityBar
from .widgets.conversation import ConversationView
from .widgets.input_box import InputBox
from .widgets.message import MessageBubble
from .widgets.status_bar import StatusBar

LOGGER = logging.getLogger(__name__)

# Image file extensions imported from AttachmentManager
# (defined in managers/attachment.py)

_STREAM_ERROR_MESSAGES: dict[type, tuple[str, str]] = {
    OllamaToolError: ("Tool error: {exc}", "Tool execution error"),
    OllamaConnectionError: ("Connection error: {exc}", "Connection error"),
    OllamaModelNotFoundError: (
        "Model not found. Verify the configured ollama.model value.",
        "Model not found",
    ),
    OllamaStreamingError: ("Streaming error: {exc}", "Streaming error"),
    OllamaChatError: (
        "Chat error. Please review settings and try again.",
        "Chat error",
    ),
}

_SlashCommand = Callable[[str], Awaitable[None]]


async def _open_native_file_dialog(
    title: str = "Open File",
    file_filter: list[tuple[str, list[str]]] | None = None,
) -> str | None:
    """Open a native Linux file picker, trying multiple backends.

    Tries xdg-desktop-portal (via gdbus), then zenity, then kdialog.
    Returns the selected file path or None if cancelled/unavailable.
    """

    # --- Portal via gdbus (Wayland/Hyprland-friendly) ---
    gdbus_bin = shutil.which("gdbus")
    if gdbus_bin is not None:
        try:
            handle_token = f"ollamaterm_{os.getpid()}"
            proc = await asyncio.create_subprocess_exec(
                gdbus_bin,
                "call",
                "--session",
                "--dest=org.freedesktop.portal.Desktop",
                "--object-path=/org/freedesktop/portal/desktop",
                "--method=org.freedesktop.portal.FileChooser.OpenFile",
                "",
                title,
                f"{{'handle_token': <'{handle_token}'>}}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=60)
            if proc.returncode == 0:
                output = stdout.decode().strip()
                if "file://" in output:
                    for token in output.split():
                        cleaned = token.strip("',()><[]")
                        if cleaned.startswith("file://"):
                            return urllib.parse.unquote(cleaned[len("file://") :])
        except (TimeoutError, OSError):
            pass

    # --- zenity ---
    zenity_bin = shutil.which("zenity")
    if zenity_bin is not None:
        cmd: list[str] = [zenity_bin, "--file-selection", f"--title={title}"]
        if file_filter:
            for name, patterns in file_filter:
                cmd.append(f"--file-filter={name} | {' '.join(patterns)}")
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=120)
            if proc.returncode == 0:
                path = stdout.decode().strip()
                if path:
                    return path
        except (TimeoutError, OSError):
            pass

    # --- kdialog ---
    kdialog_bin = shutil.which("kdialog")
    if kdialog_bin is not None:
        cmd = [kdialog_bin, "--getopenfilename", ".", title]
        if file_filter:
            filter_str = " ".join(p for _, patterns in file_filter for p in patterns)
            cmd = [kdialog_bin, "--getopenfilename", ".", filter_str]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=120)
            if proc.returncode == 0:
                path = stdout.decode().strip()
                if path:
                    return path
        except (TimeoutError, OSError):
            pass

    return None


def _is_regular_file(path: Path) -> bool:
    try:
        return path.is_file() and not path.is_symlink()
    except OSError:
        return False


def _is_within_home(path: Path) -> bool:
    try:
        home = Path.home().resolve(strict=False)
        resolved = path.resolve(strict=False)
        resolved.relative_to(home)
        return True
    except Exception:
        return False


def _validate_attachment(
    raw_path: str,
    *,
    kind: str,
    max_bytes: int,
    allowed_extensions: set[str] | None = None,
    home_only: bool = False,
) -> tuple[bool, str, Path | None]:
    expanded = Path(os.path.expanduser(raw_path))
    if not _is_regular_file(expanded):
        return False, f"{kind.capitalize()} not found: {expanded}", None
    if home_only and not _is_within_home(expanded):
        return False, f"{kind.capitalize()} must be inside your home directory.", None
    if allowed_extensions is not None:
        ext = expanded.suffix.lower()
        if ext not in allowed_extensions:
            return False, f"Unsupported {kind} type: {ext}", None
    try:
        size = expanded.stat().st_size
    except OSError:
        return False, f"Unable to read {kind} size.", None
    if size > max_bytes:
        return (
            False,
            f"{kind.capitalize()} too large ({size} bytes). Max is {max_bytes} bytes.",
            None,
        )
    return True, str(expanded), expanded


class ModelPickerScreen(ModalScreen[str | None]):
    """Modal picker for selecting a configured Ollama model."""

    CSS = """
    ModelPickerScreen {
        align: center middle;
    }

    #model-picker-dialog {
        width: 50;
        max-height: 22;
        padding: 1 2;
        border: round $panel;
        background: $surface;
    }

    #model-picker-title {
        padding-bottom: 1;
        text-style: bold;
    }

    #model-picker-help {
        padding-top: 1;
    }
    """

    BINDINGS = [Binding("escape", "cancel", "Close", show=False)]

    def __init__(self, models: list[str], active_model: str) -> None:
        super().__init__()
        self.models = models
        self.active_model = active_model

    def compose(self) -> ComposeResult:
        with Container(id="model-picker-dialog"):
            yield Static("Select model from config", id="model-picker-title")
            yield OptionList(*self.models, id="model-picker-options")
            yield Static(
                "Enter/click to select  |  Esc to cancel", id="model-picker-help"
            )

    def on_mount(self) -> None:
        options = self.query_one("#model-picker-options", OptionList)
        selected_index = 0
        for index, model_name in enumerate(self.models):
            if OllamaChat._model_name_matches(self.active_model, model_name):
                selected_index = index
                break
        options.highlighted = selected_index

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        option_index = getattr(event, "option_index", None)
        if option_index is None:
            option_index = getattr(event, "index", -1)
        try:
            selected_index = int(option_index or -1)
        except (TypeError, ValueError):
            selected_index = -1
        if 0 <= selected_index < len(self.models):
            self.dismiss(self.models[selected_index])

    def action_cancel(self) -> None:
        self.dismiss(None)


class OllamaChatApp(App[None]):
    """ChatGPT-style TUI app powered by local Ollama models."""

    CSS = """
    Screen {
        layout: vertical;
        background: $background;
    }

    #app-root {
        layout: vertical;
        width: 100%;
        height: 1fr;
        background: $background;
    }

    Header {
        border-bottom: solid $panel;
        background: $surface;
    }

    Footer {
        border-top: solid $panel;
        background: $surface;
    }

    #conversation {
        height: 1fr;
        padding: 1;
    }

    InputBox {
        height: auto;
        padding: 0 1 1 1;
        border-top: solid $panel;
        background: $surface;
    }

    #message_input {
        width: 1fr;
    }

    #attach_button {
        margin-left: 1;
        min-width: 10;
    }

    #file_button {
        margin-left: 1;
        min-width: 10;
    }

    #send_button {
        margin-left: 1;
        min-width: 10;
    }

    #input_row {
        height: auto;
    }

    #status_bar {
        height: auto;
        padding: 0 1;
        border-top: solid $panel;
        background: $surface;
    }

    #activity_bar {
        height: auto;
        min-height: 2;
        padding: 0 1 0 1;
        border-top: dashed $panel;
        background: $surface;
    }

    #slash_menu {
        max-height: 8;
        width: 60;
        margin-top: 1;
    }

    #slash_menu.hidden {
        display: none;
    }

    MessageBubble {
        width: 85%;
        margin: 1 0;
        padding: 1 2;
        border: round $panel;
    }

    .message-user {
        align-horizontal: right;
        background: $primary;
    }

    .message-assistant {
        align-horizontal: left;
        background: $surface;
    }
    """

    DEFAULT_ACTION_DESCRIPTIONS: dict[str, str] = {
        "send_message": "Send",
        "new_conversation": "New Chat",
        "quit": "Quit",
        "scroll_up": "Scroll Up",
        "scroll_down": "Scroll Down",
        "command_palette": "🧭 Palette",
        "toggle_model_picker": "Model",
        "save_conversation": "Save",
        "load_conversation": "Load",
        "export_conversation": "Export",
        "search_messages": "Search",
        "copy_last_message": "Copy Last",
        "toggle_conversation_picker": "Conversations",
        "toggle_prompt_preset_picker": "Prompt",
        "toggle_theme_picker": "Theme",
        "interrupt_stream": "Interrupt",
    }

    def __init__(self) -> None:
        self.config = load_config()
        self.window_title = str(self.config["app"]["title"])
        self.window_class = str(self.config["app"]["class"])
        configure_logging(self.config["logging"])
        LOGGER.info(
            "app.python",
            extra={
                "event": "app.python",
                "executable": sys.executable,
                "version": sys.version.split()[0],
            },
        )

        ollama_cfg = self.config["ollama"]
        configured_default_model = str(ollama_cfg["model"])
        self._configured_models = self._normalize_configured_models(
            raw_models=ollama_cfg.get("models"),
            default_model=configured_default_model,
        )
        self.chat = OllamaChat(
            host=str(ollama_cfg["host"]),
            model=configured_default_model,
            system_prompt=str(ollama_cfg["system_prompt"]),
            timeout=int(ollama_cfg["timeout"]),
            max_history_messages=int(ollama_cfg["max_history_messages"]),
            max_context_tokens=int(ollama_cfg["max_context_tokens"]),
        )
        self._prompt_presets: dict[str, str] = dict(
            ollama_cfg.get("prompt_presets") or {}
        )
        self._active_prompt_preset: str = str(
            ollama_cfg.get("active_prompt_preset") or ""
        ).strip()
        self.state = StateManager()
        self._task_manager = TaskManager()
        self.connection_manager = ConnectionManager(
            self.chat,
            check_interval_seconds=int(
                self.config["app"]["connection_check_interval_seconds"]
            ),
        )
        self.connection_manager.on_state_change(self._on_connection_state_changed)
        self._search = SearchState()
        self._attachments = AttachmentState()
        self._last_prompt: str = ""

        # Cached widget references — populated in on_mount() after compose().
        # Using cached refs avoids repeated O(widget-tree) query_one() calls in
        # hot paths (every send, every connection-monitor tick, every keystroke).
        self._w_input: Input | None = None
        self._w_send: Button | None = None
        self._w_file: Button | None = None
        self._w_activity: ActivityBar | None = None
        self._w_status: StatusBar | None = None
        self._w_conversation: ConversationView | None = None

        self._slash_commands: list[tuple[str, str]] = [
            ("/image <path>", "Attach image from filesystem"),
            ("/file <path>", "Attach file as context"),
            ("/new", "Start a new conversation"),
            ("/clear", "Clear the input"),
            ("/help", "Show help"),
            ("/model <name>", "Switch active model"),
            ("/preset <name>", "Switch prompt preset"),
            ("/conversations", "Open conversation picker"),
        ]

        # Capabilities configuration (user preferences from config — the ceiling).
        self.capabilities = CapabilityContext.from_config(self.config)

        # Initialize capability manager
        self.capability_manager = CapabilityManager(
            self.chat,
            user_preferences={
                "show_thinking": self.capabilities.show_thinking,
                "web_search_enabled": self.capabilities.web_search_enabled,
                "max_tool_iterations": self.capabilities.max_tool_iterations,
            },
        )

        # Build the tool registry unconditionally — whether tools are actually
        # used is gated at call time by _effective_caps.tools_enabled.  This
        # ensures the registry is ready when the first tool-capable model loads.
        try:
            tools_cfg = self.config.get("tools", {})
            options = ToolRegistryOptions(
                web_search_api_key=(
                    self.capabilities.web_search_api_key
                    if self.capabilities.web_search_enabled
                    else None
                ),
                runtime_options=ToolRuntimeOptions(
                    enabled=bool(tools_cfg.get("enabled", True)),
                    workspace_root=str(tools_cfg.get("workspace_root", ".")),
                    allow_external_directories=bool(
                        tools_cfg.get("allow_external_directories", False)
                    ),
                    command_timeout_seconds=int(
                        tools_cfg.get("command_timeout_seconds", 30)
                    ),
                    max_output_lines=int(tools_cfg.get("max_output_lines", 200)),
                    max_output_bytes=int(tools_cfg.get("max_output_bytes", 50_000)),
                    max_read_bytes=int(tools_cfg.get("max_read_bytes", 200_000)),
                    max_search_results=int(tools_cfg.get("max_search_results", 200)),
                    default_external_directories=tuple(
                        str(item)
                        for item in tools_cfg.get("default_external_directories", [])
                        if str(item).strip()
                    ),
                    include_web_tools=bool(self.capabilities.web_search_enabled),
                ),
            )
            self._tool_registry: ToolRegistry | None = build_registry(options)
        except OllamaToolError as exc:
            LOGGER.warning(
                "app.tools.disabled",
                extra={
                    "event": "app.tools.disabled",
                    "reason": str(exc),
                },
            )
            self._tool_registry = None

        persistence_cfg = self.config["persistence"]
        self.persistence = ConversationPersistence(
            enabled=bool(persistence_cfg["enabled"]),
            directory=str(persistence_cfg["directory"]),
            metadata_path=str(persistence_cfg["metadata_path"]),
        )

        self.conversation_manager = ConversationManager(
            self.chat,
            self.persistence,
            auto_save_enabled=bool(persistence_cfg.get("auto_save", True)),
        )
        self.command_manager = CommandManager()
        self._register_all_commands()
        self.theme_manager = ThemeManager(
            self.config,
            app_name=str(self.config["app"]["title"]).lower().replace(" ", "-"),
            app_author=str(self.config["app"]["class"]),
        )

        self.stream_manager = StreamManager(
            self.chat,
            self.state,
            self._task_manager,
            chunk_size=max(1, int(self.config["ui"]["stream_chunk_size"])),
        )
        self.stream_manager.on_subtitle_change(
            lambda text: setattr(self, "sub_title", text)
        )
        self.stream_manager.on_statusbar_update(self._update_status_bar)

        self.message_renderer = MessageRenderer(
            self.theme_manager,
            self.capability_manager,
        )

        self.attachment_manager = AttachmentManager(
            self._attachments,
            max_image_bytes=10 * 1024 * 1024,  # 10 MB
            max_file_bytes=2 * 1024 * 1024,  # 2 MB
        )
        self.attachment_manager.on_status_update(
            lambda text: setattr(self, "sub_title", text)
        )

        self._last_prompt_path = self.persistence.directory / "last_prompt.txt"
        # Ensure the slash menu/help include all registered commands.
        try:
            existing = {cmd for cmd, _ in self._slash_commands}
            for cmd, desc in self.command_manager.get_commands():
                if cmd not in existing:
                    self._slash_commands.append((cmd, desc))
        except Exception:
            pass
        # Initialize event bus and plugin manager
        self.event_bus = app_event_bus
        self.plugin_manager = PluginManager()
        self._setup_event_subscribers()
        self._load_last_prompt()
        self._binding_specs = self._binding_specs_from_config(self.config)
        self._apply_terminal_window_identity()
        super().__init__()

    def _register_all_commands(self) -> None:
        self.command_manager.register(
            "clear", self._handle_clear_command, "Clear the input"
        )
        self.command_manager.register(
            "new", self._handle_new_command, "Start a new conversation"
        )
        self.command_manager.register(
            "save", self._handle_save_command, "Save conversation"
        )
        self.command_manager.register(
            "load", self._handle_load_command, "Load most recent conversation"
        )
        self.command_manager.register(
            "image", self._handle_image_command, "Attach image from filesystem"
        )
        self.command_manager.register(
            "file", self._handle_file_command, "Attach file as context"
        )
        self.command_manager.register(
            "export", self._handle_export_command, "Export conversation to markdown"
        )
        self.command_manager.register("help", self._handle_help_command, "Show help")

    async def _handle_clear_command(self, _args: str) -> None:
        input_widget = self._w_input or self.query_one("#message_input", Input)
        input_widget.value = ""
        self.sub_title = "Input cleared."

    async def _handle_new_command(self, _args: str) -> None:
        await self.action_new_conversation()

    async def _handle_save_command(self, _args: str) -> None:
        await self.action_save_conversation()

    async def _handle_load_command(self, _args: str) -> None:
        await self.action_load_conversation()

    async def _handle_export_command(self, _args: str) -> None:
        await self.action_export_conversation()

    async def _handle_help_command(self, _args: str) -> None:
        await self.action_command_palette()

    async def _handle_image_command(self, args: str) -> None:
        raw_path = args.strip()
        if not raw_path:
            self.sub_title = "/image requires a file path"
            return
        self._attachments.add_image(raw_path)
        self.sub_title = f"Attached image: {raw_path}"

    async def _handle_file_command(self, args: str) -> None:
        raw_path = args.strip()
        if not raw_path:
            self.sub_title = "/file requires a file path"
            return
        self._attachments.add_file(raw_path)
        self.sub_title = f"Attached file: {raw_path}"

    def _load_last_prompt(self) -> None:
        """Load last prompt synchronously during init (file is small)."""
        try:
            if self._last_prompt_path.exists():
                self._last_prompt = self._last_prompt_path.read_text(
                    encoding="utf-8"
                ).strip()
        except Exception:
            self._last_prompt = ""

    async def _save_last_prompt(self, prompt: str) -> None:
        """Persist the last prompt asynchronously to avoid blocking the event loop."""
        try:
            self._last_prompt_path.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(
                self._last_prompt_path.write_text, prompt, encoding="utf-8"
            )
        except Exception:
            LOGGER.warning(
                "app.last_prompt.save_failed",
                extra={"event": "app.last_prompt.save_failed"},
            )

    @classmethod
    def _binding_specs_from_config(
        cls, config: dict[str, dict[str, Any]]
    ) -> list[Binding]:
        keybinds = config.get("keybinds", {})
        bindings: list[Binding] = []
        # Iterate over the canonical action descriptions; KEY_TO_ACTION was a
        # redundant identity map (every key mapped to itself) and has been removed.
        for action_name in cls.DEFAULT_ACTION_DESCRIPTIONS:
            binding_key = keybinds.get(action_name)
            if isinstance(binding_key, str) and binding_key.strip():
                bindings.append(
                    Binding(
                        key=binding_key.strip(),
                        action=action_name,
                        description=cls.DEFAULT_ACTION_DESCRIPTIONS[action_name],
                        show=True,
                    )
                )
        return bindings

    @staticmethod
    def _normalize_configured_models(raw_models: Any, default_model: str) -> list[str]:
        configured: list[str] = []
        if isinstance(raw_models, list):
            for item in raw_models:
                if not isinstance(item, str):
                    continue
                candidate = item.strip()
                if candidate and candidate not in configured:
                    configured.append(candidate)

        normalized_default = default_model.strip()
        if not configured:
            configured = [normalized_default]
        if normalized_default and normalized_default not in configured:
            configured.insert(0, normalized_default)
        return configured

    def _command_palette_key_display(self) -> str:
        for binding in self._binding_specs:
            if binding.action == "command_palette":
                return binding.key.upper()
        return "CTRL+P"

    def _set_idle_sub_title(self, prefix: str) -> None:
        palette_hint = f"🧭 Palette: {self._command_palette_key_display()}"
        self.sub_title = f"{prefix}  |  {palette_hint}"

    def _apply_terminal_window_identity(self) -> None:
        """Best-effort terminal identity setup for title and class."""
        self._emit_osc("0", self.window_title)
        self._emit_osc("2", self.window_title)
        if self.window_class.strip():
            self._emit_osc("1", self.window_class.strip())
        self._set_window_class_best_effort()

    @staticmethod
    def _emit_osc(code: str, value: str) -> None:
        if not value.strip():
            return
        print(f"\033]{code};{value}\007", end="", flush=True)

    def _discover_window_id(self) -> str | None:
        window_id = os.environ.get("WINDOWID", "").strip()
        if window_id:
            return window_id
        return None

    def _set_window_class_best_effort(self) -> None:
        class_name = self.window_class.strip()
        if not class_name:
            return
        xprop_bin = shutil.which("xprop")
        if xprop_bin is None:
            return

        window_id = self._discover_window_id()
        if window_id is None:
            return

        try:
            subprocess.run(
                [
                    xprop_bin,
                    "-id",
                    window_id,
                    "-f",
                    "WM_CLASS",
                    "8s",
                    "-set",
                    "WM_CLASS",
                    f"{class_name},{class_name}",
                ],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=0.2,
            )
        except Exception:
            return

    def _build_header(self) -> Header:
        """Return header with visual hamburger icon when supported."""
        try:
            return Header(name=self.window_title, icon="☰")
        except TypeError:
            return Header(name=self.window_title)

    def compose(self) -> ComposeResult:
        """Compose app widgets."""
        yield self._build_header()
        with Container(id="app-root"):
            yield ConversationView(id="conversation")
            yield InputBox()
            yield StatusBar(id="status_bar")
            yield ActivityBar(
                shortcut_hints="ctrl+p commands",
                id="activity_bar",
            )
        yield Footer()

    async def action_command_palette(self) -> None:
        """Open a lightweight command palette (help) listing available actions."""
        lines = ["Commands:", ""]
        for cmd, desc in self._slash_commands:
            lines.append(f"{cmd} - {desc}")
        lines.append("")
        lines.append("Keybind actions:")
        for binding in self._binding_specs:
            lines.append(
                f"{binding.key.upper()} - {binding.description} ({binding.action})"
            )
        await self.push_screen(InfoScreen("\n".join(lines)))

    async def action_toggle_conversation_picker(self) -> None:
        """Open conversation quick switcher."""
        if await self.state.get_state() != ConversationState.IDLE:
            self.sub_title = "Conversation picker is available only when idle."
            return
        if not self.persistence.enabled:
            self.sub_title = "Persistence is disabled in configuration."
            return
        items = self.persistence.list_conversations()
        if not items:
            self.sub_title = "No saved conversations found."
            return
        selected = await self.push_screen_wait(ConversationPickerScreen(items))
        if not selected:
            return
        await self._load_conversation_from_path(Path(selected))

    async def action_toggle_prompt_preset_picker(self) -> None:
        """Open prompt preset picker and apply selection."""
        if await self.state.get_state() != ConversationState.IDLE:
            self.sub_title = "Prompt picker is available only when idle."
            return
        if not self._prompt_presets:
            self.sub_title = "No prompt presets configured."
            return
        options = sorted(self._prompt_presets.keys())
        selected = await self.push_screen_wait(
            SimplePickerScreen("Prompt Presets", options)
        )
        if not selected:
            return
        self._active_prompt_preset = selected
        preset_value = self._prompt_presets.get(selected, "").strip()
        if preset_value:
            self.chat.system_prompt = preset_value
            self.chat.message_store = self.chat.message_store.__class__(
                system_prompt=preset_value,
                max_history_messages=self.chat.message_store.max_history_messages,
                max_context_tokens=self.chat.message_store.max_context_tokens,
            )
        self.sub_title = f"Prompt preset set: {selected}"

    async def action_toggle_theme_picker(self) -> None:
        """Open theme picker and apply selection."""
        available_themes = self.theme_manager.get_available_themes(self)
        current_theme = self.theme_manager.current_theme_name

        self.push_screen(
            ThemePickerScreen(available_themes, current_theme),
            callback=self._handle_theme_picker_result,
        )

    def _handle_theme_picker_result(self, selected: str | None) -> None:
        """Apply theme picker selection."""
        if not selected:
            return

        success = self.theme_manager.switch_theme(selected, self)
        if not success:
            self.sub_title = f"Failed to switch theme: {selected}"
            return

        self.sub_title = f"Theme switched to: {selected}"
        self._restyle_rendered_bubbles()

    async def _load_conversation_from_path(self, path: Path) -> None:
        try:
            payload = await self.conversation_manager.load_from_path(path)
        except Exception:
            self.sub_title = "Failed to load conversation."
            return
        await self._load_conversation_payload(payload)

    async def _prompt_conversation_name(self) -> str:
        try:
            value = await self.push_screen_wait(
                TextPromptScreen("Conversation name", placeholder="(optional)")
            )
        except Exception:
            return ""
        if value is None:
            return ""
        return str(value).strip()

    async def on_mount(self) -> None:
        """Apply theme and register runtime keybindings."""
        self.title = self.window_title
        self._set_idle_sub_title(f"Model: {self.chat.model}")
        LOGGER.info(
            "app.state.transition",
            extra={"event": "app.state.transition", "to_state": "IDLE"},
        )
        self._apply_theme()
        for binding in self._binding_specs:
            self.bind(
                binding.key,
                binding.action,
                description=binding.description,
                show=binding.show,
                key_display=binding.key_display,
            )

        # Populate widget cache once; avoids repeated DOM traversal on hot paths.
        self._w_input = self.query_one("#message_input", Input)
        self._w_send = self.query_one("#send_button", Button)
        self._w_file = self.query_one("#file_button", Button)
        self._w_activity = self.query_one("#activity_bar", ActivityBar)
        self._w_status = self.query_one("#status_bar", StatusBar)
        self._w_conversation = self.query_one(ConversationView)

        attach_button = self.query_one("#attach_button", Button)
        self._w_input.disabled = True
        self._w_send.disabled = True
        attach_button.disabled = not self.capabilities.vision_enabled
        self._w_file.disabled = False
        self._update_status_bar()

        # Initialize plugins and register their commands (tools are integrated via ToolRegistry build at startup)
        try:
            context = {
                "app": self,
                "config": self.config,
                "event_bus": self.event_bus,
                "command_manager": self.command_manager,
                "tool_registry": self._tool_registry,
            }
            await self.plugin_manager.initialize_all(context)
            plugin_commands = self.plugin_manager.get_all_commands()
            existing = {cmd for cmd, _ in self._slash_commands}
            for name, handler in (plugin_commands or {}).items():
                cmd_name = name.lstrip("/")

                async def _wrapped(args: str, _h=handler):
                    if inspect.iscoroutinefunction(_h):
                        await _h(args)
                    else:
                        await asyncio.to_thread(_h, args)

                self.command_manager.register(cmd_name, _wrapped, "Plugin command")
                display = "/" + cmd_name
                if display not in existing:
                    self._slash_commands.append((display, "Plugin command"))
                    existing.add(display)

            # Register plugin tools into the ToolRegistry
            if self._tool_registry is not None:
                try:
                    plugin_tools = self.plugin_manager.get_all_tools()
                except Exception:
                    plugin_tools = []
                runtime_opts = getattr(
                    self._tool_registry, "_runtime_options", ToolRuntimeOptions()
                )
                for tool in plugin_tools:
                    try:
                        name = getattr(tool, "id", "")
                        if not name:
                            continue
                        # Obtain parameters schema
                        try:
                            ollama_schema = tool.to_ollama_schema()
                            func_dict = ollama_schema.get("function", {})
                            schema = func_dict.get(
                                "parameters",
                                {
                                    "type": "object",
                                    "properties": {},
                                    "required": [],
                                    "additionalProperties": True,
                                },
                            )
                        except Exception:
                            try:
                                legacy = tool.schema()
                                schema = legacy.get(
                                    "parameters",
                                    {
                                        "type": "object",
                                        "properties": {},
                                        "required": [],
                                        "additionalProperties": True,
                                    },
                                )
                            except Exception:
                                schema = {
                                    "type": "object",
                                    "properties": {},
                                    "required": [],
                                    "additionalProperties": True,
                                }

                        def make_handler(t=tool):
                            def handler(args: dict[str, Any]) -> str:
                                async def _run() -> str:
                                    ctx = ToolContext(
                                        session_id="plugin",
                                        message_id=str(time.time_ns()),
                                        agent="ollama",
                                        abort=asyncio.Event(),
                                        extra={
                                            "project_dir": runtime_opts.workspace_root,
                                            "bypassCwdCheck": runtime_opts.allow_external_directories,
                                        },
                                    )
                                    result = await t.run(args, ctx)
                                    return str(result.output)

                                return _run_async_from_sync(_run())

                            return handler

                        spec = ToolSpec(
                            name=name,
                            description=getattr(tool, "description", name),
                            parameters_schema=schema,
                            handler=make_handler(),
                            safety_level="safe",
                            category="plugin",
                        )
                        self._tool_registry.register_spec(spec)
                    except Exception:
                        continue
        except Exception:
            pass

        # _slash_registry removed - using CommandManager instead

        palette_key = self._command_palette_key_display().lower()
        self._w_activity.set_shortcut_hints(f"{palette_key} commands")

        # The connection monitor starts only after _prepare_startup_model() completes
        # so that the startup check sets the initial connection state first, preventing the
        # monitor from immediately overwriting the startup result with a concurrent
        # check_connection() call.
        self._task_manager.add(
            asyncio.create_task(self._prepare_startup_model()), name="startup_model"
        )

    async def _prepare_startup_model(self) -> None:
        """Warm up model in background so UI stays responsive on launch."""
        try:
            await self._ensure_startup_model_ready()
        finally:
            if self._w_input:
                self._w_input.disabled = False
                self._w_input.focus()
            if self._w_send:
                self._w_send.disabled = False
            if self._w_file:
                self._w_file.disabled = False
            self._update_status_bar()
            # Start the connection monitor only after startup determines the initial
            # connection state, so the two tasks cannot race.
            await self.connection_manager.start_monitoring()

    async def _ensure_startup_model_ready(self) -> None:
        """Ensure configured model is available before interactive usage."""
        pull_on_start = bool(self.config["ollama"].get("pull_model_on_start", True))
        self.sub_title = f"Preparing model: {self.chat.model}"
        try:
            # Prefer new wrapper methods when available; fall back to legacy flag API
            # so test fakes that only implement ensure_model_ready(...) still work.
            if pull_on_start and hasattr(self.chat, "ensure_model_ready_pull"):
                await self.chat.ensure_model_ready_pull()  # type: ignore[func-returns-value]
            elif (not pull_on_start) and hasattr(
                self.chat, "ensure_model_ready_no_pull"
            ):
                await self.chat.ensure_model_ready_no_pull()  # type: ignore[func-returns-value]
            else:
                await self.chat.ensure_model_ready(pull_if_missing=pull_on_start)
            self.connection_manager._state = ConnectionState.ONLINE
            # Detect what this model actually supports and update effective caps.
            await self.capability_manager.detect_model_capabilities()
            self._set_idle_sub_title(f"Model ready: {self.chat.model}")
        except OllamaConnectionError:
            self.connection_manager._state = ConnectionState.OFFLINE
            self.sub_title = "Cannot reach Ollama. Start ollama serve."
        except OllamaModelNotFoundError:
            self.connection_manager._state = ConnectionState.ONLINE
            self.sub_title = (
                f"Model not available: {self.chat.model}. "
                "Enable pull_model_on_start or run ollama pull manually."
            )
        except OllamaStreamingError:
            self.connection_manager._state = ConnectionState.OFFLINE
            self.sub_title = "Failed while preparing model."
        except OllamaChatError:
            self.sub_title = "Model preparation failed."

    def _apply_theme(self) -> None:
        """Apply theme settings using ThemeManager and restyle mounted widgets."""
        # Initialize theme system
        self.theme_manager.initialize_theme(self)

        # Apply background for custom themes
        try:
            root = self.query_one("#app-root", Container)
            if not self.theme_manager.is_using_textual_theme:
                bg = self.theme_manager.get_background_color()
                root.styles.background = str(bg)
        except Exception:
            pass

        self._restyle_rendered_bubbles()

    def watch_theme(self, *_args: str) -> None:
        """Ensure all widgets react when a Textual theme changes."""
        self._apply_theme()

    def _setup_event_subscribers(self) -> None:
        """Subscribe to support bus events for file changes and watchers."""
        try:
            from .support.bus import bus as support_bus

            support_bus.subscribe("file.edited", self._on_support_file_event)
            support_bus.subscribe("file.watcher.updated", self._on_support_file_event)
        except Exception:
            pass

    def _on_support_file_event(self, event: str, payload: dict[str, Any]) -> None:
        """Handle file events from support bus (log-only)."""
        try:
            LOGGER.info(
                "app.file.event",
                extra={
                    "event": event,
                    "path": str(payload.get("file", "")),
                    "detail": payload,
                },
            )
            # Re-publish via app event bus asynchronously
            try:
                asyncio.create_task(self.event_bus.publish(event, payload))
            except Exception:
                pass
        except Exception:
            pass

    @property
    def show_timestamps(self) -> bool:
        return bool(self.config["ui"]["show_timestamps"])

    def _timestamp(self) -> str:
        """Generate timestamp for messages (delegates to MessageRenderer)."""
        if not self.show_timestamps:
            return ""
        return self.message_renderer.generate_timestamp()

    @staticmethod
    def _apply_custom_theme(
        bubble: MessageBubble, role: str, ui_cfg: dict[str, Any]
    ) -> None:
        """Apply user-configured colours and border to a message bubble."""
        if role == "user":
            bubble.styles.background = str(ui_cfg["user_message_color"])
        else:
            bubble.styles.background = str(ui_cfg["assistant_message_color"])
        bubble.styles.border = ("round", str(ui_cfg["border_color"]))

    def _style_bubble(self, bubble: MessageBubble, role: str) -> None:
        """Style a message bubble (delegates to MessageRenderer)."""
        bubble.styles.align_horizontal = "right" if role == "user" else "left"
        self.message_renderer.style_bubble(bubble, role)

    def _restyle_rendered_bubbles(self) -> None:
        """Restyle all bubbles (delegates to MessageRenderer)."""
        try:
            conversation = self._w_conversation or self.query_one(ConversationView)
        except Exception:
            return
        self.message_renderer.restyle_all_bubbles(conversation)

    def _using_theme_palette(self) -> bool:
        return bool(getattr(self, "theme", ""))

    def _update_status_bar(self) -> None:
        # Use non_system_count (no list copy) when the real MessageStore is available;
        # fall back to iterating the messages property for test fakes that lack it.
        ms = getattr(self.chat, "message_store", None)
        if ms is not None and hasattr(ms, "non_system_count"):
            message_count = ms.non_system_count
        else:
            message_count = sum(
                1
                for m in getattr(self.chat, "messages", [])
                if m.get("role") != "system"
            )
        status_widget = getattr(self, "_w_status", None) or self.query_one(
            "#status_bar", StatusBar
        )
        status_widget.set_status(
            connection_state=self.connection_manager.state.value,
            model=self.chat.model,
            message_count=message_count,
            estimated_tokens=self.chat.estimated_context_tokens,
            effective_caps=self.capability_manager.effective_capabilities,
        )

    async def _open_configured_model_picker(self) -> None:
        if await self.state.get_state() != ConversationState.IDLE:
            self.sub_title = "Model switch is available only when idle."
            return
        configured_models = list(self._configured_models)
        if not configured_models:
            self.sub_title = "No configured models found in config."
            return
        self.push_screen(
            ModelPickerScreen(configured_models, self.chat.model),
            callback=self._on_model_picker_dismissed,
        )

    def _on_model_picker_dismissed(self, selected_model: str | None) -> None:
        if selected_model is None:
            return
        self._task_manager.add(
            asyncio.create_task(self._activate_selected_model(selected_model))
        )

    async def _activate_selected_model(self, model_name: str) -> None:
        if await self.state.get_state() != ConversationState.IDLE:
            self.sub_title = "Model switch is available only when idle."
            return
        if model_name not in self._configured_models:
            self.sub_title = f"Model is not configured: {model_name}"
            return

        previous_model = self.chat.model
        self.chat.set_model(model_name)
        self.sub_title = f"Switching model: {model_name}"
        try:
            if hasattr(self.chat, "ensure_model_ready_no_pull"):
                await self.chat.ensure_model_ready_no_pull()  # type: ignore[func-returns-value]
            else:
                await self.chat.ensure_model_ready(pull_if_missing=False)
            self.connection_manager._state = ConnectionState.ONLINE

            # Fetch this model's actual capabilities and recompute effective flags.
            await self.capability_manager.detect_model_capabilities(model_name)

            # Build a subtitle reporting which capabilities this model lacks.
            unsupported = self.capability_manager.get_unsupported_features()
            msg = f"Active model: {model_name}"
            if unsupported and self.capability_manager.model_capabilities.known:
                msg += f"  |  Not supported: {', '.join(unsupported)}"
            self._set_idle_sub_title(msg)
        except OllamaChatError as exc:  # noqa: BLE001
            self.chat.set_model(previous_model)
            LOGGER.warning(
                "app.model.switch.failed",
                extra={
                    "event": "app.model.switch.failed",
                    "error_type": type(exc).__name__,
                    "model": model_name,
                },
            )
            if isinstance(exc, OllamaConnectionError):
                self.connection_manager._state = ConnectionState.OFFLINE
                self.sub_title = "Unable to switch model while offline."
            elif isinstance(exc, OllamaModelNotFoundError):
                self._set_idle_sub_title(
                    f"Configured model unavailable in Ollama: {model_name}"
                )
            elif isinstance(exc, OllamaStreamingError):
                self.sub_title = "Failed while validating selected model."
            else:
                self.sub_title = "Model switch failed."
        finally:
            self._update_status_bar()

    async def _on_connection_state_changed(self, old_state, new_state) -> None:
        """Handle connection state changes from ConnectionManager."""
        LOGGER.info(
            "app.connection.state",
            extra={
                "event": "app.connection.state",
                "connection_state": new_state.value,
            },
        )
        if await self.state.get_state() == ConversationState.IDLE:
            self._set_idle_sub_title(f"Connection: {new_state}")
        self._update_status_bar()

    async def _add_message(
        self, content: str, role: str, timestamp: str = ""
    ) -> MessageBubble:
        """Add a message bubble (delegates to MessageRenderer)."""
        conversation = self._w_conversation or self.query_one(ConversationView)
        bubble = await self.message_renderer.add_message(
            conversation, content, role, timestamp
        )
        # Apply horizontal alignment (not in MessageRenderer)
        bubble.styles.align_horizontal = "right" if role == "user" else "left"
        return bubble

    async def on_status_bar_model_picker_requested(
        self, _message: StatusBar.ModelPickerRequested
    ) -> None:
        """Open configured model picker from StatusBar model segment click."""
        await self._open_configured_model_picker()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle send button clicks."""
        if event.button.id == "send_button":
            await self.send_user_message()

    async def _open_attachment_dialog(self, mode: str) -> None:
        """Open attachment dialog (delegates to AttachmentManager)."""
        await self.attachment_manager.open_dialog(
            mode,
            open_native_dialog=_open_native_file_dialog,
            open_modal_dialog=lambda callback: self.push_screen(
                ImageAttachScreen(), callback=callback
            ),
        )

    async def on_input_box_attach_requested(
        self, _message: InputBox.AttachRequested
    ) -> None:
        """Open native image picker when attach button is clicked."""
        if not self.capability_manager.effective_capabilities.vision_enabled:
            self.sub_title = "Vision is not supported by this model."
            return
        await self._open_attachment_dialog("image")

    async def on_input_box_file_attach_requested(
        self, _message: InputBox.FileAttachRequested
    ) -> None:
        """Open native file picker when file button is clicked."""
        await self._open_attachment_dialog("file")

    # _on_image_attach_dismissed() moved to AttachmentManager
    # _on_file_attach_dismissed() moved to AttachmentManager
    # _is_image_path() moved to AttachmentManager

    @staticmethod
    def _extract_paths_from_paste(text: str) -> list[str]:
        """Extract file paths from pasted text (common drag/drop behavior)."""
        candidates: list[str] = []
        for token in text.strip().split():
            cleaned = token.strip().strip("'\"")
            if cleaned.startswith("file://"):
                cleaned = cleaned[len("file://") :]
            if cleaned:
                candidates.append(cleaned)
        return candidates

    def on_paste(self, event: Paste) -> None:
        """Handle drag/drop style paste events to attach files/images."""
        if not event.text:
            return
        paths = self._extract_paths_from_paste(event.text)
        if not paths:
            return
        added_images = 0
        added_files = 0
        for path in paths:
            expanded = os.path.expanduser(path)
            if not os.path.isfile(expanded):
                continue
            if (
                AttachmentManager.is_image_path(expanded)
                and self.capability_manager.effective_capabilities.vision_enabled
            ):
                self._attachments.add_image(expanded)
                added_images += 1
            else:
                self._attachments.add_file(expanded)
                added_files += 1
        if added_images or added_files:
            self.sub_title = (
                f"Attached {added_images} image(s), {added_files} file(s) via drop"
            )
            event.stop()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submit events."""
        if event.input.id == "message_input":
            await self.send_user_message()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Toggle slash menu visibility based on input prefix."""
        if event.input.id != "message_input":
            return
        value = event.value
        if value.startswith("/"):
            self._show_slash_menu(prefix=value)
        else:
            self._hide_slash_menu()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id != "slash_menu":
            return
        option_text = str(event.option.prompt)
        command = option_text.split(" ", 1)[0]
        input_widget = self._w_input or self.query_one("#message_input", Input)
        input_widget.value = f"{command} "
        input_widget.cursor_position = len(input_widget.value)
        self._hide_slash_menu()
        input_widget.focus()
        event.stop()

    def on_key(self, event: Key) -> None:
        """Handle Up-arrow recall and slash quick-open."""
        if event.key == "up":
            try:
                input_widget = self._w_input or self.query_one("#message_input", Input)
            except Exception:
                return
            if input_widget.has_focus and not input_widget.value and self._last_prompt:
                input_widget.value = self._last_prompt
                input_widget.cursor_position = len(input_widget.value)
                self.sub_title = "Restored last prompt."
                event.stop()

    def _show_slash_menu(self, prefix: str) -> None:
        """Show slash command menu (delegates to CommandManager)."""
        try:
            menu = self.query_one("#slash_menu", OptionList)
        except Exception:
            return
        self.command_manager.show_slash_menu(menu, prefix)
        if menu.options:
            menu.remove_class("hidden")
        else:
            menu.add_class("hidden")

    def _hide_slash_menu(self) -> None:
        """Hide slash command menu (delegates to CommandManager)."""
        try:
            menu = self.query_one("#slash_menu", OptionList)
        except Exception:
            return
        self.command_manager.hide_slash_menu(menu)
        menu.add_class("hidden")

    async def action_send_message(self) -> None:
        """Action invoked by keybinding for sending a message."""
        await self.send_user_message()

    async def _transition_state(self, new_state: ConversationState) -> None:
        """Transition to new_state atomically (single lock acquisition)."""
        await self.state.transition_to(new_state)
        LOGGER.info(
            "app.state.transition",
            extra={
                "event": "app.state.transition",
                "to_state": new_state.value,
            },
        )

    # _animate_response_placeholder() moved to StreamManager
    # _stop_response_indicator_task() moved to StreamManager

    async def _stream_assistant_response(
        self,
        user_text: str,
        assistant_bubble: MessageBubble,
        images: list[str | bytes] | None = None,
    ) -> None:
        """Stream assistant response (delegates to StreamManager)."""

        def _scroll() -> None:
            conv = self._w_conversation or self.query_one(ConversationView)
            conv.scroll_end(animate=False)

        opts = ChatSendOptions(
            images=images or None,
            tool_registry=(
                self._tool_registry
                if self.capability_manager.effective_capabilities.tools_enabled
                else None
            ),
            think=self.capability_manager.effective_capabilities.think,
            max_tool_iterations=self.capability_manager.effective_capabilities.max_tool_iterations,
        )

        await self.stream_manager.stream_response(
            user_text,
            assistant_bubble,
            _scroll,
            opts,
        )

    # _handle_stream_error() moved to StreamManager

    async def action_interrupt_stream(self) -> None:
        """Cancel an in-flight assistant response (delegates to StreamManager)."""
        interrupted = await self.stream_manager.interrupt_stream(self.chat.model)
        if interrupted:
            self._update_status_bar()
            await self._transition_state(ConversationState.IDLE)
        else:
            self.sub_title = "No response to interrupt."

    async def action_new_conversation(self) -> None:
        """Clear UI and in-memory conversation history."""
        active_stream = self._task_manager.get("active_stream")
        if (
            await self.state.get_state() == ConversationState.STREAMING
            and active_stream is not None
        ):
            await self._transition_state(ConversationState.CANCELLING)
            LOGGER.info(
                "chat.request.cancelling", extra={"event": "chat.request.cancelling"}
            )
            await self._task_manager.cancel("active_stream")
            LOGGER.info(
                "chat.request.cancelled", extra={"event": "chat.request.cancelled"}
            )

        self.chat.clear_history()
        self._attachments.clear()
        await self._clear_conversation_view()
        self._search.reset()
        await self._transition_state(ConversationState.IDLE)
        self._set_idle_sub_title(f"Model: {self.chat.model}")
        self._update_status_bar()

    async def _clear_conversation_view(self) -> None:
        """Remove all rendered conversation bubbles (delegates to MessageRenderer)."""
        conversation = self._w_conversation or self.query_one(ConversationView)
        await self.message_renderer.clear_conversation(conversation)

    async def _render_messages_from_history(
        self, messages: list[dict[str, Any]]
    ) -> None:
        """Render persisted messages (delegates to MessageRenderer)."""
        conversation = self._w_conversation or self.query_one(ConversationView)
        await self.message_renderer.render_history(conversation, messages)

    def _auto_save_on_exit(self) -> None:
        """Persist conversation on exit when auto_save is enabled."""
        if not self.persistence.enabled:
            return
        self.conversation_manager.auto_save_on_exit()

    async def on_unmount(self) -> None:
        """Cancel and await all background tasks during shutdown."""
        self._auto_save_on_exit()
        await self._transition_state(ConversationState.CANCELLING)
        try:
            self.plugin_manager.shutdown_all()
        except Exception:
            pass
        await self.connection_manager.stop_monitoring()
        await self._task_manager.cancel_all()
        await self._transition_state(ConversationState.IDLE)

    async def action_quit(self) -> None:
        """Exit the app."""
        self.exit()

    def action_scroll_up(self) -> None:
        """Scroll conversation up."""
        conversation = self._w_conversation or self.query_one(ConversationView)
        conversation.scroll_relative(y=-10, animate=False)

    def action_scroll_down(self) -> None:
        """Scroll conversation down."""
        conversation = self._w_conversation or self.query_one(ConversationView)
        conversation.scroll_relative(y=10, animate=False)

    async def action_toggle_model_picker(self) -> None:
        """Open configured model picker while in IDLE state."""
        await self._open_configured_model_picker()

    async def action_save_conversation(self) -> None:
        """Persist the current conversation to disk."""
        if await self.state.get_state() != ConversationState.IDLE:
            self.sub_title = "Save is available only when idle."
            return
        if not self.persistence.enabled:
            self.sub_title = "Persistence is disabled in configuration."
            return
        try:
            name = await self._prompt_conversation_name()
            path = await self.conversation_manager.save_snapshot(name)
            self.sub_title = f"Conversation saved: {path}"
            try:
                await self.event_bus.publish(
                    "conversation.saved",
                    {
                        "path": str(path),
                        "timestamp": datetime.now().isoformat(),
                    },
                )
            except Exception:
                pass
        except Exception:
            self.sub_title = "Failed to save conversation."

    async def action_load_conversation(self) -> None:
        """Load the most recently saved conversation."""
        if await self.state.get_state() != ConversationState.IDLE:
            self.sub_title = "Load is available only when idle."
            return
        if not self.persistence.enabled:
            self.sub_title = "Persistence is disabled in configuration."
            return
        try:
            payload = await self.conversation_manager.load_latest()
        except Exception:
            self.sub_title = "Failed to read saved conversations."
            return
        if payload is None:
            self.sub_title = "No saved conversation found."
            return
        await self._load_conversation_payload(payload)
        try:
            await self.event_bus.publish(
                "conversation.loaded",
                {"timestamp": datetime.now().isoformat()},
            )
        except Exception:
            pass

    async def action_export_conversation(self) -> None:
        """Export current conversation to markdown."""
        if await self.state.get_state() != ConversationState.IDLE:
            self.sub_title = "Export is available only when idle."
            return
        if not self.persistence.enabled:
            self.sub_title = "Persistence is disabled in configuration."
            return
        try:
            path = self.persistence.export_markdown(self.chat.messages, self.chat.model)
            self.sub_title = f"Exported markdown: {path}"
        except Exception:
            self.sub_title = "Failed to export conversation."

    def _jump_to_search_result(self, message_index: int) -> None:
        conversation = self._w_conversation or self.query_one(ConversationView)
        non_system_index = -1
        for index, message in enumerate(self.chat.messages):
            if message.get("role") == "system":
                continue
            non_system_index += 1
            if index == message_index:
                break
        bubbles = [
            child for child in conversation.children if isinstance(child, MessageBubble)
        ]
        if 0 <= non_system_index < len(bubbles):
            target = bubbles[non_system_index]
            if hasattr(target, "scroll_visible"):
                target.scroll_visible(animate=False)
            conversation.scroll_end(animate=False)

    async def action_search_messages(self) -> None:
        """Search messages using input box text and cycle through results."""
        input_widget = self._w_input or self.query_one("#message_input", Input)
        query = input_widget.value.strip().lower()

        if not query and self._search.has_results():
            current = self._search.advance()
            self._jump_to_search_result(current)
            self.sub_title = f"Search {self._search.position + 1}/{len(self._search.results)}: {self._search.query}"
            return
        if not query:
            self.sub_title = "Type search text in the input box, then press search."
            return

        self._search.query = query
        self._search.results = [
            index
            for index, message in enumerate(self.chat.messages)
            if message.get("role") != "system"
            and query in str(message.get("content", "")).lower()
        ]
        self._search.position = 0
        if not self._search.has_results():
            self.sub_title = f"No matches for '{query}'."
            return
        self._jump_to_search_result(self._search.results[self._search.position])
        self.sub_title = (
            f"Search {self._search.position + 1}/{len(self._search.results)}: {query}"
        )

    async def action_copy_last_message(self) -> None:
        """Copy the latest assistant reply to clipboard when available."""
        for message in reversed(self.chat.messages):
            if message.get("role") != "assistant":
                continue
            content = str(message.get("content", "")).strip()
            if not content:
                continue
            if hasattr(self, "copy_to_clipboard"):
                self.copy_to_clipboard(content)  # type: ignore[attr-defined]
                self.sub_title = "Copied latest assistant message."
            else:
                input_widget = self._w_input or self.query_one("#message_input", Input)
                input_widget.value = content
                self.sub_title = "Clipboard unavailable. Message placed in input box."
            return
        self.sub_title = "No assistant message available to copy."

    async def send_user_message(self) -> None:
        """Collect input text and stream the assistant response into the UI."""
        input_widget = self._w_input or self.query_one("#message_input", Input)
        send_button = self._w_send or self.query_one("#send_button", Button)
        file_button = self._w_file or self.query_one("#file_button", Button)
        raw_text = input_widget.value.strip()

        # Intercept slash commands before sending to LLM.
        if raw_text.startswith("/"):
            try:
                handled = await self.command_manager.execute(raw_text)
                try:
                    await self.event_bus.publish(
                        "command.executed",
                        {
                            "command": raw_text.split()[0],
                            "success": bool(handled),
                            "timestamp": datetime.now().isoformat(),
                        },
                    )
                except Exception:
                    pass
            except Exception:
                self.sub_title = "Command failed."
                return
            if handled:
                return

        directives = parse_inline_directives(
            raw_text, self.capability_manager.effective_capabilities
        )
        user_text = directives.cleaned_text
        inline_images = directives.image_paths
        inline_files = directives.file_paths

        # Combine inline images and images attached via the attach button.
        all_images: list[str] = inline_images + self._attachments.images
        all_files: list[str] = inline_files + list(self._attachments.files)

        if not user_text and not all_images and not all_files:
            self.sub_title = "Cannot send an empty message."
            return

        # Validate attachments (delegates to AttachmentManager)
        valid_images_str, valid_files, errors = (
            self.attachment_manager.validate_attachments_batch(all_images, all_files)
        )
        # Convert to list[str | bytes] for API
        valid_images: list[str | bytes] = valid_images_str

        # Log validation errors
        if errors:
            for error in errors:
                LOGGER.warning(
                    "app.attachment.validation_failed",
                    extra={"event": "app.attachment.validation_failed", "error": error},
                )
            # Show first error to user
            self.sub_title = errors[0]

        # Atomic CAS: only proceed when IDLE → STREAMING succeeds.
        # This replaces a separate can_send_message() check, eliminating
        # the TOCTOU gap between the two lock acquisitions.
        assistant_bubble: MessageBubble | None = None
        transitioned = await self.state.transition_if(
            ConversationState.IDLE, ConversationState.STREAMING
        )
        if not transitioned:
            self.sub_title = "Busy. Wait for current request to finish."
            return
        LOGGER.info(
            "app.state.transition",
            extra={
                "event": "app.state.transition",
                "from_state": "IDLE",
                "to_state": "STREAMING",
            },
        )
        input_widget.disabled = True
        send_button.disabled = True
        file_button.disabled = True
        try:
            activity = self._w_activity or self.query_one("#activity_bar", ActivityBar)
            activity.start_activity()
        except Exception:
            pass
        # Clear pending images and files now that we've consumed them.
        self._attachments.clear()
        try:
            display_text = (
                raw_text
                if raw_text
                else f"[{len(valid_images)} image(s), {len(valid_files)} file(s)]"
            )
            self.sub_title = "Sending message..."
            await self._add_message(
                content=display_text, role="user", timestamp=self._timestamp()
            )
            input_widget.value = ""
            assistant_bubble = await self._add_message(
                content="", role="assistant", timestamp=self._timestamp()
            )

            # Build file context to append to the user prompt for the API call.
            file_context_parts: list[str] = []
            for path in valid_files:
                snippet = ""
                try:
                    snippet = Path(path).read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    snippet = "<unreadable file>"
                max_chars = 4000
                if len(snippet) > max_chars:
                    snippet = snippet[:max_chars] + "\n... [truncated]"
                file_context_parts.append(
                    f"[File: {os.path.basename(path)}]\n{snippet}"
                )

            final_user_text = user_text
            if file_context_parts:
                file_context = "\n\n".join(file_context_parts)
                final_user_text = (
                    f"{user_text}\n\n{file_context}" if user_text else file_context
                )
            self._last_prompt = final_user_text
            await self._save_last_prompt(final_user_text)
            self._hide_slash_menu()

            stream_task = asyncio.create_task(
                self._stream_assistant_response(
                    final_user_text,
                    assistant_bubble,
                    images=valid_images if valid_images else None,
                )
            )
            self._task_manager.add(stream_task, name="active_stream")
            try:
                await stream_task
            finally:
                self._task_manager.discard("active_stream")
            self._set_idle_sub_title("Ready")
        except asyncio.CancelledError:
            # chat.py already logs "chat.request.cancelled"; no duplicate log here.
            self.sub_title = "Request cancelled."
            return
        except OllamaChatError as exc:  # noqa: BLE001
            msg_tpl, subtitle = _STREAM_ERROR_MESSAGES.get(
                type(exc),
                _STREAM_ERROR_MESSAGES[OllamaChatError],
            )
            await self.stream_manager.handle_stream_error(
                assistant_bubble,
                msg_tpl.format(exc=exc),
                subtitle,
                add_message_callback=self._add_message,
                timestamp_callback=self._timestamp,
            )
        finally:
            try:
                activity = self._w_activity or self.query_one(
                    "#activity_bar", ActivityBar
                )
                activity.stop_activity()
            except Exception:
                pass
            input_widget.disabled = False
            send_button.disabled = False
            file_button.disabled = False
            input_widget.focus()
            await self._transition_state(ConversationState.IDLE)
            self._update_status_bar()

    # _build_slash_registry() deleted - using CommandManager instead
    # register_slash_command() deleted - using CommandManager instead
    # _dispatch_slash_command() deleted - using CommandManager instead
