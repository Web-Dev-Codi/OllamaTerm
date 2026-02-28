"""Configuration loading and validation for the Ollama chat TUI."""

from __future__ import annotations

from copy import deepcopy
import logging
import os
from pathlib import Path
import re
import tomllib  # stdlib since Python 3.11 (project requires >=3.11)
from typing import Any
from urllib.parse import urlparse

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from .exceptions import ConfigValidationError

LOGGER = logging.getLogger(__name__)

CONFIG_DIR = Path.home() / ".config" / "ollamaterm"
CONFIG_PATH = CONFIG_DIR / "config.toml"

LEGACY_CONFIG_DIR = Path.home() / ".config" / "ollama-chat"
LEGACY_CONFIG_PATH = LEGACY_CONFIG_DIR / "config.toml"

HEX_COLOR_PATTERN = re.compile(r"^#(?:[0-9a-fA-F]{3}){1,2}$")
VALID_LOG_LEVELS = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}


class AppConfig(BaseModel):
    """Application metadata and terminal integration options."""

    model_config = ConfigDict(populate_by_name=True)
    title: str = "OllamaTerm"
    window_class: str = Field(default="ollamaterm", alias="class")
    connection_check_interval_seconds: int = Field(default=15, ge=1, le=3600)

    @field_validator("title", "window_class", mode="before")
    @classmethod
    def _validate_non_empty_string(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("Expected a string value.")
        normalized = value.strip()
        if not normalized:
            raise ValueError("String value must not be empty.")
        return normalized


class OllamaConfig(BaseModel):
    """Ollama endpoint and model settings."""

    host: str = "http://localhost:11434"
    model: str = "llama3.2"
    models: list[str] = Field(default_factory=list)
    timeout: int = Field(default=120, ge=1, le=3600)
    system_prompt: str = "You are a helpful assistant."
    prompt_presets: dict[str, str] = Field(default_factory=dict)
    active_prompt_preset: str = ""
    max_history_messages: int = Field(default=200, ge=1, le=100_000)
    max_context_tokens: int = Field(default=4096, ge=128, le=1_000_000)
    pull_model_on_start: bool = True
    api_key: str = ""

    @field_validator("host", "model", mode="before")
    @classmethod
    def _validate_required_string(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("Expected a string value.")
        normalized = value.strip()
        if not normalized:
            raise ValueError("String value must not be empty.")
        return normalized

    @field_validator("system_prompt", mode="before")
    @classmethod
    def _normalize_prompt(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("Expected a string value.")
        return value.strip()

    @field_validator("api_key", mode="before")
    @classmethod
    def _normalize_api_key(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("api_key must be a string.")
        return value.strip()

    @field_validator("active_prompt_preset", mode="before")
    @classmethod
    def _normalize_active_preset(cls, value: Any) -> str:
        if value is None:
            return ""
        if not isinstance(value, str):
            raise ValueError("active_prompt_preset must be a string.")
        return value.strip()

    @field_validator("prompt_presets", mode="before")
    @classmethod
    def _validate_prompt_presets(cls, value: Any) -> dict[str, str]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("prompt_presets must be a table/dict of name -> prompt.")
        presets: dict[str, str] = {}
        for k, v in value.items():
            if not isinstance(k, str) or not k.strip():
                raise ValueError("prompt_presets keys must be non-empty strings.")
            if not isinstance(v, str):
                raise ValueError("prompt_presets values must be strings.")
            presets[k.strip()] = v.strip()
        return presets

    @field_validator("models", mode="before")
    @classmethod
    def _validate_models(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("models must be a list of model names.")

        normalized: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError("Each model name in models must be a string.")
            candidate = item.strip()
            if not candidate:
                raise ValueError("Model names in models must not be empty.")
            if candidate not in normalized:
                normalized.append(candidate)
        return normalized

    @model_validator(mode="after")
    def _normalize_model_list(self) -> OllamaConfig:
        ordered_models = list(self.models)
        if not ordered_models:
            ordered_models = [self.model]
        if self.model not in ordered_models:
            ordered_models.insert(0, self.model)
        deduped: list[str] = []
        for model_name in ordered_models:
            if model_name not in deduped:
                deduped.append(model_name)
        self.models = deduped

        # Apply prompt preset selection if configured.
        preset_name = (self.active_prompt_preset or "").strip()
        if preset_name:
            preset_value = self.prompt_presets.get(preset_name)
            if isinstance(preset_value, str) and preset_value.strip():
                self.system_prompt = preset_value.strip()
        return self


class UIConfig(BaseModel):
    """Visual settings for Textual rendering."""

    font_size: int = Field(default=14, ge=8, le=72)
    background_color: str = "#1a1b26"
    user_message_color: str = "#7aa2f7"
    assistant_message_color: str = "#9ece6a"
    border_color: str = "#565f89"
    show_timestamps: bool = True
    stream_chunk_size: int = Field(default=8, ge=1, le=1024)

    @field_validator(
        "background_color",
        "user_message_color",
        "assistant_message_color",
        "border_color",
        mode="before",
    )
    @classmethod
    def _validate_hex_color(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("Expected a string value.")
        normalized = value.strip()
        if not HEX_COLOR_PATTERN.match(normalized):
            raise ValueError("Color must use #RGB or #RRGGBB format.")
        return normalized


class KeybindsConfig(BaseModel):
    """Keyboard action mapping."""

    send_message: str = "ctrl+enter"
    new_conversation: str = "ctrl+n"
    quit: str = "ctrl+q"
    scroll_up: str = "ctrl+k"
    scroll_down: str = "ctrl+j"
    command_palette: str = "ctrl+p"
    toggle_model_picker: str = "ctrl+m"
    toggle_theme_picker: str = "ctrl+t"
    save_conversation: str = "ctrl+s"
    load_conversation: str = "ctrl+l"
    export_conversation: str = "ctrl+e"
    search_messages: str = "ctrl+f"
    copy_last_message: str = "ctrl+y"
    interrupt_stream: str = "escape"

    @field_validator("*", mode="before")
    @classmethod
    def _validate_keybind(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("Keybind must be a string.")
        normalized = value.strip()
        if not normalized:
            raise ValueError("Keybind must not be empty.")
        return normalized


class SecurityConfig(BaseModel):
    """Security policy for remote host access."""

    allow_remote_hosts: bool = False
    allowed_hosts: list[str] = ["localhost", "127.0.0.1", "::1"]

    @field_validator("allowed_hosts", mode="before")
    @classmethod
    def _validate_allowed_hosts(cls, value: Any) -> list[str]:
        if not isinstance(value, list):
            raise ValueError("allowed_hosts must be a list.")
        normalized_hosts = [
            item.strip().lower()
            for item in value
            if isinstance(item, str) and item.strip()
        ]
        if not normalized_hosts:
            raise ValueError("allowed_hosts must contain at least one host.")
        return normalized_hosts


class LoggingConfig(BaseModel):
    """Logging behavior and output destinations."""

    level: str = "INFO"
    structured: bool = True
    log_to_file: bool = False
    log_file_path: str = "~/.local/state/ollamaterm/app.log"

    @field_validator("level", mode="before")
    @classmethod
    def _validate_level(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("Logging level must be a string.")
        normalized = value.strip().upper()
        if normalized not in VALID_LOG_LEVELS:
            raise ValueError(f"Unsupported log level {normalized!r}.")
        return normalized

    @field_validator("log_file_path", mode="before")
    @classmethod
    def _validate_log_file_path(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("log_file_path must be a string.")
        normalized = value.strip()
        if not normalized:
            raise ValueError("log_file_path must not be empty.")
        return normalized


class PersistenceConfig(BaseModel):
    """Conversation persistence settings (Tier 2 feature flags included)."""

    enabled: bool = False
    auto_save: bool = True
    directory: str = "~/.local/state/ollamaterm/conversations"
    metadata_path: str = "~/.local/state/ollamaterm/conversations/index.json"

    @field_validator("directory", "metadata_path", mode="before")
    @classmethod
    def _validate_path_string(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("Path value must be a string.")
        normalized = value.strip()
        if not normalized:
            raise ValueError("Path value must not be empty.")
        return normalized


class ToolsConfig(BaseModel):
    """Runtime policy for schema-based coding tools."""

    enabled: bool = True
    workspace_root: str = "."
    allow_external_directories: bool = False
    command_timeout_seconds: int = Field(default=30, ge=1, le=600)
    max_output_lines: int = Field(default=200, ge=1, le=10_000)
    max_output_bytes: int = Field(default=50_000, ge=256, le=5_000_000)
    max_read_bytes: int = Field(default=200_000, ge=256, le=20_000_000)
    max_search_results: int = Field(default=200, ge=1, le=10_000)
    default_external_directories: list[str] = Field(default_factory=list)

    @field_validator("workspace_root", mode="before")
    @classmethod
    def _validate_workspace_root(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("workspace_root must be a string.")
        normalized = value.strip()
        if not normalized:
            raise ValueError("workspace_root must not be empty.")
        return normalized

    @field_validator("default_external_directories", mode="before")
    @classmethod
    def _validate_external_directories(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("default_external_directories must be a list.")
        normalized: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError(
                    "default_external_directories entries must be strings."
                )
            candidate = item.strip()
            if candidate:
                normalized.append(candidate)
        return normalized


class CapabilitiesConfig(BaseModel):
    """User-facing feature preferences for thinking, tools, web search, and vision.

    Model capability detection (thinking / tools / vision support) is handled
    automatically via Ollama's ``/api/show`` endpoint.  The flags that used to
    live here (``think``, ``tools_enabled``, ``vision_enabled``) have been
    removed: auto-detection is now the sole authority for those three.

    The remaining fields are either UI preferences (``show_thinking``), app-level
    configuration (``web_search_*``), or behavioural limits
    (``max_tool_iterations``) that have no model-side equivalent.
    """

    show_thinking: bool = True
    web_search_enabled: bool = False
    web_search_api_key: str = ""
    max_tool_iterations: int = Field(default=10, ge=1, le=100)

    @field_validator("web_search_api_key", mode="before")
    @classmethod
    def _normalize_api_key(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("web_search_api_key must be a string.")
        return value.strip()


class Config(BaseModel):
    """Root configuration model for all sections."""

    model_config = ConfigDict(populate_by_name=True)
    app: AppConfig = AppConfig()
    ollama: OllamaConfig = OllamaConfig()
    ui: UIConfig = UIConfig()
    keybinds: KeybindsConfig = KeybindsConfig()
    security: SecurityConfig = SecurityConfig()
    logging: LoggingConfig = LoggingConfig()
    persistence: PersistenceConfig = PersistenceConfig()
    tools: ToolsConfig = ToolsConfig()
    capabilities: CapabilitiesConfig = CapabilitiesConfig()

    @model_validator(mode="after")
    def _validate_security_policy(self) -> Config:
        parsed = urlparse(self.ollama.host)
        scheme = parsed.scheme.lower()
        hostname = (parsed.hostname or "").strip().lower()

        if scheme not in {"http", "https"}:
            raise ValueError("ollama.host must use http or https scheme.")
        if not hostname:
            raise ValueError("ollama.host must include a hostname.")
        if not self.security.allow_remote_hosts and hostname not in set(
            self.security.allowed_hosts
        ):
            raise ValueError(
                "ollama.host is not in security.allowed_hosts while allow_remote_hosts is false."
            )
        return self


def _build_default_config() -> dict[str, dict[str, Any]]:
    """Build default config with an empty models list for clean merging."""
    data = Config().model_dump(by_alias=True)
    # Keep models empty in the base default so that user-specified `model`
    # values are not contaminated by the normalised default model list when
    # deep-merging a partial TOML that only sets `model` without `models`.
    data["ollama"]["models"] = []
    return data


DEFAULT_CONFIG: dict[str, dict[str, Any]] = _build_default_config()


def ensure_config_dir(config_dir: Path | None = None) -> Path:
    """Ensure that the config directory exists and return its path."""
    directory = config_dir or CONFIG_DIR
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        LOGGER.warning("Unable to create config directory %s: %s", directory, exc)
    return directory


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override values onto base values."""
    merged: dict[str, Any] = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _enforce_private_permissions(path: Path) -> None:
    """Best-effort enforcement of private file permissions on POSIX systems."""
    if os.name != "posix" or not path.exists():
        return
    try:
        path.chmod(0o600)
    except OSError as exc:
        LOGGER.warning("Unable to enforce 0600 permissions for %s: %s", path, exc)


def _migrate_legacy_config(target_path: Path) -> None:
    """Auto-copy the legacy config to the new location when it exists.

    Migration rules:
    - If target_path exists, do nothing.
    - If legacy config exists, copy it to target_path and warn.
    """

    if target_path.exists():
        return
    if target_path != CONFIG_PATH:
        return
    if not LEGACY_CONFIG_PATH.exists():
        return

    try:
        ensure_config_dir(CONFIG_DIR)
        target_path.write_text(
            LEGACY_CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8"
        )
        _enforce_private_permissions(target_path)
        LOGGER.warning(
            "config.migrated",
            extra={
                "event": "config.migrated",
                "legacy_path": str(LEGACY_CONFIG_PATH),
                "new_path": str(target_path),
            },
        )
    except OSError as exc:
        LOGGER.warning(
            "config.migrate_failed",
            extra={
                "event": "config.migrate_failed",
                "legacy_path": str(LEGACY_CONFIG_PATH),
                "new_path": str(target_path),
                "reason": str(exc),
            },
        )


def _safe_default_config() -> dict[str, dict[str, Any]]:
    """Return a deep copy of validated default config data."""
    return deepcopy(DEFAULT_CONFIG)


def _validate_config(raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Validate merged config and fallback to safe defaults when possible."""
    try:
        config = Config.model_validate(raw)
        return config.model_dump(by_alias=True)
    except ValidationError as exc:
        LOGGER.warning("Configuration validation failed, using safe defaults: %s", exc)
        return _safe_default_config()
    except Exception as exc:  # noqa: BLE001 - unexpected model construction failure.
        raise ConfigValidationError(f"Unable to validate configuration: {exc}") from exc


def load_config(config_path: Path | None = None) -> dict[str, dict[str, Any]]:
    """
    Load configuration from TOML, merge with defaults, and validate.

    The optional ``config_path`` argument is intended for tests and tooling.
    """
    target_path = config_path or CONFIG_PATH
    ensure_config_dir(target_path.parent)

    if config_path is None:
        _migrate_legacy_config(target_path)

    raw_data: dict[str, Any] = {}
    if target_path.exists():
        _enforce_private_permissions(target_path)
        try:
            raw_data = tomllib.loads(target_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - we must not crash on invalid user config.
            LOGGER.warning("Failed to parse config at %s: %s", target_path, exc)
            raw_data = {}

    merged = (
        _deep_merge(DEFAULT_CONFIG, raw_data)
        if isinstance(raw_data, dict)
        else _safe_default_config()
    )
    return _validate_config(merged)
