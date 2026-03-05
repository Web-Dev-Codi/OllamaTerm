<p align="center">
  <h1 align="center">OllamaTerm</h1>
  <p align="center">A keyboard-first, fully local AI chat interface for the terminal</p>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.11+-3776AB.svg?logo=python&logoColor=white" alt="Python"></a>
  <a href="https://ollama.com/"><img src="https://img.shields.io/badge/Ollama-Compatible-FF6F00.svg?logo=ollama&logoColor=white" alt="Ollama"></a>
  <a href="https://github.com/Web-Dev-Codi/OllamaTerm/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
  <a href="https://github.com/Web-Dev-Codi/OllamaTerm/stargazers"><img src="https://img.shields.io/github/stars/Web-Dev-Codi/OllamaTerm?color=yellow&logo=github" alt="Stars"></a>
  <a href="https://github.com/Web-Dev-Codi/OllamaTerm/issues"><img src="https://img.shields.io/github/issues/Web-Dev-Codi/OllamaTerm?color=0088ff" alt="Issues"></a>
</p>

<p align="center">
  Powered by <a href="https://ollama.com/">Ollama</a> and <a href="https://github.com/Textualize/textual">Textual</a> — no cloud, no API keys, no data leaving your machine.
</p>

---

## 🖥️ Preview

```
┌─────────────────────────────────────────────────────────┐
│  OllamaTerm                         [llama3.2] 🟢 Online │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  You  ────────────────────────────────────────────────  │
│  Explain async/await in Python in one paragraph.        │
│                                                         │
│  Assistant  ──────────────────────────────────────────  │
│  async/await is Python's syntax for writing coroutines  │
│  — functions that can pause execution with `await`,     │
│  yielding control back to the event loop while waiting  │
│  for I/O, then resuming where they left off...          │
│                                                         │
├─────────────────────────────────────────────────────────┤
│  > Type a message...                         ⌘P for help│
└─────────────────────────────────────────────────────────┘
```

---

## ⚠️ Disclaimer

> [!WARNING]
> **OllamaTerm is highly experimental software.** This project is under active development and may contain bugs, incomplete features, or breaking changes. Use at your own risk, especially when using coding tools that can modify files. Always review changes before committing and maintain backups of important data.

---

## 📑 Table of Contents

- [OllamaTerm](#ollamaterm)
  - [Preview](#-preview)
  - [Table of Contents](#-table-of-contents)
  - [Why OllamaTerm?](#-why-ollamaterm)
  - [Features](#-features)
  - [Requirements](#-requirements)
  - [Installation](#-installation)
  - [Quick Start](#-quick-start)
  - [Configuration](#️-configuration)
  - [Keybinds](#️-keybinds)
  - [Capabilities](#-capabilities)
  - [Desktop Integration](#️-desktop-integration)
  - [Packaging / Building](#-packaging--building)
  - [Development](#-development)
  - [Troubleshooting](#️-troubleshooting)
  - [Contributing](#-contributing)
  - [License](#-license)
  - [Show Your Support](#-show-your-support)
  - [Acknowledgments](#-acknowledgments)

---

## ❓ Why OllamaTerm?

| Feature | OllamaTerm | Web-based chat UIs |
|:--------|:-----------|:-------------------|
| 🔒 **Privacy** | 100% local — data never leaves your machine | Depends on provider |
| 📡 **Offline use** | Works after initial model pull | Requires internet |
| 💰 **Cost** | Free (you own the hardware) | Often metered |
| ⚡ **Speed** | No network latency to the model | Round-trip to cloud |
| 🎛️ **Customization** | Full TOML config, rebindable keys | Usually limited |
| 💻 **Terminal native** | Keyboard-first, scriptable | Browser tab |

---

## ✨ Features

#### 💬 Core Chat

> [!NOTE]
> All responses stream in real-time with smooth, flicker-free rendering.

- **Streaming responses** with batched rendering for smooth, flicker-free output
- **Animated "thinking" placeholder** shown while the model starts generating
- **Bounded context window** — automatically trims history to stay within token limits
- **Retry with backoff** for resilient streaming on transient Ollama failures

#### 🤖 Model Management

- **Multi-model config** — list multiple models and switch at runtime
- **Clickable model picker** in the status bar, or `ctrl+m` keyboard shortcut
- **Auto-pull on startup** — optionally pull the configured model if it is not present
- **Traffic-light connection indicator** — always know if Ollama is reachable

#### 📁 Conversation Persistence

- **Save and load** conversation history (JSON format)
- **Export** conversations as Markdown transcripts
- **Search messages** and cycle through results from the input box
- **Copy** the latest assistant reply to clipboard in one shortcut
- **Conversation picker & auto-save** — quickly switch between saved chats (via `/conversations`) with automatic saving enabled by default

#### 🧠 Capabilities

> [!TIP]
> Capabilities are auto-detected from Ollama's `/api/show` endpoint — no manual configuration needed!

- **Auto-detected per model** — thinking, tool calling, and vision support are read from Ollama's `/api/show` at load time; no config required
- **Seamless model switching** — capabilities update instantly when you switch models mid-conversation
- **Chain-of-thought reasoning** for models that support it (e.g. `qwen3`, `deepseek-r1`, `deepseek-v3.1`, `gpt-oss`)
- **Tool calling** and a full agent loop for multi-step model actions
- **Custom coding tools** (`read`, `grep`, `glob`, `ls`, `write`, `edit`, `multiedit`, `apply_patch`, `bash`, `batch`, planning/todo/task tools, and more)
- **Web search** via Ollama's built-in tools (requires an Ollama API key)
- **Vision / image attachments** for vision-capable models (e.g. `gemma3`, `llava`)
- **Context window alignment** — `max_context_tokens` is forwarded to Ollama as `options.num_ctx` so the server-side context window always matches the client-side trim budget

#### 🎨 Interface & Integration

- **Command palette** (`ctrl+p`) with searchable list of all actions
- **Fully configurable keybinds** via TOML
- **Structured JSON logging** with optional file output for debugging
- **Terminal title** and window class set on startup for WM rules
- **Desktop entry** and Hyprland/Ghostty integration examples included
- **Slash commands** for actions like `/new`, `/save`, `/load`, `/model`, `/preset`, `/image`, `/file`, `/conversations`, and `/help`
- **Theme system** with a theme picker (`ctrl+t`), built-in themes (e.g. Textual dark/light, Nord, Gruvbox, Tokyo Night) and customizable color themes with persistence

---

## 📋 Requirements

| Requirement | Details |
|:------------|:--------|
| 🐍 **Python** | 3.11 or newer |
| 🦙 **Ollama** | Installed and on your `PATH` ([install guide](https://ollama.com/download)) |
| ▶️ **Ollama daemon** | Running — `ollama serve` |
| 🌐 **Internet** | Only needed once, to pull models |

---

## 📦 Installation

### 🔧 From source (recommended)

```bash
# Clone the repository
git clone https://github.com/Web-Dev-Codi/OllamaTerm.git
cd OllamaTerm

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install the package
pip install -e .
```

### 👨‍💻 Developer / contributor install

```bash
pip install -e '.[dev]'
```

### 📦 Arch Linux (PKGBUILD)

A `PKGBUILD` is included for building a native Arch package:

```bash
makepkg -si
```

---

## 🚀 Quick Start

### 1️⃣ Start Ollama and pull a model

```bash
ollama serve
ollama pull llama3.2
```

### 2️⃣ (Optional) Copy the example config

```bash
mkdir -p ~/.config/ollamaterm
cp config.example.toml ~/.config/ollamaterm/config.toml
```

### 3️⃣ Launch the app

```bash
ollamaterm
# or
python -m ollama_chat
```

### 4️⃣ Basic workflow

| Action | How |
|:-------|:------|
| 💬 Send a message | Type in the input field → `ctrl+enter` |
| 🔄 Switch model | Click `Model` in the status bar, or `ctrl+m` |
| 📄 New conversation | `ctrl+n` |
| 🔍 Search messages | `ctrl+f`, type query, press again to cycle |
| 📋 Copy last reply | `ctrl+y` |
| 🎯 Open all actions | `ctrl+p` |
| ❌ Quit | `ctrl+q` |

---

## ⚙️ Configuration

### 📍 Config File Location

```
~/.config/ollamaterm/config.toml
```

If the file does not exist, built-in defaults are used automatically.  
Use `config.example.toml` from the repo as your starting point.

### All Options

```toml
[app]
# Window title shown in the TUI header
title = "OllamaTerm"
# WM window class set on startup (useful for Hyprland/i3 rules)
class = "ollamaterm"
# How often (seconds) to check Ollama connectivity
connection_check_interval_seconds = 15

[ollama]
# Ollama API endpoint
host = "http://localhost:11434"
# Default active model
model = "llama3.2"
# All models available in the picker
models = ["llama3.2", "qwen2.5", "mistral"]
# Request timeout in seconds
timeout = 120
# System prompt injected at the start of every conversation
system_prompt = "You are a helpful assistant."
# Maximum messages kept in history
max_history_messages = 200
# Token budget for context trimming
max_context_tokens = 4096
# Pull the model on startup if not present locally
pull_model_on_start = true

[ui]
font_size = 14
background_color = "#1a1b26"
user_message_color = "#7aa2f7"
assistant_message_color = "#9ece6a"
border_color = "#565f89"
show_timestamps = true
# Number of streaming chunks to buffer before rendering
stream_chunk_size = 8

[theme]
# Theme selection: "textual-dark", "textual-light", "nord", "gruvbox", "tokyo-night",
# "monokai", "dracula", "solarized-light", "solarized-dark", "atom-one-dark", "atom-one-light"
# or "custom" to use the ui colors above
name = "textual-dark"
# Persist theme choice across sessions
persist = true
# Custom theme definitions (optional)
[theme.custom]
# Define custom themes here - example for a "catppuccin" theme:
# [theme.custom.catppuccin]
# primary = "#89B4FA"
# secondary = "#74C7EC"
# accent = "#F5C2E7"
# foreground = "#CDD6F4"
# background = "#1E1E2E"
# surface = "#313244"
# panel = "#45475A"
# success = "#A6E3A1"
# warning = "#F9E2AF"
# error = "#F38BA8"
# dark = true

[keybinds]
send_message = "ctrl+enter"
new_conversation = "ctrl+n"
quit = "ctrl+q"
scroll_up = "ctrl+k"
scroll_down = "ctrl+j"
command_palette = "ctrl+p"
toggle_model_picker = "ctrl+m"
toggle_theme_picker = "ctrl+t"
save_conversation = "ctrl+s"
load_conversation = "ctrl+l"
export_conversation = "ctrl+e"
search_messages = "ctrl+f"
copy_last_message = "ctrl+y"
interrupt_stream = "escape"

[security]
# Set true to allow non-localhost Ollama endpoints
allow_remote_hosts = false
allowed_hosts = ["localhost", "127.0.0.1", "::1"]

[logging]
level = "INFO"          # DEBUG | INFO | WARNING | ERROR
structured = true       # JSON-formatted log lines
log_to_file = false
log_file_path = "~/.local/state/ollamaterm/app.log"

[persistence]
enabled = true
auto_save = true
directory = "~/.local/state/ollamaterm/conversations"
metadata_path = "~/.local/state/ollamaterm/conversations/index.json"

[tools]
# Enable schema-first custom coding tools
enabled = true
# Base root for file/search/edit tools
workspace_root = "."
# Allow temporary external roots via external-directory tool
allow_external_directories = false
command_timeout_seconds = 30
max_output_lines = 200
max_output_bytes = 50000
max_read_bytes = 200000
max_search_results = 200
default_external_directories = []

[capabilities]
# Show the model's reasoning trace inside the assistant bubble.
# Thinking support itself is auto-detected — this controls only the UI display.
show_thinking = true

# Built-in web_search / web_fetch (requires OLLAMA_API_KEY or web_search_api_key).
# Only active when the model also supports tool calling (auto-detected).
web_search_enabled = false
web_search_api_key = ""

# Max tool-call iterations per message before the agent loop stops.
max_tool_iterations = 10

# NOTE: thinking support, tool calling, and vision are detected automatically
# from Ollama's /api/show response — no manual flags needed.
```

---

## ⌨️ Keybinds

> [!NOTE]
> All keybinds are rebindable in `[keybinds]`. These are the defaults:

| Keybind | Action |
|---|---|
| `ctrl+enter` | Send message |
| `ctrl+n` | New conversation |
| `ctrl+q` | Quit |
| `ctrl+k` | Scroll up |
| `ctrl+j` | Scroll down |
| `ctrl+p` | Open command palette |
| `ctrl+m` | Open model picker |
| `ctrl+t` | Open theme picker |
| `ctrl+s` | Save conversation *(requires persistence enabled)* |
| `ctrl+l` | Load latest saved conversation *(requires persistence enabled)* |
| `ctrl+e` | Export Markdown transcript *(requires persistence enabled)* |
| `ctrl+f` | Search messages (press again to cycle results) |
| `ctrl+y` | Copy last assistant message to clipboard |
| `escape` | Interrupt a streaming response |

---

## 🔮 Capabilities

> [!IMPORTANT]
> Thinking, tool calling, and vision support are **detected automatically** from
> Ollama's `/api/show` endpoint each time a model is loaded or switched. No
> manual configuration is required — the status bar icons (🧠 🔧 👁) reflect
> what the active model actually supports.

> [!NOTE]
> **Requires Ollama ≥ 0.6** for capability metadata. Older Ollama versions fall
> back gracefully — all features are assumed enabled and gated only by whether
> the model responds correctly.

#### 🧠 Chain-of-thought reasoning

Automatically active when the model reports `"thinking"` in its capabilities
(e.g. `qwen3`, `deepseek-r1`, `deepseek-v3.1`, `gpt-oss`). The model's
internal reasoning trace is shown above the final answer when
`show_thinking = true` in `[capabilities]`.

**GPT-OSS note:** GPT-OSS requires a string think level rather than a boolean.
OllamaTerm detects GPT-OSS by name and automatically sends `think="medium"`.

#### 🔧 Tool calling

Automatically active when the model reports `"tools"` in its capabilities.
The agent loop allows the model to invoke tools multiple times before producing
a final answer. Control the upper bound with `max_tool_iterations` in
`[capabilities]`.

In addition to Ollama web tools, OllamaTerm now ships a schema-first local
coding toolset designed for agentic workflows:

- File and search tools: `read`, `ls`, `glob`, `grep`, `codesearch`
- Editing tools: `write`, `edit`, `multiedit`, `apply_patch`
- Runtime tools: `bash`, `batch`, `external-directory`
- Planning/state tools: `plan-enter`, `plan-exit`, `plan`, `todo`, `todoread`, `todowrite`, `task`, `question`
- Introspection tools: `registry`, `tool`, `truncation`, `invalid`

These tools are controlled by the `[tools]` config section and are constrained
by workspace-root path checks, command timeouts, and output truncation limits.

##### 🔌 Function tools with Ollama (alpha/experimental)

OllamaTerm passes tools to the Ollama Python SDK in two forms:

- JSON function tools generated from the schema-first tool specs (the majority of tools below)
- Python callables for built-in Ollama integrations when enabled (e.g. `web_search`, `web_fetch`)

The model emits `tool_calls`, the app executes them, appends a `tool` role message with the result, and continues the loop until the assistant returns a final answer.

> Warning: This tool suite is experimental. Most tools are untested and may be buggy or missing edge-case handling. Use with caution and review changes carefully, especially file edits. Outputs may be truncated according to configured limits.

##### Available tools (names and key parameters)

- Files & search
  - `list` (built-in) — List files and directories.
    - `path?: string` (default: workspace root)
  - `ls` (custom) — Alternate directory listing with tree-style output.
    - `path?: string`, `ignore?: string[]`
  - `read` — Read a file window.
    - `path: string`, `offset?: int`, `limit?: int`
  - `glob` — Find files by glob.
    - `pattern: string`, `path?: string`, `max_results?: int`
  - `grep` / `codesearch` — Search file contents.
    - `query: string`, `path?: string`, `case_sensitive?: bool`, `fixed_strings?: bool`, `max_results?: int`

- Editing
  - `write` — Atomic full-file write.
    - `path: string`, `content: string`, `overwrite?: bool`, `create_dirs?: bool`
  - `edit` — Single snippet replace.
    - `path: string`, `old_text: string`, `new_text: string`, `replace_all?: bool`
  - `multiedit` — Multiple snippet edits atomically.
    - `path: string`, `edits: { old_text, new_text, replace_all? }[]`
  - `apply_patch` — Apply structured patch hunks.
    - `path: string`, `hunks: { old_text, new_text, replace_all? }[]`

- Runtime
  - `bash` — Run a shell command (capped by time/output limits).
    - `command: string`, `cwd?: string`
  - `batch` — Run a sequence of tool calls.
    - `calls: { name: string, arguments: object }[]`, `continue_on_error?: bool`
  - `external-directory` — Manage temporary external directory allowlist for this session.
    - `action: string`, `path?: string`

- Planning & state
  - `plan-enter` | `plan-exit` | `plan`
    - `plan-enter: { goal?: string }`
    - `plan: { action?: string, content?: string }`
  - `todo` | `todoread` | `todowrite` | `task`
    - `todo: { item: string }`
    - `todowrite: { items: string[], mode?: "append"|"replace" }`
    - `task: { action?: string, name?: string, status?: string }`
  - `question` — Emit a structured clarification question.
    - `prompt: string`, `context?: string`

- Introspection & utility
  - `registry` — List available tools.
  - `tool` — Inspect a tool definition.
  - `truncation` — Show output truncation limits.
  - `invalid` — Always fails (for error-path testing).

- Web (requires tool-capable model; `web_search_enabled = true` and an API key)
  - `websearch` — Perform a web search via Ollama integration.
    - `query: string`, `max_results?: int`
  - `webfetch` — Fetch a URL via Ollama integration.
    - `url: string`

Notes:

- Directory listing may appear as `list` (built-in) or `ls` (custom) depending on which tool set is active. Both list files; prefer `list` when available.
- File and command tools will prompt for permission. Paths are restricted to the configured workspace by default.
- Large outputs are truncated. Use `offset`/`limit` (for `read`) and `max_results` (for `grep`/`glob`) to scope results.

##### Quick examples

```text
List files here → Call tool: list { "path": "." }
Search for a string → Call tool: grep { "query": "TODO", "path": "." }
Read a file window → Call tool: read { "path": "src/main.py", "offset": 1, "limit": 120 }
Make an edit → Call tool: edit { "path": "README.md", "old_text": "foo", "new_text": "bar", "replace_all": true }
```

#### 🌐 Web search

Set `web_search_enabled = true` in `[capabilities]` and provide an Ollama API
key (via `web_search_api_key` or the `OLLAMA_API_KEY` environment variable).
Web search also requires the active model to support tool calling
(auto-detected) — it is silently disabled for models that do not.

#### 👁️ Vision / image attachments

Automatically active when the model reports `"vision"` in its capabilities
(e.g. `gemma3`, `llava`). Attach images with `/image <path>` in the input box
or use the Attach button in the toolbar. Use `/file <path>` or the file attachment button to include non-image context files.

#### 📏 Context window alignment

`max_context_tokens` (in `[ollama]`) serves two purposes:

1. **Client-side** — conversation history is trimmed to stay within this token budget before being sent
2. **Server-side** — the value is forwarded to Ollama as `options.num_ctx` so the model's context window matches; without this, Ollama may use a smaller default and silently truncate longer conversations

Increase this value for models with larger native context windows (e.g. set
`max_context_tokens = 32768` for `llama3.2` or `qwen3`).

---

## 🖥️ Desktop Integration

### 🌀 Hyprland + Ghostty

The app sets the terminal window class from `app.class` on startup.
For the most reliable behavior on Wayland, also pass the class directly to
your terminal:

```bash
ghostty --class=ollamaterm-tui -e ollamaterm
```

Suggested Hyprland window rules (`~/.config/hypr/hyprland.conf`):

```conf
windowrulev2 = float,          class:^(ollamaterm-tui)$
windowrulev2 = size 1200 800,  class:^(ollamaterm-tui)$
windowrulev2 = center,         class:^(ollamaterm-tui)$
windowrulev2 = opacity 0.95,   class:^(ollamaterm-tui)$

bind = $mainMod, O, exec, ghostty --class=ollamaterm-tui -e ollamaterm
```

### 📑 Desktop Entry

Create `~/.local/share/applications/ollamaterm.desktop`:

```desktop
[Desktop Entry]
Type=Application
Name=OllamaTerm
Comment=ChatGPT-style TUI for Ollama local LLMs
Exec=ghostty --class=ollamaterm-tui -e ollamaterm
Icon=utilities-terminal
Terminal=false
Categories=Utility;TerminalEmulator;Development;
```

---

## 📦 Packaging / Building

### 🎡 Local wheel build (isolated)

```bash
python -m pip install build
python -m build --wheel
# optional: install into current env
python -m pip install --force-reinstall dist/*.whl
# or install for user with pipx
pipx install .
```

Troubleshooting: if you see `BackendUnavailable: Cannot import 'setuptools.build_meta'`, either run the isolated build above, or install/upgrade in your active environment:

```bash
python -m pip install -U setuptools wheel build
# For Python 3.14 pre-releases, you may need:
python -m pip install --pre -U setuptools
```

### 🏗️ Arch package (PKGBUILD)

This repo ships a `PKGBUILD` that builds without network access using system makedepends:

```bash
sudo pacman -S --needed base-devel python-setuptools python-build python-installer python-wheel
makepkg -si
```

The `build()` step uses `python -m build --wheel --no-isolation` so it relies on the above makedepends instead of downloading during build.

---

## 👨‍💻 Development

```bash
# Full test suite
pytest -q

# With coverage report
pytest --cov=ollama_chat --cov-report=term-missing -q

# Lint
ruff check .

# Format check
black --check .

# Type check
mypy ollama_chat/
```

Run all checks before submitting changes:

```bash
ruff check . && black --check . && mypy ollama_chat/ && pytest -q
```

---

## 🛠️ Troubleshooting

| Symptom | Fix |
|---|---|
| `Connection error` on startup | Ensure `ollama serve` is running; verify `ollama.host` in config |
| "Model not found" warning | Set `pull_model_on_start = true`, or run `ollama pull <model>` manually |
| Empty or cut-off response | Check `ollama list` to confirm the model name; review Ollama logs |
| Thinking / tools / vision not activating | Requires Ollama ≥ 0.6; run `ollama show <model>` and confirm `capabilities` is listed |
| Response cuts off mid-conversation | Increase `max_context_tokens` in `[ollama]` to match the model's native context window |
| Keybind not responding | Verify the syntax in `[keybinds]` and restart the app |
| Colors not applied | Use valid hex format: `#RRGGBB` or `#RGB` |
| Window class rule not matching | Ensure `app.class` is set; prefer launching with `ghostty --class=ollamaterm-tui` |
| Tool loop not stopping | Lower `max_tool_iterations` in `[capabilities]` |
| Web search not working | Confirm the model supports tool calling (`ollama show <model>`); set `web_search_enabled = true` and provide `OLLAMA_API_KEY` |

---

---

## 🤝 Contributing

Contributions are always welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please read the [Code of Conduct](CODE_OF_CONDUCT.md) and review our [Contributing Guide](CONTRIBUTING.md) for details.

---

## 📄 License

<p align="center">
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="MIT License">
  </a>
  <br>
  <b>MIT License — © Web-Dev-Codi</b>
</p>

---

## ⭐ Show Your Support

If you find OllamaTerm useful, please consider giving it a star!

<p align="center">
  <a href="https://github.com/Web-Dev-Codi/OllamaTerm/stargazers">
    <img src="https://img.shields.io/github/stars/Web-Dev-Codi/OllamaTerm?style=social" alt="Star this project">
  </a>
</p>

---

## 🙏 Acknowledgments

- [Ollama](https://ollama.com/) — Local LLM runtime
- [Textual](https://github.com/Textualize/textual) — TUI framework

---

<p align="center">
  <sub>Built with ❤️ and lots of ☕ by the OllamaTerm contributors</sub>
  <br>
  <sub>🚀 Made for the terminal, by terminal lovers</sub>
</p>
