"""Source-adapter registry — the one place CLIs are wired in.

To add a CLI (codex, opencode, ollama, ...): implement sources/<cli>.py with the
SessionSource Protocol, add one line to _FACTORIES, and add a [sources.<cli>]
block to config.toml. The indexer, DB, UI, watcher, and MCP server need no edits.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import sbconfig
from sources.base import SessionSource
from sources.claude import ClaudeSource


def _cli_home(configured, documented: str, env_var: str, suffix: str):
    """Where a CLI keeps its state: the configured path, unless it is still the
    documented default AND the CLI's own relocation variable is set — then
    follow the CLI (a multi-account setup with CLAUDE_CONFIG_DIR / CODEX_HOME /
    XDG_DATA_HOME would otherwise index the wrong, usually empty, tree with
    no error at all). An explicit non-default path in config.toml always wins."""
    if configured in (None, "", documented):
        base = os.environ.get(env_var)
        if base:
            return Path(os.path.expanduser(base)) / suffix
    return configured or documented


def _make_claude() -> SessionSource:
    cfg = sbconfig.source_config("claude")
    return ClaudeSource(_cli_home(cfg.get("projects_dir"), "~/.claude/projects",
                                  "CLAUDE_CONFIG_DIR", "projects"))


def _make_copilot() -> SessionSource:
    from sources.copilot import CopilotSource
    cfg = sbconfig.source_config("copilot")
    return CopilotSource(_cli_home(cfg.get("state_dir"), "~/.copilot/session-state", "", ""))


def _make_codex() -> SessionSource:
    from sources.codex import CodexSource
    cfg = sbconfig.source_config("codex")
    return CodexSource(
        _cli_home(cfg.get("sessions_dir"), "~/.codex/sessions", "CODEX_HOME", "sessions"),
        _cli_home(cfg.get("archived_sessions_dir"), "~/.codex/archived_sessions",
                  "CODEX_HOME", "archived_sessions"))


def _make_opencode() -> SessionSource:
    from sources.opencode import DATA_DIR, MIRROR_DIR, OpenCodeSource
    cfg = sbconfig.source_config("opencode")
    return OpenCodeSource(_cli_home(cfg.get("data_dir"), "~/.local/share/opencode",
                                    "XDG_DATA_HOME", "opencode") or DATA_DIR,
                          cfg.get("mirror_dir", MIRROR_DIR),
                          db=cfg.get("db"), reimport_on_restore=cfg.get("reimport_on_restore", True))


_FACTORIES: dict[str, Callable[[], SessionSource]] = {
    "claude": _make_claude,
    "copilot": _make_copilot,
    "codex": _make_codex,
    "opencode": _make_opencode,
}


def build_source_registry(only_available: bool = False) -> dict[str, SessionSource]:
    """Instantiate adapters for every enabled+known source."""
    registry: dict[str, SessionSource] = {}
    for name in sbconfig.enabled_sources():
        factory = _FACTORIES.get(name)
        if factory is None:
            continue
        try:
            adapter = factory()
        except Exception:  # noqa: BLE001 — a not-yet-built adapter shouldn't break the rest
            continue
        if only_available and not adapter.is_available():
            continue
        registry[name] = adapter
    return registry
