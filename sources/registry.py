"""Source-adapter registry — the one place CLIs are wired in.

To add a CLI (codex, opencode, ollama, ...): implement sources/<cli>.py with the
SessionSource Protocol, add one line to _FACTORIES, and add a [sources.<cli>]
block to config.toml. The indexer, DB, UI, watcher, and MCP server need no edits.
"""
from __future__ import annotations

from typing import Callable

import sbconfig
from sources.base import SessionSource
from sources.claude import ClaudeSource


def _make_claude() -> SessionSource:
    cfg = sbconfig.source_config("claude")
    return ClaudeSource(cfg.get("projects_dir", "~/.claude/projects"))


def _make_copilot() -> SessionSource:
    from sources.copilot import CopilotSource
    cfg = sbconfig.source_config("copilot")
    return CopilotSource(cfg.get("state_dir", "~/.copilot/session-state"))


def _make_codex() -> SessionSource:
    from sources.codex import CodexSource
    cfg = sbconfig.source_config("codex")
    return CodexSource(cfg.get("sessions_dir", "~/.codex/sessions"))


_FACTORIES: dict[str, Callable[[], SessionSource]] = {
    "claude": _make_claude,
    "copilot": _make_copilot,
    "codex": _make_codex,
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
