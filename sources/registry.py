"""Source-adapter registry — the one place CLIs are wired in.

To add a CLI (codex, opencode, ollama, ...): implement sources/<cli>.py with the
SessionSource Protocol, add one line to _FACTORIES, and add a [sources.<cli>]
block to config.toml.example (the committed defaults; config.toml is a
git-ignored per-machine override). The indexer, DB, UI, watcher, and MCP server
need no edits.

What this module actually does at runtime: turn the `[sources.*]` configuration into
ready-to-use adapter objects. Two jobs, both small but load-bearing:

  * Deciding WHERE each CLI keeps its files — the configured path, or the CLI's own
    relocation environment variable when the config is still on the documented default
    (see _cli_home, and the `[portability]` notes in docs/ARCHITECTURE.md).
  * Instantiating only the sources the user enabled, and never letting one broken or
    not-yet-written adapter stop the others from loading.

Callers: watcher.py (to learn which directories to watch), backfill/reconcile/prune, the
Flask UI, the MCP server, and the `cr` resume helper. The adapters themselves are cheap
to construct — no file is read until discover()/parse_header() is called — so callers
build a fresh registry rather than sharing one.

Terms (session, transcript, adapter, registry) are defined in docs/GLOSSARY.md.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import sbconfig
from sources.base import SessionSource
# Claude is imported eagerly because it is always present; the other adapters are
# imported inside their factory so that an import error (a missing optional dependency
# such as PyYAML for Copilot) disables just that source instead of the whole tool.
from sources.claude import ClaudeSource


def _cli_home(configured, documented: str, env_var: str, suffix: str):
    """Where a CLI keeps its state: the configured path, unless it is still the
    documented default AND the CLI's own relocation variable is set — then
    follow the CLI (a multi-account setup with CLAUDE_CONFIG_DIR / CODEX_HOME /
    XDG_DATA_HOME would otherwise index the wrong, usually empty, tree with
    no error at all). An explicit non-default path in config.toml always wins.

    Arguments: `configured` is whatever config.toml said (possibly None); `documented` is
    the default shipped in config.toml.example, e.g. "~/.claude/projects"; `env_var` is
    the CLI's OWN relocation variable, e.g. "CLAUDE_CONFIG_DIR"; `suffix` is what the CLI
    appends inside that home, e.g. "projects". Pass env_var="" for a CLI that has no such
    variable (Copilot), which makes this a plain "configured or documented".

    Worked example — CLAUDE_CONFIG_DIR=/Volumes/work/.claude and no `projects_dir` in
    config.toml gives /Volumes/work/.claude/projects. Set `projects_dir` to anything
    other than the documented default and that wins instead, environment or not.

    Returns a Path when it followed the environment, otherwise the configured/documented
    value unchanged (a string) — every adapter expands and normalises its own argument,
    so both are acceptable here.
    """
    if configured in (None, "", documented):
        base = os.environ.get(env_var)
        if base:
            return Path(os.path.expanduser(base)) / suffix
    return configured or documented


def _make_claude() -> SessionSource:
    """Claude Code: one directory of transcripts, relocatable with CLAUDE_CONFIG_DIR."""
    cfg = sbconfig.source_config("claude")
    return ClaudeSource(_cli_home(cfg.get("projects_dir"), "~/.claude/projects",
                                  "CLAUDE_CONFIG_DIR", "projects"))


def _make_copilot() -> SessionSource:
    """GitHub Copilot CLI: one directory per session under ~/.copilot/session-state.

    Copilot has no relocation environment variable, so env_var/suffix are empty and
    _cli_home() degrades to "the configured path, else the documented one".
    """
    from sources.copilot import CopilotSource
    cfg = sbconfig.source_config("copilot")
    return CopilotSource(_cli_home(cfg.get("state_dir"), "~/.copilot/session-state", "", ""))


def _make_codex() -> SessionSource:
    """OpenAI Codex CLI: TWO trees — live rollouts and archived ones — both under
    CODEX_HOME. The adapter reports both to the watcher via watch_roots()."""
    from sources.codex import CodexSource
    cfg = sbconfig.source_config("codex")
    return CodexSource(
        _cli_home(cfg.get("sessions_dir"), "~/.codex/sessions", "CODEX_HOME", "sessions"),
        _cli_home(cfg.get("archived_sessions_dir"), "~/.codex/archived_sessions",
                  "CODEX_HOME", "archived_sessions"))


def _make_opencode() -> SessionSource:
    """OpenCode: everything lives in one SQLite database, so this adapter also needs the
    mirror directory it projects that database into (one JSONL file per session, which is
    what every other part of this tool expects to find). `db` overrides the database
    location; `reimport_on_restore` decides whether restoring also puts the session back
    into OpenCode itself."""
    from sources.opencode import DATA_DIR, MIRROR_DIR, OpenCodeSource
    cfg = sbconfig.source_config("opencode")
    return OpenCodeSource(_cli_home(cfg.get("data_dir"), "~/.local/share/opencode",
                                    "XDG_DATA_HOME", "opencode") or DATA_DIR,
                          cfg.get("mirror_dir", MIRROR_DIR),
                          db=cfg.get("db"), reimport_on_restore=cfg.get("reimport_on_restore", True))


# The complete list of CLIs this build knows about. The key is the name used in
# config.toml's [sources.<key>] block and stored in sessions.cli_source, so it is part of
# the data format — renaming one would orphan existing rows. Adding a CLI is one line
# here plus a new sources/<cli>.py.
_FACTORIES: dict[str, Callable[[], SessionSource]] = {
    "claude": _make_claude,
    "copilot": _make_copilot,
    "codex": _make_codex,
    "opencode": _make_opencode,
}


def build_source_registry(only_available: bool = False) -> dict[str, SessionSource]:
    """Instantiate adapters for every enabled+known source.

    Returns {name: adapter}, in config-file order. This is the entry point everything
    else uses; tests call it directly and assert the expected keys are present.

    `only_available=True` additionally drops adapters with nothing to read on this
    machine. Note what "available" means (see SessionSource.is_available): transcripts
    exist on disk — NOT that the CLI is installed. Indexing and watching must keep
    working after a CLI is uninstalled, because those old transcripts are precisely what
    this tool exists to preserve. Leave it False when you want a home for a CLI that has
    not run yet: the watcher deliberately asks for all enabled sources so it can start
    watching a directory the moment it appears.

    Never raises for a bad source. An unknown name (config from a newer version) is
    skipped, and a factory that blows up — missing dependency, malformed path, an adapter
    module that does not exist yet — is skipped too, so one broken CLI cannot take the
    other three down with it.
    """
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
