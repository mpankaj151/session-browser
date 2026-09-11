"""Central config + path resolution for the Session Browser.

Loads config.toml (falling back to config.toml.example), expands ~ in paths, and
exposes the canonical filesystem locations. Importable from every script and the
Flask app so there is one source of truth.

Where this sits in the pipeline: nowhere near the data. Nothing here opens a transcript
(the file a coding CLI writes for one conversation — see docs/GLOSSARY.md) or touches
the registry database. This module only answers "where does that live, and what did the
user configure?". It is imported — usually first — by the indexer, the watcher, every
script under scripts/, the Flask UI and the MCP server, so an exception raised at import
time here takes the whole tool down. Hence the defensive style below: every lookup is
`.get()` with the documented default rather than an index that could KeyError on a
hand-edited config.

Two layers of configuration, merged at import time:
  1. config.toml.example — committed, complete, documented: the defaults.
  2. config.toml         — git-ignored, one per machine: only the overrides.
See load_config() for why layering (rather than "the file, if present") matters.

Environment variables override both, because they describe *this* machine's layout:
  SB_CONFIG          path to the override file itself
  SB_DB              path to the registry database (used by `sb demo` for a throwaway one)
  CLAUDE_CONFIG_DIR  a relocated Claude Code home; consulted here only for the legacy
                     database location (the source adapters use it via sources/registry.py)

Vocabulary used below (session, transcript, registry, enrichment, facet, token...) is
defined once in docs/GLOSSARY.md; docs/ARCHITECTURE.md draws the pipeline.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

# The checkout this file lives in. .resolve() follows symlinks, so a repo reached through
# a symlinked path still finds config.toml.example and pricing.json next to the real file.
_REPO_ROOT = Path(__file__).resolve().parent


def _expand(p: str) -> Path:
    """Turn a configured path string into a real absolute path.

    Expands a leading `~` and resolves symlinks and `..`, because several call sites
    compare paths for containment (an adapter refusing to write outside its own tree, for
    example) and two spellings of the same directory must compare equal.
    """
    return Path(os.path.expanduser(p)).resolve()


def _deep_merge(base: dict, override: dict) -> dict:
    """base with override's keys applied; nested tables merge, everything else replaces.

    Recursive so that `[sources.claude] enabled = false` in config.toml overrides only
    that one key and inherits the rest of the example's `[sources.claude]` block. A plain
    dict.update() would have replaced the whole table and dropped the other keys.
    """
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_toml(path: Path) -> dict[str, Any]:
    """Parse one TOML file into nested dicts. Binary mode: tomllib only accepts bytes,
    and decoding is TOML's own job (it is always UTF-8)."""
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def config_path() -> Path:
    """The per-machine override file: $SB_CONFIG, else <repo>/config.toml.

    $SB_CONFIG lets the test suite and `sb demo` point the whole stack at a scratch
    config without touching the user's own. Note this is deliberately not _expand()ed:
    the file need not exist yet (load_config() checks), and resolving a missing path is
    pointless here.
    """
    env = os.environ.get("SB_CONFIG")
    return Path(os.path.expanduser(env)) if env else _REPO_ROOT / "config.toml"


def load_config(path: Path | str | None = None, defaults: Path | str | None = None) -> dict[str, Any]:
    """config.toml LAYERED over config.toml.example.

    The example is the complete, documented default set; config.toml holds a
    machine's overrides. Any section or key the override omits inherits the
    example's value, so a config.toml written before a source or enrichment
    provider existed keeps working after `git pull` instead of silently
    dropping the new [sources.<cli>] block. Switch a source off explicitly with
    `enabled = false` — deleting its block no longer does that.

    Both arguments exist for the tests, which layer a temporary override over a temporary
    defaults file: `path` is the override (default: config_path()), `defaults` is the
    baseline (default: <repo>/config.toml.example). A missing file on either side is not
    an error — a fresh checkout with no config.toml must still start up.
    """
    defaults_path = Path(defaults) if defaults else _REPO_ROOT / "config.toml.example"
    cfg = _load_toml(defaults_path) if defaults_path.exists() else {}
    override_path = Path(os.path.expanduser(str(path))) if path else config_path()
    if override_path.exists():
        cfg = _deep_merge(cfg, _load_toml(override_path))
    return cfg


# Read once, at import time. Every module shares this dict, so a config change needs a
# restart of the watcher / UI — which is what the installer does anyway.
CONFIG = load_config()
CONFIG_PATH = config_path()

REPO_ROOT = _REPO_ROOT

# --- Canonical paths (dual DB-path resolution: new location, legacy fallback) --
# .get() with defaults everywhere: a hand-edited config.toml missing a key must
# degrade to the documented default, not KeyError every tool at import time.
_PATHS = CONFIG.get("paths", {})
# Where the registry lives today.
_NEW_DB = _expand(_PATHS.get("db", "~/.session-browser/registry.db"))
# Where early versions kept it, inside Claude Code's own home. Still honoured so an
# existing install keeps its history after an upgrade instead of starting from an empty
# database; CLAUDE_CONFIG_DIR is followed because that home can be relocated.
_OLD_DB = Path(os.path.expanduser(os.environ.get("CLAUDE_CONFIG_DIR") or "~/.claude")) / "session-registry.db"
# SB_DB env override wins over everything — used by `sb demo` to point the whole
# stack at a throwaway seeded database without touching the real registry.
_ENV_DB = os.environ.get("SB_DB")
if _ENV_DB:
    DB_PATH = Path(os.path.expanduser(_ENV_DB))
else:
    # Prefer whichever one actually exists; if neither does (first install) create the
    # new one. Never both: _NEW_DB wins, so a half-finished migration is not ambiguous.
    DB_PATH = _NEW_DB if _NEW_DB.exists() else (_OLD_DB if _OLD_DB.exists() else _NEW_DB)

# Enrichment output (the short structured summary a model writes about a session) is
# stored twice: as JSON files here, and in registry columns. The files are the durable
# copy — the database can be rebuilt from transcripts, these cannot.
FACETS_DIR = _expand(_PATHS.get("facets_dir", "~/.session-browser/facets"))
# The 30-second "a hook just indexed this session" note shared with the watcher; the
# file format and TTL live in hookstate.py.
HOOK_STATE = _expand(_PATHS.get("hook_state", "~/.session-browser/.hook-state.json"))
# Background jobs have no terminal, so the watcher and the nightly scripts log here.
LOG_DIR = _expand(_PATHS.get("log_dir", "~/.session-browser/logs"))

# The reasoning archive holds two things: readable/ (the assistant's own "thinking" text
# rendered as Markdown) and raw/ (a byte-for-byte copy of every transcript). The raw copy
# is what makes a session restorable after its CLI deletes the original, so switching
# `enabled` off also gives up restore. See docs/GLOSSARY.md -> "reasoning archive".
REASONING_ENABLED = bool(CONFIG.get("reasoning", {}).get("enabled", True))
REASONING_ARCHIVE = _expand(CONFIG.get("reasoning", {}).get("archive_dir", "~/claude-reasoning-archive"))

# One Markdown page per local day, assembled from per-session journals with no model call.
DAILY_DIR = _expand(CONFIG.get("digest", {}).get("daily_dir", "~/.session-browser/daily-logs"))

# Generated weekly/monthly reports, plus an optional user-written style file that is
# prepended to the report prompt (kept outside the repo so `git pull` never clobbers it).
REPORTS_DIR = _expand(CONFIG.get("reports", {}).get("out_dir", "~/.session-browser/reports"))
REPORT_STYLE_OVERRIDE = _expand(CONFIG.get("reports", {}).get(
    "style_override", "~/.session-browser/report-style.md"))

# Semantic search settings. The model name is stored alongside each vector, so changing
# it triggers a clean re-embed rather than mixing incompatible vectors; "auto" lets
# semsearch.py pick the fastest backend available on this machine.
EMBED_MODEL = CONFIG.get("embeddings", {}).get("model", "all-MiniLM-L6-v2")
EMBED_BACKEND = CONFIG.get("embeddings", {}).get("backend", "auto")

# How to present money. Under a flat subscription no session is billed individually, so
# the dollar figure computed from token counts is only a list-price equivalent — the UI
# shows it with "≈". `mode = "api"` (pay per request) is the one case where it is real.
BILLING = CONFIG.get("billing", {"mode": "subscription", "plan": "subscription", "monthly_usd": None})
BILLING_MODE = BILLING.get("mode", "subscription")
COST_IS_NOTIONAL = BILLING_MODE != "api"

# Per-model prices, committed with the repo rather than configured: they are facts about
# the vendors, not a user preference.
PRICING_PATH = _REPO_ROOT / "pricing.json"

# Title given to every headless `opencode run` the enricher spawns. Shared with
# the OpenCode source adapter, which must never index our own enrichment runs
# (they normally never persist — OPENCODE_DB=:memory: — but a title is the one
# field we control end to end, so it is the belt-and-braces marker).
OPENCODE_ENRICHMENT_TITLE = "session-browser-enrichment"


def ensure_dirs() -> None:
    """Create the runtime directories if missing (safe to call repeatedly).

    Only the three the tool writes to unconditionally: the registry's folder, the facets
    folder and the log folder. Deliberately not the reasoning archive — that one may be
    disabled or point at external storage, and the extractor creates it when it runs.
    Called at start-up by the watcher, the hooks and the installer; idempotent, so no
    caller has to check first.
    """
    for d in (_NEW_DB.parent, FACETS_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)


def source_config(name: str) -> dict[str, Any]:
    """The `[sources.<name>]` table (e.g. "claude", "copilot"), or {} if there is none.

    Callers must treat every key as optional — see sources/registry.py, which reads the
    directory keys out of this and falls back to each CLI's documented default.
    """
    return CONFIG.get("sources", {}).get(name, {})


def enabled_sources() -> list[str]:
    """Names of the CLIs to index, in config-file order.

    Only `enabled = true` counts: a source must be switched off explicitly, because the
    layering above means deleting its block just restores the example's defaults.
    A name here is not a promise that the adapter exists or that the CLI is installed —
    sources/registry.py skips unknown names, and `is_available()` decides the rest.
    """
    srcs = CONFIG.get("sources", {})
    return [name for name, c in srcs.items() if c.get("enabled")]
