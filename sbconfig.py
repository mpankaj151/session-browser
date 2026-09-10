"""Central config + path resolution for the Session Browser.

Loads config.toml (falling back to config.toml.example), expands ~ in paths, and
exposes the canonical filesystem locations. Importable from every script and the
Flask app so there is one source of truth.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

_REPO_ROOT = Path(__file__).resolve().parent


def _expand(p: str) -> Path:
    return Path(os.path.expanduser(p)).resolve()


def _deep_merge(base: dict, override: dict) -> dict:
    """base with override's keys applied; nested tables merge, everything else replaces."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_toml(path: Path) -> dict[str, Any]:
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def config_path() -> Path:
    """The per-machine override file: $SB_CONFIG, else <repo>/config.toml."""
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
    """
    defaults_path = Path(defaults) if defaults else _REPO_ROOT / "config.toml.example"
    cfg = _load_toml(defaults_path) if defaults_path.exists() else {}
    override_path = Path(os.path.expanduser(str(path))) if path else config_path()
    if override_path.exists():
        cfg = _deep_merge(cfg, _load_toml(override_path))
    return cfg


CONFIG = load_config()
CONFIG_PATH = config_path()

REPO_ROOT = _REPO_ROOT

# --- Canonical paths (dual DB-path resolution: new location, legacy fallback) --
# .get() with defaults everywhere: a hand-edited config.toml missing a key must
# degrade to the documented default, not KeyError every tool at import time.
_PATHS = CONFIG.get("paths", {})
_NEW_DB = _expand(_PATHS.get("db", "~/.session-browser/registry.db"))
_OLD_DB = Path.home() / ".claude" / "session-registry.db"
# SB_DB env override wins over everything — used by `sb demo` to point the whole
# stack at a throwaway seeded database without touching the real registry.
_ENV_DB = os.environ.get("SB_DB")
if _ENV_DB:
    DB_PATH = Path(os.path.expanduser(_ENV_DB))
else:
    DB_PATH = _NEW_DB if _NEW_DB.exists() else (_OLD_DB if _OLD_DB.exists() else _NEW_DB)

FACETS_DIR = _expand(_PATHS.get("facets_dir", "~/.session-browser/facets"))
HOOK_STATE = _expand(_PATHS.get("hook_state", "~/.session-browser/.hook-state.json"))
LOG_DIR = _expand(_PATHS.get("log_dir", "~/.session-browser/logs"))

REASONING_ENABLED = bool(CONFIG.get("reasoning", {}).get("enabled", True))
REASONING_ARCHIVE = _expand(CONFIG.get("reasoning", {}).get("archive_dir", "~/claude-reasoning-archive"))

DAILY_DIR = _expand(CONFIG.get("digest", {}).get("daily_dir", "~/.session-browser/daily-logs"))

REPORTS_DIR = _expand(CONFIG.get("reports", {}).get("out_dir", "~/.session-browser/reports"))
REPORT_STYLE_OVERRIDE = _expand(CONFIG.get("reports", {}).get(
    "style_override", "~/.session-browser/report-style.md"))

EMBED_MODEL = CONFIG.get("embeddings", {}).get("model", "all-MiniLM-L6-v2")
EMBED_BACKEND = CONFIG.get("embeddings", {}).get("backend", "auto")

BILLING = CONFIG.get("billing", {"mode": "subscription", "plan": "subscription", "monthly_usd": None})
BILLING_MODE = BILLING.get("mode", "subscription")
COST_IS_NOTIONAL = BILLING_MODE != "api"

PRICING_PATH = _REPO_ROOT / "pricing.json"

# Title given to every headless `opencode run` the enricher spawns. Shared with
# the OpenCode source adapter, which must never index our own enrichment runs
# (they normally never persist — OPENCODE_DB=:memory: — but a title is the one
# field we control end to end, so it is the belt-and-braces marker).
OPENCODE_ENRICHMENT_TITLE = "session-browser-enrichment"


def ensure_dirs() -> None:
    """Create the runtime directories if missing (safe to call repeatedly)."""
    for d in (_NEW_DB.parent, FACETS_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)


def source_config(name: str) -> dict[str, Any]:
    return CONFIG.get("sources", {}).get(name, {})


def enabled_sources() -> list[str]:
    srcs = CONFIG.get("sources", {})
    return [name for name, c in srcs.items() if c.get("enabled")]
