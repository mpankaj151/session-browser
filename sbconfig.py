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


def load_config() -> dict[str, Any]:
    """Load config.toml (or the .example if no real config exists yet)."""
    cfg_path = _REPO_ROOT / "config.toml"
    if not cfg_path.exists():
        cfg_path = _REPO_ROOT / "config.toml.example"
    with open(cfg_path, "rb") as fh:
        return tomllib.load(fh)


CONFIG = load_config()

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


def ensure_dirs() -> None:
    """Create the runtime directories if missing (safe to call repeatedly)."""
    for d in (_NEW_DB.parent, FACETS_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)


def source_config(name: str) -> dict[str, Any]:
    return CONFIG.get("sources", {}).get(name, {})


def enabled_sources() -> list[str]:
    srcs = CONFIG.get("sources", {})
    return [name for name, c in srcs.items() if c.get("enabled")]
