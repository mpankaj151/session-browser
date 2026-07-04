"""Enrichment provider Protocol, factory, and facet parsing.

A provider turns a session's turns into a validated facet dict (summary, topics,
type, outcome). All providers produce the SAME shape; parse_facet_json enforces
the contract regardless of which CLI produced the text.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

REQUIRED_KEYS = {"brief_summary", "goal_categories", "session_type", "outcome"}
_REPO = Path(__file__).resolve().parent.parent


class FacetValidationError(ValueError):
    pass


class EnrichmentProvider(Protocol):
    name: str
    def is_available(self) -> bool: ...
    def summarize(self, turns: list, cli_source: str, model: str = "", cwd: str = "") -> dict: ...


def _finalize_summary(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    if text[-1] in ".!?":
        return text
    # trim back to the last sentence terminator, else append an ellipsis
    cut = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
    if cut >= 10:
        return text[:cut + 1]
    return text + "…"


def parse_facet_json(raw: str, provider_name: str, model: str | None = None) -> dict:
    """Strip code fences / preamble, json.loads, validate, coerce, inject _meta."""
    s = raw.strip()
    s = re.sub(r"^```(?:json)?", "", s).strip()
    s = re.sub(r"```$", "", s).strip()
    brace = s.find("{")
    if brace > 0:
        s = s[brace:]
    try:
        data = json.loads(s)
    except json.JSONDecodeError as e:
        raise FacetValidationError(f"not valid JSON: {e}") from e
    missing = REQUIRED_KEYS - set(data)
    if missing:
        raise FacetValidationError(f"missing keys: {missing}")
    gc = data.get("goal_categories")
    if isinstance(gc, list):
        data["goal_categories"] = {str(k): 1 for k in gc}
    elif not isinstance(gc, dict):
        data["goal_categories"] = {}
    data["key_decisions"] = list(data.get("key_decisions") or [])
    data["files_touched"] = list(data.get("files_touched") or [])
    data["brief_summary"] = _finalize_summary(data.get("brief_summary", ""))
    data["_meta"] = {"provider": provider_name, "model": model,
                     "enriched_at": datetime.now(timezone.utc).isoformat()}
    return data


def render_prompt(turns: list, cli_source: str, model: str, cwd: str,
                  template_path: Path) -> str:
    lines = []
    for t in turns[:60]:
        role = getattr(t, "role", "?").upper()
        content = (getattr(t, "content", "") or "")[:1500]
        if content:
            lines.append(f"**{role}:** {content}")
    transcript = "\n\n".join(lines)
    template = Path(template_path).read_text(encoding="utf-8")
    return (template.replace("{cli_source}", cli_source)
                    .replace("{model}", model or "")
                    .replace("{cwd}", cwd or "")
                    .replace("{transcript}", transcript))


def get_provider(config: dict):
    """Factory: maps [enrichment].provider to a provider instance."""
    name = config.get("enrichment", {}).get("provider", "none")
    if name in ("none", "null", None):
        from .null_provider import NullProvider
        return NullProvider()
    sub = config.get("enrichment", {}).get(name.replace("-", "_"), {})
    if name == "claude-headless":
        from .claude_headless import ClaudeHeadless
        return ClaudeHeadless(sub)
    if name == "copilot-headless":
        from .copilot_headless import CopilotHeadless
        return CopilotHeadless(sub)
    from .null_provider import NullProvider
    return NullProvider()
