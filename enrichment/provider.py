"""Enrichment provider Protocol, factory, and facet parsing.

A provider turns a session's turns into a validated facet dict (summary, topics,
type, outcome). All providers produce the SAME shape; parse_facet_json enforces
the contract regardless of which CLI produced the text.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

REQUIRED_KEYS = {"brief_summary", "goal_categories", "session_type", "outcome"}
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import redact as _redact  # noqa: E402


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
    # Journal-grade keys are optional (older facets and the null provider lack
    # them) — coerce to their shape so downstream code never branches on absence.
    for key in ("accomplishments", "explorations", "open_threads"):
        data[key] = [str(x) for x in (data.get(key) or [])]
    data["goal"] = str(data.get("goal") or "").strip()
    data["reusability"] = str(data.get("reusability") or "").strip()
    data["brief_summary"] = _finalize_summary(data.get("brief_summary", ""))
    data["_meta"] = {"provider": provider_name, "model": model,
                     "enriched_at": datetime.now(timezone.utc).isoformat()}
    return data


_JOURNAL_SECTIONS = (
    ("accomplishments", "Accomplishments"),
    ("key_decisions", "Key decisions"),
    ("explorations", "Explorations (not kept)"),
    ("open_threads", "Open threads"),
)


def render_journal_markdown(facet: dict) -> str:
    """Deterministic journal markdown from a facet — the durable per-session
    record surfaced by daily digests and review reports. Empty sections are
    omitted; an all-empty facet yields ''. """
    parts: list[str] = []
    for key, heading in _JOURNAL_SECTIONS:
        items = [str(x).strip() for x in (facet.get(key) or []) if str(x).strip()]
        if items:
            parts.append(f"## {heading}\n" + "\n".join(f"- {i}" for i in items))
    reuse = str(facet.get("reusability") or "").strip()
    if reuse:
        parts.append(f"## Reusability\n{reuse}")
    return "\n\n".join(parts)


def render_prior_context(prior: dict) -> str:
    """The incremental re-enrichment block: shows the previous journal so the
    model UPDATES it from the new turns instead of starting over."""
    lines = ["", "This session was previously journaled. Prior entry:",
             f"- Summary: {prior.get('brief_summary', '')}"]
    journal = render_journal_markdown(prior)
    if journal:
        lines.append(journal)
    lines.append(
        "The transcript below contains only turns SINCE that entry. Merge: keep "
        "prior facts that still hold, integrate what is new, and move resolved "
        "open threads into accomplishments. Return the full updated JSON.")
    return "\n".join(lines) + "\n"


def render_prompt(turns: list, cli_source: str, model: str, cwd: str,
                  template_path: Path, prior: dict | None = None) -> str:
    lines = []
    for t in turns[:60]:
        role = getattr(t, "role", "?").upper()
        # Redact BEFORE the LLM sees the transcript: a summary can't echo a
        # credential it never received, and summarization doesn't need the value.
        content = _redact.redact((getattr(t, "content", "") or "")[:1500])
        if content:
            lines.append(f"**{role}:** {content}")
    transcript = "\n\n".join(lines)
    template = Path(template_path).read_text(encoding="utf-8")
    return (template.replace("{cli_source}", cli_source)
                    .replace("{model}", model or "")
                    .replace("{cwd}", cwd or "")
                    .replace("{prior_context}", render_prior_context(prior) if prior else "")
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
