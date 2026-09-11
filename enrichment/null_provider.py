"""Metadata-only enrichment — no LLM calls. Keeps the system fully functional
with zero cost. Derives a minimal facet from the turns themselves.

Every other provider in this package runs a coding CLI, which sends the session
to an AI **model** and spends the user's own quota (see docs/GLOSSARY.md, and
enrichment/provider.py for the full picture). This one spends nothing: it just
takes the first thing the user typed as the session's summary. The **facet** — the
JSON record enrichment produces per session — comes out in exactly the same shape
as a real one, so the UI, reports and MCP server cannot tell the difference.

Selected by `[enrichment].provider = "none"` in config.toml, and also used as the
safe fallback when the configured provider name is not recognised (with a warning
— see get_provider in enrichment/provider.py).

Important consequence, and the reason enrichment/provider.py has a SEPARATE
UnavailableProvider: a null facet still counts as enriched, because it writes a
non-empty summary to the session row. That permanently satisfies the staleness
check in scripts/enrich-sessions.py, so a session summarised by this provider is
never revisited by a real one unless the user runs `enrich-sessions.py --force`.
Choosing "none" is a deliberate "I do not want model calls", not a stopgap.
"""
from __future__ import annotations

from datetime import datetime, timezone


class NullProvider:
    """Zero-cost summariser. Satisfies provider.EnrichmentProvider."""

    name = "null"

    def is_available(self) -> bool:
        """Always True — there is nothing to install and nothing to authenticate."""
        return True

    def summarize(self, turns: list, cli_source: str, model: str = "", cwd: str = "",
                  prior: dict | None = None) -> dict:
        """Build a facet from the transcript alone. No subprocess, no network, no spend.

        The summary is the first line of the first thing the user typed, capped at
        160 characters — usually the best one-line description of a session that
        exists without a model reading it.

        `cli_source`, `cwd` and `prior` are accepted and ignored: the signature has
        to match the other providers so scripts/enrich-sessions.py can call any of
        them identically. `prior` in particular is pointless here — there is no
        model to merge an old entry with, and re-running simply recomputes the same
        line. `model` is echoed straight into `_meta` so the record still says which
        model the SESSION ran on even though no model was called to describe it.

        Returns a facet with every journal key present but empty. That is honest:
        the fields a real provider fills (accomplishments, key decisions,
        explorations, open threads) cannot be derived by any rule, so this provider
        leaves them empty rather than guessing. render_journal_markdown() then
        produces '' and no journal artifact is stored at all.
        """
        # `getattr(t, "role", "")` rather than t.role: turns come from whichever
        # adapter parsed the transcript, and a duck-typed object is fair game here.
        first_user = next((t.content for t in turns if getattr(t, "role", "") == "user"), "")
        # First line only: a first message is often a long paste with the real
        # request on line one.
        summary = (first_user or "").strip().split("\n")[0][:160]
        # Mirrors _finalize_summary in provider.py — an unterminated line is a
        # fragment, so mark it as one instead of pretending it is a sentence.
        if summary and summary[-1] not in ".!?":
            summary += "…"
        return {
            "brief_summary": summary or "(no summary)",
            "goal": "",
            "accomplishments": [],
            "explorations": [],
            "open_threads": [],
            "reusability": "",
            # Empty, so _persist in scripts/enrich-sessions.py leaves whatever
            # topics scripts/classify-topics.py derived from keywords in place.
            "goal_categories": {},
            # "unknown" is not in provider.SESSION_TYPES, so anything that routes
            # this through parse_facet_json coerces it to "other". Both spellings
            # mean the same thing to every report: nobody classified this session.
            "session_type": "unknown",
            "outcome": "unknown",
            "key_decisions": [],
            "files_touched": [],
            # Built by hand rather than via parse_facet_json: this facet was never
            # serialised to text, so there is nothing to parse. Same shape, minus
            # the cost keys the headless providers add.
            "_meta": {"provider": self.name, "model": model,
                      "enriched_at": datetime.now(timezone.utc).isoformat()},
        }
