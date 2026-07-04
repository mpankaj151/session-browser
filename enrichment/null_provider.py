"""Metadata-only enrichment — no LLM calls. Keeps the system fully functional
with zero cost. Derives a minimal facet from the turns themselves."""
from __future__ import annotations

from datetime import datetime, timezone


class NullProvider:
    name = "null"

    def is_available(self) -> bool:
        return True

    def summarize(self, turns: list, cli_source: str, model: str = "", cwd: str = "") -> dict:
        first_user = next((t.content for t in turns if getattr(t, "role", "") == "user"), "")
        summary = (first_user or "").strip().split("\n")[0][:160]
        if summary and summary[-1] not in ".!?":
            summary += "…"
        return {
            "brief_summary": summary or "(no summary)",
            "goal_categories": {},
            "session_type": "unknown",
            "outcome": "unknown",
            "key_decisions": [],
            "files_touched": [],
            "_meta": {"provider": self.name, "model": model,
                      "enriched_at": datetime.now(timezone.utc).isoformat()},
        }
