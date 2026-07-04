#!/usr/bin/env python3
"""Lightweight keyword->topic classifier (no LLM).

Scans a session's first_message + summary for known keywords and assigns up to 3
topics. Useful on its own (with the null provider) and as a fallback alongside
LLM enrichment. Writes topics as a JSON array string on the session row.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import indexer  # noqa: E402

TOPIC_PATTERNS = {
    "python": r"\bpython|fastapi|django|pytest|pydantic\b",
    "javascript": r"\b(javascript|typescript|node|react|vue|npm)\b",
    "database": r"\b(sql|sqlite|postgres|database|schema|migration)\b",
    "testing": r"\b(test|pytest|unittest|coverage|tdd)\b",
    "debugging": r"\b(bug|debug|error|traceback|exception|fix)\b",
    "ci-cd": r"\b(ci|cd|pipeline|github actions|deploy|docker)\b",
    "data": r"\b(data|etl|attribution|analytics|dataframe|pandas)\b",
    "finance": r"\b(stock|portfolio|trading|multibagger|paytm|dividend)\b",
    "mcp": r"\bmcp|model context protocol|tool server\b",
    "agent": r"\b(agent|subagent|orchestrat|workflow)\b",
    "review": r"\b(review|audit|gap|improvement)\b",
    "planning": r"\b(plan|brainstorm|design|spec)\b",
}


def classify(text: str, limit: int = 3) -> list[str]:
    text = (text or "").lower()
    hits = [topic for topic, pat in TOPIC_PATTERNS.items() if re.search(pat, text)]
    return hits[:limit]


def main() -> None:
    conn = indexer.connect()
    rows = conn.execute(
        "SELECT session_id, first_message, summary, title FROM sessions WHERE archived = 0"
    ).fetchall()
    n = 0
    for r in rows:
        text = " ".join(filter(None, [r["title"], r["summary"], r["first_message"]]))
        topics = classify(text)
        if topics:
            conn.execute(
                "UPDATE sessions SET topics = COALESCE(topics, ?) WHERE session_id = ?",
                (json.dumps(topics), r["session_id"]),
            )
            n += 1
    conn.commit()
    conn.close()
    print(f"Classified topics for {n} sessions.")


if __name__ == "__main__":
    main()
