#!/usr/bin/env python3
"""Lightweight keyword->topic classifier (no LLM).

Scans a session's first_message + summary for known keywords and assigns up to 3
topics. Useful on its own (with the null provider) and as a fallback alongside
LLM enrichment. Writes topics as a JSON array string on the session row.

"No LLM" means no AI model is involved and nothing is spent: this is a plain
regular-expression match over text already in the database. (An **LLM**, or large
language model, is the AI system that the coding CLIs talk to; asking one to read
a session is what scripts/enrich-sessions.py does, and it costs the user quota.
See docs/GLOSSARY.md.) So this script is safe to run on every machine, on every
nightly pass, whether or not any coding CLI is installed.

Where it sits in the pipeline: scripts/refresh-all.py runs it third, BEFORE
enrichment, so that every session has at least rough topics even if enrichment
never runs. When enrichment does run afterwards, the model's topics overwrite
these — see _persist in scripts/enrich-sessions.py, which sets the topics column
outright rather than merging.

Reads and writes only `sessions.topics`. Topics themselves are short tags
("python", "testing", "ci-cd") that the UI shows as chips and lets you filter on.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import indexer  # noqa: E402

# topic tag -> the regular expression that awards it. Each pattern is a group of
# alternatives surrounded by word boundaries: `\b(...)\b` means the match must
# start and end at a word edge, so "test" matches "add a test" but not "latest".
#
# NOTE: alternations must be grouped — r"\ba|b\b" binds \b to only the first
# and last branch, letting middle keywords match inside longer words.
# (Concretely: without the parentheses that pattern reads as "\ba" OR "b\b", so a
# middle alternative like "cd" would match inside "cdn" or "abcd".)
#
# The vocabulary is deliberately small and hand-picked — these are the topics this
# user's work actually falls into, not an attempt at a general taxonomy. `\w*` in
# the agent row is a stem match: "orchestrat" plus anything covers "orchestrate",
# "orchestration" and "orchestrator" in one alternative. Case is handled by
# lowercasing the text in classify(), not by a regex flag.
TOPIC_PATTERNS = {
    "python": r"\b(python|fastapi|django|pytest|pydantic)\b",
    "javascript": r"\b(javascript|typescript|node|react|vue|npm)\b",
    "database": r"\b(sql|sqlite|postgres|database|schema|migration)\b",
    "testing": r"\b(test|pytest|unittest|coverage|tdd)\b",
    "debugging": r"\b(bug|debug|error|traceback|exception|fix)\b",
    "ci-cd": r"\b(ci|cd|pipeline|github actions|deploy|docker)\b",
    "data": r"\b(data|etl|attribution|analytics|dataframe|pandas)\b",
    "finance": r"\b(stock|portfolio|trading|multibagger|paytm|dividend)\b",
    "mcp": r"\b(mcp|model context protocol|tool server)\b",
    "agent": r"\b(agent|subagent|orchestrat\w*|workflow)\b",
    "review": r"\b(review|audit|gap|improvement)\b",
    "planning": r"\b(plan|brainstorm|design|spec)\b",
}


def classify(text: str, limit: int = 3) -> list[str]:
    """Topic tags for a blob of text, at most `limit` of them.

    Pure function — no database, no I/O — which is what makes it easy to test and
    reusable elsewhere. Lowercases once, then tries every pattern.

    The cap keeps the UI's topic chips readable. Ties are broken by TOPIC_PATTERNS
    order, not by relevance, because a keyword hit carries no strength: one mention
    of "pytest" scores exactly like twenty. That crudeness is the whole point — the
    real ranking comes from enrichment, which counts how much of the session was
    actually about each topic.
    """
    text = (text or "").lower()
    hits = [topic for topic, pat in TOPIC_PATTERNS.items() if re.search(pat, text)]
    return hits[:limit]


def main() -> None:
    """Classify every visible session that has no topics yet, then commit.

    Side effect: updates `sessions.topics`. Prints how many rows changed.
    Idempotent and safe to re-run — see the COALESCE in the UPDATE below.
    """
    conn = indexer.connect()
    # indexer.VISIBLE = what the user should see: sessions whose transcript is
    # still on disk, plus real sessions whose transcript has since aged out. It
    # deliberately excludes subagent sidechain noise. Composing the named
    # predicate instead of spelling the flag out is enforced by a test.
    rows = conn.execute(
        f"SELECT session_id, first_message, summary, title FROM sessions WHERE {indexer.VISIBLE}"
    ).fetchall()
    n = 0
    for r in rows:
        # All three fields together, skipping the NULL ones. The title and summary
        # are the most informative when present (a model wrote them), but a session
        # that has never been enriched has only its first message to go on.
        text = " ".join(filter(None, [r["title"], r["summary"], r["first_message"]]))
        topics = classify(text)
        if topics:
            # COALESCE(topics, ?) = "keep whatever is already there; use these only
            # if the column is NULL". This script must never overwrite topics that
            # enrichment produced from actually reading the session. SQLite has no
            # array type, so the list is stored as a JSON string: '["python"]'.
            conn.execute(
                "UPDATE sessions SET topics = COALESCE(topics, ?) WHERE session_id = ?",
                (json.dumps(topics), r["session_id"]),
            )
            # Counts rows the UPDATE touched, which is not the same as rows that
            # actually changed — an already-classified session still counts here.
            n += 1
    conn.commit()
    conn.close()
    print(f"Classified topics for {n} sessions.")


if __name__ == "__main__":
    main()
