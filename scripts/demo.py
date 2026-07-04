#!/usr/bin/env python3
"""Demo mode — see the full UI in 30 seconds with zero of your own sessions.

Seeds a throwaway database with synthetic sessions (no personal data), then
launches the web UI pointed at it via SB_DB. Nothing touches your real registry.

    sb demo          # or: .venv/bin/python scripts/demo.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from importlib import util as _ilu
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

# Point the whole stack at a temp DB BEFORE importing anything that resolves
# paths. mkdtemp gives a fresh private (0700) dir — a fixed /tmp name would
# collide (and race) with other users on a shared host.
_DB = Path(tempfile.mkdtemp(prefix="session-browser-demo-")) / "demo.db"
os.environ["SB_DB"] = str(_DB)

import indexer  # noqa: E402


def _load(name: str):
    spec = _ilu.spec_from_file_location(name.replace("-", "_"), _REPO / "scripts" / f"{name}.py")
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")


# (title, source, model, folder, turns, summary, topics, type, outcome, tokens, cost, days_ago)
_DEMO = [
    ("Add JWT auth to the FastAPI service", "claude", "claude-opus-4-8", "api-server", 42,
     "Implemented OAuth2 password flow with JWT access + refresh tokens, pydantic settings for secrets, and pytest coverage for expiry and role checks.",
     ["python", "security", "api"], "feature", "completed", 3_600_000, 4.31, 0),
    ("Fix flaky Playwright checkout tests", "claude", "claude-sonnet-4-6", "webshop", 27,
     "Root-caused the flake to a race between cart hydration and the payment iframe; replaced sleeps with locator assertions and stubbed the gateway.",
     ["testing", "debugging"], "bugfix", "completed", 1_300_000, 0.87, 0),
    ("Migrate the blog to Astro content collections", "copilot", "gpt-5.4", "blog", 33,
     "Moved 120 markdown posts into typed content collections with zod frontmatter schemas and view transitions between pages.",
     ["astro", "frontend"], "refactor", "completed", 1_100_000, 1.42, 1),
    ("Design the payment webhook retry strategy", "codex", "gpt-5.5", "api-server", 18,
     "Compared outbox-table polling vs queue-backed retries; chose the outbox pattern for exactly-once semantics and sketched the migration.",
     ["architecture", "planning"], "planning", "partial", 2_100_000, 2.05, 1),
    ("Set up CI pipeline with matrix builds", "copilot", "claude-haiku-4-5", "webshop", 15,
     "GitHub Actions workflow with a python/node matrix, dependency caching, and a release job gated on the full test suite.",
     ["ci-cd", "tooling"], "feature", "completed", 217_000, 0.09, 4),
    ("Profile and fix the slow dashboard query", "claude", "claude-opus-4-8", "analytics", 51,
     "EXPLAIN ANALYZE showed a seq scan on events; added a partial covering index and an hourly materialized view. P95 4.2s -> 80ms.",
     ["postgres", "performance"], "bugfix", "completed", 6_900_000, 6.78, 5),
    ("Brainstorm plugin architecture for the CLI", "codex", "gpt-5.5", "devtool", 22,
     "Explored entry-point discovery vs a manifest registry; settled on a hybrid with lazy loading and a capability handshake.",
     ["architecture", "python"], "planning", "completed", 1_300_000, 1.90, 12),
    ("Refactor the auth module for testability", "claude", "claude-sonnet-4-6", "api-server", 30,
     "Extracted the token service behind a protocol, injected the clock, and removed the global session singleton so tests can run in parallel.",
     ["python", "testing"], "refactor", "completed", 2_400_000, 1.55, 20),
]


def seed():
    from sources.base import SessionHeader
    conn = indexer.connect(str(_DB))
    _load("migrate-db").migrate(conn)
    now = datetime.now(timezone.utc)
    for i, (title, src, model, folder, turns, summary, topics, stype, outcome, tokens, cost, ago) in enumerate(_DEMO):
        ts = now - timedelta(days=ago, hours=i)
        sid = f"demo-{i:04d}-0000-0000-000000000000"
        h = SessionHeader(session_id=sid, cli_source=src, project_path=f"/demo/{folder}",
                          cwd=f"/demo/{folder}", folder_name=folder, start_time=_iso(ts),
                          last_activity=_iso(ts), first_message=summary, turn_count=turns,
                          title=title, model_used=model)
        indexer.upsert(h, conn=conn)
        import json as _json
        conn.execute(
            "UPDATE sessions SET summary=?, topics=?, session_type=?, outcome=?, "
            "input_tokens=?, output_tokens=?, cache_read_tokens=?, cache_write_tokens=?, "
            "cost_usd=? WHERE session_id=?",
            (summary, _json.dumps(topics), stype, outcome,
             int(tokens * 0.2), int(tokens * 0.05), int(tokens * 0.72), int(tokens * 0.03),
             cost, sid))
    conn.commit()
    conn.close()


def main():
    seed()
    print(f"Demo DB seeded ({len(_DEMO)} synthetic sessions) at {_DB}")
    print("Starting the UI at http://127.0.0.1:7655  (Ctrl-C to stop; your real data is untouched)")
    # Launch the Flask app in-process; SB_DB is already set so it uses the demo DB.
    sys.path.insert(0, str(_REPO / "session-ui"))
    import app as flask_app
    flask_app.sbconfig.ensure_dirs()
    flask_app.app.run(host=flask_app.HOST, port=flask_app.PORT, debug=False, threaded=True)


if __name__ == "__main__":
    main()
