#!/usr/bin/env python3
"""Demo mode — see the full UI in 30 seconds with zero of your own sessions.

Seeds a throwaway database with synthetic sessions (no personal data), then
launches the web UI pointed at it. Nothing touches your real registry, archive
or CLI directories: the demo writes its own config override (SB_CONFIG) so the
reasoning archive and the Claude projects dir live in the same temp folder —
which is also what lets one archived demo session be genuinely restorable.

    sb demo                  # or: .venv/bin/python scripts/demo.py
    sb demo --port 7699      # another port (default: [ui].port, 7655)

Vocabulary (docs/GLOSSARY.md): a *session* is one conversation with a coding CLI; a
*transcript* is the file the CLI writes while it happens; *enrichment* is the summary a
model writes about a finished session; *archived* means the transcript is gone but the row
remains; *restore* means copying a saved transcript back so the session is live again.

HOW THE SANDBOX WORKS. Two environment variables, both set at import time, before any
module that resolves a path is imported:
  * SB_DB      the registry file to use — a fresh demo.db in the temp directory.
  * SB_CONFIG  a config.toml written on the fly. sbconfig layers it over
               config.toml.example, so it only has to override two settings: the reasoning
               archive directory and the Claude projects directory. Everything else (ports,
               providers, thresholds) comes from the shipped defaults.
That second override is the load-bearing one. It is what lets this demo show a REAL archive
lifecycle instead of a screenshot of one: the archived session below gets a genuine
transcript written into the temp tree and a genuine byte-for-byte copy in the temp
reasoning archive, so the UI's Restore button runs the same restore.py code path as on real
data, finds the copy, writes it back into the temp Claude projects directory and re-indexes
the row as live. Because both directories are inside the temp tree, nothing in ~/.claude,
~/.session-browser or ~/claude-reasoning-archive is read or written at any point.

The demo seeds one row per entry in the two tables below, starts the real Flask app
in-process (not a mock) and serves it. Nothing cleans up the temp directory: it is a
mkdtemp under the system temp dir, and leaving it lets you poke at demo.db afterwards.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from importlib import util as _ilu
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

# Point the whole stack at a temp tree BEFORE importing anything that resolves
# paths. mkdtemp gives a fresh private (0700) dir — a fixed /tmp name would
# collide (and race) with other users on a shared host.
_TMP = Path(tempfile.mkdtemp(prefix="session-browser-demo-"))
_DB = _TMP / "demo.db"
# Arguments are parsed up here, at import time, because the port has to be in the config
# file before session-ui/app.py is imported and reads it.
_ARGS = argparse.ArgumentParser(description="Session Browser demo on synthetic data")
_ARGS.add_argument("--port", type=int, default=None, help="UI port (default: [ui].port from config)")
_OPTS = _ARGS.parse_args()
# A minimal config.toml. sbconfig layers it over config.toml.example, so only the two
# directories that MUST be redirected are named — everything else inherits the defaults.
# The archive dir is where transcript copies are kept; the projects dir is where the Claude
# adapter looks for transcripts. Together they make the archived session restorable inside
# the sandbox.
_override = [
    "[reasoning]", f'archive_dir = "{_TMP / "archive"}"',
    "[sources.claude]", f'projects_dir = "{_TMP / "claude" / "projects"}"',
]
if _OPTS.port:
    _override += ["[ui]", f"port = {_OPTS.port}"]
(_TMP / "config.toml").write_text("\n".join(_override) + "\n")
# Set BEFORE the first project import below: sbconfig reads both at import time, and every
# other module takes its paths from sbconfig.
os.environ["SB_DB"] = str(_DB)
os.environ["SB_CONFIG"] = str(_TMP / "config.toml")

import indexer  # noqa: E402


def _load(name: str):
    """Import a sibling script by file name, e.g. _load("migrate-db").

    Needed because these file names contain hyphens, which are not legal in a Python module
    name, so a plain `import` cannot reach them.
    """
    spec = _ilu.spec_from_file_location(name.replace("-", "_"), _REPO / "scripts" / f"{name}.py")
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _iso(dt: datetime) -> str:
    """Format a timestamp the way Claude Code writes them, e.g. 2026-09-11T08:12:04.000Z.

    The whole tool sorts on these strings directly, so the fixed width and the trailing Z
    matter: they make lexical order equal chronological order.
    """
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")


# The live sessions. Invented, not sampled from anyone's history — the point is to show the
# UI (four different CLIs, a spread of dates, topics, costs and outcomes) with no personal
# data anywhere. `tokens` is the session total; seed() splits it into the four token
# buckets in realistic proportions. `days_ago` positions the row on the timeline.
# (title, source, model, folder, turns, summary, topics, type, outcome, tokens, cost, days_ago)
_DEMO = [
    ("Add JWT auth to the FastAPI service", "claude", "claude-opus-5", "api-server", 42,
     "Implemented OAuth2 password flow with JWT access + refresh tokens, pydantic settings for secrets, and pytest coverage for expiry and role checks.",
     ["python", "security", "api"], "feature", "completed", 3_600_000, 4.31, 0),
    ("Fix flaky Playwright checkout tests", "claude", "claude-sonnet-5", "webshop", 27,
     "Root-caused the flake to a race between cart hydration and the payment iframe; replaced sleeps with locator assertions and stubbed the gateway.",
     ["testing", "debugging"], "bugfix", "completed", 1_300_000, 0.87, 0),
    ("Migrate the blog to Astro content collections", "copilot", "gpt-5.4", "blog", 33,
     "Moved 120 markdown posts into typed content collections with zod frontmatter schemas and view transitions between pages.",
     ["astro", "frontend"], "refactor", "completed", 1_100_000, 1.42, 1),
    ("Design the payment webhook retry strategy", "codex", "gpt-5.5", "api-server", 18,
     "Compared outbox-table polling vs queue-backed retries; chose the outbox pattern for exactly-once semantics and sketched the migration.",
     ["architecture", "planning"], "planning", "partial", 2_100_000, 2.05, 1),
    ("Wire the GLM gateway into the eval harness", "opencode", "zai/glm-5.1", "eval-harness", 24,
     "Added a provider adapter for the team's GLM gateway, normalised its token accounting, and made the eval harness pick models per suite.",
     ["python", "tooling"], "feature", "completed", 910_000, 0.31, 2),
    ("Set up CI pipeline with matrix builds", "copilot", "claude-haiku-4.5", "webshop", 15,
     "GitHub Actions workflow with a python/node matrix, dependency caching, and a release job gated on the full test suite.",
     ["ci-cd", "tooling"], "feature", "completed", 217_000, 0.09, 4),
    ("Profile and fix the slow dashboard query", "claude", "claude-opus-5", "analytics", 51,
     "EXPLAIN ANALYZE showed a seq scan on events; added a partial covering index and an hourly materialized view. P95 4.2s -> 80ms.",
     ["postgres", "performance"], "bugfix", "completed", 6_900_000, 6.78, 5),
    ("Brainstorm plugin architecture for the CLI", "codex", "gpt-5.5", "devtool", 22,
     "Explored entry-point discovery vs a manifest registry; settled on a hybrid with lazy loading and a capability handshake.",
     ["architecture", "python"], "planning", "completed", 1_300_000, 1.90, 12),
    ("Refactor the auth module for testability", "claude", "claude-sonnet-5", "api-server", 30,
     "Extracted the token service behind a protocol, injected the clock, and removed the global session singleton so tests can run in parallel.",
     ["python", "testing"], "refactor", "completed", 2_400_000, 1.55, 20),
]


def seed():
    """Create the demo registry and fill it with the synthetic rows.

    Writes only to the temp tree. Goes through the real indexer.upsert() so the rows have
    exactly the shape the real pipeline produces, then fills the enrichment and token
    columns directly with an UPDATE — those are normally written by later passes (the LLM
    enrichment and the cost computation), and the demo has neither.
    """
    from sources.base import SessionHeader
    conn = indexer.connect(str(_DB))
    _load("migrate-db").migrate(conn)
    now = datetime.now(timezone.utc)
    for i, (title, src, model, folder, turns, summary, topics, stype, outcome, tokens, cost, ago) in enumerate(_DEMO):
        # `hours=i` spreads rows that share a days_ago value across the day, so the list is
        # never a block of identical timestamps.
        ts = now - timedelta(days=ago, hours=i)
        # Deliberately shaped like a UUID but obviously fake, so a demo row can never be
        # mistaken for a real session id if a demo database is ever opened by accident.
        sid = f"demo-{i:04d}-0000-0000-000000000000"
        h = SessionHeader(session_id=sid, cli_source=src, project_path=f"/demo/{folder}",
                          cwd=f"/demo/{folder}", folder_name=folder, start_time=_iso(ts),
                          last_activity=_iso(ts), first_message=summary, turn_count=turns,
                          title=title, model_used=model)
        indexer.upsert(h, conn=conn)
        import json as _json
        # The columns a real install fills later: enrichment writes summary/topics/type/
        # outcome, compute-costs writes the token counts and the dollar figure. The split
        # 20/5/72/3 (input / output / cache read / cache write) matches what a long real
        # session looks like — most of the volume is cached history being re-read cheaply.
        conn.execute(
            "UPDATE sessions SET summary=?, topics=?, session_type=?, outcome=?, "
            "input_tokens=?, output_tokens=?, cache_read_tokens=?, cache_write_tokens=?, "
            "cost_usd=? WHERE session_id=?",
            (summary, _json.dumps(topics), stype, outcome,
             int(tokens * 0.2), int(tokens * 0.05), int(tokens * 0.72), int(tokens * 0.03),
             cost, sid))
    conn.commit()
    _seed_archived(conn, now)
    conn.commit()
    conn.close()


# Two sessions whose transcripts "aged out" (Claude Code's cleanupPeriodDays):
# one still has a raw copy in the reasoning archive and can be restored with
# one click; the other never got one and is metadata-only.
# ("Aged out" = the CLI deleted the transcript itself; Claude Code removes files older than
# its cleanupPeriodDays, 30 by default. The row survives, which is the whole reason this
# tool exists.) The trailing flag in each tuple is has_copy — whether a copy reached the
# vault in time. The pair is deliberate: the Archived tab must show BOTH a restorable row
# and an unrestorable one, so what Restore can and cannot do is visible, not implied.
# Their days_ago values (41, 48) are past the 30-day cleanup window on purpose.
_ARCHIVED = [
    ("Prototype the rate limiter middleware", "claude", "claude-opus-5", "api-server", 19,
     "Sketched a token-bucket limiter as ASGI middleware with per-route budgets and a Redis-backed store; left the tests for later.",
     ["python", "api"], "feature", "partial", 1_900_000, 1.62, 41, True),
    ("Spike: swap the search index to SQLite FTS5", "codex", "gpt-5.4", "devtool", 11,
     "Measured FTS5 against the old grep-based search on 30k docs; 40x faster, but the tokenizer needs a custom stemmer.",
     ["research", "performance"], "research", "completed", 640_000, 0.55, 48, False),
]


def _seed_archived(conn, now: datetime) -> None:
    """Seed the two archived rows, one of them genuinely restorable.

    For the restorable one this writes a real (tiny) transcript and hands it to
    reasoning.archive_raw(), the same function the nightly pipeline uses — so the vault
    ends up holding a real copy at a real path, and the UI's Restore button exercises the
    production code path rather than a stub.
    """
    import json as _json
    from sources.base import SessionHeader
    import reasoning
    # Claude Code names a project directory after the working directory with the path
    # separators turned into hyphens, so /demo/api-server becomes "-demo-api-server". The
    # demo reproduces that exactly, because restore.py computes the destination the same
    # way and would otherwise put the file back where nothing looks for it.
    projects = _TMP / "claude" / "projects" / "-demo-api-server"
    projects.mkdir(parents=True, exist_ok=True)
    for i, (title, src, model, folder, turns, summary, topics, stype, outcome, tokens, cost, ago, has_copy) in enumerate(_ARCHIVED):
        ts = now - timedelta(days=ago, hours=i)
        sid = f"demo-aged-{i:04d}-0000-000000000000"
        # Only the claude row gets a path inside the sandbox's projects tree: restore is
        # decided per row by the owning adapter, which refuses to write outside its own
        # tree. The codex row keeps an invented path and so is honestly not restorable.
        project_path = str(projects) if src == "claude" else f"/demo/{folder}"
        h = SessionHeader(session_id=sid, cli_source=src, project_path=project_path,
                          cwd=f"/demo/{folder}", folder_name=folder, start_time=_iso(ts),
                          last_activity=_iso(ts), first_message=summary, turn_count=turns,
                          title=title, model_used=model)
        indexer.upsert(h, conn=conn)
        conn.execute(
            "UPDATE sessions SET summary=?, topics=?, session_type=?, outcome=?, "
            "input_tokens=?, output_tokens=?, cache_read_tokens=?, cache_write_tokens=?, "
            "cost_usd=? WHERE session_id=?",
            (summary, _json.dumps(topics), stype, outcome,
             int(tokens * 0.2), int(tokens * 0.05), int(tokens * 0.72), int(tokens * 0.03),
             cost, sid))
        # TRANSCRIPT_MISSING (rather than NOT_A_SESSION) is what keeps these rows visible in
        # the Archived tab and counted in the usage totals — they were real conversations.
        indexer.archive(sid, indexer.TRANSCRIPT_MISSING, conn=conn)
        if has_copy:
            # the nightly refresh copied this transcript into the vault before it aged out
            # A minimal but genuine Claude Code transcript: one JSON object per line, a user
            # message followed by an assistant reply, with the keys the adapter reads
            # (type, uuid, sessionId, cwd, timestamp, message). Small on purpose — it only
            # has to parse and re-index after a restore.
            lines = [
                {"type": "user", "uuid": "u1", "sessionId": sid, "cwd": f"/demo/{folder}", "timestamp": _iso(ts),
                 "message": {"role": "user", "content": title}},
                {"type": "assistant", "uuid": "a1", "sessionId": sid, "timestamp": _iso(ts),
                 "message": {"role": "assistant", "model": model, "content": [{"type": "text", "text": summary}]}},
            ]
            # Written OUTSIDE the projects tree and never moved there: the session must
            # look aged-out (no transcript where the CLI would keep it) while a copy exists
            # in the vault. archive_raw() copies it into <archive>/raw/YYYY/MM/, which is
            # exactly where restore.py will go looking for it.
            transcript = _TMP / f"{sid}.jsonl"
            transcript.write_text("".join(_json.dumps(ln) + "\n" for ln in lines))
            reasoning.archive_raw(transcript, {"session_id": sid, "last_activity": _iso(ts)})


def main():
    """Seed the sandbox and serve the real UI against it until Ctrl-C."""
    seed()
    print(f"Demo DB seeded ({len(_DEMO)} live + {len(_ARCHIVED)} archived synthetic sessions) at {_DB}")
    # Launch the Flask app in-process; SB_DB is already set so it uses the demo DB.
    # Imported here, not at the top of the file: importing it resolves paths from SB_DB and
    # SB_CONFIG, which are only set once the sandbox above exists.
    sys.path.insert(0, str(_REPO / "session-ui"))
    import app as flask_app
    print(f"Starting the UI at http://{flask_app.HOST}:{flask_app.PORT}  (Ctrl-C to stop; your real data is untouched)")
    flask_app.sbconfig.ensure_dirs()
    flask_app.app.run(host=flask_app.HOST, port=flask_app.PORT, debug=False, threaded=True)


if __name__ == "__main__":
    main()
