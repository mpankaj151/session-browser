#!/usr/bin/env python3
"""Work-journal tests: journal-grade enrichment, stale re-enrichment selection,
incremental slicing, and persistence (snapshots + journal artifacts).

Runs standalone (no pytest):

    .venv/bin/python tests/test_work_journal.py

Fully isolated: DB tests run against a temp database built by migrate(); the
suite never touches ~/.session-browser.
"""
from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
from importlib import util as _ilu
from pathlib import Path
from types import SimpleNamespace

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import indexer
import sbconfig
from enrichment.provider import (parse_facet_json, render_journal_markdown,
                                 render_prior_context, render_prompt)
from sources.base import SessionHeader


def _load_script(name: str):
    spec = _ilu.spec_from_file_location(name.replace("-", "_"), _REPO / "scripts" / f"{name}.py")
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _temp_db() -> sqlite3.Connection:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    conn = indexer.connect(tmp.name)
    _load_script("migrate-db").migrate(conn)
    return conn


def _header(sid="__wj__", **kw) -> SessionHeader:
    base = dict(session_id=sid, cli_source="claude", project_path="/proj/x",
                cwd="/x", folder_name="x", start_time="2026-01-01T00:00:00.000Z",
                last_activity="2026-01-01T00:00:00.000Z", first_message="hi",
                turn_count=1, title="T")
    base.update(kw)
    return SessionHeader(**base)


_FULL_FACET_RAW = json.dumps({
    "brief_summary": "Built the retry layer. It now survives S3 blips.",
    "goal": "Make uploads resilient",
    "accomplishments": ["Added retry with backoff to the S3 uploader"],
    "key_decisions": ["tenacity over hand-rolled loop — battle-tested jitter"],
    "explorations": ["urllib3 Retry — rejected, no async support"],
    "open_threads": ["backfill metrics for retry counts"],
    "reusability": "The backoff decorator generalizes to any boto call.",
    "goal_categories": {"python": 2, "s3": 1},
    "session_type": "feature",
    "outcome": "completed",
    "files_touched": ["uploader.py"],
})


# --- facet parsing ------------------------------------------------------------
def test_facet_json_coerces_journal_keys():
    """New journal keys are optional but always present + typed after parsing."""
    facet = parse_facet_json(_FULL_FACET_RAW, "test")
    assert facet["accomplishments"] == ["Added retry with backoff to the S3 uploader"]
    assert facet["open_threads"] == ["backfill metrics for retry counts"]
    assert facet["goal"] == "Make uploads resilient"
    # legacy facet without the new keys still validates, with empty defaults
    legacy = parse_facet_json(json.dumps({
        "brief_summary": "Did a thing.", "goal_categories": {},
        "session_type": "ops", "outcome": "completed"}), "test")
    assert legacy["accomplishments"] == [] and legacy["explorations"] == []
    assert legacy["open_threads"] == [] and legacy["reusability"] == ""
    assert legacy["goal"] == ""
    print("  ok  parse_facet_json coerces journal keys (new + legacy)")


def test_journal_markdown_sections():
    facet = parse_facet_json(_FULL_FACET_RAW, "test")
    md = render_journal_markdown(facet)
    for heading in ("## Accomplishments", "## Key decisions",
                    "## Explorations (not kept)", "## Open threads", "## Reusability"):
        assert heading in md, f"missing {heading}"
    assert "- Added retry with backoff" in md
    # empty facet -> no journal at all (never store an empty shell)
    empty = parse_facet_json(json.dumps({
        "brief_summary": "x.", "goal_categories": {},
        "session_type": "other", "outcome": "unknown"}), "test")
    assert render_journal_markdown(empty) == ""
    print("  ok  render_journal_markdown sections + empty-facet elision")


def test_prompt_prior_context_block():
    """{prior_context} renders the previous journal for re-enrichment and
    disappears entirely on first enrichment."""
    turns = [SimpleNamespace(role="user", content="hello world")]
    template = _REPO / "prompts" / "summarize-multi-source.md"
    first = render_prompt(turns, "claude", "m", "/x", template)
    assert "{prior_context}" not in first and "previously journaled" not in first
    prior = parse_facet_json(_FULL_FACET_RAW, "test")
    again = render_prompt(turns, "claude", "m", "/x", template, prior=prior)
    assert "previously journaled" in again
    assert "Built the retry layer" in again
    assert "only turns SINCE" in render_prior_context(prior)
    print("  ok  render_prompt prior-context block (present on update, absent on first)")


# --- stale selection ----------------------------------------------------------
def test_stale_predicate_selects_resumed_sessions():
    es = _load_script("enrich-sessions")
    conn = _temp_db()
    try:
        rows = [
            # never enriched -> eligible
            ("new", "2026-06-01T10:00:00.000Z", None, None),
            # enriched after last activity (canonical format) -> NOT eligible
            ("fresh", "2026-06-01T10:00:00.000Z", "done.", "2026-06-01T10:05:00.000Z"),
            # resumed after enrichment -> eligible
            ("resumed", "2026-06-02T09:00:00.000Z", "done.", "2026-06-01T10:05:00.000Z"),
            # legacy CURRENT_TIMESTAMP spelling, enriched after activity -> NOT
            # eligible (normalization makes the comparison semantically right)
            ("legacy_fresh", "2026-06-01T10:00:00.000Z", "done.", "2026-06-01 10:05:00"),
            # legacy spelling but resumed since -> eligible
            ("legacy_resumed", "2026-06-02T09:00:00.000Z", "done.", "2026-06-01 10:05:00"),
        ]
        for sid, last, summary, enriched in rows:
            indexer.upsert(_header(sid=sid, last_activity=last), conn=conn)
            conn.execute("UPDATE sessions SET summary=?, enriched_at=? WHERE session_id=?",
                         (summary, enriched, sid))
        got = {r["session_id"] for r in conn.execute(
            f"SELECT session_id FROM sessions WHERE archived=0 AND {es.STALE_PREDICATE}")}
        assert got == {"new", "resumed", "legacy_resumed"}, got
    finally:
        conn.close()
    print("  ok  stale predicate: new + resumed eligible (either spelling), fresh skipped")


def test_select_session_fast_path_honors_staleness():
    """The hook fires --session on every SessionEnd; a re-fire with no new
    activity must select nothing (cost $0), while --force always selects."""
    es = _load_script("enrich-sessions")
    conn = _temp_db()
    try:
        indexer.upsert(_header(sid="h1", last_activity="2026-06-01T10:00:00.000Z"), conn=conn)
        conn.execute("UPDATE sessions SET summary='done.', enriched_at=? WHERE session_id='h1'",
                     ("2026-06-01T10:05:00.000Z",))
        assert es._select_sessions(conn, "h1", force=False) == []
        assert len(es._select_sessions(conn, "h1", force=True)) == 1
        # resumed since enrichment -> selected again
        conn.execute("UPDATE sessions SET last_activity='2026-06-02T08:00:00.000Z' "
                     "WHERE session_id='h1'")
        assert len(es._select_sessions(conn, "h1", force=False)) == 1
        # unknown id -> empty, not an error
        assert es._select_sessions(conn, "nope", force=False) == []
    finally:
        conn.close()
    print("  ok  --session fast path: fresh=$0, resumed/force selected, unknown=empty")


# --- incremental slicing --------------------------------------------------------
def test_slice_turns():
    es = _load_script("enrich-sessions")
    turns = [SimpleNamespace(role="user", content=f"t{i}") for i in range(100)]
    # first enrichment of a long session: head + tail (tail carries the outcome)
    sliced, prior = es._slice_turns(turns, None, 0)
    assert prior is None and len(sliced) == 60
    assert sliced[0].content == "t0" and sliced[-1].content == "t99"
    # short session: everything
    short, _ = es._slice_turns(turns[:10], None, 0)
    assert len(short) == 10
    # re-enrichment: only the new turns
    facet = {"brief_summary": "x."}
    sliced, prior = es._slice_turns(turns, facet, 80)
    assert prior is facet and [t.content for t in sliced] == [f"t{i}" for i in range(80, 100)]
    # activity advanced but no new substantive turns: bounded tail
    sliced, prior = es._slice_turns(turns, facet, 100)
    assert prior is facet and len(sliced) == 30 and sliced[-1].content == "t99"
    print("  ok  _slice_turns: head+tail / full / incremental / bounded-tail")


def test_load_prior_reads_turns_seen(tmp_dir: Path | None = None):
    es = _load_script("enrich-sessions")
    conn = _temp_db()
    saved = sbconfig.FACETS_DIR
    try:
        tmp = Path(tempfile.mkdtemp())
        sbconfig.FACETS_DIR = tmp
        es.sbconfig.FACETS_DIR = tmp
        indexer.upsert(_header(sid="p1"), conn=conn)
        conn.execute("UPDATE sessions SET summary='done.' WHERE session_id='p1'")
        row = conn.execute("SELECT * FROM sessions WHERE session_id='p1'").fetchone()
        # no facet file -> full enrich
        assert es._load_prior(row, force=False) == (None, 0)
        (tmp / "p1.json").write_text(json.dumps(
            {"brief_summary": "done.", "_meta": {"turns_seen": 42}}))
        prior, seen = es._load_prior(row, force=False)
        assert seen == 42 and prior["brief_summary"] == "done."
        # --force ignores the prior entirely
        assert es._load_prior(row, force=True) == (None, 0)
    finally:
        sbconfig.FACETS_DIR = saved
        conn.close()
    print("  ok  _load_prior: facet turns_seen / missing file / --force")


# --- daily digest ---------------------------------------------------------------
def _digest_fixture(conn):
    """3 sessions across 2 projects on 2026-06-01 (UTC), one unsummarized."""
    specs = [
        ("a1", "x", "Refresh the MCP token", "2026-06-01T04:30:00.000Z",
         "Refreshed and verified the expired OAuth tokens.", "ops", "completed"),
        ("a2", "x", None, "2026-06-01T09:00:00.000Z", None, None, None),
        ("b1", "y", "Fix flaky DAG", "2026-06-01T11:15:00.000Z",
         "Stabilized the airflow DAG retries.", "debugging", "completed"),
    ]
    for sid, folder, title, start, summary, stype, outcome in specs:
        indexer.upsert(_header(sid=sid, folder_name=folder, title=title,
                               start_time=start, last_activity=start,
                               first_message="please fix the dag"), conn=conn)
        if summary:
            conn.execute(
                "UPDATE sessions SET summary=?, session_type=?, outcome=?, "
                "enriched_at=? WHERE session_id=?",
                (summary, stype, outcome, "2026-06-01T12:00:00.000Z", sid))
    conn.execute("INSERT INTO session_artifacts (session_id, type, content) "
                 "VALUES ('a1', 'journal', '## Accomplishments\n- refreshed both tokens')")
    conn.commit()


def test_daily_digest_render_day():
    from datetime import timezone as _tz
    dd = _load_script("daily-digest")
    conn = _temp_db()
    try:
        _digest_fixture(conn)
        days = dd.collect_days(conn, tz=_tz.utc)
        assert set(days) == {"2026-06-01"} and len(days["2026-06-01"]) == 3
        journals = {"a1": "## Accomplishments\n- refreshed both tokens"}
        md = dd.render_day("2026-06-01", days["2026-06-01"], journals, tz=_tz.utc)
        assert md.startswith("# Work log — 2026-06-01 (Monday)")
        assert "*3 sessions · 2 projects · claude ×3*" in md
        assert md.index("## x") < md.index("## y"), "busier project first"
        assert "### 04:30 — Refresh the MCP token  `ops · completed · claude · 0 min`" in md
        assert "#### Accomplishments" in md and "\n## Accomplishments" not in md, \
            "journal headings must demote under the session heading"
        assert "_(unsummarized)_ please fix the dag" in md
    finally:
        conn.close()
    print("  ok  render_day: grouping, badges, journal demotion, unsummarized fallback")


def test_daily_digest_local_date_grouping():
    from datetime import timedelta, timezone as _tz
    dd = _load_script("daily-digest")
    conn = _temp_db()
    try:
        indexer.upsert(_header(sid="tz1", start_time="2026-06-01T20:30:00.000Z",
                               last_activity="2026-06-01T20:30:00.000Z"), conn=conn)
        ist = _tz(timedelta(hours=5, minutes=30))
        assert set(dd.collect_days(conn, tz=ist)) == {"2026-06-02"}
        assert set(dd.collect_days(conn, tz=_tz.utc)) == {"2026-06-01"}
    finally:
        conn.close()
    print("  ok  collect_days groups by LOCAL date of start_time")


def test_daily_digest_needs_write_staleness():
    import os
    dd = _load_script("daily-digest")
    conn = _temp_db()
    try:
        _digest_fixture(conn)
        rows = dd.collect_days(conn)[next(iter(dd.collect_days(conn)))]
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "2026-06-01.md"
            assert dd.needs_write(f, rows), "missing file must be written"
            f.write_text("x")
            assert not dd.needs_write(f, rows), "fresh file (mtime now) is kept"
            # age the file to before the sessions' enriched_at -> stale
            old = 1_700_000_000  # 2023 — well before any fixture timestamp
            os.utime(f, (old, old))
            assert dd.needs_write(f, rows), "re-enriched session must refresh its day"
    finally:
        conn.close()
    print("  ok  needs_write: missing/stale rewritten, fresh kept")


# --- report data ----------------------------------------------------------------
def test_resolve_window_deterministic():
    from datetime import date
    rd = _load_script("report-data")
    today = date(2026, 7, 8)  # a Wednesday
    cases = {
        "last-week": (date(2026, 6, 29), date(2026, 7, 5)),
        "this-week": (date(2026, 7, 6), today),
        "last-month": (date(2026, 6, 1), date(2026, 6, 30)),
        "last-quarter": (date(2026, 4, 1), date(2026, 6, 30)),
        "last-6-months": (date(2026, 1, 1), date(2026, 6, 30)),
        "7d": (date(2026, 7, 2), today),
        "yesterday": (date(2026, 7, 7), date(2026, 7, 7)),
    }
    for spec, (lo, hi) in cases.items():
        got = rd.resolve_window(spec, None, None, today=today)[:2]
        assert got == (lo, hi), f"{spec}: {got}"
    lo, hi, _ = rd.resolve_window(None, "2026-01-15", "2026-03-01", today=today)
    assert (lo, hi) == (date(2026, 1, 15), date(2026, 3, 1))
    # quarter boundary: today in Q1 -> last quarter is Q4 of the prior year
    lo, hi, _ = rd.resolve_window("last-quarter", None, None, today=date(2026, 2, 10))
    assert (lo, hi) == (date(2025, 10, 1), date(2025, 12, 31))
    print("  ok  resolve_window: named windows, Nd, --from/--to, year boundary")


def test_build_report_structure():
    from datetime import date, timezone as _tz
    rd = _load_script("report-data")
    conn = _temp_db()
    try:
        _digest_fixture(conn)  # 3 sessions on 2026-06-01, one unsummarized
        conn.execute("UPDATE sessions SET turn_count=8, cost_usd=1.25 "
                     "WHERE session_id='a1'")
        conn.execute("INSERT INTO session_artifacts (session_id, type, content, turn_index) "
                     "VALUES ('a1','decision','ship it — see https://github.com/x/y/pull/7',0)")
        conn.commit()
        rep = rd.build_report(conn, date(2026, 6, 1), date(2026, 6, 30), "June",
                              tz=_tz.utc)
        assert rep["stats"]["sessions"] == 3 and rep["stats"]["active_days"] == 1
        assert rep["coverage"]["enriched"] == 2
        assert rep["coverage"]["unenriched_ids"] == ["a2"]
        assert rep["stats"]["by_source"] == {"claude": 3}
        assert rep["projects"][0]["name"] == "x", "busiest project first"
        a1 = next(s for p in rep["projects"] for s in p["sessions"]
                  if s["session_id"] == "a1")
        assert a1["journal"].startswith("## Accomplishments")
        assert a1["links"] == ["https://github.com/x/y/pull/7"]
        assert a1["week"] == "2026-W23" and not a1["trivial"]
        assert rep["stats"]["cost_usd"] == 1.25
        # out-of-window -> empty
        empty = rd.build_report(conn, date(2026, 7, 1), date(2026, 7, 31), "July",
                                tz=_tz.utc)
        assert empty["stats"]["sessions"] == 0 and empty["projects"] == []
    finally:
        conn.close()
    print("  ok  build_report: coverage, grouping, journal+links, window filter")


# --- persistence ----------------------------------------------------------------
def test_persist_writes_snapshot_and_journal_idempotently():
    es = _load_script("enrich-sessions")
    conn = _temp_db()
    try:
        indexer.upsert(_header(sid="s1"), conn=conn)
        facet = parse_facet_json(_FULL_FACET_RAW, "test")
        es._persist(conn, "s1", facet, "2026-06-01T10:05:00.000Z")
        es._persist(conn, "s1", facet, "2026-06-01T11:05:00.000Z")  # re-enrich
        conn.commit()
        row = conn.execute("SELECT * FROM sessions WHERE session_id='s1'").fetchone()
        assert row["summary"].startswith("Built the retry layer")
        assert row["session_type"] == "feature" and row["outcome"] == "completed"
        assert row["enriched_at"] == "2026-06-01T11:05:00.000Z"
        assert json.loads(row["topics"]) == ["python", "s3"]
        snap = conn.execute("SELECT * FROM session_snapshots WHERE session_id='s1'").fetchall()
        assert len(snap) == 1 and snap[0]["goal"] == "Make uploads resilient"
        assert json.loads(snap[0]["unresolved"]) == ["backfill metrics for retry counts"]
        journals = conn.execute(
            "SELECT content FROM session_artifacts WHERE session_id='s1' AND type='journal'"
        ).fetchall()
        assert len(journals) == 1 and "## Key decisions" in journals[0]["content"]
        decisions = conn.execute(
            "SELECT content FROM session_artifacts WHERE session_id='s1' AND type='decision'"
        ).fetchall()
        assert len(decisions) == 1 and decisions[0]["content"].startswith("tenacity")
    finally:
        conn.close()
    print("  ok  _persist: sessions/snapshot/journal/decisions written once, upserted on redo")


if __name__ == "__main__":
    print("Work-journal tests")
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for fn in tests:
        try:
            fn()
        except Exception as e:  # noqa: BLE001 — report, keep running the rest
            failures += 1
            print(f"  FAIL {fn.__name__}: {e}")
    if failures:
        print(f"\n{failures}/{len(tests)} test(s) FAILED.")
        sys.exit(1)
    print(f"\nAll {len(tests)} tests passed.")
