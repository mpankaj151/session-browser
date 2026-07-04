#!/usr/bin/env python3
"""Smoke + regression tests for the Session Browser. Runs standalone (no pytest):

    .venv/bin/python tests/test_smoke.py

Fully isolated: DB tests run against a temp database built by migrate(); the
suite never touches ~/.session-browser and passes on a fresh clone with no
install. Each test runs in its own try/except so one failure doesn't hide the
rest; the process exits nonzero if any test failed.
"""
from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
from importlib import util as _ilu
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import costs
import indexer
import redact
import reasoning
from enrichment.provider import FacetValidationError, parse_facet_json
from sources.base import SessionHeader, to_iso_utc


def _load_script(name: str):
    """Import a hyphen-named script module from scripts/."""
    spec = _ilu.spec_from_file_location(name.replace("-", "_"), _REPO / "scripts" / f"{name}.py")
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _temp_db() -> sqlite3.Connection:
    """A migrated, empty registry in a temp file — never the user's real DB."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    conn = indexer.connect(tmp.name)
    _load_script("migrate-db").migrate(conn)
    return conn


def _header(sid="__smoke__", **kw) -> SessionHeader:
    base = dict(session_id=sid, cli_source="claude", project_path="/proj/x",
                cwd="/x", folder_name="x", start_time="2026-01-01T00:00:00.000Z",
                last_activity="2026-01-01T00:00:00.000Z", first_message="hi",
                turn_count=1, title="T")
    base.update(kw)
    return SessionHeader(**base)


# --- redaction ------------------------------------------------------------------
def test_redaction_core():
    assert redact.redact('K=ctx7sk-00000000-aaaa-bbbb-cccc') == 'K=«REDACTED»'
    assert '«REDACTED»' in redact.redact('MY_SECRET="0123456789abcdef0123456789abcdef"')
    assert redact.redact('just normal prose') == 'just normal prose'
    assert redact.redact_count('a sk-ant-api03-aaaaaaaaaaaaaaaaaaaaaaaa and tvly-aaaaaaaaaaaa') == 2
    print("  ok  redaction core")


def test_redaction_json_and_modern_tokens():
    # JSON-form assignments (quote precedes the colon) must be caught
    r = redact.redact('{"api_key": "sup3rSecretValue123"}')
    assert 'sup3rSecretValue123' not in r and '«REDACTED»' in r
    # modern token formats
    for tok in ('github_pat_11ABCDEFG0123456789abcdefgh',
                'npm_abcdefghijklmnopqrstuvwxyz0123456789',
                'xoxc-1234567890-abcdef', 'xoxe-1234567890-abcdef'):
        assert tok.split('-')[0][:6] not in redact.redact(f"token here: {tok}"), tok
    print("  ok  redaction: JSON form + github_pat/npm/xox*")


def test_redaction_stripe_urlcreds_keys_auth():
    # Stripe underscores (the sk- patterns require a hyphen)
    for leak in ('STRIPE_KEY=sk_live_51H8xAbCdEfGhIj', 'bare sk_live_51H8xAbCdEfGhIj',
                 'whsec_AbCdEf123456789'):
        assert 'sk_live' not in redact.redact(leak) or '«REDACTED»' in redact.redact(leak), leak
        assert '«REDACTED»' in redact.redact(leak), leak
    # generic *_KEY assignments (not just *SECRET*/API_KEY)
    for leak, secret in (('ENCRYPTION_KEY=aGVsbG8xMjM0NTY=', 'aGVsbG8xMjM0NTY'),
                         ('SIGNING_KEY: 9f8e7d6c5b4a', '9f8e7d6c5b4a'),
                         ('"deploy_key": "abcdef-123456"', 'abcdef-123456')):
        r = redact.redact(leak)
        assert '«REDACTED»' in r and secret not in r, leak
    # URL basic-auth credentials — password masked, structure intact
    r = redact.redact('DATABASE_URL=postgres://admin:hunter2pw@db.internal/x')
    assert 'hunter2pw' not in r and '://admin:«REDACTED»@db.internal' in r
    # Authorization header, any/no scheme
    for leak in ('Authorization: Bearer shorttok123', 'Authorization: rawOpaque123456'):
        assert '«REDACTED»' in redact.redact(leak), leak
    # over-redaction guards: benign shapes survive
    for keep in ('primary_key=True', 'the monkey=business idiom',
                 'visit https://github.com/o/r.git today', 'http://localhost:7655/api'):
        assert redact.redact(keep) == keep, keep
    print("  ok  redaction: stripe/url-creds/*_KEY/authorization")


def test_redact_obj_walks_structures():
    facet = {"brief_summary": "Wired Stripe with sk_live_51H8xAbCdEfGhIj",
             "key_decisions": ["use STRIPE_KEY=sk_live_51H8xAbCdEfGhIj"],
             "goal_categories": {"payments": 2}, "n": 3}
    out = redact.redact_obj(facet)
    assert 'sk_live' not in json.dumps(out) and out["n"] == 3
    assert out["goal_categories"] == {"payments": 2}
    print("  ok  redact_obj masks nested facet strings")


def test_redaction_hash_scoping():
    # bare hashes in prose survive (FTS stays searchable by commit SHA)
    sha = '3031ee3891a699f0000000000000000000000000'
    assert redact.redact(f'commit {sha} fixed it') == f'commit {sha} fixed it'
    # …but the same hex in a value position is masked
    assert sha not in redact.redact(f"KEY='{sha}'")
    print("  ok  redaction: hex scoped to value positions")


def test_reasoning_trail_is_redacted():
    steps = [reasoning.ReasoningStep(
        turn_index=1, thinking="", decision="Set GITHUB_TOKEN=ghp_abcdefghij0123456789abcd now",
        actions=[{"tool": "Bash", "input": "command=export API_KEY='deadbeefdeadbeefdeadbeefdeadbeef'"}],
        signature_present=False)]
    md = reasoning.render_markdown(steps, {"session_id": "x", "title": "t"})
    assert "ghp_abcdefghij" not in md and "deadbeefdeadbeef" not in md
    assert "«REDACTED»" in md
    print("  ok  reasoning trails redacted before archive")


# --- costs ----------------------------------------------------------------------
def test_cost_mapping():
    pricing = costs.load_pricing()
    assert costs.tier_for_model("claude-opus-4-8", pricing) == "opus"
    assert costs.tier_for_model("claude-sonnet-4-6", pricing) == "sonnet"
    assert costs.tier_for_model("gpt-5-mini", pricing) == "gpt-5-mini"  # longest-alias-first
    assert costs.tier_for_model("some-future-model-9", pricing) is None  # unknown -> None, not a guess
    c = costs.cost_usd("claude-opus-4-8", {"input": 1_000_000, "output": 0,
                                           "cache_read": 0, "cache_write": 0}, pricing)
    assert abs(c - 15.0) < 1e-6
    assert costs.coerce_cache_write({"ephemeral_5m_input_tokens": 10,
                                     "ephemeral_1h_input_tokens": 5}) == 15
    print("  ok  cost mapping (tiers, unknown->None, cache dict coercion)")


def test_copilot_cost_extraction():
    cc = _load_script("compute-costs")
    event = {"type": "session.shutdown", "data": {"modelMetrics": {
        "gpt-5.4": {"usage": {"inputTokens": 1000, "outputTokens": 100,
                              "cacheReadTokens": 500, "cacheWriteTokens": 0, "reasoningTokens": 40}},
        "claude-sonnet-4.6": {"usage": {"inputTokens": 200, "outputTokens": 50,
                              "cacheReadTokens": 0, "cacheWriteTokens": 30, "reasoningTokens": 0}},
    }}}
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
        fh.write(json.dumps({"type": "user.message", "data": {"content": "hi"}}) + "\n")
        fh.write(json.dumps(event) + "\n")
        path = fh.name
    totals, per_model = cc._usage_copilot(Path(path))
    assert per_model["gpt-5.4"]["output"] == 140, "reasoningTokens should fold into output"
    assert totals["input"] == 1200 and totals["cache_read"] == 500
    print("  ok  copilot cost extraction (modelMetrics + reasoning-as-output)")


# --- timestamps -----------------------------------------------------------------
def test_timestamp_normalization():
    from datetime import datetime, timezone, timedelta
    z = to_iso_utc("2026-06-19T12:00:00.000Z")
    off = to_iso_utc("2026-06-19T12:00:00+00:00")
    dt = to_iso_utc(datetime(2026, 6, 19, 17, 30, tzinfo=timezone(timedelta(hours=5, minutes=30))))
    assert z == off == "2026-06-19T12:00:00.000Z"
    assert dt == "2026-06-19T12:00:00.000Z"
    assert to_iso_utc("garbage") == "" and to_iso_utc(None) == ""
    # canonical form is lexicographically sortable across sources
    assert to_iso_utc("2026-06-19T11:59:59Z") < to_iso_utc("2026-06-19T12:00:00+00:00")
    print("  ok  timestamp normalization (cross-source sortable)")


# --- adapters -------------------------------------------------------------------
def test_adapters():
    from sources.registry import build_source_registry
    from sources.claude import ClaudeSource
    from sources.copilot import CopilotSource
    reg = build_source_registry()
    assert "claude" in reg and "copilot" in reg
    assert ClaudeSource().session_id_for_path(Path("/p/abc-1.jsonl")) == "abc-1"
    assert CopilotSource().session_id_for_path(Path("/s/sid9/events.jsonl")) == "sid9"
    assert CopilotSource().session_id_for_path(Path("/s/sid9/other.jsonl")) is None
    print(f"  ok  adapters registered + path->id mapping: {list(reg)}")


def test_reasoning_extract():
    rec = {"type": "assistant", "timestamp": "2026-01-01T00:00:00Z",
           "message": {"content": [
               {"type": "thinking", "thinking": "", "signature": "abc"},
               {"type": "text", "text": "I'll list the files."},
               {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}},
           ]}}
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
        fh.write(json.dumps(rec) + "\n")
        path = fh.name
    steps = reasoning.extract(path)
    assert len(steps) == 1
    s = steps[0]
    assert s.decision == "I'll list the files." and s.signature_present
    assert s.actions[0]["tool"] == "Bash" and "ls" in s.actions[0]["input"]
    md = reasoning.render_markdown(steps, {"session_id": "x", "title": "t"})
    assert "Decision trail" in md and "🔒" in md
    print("  ok  claude reasoning extraction + render")


def test_copilot_reasoning():
    evt = {"type": "assistant.message", "timestamp": "2026-01-01T00:00:00Z", "data": {
        "reasoningText": "I should run the command via bash.",
        "content": "Running it now.",
        "toolRequests": [{"name": "bash", "arguments": {"command": "ls"}}]}}
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
        fh.write(json.dumps(evt) + "\n")
        path = fh.name
    steps = reasoning.extract_copilot(Path(path))
    assert len(steps) == 1 and steps[0].thinking.startswith("I should run")
    md = reasoning.render_markdown(steps, {"session_id": "x", "cli_source": "copilot"})
    assert "🧠 Reasoning" in md
    print("  ok  copilot reasoning extraction (real reasoningText)")


def test_copilot_nonstring_content():
    """A non-string content value degrades to skip, never crashes the parse."""
    from sources.copilot import CopilotSource
    import os
    d = Path(tempfile.mkdtemp()) / "sid-1"
    d.mkdir()
    (d / "workspace.yaml").write_text("cwd: /tmp\nname: t\n")
    evts = [{"type": "user.message", "data": {"content": {"weird": "dict"}}},
            {"type": "user.message", "data": {"content": "real text"}}]
    (d / "events.jsonl").write_text("\n".join(json.dumps(e) for e in evts))
    h = CopilotSource(d.parent).parse_header(d / "events.jsonl")
    assert h is not None and h.first_message == "real text" and h.turn_count == 2
    print("  ok  copilot non-string content degrades gracefully")


# --- facets ---------------------------------------------------------------------
def test_facet_parsing():
    raw = '```json\n{"brief_summary":"Did a thing","goal_categories":["python"],' \
          '"session_type":"feature","outcome":"completed"}\n```'
    f = parse_facet_json(raw, "test")
    assert f["goal_categories"] == {"python": 1}
    try:
        parse_facet_json('{"brief_summary":"x"}', "test")
        assert False, "should have raised on missing keys"
    except FacetValidationError:
        pass
    print("  ok  facet parsing + validation")


# --- DB behavior (temp DB — never the user's) -------------------------------------
def test_upsert_preserves_enrichment():
    conn = _temp_db()
    try:
        h = _header()
        indexer.upsert(h, conn=conn)
        conn.execute("UPDATE sessions SET summary='ENRICHED' WHERE session_id='__smoke__'")
        h.turn_count = 2
        h.last_activity = "2026-01-02T00:00:00.000Z"
        indexer.upsert(h, conn=conn)
        row = conn.execute("SELECT summary, turn_count FROM sessions WHERE session_id='__smoke__'").fetchone()
        assert row["summary"] == "ENRICHED" and row["turn_count"] == 2
    finally:
        conn.close()
    print("  ok  upsert preserves enrichment (COALESCE)")


def test_empty_string_fields_not_sticky():
    """B3 regression: '' stored first must be replaced by a later real value."""
    conn = _temp_db()
    try:
        indexer.upsert(_header(first_message="", cwd="", folder_name=""), conn=conn)
        indexer.upsert(_header(first_message="the real question", cwd="/real",
                               folder_name="proj"), conn=conn)
        row = conn.execute("SELECT first_message, cwd, folder_name FROM sessions "
                           "WHERE session_id='__smoke__'").fetchone()
        assert row["first_message"] == "the real question"
        assert row["cwd"] == "/real" and row["folder_name"] == "proj"
        # …and a later '' must not clobber the real value
        indexer.upsert(_header(first_message=""), conn=conn)
        row = conn.execute("SELECT first_message FROM sessions WHERE session_id='__smoke__'").fetchone()
        assert row["first_message"] == "the real question"
    finally:
        conn.close()
    print("  ok  empty-string fields neither stick nor clobber (NULLIF)")


def test_archive_resurrect_roundtrip():
    """B1/B2 regression: archive hides; a fresh upsert (file exists) resurrects."""
    conn = _temp_db()
    try:
        indexer.upsert(_header(), conn=conn)
        indexer.archive("__smoke__", conn=conn)
        assert conn.execute("SELECT archived FROM sessions WHERE session_id='__smoke__'").fetchone()[0] == 1
        indexer.upsert(_header(), conn=conn)
        assert conn.execute("SELECT archived FROM sessions WHERE session_id='__smoke__'").fetchone()[0] == 0
    finally:
        conn.close()
    print("  ok  archive -> upsert resurrects (no one-way trapdoor)")


if __name__ == "__main__":
    print("Session Browser smoke + regression tests")
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
