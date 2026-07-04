#!/usr/bin/env python3
"""Smoke tests for the Session Browser. Runs standalone (no pytest needed):

    .venv/bin/python tests/test_smoke.py

Covers: facet parsing/validation, cost mapping, reasoning extraction shape,
adapter availability, and the COALESCE upsert preserving enrichment.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import costs
import indexer
from enrichment.provider import FacetValidationError, parse_facet_json
from sources.base import SessionHeader


def test_facet_parsing():
    raw = '```json\n{"brief_summary":"Did a thing","goal_categories":["python"],' \
          '"session_type":"feature","outcome":"completed"}\n```'
    f = parse_facet_json(raw, "test")
    assert f["brief_summary"].endswith((".", "!", "?", "…")), "summary should be finalized"
    assert f["goal_categories"] == {"python": 1}, "list goal_categories coerced to dict"
    assert f["_meta"]["provider"] == "test"
    try:
        parse_facet_json('{"brief_summary":"x"}', "test")
        assert False, "should have raised on missing keys"
    except FacetValidationError:
        pass
    print("  ok  facet parsing + validation")


def test_cost_mapping():
    pricing = costs.load_pricing()
    assert costs.tier_for_model("claude-opus-4-8", pricing) == "opus"
    assert costs.tier_for_model("claude-sonnet-4-6", pricing) == "sonnet"
    # longest-alias-first: gpt-5-mini must not match gpt-5
    assert costs.tier_for_model("gpt-5-mini", pricing) == "gpt-5-mini"
    c = costs.cost_usd("claude-opus-4-8", {"input": 1_000_000, "output": 0,
                                           "cache_read": 0, "cache_write": 0}, pricing)
    assert abs(c - 15.0) < 1e-6, f"1M opus input tokens should be $15, got {c}"
    assert costs.coerce_cache_write({"ephemeral_5m_input_tokens": 10,
                                     "ephemeral_1h_input_tokens": 5}) == 15
    print("  ok  cost mapping (tier match + USD + cache dict coercion)")


def test_reasoning_extract():
    import json
    import reasoning
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
    assert s.decision == "I'll list the files."
    assert s.signature_present is True
    assert s.actions[0]["tool"] == "Bash" and "ls" in s.actions[0]["input"]
    md = reasoning.render_markdown(steps, {"session_id": "x", "title": "t"})
    assert "Decision trail" in md and "🔒" in md
    print("  ok  reasoning extraction + render")


def test_copilot_cost_extraction():
    """Copilot per-model usage parsed from a synthetic session.shutdown event."""
    import importlib.util
    import json
    spec = importlib.util.spec_from_file_location(
        "compute_costs", Path(__file__).resolve().parent.parent / "scripts" / "compute-costs.py")
    cc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cc)
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
    assert set(per_model) == {"gpt-5.4", "claude-sonnet-4.6"}
    print("  ok  copilot cost extraction (modelMetrics + reasoning-as-output)")


def test_copilot_reasoning():
    import json
    import reasoning
    evt = {"type": "assistant.message", "timestamp": "2026-01-01T00:00:00Z", "data": {
        "reasoningText": "I should run the command via bash.",
        "content": "Running it now.",
        "toolRequests": [{"name": "bash", "arguments": {"command": "ls"}}]}}
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
        fh.write(json.dumps(evt) + "\n")
        path = fh.name
    steps = reasoning.extract_copilot(Path(path))
    assert len(steps) == 1 and steps[0].thinking.startswith("I should run")
    assert steps[0].decision == "Running it now." and steps[0].actions[0]["tool"] == "bash"
    md = reasoning.render_markdown(steps, {"session_id": "x", "cli_source": "copilot"})
    assert "🧠 Reasoning" in md, "copilot reasoning text should render"
    print("  ok  copilot reasoning extraction (real reasoningText)")


def test_redaction():
    import redact
    assert redact.redact('K=ctx7sk-00000000-aaaa-bbbb-cccc') == 'K=«REDACTED»'
    assert '«REDACTED»' in redact.redact('PAYTM_SECRET="0123456789abcdef0123456789abcdef"')
    assert redact.redact('just normal prose') == 'just normal prose'
    assert redact.redact_count('a sk-ant-api03-aaaaaaaaaaaaaaaaaaaaaaaa and tvly-aaaaaaaaaaaa') >= 2
    print("  ok  secret redaction (masks keys, keeps prose)")


def test_adapters_available():
    from sources.registry import build_source_registry
    reg = build_source_registry()
    assert "claude" in reg and "copilot" in reg
    print(f"  ok  adapters registered: {list(reg)}")


def test_upsert_preserves_enrichment():
    h = SessionHeader(session_id="__smoke__", cli_source="claude", project_path="/x",
                      cwd="/x", folder_name="x", start_time="2026-01-01T00:00:00Z",
                      last_activity="2026-01-01T00:00:00Z", first_message="hi",
                      turn_count=1, title="T")
    conn = indexer.connect()
    try:
        indexer.upsert(h, conn=conn)
        conn.execute("UPDATE sessions SET summary='ENRICHED' WHERE session_id='__smoke__'")
        h.turn_count = 2
        h.last_activity = "2026-01-02T00:00:00Z"
        indexer.upsert(h, conn=conn)
        row = conn.execute("SELECT summary, turn_count FROM sessions WHERE session_id='__smoke__'").fetchone()
        assert row["summary"] == "ENRICHED", "re-upsert clobbered the summary"
        assert row["turn_count"] == 2, "turn_count should update"
        conn.execute("DELETE FROM sessions WHERE session_id='__smoke__'")
        conn.commit()
    finally:
        conn.close()
    print("  ok  upsert preserves enrichment (COALESCE)")


if __name__ == "__main__":
    print("Session Browser smoke tests")
    for fn in [test_facet_parsing, test_cost_mapping, test_reasoning_extract,
               test_copilot_cost_extraction, test_copilot_reasoning, test_redaction,
               test_adapters_available, test_upsert_preserves_enrichment]:
        fn()
    print("\nAll smoke tests passed.")
