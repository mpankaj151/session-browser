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


# --- shell-embedded python --------------------------------------------------
def test_shell_heredoc_python_compiles():
    """Python inside .sh heredocs is invisible to compileall — a 3.12-only
    f-string there once shipped green through CI while breaking 3.11 users."""
    import re as _re
    blocks = 0
    for sh in list(_REPO.glob("*.sh")) + list((_REPO / "bin").glob("*.sh")):
        text = sh.read_text(encoding="utf-8")
        # Openers may carry a suffix (<<'PYEOF' || echo ...) or spill onto the
        # next line with a backslash continuation — the python body only starts
        # after the full shell command, and none of it may escape compilation.
        for m in _re.finditer(r"<<'?(PYEOF)'?(?:[^\n]*\\\n)*[^\n]*\n(.*?)\n\1", text, _re.S):
            compile(m.group(2), f"{sh.name}:heredoc", "exec")
            blocks += 1
    assert blocks >= 3, f"expected to find python heredocs, got {blocks}"
    print(f"  ok  {blocks} shell-heredoc python blocks compile on this interpreter")


def test_install_cr_repairs_moved_repo_paths():
    """Re-running install-cr.sh after the repo moves must REPLACE the managed
    rc blocks — presence-checking alone left cr/sb pointing at the dead path."""
    import os
    import subprocess
    with tempfile.TemporaryDirectory() as home:
        rc = Path(home) / ".zshrc"
        stale = "/tmp/old-location/session-browser"
        rc.write_text(
            "export KEEP_ME=1\n\n"
            "# >>> session-browser cr >>>\n"
            f'cr() {{ "{stale}/bin/resume-here.sh" "$@"; }}\n'
            "# <<< session-browser cr <<<\n\n"
            "# >>> session-browser sb >>>\n"
            f'sb() {{ local REPO="{stale}"; }}\n'
            "# <<< session-browser sb <<<\n"
        )
        env = {**os.environ, "HOME": home, "SHELL": "/bin/zsh"}
        run = lambda: subprocess.run(  # noqa: E731
            ["bash", str(_REPO / "bin" / "install-cr.sh")],
            env=env, capture_output=True, text=True)
        r = run()
        assert r.returncode == 0, r.stderr
        assert "Updated" in r.stdout, r.stdout
        text = rc.read_text()
        assert "export KEEP_ME=1" in text, "unmanaged rc content must survive"
        assert stale not in text, "stale repo path must be gone"
        assert str(_REPO) in text, "current repo path must be installed"
        for tag in ("cr", "sb"):
            assert text.count(f"# >>> session-browser {tag} >>>") == 1
        r2 = run()  # second run: idempotent
        assert r2.returncode == 0 and rc.read_text() == text
    print("  ok  install-cr.sh replaces stale rc blocks after a repo move")


def test_semsearch_offline_gate():
    """With no cached model and no SB_ALLOW_MODEL_DOWNLOAD, get_model() must
    fail fast with the friendly RuntimeError and never attempt the network.
    Regression: the old env-var approach (HF_HUB_OFFLINE) froze into
    huggingface_hub at import, so even the AUTHORIZED download retry was a
    permanent no-op on machines without a warm cache."""
    import importlib.util
    import os
    import subprocess
    if importlib.util.find_spec("sentence_transformers") is None:
        print("  skip semsearch offline gate (sentence-transformers not installed)")
        return
    env = {**os.environ, "HF_HOME": tempfile.mkdtemp(prefix="sb-hf-empty-")}
    for k in ("SB_ALLOW_MODEL_DOWNLOAD", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        env.pop(k, None)
    r = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.path.insert(0, {str(_REPO)!r}); "
         "import semsearch; semsearch.get_model()"],
        env=env, capture_output=True, text=True, timeout=180)
    assert r.returncode != 0, "expected failure with an empty model cache"
    assert "not cached locally" in r.stderr, r.stderr[-800:]
    print("  ok  semsearch: cold cache + no opt-in -> fast friendly error, no download")


def test_install_hook_repairs_moved_repo_path():
    """install.sh's hook registration must repoint a stale session-hook entry
    (repo moved), keep foreign Stop hooks, and no-op when already correct."""
    import os
    import re as _re
    import subprocess
    block = next(
        m.group(2)
        for m in _re.finditer(r"<<'?(PYEOF)'?(?:[^\n]*\\\n)*[^\n]*\n(.*?)\n\1",
                              (_REPO / "install.sh").read_text(), _re.S)
        if "session-hook.py" in m.group(2))
    with tempfile.TemporaryDirectory() as home:
        settings = Path(home) / ".claude" / "settings.json"
        settings.parent.mkdir()
        foreign = {"hooks": [{"type": "command", "command": "echo other-tool"}]}
        stale = {"hooks": [{"type": "command",
                            "command": '"/old/path/.venv/bin/python" "/old/path/scripts/session-hook.py"'}]}
        settings.write_text(json.dumps({"hooks": {"Stop": [foreign, stale]}}))
        env = {**os.environ, "HOME": home}
        run = lambda: subprocess.run(  # noqa: E731
            [sys.executable, "-", str(_REPO)],
            input=block, env=env, capture_output=True, text=True)
        r = run()
        assert r.returncode == 0, r.stderr
        assert "registered" in r.stdout, r.stdout
        hooks = json.loads(settings.read_text())["hooks"]
        for event in ("Stop", "SessionEnd"):
            ours = [h for h in hooks[event] if "session-hook.py" in json.dumps(h)]
            assert len(ours) == 1 and str(_REPO) in ours[0]["hooks"][0]["command"], event
        assert "/old/path" not in json.dumps(hooks)
        assert foreign in hooks["Stop"], "foreign Stop hooks must survive"
        assert settings.with_suffix(".json.sb-backup").exists()
        before = settings.read_text()
        r2 = run()  # second run: correct entries -> no rewrite
        assert "already present" in r2.stdout and settings.read_text() == before
    print("  ok  install.sh registers Stop + SessionEnd, repoints stale paths")


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


def test_upsert_is_monotonic():
    """A re-parse of a shorter/older view (partial sync, second codex rollout
    file for the same id) must not walk last_activity/turn_count backwards."""
    conn = _temp_db()
    try:
        indexer.upsert(_header(turn_count=50, last_activity="2026-06-02T00:00:00.000Z",
                               project_path="/roll/06/02"), conn=conn)
        indexer.upsert(_header(turn_count=3, last_activity="2026-06-01T00:00:00.000Z",
                               project_path="/roll/06/01"), conn=conn)
        r = conn.execute("SELECT turn_count, last_activity, project_path FROM sessions "
                         "WHERE session_id='__smoke__'").fetchone()
        assert r["turn_count"] == 50 and r["last_activity"] == "2026-06-02T00:00:00.000Z"
        # canonical dir sticks with the NEWEST activity, not the latest parse
        assert r["project_path"] == "/roll/06/02"
        # ...and a genuinely newer parse advances everything, dir included
        indexer.upsert(_header(turn_count=60, last_activity="2026-06-03T00:00:00.000Z",
                               project_path="/roll/06/03"), conn=conn)
        r = conn.execute("SELECT turn_count, project_path FROM sessions "
                         "WHERE session_id='__smoke__'").fetchone()
        assert r["turn_count"] == 60 and r["project_path"] == "/roll/06/03"
    finally:
        conn.close()
    print("  ok  upsert monotonic (no backward regression)")


def test_to_iso_utc_hardening():
    # epoch milliseconds must not become a year-56000 date
    assert to_iso_utc(1777573058000).startswith("2026-")
    assert to_iso_utc(1777573058).startswith("2026-")
    # nanosecond-precision RFC3339 strings truncate instead of vanishing
    assert to_iso_utc("2026-04-30T18:17:38.123456789Z") == "2026-04-30T18:17:38.123Z"
    print("  ok  to_iso_utc: epoch-ms + nanosecond fractions")


# --- costs ----------------------------------------------------------------------
def test_cost_mapping():
    pricing = costs.load_pricing()
    assert costs.tier_for_model("claude-opus-4-8", pricing) == "opus-4.5"
    assert costs.tier_for_model("claude-opus-4-1", pricing) == "opus-4"      # retired rate kept
    assert costs.tier_for_model("claude-sonnet-4-6", pricing) == "sonnet-4"
    assert costs.tier_for_model("gpt-5-mini", pricing) == "gpt-5-mini"  # longest-alias-first
    assert costs.tier_for_model("some-future-model-9", pricing) is None  # unknown -> None, not a guess
    c = costs.cost_usd("claude-opus-4-8", {"input": 1_000_000, "output": 0,
                                           "cache_read": 0, "cache_write": 0}, pricing)
    assert abs(c - 5.0) < 1e-6
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
    from sources.opencode import OpenCodeSource
    reg = build_source_registry()
    assert "claude" in reg and "copilot" in reg and "opencode" in reg, list(reg)
    assert isinstance(reg["opencode"], OpenCodeSource)
    assert reg["opencode"].session_id_for_path(Path(f"/m/{_OC_ROOT}.jsonl")) == _OC_ROOT
    assert ClaudeSource("/p").session_id_for_path(Path("/p/-proj/abc-1.jsonl")) == "abc-1"
    assert CopilotSource().session_id_for_path(Path("/s/sid9/events.jsonl")) == "sid9"
    assert CopilotSource().session_id_for_path(Path("/s/sid9/other.jsonl")) is None
    print(f"  ok  adapters registered + path->id mapping: {list(reg)}")


def test_claude_ignores_subagent_transcripts():
    """A multi-agent run writes sidechain transcripts under <session>/subagents/.

    They are part of their parent session, not sessions themselves: every record
    is isSidechain, so they'd index as permanently-empty rows. The watcher walks
    recursive=True and gates purely on session_id_for_path(), so the gate — not
    discover()'s glob — is what has to reject them.
    """
    from sources.claude import ClaudeSource
    src = ClaudeSource("/p")
    root = "/p/-Users-me-proj/6550180f-14ff-4b91-a93d-d951ed98c2f7"
    # real transcript still maps to its id (the discover() `*/*.jsonl` shape)
    assert src.session_id_for_path(Path("/p/-Users-me-proj/abc-1.jsonl")) == "abc-1"
    # direct subagent sidechain
    assert src.session_id_for_path(Path(f"{root}/subagents/agent-a0c125.jsonl")) is None
    # nested under a workflow dir — same tree, deeper
    assert src.session_id_for_path(
        Path(f"{root}/subagents/workflows/wf_3433eeec-4b7/agent-a921527.jsonl")) is None
    # workflow journals: every one of these has stem "journal", so without the
    # gate they all collide onto a single bogus session row keyed "journal".
    assert src.session_id_for_path(
        Path(f"{root}/subagents/workflows/wf_ffba7666-ae0/journal.jsonl")) is None
    print("  ok  claude adapter rejects subagent/workflow sidechain transcripts")


def test_claude_parse_header_rejects_subagent_path():
    """Defense in depth: the Stop hook calls parse_header() directly on the path it
    is handed, never consulting session_id_for_path(). parse_header is the one
    chokepoint every entry path (hook, watcher, backfill, enrich) goes through, so
    the 'is this a session?' invariant has to hold there too."""
    from sources.claude import ClaudeSource
    rec = {"type": "user", "isSidechain": True, "cwd": "/w", "timestamp": "2026-01-01T00:00:00Z",
           "message": {"content": "spawned task"}}
    with tempfile.TemporaryDirectory() as d:
        sub = Path(d) / "sess-uuid" / "subagents"
        sub.mkdir(parents=True)
        agent = sub / "agent-a0c125.jsonl"
        agent.write_text(json.dumps(rec) + "\n")
        assert ClaudeSource().parse_header(agent) is None
        # a normal transcript at the same depth-2 shape still parses
        normal = Path(d) / "proj" / "abc-1.jsonl"
        normal.parent.mkdir(parents=True)
        normal.write_text(json.dumps({**rec, "isSidechain": False}) + "\n")
        assert ClaudeSource().parse_header(normal) is not None
    print("  ok  parse_header refuses subagent transcripts (hook path defended)")


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
        indexer.archive("__smoke__", indexer.TRANSCRIPT_MISSING, conn=conn)
        assert conn.execute("SELECT archived FROM sessions WHERE session_id='__smoke__'").fetchone()[0] == 1
        indexer.upsert(_header(), conn=conn)
        assert conn.execute("SELECT archived FROM sessions WHERE session_id='__smoke__'").fetchone()[0] == 0
    finally:
        conn.close()
    print("  ok  archive -> upsert resurrects (no one-way trapdoor)")


# --- codex adapter: paginated dialect, compression, archived root -----------
# Codex ~0.144+ writes every non-ephemeral CLI thread in "paginated" history
# mode (codex-rs/tui/src/app_server_session.rs, codex-rs/exec/src/lib.rs both
# set `history_mode: (!ephemeral).then_some(Paginated)`), and a background
# worker zstd-compresses rollouts older than 7 days. The line shapes below are
# copied from Codex's own fixture (codex-rs/tui/src/lib.rs) so these tests fail
# if we drift from what the CLI actually writes.
_CX_ID = "019e18fa-0d21-7461-922c-5ccaad36df05"
_CX_NAME = f"rollout-2026-08-01T10-00-00-{_CX_ID}.jsonl"


def _cx_line(ordinal, payload):
    return {"timestamp": "2026-08-01T10:00:00Z", "type": "event_msg",
            "payload": payload, "ordinal": ordinal}


def _cx_rollout(mode="paginated") -> str:
    """One rollout's JSONL text in the `legacy` or `paginated` dialect."""
    meta = {"id": _CX_ID, "timestamp": "2026-08-01T10:00:00Z", "cwd": "/Users/x/proj",
            "originator": "codex_cli_rs", "cli_version": "0.150.1",
            "history_mode": mode, "model_provider": "openai"}
    lines = [{"timestamp": "2026-08-01T10:00:00Z", "type": "session_meta", "payload": meta},
             _cx_line(2, {"type": "turn_context", "model": "gpt-5.5"})]
    if mode == "paginated":
        item = lambda n, t, i, c: _cx_line(n, {  # noqa: E731
            "type": "item_completed", "thread_id": _CX_ID, "turn_id": "t1",
            "item": {"type": t, "id": i, "content": c}})
        lines += [
            item(3, "UserMessage", "u0", [{"type": "text", "text": "why is codex missing?"}]),
            item(4, "AgentMessage", "a0", [{"type": "Text", "text": "the format changed"}]),
            item(5, "UserMessage", "u1", [{"type": "text", "text": "fix it"}]),
        ]
    else:
        lines += [
            _cx_line(3, {"type": "user_message", "message": "why is codex missing?"}),
            _cx_line(4, {"type": "agent_message", "message": "the format changed"}),
            _cx_line(5, {"type": "user_message", "message": "fix it"}),
        ]
    return "".join(json.dumps(rec) + "\n" for rec in lines)


def _cx_tree(tmp: Path, text: str, root="sessions", compress=False) -> Path:
    """Write one rollout into <tmp>/<root>/2026/08/01/ and return its path."""
    day = tmp / root / "2026" / "08" / "01"
    day.mkdir(parents=True, exist_ok=True)
    path = day / _CX_NAME
    if compress:
        import zstandard
        path = day / (_CX_NAME + ".zst")
        path.write_bytes(zstandard.ZstdCompressor().compress(text.encode()))
    else:
        path.write_text(text, encoding="utf-8")
    return path


def _cx_source(tmp: Path):
    from sources.codex import CodexSource
    return CodexSource(tmp / "sessions")


def test_codex_paginated_rollout_parses_turns():
    """Paginated rollouts carry item_completed/TurnItem instead of user_message.
    Parsing only the legacy events yielded turn_count==0, which parse_header
    treated as 'not browsable' — every new Codex session vanished silently."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        path = _cx_tree(tmp, _cx_rollout("paginated"))
        h = _cx_source(tmp).parse_header(path)
        assert h is not None, "paginated rollout parsed as unbrowsable"
        assert h.turn_count == 2, f"expected 2 user turns, got {h.turn_count}"
        assert h.first_message == "why is codex missing?", h.first_message
        assert h.session_id == _CX_ID and h.model_used == "gpt-5.5"
    print("  ok  codex paginated rollout yields turns + first_message")


def test_codex_legacy_rollout_still_parses():
    """Regression guard: widening the parser must not drop the old dialect."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        h = _cx_source(tmp).parse_header(_cx_tree(tmp, _cx_rollout("legacy")))
        assert h is not None and h.turn_count == 2, h
        assert h.first_message == "why is codex missing?"
    print("  ok  codex legacy rollout still parses")


def test_codex_compressed_rollout_matches_plain():
    """Codex zstd-compresses rollouts older than 7 days in place. A .jsonl.zst
    must produce the same header as its plain twin, not disappear."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        text = _cx_rollout("paginated")
        plain = _cx_source(tmp).parse_header(_cx_tree(tmp, text))
        for p in (tmp / "sessions" / "2026" / "08" / "01").iterdir():
            p.unlink()
        gz = _cx_tree(tmp, text, compress=True)
        comp = _cx_source(tmp).parse_header(gz)
        assert comp is not None, ".jsonl.zst rollout parsed as unbrowsable"
        assert (comp.session_id, comp.turn_count, comp.first_message) == \
               (plain.session_id, plain.turn_count, plain.first_message)
    print("  ok  codex .jsonl.zst parses identically to plain .jsonl")


def test_codex_session_id_for_path_variants():
    """.zst suffix and reverted-thread names (rollout-<ts>-<thread>_<rollout>)
    both broke the old 'last five dash-separated groups' heuristic."""
    with tempfile.TemporaryDirectory() as td:
        src = _cx_source(Path(td))
        base = Path(f"/s/2026/08/01/rollout-2026-08-01T10-00-00-{_CX_ID}.jsonl")
        assert src.session_id_for_path(base) == _CX_ID
        assert src.session_id_for_path(Path(str(base) + ".zst")) == _CX_ID
        rev = base.with_name(f"rollout-2026-08-01T10-00-00-{_CX_ID}_019e0000-1111-2222-3333-444455556666.jsonl")
        assert src.session_id_for_path(rev) == _CX_ID, src.session_id_for_path(rev)
        assert src.session_id_for_path(Path("/s/notes.txt")) is None
    print("  ok  codex session_id_for_path handles .zst and reverted names")


def test_codex_discovers_archived_sessions():
    """`codex archive` MOVES rollouts to ~/.codex/archived_sessions/. They are
    still the user's work and must stay searchable."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _cx_tree(tmp, _cx_rollout("paginated"), root="archived_sessions")
        found = list(_cx_source(tmp).discover())
        assert len(found) == 1, f"archived_sessions not discovered: {found}"
    print("  ok  codex discovers archived_sessions as a second root")


def test_codex_unrecognised_schema_is_not_silent():
    """The outage was invisible because 'no user turns' and 'I do not
    understand this file' both returned None. A rollout with records but no
    recognised turn types must still index (turn_count 0), not vanish."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        future = "".join(json.dumps(rec) + "\n" for rec in [
            {"timestamp": "2026-08-01T10:00:00Z", "type": "session_meta",
             "payload": {"id": _CX_ID, "timestamp": "2026-08-01T10:00:00Z",
                         "cwd": "/Users/x/proj", "cli_version": "9.9.9"}},
            _cx_line(2, {"type": "quantum_message", "message": "hello from 2027"}),
            _cx_line(3, {"type": "quantum_message", "message": "goodbye"}),
        ])
        h = _cx_source(tmp).parse_header(_cx_tree(tmp, future))
        assert h is not None, "unrecognised schema silently dropped (the original bug)"
        assert h.session_id == _CX_ID and h.turn_count == 0
    print("  ok  codex unrecognised rollout schema indexes instead of vanishing")


def test_codex_truly_empty_rollout_still_skipped():
    """A meta-only rollout genuinely has nothing to browse — it must stay
    skipped, or every aborted Codex launch litters the browser."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        only_meta = json.dumps({"timestamp": "2026-08-01T10:00:00Z", "type": "session_meta",
                                "payload": {"id": _CX_ID, "cwd": "/Users/x/proj"}}) + "\n"
        assert _cx_source(tmp).parse_header(_cx_tree(tmp, only_meta)) is None
    print("  ok  codex meta-only rollout still skipped")


def test_codex_parse_full_reads_both_dialects():
    """parse_full feeds the reasoning archive and enrichment — it needs the
    message bodies out of paginated items, not just the header counts."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        parsed = _cx_source(tmp).parse_full(_cx_tree(tmp, _cx_rollout("paginated")))
        assert parsed is not None
        roles = [(t.role, t.content) for t in parsed.turns]
        assert roles == [("user", "why is codex missing?"),
                         ("assistant", "the format changed"),
                         ("user", "fix it")], roles
    print("  ok  codex parse_full extracts paginated message bodies")


def test_codex_available_without_binary_on_path():
    """is_available() gated a filesystem watcher on `which codex`. The unified
    ChatGPT/Codex app moves the binary, so the watcher silently stopped
    watching ~/.codex/sessions with no log line at all."""
    import os
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        (tmp / "sessions").mkdir()
        old = os.environ.get("PATH", "")
        try:
            os.environ["PATH"] = str(tmp / "no-bins")
            assert _cx_source(tmp).is_available(), \
                "codex source unavailable purely because the binary left $PATH"
        finally:
            os.environ["PATH"] = old
    print("  ok  codex source availability does not depend on $PATH")


def test_watcher_ignores_compression_representation_change():
    """Compressing rollout.jsonl -> rollout.jsonl.zst deletes the plain file.
    Treating that delete as a session deletion archived live sessions out of
    the browser about a week after they were written."""
    import watcher
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        plain = tmp / _CX_NAME
        comp = tmp / (_CX_NAME + ".zst")
        comp.write_bytes(b"\x28\xb5\x2f\xfd")
        assert watcher._is_representation_change(plain) is True
        comp.unlink()
        plain.write_text("{}\n")
        assert watcher._is_representation_change(comp) is True
        plain.unlink()
        assert watcher._is_representation_change(plain) is False
    print("  ok  watcher treats compression as a representation change, not a delete")



# --- archive lifecycle: reason + visibility ----------------------------------
def test_archive_records_reason_and_resurrect_clears_it():
    """An archived row must say WHY. A transcript that aged out (Claude's
    cleanupPeriodDays) is a real session the UI should keep showing; a subagent
    sidechain that never was a session is not. Same flag, different meaning."""
    conn = _temp_db()
    try:
        indexer.upsert(_header(), conn=conn)
        indexer.archive("__smoke__", reason="transcript-missing", conn=conn)
        row = conn.execute("SELECT archived, archived_reason, archived_at FROM sessions "
                           "WHERE session_id='__smoke__'").fetchone()
        assert row["archived"] == 1
        assert row["archived_reason"] == "transcript-missing"
        assert row["archived_at"], "archived_at must be stamped"
        # a fresh upsert (file exists again) must leave no stale reason behind
        indexer.upsert(_header(), conn=conn)
        row = conn.execute("SELECT archived, archived_reason, archived_at FROM sessions "
                           "WHERE session_id='__smoke__'").fetchone()
        assert row["archived"] == 0
        assert row["archived_reason"] is None and row["archived_at"] is None
    finally:
        conn.close()
    print("  ok  archive records a reason + timestamp; resurrect clears both")


def test_infer_archive_reason_separates_noise_from_aged_out():
    """Backfilling history (rows archived before the reason column existed): a
    row with zero turns and no first message never held a conversation — it's a
    subagent sidechain or workflow journal. Anything with content was a real
    session whose transcript went missing. Derived, never guessed from the id."""
    infer = indexer.infer_archive_reason
    assert infer({"turn_count": 0, "first_message": ""}) == indexer.NOT_A_SESSION
    assert infer({"turn_count": 0, "first_message": None}) == indexer.NOT_A_SESSION
    assert infer({"turn_count": None, "first_message": "  "}) == indexer.NOT_A_SESSION
    assert infer({"turn_count": 3, "first_message": "fix the bug"}) == indexer.TRANSCRIPT_MISSING
    # one typed turn is still a conversation
    assert infer({"turn_count": 1, "first_message": "hi"}) == indexer.TRANSCRIPT_MISSING
    # content survives even when an ancient row never got a turn_count
    assert infer({"turn_count": None, "first_message": "x"}) == indexer.TRANSCRIPT_MISSING
    print("  ok  infer_archive_reason: 0 turns + no message -> noise, else aged-out")


def test_migrate_backfills_reason_onto_legacy_archived_rows():
    """Upgrading a registry that archived rows before the reason column existed:
    migrate() classifies them with the derived rule, touches nothing live, never
    overwrites a reason already recorded, and is idempotent."""
    conn = _temp_db()
    try:
        conn.executemany(
            "INSERT INTO sessions (session_id, archived, turn_count, first_message) VALUES (?,?,?,?)",
            [("agent-abc", 1, 0, ""),           # subagent sidechain noise
             ("real-1", 1, 12, "refactor x"),   # real session, transcript aged out
             ("live-1", 0, 3, "y")])            # live — must stay untouched
        conn.commit()
        migrate = _load_script("migrate-db").migrate
        migrate(conn)
        rows = dict(conn.execute("SELECT session_id, archived_reason FROM sessions").fetchall())
        assert rows["agent-abc"] == indexer.NOT_A_SESSION, rows
        assert rows["real-1"] == indexer.TRANSCRIPT_MISSING, rows
        assert rows["live-1"] is None, rows
        # a reason recorded at the source is authoritative; re-migrating keeps it
        conn.execute("UPDATE sessions SET archived_reason=? WHERE session_id='agent-abc'",
                     (indexer.TRANSCRIPT_MISSING,))
        conn.commit()
        migrate(conn)
        assert conn.execute("SELECT archived_reason FROM sessions WHERE session_id='agent-abc'"
                            ).fetchone()[0] == indexer.TRANSCRIPT_MISSING
    finally:
        conn.close()
    print("  ok  migrate backfills archived_reason on legacy rows (idempotent, non-clobbering)")


def test_visible_predicate_includes_aged_out_excludes_noise():
    """Two named SQL predicates replace the `archived = 0` literal copy-pasted
    across the UI, stats and search: LIVE (a transcript exists on disk) and
    VISIBLE (what the user should see — live rows plus real sessions whose
    transcript aged out, never sidechain noise)."""
    conn = _temp_db()
    try:
        indexer.upsert(_header("live"), conn=conn)
        indexer.upsert(_header("aged"), conn=conn)
        indexer.archive("aged", reason=indexer.TRANSCRIPT_MISSING, conn=conn)
        indexer.upsert(_header("noise", turn_count=0, first_message=""), conn=conn)
        indexer.archive("noise", reason=indexer.NOT_A_SESSION, conn=conn)
        visible = {r[0] for r in conn.execute(f"SELECT session_id FROM sessions WHERE {indexer.VISIBLE}")}
        assert visible == {"live", "aged"}, visible
        live = {r[0] for r in conn.execute(f"SELECT session_id FROM sessions WHERE {indexer.LIVE}")}
        assert live == {"live"}, live
        # an archived row with no reason yet (mid-upgrade) must not leak through
        conn.execute("UPDATE sessions SET archived_reason = NULL WHERE session_id = 'noise'")
        visible = {r[0] for r in conn.execute(f"SELECT session_id FROM sessions WHERE {indexer.VISIBLE}")}
        assert visible == {"live", "aged"}, visible
    finally:
        conn.close()
    print("  ok  VISIBLE = live + transcript-missing; LIVE = archived=0 only")


def test_no_archived_sql_literal_outside_schema_layer():
    """`archived = 0` used to be copy-pasted into ~15 queries across 8 files, so
    a visibility rule would have to be re-derived at every site. The flag's SQL
    is spelled out only in indexer.py (predicates) and migrate-db.py (schema);
    everything else composes indexer.LIVE / VISIBLE / ARCHIVED_VISIBLE."""
    import re as _re
    pat = _re.compile(r"\b(WHERE|AND|OR)\s+(\w+\.)?archived\s*=\s*[01]\b")
    allowed = {"indexer.py", "scripts/migrate-db.py"}
    offenders = []
    for py in _REPO.rglob("*.py"):
        rel = py.relative_to(_REPO)
        if rel.parts[0] in (".venv", ".worktrees", "tests") or "__pycache__" in rel.parts:
            continue
        if str(rel) in allowed:
            continue
        for i, line in enumerate(py.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if pat.search(line):
                offenders.append(f"{rel}:{i}")
    assert not offenders, ("compose indexer.LIVE / VISIBLE instead of a literal:\n  "
                           + "\n  ".join(offenders))
    print("  ok  archived-flag SQL is spelled only in the schema layer")


def test_watcher_delete_archives_as_transcript_missing():
    """The watcher reaches archive() only after proving the deleted path WAS the
    canonical transcript — a real session aged out. It must say so, or the row
    is indistinguishable from sidechain noise and vanishes from the UI."""
    import watcher
    from sources.claude import ClaudeSource
    conn = _temp_db()
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    orig_connect, orig_log = indexer.connect, watcher._log
    try:
        with tempfile.TemporaryDirectory() as td:
            proj = Path(td) / "-proj-x"
            proj.mkdir()
            gone = proj / "sid-1.jsonl"          # never created: it was deleted
            indexer.upsert(_header("sid-1", project_path=str(proj)), conn=conn)
            conn.commit()
            indexer.connect = lambda *a, **k: orig_connect(db_path)
            watcher._log = lambda msg: None      # never touch ~/.session-browser
            ev = type("Ev", (), {"is_directory": False, "src_path": str(gone)})()
            watcher._Handler(ClaudeSource(td)).on_deleted(ev)
        row = conn.execute("SELECT archived, archived_reason FROM sessions "
                           "WHERE session_id='sid-1'").fetchone()
        assert row["archived"] == 1, dict(row)
        assert row["archived_reason"] == indexer.TRANSCRIPT_MISSING, dict(row)
    finally:
        indexer.connect, watcher._log = orig_connect, orig_log
        conn.close()
    print("  ok  watcher delete -> archived as transcript-missing (stays VISIBLE)")


def test_prune_classifies_each_stale_row():
    """prune-sessions sees BOTH kinds of dead row — sidechain noise, and real
    transcripts deleted while the watcher was down — so it must classify each
    one with the shared rule, never blanket-label the batch."""
    prune = _load_script("prune-sessions")
    conn = _temp_db()
    try:
        indexer.upsert(_header("agent-1", turn_count=0, first_message=""), conn=conn)
        indexer.upsert(_header("real-1", turn_count=9, first_message="ship it"), conn=conn)
        stale = conn.execute("SELECT * FROM sessions").fetchall()
        prune.archive_stale(stale, conn=conn)
        rows = dict(conn.execute("SELECT session_id, archived_reason FROM sessions "
                                 "WHERE archived = 1").fetchall())
        assert rows == {"agent-1": indexer.NOT_A_SESSION,
                        "real-1": indexer.TRANSCRIPT_MISSING}, rows
    finally:
        conn.close()
    print("  ok  prune archives noise as not-a-session, real rows as transcript-missing")


def test_find_archived_raw_prefers_newest_version():
    """archive_raw() keeps <sid>.jsonl plus <sid>@vN.jsonl on content change.
    Restore wants the newest — numerically (v10 > v2), not lexically — and the
    list view needs one directory walk for 500 rows, not 500 walks."""
    old = reasoning.ARCHIVE
    try:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            reasoning.ARCHIVE = root
            d = root / "raw" / "2026" / "05"
            d.mkdir(parents=True)
            for name in ("sid-a.jsonl", "sid-a@v2.jsonl", "sid-a@v10.jsonl", "sid-b.jsonl"):
                (d / name).write_text(name)
            assert reasoning.find_archived_raw("sid-a") == d / "sid-a@v10.jsonl"
            assert reasoning.find_archived_raw("sid-b") == d / "sid-b.jsonl"
            assert reasoning.find_archived_raw("nope") is None
            assert reasoning.archived_raw_index() == {"sid-a": d / "sid-a@v10.jsonl",
                                                      "sid-b": d / "sid-b.jsonl"}
            reasoning.ARCHIVE = root / "never-created"
            assert reasoning.find_archived_raw("sid-a") is None
            assert reasoning.archived_raw_index() == {}
    finally:
        reasoning.ARCHIVE = old
    print("  ok  find_archived_raw: newest @vN wins numerically; index is one walk")


def _cl_transcript(first_message: str, cwd: str = "/x") -> str:
    """A minimal Claude transcript parse_header/parse_full accept. Compact JSON:
    the adapter's turn counter prefilters on the '"type":"user"' substring."""
    recs = [
        {"type": "user", "cwd": cwd, "version": "2.0.0", "promptSource": "typed",
         "timestamp": "2026-05-01T10:00:00.000Z",
         "message": {"role": "user", "content": first_message}},
        {"type": "assistant", "timestamp": "2026-05-01T10:00:05.000Z",
         "message": {"model": "claude-test", "content": [{"type": "text", "text": "done"}]}},
    ]
    return "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in recs)


def test_restore_session_from_raw_archive():
    """Claude Code's cleanup deleted the transcript; the reasoning archive still
    holds a raw copy. Restore puts the NEWEST copy back where Claude Code looks
    for it (recreating the project dir if needed) and re-indexes, so the row
    goes live again and `cr <id>` / --resume work. The archive copy stays."""
    import restore
    from sources.claude import ClaudeSource
    old = reasoning.ARCHIVE
    conn = _temp_db()
    try:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            reasoning.ARCHIVE = root / "archive"
            projects = root / "projects"
            proj = projects / "-x"                      # deliberately NOT created
            raw = root / "archive" / "raw" / "2026" / "05"
            raw.mkdir(parents=True)
            (raw / "sid-r.jsonl").write_text(_cl_transcript("older copy"))
            (raw / "sid-r@v2.jsonl").write_text(_cl_transcript("restore me please"))
            indexer.upsert(_header("sid-r", project_path=str(proj), turn_count=1,
                                   first_message="restore me please"), conn=conn)
            indexer.archive("sid-r", indexer.TRANSCRIPT_MISSING, conn=conn)
            registry = {"claude": ClaudeSource(projects)}

            res = restore.restore_session("sid-r", conn=conn, registry=registry)
            assert res.status == "restored", res
            dest = proj / "sid-r.jsonl"
            assert res.path == dest and dest.exists(), res
            assert dest.read_text() == (raw / "sid-r@v2.jsonl").read_text()
            assert (raw / "sid-r@v2.jsonl").exists(), "archive copy must survive"
            row = conn.execute("SELECT archived, archived_reason, turn_count, cwd FROM sessions "
                               "WHERE session_id='sid-r'").fetchone()
            assert row["archived"] == 0 and row["archived_reason"] is None, dict(row)
            assert row["turn_count"] == 1 and row["cwd"] == "/x", dict(row)
            # second call: the live file is back, so nothing to copy — just re-index
            again = restore.restore_session("sid-r", conn=conn, registry=registry)
            assert again.status == "already-live", again
    finally:
        reasoning.ARCHIVE = old
        conn.close()
    print("  ok  restore: newest raw copy -> project dir, row resurrected, archive kept")


def test_restore_refuses_when_nothing_to_restore():
    """Every way restore can't proceed is a distinct, honest status — never a
    silent no-op and never a write into ~/.claude/projects it can't justify."""
    import restore
    from sources.claude import ClaudeSource
    from sources.copilot import CopilotSource
    old = reasoning.ARCHIVE
    conn = _temp_db()
    try:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            reasoning.ARCHIVE = root / "archive"          # exists but empty
            (root / "archive" / "raw").mkdir(parents=True)
            projects = root / "projects"
            registry = {"claude": ClaudeSource(projects),
                        "copilot": CopilotSource(root / "copilot")}

            assert restore.restore_session("ghost", conn=conn, registry=registry).status == "not-found"

            indexer.upsert(_header("agent-z", project_path=str(projects / "-p"),
                                   turn_count=0, first_message=""), conn=conn)
            indexer.archive("agent-z", indexer.NOT_A_SESSION, conn=conn)
            assert restore.restore_session("agent-z", conn=conn, registry=registry).status == "not-a-session"

            indexer.upsert(_header("no-copy", project_path=str(projects / "-p")), conn=conn)
            indexer.archive("no-copy", indexer.TRANSCRIPT_MISSING, conn=conn)
            assert restore.restore_session("no-copy", conn=conn, registry=registry).status == "no-raw-copy"

            # copilot can be restored now (events.jsonl under its state dir):
            # with an empty vault that is an honest no-raw-copy, not unsupported
            indexer.upsert(_header("cp-1", cli_source="copilot",
                                   project_path=str(root / "copilot" / "cp-1")), conn=conn)
            indexer.archive("cp-1", indexer.TRANSCRIPT_MISSING, conn=conn)
            assert restore.restore_session("cp-1", conn=conn, registry=registry).status == "no-raw-copy"
            # a source whose adapter has no restore_path() hook stays unsupported
            from types import SimpleNamespace
            registry["gemini"] = SimpleNamespace(name="gemini")
            indexer.upsert(_header("gm-1", cli_source="gemini", project_path="/g"), conn=conn)
            indexer.archive("gm-1", indexer.TRANSCRIPT_MISSING, conn=conn)
            assert restore.restore_session("gm-1", conn=conn, registry=registry).status == "unsupported"

            assert not (projects).exists(), "no refusal may write into the projects dir"
            for sid in ("agent-z", "no-copy", "cp-1", "gm-1"):
                assert conn.execute("SELECT archived FROM sessions WHERE session_id=?",
                                    (sid,)).fetchone()[0] == 1, sid
    finally:
        reasoning.ARCHIVE = old
        conn.close()
    print("  ok  restore refuses: not-found / not-a-session / no-raw-copy / unsupported")


def test_restore_plan_lists_only_aged_out_rows():
    """`restore-session.py --all --dry-run` is the first thing to run on a machine
    that lost sessions: it says which archived rows CAN come back before anyone
    counts on them. Noise rows never appear in the plan."""
    import restore
    from sources.claude import ClaudeSource
    old = reasoning.ARCHIVE
    conn = _temp_db()
    try:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            reasoning.ARCHIVE = root / "archive"
            raw = root / "archive" / "raw" / "2026" / "06"
            raw.mkdir(parents=True)
            (raw / "have.jsonl").write_text(_cl_transcript("x"))
            for sid, reason, kw in (("have", indexer.TRANSCRIPT_MISSING, {}),
                                    ("lost", indexer.TRANSCRIPT_MISSING, {}),
                                    ("noise", indexer.NOT_A_SESSION,
                                     dict(turn_count=0, first_message=""))):
                indexer.upsert(_header(sid, project_path=str(root / "p"), **kw), conn=conn)
                indexer.archive(sid, reason, conn=conn)
            indexer.upsert(_header("live"), conn=conn)
            plan = restore.plan(conn=conn, registry={"claude": ClaudeSource(root / "p")})
            got = {p["session_id"]: p["restorable"] for p in plan}
            assert got == {"have": True, "lost": False}, got
    finally:
        reasoning.ARCHIVE = old
        conn.close()
    print("  ok  restore.plan: aged-out rows with/without a raw copy; noise omitted")


# --- Flask API: archived view --------------------------------------------------
def _load_app():
    spec = _ilu.spec_from_file_location("sb_app", _REPO / "session-ui" / "app.py")
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_api_archived_state_and_visible_stats():
    """The Archived tab lists aged-out sessions — why, when, and whether a raw
    copy exists to restore from. The default list is unchanged. Usage stats
    count aged-out sessions (their spend was real) but never noise."""
    sb = _load_app()
    conn = _temp_db()
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    orig_connect, old_archive, old_sources = indexer.connect, reasoning.ARCHIVE, sb.SOURCES
    try:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            reasoning.ARCHIVE = root
            raw = root / "raw" / "2026" / "05"
            raw.mkdir(parents=True)
            (raw / "aged-has-copy.jsonl").write_text(_cl_transcript("x"))
            # restorable is decided per ROW: the recorded project_path must be
            # inside the adapter's tree, as it is for a session indexed here.
            from sources.claude import ClaudeSource
            proj = root / "claude" / "-Users-x-proj"
            sb.SOURCES = {"claude": ClaudeSource(root / "claude")}
            indexer.upsert(_header("live", project_path=str(proj)), conn=conn)
            conn.execute("UPDATE sessions SET cost_usd=1.0 WHERE session_id='live'")
            for sid in ("aged-has-copy", "aged-no-copy"):
                indexer.upsert(_header(sid, project_path=str(proj)), conn=conn)
                indexer.archive(sid, indexer.TRANSCRIPT_MISSING, conn=conn)
                conn.execute("UPDATE sessions SET cost_usd=2.0 WHERE session_id=?", (sid,))
            indexer.upsert(_header("agent-noise", turn_count=0, first_message=""), conn=conn)
            indexer.archive("agent-noise", indexer.NOT_A_SESSION, conn=conn)
            conn.execute("UPDATE sessions SET cost_usd=100.0 WHERE session_id='agent-noise'")
            conn.commit()
            indexer.connect = lambda *a, **k: orig_connect(db_path)
            c = sb.app.test_client()

            ids = [s["session_id"] for s in c.get("/api/sessions").get_json()]
            assert ids == ["live"], ids
            arch = c.get("/api/sessions?state=archived").get_json()
            got = {s["session_id"]: (s["archived_reason"], s["restorable"]) for s in arch}
            assert got == {"aged-has-copy": (indexer.TRANSCRIPT_MISSING, True),
                           "aged-no-copy": (indexer.TRANSCRIPT_MISSING, False)}, got
            assert all(s["archived_at"] for s in arch), arch
            stats = c.get("/api/stats").get_json()
            assert stats["total"] == 3 and stats["archived"] == 2, stats
            assert stats["by_source"] == {"claude": 3}, stats
            # the source pills must match the tab they sit above
            live = c.get("/api/stats?state=live").get_json()
            assert live["total"] == 1 and live["by_source"] == {"claude": 1}, live
            arch_stats = c.get("/api/stats?state=archived").get_json()
            assert arch_stats["total"] == 2 and arch_stats["by_source"] == {"claude": 2}, arch_stats
            assert arch_stats["archived"] == 2, arch_stats
            totals = c.get("/api/stats/timeseries").get_json()["totals"]
            assert totals["sessions"] == 3 and abs(totals["cost"] - 5.0) < 1e-9, totals
    finally:
        indexer.connect, reasoning.ARCHIVE, sb.SOURCES = orig_connect, old_archive, old_sources
        conn.close()
    print("  ok  /api/sessions?state=archived + stats count aged-out, never noise")


def test_api_restore_endpoint_and_resume_refusal():
    """Resume on an aged-out session is refused with the reason (there is no
    file for `claude --resume` to open); POST restore brings it back, after
    which resume works. Refusals map to honest HTTP statuses."""
    from sources.claude import ClaudeSource
    sb = _load_app()
    conn = _temp_db()
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    import os
    orig_connect, old_archive, old_sources = indexer.connect, reasoning.ARCHIVE, sb.SOURCES
    old_path = os.environ.get("PATH", "")
    try:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            reasoning.ARCHIVE = root / "archive"
            raw = root / "archive" / "raw" / "2026" / "05"
            raw.mkdir(parents=True)
            (raw / "aged.jsonl").write_text(_cl_transcript("bring me back"))
            projects = root / "projects"
            for sid in ("aged", "no-copy"):
                indexer.upsert(_header(sid, project_path=str(projects / "-x")), conn=conn)
                indexer.archive(sid, indexer.TRANSCRIPT_MISSING, conn=conn)
            conn.commit()
            indexer.connect = lambda *a, **k: orig_connect(db_path)
            sb.SOURCES = {"claude": ClaudeSource(projects)}
            # resume also requires the CLI on PATH now: stub it (CI has no claude)
            os.environ["PATH"] = str(_stub_bin(projects.parent / "bins", "claude").parent) + os.pathsep + old_path
            c = sb.app.test_client()

            r = c.get("/api/sessions/aged/resume")
            assert r.status_code == 409 and r.get_json()["archived_reason"] == indexer.TRANSCRIPT_MISSING, r.data
            # restore writes into the CLI's session tree: same CSRF guard as bridge
            assert c.post("/api/sessions/aged/restore").status_code == 403
            assert not (projects / "-x" / "aged.jsonl").exists()
            H = {"X-Requested-With": "session-browser"}
            r = c.post("/api/sessions/aged/restore", headers=H)
            assert r.status_code == 200 and r.get_json()["status"] == "restored", r.data
            assert (projects / "-x" / "aged.jsonl").exists()
            assert c.get("/api/sessions/aged/resume").status_code == 200
            assert [s["session_id"] for s in c.get("/api/sessions").get_json()] == ["aged"]

            assert c.post("/api/sessions/nope/restore", headers=H).status_code == 404
            r = c.post("/api/sessions/no-copy/restore", headers=H)
            assert r.status_code == 409 and r.get_json()["status"] == "no-raw-copy", r.data
    finally:
        indexer.connect, reasoning.ARCHIVE, sb.SOURCES = orig_connect, old_archive, old_sources
        os.environ["PATH"] = old_path
        conn.close()
    print("  ok  resume refused (409) while aged out; POST restore -> live -> resume ok")


def test_fts_indexes_archived_sessions_from_raw_copy():
    """Full-text search must keep working for aged-out sessions: build-fts reads
    their body from the reasoning archive's raw copy when the live transcript is
    gone. Rows with no copy, and noise rows, are skipped."""
    from sources.claude import ClaudeSource
    fts = _load_script("build-fts")
    conn = _temp_db()
    old = reasoning.ARCHIVE
    try:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            reasoning.ARCHIVE = root
            raw = root / "raw" / "2026" / "05"
            raw.mkdir(parents=True)
            (raw / "aged.jsonl").write_text(_cl_transcript("zebra migration strategy"))
            for sid, reason, kw in (("aged", indexer.TRANSCRIPT_MISSING, {}),
                                    ("lost", indexer.TRANSCRIPT_MISSING, {}),
                                    ("noise", indexer.NOT_A_SESSION,
                                     dict(turn_count=0, first_message=""))):
                indexer.upsert(_header(sid, **kw), conn=conn)
                indexer.archive(sid, reason, conn=conn)
            n = fts.index_archived(conn, {"claude": ClaudeSource(root / "projects")})
            assert n == 1, n
            hits = [r[0] for r in conn.execute(
                "SELECT session_id FROM sessions_fts WHERE sessions_fts MATCH 'zebra'")]
            assert hits == ["aged"], hits
    finally:
        reasoning.ARCHIVE = old
        conn.close()
    print("  ok  build-fts indexes aged-out sessions from their raw archive copy")


# --- schema self-heal: code ahead of the registry --------------------------
def _old_schema_db(path: Path) -> None:
    """A registry as it existed before archived_reason — built WITHOUT
    indexer.connect(), which would upgrade it."""
    mig = _load_script("migrate-db")
    conn = sqlite3.connect(str(path))
    conn.executescript(mig.BASE_DDL)
    for col in mig.ADDITIVE_COLUMNS:
        if not col.startswith("archived_"):
            mig._add_column_if_missing(conn, "sessions", col)
    conn.execute("INSERT INTO sessions (session_id, archived, turn_count, first_message, "
                 "cli_source, project_path) VALUES ('legacy', 1, 4, 'x', 'claude', '/nowhere')")
    conn.commit()
    conn.close()


def test_connect_self_heals_registry_behind_the_code():
    """After a `git pull`, the Stop hook and watcher upsert BEFORE the nightly
    refresh has migrated — and the upsert SQL now names archived_reason, so
    they'd fail until 01:00. connect() reads PRAGMA user_version (one integer)
    and migrates only when the registry is behind."""
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "old.db"
        _old_schema_db(db)
        raw = sqlite3.connect(str(db))
        assert "archived_reason" not in {r[1] for r in raw.execute("PRAGMA table_info(sessions)")}
        assert raw.execute("PRAGMA user_version").fetchone()[0] == 0
        raw.close()
        conn = indexer.connect(db)
        try:
            indexer.upsert(_header("fresh"), conn=conn)          # the hook's hot path
            assert conn.execute("PRAGMA user_version").fetchone()[0] == indexer.SCHEMA_VERSION
            assert conn.execute("SELECT archived_reason FROM sessions WHERE session_id='legacy'"
                                ).fetchone()[0] == indexer.TRANSCRIPT_MISSING
        finally:
            conn.close()
    print("  ok  connect() migrates a registry the code is ahead of (PRAGMA user_version)")


def test_restore_cli_plans_on_unmigrated_registry():
    """`restore-session.py --all` is the first command to run on a laptop that
    lost sessions — before any refresh has migrated its registry. It must
    print the plan, not crash on a missing column."""
    import os
    import subprocess
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "old.db"
        _old_schema_db(db)
        out = subprocess.run([sys.executable, str(_REPO / "scripts" / "restore-session.py"), "--all"],
                             capture_output=True, text=True, cwd=str(_REPO),
                             env={**os.environ, "SB_DB": str(db)})
        assert out.returncode == 0, out.stderr
        assert "legacy" in out.stdout and "1 archived session" in out.stdout, out.stdout
    print("  ok  restore-session.py --all plans on a not-yet-migrated registry")


# --- opencode adapter: SQLite -> per-root-session JSONL mirror ----------------
# OpenCode (>= v1.2.0) keeps every session in one SQLite DB (WAL):
# session / message / part rows with JSON `data` blobs. The adapter projects
# each ROOT session (children embedded) into <mirror>/<ses_id>.jsonl so every
# downstream consumer keeps its one-file-per-session assumption. Shapes below
# follow packages/schema/src/v1/session.ts and the live 1.18.15 DB.
_OC_ROOT = "ses_fd7037a16ffeRyoMOVVyFqv3xY"
_OC_CHILD = "ses_fd7037a16ffdAbCdEfGhIjKlMn"
_OC_T0 = 1_785_542_400_000          # 2026-08-01T00:00:00.000Z, epoch ms
_OC_MODEL = ("opencode", "minimax-m2.5-free")


def _oc_schema(conn) -> None:
    conn.executescript("""
    PRAGMA journal_mode=WAL;
    CREATE TABLE project(id TEXT PRIMARY KEY, worktree TEXT NOT NULL, vcs TEXT, name TEXT,
        time_created INTEGER, time_updated INTEGER);
    CREATE TABLE session(id TEXT PRIMARY KEY, project_id TEXT NOT NULL, parent_id TEXT,
        slug TEXT NOT NULL, directory TEXT NOT NULL, path TEXT, title TEXT NOT NULL,
        version TEXT NOT NULL, share_url TEXT, summary_additions INTEGER, summary_deletions INTEGER,
        summary_files INTEGER, summary_diffs TEXT, revert TEXT, permission TEXT, metadata TEXT,
        agent TEXT, model TEXT, cost REAL NOT NULL DEFAULT 0,
        tokens_input INTEGER NOT NULL DEFAULT 0, tokens_output INTEGER NOT NULL DEFAULT 0,
        tokens_reasoning INTEGER NOT NULL DEFAULT 0, tokens_cache_read INTEGER NOT NULL DEFAULT 0,
        tokens_cache_write INTEGER NOT NULL DEFAULT 0, time_created INTEGER NOT NULL,
        time_updated INTEGER, time_compacting INTEGER, time_archived INTEGER, workspace_id TEXT);
    CREATE TABLE message(id TEXT PRIMARY KEY, session_id TEXT NOT NULL, time_created INTEGER NOT NULL,
        time_updated INTEGER, data TEXT NOT NULL);
    CREATE TABLE part(id TEXT PRIMARY KEY, message_id TEXT NOT NULL, session_id TEXT NOT NULL,
        time_created INTEGER NOT NULL, time_updated INTEGER, data TEXT NOT NULL);
    CREATE TABLE session_message(id TEXT PRIMARY KEY, session_id TEXT, type TEXT, seq INTEGER, data TEXT);
    CREATE TABLE migration(id TEXT PRIMARY KEY, time_completed INTEGER);   -- the NAME lives in `id`
    INSERT INTO migration VALUES ('20260622202450_simplify_session_input', 1786909685967);
    """)


def _oc_seed(conn, *, root=_OC_ROOT, child=_OC_CHILD,
             title="New session - 2026-08-01T10:00:00.000Z", version="1.18.15") -> dict:
    """A root session (2 real user turns, a compaction pair, an aborted assistant
    message, a part-less message) plus one child spawned by the task tool."""
    T = _OC_T0
    prov, mdl = _OC_MODEL
    conn.execute("INSERT INTO project VALUES ('proj-hash', '/Users/x/proj', 'git', 'proj', ?, ?)", (T, T))
    conn.execute("INSERT INTO session (id, project_id, parent_id, slug, directory, path, title, version, "
                 "time_created, time_updated, permission, model) VALUES (?, 'proj-hash', NULL, 'kind-canyon', "
                 "'/Users/x/proj', 'Users/x/proj', ?, ?, ?, ?, ?, ?)",
                 (root, title, version, T, T + 60_000,
                  json.dumps([{"permission": "question", "action": "deny", "pattern": "*"}]),
                  json.dumps({"id": mdl, "providerID": prov})))
    conn.execute("INSERT INTO session (id, project_id, parent_id, slug, directory, path, title, version, "
                 "time_created, time_updated) VALUES (?, 'proj-hash', ?, 'tiny-fox', '/Users/x/proj', "
                 "'Users/x/proj', 'Chunk 1 - scan (@explore subagent)', ?, ?, ?)",
                 (child, root, version, T + 3000, T + 4000))

    def msg(sid, mid, t, data):
        conn.execute("INSERT INTO message VALUES (?, ?, ?, ?, ?)", (mid, sid, t, t, json.dumps(data)))

    def part(sid, mid, pid, t, data):
        conn.execute("INSERT INTO part VALUES (?, ?, ?, ?, ?, ?)", (pid, mid, sid, t, t, json.dumps(data)))

    def assistant(parent, t, cost, inp, out, reasoning=0, cr=0, cw=0, **extra):
        return {"parentID": parent, "role": "assistant", "mode": "build", "agent": "build",
                "path": {"cwd": "/Users/x/proj", "root": "/Users/x/proj"}, "cost": cost,
                "tokens": {"total": inp + out + reasoning, "input": inp, "output": out,
                           "reasoning": reasoning, "cache": {"read": cr, "write": cw}},
                "modelID": mdl, "providerID": prov,
                "time": {"created": t, "completed": t + 3000}, "finish": "stop", **extra}
    user = {"role": "user", "agent": "build", "model": {"providerID": prov, "modelID": mdl}}

    msg(root, "msg_u1", T + 1000, {**user, "time": {"created": T + 1000}})
    part(root, "msg_u1", "prt_01", T + 1000, {"type": "text", "text": "why is opencode missing?"})
    part(root, "msg_u1", "prt_02", T + 1001, {"type": "text", "text": "hidden", "synthetic": True})
    msg(root, "msg_a1", T + 2000, assistant("msg_u1", T + 2000, 0.0125, 1000, 200, 100, 50, 10))
    part(root, "msg_a1", "prt_03", T + 2000, {"type": "step-start"})
    part(root, "msg_a1", "prt_04", T + 2001, {"type": "reasoning", "text": "check the db",
                                             "time": {"start": T + 2001, "end": T + 2002},
                                             "metadata": {"anthropic": {"signature": "abc"}}})
    part(root, "msg_a1", "prt_05", T + 2003, {"type": "text", "text": "the storage moved",
                                             "time": {"start": T + 2003, "end": T + 2004}})
    part(root, "msg_a1", "prt_06", T + 2005, {"type": "tool", "tool": "bash", "callID": "call_1",
                                             "state": {"status": "completed", "input": {"command": "ls"},
                                                       "output": "a b", "title": "ls", "metadata": {},
                                                       "time": {"start": T + 2005, "end": T + 2006}}})
    part(root, "msg_a1", "prt_07", T + 2007, {"type": "subtask", "prompt": "scan the repo",
                                             "description": "scan", "agent": "explore"})
    part(root, "msg_a1", "prt_08", T + 2008, {"type": "step-finish", "reason": "stop", "cost": 0.0125,
                                             "tokens": {"total": 1300, "input": 1000, "output": 200,
                                                        "reasoning": 100, "cache": {"read": 50, "write": 10}}})
    # compaction pair: a synthetic user message carrying a compaction part and
    # the assistant summary — real spend, but not conversation turns
    msg(root, "msg_uc", T + 5500, {**user, "time": {"created": T + 5500}})
    part(root, "msg_uc", "prt_09", T + 5500, {"type": "compaction", "auto": True})
    msg(root, "msg_a2", T + 6000, assistant("msg_uc", T + 6000, 0.001, 100, 20, summary=True))
    part(root, "msg_a2", "prt_10", T + 6000, {"type": "text", "text": "Summary of the conversation so far"})
    msg(root, "msg_u2", T + 7000, {**user, "time": {"created": T + 7000}})
    part(root, "msg_u2", "prt_11", T + 7000, {"type": "text", "text": "fix it"})
    # aborted turn: assistant with an error and no parts (old rows look like this)
    msg(root, "msg_a3", T + 8000, {**assistant("msg_u2", T + 8000, 0.0, 0, 0),
                                   "error": {"name": "MessageAbortedError", "data": {}}})
    # child session (task tool)
    msg(child, "msg_c1", T + 3000, {**user, "time": {"created": T + 3000}})
    part(child, "msg_c1", "prt_12", T + 3000, {"type": "text", "text": "scan the repo"})
    msg(child, "msg_c2", T + 4000, assistant("msg_c1", T + 4000, 0.005, 400, 80))
    part(child, "msg_c2", "prt_13", T + 4000, {"type": "text", "text": "found 3 files"})
    conn.commit()
    return {"root": root, "child": child}


def _oc_source(tmp: Path, seed=True):
    """OpenCodeSource over a seeded temp DB at <tmp>/data/opencode.db."""
    from sources.opencode import OpenCodeSource
    data = tmp / "data"
    data.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(data / "opencode.db"))
    _oc_schema(conn)
    if seed:
        _oc_seed(conn)
    conn.close()
    return OpenCodeSource(data_dir=data, mirror_dir=tmp / "mirror")


def test_opencode_sync_projects_root_sessions_to_mirror():
    """One JSONL per ROOT session; the child is embedded, never a file of its
    own; line 1 carries the export-shaped info, the children, and the stats
    every consumer reads (turns, first message, per-model spend rolled up
    across the tree). The stem MUST equal the session id — prune-sessions
    archives every row whose session_id_for_path disagrees with the header."""
    with tempfile.TemporaryDirectory() as td:
        src = _oc_source(Path(td))
        files = list(src.discover())
        assert [f.name for f in files] == [f"{_OC_ROOT}.jsonl"], files
        mirror = files[0]
        assert mirror.parent == src.mirror_dir
        lines = mirror.read_text().splitlines()
        head = json.loads(lines[0])
        assert head["type"] == "session" and head["schema"] == 1
        assert head["info"]["id"] == _OC_ROOT and head["info"]["projectID"] == "proj-hash"
        assert head["info"]["directory"] == "/Users/x/proj"
        assert head["info"]["time"] == {"created": _OC_T0, "updated": _OC_T0 + 60_000}
        assert [c["id"] for c in head["children"]] == [_OC_CHILD]
        assert head["children"][0]["parentID"] == _OC_ROOT
        st = head["stats"]
        assert st["turn_count"] == 2 and st["first_message"] == "why is opencode missing?", st
        assert st["title"] is None, st                       # placeholder title
        assert st["model_used"] == "opencode/minimax-m2.5-free"
        assert st["models"] == {"opencode/minimax-m2.5-free": {
            "input": 1500, "output": 300, "reasoning": 100, "cache_read": 50, "cache_write": 10,
            "cost": 0.0185}}, st["models"]
        assert abs(st["cost_usd"] - 0.0185) < 1e-9
        assert st["message_count"] == 8 and st["child_count"] == 1
        assert st["start_time"] == _OC_T0 and st["last_activity"] == _OC_T0 + 60_000
        # message lines: root first (chronological), then the child's
        msgs = [json.loads(l) for l in lines[1:]]
        assert all(m["type"] == "message" for m in msgs)
        assert [m["info"]["id"] for m in msgs] == \
            ["msg_u1", "msg_a1", "msg_uc", "msg_a2", "msg_u2", "msg_a3", "msg_c1", "msg_c2"]
        assert msgs[0]["session"] == _OC_ROOT and msgs[-1]["session"] == _OC_CHILD
        assert msgs[0]["info"]["sessionID"] == _OC_ROOT and msgs[0]["parts"][0]["messageID"] == "msg_u1"
        assert msgs[1]["parts"][3]["tool"] == "bash"      # parts in time order, export-shaped
        assert (src.mirror_dir / ".manifest.json").exists()
    print("  ok  opencode sync: one JSONL per root, child embedded, stats rolled up, stem == id")


def test_opencode_parse_header_reads_line_one_only():
    """The header lives on line 1 so parse_header stays cheap on multi-MB
    sessions; fields map from Session.Info + stats; a placeholder title is no
    title; times go through to_iso_utc so cross-source ordering holds."""
    with tempfile.TemporaryDirectory() as td:
        src = _oc_source(Path(td))
        path = next(iter(src.discover()))
        h = src.parse_header(path)
        assert h is not None
        assert h.session_id == _OC_ROOT and h.cli_source == "opencode"
        assert h.project_path == str(src.mirror_dir)
        assert h.cwd == "/Users/x/proj" and h.folder_name == "proj"
        assert h.start_time == "2026-08-01T00:00:00.000Z", h.start_time
        assert h.last_activity == to_iso_utc(_OC_T0 + 60_000)
        assert h.first_message == "why is opencode missing?" and h.turn_count == 2
        assert h.title is None, h.title
        assert h.model_used == "opencode/minimax-m2.5-free" and h.cli_version == "1.18.15"
        first = path.read_text().splitlines()[0]
        path.write_text(first + "\n")            # only line 1 left
        assert src.parse_header(path) == h
        path.write_text("not json\n")
        assert src.parse_header(path) is None
    with tempfile.TemporaryDirectory() as td:
        src = _oc_source(Path(td), seed=False)
        conn = sqlite3.connect(str(src.db_path))
        _oc_seed(conn, title="Refactor the retry loop")
        conn.close()
        h = src.parse_header(next(iter(src.discover())))
        assert h.title == "Refactor the retry loop"
    print("  ok  opencode parse_header: line 1 only; fields, times, placeholder title")


def test_opencode_session_id_for_path_variants():
    """Only <mirror>/ses_<26 chars>.jsonl is a session: never the manifest, a
    half-written .tmp, the DB's WAL, or an archive copy's @vN name."""
    from sources.opencode import OpenCodeSource
    src = OpenCodeSource(data_dir="/nonexistent/data", mirror_dir="/nonexistent/mirror")
    m = Path("/nonexistent/mirror")
    assert src.session_id_for_path(m / f"{_OC_ROOT}.jsonl") == _OC_ROOT
    for bad in (".manifest.json", f"{_OC_ROOT}.jsonl.tmp", "opencode.db-wal", "opencode.db",
                f"{_OC_ROOT}@v2.jsonl", "ses_short.jsonl", "notes.jsonl", f"{_OC_ROOT}.json"):
        assert src.session_id_for_path(m / bad) is None, bad
    print("  ok  opencode session_id_for_path: ses_<26>.jsonl only")


def _oc_add_child_message(db: Path, cost=0.002, inp=10, out=5, mid="msg_c3") -> None:
    conn = sqlite3.connect(str(db))
    conn.execute("INSERT INTO message VALUES (?, ?, ?, ?, ?)", (mid, _OC_CHILD, _OC_T0 + 9000, _OC_T0 + 9000,
                 json.dumps({"parentID": "msg_c1", "role": "assistant", "cost": cost,
                             "tokens": {"input": inp, "output": out, "reasoning": 0, "cache": {"read": 0, "write": 0}},
                             "modelID": _OC_MODEL[1], "providerID": _OC_MODEL[0],
                             "time": {"created": _OC_T0 + 9000}})))
    conn.commit()
    conn.close()


def test_opencode_manifest_skips_unchanged_and_rewrites_updated():
    """Rewrites are driven by a fingerprint spanning the whole tree (a child
    can update after its root; old rows have NULL time_updated). A lost or
    corrupt manifest costs one full resync, nothing more."""
    with tempfile.TemporaryDirectory() as td:
        src = _oc_source(Path(td))
        assert src.sync().written == [_OC_ROOT]
        path = src.mirror_dir / f"{_OC_ROOT}.jsonl"
        mtime = path.stat().st_mtime_ns
        r = src.sync()
        assert r.written == [] and r.skipped == 1 and path.stat().st_mtime_ns == mtime, r
        _oc_add_child_message(src.db_path)
        assert src.sync().written == [_OC_ROOT]
        head = json.loads(path.read_text().splitlines()[0])
        assert head["stats"]["message_count"] == 9 and abs(head["stats"]["cost_usd"] - 0.0205) < 1e-9
        body = path.read_text().splitlines()[1:]
        (src.mirror_dir / ".manifest.json").write_text("{corrupt")
        assert src.sync().written == [_OC_ROOT]
        assert path.read_text().splitlines()[1:] == body
        assert src.sync().written == [] and src.sync(force=True).written == [_OC_ROOT]
    print("  ok  opencode manifest: skip unchanged, child change rewrites root, corrupt -> resync")


def _oc_wipe(db: Path) -> None:
    conn = sqlite3.connect(str(db))
    conn.executescript("DELETE FROM part; DELETE FROM message; DELETE FROM session;")
    conn.close()


def test_opencode_deleted_session_archives_raw_then_unlinks():
    """`opencode session delete` cascades. The mirror file is the last copy, so
    sync hands it to the archiver BEFORE unlinking (default: reasoning.archive_raw
    into the versioned raw vault), keeps it if archiving fails, and removes
    nothing at all when the DB cannot be read. The unlink is what makes the
    watcher archive the row as transcript-missing -> Archived tab -> Restore."""
    old = reasoning.ARCHIVE
    try:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # explicit archiver: called with the live file and its header, first
            src = _oc_source(root / "a")
            src.sync()
            path = src.mirror_dir / f"{_OC_ROOT}.jsonl"
            _oc_wipe(src.db_path)
            calls = []
            r = src.sync(on_delete=lambda p, h: calls.append((p, h["session_id"], p.exists())))
            assert r.removed == [_OC_ROOT] and not path.exists(), r
            assert calls == [(path, _OC_ROOT, True)], calls
            assert _OC_ROOT not in src._load_manifest()
            # default archiver = the raw vault (month from last_activity: 2026/08)
            reasoning.ARCHIVE = root / "archive"
            src = _oc_source(root / "b")
            src.sync()
            _oc_wipe(src.db_path)
            r = src.sync()
            assert r.removed == [_OC_ROOT], r
            copy = reasoning.find_archived_raw(_OC_ROOT)
            assert copy is not None and copy.parent == root / "archive" / "raw" / "2026" / "08", copy
            assert json.loads(copy.read_text().splitlines()[0])["info"]["id"] == _OC_ROOT
            # an archiver that fails keeps the file — never destroy the last copy
            src = _oc_source(root / "c")
            src.sync()
            path = src.mirror_dir / f"{_OC_ROOT}.jsonl"
            _oc_wipe(src.db_path)

            def boom(p, h):
                raise OSError("disk full")
            r = src.sync(on_delete=boom)
            assert path.exists() and r.removed == [] and any("disk full" in w for w in r.warnings), r
            # unreadable DB: nothing is removed
            src = _oc_source(root / "d")
            src.sync()
            path = src.mirror_dir / f"{_OC_ROOT}.jsonl"
            src.db_path.unlink()
            r = src.sync(on_delete=boom)
            assert r.db_missing and path.exists() and r.removed == []
    finally:
        reasoning.ARCHIVE = old
    print("  ok  opencode deletion: archive_raw first, then unlink; failures keep the file")


def test_watcher_delete_archives_opencode_row():
    """The mirror file vanishing (sync unlinked it) goes through the watcher's
    ordinary delete path: the row is archived as transcript-missing."""
    import watcher
    conn = _temp_db()
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    orig_connect, orig_log = indexer.connect, watcher._log
    try:
        with tempfile.TemporaryDirectory() as td:
            src = _oc_source(Path(td))
            path = next(iter(src.discover()))
            indexer.upsert(src.parse_header(path), conn=conn)
            conn.commit()
            path.unlink()
            indexer.connect = lambda *a, **k: orig_connect(db_path)
            watcher._log = lambda msg: None
            ev = type("Ev", (), {"is_directory": False, "src_path": str(path)})()
            watcher._Handler(src).on_deleted(ev)
        row = conn.execute("SELECT archived, archived_reason FROM sessions WHERE session_id=?",
                           (_OC_ROOT,)).fetchone()
        assert row["archived"] == 1 and row["archived_reason"] == indexer.TRANSCRIPT_MISSING, dict(row)
    finally:
        indexer.connect, watcher._log = orig_connect, orig_log
        conn.close()
    print("  ok  watcher: deleted opencode mirror file -> row archived transcript-missing")


def test_opencode_parse_full_turn_order_and_skips_noise():
    """Root turns only, in order: synthetic text, the compaction pair, the
    part-less aborted message and the child's own turns are not turns. Tool
    calls carry {name, input} (build-fts indexes `input`); the child shows up
    as the `subtask` call that spawned it."""
    with tempfile.TemporaryDirectory() as td:
        src = _oc_source(Path(td))
        parsed = src.parse_full(next(iter(src.discover())))
        assert parsed is not None and parsed.header.session_id == _OC_ROOT
        got = [(t.role, t.content, t.tool_calls) for t in parsed.turns]
        assert got == [
            ("user", "why is opencode missing?", []),
            ("assistant", "the storage moved", [{"name": "bash", "input": "command=ls"},
                                                {"name": "subtask", "input": "agent=explore: scan the repo"}]),
            ("user", "fix it", []),
        ], got
    print("  ok  opencode parse_full: root turns in order; compaction/synthetic/child/aborted skipped")


def test_opencode_unknown_part_types_tolerated():
    """OpenCode adds part types over time; an unknown one must neither crash
    nor be dropped from the mirror (the mirror is the lossless backup)."""
    with tempfile.TemporaryDirectory() as td:
        src = _oc_source(Path(td))
        conn = sqlite3.connect(str(src.db_path))
        conn.execute("INSERT INTO part VALUES ('prt_99', 'msg_a1', ?, ?, ?, ?)",
                     (_OC_ROOT, _OC_T0 + 2009, _OC_T0 + 2009, json.dumps({"type": "hologram", "beam": 1})))
        conn.commit()
        conn.close()
        path = next(iter(src.discover()))
        parsed = src.parse_full(path)
        assert [t.content for t in parsed.turns] == ["why is opencode missing?", "the storage moved", "fix it"]
        a1 = next(json.loads(l) for l in path.read_text().splitlines()[1:] if json.loads(l)["info"]["id"] == "msg_a1")
        assert {"type": "hologram", "beam": 1, "id": "prt_99", "messageID": "msg_a1", "sessionID": _OC_ROOT} in a1["parts"]
    print("  ok  opencode unknown part types: tolerated and kept verbatim")


def test_opencode_unrecognised_part_schema_is_not_silent():
    """Messages exist but no part type is one we know: index with 0 turns and
    warn, rather than return None and vanish (how the codex dialect switch
    went unnoticed for weeks)."""
    import io
    from contextlib import redirect_stderr
    with tempfile.TemporaryDirectory() as td:
        src = _oc_source(Path(td), seed=False)
        conn = sqlite3.connect(str(src.db_path))
        T = _OC_T0
        conn.execute("INSERT INTO project VALUES ('p', '/x', 'git', 'x', ?, ?)", (T, T))
        conn.execute("INSERT INTO session (id, project_id, slug, directory, title, version, time_created) "
                     "VALUES (?, 'p', 'odd-slug', '/x', 'Something real', '9.0.0', ?)", (_OC_ROOT, T))
        for i, role in enumerate(("user", "assistant")):
            conn.execute("INSERT INTO message VALUES (?, ?, ?, ?, ?)",
                         (f"msg_{i}", _OC_ROOT, T + i, T + i, json.dumps({"role": role, "time": {"created": T + i}})))
            conn.execute("INSERT INTO part VALUES (?, ?, ?, ?, ?, ?)",
                         (f"prt_{i}", f"msg_{i}", _OC_ROOT, T + i, T + i, json.dumps({"type": "blob", "v": i})))
        conn.commit()
        conn.close()
        err = io.StringIO()
        with redirect_stderr(err):
            files = list(src.discover())
            h = src.parse_header(files[0])
        assert h is not None and h.turn_count == 0 and h.title == "Something real", h
        assert "part type" in err.getvalue(), err.getvalue()
    print("  ok  opencode unrecognised part schema: indexed with 0 turns + warning, not dropped")


def test_opencode_parse_full_on_raw_archive_copy():
    """build-fts.index_archived and Restore read the raw-archive copy, named
    <sid>@vN.jsonl in a different directory: the session id must come from
    line 1, and nothing may depend on the DB, manifest or mirror_dir."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        src = _oc_source(root)
        live = next(iter(src.discover()))
        want = [(t.role, t.content, t.tool_calls) for t in src.parse_full(live).turns]
        raw = root / "archive" / "raw" / "2026" / "08"
        raw.mkdir(parents=True)
        copy = raw / f"{_OC_ROOT}@v2.jsonl"
        copy.write_bytes(live.read_bytes())
        # the DB and mirror are gone: the copy must still parse on its own
        src.db_path.unlink()
        live.unlink()
        h = src.parse_header(copy)
        assert h is not None and h.session_id == _OC_ROOT and h.turn_count == 2, h
        assert [(t.role, t.content, t.tool_calls) for t in src.parse_full(copy).turns] == want
    print("  ok  opencode parse_full on a raw-archive copy: id from line 1, no DB needed")


def test_opencode_v2_tables_not_silently_empty():
    """`message` empty while `session_message` has rows = OpenCode moved to its
    v2 store: warn by name, keep serving the existing mirror, never pretend the
    user has no sessions. A missing column is projected around and warned."""
    import io
    from contextlib import redirect_stderr
    with tempfile.TemporaryDirectory() as td:
        src = _oc_source(Path(td))
        assert len(list(src.discover())) == 1
        conn = sqlite3.connect(str(src.db_path))
        conn.executescript("DELETE FROM part; DELETE FROM message; "
                           "INSERT INTO session_message VALUES ('sm1', 'ses_x', 'user', 1, '{}');")
        conn.close()
        err = io.StringIO()
        with redirect_stderr(err):
            r = src.sync(force=True)
        assert any("session_message" in w for w in r.warnings) and "session_message" in err.getvalue(), r
        assert r.written == [] and r.removed == []
        assert len(list(src.mirror_dir.glob("ses_*.jsonl"))) == 1     # still served
    with tempfile.TemporaryDirectory() as td:
        src = _oc_source(Path(td), seed=False)
        conn = sqlite3.connect(str(src.db_path))
        conn.executescript("ALTER TABLE session DROP COLUMN directory;")
        conn.execute("INSERT INTO project VALUES ('p', '/x', 'git', 'x', 1, 1)")
        conn.execute("INSERT INTO session (id, project_id, slug, title, version, time_created) "
                     "VALUES (?, 'p', 's', 'T', '1.0', ?)", (_OC_ROOT, _OC_T0))
        conn.execute("INSERT INTO message VALUES ('m1', ?, ?, ?, ?)",
                     (_OC_ROOT, _OC_T0, _OC_T0, json.dumps({"role": "user", "time": {"created": _OC_T0}})))
        conn.execute("INSERT INTO part VALUES ('p1', 'm1', ?, ?, ?, ?)",
                     (_OC_ROOT, _OC_T0, _OC_T0, json.dumps({"type": "text", "text": "hello"})))
        conn.commit()
        conn.close()
        err = io.StringIO()
        with redirect_stderr(err):
            files = list(src.discover())
        assert "directory" in err.getvalue(), err.getvalue()
        h = src.parse_header(files[0])
        assert h is not None and h.cwd == "" and h.folder_name == "" and h.turn_count == 1, h
    print("  ok  opencode schema drift: v2 store and missing columns warn, never silent")


def test_opencode_available_without_binary_on_path():
    """Availability means 'there is something to read', never 'the binary is on
    PATH' — the watcher runs under a PATH where the binary may be absent, and
    a mirror keeps serving after OpenCode is uninstalled. And the indexing path
    must never spawn the binary (it rewrote the WAL when merely asked for a path)."""
    import os
    import subprocess
    from sources.opencode import OpenCodeSource
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        old_path = os.environ["PATH"]
        os.environ["PATH"] = str(root / "empty")
        try:
            assert OpenCodeSource(data_dir=root / "none", mirror_dir=root / "m").is_available() is False
            src = _oc_source(root)
            assert src.is_available() is True                # DB exists
            assert src.resume_command("ses_abc") == "opencode --session ses_abc"
            files = list(src.discover())
            src.db_path.unlink()
            assert OpenCodeSource(data_dir=root / "data", mirror_dir=root / "mirror").is_available() is True  # mirror only
        finally:
            os.environ["PATH"] = old_path
        orig_run, orig_popen = subprocess.run, subprocess.Popen

        def forbidden(*a, **k):
            raise AssertionError("sync/discover must never spawn a subprocess")
        subprocess.run = subprocess.Popen = forbidden
        try:
            src = _oc_source(root / "again")
            assert len(list(src.discover())) == 1
            _oc_wipe(src.db_path)
            src.sync(on_delete=lambda p, h: None)
        finally:
            subprocess.run, subprocess.Popen = orig_run, orig_popen
    print("  ok  opencode is_available without the binary; indexing never shells out")


def test_opencode_cost_extractor_returns_authoritative_cost():
    """OpenCode stores per-message USD (from its models.dev catalogue) for
    every provider — GLM, Qwen, MiniMax, Kimi are unpriceable by pricing.json.
    The extractor returns that cost as authoritative (3-tuple) and process()
    writes it without an 'unknown model' warning. Reasoning tokens fold into
    output, the copilot precedent."""
    import io
    from contextlib import redirect_stderr
    cc = _load_script("compute-costs")
    with tempfile.TemporaryDirectory() as td:
        src = _oc_source(Path(td))
        path = next(iter(src.discover()))
        totals, per_model, cost = cc._usage_opencode(path)
        assert dict(totals) == {"input": 1500, "output": 400, "cache_read": 50, "cache_write": 10}, totals
        assert {k: dict(v) for k, v in per_model.items()} == {
            "opencode/minimax-m2.5-free": {"input": 1500, "output": 400, "cache_read": 50, "cache_write": 10}}
        assert abs(cost - 0.0185) < 1e-9
        conn = _temp_db()
        try:
            indexer.upsert(src.parse_header(path), conn=conn)
            err = io.StringIO()
            with redirect_stderr(err):
                out = cc.process(path, src, conn)
            assert out and abs(out["cost"] - 0.0185) < 1e-9, out
            assert "unknown model" not in err.getvalue(), err.getvalue()
            row = conn.execute("SELECT input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, "
                               "model_used, models_used, cost_usd FROM sessions WHERE session_id=?",
                               (_OC_ROOT,)).fetchone()
            assert tuple(row)[:4] == (1500, 400, 50, 10), tuple(row)
            assert row["model_used"] == "opencode/minimax-m2.5-free"
            assert json.loads(row["models_used"]) == ["opencode/minimax-m2.5-free"]
            assert abs(row["cost_usd"] - 0.0185) < 1e-9
        finally:
            conn.close()
    print("  ok  compute-costs: opencode extractor is authoritative (3-tuple), no pricing warning")


def test_cost_process_accepts_two_tuple_extractors():
    """The three existing extractors return (totals, per_model) and are priced
    via pricing.json; that contract must survive the 3-tuple extension."""
    cc = _load_script("compute-costs")

    class Fake:
        name = "fake"

        def parse_header(self, path):
            return _header("fk-1", cli_source="fake", model_used=None)
    toks = {"input": 1_000_000, "output": 100_000, "cache_read": 0, "cache_write": 0}
    cc._EXTRACTORS["fake"] = lambda path: (dict(toks), {"claude-sonnet-5": dict(toks)})
    conn = _temp_db()
    try:
        indexer.upsert(_header("fk-1", cli_source="fake"), conn=conn)
        out = cc.process(Path("/nowhere"), Fake(), conn)
        expected = costs.cost_usd("claude-sonnet-5", toks, costs.load_pricing())
        assert expected > 0 and abs(out["cost"] - round(expected, 4)) < 1e-6, (out, expected)
        assert conn.execute("SELECT model_used FROM sessions WHERE session_id='fk-1'").fetchone()[0] == "claude-sonnet-5"
    finally:
        cc._EXTRACTORS.pop("fake", None)
        conn.close()
    print("  ok  compute-costs: 2-tuple extractors still priced via pricing.json")


def test_opencode_reasoning_extract_real_text():
    """Unlike Claude, OpenCode persists reasoning text. One step per root
    assistant message that said or did something: reasoning, response, exact
    actions; the compaction summary and the part-less aborted turn are skipped."""
    with tempfile.TemporaryDirectory() as td:
        src = _oc_source(Path(td))
        steps = reasoning.extract_opencode(next(iter(src.discover())))
        assert [s.turn_index for s in steps] == [1], steps
        s = steps[0]
        assert s.thinking == "check the db" and s.decision == "the storage moved"
        assert s.actions == [{"tool": "bash", "input": "command=ls"},
                             {"tool": "subtask", "input": "agent=explore: scan the repo"}], s.actions
        assert s.signature_present is True
        assert s.timestamp == to_iso_utc(_OC_T0 + 2000)
    print("  ok  reasoning.extract_opencode: real reasoning text + actions per assistant turn")


def test_render_markdown_note_is_source_aware():
    """The no-thinking note explained Claude Code's signature-only storage for
    EVERY source; for other CLIs it must not claim to be Claude Code."""
    step = reasoning.ReasoningStep(turn_index=1, thinking="", decision="did x", signature_present=True)
    md = reasoning.render_markdown([step], {"cli_source": "opencode", "session_id": "ses_x", "title": "t"})
    assert "Claude Code" not in md and "opencode" in md, md[:400]
    md = reasoning.render_markdown([step], {"cli_source": "claude", "session_id": "s", "title": "t"})
    assert "Claude Code stores extended-thinking" in md
    print("  ok  render_markdown: no-thinking note names the actual source")


def test_opencode_sync_inlines_spilled_tool_output():
    """Tool outputs over 2000 lines / 50 KB are spilled to <data>/tool-output/
    tool_<id> and purged after 7 days; the part keeps a truncated preview plus
    state.metadata.outputPath. While the blob exists, the mirror inlines it —
    the mirror is the backup, and the blob will not be there next week."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        src = _oc_source(root, seed=False)
        conn = sqlite3.connect(str(src.db_path))
        _oc_seed(conn)
        spill_dir = src.data_dir / "tool-output"
        spill_dir.mkdir()
        blob = spill_dir / "tool_00c2dd9e40012bvNNrwXYZ"
        blob.write_text("line\n" * 5000)
        gone = spill_dir / "tool_gone"
        for pid, path_ in (("prt_s1", str(blob)), ("prt_s2", str(gone))):
            conn.execute("INSERT INTO part VALUES (?, 'msg_a1', ?, ?, ?, ?)",
                         (pid, _OC_ROOT, _OC_T0 + 2100, _OC_T0 + 2100, json.dumps({
                             "type": "tool", "tool": "bash", "callID": pid,
                             "state": {"status": "completed", "input": {"command": "cat big"},
                                       "output": f"...output truncated...\n\nFull output saved to: {path_}\n\nline",
                                       "title": "cat big", "time": {"start": 1, "end": 2},
                                       "metadata": {"truncated": True, "outputPath": path_}}})))
        conn.commit()
        conn.close()
        path = next(iter(src.discover()))
        a1 = next(json.loads(l) for l in path.read_text().splitlines()[1:] if json.loads(l)["info"]["id"] == "msg_a1")
        by_id = {pd["id"]: pd for pd in a1["parts"]}
        assert by_id["prt_s1"]["state"]["output"] == "line\n" * 5000
        assert by_id["prt_s1"]["state"]["metadata"]["inlined"] is True
        assert by_id["prt_s2"]["state"]["output"].startswith("...output truncated")   # blob gone: untouched
        assert "inlined" not in by_id["prt_s2"]["state"]["metadata"]
    print("  ok  opencode sync inlines spilled tool output while the blob still exists")


def test_opencode_sync_survives_unexpected_migration_table_shape():
    """The migration-version probe is informational. On the real 1.18.15 DB the
    `migration` table has no `name` column and the probe raised — taking the
    whole backfill (every source) down with it. Any shape must be tolerated,
    and no sqlite error inside sync() may escape discover()."""
    with tempfile.TemporaryDirectory() as td:
        src = _oc_source(Path(td))
        conn = sqlite3.connect(str(src.db_path))
        conn.executescript("DROP TABLE migration; CREATE TABLE migration(id INTEGER PRIMARY KEY, "
                           "hash TEXT, created_at INTEGER); INSERT INTO migration VALUES (14, 'abc', 1);")
        conn.close()
        r = src.sync()
        assert r.written == [_OC_ROOT] and not r.db_missing, r
        head = json.loads((src.mirror_dir / f"{_OC_ROOT}.jsonl").read_text().splitlines()[0])
        assert "migration" in head["source"]
        conn = sqlite3.connect(str(src.db_path))
        conn.executescript("DROP TABLE migration;")
        conn.close()
        assert src.sync(force=True).written == [_OC_ROOT]
        # a broken part row (NULL data) must not abort the whole projection either
        conn = sqlite3.connect(str(src.db_path))
        conn.execute("INSERT INTO part VALUES ('prt_bad', 'msg_u2', ?, ?, ?, 'not json')", (_OC_ROOT, _OC_T0, _OC_T0))
        conn.commit()
        conn.close()
        assert src.sync(force=True).written == [_OC_ROOT]
        assert len(list(src.discover())) == 1
    print("  ok  opencode sync tolerates odd migration tables and bad part rows")


# --- phase B: live watch, restore/re-import, DB snapshots -------------------
def test_watcher_sync_trigger_resyncs_mirror():
    """The watcher cannot watch DB rows. A write to opencode.db / its WAL is a
    change trigger: the adapter re-syncs and the resulting mirror create/modify/
    delete events flow through the ordinary handler. log/ and tool-output/
    churn constantly and must be ignored."""
    import time as _time
    import watcher
    old_log, old_deb = watcher._log, watcher.SYNC_DEBOUNCE_S
    try:
        with tempfile.TemporaryDirectory() as td:
            src = _oc_source(Path(td))
            assert [r[0] for r in src.watch_roots()] == [src.mirror_dir, src.data_dir]
            for yes in ("opencode.db", "opencode.db-wal", "opencode-beta.db-wal"):
                assert src.sync_trigger(src.data_dir / yes) is True, yes
            for no in ("opencode.db-shm", "log/2026.log", "tool-output/tool_x", "auth.json"):
                assert src.sync_trigger(src.data_dir / no) is False, no
            assert src.sync_trigger(src.mirror_dir / f"{_OC_ROOT}.jsonl") is False
            wal = src.data_dir / "opencode.db-wal"
            wal.write_bytes(b"")
            watcher._log = lambda msg: None
            watcher.SYNC_DEBOUNCE_S = 0.05
            h = watcher._Handler(src)
            # watch_roots() above created the (empty) mirror dir; the file is what sync writes
            assert not (src.mirror_dir / f"{_OC_ROOT}.jsonl").exists()
            h._process(str(wal))                       # trigger, not a session
            _time.sleep(0.4)
            assert (src.mirror_dir / f"{_OC_ROOT}.jsonl").exists()
    finally:
        watcher._log, watcher.SYNC_DEBOUNCE_S = old_log, old_deb
    print("  ok  watcher: a DB/WAL write re-syncs the mirror (debounced); log/ ignored")


def test_hookstate_mark_and_recently():
    """One module for the 30 s hook race-guard, shared by the Claude Stop hook,
    the OpenCode hook and the watcher: mark() prunes stale entries and writes
    atomically; recently() is False for unknown, stale or corrupt state."""
    import hookstate
    from datetime import datetime, timedelta, timezone
    with tempfile.TemporaryDirectory() as td:
        state = Path(td) / "hook-state.json"
        assert hookstate.recently("ses_a", path=state) is False       # no file yet
        hookstate.mark("ses_a", path=state)
        assert hookstate.recently("ses_a", path=state) is True
        assert hookstate.recently("other", path=state) is False
        stale = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        state.write_text(json.dumps({"old": stale, "ses_a": datetime.now(timezone.utc).isoformat()}))
        assert hookstate.recently("old", path=state) is False
        hookstate.mark("ses_b", path=state)
        kept = json.loads(state.read_text())
        assert "old" not in kept and {"ses_a", "ses_b"} <= set(kept)
        state.write_text("{corrupt")
        assert hookstate.recently("ses_a", path=state) is False
        hookstate.mark("ses_c", path=state)
        assert hookstate.recently("ses_c", path=state) is True
    print("  ok  hookstate: mark/recently with TTL pruning, corrupt-tolerant")


def test_watcher_race_guard_is_source_agnostic():
    """The Stop-hook race guard was `adapter.name == "claude" and ...`; the
    OpenCode hook needs the same guard, so it keys on the shared hook state
    alone (uuid and ses_ ids never collide)."""
    import hookstate
    import watcher
    conn = _temp_db()
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    orig_connect, orig_log, orig_recently = indexer.connect, watcher._log, hookstate.recently
    logs = []
    try:
        with tempfile.TemporaryDirectory() as td:
            src = _oc_source(Path(td))
            path = next(iter(src.discover()))
            indexer.connect = lambda *a, **k: orig_connect(db_path)
            watcher._log = logs.append
            hookstate.recently = lambda sid, **k: sid == _OC_ROOT
            watcher._Handler(src)._process(str(path))
            assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0, "upsert must be skipped"
            assert any("race-guard" in m for m in logs), logs
            hookstate.recently = lambda sid, **k: False
            watcher._Handler(src)._process(str(path))
            assert conn.execute("SELECT COUNT(*) FROM sessions WHERE session_id=?", (_OC_ROOT,)).fetchone()[0] == 1
    finally:
        indexer.connect, watcher._log, hookstate.recently = orig_connect, orig_log, orig_recently
        conn.close()
    print("  ok  watcher race-guard applies to every adapter via hookstate")


def test_opencode_export_docs_roundtrip_shape():
    """Restore re-imports through `opencode import`, which decodes the export
    shape {info: Session.Info, messages: [{info, parts}]} with Effect Schema —
    required keys present, ids re-attached, root doc first, then each child."""
    from sources.opencode import to_export_docs
    with tempfile.TemporaryDirectory() as td:
        src = _oc_source(Path(td))
        docs = to_export_docs(next(iter(src.discover())))
        assert [d["info"]["id"] for d in docs] == [_OC_ROOT, _OC_CHILD]
        root = docs[0]
        assert set(root) == {"info", "messages"}
        for k in ("id", "slug", "projectID", "directory", "path", "title", "version", "time"):
            assert k in root["info"], k
        assert [m["info"]["id"] for m in root["messages"]] == \
            ["msg_u1", "msg_a1", "msg_uc", "msg_a2", "msg_u2", "msg_a3"]
        m = root["messages"][1]
        assert m["info"]["sessionID"] == _OC_ROOT and m["info"]["role"] == "assistant"
        assert m["parts"][3]["tool"] == "bash" and m["parts"][3]["messageID"] == "msg_a1" \
            and m["parts"][3]["sessionID"] == _OC_ROOT
        assert docs[1]["info"]["parentID"] == _OC_ROOT
        assert [m["info"]["id"] for m in docs[1]["messages"]] == ["msg_c1", "msg_c2"]
        # JSON columns keep their real shape — `permission` is a LIST (PermissionRuleset);
        # `opencode import` rejected a mirror where it had been coerced to {}
        assert root["info"]["permission"] == [{"permission": "question", "action": "deny", "pattern": "*"}], root["info"]
        assert root["info"]["model"] == {"id": _OC_MODEL[1], "providerID": _OC_MODEL[0]}
        assert "revert" not in root["info"] and "metadata" not in root["info"]      # unset stays absent
    print("  ok  to_export_docs: {info, messages[{info, parts}]} per session, root first")


def test_opencode_restore_reimports_via_hook():
    """Restore = the raw copy back into the mirror (row live again, browsable)
    PLUS `opencode import` of root then children, run with cwd = the session's
    directory because import re-homes a session to wherever it runs. A failing
    import never un-restores the row; it is reported. Sessions still present
    in the DB are skipped (import is idempotent, but why spend it)."""
    import restore
    old_archive = reasoning.ARCHIVE
    conn = _temp_db()
    try:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            reasoning.ARCHIVE = root / "archive"
            src = _oc_source(root)
            proj = root / "proj"
            proj.mkdir()
            c = sqlite3.connect(str(src.db_path))
            c.execute("UPDATE session SET directory = ?", (str(proj),))
            c.commit()
            c.close()
            live = next(iter(src.discover()))
            indexer.upsert(src.parse_header(live), conn=conn)
            raw = root / "archive" / "raw" / "2026" / "08"
            raw.mkdir(parents=True)
            (raw / f"{_OC_ROOT}@v2.jsonl").write_bytes(live.read_bytes())
            _oc_wipe(src.db_path)                          # `opencode session delete`
            live.unlink()
            indexer.archive(_OC_ROOT, indexer.TRANSCRIPT_MISSING, conn=conn)
            calls = []

            def fake_import(file, cwd):
                calls.append((json.loads(Path(file).read_text())["info"]["id"], cwd))
                return True, ""
            src._run_import = fake_import
            res = restore.restore_session(_OC_ROOT, conn=conn, registry={"opencode": src})
            assert res.status == "restored" and res.path == src.mirror_dir / f"{_OC_ROOT}.jsonl", res
            assert res.path.exists() and res.reimported is True, res
            assert [c[0] for c in calls] == [_OC_ROOT, _OC_CHILD], calls
            assert all(c[1] == proj for c in calls), calls
            assert conn.execute("SELECT archived FROM sessions WHERE session_id=?", (_OC_ROOT,)).fetchone()[0] == 0
            # failing import: still restored in the browser, failure reported
            live.unlink()
            indexer.archive(_OC_ROOT, indexer.TRANSCRIPT_MISSING, conn=conn)
            src._run_import = lambda file, cwd: (False, "opencode import: schema decode failed")
            res = restore.restore_session(_OC_ROOT, conn=conn, registry={"opencode": src})
            assert res.status == "restored" and res.reimported is False, res
            assert "schema decode failed" in res.detail, res.detail
            # disabled per config: no import attempted, reported as not applicable
            live.unlink()
            indexer.archive(_OC_ROOT, indexer.TRANSCRIPT_MISSING, conn=conn)
            src.reimport_on_restore = False
            src._run_import = lambda file, cwd: (_ for _ in ()).throw(AssertionError("must not import"))
            res = restore.restore_session(_OC_ROOT, conn=conn, registry={"opencode": src})
            assert res.status == "restored" and res.reimported is None, res
            assert restore.RestoreResult("x", "restored").reimported is None
    finally:
        reasoning.ARCHIVE = old_archive
        conn.close()
    print("  ok  restore(opencode): mirror back + `opencode import` root then children, cwd=directory")


def test_backup_opencode_snapshot_and_rotation():
    """Weekly whole-DB copies via VACUUM INTO (consistent even mid-write, WAL
    included), rotated to `keep`, due only after `days`; a copy is a complete,
    openable database."""
    from datetime import datetime, timezone
    bk = _load_script("backup-opencode")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        src = _oc_source(root)
        out = root / "snapshots"
        day = lambda d: datetime(2026, 9, d, 12, 0, tzinfo=timezone.utc)  # noqa: E731
        assert bk.is_due(out, days=7, now=day(1)) is True                 # nothing yet
        p1 = bk.snapshot(src.db_path, out, keep=2, now=day(1))
        assert bk.is_due(out, days=7, now=day(2)) is False and bk.is_due(out, days=7, now=day(9)) is True
        p2 = bk.snapshot(src.db_path, out, keep=2, now=day(8))
        p3 = bk.snapshot(src.db_path, out, keep=2, now=day(15))
        files = sorted(out.glob("opencode-*.db"))
        assert files == [p2, p3] and not p1.exists(), files
        c = sqlite3.connect(f"file:{p3}?mode=ro", uri=True)
        assert c.execute("SELECT COUNT(*) FROM session").fetchone()[0] == 2
        assert c.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        c.close()
    print("  ok  backup-opencode: VACUUM INTO snapshots, rotation, due-after-N-days")


def test_opencode_run_import_verifies_the_session_landed():
    """`opencode import` printed "Error: Unexpected error / Expected
    PermissionRuleset" and still EXITED 0 during verification. Success is the
    session being in OpenCode's DB afterwards; a clean exit is not enough, and
    an "Error:" on stderr is a failure whose text is the detail."""
    import subprocess as sp
    from sources.opencode import to_export_docs
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        src = _oc_source(root)
        doc = root / "doc.json"
        doc.write_text(json.dumps(to_export_docs(next(iter(src.discover())))[0]))
        src.has_binary = lambda: True
        orig = sp.run
        stderr = ""
        sp.run = lambda *a, **k: sp.CompletedProcess(a[0], 0, stdout="", stderr=stderr)
        try:
            ok, detail = src._run_import(doc, root)
            assert ok is True, detail                       # exit 0, clean, session present
            stderr = "Error: Unexpected error\nExpected PermissionRuleset, got {}\n  at [\"permission\"]\n"
            ok, detail = src._run_import(doc, root)
            assert ok is False and "PermissionRuleset" in detail, detail
            stderr = ""
            _oc_wipe(src.db_path)
            ok, detail = src._run_import(doc, root)
            assert ok is False and "did not appear" in detail, detail   # exit 0, but nothing landed
        finally:
            sp.run = orig
    print("  ok  _run_import: success = session present afterwards, not exit 0")


def test_watcher_delete_event_for_existing_file_is_a_replace():
    """The OpenCode mirror is rewritten atomically (tmp + os.replace). macOS
    FSEvents reports the overwritten inode as a deletion of the target path,
    and during verification the watcher archived a live session on exactly that
    event. A path that still exists was replaced, not deleted."""
    import watcher
    conn = _temp_db()
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    orig_connect, orig_log = indexer.connect, watcher._log
    logs = []
    try:
        with tempfile.TemporaryDirectory() as td:
            src = _oc_source(Path(td))
            path = next(iter(src.discover()))
            indexer.upsert(src.parse_header(path), conn=conn)
            conn.commit()
            indexer.connect = lambda *a, **k: orig_connect(db_path)
            watcher._log = logs.append
            ev = type("Ev", (), {"is_directory": False, "src_path": str(path)})()
            watcher._Handler(src).on_deleted(ev)          # file still there
            row = conn.execute("SELECT archived FROM sessions WHERE session_id=?", (_OC_ROOT,)).fetchone()
            assert row["archived"] == 0, "a replaced file must not archive the row"
            assert any("replaced" in m for m in logs), logs
            path.unlink()
            watcher._Handler(src).on_deleted(ev)          # now it really is gone
            assert conn.execute("SELECT archived FROM sessions WHERE session_id=?", (_OC_ROOT,)).fetchone()[0] == 1
    finally:
        indexer.connect, watcher._log = orig_connect, orig_log
        conn.close()
    print("  ok  watcher: delete event on a still-existing path is a replace, not a deletion")


# --- phase C: OpenCode plugin hook --------------------------------------------
def test_opencode_hook_resolves_child_to_root_and_syncs_only_it():
    """session.idle fires for child sessions too; the hook resolves the ROOT
    (read-only parent_id walk), re-projects only that root, indexes it, marks
    the race-guard so the watcher skips the same file, and returns the root.
    --deleted re-syncs so the mirror file goes and the row gets archived."""
    import hookstate
    import sbconfig
    hook = _load_script("opencode-hook")
    old_state, old_archive = sbconfig.HOOK_STATE, reasoning.ARCHIVE
    conn = _temp_db()
    try:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sbconfig.HOOK_STATE = root / "hook-state.json"
            reasoning.ARCHIVE = root / "archive"
            src = _oc_source(root)
            assert hook.run("ses_doesnotexist000000000000", adapter=src, conn=conn, spawn=False) is None
            assert not src.mirror_dir.exists()
            out = hook.run(_OC_CHILD, adapter=src, conn=conn, spawn=False)
            assert out == _OC_ROOT, out
            assert [p.name for p in src.mirror_dir.glob("ses_*.jsonl")] == [f"{_OC_ROOT}.jsonl"]
            row = conn.execute("SELECT turn_count, archived FROM sessions WHERE session_id=?", (_OC_ROOT,)).fetchone()
            assert row is not None and row["turn_count"] == 2 and row["archived"] == 0
            assert hookstate.recently(_OC_ROOT) and not hookstate.recently(_OC_CHILD)
            _oc_wipe(src.db_path)
            assert hook.run(_OC_ROOT, adapter=src, conn=conn, deleted=True, spawn=False) is None
            assert not (src.mirror_dir / f"{_OC_ROOT}.jsonl").exists()
            assert reasoning.find_archived_raw(_OC_ROOT) is not None       # archived before unlink
    finally:
        sbconfig.HOOK_STATE, reasoning.ARCHIVE = old_state, old_archive
        conn.close()
    print("  ok  opencode-hook: child -> root, sync only it, upsert, race-guard; --deleted re-syncs")


def test_opencode_hook_always_exits_zero():
    """Same contract as the Claude Stop hook: whatever happens (bad id, no DB,
    broken config), the process exits 0 — it must never surface as an error
    inside OpenCode."""
    import os
    import subprocess
    with tempfile.TemporaryDirectory() as td:
        env = {**os.environ, "SB_DB": str(Path(td) / "reg.db"), "XDG_DATA_HOME": str(Path(td) / "nodata"),
               "HOME": td}
        for argv in (["garbage"], [], ["ses_x", "--deleted"], ["--nonsense"]):
            p = subprocess.run([sys.executable, str(_REPO / "scripts" / "opencode-hook.py"), *argv],
                               capture_output=True, text=True, env=env, timeout=60)
            assert p.returncode == 0, (argv, p.stderr[-300:])
    print("  ok  opencode-hook exits 0 on every failure path")


def test_opencode_plugin_template_renders_and_installs():
    """The plugin is rendered from a template with the venv python and hook
    paths baked in, installed to <config>/opencode/plugins/session-browser.js
    (auto-loaded for every project, no opencode.json edit), removable, and
    reported stale when the repo moved."""
    import shutil
    import subprocess
    inst = _load_script("install-opencode-plugin")
    js = inst.render(repo=_REPO, python=Path(sys.executable))
    assert "__" not in js.replace("__proto__", ""), "unrendered placeholder"
    assert "session.idle" in js and "session.deleted" in js and "--deleted" in js
    assert str(_REPO / "scripts" / "opencode-hook.py") in js and str(sys.executable) in js
    assert "unref()" in js and "try" in js
    if shutil.which("node"):
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "p.mjs"
            f.write_text(js)
            assert subprocess.run(["node", "--check", str(f)], capture_output=True).returncode == 0
    with tempfile.TemporaryDirectory() as td:
        cfg = Path(td) / "opencode"
        assert inst.status(config_dir=cfg) == "not installed"
        dest = inst.install(config_dir=cfg, repo=_REPO, python=Path(sys.executable))
        assert dest == cfg / "plugins" / "session-browser.js" and dest.read_text() == js
        assert inst.status(config_dir=cfg, repo=_REPO, python=Path(sys.executable)) == "installed"
        assert inst.status(config_dir=cfg, repo=Path("/moved/elsewhere"), python=Path(sys.executable)).startswith("stale")
        assert inst.install(config_dir=cfg, repo=_REPO, python=Path(sys.executable)) == dest   # idempotent
        assert inst.uninstall(config_dir=cfg) is True and not dest.exists()
        assert inst.uninstall(config_dir=cfg) is False
    print("  ok  opencode plugin: renders, node-checks, installs/uninstalls, stale detection")


# --- portability: config layering -------------------------------------------------
def test_config_layers_over_example_defaults():
    """A per-machine config.toml written before a source or provider existed
    (the work laptop's predates [sources.opencode]) must not silently drop it:
    sections and keys missing from config.toml inherit config.toml.example;
    an explicit value (enabled = false) still wins."""
    import os
    import sbconfig
    with tempfile.TemporaryDirectory() as td:
        cfg = Path(td) / "config.toml"
        cfg.write_text('[sources.claude]\nenabled = true\n[sources.codex]\nenabled = false\n'
                       '[enrichment]\nprovider = "claude-headless"\n')
        merged = sbconfig.load_config(cfg)
        assert merged["sources"]["opencode"]["enabled"] is True, "new source dropped by a stale config"
        assert merged["sources"]["codex"]["enabled"] is False, "explicit override lost"
        assert merged["sources"]["claude"]["projects_dir"] == "~/.claude/projects", "key-level inherit"
        assert merged["enrichment"]["provider"] == "claude-headless"
        assert "opencode_headless" in merged["enrichment"], "new provider block dropped"
        assert merged["ui"]["port"] == 7655
        # no config.toml at all == the example
        assert sbconfig.load_config(Path(td) / "absent.toml")["sources"]["opencode"]["enabled"] is True
        # SB_CONFIG points a whole run (tests, `sb demo`) at another file
        old = os.environ.get("SB_CONFIG")
        os.environ["SB_CONFIG"] = str(cfg)
        try:
            assert sbconfig.load_config()["sources"]["codex"]["enabled"] is False
        finally:
            if old is None:
                os.environ.pop("SB_CONFIG", None)
            else:
                os.environ["SB_CONFIG"] = old
    print("  ok  config.toml layers over config.toml.example (stale configs keep new sources)")


def test_claude_and_copilot_available_without_binary_on_path():
    """Availability means 'there are transcripts to read'. Gating it on `which
    <cli>` unsubscribed the watcher and skipped backfill the moment the CLI was
    uninstalled or fell off the daemon's PATH — the transcripts were still there,
    and keeping them browsable after a CLI goes away is the point of this tool
    (codex and opencode already behave this way)."""
    import os
    from sources.claude import ClaudeSource
    from sources.copilot import CopilotSource
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        (tmp / "projects").mkdir()
        (tmp / "state").mkdir()
        old = os.environ.get("PATH", "")
        try:
            os.environ["PATH"] = str(tmp / "no-bins")
            assert ClaudeSource(tmp / "projects").is_available(), "claude transcripts present but source unavailable"
            assert CopilotSource(tmp / "state").is_available(), "copilot transcripts present but source unavailable"
            assert ClaudeSource(tmp / "projects").has_binary() is False
            assert CopilotSource(tmp / "state").has_binary() is False
        finally:
            os.environ["PATH"] = old
        assert ClaudeSource(tmp / "absent").is_available() is False
        assert CopilotSource(tmp / "absent").is_available() is False
    print("  ok  claude/copilot availability = transcripts on disk, not the binary on PATH")


# --- portability: nightly pipeline + watcher on partial machines --------------------
def test_embed_step_skips_instead_of_failing_without_semantic_stack():
    """refresh-all marked EVERY nightly run failed on a --lite install (no
    sentence-transformers) and on any machine whose model is not cached while
    downloads are disallowed — both are modes the user chose, not failures.
    Only an attempted download that fails deserves a nonzero exit."""
    import os
    import semsearch
    emb = _load_script("embed-sessions")
    saved_mod = sys.modules.get("sentence_transformers")
    saved_get = semsearch.get_model
    saved_env = os.environ.get("SB_ALLOW_MODEL_DOWNLOAD")
    try:
        semsearch.get_model.cache_clear()
        sys.modules["sentence_transformers"] = None          # `import` now raises ImportError
        assert emb._load_model() is None, "lite install must be a skip"

        def not_cached():
            raise RuntimeError("embedding model 'x' is not cached locally; set SB_ALLOW_MODEL_DOWNLOAD=1")
        semsearch.get_model = not_cached
        os.environ["SB_ALLOW_MODEL_DOWNLOAD"] = "0"
        assert emb._load_model() is None, "offline by choice must be a skip"
        os.environ["SB_ALLOW_MODEL_DOWNLOAD"] = "1"
        try:
            emb._load_model()
            raise AssertionError("a failed download must exit nonzero")
        except SystemExit as e:
            assert e.code == 1
    finally:
        semsearch.get_model = saved_get
        if saved_mod is None:
            sys.modules.pop("sentence_transformers", None)
        else:
            sys.modules["sentence_transformers"] = saved_mod
        if saved_env is None:
            os.environ.pop("SB_ALLOW_MODEL_DOWNLOAD", None)
        else:
            os.environ["SB_ALLOW_MODEL_DOWNLOAD"] = saved_env
    print("  ok  embed step: lite / offline-by-choice skip with exit 0; failed download exits 1")


def test_watcher_subscribes_roots_that_appear_later():
    """A fresh laptop may install Session Browser before the CLI has written
    its first session directory (or before OpenCode's mirror exists). The
    watcher used to skip missing roots forever and exit when none existed —
    launchd never restarts a clean exit, so nothing was watched until reboot."""
    import watcher

    class FakeObserver:
        def __init__(self):
            self.scheduled = []

        def schedule(self, handler, path, recursive=True):
            self.scheduled.append(path)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        adapter = _cx_source(tmp)
        obs = FakeObserver()
        sched = watcher._RootScheduler(obs, [(tmp / "sessions", adapter), (tmp / "archived", adapter)],
                                       log=lambda m: None)
        assert sched.poll() == [] and obs.scheduled == []
        (tmp / "sessions").mkdir()
        assert sched.poll() == [tmp / "sessions"] and obs.scheduled == [str(tmp / "sessions")]
        assert sched.poll() == [] and len(obs.scheduled) == 1, "must not subscribe twice"
        assert sched.pending == [(tmp / "archived", adapter)]
        (tmp / "archived").mkdir()
        sched.poll()
        assert sched.pending == [] and len(obs.scheduled) == 2
    print("  ok  watcher subscribes to source dirs that appear after start, once each")


def test_opencode_watch_roots_creates_mirror_dir():
    """The mirror is OUR directory. If the watcher starts before the first sync
    (install --no-backfill), the root must exist to be subscribed — otherwise
    every mirror file the WAL-triggered sync writes is invisible until a restart."""
    from sources.opencode import OpenCodeSource
    with tempfile.TemporaryDirectory() as td:
        mirror = Path(td) / "mirror"
        roots = OpenCodeSource(data_dir=Path(td) / "data", mirror_dir=mirror).watch_roots()
        assert mirror.is_dir() and mirror in [r[0] for r in roots], (mirror.exists(), roots)
    print("  ok  opencode watch_roots() materialises the mirror dir")


# --- portability: shell helpers + CLI home env vars -------------------------------
def _stub_bin(dirpath: Path, name: str, body: str = 'echo "{name}-stub $*"') -> Path:
    """A fake CLI on PATH that prints how it was invoked."""
    dirpath.mkdir(parents=True, exist_ok=True)
    p = dirpath / name
    p.write_text("#!/usr/bin/env bash\n" + body.format(name=name) + "\n")
    p.chmod(0o755)
    return p


def _cr(home: Path, bins: Path, sid: str, env_extra: dict | None = None):
    import os
    import subprocess
    env = {k: v for k, v in os.environ.items() if k not in ("CLAUDE_CONFIG_DIR", "CODEX_HOME")}
    env.update({"HOME": str(home), "PATH": f"{bins}:/usr/bin:/bin"})
    env.update(env_extra or {})
    return subprocess.run(["bash", str(_REPO / "bin" / "resume-here.sh"), sid],
                          env=env, capture_output=True, text=True, timeout=30, cwd=str(home))


def test_resume_here_finds_cold_and_archived_codex_rollouts():
    """Codex zstd-compresses rollouts older than ~7 days in place and `codex
    archive` moves them to a sibling tree; both are still resumable, but `cr`
    only looked for plain .jsonl under sessions/ and said 'not found'."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        home, bins = tmp / "home", tmp / "bin"
        home.mkdir()
        _stub_bin(bins, "codex")
        _cx_tree(home / ".codex", _cx_rollout(), root="sessions", compress=True)      # cold
        r = _cr(home, bins, _CX_ID)
        assert r.returncode == 0 and f"codex-stub resume {_CX_ID}" in r.stdout, r.stdout + r.stderr
        arch_id = "019e18fa-0d21-7461-922c-aaaaaaaaaaaa"
        day = home / ".codex" / "archived_sessions" / "2026" / "08" / "02"
        day.mkdir(parents=True)
        (day / f"rollout-2026-08-02T10-00-00-{arch_id}.jsonl").write_text(_cx_rollout())
        r = _cr(home, bins, arch_id)
        assert r.returncode == 0 and f"codex-stub resume {arch_id}" in r.stdout, r.stdout + r.stderr
    print("  ok  cr resumes cold (.zst) and archived codex rollouts")


def test_resume_here_explains_missing_binary_before_touching_anything():
    """On a laptop that has the transcripts but not the CLI (uninstalled, or a
    stripped PATH), cr used to link the session into the cwd's project dir and
    then die with a bare exit 127 from `exec`. Check first, say what is missing."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        home, bins = tmp / "home", tmp / "empty-bin"
        proj = home / ".claude" / "projects" / "-Users-x-proj"
        proj.mkdir(parents=True)
        sid = "0a1b2c3d-1111-4222-8333-444455556666"
        (proj / f"{sid}.jsonl").write_text(_cl_transcript("hello", cwd="/Users/x/proj"))
        r = _cr(home, bins, sid)
        assert r.returncode == 1, (r.returncode, r.stdout, r.stderr)
        assert "claude" in r.stderr and "not on PATH" in r.stderr, r.stderr
        linked = [p for p in (home / ".claude" / "projects").glob("*/*.jsonl") if p.is_symlink()]
        assert not linked, "must not link memory when the CLI cannot run"
    print("  ok  cr reports a missing CLI binary instead of exit 127")


def test_registry_honours_cli_home_env_vars():
    """Claude Code relocates its whole state with $CLAUDE_CONFIG_DIR, Codex with
    $CODEX_HOME, OpenCode with $XDG_DATA_HOME. With the documented default in
    config, the adapters must follow the env var — otherwise a multi-account
    setup indexes the wrong (usually empty) tree with no error at all."""
    import os
    from sources import registry
    keys = ("CLAUDE_CONFIG_DIR", "CODEX_HOME", "XDG_DATA_HOME")
    saved = {k: os.environ.get(k) for k in keys}
    try:
        os.environ["CLAUDE_CONFIG_DIR"] = "/alt/claude"
        os.environ["CODEX_HOME"] = "/alt/codex"
        os.environ["XDG_DATA_HOME"] = "/alt/xdg"
        assert registry._make_claude().projects_dir == Path("/alt/claude/projects")
        cx = registry._make_codex()
        assert cx.sessions_dir == Path("/alt/codex/sessions") and cx.archived_dir == Path("/alt/codex/archived_sessions")
        assert registry._make_opencode().data_dir == Path("/alt/xdg/opencode")
        for k in keys:
            os.environ.pop(k)
        assert registry._make_claude().projects_dir == Path.home() / ".claude" / "projects"
        assert registry._make_codex().sessions_dir == Path.home() / ".codex" / "sessions"
        assert registry._make_opencode().data_dir == Path.home() / ".local" / "share" / "opencode"
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    print("  ok  registry follows CLAUDE_CONFIG_DIR / CODEX_HOME / XDG_DATA_HOME at the documented defaults")


def _install_python_block(needle: str) -> str:
    import re as _re
    return next(
        m.group(2)
        for m in _re.finditer(r"<<'?(PYEOF)'?(?:[^\n]*\\\n)*[^\n]*\n(.*?)\n\1",
                              (_REPO / "install.sh").read_text(), _re.S)
        if needle in m.group(2))


def test_install_hook_creates_settings_dir_and_honours_config_dir():
    """A brand-new Claude Code install has no ~/.claude yet — registration must
    create it rather than fail with FileNotFoundError. And with
    $CLAUDE_CONFIG_DIR set, settings.json lives THERE; writing ~/.claude/
    settings.json registers a hook Claude never reads."""
    import os
    import subprocess
    block = _install_python_block("session-hook.py")
    with tempfile.TemporaryDirectory() as home:
        base = {**os.environ, "HOME": home}
        base.pop("CLAUDE_CONFIG_DIR", None)
        r = subprocess.run([sys.executable, "-", str(_REPO)], input=block, env=base,
                           capture_output=True, text=True)
        assert r.returncode == 0 and "registered" in r.stdout, r.stdout + r.stderr
        assert (Path(home) / ".claude" / "settings.json").exists(), "missing parent dir must be created"
        alt = Path(home) / "cc"
        r = subprocess.run([sys.executable, "-", str(_REPO)], input=block,
                           env={**base, "CLAUDE_CONFIG_DIR": str(alt)}, capture_output=True, text=True)
        assert r.returncode == 0 and "registered" in r.stdout, r.stdout + r.stderr
        assert (alt / "settings.json").exists(), "must honour CLAUDE_CONFIG_DIR"
    print("  ok  install.sh hook registration creates the settings dir and honours CLAUDE_CONFIG_DIR")


# --- portability: background jobs on Linux (systemd --user) -------------------------
def test_render_job_templates_for_launchd_and_systemd():
    """Linux got only a printed hint while macOS got launchd jobs; the watcher
    and nightly refresh are what make the browser stay current, so Linux needs
    the equivalent systemd --user units. One renderer serves both: plist output
    must XML-escape (a repo path with & or < used to abort the install with a
    malformed plist), systemd output must shell-quote paths with spaces."""
    import os
    import shutil
    import subprocess
    render = _load_script("render-job")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        repo = tmp / "My Repo & Co"          # spaces and an ampersand on purpose
        # real (fake) executables so `systemd-analyze verify` below judges the
        # unit text, not the absence of a venv in this temp repo
        for rel in (".venv/bin/python", "watcher.py", "scripts/refresh-all.py"):
            f = repo / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text("#!/bin/sh\n")
            f.chmod(0o755)
        (tmp / "logs").mkdir()
        env = {"SB_VENV_PY": str(repo / ".venv/bin/python"), "SB_REPO": str(repo),
               "SB_LOG_DIR": str(tmp / "logs"), "SB_HOME_DIR": str(tmp), "SB_JOB_PATH": "/usr/bin:/bin"}
        for job in ("watcher", "refresh"):
            out = render.render(_REPO / "launchd" / f"{job}.plist.template", env, fmt="plist")
            assert "__" not in out.replace("__proto__", ""), out
            assert "My Repo &amp; Co" in out and "&amp;" in out, "plist must XML-escape &"
        units = {}
        for name in ("session-browser-watcher.service", "session-browser-refresh.service",
                     "session-browser-refresh.timer"):
            out = render.render(_REPO / "systemd" / f"{name}.template", env, fmt="systemd")
            assert "__" not in out.replace("__proto__", ""), out
            units[name] = out
        assert '"' + str(repo / ".venv/bin/python") + '"' in units["session-browser-watcher.service"], \
            "ExecStart must quote a path containing spaces"
        assert "watcher.py" in units["session-browser-watcher.service"]
        assert "refresh-all.py" in units["session-browser-refresh.service"] and "--enrich" in units["session-browser-refresh.service"]
        assert "OnCalendar=" in units["session-browser-refresh.timer"] and "Persistent=true" in units["session-browser-refresh.timer"]
        assert "PATH=/usr/bin:/bin" in units["session-browser-watcher.service"]
        # CLI entry point: render to a destination file
        dest = tmp / "out" / "w.service"
        r = subprocess.run([sys.executable, str(_REPO / "scripts" / "render-job.py"),
                            str(_REPO / "systemd" / "session-browser-watcher.service.template"),
                            str(dest), "--format", "systemd"], env={**os.environ, **env},
                           capture_output=True, text=True)
        assert r.returncode == 0 and dest.read_text() == units["session-browser-watcher.service"], r.stderr
        if shutil.which("systemd-analyze"):
            for name, text in units.items():
                (tmp / "out" / name).write_text(text)
                v = subprocess.run(["systemd-analyze", "--user", "verify", str(tmp / "out" / name)],
                                   capture_output=True, text=True)
                assert v.returncode == 0, v.stderr
    print("  ok  render-job: launchd plists XML-escaped, systemd units quoted and complete")


# --- app review: restore honesty, primer pointers, bridge/resume guards, stats ------
def test_codex_and_copilot_restore_paths():
    """Only claude/opencode had restore_path(), so the Archived tab showed a
    Restore button for codex/copilot rows (a raw copy existed) that the server
    then refused as 'unsupported'. Codex must find an existing rollout under
    the row's date dir (plain or .zst) or synthesise the canonical name; the
    name must round-trip through session_id_for_path. Copilot's transcript
    lives at <state>/<sid>/events.jsonl. Both refuse paths outside their tree."""
    from sources.copilot import CopilotSource
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        cx = _cx_source(tmp)
        cold = _cx_tree(tmp, _cx_rollout(), root="sessions", compress=True)
        row = {"project_path": str(cold.parent), "session_id": _CX_ID, "start_time": "2026-08-01T10:00:00.000Z"}
        assert cx.restore_path(row) == cold, "existing (cold) rollout must be found"
        row = {"project_path": str(cold.parent), "session_id": "019e18fa-0d21-7461-922c-bbbbbbbbbbbb",
               "start_time": "2026-08-01T10:00:00.000Z"}
        synth = cx.restore_path(row)
        assert synth == cold.parent / "rollout-2026-08-01T10-00-00-019e18fa-0d21-7461-922c-bbbbbbbbbbbb.jsonl", synth
        assert cx.session_id_for_path(synth) == row["session_id"], "synthesised name must map back to the id"
        assert cx.restore_path({"project_path": str(tmp / "elsewhere"), "session_id": "x",
                                "start_time": ""}) is None, "must refuse paths outside the codex trees"
        # a row with no usable date dir falls back to sessions/YYYY/MM/DD from start_time
        assert cx.restore_path({"project_path": "", "session_id": "abc", "start_time": "2026-09-03T01:02:03.000Z"}) \
            == tmp / "sessions" / "2026" / "09" / "03" / "rollout-2026-09-03T01-02-03-abc.jsonl"
        cp = CopilotSource(tmp / "state")
        assert cp.restore_path({"project_path": str(tmp / "state" / "sid9"), "session_id": "sid9"}) \
            == tmp / "state" / "sid9" / "events.jsonl"
        assert cp.restore_path({"project_path": str(tmp / "other" / "sid9"), "session_id": "sid9"}) is None
    print("  ok  codex/copilot restore_path: find or synthesise, id round-trips, tree-contained")


def _app_harness():
    """(app module, conn, root, restore) with indexer.connect and the archive
    pointed at temp locations; call restore() in finally."""
    import os
    sb = _load_app()
    conn = _temp_db()
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    td = tempfile.mkdtemp(prefix="sb-app-")
    root = Path(td)
    saved = (indexer.connect, reasoning.ARCHIVE, sb.SOURCES, os.environ.get("HOME"), os.environ.get("PATH"))
    orig_connect = indexer.connect
    reasoning.ARCHIVE = root / "archive"
    indexer.connect = lambda *a, **k: orig_connect(db_path)
    os.environ["HOME"] = str(root / "home")          # bridge primers land under ~/.session-browser
    (root / "home").mkdir()

    def restore():
        indexer.connect, reasoning.ARCHIVE, sb.SOURCES = saved[0], saved[1], saved[2]
        os.environ["HOME"] = saved[3]
        os.environ["PATH"] = saved[4]
        conn.close()
    return sb, conn, root, restore


def test_api_restorable_means_restore_will_succeed():
    """`restorable` used to mean 'archived + raw copy exists' and ignored whether
    the adapter can put the file back, so the UI advertised Restore and the
    server answered 409 unsupported. It must agree with restore.plan()."""
    from sources.codex import CodexSource
    sb, conn, root, restore = _app_harness()
    try:
        raw = root / "archive" / "raw" / "2026" / "08"
        raw.mkdir(parents=True)
        day = root / "codex" / "sessions" / "2026" / "08" / "01"
        # the raw copy's own session_meta carries the id; codex indexes by it
        for sid, src in ((_CX_ID, "codex"), ("gm-aged", "gemini")):
            (raw / f"{sid}.jsonl").write_text(_cx_rollout())
            indexer.upsert(_header(sid, cli_source=src, project_path=str(day),
                                   start_time="2026-08-01T10:00:00.000Z"), conn=conn)
            indexer.archive(sid, indexer.TRANSCRIPT_MISSING, conn=conn)
        conn.commit()
        sb.SOURCES = {"codex": CodexSource(root / "codex" / "sessions")}
        c = sb.app.test_client()
        rows = {s["session_id"]: s for s in c.get("/api/sessions?state=archived").get_json()}
        assert rows[_CX_ID]["restorable"] is True, rows[_CX_ID]
        assert rows["gm-aged"]["restorable"] is False, "no adapter -> must not advertise Restore"
        assert rows["gm-aged"]["restore_blocker"] == "unsupported"
        H = {"X-Requested-With": "session-browser"}
        r = c.post(f"/api/sessions/{_CX_ID}/restore", headers=H)
        assert r.status_code == 200 and r.get_json()["status"] == "restored", r.data
        dest = Path(r.get_json()["path"])
        assert dest.exists() and dest.name == f"rollout-2026-08-01T10-00-00-{_CX_ID}.jsonl", dest
        assert [s["session_id"] for s in c.get("/api/sessions").get_json()] == [_CX_ID]
        r = c.post("/api/sessions/gm-aged/restore", headers=H)
        assert r.status_code == 409 and r.get_json()["status"] == "unsupported", r.data
    finally:
        restore()
    print("  ok  restorable == restore will succeed; codex rows restore to a canonical rollout name")


def test_context_primer_pointers_are_honest():
    """The primer/bridge told the receiving agent to open a transcript path
    that cannot exist for codex (<date-dir>/<uuid>.jsonl) and, for archived
    rows, handed out `cr <id>` although /resume refuses them. Pointers must
    only name files that exist; archived rows must say 'restore first'."""
    from sources.codex import CodexSource
    sb, conn, root, restore = _app_harness()
    try:
        cx_root = root / "codex"
        cold = _cx_tree(cx_root, _cx_rollout(), root="sessions", compress=True)
        indexer.upsert(_header(_CX_ID, cli_source="codex", project_path=str(cold.parent),
                               start_time="2026-08-01T10:00:00.000Z"), conn=conn)
        indexer.upsert(_header("gone", cli_source="codex", project_path=str(cold.parent),
                               start_time="2026-08-02T10:00:00.000Z"), conn=conn)
        indexer.upsert(_header("aged", cli_source="claude", project_path=str(root / "p" / "-x")), conn=conn)
        indexer.archive("aged", indexer.TRANSCRIPT_MISSING, conn=conn)
        conn.commit()
        sb.SOURCES = {"codex": CodexSource(cx_root / "sessions")}
        md, _ = sb._build_context(conn, _CX_ID)
        assert f"Transcript: `{cold}`" in md, md
        md, _ = sb._build_context(conn, "gone")
        assert "Transcript:" not in md, "a pointer to a missing file must be omitted:\n" + md
        md, _ = sb._build_context(conn, "aged")
        assert "cr aged" not in md and "Transcript:" not in md, md
        assert "restore" in md.lower(), md
    finally:
        restore()
    print("  ok  primer pointers name only existing files; archived rows say restore first")


def test_api_sources_and_bridge_refuse_uninstalled_target():
    """Bridge offered all four CLIs on every laptop; picking one that is not
    installed wrote a primer and handed back a command that dies with
    'command not found'. The SPA needs /api/sources to know what is installed,
    and the server must refuse the target itself. Also: `copilot -p` is the
    non-interactive one-shot flag — a handoff must open a session (-i)."""
    import os
    from sources.claude import ClaudeSource
    sb, conn, root, restore = _app_harness()
    try:
        indexer.upsert(_header("s1", cwd=str(root)), conn=conn)
        conn.commit()
        sb.SOURCES = {"claude": ClaudeSource(root / "projects")}
        c = sb.app.test_client()
        H = {"X-Requested-With": "session-browser"}
        os.environ["PATH"] = str(root / "nobins")
        srcs = c.get("/api/sources").get_json()
        # every CLI the bridge knows, flagged: enabled (adapter loaded), installed, has_data
        assert set(srcs) >= {"claude", "codex", "copilot", "opencode"}, srcs
        assert srcs["claude"] == {"enabled": True, "installed": False, "has_data": False}, srcs
        assert srcs["codex"]["enabled"] is False and srcs["codex"]["installed"] is False, srcs
        r = c.post("/api/sessions/s1/bridge?target=codex", headers=H)
        assert r.status_code == 409 and "codex" in r.get_json()["error"], r.data
        _stub_bin(root / "bins", "codex")
        os.environ["PATH"] = str(root / "bins")
        r = c.post("/api/sessions/s1/bridge?target=codex", headers=H)
        assert r.status_code == 200 and 'codex "$(cat' in r.get_json()["command"], r.data
        assert "copilot -i " in sb._BRIDGE_CMD["copilot"] and " -p " not in sb._BRIDGE_CMD["copilot"]
    finally:
        restore()
    print("  ok  /api/sources reports installed/has_data; bridge refuses uninstalled targets; copilot -i")


def test_api_resume_refuses_source_without_adapter():
    """A row whose source is disabled in config (or unknown) got a 200 with
    `cr <id>` and '# unknown source' — a wasted round trip to the terminal."""
    sb, conn, root, restore = _app_harness()
    try:
        indexer.upsert(_header("g1", cli_source="gemini"), conn=conn)
        conn.commit()
        sb.SOURCES = {}
        r = sb.app.test_client().get("/api/sessions/g1/resume")
        assert r.status_code == 409 and "gemini" in r.get_json()["error"], r.data
    finally:
        restore()
    print("  ok  resume: no adapter for the row's source -> 409 with the reason")


def test_stats_filters_and_sums_are_exact():
    """Three string/NULL slips: ?days=N compared 'YYYY-MM-DDT…' against
    datetime()'s 'YYYY-MM-DD …' (T > space, so the cutoff day leaked in);
    the header folder count included ''; SUM(a+b+c+d) zeroed a whole session
    when one token column was NULL."""
    from datetime import datetime, timedelta, timezone
    sb, conn, root, restore = _app_harness()
    try:
        now = datetime.now(timezone.utc)
        old = to_iso_utc(now - timedelta(days=7, hours=6))
        new = to_iso_utc(now - timedelta(days=1))
        indexer.upsert(_header("old", last_activity=old, folder_name=""), conn=conn)
        indexer.upsert(_header("new", last_activity=new, folder_name="proj"), conn=conn)
        conn.execute("UPDATE sessions SET input_tokens=1000, output_tokens=2000, cache_read_tokens=3000, "
                     "cache_write_tokens=NULL, cost_usd=1.75, model_used='m' WHERE session_id='new'")
        conn.commit()
        c = sb.app.test_client()
        ids = [s["session_id"] for s in c.get("/api/sessions?days=7").get_json()]
        assert ids == ["new"], f"7-day filter leaked the cutoff day: {ids}"
        stats = c.get("/api/stats").get_json()
        assert stats["folders"] == len(c.get("/api/sessions/folders").get_json()) == 1, stats
        ts = c.get("/api/stats/timeseries").get_json()
        assert ts["by_model"][0]["tokens"] == 6000, ts["by_model"]
        assert ts["totals"]["tokens"] == 6000, ts["totals"]
    finally:
        restore()
    print("  ok  stats: day cutoff exact, folder count matches the dropdown, NULL token columns don't zero sums")


# --- pipeline review: archive/restore lifecycle, readers, watcher on Linux -------------
def test_readable_trail_cleanup_keys_on_full_session_id():
    """write_readable swept '*/*/<first 8 chars of id>-*.md' to keep one trail
    per session. OpenCode ids are `ses_` + a millisecond clock (those 8 chars
    repeat for ~50 days) and Codex ids are UUIDv7 (65 s window), so rendering
    one session deleted the others' trails and left their reasoning_path
    pointing at nothing. Key on the FULL id, and let persist() remove exactly
    the previous render it recorded (which also retires legacy names)."""
    old_archive = reasoning.ARCHIVE
    try:
        with tempfile.TemporaryDirectory() as td:
            reasoning.ARCHIVE = Path(td)
            a, b = "ses_01a088b5a200AbCdEfGhIjKlMn", "ses_01a08ddbfe00AbCdEfGhIjKlMn"
            steps = [reasoning.ReasoningStep(turn_index=1, thinking="t", decision="d")]
            pa = reasoning.write_readable(steps, {"session_id": a, "title": "session a", "last_activity": "2026-09-01T00:00:00.000Z"})
            pb = reasoning.write_readable(steps, {"session_id": b, "title": "session b", "last_activity": "2026-09-02T00:00:00.000Z"})
            assert pa.exists() and pb.exists(), "rendering B deleted A's trail (shared 8-char prefix)"
            pa2 = reasoning.write_readable(steps, {"session_id": a, "title": "renamed a", "last_activity": "2026-09-03T00:00:00.000Z"})
            assert pa2.exists() and not pa.exists() and pb.exists(), "re-render must replace only A's own trail"
            conn = _temp_db()
            indexer.upsert(_header(a), conn=conn)
            legacy = Path(td) / "readable" / "2026" / "08" / "ses_01a0-old-name.md"
            legacy.parent.mkdir(parents=True)
            legacy.write_text("x")
            conn.execute("UPDATE sessions SET reasoning_path=? WHERE session_id=?", (str(legacy), a))
            conn.commit()
            reasoning.persist(a, steps, pa2, conn=conn)
            assert not legacy.exists() and pa2.exists(), "persist() retires the previous render it recorded"
            assert conn.execute("SELECT reasoning_path FROM sessions WHERE session_id=?", (a,)).fetchone()[0] == str(pa2)
            conn.close()
    finally:
        reasoning.ARCHIVE = old_archive
    print("  ok  readable trails keyed on the full session id; persist retires the previous render")


def test_opencode_restore_survives_next_sync_without_reimport():
    """Restore copies the raw file into the mirror and re-indexes; if the
    re-import into OpenCode does not land (binary absent, disabled, failed),
    the next sync() saw 'a mirror file whose root is not in the DB' and deleted
    it again — 'Restored' flashed, then the row was back in Archived seconds
    later. A restored-but-not-reimported file must be kept until OpenCode has
    the session again."""
    import restore as _restore
    from dataclasses import asdict
    old_archive = reasoning.ARCHIVE
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        reasoning.ARCHIVE = root / "archive"
        try:
            src = _oc_source(root)
            src.has_binary = lambda: False           # no `opencode` here -> re-import cannot run
            src.sync()
            mirror = src.mirror_dir / f"{_OC_ROOT}.jsonl"
            header = src.parse_header(mirror)
            conn = _temp_db()
            indexer.upsert(header, conn=conn)
            _oc_wipe(src.db_path)
            rep = src.sync()                          # archives the raw copy, unlinks the mirror file
            assert rep.removed == [_OC_ROOT] and not mirror.exists()
            indexer.archive(_OC_ROOT, indexer.TRANSCRIPT_MISSING, conn=conn)
            conn.commit()
            res = _restore.restore_session(_OC_ROOT, conn, registry={"opencode": src})
            assert res.status == "restored" and res.reimported is False, res
            assert mirror.exists()
            rep = src.sync()
            assert rep.removed == [] and mirror.exists(), "sync must not un-restore a file OpenCode lacks"
            # once OpenCode has the session again the marker is gone and the file is a normal mirror
            for suffix in ("", "-wal", "-shm"):
                Path(str(src.db_path) + suffix).unlink(missing_ok=True)
            c = sqlite3.connect(str(src.db_path))
            _oc_schema(c)
            _oc_seed(c)
            c.close()
            src.sync(force=True)
            assert mirror.exists() and not list(src.mirror_dir.glob("*.restored")), list(src.mirror_dir.iterdir())
            conn.close()
        finally:
            reasoning.ARCHIVE = old_archive
    print("  ok  opencode restore without re-import survives the next sync; marker cleared once re-imported")


def _cx_rollout_with_usage() -> str:
    usage = _cx_line(6, {"type": "token_count", "info": {"total_token_usage": {
        "input_tokens": 1000, "cached_input_tokens": 400, "output_tokens": 250, "reasoning_output_tokens": 0}}})
    return _cx_rollout() + json.dumps(usage) + "\n"


def test_archive_raw_keeps_zst_representation_and_readers_decompress():
    """archive_raw copied a compressed Codex rollout byte-for-byte into a file
    named <sid>.jsonl, so the vault held a zstd frame every reader parsed as
    text (aged-out Codex sessions vanished from full-text search and restore
    wrote garbage). extract_codex and the cost extractor also opened rollouts
    with plain open(), losing reasoning and cost for every rollout older than
    a week. The archive keeps the real suffix; every reader decompresses."""
    import restore as _restore
    from dataclasses import asdict
    cc = _load_script("compute-costs")
    old_archive = reasoning.ARCHIVE
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        reasoning.ARCHIVE = root / "archive"
        try:
            cx = _cx_source(root)
            text = _cx_rollout_with_usage()
            plain = _cx_tree(root, text, root="sessions")
            cold = _cx_tree(root / "cold", text, root="sessions", compress=True)
            assert reasoning.extract_codex(cold) and len(reasoning.extract_codex(cold)) == len(reasoning.extract_codex(plain))
            assert cc._usage_codex(cold)[0]["input"] == cc._usage_codex(plain)[0]["input"] == 600, cc._usage_codex(cold)
            header = asdict(cx.parse_header(cold))
            dest = reasoning.archive_raw(cold, header)
            assert dest.name == f"{_CX_ID}.jsonl.zst", dest
            assert reasoning.find_archived_raw(_CX_ID) == dest
            assert reasoning.archived_raw_index()[_CX_ID] == dest
            assert [t.content for t in cx.parse_full(dest).turns] == [t.content for t in cx.parse_full(plain).turns]
            # restore of a compressed copy keeps the representation (codex reads both)
            conn = _temp_db()
            indexer.upsert(_header(_CX_ID, cli_source="codex", project_path=str(root / "sessions" / "2026" / "08" / "01"),
                                   start_time="2026-08-01T10:00:00.000Z"), conn=conn)
            indexer.archive(_CX_ID, indexer.TRANSCRIPT_MISSING, conn=conn)
            plain.unlink()
            res = _restore.restore_session(_CX_ID, conn, registry={"codex": cx})
            assert res.status == "restored" and res.path.name.endswith(".jsonl.zst"), res
            assert cx.session_id_for_path(res.path) == _CX_ID and cx.parse_header(res.path).turn_count == 2
            conn.close()
        finally:
            reasoning.ARCHIVE = old_archive
    print("  ok  raw archive keeps .jsonl.zst; codex reasoning/cost/restore read compressed rollouts")


def test_find_archived_raw_prefers_newest_across_months():
    """The @vN counter restarts per YYYY/MM directory but candidates were ranked
    by N alone, so after a month rollover the September @v3 beat October's v1
    and restore/full-text used a stale snapshot. Rank by the copy's mtime
    (copy2 preserves the source's)."""
    import os
    old_archive = reasoning.ARCHIVE
    with tempfile.TemporaryDirectory() as td:
        reasoning.ARCHIVE = Path(td)
        try:
            sid = "0199a1f2-aaaa-4bbb-8ccc-ddddeeeeffff"
            sep = Path(td) / "raw" / "2026" / "09"
            octo = Path(td) / "raw" / "2026" / "10"
            sep.mkdir(parents=True)
            octo.mkdir(parents=True)
            t0 = 1_790_000_000
            for i, name in enumerate((f"{sid}.jsonl", f"{sid}@v2.jsonl", f"{sid}@v3.jsonl")):
                p = sep / name
                p.write_text("x" * (10 + i))
                os.utime(p, (t0 + i, t0 + i))
            newest = octo / f"{sid}.jsonl"
            newest.write_text("x" * 20)
            os.utime(newest, (t0 + 10, t0 + 10))
            assert reasoning.find_archived_raw(sid) == newest, reasoning.find_archived_raw(sid)
            assert reasoning.archived_raw_index()[sid] == newest
        finally:
            reasoning.ARCHIVE = old_archive
    print("  ok  newest raw copy wins across a month rollover (mtime, not per-dir version)")


def test_archive_reason_uses_content_signals_and_watcher_agrees():
    """A slash-command-only Claude session (/init, /model …) has turn_count 0
    and no first_message but real assistant work (tokens, a model, often a
    trail). prune-sessions classified it not-a-session (hidden for good,
    unrestorable) while the watcher hardcoded transcript-missing for the same
    deletion. One rule, richer signals, both callers."""
    import watcher
    conn = _temp_db()
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    indexer.upsert(_header("cmd-only", turn_count=0, first_message="", project_path="/p/-proj"), conn=conn)
    indexer.upsert(_header("empty", turn_count=0, first_message="", project_path="/p/-proj"), conn=conn)
    conn.execute("UPDATE sessions SET output_tokens=3000, model_used='claude-opus-4-5' WHERE session_id='cmd-only'")
    conn.commit()
    rows = {r["session_id"]: r for r in conn.execute("SELECT * FROM sessions")}
    assert indexer.infer_archive_reason(rows["cmd-only"]) == indexer.TRANSCRIPT_MISSING
    assert indexer.infer_archive_reason(rows["empty"]) == indexer.NOT_A_SESSION
    orig_connect, old_log = indexer.connect, watcher._log
    try:
        indexer.connect = lambda *a, **k: orig_connect(db_path)
        watcher._log = lambda m: None
        from sources.claude import ClaudeSource
        h = watcher._Handler(ClaudeSource("/p"))

        class Ev:
            is_directory = False

            def __init__(self, p):
                self.src_path = p
        h.on_deleted(Ev("/p/-proj/cmd-only.jsonl"))     # <projects>/<project>/<sid>.jsonl
        h.on_deleted(Ev("/p/-proj/empty.jsonl"))
        got = {r[0]: r[1] for r in conn.execute("SELECT session_id, archived_reason FROM sessions")}
        assert got == {"cmd-only": indexer.TRANSCRIPT_MISSING, "empty": indexer.NOT_A_SESSION}, got
    finally:
        indexer.connect, watcher._log = orig_connect, old_log
        conn.close()
    print("  ok  archive reason: tokens/model/trail count as content; watcher and prune agree")


def test_watcher_root_scheduler_survives_inotify_limit():
    """On Linux, observer.schedule(recursive=True) raises OSError(ENOSPC) once
    fs.inotify.max_user_watches is exhausted; it propagated out of the daemon
    (systemd then restart-looped it). Keep the root pending, log the
    remediation, retry on the next poll."""
    import errno
    import watcher

    class Limited:
        def __init__(self):
            self.fail = True
            self.scheduled = []

        def schedule(self, handler, path, recursive=True):
            if self.fail:
                raise OSError(errno.ENOSPC, "inotify watch limit reached")
            self.scheduled.append(path)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "sessions"
        root.mkdir()
        obs, logs = Limited(), []
        sched = watcher._RootScheduler(obs, [(root, _cx_source(Path(td)))], log=logs.append)
        assert sched.poll() == [] and sched.pending and obs.scheduled == []
        assert any("max_user_watches" in m for m in logs), logs
        obs.fail = False
        assert sched.poll() == [root] and not sched.pending
    print("  ok  watcher keeps a root pending when inotify watches run out, and retries")


def test_copilot_workspace_yaml_is_utf8_and_dict_only():
    """workspace.yaml was the one file read with the locale encoding and no
    ValueError guard: a non-UTF-8 byte raised UnicodeDecodeError out of
    parse_header (that Copilot session never indexed), and a non-mapping YAML
    document crashed on .get(). Decode as UTF-8 with replacement; coerce."""
    from sources.copilot import CopilotSource
    with tempfile.TemporaryDirectory() as td:
        state = Path(td)
        d = state / "sid-latin1"
        d.mkdir()
        (d / "workspace.yaml").write_bytes(b"cwd: /home/u/projets/caf\xe9\nname: caf\xe9\n")   # latin-1 bytes
        (d / "events.jsonl").write_text(json.dumps({"type": "user.message", "data": {"content": "hi"}}) + "\n")
        h = CopilotSource(state).parse_header(d / "events.jsonl")
        assert h is not None and h.cwd.startswith("/home/u/projets/caf"), h
        d2 = state / "sid-list"
        d2.mkdir()
        (d2 / "workspace.yaml").write_text("- not\n- a mapping\n")
        (d2 / "events.jsonl").write_text(json.dumps({"type": "user.message", "data": {"content": "hi"}}) + "\n")
        h = CopilotSource(state).parse_header(d2 / "events.jsonl")
        assert h is not None and h.cwd == "" and h.turn_count == 1, h
    print("  ok  copilot workspace.yaml: utf-8 with replacement, non-mapping tolerated")


def test_opencode_reimport_empty_directory_falls_back_to_home():
    """Path('') is Path('.') and '.'.is_dir() is True, so a session whose
    directory is empty was re-imported into whatever cwd the UI process had —
    silently, because the 'directory is gone' note compared cwd == directory."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        src = _oc_source(root)
        c = sqlite3.connect(str(src.db_path))
        c.execute("UPDATE session SET directory=''")
        c.commit()
        c.close()
        src.sync(force=True)
        seen = {}
        src.has_binary = lambda: True
        src._existing_ids = lambda ids: set()
        src._run_import = lambda file, cwd: (seen.setdefault("cwd", cwd), (True, "ok"))[1]
        ok, detail = src.reimport(src.mirror_dir / f"{_OC_ROOT}.jsonl")
        assert ok is True and seen["cwd"] == Path.home(), (seen, detail)
        assert "directory" in detail and str(Path.home()) in detail, detail
    print("  ok  opencode re-import: empty directory -> $HOME, and the note says so")


# --- stats-report windows: cutoff day is not leaked --------------------------
def test_stats_report_windows_do_not_leak_the_cutoff_day():
    """`sb stats` 7-day / 30-day windows must agree with the UI's days filter:
    a row at 00:00 on the cutoff day is OUTSIDE a window that starts later
    that day. datetime('now', '-N days') spells the cutoff with a space, which
    sorts below the stored 'T' and pulled the whole day in."""
    mod = _load_script("stats-report")
    conn = _temp_db()
    try:
        row = conn.execute(
            "SELECT strftime('%Y-%m-%dT00:00:00.000Z','now','-7 days') AS edge, "
            "strftime('%Y-%m-%dT%H:%M:%S.000Z','now','-6 days') AS inside").fetchone()
        indexer.upsert(_header(sid="edge", last_activity=row["edge"]), conn=conn)
        indexer.upsert(_header(sid="inside", last_activity=row["inside"]), conn=conn)
        line = mod._window(conn, "7 days", "AND " + mod.since_sql(7))
        assert "   1 sessions" in line, line
        assert mod.since_sql(7).count("'T'") == 0 and "T%H" in mod.since_sql(7), mod.since_sql(7)
    finally:
        conn.close()
    print("  ok  stats-report windows use the transcript timestamp spelling")


# --- pricing.json tracks the published Claude list prices --------------------
def test_pricing_matches_published_claude_list_prices():
    """Per-million list prices as published on platform.claude.com/docs/en/about-claude/pricing
    (checked 2026-09-11). The old table priced every Opus at the retired Opus 4.1
    rate ($15/$75), tripling the Usage tab for Opus 4.5+ / Opus 5, priced Sonnet 5
    at the Sonnet 4.x rate, Haiku 4.5 at the Haiku 3.5 rate, and had no Fable /
    Mythos tier at all (cost counted as $0 with a nightly warning)."""
    pricing = costs.load_pricing()
    M = 1_000_000

    def usd(model, **tok):
        return round(costs.cost_usd(model, dict(tok), pricing), 4)

    # Opus 4.5 .. Opus 5 (both '4-7' and '4.7' spellings occur in transcripts)
    for m in ("claude-opus-5", "claude-opus-4-8", "claude-opus-4.7", "claude-opus-4-6", "claude-opus-4-5"):
        assert usd(m, input=M) == 5.0, (m, usd(m, input=M))
        assert usd(m, output=M) == 25.0, m
        assert usd(m, cache_read=M) == 0.5 and usd(m, cache_write=M) == 6.25, m
    # retired Opus 4 / 4.1 keep the old rate
    for m in ("claude-opus-4-1", "claude-opus-4"):
        assert usd(m, input=M) == 15.0 and usd(m, output=M) == 75.0, m
    # Sonnet 5 vs Sonnet 4.x
    assert usd("claude-sonnet-5", input=M) == 2.0 and usd("claude-sonnet-5", output=M) == 10.0
    assert usd("claude-sonnet-5", cache_read=M) == 0.2 and usd("claude-sonnet-5", cache_write=M) == 2.5
    for m in ("claude-sonnet-4.6", "claude-sonnet-4-5", "claude-sonnet-4"):
        assert usd(m, input=M) == 3.0 and usd(m, output=M) == 15.0, m
    # Haiku 4.5 vs Haiku 3.5
    assert usd("claude-haiku-4.5", input=M) == 1.0 and usd("claude-haiku-4.5", output=M) == 5.0
    assert usd("claude-haiku-4-5", cache_read=M) == 0.1 and usd("claude-haiku-4-5", cache_write=M) == 1.25
    assert usd("claude-3-5-haiku", input=M) == 0.8 and usd("claude-3-5-haiku", output=M) == 4.0
    # Fable / Mythos 5.1: cache hits are 0.025x; Fable / Mythos 5: 0.1x
    for m in ("claude-fable-5-1", "claude-mythos-5-1", "claude-fable-5.1"):
        assert usd(m, input=M) == 10.0 and usd(m, output=M) == 50.0, m
        assert usd(m, cache_read=M) == 0.25 and usd(m, cache_write=M) == 12.5, m
    for m in ("claude-fable-5", "claude-mythos-5"):
        assert usd(m, input=M) == 10.0 and usd(m, output=M) == 50.0, m
        assert usd(m, cache_read=M) == 1.0 and usd(m, cache_write=M) == 12.5, m
    # OpenAI (developers.openai.com/api/docs/pricing, 2026-09-11): each 5.x
    # generation is priced on its own, not at the launch gpt-5 rate
    for m, inp, out in (("gpt-5", 1.25, 10.0), ("gpt-5.1", 1.25, 10.0), ("gpt-5.2", 1.75, 14.0),
                        ("gpt-5.3-codex", 1.75, 14.0), ("gpt-5.4", 2.5, 15.0), ("gpt-5.5", 5.0, 30.0),
                        ("gpt-5-mini", 0.25, 2.0), ("gpt-5.4-mini", 0.75, 4.5),
                        ("gpt-5-nano", 0.05, 0.4), ("gpt-5.4-nano", 0.2, 1.25)):
        assert usd(m, input=M) == inp, (m, usd(m, input=M))
        assert usd(m, output=M) == out, (m, usd(m, output=M))
        assert usd(m, cache_read=M) == round(inp / 10, 4), m   # cached input = 0.1x
    # unknown stays unknown (loud $0), never a guess
    assert costs.tier_for_model("claude-nova-9", pricing) is None
    print("  ok  pricing.json matches the published Claude + OpenAI list prices")


# --- Codex cost extractor reads the model from a top-level turn_context ------
def test_codex_cost_extractor_reads_model_from_top_level_turn_context():
    """Real rollouts write turn_context as the RECORD type ({"type":
    "turn_context", "payload": {"model": "gpt-5.5", ...}}); the adapter reads it
    there, so rows say gpt-5.5, but the cost extractor only looked for
    payload.type == "turn_context", never saw a model, and silently billed every
    Codex session at its "gpt-5" fallback tier. Both spellings must resolve."""
    cc = _load_script("compute-costs")
    usage = {"type": "token_count", "info": {"total_token_usage": {
        "input_tokens": 1000, "cached_input_tokens": 400, "output_tokens": 250,
        "reasoning_output_tokens": 0}}}
    meta = {"timestamp": "2026-08-01T10:00:00Z", "type": "session_meta",
            "payload": {"id": _CX_ID, "timestamp": "2026-08-01T10:00:00Z", "cwd": "/x",
                        "cli_version": "0.150.1", "model_provider": "openai"}}
    top_level = [meta,
                 {"timestamp": "2026-08-01T10:00:01Z", "type": "turn_context",
                  "payload": {"cwd": "/x", "model": "gpt-5.5", "approval_policy": "never"}},
                 _cx_line(3, usage)]
    in_payload = [meta, _cx_line(2, {"type": "turn_context", "model": "gpt-5.5"}), _cx_line(3, usage)]
    with tempfile.TemporaryDirectory() as td:
        for label, recs in (("top-level", top_level), ("payload.type", in_payload)):
            p = Path(td) / f"{label}.jsonl"
            p.write_text("".join(json.dumps(r) + "\n" for r in recs))
            totals, per_model = cc._usage_codex(p)
            assert totals["input"] == 600 and totals["cache_read"] == 400, totals
            assert list(per_model) == ["gpt-5.5"], (label, dict(per_model))
    print("  ok  codex cost extractor attributes tokens to the rollout's real model (both dialects)")


# ===== final review: OpenCode mirror safety ==================================
def test_opencode_deleted_session_kept_when_reasoning_archive_is_disabled():
    """With [reasoning] enabled = false the default archiver was a silent no-op
    and _remove_deleted still unlinked: the LAST copy of a deleted OpenCode
    session was destroyed. The file is kept and the report says why."""
    import sbconfig
    old = sbconfig.REASONING_ENABLED
    with tempfile.TemporaryDirectory() as td:
        src = _oc_source(Path(td))
        path = next(iter(src.discover()))
        _oc_wipe(src.db_path)
        sbconfig.REASONING_ENABLED = False
        try:
            r = src.sync()
        finally:
            sbconfig.REASONING_ENABLED = old
        assert r.removed == [] and path.exists(), r
        assert any("reasoning" in w and "keeping" in w for w in r.warnings), r.warnings
    print("  ok  opencode: reasoning archive off -> a deleted session's mirror file is kept")


def test_opencode_sync_never_removes_mirror_files_projected_from_another_db():
    """A daemon that resolves a DIFFERENT OpenCode DB (XDG_DATA_HOME/OPENCODE_DB
    set in the shell, not in the launchd/systemd job) must not archive and
    unlink every mirror file the hook wrote from the real DB. Line 1 records
    the source DB; a mismatch is skipped, never deleted."""
    from sources.opencode import OpenCodeSource
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        real = _oc_source(root)
        path = next(iter(real.discover()))
        other_dir = root / "other"
        other_dir.mkdir()
        c = sqlite3.connect(str(other_dir / "opencode.db"))
        _oc_schema(c)
        c.close()                                            # valid, empty
        other = OpenCodeSource(data_dir=other_dir, mirror_dir=real.mirror_dir)
        r = other.sync()
        assert path.exists() and r.removed == [], r
        assert r.skipped >= 1 and any("another OpenCode DB" in w for w in r.warnings), r
    print("  ok  opencode: a mirror file projected from another DB is never archived or unlinked")


def test_opencode_atomic_writes_use_unique_temp_names_and_sync_is_serialised():
    """The hook (idle + 1.5 s) and the watcher (WAL + 2.5 s) both project the
    same root; a fixed <file>.tmp name let two writers interleave and one crash
    on os.replace. Temp names are unique per writer and sync() holds
    <mirror>/.sync.lock so concurrent syncs serialise."""
    import fcntl
    import os as _os
    import threading
    with tempfile.TemporaryDirectory() as td:
        src = _oc_source(Path(td))
        seen = []
        real_replace = _os.replace

        def spy(a, b):
            seen.append(str(a))
            real_replace(a, b)
        _os.replace = spy
        try:
            src.sync(force=True)
            src.sync(force=True)
        finally:
            _os.replace = real_replace
        tmps = [s for s in seen if ".tmp" in s]
        assert len(tmps) >= 4 and len(set(tmps)) == len(tmps), tmps
        assert not list(src.mirror_dir.glob("*.tmp")), "temp litter"
        lock = src.mirror_dir / ".sync.lock"
        fh = open(lock, "a+")
        fcntl.flock(fh, fcntl.LOCK_EX)
        done = threading.Event()
        t = threading.Thread(target=lambda: (src.sync(force=True), done.set()), daemon=True)
        t.start()
        assert not done.wait(0.5), "sync() ran while another writer held the mirror lock"
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()
        assert done.wait(10), "sync() never completed after the lock was released"
    print("  ok  opencode: unique temp names, concurrent sync() serialised on the mirror lock")


def test_opencode_discover_survives_an_unwritable_mirror_dir():
    """sync() raised OSError out of discover() (manifest write on a read-only
    mirror dir) and every batch script calls discover() outside its per-file
    try — one bad directory took the whole nightly down for EVERY source. A
    failed sync degrades to 'serve what is already mirrored' plus a warning."""
    import os as _os
    if _os.geteuid() == 0:
        print("  --  skipped: root ignores directory permissions")
        return
    with tempfile.TemporaryDirectory() as td:
        src = _oc_source(Path(td))
        path = next(iter(src.discover()))
        src.mirror_dir.chmod(0o555)
        try:
            r = src.sync(force=True)
            assert any("mirror" in w for w in r.warnings), r
            src._last_sync = 0.0
            assert list(src.discover()) == [path]
        finally:
            src.mirror_dir.chmod(0o755)
    print("  ok  opencode: an unwritable mirror dir warns instead of crashing every source's batch")


def test_opencode_reprojection_keeps_inlined_tool_output_after_the_blob_is_purged():
    """OpenCode purges spilled tool outputs after 7 days; the mirror inlined the
    blob while it existed, but every re-projection rebuilt parts from the DB
    and dropped the copy again — and the smaller file became a new raw archive
    version, so Restore would have brought back the lossy one."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        src = _oc_source(root, seed=False)
        conn = sqlite3.connect(str(src.db_path))
        _oc_seed(conn)
        spill_dir = src.data_dir / "tool-output"
        spill_dir.mkdir()
        blob = spill_dir / "tool_keepme"
        blob.write_text("line\n" * 3000)
        conn.execute("INSERT INTO part VALUES (?, 'msg_a1', ?, ?, ?, ?)",
                     ("prt_keep", _OC_ROOT, _OC_T0 + 2100, _OC_T0 + 2100, json.dumps({
                         "type": "tool", "tool": "bash", "callID": "prt_keep",
                         "state": {"status": "completed", "input": {"command": "cat big"},
                                   "output": "(truncated preview)", "title": "cat big",
                                   "metadata": {"truncated": True, "outputPath": str(blob)}}})))
        conn.commit()
        src.sync()
        blob.unlink()                                        # the 7-day purge
        conn.execute("INSERT INTO message VALUES (?, ?, ?, ?, ?)",
                     ("msg_later", _OC_ROOT, _OC_T0 + 90_000, _OC_T0 + 90_000,
                      json.dumps({"role": "user"})))
        conn.commit()
        conn.close()
        r = src.sync()
        assert r.written == [_OC_ROOT], r
        path = src.mirror_dir / f"{_OC_ROOT}.jsonl"
        a1 = next(json.loads(l) for l in path.read_text().splitlines()[1:]
                  if json.loads(l)["info"]["id"] == "msg_a1")
        part = next(pd for pd in a1["parts"] if pd["id"] == "prt_keep")
        assert part["state"]["output"] == "line\n" * 3000, part["state"]["output"][:40]
        assert part["state"]["metadata"]["inlined"] is True
    print("  ok  opencode: re-projection carries an inlined tool output forward after the blob is purged")


def test_opencode_unattributed_assistant_spend_is_bucketed_not_dropped():
    """An assistant message without providerID/modelID contributed nothing to
    the roll-up while the cost extractor still reported the total as
    authoritative — a renamed field upstream would silently zero every OpenCode
    session. Unknown spend lands in an 'unknown/unknown' bucket and sync warns."""
    with tempfile.TemporaryDirectory() as td:
        src = _oc_source(Path(td))
        conn = sqlite3.connect(str(src.db_path))
        conn.execute("INSERT INTO message VALUES (?, ?, ?, ?, ?)",
                     ("msg_nomodel", _OC_ROOT, _OC_T0 + 9000, _OC_T0 + 9000, json.dumps({
                         "role": "assistant", "cost": 9.99,
                         "tokens": {"input": 50000, "output": 9000, "reasoning": 0,
                                    "cache": {"read": 0, "write": 0}}})))
        conn.commit()
        conn.close()
        r = src.sync()
        head = json.loads(next(iter(src.discover())).read_text().splitlines()[0])
        bucket = head["stats"]["models"].get("unknown/unknown")
        assert bucket and bucket["input"] == 50000 and bucket["cost"] == 9.99, head["stats"]["models"]
        assert head["stats"]["cost_usd"] >= 9.99, head["stats"]
        assert any("unknown" in w.lower() and "model" in w.lower() for w in r.warnings), r.warnings
    print("  ok  opencode: spend without a model name is bucketed and warned, never dropped")


def test_opencode_hook_does_not_archive_a_raw_copy_per_turn():
    """session.idle fires per turn; the hook spawned extract-reasoning --archive
    every time and archive_raw writes a new @vN copy whenever the size changed,
    so a 60-turn session left ~60 full copies. The per-turn spawn refreshes the
    trail only; the nightly run and archive-then-unlink on deletion own the vault."""
    import subprocess
    import sbconfig
    hook = _load_script("opencode-hook")
    captured = []
    real = subprocess.Popen
    subprocess.Popen = lambda argv, **kw: captured.append(list(argv)) or type("P", (), {"pid": 1})()
    old_log = sbconfig.LOG_DIR
    try:
        with tempfile.TemporaryDirectory() as td:
            sbconfig.LOG_DIR = Path(td) / "logs"
            sbconfig.LOG_DIR.mkdir()
            src = _oc_source(Path(td))
            conn = _temp_db()
            try:
                hook.run(_OC_ROOT, adapter=src, conn=conn, spawn=True)
            finally:
                conn.close()
    finally:
        subprocess.Popen = real
        sbconfig.LOG_DIR = old_log
    assert captured and "--archive" not in captured[0], captured
    assert "--source" in captured[0] and "opencode" in captured[0], captured
    print("  ok  opencode hook: per-turn spawn refreshes the trail without a new raw copy")


def test_opencode_plugin_runs_under_node_and_spawns_on_the_leading_edge():
    """Bun.spawn inside a bare try/catch meant an OpenCode build that loads
    plugins under Node silently never indexed anything; and a trailing-edge
    debounce dropped the LAST turn when `opencode run` exited inside the
    window. Under node: the first idle event spawns immediately, a burst
    coalesces into one trailing spawn, and failures are logged, not swallowed."""
    import shutil
    import subprocess
    inst = _load_script("install-opencode-plugin")
    js = inst.render(repo=_REPO, python=Path("/bin/sh"))
    assert "typeof Bun" in js and "child_process" in js, "no Node fallback"
    assert "catch (_) {}" not in js, "failures are still swallowed silently"
    if not shutil.which("node"):
        print("  --  node not installed: plugin runtime check skipped")
        return
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        log = root / "spawns.log"
        hook = root / "hook.sh"
        hook.write_text(f"#!/bin/sh\necho \"$@\" >> '{log}'\n")
        hook.chmod(0o755)
        plugin = root / "plugin.mjs"
        plugin.write_text(js.replace(str(_REPO / "scripts" / "opencode-hook.py"), str(hook)))
        driver = root / "drive.mjs"
        driver.write_text(f"""
import {{ SessionBrowser }} from '{plugin}';
const h = await SessionBrowser({{}});
const idle = (id) => h.event({{ event: {{ type: 'session.idle', properties: {{ sessionID: id }} }} }});
await idle('ses_a'); await idle('ses_a'); await idle('ses_a');
await new Promise(r => setTimeout(r, 300));
const fs = await import('node:fs');
const early = fs.existsSync('{log}') ? fs.readFileSync('{log}', 'utf8').trim().split('\\n').filter(Boolean).length : 0;
await new Promise(r => setTimeout(r, 2200));
const late = fs.readFileSync('{log}', 'utf8').trim().split('\\n').filter(Boolean).length;
console.log(JSON.stringify({{ early, late }}));
""")
        p = subprocess.run(["node", str(driver)], capture_output=True, text=True, timeout=30)
        assert p.returncode == 0, p.stderr[-800:]
        got = json.loads(p.stdout.strip().splitlines()[-1])
        assert got["early"] == 1, ("first event must spawn immediately", got, p.stderr[-300:])
        assert got["late"] == 2, ("a burst must coalesce into one trailing spawn", got)
    print("  ok  opencode plugin: runs under node, leading-edge spawn, burst coalesced")


def test_opencode_data_dir_is_watched_non_recursively():
    """watch_roots() returned the whole OpenCode data dir and the watcher
    subscribed recursively — log/, tool-output/, snapshot/, project/ are hundreds
    of directories that never trigger a sync but each cost an inotify watch
    (ENOSPC on Linux, timer churn on macOS). A root may declare (path, recursive)."""
    import watcher
    with tempfile.TemporaryDirectory() as td:
        src = _oc_source(Path(td))
        roots = src.watch_roots()
        as_pairs = [(Path(r[0]), r[1]) if isinstance(r, tuple) else (Path(r), True) for r in roots]
        assert (src.mirror_dir, True) in as_pairs and (src.data_dir, False) in as_pairs, roots
        calls = []

        class Obs:
            def is_alive(self):
                return True

            def schedule(self, handler, path, recursive=True):
                calls.append((Path(path), recursive))
        pairs = watcher._build_watch_pairs()
        oc = [p for p in pairs if p[1].name == "opencode"]
        assert oc and all(len(p) == 3 for p in oc), oc
        sched = watcher._RootScheduler(Obs(), [(src.mirror_dir, src, True), (src.data_dir, src, False)],
                                       log=lambda m: None)
        sched.poll()
        assert (src.data_dir, False) in calls and (src.mirror_dir, True) in calls, calls
    print("  ok  opencode: the data dir is watched non-recursively (only the DB/WAL matter)")


def test_backup_opencode_leaves_no_stub_when_the_snapshot_fails():
    """A locked DB (OpenCode running) made VACUUM INTO raise; the fallback
    created the destination file before failing on the same lock, and the
    0-byte stub then satisfied is_due() for another 7 days."""
    bk = _load_script("backup-opencode")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        db = root / "opencode.db"
        c = sqlite3.connect(str(db))
        _oc_schema(c)
        c.close()
        out = root / "snaps"
        real_connect = sqlite3.connect

        def locked(*a, **kw):
            conn = real_connect(*a, **kw)
            if kw.get("uri") and "mode=ro" in str(a[0]):
                class C:
                    def execute(self, *x, **y):
                        raise sqlite3.OperationalError("database is locked")

                    def backup(self, *x, **y):
                        raise sqlite3.OperationalError("database is locked")

                    def close(self):
                        conn.close()
                return C()
            return conn
        bk.sqlite3.connect = locked
        try:
            try:
                bk.snapshot(db, out, keep=3)
                raised = False
            except sqlite3.OperationalError:
                raised = True
        finally:
            bk.sqlite3.connect = real_connect
        assert raised, "a locked DB must be reported, not swallowed"
        assert bk.snapshots(out) == [] and not list(out.glob("*.db")), list(out.iterdir())
        assert bk.is_due(out, 7) is True
    print("  ok  backup-opencode: a failed snapshot leaves no stub that would silence is_due()")


def test_restore_protects_the_mirror_file_before_copying_and_retries_reimport_when_live():
    """(a) restore copied the mirror file back BEFORE the .restored marker was
    written; a watcher sync in that window saw a file with no DB row and no
    marker and archived + unlinked it — un-restoring the session. The marker
    goes first now. (b) 'already-live' returned before reimport, so a restore
    whose import failed could never be retried from the UI."""
    import restore
    import shutil
    old_archive = reasoning.ARCHIVE
    conn = _temp_db()
    try:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            reasoning.ARCHIVE = root / "archive"
            src = _oc_source(root)
            live = next(iter(src.discover()))
            indexer.upsert(src.parse_header(live), conn=conn)
            raw = root / "archive" / "raw" / "2026" / "08"
            raw.mkdir(parents=True)
            (raw / f"{_OC_ROOT}.jsonl").write_bytes(live.read_bytes())
            _oc_wipe(src.db_path)
            live.unlink()
            indexer.archive(_OC_ROOT, indexer.TRANSCRIPT_MISSING, conn=conn)
            marker = live.with_name(live.name + ".restored")
            seen = []
            real_copy = shutil.copyfile
            restore.shutil.copyfile = lambda s, d: seen.append(marker.exists()) or real_copy(s, d)
            try:
                src._run_import = lambda file, cwd: (False, "import failed")
                res = restore.restore_session(_OC_ROOT, conn=conn, registry={"opencode": src})
            finally:
                restore.shutil.copyfile = real_copy
            assert res.status == "restored" and res.reimported is False, res
            assert seen == [True], "marker must exist before the copy lands"
            assert marker.exists(), "marker stays until the session is back in the DB"
            # (b) retry from the UI: file is live, DB still lacks the session
            calls = []
            src._run_import = lambda file, cwd: calls.append(file) or (True, "")
            res = restore.restore_session(_OC_ROOT, conn=conn, registry={"opencode": src})
            assert res.status == "already-live" and res.reimported is True, res
            assert calls, "already-live must still offer the re-import"
    finally:
        reasoning.ARCHIVE = old_archive
        conn.close()
    print("  ok  restore: sidecar before copy; already-live still re-imports")


# ===== final review: indexing core ==========================================
def test_migrate_backfills_archive_reason_from_the_whole_row():
    """Legacy archived rows (schema v1) were classified from a 3-column
    projection, so a slash-command-only session with tokens/cost/model was
    labelled not-a-session — hidden from the Archived tab, never offered for
    restore — and the idempotency guard made it permanent."""
    mig = _load_script("migrate-db")
    conn = _temp_db()
    try:
        indexer.upsert(_header(sid="init-only", turn_count=0, first_message=""), conn=conn)
        conn.execute("UPDATE sessions SET archived=1, archived_reason=NULL, output_tokens=1200, "
                     "cost_usd=0.42, model_used='claude-opus-4-8' WHERE session_id='init-only'")
        indexer.upsert(_header(sid="noise", turn_count=0, first_message=""), conn=conn)
        conn.execute("UPDATE sessions SET archived=1, archived_reason=NULL WHERE session_id='noise'")
        conn.commit()
        mig._backfill_archive_reason(conn)
        got = dict(conn.execute("SELECT session_id, archived_reason FROM sessions").fetchall())
        assert got == {"init-only": indexer.TRANSCRIPT_MISSING, "noise": indexer.NOT_A_SESSION}, got
    finally:
        conn.close()
    print("  ok  migrate: legacy archived rows are classified from the whole row")


def test_watcher_starts_the_observer_before_subscribing_roots():
    """watchdog defers emitter start until Observer.start(); roots scheduled
    BEFORE start raised their inotify ENOSPC out of start() — outside the
    scheduler's guard — and the daemon died (systemd respawned it into the same
    crash). The observer is started first so every failure lands in poll()."""
    import errno
    import watcher

    class Obs:
        def __init__(self):
            self.started = False
            self.deferred = []

        def is_alive(self):
            return self.started

        def schedule(self, handler, path, recursive=True):
            if not self.started:
                self.deferred.append(path)
                return
            raise OSError(errno.ENOSPC, "inotify watch limit reached")

        def start(self):
            self.started = True
            if self.deferred:
                raise OSError(errno.ENOSPC, "inotify watch limit reached")
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "projects"
        d.mkdir()
        obs = Obs()
        logs = []
        adapter = type("A", (), {"name": "claude"})()
        roots = watcher.start_watching(obs, [(d, adapter)], log=logs.append)
        assert obs.started and roots.pending and not roots.scheduled, (roots.pending, roots.scheduled)
        assert any("cannot watch" in m for m in logs), logs
    print("  ok  watcher: observer started before roots are subscribed (ENOSPC lands in poll())")


def test_watcher_pairs_follow_relocated_cli_homes():
    """_build_watch_pairs re-derived Claude/Copilot roots from raw config keys,
    bypassing registry._cli_home(): with CLAUDE_CONFIG_DIR set the watcher
    subscribed ~/.claude/projects and never indexed a live session. Roots come
    from the adapters — the single source of truth."""
    import os as _os
    import watcher
    with tempfile.TemporaryDirectory() as td:
        alt = Path(td) / "alt-claude"
        (alt / "projects").mkdir(parents=True)
        old = _os.environ.get("CLAUDE_CONFIG_DIR")
        _os.environ["CLAUDE_CONFIG_DIR"] = str(alt)
        try:
            pairs = watcher._build_watch_pairs()
        finally:
            if old is None:
                _os.environ.pop("CLAUDE_CONFIG_DIR", None)
            else:
                _os.environ["CLAUDE_CONFIG_DIR"] = old
        claude_roots = [p[0] for p in pairs if p[1].name == "claude"]
        assert claude_roots == [alt / "projects"], pairs
    print("  ok  watcher: roots follow CLAUDE_CONFIG_DIR / CODEX_HOME / XDG_DATA_HOME like the registry")


def test_backfill_holds_the_write_lock_only_for_the_burst():
    """parse_header ran inside the open write transaction (commit every 200
    rows), so a nightly backfill of multi-MB transcripts held the lock for
    5+ s and the Stop hook's extract-reasoning died with 'database is locked'
    (busy_timeout 5 s). Headers are parsed outside the transaction and written
    in one short burst."""
    import threading
    import time as _time
    bf = _load_script("backfill")
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db = f.name
    indexer.connect(db).close()                         # migrate the schema

    class Slow:
        name = "claude"

        def discover(self):
            return [Path(f"/x/s{i}.jsonl") for i in range(5)]

        def parse_header(self, p):
            _time.sleep(0.12)
            return _header(sid=p.stem)
    errors = []

    def run():
        conn = indexer.connect(db)                      # sqlite connections are per thread
        try:
            bf.index_source(conn, "claude", Slow(), commit_every=200)
        except Exception as e:  # noqa: BLE001
            errors.append(e)
        finally:
            conn.close()
    t = threading.Thread(target=run)
    t.start()
    _time.sleep(0.25)                                   # mid-backfill
    other = sqlite3.connect(db, timeout=0.25)
    try:
        other.execute("PRAGMA busy_timeout=250")
        other.execute("INSERT INTO session_artifacts (session_id, type, content) VALUES ('h', 'journal', 'x')")
        other.commit()
    finally:
        other.close()
        t.join()
    assert not errors, errors
    c = sqlite3.connect(db)
    assert c.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 5
    c.close()
    print("  ok  backfill: parse outside the write transaction; a concurrent writer gets through")


def test_codex_corrupt_zst_rollout_is_not_a_session():
    """zstandard.ZstdError is not a ValueError: a .jsonl.zst whose bytes are not
    a zstd frame raised straight out of parse_header/parse_full (and the
    reasoning and cost readers) instead of returning None — a restore that had
    already copied the file then 500'd in the UI."""
    cc = _load_script("compute-costs")
    with tempfile.TemporaryDirectory() as td:
        cx = _cx_source(Path(td))
        day = Path(td) / "sessions" / "2026" / "08" / "01"
        day.mkdir(parents=True)
        bad = day / (_CX_NAME + ".zst")
        bad.write_bytes(b"this is not a zstd frame at all")
        assert cx.parse_header(bad) is None and cx.parse_full(bad) is None
        assert reasoning.extract_codex(bad) == []
        totals, per_model = cc._usage_codex(bad)
        assert not totals and not per_model
    print("  ok  codex: a corrupt .zst rollout is 'not a session' in every reader, never a traceback")


def test_claude_session_id_gate_matches_discover_depth():
    """discover() globs exactly <projects>/*/*.jsonl, but session_id_for_path
    accepted a .jsonl at ANY depth: a future <project>/<sid>/workflows/x/
    journal.jsonl would be indexed as session 'journal' (every one colliding on
    one row) and then archived by prune because discover() never yields it."""
    from sources.claude import ClaudeSource
    with tempfile.TemporaryDirectory() as td:
        projects = Path(td) / "projects"
        proj = projects / "-Users-x-proj"
        proj.mkdir(parents=True)
        cl = ClaudeSource(projects_dir=projects)
        sid = "0f0f0f0f-0000-4000-8000-000000000001"
        assert cl.session_id_for_path(proj / f"{sid}.jsonl") == sid
        assert cl.session_id_for_path(proj / sid / "workflows" / "wf-1" / "journal.jsonl") is None
        assert cl.session_id_for_path(projects / "stray.jsonl") is None
        assert cl.session_id_for_path(proj / "subagents" / "agent-1.jsonl") is None
    print("  ok  claude: session_id_for_path rejects what discover() would never yield")


def test_copilot_factory_default_and_symlink_guard():
    """(a) an explicit state_dir = "" fell through cfg.get()'s default and made
    CopilotSource('.') 'available' (a recursive watch on the daemon's cwd);
    (b) copilot discover() did not skip symlinked session dirs, so a symlink
    indexed a second row carrying another session's turns."""
    import sbconfig
    from sources import registry as reg
    from sources.copilot import CopilotSource
    real = sbconfig.source_config
    sbconfig.source_config = lambda name: {"state_dir": ""} if name == "copilot" else real(name)
    try:
        assert reg._make_copilot().state_dir == Path("~/.copilot/session-state").expanduser()
    finally:
        sbconfig.source_config = real
    with tempfile.TemporaryDirectory() as td:
        state = Path(td) / "session-state"
        real_dir = state / "11111111-1111-4111-8111-111111111111"
        real_dir.mkdir(parents=True)
        (real_dir / "workspace.yaml").write_text("cwd: /x\n")
        (real_dir / "events.jsonl").write_text("")
        (state / "22222222-2222-4222-8222-222222222222").symlink_to(real_dir)
        assert [p.parent.name for p in CopilotSource(state).discover()] == [real_dir.name]
    print("  ok  copilot: empty state_dir means the default; symlinked session dirs are skipped")


# ===== final review: archive / restore / costs ===============================
def test_raw_copy_version_breaks_an_mtime_tie():
    """Newest-by-mtime tie-broke on the full path string, so @v9 beat @v10 on a
    1-second-granularity volume or when copy2 preserved an unchanged source
    mtime — and Restore put back the OLDER, shorter transcript."""
    import os as _os
    old_archive = reasoning.ARCHIVE
    with tempfile.TemporaryDirectory() as td:
        reasoning.ARCHIVE = Path(td)
        try:
            d = Path(td) / "raw" / "2026" / "08"
            d.mkdir(parents=True)
            for v in (9, 10):
                p = d / f"sid-a@v{v}.jsonl"
                p.write_text("x" * v)
                _os.utime(p, (1_700_000_000, 1_700_000_000))
            assert reasoning.find_archived_raw("sid-a").name == "sid-a@v10.jsonl"
            assert reasoning.archived_raw_index()["sid-a"].name == "sid-a@v10.jsonl"
        finally:
            reasoning.ARCHIVE = old_archive
    print("  ok  raw copies: an mtime tie is broken by the @vN version, not the path string")


def test_readable_trail_is_written_atomically_before_the_old_one_is_retired():
    """write_readable unlinked every older <sid>-*.md FIRST and then wrote the
    new file non-atomically: a kill or ENOSPC between the two left no trail at
    all while sessions.reasoning_path still pointed at the deleted file."""
    import os as _os
    old_archive = reasoning.ARCHIVE
    with tempfile.TemporaryDirectory() as td:
        reasoning.ARCHIVE = Path(td)
        try:
            hdr = {"session_id": "sid-trail", "last_activity": "2026-08-01T00:00:00.000Z",
                   "cli_source": "claude", "cwd": "/x", "folder_name": "x", "title": "Old"}
            step = reasoning.ReasoningStep(turn_index=0, thinking="why", decision="do", actions=[])
            first = reasoning.write_readable([step], hdr)
            assert first.exists()
            real_replace = _os.replace
            _os.replace = lambda a, b: (_ for _ in ()).throw(OSError("ENOSPC"))
            try:
                try:
                    reasoning.write_readable([step], {**hdr, "title": "New"})
                except OSError:
                    pass
            finally:
                _os.replace = real_replace
            assert first.exists(), "the previous trail must survive a failed rewrite"
            assert not list(first.parent.glob("*.tmp")), "no temp litter"
            second = reasoning.write_readable([step], {**hdr, "title": "New"})
            assert second.exists() and (second == first or not first.exists())
        finally:
            reasoning.ARCHIVE = old_archive
    print("  ok  readable trail: atomic write, previous trail retired only after the new one landed")


def test_archive_paths_never_leave_the_vault():
    """session_id and last_activity are file content (a Codex rollout's payload.id,
    a mirror header) yet were used as path components: '../x' escaped the
    archive, and a malformed last_activity produced a one-level directory the
    raw-copy glob could never see — archived, but permanently unrestorable."""
    old_archive = reasoning.ARCHIVE
    with tempfile.TemporaryDirectory() as td:
        reasoning.ARCHIVE = Path(td) / "archive"
        try:
            src = Path(td) / "t.jsonl"
            src.write_text('{"x":1}\n')
            for bad in ("../../pwn", "a/b", "a\\b", ".."):
                try:
                    reasoning.archive_raw(src, {"session_id": bad, "last_activity": "2026-08-01T00:00:00.000Z"})
                    raise AssertionError(f"archive_raw accepted {bad!r}")
                except ValueError:
                    pass
                try:
                    reasoning.write_readable([], {"session_id": bad, "last_activity": "2026-08-01T00:00:00.000Z",
                                                  "cli_source": "claude", "cwd": "/x", "folder_name": "x"})
                    raise AssertionError(f"write_readable accepted {bad!r}")
                except ValueError:
                    pass
            dest = reasoning.archive_raw(src, {"session_id": "sid-odd", "last_activity": "2026"})
            assert dest.parent.relative_to(reasoning.ARCHIVE / "raw").parts == ("0000", "00"), dest
            assert reasoning.find_archived_raw("sid-odd") == dest
        finally:
            reasoning.ARCHIVE = old_archive
    print("  ok  archive: session ids are validated and odd timestamps still land where the glob looks")


def test_compute_costs_never_overwrites_a_price_with_zero_for_unpriced_models():
    """cost_usd was written unconditionally, so a Copilot session on a model
    with no pricing alias — or a run with an unreadable pricing.json — zeroed
    a previously correct cost. When nothing in the session is priced, the
    stored cost is left alone (and the unknown-model line still warns)."""
    cc = _load_script("compute-costs")
    conn = _temp_db()
    try:
        indexer.upsert(_header(sid="cp1", cli_source="copilot"), conn=conn)
        conn.execute("UPDATE sessions SET cost_usd=1.5 WHERE session_id='cp1'")
        conn.commit()

        class Ad:
            name = "copilot"

            def parse_header(self, p):
                return _header(sid="cp1", cli_source="copilot")
        cc._EXTRACTORS["copilot"] = lambda p: ({"input": 1000, "output": 10, "cache_read": 0, "cache_write": 0},
                                              {"gemini-9-ultra": {"input": 1000, "output": 10,
                                                                  "cache_read": 0, "cache_write": 0}})
        try:
            r = cc.process(Path("/x/cp1"), Ad(), conn)
        finally:
            cc._EXTRACTORS["copilot"] = cc._usage_copilot
        assert r is not None and r["cost"] is None, r
        assert conn.execute("SELECT cost_usd FROM sessions WHERE session_id='cp1'").fetchone()[0] == 1.5
    finally:
        conn.close()
    print("  ok  compute-costs: an unpriced session keeps its stored cost instead of being zeroed")


def test_restore_session_batch_survives_one_failure_and_exits_nonzero():
    """restore-session.py --all --apply had no per-row guard: one locked DB or
    corrupt rollout aborted the batch mid-way, and main() returned 0 even when
    every restore failed — on the one command a user runs after losing data."""
    import restore
    rs = _load_script("restore-session")
    rows = [{"session_id": s, "restorable": True, "supported": True, "cli_source": "claude",
             "folder_name": "x", "last_activity": "2026-08-01T00:00:00.000Z", "title": s,
             "archived_at": None, "raw_path": None} for s in ("ok1", "boom", "ok2")]
    calls = []

    def fake(sid):
        calls.append(sid)
        if sid == "boom":
            raise sqlite3.OperationalError("database is locked")
        return restore.RestoreResult(sid, "restored")
    real_plan, real_restore = rs.restore.plan, rs.restore.restore_session
    rs.restore.plan, rs.restore.restore_session = (lambda: rows), fake
    old_argv = sys.argv
    sys.argv = ["restore-session.py", "--all", "--apply"]
    try:
        try:
            rs.main()
            code = 0
        except SystemExit as e:
            code = e.code
    finally:
        sys.argv = old_argv
        rs.restore.plan, rs.restore.restore_session = real_plan, real_restore
    assert calls == ["ok1", "boom", "ok2"], calls
    assert code not in (0, None), "a failed restore must make the batch exit non-zero"
    print("  ok  restore-session --all: one failure is reported, the rest proceed, exit is non-zero")


def test_redaction_masks_bare_platform_tokens():
    """Unnamed credentials in tool output — a Slack webhook URL, HF / GitLab /
    PyPI tokens — reached the readable trail and the MCP egress verbatim."""
    import redact
    # Built at runtime: a token-shaped literal in the source trips GitHub's
    # push protection even though every one of these is fake.
    slack = "https://hooks." + "slack.com/services/" + "T0" + "A" * 7 + "/B0" + "B" * 7 + "/" + "X" * 24
    samples = {
        "slack-webhook": "curl -X POST " + slack,
        "hf": "export HF_TOKEN=" + "hf_" + "Q" * 34,
        "gitlab": "git clone https://oauth2:" + "glpat-" + "y" * 20 + "@gitlab.com/x/y.git",
        "pypi": "twine upload -p " + "pypi-" + "AgEIcHlwaS5vcmc" + "C" * 50,
    }
    for name, text in samples.items():
        out = redact.redact(text)
        assert out != text and "REDACTED" in out.upper(), (name, out)
    print("  ok  redaction: Slack webhook, HF, GitLab and PyPI tokens are masked")


def test_build_fts_rebuild_with_source_only_clears_that_source():
    """--rebuild --source claude ran DELETE FROM sessions_fts before narrowing
    the registry, silently destroying full-text for every OTHER source."""
    import os as _os
    import subprocess
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "r.db"
        conn = indexer.connect(db)
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS sessions_fts USING fts5(session_id UNINDEXED, body)")
        for sid, src in (("c1", "claude"), ("x1", "codex")):
            indexer.upsert(_header(sid=sid, cli_source=src), conn=conn)
            conn.execute("INSERT INTO sessions_fts (session_id, body) VALUES (?, ?)", (sid, f"body of {sid}"))
        conn.commit()
        conn.close()
        env = {**_os.environ, "SB_DB": str(db), "HOME": td}
        p = subprocess.run([sys.executable, str(_REPO / "scripts" / "build-fts.py"), "--rebuild", "--source", "claude"],
                           capture_output=True, text=True, env=env, timeout=120)
        assert p.returncode == 0, p.stderr[-500:]
        conn = sqlite3.connect(str(db))
        left = sorted(r[0] for r in conn.execute("SELECT session_id FROM sessions_fts"))
        conn.close()
        assert left == ["x1"], left
    print("  ok  build-fts: --rebuild --source clears only that source's full-text rows")


def test_fts_archived_index_ignores_source_availability():
    """index_archived received the availability-narrowed registry: once the
    CLI's transcript tree was gone (the exact end state the archive exists
    for) the adapter was dropped and every archived row lost full-text — while
    Restore, using the full registry, still worked."""
    import os as _os
    import subprocess
    with tempfile.TemporaryDirectory() as td:
        home = Path(td)
        db = home / "r.db"
        raw = home / "claude-reasoning-archive" / "raw" / "2026" / "08"
        raw.mkdir(parents=True)
        sid = "aaaaaaaa-0000-4000-8000-000000000001"
        (raw / f"{sid}.jsonl").write_text(_cl_transcript("find me in fts", cwd="/Users/x/proj"))
        conn = indexer.connect(db)
        indexer.upsert(_header(sid=sid, cli_source="claude", project_path=str(home / "gone")), conn=conn)
        indexer.archive(sid, indexer.TRANSCRIPT_MISSING, conn=conn)
        conn.commit()
        conn.close()
        env = {**_os.environ, "SB_DB": str(db), "HOME": str(home)}
        env.pop("CLAUDE_CONFIG_DIR", None)
        p = subprocess.run([sys.executable, str(_REPO / "scripts" / "build-fts.py")],
                           capture_output=True, text=True, env=env, timeout=120)
        assert p.returncode == 0, p.stderr[-500:]
        assert "+1 archived" in p.stdout, p.stdout
        conn = sqlite3.connect(str(db))
        n = conn.execute("SELECT COUNT(*) FROM sessions_fts WHERE sessions_fts MATCH 'fts'").fetchone()[0]
        conn.close()
        assert n == 1, n
    print("  ok  build-fts: archived rows are indexed from the vault even when the CLI's tree is gone")


def test_restore_rechecks_the_zst_destination_before_writing():
    """dest was re-pointed to .zst AFTER the already-live check, so a live cold
    rollout could be overwritten if an adapter ever returned the plain name."""
    import restore
    old_archive = reasoning.ARCHIVE
    conn = _temp_db()
    try:
        with tempfile.TemporaryDirectory() as td:
            reasoning.ARCHIVE = Path(td) / "archive"
            raw = reasoning.ARCHIVE / "raw" / "2026" / "08"
            raw.mkdir(parents=True)
            (raw / "z1.jsonl.zst").write_bytes(b"archived-bytes")
            live = Path(td) / "live.jsonl.zst"
            live.write_bytes(b"LIVE")
            indexer.upsert(_header(sid="z1", cli_source="codex"), conn=conn)
            indexer.archive("z1", indexer.TRANSCRIPT_MISSING, conn=conn)

            class Ad:
                name = "codex"

                def restore_path(self, row):
                    return Path(td) / "live.jsonl"          # plain name; the .zst twin is live

                def parse_header(self, p):
                    return _header(sid="z1", cli_source="codex")
            res = restore.restore_session("z1", conn=conn, registry={"codex": Ad()})
            assert res.status == "already-live", res
            assert live.read_bytes() == b"LIVE", "a live transcript must never be overwritten"
    finally:
        reasoning.ARCHIVE = old_archive
        conn.close()
    print("  ok  restore: the .zst twin of the destination counts as live")


# ===== final review: UI / API / MCP honesty ==================================
def _js_function_source(name: str) -> str:
    """Source text of one top-level `function name(...){ ... }` in the SPA."""
    html = (_REPO / "session-ui" / "static" / "index.html").read_text()
    start = html.index(f"function {name}(")
    if html[max(0, start - 6):start] == "async ":
        start -= 6
    depth = 0
    for i in range(start, len(html)):
        if html[i] == "{":
            depth += 1
        elif html[i] == "}":
            depth -= 1
            if depth == 0:
                return html[start:i + 1]
    raise AssertionError(f"unterminated function {name}")


def test_api_resume_refuses_a_cli_that_is_not_installed():
    """Resume answered 200 with `cr <id>` for a source whose binary is absent on
    this machine (rows synced from the other laptop); the user switched
    terminals and got 'claude is not on PATH'. Bridge already refused — resume
    now does too, and the SPA can disable the button from /api/sources."""
    import os
    from sources.claude import ClaudeSource
    sb, conn, root, restore = _app_harness()
    try:
        indexer.upsert(_header("s1", project_path=str(root / "claude" / "-p")), conn=conn)
        conn.commit()
        sb.SOURCES = {"claude": ClaudeSource(root / "claude")}
        os.environ["PATH"] = str(root / "nobins")
        c = sb.app.test_client()
        r = c.get("/api/sessions/s1/resume")
        assert r.status_code == 409 and "not installed" in r.get_json()["error"], r.data
        assert c.get("/api/sources").get_json()["claude"]["installed"] is False
    finally:
        restore()
    print("  ok  resume: 409 when the row's CLI is not installed here")


def test_api_restorable_is_decided_per_row_not_per_source():
    """_restore_supported() asked 'does the adapter have restore_path'; the
    adapter refuses rows whose recorded path is outside its tree (a registry
    carried from another machine), so Restore was advertised and then 409'd.
    The API now asks restore.supported_for(row)."""
    from sources.claude import ClaudeSource
    sb, conn, root, restore = _app_harness()
    try:
        raw = root / "archive" / "raw" / "2026" / "08"
        raw.mkdir(parents=True)
        inside, outside = root / "claude" / "-p", Path("/Users/other-laptop/.claude/projects/-p")
        for sid, pp in (("row-inside", inside), ("row-outside", outside)):
            (raw / f"{sid}.jsonl").write_text(_cl_transcript("hi", cwd="/x"))
            indexer.upsert(_header(sid, project_path=str(pp)), conn=conn)
            indexer.archive(sid, indexer.TRANSCRIPT_MISSING, conn=conn)
        conn.commit()
        sb.SOURCES = {"claude": ClaudeSource(root / "claude")}
        c = sb.app.test_client()
        rows = {s["session_id"]: s for s in c.get("/api/sessions?state=archived").get_json()}
        assert rows["row-inside"]["restorable"] is True and rows["row-inside"]["restore_blocker"] is None
        assert rows["row-outside"]["restorable"] is False and rows["row-outside"]["restore_blocker"] == "unsupported", rows["row-outside"]
        H = {"X-Requested-With": "session-browser"}
        assert c.post("/api/sessions/row-outside/restore", headers=H).status_code == 409
        assert c.post("/api/sessions/row-inside/restore", headers=H).status_code == 200
    finally:
        restore()
    print("  ok  restorable agrees with restore_session() per row")


def test_restore_blocker_prefers_no_raw_copy_over_unsupported():
    """An archived row with no raw copy AND no adapter support was labelled
    'unsupported', whose tooltip claims 'a raw copy exists' — it does not."""
    sb, conn, root, restore = _app_harness()
    try:
        indexer.upsert(_header("gm-gone", cli_source="gemini"), conn=conn)
        indexer.archive("gm-gone", indexer.TRANSCRIPT_MISSING, conn=conn)
        conn.commit()
        sb.SOURCES = {}
        row = sb.app.test_client().get("/api/sessions?state=archived").get_json()[0]
        assert row["restorable"] is False and row["restore_blocker"] == "no-raw-copy", row
    finally:
        restore()
    print("  ok  restore_blocker names the fact that is actually missing")


def test_bridge_and_resume_say_when_the_recorded_cwd_is_gone():
    """`cd <cwd> && <cli> …` died at the cd for a directory from another
    machine, after the user had already pasted the command. With no such
    directory the command drops the cd and the primer says so; resume reports
    origin_cwd_exists."""
    import os
    from sources.claude import ClaudeSource
    sb, conn, root, restore = _app_harness()
    try:
        gone = "/Users/other-laptop/code/my proj"
        indexer.upsert(_header("s1", cwd=gone, project_path=str(root / "claude" / "-p")), conn=conn)
        conn.commit()
        sb.SOURCES = {"claude": ClaudeSource(root / "claude")}
        bins = root / "bins"
        bins.mkdir()
        for b in ("claude", "codex"):
            (bins / b).write_text("#!/bin/sh\nexit 0\n")
            (bins / b).chmod(0o755)
        os.environ["PATH"] = str(bins)
        c = sb.app.test_client()
        r = c.post("/api/sessions/s1/bridge?target=codex", headers={"X-Requested-With": "x"})
        assert r.status_code == 200, r.data
        j = r.get_json()
        assert not j["command"].startswith("cd "), j["command"]
        assert "does not exist on this machine" in j["primer"], j["primer"][:400]
        assert j["origin_cwd_exists"] is False
        r = c.get("/api/sessions/s1/resume")
        assert r.status_code == 200 and r.get_json()["origin_cwd_exists"] is False, r.data
    finally:
        restore()
    print("  ok  bridge/resume: a missing origin cwd is reported, never `cd`-ed into")


def test_api_sessions_marks_a_truncated_listing():
    """A hard LIMIT 500 with no marker: the header counted 602 sessions, the
    list showed 500, and the oldest 102 — the very rows the archive protects —
    were unreachable from any UI path without a word about it."""
    sb, conn, root, restore = _app_harness()
    try:
        for i in range(503):
            indexer.upsert(_header(f"s{i:04d}", last_activity=f"2026-01-{1 + i % 28:02d}T00:00:00.000Z"), conn=conn)
        conn.commit()
        sb.SOURCES = {}
        c = sb.app.test_client()
        r = c.get("/api/sessions")
        assert len(r.get_json()) == 500
        assert r.headers.get("X-Result-Truncated") == "1" and r.headers.get("X-Result-Total") == "503", dict(r.headers)
        r = c.get("/api/sessions?days=0&search=s0002")
        assert r.headers.get("X-Result-Truncated") in (None, "0")
    finally:
        restore()
    print("  ok  /api/sessions flags a truncated listing with the true total")


def test_spa_markdown_blockquote_and_fetch_errors():
    """(a) mdToHtml sliced the ESCAPED line by the RAW marker length, so every
    '> ' reasoning line rendered as 't; …'. (b) fetchJson threw Error(status),
    discarding the server's honest error body."""
    import shutil
    import subprocess
    if not shutil.which("node"):
        print("  --  node not installed: SPA function check skipped")
        return
    src = "\n".join(_js_function_source(n) for n in ("esc", "mdToHtml", "fetchJson"))
    driver = src + r"""
const q = mdToHtml("> I need to check the auth flow.\n>\n- item **bold**");
if (!q.includes("<blockquote") || !q.includes(">I need to check the auth flow.</blockquote>")) { console.error("BAD:" + q); process.exit(2); }
if (q.includes("t; ")) { console.error("BAD:" + q); process.exit(3); }
globalThis.fetch = async () => ({ ok: false, status: 409, json: async () => ({ error: "codex is not installed on this machine" }) });
fetchJson("/x").then(() => process.exit(4)).catch(e => { if (!String(e.message).includes("codex is not installed")) { console.error("BAD:" + e.message); process.exit(5); } console.log("ok"); });
"""
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "t.js"
        f.write_text(driver)
        p = subprocess.run(["node", str(f)], capture_output=True, text=True, timeout=30)
        assert p.returncode == 0, (p.returncode, p.stderr[-400:])
    print("  ok  SPA: blockquotes render, fetch errors carry the server's message")


def test_shell_helpers_read_the_ui_port_from_config():
    """[ui].port is honoured by app.py, but `sb ui|stop|open` and doctor
    hard-coded 7655: after the documented remedy for a busy port, sb printed
    the wrong URL, could not see the running UI, and `sb stop` killed whatever
    unrelated process held 7655."""
    import os
    import subprocess
    with tempfile.TemporaryDirectory() as td:
        env = {**os.environ, "HOME": td, "SHELL": "/bin/zsh"}
        p = subprocess.run(["bash", str(_REPO / "bin" / "install-cr.sh")], env=env, capture_output=True, text=True, timeout=60)
        assert p.returncode == 0, p.stderr
        rc = (Path(td) / ".zshrc").read_text()
        block = rc[rc.index("# >>> session-browser sb >>>"):rc.index("# <<< session-browser sb <<<")]
        for literal in ("tcp:7655", "127.0.0.1:7655", "7655/tcp"):
            assert literal not in block, (literal, block)
        assert "sbconfig" in block, block          # resolved from config; 7655 survives only as the fallback
    doctor = (_REPO / "bin" / "doctor.sh").read_text()
    for literal in ("tcp:7655", "localhost:7655", ":7655 "):
        assert literal not in doctor, ("doctor still hard-codes the port", literal)
    print("  ok  sb / doctor resolve the UI port from config")


def test_stats_report_token_sums_tolerate_null_columns():
    """_TOK summed the four token columns without per-column COALESCE, so one
    NULL zeroed a row's whole token count (the dashboard COALESCEs each)."""
    mod = _load_script("stats-report")
    conn = _temp_db()
    try:
        indexer.upsert(_header(sid="a"), conn=conn)
        indexer.upsert(_header(sid="b"), conn=conn)
        conn.execute("UPDATE sessions SET input_tokens=100, output_tokens=50, cache_read_tokens=NULL, cache_write_tokens=NULL WHERE session_id='a'")
        conn.execute("UPDATE sessions SET input_tokens=10, output_tokens=10, cache_read_tokens=10, cache_write_tokens=10 WHERE session_id='b'")
        line = mod._window(conn, "all", "")
        assert "190 tok" in line, line
    finally:
        conn.close()
    print("  ok  stats-report token sums COALESCE per column")


def test_mcp_descriptors_flag_archived_rows_and_tolerate_odd_bytes():
    """search/list returned aged-out sessions with no `archived` field, so the
    consuming agent suggested `cr <id>` for a transcript that no longer exists;
    get_reasoning read the trail strictly and one bad byte became a tool error."""
    import os
    tmp = tempfile.mkdtemp(prefix="sb-mcp2-")
    db = str(Path(tmp) / "r.db")
    os.environ["SESSION_MEMORY_DB"] = db
    try:
        conn = indexer.connect(db)
        try:
            for sid in ("live", "aged"):
                indexer.upsert(_header(sid=sid, last_activity="2026-09-01T00:00:00.000Z", title="find me"), conn=conn)
            indexer.archive("aged", indexer.TRANSCRIPT_MISSING, conn=conn)
            trail = Path(tmp) / "trail.md"
            trail.write_bytes(b"# Decision trail\n\xff\xfe odd bytes\n")
            conn.execute("UPDATE sessions SET reasoning_path=? WHERE session_id='live'", (str(trail),))
            conn.commit()
        finally:
            conn.close()
        srv_dir = _REPO / "mcp" / "session-memory"
        sys.path.insert(0, str(srv_dir))
        spec = _ilu.spec_from_file_location("sb_mcp_server2", srv_dir / "server.py")
        srv = _ilu.module_from_spec(spec)
        spec.loader.exec_module(srv)
        srv.common.DB_PATH = db
        recent = {r["session_id"]: r for r in srv.list_recent(days=36500)}
        assert recent["aged"]["archived"] is True and recent["live"]["archived"] is False, recent
        found = {r["session_id"]: r for r in srv.search_sessions("find me", limit=5)}
        assert "archived" in found.get("aged", {}), found
        got = srv.get_reasoning("live")
        assert "markdown" in got and "odd bytes" in got["markdown"], got
        assert "Claude" not in (srv.get_reasoning.__doc__ or "") or "CLI" in (srv.get_reasoning.__doc__ or "")
    finally:
        os.environ.pop("SESSION_MEMORY_DB", None)
    print("  ok  MCP: archived flag on descriptors, lenient trail decoding")


# ===== final review: installer, jobs, CI, docs ===============================
def test_render_job_passes_cli_home_env_through_to_the_jobs():
    """The launchd/systemd jobs propagated only PATH, so CLAUDE_CONFIG_DIR /
    CODEX_HOME / XDG_DATA_HOME / OPENCODE_DB set in the shell were unknown to
    the watcher and the nightly refresh — which then indexed the default trees
    (usually empty) or, worse, a different OpenCode DB."""
    rj = _load_script("render-job")
    env = {"SB_VENV_PY": "/v/bin/python", "SB_REPO": "/r", "SB_LOG_DIR": "/l", "SB_HOME_DIR": "/h",
           "SB_JOB_PATH": "/usr/bin", "SB_JOB_ENV": "CLAUDE_CONFIG_DIR=/alt/claude\nCODEX_HOME=/x y/codex"}
    plist = rj.render(_REPO / "launchd" / "watcher.plist.template", env, "plist")
    assert "<key>CLAUDE_CONFIG_DIR</key><string>/alt/claude</string>" in plist, plist
    assert "<key>CODEX_HOME</key><string>/x y/codex</string>" in plist, plist
    unit = rj.render(_REPO / "systemd" / "session-browser-watcher.service.template", env, "systemd")
    assert 'Environment="CLAUDE_CONFIG_DIR=/alt/claude"' in unit and 'Environment="CODEX_HOME=/x y/codex"' in unit, unit
    # no extra env: the markers render to nothing, never a dangling key
    plain = rj.render(_REPO / "launchd" / "refresh.plist.template", {k: v for k, v in env.items() if k != "SB_JOB_ENV"}, "plist")
    assert "__JOB_ENV__" not in plain and "<key></key>" not in plain, plain
    print("  ok  render-job passes the CLI-home env vars through to the background jobs")


def test_installer_survives_a_failing_pipeline_step_and_uninstall_purge_exits_zero():
    """install.sh ran refresh-all unguarded under set -e: one failing pipeline
    step (FTS5 missing, a pinned provider absent) aborted BEFORE the hooks and
    scheduler were installed, leaving a database nothing keeps fresh. And
    uninstall.sh --purge always exited 1 (its last command was a false test)."""
    import os
    import shutil
    import subprocess
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        clone = root / "repo"
        clone.mkdir()
        files = subprocess.run(["git", "ls-files"], cwd=_REPO, capture_output=True, text=True).stdout.split()
        for f in files:
            src = _REPO / f
            if src.is_file():
                dst = clone / f
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        # A "venv" whose python is THIS interpreter (the venv locally; the
        # CI runner's python with requirements installed): the installer then
        # skips venv creation and pip is already satisfied — no network, no
        # dependence on a .venv existing in the checkout.
        (clone / ".venv" / "bin").mkdir(parents=True)
        os.symlink(sys.executable, clone / ".venv" / "bin" / "python")
        stub = clone / "scripts" / "refresh-all.py"
        stub.write_text("#!/usr/bin/env python3\nimport sys\nprint('boom: simulated pipeline failure')\nsys.exit(1)\n")
        home = root / "home"
        home.mkdir()
        env = {**os.environ, "HOME": str(home), "PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "SHELL": "/bin/zsh"}
        for k in ("CLAUDE_CONFIG_DIR", "CODEX_HOME", "XDG_DATA_HOME", "OPENCODE_DB"):
            env.pop(k, None)
        p = subprocess.run(["bash", "install.sh", "--lite", "--no-scheduler"], cwd=clone, env=env,
                           capture_output=True, text=True, timeout=600)
        assert p.returncode == 0, (p.returncode, p.stdout[-1500:], p.stderr[-800:])
        assert "Session Browser installed" in p.stdout, p.stdout[-800:]
        assert "pipeline step" in p.stdout or "re-run" in p.stdout, p.stdout[-800:]
        assert (clone / "config.toml").exists()
        q = subprocess.run(["bash", "uninstall.sh", "--purge"], cwd=clone, env=env,
                           capture_output=True, text=True, timeout=120)
        assert q.returncode == 0, (q.returncode, q.stdout[-600:], q.stderr[-400:])
        assert not (home / ".session-browser").exists()
    print("  ok  install.sh continues past a failing pipeline step; uninstall --purge exits 0")


def test_ci_runs_every_suite_the_docs_promise():
    """tests/test_work_journal.py — the enrichment/journal suite — ran in
    neither CI nor the CONTRIBUTING checklist, so a change to the journal path
    could pass CI green."""
    ci = (_REPO / ".github" / "workflows" / "ci.yml").read_text()
    contributing = (_REPO / "CONTRIBUTING.md").read_text()
    for suite in ("tests/test_smoke.py", "tests/test_work_journal.py", "tests/test_portability.py"):
        assert suite in ci, f"{suite} missing from CI"
        assert suite in contributing, f"{suite} missing from CONTRIBUTING"
    print("  ok  CI and CONTRIBUTING run all three suites")


def test_docs_reference_only_files_flags_and_keys_that_exist():
    """Documentation drift, checked mechanically: every scripts/*.py and
    bin/*.sh a doc or skill names must exist; every [ui] key documented in
    config.toml.example must be read by app.py; the ADDING-A-CLI sample must
    not gate availability on the binary; setup/README invoke repo scripts via
    the venv; the Linux prerequisite names a 3.11+ Python; log names are the
    real ones."""
    import re
    import tomllib
    docs = [_REPO / "README.md", _REPO / "CONTRIBUTING.md", *(_REPO / "docs").glob("*.md"),
            *(_REPO / "skills").glob("*/SKILL.md")]
    for doc in docs:
        text = doc.read_text()
        for ref in set(re.findall(r"\b(?:scripts|bin)/[A-Za-z0-9_\-]+\.(?:py|sh)\b", text)):
            assert (_REPO / ref).exists(), f"{doc.name} names {ref}, which does not exist"
    example = tomllib.loads((_REPO / "config.toml.example").read_text())
    app_src = (_REPO / "session-ui" / "app.py").read_text()
    for key in example.get("ui", {}):
        assert f'"{key}"' in app_src, f"[ui].{key} is documented but nothing reads it"
    adding = (_REPO / "docs" / "ADDING-A-CLI.md").read_text()
    sample = adding[adding.index("def is_available"):adding.index("def is_available") + 200]
    assert "shutil.which" not in sample, "ADDING-A-CLI's is_available() sample gates on the binary"
    assert "config.toml.example" in adding, "step 3 must point at the committed defaults"
    setup = (_REPO / "docs" / "SETUP.md").read_text()
    assert "python3.12" in setup and "sudo apt install python3 python3-venv" not in setup
    assert "refresh.out.log" in setup and "`refresh.log`" not in setup
    assert "claude mcp add" in setup.split("**Claude Code**")[1].split("**Codex")[0].split("\n")[0]
    for doc in (setup, (_REPO / "README.md").read_text()):
        assert not re.search(r"^\s*scripts/[a-z\-]+\.py", doc, re.M), "bare script invocation runs the system python3"
    readme_head = (_REPO / "README.md").read_text()[:600]
    assert "OpenCode" in readme_head, "README's pitch omits OpenCode"
    for skill in ("checkpoint", "snapshot"):
        head = (_REPO / "skills" / skill / "SKILL.md").read_text()[:400]
        assert "scaffold" in head.lower() and "not implemented" in head.lower(), skill
    for src in ("sources/base.py", "sources/registry.py"):
        head = (_REPO / src).read_text()[:700]
        assert "app.py and watcher.py" not in head and "config.toml.example" in head, src
    print("  ok  docs, config example, skills and docstrings match the code")


def test_spa_heatmap_cells_are_theme_aware():
    """The heatmap painted every empty cell with an inline dark-navy fill, so
    in light mode the activity chart was a black block (and a theme toggle
    never re-renders it). Cells carry hm0..hm4 classes styled per theme."""
    import shutil
    import subprocess
    html = (_REPO / "session-ui" / "static" / "index.html").read_text()
    assert ".hm0{fill:#e5e7eb}" in html and ".dark .hm0{fill:#1e293b}" in html, "missing theme rules"
    if not shutil.which("node"):
        print("  --  node not installed: heatmap check skipped")
        return
    src = _js_function_source("heatmap")
    driver = src + r"""
const svg = heatmap([{day: new Date().toISOString().slice(0,10), sessions: 3}]);
if (/fill="#1e293b"/.test(svg)) { console.error("inline dark fill"); process.exit(2); }
if (!/class="hm hm0"/.test(svg) || !/class="hm hm4"/.test(svg)) { console.error("classes missing: " + svg.slice(0, 200)); process.exit(3); }
console.log("ok");
"""
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "t.js"
        f.write_text(driver)
        p = subprocess.run(["node", str(f)], capture_output=True, text=True, timeout=30)
        assert p.returncode == 0, p.stderr[-300:]
    print("  ok  SPA heatmap: theme-aware cell classes, no inline dark fill")


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
