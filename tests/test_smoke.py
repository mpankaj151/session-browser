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
    from sources.opencode import OpenCodeSource
    reg = build_source_registry()
    assert "claude" in reg and "copilot" in reg and "opencode" in reg, list(reg)
    assert isinstance(reg["opencode"], OpenCodeSource)
    assert reg["opencode"].session_id_for_path(Path(f"/m/{_OC_ROOT}.jsonl")) == _OC_ROOT
    assert ClaudeSource().session_id_for_path(Path("/p/abc-1.jsonl")) == "abc-1"
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
    src = ClaudeSource()
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

            indexer.upsert(_header("cp-1", cli_source="copilot",
                                   project_path=str(root / "copilot" / "cp-1")), conn=conn)
            indexer.archive("cp-1", indexer.TRANSCRIPT_MISSING, conn=conn)
            assert restore.restore_session("cp-1", conn=conn, registry=registry).status == "unsupported"

            assert not (projects).exists(), "no refusal may write into the projects dir"
            for sid in ("agent-z", "no-copy", "cp-1"):
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
    orig_connect, old_archive = indexer.connect, reasoning.ARCHIVE
    try:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            reasoning.ARCHIVE = root
            raw = root / "raw" / "2026" / "05"
            raw.mkdir(parents=True)
            (raw / "aged-has-copy.jsonl").write_text(_cl_transcript("x"))
            indexer.upsert(_header("live"), conn=conn)
            conn.execute("UPDATE sessions SET cost_usd=1.0 WHERE session_id='live'")
            for sid in ("aged-has-copy", "aged-no-copy"):
                indexer.upsert(_header(sid), conn=conn)
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
        indexer.connect, reasoning.ARCHIVE = orig_connect, old_archive
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
    orig_connect, old_archive, old_sources = indexer.connect, reasoning.ARCHIVE, sb.SOURCES
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
            assert src.watch_roots() == [src.mirror_dir, src.data_dir]
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
            assert not src.mirror_dir.exists()
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
