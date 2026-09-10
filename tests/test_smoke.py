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
    reg = build_source_registry()
    assert "claude" in reg and "copilot" in reg
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
