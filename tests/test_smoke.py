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
        indexer.archive("__smoke__", conn=conn)
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
