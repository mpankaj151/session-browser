#!/usr/bin/env python3
"""Single-CLI laptop scenarios — the promise that Session Browser works on any
laptop that has ANY ONE of claude / codex / copilot / opencode (or only the
transcripts of one whose binary is gone).

Each scenario seeds ONE CLI's sessions into an isolated HOME, puts only that
CLI's stub binary on a stripped PATH, points the config at an empty override
file (so config.toml.example's defaults apply, exactly like a fresh install),
and runs the REAL scripts as subprocesses: the nightly pipeline, enrichment,
and `cr`. Runs standalone, no pytest:

    .venv/bin/python tests/test_portability.py
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "tests"))

import test_smoke as T  # noqa: E402  — fixture builders live there

FACET = json.dumps({"brief_summary": "Stub summary.", "goal_categories": {"testing": 1},
                    "session_type": "debugging", "outcome": "completed", "key_decisions": ["d1"],
                    "files_touched": [], "accomplishments": ["a1"], "open_threads": [],
                    "goal": "g", "reusability": ""})
CL_SID = "0a1b2c3d-1111-4222-8333-444455556666"
CP_SID = "b2c3d4e5-2222-4333-8444-555566667777"

# Stub CLIs: log every invocation; answer the headless-enrichment call shapes
# each real CLI has (claude: prompt on stdin; copilot: prompt as -p argv;
# opencode: prompt on stdin, NDJSON out) with the fixed facet above.
_STUBS = {
    "claude": r'''echo "claude $*" >> "$STUB_LOG"
case " $* " in *" --print "*) cat >/dev/null; printf '%s' "$FACET";; *) echo "claude-stub $*";; esac''',
    "copilot": r'''echo "copilot $1" >> "$STUB_LOG"
case "$1" in -p) printf '%s' "$FACET";; *) echo "copilot-stub $*";; esac''',
    "codex": r'''echo "codex $*" >> "$STUB_LOG"
echo "codex-stub $*"''',
    "opencode": r'''echo "opencode $*" >> "$STUB_LOG"
case "$1" in
  run) cat >/dev/null
       printf '{"type":"text","part":{"text":%s}}\n' "$FACET_JSON"
       printf '{"type":"step_finish","part":{"cost":0.01,"tokens":{"input":10,"output":5}}}\n';;
  *) echo "opencode-stub $*";;
esac''',
}


def _seed(home: Path, cli: str) -> str:
    """One indexable session for `cli` under home; returns its id."""
    if cli == "claude":
        d = home / ".claude" / "projects" / "-Users-x-proj"
        d.mkdir(parents=True)
        (d / f"{CL_SID}.jsonl").write_text(T._cl_transcript("hello claude", cwd="/Users/x/proj"))
        return CL_SID
    if cli == "codex":   # a COLD rollout: zstd-compressed in place, as Codex does after ~7 days
        T._cx_tree(home / ".codex", T._cx_rollout(), root="sessions", compress=True)
        return T._CX_ID
    if cli == "copilot":
        d = home / ".copilot" / "session-state" / CP_SID
        d.mkdir(parents=True)
        (d / "workspace.yaml").write_text(
            "cwd: /Users/x/proj\nname: hello copilot\n"
            "created_at: 2026-08-01T10:00:00Z\nupdated_at: 2026-08-01T10:05:00Z\n")
        events = [{"type": "session.model_change", "data": {"newModel": "gpt-5.4"}},
                  {"type": "user.message", "data": {"content": "hello copilot"}},
                  {"type": "assistant.message", "data": {"content": "hi there"}}]
        (d / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
        return CP_SID
    if cli == "opencode":
        d = home / ".local" / "share" / "opencode"
        d.mkdir(parents=True)
        conn = sqlite3.connect(str(d / "opencode.db"))
        T._oc_schema(conn)
        T._oc_seed(conn)
        conn.close()
        return T._OC_ROOT
    raise ValueError(cli)


class Laptop:
    """An isolated HOME with one CLI's sessions and (optionally) its stub binary."""

    def __init__(self, cli: str, with_binary: bool = True):
        self.cli, self.with_binary = cli, with_binary
        self.tmp = Path(tempfile.mkdtemp(prefix=f"sb-laptop-{cli}-"))
        self.home = self.tmp / "home"
        self.home.mkdir()
        self.bins = self.tmp / "bin"
        self.bins.mkdir()
        self.stub_log = self.tmp / "stub.log"
        if with_binary:
            stub = self.bins / cli
            stub.write_text("#!/usr/bin/env bash\n" + _STUBS[cli] + "\n")
            stub.chmod(0o755)
        self.sid = _seed(self.home, cli)
        (self.tmp / "config.toml").write_text("# fresh laptop: no overrides\n")
        env = {k: v for k, v in os.environ.items()
               if k not in ("XDG_DATA_HOME", "OPENCODE_DB", "CLAUDE_CONFIG_DIR", "CODEX_HOME")}
        env.update({
            "HOME": str(self.home), "PATH": f"{self.bins}:/usr/bin:/bin",
            "SB_CONFIG": str(self.tmp / "config.toml"),
            "SB_DB": str(self.home / ".session-browser" / "registry.db"),
            # never fetch the embedding model in a test; the step must SKIP, not fail
            "SB_ALLOW_MODEL_DOWNLOAD": "0", "HF_HUB_OFFLINE": "1",
            "STUB_LOG": str(self.stub_log), "FACET": FACET, "FACET_JSON": json.dumps(FACET),
        })
        self.env = env

    def run(self, *cmd: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(list(cmd), env=self.env, capture_output=True, text=True,
                              timeout=600, cwd=str(cwd or self.home))

    def script(self, name: str, *args: str) -> subprocess.CompletedProcess:
        return self.run(sys.executable, str(_REPO / "scripts" / name), *args)

    def rows_by_source(self) -> dict[str, int]:
        db = Path(self.env["SB_DB"])
        if not db.exists():
            return {}
        conn = sqlite3.connect(str(db))
        try:
            return dict(conn.execute("SELECT cli_source, COUNT(*) FROM sessions GROUP BY 1").fetchall())
        finally:
            conn.close()

    def facet(self) -> dict | None:
        p = self.home / ".session-browser" / "facets" / f"{self.sid}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def raw_copies(self) -> list[Path]:
        root = self.home / "claude-reasoning-archive" / "raw"
        return sorted(root.rglob(f"{self.sid}*.jsonl")) if root.exists() else []

    def stub_calls(self) -> list[str]:
        return self.stub_log.read_text().splitlines() if self.stub_log.exists() else []


def _out(p: subprocess.CompletedProcess) -> str:
    return (p.stdout or "") + (p.stderr or "")


def _check_pipeline(lap: Laptop) -> None:
    r = lap.script("refresh-all.py")
    assert r.returncode == 0, f"[{lap.cli}] refresh-all exit {r.returncode}:\n{_out(r)[-1500:]}"
    assert "✗" not in _out(r), _out(r)[-1500:]
    assert lap.rows_by_source() == {lap.cli: 1}, (lap.cli, lap.rows_by_source())
    assert lap.raw_copies(), f"[{lap.cli}] no raw archive copy — the session is not restorable"


def _check_enrichment(lap: Laptop, expect_provider: str | None) -> None:
    r = lap.script("enrich-sessions.py")
    assert r.returncode == 0, f"[{lap.cli}] enrich exit {r.returncode}:\n{_out(r)[-800:]}"
    facet = lap.facet()
    if expect_provider is None:
        assert facet is None and "enrichment skipped" in _out(r), _out(r)[-400:]
        return
    assert facet is not None, f"[{lap.cli}] no facet written:\n{_out(r)[-800:]}"
    assert facet["_meta"]["provider"] == expect_provider, facet["_meta"]
    assert facet["brief_summary"] == "Stub summary."


def _check_resume(lap: Laptop, expect_fragment: str | None) -> None:
    r = lap.run("bash", str(_REPO / "bin" / "resume-here.sh"), lap.sid)
    if expect_fragment is None:   # binary absent: a clear refusal, not exit 127
        assert r.returncode == 1 and "not on PATH" in r.stderr, (r.returncode, _out(r))
        return
    assert r.returncode == 0, (lap.cli, r.returncode, _out(r))
    assert expect_fragment in r.stdout, (lap.cli, r.stdout)


def test_claude_only_laptop():
    lap = Laptop("claude")
    _check_pipeline(lap)
    _check_enrichment(lap, "claude-headless")
    assert any(c.startswith("claude --print") for c in lap.stub_calls()), lap.stub_calls()
    _check_resume(lap, f"claude-stub --resume {lap.sid}")
    print("  ok  claude-only laptop: index, auto-enrich via claude, cr")


def test_codex_only_laptop():
    lap = Laptop("codex")
    _check_pipeline(lap)
    _check_enrichment(lap, None)          # no summariser CLI: clean skip, exit 0
    _check_resume(lap, f"codex-stub resume {lap.sid}")   # cold (.zst) rollout
    print("  ok  codex-only laptop: index a cold rollout, enrichment skips cleanly, cr")


def test_copilot_only_laptop():
    lap = Laptop("copilot")
    _check_pipeline(lap)
    _check_enrichment(lap, "copilot-headless")
    _check_resume(lap, f"copilot-stub --resume={lap.sid}")
    print("  ok  copilot-only laptop: index, auto-enrich via copilot, cr")


def test_opencode_only_laptop():
    lap = Laptop("opencode")
    _check_pipeline(lap)
    assert (lap.home / ".session-browser" / "opencode-mirror" / f"{lap.sid}.jsonl").exists()
    _check_enrichment(lap, "opencode-headless")
    run_call = next((c for c in lap.stub_calls() if c.startswith("opencode run")), "")
    assert "--pure" in run_call and "--auto" not in run_call, run_call
    _check_resume(lap, f"opencode-stub --session {lap.sid}")
    print("  ok  opencode-only laptop: mirror + index, auto-enrich via opencode run, cr")


def test_claude_transcripts_without_binary():
    """The work-laptop trajectory: Claude Code uninstalled, its transcripts
    still on disk. They must stay indexed and restorable; resume says why not."""
    lap = Laptop("claude", with_binary=False)
    _check_pipeline(lap)
    _check_enrichment(lap, None)
    _check_resume(lap, None)
    print("  ok  claude transcripts with no binary: still indexed + archived; cr explains")


if __name__ == "__main__":
    print("Session Browser portability scenarios (one CLI per laptop)")
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for fn in tests:
        try:
            fn()
        except Exception as e:  # noqa: BLE001 — report, keep running the rest
            failures += 1
            print(f"  FAIL {fn.__name__}: {e}")
    if failures:
        print(f"\n{failures}/{len(tests)} scenario(s) FAILED.")
        sys.exit(1)
    print(f"\nAll {len(tests)} scenarios passed.")
