#!/usr/bin/env python3
"""Claude Code Stop hook — instant indexing + deferred reasoning archive.

Two tiers so perceived latency stays in the tens of milliseconds:
  Inline (waited): read transcript_path from stdin JSON -> parse_header ->
                   indexer.upsert -> write .hook-state.json -> exit.
  Deferred (detached, not waited): spawn extract-reasoning.py to archive the raw
                   transcript and render the readable decision trail.

Registered in ~/.claude/settings.json as:
  { "hooks": { "Stop": [ { "hooks": [ {
      "type": "command",
      "command": "<venv-python> <repo>/scripts/session-hook.py" } ] } ] } }
The command MUST use the absolute venv interpreter (launchd/hook PATH lacks pyenv).
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import indexer  # noqa: E402
import sbconfig  # noqa: E402
from sources.claude import ClaudeSource  # noqa: E402

VENV_PY = _REPO / ".venv" / "bin" / "python"
EXTRACT = _REPO / "scripts" / "extract-reasoning.py"


def _read_payload() -> dict:
    try:
        raw = sys.stdin.read()
        return json.loads(raw) if raw.strip() else {}
    except (json.JSONDecodeError, ValueError):
        return {}


def _transcript_path(payload: dict) -> Path | None:
    for key in ("transcript_path", "transcriptPath", "transcript"):
        if payload.get(key):
            return Path(payload[key]).expanduser()
    # fallback: positional arg
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).expanduser()
    return None


def _write_hook_state(session_id: str) -> None:
    state = {}
    if sbconfig.HOOK_STATE.exists():
        try:
            state = json.loads(sbconfig.HOOK_STATE.read_text())
        except (json.JSONDecodeError, OSError):
            state = {}
    state[session_id] = datetime.now(timezone.utc).isoformat()
    sbconfig.HOOK_STATE.parent.mkdir(parents=True, exist_ok=True)
    sbconfig.HOOK_STATE.write_text(json.dumps(state))


def main() -> None:
    sbconfig.ensure_dirs()
    payload = _read_payload()
    path = _transcript_path(payload)
    if path is None or not path.exists():
        sys.exit(0)  # nothing to do; never block the CLI

    # --- inline tier: cheap upsert ---
    try:
        header = ClaudeSource().parse_header(path)
        if header is not None:
            indexer.upsert(header)
            _write_hook_state(header.session_id)
    except Exception as e:  # noqa: BLE001 — a hook must never crash the session
        print(f"[session-hook] index error: {e}", file=sys.stderr)

    # --- deferred tier: detached archive + reasoning render (fire-and-forget) ---
    try:
        log = open(sbconfig.LOG_DIR / "reasoning-hook.log", "a")
        subprocess.Popen(
            [str(VENV_PY), str(EXTRACT), "--session", str(path), "--archive"],
            stdin=subprocess.DEVNULL, stdout=log, stderr=log,
            start_new_session=True,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[session-hook] spawn error: {e}", file=sys.stderr)

    sys.exit(0)


if __name__ == "__main__":
    main()
