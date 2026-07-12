#!/usr/bin/env python3
"""Claude Code Stop + SessionEnd hook — instant indexing, deferred heavy work.

Two tiers so perceived latency stays in the tens of milliseconds:
  Inline (waited): read transcript_path from stdin JSON -> parse_header ->
                   indexer.upsert -> write .hook-state.json -> exit.
  Deferred (detached, not waited): spawn extract-reasoning.py to archive the raw
                   transcript and render the readable decision trail. On
                   SessionEnd ONLY, additionally spawn
                   `enrich-sessions.py --session <id>` so the session gets its
                   journal-grade summary the moment it ends. (Stop fires after
                   EVERY assistant response — enriching there would pay one LLM
                   call per turn; SessionEnd fires once, and the enricher's
                   staleness check makes a no-activity re-fire cost nothing.)

Registered in ~/.claude/settings.json under BOTH events (install.sh does this):
  { "hooks": { "Stop":       [ { "hooks": [ { "type": "command", "command": CMD } ] } ],
               "SessionEnd": [ { "hooks": [ { "type": "command", "command": CMD } ] } ] } }
  where CMD = "\"<venv-python>\" \"<repo>/scripts/session-hook.py\""
The command MUST use the absolute venv interpreter (launchd/hook PATH lacks pyenv).

CONTRACT: this process must ALWAYS exit 0. A nonzero Stop-hook exit blocks
Claude Code's session end, so even an unimportable config or a missing
dependency must degrade to a silent no-op. Hence all project imports happen
inside the guarded main(), and __main__ catches BaseException.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

EXTRACT = _REPO / "scripts" / "extract-reasoning.py"
ENRICH = _REPO / "scripts" / "enrich-sessions.py"

# Our own enrichment providers run `claude --print` with this set, so the hook
# their headless sessions trigger is a no-op (belt to the entrypoint=="sdk-cli"
# suspenders in the adapter).
_SUPPRESS_ENV = "SESSION_BROWSER_SUPPRESS_HOOK"

# Entries older than this are useless to the watcher's 30s race-guard; pruning
# on every write keeps .hook-state.json from growing forever.
_STATE_TTL_S = 300


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


def _write_hook_state(hook_state: Path, session_id: str) -> None:
    state = {}
    if hook_state.exists():
        try:
            state = json.loads(hook_state.read_text())
        except (json.JSONDecodeError, OSError):
            state = {}
    now = datetime.now(timezone.utc)
    state[session_id] = now.isoformat()
    # prune stale entries; tolerate junk values
    fresh = {}
    for sid, ts in state.items():
        try:
            if (now - datetime.fromisoformat(ts)).total_seconds() < _STATE_TTL_S:
                fresh[sid] = ts
        except (ValueError, TypeError):
            continue
    hook_state.parent.mkdir(parents=True, exist_ok=True)
    # atomic replace: a concurrent reader never sees a half-written file
    tmp = hook_state.with_suffix(f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(fresh))
    os.replace(tmp, hook_state)


def main() -> None:
    if os.environ.get(_SUPPRESS_ENV):
        return  # our own headless enrichment call — nothing to index

    payload = _read_payload()
    path = _transcript_path(payload)
    if path is None or not path.exists():
        return  # nothing to do; never block the CLI

    # Project imports INSIDE the guard: a broken config.toml or missing dep must
    # not take down the hook (see module docstring contract).
    import indexer
    import sbconfig
    from sources.claude import ClaudeSource

    sbconfig.ensure_dirs()

    # --- inline tier: cheap upsert ---
    header = None
    try:
        header = ClaudeSource().parse_header(path)
        if header is not None:
            indexer.upsert(header)
            _write_hook_state(sbconfig.HOOK_STATE, header.session_id)
    except Exception as e:  # noqa: BLE001 — a hook must never crash the session
        print(f"[session-hook] index error: {e}", file=sys.stderr)

    # --- deferred tier: detached archive + reasoning render (fire-and-forget) ---
    # header is None for headless/sdk-cli and unparseable transcripts — the same
    # sessions the index excludes, so don't archive/render them either.
    if header is None:
        return
    try:
        log = open(sbconfig.LOG_DIR / "reasoning-hook.log", "a")
        subprocess.Popen(
            [sys.executable, str(EXTRACT), "--session", str(path), "--archive"],
            stdin=subprocess.DEVNULL, stdout=log, stderr=log,
            start_new_session=True,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[session-hook] spawn error: {e}", file=sys.stderr)

    # --- SessionEnd only: journal-grade enrichment for the ended session ---
    # Detached like the reasoning tier; the enricher itself skips fresh sessions
    # (staleness check) and its headless LLM child inherits _SUPPRESS_ENV, so
    # this can neither block shutdown nor recurse.
    if payload.get("hook_event_name") == "SessionEnd":
        try:
            elog = open(sbconfig.LOG_DIR / "enrich-hook.log", "a")
            subprocess.Popen(
                [sys.executable, str(ENRICH), "--session", header.session_id],
                stdin=subprocess.DEVNULL, stdout=elog, stderr=elog,
                start_new_session=True,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[session-hook] enrich spawn error: {e}", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:  # noqa: BLE001 — contract: always exit 0
        try:
            print(f"[session-hook] fatal (suppressed): {e}", file=sys.stderr)
        except Exception:  # noqa: BLE001
            pass
    sys.exit(0)
