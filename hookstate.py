"""The hook race-guard state, shared by every "index it now" hook and the watcher.

A hook (Claude's Stop hook, the OpenCode plugin hook) indexes a session the
instant it ends and records the id here; the watcher, which sees the same file
change moments later, skips anything hooked within RACE_GUARD_S so the two paths
never double-process. Ids never collide across CLIs (uuid vs ses_…), so one file
serves all of them.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import sbconfig

RACE_GUARD_S = 30
# Entries older than this are useless to the guard; pruning on every write
# keeps the file from growing forever.
STATE_TTL_S = 300


def _load(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def mark(session_id: str, path: Path | None = None) -> None:
    """Record that `session_id` was just indexed by a hook. Atomic write, TTL prune."""
    path = Path(path) if path else sbconfig.HOOK_STATE
    now = datetime.now(timezone.utc)
    state = {}
    for sid, ts in _load(path).items():
        try:
            if (now - datetime.fromisoformat(ts)).total_seconds() < STATE_TTL_S:
                state[sid] = ts
        except (TypeError, ValueError):
            continue
    state[session_id] = now.isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    import tempfile
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(state))
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def recently(session_id: str, within_s: float = RACE_GUARD_S, path: Path | None = None) -> bool:
    """Was `session_id` hooked within the last `within_s` seconds?"""
    path = Path(path) if path else sbconfig.HOOK_STATE
    ts = _load(path).get(session_id)
    if not ts:
        return False
    try:
        return (datetime.now(timezone.utc) - datetime.fromisoformat(ts)).total_seconds() < within_s
    except (TypeError, ValueError):
        return False
