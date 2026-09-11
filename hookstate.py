"""The hook race-guard state, shared by every "index it now" hook and the watcher.

A hook (Claude's Stop hook, the OpenCode plugin hook) indexes a session the
instant it ends and records the id here; the watcher, which sees the same file
change moments later, skips anything hooked within RACE_GUARD_S so the two paths
never double-process. Ids never collide across CLIs (uuid vs ses_…), so one file
serves all of them.

Background for a reader new to this vocabulary (fuller definitions in docs/GLOSSARY.md):
a *session* is one conversation with a coding CLI; a *hook* is a small program the CLI
runs at a fixed moment, here when a session ends; the *watcher* is our always-on
background process that notices transcript files changing on disk. Both paths lead to
the same registry row, which is why they have to be told apart.

The state itself is one small JSON object, `{"<session-id>": "<UTC ISO timestamp>"}`,
at sbconfig.HOOK_STATE (default `~/.session-browser/.hook-state.json`), e.g.

    {"6550180f-14ff-4b91-a93d-d951ed98c2f7": "2026-06-19T12:00:00.123456+00:00"}

Design notes:
  * It is a hint, never a lock. Worst case on a lost or corrupted file is that the
    watcher indexes a session a second time — an upsert, so harmless. That is why every
    read swallows its errors and returns "not recently hooked" (see _load()).
  * Writers are separate short-lived processes (one per hook invocation) with no
    coordination, so mark() rewrites the whole file atomically instead of appending.
  * Nothing ever deletes entries explicitly; each write drops entries older than
    STATE_TTL_S, which keeps the file a few lines long forever.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import sbconfig

# How long a hook's claim on a session suppresses the watcher. Generous on purpose: the
# window has to cover the hook's own work (parse + upsert) plus the filesystem event's
# trip through the watcher's 0.5 s debounce, on a loaded laptop. Indexing twice is
# harmless, so erring long costs nothing; erring short would.
RACE_GUARD_S = 30
# Entries older than this are useless to the guard; pruning on every write
# keeps the file from growing forever.
STATE_TTL_S = 300


def _load(path: Path) -> dict:
    """Read the state file, or {} if it is missing, unreadable, half-written or not a
    JSON object.

    Every failure is deliberately silent. This file is advisory: a mangled one must make
    the watcher fall back to "nobody hooked this session", never raise inside a hook that
    is contractually required to exit 0 and never block the CLI that spawned it.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def mark(session_id: str, path: Path | None = None) -> None:
    """Record that `session_id` was just indexed by a hook. Atomic write, TTL prune.

    Called by every hook right after its upsert. `path` overrides the configured state
    file and exists for the tests. Side effect: rewrites the whole state file (creating
    its parent directory if needed); entries older than STATE_TTL_S are dropped on the
    way through, so this is also the only garbage collection the file gets.

    Raises only if the temporary file cannot be written or renamed — a caller inside a
    hook is expected to swallow that, since failing to mark is not worth failing a
    session end over.
    """
    path = Path(path) if path else sbconfig.HOOK_STATE
    now = datetime.now(timezone.utc)
    state = {}
    for sid, ts in _load(path).items():
        try:
            # A timestamp we cannot parse (older format, truncated write) is simply
            # dropped: it can no longer suppress anything, which is the safe direction.
            if (now - datetime.fromisoformat(ts)).total_seconds() < STATE_TTL_S:
                state[sid] = ts
        except (TypeError, ValueError):
            continue
    state[session_id] = now.isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Imported here, not at module scope: hooks run on every session end and this is the
    # only code path that needs tempfile, so the import cost is paid only when writing.
    import tempfile
    # Write-then-rename. os.replace() is atomic within a filesystem, so a reader either
    # sees the whole old file or the whole new one — never a half-written object. The
    # temp file is created in the SAME directory precisely so the rename stays on one
    # filesystem, and mkstemp gives each racing hook process its own unique name.
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(state))
        os.replace(tmp, path)
    except BaseException:
        # BaseException, not Exception: a KeyboardInterrupt or SystemExit mid-write must
        # still clean up, or the state directory slowly fills with .tmp leftovers. The
        # original error is re-raised untouched after the cleanup attempt.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def recently(session_id: str, within_s: float = RACE_GUARD_S, path: Path | None = None) -> bool:
    """Was `session_id` hooked within the last `within_s` seconds?

    The watcher asks this before every upsert and skips the session when it is True.
    Answers False for anything it cannot establish — no file, no entry, unparseable
    timestamp — because the fallback (index it again) is safe and silently skipping a
    real session is not. `within_s` and `path` are overridable for the tests.
    """
    path = Path(path) if path else sbconfig.HOOK_STATE
    ts = _load(path).get(session_id)
    if not ts:
        return False
    try:
        return (datetime.now(timezone.utc) - datetime.fromisoformat(ts)).total_seconds() < within_s
    except (TypeError, ValueError):
        return False
