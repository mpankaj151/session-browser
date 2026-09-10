#!/usr/bin/env python3
"""OpenCode plugin hook — the Claude Stop-hook equivalent.

    opencode-hook.py <session_id>            a turn settled: index that session now
    opencode-hook.py <session_id> --deleted  the session was deleted: re-sync

Spawned (detached) by plugins/opencode/session-browser.js on session.idle /
session.deleted. session.idle fires for child sessions too, so the id is
resolved to its ROOT (read-only parent_id walk) and only that root is
re-projected, indexed and marked in the hook race-guard so the watcher skips
the same file moments later. --deleted runs a plain sync, which archives the
mirror file to the raw vault and unlinks it; the ordinary delete path then
archives the row as transcript-missing.

CONTRACT: this process always exits 0 and never blocks. Project imports live
inside main() so a broken config or missing dependency cannot change that.
No enrichment spawn — idle fires every turn; the nightly --enrich covers it.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
EXTRACT = _REPO / "scripts" / "extract-reasoning.py"


def _root_of(adapter, session_id: str) -> str | None:
    """Walk parent_id to the root, read-only. None if the id is unknown."""
    conn = adapter._open_ro()
    if conn is None:
        return None
    try:
        cur = session_id
        for _ in range(32):                       # cycle guard
            row = conn.execute("SELECT parent_id FROM session WHERE id = ?", (cur,)).fetchone()
            if row is None:
                return None
            if not row[0]:
                return cur
            cur = row[0]
        return None
    finally:
        conn.close()


def run(session_id: str, *, adapter=None, conn=None, deleted: bool = False, spawn: bool = True) -> str | None:
    """Index `session_id`'s root now. Returns the root id, or None when there
    was nothing to do. `spawn=False` skips the detached reasoning extraction."""
    import hookstate
    import indexer
    if adapter is None:
        from sources.registry import build_source_registry
        adapter = build_source_registry().get("opencode")
        if adapter is None:
            return None
    if deleted:
        adapter.sync()
        return None
    root = _root_of(adapter, session_id)
    if root is None:
        return None
    adapter.sync(only=[root], force=True)
    path = adapter.mirror_dir / f"{root}.jsonl"
    header = adapter.parse_header(path)
    if header is None:
        return None
    own = conn is None
    conn = conn or indexer.connect()
    try:
        indexer.upsert(header, conn=conn)
        if own:
            conn.commit()
    finally:
        if own:
            conn.close()
    hookstate.mark(root)
    if spawn:
        import subprocess
        import sbconfig
        try:
            log = open(sbconfig.LOG_DIR / "reasoning-hook.log", "a")
            subprocess.Popen([sys.executable, str(EXTRACT), "--source", "opencode",
                              "--session", str(path), "--archive"],
                             stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
        except Exception as e:  # noqa: BLE001
            print(f"[opencode-hook] spawn error: {e}", file=sys.stderr)
    return root


def main() -> None:
    try:
        args = [a for a in sys.argv[1:]]
        deleted = "--deleted" in args
        ids = [a for a in args if not a.startswith("--")]
        if not ids:
            return
        import sbconfig
        sbconfig.ensure_dirs()
        try:
            log = open(sbconfig.LOG_DIR / "opencode-hook.log", "a")
        except OSError:
            log = None
        root = run(ids[0], deleted=deleted)
        if log:
            from datetime import datetime, timezone
            log.write(f"{datetime.now(timezone.utc).isoformat()}  {'deleted' if deleted else 'idle'} "
                      f"{ids[0]} -> {root or '-'}\n")
            log.close()
    except BaseException as e:  # noqa: BLE001 — never surface inside OpenCode
        try:
            print(f"[opencode-hook] {type(e).__name__}: {e}", file=sys.stderr)
        except Exception:  # noqa: BLE001
            pass
    finally:
        sys.exit(0)


if __name__ == "__main__":
    main()
