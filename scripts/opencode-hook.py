#!/usr/bin/env python3
"""OpenCode plugin hook — the Claude Stop-hook equivalent.

    opencode-hook.py <session_id>            a turn settled: index that session now
    opencode-hook.py <session_id> --deleted  the session was deleted: re-sync

Vocabulary (docs/GLOSSARY.md): a *session* is one conversation with a coding CLI; OpenCode
calls its hook mechanism a *plugin* (a JavaScript file it loads at start-up); *indexing*
means writing a session's cheap header facts into registry.db; the *mirror* is the JSONL
file this repo projects out of OpenCode's SQLite database, because every other part of the
tool assumes one plain-text transcript file per session.

Spawned (detached) by plugins/opencode/session-browser.js on session.idle /
session.deleted. session.idle fires for child sessions too, so the id is
resolved to its ROOT (read-only parent_id walk) and only that root is
re-projected, indexed and marked in the hook race-guard so the watcher skips
the same file moments later. --deleted runs a plain sync, which archives the
mirror file to the raw vault and unlinks it; the ordinary delete path then
archives the row as transcript-missing.

Why child-to-root matters: OpenCode starts a helper conversation ("child" / subagent) as a
row in the same `session` table, linked to its parent by `parent_id`. Only the root is a
session in this tool's sense — children roll their turns and cost into it. Indexing a child
id directly would create a duplicate row that the rest of the pipeline cannot restore or
resume, so the id from the plugin is always walked up first.

CONTRACT: this process always exits 0 and never blocks. Project imports live
inside main() so a broken config or missing dependency cannot change that.
No enrichment spawn — idle fires every turn; the nightly --enrich covers it.
Same reason as the Claude hook: an error surfacing out of a plugin would show up inside the
user's OpenCode session. tests/test_smoke.py runs this script with a bad id, no arguments,
an unknown flag and a missing database, and asserts exit 0 every time.
"""
from __future__ import annotations

import sys
from pathlib import Path

# OpenCode runs this file by absolute path; the repo root is not on sys.path by default.
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
EXTRACT = _REPO / "scripts" / "extract-reasoning.py"


def _root_of(adapter, session_id: str) -> str | None:
    """Walk parent_id to the root, read-only. None if the id is unknown.

    OpenCode's `session` table stores child (subagent) conversations alongside their
    parents, linked by `parent_id`; a root session has it empty. The SELECT below fetches
    one parent link at a time and follows it until the link is empty.

    Read-only on purpose: `adapter._open_ro()` opens opencode.db with `mode=ro` so this
    hook can never write to, or rewrite the write-ahead log of, the database OpenCode is
    actively using. (The OpenCode binary itself is never invoked on this path for the same
    reason — even `opencode db path` rewrote the WAL.)

    Returns the root id, or None when the id is not in the database at all (already deleted,
    or a different OpenCode install) — the caller then does nothing.
    """
    conn = adapter._open_ro()
    if conn is None:
        return None
    try:
        cur = session_id
        for _ in range(32):                       # cycle guard
            # Bounded to 32 hops so a corrupt parent_id cycle (a -> b -> a) ends in None
            # instead of spinning forever inside a user's editor session.
            row = conn.execute("SELECT parent_id FROM session WHERE id = ?", (cur,)).fetchone()
            if row is None:
                return None                       # id not in this DB
            if not row[0]:
                return cur                        # empty parent_id == this IS the root
            cur = row[0]
        return None
    finally:
        conn.close()


def run(session_id: str, *, adapter=None, conn=None, deleted: bool = False, spawn: bool = True) -> str | None:
    """Index `session_id`'s root now. Returns the root id, or None when there
    was nothing to do. `spawn=False` skips the detached reasoning extraction.

    This is the whole hook, factored out of main() so tests can drive it in-process.
    tests/test_smoke.py calls it as `run(child_id, adapter=..., conn=..., spawn=False)` and
    asserts the child resolves to the root, that ONLY the root's mirror file was written,
    that the row landed, and that the race guard marks the root and not the child.

    Arguments worth spelling out:
      adapter  the OpenCode source adapter; built from the registry when omitted.
      conn     an open registry connection to reuse. When omitted this function opens and
               commits its own — the hook is a one-shot process, so that is the normal case.
      deleted  the plugin saw session.deleted rather than session.idle.
      spawn    False suppresses the detached reasoning-extraction child (tests, dry runs).

    Side effects: rewrites one mirror JSONL, upserts one registry row, writes the race-guard
    mark, and (unless spawn=False) starts one detached subprocess.
    """
    # Imported here, not at module level: an import failure at module scope would make the
    # process exit nonzero before main()'s guard could catch it.
    import hookstate
    import indexer
    if adapter is None:
        from sources.registry import build_source_registry
        adapter = build_source_registry().get("opencode")
        if adapter is None:
            return None
    if deleted:
        # A FULL sync, not `only=[id]`: the row is already gone from opencode.db, so there
        # is nothing to project. What we need is the sweep that notices a mirror file with
        # no matching DB row, copies it into the raw vault and unlinks it. The ordinary
        # delete path (watcher or prune) then archives the registry row as
        # transcript-missing, which keeps it visible and restorable.
        adapter.sync()
        return None
    root = _root_of(adapter, session_id)
    if root is None:
        return None
    # `only=[root]` limits the projection to this one session tree — a full sync would
    # re-read every session in the database on every settled turn. `force=True` bypasses
    # the fingerprint manifest that normally skips unchanged trees: the whole point of the
    # hook is that this tree changed a moment ago, and the fingerprint may not reflect the
    # write yet.
    adapter.sync(only=[root], force=True)
    path = adapter.mirror_dir / f"{root}.jsonl"
    header = adapter.parse_header(path)
    if header is None:
        return None
    # Tests pass a connection in and commit themselves; the real hook owns its connection.
    own = conn is None
    conn = conn or indexer.connect()
    try:
        indexer.upsert(header, conn=conn)
        if own:
            conn.commit()
    finally:
        if own:
            conn.close()
    # Mark the ROOT (the id that was indexed): within 30 s the watcher will see the mirror
    # file change and must skip it rather than index the same session again.
    hookstate.mark(root)
    if spawn:
        import subprocess
        import sbconfig
        try:
            log = open(sbconfig.LOG_DIR / "reasoning-hook.log", "a")
            # No --archive here: session.idle fires per TURN, and archive_raw
            # writes a new @vN copy whenever the size changed — a 60-turn session
            # left ~60 full copies. The nightly run and archive-then-unlink on
            # deletion own the raw vault; this only refreshes the trail.
            subprocess.Popen([sys.executable, str(EXTRACT), "--source", "opencode",
                              "--session", str(path)],
                             stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
        except Exception as e:  # noqa: BLE001
            # Swallowed: the trail is a nice-to-have that the nightly run rebuilds anyway,
            # and nothing the plugin does may raise inside OpenCode.
            print(f"[opencode-hook] spawn error: {e}", file=sys.stderr)
    return root


def main() -> None:
    """Command-line entry point: parse argv, run the hook, append one audit line, exit 0.

    Everything lives inside one try/finally whose finally is `sys.exit(0)` — the contract.
    The audit line in opencode-hook.log is how you tell "the plugin never fired" apart from
    "the plugin fired and found nothing", which is otherwise invisible from inside OpenCode.
    """
    try:
        args = [a for a in sys.argv[1:]]
        deleted = "--deleted" in args
        # Positional arguments are session ids; anything starting with "--" is a flag. Only
        # the first id is used — the plugin spawns one process per event.
        ids = [a for a in args if not a.startswith("--")]
        if not ids:
            return
        import sbconfig
        sbconfig.ensure_dirs()
        try:
            log = open(sbconfig.LOG_DIR / "opencode-hook.log", "a")
        except OSError:
            log = None  # unwritable log dir must not stop the indexing below
        root = run(ids[0], deleted=deleted)
        if log:
            # One line per event, e.g.
            #   2026-09-11T08:12:04+00:00  idle ses_7f21… -> ses_1a90…
            # "-" as the target means the id resolved to nothing (unknown / deleted).
            from datetime import datetime, timezone
            log.write(f"{datetime.now(timezone.utc).isoformat()}  {'deleted' if deleted else 'idle'} "
                      f"{ids[0]} -> {root or '-'}\n")
            log.close()
    except BaseException as e:  # noqa: BLE001 — never surface inside OpenCode
        # BaseException, not Exception: SystemExit or KeyboardInterrupt escaping here would
        # become a nonzero exit, which the plugin would report to the user mid-session.
        try:
            print(f"[opencode-hook] {type(e).__name__}: {e}", file=sys.stderr)
        except Exception:  # noqa: BLE001
            pass  # reporting the failure must not itself fail
    finally:
        # In `finally`, so even the success path and the early `return` above exit 0.
        sys.exit(0)


if __name__ == "__main__":
    main()
