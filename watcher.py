#!/usr/bin/env python3
"""Filesystem watcher daemon — covers every enabled source.

watchdog Observer over each source directory with a 500ms per-path debounce.
On create/modify -> resolve owning adapter -> parse_header -> indexer.upsert.
On delete -> indexer.archive. For Claude files it consults .hook-state.json and
skips any session the Stop hook indexed within the last 30s (race-guard), so the
two indexing paths never double-process the same file.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

_REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO))

import hookstate  # noqa: E402
import indexer  # noqa: E402
import sbconfig  # noqa: E402
from sources.registry import build_source_registry  # noqa: E402

DEBOUNCE_S = 0.5
# A DB-backed source (OpenCode) reports a change trigger per WAL write, which
# is many times a second while a session is active; one re-sync per burst.
SYNC_DEBOUNCE_S = 2.0
_SYNC_KEY = "__sync__"
LOG = sbconfig.LOG_DIR / "watcher.log"


_LOG_MAX_BYTES = 5 * 1024 * 1024


def _log(msg: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()}  {msg}\n"
    try:
        # crude size cap: a long-lived daemon logging one line per index event
        # would otherwise grow the file forever
        if LOG.exists() and LOG.stat().st_size > _LOG_MAX_BYTES:
            LOG.rename(LOG.with_suffix(".log.1"))
        with open(LOG, "a") as fh:
            fh.write(line)
    except OSError:
        pass
    print(line, end="")


def _build_watch_pairs() -> list[tuple[Path, object]]:
    """(directory, adapter) pairs for every available source.

    An adapter may own more than one root — codex spreads live and archived
    rollouts across two sibling trees — so it gets to declare them via an
    optional watch_roots(). Sources without one keep the single configured dir.
    """
    pairs = []
    for name, adapter in build_source_registry(only_available=True).items():
        roots = getattr(adapter, "watch_roots", None)
        if callable(roots):
            pairs.extend((Path(d).expanduser(), adapter) for d in roots())
            continue
        cfg = sbconfig.source_config(name)
        d = cfg.get("projects_dir") or cfg.get("state_dir") or cfg.get("sessions_dir")
        if d:
            pairs.append((Path(d).expanduser(), adapter))
    return pairs


def _is_representation_change(path: Path) -> bool:
    """True when `path` vanished only because the same transcript now exists in
    the other representation.

    Codex zstd-compresses cold rollouts in place (rollout.jsonl ->
    rollout.jsonl.zst) and materializes them back to append. Both show up as a
    delete of a real transcript path; archiving on them hid live sessions from
    the browser about a week after they were written.
    """
    name = str(path)
    twin = name[:-4] if name.endswith(".zst") else name + ".zst"
    return Path(twin).exists()


class _Handler(FileSystemEventHandler):
    def __init__(self, adapter):
        self.adapter = adapter
        self._timers: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()

    def _schedule(self, path: str):
        with self._lock:
            t = self._timers.get(path)
            if t:
                t.cancel()
            timer = threading.Timer(DEBOUNCE_S, self._process, args=(path,))
            self._timers[path] = timer
            timer.start()

    def _process(self, path_str: str):
        with self._lock:
            self._timers.pop(path_str, None)  # fired — drop the dead Timer
        path = Path(path_str)
        if not path.exists():
            return
        # `cr` links sessions into other project dirs as resume conduits; the
        # canonical transcript is the real file (same invariant as discover()).
        if path.is_symlink():
            return
        # Let the adapter decide whether this path is a session transcript at all
        # (e.g. only <sid>/events.jsonl counts for copilot, not future siblings).
        if self.adapter.session_id_for_path(path) is None:
            # Not a session — but maybe the SOURCE changed (a write to OpenCode's
            # DB/WAL): let the adapter re-project, and the resulting mirror
            # events flow through this same handler.
            trigger = getattr(self.adapter, "sync_trigger", None)
            if callable(trigger) and trigger(path):
                self._schedule_sync()
            return
        try:
            header = self.adapter.parse_header(path)
            if header is None:
                return
            # Any hook (Claude Stop hook, OpenCode plugin) that just indexed this
            # session wins; ids never collide across CLIs.
            if hookstate.recently(header.session_id):
                _log(f"skip (hook race-guard) {header.session_id}")
                return
            indexer.upsert(header)
            _log(f"index [{self.adapter.name}] {header.session_id} ({header.turn_count} turns)")
        except Exception as e:  # noqa: BLE001
            _log(f"error {path.name}: {e}")

    def _schedule_sync(self):
        with self._lock:
            t = self._timers.get(_SYNC_KEY)
            if t:
                t.cancel()
            timer = threading.Timer(SYNC_DEBOUNCE_S, self._run_sync)
            self._timers[_SYNC_KEY] = timer
            timer.start()

    def _run_sync(self):
        with self._lock:
            self._timers.pop(_SYNC_KEY, None)
        try:
            report = self.adapter.sync()
            if report.written or report.removed or report.warnings:
                _log(f"sync [{self.adapter.name}] written={len(report.written)} "
                     f"removed={len(report.removed)} warnings={len(report.warnings)}")
        except Exception as e:  # noqa: BLE001
            _log(f"sync error [{self.adapter.name}]: {e}")

    def on_created(self, event):
        if not event.is_directory:
            self._schedule(event.src_path)

    def on_modified(self, event):
        if not event.is_directory:
            self._schedule(event.src_path)

    def on_moved(self, event):
        # A rename is a delete at src + a create at dest (Finder's "Move to
        # Trash" is a rename out of the tree) — treat it as exactly that.
        if event.is_directory:
            return
        self.on_deleted(event)
        dest = getattr(event, "dest_path", "")
        if dest and self.adapter.session_id_for_path(Path(dest)) is not None:
            self._schedule(dest)

    def on_deleted(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        sid = self.adapter.session_id_for_path(path)
        if sid is None:
            return
        # An atomic rewrite (tmp + os.replace — how the OpenCode mirror is
        # written) surfaces on macOS as a delete of the overwritten inode. A
        # path that still exists was replaced, not deleted; the create/modify
        # event that follows re-indexes it.
        if path.exists():
            _log(f"skip archive (replaced in place) {sid}")
            return
        if _is_representation_change(path):
            _log(f"skip archive (compression/materialization) {sid}")
            return
        try:
            # Archive only when the deleted path IS the canonical transcript. A
            # deleted `cr` symlink shares the session's filename but lives in a
            # different project dir — archiving on it would hide a live session.
            conn = indexer.connect()
            try:
                row = conn.execute(
                    "SELECT project_path FROM sessions WHERE session_id = ?", (sid,)
                ).fetchone()
                canonical_dir = row["project_path"] if row else None
            finally:
                conn.close()
            if canonical_dir is not None and Path(canonical_dir) != path.parent:
                _log(f"skip archive (non-canonical copy deleted) {sid}")
                return
            # Proven above: the path that vanished IS the canonical transcript,
            # so this is a real session aged out — keep it visible as such.
            indexer.archive(sid, indexer.TRANSCRIPT_MISSING)
            _log(f"archive [{self.adapter.name}] {sid} ({indexer.TRANSCRIPT_MISSING})")
        except Exception as e:  # noqa: BLE001
            _log(f"archive error {sid}: {e}")


def _acquire_singleton_lock():
    """Ensure only one watcher runs, no matter how it was started (launchd, shell,
    manual). Returns the held lock file handle, or None if another instance owns it."""
    import fcntl
    lock_path = sbconfig.LOG_DIR.parent / ".watcher.lock"
    # "a+" not "w": opening must not truncate the pid a live holder recorded
    fh = open(lock_path, "a+")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fh.seek(0)
        fh.truncate()
        fh.write(str(os.getpid()))
        fh.flush()
        return fh
    except OSError:
        return None


def main() -> None:
    sbconfig.ensure_dirs()
    lock = _acquire_singleton_lock()
    if lock is None:
        _log("another watcher instance is already running; exiting")
        return
    pairs = _build_watch_pairs()
    if not pairs:
        _log("no available sources to watch; exiting")
        return
    observer = Observer()
    for directory, adapter in pairs:
        if directory.exists():
            observer.schedule(_Handler(adapter), str(directory), recursive=True)
            _log(f"watching [{adapter.name}] {directory}")
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
