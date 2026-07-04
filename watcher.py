#!/usr/bin/env python3
"""Filesystem watcher daemon — covers every enabled source.

watchdog Observer over each source directory with a 500ms per-path debounce.
On create/modify -> resolve owning adapter -> parse_header -> indexer.upsert.
On delete -> indexer.archive. For Claude files it consults .hook-state.json and
skips any session the Stop hook indexed within the last 30s (race-guard), so the
two indexing paths never double-process the same file.
"""
from __future__ import annotations

import json
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

import indexer  # noqa: E402
import sbconfig  # noqa: E402
from sources.registry import build_source_registry  # noqa: E402

DEBOUNCE_S = 0.5
RACE_GUARD_S = 30
LOG = sbconfig.LOG_DIR / "watcher.log"


def _log(msg: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()}  {msg}\n"
    try:
        with open(LOG, "a") as fh:
            fh.write(line)
    except OSError:
        pass
    print(line, end="")


def _build_watch_pairs() -> list[tuple[Path, object]]:
    """(directory, adapter) pairs for every available source."""
    pairs = []
    for name, adapter in build_source_registry(only_available=True).items():
        cfg = sbconfig.source_config(name)
        d = cfg.get("projects_dir") or cfg.get("state_dir") or cfg.get("sessions_dir")
        if d:
            pairs.append((Path(d).expanduser(), adapter))
    return pairs


def _recently_hooked(session_id: str) -> bool:
    try:
        state = json.loads(sbconfig.HOOK_STATE.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    ts = state.get(session_id)
    if not ts:
        return False
    try:
        when = datetime.fromisoformat(ts)
        return (datetime.now(timezone.utc) - when).total_seconds() < RACE_GUARD_S
    except ValueError:
        return False


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
        path = Path(path_str)
        if not path.exists() or path.suffix != ".jsonl":
            return
        try:
            header = self.adapter.parse_header(path)
            if header is None:
                return
            if self.adapter.name == "claude" and _recently_hooked(header.session_id):
                _log(f"skip (hook race-guard) {header.session_id}")
                return
            indexer.upsert(header)
            _log(f"index [{self.adapter.name}] {header.session_id} ({header.turn_count} turns)")
        except Exception as e:  # noqa: BLE001
            _log(f"error {path.name}: {e}")

    def on_created(self, event):
        if not event.is_directory:
            self._schedule(event.src_path)

    def on_modified(self, event):
        if not event.is_directory:
            self._schedule(event.src_path)

    def on_deleted(self, event):
        if event.is_directory or not str(event.src_path).endswith(".jsonl"):
            return
        sid = Path(event.src_path).stem
        try:
            indexer.archive(sid)
            _log(f"archive [{self.adapter.name}] {sid}")
        except Exception as e:  # noqa: BLE001
            _log(f"archive error {sid}: {e}")


def _acquire_singleton_lock():
    """Ensure only one watcher runs, no matter how it was started (launchd, shell,
    manual). Returns the held lock file handle, or None if another instance owns it."""
    import fcntl
    lock_path = sbconfig.LOG_DIR.parent / ".watcher.lock"
    fh = open(lock_path, "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
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
