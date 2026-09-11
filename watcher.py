#!/usr/bin/env python3
"""Filesystem watcher daemon — covers every enabled source.

watchdog Observer over each source directory with a 500ms per-path debounce.
On create/modify -> resolve owning adapter -> parse_header -> indexer.upsert.
On delete -> indexer.archive. For Claude files it consults .hook-state.json and
skips any session the Stop hook indexed within the last 30s (race-guard), so the
two indexing paths never double-process the same file.

--- What this process is --------------------------------------------------------------

A *watcher* is a program that sits idle and is woken by the operating system whenever a
file it cares about changes. This one is started at login by launchd (macOS) or a systemd
--user unit (Linux), runs forever, and is the safety net of the whole pipeline: hooks
(small programs a CLI runs when a session ends) index instantly but only exist for Claude
Code and OpenCode, and only when their configuration is intact. Everything else — Copilot,
Codex, a session ended by closing the terminal, a transcript deleted last night — reaches
the registry through here. Terms: see docs/GLOSSARY.md.

Its whole job is three lines of logic per event:

    file created/changed  ->  adapter.parse_header(path)  ->  indexer.upsert(header)
    file deleted          ->  adapter.session_id_for_path(path) -> indexer.archive(...)
    database write        ->  adapter.sync()  (a source with no per-session file)

Everything else in this module is defence against the ways that goes wrong in practice.

--- The five hazards this file exists to handle ----------------------------------------

1. Bursts. An active transcript is appended to constantly, and each append is an event.
   Every path gets a 0.5 s debounce timer (SYNC_DEBOUNCE_S = 2 s for database writes, which
   are far noisier), so a session being typed into costs two parses a second, not fifty.

2. The hook race. A hook and this daemon both see the same session end. The hook records
   the id in hookstate.py and this process skips anything claimed in the last 30 seconds.

3. Directories that do not exist yet. A laptop can install Session Browser before it ever
   runs Codex. Roots are therefore subscribed lazily by _RootScheduler, which retries
   every ROOT_POLL_S seconds forever, and the process stays alive with zero roots.

4. Deletes that are not deletes. Three separate cases, all handled in on_deleted():
   an atomic rewrite (write to a temp file, rename over the target) surfaces on macOS as
   a delete; Codex compresses cold transcripts in place, which is a delete of the
   uncompressed name; and a symlink to a transcript can be deleted while the transcript
   itself is fine. Archiving on any of these hides a live session from the user.

5. Watch-limit exhaustion. Linux allows a fixed number of watched directories per user
   (fs.inotify.max_user_watches). Exceeding it raises ENOSPC, which used to kill the
   daemon on start-up, with systemd respawning it straight back into the same crash.

--- Lifecycle ---------------------------------------------------------------------------

main() takes a singleton lock (so a manual run cannot double up with the launchd one),
builds the (directory, adapter) pairs, starts the observer, and then loops once a second
doing nothing except re-polling for roots that have not appeared yet. The real work
happens on watchdog's own threads, inside _Handler.

Everything is logged to LOG (~/.session-browser/logs/watcher.log) because a background
daemon has no terminal; that file is the only way to see what it has been doing.
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

# This file is launched by launchd/systemd as a bare script, not as part of a package, so
# the repo is not on sys.path yet and `import indexer` below would fail. Prepending it
# here — before those imports, which is why they carry noqa: E402 — makes the daemon
# start correctly no matter what directory it was started from.
_REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO))

import hookstate  # noqa: E402
import indexer  # noqa: E402
import sbconfig  # noqa: E402
from sources.registry import build_source_registry  # noqa: E402

# How long a path must stay quiet before it is parsed. Long enough to collapse the
# append-per-keystroke burst of a live session, short enough that a finished session is
# in the registry before the user can switch to the browser tab.
DEBOUNCE_S = 0.5
# A DB-backed source (OpenCode) reports a change trigger per WAL write, which
# is many times a second while a session is active; one re-sync per burst.
SYNC_DEBOUNCE_S = 2.0
# Re-projecting a whole database is not per-path work, so its pending timer is filed
# under this reserved key in the same timer dict. It can never collide with a real path.
_SYNC_KEY = "__sync__"
LOG = sbconfig.LOG_DIR / "watcher.log"


# Rotate at 5 MB. A daemon that logs a line per indexed session would otherwise fill the
# disk over a laptop's lifetime; one previous file (.log.1) is kept.
_LOG_MAX_BYTES = 5 * 1024 * 1024


def _log(msg: str) -> None:
    """Append one timestamped line to the watcher log, and echo it to stdout.

    Both destinations matter: launchd/systemd capture stdout, and the file is what
    `sb doctor` and a curious user read. Timestamps are UTC so they line up with the
    registry's own columns.

    Every filesystem error is swallowed — a daemon that cannot write its log must keep
    indexing, and the print() below still reaches the service manager's own log.
    """
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


# How often the watcher re-checks for source directories that did not exist
# when it started (a CLI installed or first run after us). Also the retry interval after
# a failed subscribe (see _RootScheduler.poll). Cheap: one is_dir() per pending root.
ROOT_POLL_S = 30


def _build_watch_pairs() -> list[tuple[Path, object]]:
    """(directory, adapter) pairs for every ENABLED source — existing or not.

    An adapter may own more than one root — codex spreads live and archived
    rollouts across two sibling trees — so it gets to declare them via an
    optional watch_roots(). Sources without one keep the single configured dir.
    Existence is not checked here: _RootScheduler subscribes each root the
    moment it appears, so a CLI first run after the watcher started is covered.

    What comes back, concretely: three-element tuples (directory, adapter, recursive).
    The annotation says pairs because that is how the rest of the file reads them — the
    third element is a flag _RootScheduler picks off, not part of the identity. An
    adapter's watch_roots() may yield either a bare path (meaning: watch it recursively)
    or a (path, recursive) tuple; `recursive=False` is how a source says "only the files
    directly in here matter", which for OpenCode's data directory avoids hundreds of
    pointless kernel watches on log/, tool-output/, snapshot/ and project/.

    Called once at start-up by main(). Pure: no directory is created, opened or watched.
    """
    pairs = []
    # Note: build_source_registry() without only_available — a directory that does not
    # exist yet is exactly the case _RootScheduler is for, so filtering here would
    # permanently blind the watcher to a CLI the user installs later.
    for name, adapter in build_source_registry().items():
        roots = getattr(adapter, "watch_roots", None)
        if callable(roots):
            for r in roots():
                # Accept both spellings an adapter may use: a bare path, or an explicit
                # (path, recursive) pair.
                d, rec = (r[0], bool(r[1])) if isinstance(r, tuple) else (r, True)
                pairs.append((Path(d).expanduser(), adapter, rec))
            continue
        # The adapter — not the raw config key — is the source of truth: it has
        # already applied CLAUDE_CONFIG_DIR / CODEX_HOME / XDG_DATA_HOME via
        # registry._cli_home(), which a config re-read here silently bypassed.
        # Each adapter names its root directory differently; take whichever it has.
        d = (getattr(adapter, "projects_dir", None) or getattr(adapter, "state_dir", None)
             or getattr(adapter, "sessions_dir", None))
        if d:
            pairs.append((Path(d).expanduser(), adapter, True))
    return pairs


def _is_representation_change(path: Path) -> bool:
    """True when `path` vanished only because the same transcript now exists in
    the other representation.

    Codex zstd-compresses cold rollouts in place (rollout.jsonl ->
    rollout.jsonl.zst) and materializes them back to append. Both show up as a
    delete of a real transcript path; archiving on them hid live sessions from
    the browser about a week after they were written.

    The test is simply "does the other spelling of this name exist right now?":

        /…/rollout-2026-08-01-abc.jsonl      -> look for …abc.jsonl.zst
        /…/rollout-2026-08-01-abc.jsonl.zst  -> look for …abc.jsonl

    (zstd is a compression format; ".zst" is its four-character suffix, hence `[:-4]`.)
    Returns True — meaning "do not archive" — when the twin is there. Called only from
    the delete path, after the plain "was it replaced in place?" check.
    """
    name = str(path)
    twin = name[:-4] if name.endswith(".zst") else name + ".zst"
    return Path(twin).exists()


class _RootScheduler:
    """Subscribes each (root, adapter) pair once its directory exists and keeps
    retrying the rest. Missing roots used to be skipped for the life of the
    process, and a watcher that found none exited 0 — which launchd (KeepAlive
    on failure only) never restarts, so a laptop that installed Session Browser
    before its first CLI session watched nothing until the next login.

    State: `pending` holds the (directory, adapter) pairs not yet subscribed, `scheduled`
    those that are. poll() moves entries from one to the other and is safe to call
    repeatedly — main() calls it every ROOT_POLL_S seconds for as long as anything is
    pending, which may be forever on a machine that never installs a second CLI.

    Constructed by start_watching(); the tests drive it directly with a fake observer.
    """

    def __init__(self, observer, pairs, log=None):
        """`observer` is the (already started) watchdog Observer; `pairs` are the
        (directory, adapter[, recursive]) entries from _build_watch_pairs(); `log` is
        injectable so the tests capture lines instead of writing to the real log file."""
        self.observer = observer
        # (dir, adapter[, recursive]); recursive defaults to True. Kept as pairs
        # for callers; the flag lives alongside.
        self.pending = [(p[0], p[1]) for p in pairs]
        self.recursive = {(p[0], p[1]): (bool(p[2]) if len(p) > 2 else True) for p in pairs}
        self.scheduled: list[tuple[Path, object]] = []
        self._log = log or _log

    def poll(self) -> list[Path]:
        """Try to subscribe every still-pending root; return the ones newly subscribed.

        Idempotent: a root is removed from `pending` only after the observer accepted it,
        and a root that is still missing or still refused simply stays pending for the
        next call. Side effects: registers watches on the observer and writes log lines.
        Never raises — a failure here must not take the daemon down (hazard 5 in the
        module docstring).
        """
        # Local import: only needed to recognise one errno, and only on the failure path.
        import errno
        newly: list[Path] = []
        # Iterate over a copy — the loop removes from self.pending as it succeeds.
        for directory, adapter in list(self.pending):
            if not directory.is_dir():
                continue        # not created yet; try again in ROOT_POLL_S seconds
            try:
                self.observer.schedule(_Handler(adapter), str(directory),
                                       recursive=self.recursive.get((directory, adapter), True))
            except OSError as e:
                # Linux: one inotify watch per directory; past
                # fs.inotify.max_user_watches watchdog raises ENOSPC. Keep the
                # root pending and retry rather than take the daemon down.
                hint = (" — raise the limit: sudo sysctl fs.inotify.max_user_watches=524288"
                        if e.errno == errno.ENOSPC else "")
                self._log(f"cannot watch [{adapter.name}] {directory}: {e}{hint}; retrying in {ROOT_POLL_S}s")
                continue
            self.pending.remove((directory, adapter))
            self.scheduled.append((directory, adapter))
            newly.append(directory)
            self._log(f"watching [{adapter.name}] {directory}")
        return newly


def start_watching(observer, pairs, log=None) -> "_RootScheduler":
    """Start the observer FIRST, then subscribe roots. watchdog defers each
    emitter's start until Observer.start(); a root scheduled before that
    raised its inotify ENOSPC out of start() — outside the scheduler's guard —
    and the daemon died (systemd respawned it straight into the same crash).
    Started first, every failure lands in poll(), is logged with the sysctl
    hint, and is retried.

    Returns the live _RootScheduler so main() (or a test) can keep polling it. Side
    effects: starts the observer's threads and subscribes whatever roots already exist.
    Note it deliberately does NOT fail when nothing could be subscribed — it logs and
    returns, and the daemon keeps waiting.
    """
    log = log or _log
    # Order matters, and this line is the fix: start(), THEN schedule().
    observer.start()
    roots = _RootScheduler(observer, pairs, log=log)
    roots.poll()
    if not roots.scheduled:
        log(f"no source directory exists yet ({len(roots.pending)} pending) — "
            f"waiting for the first one to appear, re-checking every {ROOT_POLL_S}s")
    return roots


class _Handler(FileSystemEventHandler):
    """Receives filesystem events for ONE adapter's roots and turns them into DB writes.

    watchdog calls the on_* methods below on its own threads, so this class is shared
    mutable state across threads and every touch of `_timers` is under `_lock`. One
    handler instance is created per subscribed root (see _RootScheduler.poll), which is
    why it carries its adapter with it — the adapter is what knows whether a given path
    is a session at all.

    The debounce works by *rescheduling*: each event for a path cancels that path's
    pending timer and starts a new one, so the parse happens once, DEBOUNCE_S after the
    last write, rather than once per write.
    """

    def __init__(self, adapter):
        """`adapter` owns the directory this handler was subscribed to; every decision
        about a path is delegated to it."""
        self.adapter = adapter
        # path (or _SYNC_KEY) -> the pending threading.Timer for it.
        self._timers: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()

    def _schedule(self, path: str):
        """(Re)start this path's debounce timer: _process(path) runs DEBOUNCE_S from now.

        Cancelling the previous timer is the debounce — a file being appended to
        constantly keeps pushing its own parse into the future until the writing stops.
        """
        with self._lock:
            t = self._timers.get(path)
            if t:
                t.cancel()
            timer = threading.Timer(DEBOUNCE_S, self._process, args=(path,))
            self._timers[path] = timer
            timer.start()

    def _process(self, path_str: str):
        """Parse one settled path and upsert it. Runs on a Timer thread.

        The main path of the whole daemon, and the one place a registry row is written
        from a file change. Catches everything: an exception here would kill a Timer
        thread silently and the daemon would look alive while quietly indexing nothing,
        so every failure is logged and the process carries on.
        """
        with self._lock:
            self._timers.pop(path_str, None)  # fired — drop the dead Timer
        path = Path(path_str)
        if not path.exists():
            return        # created and removed again inside the debounce window
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
        """Same debounce, but for "the source database changed" rather than one path.

        Filed under the reserved _SYNC_KEY in the same timer dict, with the longer
        SYNC_DEBOUNCE_S: a database in write-ahead-log mode reports a change on every
        flush, many times a second, and re-projecting is much more expensive than
        parsing one file.
        """
        with self._lock:
            t = self._timers.get(_SYNC_KEY)
            if t:
                t.cancel()
            timer = threading.Timer(SYNC_DEBOUNCE_S, self._run_sync)
            self._timers[_SYNC_KEY] = timer
            timer.start()

    def _run_sync(self):
        """Ask the adapter to re-project its database into per-session files.

        Writes and deletes mirror files, each of which comes back as an ordinary create /
        modify / delete event through this same handler and is indexed by the normal
        path — so this method itself never touches the registry. Logs only when something
        actually changed, to keep an idle machine's log quiet. Errors are logged and
        swallowed: a locked or mid-migration source database must not kill the daemon.
        """
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
        """A new file appeared — a session starting. Directories are ignored: a new
        project directory holds no transcript yet, and its files arrive as their own
        events."""
        if not event.is_directory:
            self._schedule(event.src_path)

    def on_modified(self, event):
        """A file was written to — the common case, fired on every append to a live
        transcript. Debounced like everything else."""
        if not event.is_directory:
            self._schedule(event.src_path)

    def on_moved(self, event):
        """A rename. Handled as its two halves rather than as an event of its own."""
        # A rename is a delete at src + a create at dest (Finder's "Move to
        # Trash" is a rename out of the tree) — treat it as exactly that.
        if event.is_directory:
            return
        self.on_deleted(event)
        dest = getattr(event, "dest_path", "")
        # Only index the destination if it is still a transcript of ours: renaming a
        # transcript to notes.txt is a deletion as far as this tool is concerned.
        if dest and self.adapter.session_id_for_path(Path(dest)) is not None:
            self._schedule(dest)

    def on_deleted(self, event):
        """A file vanished — possibly. Archive the session only once it is certain.

        This is hazard 4 from the module docstring, and the most defensive code in the
        file, because a wrong archive silently hides a session the user still has. Four
        things are ruled out before indexer.archive() is called:

          1. the path is not one of this source's transcripts at all;
          2. the path still exists — so it was replaced in place, not deleted;
          3. its compressed/uncompressed twin exists — a representation change;
          4. the deleted file was not the row's canonical transcript, i.e. it was a
             symlinked copy in another project directory.

        Only then is the row archived, with the reason worked out from the row's own
        content (see indexer.infer_archive_reason). Runs directly on watchdog's thread —
        no debounce, because a deletion does not repeat — and logs every decision,
        including the skips, since "why is this session missing / still here?" is
        answered from that log.
        """
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
                # The whole row, not just project_path: infer_archive_reason() below
                # needs its turn count, first message, tokens and summary to tell a real
                # session from subagent noise, and one read is cheaper than two.
                row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (sid,)).fetchone()
                canonical_dir = row["project_path"] if row else None
            finally:
                # Closed before archive() opens its own connection: holding a second
                # connection open invites a write-lock wait for no reason.
                conn.close()
            if canonical_dir is not None and Path(canonical_dir) != path.parent:
                _log(f"skip archive (non-canonical copy deleted) {sid}")
                return
            # Proven above: the path that vanished IS the canonical transcript.
            # The same rule prune-sessions applies decides whether it was a
            # real session (kept visible, restorable) or sidechain noise — the
            # two paths used to disagree, so WHICH process saw the deletion
            # decided whether a session stayed browsable.
            # No row at all (a transcript this watcher never indexed) is treated as a
            # real session that has gone: archive() on an unknown id is a harmless no-op.
            reason = indexer.infer_archive_reason(row) if row is not None else indexer.TRANSCRIPT_MISSING
            indexer.archive(sid, reason)
            _log(f"archive [{self.adapter.name}] {sid} ({reason})")
        except Exception as e:  # noqa: BLE001
            _log(f"archive error {sid}: {e}")


def _acquire_singleton_lock():
    """Ensure only one watcher runs, no matter how it was started (launchd, shell,
    manual). Returns the held lock file handle, or None if another instance owns it.

    Uses an advisory whole-file lock (flock) rather than a pid file, because the kernel
    releases it automatically when the process dies — including a kill -9 or a crash — so
    there is no stale lock to clean up after a reboot. The returned handle MUST stay
    referenced for the life of the process: closing it, or letting it be garbage
    collected, releases the lock. The pid is written purely so a human reading the file
    can see who holds it.
    """
    # Local import: this is POSIX-only and is called exactly once.
    import fcntl
    lock_path = sbconfig.LOG_DIR.parent / ".watcher.lock"
    # "a+" not "w": opening must not truncate the pid a live holder recorded
    fh = open(lock_path, "a+")
    try:
        # LOCK_NB = do not block: if someone else holds it we want to exit immediately,
        # not queue up behind them.
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Only now, having won, replace the previous holder's pid with ours.
        fh.seek(0)
        fh.truncate()
        fh.write(str(os.getpid()))
        fh.flush()
        return fh
    except OSError:
        return None


def main() -> None:
    """Run the daemon: take the lock, subscribe the roots, then idle forever.

    Two early exits, both clean (exit code 0): another instance already holds the lock,
    and no sources enabled in config. Note the contrast with "no source DIRECTORY exists
    yet", which is deliberately NOT an exit — see _RootScheduler.

    After start_watching() all real work happens on watchdog's threads; this loop only
    wakes once a second to re-poll for roots that have not appeared yet. `tick %
    ROOT_POLL_S` is what turns a 1-second loop into a 30-second retry while keeping the
    process responsive to Ctrl-C.
    """
    sbconfig.ensure_dirs()
    lock = _acquire_singleton_lock()
    if lock is None:
        _log("another watcher instance is already running; exiting")
        return
    pairs = _build_watch_pairs()
    if not pairs:
        _log("no sources enabled in config; exiting")
        return
    observer = Observer()
    roots = start_watching(observer, pairs)
    try:
        tick = 0
        while True:
            time.sleep(1)
            tick += 1
            if roots.pending and tick % ROOT_POLL_S == 0:
                roots.poll()
    except KeyboardInterrupt:
        # Ctrl-C from a manual run, or the service manager's stop signal: ask the
        # observer to wind down, then wait for its threads below.
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
