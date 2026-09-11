"""OpenCode source adapter — a SQLite database projected to one JSONL per session.

In plain words: OpenCode is a command-line AI coding assistant, and unlike the others it
does not leave a log file per conversation — it writes everything into one database. This
module reads that database and WRITES OUT one plain text file per conversation, because
every other part of Session Browser was built around "one file per session". That
rewriting step is called the *mirror*, and it is the only adapter that produces files
rather than just reading them.

Vocabulary, defined once here; docs/GLOSSARY.md has the rest. A **session** is one
conversation with the CLI. A **transcript** is the record of it — for OpenCode, rows in
the database, and after mirroring, one JSONL file. A **turn** is one user message plus
the assistant reply it drew. A **token** is the word-piece unit models read and write in,
and what vendors bill by; OpenCode records both token counts and its own dollar figure
per message. A **provider/model** pair names who served the AI and which one
("anthropic/claude-sonnet-5"). A **sub-agent** is a helper conversation the assistant
starts for a sub-task; OpenCode stores it as a CHILD session, and this adapter folds
children into their parent so one conversation stays one row.

Where it sits: sources/registry.py builds one OpenCodeSource; discover() is the entry
point every caller uses, and it quietly re-syncs the mirror first, so the indexer, the
watcher, backfill and `sb doctor` all see fresh files without knowing a database exists.
The OpenCode plugin (plugins/opencode/ + scripts/opencode-hook.py) calls sync() for one
session the moment it ends. docs/ARCHITECTURE.md, "DB-backed sources: the mirror
pattern", has the design rationale.

Since v1.2.0 OpenCode keeps every session in ONE SQLite DB (WAL mode):
  ~/.local/share/opencode/opencode.db     ($XDG_DATA_HOME/opencode; $OPENCODE_DB
                                           overrides; channel builds: opencode-<ch>.db)
  session(id, project_id, parent_id, slug, directory, path, title, version, ...
          time_created/time_updated — epoch ms)
  message(id, session_id, time_created, data JSON)      role, providerID/modelID,
                                                        cost, tokens, finish, error
  part(id, message_id, session_id, time_created, data JSON)   text / reasoning /
                                                        tool / step-* / patch / ...
The JSON `data` omits the promoted columns (id, sessionID, messageID), so the projection
puts them back — see _project_root. A *part* is one piece of a message: the text the
assistant wrote, its reasoning, a tool call and its output, a file it touched.

Every other part of this repo assumes ONE plain-text file per session at the
path discover() yields: the watcher's delete handler, reasoning.archive_raw,
restore.py, build-fts, the cost and reasoning extractors. So this adapter
PROJECTS the DB into <mirror_dir>/<ses_id>.jsonl — one file per ROOT session,
sub-agent children embedded, their tokens/cost rolled up — and everything
downstream works unchanged. The mirror doubles as the backup: OpenCode never
auto-deletes, but `opencode session delete` hard-cascades, spilled tool
outputs are purged after 7 days, and the DB grows fast enough that people prune it.

    L1  {"type":"session","schema":1,"info":{...Session.Info...},
         "children":[...],"stats":{...},"source":{...}}
    L2+ {"type":"message","session":"ses_…","info":{...},"parts":[...]}

Shortened but real, from a two-turn session with one sub-agent child (the session ids are
abbreviated to ROOT/CHILD here; on disk each of these is a single line):

    L1  {"type":"session","schema":1,
         "info":{"id":"ses_ROOT","slug":"kind-canyon","projectID":"proj-hash",
                 "directory":"/Users/x/proj","version":"1.18.15",
                 "title":"New session - 2026-08-01T10:00:00.000Z",
                 "time":{"created":1785542400000,"updated":1785542460000}},
         "children":[{"id":"ses_CHILD","parentID":"ses_ROOT", …}],
         "stats":{"turn_count":2,"first_message":"why is opencode missing?",
                  "title":null,"model_used":"opencode/minimax-m2.5-free",
                  "models":{"opencode/minimax-m2.5-free":{"input":1500,"output":300,
                            "reasoning":100,"cache_read":50,"cache_write":10,
                            "cost":0.0185}},
                  "cost_usd":0.0185,"start_time":1785542400000,
                  "last_activity":1785542460000,"message_count":8,"child_count":1,
                  "recognised_parts":13},
         "source":{"db":"/Users/x/.local/share/opencode/opencode.db",
                   "migration":"20260622202450_simplify_session_input",
                   "mirrored_at":1785542461000}}
    L2  {"type":"message","session":"ses_ROOT",
         "info":{"id":"msg_u1","sessionID":"ses_ROOT","role":"user", …},
         "parts":[{"id":"prt_01","messageID":"msg_u1","sessionID":"ses_ROOT",
                   "type":"text","text":"why is opencode missing?"}]}

Note `info.title` versus `stats.title` in that example: OpenCode named the session before
it knew anything about it, so the projection reports no title at all and the browser falls
back to the first message (see _PLACEHOLDER_TITLE). The 8 messages and 13 recognised parts
span the root AND its child, which is also why `models` totals more than the root spent.

Line 1 is the whole point of the layout: parse_header reads ONE line and already has the
turn count, the first message, the title, the model and the rolled-up spend, so listing
hundreds of sessions never opens a multi-megabyte file. Times on line 1 are OpenCode's
raw epoch milliseconds; to_iso_utc converts them when a header is built. `source` records
provenance — which database this file came from — which is what keeps a mirror belonging
to another OpenCode installation from being deleted here (see _remove_deleted).

Which files get rewritten is decided by a fingerprint manifest, <mirror>/.manifest.json:
{"schema": 1, "roots": {"ses_…": [newest activity ms, message count]}}. A root whose pair
still matches is skipped untouched, so a sync over hundreds of sessions writes only the
handful that changed. Losing or corrupting the manifest costs exactly one full resync.

Never invoke the `opencode` binary on this path: `opencode export` truncates at
64 KiB on a pipe, merely running `opencode db path` opened the DB read-write
and checkpointed the WAL, and a newer binary would run migrations. The DB is
opened read-only (`mode=ro`, the WAL must be readable) and the path comes from
env/config only. Ids are NOT chronologically sortable (36-bit truncated clock)
— always order by time_created. The single exception is restore, which the user asks
for by hand: reimport() runs `opencode import` to put a session back (see _run_import).

Concurrency: several processes project the same mirror — the watcher, the OpenCode
plugin's hook, the nightly refresh — so sync() takes an exclusive lock on
<mirror>/.sync.lock and every file is written to a per-writer temp name and renamed into
place. A session deleted inside OpenCode is archived to the raw vault BEFORE its mirror
file is unlinked, because by then that file is the last copy in existence; the unlink is
what the watcher turns into an archived row the user can restore.
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import shlex
import shutil
import sqlite3
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from .base import ParsedSession, SessionHeader, Turn, to_iso_utc

# Sidecar left beside a restored mirror file until OpenCode holds the session again.
# The file <mirror>/ses_….jsonl.restored means "a human deliberately put this back, and
# OpenCode does not know about it yet" — without it the next sync would see a mirror file
# with no matching database row and archive it straight back out. See protect()/reimport().
_RESTORED = ".restored"


def _now_iso() -> str:
    """Now, as a UTC ISO-8601 string. Only used as human-readable content for the
    .restored marker, so the exact spelling does not matter. Imported lazily because
    every adapter module is imported on start-up and this is a cold path."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


# Defaults only — sources/registry.py resolves config.toml and the environment and passes
# the real paths to __init__. $XDG_DATA_HOME is the standard "where apps keep their data"
# variable; OpenCode honours it, so this adapter must too, or a customised setup points
# at an empty directory.
_XDG = os.environ.get("XDG_DATA_HOME")
DATA_DIR = (Path(_XDG) if _XDG else Path.home() / ".local" / "share") / "opencode"
# Our own directory, not OpenCode's: one JSONL per root session plus the manifest and
# lock. It is also the backup, which is why it lives under ~/.session-browser and not in
# a temp directory.
MIRROR_DIR = Path(os.path.expanduser("~/.session-browser/opencode-mirror"))

# Version stamped on line 1 of every mirror file and in the manifest. Bump it if the
# projection's shape changes in a way older readers cannot cope with.
SCHEMA = 1
# A valid OpenCode session id: "ses_" followed by exactly 26 letters/digits, e.g.
# "ses_fd7037a16ffeRyoMOVVyFqv3xY". Anchored at both ends, so it matches the WHOLE stem
# and nothing else — that is what keeps ".manifest.json", "opencode.db-wal", a
# half-written ".tmp" and an archive copy's "ses_…@v2" out of session_id_for_path().
_SID = re.compile(r"^ses_[A-Za-z0-9]{26}$")
# OpenCode names a session before it knows what it is about: "New session - 2026-08-01T…"
# or "Child session - 2026-08-01T…". Matching that prefix plus a date means such a title
# is treated as no title at all, so the UI falls back to the first message instead of
# showing every session the same useless name. `\d{4}-\d{2}-\d{2}T` is the date-and-T of
# an ISO timestamp; the rest of it is not worth matching.
_PLACEHOLDER_TITLE = re.compile(r"^(New session|Child session) - \d{4}-\d{2}-\d{2}T")
# Finds a spilled tool-output path inside preview text, for older OpenCode builds that
# mention the file but do not record it in metadata. Matches a run of non-space
# characters ending in "tool-output/tool_<alphanumerics>", e.g. the
# "/Users/x/.local/share/opencode/tool-output/tool_00c2dd9e40012bvNNrwXYZ" inside
# "...output truncated...\n\nFull output saved to: <path>". `\S*` grabs the leading
# directories; the path itself never contains a space.
_SPILL_RE = re.compile(r"\S*tool-output/tool_[A-Za-z0-9]+")
# Part types this adapter understands — only to tell "no user turns" apart
# from "a schema I have never seen" (the codex lesson: never index nothing silently).
#
# NOT a filter: unknown part types are copied into the mirror verbatim (the mirror is
# the lossless backup) and simply are not counted here. What the count feeds is one
# decision in parse_header: a session with messages but zero recognised parts is in a
# schema this adapter does not understand, so it is indexed with 0 turns AND a warning
# instead of vanishing. Meanings: text = what was written; reasoning = the model's own
# thinking; tool = a command or edit it ran, with the result; step-start/step-finish =
# request boundaries carrying token counts; patch/file = code changes; subtask/agent =
# a sub-agent being spawned; retry = a failed request repeated; compaction = OpenCode
# summarising older history to fit the context window; snapshot = a saved workspace state.
_KNOWN_PARTS = frozenset({
    "text", "reasoning", "tool", "step-start", "step-finish", "patch", "file",
    "subtask", "agent", "retry", "compaction", "snapshot",
})
# The minimum shape the projection needs from each table. _schema_check compares this
# against the live database and WARNS about anything missing rather than raising: a
# missing column is projected around (the field is simply absent), while a missing table
# means nothing can be read at all and the sync stops so the existing mirror keeps
# serving. Columns beyond these are used opportunistically via _get().
_REQUIRED = {
    "session": {"id", "project_id", "parent_id", "slug", "directory", "title", "version", "time_created"},
    "message": {"id", "session_id", "time_created", "data"},
    "part": {"id", "message_id", "session_id", "time_created", "data"},
}
# Newest OpenCode DB migration (YYYYMMDD prefix) this projection was verified against.
# A *migration* is one recorded change to the database's shape; OpenCode names them
# "20260622202450_simplify_session_input". If the database has a newer one than this,
# _schema_check warns that the projection may now be incomplete — a nudge to re-check
# this adapter, never a refusal to run. Bump it after verifying against a newer OpenCode.
_KNOWN_MIGRATION_DATE = "20260622"

# Our own headless enrichment runs carry this title (sbconfig.OPENCODE_ENRICHMENT_TITLE
# once the enrichment provider lands); they must never be indexed as sessions.
# Enrichment asks a CLI to summarise a session; when that CLI is OpenCode, the summarising
# run is itself a session in OpenCode's database. Indexing those would fill the browser
# with this tool's own bookkeeping, so _tree() skips them by title.
try:  # pragma: no cover - the constant may not exist on older branches
    # Imported defensively: sbconfig may be absent (a bare checkout) or predate the
    # constant, and failing to import config must never break transcript reading.
    import sbconfig as _sbconfig
    ENRICHMENT_TITLE = getattr(_sbconfig, "OPENCODE_ENRICHMENT_TITLE", "session-browser-enrichment")
except Exception:  # noqa: BLE001
    ENRICHMENT_TITLE = "session-browser-enrichment"

# Messages already printed, so a persistent problem (an unreadable DB, a schema this
# adapter does not know) costs one stderr line per process instead of one per sync.
# Process-local by design: a fresh run should say it again.
_WARNED: set[str] = set()


def _warn(msg: str) -> None:
    """Loud, once per distinct message per process.

    stderr, not a logger: this module runs inside the watcher daemon (launchd/systemd
    capture stderr into its log), inside short-lived hook processes and inside tests that
    assert on the text. flush=True so a crash right after does not swallow the line.
    """
    if msg in _WARNED:
        return
    _WARNED.add(msg)
    print(f"[opencode] {msg}", file=sys.stderr, flush=True)


def _get(row: sqlite3.Row, col: str, default=None):
    """A database column if this OpenCode version has it, else `default`.

    Indexing a sqlite3.Row by a column the query did not return raises IndexError, and
    OpenCode adds and drops columns between releases. Asking through this helper is what
    lets one projection run against several schema versions (see _schema_check, which
    warns about what is missing).
    """
    return row[col] if col in row.keys() else default


def _loads(blob) -> dict:
    """Parse a JSON column into a dict; {} for anything unparseable or not an object.

    OpenCode stores each message and part as a JSON blob. A row with a truncated or
    corrupt blob must cost that one row, never the session — see the test that inserts a
    'not json' part and still expects the projection to succeed.
    """
    try:
        v = json.loads(blob) if isinstance(blob, (str, bytes)) else blob
    except (json.JSONDecodeError, TypeError):
        return {}
    return v if isinstance(v, dict) else {}


def _loads_any(blob):
    """A JSON column as-is — dict OR list (session.permission is a
    PermissionRuleset list; coercing it to {} made `opencode import` reject
    every re-import). None when empty or unparseable."""
    if isinstance(blob, (dict, list)):
        return blob or None
    if not isinstance(blob, (str, bytes)) or not blob:
        return None
    try:
        v = json.loads(blob)
    except (json.JSONDecodeError, TypeError):
        return None
    return v if isinstance(v, (dict, list)) and v else None


@dataclass
class SyncReport:
    """What one sync() pass did. Returned to callers, asserted on by the tests, and
    printed by `sb doctor`; sync() itself never raises, so this is how a caller learns
    that something went wrong."""

    # Root session ids whose mirror file was (re)written this pass.
    written: list[str] = field(default_factory=list)
    # Root ids whose session is gone from the database: archived, then unlinked.
    removed: list[str] = field(default_factory=list)
    # Files left alone — unchanged since the manifest was written, restored-but-not-yet
    # re-imported, or projected from a different OpenCode database.
    skipped: int = 0
    # Human-readable problems, each also printed once via _warn().
    warnings: list[str] = field(default_factory=list)
    # True when there is no database to read at all. Distinct from "no sessions": a
    # missing DB must never be read as "the user deleted everything".
    db_missing: bool = False


class OpenCodeSource:
    """The OpenCode adapter: reads one OpenCode database, maintains one mirror directory.

    Implements the SessionSource protocol from sources/base.py (discover, parse_header,
    parse_full, session_id_for_path, resume_command, is_available) plus the optional
    hooks the rest of the tool looks up with getattr(): watch_roots(), sync_trigger(),
    has_binary(), restore_path(), protect() and reimport().

    The unusual part is that discover() has a side effect — it re-projects the database
    into the mirror first. That is deliberate: it means no caller needs to know about the
    database, and a session is never missing merely because whoever asked forgot to sync.

    Instances hold only paths and a throttle timestamp, so constructing one is free and
    safe on a machine with no OpenCode at all.
    """

    # Identifies rows from this adapter in the registry's cli_source column.
    name = "opencode"

    def __init__(self, data_dir: Path | str = DATA_DIR, mirror_dir: Path | str = MIRROR_DIR,
                 db: Path | str | None = None, reimport_on_restore: bool = True):
        """Point the adapter at one OpenCode installation and one mirror directory.

        `data_dir` is OpenCode's own data directory (where the database and the spilled
        tool outputs live); `db` overrides the database file within it, for a channel
        build or a test; `reimport_on_restore` is the [sources.opencode] switch that
        decides whether a restore also puts the session back INTO OpenCode (the only
        place this adapter ever runs the binary).

        Tests construct one directly against a temporary directory: see `_oc_source` in
        tests/test_smoke.py.
        """
        # No disk access here: CI has no OpenCode, and the registry builds every adapter.
        self.data_dir = Path(os.path.expanduser(str(data_dir)))
        self.mirror_dir = Path(os.path.expanduser(str(mirror_dir)))
        self._db = Path(os.path.expanduser(str(db))) if db else None
        self.reimport_on_restore = reimport_on_restore
        # Throttle for discover(): a batch script may call it repeatedly, and projecting
        # the whole database on each call would be pure waste. The watcher's own
        # sync_trigger path is not throttled by this.
        self.sync_interval = 5.0     # discover() re-syncs at most this often per process
        self._last_sync = 0.0

    # -- database ----------------------------------------------------------------
    @property
    def db_path(self) -> Path:
        """Resolved like OpenCode does, without asking the binary: $OPENCODE_DB
        (absolute, or relative to the data dir) > config `db` > opencode.db >
        the newest channel DB (opencode-<channel>.db).

        Asking the binary (`opencode db path`) would be the obvious way to get this and
        is exactly what must not happen: that command opens the database read-write and
        checkpoints its write-ahead log, which is a write to the user's data just to
        answer a question. Recomputed on each access — cheap, and it picks up an
        environment change without rebuilding the adapter. ":memory:" is ignored because
        an in-memory database has no file to read."""
        env = os.environ.get("OPENCODE_DB")
        if env and env != ":memory:":
            p = Path(os.path.expanduser(env))
            return p if p.is_absolute() else self.data_dir / p
        if self._db is not None:
            return self._db
        default = self.data_dir / "opencode.db"
        if default.exists() or not self.data_dir.is_dir():
            return default
        # No opencode.db, but the directory exists: this is probably a channel install
        # (opencode-beta.db, opencode-nightly.db). Most-recently-modified wins, because
        # that is the one being used.
        channel = sorted(self.data_dir.glob("opencode*.db"), key=lambda p: p.stat().st_mtime, reverse=True)
        return channel[0] if channel else default

    def _open_ro(self) -> sqlite3.Connection | None:
        """A read-only connection to OpenCode's database, or None if there is none usable.

        `file:<path>?mode=ro` with uri=True is SQLite's read-only open: this process
        cannot modify the user's live data even by accident, and it does not checkpoint
        the write-ahead log out from under OpenCode. The write-ahead log file must still
        be READABLE, or recent messages would be invisible.

        timeout=5 waits politely if OpenCode is mid-write rather than failing instantly.
        The `SELECT 1 FROM sqlite_master LIMIT 1` is a probe: sqlite3.connect() succeeds
        lazily, so without touching the file a corrupt or permission-denied database only
        fails later, somewhere less convenient. Every failure is warned once and returned
        as None — the caller then serves whatever is already mirrored.

        Callers close the connection themselves (see _sync_locked's try/finally).
        """
        db = self.db_path
        if not db.exists():
            return None
        try:
            conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5)
            # Rows addressable by column name, which is what _get() and the projection
            # rely on across schema versions.
            conn.row_factory = sqlite3.Row
            conn.execute("SELECT 1 FROM sqlite_master LIMIT 1")
            return conn
        except sqlite3.Error as e:
            _warn(f"cannot open {db} read-only: {e}")
            return None

    def _schema_check(self, conn: sqlite3.Connection) -> list[str]:
        """Compare the live database against what this projection expects.

        Returns a list of human-readable warnings; never raises, never stops anything by
        itself. _sync_locked decides what to do with them: a missing TABLE or the v2-store
        signal below aborts the pass (keep serving the existing mirror), while a missing
        column is merely reported and projected around.

        This exists because OpenCode's storage has already changed shape twice under this
        tool, and the failure mode that matters is the silent one — a user opening the
        browser and concluding their sessions are gone.
        """
        # sqlite_master is SQLite's own catalogue of what exists in the file; this reads
        # the set of table names.
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        warnings: list[str] = []
        for table, cols in _REQUIRED.items():
            if table not in tables:
                warnings.append(f"table '{table}' is missing from {self.db_path} — nothing can be indexed")
                continue
            # PRAGMA table_info(<table>) lists one row per column; field 1 is the name.
            # The table name is interpolated rather than bound because SQLite does not
            # allow a parameter where an identifier goes — the values come from
            # _REQUIRED, a constant in this file, so nothing user-supplied reaches here.
            have = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
            missing = sorted(cols - have)
            if missing:
                warnings.append(f"table '{table}' lacks columns {missing} — projecting what is there")
        # The v2-store detector. OpenCode is migrating from message/part to a single
        # `session_message` table. During the transition both exist, so the test is not
        # "does session_message exist" but "is the OLD store empty while the NEW one has
        # rows" — that combination means this adapter would project zero messages and
        # quietly report every session as empty. Counting rows is cheap next to being
        # wrong; the warning names the table so the fix is obvious.
        if "message" in tables and "session_message" in tables:
            n_v1 = conn.execute("SELECT COUNT(*) FROM message").fetchone()[0]
            n_v2 = conn.execute("SELECT COUNT(*) FROM session_message").fetchone()[0]
            if n_v1 == 0 and n_v2 > 0:
                warnings.append("message/part tables are empty but session_message has rows — OpenCode "
                                "has moved to its v2 session store; this adapter reads message/part "
                                "and needs updating (existing mirror files are still served)")
        # Migration names begin with a date, so a plain string comparison of the first 8
        # characters ("20260701" > "20260622") is a date comparison.
        newest = self._migration_fingerprint(conn)
        if newest[:8].isdigit() and newest[:8] > _KNOWN_MIGRATION_DATE:
            warnings.append(f"DB migration {newest} is newer than this adapter was verified against "
                            f"({_KNOWN_MIGRATION_DATE}); the projection may be incomplete")
        return warnings

    @staticmethod
    def _migration_fingerprint(conn: sqlite3.Connection) -> str:
        """Newest applied migration name, informational only — never raises.
        On 1.18.15 `migration(id, time_completed)` keeps the NAME in `id`;
        `__drizzle_migrations` has a `name` column; either may change shape.

        Returns something like "20260622202450_simplify_session_input", or "" when the
        version cannot be determined. Two uses, both soft: the staleness warning in
        _schema_check, and the `source.migration` stamp on line 1 of every mirror file,
        which makes a file say which schema produced it.

        "Never raises" is the contract that matters. An earlier version assumed a `name`
        column, and on a real 1.18.15 database the resulting sqlite3.Error escaped
        sync(), escaped discover(), and took the whole nightly backfill down — for every
        source, not just this one. Hence: try each known table, ask what columns it
        actually has, and swallow any error."""
        for table in ("migration", "__drizzle_migrations"):
            try:
                # Identifiers cannot be bound as parameters; both names are constants here.
                cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
                col = next((c for c in ("name", "id") if c in cols), None)
                if col is None:
                    continue
                # Names sort chronologically because they start with a timestamp, so
                # MAX() is "the newest applied migration".
                v = conn.execute(f"SELECT MAX({col}) FROM {table}").fetchone()[0]
                # Guards against a table whose `id` is an integer counter rather than a
                # name — that is a shape this probe cannot read, so keep looking.
                if isinstance(v, str) and v[:8].isdigit():
                    return v
            except sqlite3.Error:
                continue
        return ""

    # -- tree / fingerprints -----------------------------------------------------
    @staticmethod
    def _tree(conn: sqlite3.Connection) -> tuple[dict, dict[str, list[str]]]:
        """by_id, {root_id: [descendant ids, breadth-first, by time_created]}.
        A child whose parent is gone is treated as a root; our own enrichment
        runs are skipped outright.

        OpenCode records a sub-agent — a helper conversation the assistant starts for a
        sub-task — as a separate session row with `parent_id` pointing at its parent, and
        those can nest. This turns that parent/child table into the shape the projection
        needs: for each top-level conversation, every descendant beneath it. Only roots
        become files; descendants are embedded in their root's file and their spend rolls
        up into it, so one conversation stays one row in the registry.

        Orphans (parent deleted, parent hidden as an enrichment run) become roots rather
        than disappearing — a session with no reachable ancestor still belongs to the
        user.

        `by_id` maps EVERY session id to its row, including enrichment runs, because
        _fingerprints and _info look rows up through it."""
        # SELECT * because the column set differs between OpenCode versions; _get()
        # handles whatever is or is not there. Session rows are metadata only (no
        # message bodies), so reading them all is cheap.
        rows = conn.execute("SELECT * FROM session").fetchall()
        by_id = {r["id"]: r for r in rows}
        kids: dict[str, list[str]] = defaultdict(list)
        roots: list[str] = []
        for r in rows:
            if (_get(r, "title") or "") == ENRICHMENT_TITLE:
                continue
            pid = _get(r, "parent_id")
            # "and pid in by_id" is the orphan rule: a parent that no longer exists
            # means this session is promoted to a root of its own.
            if pid and pid in by_id:
                kids[pid].append(r["id"])
            else:
                roots.append(r["id"])
        # Chronological order among siblings. Session ids are NOT sortable by time (they
        # embed a truncated clock), so time_created is the only correct key.
        for k in kids:
            kids[k].sort(key=lambda i: _get(by_id[i], "time_created") or 0)
        tree: dict[str, list[str]] = {}
        # Breadth-first walk down from each root, collecting descendants at every depth:
        # take the front of the queue, append its children to the output, and push those
        # children on to be walked in turn. Breadth-first keeps a grandchild after its
        # parent, which is the order the mirror file's message lines follow.
        for root in roots:
            out: list[str] = []
            queue = [root]
            while queue:
                cur = queue.pop(0)
                out += kids.get(cur, [])
                queue += kids.get(cur, [])
            tree[root] = out
        return by_id, tree

    @staticmethod
    def _fingerprints(conn: sqlite3.Connection, by_id: dict, tree: dict) -> dict[str, list]:
        """[newest activity ms, message count] per root, spanning the whole tree —
        a child can update after its root, and old rows have NULL time_updated.

        This pair is the change detector. sync() compares it against .manifest.json and
        rewrites a mirror file only when the pair differs, which is what keeps a sync
        over hundreds of sessions down to the few that moved. It has to be conservative
        in one direction only: if it says "unchanged" when something did change, the
        mirror goes stale — hence spanning the whole tree and taking the newest of
        several clocks.

        The timestamp also becomes the session's `last_activity` on line 1, so the
        browser's ordering depends on it too.
        """
        # One row per session: how many messages it has and the newest message time.
        # GROUP BY session_id does the counting inside SQLite rather than pulling every
        # message row into Python — this runs on every sync. `or 0` covers a session
        # with no messages, where MAX() is NULL.
        stats = {r[0]: (r[1], r[2] or 0) for r in conn.execute(
            "SELECT session_id, COUNT(*), MAX(time_created) FROM message GROUP BY session_id")}
        fps = {}
        for root, kids in tree.items():
            ids = [root, *kids]
            # Newest of three clocks, because no single one is reliable: the session's
            # own time_updated (NULL on older rows, hence the fall back to time_created),
            # and the newest message in the tree (a title edit bumps the session row, a
            # new reply does not).
            newest = max((_get(by_id[i], "time_updated") or _get(by_id[i], "time_created") or 0) for i in ids)
            newest = max(newest, *(stats.get(i, (0, 0))[1] for i in ids))
            # The count catches the case a timestamp cannot: rows deleted or re-added
            # without any clock moving forward.
            fps[root] = [int(newest), sum(stats.get(i, (0, 0))[0] for i in ids)]
        return fps

    # -- projection --------------------------------------------------------------
    @staticmethod
    def _info(row: sqlite3.Row) -> dict:
        """Session.Info in the export's camelCase shape (what `opencode import` decodes).

        One database row in, one dictionary out — the object that becomes `info` on line 1
        of a mirror file (and each entry of `children`). The shape is not ours to choose:
        it has to match what `opencode export` produces, because restore feeds these
        straight back to `opencode import`, which validates them strictly. So database
        `project_id` becomes `projectID`, the two time columns collapse into a nested
        `time` object, and so on.

        Everything is optional. Columns this OpenCode version lacks are read through
        _get() and simply do not appear; keys whose value is None are dropped at the end,
        because an explicit null is a decode error on import where an absent key is fine.
        """
        created = _get(row, "time_created")
        info = {
            "id": row["id"], "slug": _get(row, "slug") or "", "projectID": _get(row, "project_id"),
            "directory": _get(row, "directory") or "", "path": _get(row, "path"),
            "title": _get(row, "title") or "", "version": _get(row, "version") or "",
            "time": {"created": created, "updated": _get(row, "time_updated") or created},
        }
        # Present only on a child: this is how line 1's `children` say who they belong to.
        if _get(row, "parent_id"):
            info["parentID"] = row["parent_id"]
        # Straight copies whose only change is the key's spelling.
        for col, key in (("agent", "agent"), ("workspace_id", "workspaceID")):
            if _get(row, col):
                info[key] = row[col]
        # Columns that hold JSON themselves. _loads_any, not _loads: `permission` is a
        # LIST (a rule set), and flattening it to {} made `opencode import` reject every
        # re-import with a schema-decode error.
        for col in ("model", "revert", "permission", "metadata"):
            v = _loads_any(_get(row, col))
            if v is not None:
                info[col] = v
        if _get(row, "share_url"):
            info["share"] = {"url": row["share_url"]}
        # Lines added/removed/files touched, stored as three flat columns and exported as
        # one nested object; included only if OpenCode recorded any of them.
        summary = {k: _get(row, f"summary_{k}") for k in ("additions", "deletions", "files")}
        if any(v is not None for v in summary.values()):
            info["summary"] = {k: v for k, v in summary.items() if v is not None}
            diffs = _loads_any(_get(row, "summary_diffs"))
            if diffs is not None:
                info["summary"]["diffs"] = diffs
        # OpenCode's OWN notion of archived (the user hid the session in its UI) and of
        # compacting. Unrelated to this tool's archived rows — see docs/GLOSSARY.md.
        for col, key in (("time_archived", "archived"), ("time_compacting", "compacting")):
            if _get(row, col):
                info["time"][key] = row[col]
        if _get(row, "cost"):
            info["cost"] = row["cost"]
        # OpenCode's own running totals for the session. Input/output are tokens sent and
        # generated; cache read/write are the discounted tokens of a conversation prefix
        # the provider kept on its side. These are the session row's figures; the
        # authoritative per-model roll-up the cost extractor reads is computed
        # message-by-message in _project_root.
        tok = {k: _get(row, f"tokens_{k}") or 0 for k in ("input", "output", "reasoning", "cache_read", "cache_write")}
        if any(tok.values()):
            info["tokens"] = {"input": tok["input"], "output": tok["output"], "reasoning": tok["reasoning"],
                              "cache": {"read": tok["cache_read"], "write": tok["cache_write"]}}
        return {k: v for k, v in info.items() if v is not None}

    def _project_root(self, conn: sqlite3.Connection, root: str, kids: list[str],
                      by_id: dict, fingerprint: list, migration: str,
                      prev_inlined: dict | None = None, warnings: list | None = None) -> tuple[dict, list[dict]]:
        """Turn one conversation tree into the two halves of its mirror file.

        Returns (head, lines): the line-1 dictionary and the message lines that follow.
        _write_atomic serialises them; nothing is written here.

        Arguments beyond the obvious: `kids` is the root's descendants from _tree(), in
        the order their messages are written; `by_id` maps ids to session rows;
        `fingerprint` is the [newest activity, message count] pair, whose first element
        becomes `last_activity`; `migration` stamps the schema version on line 1;
        `prev_inlined` carries tool outputs forward from the file being replaced (see
        _inline_spill); `warnings`, when given, collects problems worth telling the user
        about without interrupting the projection.

        Two things happen in the same pass. The file's message lines are built, with the
        ids the database promoted out of the JSON put back so each line stands alone; and
        the line-1 `stats` are computed — turn count, first message, title, and spend
        per provider/model summed across the whole tree, which is what makes a sub-agent's
        cost show up on its parent's row.

        Reads the database, writes nothing, never spawns anything. One bad row degrades
        to an empty dictionary rather than an exception (see _loads).
        """
        lines: list[dict] = []
        # provider/model -> running token and cost totals, across the whole tree.
        models: dict[str, dict] = {}
        unattributed = 0
        turn_count = 0
        first_message = ""
        # message_count / recognised feed the "is this a schema I know?" test in
        # parse_header, the same guard the codex adapter learned to need.
        message_count = recognised = 0
        # Root first, then descendants, so the conversation reads in order and a reader
        # that only wants the main thread can stop at the first child id.
        for sid in [root, *kids]:
            # All of one session's messages, oldest first. ORDER BY time_created, id:
            # the timestamp is the real order, and the id breaks ties deterministically
            # so two syncs of an unchanged session produce byte-identical files.
            # `?` is a bound parameter — the id never goes into the SQL text.
            msgs = conn.execute("SELECT id, time_created, data FROM message WHERE session_id = ? "
                                "ORDER BY time_created, id", (sid,)).fetchall()
            # All of that session's parts in one query, then grouped in Python by the
            # message they belong to. One query per session beats one per message: a
            # long session has thousands of parts.
            parts_by_msg: dict[str, list] = defaultdict(list)
            for pr in conn.execute("SELECT id, message_id, time_created, data FROM part WHERE session_id = ? "
                                   "ORDER BY time_created, id", (sid,)):
                parts_by_msg[pr["message_id"]].append(pr)
            for m in msgs:
                info = _loads(m["data"])
                # The database promoted these out of the JSON blob to make them
                # indexable columns; put them back, or the exported document is missing
                # the keys `opencode import` requires.
                info["id"], info["sessionID"] = m["id"], sid
                parts = []
                for pr in parts_by_msg.get(m["id"], []):
                    pd = _loads(pr["data"])
                    pd["id"], pd["messageID"], pd["sessionID"] = pr["id"], m["id"], sid
                    # Large tool outputs live in a separate file that OpenCode deletes
                    # after a week; pull the content in while it still exists.
                    if pd.get("type") == "tool":
                        self._inline_spill(pd, prev_inlined or {})
                    parts.append(pd)
                # Every message is written out verbatim, including part types this
                # adapter does not recognise: the mirror is the lossless backup.
                lines.append({"type": "message", "session": sid, "info": info, "parts": parts})
                message_count += 1
                recognised += sum(1 for pd in parts if pd.get("type") in _KNOWN_PARTS)
                role = info.get("role")
                # Turns are counted on the ROOT only. A sub-agent's internal prompts are
                # the assistant talking to itself, not the user taking another turn.
                if role == "user" and sid == root:
                    if any(pd.get("type") == "compaction" for pd in parts):
                        continue                      # OpenCode's own context summary, not a turn
                    text = _user_text(parts)
                    if text:
                        turn_count += 1
                        first_message = first_message or text
                elif role == "assistant":
                    # Spend is counted on EVERY session in the tree, root and children
                    # alike — a sub-agent's tokens were really spent on this conversation.
                    key = _model_key(info)
                    tk = info.get("tokens") if isinstance(info.get("tokens"), dict) else {}
                    if not key and (tk or info.get("cost")):
                        # Spend without providerID/modelID (a renamed field upstream, an
                        # aborted message): bucket it, never drop it — the cost extractor
                        # treats this roll-up as authoritative.
                        #
                        # In other words: the totals here ARE the session's cost, so a
                        # message whose model cannot be named must still contribute, or
                        # the reported spend silently under-counts. A visible
                        # "unknown/unknown" row plus the warning below is the honest
                        # answer; a quietly smaller number is not.
                        key = "unknown/unknown"
                        unattributed += 1
                    if key:
                        cache = tk.get("cache") if isinstance(tk.get("cache"), dict) else {}
                        # setdefault creates the bucket on first sight of this model; a
                        # session may switch models part-way, so several buckets is normal.
                        acc = models.setdefault(key, {"input": 0, "output": 0, "reasoning": 0,
                                                      "cache_read": 0, "cache_write": 0, "cost": 0.0})
                        acc["input"] += int(tk.get("input") or 0)
                        acc["output"] += int(tk.get("output") or 0)
                        acc["reasoning"] += int(tk.get("reasoning") or 0)
                        acc["cache_read"] += int(cache.get("read") or 0)
                        acc["cache_write"] += int(cache.get("write") or 0)
                        acc["cost"] += float(info.get("cost") or 0)
        # Round once at the end: summing floats a few hundred times leaves a trail of
        # 0.018500000000000003, and six decimal places is well below a cent.
        for acc in models.values():
            acc["cost"] = round(acc["cost"], 6)
        if unattributed and warnings is not None:
            warnings.append(f"{root}: {unattributed} assistant message(s) carry tokens/cost but no "
                            f"providerID/modelID — spend bucketed under unknown/unknown")
        row = by_id[root]
        raw_title = _get(row, "title") or ""
        # Everything parse_header needs, on line 1, so listing sessions never reads past it.
        stats = {
            "turn_count": turn_count,
            # A preview for the browser list, not the message: the registry column is
            # capped at the same 500 characters.
            "first_message": first_message[:500],
            # A placeholder title ("New session - 2026-08-01T…") is no title: None lets
            # the UI fall back to the first message instead of showing the same string
            # on every row.
            "title": None if (not raw_title or _PLACEHOLDER_TITLE.match(raw_title)) else raw_title,
            # "The" model of a session that may have used several: the one that generated
            # the most output tokens, with input tokens breaking a tie. Output is the
            # better signal because that is the model doing the actual work.
            "model_used": max(models, key=lambda k: (models[k]["output"], models[k]["input"])) if models else None,
            "models": models,
            # OpenCode's own dollar figure, summed. Real money for a pay-per-use
            # provider; scripts/compute-costs.py takes it as authoritative rather than
            # re-pricing it, because OpenCode knows providers pricing.json does not.
            "cost_usd": round(sum(a["cost"] for a in models.values()), 6),
            # Epoch milliseconds, as OpenCode stores them; to_iso_utc converts when a
            # SessionHeader is built.
            "start_time": _get(row, "time_created"),
            # From the fingerprint, so "changed" and "last active" can never disagree.
            "last_activity": fingerprint[0],
            "message_count": message_count,
            "child_count": len(kids),
            "recognised_parts": recognised,
        }
        # `source` is provenance: which database produced this file, under which schema,
        # and when. _remove_deleted reads source.db to refuse to delete a mirror file
        # belonging to a DIFFERENT OpenCode installation.
        head = {"type": "session", "schema": SCHEMA, "info": self._info(row),
                "children": [self._info(by_id[k]) for k in kids], "stats": stats,
                "source": {"db": str(self.db_path), "migration": migration,
                           "mirrored_at": int(time.time() * 1000)}}
        return head, lines

    # Refuse to inline a spilled output larger than 16 MiB: the mirror file is read whole
    # by several consumers, and one runaway command's output must not make a session
    # unopenable. Such a part keeps its truncated preview.
    _SPILL_MAX = 16 * 1024 * 1024

    def _inline_spill(self, pd: dict, prev_inlined: dict | None = None) -> None:
        """Outputs over 2000 lines / 50 KB are spilled to <data>/tool-output/tool_<id>
        (state.metadata.outputPath; older builds only leave the path in the
        preview text) and purged after SEVEN days. The mirror is the backup, so
        inline the blob while it exists. Once the blob is gone, the copy the
        previous projection already inlined is carried forward — a re-projection
        must never be lossier than the file it replaces.

        Modifies `pd` (one tool part) in place and returns nothing. `prev_inlined` is
        {part id: output text} from the mirror file about to be overwritten, built by
        _inlined_outputs. Marks what it did with state.metadata.inlined = True, which is
        what lets the next projection recognise its own work. Never raises: a missing or
        unreadable blob simply leaves the part with its truncated preview."""
        st = pd.get("state")
        if not isinstance(st, dict):
            return
        meta = st.get("metadata") if isinstance(st.get("metadata"), dict) else {}
        # Newer OpenCode records the spill path properly...
        ref = meta.get("outputPath")
        # ...older builds only mention it inside the preview text ("Full output saved
        # to: /…/tool-output/tool_abc"), so dig it out with _SPILL_RE.
        if not ref:
            m = _SPILL_RE.search(st.get("output") or "")
            ref = m.group(0) if m else None
        if not ref:
            return
        blob = Path(ref)
        # A relative path is relative to OpenCode's data directory, which may not be the
        # one this process is running in.
        if not blob.is_absolute():
            blob = self.data_dir / "tool-output" / blob.name
        text = None
        try:
            if blob.is_file() and blob.stat().st_size <= self._SPILL_MAX:
                text = blob.read_text(encoding="utf-8", errors="replace")
        except OSError:
            # Purged between the stat and the read, or unreadable: fall through to the
            # carry-forward below rather than failing the projection.
            text = None
        # The blob is gone (older than a week). If a previous projection saved a copy,
        # reuse it — rewriting this file must never lose what the old one held.
        if text is None:
            text = (prev_inlined or {}).get(pd.get("id"))
        if text is None:
            return
        st["output"] = text
        meta["inlined"] = True
        st["metadata"] = meta

    @staticmethod
    def _inlined_outputs(path: Path) -> dict[str, str]:
        """part id -> output text for every tool part the existing mirror file
        already inlined (the blob may be purged by now).

        Read from the file that is about to be replaced and handed to _project_root as
        `prev_inlined`, so a rewrite carries forward content OpenCode has since deleted.
        Returns {} for a missing or unreadable file — losing the carry-forward is a
        degradation, never a failure."""
        out: dict[str, str] = {}
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                # Skip line 1: it is the session header, and only message lines carry
                # parts. next(fh, None) rather than readline() so an empty file is fine.
                next(fh, None)
                for line in fh:
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    for pd in rec.get("parts") or []:
                        st = pd.get("state") if isinstance(pd, dict) else None
                        meta = st.get("metadata") if isinstance(st, dict) and isinstance(st.get("metadata"), dict) else {}
                        # Only parts a previous projection inlined: the marker is what
                        # distinguishes a rescued blob from an ordinary short output.
                        if meta.get("inlined") and isinstance(st.get("output"), str) and pd.get("id"):
                            out[pd["id"]] = st["output"]
        except OSError:
            pass
        return out

    @staticmethod
    def _write_atomic(path: Path, head: dict, lines: list[dict]) -> None:
        """Write one mirror file so a reader never sees it half-written.

        Everything goes to a temp file in the same directory, then os.replace() swaps it
        into place — on POSIX that rename is atomic, so any process opening the path gets
        either the whole old file or the whole new one. Same directory matters: a rename
        across filesystems is not atomic.

        JSON is written compactly (no spaces after separators) because these files are
        read by programs and can reach tens of megabytes.

        On ANY failure the temp file is removed and the error re-raised — a stray
        `.tmp` would otherwise accumulate on every failed sync. BaseException, not
        Exception, so a Ctrl-C cleans up too. _sync_locked catches what comes out and
        records it as a per-session warning."""
        # A temp name unique to this writer: the hook and the watcher project the
        # same root within seconds of each other, and a shared <file>.tmp let them
        # interleave bytes and race on os.replace.
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(head, separators=(",", ":")) + "\n")
                for line in lines:
                    fh.write(json.dumps(line, separators=(",", ":")) + "\n")
            os.replace(tmp, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    # -- manifest ----------------------------------------------------------------
    @property
    def _manifest_path(self) -> Path:
        """<mirror>/.manifest.json — the record of what was projected last time.

        Dot-prefixed and ending in .json, so session_id_for_path() (which demands a
        ses_<26>.jsonl name) can never mistake it for a session."""
        return self.mirror_dir / ".manifest.json"

    def _load_manifest(self) -> dict[str, list]:
        """{root id: [newest activity ms, message count]} as of the last sync, or {}.

        Compared against freshly computed fingerprints to decide which files to rewrite.
        A missing, unreadable or corrupt manifest is not an error worth reporting: the
        worst case is that every session is re-projected once, and the files that result
        are identical."""
        try:
            data = json.loads(self._manifest_path.read_text(encoding="utf-8"))
            # Nested under "roots" so the file has room for other keys later; anything
            # of an unexpected shape is treated as absent.
            roots = data.get("roots") if isinstance(data, dict) else None
            return dict(roots) if isinstance(roots, dict) else {}
        except (OSError, ValueError):
            return {}          # lost or corrupt: one full resync, no harm

    def _save_manifest(self, roots: dict[str, list]) -> None:
        """Write the manifest back, atomically (same temp-then-rename as a mirror file).

        Called once at the end of a sync, inside the lock. Atomicity matters because a
        half-written manifest that still parsed would claim some sessions are up to date
        when they are not — silently stale mirror files, the one failure mode this
        design must not have. Raises on I/O failure; sync() turns that into a warning."""
        fd, tmp = tempfile.mkstemp(dir=self.mirror_dir, prefix=".manifest.json.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"schema": SCHEMA, "roots": roots}, separators=(",", ":")))
            os.replace(tmp, self._manifest_path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    @contextlib.contextmanager
    def _mirror_lock(self):
        """Serialise sync() across processes (hook vs watcher vs nightly) on
        <mirror>/.sync.lock. No-op where flock is unavailable.

        Three independent processes project this same directory: the watcher daemon
        (woken by a database write), the OpenCode plugin's hook (spawned when a session
        ends) and the nightly refresh. Without this they overlap within seconds of each
        other, and the dangerous overlap is not a torn file — temp-then-rename already
        prevents that — but the DELETION pass: one process can be deciding "this mirror
        file has no session in the database, archive and unlink it" while another is
        mid-write, and the manifest each writes would clobber the other's.

        flock is an advisory whole-file lock held by the OPERATING SYSTEM for as long as
        the file is open, so it is released even if this process is killed — unlike a
        lock file whose existence means "locked", which a crash leaves behind forever.
        The file is opened "a+" (append/read) purely so it is created if absent and never
        truncated; nothing is ever written into it. Blocking is intentional: the loser
        waits and then sees the winner's fresh manifest, so it skips everything.

        On a platform with no fcntl (Windows) this yields without locking rather than
        failing — the pipeline still works, it just loses the guarantee."""
        try:
            import fcntl
        except ImportError:   # pragma: no cover — non-POSIX
            yield
            return
        with open(self.mirror_dir / ".sync.lock", "a+") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield
            finally:
                # Explicit unlock before the file closes, so the ordering is obvious;
                # closing would release it anyway.
                fcntl.flock(fh, fcntl.LOCK_UN)

    # -- sync --------------------------------------------------------------------
    def sync(self, only: list[str] | None = None, *, force: bool = False,
             on_delete=None) -> SyncReport:
        """Project every changed root session into the mirror. Read-only on the
        DB, never spawns a subprocess. An unreadable DB changes nothing.

        This is the heart of the adapter: afterwards, <mirror>/ holds one up-to-date
        JSONL per conversation and the rest of the tool can forget a database was ever
        involved. Callers: discover() (throttled), the watcher when it sees a database
        write, and the OpenCode hook for a single session.

        `only` restricts the pass to the listed root ids — what the hook uses, so ending
        one session does not re-examine hundreds. It also suppresses the deletion pass:
        "not in this list" must never be read as "gone from OpenCode".
        `force` re-projects even sessions the manifest calls unchanged.
        `on_delete` replaces the archiver used before a vanished session's mirror file is
        unlinked; it is called as on_delete(path, header_dict) and tests pass their own.

        Returns a SyncReport and NEVER raises — see the comment below."""
        report = SyncReport()
        # Nothing in here may escape: every batch script calls discover() outside
        # its per-file try, so an OSError here used to take the whole nightly down
        # for EVERY source. A failed sync degrades to "serve what is mirrored".
        try:
            self.mirror_dir.mkdir(parents=True, exist_ok=True)
            with self._mirror_lock():
                self._sync_locked(only, force, on_delete, report)
        except (OSError, sqlite3.Error) as e:
            msg = f"mirror sync failed under {self.mirror_dir}: {e}"
            _warn(msg)
            report.warnings.append(msg)
        return report

    def _sync_locked(self, only, force, on_delete, report: SyncReport) -> None:
        """The body of sync(), running with the mirror lock held and the directory made.

        Split out so sync() is nothing but "make the directory, take the lock, catch
        everything". Fills `report` in place; may raise OSError/sqlite3.Error, which
        sync() catches."""
        conn = self._open_ro()
        if conn is None:
            # No database (OpenCode not installed here, or it was moved). Distinct from
            # "no sessions": leave every mirror file exactly as it is.
            report.db_missing = True
            return
        try:
            for w in self._schema_check(conn):
                _warn(w)
                report.warnings.append(w)
            # Two warnings are fatal to a pass: a table this projection reads is gone, or
            # OpenCode has moved to its v2 store. Either way this adapter would project
            # zero messages, so stop and keep serving what is already mirrored rather
            # than overwrite good files with empty ones — and never run the deletion
            # pass, which would read "no sessions" as "the user deleted everything".
            if any("is missing" in w or "v2 session store" in w for w in report.warnings):
                return
            migration = self._migration_fingerprint(conn)
            try:
                by_id, tree = self._tree(conn)
                fps = self._fingerprints(conn, by_id, tree)
            except sqlite3.Error as e:
                # An odd DB must never take the whole backfill (every source) down.
                msg = f"cannot read sessions from {self.db_path}: {e}"
                _warn(msg)
                report.warnings.append(msg)
                return
            manifest = self._load_manifest()
            for root, kids in tree.items():
                if only is not None and root not in only:
                    continue
                # The filename IS the session id, which is what lets every consumer map
                # a file back to a row without opening it.
                path = self.mirror_dir / f"{root}.jsonl"
                # Unchanged since last time and the file is still there: leave it
                # untouched, mtime included (the watcher would otherwise re-index it).
                if not force and path.exists() and manifest.get(root) == fps[root]:
                    report.skipped += 1
                    continue
                try:
                    # Rescue anything the file being replaced had inlined, before it is
                    # overwritten.
                    prev = self._inlined_outputs(path) if path.exists() else {}
                    head, lines = self._project_root(conn, root, kids, by_id, fps[root], migration,
                                                     prev_inlined=prev, warnings=report.warnings)
                    self._write_atomic(path, head, lines)
                    manifest[root] = fps[root]
                    report.written.append(root)
                except Exception as e:  # noqa: BLE001 — one bad session must not stop the rest
                    msg = f"projection of {root} failed: {e}"
                    _warn(msg)
                    report.warnings.append(msg)
            # Deletions only on a full pass: with `only` set, the roots NOT listed were
            # never examined, so their absence here means nothing.
            if only is None:
                self._remove_deleted(set(tree), manifest, on_delete or self._archive_before_delete, report)
            # Last, so a crash mid-pass leaves the manifest describing less than what is
            # on disk — which costs a re-projection, never a stale file served as fresh.
            self._save_manifest(manifest)
        finally:
            conn.close()

    # -- deletions ---------------------------------------------------------------
    @staticmethod
    def _archive_before_delete(path: Path, header: dict) -> Path:
        """Default archiver: the versioned raw vault (<archive>/raw/YYYY/MM/<sid>.jsonl),
        the same copy refresh-all makes nightly and Restore reads back. With the
        archive switched off there is nowhere to put the last copy, so this
        RAISES — _remove_deleted then keeps the mirror file instead of destroying it.

        `header` is the session's parsed header as a plain dict; reasoning.archive_raw
        uses it to file the copy under the right year/month. Returns the path written.

        Read the raise as the design, not an oversight: this is the only moment the last
        copy of a session could be destroyed, so "nowhere safe to put it" must stop the
        deletion, not proceed with it."""
        # Imported here rather than at module scope: these modules read config and would
        # otherwise be pulled in by every process that merely imports this adapter.
        import reasoning
        import sbconfig
        if not sbconfig.REASONING_ENABLED:
            raise RuntimeError("reasoning archive disabled ([reasoning] enabled = false) — "
                               "no vault to hold the last copy")
        return reasoning.archive_raw(path, header)

    @staticmethod
    def _source_db_of(path: Path) -> str | None:
        """The DB a mirror file was projected from (line 1, source.db).

        The provenance check behind _remove_deleted: a file stamped with a different
        database path was produced by a different OpenCode installation, and this one
        knows nothing about whether its session still exists. Returns None when line 1
        cannot be read or carries no stamp (a file from before the stamp existed), which
        the caller treats as "no reason to doubt it"."""
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                # One line is enough — the stamp is on line 1 by construction.
                head = json.loads(fh.readline())
            src = head.get("source") if isinstance(head, dict) else None
            return src.get("db") if isinstance(src, dict) else None
        except (OSError, ValueError):
            return None

    def _remove_deleted(self, live_roots: set[str], manifest: dict, on_delete, report: SyncReport) -> None:
        """A mirror file whose root is gone from the DB is the LAST copy of that
        session: hand it to the archiver first, unlink only if that succeeded.
        The unlink is what the watcher turns into archived=transcript-missing.

        `live_roots` is every root id currently in the database; `manifest` is edited in
        place so a removed session stops being tracked; `on_delete(path, header)` is the
        archiver. Runs only on a full sync pass, with the mirror lock held.

        Why a delete path exists at all: OpenCode never prunes by itself, but
        `opencode session delete` cascades away every message of a session, and people do
        prune a growing database. When that happens the mirror file is all that is left,
        so it is archived into the raw vault and then removed — and its removal is
        ordinary news to the watcher, which marks the row as having no transcript. The
        user sees it in the Archived tab with a working Restore button
        (docs/GLOSSARY.md: Archived, Restore).

        Three guards below make deletion refuse rather than risk destroying work: a
        restore marker, foreign provenance, and a failing archiver."""
        from dataclasses import asdict
        # A restore that could not (yet) re-import leaves <file>.restored beside
        # the mirror file: OpenCode does not have that session, but the user
        # deliberately put it back — deleting it here would un-restore it on
        # the next WAL write. The marker goes once the session is in the DB.
        # Clear markers first: OpenCode now HAS these sessions, so the restore is
        # complete and the protection is no longer needed. Strips ".jsonl.restored" off
        # the name to recover the session id.
        for marker in self.mirror_dir.glob("ses_*.jsonl" + _RESTORED):
            if marker.name[:-len(".jsonl" + _RESTORED)] in live_roots:
                marker.unlink(missing_ok=True)
        foreign_warned = False
        # sorted() only so the order (and any warning) is reproducible run to run.
        for path in sorted(self.mirror_dir.glob("ses_*.jsonl")):
            sid = path.stem
            # Not a session file at all, or its session is alive: nothing to do. The
            # _SID test also keeps a stray "ses_notes.jsonl" from being archived.
            if not _SID.match(sid) or sid in live_roots:
                continue
            # Guard 1: restored by hand and not yet back in OpenCode. Deleting it now
            # would undo the restore seconds after the user asked for it.
            if path.with_name(path.name + _RESTORED).exists():
                report.skipped += 1
                continue
            # Guard 2: provenance.
            origin = self._source_db_of(path)
            if origin and origin != str(self.db_path):
                # Projected from ANOTHER OpenCode DB (XDG_DATA_HOME / OPENCODE_DB set
                # in the shell but not in the daemon's environment). Absent from
                # this DB proves nothing — never archive-and-unlink it from here.
                report.skipped += 1
                if not foreign_warned:
                    msg = (f"{sid}: mirror file was projected from another OpenCode DB ({origin}); "
                           f"this daemon resolves {self.db_path} — leaving it alone")
                    _warn(msg)
                    report.warnings.append(msg)
                    foreign_warned = True
                continue
            # The archiver files the copy by date, so it needs the header. A file too
            # damaged to parse still gets archived — with a minimal header, because the
            # bytes are worth keeping even when the metadata is not readable.
            header = self.parse_header(path)
            hdr = asdict(header) if header is not None else {"session_id": sid, "last_activity": ""}
            # Guard 3: archive BEFORE unlinking, and only unlink if it worked. Any
            # failure at all (disk full, archive disabled, permissions) keeps the file.
            try:
                on_delete(path, hdr)
            except Exception as e:  # noqa: BLE001 — keep the file, say why
                msg = f"{sid}: deleted in OpenCode but archiving the mirror failed ({e}); keeping it"
                _warn(msg)
                report.warnings.append(msg)
                continue
            try:
                path.unlink()
            except OSError as e:
                report.warnings.append(f"{sid}: could not unlink mirror file: {e}")
                continue
            # Only now is the session really gone from the mirror: stop tracking it, and
            # report it so the caller (and the tests) can see what happened.
            manifest.pop(sid, None)
            report.removed.append(sid)

    # -- identity ----------------------------------------------------------------
    def session_id_for_path(self, path: Path) -> Optional[str]:
        """The stem, for exactly <mirror>/ses_<26>.jsonl — never the manifest, a
        half-written .tmp, the DB/WAL, or an archive copy's ses_…@vN name. The
        file may already be gone (the watcher's delete handler).

        The mirror directory holds more than session files, and the watcher reports an
        event for every one of them, so "is this path a session, and which one?" has to
        be answerable from the name alone. Returning None means "not mine, ignore it".

        Deliberately strict about the archive copy. A vault copy is named ses_…@v2.jsonl,
        and answering with the bare id would let a consumer treat an old snapshot as the
        live file. parse_header reads the id from line 1 instead, which is why
        parse_full() works on a vault copy while this does not."""
        if path.suffix != ".jsonl":
            return None
        # _SID is anchored, so "ses_…@v2" and "ses_short" both fail to match.
        return path.stem if _SID.match(path.stem) else None

    # -- cheap header ------------------------------------------------------------
    @staticmethod
    def _read_head(path: Path) -> Optional[dict]:
        """Line 1 only — the whole point of putting the stats there.

        One readline() regardless of how big the file is, so listing a thousand sessions
        costs a thousand short reads. Returns the parsed dictionary, or None if the file
        is unreadable, is not JSON, or is not a mirror file (both the "session" type and
        an `info` object must be there — that pair is the recognition test)."""
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                line = fh.readline()
        except OSError:
            return None
        head = _loads(line.strip())
        return head if head.get("type") == "session" and isinstance(head.get("info"), dict) else None

    def parse_header(self, path: Path) -> Optional[SessionHeader]:
        """The facts one registry row needs, read from line 1 of a mirror file.

        Returns a SessionHeader (see sources/base.py) or None when there is nothing
        browsable: an unreadable file, a file that is not a mirror file, or a session
        that never held a conversation (an aborted launch). The indexer reads None as
        "skip this", so a session in a part schema this adapter does not recognise is
        indexed with 0 turns and a warning instead — the same rule the codex adapter
        learned the hard way.

        Works on a raw-archive copy as well as a live mirror file: everything comes from
        line 1, so nothing here touches the database, the manifest or mirror_dir. That is
        what lets Restore and the full-text indexer read a vault copy directly.
        """
        head = self._read_head(path)
        if head is None:
            return None
        info, st = head["info"], head.get("stats") or {}
        # From info, never the stem: archive copies are named <sid>@vN.jsonl.
        session_id = info.get("id") or path.stem
        turn_count = int(st.get("turn_count") or 0)
        first_message = st.get("first_message") or ""
        if turn_count == 0 and not first_message:
            if st.get("message_count") and not st.get("recognised_parts"):
                # Messages exist but no part type we know: a schema we have not
                # seen. Index it (0 turns) so the warning is actionable rather
                # than silently returning None — the codex outage lesson.
                _warn(f"{path.name}: messages carry no part type this adapter knows — "
                      f"indexing with 0 turns; OpenCode's part schema may have changed")
            else:
                return None            # no conversation at all (aborted / meta-only)
        cwd = info.get("directory") or ""
        # The times go through to_iso_utc, which turns OpenCode's epoch milliseconds into
        # the one UTC spelling every source shares, so a mixed Claude/Codex/OpenCode list
        # sorts correctly under a plain ORDER BY. The stats value is preferred over
        # info.time because stats.last_activity spans the sub-agent children too.
        return SessionHeader(
            session_id=session_id,
            cli_source=self.name,
            # The mirror directory, not a project directory: it is where the transcript
            # file lives, which is what restore and the watcher need.
            project_path=str(path.parent),
            cwd=cwd,
            # "/Users/x/proj" -> "proj", the short label the browser groups by.
            folder_name=Path(cwd).name if cwd else "",
            start_time=to_iso_utc(st.get("start_time") or (info.get("time") or {}).get("created")),
            last_activity=to_iso_utc(st.get("last_activity") or (info.get("time") or {}).get("updated")),
            first_message=first_message[:500],
            turn_count=turn_count,
            title=st.get("title") or None,
            model_used=st.get("model_used") or None,
            cli_version=info.get("version") or None,
        )

    # -- full parse ----------------------------------------------------------------
    def parse_full(self, path: Path) -> Optional[ParsedSession]:
        """Root turns only, in order. Children stay lossless in the file and
        surface as the `subtask` tool call that spawned them; the compaction
        pair (OpenCode's own context summary), synthetic/ignored text and
        part-less aborted messages are not turns. Works on an archive copy too:
        nothing here touches the DB, the manifest or mirror_dir.

        Returns a ParsedSession (header plus ordered turns) or None. Used by everything
        that needs the conversation itself rather than a summary: the full-text index,
        the reasoning trail, the enrichment prompt, Copy Context and Export.

        "Root turns only" is the deliberate part. A sub-agent's internal back-and-forth
        is the assistant working, not the user conversing, so it stays in the file (for
        the backup and for search) but is not replayed as dialogue; what the reader sees
        instead is the `subtask` tool call that started it."""
        header = self.parse_header(path)
        if header is None:
            return None
        root = header.session_id
        turns: list[Turn] = []
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    rec = _loads(line.strip())
                    # Skips line 1 (type "session") and every child's message lines in
                    # one test. Comparing against the header's id, not the filename, is
                    # what makes this work on a ses_…@v2.jsonl archive copy.
                    if rec.get("type") != "message" or rec.get("session") != root:
                        continue
                    info = rec.get("info") if isinstance(rec.get("info"), dict) else {}
                    parts = [pd for pd in (rec.get("parts") or []) if isinstance(pd, dict)]
                    role = info.get("role")
                    if role == "user":
                        # A compaction part marks OpenCode's own summary of older
                        # history, written to fit the context window. It arrives as a
                        # user message but the user never typed it.
                        if any(pd.get("type") == "compaction" for pd in parts):
                            continue
                        text = _user_text(parts)
                        if text:
                            turns.append(Turn(role="user", content=text))
                    elif role == "assistant":
                        # The other half of the compaction pair: the summary itself.
                        if info.get("summary"):
                            continue
                        text = _user_text(parts)
                        tools = _tool_calls(parts)
                        # A reply that neither said nor did anything is an aborted turn
                        # (the user pressed Escape); it has no parts and is skipped.
                        if text or tools:
                            turns.append(Turn(role="assistant", content=text, tool_calls=tools))
        except OSError:
            return None
        return ParsedSession(header=header, turns=turns)

    # -- resume / availability ---------------------------------------------------
    def resume_command(self, session_id: str) -> str:
        """The command that reopens this session in OpenCode with its history intact.

        Shown by the UI and run by the `cr` shell helper (bin/resume-here.sh); nothing is
        executed here. shlex.quote keeps an odd id from being split or interpreted by the
        shell."""
        # A global lookup by id that runs in the CURRENT cwd — exactly `cr`'s meaning.
        return f"opencode --session {shlex.quote(session_id)}"

    def has_binary(self) -> bool:
        """Whether `opencode` is runnable from here — a UI hint for resume/re-import only.

        Never consulted by indexing (see is_available). The one code path that acts on a
        False answer is _run_import, which then tells the user the command to run by hand
        instead of failing silently."""
        return shutil.which("opencode") is not None

    def is_available(self) -> bool:
        """Something to read: the DB, or a mirror that keeps serving after
        OpenCode is gone. Never gated on the binary being on PATH.

        The question is about DATA, not installed software. A machine that never had
        OpenCode answers False and the adapter stays out of the way; a machine where it
        was uninstalled still has a mirror full of history, and keeping that browsable is
        the point of the tool. The watcher in particular runs under a minimal PATH where
        the binary may well be invisible."""
        if self.db_path.exists():
            return True
        return self.mirror_dir.is_dir() and any(self.mirror_dir.glob("ses_*.jsonl"))

    # -- watcher hooks -------------------------------------------------------------
    def watch_roots(self) -> list[Path]:
        """The mirror (session files → the ordinary handler) and the data dir
        (DB/WAL writes → sync_trigger). The mirror is our own directory, so it
        is created here: a watcher started before the first sync must be able
        to subscribe to it, or the files that sync writes go unseen.

        Returns (path, recursive) PAIRS, not bare paths — this adapter is the reason the
        protocol allows them. Every other source returns plain paths and gets recursive
        watching; here the second root must NOT be recursive (see below), so the flag has
        to travel with the path. watcher.py accepts either form.

        Failure to create the mirror directory is ignored: a read-only or missing parent
        must not stop the watcher from subscribing to the other roots."""
        try:
            self.mirror_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        # The data dir is watched NON-recursively: only opencode*.db / -wal at
        # its top level ever trigger a sync, while log/, tool-output/, snapshot/
        # and project/ are hundreds of directories (one inotify watch each).
        return [(self.mirror_dir, True), (self.data_dir, False)]

    def sync_trigger(self, path: Path) -> bool:
        """A write to opencode*.db or its WAL means sessions changed. Never the
        -shm, log/, tool-output/, auth.json — those churn constantly.

        The bridge between "a file changed" and "a database row changed". A filesystem
        watcher cannot see rows, so the watcher asks this of every event under the data
        directory: True means "re-project the mirror", and the create/modify/delete
        events that produces then flow through the ordinary per-file handler. The watcher
        debounces, so a burst of writes costs one sync.

        The write-ahead log (`.db-wal`) matters more than the database file itself: in
        WAL mode a new message lands there first and the main file may not be touched for
        a long time. `-shm` is a shared-memory index rewritten constantly with no new
        content, so triggering on it would mean syncing in a loop."""
        name = path.name
        return name.startswith("opencode") and (name.endswith(".db") or name.endswith(".db-wal"))

    # -- restore -------------------------------------------------------------------
    def restore_path(self, row) -> Optional[Path]:
        """Where this row's mirror file lives; restoring the raw copy here makes
        the row live again. Contained to the mirror dir so a corrupted row can
        never make restore write elsewhere.

        `row` is one `sessions` row; only session_id is read. Nothing is written here —
        restore.py copies the vault copy to the returned path. None means "refuse", and
        the UI then shows the truth instead of a Restore button that fails.

        Unlike the file-based adapters this ignores the row's recorded project_path
        entirely: an OpenCode session's home is this mirror directory and nowhere else,
        so the id alone determines the destination."""
        dest = self.mirror_dir / f"{row['session_id']}.jsonl"
        # Containment check: relative_to() raises ValueError when the resolved path is
        # outside the mirror, which is what stops a session_id like "../../etc/x" from
        # steering a write out of the directory.
        try:
            dest.resolve().relative_to(self.mirror_dir.resolve())
        except ValueError:
            return None
        return dest

    def protect(self, path: Path) -> None:
        """Called by restore BEFORE the raw copy is written back: the marker
        must already exist when the file appears, or a watcher sync in between
        sees a mirror file with no DB row and no marker and archives it away.

        Writes <path>.restored, whose mere existence is the signal (its content is just a
        timestamp for a human reading the directory). _remove_deleted honours it;
        reimport() and the next full sync clear it once OpenCode holds the session again.

        The ORDER is the whole point: the window between the file appearing and the
        marker appearing is exactly how long another process has to mistake a deliberate
        restore for an orphan and delete it. Failures are ignored — a restore that cannot
        write a marker is still better than no restore, and the file is protected by the
        row being live in the common case."""
        marker = path.with_name(path.name + _RESTORED)
        try:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(_now_iso(), encoding="utf-8")
        except OSError:
            pass

    def reimport(self, path: Path) -> tuple[bool | None, str]:
        """Second half of a restore: put the session back into OpenCode itself
        with `opencode import`, root first, then each child. Import re-homes a
        session to the directory it runs in, so it runs in the session's own
        directory ($HOME when that is gone). Idempotent upstream, but sessions
        still present are skipped. None = disabled by config.

        Returns (ok, detail): True with a summary, False with a reason the UI shows, or
        None when re-import is switched off. Failure is never fatal — the session is
        already browsable again by the time this runs; what is missing is only the
        ability to CONTINUE it inside OpenCode.

        Two halves of a restore, and only this one runs the binary: the first half put
        the transcript back where this tool can read it, this one puts the session back
        where OpenCode can resume it. Both are user-initiated, which is why spawning a
        process is acceptable here and nowhere else in this module."""
        # Mark first: until OpenCode holds the session again, sync() must not
        # treat this mirror file as "deleted in OpenCode" (see _remove_deleted).
        marker = path.with_name(path.name + _RESTORED)
        try:
            marker.write_text(_now_iso(), encoding="utf-8")
        except OSError:
            pass
        ok, detail = self._reimport(path)
        # Only a confirmed success clears the protection. If the import failed or was
        # skipped, OpenCode still lacks the session and the marker must stay.
        if ok is True:
            marker.unlink(missing_ok=True)
        return ok, detail

    def _reimport(self, path: Path) -> tuple[bool | None, str]:
        """The import work itself, without the marker bookkeeping reimport() adds.

        Rebuilds export documents from the mirror file, skips whatever OpenCode already
        has, writes one JSON file per session and imports them root first. Returns the
        same (ok, detail) triple-state as reimport(). Tests replace _run_import to drive
        this without a real binary."""
        if not self.reimport_on_restore:
            return None, "re-import into OpenCode disabled ([sources.opencode] reimport_on_restore = false)"
        # One document per session in the tree: the root, then each sub-agent child.
        docs = to_export_docs(path)
        if not docs:
            return False, "mirror file holds no session document"
        # Importing a session OpenCode already has is harmless but wasteful, and a
        # partly-restored tree is the normal case after a failed attempt.
        existing = self._existing_ids([d["info"]["id"] for d in docs])
        todo = [d for d in docs if d["info"]["id"] not in existing]
        if not todo:
            return True, "already present in OpenCode — nothing to re-import"
        # Path("") is Path(".") and "." is a directory: an empty recorded
        # directory used to re-home the session into the UI's own cwd, silently.
        raw_dir = docs[0]["info"].get("directory") or ""
        directory = Path(raw_dir) if raw_dir else None
        cwd = directory if directory is not None and directory.is_dir() else Path.home()
        note = ("" if directory is not None and cwd == directory else
                f" (original directory {raw_dir or '(none recorded)'} is gone; imported from {cwd})")
        # Export docs live under the mirror, not a temp dir: a failure message
        # that says "run: opencode import <file>" must name a file that exists.
        export_dir = self.mirror_dir / ".reimport"
        export_dir.mkdir(parents=True, exist_ok=True)
        done = 0
        # Root first, children after: a child references its parent, and importing it
        # into a database that does not have the parent yet leaves it orphaned.
        for doc in todo:
            f = export_dir / f"{doc['info']['id']}.json"
            f.write_text(json.dumps(doc), encoding="utf-8")
            ok, detail = self._run_import(f, cwd)
            # Stop at the first failure and say how far it got — the successfully
            # imported sessions stay, and a retry skips them via _existing_ids.
            if not ok:
                return False, (f"opencode import failed for {doc['info']['id']} "
                               f"({done} of {len(todo)} imported): {detail}")
            # Only a succeeded document is cleaned up; a failed one stays so the error
            # message can name a file the user can import by hand.
            f.unlink(missing_ok=True)
            done += 1
        return True, f"re-imported {done} session(s) into OpenCode{note}"

    def _run_import(self, file: Path, cwd: Path) -> tuple[bool, str]:
        """The only place this adapter runs the binary — user-initiated Restore.

        Runs `opencode import <file>` with the working directory set to `cwd`, and
        returns (ok, detail) where detail is the tail of the command's output. Success is
        defined as THE SESSION BEING IN THE DATABASE AFTERWARDS, not as a clean exit —
        see the three checks at the end. Never raises; a missing binary, a crash and a
        timeout all come back as False with an explanation.

        Tests replace this method wholesale to exercise the restore flow offline."""
        if not self.has_binary():
            # Not an error the user can do nothing about: hand them the exact command.
            return False, f"`opencode` is not on PATH — run manually: opencode import {file}"
        # Imported here, not at module scope, so that merely importing this adapter never
        # pulls in subprocess machinery — the indexing path must never shell out.
        import subprocess
        # Read the id now: it is what the post-import verification looks for.
        try:
            sid = json.loads(Path(file).read_text(encoding="utf-8"))["info"]["id"]
        except (OSError, ValueError, KeyError, TypeError) as e:
            return False, f"not an export document: {e}"
        try:
            # cwd matters: `opencode import` re-homes the session to whatever directory
            # it runs in, so running it anywhere else would silently move the restored
            # session to the wrong project. Output is captured rather than printed
            # because the caller turns it into a UI message. The timeout bounds a hung
            # binary. OPENCODE_DISABLE_AUTOUPDATE stops the CLI from deciding to upgrade
            # itself mid-restore — which would also mean running migrations on the DB.
            proc = subprocess.run(["opencode", "import", str(file)], cwd=str(cwd),
                                  capture_output=True, text=True, timeout=120,
                                  env={**os.environ, "OPENCODE_DISABLE_AUTOUPDATE": "1"})
        except (OSError, subprocess.TimeoutExpired) as e:
            return False, str(e)
        err = (proc.stderr or "").strip()
        # Last 300 characters of stderr then stdout: enough to identify the failure,
        # short enough to show in the UI.
        tail = (err + "\n" + (proc.stdout or "")).strip()[-300:]
        # Check 1: the ordinary one.
        if proc.returncode != 0:
            return False, f"exit {proc.returncode}: {tail}"
        # Check 2. A schema-decode failure prints "Error: Unexpected error" and STILL
        # exits 0 (a Bun defect): the session landing in the DB is the proof.
        if "Error:" in err:
            return False, tail
        # Check 3, the authoritative one: re-read the database and confirm the session is
        # really there. Reporting a restore that did not happen is the worst outcome —
        # the user would think their work is back when it is not.
        if sid not in self._existing_ids([sid]):
            return False, f"{sid} did not appear in {self.db_path} after import (exit 0): {tail}"
        return True, tail

    def _existing_ids(self, ids: list[str]) -> set[str]:
        """Which of these session ids OpenCode's database currently holds.

        Used to skip sessions already present before an import and to verify one landed
        afterwards. Returns an empty set if the database cannot be read, which the
        callers read conservatively: "assume nothing is there" means re-import attempts
        rather than a false success."""
        conn = self._open_ro()
        if conn is None:
            return set()
        try:
            # One "?" placeholder per id, joined into "?,?,?" — the only way to bind a
            # variable-length IN list in SQLite. The ids are still BOUND values, never
            # spliced into the SQL text, so nothing here can be injected.
            marks = ",".join("?" * len(ids))
            return {r[0] for r in conn.execute(f"SELECT id FROM session WHERE id IN ({marks})", ids)}
        except sqlite3.Error:
            return set()
        finally:
            conn.close()

    # -- discovery ---------------------------------------------------------------
    def discover(self) -> Iterator[Path]:
        """Every mirror file, after quietly bringing the mirror up to date.

        The entry point every caller uses — the indexer, backfill, prune, `sb doctor`.
        The side effect is the design: because discover() syncs first, no caller has to
        know that OpenCode keeps its sessions in a database, and a freshly ended session
        is never missing merely because nobody thought to sync.

        Throttled to one sync per `sync_interval` seconds per process, so a batch script
        calling discover() in a loop does not re-project everything each time; the
        watcher has its own trigger for immediacy. sync() never raises, so a broken
        database degrades to "serve whatever is already mirrored".

        Yields lazily, sorted by name for a reproducible order. Symlinks are skipped (a
        link could yield a session twice or point outside the mirror), and _SID filters
        out the manifest, the lock, temp files and any archive copy that was dropped in."""
        if time.time() - self._last_sync >= self.sync_interval:
            self.sync()
            self._last_sync = time.time()
        if not self.mirror_dir.is_dir():
            return
        for p in sorted(self.mirror_dir.glob("ses_*.jsonl")):
            if not p.is_symlink() and _SID.match(p.stem):
                yield p


def to_export_docs(path: Path | str) -> list[dict]:
    """The mirror file as `opencode export` documents — {info, messages:[{info,
    parts}]} — root first, then each embedded child, for `opencode import`.

    The inverse of the projection: mirroring flattened a conversation tree into one file,
    and this unpacks it back into the per-session documents OpenCode's importer expects.
    Returns [] for a missing, unreadable or non-mirror file rather than raising.

    A module-level function, not a method, because it needs nothing from the adapter —
    restore and the tests call it directly on a path.
    """
    path = Path(path)
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            head = _loads(fh.readline().strip())
            if head.get("type") != "session" or not isinstance(head.get("info"), dict):
                return []
            # Import order: the root, then its children. `order` fixes it; `infos` holds
            # the Session.Info each document needs.
            order = [head["info"]["id"]] + [c["id"] for c in head.get("children") or [] if isinstance(c, dict)]
            infos = {head["info"]["id"]: head["info"], **{c["id"]: c for c in head.get("children") or []}}
            messages: dict[str, list] = {sid: [] for sid in order}
            # Note the loop continues from the same handle, so line 1 is already consumed.
            for line in fh:
                rec = _loads(line.strip())
                if rec.get("type") != "message":
                    continue
                # Sort each message back under the session it belongs to. An id that is
                # not in `order` is silently dropped — it has no document to go in.
                sid = rec.get("session")
                if sid in messages:
                    messages[sid].append({"info": rec.get("info") or {}, "parts": rec.get("parts") or []})
    except OSError:
        return []
    return [{"info": infos[sid], "messages": messages[sid]} for sid in order]


def _user_text(parts: list[dict]) -> str:
    """The human-typed text of a user message: non-synthetic, non-ignored text parts.

    OpenCode adds text of its own to a user message — file contents it attached, system
    reminders, instructions — and flags those `synthetic`; `ignored` marks text the model
    was told to disregard. Neither is something the person typed, so neither belongs in a
    first-message preview or a turn count. Several kept parts are joined with newlines;
    "" means this message said nothing the user wrote.

    Also used for assistant messages in parse_full, where the same rule applies."""
    out = []
    for pd in parts:
        if pd.get("type") == "text" and not pd.get("synthetic") and not pd.get("ignored"):
            t = pd.get("text")
            if isinstance(t, str) and t.strip():
                out.append(t.strip())
    return "\n".join(out)


# The argument worth showing for a tool call, most identifying first: the shell command,
# then the file, then the search term. Both spellings of each are listed because tool
# inputs are camelCase in OpenCode (filePath), snake_case elsewhere.
_INPUT_KEYS = ("command", "cmd", "filePath", "file_path", "path", "pattern", "query",
               "description", "url")


def _summarize_input(inp) -> str:
    """One short line describing what a tool call was asked to do.

    A tool call's input can be anything from a string to a deeply nested object, and the
    whole thing is neither readable in a decision trail nor useful in a search index. So:
    take the first recognised key ("command=ls"), truncated to 140 characters, and fall
    back to naming up to five keys when none is recognised. Returns "" for input that is
    neither a string nor an object."""
    if isinstance(inp, str):
        return inp[:140]
    if not isinstance(inp, dict):
        return ""
    for key in _INPUT_KEYS:
        if key in inp:
            return f"{key}={str(inp[key])[:140]}"
    # Unrecognised shape: the key names at least say what kind of call it was.
    return ", ".join(list(inp.keys())[:5])


def _tool_calls(parts: list[dict]) -> list[dict]:
    """[{name, input}] — build-fts indexes `input`, the decision trail lists both.

    What an assistant turn actually DID, as a short list: the commands it ran, the files
    it read or changed, the sub-agents it started. Attached to each assistant Turn by
    parse_full. Returns [] for a message that only wrote text."""
    calls = []
    for pd in parts:
        t = pd.get("type")
        if t == "tool":
            st = pd.get("state") if isinstance(pd.get("state"), dict) else {}
            calls.append({"name": pd.get("tool") or "", "input": _summarize_input(st.get("input"))})
        elif t == "subtask":
            # A sub-agent being spawned. Its own conversation is a child session embedded
            # further down the file; here it shows as the single call that started it,
            # which is how a reader sees where the work went. `?` when the agent is
            # unnamed; capped at 160 characters like any other call summary.
            what = pd.get("prompt") or pd.get("description") or ""
            calls.append({"name": "subtask", "input": f"agent={pd.get('agent') or '?'}: {what}"[:160]})
    return calls


def _model_key(info: dict) -> str:
    """'provider/model' — OpenCode's own spelling (-m provider/model).

    The bucket key spend is accumulated under, e.g. "anthropic/claude-sonnet-5": the
    provider is who served the model, the model is which one. OpenCode has moved these
    fields around between versions, so both the flat spelling and the nested `model`
    object are accepted. Returns "" when no model can be named at all — _project_root
    then buckets that message's spend under "unknown/unknown" rather than losing it.

    A provider-less model keeps its bare name rather than gaining an empty prefix, so
    pricing.json's alias matching still has something clean to match."""
    # Newer shape: the ids sit directly on the message.
    model = info.get("modelID") or ""
    provider = info.get("providerID") or ""
    # Older shape: a nested object, where the model id may be called `id`.
    if not model:
        m = info.get("model")
        if isinstance(m, dict):
            model, provider = m.get("modelID") or m.get("id") or "", m.get("providerID") or provider
    if not model:
        return ""
    return f"{provider}/{model}" if provider else model
