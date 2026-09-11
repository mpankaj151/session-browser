"""Data model + adapter Protocol shared by every CLI source.

Adding a new CLI (codex, opencode, ollama, ...) means writing one module that
implements SessionSource and adding it to _FACTORIES in sources/registry.py
(plus a [sources.<cli>] block in config.toml.example). Nothing else in the
system needs to change — the indexer, DB, UI, watcher and MCP server are all
source-agnostic and speak only SessionHeader / Turn / ParsedSession.

Vocabulary, defined once in docs/GLOSSARY.md and used throughout: a *session* is one
conversation with a coding CLI; its *transcript* is the file that CLI wrote to record it;
a *turn* is one exchange within it (what the user typed plus the reply). An *adapter*
(or *source*) is the one class per CLI that knows where those files live and how to read
them.

The three data shapes below are the whole contract between "how some CLI stores things"
and "the rest of this program":

    SessionHeader   the cheap facts — id, directory, time span, turn count, first
                    message, model. Enough to build a registry row without reading a
                    possibly multi-megabyte file end to end.
    Turn            one message, already reduced to plain text (+ which tools it ran).
    ParsedSession   a header plus every turn, for the jobs that do need the full text
                    (full-text search, the reasoning extractor, export, enrichment).

Invariants every adapter must honour, because callers rely on them:

  * `session_id_for_path(p)` and `parse_header(p).session_id` agree for any transcript
    path. The first works without opening the file (it is used on deletion, when the file
    is already gone); the second is what actually gets stored. When they disagree, the
    row and the file part company and housekeeping archives a live session.
  * `parse_header()` returns None — rather than raising — for anything that is not a real
    session of this CLI: an unreadable file, a subagent sidechain, a headless run this
    tool itself started.
  * Every timestamp is passed through to_iso_utc() first (see below).
  * `discover()` yields the canonical file for each session exactly once, and never a
    symlink to one.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Literal, Optional, Protocol, runtime_checkable

# Just an alias for readability: wherever you see SourceName, the value is the short
# adapter key ("claude"), the same string used in config.toml, in _FACTORIES, and stored
# in the sessions.cli_source column.
SourceName = str  # "claude" | "copilot" | "codex" | "opencode" | ...


def to_iso_utc(value) -> str:
    """Normalize any timestamp an adapter sees to one canonical, lexicographically
    sortable form: `YYYY-MM-DDTHH:MM:SS.mmmZ` in UTC.

    The DB orders sessions by string comparison on these columns, so every source
    MUST emit the same spelling — `...+00:00` vs `...Z` vs naive-local strings
    sort wrong against each other. Accepts ISO strings (any offset spelling),
    datetime objects (naive = assume local), and epoch seconds. Returns '' for
    anything unparseable.

    Worked examples (all four land on the same string, which is the point):

        to_iso_utc("2026-06-19T12:00:00Z")            -> "2026-06-19T12:00:00.000Z"
        to_iso_utc("2026-06-19T12:00:00+00:00")       -> "2026-06-19T12:00:00.000Z"
        to_iso_utc("2026-06-19T17:30:00+05:30")       -> "2026-06-19T12:00:00.000Z"
        to_iso_utc(1781870400)                        -> "2026-06-19T12:00:00.000Z"
        to_iso_utc("garbage") / to_iso_utc(None)      -> ""

    Returning '' rather than raising is deliberate: a transcript with one malformed
    timestamp should still index, and '' is exactly what the upsert treats as "not known
    yet" (it neither sticks nor overwrites a good value).
    """
    if value is None or value == "":
        return ""
    if isinstance(value, (int, float)):
        # epoch MILLISECONDS from a future adapter would otherwise become a
        # year-56000 date silently; anything past ~5138 AD in seconds is ms.
        if value > 1e11:
            value /= 1000.0
        dt = datetime.fromtimestamp(value, tz=timezone.utc)
    elif isinstance(value, datetime):
        dt = value if value.tzinfo else value.astimezone()
    elif isinstance(value, str):
        # fromisoformat rejects >6 fractional digits (nanosecond RFC3339);
        # truncate rather than silently returning '' for a valid timestamp.
        # The regex keeps the first 6 fractional digits and drops the rest:
        # "2026-04-30T18:17:38.123456789Z" -> "...38.123456+00:00". `\1` is the kept
        # ".123456"; the trailing `\d+` (the nanosecond tail) is what gets deleted.
        s = re.sub(r"(\.\d{6})\d+", r"\1", value.replace("Z", "+00:00"))
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            return ""
        if dt.tzinfo is None:
            # No offset in the string: the CLI wrote local wall-clock time, so interpret
            # it in this machine's zone rather than pretending it was already UTC.
            dt = dt.astimezone()
    else:
        return ""
    # One spelling, always: UTC, exactly three fractional digits, a literal "Z".
    # Milliseconds are truncated rather than rounded — ordering is what matters, and
    # rounding could push a timestamp past the next event's.
    dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


@dataclass
class SessionHeader:
    """The cheap facts about one session — exactly the columns indexer.upsert() writes.

    "Cheap" is the contract: producing one must not require reading a whole transcript,
    because the watcher rebuilds a header on every debounced file change while a session
    is still being written to.

    Empty string vs None matters here. '' means "this source has no value for that, or
    not yet" and the upsert skips over it; it is NOT stored as an empty value. Fields
    that are Optional simply default to None when the CLI does not record them.
    """
    # Whatever the CLI calls this session. Unique across CLIs in practice (Claude uses a
    # UUID, OpenCode a `ses_…` string), which is why the registry can key on it alone.
    session_id: str
    # Which adapter produced this — "claude", "copilot", ... Stored so the UI can offer
    # the right resume command.
    cli_source: SourceName
    # The directory holding the canonical transcript. The watcher compares a deleted
    # file's parent against this to tell the real transcript from a symlinked copy, and
    # restore writes back here.
    project_path: str
    # The working directory the user was in when they ran the CLI — the project the
    # session was ABOUT, as opposed to project_path, which is where the CLI stored it.
    cwd: str
    # Last component of cwd ("session-browser"): what the UI shows as the project name.
    folder_name: str
    # First and last activity, both already normalised by to_iso_utc().
    start_time: str
    last_activity: str
    # The user's opening message, trimmed by the adapter (500 chars) — the list view's
    # fallback label before enrichment produces a real title.
    first_message: str
    # How many times the human actually typed something. Deliberately not "messages":
    # tool results, system reminders and compaction notices are not turns, so a row with
    # 0 turns is strong evidence nothing real happened here.
    turn_count: int
    title: Optional[str] = None           # human-friendly title (e.g. Claude ai-title)
    topics: Optional[str] = None          # JSON array string e.g. '["python","testing"]'
    # Name of the model the CLI was talking to, e.g. "claude-opus-5" — the key the cost
    # calculator matches against pricing.json.
    model_used: Optional[str] = None
    # Version of the CLI itself, when the transcript records it.
    cli_version: Optional[str] = None
    # Per-source extras (git branch, workspace name...). Not a column: nothing in the
    # pipeline may depend on it, so adapters can put anything useful here.
    metadata: dict = field(default_factory=dict)


@dataclass
class Turn:
    """One message in a session, flattened to plain text.

    Adapters do the flattening: a transcript message can be a list of typed blocks
    (text, the model's private "thinking", a tool request, a tool result), and consumers
    of Turn — search indexing, export, the bridge primer — only ever want readable text.
    `tool_calls` keeps a light record of what the assistant asked to run, as
    {"name": ..., "input": <short summary>} entries, not the full arguments.
    """
    role: Literal["user", "assistant"]
    content: str
    tool_calls: list[dict] = field(default_factory=list)


@dataclass
class ParsedSession:
    """A header plus every turn, in the order they happened.

    The expensive counterpart to a bare header: produced by parse_full() only for jobs
    that genuinely need the conversation text.
    """
    header: SessionHeader
    turns: list[Turn]


@runtime_checkable
class SessionSource(Protocol):
    """What every CLI adapter must implement.

    A Protocol is a structural interface: an adapter does not inherit from this class,
    it merely has to have these members. `@runtime_checkable` lets the tests assert
    `isinstance(adapter, SessionSource)` — a check on method NAMES only, not signatures,
    which is why the optional extras at the bottom are kept out of the Protocol body.

    Six required members, and the reason each exists:

      discover()            enumerate everything (backfill, reconcile, FTS, costs)
      parse_header()        cheap facts for a registry row (hook, watcher, backfill)
      session_id_for_path() identity WITHOUT opening the file (the watcher's delete path)
      parse_full()          the whole conversation (search, reasoning, export, enrich)
      resume_command()      how the user reopens this session in its own CLI
      is_available()        is there anything of this CLI on this machine at all
    """
    # Short key for this source, matching config.toml and sessions.cli_source.
    name: SourceName

    def discover(self) -> Iterator[Path]:
        """Yield transcript file paths for this CLI.

        One path per session, the canonical file only — never a symlink to it, or the
        same session would be indexed twice under two locations. An iterator rather than
        a list so a machine with thousands of sessions streams instead of materialising.
        Yields nothing if the CLI's directory does not exist.
        """
        ...

    def parse_header(self, path: Path) -> Optional[SessionHeader]:
        """Cheap metadata extraction — must NOT read the whole transcript.

        Cheap means: read the first and last handful of lines, not the middle. The
        watcher calls this every time an active transcript changes on disk.

        Returns None when `path` is not a real session of this CLI — unreadable, empty,
        a subagent sidechain, or a headless run this tool itself started. None means
        "skip silently"; it is not an error. This is the chokepoint every entry path
        shares (hook, watcher, backfill, enrichment), so the "is this a session?" rule
        has to be enforced here as well as in session_id_for_path().
        """
        ...

    def session_id_for_path(self, path: Path) -> Optional[str]:
        """Map a transcript file path to its session id WITHOUT reading the file
        (the file may already be deleted — used by the watcher's delete handler).
        Return None if the path isn't a session transcript for this source.

        Two hard requirements:
          * Pure path arithmetic. By the time the watcher asks, the file is gone.
          * Agreement with parse_header(): for any real transcript,
            `session_id_for_path(p) == parse_header(p).session_id`. Housekeeping
            archives rows where the two disagree, so a mismatch quietly loses sessions.

        It is also the watcher's gate for "is this path interesting at all" — the watcher
        walks each source tree recursively, so this must reject everything discover()
        would not have yielded (sidechains, nested journals, stray files).
        """
        ...

    def parse_full(self, path: Path) -> Optional[ParsedSession]:
        """Full transcript parse into ordered turns.

        Reads the entire file, so callers use it only when they need the text: full-text
        indexing, reasoning extraction, export, the enrichment prompt. Returns None on
        the same "not a session" grounds as parse_header().
        """
        ...

    def resume_command(self, session_id: str) -> str:
        """The CLI's own command to resume this session in that CLI.

        A shell-quoted string such as `claude --resume 6550180f-…`, shown in the UI and
        used by the `cr` helper. Built as text, never executed here.
        """
        ...

    def is_available(self) -> bool:
        """True if there are transcripts to read (the session directory / DB /
        mirror exists). Deliberately NOT "the binary is on PATH": indexing and
        watching must keep working after a CLI is uninstalled or falls off a
        daemon's PATH — those transcripts are what this tool keeps."""
        ...

    # Optional — NOT Protocol members, so adapters without them still satisfy
    # isinstance(). Looked up with getattr():
    #
    #   def has_binary(self) -> bool:
    #       """Whether the CLI is runnable from here — a hint for resume/bridge
    #       and `sb doctor` only; never consulted by indexing."""
    #
    #   def restore_path(self, row) -> Optional[Path]:
    #       """Where a restored transcript for this `sessions` row must be
    #       written so the CLI's own resume finds it, or None if this source
    #       can't be restored from a bare <session_id>.jsonl raw copy."""
    #
    #   def watch_roots(self) -> list:
    #       """Directories the watcher should subscribe to, for an adapter that
    #       owns more than one (Codex keeps live and archived rollouts in
    #       sibling trees). Each entry is either a path or a (path, recursive)
    #       tuple; without this method the watcher falls back to the adapter's
    #       single configured directory, watched recursively. Say
    #       recursive=False for a big tree where only one file matters — every
    #       subdirectory costs a kernel watch, and Linux runs out of them."""
    #
    #   def sync_trigger(self, path: Path) -> bool:
    #       """For a source that does not store one file per session: True when
    #       `path` is a write to the underlying store (OpenCode's SQLite
    #       database or its write-ahead log) that should make the adapter
    #       re-project its mirror files. The watcher then calls sync()."""
