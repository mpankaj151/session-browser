"""OpenAI Codex CLI source adapter.

In plain words: the `codex` command-line assistant writes a log of every conversation it
has with you; this module finds those logs and pulls out the handful of facts the rest of
Session Browser stores (who, where, when, how many exchanges, which model). Nothing else
in this repo knows Codex exists — see sources/base.py for the small protocol every
adapter implements and docs/ARCHITECTURE.md for the pipeline picture.

Vocabulary, defined once here; docs/GLOSSARY.md has the rest. A **session** is one
conversation with the CLI, from the first prompt to the last reply. Its **transcript** is
the file the CLI wrote while that happened — Codex's own name for one is a **rollout**. A
**turn** is one user message plus the assistant reply it drew. A **model** is the AI
system that writes those replies; Codex records which one a session used (e.g. "gpt-5.5").

Who calls this: sources/registry.py builds one CodexSource (pointed at $CODEX_HOME or the
paths in config.toml); indexer.py, watcher.py, backfill.py, the MCP server and the `cr`
shell helper then go through the SessionSource protocol. Everything here is READ-ONLY: no
file under ~/.codex is written, and the `codex` binary is never executed (has_binary()
is a UI hint for the Resume button, nothing more). reasoning.py and
scripts/compute-costs.py import open_rollout()/ROLLOUT_ERRORS from here so that rollout
I/O lives in exactly one place.

Sessions: ~/.codex/sessions/YYYY/MM/DD/rollout-<ts>-<uuid>.jsonl
Archived: ~/.codex/archived_sessions/YYYY/MM/DD/<same>   (`codex archive` MOVES)
Each line is {"type": ..., "payload": {...}} (or a top-level record).

Two on-disk dialects, both live:

  legacy    - payload.type "user_message"/"agent_message" carry payload.message.
  paginated - Codex >=~0.144 starts every non-ephemeral CLI thread in
              `paginated` history mode (codex-rs/tui/src/app_server_session.rs
              and codex-rs/exec/src/lib.rs both set
              `history_mode: (!ephemeral).then_some(Paginated)`). In that mode
              codex-rs/rollout/src/policy.rs persists the legacy events ONLY
              when the mode is Legacy; the same content arrives as
              "item_completed" wrapping a TurnItem instead.

One real line of each, so the difference is concrete (indented here for reading; on disk
each is a single line):

  legacy     {"timestamp": "2026-08-01T10:00:00Z", "type": "event_msg",
              "payload": {"type": "user_message", "message": "why is codex missing?"}}

  paginated  {"timestamp": "2026-08-01T10:00:00Z", "type": "event_msg",
              "payload": {"type": "item_completed", "thread_id": "019e18fa-0d21-...",
                          "turn_id": "t1",
                          "item": {"type": "UserMessage", "id": "u0",
                                   "content": [{"type": "text",
                                                "text": "why is codex missing?"}]}}}

Both dialects have to keep working: a laptop holds months of old legacy rollouts beside
today's paginated ones. Reading only the legacy shape is exactly what made every NEW
Codex session disappear from the browser for weeks — the turn count came out 0, and 0
turns used to mean "nothing to browse, skip it". tests/test_smoke.py pins both dialects
(test_codex_paginated_rollout_parses_turns / test_codex_legacy_rollout_still_parses).

Rollouts older than 7 days are zstd-compressed in place to `<name>.jsonl.zst`
by the worker spawned from codex-rs/core/src/thread_manager.rs, so every read
path has to handle both representations. zstd is an ordinary compression format (like
gzip); open_rollout() below hides which representation is on disk from every caller.

Other records, unchanged across dialects:
  - first line: type "session_meta", payload {id, timestamp, cwd, cli_version,
    model_provider, history_mode}
  - model: type "turn_context", payload.model (e.g. "gpt-5.5")
  - reasoning: payload.type "agent_reasoning"/"agent_reasoning_raw_content"
    (the model's own "thinking" text — scripts/extract-reasoning.py turns it into a
    readable trail)
  - title: payload.type "thread_name_updated" (also ~/.codex/session_index.jsonl)
  - tokens: payload.type "token_count", payload.info.total_token_usage (cumulative)
    — a *token* is the word-piece unit models read and write in, and what vendors bill
    by; scripts/compute-costs.py turns these counts into a dollar figure
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import shlex
import shutil
import sys
from pathlib import Path
from typing import Iterator, Optional

from .base import ParsedSession, SessionHeader, Turn, to_iso_utc

# Codex's default transcript root. Only a fallback: sources/registry.py resolves
# $CODEX_HOME and the [sources.codex] config block and passes the real paths to
# __init__, so a machine with several Codex homes still works.
SESSIONS_DIR = Path(os.path.expanduser("~/.codex/sessions"))

# Filename pieces of `rollout-2026-08-01T10-00-00-<uuid>.jsonl[.zst]`, spelled once
# because discovery, identity and restore all have to agree on them.
_JSONL = ".jsonl"
_ZST = ".zst"
_PREFIX = "rollout-"
# "2026-08-01T10-00-00" — the fixed-width stamp codex-rs/rollout/src/
# rollout_file_name.rs writes between the prefix and the ids. 19 characters:
# 10 for the date, "T", 8 for the time with dashes instead of colons (a colon is
# not safe in a filename). session_id_for_path() counts on that width.
_TS_LEN = 19

# parse_header streams the whole rollout (last_activity lives on the final
# line), so memoize by (size, mtime): the watcher re-parses on a 500ms debounce
# and would otherwise pin a core on an active multi-MB session.
#
# Shape: {path: ((size in bytes, mtime in nanoseconds), header or None)}. Size plus
# mtime is a cheap "has this file changed?" test — a rollout only ever grows, and a
# nanosecond mtime does not collide the way a second-resolution one would. A cached
# None is meaningful ("this file has nothing browsable in it"), which is what keeps
# aborted rollouts from being re-read on every pass. The cache is per process and is
# emptied wholesale past 4096 entries (see parse_header) rather than evicted one by
# one: a long-lived watcher must not grow without bound, and a full re-read costs
# one pass.
_HDR_CACHE: dict[str, tuple[tuple[int, int], "SessionHeader | None"]] = {}

# Record types this adapter understands. Used only to tell "this session has no
# user turns" apart from "this file is in a dialect I have never seen" — the
# distinction whose absence hid the paginated-rollout outage for weeks.
#
# Read it as a recognition test, NOT a filter: nothing is dropped for being absent
# from this set. parse_header counts how many of a file's records appear here; if a
# rollout has records but NONE of them are recognised, the file is in some future
# Codex dialect and gets indexed with 0 turns plus a loud warning (see
# _warn_unknown_dialect) instead of being silently skipped. The entries below are
# simply every payload.type and TurnItem kind seen across Codex 0.1x–0.15x: session
# bookkeeping (session_meta, turn_context, turn_started/complete/aborted, task_*),
# conversation (user_message/agent_message and their paginated item equivalents),
# tool activity (function_call, local_shell_call, web_search_call, patch_apply_end,
# mcp_tool_call_end ...), accounting (token_count) and compaction.
_KNOWN_TYPES = frozenset({
    "session_meta", "turn_context", "user_message", "agent_message",
    "item_completed", "item_started", "token_count", "task_started",
    "task_complete", "turn_started", "turn_complete", "turn_aborted",
    "thread_name_updated", "agent_reasoning", "agent_reasoning_raw_content",
    "message", "reasoning", "function_call", "function_call_output",
    "custom_tool_call", "custom_tool_call_output", "local_shell_call",
    "web_search_call", "web_search_end", "patch_apply_end", "mcp_tool_call_end",
    "compacted", "context_compacted", "world_state", "security_risk_score",
})

# Which paginated TurnItem kinds are conversation, and whose voice they are.
# TurnItem tags are PascalCase (codex-rs/protocol/src/items.rs has no
# rename_all on the enum); matched case-insensitively for safety — the keys here
# are therefore lower-cased ("UserMessage" -> "usermessage"). Items not listed
# (Reasoning, CommandExecution, FileChange, WebSearch, ...) are not turns.
_ITEM_ROLES = {"usermessage": "user", "agentmessage": "assistant"}

# Session ids already warned about, so an unknown dialect produces ONE stderr line
# per session per process instead of one per re-parse (the watcher re-parses on
# every file change). Process-local and deliberately never persisted.
_WARNED: set[str] = set()


def _payload(rec: dict) -> tuple[str, dict]:
    """(record type, payload dict). Codex nests real data under 'payload'.

    A rollout line is usually an envelope — {"type": "event_msg", "payload": {"type":
    "user_message", ...}} — where the OUTER type names the transport and the inner one
    names the event. A few records (older builds, session_meta on some versions) are flat
    instead. Returning the pair lets every caller ask one question ("what kind of record
    is this, and where are its fields?") without caring which shape it got. The type is
    "" when neither level names one; no exception is ever raised here.
    """
    p = rec.get("payload")
    if isinstance(p, dict):
        return p.get("type") or rec.get("type") or "", p
    return rec.get("type") or "", rec


def _rollout_errors() -> tuple:
    """What a reader must tolerate: I/O, a truncated frame (ValueError from the
    text wrapper) and zstandard's own ZstdError — which is NOT a ValueError.

    Built once at import time into ROLLOUT_ERRORS so callers can write
    `except ROLLOUT_ERRORS`. Reading a rollout goes wrong in ways that are normal, not
    exceptional: the file is being written right now, or Codex was killed mid-compression
    and left half a zstd frame. All of those mean "this file is not a readable session",
    never "crash the watcher". The zstandard import is optional because a laptop with no
    compressed rollouts need not have the library installed.
    """
    errs: tuple = (OSError, ValueError)
    try:
        import zstandard
        errs += (zstandard.ZstdError,)
    except ImportError:
        pass
    return errs


# Public: scripts/compute-costs.py catches these around its own open_rollout() loop.
ROLLOUT_ERRORS = _rollout_errors()


@contextlib.contextmanager
def open_rollout(path: Path):
    """Line reader for a rollout, plain `.jsonl` or zstd `.jsonl.zst`.

    Mirrors codex_rollout::open_rollout_line_reader: callers should not need to
    know which representation is currently on disk. Public because every
    consumer of a rollout (reasoning, costs, full-text) must go through it —
    a plain open() reads a zstd frame as garbage and silently yields nothing.

    Used as `with open_rollout(path) as fh: for line in fh: ...`; the yielded object is a
    text file you iterate a line at a time, so a multi-megabyte session is never held in
    memory whole. Decoding errors are replaced rather than raised (errors="replace"): one
    bad byte in a long transcript must not cost the whole session. Raises the usual
    filesystem errors if the file is missing or unreadable — see ROLLOUT_ERRORS.
    """
    if path.name.endswith(_ZST):
        import zstandard  # imported lazily: only compressed rollouts need it
        with open(path, "rb") as raw:
            with zstandard.ZstdDecompressor().stream_reader(raw) as reader:
                yield io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
    else:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            yield fh


def _item_text(item: dict) -> str:
    """Flatten a paginated TurnItem's content[] to plain text.

    A paginated item carries its text as a list of chunks, e.g.
    "content": [{"type": "text", "text": "why is codex missing?"}]. Non-text chunks
    (images, file references) are skipped, the text ones are joined, and the result is
    stripped; "" means "this item said nothing quotable".

    Codex tags the chunks inconsistently — UserInput is snake_case
    ({"type": "text"}), AgentMessageContent is PascalCase ({"type": "Text"}) —
    so the type is compared case-insensitively.

    Also imported by reasoning.py, which flattens Reasoning items the same way.
    """
    parts = []
    for chunk in item.get("content") or []:
        if isinstance(chunk, dict) and str(chunk.get("type", "")).lower() == "text":
            text = chunk.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts).strip()


def _turn_of(pt: str, p: dict) -> Optional[tuple[str, str]]:
    """(role, content) for any record carrying a message, in either dialect.

    `pt` is the record type and `p` the payload, exactly as _payload() returned them.
    This is the one place that knows both dialects, which is why parse_header and
    parse_full stay dialect-agnostic.

    Returns None for every other record. The two dialects never overlap: policy
    .rs persists the legacy events only in Legacy mode and UserMessage/
    AgentMessage items only in Paginated, so a turn is never counted twice.

    Empty messages return None too, so a blank prompt never inflates turn_count.
    """
    # Legacy dialect: the text sits directly on payload.message.
    if pt in ("user_message", "agent_message"):
        message = p.get("message")
        if isinstance(message, str) and message.strip():
            return ("user" if pt == "user_message" else "assistant", message.strip())
        return None
    # Paginated dialect: payload.item is a TurnItem whose kind gives the role and
    # whose content[] holds the text. Kinds that are not conversation fall through.
    if pt == "item_completed":
        item = p.get("item")
        if isinstance(item, dict):
            role = _ITEM_ROLES.get(str(item.get("type", "")).lower())
            if role:
                text = _item_text(item)
                if text:
                    return (role, text)
    return None


def _warn_unknown_dialect(path: Path, session_id: str) -> None:
    """Surface schema drift loudly ONCE per session.

    Writes one line to stderr (which launchd/systemd capture into the watcher's log) and
    remembers the session id in _WARNED so re-parses stay quiet. Returns nothing; the
    caller carries on and indexes the session with 0 turns.

    The paginated switch was invisible because an unparseable rollout and an
    empty one both returned None, and the watcher drops None without logging.
    """
    if session_id in _WARNED:
        return
    _WARNED.add(session_id)
    print(f"[codex] unrecognised rollout dialect in {path.name} — indexing with "
          f"0 turns; the adapter likely needs updating for a new Codex format",
          file=sys.stderr, flush=True)


class CodexSource:
    """The Codex adapter: one instance per configured Codex home.

    Implements the SessionSource protocol from sources/base.py (discover, parse_header,
    parse_full, session_id_for_path, resume_command, is_available) plus the optional
    extras the rest of the tool looks up with getattr(): watch_roots(), has_binary() and
    restore_path().

    It holds two directories — the live `sessions/` tree and the sibling
    `archived_sessions/` tree that `codex archive` MOVES rollouts into. Both are the
    user's work, so both are discovered and watched; "archived" in Codex's sense has
    nothing to do with this tool's own archived rows (docs/GLOSSARY.md).

    Instances are cheap and stateless apart from the module-level header cache, so tests
    construct a fresh one per temporary directory.
    """

    # Identifies rows from this adapter in the registry's cli_source column.
    name = "codex"

    def __init__(self, sessions_dir: Path | str = SESSIONS_DIR,
                 archived_dir: Path | str | None = None):
        """Point the adapter at one Codex home. No disk access: every adapter is
        constructed on start-up, including on machines that have no Codex at all.

        `archived_dir` defaults to the `archived_sessions` directory beside `sessions_dir`
        — the layout Codex itself uses — so callers normally pass only the first path.
        `~` is expanded here so config values like "~/.codex/sessions" just work.
        """
        self.sessions_dir = Path(os.path.expanduser(str(sessions_dir)))
        self.archived_dir = (
            Path(os.path.expanduser(str(archived_dir))) if archived_dir
            else self.sessions_dir.parent / "archived_sessions"
        )

    # -- discovery -------------------------------------------------------------
    def watch_roots(self) -> list[Path]:
        """Every directory the watcher must subscribe to.

        watcher.py subscribes to each path returned here (recursively) and re-indexes on
        every create/modify, archiving the row on delete.

        `codex archive` MOVES a rollout into the sibling archived_sessions tree.
        Watching only sessions/ made that move look like a deletion — the session was
        still on disk, but its row dropped out of the browser.

        Paths that do not exist yet are fine: the watcher skips them, and discover()
        below checks again on every pass.
        """
        return [self.sessions_dir, self.archived_dir]

    def discover(self) -> Iterator[Path]:
        """Yield every rollout file under both trees, newest-first order not guaranteed.

        Callers: backfill.py (index everything), prune/reconcile, `sb doctor`. Yields
        lazily, so a tree with thousands of rollouts is never listed into memory at once.

        The two glob patterns are the same shape twice, once per representation:
        `*/*/*/rollout-*.jsonl` and `*/*/*/rollout-*.jsonl.zst`. The three `*/` segments
        are Codex's YYYY/MM/DD directories, so the glob cannot wander outside the date
        layout. Symlinks are skipped: following one could yield the same session twice,
        or escape the tree entirely.
        """
        for root in self.watch_roots():
            if not root.exists():
                continue
            for pattern in (f"*/*/*/{_PREFIX}*{_JSONL}", f"*/*/*/{_PREFIX}*{_JSONL}{_ZST}"):
                for p in root.glob(pattern):
                    if not p.is_symlink():
                        yield p

    def session_id_for_path(self, path: Path) -> Optional[str]:
        """Thread id from the filename alone — the file may already be gone.

        This is what the watcher's delete handler calls when a file disappears: it has
        only the path, so identity has to come from the NAME. Returns None for anything
        that is not a Codex rollout (a stray notes.txt, a half-written temp file), which
        tells the caller "not mine, ignore it".

        Mirrors codex-rs/rollout/src/rollout_file_name.rs: `rollout-` + a
        19-char timestamp + `-` + ids, where a reverted thread appends
        `_<rollout_id>` after the stable thread id. Anchoring on the timestamp
        width keeps both that shape and `.jsonl.zst` correct, which the old
        "last five dash-separated groups" heuristic did not.

        Worked example, step by step, for
        `rollout-2026-08-01T10-00-00-019e18fa-0d21-7461-922c-5ccaad36df05.jsonl.zst`:
          1. drop a trailing `.zst`, so compressed and plain names resolve alike;
          2. require the `rollout-` prefix and the `.jsonl` suffix;
          3. `core` is what is left: "2026-08-01T10-00-00-019e18fa-...-5ccaad36df05";
          4. character 19 of `core` must be the dash after the fixed-width stamp;
          5. everything after it is the id, and a `_<rollout_id>` suffix (added when a
             thread is reverted) is cut off so both files map to the same session.
        The counting matters because the timestamp itself contains dashes, which is why
        splitting on "-" could never work.
        """
        name = path.name
        if name.endswith(_ZST):
            name = name[: -len(_ZST)]
        if not (name.startswith(_PREFIX) and name.endswith(_JSONL)):
            return None
        core = name[len(_PREFIX):-len(_JSONL)]
        if len(core) <= _TS_LEN or core[_TS_LEN] != "-":
            return None
        return core[_TS_LEN + 1:].split("_", 1)[0] or None

    # -- cheap header ----------------------------------------------------------
    def parse_header(self, path: Path) -> Optional[SessionHeader]:
        """The facts one registry row needs, from one pass over the rollout.

        Returns a SessionHeader (see sources/base.py) or None when the file is not a
        browsable session — unreadable, corrupt, or a rollout with no user turns at all
        (an aborted `codex` launch). The indexer treats None as "skip this file", so
        returning None must always mean "there is genuinely nothing here"; a file in a
        dialect this adapter does not understand is indexed with turn_count 0 and a
        warning instead. See _warn_unknown_dialect for why that distinction exists.

        Unlike the other adapters this one reads the WHOLE file even though the protocol
        calls parse_header "cheap": last_activity is the timestamp on the final record,
        and Codex writes no footer. _HDR_CACHE makes the repeated reads bearable.

        Never raises: every filesystem, decompression and JSON failure is turned into
        None or into skipping one line.
        """
        # stat() first: it is also the cache key, so a file that has not changed since
        # the last parse costs one syscall instead of a full read.
        try:
            st = path.stat()
        except OSError:
            return None
        cache_key = (st.st_size, st.st_mtime_ns)
        hit = _HDR_CACHE.get(str(path))
        if hit is not None and hit[0] == cache_key:
            return hit[1]
        # Fall back to the bare filename if the name is not a rollout shape; session_meta
        # below normally overrides this with the id Codex itself recorded.
        session_id = self.session_id_for_path(path) or path.stem
        cwd = ""
        start_time = ""
        version = None
        model = None
        title = None
        first_message = ""
        last_ts = ""
        turn_count = 0
        events = 0        # records that are not the session_meta preamble
        recognised = 0    # ...of those, ones this adapter knows
        try:
            with open_rollout(path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        # One unparseable line (the session is being appended to right
                        # now) must not cost the other thousand.
                        continue
                    # Every record is stamped, so the LAST stamp seen is the session's
                    # last activity. Overwritten on each line on purpose.
                    ts = rec.get("timestamp") or ""
                    if ts:
                        last_ts = ts
                    pt, p = _payload(rec)
                    # The preamble Codex writes first: id, working directory, start time
                    # and CLI version. Checked at both nesting levels because older
                    # builds wrote it flat. `or`-guards keep the FIRST value seen, since
                    # a resumed session can carry a second session_meta.
                    if pt == "session_meta" or rec.get("type") == "session_meta":
                        session_id = p.get("id") or session_id
                        cwd = cwd or p.get("cwd", "")
                        start_time = start_time or p.get("timestamp") or ts
                        version = version or p.get("cli_version")
                        continue
                    # Everything past the preamble is counted, and separately counted if
                    # this adapter recognises it. Those two numbers are the whole basis
                    # of the unknown-dialect decision made after the loop.
                    events += 1
                    if pt in _KNOWN_TYPES:
                        recognised += 1
                    # turn_context is emitted whenever the model or settings change; the
                    # first one names the model the session ran on (e.g. "gpt-5.5") and
                    # carries a cwd for rollouts whose session_meta had none.
                    if pt == "turn_context":
                        model = model or p.get("model")
                        cwd = cwd or p.get("cwd", "")
                    elif pt == "thread_name_updated":
                        # Codex's own short name for the thread; the UI prefers it over
                        # the first message. Last one wins — it can be renamed.
                        title = p.get("name") or p.get("thread_name") or title
                    # _turn_of understands both dialects, so nothing here has to.
                    # Only USER turns are counted: turn_count means "how many times did
                    # the human speak", and the first of them is the session's headline.
                    turn = _turn_of(pt, p)
                    if turn and turn[0] == "user":
                        turn_count += 1
                        first_message = first_message or turn[1]
        except ROLLOUT_ERRORS:
            # a truncated or corrupt zstd frame is "not a session", never a traceback
            return None

        # Bound the cache before inserting below. Clearing it whole is deliberate: the
        # alternative (tracking recency) costs more than re-reading, and the watcher's
        # working set is "the few sessions touched today" anyway.
        if len(_HDR_CACHE) > 4096:
            _HDR_CACHE.clear()
        if turn_count == 0 and not first_message:
            # Nothing the user said was found. Two very different reasons, and telling
            # them apart is the point of the `events`/`recognised` counters above.
            if events and not recognised:
                # A dialect we do not know. Index it anyway so the session is
                # visible and the warning is actionable — silently returning
                # None here is exactly how the paginated switch went unnoticed.
                _warn_unknown_dialect(path, session_id)
            else:
                # A rollout with no user turns (aborted / meta-only) — not
                # browsable. Cached too: these files are exactly the ones
                # re-scanned pointlessly.
                _HDR_CACHE[str(path)] = (cache_key, None)
                return None

        # Short label for the UI: the last component of the project directory
        # ("/Users/x/proj" -> "proj"). With no cwd recorded, the containing date
        # directory is at least something stable to group by.
        folder = Path(cwd).name if cwd else path.parent.name
        # first_message is capped at 500 characters because the registry column is a
        # preview, not the transcript; to_iso_utc normalises Codex's timestamps into the
        # one UTC spelling every source shares, so mixed lists sort correctly.
        # metadata is the per-source extras bag: it has no database column (see
        # indexer._params) and is only read in-process. This one records the on-disk
        # filename, including any .zst, which the row's project_path does not.
        header = SessionHeader(
            session_id=session_id,
            cli_source=self.name,
            project_path=str(path.parent),
            cwd=cwd,
            folder_name=folder,
            start_time=to_iso_utc(start_time),
            last_activity=to_iso_utc(last_ts or start_time),
            first_message=first_message[:500],
            turn_count=turn_count,
            title=title,
            model_used=model,
            cli_version=version,
            metadata={"transcript": path.name},
        )
        _HDR_CACHE[str(path)] = (cache_key, header)
        return header

    # -- full parse ------------------------------------------------------------
    def parse_full(self, path: Path) -> Optional[ParsedSession]:
        """Header plus every message body, in the order they were written.

        Used by the consumers that need the conversation itself rather than a summary:
        the full-text index, the reasoning/decision trail, the enrichment prompt, Copy
        Context and Export. A second pass over the file is cheap next to what those do
        with the result, and it keeps parse_header's fast path free of turn storage.

        Returns None when the header does (nothing browsable) or when reading fails
        part-way. tool_calls stay empty here: Codex records tool activity as separate
        records, and no consumer of this adapter asks for them.
        """
        header = self.parse_header(path)
        if header is None:
            return None
        turns: list[Turn] = []
        try:
            with open_rollout(path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    # Both roles this time, not just the user's, so the trail reads as a
                    # conversation. Dialect handling lives entirely in _turn_of.
                    turn = _turn_of(*_payload(rec))
                    if turn:
                        turns.append(Turn(role=turn[0], content=turn[1]))
        except ROLLOUT_ERRORS:
            return None
        return ParsedSession(header=header, turns=turns)

    # -- resume / availability -------------------------------------------------
    def resume_command(self, session_id: str) -> str:
        """The command that reopens this session in Codex with its history intact.

        Shown by the UI and run by the `cr` shell helper (bin/resume-here.sh). shlex.quote
        keeps an odd id from being split or interpreted by the shell; nothing is executed
        here, the string is only handed back.
        """
        return f"codex resume {shlex.quote(session_id)}"

    # -- restore / locate --------------------------------------------------------
    def _contained(self, path: Path) -> bool:
        """True when `path` really is inside one of this adapter's two trees.

        The guard behind restore: a registry row carried over from another laptop can
        record any directory at all, and restore writes a file at whatever path it is
        given. relative_to() raises ValueError when the path is outside the root, which
        is the containment test; resolve() is what makes `..` and symlinks unable to
        walk out. OSError (a broken symlink, a vanished directory) counts as "not
        contained" — refusing is always the safe answer here.
        """
        for root in (self.sessions_dir, self.archived_dir):
            try:
                path.resolve().relative_to(root.resolve())
                return True
            except (ValueError, OSError):
                continue
        return False

    def restore_path(self, row) -> Optional[Path]:
        """Where this row's rollout lives — or, for restore, must be written.

        Codex names files rollout-<19-char ts>-<id>.jsonl inside a date dir
        and may zstd-compress or `codex archive` them; the row records the dir
        (project_path) but not the name. Look the file up (row dir first, then
        both trees), else synthesise the canonical name from start_time so a
        restored copy is exactly what discover()/session_id_for_path expect.
        Refuses anything outside the codex trees, like ClaudeSource.

        `row` is one `sessions` row (a sqlite3.Row, or anything that indexes by column
        name); only session_id, project_path and start_time are read. Nothing is written
        and no directory is created — restore.py does that with the path returned here.
        None means "this row cannot be restored by this adapter", which is what makes the
        UI show the truth instead of a Restore button that fails."""
        sid = row["session_id"]
        project_path = row["project_path"] or ""
        day = Path(os.path.expanduser(project_path)) if project_path else None
        if day is not None and not self._contained(day):
            return None
        # Both representations of "the rollout for this id", e.g.
        # `rollout-*<id>.jsonl` and `rollout-*<id>.jsonl.zst`. The `*` swallows the
        # timestamp, which the row does not record.
        patterns = (f"{_PREFIX}*{sid}{_JSONL}", f"{_PREFIX}*{sid}{_JSONL}{_ZST}")
        found: list[Path] = []
        # Cheapest first: the date directory the row itself recorded.
        if day is not None and day.is_dir():
            for pat in patterns:
                found += sorted(day.glob(pat))
        # Otherwise sweep both trees. `codex archive` may have moved the file, or the
        # row's directory may predate a Codex-home change.
        if not found:
            for root in (self.sessions_dir, self.archived_dir):
                if root.exists():
                    for pat in patterns:
                        found += sorted(root.glob(f"*/*/*/{pat}"))
        # A glob on "*<id>.jsonl" can also match a file whose id merely ENDS with this
        # one, so confirm identity the same way the watcher does.
        for candidate in found:
            if self.session_id_for_path(candidate) == sid:
                return candidate
        # Nothing on disk: build the canonical name Codex would have used. sqlite3.Row
        # raises IndexError for a column the query did not select, so start_time is
        # optional.
        try:
            start = row["start_time"]
        except (KeyError, IndexError):
            start = ""
        # "2026-08-01T10:00:00.123Z"[:19] -> "2026-08-01T10:00:00", then colons become
        # dashes: exactly the 19-character stamp rollout_file_name.rs writes.
        stamp = (start or "")[:19].replace(":", "-")
        # An empty or malformed start_time would produce a name session_id_for_path
        # rejects, so fall back to now — the file still has to be findable.
        if len(stamp) != _TS_LEN:
            from datetime import datetime, timezone
            stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        # Slice the stamp back into Codex's YYYY/MM/DD directory layout.
        if day is None:
            day = self.sessions_dir / stamp[0:4] / stamp[5:7] / stamp[8:10]
        # The plain .jsonl name. restore.py appends `.zst` itself when the vault copy it
        # is about to write is a zstd frame, so the representation is never mismatched
        # against the name.
        return day / f"{_PREFIX}{stamp}-{sid}{_JSONL}"

    def has_binary(self) -> bool:
        """Whether `codex` is runnable from here — a UI hint for resume, only.

        Never consulted by indexing (see is_available). `sb doctor` and the Resume/Bridge
        buttons use it to say "installed" or to grey a button; a False answer must never
        stop a single transcript from being read.
        """
        return shutil.which("codex") is not None

    def is_available(self) -> bool:
        """Whether there are transcripts to read.

        This is the question the watcher and the indexer ask, and the answer is about
        FILES, not about software being installed: an uninstalled Codex leaves months of
        readable history behind, and keeping it browsable is the whole point of this tool.

        Deliberately NOT gated on `which codex`: the unified ChatGPT/Codex app
        moves the binary out of the PATH the watcher runs under, and gating a
        filesystem watcher on that silently unsubscribed ~/.codex/sessions with
        no log line at all. A test in tests/test_smoke.py empties $PATH and asserts this
        still answers True (test_codex_available_without_binary_on_path).
        """
        return self.sessions_dir.exists() or self.archived_dir.exists()


# open_rollout used to be private. reasoning.py and scripts/compute-costs.py now import
# the public name; this alias keeps an older checkout or a stray import working.
_open_rollout = open_rollout  # backwards-compatible alias
