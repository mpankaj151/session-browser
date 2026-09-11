"""Claude Code source adapter.

Transcripts: ~/.claude/projects/<encoded-cwd>/<session-uuid>.jsonl
Newer-format aware (verified against real transcripts):
  - substantive user turn = type=="user" && promptSource=="typed" && isMeta!=true
  - free title from `ai-title` records (aiTitle field)
  - model at assistant.message.model; usage at assistant.message.usage
  - reasoning lives in assistant.message.content blocks of type "thinking"

--- On-disk layout -----------------------------------------------------------------

Claude Code stores one *transcript* (the file recording one conversation — see
docs/GLOSSARY.md) per session, in a directory named after the working directory the
session ran in, with "/" replaced by "-":

    ~/.claude/projects/-Users-me-Documents-app/6550180f-14ff-4b91-a93d-d951ed98c2f7.jsonl
                       └── encoded cwd ─────┘  └── session id (a UUID) ───────────┘

That two-level shape is the identity rule: the file's stem IS the session id, and its
parent's parent must be the projects directory. Anything deeper is not a session. In
particular a session that spawns helper conversations ("subagents", also called
sidechains) writes them underneath its own id:

    <project>/<session-uuid>/subagents/agent-<id>.jsonl
    <project>/<session-uuid>/subagents/workflows/<wf-id>/{agent-<id>,journal}.jsonl

Those belong to the parent session and must never become rows of their own — every
record in them is flagged isSidechain, so they would index as permanently empty
sessions, and worse, every workflow journal has the stem "journal", so they would all
collide onto one bogus row. Both session_id_for_path() and parse_header() reject them.

--- JSONL line shapes this module reads ---------------------------------------------

The file is JSON Lines: one JSON object per line, appended as the session goes on. Only
three `type`s matter here (fields trimmed for readability, real lines are much longer):

  a typed user message — the only thing counted as a *turn*:
    {"type":"user","promptSource":"typed","isSidechain":false,"isMeta":false,
     "cwd":"/Users/me/app","version":"2.1.207","gitBranch":"main",
     "timestamp":"2026-07-13T00:36:10.534Z","message":{"role":"user",
     "content":"verify if mcp is working"}}

  an assistant reply — a LIST of typed blocks, which is why message.content needs
  flattening; `usage` is where the token counts live (tokens are the word-pieces a model
  reads and writes; usage is what a session is billed by):
    {"type":"assistant","timestamp":"...","message":{"model":"claude-opus-5",
     "usage":{"input_tokens":12,"output_tokens":340,"cache_read_input_tokens":91000},
     "content":[{"type":"thinking","thinking":"the hook never fired..."},
                {"type":"text","text":"I'll check the logs."},
                {"type":"tool_use","name":"Bash","input":{"command":"ls"}}]}}

  a title Claude wrote for the session, appearing (and being rewritten) as it goes:
    {"type":"ai-title","aiTitle":"Review solution for gaps","sessionId":"6550180f-…"}

Other lines (mode, permission-mode, last-prompt, hook attachments, tool results,
compaction notices) are simply ignored — this parser looks for what it needs and skips
the rest, which is also what makes it tolerant of format changes.

--- Reading strategy ----------------------------------------------------------------

Transcripts routinely reach tens of megabytes, and the watcher re-parses the header of
an ACTIVE one every half second. So parse_header() reads only the first 60 and last 60
lines, falls back to a wider scan only when the first typed message was not in that
window, and memoises the turn count by (size, mtime). parse_full() is the deliberate
exception: it streams the whole file, and only the jobs that need the text call it.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
from itertools import islice
from pathlib import Path
from typing import Iterator, Optional

from .base import ParsedSession, SessionHeader, Turn, to_iso_utc

# The documented default; sources/registry.py overrides it from config.toml or from
# CLAUDE_CONFIG_DIR, so nothing outside the tests should rely on this constant.
PROJECTS_DIR = Path(os.path.expanduser("~/.claude/projects"))

# Matches a line of machinery that Claude Code wraps around a message but that a human
# never typed, so it can be stripped before the text is stored or searched. It fires on a
# line whose first non-whitespace characters are one of these XML-ish opening tags:
#   "<command-name>/init</command-name>"        -> stripped (a slash command)
#   "<system-reminder>Skill X is available"     -> stripped (injected instructions)
#   "<bash-stdout>total 24"                     -> stripped (output of a ! shell line)
#   "why is the build failing?"                 -> kept (no tag, a real question)
# Anchored with ^ and used with .match(), so a "<command-name>" appearing mid-sentence
# does not cost the line.
_CMD_LINE = re.compile(r"^\s*<(command-name|command-message|command-args|local-command|"
                       r"bash-input|bash-stdout|bash-stderr|system-reminder)")
# A cheap substring test applied to the raw line BEFORE json.loads(). Most lines in a big
# transcript are assistant replies and tool results; skipping the JSON parse for those is
# what makes counting turns over a 50 MB file affordable. It relies on Claude Code
# writing compact JSON with no space after the colon — if that ever changes, the count
# silently drops to 0, which is why turn counts are also sanity-checked by the tests.
_USER_MARKER = '"type":"user"'
# Subdir a session's multi-agent sidechain transcripts live under; never a session.
_SUBAGENT_DIR = "subagents"
# Claude encodes a cwd as one project dir name ("/" -> "-"); enrichment runs
# from <data>/enrichment-cwd, so its transcripts always sit in a dir with this tail.
_ENRICH_CWD_SUFFIX = "-enrichment-cwd"


class ClaudeSource:
    """Adapter implementing sources/base.py's SessionSource protocol for Claude Code.

    Holds no state beyond its projects directory, so it is cheap to construct and safe to
    build fresh per call (which every caller does). The one exception is the module-level
    turn-count cache below, which is shared across instances on purpose — the long-lived
    watcher benefits from it, and it is keyed by path plus file identity, so a stale entry
    is impossible.
    """
    name = "claude"

    def __init__(self, projects_dir: Path | str = PROJECTS_DIR):
        """`projects_dir` is where Claude Code keeps transcripts; the tests pass a
        temporary directory. Accepts a str so config values need no conversion, and
        expands a leading `~` here rather than making every caller remember to."""
        self.projects_dir = Path(os.path.expanduser(str(projects_dir)))

    # -- discovery -------------------------------------------------------------
    def discover(self) -> Iterator[Path]:
        """Yield every Claude transcript, one per session.

        The glob is exactly `<projects>/*/*.jsonl` — one directory deep, matching the
        layout described at the top of this file. Yields nothing (rather than raising)
        when Claude Code has never run on this machine. Note the exclusions below are
        mirrored in session_id_for_path(); the two must agree.
        """
        if not self.projects_dir.exists():
            return
        # Skip symlinks: `cr` links a session into other project dirs as resume
        # conduits, but the canonical transcript is the real file at its origin.
        # Indexing only real files keeps one row per session, at its true home.
        for p in self.projects_dir.glob("*/*.jsonl"):
            if p.is_symlink() or p.parent.name.endswith(_ENRICH_CWD_SUFFIX):
                continue        # resume conduits; our own headless enrichment runs
            yield p

    # -- cheap header ----------------------------------------------------------
    def parse_header(self, path: Path) -> Optional[SessionHeader]:
        """Build a SessionHeader from a few lines at each end of the transcript.

        What comes from where:
          head (first 60 records) — cwd, CLI version, git branch, start timestamp, the
                first typed message, and the `entrypoint` that identifies a headless run.
          tail (last 60 records)  — last activity timestamp, the current ai-title, and
                the model in use. Reading the END for the model is what makes a session
                that switched models report the one it finished on.

        Returns None for anything that is not a browsable session: an unreadable or empty
        file, a subagent sidechain, or a headless run this tool itself started. Reads the
        file (up to three passes in the worst case) but never writes.

        Called by the Stop hook, the watcher (repeatedly, while a session is live), the
        backfill and reconcile.
        """
        # Defense in depth: the Stop hook parses whatever transcript_path it is
        # handed without ever calling session_id_for_path(). parse_header is the
        # chokepoint every entry path shares, so the not-a-session check lives here
        # too — same shape as the sdk-cli exclusion below.
        if _SUBAGENT_DIR in path.parts:
            return None
        head = _read_head(path, 60)
        if not head:
            # No parseable records at all: the file is missing, empty, or still being
            # created. Not an error — the watcher will be back on the next write.
            return None
        tail = _read_tail_lines(path, 60)

        # The filename IS the id. This is the invariant session_id_for_path() relies on.
        session_id = path.stem
        cwd = ""
        version = None
        git_branch = None
        start_time = ""
        first_message = ""
        entrypoint = None

        # First-wins over the head window: these facts are stamped on many records, and
        # the earliest one is the session's own. `x = x or rec.get(...)` keeps the first
        # non-empty value and ignores every later repeat.
        for rec in head:
            cwd = cwd or rec.get("cwd", "")
            version = version or rec.get("version")
            git_branch = git_branch or rec.get("gitBranch")
            entrypoint = entrypoint or rec.get("entrypoint")
            if not start_time and rec.get("timestamp"):
                start_time = rec["timestamp"]
            if not first_message and _is_substantive_user(rec):
                first_message = _extract_text(rec.get("message", {}).get("content"))

        # Skip headless/SDK sessions (e.g. our own `claude --print` enrichment calls,
        # entrypoint "sdk-cli") so they never pollute the browsable index.
        if entrypoint == "sdk-cli":
            return None
        # if first typed message wasn't in the head window, scan a bit more
        if not first_message:
            first_message = self._first_typed_message(path)

        # Last-wins over the tail window (the mirror image of the head loop): each of
        # these is overwritten by every later record, so what survives is the newest.
        # Seeded with start_time so a session with a single record still has a sane end.
        last_activity = start_time
        title = None
        model = None
        for rec in tail:
            if rec.get("timestamp"):
                last_activity = rec["timestamp"]
            if rec.get("type") == "ai-title" and rec.get("aiTitle"):
                title = rec["aiTitle"]
            if rec.get("type") == "assistant":
                m = rec.get("message", {}).get("model")
                # "<synthetic>" marks a message Claude Code generated locally (an error
                # notice, an interrupt) rather than one a model produced. Recording it
                # would make the session look like it ran on a model called "<synthetic>"
                # and price it at $0.
                if m and m != "<synthetic>":
                    model = m
        if title is None:
            # No title in the last 60 records — it is written early and only rewritten
            # occasionally, so look back further before giving up.
            title = self._last_ai_title(path)

        # Where the transcript lives vs. what the session was about. project_path is the
        # encoded-cwd directory (used by restore and by the watcher's delete check);
        # folder_name is the human-readable project name, from the real cwd when the
        # transcript records one and from the encoded directory name otherwise.
        project_path = str(path.parent)
        folder_name = _folder_from_cwd(cwd) or path.parent.name

        return SessionHeader(
            session_id=session_id,
            cli_source=self.name,
            project_path=project_path,
            cwd=cwd,
            folder_name=folder_name,
            start_time=to_iso_utc(start_time),
            last_activity=to_iso_utc(last_activity),
            # Capped at 500 characters: this is a list-view label, not the message.
            first_message=first_message[:500],
            # A separate streaming pass — the head window is nowhere near enough to count
            # a long session's turns. Memoised; see _count_typed_turns.
            turn_count=_count_typed_turns(path),
            title=title,
            model_used=model,
            cli_version=version,
            metadata={"gitBranch": git_branch} if git_branch else {},
        )

    # -- full parse ------------------------------------------------------------
    def parse_full(self, path: Path) -> Optional[ParsedSession]:
        """Stream the whole transcript into ordered Turn objects.

        Expensive by design — used by full-text indexing, reasoning extraction, export
        and the enrichment prompt, never by the watcher. Keeps the session's real order
        by appending as it reads. Returns None for the same non-sessions parse_header()
        rejects, since it delegates that decision to it.

        Reads with errors="replace" and skips unparseable lines: a transcript truncated
        mid-write (the CLI is still running) must yield the turns it does have rather
        than nothing at all.
        """
        header = self.parse_header(path)
        if header is None:
            return None
        turns: list[Turn] = []
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue        # a partially flushed last line, typically
                t = rec.get("type")
                if t == "user" and _is_substantive_user(rec):
                    content = _extract_text(rec.get("message", {}).get("content"))
                    if content:
                        turns.append(Turn(role="user", content=content))
                elif t == "assistant":
                    msg = rec.get("message", {})
                    text, tools = _assistant_text_and_tools(msg.get("content"))
                    # A reply that is only a tool call still counts: "ran the tests" is
                    # part of the conversation even with no prose attached.
                    if text or tools:
                        turns.append(Turn(role="assistant", content=text, tool_calls=tools))
        return ParsedSession(header=header, turns=turns)

    # -- restore -----------------------------------------------------------------
    def restore_path(self, row) -> Optional[Path]:
        """Claude Code looks a session up at <projects>/<encoded-cwd>/<id>.jsonl —
        exactly the row's recorded project_path, which parse_header set from the
        transcript's real location. Refuse anything outside the projects tree so a
        corrupted row can never make restore write elsewhere on disk.

        `row` is a registry row (anything indexable by column name). Returns the file to
        write, or None to mean "this row cannot be restored here" — which is how the UI
        knows whether to offer a Restore button at all, per row: a registry copied from
        another machine records paths that are not under THIS projects directory, and the
        honest answer there is None rather than a button that fails when clicked.
        """
        project_path = row["project_path"]
        if not project_path:
            return None
        dest = Path(project_path) / f"{row['session_id']}.jsonl"
        try:
            dest.resolve().relative_to(self.projects_dir.resolve())
        except ValueError:
            return None
        return dest

    # -- identity / resume / availability ----------------------------------------
    def session_id_for_path(self, path: Path) -> Optional[str]:
        """The session id for a transcript path, or None if it is not one of ours.

        Pure path arithmetic — the file is usually already gone when this is called, and
        the answer must still be right. For a real transcript the result equals
        parse_header(path).session_id, which is the invariant everything downstream (the
        watcher's archive decision, prune, restore) depends on.

        The rejections below are not an optimisation: the watcher walks each project tree
        recursively and gates purely on this method, so anything discover()'s glob would
        not have produced has to be rejected here or it becomes a bogus registry row.
        Does not touch the filesystem except for the resolve() of the parent, which is
        needed to compare directories through symlinks.
        """
        # Mirrors discover()'s `*/*.jsonl`: a session transcript is a direct child
        # of a project dir. This gate — not the glob — is what the watcher uses to
        # decide if a path is a session at all, and it walks recursive=True, so it
        # must reject the same things the glob does.
        #
        # A multi-agent run writes its sidechains under the parent session's dir:
        #   <project>/<session-uuid>/subagents/agent-<id>.jsonl
        #   <project>/<session-uuid>/subagents/workflows/<wf-id>/{agent-<id>,journal}.jsonl
        # Those belong to the parent session. Every record in them is isSidechain,
        # so _is_substantive_user() rejects all of them and they index as
        # permanently-empty rows — one per subagent, swamping the UI. Worse, every
        # workflow journal has stem "journal", so they'd all collide onto a single
        # bogus session row.
        if path.suffix != ".jsonl" or _SUBAGENT_DIR in path.parts:
            return None
        if path.parent.name.endswith(_ENRICH_CWD_SUFFIX):
            return None          # our own headless enrichment runs live here
        # Exactly <projects>/<project>/<sid>.jsonl — the depth discover() globs.
        # Deeper files (a future <sid>/workflows/x/journal.jsonl) would index as
        # a session named after the file and then be archived by prune.
        try:
            if path.parent.parent.resolve() != self.projects_dir.resolve():
                return None
        except OSError:
            # resolve() can fail on a vanished or permission-denied path; treat "cannot
            # prove it belongs to us" as "not ours".
            return None
        return path.stem

    def resume_command(self, session_id: str) -> str:
        """The command that reopens this session in Claude Code, with its history intact.

        Returned as text for the UI and the `cr` helper to display or hand to a shell;
        nothing is run here. shlex.quote() guards the shell against an id that is not the
        plain UUID we expect.
        """
        return f"claude --resume {shlex.quote(session_id)}"

    def has_binary(self) -> bool:
        """Whether `claude` is runnable from here — a UI hint for resume/bridge only."""
        return shutil.which("claude") is not None

    def is_available(self) -> bool:
        """Whether there are transcripts to read.

        Deliberately NOT gated on `which claude`: the watcher and backfill must
        keep indexing ~/.claude/projects after Claude Code is uninstalled or
        drops off the daemon's PATH — those transcripts are exactly what this
        tool exists to keep browsable (codex/opencode already behave this way).
        """
        return self.projects_dir.exists()

    # -- helpers ---------------------------------------------------------------
    def _first_typed_message(self, path: Path) -> str:
        """Second-chance scan for the user's opening message.

        A session can start with a long run of machinery — hook output, injected skill
        instructions, a resumed history — before the human's first words, pushing them
        past parse_header()'s 60-line window. 1200 lines is a compromise: far enough to
        cover that preamble, short enough not to stream a huge file on every watcher tick.
        Returns '' if nothing typed is found, which the upsert reads as "not known yet"
        and will happily fill in on a later parse.
        """
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in islice(fh, 0, 1200):
                if _USER_MARKER not in line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if _is_substantive_user(rec):
                    text = _extract_text(rec.get("message", {}).get("content"))
                    if text:
                        return text
        return ""

    def _last_ai_title(self, path: Path) -> Optional[str]:
        """The most recent ai-title record within the last 200 lines, or None.

        Claude writes a title early and rewrites it as the topic drifts, so the LAST one
        is the current one — hence the loop keeps overwriting instead of returning on the
        first hit. A wider window than parse_header's tail because a busy session can
        append hundreds of records after the final retitle.
        """
        title = None
        for rec in _read_tail_lines(path, 200):
            if rec.get("type") == "ai-title" and rec.get("aiTitle"):
                title = rec["aiTitle"]
        return title


# --- module-level parsing helpers ---------------------------------------------
def _is_substantive_user(rec: dict) -> bool:
    """A real human-typed turn, format-agnostic.

    Newer transcripts mark these with promptSource=="typed"; older ones lack that
    field. The cross-format signal: a non-meta, non-sidechain user record whose
    message.content is a *string* (tool-results are lists) that is non-empty after
    stripping command/caveat wrapper lines.

    This single predicate defines what `turn_count` counts and which record supplies
    `first_message`, so it is the most consequential function in the file: loosen it and
    every sidechain becomes a session, tighten it and real sessions report 0 turns and
    get classified as noise when their transcript later disappears.

    The four gates, in order:
      1. type "user", not isMeta (a system-injected pseudo-message), not isSidechain
         (a subagent's conversation, not the user's).
      2. promptSource, when present, must be "typed" — a pasted resume, a queued
         command or a replay is not a turn. Absent on older transcripts, hence the
         `rec.get(...) and` guard rather than a straight comparison.
      3. message.content must be a *string*. Tool results arrive as user-role records
         whose content is a list of blocks; this one check excludes all of them.
      4. Something must survive stripping the wrapper lines — a bare "/model" line or a
         lone system reminder is not a turn.
    """
    if rec.get("type") != "user" or rec.get("isMeta") or rec.get("isSidechain"):
        return False
    if rec.get("promptSource") and rec.get("promptSource") != "typed":
        return False
    content = rec.get("message", {}).get("content")
    if not isinstance(content, str):
        return False
    return bool(_strip_command_lines(content).strip())


def _extract_text(content) -> str:
    """Normalize message.content (string or block list) to clean text.

    `content` is whatever the transcript held: a plain string (typed messages), or the
    list of typed blocks an assistant reply uses. Text blocks are joined with newlines in
    order; tool_result blocks are dropped, because a tool's output is not part of what
    the message SAID and would swamp the text with file dumps. Anything else — a number,
    None, a shape from a future format — yields ''.
    """
    if isinstance(content, str):
        return _strip_command_lines(content)
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" and block.get("text"):
                    parts.append(block["text"])
                elif block.get("type") == "tool_result":
                    continue  # skip tool results for "message" text
            elif isinstance(block, str):
                parts.append(block)
        return _strip_command_lines("\n".join(parts))
    return ""


def _strip_command_lines(text: str) -> str:
    """Drop the machinery lines (_CMD_LINE) from a message and trim the rest.

    Line-by-line rather than a regex over the whole text, so a genuine message that
    merely *mentions* one of those tags keeps everything else it said.
    """
    lines = [ln for ln in text.splitlines() if not _CMD_LINE.match(ln)]
    return "\n".join(lines).strip()


def _assistant_text_and_tools(content) -> tuple[str, list[dict]]:
    """Split an assistant reply's blocks into (prose, tool calls).

    Returns the visible text joined in order, plus one {"name", "input"} entry per
    tool_use block. "thinking" blocks — the model's private reasoning — are deliberately
    NOT included: they are extracted separately into the reasoning archive, and mixing
    them into the reply text would put them into exports and search results.
    """
    text_parts: list[str] = []
    tools: list[dict] = []
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            bt = block.get("type")
            if bt == "text" and block.get("text"):
                text_parts.append(block["text"])
            elif bt == "tool_use":
                tools.append({"name": block.get("name", ""),
                              "input": _summarize_input(block.get("input"))})
    elif isinstance(content, str):
        text_parts.append(content)
    return "\n".join(text_parts).strip(), tools


def _summarize_input(inp) -> str:
    """One short, readable line describing what a tool was asked to do.

    A tool's arguments can be an entire file's new contents, so storing them verbatim is
    out of the question. The key list is ordered by how well each identifies the action —
    a shell command beats a path, a path beats a search pattern — and the first one
    present wins, truncated to 120 characters:

        {"command": "pytest -x"}                 -> 'command=pytest -x'
        {"file_path": "app.py", "content": ...}  -> 'file_path=app.py'
        {"foo": 1, "bar": 2}                     -> 'foo, bar'   (nothing recognised)
    """
    if not isinstance(inp, dict):
        return ""
    for key in ("command", "file_path", "path", "pattern", "query", "description"):
        if key in inp:
            return f"{key}={str(inp[key])[:120]}"
    # Nothing recognisable: name the first few argument keys so the call is still
    # identifiable, without risking a dump of their values.
    return ", ".join(list(inp.keys())[:4])


def _folder_from_cwd(cwd: str) -> str:
    """Project name from a working directory: "/Users/me/app" -> "app"; '' stays ''."""
    return Path(cwd).name if cwd else ""


def _read_head(path: Path, n: int) -> list[dict]:
    """Parse at most the first `n` non-empty lines into dicts.

    islice() keeps this lazy — the file is never read past line `n`, which is the whole
    point for a multi-megabyte transcript. Blank and unparseable lines are skipped, and
    an unreadable file yields [] rather than raising, so the caller's only failure mode
    is "no records".
    """
    out = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in islice(fh, 0, n):
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return out


def _read_tail_lines(path: Path, n: int, blocksize: int = 65536) -> list[dict]:
    """Read approximately the last n JSONL records without loading the whole file.

    A `tail` implementation: seek towards the end and read 64 KB blocks BACKWARDS,
    prepending each to what we have, until the buffer holds more than n newlines or the
    start of the file is reached. Binary mode because seeking has to be in bytes; the
    decode to text happens per line afterwards.

    "Approximately" because the first line in the buffer is usually a fragment of a
    longer record — it simply fails to parse and is dropped, which is why the slice takes
    n+1 pieces. Returns [] for an unreadable file. Used for the facts that live at the
    end of a session: its last timestamp, its final model, its current title.
    """
    try:
        size = path.stat().st_size
        with open(path, "rb") as fh:
            data = b""
            read = 0
            while read < size and data.count(b"\n") <= n:
                step = min(blocksize, size - read)
                read += step
                fh.seek(size - read)      # absolute offset, measured back from the end
                data = fh.read(step) + data
        lines = data.split(b"\n")[-(n + 1):]
    except OSError:
        return []
    out = []
    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            out.append(json.loads(raw.decode("utf-8", errors="replace")))
        except json.JSONDecodeError:
            continue
    return out


# (path -> (size, mtime_ns, count)) — parse_header runs on every debounced watcher
# tick; without this an active multi-MB transcript would be fully re-streamed
# every ~0.5s just to recount turns.
_TURN_CACHE: dict[str, tuple[int, int, int]] = {}


def _count_typed_turns(path: Path) -> int:
    """Count substantive user turns. Streams the file and JSON-parses only the
    lines that look like user records (cheap substring prefilter), so it works
    across both transcript formats without materializing the whole file.
    Results are memoized by (size, mtime) so unchanged files cost one stat().

    Why (size, mtime_ns) and not just mtime: a transcript is only ever appended to, so a
    changed size alone is decisive, and the nanosecond mtime catches the rare rewrite
    that happens to land on the same length. The cache can never go stale in the wrong
    direction — any write changes at least one of the two.

    Returns 0 for an unreadable file. 0 is meaningful downstream: a row with no turns and
    no first message is what indexer.infer_archive_reason() treats as sidechain noise.
    """
    try:
        st = path.stat()
    except OSError:
        return 0
    key = str(path)
    cached = _TURN_CACHE.get(key)
    if cached and cached[0] == st.st_size and cached[1] == st.st_mtime_ns:
        return cached[2]
    n = 0
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if _USER_MARKER not in line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if _is_substantive_user(rec):
                    n += 1
    except OSError:
        return 0
    if len(_TURN_CACHE) > 4096:  # bound memory in the long-lived watcher
        _TURN_CACHE.clear()
    _TURN_CACHE[key] = (st.st_size, st.st_mtime_ns, n)
    return n
