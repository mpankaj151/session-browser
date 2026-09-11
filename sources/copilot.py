"""GitHub Copilot CLI source adapter.

Sessions: ~/.copilot/session-state/<uuid>/events.jsonl  (+ workspace.yaml)
The sibling session.db holds only todos/inbox state — NOT the transcript — so we
parse events.jsonl for turns and read workspace.yaml for cheap header metadata.

Event stream (verified): session.start, session.model_change (data.newModel),
user.message (data.content), assistant.message (data.content, data.reasoningText,
data.toolRequests, data.outputTokens). Unlike Claude, Copilot DOES persist
reasoning text in assistant.message.reasoningText.

--- On-disk layout -----------------------------------------------------------------

One DIRECTORY per session, named after the session id, rather than one file:

    ~/.copilot/session-state/40f3e815-9be0-410c-9737-0a05cb06a3da/
        events.jsonl          the transcript: one JSON event per line
        workspace.yaml        cheap metadata: name, cwd, created_at, updated_at
        session.db            todos / inbox state only — NOT the conversation
        checkpoints/ files/ research/ vscode.metadata.json

The consequence for identity: the session id is the DIRECTORY name, not the file stem.
Every session's transcript is called "events.jsonl", so using the stem would collide
every Copilot session onto a single row — see session_id_for_path().

Header facts come from two files, which is why both are in the cache key: the times and
the title live in workspace.yaml, while the turns, the first message and the model live
in events.jsonl. A real workspace.yaml:

    id: 40f3e815-9be0-410c-9737-0a05cb06a3da
    cwd: /Users/me/Documents/app
    name: copilot plugin marketplace add
    user_named: false
    created_at: 2026-05-12T19:43:45.811Z
    updated_at: 2026-05-12T20:06:06.223Z

--- events.jsonl line shapes --------------------------------------------------------

Every line is one event: {"type": ..., "data": {...}, "id", "timestamp", "parentId"}.
The three this module reads (trimmed; `data` differs per type):

    {"type":"session.start","data":{"sessionId":"40f3e815-…","copilotVersion":"1.0.44",
     "context":{"cwd":"/Users/me/app"}},"timestamp":"2026-05-12T19:43:45.825Z"}

    {"type":"session.model_change","data":{"newModel":"claude-haiku-4.5"},
     "timestamp":"2026-05-12T19:43:47.132Z"}

    {"type":"user.message","data":{"content":"add a retry around the fetch"},…}

    {"type":"assistant.message","data":{"content":"Done — one retry with backoff.",
     "reasoningText":"the timeout is the flaky part…",
     "toolRequests":[{"name":"str_replace"}],"outputTokens":412},…}

Other types (system.message, tool results, telemetry) are ignored. Counting turns is
therefore simple compared with Claude: one user.message line is one turn, with none of
the isMeta / isSidechain / promptSource filtering, because Copilot does not write
machinery into the user's own events.

Vocabulary (session, transcript, turn, token, reasoning) is in docs/GLOSSARY.md.
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
from pathlib import Path
from typing import Iterator, Optional

import yaml

from .base import ParsedSession, SessionHeader, Turn, to_iso_utc

# (events.jsonl size, events mtime, workspace.yaml mtime) -> parsed header
# Module level, so the long-lived watcher shares one cache across adapter instances.
# Keyed by the transcript path; the triple is the file-identity fingerprint that decides
# whether the cached header is still valid. Bounded in parse_header (cleared past 4096
# entries) so a daemon running for months cannot grow it without limit.
_HDR_CACHE: dict[str, tuple[tuple[int, int, int], "SessionHeader | None"]] = {}

# Documented default; sources/registry.py may override it from config.toml. Copilot has
# no relocation environment variable, so there is nothing else to follow.
STATE_DIR = Path(os.path.expanduser("~/.copilot/session-state"))


def _text(v) -> str:
    """Event content defensively coerced to str (a non-string content value must
    degrade to '' rather than crash the whole session parse)."""
    return v if isinstance(v, str) else ""


class CopilotSource:
    """Adapter implementing sources/base.py's SessionSource protocol for Copilot CLI.

    Stateless apart from its state directory (the header cache is module level), so it is
    cheap to construct and every caller builds its own.
    """
    name = "copilot"

    def __init__(self, state_dir: Path | str = STATE_DIR):
        """`state_dir` is the folder of per-session directories; the tests pass a
        temporary one. Takes a str so config values need no conversion, and expands `~`
        here rather than at each call site."""
        self.state_dir = Path(os.path.expanduser(str(state_dir)))

    def discover(self) -> Iterator[Path]:
        """Yield each session's events.jsonl — one per session directory.

        Skips a symlinked session directory as well as a symlinked events.jsonl: either
        would index the same conversation a second time under a second path. A session
        directory with no events.jsonl (created but never used) is skipped silently.
        Yields nothing when Copilot has never run here.
        """
        if not self.state_dir.exists():
            return
        for d in self.state_dir.iterdir():
            if d.is_symlink():   # a symlinked session dir would index a second row of the same turns
                continue
            ev = d / "events.jsonl"
            if ev.exists() and not ev.is_symlink():
                yield ev

    def parse_header(self, path: Path) -> Optional[SessionHeader]:
        """Build a SessionHeader for the session owning `path` (its events.jsonl).

        Combines the two files described at the top of this module: workspace.yaml for
        title, cwd and the time span, events.jsonl for the turn count, the first message
        and the model.

        Unlike the Claude adapter this streams the WHOLE events file rather than reading
        both ends. It can afford to because the model is announced by a session.model_change
        event that may appear anywhere, and Copilot sessions are small; the memoisation
        below is what keeps the watcher's repeated calls cheap.

        Returns None only when the files cannot be read (`stat` or `open` fails) — a
        Copilot session directory is by construction a real session, so there is no
        sidechain-style rejection here. Never raises.
        """
        sess_dir = path.parent
        # The id is the DIRECTORY name: every transcript here is called "events.jsonl".
        session_id = sess_dir.name
        # Memoized like codex: the watcher debounce would otherwise re-stream
        # events.jsonl every 500ms. Key covers workspace.yaml too — the title
        # and updated_at live there, not in the events file.
        try:
            st = path.stat()
            ws_path = sess_dir / "workspace.yaml"
            ws_mtime = ws_path.stat().st_mtime_ns if ws_path.exists() else 0
        except OSError:
            return None
        # Fingerprint of both files. A missing workspace.yaml contributes 0, so the entry
        # is correctly invalidated the moment one appears.
        cache_key = (st.st_size, st.st_mtime_ns, ws_mtime)
        hit = _HDR_CACHE.get(str(path))
        if hit is not None and hit[0] == cache_key:
            return hit[1]
        ws = self._workspace(sess_dir)
        cwd = ws.get("cwd", "")
        title = ws.get("name") or None
        # PyYAML auto-parses ISO timestamps into datetime objects; to_iso_utc
        # normalizes both datetimes and strings to the canonical sortable form.
        start = to_iso_utc(ws.get("created_at", ""))
        # Fall back to the start time so a session that has not been updated still has a
        # sortable last_activity instead of '' (which would sort before everything).
        last = to_iso_utc(ws.get("updated_at", "")) or start

        first_message = ""
        model = None
        turn_count = 0
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    # Cheap prefilter before the JSON parse: skip anything that cannot be
                    # an event (a blank line, a partially flushed final line).
                    if '"type"' not in line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    t = rec.get("type")
                    # Defensive: `data` is normally an object, but a malformed event must
                    # degrade to "no fields" rather than raise on .get() below.
                    data = rec.get("data", {}) if isinstance(rec.get("data"), dict) else {}
                    if t == "user.message":
                        turn_count += 1
                        if not first_message:
                            first_message = _text(data.get("content"))[:500]
                    elif t == "session.model_change" and data.get("newModel"):
                        # Last one wins: a session that switched models reports the one
                        # it ended on, matching what the Claude adapter does.
                        model = data["newModel"]
        except OSError:
            return None

        header = SessionHeader(
            session_id=session_id,
            cli_source=self.name,
            # The session's own directory: where its transcript lives, and where restore
            # writes it back. (For Claude this is a shared per-project directory; here it
            # is one directory per session.)
            project_path=str(sess_dir),
            cwd=cwd,
            # The project name for the UI. Falls back to the session id when
            # workspace.yaml has no cwd — ugly, but never blank.
            folder_name=Path(cwd).name if cwd else sess_dir.name,
            start_time=start,
            last_activity=last,
            first_message=first_message,
            turn_count=turn_count,
            title=title,
            model_used=model,
            metadata={"workspace_name": ws.get("name")},
        )
        # Crude bound on the long-lived watcher's memory: clear the whole cache rather
        # than evicting cleverly. Losing it costs one re-parse per active session.
        if len(_HDR_CACHE) > 4096:
            _HDR_CACHE.clear()
        _HDR_CACHE[str(path)] = (cache_key, header)
        return header

    def parse_full(self, path: Path) -> Optional[ParsedSession]:
        """Stream events.jsonl into ordered Turn objects.

        For the jobs that need the conversation text: full-text indexing, reasoning
        extraction, export, the enrichment prompt. Returns None when parse_header() does.

        Note what is NOT carried into a Turn: data.reasoningText, Copilot's record of the
        model's private thinking. That is extracted separately into the reasoning archive
        so it never leaks into exports and search results.
        """
        header = self.parse_header(path)
        if header is None:
            return None
        turns: list[Turn] = []
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue        # blank or half-written line; keep the rest
                t = rec.get("type")
                data = rec.get("data", {}) if isinstance(rec.get("data"), dict) else {}
                if t == "user.message":
                    content = _text(data.get("content")).strip()
                    if content:
                        turns.append(Turn(role="user", content=content))
                elif t == "assistant.message":
                    content = _text(data.get("content")).strip()
                    # Only the tool NAMES: Copilot records the arguments too, and they can
                    # be whole file contents. `or []` covers the field being absent or
                    # null; the isinstance guard covers a malformed entry.
                    tools = [{"name": tr.get("name", ""), "input": ""}
                             for tr in (data.get("toolRequests") or []) if isinstance(tr, dict)]
                    # A reply that only ran tools is still part of the conversation.
                    if content or tools:
                        turns.append(Turn(role="assistant", content=content, tool_calls=tools))
        return ParsedSession(header=header, turns=turns)

    def session_id_for_path(self, path: Path) -> Optional[str]:
        """The session id for a path, without opening anything, or None if it is not a
        Copilot transcript.

        Only the file literally named events.jsonl counts. The watcher walks these
        directories recursively and gates on this method, so the siblings — session.db,
        workspace.yaml, checkpoints/, files/ — and any future addition must all answer
        None here or they become bogus registry rows. Matches
        parse_header().session_id by construction: both take the directory name.
        """
        # Transcript lives at <state_dir>/<session-id>/events.jsonl; the id is the
        # directory name, NOT the filename stem ("events").
        return path.parent.name if path.name == "events.jsonl" else None

    def resume_command(self, session_id: str) -> str:
        """The command that reopens this session in Copilot CLI, history intact.

        Text for the UI and the `cr` helper; nothing is executed here. Note Copilot's
        `--resume=<id>` spelling differs from Claude's `--resume <id>` — exactly the kind
        of per-CLI detail that lives in an adapter and nowhere else.
        """
        return f"copilot --resume={shlex.quote(session_id)}"

    def restore_path(self, row) -> Optional[Path]:
        """<state_dir>/<sid>/events.jsonl — the row's project_path IS that
        session dir. Refuses anything outside the state tree.

        `row` is a registry row. Returns the file a restore must write, or None meaning
        "not restorable here" — which is how the UI decides, per row, whether to show a
        Restore button at all. Three things must hold, all checked below: the path is
        inside this state directory, it is exactly one level deep, and that level's name
        is the row's own session id. The last check matters because the id and the
        directory name are the same fact; if they have drifted, the restored file would
        be invisible to Copilot's own resume.
        """
        project_path = row["project_path"] or ""
        if not project_path:
            return None
        sess_dir = Path(os.path.expanduser(project_path))
        try:
            rel = sess_dir.resolve().relative_to(self.state_dir.resolve())
        except (ValueError, OSError):
            return None
        if len(rel.parts) != 1 or rel.parts[0] != row["session_id"]:
            return None
        return sess_dir / "events.jsonl"

    def has_binary(self) -> bool:
        """Whether `copilot` is runnable from here — a UI hint for resume/bridge only."""
        return shutil.which("copilot") is not None

    def is_available(self) -> bool:
        """Transcripts to read — never gated on the binary (see ClaudeSource)."""
        return self.state_dir.exists()

    # -- helpers --
    @staticmethod
    def _workspace(sess_dir: Path) -> dict:
        """Parse a session's workspace.yaml into a dict; {} if there isn't a usable one.

        YAML (not JSON) is Copilot's choice, hence the PyYAML dependency; safe_load is
        used because a config file must never be able to construct arbitrary objects.

        Returning {} for every failure is what keeps a session indexable with whatever
        events.jsonl alone can tell us — the caller then reads '' for cwd and title,
        which the upsert treats as "not known yet" rather than storing as empty.
        """
        wf = sess_dir / "workspace.yaml"
        if not wf.exists():
            return {}
        try:
            # utf-8 with replacement like every other reader: the locale default
            # raised UnicodeDecodeError (a ValueError) out of parse_header
            data = yaml.safe_load(wf.read_text(encoding="utf-8", errors="replace"))
        except (yaml.YAMLError, OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}
