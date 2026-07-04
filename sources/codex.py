"""OpenAI Codex CLI source adapter.

Sessions: ~/.codex/sessions/YYYY/MM/DD/rollout-<ts>-<uuid>.jsonl
Each line is {"type": ..., "payload": {...}} (or a top-level record). Verified
against real rollout files:
  - first line: type "session_meta", payload {id, timestamp, cwd, cli_version,
    model_provider}
  - model: type "turn_context", payload.model (e.g. "gpt-5.5")
  - user turns: type "event_msg", payload.type "user_message", payload.message
    (a plain string — the response_item/message role=user records are wrapped in
    environment_context noise, so we prefer event_msg/user_message)
  - agent turns: payload.type "agent_message", payload.message
  - reasoning: payload.type "agent_reasoning"/"agent_reasoning_raw_content" carry
    Codex's reasoning text
  - title: payload.type "thread_name_updated" (also ~/.codex/session_index.jsonl)
  - tokens: payload.type "token_count", payload.info.total_token_usage (cumulative)
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Iterator, Optional

from .base import ParsedSession, SessionHeader, Turn, to_iso_utc

SESSIONS_DIR = Path(os.path.expanduser("~/.codex/sessions"))

# parse_header streams the whole rollout (last_activity lives on the final
# line), so memoize by (size, mtime): the watcher re-parses on a 500ms debounce
# and would otherwise pin a core on an active multi-MB session.
_HDR_CACHE: dict[str, tuple[tuple[int, int], "SessionHeader | None"]] = {}


def _payload(rec: dict) -> tuple[str, dict]:
    """(record type, payload dict). Codex nests real data under 'payload'."""
    p = rec.get("payload")
    if isinstance(p, dict):
        return p.get("type") or rec.get("type") or "", p
    return rec.get("type") or "", rec


class CodexSource:
    name = "codex"

    def __init__(self, sessions_dir: Path | str = SESSIONS_DIR):
        self.sessions_dir = Path(os.path.expanduser(str(sessions_dir)))

    # -- discovery -------------------------------------------------------------
    def discover(self) -> Iterator[Path]:
        if not self.sessions_dir.exists():
            return
        for p in self.sessions_dir.glob("*/*/*/rollout-*.jsonl"):
            if not p.is_symlink():
                yield p

    def session_id_for_path(self, path: Path) -> Optional[str]:
        # rollout-2026-05-10T13-32-08-019e11df-a087-7f82-8717-e023d8e8bf32.jsonl
        # -> the trailing UUID (last five dash-separated groups). No file read.
        if not (path.name.startswith("rollout-") and path.suffix == ".jsonl"):
            return None
        parts = path.stem.split("-")
        return "-".join(parts[-5:]) if len(parts) >= 5 else path.stem

    # -- cheap header ----------------------------------------------------------
    def parse_header(self, path: Path) -> Optional[SessionHeader]:
        try:
            st = path.stat()
        except OSError:
            return None
        cache_key = (st.st_size, st.st_mtime_ns)
        hit = _HDR_CACHE.get(str(path))
        if hit is not None and hit[0] == cache_key:
            return hit[1]
        session_id = self.session_id_for_path(path) or path.stem
        cwd = ""
        start_time = ""
        version = None
        model = None
        title = None
        first_message = ""
        last_ts = ""
        turn_count = 0
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    ts = rec.get("timestamp") or ""
                    if ts:
                        last_ts = ts
                    pt, p = _payload(rec)
                    if pt == "session_meta" or rec.get("type") == "session_meta":
                        session_id = p.get("id") or session_id
                        cwd = cwd or p.get("cwd", "")
                        start_time = start_time or p.get("timestamp") or ts
                        version = version or p.get("cli_version")
                    elif pt == "turn_context":
                        model = model or p.get("model")
                        cwd = cwd or p.get("cwd", "")
                    elif pt == "thread_name_updated":
                        title = p.get("name") or p.get("thread_name") or title
                    elif pt == "user_message":
                        turn_count += 1
                        if not first_message and isinstance(p.get("message"), str):
                            first_message = p["message"]
        except OSError:
            return None

        if len(_HDR_CACHE) > 4096:
            _HDR_CACHE.clear()
        if turn_count == 0 and not first_message:
            # a rollout with no user turns (aborted / meta-only) — not browsable.
            # Cached too: these files are exactly the ones re-scanned pointlessly.
            _HDR_CACHE[str(path)] = (cache_key, None)
            return None

        folder = Path(cwd).name if cwd else path.parent.name
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
        header = self.parse_header(path)
        if header is None:
            return None
        turns: list[Turn] = []
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    pt, p = _payload(rec)
                    if pt == "user_message" and isinstance(p.get("message"), str):
                        turns.append(Turn(role="user", content=p["message"].strip()))
                    elif pt == "agent_message" and isinstance(p.get("message"), str):
                        turns.append(Turn(role="assistant", content=p["message"].strip()))
        except OSError:
            return None
        return ParsedSession(header=header, turns=turns)

    # -- resume / availability -------------------------------------------------
    def resume_command(self, session_id: str) -> str:
        return f"codex resume {session_id}"

    def is_available(self) -> bool:
        return shutil.which("codex") is not None and self.sessions_dir.exists()
