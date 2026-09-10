"""Reasoning extraction — the headline feature.

Reconstructs Claude's *decision path* for a session: for each assistant turn, the
visible reasoning it wrote (text), the actions it took (tool_use), and a flag for
whether extended (hidden) thinking occurred on that turn.

IMPORTANT — what Claude Code actually stores: transcripts DO contain `thinking`
content blocks, but in current Claude Code versions their text is EMPTY — only a
cryptographic `signature` is persisted, not the plaintext chain-of-thought. So the
true internal reasoning text is not recoverable from disk. What we CAN reconstruct
faithfully is the *visible* reasoning (what Claude said in its responses) plus the
exact sequence of actions — which together explain how it reached each decision.
Turns where hidden thinking occurred are marked so the trail is honest about the
gap. This is deterministic, local, and free — it runs even with LLM enrichment off.

Storage:
  - readable Markdown -> <archive>/readable/YYYY/MM/<session>-<slug>.md
  - raw transcript copy -> <archive>/raw/YYYY/MM/<session>.jsonl (idempotent, @vN)
  - per-step rows in session_artifacts (type='reasoning') + sessions.reasoning_path
"""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import redact as _redact
import sbconfig

ARCHIVE = sbconfig.REASONING_ARCHIVE


@dataclass
class ReasoningStep:
    turn_index: int
    thinking: str
    decision: str
    actions: list[dict] = field(default_factory=list)  # [{tool, input}]
    signature_present: bool = False
    timestamp: Optional[str] = None


class _Closing:
    """Run an already-entered context manager's __exit__ when the block ends."""
    def __init__(self, cm):
        self._cm = cm

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return self._cm.__exit__(*exc)


# --- extraction ---------------------------------------------------------------
def extract(transcript_path: Path | str) -> list[ReasoningStep]:
    path = Path(transcript_path)
    steps: list[ReasoningStep] = []
    turn = 0
    try:
        fh = open(path, "r", encoding="utf-8", errors="replace")
    except OSError:
        return steps
    with fh:
        for line in fh:
            line = line.strip()
            # cheap, spacing-robust prefilter before the full JSON parse
            if not line or "assistant" not in line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") != "assistant":
                continue
            content = rec.get("message", {}).get("content")
            if not isinstance(content, list):
                continue
            thinking_parts, decision_parts, actions = [], [], []
            sig = False
            for block in content:
                if not isinstance(block, dict):
                    continue
                bt = block.get("type")
                if bt == "thinking":
                    if block.get("thinking"):
                        thinking_parts.append(block["thinking"].strip())
                    if block.get("signature"):
                        sig = True
                elif bt == "redacted_thinking":
                    thinking_parts.append("[redacted reasoning]")
                    sig = True
                elif bt == "text" and block.get("text"):
                    decision_parts.append(block["text"].strip())
                elif bt == "tool_use":
                    actions.append({
                        "tool": block.get("name", ""),
                        "input": _summarize_input(block.get("input")),
                    })
            if not (thinking_parts or decision_parts or actions):
                continue
            turn += 1
            steps.append(ReasoningStep(
                turn_index=turn,
                thinking="\n\n".join(thinking_parts),
                decision="\n\n".join(decision_parts),
                actions=actions,
                signature_present=sig,
                timestamp=rec.get("timestamp"),
            ))
    return steps


def extract_copilot(events_path: Path | str) -> list[ReasoningStep]:
    """Copilot decision trail. Unlike Claude, Copilot persists real reasoning text
    in assistant.message.reasoningText, so these trails include actual reasoning."""
    path = Path(events_path)
    steps: list[ReasoningStep] = []
    turn = 0
    try:
        fh = open(path, "r", encoding="utf-8", errors="replace")
    except OSError:
        return steps
    with fh:
        for line in fh:
            if "assistant.message" not in line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") != "assistant.message":
                continue
            data = rec.get("data", {}) if isinstance(rec.get("data"), dict) else {}
            thinking = (data.get("reasoningText") or "").strip()
            decision = (data.get("content") or "").strip()
            actions = [{"tool": tr.get("name", ""), "input": _summarize_input(tr.get("arguments") or tr.get("input"))}
                       for tr in (data.get("toolRequests") or []) if isinstance(tr, dict)]
            if not (thinking or decision or actions):
                continue
            turn += 1
            steps.append(ReasoningStep(
                turn_index=turn, thinking=thinking, decision=decision, actions=actions,
                signature_present=bool(thinking or data.get("reasoningOpaque")),
                timestamp=rec.get("timestamp"),
            ))
    return steps


def extract_codex(rollout_path) -> list:
    """Codex decision trail; a corrupt or truncated zstd rollout is 'no trail',
    never a traceback (zstandard.ZstdError is not a ValueError)."""
    from sources.codex import ROLLOUT_ERRORS
    try:
        return _extract_codex(rollout_path)
    except ROLLOUT_ERRORS:
        return []


def _extract_codex(rollout_path: Path | str) -> list[ReasoningStep]:
    """Codex decision trail. Codex stores reasoning as `type:reasoning` records but
    their text is encrypted (encrypted_content only) — like Claude's empty thinking,
    the plaintext isn't recoverable. So we reconstruct the visible agent messages
    plus the exact tool-call sequence, flagging turns that had encrypted reasoning."""
    from sources.codex import _item_text, open_rollout   # the adapter owns rollout I/O + item shapes
    path = Path(rollout_path)
    steps: list[ReasoningStep] = []
    turn = 0
    pending_actions: list[dict] = []
    saw_reasoning = False
    try:
        cm = open_rollout(path)
        fh = cm.__enter__()
    except (OSError, ImportError):
        return steps
    # `cm` must outlive the loop: it owns the (possibly zstd) file handles.
    with cm if False else _Closing(cm):
        for line in fh:
            if '"payload"' not in line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            p = rec.get("payload")
            if not isinstance(p, dict):
                continue
            pt = p.get("type")
            if pt == "reasoning":
                saw_reasoning = True
            elif pt in ("function_call", "custom_tool_call"):
                pending_actions.append({
                    "tool": p.get("name", ""),
                    "input": _summarize_input_str(p.get("arguments") or p.get("input")),
                })
            elif pt == "agent_message" and isinstance(p.get("message"), str):
                turn += 1
                steps.append(ReasoningStep(
                    turn_index=turn, thinking="", decision=p["message"].strip(),
                    actions=pending_actions, signature_present=saw_reasoning,
                    timestamp=rec.get("timestamp"),
                ))
                pending_actions = []
                saw_reasoning = False
            elif pt == "item_completed" and isinstance(p.get("item"), dict):
                # Paginated dialect (the current Codex format): every TurnItem
                # arrives wrapped in item_completed. Same trail, other spelling —
                # without this branch every recent Codex session had no trail.
                item = p["item"]
                it = str(item.get("type", "")).lower()
                if it == "reasoning":
                    saw_reasoning = True
                elif it == "commandexecution":
                    pending_actions.append({"tool": "shell",
                                            "input": _summarize_input_str(item.get("command"))})
                elif it == "filechange":
                    pending_actions.append({"tool": "apply_patch",
                                            "input": _summarize_input_str(item.get("changes") or item.get("path"))})
                elif it in ("mcptoolcall", "functioncall", "customtoolcall"):
                    name = item.get("tool") or item.get("name") or ""
                    if item.get("server"):
                        name = f"{item['server']}/{name}"
                    pending_actions.append({"tool": name,
                                            "input": _summarize_input_str(item.get("arguments") or item.get("input"))})
                elif it == "websearch":
                    pending_actions.append({"tool": "web_search",
                                            "input": _summarize_input_str(item.get("query"))})
                elif it == "agentmessage":
                    text = _item_text(item)
                    if text:
                        turn += 1
                        steps.append(ReasoningStep(
                            turn_index=turn, thinking="", decision=text,
                            actions=pending_actions, signature_present=saw_reasoning,
                            timestamp=rec.get("timestamp"),
                        ))
                        pending_actions = []
                        saw_reasoning = False
    # trailing actions with no closing message
    if pending_actions:
        turn += 1
        steps.append(ReasoningStep(turn_index=turn, thinking="", decision="",
                                   actions=pending_actions, signature_present=saw_reasoning))
    return steps


def extract_opencode(mirror_path: Path | str) -> list[ReasoningStep]:
    """OpenCode decision trail — from the adapter's mirror file. Unlike Claude,
    OpenCode persists reasoning text (`reasoning` parts), so the trail carries
    the real chain of thought where the provider returned one, plus the visible
    response and the exact tool sequence. Root session only: a child (task tool)
    is a separate agent and shows up as the `subtask` action that spawned it."""
    from sources.opencode import _loads as _oc_loads, _tool_calls, _user_text
    from sources.base import to_iso_utc
    path = Path(mirror_path)
    steps: list[ReasoningStep] = []
    try:
        fh = open(path, "r", encoding="utf-8", errors="replace")
    except OSError:
        return steps
    with fh:
        head = _oc_loads(fh.readline().strip())
        root = (head.get("info") or {}).get("id") if head.get("type") == "session" else None
        if not root:
            return steps
        turn = 0
        for line in fh:
            rec = _oc_loads(line.strip())
            if rec.get("type") != "message" or rec.get("session") != root:
                continue
            info = rec.get("info") if isinstance(rec.get("info"), dict) else {}
            if info.get("role") != "assistant" or info.get("summary"):
                continue
            parts = [pd for pd in (rec.get("parts") or []) if isinstance(pd, dict)]
            thinking = "\n\n".join(
                (pd.get("text") or "").strip() for pd in parts
                if pd.get("type") == "reasoning" and isinstance(pd.get("text"), str) and pd.get("text").strip())
            signed = any(pd.get("type") == "reasoning" and _has_signature(pd.get("metadata")) for pd in parts)
            decision = _user_text(parts)
            actions = [{"tool": c["name"], "input": c["input"]} for c in _tool_calls(parts)]
            if not (thinking or decision or actions):
                continue          # aborted / part-less turn
            turn += 1
            steps.append(ReasoningStep(
                turn_index=turn, thinking=thinking, decision=decision, actions=actions,
                signature_present=bool(thinking) or signed,
                timestamp=to_iso_utc((info.get("time") or {}).get("created")) or None,
            ))
    return steps


def _has_signature(meta) -> bool:
    """Provider reasoning metadata (e.g. {"anthropic": {"signature": ...}})."""
    if not isinstance(meta, dict):
        return False
    return any(isinstance(v, dict) and "signature" in v for v in meta.values()) or "signature" in meta


def _summarize_input_str(inp) -> str:
    """Codex tool args arrive as a JSON string or a dict; summarize either."""
    if isinstance(inp, str):
        try:
            inp = json.loads(inp)
        except (json.JSONDecodeError, ValueError):
            return inp[:140]
    return _summarize_input(inp) if isinstance(inp, dict) else ""


def _summarize_input(inp) -> str:
    if not isinstance(inp, dict):
        return ""
    for key in ("command", "cmd", "file_path", "path", "pattern", "query", "description", "url"):
        if key in inp:
            return f"{key}={str(inp[key])[:140]}"
    return ", ".join(list(inp.keys())[:5])


# --- rendering ----------------------------------------------------------------
def render_markdown(steps: list[ReasoningStep], header: dict) -> str:
    title = header.get("title") or header.get("first_message", "")[:60] or header.get("session_id", "")
    has_thinking_text = any(s.thinking for s in steps)
    n_reasoning = sum(1 for s in steps if s.thinking)
    n_visible = sum(1 for s in steps if s.decision)
    source = header.get("cli_source", "claude")
    if has_thinking_text:
        note = ("> This trail includes the reasoning text the CLI persisted, plus the "
                "stated response and exact action sequence for each turn.")
    elif source == "claude":
        note = ("> **Note:** Claude Code stores extended-thinking blocks without their text "
                "(only a cryptographic signature), so the *internal* chain-of-thought is not "
                "recoverable. This trail reconstructs the **visible** reasoning plus the exact "
                "action sequence; turns marked 🔒 had hidden thinking whose text was not persisted.")
    else:
        note = (f"> **Note:** {source} did not persist reasoning text for these turns (a "
                "signature or nothing at all), so the *internal* chain-of-thought is not "
                "recoverable here. This trail reconstructs the **visible** reasoning plus the exact "
                "action sequence; turns marked 🔒 had hidden thinking whose text was not persisted.")
    lines = [
        f"# Decision trail — {title}",
        "",
        f"- **Session:** `{header.get('session_id','')}`",
        f"- **Source:** {header.get('cli_source','claude')}  ·  **Model:** {header.get('model_used','')}",
        f"- **Folder:** {header.get('folder_name','')}",
        f"- **Assistant turns:** {len(steps)}  ·  **with reasoning:** {n_reasoning}  ·  "
        f"**with a response:** {n_visible}",
        f"- **Span:** {header.get('start_time','')} → {header.get('last_activity','')}",
        "",
        note,
        "",
        "---",
        "",
    ]
    for s in steps:
        head = f"## Turn {s.turn_index}"
        if s.signature_present and not s.thinking:
            head += "  🔒"
        if s.timestamp:
            head += f"  ·  _{s.timestamp}_"
        lines.append(head)
        if s.thinking:
            lines.append("\n**🧠 Reasoning**\n")
            lines.append(_quote(s.thinking[:4000]))
        elif s.signature_present:
            lines.append("\n_🔒 Extended thinking occurred here; text not stored by the CLI._")
        if s.decision:
            lines.append("\n**💬 Response**\n")
            lines.append(_quote(s.decision[:4000]))
        if s.actions:
            lines.append("\n**⚙️ Actions**\n")
            for a in s.actions:
                lines.append(f"- `{a['tool']}` {a['input']}")
        lines.append("\n---\n")
    # Trails are shareable artifacts — a tool command like `export TOKEN=ghp_...`
    # must not survive into the archive verbatim.
    return _redact.redact("\n".join(lines))


def _quote(text: str) -> str:
    return "\n".join("> " + ln if ln else ">" for ln in text.splitlines())


# --- archive + persist --------------------------------------------------------
def _slug(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", (text or "").lower()).strip("-")
    return s[:60] or "session"


_YM = re.compile(r"^\d{4}-\d{2}")


def _ym_dir(base: Path, last_activity: str) -> Path:
    # Exactly two components: the raw-copy glob is */*/*.jsonl, so anything
    # else would be archived where nothing looks (unrestorable, silently).
    head = (last_activity or "")[:7]
    ym = head.replace("-", "/") if _YM.match(head) else "0000/00"
    d = base / ym
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe_sid(sid) -> str:
    """session_id is file CONTENT (a rollout's payload.id, a mirror header) used
    as a path component: refuse anything that could leave the vault."""
    sid = str(sid or "")
    if not sid or "/" in sid or "\\" in sid or sid in (".", "..") or sid.startswith(".."):
        raise ValueError(f"refusing to archive with an unsafe session id {sid!r}")
    return sid


# Raw copies keep the source's representation: Codex zstd-compresses cold
# rollouts in place, and a byte copy named .jsonl would be a zstd frame every
# reader parses as text. The codex adapter reads either suffix transparently.
_ZST = ".zst"


def _raw_suffix(src: Path) -> str:
    return ".jsonl" + _ZST if src.name.endswith(_ZST) else ".jsonl"


def archive_raw(transcript_path: Path, header: dict) -> Path:
    """Copy the raw transcript into the archive, versioning on content change."""
    src = Path(transcript_path)
    dest_dir = _ym_dir(ARCHIVE / "raw", header.get("last_activity", ""))
    sid = _safe_sid(header.get("session_id") or src.name.split(".")[0])
    suffix = _raw_suffix(src)
    dest = dest_dir / f"{sid}{suffix}"
    if dest.exists():
        if dest.stat().st_size == src.stat().st_size:
            return dest  # unchanged — idempotent
        v = 2
        while (dest_dir / f"{sid}@v{v}{suffix}").exists():
            v += 1
        dest = dest_dir / f"{sid}@v{v}{suffix}"
    shutil.copy2(src, dest)
    return dest


_RAW_STEM = re.compile(r"^(?P<sid>.+?)(?:@v(?P<v>\d+))?$")


def _raw_copies():
    """Every raw transcript copy as (session_id, mtime, path). One walk. The
    @vN counter restarts per YYYY/MM directory, so ordering is by the copy's
    mtime (copy2 preserves the source's) — a September @v3 must not outrank
    October's first copy."""
    raw = ARCHIVE / "raw"
    if not raw.is_dir():
        return
    for pattern in ("*/*/*.jsonl", f"*/*/*.jsonl{_ZST}"):
        for p in raw.glob(pattern):
            base = p.name[:-len(".jsonl" + _ZST)] if p.name.endswith(_ZST) else p.stem
            m = _RAW_STEM.match(base)
            try:
                mtime = p.stat().st_mtime
            except OSError:
                continue
            # mtime first, then the @vN version: equal mtimes (1 s volumes, an
            # unchanged source mtime) used to tie-break on the path STRING, so
            # @v9 beat @v10 and Restore put back the older, shorter copy.
            yield m.group("sid"), (mtime, int(m.group("v") or 1)), p


def archived_raw_index() -> dict[str, Path]:
    """session_id -> newest raw transcript copy in the archive. A single
    directory walk, so the UI can label 500 archived rows without 500 walks."""
    best: dict[str, tuple[tuple, Path]] = {}
    for sid, key, p in _raw_copies():
        if sid not in best or (key, str(p)) > (best[sid][0], str(best[sid][1])):
            best[sid] = (key, p)
    return {sid: p for sid, (_, p) in best.items()}


def find_archived_raw(session_id: str) -> Path | None:
    """Newest raw copy of one session's transcript, or None. This is what makes
    an aged-out session restorable after Claude Code's cleanup deleted the
    original: refresh-all copies every indexable transcript here first."""
    hits = [(key, str(p), p) for sid, key, p in _raw_copies() if sid == session_id]
    return max(hits)[2] if hits else None


def write_readable(steps: list[ReasoningStep], header: dict) -> Path:
    dest_dir = _ym_dir(ARCHIVE / "readable", header.get("last_activity", ""))
    sid = _safe_sid(header.get("session_id", ""))
    fname = f"{sid}-{_slug(header.get('title') or header.get('first_message',''))}.md"
    dest = dest_dir / fname
    # Write atomically FIRST, retire the previous renders AFTER: the old order
    # (unlink, then a plain write) left no trail at all on ENOSPC or a kill
    # between the two, while sessions.reasoning_path still pointed at the
    # deleted file.
    fd, tmp = tempfile.mkstemp(dir=dest_dir, prefix=fname + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(render_markdown(steps, header))
        os.replace(tmp, dest)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    # A later title (from enrichment) or a month rollover changes the path; remove
    # the session's previous renders so the archive holds exactly one trail per
    # session. Keyed on the FULL id: an 8-char prefix collides for OpenCode
    # (`ses_` + a ms clock) and Codex (UUIDv7) ids and deleted other sessions'
    # trails. Legacy prefix-named files are retired by persist(), which knows
    # the path each row recorded.
    for old in (ARCHIVE / "readable").glob(f"*/*/{sid}-*.md"):
        if old != dest:
            old.unlink(missing_ok=True)
    return dest


def persist(session_id: str, steps: list[ReasoningStep], readable_path: Path,
            conn=None) -> None:
    """Record reasoning_path on the session and store per-step artifacts."""
    import indexer
    own = conn is None
    conn = conn or indexer.connect()
    try:
        # Retire the previous render this row pointed at (a legacy prefix name,
        # or last month's directory) — exactly that file, never a prefix sweep.
        prev = conn.execute("SELECT reasoning_path FROM sessions WHERE session_id = ?",
                            (session_id,)).fetchone()
        prev_path = Path(prev[0]) if prev and prev[0] else None
        if prev_path and prev_path != Path(readable_path):
            try:
                prev_path.resolve().relative_to((ARCHIVE / "readable").resolve())
                prev_path.unlink(missing_ok=True)
            except (ValueError, OSError):
                pass
        conn.execute(
            "UPDATE sessions SET reasoning_path = ? WHERE session_id = ?",
            (str(readable_path), session_id),
        )
        conn.execute(
            "DELETE FROM session_artifacts WHERE session_id = ? AND type = 'reasoning'",
            (session_id,),
        )
        for s in steps:
            # Hidden thinking text is empty in the transcript; the visible reasoning
            # Claude wrote (decision text) is the searchable reasoning content.
            content = s.thinking or s.decision
            if not content:
                continue
            conn.execute(
                "INSERT INTO session_artifacts (session_id, type, content, turn_index) "
                "VALUES (?, 'reasoning', ?, ?)",
                (session_id, _redact.redact(content[:8000]), s.turn_index),
            )
        if own:
            conn.commit()
    finally:
        if own:
            conn.close()
