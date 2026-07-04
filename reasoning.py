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
import re
import shutil
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


def extract_codex(rollout_path: Path | str) -> list[ReasoningStep]:
    """Codex decision trail. Codex stores reasoning as `type:reasoning` records but
    their text is encrypted (encrypted_content only) — like Claude's empty thinking,
    the plaintext isn't recoverable. So we reconstruct the visible agent messages
    plus the exact tool-call sequence, flagging turns that had encrypted reasoning."""
    path = Path(rollout_path)
    steps: list[ReasoningStep] = []
    turn = 0
    pending_actions: list[dict] = []
    saw_reasoning = False
    try:
        fh = open(path, "r", encoding="utf-8", errors="replace")
    except OSError:
        return steps
    with fh:
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
    # trailing actions with no closing message
    if pending_actions:
        turn += 1
        steps.append(ReasoningStep(turn_index=turn, thinking="", decision="",
                                   actions=pending_actions, signature_present=saw_reasoning))
    return steps


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
    if has_thinking_text:
        note = ("> This trail includes the reasoning text the CLI persisted, plus the "
                "stated response and exact action sequence for each turn.")
    else:
        note = ("> **Note:** Claude Code stores extended-thinking blocks without their text "
                "(only a cryptographic signature), so the *internal* chain-of-thought is not "
                "recoverable. This trail reconstructs the **visible** reasoning plus the exact "
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


def _ym_dir(base: Path, last_activity: str) -> Path:
    ym = (last_activity or "0000-00")[:7].replace("-", "/")  # YYYY/MM
    d = base / ym
    d.mkdir(parents=True, exist_ok=True)
    return d


def archive_raw(transcript_path: Path, header: dict) -> Path:
    """Copy the raw transcript into the archive, versioning on content change."""
    src = Path(transcript_path)
    dest_dir = _ym_dir(ARCHIVE / "raw", header.get("last_activity", ""))
    dest = dest_dir / f"{header.get('session_id', src.stem)}.jsonl"
    if dest.exists():
        if dest.stat().st_size == src.stat().st_size:
            return dest  # unchanged — idempotent
        v = 2
        while (dest_dir / f"{dest.stem}@v{v}.jsonl").exists():
            v += 1
        dest = dest_dir / f"{dest.stem}@v{v}.jsonl"
    shutil.copy2(src, dest)
    return dest


def write_readable(steps: list[ReasoningStep], header: dict) -> Path:
    dest_dir = _ym_dir(ARCHIVE / "readable", header.get("last_activity", ""))
    sid8 = header.get("session_id", "")[:8]
    fname = f"{sid8}-{_slug(header.get('title') or header.get('first_message',''))}.md"
    dest = dest_dir / fname
    # A later title (from enrichment) or a month rollover changes the path; remove
    # the session's previous renders so the archive holds exactly one trail per session.
    if sid8:
        for old in (ARCHIVE / "readable").glob(f"*/*/{sid8}-*.md"):
            if old != dest:
                old.unlink(missing_ok=True)
    dest.write_text(render_markdown(steps, header), encoding="utf-8")
    return dest


def persist(session_id: str, steps: list[ReasoningStep], readable_path: Path,
            conn=None) -> None:
    """Record reasoning_path on the session and store per-step artifacts."""
    import indexer
    own = conn is None
    conn = conn or indexer.connect()
    try:
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
