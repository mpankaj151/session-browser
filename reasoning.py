"""Reasoning extraction — the headline feature.

A **reasoning trail** (or decision trail) is a readable Markdown reconstruction of HOW an
AI coding assistant reached its decisions in one session. Modern assistants can "think"
before answering: the model writes out its private deliberation first, then its visible
reply, then asks the CLI to run commands or edit files on its behalf (a *tool call*). A
trail lays those out turn by turn — the thinking text where the CLI stored it, the
response the user actually saw, and the exact sequence of actions — so months later you
can read why a decision was made, not just what changed. See docs/GLOSSARY.md
("Reasoning trail", "Tool call", "Turn").

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

Who calls this: scripts/extract-reasoning.py (driven by the nightly refresh and, detached,
by the Claude Stop hook), the OpenCode adapter just before it unlinks a mirror file, and
restore.py / scripts/build-fts.py for the raw-copy lookups. Reading is the UI (which
serves the Markdown trail) and full-text search (which indexes the per-step rows).

Two jobs in one module, and the second one matters more than its name suggests. Besides
rendering trails, `archive_raw()` keeps a byte-for-byte copy of every transcript it is
handed under <archive>/raw/. That vault is what makes a session survivable after its CLI
deletes the original — `restore.py` reads exactly these copies. So the reasoning archive
is also the transcript backup; see docs/ARCHITECTURE.md, "Archive lifecycle".

One extractor per CLI, because each records reasoning differently:
    extract()           Claude Code   — thinking blocks, text blocks, tool_use blocks
    extract_copilot()   Copilot CLI   — assistant.message events with reasoningText
    extract_codex()     Codex CLI     — rollout payloads, two dialects, maybe zstd
    extract_opencode()  OpenCode      — the adapter's mirror JSONL, reasoning parts
All four return the same list[ReasoningStep], so render_markdown() and persist() are
source-agnostic. Every one of them treats an unreadable or malformed file as "no trail"
(an empty list) rather than an exception: a single corrupt transcript must never abort a
nightly pass over thousands of sessions.
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

# Root of the reasoning archive (default ~/claude-reasoning-archive), holding readable/
# and raw/. A module-level name, not a function call, on purpose: the tests reassign
# reasoning.ARCHIVE to a temporary directory so nothing touches the real vault.
ARCHIVE = sbconfig.REASONING_ARCHIVE


@dataclass
class ReasoningStep:
    """One assistant turn of a trail, in the shape every extractor produces.

    turn_index         1-based position among the turns that had any content at all;
                       counted here, not taken from the transcript, so the numbering is
                       continuous even when a CLI records empty or aborted turns.
    thinking           the model's private deliberation, where the CLI persisted its text
                       (Copilot and OpenCode do; Claude and Codex store it unreadably —
                       see the module docstring). Empty string when there is none.
    decision           the visible reply the user saw for this turn.
    actions            the tool calls this turn made, in order, each already summarised
                       to one short line: {"tool": "Bash", "input": "command=pytest -q"}.
    signature_present  True when the transcript proves hidden thinking happened on this
                       turn even though its text was not stored (a cryptographic
                       signature, an encrypted blob, a bare `reasoning` record). The
                       renderer marks those turns 🔒 so the trail is honest about the gap
                       instead of silently looking like the model never thought.
    timestamp          the turn's time as the source recorded it, or None.
    """
    turn_index: int
    thinking: str
    decision: str
    actions: list[dict] = field(default_factory=list)  # [{tool, input}]
    signature_present: bool = False
    timestamp: Optional[str] = None


class _Closing:
    """Run an already-entered context manager's __exit__ when the block ends.

    Needed because _extract_codex() must call `open_rollout(path).__enter__()` inside a
    try/except (so a corrupt zstd file becomes "no trail" rather than a traceback) and
    only then start iterating. The context manager is therefore already entered by the
    time we reach the `with`, and re-entering it would be wrong; this shim just makes
    sure its __exit__ still runs — closing the possibly-decompressing file handles — when
    the loop finishes or raises.
    """
    def __init__(self, cm):
        """Wrap `cm`, a context manager whose __enter__ has ALREADY been called."""
        self._cm = cm

    def __enter__(self):
        """Do not re-enter the wrapped manager; just hand back this shim."""
        return self

    def __exit__(self, *exc):
        """Forward the exit (and any exception) to the wrapped manager so it closes."""
        return self._cm.__exit__(*exc)


# --- extraction ---------------------------------------------------------------
def extract(transcript_path: Path | str) -> list[ReasoningStep]:
    """Claude Code decision trail, from one ~/.claude/projects/<slug>/<uuid>.jsonl file.

    The transcript is JSONL — one JSON object per line. Only `type == "assistant"` lines
    matter here; each holds a `message.content` LIST of typed blocks, e.g.

      {"type":"assistant","timestamp":"2026-09-11T09:00:00Z","message":{"content":[
         {"type":"thinking","thinking":"","signature":"Er0BCk…"},
         {"type":"text","text":"I'll run the tests first."},
         {"type":"tool_use","name":"Bash","input":{"command":"pytest -q"}}]}}

    Note the empty `thinking` next to a non-empty `signature` — that is the norm, and the
    reason this function reconstructs the VISIBLE reasoning plus the exact action order
    and merely flags the turn as having had hidden thinking.

    Returns the steps in file order (an empty list for a missing, unreadable, or
    reasoning-free file — never raises). Lines that are not valid JSON are skipped:
    a transcript being appended to while we read can end in a half-written line.
    """
    path = Path(transcript_path)
    steps: list[ReasoningStep] = []
    turn = 0
    try:
        fh = open(path, "r", encoding="utf-8", errors="replace")
    except OSError:
        # Deleted between discovery and now, or unreadable: "no trail", not a crash.
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
                    # The vendor itself withheld this thinking block. Record the fact so
                    # the turn is not silently blank; there is nothing to recover.
                    thinking_parts.append("[redacted reasoning]")
                    sig = True
                elif bt == "text" and block.get("text"):
                    decision_parts.append(block["text"].strip())
                elif bt == "tool_use":
                    actions.append({
                        "tool": block.get("name", ""),
                        "input": _summarize_input(block.get("input")),
                    })
            # Nothing usable on this turn (an empty continuation record, a tool result
            # echoed back): don't burn a turn number on it.
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
    in assistant.message.reasoningText, so these trails include actual reasoning.

    Reads ~/.copilot/session-state/<id>/events.jsonl, whose lines are events rather than
    messages. One assistant turn looks like:

      {"type":"assistant.message","timestamp":"…","data":{
         "reasoningText":"The failing test is in checkout…",
         "content":"Let me fix the fixture.",
         "toolRequests":[{"name":"str_replace","arguments":{"path":"tests/conftest.py"}}]}}

    `reasoningOpaque` may appear instead of readable text when the provider returned an
    encrypted blob; that still counts as "thinking happened here" for the 🔒 marker.
    Same contract as extract(): file order, empty list on any failure, never raises.
    """
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
    never a traceback (zstandard.ZstdError is not a ValueError).

    A thin guard around _extract_codex(). Codex compresses rollouts older than about a
    week to `.jsonl.zst`, and a frame truncated by a crash or a half-finished copy raises
    `zstandard.ZstdError` — which inherits from Exception, not from ValueError or OSError,
    so none of the ordinary `except` clauses inside would have caught it. ROLLOUT_ERRORS
    is the adapter's canonical tuple of "this rollout is unreadable" exceptions, kept in
    one place so every reader agrees.
    """
    from sources.codex import ROLLOUT_ERRORS
    try:
        return _extract_codex(rollout_path)
    except ROLLOUT_ERRORS:
        return []


def _extract_codex(rollout_path: Path | str) -> list[ReasoningStep]:
    """Codex decision trail. Codex stores reasoning as `type:reasoning` records but
    their text is encrypted (encrypted_content only) — like Claude's empty thinking,
    the plaintext isn't recoverable. So we reconstruct the visible agent messages
    plus the exact tool-call sequence, flagging turns that had encrypted reasoning.

    Codex rollouts are JSONL where every line wraps a `payload`. Two dialects exist and
    both are handled below:

      legacy      one record per item, typed directly —
                  {"timestamp":"…","payload":{"type":"function_call","name":"shell",
                                              "arguments":"{\\"command\\":\\"ls\\"}"}}
      paginated   the current format, every item wrapped —
                  {"payload":{"type":"item_completed","item":{"type":"CommandExecution",
                                                              "command":"ls"}}}

    Actions accumulate in `pending_actions` and are attached to the NEXT agent message,
    because Codex records the tool calls a turn made before the message that concludes
    it. Anything left over at the end becomes a final step with no response text, so a
    session that ended mid-work still shows what it was doing.

    Returns steps in file order; an unreadable file gives an empty list. Corrupt-zstd
    errors propagate to the extract_codex() wrapper above, which swallows them.
    """
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
        # ImportError too: a `.zst` rollout needs the optional zstandard package, and a
        # machine without it should simply have no Codex trails, not a crashing job.
        return steps
    # `cm` must outlive the loop: it owns the (possibly zstd) file handles.
    with cm if False else _Closing(cm):
        for line in fh:
            # Cheap substring prefilter: skip lines that cannot be item records before
            # paying for json.loads on a multi-megabyte rollout.
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
                # Legacy dialect. The record exists but carries only encrypted_content,
                # so all we can honestly say is "thinking happened on this turn".
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
                # Item types are CamelCase in the wire format ("CommandExecution",
                # "McpToolCall"); lower-casing once lets the comparisons below be plain
                # lowercase literals and survives casing changes between Codex versions.
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
                    # An MCP tool (a tool served by an external helper process) is only
                    # identified by server+tool; show it as "server/tool" so two servers
                    # offering a "search" tool stay distinguishable in the trail.
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
    is a separate agent and shows up as the `subtask` action that spawned it.

    Input is the mirror file the OpenCode adapter projects from OpenCode's SQLite
    database (docs/GLOSSARY.md, "Mirror"). Line 1 is the session header; every later line
    is one message with its `parts`:

      {"type":"session","info":{"id":"ses_abc","title":"…"}}
      {"type":"message","session":"ses_abc","info":{"role":"assistant","time":{…}},
       "parts":[{"type":"reasoning","text":"…","metadata":{"anthropic":{"signature":"…"}}},
                {"type":"text","text":"Fixed."},{"type":"tool","tool":"bash", …}]}

    Lines whose `session` is not the root id belong to a child agent and are skipped —
    a child is its own conversation, and including it would interleave two trains of
    thought. Messages flagged `summary` are compaction recaps, not real turns.
    Returns an empty list if the file is unreadable or line 1 is not a session header.
    """
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
    """Provider reasoning metadata (e.g. {"anthropic": {"signature": ...}}).

    True when a reasoning part carries a cryptographic signature — proof the model did
    think here even if OpenCode stored no text for it. The shape is provider-specific, so
    both the nested form above and a flat {"signature": …} are accepted.
    """
    if not isinstance(meta, dict):
        return False
    return any(isinstance(v, dict) and "signature" in v for v in meta.values()) or "signature" in meta


def _summarize_input_str(inp) -> str:
    """Codex tool args arrive as a JSON string or a dict; summarize either.

    Codex records `arguments` as a JSON-encoded STRING ('{"command":"ls"}') in the legacy
    dialect and as a real object in places. Decode when possible, then hand off to
    _summarize_input(); a plain non-JSON string (a raw shell command) is kept as-is,
    truncated. Never raises — a bad argument blob just yields a shorter line.
    """
    if isinstance(inp, str):
        try:
            inp = json.loads(inp)
        except (json.JSONDecodeError, ValueError):
            return inp[:140]
    return _summarize_input(inp) if isinstance(inp, dict) else ""


def _summarize_input(inp) -> str:
    """One short line describing a tool call's arguments, for the Actions list.

    A tool's real input can be a whole file's contents or a 200-line patch; a trail needs
    a glance, not the payload. So: take the first of a few well-known "what was this
    about" keys, in priority order (what was run > which file > what was searched for >
    prose > a URL), and cap it at 140 characters —
        {"command": "pytest -q tests/"}          -> "command=pytest -q tests/"
        {"file_path": "app.py", "old": "…"}      -> "file_path=app.py"
    When no known key is present, fall back to naming the first five keys so the reader
    at least sees the shape. Returns "" for anything that is not a dict.
    """
    if not isinstance(inp, dict):
        return ""
    for key in ("command", "cmd", "file_path", "path", "pattern", "query", "description", "url"):
        if key in inp:
            return f"{key}={str(inp[key])[:140]}"
    return ", ".join(list(inp.keys())[:5])


# --- rendering ----------------------------------------------------------------
def render_markdown(steps: list[ReasoningStep], header: dict) -> str:
    """Turn steps + a session header into the Markdown page stored under readable/.

    `header` is the adapter's parsed header as a plain dict (session_id, title,
    first_message, cli_source, model_used, folder_name, start_time, last_activity); every
    field is optional here, so a partially-parsed header still renders.

    The leading note is SOURCE-AWARE, and that is the point: a trail with no thinking text
    means something different for each CLI, and a single generic apology would be wrong
    three ways. If any turn has thinking text, say so plainly. If not and the source is
    Claude Code, explain that it stores thinking blocks without their text. Otherwise name
    the actual source in the note, so a Codex or OpenCode reader is not told about a
    Claude limitation that has nothing to do with their file.

    The whole page goes through redact.redact() before being returned — see the comment
    at the end of the function. Per-turn text is capped at 4000 characters so one
    runaway turn cannot produce a megabyte of Markdown.
    """
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
        # 🔒 means "the model thought here, but the CLI did not keep the text". Only when
        # there is no thinking text to show — otherwise the reader has the real thing.
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
    """Render text as a Markdown blockquote, one "> " per line.

    Quoting matters because the quoted text is itself Markdown the model wrote — headings
    and code fences inside it would otherwise restructure the trail's own page. Blank
    lines become a bare ">" so the quote is not broken into separate blocks.
    """
    return "\n".join("> " + ln if ln else ">" for ln in text.splitlines())


# --- archive + persist --------------------------------------------------------
def _slug(text: str) -> str:
    """A short filename-safe tag from a session title, e.g. "Fix flaky checkout tests"
    -> "fix-flaky-checkout-tests".

    The regex replaces every run of non-alphanumeric characters with a single hyphen, so
    spaces, punctuation, emoji and non-Latin scripts all collapse safely; leading and
    trailing hyphens are trimmed and the result capped at 60 characters. Falls back to
    "session" when nothing survives (an all-emoji title, an empty one). Purely cosmetic —
    the session id in front of it is what identifies the file.
    """
    s = re.sub(r"[^a-zA-Z0-9]+", "-", (text or "").lower()).strip("-")
    return s[:60] or "session"


# Matches a timestamp that begins with "YYYY-MM", e.g. "2026-09-11T09:00:00.000Z".
_YM = re.compile(r"^\d{4}-\d{2}")


def _ym_dir(base: Path, last_activity: str) -> Path:
    """The YYYY/MM subdirectory of `base` for a session, created if missing.

    Sessions are filed by the month they were last active so the vault stays browsable
    (and a month's worth can be pruned or moved as a unit).

    The guard is the important part. `last_activity` comes from a transcript, so it can
    be empty, a bare year, or something unexpected. Anything that is not "YYYY-MM…" is
    filed under the sentinel "0000/00" instead — see the existing note below for why the
    shape must be exactly two path components.
    """
    # Exactly two components: the raw-copy glob is */*/*.jsonl, so anything
    # else would be archived where nothing looks (unrestorable, silently).
    head = (last_activity or "")[:7]
    ym = head.replace("-", "/") if _YM.match(head) else "0000/00"
    d = base / ym
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe_sid(sid) -> str:
    """session_id is file CONTENT (a rollout's payload.id, a mirror header) used
    as a path component: refuse anything that could leave the vault.

    Nothing validates a session id before it reaches us — it is read out of a file we did
    not write. Used unchecked as a filename, a value like "../../.ssh/id_rsa" would make
    archive_raw() write outside the archive. So: reject empty ids, anything containing a
    forward or back slash, "." and "..", and anything starting with "..". Raises
    ValueError, which the callers let propagate — refusing to archive is the correct
    outcome, and the driver reports it per file without stopping the run.
    """
    sid = str(sid or "")
    if not sid or "/" in sid or "\\" in sid or sid in (".", "..") or sid.startswith(".."):
        raise ValueError(f"refusing to archive with an unsafe session id {sid!r}")
    return sid


# Raw copies keep the source's representation: Codex zstd-compresses cold
# rollouts in place, and a byte copy named .jsonl would be a zstd frame every
# reader parses as text. The codex adapter reads either suffix transparently.
_ZST = ".zst"


def _raw_suffix(src: Path) -> str:
    """".jsonl" normally, ".jsonl.zst" when the source is already compressed."""
    return ".jsonl" + _ZST if src.name.endswith(_ZST) else ".jsonl"


def archive_raw(transcript_path: Path, header: dict) -> Path:
    """Copy the raw transcript into the archive, versioning on content change.

    This is the durable transcript vault: `refresh-all` calls it for every indexable
    transcript before extracting reasoning, and the OpenCode adapter calls it just before
    unlinking a mirror file. What lands here is what restore.py can put back later, and
    what build-fts.py indexes once the original is gone.

    Destination: <archive>/raw/YYYY/MM/<session_id>.jsonl (or .jsonl.zst). Returns the
    path written or reused.

    Versioning — the "@vN" in the name. Sessions grow: the same transcript is archived
    again the next night with more turns in it. Overwriting would throw away the older
    state; writing a new file every time would balloon the vault. So:
      * destination absent            -> write it
      * destination exists, same SIZE -> assume unchanged, return it, copy nothing
      * destination exists, different -> write <sid>@v2.jsonl, then @v3, @v4 …
    Size, not a content hash, is the change test: transcripts are append-only, so a
    changed transcript is a longer one, and comparing sizes costs one stat() instead of
    re-reading megabytes. That makes the call idempotent, which matters because the
    OpenCode plugin fires on every idle turn.

    copy2 (not copyfile) here, deliberately the opposite of restore.py: the vault copy
    must keep the ORIGINAL modification time, because _raw_copies() orders versions by
    mtime to decide which copy is newest.

    Raises ValueError via _safe_sid() for an unsafe session id, and OSError if the copy
    itself fails; callers report per file rather than aborting a batch.
    """
    src = Path(transcript_path)
    dest_dir = _ym_dir(ARCHIVE / "raw", header.get("last_activity", ""))
    # Fall back to the file's own stem when the header has no id — "<uuid>.jsonl" and
    # "<uuid>.jsonl.zst" both yield "<uuid>" because split(".") takes the first part.
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


# Splits an archived copy's stem back into (session id, version number):
#   "ses_abc"      -> sid="ses_abc",  v=None (treated as version 1)
#   "ses_abc@v10"  -> sid="ses_abc",  v="10"
# The `.+?` is non-greedy so the LAST "@vN" is the one peeled off, leaving ids that
# themselves contain "@" intact.
_RAW_STEM = re.compile(r"^(?P<sid>.+?)(?:@v(?P<v>\d+))?$")


def _raw_copies():
    """Every raw transcript copy as (session_id, mtime, path). One walk. The
    @vN counter restarts per YYYY/MM directory, so ordering is by the copy's
    mtime (copy2 preserves the source's) — a September @v3 must not outrank
    October's first copy.

    A generator yielding `(sid, (mtime, version), path)` — note the middle element is a
    TUPLE, and that tuple is the sort key. Python compares tuples element by element, so
    the newest modification time wins first and the @vN number only breaks ties. Both
    callers below simply take the maximum.

    Missing archive directory -> yields nothing (the vault may not exist yet). A file that
    disappears between the glob and the stat() is skipped rather than raising: the
    nightly job and this walk can run at the same time.
    """
    raw = ARCHIVE / "raw"
    if not raw.is_dir():
        return
    # Two globs, one pass each: plain copies and zstd-compressed ones. The pattern is
    # exactly */*/ — year, then month — which is why _ym_dir() must never produce a
    # deeper or shallower path.
    for pattern in ("*/*/*.jsonl", f"*/*/*.jsonl{_ZST}"):
        for p in raw.glob(pattern):
            # Strip the suffix to get the stem. Path.stem only removes the LAST
            # extension, so "x.jsonl.zst".stem is "x.jsonl" — hence the manual slice.
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
    directory walk, so the UI can label 500 archived rows without 500 walks.

    Used by restore.plan() and build-fts.py's index_archived(), both of which need the
    answer for many sessions at once. The path string is appended to the sort key purely
    as a deterministic tie-break, so two copies with the same mtime and version always
    resolve the same way rather than depending on directory order.
    """
    best: dict[str, tuple[tuple, Path]] = {}
    for sid, key, p in _raw_copies():
        if sid not in best or (key, str(p)) > (best[sid][0], str(best[sid][1])):
            best[sid] = (key, p)
    return {sid: p for sid, (_, p) in best.items()}


def find_archived_raw(session_id: str) -> Path | None:
    """Newest raw copy of one session's transcript, or None. This is what makes
    an aged-out session restorable after Claude Code's cleanup deleted the
    original: refresh-all copies every indexable transcript here first.

    The single-session counterpart of archived_raw_index(); restore.py calls it. The
    triple (key, path-string, path) is built so `max()` compares the sort key first and
    only falls back to the path string — the trailing `[2]` then picks the Path itself.
    """
    hits = [(key, str(p), p) for sid, key, p in _raw_copies() if sid == session_id]
    return max(hits)[2] if hits else None


def write_readable(steps: list[ReasoningStep], header: dict) -> Path:
    """Render the trail and write it to <archive>/readable/YYYY/MM/<sid>-<slug>.md.

    Returns the path written. Writing is ATOMIC — the Markdown goes to a temporary file
    in the same directory and is then moved into place with os.replace(), which on a
    POSIX filesystem either fully succeeds or leaves the previous file untouched. A
    reader (the UI serving the trail) therefore never sees a half-written page, and a
    crash mid-write cannot destroy the old one. The temp file is removed on any failure.

    After the new file is safely in place, the session's OLDER renders are deleted, so
    the archive holds exactly one trail per session even though the filename changes
    whenever the title or the month changes.

    Raises ValueError for an unsafe session id, or OSError if the write fails.
    """
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
    # Why prefixes collide: OpenCode ids are "ses_" plus a millisecond clock and Codex
    # uses time-ordered UUIDv7, so two sessions started the same minute share their first
    # eight characters — sweeping on a prefix deleted a stranger's trail. The glob is
    # "*/*/<full id>-*.md": every month directory, every title slug, this session only.
    for old in (ARCHIVE / "readable").glob(f"*/*/{sid}-*.md"):
        if old != dest:
            old.unlink(missing_ok=True)
    return dest


def persist(session_id: str, steps: list[ReasoningStep], readable_path: Path,
            conn=None) -> None:
    """Record reasoning_path on the session and store per-step artifacts.

    Two writes into registry.db:
      * sessions.reasoning_path — where the UI finds the Markdown trail for this row.
      * session_artifacts — one row per turn that had text, type='reasoning', so search
        can find a session by something the model only said inside its reasoning.
    The artifact rows are deleted and re-inserted, which makes re-running the extractor
    idempotent instead of duplicating every turn.

    Pass `conn` to join a caller's transaction (the backfill does, and commits per file);
    with conn=None the connection is opened, committed and closed here. Tests call it as
    persist(sid, steps, path, conn=conn).

    `indexer` is imported inside the function, not at module level: this is the only
    function here that touches the database, while the extractors and archive_raw() are
    imported on hot paths (the Stop hook, the OpenCode adapter) that must not pay for the
    database layer just to copy a file.
    """
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
                # Containment check before deleting anything: relative_to() raises
                # ValueError unless the recorded path really is inside the readable
                # archive. The stored value came from a database that may have been
                # written by another machine or an older version, so it is not trusted
                # to point somewhere we are allowed to unlink. Both failure modes
                # (outside the archive, or an I/O error) mean "leave it alone".
                prev_path.resolve().relative_to((ARCHIVE / "readable").resolve())
                prev_path.unlink(missing_ok=True)
            except (ValueError, OSError):
                pass
        conn.execute(
            "UPDATE sessions SET reasoning_path = ? WHERE session_id = ?",
            (str(readable_path), session_id),
        )
        # Clear this session's previous reasoning artifacts before re-inserting, so a
        # re-run replaces the turns instead of appending a second copy of them. Only
        # type='reasoning' rows — other artifact kinds belong to other producers.
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
            # Redact before storing, and cap at 8000 characters: these rows are searched
            # and shown, so they are an egress point like any other, and one enormous
            # turn should not bloat the registry.
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
