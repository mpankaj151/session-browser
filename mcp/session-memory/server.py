#!/usr/bin/env python3
"""session-memory MCP server.

Exposes the Session Browser registry to the coding CLI as tools so it can recall and
reason over past sessions across CLIs. Five core tools plus get_reasoning — a
deliberate extension surfacing the project's headline decision-trail feature.

Run over stdio:  <venv-python> mcp/session-memory/server.py

What MCP is
-----------
MCP (Model Context Protocol) is a standard way for an AI assistant to call external tools.
A *server* like this one advertises a list of tools — each with a name, a description and
typed arguments — and the assistant's CLI decides when to call one and feeds the result
back into the conversation. So after registering this server, you can ask Claude Code
"what did I decide about the archive lifecycle last week?" and it will call `list_recent`
and `get_decisions` here instead of guessing.

How it is run
-------------
Over **stdio**: the CLI starts this file as a child process and speaks JSON-RPC over its
standard input and output. There is no port, no URL, and no `--help` — running it in a
terminal looks like a hang, because it is correctly waiting for a client to say something.
Register it instead, e.g. `claude mcp add session-memory -- <venv-python>
mcp/session-memory/server.py` (docs/SETUP.md has the incantation for each CLI). Never
print to standard output from anywhere in this file: stdout IS the protocol channel.

The six tools, all read-only:
  search_sessions(query, limit=5)          find sessions by meaning, else by keyword
  get_session_summary(session_id)          one session's metadata plus its decisions
  get_session_snippet(session_id, query)   matching fragments inside one session
  list_recent(folder="", days=7)           what was worked on lately
  get_decisions(topic, limit=10)           decisions across every session with a topic tag
  get_reasoning(session_id, query="")      the session's decision trail, whole or searched

Contracts that apply to all six:
  * **Every result is redacted.** A tool result is not read by a human — it is inserted
    into the assistant's conversation and therefore sent to the model provider. common.py's
    `sanitize`/`clamp` mask API keys, tokens and passwords before anything is returned.
  * **Every result is capped** (common.MAX_BYTES), because the reply is spent out of the
    assistant's context window. A trimmed list carries a `_truncated` marker so the
    assistant knows there was more.
  * **Visibility.** Queries compose `indexer.VISIBLE` — live sessions plus real ones whose
    transcript aged out — so subagent side-conversations never surface as results.
  * **Failure is a returned value, not a crash**: a missing session is
    `{"error": "not found"}`, not an exception, so the assistant can say so and move on.

Vocabulary: **session** (one conversation with a coding CLI), **transcript**, **resume**,
**token**, **reasoning trail** — all defined in docs/GLOSSARY.md.

Nothing here writes to the registry; see session-ui/app.py for the surfaces that do.
Tests: tests/test_smoke.py loads this file as a module, points `common.DB_PATH` at a
temporary registry and calls the tool functions directly (`test_mcp_*`).
"""
from __future__ import annotations

import json
from pathlib import Path

# FastMCP is the 1.x SDK's decorator-based server: `@mcp.tool()` turns a plain function
# into an advertised tool, deriving its name, argument schema and description from the
# signature, type annotations and docstring. Hence the version pin in requirements.txt —
# `mcp>=1.0,<2`; 2.x renamed the class and changed the tool API, so this import is the one
# place that would break, and it is caught here.
#
# SystemExit rather than a traceback: the CLI starts this file as a child process, and a
# stack trace in its log tells the user nothing actionable. This prints one sentence with
# the fix in it.
try:
    from mcp.server.fastmcp import FastMCP
except ImportError as e:  # mcp 2.x renamed FastMCP to MCPServer and changed the tool API
    raise SystemExit("session-memory needs the mcp 1.x SDK: pip install 'mcp>=1.0,<2' "
                     f"(installed SDK lacks mcp.server.fastmcp: {e})") from e

# `common` is next door in this same directory, and importing it is also what puts the
# repo root on sys.path — which is what makes the `indexer` import below resolve at all.
# Order matters here: swapping these two lines breaks the server.
import common  # noqa: E402  (same dir — also puts the repo root on sys.path)
import indexer  # noqa: E402

# The server object. "session-memory" is the name the client shows and the prefix its
# tools appear under; every @mcp.tool() below attaches to it, and mcp.run() at the bottom
# is what actually starts serving.
mcp = FastMCP("session-memory")


# A note on the docstrings below: for an @mcp.tool() function the docstring is not just
# documentation, it is the tool's DESCRIPTION — the SDK ships it to the assistant, which
# reads it to decide when to call the tool and is billed for the words. So they stay short
# and behavioural (what it does, what comes back, what to watch out for), and the
# explanation for a human reading the code lives in ordinary comments like this one.


# Returns compact descriptors, not whole sessions: this is the "which conversation was
# that?" tool, meant to be followed by get_session_summary or get_reasoning on the one id
# that looks right. Ordering is whatever common.semantic_or_keyword ranked, best first.
@mcp.tool()
def search_sessions(query: str, limit: int = 5) -> list[dict]:
    """Search past sessions (semantic, falling back to keyword). Returns compact
    session descriptors: session_id, summary, topics, last_activity, folder_name."""
    ids, scores = common.semantic_or_keyword(query, limit)
    if not ids:
        return []
    conn = common.connect()
    try:
        # One query for the whole shortlist: "?,?,?" placeholders, one per id, all bound
        # as parameters rather than pasted into the SQL. Named columns, not SELECT * —
        # every extra column would be bytes out of the caller's budget.
        ph = ",".join("?" * len(ids))
        rows = conn.execute(
            f"SELECT session_id, summary, title, topics, last_activity, folder_name, cli_source, archived "
            f"FROM sessions WHERE session_id IN ({ph})", ids
        ).fetchall()
    finally:
        conn.close()
    # SQL returns rows in no particular order, so index them by id and walk `ids` instead:
    # that is what preserves the search ranking.
    by_id = {r["session_id"]: r for r in rows}
    out = []
    for sid in ids:
        r = by_id.get(sid)
        if not r:
            # An id the search knew about but the registry no longer has (an embedding
            # left behind by a deleted row). Skip it rather than returning a hole.
            continue
        d = {
            "session_id": sid,
            # Summary if enrichment wrote one, else the title, else nothing — and cut to
            # 160 characters, which is a line of context, not a paragraph of it.
            "summary": (r["summary"] or r["title"] or "")[:160],
            "topics": _json(r["topics"]),
            "last_activity": r["last_activity"],
            "folder_name": r["folder_name"],
            "cli_source": r["cli_source"],
            # An aged-out session has no transcript to resume: say so, or the
            # consuming agent suggests `cr <id>` for a file that no longer exists.
            "archived": bool(r["archived"]),
        }
        # Only present on the semantic path (0..1, higher = closer in meaning); a keyword
        # match has no score, and an absent key says that more honestly than a fake 1.0.
        if sid in scores:
            d["_score"] = scores[sid]
        out.append(d)
    # clamp() redacts and trims to the byte budget, appending a `_truncated` marker if it
    # had to cut. Every list-returning tool here ends this way.
    return common.clamp(out)


# The follow-up to search_sessions: everything worth knowing about one session without
# opening its transcript. `outcome` and `session_type` are enrichment's own labels (e.g.
# "completed" / "debugging"); `turn_count` is how many user messages it held; `decisions`
# come from session_artifacts, the per-turn items scripts/extract-reasoning.py pulled out.
# Returns {"error": "not found"} rather than raising — the assistant can then say so.
@mcp.tool()
def get_session_summary(session_id: str) -> dict:
    """Full metadata for one session: summary, topics, outcome, session_type,
    turn_count, and up to 5 extracted decisions."""
    conn = common.connect()
    try:
        r = conn.execute(
            "SELECT summary, title, topics, outcome, session_type, turn_count, "
            "model_used, cost_usd FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if not r:
            return {"error": "not found"}
        # Oldest first (turn_index ascending) so the decisions read in the order they were
        # made; 200 characters each is a decision, not the discussion around it.
        decisions = [
            row[0][:200] for row in conn.execute(
                "SELECT content FROM session_artifacts WHERE session_id = ? AND type='decision' "
                "ORDER BY turn_index LIMIT 5", (session_id,)
            ).fetchall()
        ]
    finally:
        conn.close()
    # A dict, so clamp() does not apply: redact explicitly. The fields are short and
    # bounded above, so there is nothing to trim.
    return common.sanitize({
        "session_id": session_id,
        "summary": r["summary"] or r["title"],
        "topics": _json(r["topics"]),
        "outcome": r["outcome"],
        "session_type": r["session_type"],
        "turn_count": r["turn_count"],
        "model_used": r["model_used"],
        "cost_usd": r["cost_usd"],
        "decisions": decisions,
    })


# Zoom in: "in THAT session, what was said about X?". Searches the extracted artifacts
# (reasoning steps and decisions), not the raw transcript — so it is one indexed lookup
# rather than parsing a multi-megabyte file, at the cost of only seeing what extraction
# kept. Returns [] when nothing matches, which is a valid answer, not a failure.
@mcp.tool()
def get_session_snippet(session_id: str, query: str) -> list[dict]:
    """Find up to 3 relevant snippets within a session by matching its artifacts
    (reasoning/decisions) against a query."""
    conn = common.connect()
    try:
        # Substring match, oldest first, at most 3. `type` is returned so the caller can
        # tell a recorded decision from a passing thought.
        like = f"%{query}%"
        rows = conn.execute(
            "SELECT type, content, turn_index FROM session_artifacts "
            "WHERE session_id = ? AND content LIKE ? ORDER BY turn_index LIMIT 3",
            (session_id, like),
        ).fetchall()
    finally:
        conn.close()
    return common.clamp([
        {"type": r["type"], "turn": r["turn_index"], "content": r["content"][:400]}
        for r in rows
    ])


# The "what have I been working on?" tool — the one behind questions like "summarise last
# week". The window is measured against `last_activity` (when the session was last
# touched), not when it started, so a long-running conversation counts as recent for as
# long as it is being used. Newest first, at most 20.
@mcp.tool()
def list_recent(folder: str = "", days: int = 7) -> list[dict]:
    """List sessions active within the last N days, optionally filtered by folder."""
    conn = common.connect()
    try:
        # last_activity is stored as TEXT, so the cutoff is compared as a string and must
        # be rendered in exactly the spelling the column uses. strftime() with that format
        # builds e.g. '2026-09-04T00:00:00.000Z' for ?='-7 days'.
        #
        # strftime with the column's own 'T' spelling: datetime('now', ...) yields
        # 'YYYY-MM-DD HH:MM:SS', which sorts below every row of the cutoff day.
        # (A space sorts before 'T', so that form would silently include up to an extra
        # day of sessions — the same bug the UI hit; see session-ui/app.py.)
        sql = ("SELECT session_id, title, summary, folder_name, last_activity, cli_source, archived "
               f"FROM sessions WHERE {indexer.VISIBLE} AND "
               "last_activity >= strftime('%Y-%m-%dT%H:%M:%S.000Z', 'now', ?)")
        # int(days) before interpolating into the modifier string: it is the one place a
        # caller's value reaches SQL as text rather than as a bound parameter, and a
        # non-integer would raise here rather than reaching SQLite.
        params = [f"-{int(days)} days"]
        if folder:
            sql += " AND folder_name = ?"
            params.append(folder)
        sql += " ORDER BY last_activity DESC LIMIT 20"
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    # `archived` travels with every descriptor: an aged-out session has no transcript
    # left, so an assistant that did not know would happily suggest resuming it.
    return common.clamp([
        {"session_id": r["session_id"], "title": r["title"] or (r["summary"] or "")[:80],
         "folder_name": r["folder_name"], "last_activity": r["last_activity"],
         "cli_source": r["cli_source"], "archived": bool(r["archived"])}
        for r in rows
    ])


# Across sessions rather than within one: "what have I decided about testing?". A topic is
# a short keyword tag ('python', 'ci-cd') attached to a session by enrichment or by
# scripts/classify-topics.py.
@mcp.tool()
def get_decisions(topic: str, limit: int = 10) -> list[dict]:
    """Decisions extracted from sessions tagged with a given topic."""
    conn = common.connect()
    try:
        # Join the artifacts to their sessions so the tag can be matched on the session and
        # the results ordered by when that session was last active — newest thinking first.
        # topics is JSON text like '["python","testing"]', so the tag is matched with its
        # surrounding quotes: '%"ci"%' hits '["ci"]' but not '["ci-cd"]'.
        rows = conn.execute(
            "SELECT a.session_id, a.content, a.turn_index FROM session_artifacts a "
            "JOIN sessions s ON s.session_id = a.session_id "
            "WHERE a.type = 'decision' AND s.topics LIKE ? ORDER BY s.last_activity DESC LIMIT ?",
            (f'%"{topic}"%', limit),
        ).fetchall()
    finally:
        conn.close()
    return common.clamp([
        {"session_id": r["session_id"], "turn": r["turn_index"], "decision": r["content"][:200]}
        for r in rows
    ])


# This project's headline feature, exposed as a tool: not just WHAT a past session did but
# WHY. scripts/extract-reasoning.py renders each session's thinking-plus-actions into a
# Markdown file under the reasoning archive; the row keeps only the path in
# `reasoning_path`.
#
# Two modes in one function, which is why the control flow below is worth reading slowly:
#   * with a `query`, it returns matching reasoning STEPS from session_artifacts (an
#     indexed lookup) and returns from inside the try block, so the `finally` closes the
#     connection on the way out;
#   * with no `query`, it falls through past the try/finally — connection already closed —
#     and reads the whole rendered Markdown file from disk.
@mcp.tool()
def get_reasoning(session_id: str, query: str = "") -> dict:
    """The decision/reasoning trail for a session — how the coding CLI reached
    its decisions. Returns the rendered Markdown trail (or, with a query, the
    matching reasoning steps). Note: Claude Code stores hidden extended-thinking
    text empty; this surfaces the visible reasoning + action sequence (Copilot
    and OpenCode include their reasoning text)."""
    conn = common.connect()
    try:
        r = conn.execute(
            "SELECT reasoning_path FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if query:
            like = f"%{query}%"
            steps = [
                {"turn": row["turn_index"], "content": row["content"][:400]}
                for row in conn.execute(
                    "SELECT content, turn_index FROM session_artifacts "
                    "WHERE session_id = ? AND type='reasoning' AND content LIKE ? "
                    "ORDER BY turn_index LIMIT 5", (session_id, like)
                ).fetchall()
            ]
            return {"session_id": session_id, "steps": common.clamp(steps)}
    finally:
        conn.close()
    # Three distinct misses, one answer, because the remedy is the same in all of them: no
    # such session, a session nothing was ever extracted for, or a recorded path whose file
    # has since been removed. The message names the script that fixes it.
    if not r or not r["reasoning_path"] or not Path(r["reasoning_path"]).exists():
        return {"error": "no reasoning trail; run extract-reasoning.py for this session"}
    # errors="replace" rather than strict: a single bad byte in a years-old trail must
    # render as a replacement character, not turn the whole tool call into an error.
    md = Path(r["reasoning_path"]).read_text(encoding="utf-8", errors="replace")
    # 6000 characters is this tool's own budget (trails run to tens of thousands), and
    # `truncated` tells the assistant it is reading a beginning, not a whole. `path` lets
    # it open the rest itself with its ordinary file-reading tools if it needs to.
    return common.sanitize({"session_id": session_id, "markdown": md[:6000],
                            "truncated": len(md) > 6000, "path": r["reasoning_path"]})


def _json(s):
    """Decode a JSON-text column (topics is stored as '["python","testing"]') into a list.

    SQLite has no array type, hence the encoding. NULL, empty or malformed text all become
    [] — one bad row must not fail a tool call, and every caller wants a list.
    """
    try:
        return json.loads(s) if s else []
    except (json.JSONDecodeError, TypeError):
        return []


if __name__ == "__main__":
    # Blocks forever, reading JSON-RPC requests from standard input and writing replies to
    # standard output, until the client that launched this process closes the pipe. This
    # is why running the file by hand looks like a hang: it is waiting for a client.
    mcp.run()
