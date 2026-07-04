#!/usr/bin/env python3
"""session-memory MCP server.

Exposes the Session Browser registry to Claude as tools so it can recall and
reason over past sessions across CLIs. Five core tools plus get_reasoning — a
deliberate extension surfacing the project's headline decision-trail feature.

Run over stdio:  <venv-python> mcp/session-memory/server.py
"""
from __future__ import annotations

import json
from pathlib import Path

from mcp.server.fastmcp import FastMCP

import common  # noqa: E402  (same dir)

mcp = FastMCP("session-memory")


@mcp.tool()
def search_sessions(query: str, limit: int = 5) -> list[dict]:
    """Search past sessions (semantic, falling back to keyword). Returns compact
    session descriptors: session_id, summary, topics, last_activity, folder_name."""
    ids, scores = common.semantic_or_keyword(query, limit)
    if not ids:
        return []
    conn = common.connect()
    try:
        ph = ",".join("?" * len(ids))
        rows = conn.execute(
            f"SELECT session_id, summary, title, topics, last_activity, folder_name, cli_source "
            f"FROM sessions WHERE session_id IN ({ph})", ids
        ).fetchall()
    finally:
        conn.close()
    by_id = {r["session_id"]: r for r in rows}
    out = []
    for sid in ids:
        r = by_id.get(sid)
        if not r:
            continue
        d = {
            "session_id": sid,
            "summary": (r["summary"] or r["title"] or "")[:160],
            "topics": _json(r["topics"]),
            "last_activity": r["last_activity"],
            "folder_name": r["folder_name"],
            "cli_source": r["cli_source"],
        }
        if sid in scores:
            d["_score"] = scores[sid]
        out.append(d)
    return common.clamp(out)


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
        decisions = [
            row[0][:200] for row in conn.execute(
                "SELECT content FROM session_artifacts WHERE session_id = ? AND type='decision' "
                "ORDER BY turn_index LIMIT 5", (session_id,)
            ).fetchall()
        ]
    finally:
        conn.close()
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


@mcp.tool()
def get_session_snippet(session_id: str, query: str) -> list[dict]:
    """Find up to 3 relevant snippets within a session by matching its artifacts
    (reasoning/decisions) against a query."""
    conn = common.connect()
    try:
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


@mcp.tool()
def list_recent(folder: str = "", days: int = 7) -> list[dict]:
    """List sessions active within the last N days, optionally filtered by folder."""
    conn = common.connect()
    try:
        sql = ("SELECT session_id, title, summary, folder_name, last_activity, cli_source "
               "FROM sessions WHERE archived = 0 AND last_activity >= datetime('now', ?)")
        params = [f"-{int(days)} days"]
        if folder:
            sql += " AND folder_name = ?"
            params.append(folder)
        sql += " ORDER BY last_activity DESC LIMIT 20"
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    return common.clamp([
        {"session_id": r["session_id"], "title": r["title"] or (r["summary"] or "")[:80],
         "folder_name": r["folder_name"], "last_activity": r["last_activity"],
         "cli_source": r["cli_source"]}
        for r in rows
    ])


@mcp.tool()
def get_decisions(topic: str, limit: int = 10) -> list[dict]:
    """Decisions extracted from sessions tagged with a given topic."""
    conn = common.connect()
    try:
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


@mcp.tool()
def get_reasoning(session_id: str, query: str = "") -> dict:
    """The decision/reasoning trail for a session — how Claude reached its
    decisions. Returns the rendered Markdown trail (or, with a query, the matching
    reasoning steps). Note: Claude Code stores hidden extended-thinking text empty;
    this surfaces the visible reasoning + action sequence (Copilot includes its
    reasoning text)."""
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
    if not r or not r["reasoning_path"] or not Path(r["reasoning_path"]).exists():
        return {"error": "no reasoning trail; run extract-reasoning.py for this session"}
    md = Path(r["reasoning_path"]).read_text(encoding="utf-8")
    return common.sanitize({"session_id": session_id, "markdown": md[:6000],
                            "truncated": len(md) > 6000, "path": r["reasoning_path"]})


def _json(s):
    try:
        return json.loads(s) if s else []
    except (json.JSONDecodeError, TypeError):
        return []


if __name__ == "__main__":
    mcp.run()
