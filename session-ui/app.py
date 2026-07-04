#!/usr/bin/env python3
"""Flask backend for the Session Browser.

Serves the single-file SPA and a REST API over registry.db. Endpoints are added
slice by slice; this module is the home for all of them.
"""
from __future__ import annotations

import json
import re
import shlex
import sys
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import indexer  # noqa: E402
import redact as _redact  # noqa: E402
import sbconfig  # noqa: E402
from sources.registry import build_source_registry  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"
HOST = sbconfig.CONFIG["ui"]["host"]
PORT = int(sbconfig.CONFIG["ui"]["port"])

app = Flask(__name__, static_folder=None)
SOURCES = build_source_registry()


# --- helpers ------------------------------------------------------------------
def _row_to_dict(row) -> dict:
    d = dict(row)
    # topics / models_used are JSON-encoded text columns.
    for key in ("topics", "models_used"):
        if d.get(key):
            try:
                d[key] = json.loads(d[key])
            except (json.JSONDecodeError, TypeError):
                d[key] = []
        else:
            d[key] = []
    d["is_active"] = _is_active(d.get("last_activity"))
    d["has_reasoning"] = bool(d.get("reasoning_path"))
    d["cost"] = {
        "usd": d.get("cost_usd"),
        "input": d.get("input_tokens"), "output": d.get("output_tokens"),
        "cache_read": d.get("cache_read_tokens"), "cache_write": d.get("cache_write_tokens"),
        "notional": sbconfig.COST_IS_NOTIONAL,
    }
    return d


def _is_active(last_activity: str | None) -> bool:
    if not last_activity:
        return False
    try:
        from datetime import datetime, timezone
        ts = datetime.fromisoformat(last_activity.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - ts).total_seconds() < 7200
    except (ValueError, TypeError):
        return False


# --- routes -------------------------------------------------------------------
@app.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/api/sessions")
def api_sessions():
    search = (request.args.get("search") or "").strip()
    folder = request.args.get("folder") or ""
    source = request.args.get("source") or ""
    topic = request.args.get("topic") or ""
    days = request.args.get("days")
    mode = request.args.get("mode") or ""

    # Semantic mode: rank by embedding similarity, then apply the same filters.
    sem_ids: list[str] | None = None
    if mode == "semantic" and search:
        try:
            import semsearch
            hits = semsearch.search(search, limit=40)
            sem_ids = [sid for sid, _ in hits]
        except Exception:  # noqa: BLE001 — fall back to keyword on any failure
            sem_ids = None
    # Full-text mode: match against transcript body via FTS5.
    if mode == "fulltext" and search:
        toks = re.findall(r"\w+", search)
        if toks:
            conn0 = indexer.connect()
            try:
                rows0 = conn0.execute(
                    "SELECT session_id FROM sessions_fts WHERE sessions_fts MATCH ? LIMIT 200",
                    (" ".join(toks),)).fetchall()
                sem_ids = [r[0] for r in rows0]
            except Exception:  # noqa: BLE001 — FTS missing/not built -> keyword fallback
                sem_ids = None
            finally:
                conn0.close()

    where = ["archived = 0"]
    params: list = []
    if folder:
        where.append("folder_name = ?")
        params.append(folder)
    if source and source != "all":
        where.append("cli_source = ?")
        params.append(source)
    if topic:
        where.append("topics LIKE ?")
        params.append(f'%"{topic}"%')
    if days:
        try:
            where.append("last_activity >= datetime('now', ?)")
            params.append(f"-{int(days)} days")
        except ValueError:
            pass
    if sem_ids is not None:
        if not sem_ids:
            return jsonify([])
        placeholders = ",".join("?" * len(sem_ids))
        where.append(f"session_id IN ({placeholders})")
        params += sem_ids
    elif search:
        where.append("(first_message LIKE ? OR summary LIKE ? OR title LIKE ? OR topics LIKE ?)")
        like = f"%{search}%"
        params += [like, like, like, like]

    order = "last_activity DESC"
    sql = "SELECT * FROM sessions WHERE " + " AND ".join(where) + f" ORDER BY {order} LIMIT 500"
    conn = indexer.connect()
    try:
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    results = [_row_to_dict(r) for r in rows]
    # In semantic mode, preserve similarity ranking from sem_ids.
    if sem_ids is not None:
        rank = {sid: i for i, sid in enumerate(sem_ids)}
        results.sort(key=lambda d: rank.get(d["session_id"], 1e9))
    return jsonify(results)


@app.get("/api/sessions/folders")
def api_folders():
    conn = indexer.connect()
    try:
        rows = conn.execute(
            "SELECT DISTINCT folder_name FROM sessions "
            "WHERE archived = 0 AND folder_name <> '' ORDER BY folder_name"
        ).fetchall()
    finally:
        conn.close()
    return jsonify([r[0] for r in rows])


@app.get("/api/sessions/topics")
def api_topics():
    conn = indexer.connect()
    try:
        rows = conn.execute(
            "SELECT topics FROM sessions WHERE archived = 0 AND topics IS NOT NULL"
        ).fetchall()
    finally:
        conn.close()
    seen: set[str] = set()
    for r in rows:
        try:
            for t in json.loads(r[0]):
                seen.add(t)
        except (json.JSONDecodeError, TypeError):
            continue
    return jsonify(sorted(seen))


@app.get("/api/sessions/<sid>/resume")
def api_resume(sid: str):
    conn = indexer.connect()
    try:
        row = conn.execute(
            "SELECT cli_source, cwd FROM sessions WHERE session_id = ?", (sid,)
        ).fetchone()
    finally:
        conn.close()
    if not row:
        return jsonify({"error": "not found"}), 404
    src = SOURCES.get(row["cli_source"])
    raw = src.resume_command(sid) if src else f"# unknown source {row['cli_source']}"
    cwd = row["cwd"] or ""
    # Primary: the `cr` shell shortcut (installed via bin/install-cr.sh). Paste it in
    # the directory where you want to continue — it ports the session's memory there
    # and resumes. command_full is the direct script call if `cr` isn't installed.
    wrapper = _REPO / "bin" / "resume-here.sh"
    command = f"cr {sid}"
    command_full = f'{shlex.quote(str(wrapper))} {sid} {row["cli_source"]}'
    return jsonify({"command": command, "command_full": command_full,
                    "raw_command": raw, "origin_cwd": cwd, "cli_source": row["cli_source"]})


def _build_context(conn, sid: str) -> tuple[str, str] | None:
    """Build a portable context primer (markdown) from indexed metadata — fast,
    no transcript parse. Designed to paste into a new session to carry work over."""
    r = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (sid,)).fetchone()
    if not r:
        return None
    d = _row_to_dict(r)
    decisions = [row[0] for row in conn.execute(
        "SELECT content FROM session_artifacts WHERE session_id = ? AND type='decision' "
        "ORDER BY turn_index LIMIT 8", (sid,)).fetchall()]
    reasoning = conn.execute(
        "SELECT content, turn_index FROM session_artifacts WHERE session_id = ? "
        "AND type='reasoning' ORDER BY turn_index DESC LIMIT 5", (sid,)).fetchall()
    reasoning = list(reversed(reasoning))

    cost = d.get("cost", {})
    tot = sum(int(cost.get(k) or 0) for k in ("input", "output", "cache_read", "cache_write"))
    src = SOURCES.get(d["cli_source"])
    resume = f"cr {sid}"

    L = [
        f"# Context primer — {d.get('title') or (d.get('first_message') or '')[:60]}",
        "",
        f"- **Session:** `{sid}`  ·  **Source:** {d['cli_source']}  ·  **Model:** {d.get('model_used') or '—'}",
        f"- **Project:** {d.get('folder_name')}  ·  **cwd:** `{d.get('cwd') or ''}`",
        f"- **Activity:** {d.get('start_time','')} → {d.get('last_activity','')}  ·  {d.get('turn_count',0)} turns",
        f"- **Usage:** {tot:,} tokens" + (f"  ·  ≈${cost['usd']:.2f} (API list-price equiv)" if cost.get('usd') is not None else ""),
        f"- **Resume this session:** `{resume}`",
        "",
        "## Goal",
        (d.get("first_message") or "(not recorded)").strip(),
        "",
        "## Summary",
        (d.get("summary") or "(not enriched yet)").strip(),
    ]
    if d.get("topics"):
        L.append("")
        L.append("**Topics:** " + ", ".join(d["topics"])
                 + (f"  ·  **Type:** {d.get('session_type')}" if d.get('session_type') else "")
                 + (f"  ·  **Outcome:** {d.get('outcome')}" if d.get('outcome') else ""))
    if decisions:
        L += ["", "## Key decisions"] + [f"- {x}" for x in decisions]
    if reasoning:
        L += ["", "## Recent reasoning (visible)"]
        for content, turn in reasoning:
            L.append(f"- _turn {turn}:_ {content[:400].strip()}")
    L += [
        "",
        "## Pointers",
        f"- Transcript: `{d.get('project_path')}/{sid}.jsonl`",
    ]
    if d.get("reasoning_path"):
        L.append(f"- Full decision trail: `{d['reasoning_path']}`")
    L += [
        "",
        "---",
        "_To continue this work in a new session: read the transcript and decision trail "
        "referenced above, then proceed from the goal/summary/decisions._",
    ]
    folder = re.sub(r"[^a-zA-Z0-9]+", "-", (d.get("folder_name") or "session")).strip("-")
    filename = f"context-{folder}-{sid[:8]}.md"
    # Redact secrets before this primer can be copied / downloaded / bridged out.
    markdown = _redact.redact("\n".join(L))
    return markdown, filename


@app.get("/api/sessions/<sid>/context")
def api_context(sid: str):
    conn = indexer.connect()
    try:
        built = _build_context(conn, sid)
    finally:
        conn.close()
    if built is None:
        return jsonify({"error": "not found"}), 404
    markdown, filename = built
    if request.args.get("download"):
        return Response(markdown, mimetype="text/markdown",
                        headers={"Content-Disposition": f'attachment; filename="{filename}"'})
    return jsonify({"markdown": markdown, "filename": filename})


_BRIDGE_CMD = {
    # start a NEW session in the target CLI, seeded with the primer file as the
    # opening prompt, in the original project dir. No CLI can resume another's
    # session, so this transfers the full context instead.
    "claude":  'cd {cwd} && claude "$(cat {file})"',
    "copilot": 'cd {cwd} && copilot -p "$(cat {file})"',
    "codex":   'cd {cwd} && codex "$(cat {file})"',
}


def _build_bridge(conn, sid: str, target: str) -> dict | None:
    built = _build_context(conn, sid)
    if built is None:
        return None
    context_md, _ = built
    row = conn.execute(
        "SELECT cli_source, cwd, folder_name FROM sessions WHERE session_id = ?", (sid,)
    ).fetchone()
    source = row["cli_source"]
    cwd = row["cwd"] or ""

    header = (
        f"# Handoff: continue this {source} session in {target}\n\n"
        f"You are taking over an in-progress task that was being worked on in the "
        f"**{source}** CLI. No transcript is being resumed — the full context is below. "
        f"Read it, then continue the work from where it left off. For deeper detail you "
        f"may open the referenced transcript and decision-trail files directly.\n\n"
        f"---\n\n"
    )
    primer = header + context_md

    bridges = Path.home() / ".session-browser" / "bridges"
    bridges.mkdir(parents=True, exist_ok=True)
    fpath = bridges / f"{sid[:8]}-{source}-to-{target}.md"
    fpath.write_text(primer, encoding="utf-8")

    tmpl = _BRIDGE_CMD.get(target)
    command = (tmpl.format(cwd=shlex.quote(cwd or "."), file=shlex.quote(str(fpath)))
               if tmpl else f"# unsupported target {target}")
    return {"command": command, "primer": primer, "path": str(fpath),
            "target": target, "source": source}


@app.get("/api/sessions/<sid>/bridge")
def api_bridge(sid: str):
    target = (request.args.get("target") or "").lower()
    if target not in _BRIDGE_CMD:
        return jsonify({"error": f"target must be one of {list(_BRIDGE_CMD)}"}), 400
    conn = indexer.connect()
    try:
        built = _build_bridge(conn, sid, target)
    finally:
        conn.close()
    if built is None:
        return jsonify({"error": "not found"}), 404
    if request.args.get("download"):
        fname = Path(built["path"]).name
        return Response(built["primer"], mimetype="text/markdown",
                        headers={"Content-Disposition": f'attachment; filename="{fname}"'})
    return jsonify(built)


@app.get("/api/sessions/<sid>/thread")
def api_thread(sid: str):
    """Sibling sessions in the same project (excluding this one), newest first.

    Matches on the exact cwd (precise — never collides across different project
    locations that happen to share a basename); falls back to folder_name only if
    this session has no recorded cwd.
    """
    conn = indexer.connect()
    try:
        row = conn.execute(
            "SELECT cwd, folder_name FROM sessions WHERE session_id = ?", (sid,)
        ).fetchone()
        if not row:
            return jsonify({"error": "not found"}), 404
        if row["cwd"]:
            rows = conn.execute(
                "SELECT * FROM sessions WHERE cwd = ? AND session_id <> ? "
                "AND archived = 0 ORDER BY last_activity DESC LIMIT 50",
                (row["cwd"], sid),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM sessions WHERE folder_name = ? AND session_id <> ? "
                "AND archived = 0 ORDER BY last_activity DESC LIMIT 50",
                (row["folder_name"], sid),
            ).fetchall()
    finally:
        conn.close()
    return jsonify({"folder": row["folder_name"], "cwd": row["cwd"],
                    "siblings": [_row_to_dict(r) for r in rows]})


@app.get("/api/sessions/<sid>/reasoning")
def api_reasoning(sid: str):
    conn = indexer.connect()
    try:
        row = conn.execute(
            "SELECT reasoning_path, title, first_message FROM sessions WHERE session_id = ?",
            (sid,),
        ).fetchone()
    finally:
        conn.close()
    if not row:
        return jsonify({"error": "not found"}), 404
    rp = row["reasoning_path"]
    if not rp or not Path(rp).exists():
        return jsonify({"error": "no reasoning captured", "markdown": ""}), 404
    markdown = Path(rp).read_text(encoding="utf-8")
    if request.args.get("format") == "md":
        from flask import Response
        return Response(markdown, mimetype="text/markdown")
    return jsonify({"markdown": markdown, "path": rp,
                    "title": row["title"] or row["first_message"]})


@app.get("/api/stats")
def api_stats():
    conn = indexer.connect()
    try:
        total = conn.execute("SELECT COUNT(*) FROM sessions WHERE archived = 0").fetchone()[0]
        enriched = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE archived = 0 AND summary IS NOT NULL"
        ).fetchone()[0]
        folders = conn.execute(
            "SELECT COUNT(DISTINCT folder_name) FROM sessions WHERE archived = 0"
        ).fetchone()[0]
        by_source = {
            r[0]: r[1] for r in conn.execute(
                "SELECT cli_source, COUNT(*) FROM sessions WHERE archived = 0 GROUP BY cli_source"
            ).fetchall()
        }
    finally:
        conn.close()
    return jsonify({"total": total, "enriched": enriched, "folders": folders,
                    "by_source": by_source, "billing": sbconfig.BILLING})


if __name__ == "__main__":
    sbconfig.ensure_dirs()
    print(f"Session Browser on http://{HOST}:{PORT}  (sources: {list(SOURCES)})")
    app.run(host=HOST, port=PORT, debug=False)
