#!/usr/bin/env python3
"""Flask backend for the Session Browser.

Serves the single-file SPA and a REST API over registry.db. Endpoints are added
slice by slice; this module is the home for all of them.
"""
from __future__ import annotations

import json
import re
import shlex
import shutil
import sys
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import indexer  # noqa: E402
import reasoning  # noqa: E402
import redact as _redact  # noqa: E402
import restore  # noqa: E402
import sbconfig  # noqa: E402
from sources.registry import build_source_registry  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"
_UI = sbconfig.CONFIG.get("ui", {})
HOST = _UI.get("host", "127.0.0.1")
PORT = int(_UI.get("port", 7655))

app = Flask(__name__, static_folder=None)
SOURCES = build_source_registry()

# DNS-rebinding guard: a malicious website can point its own domain at
# 127.0.0.1 and read this API cross-origin. Only accept requests addressed to
# the loopback names / the configured host. Wildcard binds are deliberately NOT
# allowlisted — "0.0.0.0" is never a legitimate Host header.
_LOOPBACK = {"localhost", "127.0.0.1", "[::1]", "::1"}
_ALLOWED_HOSTS = _LOOPBACK | ({HOST} - {"0.0.0.0", "::"})


@app.before_request
def _check_host():
    raw = request.host or ""
    host = raw.split("]")[0] + "]" if raw.startswith("[") else raw.rsplit(":", 1)[0]
    if host not in _ALLOWED_HOSTS:
        return Response("Forbidden: bad Host header", status=403)


# --- helpers ------------------------------------------------------------------
def _restore_supported() -> set[str]:
    """Sources whose adapter can put a raw copy back (restore_path)."""
    return {name for name, a in SOURCES.items() if callable(getattr(a, "restore_path", None))}


def _row_to_dict(row, raw_index: dict | None = None) -> dict:
    """raw_index: reasoning.archived_raw_index(), computed once per request by
    the callers that list archived rows — it labels which of them still have
    a raw transcript copy to restore from. `restorable` means exactly what
    restore.plan() means (raw copy AND an adapter that can place it), so the
    UI never advertises a Restore the server would refuse; `restore_blocker`
    says why not."""
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
    has_raw = raw_index is not None and d["session_id"] in raw_index
    supported = d.get("cli_source") in _restore_supported()
    d["restorable"] = bool(d.get("archived")) and has_raw and supported
    d["restore_blocker"] = (None if not d.get("archived") or d["restorable"]
                            else "unsupported" if not supported else "no-raw-copy")
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


@app.get("/static/<path:filename>")
def static_files(filename: str):
    return send_from_directory(STATIC_DIR, filename)


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
    # live (default): a transcript exists. archived: real sessions whose
    # transcript aged out — the Archived tab. Noise rows are in neither.
    state = request.args.get("state") or "live"

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

    where = [indexer.ARCHIVED_VISIBLE if state == "archived" else indexer.LIVE]
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
            days_i = int(days)  # parse BEFORE touching where/params — a bad value
        except ValueError:      # must not leave a placeholder without its param
            days_i = None
        if days_i is not None:
            # Same spelling as the column (to_iso_utc: 'YYYY-MM-DDTHH:MM:SS.mmmZ');
            # datetime()'s 'YYYY-MM-DD HH:MM:SS' sorts BELOW every row of the
            # cutoff day ('T' > ' '), which leaked up to 24 extra hours.
            where.append("last_activity >= strftime('%Y-%m-%dT%H:%M:%S.000Z', 'now', ?)")
            params.append(f"-{days_i} days")
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
    raw_index = reasoning.archived_raw_index() if state == "archived" else None
    results = [_row_to_dict(r, raw_index) for r in rows]
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
            f"WHERE {indexer.VISIBLE} AND folder_name <> '' ORDER BY folder_name"
        ).fetchall()
    finally:
        conn.close()
    return jsonify([r[0] for r in rows])


@app.get("/api/sessions/topics")
def api_topics():
    conn = indexer.connect()
    try:
        rows = conn.execute(
            f"SELECT topics FROM sessions WHERE {indexer.VISIBLE} AND topics IS NOT NULL"
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
            "SELECT cli_source, cwd, archived, archived_reason FROM sessions WHERE session_id = ?",
            (sid,),
        ).fetchone()
    finally:
        conn.close()
    if not row:
        return jsonify({"error": "not found"}), 404
    if row["archived"]:
        # No file for `claude --resume` to open. Say why instead of handing
        # back a command that fails in the terminal.
        return jsonify({"error": "transcript missing — restore the session first",
                        "archived_reason": row["archived_reason"]}), 409
    src = SOURCES.get(row["cli_source"])
    if src is None:
        # Disabled in config (or a source this build doesn't know): `cr` would
        # only report "not found" after a round trip to the terminal.
        return jsonify({"error": f"no adapter for source '{row['cli_source']}' — enable "
                                 f"[sources.{row['cli_source']}] in config.toml"}), 409
    raw = src.resume_command(sid)
    cwd = row["cwd"] or ""
    # Primary: the `cr` shell shortcut (installed via bin/install-cr.sh). Paste it in
    # the directory where you want to continue — it ports the session's memory there
    # and resumes. command_full is the direct script call if `cr` isn't installed.
    wrapper = _REPO / "bin" / "resume-here.sh"
    command = f"cr {shlex.quote(sid)}"
    command_full = f'{shlex.quote(str(wrapper))} {shlex.quote(sid)} {shlex.quote(row["cli_source"])}'
    return jsonify({"command": command, "command_full": command_full,
                    "raw_command": raw, "origin_cwd": cwd, "cli_source": row["cli_source"]})


@app.post("/api/sessions/<sid>/restore")
def api_restore(sid: str):
    """Copy the newest raw transcript from the reasoning archive back to where
    the CLI looks for it and re-index — the row leaves the Archived view."""
    # Writes into the CLI's own session tree: same drive-by guard as bridge.
    if not request.headers.get("X-Requested-With"):
        return jsonify({"error": "missing X-Requested-With header"}), 403
    res = restore.restore_session(sid, registry=SOURCES)
    if res.status in ("restored", "already-live"):
        code = 200
    elif res.status == "not-found":
        code = 404
    else:
        code = 409
    return jsonify({"status": res.status, "path": str(res.path) if res.path else None,
                    "detail": res.detail, "reimported": res.reimported}), code


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
    # /resume refuses archived rows; the primer must not hand out `cr` for them.
    resume = (f"restore it first (transcript aged out{' on ' + d['archived_at'][:10] if d.get('archived_at') else ''})"
              if d.get("archived") else f"`cr {sid}`")

    L = [
        f"# Context primer — {d.get('title') or (d.get('first_message') or '')[:60]}",
        "",
        f"- **Session:** `{sid}`  ·  **Source:** {d['cli_source']}  ·  **Model:** {d.get('model_used') or '—'}",
        f"- **Project:** {d.get('folder_name')}  ·  **cwd:** `{d.get('cwd') or ''}`",
        f"- **Activity:** {d.get('start_time','')} → {d.get('last_activity','')}  ·  {d.get('turn_count',0)} turns",
        f"- **Usage:** {tot:,} tokens" + (f"  ·  ≈${cost['usd']:.2f} (API list-price equiv)" if cost.get('usd') is not None else ""),
        f"- **Resume this session:** {resume}",
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
    # An adapter's restore_path() doubles as "where this row's transcript
    # lives". The primer is pasted into another agent as its opening prompt,
    # so a pointer is only printed when the file actually exists — a guessed
    # path sends the receiving agent hunting for a file that is not there.
    locate = getattr(SOURCES.get(d["cli_source"]), "restore_path", None)
    try:
        located = locate(r) if callable(locate) else None
    except Exception:  # noqa: BLE001 — a bad row must not break the primer
        located = None
    if located is None and d.get("project_path"):
        located = Path(d["project_path"]) / f"{sid}.jsonl"
    pointers = []
    if located is not None and not d.get("archived") and Path(located).exists():
        pointers.append(f"- Transcript: `{located}`")
    if d.get("reasoning_path") and Path(d["reasoning_path"]).exists():
        pointers.append(f"- Full decision trail: `{d['reasoning_path']}`")
    if pointers:
        L += ["", "## Pointers"] + pointers
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
    # -i, not -p: -p is Copilot's NON-interactive one-shot (the flag headless
    # enrichment uses); a handoff must open a session the user can continue.
    "copilot": 'cd {cwd} && copilot -i "$(cat {file})"',
    "codex":   'cd {cwd} && codex "$(cat {file})"',
    "opencode": 'cd {cwd} && opencode --prompt "$(cat {file})"',
}


def _installed(source: str) -> bool:
    """Is that CLI runnable from this process? The adapter's has_binary() when
    the source is enabled, else the configured binary name on PATH."""
    adapter = SOURCES.get(source)
    hb = getattr(adapter, "has_binary", None)
    if callable(hb):
        return bool(hb())
    binary = sbconfig.source_config(source).get("binary", source)
    return shutil.which(binary) is not None


@app.get("/api/sources")
def api_sources():
    """Every CLI this build knows: enabled (adapter loaded), installed (binary
    on PATH — resume/bridge will work), has_data (transcripts to index). The
    SPA uses it to offer only bridge targets that exist on this machine."""
    names = list(dict.fromkeys([*SOURCES, *_BRIDGE_CMD]))
    out = {}
    for name in names:
        adapter = SOURCES.get(name)
        out[name] = {"enabled": adapter is not None, "installed": _installed(name),
                     "has_data": bool(adapter.is_available()) if adapter is not None else False}
    return jsonify(out)


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


@app.post("/api/sessions/<sid>/bridge")
def api_bridge(sid: str):
    # This endpoint writes a primer file to disk. POST + a custom header keeps
    # cross-origin pages out: an <img>/<form> can send neither, and fetch()
    # with a custom header is blocked by CORS preflight. (The SPA downloads
    # the primer client-side from the JSON — no separate download route.)
    if not request.headers.get("X-Requested-With"):
        return jsonify({"error": "missing X-Requested-With header"}), 403
    target = (request.args.get("target") or "").lower()
    if target not in _BRIDGE_CMD:
        return jsonify({"error": f"target must be one of {list(_BRIDGE_CMD)}"}), 400
    if not _installed(target):
        # Refuse BEFORE writing a primer: the command we would hand back dies
        # with "command not found" once pasted into a terminal.
        return jsonify({"error": f"{target} is not installed on this machine (binary not on PATH)"}), 409
    conn = indexer.connect()
    try:
        built = _build_bridge(conn, sid, target)
    finally:
        conn.close()
    if built is None:
        return jsonify({"error": "not found"}), 404
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
                f"AND {indexer.VISIBLE} ORDER BY last_activity DESC LIMIT 50",
                (row["cwd"], sid),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM sessions WHERE folder_name = ? AND session_id <> ? "
                f"AND {indexer.VISIBLE} ORDER BY last_activity DESC LIMIT 50",
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
    # trails are written redacted, but re-redact on the way out: archives from
    # before that fix (and hand-edited files) shouldn't leak. 2MB cap.
    with open(rp, "r", encoding="utf-8", errors="replace") as fh:
        markdown = _redact.redact(fh.read(2_000_000))
    if request.args.get("format") == "md":
        from flask import Response
        return Response(markdown, mimetype="text/markdown")
    return jsonify({"markdown": markdown, "path": rp,
                    "title": row["title"] or row["first_message"]})


@app.get("/api/stats/timeseries")
def api_stats_timeseries():
    """Aggregates for the usage dashboard — all derived from existing columns via
    GROUP BY (no schema change). Powers the activity heatmap, per-day tokens/cost,
    and per-model / per-project / per-source breakdowns."""
    conn = indexer.connect()
    try:
        # per-day: session count, tokens, cost — last 365 days
        per_day = [dict(r) for r in conn.execute(
            "SELECT substr(last_activity,1,10) AS day, COUNT(*) AS sessions, "
            "COALESCE(SUM(COALESCE(input_tokens,0)+COALESCE(output_tokens,0)+COALESCE(cache_read_tokens,0)+COALESCE(cache_write_tokens,0)),0) AS tokens, "
            "COALESCE(SUM(cost_usd),0) AS cost "
            f"FROM sessions WHERE {indexer.VISIBLE} AND last_activity!='' "
            "GROUP BY day ORDER BY day"
        ).fetchall()]
        by_model = [dict(r) for r in conn.execute(
            "SELECT COALESCE(model_used,'unknown') AS model, COUNT(*) AS sessions, "
            "COALESCE(SUM(COALESCE(input_tokens,0)+COALESCE(output_tokens,0)+COALESCE(cache_read_tokens,0)+COALESCE(cache_write_tokens,0)),0) AS tokens, "
            f"COALESCE(SUM(cost_usd),0) AS cost FROM sessions WHERE {indexer.VISIBLE} "
            "GROUP BY model ORDER BY cost DESC LIMIT 20"
        ).fetchall()]
        by_project = [dict(r) for r in conn.execute(
            "SELECT COALESCE(NULLIF(folder_name,''),'—') AS project, COUNT(*) AS sessions, "
            "COALESCE(SUM(COALESCE(input_tokens,0)+COALESCE(output_tokens,0)+COALESCE(cache_read_tokens,0)+COALESCE(cache_write_tokens,0)),0) AS tokens, "
            f"COALESCE(SUM(cost_usd),0) AS cost FROM sessions WHERE {indexer.VISIBLE} "
            "GROUP BY project ORDER BY cost DESC LIMIT 15"
        ).fetchall()]
        by_source = [dict(r) for r in conn.execute(
            "SELECT cli_source AS source, COUNT(*) AS sessions, "
            "COALESCE(SUM(COALESCE(input_tokens,0)+COALESCE(output_tokens,0)+COALESCE(cache_read_tokens,0)+COALESCE(cache_write_tokens,0)),0) AS tokens, "
            f"COALESCE(SUM(cost_usd),0) AS cost FROM sessions WHERE {indexer.VISIBLE} "
            "GROUP BY source ORDER BY cost DESC"
        ).fetchall()]
        totals = dict(conn.execute(
            "SELECT COUNT(*) AS sessions, "
            "COALESCE(SUM(COALESCE(input_tokens,0)+COALESCE(output_tokens,0)+COALESCE(cache_read_tokens,0)+COALESCE(cache_write_tokens,0)),0) AS tokens, "
            f"COALESCE(SUM(cost_usd),0) AS cost FROM sessions WHERE {indexer.VISIBLE}"
        ).fetchone())
        # this-calendar-month cost (the headline "≈$X of API-equivalent" number)
        month_cost = conn.execute(
            f"SELECT COALESCE(SUM(cost_usd),0) FROM sessions WHERE {indexer.VISIBLE} "
            "AND last_activity >= strftime('%Y-%m-01', 'now')"
        ).fetchone()[0]
    finally:
        conn.close()
    return jsonify({"per_day": per_day, "by_model": by_model, "by_project": by_project,
                    "by_source": by_source, "totals": totals, "month_cost": month_cost,
                    "billing": sbconfig.BILLING, "notional": sbconfig.COST_IS_NOTIONAL})


@app.get("/api/stats")
def api_stats():
    # ?state=live|archived scopes the counts to one tab so the source pills
    # match the list beneath them; the default (everything visible) is what
    # the header line and other consumers want.
    pred = {"live": indexer.LIVE, "archived": indexer.ARCHIVED_VISIBLE}.get(
        request.args.get("state") or "", indexer.VISIBLE)
    conn = indexer.connect()
    try:
        total = conn.execute(f"SELECT COUNT(*) FROM sessions WHERE {pred}").fetchone()[0]
        enriched = conn.execute(
            f"SELECT COUNT(*) FROM sessions WHERE {pred} AND summary IS NOT NULL"
        ).fetchone()[0]
        folders = conn.execute(   # same rule as /api/sessions/folders: '' is not a folder
            f"SELECT COUNT(DISTINCT folder_name) FROM sessions WHERE {pred} AND folder_name <> ''"
        ).fetchone()[0]
        by_source = {
            r[0]: r[1] for r in conn.execute(
                f"SELECT cli_source, COUNT(*) FROM sessions WHERE {pred} GROUP BY cli_source"
            ).fetchall()
        }
        archived = conn.execute(
            f"SELECT COUNT(*) FROM sessions WHERE {indexer.ARCHIVED_VISIBLE}"
        ).fetchone()[0]
    finally:
        conn.close()
    return jsonify({"total": total, "enriched": enriched, "folders": folders,
                    "archived": archived, "by_source": by_source, "billing": sbconfig.BILLING})


if __name__ == "__main__":
    sbconfig.ensure_dirs()
    if HOST not in _LOOPBACK:
        # The Host-header check only stops browser-based rebinding; a direct
        # socket peer sends whatever Host it likes. Binding beyond loopback
        # exposes every session title/summary/reasoning trail to the network.
        print("WARNING: [ui].host is not loopback — the whole session index "
              "becomes readable by anyone on your network. Prefer 127.0.0.1 "
              "plus an SSH tunnel for remote access.")
    print(f"Session Browser on http://{HOST}:{PORT}  (sources: {list(SOURCES)})")
    # threaded: the first semantic query loads the embedding model for seconds —
    # a single-threaded server would freeze every other request meanwhile.
    app.run(host=HOST, port=PORT, debug=False, threaded=True)
