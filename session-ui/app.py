#!/usr/bin/env python3
"""Flask backend for the Session Browser.

Serves the single-file SPA and a REST API over registry.db. Endpoints are added
slice by slice; this module is the home for all of them.

What this program is
--------------------
A **local-only** web server. Start it with `python session-ui/app.py` (or `sb ui`): it
binds 127.0.0.1 on the port from `[ui].port` in config.toml, and a browser on the same
machine loads `static/index.html` — one page of plain JavaScript, no build step — which
then calls the `/api/...` routes below. There is no login and no multi-user notion because
there is no network exposure: nothing here is meant to be reachable from another machine.

Vocabulary used throughout (full definitions in docs/GLOSSARY.md):
  * **session** — one conversation with an AI coding CLI (Claude Code, GitHub Copilot CLI,
    OpenAI Codex CLI, OpenCode), first prompt to last reply. One registry row each.
  * **transcript** — the file the CLI wrote while that conversation happened.
  * **resume** — reopening an old session inside its own CLI so the conversation continues
    with its history intact (`claude --resume <id>`). Needs the transcript to still exist.
  * **bridge / primer** — a short Markdown briefing about a session (goal, decisions, open
    threads) written so a *different* CLI can take the work over. No CLI can resume
    another CLI's session, so the primer carries the context across instead.
  * **tokens** — the word-pieces a model reads and writes; vendors bill per token, so the
    token counts shown here are the raw measure of how much a session used.

Where it sits in the pipeline (see docs/ARCHITECTURE.md for the whole map): this module is
a *consumer*. The Stop hook, the watcher and the backfill create rows; the nightly
enrichment scripts fill in summaries, topics, costs and reasoning-trail paths; this server
only reads `~/.session-browser/registry.db` and the files those rows point at. It performs
exactly two writes, both of them explicit user actions:
  * POST /api/sessions/<sid>/restore — copies an aged-out transcript back into the CLI's
    own directory (restore.py) so the session can be resumed again;
  * POST /api/sessions/<sid>/bridge  — writes a primer file under
    `~/.session-browser/bridges/` for the handoff command it hands back.

Cross-cutting rules worth knowing before reading any single route:
  * **Visibility.** No query here spells the archive flag itself. They compose the named
    predicates from indexer.py: `LIVE` (the transcript still exists on disk),
    `ARCHIVED_VISIBLE` (real sessions whose transcript aged out — the Archived tab) and
    `VISIBLE` (LIVE plus ARCHIVED_VISIBLE — everything a human should see, never the
    subagent/sidechain rows that were archived as noise). A test greps the whole tree to
    keep the spelling in one place.
  * **Redaction at every egress.** Any text that can leave the tool — Copy Context,
    Export, the Bridge primer, the reasoning viewer — goes through redact.py first, so an
    API key that was pasted into a session a year ago cannot be copied back out.
  * **CSRF.** The two state-changing routes are POST *and* require an `X-Requested-With`
    request header; api_bridge() explains why a header is enough here.
  * **Honest refusals.** A route would rather answer 409 with a sentence than hand back a
    command that dies once pasted into a terminal. Hence the checks for "is that CLI
    actually installed here" and "is this row archived" before building resume/bridge
    commands.

Tests: tests/test_smoke.py loads this file as a module (`_load_app`), points
`indexer.connect` at a temporary registry and drives the routes with Flask's test client
(see `_app_harness` and the `test_api_*` cases) — so the contracts documented on each
route below are asserted, not aspirational.
"""
from __future__ import annotations

import json
import re
import shlex
import shutil
import sys
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory

# This file lives in session-ui/, but the modules it needs (indexer, redact, restore,
# sbconfig, sources/) sit at the repo root. Put the root on the import path first, then
# import them — hence the `# noqa: E402` markers, which tell the linter that these
# imports are deliberately below other code rather than an accident.
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import indexer  # noqa: E402
import reasoning  # noqa: E402
import redact as _redact  # noqa: E402
import restore  # noqa: E402
import sbconfig  # noqa: E402
from sources.registry import build_source_registry  # noqa: E402

# The SPA and its assets: session-ui/static/index.html plus the compiled tailwind.css.
STATIC_DIR = Path(__file__).resolve().parent / "static"
# The `[ui]` table of config.toml (layered over config.toml.example by sbconfig), e.g.
#   [ui]
#   host = "127.0.0.1"
#   port = 7655
# The literal defaults below are the fallback if the table or a key is missing. Every key
# documented under [ui] in config.toml.example must appear here as a quoted string — a
# docs-consistency test asserts it, so nothing can be documented but unread.
_UI = sbconfig.CONFIG.get("ui", {})
HOST = _UI.get("host", "127.0.0.1")
PORT = int(_UI.get("port", 7655))

# static_folder=None: Flask's built-in /static handler is switched off so static_files()
# below is the single place that serves files, under the path the SPA actually requests.
app = Flask(__name__, static_folder=None)
# name -> adapter instance, one per CLI that is enabled in config.toml (sources/*.py).
# Built once at import time: the routes only ever ask an adapter for a resume command, a
# restore destination, or whether its binary exists, so there is nothing per-request here.
SOURCES = build_source_registry()

# DNS-rebinding guard: a malicious website can point its own domain at
# 127.0.0.1 and read this API cross-origin. Only accept requests addressed to
# the loopback names / the configured host. Wildcard binds are deliberately NOT
# allowlisted — "0.0.0.0" is never a legitimate Host header.
_LOOPBACK = {"localhost", "127.0.0.1", "[::1]", "::1"}
_ALLOWED_HOSTS = _LOOPBACK | ({HOST} - {"0.0.0.0", "::"})


@app.before_request
def _check_host():
    """Reject any request whose `Host:` header is not a name we expect (403).

    Runs before every route (Flask's `before_request`). Returning a response here short
    circuits the route; returning None lets it proceed.

    The attack it blocks: a browser will happily send requests to `http://evil.example`
    whose DNS answer is 127.0.0.1, and the page at that origin then reads this API as if
    it were its own. The requests still carry `Host: evil.example`, which is the one part
    a page cannot forge, so comparing it is a reliable filter.
    """
    raw = request.host or ""
    # Strip the port so "localhost:7655" matches "localhost". IPv6 hosts arrive in
    # brackets ("[::1]:7655"), where rsplit on ":" would cut inside the address — keep
    # everything up to and including the closing bracket instead.
    host = raw.split("]")[0] + "]" if raw.startswith("[") else raw.rsplit(":", 1)[0]
    if host not in _ALLOWED_HOSTS:
        return Response("Forbidden: bad Host header", status=403)


# --- helpers ------------------------------------------------------------------
def _row_to_dict(row, raw_index: dict | None = None) -> dict:
    """Turn one `sessions` row into the JSON object the SPA expects.

    Every listing route funnels through here, so the shape is the same everywhere: all the
    stored columns, plus these derived keys —
      * `topics`, `models_used` — decoded from their JSON-text columns into real lists;
      * `is_active`     — true if the session was touched in the last two hours;
      * `has_reasoning` — true if a decision trail was extracted for it;
      * `restorable`    — true if the Restore button should be offered (archived rows only);
      * `restore_blocker` — None, "no-raw-copy" or "unsupported": why Restore is not offered;
      * `cost`          — the token counts and dollar figure gathered into one sub-object.

    raw_index: reasoning.archived_raw_index(), computed once per request by
    the callers that list archived rows — it labels which of them still have
    a raw transcript copy to restore from. `restorable` means exactly what
    restore.plan() means (raw copy AND an adapter that can place it), so the
    UI never advertises a Restore the server would refuse; `restore_blocker`
    says why not.

    Pure: reads the row and the filesystem index it is handed, writes nothing.
    """
    d = dict(row)
    # topics / models_used are JSON-encoded text columns.
    # SQLite has no array type, so they are stored as JSON text — e.g. '["python","ci-cd"]'
    # — and decoded here. A malformed or NULL value becomes [] rather than an exception:
    # one bad row must not blank the whole listing, and the SPA always wants a list.
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
    # Per ROW (restore.supported_for): an adapter refuses rows whose recorded
    # path is outside its tree, so a per-source answer advertised Restores the
    # server then refused. Only archived listings pay for the per-row check.
    supported = bool(d.get("archived")) and restore.supported_for(d, SOURCES)
    d["restorable"] = bool(d.get("archived")) and has_raw and supported
    # Name the fact that is actually missing: with no raw copy there is nothing
    # to place back, whatever the adapter could do.
    d["restore_blocker"] = (None if not d.get("archived") or d["restorable"]
                            else "no-raw-copy" if not has_raw else "unsupported")
    # One nested object so the SPA can render usage without knowing four column names.
    # The four token counts are what the CLI reported; `usd` is what compute-costs.py
    # derived from them at public list prices. `notional` carries the caveat with the
    # number: under a flat monthly subscription no per-session money is actually billed,
    # so the UI shows the figure prefixed with "≈" as an intensity signal, not a bill.
    d["cost"] = {
        "usd": d.get("cost_usd"),
        "input": d.get("input_tokens"), "output": d.get("output_tokens"),
        "cache_read": d.get("cache_read_tokens"), "cache_write": d.get("cache_write_tokens"),
        "notional": sbconfig.COST_IS_NOTIONAL,
    }
    return d


def _is_active(last_activity: str | None) -> bool:
    """Was this session touched recently enough to show a "live now" dot in the list?

    `last_activity` is the canonical UTC stamp the indexer writes, e.g.
    '2026-09-11T14:03:22.000Z'. 7200 seconds = two hours: long enough to cover a coffee
    break in an open session, short enough that yesterday's work is not flagged.
    Anything unparseable (empty column, a legacy format) counts as not active — this is a
    cosmetic badge, never worth raising for.
    """
    if not last_activity:
        return False
    try:
        from datetime import datetime, timezone
        # fromisoformat does not accept the trailing 'Z' on older Pythons; '+00:00' is the
        # same instant spelled the way it does accept.
        ts = datetime.fromisoformat(last_activity.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - ts).total_seconds() < 7200
    except (ValueError, TypeError):
        return False


# --- routes -------------------------------------------------------------------
# Every route below is GET unless marked otherwise, answers JSON unless marked otherwise,
# and takes its parameters from the query string (?key=value). Error bodies are always
# `{"error": "<sentence>"}` with a status that says what kind of refusal it was:
#   400 the request asked for something that does not exist as an option (bad bridge target)
#   403 the CSRF header was missing on a state-changing route
#   404 no such session / no such artefact for it
#   409 the request is well-formed but cannot be honoured right now (archived row, a CLI
#       that is not installed here, nothing to restore from)
@app.get("/")
def index():
    """GET / -> the SPA itself (session-ui/static/index.html)."""
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/static/<path:filename>")
def static_files(filename: str):
    """GET /static/<name> -> a file from session-ui/static (the stylesheet, icons).

    send_from_directory refuses paths that escape STATIC_DIR, so a request for
    '../../config.toml' is a 404 rather than a file read.
    """
    return send_from_directory(STATIC_DIR, filename)


@app.get("/health")
def health():
    """GET /health -> {"status": "ok"}. Used by `sb ui` / doctor.sh to tell "the server is
    up on this port" from "something else has the port", without parsing the SPA."""
    return jsonify({"status": "ok"})


PAGE = 500   # rows per listing; a truncated page is flagged in the response headers
# Why a cap at all: the SPA renders the whole list client-side, and a registry with tens of
# thousands of rows would ship megabytes of JSON per keystroke. Why a *flagged* cap: see
# api_sessions() — a silent cut once hid the oldest rows with no way to reach them.


@app.get("/api/sessions")
def api_sessions():
    """GET /api/sessions -> a JSON array of session objects (`_row_to_dict` shape).

    This is the main list the SPA shows. Query parameters, all optional and combinable:
      search=<text>   match against the session's first message, summary, title or topics
      mode=semantic   rank `search` by meaning instead (embeddings; see semsearch.py)
      mode=fulltext   match `search` against the transcript body via the SQLite FTS index
      folder=<name>   only sessions whose project folder is exactly this
      source=<cli>    only sessions from one CLI ("claude", "codex", ...); "all" = no filter
      topic=<tag>     only sessions tagged with this topic
      days=<n>        only sessions active within the last n days
      state=live      (default) sessions whose transcript still exists
      state=archived  real sessions whose transcript aged out — the Archived tab

    Newest first (`last_activity DESC`), except in semantic mode where the order is the
    similarity ranking. At most PAGE (500) rows; if more matched, two response headers say
    so: `X-Result-Truncated: 1` and `X-Result-Total: <true count>`. The SPA shows that
    number and tells the user to narrow the filters — without it, the header once claimed
    602 sessions while the list held 500 and the missing 102 were unreachable.

    Always 200, even for no matches (an empty array). Read-only.
    """
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
    # "Semantic" = search by meaning: each session's text was turned into a list of numbers
    # (an embedding) by a small local model, and the query is scored against them, so "the
    # time the checkout tests went flaky" finds the session even with no word in common.
    # sem_ids is the shortlist of ids that search produced, in rank order; None means "no
    # shortlist — use the plain LIKE search below". An empty list means "searched, matched
    # nothing", which is why the two cases must stay distinguishable.
    sem_ids: list[str] | None = None
    if mode == "semantic" and search:
        try:
            # Imported lazily: it pulls in numpy and, on first use, loads the embedding
            # model (seconds). A user who never runs a semantic search never pays for it.
            import semsearch
            hits = semsearch.search(search, limit=40)
            sem_ids = [sid for sid, _ in hits]
        except Exception:  # noqa: BLE001 — fall back to keyword on any failure
            # Deliberately broad: no embeddings built yet, the model not installed, a
            # dimension mismatch. Search degrading to keyword matching beats a 500.
            sem_ids = None
    # Full-text mode: match against transcript body via FTS5.
    # FTS5 is SQLite's built-in word index, built by scripts/build-fts.py over the
    # transcripts themselves — so this finds a phrase that only ever appeared inside the
    # conversation, not just in the summary columns.
    if mode == "fulltext" and search:
        # Keep only word characters and rejoin with spaces: raw user text can contain
        # FTS5 operators ('"', '*', 'NEAR', '-') that would either error or mean something
        # the user did not type. ['fix', 'flaky', 'tests'] is then an implicit AND.
        toks = re.findall(r"\w+", search)
        if toks:
            conn0 = indexer.connect()
            try:
                # `sessions_fts MATCH ?` is FTS5's full-table match syntax: it searches
                # every indexed column of the virtual table for the query terms. 200 is a
                # shortlist; the filters below narrow it further.
                rows0 = conn0.execute(
                    "SELECT session_id FROM sessions_fts WHERE sessions_fts MATCH ? LIMIT 200",
                    (" ".join(toks),)).fetchall()
                sem_ids = [r[0] for r in rows0]
            except Exception:  # noqa: BLE001 — FTS missing/not built -> keyword fallback
                sem_ids = None
            finally:
                conn0.close()

    # Build the WHERE clause as a list of fragments plus a parallel list of bound values.
    # Everything user-supplied goes in as a `?` parameter, never string-formatted into the
    # SQL — that is what keeps a session title containing a quote from breaking the query
    # (and what makes SQL injection impossible here).
    #
    # The first fragment is the visibility rule and is never user-supplied: the Archived
    # tab asks for indexer.ARCHIVED_VISIBLE (rows archived because their transcript aged
    # out), everything else for indexer.LIVE (rows whose transcript is still on disk).
    # Rows archived as not-a-session — subagent side-conversations that were never real
    # sessions — match neither, so they are invisible in both tabs.
    where = [indexer.ARCHIVED_VISIBLE if state == "archived" else indexer.LIVE]
    params: list = []
    if folder:
        where.append("folder_name = ?")
        params.append(folder)
    if source and source != "all":
        where.append("cli_source = ?")
        params.append(source)
    if topic:
        # topics is JSON text like '["python","testing"]', so the tag is matched with its
        # surrounding quotes: '%"ci"%' hits '["ci"]' but not '["ci-cd"]'.
        where.append("topics LIKE ?")
        params.append(f'%"{topic}"%')
    if days:
        try:
            days_i = int(days)  # parse BEFORE touching where/params — a bad value
        except ValueError:      # must not leave a placeholder without its param
            days_i = None
        if days_i is not None:
            # last_activity is plain text, so the cutoff has to be rendered in exactly the
            # same spelling the column uses or the string comparison is wrong.
            # strftime(...) builds '2026-09-04T00:00:00.000Z' for ?='-7 days'.
            #
            # Same spelling as the column (to_iso_utc: 'YYYY-MM-DDTHH:MM:SS.mmmZ');
            # datetime()'s 'YYYY-MM-DD HH:MM:SS' sorts BELOW every row of the
            # cutoff day ('T' > ' '), which leaked up to 24 extra hours.
            where.append("last_activity >= strftime('%Y-%m-%dT%H:%M:%S.000Z', 'now', ?)")
            params.append(f"-{days_i} days")
    if sem_ids is not None:
        # A semantic / full-text search ran. Restrict to the ids it returned.
        if not sem_ids:
            # It ran and matched nothing. Answer honestly rather than falling through to
            # the LIKE branch, which would quietly show unrelated keyword hits instead.
            return jsonify([])
        # One '?' per id: "session_id IN (?,?,?)". The ids are still bound, not inlined.
        placeholders = ",".join("?" * len(sem_ids))
        where.append(f"session_id IN ({placeholders})")
        params += sem_ids
    elif search:
        # Plain keyword mode: a substring match across the four columns a user would
        # expect to search — what they asked for, what the summariser wrote, the title and
        # the topic tags. LIKE '%x%' cannot use an index, but at this row count it is
        # instant and it needs no FTS index to have been built.
        where.append("(first_message LIKE ? OR summary LIKE ? OR title LIKE ? OR topics LIKE ?)")
        like = f"%{search}%"
        params += [like, like, like, like]

    # Newest activity first: the session you were just in is the one you want to find.
    order = "last_activity DESC"
    clause = "SELECT * FROM sessions WHERE " + " AND ".join(where)
    # One row past the page: a hard cap with no marker made the header count
    # 602 sessions while the list showed 500 and the oldest 102 — the very rows
    # the archive protects — were unreachable from any UI path.
    # Asking for PAGE + 1 rows is the cheap way to learn "is there more?": if 501 come
    # back, the page is truncated and only then is the (more expensive) COUNT(*) run to
    # report the true total. The common case pays nothing extra.
    sql = clause + f" ORDER BY {order} LIMIT {PAGE + 1}"
    conn = indexer.connect()
    try:
        rows = conn.execute(sql, params).fetchall()
        total = len(rows)
        if len(rows) > PAGE:
            rows = rows[:PAGE]
            total = conn.execute("SELECT COUNT(*) FROM sessions WHERE " + " AND ".join(where),
                                 params).fetchone()[0]
    finally:
        conn.close()
    # One directory walk of the raw-transcript vault for the whole page, not one per row:
    # archived rows need to know whether a copy exists to restore from. Live rows do not
    # need it at all, so the live tab never pays for the walk.
    raw_index = reasoning.archived_raw_index() if state == "archived" else None
    results = [_row_to_dict(r, raw_index) for r in rows]
    # In semantic mode, preserve similarity ranking from sem_ids.
    # SQL returned the rows in date order; re-sort them into the order the search ranked
    # them. An id somehow missing from the rank map sorts last (1e9) instead of raising.
    if sem_ids is not None:
        rank = {sid: i for i, sid in enumerate(sem_ids)}
        results.sort(key=lambda d: rank.get(d["session_id"], 1e9))
    resp = jsonify(results)
    # Headers, not a wrapper object, so the body stays a plain array for every caller;
    # the SPA reads them to render "showing 500 of 602 — narrow your filters".
    if total > len(results):
        resp.headers["X-Result-Truncated"] = "1"
        resp.headers["X-Result-Total"] = str(total)
    return resp


@app.get("/api/sessions/folders")
def api_folders():
    """GET /api/sessions/folders -> a sorted JSON array of project folder names.

    Fills the folder dropdown, so it must offer exactly the values ?folder= can match.
    The SQL lists each distinct folder_name across every visible row (live plus aged-out;
    subagent noise excluded) and skips the empty string — a session whose working
    directory was never recorded is not a folder anyone can pick.
    """
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
    """GET /api/sessions/topics -> a sorted JSON array of every topic tag in use.

    Fills the topic dropdown. Topics are short keyword tags ('python', 'ci-cd') attached
    to a session by enrichment or by the keyword rules in scripts/classify-topics.py.

    They are stored one JSON list per row, so there is no way to ask SQLite for the
    distinct tags: the query pulls the JSON text of every visible row and the union is
    built here. A row with malformed JSON is skipped rather than failing the dropdown.
    """
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
    """GET /api/sessions/<sid>/resume -> the shell command that reopens this session.

    "Resume" means reopening the conversation inside the CLI that recorded it, with its
    history intact. This route does not run anything: it hands the SPA a string to show in
    a Copy box, which the user pastes into their own terminal.

    200 body:
      command       the short form: `cr <id>` (the shell helper from bin/install-cr.sh)
      command_full  the same thing without that helper: a direct call to bin/resume-here.sh
      raw_command   what the CLI itself would need, from the adapter (`claude --resume <id>`)
      origin_cwd    the directory the session was originally worked in
      origin_cwd_exists  whether that directory exists on THIS machine (it may not: the
                    registry can be carried over from another laptop)
      cli_source    which CLI recorded it

    Refusals, each with an `error` sentence, because the alternative is a command that
    fails after the user has already switched to a terminal:
      404  no row with that id
      409  the row is archived — its transcript is gone, so there is nothing to reopen;
           the body also carries `archived_reason`, and the fix is POST .../restore
      409  no adapter for the row's CLI (disabled in config.toml, or unknown to this build)
      409  that CLI's binary is not on PATH here

    Read-only.
    """
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
    if not _installed(row["cli_source"]):
        # Same rule as bridge: never hand back a command that dies with
        # "command not found" after the user has switched terminals.
        return jsonify({"error": f"{row['cli_source']} is not installed on this machine "
                                 f"(binary not on PATH) — resume it where it is"}), 409
    # Each adapter knows its own CLI's flag spelling (`claude --resume <id>`,
    # `codex resume <id>`, `opencode --session <id>`); nothing here hard-codes it.
    raw = src.resume_command(sid)
    cwd = row["cwd"] or ""
    # Primary: the `cr` shell shortcut (installed via bin/install-cr.sh). Paste it in
    # the directory where you want to continue — it ports the session's memory there
    # and resumes. command_full is the direct script call if `cr` isn't installed.
    # shlex.quote wraps anything containing spaces or shell metacharacters in quotes, so a
    # path like /Users/me/code/my proj survives being pasted into a shell verbatim.
    wrapper = _REPO / "bin" / "resume-here.sh"
    command = f"cr {shlex.quote(sid)}"
    command_full = f'{shlex.quote(str(wrapper))} {shlex.quote(sid)} {shlex.quote(row["cli_source"])}'
    return jsonify({"command": command, "command_full": command_full,
                    "raw_command": raw, "origin_cwd": cwd, "cli_source": row["cli_source"],
                    "origin_cwd_exists": bool(cwd) and Path(cwd).is_dir()})


@app.post("/api/sessions/<sid>/restore")
def api_restore(sid: str):
    """Copy the newest raw transcript from the reasoning archive back to where
    the CLI looks for it and re-index — the row leaves the Archived view.

    POST /api/sessions/<sid>/restore. One of the two routes that changes anything on disk.

    Why this exists: CLIs delete their own old transcripts (Claude Code's
    `cleanupPeriodDays` defaults to 30). This tool keeps a byte-for-byte copy of every
    transcript it indexed under `~/claude-reasoning-archive/raw/`, so the file can be put
    back where the CLI expects it and the session becomes resumable again. For OpenCode,
    whose sessions live in a database rather than a file, restore.py also re-imports it —
    `reimported` in the body is True/False when that ran, null when it did not apply.

    Body: {status, path, detail, reimported}. `status` is restore.py's own vocabulary and
    maps to the HTTP code:
      200  "restored"      the file is back and the row is live again (`path` says where)
      200  "already-live"  nothing to do; the transcript was never missing
      404  "not-found"     no row with that id
      409  "no-raw-copy"   the row is archived but no copy was ever archived for it
      409  "unsupported"   no adapter can place this row's file back (e.g. its recorded
                           path belongs to another machine's home directory)
      409  "not-a-session" the row is subagent noise, not a conversation worth restoring
    403 if the CSRF header is missing (see below).
    """
    # Writes into the CLI's own session tree: same drive-by guard as bridge.
    # A page on another origin can make a browser POST here (a form submit needs no
    # permission), but it cannot attach a custom header — doing so triggers a CORS
    # preflight that this server never answers. So requiring any X-Requested-With value
    # is enough to prove the request came from our own page, not a drive-by web page.
    if not request.headers.get("X-Requested-With"):
        return jsonify({"error": "missing X-Requested-With header"}), 403
    res = restore.restore_session(sid, registry=SOURCES)
    if res.status in ("restored", "already-live"):
        code = 200
    elif res.status == "not-found":
        code = 404
    else:
        # Everything else is a well-formed request that cannot be honoured: 409 Conflict.
        code = 409
    return jsonify({"status": res.status, "path": str(res.path) if res.path else None,
                    "detail": res.detail, "reimported": res.reimported}), code


def _build_context(conn, sid: str) -> tuple[str, str] | None:
    """Build a portable context primer (markdown) from indexed metadata — fast,
    no transcript parse. Designed to paste into a new session to carry work over.

    A "primer" is a one-page briefing about a past session: what it was trying to do, what
    was decided, what was still open, and where the full record lives. Pasting it as the
    first message of a brand-new conversation lets any CLI pick the work up — which is how
    work moves between CLIs at all, since none of them can open another's transcript.

    Returns (markdown, suggested filename), or None if there is no such session. The
    filename is derived from the project folder and the first 8 characters of the id, e.g.
    'context-session-browser-a1b2c3d4.md'.

    Cheap on purpose: everything comes from columns the indexer and the enrichment
    pipeline already wrote, plus two small queries against `session_artifacts` (the
    per-turn decisions and reasoning snippets extracted earlier). No transcript is opened,
    so the button feels instant even on a large session.

    Side effects: none — it only reads. The output is redacted on the last line before
    being returned, because from here it goes to the clipboard, a download, or another CLI.

    `conn` is passed in (never opened here) so callers can share one connection and so
    tests can hand it a temporary registry.
    """
    r = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (sid,)).fetchone()
    if not r:
        return None
    d = _row_to_dict(r)
    # Up to 8 decisions, oldest first: the primer should read in the order the work
    # happened. `session_artifacts` holds one row per extracted item, keyed by session and
    # tagged with a type ('decision', 'reasoning'), with turn_index giving their order.
    decisions = [row[0] for row in conn.execute(
        "SELECT content FROM session_artifacts WHERE session_id = ? AND type='decision' "
        "ORDER BY turn_index LIMIT 8", (sid,)).fetchall()]
    # The five most RECENT reasoning steps (DESC + LIMIT picks the tail of the session),
    # then flipped back into chronological order for display. Where the session ended up
    # is what the next agent needs; where it started is already in the summary.
    #
    # NOTE: this local name deliberately shadows the imported `reasoning` module for the
    # rest of the function — the module is not used again below this point.
    reasoning = conn.execute(
        "SELECT content, turn_index FROM session_artifacts WHERE session_id = ? "
        "AND type='reasoning' ORDER BY turn_index DESC LIMIT 5", (sid,)).fetchall()
    reasoning = list(reversed(reasoning))

    cost = d.get("cost", {})
    # Total tokens = everything the session sent and received, including the cached parts.
    # `or 0` per column, not one SUM: a single NULL column would otherwise zero the lot.
    tot = sum(int(cost.get(k) or 0) for k in ("input", "output", "cache_read", "cache_write"))
    # /resume refuses archived rows; the primer must not hand out `cr` for them.
    resume = (f"restore it first (transcript aged out{' on ' + d['archived_at'][:10] if d.get('archived_at') else ''})"
              if d.get("archived") else f"`cr {sid}`")

    # L is the primer, built as a list of Markdown lines and joined at the end. Sections in
    # order: a metadata header, the goal (the user's first message), the summary, topics,
    # key decisions, recent reasoning, pointers to the files, and a closing instruction.
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
    # Slugify the folder name for the download filename: every run of characters that is
    # not a letter or digit collapses to one '-', and leading/trailing dashes are trimmed.
    # "my proj (v2)" -> "my-proj-v2". Keeps the name safe on every filesystem.
    folder = re.sub(r"[^a-zA-Z0-9]+", "-", (d.get("folder_name") or "session")).strip("-")
    filename = f"context-{folder}-{sid[:8]}.md"
    # Redact secrets before this primer can be copied / downloaded / bridged out.
    # This is the last point the text is under our control: after this it is on a
    # clipboard, in a downloaded file, or being read by another CLI (which sends it to a
    # model provider). redact.py masks API keys, tokens, passwords in URLs, private keys.
    markdown = _redact.redact("\n".join(L))
    return markdown, filename


@app.get("/api/sessions/<sid>/context")
def api_context(sid: str):
    """GET /api/sessions/<sid>/context -> the session's context primer (see _build_context).

    Default: JSON `{"markdown": "...", "filename": "context-<project>-<id8>.md"}`, which
    the SPA's "Copy Context" button puts on the clipboard.
    With `?download=1`: the same Markdown as a text/markdown body with a
    `Content-Disposition: attachment` header, so the browser saves it as a file.
    404 `{"error": "not found"}` if there is no session with that id.

    Read-only, and the text is redacted before it leaves.
    """
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


# Per-CLI template for the handoff ("bridge") command, keyed by the TARGET CLI — the one
# taking the work over. `{cwd}` is the session's original project directory and `{file}`
# the primer file written by _build_bridge(); both are shell-quoted before formatting.
# `"$(cat <file>)"` makes the shell read the primer and pass it as a single argument —
# that is how each CLI accepts an opening prompt on its command line.
# Adding a CLI here also adds it to /api/sources, since that route unions the two maps.
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
    the source is enabled, else the configured binary name on PATH.

    Note the distinction this whole file rests on: "installed" (the program exists here,
    so a resume/bridge command will actually run) is NOT the same as "has data" (there are
    transcripts on disk to index, which is what an adapter's is_available() reports). A
    registry synced from another laptop is full of sessions whose CLI is not installed
    here; a freshly installed CLI has a binary but no sessions yet. Both happen.

    The fallback branch exists for a CLI this build knows how to bridge TO but whose
    adapter is disabled in config.toml — there is no adapter object to ask, so the binary
    name from `[sources.<name>].binary` (defaulting to the source name) is looked up on
    PATH directly.
    """
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
    SPA uses it to offer only bridge targets that exist on this machine.

    GET /api/sources -> {"claude": {"enabled": true, "installed": false, "has_data": true},
    "codex": {...}, ...}. Always 200.

    The three flags answer three different questions and are genuinely independent:
      enabled   — this build has an adapter for it and config.toml did not turn it off
      installed — the binary is on PATH here, so a command we hand back will run
      has_data  — the adapter found transcripts on disk, so there is something to index

    The SPA greys out a Bridge target that is not installed instead of letting the user
    pick it and receive a 409 afterwards.
    """
    # Union of "sources we can index" and "sources we can bridge to", de-duplicated while
    # keeping order (dict.fromkeys preserves first-seen order; set() would not).
    names = list(dict.fromkeys([*SOURCES, *_BRIDGE_CMD]))
    out = {}
    for name in names:
        adapter = SOURCES.get(name)
        out[name] = {"enabled": adapter is not None, "installed": _installed(name),
                     "has_data": bool(adapter.is_available()) if adapter is not None else False}
    return jsonify(out)


def _build_bridge(conn, sid: str, target: str) -> dict | None:
    """Write the handoff primer for `sid` and return the command that opens it in `target`.

    Takes the context primer from _build_context(), puts a short instruction header in
    front of it ("you are taking over an in-progress task ..."), saves the result to
    `~/.session-browser/bridges/<id8>-<source>-to-<target>.md`, and formats the matching
    _BRIDGE_CMD template around that path.

    Returns {command, primer, path, target, source, origin_cwd_exists}, or None if there
    is no session with that id.

    Side effect: creates the bridges/ directory if needed and writes (or overwrites) that
    one file. Overwriting is intended — the same handoff rebuilt is the same file, and the
    id-plus-direction name keeps distinct handoffs apart.

    The primer is already redacted: _build_context() does it before returning.
    """
    built = _build_context(conn, sid)
    if built is None:
        return None
    context_md, _ = built
    row = conn.execute(
        "SELECT cli_source, cwd, folder_name FROM sessions WHERE session_id = ?", (sid,)
    ).fetchone()
    source = row["cli_source"]
    cwd = row["cwd"] or ""
    # Does the directory this session was worked in still exist HERE? A registry synced
    # from another laptop is full of paths that do not. This one boolean decides both the
    # warning in the primer and whether the command keeps its leading `cd`.
    cwd_exists = bool(cwd) and Path(cwd).is_dir()

    # Addressed to the receiving assistant, not to the user: it is the first thing that
    # CLI reads, and without it the primer looks like a report rather than an instruction.
    header = (
        f"# Handoff: continue this {source} session in {target}\n\n"
        f"You are taking over an in-progress task that was being worked on in the "
        f"**{source}** CLI. No transcript is being resumed — the full context is below. "
        f"Read it, then continue the work from where it left off. For deeper detail you "
        f"may open the referenced transcript and decision-trail files directly.\n\n"
    )
    if cwd and not cwd_exists:
        # `cd <cwd> && …` would die at the cd — after the user pasted it.
        header += (f"Note: the original project directory `{cwd}` does not exist on this "
                   f"machine; the command below starts in the current directory.\n\n")
    header += "---\n\n"
    primer = header + context_md

    # The primer goes to a file rather than into the command itself: it is a page of
    # Markdown, far past what a shell will accept as one pasted argument, and a file can
    # be re-read or edited before the handoff is actually run.
    bridges = Path.home() / ".session-browser" / "bridges"
    bridges.mkdir(parents=True, exist_ok=True)
    fpath = bridges / f"{sid[:8]}-{source}-to-{target}.md"
    fpath.write_text(primer, encoding="utf-8")

    tmpl = _BRIDGE_CMD.get(target)
    # Drop the `cd <dir> &&` prefix when that directory is gone: `&&` means the CLI only
    # runs if the cd succeeded, so the whole command would die at the first step — after
    # the user had already pasted it. Without the prefix the CLI starts in whatever
    # directory the terminal is in, and the primer says why.
    if tmpl and not cwd_exists:
        tmpl = tmpl.replace("cd {cwd} && ", "")
    # The `if tmpl else` arm is unreachable through the HTTP route (api_bridge rejects an
    # unknown target with 400 first); it keeps this helper safe to call directly.
    command = (tmpl.format(cwd=shlex.quote(cwd), file=shlex.quote(str(fpath)))
               if tmpl else f"# unsupported target {target}")
    return {"command": command, "primer": primer, "path": str(fpath),
            "target": target, "source": source, "origin_cwd_exists": cwd_exists}


@app.post("/api/sessions/<sid>/bridge")
def api_bridge(sid: str):
    """POST /api/sessions/<sid>/bridge?target=<cli> -> a handoff into another CLI.

    "Bridging" moves work from the CLI that recorded a session to a different one. Since
    no CLI can open another's transcript, this writes a primer (the briefing built by
    _build_context) to a file and returns the command that starts a NEW session in the
    target CLI with that primer as its opening prompt.

    Query parameter `target` must be one of _BRIDGE_CMD: claude, copilot, codex, opencode.

    200 body: {command, primer, path, target, source, origin_cwd_exists} — `command` is
    for the user's terminal, `primer` is the full text (the SPA offers it as a download
    without a second request), `path` is where it was saved.
    403 the CSRF header is missing. 400 the target is unknown. 409 the target CLI is not
    installed on this machine. 404 no session with that id.

    Side effect: writes one file under ~/.session-browser/bridges/. Nothing is executed.
    """
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

    GET /api/sessions/<sid>/thread -> {"folder": ..., "cwd": ..., "siblings": [row, ...]}
    where each sibling has the usual `_row_to_dict` shape. 404 if there is no such session.

    This is what lets the detail view show "the other 12 conversations about this project",
    which is usually how you find the session you actually meant. Capped at 50 siblings;
    only visible rows (live plus aged-out, never subagent noise) are listed.
    """
    conn = indexer.connect()
    try:
        row = conn.execute(
            "SELECT cwd, folder_name FROM sessions WHERE session_id = ?", (sid,)
        ).fetchone()
        if not row:
            return jsonify({"error": "not found"}), 404
        # Preferred: the exact working directory. `folder_name` is only the last path
        # segment, so two unrelated checkouts both called "api" would look like one thread.
        if row["cwd"]:
            rows = conn.execute(
                "SELECT * FROM sessions WHERE cwd = ? AND session_id <> ? "
                f"AND {indexer.VISIBLE} ORDER BY last_activity DESC LIMIT 50",
                (row["cwd"], sid),
            ).fetchall()
        else:
            # No cwd recorded (some older rows, some sources): fall back to the folder
            # name. Less precise, but better than showing no thread at all.
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
    """GET /api/sessions/<sid>/reasoning -> the session's decision trail.

    A "reasoning trail" is a readable Markdown rendering of how the assistant worked
    through the session — its own thinking text plus the actions it took — extracted
    earlier by scripts/extract-reasoning.py and stored as a file under the reasoning
    archive; the row only holds the path in `reasoning_path`.

    Default: JSON `{"markdown": ..., "path": ..., "title": ...}`.
    With `?format=md`: the raw Markdown as a text/markdown body (what "Open trail" uses).
    404 `{"error": "not found"}` if there is no such session; 404
    `{"error": "no reasoning captured", "markdown": ""}` if the row has no trail, or the
    file it points at has since been deleted.

    Read-only, and redacted on the way out.
    """
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
    # errors="replace" rather than strict: one bad byte in a years-old trail must render
    # as a replacement character, not turn the whole page into an error. The 2MB read cap
    # keeps a runaway trail from being loaded into memory and shipped to the browser.
    with open(rp, "r", encoding="utf-8", errors="replace") as fh:
        markdown = _redact.redact(fh.read(2_000_000))
    if request.args.get("format") == "md":
        # Redundant with the module-level `from flask import Response` — same class, same
        # behaviour; harmless, and left alone rather than churned.
        from flask import Response
        return Response(markdown, mimetype="text/markdown")
    return jsonify({"markdown": markdown, "path": rp,
                    "title": row["title"] or row["first_message"]})


@app.get("/api/stats/timeseries")
def api_stats_timeseries():
    """Aggregates for the usage dashboard — all derived from existing columns via
    GROUP BY (no schema change). Powers the activity heatmap, per-day tokens/cost,
    and per-model / per-project / per-source breakdowns.

    GET /api/stats/timeseries -> {per_day, by_model, by_project, by_source, totals,
    month_cost, billing, notional}. No parameters. Always 200.

    Every list is `[{"...": name, "sessions": n, "tokens": n, "cost": usd}, ...]`. All five
    queries share the same shape, so read the first one and the rest follow:
      * `COALESCE(x, 0)` per token column before adding, never `SUM(a+b+c+d)` — in SQL
        anything + NULL is NULL, so one missing column would zero a whole session's tokens;
      * `indexer.VISIBLE` everywhere, so aged-out sessions still count (that spend really
        happened) while subagent noise never does;
      * `cost` is the public list-price equivalent, and `notional` in the response says
        whether it is money actually billed — under a flat subscription it is not, and the
        UI prefixes it with "≈".

    Read-only.
    """
    conn = indexer.connect()
    try:
        # per-day: session count, tokens, cost — last 365 days
        # substr(last_activity,1,10) takes the 'YYYY-MM-DD' out of the stored timestamp —
        # cheap because the column's own text layout makes the date a fixed-width prefix.
        # Rows with an empty timestamp are dropped; they would group under a blank day.
        # Ordered ascending so the heatmap can be drawn straight through.
        per_day = [dict(r) for r in conn.execute(
            "SELECT substr(last_activity,1,10) AS day, COUNT(*) AS sessions, "
            "COALESCE(SUM(COALESCE(input_tokens,0)+COALESCE(output_tokens,0)+COALESCE(cache_read_tokens,0)+COALESCE(cache_write_tokens,0)),0) AS tokens, "
            "COALESCE(SUM(cost_usd),0) AS cost "
            f"FROM sessions WHERE {indexer.VISIBLE} AND last_activity!='' "
            "GROUP BY day ORDER BY day"
        ).fetchall()]
        # Per model, most expensive first, top 20. A session whose model was never
        # recorded groups under the literal 'unknown' rather than vanishing into NULL.
        by_model = [dict(r) for r in conn.execute(
            "SELECT COALESCE(model_used,'unknown') AS model, COUNT(*) AS sessions, "
            "COALESCE(SUM(COALESCE(input_tokens,0)+COALESCE(output_tokens,0)+COALESCE(cache_read_tokens,0)+COALESCE(cache_write_tokens,0)),0) AS tokens, "
            f"COALESCE(SUM(cost_usd),0) AS cost FROM sessions WHERE {indexer.VISIBLE} "
            "GROUP BY model ORDER BY cost DESC LIMIT 20"
        ).fetchall()]
        # Per project, top 15 by cost. NULLIF(folder_name,'') turns the empty string into
        # NULL so COALESCE can catch both "never recorded" and "recorded as empty" and
        # label them with a dash — the dropdown drops such rows, the chart shows them.
        by_project = [dict(r) for r in conn.execute(
            "SELECT COALESCE(NULLIF(folder_name,''),'—') AS project, COUNT(*) AS sessions, "
            "COALESCE(SUM(COALESCE(input_tokens,0)+COALESCE(output_tokens,0)+COALESCE(cache_read_tokens,0)+COALESCE(cache_write_tokens,0)),0) AS tokens, "
            f"COALESCE(SUM(cost_usd),0) AS cost FROM sessions WHERE {indexer.VISIBLE} "
            "GROUP BY project ORDER BY cost DESC LIMIT 15"
        ).fetchall()]
        # Per CLI — no LIMIT, there are only a handful of them.
        by_source = [dict(r) for r in conn.execute(
            "SELECT cli_source AS source, COUNT(*) AS sessions, "
            "COALESCE(SUM(COALESCE(input_tokens,0)+COALESCE(output_tokens,0)+COALESCE(cache_read_tokens,0)+COALESCE(cache_write_tokens,0)),0) AS tokens, "
            f"COALESCE(SUM(cost_usd),0) AS cost FROM sessions WHERE {indexer.VISIBLE} "
            "GROUP BY source ORDER BY cost DESC"
        ).fetchall()]
        # Grand totals across everything visible — the same sums with no GROUP BY.
        totals = dict(conn.execute(
            "SELECT COUNT(*) AS sessions, "
            "COALESCE(SUM(COALESCE(input_tokens,0)+COALESCE(output_tokens,0)+COALESCE(cache_read_tokens,0)+COALESCE(cache_write_tokens,0)),0) AS tokens, "
            f"COALESCE(SUM(cost_usd),0) AS cost FROM sessions WHERE {indexer.VISIBLE}"
        ).fetchone())
        # this-calendar-month cost (the headline "≈$X of API-equivalent" number)
        # strftime('%Y-%m-01','now') is the 1st of the current month, '2026-09-01'. That is
        # a prefix of the stored 'YYYY-MM-DDT...' spelling, and '2026-09-01' sorts at or
        # below every stamp on that day, so the whole 1st is included.
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
    """GET /api/stats -> the counts in the page header.

    Body: {total, enriched, folders, archived, by_source, billing}. `enriched` is how many
    of those sessions have a written summary; `by_source` is {"claude": 41, "codex": 7};
    `archived` is always the full Archived-tab count regardless of scope, because it is
    what the tab's own badge shows. Always 200, read-only.

    Optional `state=live|archived` scopes the first four numbers and `by_source` to one
    tab; with no parameter they cover everything visible.
    """
    # ?state=live|archived scopes the counts to one tab so the source pills
    # match the list beneath them; the default (everything visible) is what
    # the header line and other consumers want.
    # `pred` is one of the named predicates from indexer.py, chosen by ?state and NEVER
    # built from user text — it is interpolated straight into the SQL below, so the lookup
    # (with VISIBLE as the default for any unrecognised value) is what keeps that safe.
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
        # Deliberately NOT scoped by `pred`: this number is the Archived tab's own badge,
        # so it must read the same whichever tab you are looking at.
        archived = conn.execute(
            f"SELECT COUNT(*) FROM sessions WHERE {indexer.ARCHIVED_VISIBLE}"
        ).fetchone()[0]
    finally:
        conn.close()
    return jsonify({"total": total, "enriched": enriched, "folders": folders,
                    "archived": archived, "by_source": by_source, "billing": sbconfig.BILLING})


# Only when this file is RUN (`python session-ui/app.py`), not when it is imported — the
# test suite imports it as a module and drives `app` with Flask's test client, which must
# not start a server.
if __name__ == "__main__":
    # Create ~/.session-browser and its subdirectories if this is a first run, so the
    # bridges/ write later cannot be the thing that discovers they are missing.
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
