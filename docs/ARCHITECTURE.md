# Architecture

Session Browser is a local pipeline that indexes AI-CLI transcripts into one
SQLite database, plus a Flask UI and an MCP server on top. Everything is
source-agnostic behind an adapter protocol.

## Data flow

```mermaid
flowchart TD
    subgraph Sources
      C[~/.claude/projects/*.jsonl]
      P[~/.copilot/session-state/*/events.jsonl]
      X[~/.codex/sessions/**/rollout-*.jsonl]
      O[~/.local/share/opencode/opencode.db<br/>→ opencode-mirror/ses_*.jsonl]
    end

    C & P & X & O --> AD[sources/*.py adapters<br/>SessionSource protocol]

    subgraph Indexing
      HOOK[session-hook.py<br/>Claude Stop hook] --> IDX
      WATCH[watcher.py<br/>launchd / systemd daemon] --> IDX
      BACK[backfill.py] --> IDX
      IDX[indexer.py<br/>COALESCE upsert] --> DB[(registry.db)]
    end
    AD --> HOOK & WATCH & BACK

    subgraph Enrichment pipeline - nightly
      COST[compute-costs.py] --> DB
      REAS[extract-reasoning.py] --> DB
      REAS --> ARCH[~/claude-reasoning-archive]
      FTS[build-fts.py] --> DB
      EMB[embed-sessions.py] --> DB
      ENR[enrich-sessions.py<br/>claude --print | opencode run] --> DB
    end
    DB --> COST & REAS & FTS & EMB & ENR

    subgraph Consumers
      UI[session-ui/app.py<br/>Flask + SPA]
      MCP[mcp/session-memory<br/>6 tools]
      CR[cr / sb shell]
    end
    DB --> UI & MCP
    AD --> CR
```

## Key design decisions

**Adapter protocol (`sources/base.py`).** Every CLI implements `SessionSource`:
`discover`, `parse_header` (cheap, no full read), `parse_full`,
`session_id_for_path` (identity without reading — used on delete),
`resume_command`, `is_available`. The indexer, DB, UI, watcher, and MCP server
never mention a specific CLI — adding one is a new file + one registry line.

**DB-backed sources: the mirror pattern (`sources/opencode.py`).** Every
consumer — the watcher's delete handler, `archive_raw`, restore, FTS, the cost
and reasoning extractors — assumes one plain-text file per session at the path
`discover()` yields. OpenCode keeps everything in one SQLite DB, so its adapter
*projects* the DB (opened read-only; the binary is never invoked on this path
because even `opencode db path` rewrote the WAL) into one JSONL per root
session: line 1 is the export-shaped `Session.Info` + children + the stats every
consumer reads, then one line per message with its parts. Children roll their
cost into the root. A manifest of per-tree fingerprints limits rewrites to
changed sessions; a session gone from the DB is archived to the raw vault
before its mirror file is unlinked, so the ordinary delete path archives the
row as transcript-missing and Restore can bring it back.

**COALESCE upsert (`indexer.py`).** Re-indexing a session must never clobber
enrichment (summary, topics, cost, reasoning_path). The upsert updates cheap
fields (last_activity, turn_count) but `COALESCE(NULLIF(old,''), NULLIF(new,''))`
preserves everything derived, and treats `''` as absent so a header parsed
before the first user turn doesn't pin a field empty. Every upsert also sets
`archived=0` and clears `archived_reason` — a parsed file exists, so the
session is alive.

**Archive lifecycle (`archived`, `archived_reason`).** Rows are never deleted.
`archive()` flips `archived=1` and must say why: `transcript-missing` (the
canonical file vanished — Claude Code's `cleanupPeriodDays`, a manual `rm`) or
`not-a-session` (a subagent sidechain / workflow journal that never held a
conversation). Without the reason the two are indistinguishable, and an
Archived view built on the flag alone drowns real sessions under sidechain
noise. Consumers never spell the flag: `indexer.LIVE` (a file exists — for
enrichment, reasoning extraction, prune), `indexer.VISIBLE` (what the user sees
and what usage stats count: live + transcript-missing), `indexer.ARCHIVED_VISIBLE`
(the Archived tab); a test greps the tree to keep it that way. `restore.py`
brings a transcript-missing session back from `<archive>/raw/` — `refresh-all`
copies every indexable transcript there before extracting reasoning, which
makes the reasoning archive a durable transcript vault. Restore uses
`copyfile`, not `copy2`: the restored file needs a fresh mtime or an age-based
cleanup would delete it again on its next pass.

**Two-tier live indexing.** The Claude **Stop hook** (and, opt-in, the
**OpenCode plugin** — `scripts/opencode-hook.py` spawned on `session.idle` /
`session.deleted`) indexes a session the instant it ends (tens of ms) and
detaches reasoning extraction. The **watcher** (a launchd / systemd --user daemon,
singleton-locked) catches everything else — Copilot, Codex, OpenCode DB writes
via `sync_trigger()`, and anything a hook missed — via filesystem events, with
a 30s race-guard (`hookstate.py`, shared by every hook) so the two paths never
double-process. Hooks are contractually exit-0 so a broken config can never
block a CLI's session end.

**Timestamps.** `to_iso_utc()` normalizes every source to one canonical,
lexicographically-sortable UTC form, so mixed Claude/Copilot/Codex lists order
correctly under a plain `ORDER BY`.

**Vector search without a native extension.** pyenv's `sqlite3` can't load
`sqlite-vec`, so embeddings are stored as float32 BLOBs and searched with a
numpy brute-force cosine — trivially fast for thousands of sessions. If an
extension-capable interpreter is present, the `sessions_vec` fast path is used
automatically. The stored `dim` is the true vector length, so changing
`[embeddings].model` triggers a clean re-embed instead of a crash.

**Redaction at every egress.** Anything that leaves the tool — Copy Context,
Export, Bridge, the full-text index, the reasoning archive, the LLM enrichment
prompt and its persisted output, and every MCP tool result — passes through
`redact.py`, which masks API keys, tokens (`sk-`, `sk_live_`, `github_pat_`,
`npm_`, `xox*`, AWS, Google, JWTs, `Authorization:` headers), `*_SECRET`/`*_KEY`
assignments (including JSON form), `user:password@` URL credentials, and
private-key blocks. Bare hashes in prose survive so the FTS index stays
searchable by commit SHA.

**Cost is notional under a subscription.** Token totals are real; the dollar
figure is the public-API list-price equivalent. Under a flat plan (`[billing]
mode = "subscription"`) it's shown with `≈` as an intensity signal, not money
billed.

## Storage

| What | Where |
|------|-------|
| Registry (sessions, artifacts, embeddings, FTS) | `~/.session-browser/registry.db` (WAL) |
| Enrichment facets / bridge primers | `~/.session-browser/{facets,bridges}/` |
| Raw transcripts + readable reasoning trails | `~/claude-reasoning-archive/{raw,readable}/YYYY/MM/` |
| OpenCode mirror (one JSONL per root session; also the backup) | `~/.session-browser/opencode-mirror/` |

## Module map

```
sources/{base,claude,copilot,codex,opencode,registry}.py   adapters + protocol
indexer.py                                         upsert / archive
watcher.py + scripts/session-hook.py               two-tier live indexing
reasoning.py + scripts/extract-reasoning.py        decision trails
costs.py + scripts/compute-costs.py                tokens -> USD
semsearch.py + scripts/embed-sessions.py           numpy cosine vector search
redact.py                                          secret masking
enrichment/                                         pluggable LLM summarizers
session-ui/app.py + static/                         Flask API + vanilla-JS SPA
mcp/session-memory/                                 6 MCP tools
```
