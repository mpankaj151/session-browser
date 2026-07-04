# Session Browser

A local, privacy-first browser for your AI coding CLI sessions (Claude Code +
GitHub Copilot CLI, extensible to others). One searchable UI over every past
session — keyword, semantic, and full-text search, LLM summaries and topic tags,
per-session token usage and cost, one-click **resume from any directory**,
cross-CLI session **bridging**, an MCP server so Claude can recall past work,
and a **decision/reasoning trail** for every session.

Everything runs and stays on your machine: SQLite + Flask + vanilla JS. No
cloud services, no telemetry.

## Requirements

- **Python 3.11+** (3.12 recommended)
- **macOS or Linux.** Background scheduling (live watcher + nightly refresh) is
  automated via launchd on macOS; on Linux run the same two scripts via
  systemd/cron (the installer prints the exact commands).
- At least one supported CLI with session history:
  - [Claude Code](https://claude.com/claude-code) (`~/.claude/projects`)
  - [GitHub Copilot CLI](https://github.com/github/copilot-cli) (`~/.copilot/session-state`)
- Optional: ~2 GB disk for the semantic-search stack (torch +
  sentence-transformers). Skip with `--lite`; search falls back to
  keyword + full-text.

## Quick start

```bash
git clone <this-repo> && cd session-browser
./install.sh                 # venv, deps, schema, backfill, hook, launchd
./bin/start-session-ui.sh    # http://127.0.0.1:7655
./bin/doctor.sh              # health check — every subsystem, one screen
```

Install flags:

| Flag | Effect |
|------|--------|
| `--lite` | skip torch/sentence-transformers (~2 GB); no semantic search |
| `--enrich` | run LLM summarization during install (uses your Claude quota) |
| `--no-hook` | don't register the Claude Stop hook |
| `--no-launchd` | don't install the background jobs (macOS) |
| `--no-backfill` | skip indexing existing sessions |

Reverse everything with `./uninstall.sh` (`--purge` also drops the database).

Then install the shell helpers (adds `cr` and `sb` functions to your rc file):

```bash
./bin/install-cr.sh && source ~/.zshrc
```

```bash
cr <session_id>          # resume ANY session in the CURRENT directory
sb ui / open / stop      # start / open / stop the web UI
sb doctor                # health check
sb refresh [--enrich]    # run the indexing pipeline now
```

## Features

### Browse & search everything

Cards grouped by recency with title, summary, topics, source badge, token/cost
badge, and turn count. Filter by folder, source, topic, or time window. The
search bar has three modes: **Keyword** (title/summary/topics), **Semantic**
(vector similarity over embeddings), and **Full-text** (FTS5 over the actual
conversation text — find sessions by what was *discussed*).

### Resume from anywhere (`cr <session_id>`)

A plain `claude --resume <id>` only works from the session's original project
directory. `cr` ports the session's memory to wherever you are, then resumes:

- **Claude:** symlinks `~/.claude/projects/<orig>/<id>.jsonl` into the encoded
  project dir for `$PWD`, then `claude --resume <id>`. Symlinks (not copies)
  mean one canonical transcript — origin dir, browser, and reasoning archive
  stay in sync as you keep working.
- **Copilot:** repoints the session's `workspace.yaml` cwd to `$PWD` (backup
  kept), then `copilot --resume=<id>`.

The UI's **Resume** button copies exactly this `cr <id>` command.
`bin/doctor.sh` flags any legacy diverged copies;
`scripts/reconcile-sessions.py --apply` consolidates them safely (forks are
archived, never deleted).

### The reasoning / decision trail

For each session it reconstructs, turn by turn, the **visible reasoning** the
assistant wrote and the **exact action sequence** it took, rendered as a
readable Markdown trail — per-card **🧠 Reasoning** button, plus raw + readable
archives under `~/claude-reasoning-archive/{raw,readable}/YYYY/MM/`.

> **Honest limitation:** current Claude Code versions store extended-thinking
> blocks with their text **empty** (only a cryptographic signature) — internal
> chain-of-thought is not persisted to disk and cannot be recovered. The trail
> captures visible reasoning + actions and flags turns where hidden thinking
> occurred. Copilot, by contrast, *does* persist `reasoningText`.

### Carry context between sessions

**📋 Copy Context** / **⬇ Export** build a portable markdown primer — goal,
summary, topics, key decisions, recent visible reasoning, token/cost, the `cr`
resume command, and pointers to the transcript + decision trail. Paste it into
a new session to hand the work over.

### Bridge — hand a session off to another CLI

No CLI can natively resume another's session, so **🌉 Bridge** writes a
target-specific handoff primer and gives you a command that starts a fresh
session in the other CLI, in the same project dir, seeded with the full
context:

```bash
cd <cwd> && codex   "$(cat ~/.session-browser/bridges/<id>-claude-to-codex.md)"
cd <cwd> && copilot -p "$(cat ~/.session-browser/bridges/<id>-claude-to-copilot.md)"
cd <cwd> && claude  "$(cat ~/.session-browser/bridges/<id>-copilot-to-claude.md)"
```

A preview modal shows the command, a copy button, a download button, and the
full primer.

### Thread (sibling sessions)

**🧵 Thread** expands the other sessions in the same folder — cross-CLI, each
with its own description and Resume button.

### Privacy: secret redaction

Any primer that leaves the tool (Copy Context, Export, Bridge) passes through
`redact.py`, which masks API keys, tokens, `*_SECRET`/`*_KEY` assignments,
JWTs, bearer tokens, and private-key blocks before it reaches your clipboard, a
file, or another CLI.

### Token usage & billing

Badges show **token volume + public-API list-price cost** (e.g. `33.1M tok
≈$87.86`). Configure `[billing]` in `config.toml`:

- `mode = "subscription"` (default — e.g. Claude Max): the dollar figure is a
  **notional API-equivalent** (`≈` prefix) — an intensity signal, not money you
  are billed.
- `mode = "api"`: you actually pay per token; the `≈` goes away.

Claude usage comes from each assistant record's `usage`; Copilot from the
per-model `modelMetrics` in its session-shutdown event (reasoning tokens billed
as output). Rates and model→tier aliases live in `pricing.json`.

### MCP server — let Claude recall past work

`mcp/session-memory/server.py` exposes six tools over stdio:
`search_sessions`, `get_session_summary`, `get_session_snippet`,
`list_recent`, `get_decisions`, `get_reasoning`. Register it in your Claude
Code MCP config:

```json
{
  "mcpServers": {
    "session-memory": {
      "command": "/path/to/session-browser/.venv/bin/python",
      "args": ["/path/to/session-browser/mcp/session-memory/server.py"]
    }
  }
}
```

## How it stays fresh

Three tiers, all installed user-level (not per-project):

1. **Stop hook** (`~/.claude/settings.json`) — indexes a Claude session the
   moment it ends, then detaches a background reasoning-trail extraction.
2. **Watcher** (launchd daemon, singleton-locked) — filesystem events over
   every enabled source; catches Copilot sessions and anything the hook missed.
3. **Nightly refresh** (launchd, 01:00) — full pipeline: backfill → topics →
   costs → reasoning → full-text → embeddings → LLM enrichment.

On Linux, run tier 2 and 3 yourself (systemd/cron):
`.venv/bin/python watcher.py` and `.venv/bin/python scripts/refresh-all.py --enrich`.

Headless `claude --print` sessions (`entrypoint: sdk-cli` — e.g. enrichment's
own calls) are excluded from the index, so the tool can't pollute itself.

## Architecture

```
sources/{claude,copilot}.py   adapters (SessionSource protocol)
indexer.py                    COALESCE upsert / archive into registry.db
watcher.py + session-hook.py  two-tier live indexing (daemon + instant hook)
reasoning.py                  decision-trail extraction + render + archive
costs.py / compute-costs.py   tokens -> USD (multi-provider)
semsearch.py / embed-sessions numpy brute-force cosine, 384-dim MiniLM
redact.py                     secret masking for anything that leaves the tool
enrichment/                   pluggable LLM summarizers (claude-headless / null)
session-ui/app.py + static/   Flask API + single-file vanilla-JS SPA
mcp/session-memory/server.py  6 MCP tools incl. get_reasoning
```

### What gets stored where

| What | Location |
|------|----------|
| Registry DB (sessions, artifacts, embeddings, FTS) | `~/.session-browser/registry.db` |
| Enrichment facets | `~/.session-browser/facets/` |
| Bridge primers | `~/.session-browser/bridges/` |
| Logs, hook-state | `~/.session-browser/logs/`, `~/.session-browser/.hook-state.json` |
| Raw transcript archive + readable reasoning trails | `~/claude-reasoning-archive/{raw,readable}/YYYY/MM/` |

All locations are configurable in `config.toml` (created from
`config.toml.example` on install).

### Adding a CLI (codex, opencode, ollama, …)

1. Write `sources/<cli>.py` implementing the `SessionSource` protocol
   (`sources/base.py`).
2. Add it to `_FACTORIES` in `sources/registry.py`.
3. Add a `[sources.<cli>]` block to `config.toml`.

Nothing else changes — the indexer, DB, UI, watcher, and MCP server are
source-agnostic.

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/migrate-db.py` | idempotent schema (WAL, additive columns) |
| `scripts/backfill.py` | index all sessions from enabled sources |
| `scripts/compute-costs.py` | token usage → USD |
| `scripts/extract-reasoning.py` | decision trails (`--backfill --archive`) |
| `scripts/embed-sessions.py` | semantic-search embeddings |
| `scripts/classify-topics.py` | keyword topic tags (no LLM) |
| `scripts/enrich-sessions.py` | LLM summaries (claude-headless) |
| `scripts/build-fts.py` | full-text index over transcript text (FTS5) |
| `scripts/refresh-all.py` | run the whole pipeline (`--enrich` for LLM) |
| `scripts/reconcile-sessions.py` | consolidate diverged session copies (safe) |

## Tests

```bash
.venv/bin/python tests/test_smoke.py
```

Covers facet validation, cost mapping (both providers), reasoning extraction
(both providers), redaction, adapter registration, and the COALESCE upsert
that protects enrichment data.

## Environment notes

- Semantic search uses a **numpy brute-force backend** by default — no native
  SQLite extension needed, trivially fast up to a few thousand sessions. If an
  extension-capable interpreter can load `sqlite-vec`, that fast path is used
  automatically.
- The Stop hook and launchd jobs always use the **absolute** venv interpreter
  path (version managers' shims aren't on PATH in those contexts).
- The embedding model (`all-MiniLM-L6-v2`, ~90 MB) downloads once on first use,
  then runs fully offline.

## License

MIT — see [LICENSE](LICENSE).
