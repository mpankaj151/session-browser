# Glossary

Plain-language definitions of the terms this codebase uses, written for someone who has
never used an AI coding assistant. Module docstrings point here instead of re-explaining
each term. Terms are grouped by theme; **bold** marks the defined term.

## The tools being indexed

- **AI coding assistant / coding CLI** — a terminal program (Claude Code, GitHub Copilot
  CLI, OpenAI Codex CLI, OpenCode) that you chat with about your code. It reads files,
  runs commands and edits code on your behalf, sending your conversation to a large
  language model over the network.
- **Large language model (LLM), model** — the AI system that produces the replies. Each
  vendor offers several models (e.g. `claude-opus-5`, `gpt-5.5`) that differ in
  capability, speed and price. This tool never talks to a model directly; it only reads
  what the CLIs recorded and, for enrichment, asks a CLI to run one for it.
- **Provider** — the company or gateway serving a model (Anthropic, OpenAI, GitHub, a
  local Ollama server). OpenCode names models as `provider/model`, e.g.
  `anthropic/claude-sonnet-5`.
- **Session** — one conversation with a coding CLI, from the first prompt to the last
  reply. Every session has an id, a working directory and a time span. A session is the
  unit this tool indexes: one row in the registry per session.
- **Transcript** — the file (or, for OpenCode, database rows) in which a CLI records a
  session: every message, tool call and result, with timestamps. Claude Code, Copilot and
  Codex write one JSONL file per session under their home directories.
- **Turn** — one exchange in a session: a user message plus the assistant reply that
  answers it. `turn_count` counts user messages that carry real text.
- **Prompt** — the text sent to a model: what the user typed, plus any instructions and
  context the CLI adds.
- **Context window** — the maximum amount of text a model can consider at once. When a
  session grows past it the CLI *compacts* (summarises older history into a short
  recap). Compaction messages are noise for our purposes and are skipped.
- **Tool call** — a step where the model asks the CLI to do something (run a shell
  command, read or edit a file) and gets the result back. Transcripts record the request
  and its output; large outputs may be spilled to separate files.
- **Subagent / sidechain** — a helper conversation a CLI starts to work on a sub-task in
  parallel. Its transcript is a child of the main session, not a session of its own, so
  it must never become a registry row.
- **Resume** — reopening an old session in its CLI so the conversation continues with
  its history intact (`claude --resume <id>`, `opencode --session <id>` ...). The
  `cr` shell helper (`bin/resume-here.sh`) picks the right command for a session id.
- **Rollout** — Codex's name for a session transcript file (`rollout-<timestamp>-<id>.jsonl`).
  Old rollouts are compressed to `.jsonl.zst` (zstd) after a week.
- **Hook** — a small program a CLI runs at fixed moments. Claude Code runs a *Stop* hook
  when a reply finishes and a *SessionEnd* hook when the session closes; this tool installs
  `scripts/session-hook.py` on both to index the session instantly. OpenCode calls the
  same idea a **plugin** (a JavaScript file it loads at start-up).
- **Headless run** — running a coding CLI non-interactively, with the prompt on standard
  input and the answer on standard output, no terminal UI (`claude --print`,
  `opencode run`, `copilot -p`). Enrichment uses headless runs.

## Usage and money

- **Token** — the unit models read and write text in: a word piece, roughly four
  characters of English. Every request is billed per token, so token counts are the raw
  measure of how much a session "used".
- **Input tokens / output tokens** — tokens sent to the model (your prompt and history)
  versus tokens it generated (its reply). Output tokens cost several times more.
- **Cache read / cache write tokens** — vendors let a CLI *cache* the unchanged start of
  a long conversation on their side. Writing to that cache costs a little more than a
  normal input token; reading from it costs about a tenth. Long sessions are mostly cache
  reads, which is why they are cheaper than their raw size suggests.
- **Cost (USD)** — the public list price for a session's tokens, computed from
  `pricing.json` by `scripts/compute-costs.py`. Under a flat subscription nothing is
  actually billed per session; the figure is then shown with `≈` as a measure of
  intensity, not money spent. OpenCode records a per-message cost itself, which is used
  as-is.
- **Pricing tier** — one row in `pricing.json`: the per-million-token prices for a model
  generation, matched to a session's model name by the longest matching alias.

## What this tool derives from a session

- **Registry** — the SQLite database (`registry.db`) with one row per session plus the
  derived data below. Everything the UI, the reports and the MCP server show comes from
  it.
- **Indexing** — reading a transcript's cheap header facts (id, directory, times, turn
  count, first message, model) and upserting them into the registry. Done by the hook,
  the watcher and the backfill; never touches derived columns.
- **Watcher** — the always-on background process (`watcher.py`, run by launchd on macOS
  or a systemd user unit on Linux) that notices transcript files being created, changed
  or deleted and indexes or archives the matching rows.
- **Enrichment** — asking a model, through a headless CLI run, to read a session and
  write a short structured summary: title, summary, topics, session type, outcome,
  decisions, open threads. Produced by `scripts/enrich-sessions.py`; the result is called
  a **facet** and is stored both as JSON under `facets/` and in registry columns. This is
  the tool's only LLM call and it spends your own CLI quota.
- **Summariser** — the model plus CLI that enrichment runs through, chosen by
  `[enrichment].provider` in `config.toml` (`claude-headless`, `opencode-headless`,
  `copilot-headless`, or `auto` to take whichever CLI is installed).
- **Work journal** — the richer, journal-grade enrichment (accomplishments, key decisions
  with rationale, explorations set aside, open threads) that the SessionEnd hook produces
  per session, plus the **daily digest**: a Markdown page per local day assembled from
  those journals without any model call (`scripts/daily-digest.py`).
- **Topics / classification** — short keyword tags per session (`python`, `testing`,
  `ci-cd`), either from enrichment or from the keyword rules in
  `scripts/classify-topics.py`.
- **Reasoning trail** — the model's own "thinking" text (Claude Code and OpenCode record
  it; Codex records a summary) extracted per session into a readable Markdown file by
  `scripts/extract-reasoning.py`. Shows *why* the assistant made each decision.
- **Reasoning archive / raw vault** — the folder (default `~/claude-reasoning-archive`)
  holding those Markdown trails under `readable/` and, under `raw/`, a byte-for-byte copy
  of every transcript, versioned when the file grows. The raw copy is what makes a
  session restorable after its CLI deletes the original.
- **Full-text search (FTS)** — SQLite's built-in word index (FTS5) over transcript text,
  built by `scripts/build-fts.py`. Finds sessions by exact words, like grep.
- **Embedding** — a list of numbers (a *vector*, here 384 of them) that a small local
  model produces from a piece of text, arranged so that texts with similar *meaning* get
  similar numbers. Produced by `scripts/embed-sessions.py` with the `sentence-transformers`
  library and stored as raw float bytes in the registry.
- **Semantic search / cosine similarity** — searching by meaning: embed the query, then
  rank sessions by the angle between the query vector and each session vector (cosine
  similarity, 1.0 = identical direction). Finds "the time I fixed the flaky checkout
  tests" even when no word matches. Done with numpy in `semsearch.py`; no vector database
  needed at this scale.
- **Bridge / primer** — a short Markdown briefing about a session (goal, decisions, open
  threads) written so another CLI can pick the work up: "continue this Claude session in
  Codex". The UI's Bridge button builds one and hands it to the target CLI.
- **Redaction** — masking secrets (API keys, tokens, passwords in URLs, private keys)
  in any text that leaves the tool: exports, bridges, search indexes, enrichment prompts
  and MCP replies. Implemented in `redact.py`.

## Lifecycle of a row

- **Live** — the session's transcript still exists where its CLI keeps it.
- **Archived** — the transcript is gone but the row stays. `archived_reason` says why:
  `transcript-missing` (the CLI deleted an old file, e.g. Claude Code's 30-day
  `cleanupPeriodDays`, or `opencode session delete`) keeps the row visible and counted;
  `not-a-session` (a subagent sidechain that never held a conversation) hides it.
- **Restore** — copying the newest raw-vault copy back to where the CLI expects it and
  re-indexing, so the session is live and resumable again (`scripts/restore-session.py`).
  For OpenCode it also re-imports the session into OpenCode's database.
- **Mirror (OpenCode)** — OpenCode stores all sessions in one SQLite database. The
  adapter *projects* that database into one JSONL file per root session under
  `opencode-mirror/`, because every other part of this tool works on one file per
  session. The mirror is also the backup; `scripts/backup-opencode.py` additionally
  snapshots the whole database weekly.
- **Prune / reconcile** — housekeeping passes (`scripts/prune-sessions.py`,
  `scripts/reconcile-sessions.py`) that archive rows whose transcript no longer maps to
  any file and repair rows the watcher missed.

## Integration surfaces

- **MCP (Model Context Protocol)** — a standard way for an AI assistant to call external
  tools. `mcp/session-memory` is an MCP server exposing six tools (recent sessions,
  search, session detail ...) so a coding CLI can ask "what did I work on last week"
  and get answers from the registry.
- **Skill** — a Markdown instruction file (`skills/*/SKILL.md`) that Claude Code loads on
  demand to know how to use a workflow, e.g. the work-journal reports.
- **UI** — the local web page (`session-ui/app.py`, a Flask server, plus one vanilla
  JavaScript file) served on the configured port; never reachable from other machines.

## Plumbing words

- **JSONL** — "JSON Lines": a text file with one JSON object per line. All transcripts
  and the OpenCode mirror use it because it can be appended to and read one line at a
  time.
- **WAL** — SQLite's write-ahead-log mode: writes go to a side file first, so readers are
  never blocked by a writer. Both the registry and OpenCode's database use it; a WAL
  write is what the watcher listens for to notice new OpenCode messages.
- **Upsert with COALESCE** — inserting a row, or updating it if it exists, while keeping
  every existing non-empty value. This is how re-indexing a session never wipes out its
  enrichment.
- **Adapter / source** — one Python class per CLI (`sources/*.py`) implementing the same
  small protocol (discover transcripts, parse a header cheaply, parse a full transcript,
  build the resume command). Nothing else in the tool knows which CLI a row came from.
- **launchd / systemd** — the macOS and Linux services that keep the watcher running and
  run the nightly refresh; `scripts/render-job.py` writes their job files from templates.
- **venv** — the private Python environment under `.venv/` that the installer creates so
  the tool's dependencies never touch the system Python.
- **Race guard (hookstate)** — a 30-second note that "the hook just handled this
  session", so the watcher does not index the same file a second time.
