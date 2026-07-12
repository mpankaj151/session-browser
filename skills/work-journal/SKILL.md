---
name: work-journal
description: Generate work summaries from the session-browser registry — across
  Claude Code, Codex CLI, and Copilot CLI. Use when the user asks "what did I do
  today/yesterday/last week/last month/last quarter", requests a weekly-summary,
  monthly-summary, review-summary, performance review, self-assessment,
  accomplishments report, brag doc, stand-up status, or a timeline of their
  coding work. Reads only the enriched work journal (never raw transcripts).
---

# work-journal

Turn the registry's journal-grade session records into a report for any time
window. Capture is automatic (Stop/SessionEnd hooks + watcher + nightly
enrichment); this skill is the read layer.

## Resolve paths first

The skill directory may be a symlink into `~/.claude/skills`. Resolve the repo:

```bash
REPO="$(cd "$(dirname "$(readlink -f "<this skill dir>/SKILL.md")")/../.." && pwd)"
PY="$REPO/.venv/bin/python"
```

Output paths come from `[reports]` / `[digest]` in the repo's config
(defaults: `~/.session-browser/reports`, `~/.session-browser/daily-logs`).

## Mode: daily — "what did I do today / yesterday"

1. `"$PY" "$REPO/scripts/daily-digest.py" --date <YYYY-MM-DD>` (local date; this
   regenerates the file even for today).
2. Read `<daily_dir>/<YYYY-MM-DD>.md` and present it conversationally in
   stand-up framing. No report files are written.

## Modes: weekly-summary / monthly-summary / review-summary

**Step 1 — pre-flight (always):**

```bash
"$PY" "$REPO/scripts/report-data.py" --check-only --window <window>
```

If `unenriched_ids` or `stale_ids` is non-empty, or `on_disk_all_time` counts
clearly exceed what the registry has indexed, heal before reporting:

```bash
"$PY" "$REPO/scripts/backfill.py"            # index anything the hooks missed
"$PY" "$REPO/scripts/enrich-sessions.py"     # journals new + stale (resumed) sessions
```

Re-run the check and tell the user what was healed (e.g. "journaled 6 codex
sessions that were never enriched"). If enrichment fails (provider down, quota),
DO NOT block: generate the report with `(unsummarized)` fallbacks and say so.

**Step 2 — data:** resolve the user's phrasing to a window
(`last-week`, `last-month`, `last-quarter`, `last-6-months`, `Nd`, or
`--from/--to` for anything custom — the script owns date math, don't compute
dates yourself), then:

```bash
"$PY" "$REPO/scripts/report-data.py" --window <window> > /tmp/worklog-report.json
```

**Step 3 — Markdown report:** read `references/report-style.md` for the
structure of the requested mode. If the file named by `[reports].style_override`
exists, read it too — its instructions WIN over the defaults (personal /
company framing). Write the report to `<out_dir>/summary-<from>-to-<to>.md`.

Non-negotiable: every non-trivial session in the window appears in its
project's section. Compression happens only in the narrative sections
(executive summary, highlights, themes) — never by dropping sessions.

**Step 4 — HTML timeline:**

```bash
cp "<this skill dir>/templates/timeline.html" "<out_dir>/timeline-<from>-to-<to>.html"
```

Then edit the copy: replace the `{}` inside
`<script id="worklog-data" type="application/json">{}</script>` with the
report JSON from Step 2, plus one added key you author:
`"narrative": {"summary": "<2-4 sentences>", "highlights": ["...", ...]}`.

**Step 5 —** report both file paths (and the healing note from Step 1). Do not
open the files.

## Mode: checkpoint (optional, mid/end-session)

Delegate to the repo's `snapshot` / `checkpoint` skills — not required for the
journal to function; enrichment covers every session automatically.
