You are writing the work-journal entry for one AI coding CLI session — the
record a senior engineer will rely on for stand-ups, 1:1s, and performance
reviews. Capture not just what happened but what it meant: decisions and their
rationale, approaches tried and discarded, and what remains open.

- CLI: {cli_source}
- Model: {model}
- Working directory: {cwd}
{prior_context}
Transcript (truncated):
---
{transcript}
---

Analyze the session and respond with **ONLY a JSON object** (no prose, no code
fences) with exactly these keys:

- "brief_summary": 2-4 complete sentences describing what the session
  accomplished and why it mattered. Written in past tense, self-contained
  (understandable months later without the transcript).
- "goal": ONE short sentence — what the session set out to do.
- "accomplishments": array of short past-tense strings, one per concrete thing
  achieved (may be empty). Specific over generic: "Added retry with backoff to
  the S3 uploader" beats "improved reliability".
- "key_decisions": array of short strings, each "decision — rationale" (may be
  empty).
- "explorations": array of short strings — approaches considered or tried and
  NOT kept, each with why it was set aside (may be empty).
- "open_threads": array of short strings — unresolved items, follow-ups, or
  known next steps (may be empty).
- "reusability": ONE short sentence on anything produced that is reusable
  beyond this session (pattern, script, learning), or "" if nothing.
- "goal_categories": object mapping topic -> count, e.g. {"python": 3, "testing": 2}.
- "session_type": one of "debugging", "feature", "refactor", "review", "research",
  "planning", "ops", "other".
- "outcome": one of "completed", "partial", "abandoned", "unknown".
- "files_touched": array of file paths (may be empty).

Respond with the JSON object only.
