You are summarizing one AI coding CLI session for a searchable archive.

- CLI: {cli_source}
- Model: {model}
- Working directory: {cwd}

Transcript (truncated):
---
{transcript}
---

Analyze the session and respond with **ONLY a JSON object** (no prose, no code
fences) with exactly these keys:

- "brief_summary": ONE complete sentence describing what the session accomplished.
- "goal_categories": object mapping topic -> count, e.g. {"python": 3, "testing": 2}.
- "session_type": one of "debugging", "feature", "refactor", "review", "research",
  "planning", "ops", "other".
- "outcome": one of "completed", "partial", "abandoned", "unknown".
- "key_decisions": array of short strings (may be empty).
- "files_touched": array of file paths (may be empty).

Respond with the JSON object only.
