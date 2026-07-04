---
name: checkpoint
description: Mid-session compaction — summarize the turns so far into session_checkpoints so a long session can be resumed with a compact primer instead of the full transcript.
---

# checkpoint

Use mid-session when context grows large. Summarize turns `1..(N-10)` into a
checkpoint row; on resume, load the checkpoint summary + the last 10 turns instead
of the entire transcript.

Implementation: `scripts/checkpoint.py --session <id>` (writes to
`session_checkpoints`). The Flask `/api/sessions/<id>/context` endpoint prefers the
latest checkpoint when present.

Status: scaffold — table (`session_checkpoints`) and resume wiring exist; the
summarizer reuses the enrichment provider.
