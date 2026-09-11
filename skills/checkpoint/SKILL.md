---
name: checkpoint
description: NOT IMPLEMENTED (scaffold, do not use) — planned mid-session compaction into session_checkpoints; the nightly enrichment pass already journals every session.
---

# checkpoint

Use mid-session when context grows large. Summarize turns `1..(N-10)` into a
checkpoint row; on resume, load the checkpoint summary + the last 10 turns instead
of the entire transcript.

Status: scaffold — not implemented. Only the `session_checkpoints` table exists
(scripts/migrate-db.py); no script writes it and the context endpoint does not
read it yet. The nightly enrichment pass (`sb refresh --enrich`) covers the
journal use case today.
