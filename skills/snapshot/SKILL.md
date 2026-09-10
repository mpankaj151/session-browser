---
name: snapshot
description: NOT IMPLEMENTED (scaffold, do not use) — planned end-of-session capture into session_snapshots; the nightly enrichment pass already extracts decisions.
---

# snapshot

Use at the end of a session to capture a structured snapshot:
goal · decisions · artifacts · unresolved. Stored in `session_snapshots` (one row
per session) and surfaced by the MCP `get_session_summary` tool and the UI.

Status: scaffold — not implemented. Only the `session_snapshots` table exists
(scripts/migrate-db.py); nothing writes or reads it yet. The nightly enrichment
pass already extracts `key_decisions` into `session_artifacts` and the journal.
