---
name: snapshot
description: End-of-session capture — record the session's goal, key decisions, artifacts, and unresolved items into session_snapshots for durable recall.
---

# snapshot

Use at the end of a session to capture a structured snapshot:
goal · decisions · artifacts · unresolved. Stored in `session_snapshots` (one row
per session) and surfaced by the MCP `get_session_summary` tool and the UI.

Implementation: `scripts/snapshot.py --session <id>`. Reuses the enrichment
provider to derive the four fields, then writes the snapshot row.

Status: scaffold — table (`session_snapshots`) exists; the nightly enrichment pass
already extracts `key_decisions` into `session_artifacts`.
