#!/usr/bin/env python3
"""Embed sessions for semantic search.

Encodes title || summary || first_message into a 384-dim unit vector and stores it
as a float32 BLOB in session_embeddings. Batched. Re-runnable (re-embeds when the
source text changes). No-op rows are skipped.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import indexer  # noqa: E402
import semsearch  # noqa: E402


def _source_text(row) -> str:
    parts = [row["title"] or "", row["summary"] or "", row["first_message"] or ""]
    return "  ".join(p for p in parts if p).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--force", action="store_true", help="re-embed all sessions")
    args = ap.parse_args()

    conn = indexer.connect()
    rows = conn.execute(
        "SELECT session_id, title, summary, first_message FROM sessions WHERE archived = 0"
    ).fetchall()
    existing = {
        r[0]: (r[1], r[2]) for r in conn.execute(
            "SELECT session_id, source_text, dim FROM session_embeddings"
        ).fetchall()
    }

    # A config [embeddings].model change means a new dimensionality — stored
    # rows at the old dim are unusable and must be re-embedded even if their
    # source_text is unchanged.
    model_dim = semsearch.get_model().get_sentence_embedding_dimension()

    todo = []
    for r in rows:
        text = _source_text(r)
        if not text:
            continue
        prev_text, prev_dim = existing.get(r["session_id"], (None, None))
        if not args.force and prev_text == text and prev_dim == model_dim:
            continue
        todo.append((r["session_id"], text))

    if not todo:
        print("Embeddings up to date.")
        return

    model = semsearch.get_model()
    t0 = time.time()
    for i in range(0, len(todo), args.batch):
        chunk = todo[i:i + args.batch]
        vecs = model.encode([t for _, t in chunk], normalize_embeddings=True)
        for (sid, text), vec in zip(chunk, vecs):
            import numpy as np
            arr = np.asarray(vec, dtype=np.float32)
            blob = semsearch.pack(arr)
            conn.execute(
                "INSERT INTO session_embeddings (session_id, dim, embedding, source_text) "
                "VALUES (?,?,?,?) ON CONFLICT(session_id) DO UPDATE SET "
                "dim=excluded.dim, embedding=excluded.embedding, source_text=excluded.source_text, "
                "updated_at=CURRENT_TIMESTAMP",
                (sid, len(arr), blob, text),  # store the TRUE dim, never a constant
            )
        conn.commit()
        print(f"  embedded {min(i+args.batch, len(todo))}/{len(todo)}")
    conn.close()
    print(f"Embedded {len(todo)} sessions in {time.time()-t0:.1f}s.")


if __name__ == "__main__":
    main()
