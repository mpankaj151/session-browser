#!/usr/bin/env python3
"""Embed sessions for semantic search.

Encodes title || summary || first_message into a 384-dim unit vector and stores it
as a float32 BLOB in session_embeddings. Batched. Re-runnable (re-embeds when the
source text changes). No-op rows are skipped.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# This is the explicit batch job for the semantic stack the user opted into at
# install — fetching a missing model here is expected. Interactive queries
# (UI / MCP) stay strictly offline.
os.environ.setdefault("SB_ALLOW_MODEL_DOWNLOAD", "1")
# Loading a cached model printed a tqdm "Loading weights" bar and a
# FutureWarning to stderr on EVERY nightly run, so refresh.err.log was never
# empty and stopped being a health signal.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import warnings  # noqa: E402
warnings.filterwarnings("ignore", category=FutureWarning)

import indexer  # noqa: E402
import semsearch  # noqa: E402


def _source_text(row) -> str:
    parts = [row["title"] or "", row["summary"] or "", row["first_message"] or ""]
    return "  ".join(p for p in parts if p).strip()


def _load_model():
    """The SentenceTransformer, or None when embeddings are simply not part of
    this install. Two skips and one failure:
      - sentence-transformers not importable  -> --lite install: skip, exit 0
      - model not cached, downloads disallowed -> offline by choice: skip, exit 0
      - download attempted and failed          -> a real problem: exit 1
    refresh-all reports a nonzero step as a nightly FAILURE, so the two modes
    must never look like one."""
    try:
        return semsearch.get_model()
    except ImportError as e:
        print(f"! semantic search not installed ({e.__class__.__name__}: {e}) — skipping embeddings")
        print("  (--lite install: search falls back to keyword + full-text; re-run install.sh without --lite to enable)")
        return None
    except Exception as e:  # noqa: BLE001
        msg = str(e).strip().splitlines()
        tail = msg[-1] if msg else str(e)
        if os.environ.get("SB_ALLOW_MODEL_DOWNLOAD") != "1":
            print(f"! embedding model not cached and downloads are disallowed — skipping embeddings ({tail})")
            return None
        print(f"! embedding model unavailable: {type(e).__name__}: {tail}")
        print("  Semantic search falls back to keyword + full-text until the model is cached.")
        print("  Check access to huggingface.co (proxy/firewall?), then re-run: sb refresh")
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--force", action="store_true", help="re-embed all sessions")
    args = ap.parse_args()

    # Resolve the model BEFORE touching the DB: a skip must leave no trace.
    model = _load_model()
    if model is None:
        return
    conn = indexer.connect()
    rows = conn.execute(
        f"SELECT session_id, title, summary, first_message FROM sessions WHERE {indexer.VISIBLE}"
    ).fetchall()
    existing = {
        r[0]: (r[1], r[2]) for r in conn.execute(
            "SELECT session_id, source_text, dim FROM session_embeddings"
        ).fetchall()
    }

    # A config [embeddings].model change means a new dimensionality — stored
    # rows at the old dim are unusable and must be re-embedded even if their
    # source_text is unchanged.
    model_dim = model.get_sentence_embedding_dimension()

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
