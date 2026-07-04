"""Semantic search over session embeddings.

Default backend: numpy brute-force cosine over float32 BLOBs stored in
session_embeddings. No native SQLite extension required and trivially fast for up
to a few thousand sessions. The SentenceTransformer model is loaded lazily and
cached so the first query pays the load cost, not import time.
"""
from __future__ import annotations

import os
import struct
from functools import lru_cache

# Local-first: the model is cached after first download; don't hit the network.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np

import indexer
import sbconfig

DIM = 384


@lru_cache(maxsize=1)
def get_model():
    from sentence_transformers import SentenceTransformer
    try:
        return SentenceTransformer(sbconfig.EMBED_MODEL)
    except Exception:
        # Model not in the local cache. Fetching it silently would contradict
        # the no-external-requests promise, so going online is opt-in;
        # install.sh (non --lite) pre-downloads, making this branch rare.
        # Callers catch this and fall back to keyword search.
        if os.environ.get("SB_ALLOW_MODEL_DOWNLOAD") == "1":
            os.environ.pop("HF_HUB_OFFLINE", None)
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
            return SentenceTransformer(sbconfig.EMBED_MODEL)
        raise RuntimeError(
            f"embedding model '{sbconfig.EMBED_MODEL}' is not cached locally; "
            "set SB_ALLOW_MODEL_DOWNLOAD=1 to fetch it once (or re-run "
            "install.sh without --lite)")


def embed_text(text: str) -> np.ndarray:
    vec = get_model().encode([text or ""], normalize_embeddings=True)[0]
    return np.asarray(vec, dtype=np.float32)


def pack(vec: np.ndarray) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec.tolist())


def unpack(blob: bytes) -> np.ndarray:
    n = len(blob) // 4
    return np.asarray(struct.unpack(f"{n}f", blob), dtype=np.float32)


def search(query: str, limit: int = 20, conn=None) -> list[tuple[str, float]]:
    """Return [(session_id, similarity)] sorted high→low. Empty if no embeddings."""
    own = conn is None
    conn = conn or indexer.connect()
    try:
        rows = conn.execute(
            "SELECT e.session_id, e.embedding FROM session_embeddings e "
            "JOIN sessions s ON s.session_id = e.session_id WHERE s.archived = 0"
        ).fetchall()
    finally:
        if own:
            conn.close()
    if not rows:
        return []
    q = embed_text(query)                                  # normalized
    # Only rows embedded at the query's dimensionality are comparable. Stale
    # rows from a previous [embeddings].model would otherwise crash np.vstack;
    # embed-sessions.py re-embeds them on its next run.
    usable = [(r[0], r[1]) for r in rows if len(r[1]) // 4 == q.shape[0]]
    if len(usable) < len(rows):
        print(f"[semsearch] skipping {len(rows) - len(usable)} embeddings with a stale "
              f"dimension — run scripts/embed-sessions.py to refresh them")
    if not usable:
        return []
    ids = [sid for sid, _ in usable]
    mat = np.vstack([unpack(blob) for _, blob in usable])  # (N, dim), already normalized
    sims = mat @ q                                         # cosine since unit vectors
    order = np.argsort(-sims)[:limit]
    return [(ids[i], float(sims[i])) for i in order]
