"""Semantic search over session embeddings.

An **embedding** is a list of numbers — here 384 of them — that a small language model
produces from a piece of text, arranged so that texts with a similar *meaning* land close
together in that 384-dimensional space. Two sessions about flaky checkout tests get
similar number lists even if they share no words. Searching is then geometry: turn the
query into its own 384 numbers and rank stored sessions by how closely their direction
matches. That closeness measure is **cosine similarity** — the cosine of the angle
between the two vectors, 1.0 for identical direction, 0.0 for unrelated. See
docs/GLOSSARY.md ("Embedding", "Semantic search / cosine similarity").

This is what lets the UI's search box answer "the time I fixed the flaky checkout tests"
when the transcript never used those words. It complements, and never replaces, the exact
word index built by scripts/build-fts.py — when the model is unavailable the callers fall
back to keyword + full-text search.

Default backend: numpy brute-force cosine over float32 BLOBs stored in
session_embeddings. No native SQLite extension required and trivially fast for up
to a few thousand sessions. The SentenceTransformer model is loaded lazily and
cached so the first query pays the load cost, not import time.

Why brute force rather than a vector database or a SQLite vector extension: at a few
thousand sessions the whole matrix is a couple of megabytes, and one numpy matrix-vector
multiply over it takes single-digit milliseconds — far less than the network-free model
load already paid. A real vector index would add a dependency that pyenv's bundled
sqlite3 cannot even load (it is built without extension support), for no measurable win.
scripts/migrate-db.py still creates a native `sessions_vec` table when an
extension-capable interpreter plus the `sqlite_vec` package happen to be present, so an
installation that can use the fast path gets it for free; everything here works either
way. See docs/ARCHITECTURE.md, "Vector search without a native extension".

Storage shape: one row per session in session_embeddings — `embedding` is the raw bytes
of `dim` little-endian float32 values (packed by pack(), read back by unpack()), `dim` is
the TRUE vector length of whatever model produced it, and `source_text` is the exact text
that was embedded so scripts/embed-sessions.py can tell when a row is stale.

Who calls this: session-ui/app.py (the search box) and the MCP server's search tool,
both at query time; scripts/embed-sessions.py uses pack()/get_model() at write time.
"""
from __future__ import annotations

import os
import struct
from functools import lru_cache

import numpy as np

import indexer
import sbconfig

# Vector length of the default model (all-MiniLM-L6-v2). Informational only — nothing
# here assumes it: every stored row carries its own `dim`, and search() compares against
# the query vector's actual shape, so switching [embeddings].model in config.toml to a
# model of a different size cannot corrupt or crash a search.
DIM = 384


@lru_cache(maxsize=1)
def get_model():
    """Load (once per process) the local sentence-embedding model named by config.

    The model is a few tens of megabytes of weights that live in the Hugging Face cache
    under the user's home directory; loading it takes a second or two, so `lru_cache`
    keeps exactly one instance alive for the life of the process. Importing this module
    does NOT load it — the first query does, which keeps `import semsearch` cheap for
    callers that may never search (the Flask app imports it at start-up).

    Everything runs locally: the text never leaves the machine, and no request is made to
    any model vendor. The only network this function could ever touch is a one-time
    download of the weights themselves, and that is opt-in — see the branches below.

    Raises ImportError when `sentence-transformers` is not installed (a `--lite` install),
    and RuntimeError when the weights are not cached and downloading was not permitted.
    Callers are expected to catch both and fall back to keyword search;
    scripts/embed-sessions.py turns each into a clean skip rather than a failure.
    """
    from sentence_transformers import SentenceTransformer
    try:
        # Offline-first via the local_files_only KWARG, never via the
        # HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE env vars: huggingface_hub freezes
        # those into module constants at import time, so flipping the env after
        # this first attempt can't re-enable the network — the authorized
        # download below would be a guaranteed no-op on a cold machine.
        return SentenceTransformer(sbconfig.EMBED_MODEL, local_files_only=True)
    except Exception:
        # Model not in the local cache. Fetching it silently would contradict
        # the no-external-requests promise, so going online is opt-in;
        # install.sh (non --lite) pre-downloads, making this branch rare.
        # Callers catch this and fall back to keyword search.
        if os.environ.get("SB_ALLOW_MODEL_DOWNLOAD") == "1":
            return SentenceTransformer(sbconfig.EMBED_MODEL)
        raise RuntimeError(
            f"embedding model '{sbconfig.EMBED_MODEL}' is not cached locally; "
            "set SB_ALLOW_MODEL_DOWNLOAD=1 to fetch it once (or re-run "
            "install.sh without --lite)")


def embed_text(text: str) -> np.ndarray:
    """One piece of text as a unit-length float32 vector.

    `normalize_embeddings=True` scales every vector to length 1, which is what makes the
    plain dot product in search() equal to the cosine similarity — no division by
    magnitudes at query time. float32 (not the default float64) halves the stored size
    and matches what pack() writes. Loads the model on first call; see get_model() for
    what can go wrong there.
    """
    vec = get_model().encode([text or ""], normalize_embeddings=True)[0]
    return np.asarray(vec, dtype=np.float32)


def pack(vec: np.ndarray) -> bytes:
    """Vector -> raw bytes for the session_embeddings.embedding BLOB column.

    `"384f"` means "384 native-endian 4-byte floats", so a 384-value vector becomes
    exactly 1536 bytes with no JSON, no separators and no precision loss beyond float32.
    SQLite stores and returns those bytes verbatim. The length is taken from the vector,
    never hard-coded, so a different model's longer or shorter vector round-trips too.
    """
    return struct.pack(f"{len(vec)}f", *vec.tolist())


def unpack(blob: bytes) -> np.ndarray:
    """Inverse of pack(): BLOB bytes -> float32 vector.

    The vector length is recovered from the byte count (4 bytes per float), so unpack()
    needs no schema knowledge and a row written by an older, differently-sized model
    still reads back correctly — search() then filters those rows out by length.
    """
    n = len(blob) // 4
    return np.asarray(struct.unpack(f"{n}f", blob), dtype=np.float32)


def search(query: str, limit: int = 20, conn=None) -> list[tuple[str, float]]:
    """Return [(session_id, similarity)] sorted high→low. Empty if no embeddings.

    `similarity` is a cosine score in roughly -1.0 … 1.0; in practice this model's scores
    sit around 0.1-0.7 and only the ORDER is meaningful, so callers rank by it rather
    than thresholding on an absolute value.

    Pass `conn` to reuse an open registry connection (the Flask app and the tests do);
    with `conn=None` a connection is opened and closed here. The read is finished and the
    connection released BEFORE the model is loaded and the arithmetic runs, so a slow
    first query never holds a database handle open.

    Raises whatever get_model() raises when the embedding model is unavailable — callers
    catch that and fall back to keyword/full-text search.
    """
    own = conn is None
    conn = conn or indexer.connect()
    try:
        # Join back to sessions so hidden rows never surface: indexer.VISIBLE is the
        # shared predicate for "a session the user should see" (live, plus rows whose
        # transcript aged out but which still represent real conversations). Subagent
        # noise is excluded by the same predicate. Embeddings for such rows may exist —
        # the join, not a delete, is what keeps them out of results.
        rows = conn.execute(
            "SELECT e.session_id, e.embedding FROM session_embeddings e "
            f"JOIN sessions s ON s.session_id = e.session_id WHERE {indexer.VISIBLE}"
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
        import sys
        print(f"[semsearch] skipping {len(rows) - len(usable)} embeddings with a stale "
              f"dimension — run scripts/embed-sessions.py to refresh them", file=sys.stderr)
    if not usable:
        return []
    ids = [sid for sid, _ in usable]
    mat = np.vstack([unpack(blob) for _, blob in usable])  # (N, dim), already normalized
    # One matrix-vector product scores every session at once: row i of `mat` dotted with
    # the query gives session i's similarity. Because every vector has length 1, that dot
    # product IS the cosine of the angle between them — no normalisation step needed.
    sims = mat @ q                                         # cosine since unit vectors
    # argsort is ascending, so negate to get best-first, then keep the top `limit`.
    order = np.argsort(-sims)[:limit]
    return [(ids[i], float(sims[i])) for i in order]
