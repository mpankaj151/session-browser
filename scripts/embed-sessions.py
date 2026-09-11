#!/usr/bin/env python3
"""Embed sessions for semantic search.

An **embedding** is a list of numbers (a vector — 384 of them with the default model)
that a small language model produces from a piece of text, arranged so that texts with a
similar *meaning* end up with similar numbers. Storing one per session is what lets the
search box find "the time I fixed the flaky checkout tests" even when the transcript
never used those words: the query is embedded too, and sessions are ranked by how closely
their vectors point the same way. See semsearch.py and docs/GLOSSARY.md ("Embedding",
"Semantic search / cosine similarity").

Encodes title || summary || first_message into a 384-dim unit vector and stores it
as a float32 BLOB in session_embeddings. Batched. Re-runnable (re-embeds when the
source text changes). No-op rows are skipped.

The model runs locally on this machine; no session text is sent anywhere. The only
network this script can touch is a one-time download of the model weights, and that is
opt-in — enabled here (see the env var below) because embedding is the batch job the user
explicitly asked for at install time, while interactive queries stay strictly offline.

Skipping is a first-class outcome, not a failure: on a `--lite` install
sentence-transformers is not present at all, and on a deliberately offline machine the
weights are not cached. Both exit 0 so the nightly refresh does not report a failed run.
Only an attempted download that actually fails exits 1. See _load_model().

Pipeline position: run by the nightly `refresh-all` after enrichment (so freshly written
summaries get embedded the same night). Reads the sessions table; writes only
session_embeddings. Nothing else in the tool depends on it — with no embeddings at all,
search silently falls back to keyword and full-text matching.
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
# Hugging Face is the library ecosystem the embedding model is loaded through; it is
# chatty by default. Four knobs, all set BEFORE the libraries are imported because each
# is read once at import time:
#   HF_HUB_DISABLE_PROGRESS_BARS  no download/load progress bars on a non-interactive run
#   TRANSFORMERS_VERBOSITY=error  only real errors, not informational chatter
#   TOKENIZERS_PARALLELISM=false  silences a fork-safety warning; also avoids thread
#                                 churn we would gain nothing from at this batch size
# setdefault, not assignment, so an operator who exports one of these keeps their value.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import warnings  # noqa: E402
# The library emits a FutureWarning about an upcoming API change on every load. Nothing
# here can act on it, and it is the last thing that would otherwise pollute the log.
warnings.filterwarnings("ignore", category=FutureWarning)

import indexer  # noqa: E402
import semsearch  # noqa: E402


def _source_text(row) -> str:
    """The text that represents a session to the embedding model.

    Title, then summary, then the user's first message, joined with two spaces — the
    three shortest things that say what a session was about. The whole transcript is
    deliberately NOT used: it is far longer than the model's input limit, and most of it
    is tool output that would blur the meaning rather than sharpen it.

    Order matters a little (the model weights earlier text slightly more) and stability
    matters a lot: the exact string is stored alongside the vector as `source_text`, and
    the next run re-embeds precisely those rows whose string changed. Returns "" when the
    row has none of the three, and such rows are skipped entirely.
    """
    parts = [row["title"] or "", row["summary"] or "", row["first_message"] or ""]
    return "  ".join(p for p in parts if p).strip()


def _load_model():
    """The SentenceTransformer, or None when embeddings are simply not part of
    this install. Two skips and one failure:
      - sentence-transformers not importable  -> --lite install: skip, exit 0
      - model not cached, downloads disallowed -> offline by choice: skip, exit 0
      - download attempted and failed          -> a real problem: exit 1
    refresh-all reports a nonzero step as a nightly FAILURE, so the two modes
    must never look like one.

    Tests drive all three branches directly (tests/test_smoke.py: a stubbed missing
    `sentence_transformers` module, a get_model() that raises "not cached", and the same
    with downloads allowed), so the three outcomes are pinned: None, None, SystemExit(1).
    """
    try:
        return semsearch.get_model()
    except ImportError as e:
        print(f"! semantic search not installed ({e.__class__.__name__}: {e}) — skipping embeddings")
        print("  (--lite install: search falls back to keyword + full-text; re-run install.sh without --lite to enable)")
        return None
    except Exception as e:  # noqa: BLE001
        # Model-loading failures arrive as long multi-line messages (stack hints, URLs);
        # the last line is the useful part, so only that is printed.
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
    """Embed every session whose source text (or vector size) is out of date.

    --batch N  how many texts to encode per call to the model; 64 keeps memory modest
               while still amortising the per-call overhead. Purely a tuning knob.
    --force    re-embed everything, ignoring the up-to-date check. Use after changing
               how _source_text() is built.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--force", action="store_true", help="re-embed all sessions")
    args = ap.parse_args()

    # Resolve the model BEFORE touching the DB: a skip must leave no trace.
    model = _load_model()
    if model is None:
        return
    conn = indexer.connect()
    # indexer.VISIBLE is the shared predicate for "a session the user should see" — live
    # rows plus those whose transcript aged out but which were real conversations.
    # Subagent-noise rows are excluded, so they never take up space in the index.
    rows = conn.execute(
        f"SELECT session_id, title, summary, first_message FROM sessions WHERE {indexer.VISIBLE}"
    ).fetchall()
    # What is already embedded, as session_id -> (source_text, dim). Read in one query
    # rather than per row: the comparison below is a dictionary lookup, not a round trip.
    existing = {
        r[0]: (r[1], r[2]) for r in conn.execute(
            "SELECT session_id, source_text, dim FROM session_embeddings"
        ).fetchall()
    }

    # A config [embeddings].model change means a new dimensionality — stored
    # rows at the old dim are unusable and must be re-embedded even if their
    # source_text is unchanged.
    # Vectors of different lengths cannot be compared at all, so semsearch.search() drops
    # stale-dimension rows from its results; including `dim` in the up-to-date test is
    # what makes that state self-healing on the next nightly run instead of permanent.
    model_dim = model.get_sentence_embedding_dimension()

    # Two reasons to (re-)embed a row: its text changed (a new title or summary arrived
    # from enrichment) or its stored vector is the wrong size. --force overrides both.
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
        # normalize_embeddings=True scales each vector to length 1, which is what lets
        # search() use a plain dot product as the cosine similarity.
        vecs = model.encode([t for _, t in chunk], normalize_embeddings=True)
        for (sid, text), vec in zip(chunk, vecs):
            import numpy as np
            arr = np.asarray(vec, dtype=np.float32)
            blob = semsearch.pack(arr)
            # Upsert: one row per session, replaced in place when it is re-embedded, so
            # a re-run never accumulates duplicates. source_text is stored with the
            # vector because it IS the freshness check the next run performs.
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
