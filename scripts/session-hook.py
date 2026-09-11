#!/usr/bin/env python3
"""Claude Code Stop + SessionEnd hook — instant indexing, deferred heavy work.

Vocabulary (docs/GLOSSARY.md has the rest): a *session* is one conversation with a coding
CLI; its *transcript* is the file that CLI writes while the conversation happens (for
Claude Code, one JSONL — one JSON object per line — per session); a *hook* is a small
program the CLI runs at a fixed moment; *enrichment* is asking a model, through a
non-interactive CLI run, to read a finished session and write a structured summary.

Claude Code runs this script on two events. **Stop** fires every time an assistant reply
finishes, so several times per conversation. **SessionEnd** fires once, when the session
closes. Each run gets a single JSON object on standard input; a real payload looks like:

    {"session_id": "0f6c1a2b-6f0e-4a1e-9a3c-5f2b0c7d1e44",
     "transcript_path": "/Users/me/.claude/projects/-Users-me-app/0f6c1a2b-....jsonl",
     "cwd": "/Users/me/app",
     "hook_event_name": "SessionEnd",
     "reason": "clear"}

Only `transcript_path` and `hook_event_name` are read. The session id is deliberately NOT
taken from the payload: the claude adapter re-derives it from the transcript, and that is
also where it decides a file is a subagent sidechain and must never become a registry row
(see sources/claude.py). Where it fits in the pipeline: this is the fast lane of the
two-tier live indexing described in docs/ARCHITECTURE.md — the watcher daemon is the slow
lane that catches whatever a hook missed.

Two tiers so perceived latency stays in the tens of milliseconds:
  Inline (waited): read transcript_path from stdin JSON -> parse_header ->
                   indexer.upsert -> write .hook-state.json -> exit.
  Deferred (detached, not waited): spawn extract-reasoning.py to archive the raw
                   transcript and render the readable decision trail. On
                   SessionEnd ONLY, additionally spawn
                   `enrich-sessions.py --session <id>` so the session gets its
                   journal-grade summary the moment it ends. (Stop fires after
                   EVERY assistant response — enriching there would pay one LLM
                   call per turn; SessionEnd fires once, and the enricher's
                   staleness check makes a no-activity re-fire cost nothing.)

"Detached" here means `start_new_session=True` and no `wait()`: the child is put in its own
process group, so it survives this hook exiting AND Claude Code shutting down, and the
user's terminal never waits on a multi-second archive or an LLM call. A detached child has
no terminal to print to, so both spawns get an append-mode log file under <log dir>/.

The race-guard mark is `hookstate.mark()`, which writes "<session id> was handled at <now>"
into .hook-state.json. The watcher daemon sees the same transcript change moments later and
calls `hookstate.recently()`; within 30 seconds it skips the file, so the two tiers never
index the same session twice. Every hook in this repo uses the same file.

Registered in ~/.claude/settings.json under BOTH events (install.sh does this):
  { "hooks": { "Stop":       [ { "hooks": [ { "type": "command", "command": CMD } ] } ],
               "SessionEnd": [ { "hooks": [ { "type": "command", "command": CMD } ] } ] } }
  where CMD = "\"<venv-python>\" \"<repo>/scripts/session-hook.py\""
The command MUST use the absolute venv interpreter (launchd/hook PATH lacks pyenv).

CONTRACT: this process must ALWAYS exit 0. A nonzero Stop-hook exit blocks
Claude Code's session end, so even an unimportable config or a missing
dependency must degrade to a silent no-op. Hence all project imports happen
inside the guarded main(), and __main__ catches BaseException.

That is why the bottom of this file catches `BaseException` and not `Exception`: a plain
`except Exception` would let KeyboardInterrupt, SystemExit or a MemoryError escape and
become a nonzero exit — i.e. a user whose `git pull` half-installed a dependency would find
their coding CLI refusing to end sessions. Diagnostics go to stderr (Claude Code shows hook
stderr but does not act on it) and even that print is itself wrapped, so a closed stderr
cannot turn into an exit code either.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Claude Code invokes this file by absolute path from its own working directory, so the
# repo root is not importable by default — put it on sys.path before any project import.
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

# The two deferred workers, spawned as separate processes (never imported): keeping them
# out of this process is what lets the hook exit while they keep running.
EXTRACT = _REPO / "scripts" / "extract-reasoning.py"
ENRICH = _REPO / "scripts" / "enrich-sessions.py"

# Our own enrichment providers run `claude --print` with this set, so the hook
# their headless sessions trigger is a no-op (belt to the entrypoint=="sdk-cli"
# suspenders in the adapter).
# Without it the loop is: enrichment starts a Claude Code run -> that run ends -> Claude
# Code fires this hook -> the hook spawns enrichment again. Any value counts as "on".
_SUPPRESS_ENV = "SESSION_BROWSER_SUPPRESS_HOOK"

def _read_payload() -> dict:
    """Read the hook's JSON payload from standard input.

    Claude Code writes one JSON object and closes the pipe. Anything unreadable — empty
    input (someone ran the hook by hand), truncated JSON, a payload shape from a future
    release — returns an empty dict instead of raising: an exception here would end the
    process nonzero and block the CLI.
    """
    try:
        raw = sys.stdin.read()
        return json.loads(raw) if raw.strip() else {}
    except (json.JSONDecodeError, ValueError):
        return {}


def _transcript_path(payload: dict) -> Path | None:
    """Pick the transcript file this event is about, or None if the payload names none.

    Three key spellings are accepted because they have all appeared across Claude Code
    releases: `transcript_path` is the current one, the other two are older/defensive.
    Returning None (rather than guessing) means the caller quietly does nothing.
    """
    for key in ("transcript_path", "transcriptPath", "transcript"):
        if payload.get(key):
            return Path(payload[key]).expanduser()
    # fallback: positional arg
    # Lets the hook be exercised by hand while debugging, with no JSON on stdin:
    #   .venv/bin/python scripts/session-hook.py ~/.claude/projects/-Users-me-app/<id>.jsonl
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).expanduser()
    return None


def main() -> None:
    """Handle one hook invocation: index the session now, defer everything slow.

    Side effects: upserts one row in registry.db, writes the race-guard mark, and spawns up
    to two detached child processes (reasoning extraction always; enrichment on SessionEnd
    only). Returns early — never raises — on every "nothing to do" path, because the only
    hard requirement on this process is that it finish quickly and exit 0.
    """
    if os.environ.get(_SUPPRESS_ENV):
        return  # our own headless enrichment call — nothing to index

    payload = _read_payload()
    path = _transcript_path(payload)
    if path is None or not path.exists():
        return  # nothing to do; never block the CLI

    # Project imports INSIDE the guard: a broken config.toml or missing dep must
    # not take down the hook (see module docstring contract).
    import hookstate
    import indexer
    import sbconfig
    from sources.registry import _make_claude

    sbconfig.ensure_dirs()  # first run after install: create ~/.session-browser/{logs,…}

    # --- inline tier: cheap upsert ---
    # parse_header reads only what it needs (first/last lines, a bounded scan) rather than
    # the whole transcript, so this stays in the tens of milliseconds on a 40 MB file.
    # `_make_claude()` builds the Claude adapter honouring $CLAUDE_CONFIG_DIR; upsert()
    # merges with COALESCE so re-indexing never wipes a summary, cost or topic list.
    header = None
    try:
        header = _make_claude().parse_header(path)
        if header is not None:
            indexer.upsert(header)
            hookstate.mark(header.session_id)
    except Exception as e:  # noqa: BLE001 — a hook must never crash the session
        # Swallowed on purpose: a corrupt transcript, a locked database or a schema the
        # code has not migrated yet must cost the user a stderr line, not a stuck CLI.
        print(f"[session-hook] index error: {e}", file=sys.stderr)

    # --- deferred tier: detached archive + reasoning render (fire-and-forget) ---
    # header is None for headless/sdk-cli and unparseable transcripts — the same
    # sessions the index excludes, so don't archive/render them either.
    if header is None:
        return
    try:
        # Appended to, never truncated, and deliberately not closed: the child inherits the
        # file descriptor and keeps writing after this process is gone.
        log = open(sbconfig.LOG_DIR / "reasoning-hook.log", "a")
        # --archive also copies the transcript byte-for-byte into the raw vault. That copy
        # is what makes the session restorable later, when Claude Code's own 30-day
        # cleanup deletes the original. Safe per-Stop here (unlike the OpenCode hook)
        # because archive_raw only writes a new version when the content actually changed.
        subprocess.Popen(
            [sys.executable, str(EXTRACT), "--session", str(path), "--archive"],
            stdin=subprocess.DEVNULL, stdout=log, stderr=log,
            start_new_session=True,  # own process group: outlives the hook and the CLI
        )
    except Exception as e:  # noqa: BLE001
        # A failed spawn (no fork slots, unwritable log dir) is not worth a nonzero exit:
        # the nightly refresh-all re-runs the same extraction for every session.
        print(f"[session-hook] spawn error: {e}", file=sys.stderr)

    # --- SessionEnd only: journal-grade enrichment for the ended session ---
    # Detached like the reasoning tier; the enricher itself skips fresh sessions
    # (staleness check) and its headless LLM child inherits _SUPPRESS_ENV, so
    # this can neither block shutdown nor recurse.
    # `hook_event_name` is the only way to tell the two registrations apart — the same
    # command is registered for Stop and for SessionEnd.
    if payload.get("hook_event_name") == "SessionEnd":
        try:
            elog = open(sbconfig.LOG_DIR / "enrich-hook.log", "a")
            # Enrichment runs a model through a non-interactive CLI run, which can take
            # tens of seconds and spends the user's own CLI quota — hence detached, and
            # hence only here rather than on every Stop.
            subprocess.Popen(
                [sys.executable, str(ENRICH), "--session", header.session_id],
                stdin=subprocess.DEVNULL, stdout=elog, stderr=elog,
                start_new_session=True,
            )
        except Exception as e:  # noqa: BLE001
            # Same reasoning as the extraction spawn: the nightly `refresh-all --enrich`
            # will pick this session up anyway.
            print(f"[session-hook] enrich spawn error: {e}", file=sys.stderr)


if __name__ == "__main__":
    # The exit-0 contract, enforced in one place. BaseException (not Exception) so that
    # SystemExit/KeyboardInterrupt/MemoryError cannot leak a nonzero status either.
    try:
        main()
    except BaseException as e:  # noqa: BLE001 — contract: always exit 0
        try:
            print(f"[session-hook] fatal (suppressed): {e}", file=sys.stderr)
        except Exception:  # noqa: BLE001
            pass  # even reporting the failure must not fail (e.g. stderr already closed)
    sys.exit(0)
