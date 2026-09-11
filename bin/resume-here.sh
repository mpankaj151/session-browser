#!/usr/bin/env bash
# resume-here.sh — resume a past session in the CURRENT directory by relocating
# its memory (transcript / workspace) into this directory's project namespace.
#
#   Usage: resume-here.sh <session_id> [cli_source]
#
# cli_source is auto-detected when omitted. Typically invoked via the `cr` shell
# function (see install: bin/install-cr.sh):  cr <session_id>
#
# Unlike a plain `claude --resume <id>` (which only finds the session if you're
# standing in its original project dir), this ports the session's memory to wherever
# you run it, so you can continue the conversation in a new project/worktree.
#
# Claude:  copies ~/.claude/projects/<orig>/<id>.jsonl into the encoded project dir
#          for $PWD, then `claude --resume <id>`.
# Copilot: repoints the session's workspace.yaml cwd to $PWD, then `copilot --resume`.
#
# The original copy is left intact (this forks the session into the new location).
#
# WHAT "RESUME" MEANS
#   Reopening a past conversation in the CLI that recorded it, so the model still has the
#   whole history. Each CLI has its own command for that; this script's job is to pick the
#   right one for a session id and to make the session findable from the current
#   directory. See docs/GLOSSARY.md (session, transcript, resume).
#
# WHERE IT SITS
#   Called almost always through the one-line `cr` shell function that bin/install-cr.sh
#   writes into your rc. The UI's session list shows the id; `cr <id>` is the shortcut.
#   It reads the CLIs' own state directories directly and never consults the registry
#   database — so it works even when the indexer has never seen the session.
#
# WHAT IT WRITES
#   Claude:  a symlink (or, if symlinks are unavailable, a copy) of the transcript into
#            the encoded project directory for $PWD. The original is never modified.
#   Copilot: rewrites the `cwd:` line of that session's workspace.yaml, keeping a .bak.
#   Codex / OpenCode: nothing at all — both look sessions up globally by id.
#   Then it `exec`s the CLI, replacing this process, so the user lands straight in it.
#
# EXIT CODES
#   1 with a one-line explanation on stderr for: an unknown/ambiguous session id, a
#   dangling transcript link, or a CLI binary that is not on PATH. Otherwise it never
#   returns — exec hands the terminal to the CLI, whose exit status becomes ours.
set -euo pipefail

# ${1:?msg} is the shell's "required argument" form: with no id, print msg and exit.
SID="${1:?usage: resume-here.sh <session_id> [cli_source]}"
# Second argument forces a CLI; "auto" (the normal case) detects it from the id below.
CLI="${2:-auto}"
# The directory the user is standing in — the whole point of "resume HERE".
CUR="$(pwd)"
# Each CLI can relocate its state; follow the same env vars the CLIs read.
CLAUDE_DIR="${CLAUDE_CONFIG_DIR:-$HOME/.claude}"
CODEX_DIR="${CODEX_HOME:-$HOME/.codex}"

# codex_rollout — print the path of the Codex transcript for a session id, or nothing.
#   $1  the session id.
# Codex calls a transcript file a "rollout" and names it
# `rollout-<19-char timestamp>-<id>.jsonl`, e.g.
# `rollout-2026-08-01T10-00-00-019e18fa-0d21-7461-922c-….jsonl`, filed under
# ~/.codex/sessions/YYYY/MM/DD/. Two things move it afterwards, and both still resume:
# Codex zstd-compresses rollouts older than about a week in place (`.jsonl.zst`), and
# `codex archive` moves them into the sibling archived_sessions tree. Hence both roots and
# both suffixes below. `head -1` keeps only the first hit; stderr is discarded because a
# missing root directory is normal on a machine that never ran Codex.
codex_rollout() {
  find "$CODEX_DIR/sessions" "$CODEX_DIR/archived_sessions" \
       \( -name "rollout-*$1.jsonl" -o -name "rollout-*$1.jsonl.zst" \) 2>/dev/null | head -1
}

# need_cli — refuse clearly when the CLI binary is not installed in this shell.
#   $1  the binary name (claude / copilot / codex / opencode).
# Returns 0 if it is on PATH; otherwise prints two lines on stderr (what is missing, and
# how to retry) and exits 1. Indexing, search and the Archived/Restore flow never need a
# CLI binary — only resume and bridge do — so "the session exists but cannot be resumed
# here" is a genuinely common, non-broken state on a second laptop.
# Fail BEFORE any side effect when the CLI itself is missing: exec'ing an absent
# binary died with a bare 127 after the session memory had already been linked.
need_cli() {
  command -v "$1" >/dev/null 2>&1 && return 0
  echo "cr: $1 is not on PATH on this machine — session $SID exists but cannot be resumed here" >&2
  echo "cr: install $1 (or open a shell where it is available) and re-run: cr $SID" >&2
  exit 1
}

# Session ids are used in find -name patterns; refuse glob/path metacharacters.
# The pattern rejects any id containing * ? [ ] or / — a real id is a UUID, a `ses_`
# string or a Codex id, none of which need them. Without this, `cr '*'` would match every
# transcript on disk and link an arbitrary one into the current directory.
case "$SID" in
  *[\*\?\[\]/]*) echo "cr: invalid session id '$SID'" >&2; exit 1;;
esac

# encode_path — turn a directory path into Claude Code's project-directory name.
#   $1  an absolute path, e.g. /Users/me/My App
#   ->  the encoded name, e.g. -Users-me-My-App
# Claude Code files each session under ~/.claude/projects/<encoded cwd>/<id>.jsonl, and
# encodes the path by replacing every non-alphanumeric character with '-' (spaces, dots,
# underscores, slashes — all of them). This must match Claude Code's own rule exactly, or
# the relocated transcript lands in a directory the CLI never looks in.
encode_path() { printf '%s' "$1" | sed 's/[^A-Za-z0-9]/-/g'; }

# Auto-detect which CLI owns this session id, cheapest and most certain test first:
#   1. `ses_` prefix           -> OpenCode. Only OpenCode prefixes its ids, so this is
#                                 decidable from the string alone, with no disk access.
#   2. <id>.jsonl exists under the Claude projects tree  -> Claude Code. -maxdepth 2
#                                 because the layout is exactly projects/<encoded>/<id>.
#                                 `grep -q .` turns "find printed something" into a status.
#   3. a ~/.copilot/session-state/<id> DIRECTORY         -> Copilot, which keeps a folder
#                                 per session rather than a single file.
#   4. a matching rollout file -> Codex (see codex_rollout above).
# If none matches, the id is not on this machine at all — say so and stop, rather than
# guessing a CLI and letting it fail with its own obscure message.
if [ "$CLI" = "auto" ]; then
  if [[ "$SID" == ses_* ]]; then
    CLI=opencode          # OpenCode ids are prefixed; nothing else looks like this
  elif find "$CLAUDE_DIR/projects" -maxdepth 2 -name "$SID.jsonl" 2>/dev/null | grep -q .; then
    CLI=claude
  elif [ -d "$HOME/.copilot/session-state/$SID" ]; then
    CLI=copilot
  elif [ -n "$(codex_rollout "$SID")" ]; then
    CLI=codex
  else
    echo "cr: session '$SID' not found for claude, copilot, codex, or opencode" >&2
    exit 1
  fi
fi

# One branch per CLI. Each follows the same order: locate the session -> need_cli ->
# do whatever relocation that CLI needs -> exec it. need_cli always comes BEFORE the
# first write, so a missing binary leaves the machine exactly as it was.
case "$CLI" in
  claude)
    PROJECTS="$CLAUDE_DIR/projects"
    SRC="$(find "$PROJECTS" -maxdepth 2 -name "$SID.jsonl" 2>/dev/null | head -1)"
    if [ -z "$SRC" ]; then
      echo "cr: claude session $SID not found under $PROJECTS" >&2; exit 1
    fi
    need_cli claude
    # Resolve to the REAL file (in case SRC is itself a symlink from a prior cr),
    # so every location links back to one canonical transcript — always in sync.
    SRC_REAL="$(realpath "$SRC" 2>/dev/null || echo "$SRC")"
    if [ ! -e "$SRC_REAL" ]; then
      echo "cr: found only a dangling link for $SID — the canonical transcript was deleted" >&2
      echo "cr: ($SRC -> $SRC_REAL)" >&2
      exit 1
    fi
    # Where Claude Code will look for this session when started in $PWD.
    DEST_DIR="$PROJECTS/$(encode_path "$CUR")"
    DEST="$DEST_DIR/$SID.jsonl"
    # Skip when the transcript is already reachable from here: either this IS its original
    # directory, or a previous `cr` already linked it. Never overwrite an existing file.
    if [ "$SRC_REAL" != "$DEST" ] && [ ! -e "$DEST" ]; then
      mkdir -p "$DEST_DIR"
      # A symlink is strongly preferred: both locations then read and write ONE file, so
      # continuing the conversation here also updates it there. Copying is the fallback
      # for filesystems without symlinks, and the message warns that the two will diverge.
      if ln -s "$SRC_REAL" "$DEST" 2>/dev/null; then
        echo "cr: linked session memory (stays in sync) -> $DEST"
      else
        cp "$SRC_REAL" "$DEST"
        echo "cr: copied session memory (symlinks unavailable; will diverge) -> $DEST"
      fi
    fi
    exec claude --resume "$SID"
    ;;
  copilot)
    # Copilot keeps a directory per session and records the working directory inside it,
    # so "resume here" means repointing that one `cwd:` line rather than moving files.
    STATE="$HOME/.copilot/session-state/$SID"
    WS="$STATE/workspace.yaml"
    if [ ! -d "$STATE" ]; then
      echo "cr: copilot session $SID not found under ~/.copilot/session-state" >&2; exit 1
    fi
    need_cli copilot
    if [ -f "$WS" ]; then
      cp "$WS" "$WS.bak"
      if grep -q '^cwd:' "$WS"; then
        # awk via ENVIRON, not sed: & and | in the directory path are literal here,
        # where a sed replacement would corrupt workspace.yaml
        CUR="$CUR" awk 'BEGIN{cur=ENVIRON["CUR"]} /^cwd:/{print "cwd: " cur; next} {print}' \
          "$WS" > "$WS.tmp" && mv "$WS.tmp" "$WS"
      # No cwd: line yet (an older Copilot, or a session that never had one) — append it.
      else
        printf 'cwd: %s\n' "$CUR" >> "$WS"
      fi
      echo "cr: pointed copilot session cwd -> $CUR (backup: workspace.yaml.bak)"
    fi
    exec copilot --resume="$SID"
    ;;
  opencode)
    # `opencode --session` is a global lookup by id that runs in the CURRENT
    # directory — exactly what "resume here" means. No memory to port.
    need_cli opencode
    exec opencode --session "$SID"
    ;;
  codex)
    # Codex stores sessions by date, not by an encoded cwd, so `codex resume`
    # finds the session from any directory — just resume in place.
    if [ -z "$(codex_rollout "$SID")" ]; then
      echo "cr: codex session $SID not found under $CODEX_DIR/{sessions,archived_sessions}" >&2; exit 1
    fi
    need_cli codex
    exec codex resume "$SID"
    ;;
  # Only reachable when the caller passed an explicit second argument that is not one of
  # the four known CLIs — auto-detection can only ever produce a valid name.
  *)
    echo "cr: unknown cli_source '$CLI'" >&2; exit 1
    ;;
esac
