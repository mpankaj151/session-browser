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
set -euo pipefail

SID="${1:?usage: resume-here.sh <session_id> [cli_source]}"
CLI="${2:-auto}"
CUR="$(pwd)"

# Session ids are used in find -name patterns; refuse glob/path metacharacters.
case "$SID" in
  *[\*\?\[\]/]*) echo "cr: invalid session id '$SID'" >&2; exit 1;;
esac

# Claude Code encodes a project path by replacing every non-alphanumeric
# character with '-' (spaces, dots, underscores, slashes — all of them).
encode_path() { printf '%s' "$1" | sed 's/[^A-Za-z0-9]/-/g'; }

# Auto-detect which CLI owns this session id.
if [ "$CLI" = "auto" ]; then
  if find "$HOME/.claude/projects" -maxdepth 2 -name "$SID.jsonl" 2>/dev/null | grep -q .; then
    CLI=claude
  elif [ -d "$HOME/.copilot/session-state/$SID" ]; then
    CLI=copilot
  else
    echo "cr: session '$SID' not found for claude or copilot" >&2
    exit 1
  fi
fi

case "$CLI" in
  claude)
    PROJECTS="$HOME/.claude/projects"
    SRC="$(find "$PROJECTS" -maxdepth 2 -name "$SID.jsonl" 2>/dev/null | head -1)"
    if [ -z "$SRC" ]; then
      echo "cr: claude session $SID not found under $PROJECTS" >&2; exit 1
    fi
    # Resolve to the REAL file (in case SRC is itself a symlink from a prior cr),
    # so every location links back to one canonical transcript — always in sync.
    SRC_REAL="$(realpath "$SRC" 2>/dev/null || echo "$SRC")"
    if [ ! -e "$SRC_REAL" ]; then
      echo "cr: found only a dangling link for $SID — the canonical transcript was deleted" >&2
      echo "cr: ($SRC -> $SRC_REAL)" >&2
      exit 1
    fi
    DEST_DIR="$PROJECTS/$(encode_path "$CUR")"
    DEST="$DEST_DIR/$SID.jsonl"
    if [ "$SRC_REAL" != "$DEST" ] && [ ! -e "$DEST" ]; then
      mkdir -p "$DEST_DIR"
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
    STATE="$HOME/.copilot/session-state/$SID"
    WS="$STATE/workspace.yaml"
    if [ ! -d "$STATE" ]; then
      echo "cr: copilot session $SID not found under ~/.copilot/session-state" >&2; exit 1
    fi
    if [ -f "$WS" ]; then
      cp "$WS" "$WS.bak"
      if grep -q '^cwd:' "$WS"; then
        # -i.tmp works on both BSD (macOS) and GNU sed; .bak above is the real backup
        sed -i.tmp "s|^cwd:.*|cwd: $CUR|" "$WS" && rm -f "$WS.tmp"
      else
        printf 'cwd: %s\n' "$CUR" >> "$WS"
      fi
      echo "cr: pointed copilot session cwd -> $CUR (backup: workspace.yaml.bak)"
    fi
    exec copilot --resume="$SID"
    ;;
  *)
    echo "cr: unknown cli_source '$CLI'" >&2; exit 1
    ;;
esac
