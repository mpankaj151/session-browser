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
# Each CLI can relocate its state; follow the same env vars the CLIs read.
CLAUDE_DIR="${CLAUDE_CONFIG_DIR:-$HOME/.claude}"
CODEX_DIR="${CODEX_HOME:-$HOME/.codex}"

# Codex rollouts may be zstd-compressed in place (cold, >~7 days) and/or moved
# to the sibling archived_sessions tree by `codex archive`; all still resume.
codex_rollout() {
  find "$CODEX_DIR/sessions" "$CODEX_DIR/archived_sessions" \
       \( -name "rollout-*$1.jsonl" -o -name "rollout-*$1.jsonl.zst" \) 2>/dev/null | head -1
}

# Fail BEFORE any side effect when the CLI itself is missing: exec'ing an absent
# binary died with a bare 127 after the session memory had already been linked.
need_cli() {
  command -v "$1" >/dev/null 2>&1 && return 0
  echo "cr: $1 is not on PATH on this machine — session $SID exists but cannot be resumed here" >&2
  echo "cr: install $1 (or open a shell where it is available) and re-run: cr $SID" >&2
  exit 1
}

# Session ids are used in find -name patterns; refuse glob/path metacharacters.
case "$SID" in
  *[\*\?\[\]/]*) echo "cr: invalid session id '$SID'" >&2; exit 1;;
esac

# Claude Code encodes a project path by replacing every non-alphanumeric
# character with '-' (spaces, dots, underscores, slashes — all of them).
encode_path() { printf '%s' "$1" | sed 's/[^A-Za-z0-9]/-/g'; }

# Auto-detect which CLI owns this session id.
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
    need_cli copilot
    if [ -f "$WS" ]; then
      cp "$WS" "$WS.bak"
      if grep -q '^cwd:' "$WS"; then
        # awk via ENVIRON, not sed: & and | in the directory path are literal here,
        # where a sed replacement would corrupt workspace.yaml
        CUR="$CUR" awk 'BEGIN{cur=ENVIRON["CUR"]} /^cwd:/{print "cwd: " cur; next} {print}' \
          "$WS" > "$WS.tmp" && mv "$WS.tmp" "$WS"
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
  *)
    echo "cr: unknown cli_source '$CLI'" >&2; exit 1
    ;;
esac
