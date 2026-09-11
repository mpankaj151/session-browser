#!/usr/bin/env bash
# Reverse install.sh. Leaves registry.db and the reasoning archive intact unless
# you pass --purge.
#
#   ./uninstall.sh            unhook everything, keep the data
#   ./uninstall.sh --purge    also delete ~/.session-browser (the registry database)
#
# WHEN IT RUNS
#   By hand, from the repo. It is a cleanup script, so it is written to keep going after
#   any individual failure and to be safe to run twice (or on a half-installed machine).
#
# WHAT IT REMOVES — everything install.sh wrote OUTSIDE this repo directory:
#   the launchd agents (macOS) or systemd --user units (Linux);
#   the Stop + SessionEnd entries pointing at scripts/session-hook.py in Claude Code's
#     settings.json ($CLAUDE_CONFIG_DIR honoured) — other people's hooks are left alone;
#   the skill symlinks under <claude dir>/skills that point INTO this repo;
#   the OpenCode plugin file;
#   the marker-delimited `cr` and `sb` blocks in ~/.zshrc and ~/.bashrc (a .sb-uninstall-
#     backup copy of each rc is kept);
#   with --purge, ~/.session-browser (registry, logs, facets, OpenCode mirror).
#
# WHAT IT NEVER REMOVES
#   ~/claude-reasoning-archive — the raw transcript vault and the readable reasoning
#   trails. That is the one place a session survives after its CLI deleted the original,
#   so destroying it can lose history no reinstall can recover.
#   The repo itself, including .venv and config.toml: delete the directory if you want it
#   gone. And no transcript belonging to any CLI is ever touched.
#
# NOTE `set -uo pipefail` WITHOUT -e, unlike install.sh: a cleanup must not abort halfway
# because one artefact was already gone. See the exit 0 at the very bottom.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Only the exact flag `--purge` enables data deletion; anything else is ignored, so a
# mistyped flag can never delete the database by accident.
PURGE=0; [ "${1:-}" = "--purge" ] && PURGE=1
# Must resolve the same way install.sh did, or the hooks and skill links are looked for
# in the wrong directory and silently "already removed".
CLAUDE_DIR="${CLAUDE_CONFIG_DIR:-$HOME/.claude}"

# Background jobs first: stop the watcher before pulling the rest out from under it.
# Mirror of install.sh step 7 — launchd on macOS, systemd --user elsewhere. Unlike the
# installer this does not test for a reachable user bus: `systemctl --user disable` on a
# machine without one just fails, and every command here tolerates failure.
if [ "$(uname)" = "Darwin" ]; then
  echo "==> removing launchd jobs"
  AGENTS="$HOME/Library/LaunchAgents"
  # unload stops the running job; rm deletes the plist so it never starts again. The `;`
  # (not `&&`) before rm is deliberate: the file must go even if the unload failed.
  for job in watcher refresh; do
    P="$AGENTS/com.sessionbrowser.$job.plist"
    [ -f "$P" ] && launchctl unload "$P" 2>/dev/null; rm -f "$P"
  done
elif command -v systemctl >/dev/null 2>&1; then
  echo "==> removing systemd --user units"
  UNITS="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
  systemctl --user disable --now session-browser-watcher.service session-browser-refresh.timer 2>/dev/null || true
  rm -f "$UNITS/session-browser-watcher.service" "$UNITS/session-browser-refresh.service" "$UNITS/session-browser-refresh.timer"
  systemctl --user daemon-reload 2>/dev/null || true
fi

echo "==> removing Stop + SessionEnd hooks"
# The repo venv may already be gone — any python3 can strip the hooks.
# (The block below imports only the standard library for exactly this reason, so even a
# stock macOS python3.9 can run it.)
UNPY="$REPO/.venv/bin/python"
[ -x "$UNPY" ] || UNPY="$(command -v python3 || true)"
if [ -n "$UNPY" ]; then
  "$UNPY" - <<'PYEOF' || echo "   ! could not edit ~/.claude/settings.json — remove the session-hook.py hooks manually"
# Surgical edit of someone else's config file: keep every hook that is not ours, and
# drop the event key entirely when ours was the only entry, so uninstalling leaves no
# empty "Stop": [] behind. json.dumps(h) is a depth-agnostic "does this entry mention
# our script" test. No parse failure is caught here — a malformed settings.json raises,
# and the `|| echo` on the shell side turns that into the "remove it manually" hint.
import json, os
from pathlib import Path
s = Path(os.path.expanduser(os.environ.get("CLAUDE_CONFIG_DIR") or "~/.claude"))/"settings.json"
if s.exists():
    cfg = json.loads(s.read_text())
    hooks = cfg.get("hooks", {})
    for event in ("Stop", "SessionEnd"):
        entries = hooks.get(event, [])
        if isinstance(entries, list):
            entries = [h for h in entries if "session-hook.py" not in json.dumps(h)]
            if entries: hooks[event] = entries
            elif event in hooks: del hooks[event]
    s.write_text(json.dumps(cfg, indent=2))
    print("   hooks removed")
PYEOF
else
  echo "   ! no python3 found — remove the session-hook.py hooks from ~/.claude/settings.json manually"
fi

echo "==> removing Claude skill links"
# Iterate over the skills this repo SHIPS, not over what is in the skills directory: that
# way an unrelated skill of the same name is never even considered.
for d in "$REPO"/skills/*/; do
  name="$(basename "$d")"
  target="$CLAUDE_DIR/skills/$name"
  # only remove links that point INTO this repo — never a user's own skill
  if [ -L "$target" ] && [ "$(readlink "$target")" = "${d%/}" ]; then
    rm -f "$target" && echo "   unlinked $name"
  fi
done

echo "==> removing the OpenCode plugin (if installed)"
OCP="${XDG_CONFIG_HOME:-$HOME/.config}/opencode/plugins/session-browser.js"
if [ -f "$OCP" ]; then rm -f "$OCP" && echo "   removed $OCP"; fi

echo "==> removing cr/sb shell functions"
# Both rc files are swept, not just the one install-cr.sh would pick today: the login
# shell may have changed since installing, and leaving a dead `cr` behind is worse than
# checking a file that never had one.
for RC in "$HOME/.zshrc" "$HOME/.bashrc"; do
  [ -f "$RC" ] || continue
  if grep -q "# >>> session-browser" "$RC"; then
    cp "$RC" "$RC.sb-uninstall-backup"
    # delete both marker-delimited blocks (cr and sb)
    # `/start/,/end/d` is sed's range-delete: everything from the >>> marker line through
    # the matching <<< marker line, inclusive. Two such ranges, separated by `;`.
    # `-i.tmp` is the portable in-place form (BSD sed on macOS insists on a suffix);
    # the leftover .tmp file is removed on the next line.
    sed -i.tmp '/# >>> session-browser cr >>>/,/# <<< session-browser cr <<</d;/# >>> session-browser sb >>>/,/# <<< session-browser sb <<</d' "$RC"
    rm -f "$RC.tmp"
    echo "   removed from $RC (backup: $RC.sb-uninstall-backup)"
  fi
done

# --purge only: the registry, the logs, the enrichment facets and the OpenCode mirror all
# live under ~/.session-browser. Sessions themselves are re-indexable from the CLIs'
# transcripts; what is genuinely lost is the enrichment (LLM summaries) and the costs.
if [ "$PURGE" -eq 1 ]; then
  echo "==> purging data (~/.session-browser)"
  rm -rf "$HOME/.session-browser"
  echo "   (reasoning archive at ~/claude-reasoning-archive left intact)"
fi
# Leftovers we deliberately keep but should mention. Each line is a `[ test ] && echo`,
# so it prints only when the thing exists / applies.
echo "==> uninstalled. Also present if you want them gone:"
[ -f "$CLAUDE_DIR/settings.json.sb-backup" ] && echo "    $CLAUDE_DIR/settings.json.sb-backup   (pre-install settings backup)"
[ "$PURGE" -eq 0 ] && echo "    ~/.session-browser                  (data — rerun with --purge)"
# A script's exit status is that of its last command. With --purge the test just above is
# FALSE, so without this line `./uninstall.sh --purge` always exited 1 and looked like a
# failure to any caller or CI step. tests/test_smoke.py asserts this exits 0.
exit 0   # the last test above is false after --purge; the uninstall itself succeeded
