#!/usr/bin/env bash
# One-command install for Session Browser. Clone (or update) + install.
#
#   curl -fsSL https://raw.githubusercontent.com/mpankaj151/session-browser/main/bootstrap.sh | bash
#
# Env knobs:
#   SB_HOME=~/somewhere    where to clone (default: ~/session-browser)
#   SB_INSTALL_ARGS="--lite --no-scheduler"   passed through to install.sh
#     (--lite skips the ~2 GB semantic-search ML stack)
#
# WHEN IT RUNS
#   Only on a machine that does not have the repo yet, straight off the internet via the
#   curl line above. Everything it does is: sanity-check the machine, clone (or fast-
#   forward) the repo, then hand over to install.sh and bin/install-cr.sh. It writes
#   nothing of its own — $SB_HOME is the only path it creates.
#
# WHY IT IS SHAPED LIKE THIS
#   `curl | bash` feeds the script to bash a chunk at a time, and bash executes each
#   complete command as it arrives. If the download is cut off mid-way (flaky wifi, a
#   proxy closing the connection) a top-level script would already have run half of
#   itself. Wrapping the whole body in main() and calling it on the LAST line makes a
#   truncated download parse to nothing executable: worst case, nothing happens at all.
#   For the same reason nothing here uses `sudo` or touches anything outside $SB_HOME.
set -euo pipefail

# Everything lives inside main() and the LAST line calls it: a truncated
# `curl | bash` download parses nothing executable, so it can never run half
# a script.
# main — the whole bootstrap: preflight checks, clone/update, install, shortcuts, hints.
# Takes no arguments of its own (it is called as `main "$@"` only for tidiness; the knobs
# are the SB_* environment variables). Exits non-zero via die() on any unmet prerequisite.
main() {
  local REPO_URL="https://github.com/mpankaj151/session-browser.git"
  local SB_HOME="${SB_HOME:-$HOME/session-browser}"
  local INSTALL_ARGS="${SB_INSTALL_ARGS:-}"

  # Preflight. Fail with a fixable sentence BEFORE cloning anything, so a machine that
  # cannot run the tool is not left with a stray half-set-up directory.
  command -v git >/dev/null 2>&1 || die "git is required. Install it and re-run."
  # macOS's stock `python3` is often 3.9 (Xcode CLT); accept any 3.11+ interpreter.
  # Same search as install.sh, duplicated on purpose: this file is fetched and run on its
  # own, so it cannot source anything from the repo it has not cloned yet. PYOK is only a
  # yes/no answer — install.sh re-runs the search and picks the interpreter for the venv.
  local PYOK=""
  for c in python3.13 python3.12 python3.11 python3; do
    if command -v "$c" >/dev/null 2>&1 && \
       "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
      PYOK="$c"; break
    fi
  done
  [ -n "$PYOK" ] || die "Python 3.11+ is required (found: $(python3 -V 2>/dev/null || echo none)). macOS: brew install python@3.12 — Linux: sudo apt install python3.12 python3.12-venv. Then re-run."

  # macOS and Linux only: the background jobs are launchd/systemd, and the filesystem
  # watching the watcher relies on is POSIX. Refuse elsewhere instead of half-working.
  case "$(uname)" in
    Darwin|Linux) : ;;
    *) die "Unsupported OS '$(uname)'. macOS and Linux only." ;;
  esac

  # Re-running the one-liner on an already-bootstrapped machine is the documented upgrade
  # path, so an existing clone is fast-forwarded rather than treated as an error.
  # --ff-only refuses to merge: if the user has local commits or edits, stop and say so
  # instead of creating a merge commit or silently discarding their work.
  # --depth 1 on the fresh clone keeps the download small; nothing here needs history.
  if [ -d "$SB_HOME/.git" ]; then
    say "Updating existing clone at $SB_HOME"
    git -C "$SB_HOME" pull --ff-only || die \
      "Update blocked: local changes in $SB_HOME. Commit/stash them there, or set SB_HOME to a fresh path and re-run."
  else
    say "Cloning into $SB_HOME"
    git clone --depth 1 "$REPO_URL" "$SB_HOME"
  fi

  say "Running installer"
  # $INSTALL_ARGS is intentionally UNQUOTED so "--lite --no-scheduler" splits into two
  # separate flags — that is what the shellcheck suppression is for. Quoting it would
  # pass the whole string as a single (rejected) argument.
  # shellcheck disable=SC2086
  "$SB_HOME/install.sh" $INSTALL_ARGS

  # Unlike install.sh, the one-command path DOES add the shell shortcuts: someone running
  # a curl one-liner expects `sb ui` to work afterwards. `|| true` because a rc file that
  # cannot be written is a cosmetic loss — the installed tool still works via ./bin/*.
  say "Installing shell shortcuts (cr / sb)"
  "$SB_HOME/bin/install-cr.sh" || true

  # Mirror install-cr.sh's rc choice so "Next steps" names the right file.
  # Keep this `case` identical to the one in bin/install-cr.sh: if the two ever disagree,
  # the instructions tell the user to `source` a file the functions were not written to.
  local RC
  case "${SHELL:-}" in
    */zsh)  RC="$HOME/.zshrc" ;;
    */bash) RC="$HOME/.bashrc" ;;
    *)      RC="$HOME/.zshrc"; [ -f "$RC" ] || RC="$HOME/.bashrc" ;;
  esac

  printf '\n\033[1;32mSession Browser installed.\033[0m\n\n'
  printf 'Next:\n'
  printf '  source %s          # load the cr / sb shortcuts (or open a new terminal)\n' "$RC"
  printf '  sb ui                    # start the web UI  -> http://127.0.0.1:7655\n'
  printf '  sb doctor                # health check\n'
  printf '  sb stats                 # usage report in the terminal\n'
  printf '  sb demo                  # a synthetic-data demo (no real sessions needed)\n'
  printf '\nRepo: %s\n' "$SB_HOME"
}

# say — print one progress line in cyan. Argument: the message.
say() { printf "\033[1;36m==>\033[0m %s\n" "$1"; }
# die — print one message in red on stderr and abort with status 1. Used for every
# unmet prerequisite, so the caller (and CI) can tell a refusal from a crash.
die() { printf "\033[1;31m!!\033[0m %s\n" "$1" >&2; exit 1; }

# The last line, and the only top-level call — see the header: this is what makes a
# truncated `curl | bash` download a no-op instead of a half-run install.
main "$@"
