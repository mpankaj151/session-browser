#!/usr/bin/env bash
# Verify each CLI binary is reachable (used by enrichment / resume).
#
#   ./bin/check-cli-access.sh        no arguments, reads nothing, always exits 0
#
# For each of the four supported coding CLIs it prints one line: green with the resolved
# absolute path if the binary is on PATH, amber if it is not. That is the whole script —
# it is the smallest possible answer to "which CLIs can this shell actually run?".
#
# WHY THAT QUESTION MATTERS, and what it is NOT
#   Having transcripts on disk is what indexing, search, stats and Restore need; having
#   the BINARY is a separate thing, needed only to resume a session (bin/resume-here.sh)
#   and to run enrichment, which summarises sessions by invoking a CLI non-interactively.
#   So an amber line here never means sessions are missing — see bin/doctor.sh, whose
#   [sources] section reports transcripts and binary side by side.
#   PATH differs between your terminal and a background job, which is exactly the bug this
#   helps diagnose: install.sh bakes the directory of each CLI found HERE into the launchd
#   plist / systemd unit, so running this in the same shell you ran the installer from
#   tells you what the nightly enrichment job will see.
#
# Never fails: a missing binary is a normal state, not an error, so there is no `-e` and
# no non-zero exit — callers can print the output without special-casing anything.
set -uo pipefail
# `command -v` is the portable "where is this binary" builtin (`which` is not guaranteed).
# %-8s pads the name so the four lines align.
for cli in claude copilot codex opencode; do
  if command -v "$cli" >/dev/null 2>&1; then
    printf "  \033[32m✓\033[0m %-8s -> %s\n" "$cli" "$(command -v "$cli")"
  else
    printf "  \033[33m∼\033[0m %-8s not on PATH\n" "$cli"
  fi
done
