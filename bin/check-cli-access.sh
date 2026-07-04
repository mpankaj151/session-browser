#!/usr/bin/env bash
# Verify each CLI binary is reachable (used by enrichment / resume).
set -uo pipefail
for cli in claude copilot codex; do
  if command -v "$cli" >/dev/null 2>&1; then
    printf "  \033[32m✓\033[0m %-8s -> %s\n" "$cli" "$(command -v "$cli")"
  else
    printf "  \033[33m∼\033[0m %-8s not on PATH\n" "$cli"
  fi
done
