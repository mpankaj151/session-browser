#!/usr/bin/env bash
# Session Browser installer.
#   ./install.sh [--no-hook] [--no-scheduler] [--no-backfill] [--enrich] [--lite] [--opencode-plugin]
# Idempotent. Creates the venv, builds the DB, backfills, optionally registers the
# Claude Stop hook and the background jobs (launchd on macOS, systemd --user on
# Linux). --no-launchd is kept as an alias of --no-scheduler.
#   --lite  skip sentence-transformers/torch (~2 GB download); semantic search
#           falls back to keyword + full-text.
#
# ---------------------------------------------------------------------------
# WHAT THIS TOOL IS (for a reader new to AI coding assistants)
#   Claude Code, Copilot CLI, Codex and OpenCode are terminal programs you chat with about
#   your code. Each of them records every conversation ("session") to a file on disk (a
#   "transcript"). Session Browser reads those files and keeps one row per session in a
#   small SQLite database (the "registry"), then puts a local web UI, a terminal report
#   and a plug-in server on top. docs/GLOSSARY.md defines every term used below;
#   docs/ARCHITECTURE.md draws the pipeline.
#
# WHEN IT RUNS
#   By hand, from a clone of this repo (`./install.sh`), or via bootstrap.sh which clones
#   first. Re-running it is normal and is the documented repair after the repo directory
#   is moved: every step rewrites its own artefacts with the CURRENT absolute paths.
#
# WHAT IT WRITES, AND WHERE
#   <repo>/.venv/                 the private Python environment ("venv") this tool runs
#                                 in, so its dependencies never touch the system Python.
#                                 Recreated if it turns out to be older than Python 3.11.
#   <repo>/config.toml            a small per-machine OVERRIDE file (git-ignored), written
#                                 only when absent — never a copy of config.toml.example.
#   ~/.session-browser/logs/      where the background jobs and the UI write their logs.
#   ~/.session-browser/registry.db  the registry itself, created by scripts/migrate-db.py
#                                 and filled by scripts/refresh-all.py.
#   ~/claude-reasoning-archive/   raw transcript copies + readable reasoning trails, both
#                                 produced by the refresh pipeline (not by this script).
#   ~/.claude/settings.json       the two Claude Code "hooks" (a hook is a command the CLI
#                                 runs at a fixed moment): Stop and SessionEnd, each
#                                 invoking scripts/session-hook.py. A .sb-backup copy of
#                                 the file is kept. $CLAUDE_CONFIG_DIR relocates all this.
#   ~/.claude/skills/<name>       one symlink per shipped skill directory in skills/.
#   ~/Library/LaunchAgents/com.sessionbrowser.{watcher,refresh}.plist    (macOS)
#   ~/.config/systemd/user/session-browser-{watcher.service,refresh.service,refresh.timer}
#                                 (Linux) — the always-on watcher + the nightly refresh.
#   ~/.config/opencode/plugins/session-browser.js   only with --opencode-plugin.
#
# WHAT IT NEVER TOUCHES
#   Your shell rc. The `cr` / `sb` shell functions are a separate, opt-in step
#   (./bin/install-cr.sh); this script only prints it as the next thing to run.
#   Any transcript a CLI wrote — the whole tool is read-only towards those.
#   ~/.claude.json and .mcp.json: registering the MCP server is manual (docs/SETUP.md §6).
#
# EXIT BEHAVIOUR
#   `set -e` aborts on an unexpected failure, but every step that can legitimately fail on
#   a given laptop (a pipeline stage, hook registration, launchctl) is guarded with
#   `|| echo ...` so a partial environment still ends up with a working install.
# ---------------------------------------------------------------------------
set -euo pipefail
# REPO is the absolute path of the directory holding THIS file, resolved rather than
# assumed: the installer may be invoked as `./install.sh`, `bash ~/x/install.sh`, or from
# bootstrap.sh, and every artefact it writes (hook command, plist, systemd unit, shell
# function) bakes this absolute path in.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The venv interpreter. It may not exist yet — step 1 creates it — so every use before
# then is guarded with `[ -x "$PY" ]`.
PY="$REPO/.venv/bin/python"
# $HOME is captured because the background jobs need it spelled out literally: launchd
# and systemd start them with a near-empty environment.
HOME_DIR="$HOME"
LOG_DIR="$HOME/.session-browser/logs"
# Claude Code relocates its state with CLAUDE_CONFIG_DIR; hooks/skills go there.
CLAUDE_DIR="${CLAUDE_CONFIG_DIR:-$HOME/.claude}"
# Claude-specific steps (hook, skills) only make sense where Claude Code is —
# on an OpenCode/Codex/Copilot-only laptop they would just create ~/.claude.
# Either test is enough: the binary on PATH, or a state directory left behind by a past
# install (Claude Code may be installed but not on this shell's PATH).
HAVE_CLAUDE=0
if command -v claude >/dev/null 2>&1 || [ -d "$CLAUDE_DIR" ]; then HAVE_CLAUDE=1; fi

# Flag parser. Defaults are the safe, cheap install: hook + scheduler + backfill ON,
# LLM enrichment OFF (it spends your CLI subscription's quota), the ~2 GB semantic-search
# stack ON, OpenCode plugin OFF. Each variable is 1 when the behaviour is DISABLED, except
# NO_ENRICH which starts at 1 because enrichment is opt-in. An unknown flag is fatal
# rather than ignored — a typo'd `--no-hooks` silently installing the hook is worse.
NO_HOOK=0; NO_LAUNCHD=0; NO_BACKFILL=0; NO_ENRICH=1; LITE=0; OC_PLUGIN=0   # enrich off by default (uses LLM quota)
for a in "$@"; do case "$a" in
  --no-hook) NO_HOOK=1;; --no-launchd|--no-scheduler) NO_LAUNCHD=1;;
  --no-backfill) NO_BACKFILL=1;; --enrich) NO_ENRICH=0;; --lite) LITE=1;;
  --opencode-plugin) OC_PLUGIN=1;;
  *) echo "unknown flag: $a"; exit 1;; esac; done

echo "==> Session Browser install ($REPO)"

# Python 3.11+ required (tomllib, sqlite FTS5). Don't hard-require that plain
# `python3` be new: stock macOS resolves it to the Xcode CLT build (often 3.9)
# even when a modern interpreter is installed as python3.12 — hunt for one.
# PYBOOT is the "bootstrap" interpreter: the one used ONLY to create the venv. Once the
# venv exists, everything else runs through $PY. The probe is a one-liner that exits 0
# when the interpreter is new enough, so an interpreter that is present but too old is
# rejected the same way as one that is missing.
PYBOOT=""
# An existing venv that is new enough needs no bootstrap interpreter at all —
# re-running the installer must not fail just because python3.12 left PATH.
if [ -x "$PY" ] && "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
  PYBOOT="$PY"
fi
# Newest first, `python3` last: prefer an explicitly versioned binary over whatever the
# bare name happens to resolve to. The `[ -n "$PYBOOT" ] && break` guard makes the whole
# loop a no-op when the existing venv already qualified above.
for c in python3.13 python3.12 python3.11 python3; do
  [ -n "$PYBOOT" ] && break
  if command -v "$c" >/dev/null 2>&1 && \
     "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
    PYBOOT="$(command -v "$c")"; break
  fi
done
# Nothing usable anywhere: stop with the exact install command for each OS rather than
# letting a 3.9 interpreter get as far as `import tomllib` and a confusing traceback.
if [ -z "$PYBOOT" ]; then
  echo "!! Python 3.11+ is required (found: $(python3 -V 2>/dev/null || echo none))."
  echo "   macOS:  brew install python@3.12"
  echo "   Linux:  sudo apt install python3.12 python3.12-venv  (or your distro's equivalent)"
  echo "   Then re-run this script."
  exit 1
fi

# 1. venv + dependencies (idempotent — pip skips what's already satisfied).
#    --system-site-packages reuses an existing torch/sentence-transformers
#    install when one is present; harmless otherwise.
# A venv built on an old interpreter cannot be upgraded in place — the interpreter is
# baked into it — so the only fix is to delete and recreate. Deleting .venv is safe: it
# holds no user data, only downloaded packages.
if [ -x "$PY" ] && ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
  # An existing venv on an interpreter older than 3.11 was detected as too old
  # and then used anyway (tomllib missing at the first script). Rebuild it.
  echo "==> existing venv is older than Python 3.11 — recreating it with $PYBOOT"
  rm -rf "$REPO/.venv"
fi
# Create only when missing (or just deleted): re-running the installer on a healthy
# machine must not throw the environment away and re-download everything.
if [ ! -x "$PY" ]; then
  echo "==> creating venv ($PYBOOT)"
  "$PYBOOT" -m venv --system-site-packages "$REPO/.venv"
  "$PY" -m pip install --quiet --upgrade pip
fi
echo "==> installing dependencies"
"$PY" -m pip install --quiet -r "$REPO/requirements.txt"
# The semantic-search stack (sentence-transformers + torch) is big and optional. Its
# absence is a MODE, not a failure: semsearch.py falls back to keyword + full-text search,
# so a failed install only warns. --lite skips the attempt entirely.
if [ "$LITE" -eq 0 ]; then
  "$PY" -m pip install --quiet "sentence-transformers>=2.7" \
    || echo "   ! sentence-transformers install failed — semantic search will fall back to keyword"
  # Pre-download the embedding model NOW (the one moment a big fetch is
  # expected). Runtime queries never go online — semsearch is offline-only
  # unless SB_ALLOW_MODEL_DOWNLOAD=1.
  echo "==> caching the embedding model (one-time download)"
  SB_REPO="$REPO" SB_ALLOW_MODEL_DOWNLOAD=1 "$PY" - <<'PYEOF' \
    || echo "   ! model download failed — the nightly embed job will retry"
# This runs on the venv's python, fed on stdin (`"$PY" -`), so the repo is not on the
# import path by default — SB_REPO is exported by the shell line above for exactly this.
import os, sys
sys.path.insert(0, os.environ["SB_REPO"])
# get_model() downloads and caches the sentence-transformers model on first call. Later
# calls (searching, the nightly embed job) find it in the cache and stay offline.
import semsearch
semsearch.get_model()
PYEOF
fi

# 2. config — a minimal OVERRIDE file, not a copy of the example. config.toml is
#    layered over config.toml.example at load time, so a full copy would freeze
#    today's defaults and silently miss every source/provider a later git pull
#    adds (this is exactly how a pre-OpenCode config hid OpenCode).
if [ ! -f "$REPO/config.toml" ]; then
  cat > "$REPO/config.toml" <<'EOF'
# Per-machine overrides for Session Browser (this file is git-ignored).
# Layered over config.toml.example: anything NOT set here keeps the example's
# documented default, so sources and providers added by `git pull` just work.
# Only write the keys you want to change, e.g.
#
# [enrichment]
# provider = "opencode-headless"     # summarise through OpenCode on this machine
#
# [sources.codex]
# enabled = false                    # never index Codex here
EOF
  echo "==> wrote config.toml (overrides only; defaults come from config.toml.example)"
fi

# 3. runtime dirs + schema. migrate-db.py creates ~/.session-browser/registry.db if it is
#    missing and otherwise adds whatever columns/tables this version needs — it is a
#    forward-only migration and is safe to run on every install.
mkdir -p "$LOG_DIR"
"$PY" "$REPO/scripts/migrate-db.py"

# 4. backfill + full processing pipeline (idempotent). refresh-all.py is the same script
#    the nightly job runs: index every transcript found on disk, then compute costs,
#    extract reasoning trails, build the full-text index and the embeddings. --enrich
#    additionally asks a coding CLI to summarise each session, which spends that CLI
#    subscription's quota — hence opt-in.
if [ "$NO_BACKFILL" -eq 0 ]; then
  echo "==> running full pipeline (backfill, cost, reasoning, full-text, embeddings)"
  # Never abort here: a failing pipeline step (no FTS5 in this sqlite, a pinned
  # provider missing) used to stop the installer BEFORE the hooks and the
  # scheduler were installed, leaving a database nothing keeps fresh.
  if [ "$NO_ENRICH" -eq 0 ]; then
    "$PY" "$REPO/scripts/refresh-all.py" --enrich || echo "   ! a pipeline step failed (see above) — continuing; re-run 'sb refresh' after fixing it"
  else
    "$PY" "$REPO/scripts/refresh-all.py" || echo "   ! a pipeline step failed (see above) — continuing; re-run 'sb refresh' after fixing it"
  fi
fi

# 6. Stop + SessionEnd hooks (Stop = instant indexing; SessionEnd = indexing +
#    journal-grade enrichment of the just-ended session)
#    A "hook" is a shell command Claude Code runs itself at a fixed moment, configured in
#    its settings.json. This is the fast half of the two-tier live indexing described in
#    docs/ARCHITECTURE.md: the hook indexes a session within tens of milliseconds of it
#    ending, and the always-on watcher (step 7) catches everything the hook missed and
#    everything the other CLIs do. Losing the hook therefore degrades latency, never
#    correctness — which is why this whole step is optional and never fatal.
if [ "$NO_HOOK" -eq 0 ] && [ "$HAVE_CLAUDE" -eq 0 ]; then
  echo "==> Claude Code not detected — skipping its Stop/SessionEnd hook and skills (the watcher indexes everything)"
fi
if [ "$NO_HOOK" -eq 0 ] && [ "$HAVE_CLAUDE" -eq 1 ]; then
  echo "==> registering Claude Stop + SessionEnd hooks"
  "$PY" - "$REPO" <<'PYEOF' || echo "   ! hook registration failed — everything else still works (watcher covers indexing)"
# Editing someone else's JSON config by hand in shell would be reckless, so the rewrite is
# done in Python on stdin. tests/test_smoke.py extracts this exact block out of the file
# and runs it against a temporary HOME, so it must stay self-contained: argv[1] is the
# repo path, and nothing here may import from the repo.
import json, os, sys, shutil
from pathlib import Path
repo = Path(sys.argv[1])
# A brand-new install has no ~/.claude yet; CLAUDE_CONFIG_DIR relocates it.
settings = Path(os.path.expanduser(os.environ.get("CLAUDE_CONFIG_DIR") or "~/.claude"))/"settings.json"
settings.parent.mkdir(parents=True, exist_ok=True)
# Back up BEFORE parsing: if the user's file is malformed we must not have
# touched anything, and they keep a copy either way.
if settings.exists():
    shutil.copy2(settings, settings.with_suffix(".json.sb-backup"))
try:
    cfg = json.loads(settings.read_text()) if settings.exists() else {}
except (json.JSONDecodeError, OSError) as e:
    print(f"   ! {settings} is not valid JSON ({e}) — skipping hook registration.")
    print("     Fix the file, then re-run: ./install.sh --no-backfill --no-scheduler")
    sys.exit(0)
# Every bail-out below exits 0, not 1: a weird settings.json must not fail the install.
# The user keeps a working watcher, and the message says exactly how to retry.
if not isinstance(cfg, dict):
    print("   ! settings.json is not a JSON object — skipping hook registration"); sys.exit(0)
# Paths quoted: the hook command runs through a shell, and the repo path may
# contain spaces (e.g. ~/My Projects/session-browser).
cmd = f'"{repo}/.venv/bin/python" "{repo}/scripts/session-hook.py"'
hooks = cfg.setdefault("hooks", {})
if not isinstance(hooks, dict):
    print("   ! settings.json 'hooks' is not an object — skipping hook registration"); sys.exit(0)
# Claude Code's settings.json shape, for reference:
#   {"hooks": {"Stop": [{"hooks": [{"type": "command", "command": "..."}]}], ...}}
# Stop fires when a reply finishes; SessionEnd when the whole session closes.
changed = []
for event in ("Stop", "SessionEnd"):
    entries = hooks.get(event)
    if not isinstance(entries, list):
        entries = hooks[event] = []
    # Presence isn't enough: after the repo is moved, a session-hook.py entry
    # still exists but points at the dead path. Rebuild the desired state (all
    # non-ours entries + exactly one entry with the CURRENT path).
    # json.dumps(h) is a cheap "does this entry mention our script anywhere" test that
    # does not care how deeply Claude Code nests the command in a future format.
    kept = [h for h in entries if "session-hook.py" not in json.dumps(h)]
    desired = kept + [{"hooks": [{"type": "command", "command": cmd}]}]
    # Compare canonically (sorted keys) so a pure key-order difference is not mistaken
    # for a change — that would rewrite settings.json on every single install.
    if json.dumps(desired, sort_keys=True) != json.dumps(entries, sort_keys=True):
        hooks[event] = desired
        changed.append(event)
if not changed:
    print("   hooks already present")
else:
    settings.write_text(json.dumps(cfg, indent=2))
    print(f"   hook(s) registered for {', '.join(changed)} (backup: settings.json.sb-backup)")
PYEOF
fi

# 6a'. OpenCode plugin (opt-in) — the Stop-hook equivalent: index a session the
#      moment a turn settles, re-sync on deletion. Auto-loaded from
#      ~/.config/opencode/plugins/; the watcher already covers OpenCode within
#      seconds, so this is for the instant-index feel.
#      The elif exists purely to advertise the flag: when OpenCode is installed but the
#      plugin was not requested, print one line rather than silently doing nothing.
if [ "$OC_PLUGIN" -eq 1 ]; then
  echo "==> installing the OpenCode plugin"
  "$PY" "$REPO/scripts/install-opencode-plugin.py"
elif command -v opencode >/dev/null 2>&1; then
  echo "==> OpenCode detected: add --opencode-plugin for instant indexing (the watcher already covers it within seconds)"
fi

# 6b. Claude skills — symlink each shipped skill into ~/.claude/skills so the
#     work-journal / snapshot / checkpoint skills are discoverable. Symlinks (not
#     copies) so a repo update updates the skills; skill updates ride git pull.
#     A "skill" here is just a Markdown instruction file (skills/<name>/SKILL.md) that
#     Claude Code loads on demand — see docs/GLOSSARY.md. Linking them is tied to the
#     --no-hook flag because both are the same "integrate with Claude Code" decision.
#     `${d%/}` strips the trailing slash the glob adds, so readlink comparisons match.
if [ "$NO_HOOK" -eq 0 ] && [ "$HAVE_CLAUDE" -eq 1 ]; then
  echo "==> linking Claude skills"
  mkdir -p "$CLAUDE_DIR/skills"
  for d in "$REPO"/skills/*/; do
    name="$(basename "$d")"
    # Scaffold skills document a future feature; linking them made the agent run
    # scripts that do not exist yet.
    # A skill opts out by saying "scaffold" in its SKILL.md `description:` front-matter
    # line — the marker lives with the skill, so no list has to be maintained here.
    target="$CLAUDE_DIR/skills/$name"
    if grep -qi "^description:.*scaffold" "$d/SKILL.md" 2>/dev/null; then
      # an earlier install linked it: retire our own link, never a user's file
      if [ -L "$target" ] && [ "$(readlink "$target")" = "${d%/}" ]; then rm -f "$target"; echo "   unlinked scaffold skill $name"; fi
      continue
    fi
    if [ -L "$target" ]; then
      # repoint after a repo move; no-op when already correct
      [ "$(readlink "$target")" = "${d%/}" ] || { ln -sfn "${d%/}" "$target"; echo "   repointed $name"; }
    # A real file/directory of that name is the user's own skill. Never overwrite it;
    # say so and move on.
    elif [ -e "$target" ]; then
      echo "   ! $target exists and is not ours — leaving it alone"
    else
      ln -s "${d%/}" "$target"
      echo "   linked $name"
    fi
  done
fi

# 7. background jobs — launchd (macOS) or systemd --user (Linux). The watcher is
#    live indexing; refresh is the nightly full pipeline (cost/reasoning/fts/
#    embed/enrich). Both are what keep the browser current without a hand run.
#    launchd (macOS) and systemd --user (Linux) are the operating system's own "keep this
#    running / run this at 01:00" services. Their job files are rendered from the
#    templates in launchd/ and systemd/ by scripts/render-job.py, which substitutes the
#    SB_* variables exported just below. The key difficulty: a background job starts with
#    an almost empty environment — no PATH worth the name, none of your shell's exports —
#    so anything the jobs need has to be written INTO the job file. That is what the next
#    two blocks compute.
if [ "$NO_LAUNCHD" -eq 0 ]; then
  # Job PATH: system dirs + Homebrew (both arches) + wherever the user's CLI
  # binaries actually live (npm globals, volta, nvm…) — enrichment shells out
  # to the CLI, and a PATH miss silently disables it.
  JOB_PATH="/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin:$HOME_DIR/.local/bin"
  # Then append the real directory of each CLI found in THIS shell. The `case` is the
  # shell idiom for "is this directory already in the list": it wraps both sides in
  # colons so a prefix like /usr/bin cannot match /usr/bin-other.
  for c in claude copilot codex opencode; do
    B="$(command -v "$c" 2>/dev/null || true)"
    if [ -n "$B" ]; then D="$(dirname "$B")"; case ":$JOB_PATH:" in *":$D:"*) ;; *) JOB_PATH="$JOB_PATH:$D";; esac; fi
  done
  # CLI homes relocated in this shell must reach the jobs too, or they index
  # the default (usually empty) trees — or a different OpenCode DB.
  # Each of these four env vars moves one CLI's data somewhere non-default:
  #   CLAUDE_CONFIG_DIR  Claude Code's state dir  (default ~/.claude)
  #   CODEX_HOME         Codex's state dir        (default ~/.codex)
  #   XDG_DATA_HOME      where OpenCode keeps its database (default ~/.local/share)
  #   OPENCODE_DB        an explicit path to that database, overriding the above
  # The daemons run outside your shell, so an unexported-to-them CLAUDE_CONFIG_DIR means
  # the watcher happily watches an empty ~/.claude while your real sessions pile up
  # elsewhere — and, worse for OpenCode, a job pointed at a DIFFERENT database would treat
  # every session it cannot see as deleted. JOB_ENV collects the ones that are actually
  # set, one `NAME=value` per line; render-job.py turns that into the plist/unit syntax.
  # (Changed one of these since installing? Re-run ./install.sh to re-bake the jobs.)
  JOB_ENV=""
  # `eval` is how POSIX shell reads a variable whose NAME is itself in a variable:
  # it expands to `val=${CLAUDE_CONFIG_DIR:-}` and so on, one loop pass at a time.
  for v in CLAUDE_CONFIG_DIR CODEX_HOME XDG_DATA_HOME OPENCODE_DB; do
    eval "val=\${$v:-}"
    [ -n "$val" ] && JOB_ENV="${JOB_ENV}${v}=${val}
"
  done
  # scripts/render-job.py reads these SB_* variables out of the environment and
  # substitutes them into launchd/*.plist.template and systemd/*.template.
  export SB_VENV_PY="$PY" SB_REPO="$REPO" SB_LOG_DIR="$LOG_DIR" SB_HOME_DIR="$HOME_DIR" SB_JOB_PATH="$JOB_PATH" SB_JOB_ENV="$JOB_ENV"
  if [ "$(uname)" = "Darwin" ]; then
    echo "==> installing launchd jobs"
    AGENTS="$HOME/Library/LaunchAgents"; mkdir -p "$AGENTS"
    # unload-then-load, because launchd keeps the OLD plist in memory: without the unload
    # a re-install after a repo move would leave the previous paths running. The unload is
    # `|| true` since it legitimately fails on a first install (nothing loaded yet), and
    # the load only warns — a machine where launchd refuses (SSH session, locked-down
    # profile) still has a complete install, it just has to refresh by hand.
    for job in watcher refresh; do
      "$PY" "$REPO/scripts/render-job.py" "$REPO/launchd/$job.plist.template" \
        "$AGENTS/com.sessionbrowser.$job.plist" --format plist
      launchctl unload "$AGENTS/com.sessionbrowser.$job.plist" 2>/dev/null || true
      launchctl load "$AGENTS/com.sessionbrowser.$job.plist" \
        || echo "   ! could not load $job — check: launchctl list | grep sessionbrowser"
    done
  # Both tests are needed: systemctl can exist while there is no per-user service manager
  # to talk to (a bare SSH login, a container). `show-environment` is the cheap probe that
  # the user bus is actually reachable; without it `systemctl --user enable` would fail
  # confusingly instead of falling through to the "schedule these yourself" branch.
  elif command -v systemctl >/dev/null 2>&1 && systemctl --user show-environment >/dev/null 2>&1; then
    echo "==> installing systemd --user units"
    UNITS="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"; mkdir -p "$UNITS"
    for u in session-browser-watcher.service session-browser-refresh.service session-browser-refresh.timer; do
      "$PY" "$REPO/scripts/render-job.py" "$REPO/systemd/$u.template" "$UNITS/$u" --format systemd
    done
    # daemon-reload makes systemd re-read the unit files just written; `enable --now`
    # both starts them and marks them to start at every login. The .timer is what fires
    # the nightly refresh.service — timers replace cron in the systemd world.
    systemctl --user daemon-reload
    systemctl --user enable --now session-browser-watcher.service session-browser-refresh.timer \
      || echo "   ! could not enable the units — check: systemctl --user status session-browser-watcher"
    # --now starts a unit only if it is not already active: after a repo move the
    # OLD watcher process kept running from the previous path.
    # `try-restart` restarts the unit if (and only if) it is running — the systemd
    # equivalent of the launchd unload/load pair above, and a harmless no-op on a first
    # install, which is why its failure is swallowed.
    systemctl --user try-restart session-browser-watcher.service 2>/dev/null || true
    echo "   (units run while you are logged in; to keep them after logout: loginctl enable-linger $USER)"
  # No scheduler at all (a container, a plain SSH session, an unusual distro): print the
  # two commands so the user can wire them into cron / their rc file themselves. Nothing
  # is broken — indexing simply stops being automatic.
  else
    echo "==> no launchd / systemd --user session here. Schedule these yourself (docs/SETUP.md §5):"
    echo "      watcher (live indexing):  $PY $REPO/watcher.py"
    echo "      nightly refresh:          $PY $REPO/scripts/refresh-all.py --enrich"
  fi
fi

# Closing summary. install-cr.sh is deliberately NOT run from here: adding functions to
# someone's shell rc is a separate consent. tests/test_smoke.py greps this script's output
# for "Session Browser installed", so the success line has to stay printed on every path.
printf '\n\033[1;32m✓ Session Browser installed.\033[0m\n\n'
echo "Next:"
echo "  ./bin/install-cr.sh      # once — adds the cr / sb shell commands"
echo "  source ~/.zshrc          # or open a new terminal (installer prints which rc)"
echo "  sb ui                    # start the web UI  ->  http://127.0.0.1:<[ui].port, default 7655>"
echo "  sb demo                  # or try it on synthetic data first"
echo "  sb doctor                # health check"
