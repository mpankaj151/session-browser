#!/usr/bin/env bash
# Health check for the Session Browser install.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO/.venv/bin/python"
ok(){ printf "  \033[32m✓\033[0m %s\n" "$1"; }
bad(){ printf "  \033[31m✗\033[0m %s\n" "$1"; }

echo "Session Browser — doctor"
echo "repo: $REPO"

echo "[python & deps]"
if [ -x "$PY" ]; then ok "venv python present"; else bad "venv python missing — run install.sh"; fi
"$PY" - <<'PYEOF'
import importlib, sys
for m in ["flask","numpy","watchdog","yaml"]:
    try: importlib.import_module(m); print(f"  \033[32m✓\033[0m import {m}")
    except Exception as e: print(f"  \033[31m✗\033[0m import {m}: {e}")
# optional (--lite installs skip it): absence is a mode, not a failure
try: importlib.import_module("sentence_transformers"); print("  \033[32m✓\033[0m import sentence_transformers")
except Exception: print("  \033[33m∼\033[0m sentence_transformers absent (--lite: semantic search falls back to keyword/full-text)")
# sqlite extension capability (optional fast path)
import sqlite3
c=sqlite3.connect(":memory:")
# no backslashes inside f-string {} — that's a SyntaxError before Python 3.12 (PEP 701)
loadable = hasattr(c, "enable_load_extension")
mark = "\033[32m✓\033[0m" if loadable else "\033[33m∼\033[0m"
state = "available" if loadable else "absent (numpy backend in use)"
print(f"  {mark} sqlite loadable-extensions {state}")
PYEOF

echo "[database]"
"$PY" - "$REPO" <<'PYEOF'
import sys; sys.path.insert(0, sys.argv[1])   # repo path, NOT cwd — doctor may run from anywhere
import indexer, sbconfig
try:
    c=indexer.connect()
    n=c.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    e=c.execute("SELECT COUNT(*) FROM session_embeddings").fetchone()[0]
    r=c.execute("SELECT COUNT(*) FROM sessions WHERE reasoning_path IS NOT NULL").fetchone()[0]
    print(f"  \033[32m✓\033[0m db at {sbconfig.DB_PATH}")
    print(f"  \033[32m✓\033[0m {n} sessions · {e} embedded · {r} with reasoning trails")
except Exception as ex:
    print(f"  \033[31m✗\033[0m db error: {ex}")
PYEOF

echo "[resume sync]"
"$PY" - "$REPO" <<'PYEOF'
import collections, sys; sys.path.insert(0, sys.argv[1])
from pathlib import Path
from sources.registry import _make_claude
proj = _make_claude().projects_dir          # honours $CLAUDE_CONFIG_DIR
real = collections.defaultdict(list)   # session_id -> [dirs] for REAL files
links = 0
if proj.exists():
    for d in proj.iterdir():
        if not d.is_dir(): continue
        for f in d.glob("*.jsonl"):
            if f.is_symlink(): links += 1
            else: real[f.stem].append(d.name)
dupes = {k: v for k, v in real.items() if len(v) > 1}
print(f"  \033[32m✓\033[0m {links} resume symlink(s) (always in sync with origin)")
if dupes:
    print(f"  \033[33m∼\033[0m {len(dupes)} session(s) have diverged REAL copies in >1 dir:")
    for sid, dirs in list(dupes.items())[:5]:
        print(f"      {sid[:8]} in {len(dirs)} dirs")
    print("      -> reconcile safely (forks preserved):  scripts/reconcile-sessions.py --apply")
else:
    print("  \033[32m✓\033[0m no diverged copies — every session has a single canonical transcript")
PYEOF

echo "[sources]"
"$PY" - "$REPO" <<'PYEOF'
import sys; sys.path.insert(0, sys.argv[1])
from sources.registry import build_source_registry
for name,a in build_source_registry().items():
    avail=a.is_available()          # transcripts on disk (indexing never needs the binary)
    hb=getattr(a, "has_binary", None)
    binary="" if hb is None else (" · binary on PATH" if hb() else " · binary NOT on PATH (resume/bridge from this shell will fail)")
    mark="\033[32m✓\033[0m" if avail else "\033[33m∼\033[0m"
    print(f"  {mark} {name}: {'transcripts found' if avail else 'no transcripts yet (nothing to index)'}{binary}")
PYEOF

echo "[enrichment]"
"$PY" - "$REPO" <<'PYEOF'
import shutil, sys; sys.path.insert(0, sys.argv[1])
import sbconfig
from enrichment.provider import _PROVIDERS, get_provider, resolve_provider_name
configured = sbconfig.CONFIG.get("enrichment", {}).get("provider", "auto")
name = resolve_provider_name(sbconfig.CONFIG)
label = f"{configured} -> {name}" if configured == "auto" and name else str(configured)
if name is None:
    print("  \033[33m∼\033[0m provider: auto found no summariser CLI on PATH (claude / opencode / copilot) — "
          "enrichment is skipped; install one or set [enrichment].provider")
elif name == "none":
    print("  \033[33m∼\033[0m provider: none (no LLM summaries)")
elif name not in _PROVIDERS:
    print(f"  \033[31m✗\033[0m provider: {name!r} is not a known provider — summaries will be empty")
else:
    prov = get_provider(sbconfig.CONFIG)
    binary = getattr(prov, "binary", _PROVIDERS[name][2])
    model = getattr(prov, "model", "")
    have = shutil.which(binary) is not None
    mark = "\033[32m✓\033[0m" if have else "\033[31m✗\033[0m"
    print(f"  {mark} provider: {label}  model: {model or '(CLI default)'}  binary: {binary} "
          f"{'found' if have else 'NOT on PATH'}")
PYEOF

echo "[hook]"
SETTINGS="${CLAUDE_CONFIG_DIR:-$HOME/.claude}/settings.json"
if grep -q "session-hook.py" "$SETTINGS" 2>/dev/null; then ok "Claude Stop hook registered"; else
  if command -v claude >/dev/null 2>&1; then
    printf "  \033[33m∼\033[0m Claude Stop hook NOT registered (live indexing still works via watcher; ./install.sh --no-backfill --no-scheduler)\n"
  else
    printf "  \033[33m∼\033[0m Claude Code not installed here — no Stop hook needed\n"
  fi
fi
OCP_STATUS="$("$PY" "$REPO/scripts/install-opencode-plugin.py" --status 2>/dev/null || echo "unknown")"
case "$OCP_STATUS" in
  installed) ok "OpenCode plugin installed";;
  stale*) printf "  \033[31m✗\033[0m OpenCode plugin %s\n" "$OCP_STATUS";;
  *) printf "  \033[33m∼\033[0m OpenCode plugin not installed (./install.sh --opencode-plugin; the watcher covers OpenCode within seconds)\n";;
esac

echo "[watcher / ui]"
# One line per background job, phrased for the scheduler this OS actually has.
if [ "$(uname)" = "Darwin" ]; then
  # Capture once; grep against a here-string so `grep -q` closing early can't
  # SIGPIPE launchctl and trip pipefail (which would falsely report "not loaded").
  JOBS="$(launchctl list 2>/dev/null || true)"
  if grep -q sessionbrowser.watcher <<< "$JOBS"; then ok "watcher launchd job loaded"; else
    printf "  \033[33m∼\033[0m watcher launchd job not loaded (./install.sh --no-backfill)\n"; fi
  if grep -q sessionbrowser.refresh <<< "$JOBS"; then ok "nightly refresh job loaded"; else
    printf "  \033[33m∼\033[0m nightly refresh job not loaded (run refresh-all.py manually to update)\n"; fi
elif command -v systemctl >/dev/null 2>&1; then
  if systemctl --user is-active --quiet session-browser-watcher.service 2>/dev/null; then ok "watcher systemd --user unit active"; else
    printf "  \033[33m∼\033[0m watcher systemd --user unit not active (docs/SETUP.md §5)\n"; fi
  if systemctl --user is-active --quiet session-browser-refresh.timer 2>/dev/null; then ok "nightly refresh systemd timer active"; else
    printf "  \033[33m∼\033[0m nightly refresh systemd timer not active (docs/SETUP.md §5; or run refresh-all.py manually)\n"; fi
else
  printf "  \033[33m∼\033[0m no launchd/systemd here — run watcher.py and scripts/refresh-all.py yourself (docs/SETUP.md §5)\n"
fi
# Is something listening on the UI port? lsof is not a given on Linux.
port_held() {
  if command -v lsof >/dev/null 2>&1; then lsof -ti tcp:7655 >/dev/null 2>&1
  elif command -v ss >/dev/null 2>&1; then ss -ltn 2>/dev/null | grep -q ':7655 '
  else return 1; fi
}
# --max-time: a wedged/suspended process holding the port accepts the TCP
# connect but never answers — without a deadline this health check hangs forever.
if curl -s --max-time 3 localhost:7655/health >/dev/null 2>&1; then ok "UI responding on :7655"; else
  if port_held; then
    printf "  \033[31m✗\033[0m :7655 is held by a process that isn't answering — try: sb stop, then sb ui\n"
  else
    printf "  \033[33m∼\033[0m UI not running (start with: sb ui)\n"
  fi
fi
