#!/usr/bin/env python3
"""Render a background-job template — a launchd plist (macOS) or a systemd
--user unit (Linux) — with this install's paths baked in.

    render-job.py TEMPLATE DEST --format plist|systemd

Markers and the environment variables install.sh sets for them:
    __VENV_PY__  SB_VENV_PY     __REPO__  SB_REPO      __LOG_DIR__  SB_LOG_DIR
    __HOME__     SB_HOME_DIR    __PATH__  SB_JOB_PATH

Rendered in Python, not sed: a repo path containing &, <, > or sed
metacharacters produced a malformed plist and aborted the install half-done.
plist values are XML-escaped; systemd values are escaped for the double-quoted
strings the templates put them in (ExecStart="…" "…", Environment="PATH=…").
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from xml.sax.saxutils import escape as _xml_escape

MARKERS = (("__VENV_PY__", "SB_VENV_PY"), ("__REPO__", "SB_REPO"), ("__LOG_DIR__", "SB_LOG_DIR"),
           ("__HOME__", "SB_HOME_DIR"), ("__PATH__", "SB_JOB_PATH"))


def _systemd_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def render(template: Path, env: dict, fmt: str) -> str:
    esc = _xml_escape if fmt == "plist" else _systemd_escape
    text = Path(template).read_text(encoding="utf-8")
    for marker, var in MARKERS:
        if marker in text:
            if var not in env:
                raise SystemExit(f"render-job: {var} is not set (needed for {marker} in {template})")
            text = text.replace(marker, esc(env[var]))
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("template")
    ap.add_argument("dest")
    ap.add_argument("--format", choices=("plist", "systemd"), required=True)
    args = ap.parse_args()
    out = render(Path(args.template), os.environ, args.format)
    dest = Path(args.dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(out, encoding="utf-8")
    print(f"   rendered {dest}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        if e.code and not isinstance(e.code, int):
            print(e.code, file=sys.stderr)
            sys.exit(1)
        raise
