"""Secret redaction for anything that leaves the tool (context / bridge / export).

Transcripts and primers can echo real credentials. Before a primer is copied,
downloaded, or handed to another CLI, mask common secret shapes. Conservative by
design: prefers a few false-positives (masking a harmless token) over leaking a key.
"""
from __future__ import annotations

import re

# (label, compiled pattern). Order matters: more specific first.
# Labels starting with "assigned" keep match group 1 (the name+separator) and mask
# only group 2 (the value); every other pattern masks the whole match.
_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("tavily-key", re.compile(r"tvly-[A-Za-z0-9_\-]{10,}")),
    ("context7-key", re.compile(r"ctx7sk-[A-Za-z0-9_\-]{10,}")),
    ("anthropic-key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}")),
    ("openai-key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{20,}")),
    ("github-fine-pat", re.compile(r"github_pat_[A-Za-z0-9_]{22,}")),
    ("github-token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}")),
    ("npm-token", re.compile(r"\bnpm_[A-Za-z0-9]{30,}")),
    ("slack-token", re.compile(r"xox[a-z]-[A-Za-z0-9-]{10,}")),
    ("google-key", re.compile(r"AIza[0-9A-Za-z_\-]{30,}")),
    ("aws-key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("bearer", re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._\-]{20,}")),
    ("jwt", re.compile(r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}")),
    ("private-key-block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S)),
    # KEY=VALUE / KEY: value / "key": "value" (JSON) assignments for secret-ish names.
    # The optional quotes around the name and before the value make JSON forms match.
    ("assigned-secret", re.compile(
        r"(?i)(['\"]?\b[A-Z0-9_]*(?:secret|api[_-]?key|token|password|passwd|access[_-]?key)[A-Z0-9_]*\b['\"]?"
        r"\s*[:=]\s*['\"]?)([^\s'\"]{6,})")),
    # Long hex runs, but only in a value position (after = : or an opening quote).
    # A bare hash in prose/git-log output (commit SHAs, digests) is left alone so the
    # full-text index stays searchable by hash.
    ("assigned-hex", re.compile(r"([=:]\s*['\"]?|['\"])([0-9a-fA-F]{32,})\b")),
]

_MASK = "«REDACTED»"


def redact(text: str) -> str:
    if not text:
        return text
    out = text
    for label, pat in _PATTERNS:
        if label.startswith("assigned"):
            out = pat.sub(lambda m: f"{m.group(1)}{_MASK}", out)
        else:
            out = pat.sub(_MASK, out)
    return out


def redact_count(text: str) -> int:
    """How many secrets get masked (exact: diff of mask occurrences)."""
    if not text:
        return 0
    return redact(text).count(_MASK) - text.count(_MASK)
