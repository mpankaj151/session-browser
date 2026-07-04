"""Secret redaction for anything that leaves the tool (context / bridge / export).

Transcripts and primers can echo real credentials. Before a primer is copied,
downloaded, or handed to another CLI, mask common secret shapes. Conservative by
design: prefers a few false-positives (masking a harmless token) over leaking a key.
"""
from __future__ import annotations

import re

# (label, compiled pattern). Order matters: more specific first.
_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("tavily-key", re.compile(r"tvly-[A-Za-z0-9_\-]{10,}")),
    ("context7-key", re.compile(r"ctx7sk-[A-Za-z0-9_\-]{10,}")),
    ("anthropic-key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}")),
    ("openai-key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{20,}")),
    ("github-token", re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}")),
    ("slack-token", re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}")),
    ("google-key", re.compile(r"AIza[0-9A-Za-z_\-]{30,}")),
    ("aws-key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("bearer", re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._\-]{20,}")),
    ("jwt", re.compile(r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}")),
    ("private-key-block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S)),
    # KEY=VALUE / "key": "value" assignments for secret-ish names
    ("assigned-secret", re.compile(
        r"(?i)(\b[A-Z0-9_]*(?:secret|api[_-]?key|token|password|passwd|access[_-]?key)[A-Z0-9_]*\b)"
        r"(\s*[:=]\s*['\"]?)([^\s'\"]{6,})")),
    # long hex / base64 blobs that look like keys (32+ chars)
    ("hex-blob", re.compile(r"\b[0-9a-fA-F]{32,}\b")),
]

_MASK = "«REDACTED»"


def redact(text: str) -> str:
    if not text:
        return text
    out = text
    for label, pat in _PATTERNS:
        if label == "assigned-secret":
            out = pat.sub(lambda m: f"{m.group(1)}{m.group(2)}{_MASK}", out)
        else:
            out = pat.sub(_MASK, out)
    return out


def redact_count(text: str) -> int:
    """How many secrets would be masked (for logging/preview)."""
    if not text:
        return 0
    n = 0
    for label, pat in _PATTERNS:
        n += len(pat.findall(text))
    return n
