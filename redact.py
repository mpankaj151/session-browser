"""Secret redaction for anything that leaves the tool (context / bridge / export).

Transcripts and primers can echo real credentials. Before a primer is copied,
downloaded, or handed to another CLI, mask common secret shapes. Conservative by
design: prefers a few false-positives (masking a harmless token) over leaking a key.

Why this module exists at all: a transcript is a recording of a terminal session, so it
contains whatever scrolled past — an `export` line, a `curl -H "Authorization: ..."`, a
database URL with its password, the contents of a `.env` file someone asked about. All of
that is faithfully stored on disk. Every path that carries such text OUT of this tool
funnels through `redact()` first: Copy Context and Export in the UI, the bridge primer
handed to another CLI, the full-text search index, the reasoning archive's Markdown
trails, the prompt sent to a summariser model and the summary it returns, and every MCP
tool result. See docs/ARCHITECTURE.md, "Redaction at every egress".

How it works: a fixed, ordered list of regular expressions, each labelled. Most patterns
mask their whole match. Patterns whose label starts with "assigned" keep the *name* part
(`API_KEY=`, `Authorization: Bearer `, `://user:`) and mask only the value, so the reader
can still see WHICH credential was present and what shape the line had — useful when
reading a trail, and it keeps the surrounding text parseable.

Two deliberate limits a newcomer should know:
  * This is pattern matching, not parsing. A credential in a shape nobody listed here
    survives. The list is meant to be extended as new token formats appear.
  * A bare 40-character hex string in prose is NOT masked. Commit SHAs, file digests and
    container image ids look exactly like a secret to a regex, and masking them would
    make the full-text index useless for "find the session where I touched 3031ee38…".
    Hex is only masked in a value position (after `=`, `:` or an opening quote).

Nothing here mutates its input: `redact()` returns a new string. It is pure and has no
I/O, so it is safe to call on every row in a loop.
"""
from __future__ import annotations

import re

# (label, compiled pattern). Order matters: more specific first.
# Labels starting with "assigned" keep match group 1 (the name+separator) and mask
# only group 2 (the value); every other pattern masks the whole match.
#
# Why order matters concretely: the generic "assigned-secret" rule near the bottom would
# happily eat `OPENAI_API_KEY=sk-xxxx…` as a name/value pair and keep the name. The
# specific vendor rules run FIRST so the token is masked even when it appears bare, with
# no `NAME=` in front of it (pasted into prose, inside a curl command, in tool output).
#
# Every example below is synthetic — an obviously fake shape, never a real key. `xxxx`
# stands for the long random tail each vendor appends.
_PATTERNS: list[tuple[str, re.Pattern]] = [
    # Tavily (a web-search API) keys: literal "tvly-" then 10+ letters/digits/_/-.
    #   "TAVILY_KEY=tvly-xxxxxxxxxxxx"  ->  "TAVILY_KEY=«REDACTED»"
    ("tavily-key", re.compile(r"tvly-[A-Za-z0-9_\-]{10,}")),
    # Context7 (a docs-lookup service) keys: literal "ctx7sk-" then 10+ of the same.
    #   "ctx7sk-00000000-aaaa-bbbb"  ->  "«REDACTED»"
    ("context7-key", re.compile(r"ctx7sk-[A-Za-z0-9_\-]{10,}")),
    # Anthropic API keys: "sk-ant-" at a word boundary then 20+ chars.
    #   "sk-ant-api03-xxxxxxxxxxxxxxxxxxxxxxxx"  ->  "«REDACTED»"
    ("anthropic-key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}")),
    # OpenAI API keys, classic and project-scoped: "sk-" or "sk-proj-" then 20+ chars.
    #   "sk-proj-xxxxxxxxxxxxxxxxxxxxxxxx"  ->  "«REDACTED»"
    # Listed AFTER the Anthropic rule only for readability; "sk-ant-…" matches both, and
    # either way the whole run is replaced by one mask, so the outcome is identical.
    ("openai-key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{20,}")),
    # GitHub fine-grained personal access token: "github_pat_" then 22+ chars.
    #   "github_pat_" + a long run of letters/digits/underscores  ->  "«REDACTED»"
    ("github-fine-pat", re.compile(r"github_pat_[A-Za-z0-9_]{22,}")),
    # GitHub classic tokens. The letter after "gh" is the token KIND — p=personal,
    # o=OAuth, u=user-to-server, s=server-to-server, r=refresh — hence the [pousr] class.
    #   "ghp_xxxxxxxxxxxxxxxxxxxx"  ->  "«REDACTED»"
    ("github-token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}")),
    # npm registry automation token: "npm_" then 30+ alphanumerics.
    #   "npm_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  ->  "«REDACTED»"
    ("npm-token", re.compile(r"\bnpm_[A-Za-z0-9]{30,}")),
    # Slack tokens of every flavour: "xox" + one letter (b/p/a/c/e/s…) + "-" + 10+ chars.
    #   "xoxb-0000000000-xxxxxxxx"  ->  "«REDACTED»"
    ("slack-token", re.compile(r"xox[a-z]-[A-Za-z0-9-]{10,}")),
    # Google API keys always start with the literal "AIza", then 30+ chars.
    #   "AIzaxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  ->  "«REDACTED»"
    ("google-key", re.compile(r"AIza[0-9A-Za-z_\-]{30,}")),
    # AWS access key id: "AKIA" plus exactly 16 upper-case/digit characters (fixed
    # length, unlike most of the others).
    #   "AKIA" + 16 upper-case letters/digits  ->  "«REDACTED»"
    ("aws-key", re.compile(r"AKIA[0-9A-Z]{16}")),
    # Stripe uses underscores (sk_live_...), so the hyphenated sk- patterns miss it.
    # r/p/s = restricted / publishable / secret key; live or test mode.
    #   "STRIPE_KEY=sk_live_xxxxxxxxxx"  ->  "STRIPE_KEY=«REDACTED»"
    ("stripe-key", re.compile(r"\b[rps]k_(?:live|test)_[A-Za-z0-9]{10,}")),
    # Stripe webhook signing secret: "whsec_" then 10+ chars.
    #   "whsec_xxxxxxxxxxxxxx"  ->  "«REDACTED»"
    ("stripe-webhook-secret", re.compile(r"\bwhsec_[A-Za-z0-9]{10,}")),
    # Unnamed credentials that show up in tool output (curl lines, git URLs, env dumps).
    # A Slack incoming-webhook URL is itself the credential — anyone holding the URL can
    # post — so the path after /services/ is masked, host and all.
    #   "https://hooks.slack.com/services/T000/B000/xxxxxxxxxxxxxxxxxxxxxxxx"
    ("slack-webhook", re.compile(r"hooks\.slack\.com/services/[A-Za-z0-9/_\-]{20,}")),
    # Hugging Face access token (used to download models): "hf_" then 30+ chars.
    #   "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  ->  "«REDACTED»"
    ("huggingface-token", re.compile(r"\bhf_[A-Za-z0-9]{30,}")),
    # GitLab personal access token: "glpat-" then 20+ chars.
    #   "glpat-" + 20 letters/digits  ->  "«REDACTED»"
    ("gitlab-pat", re.compile(r"\bglpat-[A-Za-z0-9_\-]{20,}")),
    # PyPI upload token: "pypi-" then 32+ chars (they are long base64-ish blobs).
    #   "pypi-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  ->  "«REDACTED»"
    ("pypi-token", re.compile(r"\bpypi-[A-Za-z0-9_\-]{32,}")),
    # Authorization header with any (or no) scheme — raw opaque tokens included.
    # Group 1 keeps `Authorization: Bearer ` (or `"authorization":"`, or no scheme at
    # all); group 2 is the credential and is the only part masked. (?i) makes the whole
    # pattern case-insensitive, since headers are written every possible way.
    #   'Authorization: Bearer xxxxxxxx'  ->  'Authorization: Bearer «REDACTED»'
    #   'Authorization: rawOpaque123456'  ->  'Authorization: «REDACTED»'
    ("assigned-auth-header", re.compile(
        r"(?i)(\bauthorization\b['\"]?\s*[:=]\s*['\"]?(?:(?:basic|bearer|token)\s+)?)"
        r"([A-Za-z0-9._~+/=\-]{8,})")),
    # A bare "Bearer <20+ chars>" with no Authorization word in front (a curl example, a
    # log line). This label does NOT start with "assigned", so the whole match including
    # the word Bearer is replaced.
    #   "send Bearer xxxxxxxxxxxxxxxxxxxx"  ->  "send «REDACTED»"
    ("bearer", re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._\-]{20,}")),
    # user:password@host in connection strings / authenticated git remotes.
    # Group 1 is `://user:` and is kept; group 2 is the password. The trailing `(?=@)` is
    # a lookahead — it requires the "@" without consuming it, so the rest of the URL is
    # left intact and still readable.
    #   "postgres://admin:xxxxxxxx@db.internal/x"  ->  "postgres://admin:«REDACTED»@db.internal/x"
    ("assigned-url-credentials", re.compile(r"(://[^/\s:@'\"]{1,64}:)([^/\s:@'\"]{3,256})(?=@)")),
    # JSON Web Token: three dot-separated base64url chunks. Real JWTs begin "eyJ" because
    # that is base64 for '{"' — the start of the JSON header — which makes them easy to
    # spot without decoding.
    #   "eyJ<header>.eyJ<payload>.<signature>"  ->  "«REDACTED»"
    ("jwt", re.compile(r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}")),
    # A whole PEM private-key block, header line through footer line. `re.S` (DOTALL)
    # lets `.` match newlines so the body in between is swallowed; `.*?` is non-greedy so
    # two keys in one file become two masks, not one giant one.
    #   the whole PEM block, from its BEGIN ... KEY line to its END line  ->  "«REDACTED»"
    ("private-key-block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S)),
    # KEY=VALUE / KEY: value / "key": "value" (JSON) assignments for secret-ish names.
    # The optional quotes around the name and before the value make JSON forms match.
    # [_-]key (not bare "key") so ENCRYPTION_KEY/SIGNING_KEY/APP_KEY match without
    # masking every dict-literal `key=` in code discussions.
    # The name must contain one of: secret / api_key / api-key / apikey / token /
    # password / passwd / access_key / <something>_key / credential(s). The value is 6+
    # characters with no space or quote in it, so a short placeholder like `key=x` and a
    # sentence after a colon are both left alone.
    #   'MY_SECRET="xxxxxxxxxxxxxxxx"'  ->  'MY_SECRET="«REDACTED»'
    #   '{"api_key": "xxxxxxxxxxxx"}'   ->  '{"api_key": "«REDACTED»}'
    # (the closing quote falls outside group 2 — harmless, the secret is gone)
    ("assigned-secret", re.compile(
        r"(?i)(['\"]?\b[A-Z0-9_]*(?:secret|api[_-]?key|token|password|passwd|access[_-]?key|[_-]key|credentials?)[A-Z0-9_]*\b['\"]?"
        r"\s*[:=]\s*['\"]?)([^\s'\"]{6,})")),
    # Long hex runs, but only in a value position (after = : or an opening quote).
    # A bare hash in prose/git-log output (commit SHAs, digests) is left alone so the
    # full-text index stays searchable by hash.
    #   "commit 3031ee38…  fixed it"  ->  unchanged (a 40-char SHA in prose)
    #   "KEY='3031ee38…'"             ->  "KEY='«REDACTED»'"
    # 32 is the shortest hash length worth worrying about (an MD5 digest); the trailing
    # \b stops a longer alphanumeric word from being clipped in the middle.
    ("assigned-hex", re.compile(r"([=:]\s*['\"]?|['\"])([0-9a-fA-F]{32,})\b")),
]

# The replacement text. French quotation marks («») are used deliberately: they almost
# never occur in code or transcripts, so counting them is a reliable way to count
# redactions (see redact_count) and a masked value is unmistakable when read back.
_MASK = "«REDACTED»"


def redact(text: str) -> str:
    """Return `text` with every recognised secret shape replaced by «REDACTED».

    Applies all patterns in order, each to the output of the previous one. "assigned"
    patterns keep group 1 (the name and separator) and drop group 2 (the value); the rest
    replace their whole match. Empty input (or None-ish) is returned unchanged.

    Pure and idempotent: redacting already-redacted text is a no-op, because «REDACTED»
    matches none of the patterns. Never raises — worst case it masks something harmless.
    """
    if not text:
        return text
    out = text
    for label, pat in _PATTERNS:
        if label.startswith("assigned"):
            # Keep the name/separator prefix so the line still reads as an assignment.
            out = pat.sub(lambda m: f"{m.group(1)}{_MASK}", out)
        else:
            out = pat.sub(_MASK, out)
    return out


def redact_obj(obj):
    """Recursively redact every string in a JSON-like structure (dict keys included —
    LLM-generated topic keys are content too). Non-string leaves pass through.

    Used on enrichment facets (the JSON summary a model wrote about a session) and on MCP
    tool results before they are handed back, where the payload is nested dicts and lists
    rather than one flat blob. Numbers, booleans and None are returned as-is, so counts
    and flags survive; the structure itself is never changed, only string leaves and keys.
    """
    if isinstance(obj, str):
        return redact(obj)
    if isinstance(obj, dict):
        return {(redact(k) if isinstance(k, str) else k): redact_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact_obj(v) for v in obj]
    return obj


def redact_count(text: str) -> int:
    """How many secrets get masked (exact: diff of mask occurrences).

    Counting masks in the output alone would over-report on text that already contained
    the literal «REDACTED» (a transcript discussing this very tool, or a re-export of an
    earlier export), so the pre-existing occurrences are subtracted. Used for the UI's
    "N secrets masked" badge on Copy Context / Export — the user should know something
    was removed from what they are about to paste elsewhere.
    """
    if not text:
        return 0
    return redact(text).count(_MASK) - text.count(_MASK)
