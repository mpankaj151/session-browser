# Report structures (defaults)

A personal override file (config `[reports].style_override`) may replace any of
this — its instructions win. These defaults define the three report modes.

## Writing rules (all modes)

- Past tense, accomplishment-framed, self-contained: a bullet must make sense
  months later without the transcript. "Built X that does Y" over "worked on X".
- Prefer the session's journal (accomplishments/decisions) over its summary
  when both exist; quote PR/issue links inline.
- Every non-trivial session in the window appears exactly once, under its
  project, with its date. Trivial sessions (< 2 turns) are counted in stats
  only.
- Unenriched sessions appear with their title/first message and the marker
  *(unsummarized)* — visible gaps beat silent ones.
- No tool mechanics: never mention prompts, transcripts, enrichment, or Claude
  itself in the prose. The subject is the work.

## weekly-summary — stand-up ready

```
# Week of <from> – <to>

**Done** (grouped by project, dated bullets)
**In progress / carried over** (from open_threads of the week's sessions)
**Decisions** (the week's key_decisions worth surfacing)
**Next week** (open_threads that look actionable)
**Stats** (one line: sessions, active days, per-source)
```

Keep it pasteable: no H1 fluff above the fold, bullets over paragraphs.

## monthly-summary — 1:1 prep

```
# <Month YYYY> — work summary

**Themes** (2-4 sentences: what the month was about)
**Per-project progress** (H3 per project: dated accomplishment bullets,
  notable decisions, links)
**Decisions worth remembering** (cross-project, with rationale)
**Open threads going into next month**
**Stats appendix** (sessions, active days, by_type, by_outcome, per-source)
```

## review-summary — performance review / self-assessment

```
# Work summary — <window label> (<from> – <to>)

**Executive summary** (3-5 sentences)
**Highlights** (5-8 cross-project bullets — the strongest, most impactful work)
**Per-project accomplishments** (H3 per project: every non-trivial session as
  a dated bullet from its journal; decisions where meaningful; links)
**Impact** (per project with real substance: scope, what changed
  before → after, who benefits)
**Themes for self-assessment** (3-4 narrative arcs that span projects —
  e.g. "raised reliability of X", "built reusable Y tooling")
**Skills & tools** (derived from topics and session types — technologies
  exercised, not a keyword dump)
**Stats appendix** (table: sessions, non-trivial, active days, by_type,
  by_outcome, by_source, notional cost)
```

Impact and Themes are the sections managers read — write them from the
accomplishments/decisions evidence, not adjectives.
