# Full Analysis: sci_fi_heist__c05349c6 (300-turn soak)

Written: 2026-10-13 (consolidated from parent + 3 subagent reports)

## Run Overview

- **Score:** 4.0/5 (final overview)
- **Turns:** 300/300 complete, all with non-empty replies
- **Total reply chars:** 432,587 (mean 1,441/turn)
- **Distinct scenes (current_scene):** 114 (2.6 turns/scene)
- **Judged turns:** 99/300 (mean 4.47, 83% at 4.5+)
- **Wall time:** ~39h wall clock (18.2h active turn time, rest = resume gaps + cloud retries)

## Critical Systemic Findings

### 1. arcs.json is EMPTY for all 300 turns (and all prior runs)

`arcs.json` is `{}` (2 bytes) for every single turn in this run, and in both
prior greek_mythology runs. The DSS never created a single arc across 585 total
turns. This is NOT a resume artifact — it's systemic.

**Impact:** The judge's `memory_fidelity` score is capped at ~2.0 for 36 turns
because it sees "empty memory store." This drags the quality_score down
(quality = f(style, memory_fidelity, notes)). Turns 89 (2.5), 119 (3.5), 199
(3.0) are all dragged down by this.

**Root cause:** Need to trace why `dss_close_arc` / arc creation is never
triggered. The DSS tool executor is initialized (log line 15) but no arc
creation calls appear in the log.

### 2. events.json scene tracking is 16x behind current_scene.json

- `current_scene.json` tracks 114 distinct scenes (scene_number 1-114)
- `events.json` only has 7 named scenes (plus a header entry)
- `events.json` stops growing at ~turn 150 (8 entries) and never updates again

The DSS is creating granular scene tracking in `current_scene.json` but NOT
aggregating them into `events.json` scenes/chapters/arcs. The higher-level
narrative structure (what the judge and overview see) is severely underpopulated.

### 3. Judge "parroting" criticism is a false positive

Turn 89 (score 2.5, the lowest) is flagged for "parroting and re-quoting
phrases directly from the instruction block." Actual 5-gram overlap: **1.9%**.
Only 1 of 3 flagged phrases is actually in the instruction. The other two are
original prose the judge misattributed.

Systematic check across all 99 judged turns:
- Mean overlap: 1.1%, max: 7.2%, p95: 3.9%
- Only 4 turns have the judge mentioning "copy/parrot/mirror"
- None have overlap > 4%
- **Verdict:** The transcription failure mode (78% in Qwen) does NOT exist in
  Gemma. The judge is pattern-matching on shared vocabulary (technical terms
  that appear in both instruction and reply) and calling it "parroting."

### 4. n02 "missed" 11x is a prop-continuity false positive

Note n02: "Dex's wrist-reader has a scuffed corner where the serial number is
nearly worn off." The judge marks this "missed" on 11 turns. In 10 of those 11
turns, the wrist-reader IS mentioned — just without the serial number detail.
The judge is charging the model for not re-stating a specific sub-detail of a
prop that IS present in the scene.

This is the same false positive class as the greek run's "borrowed sword"
criticism. The two-list fix (in-beat vs past) should help, but the underlying
issue is that the judge treats "prop mentioned without its sub-detail" as
"missed" when it should be "echoed (partial)."

### 5. Score compression persists

83% of judged turns at 4.5+ (vs 52% in the original Qwen greek run). The floor
is 2.5 (turn 89, a judge artifact). The overview identifies real weaknesses
("omission of specific physical character plant details") but the per-turn
scores don't reflect them.

## Timing Summary

| Category | Count | Mean | Notes |
|----------|-------|------|-------|
| Writer (reply) | 300 | 8.8s | Very consistent, p95=10.1s |
| DSS/summarizer | 6,647 | 18.0s | **Dominant cost**, 22 calls/turn |
| Cloud guide | 315 | 17.2s | 13 failures (all guide) |
| Cloud judge | 99 | ~5s | |
| Cloud auditor | 514 | 0.0s median | |

- Turn 55 outlier: 58min (cloud retry storm, 13 failed guide calls)
- Writer input tokens: 23,801 mean (context size hitting the model)
- DSS input tokens: 24,575 mean (similar context, 22x more calls)
- **The DSS is 22x the writer's call count and 2x the per-call time** — it's
  the dominant compute cost by a wide margin

### 6. ROOT CAUSE: StoryEvents schema missing new_entry templates (systemic, all runs)

The `soak_optimization_plan.md` (line 520) documents a fix: "Fixed by adding
generic `new_entry_query_prompt_template` + `new_entry_prompt_template` to
StoryEvents defaults. Verified hermetically." 

**This fix was never actually applied to the schema files.** Both
`user_data/example/subjects_schema.json` and
`user_data/example/schemas/subjects_schema_sceneagg.json` are missing both
templates on `StoryEvents`. The run's snapshot schema confirms it:
`StoryEvents.new_entry_query_prompt_template: False`.

This is the root cause of the entire chain:
1. `StoryEvents` missing templates → `_detect_and_add_new_entries_to_branch`
   skips → scenes never archived into `events.json` (stays at 7 initial entries)
2. 0 archived scenes → chapter gate finds 0 scenes → always skips ("Chapter
   has 0 scenes (min suggested: 4). Skipping chapter check.")
3. 0 chapters → arc gate finds 0 chapters → always skips
4. Empty `arcs.json` → judge sees "empty memory store" → low `memory_fidelity`
5. Low `memory_fidelity` drags down `quality_score` on 36 turns

**Impact on this run's score:** 36 turns have `memory_fidelity` ≤ 2.0 purely
because the DSS never populated the structural state the judge checks. The
4.0/5.0 final score is inflated by the judge not being able to see the
story's structural history (no chapters, no arcs, frozen scenes).

**Fix:** Add `new_entry_query_prompt_template` and `new_entry_prompt_template`
to the `StoryEvents` definition in both schema files. The plan doc describes
what they should do; the templates need to be written.

## Subagent Reports (merged)

- [x] Timing deep-dive → `scifi_heist_c05349c6_timing.md`
- [x] Context/state evolution → `scifi_heist_c05349c6_context.md`
- [x] Prose quality + flagged turns → `scifi_heist_c05349c6_prose.md`

## Timing Highlights (from timing report)

- 300/300 turns in 1091.5 min (18.2h active)
- Writer: 300 calls, 8.8s mean, very consistent (p95=10.1s)
- DSS: 6,647 calls, 18.0s mean, high variance (p95=60.8s, max=120.3s)
- Turn 55 outlier: 58min (13 failed cloud guide calls, retry storm)
- DSS input tokens: 24,575 mean (same context as writer, 22x more calls)
- Sum of local call durations (121,972s) exceeds wall time (65,487s) by 86% — the `total` field includes non-GPU overhead

## Context/State Highlights (from context report)

- 114 distinct scenes, 2.6 turns/scene, perfectly monotonic
- Character roster static: 2 chars (Sable, Dex), no entry/exit
- Entity graph: elements front-loaded (120 by turn 100, flat after), arcs=0, events=4 (static)
- Rolling summaries: 37 files, 3KB → 50KB (15x growth), the main growth vector
- Entire run stays in Chapter 1, Arc 1 — scene counter is the only progression axis

## Prose Highlights (from prose report)

- Reply length: 1017–2140 chars, median 1418, no degradation over 299 turns
- 5-gram overlap: all 20 sampled turns < 7% (no transcription)
- Voice: consistent first-person present-tense, no person/tense shifts
- Turn 89 (score 2.5): judge artifact — 1.1% overlap, not verbatim copying
- 125/132 notes never present in structured state → memory_fidelity=1.0 → infrastructure issue, not prose defect

## Consolidated Assessment

### What's working well
1. **Prose quality is strong.** No transcription, consistent voice, stable output length over 300 turns. Gemma-4-26B-StyleTune is a solid creative writer.
2. **Scene tracking is robust.** 114 scenes, monotonic, no anomalies.
3. **Entity graph tracks the story.** 120 elements, 1 group, properly populated.

### What's broken (systemic, affects all runs)
1. **StoryEvents schema missing new_entry templates** → scenes never archived into events.json → chapters never close → arcs never create → judge sees empty structural state → 36 turns get memory_fidelity≤2.0 → quality scores dragged down. **This is the #1 fix.**
2. **DSS is the dominant compute cost.** 6,647 calls (22/turn) at 18s mean = ~33h of DSS time vs ~44min of writer time. The DSS is doing 22 LLM calls per turn to maintain state that the judge can't even see because of fix #1.
3. **Rolling summaries grow unbounded.** 3KB → 50KB over 300 turns. At 700 turns this will be ~120KB+ and will dominate the context budget.

### Judge artifacts (not model defects)
1. **Turn 89 "transcription"** — 1.1% overlap, judge pattern-matching on shared vocabulary
2. **n02 "missed" 11x** — prop present without sub-detail, same class as greek "borrowed sword"
3. **Score compression** — 83% at 4.5+ vs 52% in Qwen run, floor at 2.5 (a judge artifact)