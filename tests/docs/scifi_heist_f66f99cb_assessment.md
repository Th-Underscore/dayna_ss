# Full Assessment: sci_fi_heist__f66f99cb (150 turns, seed 514)

## Run Config

| Parameter | Value |
|-----------|-------|
| Turns | 150 |
| Seed | 514 |
| Writer model | Gemma-4-26B-A4B-StyleTune-V2 (local, Q5_K_M) |
| Cloud models | gemini-3.5-flash-lite (guide, judge, auditor) |
| do_instr | false (Option D) |
| enable_thinking | true (reasoning_effort: low) |
| Judge memory limits | 50k per entry, 200k total |
| Character attributes | Added to Sable + Dex in spreadsheet |

## 1. Structure (DSS State)

| Metric | Turn 0 | Turn 149 | Baseline (300t) |
|--------|--------|----------|-----------------|
| Characters | 2 | 4 | 2→4 |
| Elements | 21 | 120 | 25→123 |
| Groups | 0 | 4 | 0→6 |
| Beats | 0 | 19 | 3→47 |
| Event Scenes | 0 | 7 | 0→121 |
| Chapters | 0 | 1 | 0→2 |
| Arcs | 0 | 0 | 0 |
| State Size | 34KB | 310KB | 54→405KB |

**Observations:**
- Entity growth tracks the baseline closely (elements plateau at ~120 by turn 100, same as baseline's 123 by turn 300)
- **Event scenes are dramatically lower**: 7 in 150 turns vs 121 in 300 turns. The baseline averaged 2.5 turns/scene; this run averages ~21 turns/scene. The narrative is more continuous with fewer hard scene cuts.
- No entity drops, no scene resets, no empty state files detected
- Rolling overviews: 18 files, 2KB→26KB (×9.8 growth)

## 2. Prose Quality

| Metric | This Run | Baseline (300t) | Thinking-on B (60t) |
|--------|----------|-----------------|---------------------|
| Mean words | 208 | 267 | 243 |
| Median | 210 | ~267 | — |
| Range | 148–276 | 161–351 | — |
| P95 | 246 | ~351 | — |
| Proper noun overlap | **0.6** | ~3 | 0.7 |
| Voice (1st:3rd) | 272:15 (18:1) | 147:1 (~150:1) | — |

**Key findings:**
- **Replies are 22% shorter than baseline** (208 vs 267 words). Tighter range (148–276 vs 161–351). The prose is more compact/clipped.
- **Proper noun overlap is 0.6** — same as the thinking-on Option B run (0.7), far below the no-thinking baseline (~3). Thinking is NOT improving specificity.
- **Voice is solid first-person** (18:1 ratio). The 15 third-person instances are in dialogue, not narration.
- **Turn 150 has 0 words** — the final turn produced an empty reply (likely a timeout or generation failure at the very end).

## 3. Judge Scores

| Axis | Mean | Min | Max | % at ceiling (5.0) |
|------|------|-----|-----|-------------------|
| Style | 4.81 | 3.0 | 5.0 | 74% |
| Quality | 4.57 | 3.5 | 5.0 | 29% |
| Memory | **4.83** | 2.0 | 5.0 | 90% |
| Combined | **4.74** | — | — | — |

**vs baseline:** Memory jumped +0.57 (4.26→4.83) from the 50k/200k limit bump. Style/Quality dipped slightly (-0.12/-0.05).

**Flagged turns:**
- **Turn 84** (style=3.0, quality=3.5, memory=2.0): **Valid criticism.** Genuine continuity error — reply narrates a drainage pipe scene while stored state shows the crew already aboard the skimmer in flight.
- **Turn 104** (style=3.0, quality=3.5, memory=5.0): **Judge false positive.** Judge claims "third-person narration ('I lift my legs')" but "I" IS first-person. The prose is correct; the judge mislabelled it.

## 4. Timing

| Metric | This Run (150t) | Baseline (300t) | Per-turn Δ |
|--------|-----------------|-----------------|------------|
| Wall time | 11.5h | 23.4h | — |
| Median turn | 168s | 167s | 0% |
| P95 turn | 693s | 598s | +16% |
| Max turn | 1178s (turn 51) | 6537s (turn 187) | — |
| DSS calls | 3,785 | 7,631 | — |
| DSS time/turn | 571s | 527s | +8% |
| Mean prompt tok/turn | 553K | 596K | -7% |
| Cloud calls | 420 | — | — |

**Observations:**
- Median turn is identical to baseline (168s vs 167s) — thinking adds negligible per-turn latency without instr_prompt
- P95 is 16% higher (693s vs 598s) — the tail is slightly fatter
- DSS is still 98% of total LLM time (85,674s of 88,799s local)
- No catastrophic outliers (max 1178s vs baseline's 6537s hang)
- 47 outlier turns (>2× median) — consistent with DSS summarization bursts

## 5. Note Adherence

| Status | Count | Notes |
|--------|-------|-------|
| planted | 6 | n03, n18, n32, n47, n65, n108 |
| echoed | 2 | n01, n17 |
| superseded | 4 | n04, n16, n30, n63 |

50% of tracked notes were planted into the narrative. The 4 "superseded" notes are all attributed to other characters' POVs (not Dex's), which is expected for a single-POV run.

## 6. Summary Assessment

### What improved vs baseline
1. **Memory fidelity: 4.26 → 4.83** (+0.57). The 50k/200k judge limit bump is the clear driver. The judge now sees full state data and no longer penalizes for "truncation."
2. **No catastrophic outliers.** Max turn 1178s vs baseline's 6537s hang.
3. **Note adherence is measurable.** 6 planted + 2 echoed out of 12 tracked notes.

### What did not improve
1. **Proper noun overlap: still 0.6.** Thinking (Option D) does NOT improve specificity. The model writes generic action prose without referencing the specific names/locations in its context. This is the same finding as Option B (0.7).
2. **Replies are shorter (208 vs 267 words).** Thinking consumes output budget; the prose is more clipped.
3. **Scene count is very low (7 in 150t vs 121 in 300t).** The narrative is more continuous but has far fewer formal scene boundaries.

### What's neutral
- **Style/Quality:** Slightly down (-0.12/-0.05) but within sampling variance.
- **Voice:** Rock-solid first-person, same as baseline.
- **Timing:** Median turn identical; P95 slightly higher.

### Verdict

**Option D (no instr_prompt + thinking on reply) does not improve specificity.** The proper noun overlap (0.6) is identical to Option B (0.7) and both are far below the no-thinking baseline (~3). The thinking tokens are spent on planning/reasoning but do NOT translate into more specific references to the story world in the prose output.

The memory fidelity improvement is real and significant, but it's entirely from the judge limit bump (a measurement fix), not from thinking.

**The specificity problem is a writer model limitation, not a prompt architecture problem.** Gemma-4-26B-StyleTune-V2 simply does not weave in specific proper nouns from its context, regardless of whether it has reasoning enabled or an instruction prompt directing it to.

### Recommendations

1. **Keep the 50k/200k judge limits** — clear improvement, no downside.
2. **Keep character attributes in the spreadsheet** — enables the judge to score ability usage.
3. **Revert thinking for the writer** — it adds 8% DSS time, shortens replies by 22%, and does not improve specificity. The `custom_reply_parameters` should be set to `{}` or `enable_thinking: false`.
4. **The specificity gap is a model capability issue.** Options: (a) accept it as a model limitation, (b) try a different writer model with stronger instruction-following, (c) post-process to inject key names (violates the pure-emulator principle).