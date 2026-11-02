# s536 (sci_fi_heist__adff125b) — Full Analysis

**Run:** seed 536, 174 turns (stopped at 174/300 for maintenance), 12h wall time
**Model:** gemma-4-26b-a4b (NVFP4, vLLM 1Cat, TP=2)
**Judge:** gemini-3.6-flash (tightened rubric)
**Tested:** Full perf stack integration + rules harvest + WorldRule + _history + judge variance

---

## 1. Timing

### Per-turn DSS time
| Category | Turns | Mean | Median | P95 | Max |
|----------|-------|------|--------|-----|-----|
| Non-scene | 114 | 48.5s | 45.5s | 72.3s | 91.7s |
| Scene | 61 | 503.8s | 510.7s | 783.2s | 837.7s |

**DSS fraction of wall time: 83.9%** (36,260s of 43,200s)

### Input token growth (median per subject, by turn range)
| Subject | T0-19 | T20-49 | T50-99 | T100-149 | T150-173 | Ratio |
|---------|-------|--------|--------|----------|----------|-------|
| characters.entries | 11,544 | 15,366 | 17,472 | 19,907 | 24,535 | **2.13x** |
| elements.entries | 8,148 | 15,426 | 16,978 | 18,898 | 23,261 | **2.85x** |
| groups.entries | 8,762 | 16,057 | 17,791 | 20,299 | 25,673 | **2.93x** |
| events.past | 7,147 | 9,691 | 10,828 | 14,338 | 19,204 | **2.69x** |
| events.scenes | 7,191 | 9,893 | 10,756 | - | - | 1.50x |
| events.beats | 7,168 | 9,362 | 19,258 | - | - | 2.69x |
| current_scene | 7,061 | 9,296 | 10,917 | 13,618 | 18,565 | **2.63x** |
| general_info | 9,793 | 13,682 | 15,243 | 18,184 | 24,903 | **2.54x** |
| chapters | 5,766 | 7,212 | 8,685 | 11,650 | 17,148 | **2.97x** |
| arcs | - | 7,788 | 8,578 | 11,382 | 16,881 | **2.17x** |

### Prefix cache (TTFT-based classification)
- Hits (<3s): **50%** (2,021 calls)
- Partial (3-8s): 27% (1,075 calls)
- Misses (>8s): 23% (930 calls)

### 10 slowest turns
| Turn | Time | Calls | Slowest call |
|------|------|-------|-------------|
| 171 | 837.7s | 57 | events.past/apply_updates 57.7s (in=53,332) |
| 162 | 807.9s | 63 | events.past/apply_updates 69.1s (in=49,330) |
| 156 | 791.0s | 64 | events.past/apply_updates 61.7s (in=46,097) |
| 76 | 783.2s | 52 | events.past/apply_updates 99.5s (in=24,080) |
| 165 | 752.3s | 50 | events.past/apply_updates 95.6s (in=51,152) |
| 174 | 745.0s | 51 | groups.entries/add_new 37.3s (in=21,352) |
| 61 | 735.4s | 49 | events.beats/apply_updates 74.9s (in=20,115) |
| 168 | 710.2s | 44 | events.past/apply_updates 70.1s (in=52,201) |
| 132 | 668.8s | 57 | events.past/apply_updates 72.6s (in=38,249) |
| 25 | 664.7s | 50 | events.beats/apply_updates 75.6s (in=15,381) |

**Pattern:** `events.past/apply_updates` dominates the slowest turns in the second half of the run, with input tokens reaching 53K.

---

## 2. Context / State Evolution

### Entity counts (every 10 turns)
| Turn | Chars | Elems | Groups | Scenes | Beats | Chaps | Past | Arcs | Rules |
|------|-------|-------|--------|--------|-------|-------|------|------|-------|
| 0 | 2 | 9 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| 20 | 4 | 34 | 7 | 12 | 9 | 1 | 12 | 0 | 0 |
| 50 | 6 | 52 | 7 | 24 | 20 | 2 | 20 | 0 | 0 |
| 100 | 11 | 79 | 11 | 37 | 35 | 3 | 42 | 0 | 0 |
| 150 | 20 | 112 | 14 | 37 | 35 | 3 | 68 | 0 | 0 |
| 173 | 24 | 125 | 14 | 37 | 35 | 3 | 76 | 0 | 0 |

### Description/Status split — **WORKING PERFECTLY**
| Turn | Sable desc | Sable status | Dex desc | Dex status | Ijo desc | Ijo status |
|------|-----------|-------------|----------|-----------|----------|-----------|
| 0 | 1e/81c | 134c | 1e/89c | 62c | - | - |
| 20 | 1e/81c | 125c | 1e/161c | 240c | 1e/134c | 76c |
| 50 | 1e/81c | 189c | 1e/157c | 191c | 1e/134c | 76c |
| 100 | 1e/126c | 95c | 1e/157c | 154c | 1e/134c | 76c |
| 150 | 1e/126c | 310c | 1e/157c | 225c | 1e/134c | 76c |
| 173 | 1e/126c | 125c | 1e/157c | 215c | 1e/134c | 76c |

Description stays flat (1 entry, 81-161 chars). Status is a single string (62-310c range), replaced not appended.

### `_history` — **WORKING**
| Turn | Chars with _history | Max entries |
|------|-------------------|-------------|
| 0 | 0/2 | 0 |
| 10 | 2/3 | 13 |
| 20 | 2/4 | 22 |
| 50 | 3/6 | 45 |
| 100 | 6/11 | 87 |
| 150 | 12/20 | 122 |
| 173 | 16/24 | 133 |

### Rules harvest — **BROKEN**
- `rules` is a `list` (not `dict[str, WorldRule]`) at every turn
- Length is 0 at every turn (0→0→0→...→0)
- The schema migration to `dict[str, WorldRule]` did NOT take effect in this run
- 194 `general_info` calls were made (including `query_changes`), but rules were never populated

---

## 3. Prose Quality / Judge

### Score distribution (34 turns judged, 6 had 0.0 = parse failures)
| Metric | Mean | Range | Variance |
|--------|------|-------|----------|
| Style | 4.89 (excl. zeros) | 4.5-5.0 | Low — still clustered |
| Quality | 4.05 (excl. zeros) | 3.5-4.5 | **Real variance** |
| Memory | 3.54 (excl. zeros) | 2.5-5.0 | **Strong differentiation** |

### Zero-score turns (6): turns 25, 35, 75, 115, 140, 165
All have empty summary. Likely cloud call failures (the judge model endpoint returned empty/malformed).

### Relative comparison: **NOT PRESENT**
0/34 turns contain relative comparison language ("comparable to", "weaker than", "strongest turn", etc.). The relative comparison feature (E) from the rubric tightening is not being exercised by the judge.

### Memory fidelity < 4.0 (18/28 valid turns)
The judge consistently cites:
- "state tracking exhibits notable corruption"
- "memory fidelity suffers from stale/recycled imagery"
- "repetitive physical distress tropes from earlier beats"
- "relies heavily on recurring [specific imagery]"

This is the completeness/freshness check working — the judge is catching that the writer keeps recycling the same physical-state descriptions without updating to match current DSS state.

---

## 4. Temporary Checks Summary

| Check | Status | Detail |
|-------|--------|--------|
| Input token growth (chars ~1.5-1.8x) | ⚠️ PARTIAL | 2.13x — better than 3x pre-split, but not at target |
| Per-turn wall time (non-scene 15-40s) | ⚠️ PARTIAL | 48.5s mean — above 40s target |
| Scene turn 200-400s | ❌ OVER | 503.8s mean — significantly above target |
| Prefix cache 40-50% | ✅ PASS | 50% hit rate |
| No perform_update on arcs/chapters | ✅ PASS | 0 calls |
| Rules harvest (8-20+ WorldRule entries) | ❌ FAIL | 0 entries, field is still a list |
| _history present on updated entries | ✅ PASS | 16/24 chars, max 133 entries |
| Description stays 1-3 entries | ✅ PASS | All 1 entry, flat |
| Status stays single string | ✅ PASS | 62-310c, replaced not appended |
| Score variance (quality 3.0-3.5 range) | ✅ PASS | 3.5-4.5 range |
| Memory fidelity differentiation | ✅ PASS | 2.5-5.0 range |
| Relative comparison in judge notes | ❌ FAIL | 0/34 turns |

---

## 5. Key Findings & Next Steps

### Working well
1. **Description/status split** — perfectly flat, zero bloat
2. **_history** — recording correctly, growing as expected
3. **Prefix cache** — 50% hit rate, stable
4. **Path-patch conversion** — zero perform_update on arcs/chapters
5. **Judge variance** — quality and memory now differentiate meaningfully

### Broken / needs attention
1. **Rules harvest is completely non-functional** — the schema in the run's `state_snapshot/` still has `rules` as a `list`. The `dict[str, WorldRule]` migration + `rules_query_prompt_template` trigger did not take effect. Need to check: (a) was the schema file the run actually loaded the updated one? (b) is the trigger config correct?
2. **`events.past` is the dominant late-run cost** — 53K input tokens by turn 171, 99.5s TTFT on the worst call. This is the archival accumulation problem.
3. **Scene turns are 504s mean** — well above the 200-400s target. Driven by `events.past` + `events.beats` + high entity counts.
4. **No relative comparison in judge output** — the rubric change (E) isn't producing the expected behavior. The judge may not be receiving the window of prior turns, or the prompt language isn't strong enough.
5. **Character input tokens still growing 2.13x** — better than 3x but the entity accumulation (24 chars by T173) is the driver, not field bloat.

### Priority fix list
1. **Diagnose rules harvest failure** — check which schema file the run loaded, verify the trigger config is in the right place
2. **Cap `events.past` input** — 53K tokens is way over the 20K prefix-cache cliff. Need a top-N or recency window on past events
3. **Investigate scene turn cost** — 504s is 2x the target. The `events.past` + `events.beats` calls on scene turns are the likely culprits
4. **Fix relative comparison** — check if the judge prompt actually includes the window of prior turns and the comparison instruction