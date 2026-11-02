# DSS Optimization Report: sci_fi_heist__f42ea382

**Run:** 150 turns, seed 518, Gemma-4-26B-A4B-StyleTune-V2.i1-IQ4_XS
**Log:** `scifi_heist_gemma4_s518.log` (164 MB, 952K lines)
**Total LLM time:** 85,749s (23.8h local) + 9,687s (2.7h cloud) = 95,436s
**Total local calls:** 4,406 | **Total cloud calls:** 407
**Mean per-turn:** 29.4 local calls (median 21, max 144)

## 1. Call Time Budget

| Category | N | Total (s) | % of Total |
|----------|---|-----------|-----------|
| DSS (all subjects combined) | ~4,255 | ~83,390 | ~97.2% |
| Writer (reply/generate) | 151 | 2,359 | 2.8% |

The writer is trivial. DSS is the entire problem.

### DSS Per-Subject Breakdown

| Subject | N | Mean (s) | Total (s) | % of DSS |
|---------|---|----------|-----------|----------|
| *(uncategorized branch updates)* | 1,926 | 16.3 | 31,454 | 37.7% |
| element | 465 | 23.9 | 11,106 | 13.3% |
| *(gate checks)* | 404 | 26.8 | 10,836 | 12.6% |
| character | 158 | 36.3 | 5,741 | 6.9% |
| storyevents | 181 | 29.8 | 5,388 | 6.5% |
| scenecharacter | 184 | 20.5 | 3,777 | 4.5% |
| currentscene | 61 | 43.0 | 2,621 | 3.1% |
| generalinfo | 138 | 17.6 | 2,434 | 2.9% |
| importance | 171 | 12.3 | 2,104 | 2.5% |
| arcs | 93 | 21.5 | 2,001 | 2.4% |
| scene sub-fields (when/why/group/element) | 334 | ~10 | ~3,547 | 4.3% |
| other (arc, map, relationships, timepoint) | 108 | ~15 | ~1,543 | 1.9% |

> Note: The "uncategorized" 1,926 calls are branch update calls that the log categorizer
> couldn't attribute to a specific subject (the `Processing Schema Node` line was more than
> 5 lines before the call). They are the same class as the named DSS subjects.

## 2. Gate Check Analysis

**423 gate checks** consumed **10,836s (12.6% of DSS time)**.

| Subject | YES | NO | YES Rate |
|---------|-----|----|----------|
| elements.entries | 124 | 27 | 82.1% |
| characters.entries | 45 | 16 | 73.8% |
| general_info.objectives[0] | 34 | 116 | **22.7%** |
| groups.entries | 4 | 57 | **6.6%** |

### Findings

- **`groups.entries`** is 93.4% NO — 57 of 61 gate checks said "nothing changed." This is the single most wasteful gate. Groups barely change in a heist story; the gate is almost always correct to skip, but it still costs an LLM call (~27s each) to determine that.
- **`general_info.objectives[0]`** is 77.3% NO — 116 of 150. Objectives are set early and rarely change. Same pattern.
- **`elements.entries`** is 82% YES — the gate earns its cost here. Elements are discovered continuously.
- **`characters.entries`** is 74% YES — also earns its cost.

### Optimization Opportunity

For subjects with <25% YES rate (groups, objectives), the gate check is pure overhead. Options:
1. **Raise the gate threshold** — require stronger evidence to say YES (e.g. "YES only if a NEW group/objective is explicitly introduced in the exchange")
2. **Reduce gate frequency** — check these subjects every N turns instead of every turn
3. **Remove the gate** — let the branch update run and rely on NO_UPDATES to skip (saves the gate call, adds a cheap update call that returns NO_UPDATES)

## 3. Discovery Analysis

- **101 "Failed to extract new entry names"** — this is the fence-parsing bug (fixed in the latest commit). All 101 were for `elements.entries` discovery.
- **94 "already has 120 entries; skipping"** — the element entry cap is working correctly, preventing unbounded growth.
- **2 entity identification calls** — only 2 turns triggered entity identification (turn 0 initial population + one mid-run). Discovery is not a hot path.

Discovery is not a significant cost driver. The fence-parsing fix will eliminate the 101 spurious failures.

## 4. Branch Update Redundancy

**1,175 "Summarizing" calls** (branch updates), **229 returned NO_UPDATES** (19.5% no-ops).

### Generation Time Distribution (proxy for response size)

| Stat | Value |
|------|-------|
| N | 1,925 |
| Mean | 2.4s |
| Median | 0.5s |
| P25 | 0.1s |
| P75 | 3.6s |
| P95 | 8.8s |
| Max | 41.2s |

### Size Buckets

| Bucket | Count | % |
|--------|-------|---|
| <2s | 1,229 | 63.9% |
| 2-5s | 372 | 19.3% |
| 5-10s | 265 | 13.8% |
| 10-20s | 44 | 2.3% |
| 20-50s | 15 | 0.8% |
| >50s | 0 | 0% |

**83.2% of branch updates generate <5s of content.** These are trivial updates (a few words of change) that each cost a full LLM round-trip.

### Optimization Opportunity

The 1,229 calls with <2s generation time are the biggest waste. Each one costs ~10-15s of prefill (reading the prompt) to produce <2s of output. Options:
1. **Batch updates** — combine multiple small updates into one LLM call (e.g. update all scene sub-fields in one call instead of 4 separate calls)
2. **Skip trivial updates** — if the gate check says YES but the expected change is small, skip the update
3. **Reduce update frequency** — update less-frequently-changed subjects (groups, objectives, importance) every N turns

## 5. Per-Turn Call Distribution

| Stat | Value |
|------|-------|
| Turns | 150 |
| Mean calls/turn | 29.4 |
| Median | 21 |
| P95 | 57 |
| Max | 144 |

Turn 23 had 144 calls (the outlier). Turn 110 had 61 calls and was the slowest turn (6,888s wall time).

## 6. Ranked Optimization Opportunities

| # | Opportunity | Est. Savings | Complexity |
|---|-----------|-------------|------------|
| 1 | **Batch scene sub-field updates** (when/why/group/element are 4 separate calls per scene) | ~3,547s (4.3%) | Low |
| 2 | **Reduce/remove gate for groups + objectives** (93% and 77% NO) | ~6,000s (7%) | Low |
| 3 | **Skip trivial branch updates** (<2s generation = 64% of calls) | ~10,000s+ (12%+) | Medium |
| 4 | **Batch element updates** (465 calls, 13.3% of DSS) | ~5,000s (6%) | Medium |
| 5 | **Reduce importance update frequency** (171 calls, 2.5%) | ~1,000s (1.2%) | Low |
| 6 | **Fence-parsing fix** (already done) | Eliminates 101 spurious failures | Done |

**Combined potential:** ~20-25% reduction in DSS LLM time if all are implemented.

## 7. Cloud Call Summary

| Type | N | Mean (s) | Total (s) |
|------|---|----------|-----------|
| Guide (user turn) | 158 | 28.6 | 4,511 |
| Auditor | 168 | 23.1 | 3,886 |
| Judge | 51 | 14.0 | 714 |
| Replan | 30 | 19.3 | 577 |
| **Total** | **407** | | **~9,687** |

Cloud is 10% of total LLM time. Not the bottleneck.

## 8. Deliverables Index

| Doc | Location |
|-----|----------|
| Interim findings | `tests/docs/scifi_heist_f42ea382_interim.md` |
| Timing analysis | `tests/docs/scifi_heist_f42ea382_timing.md` |
| Context/state evolution | `tests/docs/scifi_heist_f42ea382_context.md` |
| Call profile | `docs/plans/scifi_heist_f42ea382_call_profile.md` |
| Update redundancy | `docs/plans/scifi_heist_f42ea382_update_redundancy.md` |
| Gate check audit | `docs/plans/gate_check_audit.md` |
| This report | `docs/plans/scifi_heist_f42ea382_optimization_report.md` |