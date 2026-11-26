# DSS Performance Optimization Plan

**Supersedes:** `prefix_caching_optimization.md` (APC analysis preserved in §2)
**Reference:** `importance_decay_design.md` (relevance scoring mechanism)
**Baseline runs:** s573 (83 turns, 2,059 DSS calls), s574 (66 turns, 2,242 DSS calls)

---

## 1. Performance Model (Measured)

### Calibration Data

```
Model: gemma-4-26b-a4b | prefix-cache(relative)
   500 target (  517 in,   517 new,     0 cached)  ttft= 0.24s  prefill= 2159 tok/s  decode= 35.0 tok/s
  2000 target ( 1954 in,  1460 new,   494 cached)  ttft= 0.63s  prefill= 2317 tok/s  decode= 34.9 tok/s
  5000 target ( 4822 in,  2891 new,  1931 cached)  ttft= 2.00s  prefill= 1448 tok/s  decode= 34.8 tok/s
 10000 target ( 9553 in,  4754 new,  4799 cached)  ttft= 3.83s  prefill= 1241 tok/s  decode= 34.3 tok/s
 20000 target (19072 in,  9542 new,  9530 cached)  ttft=10.82s  prefill=  882 tok/s  decode= 32.4 tok/s
 30000 target (28591 in,  9542 new, 19049 cached)  ttft=14.46s  prefill=  660 tok/s  decode= 30.9 tok/s
 45000 target (42841 in, 14273 new, 28568 cached)  ttft=32.92s  prefill=  434 tok/s  decode= 28.8 tok/s
```

### Key Findings

1. **Prefill degrades 5x** as new-token count grows (2159→434 tok/s). This is the dominant cost.
2. **Decode is stable** (35→29 tok/s, 18% drop). Output length is a secondary factor.
3. **APC makes the cached prefix free.** The `cached` column is the portion handled by prefix caching; only `new` tokens contribute to prefill cost.
4. **The uncached suffix is the cost.** Reducing new-token count is the primary optimization lever.

### Cost Decomposition

For a DSS call with a 20K-token prompt where 15K is cached:
- Prefill: 5K new tokens at ~880 tok/s ≈ 5.7s
- Decode: 150 output tokens at ~32 tok/s ≈ 4.7s
- Total: ~10.4s

Reducing the uncached suffix from 5K to 2K tokens:
- Prefill: 2K new tokens at ~1200 tok/s ≈ 1.7s
- Decode: unchanged ≈ 4.7s
- Total: ~6.4s (38% reduction)

### Optimization Levers (Ranked)

| Lever | Mechanism | Impact |
|-------|-----------|--------|
| **Reduce uncached suffix** | Relevance filtering, two-tier prompts, select_entries_to_update | Primary — directly reduces prefill |
| **Reduce call count** | Gate elimination, skip_query, gate add_new_query | Secondary — removes entire decode+prefill cycles |
| **Reduce output length** | Delta format, shorter responses | Tertiary — decode is stable, so impact is modest |

---

## 2. Background: APC Analysis (from prefix_caching_optimization.md)

vLLM APC operates at block level (block_size=16). It hashes the first N tokens of each prompt and reuses existing KV cache blocks. The cached prefix is free; only the divergent suffix is prefilled.

**Template ordering** (section 2 of the old doc): moving turn-static content before per-entry variable content extends the LCP. This was the original optimization target and is partially addressed by the current template structure.

**Dynamic-suffix optimizations** (section 7 of the old doc): relevance-filter `_context_restatement`, enable `skip_query`, gate `add_new_query`. These remain valid and are carried into §5 below.

The old doc's TTFT-based cache-hit estimates are superseded by the calibration data above. The core insight (reduce the uncached suffix) is the same, but the magnitude is now measurable.

---

## 3. The Growth Problem

### Observed Pattern (s573/s574)

Per-subject prompt size grows linearly with story length:

| Subject | Turn 1 | Turn 30 | Turn 60 | Growth |
|---------|--------|---------|---------|--------|
| events.beats | 14K | 35K | 57K | 4x |
| events.scenes | 10K | 28K | 45K | 4.5x |
| chapters (apply_updates) | 8K | 25K | 40K | 5x |

The root cause: as the story accumulates entities, more of them pass the relevance filter (threshold 75), and the full branch data is sent to the LLM on every scene turn. The filter is working correctly — the relevance set is just growing because the story world is growing.

### Per-Chapter Recursion (s574 Turn 62)

Turn 62 took 50 minutes. 42 per-chapter calls (14 chapters × 3 calls each) consumed 47.7% of the turn (1,438s):

| Call type | n | Avg TTFT | Total | Avg input |
|-----------|---|----------|-------|----------|
| gate_check | 14 | 34.4s | 482s | 33,622 |
| query_changes | 14 | 10.1s | 141s | 39,260 |
| apply_updates | 14 | 45.5s | 815s | 40,300 |

Each chapter costs ~102s. The gate_check calls are 100% NO in this run (482s wasted). The query_changes calls are pure overhead (141s wasted). Together: 623s (21% of the turn) eliminable by gate elimination + skip_query.

The remaining 815s of apply_updates is real work, but at 40K input tokens each, it's hitting the worst part of the prefill curve (~434 tok/s at 14K new tokens).

### Chapter Over-creation (s574 vs s573)

s573 had 3 chapter entries; s574 had 14. Expected count is ~6. The chapter gate check is saying YES too often in s574, creating chapters that don't correspond to genuine story boundaries. This compounds the recursion problem: 14 chapters × 3 calls = 42 calls instead of the expected ~18. The gate check prompt (signal-detection framing) needs validation — it may be too loose for chapters specifically.

### Why s574 Got Fewer Turns Than s573

s573: 83 turns in 517 min local. s574: 66 turns in 534 min local. The 17-min difference doesn't account for 17 fewer turns. The answer: s574's per-turn cost grew faster because the story accumulated more entities (chapters, beats, scenes) before hitting the time limit. The linear growth in prompt size means each successive turn is more expensive, so the run's average per-turn time increases as it progresses. s573 happened to have a slower entity accumulation rate (different seed/story trajectory).

---

## 4. select_entries_to_update: Structural Fix

### Problem

The DSS iterates ALL entries in a branch on every scene turn. For `events.beats` with 30+ beats, every beat gets a gate_check + query_changes + apply_updates cycle. Most beats don't need updating on a given turn, but the LLM has to process all of them to determine which ones do.

### Fix: Select-Then-Update Pattern

Replace the per-entry recursion with a two-phase approach:

**Phase 1: Select** (cheap, ~2-3s)
```
Given the current scene context and the list of existing beats:
"Which of these beats may require updates based on what happened in this scene?"
→ Returns: ["Beat A", "Beat C"]
```

**Phase 2: Update only selected** (bounded)
```
Update these specific beats: [Beat A, Beat C]
```

The select call is cheap because it only needs names + one-line descriptions (not full JSON). The update calls are bounded because only 2-3 entries are sent, not 30+.

### Application Targets

| Branch | Current per-turn cost | With select-then-update |
|--------|----------------------|------------------------|
| events.beats | N_beats × 102s | 1 select (~3s) + N_selected × 30s |
| events.scenes | N_scenes × 80s | 1 select (~3s) + N_selected × 25s |
| events.chapters | N_chapters × 102s | 1 select (~3s) + N_selected × 30s |
| events.offscreen | N_offscreen × 60s | 1 select (~3s) + N_selected × 20s |

For a typical scene turn where 1-2 entries need updating out of 20-40, this reduces the per-branch cost by 80-95%.

### Implementation Notes

- The `select_entries` action already exists and is proven fast (0.5% of total time, TTFT 0.95s, decode 0.86s).
- The select prompt needs a roster of existing entries (name + one-line desc) — see §5 for the two-tier structure.
- The select prompt should err on the side of including ("err on the side of naming events over omitting them") to avoid missing updates.
- Schema wiring: `select_entries_to_update` already exists as an action; `select_entries` is wired into it. The change is in what the prompt asks for (which entries need updating) and how the results gate the downstream `apply_updates` call.

---

## 5. Two-Tier Prompt Structure

### Design

For branches with accumulating entries (beats, scenes, chapters, offscreen), structure the prompt as:

```
### Events already in context
**Event1**: One-line description
**Event2**: One-line description
**Event3**: One-line description
...
(~30 tokens per entry, flat list)

### Possibly relevant events (full data)
**EventA**: Full JSON with all properties
**EventB**: Full JSON with all properties
...
(bounded by relevance threshold)

Which of these events may require updates? Err on the side of naming events over omitting them.
```

### How It Works

- **Tier 1 (roster):** All entries in the branch, name + one-line description. Uses the **broad threshold** (42) or lists everything. Cost: ~30 tokens per entry. 50 entries = 750 tokens. This gives the LLM full awareness of what exists without exponential growth.

- **Tier 2 (full data):** Only entries passing the **final threshold** (75). Full JSON. Bounded by the threshold — as the story grows, more entries pass, but the select-then-update pattern (§4) means only the selected ones get full-data updates.

### Token Budget Comparison

| Approach | 20 entries | 50 entries | 100 entries |
|----------|-----------|-----------|-------------|
| Current (all full data) | 40K | 100K | 200K |
| Two-tier (roster + top-5 full) | 15K | 18K | 23K |
| Two-tier + select (roster + 2-3 full) | 8K | 10K | 12K |

The two-tier structure converts unbounded linear growth into bounded growth.

### Relevance Threshold for Tier 2

The final_threshold (75) determines which entries get full data in Tier 2. This is the knob to tune — see §6.

---

## 6. Relevance Threshold Tuning

### Current State

Thresholds are set but never validated against actual run data:
- `final_threshold: 75` (L7) — entries below this are excluded from `relevant_entities`
- `broad_threshold: 42` (L4) — Pass 1 discovery threshold
- `re_traverse_threshold: 65` (L6) — entities at 65+ become new seeds for Pass 3

The `re_traverse` pass is the main growth vector: every entity scoring 65+ becomes a seed, pulling in its neighbors. As the story accumulates, the 65+ set grows, and so does the 75+ set.

### What to Measure

From a soak run, log the relevance score distribution per entity type:
- How many entities score in each band: 42-54, 54-65, 65-75, 75-85, 85-100
- How many entries in each branch pass the final_threshold (75)
- How the distribution shifts over the course of the run (turn 1 vs turn 30 vs turn 60)

This tells us:
1. Whether 75 is too loose (too many entries passing) or too tight (missing relevant entries)
2. Whether the re_traverse threshold (65) is the main growth driver
3. What threshold would keep the Tier 2 set at a reasonable size (3-5 entries)

### Tuning Strategy

- If too many entries pass 75: raise the threshold (e.g., 80 or 85) or disable re_traverse
- If too few entries pass: lower the threshold (e.g., 70) or enable a looser Tier 2 for the select prompt
- The two-tier structure (§5) makes threshold tuning less critical — even a loose threshold only affects the Tier 2 full-data set, not the Tier 1 roster

---

## 7. Score-Graded Relevance Filter

### Current Limitation

`_get_relevance_names` (prompts.py:84) is a **binary gate**: it takes `set(rel[lower].keys())` from `relevant_entities`, discarding all score values. An entity at relevance 75 (barely passed) is treated identically to one at 99.

### Fix

The scoring pipeline (`_unified_entity_aggregation`) produces graded scores (float values 0-100). The consumption side throws them away. A score-graded filter would:

- **Tier 1 (roster):** All entities at `broad_threshold` (42)+ — name + desc only
- **Tier 2 (full data):** Entities at `final_threshold` (75)+ — full JSON
- **Tier 2+ (priority):** Entities at 85+ — full JSON, listed first

This uses the existing float values — no new mechanism, just stop discarding the scores.

### Implementation

In `_get_relevance_names`, return the score dict instead of just the keys. In the prompt builder, partition entries by score band and render each tier accordingly.

---

## 8. Carried-From prefix_caching_optimization.md §7

These optimizations remain valid and are independent of the new sections above:

### 8.1 Enable `skip_query` (DEFERRED — template rewording required first)

`query_changes` pre-query runs 56×/run × 2.66s = ~149s/run. The mechanism exists (`skip_query` in branch defaults, core.py:344). Setting it eliminates the pre-query.

**Risk (s577 data):** 94% of query calls return YES (51/54), confirming the pre-query is pure overhead. However, 8 of 11 update templates contain an active "fill gaps" instruction ("Also check for fields that are empty, contain 'N/A', or appear incomplete. If the story context... contain information that should fill those gaps, include those as updates too.") that biases the LLM toward producing updates. With the query pre-filter in place, this bias was harmless — the query confirmed something needed attention first. Without it, the instruction fires every turn and pushes the LLM away from `NO_UPDATES_REQUIRED`.

**Affected templates (8/11):** Character, Group, SceneState, StoryEvents, Scenes, OffscreenEvents, Element, GeneralInfo. Not affected: Knowledge (different structure), Arc, Chapter (no fill-gaps instruction).

**Prerequisite:** Reword the "fill gaps" instruction in all 8 templates from active ("check for and fill gaps") to passive ("if the current exchange directly changes a field's value, include it") before enabling `skip_query`. Without this, the update call becomes the sole gate and the biased framing increases spurious updates.

**Status:** Deferred. Minor optimization (~149s/run) that introduces new risk without the template rewording.

### 8.2 Gate `add_new_query` with entity graph presence

`add_new_query` runs 83×/run × 7.24s = ~600s/run. Most return NO/[]. Gate: skip if the subject type has zero presence in `relevant_entities`.

### 8.3 Relevance-filter `_context_restatement`

`_context_restatement` (prompts.py:112-162) renders the full subject map with no filtering. Thread `rel_names` through; filter both the full-map path and the fallback roster path. (Superseded by the two-tier structure in §5 — the roster IS the filtered restatement.)

---

## 9. Implementation Order

| # | Change | Status | Notes |
|---|--------|--------|-------|
| 1 | Gate `add_new_query` with relevance presence | ⬜ pending | |
| 2 | Two-tier prompt structure (roster + full data) | ⬜ pending | |
| 3 | `select_entries_to_update` for accumulating branches | ✅ done | events `apply_updates` 78→5 (s573→s585) |
| 4 | Score-graded relevance filter | ⬜ pending | |
| 5 | Relevance threshold tuning (data-driven) | ⬜ pending | |
| 6 | Gate elimination (8 grounded gate checks) | ⬜ pending | |
| 7 | Enable `skip_query` (after template rewording) | ⬜ pending | deferred — see §8.1 |

**Additional completed work (not in original plan):**
- Cadence tuning: `suggested_min_scenes` 4→6 (schema)
- Validation consolidation: shared `_generate_validated_entry` helper across all data-write paths
- Underscore structural fields: schema-driven, LLM-invisible, always restored from programmatic template
- Off-by-one fix: `_ending_*` now inclusive (`- 1` in template)
- `span_context`: archive refine prompts get actual scene/chapter data for the span
- Spurious list-index skip in `_resolve_path_keys` (parsing.py)
- Memory fidelity: removed `_DEMOTE` demotion of `partial`/`stale` verdicts

**Combined target:** 50-60% reduction in DSS time (from ~28Ks to ~11-14Ks for a 100-turn run).

---

## 10. Programmatic Override for Archive Templates

**Motivation:** The LLM refinement step in `_archive_boundary_entry` can silently corrupt structural fields (e.g. setting `ending_scene` to `null`), which cascades into `last_boundary` being stuck at 0, unbounded `span_count` growth, and gate bypass. Even with the `or 0` template fix, the LLM can still null out `ending_scene` on subsequent archives.

**Proposal:** Add an optional `archive_override` key alongside `archive_template` in schema defaults. This is a pure-JSON template (same Jinja variables available) that is merged **on top of** the LLM's refined response, guaranteeing programmatic values for structural fields.

Example for chapters:
```json
"archive_override": {
  "starting_scene": "{{ start }}",
  "ending_scene": "{{ current_scene._scene_number }}",
  "scenes": "{{ start | range_list(current_scene._scene_number) | tojson }}",
  "status": "complete"
}
```

The override values are rendered first, then merged into the LLM's response (override wins on key conflicts). This lets the LLM freely name the entry and fill narrative fields while structural integrity is guaranteed.

**Status:** Superseded. The underscore-prefixing approach achieves the same goal more broadly: structural fields are invisible to the LLM in both the prompt and the update path, and `_archive_boundary_entry` always restores them from the programmatic template output (not just when nulled). No separate `archive_override` mechanism needed.