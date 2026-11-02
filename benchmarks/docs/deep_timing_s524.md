# s524 Deep Timing Analysis (vLLM NVFP4, 100 turns)

Run: `sci_fi_heist__640b555f` seed 524, vLLM NVFP4, sequential (`--max-subject-workers 1`)

## Headline numbers

| Metric | Value |
|--------|-------|
| Total LLM time | 32,559s (~9.0h) |
| DSS share | 96.1% (31,287s) |
| Writer share | 3.9% (1,271s) |
| Avg LLM/turn | 325.6s |
| Scene turns (43) | avg ~530s |
| Non-scene turns (57) | avg ~160s |

## The three cost centers

### 1. `perform_update` — 46% of all DSS time (14,379s)

The all-or-nothing full-JSON rewrite. Every time the traversal visits a leaf node (element description, character relationship, importance field, etc.), it calls the LLM to either confirm "no changes" or rewrite the entire entry JSON.

Per-subject breakdown:

| Subject | Calls | Time | Avg/call | Notes |
|---------|-------|------|----------|-------|
| elements | 667 | 6,001s | 9.0s | 32 entries × ~2 sub-branches each = ~96 leaf calls/scene turn |
| characters | 131 | 4,617s | 35.2s | Highest per-call cost — large JSON (bio, prefs, relationships, group_status, importance) |
| groups | 127 | 1,701s | 13.4s | |
| events | 78 | 1,098s | 14.1s | |

**Characters are the worst offender per call** (35s avg) because each character entry is a large JSON blob. A single `perform_update` on a character rewrites the entire entry including biography, preferences, all relationships, group status, and importance.

### 2. Elements recursion — 27% of all DSS time (8,422s)

On scene turns, elements account for 42% of the turn's LLM time. The pattern on turn 5 (worst scene turn, 1,585s):

- **78 element DSS calls** for **32 unique element entries**
- Two distinct waves:
  - **Wave 1** (37 calls, 01:20–01:24): ElementMap `query_branch` + per-element `perform_update` on the main entry
  - **Wave 2** (41 calls, 02:10–02:18): Per-element sub-branch `perform_update` (relationships × N characters + importance)
- 46-minute gap between waves (other subjects processing in between)

The recursion is: `ElementMap` → 32 `Element` entries → each has `relationships` (one per character, ~2-3 each) + `importance`. That's ~96 leaf nodes per scene turn, each getting its own LLM call with a ~10K token prompt.

**This is the primary target for the scene-turn optimization.** Batching sub-branch updates (relationships + importance) into the parent element's `perform_update` call would eliminate ~40 calls per scene turn.

### 3. `query_branch` — 17% of all DSS time (5,289s)

The "what changed?" check before deciding whether to update. These are the gate checks that determine if an entry needs attention.

| Subject | Calls | Time | Avg/call |
|---------|-------|------|----------|
| events | 146 | 2,096s | 14.4s |
| elements | 76 | 1,103s | 14.5s |
| groups | 82 | 1,106s | 13.5s |
| characters | 69 | 984s | 14.3s |

These are relatively uniform across subjects. The prompt is ~16K tokens (larger than `perform_update`'s ~14K) because it includes the full branch context.

## Non-scene turn breakdown (57 turns, avg 160s)

| Component | Time | % | What it is |
|-----------|------|---|------------|
| Setup (instr gen + writer) | 1,549s | 55.5% | Instruction prompt generation (with collision regeneration), writer reply, DSS context prep |
| current_scene | 757s | 27.1% | Scene tracking updates |
| arcs | 429s | 15.4% | Arc `query_branch` + `perform_update` |
| events | 35s | 1.2% | Minimal |
| elements + groups + chars | 18s | 0.6% | Nearly zero — trigger reorg working |

**Key insight: the trigger reorg is working.** Non-scene turns now only touch `current_scene`, `arcs`, and minimal `events`/`elements`/`groups`. The 55% "setup" is the instruction generation + writer pipeline, which is not a DSS optimization target.

The `arcs` subject is unexpectedly expensive on non-scene turns (15.4%, 429s). With 114 calls across 57 turns, that's 2 calls/turn × 7.5s avg. This is the `query_branch` + conditional `perform_update` on arcs.

## Scene turn breakdown (43 turns, avg 530s)

| Subject | Calls | Time | % | Avg/call |
|---------|-------|------|---|----------|
| elements | 915 | 8,413s | 29.5% | 9.2s |
| events | 401 | 5,855s | 20.5% | 14.6s |
| characters | 194 | 5,607s | 19.7% | 28.9s |
| groups | 239 | 3,281s | 11.5% | 13.7s |
| current_scene | 351 | 1,895s | 6.6% | 5.4s |
| general_info | 90 | 1,622s | 5.7% | 18.0s |
| setup | 157 | 1,428s | 5.0% | 9.1s |
| arcs | 124 | 397s | 1.4% | 3.2s |

## Prefill vs decode (all DSS calls)

| Metric | Value |
|--------|-------|
| Total prefill (ttft) | 15,616s (50.3%) |
| Total decode | 15,370s (49.7%) |
| Avg in-tok/call | 13,435 |
| Avg out-tok/call | 0 (logging bug) |

Prefill and decode are nearly equal. With vLLM's ~4,400 tok/s prefill and ~36 tok/s decode, a typical 14K-token call with ~200 output tokens costs ~3.2s prefill + ~5.6s decode = ~8.8s total.

## Optimization targets (ranked by impact)

1. **Batch element sub-branch updates** (relationships + importance into parent `perform_update`): eliminates ~40 calls/scene turn, ~10-12s each = **~400-480s per scene turn** (40% of scene turn time). This is the elements recursion.

2. **Reduce `perform_update` frequency on characters**: 35s/call × 194 calls = 5,607s. If half the calls could be skipped (no actual changes), that's **~2,800s saved** across the run.

3. **Batch event sub-branch updates** (participants + importance into parent): similar pattern to elements, ~15 calls/scene turn eliminable = **~200-250s per scene turn**.

4. **Reduce `query_branch` prompt size**: 16K avg tokens is larger than needed. If the branch context could be trimmed to ~10K, prefill drops by ~37% on these calls = **~1,900s saved** across the run.

5. **Non-scene: reduce arcs calls**: 2 calls/turn × 7.5s = 15s/turn. If `query_branch` could be gated more aggressively (e.g., only on scene turns), that's **~430s saved** across 57 non-scene turns.