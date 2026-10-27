# DSS Prefix-Caching Optimization Plan (vLLM)

**Target:** Reduce per-call prefill cost in sequential DSS (`--max-subject-workers 1`) by maximizing vLLM's Automatic Prefix Caching (APC) hit rate.

**Baseline (s524, 100 turns, vLLM NVFP4):**
- 3,022 DSS calls, 31,287s total LLM time
- TTFT (prefill): 15,942s (51%)
- Decode: 15,345s (49%)
- Avg in-tok/call: 13,435
- Avg TTFT: 5.27s/call

---

## 1. The Problem: Template Ordering Defeats APC

### How vLLM APC works

vLLM's prefix caching operates at the **block level** (block_size=16 tokens). It hashes the first N tokens of each prompt and looks up existing KV cache blocks. If the prefix matches an existing entry, those blocks are reused and only the suffix is prefilled.

**Critical property:** APC matches on **exact prefix**. If token 5,000 differs between two prompts, everything from token 5,000 onward is a cache miss — even if tokens 5,001–10,000 are identical.

### Current template structure

Every per-entry DSS call (element update, character update, etc.) is assembled as:

```
[1] System context (story state, importance scale)     ~2-4K tok  STATIC within turn
[2] Internal history (recent DSS exchanges)             ~3.6K tok  STATIC within turn
[3] Per-call instruction:
    [3a] Schema (json-schema for this entry type)      ~500-1,200 tok  STATIC per entry type
    [3b] Example (sample JSON)                         ~200-500 tok   STATIC per entry type
    [3c] {{value}} (current entry JSON)                ~200-2,000 tok  VARIES per entry  ← BREAKS LCP HERE
    [3d] Boilerplate instructions ("update if changed") ~200-400 tok   STATIC
    [3e] {{scene_events}} (current scene context)      ~1-3.3K tok    STATIC within turn
```

**The LCP between any two calls in a turn is: [1] + [2] + [3a] + [3b] ≈ 5-8K tokens.**

After that, every call diverges at [3c] (the per-entry value). The shared [3e] (scene_events, 1-3.3K tokens) is AFTER the divergence point, so it's never cached.

### Measured impact (s524 data)

| Prompt size | Avg TTFT | Implied cache hit | Expected if scene_events were cached |
|---|---|---|---|
| 8-10K | 1.84s | ~59% | ~85% |
| 10-12K | 1.78s | ~68% | ~88% |
| 12-15K | 2.89s | ~57% | ~82% |
| 15-20K | 8.34s | ~5% | ~60% |

Scene turns (40-141 calls) show 28-55% hit rate. Non-scene turns (8-10 calls) show 54-66%. The difference is the number of distinct per-entry values breaking the prefix.

### Why it's NOT:
- **Batch size / concurrency:** `--max-subject-workers 1` means one request at a time. No competition.
- **KV cache size:** 240K tokens available, 141 sequential 10K-token calls = 1.4M tokens total. Even without reuse, the cache never overflows (only 10K is "live" at any time).
- **ISWA / model architecture:** The cache DOES work — retries with +150 tokens get 0.3s TTFT (near-full hit). The mechanism is fine; the prefix just isn't long enough.
- **Eviction:** With sequential requests and a 240K token cache, there's no eviction pressure.

---

## 2. Fix: Reorder Templates (Highest ROI)

### Principle

Move all **turn-static** content BEFORE the **per-entry variable** content. The shared `scene_events` block (1-3.3K tokens) is currently after the per-entry value; moving it before extends the LCP from ~5-8K to ~8-11K tokens.

### New template structure

```
[1] System context (story state, importance scale)     ~2-4K tok  STATIC
[2] Internal history (recent DSS exchanges)             ~3.6K tok  STATIC
[3] Per-call instruction:
    [3a] Schema (json-schema for this entry type)      ~500-1,200 tok  STATIC per type
    [3b] Example (sample JSON)                         ~200-500 tok   STATIC per type
    [3c] {{scene_events}} (current scene context)      ~1-3.3K tok    STATIC within turn  ← MOVED UP
    [3d] Boilerplate instructions ("update if changed") ~200-400 tok   STATIC
    [3e] {{value}} (current entry JSON)                ~200-2,000 tok  VARIES per entry  ← NOW LAST
```

**New LCP between any two calls: [1] + [2] + [3a] + [3b] + [3c] + [3d] ≈ 8-11K tokens.**
**Variable suffix: [3e] only ≈ 200-2,000 tokens.**

### Expected impact

| Scenario | Before (LCP ~6K) | After (LCP ~10K) |
|---|---|---|
| 10K call | 6K cached, 4K fresh → 2.0s TTFT | 10K cached, 0K fresh → 0.3s TTFT |
| 12K call | 6K cached, 6K fresh → 3.0s TTFT | 10K cached, 2K fresh → 1.0s TTFT |
| 15K call | 6K cached, 9K fresh → 4.5s TTFT | 10K cached, 5K fresh → 2.5s TTFT |

**Conservative estimate (accounting for block alignment, hash collisions):**
- Current: 37% full-miss calls cost 12,198s TTFT
- After reorder: most calls become "near hit" (only the 200-2K value suffix is fresh)
- **Expected TTFT reduction: 60-75% on DSS calls**
- **Total DSS time saving: ~8,000-10,000s (25-32% of total LLM time)**

### Implementation

The templates are in `format_templates.json` (git-ignored, generated from `subjects_schema_sceneagg.json`). The reorder is a template edit: move the `{{scene_events}}` (or equivalent context block) from after the per-entry value to before it.

**Caveat:** This changes the prompt the LLM sees. The model is being asked to "update this entry given the scene context" — moving the context before the entry is semantically neutral (the model reads the whole prompt before generating). But it should be validated with a smoke run to confirm no quality regression.

---

## 3. Secondary Optimizations

### 3a. Slim the DSS prompt (reduce cold-call cost)

Even with perfect caching, the FIRST call in a turn (or after a cache reset) pays the full prefill. Reducing the total prompt size helps these cold calls and reduces the KV cache footprint.

- **Drop writer-oriented context from patch calls:** The current_scene narrative, the writer's last reply, and other RP-specific context are not needed for a "does this element need updating?" call. Estimate: ~2-3K tokens removable.
- **Shorten the importance scale:** 3,039 chars (~800 tok) is in every call. Only needed for importance-specific calls. Conditionally include it.
- **Target: 10-12K avg in-tok → 7-9K.**

### 3b. Reduce retry cost

234 retry calls cost 2,370s (7.6% of DSS time). Causes:
- "Parent path does not exist" (405 occurrences): the LLM returns a path that doesn't exist in the schema. Fix: auto-create the parent container, or validate the path before the LLM call.
- Unparseable output (110 occurrences): the LLM returns malformed JSON. Fix: lower temperature for DSS calls (currently 1.0).

**Expected saving: 1,500-2,000s (5-6% of DSS time).**

### 3c. Wire `select_entries_to_update` to StoryEvents

170 whole-map patch calls (~19K tokens in) for events could be reduced by using the select mechanism to pick which events to update, rather than sending the full map.

**Expected saving: 1,500-2,500s (5-8% of DSS time).**

### 3d. Lower temperature for DSS calls

The harness sends temperature=1.0, top_p=0.95, top_k=64 to ALL local calls. DSS calls (structured JSON extraction) benefit from lower temperature (0.3-0.5). This should reduce:
- Invalid path returns (fewer hallucinated paths)
- Malformed JSON (tighter token selection)
- **Free test, no quality risk for structured extraction.**

### 3e. Character `description` → `description` + `status` split

517 of 576 applied character updates target `description`. By turn 99, Dex's description is 10.9K chars of near-paraphrase "she now maintains position behind..." lines. This makes the prompt enormous AND the output large (31s decode/call).

Split into:
- `description`: stable physical/personality description (set once, rarely changes)
- `status`: current state (overwritten each update, short)

**Expected saving: 2,500-3,000s (8-10% of DSS time) from reduced prompt + output size.**

---

## 4. What We're NOT Doing (and why)

| Idea | Why not |
|---|---|
| Reduce `--max-num-batched-tokens` to 512 | One request at a time; batch size is irrelevant for sequential processing |
| Reduce `--max-num-seqs` | Already sequential (1 request); no effect |
| Increase KV cache size | 240K tokens is more than enough; the issue is prefix matching, not capacity |
| Switch to llama.cpp LCP | vLLM APC is the correct mechanism; the issue is template ordering, not the caching system |
| Reduce call count (batch sub-branches) | The traversal already skips 90% of nodes; the patch call IS the field selector |
| Deterministic name-matching for select | The LLM selector has 69% precision; name-matching only reaches 77-88% recall. The select call is 1% of time. Not worth the complexity. |

---

## 5. Validation Plan

Before committing to the template reorder, run a controlled A/B test:

1. **Baseline:** current template order, 30-turn run, measure TTFT distribution
2. **Reordered:** `scene_events` before `{{value}}`, same 30 turns, same seed
3. **Compare:** TTFT p50/p90, full-miss rate, total DSS time, AND quality (judge score, memory adherence)

The quality check is critical: if the model's update quality degrades when the context is before the entry (rather than after), the reorder is a no-go and we fall back to prompt slimming + temperature reduction.

**For the next full run:** enable vLLM's `--log-prefix-cache-stats` (if available) or add per-request `cached_tokens` logging to get ground-truth cache hit data instead of inferring from TTFT.

---

## 6. Implementation Order

| # | Change | Type | Risk | Expected saving |
|---|---|---|---|---|
| 1 | Template reorder (scene_events before value) | Template edit | Low (validate with smoke) | 8,000-10,000s (25-32%) |
| 2 | Lower DSS temperature to 0.3 | Config | None | 1,500-2,000s (5-6%) |
| 3 | Slim prompt (drop writer context from patch calls) | Code | Low | 3,000-5,000s (10-16%) |
| 4 | Character description/status split | Schema + prompt | Medium (quality) | 2,500-3,000s (8-10%) |
| 5 | Wire select to StoryEvents | Code | Low | 1,500-2,500s (5-8%) |

Items 1-3 are independent and can be applied together. Item 4 needs a separate quality validation. Item 5 is a nice-to-have.

**Combined realistic target: 40-55% reduction in DSS time** (from 31,287s to ~14,000-19,000s for a 100-turn run).