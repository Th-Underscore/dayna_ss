# DSS Prefix-Caching & Prompt-Size Optimization Plan

**Target:** Reduce per-call prefill cost in sequential DSS (`--parallel 1`, `--max-subject-workers 1`) by maximizing LCP prefix reuse and minimizing the variable suffix.

**Baseline (s521, 50 turns):** 10,644 tok/call avg, 61% prefill, 17–73 calls/turn.

---

## 1. Current Prompt Anatomy

Every DSS LLM call is assembled from:

```
[system message]  system context (formatted story state)
[user message]    internal context (history)  +  per-call instruction
```

| Component | Turn 0 | Turn 48 | Within-turn variance |
|-----------|--------|---------|---------------------|
| System context | 1,314 tok | 4,683 tok | **0%** (static) |
| Internal context (history) | 3,589 tok | ~3,589 tok | **0%** (static) |
| Per-call instruction | 166–7,455 tok | same range | **100%** (varies per call) |

**Stable prefix = system + history ≈ 5,000–8,200 tok (75–80% of total).**
**Variable suffix = instruction ≈ 1,300–7,455 tok (20–25% of total).**

The LCP mechanism in llama.cpp matches the stable prefix and only prefills the variable suffix. This is already working — the 54–58s prefill for 150K input tokens on non-scene turns reflects effective ~2.7K tok/s (vs raw 1K), consistent with ~75% LCP match.

**Goal: reduce the variable suffix to <10% of total prompt.** That means the instruction needs to average <1,000 tokens (down from ~2,600).

---

## 2. Trigger Reorganization

### Problem

13–14 `always` triggers fire on every turn, generating 17–20 LLM calls on non-scene turns where most return NO/UNCHANGED. Each call costs ~10K tokens of prefill (mostly the stable prefix re-verified via LCP, but the variable suffix is still fresh).

### Current `always` triggers

| Subject | Triggers | Calls/turn | Non-scene usefulness |
|---------|----------|------------|---------------------|
| CurrentScene | `query_branch_for_changes` | 1 | **High** — the core work |
| SceneCharacter | `perform_update` | 1 | Medium — tracks who's in scene |
| SceneGroup | `perform_update` | 1 | Low — rarely changes mid-scene |
| SceneElement | `perform_update` | 1 | Low — rarely changes mid-scene |
| SceneWhen | `perform_update` | 1 | Low |
| SceneWhy | `add_new` + `perform_update` | 2 | Low |
| Arc | `query_branch_for_changes` | 1 | **None** — arcs don't change mid-scene |
| Chapter | `query_branch_for_changes` | 1 | **None** — chapters don't change mid-scene |
| Objective | `gate_check` + `query` + `importance` | 2–3 | **None** — objectives are stable |
| GeneralInfo | `query_branch_for_changes` | 1 | Low — occasionally updates |
| ElementMap | `add_new` + `gate_check` | 2 | **None** — new elements only appear in new scenes |

### Proposed reorganization

Move all non-`current_scene` subjects to `on_new_scene` only:

| Subject | Before | After |
|---------|--------|-------|
| CurrentScene | `always: [query]` | **unchanged** |
| SceneCharacter | `always: [perform_update]` | **unchanged** (tracks per-turn presence) |
| SceneGroup | `always: [perform_update]` | `on_new_scene: [perform_update]` |
| SceneElement | `always: [perform_update]` | `on_new_scene: [perform_update]` |
| SceneWhen | `always: [perform_update]` | `on_new_scene: [perform_update]` |
| SceneWhy | `always: [add_new, perform_update]` | `on_new_scene: [add_new, perform_update]` |
| Arc | `always: [query]` | `on_new_scene: [perform_update]` (drop query; see §6) |
| Chapter | `always: [query]` | `on_new_scene: [perform_update]` (drop query; see §6) |
| Objective | `always: [gate, query, importance]` | `on_new_scene: [gate, query, importance]` |
| GeneralInfo | `always: [query]` | `on_new_scene: [perform_update]` (drop query; see §6) |
| ElementMap | `always: [add_new, gate]` | `on_new_scene: [add_new, gate, select_entries]` |

**Non-scene turn call count: 17–20 → ~5–6** (CurrentScene query + SceneCharacter update + a few residual).

### `perform_update` concern (Arc, Chapter)

The user flagged that `perform_update` is all-or-nothing (no diff, full JSON rewrite) and will be lossy on large sets. Arc and Chapter will grow over the run. Mitigations:

- **Arc:** Small set (typically <10 arcs in a 150-turn run). `perform_update` is safe at this scale. The `query_branch_for_changes` gate was 93% NO — pure overhead. Replacing it with `perform_update` on scene transitions is a net win.
- **Chapter:** Grows to ~10–15 entries. `perform_update` rewrites the full chapter list each scene transition. At 15 entries this is a ~2K token JSON. Risk: LLM drops or mangles earlier chapters. **Mitigation:** use `query_branch_for_changes` as the trigger (not `perform_update`) — it asks "does this need updating?" and only fires `perform_update` on YES. This is the same pattern as Objective, which has been working. The gate adds one extra call on scene turns, but scene turns are already expensive.

**Revised Chapter/Arc/GeneralInfo:**
| Subject | After |
|---------|-------|
| Arc | `on_new_scene: [gate_check, perform_update]` |
| Chapter | `on_new_scene: [gate_check, perform_update]` |
| GeneralInfo | `on_new_scene: [gate_check, perform_update]` |

This adds 3 gate-check calls on scene turns (which are already 40–70 calls), but eliminates 3 calls on every non-scene turn (17–20 calls). Net win.

---

## 3. Prompt Structure for LCP Optimization

### The LCP mechanism

llama.cpp's LCP finds the longest common prefix between the new prompt and existing KV cache. With `--parallel 1`, all calls share one slot and one KV cache. Sequential calls with the same prefix → the prefix is already in the cache → only the suffix is prefilled.

**For LCP to work well, the prompt must be structured as `[long stable prefix] + [short variable suffix]`.**

### Current structure (already optimal in shape)

```
[system: story state]        ← stable within turn
[user: history + instruction] ← history stable, instruction varies
```

The LCP matches system + history, and only the instruction is fresh. This is already the correct shape.

### Can we extend the stable prefix?

The question: can any part of the per-call instruction be moved into the stable prefix?

**The user's rejected idea:** put JSON schemas in the general prompt. Rejected because it inflates the prefix for ALL calls, most of which don't need the schema. A 1,200-token schema × 17 calls = 20K extra tokens/turn of wasted prefix for the 2–3 calls that actually use it.

**Alternative approaches:**

#### A. Shorten the schema (not move it)

The Element schema is 4,623 chars (~1,156 tok). Most of that is field descriptions. A minimal schema (field names + types only, no descriptions) would be ~800–1,200 chars (~200–300 tok). The LLM doesn't need "a list of relationships to other entities, specifying the type and importance of each relationship" — it needs `relationships: list[Relationship]`.

**Impact:** 1,156 → ~300 tok per schema-gen call. With 3–4 schema-gen calls on scene turns, saves ~3K tokens/scene turn. Marginal on non-scene turns (no schema-gen calls).

#### B. Conditional system context

The importance_scale (3,039 chars, `no_prompt=True`) is in the system context for every call. It's only useful for importance-check calls (1–2 per turn). For the other 15–18 calls, it's dead weight in the prefix.

If we conditionally exclude it from non-importance calls:
- Prefix shrinks by ~800 tok for 15–18 calls
- LCP match rate stays the same (the remaining prefix is still stable)
- Fresh prefill per call drops by ~800 tok
- **Net saving: 15 × 800 = 12K tokens/turn** (on non-scene turns)

This is a real win. The implementation: the system context builder needs to know which call type is about to be made, and conditionally include/exclude the importance_scale.

**Risk:** if the call type classification is wrong, the LLM gets a prompt without the importance scale when it needs it. Mitigation: only exclude it for call types that demonstrably never reference importance (gate checks, scene updates, element discovery).

#### C. Stable instruction preamble

The per-call instruction starts with a type-specific header ("You are checking if branch X needs updates" vs "Generate the JSON for entity Y" vs "Discover new entities in this scene"). These headers are different for each call type, so they break the LCP at the very start of the instruction.

If we restructure so that ALL instructions share a common preamble (e.g., the current scene summary + the task category), and only the specific task details vary at the end, the LCP would match further into the instruction.

Example:
```
[stable prefix: system + history]
[semi-stable: "Current scene: Sable is in the crawlspace. The comms unit is dead." — same for all calls this turn]
[variable: "Task: Check if branch 'elements.entries' needs updates." — differs per call]
```

The "current scene" summary is already in the system context, so this might be redundant. The real question is whether the instruction's first 200–500 tokens are typically shared across calls of the same turn. If they're not (because each call has a completely different instruction), this doesn't help.

**Verdict:** Low impact. The instruction is already the variable suffix. Making a small part of it semi-stable would extend the LCP by 200–500 tokens, which is <5% of the total. Not worth the complexity.

### Summary of prompt-structure optimizations

| Optimization | Tokens saved/call | Calls affected | Complexity |
|-------------|------------------|----------------|------------|
| B. Conditional importance_scale | ~800 | 15–18/turn | Medium |
| A. Shorten JSON schema | ~800 | 3–4/turn (scene only) | Low |
| C. Stable instruction preamble | ~300 | all | High, low ROI |

---

## 4. Entity List Capping (Discovery Prompt)

### Problem

The entity discovery prompt includes the full list of all existing entity names (140 entries, 26,711 chars, ~6,700 tok) so the LLM can avoid creating duplicates. This grows linearly with the story.

### Capping strategy: lenient entity graph

The user's suggestion: use a "lenient version of the entity graph" — lower importance thresholds to select which entities to include in the discovery prompt.

The entity graph already computes per-node relevance from connection counts, importance, and bidirectional counts. A "lenient" mode would:

1. **Always include:** all characters (by name), all groups, the current scene's elements
2. **Include by relevance:** elements/events connected to the current scene's characters (1-hop neighbors in the entity graph)
3. **Cap:** if the result exceeds N entries (e.g., 50), truncate by importance score

This keeps the list bounded (~50 entries, ~2,000 tok) while covering the entities the LLM is most likely to accidentally duplicate. The LLM doesn't need to see all 140 entity names — it needs to see the ones that are **topically adjacent** to the current scene.

**Implementation:** the discovery prompt builder queries the entity graph with a lower importance threshold (e.g., 0.3 instead of 0.5) and a hop limit (1 hop from current scene participants). The result is capped at 50 entries.

**Risk:** the LLM might create a duplicate of an entity that's not in the capped list. Mitigation: the `add_new` path already has a dedup check (exact name match against the full list) that runs after the LLM response. The list in the prompt is a soft guide; the post-hoc check is the hard gate.

---

## 5. Batch Scene Sub-field Updates

### Problem

Scene sub-fields (when, why, group, element) are 4 separate LLM calls, each with a ~10K token prompt. They could be batched into 1 call.

### Approach

Merge the 4 sub-field update prompts into a single prompt that asks the LLM to update all 4 at once, returning a single JSON with all 4 fields. The schema for the combined response is the union of the 4 individual schemas.

**Savings:** 3 calls × ~10K tokens = 30K tokens saved per scene turn. At 1K tok/s prefill, that's ~30s per scene turn.

**Risk:** the combined prompt is larger (4× the individual instructions), and the LLM might struggle with a more complex output format. Mitigation: test with a smoke run.

---

## 6. Expected Impact Summary

| Optimization | Non-scene turn saving | Scene turn saving |
|-------------|----------------------|-------------------|
| Trigger reorg (§2) | 11–14 calls eliminated → ~100–140s | 0 (already scene-gated) |
| Conditional importance_scale (§3B) | ~12K tokens → ~12s prefill | ~15K tokens → ~15s |
| Shorten JSON schema (§3A) | 0 | ~3K tokens → ~3s |
| Entity list capping (§4) | 0 (no discovery on non-scene) | ~5K tokens → ~5s |
| Batch scene sub-fields (§5) | 0 | ~30s |
| **Total** | **~112–152s/turn (from ~90–110s)** | **~53s/turn (from ~200–300s)** |

Wait — the non-scene turn numbers don't add up. Let me recalculate.

Non-scene turn baseline: 17–20 calls, 86–108s LLM time.
- Trigger reorg: 17–20 → 5–6 calls. That's 11–14 calls eliminated. At ~5s/call (LCP-cached), that's 55–70s saved.
- Conditional importance_scale: 5 calls × 800 tok = 4K tokens → ~4s saved.
- **Total non-scene: ~59–74s saved. Remaining: ~12–34s (5–6 calls × ~2–6s each).**

Scene turn baseline: 40–70 calls, 180–336s LLM time.
- Conditional importance_scale: ~15K tokens → ~15s
- Shorten schema: ~3K tokens → ~3s
- Entity list capping: ~5K tokens → ~5s
- Batch sub-fields: ~30s
- **Total scene: ~53s saved. Remaining: ~127–283s.**

The scene turns are still dominated by the sheer number of calls (40–70). The trigger reorg doesn't help scene turns because all the triggers are already firing. The main lever for scene turns is reducing per-call cost (prefix caching is already doing this) and batching.

**Overall: non-scene turns go from ~90–110s to ~15–35s (65–75% reduction). Scene turns go from ~200–300s to ~150–250s (25–30% reduction).**

---

## 7. Implementation Order

1. **Trigger reorg** (§2) — schema JSON edit, no code changes. Test with smoke run.
2. **Shorten JSON schema** (§3A) — edit the schema definitions in `subjects_schema_sceneagg.json`. Test with smoke run.
3. **Entity list capping** (§4) — code change in the discovery prompt builder. Test with smoke run.
4. **Conditional importance_scale** (§3B) — code change in the system context builder. Test with smoke run.
5. **Batch scene sub-fields** (§5) — code change in the traversal logic. Test with smoke run.

Each step is independently testable. The trigger reorg is the highest-impact, lowest-risk change.