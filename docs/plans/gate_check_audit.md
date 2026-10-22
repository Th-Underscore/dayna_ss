# DSS Gate Check Audit

**Schema:** `user_data/example/schemas/subjects_schema_sceneagg.json`
**Code:** `agents/data_summarizer/updates.py`, `traversal.py`, `core.py`

## 1. Gate Check Mechanics (from code)

### `_perform_gate_check` (updates.py:69-181)

| Aspect | Behavior |
|--------|----------|
| Empty prompt | Returns `True` (fail-open: process the branch) |
| Stopping strings | `["NO", "UNCHANGED", "YES"]` with `match_prefix_only=True` |
| Negative verdict | `stop in ("NO","UNCHANGED")` OR text starts with `"NO"` OR text contains `"UNCHANGED"` |
| Affirmative verdict | `stop == "YES"` OR text starts with `"YES"` |
| Non-verdict (e.g. story continuation) | One bounded retry with strict re-frame, then **defaults to YES** |
| `runtime.stop_everything` | Returns `False` (skip branch) |
| Net effect | Fail-open: any ambiguous/failed LLM response → YES (process) |

### `_update_recursive` (traversal.py:68-159) — trigger walking

1. Collects `current_event_triggers` = `[ALWAYS]` + `[ON_NEW_SCENE | ON_EXISTING_SCENE]`
2. Walks `trigger_map` for each matching trigger, in schema order
3. **Skips** any config with `when: "gate_check_fail"` in the normal loop
4. On `gate_failed=True` from a `perform_gate_check` action:
   - Scans remaining configs for `when: "gate_check_fail"` and executes them
   - Returns immediately (stops the branch)
5. On `stop_processing=True` or `skip_children=True`: returns immediately
6. If all actions complete without stopping: drills into children via `_traverse_structure`

### `_execute_action` (core.py:240-380) — gate check path

For `Action.PERFORM_GATE_CHECK`:
- Resolves `gate_check_prompt_template` via `_get_effective_setting`
- If no template found: logs "Gate: No template" and **continues** (no gate, branch proceeds)
- If template found: calls `_perform_gate_check`
  - Returns `(False, True, False)` on gate FAIL (gate_failed=True)
  - Returns `(False, False, False)` on gate PASS (continue)

---

## 2. Subject × Trigger × Gate Table

| Subject | Map Class | Triggers (active) | Gate Template | Gate Fires On | Expected YES Rate | Optimization |
|---------|-----------|-------------------|---------------|---------------|-------------------|--------------|
| **characters** | `CharacterMap` | `on_new_scene`: [add_new, **gate_check**, importance_update*(fail), select_entries] | Generic "does section need review?" + scene_recap + scene_events | **New scene only** | **~90% YES** (new scene = new context) | **REPLACE with cheaper heuristic** — see §4 |
| **groups** | `GroupMap` | `on_new_scene`: [add_new, **gate_check**, select_entries] | Generic "does section need review?" + "Section scope: (all groups)" + scene_recap + scene_events | **New scene only** | **~50-70% YES** (not every scene involves groups) | **KEEP** — variable enough to be useful |
| **elements** | `ElementMap` | `always`: [add_new, **gate_check**] + `on_new_scene`: [select_entries] | Generic "does section need review?" + scene_recap + scene_events | **EVERY TURN** | **~10-20% YES** (most turns don't change elements) | **REDUCE to on_new_scene** — see §4 |
| **current_scene** | `CurrentScene` | `always`: [branch_query(now)] + `on_new_scene`: [branch_query(start)] | None (uses branch_query as implicit gate) | N/A | N/A | N/A |
| **events** | `Events` (top) | **NONE** (`triggers: {}`) | Defined but **DEAD** — no trigger calls perform_gate_check | **NEVER** | N/A | **REMOVE dead template** |
| **events.beats/scenes/past** | `StoryEvents` | `on_new_scene`: [add_new, branch_query(skip), importance_update*(dead)] | None in trigger list; `gate_check_fail` conditional is **DEAD** | **NEVER** | N/A | **REMOVE dead conditional** |
| **events.chapters** | `Chapters` | **NONE** (`triggers: {}`) | Defined but **DEAD** | **NEVER** | N/A | **REMOVE dead template** |
| **events.chapters (per-chapter)** | `Chapter` | `always`: [branch_query] + `on_new_scene`: [perform_update] | None | N/A | N/A | N/A |
| **general_info** | `GeneralInfo` | `always`: [branch_query(skip_query)] | Defined but **DEAD** — no perform_gate_check in triggers | **NEVER** | N/A | **REMOVE dead template** |
| **general_info.objectives** | `Objective` | `always`: [**gate_check**, branch_query, importance_update*(fail)] | "Has importance changed for any secondary objectives?" | **EVERY TURN** | **~10-20% YES** (importance rarely changes) | **REDUCE to on_new_scene** — see §4 |
| **arcs** | `Arcs` (top) | **NONE** (`triggers: {}`) | Defined but **DEAD** | **NEVER** | N/A | **REMOVE dead template** |
| **arcs (per-arc)** | `Arc` | `always`: [branch_query] + `on_new_scene`: [perform_update] | None | N/A | N/A | N/A |

*( = `when: "gate_check_fail"` conditional action

---

## 3. Code Path Findings

### 3.1 Dead Gate Templates (template defined, never called)

| Class | Template Key | Why Dead |
|-------|-------------|----------|
| `Events` | `gate_check_prompt_template` | `triggers: {}` — no trigger fires `perform_gate_check` |
| `Chapters` | `gate_check_prompt_template` | `triggers: {}` — no trigger fires `perform_gate_check` |
| `Arcs` | `gate_check_prompt_template` | `triggers: {}` — no trigger fires `perform_gate_check` |
| `GeneralInfo` | `gate_check_prompt_template` | Triggers only have `query_branch_for_changes`, no `perform_gate_check` |

**Impact:** Zero runtime cost (never rendered/sent), but schema bloat and maintenance confusion. A future schema edit that adds `perform_gate_check` to these classes' triggers would silently activate a stale template.

### 3.2 Dead `gate_check_fail` Conditional (StoryEvents)

```json
"on_new_scene": [
  "add_new",
  {"action": "query_branch_for_changes", "skip_query": true},
  {"action": "query_branch_for_changes",
   "prompt_template": "importance_update_prompt_template",
   "when": "gate_check_fail"}   ← DEAD: no perform_gate_check in this list
]
```

The third action can never execute because no preceding action in the same trigger list produces `gate_failed=True`. The `importance_update_prompt_template` LLM call is unreachable.

**Impact:** Zero runtime cost, but if someone adds `perform_gate_check` to StoryEvents' triggers expecting the importance fallback to work, it would silently activate.

### 3.3 Gate Fail Does NOT Prevent `add_new` (by design)

**ElementMap:** `always: [add_new, perform_gate_check]`

The `add_new` action executes BEFORE the gate check. If the gate then fails, the branch stops — but any entries added by `add_new` are already in `data`. This is intentional (per architecture memory #598: add_new is reordered before gate_check to prevent new entities from being swallowed by stale gates).

**Not a leakage bug.** The design intent: new entries always get created; the gate only gates *updates to existing entries*.

### 3.4 CharacterMap Gate Fail → Importance Update Still Runs

```json
"on_new_scene": [
  "add_new",
  "perform_gate_check",
  {"action": "query_branch_for_changes",
   "prompt_template": "importance_update_prompt_template",
   "when": "gate_check_fail"},   ← runs when gate says NO
  "select_entries_to_update"     ← runs when gate says YES
]
```

When the gate returns NO (section is up-to-date), the `importance_update_prompt_template` branch query still executes a full LLM call. This is a **deliberate bypass**: importance scores are updated even when content is stable.

**Assessment:** This is a feature, not a leakage. Importance drift is a separate axis from content change. However, it means a "NO" gate still costs one LLM call, reducing the gate's savings.

### 3.5 No Gate on GeneralInfo (always branch_query with skip_query)

```json
"always": [{"action": "query_branch_for_changes", "skip_query": true}]
```

`skip_query: true` means the "is there a change?" query is skipped and the update runs unconditionally. GeneralInfo gets a **full update LLM call every single turn** with no gate.

**Assessment:** GeneralInfo (writing_style, tone, themes, setting, rules, objectives, synopsis) rarely changes. An unconditional update every turn is wasteful. The `branch_query_prompt_template` IS defined (asks "are there significant changes?") but `skip_query: true` bypasses it.

### 3.6 No Gate on Chapter/Arc (always branch_query)

```json
// Chapter and Arc both:
"always": ["query_branch_for_changes"],
"on_new_scene": ["perform_update"]
```

The `query_branch_for_changes` without `skip_query` acts as an **implicit gate**: it asks "are there changes?" and if the answer is NO, returns `skip_children=True` (no further LLM calls for sub-fields). This is functionally equivalent to a gate check but uses the branch query mechanism.

**Assessment:** Acceptable. The branch query serves as a cheap gate. No change needed.

---

## 4. Optimization Recommendations

### 4.1 ElementMap: Reduce from `always` to `on_new_scene`

**Current:** Gate check runs EVERY turn. Expected YES rate: ~10-20%.
**Problem:** 80-90% of gate calls are wasted LLM round-trips that return NO.

**Recommendation:** Move `perform_gate_check` from `always` to `on_new_scene`:
```json
"triggers": {
  "always": ["add_new"],
  "on_new_scene": ["perform_gate_check", "select_entries_to_update"]
}
```

**Rationale:** New elements are introduced at scene boundaries. Between scenes, the element roster is stable. The `add_new` action (which handles new element detection) already runs every turn via `always` — it just won't be gated.

**Savings:** ~80% reduction in gate LLM calls for elements.

### 4.2 Objective: Reduce from `always` to `on_new_scene`

**Current:** Gate check runs EVERY turn. Expected YES rate: ~10-20%.
**Problem:** Objective importance rarely changes within a scene.

**Recommendation:** Move `perform_gate_check` to `on_new_scene`:
```json
"triggers": {
  "always": ["query_branch_for_changes"],
  "on_new_scene": ["perform_gate_check", {"action": "query_branch_for_changes", "prompt_template": "importance_update_prompt_template", "when": "gate_check_fail"}]
}
```

**Savings:** ~80% reduction in objective gate LLM calls.

### 4.3 CharacterMap: Replace LLM Gate with Cheaper Heuristic

**Current:** Gate check on every new scene. Expected YES rate: ~90%.
**Problem:** A new scene almost always involves character changes. The gate is a tax on a near-certain YES.

**Options (in order of preference):**

1. **Remove the gate entirely** for CharacterMap. Let `select_entries_to_update` do the filtering (it already asks "which entries need updating?" and can return an empty list). The gate is redundant with the selection step.

2. **Replace with a deterministic heuristic:** Skip the gate if the scene recap contains no character names (string match against the entry roster). Only call the LLM gate when at least one known character name appears in the scene context.

3. **Keep as-is** if measured data shows the 90% YES rate is overstated (some new scenes may be pure dialogue with no state changes).

**Savings (option 1):** 1 LLM call per new scene eliminated.

### 4.4 GeneralInfo: Add a Gate or Remove skip_query

**Current:** `always: [branch_query {skip_query: true}]` — unconditional update every turn.
**Problem:** Writing style, tone, themes, setting, rules rarely change. Synopsis and objectives change occasionally.

**Options:**

1. **Remove `skip_query: true`** so the branch query acts as a gate (asks "are there changes?" before updating). This is the cheapest fix — the `branch_query_prompt_template` is already defined.

2. **Add a `perform_gate_check`** with the existing (currently dead) `gate_check_prompt_template`, triggered on `on_new_scene` only.

**Savings (option 1):** Eliminates the update call on turns where nothing changed (estimated ~70-80% of turns).

### 4.5 Remove Dead Schema Artifacts

| Item | Action |
|------|--------|
| `Events.gate_check_prompt_template` | Remove (dead) |
| `Chapters.gate_check_prompt_template` | Remove (dead) |
| `Arcs.gate_check_prompt_template` | Remove (dead) |
| `GeneralInfo.gate_check_prompt_template` | Remove (dead) OR activate via §4.4 |
| `StoryEvents` `gate_check_fail` conditional | Remove (dead) |

**Rationale:** Dead templates are a maintenance hazard. If a future edit adds `perform_gate_check` to these classes' triggers, the stale template silently activates with potentially incorrect wording.

---

## 5. Summary: Gate Call Frequency Per Turn

| Turn Type | CharacterMap | GroupMap | ElementMap | Objective | GeneralInfo | Total Gate Calls |
|-----------|-------------|----------|------------|-----------|-------------|-----------------|
| **Existing scene** | 0 | 0 | **1** | **1** | 0 (unconditional update) | **2** |
| **New scene** | **1** | **1** | **1** | **1** | 0 (unconditional update) | **4** |

After recommended optimizations:

| Turn Type | CharacterMap | GroupMap | ElementMap | Objective | GeneralInfo | Total Gate Calls |
|-----------|-------------|----------|------------|-----------|-------------|-----------------|
| **Existing scene** | 0 | 0 | 0 | 0 | 0 (gated by branch_query) | **0** |
| **New scene** | 0 (removed) | **1** | **1** | **1** | 0 (gated by branch_query) | **3** |

**Net reduction:** 2 gate calls per existing-scene turn eliminated; 1 per new-scene turn eliminated.

---

## 6. Residual Risks

1. **CharacterMap gate removal (§4.3 option 1):** If `select_entries_to_update` returns an empty list, no per-entry updates run. This is correct behavior. But if the selection call itself is unreliable (returns names for entries that don't need updating), we lose the gate's "is this section even relevant?" filter. Mitigation: the selection prompt already says "an empty array means none need updating."

2. **ElementMap gate reduction (§4.1):** If a mid-scene exchange introduces a new element AND changes an existing one, the existing element won't get updated until the next scene boundary. Mitigation: `add_new` still runs every turn (just not gated), so new elements are caught. Existing element changes are rarer mid-scene.

3. **Objective gate reduction (§4.2):** If objective importance shifts mid-scene (e.g., a secondary objective becomes the main goal), it won't be updated until the next scene. Mitigation: `main_objective` is a separate field on GeneralInfo, updated by the GeneralInfo branch query.

4. **GeneralInfo skip_query removal (§4.4):** The branch query adds one LLM call (the query itself) before the update. On turns where changes ARE needed, total cost is the same (query + update). On turns where nothing changed, we save the update call. Net positive.