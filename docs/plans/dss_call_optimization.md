# DSS Call Optimization Plan

**Goal:** Answer "what LLM calls were unnecessary?" by profiling the per-turn call
distribution in a completed soak run and mapping each category back to the code path
that produces it.

## Per-Turn DSS Call Flow (from code)

Each turn, `_process_all_subjects` iterates over every subject (characters, elements,
groups, events, current_scene, general_info, arcs). For each subject, `generate()` →
`_update_recursive()` walks the schema's trigger map:

| Trigger | Code path | LLM calls | Purpose |
|---------|-----------|-----------|---------|
| `add_new` | `_detect_and_add_new_entries_to_branch` | 1 | Discover new entities to create |
| `perform_gate_check` | `_perform_gate_check` | 1 (+1 retry) | Decide if the branch needs updating |
| `branch_query` | `_perform_full_branch_update` | 1 | Generate the actual field updates |

After gate check passes, `_traverse_structure` drills into sub-branches, each of which
may have its own triggers (e.g. `characters.entries.Sable.relationships` could have a
gate check). After all subjects, `_run_boundary_checks` may trigger chapter/arc archival
(1 LLM call each) and canon digest regeneration (1 LLM call).

**Worst case per turn:** 7 subjects × 3 triggers = 21 calls + boundary + digest = ~23.
**Actual (from 300-turn baseline):** ~25 calls/turn, 7,631 over 300 turns.

## Analysis Axes

### Axis 1: Gate Check Efficiency
**Question:** What fraction of gate checks returned NO/UNCHANGED (wasted call)?

**Log signals:**
- `Gate check for 'X' returned YES` → useful
- `Gate check for 'X' returned NO/UNCHANGED` → wasted
- `Gate check for 'X' returned a non-verdict; retrying` → wasted ×2

**Per-subject breakdown:** Which subjects have the lowest YES rate? If a subject's gate
check returns NO >80% of the time, it's a candidate for:
- Skipping the gate check entirely (always process)
- Reducing frequency (check every N turns)
- Replacing with a cheaper heuristic (e.g. "only check if the subject's JSON size changed")

**Deliverable:** Table of gate check YES/NO/retry counts per subject, with the ratio.

### Axis 2: Discovery Call Efficiency
**Question:** What fraction of `add_new` calls returned an empty array (no new entities)?

**Log signals:**
- `Failed to extract new entry names` → empty result (wasted)
- `New entries detected: [...]` → useful
- The discovery prompt asks "what NEW entries should be added?" — if the answer is
  consistently `[]`, the call is pure overhead.

**Per-subject breakdown:** Elements likely have the highest discovery rate (new items
appear frequently). Characters and groups likely have the lowest (new characters are rare).

**Deliverable:** Table of discovery call counts per subject, with empty-result rate.

### Axis 3: Branch Update Size Distribution
**Question:** How many branch updates were trivial (1-2 field changes) vs substantial?

**Log signals:**
- The branch update LLM response is the JSON of field updates. Small responses (a few
  fields) suggest the update was marginal.
- `NO_UPDATES_REQUIRED` responses = the LLM looked and found nothing to change.

**Deliverable:** Distribution of branch update response sizes (token count), with
percentiles. Flag updates below a threshold as "trivial."

### Axis 4: Redundant Recursion
**Question:** Did the code recurse into sub-branches after a parent gate check failed?

**Code path:** In `_update_recursive`, when `gate_failed` is True, the code executes
`gate_check_fail` conditional actions and then `return data` (line 138). This should
prevent recursion. But verify: are there code paths where a sub-branch is processed
despite the parent being skipped?

**Deliverable:** Code audit of the gate-fail → skip path, confirming no leakage.

### Axis 5: Boundary Check & Digest Frequency
**Question:** How often do chapter/arc archival and canon digest calls fire, and are
they always useful?

**Log signals:**
- `Archiving chapter` / `Archiving arc` lines
- `canon` phase calls

**Deliverable:** Count and timing of boundary/digest calls across the run.

### Axis 6: Per-Subject Call Budget
**Question:** What's the total LLM call count per subject per turn, and which subject
dominates?

**Deliverable:** Per-subject call count histogram across all turns. Identify the top
contributors to the 25 calls/turn.

## Execution Plan

### Phase 1: Log Analysis (subagents)
Dispatch 2 subagents in parallel:

**Subagent A: Call Distribution Profiling**
- Parse the run log for all `local-call` lines
- Categorize each call by `phase_id/step_id` (gate_check, branch_update, discovery,
  boundary, digest, writer, etc.)
- Compute per-category: total count, per-turn distribution, mean/median/p95 duration
- Cross-reference with gate check verdict lines (YES/NO) to compute wasted-call rates
- Cross-reference with discovery result lines to compute empty-result rates
- Deliverable: `<run_id>_call_profile.md`

**Subagent B: Update Size & Redundancy Analysis**
- Parse the run log for branch update responses
- Measure response token counts per update
- Identify NO_UPDATES_REQUIRED responses
- Check for repeated updates to the same field within a short window (redundant edits)
- Deliverable: `<run_id>_update_redundancy.md`

### Phase 2: Code Audit (subagents)
Dispatch 2 subagents in parallel:

**Subagent C: Gate Check Code Path Audit**
- Read `_perform_gate_check` and `_update_recursive` in full
- Map every subject's trigger map from the schema file
- Identify which subjects have gate checks that are likely always-YES or always-NO
- Check for the `gate_check_fail` conditional action paths
- Identify any code paths where a failed gate doesn't prevent downstream work
- Deliverable: `gate_check_audit.md`

**Subagent D: Discovery & Recursion Code Path Audit**
- Read `_detect_and_add_new_entries_to_branch` in full
- Map which subjects have `add_new` triggers
- Check the `_traverse_structure` recursion: does it re-invoke triggers on sub-branches?
- Identify any redundant LLM calls in the discovery path (e.g. double strip_response)
- Deliverable: `discovery_recursion_audit.md`

### Phase 3: Consolidation
Parent reads all 4 subagent deliverables, merges into a single optimization report with:
- Ranked list of unnecessary call categories (by total time saved if eliminated)
- Concrete code changes proposed for each
- Estimated time savings per turn