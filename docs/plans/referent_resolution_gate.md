# Pre-Mutation Referent-Resolution Gate — Patch Scope

Status: **H1-H4 IMPLEMENTED (uncommitted)** + dangling-edge regression test landed.
Companion to `rag_redesign_p1.md`. See "Landed state (2026-09-10)" below for the
verified coverage table and the one genuine residual (H3 value-referent retarget).
Root-cause evidence: `tests/split_node_audit_report.md` (mechanical) +
`tests/split_adjudication.md` (story-level) + `tests/alias_branchquery_map.md`
(prompt map) + `tests/dangling_edge_test.py` (standing dangling-edge regression,
GREEN 6/6) + turn-84 live-graph replay (176 nodes / 525 edges, 18.1% dangling).

## Problem (one sentence)

The model emits entity data *addressed at the wrong subject* — a new lookalike key
(split), a foreign name into another's alias array (poisoning), foreign prose into
another's description (contamination), or a nonexistent partner (dangle). Same root,
four symptoms, growing linearly with run length (323 dangling edges, 10→332 over one run).

## Root cause (verified in primary source)

The branch edit path is **THIN**: both branch update templates receive a fully-computed
`{{ value }}` (the target node's complete JSON, computed at `prompts.py:395-410`) but
**never emit it**. So at the branch boundary the model is asked to regenerate the entire
alias/relationship array *from scratch* with only a names-only sibling roster in view
(the degraded `_context_restatement`, `prompts.py:129-139`, once the map outgrew the
10000-char gate). The field-level path, by contrast, *is* inlined (`_update_field` at
`updates.py:719` emits `Current value:\n\`\`\`\n{{ value }}`). The defect is precisely
the thin path.

## Budget constraint (non-negotiable)

The fix must NOT re-arm the whole-map restatement. It inlines **the target node only**
(one subject's own arrays/rows — order of KBs), never the O(whole map) block the
`restate_map_threshold_chars` gate (core.py:53) exists to suppress. Target-node-scoped
inline is cache-favorable and budget-neutral relative to the existing field-level inline,
which already pays this cost.

## Four hunks

### H1 — Target-node-scoped inline (primary lever, #278-compliant)

Make the branch templates EMIT the already-computed `{{ value }}`.

- Character/Group/SceneState `branch_update_prompt_template`: add
  `Current entry state:\n\`\`\`\n{{ value }}\n\`\`\`` (mirroring the field-level form at
  `updates.py:719`). The value is already computed — this is a template-string change,
  not new plumbing.
- Same for the branch *query* templates (the gate step sees the same ground truth).
- Confirm `value_str` is populated for the whole-node path (keys=[entry], field_name="")
  — if recursive_get already returns the node dict, no code change; only the template.

Effect: model edits alias/relationship arrays *against ground truth* instead of
regurgitating. Converts the split/poison/contaminate family from "model invents foreign
data as if it were the target's" to "model diffs against what's actually there".

### H2 — Canonical-key constraint (prompt-side)

Add to each branch update template a hard constraint:

> Every referent you emit (a relationship partner, an alias, a participant, a
> milestone subject) MUST resolve to a key already present in the entry list below
> [names-only sibling roster — already in view via the degraded restatement] or to this
> entry's own keys. If the referent is a NEW entity, it belongs in an add_new proposal,
> not as a relationship/alias value here. Never write a partner/alias that names a
> different existing entry.

The sibling-roster names are ALREADY computed (prompts.py:129-139 degraded path) — reuse
the same source, so this adds no budget.

Effect: model is told, at the boundary, that a foreign referent is *invalid output*, not
a new node — collapsing the split class at generation time.

### H3 — Parse-time retargeting (belt, non-eventizing)

In `_apply_branch_updates` (updates.py:417), after `_resolve_path_keys` canonicalizes the
*path* (parsing.py:117), add a symmetric pass over the *value*: for any alias/relationship
partner string in the value that normalizes-collides with a *distinct* existing node,
route it through `_resolve_dict_key` (parsing.py:95: exact→normalized→alias) and
retarget onto the matched canonical node. A genuinely shared identity (undercover/impersonation)
registers as an alias on the *correct owner's* array — not prohibited, only re-addressed.

- Gap to close: `_filter_new_entry_names` (parsing.py:241) today only *rejects* a novel
  proposal on containment (secondrunner ⊂ thesecondrunner) and never retargets. Add the
  retargeting branch: if a proposed name/alias is a near-dup of an existing node, map the
  write to that node instead of creating a sibling.
- Retargeting must be **log-verbose** (which write moved where) so it's auditable, and
  must be a **no-op** when no collision exists (idempotent, never rewrites a legitimate
  shared identity).

Effect: the model's residual slips become non-events instead of permanent store corruption.

### H4 — Alias placeholder hardening (schema)

The entire alias contract is two example-phrasing lines (schema.json:27,58) + bare
`list[str]` for two of four alias-bearing fields. Add to each `aliases_placeholder`:

> A genuine alias is a name for THIS entry only — a nickname, title, or handle that
> refers to this same entity. It is NOT the name of a different, separately-tracked
> entity. If a term names another character/group/element in the roster, do not list it
> here; it belongs on that entity's own entry. Shared/undercover identities are recorded
> as a relationship/milestone to that entity, not as an alias on this one.

Apply to all four classes: CharacterRelationship, CharacterGroupStatus, Group,
StoryEvent (aliases field).

Effect: closes the "alias = other node's key" ambiguity that produced the poisoning.

## Hermetic smoke gate

`tests/referent_gate_test.py` (landed; read-only, stdlib, model-free, wired into
`run_tests.py` with dump recording) over a frozen store — 27/27 GREEN. A second
standing gate, `tests/dangling_edge_test.py` (also landed, GREEN 6/6), audits the
entity graph's edge set for dangling endpoints and classifies each as `phantom`
(referent spelled with no backing node anywhere) or `type_mismatch` (referent's
bare name exists under a different type prefix) — the mechanical regression for
the turn-84 finding below.

1. **Inline presence**: build a branch update prompt for a node; assert the target's own
   alias array + relationship rows appear verbatim in the emitted prompt (H1 works).
2. **No whole-map re-arm**: assert the prompt does NOT contain other entries' field
   content (only names) — budget discipline held (H1 scoped, not re-armed).
3. **Constraint present**: assert the canonical-key constraint line is emitted (H2).
4. **Retargeting idempotent**: feed a value with a partner that collides with a distinct
   node → assert it is re-addressed to the canonical key AND a value with no collision
   passes through unchanged (H3). Run twice → identical store (no drift).
5. **Shared-identity preserved**: a value representing a legitimate undercover/impersonation
   edge is NOT collapsed — retargeting re-addresses, never erases the distinct-target
   relationship (H3 + adjudication C4/C3 boundary holds: Pike wore Vell's frequency).
6. **Alias placeholder**: assert each of the four classes' alias placeholder now
   contains the "not a different entity" clause (H4).

Gate: 6/6 green, no existing `run_tests.py` regression. Then a 100-turn soak A/B:
dangling-edge count + split-pair count + alias-poisoning count over the run (compare
pre-patch baseline: 323 dangling, 7 splits, 14 contamination entries).

## Scope / non-goals

- In scope: 4 hunks (H1-H4) + smoke gate. No model spend, no RAG, no small-model
  dependency (per #294).
- Out of scope: merging already-corrupted existing run data (that's a one-time migration,
  not engine logic); the small helper model (optional/progressive, separate track).

## Landed state (verified 2026-09-10, uncommitted)

H1-H4 are **implemented** and green. The original "not yet implemented" status
above is STALE. Verified sentinel coverage across BOTH schema files
(`user_data/example/subjects_schema.json` + `.../schemas/subjects_schema_sceneagg.json`):

- **H1 (target-node inline) + H2 (canonical-key constraint)** land on the
  `branch_update_prompt_template` of **6/7 subjects**: `Character`, `Group`,
  `SceneState`, **`StoryEvents`**, `GeneralInfo`, **`Element`**. Whole-map
  re-arm sentinel: **False** on event/element templates (budget discipline held —
  the inline is target-node-scoped, mirroring the field-level form, not the
  O(whole-map) block the `restate_map_threshold_chars` gate suppresses).
- **H4 (alias hardening)** lands on the alias placeholders of **4 classes**:
  `CharacterRelationship`, `CharacterGroupStatus`, `Group`, **`StoryEvent`**.
- **H3 (parse-time retargeting)**: `_resolve_path_keys` (path, parsing.py:172) +
  `_retarget_value`/`_retarget_dict_key` (value, parsing.py:142-169), wired into
  `_apply_branch_updates` (updates.py:445/455) — value-side retarget of
  **dict keys** only.
- **`tests/referent_gate_test.py`**: landed, 27/27 GREEN, dump-wired.
- **`tests/dangling_edge_test.py`**: landed, 6/6 GREEN, dump-wired.

**Consequence for the turn-84 evidence:** the 18.1% dangling-edge gap (95/525
edges) + 6-of-7 phantom top-ranked events was produced by a run on the
**PRE-PATCH** schema. The gate as landed is *already designed to prevent that
class* — it simply predates the evidence. No new template hunks are required to
"extend the gate to event/scene/element": those branch templates already carry
H1+H2 and the StoryEvent alias already carries H4.

## Residual (the one genuinely open lever)

The single remaining gap, confirmed against source: **H3 does not retarget
value-referents inside list-of-dicts rows.**

- `_retarget_value` (parsing.py:142) retargets **dict keys and top-level keys of
  dict rows** only. Its contract states it "leaving **string items, scalars, and
  prose untouched**" (line 149-151).
- But the dangling milestone/participant referents are stored as **list-of-dicts
  where the target is a STRING *VALUE***, not a key: the graph reader reads
  `rel_item.get("title", rel_item.get("name"))` (entity_graph.py:516, 528) and
  the dangle enters via that title *value*.
- Therefore a dangling milestone/participant **title value** passes through
  `_retarget_value` un-retargeted and mints the dangling edge exactly as the
  turn-84 replay recorded (38 milestone + 48 relationship + 9 character
  dangling, all target-side).

**Resolution (scoped, non-eventizing):** extend the value-side pass so that, for
a list-of-dicts row, the string *value* fields that name a referent (the
`title`/`name`/partner fields) are also run through `_resolve_dict_key`: if the
value resolves to a **distinct** existing node, rewrite the value onto that
canonical key. Same guarantees as the existing key-retarget — fires only on a
distinct collision, guaranteed no-op otherwise, so legitimate shared/undercover
identities (one canonical referent) and cross-subject prose are never rewritten.
This is a **parse-time correction**, not a model change, and is the lever that
closes the specific dangling class the turn-84 audit proves live.

**Verification before any soak:** extend `tests/dangling_edge_test.py` (or the
referent-gate test) with a fixture where a list-of-dicts milestone row carries a
dangling title *value* distinct from an existing node; assert it is re-addressed
onto the canonical key (and a non-colliding title value passes through
unchanged). Then the existing 100-turn soak A/B: dangling-edge count +
split-pair count + alias-poisoning count vs the pre-patch baseline
(323 dangling, 7 splits, 14 contamination entries).

## File touch list

Landed (uncommitted; do not re-apply):
- `user_data/example/subjects_schema.json` + `.../schemas/subjects_schema_sceneagg.json`:
  H1 (branch update+query templates emit `{{ value }}`), H2 (constraint line),
  H4 (4× alias placeholders). NOTE: not git-tracked — JSON round-trip edits, ensure_ascii=False.
- `agents/data_summarizer/prompts.py`: `value_str` populated for node path (H1) — confirmed.
- `agents/data_summarizer/updates.py` `_apply_branch_updates`: path canonicalization
  (line 445) + value retarget (line 455) — landed.
- `agents/data_summarizer/parsing.py`: `_resolve_dict_key`/`_retarget_dict_key`/
  `_retarget_value`/`_resolve_path_keys` — landed. (`_filter_new_entry_names` left
  untouched — its containment-rejection already handles the add_new path.)
- `tests/referent_gate_test.py` (landed, 27/27, wired into `run_tests.py`).
- `tests/dangling_edge_test.py` (landed, 6/6, wired into `run_tests.py`).

Residual (open lever — see "Residual" above):
- `agents/data_summarizer/parsing.py`: extend `_retarget_value` to retarget the
  string *value* fields of list-of-dicts rows (title/name/partner) via
  `_resolve_dict_key`, preserving the distinct-collision/no-op guarantees.
- `tests/dangling_edge_test.py` (or `referent_gate_test.py`): value-referent
  fixture proving the milestone/participant title-value dangle is re-addressed.