# Scene start→now deterministic copy

## Problem
At a scene boundary the model is prompted (via `scene_start_update_prompt_template`)
to (a) completely rewrite `current_scene.start` as the new scene's opening and
(b) **reset `current_scene.now` to that same opening moment** so `start` and
`now` render identically at the boundary. Empirically the model performs (a)
reliably but (b) unreliably — in **20 of 29 first-boundary turns (69%)** `now`
is already ahead of `start` at the transition (e.g. 4dbf1435:
`start.where='Ashenfall Skyway'` vs `now.where='Graft stair-clusters…'`).
This is the same failure class as the `who.characters` nameless entry and the
RAG boilerplate collapse: the model violates a *declared* contract because
structure is enforced but field-level compliance is left to the LLM.

The result: at a boundary `now` carries the *prior* scene's closed state forward,
so the new scene's opening is recorded inconsistently (start says "the alley",
now says "the Graft") and downstream scene-bounded windows render the new
scene as if it began mid-motion.

## Fix
Make `now` a **deterministic** function of `start` at the boundary. The model
writes the opening (`start`, a model-reliable structural act); DSS copies
`start → now` (a model-unreliable structural act) so the two are identical at
the boundary and then diverge over the scene's subsequent turns.

## Insertion point (verified)
`DataSummarizer.generate()` in `agents/data_summarizer/core.py`, immediately
**after** `_update_recursive` (line 418) and **before** the `save_json`
persist (line 441). This is the sole point where BOTH:

1. the model's fresh `start` rewrite has already landed in `data["start"]`
   (written in place by `_update_recursive`), AND
2. the in-place mutation will ride the existing single `save_json` persist,

so no separate re-save and no extra write ordering is required.

Guard: `data_type == "current_scene"` **AND**
`self.summarizer.last.is_new_scene_turn` (the same flag that drives
`Trigger.ON_NEW_SCENE` at line 207). This makes the copy scene-specific and
boundary-specific.

## Semantics (start-gated — NOT a blind clobber)
Copy `data["now"] = deepcopy(data["start"])` **only when** `data["start"]` is a
*genuinely populated* opening. If `start` is the empty NO_UPDATES shell
(meta-only, no content keys) and `now` is already populated, **skip** — a blind
clobber would destroy the only good `now` data (the 5 start-absent cases in
tests/runs/* have populated `now`; clobbering those would wipe real state).

- `start` **populated** → model is opening a new scene → now must equal it.
- `start` **empty/NO_UPDATES**, `now` populated → model declined to rewrite the
  opening → preserve `now`; only the model's partial update (if any) applies.

`deepcopy` is mandatory, never an alias — `start` and `now` must be independent
objects so subsequent scene-internal updates to `now` do not mutate `start`
(`start` is declared immutable: `no_update: true` in the schema, enforced at
`traversal.py:362`).

## Why not the main loop
`_stamp_new_scene_meta` (summarizer/core.py:836) runs *before*
`_process_all_subjects` (line 853), i.e. **before** the model's `start` rewrite
exists. A copy placed there would see the *prior* scene's `start` (or the stamp's
own empty shell) and clobber the new scene with stale data. `generate()` is the
first point at which the model's fresh `start` is observable.

## Invariants preserved
- `start` immutability (schema `no_update: true`) is untouched — we only write
  to `now`.
- Single atomic persist per subject file (the copy rides the existing
  `save_json`), no new write ordering.
- Scene-internal turns (non-boundary) are untouched — `now` evolves freely
  relative to `start` after the boundary; the copy fires only at
  `is_new_scene_turn`.

## Out of scope
- RAG retrieval (separate track).
- Any schema change — `start`/`now` shapes are unchanged.
- The model's `scene_start_update_prompt_template` instruction to reset `now` is
  left as-is (now redundant but harmless); the deterministic copy is
  authoritative and idempotent with it.