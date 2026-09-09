> **Matrix status 2026-08-25:** Type 1 Mode A = locked baseline (seed-93, 30t). Type 2 Mode B =
> default test config (multiple 40t runs; best `fantasy_epic__551bcf41` 4.0/5;
> `cyberpunk_thriller__4e29a512` 3.5/5 post-calibration-ladder). Type 2 Mode A deliberately
> skipped — two maximally different combos converged on the same quality ceiling, so attribution
> moved to other axes (instructions/retrieval/render). Type 1 Mode B untested.
> Mode B machinery (rolling window + rolling summaries sticky lock) shipped engine-side.

# Schema types + soak test matrix (2026-08-12)

Test conditions for the dayna_ss soak harness, and the schema-type dimension
(trigger architectures) that drives the biggest behavioral differences.

## 1. Schema types (trigger architectures)

A schema type is a variant of `subjects_schema.json` whose *trigger map* places
each subject's actions (add_new, gate check, branch query, branch update) on the
`ALWAYS` (every-message) vs `ON_NEW_SCENE` (scene-transition) axes. The goal is
to test a few distinct architectures head-to-head and pick the best, not to
rewrite the engine.

### Type 1 — "incremental-per-turn" (current default `subjects_schema.json`)

Everything is touched often; nothing relies on scene boundaries.

| subject | ALWAYS (every message) | ON_NEW_SCENE |
|---|---|---|
| characters | gate check, per-entry query | add_new |
| groups | add_new, gate check | — |
| elements | add_new, gate check | — |
| current_scene | now (SceneState query), when/why update | — |
| events | — | gate check → (StoryEvents add_new) |
| general_info | — | gate check + update |
| arcs / chapters | — | gate check + add_new |

Pros: fresh every turn; robust to scene-transition detection failures.
Cons: most local LLM calls per turn; scene-wide narrative shape is rebuilt
incrementally from the latest exchange only.

### Type 2 — "scene-aggregation" (target, `subjects_schema_sceneagg.json`)

The user's original intent, split into two cadences:

- **a. Every message**: add new **Events** + update **current_scene**.
- **b. On new scene turn**: update *everything else* (new/updated characters,
  groups, elements, general_info, arcs) using an **aggregated scene context** —
  the scene's events + all sequential message summaries + a plain listing of
  exactly what happened during the scene / scene part — rather than the raw
  latest exchange.

| subject | ALWAYS (every message) | ON_NEW_SCENE |
|---|---|---|
| characters | — | add_new + per-entry update w/ scene context |
| groups | — | add_new + update w/ scene context |
| elements | add_new (+ gate) | — |
| current_scene | now + when/why update | — |
| events | — | add_new (StoryEvents) + per-event query (review) |
| general_info | branch-update (single skip_query call) | — |
| arcs / chapters | — | gate check + add_new w/ scene context |

### Type 2 (refined, 2026-08-13) — minimal-runtime summarizer

The user's clarified spec makes Type 2 the *minimal-runtime* architecture —
the lightest possible every-message load; heavy subjects run only on scene
transitions (the scene-part budget at `max_scene_messages=12` guarantees a
transition every ~6 turns):

1. **Every message** (ALWAYS):
   - branch-update `current_scene.now` (SceneState query, gates the
     who/when/why drill-downs),
   - branch-update `general_info` (`skip_query: true` — one self-detecting
     call, no separate pre-query),
   - add new **elements** (ElementMap `always: [add_new, perform_gate_check]`,
     per-entry element updates stay on scene cadence).
2. **On new scene** (ON_NEW_SCENE):
   - archive the scene,
   - add new + branch-update **characters**, **groups**, **events**
     (events `add_new` moved here from ALWAYS — event collection is
     scene-bounded), still grounded in the aggregated scene context.

Trigger deltas applied to `subjects_schema_sceneagg.json` (verified vs the
base Type 1 schema):
- `GeneralInfo`: `on_existing_scene/on_new_scene` → `always: [query_branch_for_changes, skip_query]`.
- `ElementMap`: `on_new_scene` → `always: [add_new, perform_gate_check]` (parity with Type 1; add_new dedupe caps apply).
- `StoryEvents`: `always: [add_new]` → `on_new_scene: [add_new, ...]`.

Per-message cost ≈ 1 (scene) + 1 (general_info) + 1-4 (elements add_new) calls
+ the always drill-downs, vs ~40-50 for Type 1. Aggregation injection
(`scene_events`, `message_summaries`, `scene_recap`) unchanged on the b-pass.

Pros: far fewer calls per turn; scene-aware updates grounded in the full scene,
not one exchange.
Cons: relies on regular scene transitions for freshness; character/element
stale between scene parts; higher risk of the freeze class if a gate answers NO
(mitigation: add_new unconditional, gate only short-circuits per-entry queries).

### Type 3 — "hybrid" (optional)

Events per-message (Type 2 a) but characters/groups/elements keep per-turn
incremental updates (Type 1) — no aggregation. Useful as an A/B control to
isolate the value of the aggregation pass vs the trigger reshuffle.

## 2. Schema resolution order (per-genre × per-type)

Schema files live in `user_data/example/schemas/` (name-suffixed variants) with
the legacy path as final fallback. Resolution:

1. Explicit override: `subjects_schema` config key or soak `--schema <path>`.
2. Per-genre + per-type: `subjects_schema_<genre>_<type>.json`.
3. Per-type: `subjects_schema_<type>.json` (e.g. `subjects_schema_sceneagg.json`).
4. Per-genre: `subjects_schema_<genre>.json`.
5. Legacy default: `user_data/example/subjects_schema.json`.

Genre variants tune sensitivity/traits (rule #35: schema = data). E.g. the
detective genre raises add_new sensitivity and tightens element traits so the
plot store (items, places, people) is denser than a romance's.

## 3. Soak test matrix

Each row is a dimension; a run is a tuple of choices. The dashboard/report
already key on the config hash — the schema type must be part of that hash so
identical args with different schemas don't collide.

| dimension | options | notes |
|---|---|---|
| spreadsheet / genre | cozy_mystery, noir, fantasy, sci-fi, horror, western, romance | 7 fixtures; beats + notes differ |
| **schema type** | 1 (incremental), 2 (scene-aggregation), 3 (hybrid) | `--schema-type N` |
| director | on / off | `--director`; instruction block presence |
| level | 1 (fast per-subject) / 2 (full engine path) | L2 is the realistic test |
| local model | Qwen3.6-35B-A3B (current), 9B Qwen3.5 (legacy), Magnum-v4-12b (Mistral alias, pending), Gemma-4-E4B (llama.cpp) | see model_test_slate.md |
| cloud roles | guide / judge / auditor / replan | deepseek-v4-flash default; judge/auditor swap free at resume |
| cadence | turns, judge-every, plan-every, audit-every | quality vs cloud $ |
| scene part budget | `--max-scene-messages` (0 disables, default 12) | drives ON_NEW_SCENE cadence |
| parallelism | `--max-subject-workers` (0 serial … 4) | ~1.8x |
| embedding device | cuda:0 (1060) vs cuda:2 | model placement |
| context mode | A: enumerated prompt-start (baseline) vs B: rolling + `{system,inline}` (§3) | `--message-mode` / `--retrieval-placement`; prefill + recall |

### Suggested comparisons (choose the winner)

1. **Schema Type 1 vs Type 2**, same genre + seed-class, 20 turns: the primary
   open question. Watch: final overview, memory fidelity curve, events
   population, local call count (expect Type 2 materially lower), recall.
2. **Director on/off**, same schema: instruction-block influence on style/echo.
3. **Model size**: 35B vs 9B vs Magnum on the winning schema type.
4. **Genre sweep** on the winning schema type: does detective density survive
   the same schema as romance?

---

## 3. Mode B — rolling message history + inline retrieval (design, 2026-08-13)

New test condition (orthogonal to schema type). Today every generation call
renders the same assembled prompt:

```
system   : harness context (premise/setting/style/directives)
           + importance_scale + general_info + current_scene   (to_context:true subjects)
history  : [simulated retrieval Q&A pairs]  -> "What are the current narrative arcs?"
           ...                                "Now, describe each of the relevant characters."
           [enumerated dialogue]           -> "What were the last N exchanges (pairs of messages)?"
                                               "14. 'Prudence' >> ...  / 13. 'Evelyn' >> ..."
           ["Analyze all of the above information...", "Analysis complete."]
prompt   : mark_field(<branch>) + the schema template (add_new / gate / branch update)
```

Problems (the seed-89 O(N²) driver): the retrieval data is a *fake conversation*
at the prompt start (start-of-prompt attention dilution, big prefill, all
subjects embedded in every per-entry call), and the recent dialogue is
*enumerated* (numbered re-statement of every message — token-heavy, and the
model re-reads the scene even though it just wrote it).

### Mode B changes

Two config keys on the Summarizer (soak-overridable, folded into the run-id
hash), both data-driven through the existing `_context_order`/format-template
machinery — no hardcoded dispatch:

1. `message_mode: "rolling"` (default `"enumerated"`)
   - Drop the `format_dialogue` numbered block. The recent window
     (`history[-last_x:]`, still scene-bounded via `_scene_dialogue_window`)
     is appended to the internal history as *raw user/assistant pairs* — the
     model sees the actual conversation, not a re-statement.
   - Drop the "Analyze all of the above information" marker (it exists only to
     close the Q&A framing).
   - `last_x` becomes a config `rolling_window` (default 6).

2. `retrieval_placement: "system" | "inline" | "prompt_start"`
   - `"prompt_start"` (default / baseline): current behaviour — simulated
     Q&A pairs at the head of the internal history.
   - `"system"` (PRIMARY DESIGN, user-confirmed 2026-08-14): the non-to_context
     subjects render once per turn into ONE `CURRENT CONTEXT` block appended to
     the **system message** (`custom_state["context"]`). Because the same
     custom_state is shared, the block is in context for BOTH the reply AND
     every DataSummarizer call — per-entry calls keep full cross-subject
     visibility (all subjects) with zero extra engineering, and within a turn
     the stable system prefix is prefix-cached (encoded once, reused by every
     call in the turn).
   - `"inline"`: same block, but injected at the generation boundary (prepended
     to the instruction) instead of system — control arm to attribute the gain
     to placement vs message-mode.

### Confirmed prompt structure (user design, 2026-08-14)

```
[CONTEXT]                         <- harness premise/setting/style/directives/cast (as today)
[GENERAL_INFO]                    <- to_context appends (general_info, current_scene, importance_scale)
[CURRENT_SCENE]
[OTHER RETRIEVED RELEVANT SUBJECTS]  <- arcs / character_list / groups / elements / events /
                                        chapters / characters / lines(RAG), rendered once
                                        into ONE block (NEW: moved out of the Q&A framing)

<blank line>

[rolling user/assistant pairs]    <- the recent window, raw pairs (no enumeration)
[prompt]                          <- branch template instruction (summarizer call)
                                     or instr_prompt (reply call)
```

The four header blocks are ALWAYS present — reply and summarization share the
same system prefix per turn. `[prompt]` is the only per-call part; the rolling
messages sit immediately above it as real speaker turns (speaker attribution
kept — raw pairs, not a numbered re-statement).

### How the window stays dynamic (the user's open question)

`last_x` is not a constant — it's computed per turn by
`_scene_dialogue_window` (scene-bounded, `last_x_max`/`last_x_min`) and
`retrieve_and_format_context` slices `history[-last_x:]`. In Mode B the same
computation feeds the raw-pair window; the only changes are (a) the slice is
rendered as raw pairs instead of `format_dialogue` enumeration, (b) the window
size config name (`rolling_window`) flows through the Summarizer config to
`get_retrieval_context` kwargs, exactly like `last_x`/`last_x_max` do today —
one plumbing point, no hardcoding.

`rolling_window` and `--max-update-history` are DIFFERENT layers and both stay:
`rolling_window` = the model-visible semantic window (how many recent exchanges
the templates reference as "the latest exchange(s)");
`--max-update-history` = a transport cap on how much history the client
physically sends per DataSummarizer call. They compose:
`sent = history[-min(rolling_window, max_update_history):]`. Under
`retrieval_placement=system` the bound no longer risks blinding per-entry calls
to the subjects (those live in the system block, never bounded) — it only clamps
the rolling window, which is exactly what it should do.

### Sticky-roll message summaries (engine upgrade, 2026-08-16)

**Problem**: rolling mode drops messages outside `history[-last_x:]` entirely —
older story context (earlier scenes' details, plants, decisions) simply stops
existing for the model. For a long horizon this is a hard retention cliff.

**Fix** (engine-side, `agents/summarizer.py`): when the config key
`rolling_summaries` > 0, `retrieve_and_format_context` injects a single pair
ahead of the raw recent pairs:

```
Summaries of earlier messages (messages S-E, before the most recent window):
- [message 30] Prudence deduced that Mrs. Arbuthnot's map was a forgery...
- [message 31] ...
```

The summaries come from the accumulated `message_index` store in the current
history dir (llama_index docstore; `is_summary` nodes). Empirically the store
accumulates the full summary history every turn (whole-index persist + retriever
instance reuse), so a fresh chunker load at retrieval time sees every summary
from the start of the run — verified on the real 40-turn `b884af4a` store (idx
2..81).

**Sticky roll** (`_rolling_summary_window`): the summary range reaches back at
least `rolling_summaries` exchanges (the floor), OR to the start of the last
`ROLLING_SUMMARY_SCENE_ROLL` (5) scenes
(`Math.max(floor, last_5_scenes)`), whichever reaches further. Scene starts come
from the archived `events.scenes[*].start._message_node` plus the current
scene's start; with fewer than 5 scenes, all of them count. The range always
ENDS where the raw rolling window begins, so summaries never duplicate
in-window messages.

**Sticky lock** (2026-08-16): the computed `start` is LOCKED in memory on the
Summarizer instance until the next scene turn (`_current_scene_key` identifies
the scene by its start message-node, else `_scene_number`). Within a scene,
every subsequent turn reuses the SAME `start` and only advances `end` — so the
summaries block stays byte-stable across turns for prefix caching (encode once,
reuse for every call in every turn of the scene). A scene change recomputes
with the new bound list. In-memory only by design: a process restart
legitimately recomputes, since the prefix cache is gone with it.

**Bound interplay**: the harness `LocalModel._engine_internal_messages`
`--max-update-history` bound now PRESERVES the summaries pair and bounds only
the raw rolling pairs behind it — otherwise the pair (injected at the head of
the dialogue section) was trimmed on DataSummarizer per-entry calls and the
feature only ever served REPLY calls, where prefill is cheap. Per-entry calls
now carry the sticky-roll context too.

**Flow**: message_idx numbering matches production (`greeting` = idx 0/1, turn N
user = idx 2N+2 — `user_input_message_idx = len(history) * 2`). The block lands
inside `custom_state["history"]["internal"]` (with the raw pairs), so the harness
`_engine_internal_messages` renders it automatically and every engine call that
sees the dialogue sees it too.

**Wiring**: `--rolling-summaries N` (0 = off, default) → Summarizer config
`rolling_summaries`; folded into the run-id hash (a different value is a fresh,
comparable run dir). `run_tests.py --rolling-summaries` passthrough added.

**Open design question (a vs b) — mentioned subjects**: whether per-message
"mentioned entities" should be (a) a separate message node (structured, extra
LLM call per message) or (b) embedded in the free-text summary (zero extra
calls, less structured). See `long_horizon_soak_plan.md` §21.3 for the current
answer and follow-up.

### Why it should help

- **Prefill / prefix caching**: one stable system prefix (the four blocks) per
  turn, shared by all calls → encoded once, reused by every per-entry call and
  the reply. The fake-conversation Q&A pairs and the enumerated dialogue block
  vanish entirely.
- **Per-entry context**: with `retrieval_placement=system`, per-entry calls see
  ALL subjects (user decision) — strictly better than today, where
  `--max-update-history` trims the Q&A tail and can blind them to early subjects.
- **Retention**: the data block is present in context for the reply; combined
  with the Q-P3 memory-surfacing directive, it's the recall lever.
- **Control arms**: `message_mode × retrieval_placement` gives
  enumerated-start (baseline) / rolling-system (primary) / rolling-inline
  (placement control) / rolling-start (message-mode control) for attribution.

### Open question (implementation detail, not design)

**Injection mechanism for `"inline"`** only: engine prepends the block to the
instruction text (`generate_instr_prompt`) vs harness appends a history pair.
Recommend the engine prepend — self-contained, lands right before generation.
For `"system"` (the primary), no question: `retrieve_and_format_context` already
appends to `custom_state["context"]`; the OTHER-SUBJECTS block is just a new
append after current_scene. `"prompt_start"` keeps the current Q&A path.

### Harness wiring

New flags `--message-mode {enumerated,rolling}` + `--retrieval-placement
{prompt_start,system,inline}`, passed to the Summarizer config, folded into the
run-id config hash so modes get distinct run dirs.
Add a **Mode A (baseline) vs Mode B** comparison to the test matrix.
