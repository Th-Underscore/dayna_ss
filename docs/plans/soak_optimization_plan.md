# Soak optimization plan — performance + quality (cozy_mystery__2481faa4 baseline)

Status legend: [ ] open · [x] done · [~] in progress · [ ] blocked/needs input

## Baseline (2026-08-11, 20-turn run, local 9B AWQ + cloud guide/judge/auditor)

- **Runtime ~3h (2.9h)**, 894 local calls / 10.1M prompt tokens. Judge 3/5.
- Per-turn local call deltas: ~25–46 baseline, spikes at t06=109 (1873s), t16=89 (1395s), t17=73 (821s).
- **Perf root causes (subagent analysis → /tmp/opencode/perf_analysis.md):**
  - ElementMap + GroupMap carry `add_new` under `always` (schema) → every turn the LLM
    fabricates ~10–74 micro-elements (`The_Bakery_Air`, `The_Florist_Gaze`) as full-JSON
    calls. t06 +74 elements, t16 +43. Dominant cost driver. (Characters correctly do
    `add_new` only under `on_new_scene`.)
  - Minimum all-NO baseline is 18 calls/turn (gate + branch + update checks per subject).
  - `NO_UPDATES_REQUIRED` token is demanded by every template but not treated as a negative
    in `_perform_branch_query` → wastes a follow-up apply call.
  - current_scene's `now` runs 7 unconditional `perform_update` full-branch calls even when
    the branch query returned UNCHANGED (no short-circuit on negative).
- **Quality root causes (subagent analysis → /tmp/opencode/quality_analysis.md):**
  - Q1 — Turn-17 auto-detected "scene transition" re-ran `_populate_from_first_scene`
    (`summarizer.py:1090` gates on `not has_archived_scenes`, which was False all run).
    The `.populated_from_first_scene` marker is written (line 1104) but NEVER checked.
    Wipe destroyed accumulated memory: elements 206→18, general_info/characters rebuilt from
    a bare window → Evelyn/Mrs. Winters conflation, c13 probe lost, t17 spike.
  - Q2 — Engine director feedback loop: per-turn instructions DICTATE imagery that then
    recurs verbatim in replies ("heavy crust, light filling" in 10/20, "crumbs like tiny
    white questions" in 7/20). turn_002 instr invents it; turn_010 copies it from the
    model's own turn 3. No anti-repetition/freshness rule.
  - Q3 — DSS stored+retained all planted notes (probes present t3–16) but never surfaced
    them in replies (model selection failure); c13 wiped by the Q1 rebuild.
  - Q4 — Beat/plant mismatch: d1 fires at t2 before Mrs. Arbuthnot is introduced (t9);
    d2/d3/d4 beats reference Wattle/committee/paste-jar that never appear in-story.
  - Q5 — Verbatim quoting of Prudence (turns 6, 9): all M1/M3 frames present but ineffective;
    guide writes quotable aphorisms; engine anti-echo list targets wrong phrases.
  - Q6 — 3 paragraphs vs 2 (turns 8/12/13/18): minor, accepted model limitation.

## Workstreams

### Performance
- [x] P-P1 (engine): dedupe + cap guard on `add_new`. `_filter_new_entry_names` rejects
      exact/near-duplicate names against existing state + caps batch (8) and total (120).
      Mid-scene discovery preserved (user choice: keep `add_new` on `always`). Verified:
      fabricated micro-elements rejected live ("Prune", "The Larder", "The Cooling Rack"),
      elements stayed 9→13 across 2-turn smoke (was +74/+43 spikes).
- [x] P-P2 (engine): `_perform_branch_query` now treats NO/UNCHANGED/NO_UPDATES_REQUIRED/"n"
      (+prefix NO) as a negative → skips the wasted apply_updates call.
- [x] P-P3 (engine): negative branch query short-circuits recursion — `_execute_action`
      returns (stop, gate_failed, skip_children); `_update_recursive` skips drilling into
      sub-branches after a NO verdict. Fired live in smoke.
- [ ] (Optional, harness) per-step call histogram in result.json; `dump.txt` append-only.
- Measured on 2-turn smoke: turn-0 82→51 calls (incl. population), turn-1 delta 46→33,
  element spikes eliminated. Expect ~30-40% on normal turns + spike collapse on the 20-turn.

### Quality
- [x] Q-P1 (engine, HIGH): gate `_populate_from_first_scene` on the `.populated_from_first_scene`
      marker (written to both last+new history dirs). First-scene population now runs exactly
      once per session — the t17 re-population that wiped memory (elements 206→18, Evelyn/Mrs.
      Winters conflation, c13 loss) can no longer fire.
- [ ] Q-P2 (engine): director anti-repetition rule — forbid the instruction block from
      dictating imagery copied from prior replies / user input.
- [x] Q-P3 (engine/harness, DONE 2026-08-12): memory-surface step — the reply prompt now
      carries "Draw naturally on the characters, items, and past events established in your
      system context (General Info, Current Scene, Characters, Events sections)" so DSS
      surfaces its stored memory instead of just storing it. Applies to both the do_instr
      and do_instr=False branches in generate_instr_prompt (summarizer.py) + the level-1
      prompt in long_horizon_soak.py. The contamination guard is UNCHANGED — note text is
      still never injected; recall must come from DSS's own stored state. Live smoke on the
      35B (seed 8001, 3 turns): directive present in all instr_prompts; replies now weave in
      established items/characters/events across turns (Le Creuset dish, oven-door, Agnes's
      handling, Constable Finch's notebook, Mrs. Ogilvy). Also fixed: `--judge-every 0`
      crashed the turn loop (`turn % 0` ZeroDivisionError at long_horizon_soak.py:1283/1396)
      — both judge sites now guard `args.judge_every` (0 = disabled, matching audit_every).
- [ ] Q-P4 (spreadsheet): re-align dss_beats with in-story plants (fires after character
      intro; ensure sources exist).
- [ ] Q-P5 (engine): strengthen anti-echo n-gram overlap check for verbatim quoting.

### Second soak run (cozy_mystery__5b95ec83, seed 18470, 2026-08-11)

Still slow AND auditor judged "missing" everywhere. Two findings:

### Perf: the per-entry query+apply pair dominates (~48%)
Measured over 9 turns (706 local calls): characters per-entry 99 queries (~11/turn),
groups 66 (~7/turn). Each `query_branch_for_changes` on an entry = **TWO LLM calls**
(query "does it need updates?" + separate update list, data_summarizer.py:1643+1706).
The query returns YES almost every turn in a live story → the pre-filter is wasted
overhead. Plus current_scene.now drill-downs (~13-24 perform_update calls/turn,
unconditional), elements add_new generates (~5/turn), gates (~4), general_info (~6).
Steady state ~65-85 calls/turn → ~9-10 min/turn.

Fix options (ranked):
- **E1 (engine/schema): change Character/Element/Group entry triggers from
  `always: [query_branch_for_changes]` to `always: [perform_update]`** — single
  self-detecting call per entry (the full-branch update template already supports
  NO_UPDATES_REQUIRED). Halves entry cost: ~36 → ~18 calls/turn. Schema-driven.
- **E2 (engine): skip current_scene.now child drill-downs when the who/when/why
  content is byte-identical to the previous turn's snapshot** (deterministic, no LLM).
  Saves ~13-24 calls/turn on static scenes.
- **E3 (engine): lower ADD_NEW_MAX_PER_CALL 8 → 4** — saves ~2-4 calls/turn.
- Combined ≈ 40-55% (≈3h → ~1.5h).

### Auditor/judge/guide were BLIND to most of DSS's memory (harness bug — FIXED)
`_memory_summary` dumped `json.dumps(entries)[:2500]` — a raw JSON prefix. With 9
characters (15,813 chars) the prefix showed only Evelyn + half of Prudence, and
groups/elements/events/arcs weren't included at all. So the auditor judged "missing"
on state that WAS saved (turn-4 state: 9 characters, 6 groups, 35 elements, events
present). FIXED: `_memory_summary` now renders a compact full inventory — every entry
name with ~160 chars of detail (name-only fallback if too big), plus events/arcs.
Judge's memory_fidelity + guide's planning were undercut by the same truncation.

### Jinja crash `'list object' has no attribute 'items'` (engine — FIXED)
Events format template did `category_data.items()` on `data.past/scenes/events`;
when the LLM wrote one of those as a list (schema says dict[str, StoryEvent]) the
render raised. It was CAUGHT (returned ""), silently dropping the events context.
FIXED two ways: (1) the events template now guards `category_data is mapping` and
renders list categories as indexed entries; (2) `_render_jinja_template` falls back
to a compact raw-JSON dump on any render failure so no subject is ever dropped.

## Tracking
- [~] Re-run cozy_mystery 20-turn after E-fixes → measure calls/time vs baseline.
- [ ] Re-run and watch judge curve for the 3/5 → goal.

## Open questions for user
- P-P1a semantics: OK to restrict element/group `add_new` to scene starts, or keep
  mid-scene discovery with a dedupe/cap guard instead?
- Priority order to implement?

## Parallel subject processing (`max_subject_workers`, engine + harness — DONE, 2026-08-11)
The server-side batching blocker was cleared (see `--disable-prefix-caching` note below),
so the subject loop can now run concurrently. Engine side (`summarizer.py`):
- New config `max_subject_workers` (default **0 = serial, production byte-identical** unless
  someone opts in). `dss_config.json` may set it; the soak harness overrides via
  `--max-subject-workers` (wired into `_make_summarizer`).
- Subject loop → `ThreadPoolExecutor`. Each worker runs its own `DataSummarizer` with a
  **deep-copied `custom_state`** and a hybrid `all_subjects_data` snapshot (deep copy of all
  subjects + the live current subject swapped in), so the post-loop chapter/arc/message
  summary reads the fully-updated state without merging.
- The `PhaseManager` is shared behind one RLock wrapper (`_LockedPhaseManager`); only
  mutating methods (`start_phase`/`end_phase`/`fail_phase`/`warn_step`) take the lock, so
  attribute reads stay unlocked (no UI latency). Each subject still writes its own
  `<history_path>/<subject>.json` (per-subject dicts are process-local, so writes don't collide).
- `processed_subjects_data` reassembled in schema order; post-loop chapter/arc work uses the
  shared `data_summarizer` whose `all_subjects_data` reflects every subject's final state.
- Bug caught in verification: `DataSummarizer` was imported *locally* inside
  `summarize_latest_state` (to break a circular import — `data_summarizer` imports `Summarizer`
  at module level), so the new `_process_subject_parallel` worker hit `NameError`. Fixed by
  re-importing it inside `_process_subject_parallel` (same deferred-import pattern).

Measured on a clean 2-turn Level-2 smoke (cozy, seed 1003, 4 workers, server with
`--max-batch-size 8 --disable-prefix-caching`):
- Turn 0 (population-dominated): **406s / 51 calls** vs serial baseline 650s / 82 calls.
- Turn 1 (pure subject loop): **128s / 26 calls** vs serial baseline 230s / 46 calls (~1.8x).
- All subject files written to every turn's history dir; current_scene populated
  ("Evelyn and Prune at the fete…", "The village green"); reply stays third-person Evelyn.

Caveat (pre-existing, environmental): the soak's SIGINT/SIGTERM handler sets `soak._stop`,
which the engine checks as `runtime.stop_everything`. If a signal arrives DURING first-scene
population, population aborts and writes **empty `{}`** for the remaining subjects
(`summarizer.py:2187`) and the run continues gutted. Seen once when a shell `timeout` fired
mid-run. Not introduced by threading; worth hardening later (abort the whole turn on a
population stop rather than silently writing empty subjects).

## `--disable-prefix-caching` (LMDeploy server, DONE — see textgen fork notes)
Crashed the server under 4+ concurrent requests (BlockTrie/SequenceManager race in the
shared prefix-cache path). With prefix caching OFF, concurrency is stable: 8 concurrent
= 5.2s vs 9.6s serial (~1.85x, 3.1x for identical prompts). Flag is additive; default stays
caching-on. Restart server with `./strt mp --max-batch-size 8 --disable-prefix-caching`.

## lmdeploy v0.15 empty-completion-under-concurrency bug (FIXED in harness, 2026-08-11)
The v0.15 engine occasionally returns a **genuinely empty completion** (`content=""`,
`reasoning_content=""`, `finish="stop"`) under concurrent load. Measured 1/24 calls under
4-way threads; sequential calls never reproduce it. Impact on the soak:
- An empty **branch query** was treated as "changes needed" → cascaded ~10 direct-update
  calls (waste) whose per-field responses were also empty → **current_scene never updated**
  even though the story moved (query said YES 7/11 turns, yet current_scene stayed
  byte-identical across turns 0-9; the oak-tree scene never reached memory).
- An empty **field update** hit `jsonc.loads("")` IndexError (`data_summarizer.py:2047`) →
  caught → update silently dropped (32 occurrences in the 20-turn run).

Fixes (harness + engine, land next run):
1. `LocalModel._complete` (long_horizon_soak.py) retries up to 3x with backoff when both
   content and reasoning_content are empty. The engine bug is probabilistic (~4%), so a
   retry catches it without slowing normal calls.
2. `_generate_field_update` (data_summarizer.py) guards `not text.strip()` → clean retry
   (`last_error="The model returned an empty response."`) instead of the jsonc IndexError.
3. `_perform_branch_query` negative-detection now treats an **empty** response as negative
   (skip), so a blank verdict no longer spawns the direct-update cascade.

Note: fixes do NOT affect the in-flight `cozy_mystery__50761bf6` run (code already loaded);
that run stands as the perf baseline + a documented demonstration of the memory staleness.

## Audit misses in the seed-77 run: three-layer finding (2026-08-11)
1. DESIGN: `--audit-every N` polls ALL beats every N turns — d8 (turn 24) / d9 (turn 30)
   are beyond a 20-turn run and always "missing"; d2-d7 get checked before their due
   turns. Beats-only mode (audit_every=0) evaluates each beat only at its due turn and
   gives per-beat variety. Consider: in audit-every mode, only evaluate beats whose
   due turn has PASSED.
2. REAL save failures (auditor correct): Mrs. Arbuthnot saved turn 2 (wrong role) then
   LOST by turn 12 (characters rewritten to Prune/Evelyn/Mr. Beech/Constable Whitlow —
   the Q-P1 first-scene population re-run / wholesale replacement); Marmalade the cat
   (mentioned 2x) and crooked oak (mentioned) never saved; events subject EMPTY all run.
   Root cause = v0.15 empty-completion bug dropping add_new/field-update saves.
3. BEAT ALIGNMENT (Q-P4): quince paste jar + Reverend Wattle appear 0x in-story — the
   guide never plants them; those beats are unsatisfiable as written.
Dashboard: `_run_audits` now returns the FULL audit log (all verdicts, per-turn wall
time) instead of collapsing to last-verdict-per-beat (which showed every row at the
viewing turn under audit-every polling).

## df2fce53 (35B, seed 79) all-audits-missing: world-empty root cause (2026-08-12, FIXED)
The 35B run's audits were ~all "missing" because the CHARACTERS/GROUPS/EVENTS subjects were
EMPTY the entire run — verified in the real history dirs, not just snapshots (turn 0 had
characters 0 / elements 0; a same-day smoke control populated 5 characters + 8 elements).
Root cause (engine bug, summarizer.py): `is_new_scene = True` was only set in the initial-
world CACHE-MISS branch. On a cache HIT it stayed False → flowed to
`SummarizationContextCache(is_new_scene_turn=False)` → `_populate_from_first_scene` gate
(self.last.is_new_scene_turn) never passed → no first-scene population. The cache only ever
holds EMPTY placeholders + seeded general_info (population writes to the session path, never
back to the cache), so a cache hit always needs population. df2fce53's first launch crashed
(Broken pipe) AFTER writing the cache but BEFORE population; the restart hit that cache and
skipped population → gutted world; only `add_new` (elements) trickled in 5 items late. Fix:
set `is_new_scene = True` in the cache-hit branch too (both branches now set it; compile +
hermetic green). Second layer: the beat EXPECTATIONS partly didn't match the story the guide
actually wrote (d2 quince paste jar / d3 bake-off committee / d4 Reverend Wattle appeared 0x
in-story; DSS saved "Wooden Quince Press" instead) — Q-P4. Third layer: wrong-subject routing
(Mrs. Arbuthnot + Marmalade Cat saved to ELEMENTS, never to characters — characters
add_new never fired on the 35B). The seed-80 run (cozy_mystery__418b4693) is a clean
cache-miss population and will produce the real audit picture for the fixed code.

## Wrong-subject routing fix: dynamic subject guide in add_new queries (2026-08-12, DONE)
Mrs. Arbuthnot + Marmalade Cat were saved to ELEMENTS, never characters — the elements
`new_entry_query_prompt_template` was a permissive general fallback ("Elements include
creatures... Any new element should be added") with no routing constraints. Fix (3 parts):
(1) subjects_schema.json gained a top-level `subject_routing` map — one-line per-subject
description of what belongs where (data-driven, no code hardcoding; SchemaParser parses it
into `subject_routing`). (2) DataSummarizer._build_subject_routing_guide(subject_name)
composes a dynamic "SUBJECT ROUTING" block at runtime from the schema (lists every subject +
description + a "<<< YOU ARE NOW POPULATING THIS SUBJECT >>>" marker on the current one +
a route-correctly rule), prepended to every add_new query prompt in
_detect_and_add_new_entries_to_branch. (3) Templates tightened: elements query now has an
explicit "WHAT NOT TO INCLUDE" (no named people/speaking beings → characters; no orgs/committees
→ groups; no narrative events → events; creatures only as props/background fauna); elements
entry creature-kind tightened ("does not speak or recur as a story participant"); characters
query REMEMBER now states named animals/pets count as characters. Verified: schema JSON valid,
compile OK, guide composes with correct current-subject marker, hermetic suite green. Applies
to the NEXT run (running seed-80 process has the old schema copy).

## cozy_mystery__9284c44d (35B, seed 81) — analysis + strict-rating rubric (2026-08-12)
Run: 20 turns, 842 local calls, judged 4/5 under the OLD rubric (user: "should be ~2.8-3/5").
Root causes (2 subagents, /tmp/opencode/soak_9284c44d_facts.md):
- F1 IDENTITY CONFLATION (memory_fidelity 2 at 3/5 judge turns): Evelyn's stored bio became
  Prudence's ("Retired librarian turned amateur sleuth. Niece of the florist..."). Seeded at
  turn 0 by _populate_from_first_scene (feeds the first exchange verbatim; Evelyn's own reply
  claimed the bakery/sleuth identity; population prompts have NO name1/name2 role binding).
  Compounded at turn 12 by the Character skip_query branch_update_prompt_template (renders the
  whole characters list + latest exchange, no canonical-identity anchor). Spreadsheet
  characters.name1/name2.role are HARNESS-ONLY; after the first scene archives
  retrieve_and_format_context drops the premise/roster from context entirely.
- F2 SAVE FAILURES: (a) characters add_new is gated behind on_new_scene (only ONE scene
  transition in 20 turns) — the model was never ASKED about Mrs. Arbuthnot (in-story turns
  1/2/7/14+). (b) elements add_new is ordered AFTER the gate check and a negative gate
  short-circuits the whole branch — elements.json was byte-identical for 12+8 turns, so the
  turn-11 watering can never landed. (c) population guard hole: .populated_from_first_scene
  marker only in last+new history dirs → population RE-RAN at turn 12, wiping characters
  (Queen Victoria dropped, Marmalade became "Evelyn's cat").
- F3 RECALL 0% = METRIC BUG: all 13 recall notes have recall.due ∈ [62,78] — beyond the 20-turn
  run, so no recall probe ever fired; report.py counts never-due notes as failures.
- F4 Q-P4: d2/d3/d4 beats (quince paste jar, bake-off committee, Reverend Wattle) reference
  subjects never planted in-story — unsatisfiable as written.
- F5 scene inconsistency (Mrs. Pemberton present when left) + repeated closing imagery.
- F6 harness: turn-15 judge JSON rejected ('judge output was not JSON') — lost judge point.
STRICT RUBRIC SHIPPED: JUDGE_SYSTEM_TMPL + FINAL_OVERVIEW_SYSTEM_TMPL now carry a SCORING
RUBRIC with per-dimension anchors + MANDATORY FLOORS (identity confusion caps memory_fidelity
≤2 and overall ≤3; >half notes missed caps quality ≤3 and overall ≤3; verbatim re-quote caps
style ≤3; scene inconsistency caps quality ≤3). overall_score = weighted (voice 30 / memory 25
/ notes 25 / progression 20), not a straight average, and must be consistent with the per-turn
judge records. Backfilled this run: 4/5 → 3/5, matching the user's read.
PROPOSED FIX BATCH (pending approval): A1 characters add_new into always triggers; A2 don't let
a negative map gate short-circuit add_new (reorder ElementMap triggers); A3 session-wide
population marker (in general_info.json) so population runs exactly once; A5 re-scope
new_entry_query_prompt_template to "latest exchange(s)"; B1 move recall.due inside the run
horizon; B2 report.py counts only due-and-reached recalls; + identity anchor (canonical roster
in general_info + name1/name2 binding in population + character update prompts).

## 16. Audit capability awareness + scene-part budget + elements gate + population marker (2026-08-12)

User approved: "proceed with all #1-2+A2-3".

**#1 — Auditor capability awareness (harness, long_horizon_soak.py).** `_run_audit` now computes a
deterministic per-beat "DSS CAPABILITY CONTEXT" from run data via `_audit_capability_context(beat, turn)`:
which turns each subject's JSON actually changed, which turns were scene transitions (characters add_new
gating), and whether the beat's expected entity ever appeared in the story by the audit turn. Injected
into `build_audit_messages`; AUDITOR_SYSTEM_TMPL now has a 5th status `unassessable` (entity never planted
or subject frozen/gated → not a DSS failure; only 'missing' when it appeared AND was updatable). report.py
`_audits` + markdown summary + dashboard tables pass the new status through (counted separately).

**#2 — Hard message-budget scene split (engine, summarizer.py).** New config `max_scene_part_messages`
(default 12; dss_config.json override; soak `--max-scene-messages`). Tracks `_scene_start_message`
(new int field on CurrentScene schema, engine-meta like `_scene_number`; initialized on population turn,
set on every scene start). In the auto-scene-detection block of summarize_latest_state: if
msgs_in_part >= budget, force `is_new_scene_turn=True` (+ auto flag) and log "Budget scene split" — the
LLM transition check is skipped that turn. This fires the on_new_scene triggers (characters/elements
add_new, scene archive) on a regular cadence. Live smoke (budget 4, 3 turns): 2 splits fired (scene 1→2→3),
characters add_new fired at turn 1 and saved Mrs. Arbuthnot (d1 audit = saved). NOTE: current_scene `what`
is not regenerated on splits (pre-existing: no update trigger on the field), so scene-parts archive under
the same title → events.scenes dedups parts into one entry. Scene NUMBER still advances. Follow-up:
on_new_scene-triggered `what` regeneration (or "(Part N)" labeling) for distinct archives.

**A2 — Elements gate (schema).** ElementMap `always` triggers reordered `[perform_gate_check, add_new]` →
`[add_new, perform_gate_check]`, so add_new runs every turn even when the gate would answer NO (the
watering-can class: new entity present but section gate froze the whole branch). GroupMap left unchanged.

**A3 — Session-wide population marker (engine, summarizer.py).** `.populated_from_first_scene` guard now
also checks + writes a `_populated_from_first_scene: true` key inside general_info.json (written at
population time, carried forward every turn by the GeneralInfo subject). This fixes the turn-12 population
re-run that wiped accumulated memory when the marker FILE wasn't in that turn's history dir.

VERIFIED: hermetic suite green; report builds with `unassessable`; dashboard builds both run shapes;
live 3-turn smoke (cozy_mystery seed 8002, budget 4): budget splits fire, characters add_new on split
saved Mrs. Arbuthnot, marker persists (`_populated_from_first_scene: True` in every turn's general_info),
capability context correct (d1 assessable → 'saved'; d2 unassessable facts: quince paste jar never planted).

## 17. A5 + A6 + B1 + B2 batch (2026-08-12)

The remaining queued fixes from the seed-81 analysis, approved as one batch.

**A5 — add_new discovery scope (schema, subjects_schema.json).** Re-scoped the five
`new_entry_query_prompt_template`s (CharacterMap, GroupMap, ElementMap, PastEvents, CrucialEvents)
from "Throughout the last scene" to "In the latest exchange(s) (the most recent user and assistant
messages, whether or not a scene boundary has been crossed)". Same-turn entities (the watering-can
class) are now catchable even before a scene boundary. The Scenes archive template stays scene-scoped.

**A6 — identity anchor (engine + harness).** The Evelyn→Prudence identity conflation (F1) is now
explicitly bound in the prompts. (1) `_create_update_prompt` (data_summarizer.py) gained clean
`name1`/`name2` format keys alongside the legacy `{user}`/`{char}`. (2) IDENTITY BINDING block added
to five character templates: Character.branch_update_prompt_template + branch_query_prompt_template,
CharacterMap.new_entry_prompt_template, Characters initial_population identification + population
prompts — "{{ name2 }} is the AI character being roleplayed (protagonist); {{ name1 }} is the other
principal character; never assign {{ name1 }}'s identity/occupation/biography/voice to {{ name2 }}".
(3) summarizer.py passes name1/name2 into the population/identification prompt formatting (4 sites).
(4) Harness build_state context gained a "Canonical cast:" line; _seed_general_info_style also seeds
`_cast` (name1/name2 + role + description) into general_info.json; format_templates.json general_info
gained a conditional "Cast ---" line rendering _cast when present (N/A otherwise — production safe).

**B1 — recall horizon clamp (harness).** SoakRun._clamp_note_windows (called in __init__) clamps
every note's recall.due and plant.turn to `turns - 2`, so short runs actually fire the recall
probes (seed-81's 13 echo notes had due 62-78 → never fired). Spreadsheet files untouched.

**B2 — report.py counts only due-and-reached recalls.** `_recalls` adds a `reached` flag
(due < len(results)); the echo/recall success denominator now counts reached recalls only —
never-due notes no longer drag recall_success to 0%. Markdown table gains a "reached" column.
Dashboard unaffected (reads per-turn probes only).

VERIFIED: hermetic suite green; bindings render with real names (Prudence/Evelyn) in all five
templates; B1 clamp (13 due 62-78 → 18); B2 against seed-81 run (13 rows, all unreached → excluded,
recall_success no longer penalized); compile + schema valid.

## 18. Scene-turn visibility + auditor savable-hint (2026-08-12)

User asked whether scene transitions fired on the seed-82 run and to note scene turns in the dashboard.

Finding: scene-part budget splits DID fire — transitions at turns 5 (1→2), 10 (2→3), 13 (3→4), 15 (4→5)
on cozy_mystery__2502c5ad. Mrs. Arbuthnot was saved as a character at the turn-10 transition. Yet the
auditor marked everything "unassessable" for two reasons: (a) genuine — d2/d3/d4 reference entities
the guide never planted (Q-P4, unsatisfiable as written); (b) over-flagging — the d1@10 verdict
ignored that both the transition (turn 10) and the in-story appearance (turns 8-9) had landed by
then, and marked unassessable based on the beat's scheduled turn (2) instead of the evaluation turn.

Changes:
1. soak_dashboard.py — new `_scene_turns(run_dir)` helper reads each turn's
   state_snapshot/current_scene.json `_scene_number` and returns {turn: (from, to)} for transitions.
   `_build_turn_accordion` now tags transition turns with "· ⛨ SCENE n→m" in the accordion label;
   `_run_display` prints a "**Scene turns** (scene-part budget splits): turn 5: scene 1→2, ..." line
   under the stat cards.
2. long_horizon_soak.py `_audit_capability_context` — name/hit_turns hoisted so both the characters
   block and the entity block can use them; characters beats now get an explicit deterministic
   conclusion: "DSS WAS able to save '<name>': a scene transition occurred at turn(s) [X] on or
   after first in-story appearance (turn N). The beat is ASSESSABLE — if not saved, mark 'missing',
   not 'unassessable'." This prevents the cloud auditor over-flagging satisfiable beats.

## 19. Dashboard bugs + dark mode + help tooltips (2026-08-12)

Two bugs fixed:
1. Dashboard loaded with zero runs stayed on "No runs" even after runs started —
   the `@gr.render(inputs=[run_menu])` block was only registered when `choices`
   was non-empty. Now it is always registered; the empty state ("No run dirs yet,
   run the soak then Refresh") renders inside the block, so new runs appear on
   Refresh and are selectable.
2. "Local ctx (prompt tok)" always 0 — lmdeploy v0.15's /chat/completions returns
   `usage.prompt_tokens: 0` (verified live). Fixed twice: LocalModel._complete now
   estimates prompt tokens from the actual message payload (chars/4) when the server
   reports 0 (future runs), and _usage_series falls back to an instr_prompt estimate
   when stored local_usage.prompt_tokens is 0 (retroactive for existing runs).

Enhancements (soak_dashboard.py):
- Dark mode: 🌙 Dark mode toggle button (persisted in localStorage); _DARK_CSS
  overrides gradio's CSS variables under `body.tg-dark` (bg, text, tables, inputs,
  panels, links) plus !important fixes for the hardcoded light stat-card colors.
- Help tooltips: HELP dict defines every term (retention, guide_failure,
  dss_retention_loss, superseded, style, fidelity, quality, scene_part, recall,
  unassessable). Stat/live cards carry native title= tooltips; judge curves,
  retention, recalls, probes, judge-result, audit and DSS save-beat tables carry
  gr.Dataframe info=; the Scene turns line includes a scene_part definition.

## 19b. Dark-mode fixes round 2 (2026-08-12)

Two issues after the first dark-mode pass:
1. "No tables" — gr.Dataframe in gradio 4.37 does NOT accept `info=` (TypeError at
   component construction). The whole gr.render body aborted at the first Dataframe,
   so no tables rendered. Fixed: removed all `info=` kwargs and added a `_help_html()`
   marker — a muted "ⓘ caption" line with a native hover tooltip (title=) above each
   table (judge curves, retention by type, per-note retention, judge result, probes,
   recalls, per-turn audit, run-level audit).
2. White dropdown/refresh — gradio 4.37 uses `--background-fill-primary`,
   `--background-fill-secondary`, `--table-even/odd-background-fill` and
   `--button-secondary-*` (NOT the names I had set). Rebuilt _DARK_CSS against the
   real 204-variable theme list (verified via gr.themes.Base()._get_theme_css()).
Verified: `_run_display` executes through every table build with no info= TypeError;
dashboard relaunched clean (HTTP 200, no log errors).

## 20. Big ideas for consideration (user-raised, 2026-08-12 — NOT yet acted on)

Parked design directions from the user. None implemented yet; they are listed here
with how each connects to existing work.

### 20.1 Populate `events` more aggressively
`events.json` holds only the single initial "The Dawn Baking" event across the whole
run — the story's actual occurrences never enter structured memory. Root causes are
the SAME two bugs as the elements freeze (§20.4): the events `new_entry_query_prompt_template`
carries the `NO_NEW_ENTRIES_REQUIRED` escape and events entries lack the update
templates. Fixing the elements batch (§20.4) should unblock events too — verify
events.json changes per turn afterward. Beyond the bug fix, consider whether PastEvents
vs CrucialEvents scoping is right for retention of mundane occurrences.

### 20.2 Spreadsheet-specific schemas (per-genre schema tuning)
Schemas are the tuning surface: per-genre `subjects_schema.json` variants. A detective
story needs exponentially more detail retention than most genres, so the detective
variant should raise add_new sensitivity (weaker NO-escapes, higher caps), make
`branch_list` grounding richer, tighten element trait specificity, and possibly lower
the `max_scene_part_messages` budget so objects get saved more often. Generalization
rule (#35): a schema is DATA — the engine must stay data-driven; swapping
`subjects_schema.json` per spreadsheet is configuration, not new code. Consider wiring
the soak to pick `spreadsheets/<genre>_schema.json` when present, falling back to the
default. Keep the schema diff minimal per genre and measure retention deltas.

### 20.3 RAG is untested — needs a long-horizon soak or synthetic history
DSS's RAG path has never been properly exercised. A real test needs ~50 messages
(long-horizon soak) to cross the scene-archive boundary repeatedly and force real
retrieval decisions. Two test vectors:
- Extend `--turns` for a dedicated 50-turn run (cost: cloud calls ~2-3x a 20-turn run).
- Synthetic chat history: inject a prebuilt multi-scene `history.json` + warm state
  dirs so a short run still drives heavy retrieval. Cheaper, reproducible.
Expected findings: retrieval hit quality across scene archives, element/event recall
under context pressure, and whether `last_x` scene-bounded windows surface the right
plants. See also the `context_retriever` note in §20.5 about old plants dropping out
of context when the scene moves.

### 20.4 Elements/objects memory freeze — engine batch (IMPLEMENTED, 2026-08-12)
Confirmed root causes (subagent analysis, 2026-08-12, verified against schema + run):
1. `NO_NEW_ENTRIES_REQUIRED` escape in ElementMap/GroupMap/events
   `new_entry_query_prompt_template` (subjects_schema.json:839) → the model answers
   "no new entries" every turn (ablation verified: removing the sentence flips it to
   proposing entries). Characters template says "empty array acceptable but not ideal"
   → characters is the only subject that grows.
2. `Element` is the only entry class missing `branch_update_prompt_template`
   (subjects_schema.json:796-815) → its `always: [query_branch_for_changes]` is
   silently skipped (data_summarizer.py:396-402 `if bq_template and bu_template`).
3. `branch_list` renders `<EMPTY>` (data_summarizer.py:2501-2503 builds
   `FormattedData(data, "elements.entries_list")`, no format template) → add_new
   queries show zero existing entries to the model.
4. `GroupMap` triggers are gate-first `[perform_gate_check, add_new]` → a NO gate
   short-circuits add_new (groups freeze at 2 entries). Reorder to match ElementMap.
DONE 2026-08-12 (verified: schema parses Element with both templates, ElementMap/GroupMap/
CrucialEvents/PastEvents NO-escape removed, GroupMap triggers reordered, branch_list renders
real entry keys in the add_new prompt — unit-checked; hermetic suite green; live 2-turn smoke
cozy_mystery__940a87d2 in flight).

### 20.5 Reply quality vs memory quality — isolate generation from summarization (synthetic chat)
Design to isolate DSS's two failure surfaces: in "synthetic chat" mode the cloud
model (guide) writes BOTH Prudence's AND Evelyn's turns; DSS ONLY runs DataSummarizer
(memory updates), never generates replies. Then compare: if memory still degrades,
the summarization is at fault; if memory is fine but the full-run replies were bad,
the failure was the local generator. This directly tests the 4/5 vs 3/5 question.
Harness shape: `--synthetic-chat` flag; guide prompt gets an extra "also write
Evelyn's reply for this turn" duty; skip the local reply call + its judge; keep
probes/recalls/audits against the guide-written Evelyn. Also a precursor RAG test
(could combine with 20.3): run DSS with RAG on/off and diff memory fidelity.

### 20.6 Dashboard: local model actual context-size usage per stat
Per-turn detail should show the local model's REAL context-size usage (the window
sent per call), not just token in/out. We already record per-turn prompt/completion
deltas and "current ctx sent" on the live stat cards; the ask is to surface it
per-stat in the per-turn detail section (e.g. a "ctx size (tok)" column / line per
turn showing the largest prompt window used that turn, from LocalModel's last
prompt-token measurement). Cheap to add once the p=0 estimate fix lands in a real run.

### 20.7 Current model is Qwen3.6-35B-A3B (not 9B)
Reference for all future writeups: the local writer under test is the 35B MoE
Qwen3.6-35B-A3B-abliterated-AWQ (3B active/token, 256 experts), served by LMDeploy
v0.15 TurboMind. The earlier 9B-era baselines (seed-77 cozy 3/5, etc.) are the
comparison points, not the current engine's characteristics. Any analysis that says
"the 9B model does X" for current runs is wrong unless it quotes a 9B-era run.


## 21. Schema types + synthetic-chat isolation + dashboard peak-ctx (2026-08-12, DONE)

Test conditions are now a first-class dimension — see
`schema_types_and_test_matrix.md` (full matrix + schema resolution order).

- **Schema-type mechanism**: `--schema-type 1|2|3` (+ `--schema <path>` override)
  in the soak harness; `Summarizer.config["subjects_schema"]` picks the schema
  file (runtime.extension_dir-relative). Type 1 = current default
  `subjects_schema.json` (incremental-per-turn, unchanged baseline). Type 2 =
  `user_data/example/schemas/subjects_schema_sceneagg.json` — events + current_scene
  every message, everything else on new-scene turns with an aggregated SCENE
  CONTEXT block ({{ scene_recap }} + {{ scene_events }}, injected into 19 b-pass
  templates). Resolution order: --schema, per-genre+type, per-type, per-genre,
  legacy default. `schema_type` is part of the run-id hash (fresh run dirs).
  The two aggregation format vars are lazy lambdas in `_create_update_prompt`
  (data_summarizer.py) — zero cost unless a template references them.

- **REAL events-freeze root cause (found while smoke-testing Type 2, 2026-08-12, FIXED in BOTH schemas)**: the DataSummarizer traversal resolves each events-branch field (`past`/`scenes`/`events`/`chapters`) through its wrapper alias to the shared `StoryEvents` dict class — so the per-branch `new_entry_query_prompt_template`/`new_entry_prompt_template` living on the wrapper aliases (CrucialEvents/PastEvents/Scenes) were UNREACHABLE dead code. `_detect_and_add_new_entries_to_branch` requires both templates on the class it actually reaches (StoryEvents) and skipped when missing (data_summarizer.py:514) — that is why events.json never populated outside first-scene init, on EVERY run. Fixed by adding generic `new_entry_query_prompt_template` + `new_entry_prompt_template` to StoryEvents defaults. Verified hermetically (sceneagg smoke: events add_new query now dispatches) + full suite green.
- **Type 2 trigger map**: Events + StoryEvents.add_new → ALWAYS; StoryEvents
  per-event query → ON_NEW_SCENE; CharacterMap/GroupMap/ElementMap and the
  Character/Group/Element per-entry queries → ON_NEW_SCENE; CurrentScene stays
  ALWAYS. Note: events.json only ever populated at first-scene init because the
  Events gate answered NO on every transition — Type 2 removes the gate so the
  StoryEvents add_new fires every message.
- **20.5 synthetic-chat isolation** (`--synthetic-chat`, Level 2 only): the cloud
  guide writes BOTH sides; DSS runs only summarize_latest_state (no local reply);
  judge gets a SYNTHETIC-CHAT-ISOLATION note to score memory/note-retention only.
  Isolates memory/summarization flaws from the local writer's prose.
- **20.6 dashboard peak ctx**: LocalModel.stats() now returns max_prompt_tokens
  (peak single-call prompt, reset per stats() snapshot) + ctx_size; stored in
  per-turn local_usage; the per-turn accordion shows "peak ctx X/32768 (N%)".

## 21. Rolling overviews (--overview-every) — shipped 2026-08-13
Opt-in mid-run judge pass: `--overview-every N` writes `rolling_overview_XXX.json` (turn-count suffix) every N turns, covering {name2}'s performance so far with an extra `trajectory` (improving/stable/declining) field. Same FINAL_OVERVIEW_SYSTEM_TMPL with a MID-RUN preamble; 6000-token budget, reasoning_effort="none". Report gets a "## Rolling overviews" section; dashboard gets a "Rolling overviews (mid-run judge)" accordion (one sub-accordion per checkpoint: score/trajectory/timestamp). Verified hermetically + in-process render tree (174 accordions incl. rolling).

## 22. seed-89 diagnosis (cozy_mystery__4063405d, stopped t15) — subagent analysis
PERF: (a) elements = sole critical path, ~2×N calls (query+update, skip_query absent) = ~90/45 entries at t15; (b) O(N²) prompt growth is the real superlinear driver: every per-entry call embeds the ENTIRE elements map (32KB→143KB) + growing internal history (100-153KB chars); per-call latency 7.5s→12s; max_prompt_tokens 54856 > ctx 32768 → server truncation wastes prefill; (c) events on_new_scene archival = O(accumulated state): 78-150 calls on scene turns (events worker 1176s at t10); (d) current_scene ~13 calls/turn constant, general_info 8-11. Element skip_query (shipped in the elements-fix batch) halves element calls but NOT per-call latency → t20 still ~14-15min. HIGHEST-LEVERAGE FIX: scope per-entry context to target entry + compact sibling index + bound internal-history payload (50K→8-12K tok/call, ~3s/call → t20 ~4-5min).
QUALITY: (a) note-surfacing is instruction-steered not recall-driven — the per-turn instruction block names 2-3 notes, the judge grades against ALL in-scope notes (incl. never-planted); (b) d1 beat STALE (expects Arbuthnot="constable's wife" — leaked Evelyn's constable-connection; story makes her the society chairman culprit); d5 unplanted by guide (watering can 0 hits); d3 "pie tent access" never planted; (c) entity-fitting defects: d2 "quince paste" vs "quince paste jar" (location dropped), d3 committee = verbatim duplicate of society group; (d) t15 glove possession slip seeded t14, propagated (guide assumed possession); (e) retention attribution bias — lavender hat WAS planted (t4) but never stored by DSS → genuine DSS loss mislabeled "guide failure"; notes planted beyond run horizon inflate guide_failure; (f) guide truncation t9 (user.txt ends mid-sentence; DSS invented the rose → retcon).

## 23. Importance-weighted detail batch (2026-08-13) — shipped + smoke-verified
From the seed-89 O(N²) diagnosis: every per-entry call embedded the ENTIRE subject map
(mark_field strips only the marker tail, keeping line content) + full internal history,
growing 32KB→143KB per call as entries accumulated. User's design: entity-level
`importance` should gate detail level — high-importance entities full profile, low-
importance one-line roster entries (`element_list`/`group_list` akin to `character_list`).

SHIPPED (both subjects_schema.json + sceneagg variant):
- **A1** entry-level `importance` field (Importance: score/faction/reason) on Character,
  Group, Element, StoryEvent entry classes; `importance_detail_threshold: 50` on the map
  classes; IMPORTANCE directive + BRIEF-detail instruction for minor entries in every
  population/new_entry generation template.
- **A2** `list_template` lookup: new-entry discovery resolves `{item_name}_list` to a
  `list_template` key; added `character_list` (existed) + `element_list`/`group_list`
  compact-roster format templates; _context_order unchanged (roster renders to context).
- **A3** importance-weighted rendering in the format templates: entries with
  importance.score >= threshold render FULL; below threshold render a one-line roster
  entry (name + score + reason + first line of description); legacy entries without
  importance default to FULL (back-compat). Covers characters/elements/groups/events.
- **A4** compact add_new: minor entities (importance < 30) generated with BRIEF detail.
- **A5** internal-history bound: LocalModel gains `--max-update-history N` (default 12);
  DataSummarizer update calls (generate_with_sse / generate_using_tgwui /
  generate_with_streaming-when-bound_history) send only the N newest internal exchanges;
  reply generation (complete) stays FULL history. Unit-verified: update path 8-of-20
  exchanges (keeps newest), reply path full, 0=unlimited.
- **A6** events skip_query: Element/Group/Character per-entry branches now
  `{"action": "query_branch_for_changes", "skip_query": true}` — single self-detecting
  update call (they have branch_update_prompt_template); events map-level scene-turn
  pre-query collapsed.

ALSO SHIPPED (same batch, spreadsheet/auditor/reply quality):
- **B1** stale-beat fixes in cozy_mystery.json: d1 (Mrs. Arbuthnot) — the beat + note c11
  said "constable's wife" (leaked Evelyn's constable-connection) while the story makes her
  the historical-society chairman culprit → aligned to chairman; d2 name "quince paste"
  (story's exact term, was "quince paste jar"); d3 relevance "runs the pie tent" (was
  "access to the pie tent before judging", never planted).
- **B2** NAME EXACTLY directive appended to all six discovery new_entry_query_prompt_templates
  (both schemas): "use the exact name the story uses — do not paraphrase, shorten, or reword".
- **B3** reply-instruction hardening (generate_instr_prompt, do_instr + do_instr=False):
  item 10 bans purely-atmospheric/observational replies + premature conclusions (no "case
  closed", no culprit-naming before reveal) + imagery repetition; item 11 ROLE BINDING
  (name2 = protagonist writing the reply, name1 = the character whose turn just ended,
  never swap / never assign name1's identity to name2); item 8 cleaned (drops the legacy
  1st/2nd-person example garbage, points at general_info.writing_style as authoritative);
  the reply prompt itself gained the same anti-atmosphere/anti-premature-conclusion line.

SMOKE (2-turn, seed 9005, isolated runs-dir): 5 characters + 24 elements populated with
importance scores (Evelyn/Prudence 95, Harold Finch 85, Geraldine Moss 75; elements 20-95);
replies clean third-person; skip_query per-entry update path firing ("Update-only: checking
'elements.entries.X' ... No updates needed"). One flake: Harold Finch population failed
3/3 attempts (missing `]` closing a relationship list — model structural JSON error, not
harness); he still landed in characters.json via a later pass. Turn-1 in progress at
close of day.

## 24. Guide reasoning-leak resilience (2026-08-13) — shipped
seed-90 run (cozy_mystery__ce3305c5) died at turn 9-10: CONTAMINATION guard raised on
note c20. Root cause chain: turns 7-9 outputs were ~12.9K chars (vs ~2K clean for 0-6) —
impossible at the guide's max_tokens=1500, so the text was the REASONING-CONTENT
FALLBACK (cloud_client.py surfaces reasoning_content ONLY when content is empty). The
guide's deepseek-v4-flash burned the whole 1500-token budget on reasoning_content, leaving
content empty; the fallback then returned the giant reasoning blob — which contained a
VERBATIM c20 quote (the guide reasons over the outline notes) — and the contamination guard
caught it. Turns 0-6 were clean because content landed. So the guard trips ONLY when the
reasoning becomes the output (content empty), never when reasoning merely contains notes.

FIXES (shipped):
1. cloud_client.py CloudModel gains `fallback_reasoning` (default True). False (the guide):
   empty content after the doubled-budget retry is a FAILURE → returns "" (never the reasoning
   blob). Judge/auditor keep the fallback (JSON salvage may legitimately use it).
2. Guide constructed with reasoning_effort="none" (kills the reasoning burn at the root;
   replan still passes reasoning_effort="low" explicitly) + max_tokens 1500→2500.
3. New `guide_turn_with_retry(guide, spreadsheet, msgs, label)` replaces all three
   guide.complete call sites (L2 user turn, L1 user turn, synthetic reply): cleans output,
   validates non-empty + no verbatim note content, retries up to 3x with a harness note naming
   the offending note id (never its content), raises RuntimeError only if ALL attempts fail.
4. Post-instr_prompt contamination guard stays as a final backstop (now unreachable for
   guide-origin leaks; still guards writing_style/instructions leaks).

Verified: compile, hermetic suite green, 4-case unit test of the retry (contaminated→retry→
clean; empty→retry→clean; 3x-contaminated→RuntimeError; clean→no retry), live probe of
reasoning_effort="none" (clean 621-char fiction turn, 152 completion tokens, 1 call).
Resuming cozy_mystery__ce3305c5 now runs the new code at turn 10.

## 25. Chapter/arc archival list-vs-dict shape bug (2026-08-13) — shipped
User reported a chapter-archival crash mid-run:
    events_data["chapters"][chapter_data["title"]] = chapter_data
    TypeError: list indices must be integers or slices, not str   (data_summarizer.py:829)
Root cause: the schema declares `chapters: list[Chapter]` (subjects_schema.json:698, Chapters
alias) and the events-population fix now actually populates it as a LIST — but the archive
functions were written dict-keyed-by-title. Only check_and_archive_chapter + check_and_archive_arc
were dict-shaped; every other consumer was already list-aware (events format template, chapters
template, the sceneagg helper at 2587). Once events populated, chapters became a list and every
chapter/arc archival silently failed (caught by the outer try, logged, chapter tracking frozen).
Also fixed a second latent bug in both functions: `current_custom_state` was only defined on the
gate-check branch, so a FORCED transition crashed with UnboundLocalError at the generation call.
Fix: `_entries_as_list()` normalization helper (dict→list.values(), list passthrough) used by
check_and_archive_chapter (append/replace-by-title), check_and_archive_arc (total_chapters,
recent_chapter read, json.dumps), and the sceneagg aggregation context (chapters_count,
scenes_in_chapter); `current_custom_state` initialized early in both archive functions. Legacy
dict-keyed runs are read transparently (converted on next archive write). Verified: 5-case unit
test (list append = the crash case, absent→list, legacy dict→list, forced transition, arc
archival with list chapters) + hermetic suite green. NOTE: the running run has old code loaded —
resume (kill + re-run) needed for chapters/arcs to archive for the remaining turns.

## 26. Detail-loss crash (dotted names) + general_info empty render + alias-aware dict-get-set (2026-08-13) — shipped
User reported two context problems from the turn-12 dump of cozy_mystery__500fef50: (a) only
general_info + current_scene + RAG rendered as entries — characters/groups/elements/events were
missing entirely; (b) general_info itself rendered near-empty (Synopsis/Setting/Writing Style/Cast
blank or N/A). User also asked for the number of messages in the current scene in the
current_scene template, and to enhance dict-get-set to resolve nicknames/aliases/titles.
Root causes + fixes:
  A. Dotted-name split crash (the detail-loss bug): the LLM wrote relationship key "Mrs.
     Arbuthnot" WITHOUT [brackets], so the update landed as `relationships.Mrs` = {"Arbuthnot":[...]}.
     The relationship scan in context_retriever `_get_character_important_relationships` iterated
     that dict's KEYS as relationship records -> `'Arbuthnot'.get()` AttributeError at
     `_get_importance` -> the whole retrieve_context try-block aborted -> every subject (characters/
     groups/elements/events) stayed {} for the rest of the turn. Fixed: `_get_importance` guards
     non-dicts; the rel iteration normalizes dict-keyed rel lists; each subject extraction wrapped
     in its own try/except so one bad entry can never empty the rest.
  B. general_info always empty: the general_info format template used bare `{{synopsis}}`,
     `{{writing_style}}`, `{{_cast}}` etc. — none of those are in the render context (data is the
     root). All bare vars are now `{{data.*}}` (verified 2234-char render vs ~115 before).
  C. Dict-get-set alias resolution (user ask): new helpers in data_summarizer.py —
     `_collect_entry_aliases` (schema-agnostic walk for `aliases` arrays), `_resolve_dict_key`
     (exact -> normalized -> alias), `_resolve_path_keys` (canonicalize a path against the live
     structure: exact, dot-split-join for "Mr"+"Peters" -> "Mr. Peters", normalized, alias).
     Wired into `_apply_branch_updates` (write path), `_resolve_fuzzy_path` (reference reads),
     and `_filter_new_entry_names` (add_new dedupe now rejects known aliases). Unit-tested all
     paths. NOTE: recursive_set requires list indices as STRING-digit keys (converts internally);
     int indices raise — _resolve_path_keys keeps them as strings.
  D. Bracket-safe path markers (prompt-side): new `bracket` jinja filter (utils/helpers.py) wraps
     dotted names in [brackets]; format templates' path markers (characters/groups/elements/events/
     *_list) now render `{{path}}.characters.[Mrs. Arbuthnot]` so the model copies bracket-safe
     paths instead of emitting split dotted paths.
  E. current_scene template now shows "Messages in this scene (from message metadata) -- N" via the
     existing `scene_messages` filter keyed on `data._scene_number` (metadata passed in
     extra_context for every subject).
Verified: retrieve_context completes end-to-end (no crash) on the real run dir; resolver unit
tests; all subjects render; hermetic suite green (soak: 2/3 retained, f2_raid survives); both
schemas + format_templates JSON valid; spreadsheets validate. Pre-existing latent issue noted
(not fixed): mark_field's per-entry isolation never matches — identifiers like `.characters.
Prudence` never equal `branch_name`, so per-entry gate/branch contexts strip to headers only.

## 27. Type 2 refinement (minimal-runtime) + Mode B rolling-context design (2026-08-13)

Type 2 sceneagg schema refined to the user's minimal-runtime spec (the lightest
possible every-message load; heavy subjects only on scene transitions):
- Every message: branch-update current_scene.now (SceneState query), branch-update
  general_info (single skip_query self-detecting call), add_new elements.
- On new scene: archive scene + add_new/branch-update characters, groups, events
  (StoryEvents add_new moved off ALWAYS to on_new_scene).
Applied to subjects_schema_sceneagg.json; trigger dispatch verified (non-scene:
general_info+elements only; new-scene: events+characters+groups fire) via a
scripted DataSummarizer smoke; hermetic suite green. Type 1 vs Type 2 A/B queued.

New test condition (design only, per user "let's plan designing it"): Mode B —
rolling message history + inline retrieval. Today every call renders simulated
retrieval Q&A at the prompt head + format_dialogue enumeration ("N. 'name1' >> ...")
+ "Analyze all" marker — the prefill/attention cost driver. Mode B: (1)
message_mode=rolling — raw last-N user/assistant pairs replace the enumeration,
drop the marker; (2) retrieval_placement=system|inline — the non-to_context
subjects render ONCE per turn into a CURRENT CONTEXT block. last_x stays
dynamic via the existing scene-bounded _scene_dialogue_window computation
(config plumbing only).

User confirmed the design 2026-08-14: per-entry calls KEEP full cross-subject
visibility — the four header blocks ([CONTEXT]/[GENERAL_INFO]/[CURRENT_SCENE]/
[OTHER RETRIEVED SUBJECTS]) become a single shared SYSTEM prefix per turn
(retrieval_placement=system is the PRIMARY design — prefix-cache-friendly, no
per-call re-embed of subjects, shared by reply AND every DataSummarizer call);
the tail is raw rolling pairs + the per-call instruction. rolling_window
(semantic window) and --max-update-history (transport cap) COMPOSE:
sent = history[-min(window, cap):]. Control arms: message_mode ×
retrieval_placement (enumerated-start baseline / rolling-system primary /
rolling-inline placement control). Full design in
schema_types_and_test_matrix.md §3.

## 21. cozy_mystery__dded7733 (Type 2 Mode B, 30 turns, 3.5/5) — analysis findings

**RUN**: seed 93, schema-type 2, message-mode rolling, placement system. 3.5/5. Style/memory strong (judge 4.5-5), quality the weak axis (2-4, mostly 3).

**A. The characters-subject wipe (dominant engine bug, FIXED)** — t13 transition pass threw during characters generation; generate() swallowed the exception and skipped the write. 10 characters -> missing file -> `{}` for turns 14-27 -> rebuilt t28 with only 3 (Evelyn lost). Silent: judge scored memory=5 (memory summary omits empty subjects). Fixes shipped: persist-on-error in generate() + `_last_good_subjects` load-time restore.

**B. Prompting pipeline (the quality bottleneck — the instruction generator authors the penalized flaws)** — verified verbatim in instr_prompt files: the same Qwen model writes the instruction block AND the reply. t21 instruction literally contains "Jasper the tabby cat, who is weaving between your ankles" + "leaving the next move in Prudence's hands"; t27 contains "Ensure your response remains strictly in the third person... end on a specific action"; t28 "end on a specific action, such as adjusting the strap of your basket or turning toward the path, leaving the next move in Prudence's hands". The reply model transcribes these verbatim, then the judge blames Evelyn. P1 meta-instructions leak into prose; P2 motif repetition (sweet pea 26/30, defer-ending 8 turns); P3 instructions place cats/items; P4 two-paragraph directive = 2 mega-paragraphs (~600-700 words); P5 t9 verbatim re-quote.

**C. Scoring artifacts** — quality is mechanically the `>half in-scope notes missed -> quality<=3` floor (fires 6/8 judge turns); judge can't see the instr_prompt; memory=5 with empty subjects (empty-subject omission); judge charges DSS for guide-failure notes (c2 'feud', c15 'pressed rose' never planted by the guide); glove "pantry" misattribution is a judge read error (DSS says bandstand steps). Retention 21/26 = 81%; DSS-side 21/23 = 91%.

**D. Schema/engine gaps** — NAME EXACTLY absent from initial_population templates (watering can -> 'galvanised_watering_can'); 3 malformed auditor records + 3 not-JSON errors; characters add_new on_new_scene-gated (by design, audit capability context handles it).

**Fix priorities** (user steers): 1. deterministic `_clean_generated_instructions` extension (strip meta/deferral phrases) — kills P1/P2/P3 verbatim transcription; 2. ban deferral endings in gen requirements + reply hard rule; 3. per-paragraph word cap in dss_directive + cap beats at 3; 4. judge calibration: give judge the instr_prompt, split note-adherence from quality, ground memory in exact fields, don't charge DSS for guide failures; 5. NAME EXACTLY in population templates; 6. props-already-used list to the instruction generator.

## 28. Judge-calibration + instruction-hardening batch (2026-08-14) — shipped, hermetic green

Fixes for the cozy_mystery__dded7733 3.5/5 analysis (instruction blocks author the penalized flaws; scoring artifacts; empty-subject invisibility).

**F1 — `_clean_generated_instructions` extension (summarizer.py):** deterministic post-check now strips meta/deferral phrases in three passes — whole meta sentences ("Ensure your response...", "Your response should end on..."), deferral/construction clauses within kept sentences ("leaving the next move in X's hands", "as a prop to emphasize", "end on ... rather than a summary", "Avoid declaring the mystery resolved or naming the culprit definitively;"), and near-duplicate sentences. `_SENT_SPLIT_RE` has abbreviation-aware lookbehinds (Mrs./Dr./St.). Verified on real run instr_prompts (t9/21/27/28): target phrases all gone, actionable content preserved.

**F2 — prompt-side bans (summarizer.py):** generation-requirements #10 now bans deferral endings + repeated imagery explicitly; new #12 (every instruction = concrete imperative action; no meta-construction phrases); new #13 (max 3 beats per block); #5 reworded to reference the writing-style directive instead of "The final response should be...". Reply hard rules (both do_instr paths): never end by handing the turn to {name1}, never narrate the reply's own construction; extend anti-repetition to closing lines/ending gestures.

**F3 — dss_directive word caps (3 spreadsheets):** cozy "~80 words each, never more than 100" (two paragraphs); fantasy "~150 words"; romance "~60 words each". Rule #36: every paragraph-count clause got the cap.

**F4 — judge calibration (long_horizon_soak.py):**
- Feed the judge the current turn's INSTRUCTIONS block (extracted from instr_prompt.txt) with an ATTRIBUTION RULE: verbatim transcription = poor execution; instruction-MANDATED content (prop/animal/ending) is not charged as creative choice or memory error.
- Note-coverage decoupled from quality: removed the ">half in-scope notes missed -> quality<=3" floor; quality anchors reframed (notes lower quality only via the 4/2 anchors, never double-penalized).
- `_planted_note_ids()`: unplanted notes (needle never in story text; style-holds always in scope) are excluded from the in-scope set and listed for the judge/overview — never markable 'missed'. Wired into `_run_judge`, `_run_overview`, AND the final-overview builder (now `_backfill_overview`, 2026-08-16).
- Memory grounded in stored state: judges told to verify against the exact fields shown and not invent contradictions; `_memory_summary` now emits "(EMPTY — no entries saved)" markers so an empty map can't pass as memory=5.

**F5 — NAME EXACTLY (both schemas):** appended to characters/elements initial_population identification_prompt + population_prompt (8 templates) — was only in new_entry_query_prompt_template, so first-scene population produced 'galvanised_watering_can'.

## 29. Async judge/audit/rolling-overview workers (2026-08-14) — shipped, unit-tested

Per-turn judge, audit, and rolling-overview cloud calls were on the turn critical
path (blocking the next turn). They are pure side effects (reports, judge_records,
abort flag) — nothing in the next turn depends on their output — so they now run
on two background worker threads, one per CloudModel instance (serialized per
role: each role's tasks run in dispatch order on a single thread, so a CloudModel
is never used by two threads at once).

Design:
- SoakRun gains `_judge_q`/`_audit_q` (queue.Queue), worker threads (daemon),
  `_pending_judge`/`_pending_audit` buffers, `_cloud_lock` (RLock).
- run_turn (L1+L2) no longer runs judge/audit inline; it appends task closures
  (capturing `judge_window` + `_planted_note_ids()` SNAPSHOTS so a grading turn
  never reads later history) and returns `judge=None, audit=None`.
- main() checkpoints result.json, then `soak.flush_pending_tasks()` enqueues them.
  The workers call `_run_judge`/`_run_audit`, compute the per-task usage delta
  (base stats snapshot before the call — exact because the instance is
  single-threaded), merge `{judge, judge_usage}` / `{audit, auditor_usage}` back
  into the turn's result.json atomically (tmp+rename, under the lock).
- Abort early-stop moved into the workers: `_note_abort` increments/resets
  `abort_flags` and sets `_stop` (the loop notices at the next loop top — the
  stop is delayed by up to ~1 turn, acceptable for a safety valve).
- Rolling overviews run on the JUDGE worker (serialized after per-turn judges so
  `judge_records` is complete); they snapshot history + planted set at dispatch
  and read judge_records at run time. Final overview stays main-thread after
  `_join_cloud_workers()`.
- `soak_log` is now lock-guarded (thread-safe file writes); `_join_cloud_workers`
  is idempotent (sentinel protocol + `_workers_joined` guard).

Latency: judge/audit (typically ~10-60s of cloud time per cadence turn) now
overlaps the next turn's guide+reply+summarize (minutes), so it's hidden almost
entirely. The guide stays on the main thread — its output is the next turn's
input, and it depends on this turn's summarize having finished.

Also fixed en route: a latent `NameError: name 're' is not defined` in
`_run_judge` (the F4 instr-block extraction used `re` without importing it — the
judge path is only exercised live so it was never caught).

Verified: unit test with scripted cloud models — workers genuinely in flight
while the main thread runs; verdicts + usage deltas merged into result.json;
judge_records/audit_records populated; two consecutive abort verdicts set
`_stop`; join idempotent. Compile + full hermetic suite green.

## 30. Entry-selection action (select_entries_to_update) — 2026-08-14, shipped + hermetic + unit-tested

The "querying for each entity is still very slow" fix. On every per-entry sweep
(Type 1: `always` per-entry triggers every message; Type 2: `on_new_scene` heavy
pass) each Character/Group/Element/Event entry runs its own self-detecting update
call. A new MAP-level action now asks the model FIRST which entries plausibly
changed, then only the whitelisted entries run their per-entry triggers.

Design (user-specified, option "a" — whitelist-gated traversal, no new
`selected_to_update` trigger):
- New `Action.SELECT_ENTRIES_TO_UPDATE` (schema_parser.py).
- `DataSummarizer._select_entries_to_update` (called from `_execute_action`):
  renders `select_entries_to_update_prompt_template` (map class) via
  `_create_update_prompt` (`{{ branch_list }}` compact roster + scene context),
  calls generate_with_sse, parses a JSON name array, resolves each name via
  `_resolve_dict_key` (bracket-form + alias safe), stores the whitelist in
  `self._entry_whitelists[branch_name]`.
- `_traverse_structure` case 3a (dict-of-schema-classes loop) consults the
  whitelist: non-whitelisted entries are SKIPPED (no per-entry LLM call).
  Keyed by the map's branch_name (`characters.entries`, ...) so relationship
  sub-dicts are untouched. Per-instance dicts => parallel-worker-safe.
- Fail-open: unparseable response, non-`[`-leading garbage, json_repair
  reparsed-nested junk, missing template, or a map below
  `select_entries_min_size` (default 4) => whitelist None => run ALL entries.
  Empty-but-valid `[]` = the win case (skip all per-entry updates).
- New-entry union: entries added by add_new this pass are ALWAYS run (initial
  population), checked at loop time (`_new_entry_names`) so it works regardless
  of trigger ordering (add_new may fire after selection).

Cadence wiring (both schemas):
- Default (Type 1): selection on `always` after `perform_gate_check` for
  CharacterMap/GroupMap/ElementMap.
- Sceneagg (Type 2): selection on `on_new_scene` after `add_new` for
  CharacterMap/GroupMap; ElementMap gains `on_new_scene: [select_entries_to_update]`
  (its per-entry updates fire on new scene while add_new stays `always`).

Verified: hermetic suite green; focused unit test (10 cases across both schemas):
selection filters, empty-selection skips all, fail-open on garbage, min-size
guard, new-entry union. Live smoke (2-turn, Type 1, 27B, cozy_mystery__b2741b96)
green: min-size guard fired (characters 3 / groups 1 -> run all); elements (26
entries) selection returned [] -> all skipped (~26 per-entry calls saved, only
gate+selection+add_new ran); the [] was VERIFIED correct — 0 of 24 shared
elements changed across the turn, and the 2 new entries came from add_new
(independent of selection). Turn 1 = 69 calls total.

## 31. Prefix-caching prompt ordering: static-first / dynamic-last (2026-08-14) — shipped

User-raised principle: the DSS workload is ~99% prefill by tokens (128:1
prompt:completion), so TurboMind prefix caching is the dominant lever. Any
dynamic content placed BEFORE the static template text (or before a large
static block) splits the shared prefix there and kills the cache for every
later call. Fix: **static template content first, dynamic descriptors last** —
"this could partially help with LLM prompting (descriptor near the end) and
would certainly save thousands in token prefill time."

### The bug that surfaced it
`Elements.initial_population.population_prompt` rendered
`generate the full data for the element named ''` — the template used
`{{ element_name }}` but the population code (summarizer.py) only passes
`entity_name`, so jinja rendered it empty.

### Transformations applied to BOTH schemas (subjects_schema.json + sceneagg)
- **TR1** per-entry `branch_query`/`branch_update` (Character/Group/Element/
  SceneState/GeneralInfo/StoryEvents): all `'{{ branch_name }}'` mentions
  removed from the static body; a single trailing line
  `Entry to review: '{{ branch_name }}'` names the entry last. Now the whole
  body (IDENTITY BINDING + schema + example + output format) is a shared prefix
  across every per-entry call.
- **TR2** `new_entry_prompt_template` (all maps + StoryEvents wrappers):
  `named '{{ entry_name }}'` -> static "described here"; trailing
  `The entry to populate: '{{ entry_name }}'`.
- **TR3** `new_entry_query` + `select_entries_to_update`: the big
  `Here is the (full|current) list: {{ branch_list }}` block moved from mid-
  prompt to the very end (`Here is the current list of existing entries:`).
- **TR4** `population_prompt` (Characters + Elements): `element_name` empty-name
  bug fixed; `Descriptor: {{ descriptor }}` moved from the head to the tail.
- **TR5** `CharacterMap.importance_update_prompt_template`: the big
  `Current characters and their relationships: {{ value }}` block moved to the
  end.

### Code-level fix (data_summarizer.py — 5 sites)
All five call sites PREPENDED a dynamic block before the template:
`f"Current context for '{branch_name}':\n{formatted_data.mark_field(branch_name)}\n\n{template}"`.
That put the huge whole-subject render (up to ~143KB, identical across a
subject's per-entry calls) BEFORE the static template, so no two calls shared
more than the system+history prefix. Swapped to
`f"{template}\n\nCurrent context for '{branch_name}':\n{mark_field}"`.
Covers `_perform_gate_check`, `_perform_branch_query`, `_perform_branch_update`
(both full and update-only paths), and `_generate_field_update`. Note
`mark_field` does NOT isolate the entry (identifiers like
`data.characters.Evelyn` never match the bare `branch_name` passed) — it
returns the whole subject render; that block is IDENTICAL across a subject's
per-entry calls, so after the swap it sits in the cacheable tail region.
The mark_field-isolation bug remains a separate latent issue (memory #114).

### Initial Greeting empty — root cause + fix
The population prompts showed an empty `Initial Greeting:` block. Production
stores the character greeting in the assistant slot of the first internal pair
(`['<|BEGIN-VISIBLE-CHAT|>', greeting]`, chat.py:1801) and the engine read
`internal[0][1]`. The harness has no character greeting — its opening
`"You walk into the village green."` sits in `internal[0][0]` (the user slot),
so `internal[0][1]` was legitimately empty. Fix: the engine's four
`internal[0][1]` extractions now fall back to the opening user line
(`_first_pair[1] or _first_pair[0]`), so the population prompt shows the real
opening context instead of an empty block.

Verified: both schemas JSON-valid + parse (46 definitions), render checks for
every transformed template (name/descriptor/branch_list only at the tail),
compile clean, hermetic suite green, 12 spreadsheets validate. The running
seed-96 soak is on old templates (loaded at init) — the fixes land on the next
run.

## 32. Initial Greeting — production-parity redesign (2026-08-14) — shipped

Section 31's `internal[0][1] or internal[0][0]` fallback was **reverted** after
review. Production reserves "Initial Greeting" for name2's own opening
(assistant slot, chat.py:1801: `# Add timestamp for assistant's greeting`) —
`_populate_subject_identify` renders `Initial Greeting` and `First User Input`
as two distinct concepts. Falling back to the user's line put a *name1*
second-person address under the greeting label, and since population builds
name2's identity from these blocks, a model told "this is the greeting" could
attribute that voice to name2 — the exact identity/voice conflation memory #46
and #103 fight.

The real fix is **harness-side simulation, not engine fallback** (user: "we
should be trying to simulate `state` as much as possible in the harness,
meaning writing a greeting beforehand (e.g. in the spreadsheet)"):

1. New required spreadsheet field `greeting`: name2's opening line, written in
   the dss voice. `validate_spreadsheet.py` enforces it. All 12 sheets carry one
   (cozy/night the originals, greetings authored in the build scripts —
   `new_spreadsheets.py` gained a `GREETINGS` map; `revoice_spreadsheets.py`
   preserves it since it rewrites the full dict).
2. `build_state` sets `state["greeting"]` and `state["history"]["internal"] =
   [["<|BEGIN-VISIBLE-CHAT|>", greeting]]` — production shape exactly, so the
   fresh-chat world-cache key (`char_context + char_greeting + user_bio`,
   summarizer.py:1894) includes it. Also added `state["user_bio"] = name1
   description` (cache key parity; the engine only reads it there).
3. The harness no longer zeroes `history["internal"][0][1]`. The Level-2
   engine-view history passed to `generate_instr_prompt` / `summarize_latest_state`
   is `[["<|BEGIN-VISIBLE-CHAT|>", greeting]] + self.history`; the summarization
   call passes the same greeting-inclusive history (production's message-node
   numbering includes the greeting). `self.history` stays pure `[user, reply]`
   (judge/guide/probes/plan unaffected). Level-1 smoke mirrors it in both the
   `recent` window and `custom_state["history"]["internal"]`.
4. Rendering verified: `_engine_internal_messages` skips the marker and emits
   the greeting as the first **assistant** message (before the first user line);
   `format_dialogue` renders `3. 'Evelyn' >> Evelyn was weighing lavender.`
   first, marker excluded. Population prompts now show a real
   `Initial Greeting:` block and `First User Input:` distinct.

**Breakage note — hash discontinuity on old runs.** The engine's
`retrieve_history_path` hashes the full history, and the greeting is now part of
it. A run started on the OLD code has turn dirs keyed WITHOUT the greeting;
resuming it with the new code changes the hash mid-run → the engine computes a
fresh history dir with an empty world. Same-seed runs are unaffected for NEW
runs (greeting constant → consistent hashes). The in-flight seed-96 run
(40 turns, old code) must be left to finish on old behavior or restarted fresh.

### Remaining harness↔production state divergences (audited 2026-08-14)
- **visible history / per-message metadata** (`state["history"]["visible"]`,
  `messages_metadata` with role timestamps): the engine only reads `internal`
  + RAG-derived metadata. The `scene_messages` count and scene-boundary
  detection are already computed from `internal`. Metadata-driven features
  (timestamps, per-msg roles) would need the RAG index — tracked under 20.3.
- **chat_template_str**: the engine's only consumer is commented out. No action.
- **character generation params**: transport-level (LocalModel payload), not
  state. Production's `auto_max_new_tokens`/seed handling lives in
  `modules/chat.py`; the harness controls these via LocalModel kwargs.
- **seed**: production randomizes on seed -1; the harness pins the seed
  (reproducibility across a run). Intentional.
- **internal-history in-place mutation**: production mutates live state per
  turn; the harness passes fresh deep-copies each turn + keeps its own
  `self.history`. Functionally equivalent.


### New-entry name extraction + arcs/chapters template fix (2026-08-14)
**Bug**: add_new name queries on a small model answered bare ("The Unsealing of
the North Door") instead of a JSON array; the parser required JSON → entry never
added. Root cause was HALF a template bug: the Arcs/Chapters
`new_entry_query_prompt_template` literally asked for a bare title ("Respond with
arc title ... otherwise respond NO.") while every other discovery template
demands `["..."]`. The model followed the prompt correctly.
**Fix** (both schemas):
1. `_extract_entry_names()` (data_summarizer.py) — lenient extractor:
   strict JSON → json_repair → quoted strings → bare lines (bullets/brackets/
   quotes/trailing commas stripped, `_NO_NEW_ENTRY_PREFIXES` negative guard).
   Wired into `_detect_and_add_new_entries_to_branch` (was `jsonc.loads` only).
2. Arcs/Chapters templates now demand a one-element JSON array
   (`["The Unsealing of the North Door"]`); negative stays exactly `NO`
   (parser's stopping-string + negative-verdict paths already cover it).
`select_entries_to_update` deliberately stays strict-fail-open (a junk whitelist
would skip updates; add_new has no such failure mode — a miss just defers).

### Model choice for the soak (2026-08-14)
Decode is now the bottleneck (prefill optimized via prefix caching — arcs call
was 74% cache-matched; the remaining ~2K fresh tokens + ~8 output tokens still
took ~6s/call under 5-way concurrency: effective 1.34 tok/s on Qwen3.8-27B
dense at tp=2). A 3-call branch (gate check + add_new query + main update) ×
~6s ≈ 20s. Recommendation: serve the soak on Qwen3.6-35B-A3B-abliterated-AWQ —
3B active params/token vs 27B active = ~9x cheaper decode AND prefill; prior
35B runs held 4/5 + 5.0 style; the 27B never produced a clean baseline (wipe
bug). 27B decode-heavy strengths are irrelevant to a call-heavy, short-output
workload.

### 45adee46 fixes: greedy decode + engine-native instruction-repetition guard (2026-08-15)
**Root cause of byte-identical reply clusters (turns 7/8/9, 25/26, 33/34, 37/38)**:
lmdeploy 0.15 `GenerationConfig.do_sample` defaults to **False**
(messages.py:117); async_engine `_determine_gen_config` then forced greedy
decode (top_k=1, temp=1.0) regardless of the request temperature. The v0.15
upgrade regression — pre-upgrade the default was sampling. FIXED in the TGWUI
loader (`modules/lmdeploy.py _prepare_generation_config`: `do_sample =
temperature > 0`), server restarted, verified (temp 0.8 → 3 distinct outputs;
temp 0.0 → greedy).
**F1 (schema, both schemas)**: CurrentScene gained `now_query_prompt_template` +
`now_update_prompt_template`; its trigger is now
`always: [{action: query_branch_for_changes, prompt_template: now_update_prompt_template}]`;
SceneState's `always: [query_branch_for_changes]` removed. So the frozen
`what` headline + stale `now` are updated together every turn instead of the
`what` field being permanently frozen (it had no update template; 1 distinct
value across 40 turns while `where` said cottage/vestry — the contradictory
context resolved to the market stall). Verified: scripted CurrentScene
query→update test (what + now.where + now.who.characters[0].location all
update).
**F5 (engine-native)**: NO reply-text post-check in the harness — production
reply text is generated outside the extension (TGWUI modules/chat.py), so a
harness-side guard would never protect production. Instead the guard lives in
`generate_instr_prompt` (agents/summarizer.py): `_instructions_similar()`
(deterministic normalized difflib, threshold 0.85) compares the fresh
instruction block against `Summarizer._prev_instruction`; a collision triggers
ONE regeneration with `_INSTR_ANTI_REPEAT_DIRECTIVE`. Also fixed a latent bug:
the cleaned block was cached but the CURRENT prompt used the raw block
(`full_instr = instr`) — now `instr = cleaned_instr` so the cleaner AND the
guard reach the model on the generating turn. Verified: scripted 2-turn
generate_instr_prompt test + hermetic green.

## Round: retrieval + prompting (Type 2 focus) — 2026-08-15

**14714d8e (Type 2 Mode B, seed 103, 40 turns) = 3/5.** Style 4.0, quality 3.1,
memory 5.0. 38 note misses vs 15 echoes (9 judge turns); c5 + c14 missed in 9/9.
Data summarization is now solid — failures are reply-side only.

**Subagent loop diagnosis (023→038)**: NOT byte-copying (blocks differ ~0.08
difflib). It is prop-recycling + transcription, 5 mechanisms:
- M1 reply transcribes the instruction block near-verbatim (030==031 and
  032==033 replies byte-identical).
- M2 the instruction generator re-anchors on the STABLE top of context: Evelyn's
  byte-identical character entry + the first never-pruned elements (dried
  lavender first, ~860-1000-line block). "Weigh the dried lavender against the
  torn envelope…" appears 9×.
- M3 rolling window self-reinforces (replies → next instructions).
- M4 memory write-back: reply prose gets appended into elements.json
  descriptions AND importance bumped 50→70 (the loop's own output becomes canon).
- M5 no repetition penalty (repetition_penalty 1.0) + instruction directive
  rewards "weave in characters/items/past events" → re-anchors on stable props.
- current_scene.now WHERE advances, but TIME frozen ("Late morning" all 16 turns,
  11:30→11:10→11:00 regression), `start` never updates.

**Verified facts (answer P1 question)**: `current_scene` + `general_info` are
`to_context: true` → ALWAYS in `custom_state["context"]` regardless of
`retrieval_placement` (summarizer.py:1776-1777 unconditional). Format templates
are IDENTICAL across placements — placement only moves the OTHER subjects
(arcs/character_list/groups/elements/events/chapters/characters/lines):
prompt_start = Q&A head, system = CURRENT CONTEXT block, inline = last history
pair. So Mode A already had current_scene in context; "inject current_scene" is
a no-op. The real gap is the generator anchors on top elements + frozen time.

**Change list (this round, choose scope):**
- C1 — log `custom_state["context"]` to `turn_XXX/context.txt` beside
  instr_prompt.txt (user-requested; dump.txt is too opaque).
- P1a — inject a compact CURRENT SCENE recap (where/when + latest beats + what's
  NEW) at the top of the instruction-generation prompt, before the "weave in
  characters/items" line → generator plans around the live scene, not top props.
- P1b — CurrentScene.now query/update templates: add explicit `when` update
  guidance (time currently freezes) in both schemas.
- P2a — instruction-gen prompt hard rule: never reuse imagery/beats/actions from
  earlier replies or memory unless a deliberate stylistic choice (poignant/comedic).
- P2b — extend `_instructions_similar` past byte-similarity to prop/verb
  overlap (extract named props + action verbs from previous block; regenerate
  once if the new block reuses too many).
- P3 — beat-steering: user proposes the GUIDE (acting as user, production-
  faithful) deliberately points at one plot beat per turn instead of engine-side
  note injection. Needs design confirmation.
- P4 — F7-F9 completion: judge retry on non-JSON; report.py guide_failure vs
  dss_retention_loss attribution; merge duplicate cozy notes c14/c26.
Then a full 40-turn Type 2 Mode B run on the fixed stack vs 14714d8e.

## 30. cozy_mystery__b884af4a repetition regression (2026-08-15, Type 2 Mode B, 40t, 3.5/5) — analysis

Final overview 3.5/5; major detractor is the recurring "snip dried lavender /
place into grey ceramic jug" gesture (in ~21/26 replies turns 10-35). Three
subagents converged on the mechanism; none of the active defenses caught it:

1. **Instruction generator has no used-imagery memory** — req 9 ("weave in
   stored items") and req 14 ("never re-anchor on stored objects") are
   unsatisfiable together because nothing enumerates which stored objects prior
   replies used. The generator resolves it by picking the most positionally
   salient elements: "dried lavender" is elements.json entry #1 in EVERY turn
   (dict insertion order; frozen imp 50 == full-profile threshold), jug imp 60.
   17/20 instruction blocks (t11-30) carry the lavender/jug beat.
2. **Reply transcribes the instruction block 100%** (13/13 instructed beats
   performed) **and self-anchors on its own prior sentences** — rolling mode
   puts DSS's own last 6 replies as assistant messages directly above the
   generation boundary; t18 vs t16 reply ratio 0.79 (judge flagged "verbatim
   repetition"); t26 self-injected the gesture with NO instruction beat.
3. **Guards structurally blind** — all 40 blocks distinct md5s, consecutive LCS
   16-85 (fired once at t28), byte ratio max 0.17 vs 0.85 threshold; the beat is
   re-lexicalized every turn. Storage passive: descriptions frozen, importance
   pinned 50-60; salience purely positional. P1a scene recap was correct.

Fix batch (status as of 2026-08-16):
- P1 — INJECTED in instruction-GENERATION prompt after the scene recap; req 14
  now references the RECENTLY USED IMAGERY block (exhaustive for the last N
  blocks). SHIPPED.
- P2 — semantic prop-overlap post-check: overlap fires when the new block
  shares >=2 props with the previous block or the last-3 ring (tolerates 1
  prop per the "a prop can be important enough to mention every message"
  caveat); regenerates once NAMING the offending props. Prop extraction =
  elements-vocab exact match + head-noun fallback ("lavender", "jug") + a
  conservative verb-phrase regex. SHIPPED.
- R1 — DSS's own last 2-3 replies appended to the REPLY prompt as negative
  exemplars ("these are YOUR OWN most recent replies — do not reuse..."), with
  the essential-prop one-use allowance. SHIPPED.
- R4 — soak default repetition_penalty 1.0 -> 1.1 (only repetition knob
  lmdeploy 0.15 exposes; no presence_penalty). SHIPPED.
- S1 — dedupe strengthening in `_filter_new_entry_names`: within-batch checks,
  token-set containment ("grey jug" ⊆ "grey ceramic jug"), post-generation
  description-similarity merge (would catch jug pair + quince/quince pie 100%
  dup + trestle/judging table 98%). PENDING.
- S2 — importance decay / used-recently flag for over-referenced props.
  PENDING.
- S3 — render-order control so entry #1 can't hold slot #1 forever; fix latent
  bug format_templates.json:57 (`defaults.CharacterMap.importance_detail_threshold`
  is a stale path — threshold effectively hardcoded at 50). PENDING.

Also shipped this round: dashboard "history (unbound)" stat (per-turn
`full_history_tokens` = sum(len(u)+len(r) for u,r in soak.history)//4 in
result.json local_usage; shown beside peak ctx in the turn accordion + last-turn
in the live local-model card; "—" fallback for old runs). Expected curve:
flat -> medium -> high after ~20t -> ideally leveling off logarithmically.

## 31. Container-shape repair (coerce_container_types) — 2026-08-17, shipped

The 35B started emitting `[]` where the schema declares `dict[str, ...]` —
`Character.relationships` / `group_status` (73 rel-list entries in
cozy_mystery__3fbf1a99 vs 0 in every prior run). Every name-keyed update path
(`relationships.Glove.owner.events.0 = ...`) then crashed in `recursive_set`
with "Cannot use non-integer key 'Glove' on a list" and the update was SILENTLY
dropped. Root cause: the engine never coerced container shapes — 
`unexpand_lists_in_data_from_llm` is shape-preserving.

Fix: `coerce_container_types(data, schema_type, parser)` (utils/helpers.py) — a
schema-driven in-place traversal mirroring `unexpand_lists_in_data_from_llm`
that coerces dict-declared fields written as lists (empty -> `{}`, populated ->
keyed by a name-ish element field via `_infer_collection_key`, e.g.
`group_status`; CharacterRelationship objects carry no name field so real rel
lists are LEFT untouched) and int-keyed dicts under list-declared fields back to
lists. Wired at the top of `DataSummarizer.generate()` (data_summarizer.py:713),
before FormattedData construction and `_update_recursive`, so BOTH update
resolution and the save path see schema-consistent containers — self-heals
across turns. Root object identity preserved (in-place mutation), so
`all_subjects_data`/`processed_subjects_data` aliasing and the post-loop
chapter/arc checks stay consistent with what the workers wrote.

Verified: 7-case unit test (empty-list→{}, name-keyed resolve no-crash,
populated-no-name-list untouched, name-field list→dict, int-keyed dict→list,
scalar passthrough, real turn_013 data — 2 chars with `[]`→`{}`), full hermetic
suite green. Applies to every subject automatically (schema-driven, rule #35).

Residual (intentional): when the model ALSO writes a spurious nesting level
(`relationships.Glove.owner.<sub>` instead of the schema's `relationships.Glove.0.<sub>`),
the update now lands as a nested dict under Glove (data preserved, gracefully
skipped in rendering) rather than crashing. Prompt-level teaching of the
`Name.0.field` addressing form is a possible follow-up.

## 32. Real TTFT for local calls (2026-08-17) — shipped

The `local-call` log line labeled `ttft=` was actually the FULL round-trip
latency: `_complete` did a synchronous non-streaming POST (no `stream:true`) and
timed request-issue → full-body-read, i.e. prefill+decode+transport+queue. The
harness never saw a single token incrementally. Verified in code before the fix:
every "streaming"-named LocalModel method (`generate_with_sse`, etc.) delegates
to `_complete`, and the server's `/v1/chat/completions` (modules/api/script.py
+ completions.py) fully supports SSE streaming with per-token
`chat.completion.chunk` deltas + `stream_options.include_usage`.

Fix: `_complete` now sends `"stream": True` + `stream_options.include_usage`
and parses the SSE body (`_stream_post`): ttft = wall time to the FIRST delta
carrying content/reasoning, total = time to `[DONE]`, pt/ct from the usage
chunk (0-valued pt on v0.15 still falls back to the payload estimate). The
empty-completion retry (3x) is preserved; content/reasoning are concatenated
from deltas. Log line now `local-call <phase>/<step>: ttft=X.XXs total=Y.YYs
in=PT out=CT tok (pt/total K tok/s)`. Measured live: bot 9-token reply
ttft=0.12s total=0.35s; 58-token prose ttft=0.06s total=1.29s. Caveat: under
concurrent workers, client-side ttft includes server queueing time, not just
prefill+first-token decode. Hermetic green.

## 34. Overnight cloud robustness (2026-08-17)

`CloudModel.complete()` re-rolls empty-content draws up to `max_empty_retries`
(default 3) with `empty_cooldown` (default 300s) sleeps between re-rolls. A
reasoning-only draw is time-correlated on the Go endpoint, so a cooldown lets it
recover instead of killing the run. Budget tiering unchanged (small budgets
doubled once, large +2048 once, re-rolls stay there). `guide_turn_with_retry`
defaults max_attempts 3->5 and sleeps the cooldown on empty/truncated draws
(skips the sleep on verbatim-quote failures). Flags: `--guide-max-attempts`,
`--cloud-empty-retries`, `--cloud-empty-cooldown` — passthrough in run_tests.py.
Deliberately NOT folded into the run-id hash (never changes what the model sees
on success), so a resume picks them up without a new dir. Worst case on a dead
endpoint: guide aborts much later than before, after ~5 attempts of layered
cooldowns — still protects DSS prompts. Unit-tested (8 cases) + hermetic green.

## 35. Standing rule — context budget (2026-08-24)

Total rendered context should stay under **40k tokens** whenever possible. The
rig tolerates >60k, but current retrieval is considered inflated; every render,
prompt-template, or retrieval change must respect this budget. The P0 Canon
Synopsis (closed-arc entities demoted to roster lines) is the main planned lever.

## 33. Aggregation-unit fixes (2026-08-25, cyberpunk_thriller__4e29a512 post-mortem)

Engine bugs fixed (all verified: 8-case unit test /tmp/opencode/test_agg_units.py + hermetic suite):

- **List-branch add_new** (`discovery.py` + `core.py:_execute_action`): `_detect_and_add_new_entries_to_branch`
  now accepts `list[...]` aliases (was dict-only; Chapters' add_new was a silent no-op). List entries dedupe
  against title-ish fields (`title`/`name`/`formal_name`/`id`), missing titles fall back to the proposed name,
  and the formatted_data mirror append is identity-guarded (aliasing would otherwise duplicate entries —
  caught by unit test).
- **Chapters/Arcs triggers removed in BOTH schemas**: creation + gating are owned solely by the deterministic
  archive path (`check_and_archive_chapter/_arc`). The old `on_new_scene: [perform_gate_check, add_new]`
  triggers double-gated per scene turn and, with list add_new enabled, would have double-created chapters.
- **Post-loop gates now render with real content**: they previously passed no scene/chapter variables, so the
  templates' `{{ scene_recap }}`/`{{ scene_events }}` rendered EMPTY — the chapter gate judged on bare counters
  with zero story evidence (why it answered NO at 6-7 scenes). Gates now receive a compact span digest.
- **Real span math**: scenes_in_chapter = archived scenes since the last chapter's `ending_scene` (was
  total_scenes ever — never reset); chapters_in_arc = chapters since the last arc's `ending_chapter`
  (was a `# TODO`). Chapter-archive fallback data spans only the current chapter's scene indices.
- **Chapter generation prompt** now lists every scene of the span (name + summary), not just the newest.
- **Scene-transition prompt** (context_engine.py) aligned with the agreed unit semantics: action-phase shifts
  and material cast-presence changes are valid transition signals; passing tonal color is not.

Unit definitions encoded in the rewritten gate templates: SCENE = one continuous stretch (action/tone/cast/
setting shifts start a new one); CHAPTER = full TV episode / few graphic-novel chapters (setup -> complication
-> resolution or deliberate pause); ARC = full TV season / few GN volumes (long-range conflict premise ->
resolution). Suggested cadence unchanged (chapter 4-8 scenes, hard 10; arc 3-6 chapters, hard 12).

Open follow-ups from the same analysis (not yet implemented): uncertainty-preservation directives in character
update templates (Pike/Kell class), semantic similarity guard on discovery (P2), continuity contradiction probe
(P1), rolling-overview findings feedback loop into instruction generation, canon synopsis P0 (now unblocked —
it hooks chapter archival).

## 34. Live validation of §33 + the persistence bug it caught (2026-08-25)

Short Type 2 Mode B smoke (`--smoke 16 --max-scene-messages 4 --force-chapter-turn 6`, cyberpunk_thriller
seed 7, local Qwen3.6-35B-A3B + stealth/ox-alpha guide) to live-validate the hermetic-only §33 fixes.

**Found: boundary archives were never persisted (fixed same day).** Budget splits fired every ~2 turns;
at t2 with 4 archived scenes the natural gate ran the LLM check on a real span digest and answered YES —
"Archiving chapter 1" logged — but every events.json on disk still had `chapters=[]`. Cause: per-subject
files save inside DataSummarizer.generate BEFORE `_run_boundary_checks` mutates shared in-memory state, so
the chapter append and the `_chapter_number` bump were memory-only; next turn loaded from disk and lost
them. The arc path saved arcs.json but leaked its `_arc_number` stamp the same way. Fix in archives.py:
explicit saves of events.json/current_scene.json after successful archive mutations (arc path:
current_scene.json). Regression test asserts ON-DISK state: tests/boundary_persistence_test.py.

Also found/fixed in the same session:

- **NEXT CHAPTER:/NEXT ARC: prefixes were dead in production** (and harness `--arc-breaks` was a silent
  no-op): flags set in chat_input_modifier landed on the previous exchange's SummarizationContextCache,
  which get_retrieval_context replaces every exchange. Fix: staged intent in persistent_ui_state, consumed
  by `_consume_forced_unit_boundaries` inside prepare_context post-recreation (tests/force_boundary_test.py).
  Harness gained `--force-chapter-turn N`.
- **cloud_client DEFAULT_ENDPOINT still pointed at the retired OpenCode Go endpoint** → instant 401 once
  resolve_api_key started preferring the OpenRouter key. Flipped to openrouter.ai/api/v1.
- **Cadence profiles shipped** (decisions ledger #3): UNIT_CADENCE_PROFILES overlay in archives.py
  (compressed = schema numbers; campaign = chapters 15/40/50, arcs 8/24/30), config key
  `cadence_profile`, harness `--cadence-profile`, run-id hash + run_tests passthrough. Note: real arc
  bounds are 2/5/8 (both schemas), not the 3–6/hard-12 in older docs.

Validated working live post-fix: budget scene splits at `--max-scene-messages 4`, content-bearing gate
(LLM YES on span digest), list-shaped chapters rendering in context via the events template's roster
branch, forced boundaries landing on the current cache. Canon Synopsis P0 design doc written pre-code:
docs/plans/canon_synopsis_p0.md (home = general_info.synopsis; regenerate-from-inputs; render-time
staleness demotion).

### §34 addendum (run 2, same day): mis-heal chapter corruption + span hallucination

Second smoke (seed 11→94613841 era, `--max-scene-messages 4`) surfaced three more defects:

- **Chapters array corrupted by the array-wrapped-instance heal**: the field-level heal in
  traversal.py unwrapped a LEGAL `[chapter_dict]` on the list-alias field events.chapters to a bare
  dict; `_entries_as_list` then flattened it via `list(dict.values())` into positional scalars
  (`['Shadow Side Run', 1, 4, [1,2,3,4], summary, [...], 'concluded'`). Archival appended onto the
  monolith and (with the §34 persistence fix) faithfully wrote it to disk. Fix: heal skips fields whose
  type origin is `list`.
- **Hallucinated archive spans accepted**: forced archive wrote `ending_scene: 14` with 5 scenes on
  disk because model-provided span fields were only fallback-filled. Fix: computed span indices are now
  unconditional overrides; the model owns title/summary/key_changes/status only.
- **Malformed entries now sanitized**: chapter/arc paths drop non-dict entries with a warning before
  boundary math, so a corrupted array can no longer feed ending_scene scans or arc gate counts.

Also clarified: scene counts legitimately grow faster than budget splits — events.scenes has its own
ON_NEW_SCENE add_new trigger (LLM-proposed scene entries) and auto-detection runs alongside splits when
min_scene_part_messages=0. And `--max-scene-messages` counts inclusively (both endpoint exchanges), so
4 makes every exchange a boundary candidate; multi-exchange scene parts want 8+.

Regression coverage extended in tests/boundary_persistence_test.py (sanitize + span-clamp cases);
full hermetic suite green.

### §34 addendum 2 (run 3, d448ef5f): render-path chapter corruption + junk-key span inflation

The clean rerun (16 turns, `--max-scene-messages 8`, forced chapter at t6) validated the heal fix,
sanitizers, deterministic spans and on-disk persistence end-to-end: chapter 'Route Adjustment'
archived at the forced turn with span [1-3] persisted, counters coherent, no array corruption.
Two residual defects surfaced and were fixed:

- **Render path**: archived chapters rendered as `Chapter [1] --- 0` — FormattedData.__init__ ran
  expand_lists_in_data_for_llm with schema_type=None for hintless data_types ('chapters', 'arcs');
  the expander's terminal branch dict-expanded `[chapter_dict]` → `{"0": chapter_dict}` so the
  list-iterating template saw stringified keys. Fix: untyped object lists stay lists; scalar lists
  still expand. Regression test tests/chapters_render_test.py.
- **Junk-key span inflation**: a misdirected add_new nested a whole `{"scenes": {...}}` wrapper under
  events.scenes['events'], inflating scene-count-based span math by one phantom scene. Fix: archives.py
  skips reserved field-name keys {events, past, chapters, arcs} when counting scenes.

Operational note: the first instance of run 3 died at t12 with an unhandled
URLError (Connection refused) when the local server blipped — checkpoint resume worked cleanly
(relaunch resumed at t12 as sole writer), but the harness should retry transient local-server errors
before giving up (candidate hardening for the 100-turn run).
