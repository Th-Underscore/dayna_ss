# Long-Horizon Soak Test — Plan

**Status:** Implementation stage — Milestone 0 (Q5 prompt fix) DONE; harness
milestones 1-5 (below) not yet implemented. Ready for the soak implementation.
**Date:** 2026-08-09
**Related:** `dayna_ss_strategy_review.md` §9 (live environment) + §3 (soak rationale); existing harness in `extensions/dayna_ss/tests/`

## 0. Environment & live-serving cheat sheet (handoff-critical)

```
conda env:       lmd-tg            (gradio + lmdeploy; run everything in it)
hermetic tests:  cd extensions/dayna_ss && conda run -n lmd-tg python tests/run_tests.py
                 # --live additionally runs live_benchmark/live_director/live_soak
                 #   (needs DSS_BENCH_BASE_URL); --update-golden regenerates goldens

Local model server (V100 GPU 1, LMDeploy TurboMind):
  CMD_FLAGS_m.txt in text-generation-webui root:
    --model Huihui-Qwen3.5-9B-abliterated-AWQ-4bit --n_ctx 32768
    --loader LMDeploy --cache-type q8
  CMD_FLAGS_p.txt: --api-key 764d7ca1-7a25-405e-ad27-4fe7ff0bf1ae
  Start:   ./link.sh && setsid nohup ./strt mp &   # link.sh symlinks user_data/models -> ../../models
  Serving: http://0.0.0.0:5000/v1   (OpenAI-compatible)
  Model dir: /mnt/c/Users/there/Downloads/Projects/Programming/models

Cloud judge/guide (OpenCode Go, Route A):
  Endpoint: https://opencode.ai/zen/go/v1/chat/completions  (Bearer auth)
  Key:      "opencode-go" entry in ~/.local/share/opencode/auth.json
            (or OPENCODE_GO_API_KEY env)
  Default model: deepseek-v4-flash  (Q7); judge/guide ids configurable (Q2)

Important behavioral knobs:
  DSS_BENCH_THINKING=0   # Qwen3.5-era models burn tokens on reasoning_content
                         # by default; live tests disable thinking via
                         # enable_thinking:false. Default is OFF (0).
  DSS_BENCH_MODEL / DSS_BENCH_BASE_URL   # live-benchmark endpoint/model overrides
```

---

## 1. Goal

Run **one continuous conversation of ~100 messages** through the real dayna_ss
engine on a live local model, while a **larger cloud model** (via OpenCode Go)
steers the story and **judges quality and memory retention**. Repeat across
**many genres and writing styles** to find where DSS memory degrades over long
horizons.

Unlike the current `soak_conversation` fixture (20 scripted turns) and
`live_soak.py` (20 live turns), this is a **true long-horizon** test: hours of
compute are expected and fine — but the harness must be **foolproof**
(checkpoint/resume, deterministic, fully logged, rate-limit aware).

### Success criteria
- A 100-message conversation completes without human intervention.
- Crash-safe: a killed run resumes from the last completed turn with identical
  artifacts (no recompute, no divergence).
- Every prompt, response, and state snapshot is on disk under a run directory.
- Output is a structured report: per-genre style-consistency score, per-detail
  retention timeline, and memory-fidelity judge scores.

---

## 2. System under test (SUT)

The real engine, driven exactly like production (`script.py`):

```
1. guide/user turn arrives
2. Summarizer.generate_instr_prompt(user_input, state, history, do_instr=...)
   -> reply prompt        (calls retrieve_and_format_context internally)
3. local model generates the character reply from that prompt
4. Summarizer.summarize_latest_state(output, user_input, state, history)
   -> memory update (gate check, branch queries, field updates, scenes, chunks)
```

Local model: `Huihui-Qwen3.5-9B-abliterated-AWQ-4bit` served via LMDeploy on
V100 (port 5000), or any `DSS_BENCH_MODEL` (see §0 cheat sheet). Director pass
(`do_instr`) is a test factor (on/off). Note: Qwen3.5-era models default to
thinking (empty `content`); pass `enable_thinking: false` on every request
(`DSS_BENCH_THINKING=0`).

---

## 3. Roles

| Role | Who | What it does |
|---|---|---|
| **Character / SUT** | DSS + local 9B | Writes name2 (character) replies, maintains structured memory. |
| **Guide (cloud)** | OpenCode Go model | Writes name1 (user) turns, loosely following the spreadsheet's *active* notes (never shown to DSS). |
| **Judge (cloud)** | OpenCode Go model | Every N turns: scores style consistency, narrative quality, memory fidelity, and per-note adherence. |
| **Auditor (cloud)** | OpenCode Go model | At `dss_beats` turns: compares DSS's actual subject files against the spreadsheet's explicit "DSS should save a new X with Y" beats; verdicts `saved|partial|wrong|missing`. `--audit-every N` forces a full-beat audit every N turns; 0 = beats only. |
| **Harness** | `long_horizon_soak.py` | Orchestrates the loop; probes needles + recall; guards contamination; checkpoints; aggregates metrics. |

The guide, judge and auditor **may be the same cloud model** (different system
prompts) or different models — the harness treats them as three configurable
endpoints (`--guide-model`, `--judge-model`, `--auditor-model`).

---

## 4. Route analysis

The question: *how do we get the large cloud model to guide and judge?*

### Route A — Direct HTTP to the OpenAI-compatible endpoint (RECOMMENDED)
OpenCode Go exposes `https://opencode.ai/zen/go/v1/chat/completions` with
standard Bearer auth (endpoint + key resolution in §0). The harness calls it
with `urllib` exactly like `live_benchmark.py` already calls the local server.
- **Pros:** trivial to implement, no extra process, no JSON-shell fragility,
  full control over retries/backoff, works headless for hours.
- **Cons:** none material.

### Route B — Drive OpenCode as an agent (`opencode run`) per turn
Invoke the OpenCode CLI with a custom tool that calls the endpoint.
- **Pros:** reuses OpenCode's model routing/auth niceties.
- **Cons:** per-turn subprocess spawn overhead, output-parsing fragility,
  harder to make deterministic/retryable for a 100-turn × multi-genre matrix.
  No benefit over A because Go already exposes a plain OpenAI endpoint.

### Route C — Guide local, judge cloud
Local 9B writes user turns, cloud only judges.
- **Pros:** saves cloud quota.
- **Cons:** the whole point is a *larger* model steering; a small model guides
  poorly, confounding the "guidance" factor.

### Route D — Judge local, guide cloud
Judge = local large model (e.g. `qwen3.6-27b` if served).
- **Pros:** no cloud judge cost.
- **Cons:** a 27B on 16GB V100 alongside the 9B SUT is tight (VRAM), and using
  the same hardware class to grade itself weakens the "independent judge"
  property the user wants.

### Decision
**Route A**, single cloud endpoint with two configurable model roles (see §3
and Q7): **default `deepseek-v4-flash`** for both guide and judge (Q7 decision —
the `-0731` checkpoint is sometimes better than Pro and ~10× cheaper). The
guide/judge role split, `--guide-model` / `--judge-model` flags, and a Pro-vs-
Flash comparison (Q2 self-grading test) all remain supported. `deepseek-v4-flash`
budget (~158K req/mo) comfortably covers a 100-turn soak and the full matrix.

---

## 5. Story spreadsheet schema

The spreadsheet is a **story skeleton**: a loose outline of **dozens of notes
at varying specificity** that the cloud **guide** follows (never DSS). Each
note is an instruction about what the story should eventually contain, do, or
become — the guide interprets them freely, and the judge later grades how well
each one landed. Notes mix tight facts ("the crimson scarf is an heirloom")
with fuzzy directions ("steer toward a betrayal twist without being explicit",
"X location should be a turning point", "introduce Ruiz early, never mention
him again, and wait for dayna_ss to recall him when it becomes relevant").

The notes are **only ever seen by the cloud model**. The harness never injects
them into DSS's prompts, so the test measures whether DSS's *own memory*
carries the story forward — not leaked authorial intent. (A future feature will
hand a plot outline / notes directly to dayna_ss; out of scope here.)

```jsonc
{
  "id": "noir_detective",
  "genre": "Noir detective",
  "premise": "A burnt-out detective takes a case that drags him back into the city's underworld.",
  "setting": "Rainy 1950s harbor city, jazz clubs, crooked docks.",
  "characters": {
    "name1": { "name": "Marlowe", "role": "detective (guide writes these turns)" },
    "name2": { "name": "Vivienne", "role": "nightclub singer (DSS writes these turns)" }
  },
  "writing_style": {
    "directive": "Hardboiled noir, cynical and weary. Short declarative sentences. One paragraph per beat.",
    "guide_directive": "First-person past tense as Marlowe, the detective. Terse, wry, world-weary; narrate only what Marlowe sees, feels, or decides. Hard-boiled idiom.",
    "dss_directive": "Third-person past tense following Vivienne, the nightclub singer. Close but not interior — observe her words, gestures, and withheld secrets from the outside. Guarded, faintly dangerous.",
    "taboos": ["No purple prose", "No happy endings"]
  },
  "specificity_profile": "mixed",          // "loose" | "mixed" | "exact"
  "notes": [
    { "id": "n1", "type": "character_plant", "specificity": "exact",
      "content": "Introduce Vivienne's crimson scarf as a family heirloom.",
      "plant": { "turn": 5 } },
    { "id": "n2", "type": "character_echo", "specificity": "loose",
      "content": "Mention Marlowe's old partner Ruiz early, tied to Vivienne, then never mention him again. Wait for dayna_ss to recall him when it becomes relevant around the docks.",
      "plant": { "turn": 8 },
      "recall": { "due": 70, "hint": "a dockside reunion should surface Ruiz" } },
    { "id": "n3", "type": "plot_twist", "specificity": "loose",
      "content": "Steer the story toward a betrayal twist around turn 45 without being explicit." },
    { "id": "n4", "type": "location_turning_point", "specificity": "loose",
      "content": "The old docks should be a turning point, not just scenery." },
    { "id": "n5", "type": "supersession", "specificity": "exact",
      "content": "The ledger is burned at turn 60; after that DSS should treat it as gone, not still hiding in the safe.",
      "plant": { "turn": 60 } },
    { "id": "n6", "type": "foreshadow", "specificity": "loose",
      "content": "Foreshadow the fire that burned the harbor office before it happens." },
    { "id": "n7", "type": "style_hold", "specificity": "exact",
      "content": "Keep the cynicism and short sentences even during the climax." }
    // ... dozens more
  ],
  "dss_beats": [
    { "id": "d1", "turn": 2, "subject": "characters",
      "content": "DSS should save a new character: Evelyn's aunt, the constable's wife, who runs the bake-off registration table.",
      "expect": { "name": "Mrs. Arbuthnot", "role": "constable's wife", "detail": "bake-off registration" } }
  ]
}
```

`dss_beats` is the **auditor's job list**: explicit "DSS should save a new X
with Y data" expectations, each scheduled at a `turn`. On that turn the
harness hands the beat + DSS's actual subject JSON to the **auditor** cloud
model, which returns `saved|partial|wrong|missing` with a one-sentence detail.
Unlike notes (guide-only, graded post-hoc by the judge), a beat checks DSS's
*structured memory directly* at the moment the save was supposed to happen.
Optional `expect` gives the auditor the exact data the entry should carry.

### Note population guidance
Each spreadsheet carries **roughly 18-22 notes + 5-9 beats (a couple dozen
items total)**, and the *majority of notes are mundane detail-retention
notes*: everyday objects, habits, minor NPCs and small places that are part of
the story's texture and that DSS must keep in its structured memory for tone
and detail retention (a watering can, a stuck piano key, a cat, a cracked
mirror). These are usually `character_plant` (short window: plant it, DSS
must hold it) or `character_echo`/`foreshadow` with a late recall so the
mundane detail must *resurface*. The plot-driving notes (`plot_twist`,
`supersession`, `location_turning_point`, tight `character_echo` beats) are a
smaller minority. Plant turns are spread across the run so the guide's
windowed outline (spread 8) never carries the full outline at once — peaks at
~10-13 notes mid-run, then tapers.

### Note taxonomy
Notes are **typed** so the judge grades adherence per cognitive operation and
the report breaks down failures by kind:

| type | What it tests | Judge check |
|---|---|---|
| `character_plant` | DSS captures a new detail at plant time | detail present in state + reply near `plant.turn` |
| `character_echo` | **Recall**: planted early, DSS must surface it much later with no re-introduction | state still holds it at `recall.due` AND the reply at that turn references it — memory-driven, not guide-driven |
| `plot_twist` | Directional steering | the beat was reached; the twist changed the state |
| `location_turning_point` | Engine scene/arc machinery carries location weight | location appears in state; the turning-point reply engages it |
| `supersession` | DSS *forgets on purpose* (overwrite, not drop) | detail absent after the superseding event |
| `foreshadow` | Setup→payoff chaining | payoff turn's reply echoes the foreshadowed element |
| `style_hold` | Style consistency under pressure | judge `style_score` at the specified window |

Per-note `specificity`:
- `exact` — the judge expects the literal needle in state/reply (string probe
  + semantic confirm). Tests **verbatim preservation**.
- `loose` — the judge grades semantically (was the *idea* present?). Tests
  **semantic capture**.

### Test-factor: specificity profile
A spreadsheet declares `specificity_profile` (`loose` / `mixed` / `exact`)
describing the *mix* of its notes — so "spreadsheet specificity" stays a real
knob while individual notes still vary within it.

### Test-factor: guide steering style
- **Explicit** — the guide openly works the notes into its user turns: *"User
  (Marlowe): 'You're still wearing that scarf, aren't you?'"* (obvious
  steering; tests the strong-signal case).
- **Implicit** — the guide advances the scene naturally and lets DSS pick up
  the planted details on its own. (hidden steering; the stronger memory test)
- These are two system-prompt variants for the guide role.

### Test-factor: voice / person combination (12 spreadsheet matrix)
Every spreadsheet carries a `guide_directive` (the cloud guide's voice for
name1) and a `dss_directive` (the desired DSS voice for name2). Until 2026-08-14
all sheets used the same 1st-person-guide / 3rd-person-DSS split. The set is now
a **voice matrix** — different person/tense combinations per spreadsheet (cozy
kept as the 1P/3P control):

| Spreadsheet | Guide (name1) | DSS (name2) | Combo |
|---|---|---|---|
| cozy_mystery | 1st past (Prudence) | 3rd past (Evelyn) | **control** |
| noir_detective | 1st past (Marlowe) | 1st past (Vivienne) | two 1st-person perspectives |
| fantasy_epic | 2nd present (narrator→Ysara) | 2nd present (Ysara) | second person both |
| sci_fi_heist | 3rd present (story master) | 1st present (Dex) | story master + in-world |
| horror | 1st present (shared 'I') | 1st present (shared 'I') | same-perspective 1st |
| romance | 3rd past (Amelia) | 3rd past (Ashworth) | dual close 3rd |
| western | 3rd past (Callan) | 3rd past (Rosa) | dual close 3rd |
| cyberpunk_thriller | 2nd present (Handler→'you') | 1st present (Juno) | handler + runner |
| greek_mythology_retelling | 3rd past (muse) | 2nd present (hero as 'you') | muse + interactive hero |
| postapocalyptic_survival | 1st present (Sal) | 3rd present (Rook) | classic split, present tense |
| heist_caper | 2nd present (Fixer→'you') | 2nd present (Diamond) | second person both |
| gothic_haunted_mansion | 1st past (Isobel journal) | 2nd present (ghost→'you') | journal + ghost address |

Because DSS voice is carried by `general_info.writing_style` (see §voice
carrier), the person/tense experiments flow through the production path. The
harness's scoring prompts are **person-agnostic** (2026-08-14): judge /
final-overview / synthetic-reply templates grade the DSS turn against
`{dss_directive}` instead of assuming third person ("wrong-person narration"
anchors replace the old "first-person throughout" anchors), and the guide
template no longer says "You play {name1}" (it writes name1's turn in whatever
person the directive specifies). Same-perspective identity-sharing is carved
out of the judge's identity-confusion floor.


A note can fail for three different reasons, and the report must distinguish
them:
- **Guide failure** — the guide never planted the detail (state never had it).
- **DSS retention failure** — planted, then lost from state.
- **DSS supersession** — correctly dropped per a `supersession` note (a pass).
The judge's per-note grading plus the deterministic state timeline provides the
evidence for attribution; never count a "guide never planted it" as a DSS loss.

---

## 6. Harness design (`long_horizon_soak.py`)

### Turn loop
```
for turn in range(resume_from, TOTAL_TURNS):
    # 0. Live plan (optional, --plan-every N): guide revises its own short-term plan
    #    every N turns against the FULL outline; the plan (not the outline) then
    #    steers the in-between turns. plan.json snapshotted per turn.
    if plan_every and turn % plan_every == 0:
        plan = guide.replan(full_outline, prev_plan, recent_window, dss_memory_summary)
        persist(plan)                                       # run_dir/plan.json + turn plan.json

    # 1. Guide writes the next user turn from the notes relevant to this window
    active_notes = notes_visible_at(notes, turn)          # planted/echo-due/twist-window
    user_input = guide.call(spreadsheet, active_notes, recent_window, dss_memory_summary,
                            plan_view=plan)               # plan injected on top of notes window
    if arc_boundary(turn, arc_breaks): user_input = "NEXT ARC: " + user_input  # optional factor

    # 2. Engine reply prompt (director pass configurable)
    instr_prompt, custom_state, history_path, ts = summarizer.generate_instr_prompt(
        user_input, state, history_internal, do_instr=DIRECTOR_ON)

    # 3. Local model generates the character reply
    reply = local_model.call(instr_prompt)

    # 4. Engine summarization (real memory update)
    summarizer.summarize_latest_state(reply, user_input, state, history_internal)

    # 5. Deterministic needle probes on DSS state JSON (free)
    probe_report = probe_needles(notes, history_path)

    # 6. Echo/recall probe: is the reply itself referencing a due recall? (free)
    if turn in recall_due_turns: recall_report = recall_probe(notes, reply, state)

    # 7. Optional judge pass every JUDGE_EVERY turns
    if turn % JUDGE_EVERY == 0:
        judge_score = judge.call(spreadsheet, active_notes, window, dss_state_summary, reply)

    # 7b. DSS-save audit at dss_beats turns (or every AUDIT_EVERY turns)
    if turn in beat_turns or (AUDIT_EVERY and turn % AUDIT_EVERY == 0):
        audit = auditor.call(spreadsheet, due_beats, dss_state_summary)

    # 8. Checkpoint (atomic) — see §7
    checkpoint(run_dir, turn, user_input, reply, instr_prompt, custom_state,
               probe_report, recall_report, judge_score)
```

### Guide prompt (role: name1)
System: *"You are the user in a collaborative fiction. Continue the story as
{name1}. Advance the plot using ONLY the notes attached for this turn's window
— interpret them loosely and naturally. [EXPLICIT] Work the notes' specifics
into what you say. [IMPLICIT] Never reveal the notes or the outline; just
continue the scene."* Context: recent dialogue window, current DSS memory
summary (so the guide knows what's established), the active notes.

Note: the guide sees only the *active* notes for its window, not the full
outline, to avoid it front-loading future beats.

### Live short-term plan (role: guide's own agenda)
The guide maintains a **rolling ~15-exchange plan** that it owns and revises
on a fixed cadence (`--plan-every N`, default 5; 0 disables). The plan is the
guide's answer to "what am I trying to land in the near future", and it is the
*only* thing that follows the guide between replans — the full outline is not
re-sent on steering turns.

- **Replan turns** (`turn % N == 0`): the guide is given the FULL outline,
  its previous plan, recent exchanges, and the DSS memory summary, and emits a
  fresh plan JSON via `json_mode` + `reasoning_effort=low` (4000-token budget;
  `parse_plan` repairs truncated JSON). Beats carry `note_ids` and `recalled`
  content — concrete details copied out of the spreadsheet that the guide
  wants to work in over the window. Persisted to `run_dir/plan.json` and a
  per-turn snapshot.
- **Steering turns** (in between): the guide gets its current plan (`_plan_view`,
  one line per beat) *on top of* the normal notes window + static style block.
  The plan does not replace `notes_visible_at` — it layers the guide's own
  agenda over the deterministic window.
- **Why**: the guide manages its own context instead of the harness deciding
  relevance. Replanning every few turns means a beat planned ~5 exchanges out
  is seen on the refresh that covers it, giving the model a wider working
  understanding than a per-turn window alone.
- **Resume**: `plan.json` is reloaded on resume so the agenda survives
  checkpoints.

### Judge prompt (role: judge)
System: *"You are a literary editor grading a collaborative story against a
private outline. Score: (1) style consistency vs the two writing-style
directives (name1's guide voice and name2's DSS voice),
(2) narrative quality, (3) memory fidelity (does DSS's memory reflect the story
so far), and (4) per-note adherence for the notes in scope this window."*
Output: **strict JSON**:
```json
{"style_score": 0-5, "quality_score": 0-5, "memory_fidelity": 0-5,
 "notes": [{"id": "n2", "status": "planted|echoed|missed|superseded", "detail": "..."}],
 "summary": "..."}
```

### Deterministic probes
- **State needles:** reuse `live_soak._find_needle` against the per-subject
  `*.json` under `history_path` after every turn → per-detail presence timeline
  (`plant_at`, `last_seen`, `loss_turn`, `survived`), identical to the hermetic
  soak report. Free and deterministic — the backbone of retention.
- **Echo/recall probe (free, string-based):** at a note's `recall.due` turn,
  check whether the *reply* text mentions the planted needle (so we can tell
  "DSS surfaced it from memory" vs "the guide reintroduced it"). Combined with
  the state timeline this attributes the recall to DSS rather than the guide.

### Notes → DSS contamination guard
A hard harness invariant: **note text must never appear in any prompt sent to
DSS** (in `user_input`, `instr_prompt`, or the engine's context). A validation
step after each turn asserts none of the spreadsheet's note strings appear in
`instr_prompt`/`state` input paths; violation aborts the run (loudly) so the
test can never be silently leaked. (The guide's *output* may naturally echo a
detail — that is legitimate story, not contamination; the guard only checks the
raw note text.)

### Metrics produced
- **Retention rate**: needles surviving to turn 100, split by
  `character_plant` vs `character_echo` (recall) vs `foreshadow` — retention
  broken down by note type.
- **Recall success**: for echo notes, did the reply reference the planted
  detail at `recall.due` *without* the guide re-introducing it? (the strongest
  DSS-memory signal).
- **Style-consistency curve**: judge `style_score` over time (drift detection).
- **Memory-fidelity curve**: judge `memory_fidelity` over time.
- **Supersession correctness**: were `supersession` notes correctly dropped?
- **Attribution**: guide-failure vs DSS-retention-failure per note (see §5).
- **Cost**: local + cloud LLM calls and tokens per turn (cloud usage returned
  by the endpoint).
- **Director on/off delta** when both are run for the same spreadsheet.

### Per-turn usage + timestamps (result.json)
Each `turn_XXX/result.json` now records, in addition to `dt_s` and the old
cumulative fields:
- `local_usage` / `cloud_usage` / `judge_usage` / `auditor_usage`: **per-turn
  deltas** `{calls, prompt_tokens, completion_tokens, total_tokens, model}`
  (snapshotted around `run_turn` in `main()`). Earlier runs stored *cumulative*
  counters in `cloud_usage` that reset on resume, which overcounted when
  summed — the dashboard's `_usage_series` reconstructs per-turn deltas from
  those (detecting the resume reset via the calls counter), so old run dirs
  display correct totals too.
- `wall_ts`: wall-clock ISO timestamp of the turn's completion (the pre-existing
  `ts` field is story-scene time, not wall clock).
- `final_overview.json` gains `generated_at`.

### Soak dashboard live-stats panel
`tests/soak_dashboard.py` shows a top-of-page live panel refreshed every 10 s
(`show_progress="hidden"` so no loading spinner): local server status (model,
loader, **max ctx** from `CMD_FLAGS_m.txt` `--n_ctx`, **last-turn ctx sent** =
that turn's per-turn `prompt_tokens`), cloud models (per-role ctx + last-turn
ctx sent), run progress, per-turn usage totals, and GPU load — each card
carrying the `HH:MM:SS` it was measured. Run selection uses `gr.render`
(re-invokes `_run_display` on dropdown change). Timestamps appear on retention
tables/rows (planted/last-seen `turn@time`), judge curves, recalls, per-turn
accordion headers (`at`), and the final overview (`written at`).

---

## 7. Foolproofing (hard requirements)

1. **Checkpoint / resume.** After each turn, atomically write
   `run_dir/turn_{NNN}/` containing `user.txt`, `reply.txt`, `instr_prompt.txt`,
   `state.json` (custom_state), `probes.json`, `judge.json`, plus the engine's
   own history files. A top-level `run_dir/manifest.json` records the last
   completed turn + full config hash. On start: `--resume run_dir` detects the
   last turn and continues; never re-runs a completed turn.
2. **Deterministic seeds.** Local model `temperature=0`; cloud guide
   `temperature` configurable but fixed per run; seed recorded in manifest.
3. **Retry / backoff.** Cloud calls retried with exponential backoff on
   `429/5xx/timeout` (max retries configurable). Local server health-checked
   before each turn; if down, wait and retry (server restart flow documented).
4. **Rate-limit awareness.** OpenCode Go limits are **$12/5hr, $30/wk,
   $60/mo**. `deepseek-v4-flash` (default, Q7) ≈ 158K req/mo — comfortable for
   the full matrix; but the harness tracks cumulative cloud usage and
   hard-stops with a clean checkpoint when a configurable budget is exceeded.
   (If a Pro/GLM judge experiment runs, that model's own 5-hr cap applies.)
5. **Smoke mode.** `--smoke 5` runs 5 turns to validate a spreadsheet + config
   before committing hours to 100.
6. **Full logging.** Every cloud/local request/response body appended to
   `run_dir/raw_requests.log` (timestamps, model, tokens). Reproducible
   debugging without re-running.
7. **Graceful stop.** `SIGINT`/`SIGTERM` flush a checkpoint after the current
   turn.
8. **Idempotent artifacts.** Turn dirs are never overwritten; resume asserts
   content hash of prior turns matches (drift detection).
9. **Contamination guard.** Notes are never passed to DSS; a post-turn
   validation asserts no note string leaked into `user_input`/`instr_prompt`/
   engine context and aborts loudly on violation (see §6).

---

## 8. Test matrix

```
spreadsheets     = [cozy_mystery, noir_detective, fantasy_epic, sci_fi_heist, horror, romance, western,
                    cyberpunk_thriller, greek_mythology_retelling, postapocalyptic_survival, heist_caper,
                    gothic_haunted_mansion]
specificity      = [loose, mixed, exact]  # per-note mix inside each spreadsheet
guide_style      = [explicit, implicit]
director         = [on, off]     # cost-quality factor; may be a subset
arc_breaks       = [none, 3_arcs]  # exercise NEXT ARC scene machinery
judge_every      = 10            # judge cadence (config)
```

A full matrix = 7 × 3 × 2 × 2 × 2 = 168 runs × ~100 turns — far beyond a
weekend. Practical strategy:
- **Phase 1 (short):** 2 genres × 2 specificity × 1 guide style, ~100 turns
  each, director on, no arc breaks → validate harness + metrics + judge
  calibration.
- **Phase 2 (matrix):** run the full matrix as **background jobs**, one
  spreadsheet+config per `run_dir`, queued sequentially on the V100, writing a
  summary table as runs complete.
- **Phase 3 (recall focus):** on the best genres, run the `character_echo` +
  arc-break variants to probe the recall path specifically.

Each run is fully independent (own `run_dir`) so the matrix can be extended,
cancelled, or re-run without touching prior results.

---

## 9. Implementation plan

**Milestone 0 (prerequisite): scene-bounded `last_x` prompt fix (Q5) — DONE.**
Implemented 2026-08-09: `_scene_dialogue_window` helper in
`agents/summarizer.py` (bounded by `last_x_max=8` / `last_x_min=2`, flat-6
fallback, degenerate-boundary guard), wired into `get_retrieval_context`; 
`base_state["truncation_length"]` raised 16384 → 32768; hermetic regression
`tests/scene_window_test.py` wired into `run_tests.py` (9 checks green; full
hermetic suite green). The soak can now proceed.

1. **`spreadsheets/` library** — define the notes schema + validator
   (`validate_spreadsheet.py`: note-type taxonomy, needle paths resolvable
   against `subjects_schema.json`, `plant`/`recall` turn sanity checks) + 7
   genre files, each with dozens of typed notes at a mix of specificity.
2. **`cloud_client.py`** — thin wrapper: `CloudModel(base_url, api_key, model)`
   reading the key from `auth.json` (or `OPENCODE_GO_API_KEY` env), JSON-mode
   helper for judge output, retry/backoff, token accounting. Default model
   `deepseek-v4-flash`. (Reuses the pattern from
   `live_soak.LiveSoakModel._complete`.)
3. **`long_horizon_soak.py`** — orchestrator: config parsing, run dir setup,
   guide/judge prompts (notes-windowed), turn loop (engine via the same runtime
   binding as `live_director._make_summarizer` + real `summarize_latest_state`),
   state needles + echo/recall probes, contamination guard, checkpointing,
   resume, smoke mode, report writer.
 4. **`report.py`** — aggregate metrics → `report.md` + `report.json` +
    per-note retention/recall/attribution table, style + fidelity curves,
    and the final-overview section (when `final_overview.json` exists).
 5. **Final overview (end-of-run judge pass)** — after the last turn, the judge
    reviews the WHOLE chat (full transcript + all notes + per-turn judge records
    + DSS private memory) and writes `run_dir/final_overview.json`:
    `overall_score`, `verdict`, `arc_progression`, `style_consistency`,
    `memory_fidelity`, per-note final adherence, `strengths`/`weaknesses`,
    `summary`. Backfill for already-completed runs:
    `python long_horizon_soak.py --final-overview <run_dir>` (needs only the
    cloud judge + spreadsheet; no local server required).
 6. **Wire into `run_tests.py`** as `--long-soak <spreadsheet>` (opt-in, never
    part of the default suite) or as a standalone CLI entry.
 7. **V100 serving** — exact `CMD_FLAGS_m.txt`/`CMD_FLAGS_p.txt`/`strt mp`
    flow + key documented in §0 (LMDeploy + `--loader LMDeploy --cache-type q8`,
    `--n_ctx 32768`, key `764d7ca1-...`, server currently running on
   0.0.0.0:5000). Remaining: a health-check/restart helper for overnight runs.

### Open questions / risks
**Resolved 2026-08-09.** Detailed analysis and decisions below.
- **Q1 Pipeline depth → DECIDED: full production path** (`summarize_latest_state`), embedding on GPU 2 or `--cache-max-entry-count 0.4`.
- **Q2 Judge drift → DECIDED:** separate guide/judge model IDs; rest of mitigations adopted.
- **Q3 Paraphrase probe → DECIDED:** two-tier detection + synonym needles + false-positive guard.
- **Q4 Guide hiding outline → DECIDED:** windowed notes + split echo note halves.
- **Q5 Context growth → DECIDED:** implement the scene-bounded `last_x` prompt fix **before** the soak (see below).
- **Q6 Judge memory view → DECIDED:** bounded `FormattedData` summary + state diff.
- **Q7 Cloud quota → DECIDED:** `deepseek-v4-flash` (0731) for both roles.
- **Q8 Guide persona → DECIDED:** name1 = in-fiction character, per genre.

---

## 10. What "done" looks like

- `python long_horizon_soak.py --spreadsheet spreadsheets/noir_detective.json
  --turns 100 --guide-style implicit --specificity-profile mixed` completes 100
  turns unattended, resumes cleanly after a kill, and writes
  `runs/noir_detective/` with full artifacts + `report.md`.
- The report shows per-note retention/recall/attribution (guide-failure vs
  DSS-loss vs supersession), style + fidelity curves, and the echo/recall
  probe results.
- The matrix runner produces a comparison table across genres/styles, showing
  where retention and style-consistency decay.
- Findings feed back into the strategy review (§3 soak / §4 design forks):
  e.g. tagged-text fallback for low-compliance genres, director on/off cost
  justification, scene/arc-boundary retention cliffs, and whether the
  future "explicit notes to dayna_ss" feature is worth building (i.e. does
  hidden-note steering leave measurable gaps that explicit steering would fill).

---

## 11. Open questions — detailed analysis

### Q1. Pipeline depth: `summarize_latest_state` (full production) vs per-subject `DataSummarizer.generate`

There are two levels of engine integration, and the choice changes what the
soak actually measures.

**Level 1 — per-subject (what `live_soak.py` does today).**
Each turn, for each subject in the schema, call
`DataSummarizer.generate(subject, data, schema_class)`. This exercises the
gate-check → branch-query → field-update pipeline against the *current state
files* and writes them back. It is fast, deterministic, and already proven.
**But it skips the parts of the engine the long-horizon test cares about most:**

- `retrieve_and_format_context` / `get_retrieval_context` — the llama_index
  embedding + retrieval layer (`all-mpnet-base-v2`, spaCy, NLTK, the persisted
  `message_index` vector store). This is where "what does DSS remember from 70
  turns ago" actually comes from. Without it, an echo/recall note ("wait for
  DSS to recall Ruiz") cannot be tested at all — there is no retrieval to
  recall from.
- Scene/chapter/arc boundary detection (`is_new_scene_turn`,
  `force_next_arc`, message-node bookkeeping). The `location_turning_point`,
  `foreshadow`, and arc-break factors are meaningless without it.
- Message summarization + chunking (`summarizer.generate(...)` writing summary
  chunks into `history_path/message_index`), which feeds the retrieval layer.

**Level 2 — full production (`summarize_latest_state`).**
`script.py` calls `summarizer.summarize_latest_state(output, user_input, state,
history)`, which internally does `prepare_context` →
`retrieve_and_format_context` → per-subject `DataSummarizer.generate` →
chapter/arc checks → message summary + chunking. This is literally the
production path, so a soak through it tests what users actually get. The costs:

- Loads the embedding model (`all-mpnet-base-v2`, ~420 MB) plus spaCy/NLTK.
  In `lmd-tg` this imports and runs, but VRAM is tight (V100 GPU 1 already
  holds the 9B at ~14.3/16 GB). The embedding model is small and can run on CPU
  or the idle GPU 2 — needs an `--embed-device cpu|2` flag and a smoke check.
- Each turn makes extra LLM calls (message summary + chunking on top of the
  per-subject updates), so wall-time and token cost per turn rise.
- `summarize_latest_state` needs a fully wired `Summarizer` (history_path,
  schema_parser, `last.context`, retrieval context) — more setup than the
  per-subject path, and it mutates shared `history` state.

**Recommendation (ADOPTED):** run **Level 2 (full production path)** for the
soak — anything less silently drops the retrieval and chunking machinery, which
are the actual subject of a long-horizon *memory* test. Keep Level 1 as the
smoke path (`--smoke 5` on Level 1 validates a spreadsheet/config fast), then
run the real thing on Level 2.

**VRAM decision:** run `all-mpnet-base-v2` on GPU 2, **or** add
`--cache-max-entry-count 0.4` to `CMD_FLAGS_m.txt`. LMDeploy TurboMind reserves
more VRAM than the model actually needs (its memory-management implementation),
so lowering the cache-entry count frees headroom without hurting the model.
Whichever is chosen is recorded in the run manifest. A smoke check must assert
the retrieval layer actually feeds the prompts (non-empty context with an empty
`message_index`), else the echo test silently degrades to "no memory to recall".

### Q2. Judge drift / self-grading

**Problem.** The same cloud model (default `deepseek-v4-flash`, Q7) both
*writes* the user turns (guide) and *scores* them (judge). Risks:

- **Leniency bias:** the judge implicitly credits its own authorial intent;
  per-note `planted`/`echoed` statuses may be inflated because the judge knows
  what it meant.
- **Drift across a 100-turn window:** later judge calls run against much longer
  windows, so raw scores are not directly comparable over time — a 4/5 at turn
  10 vs 4/5 at turn 95 may mean different things.
- **Prompt-sensitivity:** small changes in the judge's context (how the DSS
  memory summary is formatted) shift the whole scale.

**Mitigations (adopted).**

- **Separate model IDs for guide vs judge** (default same model):
  run one comparison with guide=`deepseek-v4-pro` / judge=`glm-5.2` to measure
  whether scores move materially. If they do, bias is real and the two-role
  split becomes the default. (Validated against OpenCode's API: different model
  IDs on the same endpoint work as long as prefix matching of model names works;
  smoke test confirms the exact model IDs before the matrix starts.)
- **Calibration anchors:** in each judge call, include 2 fixed reference
  samples (one deliberately good, one deliberately broken) drawn from a
  pre-scored pool; the judge's scores on these anchors reveal scale drift and
  let the report rescale the live scores (e.g. "live 4.2 vs anchored 4.0").
- **Score-decomposition honesty:** the deterministic string probes are the
  *ground truth* backbone; the judge's scores are treated as *secondary,
  drift-annotated* metrics, never the primary retention number.
- **Window normalization:** judge always receives a fixed-shape window (last
  `JUDGE_WINDOW` exchanges + a *bounded* DSS-state summary, §Q6), not the full
  history, so the input the judge sees stays roughly constant in scale.

### Q3. Exact-string recall probe is brittle under paraphrase

**Problem.** A note planted as "Ruiz" may come back in the reply as "his old
partner" with no string overlap. `live_soak` already showed this: "gravelly
voice" never appeared as a string even when the concept was retained. So the
deterministic probe alone undercounts recall.

**Mitigations.**

- **Two-tier recall detection:** (1) *exact* — the needle string (or a
  lemma-stemmed variant) appears in reply/state; (2) *semantic* — the judge
  marks `echoed` when the reply clearly refers to the planted entity. Report
  both; exact is the floor, semantic is the ceiling.
- **Canonical needle set:** each note carries `needle_syns` (Ruiz → ["Ruiz",
  "his old partner", "the partner"]) so the deterministic probe can match
  near-synonyms without a judge call.
- **False-positive guard:** a recall is only counted if (a) the *state* still
  holds the detail (so it is memory, not the guide re-planting it) and (b) the
  guide did not reintroduce the name in the immediately preceding user turn
  (else it is echo-by-hint, not echo-by-memory). This is the echo probe's job
  (§6).

### Q4. Guide hiding the outline

**Problem.** If the guide sees the full notes list, it can front-load future
beats ("introduce Ruiz at turn 8" + "Ruiz returns at turn 70" both visible ⇒ it
keeps Ruiz alive itself, and the echo test never isolates DSS memory).

**Mitigations.**

- **Windowed notes (§6):** the guide receives only the notes whose `plant` /
  `recall.due` / twist window covers the current turn. It never sees the
  spreadsheet's full outline.
- **Split the echo note into two invisible halves:** the plant-side note
  ("introduce Ruiz early, tied to Vivienne") is given at plant time; the
  recall-side note ("a dockside reunion should surface Ruiz") is only given
  when the recall window arrives. The harness keeps both halves; the guide sees
  each at its own time, so it cannot deliberately hold Ruiz in play.
- **Guard-rail assertion:** the harness verifies the guide's *output* at a
  recall turn did not re-state the plant note verbatim (contamination guard,
  §6/§7) — a recall that only works because the guide repeated its own
  instruction is flagged as `echo-by-hint`, not counted as DSS memory.

### Q5. Context growth over 100 turns vs the local model's `n_ctx` (DECIDED: fix first)

**Correction (user review):** an earlier draft of this section claimed
`custom_state` is built once and cached from turn 0. That was wrong. The
engine **regenerates `custom_history` every turn**: `retrieve_history_path`
hashes the full `history`, so the hash changes as the conversation grows,
`get_retrieval_context` then recreates `self.last` each turn
(`if not last or history_path != last.history_path`), `history_length` resets
to `None`, and the line-1340 gate (`if not self.last.history_length is None`)
does not early-return. The gate only dedups the *second* `prepare_context`
call within a single turn (`generate_instr_prompt` then `summarize_latest_state`).

**The real open question is boundedness of the prompt *size*, not staleness.**

- **Recent-dialogue window:** `get_retrieval_context` capped the visible window
  at `last_x = min(len(history), kwargs.get("last_x", 6))` — a **flat 6
  exchanges across all scenes** (TODO "Get all in current scene"). As scenes
  accumulate, the last 6 exchanges can span a scene boundary, pulling
  cross-scene dialogue into the prompt and diluting the current scene's context.
  **[FIXED — §12/Milestone 0: now scene-bounded via
  `_scene_dialogue_window`.]**
- **Retrieved-context blocks:** each turn `retrieve_context` injects
  characters/groups/events (importance-decayed) + 5 semantically similar
  messages. These blocks grow with the story; importance decay bounds them in
  principle but the ceiling is unmeasured over 100 turns.
- **Truncation:** `base_state["truncation_length"] = 16384` and the served
  `--n_ctx 32768` bound the final prompt; if context + dialogue exceed them,
  tgwui truncates — silently cutting retrieved memory. **Unverified at 100
  turns** (longest live run so far: 20).

**Decision (user-confirmed): implement the prompt-management fix first.**
The intended behavior is to bound the visible dialogue to the **current
scene**: `scene_bounds + last_x` (i.e. `last_x` **within the current scene**).
This makes each turn's prompt contain the current scene's recent
dialogue plus retrieved structured memory at a bounded size, and later scenes
enter via retrieval rather than raw replay. Full spec below.
**[IMPLEMENTED — see §12.]**

### Q6. How to show the judge "DSS's memory" without overflowing its context

**Problem.** After 100 turns the per-subject JSON files are large; feeding them
raw to the judge each call is expensive and noisy. But the judge cannot score
`memory_fidelity` without seeing what DSS believes.

**Mitigation.** Reuse the engine's own serialization: `FormattedData` /
`format_general_info_static` already produce compact LLM-friendly renderings of
the structured subjects. Feed the judge a **bounded DSS-state summary** — the
same `FormattedData` view the engine uses for its own prompts, capped to the
subjects referenced by the active notes, plus a per-turn diff (`what changed in
state this window`) so the judge scores against a small, focused delta rather
than the whole memory. This keeps judge inputs constant in scale (§Q2) and
avoids re-implementing a serializer.

**Decision (user-confirmed):** adopt as the best current option; may be revised
if the judge's fidelity scores prove unstable against the bounded summary.

### Q7. Cloud quota across the matrix (DECIDED)

**Decision (user-confirmed):** use **`deepseek-v4-flash` for both guide and
judge** (default). The `deepseek-v4-flash-0731` checkpoint is the actual model
behind `deepseek-v4-flash`, is sometimes **better than Pro**, and is far
cheaper (~158K requests/mo vs ~17K/mo) — so the matrix fits comfortably and
role-splitting for cost is unnecessary. The `--judge-model` / `--guide-model`
flags remain for experiments (e.g. judge=`glm-5.2` for the self-grading bias
test in Q2), defaulting to `deepseek-v4-flash`.

Remaining mitigations still apply: budget-aware matrix runner (skip a run if it
would exceed the cap), `judge_every` cadence knob, and a 1-call smoke test of
the `auth.json` `opencode-go` key against the Go endpoint before the matrix.

### Q8. What "name1/name2" means for the guide's persona

**Open issue:** in the example schema, `name1` (User) is the human-role and
`name2` (Assistant) is the character DSS plays. The guide writes name1's turns.
For genres where the "user voice" is itself a protagonist (Marlowe the
detective), the spreadsheet must define whether name1 is (a) an in-fiction
character whose lines advance the story, or (b) an out-of-fiction narrator.
This changes guide prompting and judge expectations of style. Default: name1 =
in-fiction character the guide role-plays; spreadsheet `characters.name1` must
define it, and the guide's system prompt is explicit about which one.

---

## 12. Q5 fix spec — scene-bounded `last_x` prompt management

Prerequisite milestone for the soak (see §9 Milestone 0). Touches the engine
(`agents/summarizer.py`), not the harness.

### 12.1 Goal

Bound the recent-dialogue window injected into each reply prompt to the
**current scene**, instead of a flat `last 6 exchanges across all scenes`.
Two properties must hold after the fix:

- **Bounded size:** the dialogue window is `≤ last_x_max` exchanges, so the
  prompt stays within `truncation_length`/`n_ctx` headroom at 100 turns.
- **Scene-scoped recency:** the window never reaches back across a scene
  boundary — a scene-1 fact reaches a scene-2 prompt *only* via retrieval, not
  raw dialogue replay. This is the property the soak's `character_echo` /
  `location_turning_point` notes depend on.

### 12.2 Current behavior (confirmed) — as of the pre-fix baseline

- `get_retrieval_context` (line 1622, pre-fix):
  `last_x = min(len(history), kwargs.get("last_x", 6))`
  — flat 6, TODO "Get all in current scene" unimplemented (now implemented).
- `last_x_messages = self.format_dialogue(state, history[-last_x:])` (1623),
  then injected into `custom_history` in `retrieve_and_format_context`
  (1436-1437): `["What were the last {last_x} exchanges?", last_x_messages]`.
- `custom_history` is **rebuilt every turn** (not cached): `retrieve_history_path`
  hashes the full `history`, so the hash changes as the conversation grows,
  `get_retrieval_context` recreates `self.last` each turn
  (`if not last or history_path != last.history_path`), `history_length` resets
  to `None`, and the line-1340 gate does not early-return. Confirmed by hashing
  `history` at successive turn lengths (distinct hashes). The line-1340 gate
  only dedups the second `prepare_context` call within a single turn.
- Scene boundaries are tracked per-scene in the persisted events data as
  `_message_node` fields (`scene.start._message_node` / `scene.end._message_node`,
  e.g. `"140_1_1"`), written in `summarize_latest_state` (1223-1238) and used
  to tag `message_index` chunks with `scene_id`. `prepare_context` also sets
  `self.last.new_scene_start_node` (696) when a new scene turn is flagged.

### 12.3 Design

Compute the scene window at prompt-build time and use it as `last_x`:

```
scene_start_node = current_scene.start._message_node   # or new_scene_start_node
scene_start_idx  = parse_node_index(scene_start_node)  # e.g. "140_1_1" -> 140
                  # message nodes are 0-based message indices (user@even 2*turn,
                  # assistant@odd 2*turn+1); scene_start_idx maps to a history index
window_start = scene_start_idx // 2                    # history entry (exchange)
last_x       = min(len(history) - window_start, last_x_max)
last_x       = max(last_x, last_x_min)                 # never shrink to 0
```

- **`last_x_max`** (new config, default e.g. 8): hard ceiling on the dialogue
  window regardless of scene length — keeps the prompt bounded even for very
  long scenes.
- **`last_x_min`** (default 2): floor so very fresh turns (start of a scene,
  no `_message_node` yet) still get some recent dialogue.
- **Fallback:** if no scene boundary is resolvable (first turn, or
  `_message_node` absent), fall back to today's `last_x = min(len(history), 6)`.
- The `kwargs.get("last_x", ...)` override stays (harness can force a window).
- Node→history mapping detail: message nodes are `"{index}_1_1"` where `index`
  is the **0-based** message index — user messages at even `2*turn`, assistant
  at odd `2*turn+1` (`len(history)*2` at line 696 for a new scene turn ⇒
  `scene_start_exchange = node_idx // 2`). **Verified empirically during
  implementation** (see §12.6).

### 12.4 Where to change — **IMPLEMENTED**

- `agents/summarizer.py` `get_retrieval_context`: `last_x` now comes from
  `self._scene_dialogue_window(history, context_retriever, kwargs)`; the
  existing `retrieve_context(current_context, last_x_messages)` call is
  unchanged.
- New helper `Summarizer._scene_dialogue_window(history, context_retriever,
  kwargs)` reads the persisted scene boundary
  (`context_retriever.get_current_scene()` → `start.when._message_node`, with
  `new_scene_start_node` + flat-6 fallbacks), returns the exchange-based
  `last_x` (bounded, floor, degenerate-guard).
- `retrieve_and_format_context` unchanged: it already formats whatever
  `last_x`/`last_x_messages` it receives.
- `base_state["truncation_length"]` raised 16384 → 32768.

### 12.5 Regression fixtures (hermetic)

Add to the hermetic suite (no LLM / llama_index / torch):

1. **Bounded-window fixture** — **IMPLEMENTED as `tests/scene_window_test.py`**
   (drives `Summarizer._scene_dialogue_window` directly): asserts clamping to
   `last_x_max`, non-clamped in-scene windows, fresh-scene/degenerate-boundary
   handling, `last_x_min` floor, both fallbacks, short-history cap,
   `new_scene_start_node` fallback, and `last_x_max` override. 9 checks; wired
   into `run_tests.py` as `[scene_bounded_window]`. Green as of 2026-08-09.
2. **Scene-recall-only fixture** (fact planted in scene 1 survives a scene
   transition, is absent from scene-2 `last_x_messages`, still present in
   persisted subject state) — **NOT implemented**: this requires driving the
   full `get_retrieval_context` + llama_index path (embedding model, persisted
   `message_index`), which the hermetic suite deliberately avoids. Deferred to
   Level-2 soak validation (smoke turns 90-100, §Q1/§12.6): assert a scene-1
   fact does not appear in scene-2 dialogue but is present in state.
3. Existing `gate_check_update_vs_preserve` and `soak_conversation` goldens
   must remain green (the change only affects the *recent-dialogue window*, and
   those fixtures drive `DataSummarizer.generate` directly, so they are
   unaffected — but re-run them to catch regressions). **Confirmed green.**

### 12.6 Verification steps / risks

- **Node-scheme mapping — VERIFIED.** Message indices are **0-based**, user at
  even `2*turn`, assistant at odd `2*turn+1`; `new_scene_start_node =
  f"{len(history)*2}_1_1"` (line 696) so `scene_start_exchange = node_idx // 2`.
  Confirmed by tracing `script.py:142` (`index = len(history)*2` user input) and
  `summarizer.py:767/1188` (`len(history)*2` / `len(history)*2-1`), and by unit
  tests in `tests/scene_window_test.py`.
- **Scene-boundary timing — DECIDED.** `_message_node` is written during
  `summarize_latest_state` *after* the reply prompt is built for the same turn.
  The helper reads the **persisted current scene** (via
  `context_retriever.get_current_scene()`, `start.when._message_node`) as
  primary, `self.last.new_scene_start_node` (freshest signal, set in
  `prepare_context` line 696) as fallback, then flat 6. A freshly-flagged scene
  turn may see the previous scene's boundary for one turn — accepted; the
  window is only ever *larger* by at most the previous scene's tail, never
  unbounded.
- **VRAM / n_ctx:** after the fix, the smoke soak (turns 90-100) must report
  per-turn prompt token counts under `n_ctx` with headroom (§6), and the
  retrieved-context blocks must be sampled for size (importance decay should
  hold them, but measure, don't assume). `truncation_length` is now 32768
  (`base_state`), matching `--n_ctx 32768` on the server.

## 13. Engine regression fix — empty subject states (2026-08-10)

**Reported:** the cozy_mystery run showed every per-turn custom-history state
rendered as `<EMPTY>` for characters/groups/elements — the retrieval layer was
feeding DSS nothing, and only turns 0/11/12 had subject files at all.

**Root cause — two compounding regressions in `agents/summarizer.py`:**

1. **Fresh-chat flag clobbered** (line 679). Commit `2af4d9cf` ("fix: Strip
   think section in data generations") changed the next-scene handling from
   `if persistent_ui_state.get("next_scene"): is_new_scene_turn = True` to an
   **unconditional** `is_new_scene_turn = persistent_ui_state.get("next_scene", False)`.
   `get_retrieval_context` sets `is_new_scene=True` for a brand-new chat
   (line 1555), but `prepare_context` then overwrote it to False (checkbox
   unchecked). Result: `_populate_from_first_scene` never ran on turn 0, so
   subjects stayed empty `{}` placeholders for the entire first scene.

2. **Early-return dropped subject files** (line 1110). The
   `if not has_archived_scenes and not is_new_scene_turn: return` early-return
   (introduced by `d55ab7a3` "enhance first-scene population logic", marked
   UNTESTED) skipped the DataSummarizer on regular first-scene turns — and
   because each turn hashes a NEW history dir, the populated subjects never
   propagated to the next turn's retrieval dir. Even with (1) fixed, turns 2+
   would have re-read empty dirs.

**Fix (both in `agents/summarizer.py`):**

- Restored the conditional: only *set* `is_new_scene_turn = True` when the
  checkbox/`NEXT SCENE:` prefix is present, never force it to False — so the
  fresh-chat flag survives `prepare_context`.
- On the first-scene early-return path, copy all subject `*.json` files from
  `last_history_path` to `new_history_path` (schema/templates were already
  copied by `30cf00d7`), so the next turn's retrieval finds them.

**Verified:** fresh Level-2 3-turn smoke — turn 0 shows populated characters
(Evelyn, Prune, Martin Kettlewell, Mrs. Arbuthnot), 13-16 elements, current
scene, general_info in every turn's dir + state snapshot; hermetic suite still
green.

## 14. Frozen-output-loop fixes (2026-08-10)

Follow-up to the subagent diagnosis of the cozy run's frozen replies (only 3
distinct reply files across 18 turns, driven by `temperature=0.0` greedy
decoding + the model re-reading its own output + frozen memory). Fixes applied:

- **Temperature** — `LocalModel` in `tests/long_horizon_soak.py` now takes a
  `temperature` (default **0.8**) and sends it in the payload; the hardcoded
  `0.0` is gone. Greedy decoding was locking one bad reply forever.

- **`instructions.json` forward-copy bug** — the section-13 early-return copy
  path (`agents/summarizer.py:1110-1116`) copied ALL `*.json` forward,
  *including* `instructions.json`. But instructions are a seed-keyed **cache**
  (`generate_instr_prompt` looks up `input_key=str(state["seed"])`,
  summarizer.py:790), not a subject. With a fixed `--seed 43` every turn hit
  the cache and reused the same byte-identical stale instructions forever.
  Fixed: `instructions.json` excluded from the forward-copy set
  (summarizer.py:1112) → each fresh history dir cache-misses → per-turn
  regeneration.

- **`current_scene.now` frozen** — SceneState (the `now` field's schema) had the
  `always: [query_branch_for_changes]` trigger but **lacked** the
  `branch_query_prompt_template`/`branch_update_prompt_template` defaults that
  the action requires; `data_summarizer.py:299` (`if bq_template and
  bu_template`) silently no-ops, so the per-turn refresh never fired and `now`
  was written once. Fixed in `user_data/example/subjects_schema.json`: SceneState
  now defines both templates (modeled on GeneralInfo's working ones, scene
  specific). `start` stays frozen via SceneStart `no_update: true`.

## 15. Voice carrier = `general_info.writing_style` (production parity, 2026-08-11)

Follow-up to the frozen-output-loop work: instead of bolting harness-level
prompt guards onto the reply (turn-taking markers, second-inference catch-guards),
the DSS voice is carried by the **production field** `general_info.writing_style`.

- **Production already renders it.** The `general_info` format template
  ("Writing Style --- {{writing_style}}") is `_context_order` `to_context: true`
  (format_templates.json:36), so it is appended to the system context every
  turn; `_format_general_info_static` (summarizer.py:1357) re-injects it once a
  scene archives. The soak had been *bypassing* this by stuffing the directive
  into the raw `state["context"]` line and relying on the engine's lossy LLM
  paraphrase of it into `writing_style`.
- **`SoakRun._seed_general_info_style(history_path)`** (long_horizon_soak.py,
  called in the Level-2 turn right after `generate_instr_prompt` returns) pins
  `general_info.json`'s `writing_style` to the exact spreadsheet `dss_directive`
  in the session dir; the engine's copy-forward (summarizer.py:1111-1115)
  propagates it to every subsequent turn's dir. Turn 0 still uses the LLM
  paraphrase (the seed lands after turn 0's prompt is built) — an accepted
  one-turn lag, and the "Your voice" line in `build_state` context still feeds
  the turn-0 seed.
- **Engine prompt decoupling.** summarizer.py:804/820/860/870 now reference
  "the Writing Style directive (general_info.writing_style) in your system
  context" instead of the literal harness label "Your voice ({name2}, DSS
  turns)". All four call sites updated in one pass.
- **Operational rules live in the directives.** All 7 spreadsheets' `dss_directive`
  now end with hard rules: "narrate {name2} strictly in the third person —
  never first person ('I', 'my', 'me'); never quote or repeat {name1}'s lines
  back verbatim — respond to them, do not re-speak them; end on action or
  observation, not summary." Because the directive flows through the production
  renderer verbatim, the strictness is enforced where production expects it.

## 16. 3/5-verdict root-cause fixes: memory freeze, voice anchor, instruction cleanup, early-stop (2026-08-11)

A 20-turn run (`cozy_mystery__48ec5b7f`) came back 3/5 with every judge record
reporting the same DSS failures (first-person slips, verbatim re-quoting of
Prudence, recycled damp-boot/rippling-cloth imagery, stale private memory).
Subagent diagnosis found four mechanisms, all now fixed:

- **M4 — memory froze (biggest).** `summarize_latest_state` had an early-return
  (`agents/summarizer.py:1133`, added during the modular refactor) that skipped
  the DataSummarizer entirely whenever no scene was archived and no transition
  was detected — i.e. EVERY turn of the first scene. The copy-forward then just
  preserved byte-identical `current_scene/characters/events` (md5-verified across
  20 turns). **Fix: the early-return is removed** — the DataSummarizer now runs
  every turn (except initial population on the fresh-chat turn), so `always`-triggers
  fire and `current_scene.now` etc. actually update. Verified: `now.who/now.why`
  change per turn (e.g. "The village green"→"Market Lane"). Cost: many more local
  LLM calls per turn (~43 in turn 2 of a smoke) — this is production behavior.
- **M1 — voice anchor (engine).** `generate_instr_prompt` built the reply prompt
  as `This is the latest user input: <name1's verbatim first-person turn>` then a
  single weak `REMEMBER: Write from {name2}'s perspective` (ambiguous — could read
  as first-person-as-name2). The 9B model anchored on the big first-person block.
  **Fix (engine):** a `[VOICE: ...]` switch-frame is now inserted between the quoted
  user input and the reply instructions, and the REMEMBER line is explicitly
  THIRD PERSON ("Narrate {name2} in the THIRD PERSON from outside — never from
  {name2}'s own 'I' ..."). Applied to both the `do_instr` and `do_instr=False`
  branches (rule #36).
- **M3 — hard rule buried (engine).** The reply prompt only *referenced* the
  writing-style directive in the buried system context. **Fix:** the exact text is
  now inlined as `WRITING STYLE DIRECTIVE (you MUST follow it):` right above
  "Adhere loosely", pulled from `retrieval_ctx.general_info.writing_style` (the
  production carrier — seeded by the harness, so turns 1+ carry the exact
  `dss_directive` with hard rules; turn 0 falls back to the system-context
  reference).
- **M2 — instruction block planted repetition (engine).** The per-turn generated
  instructions (same weak local model) literally dictated imagery that then
  appeared verbatim in replies. **Fix:** `_clean_generated_instructions` —
  a deterministic post-check (no extra inference) that drops near-duplicate
  sentences (template lock-in) and sentences that near-verbatim echo the user
  input; fires only when clearly degenerate.
- **Early-stop (harness).** New `--abort-after N` (default 2, 0 disables): the
  judge and auditor prompts now carry an ABORT CLAUSE telling them to set
  `"abort": true` ONLY for catastrophic, multi-turn failures (gibberish, wholesale
  hard-rule violation across turns) — a single style slip or missed note is never
  enough. The harness stops the run after N consecutive `abort:true` flags (turn
  still checkpointed, then report + final overview run). Judge/auditor JSON gain an
  additive `abort` field (report/dashboard ignore it).
- `run_tests.py` passes through `--abort-after`.



---

## Post-45adee46 batch: instruction-loop fixes (C1/P1a/P1b/P2a/P2b/P4) — 2026-08-15

Drivers: 45adee46 (Type 2 Mode B, 40t) hit 2.5/5; the subagents pinned the failure
chain — instruction generator anchors on the STABLE top of the subject block
(byte-identical Evelyn entry + never-pruned first elements) instead of the live
scene; `current_scene.now` time frozen ("Late morning" all 16 turns, specific_time
regressing); replies transcribe the recycled instruction blocks; 3/13 judge turns
dropped for non-JSON.

### Shipped

- **C1 — log `custom_state["context"]`** to `turn_XXX/context.txt` beside
  `instr_prompt.txt` (L2 writes the real system context; L1 writes empty). The
  dashboard's per-turn "State snapshot" viewer now lists `context.txt` (plain text
  handled in `_load_state_file`).
- **P1a — CURRENT SCENE recap in the instruction-gen prompt.** `generate_instr_prompt`
  now prepends a compact authoritative recap of the LIVE `current_scene`
  (`_current_scene_recap`): scene number, `what`, `now.when`, `now.where`, present
  characters+locations. Grounds the plan in the actual scene instead of the stable
  context head.
- **P1b — CurrentScene.now templates always advance the in-story clock.** Both
  schemas' `now_query_prompt_template` now treat time/movement/adds/removes as
  NOT-unchanged (answer YES), and `now_update_prompt_template` gains a hard rule to
  ALWAYS write an updated `now.when.specific_time` (e.g. "11:15") whenever the
  exchange implies a clock time (never 'N/A' mid-scene), moving `now.when.time` in
  step; example updated with a specific_time line.
- **P2a — STRICT IMAGERY RULE (instr item 14).** The generator may not reuse
  imagery/gestures/props/beats from earlier replies or re-anchor on stored memory
  objects, except as a deliberate poignant/comedic callback; every reply must
  introduce a fresh concrete detail.
- **P2b — semantic anti-repeat guard.** New `_instructions_phrase_overlap` (longest
  verbatim common substring ≥ 120 chars) catches blocks that reuse whole phrases
  from the previous turn even at byte-ratio ~0.08 (calibrated on 14714d8e: loop
  fired 13/39 with the near-copy at LCS=870; healthy dded7733 fired 3/29 borderline).
  Wired into both the regenerate trigger and the retry-acceptance check.
- **P4 — judge/attribution/notes.** (1) `json_complete` gets a THIRD attempt that
  feeds the actual parse error back to the model ("your previous output could not be
  parsed as JSON: ... return ONLY valid JSON") — re-sending the identical prompt
  twice was rarely the right retry. (2) `report.py` attribution split: a note whose
  needle appears in the guide's story text but never in DSS state is now
  `dss_retention_loss`; `guide_failure` is reserved for notes whose needle never
  appeared in the story at all (or with no plant schedule → `not_planted`). (3)
  Duplicate cozy notes c14/c26 (both plant Marmalade at turn 6) merged — c26 removed,
  c14 kept (has the recall).

### Design ideas parked (user)

1. **In-world timestamps on enumerated messages.** For `message_mode: enumerated`,
   add the in-world time/date of each message to its `N. 'name' >> ...` line.
2. **Prompt layout question** — does this structure make sense?
   ```
   [GENERAL INFO]
   [CURRENT SCENE.START]
   [<ENTITIES>]

   [MESSAGES]
   [CURRENT_SCENE.NOW]
   [PROMPT]
   ```
3. **Rolling retrieval placement.** Include *newly retrieved* entities inline before
   each message in `[MESSAGES]` to avoid prefix-cache misses from placing new
   retrievals at the top — with the caveat that interleaving new blocks in the middle
   risks LITM (lost-in-the-middle) attention issues.

## 22. Prefix-cache efficiency work (2026-08-15)

### 22.1 Per-call latency instrumentation
LocalModel `_complete` now logs every local call: `local-call <phase>/<step>: ttft=X.XXs in=PT out=CT tok`. Correlates soak calls with the LMDeploy server log (match on input_tokens). Labels come from phase_id/step_id threaded through generate_with_sse (already present) + generate_using_tgwui (NEW: added phase_id/step_id at all 6 DataSummarizer call sites — add_new_query, add_new, chapters/check_archive, chapters/archive, arcs/check_archive, arcs/archive; the engine Summarizer.generate_using_tgwui pops them from kwargs). Reply call labeled reply/generate.

### 22.2 Server-log reading
Good case: input=17244, matched [0,17024) 95% checkpoint-reused, 859 tok computed.
Bad case: input=31920, matched only [0,18176); computed [16384,31920) = ~15.5K tok for a 4-token output. Of the non-reused tokens: ~1.8K matched-but-uncheckpointed (GDN 4096-boundary checkpoint clamp — after this request the engine publishes ckpts every 2048 tok so the NEXT identical call reuses fully), ~13.7K genuinely new (never in trie — the full-history reply tail / first-per-turn subject re-renders). Both fine; the lever is the DSS-prompt tail layout.

### 22.3 Subagent audit findings (subjects_schema_sceneagg.json + default)
- Already prefix-optimal (DO NOT TOUCH): per-entry branch_update_prompt_templates (24K/9.3K shared), new_entry_prompt generation (17.7K), population (18.3K), now_query/now_update.
- Byte leaks ranked by (calls/turn × hostility): (1) SceneState field-update templates put item_name at char 33 → 5.5K schema block re-encoded per call; (2) mark_field never matches the format-marker identifiers (`.characters.Evelyn` vs `characters.entries.Evelyn`, summarizer.py:3441) → per-entry calls get the WHOLE subject map, re-encoded every entry; (3) gate checks / select_entries / add_new queries put branch_name at char 47-60 → schema+example after it never cached; (4) add_new routing-guide head.

### 22.4 Applied: static-first/dynamic-last reorders (both schemas, 22 templates each + arc/chapter value block)
Gate checks (Character/Group/Element/Events), select_entries (×3), SceneState/SceneWhy/StoryEvents.update, StoryEvents.importance_update, Arc/Chapter.update — all now end with a single trailing dynamic reference line ("Section to review: '{{ branch_name }}'", "Value to review: '{{ item_name }}'", etc.); the static body + schema + example precede it. Measured shared-prefix (two consecutive calls, different branch):
- default CharacterMap.gate_check: 47 → 335 ch (98%)
- sceneagg gate_check: 47 → 780 ch (99%)
- sceneagg select_entries: 60 → 1260 ch (99%)
- sceneagg new_entry_query: 275 → 1508 ch (87%)
Sceneagg templates: the SCENE CONTEXT block (per-turn constant) now sits before the reference too, so within-turn calls share head+recap+events+closer.

### 22.5 Deferred (need user sign-off — change what the model sees)
- (A) mark_field per-entry isolation: make identifiers match (normalize `.subject.X` → `subject.entries.X`) so per-entry "Current context" shrinks from the whole map to the single entry. Biggest remaining byte-leak on transition turns. Quality tradeoff: model loses cross-entry reference.
- (B) Hoist the schema snippet + example JSON + scene recap into the per-turn system context (custom_state['context'], already built once in retrieve_and_format_context) instead of re-embedding per call. Largest structural change; templates keep placeholders.
- Note: seed-104 run (Type 2 Mode B, 40t) is on OLD code — restart to pick up 22.1 + 22.4 (schema refresh happens at startup).

## 23. Whole-subject context re-statement gating (mark_field) — 2026-08-15

### 23.1 The problem
Every per-entry DataSummarizer call re-stated the ENTIRE rendered subject map near the generation boundary (`... \nCurrent context for 'X':\n {formatted_data.mark_field(...)}`) at 5 sites (gate check, branch query ×2, branch_update_only, field update). The block:
- sits AFTER the per-call entry reference → never in the shared prefix → re-encoded per call (O(map) each);
- is byte-different across calls (the target-line marker changes) → no cache reuse even within a subject;
- grows with the story (elements map 32KB→143KB by t15 in seed-89) and pressures the context window, forcing history truncation (max_prompt_tokens 54856 vs ctx 32768) — a QUALITY loss, not just perf.

mark_field does NOT isolate an entry — it toggles the marker annotation only; every non-target line keeps its content. So the tail was never "exactly what is what"; it was the whole map with one subtle annotation. Boundary accentuation of the TARGET is already provided by `{{ value }}` (the entry's full current data) in _create_update_prompt.

### 23.2 Decision (user-approved)
- `restate_map_context` config: `auto` (default) | `always` | `never`.
  - always = quality-first, full map every call (for small-map / short-run users).
  - never = perf-first, no boundary re-statement at all.
  - auto = keep the full map while len(mark_field output) <= `restate_map_threshold_chars` (default 10000 — cheap accentuation insurance for small maps, since a mid-story map re-encode is the dominant per-call cost), else drop it and append a COMPACT SIBLING ROSTER (`Other entries in this section:\n- name` from keys[:-1]) so per-entry calls keep boundary-near cross-reference names for relationships. The full map stays in the shared prefix (system context / internal-history retrieval), which is encoded once per turn and prefix-cached.
- Harness flags: `--restate-map-context {auto,always,never}` + `--restate-map-threshold N`, folded into the run-id hash (distinct dirs per mode), passthrough in run_tests.py.
- All 5 mark_field sites route through the new DataSummarizer._context_restatement (data_summarizer.py). `never` returns "" (no roster); roster is auto-mode only.
- Note: mid-run mode is a single cache checkpoint per subject (shape change once); per-subject maps flip small→large at most once.

## 24. Sticky-roll message summaries (engine upgrade, 2026-08-16)

### 24.1 Problem
Rolling mode (`message_mode=rolling`) drops messages outside `history[-last_x:]`
entirely. Out-of-window story context (earlier scenes, plants, decisions,
character developments) simply stops existing for the model — a hard retention
cliff over a long horizon (the 40-turn Type 2 Mode B runs showed exactly this:
notes from turns < 6 unreachable because the rolling window only ever shows ~6
exchanges).

### 24.2 Fix (implemented, engine-side)
Config `rolling_summaries: N` (>0 enables). In `retrieve_and_format_context`
(`agents/summarizer.py`), a single pair is injected ahead of the raw rolling
pairs:

```
Summaries of earlier messages (messages S-E, before the most recent window):
- [message 30] Prudence deduced that Mrs. Arbuthnot's map was a forgery...
- [message 31] ...
```

- Source: the accumulated `message_index` store in the current history dir
  (`is_summary` nodes). Verified empirically on the real 40-turn `b884af4a`
  store: each new dir's message_index holds the FULL accumulated summary set
  (whole-index persist + retriever instance reuse), so a fresh chunker load at
  retrieval sees the entire run so far.
- Sticky roll (`_rolling_summary_window`): the range reaches back at least
  `rolling_summaries` exchanges OR to the start of the last
  `ROLLING_SUMMARY_SCENE_ROLL` (5) scenes (`Math.max(N, last_5_scenes)`),
  whichever reaches FURTHER. Scene boundaries = archived
  `events.scenes[*].start._message_node` + current scene start; <5 scenes → all
  of them count. The range always ends where the raw window begins (no
  duplication).
- Sticky LOCK (2026-08-16, user requested for prefix-cache friendliness): the
  computed `start` is locked in memory on the Summarizer instance until the
  next scene turn (`_current_scene_key` = scene start message-node, else
  `_scene_number`). Within a scene, later turns reuse the same `start` and only
  advance `end` → the summaries block is byte-stable across turns, so the
  shared prefix survives for prefix caching (encode once, reused by every call
  in every turn of the scene). Scene change → recompute with the new bounds.
  In-memory only: a process restart legitimately recomputes (the prefix cache
  is gone with it anyway).
- Placement: inside `custom_state["history"]["internal"]` before the raw pairs
  → the harness `_engine_internal_messages` and every engine call that sees the
  dialogue see the summaries too.
- Bound interplay (2026-08-16): the harness `_engine_internal_messages`
  `--max-update-history` bound now PRESERVES the summaries pair and bounds only
  the raw rolling pairs behind it. Previously the pair sat at the head of the
  dialogue section and was trimmed on DataSummarizer per-entry calls — so the
  feature only ever reached REPLY calls (where prefill is cheap), never the
  per-entry update calls where prefill is the real cost.
- Harness: `--rolling-summaries N` (default 0 = off) → config; folded into the
  run-id hash; run_tests.py passthrough. Cross-doc: schema_types_and_test_matrix.md §3.

### 24.3 Open design question (a vs b) — "mentioned subjects" in summaries
The user asked whether per-message mentioned-entities should be stored as
(a) a separate message node, or (b) embedded in the raw text summary.
Assessment:
- Current state: `subjects_referenced` metadata is heuristic regex extraction
  (`_extract_entities`, characters/groups/events only) — one of the oldest
  pieces (RetrievalContext-era), NOT a sound baseline (user confirmed).
- (a) is cleaner/machine-readable (could drive the instruction generator's
  used-imagery awareness, judge grounding, retrieval) but costs +1 LLM call per
  message on a pipeline that is already call-heavy on transition turns.
- (b) is nearly free (one prompt clause) and lands the entity names IN the
  summary text, so the sticky-roll block already carries them; less structured,
  and the heuristic parser would need extending (elements/props are missing
  today — that's the real gap for the repetition tic).
- RECOMMENDATION: (b) now — extend the message-summarization prompt with a
  "Mentioned in this message: [comma-separated people/items/places]" tail and
  parse it back into `subjects_referenced` (extend to elements). Revisit (a)
  only if the free-text list proves unreliable for a structured consumer.

## 25. Message-node provenance: real history vs custom_state internal (2026-08-16)

### 25.1 The bug (`summarizer.py:952` TODO "SHOULD BE ORIGINAL STATE")
`current_message_node` (written as `start._message_node` on new entries) was
`len(custom_state["history"]["internal"]) * 2` — the ARTIFICIAL retrieval-context
list, bounded to the rolling window (~6-8 pairs) + the "What was the very last
exchange?" pair. Confirmed in cozy_mystery__b884af4a: every scene/event node
froze at `16_1_1` from turn 11 on while the story was 30+ messages in.

### 25.2 Fix (implemented, engine + harness)
- DataSummarizer takes `real_history` (same list `summarize_latest_state`
  receives: greeting pair + exchanges INCLUDING the one being summarized);
  `_resolve_current_message_node()` = `len(real_history) * 2` (matches
  `new_scene_start_node` / `_scene_start_message`), legacy internal-fallback
  only for construction sites without it. Threaded through serial +
  `_process_subject_parallel` + harness L1.
- `_scene_recap_text` (sceneagg Type-2 recap) read the artificial internal list,
  which in enumerated mode is laced with retrieval Q&A pairs; now renders real
  dialogue (name1/name2 per real pair, greeting marker skipped) when
  real_history is set.
- Harness parity: `build_state` handed the engine a greeting-ONLY
  `state["history"]["internal"]`, which silently killed generate_instr_prompt's
  R1 negative-exemplar block (summarizer.py:1083). Now carries greeting pair at
  [0] + every exchange, updated to include the current exchange before
  summarize_latest_state. The harness never truncated engine-visible history
  (`--max-update-history` bounds only the LocalModel transport payload via a NEW
  list in `_engine_internal_messages`) — the ~16 cap was engine-side, not
  harness-side.

### 25.3 Other audit results (no change needed)
- summarizer.py:570 (`generate_using_tgwui` dump) — debug-only write of the
  internal context, not a logic consumer.
- summarizer.py:2244 / 3393 — greeting reads `state["history"]["internal"][0][1]`,
  correct (production TGWUI internal[0] = greeting); harness now provides it.
- summarizer.py:1066 `user_input_message_idx = len(history) * 2` — already the
  real-history convention.

## 26. Missing/errored overviews + judge records at the tail of long runs (2026-08-16)

### 26.1 The bug (cozy_mystery__0c82cf38, 40 turns)
The run completed with THREE failed cloud passes: `rolling_overview_032` =
"network error (read timed out)", `rolling_overview_040` = "judge output was
not JSON:" (empty), final overview = "HTTP 500 after 4 retries". Per-turn judge
also dropped turn 30 (network timeout).

Root cause (proven by live probes, not speculation): deepseek-v4-flash on the
/zen/go endpoint ALWAYS emits a large `reasoning` field on big prompts —
21-33K chars regardless of `reasoning_effort="none"` / `enable_thinking:false`
(the "none kills reasoning" behavior seen with OpenCode Go's other endpoint
does NOT hold here). With the old overview budget of max_tokens=6000 the
reasoning block alone ate the whole output budget → `finish=length` with empty
`content`. Two amplifiers hid it:
- `complete()` only read the `reasoning_content` fallback (which this endpoint
  never returns — the field is `reasoning`), so content stayed "".
- The empty-content doubled-budget retry was gated `budget < 4096`, so a 6000
  budget never retried; json_complete's three attempts each re-ran the same
  sized request under a 120s CloudModel socket timeout (a request can take
  70-150s at 6K budget, >120s under run load) → surfaced as network/500 errors.

Verified sizes: overview request grows ~4K tokens per 4 turns (24 turns ≈ 23.5K
tok, 40 turns ≈ 38-40K tok). The successful checkpoints (8/16/24) were under
the budget the reasoning + content could fit in; 32+ crossed it.

### 26.2 Fix (implemented)
- **cloud_client.py**: `complete()` rerolls empty content once with a TIERED
  budget — small (<4096) DOUBLES (truncated JSON is their failure mode), large
  gets only +2048 headroom (doubling invites proportionally longer reasoning
  with no content, costing minutes per attempt). `json_complete` keeps its
  doubling only below 8192. New `max_retry_budget` (32768) cap.
- **long_horizon_soak.py**: overview calls max_tokens 6000→16000 (final
  ~38-40K-token prompt needs the model's ~8.5K reasoning + ~9K content);
  per-turn judge 3000→8000; auditor 800→1600; judge/auditor CloudModels
  timeout 120→300s (a 16K-budget overview takes 70-150s).
- Proven live: the exact failing final-40 request now completes (finish=stop,
  9.2K content tokens). `--final-overview` backfill recovered the run's final
  overview (3.5/5); new `--backfill-rolling-overview RUN_DIR N` recovers any
  specific rolling checkpoint (32 → 3.0, 40 → 3.5).

### 26.3 Residual variance
At 16K budget the overview PROBES sometimes still return empty content even
though budget is not binding — the model randomly enters a reasoning-only state
on huge transcripts at temp 0.8. The reroll policy handles this (bounded, one
reroll, +2048). Note for future runs: rolling overviews can take 2-4 min each
(the output is 2-3x downgraded over the old budget).

## Batch: scene-start rewrite + full-ring repetition guards + retriever read-side repair (2026-08-17)

Follow-on to the fantasy_epic__40c80fa4 3.0/5 post-mortem. Three root causes, all addressed.

### 1. current_scene.start freeze — now rewritten on every new-scene turn
- SceneStart carried `no_update: true` and CurrentScene had only an `always` now-update trigger, so NOTHING rewrote `start` after turn-0 population (byte-identical across all 40 turns, 9 scene transitions fired). The render contradicted itself every turn ("Scene Start — Frost-hold stair" above "Scene Now — Bone Sea").
- BOTH schemas: CurrentScene gained `scene_start_query_prompt_template` + `scene_start_update_prompt_template` defaults + `on_new_scene: [{action: query_branch_for_changes, prompt_template: scene_start_update_prompt_template, skip_query: true}]`. The template COMPLETELY rewrites `what`+`start.*` (+`now.*` reset) from the transition exchange, and stamps `start.when._message_node = {{current_message_node}}`.
- Engine side: `summarize_latest_state`'s scene handler now defensively stamps `start.when._message_node = (len(history)*2)-2 _1_1` even if the LLM answers NO_UPDATES_REQUIRED — fixes the scene-bounded dialogue window + rolling-summaries scene-key (`start.when._message_node` was never written; the window elif was dead).
- Verified: schema parse + render via `_create_update_prompt` (substitution lands), branch-rewrite `_apply_branch_updates` rewrites every field in place (tested on the frozen-fantasy shape), live 3-turn smoke: node stamped 4/6 on budget splits, ON_NEW_SCENE trigger registered + update-only call fires each transition, trigger map confirmed at runtime.

### 2. Repetition loop — full-ring guard + reply-side post-check
- `_instructions_similar` now uses `autojunk=False` (the default deflates re-lexicalized prose to ~0.75 so the 0.85 threshold never fired).
- New `_block_collides` scans the WHOLE 8-ring (label, props, TEXT): byte/LCS vs any ring block, a 2-prop re-anchor cluster vs any single block, AND a persistent-prop signal (a prop in >=3 blocks reappears). Regen names the actual offending props (t28 lesson); hard-reject acceptance keeps the LEAST-colliding of up to `_INSTR_REGEN_MAX=2` regenerations.
- `_ring_blocks` folds the previous instruction in explicitly (resumed runs with an empty in-memory deque regenerate with context).
- `_current_scene_recap` prefers live `now.what` over the lagging top-level `what`.
- Reply-side: harness `SoakRun._generate_reply` post-checks each draft against the steering prompt (transcription >=0.70) and the previous reply (self-anchor >=0.80, 200+ chars), regenerates ONCE with an explicit anti-repeat directive. Harness-side by necessity — production reply text is emitted by TGWUI chat.py outside the engine.
- Verified: unit locks green on all four signals + both reply guards; hermetic suite green.

### 3. Retriever read-side repair (memory stored, context dead)
- Events selection (was 0 sections): `retrieve_context` surfaced events only via entity-graph milestone edges AND returned them as `{"entries": ...}` — a shape the events format template NEVER reads. New `_select_relevant_events` selects by name/alias match + current-scene events + recency fill (max 8), returns the FULL category shape. `Result.events_full` (uncapped) feeds boundary/`has_archived_scenes` consumers so the cap can't drop far-away archives. Verified: real turn-039 events -> 8 selected + full render; live smoke renders `Event --` lines.
- RAG recency: `query_messages` fetches 6x candidates and re-ranks (recency band, summary preferred, sim tiebreak); stale beats no longer dominate (stub-chunker verified).
- Per-entity Recent-state line: `_recent_state_map` computes each entity's freshest story fragment; `_recent_state` attached to copied entry dicts + rendered by a new optional `Recent state --` line in the characters/elements/groups format templates. Live smoke renders it.

## 27. Live cloud-call status reporting (2026-08-24)

Motivation: stealth/ox-alpha via OpenRouter turns out to be SLOW (guide turn observed at
287.6s; earlier "9-minute hang" was one long cloud call, not a server stall). There was no
way to distinguish a slow cloud model from a hang. OpenRouter's status endpoints only cover
completed generations, so in-flight progress requires streaming.

Implementation:
- cloud_client.CloudModel gained `status_dir` + `role`. When status_dir is set (soak only;
  standalone probes unchanged), `_retry_post` dispatches to a new `_post_stream`: SSE
  streaming (`stream: true`, `stream_options.include_usage`) with the SAME normalized return
  shape as blocking `_post`, so empty-draw cooldowns, budget tiering, accounting, and
  fallback_reasoning all work unchanged.
- Live status file per in-flight call: `{run_dir}/live/cloud_{role}_{pid}.json`
  (role, model, phase generating/cooldown/backoff, elapsed_s, content_chars,
  reasoning_chars, head/tail snippet), throttled to 1 write/s, atomically replaced,
  REMOVED on completion AND on exception (try/finally around both _retry_post sites).
- stdout heartbeat `[cloud {role}] ...` every 60s while generating / during backoffs /
  cooldowns — lands in the tee'd full log without flooding it.
- Harness wires status_dir={run_dir}/live + roles guide/judge/auditor (main) and judge
  (--final-overview backfill path); stale cloud_*.json cleaned at startup.
- Dashboard: `_cloud_activity_html` renders an in-flight table inside the existing 10s
  auto-refreshed live-stats block (per-call role/phase/elapsed/chars/rate/snippet;
  >45s without an update = "⚠ stale?"; "no call in flight" otherwise).

Verified live: local-server SSE probe — status file lifecycle correct, counters grow across
1s samples (0→171→335→509 ch), heartbeat fires, content intact, char-estimate usage fallback
when a provider omits stream usage; hermetic suite green; dashboard renders synthetic
in-flight files incl. stale flag.

Note: ox-alpha 429-rate-limits pings while a soak is running (shared per-minute budget) —
probes compete with the soak's own calls.

## 28. Traversal heal for array-wrapped class instances + full-scale judge calibration (2026-08-24)

Bug (cyberpunk run, repeating): "TypeError: list indices must be integers or slices, not str"
at traversal._initialize_field after branch queries. Root cause: the model writes schema-class
objects as arrays in update responses (relationships: {"Juno": [{...}]}); entry-time
coerce_container_types can't rebuild them (items lack a name field) and traversal then treats
the list as the class dict. Disk stayed clean because the crash aborted the subject before save;
the model re-emitted the shape every turn. Fix: DSTraversalMixin._normalize_schema_instance
([dict]->unwrap, multi-dict->merge) wired at both descent sites (heals in-memory data BEFORE
the whitelist check so saves propagate), plus a fail-open non-dict guard in _initialize_field.

Judge calibration: _SCORE_ANCHORS ladder injected into JUDGE_SYSTEM_TMPL and
FINAL_OVERVIEW_SYSTEM_TMPL (incl. rolling overviews): 5.0 perfect/masterful, 4.5 extraordinary
(experienced professional writer), 4.0 excellent/production-ready ("good enough" ceiling),
3.5 good, 3.0 functional, 2.5 weak, 2.0 poor, <=1.5 failing; competent-but-flawed lands 2.5-3.5,
4.0 must be earned, do not cluster at 3-4; MANDATORY FLOORS still cap. Verified: formats cleanly
against all 12 spreadsheets; hermetic green.
