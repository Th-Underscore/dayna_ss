# DSS Architecture Notes

Cross-cutting architectural decisions and proposals. Per-topic plans live in sibling docs
(`performance_optimization.md`, `quality_enhancements.md`, `retrieval_plan.md`, `async_dss.md`,
`headless_engine_ui_plan.md`); this file records the structural layer that spans them.

---

## 1. Subject processing order = "sourcehood"

**Current state (traced 2026-11-26):** the per-turn subject processing order is the dict
iteration order of `all_subjects_data` (`summarizer/core.py:_process_all_subjects`), which derives
from the schema's `subjects` mapping key order at first creation and persists turn-to-turn through
directory copying. It is *implicit* — invisible, easy to disturb, and documented nowhere.
`retrieval_order` (schema root) is unrelated: it orders the retrieval relevance dispatcher only.

**Why it matters:** cross-subject data dependencies (e.g. `Character.knows` referencing
`KnowledgeItem` slugs) are only satisfiable within a turn if the referenced subject is processed
first. Today that ordering is accidental.

**Proposal:**

1. **Explicit `source_order`** — a schema-root key listing subjects in dependency (sourcehood)
   order, consumed by `_process_all_subjects` (fallback: existing dict order when absent).
   Expected shape: `current_scene → events (scenes) → knowledge → characters/groups/elements →
   chapters/arcs → general_info`.
2. **Turn-scoped new-entry registry** — `_new_entries: dict[branch_path, set[str]]` on the engine,
   cleared at turn start, populated by `add_new` and `check_archive` at commit time. Exposed to
   templates as a **data object** (`{{ new_entries }}`, same precedent as `enum_constraints`) so
   each template decides whether to render it. Consumers: `knows`, `learned`, `relationships`
   templates ("created this turn: X, Y").

Both are schema-driven, zero new LLM calls. **`deepdiff` stays out of scope here** — it remains
the candidate replacement for the long-term `_history` audit (quality doc 3.1 follow-up); the
registry is turn-scoped and tracked at creation sites, which is exact and free.

**Synergy:** with an explicit order + registry, the deferred `_references` post-apply membership
check (quality doc §10) becomes non-flaky — referential integrity for `knows`/`learned`/
`relationships` is the combined payoff.

---

## 2. Schema-to-workflow compilation (todo.md Far TODO #22)

**Assessment (2026-11-26): the direction is right; the full compile is not the next step.**

- The value of compiling is legibility, ordering guarantees, and testability — **not speed**.
  LLM calls are ~95% of turn time; traversal overhead is noise.
- The traversal encodes hard-won invariants (two-copy expansion parity, underscore merge-preserve,
  entry whitelists, fail-open paths) that a compile must preserve. It just stabilized; a big-bang
  refactor would force re-validating all of it.
- The ordering pain that motivates it (`on_chapter_archived`) is already solved adequately by the
  number-snapshot mechanism (validated s579).

**Staged path:**

| Phase | What | Cost | Value |
|---|---|---|---|
| 0 | **Plan-emission telemetry:** instrument the traversal to emit the resolved per-turn step plan (subject → branch → action → trigger → gate result → selected entries) to `dss_plan.jsonl`, beside `dss_calls.jsonl` | Small | Diffable record of what the dynamic system decided; test oracle for a future compile; analysis telemetry |
| 1 | **Turn-scoped event queue (proper triggers lite):** archival path enqueues `scene_archived`/`chapter_archived` events; trigger evaluation consults the queue instead of snapshot-diffing `_{unit}_number` | ~30–60 lines | Causal ordering by construction; removes snapshot-timing fragility; triggers become "proper" while staying schema-declared |
| 2 | Full compile: execute from the emitted plan | Large | Explicit decision point after the performance queue drains |

Phases 0–1 can ride the same soak window as §1.

---

## 3. Scene-stable prefix architecture (Prefix A / B / C)

*User proposal, 2026-11-26. The enabling foundation for async DSS (`async_dss.md`).*

### 3.1 The problem (measured structure)

`_context_order` (format_templates.json) renders the system context as:
`importance_scale → capability_ladders → general_info → current_scene → arcs → character_list →
groups → elements → events → chapters → characters → knowledge → lines → current_scene2`.

`current_scene` sits at **position 4 of 14** and changes every turn (`now.when` advances, cast
shifts). Under vLLM APC the cache survives only up to the first changed token — so mid-scene,
everything after `general_info` (arcs, characters, groups, elements, events, chapters, knowledge,
lines, current_scene2, rolling pairs) **re-prefills every turn**. The cached head is the smallest,
slowest-changing part; the volatile part sits in the middle and poisons the tail.

Prefill degrades 5x with new-token count (performance doc §1: 2159 → 434 tok/s), so this per-turn
re-prefill is the dominant per-turn cost — and it is paid again by the first DSS call and the
writer call of every turn, even non-scene turns (>60s today).

### 3.2 Design

Three prefix zones, ordered by change frequency:

| Zone | Content | Stability |
|---|---|---|
| **A** (~20k) | The full `retrieved_context` inherited from the previous scene's freeze — reflecting the DSS run that **completed during the previous scene** (scene 11 uses scene 9's summarization data while scene 10 summarizes asynchronously): harness context, importance_scale, capability ladders, general_info, subject renders, scene data | Frozen for the whole scene; rebuild trigger = open decision (§3.4: scene boundary recommended vs DSS-complete) |
| **B** (~3k) | History until scene start: sticky rolling summaries (existing sticky-lock machinery) + raw pairs | Frozen **by construction** once the scene-start point is fixed — no new mechanism needed (resolves the "freeze B(?)" question) |
| **C** (~0.5–1k/turn) | History since scene start (raw pairs) + **per-turn retrieval deltas**: only what differs from the previous checkpoint (A at the scene's first turn, the previous C delta after), per-turn-labeled ("CONTEXT FOR USER MSG N"), never spliced into A | Append-only within the scene; dropped at the next scene turn |

Key correctness property: **DSS update targets are always live** — `branch_update` templates
render `{{ value | tojsonc }}` per-call from current state. Freezing A only affects the *ambient*
context the writer/DSS reasons over, not the data being edited. This is the same insight
`async_dss.md` rests on ("last confirmed" vs live-inferred state).

### 3.3 Synthesis with the two-tier prompt structure (performance doc §5)

A carries the full retrieved render as of its freeze — whatever tiering is active applies *inside*
A's render (roster lines for stale entities, detail blocks for relevant ones). C then carries
**detail-tier deltas**: only entries that changed state or newly entered the relevance set,
score-graded and capped (performance doc §7), plus the `current_scene.now` update. Two-tier
provides the content policy for both zones; scene-stability provides the skeleton (which turn
renders where). The existing trailing `current_scene2` ("Scene Focus") section is the seed of this
pattern — a fast-changing section already placed last so its changes only invalidate its own tail.

### 3.4 Async DSS contract (delay budget)

- Background DSS calls run **A + B + prompt**; foreground writer turns run **A + B + C + prompt**.
  Both share the A+B base, so interleaving no longer thrashes the cache — this is what makes
  `async_dss.md` viable.
- **Delay budget:** scene turns are synchronous sync points (scene → chapter → arc cascade needs
  settled state). The background DSS must complete before the next scene turn; within the scene it
  runs behind the writer with a bounded lag (coalesce per `async_dss.md` Strategy B on expiry).
  Non-scene turns stop paying DSS latency entirely; scene turns pay it once.

**Turn lifecycle (canonical, clarified 2026-11-26):**

1. **Scene N ends** (asst msg M): async DSS starts processing scene N; A and B freeze; scene N+1
   opens at msg M+1.
2. **Each user turn in scene N+1:** retrieval runs (cheap, no LLM), diffed against the previous
   checkpoint; the delta becomes a labeled C block appended to the stream.
3. **Scene N's async DSS completes mid-scene:** state files update; the next turn's diff naturally
   surfaces the new state as a C delta — the one-scene A lag self-heals through C.
4. **Scene N+1 ends:** the delay budget requires scene N+1's DSS turn complete (the scene →
   chapter → arc cascade needs settled state); A rebuilds from the last completed DSS at the next
   boundary (recommended trigger — see "A rebuild trigger" below).

**Completion-order constraint:** A's freshness = the last *completed* async DSS. The delay budget
is what bounds the lag to ~one scene: if a scene's DSS turn has not completed by the next scene
boundary, the boundary either blocks (within budget) or coalesces (`async_dss.md` Strategy B).

**A rebuild trigger — open decision (noted 2026-11-26; recommendation: scene boundary).** A at
scene N reflects the DSS run that *completed during scene N-1* (scene 11 uses scene 9's
summarization data while scene 10 summarizes asynchronously). Two possible refresh triggers:

| | rebuild when async DSS completes (mid-scene) | rebuild only at scene boundary |
|---|---|---|
| cache cost | 1 tail re-prefill per scene, landing mid-scene | 1 tail re-prefill per scene, at the boundary (no-op when nothing completed since) |
| A freshness | ~0.5–1 scene behind | ~1.5 scenes behind on *derived views* — but **exactly equal to disk state** at the boundary (state files also update only on DSS completion), and C deltas self-heal state changes |
| C checkpoint | cumulative diff base jumps mid-scene, under an already-appended C stream | fixed for the whole scene — simpler diff bookkeeping |
| concurrency | A swap must be deferred to a turn boundary; two trigger paths | one trigger path at an existing sync point |
| edge cases | two DSS completions in one scene → two cache breaks | always ≤1 break per scene |

Recommendation: **scene-boundary rebuild.** The cache cost is identical (one rebuild per scene
either way — a mid-scene rebuild makes the boundary rebuild a no-op, and vice versa), the state A
carries is exactly as fresh as the state files can be, the writer's live moment comes from the
pairs + C deltas rather than A, and the residual gap (derived views such as scene recaps lagging
~1.5 scenes) is measurable in the A/B. If that gap shows up as judge-visible artifacts, flip the
trigger to DSS-complete with a deferred swap at the next turn boundary — same C design, no
structural change.

### 3.5 Open decisions

**0. C policy and placement (clarified 2026-11-26 — user design adopted as baseline).** Two
independent axes: content policy (re-render the current relevance set each turn vs accumulate
per-change deltas) × placement (deltas interleaved between exchanges vs one trailing region after
the pairs). The cache behavior is NOT symmetric, because both the pair history and the C content
grow within a scene:

| | re-render (replace) | accumulate (deltas) |
|---|---|---|
| interleaved | broken — dropping a superseded block mid-stream invalidates everything after it (APC hashes are chain-linked to preceding tokens) | **cache-optimal** — strict prefix extension; miss ≈ reply + new delta + msg + instr (~1–1.5k); old deltas never shift position |
| trailing | miss ≈ reply + msg + block + instr (~1.5–2.5k, bounded); single authoritative snapshot | miss = reply + msg + **the entire accumulated C region** + instr — the region sits after the growing pairs, so it shifts (and re-prefills) every turn; grows with scene length |

Interleaved + accumulate is the unique zero-re-prefill arrangement: when two sections both grow,
only strict chronological interleaving (append order) keeps every earlier byte in place. The
stale-in-stream surface is bounded under diff-based deltas: few blocks, per-turn-labeled,
chronologically ordered, freshest nearest the prompt — and the DSS's authoritative data is the
live `{{ value }}` suffix, not the ambient stream.

**Baseline: interleaved + accumulate-deltas.** Trailing + re-render is the conservative fallback
(bounded miss, single snapshot, no history-render surgery). Trailing + accumulate is dominated
(worst cache of the three, no quality advantage) and dropped.

1. ~~C semantics: replace vs accumulate~~ — **resolved by decision 0**: accumulate-deltas,
   interleaved. The delta computation (diff vs the previous checkpoint) is the load-bearing new
   logic and needs a hermetic test.
2. **What triggers a C block:** relevance-delta vs state-diff (entry changed since scene start).
   State-diff is precise; relevance-delta matches what the renderer already computes. Likely both:
   changed entries ∪ newly-relevant entries, score-graded, capped.
3. **Writer freshness for `now`:** C carries the updated `now` block each turn; the raw pairs in B
   carry the live moment. Matches the async doc's option 1 ("last confirmed" + history as
   current-moment source).
4. **Template mechanics:** a `stability: scene | turn` flag per `_context_order` entry (data, not
   code) drives which sections render into A vs C; A's rendered string is cached per scene key
   (same key as the sticky lock). `format_templates.json` restructuring follows from this flag —
   no second template set needed. Path-marker contract is per-line and survives the split.
5. **Duplicate visibility:** when an entity appears in A (roster line) and C (detail block), the
   LLM sees it twice. Mitigation: C blocks carry a "latest" header convention; roster line stays
   one line. Watch for confusion in the A/B smoke.

### 3.6 Validation plan

A/B smoke, same seed, current vs scene-stable:
- Timing: TTFT distribution + per-call uncached tokens (deep_timing.py). Prediction: uncached
  suffix per call drops from ~5–15k to ~1–3k; DSS-call TTFT median from ~2.5s to <1s.
- Quality guards: judge scores, staleness artifacts (writer acting on outdated positions —
  STATE LAG class), duplicate-entity confusion, placeholder leaks, and stale-value copying by DSS
  calls from accumulated C deltas (the #828 pattern-matching class — the live `{{ value }}` suffix
  must remain the authoritative source).

### 3.7 Risks

- **Staleness bugs:** a mid-scene relevance shift must surface in C or the writer misses the new
  entity. The delta computation is the load-bearing new logic — test it hermetically.
- **Expansion parity:** A is a cached rendered *string*, so the two-copy expansion invariant is
  untouched — but the C renderer must expand with the same machinery.
- **KV pressure from variant tails:** writer (A+B+C_w+instr) and DSS streams (A+B+prompt_i) share
  A+B; tails are small. Fine at n_ctx 49152.

---

## 4. Smaller notes

- **Apply-path unknown-key validation gap:** `validate_data` rejects unexpected fields
  ("Unexpected field"), but only whole-entry paths call it (`add_new`, `check_archive` refine).
  The path-based `apply_updates` validation never checks whether a path segment is a declared
  field — how `scenes: None` landed on chapter entries in s595. A schema-driven unknown-key prune
  at commit would close a whole error class (with the `str item assignment` family).
- **R0 retrieval eval (retrieval_plan.md)** remains unbuilt and gates all threshold tuning
  (two-tier Tier-2 sizing, decay weights, re_traverse). Replay-only, no LLM, minutes per run.
- **Knowledge stance rendering (quality 4.2 Consumption):** per-stance view
  (`knows: A, B | suspects: C | unaware: D`) computed in code from edges + scene cast — the
  uncollected epistemics payoff; knowledge currently renders flat. Zero LLM cost.
- **Scene `end._message_node` stamp gap:** the post-apply stamp in `generate()` does not fire for
  `check_archive`-created scenes (`_archive_boundary_entry` path). Minor consistency gap.
- **Population wrapper-vs-individual mismatch** (todo.md AI notes): still open, Options A/B/C
  documented there.
- **Housekeeping:** untracked one-off `scripts/*.py`, `benchmarks/payload.json`, and the
  `shounen_twins.json` dual-POV spreadsheet need a commit/gitignore decision.
