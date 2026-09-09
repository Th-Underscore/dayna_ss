# DAYNA DSS — Project State & Session Handoff

*Hub doc for cross-session continuity. Last updated: 2026-08-25.*

Durable engine facts live in agent memory (magic-context); run-level forensics live in
`soak_optimization_plan.md` and `long_horizon_soak_plan.md`; this file holds **current position**,
the **decisions ledger**, the **sequenced queue**, and anything that exists only in conversation.

---

## Current position (2026-08-25, evening)

- Aggregation-unit fixes (§33) are now **live-validated** (§34): budget splits, content-bearing gate
  (LLM YES on a real span digest), list-chapter rendering all confirmed working — and the smoke caught a
  real bug the hermetic tests missed: boundary archives were never persisted to disk (fixed + regression-
  tested). Same session: dead NEXT CHAPTER:/NEXT ARC: prefixes fixed via persistent_ui_state staging;
  cloud_client endpoint flipped to OpenRouter; **cadence profiles shipped** (`--cadence-profile`).
- Canon Synopsis P0 **design doc written pre-code**: docs/plans/canon_synopsis_p0.md (home =
  general_info.synopsis; regenerate-from-inputs at each archive; render-time staleness demotion). Next
  step is implementation against that design.
- Engine stabilized after the T4 package refactor (five casualty classes swept) and the aggregation-unit
  fixes (soak_optimization_plan.md §33): Chapters/Arcs archival chain is now functional end-to-end — real span math, content-bearing gates (span digests), list-branch add_new, duplicate-trigger removal.
- Unit semantics encoded in gate + transition templates: SCENE = one continuous stretch
  (action/tone/cast-presence/setting shifts start a new one); CHAPTER = TV episode / few GN chapters; ARC = TV season / few GN volumes.
- Latest baseline: `cyberpunk_thriller__4e29a512` — 40 turns, **3.5/5 under the new calibration ladder**, flaws fully root-caused (memory #221). Best prior run: `fantasy_epic__551bcf41` 4.0/5.
- Cloud stack migrated to **OpenRouter** (`stealth/ox-alpha` trio; cloud_client default flipped 2026-08-25);
  local server currently Qwen3.6-35B-A3B-abliterated-AWQ (27B Queen config in `CMD_FLAGS_2.txt`).
- Default test configuration: Type 2 Mode B — `--schema-type 2 --message-mode rolling
  --retrieval-placement system --rolling-summaries 20`. Standing rule: rendered context < 40k tokens.

## Decisions ledger

1. **Unit cadence numbers are calibration, not truth** (2026-08-25). Unit lengths are goal-relative
   (D&D-campaign framing): scene/chapter/arc size depends on the campaign's intended scope. The
   schema's current numbers (chapter 4–8 scenes / hard 10; arc 3–6 chapters / hard 12) are one
   profile chosen low because test runs only produce ~7–10 scenes total — at true scale a 40-turn
   run is barely one episode.
2. **Machinery ≠ judgment.** Aggregation machinery (span math, archival chaining, caps, rendering)
   is scale-free and verifiable hermetically at any cadence. Boundary *judgment* ("is this a
   complete episode?") is only meaningful at realistic scale. Numeric bounds stay in prompts as
   anti-drift guardrails but become config-declared.
3. **Cadence profiles** (BUILT 2026-08-25): named profiles selectable via config/harness flags —
   `compressed` (schema numbers: chapter 4–8 scenes / hard 10; arc 2–5 chapters / hard 8) vs `campaign`
   (chapter ≈ 15–40 scenes; arc ≈ 8–24 chapters; true-scale deployments). Same engine/templates;
   pure plumbing via UNIT_CADENCE_PROFILES + `summarizer.config["cadence_profile"]`.
4. **Test ladder** (the reconciliation of small spreadsheets with true-scale targets):
   - Tier 1 — hermetic stress: script 300–500 turn-cycles, assert archives chain, caps hold,
     render bounded, no O(N²) resurgence. Free, no GPU.
   - Tier 2 — single 100-turn live run on the compressed profile. Requires **Canon Synopsis P0
     first** (render pressure valve; elements hit 173KB by t40).
   - Tier 3 — chained-session campaigns: resolved-state export/import makes each ~40-turn soak one
     "session" (D&D model). Chain 4–6 sessions to reach 160–240 effective turns; notes planted in
     session 1 come due in session 3; arcs fire at campaign cadence. Production campaigns ARE
     multi-session, so this tests the real usage pattern instead of an artificial marathon.
5. **Spreadsheet implication**: don't bloat single sheets to 100 notes — distribute notes across
   chained sessions (cheaper per run, tests cross-session retention).
6. **P0 order**: Canon Synopsis BEFORE Epistemics — the pressure valve precedes features that add
   rendered content, and the synopsis settles the render shape (closed-arc entities demoted to
   roster lines) that secrets rendering composes with.
7. **`[[...]]` inline user-steering directives**: worthwhile guide-side steering feature
   (structured version of parked P3 beat-steering; addresses guide thread-abandonment like the
   drone failure). Not required for 100-turn validity; schedule after canon synopsis.
8. **Engine-native spreadsheet/planning system**: PARKED until good 100-turn results — a persistent
   planner on top of a repetition-prone director just hides loops better.
9. **Judge calibration ladder shipped** (long_horizon_soak_plan.md §28): never compare judge scores
   across cloud-model generations; each judge model is its own scale.

## Sequenced queue

1. Canon Synopsis P0 implementation (design done: canon_synopsis_p0.md — general_info.synopsis home,
   regenerate-from-inputs at archive, render-time staleness demotion; settle the four open questions, then code)
2. Export/import session chaining (Tier-3 enabler; was P1 on the comparison.html roadmap)
3. Campaign-scale spreadsheet generation for the chained pilot
4. Post-mortem follow-ups from cyberpunk_thriller__4e29a512:
   uncertainty-preservation directives in character update templates (Pike/Kell class);
   discovery similarity guard (P2); continuity contradiction probe (P1);
   rolling-overview findings → instruction-generation feedback loop
5. Then: Epistemics P0 (adopted design in memory #212)

## Run inventory (validity eras)

| Run | Status |
|---|---|
| `cyberpunk_thriller__4e29a512` | valid; post-ladder baseline, 3.5/5, analyzed |
| `fantasy_epic__551bcf41` | valid; 4.0/5 best-yet (old ladder) |
| cozy_mystery seeds ≤ 104 | historical only — pre-greedy-fix / pre-calibration eras |
| runs suffixed `_TRUNCBUG`, `_baseline` | preserved debris of fixed bugs (drain race, greedy decode) |
| dots-model-guided runs | dead class: free-tier endpoints reject long prompts with HTTP 400 |

## Ops notes

- `~/.local/bin/llmping` — quick liveness probe for local/OpenRouter endpoints.
- Frozen soak + idle GPUs → probe/kick; frozen + busy GPUs → genuinely decoding, wait.
  Avoid `kill -9` mid-generation (zombie sequences stall the TurboMind scheduler).
- From the user's own terminal `| tee -a runs/logs/*.log` is safe and gives full stdout; the
  self-logs are filtered/concise (memory #215).
- Cloud live status streams to `{run_dir}/live/cloud_{role}_{pid}.json`, surfaced in the dashboard's
  Cloud activity block.

## Where things live

- `docs/comparison.html` — Smart-Memory feature matrix + P0–P3 adoption roadmap
- `docs/performance_estimator.py` — prefill/decode cost model (referenced by comparison.html)
- `docs/architecture.html` — engine architecture
- `docs/plans/*.md` — per-topic plans (this directory)
- Agent memory — durable engine facts: endpoint configs, parallelism/prefix-cache rules, guard
  architectures, importance system, epistemics design (#212), aggregation fixes (#221/#222)

---

## Appendix: fix-batch label glossary

These labels were coined per analysis session; several reuse letters across eras, so
they are grouped by when they were coined. Full detail lives in soak_optimization_plan.md /
long_horizon_soak_plan.md sections noted in parentheses.

**Soak build era (2026-08-09)**
- M1–M5: original harness build milestones (fixtures → cloud client → orchestrator → report → run_tests wiring).

**48ec5b7f mechanism analysis (2026-08-11)**
- M1 instr_prompt opens with guide's verbatim first-person turn · M2 weak local model plants its own repeated imagery · M3 third-person hard rule buried vs "adhere loosely" · M4 subjects byte-frozen across turns (first-scene early-return skip) · M5 latent seed-keyed instruction-cache reuse.

**Perf/quality split, seed-77 prep (2026-08-11)**
- P-P1 add_new dedupe + caps (8→4/call, 120 total) · P-P2 negative short-circuit (NO/UNCHANGED/empty ⇒ skip branch) · P-P3 _execute_action returns (stop, gate_failed, skip_children).
- Q-P1 population re-run wipe (population-once marker) · Q-P2 director imagery feedback loop · Q-P3 notes stored but never surfaced · Q-P4 beats referencing never-planted entities · Q-P5 verbatim quoting persists.

**Audit-fix batch (2026-08-12)**
- #1 auditor capability awareness ("unassessable" status + DSS-capability context) · #2 max_scene_part_messages budget split (default 12) · A2 ElementMap trigger reorder [add_new, gate_check] · A3 population-runs-once marker in general_info.json. (A1 superseded by #2; A5/A6 same day below.)

**Importance batch (2026-08-13)**
- A1 entry-level importance field on entries (+ importance_detail_threshold) · A2 events map-level trigger collapse · A3 importance-weighted rendering (full/dormant/roster tiers) · A4 data-driven list_template lookup (fixed {item_name}_list mismatch) · A5 latest-exchange scope + internal-history bound · A6 identity-binding blocks (name1/name2) across 5 character templates.
- B1 recall horizon clamp (--turns − 2) · B2 retention "reached" flag (due-and-reached denominator) · B3 spreadsheet beat alignment + NAME EXACTLY propagation.

**Five-fix batch (2026-08-14)**
- F1 _clean_generated_instructions meta/deferral strip · F2 gen requirements (#10/#12/#13) + reply hard rules (no hand-over endings/self-narration) · F3 per-paragraph dss_directive word caps · F4 judge calibration (planted-note filter, memory grounded in stored fields, judge fed extracted instruction block) · F5 NAME EXACTLY in initial-population templates.
- F7–F9 (second pass, 2026-08-15): merge dup notes c14/c26 · in-window planted-note grading · judge retry on non-JSON · report attribution guide_failure vs dss_retention_loss.

**45adee46 post-mortem (2026-08-15)**
- H1–H4 analysis hypotheses: H1 instruction-block repetition · H2 reply transcription mechanism · H3 current_scene staleness/lag · H4 note engagement + judge artifacts.
- F1 drop frozen `what` headline → redirected by user into CurrentScene.now query_branch template + remove SceneState query action · F2 inject live location into prompts · F4 reply-prompt reorder (rejected by user) · F5 reply-repetition guard (reworked engine-native).

**Type 2 slowdown levers (2026-08-14)**
- L1 stricter transition prompt + cooldown floor · L2 bound the events pass (skip_query on events trigger, current-scene-only Importance updates) · L3 compact events render.

**Instruction-loop batch 14714d8e (2026-08-15)**
- C1 log state["context"] to turn dirs (context.txt) · P1a CURRENT SCENE recap at top of instruction-generation prompt · P1b specific_time must advance every exchange · P2a no imagery reuse unless deliberate callback · P2b prop/verb overlap regen guard (LCS ≥ 120) · P3 guide beat-steering (PARKED) · P4 cloud retry with parse-error feedback + report attribution fix.
- TR1–TR5 prefix-order template transforms (per-entry name to tail, entry_name tail, branch_list tail, population descriptor tail, importance value block tail).

**mark_field restatement (2026-08-15)**
- Options a/b/c user's marker ideas; Option d adopted: size-gated _context_restatement (auto/always/never, 10k-char threshold) replacing whole-map tails.

**select_entries_to_update (2026-08-14)**
- (i)/(ii) stage designs → (ii) one combined call chosen; (a)/(b) processing choice → (a) whitelist-gated traversal chosen; C3 selection fail-open strictness (raw response must start `[` AND parse flat).

**Anti-repetition batch b884af4a (2026-08-16)**
- P1 recently-used props ring in instruction generation · P2 semantic prop-overlap regen naming offending props · R1 negative exemplars (DSS's own last replies quoted as anti-patterns) · R2/R3 proposed, not shipped (superseded by later guards) · R4 repetition_penalty 1.0 → 1.15.
- S1–S3 storage hygiene (DEFERRED): S1 dedupe strengthening · S2 importance decay / used-recently flag · S3 render-order rotation + stale threshold fix (soak_optimization_plan.md §30).

**Rolling message summaries (2026-08-16)**
- a/b mentioned-subjects storage fork → (b) recommended (ride inside summary text); sticky window lock (last-20-or-last-5-scenes, held until next scene turn).

**Refactor casualty classes (2026-08-24)**
- T1–T3 earlier user refactor rounds (pre-session); T4 = the big split-and-shrink pass — casualty classes fixed: lost import block, module-globals stranded, missing self, decorator contradictions, lost return statements.

**comparison.html roadmap (standing)**
- P0 Canon Synopsis / P0 Epistemics / P1 continuity probe + export-import / P2 similarity guard + ledger pack + dyad canonicalization / P3 polish tier. Order decision: Synopsis BEFORE Epistemics.

**Test ladder (2026-08-25)**
- Tier 1 hermetic stress (300–500 scripted cycles) · Tier 2 single 100-turn live run (compressed cadence) · Tier 3 chained-session campaigns via export/import. (Collides with the older seed-82 "Tier 1/2/3" fix tiers: engine bugs / spreadsheet plants / reply-quality design.)
