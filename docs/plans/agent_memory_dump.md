# Magic-context memory dump — dayna_ss (textgen)

*Generated 2026-08-25 from context.db (identity `git:0b86ac38…`). 113 active + 19 archived. Revise here, then tell the agent to update/archive by id.*


## ARCHITECTURE  [ARCHIVED]

### #32 ~~ARCHIVED~~

*updated 2026-08-10 16:08:42*

dayna_ss empty-subjects regression (FIXED 2026-08-10, agents/summarizer.py): two compounding bugs made ALL per-turn FORMATTED CONTEXT subjects render <EMPTY> (only turns 0/11/12 of the cozy run wrote subject files). (1) Commit 2af4d9cf changed prepare_context's next-scene line from conditional `if next_scene: is_new_scene_turn=True` to unconditional `is_new_scene_turn = persistent_ui_state.get("next_scene", False)` — clobbering the fresh-chat is_new_scene=True that get_retrieval_context sets (line ~1555), so _populate_from_first_scene never ran on turn 0. (2) The `if not has_archived_scenes and not is_new_scene_turn: return` early-return (added by d55ab7a3 "enhance first-scene population logic", UNTESTED) skipped the DataSummarizer on regular first-scene turns, and since every turn hashes a NEW history dir, populated subjects never propagated to the next turn's dir. Fix: restore conditional-set for is_new_scene_turn (line 679), and on the first-scene early-return path copy all subject *.json from last_history_path to new_history_path (schema/templates were already copied by 30cf00d7). Verified via fresh 3-turn L2 smoke (populated chars/elements/scene in every turn dir) + hermetic suite green.

### #37 ~~ARCHIVED~~

*updated 2026-08-23 22:45:09*

soak_dashboard.py run-selection fix + live stats: the old build used a visibility-toggle (run_menu.change only flipped run_body.visible — switching runs showed the first run's content forever). Fixed with @gr.render(inputs=[run_menu]) so render_run(run_id) re-invokes _run_display on dropdown change (gradio 4.37 supports gr.render; it re-renders per browser session, so config API won't list dynamic components — verify via /run/predict fn_index of the 'apply' dep returning a render_config layout tree whose node count differs per run). A hidden Button with elem_id='soak-live-refresh' + a demo.load(None, js=setInterval click) refreshes live stats every 10s. _live_stats_html(run_id, runs_dir) builds HTML stat cards: local server status (probe {base}/models + internal/model/info with the API key; ctx from --n_ctx in CMD_FLAGS_m.txt), cloud models from manifest args (guide/judge/auditor) with ctx+USD cost per MTok from the hardcoded MODEL_INFO dict (deepseek-v4-flash ctx=1048576 no published cost; glm-5.2 1048576 $1.2/$4.2; gpt-5.6-luna 1050000 $1.1/$6.6 etc), run progress (turns done/total from manifest last_turn), cloud usage (calls, total_tokens, ~est cost), and GPU load via nvidia-smi. Every card shows the HH:MM:SS it was measured at. helpers: _local_api_key(), _local_ctx_from_flags(), _local_server_status(), _gpu_stats(), _est_cost(), _fmt_tokens().

### #39 ~~ARCHIVED~~

*updated 2026-08-23 22:45:09*

soak_dashboard.py live-stats panel + timestamps (2026-08-10): the top-of-page live panel refreshes every 10s via a hidden gr.Button(elem_id='soak-live-refresh') clicked by a demo.load(None, js=setInterval) — ALL live updates use show_progress="hidden" so no loading spinner overlay. Live cards are now CONSISTENT between local and cloud: model name is the card VALUE, and the SUB line reads 'max ctx N · last turn ~X tok sent · <loader/roles> · HH:MM:SS' (local ctx from CMD_FLAGS_m.txt --n_ctx; cloud per-role ctx from MODEL_INFO dict). Run selection uses @gr.render(inputs=[run_menu]) so switching runs rebuilds. Timestamps (HH:MM) shown: retention tables + per-note rows (planted/last_seen as 'turn@time'), judge curves, recalls, per-turn accordion headers ('at'), final overview ('written at'), live cards, and all stat cards. 'Local ctx (prompt tok)' and 'Cloud ctx (prompt tok)' stat cards show per-turn prompt-token sums.

### #41 ~~ARCHIVED~~

*updated 2026-08-23 22:45:09*

dayna_ss current_scene.now frozen root cause (2026-08-10): SceneState (the `now` field's schema) had the `always: [query_branch_for_changes]` trigger but LACKED the required `branch_query_prompt_template` + `branch_update_prompt_template` defaults. data_summarizer.py:299 `if bq_template and bu_template:` silently no-ops when either template is missing, so the branch query never fired every turn — `now` was written once by first-scene population and never updated (GeneralInfo DID update because it defines both templates). Fix: added both templates to SceneState in user_data/example/subjects_schema.json (modeled on GeneralInfo's, scene-specific). `start` stays frozen via SceneStart's `no_update: true` (honored at data_summarizer.py:1240 field_def.no_update). Note: SceneState's always-trigger fires on EVERY turn — that's the intended per-turn `now` refresh.

### #68 ~~ARCHIVED~~

*updated 2026-08-12 03:17:12*

TGWUI LMDeploy fork prefix caching: the fork's build is based on lmdeploy PR #4465 ("Turbomind linear gdn prefix caching") which was REJECTED and closed unmerged (lvhan028 declined; xliangwu reported a parallel-run crash on it). Its 4096-token GatedDeltaNet clamp (SequenceManager.cc:761-765) and the concurrent BlockTrie/SequenceManager segfault are fork-local. Upstream lmdeploy main (post-v0.14.0a1 #4557, v0.15.0 #4717, Jul 2026) has lzhangzz's scheduler/object-cache rewrite with REAL Gated DeltaNet recurrent-state prefix caching (engine/scheduler.cc, prefix_trie.h, EngineConfig cache_prompt/cache_generation/cache_checkpoint_interval), race-free by single-engine-thread design. So: patching the fork's race in-place = dead end; upgrading lmdeploy is viable but requires reworking the custom 781-line modules/lmdeploy.py (asyncio wrapper, auto_max_new_tokens path) against the new API. Prefix caching only cuts ENCODE cost, not decode. Concurrency+cache should be combined post-upgrade (upstream supports both).

### #95 ~~ARCHIVED~~

*updated 2026-08-23 22:45:06*

dayna_ss soak: the recall metric can report 0% spuriously. The per-turn recall probe in long_horizon_soak.py only fires when `int(recall.due) == turn`, and report.py's _recalls counts EVERY note with a recall block (defaulting echoed=False) — so if a spreadsheet schedules recall.due beyond the run's turn count (e.g. due 62-78 in a 20-turn run), no probe ever runs yet recall_success reports 0/13=0%. Always check recall.due values against turns when a run shows 0% recall. Fix options: move due turns inside the horizon, or have report.py only count due-and-reached recalls. Also: subagent root-caused that seed-81's 13/22 notes 'missed' was partly the elements/characters add_new gating (see subject-add_new gating memory), not DSS model failure.

### #104 ~~ARCHIVED~~

*updated 2026-08-23 22:45:06*

dayna_ss soak B1/B2 (2026-08-12): (B1) SoakRun._clamp_note_windows (called in __init__) clamps every note's recall.due and plant.turn to `args.turns - 2`, so short runs actually fire recall probes (seed-81's 13 echo notes had due 62-78 → never fired → recall 0% was a metric artifact). (B2) report.py `_recalls` adds a `reached` flag (due < len(results)); the echo/recall_success denominator counts reached recalls only — never-due notes no longer count as failures; the report table has a "reached" column.

### #112 ~~ARCHIVED~~

*updated 2026-08-23 22:45:09*

dayna_ss soak perf: per-entry DataSummarizer calls embed the ENTIRE subject map via mark_field (FormattedData._str of the whole `elements` dict) + the full internal history (~100-153KB chars ≈ 25-38K tokens), so total element time is O(N²) in element count (N calls × N-token prompts) — per-call latency 7.5s→12s as elements.json grew 32KB→143KB, and max_prompt_tokens (54856) exceeded ctx 32768 causing server-side truncation that wastes prefill. skip_query halves per-entry CALLS but not per-call latency. Highest-leverage perf fix: scope per-entry context to just the target entry + compact sibling index and bound the internal-history payload (per-call tokens 50K→8-12K, latency ~12s→~3s, projected t20 ~4-5min/turn vs 25min). Second: events on_new_scene archival is O(accumulated state) (78-150 calls on scene turns, events worker 1176s). (seed-89 analysis, 2026-08-13)

### #113 ~~ARCHIVED~~

*updated 2026-08-23 22:45:10*

dayna_ss importance-weighted detail (user design, 2026-08-13): entity-level importance does NOT exist — importance (0-100 + faction + reason) is only on sub-fields (relationships, group membership, event participants/events, objectives), each with its own query/update templates. `character_list` format template exists (compact roster: name + relationships keys only) and renders into system context via _context_order, but there is NO element_list/group_list, and the new-entry path's `{item_name}_list` lookup (data_summarizer.py:2560) misses the existing `character_list` key (characterS_list/elementS_list). WHY per-entry prompts are huge: mark_field(branch_name) (summarizer.py:3142-3177) strips only the path MARKER tail of non-matching lines, leaving the line CONTENT intact — so every per-entry update prompt embeds the entire rendered subject map (elements 143KB at 45 entries). Proposed fix: add entry-level importance field + importance-weighted rendering (high-importance entries full detail, low-importance = roster line) as the principled replacement for brute-force per-entry scoping.

### #156 ~~ARCHIVED~~

*updated 2026-08-23 22:45:05*

In the dayna_ss importance system, importance values exist only on sub-parts such as relationships, group memberships, and event participant roles, nested under a parent entry like a Character, so a user request to query or update importance targets the grandparent entry to cover all of its importance-bearing children at once, not the immediate parent field. Run the lightweight importance query every message and defer the full importance perform_update to a new-scene pass after gate checks; give the LLM the complete list of the parent entry's importance-bearing children including score and reason, and keep in mind a character can hold several importance types toward the same other character simultaneously rather than one value per character pair.

### #160 ~~ARCHIVED~~

*updated 2026-08-23 22:45:06*

dayna_ss engine: current_scene.start is written ONLY by first-scene population (summarizer._populate_from_first_scene, guarded by the .populated_from_first_scene marker); it is frozen forever after because (1) the SceneStart alias carries `no_update: true` in BOTH subjects_schema.json and subjects_schema_sceneagg.json, enforced at data_summarizer.py:1861 (`_process_field`), and (2) CurrentScene's only trigger is `always: query_branch_for_changes` with now_update_prompt_template (targets now.*/what only). On a new-scene turn summarizer.py:1518-1543 only bumps `_scene_number` and `_scene_start_message`; nothing rewrites `start` or resets `now`. The old scene IS archived to events.scenes via StoryEvents' `on_new_scene: [add_new,...]` trigger. Minimal fix = schema-only: add `scene_start_query_prompt_template` + `scene_start_update_prompt_template` to CurrentScene.defaults and `on_new_scene: [{"action":"query_branch_for_changes","prompt_template":"scene_start_update_prompt_template","skip_query":true}]` to CurrentScene.triggers, in BOTH schema files (bq_template must be non-empty even with skip_query — data_summarizer.py:646). `perform_update` action CANNOT use custom template names (only reads key 'update_prompt_template', _get_effective_setting at data_summarizer.py:431). Path-based branch updates bypass no_update (only field traversal at :1861 checks it). Live current_scene.start never carries start.when._message_node in any turn, so the scene-bounded dialogue window at summarizer.py:2573-2577 always falls back to flat last_x; summarizer.py:2578 elif is dead code.

### #162 ~~ARCHIVED~~

*updated 2026-08-23 22:45:09*

current_scene.json scene `start` block is permanently stuck at the opening scene (frost-hold stair) for every scene in the fantasy_epic__40c80fa4 run; `now.what` latches at scene creation and only refreshes on scene transition (scene-transition detection in summarizer.py:1463-1516; budget split via _scene_start_message). _current_scene_recap (summarizer.py:243) feeds instruction generation from this stale scene state → stale instructions.

### #164 ~~ARCHIVED~~

*updated 2026-08-23 22:45:06*

dayna_ss current_scene.start rewrite (shipped 2026-08-17, both schemas): SceneStart had no_update:true + CurrentScene only an `always` now-update trigger → `start` was byte-identical forever after turn-0 population (verified 40 turns/9 transitions, fantasy run). Fix: CurrentScene.defaults gained scene_start_query_prompt_template + scene_start_update_prompt_template (rewrite what+start.*+reset now.* from the transition exchange, stamp start.when._message_node={{current_message_node}}; bq+bu templates BOTH required — data_summarizer.py:646 gates on both even with skip_query), and triggers gained `on_new_scene: [{action: query_branch_for_changes, prompt_template: scene_start_update_prompt_template, skip_query: true}]`. Engine defensive stamp in summarize_latest_state's scene handler writes start.when._message_node = f"{max(0, len(history)*2 - 2)}_1_1" (same value as new_scene_start_node since prepare_context got history[:-1]) so the scene-bounded dialogue window + rolling-summaries scene key always resolve even on NO_UPDATES_REQUIRED. Live-verified: trigger registered at runtime, update-only fires each transition, nodes stamped.

### #171 ~~ARCHIVED~~

*updated 2026-08-23 22:45:05*

Importance values in dayna_ss live only on sub-parts nested under a parent entry, such as relationships, group memberships, and event participant roles under a Character. When the user asks to query importance, run the branch query at the parent node to collect changes across all its children in one pass, and give the query prompt the full list of that parent's relationship entries with scores and reasons. Importance is keyed per relationship type, so the same pair of characters can have several distinct entries.


## CONFIG_VALUES  [ARCHIVED]

### #4 ~~ARCHIVED~~

*updated 2026-08-10 16:08:36*

OpenCode Go subscription exposes OpenAI-compatible endpoint at `https://opencode.ai/zen/go/v1` (chat/completions). Models: grok-4.5, glm-5.2, glm-5.1, gpt-5.6-luna, kimi-k3, kimi-k2.7-code, kimi-k2.6, deepseek-v4-pro, deepseek-v4-flash, mimo-v2.5(-pro), minimax-m3/m2.7, qwen3.8/3.7-max, qwen3.6-plus, hy3. API key stored in ~/.local/share/opencode/auth.json under `opencode-go`. Same pattern as Zen (opencode.ai/zen/v1).

### #6 ~~ARCHIVED~~

*updated 2026-08-10 16:08:36*

OpenCode Go endpoint call pattern: POST https://opencode.ai/zen/go/v1/chat/completions, Authorization Bearer auth.json["opencode-go"]["key"] (a dict; must read .key, not the whole entry). Cloudflare blocks default urllib UA (error 1010) — MUST send a browser User-Agent header (e.g. Mozilla/5.0 ... Chrome/126.0). deepseek-v4-flash is a thinking model: returns reasoning_content and can leave content empty at low max_tokens; pass enable_thinking:false and give max_tokens headroom (~100+). Response includes usage + cost.

### #69 ~~ARCHIVED~~

*updated 2026-08-12 03:17:12*

text-generation-webui fork git state: the real git dir is .git.og (plain .git was deleted — a program loading in the dir misbehaved with .git present). Use `git --git-dir=.git.og ...` or `export GIT_DIR=.git.og` for all git commands; the origin URL is intact inside .git.og (origin = oobabooga/text-generation-webui, fork = Th-Underscore/text-generation-webui). Branch is `lmdeploy` with 19 local commits (all additive LMDeploy wiring) on an upstream `dev` base from 2026-05-16. Upstream's ACTIVE default branch is `main`, and the fork is 33 commits behind origin/main (main went 2026-05-16 -> 05-31: llama.cpp bumps, MTP support, CORS localhost hardening, path-traversal fixes, torch 2.9/xformers pins, UI). Upstream main has NEVER had modules/lmdeploy.py — the LMDeploy loader is 100% fork-owned. lmdeploy pip = 0.12.3 built from the repo-root lmdeploy/ source tree (git repo on branch pr-4465 = the REJECTED GDN prefix-caching PR). The model: Huihui-Qwen3.5-9B-abliterated-AWQ-4bit, loaded via `./strt mp` (CMD_FLAGS_m.txt has --loader LMDeploy --cache-type q8 --cache-max-entry-count 0.8 --disable-prefix-caching --n_ctx 32768).


## CONSTRAINTS  [ARCHIVED]

### #8 ~~ARCHIVED~~

*updated 2026-08-23 22:45:07*

deepseek-v4-flash on the OpenCode Go cloud endpoint ALWAYS emits reasoning_content even with enable_thinking:false; it consumes the token budget, so a low max_tokens truncates/empties the real content (finish_reason=length). Fix: give max_tokens generous headroom — soak defaults are guide ~2500 (guide_turn_with_retry default max_tokens=2500; the guide CloudModel also uses reasoning_effort="none", fallback_reasoning=False), judge ~3000. The judge returns real JSON in `content`; the fallback to reasoning_content only fires if content is empty.

### #152 ~~ARCHIVED~~

*updated 2026-08-23 22:45:07*

deepseek-v4-flash on opencode.ai/zen/go ALWAYS emits a large `reasoning` field on big prompts (21-33K+ chars) even with reasoning_effort="none" + enable_thinking:false — the "none kills reasoning" rule (#25) does NOT hold on this endpoint. On large requests the reasoning alone can eat a small max_tokens budget → finish=length with EMPTY `content`. Verified empirically (2026-08-16): a 38-40K-token overview request needs max_tokens≈16000 (covers ~8.5K reasoning tokens + ~9K content), and the response field is `reasoning` (NOT `reasoning_content`) so the complete() fallback never surfaces it. The failure presents as "judge output was not JSON:" (empty), socket read-timeouts (large calls take 70-150s, over the old 120s timeout), or occasional HTTP 500. Soak fix: overview/judge/auditor budgets 16000/8000/1600 + timeouts 300s; cloud_client.py empty-content retry is tiered — small budgets (<4096) double, large budgets just +2048 (doubling large budgets invites proportionally longer reasoning with no content and triple-spends minutes). Even at adequate budgets the model randomly enters reasoning-only states on huge transcripts (temp variance), so bounded rerolls are load-bearing.


## ARCHITECTURE

### #3

*updated 2026-08-09 15:48:24*

dayna_ss engine feeds the "latest exchange" into LLM prompts via custom_state["history"]["internal"] (summarizer.py appends `[What was the very last exchange?, <dialogue>]`). Live tests (live_soak.py) must replicate this or the model sees templates referencing an exchange that isn't there. Live models also write details to different schema paths than scripted fixtures, so live soak probes search whole subject state, not fixed dot-paths.

### #5

*updated 2026-08-24 12:06:07*

dayna_ss message-node scheme: message_idx is 0-based with user messages at `2*turn` (assistant at `2*turn+1`); scene start nodes stored as `_message_node` strings like "140_1_1" whose first int is the message idx, so `exchange = node_idx // 2`. Scene-bounded dialogue window: `Summarizer._scene_dialogue_window(history, context_retriever, kwargs)` (summarizer.py) returns `last_x` capped at `last_x_max=8`, floor `last_x_min=2`, flat-6 fallback. `base_state["truncation_length"]` is 65536 (summarizer.py base_state and the harness build_state both use 65536). Hermetic test: tests/scene_window_test.py.

### #9

*updated 2026-08-24 12:05:57*

DSS soak dashboard is a standalone Gradio app at extensions/dayna_ss/tests/soak_dashboard.py. Launch: `python tests/soak_dashboard.py --port 7861` (no TGWUI server needed). Reads tests/runs/<config_hash>/ dirs: turn_XXX/result.json (keys: turn, user_input, reply, instr_prompt, history_path, probes{note_id:{needle,present,paths:[(subj,dotpath)...]}}, recalls{}, judge{style_score,quality_score,memory_fidelity,notes[],summary}, ts, local_calls, cloud_usage{calls,prompt_tokens,completion_tokens,total_tokens,model}, dt_s) + user.txt/reply.txt/instr_prompt.txt/state_snapshot/*.json + manifest.json + report.json. Gradio is 4.37 — gr.Statistic does NOT exist, use gr.HTML cards; gr.Accordion supported. Dashboard rebuilds retention timeline from raw turn probes (report-agnostic) via soak_dashboard._per_note_timeline.

### #26

*updated 2026-08-17 11:58:29*

DSS soak live plan: the cloud guide owns a rolling ~15-exchange plan (run_dir/plan.json + per-turn snapshots). Replan on `--plan-every N` turns (default 5, 0 disables) via build_plan_messages (full outline + prev plan + memory + recent) → PLAN_SYSTEM_TMPL → json_mode JSON with beats{turn,action,note_ids,recalled}. parse_plan has _repair_truncated_json fallback. Steering turns inject plan_view (one line/beat) ON TOP of notes_visible_at window + static style block — plan does not replace the window. Replan calls counted in guide.stats()/result.json plan key. Dashboard shows Live plan accordion (replan timeline via _plan_timeline) + per-turn Live plan snapshot.

### #28

*updated 2026-08-24 12:05:56*

dayna_ss new-entry generation (_detect_and_add_new_entries_to_branch, data_summarizer.py) is the ONLY path that does one-shot full-JSON generation (schema snippet + example, the entire nested object in one call) and is the #1 source of parse failures on small models — the recursive field-by-field machinery that already works for existing objects (_update_recursive → _traverse_structure → _process_field → _update_field → _generate_field_update) is NOT used for new entries. _generate_field_update has a comma-split fallback for list[str].

### #29

*updated 2026-08-17 11:59:08*

dayna_ss soak final overview: an end-of-run judge pass writes run_dir/final_overview.json (overall_score, verdict, arc_progression, style_consistency, memory_fidelity, notes[], strengths[], weaknesses[], summary) auto at the end of long_horizon_soak.py main(), or as standalone backfill via `python long_horizon_soak.py --final-overview <run_dir>`. Per-turn judge notes live in turn_XXX/result.json under `judge`. report.py renders a "## Final overview (end-of-run judge)" section; soak_dashboard.py shows a "Final overview" accordion. IMPORTANT (2026-08-11): both the final overview AND the per-turn judge now score DSS-ONLY — {name2}'s output (voice adherence to dss_directive, memory fidelity, note honoring, beat advancement) — with {name1} (cloud guide) lines shown as [CONTEXT — NOT scored] and {name2} lines as [EVALUATED] (JUDGE_SYSTEM_TMPL + FINAL_OVERVIEW_SYSTEM_TMPL in long_horizon_soak.py). JSON shapes unchanged for report.py/dashboard. Caveat: in soak_dashboard _run_display, do NOT name local variables `rows` — it shadows the outer per-note timeline rows and crashes the "Retention by type" accordion.

### #33

*updated 2026-08-24 12:06:03*

dayna_ss soak has a THIRD cloud role — the auditor (long_horizon_soak.py _run_audit + build_audit_messages). Spreadsheets carry dss_beats: [{id, turn, subject, content, expect?}] = "DSS should save a new X with Y data". On a beat's turn the auditor gets the beat + DSS's actual subject JSON (via _memory_summary) and returns {id, status: saved|partial|wrong|missing, detail}. Runs on beats' turn, or every --audit-every N (0=beats only). Verdicts live in turn result.json under "audit"[]; report.py _audits() aggregates them into report["audits"] (rows/total/saved/partial/missing/wrong/error) rendered as a "## DSS save-beat audits" md table; soak_dashboard shows per-turn "DSS save-beat audit" accordion + run-level "DSS save-beat audits" accordion (via _run_audits). Auditor config: --auditor-model (default deepseek-v4-flash), --audit-every. run_tests.py passes through --audit-every/--auditor-model/--judge-model. Harness logging: all [soak] lines go through soak_log() which timestamps AND appends to a dedicated logfile (--log-file; default runs/logs/<run_id>.log where run_id = <spreadsheet>__<hash>). Phases marked with soak_log + _elapsed(t0) so pauses are attributable.

### #38

*updated 2026-08-24 12:06:03*

dayna_ss soak per-turn usage format: each turn_XXX/result.json now has per-turn usage DELTAS in local_usage/cloud_usage/judge_usage/auditor_usage {calls, prompt_tokens, completion_tokens, total_tokens, model}, plus wall_ts (wall-clock ISO; the old 'ts' field is story-scene time), and final_overview.json has generated_at. Snapshots are taken around run_turn in main() of long_horizon_soak.py. IMPORTANT: pre-change runs stored CUMULATIVE cloud_usage counters that RESET on resume — the dashboard's _usage_series() reconstructs per-turn deltas from those by detecting the resume boundary via the calls counter decreasing (the reset is NOT visible in prompt_tokens which keep increasing); without this the totals overcount badly (197 vs 26 real cloud calls). _turn_wall_map() maps turn -> HH:MM:SS from wall_ts, else file mtime. New-format runs use per-turn fields directly.

### #40

*updated 2026-08-11 13:50:05*

dayna_ss soak frozen-output-loop root cause (2026-08-10, cozy run): not model degeneration but a deterministic loop. 18-turn run had only 3 distinct reply files (turns 1-4 md5-identical, 5-17 md5-identical, one-word diff). Four mechanisms: (1) LocalModel._complete hardcodes temperature=0.0 (long_horizon_soak.py:195) — greedy decoding locks one bad reply forever; (2) _wrap (long_horizon_soak.py:177-184) prepends last 6 raw exchanges incl. the model's OWN prior replies labelled name2: immediately before instr_prompt — the verbatim-copy source (harness prompt_builder is identity so engine retrieval never reaches the model, long_horizon_soak.py:909); (3) instructions are written by the SAME collapsed model from the same last-6 window (summarizer.py:800-831) then fed back verbatim (855-863), byte-identical every turn, describing a stale scene; summarizer.py:860 tells it to match tone of BOTH voices — speaker-swap voice leak (Evelyn's dialogue = Prudence's guide lines re-spoken); (4) current_scene is frozen (no trigger_map, _update_field refuses non-trigger leaf updates) yet to_context:true and re-injected in system msg every turn (summarizer.py:1398-1435). Secondary bug: base_state overwrites name1/name2 to SYSTEM/DAYNA (summarizer.py:85-87,1348) while instr says "You are Evelyn". Fixes: temp>=0.7, stop feeding own replies, voice-guard instructions against dss_directive, unfreeze current_scene.

### #42

*updated 2026-08-10 19:32:19*

dayna_ss instructions.json forward-copy bug (2026-08-10): the empty-subjects early-return copy path (summarizer.py:1110-1116, copies all *.json from last_history_path to new_history_path) ALSO copied instructions.json — but instructions.json is a seed-keyed CACHE (generate_instr_prompt looks up input_key=str(state['seed']) in it, summarizer.py:790), not a subject. So with --seed 43 (fixed seed), every turn hit the cache and reused the SAME byte-identical instructions generated once, never regenerating per-turn → stale self-written instructions fed back verbatim (a frozen-output-loop contributor). Fix: exclude instructions.json from the forward-copy set (summarizer.py:1112). Now each fresh history dir has no instructions.json → cache miss → per-turn regeneration.

### #44

*updated 2026-08-24 12:06:07*

dayna_ss soak LocalModel now renders the ENGINE's internal history instead of the harness _wrap: `LocalModel._engine_internal_messages(state)` converts `state['history']['internal']` pairs into user/assistant messages exactly like modules.chat.generate_chat_prompt does in production (skips `<|BEGIN-VISIBLE-CHAT|>` on the user side, keeps the assistant "I am ready..." reply, renders retrieval Q&A + scene-bounded last-X window + "Analyze all..." marker). `_complete` sends system=context + those pairs + the prompt as the final user message; `_wrap` (last-6 raw exchanges via history_provider) is now only the fallback when there's no internal history (Level 1). This fixed the frozen-output anchor: the model's own previous reply is no longer the last text before the instruction block.

### #45

*updated 2026-08-24 12:06:24*

dayna_ss implicit voice guard (Q3, 2026-08-10): summarizer.py's instruction-gen prompt now says the instructions must be in {name2}'s own voice per the writing-style directive in system context and must NOT instruct imitating {name1}'s voice/register/phrasing; the reply instr_prompt (both do_instr True and False branches) now says "Write strictly in {name2}'s own voice as defined by the writing-style directive in your context/system instructions; do NOT imitate {name1}'s voice... never quote or re-use {name1}'s lines" replacing the old "Maintain the style and tone consistent with recent messages from both {name1} and {name2}" (that 'both' line was the direct voice-blend leak). Verified live: replies are third-person Evelyn. This is implicit (single inference, stricter prompting), NOT an explicit generate-check-regenerate guard.

### #46

*updated 2026-08-24 12:05:52*

dayna_ss soak voice carrier is general_info.writing_style, NOT the harness's raw context line. Production renders it into the reply prompt two ways: the `general_info` format template ("Writing Style --- {{writing_style}}") is `_context_order` to_context:true so it's appended to the system context every turn (format_templates.json:36, summarizer.py:1399-1436), and `_format_general_info_static` (summarizer.py:1357) re-injects it once a scene archives. The soak pins it verbatim via SoakRun._seed_general_info_style(history_path) (long_horizon_soak.py, called in Level-2 run_turn right after generate_instr_prompt returns) which writes general_info.json's writing_style = spreadsheet dss_directive into the session dir; the engine's copy-forward (summarizer.py:1111-1115) propagates it every turn. Turn 0 still uses the engine's LLM paraphrase (seed lands after turn 0's prompt is built) — designed one-turn lag. Engine prompts (summarizer.py:804/820/860/870) reference "the Writing Style directive (general_info.writing_style) in your system context" — do NOT rename that concept back to a harness label.

### #48

*updated 2026-08-24 12:05:52*

dayna_ss engine fixes for the 3/5-verdict run (2026-08-11, summarizer.py): (1) M4 — the first-scene early-return that skipped the DataSummarizer whenever no scene was archived and no transition was detected is REMOVED; the DataSummarizer now runs every turn (except fresh-chat initial population), so always-triggers fire and current_scene.now etc. update per turn. Cost: many more local LLM calls per turn. (2) M1 — generate_instr_prompt now inserts a `[VOICE: ...]` switch-frame between the quoted latest-user-input and the reply instructions, and the REMEMBER anchor is explicitly THIRD PERSON, in both the do_instr and do_instr=False branches. (3) M3 — the reply prompt inlines the exact writing-style text as `WRITING STYLE DIRECTIVE (you MUST follow it):` pulled from retrieval_ctx.general_info.writing_style (the production carrier, seeded by the harness), falling back to a system-context reference on turn 0. (4) M2 — `_clean_generated_instructions` deterministically post-checks the generated instruction block (drops near-duplicate sentences + near-verbatim echoes of the user input; no extra inference). The 9B local model anchors on the last big text block, so generation-adjacent framing beats buried system-message directives.

### #55

*updated 2026-08-18 05:14:41*

dayna_ss soak perf/quality fixes (2026-08-11, engine, for the 3h/894-call + 3/5 run): (1) P-P1 — `_filter_new_entry_names` in data_summarizer.py deterministically dedupes `add_new` proposals against existing dict keys (exact + normalized containment, min 4 chars) and caps batch (ADD_NEW_MAX_PER_CALL=8) + total (ADD_NEW_MAX_TOTAL_ENTRIES=120). Kills the ElementMap/GroupMap micro-element fabrication that ran add_new every turn (always trigger). (2) P-P2 — `_perform_branch_query` now returns False on a negative (NO/UNCHANGED/NO_UPDATES_REQUIRED/"n"/prefix NO) instead of proceeding to the wasted apply_updates call; returns True otherwise. (3) P-P3 — `_execute_action` returns a 3-tuple (stop_processing, gate_failed, skip_children); a negative branch query short-circuits recursion so `_update_recursive` skips drilling into sub-branches that would also answer NO. (4) Q-P1 — `summarize_latest_state` first-scene population now gates on the `.populated_from_first_scene` marker (written to BOTH last_history_path and new_history_path); population runs exactly once per session, so an auto-detected scene transition before any archive can no longer re-run `_populate_from_first_scene` and wipe accumulated memory (was: elements 206→18, Evelyn/Mrs. Winters conflation, c13 probe lost). All four verified: hermetic green, live 2-turn smoke turn-0 82→51 calls, turn-1 delta 46→33, elements 9→13 no spikes, replies third-person.

### #67

*updated 2026-08-24 12:05:57*

dayna_ss subject-loop parallelism: Summarizer config max_subject_workers (default 0 = serial, production byte-identical unless opted in; dss_config.json may set it; soak harness overrides via --max-subject-workers, wired in long_horizon_soak._make_summarizer). Parallel path uses ThreadPoolExecutor — each worker its own DataSummarizer with deep-copied custom_state + hybrid all_subjects_data snapshot (deep copy of all + live current subject swapped in) so post-loop chapter/arc checks read fully-updated state; shared PhaseManager behind a _LockedPhaseManager RLock wrapper (only mutating methods lock); processed_subjects_data reassembled in schema order. DataSummarizer is imported LOCALLY inside both summarize_latest_state and _process_subject_parallel (circular import: data_summarizer imports Summarizer at module level). Measured clean 2-turn L2 smoke (4 workers): turn 1 = 128s/26 calls vs 230s/46 serial (~1.8x). Server MUST run with --disable-prefix-caching + --max-batch-size 8 for safe concurrency (prefix caching segfaults under 4+ concurrent requests).

### #88

*updated 2026-08-24 12:06:18*

textgen modules/lmdeploy.py generate_with_streaming had a drain race that truncated EVERY generation at arbitrary chunk boundaries: the sync loop `while not future.done()` exited the moment the async future completed, dropping the final chunks (the last chunk carries the JSON tail). Worse under concurrency (shared event loop widens the race window) — produced 1-3 token completions ('{', 'UNCH'), 'Failed to parse' errors at varying lengths, AND cut-off story replies. Same race was the true cause of the earlier 'empty completion' bug (the empty-retry in the harness masked it). FIX (2026-08-12): drain until BOTH future.done() AND queue empty, with a 5ms sleep on QueueEmpty, plus future.exception() surface. Verified: 18/18 full-length valid JSON under 6-way concurrency. Companion engine fixes in data_summarizer.py: json_repair fallback (_tolerant_json_loads) for token-level JSON slips (colon-for-comma), _is_negative_verdict treating NO/NO_UPDATES_REQUIRED/UNCHANGED as no-update in _parse_llm_field_updates AND _generate_field_update (was logging scary errors + losing updates). json_repair installed in lmd-tg env.

### #89

*updated 2026-08-14 21:35:47*

dayna_ss Q-P3 memory-surfacing fix (2026-08-12): DSS stored planted notes (retention 31.8% on the 35B run) but recall stayed 0% — notes never surfaced in replies. Root cause: the contamination guard deliberately keeps note text out of DSS prompts (by design), but the reply prompt never told DSS to USE its own stored memory (which IS in system context every turn via Characters/Events/general_info). Fix: added "Draw naturally on the characters, items, and past events established in your system context (General Info, Current Scene, Characters, Events sections); reference them concretely" to the reply instructions — in BOTH do_instr branches of generate_instr_prompt (summarizer.py, right before the REMEMBER anchor) AND the level-1 reply prompt in long_horizon_soak.py (rule #36 propagation). Verified on live 3-turn smoke (Qwen3.6-35B, seed 8001): directive present in every instr_prompt; replies weave in established details across turns (Le Creuset dish, oven-door, Agnes's handling, Constable Finch, Mrs. Ogilvy). Also: long_horizon_soak.py judge calls at lines 1283/1396 now guard `args.judge_every` (0 disables — `turn % 0` was a ZeroDivisionError).

### #91

*updated 2026-08-15 05:03:09*

dayna_ss empty-world-on-restart bug (2026-08-12, FIXED): the 35B run cozy_mystery__df2fce53 had ALL audit beats "missing" because characters/groups/events subjects were EMPTY the whole run. Root cause in summarizer.py retrieve_and_format_context: `is_new_scene = True` was only set in the initial-world CACHE-MISS branch; on a cache HIT it stayed False → SummarizationContextCache(is_new_scene_turn=False) → _populate_from_first_scene gate (self.last.is_new_scene_turn) never passed → no first-scene population. The initial world cache only ever holds EMPTY placeholders + seeded general_info (population writes to the session path, never back to the cache), so a cache hit ALWAYS needs population. Trigger: the run's first launch crashed (Broken pipe) after writing the cache but before population; the restart hit that cache and skipped population. FIX: set is_new_scene = True in the cache-hit branch too (both branches now set it). The initial world cache path is per-run-dir sandbox: <run>/sandbox/user_data/history/soak_<seed>/initial_world_cache/<content_hash>/ — content hash = hash(char_context + greeting + user_bio), seed-INDEPENDENT. Diagnostic note: on a fresh run turn-0 summarization ~23s + empty subjects = population skipped (cache hit); ~11 min + populated subjects = population ran (cache miss).

### #93

*updated 2026-08-24 12:06:15*

dayna_ss wrong-subject routing fix (2026-08-12): DSS was saving characters into the ELEMENTS subject (Mrs. Arbuthnot, Marmalade Cat as elements) because the elements `new_entry_query_prompt_template` was a permissive fallback with no routing constraints. Fix: (1) user_data/example/subjects_schema.json gained a top-level `subject_routing` map (per-subject "what belongs here" descriptions — data-driven, no hardcoded dispatch per rule #35); SchemaParser parses it into `self.subject_routing`. (2) DataSummarizer._build_subject_routing_guide(subject_name) composes a dynamic "SUBJECT ROUTING" block from the schema at runtime (lists every subject + description + "<<< YOU ARE NOW POPULATING THIS SUBJECT >>>" marker on the current one), prepended to every add_new query prompt in _detect_and_add_new_entries_to_branch. (3) Templates tightened: elements query has "WHAT NOT TO INCLUDE" (no named people/speaking beings→characters; no orgs/committees→groups; no narrative events→events; creatures only as props/background fauna); elements entry creature-kind requires "does not speak or recur as a story participant"; characters query states named animals/pets count as characters. The schema is copied into each run's sandbox on fresh-chat init, so the fix applies to runs started after the edit.

### #96

*updated 2026-08-23 22:46:36*

dayna_ss engine: subject add_new triggers have two gating gaps that cause 'DSS never saved X' audits. (1) characters add_new is gated behind `on_new_scene` (subjects_schema.json CharacterMap.triggers) — in a 20-turn run with only 1 scene transition, characters are only ever queried for new entries once; mid-scene character introductions (e.g. Mrs. Arbuthnot in-story turns 1/2/7/14+) are never proposed. (2) elements add_new is ordered AFTER perform_gate_check in `always: [perform_gate_check, add_new]`, and a negative gate check short-circuits the whole branch (data_summarizer.py _update_recursive returns early on gate_failed) — one NO freezes the subject for turns (elements.json was byte-identical 12+8 turns, swallowing a turn-11 watering can). Fix directions: A1 move characters add_new into always triggers; A2 reorder elements triggers to [add_new, perform_gate_check] or run add_new even on gate_failed. Also the new_entry_query_prompt_template scopes 'Throughout the last scene' which misses same-turn objects.

### #97

*updated 2026-08-17 11:58:29*

dayna_ss soak judge/final-overview templates carry a STRICT SCORING RUBRIC (2026-08-12) with per-dimension anchors and MANDATORY FLOORS in JUDGE_SYSTEM_TMPL + FINAL_OVERVIEW_SYSTEM_TMPL (long_horizon_soak.py). Key floors: recurring identity confusion (name2 casting herself as another character/suspect) caps memory_fidelity ≤2 and overall_score ≤3; >half notes missed caps quality ≤3 and overall ≤3; verbatim re-quote of name1 or repeated closing lines caps style ≤3; scene inconsistency caps quality ≤3. overall_score is weighted (voice 30/memory 25/notes 25/progression 20), NOT a straight average, and must be consistent with the per-turn judge records passed in. Backfilled on cozy_mystery__9284c44d: 4/5 → 3/5 (matches human read). Old-run overviews remain under the old rubric; regenerate with `python long_horizon_soak.py --final-overview <run_dir>`.

### #98

*updated 2026-08-24 12:05:57*

dayna_ss scene-part budget split (2026-08-12): Summarizer config `max_scene_part_messages` (default 12; soak --max-scene-messages, 0 disables). In summarize_latest_state's auto-scene-detection block, if `(len(history)*2) - _scene_start_message >= budget`, force is_new_scene_turn=True (skip the LLM transition check), so on_new_scene triggers (characters/elements add_new, scene archive) fire on a regular cadence. `_scene_start_message` is a CurrentScene schema field (engine-meta, like _scene_number), set on every scene start + initialized on the population turn. Known limitation: current_scene `what` has no update trigger so it never advances on splits → events.scenes dedups scene-parts under one title (scene NUMBER still advances; archive-title regeneration is a pending follow-up).

### #99

*updated 2026-08-24 12:06:10*

dayna_ss population-once marker is NOW session-persistent via general_info.json: `_populated_from_first_scene: true` key written at population time (in addition to the `.populated_from_first_scene` file in last+new history dirs). The general_info key is carried forward every turn by the GeneralInfo subject, so it survives history-dir rotation (the file marker alone did NOT — population re-ran at turn 12 wiping memory, bug A3, fixed 2026-08-12). Guard in summarize_latest_state checks all three.

### #100

*updated 2026-08-24 12:06:18*

dayna_ss soak auditor statuses: AUDITOR_SYSTEM_TMPL (long_horizon_soak.py) has FIVE statuses — saved|partial|wrong|missing|unassessable. `unassessable` = the beat couldn't be satisfied (entity never planted in-story, or subject frozen/gated that turn) — NOT a DSS failure. `_run_audit` injects a deterministic DSS CAPABILITY CONTEXT per beat via `_audit_capability_context(beat, turn)`: subject-change turns (state_snapshot md5 diffs), scene-transition turns (characters add_new gating), and whether the beat's expected entity appears in user.txt/reply.txt. report.py _audits + dashboard pass the new status through (counted separately).

### #101

*updated 2026-08-24 12:06:03*

dayna_ss ElementMap triggers reordered (2026-08-12, A2): `always: [add_new, perform_gate_check]` — add_new now runs EVERY turn even when the map-level gate would answer NO (fixes the watering-can class: new entity present but gate froze the whole branch). GroupMap left as `[perform_gate_check, add_new]`.

### #102

*updated 2026-08-24 12:05:56*

dayna_ss A5 (2026-08-12): the five discovery `new_entry_query_prompt_template`s (CharacterMap, GroupMap, ElementMap, PastEvents, CrucialEvents) in subjects_schema.json are scoped to "In the latest exchange(s) (the most recent user and assistant messages, whether or not a scene boundary has been crossed)" — so same-turn entities (the watering-can class) are catchable before a scene boundary. The Scenes archive template stays scene-scoped.

### #103

*updated 2026-08-24 12:06:28*

dayna_ss identity anchor (2026-08-12, A6, fixes the Evelyn→Prudence conflation): an IDENTITY BINDING block ("{{ name2 }} is the AI character being roleplayed (protagonist); {{ name1 }} is the other principal character; never assign {{ name1 }}'s identity/occupation/biography/voice to {{ name2 }}") is in five character templates — Character.branch_update_prompt_template + branch_query_prompt_template, CharacterMap.new_entry_prompt_template, Characters initial_population identification + population prompts. To render it: `_create_update_prompt` (data_summarizer.py) passes clean `name1`/`name2` format keys (beside legacy `{user}`/`{char}`), and _populate_subject_identify (summarizer.py, 4 format sites) passes name1/name2 into the identification/population/retry prompts. Harness build_state context has a "Canonical cast:" line and _seed_general_info_style seeds `_cast` (name1/name2+role+description) into general_info.json; format_templates.json general_info renders a conditional "Cast ---" line (N/A when _cast absent).

### #106

*updated 2026-08-24 12:06:13*

dayna_ss elements/events store freeze — FIXED 2026-08-12. Root causes were: (1) ElementMap/GroupMap/CrucialEvents/PastEvents new_entry_query_prompt_template carried "respond with the exact str: NO_NEW_ENTRIES_REQUIRED" (the model always picked it — ablation verified; characters said "empty array acceptable but not ideal" and was the only subject that grew); (2) Element was the only entry class missing branch_update_prompt_template so its always: [query_branch_for_changes] was silently skipped (data_summarizer.py:396-402); (3) branch_list rendered <EMPTY> (data_summarizer.py:2501-2503) so add_new queries saw no existing entries; (4) GroupMap triggers were gate-first. Fixes applied: dropped the NO-escape (→ "empty array acceptable but not ideal") from the three discovery templates; added Element branch_update_prompt_template (modeled on Character's, element-scoped); branch_list now renders real entry keys via recursive_get(formatted_data.data, keys); GroupMap triggers reordered to [add_new, perform_gate_check]. Verified: schema parses, hermetic green, live 2-turn smoke (cozy_mystery__940a87d2) — elements/groups files now change across turns, groups added "the Historical Society", per-element updates land real content (bakery attributes, kitchen traits, events). NOTE: events.json only gates on ON_NEW_SCENE (by design); its population on scene transitions still needs a longer-run verification.

### #107

*updated 2026-08-12 21:07:13*

dayna_ss soak auditor scheduled-turn framing bug (2026-08-12): the auditor treats each dss_beat's `turn` as a DEADLINE, but it should be the EARLIEST intended save — a beat is satisfied whenever the entity is saved once it appeared in-story and the subject is writable, judged at the EVALUATION (poll) turn. In cozy_mystery__2502c5ad, d1 (Mrs. Arbuthnot) was saved into characters.json at turn 10 but every poll still said unassessable ("outside the beat's window"). Fix sites in tests/long_horizon_soak.py: build_audit_messages beat header (~:918), AUDITOR_SYSTEM_TMPL beat definition (~:886-888) and unassessable status (~:907-909), and the capability-guide text (~:1168-1171) — all should say "by the evaluation turn N" and "if the entry IS present at the evaluation turn, mark saved, not unassessable".

### #108

*updated 2026-08-24 12:06:03*

dayna_ss schema types (2026-08-12): the soak harness supports schema-type test conditions. `--schema-type 1|2|3` (+ `--schema <path>` override) resolves the subjects_schema file (order: --schema, per-genre+type, per-type, per-genre, legacy default) and sets Summarizer.config["subjects_schema"] (runtime.extension_dir-relative); schema_type is part of the run-id hash so different schemas get fresh run dirs. Type 1 = current default subjects_schema.json (incremental-per-turn, baseline). Type 2 = user_data/example/schemas/subjects_schema_sceneagg.json: events + current_scene update EVERY message; everything else (characters/groups/elements/general_info/arcs) on ON_NEW_SCENE with an aggregated "SCENE CONTEXT" block ({{ scene_recap }} + {{ scene_events }} lazy lambdas injected into 19 b-pass templates via _create_update_prompt format_kwargs in data_summarizer.py — zero cost unless referenced). StoryEvents add_new is ALWAYS, per-event query ON_NEW_SCENE. Events subject gate removed entirely (the gate answered NO every transition, freezing events). 20.5 synthetic-chat isolation: `--synthetic-chat` (L2 only) — cloud guide writes BOTH sides, DSS runs only summarize_latest_state, judge scores memory/note-retention only. 20.6: LocalModel.stats() returns max_prompt_tokens (peak single-call prompt, reset each stats() snapshot) + ctx_size (32768), stored per-turn; dashboard shows "peak ctx X/32768 (N%)". Full matrix + design: extensions/dayna_ss/schema_types_and_test_matrix.md.

### #109

*updated 2026-08-23 22:46:02*

dayna_ss events-freeze TRUE root cause (FIXED 2026-08-12, both schemas): DataSummarizer's traversal resolves each events-branch field (events.past/scenes/events/chapters) through its wrapper alias to the shared `StoryEvents` dict class. The per-branch `new_entry_query_prompt_template`/`new_entry_prompt_template` living on the wrapper aliases (CrucialEvents/PastEvents/Scenes) were UNREACHABLE dead code — the effective schema inherits only defaults_to_inherit = [gate_check/branch_query/branch_update/update] templates (data_summarizer.py:51), and `_detect_and_add_new_entries_to_branch` (data_summarizer.py:507-518) requires BOTH new-entry templates on the class it actually reaches, skipping silently when missing. That is why events.json never populated outside first-scene init on every run. Fix: added generic new_entry_query_prompt_template + new_entry_prompt_template to StoryEvents defaults in BOTH subjects_schema.json and the sceneagg variant (wrapper-specific phrasing like "archive this scene" is intentionally lost — the generic template covers events/backstory/scenes via branch_list). Verified: hermetic sceneagg smoke — events add_new query now dispatches (22 calls vs 10), full suite green.

### #114

*updated 2026-08-23 22:46:26*

dayna_ss importance-weighted detail batch (shipped 2026-08-13, both schemas): entity-level `importance` (score 0-100/faction/reason) on Character/Group/Element/StoryEvent entry classes + `importance_detail_threshold` (default 50) on map classes; format templates render entries >= threshold as FULL profiles, below threshold as one-line roster entries (name + score + reason + desc-first-line); entries without importance default to FULL (back-compat). `{item_name}_list` lookup in new-entry discovery now resolves via a `list_template` key (character_list existed; element_list/group_list added). LocalModel has `--max-update-history N` (default 12, 0=unlimited): DataSummarizer calls send only the N NEWEST internal exchanges (bound_history=True path = generate_with_sse/generate_using_tgwui/generate_with_streaming), reply generation stays full. Element/Group/Character per-entry triggers now `{"action": "query_branch_for_changes", "skip_query": true}` (single self-detecting update call). Discovery templates carry a NAME EXACTLY directive ("use the exact name the story uses"). Reply instructions ban atmosphere-only + premature-conclusion phrasing and carry ROLE BINDING (name2=protagonist writing the reply, name1=whose turn just ended, never swap).

### #115

*updated 2026-08-17 11:58:29*

dayna_ss soak contamination guard trips ONLY when reasoning becomes output: CloudModel.complete surfaces msg.reasoning_content ONLY when content is empty (after a doubled-budget retry). deepseek-v4-flash on the guide (max_tokens 1500) burned the WHOLE budget on reasoning_content at high context (turn 7+), leaving content empty → the fallback returned the giant reasoning blob → it contained verbatim outline-note quotes (the guide reasons over notes) → guard fired. Fixes (2026-08-13): CloudModel gains fallback_reasoning (False for the guide = empty content is a failure, never the reasoning blob); guide constructed with reasoning_effort="none" + max_tokens 2500 (replan still passes reasoning_effort="low" explicitly); new guide_turn_with_retry(guide, spreadsheet, msgs, label) replaces all three guide.complete call sites (L2/L1 user turn, synthetic reply) — cleans, validates non-empty + no verbatim note text, retries 3x with a harness note naming the note id (never its content), RuntimeError only if all fail. The post-instr_prompt guard remains as a final backstop.

### #116

*updated 2026-08-13 13:57:29*

dayna_ss events.chapters schema shape is `list[Chapter]` (Chapters alias, subjects_schema.json:698) — the old archive code (check_and_archive_chapter/arc, data_summarizer.py) wrote it as dict-keyed-by-title and CRASHED once the events-population fix made chapters populate as a list (`TypeError: list indices must be integers` at the `events_data["chapters"][title]` write). Fix (2026-08-13): `_entries_as_list()` helper (dict→list(values), list passthrough, None/empty→[]) used by both archive functions + the sceneagg aggregation context; chapter archive now appends/replaces-by-title in the list; arc reads recent chapter as chapters[-1]; legacy dict-shaped runs are read transparently. Also: `current_custom_state` was only defined on the gate-check branch of both archive functions — a FORCED chapter/arc transition crashed with UnboundLocalError at the generation call; now initialized early in both. All other consumers (events/chapters format templates, sceneagg helper) were already list-aware. NOTE: the running soak process loads modules at init — engine fixes require a kill+resume to take effect.

### #117

*updated 2026-08-17 11:58:10*

dayna_ss dotted-name + alias resolution (2026-08-13): LLM-written paths with dotted entity names ("Mrs. Arbuthnot" without [brackets]) split into nested dicts (`relationships.Mrs` = {"Arbuthnot":[...]}) that crash context_retriever's relationship scan (`_get_character_important_relationships` iterates dict keys → `_get_importance` AttributeError → the whole retrieve_context try-block aborts → ALL subjects empty for the turn). Fixes: (1) `_get_importance` guards non-dicts, rel iteration normalizes dict-keyed rel lists, each subject extraction wrapped in its own try/except in retrieve_context; (2) new helpers in data_summarizer.py — `_collect_entry_aliases` (schema-agnostic walk for `aliases` arrays), `_resolve_dict_key` (exact→normalized→alias), `_resolve_path_keys` (exact, dot-split-join "Mr"+"Peters"→"Mr. Peters", normalized, alias) — wired into `_apply_branch_updates`, `_resolve_fuzzy_path`, `_filter_new_entry_names`; (3) `bracket` jinja filter (utils/helpers.py) + format-template path markers now render `[Mrs. Arbuthnot]`. IMPORTANT: recursive_set requires list indices as STRING-digit keys (int raises "non-integer key on a list"). Also fixed: general_info format template used bare `{{synopsis}}`/`{{writing_style}}`/`{{_cast}}` vars that don't exist in render context (root is `data`) → always rendered empty; now `{{data.*}}`. current_scene template gained "Messages in this scene" via `scene_messages` filter + `data._scene_number`.

### #118

*updated 2026-08-17 11:59:08*

dayna_ss schema edits DON'T reach resumed soak runs without a refresh: the engine's SchemaParser is built per-turn from `history_path/subjects_schema.json` (summarizer.py:1932-1933 `initial_schema_parser or SchemaParser(history_path/"subjects_schema.json")`), and each session dir's copy is made at fresh-chat init only — on resume the stale per-session copy is loaded and schema changes silently never apply (the per-turn copy-forward at 1259-1260 propagates the stale copy). Symptom (2026-08-14, Type 2 Mode B run cozy_mystery__dded7733): turn 22 = 244 calls / 34 min because the OLD Importance.on_new_scene trigger still fired 202 importance updates, while the source+top-level sandbox schema had the fix. FIX: harness main() now refreshes every `sandbox/user_data/history/**/{subjects_schema,format_templates}.json` from the resolved variant at startup (after the existing copy block, long_horizon_soak.py). Verified: transition turn 28 dropped to 46 calls / 6.3 min with 0 Importance generations.

### #119

*updated 2026-08-24 12:06:22*

dayna_ss character-subject wipe root cause + fix (2026-08-14, cozy_mystery__dded7733): the DataSummarizer.generate() try/except (data_summarizer.py ~571) swallowed ANY exception during _update_recursive and skipped the save_json write — leaving the history dir WITHOUT the subject file. Next turn loads `{}` (summarizer.py load site), and the copy-forward propagates the empty subject forever: 10 characters -> missing file at t13 -> `{}` for 15+ turns -> rebuilt with only 3 (Evelyn herself lost). Ran silently: the judge gave memory=5 while the character map was empty because _memory_summary omits empty subjects. FIX (shipped, both live): (1) generate() now persists the in-memory data (mutated in place, still holds prior entries) on exception, normalized-then-raw fallback; (2) Summarizer gains `_last_good_subjects` cache (deepcopy recorded on every successful save) restored at load time when a subject file is missing/empty. Both sites use getattr-guard for the hermetic ScriptedModel. Also: NAME EXACTLY directive was absent from the initial_population identification/population templates (elements became 'galvanised_watering_can' instead of 'watering can') — only present in new_entry_query_prompt_template.

### #120

*updated 2026-08-14 14:04:30*

dayna_ss judge-calibration + instruction-hardening batch (shipped 2026-08-14, hermetic green, for the dded7733 3.5/5 analysis): F1 `_clean_generated_instructions` (summarizer.py) is a 3-pass deterministic post-check — strips whole meta sentences ("Ensure your response...", "Your response should end on..."), deferral/construction clauses within kept sentences ("leaving the next move in X's hands", "as a prop to emphasize", "rather than a summary", "Avoid declaring the mystery resolved or naming the culprit definitively;"), and near-duplicate sentences; `_SENT_SPLIT_RE` is abbreviation-aware (Mrs./Dr.). F2 gen-requirements #10 bans deferral endings + repeated imagery, new #12 (every instruction = concrete imperative action, no meta-construction phrases), new #13 (max 3 beats/block), #5 reworded away from "The final response should be..."; reply hard rules (BOTH do_instr paths) ban turn-handover endings + self-narration of the reply. F3 word caps in dss_directive (cozy ~80/para, fantasy ~150, romance ~60). F4 judge calibration (long_horizon_soak.py): judge gets the turn's INSTRUCTIONS block with ATTRIBUTION RULE (verbatim transcription = poor execution; mandated content not charged as memory/creative error); ">half notes missed -> quality<=3" floor REMOVED; `_planted_note_ids()` excludes unplanted notes (needle never in story; style-holds always in scope) from the in-scope set, wired into _run_judge/_run_overview/_backfill_final_overview; memory grounded in exact stored fields + `_memory_summary` emits "(EMPTY — no entries saved)" so an empty map can't pass as memory=5. F5 NAME EXACTLY appended to characters/elements initial_population identification+population templates (8 templates, both schemas) — first-scene population was producing 'galvanised_watering_can' style renames.

### #126

*updated 2026-08-14 17:26:08*

dayna_ss soak harness runs judge/audit/rolling-overview on background worker threads (shipped 2026-08-14): per-turn judge and audit were on the turn critical path but are pure side effects, so SoakRun has `_judge_q`/`_audit_q` (queue.Queue) + one daemon worker thread per CloudModel instance (serialized per role — a CloudModel is never used by two threads). run_turn (L1+L2) appends task closures (snapshotting judge_window history + `_planted_note_ids()` at dispatch) and returns judge=None/audit=None; main() checkpoints result.json THEN calls `soak.flush_pending_tasks()` (worker merges would race the checkpoint write otherwise). Workers compute exact per-task usage deltas (base stats snapshot before the call; instance is single-threaded) and merge `{judge, judge_usage}`/`{audit, auditor_usage}` into the turn's result.json atomically (tmp+rename under `_cloud_lock` RLock). Abort early-stop moved into workers via `_note_abort` (sets `_stop`; loop notices next loop top, so stop is delayed up to ~1 turn). Rolling overviews run on the JUDGE worker (serialized after per-turn judges so judge_records is complete); final overview stays main-thread after `_join_cloud_workers()`. `soak_log` is lock-guarded; `_join_cloud_workers` is idempotent (sentinel + `_workers_joined`). The guide stays main-thread (its output is the next turn's input). Also fixed: latent `NameError: name 're' is not defined` in `_run_judge` (F4 instr-block extraction used re without import — judge path is only live-exercised).

### #128

*updated 2026-08-15 03:06:10*

dayna_ss / textgen MTP (speculative decoding) status (verified 2026-08-14): LMDeploy 0.15 supports Qwen3.5 MTP — `qwen3_5_mtp` proposer (pytorch/spec_decode/proposers/qwen3_5_mtp.py: class Qwen3_5MTP(DeepseekMTP), reuses the target model's embeddings via set_input_embeddings) + `Qwen3_5MTPModel` wrapper (pytorch/models/module_map.py:206 → qwen3_5_mtp.Qwen3_5MTPModel) — but ONLY in the **PyTorch engine**, NOT TurboMind (turbomind/ has no mtp mentions). Activation: `Pipeline(model, backend_config=PytorchEngineConfig(...), speculative_config=SpeculativeConfig(method='qwen3_5_mtp', model=<dir>, num_speculative_tokens=N))`; CLI equivalents `--speculative-algorithm qwen3_5_mtp` (cli/utils.py:796, choices eagle/eagle3/deepseek_mtp/hy3_mtp/qwen3_5_mtp) + `--speculative-num-draft-tokens` (default 1). Qwen3.8-27B genuinely ships MTP: text_config has `mtp_num_hidden_layers: 1` + `mtp_use_dedicated_embeddings: False`, 15 `mtp.*` tensors (1 decoder layer + fc + norms), stored separately in `model-mtp.safetensors`. The model is multimodal `Qwen3_5ForConditionalGeneration` (dense GDN hybrid, layer_types linear/full_attention every 4, 64 layers, head_dim 256); `PytorchEngineConfig(language_model_only=True)` (pytorch/config.py:629) skips the 333 vision tensors at build (BuildModelContext in pytorch/engine/model_agent/agent.py ~1190). GATES for V100: (1) base bf16 54GB won't fit 2x16GB; use W4A16 (~14GB, tp=2). (2) The downloaded Qwen3.8-27B-W4A16-AWQ is compressed-tensors `pack-quantized` — pytorch engine quant loader FAILS on it (`TypeError: Unsupported quant method: compressed-tensors`; supports only awq/smooth_quant/fp8); it WILL load on TurboMind (no MTP) instead. Classic-AWQ variant (Qwen3.8-27B-AWQ_ct) is the one for the pytorch/MTP path. (3) V100 sm_70 kernel support for the GDN TileLang kernels + Triton flash attn is UNVERIFIED — needs a live smoke. textgen's modules/lmdeploy.py has a `--backend pytorch` branch (lines 312-326, eager_mode=True) but does NOT pass speculative_config yet (it's a Pipeline-level arg, not in PytorchEngineConfig).

### #129

*updated 2026-08-18 05:14:41*

dayna_ss entry-selection action (shipped 2026-08-14, both schemas): new `Action.SELECT_ENTRIES_TO_UPDATE` filters per-entry updates. `DataSummarizer._select_entries_to_update` (from `_execute_action`) renders the map's `select_entries_to_update_prompt_template` (`{{ branch_list }}` compact roster + scene context), gets a JSON name array, resolves via `_resolve_dict_key`, stores `self._entry_whitelists[branch_name]`. `_traverse_structure` case 3a (dict-of-schema-classes loop) SKIPS non-whitelisted entries (no per-entry LLM call). Guards: fail-open on unparseable/`json_repair`-repaired-nested/garbage or missing template or map < `select_entries_min_size` (default 4) -> whitelist None = run all; valid `[]` = skip all; entries added by add_new this pass always run via `self._new_entry_names` union checked at loop time (add_new may fire after selection in trigger order). Keyed by map branch_name so relationship sub-dicts untouched. Per-instance dicts = parallel-worker-safe. Wiring: Type 1 selection on `always` after gate for CharacterMap/GroupMap/ElementMap; Type 2 on `on_new_scene` after add_new (+ ElementMap gains on_new_scene select while add_new stays always). CRITICAL: `_tolerant_json_loads` json_repair can repair garbage into nested lists — selection requires raw response to start with `[` AND parse as a flat string list.

### #130

*updated 2026-08-17 11:58:49*

dayna_ss spreadsheet voice matrix (2026-08-14): the 7 original spreadsheets all used 1st-person-guide / 3rd-person-DSS; now a 12-sheet voice/person matrix, cozy_mystery kept as the 1P/3P control. Combos: noir_detective = dual-1st-person (two 'I's); fantasy_epic + heist_caper = 2nd person both ('you' = name2 throughout); sci_fi_heist = 3rd-person story-master guide + 1st-person Dex; horror = same-perspective 1st (Merrick & Ellery one shared 'I', memory-erasure premise); romance + western = dual close-3rd; cyberpunk_thriller = 2nd-person handler addressing 'you' + 1st-person Juno; greek_mythology_retelling = 3rd muse + 2nd-person hero; postapocalyptic_survival = classic 1P/3P in present tense; gothic_haunted_mansion = 1st-person journal + 2nd-person ghost addressing the heir as 'you'. Voice lives entirely in writing_style.guide_directive/dss_directive (flows via general_info.writing_style, memory #46). To support this the scoring prompts became person-agnostic: judge/final-overview/synthetic-reply grade against {dss_directive} with 'wrong-person narration' anchors instead of hardcoded 'first-person' anchors; guide template now 'writes name1's turn' in whatever person the directive specifies (no more 'You play {name1}'); same-perspective identity-sharing is carved out of the judge's identity-confusion floor. Build scripts kept: tests/revoice_spreadsheets.py + tests/new_spreadsheets.py (regenerate the matrix).

### #131

*updated 2026-08-23 22:46:23*

dayna_ss prompt prefix-caching ordering principle (both schemas): DSS workload is ~99% prefill by tokens (128:1 prompt:completion), so TurboMind prefix caching is the dominant perf lever — any dynamic content before a large static block splits the shared prefix and kills cache reuse. Rule: STATIC template content first, dynamic descriptors last. Applied: (1) data_summarizer.py 5 code sites (gate check ~1905, branch query ~2068, branch update ~2133, ~2182, importance update ~2412) — the `Current context for '<name>': {mark_field(branch_name)}` whole-subject render block is appended AFTER the rendered template; (2) TR1 per-entry branch_query/branch_update: `'{{ branch_name }}'` stripped from the static body → single trailing `Entry to review: '{{ branch_name }}'` line; (3) TR2 new_entry_prompt: `named '{{ entry_name }}'` → trailing `The entry to populate:`; (4) TR3 new_entry_query + select_entries: `{{ branch_list }}` block moved to the very end; (5) TR4 population_prompt: FIXED the empty-name bug — Elements population rendered `generate the full data for the element named ''` because the template used `{{ element_name }}` while the population code (summarizer.py) only passes `entity_name`; now "described element" + trailing `Entry to populate: '{{ entity_name }}'` + `Descriptor:` at the tail; (6) TR5 CharacterMap importance_update value block to end. Also: mark_field does NOT isolate the entry (identifiers like `data.characters.Evelyn` never match the bare branch_name) — it returns the WHOLE subject render, which is IDENTICAL across a subject's per-entry calls, so post-swap it sits in the cacheable tail. Initial Greeting: production stores the char greeting in internal[0][1] (assistant slot, chat.py:1801); the harness's old opening line sat in [0][0] (user slot). The fix is harness-side: the engine's four `internal[0][1]` extractions (summarizer.py:1892, 2345, 2491, 2904) read the greeting directly because build_state now supplies a real spreadsheet `greeting` in the assistant slot — the `_first_pair[1] or _first_pair[0]` fallback was REVERTED (see #132).

### #132

*updated 2026-08-17 11:58:41*

dayna_ss Initial Greeting production-parity (shipped 2026-08-14): production reserves the greeting for name2's OWN opening line in the assistant slot of the first internal pair (chat.py:1801 `['<|BEGIN-VISIBLE-CHAT|>', greeting]`); the engine's population prompts render `Initial Greeting` and `First User Input` as distinct concepts. The short-lived `internal[0][1] or internal[0][0]` fallback was REVERTED (it put a name1 second-person address under the greeting label, risking voice/identity conflation — memory #46/#103). The real fix is harness-side simulation: new required spreadsheet field `greeting` (name2's opening in the dss voice; all 12 sheets carry one; validate_spreadsheet.py enforces it; new_spreadsheets.py GREETINGS map; revoice preserves it). build_state sets state["greeting"] + internal=[["<|BEGIN-VISIBLE-CHAT|>", greeting]] + state["user_bio"]=name1 description (world-cache key parity, summarizer.py:1894). L2 engine-view history = greeting pair + self.history (self.history stays pure [user,reply] — judge/guide/probes unaffected); summarize_latest_state receives the same greeting-inclusive history. _engine_internal_messages skips the marker → greeting renders as the first assistant message; format_dialogue renders it first. HASH BREAKAGE: the greeting is part of retrieve_history_path's history hash — a run started on old code resumed with new code changes the turn-dir hash mid-run → fresh empty world. In-flight old-code runs must finish or restart fresh; new runs are consistent.

### #133

*updated 2026-08-24 12:05:56*

dayna_ss add_new name-parse + arcs/chapters template fix (2026-08-14): the Arcs/Chapters new_entry_query_prompt_template used to demand a BARE title ("Respond with arc title...") while every other discovery template demands a JSON array — parser/format mismatch, entries never added. Fixed BOTH schemas: those two templates now demand a one-element JSON array (["Title"]), and the parser got `_extract_entry_names` (data_summarizer.py) — lenient: strict JSON -> json_repair -> double-quoted strings -> bare lines (bullets/brackets/quotes/trailing commas stripped, _NO_NEW_ENTRY_PREFIXES negative guard). select_entries_to_update deliberately stays strict-fail-open. Also: soak model choice — Qwen3.8-27B dense at tp=2 decodes ~1.34 tok/s under 5-way concurrency (~6s for a 16.6K-token call with only ~8 output tokens; a 3-call arcs branch = ~20s); Qwen3.6-35B-A3B (3B active) is ~9x cheaper per token for BOTH prefill+decode and is the established soak model (4/5, 5.0 style). Serve the soak on the 35B; CMD_FLAGS_m.txt already points at it.

### #137

*updated 2026-08-24 12:03:20*

dayna_ss reply/instruction repetition guard is ENGINE-NATIVE: a reply-text post-check must NOT live in the soak harness (tests/long_horizon_soak.py) because production reply text is generated OUTSIDE the extension by TGWUI modules/chat.py (extensions/dayna_ss/script.py custom_generate_chat_prompt returns a prompt; chat.py streams the reply). The engine-native lever is the INSTRUCTION BLOCK via generate_instr_prompt (now extensions/dayna_ss/agents/summarizer/instruction_generator.py). Implemented via pure helpers in extensions/dayna_ss/agents/instruction_blocks.py: _instructions_similar() (normalized difflib ratio, autojunk=False, threshold 0.85, min 80 chars) + _instructions_phrase_overlap (LCS >= 120 chars) + _INSTR_ANTI_REPEAT_DIRECTIVE; Summarizer._prev_instruction tracks the previous turn's block (init in summarizer/core.py, set on both cache-hit and generating paths); the guard is now a FULL-RING scan (_block_collides vs the recent-block ring, plus _ring_blocks folding prev in) checking byte-ratio, LCS, a 2-prop re-anchor cluster, and a persistent-prop signal, naming the offending props in the regen directive. After _clean_generated_instructions, a colliding block triggers up to _INSTR_REGEN_MAX=2 deterministic hard-reject regenerations keeping the least-colliding candidate. Stale-cleaning bug fixed: instr = cleaned_instr so the cleaner AND the guard reach the model on the generating turn (previously the cleaned block was cached but the raw block fed the reply prompt). The reply-side anti-repetition stays prompt-side only (existing hard rules).

### #138

*updated 2026-08-24 12:06:18*

dayna_ss new-entry schema-echo failure + retry fix (2026-08-15, engine data_summarizer.py _detect_and_add_new_entries_to_branch): the model sometimes reproduces the prompt's embedded JSONSchema instead of generating entry data — `{"main_schema": {"$ref": "#/definitions/Character"}, "definitions": {...}}` is literally get_relevant_json_schema_definitions output (Character schema = 9,779 bytes; example = 6,789 bytes, so ~16.5KB of JSON scaffolding precedes the generation boundary). Because the echo is VALID JSON it would have been stored as a garbage entry; truncation (max_tokens) was what made it a parse failure. Fix: (1) entry generation now retries 2x with parse-error feedback ("your attempt was rejected: ... do NOT reproduce the schema/example"); (2) `_is_schema_echo()` detects valid-JSON echoes (top-level main_schema/definitions/schema keys or any `$ref` value) and treats them as failures; (3) `_tolerant_json_loads` (json_repair) before giving up; (4) empty-dict guard — json_repair turns even 'not json {{' into {}, which would store an empty stub; (5) universal anti-echo directive appended to the entry-generation prompt. json_repair also silently 'completes' truncated JSON like `{"name": "Marmalade", ` — accepted (consistent with engine-wide tolerance). Verified: 9-case unit test + hermetic suite green.

### #139

*updated 2026-08-17 11:59:08*

dayna_ss memory-explosion bug (ROOT-CAUSED + FIXED 2026-08-15): DSS soak grew to 128GB and OOM'd. Root cause: `EntityGraph.traverse_graph_detailed` (rag/structured_rag/entity_graph.py:1560+) — a depth-bounded BFS that accumulates EVERY path in `PathRecord.paths` and re-combines the source's full path list on every neighbor. Cyclic graphs (characters mutually listed in each other's relationships) cause combinatorial path multiplication (avg branching^depth); with max_depth=10 on a 32-node/96-rel graph it allocated 45-113GB and spun at 100% CPU. Nondeterministic: hash-order-dependent, so it could pass once then explode on the next run. FIX: path caps — PATH_CAP_PER_RECORD=64 (keep top-64 paths per record by effective_imp, trim via `_trim_path_record`) + TOTAL_PATH_CAP=20000 (global hard stop). Scoring is EXACTLY preserved: `record.best_effective` is updated before the cap check and `source_ids` is a set, and both are the only scoring inputs; the cap only limits which paths survive for deeper extension. Verified: 10× retrieve_context iterations flat at 1.4GB (was 45GB+ climbing), hermetic suite green. Watch: the soak loads modules at init — a running soak needs restart to pick this up.

### #140

*updated 2026-08-15 17:37:11*

dayna_ss arcs/chapters null-ending crash (FIXED 2026-08-15): `_create_update_prompt` (data_summarizer.py:3015-3021) crashed with `TypeError: unsupported operand type(s) for -: 'int' and 'NoneType'` whenever an arc dict had `ending_chapter: None` (a freshly added, unconcluded arc legitimately stores null) or a chapter had `ending_scene: None`. `.get(key, 0)` only guards a MISSING key — present-but-null flows through and crashes arithmetic. Because generate() swallows exceptions, the arcs subject silently failed to update EVERY turn. Fix: added module-level `_safe_int(value, default=0)` (bool→default, int passthrough, str coerced, None/junk→default) and used it at all three read sites; `last_arc` selection now filters to dicts and uses `max(..., default=None)`. Semantics: an unconcluded arc/chapter (null ending) correctly means all chapters/scenes so far are in it. Verified: 6-case unit test incl. the exact crash + hermetic suite green.

### #141

*updated 2026-08-23 22:46:24*

dayna_ss story-continuation contamination in meta-templates (FIXED 2026-08-15): sceneagg (Type 2) gate_check/new_entry_query templates end with the SCENE CONTEXT block ({{ scene_recap }} = recent exchanges rendered as prose, {{ scene_events }}), so the generation boundary is raw story prose → the model "continues the story" instead of answering the meta-question. Observed: arcs add_new query returned the DSS reply verbatim ("Evelyn weighed the amber jar...") → "Failed to extract new entry names"; and `_perform_gate_check` is FAIL-OPEN (only a clean NO/UNCHANGED skips; anything else, incl. prose, passes) so the prose gate response let add_new run. FIX 1 (schema, 22 templates in subjects_schema_sceneagg.json): appended a universal tail to every SCENE CONTEXT template — "This is a MEMORY-MANAGEMENT TASK, not a story continuation. Do NOT write fiction. Re-read the instruction at the top..." — plus kind-specific closers (gate_check: "Respond ONLY with: YES or NO"; new_entry_query: "...JSON array... or exactly NO."; select_entries_to_update: same). FIX 2 (engine, data_summarizer.py `_perform_gate_check`): bounded strict-verdict retry — if the response is neither negative (NO/UNCHANGED prefix) nor affirmative (YES prefix) by stop_reason OR text, it retries ONCE with a strict re-frame, then fails open (preserving the A2 no-freeze property). Verified: 8-case verdict unit test + hermetic green + schema parses. Both fixes need a soak restart (code at init, per-session schema copies refreshed at startup).

### #142

*updated 2026-08-24 12:06:12*

dayna_ss instruction-loop batch (2026-08-15, C1/P1a/P1b/P2a/P2b/P4): P2b semantic anti-repeat guard: naive keyword-overlap fires EVERY turn in long runs (story vocab always overlaps — measured 39/39 on 14714d8e); rare-word overlap and consecutive-block LCS≥100 also over-fire (healthy dded7733 = 3/29). The signal that separates loop from healthy is LONGEST COMMON SUBSTRING ≥120 chars of verbatim text (loop fired 13/39 with the near-copy at LCS=870; healthy 3/29 borderline). Final detector `_instructions_phrase_overlap` (summarizer.py) = normalized-LCS ≥120, wired alongside the 0.85 byte-ratio `_instructions_similar` into both the regenerate trigger and retry-acceptance in generate_instr_prompt. P2a = instr item 14 STRICT IMAGERY RULE. P1a = `_current_scene_recap` prepends live scene (what/when/where/present) to the instruction-GENERATION prompt (NOT the reply instr_prompt — that's why it's invisible in instr_prompt.txt). P1b = CurrentScene.now templates now say the clock MUST advance on every exchange — ALWAYS write updated now.when.specific_time, NEVER identical to previous turn; both schemas. P4: CloudModel.json_complete got a THIRD retry that feeds the parse error back to the model; report.py `_attribute` now checks whether the note's needle appeared in the guide's story text (`story_text` = joined user_inputs) — story-planted-but-never-stored = dss_retention_loss; never-in-story = guide_failure; C1 = turn dirs now include context.txt (the full system context) + dashboard state-file viewer handles it via `_load_state_file`.

### #143

*updated 2026-08-24 12:06:15*

dayna_ss DSS LLM prompts (data_summarizer.py) are wrapped by modules/chat.generate_chat_prompt, so every call = [system context + full history] + DSS prompt; within one turn the history prefix is byte-identical, so prefix-cache reuse = history (~17-18K tok) + DSS static head. Two systemic facts: (1) FormattedData renders format templates with {{path}} = "" (no prefix passed in summarizer.py:3238), so mark_field identifiers look like ".characters.Evelyn", which NEVER match branch_name strings like "characters.entries.Evelyn" -> mark_field returns the FULL subject map with markers stripped for every per-entry call (data_summarizer.py:2540, 2003, 2197, 2262). (2) Schema snippet + example JSON blocks are deterministic per schema class and embedded in every call via _create_update_prompt (data_summarizer.py:3026-3029); reordering templates to put dynamic placeholders (branch_name, item_name, value) AFTER schema/example + recap maximizes TurboMind prefix reuse.

### #144

*updated 2026-08-23 22:46:09*

dayna_ss prefix-cache efficiency (2026-08-15): LocalModel._complete logs every call (`local-call <phase>/<step>: ttft=X.XXs in=PT out=CT tok`) — matches the LMDeploy server log via input_tokens. Labels threaded: generate_with_sse already had phase_id/step_id; generate_using_tgwui gained them at all 6 DataSummarizer call sites (add_new_query, add_new, chapters/arcs check_archive+archive); the engine Summarizer.generate_using_tgwui POPS phase_id/step_id from kwargs before forwarding (else double-bind TypeError in production). Applied static-first/dynamic-last reorders to 22 templates in BOTH schemas (gate checks Character/Group/Element/Events, select_entries ×3, SceneState/SceneWhy/StoryEvents.update, StoryEvents.importance_update, Arc/Chapter.update): dynamic reference moved to a single trailing line ("Section to review: '{{ branch_name }}'" etc.), static body+schema+example first. Measured shared-prefix across two different-branch calls: default gate_check 47→335 ch (98%), sceneagg gate_check 47→780 (99%), select_entries 60→1260 (99%), new_entry_query 275→1508 (87%). DEFERRED (change what model sees): (A) mark_field never matches format-marker identifiers (`.characters.Evelyn` vs `characters.entries.Evelyn`, summarizer.py:3441) so per-entry calls embed the WHOLE subject map — biggest transition-turn byte leak; (B) hoist schema_snippet/example_json/scene recap into the per-turn system context. ALREADY-OPTIMAL (don't touch): per-entry branch_update templates (24K/9.3K shared), new_entry generation (17.7K), population (18.3K), now_query/now_update.

### #145

*updated 2026-08-17 17:27:26*

dayna_ss per-entry whole-subject context re-statement gating (2026-08-15, FIXED O(map) per-call leak): every per-entry DataSummarizer call re-stated the WHOLE rendered subject map at the boundary (`\nCurrent context for 'X':\n {mark_field(...)}`) at 5 sites in data_summarizer.py (gate check, branch query ×2, branch_update_only, _generate_field_update). mark_field does NOT isolate entries — it toggles a marker annotation only; all non-target lines keep their content. The tail sat after the per-call entry ref (never prefix-cached), was byte-different per call, grew with the story (elements map 32KB→143KB by t15 in seed-89), and pressured the context window forcing history truncation (54856 vs 32768). Fix: new DataSummarizer._context_restatement(formatted_data, branch_name, keys, marker_paths) — config restate_map_context ∈ {auto(default), always, never}, restate_map_threshold_chars (default 10000): auto keeps the full map while under threshold (cheap accentuation; boundary target accentuation already served by `{{ value }}`), else drops it for a COMPACT SIBLING ROSTER (`Other entries in this section:` names from keys[:-1], auto-mode only); never = "". The full map stays in the shared prefix (system context/internal-history retrieval) — encoded once per turn, prefix-cached. Harness: --restate-map-context + --restate-map-threshold folded into the run-id hash + run_tests.py passthrough (all verified). Full map also still rendered in the retrieval context that every call shares, so the model loses nothing except boundary-near full profiles of OTHER entries.

### #146

*updated 2026-08-17 11:58:49*

dayna_ss soak repetition loop (b884af4a, 40-turn Type 2 Mode B, 3.5/5): the "snip lavender / grey ceramic jug" tic. Three converging causes, NONE caught by the F5 byte-ratio guard (0.85) or P2b LCS guard (120): (1) instruction generator CANNOT satisfy req 9 ("weave in stored items") + req 14 ("never re-anchor on stored objects") because it has NO used-imagery memory — nothing enumerates which stored objects prior replies used, so it picks the most positionally salient ones: "dried lavender" is elements.json entry #1 in EVERY turn (dict insertion order; frozen imp 50 == full-profile threshold), jug imp 60. 17/20 instruction blocks (t11-30) carry the lavender/jug beat; consecutive blocks LCS only 16-85 (fired once at t28). (2) Reply transcribes the instruction block 100% (13/13 instructed beats performed) AND self-anchors on its own prior sentences — rolling mode puts DSS's own last 6 replies as assistant messages right above the generation boundary; t18 vs t16 reply ratio 0.79 (judge flagged "verbatim repetition"); t26 self-injected the gesture with no instruction beat. (3) Storage is passive, not strengthening: lavender/jug descriptions frozen all 40 turns, importance pinned 50-60; salience is purely positional. P1a scene recap was CORRECT (vestry→post-office→ruins) — not the cause. Fix design (approved-draft): P1 inject "RECENTLY USED IMAGERY/PROPS" ring (last ~8 blocks' props via elements-vocab match + imperative regex) into the instruction-GENERATION prompt after the scene recap + amend req 14 to reference it; P2 semantic prop-overlap post-check (regen naming offending props — turn 28 proved regen without names can't de-loop); R1 append DSS's own last 2-3 replies into the reply prompt as negative exemplars; S1-3 storage hygiene (dedupe within-batch + token-set containment + description-similarity merge; importance decay; render-order shuffle) + latent bug format_templates.json:57 reads stale `defaults.CharacterMap.importance_detail_threshold` (threshold hardcoded 50).

### #147

*updated 2026-08-18 05:14:37*

dayna_ss soak dashboard + harness: "history (unbound)" stat. The harness records per turn `full_history_tokens` (and `full_history_chars`) in result.json local_usage = sum(len(u)+len(r) for u,r in soak.history) // 4 — the FULL accumulated conversation's estimated token size if NOT bounded by --max-update-history, matching the existing //4 chars->tokens convention (lmdeploy v0.15 returns prompt_tokens=0 so _complete falls back to chars//4). Computed at the main() checkpoint site (tests/long_horizon_soak.py ~2535-2541; self.history already holds the turn's pair by then, both L1 and L2 append before returning). soak_dashboard.py `_build_turn_accordion` (~640-648) renders `· **history (unbound)** {tokens}/{ctx} ({pct}%)` beside `**peak ctx**`, and `—` when the field is absent (old runs). `_live_stats_html` local card (~798) appends `· unbound history ~N tok` for the last turn. Expected curve: flat early -> medium after a few turns -> high after ~20 -> ideally leveling off (logarithmic). Old runs predating the field show "—".

### #149

*updated 2026-08-24 12:05:57*

dayna_ss sticky-roll message summaries (engine upgrade, 2026-08-16): rolling mode (`message_mode=rolling`) drops out-of-window messages entirely, so `retrieve_and_format_context` (agents/summarizer.py) now injects a "Summaries of earlier messages (messages S-E, before the most recent window)" pair ahead of the raw rolling pairs when config `rolling_summaries: N` > 0. Source = the accumulated `message_index` llama_index docstore (`is_summary` nodes); empirically each new history dir's message_index holds the FULL accumulated summary set (whole-index persist + retriever instance reuse, verified on the real 40-turn b884af4a store, idx 2..81), so a fresh chunker load at retrieval sees the whole run. Helpers: `_collect_scene_boundaries` (archived events.scenes[*].start._message_node + current scene start), `_rolling_summary_window` (start = min(2*(len-20), last-3-scenes boundary) = the FURTHER boundary = max span; end = 2*(len(history)-last_x) so no duplication with the raw window; (0,0) when nothing), `_load_summary_nodes` (dedup by message_idx, last wins), `_format_rolling_summaries` ("- [message N] text"). message_idx maps production: greeting = idx 0/1, turn N user = 2N+2. Harness: `--rolling-summaries N` (default 0) folded into the run-id hash + run_tests.py passthrough. Doc'd in schema_types_and_test_matrix.md §3 + long_horizon_soak_plan.md §24. Open design (a-vs-b): mentioned-entities per message — recommendation is (b) embed a "Mentioned in this message:" comma-list tail in the summary text (zero extra LLM calls; extend subjects_referenced parse to elements/props), revisit separate-node (a) only if a structured consumer needs it.

### #150

*updated 2026-08-24 12:05:57*

dayna_ss rolling message-summaries sticky lock (2026-08-16, engine): the summaries block from `rolling_summaries` config is byte-stable WITHIN a scene so the shared prefix survives for prefix caching. `_current_scene_key` (summarizer.py) identifies the scene by `current_scene.start._message_node`, else `_scene_number`; `_rolling_summary_window(history_len, last_x, scene_bounds, floor, scene_key)` locks `_locked_summary_start`/`_locked_summary_scene` on the Summarizer instance — same scene = same start (end still advances with history), scene change = recompute. Scene roll changed from 3 to `ROLLING_SUMMARY_SCENE_ROLL = 5` scenes (`Math.max(floor, last_5_scenes)`; fewer than 5 scenes → all count). The Summarizer instance persists across turns in both production and the soak harness (self.summarizer, long_horizon_soak.py:1754). IMPORTANT bound interplay: the harness `LocalModel._engine_internal_messages` (long_horizon_soak.py:310-323) now PRESERVES the "Summaries of earlier messages" pair under the `--max-update-history` bound and bounds only the raw rolling pairs behind it — before this, the pair (injected at the head of the dialogue section) was trimmed on DataSummarizer per-entry calls, so the feature only served REPLY calls, never the per-entry calls where prefill is the real cost. In-memory only: a process restart recomputes, which is fine because the prefix cache dies with it. Verified: 27-case unit test (/tmp/opencode/test_rolling_summaries.py), 3-case bound test, hermetic suite green.

### #151

*updated 2026-08-16 02:38:45*

dayna_ss message-node provenance (FIXED 2026-08-16): DataSummarizer's `current_message_node` (_message_node written on new entries) was derived from `len(custom_state['history']['internal']) * 2` — the ARTIFICIAL retrieval-context list, bounded to the rolling window (~6-8 pairs) + the "What was the very last exchange?" pair, so message nodes capped at ~16 forever (confirmed in cozy_mystery__b884af4a: every scene/event node frozen at 16_1_1 from turn 11 on). Fix: DataSummarizer now takes `real_history` (the same list summarize_latest_state receives — greeting pair + exchanges INCLUDING the one being summarized); `_resolve_current_message_node()` returns `len(real_history) * 2` (the engine's canonical current-message index, matching new_scene_start_node and _scene_start_message), falling back to the old internal-derived formula only for construction sites without real_history. real_history is threaded through summarize_latest_state's serial + parallel (_process_subject_parallel) paths and the harness L1 path. The other sibling bug: `_scene_recap_text` (sceneagg Type-2 scene recap) read the artificial internal list, which in enumerated mode is laced with retrieval Q&A pairs; it now renders real dialogue (name1/name2 per real pair, greeting marker skipped) when real_history is set. Harness-side: build_state was handing the engine a greeting-ONLY `state['history']['internal']` (production TGWUI = full internal history), which silently killed generate_instr_prompt's R1 negative-exemplar block (summarizer.py:1083) — build_state now carries greeting pair at [0] + every exchange, updated to include the current exchange before summarize_latest_state. The harness never truncated engine-visible history (`--max-update-history` only bounds the LocalModel transport payload via a NEW list in `_engine_internal_messages`); the ~16 cap was the engine's artificial-internal derivation, not harness truncation.

### #155

*updated 2026-08-17 11:47:47*

dayna_ss container-shape repair (coerce_container_types, utils/helpers.py, 2026-08-17): small models (Qwen3.6-35B) began writing `[]` where the schema declares `dict[str, ...]` — e.g. Character.relationships / group_status (73 rel-list entries in cozy_mystery__3fbf1a99, 0 in every prior run). The stored list then made name-keyed update paths (`relationships.Glove.owner.status`) crash in recursive_set with "Cannot use non-integer key 'Glove' on a list" and silently drop the update. Fix: `coerce_container_types(data, schema_type, parser)` — a schema-driven in-place traversal mirroring unexpand_lists_in_data_from_llm that coerces dict-declared fields written as lists (empty -> {}, populated -> keyed by name-ish element field via _infer_collection_key when unambiguous, e.g. group_status; CharacterRelationship objects have no name field so real rel lists are LEFT untouched) and int-keyed dicts under list-declared fields back to lists. Wired at the top of DataSummarizer.generate() (data_summarizer.py:713) BEFORE FormattedData construction and _update_recursive, so update resolution AND the save path both see schema-consistent containers (self-heals across turns; root object identity preserved so all_subjects_data/processed_subjects_data aliasing and post-loop chapter/arc checks stay consistent). The engine NEVER coerced container shapes before — unexpand_lists_in_data_from_llm was shape-preserving. Residual (intentional): when the model ALSO writes a spurious nesting level (relationships.Glove.owner.<sub> instead of relationships.Glove.0.<sub>), the update now lands as a nested dict under Glove (data preserved, gracefully skipped in rendering) rather than crashing — prompt-level teaching of the `Name.0.field` addressing form is a possible follow-up.

### #159

*updated 2026-08-24 12:06:03*

dayna_ss harness local-call latency metrics (2026-08-17): the soak's `ttft=` log line was previously the FULL round-trip (synchronous non-streaming POST via _complete, timed request-issue → full body read) — never real time-to-first-token; all "streaming"-named LocalModel methods delegate to _complete. Now `_complete` sends `"stream": True` + `stream_options.include_usage` and parses SSE in `_stream_post`: ttft = wall time to the FIRST delta carrying content/reasoning, total = time to [DONE], pt/ct from the include_usage chunk (falls back to payload-length estimate when v0.15 returns prompt_tokens=0). The 3x empty-completion retry is preserved. TGWUI's `/v1/chat/completions` (modules/api/completions.py chat_completions_common) fully supports this. Caveat: under concurrent --max-subject-workers, client-side ttft includes server queue wait, not just prefill. Log format: `local-call <phase>/<step>: ttft=X total=Y in=PT out=CT tok (pt/total K tok/s)`.

### #161

*updated 2026-08-17 14:33:57*

The dayna_ss retrieval stack per-turn path: StoryContextRetriever.retrieve_context (context_retriever.py:977) builds RetrievalContext from current_scene.json + entity graph traversal; elements are picked by regex name-in-context (_get_relevant_elements, context_retriever.py:257) → ~all elements served; events come only from graph traversal over character→event milestones edges (never real events.json keys) → events block is always empty; RAG messages = query_messages(context, n_results=5) pure similarity over MessageChunker llama-index sentence chunks (query_similar context_retriever.py:1589), scene_id/event_id metadata never filtered on; index-based _get_message_chunks is commented out (context_retriever.py:1090).

### #165

*updated 2026-08-24 12:06:18*

dayna_ss repetition-guard hardening (2026-08-17): the ring deque `_recent_instr_blocks` is now 3-tuples (label, prop-set, TEXT). `_instructions_similar` uses autojunk=False (default deflated re-lexicalized prose to ~0.75 so the 0.85 byte-ratio never fired). New `_block_collides(block, ring, known_names)` scans the WHOLE ring: byte/LCS vs any block text, a 2-prop re-anchor CLUSTER vs any single block (not the union), and a PERSISTENT-prop signal (a prop in >=3 blocks reappears — catches the single-prop knife loop). Regen names actual offending props; hard-reject acceptance keeps the least-colliding of up to _INSTR_REGEN_MAX=2 regenerations. `_ring_blocks` folds prev instruction in explicitly. `_current_scene_recap` prefers live `now.what` over lagging top-level `what`. Harness-side reply post-check `_generate_reply` (long_horizon_soak.py): transcription-ratio >=0.70 vs steering prompt, self-anchor >=0.80 vs previous reply (>=200 chars), one bounded regen with `_REPLY_ANTI_REPEAT_DIRECTIVE`; reply text is TGWUI-owned in production so this is necessarily harness-only.

### #166

*updated 2026-08-17 15:30:38*

dayna_ss retriever read-side repair (2026-08-17): (1) EVENT RENDER WAS BROKEN AT THE SHAPE LEVEL — retrieve_context returned events as {"entries": {...}} but the events format template iterates data.past/scenes/events/chapters, so the block rendered ZERO sections even when events.json was full, and the graph-aggregation path never called _get_relevant_events. New `_select_relevant_events` (context_retriever.py): name/alias match + current-scene events (start._message_node >= scene start node) + recency fill, max 8, returns the FULL category shape. RetrievalContext gained `events_full` (uncapped bucket view) so boundaries/_collect_scene_boundaries/has_archived_scenes/scene_names-map keep every archived scene; retrieve_context sets result.events_full and summarizer.py reads it at all four sites (summarizer.py:1499,2052,2134 + _collect_scene_boundaries). (2) RAG recency: query_messages fetches max(n_results*6,30) candidates and re-ranks by band (within _RAG_RECENCY_MESSAGES=15 of newest first), summary-node preference, similarity-order tiebreak, dedupe by message_idx — stale beats no longer dominate. (3) Per-entity Recent-state: _recent_state_map (newest message-summary mention, events fallback) attached to copied entry dicts as _recent_state; format_templates.json characters/elements/groups gained an optional `Recent state -- {{d._recent_state}}` line before the {% else %} roster branch.

### #167

*updated 2026-08-23 22:46:09*

dayna_ss prefix-cache schema reorder (2026-08-18): drafts live in user_data/example/schemas/subjects_schema_sceneagg.prefixcache.json (review copy; live file unchanged). Generator re-runnable: tools/reorder_prompts_for_prefix_cache.py (block-level extraction, token-parity 40/40, no new jinja render warnings). Rules: static LEAD ("This is a MEMORY-MANAGEMENT TASK.../Re-read the instruction below...") first; schema/example slabs before per-branch guidance; apertures FULLY generic ("...any new entries..."/"generate the full data for the requested new entry...") with subject clarified ONLY near the bottom via "Subject being populated: <label>." + each branch's own guidance; "{{ branch_name }}" lifted out of apertures into trailing "List to update:'<branch>'" line; recap/events/scene dynamic values absolute last; IDENTITY BINDING near bottom (Character family). Gains: discovery earliest-var 196→1057, family shared 147→576; new_entry family shared 0→332 (incl Scenes); gate-check map trio 63→535; branch_update 19→176. FLAGGED ENGINE NOTE: _build_subject_routing_guide (data_summarizer.py) PREPENDS a subject-routing block to every add_new query that diverges per-subject at the "<<< YOU ARE NOW POPULATING THIS SUBJECT >>>" marker inside line 2; moving the marker to a trailing line would make the whole routing block a shared prefix.

### #168

*updated 2026-08-18 01:37:51*

dayna_ss soak cloud resilience (2026-08-17 overnight-robustness pass): CloudModel.complete() now re-rolls empty-content draws up to max_empty_retries (default 3) with empty_cooldown (default 300s) sleeps between re-rolls — reasoning-only draws (deepseek-v4-flash emits 2-8K reasoning at every effort level) are time-correlated so a cooldown lets the endpoint recover instead of aborting the run. Budget tiering unchanged (small budgets doubled once, large +2048 once; re-rolls then stay at the bumped value). guide_turn_with_retry defaults max_attempts 3->5 and sleeps the cloud-empty-cooldown between empty/truncated draws but NOT between verbatim-quote failures (transient model quirk, retry immediately). New harness flags: --guide-max-attempts, --cloud-empty-retries, --cloud-empty-cooldown — wired into the three CloudModel constructions and run_tests.py passthrough. These robustness knobs are deliberately NOT folded into the run-id hash (sampling params are, but these never change what the model sees on success), so a resume picks them up without changing the run dir. Worst case for a truly dead endpoint: guide aborts after 5 attempts x (3 cooldown re-rolls + outer cooldown) — much later than the old 3x-immediate abort, still protecting DSS prompts.

### #172

*updated 2026-08-24 12:06:30*

Initial population of the dayna_ss world model should be schema-driven rather than hardcoded: the schema should declare which subjects to populate and provide a base_prompt_template for each, driving the entire _populate_..._llm method family, and the validation-retry loop used when parsing LLM output should exist in data_summarizer.py as well as in the population code.

### #196

*updated 2026-08-23 22:45:11*

dayna_ss importance system is a UNIVERSAL conjoined rubric applied across all entity types (characters/groups/elements/events). The canonical scale lives ONLY in format_templates.json `importance_scale` entry (first, sole to_context:true/no_prompt:true item in `_context_order`, injected into system context every turn so the model sees it during reply + summarization). It blends SALIENCE (how often the subject's thoughts/actions are shaped by this) with WEIGHT (how central/irreplaceable/defining), each level 0-100 linear (100 Obsessive, 90 Paramount, 80 Profound, 75 Strong, 70 Significant, 60 Notable, 50 Meaningful, 40 Moderate, 30 Mild, 20 Slight, 10 Negligible, 0 None) with Positive/Negative faction flavor per level; a high score requires BOTH axes (constant presence alone, a scene-setpiece, caps near 50), measures LONG-TERM standing not a single scene's urgency. The schema (subjects_schema_sceneagg.json) IMPORTANCE lines / importance_placeholders are SHORT POINTERS to this in-context scale (not duplicated scales); Element new_entry + placeholder carry a setpiece guardrail. The schema file is NOT git-tracked (it needs json round-trip edits; ensure_ascii=False).

### #199

*updated 2026-08-23 22:46:26*

dayna_ss importance values live only on sub-parts nested under a parent entry (relationships, group memberships, event participant roles), never on the parent itself. A user request to query/update importance therefore targets the parent (grandparent) entry to collect changes across all its importance-bearing children in one pass, not the immediate parent field; give the LLM the complete list of that parent's importance-bearing children including score and reason. Run the lightweight importance query every message, and defer the full importance perform_update to a new-scene pass after gate checks. Importance is keyed per relationship type, so the same pair of characters can hold several distinct importance entries toward each other simultaneously.

### #200

*updated 2026-08-23 22:46:12*

dayna_ss current_scene.start is written ONLY by first-scene population and would otherwise stay frozen forever: SceneStart carries no_update:true in BOTH schemas (enforced at data_summarizer.py `_process_field`), and CurrentScene's only trigger was an `always` branch query targeting now.*/what only. Fix in both subjects_schema.json and subjects_schema_sceneagg.json: CurrentScene.defaults gains scene_start_query_prompt_template + scene_start_update_prompt_template (rewrite what+start.*+reset now.* from the transition exchange, stamp start.when._message_node={{current_message_node}}; bq+bu templates BOTH required — data_summarizer.py:646 gates on both even with skip_query), and triggers gain on_new_scene: [{action: query_branch_for_changes, prompt_template: scene_start_update_prompt_template, skip_query: true}]. perform_update CANNOT use a custom template name (it reads only key 'update_prompt_template'); path-based branch updates bypass no_update (only field traversal checks it). Engine defensive stamp in summarize_latest_state's scene handler writes start.when._message_node = f"{max(0, len(history)*2 - 2)}_1_1" so the scene-bounded dialogue window + rolling-summaries scene key always resolve even on NO_UPDATES_REQUIRED.

### #201

*updated 2026-08-23 22:45:06*

dayna_ss soak recall metric can report 0% spuriously: the per-turn recall probe fires only when recall.due == turn, and report.py counts every note's recall block (defaulting echoed=False), so notes scheduled beyond the run's turn count (e.g. due 62-78 in a 20-turn run) never probe yet still count as echo/recall failures. Fixes (shipped): SoakRun._clamp_note_windows (called in __init__) clamps every note's recall.due and plant.turn to args.turns - 2 so short runs actually fire recall probes; report.py `_recalls` adds a `reached` flag (due < len(results)) and the echo/recall_success denominator counts only reached recalls, rendered as a 'reached' column in the report table. Diagnostic rule: when a run shows 0% recall, first check recall.due values against turns.

### #203

*updated 2026-08-24 12:05:57*

soak_dashboard.py (Gradio 4.37): run selection uses @gr.render(inputs=[run_menu]) so switching runs re-invokes render_run/_run_display (the older visibility-toggle kept showing the first run's content forever); config API won't list these dynamic components — verify via the /run/predict fn_index of the 'apply' dep returning a render_config tree. A hidden gr.Button(elem_id='soak-live-refresh') clicked by demo.load(None, js=setInterval) refreshes live stats every 10s; ALL live updates use show_progress='hidden' so no spinner overlay. Live stat cards are consistent local/cloud: model name is the card VALUE; the sub line reads 'max ctx N · last turn ~X tok sent · <loader/roles> · HH:MM:SS' (local ctx from CMD_FLAGS_m.txt --n_ctx; cloud per-role ctx from MODEL_INFO). _live_stats_html builds cards: local server status (probe {base}/models + internal/model/info with the API key; ctx from --n_ctx), cloud models from manifest args with ctx + USD/MTok from the hardcoded MODEL_INFO dict (deepseek-v4-flash 1048576 no stated cost, glm-5.2 1048576 $1.2/$4.2, gpt-5.6-luna 1050000 $1.1/$6.6 ...), run progress (turns from manifest last_turn), cloud usage (calls/total_tokens/~est cost), GPU load via nvidia-smi. Timestamps (HH:MM) shown on retention tables, per-note rows ('turn@time'), judge curves, recalls, per-turn accordion headers ('at'), final overview, live cards, and stat cards; 'Local ctx (prompt tok)' / 'Cloud ctx (prompt tok)' cards show per-turn prompt-token sums. Helpers: _local_api_key, _local_ctx_from_flags, _local_server_status, _gpu_stats, _est_cost, _fmt_tokens.

### #206

*updated 2026-08-23 23:30:48*

dayna_ss utils/helpers.py is now a thin re-export shim; the implementations live in five theme modules: utils/console.py (colors, TypedKey, History/Histories), utils/json_utils.py (load/save/validate_path), utils/path_utils.py (recursive_get/set, split_keys_to_list), utils/data_expansion.py (expand/coerce/unexpand lists — imports schema_parser lazily), utils/formatting.py (format_str_or_jinja etc., owns the shared _jinja_env), utils/text_parsing.py (strip_json/thinking/response, extract_meaningful_paragraphs). All 11 consumers still import from helpers. Verified: hermetic test suite green + behavior parity vs pre-refactor file. Do NOT split or delete utils/memory_management.py — user plans to use it in the future (may be replaced by native platform options for textgen vs SillyTavern); it is currently unreferenced but intentionally kept.

### #207

*updated 2026-08-23 23:36:31*

dayna_ss T2 refactor (2026-08-23): DecayConfig moved to neutral module rag/structured_rag/decay_config.py — entity_graph's two lazy imports now use `.decay_config`, so the entity_graph↔context_retriever circular import is structurally GONE (entity_graph has zero context_retriever references). MessageChunker split into rag/structured_rag/message_chunker.py (~710 lines, owns the nltk/spacy/HuggingFaceEmbedding/Settings background-import globals and class-level singletons _embed_model/_nlp); context_retriever.py re-exports it at its bottom (`from .message_chunker import MessageChunker`) because agents/summarizer.py imports it from that path. context_retriever.py is now 1392 lines. Hermetic suite green after both moves.

### #208

*updated 2026-08-23 23:56:58*

dayna_ss T3 refactor (2026-08-23): (1) instruction-block engine extracted from agents/summarizer.py into agents/instruction_blocks.py (321 lines: _SENT_SPLIT_RE, _INSTR_SIM_THRESHOLD/_INSTR_RING_MAXLEN/_INSTR_ANTI_REPEAT_DIRECTIVE, _instructions_similar/_phrase_overlap/_block_collides/_ring_blocks/_extract_props/_recently_used_block/_collect_known_names/_own_replies_block/_current_scene_recap); summarizer imports them at the old location. Also fixed a latent invalid-escape bug there ("\n\Do NOT" -> "\n\nDo NOT" — prompts previously carried a stray backslash). (2) summarize_latest_state (was 522 lines) split into 9 Summarizer helpers: _ensure_summarization_phases, _load_all_subjects_data (returns data+missing_schemas; abort stays in orchestrator), _auto_detect_scene_transition, _stamp_new_scene_meta, _maybe_populate_first_scene (returns possibly-reloaded all_subjects_data), _copy_static_session_files, _process_all_subjects (returns (data, stopped) tuple; serial+parallel branches verbatim), _run_boundary_checks (returns stopped bool), _update_chunk_scene_event_metadata. Orchestrator keeps phase bookkeeping + stop handling + message-summary block verbatim. summarizer.py now 3686 lines; hermetic suite green incl. 40-call soak; script.py imports clean.

### #209

*updated 2026-08-24 00:30:59*

dayna_ss T4 refactor (2026-08-24) complete: summarizer.py and data_summarizer.py are now PACKAGES whose __init__.py re-exports the public classes so all consumer imports are unchanged. agents/summarizer/: core.py (1051; Summarizer(LLMClientMixin, InstructionGeneratorMixin, ContextEngineMixin, PopulationMixin) + SummarizationContextCache/_LockedPhaseManager/_build_worker_all_subjects/DualStream(dead)/save_message_chunks/summarize_latest_state family/backtrack_history/format_dialogue), llm_client.py (413), instruction_generator.py (513, + do_enc), context_engine.py (813), population.py (738). agents/data_summarizer/: core.py (728; DataSummarizer(DSPromptsMixin, DSDiscoveryMixin, DSUpdatesMixin, DSTraversalMixin, DSArchivesMixin); keeps generate/_execute_action/triggers/fuzzy-resolution/_inherit_defaults_from_parent), prompts.py (521, owns RESTATE_MAP_THRESHOLD_CHARS copy), discovery.py (371, ADD_NEW_MAX_TOTAL_ENTRIES copy), updates.py (729), traversal.py (453), archives.py (469), parsing.py (347; the pure free functions + defaults_to_inherit constant). Shared flat modules: agents/formatted_data.py (392; FormattedData + MessageSummarizer; imports runtime/helpers/MessageChunker at runtime, Summarizer only as lazy forward-ref), agents/instruction_blocks.py unchanged. Mixin modules access everything via self; forward refs to core classes use `if False:` imports (annotations are lazy). Gotchas hit: module-to-package swaps need __init__ re-exports incl. helper names tests import from the old module path (strip_thinking/strip_response from agents.summarizer); relative-import depth bumps by one inside packages (agents/summarizer/__init__ needs ...utils not ..utils); shared module-level constants used across split modules must move to a leaf (parsing.py) to avoid cycles. Full hermetic suite green.

### #210

*updated 2026-08-24 02:13:37*

dayna_ss/docs/ has three companion artifacts (created 2026-08-24): architecture.html (DSS internal architecture, same visual language as Smart-Memory's doc), comparison.html (SM v1.8.1 vs DSS feature matrix — verdicts: should-have = epistemics/perspectives subject, canon synopsis at chapter archival, gate-time continuity probe, resolved-state export; nice-to-have = discovery similarity guard, ledger field packs, dry-run turns+context telemetry over SSE, away recap/entity timeline/graph canvas; doesn't-need = profile snapshots, confidence decay, ST macros, unified-injection slots, migration framework, group chat), and performance_estimator.py (standalone CLI: --speed {laptop-8b,desktop-35b-moe,desktop-70b,cloud-fast,custom}, --json, dataclass knobs). Estimator is anchored to the measured 46-call/230s serial smoke (predicts 21.7 calls balanced) and models DSS with a 65% engine prefix-cache discount. SM reference clone lives at extensions/dayna_ss/.ref/Smart-Memory (SillyTavern JS extension, NOT TGWUI). Gotcha: preset is laptop-8b (not laptop-7b); argparse errors go to stderr silently if 2>/dev/null'd.

### #212

*updated 2026-08-24 12:12:05*

dayna_ss epistemics ADOPTED DESIGN (2026-08-24, recorded in docs/comparison.html P0 roadmap): normalized `secrets` map subject where each row IS one perspective — {summary, stance: knows|suspects|believes_falsely|unaware|hiding, owner, target?, about?, status: intact|suspected|exposed, importance}. One fact = one row; the counterpart's side is a second row or nothing (never duplicated prose). POV injection groups rows by stance over owner ∈ current_scene.now.who — NO per-character five-tag lists. Reveal flips status/stance on a single branch under mark_field. Relationships/characters may reference secret names like relevant_events for render-time joins, but authoritative linkage lives in the row so non-dyadic secrets stay first-class. Group-level rows = institutional semantics only (shared secrets, collective stances, rumor penetration). Guardrails: significance threshold in discovery prompts (only revelation-relevant facts become rows; trivial awareness stays unstructured — long soaks would mint stub entities against 120-entry caps); alias-table rename discipline. REJECTED alternatives: per-relationship-row storage (topical-not-dyadic problem, dual-sided duplication, fattens characters.json payloads); replacing relationship faction with epistemic stance (orthogonal affect axis, feeds EdgeRef scoring). Phase-2 optional: derived person↔secret edges for visualization/vantage queries; structurally converges on SM's bipartite shape but declarative-schema-driven. NWM paper vocabulary: focalization, event-vs-reveal order, promise/payoff.

### #216

*updated 2026-08-24 15:19:22*

dayna_ss engine refactor (user, 2026-08-24): the monoliths are gone. agents/summarizer.py -> package agents/summarizer/ {core.py (Summarizer+SummarizationContextCache+_LockedPhaseManager+format_number), context_engine.py (ContextEngineMixin: prepare_context/retrieve_and_format_context/get_retrieval_context/scene windows + MODULE-HOME OF base_state dict), instruction_generator.py, llm_client.py, population.py}; agents/data_summarizer.py -> package agents/data_summarizer/ {core, discovery, parsing, prompts, traversal, updates, archives}. Shared leaves at agents/formatted_data.py (FormattedData, MessageSummarizer; class-called @staticmethods). Package __init__ re-exports Summarizer/SummarizationContextCache/strip_thinking/strip_response/do_enc and DataSummarizer. CYCLE RULES: core imports all four mixins; mixin modules must NOT import core at module level - SummarizationContextCache uses deferred function-level import in context_engine.get_retrieval_context; base_state lives in context_engine.py, imported from there by core (re-export) and population. FIVE refactor casualty classes fixed at the cyberpunk turn-19+ resume: (1) missing import block in context_engine; (2) base_state unreachable; (3) five mixin methods missing self (_format_general_info_static/_collect_scene_boundaries/_current_scene_key/_load_summary_nodes/_format_rolling_summaries); (4) format_number @staticmethod+self contradiction; (5) _load_all_subjects_data LOST ITS RETURN STATEMENT (returned None -> unpack TypeError, swallowed non-terminating) - fixed; plus generate_instr_prompt latent fall-through when history_path falsy (defensive terminal return added). AUDIT TOOLING for refactors (all currently CLEAN): pyflakes in lmd-tg env; AST sweeps for self-less methods, kwargs-aware arg-count mismatches, decorator/signature contradictions, and annotated-function fall-through paths (try/finally-only counts as terminating; backtrack_history is an intentional documented stub). retrieve_and_format_context verified end-to-end via stubbed get_retrieval_context probe; real retriever path cannot run in THIS WSL env (llama_index _DeadlockError, known limitation) but works on user's machine.

### #217

*updated 2026-08-24 17:03:49*

dayna_ss context-budget overhaul (2026-08-24, targets the <40k-token rule #213): (1) general_info single-source — _format_general_info_static is now a FALLBACK only (gi_fallback_render appended after the context_order loop iff the format template produced no GI render); previously the full synopsis+premise rendered TWICE (~14.6K chars). (2) events template rewritten: Past Events/Scenes = one-line records (name + importance + first summary line truncated 140); Events keeps two-tier detail; participants render NAMES (mapping→keys(), list→p.name/p) not dict reprs; empty categories skipped. Measured on real cyberpunk turn-24 stores: events 20.1K→7.7K chars. (3) elements template three-tier: roster (<threshold), DORMANT compact (importance>=thr but _recent_state defined-and-empty), FULL (recent activity OR key undefined so per-call {{value}} renders keep full detail); threshold class fixed to ElementMap (was CharacterMap copy-paste); descriptions truncate(420/260), traits [:3], Importance compact "N, faction — reason". (4) characters/groups: same compact-importance + description truncate(420) + GroupMap threshold class fix + {path} marker brace fixes. (5) rolling summaries pair capped at render time via config rolling_summary_max_chars (default 280, sentence-boundary backoff). (6) sceneagg schema: SCENE RECAP sub-block stripped from all 8 per-entry branch_query/branch_update templates (Character/Group/Element/GeneralInfo) — recent exchanges ride the shared prefix; gate_check/new_entry_query/select_entries keep scene_recap+scene_events. (7) bugs fixed: arcs/chapters templates tolerate string-title entries (the <built-in method title> leak), bare <EMPTY> renders suppressed in context assembly, writing_style_placeholder macros de-{{char}}/{{user}}ed, \u2014 double-escapes sanitized in both schemas. JINJA GOTCHA: trim_blocks eats newlines after block tags — every template branch must END with an explicit \n or post-render line fusion swallows headers (bit twice). Verified: end-to-end stubbed retrieve_and_format_context on real turn-24 data = context 82,366→51,687 chars (-37%), synopsis once, zero leaks; both schemas parse; hermetic green. Restart picks up (startup schema/template refresh).

### #219

*updated 2026-08-24 19:20:44*

dayna_ss live cloud-call status reporting (2026-08-24): CloudModel(status_dir=run_dir/live, role=guide|judge|auditor) STREAMS all calls (SSE, same normalized return shape — retries/cooldowns/accounting unchanged) and writes {run_dir}/live/cloud_{role}_{pid}.json (phase generating/cooldown/backoff, elapsed, content+reasoning chars, head/tail snippet; 1 write/s throttle; removed on completion AND exception). Stdout heartbeat "[cloud {role}]" every 60s during generation/backoff/cooldown lands in the tee'd log. Dashboard shows the in-flight table inside the 10s auto-refreshed live-stats block (>45s no update = stale flag). Motivation: stealth/ox-alpha via OpenRouter is SLOW (guide turn 287.6s observed) — earlier "9-min hang" was one long cloud call, not a server stall; OpenRouter has no in-flight status API so streaming is the only visibility. Gotchas: ox-alpha 429s ping probes while a soak runs (shared per-minute budget); local server streaming omits prompt_tokens → char-estimate fallback.

### #220

*updated 2026-08-24 19:37:22*

dayna_ss traversal array-wrapped-instance heal (2026-08-24): the 27B/35B sometimes writes schema-class objects as ARRAYS in update responses — e.g. relationships: {"Juno": [{events, importance, ...}]} — which entry-time coerce_container_types cannot repair (items carry no name field for _infer_collection_key; the documented leave-as-list case) and which crashed traversal with "TypeError: list indices must be integers or slices, not str" at _initialize_field (parent_data is a list), aborting the elements subject EVERY turn the model emitted that shape (disk stayed clean because the crash prevented the save). Fix in agents/data_summarizer/traversal.py: DSTraversalMixin._normalize_schema_instance(value) unwraps [dict]→dict and merges multi-dict arrays (non-repairable passes through); wired at two descent sites — dict-of-classes case 3a loop (heals data[key] BEFORE the select_entries whitelist check, so healed values also get saved via recursive_set) and _process_field step 4 nested-class recursion (writes back parent_data[field_name]); plus a fail-open isinstance(data, dict) guard in _initialize_field. Unit tests + hermetic green.

### #221

*updated 2026-08-25 01:54:12*

dayna_ss cyberpunk_thriller__4e29a512 post-mortem (2026-08-25, 3.5/5 under new calibration ladder): (1) ANTICIPATORY EVENT RECORDS — the false "chip delivered" event was written at END OF TURN 0, four turns BEFORE the handoff scene: CrucialEvents add_new query asks "mentioned or implied" (invites planned actions) and StoryEvent schema REQUIRES outcome with no pending state, so the model fabricated completion + hallucinated date; record persisted byte-identical all 40 turns because select_entries_to_update never revisits stabilized entries and no contradiction audit exists. (2) ARCS/CHAPTERS CHAIN STARVED + ENGINE BUG: arcs need chapters>=2, chapters gate fires only on scene turns and needs scenes>=4 (early scene turns had 0-2), force threshold 10 never hit; AND _execute_action implements add_new only for dict branches (`if is_dict:` no else) so Chapters (list[Chapter]) add_new is a SILENT NO-OP — this also blocks the P0 canon-synopsis feature which hooks chapter archival. (3) ENTITY CONFLATION: one turn-30 pass simultaneously rewrote Pike INTO Kell and created standalone Kell (story never confirmed; zero uncertainty-preservation language in any update template); buyer+vendor duplicated same-pass; Vell/Handler duplicated same pass as the correct Handler reveal; _filter_new_entry_names dedupe is purely lexical. (4) DRONE DROPOUTS: instruction generator primary owner (7/9 miss-turns dropped even same-turn guide beats); retrieval NOT guilty (drone rendered in every context.txt); drone importance frozen at 60, entity status frozen "approaching" 25 turns; ROLLING OVERVIEWS DIAGNOSED the pattern repeatedly but there is NO feedback loop from rolling-overview findings into instruction generation.

### #222

*updated 2026-08-25 13:52:38*

dayna_ss aggregation-unit fixes (2026-08-25, post cyberpunk_thriller__4e29a512): (1) list-branch add_new now works — `_detect_and_add_new_entries_to_branch` accepts `list[...]` aliases (Chapters was a silent no-op: `_execute_action` had `if is_dict:` with no else); list entries dedupe against title/name/formal_name/id fields with title fallback; the formatted_data mirror append is identity-guarded (`expanded_data is not data`) because aliasing + append duplicates entries (dict path never noticed — key assignment is idempotent). (2) Chapters/Arcs `on_new_scene` triggers REMOVED in both schemas — creation+gating owned solely by check_and_archive_chapter/_arc; old triggers double-gated per scene turn and would have double-created. (3) Post-loop gates previously rendered `{{ scene_recap }}`/`{{ scene_events }}` EMPTY (vars never passed) — judged on bare counters; now receive span_digest/chapters_digest. (4) Real span math: scenes_in_chapter = scenes since last chapter's ending_scene (was total ever); chapters_in_arc = chapters since last arc's ending_chapter (was TODO). Unit semantics encoded in gate templates: scene = action/tone/cast/setting stretch; chapter = TV episode (4-8 scenes, hard 10); arc = TV season (3-6 chapters, hard 12); transition prompt (context_engine.py) now accepts action-phase shifts + material cast-presence changes as signals. Canon synopsis P0 is unblocked (hooks chapter archival). Verified: 8-case unit test + hermetic green.


## CONFIG_VALUES

### #1

*updated 2026-08-24 12:05:52*

dayna_ss live model server: run `./strt mp` in text-generation-webui root. CMD_FLAGS_m.txt holds `--model <NAME> --n_ctx <CTX> --loader LMDeploy --cache-type q8`. CMD_FLAGS_p.txt supplies --api-key 764d7ca1-7a25-405e-ad27-4fe7ff0bf1ae. `./link.sh` symlinks user_data/models to /mnt/c/Users/there/Downloads/Projects/Programming/models. Server listens on http://0.0.0.0:5000/v1. Use `setsid nohup ... &` so it survives tool timeouts.

### #7

*updated 2026-08-11 13:50:05*

dayna_ss long-horizon soak harness lives at extensions/dayna_ss/tests/. Entry: `python tests/long_horizon_soak.py --spreadsheet spreadsheets/<genre>.json --smoke N --level 1|2`. Level 1 = fast per-subject path (live_soak-style, no retrieval stack); Level 2 = full production path (requires llama_index/sentence-transformers warm-imports, embed-device cpu option). Invoked via run_tests.py --long-soak <spreadsheet> (uses local server at DSS_BENCH_BASE_URL + cloud guide/judge). Run dirs under tests/runs/<id>__<hash>/ with per-turn turn_XXX/{result.json,state_snapshot/} and atomic resume via manifest.json's last_turn.

### #25

*updated 2026-08-25 14:20:13*

(HISTORICAL — measured on deepseek-v4-flash at the legacy OpenCode Go endpoint, retired 2026-08-24; see #223) deepseek-v4-flash accepted reasoning_effort "none"|"low"|"medium"|"high" but emitted a substantial `reasoning` block at EVERY effort level (2-8K chars at none, verified 2026-08-16); manual think-token injection (ASCII "thinking\nresponse", <|think|> stubs) did NOT suppress it. "low" was the most content-consistent level. Budget lessons that carried over to the OpenRouter stack: reasoning-only draws eat max_tokens and leave content empty → tiered empty-content reroll in CloudModel.complete (small budgets double once, large +2048 once) + cooldown re-rolls; overview calls need 16000 tokens, judge 8000, auditor 1600, timeouts 300s; json_complete feeds parse errors back to the model on the third try.

### #34

*updated 2026-08-25 14:19:56*

(HISTORICAL — endpoint retired 2026-08-24, see #223 for the current OpenRouter stack) OpenCode Go cloud endpoint (opencode.ai/zen/go/v1): POST https://opencode.ai/zen/go/v1/chat/completions (OpenAI-compatible). Auth: Authorization Bearer auth.json["opencode-go"]["key"] — the opencode-go entry is a dict, must read .key; auth.json lives at ~/.local/share/opencode/auth.json. Cloudflare blocks default urllib UA (error 1010) — MUST send a browser User-Agent header (e.g. Mozilla/5.0 ... Chrome/126.0).

### #72

*updated 2026-08-11 20:32:35*

text-generation-webui fork migrated 2026-08-11: the working repo is now /mnt/c/Users/there/Downloads/Projects/Programming/Python/textgen (fresh `git clone --branch lmdeploy` from the fork Th-Underscore/text-generation-webui — branch lmdeploy at commit 22510aed, all 20 commits incl. the LMDeploy fixes, clean .git). Upstream remote added (oobabooga/text-generation-webui). The OLD dir text-generation-webui/ is preserved untouched (its git dir is .git.og). Non-git data copied: user_data/ (models/ symlinks to /mnt/c/.../Programming/models/, characters, settings), extensions/dayna_ss/ (entire DSS project, untracked+gitignored), lmdeploy/ source tree (git repo branch pr-4465, has its own .git + 1 local file granite.py), llm-compressor/, CMD_FLAGS*.txt, strt, .ignore. Server launch unchanged: `./strt mp` from textgen (CMD_FLAGS_m.txt carries --loader LMDeploy --disable-prefix-caching --cache-type q8 --cache-max-entry-count 0.8 --n_ctx 32768). Pending: Phase 0 merge of upstream main (fork is 33 commits behind), and session/memory re-link to the new project root.

### #86

*updated 2026-08-12 12:14:29*

~/.local/share/opencode/opencode.db is ~1.6GB and WSL disk is frequently at 98% — DB backups made by migrate_opcc_ses.py double that footprint, so test copies should use a tiny stub opencode.db (create schema + one session row) instead of copying the real one, and context.db copies go to /tmp/opencode.

### #105

*updated 2026-08-17 17:27:26*

dayna_ss current local writer under test (since 2026-08-12) is Qwen3.6-35B-A3B-abliterated-AWQ — a 35B MoE (3B active/token, 256 experts), served by LMDeploy v0.15 TurboMind via TGWUI on both V100s (--tensor-parallel 2, requires --disable-prefix-caching + --max-batch-size 8 for safe concurrency). The earlier "9B model" claims in analysis/plan docs refer only to 9B-era runs (seed-77 baselines); they do NOT apply to current runs. Embeddings use cuda:0 (the GTX 1060).

### #214

*updated 2026-08-24 14:03:00*

dayna_ss cloud provider migrated to OpenRouter (2026-08-24 state): cloud_client.py DEFAULT_ENDPOINT is now https://openrouter.ai/api/v1/chat/completions (OpenCode Go endpoint commented out — deepseek-v4-flash price hikes + rate limits forced the move; free-tier OpenRouter models 400-crashed runs on context-length limits). CloudModel.base_delay raised to 90s for rate-limit backoff. Current cloud trio: stealth/ox-alpha (free/unlimited for a few days from 2026-08-24) as guide+judge+auditor. KEY RESOLVER FIXED 2026-08-24: resolve_api_key() prefers OPENROUTER_API_KEY env -> auth.json 'openrouter' -> legacy OPENCODE_GO_API_KEY -> auth.json 'opencode-go' (the week's migration had moved DEFAULT_ENDPOINT but not the resolver — soak runs without the old env var sent the stale Go key and 401'd; user had been papering over it with OPENCODE_GO_API_KEY=<openrouter key>). guide_turn_with_retry fail-fasts on HTTP 401/403 instead of cooldown-retrying. Verified live: reasoning_effort="low" + json_mode accepted by ox-alpha through the client. Local server: Qwen3.8-Queen-27B-W4A16-AWQ on CMD_FLAGS_2.txt at :5000 WITH API-key auth enabled — harness resolves the key via auth.json 'localhost'/'tgw'/'text-generation-webui' entries.

### #223

*updated 2026-08-25 14:19:41*

dayna_ss CLOUD STACK (current since 2026-08-24): the soak's CloudModel migrated from the legacy OpenCode Go endpoint (opencode.ai/zen/go/v1) to OpenRouter — DEFAULT_ENDPOINT = https://openrouter.ai/api/v1/chat/completions in tests/cloud_client.py. Key resolution is ENDPOINT-AWARE: OPENROUTER_API_KEY env → auth.json["openrouter"] → legacy OPENCODE_GO_API_KEY → auth.json["opencode-go"]; no env prefix needed on launches (an old-env workaround masked the stale resolver for a week). Models: guide/judge/auditor/overviews use the ox-alpha trio via `stealth/*` slugs (free + effectively unlimited); deepseek-v4-flash was dropped (price hikes + rate limits). Behavior notes: free-tier models reject long prompts with hard HTTP 400 (killed cyberpunk_thriller__40c80fa4 at t19) — guide_turn_with_retry treats CloudError as empty-draw cooldown and FAILS FAST on 401/403; 429 rate limits back off base_delay=90s and share the per-minute budget with any manual probes (llmping pings compete with a running soak). reasoning_effort:"low" is sent and tolerated by OpenRouter; ox-alpha streams SSE when status_dir is set (live/cloud_{role}_{pid}.json). Legacy Go-endpoint facts in older memories (#25/#34) are HISTORICAL.


## CONSTRAINTS

### #2

*updated 2026-08-17 11:58:49*

Qwen3.5-era thinking models (e.g. Huihui-Qwen3.5-9B-abliterated-AWQ-4bit) burn their whole token budget on `reasoning_content` by default, returning empty `content`; pass `enable_thinking: false` in OpenAI-compatible requests to get plain text. dayna_ss live tests default to this via DSS_BENCH_THINKING=0.

### #27

*updated 2026-08-24 12:06:03*

dayna_ss soak harness LocalModel must (1) send state["context"] as a system message (production TGWUI does this via generate_chat_prompt(chat.py:399-401); the engine's instr_prompt NEVER contains context/dss_directive — it only says "write a reply in character as {name2}" + generated instructions), and (2) use max_tokens >= 2048 for engine JSON summarization calls — the old 512 cap truncated EVERY structured JSON response (current_scene/new-entry/events/entity identification), causing 'Failed to parse LLM response' errors AND a jinja 'dict object has no attribute start' (truncated current_scene lacked the schema-required 'start' key). Both fixed in long_horizon_soak.py LocalModel.

### #43

*updated 2026-08-24 12:05:52*

dayna_ss soak LocalModel temperature: was hardcoded temperature=0.0 (long_horizon_soak.py:195) — greedy decoding was a key driver of the frozen-output loop (near-identical prompts → byte-identical replies). Now LocalModel accepts temperature (default 0.8) and sends it in the payload (long_horizon_soak.py:166,196). The harness still uses an identity prompt_builder (line 909) so the engine's rich retrieval never reaches the model — engine-path context assembly is a known open improvement.

### #136

*updated 2026-08-17 17:27:26*

textgen lmdeploy 0.15+ loader regression (modules/lmdeploy.py _prepare_generation_config): lmdeploy GenerationConfig.do_sample defaults to False (messages.py:117), and async_engine._determine_gen_config (serve/core/async_engine.py:412-417) then forces greedy decode (top_k=1, temperature=1.0, repetition_penalty=1.0) regardless of the request temperature, so every generation was deterministic argmax. FIXED 2026-08-15: the loader sets do_sample=temperature > 0. This was the byte-identical reply/instruction amplifier in cozy_mystery__45adee46 (turns 7/8/9 same md5, turns 8/9 same instructions.json md5). NOTE: base_state['temperature']=0.3 in agents/summarizer.py is engine-side and DEAD in the soak harness — LocalModel ignores state temperature and sends its own (CLI --temperature, default 0.7) — not a cause. After the fix, sampling is live.

### #163

*updated 2026-08-17 14:33:59*

MessageSummarizer stores is_summary nodes with id '{message_idx}_summary' and indices [idx,0,0], scene_id/event_id None at creation then backfilled for body chunks via update_node_metadata_by_message_idx (summarizer.py:1784-1830). Summaries of earlier messages sticky roll reads them via _load_summary_nodes (summarizer.py:2515); rolling_summaries=20 config. _collect_scene_boundaries reads start._message_node from archived scenes and current scene start — current_scene.json uses _scene_start_message (int), not _message_node, so current-scene bounds are missing unless archived.

### #202

*updated 2026-08-23 22:45:07*

deepseek-v4-flash on the OpenCode Go cloud endpoint (opencode.ai/zen/go) ALWAYS emits a large reasoning block on big prompts even with enable_thinking:false + reasoning_effort='none' — the reasoning consumes the token budget, so a low max_tokens truncates/empties the real content (finish_reason=length with empty `content`). The response field is `reasoning` (NOT `reasoning_content`, so the complete() fallback never surfaces it). On a 38-40K-token overview request, max_tokens≈16000 is needed (~8.5K reasoning + ~9K content). Failure presents as 'judge output was not JSON:' (empty), socket read-timeouts (large calls take 70-150s, beyond a 120s timeout), or HTTP 500. Soak budgets are overview/judge/auditor 16000/8000/1600 with timeouts 300s; cloud_client empty-content retry is tiered — small budgets (<4096) double, large budgets just +2048. Even at adequate budgets the model randomly enters reasoning-only states on huge transcripts (temp variance), so bounded rerolls are load-bearing.

### #213

*updated 2026-08-24 13:53:31*

dayna_ss context budget rule (user directive 2026-08-24): total rendered context should stay under 40k tokens whenever possible. The rig can handle >60k, but context retrieval is currently "very inflated" — the <40k budget is a standing design target for every render/prompt change. The P0 Canon Synopsis feature (comparison.html roadmap) directly serves this: closed-arc entities demote to roster lines to cap linear token growth.

### #218

*updated 2026-08-24 18:08:28*

dayna_ss / TurboMind scheduler-stall kick (2026-08-24): the LM server can STALL entirely — GPUs idle (0-6%), queue jammed, no output — after KV-eviction churn or a client being SIGKILL'd mid-stream; every queued request waits indefinitely. Empirically verified: any NEW arrival at /v1/chat/completions (even a 5-token ping) kicks the scheduler loop awake and unblocks everything instantly. Remedy: `llmping` (~/.local/bin/llmping) — presets: `llmping` / `llmping local` (:5000, key auto-resolved from auth.json localhost/tgw/text-generation-webui), `llmping openrouter` (stealth/ox-alpha; may 429 on ping-rate limits — still proves reachability), `llmping go` (legacy), or a raw URL via LLMPING_KEY/LLMPING_MODEL env. Diagnosis rule: soak frozen + GPUs idle = server-side stall (kick it); soak frozen + GPUs busy = slow decode, be patient. Also: avoid SIGKILL'ing soaks mid-generation (orphans can trigger the stall); Ctrl+C and wait for graceful shutdown first.


## PROJECT_RULES

### #35

*updated 2026-08-17 11:58:41*

In dayna_ss, when implementing template-based or data-driven systems, do not reintroduce hardcoded schema dispatchers, key maps, or per-data-type duplication. Compute template keys directly from the data type (such as template_{data_type}) and structure template definitions so every data type is handled without a hardcoded class mapping; a per-data-type dispatch function that maps types to schema classes repeats the hardcoding the user asked to eliminate, and shared helper modules must stay free of hardcoded keys as well.

### #36

*updated 2026-08-23 22:46:49*

When a change is applied to one prompt template or one formatting function, propagate the same change to all other prompt templates and call sites in the same pass. The user repeatedly has to interrupt to request that similar changes be applied to the remaining templates, so audit every template and shared utility in the same edit rather than fixing only the demonstrated case.

### #47

*updated 2026-08-11 00:15:03*

All 7 dayna_ss soak spreadsheets carry operational hard rules in writing_style.dss_directive: "Hard rules: narrate {name2} strictly in the third person — never first person ('I','my','me'); never quote or repeat {name1}'s lines back verbatim — respond to them, do not re-speak them; end on action or observation, not summary." This text flows everywhere production-style (build_state context turn 0, general_info.writing_style turns 1+, Level-1 inline prompt, judge style scoring). If you change a spreadsheet's dss_directive, apply the same operational rules to ALL spreadsheets and re-run validate_spreadsheet.py per-file (it does NOT accept a directory).

### #49

*updated 2026-08-11 13:50:05*

dayna_ss soak early-stop (`--abort-after N`, default 2, 0 disables): the judge and auditor prompts carry an ABORT CLAUSE that allows `"abort": true` ONLY for catastrophic multi-turn failures (gibberish, wholesale hard-rule violation across consecutive turns) — a single style slip or missed note is NEVER enough. The harness stops the run after N consecutive `abort:true` flags (current turn still checkpointed, then report + final overview still run). Judge/auditor JSON gained an additive `abort` field; report.py/soak_dashboard.py ignore it. Judge/auditor JSON shapes otherwise unchanged.

### #87

*updated 2026-08-12 12:14:31*

migrate_opcc_ses.py (in ~/) rekeys magic-context context.db identities when moving opencode sessions between directories. Collision policy must mirror the plugin's mergeProjectIdentities (dist/index.js): generic tables with a unique index containing project_path → target row wins, colliding source rows DELETEd (collision_deleted); memories → merged into target (max seen_count, embeddings copied, source deleted); embedding_identity_active/embedding_registrations handled separately. Writes v22_identity_rekey_map, identity_merge_log, project_state epoch bump. Supports --rekey-context OLD NEW to recover when the opencode.db half already committed but the context rekey crashed.

### #157

*updated 2026-08-24 12:06:17*

Before writing an update or query prompt template for a dataclass in dayna_ss, check whether that dataclass is shared across multiple parent classes such as relationships and character milestones, and keep the template wording generic to all of its uses instead of mentioning the domain of one parent. When a parent class needs specialized wording, prefer a per-class default template override over hardcoding domain-specific text into the shared dataclass's template.

### #170

*updated 2026-08-24 12:06:17*

Renaming or deprecating a prompt template in the dayna_ss schema must not change the action bound to it: when the user retires branch_query prompt templates in favor of gate_check, keep the associated action as-is, so importance querying stays query_branch_for_changes and is never swapped for perform_update.

### #204

*updated 2026-08-24 12:06:17*

When the user renames or deprecates a prompt template in the dayna_ss schema, keep the action bound to that template unchanged and propagate the rename across every variant plus the Python that auto-derives template names from linked fields (such as deriving the branch_query name from branch_update); this includes applying the same paired rename symmetrically to the importance query/update templates rather than only to the branch pair.

### #205

*updated 2026-08-24 12:06:30*

Initial population of the dayna_ss world model should be schema-driven, declaring which subjects to populate and supplying a base_prompt_template for each, and the validation error retry loop that exists in one generation path must be applied uniformly to the other paths such as data_summarizer; the user repeatedly flags when a code path omits a validation loop that another path already has.

### #211

*updated 2026-08-24 12:06:34*

When the initial world population detail-extraction prompt causes repeated validation failures because the LLM wraps its output under an entries key that contradicts the schema, fix the correction at the prompt source so the LLM generates the schema-conformant structure directly, rather than patching the returned JSON with a hardcoded unwrap of the entries wrapper. Do not add downstream special-case code that descends into an entries key; ensure the example and instructions in the prompt itself produce the correct top-level shape that validates on the first attempt.

### #215

*updated 2026-08-24 14:04:30*

dayna_ss soak launch logging convention: the USER launches soaks with `| tee -a runs/logs/<ephemeral>.log` on purpose — the harness self-log (runs/logs/<run_id>.log) is deliberately filtered/concise, and they want the full unfiltered stdout preserved separately. Do NOT advise dropping tee for user-run launches. The old "tee gets killed at timeout" issue only applies to agent-side background launches via the bash tool (process-group kill) — when an AGENT launches a soak in the background, omit tee and rely on the self-log; when the USER runs it interactively, tee stays.
