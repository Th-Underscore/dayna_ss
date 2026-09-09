# LMDeploy upgrade + `modules/lmdeploy.py` revamp plan

Status: **Phases 1-3 COMPLETED (2026-08-11)** · Driver: DSS soak needs **concurrency AND real prefix caching** together (see `soak_optimization_plan.md`).
The fork's current prefix caching is built on a rejected upstream PR, never hits, and segfaults under 4+ concurrent requests.

## 0. Result summary (Phases 1-3 done)

| Item | Result |
|---|---|
| Phase 0 | TGWUI merged to upstream `main` (33 commits behind → current); LMDeploy wiring re-applied |
| lmdeploy | **0.12.3 → 0.15.0** (prebuilt cp312 wheel; 0.12.3 backed up to `/tmp/opencode/lmd123_backup`, rollback = `pip install lmdeploy==0.12.3`) |
| V100 gate | PASS — wheel runs on sm_70, no source build needed |
| Conversion VRAM | **CPU checkpoint patch** (`_cpu_checkpoint_load`) replaces the old 6-function `_cpu_realtime_conversion`: strip `.cuda()` from `SafetensorsCheckpoint`/`PytorchCheckpoint.get/pop`; `_copy_shard_to_param`'s `copy_from` does the single H2D. `--cpu-realtime-conversion` default flipped to **True** (required on 16 GB) |
| Session_len | **32768 loads fine** with the CPU patch (14.2 GB peak). The pre-patch OOM was conversion transients, NOT the GDN state cache |
| Prefix caching | **WORKS**: identical 5.5 K-token prompt → 1.4 s → **0.2 s** (7x), `matched [0,2816) 44 blk`. `cache_prompt='all'` + `cache_generation='all'` + `cache_checkpoint_interval=4096`; short prompts (< ~2816 tok) still hit via KV blocks |
| Concurrency | **8-way race-free** (v0.15 single-scheduler-thread design), wall 4.4 s for 8×64 tok |
| Thinking | Qwen3.5 template emits `<think>\n` by default → model burns budget on "Thinking Process". Harness `LocalModel` already sends `enable_thinking: false`; API accepts it (verified `content: 'Hello there'`, no reasoning) |
| `modules/lmdeploy.py` | Revamped: `_workspace_load_hook` + `_cpu_cast_export_weight` removed (no workspace format in v0.15), `convert_model_to_turbomind` = load-time validation + `.source_model` pointer, `generate_with_streaming` bridge unchanged (API-compatible) |
| Gotcha | Saved per-model settings in `user_data/models/config-user.yaml` override parser defaults via `update_model_parameters(initial=True)` — `Huihui-…` entry had `cpu_realtime_conversion: false` (old default); fixed to `true` |
| Server flags | `CMD_FLAGS_m.txt` now WITHOUT `--disable-prefix-caching` (caching ON); `--cache-max-entry-count 0.8` kept |
| Soak smoke | 2-turn L2 (`--max-subject-workers 4`): turn0 46 calls/5.5 min, turn1 25 calls/1.5 min (vs serial ~5-9 min); third-person Evelyn replies; report + final overview valid |
| Hermetic suite | PASS (40 scripted LLM calls) |

---

## 1. Verified current state (2026-08-11)

| Item | Value | Note |
|---|---|---|
| TGWUI branch | `lmdeploy` | 19 local commits (all additive LMDeploy wiring) |
| TGWUI upstream base | `dev` @ 2026-05-16 | fork tracks `dev`; **upstream's active branch is `main`** |
| TGWUI gap | **33 commits behind origin/main** | main went 05-16 → 05-31 (llama.cpp, MTP, CORS, security, UI) |
| Upstream LMDeploy loader | **nonexistent** | `modules/lmdeploy.py` never existed on main — 100% fork-owned |
| lmdeploy pip | 0.12.3 | built from repo-root `lmdeploy/` source (branch `pr-4465`) |
| `_turbomind.so` | 328 MB, built 2026-06-09 | from the pr-4465 source tree, installed into site-packages |
| lmdeploy upstream | **v0.15.0 on PyPI** (prebuilt cp312 manylinux x86_64 wheel) | has lzhangzz scheduler/object-cache rewrite + real GDN prefix caching |
| Served model | `Huihui-Qwen3.5-9B-abliterated-AWQ-4bit` | `./strt mp`; CMD_FLAGS_m.txt already has `--disable-prefix-caching` |

### Why the current build can't just be tuned
- PR #4465 ("Turbomind linear gdn prefix caching") was **rejected & closed unmerged** (lvhan028 declined; an upstream commenter reported the same parallel-run crash we hit).
- Observed in our build: `[SeqMgr][match] ... hit blocks 0` even for byte-identical prompts (cache never hits); native **Segfault at 4+ concurrent requests** (BlockTrie/SequenceManager race); 4096-token GatedDeltaNet reuse clamp is fork-local.
- Upstream `main` (post-#4557, v0.14.0a1 → v0.15.0) replaced all of that with `engine/scheduler.cc`, `prefix_trie.h`, `cache_registry.cc` and the `EngineConfig` surface (`cache_prompt` / `cache_generation` / `cache_checkpoint_interval`), **race-free by single-engine-thread design**, with GDN recurrent-state checkpoints. Support for Qwen3.5/GDN landed in #4757 / #4744 / #4688 / #4700.

---

## 2. Git hygiene (`.git` was deleted)

All git commands currently fail without `--git-dir`. Fix options (pick one):
- `export GIT_DIR=.git.og` in the session shell, then normal `git fetch/pull/merge/log` work.
- Or a 30-byte `.git` **file** with `gitdir: /abs/path/to/.git.og` (git dot-git pointer) — but only if the program that choked on `.git` tolerated a file (test it).
Origin URL is already correct inside `.git.og` — **do not re-add it**.

**Recommended tracking change:** switch the fork to track `main` (upstream's real branch): the fork's `dev`-based merge predates 10+ commits that main already absorbed (see the `Merge pull request … from oobabooga/dev` commits). `git fetch origin main` and merge.

---

## 3. Decision summary

- **TGWUI upgrade: DO IT, but as its own low-urgency step, FIRST** (before the lmdeploy revamp) so the revamp lands on a current base. Not required for the soak, but the 33 commits include security fixes (CORS-to-localhost, path-traversal in `load_character`/`load_template_by_name`) and MTP support. Moderate conflict risk, confined to files the LMDeploy wiring touched: `modules/shared.py`, `modules/text_generation.py`, `modules/loaders.py`, `modules/chat.py`, `modules/models.py`, `modules/ui_model_menu.py`, `api/models.py`, `api/script.py`, plus a new `modules/windows_subprocess.py`. The fork's 19 commits re-apply cleanly on top (LMDeploy files are untouched by main, so no conflict in `lmdeploy.py` itself).
- **lmdeploy package: upgrade to v0.15.0** — prebuilt wheel if it passes the spike, else source build.
- **`modules/lmdeploy.py`: full revamp** against the new engine API (Phase 2).

---

## 4. Phase 0 — TGWUI merge to `main` (≈1-2h, conflicts expected)

1. `git fetch origin main`
2. `git merge origin/main` on branch `lmdeploy`
3. Resolve conflicts. Fork-owned LMDeploy hunks to re-apply / re-verify after merge:
   - `modules/shared.py` — the LMDeploy arg group (`--backend`, `--max-batch-size`, `--cache-type`, `--cache-max-entry-count`, `--disable-prefix-caching`, `--cpu-realtime-conversion`, `--extra-flags`, `--ctx-size`)
   - `modules/text_generation.py` — `LMDeployModel` in the `use_parallel` whitelist (upstream changed this block for MTP)
   - `modules/loaders.py` — `--max-batch-size` / `--disable-prefix-caching` args surfaced in the loader registry
   - `modules/chat.py` / `api/completions.py` / `modules/models.py` — `last_prompt_token_count` read sites
4. Smoke: `./strt mp`, one chat completion via the API, one DSS Level-1 smoke.

## 5. Phase 1 — lmdeploy package spike (≈30-60 min, gating)

Goal: decide **wheel vs source build**.

1. Scratch venv → `pip install lmdeploy==0.15.0` (prebuilt cp312 manylinux_2_28 x86_64 wheel).
2. Verify wheel contents: `unzip -l <wheel> | grep _turbomind` (turbomind runtime present, not pytorch-only).
3. **V100 gate (sm_70)**: `python -c "import lmdeploy, _turbomind"` + attempt to load the existing Qwen3.5 workspace on the V100. Prebuilt wheels sometimes ship only sm_80+ kernels — if it fails to load on sm_70, fall back to a source build of v0.15.0/main (the repo-root `lmdeploy/` tree → `git fetch && git checkout v0.15.0` → `pip install .`, ~1-2h compile).
4. **Workspace-format gate**: can the new engine read the existing pr-4465-era converted workspace (`config.yaml` + rank files)? If not, re-convert via `convert_model_to_turbomind` (which itself needs the revamped deploy internals — do the re-conversion with the `--cpu-realtime-conversion` path).
5. Quick generation parity check (same prompt, same seed) vs the current build.

## 6. Phase 2 — `modules/lmdeploy.py` revamp (≈2-4h)

Mirror of the current 785-line file, area by area. Keep the public surface stable:
`LMDeployModel.from_pretrained / generate / generate_with_streaming / encode / decode / unload /
last_prompt_token_count` + `LMDeployTokenizerWrapper` + `convert_model_to_turbomind` + the three context-manager patches.

| # | Area (current lines) | What changes for v0.15.0 |
|---|---|---|
| 1 | `_parse_extra_flags` (25-66) | Add new engine flags: `cache_prompt`, `cache_generation`, `cache_checkpoint_interval`. `--disable-prefix-caching` maps to `cache_prompt=False, cache_generation=False`. |
| 2 | Config assembly in `from_pretrained` (617-665) | `TurbomindEngineConfig(tp, session_len, max_batch_size, quant_policy, cache_max_entry_count, max_prefill_token_num, num_tokens_per_iter, enable_prefix_caching)` → new `EngineConfig` surface. Verify which params survived (likely renames: `session_len`, `max_batch_size` probably stay; `enable_prefix_caching` becomes `cache_prompt`/`cache_generation`; `num_tokens_per_iter`/`max_prefill_token_num` may move). Pytorch backend (`PytorchEngineConfig`) re-verify similarly. |
| 3 | `_workspace_load_hook` (73-267) — **HIGHEST RISK** | Patches `TurboMind._from_hf`/`._load_weights`, `archs.autoget_backend`, `HuggingFaceTokenizer.__init__` — internals the rewrite removed. The new engine loads via `engine/…` (scheduler + object cache), not `TurboMind`. Rework = re-implement "load pre-converted workspace without re-conversion" against the new loading path, OR drop the hook and load HF directly (standard `Pipeline(model_path)` does on-the-fly conversion; the `--cpu-realtime-conversion` patch then covers the VRAM peak). This is where the design decision lives. |
| 4 | `_cpu_realtime_conversion` (274-429) + `_cpu_cast_export_weight` (436-463) | Patch `lmdeploy.turbomind.deploy.policy` (`to_cuda`, `process_awq_gemm`, …) and `BaseOutputModel.export_weight`. Verify these deploy internals still exist on v0.15 (deploy module mostly lives on; signature check). |
| 5 | `convert_model_to_turbomind` (498-560) | `get_tm_model` + `tm_model.export()` + tp-aware `save_split` patch — same deploy-internal dependency. Re-verify `get_tm_model` signature. |
| 6 | `generate_with_streaming` (683-729) | The asyncio bridge: `self.pipeline.session()` + `async_engine.generate(..., stream_response=True)` via `run_coroutine_threadsafe(self.pipeline.internal_thread.loop)`. The new engine exposes cache toggles per request (`cache_prompt`/`cache_generation` on the generate call); verify the session/loop plumbing survived. Keep the `shared.stop_everything` break + local `prompt_token_count` computation (the race fix). |
| 7 | `_prepare_generation_config` (737-750) | `GenerationConfig` — verify fields unchanged (likely yes). |
| 8 | `last_prompt_token_count` (761-770) | Keep exactly as-is (external readers: `text_generation.py`, `chat.py`, `api/completions.py`). |

**Parallel-safety invariant:** the new engine is single-scheduler-thread; request concurrency must come from batched/parallel generate calls, NOT from touching engine internals from many threads. `generate_with_streaming` must remain safe under the soak's `--max-subject-workers N`.

## 7. Phase 3 — verification

1. Model loads on the V100; `./strt mp` brings the API up; one chat completion matches current output quality.
2. **Concurrency probe** (reuse `tests/probe_steps.py`): 4 / 6 / 8 parallel OpenAI calls — must be stable (no segfault), measure speedup vs serial (expect ≥1.85x).
3. **Prefix-cache probe**: identical repeated prompts → server log must show cache hits (`hit blocks > 0` or the v0.15 equivalent), latency for the 2nd+ call drops; then a DSS-style large-shared-prefix probe (same system message, varying tail) at 8K/16K ctx to confirm reuse scales with ctx.
4. **Soak smoke**: `--max-subject-workers 4` 3-turn Level-2 smoke; confirm subjects written, replies third-person, per-turn time materially below the serial baseline.
5. Hermetic suite (`run_tests.py` core) green.
6. Flip `--disable-prefix-caching` OFF (default now on) and re-probe concurrency + cache hits together.

## 8. Risks & fallbacks

| Risk | Fallback |
|---|---|
| v0.15.0 wheel lacks sm_70 / GDN kernels | Source build of v0.15.0 (or main) from repo-root `lmdeploy/` |
| Workspace format changed | Re-convert via `convert_model_to_turbomind` (with cpu-realtime to survive V100 VRAM) |
| `_workspace_load_hook` impossible on new engine | Load HF directly + rely on cpu-realtime conversion path; accept slower first load |
| TGWUI merge conflicts | Resolve per-file; LMDeploy files themselves are conflict-free (main never had them) |
| New engine drops `num_tokens_per_iter`/`max_prefill_token_num` | Drop from config; defaults acceptable |
| v0.15.0 regression vs 0.12.3 output | Keep the 0.12.3 venv + pr-4465 source intact; `pip install lmdeploy==0.12.3` to roll back while keeping the revamped wrapper (config-version shim) |

## 9. Sequencing & effort

Recommended order: **Phase 0 (TGWUI→main) → Phase 1 (spike) → Phase 2 (revamp) → Phase 3 (verify)**.
Total ≈ half a day to a day of focused work. The DSS soak can run in parallel only if it can share the V100 with the upgrade work — otherwise run the 20-turn perf soak BEFORE starting the upgrade (recommended: it's already queued with seed 77 and isolates the current batch of fixes).
