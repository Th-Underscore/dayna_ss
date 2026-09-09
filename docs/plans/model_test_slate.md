> **Status update 2026-08-25:** Qwen3.6-35B-A3B-abliterated-AWQ proven the best soak model
> (4.0/5-class runs, clean voice/memory). Current server = **Qwen3.8-Queen-27B-W4A16-AWQ**
> (tp=2 + `--language-model-only`, `CMD_FLAGS_2.txt`; 35B fallback in `CMD_FLAGS_m.txt`).
> MTP evaluated and rejected for this workload (~99% prefill; spec decode only helps decode).
> Magnum-v4-12B: Mistral alias patch applied upstream-side, not yet live-tested.
> Gemma4: needs llama.cpp or the C++ port — see `lmdeploy_gemma4_pr_plan.md`.

# DSS local-model test slate

What we're optimizing: **narrative prose quality** for the dayna_ss soak harness
(the local model writes third-person character narration + JSON memory updates).
Current DSS model = `Huihui-Qwen3.5-9B-abliterated-AWQ-4bit` (Qwen3.5, coding-lean,
weak/echoing prose). Researched 2026-08-11 by three subagents (HF hub, r/LocalLlama,
EQ-Bench Creative Writing v3 signals).

## Hard constraints

| Constraint | Value |
|---|---|
| GPUs | 2× V100 16GB (sm_70, NVLink NV2) + GTX 1060 3GB |
| Backend | lmdeploy 0.15 TurboMind (whitelist: Qwen2/3/3.5/3.6, Llama, Mixtral, InternLM/VL, GLM-4-moe-lite, GPT-OSS) OR llama.cpp loader |
| Quantization | FP16 <14GB → single V100; up to ~28GB via `--tensor-parallel 2`; larger needs AWQ INT4 |
| Preference | abliterated/uncensored over stock for non-RP-tuned; INT8-grade over INT4 for tiny models |
| License | Apache-2.0 preferred (Gemma license gated; LFM $10M-revenue clause is a hard filter) |

## All candidates (how each was found)

### Small 1–6B — subagent A (HF search "storytelling/roleplay/creative writing" + r/LocalLlama)

| Model | Params/arch | Ctx | Quants | License | Pros / Cons | Backend |
|---|---|---|---|---|---|---|
| **google/gemma-4-E4B-it** | 4.5B eff dense (8B total) | 131K | GGUF Q8/Q6, mlx 8bit, no AWQ | Apache-2.0, ungated | **Community prose/RP darling** ("acts like a real big model", "promotes creativity"); native system prompts. New arch → lmdeploy **unsupported** | llama.cpp |
| google/gemma-4-E2B-it | 2.3B eff | 131K | GGUF Q8 | Apache-2.0 | Same family, cheaper; the fallback if E4B arch fails on the stack | llama.cpp |
| google/gemma-3-4b-it | 4B dense | 131K | AWQ-4bit exists, Google QAT | Gemma license, gated | Battle-tested prose; roleplay-tuned variants exist (`Indexnusrefather/gemma-3-4b-it-roleplay-tuned-v2`); won Mar-2025 under-15B writing test (but "flowery/slop" tendency) | llama.cpp |
| LiquidAI/LFM2.5-2.6B | 2.69B | 131K | GGUF Q8, MXFP8, no AWQ | LFM v1.0 ($10M cap) | Fastest-in-class edge; agentic/reasoning; "not for knowledge-heavy"; prose unproven | llama.cpp |
| LiquidAI/LFM2.5-8B-A1B | 8.3B/1.5B act MoE | 128K | community AWQ INT4/FP8 | LFM v1.0 ($10M cap) | Excellent IFEval 91.8 (JSON side); 1.5B active = fast; reasoning spends tokens; prose unproven | llama.cpp |
| Qwen/Qwen3.5-4B | 4B | 262K | AWQ INT4, FP8 | Apache-2.0 | Same arch family as current 9B (drop-in LMDeploy); Qwen prose is "dry/STEM" | LMDeploy |

### Medium 8–16B — subagent B (HF + 37-model narrative sweep + community threads)

| Model | Params/arch | Ctx | Quants | License | Pros / Cons | Backend |
|---|---|---|---|---|---|---|
| **google/gemma-4-12B-it** | 12B dense | 256K | QAT INT4 w4a16, GGUF, no AWQ | Apache-2.0 | Gemma-family creative-writing reputation; strong 128K retention (8-needle MRCR 43%); multimodal overhead | llama.cpp |
| google/gemma-4-26B-A4B-it | 26B/4B act MoE | 256K | QAT INT4 (~13GB!) | Apache-2.0 | Best prose-per-VRAM (4.66 narrative sweep, beats 31B); QAT INT4 fits ONE V100; RP-favorite uncensored quants (HauhauCS) | llama.cpp |
| **anthracite-org/magnum-v4-12b** | 12B dense (Nemo base) | 128K (rope 1M) | GGUF Q4-Q8, EXL2, fp8, W8A8 | Apache-2.0 | **Community's most-recommended prose/RP model**; "Claude-grade prose"; SillyTavern templates; Mistral arch = 1-line lmdeploy patch | LMDeploy (1-line patch) |
| mistralai/Ministral-3-8B-Instruct-2512 | 8.4B | 256K | GGUF, FP8 | Apache-2.0 | **4.76 narrative sweep — best ≤12B**; safe generalist floor; excellent instruction/JSON | llama.cpp |
| mistralai/Mistral-Nemo-Instruct-2407 | 12B | 128K | many | Apache-2.0 | Magnum's base; good prose control model | LMDeploy (patch) |
| puwaer/Doujinshi-14b-roleplay | 14B (Qwen3) | 40K | GGUF only | Apache-2.0 | Popular RP finetune; Qwen base prose is dry; GGUF-only | llama.cpp |
| google/gemma-3-12b-it | 12B | 131K | INT4 official | gated | Top creative-writing pick, flowery-tendency | llama.cpp |

### Larger (future) — subagent C (factual verification + r/LocalLLaMA)

| Model | Params/arch | Ctx | Quants | License | Notes | Backend |
|---|---|---|---|---|---|---|
| **Qwen/Qwen3.6-35B-A3B-abliterated-AWQ** | 35B/3B act MoE | 262K | AWQ INT4 (~23GB), GGUF | Apache-2.0 | REAL. Agentic-coding focus but our size-gap test; INT4 → tensor-parallel 2 | LMDeploy ✅ (already on disk) |
| zhipu GLM-5.2 | ~50B+ | — | — | — | **Top open-weight creative-writing (EQ-Bench CW v3 Elo 1720)**; too big/arch-unsupported for now | n/a |
| Kimi K2 / K2.6 | 1T MoE | — | — | — | Second-best writing; far too big | n/a |
| genevera/abhishekchohan Qwen3.6-35B-A3B-Abliterated-AWQ | " | " | AWQ INT4 | Apache-2.0 | Two abliterated-AWQ variants exist; tiny/less-vetted repos vs HauhauCS | LMDeploy |

### Community favorites that DON'T fit the stack (subagent C, r/LocalLlama scan)
- GLM-5.2, Kimi K2 — top EQ-Bench prose, too large / unsupported arch.
- DeepSeek-V4-Flash — elite coder, explicitly weak prose nuance (fine as our cloud guide).
- Older gems (Miqu/Midnight-Miqu-70B, Magnum-72B, Fimbulvetr-11B-v2, Soliloquy-8B, Dolphin-Mistral-24B-Venice) — superseded by 2026 threads; some need 70B VRAM.
- Nemotron-3-Nano (30B-A3B / 4B) — coding-lean, LM-Studio small-class pick.

### Already on disk (models dir)
- `Huihui-Qwen3.5-9B-abliterated-AWQ-4bit` — current DSS model (abliterated ✓).
- `Qwen3.6-35B-A3B-abliterated-AWQ` — the large test, downloading done (23.7GB).
- `Nex-N2-mini-AWQ-INT4` — IDENTICAL config to Qwen3.6-35B-A3B (40L/256E/8tok/262K): likely the same model; do NOT test both.
- `ornith-1.0-9b-Q6_K.gguf` — Qwen3.5-dense 9B GGUF (llama.cpp); abliterated/heretic community variants exist; coding/agentic-leaning lineage.
- `Llama-3.2-1B-Instruct-AWQ`, `Nandi-Mini-600M` — tiny controls.

## Test slate (final)

Priority order — all opt-in via `CMD_FLAGS_m.txt` (model + `--tensor-parallel 2` for the 35B):

1. **Qwen3.6-35B-A3B-abliterated-AWQ** — LMDeploy tp=2, in progress. Answers the
   size-vs-quality gap. Note: occupies both V100s → soak embeddings on GPU 0
   (`--embed-device cuda:0`).
2. **Magnum-v4-12b** — LMDeploy via the 1-line Mistral whitelist patch
   (`MistralForCausalLM='llama'`). Community prose champion, no rebuild.
3. **Gemma-4-E4B** — llama.cpp GGUF (Option C fallback). Small-class prose champion.
   Longer-term: the Gemma4 LMDeploy PR (see lmdeploy_gemma4_pr_plan.md).
4. Control: **Qwen3.5-4B** (LMDeploy) if a same-arch small baseline is wanted.
5. **Qwen3.8-27B** — new release expected 2026-08-14 (2 days). Verify the HF arch + LMDeploy
   whitelist support when it drops; if `Qwen3_*ForConditionalGeneration`-compatible it runs on
   the current engine (likely tp=2, same V100 story as the 35B). Next candidate after Magnum.

Deferred: Gemma-4-26B-A4B (needs llama.cpp or the PR; best prose-per-VRAM when it lands),
LFM2.5-8B-A1B (license + llama.cpp + prose unproven), Gemma-3-4B roleplay-tuned (gated,
llama.cpp), ornith GGUF (llama.cpp; verify prose quality first).

## Progress
- 2026-08-11: 3 research subagents + arch verification + feasibility subagent done.
- 2026-08-12: Qwen3.6-35B-A3B server switch (tp=2; CPU-patch gated to tp=1). Found + fixed the
  generate_with_streaming drain race (truncated every response under concurrency) + json_repair
  fallback + negative-verdict handling. **Completed 35B soak (seed 79, cozy_mystery): 4/5 final
  overview (vs 3/5 ceiling on the 9B), avg style 5.0, fidelity 4.5, retention 31.8%, ~39 min.
  Remaining gap: notes stored but never surfaced (recall 0%).** Harness now resolves the local
  model id from /v1/models (no more stale 9B label).
- Q-P3 memory-surfacing fix landed (2026-08-12): reply prompt now directs DSS to draw on its
  stored characters/items/events; verified on a 3-turn live smoke (seed 8001) — replies now
  reference established details across turns. Needs a fresh full run to measure recall impact.
- Qwen3.8-27B added to slate (release 2026-08-14); verify arch/whitelist when it drops.
