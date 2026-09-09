# LMDeploy Gemma 4 support — PR plan

Goal: add `Gemma4ForConditionalGeneration` (Gemma 4 text tower) to lmdeploy TurboMind
as an upstream contribution, so Gemma-4-E2B/E4B/12B/26B-A4B can run on the LMDeploy
loader (prefix caching + concurrency + tensor-parallel) instead of the llama.cpp
fallback.

Context: feasibility assessed 2026-08-11 by subagent against
`textgen/lmdeploy` (lmdeploy main @ v0.15.0+23, `ad242e61`). Bottom line:
**2–5 days, C++ kernel work required** — not minimal, but a genuinely valuable
upstream PR (no gemma support exists in TurboMind today).

## Facts (from google/gemma-4-e4b-it config.json)

- Wrapper arch `Gemma4ForConditionalGeneration` (text+vision+audio). Text tower =
  `model_type: gemma4_text` (42 layers). Loader must unwrap to text-only.
- Attention mix: 36 × sliding (window **512**) + 6 × full (layers 5,11,17,23,29,35).
- hidden 2560, 8 heads, **2 KV heads**, head_dim **256**, `global_head_dim` 512
  (= kv_heads × head_dim, cache is NOT shared; do not confuse with sharing).
- `num_kv_shared_layers: 18` — cross-layer **KV-weight sharing** (one k/v proj per
  group). Checkpoint layout unverified: does one shared tensor exist per group
  (loader must broadcast) or one per layer (nothing to do)? MUST be confirmed
  against the safetensors keys before coding.
- `final_logit_softcapping: 30.0` — post-logits tanh cap. No softcap exists in
  TurboMind. Either add to the C++ decode or apply in the python logits processor.
- RoPE: `rope_parameters` is a **dict keyed by layer type**
  (`full_attention`: partial_rotary_factor 0.25, `rope_type: proportional`, θ=1e6;
  `sliding_attention`: θ=1e4, default). Also per-layer attention split.
- gelu_pytorch_tanh, vocab 262144, tie_word_embeddings.

## Work items (in dependency order)

1. **Arch registration (python, ~30 min)**
   - `lmdeploy/turbomind/supported_models.py`: add
     `Gemma4ForConditionalGeneration='gemma4'`.
   - `lmdeploy/turbomind/converter.py` `get_registered_name` + INPUT_MODELS entry,
     mirroring `models/llama.py` `LlamaModel` (generic loader already reads
     q/k/v/o_proj, gate/up/down, norms, embed/lm_head).
   - Multimodal unwrap: reuse the wrapper→text unwrap already in
     `models/utils.py:17-22`.

2. **Per-layer rope/attention (python, ~1 day)**
   - `models/utils.py:86-102` `parse_rope_param` currently treats
     `rope_parameters` as rope_scaling; add dict-by-layer-type handling.
   - `models/utils.py:197-213` `make_attention_config`: set per-layer
     `window_size` (512) + full-attention layers get window 0/None (pattern:
     `models/gpt_oss.py:94`).
   - `builders/_base.py:50` `_act_type_id`: add gelu_pytorch_tanh.

3. **`rope_type: proportional` (C++ kernel, 1–3 days — the big one)**
   - `src/turbomind/models/llama/llama_rope.h:8-15`: new RopeType enum.
   - `src/turbomind/kernels/attention/rotary_embedding.h:93`: partial-rotary is
     already kernel-supported (`idx < param_.dim`); proportional extends it
     (scaling factor proportional to position). Needs a new kernel path + tuning.
   - Wire from `unified_attention_layer.cc` per-layer.

4. **Shared-KV broadcast (python, 0.5–1 day, layout risk)**
   - If the checkpoint holds one k/v proj per shared group: broadcast into each
     layer's fused `w_qkv` (`builders/attention.py:96-108`).
   - **Gate on verifying the safetensors key layout first.**

5. **Logit softcap (0.5 day, optional for v1)**
   - Add `final_logit_softcapping` to the python logits processor
     (`tanh(logits/30)*30`) as a first cut; C++ decode-side later if needed.

## Local build (for iteration)

- Toolchain proven: cmake 3.28.3, g++ 13.3, **nvcc 12.9** (`/usr/local/cuda`),
  python 3.12, ninja, 40 cores, 125 GB RAM, 212 G free on /mnt/c.
- `CMakeLists.txt:237-272` already includes sm_70 (V100) — prior local build of
  `_turbomind.cpython-312.so` exists (Jun 9) but points at the old dir; fresh
  configure needed. Build time ~45–90 min.
- Iteration loop: edit `.cc`/`.cu` → rebuild (incremental, kernels are the slow
  part) → load test on the 16GB V100s.

## Validation

- Unit: `make_attention_config` / `parse_rope_param` against the real config.
- Load + generate: bare `Pipeline(model, backend_config=TurbomindEngineConfig(tp=1))`
  for E4B, then tp=2.
- Parity: compare a fixed prompt's logits/softmax vs transformers reference for a
  few sliding and full-attention layers.
- Backend regression: existing Qwen3.5-9B + Mistral-Magnum still load.
- Multi-size: E2B (5.1B), E4B, 12B, 26B-A4B (MoE — 26B needs its own expert
  handling, separate PR; v1 scope = dense E2B/E4B/12B).

## Contribution shape (upstream)

- Single PR to InternLM/lmdeploy, small commit series per work item.
- Include the gemma4_text config fixture + a smoke test.
- Coordinate with any upstream Gemma-3 remnants (none in TurboMind).
- ETA realistic: v1 dense support 2–5 days incl. build; MoE (26B-A4B) a follow-up.

## Fallback (until merged)

- Gemma-4 runs via llama.cpp loader (GGUF) — Option C. ~half eval speed, order of
  magnitude lower prefill. Fine for the soak, not for production.
