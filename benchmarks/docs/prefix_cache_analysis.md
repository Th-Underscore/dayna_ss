# Prefix Cache Stability Analysis — sci_fi_heist__d9b155be (300-turn soak)

## Summary

**The DSS prompt prefix IS architecturally stable intra-turn.** Context retrieval (`retrieve_and_format_context`) runs exactly ONCE per turn, before all ~25 DSS calls. The resulting `custom_state["context"]` string (the system/base prefix) is immutable for the duration of the turn. Each DSS call receives the identical base prefix and only the instruction suffix differs. **This is not an architecture bug.**

The "forcing full prompt re-processing" the user observed in the second half of the run is a **llama.cpp/LMDeploy KV-cache capacity issue**, not a prefix-stability issue. The context window is 65,536 tokens; individual prompts reach 28–43K tokens (system + history + instruction). With 17–33 DSS calls per turn, the server's KV cache cannot retain the prefix across all calls — it evicts entries between calls, forcing full re-prefill. The fix is server-side cache tuning (larger `n_ctx`, `cache_type q8_0`, or reducing per-call prompt size), not an engine architecture change.

## Per-turn context sizes

| Turn | context.txt (chars) | instr_prompt.txt (chars) | context sha256[:16] | instr sha256[:16] |
|------|-------------------|------------------------|---------------------|-------------------|
| 0    | 4,975             | 4,937                  | bb27acc50c6ff000    | 6c3c52c30f514a2e  |
| 50   | 30,406            | 8,866                  | 498b9a4386281635    | 45ebce1665ac07b8  |
| 100  | 34,976            | 9,002                  | 087430943e678021    | 3420000341d7ba53  |
| 150  | 41,110            | 9,329                  | fdf4158d00d192d2    | f626ac2b2debc32f  |
| 200  | 38,788            | 8,880                  | 89e89be715fd24f6    | e989eab1aa210211  |
| 250  | 34,895            | 9,244                  | 30045b1fdc0b07e5    | 64cc195694f93a09  |
| 299  | 37,059            | 9,090                  | 5593a82294d7e2c8    | f912111105ec6880  |

The `context` field grows from ~5K (turn 0, pre-story) to a steady-state 35–41K chars (turns 50–299). The `instr_prompt` stays stable at 8.8–9.3K chars. The context hash is unique per turn (as expected — the story state changes between turns), but is a SINGLE value per turn.

## LLM call metrics (from result.json `local_usage`)

| Turn | calls | prompt_tokens | completion_tokens | max_prompt_tokens | ctx_size | full_history_tokens | wall_time_s |
|------|-------|--------------|-------------------|-------------------|----------|--------------------|
| 0    | 85    | 336,184      | 0                 | 6,515             | 65,536   | 406                | 518.0 |
| 50   | 33    | 618,961      | 0                 | 28,346            | 65,536   | 28,292             | 237.6 |
| 100  | 16    | 397,458      | 0                 | 33,501            | 65,536   | 56,322             | 144.3 |
| 150  | 17    | 582,545      | 0                 | 43,514            | 65,536   | 88,576             | 138.5 |
| 200  | 19    | 302,045      | 0                 | 25,625            | 65,536   | 123,898            | 150.9 |
| 250  | 19    | 398,124      | 0                 | 30,745            | 65,536   | 156,812            | 156.7 |
| 299  | 21    | 564,496      | 0                 | 35,374            | 65,536   | 185,471            | 167.6 |

**Key observation:** `full_history_tokens` exceeds `ctx_size` (65,536) starting around turn 100 (56K → approaching) and clearly exceeding it by turn 150 (88K). The server MUST truncate history to fit the window. The average prompt tokens per call (e.g., 582,545/17 = 34,267 at turn 150) is far higher than the instruction suffix alone (~2–3K tokens), confirming full re-prefill is happening on most calls.

## Call flow analysis

### The per-turn sequence (from `summarize_latest_state`, core.py:768-912):

```
1. prepare_context(user_input, state, history[:-1])
   └─ retrieve_and_format_context(state, history)     ← RUNS ONCE
      └─ builds custom_state["context"] (the base prefix)
      └─ builds custom_state["history"]["internal"] (the conversation history)

2. DataSummarizer(self, exchange, custom_state, ...)  ← constructed once

3. _process_all_subjects(data_summarizer, ...)        ← ~25 DSS calls
   For each subject (characters, groups, elements, events, current_scene, ...):
     DataSummarizer.generate(data_type, data, schema)
       └─ _update_recursive → _traverse_structure → _create_update_prompt
          └─ generate_using_tgwui(prompt, state=custom_state)
             └─ generate_with_sse(prompt, state)
                └─ sends to LLM: system=state["context"], history=state["history"]["internal"], user=prompt
```

### Critical finding:

- `retrieve_and_format_context` is called **exactly once** per turn (line 794, inside `prepare_context`).
- The `custom_state` dict (containing `context` and `history`) is **deep-copied once** (line 819) and passed to `DataSummarizer`.
- Inside `DataSummarizer`, `self.custom_state` is **never mutated** between calls. Each call to `generate_using_tgwui` passes the same `state` reference.
- The ONLY thing that changes per DSS call is the `prompt` argument (the branch-specific instruction, assembled by `_create_update_prompt`).

### Therefore:

**The base prefix (system context + history) is byte-identical across all DSS calls within a turn.** Prefix caching SHOULD work at the architecture level. The cache misses are caused by the server's KV cache management, not by prefix instability.

## Why the cache still misses (server-side)

The problem is **KV cache pressure**, not prefix instability:

1. **Prompt size vs. context window:** Individual prompts are 25–43K tokens. The context window is 65,536 tokens. A single prompt consumes 40–66% of the window.

2. **Cache entry count:** With 17–33 calls per turn, the server needs to cache 17–33 separate prompt prefixes. Even if they share a common base, the KV cache must hold the full token sequence for each (llama.cpp's cache is per-sequence, not per-prefix by default).

3. **History growth:** `full_history_tokens` grows linearly (406 → 185,471 over 300 turns). Once it exceeds `ctx_size`, the server truncates. The truncation point shifts as history grows, changing the effective prefix between turns (though NOT within a turn).

4. **The "forcing full prompt re-processing" message** appears when the server's cache has evicted the prefix due to memory pressure — it has no cached KV for the prefix tokens, so it must recompute them.

## Recommendation

### Primary fix: Server-side (LMDeploy/llama.cpp configuration)

1. **Increase `n_ctx`** from 65,536 to 131,072 (or 196,608). This gives the KV cache room to retain multiple prompt prefixes simultaneously. On 2× V100 16GB, a Q4 model with 131K ctx should fit (KV cache at q8_0 is ~2-4 bytes/token/layer).

2. **Use `--cache-type-k q8_0 --cache-type-v q8_0`** (or `q4_0` for more savings). This halves the KV cache memory footprint, allowing more prefixes to coexist.

3. **Enable `--flash-attn`** if available for the model — reduces KV memory further.

4. **Consider `--n-predict` tuning** — if the server is configured to free cache between requests, ensure it retains cache for the duration of a turn (LMDeploy's session/queue management).

### Secondary fix: Engine-side (reduce per-call prompt size)

The `soak_optimization_plan.md` already identifies this: the per-call prompt embeds the ENTIRE subject map (via `_context_restatement`) + schema snippets + example JSON. At steady state, the instruction portion is 9K chars (~2.5K tokens) but the full prompt (system + history + instruction) is 25–43K tokens.

1. **Reduce `restate_map_threshold_chars`** — if the full map is >10K chars, it's already dropped to a compact roster. Verify this is happening.

2. **Trim history depth** — `full_history_tokens` at 185K is 3× the context window. The server truncates, but the truncation wastes prefill on tokens that will be cut. Cap the internal history to ~40K tokens (the most recent N exchanges) so the full prompt fits comfortably in 65K.

3. **Hoist static prompt components** (schema snippets, example JSON) into the system context — they're identical across all calls for a given subject type, so they belong in the prefix, not the suffix.

### What NOT to do:

- Do NOT change the context retrieval to run per-call — it already runs once per turn and the prefix is stable.
- Do NOT restructure the DSS call loop — the architecture is correct for prefix caching.