# DSS Retrieval Plan

Scope: the read side. How stored state and message history become the context the writer sees.

- What DSS stores: `quality_enhancements.md`.
- Speed and call counts: `soak_optimization_plan.md`.
- Engine and UI boundary: `headless_engine_ui_plan.md`.
- How DSS compares with Smart-Memory, TunnelVision and Talemate: `memory_comparisons.md`.

Evidence: a static read of `context_retriever.py`, `entity_graph.py` and `context_engine.py` at `a3c64b4` (2026-11-04), plus the reference systems in `memory_comparisons.md`. There are no retrieval-only measurements yet, which is why R0 comes first.

## 1. Summary

| Item | What | Extra LLM cost | Priority |
|------|------|----------------|----------|
| R0 | Retrieval eval: replay only, no LLM | 0 | First |
| R1 | Generic subject path and loud failure | 0 | High |
| R2 | Entity resolution in the retriever (design in quality 3.3) | 0 | High |
| R3 | Message ranking: dedupe against the live window, blended recency, MMR | 0 | High |
| R4 | Lexical (BM25) fusion and optional cross-encoder rerank | 0 (small local model) | Medium |
| R5 | Entity-card embeddings as extra graph seeds | 0 (embedder only) | Medium |
| R6 | Relevance feedback from injected items | 0 | Low to medium |
| R7 | Budgeted assembly with provenance | 0 | High |
| R8 | One shared scorer | 0 | Medium |
| R9 | Intent-aware routing for broad recap queries | 0 | Low |

## 2. Current pipeline (as read)

1. **Seeds.** Scene characters and elements from `current_scene`, plus names matched in recent text (`_extract_character_names`, `_extract_element_names`, regex).
2. **Graph.** `entity_graph` unified aggregation with decayed traversal (`DecayConfig`) and convergence, producing `unified_*` scores.
3. **Per-subject selection** in `retrieve_context`: characters, groups, elements (scene elements plus graph-scored extras, capped at `_MAX_CONTEXT_ELEMENTS = 25`), events (`_select_relevant_events` plus graph picks), arcs, chapters, then a `_recent_state` annotation.
4. **Messages.** `_build_retrieval_anchors`, then `_fanout_query_messages(n_results=5)`, then `query_similar` (30 candidates, `all-mpnet-base-v2`), then a recency-band re-rank, then a per-message dedupe that prefers summary nodes.
5. **Render.** `context_engine` walks `_context_order` from `format_templates.json`. Each section reads `getattr(retrieval_context, attr, None)` and is skipped when the value is `None`.

Not reviewed: the internals of `_unified_entity_aggregation`, event selection, and the rendering budget.

## 3. Principles

- **No LLM calls on the read path by default.** An LLM-navigated lookup (TunnelVision) can be an optional mode later.
- **Deterministic and replayable.** The same transcript and state give the same context. R0 depends on this.
- **Budget is a parameter.** The 40k standing rule (`soak_optimization_plan.md` §35) is enforced in one place, not in every template.
- **Every injected item can say why it is there.**
- **Weights live in config and are tuned against R0.** No unexplained constants in code.
- **Keep fixes that were measured.** The recency band exists because pure similarity re-served a stale beat (the pact-hall beat stayed a top hit at consecutive checkpoints). Any ranking change must keep that case passing.
- **Degrade, never fail.** Optional components (reranker, embedder) fall back to the simpler path and report once.

## 4. Items

### R0 Retrieval eval (do first)

Problem: the RAG path has never been properly exercised (`soak_optimization_plan.md` §20.3), and end-to-end judge scores mix writer quality with retrieval quality, so there is no way to tell whether a retrieval change helped.

Design: replay a recorded transcript with its per-turn state through `retrieve_context` only. Gold lists come from the planted notes the soak spreadsheets already carry (who knows a secret, a promise due at a scene, a technique learned at a turn, a rule stated once; see `quality_enhancements.md` section 9). A synthetic multi-scene history, as proposed in §20.3, works too. Metrics per checkpoint:

| Metric | Meaning |
|--------|---------|
| Recall | Share of gold facts present in the rendered context |
| Tokens per gold hit | Precision per injected token |
| Live-window duplication | Share of RAG-selected messages already in `last_x_messages` or the rolling summaries |
| False-seed rate | Entities seeded by name matching that no gold mention supports |
| Stale re-serve | A chunk served at consecutive checkpoints with no new mention of its entities |
| Knowledge render | Correct per-character stances (`quality_enhancements.md` 4.2) |
| Cost | Retrieval wall time and rendered tokens |

Cost: no LLM calls, so a run takes minutes. Exit criterion: baseline numbers recorded before R1 to R8 land.

### R1 Generic subject path and loud failure

Problem: `retrieve_context` has bespoke blocks for characters, groups, elements, events, arcs and chapters. A schema subject without one gets no `RetrievalContext` field, and `context_engine` skips it silently. That is how the Knowledge gap went unnoticed, and it conflicts with `quality_enhancements.md` section 2 ("no hardcoded subject structures").

Fix:

- **Default selector** for any `_context_order` subject without a bespoke one: graph-scored entries intersected with scene relevance, capped by a per-subject cap read from the schema, rendered with the subject's format template.
- **Generic storage.** Keep results in a `subjects: dict[str, dict]` map on `RetrievalContext`, and make the `getattr` lookup fall back to it.
- **Loud failure.** A one-time warning when a `_context_order` type resolves to `None`, and a hermetic test that every `_context_order` type in the example `format_templates.json` yields a non-`None` value for a populated fixture.
- **Per-subject injection policy.** A schema-level flag, `context: always | retrieved` (Luker's `alwaysInject`), so small stable subjects such as rules are injected every turn and leave the retrieval pool, while large subjects are retrieved. Luker also buckets recall candidates by type, because a shared top-K fills with the most numerous type (events) and starves the rest. The per-subject caps here already avoid that; keep them as floors in R7.

### R2 Entity resolution in the retriever

Design and evidence: `quality_enhancements.md` 3.3. Retriever-side work:

- Replace `_create_character_patterns`, `_create_element_patterns` and the unescaped `re.search(name, ...)` calls with the shared resolver.
- Use match tiers as seed weights in the graph aggregation.
- Do not seed on an ambiguous partial token.
- Test cases: `The Vault`, `Ray` against `array`, `Kit` against `kitchen`, a name with parentheses, two characters sharing a surname.

### R3 Message ranking

Current behavior (`query_messages`, `_fanout_query_messages`): 30 candidates (`max(n_results * 6, 30)`), sorted by `(band, -message_idx, summary_pref, similarity_position)`, where band 0 is within `_RAG_RECENCY_MESSAGES` (15) of the newest message. Consequences:

- Within a band the order is newest-first. Similarity only breaks ties between nodes of the same message. The docstring says the raw similarity order is kept within a band, and it is not.
- Any candidate in the last 15 messages outranks every older one regardless of score. With 5 results, an older and highly relevant message appears only if fewer than 5 recent candidates exist.
- Recent messages are probably already in the prompt (`last_x_messages`, rolling summaries), so the recent band may mostly return duplicates. Measure this with R0 before changing anything.

Changes, each gated by R0:

1. **Dedupe against the live window.** Drop candidates already present in the injected window before ranking.
2. **Blended recency.** Replace hard banding with an additive score: `distance + w_recency * age`, with `age` normalized from 0 (newest) to 1 (oldest) within the candidate set. Talemate's default `w_recency` is 0.1.
3. **MMR.** Skip a candidate whose cosine similarity to an already selected one exceeds a threshold. Talemate's default is 0.92.
4. **Recently-served penalty.** A chunk injected in the last N turns needs a higher score to be served again. This targets the pact-hall case directly, and it is what allows the hard band to go.
5. Keep the summary-node preference and the per-message dedupe.
6. **Score floor.** Weak matches are omitted instead of padding the result to 5.
7. **Age buckets instead of a blend (alternative to 2).** OpenVault splits candidates into old, mid and recent buckets, guarantees each a minimum share (20%, per its docs) and fills the rest by score. It attacks the same problem (old memories never surfacing) with less tuning than a recency weight. Choose between 2 and 7 with R0.
8. **Two-pass scoring.** OpenVault scores everything cheaply (lexical plus decay) and computes embedding similarity only for the top ~200. Worth it if the chunk pool grows large.

Fix the docstring whichever way the code goes.

### R4 Lexical fusion and optional rerank

- **BM25 over message chunks,** fused with the vector ranking by reciprocal rank fusion (k = 60). It targets proper nouns, code names and item names, where a 2021-era sentence embedder is weakest. SQLite FTS5 avoids a new dependency; check that the bundled SQLite supports it.
- **Optional cross-encoder rerank** of the top 30 candidates, keeping 5. Talemate ships this opt-in with `cross-encoder/ms-marco-MiniLM-L-6-v2` and falls back to the unranked list if the model fails to load. `scripts/cross_encoder.py` already exists in this repo; check what it found. Run it on CPU so it does not take VRAM from the LLM server.
- **Do not swap the embedder** (`all-mpnet-base-v2`) without an R0 comparison. A better benchmark score elsewhere does not imply better recall on these stories.
- **Precedents for the lexical pass.** OpenVault blends BM25 and cosine similarity with a tunable alpha and boosts exact-phrase and entity tokens. Graphiti runs BM25, cosine and graph expansion together and picks a reranker per search recipe (reciprocal rank fusion, MMR, cross-encoder, node distance).

### R5 Entity-card embeddings as seeds

Entities that matter but are never named ("the heist plan" for `Vault Blueprint`) are invisible to name matching. Embed `name + one-line summary` for each entry (characters, elements, groups, events), keyed by a hash of the text so only changed entries re-embed. At retrieval, the nearest cards to the anchors become extra seeds with weight `similarity * 0.5`, and the graph traversal expands them as usual. Vector seeds into graph expansion is the usual GraphRAG shape. Verify with R0 gold items that use unnamed references. Store the embedding model id with the vectors and rebuild when it changes: OpenVault validates a model fingerprint on load and re-embeds on a mismatch, which prevents silently mixing vector spaces after a model swap.

### R6 Relevance feedback

Track whether each injected item was used in the writer's reply: a name or alias hit, or embedding similarity above a threshold. Keep a small counter in a private field (like `_history`). Use it to demote items that were injected many times and never used, and as offline data for tuning weights. TunnelVision keeps a per-entry feedback map and Smart-Memory keeps `retrieval_count`.

Caution: "not mentioned" is not "not useful", because continuity facts work silently. So this is a demotion hint with a floor, and items above an importance threshold are exempt. The direction is not settled: OpenVault does the opposite and slows the decay of memories that keep getting retrieved. Let R0 decide which helps.

### R7 Budgeted assembly with provenance

Replace the single large `RetrievalContext` handoff with a list of blocks:

```
ContextBlock:
  subject, key, text
  tokens
  score
  reason      # seed:name | alias | hop 2 via X | rag:0.71 | scene cast
```

One allocator enforces the budget per purpose (writer context, DSS call) and drops the lowest-scoring blocks first, above each subject's floor. This puts the 40k rule in one place. Provenance also feeds the UI's "why was this retrieved" view (`headless_engine_ui_plan.md` P2). Keep `RetrievalContext` as a view built from the blocks so existing templates keep working.

Scoping the context that DSS update calls receive is the same allocator with a different purpose. See `soak_optimization_plan.md` §36.

### R8 Shared scorer

Knowledge selection (`quality_enhancements.md` 4.2), thread ranking (7.2), rule scoping (4.1) and message ranking all need a score built from importance, recency or staleness, graph proximity and similarity. Build one scorer module with weights in config instead of an ad-hoc function per consumer. Smart-Memory's `hybridScore` is the cautionary example: linear weights with hand-set constants (100, 60, 30, 80, 40, 50) and no eval behind them. Here every weight is tuned against R0 and recorded in config with the run that justified it.

### R9 Intent-aware routing (low priority)

OpenVault classifies a query as broad ("what has happened so far") or local (a scene question) and, for broad ones, injects the global world summary instead of scene-level memories. DSS has the pieces: the canon synopsis (`canon_synopsis_p0.md`) and the archives. A cheap rule with no LLM call could route recap-style user messages to the synopsis block. Do it only if R0 shows recap questions failing.

## 5. Rollout

0. R0 baselines.
1. R1. Small, and it prevents the next silent drop.
2. R2, together with the Resolution probe (`quality_enhancements.md` section 9).
3. R3 steps 1 and 4 first (dedupe, recently-served), measured. Then steps 2 and 3.
4. R7. It also unlocks `soak_optimization_plan.md` §36.
5. R4 and R5, only if R0 shows the gap they target.
6. R6 and R8, then R9 if R0 shows recap failures.

## 6. Not planned

- **LLM-navigated retrieval** (TunnelVision's tree walk). It costs a call per turn, and DSS already averages about 30 calls per turn. Revisit only as an opt-in mode for explicit recall questions.
- **Presence-derived knowledge** (Talemate). It needs presence metadata on every message and heavy invalidation machinery. The schema-driven edges in `quality_enhancements.md` 4.2 are the chosen design.
- **Swapping the vector store.**

## 7. Open questions

- How much of the RAG-selected messages duplicates the live window (R3 step 1)?
- `_safe_subject` silently drops a whole subject when a name raises `re.error`. Should it warn visibly (`quality_enhancements.md` 3.3)?
- What did `scripts/cross_encoder.py` find, and was it ever run on a long-horizon history?
- Once seeds are cleaner (R2), do the `DecayConfig` weights need retuning? They may be compensating for noisy seeds today.
- Where do the `_unified_entity_aggregation` thresholds come from, and are they tuned per story type?
