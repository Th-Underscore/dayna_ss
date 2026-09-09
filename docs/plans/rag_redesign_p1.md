# RAG Redesign P1 — Pointed Fan-Out + Event-Bounded Retrieval

*Status: DESIGN 2026-09-07 · **Phase 1 IMPLEMENTED 2026-09-08** (uncommitted, +239 lines
across `context_retriever.py` + `context_engine.py`) · verification GREEN on 4dbf1435 turn-84
(6 anchors, mean pairwise cosine 0.658 vs old 1.00000 collapse, all markers present, 0%
boilerplate). Formal `bench_embed_model.py` A/B gate pending.*
*Gate: the collapse below is measured, not hypothesized. Phase 1 is green before anything
else is spent.*

## Problem (measured)

`retrieve_context` (context_retriever.py:1219) builds
`context_to_search = current_context + "\n" + "\n".join(last_x_messages)` (line 1234) — a
~14.7k-token rendered-context blob — and feeds it as ONE query to
`query_messages(context_to_search, n_results=5)` (line 1383 → 483). The embedder is
all-mpnet-base-v2 (384-token cap, 768-dim). Truncation keeps the first 384 tokens = 100% static
system-prompt boilerplate. Every story marker (Handler, Juno, chip, ledger, "grey coat",
Exchange) sits past the cut. Result (bench_embed_model.py): **cross-turn query-vectors at cosine
1.00000** — total collapse. Cosine ranking is noise; only the recency band (lines 519-522) does
any real work.

The model is NOT the bottleneck: corpus side is 0/168 nodes truncated (max 202 tokens), ~148ms/
encode on CPU. The bug is **query construction**, and no model swap touches it.

## Research consensus (8 repos, 2026-09-07)

Cloned + code-mapped under /tmp/opernode/: superbooga (superbig origin), Smart-Memory
(senjinthedragon), SillyTavern base + vectors, st-memory-enhancement (muyoou),
Extension-Summaryception, SillyTavern-MemoryBooks, SillyTavern-STARmem, TaleMate.

1. **No project embeds a large context blob as one query.** All use a SHORT query:
   - superbooga: the isolated `### Instruction` line only (alpaca.py:51, never full context).
   - SillyTavern vectors: last-N raw messages (vectors/index.js:901 `getQueryText`).
   - Smart-Memory: user-typed text in `/sm-search` only (index.js:2239).
2. **Fan-out is proven and is the direct cure for the collapse.** superbooga `chromadb.py:307-326`
   loops N short `search_strings`, each `n_results=ceil(N/len)`, pools. TaleMate
   `rag.py:285-342` (multi_query) + an LLM synthesizing up to 3 "short, focused keyword phrase"
   queries.
3. **Event/structure-bounded, importance-gated, DETERMINISTIC (no embeddings) exists — STARmem.**
   A 5-tier ladder, 100% deterministic (BM25+, not embeddings): Tier-2 top-K → top-3 become graph
   *seeds* → Tier-3 beam-search (2 hops, width 5) over entity edges; a regex *intent* classifier
   (factual/relational/temporal) reweights which edge types traverse; per-hop beam score
   `exp(λ1·edgeType + λ2·BM25)`; final **multiplicative** rescore
   `score = BM25 × (1+importance/100) × recency(Δt) × maturity` (multiplicative ⇒ low-relevance
   zeroes regardless of importance). Plus a BM25-free *floor* (recency×(1+imp/100)×maturity) and a
   decay-immune `Infinity`-scored working buffer.
4. **Pointedness is solved by a hybrid score, not narrower embeddings.** Smart-Memory
   `hybridScore` (memory-utils.js:573): 5-signal NON-embedding blend — entity-mention-overlap
   (100×) as the FIRST-CLASS signal (which of a memory's registered entities the turn mentions),
   +60× arc-cosine +30× temporal +trigger(80/40) −50 contradiction. This is the exact fix for
   "a bare entity name is too broad."
5. **No project puts a small model on the RETRIEVAL path.** All keep retrieval 100% deterministic.
   The small/fast model appears only on the WRITE side: Smart-Memory's 5-token scene-break gate +
   model-test gating (model-test.js); SillyTavern vectors' 3-endpoint summarizer selector
   (vectors/index.js:383); Summaryception's independent summarizer connection with
   snapshot/disable/restore of roleplay presets (index.js:818-820) so a weak model does clean
   extraction.
6. **Compact structured chunks > prose.** The collapse-immune projects store compact structured
   units (st-memory-enhancement: numeric-tier tables; MemoryBooks: entry text + keyword sets —
   "embed only the compact structured entry, never the rendered conversation", en.md:2413).
   Structure/compactness of the stored unit is what makes injection safe.

## Decisions

### 1 — Fan-out replaces the single-blob embed (the cure)

`query_messages` (line 483) already fetches 6× candidates (`candidate_n = max(n_results*6, 30)`)
and recency re-ranks — the plumbing is sound. The defect is upstream: one collapsed query in.
Replace the single embed at line 1383 with **N short anchor queries**, each **< 300 tokens**
(within the 384 window ⇒ each a genuinely discriminative vector). Pool → per-message dedup →
existing recency re-rank (lines 514-523) → budget. **N is user-configurable (default 3-5).**
The collapse (cosine 1.00000) is structurally eliminated, not worked around.
Lift: superbooga `chromadb.py:307`, TaleMate `rag.py:285-342`.

### 2 — Deterministic anchor construction (the "elegant aggregator", no new model)

Every input to an anchor is already computed deterministically each turn:
- **cast** — `current_scene.now.who.characters` (line 1227-1231),
- **mentioned entities** — `_extract_character_names` / `_extract_element_names` (1235/1248),
- **event anchors** — the on-disk event store (below),
- **stale roster** — `compute_stale_entities` (canon P0),
- **active chapter/arc spans** — already treated as deterministic truth.

Build each anchor from a *high-relevance subset* of these, each as a short string:
`<entity/event title> + its scoped cast + story direction`, framed as **"what is new/changing
within <scope> relative to <loaded state>"** — the *delta*, never a re-surfacing of state
already carried by general_info, chapter/arc digests, and the entity graph. A bare entity name
is too broad; an entity **plus its event bounds** is the narrowest honest unit. This is the
user's "high-relevance aggregator" — it needs no model, ~0.3s, zero VRAM.

### 3 — Event-bounded deterministic retrieval (no embeddings)

A new retrieval path orthogonal to (not competing with) the semantic fetch, running every turn:
- pull **summaries of events whose `importance.score` is in the medium/high tier**,
- for the **high-relevance** events, list the **per-message summaries within
  `[start._message_node … end._message_node]`**.

This is directly computable today from the on-disk event store (memory #287): each event already
carries `importance{score,reason}`, `summary`, `start`/`end` with `_message_node` anchors, and
per-participant importance. It needs no embeddings, is verbatim-safe (selection over stored
summaries, not transforms), and its output flows through the existing hygiene layer for dedup.
This is the user's second explicit request.

### 4 — Small helper model: read/write split, write-side only

The RAG fix (Decisions 1-3) has **zero hard dependency** on a small model; it only *upgrades*
when one is present. When it is:
- **small model (0.8-4B), read-side only** — synthesize the N anchor queries (its best job:
  1-5 entity-anchored natural-language queries at a fraction of the 27B's latency). Optional;
  deterministic anchors (Decision 2) are the fallback.
- **27B, write-side only** — gates, branches, importance, archives. A mis-gate forces a false
  archive; the small model's decision error-profile is where you lose, so those calls never leave
  the 27B.
- **VRAM is BINDING (corrected 2026-09-08)**: 2× V100-SXM2-**16GB** (32GB total), most of it
  pinned by the LMDeploy LLM server + tgwui bot. A 0.8-4B helper model can only co-reside in
  the residual free partition after those are accounted for — this is a **hard** constraint,
  not a soft one. The split above (read-side small / write-side 27B) is the correct
  dependency architecture; the VRAM question is *when* the helper can be scheduled, not
  *whether* the split enforces dependency. DSS must degrade to zero (deterministic anchors
  only) when no VRAM is available for the small model.
Lift: Smart-Memory 5-token gate + model-test gating; Summaryception preset isolation.

### 5 — Hybrid scoring + injection (borrow, don't rebuild)

Extend the existing recency re-rank into a **hybrid score** (Smart-Memory `hybridScore`,
memory-utils.js:573): entity-mention-overlap as a first-class signal (scores *which of an
entity's registered names the turn mentions*, normalized by entity count) + recency + a
**multiplicative** importance term (STARmem: low-relevance ≈ 0 zeroes regardless of importance;
stale ≈ 0 regardless of importance). For injection, adopt SillyTavern's budget-aware greedy
fill (priority-ordered, running token sum, stop at budget, per-entry override) + per-slot
positional dispatch, rather than one lump.

## Rollout (each phase gated on the previous)

- **Phase 1** — Decisions 1 + 2. No new model. **IMPLEMENTED (2026-09-08, uncommitted).**
  `StoryContextRetriever.__init__` takes `rag_anchor_count` (threaded from
  `dss_config.json:rag_anchor_count`, default `_RAG_ANCHOR_COUNT_DEFAULT=5`). `_build_retrieval_anchors`
  constructs up to N short POINTED delta-framed anchors (event anchors → mentioned-entity →
  cast → stale-roster → recency fallback), each clamped to `_ANCHOR_CHAR_BUDGET=240` chars
  (well inside the 384-token window). `_fanout_query_messages` replaces the single collapsed-blob
  embed at the old line-1383 call site: each anchor queried independently, pooled, per-anchor
  recency re-rank, with a **legacy single-blob fallback when zero anchors are available** (never
  worse than before). Verified GREEN on 4dbf1435 turn-84: 6 anchors, mean pairwise cosine **0.658**
  (was **1.00000** = total collapse), all story markers present (Juno/Handler/Second Runner/chip/
  ledger/grey-coat), 0% boilerplate, 0 markers in the old kept window. Formal A/B via
  `bench_embed_model.py` (markers-survive + cross-turn-cosine + recall top-K) is the remaining
  gate before merge. **Cast display-name dedup** added to the anchor path (presentation-only,
  order-preserving, no store mutation) to prevent a live cast rendering "Second Runner" thrice
  from three distinct graph nodes sharing one display name.
- **Phase 2** — Decision 3. Event-bounded deterministic retrieval path.
- **Phase 3** — Decision 4. Small-model query-synth slot, read/write split, VRAM-gated.
- **Phase 4** — Decision 5. Hybrid scoring + injection merger.

## Model question (settled)

Keep **all-mpnet-base-v2**. Speed is already adequate (1-3 query encodes/turn ≈ 0.3-0.5s;
chunk-encode amortized against the 128:1 prefill:completion ratio). The axis that matters once
the query stops being collapsed is **discriminative quality**, not speed — MPNet's 768-d beats
MiniLM's 384-d on fine-grained event semantics, and MiniLM has a *128*-token cap, not 512.
Swapping spends the one axis that decides recall to buy ~100ms/turn. Not a trade the data
supports.

## Cross-reference — parallel track (interacts, does not block)

The **referent-resolution gate** (`referent_resolution_gate.md`) is an orthogonal,
parallel workstream landed in the same 2026-09-08 window. It targets the *write* side
(split-node / alias-poison / contamination at the branch-edit boundary) — a different
code path from this retrieval work. Two touchpoints to keep straight:

- The **cast display-name dedup** added to `_build_retrieval_anchors` (Phase 1, above)
  is a *presentation-only* anchor-string fix — it renders distinct same-named nodes
  without padding the anchor with repeats. It does **not** merge store nodes; that is the
  referent-gate's domain (H3 retargeting), which is the authoritative fix for the split
  `Second Runner` / `The_Second_Runner` class.
- The event-anchor + scoped-cast construction (Decision 2) is *read* of the same
  `events.json` / entity-graph store the referent-gate's parse-time retargeting
  *writes* to. They must not be entangled: retrieval reads whatever is on disk; the
  referent-gate decides what lands there. Ship them independently.

## Open questions

- ~~**Anchor count N default** — 3-5 proposed; user to confirm.~~ **RESOLVED**: default is
  `5` in code (`_RAG_ANCHOR_COUNT_DEFAULT`), user-overridable via `dss_config.json
  :rag_anchor_count`. Verified GREEN at N=6 (turn-84 produced 6 distinct anchors).
- **BM25 vs cosine for the deterministic ladder** — STARmem uses BM25+ (no embeddings). DSS has
  an embedding index but no BM25 index. Decide whether the event-bounded path (Decision 3) and
  any STARmem-style graph expansion adopt BM25 (new dependency, stdlib-portable) or extend the
  existing embedding index with the hybrid score (Decision 5).
- **Which small model** — 0.8-4B, read-side only, must survive model-test gating before being
  trusted with query synthesis. Candidate selection is a separate, later task.