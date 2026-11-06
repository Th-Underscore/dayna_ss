# Memory System Comparisons

Purpose: compare DSS with seven other memory systems, criterion by criterion, and point each gap at the recommendation that addresses it. Three were read first (Smart-Memory, TunnelVision, Talemate). Luker, OpenVault, Graphiti and Summaryception were added on 2026-11-06.

Reference format: `quality_enhancements.md` is written `quality §x`, `retrieval_plan.md` as `retrieval R#`, `headless_engine_ui_plan.md` as `headless P#`, and `soak_optimization_plan.md` as `soak §x`.

This extends the Smart-Memory-only matrix in `docs/comparison.html` (referenced in `PROJECT_STATE.md`; it is not in the public clone, so I did not review it).

## 1. Systems and method

| System | Kind | Host | Revision read | Stars (2026-11-06) |
|--------|------|------|---------------|--------|
| DSS (dayna_ss) | Schema-driven structured story state | text-generation-webui extension (Python) | `a3c64b4`, 2026-11-04 | 6 |
| Smart-Memory | Tiered memory injected as text | SillyTavern extension (JS) | `a7f6c8e`, 2026-10-01 | 73 |
| TunnelVision | LLM-navigated tree index over lorebooks | SillyTavern extension (JS) | `cf84ad1`, 2026-10-11 | 14 |
| Talemate | Full roleplay and narrative application | Standalone (FastAPI and Vue) | `7db12b0`, 2026-10-27 | 432 |
| Luker | Typed memory graph with per-card schemas, LLM or vector recall, commit-log state | SillyTavern fork (JS) | `da96c88`, 2026-11-06 | 550 |
| OpenVault | Witness-filtered memories, hybrid retrieval, reflections, entity graph | SillyTavern extension (JS), upstream `vadash/openvault` v26.00 | `b6284c7`, 2026-07-20 | 47 |
| Graphiti | Bi-temporal knowledge graph for agents | Python library, server and MCP server (v0.31.2) | `747b480`, 2026-11-06 | 32.6k |
| Summaryception | Layered rolling summaries | SillyTavern extension (JS, v5.5.4) | `6a0afdc`, 2026-09-06 | 187 |

Method: shallow clones and a static read of the source, not the READMEs. Things I ran:

- **TunnelVision:** the vitest suite. 725 of 756 tests passed, 3 were skipped, and all 28 failures are in `tests/entry-manager.test.js`, which still assumes per-entry metadata in chat metadata after a refactor moved it to in-memory maps.
- **Smart-Memory:** `node --test` on `tests/memory-utils.test.js` and `tests/parsers.test.js` (75 of 76 pass), plus the golden-file harness (48 of 48 assertions).
- **Talemate:** not run. About 8.8k test functions in about 400 files were counted.
- **DSS:** not run. Timing figures come from a `deep_timing.py` report for `sci_fi_heist__26d93cf1`.
- **OpenVault:** the vitest suite. 1,184 of 1,184 tests pass in 64 files (the dependency install needed `--legacy-peer-deps`). I read its `AGENTS.md` files and `include/DATA_SCHEMA.md` more closely than its code, so its formulas come from its own documentation, checked against the source layout and tests. I first read the README of a further fork (`7n1m4/vault`) for the feature list and could not clone it; the code reviewed is upstream `vadash/openvault` (v26.00). The `Shaderx/openvault` fork shows an older version number (v23.00).
- **Luker, Graphiti, Summaryception:** not run. Luker's Memory Graph extension (about 33.7k lines, 52 test files) and Graphiti (about 40.8k lines of core code, about 65 test files, plus an evaluation harness that needs a database and an LLM) were read statically. Summaryception is one JavaScript file (2,779 lines) with no tests.

Limits: no system was run end-to-end against a model, so there are no cross-system speed or quality numbers. Everything here is a reading of code and should be treated as a hypothesis until `retrieval R0` and the `quality §9` probes produce measurements. Items marked "unverified" were not traced. Smart-Memory's golden replay uses recorded model output, so it checks parsing and not model quality. The four systems added on 2026-11-06 were read once and in less depth than the first four, so weight their entries accordingly.

### Gap scale

Gap is how far DSS is from the best other system on a criterion, measured in capability and not in effort.

| Gap | Label | Meaning |
|-----|-------|---------|
| +1 ▲ | Lead | DSS does something none of the others do, or does it better by design. |
| 0 ▯▯▯ | Par | Comparable capability. Differences are polish. |
| −1 ▮▯▯ | Slight | DSS has the capability. The best system's version is more complete or better tuned, and closing it is a local change. |
| −2 ▮▮▯ | Moderate | A capability the best system has is missing or only partial in DSS, and it affects results. |
| −3 ▮▮▮ | Far | An entire capability is absent in DSS, or the difference on a measurable quantity is an order of magnitude. |

Effort to close (rough): **S** is a few days in one module. **M** is one to two weeks, a new module or a cross-cutting change. **L** is several weeks or a new subsystem.

Confidence: **High** means both sides were read in code or measured. **Medium** means the DSS side was read in code and the other side partly from documentation, or nothing was run. **Low** means the rating depends on something unmeasured or unverified.

## 2. Scoreboard

| # | Criterion | Best now | DSS gap | Effort | Confidence | Recommendations for DSS |
|---|-----------|----------|---------|--------|------------|-------------------------|
| 1 | Knowledge and epistemics | OpenVault, Smart-Memory, Talemate | 0 ▯▯▯ (unproven) | S | Low | quality §4.2 Consumption |
| 2 | State representation and schema flexibility | DSS, Luker | 0 ▯▯▯ | M | Medium | headless P6, retrieval R1 |
| 3 | Entity resolution | Graphiti, Smart-Memory | −2 ▮▮▯ (write side 0) | M | High | quality §3.3, retrieval R2 |
| 4 | Structured (graph) retrieval | DSS (Graphiti closest) | +1 ▲ (unmeasured) | n/a | Low | retrieval R0, R5 |
| 5 | Text (message) retrieval | Graphiti, OpenVault, Talemate | −2 ▮▮▯ | M | Medium | retrieval R0, R3, R4, R5 |
| 6 | Supersession, dedup and change history | Graphiti | −1 ▮▯▯ | S | Medium | quality §3.1 follow-up, §3.4 |
| 7 | Write safety: user edits and locks | Luker, TunnelVision | −2 ▮▮▯ | M | Medium | quality §3.2, headless P1, P4, P5 |
| 8 | Rewind: swipes, edits and deletes | Luker, TunnelVision | −2 ▮▮▯ (latest turn 0) | S to M | Medium | quality §3.4, headless P1 |
| 9 | LLM cost per turn | Summaryception, Smart-Memory | −3 ▮▮▮ (raw calls) | L | Medium | soak §36 S1 to S5, retrieval R7 |
| 10 | Context budgeting and injection control | OpenVault, Smart-Memory | −2 ▮▮▯ | M | Medium | retrieval R7, R1, R3 |
| 11 | Summarization hierarchy | Summaryception, Luker | 0 ▯▯▯ | S | Medium | none new (`canon_synopsis_p0.md`) |
| 12 | Robustness and failure handling | OpenVault, TunnelVision | −2 ▮▮▯ | S to M | Medium | soak §36 S6 |
| 13 | Evaluation and tests | OpenVault, Graphiti, Talemate | +1 ▲ state eval, −2 ▮▮▯ retrieval eval | M | Medium | retrieval R0, quality §9 |
| 14 | User editability and UI | Luker, Talemate | −3 ▮▮▮ | L | High | headless P3 to P6, quality §3.2 |
| 15 | Portability | Graphiti, Talemate | −2 ▮▮▯ (0 vs the ST extensions) | M | Medium | headless P0 to P2 |
| 16 | Observability and debuggability | Luker, DSS (cost) | 0 ▯▯▯ | M | Medium | retrieval R7, headless P2, P3, soak §36 S1 |
| 17 | Temporal validity of facts | Graphiti | −1 ▮▯▯ | S | Medium | quality §3.1 follow-up, headless P1, P2 |
| 18 | Schema versioning and migration | OpenVault | −2 ▮▮▯ | M | Medium | headless P1, P6 |

**Gap map.**

- **Far (−3):** 9 (raw LLM calls) and 14 (UI).
- **Moderate (−2):** 3, 5, 7, 8, 10, 12, 15, 18, and the retrieval-evaluation half of 13.
- **Slight (−1):** 6 and 17.
- **Par (0):** 1, 2, 11, 16.
- **Lead (+1):** 4 and the state-evaluation half of 13.

The cheapest meaningful closes are 8 (nearest-prefix snapshot), 12 (a bounded transport retry), 17 (a turn-to-snapshot index) and 6 (`node` and `actor` on `_history`), each an S-size change. The two far gaps are both L.

## 3. Criteria

### 1. Knowledge and epistemics

Who knows what, and whether the writer respects it.

- **DSS:** a schema-driven `Knowledge` subject with `type`, `status` and `source`, and `knows` and `learned` edges with five stances, generated at one call per scene. Writer-side rendering was missing and has been fixed. The Leak probe has not been run against it yet.
- **Smart-Memory:** a per-character private map extracted at scene breaks (knows, suspects, believes falsely, unaware, concealing from a target). Existing entries go into the extraction prompt so the model can mark stale ones `[retire]` in the same call. Only the responding character's block is injected, inside a token budget.
- **TunnelVision:** no explicit model. A secret-tag convention exists (`SECRET_TAG_RE`) that I did not trace.
- **Talemate:** knowledge is derived from per-message presence (computed in code), plus explicit disclosures and per-character perspective views with provenance and 12 kinds of staleness check. Unknown presence is never treated as "nobody". It is opt-in (`perception` mode, next to `legacy`).
- **OpenVault:** each event records its witnesses and each character keeps a `known_events` list. Retrieval is filtered to what the point-of-view characters know (names and aliases expanded), and reflections are limited to the same view. A strictness setting controls how hard the filter is. Witnesses are recorded at extraction time, which makes this the cheapest design of the four to run; DSS's `knows` edges are richer but cost a call per scene.
- **Luker:** I found no per-character knowledge model. Nothing filters recall by who knows what.
- **Graphiti:** facts are global to a graph (partitioned by group), with no per-character perspective.
- **Summaryception:** none; the summary is the same for every character.

Gap: **0, par** ▯▯▯ (unproven). DSS has more stance types than OpenVault's witness lists and now a consumption path, but no probe has confirmed that the writer respects it.
Effort to close: S. Confidence: Low (unmeasured; the consumption fix is recent).
Recommendations: quality §4.2 Consumption (per-scene-cast stances, with `unaware` derived from missing edges); Leak and Knowledge render probes in quality §9.

### 2. State representation and schema flexibility

- **DSS:** a user-defined JSON schema (subjects, field types, `field_options` enums, templates, graph edges declared in the schema). New subjects need no engine code in principle; the retriever gap behind the Knowledge bug showed that this does not yet hold for reading (retrieval R1).
- **Smart-Memory:** fixed tiers, plus state cards whose fields are hardcoded per entity type (`STATE_CARD_FIELDS`).
- **TunnelVision:** native lorebook entries, a tree index, tracker entries, and arcs kept in chat metadata.
- **Talemate:** scene JSON, world-state entries, user-declared tracked states (a question answered every N turns), and layered history.
- **Luker:** node types and fields are defined by the user and saved with the character card. Each type has its own extraction instructions, extraction interval and an `alwaysInject` flag, all editable in the UI.
- **OpenVault:** a fixed schema (events, reflections, entities and edges, scene state) that is validated and versioned (criterion 18).
- **Graphiti:** developer-defined entity and edge types as Pydantic models with identity properties. Defined in code, not by the end user.
- **Summaryception:** none; layered text.

Gap: **0, par** ▯▯▯. The engine side matches Luker: user-defined types, fields and enums. The editing UI Luker has is counted under criterion 14.
Effort to close: M (the injection-policy flag and the schema editor). Confidence: Medium.
Recommendations: headless P6 (schema editor); retrieval R1 (per-subject injection policy). Protect the schema (quality §2).

### 3. Entity resolution

How text becomes an entity key.

- **DSS:** strong on the write side (normalized keys, alias match, lookalike-key retargeting in `parsing.py`). Weak on the read side: regexes on the full name plus first and last tokens, no word boundaries, unescaped `re.search(name, ...)` in element and group lookups, and aliases used only for groups. A regex error is caught by `_safe_subject`, which prints a traceback and substitutes an empty subject, so one bad name silently removes the whole elements or groups section for that turn.
- **Smart-Memory:** an entity registry with ids and aliases. Entity overlap in scoring matches a memory's entity names and aliases against the turn's mentions (exact lowercase match). Reconciliation handlers cover merges and type changes.
- **TunnelVision:** lorebook key matching through SillyTavern's activation plus its own scoring. I did not review its alias handling.
- **Talemate:** an entity lookup for off-stage characters mentioned in recent lines, by name matching, with an optional small decision model for pronouns, titles and nicknames that falls back to name matching.
- **Graphiti:** a staged pipeline. Candidate search (semantic, exact name, identity properties), then deterministic identity-property, exact and fuzzy matching (MinHash over name shingles, Jaccard at least 0.9, with an entropy guard that passes short or low-information names to the next stage), then an LLM call only for what is still unresolved.
- **OpenVault:** a merge leaves a redirect (`_mergeRedirects`, followed through chains with a cycle guard), so later mentions of the old name resolve to the canonical entity. Per its data-schema doc, merge rules depend on entity type.
- **Luker:** the AI revision workbench proposes duplicate merges as reviewable cards. I did not trace its matching.
- **Summaryception:** no entities.

Gap: **−2, moderate** ▮▮▯ vs Graphiti and Smart-Memory on the read side; 0 on the write side. The retriever lacks word boundaries, escaping, aliases for characters and elements, and any fuzzy stage, and its failures are silent. That produces false or missing seeds, not cosmetic misses.
Effort to close: M. Confidence: High (both sides read in code).
Recommendations: quality §3.3 (now including the staged resolver, redirects, type-aware thresholds and identity fields); retrieval R2; Resolution probe in quality §9.

### 4. Structured (graph) retrieval

- **DSS:** weighted multi-hop traversal over an entity graph with decay and convergence, per-subject caps, and a per-entity recent-state annotation. No other system here has an equivalent.
- **Smart-Memory:** an entity graph module (`graph.js`) feeds entity overlap in a hand-weighted linear score. Not reviewed in depth.
- **TunnelVision:** an LLM walks a collapsed tree overview and picks nodes, plus local scoring with tree-proximity heuristics.
- **Talemate:** world-state entries and knowledge edges; retrieval itself is vector-driven. Not reviewed in depth.
- **Graphiti:** hybrid search over a temporal graph (BM25, cosine similarity and breadth-first expansion), with a node-distance reranker that favors results close to a focal entity. Community summaries come from label propagation. It is the closest to DSS's traversal, but over generic facts.
- **OpenVault:** an entity graph with edges and community-based world summaries (about every 50 messages). Retrieval itself is memory-based, with graph entities used as query anchors.
- **Luker:** recall over typed nodes by LLM exploration or vector search. No weighted traversal found.
- **Summaryception:** none.

Gap: **+1, lead** ▲ by design, with Graphiti the closest. No other system has weighted, decayed multi-hop traversal. Unmeasured, and noisy seeds (criterion 3) may be hiding its quality.
Effort to close: n/a (measure it with retrieval R0). Confidence: Low.
Recommendations: retrieval R0 to measure it; retrieval R5 for non-named seeds.

### 5. Text (message) retrieval

- **DSS:** `all-mpnet-base-v2`, anchor fan-out, 30 candidates, a hard recency band (newest message index first within the band), and a per-message dedupe that prefers summary nodes. No lexical path and no reranker in the main path. The band was added to stop a stale beat being re-served, which is worth keeping.
- **Smart-Memory:** embeddings with a Jaccard fallback feed arc relevance and turn similarity inside a linear score, plus trigger-word bonuses.
- **TunnelVision:** keyword activation and scoring, with optional sidecar-LLM navigation.
- **Talemate:** ChromaDB with a distance cutoff, an additive recency weight (default 0.1), MMR (default 0.92), round-robin merging across queries, an entity lookup, and an opt-in cross-encoder rerank of the top 30 down to 5 that falls back cleanly if the model fails to load.
- **Graphiti:** BM25 and cosine similarity with a choice of rerankers (reciprocal rank fusion, MMR, cross-encoder, node distance, episode mentions), packaged as search recipes.
- **OpenVault:** two passes. A cheap lexical-plus-decay score runs over everything, then cosine similarity is computed for the top ~200. The score is `(base + alpha * vector + (1 - alpha) * BM25) * frequency`, where base is importance times exponential decay. Chronological buckets (old, mid, recent) each keep a minimum share and the rest is filled by score.
- **Luker:** RAG recall is an optional LLM query rewrite, then a vector pre-filter, then per-type bucketing (so a shared top-K is not filled entirely with events), then an optional per-bucket rerank. The default recall mode is LLM-driven.
- **Summaryception:** none; everything is injected.

Gap: **−2, moderate** ▮▮▯ vs Talemate, OpenVault and Graphiti. DSS has embeddings, anchors and a recency band. It lacks a lexical path, a reranker, MMR, de-duplication against the live window and a blended recency score; each of the three has most of those.
Effort to close: M. Confidence: Medium (OpenVault's formulas come from its documentation).
Recommendations: retrieval R0 (measure first), R3 (now with age buckets and two-pass scoring), R4 (BM25 and optional rerank), R5 (entity-card seeds).

### 6. Supersession, dedup and change history

- **DSS:** per-field `_history` of old and new values (capped at 20 per path), a hygiene pass, and canon-synopsis regeneration. No `node` or `actor` on entries yet and no contradiction flags.
- **Smart-Memory:** memory-level `superseded_by`, `contradicts` (a flat score penalty), `retrieval_count`, time scope and expiration, and embedding dedup with a Jaccard fallback.
- **TunnelVision:** periodic consolidation, compression and reorganization, entry versions, and a quality score per entry (specificity, freshness, retrieval rate, key coverage).
- **Talemate:** revisions with content digests and receipts, a changelog, and staleness checks on derived views.
- **Graphiti:** bi-temporal edges: `valid_at` and `invalid_at` (when a fact became and stopped being true) plus `expired_at` (when the system learned it was superseded). A contradicted fact is invalidated, not deleted, and each edge records its source episodes.
- **OpenVault:** multi-tier dedup of events and reflections (reject, replace or add, by similarity), retrieval-hit counters, and per-memory importance with decay.
- **Luker:** a commit log of operations is the source of truth (criterion 8), and the revision workbench merges duplicates with approval.
- **Summaryception:** none beyond promoting snippets upward.

Gap: **−1, slight** ▮▯▯ vs Graphiti. Field-level history is the most precise of the eight. It lacks `node` and `actor`, validity intervals (criterion 17) and contradiction flags.
Effort to close: S. Confidence: Medium.
Recommendations: quality §3.1 follow-up (`node`, `actor`, as-of reads) and §3.4; headless P1.

### 7. Write safety: user edits and locks

- **DSS:** schema validation and enum coercion. No user lock and no record of who made an edit.
- **Smart-Memory:** not reviewed in depth.
- **TunnelVision:** background automation never rewrites authored (`constant`) entries, per-book read and write permissions, tool-call confirmation, and out-of-character turns suppress memory writes.
- **Talemate:** revisions and a changelog. Not reviewed for per-field locks.
- **Luker:** every write goes through an op batch that returns applied and rejected lists and never throws. AI edits arrive as reviewable cards the user approves, and types can be marked persistent.
- **OpenVault:** the UI can edit, delete and merge memories and entities, through a repository layer.
- **Graphiti:** a library API with manual fact insertion. I found no per-field locks.
- **Summaryception:** a snippet browser with edit and delete per snippet; nothing protects an edit from later regeneration.

Gap: **−2, moderate** ▮▮▯ vs Luker and TunnelVision. DSS has schema validation and enum coercion. It lacks a user lock, an `actor` stamp, a rejected-operations report and an approval flow. Nothing breaks until a UI allows editing, which is why it is not rated lower.
Effort to close: M (stamping and locking are S; approval needs the UI). Confidence: Medium.
Recommendations: quality §3.2 (`user` rung, locked paths); headless P1 (single write path), P4, P5 (modeled on Luker's approval cards and `{applied, rejected}` results); User edit probe in quality §9.

### 8. Rewind: swipes, edits and deletes

- **DSS:** state is stored in one directory per turn, keyed by a hash of the full history (`retrieve_history_path`), and each turn loads the previous turn's directory. A swipe or regeneration therefore restarts from the same pre-exchange directory, so the latest turn rewinds by construction (orphaned directories accumulate). An edit or deletion earlier in the chat changes the hash, no matching directory exists, and an empty directory holding only the schema files is created, so state appears to be lost. `backtrack_history`, the hook meant to reconcile this, is a placeholder that does nothing. I read this path in the code and did not run it. The message index lives inside each turn's directory, so it follows the snapshot.
- **Smart-Memory:** subscribes to swipe and delete events. Depth not traced.
- **TunnelVision:** snapshots each write. `revertInvalidSnapshots` undoes any whose message was swiped, edited or deleted, keyed on message identity so it does not depend on event arguments. Swipe behavior has its own tests.
- **Talemate:** scene save and restore and a revision history. Not traced for swipe-specific behavior.
- **Luker:** the most complete. A commit log is the only source of truth, and state is the replay of commits whose message still exists in the chat, filtered by (floor, swipe id). Edits, swipes and deletes roll memory back to before the affected message automatically, and recall results are reused when a message is regenerated. A code comment records that the log once grew to about 19 MB for about 1 MB of data before commits were made incremental.
- **OpenVault:** avoids the problem instead of solving it. The most recent complete turns are left out of extraction, so swiped replies never enter memory, and memories refer to messages by fingerprint (send date or content hash) so they survive index shifts. It has no swipe or delete handler.
- **Summaryception:** the same idea. Only turns that have aged out of the verbatim window are summarized, summarized messages are hidden and not deleted, and the summarized-up-to pointer is repaired on branch or truncation.
- **Graphiti:** `remove_episode` removes edges and nodes whose only source was that episode and rebuilds summaries from the rest, but the host has to call it.

Gap: **−2, moderate** ▮▮▯ vs Luker. The latest turn is already safe (swipes and regenerations restart from the pre-exchange snapshot), which is par. Edits and deletions earlier in the chat appear to lose state, because `backtrack_history` is a placeholder.
Effort to close: S to restore the nearest snapshot, M to replay the missing turns. Confidence: Medium (read in code, not run).
Recommendations: quality §3.4 (first step: implement `backtrack_history` by nearest-prefix snapshot; then the commit-log and commit-lag alternatives) and the §3.1 follow-up; headless P1; Rewind probe in quality §9.

### 9. LLM cost per turn

- **DSS:** the heaviest. The `sci_fi_heist` run made 961 DSS calls over 56 turns (about 17 per turn on average, 33 to 53 on a scene turn), took about 112 s of DSS time per turn on average, and averaged 18k to 20k input tokens per call. An earlier 100-turn soak averaged about 30 calls per turn. Mitigations in place: gates, `skip_query`, cadence profiles, parallel subject workers, and static-first templates for prefix caching.
- **Smart-Memory:** few background calls (extraction on a cadence), a separate memory model can be configured, and per-tier budgets cap injection.
- **TunnelVision:** background features are off by default. With the sidecar on, it adds roughly one call per turn, with cooldowns and timeouts.
- **Talemate:** agents are toggled individually, and a small decision model with budgets and gates handles classification. A search found no prefix-cache awareness.
- **Summaryception:** the cheapest. About one summarization call per 3 turns plus layer promotions, with the layered summary injected whole. The injected block changes only when a snippet is added, so the prompt prefix stays stable between turns.
- **OpenVault:** batched extraction through a scheduler, reflections triggered when accumulated importance passes a threshold, and a world-state summary about every 50 messages (from its docs).
- **Luker:** extraction after replies at a configurable interval per type, and LLM recall by default (a call per message), with RAG recall as the cheaper option.
- **Graphiti:** several LLM calls per episode (extraction, edge extraction and resolution, timestamps, summaries, dedup fallbacks), with a concurrency semaphore (20 by default) and an LLM response cache.

Gap: **−3, far** ▮▮▮ on raw calls: about 17 DSS calls per turn on average (33 to 53 on a scene turn) against about 0.33 for Summaryception, roughly 50 times as many. The other systems were not counted. Not adjusted for how much structure DSS tracks.
Effort to close: L. Confidence: Medium.
Recommendations: soak §36 S1 (prompt anatomy), S2, S3, S4, S5; retrieval R7.

### 10. Context budgeting and injection control

- **DSS:** a standing rule of under 40k rendered tokens (soak §35) and per-subject caps (for example 25 elements). No central allocator, so budgets live in templates.
- **Smart-Memory:** per-tier token budgets with per-chat overrides, trim statistics, and unified injection with a stable order.
- **TunnelVision:** a combined character budget across mandatory, world-state, smart-context and notebook injections.
- **Talemate:** not reviewed in detail.
- **OpenVault:** chronological bucket balancing and a final token budget on injection (per its docs).
- **Luker:** per-type recall buckets, plus an `alwaysInject` flag that takes persistent types out of the recall pool.
- **Summaryception:** fixed structural caps (verbatim turns, snippets per layer, layer count) and a fixed injection depth.

Gap: **−2, moderate** ▮▮▯ vs OpenVault and Smart-Memory. DSS has a standing rule and per-subject caps, but no allocator, no per-purpose budget and no trim telemetry.
Effort to close: M. Confidence: Medium.
Recommendations: retrieval R7 (one allocator with per-subject floors); R1 (per-subject `always | retrieved` policy); R3 (age buckets); soak §36 S3.

### 11. Summarization hierarchy

- **DSS:** scene, chapter and arc archival with content-bearing gates and span digests. A canon synopsis (`canon_synopsis_p0.md`, not in the public clone) is in design.
- **Smart-Memory:** session summaries, scenes, arcs and canon tiers.
- **TunnelVision:** a summary hierarchy module, arcs and scene archiving.
- **Talemate:** layered history with a summary per layer, and a guard against a summary longer than its source.
- **Summaryception:** the purest version. 10 turns stay verbatim, 3 turns make a snippet, up to 20 snippets fill a layer, 3 snippets are promoted into one at the next layer, up to 5 layers, injected oldest-deepest first.
- **Luker:** compression rolls events into higher-level summaries (depth-aware) while keeping the detail nodes.
- **OpenVault:** reflections synthesized from accumulated importance, plus community-based world summaries.
- **Graphiti:** community summaries, and node summaries rebuilt from their episodes.

Gap: **0, par** ▯▯▯. Scene, chapter and arc archival match the layered designs. The canon synopsis is still in design.
Effort to close: S. Confidence: Medium.
Recommendations: none new.

### 12. Robustness and failure handling

- **DSS:** tolerant JSON parsing, enums coerced instead of retried, container-shape repair (`coerce_container_types`), and cross-turn instruction anti-loop guards. The harness died on an unhandled `URLError` once (soak §34), and the engine has the same gap: its retries are content-level (`add_new`, the gate check), and I found no transport retry or backoff in `llm_client.py` or elsewhere in the engine.
- **Smart-Memory:** embedding-to-Jaccard fallback and parser golden tests.
- **TunnelVision:** a circuit breaker on the sidecar and diagnostics.
- **Talemate:** optional components degrade (the reranker falls back and reports its error), plus retries and structured logging.
- **Summaryception:** retry with exponential backoff, honors retry-after hints, stops after consecutive failures, and offers a catch-up for backlogs.
- **OpenVault:** an AIMD queue (adaptive concurrency), abort-signal handling covered by tests, and a backup LLM profile.
- **Luker:** op batches are validated and report rejected operations instead of throwing.
- **Graphiti:** retries on rate limits, empty responses and JSON decode errors, a response cache and a concurrency semaphore.

Gap: **−2, moderate** ▮▮▯ vs OpenVault, Summaryception and TunnelVision. DSS has content-level retries and repair, but no transport retry, backoff, failure limit or circuit breaker, so a server blip fails the turn.
Effort to close: S for a bounded retry, M for adaptive concurrency. Confidence: Medium (searched the engine, not run).
Recommendations: soak §36 S6 (now with Retry-After handling, a failure limit and AIMD sizing of parallel workers).

### 13. Evaluation and tests

- **DSS:** the most developed narrative-state evaluation: soak runs with a judge, an auditor, planted-note probes and a calibration ladder, plus hermetic golden scenarios. There is no retrieval-only evaluation. In the public clone, `run_tests.py` imports `boundary_persistence_test`, `cadence_profiles_test` and `force_boundary_test`, which are not present; they may be unpushed.
- **Smart-Memory:** 75 of 76 unit tests pass, and the golden replay covers parsing.
- **TunnelVision:** 756 tests with 28 stale failures in one file.
- **Talemate:** about 8.8k test functions, with more test code than source code.
- **OpenVault:** 64 test files (about 18.2k lines, roughly the size of the source). 1,184 of 1,184 tests pass.
- **Luker:** 52 memory-graph test files (about 10.6k lines). Not run.
- **Graphiti:** about 65 test files (about 23.7k lines), plus `tests/evals` with a LongMemEval dataset and an end-to-end graph-building evaluation. Not run (it needs a database and an LLM). LongMemEval measures conversational recall, not story-state consistency.
- **Summaryception:** none.

Gap: **+1, lead** ▲ on state evaluation (probes, judge, planted notes) and **−2, moderate** ▮▮▯ on retrieval-only evaluation, which does not exist. Test volume was not compared.
Effort to close: M (retrieval R0). Confidence: Medium.
Recommendations: retrieval R0; the new probes in quality §9. The missing test modules are a repo check, not a plan item.

### 14. User editability and UI

- **DSS:** state is JSON files. The planned subject-data tree UI is not built.
- **Smart-Memory:** a settings panel and memory views with edit and delete handlers (not exercised).
- **TunnelVision:** a drag-and-drop tree editor over native lorebooks, an activity feed, and entry health indicators.
- **Talemate:** a world editor, node editor, character editor with state tracking, perspective views, and 268 documentation pages.
- **Luker:** graph, node and recall views, a revision workbench with approval cards, and a per-card schema editor.
- **OpenVault:** UI for editing, deleting and merging memories and entities, plus a debug export.
- **Summaryception:** a snippet browser with edit and delete.
- **Graphiti:** none; it is a library with a server and an MCP interface.

Gap: **−3, far** ▮▮▮ vs Luker and Talemate. The planned UI is not built, and state is edited as JSON files.
Effort to close: L. Confidence: High.
Recommendations: headless P3 (read-mostly UI), P4 (editing), P5 (LLM-assisted edits), P6 (schema editor); quality §3.2 for safe edits.

### 15. Portability

- **DSS:** tied to text-generation-webui, with a partial seam in `runtime.py`. State is plain JSON on disk.
- **Smart-Memory and TunnelVision:** SillyTavern only. TunnelVision's data is ordinary lorebooks, which is portable inside SillyTavern.
- **Talemate:** a standalone application with several backends (llama.cpp, KoboldCpp, TabbyAPI, text-generation-webui, OpenAI-compatible and cloud providers).
- **Graphiti:** a Python library plus a server and an MCP server, with several graph-database backends (for example Neo4j, FalkorDB, Kuzu) and several LLM providers. It needs infrastructure DSS does not.
- **Luker:** a SillyTavern fork, so users must switch applications. Import and export bind memory to chat floors in documented modes.
- **OpenVault and Summaryception:** SillyTavern only. OpenVault needs no external database.

Gap: **−2, moderate** ▮▮▯ vs Graphiti and Talemate, and 0 vs the SillyTavern extensions. DSS is tied to text-generation-webui with a partial host seam.
Effort to close: M. Confidence: Medium.
Recommendations: headless P0 (boundary audit), P1 and P2 (store and API). Export and import session chaining is already in the `PROJECT_STATE.md` queue; Luker's import modes are a model for it.

### 16. Observability and debuggability

- **DSS:** the strongest cost instrumentation (timing by subject and action, prefill and decode split, per-turn detail). No view of why an item was retrieved.
- **Smart-Memory:** telemetry and trim statistics per tier, and logging.
- **TunnelVision:** an activity feed, diagnostics and entry quality scores.
- **Talemate:** structured logging and a decision log.
- **Luker:** a recall view showing the last injection, a request inspector, and a node view.
- **OpenVault:** performance metrics and a debug-export module.
- **Summaryception:** a snippet browser; no provenance for what was injected.

Gap: **0, par** ▯▯▯. DSS leads on cost timing (+1) and trails Luker on retrieval provenance (−1).
Effort to close: M (retrieval R7). Confidence: Medium.
Recommendations: retrieval R7 (provenance per injected block); headless P2 and P3 (explain endpoint and view); soak §36 S1 (prompt anatomy).

### 17. Temporal validity of facts

Whether the system can say when a fact was true, and what was true at a past point.

- **DSS:** `_history` stores old and new values per field, and the per-turn snapshot directories (criterion 8) already hold the full state at every turn. There is no "valid from, valid until" interval and no turn-to-directory index, so an as-of read is possible in principle but not exposed.
- **Graphiti:** the reference (criterion 6). Each fact carries when it was true and when the system learned it was superseded, so "what was true then" is a query.
- **Luker:** replaying the commit log up to a floor gives the state as of that floor.
- **OpenVault:** memories carry message ranges and decay, but no validity intervals. Smart-Memory has `superseded_by` and expiration; TunnelVision has neither.

Gap: **−1, slight** ▮▯▯ vs Graphiti. The data exists (history diffs and per-turn snapshots). Validity intervals, a turn-to-snapshot index and an as-of read do not.
Effort to close: S. Confidence: Medium.
Recommendations: quality §3.1 follow-up (a turn-to-snapshot index, and `node` on `_history`); headless P1 and P2 (`as_of` reads).

### 18. Schema versioning and migration

How stored data survives changes to its own format.

- **OpenVault:** schema-validated stored data (Zod), schema version 8 with ordered migrations that backfill fields, and an authoritative data-schema doc.
- **Luker:** a migration pipeline over versioned store shapes (the current persisted shape is an op log).
- **Graphiti and Summaryception:** not applicable or none found.
- **DSS:** schema changes are manual (`quality_enhancements.md` section 10, open question).

Gap: **−2, moderate** ▮▮▯ vs OpenVault and Luker. Schema changes are manual, with no version stamp or ordered migrations.
Effort to close: M. Confidence: Medium.
Recommendations: headless P1 (`state_version` and ordered, backfill-only migrations); headless P6 (dry-run migration for schema edits).

## 4. Where DSS leads (protect these)

- User-defined schema with typed fields, enums and declared graph edges (criterion 2). Only Luker also has user-defined types, and it has no weighted traversal. Do not trade the schema for a fixed taxonomy (OpenVault, Smart-Memory) to fix a gap elsewhere.
- Weighted multi-hop graph retrieval (criterion 4). Graphiti's search is the closest (BM25, cosine and breadth-first expansion with a node-distance reranker). Measure before changing anything.
- Field-level change history (criterion 6), which is the base for rewind and user-edit precedence.
- Content-addressed per-turn snapshots (criteria 8 and 17): the latest turn rewinds by construction and any past turn's state can be loaded. Luker reaches the same goal with a commit log.
- Cost instrumentation and cache-aware prompt ordering (criteria 9 and 16).
- Typed epistemic stances (criterion 1).

## 5. Ideas from the others considered and not adopted

See `retrieval_plan.md` section 6 for the reasoning:

- LLM-navigated tree retrieval (TunnelVision): a call per turn on top of about 17 to 30.
- Presence-derived knowledge (Talemate): needs presence on every message and invalidation machinery.
- Hand-weighted linear scoring with fixed constants (Smart-Memory): adopted only with weights in config tuned against `retrieval R0`.
- Fixed per-type state fields (Smart-Memory): conflicts with `quality §2`. Its place-state fields are taken as candidates for the schema instead (`quality §5.4`).
- Graphiti as a dependency: it needs a graph database and several LLM calls per episode, and its facts are global, with no scenes, arcs or per-character perspective. Take the staged entity resolution and as-of reads (`quality §3.3`, `headless P1`), not the system.
- Luker's LLM-driven recall by default (a call per message): the same reasoning as TunnelVision's tree walk. Its RAG recall mode, with per-type buckets, is the relevant one (`retrieval R1`).
- Luker as a host: a fork is a source of designs, not an option for DSS.
- Summaryception's inject-everything summary: fine for prose memory, but it does not scale to structured state. Its verbatim window and layer caps match what DSS's archives already do.
- OpenVault's fixed schema: conflicts with `quality §2`. Its witness lists are a cheaper alternative to `knows` edges only if the schema allowed them.

## 6. Unverified items

- TunnelVision's secret-tag handling, alias handling, and write permissions in practice.
- Smart-Memory's graph and edit UI behavior; Talemate's per-field locks, swipe behavior and prompt budgets.
- What `scripts/cross_encoder.py` found.
- What a mid-history edit does in practice: I read that an empty directory is created and state is lost, but I did not run it (criterion 8).
- Resolved on 2026-11-06 by reading the DSS code: how history-hash directories behave (criterion 8), that `_safe_subject` silently drops a subject (criterion 3), and that the engine has no transport retry (criterion 12).
- Luker: how well its LLM recall performs, whether its rollback covers every path (for example after compression rollups), and the per-message variables and state-system pages, which I did not read.
- OpenVault: retrieval formulas, merge guards and token budgets come from its documentation. Current-code behavior is checked only against the layout and its tests. The features in the `7n1m4/vault` fork README are not reviewed.
- Graphiti: the number of LLM calls per episode, which I describe as "several" without counting.
- Summaryception: injection-depth semantics beyond the defaults, and whether snippet edits survive a later promotion.

## 7. Refreshing this document

Re-clone each system at its latest revision, update the table in section 1, re-run the TunnelVision and Smart-Memory tests, recount Talemate's tests, and re-check each unverified item. When a recommendation ships, change the position here and link to the commit. When `retrieval R0` has numbers, replace the criterion 4 and 5 positions with measured ones.
