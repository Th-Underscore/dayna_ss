# DAYNA Story Summarizer — Strategy Review

*Compiled from a full architecture and competitive-landscape review, August 2026.*

---

## 1. The core decision: keep building, don't rewrite

- **text-generation-webui is not dead.** Renamed to "TextGen" in 2026, actively developed (Electron app, tool-calling, Anthropic-compatible API shipped this year). The old repo URL redirects; nothing is abandoned. This removes "my platform is dying" as a reason to restart.
- **A from-scratch rewrite on a new agent framework is not warranted.** The codebase (~18.8k lines) already splits cleanly into a portable core and a tgwui-specific adapter. The core — `schema_parser.py`, `entity_graph.py`, `context_retriever.py`, the recursive engine in `data_summarizer.py`/`summarizer.py` — has zero framework lock-in already. What needs to change is the *boundary*, not the engine.
- **The real fix is decoupling, not migrating.** Extract the core into a standalone package with no `modules.shared`/Gradio imports. tgwui becomes one adapter among possibly several (an MCP server being the other, high-leverage one — see §4).
- No single orchestration framework (LangGraph, CrewAI, etc.) should be adopted wholesale — binding to one would work against the stated goal of a framework-agnostic, extensible engine. Adopt individual capabilities (durable execution / checkpointing patterns) if useful, not the framework itself.

---

## 2. Competitive landscape

### From the original list (already known)
- **Lorebook for Oobabooga** — static keyword injection, no summarization. Different layer entirely.
- **MemoryBooks (ST)** — scene → JSON summary → lorebook, multi-tier, user-driven scene markers. Polished, less architecturally ambitious.
- **TunnelVision (ST)** — reasoning-based retrieval via tool calls, no embeddings.
- **timeline-memory (ST)** — closest prior competitor: auto chapter detection, agentic timeline fill. More mature, ST-only, no schema-driven declarative logic.
- **Memoir / long_term_memory (tgwui)** — flat embedding stores, the naive-RAG baseline DSS supersedes.

### Found in this review
- **Smart-Memory (ST extension)** — the closest structural peer found. Covered in full in §3–5.
- **Talemate (vegu-ai)** — standalone, framework-agnostic app (works with tgwui, KoboldCpp, TabbyAPI as backends). Multi-agent (dialogue, narration, summarization, direction, editing, world-state), long-term memory via ChromaDB RAG, dedicated "world state" agent for reinforcing narrative truths. Most mature *standalone* competitor — proof that people want this decoupled from any one chat frontend.
- **Extension-Summaryception (ST)** — recursive layered summarization scaling indefinitely, non-destructive, proposes World Info entries from stable facts separately from event summaries. Different philosophy: compression layers rather than typed schema fields.
- **NarrativeEngine-P** — standalone self-hosted TTRPG "AI Dungeon Master." Philosophical opposite of DSS: archives every turn verbatim, nothing is summarized away; recall via two-stage retrieval (chapter-scan, then vector scene retrieval). Write-cheap/read-expensive vs. DSS's write-expensive/read-cheap model — a real, deliberate tradeoff worth naming to yourself (see §6).
- **MemoryMesh (MCP server)** — schema-based knowledge graph server with auto-generated CRUD tools, domain-specific handling for game elements (factions, NPCs, quests, locations), exposed via MCP to clients like Claude Desktop. This is prior art for exactly the "expose the engine over MCP" move recommended in §4 — worth studying directly.
- **"Narrative World Model" (NWM)** — arXiv preprint, ~July 2026, not a shipping tool but important: proposes typed memory records, a temporal knowledge graph with validity intervals, narratology-grounded types (focalization, epistemic state, event-vs-reveal order, dramatic function, promise/payoff), benchmarked explicitly against GraphRAG and Graphiti/Zep. Validates that "narrative-specific structured memory beats generic memory frameworks" is a live research thesis, not just a hobby hunch. Also a near-collision with the "DAYNA World Model" rename idea — read before finalizing that name, and consider borrowing its evaluation methodology and vocabulary.
- **General-purpose memory frameworks** (Mem0, Zep/Graphiti, Letta, LangMem) — the "generic vector/temporal-graph memory" category DSS's schema-driven, narrative-typed approach is positioned against, not competing head-on with. Good backdrop, not a direct competitor for the near term (see §6 on scope discipline).

---

## 3. Feature-by-feature comparison vs. Smart-Memory

| Smart-Memory feature | DSS status | Note |
|---|---|---|
| Token usage display / auto-tune budgets | **Missing** | No injection-budget UI or auto-sizing; open TODO item already. |
| Long-term persistent facts, retire-and-replace | **Partial, different mechanism** | Gate-check→update is the conceptual equivalent but overwrites in place — no semantic-similarity dedup, no supersession chain/history. |
| Activation-trigger keyword boosting | **Different approach** | Structured RAG (entity graph + llama-index) instead of keyword triggers. More principled, unproven in practice. |
| Session memory (within-chat detail) | **Overlaps** | `current_scene["now"]` serves this role structurally. |
| Short-term rolling summary | **Missing** | No standalone rolling chat-level summary independent of schema state. |
| Scene detection (auto boundaries) | **Implemented, undocumented** | `_check_scene_transition` already does LLM-based auto-detection — README/roadmap is stale on this point. |
| Story arcs / chapters | **Schema exists, engine support unconfirmed** | `Arc`/`Chapter`/`CrucialEvents` fully defined with prompt templates; unclear if the summarization loop populates them yet — verify before assuming done. |
| Canon (story-bible generation) | **Missing** | No aggregation-to-prose step, but `Arc`/`Chapter` data already provides the inputs. |
| Character/world profiles (compact snapshots) | **Different approach, arguably ahead** | Your schema state is already structured and current; you don't need a separate snapshot-generation step, just formatting for injection. |
| Relationship history | **Implemented, different representation** | Yours: structured `relation`/`status`/`importance`/`disposition`. Theirs: compact `word(magnitude)` tags. Consider a compact-summary *projection* of your structured data for injection efficiency — additive, not a replacement. |
| Perspectives & Secrets (epistemic states) | **Largely unimplemented** | No epistemic fields anywhere in the schema. Highest-leverage gap given your stated goals — see §7. |
| Entity registry (type, merge, trash, timeline) | **Backend exists, no UI** | `entity_graph.py` already builds and types entities. |
| Relationship graph visualization | **Data model ready, no UI** | Your `entity_graph.py` has real typed `source_id`→`target_id` edges. Their graph is bipartite (entity↔memory + supersession chain), not true entity-entity edges — visually similar, structurally different, and less rich than what you already have on disk. |
| State Ledger (current physical/observable state) | **Missing** | No location/outfit/mood/carried-items fields on `Character`. Concrete, cheap-to-add gap. |
| Away recap | **Missing** | Low-priority polish. |
| Continuity checker (+ auto-repair) | **Missing** | No contradiction-detection pass. Good post-harness addition. |
| Per-character isolated memory / group scoping | **Architecturally different, not missing** | They give each character its own memory store (fits ST group-chat character-swapping). You model one shared world graph — more aligned with the "whole world" goal, and a prerequisite for doing Perspectives & Secrets well without duplicating stores per character. |

---

## 4. Prompting & watcher architecture comparison

| Dimension | Smart-Memory | DSS |
|---|---|---|
| Pre-reply LLM calls | None — context is injected, not planned | One extra completion per turn (director → instructions), **confirmed optional**, a prime test-harness candidate |
| Model separation | Explicit, user-configured separate "Memory LLM," raw completion, no instruct template required | Same persona machinery (name1/name2, chat template) reused for director, reply, and field updates by default |
| Output format | Deliberately plain, bracket-tagged text — avoids JSON because local models are unreliable at it | Typed JSON per field, schema-validated, retry-on-error loop (max 2 retries, error fed back into prompt) |
| Injection mechanism | Platform-native extension-prompt slots (`setExtensionPrompt`), ordered stable→immediate, discretely measurable per tier | Folded directly into prompt construction inside `custom_generate_chat_prompt` — not separately measurable per section |
| Watcher breadth | 9+ distinct ST events: chat changed/loaded, message sent, generation started, message swiped (aborts in-flight generation), message deleted, group wrapper/member/updated events | `NEXT SCENE:` prefix + auto-detected transitions; narrower event coverage — maps directly to open reliability TODOs (continue/regenerate/message-edit handling) |
| Post-reply timing | `CHARACTER_MESSAGE_RENDERED` | `handle_output` — same timing, independently converged |

**Two live design forks worth testing deliberately, not assuming:**
1. **Director pass on/off.** Confirmed optional. Test whether the explicit two-call plan-then-write pattern measurably improves output vs. its cost (2× generation calls per turn), against single-call generation with either a native reasoning model or structured output. The director pass's real advantage over hidden reasoning is that its output is plain text *you* can inspect and edit — worth preserving that property in whatever the harness concludes.
2. **JSON-typed output vs. tagged plain text.** Smart-Memory's anti-JSON stance is evidence from real testing down to `gemma3:4b` on 8GB cards — exactly your target hardware profile. Worth measuring actual JSON-compliance rate on smaller local models before assuming the retry loop is sufficient; a tagged-text fallback for weaker models (mirroring their Profile A/B hardware tiers) may be worth adding if compliance is shaky.

---

## 5. Extensibility assessment

- Your schema parser (`Action`/`Trigger` enums, dataclass-style field definitions, gate-check → query-branch → update as generic operations) is **already general-purpose** — the fiction-specificity lives entirely in `subjects_schema.json`, not the engine. This is the right shape for the "compete with general memory systems eventually" goal; it doesn't need to be rebuilt, just exercised with more schema packs later.
- Smart-Memory's entity/memory taxonomy is hardcoded (`constants.js`) — good for a tuned single-purpose product, but not extensible the way your schema is. This remains your clearest differentiator on the extensibility axis specifically.
- Smart-Memory is coupled to SillyTavern the same way DSS is coupled to tgwui (direct imports of ST's own loaders in browser JS) — they are not ahead of you on framework-agnosticism, arguably behind, since your core Python has no Gradio imports today.
- Highest-leverage extensibility move: **expose the core engine over MCP**, not bound to any single agent framework. MemoryMesh is prior art for this exact pattern in the narrative/game-state space.

---

## 6. Scope discipline (say this to yourself once and move on)

- Competing head-on with funded general-memory platforms (Mem0, Zep, Letta) as a solo project restarting after months away is a multi-year bet, not a v1 target.
- The wedge is real and underserved: schema-driven, narrative-typed memory for long-form fiction/roleplay. Ship that, get real users, let the schema-authoring experience mature under real use — then generalize to a second schema pack.
- Decide deliberately whether you're on the "consolidate at write time, cheap reads" side of the tradeoff (current DSS design) or the "never lose detail, expensive precise reads" side (NarrativeEngine-P) — both are legitimate, they are not the same bet, and the gate-check engine only makes sense if you're intentionally choosing the former.

---

## 7. Where DSS can genuinely beat Smart-Memory — and where not to bother trying

Not "better at everything" — that's not a winnable race against an actively-developed competitor with a large iteration head start. Two specific axes where the architecture gives a real ceiling advantage, if executed well:

- **Asymmetric/perspective-relative state at scale.** Smart-Memory's Perspectives & Secrets is a retrofit — epistemic tags on a bipartite entity↔memory graph, per-character isolated stores. DSS's unified typed graph can make this a first-class relation instead (what Alice's node says about Bob's node, visible only from queries with the right vantage point). A retrofit tends to accumulate "why does this character suddenly know something they shouldn't" drift as cast/faction complexity grows; a relational model resists that structurally. This should show up specifically at the long-form scale DSS is targeting, not in short sessions.
- **Consistency under schema growth.** Smart-Memory is many separately hand-tuned extraction passes — each new feature is a new prompt to keep in sync by hand. DSS's single declarative engine means a reliability fix to the gate-check mechanism improves every field at once, including ones not yet built. Bigger upfront lift, nonlinear payoff as the schema grows — the actual case for having generalized at all.
- **Generation-time steering** (director pass), *if* the harness shows it earns its keep — Smart-Memory doesn't touch generation at all, so this is a category it can't compete on without rearchitecting.

Where **not** to compete: token-budget accounting, watcher event coverage, small-model output robustness, hardware-tiered fallbacks. That's accumulated iteration against real failure cases, already paid for on their side. Don't re-derive it — copy their validated answers (event list, hardware-tiering approach, plain-text-output evidence) to reach parity cheaply, and spend the saved effort on the two axes above. Goal is "good enough" on the rest, shipped, not parity-by-original-engineering everywhere.

---

## 8. Prioritized roadmap

1. **Decouple the core engine from tgwui.** Strip `modules.shared`/Gradio imports from `schema_parser.py`, `entity_graph.py`, `context_retriever.py`, `data_summarizer.py`/`summarizer.py`. This is the prerequisite for #2, not a separate track.
2. **Build the golden-fixture test harness.** Model it on Smart-Memory's `test-harness.mjs` + `tests/golden/` + `tests/fixtures/` structure: scripted transcripts in, structured schema-state assertions out. First tests to write:
   - Director pass on vs. off — quality and cost comparison.
   - JSON compliance rate on small local models (down to your actual target hardware floor).
   - Gate-check update-vs-preserve correctness on a scripted scene with known state changes.
   - **Status: harness core shipped** (`tests/harness.py` + `tests/run_tests.py`, `--update-golden`). Engine imports hermetically (llama_index lazy-imported in `context_retriever.py`; `torch.no_grad` now flows through a `runtime.no_grad()` seam, so the core imports with zero torch/llama_index/gradio). Four test layers green: (1) golden fixture `gate_check_update_vs_preserve` — turn 0 gate-check `NO` preserves the initial state exactly, turn 1 gate-check `YES` applies a scripted branch update; (2) `director_pass_on_off` — drives the real `Summarizer.generate_instr_prompt` with a counting fake model: `do_instr=True` costs exactly 1 extra LLM call/turn, embeds instructions, and is cache-served on a repeated seed; (3) `soak_conversation` — a 20-turn scripted conversation (44 scripted LLM calls) with planted facts, reporting per-fact retention (planted turn, last seen, loss turn) and asserting each fact's survival expectation; mutation-verified to catch drift; (4) `live_benchmark.py` (`--live`) — scores a real model's output against the engine's own parsers (gate YES/NO, field-update JSON, new-entry names); skips cleanly when `DSS_BENCH_BASE_URL` is unset. Run: `cd extensions/dayna_ss/tests && python run_tests.py [--live]`.
   - **Live-testing environment now available (native Ubuntu).** See §9 for the hardware profile, power constraint, model tiers, and how to point the live benchmark at a served model.
3. **Close the two structural gaps that serve the stated goals, not just polish:**
   - **State Ledger equivalent** — current location/outfit/mood/carried-items per character. Cheap: new dataclass + schema wiring, no engine changes.
   - **Perspectives & Secrets** — epistemic-state fields per subject. Your unified-world-graph architecture is actually a *better* fit for this than Smart-Memory's per-character-store model, since you don't need duplicate stores to represent asymmetric knowledge. This is where your engine's generality should start visibly paying off.
4. **Consolidation/deduplication on write** — decide deliberately: supersession chain (auditable history) vs. overwrite-in-place (current behavior). Make this a harness test case, not a blind build.
5. **Continuity checker** — cheap, decoupled, single LLM call comparing latest output to stored state. Good first "visible win" once the harness exists to verify it actually catches injected contradictions.
6. **Entity registry UI + relationship graph visualization** — lower priority than the above, but genuinely cheaper than it looks: `entity_graph.py` already has real typed edges, so this is a renderer on existing data, not new modeling. Structurally richer than Smart-Memory's bipartite entity↔memory graph.
7. **Canon generation, token-budget UI, away-recap** — real but pure polish. Canon is a small lift given `Arc`/`Chapter` are already schema-defined; the others are deferrable.
8. **MCP server wrapper** around the decoupled core, once it's stable — the framework-agnostic distribution mechanism, and the concrete first step toward the "usable by others, truly extensible" goal.

---

## 9. Live test environment (native Ubuntu)

Hardware available for DSS live testing:

| Device | Role | Notes |
|---|---|---|
| GTX 1060 3GB (GPU 0) | **Display only** | Not used for CUDA compute during testing — do not target it. |
| Tesla V100-SXM2-16GB (GPU 1) | Compute | Primary test device. |
| Tesla V100-SXM2-16GB (GPU 2) | Compute | Second test device (usable concurrently). |
| Driver / CUDA | 580.173.02 / CUDA 13.0 | |

**Operating constraints:**
- Keep power limits and clock speeds as-is. Do not raise power caps; avoid sustained high-power draw during benchmarking.
- Small-model tests are still run on a V100 using smaller models/quants (Qwen2.5-0.5B/1.5B, Llama-3.2-1B/3B, Q4/Q8) — the 1060 is not a test target.

**Serving options** (pick one, both OpenAI-compatible on `127.0.0.1`):
- **LMDeploy TurboMind** — fast llama-family serving, `lmdeploy serve api_server ...`.
- **text-generation-webui / TextGen** — the extension's native host; run the usual server and point the benchmark at its `/v1` endpoint.

**How to run the live benchmark against a served model:**
```bash
export DSS_BENCH_BASE_URL=http://127.0.0.1:1234/v1   # or the chosen server's /v1
export DSS_BENCH_MODEL=<model-name>
export DSS_BENCH_RUNS=10
cd extensions/dayna_ss/tests && python run_tests.py --live
```
The benchmark scores the model's output against the engine's own parsers (gate-check YES/NO, field-update JSON via `_parse_llm_field_updates`, new-entry name lists) and prints a per-prompt-type compliance rate plus an overall rate.

**Recommended first live runs (V100):**
1. **JSON-compliance sweep** — run `--live` across the small-model tier (Qwen2.5-1.5B-Instruct-Q4, Llama-3.2-1B-Instruct-Q8) to see where the JSON retry loop holds and where a tagged-text fallback may be warranted (see §4, design fork #2).
2. **Director on/off** — with a real model, compare output quality and cost (2× generation calls/turn) with `do_instr` on vs off; the harness already proves the cost side hermetically, the live run adds the quality side.
3. **Soak with a live backend** — extend the `soak_conversation` fixture's scripted responses with real model output (a future harness mode) to measure detail/writing-style retention over dozens of messages against an actual model.
