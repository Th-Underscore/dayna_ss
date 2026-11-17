The remaining hard-coded surface is the **per-subject retrieval layer** — the four blocks in `retrieve_context` that populate `result.subjects["characters"]`, `["groups"]`, `["elements"]`, `["events"]`.

These are not parameterizable variants of one algorithm. Each subject uses a distinct retrieval strategy:

| Subject | Method | Behavior |
|---------|--------|----------|
| Character | `_get_all_relevant_character_relationships` (87 lines) | Graph relationship enrichment, milestone filtering/attachment, scene-to-scene second pass |
| Group | inline dict comprehension (3 lines) | Simple entry lookup by name |
| Element | `_graph_scored_elements` inline (14 lines) | Capped graph-scored selection (max 25), scene-first ordering |
| Event | `_select_relevant_events` (88 lines) | 4-bucket read, `_message_node` recency sort, 3-stage selection, chapters list-shape handling |

**Deferred:** Making the per-subject retrieval methods themselves schema-driven (a new subject works with zero code changes) would require a strategy/plugin system where the schema declares the retrieval behavior. The character relationship enrichment and events multi-bucket recency selection are too different to unify under one parameterized algorithm.

**Practical next step (when needed):** Make the *dispatch* dynamic — loop over `retrieval_order` and route to per-subject methods by convention (`_retrieve_character`, `_retrieve_group`, etc.). The per-subject methods stay as-is; adding a new subject requires a new method + `retrieval_order` entry but no changes to `retrieve_context` itself. ~40–60 lines.