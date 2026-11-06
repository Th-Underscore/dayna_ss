# DSS Headless Engine and Standalone UI Plan

Goal: a UI that can edit schemas, show the entity graph together with its edits, edit entries and the tree, and show why an item was retrieved. It should embed in text-generation-webui without depending on Gradio, and the engine underneath should be host-independent so other hosts are possible later.

Related plans: `quality_enhancements.md` (what is stored, precedence, rewind), `retrieval_plan.md` (provenance), `soak_optimization_plan.md` (parallel workers), `memory_comparisons.md` (criteria 7, 8, 14 and 15).

## 1. Evidence

Static read of `a3c64b4` (2026-11-04).

- **A host seam exists.** `runtime.py` exposes `extension_dir`, `current_character` and `update_config`, and the core engine mostly goes through it.
- **Host imports remain** in `script.py`, `ui/`, `shared.py` and `utils/model_loader.py` (text-generation-webui `modules.*` and Gradio).
- **Coupling inside the engine.** The summarizer is composed from mixins that use a duck-typed `self` (`agents/data_summarizer/`), module-level singletons, `print`-based debug logging, and state held as JSON files read per call. State lives under a history-hash directory (`retrieve_history_path` in `context_engine.py`).
- **A push channel already exists:** `ui/sse_server.py` and `ui/update_queue.py`.
- **The existing UI predates the schema-driven design.** The planned subject-data tree UI (`todo.md`) is not built, and there is no schema editor, entity-graph viewer or entry editor.

## 2. Principles

1. **The engine imports no UI or host framework** outside adapters.
2. **One write path.** UI edits, LLM-assisted edits and DSS patches all go through the same function, so validation, provenance, locking and history happen once.
3. **The UI knows no subject names.** Forms and views are generated from `subjects_schema.json`, consistent with the "no hardcoded subject structures" rule in `quality_enhancements.md` section 2.
4. **No new background LLM calls.** LLM-assisted edits are user-initiated.
5. **Each phase ships on its own** and leaves the engine working.

## 3. Phases

| Phase | What | Depends on | Extra LLM cost |
|-------|------|------------|----------------|
| P0 | Boundary audit | none | 0 |
| P1 | `StateStore`: one write path | P0 | 0 |
| P2 | Engine API and event stream | P1 | 0 |
| P3 | UI v1, read-mostly | P2, `retrieval_plan.md` R7 for "why retrieved" | 0 |
| P4 | Editing from schema-generated forms | P1, `quality_enhancements.md` 3.2 | 0 |
| P5 | LLM-assisted edits | P4 | User-initiated only |
| P6 | Schema editor | P4 | 0 |

### P0 Boundary audit

List every import of `modules.*`, `gradio` and `extensions.*` outside an adapter directory. The target is zero in the engine packages (`agents/`, `rag/`), with the rest moved behind `runtime.py` or into `adapters/tgwui/`. Measure by a grep count in CI, so the number can only go down. Also replace `print` debug output in the engine with a logger, since the API needs structured events.

### P1 StateStore

An interface the engine uses for all reads and writes of subject data:

```
read(subject, key=None, as_of=None) -> data, version
apply(patch, actor, node, reason) -> result     # validated, atomic
subscribe(callback)                             # change events
```

- **JSON on disk stays** as the backend. Write atomically (temporary file, then rename) and keep a version counter per file.
- **Validation** against `subjects_schema.json` (types, `field_options` enums) happens in `apply`, once.
- **Provenance.** `actor` is `llm`, `user` or `system`. `apply` appends the `_history` entry with `node` and `actor` (`quality_enhancements.md` 3.1).
- **Locking.** A path stamped `user` rejects `llm` patches and logs the attempt (`quality_enhancements.md` 3.2).
- **Rewind.** `rewind(to_node)` is built on this (`quality_enhancements.md` 3.4).
- **As-of reads.** `read(..., as_of=node)` loads the snapshot directory for that node (DSS already keeps one per turn), or replays `_history` backwards. Graphiti's bi-temporal edges and Luker's commit-log replay are the precedents (`memory_comparisons.md`, criteria 8 and 17).
- **State version and migrations.** Stamp each state file with a `state_version` and apply ordered migrations on load that backfill new fields and never drop data. OpenVault validates stored data with a schema and runs numbered migrations; Luker versions its persisted store shapes (`memory_comparisons.md`, criterion 18).
- **Concurrency.** DSS runs subject workers in parallel (`max_subject_workers`, `soak_optimization_plan.md` "Parallel subject processing"), and a user edit can race with a worker. Use optimistic concurrency: an edit carries the version it was based on, and a mismatch returns the current value for the UI to re-read. Never block a worker on the UI.

### P2 Engine API and event stream

Prefer extending the existing SSE server, or a small FastAPI app if that is cleaner. Endpoints:

| Endpoint | Purpose |
|----------|---------|
| `GET /schema` | `subjects_schema.json` and format templates, for generated forms |
| `GET /subjects/{subject}` and `/entries/{key}` | Read state with version, optionally `?as_of=node` |
| `PATCH /subjects/{subject}/entries/{key}` | User edit through `apply(actor=user)` |
| `GET /graph` | Nodes, edges and edge weights from `entity_graph` |
| `GET /history?path=` | `_history` entries for a path |
| `GET /retrieval/explain?turn=` | The `ContextBlock` list with reasons (`retrieval_plan.md` R7) |
| `POST /rewind` | Rewind to a node |
| `GET /events` (SSE) | Change feed from `subscribe` |

Local only by default (bind to localhost). Document the port and the embedding method.

### P3 UI v1, read-mostly

A static single-page app served by the engine. Views:

- **Entity graph** with edge weights and node importance, and a click-through to the entry.
- **Entry browser** generated from the schema.
- **History diff** per field, from `_history`, showing `actor`.
- **Why retrieved** for a chosen turn, from the explain endpoint.

Embed it in text-generation-webui as a tab that loads the engine's page in an iframe. That avoids fighting Gradio and leaves the UI usable outside it. Bundle the graph library locally; do not rely on a CDN for a local tool.

### P4 Editing

- **Forms** generated from the schema (field type, `field_options` enums, keyed lists).
- **Pin and lock toggle** per path, which stamps or clears `user`.
- **Diff preview** before saving.
- **Undo** by applying the inverse of a `_history` entry.

### P5 LLM-assisted edits

The user describes a change ("merge these two characters", "Dex's goal is now the vault"). The model proposes a constrained patch through a small set of operations: set path, rename, merge, delete. The patch is validated by the schema and shown as a diff for approval; approved edits are stamped `user`. This uses the existing tool definitions (`tools/definitions/`; check how they are registered before reusing them). It adds no background calls because it only runs when the user asks. Luker's AI revision workbench is the closest precedent: proposed changes arrive as reviewable cards the user approves or rejects (including merging duplicates), and its write API reports applied and rejected operations instead of failing the whole batch (`memory_comparisons.md`, criteria 6 and 7). Do the same: `apply` returns `{applied, rejected}`.

### P6 Schema editor

Edit `subjects_schema.json` with validation, plus a dry-run migration that reports what would change in existing state. This is where the open migration question in `quality_enhancements.md` section 10 (`attributes: list[str]` to keyed structures) gets a home. Reuse the ordered migrations from P1, so a schema edit and a state migration are one operation.

## 4. Not planned

- Rewriting the UI inside Gradio.
- Moving state to a database. JSON on disk stays until per-call reads show up as a bottleneck in timing data.
- A SillyTavern adapter. P0 to P2 make one possible; building it is a separate decision (`memory_comparisons.md`, criterion 15).

## 5. Open questions

- Extend `ui/sse_server.py` or add a separate FastAPI app? Which has fewer moving parts for the textgen embedding?
- How should the iframe be served (same origin through a textgen route, or a separate port with CORS)?
- Which graph library, and does it handle a few hundred nodes with edge weights at an acceptable frame rate?
- The history-hash directories are DSS's current snapshot mechanism (one full state directory per turn, loaded from the previous turn's directory). Keep them behind the store, or replace them with a commit log (`quality_enhancements.md` 3.4)?
- Should user edits to the schema be allowed mid-run, or only between sessions?
- Should `_history` become the source of truth (Luker's design: state is the replay of commits) or stay an audit trail with inverse patches (`quality_enhancements.md` 3.4)?
