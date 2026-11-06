# Design: Schema-Driven `check_archive` Action

**Status:** Proposed
**Date:** 2026-11-06
**Replaces:** `check_and_archive_chapter()`, `check_and_archive_arc()` in `archives.py`; the `archive_template` special-case in `Action.ADD_NEW` dispatch

## Summary

Introduce a new `Action.CHECK_ARCHIVE` that consolidates the archive boundary logic (cadence gating, gate-check LLM call, span computation, entry creation, numbering) into a single schema-driven action handler. All per-branch configuration lives in the schema definition's `defaults`, making the entire archive pipeline declarative.

## Current State

### Trigger Dispatch (schema-driven)

```
StoryEvents.triggers.on_new_scene: ["add_new", "query_branch_for_changes", ...]
Scenes (alias → StoryEvents): inherits the above
```

When `ADD_NEW` fires for `Scenes`, `_execute_action` checks for `archive_template` in defaults. If present, it calls `_archive_boundary_entry` (template render → LLM refine → store). If absent, it calls `_detect_and_add_new_entries_to_branch` (query → create).

### Boundary Checks (hardcoded, outside trigger dispatch)

```python
# core.py: _run_boundary_checks (called on scene turns)
data_summarizer.check_and_archive_chapter()  # archives.py:273
data_summarizer.check_and_archive_arc()      # archives.py:555
```

These are **not** part of the trigger dispatch system. They're called imperatively from the summarizer core. Each contains:
1. Cadence gate (hardcoded fallback `(4,8,10)` / `(2,5,8)`, overlaid by schema defaults)
2. Gate-check LLM call (YES/NO)
3. Span math (scene/chapter index computation)
4. Template render → LLM refine → store (duplicates `_archive_boundary_entry`)
5. Numbering increment (`_chapter_number` / `_arc_number`)
6. Explicit `save_json` persistence
7. Canon digest regeneration

### Problems

| # | Problem |
|---|---------|
| 1 | Two parallel code paths for the same operation (trigger dispatch vs imperative call) |
| 2 | Execution logic duplicated between `_archive_boundary_entry` and `check_and_archive_*` |
| 3 | Cadence values hardcoded in Python with schema overlay (fragile) |
| 4 | No schema-level way to configure: span source, end field, numbering field, storage type |
| 5 | Chapters stored as list, Arcs as dict — inconsistent |
| 6 | `_archive_boundary_entry` has hardcoded `key_field` logic (`"name"` for scenes, `"title"` for others) |

## Proposed Design

### New Action

```python
class Action(IntEnum):
    ADD_NEW = 1
    PERFORM_GATE_CHECK = 2
    QUERY_BRANCH_FOR_CHANGES = 3
    PERFORM_UPDATE = 4
    CHECK_ARCHIVE = 5        # was QUERY_FOR_ARCHIVE (unused)
    SELECT_ENTRIES_TO_UPDATE = 6
```

Rename `QUERY_FOR_ARCHIVE` → `CHECK_ARCHIVE` (same value, clearer name).

### Trigger Inheritance Rule

Aliases use **all-or-nothing** trigger inheritance: if the alias definition declares its own `triggers`, they completely replace the aliased type's triggers. If absent, the aliased type's triggers are used as-is. No merge. This is the existing parser behavior (`_inherit_defaults_from_parent` has `do_inherit_triggers=False` by default; the alias resolution path passes through the target's triggers directly).

Practical effect: `Scenes` defines `triggers: {"on_new_scene": ["check_archive"]}` → it gets ONLY `check_archive`, not `StoryEvents`'s `add_new` + `query_branch_for_changes`.

### Schema Defaults (per definition)

```json
{
  "Chapters": {
    "triggers": { "on_new_scene": ["check_archive"] },
    "defaults": {
      "check_archive_prompt_template": "...",
      "archive_template": "...",
      "new_entry_prompt_template": "...",
      "suggested_min_scenes": 4,
      "suggested_max_scenes": 8,
      "max_scenes_before_required": 10,
      "span_source": "scenes",
      "span_end_field": "ending_scene",
      "numbering_field": "_chapter_number",
      "key_field": "title",
      "storage": "dict"
    }
  },
  "Arcs": {
    "triggers": { "on_new_scene": ["check_archive"] },
    "defaults": {
      "check_archive_prompt_template": "...",
      "archive_template": "...",
      "new_entry_prompt_template": "...",
      "suggested_min_chapters": 2,
      "suggested_max_chapters": 5,
      "max_chapters_before_required": 8,
      "span_source": "chapters",
      "span_end_field": "ending_chapter",
      "numbering_field": "_arc_number",
      "key_field": "title",
      "storage": "dict"
    }
  },
  "Scenes": {
    "triggers": { "on_new_scene": ["check_archive"] },
    "defaults": {
      "check_archive_prompt_template": null,
      "archive_template": "...",
      "new_entry_prompt_template": "...",
      "suggested_min_scenes": 1,
      "suggested_max_scenes": 1,
      "max_scenes_before_required": 1,
      "span_source": null,
      "span_end_field": null,
      "numbering_field": "_scene_number",
      "key_field": "title",
      "storage": "dict",
      "stamp_message_node": true
    }
  }
}
```

**Key design decisions:**

| Decision | Rationale |
|----------|-----------|
| `key_field: "title"` for all | Uniform; `StoryEvents.name` → `StoryEvents.title` rename |
| `storage: "dict"` for all | Consolidate Chapters from list→dict; uniform store logic |
| `check_archive_prompt_template: null` for Scenes | Scenes always archive on scene boundary (no gate needed); null = skip gate check |
| `suggested_min_scenes: 1` for Scenes | Gate is effectively always-open (min=1 means "archive when ≥1 scene transition occurs") |
| `span_source: null` for Scenes | No span math needed (scenes don't aggregate other types) |
| `stamp_message_node: true` for Scenes | Schema flag replaces hardcoded `if branch_key == "scenes"` |

### Handler Logic (`discovery.py`)

```python
def _handle_check_archive(
    self,
    branch_name: str,
    data: dict,
    formatted_data: FormattedData,
    branch_schema_class: ParsedSchemaClass,
    keys: list,
) -> dict:
    """Schema-driven archive boundary handler.
    
    Replaces check_and_archive_chapter/arc and the ADD_NEW archive special-case.
    All configuration is read from branch_schema_class.defaults.
    """
    d = branch_schema_class.defaults or {}
    
    # 1. Cadence gate
    span_source = d.get("span_source")
    if span_source:
        end_field = d["span_end_field"]
        child_unit = span_source  # "scenes" or "chapters"
        suggested_min = d.get(f"suggested_min_{child_unit}", 1)
        max_hard = d.get(f"max_{child_unit}_before_required", suggested_min)
        
        # Count units since last boundary
        source_data = self.all_subjects_data.get(
            "events" if span_source != "chapters" else "events", {}
        ).get(span_source, {})
        # ... count logic (see below)
        
        if count < suggested_min:
            print(f"Archive gate: {count} {child_unit} < min {suggested_min}. Skipping.")
            return data
    
    # 2. Gate check LLM call (skip if no template)
    gate_template = d.get("check_archive_prompt_template")
    force = self._check_force_override(d.get("numbering_field"))
    if gate_template and not force:
        gate_prompt = render_jinja_template(gate_template, ...)
        response, stop = self.summarizer.generate_using_tgwui(
            prompt=gate_prompt, ..., stopping_strings=["YES", "NO"], match_prefix_only=True
        )
        if "NO" in response.upper() and count < max_hard:
            print(f"Archive gate: LLM says NO. Skipping.")
            return data
    elif force:
        print(f"Forcing archive via {d['numbering_field']} override.")
    
    # 3. Span math (if applicable)
    span = None
    if span_source:
        span = self._compute_archive_span(data, span_source, end_field)
    
    # 4. Execute: _archive_boundary_entry (existing, already schema-driven)
    data = self._archive_boundary_entry(
        branch_name, data, formatted_data, branch_schema_class, keys
    )
    
    # 5. Numbering increment
    num_field = d.get("numbering_field")
    if num_field:
        current_scene = self.all_subjects_data.get("current_scene", {})
        current_scene[num_field] = current_scene.get(num_field, 1) + 1
    
    # 6. (Canon digest is a separate trigger, see below)
    
    return data
```

### `_archive_boundary_entry` Changes

Minimal — just make `key_field` and `stamp_message_node` schema-driven:

```python
# Before (hardcoded):
key_field = "name" if branch_key == "scenes" else "title"

# After (schema-driven):
key_field = (branch_schema_class.defaults or {}).get("key_field", "title")

# Before (hardcoded):
if branch_key == "scenes":
    current_message_node = self._resolve_current_message_node()
    ...

# After (schema-driven):
if (branch_schema_class.defaults or {}).get("stamp_message_node"):
    current_message_node = self._resolve_current_message_node()
    ...
```

### `Action.ADD_NEW` Simplification

Remove the `archive_template` special-case from `_execute_action`:

```python
# Before:
if action == Action.ADD_NEW:
    if target_schema_class.definition_type == "alias":
        archive_tpl = (target_schema_class.defaults or {}).get("archive_template")
        if archive_tpl:
            self._archive_boundary_entry(...)  # archive path
        else:
            self._detect_and_add_new_entries_to_branch(...)  # normal path

# After:
if action == Action.ADD_NEW:
    self._detect_and_add_new_entries_to_branch(...)  # always normal path
```

Archives are now exclusively `check_archive`. The `ADD_NEW` action is purely for "discover new entities from the scene" (characters, elements, knowledge, etc.).

### Canon Digest: Separate Trigger

Add to `Chapters` and `Arcs` definitions:

```json
{
  "Chapters": {
    "triggers": {
      "on_new_scene": ["check_archive"],
      "on_archive_success": ["regenerate_canon_digest"]
    }
  }
}
```

This requires a new `Trigger` value: `ON_ARCHIVE_SUCCESS`. The `regenerate_canon_digest` action is a thin handler that calls `self._regenerate_canon_digest(trigger_name)`.

**Alternative (simpler, no new trigger):** The `check_archive` handler calls `_regenerate_canon_digest` directly after a successful archive, gated by a schema flag:

```json
{ "regenerate_digest": true }
```

I prefer the **flag** approach for v1 — it avoids adding a new trigger type and the dispatch complexity. The separate-trigger approach is cleaner long-term but can be a follow-up.

### Force Override

Currently: `self.summarizer.last.force_next_chapter = True`

Schema-driven: the `check_archive` handler checks for a force flag on `self.summarizer.last` using the `numbering_field` as the key:

```python
def _check_force_override(self, numbering_field: str) -> bool:
    if not self.summarizer.last:
        return False
    # force_next_chapter, force_next_arc, force_next_scene
    force_attr = f"force_next_{numbering_field.lstrip('_').replace('_number', '')}"
    return getattr(self.summarizer.last, force_attr, False)
```

This keeps the existing force mechanism working without schema changes.

### `core.py` Changes

**Remove:** `_run_boundary_checks` method entirely.

The chapter/arc checks now fire through the normal trigger dispatch when `ON_NEW_SCENE` triggers are processed. No imperative call needed.

### Schema Changes Summary

| Definition | Change |
|------------|--------|
| `StoryEvents` | `name` → `title` (field rename) |
| `Scenes` | Add `triggers.on_new_scene: ["check_archive"]`; add archive defaults; remove inherited `add_new` |
| `Chapters` | Add `triggers.on_new_scene: ["check_archive"]`; add archive defaults; `storage: "dict"` |
| `Arcs` | Add `triggers.on_new_scene: ["check_archive"]`; add archive defaults |
| `KnowledgeMap` | Unchanged (still `add_new`) |
| `CharacterMap` | Unchanged |
| `ElementMap` | Unchanged |

**Critical:** `Scenes` currently inherits `add_new` from `StoryEvents`. We give `Scenes` its **own** `triggers` block (`{"on_new_scene": ["check_archive"]}`), which completely replaces `StoryEvents`'s triggers per the all-or-nothing rule above. No parser change needed.

### Storage Consolidation: Chapters list → dict

Current `events.chapters` is a list:
```json
{"chapters": [{"title": "Chapter 1", ...}, {"title": "Chapter 2", ...}]}
```

Target:
```json
{"chapters": {"Chapter 1": {...}, "Chapter 2": {...}}}
```

This requires:
1. Schema: `Chapters` type changes from `list[Chapter]` to `dict[str, Chapter]`
2. Code: `_entries_as_list` helper already handles both shapes (it's used in `check_and_archive_*` to normalize)
3. Existing run data: no migration needed (chapters were always 0 in all runs)

### File Impact

| File | Change |
|------|--------|
| `utils/schema_parser.py` | Rename `QUERY_FOR_ARCHIVE` → `CHECK_ARCHIVE` |
| `agents/data_summarizer/core.py` | Add `CHECK_ARCHIVE` branch to `_execute_action`; remove `_run_boundary_checks` |
| `agents/data_summarizer/discovery.py` | Add `_handle_check_archive`; modify `_archive_boundary_entry` (key_field, stamp flag) |
| `agents/data_summarizer/archives.py` | Delete `check_and_archive_chapter`, `check_and_archive_arc` (or keep as deprecated wrappers) |
| `agents/summarizer/core.py` | Remove `_run_boundary_checks` call site |
| `subjects_schema_sceneagg.json` | Add triggers + defaults to Scenes/Chapters/Arcs; rename `name`→`title` |
| `scripts/gen_noschema.py` | Regenerate `_noschema` |

### Migration / Compatibility

- **Existing run data:** No migration needed. Scenes/Chapters/Arcs were all empty in every run to date.
- **`_entries_as_list`:** Already handles both list and dict shapes — no change needed.
- **`expand_lists_in_data_for_llm`:** Needs a check that `dict[str, Chapter]` expansion works (it currently handles `list[Chapter]`). The alias-hint path in `FormattedData.__init__` should handle this.
- **Writer prompts:** The `events.chapters` section in writer context uses the list shape. Changing to dict changes the rendered context format. Need to verify `context_engine.py` handles both.

### Remaining Design Decisions

1. **`span_source` resolution:** For `Chapters`, `span_source: "scenes"` means "count scenes in `events.scenes`". The handler resolves this to `self.all_subjects_data["events"][span_source]`. For v1, span source is always under `events.*`. If a future type needs to aggregate across subjects, the value can become a dotted path (e.g., `"events.scenes"`).

2. **Gate check prompt variables:** The generic handler computes the standard set from schema config: `scenes_count`/`chapters_count` (from span source), `current_scene_number`/`current_chapter_number`/`current_arc_number` (from `current_scene`), `scenes_in_chapter`/`chapters_in_arc` (span math), `suggested_min`/`suggested_max`/`max_*_before_required` (from defaults), and `span_digest` (compact summaries from the source data). The variable names are derived from `span_source` + `numbering_field`.

3. **`_archive_boundary_entry` template vars:** `_get_archive_template_vars` stays as-is for v1. It's already parameterized by `branch_key` and `current_scene`. Full schema-driven template vars (a `template_vars` config listing which variables to compute) is a follow-up if/when new boundary types need non-standard vars.

### Implementation Order

1. Rename `QUERY_FOR_ARCHIVE` → `CHECK_ARCHIVE` in `Action` enum
2. Add `key_field` + `stamp_message_node` to `_archive_boundary_entry` (schema-driven)
3. Write `_handle_check_archive` in `discovery.py`
4. Add `CHECK_ARCHIVE` branch to `_execute_action` in `core.py`
5. Update schema: add triggers + defaults to Scenes/Chapters/Arcs
6. Remove `_run_boundary_checks` from summarizer core
7. Deprecate `check_and_archive_chapter`/`arc` in `archives.py`
8. Rename `StoryEvents.name` → `title`
9. Change `Chapters` type from `list[Chapter]` to `dict[str, Chapter]`
10. Regenerate `_noschema`
11. Smoke test: verify trigger dispatch fires `check_archive` for all three types