# Chapter & Arc Archival Plan

Extends the scene archival pattern (commit `43ee1f6e`) to chapters and arcs.

## Current state

| Boundary | Trigger | Boundary detection | Entry creation |
|----------|---------|-------------------|----------------|
| Scene | `ON_NEW_SCENE` | Deterministic (`_scene_number` increment) | **Programmatic + LLM refine** (done) |
| Chapter | `ON_NEW_SCENE` | LLM gate check (`gate_check_prompt_template`) | Generic LLM `add_new` (query + generate) |
| Arc | `ON_NEW_SCENE` | LLM gate check (`gate_check_prompt_template`) | Generic LLM `add_new` (query + generate) |

The gate check already works — it correctly detects chapter/arc boundaries. The problem is the entry creation step: the LLM generates all fields from scratch, which is slow and produces inconsistent structural data (scene indices, chapter indices).

## Target flow

All three boundaries use the same 3-step pattern:

1. **LLM boundary detection** (unchanged) — gate check decides *when* the boundary is reached
2. **Programmatic data** (new) — render a Jinja archive template against known state to pre-fill deterministic fields
3. **LLM refinement** (new) — pass the pre-filled entry as `{{ temp_data }}`; LLM names it and fills narrative fields

## Schema changes

### `Chapters.defaults` — add:

```
chapter_archive_template: (Jinja)
  Renders a pre-filled Chapter object:
  - title: "CHAPTER #{{ chapter_number }}"
  - starting_scene: {{ starting_scene }}
  - ending_scene: {{ ending_scene }}
  - scenes: {{ scene_span | tojson }}
  - summary: ""
  - key_changes: []
  - status: "complete"

new_entry_prompt_template: (updated)
  Same pattern as Scenes — "here's the pre-filled data in {{ temp_data }},
  give it a proper title and fill the empty narrative fields"
```

### `Arcs.defaults` — add:

```
arc_archive_template: (Jinja)
  Renders a pre-filled Arc object:
  - title: "ARC #{{ arc_number }}"
  - starting_chapter: {{ starting_chapter }}
  - ending_chapter: {{ ending_chapter }}
  - chapters: {{ chapter_span | tojson }}
  - summary: ""
  - character_arcs: ""
  - relationship_shifts: ""
  - plot_threads: []
  - status: "complete"

new_entry_prompt_template: (updated)
  Same pattern as Scenes
```

## Code changes

### 1. Generalize `_archive_scene_from_current_scene` → `_archive_boundary_entry`

The current method is scene-specific (reads `current_scene`, builds `participants` dict). Generalize it:

```python
def _archive_boundary_entry(self, branch_name, data, formatted_data,
                            branch_schema_class, keys):
    # 1. Determine the archive template key from the branch name:
    #    scenes → scene_archive_template
    #    chapters → chapter_archive_template
    #    arcs → arc_archive_template
    #
    # 2. Compute programmatic data based on branch type:
    #    scenes:   current_scene → participants dict
    #    chapters: events.scenes + current_scene → scene span
    #    arcs:     events.chapters + current_scene → chapter span
    #
    # 3. Render the Jinja template → temp_data (JSON string)
    # 4. Build refine prompt with temp_data as kwarg to _create_update_prompt
    # 5. Single LLM call → parse → derive key from "name"/"title" → store
    # 6. Fail-open: store programmatic entry on LLM failure
```

The branch-specific data extraction is the only part that differs. It can be a small dispatch:

```python
if branch_name.lower() == "scenes":
    template_vars = self._scene_archive_vars()
elif branch_name.lower() == "chapters":
    template_vars = self._chapter_archive_vars()
elif branch_name.lower() == "arcs":
    template_vars = self._arc_archive_vars()
```

### 2. Programmatic data sources

**Chapters** (`_chapter_archive_vars`):
| Field | Source |
|-------|--------|
| `chapter_number` | `current_scene._chapter_number` |
| `ending_scene` | `current_scene._scene_number` |
| `starting_scene` | `last_chapter.ending_scene + 1` (or 1 if no prior chapter) |
| `scene_span` | `list(range(starting_scene, ending_scene + 1))` |

**Arcs** (`_arc_archive_vars`):
| Field | Source |
|-------|--------|
| `arc_number` | `current_scene._arc_number` |
| `ending_chapter` | `current_scene._chapter_number` |
| `starting_chapter` | `last_arc.ending_chapter + 1` (or 1 if no prior arc) |
| `chapter_span` | `list(range(starting_chapter, ending_chapter + 1))` |

### 3. Routing in `_execute_action`

Generalize the current scene-specific check:

```python
# Current (scene-only):
archive_tpl = (target_schema_class.defaults or {}).get("scene_archive_template")

# New (generic):
archive_tpl = (target_schema_class.defaults or {}).get(
    f"{branch_name.lower()}_archive_template"
)
```

This makes it fully schema-driven: any branch that defines a `*_archive_template` gets the programmatic archival path. No code change needed when adding new boundary types.

### 4. Rename the stored key

Scenes use `name` as the entry key. Chapters and Arcs use `title`. The refine step needs to derive the key from the right field:

```python
key_field = "name" if branch_name.lower() == "scenes" else "title"
final_name = refined.get(key_field, programmatic_name)
```

## Files to change

| File | Change |
|------|--------|
| `subjects_schema_sceneagg.json` | Add `chapter_archive_template` + update `new_entry_prompt_template` for `Chapters`; same for `Arcs` |
| `discovery.py` | Rename `_archive_scene_from_current_scene` → `_archive_boundary_entry`; add `_chapter_archive_vars` + `_arc_archive_vars` |
| `core.py` | Generalize the archive template key lookup in `_execute_action` |

## Open questions

1. **Key changes for chapters**: the `key_changes` field is `list[KeyChange]` where `KeyChange = {description, scene}`. Should the programmatic template pre-fill an empty list, or should the LLM be prompted to extract key changes from the scene recap? Leaning: empty list — the LLM fills it during refinement, and the scene recap is already in the prompt context.

2. **Arc `character_arcs` / `relationship_shifts`**: these are narrative fields that benefit from the full chapter digest. The refine prompt should include the chapter span's summaries as context (the `chapters_digest` variable already exists in `_create_update_prompt`).

3. **Testing**: smoke test with a short run (20-30 turns) to verify chapters and arcs are being archived with correct structural data. Check `events.json` for `chapters` and `arcs` keys.