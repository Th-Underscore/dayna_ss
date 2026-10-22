# Branch Update Prompt Diagnosis

Smoke run: `sci_fi_heist__61dd3e94` (seed 206, 20 turns, Gemma-4-26B)
Log: `extensions/dayna_ss/tests/runs/logs/scifi_heist_gemma4_smoke20_206.log`

45 error lines across 4 categories. Root cause: the `branch_update_prompt_template` is ambiguous about path scope, doesn't handle empty entries, and doesn't disambiguate field types in its examples.

---

## 1. Prompt Template (Verbatim from Schema)

The `branch_update_prompt_template` (shared across most dataclass/alias definitions in `subjects_schema.json`):

```
Based on the latest exchange, review the current state of the entry below.
Relevant Schema:
```json
{{ schema_snippet }}
```

Example JSON structure:
```json
{{ example_json }}
```

If any specific fields within the entry require updates (additions, modifications, or deletions), please provide a list of these changes.
Each change should specify the 'path' to the field and the 'value' for the update.
The path should be a dot-separated str representing the nested structure, using numeric indices or square brackets for list items (e.g., 'list_field.0.attribute'). If a key name contains a dot, wrap it in square brackets (e.g., '[Mrs. Patterson]').

Example of expected output if updates are needed:
[
  {"path": "description.0", "value": "Updated first line of description"},
  {"path": "status.current_location", "value": "New Location"},
  {"path": "relationships.Bob.0.status", "value": "ally"}
]

If NO updates are required for this entry, respond with the exact str: NO_UPDATES_REQUIRED

Please provide your response in the specified JSON format or the 'NO_UPDATES_REQUIRED' str.



Current entry state (the entry's actual stored data — diff against this ground truth; do NOT restate, re-derive, or re-derive-from-recall any of it):
```
{{ value }}
```


CANONICAL KEY CONSTRAINT — Every referent you emit (a relationship partner, an alias, a participant, a milestone subject) MUST resolve to a key already present in the entry list, or to this entry's own keys. If the referent denotes a NEW entity, it belongs in an add_new proposal, not in a relationship/alias value here. Never write a partner or alias that names a DIFFERENT existing entry.

Entry to review: '{{ branch_name }}'
```

The StoryEvents variant (used for `events.beats`, `events.past`) differs in the path example:
```
The path should be a dot-separated str representing the nested structure (e.g., 'Example Event.summary', 'Example Event.start.date').
```

---

## 2. Error Categories & Prompt Deficiencies

### Category A: Branch-name prefix in paths (`events.beats` as path)

**Log evidence (lines 4169, 7660, 15848):**
```
[events.beats] REJECTED update (parent path missing): events.beats
```

The LLM outputs `{"path": "events.beats", "value": "..."}` — it's including the branch name itself as the first path segment. The system rejects it because the path is supposed to be relative to the entry, and `events.beats` is the entry itself.

**Root cause in the prompt:**
- The prompt says "The path should be a dot-separated str representing the nested structure" but never explicitly states that paths are **relative to the entry being reviewed** and must NOT include the entry's own name.
- The StoryEvents example paths use the event name as the first segment: `'Example Event.summary'`. This is correct for a dict-of-events entry (where the top-level keys are event names), but the LLM generalizes this pattern to include the branch name `events.beats` as a prefix.
- The final line "Entry to review: '{{ branch_name }}'" identifies the entry but doesn't say "paths below are relative to this entry."

**Fix:** Add an explicit scope statement:
```
PATH SCOPE — All paths you emit are RELATIVE to the entry identified below. Do NOT include the entry name (e.g., 'events.beats', 'general_info') as a path prefix. The first segment of your path must be a top-level key WITHIN the entry.
```

### Category B: Non-existent keys referenced (`relationships.Dex.0.status` when Dex doesn't exist)

**Log evidence (lines 9164-9167, 16005, 19820-19824):**
```
[elements.entries.coolant housing] REJECTED update (parent path missing): relationships.Dex.0.status
[elements.entries.coolant housing] REJECTED update (parent path missing): relationships.Dex.0.importance.score
```

The LLM references `relationships.Dex` as if it exists in the element entry, but the element only has `relationships` as a dict that may be empty or not contain "Dex" as a key.

**Root cause in the prompt:**
- The `{{ value }}` section shows the actual entry data, which should reveal that `Dex` is not a key in `relationships`. However:
  1. The CANONICAL KEY CONSTRAINT says referents "MUST resolve to a key already present in the entry list" — but this is phrased for relationship *partners* (the values), not for the *keys* in the relationships dict. The LLM interprets "Dex" as a valid referent (a character in the story) and doesn't check whether it's actually a key in the entry's relationships dict.
  2. The prompt doesn't say "Only emit paths whose intermediate keys exist in the Current entry state shown below."
  3. The example `relationships.Bob.0.status` reinforces the pattern without showing what happens when Bob isn't in the entry.

**Fix:** Strengthen the constraint:
```
PATH VALIDITY — Every path you emit must be fully resolvable against the Current entry state shown below. Do NOT emit a path that traverses a key not present in that state. If a relationship partner is not yet a key in the entry's relationships dict, do NOT propose an update to it — that is an add_new operation, not an update.
```

### Category C: Prose refusals when entry is empty

**Log evidence (lines 969, 3902, 7315, 7864, 8990, 9215):**
```
[events.past] Failed to parse LLM response... Raw response: Since no existing entry for `events.past` was provided in the "Current entry state," there is no ground truth to compare against. Therefore, I cannot perform a diff or identify necessary updates to an empty object.

[events.beats] Failed to parse LLM response... Raw response: Since there are no existing entries under the key `events.beats` to review (the current state is an empty object `{}`), I cannot perform an update.
```

**Root cause in the prompt:**
- When `{{ value }}` is `{}` (empty dict), the LLM correctly identifies it as empty and refuses to operate on it.
- The prompt only offers two outputs: a JSON list of updates, or `NO_UPDATES_REQUIRED`. There is no instruction for what to do when the entry is empty.
- The LLM is logically correct: you can't diff against an empty object. But the system expects it to either return `NO_UPDATES_REQUIRED` or create content.
- The prompt's framing ("review the current state... if any fields require updates") implies there IS something to review, which contradicts the empty state.

**Fix:** Add an explicit empty-entry instruction:
```
EMPTY ENTRY — If the Current entry state is empty (`{}` or `[]`), respond with the exact str: NO_UPDATES_REQUIRED. Do not explain, refuse, or request additional context.
```

### Category D: Dict-style paths on string fields (`writing_style.0 = '...'`)

**Log evidence (lines 11541, 12128, 13077, 13079, 16741, 16743, 22352):**
```
[general_info] Failed to apply update writing_style.0 = 'Dex: 1st-person present tense prose...': 'str' object does not support item assignment
[general_info] Failed to apply update writing_style.Dex_specific = "1st-person present tense...": 'str' object does not support item assignment
```

The schema clearly defines `writing_style: str` (a plain string field). But the LLM emits `writing_style.0` (treating it as a list) or `writing_style.Dex_specific` (treating it as a dict).

**Root cause in the prompt:**
- The path format example says "using numeric indices or square brackets for list items (e.g., 'list_field.0.attribute')". This primes the model to use `.0` indexing liberally.
- The example output shows `{"path": "description.0", ...}` — for the Character entry, `description` IS a list (Text = list[str]). But for GeneralInfo, `writing_style` is a plain `str`.
- The `{{ schema_snippet }}` DOES show the field types, but the LLM (a 26B model) doesn't reliably cross-reference the schema types against its path choices. The examples are more salient than the schema.
- There's no instruction saying "if a field is a scalar (str), the path is just the field name with no index."

**Fix:** Add a type-awareness instruction:
```
PATH TYPES — Check the Relevant Schema for each field's type before emitting a path. Scalar fields (str, int) use the bare field name as the full path (e.g., "writing_style", not "writing_style.0"). List fields use numeric indices (e.g., "description.0"). Dict fields use the key name (e.g., "relationships.Dex.status").
```

---

## 3. Summary of Fixes (Priority Order)

| # | Fix | Addresses | Effort |
|---|-----|-----------|--------|
| 1 | Add "PATH SCOPE" line: paths are relative to the entry, never include the entry name | Category A (3 errors) | 1 line |
| 2 | Add "EMPTY ENTRY" instruction: return NO_UPDATES_REQUIRED when value is `{}` | Category C (6+ errors) | 1 line |
| 3 | Add "PATH TYPES" instruction: scalar fields use bare name, no index | Category D (7+ errors) | 2-3 lines |
| 4 | Strengthen CANONICAL KEY CONSTRAINT to require full path resolvability against the shown state | Category B (10+ errors) | 2 lines |

All four fixes are additive lines to the existing `branch_update_prompt_template`. They don't restructure the prompt.

---

## 4. Additional Observations

- The `general_info` `objectives.0.*` rejections (lines 933-936) are a Category B variant: the LLM proposes `objectives.0.description` but `objectives` is an empty list `[]`, so index 0 doesn't exist.
- The `current_scene.now.who.characters.1.*` rejections (lines 10026-10027, 11592-11593, etc.) are the same: index 1 doesn't exist in a single-element list.
- The prompt's "diff against this ground truth" language is good intent but the LLM (especially a 26B model) treats it as a permission to be creative rather than a constraint.