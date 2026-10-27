# DSS Quality Recommendations

Scope: state quality for LLM roleplay (current goal), kept general enough for other memory-system uses later.
Evidence base: the `scifi_heist` seed-524 soak (100 turns, `compressed` cadence) and the v0.2.0 source. Performance work is tracked separately.

## Summary

| # | Item | Type | Extra LLM cost | Priority |
|---|------|------|----------------|----------|
| 1 | Fix `description` semantics (stable vs. current status) | Schema fix | Negative (fewer tokens) | Do first |
| 2 | Epistemics (who knows what) | New field/subject | 1 call per scene | High |
| 3 | Threads (ongoing/recurring/open items) | New subject + `StoryEvent` fields | 0–1 call per scene | High |
| 4 | Change history (per-field diffs) | Engine hook | 0 | High, cheap |
| 5 | Goals / intent | New field | Folded into scene-boundary update | Medium |
| 6 | Spatial and physical state | Mostly code-derived | 0 | Medium |
| 7 | Event provenance and milestone types | Schema fields + dedup guard | ~1–3 tokens per item | High (provenance), Medium (types) |
| 8 | Other clarification enums | Small schema fields | ~1–3 tokens per item | Low–Medium |
| 9 | Values for characters (`red_lines` for ideological groups) | New field, set at creation | ~100–200 tokens per entry | Medium |
| 10 | Shared `source` convention and precedence rule | Enum + code guard | ~1 token per item | Medium |

Relationship asymmetry is already handled. Bidirectionality exists for entity-graph retrieval, not for the relationship data itself, so there is nothing to add.

## Design constraints

Every addition below should respect what the timing data showed, otherwise quality gains get paid for in minutes per turn.

- **No new per-entry LLM calls.** Entry-level patch calls were ~57% of DSS time. New information should ride on existing calls or on one batched call per scene boundary.
- **Prefer code-derived over LLM-derived.** If a value can be computed from `current_scene`, event participants or an applied-update log, compute it.
- **Patch-style, append-bounded fields.** The path/value patch format already works (path-scoped, `NO_UPDATES_REQUIRED` as the cheap exit). New fields should fit it and should not become unbounded logs.
- **Scene-boundary batching.** The same trigger the heavy subjects already use (`on_new_scene`).

## 1. Fix `description` semantics first

### Evidence

- 517 of 576 applied character updates (90%) target `description.*`.
- `Dex.description` reached 71 items (10.9k chars) by turn 99. `Sable.description` reached 20 items (4.8k chars).
- The items are not exact duplicates (hygiene collapses those). They are paraphrased situation snapshots, for example "she now maintains position behind a corroded industrial coupling…", appended every scene.
- Characters are the decode hogs: 101 patch calls, ~31 s decode each, only 6% no-ops.
- The same pattern appears elsewhere: the `events` list on elements (`modified keypad`: 49 items, 7k chars) and on groups (`Security signatures`: 31 items).

### Cause

The field placeholder describes "the character's current state, appearance, and how they present themselves". That mixes three different things: stable appearance, current status, and a history log. The model resolves the ambiguity by appending a new snapshot each scene.

### Recommendation

Split by lifetime:

| Field | Semantics | Update rule |
|-------|-----------|-------------|
| `description` | Stable appearance, demeanor, manner | Rewrite-only. Soft cap (for example 3 items). Changes rarely |
| `status` | Current physical/situational state, single string | Overwritten, never appended |
| (history) | What happened to them | Lives in events and the change log (item 4), not in `description` |

Apply the same split to any `list[str]` field that is acting as a log: elements' `events`, groups' `events`. Where a list really is a log, give it a hard cap and let hygiene drop or merge the oldest.

Expected effect: smaller entries in every prompt, far less decode on character calls, and descriptions the writer can actually use.

## 2. Epistemics

Roleplay failures where a character acts on something they shouldn't know are the most visible memory bugs, and nothing in the schema currently models knowledge.

### Shape (sketch, adapt to the schema DSL)

```
Knowledge item:
  fact:          str           # one atomic statement
  known_by:      list[str]     # character/group names
  secret_from:   list[str]     # explicit "must not know"
  status:        enum          # true | false | suspected | rumor
  source:        str           # witnessed | told_by:<name> | inferred | overheard
  learned_scene: str           # scene/event name
```

Stored once in a shared `knowledge` subject. Per-character and per-group views are derived, not separately generated.

### How to populate it cheaply

One call at the scene boundary, over the scene recap:

> Who learned or concealed what in this scene? Return only new or changed items.

The code then fans the result out into each character's and group's knowledge view. This is 1 call per scene, not N entry calls. It also keeps `known_by` consistent by construction, since there is a single source of truth.

### Retrieval

- Include a character's knowledge view in context only when they are in the current scene or about to speak.
- Always include `secret_from` constraints for characters in the scene. Those are the entries that prevent leaks.

## 3. Threads (ongoing and recurring items)

"Threads" covers promises, debts, mysteries, foreshadowing, schedules and recurring events.

### Part of it already exists

Events that stay in progress are already being maintained as threads, just implicitly. In the soak, 44% of applied event updates (310 of 700) were `end.*` time bumps and 24% (165) were `outcome` rewrites on events still in progress. Make this explicit instead of letting it happen by accident.

### Recommendation

Add to `StoryEvent`:

- `status`: `open | resolved | dormant`
- `cadence` (optional): for recurring events, for example "nightly patrol", "every third scene"

Add a thin `threads` subject for items that are not events:

```
Thread:
  kind:               promise | debt | mystery | foreshadowing | schedule | recurring
  owner:              str
  summary:            str
  due:                str | null     # when relevant
  cadence:            str | null
  status:             open | resolved | dormant
  last_touched_scene: str
```

### Make most of it deterministic

- While an event is `open` and overlaps the current scene, set its `end.*` from `current_scene.now.when` in code. This removes close to half of the events patch output.
- Compute "stale" (open and `last_touched_scene` far behind) and "due" in code. Surface stale threads as reminders in context.
- Only creation and resolution need an LLM. They can share the scene-boundary call from item 2 (for example, "new or resolved threads").

## 4. Change history

The goal is a git-diff-style record per field, so the system can answer "what changed, and when" without rescanning old scenes.

### Hook point

The update-apply path already computes `old_value` for every validated update; the soak log prints `Applying update: <path> == <old value>`. Persist it instead of discarding it:

```
_history: list of { turn, scene, path, old, new, reason? }
```

- Store it in a private field (leading underscore) that the traversal skips, so it adds no per-turn prompt cost.
- Zero extra LLM calls.
- Optional `reason`: leave empty by default. Only populate when the model supplied one in the patch.

### Using it

- Retrieve on demand, for example via the existing `dss_get_info` tool or when a field is queried, not by default.
- Relevance filter at read time: skip `importance`-only and time-bump changes (`end.*`), show description/relationship/status changes.
- Compaction: when a field's history exceeds a cap, collapse old entries into one summary line per scene.

The change log also makes item 1 enforceable. If a field changes every scene, that is visible and can be flagged.

## 5. Goals / intent

Short-term intent is what separates a character that reacts from one that acts.

```
Character.goals:
  - want:      str       # what they are trying to achieve now
    hiding:    str|null  # what they are concealing (links to knowledge.secret_from)
    blocked_by: str|null
    horizon:   scene | arc
```

- Fold the update into the scene-boundary call, not into per-entry patches.
- Keep it small (1–3 items per character). Overwrite rather than append.
- `hiding` and `secret_from` should reference the same facts, so goals and epistemics stay consistent.

## 6. Spatial and physical state

`current_scene` already tracks where characters are now. The gap is "where are they when not in the scene", and the system must not hallucinate movement.

### Derive in code

After each turn, update from `current_scene.now.who`:

```
last_seen:
  scene:    <scene name>
  location: <location at that time>
  turn:     <turn number>
```

Zero LLM calls.

### Handle implicit movement honestly

Do not guess where an unseen character went. Store age and confidence instead:

- Surface "last seen at X, N scenes ago" in context, not "is at X".
- Optionally downgrade `confidence` as scenes pass, and mark `uncertain` if the scene location changed substantially since they were last seen.
- Only an explicit statement ("Dex left for the docks") should set a new location, and that would come from the scene-boundary call.

### Physical state

Keep a small struct on characters and groups: `held_items`, `injuries`, `attire`. Treat it as overwritten state like `status` in item 1. The `elements` subject covers object continuity but not who currently holds what, so add `held_by` there or derive it from character `held_items`. Pick one source of truth.

## 7. Event provenance and milestone types

### Evidence

- **Milestones are never populated.** Both characters had zero milestones at every one of the 100 turns, and the log has no milestone-related validation failures. A type enum alone would do nothing until something prompts creation.
- **Milestones cannot link to events.** `CharacterMilestone` is `{title, scenes: list[int], importance}`, while events are keyed by name. As far as I can see, nothing resolves a scene number to an event.
- **The container is a weak backstory classifier.** The prompt says `past` is BACKSTORY ONLY and that anything narrated now is not backstory. In the soak, all 28 `past` entries carry an in-story message node (for example "The Severed Security Cable"). The same occurrence is also stored across tiers: 14 of 17 `past` entries share a start node with a scene or beat, and 20 of 35 beats share one with a scene.

### Where each attribute should live

Put each attribute at the level where it is true by definition, and derive the rest.

| Attribute | Lives on | Why |
|-----------|----------|-----|
| `origin`: `backstory \| on_page \| off_screen \| reported` | The event | A property of the occurrence. `off_screen` and `reported` also feed epistemics (who could have witnessed it) |
| Milestone `type` | The event participant (`EventParticipant`) | One event can be a feat for one character, a loss for another and a revelation for a third |
| `character.milestones` | Derived view | Built in code from participants above an importance threshold |

The event's `add_new` call already emits `participants`, so a type per participant adds a few output tokens and no new call. That also addresses the empty-milestones problem without a per-character call.

If you prefer to keep a stored milestone list, give `CharacterMilestone` an `event` key and a `type`, and drop `scenes: list[int]`. Characters with pre-story milestones can point at a `past` event. Milestones inherit `origin` through the event link.

### Milestone type taxonomy

Keep it to 6–8 mutually exclusive values with a one-line decision rule each, for example:

- `feat`: an achievement through action
- `setback`: a loss or failure
- `revelation`: learned or discovered something significant
- `bond` / `betrayal`: a relationship shift
- `transformation`: a change in identity, ability, allegiance or status
- `oath`: a commitment (links to threads, item 3)

A revelation overlaps with epistemics (item 2). Derive revelation milestones from `knowledge` items whose `source` is learned, instead of generating the same fact twice.

### Guardrails

- **Cross-tier dedup in code.** Reject a `past` add whose start node matches an existing scene or beat entry. This fixes the misclassification and saves `add_new` calls; events `add_new` was 2,673 s in the soak.
- **Validate `origin` against the node.** If a `past` event's message node falls inside the story's message range, flag or re-tag it `on_page` rather than trusting the model. Check first whether `_message_node` is stamped at creation or at event start, since stamping at creation cannot distinguish a backstory event referenced at that message.
- **Coerce enums in code.** Map unknown values to the nearest valid one or a default. Do not retry.

## 8. Other clarification enums

Each costs roughly 1–3 output tokens per item.

| Field | Values | Where | Benefit |
|-------|--------|-------|---------|
| `outcome_status` | `success \| partial \| failure \| ongoing \| unresolved` | Event, next to free-text `outcome` | Lets code drive thread `status` (item 3). Replaces part of the `outcome` rewrites, which were 24% of events updates |
| `seq` | integer, code-assigned from message-node order | Event | Dates were `Unknown` or `N/A` in the sampled entries, so ordering is unreliable without it |
| `source` | `witnessed \| told \| inferred \| overheard` | Knowledge item (item 2) | Separates what a character saw from what they were told |
| `condition` | `intact \| damaged \| lost \| consumed` | Element | Physical continuity (item 6) |
| `lifecycle` | `active \| absent \| deceased \| unknown` | Character | Only if not already tracked; I did not check |

### Cost note

The expense of an enum is not its output tokens. It is the placeholder text and example JSON it adds to every call's prompt, which currently sits in the uncached part of each prompt. Define each taxonomy once in the system context, the way the universal Importance Scale already is, and keep per-field placeholders to a single line.

## 9. Values for characters

### Why

Characters currently have `traits`, `attributes`, `voice`, `quirks`, `fears` and `preferences`, but nothing that explains why they act. Traits are adjectives (Sable: "sharp", "paranoid"), and `fears` is the only driver. A value with a built-in tension, for example "the longer you survive, the longer you can help", versus heroic instinct, tells the writer how the character resolves a dilemma under pressure.

### Shape (2–4 items per character)

```
Value:
  maxim:           str          # in their own words
  drives:          str          # the behavior it produces
  in_tension_with: str | null   # another value, fear or instinct it fights
  source:          stated | demonstrated | inferred   # see section 10
  coping:          str | null   # optional: how they justify breaking it
```

- **Populate at creation.** Use the existing `add_new` call, so there is no new call.
- **Revise rarely.** Overwrite only, with a cap. Revise when a milestone of type `transformation`, `betrayal` or `oath` is created (section 7). Otherwise it becomes another `description`-style log (section 1). The change log (section 4) then shows value drift, which is character development.
- **`coping` is optional.** It feeds inner monologue and dissonance, and it is the easiest part to over-generate. Add it only if that depth is wanted.
- **Relationship to nearby fields:**

| Field | Lifetime | Role |
|-------|----------|------|
| Values | Stable | Why they act |
| Goals (section 5) | Short-term | What they want now |
| `fears` | Stable | Negative driver |
| Knowledge and secrets (section 2) | State | What they know and hide |
| `voice`, `quirks` | Stable | How they sound; a literal catchphrase belongs here |

- **Writer context.** Include a one-line summary only for characters in the current scene, for example "Values: survive to help (tension: heroic instinct)".
- **Cost.** About 100–200 tokens per entry, small next to a ~3k-token character entry. Fix the `description` log first, otherwise it keeps inflating every character prompt.

### Groups

Skip a motto field. Groups already have `purpose` ("goals and motivations") and `culture` ("norms, values, traditions"). In the sample, the three corporate groups have near-identical `purpose` and `culture` text ("absolute control", "rigid", "absolute complicity"), so a new field would add boilerplate. Values also mean little for entities like a security grid or a patrol drone, so any new field must be allowed to stay empty.

For factions with real ideology, add `red_lines` instead: what they will always or never do. That predicts reactions to the party better than a motto does.

## 10. Shared `source` convention and precedence rule

`source` earns its place only where something consumes it. The consumers are:

1. **Writer context.** Hedge or omit `inferred` items.
2. **Update precedence, enforced in code.** A patch with a weaker source cannot overwrite a stronger one. An upgrade is allowed. Conflicts go to the change log.
3. **Capping.** Drop `inferred` items first.
4. **Probes.** Check that inferred values are not asserted as canon.

### Vocabulary

Use one base enum everywhere, with subject-specific refinements:

| Base | Refinements | Used by |
|------|-------------|---------|
| `observed` | `witnessed`, `stated`, `demonstrated` | Knowledge (`witnessed`), values and goals (`stated`, `demonstrated`) |
| `told` | `overheard` | Knowledge |
| `inferred` | none | Everywhere |

Precedence: `observed` > `told` > `inferred`. Values and goals keep `stated | demonstrated | inferred`, because self-professed versus shown through action is a distinction that matters for them.

### Where to apply it

| Subject / field | Decision | Notes |
|-----------------|----------|-------|
| Knowledge items (section 2) | Yes | Already specified |
| Character values (section 9) | Yes | Already specified |
| Goals (section 5) | Yes | Goals inferred from behavior are often wrong |
| Threads (section 3) | Yes, as `source` | Separates a stated promise from model-inferred foreshadowing, so an inferred thread is not treated as an obligation |
| Relationships on characters and groups | Yes, trial | See evidence below; measure the revisit rate before and after |
| Event `origin` (section 7) | Already covers it | `reported` is the event-level analog of `told` |
| `traits`, `quirks`, `preferences`, `fears` (`list[str]`) | No | Structuring plain string lists costs more than it returns. Use the change log instead |
| Element `properties` and hidden facts | No | Route secret or unknown properties through `knowledge` with `known_by` |
| Biography claims made by characters | Route through `knowledge` with `status` | A character's self-told backstory should not be asserted as true |
| Group `purpose` / `culture` | Open | See open questions |

### Evidence for the relationship decision

Per-path churn from the soak's applied updates. "Revisited" means a later patch's old value equals an earlier one, so the field returned to a previous state.

| Subject | Field | Paths | Updates | Updated 2+ times | Revisited |
|---------|-------|-------|---------|------------------|-----------|
| elements | relationships | 134 | 260 | 51 | 30 |
| elements | description | 76 | 193 | 36 | 19 |
| groups | resources | 42 | 111 | 23 | 18 |
| groups | aliases | 22 | 94 | 20 | 12 |
| characters | relationships | 14 | 34 | 3 | 1 |

Caveats on reading this table:

- **Index-addressed lists inflate it.** `resources`, `aliases` and `description` are addressed by list position (for example `resources.2`), and positions shift when items are added or removed. Their revisit counts overstate real flip-flopping. For those fields, use keyed or deduplicated lists rather than `source`.
- **Element relationships are the cleaner signal.** They are dict-keyed, so paths are stable, and 22% (30 of 134) revisited an earlier value.
- **The churn is in the lowest-value place.** Element relationships are relationships between props and locations. Before adding `source` there, consider whether they need to be tracked at all, or at least capped to important elements. Characters and groups have far fewer relationship updates in this run, so there is little data on them yet.

## Rollout order

1. **Item 1** (`description` split). Smallest change, largest immediate effect on both quality and cost.
2. **Item 4** (change history). Pure engine hook, no schema risk, and it gives you data to judge everything else.
3. **Item 3, deterministic parts** (`end.*` from `current_scene`, stale/due computation).
4. **Item 6, code-derived parts** (`last_seen`).
5. **Item 7 guardrails** (cross-tier dedup, `origin` validation). Deterministic and independent of the other items.
6. **Items 2, 3 (creation/resolution) and 5** behind one batched scene-boundary call.
7. **Item 7 milestone types and item 8 enums**, after milestones are actually being populated.
8. **Item 9 values**, set at creation first. Add revision on `transformation`, `betrayal` and `oath` milestones once milestones populate.
9. **Item 10 precedence rule**, then `source` on goals, threads and character/group relationships, after knowledge and values exist.

## How to measure it

The soak already has needle probes (for example `n01`…). Add probe types for the new fields:

- **Leak probes:** plant a fact with `secret_from: X`, then check that X's replies never use it.
- **Thread probes:** plant a promise or scheduled event, then check it resurfaces when due or stale.
- **Drift probes:** track `description` size and item count per character over 100 turns; both should stay roughly flat after item 1.
- **Provenance probes:** count `past` entries whose message node falls inside the story range, and count events duplicated across tiers by start node. Both should be zero after the dedup guard.
- **Value probes:** plant a dilemma and check that replies follow the character's stated tension. Count value edits per 100 turns; it should stay near zero outside trigger milestones.
- **Source probes:** check that `inferred` items are never asserted as canon in writer output, and that no `inferred` patch overwrites an `observed` value.
- **Cost guard:** per-scene-turn call count and seconds should not rise after each addition. If one does, that addition is wrong-shaped.

## Open questions

- Should `knowledge` be a shared subject or live on each entity? Shared is easier to keep consistent. Per-entity is simpler to retrieve. The doc assumes shared with derived views.
- Should threads and events merge into one subject long-term? Events with `status` and `cadence` cover a lot. Keep them separate until the overlap proves too large.
- How much history should the change log keep per field before compaction? Start at 20 entries and tune.
- Should milestones remain a stored list or become a derived view over event participants? The derived view is single-source and needs no per-character call, but it needs backstory events for pre-story milestones.
- Is `_message_node` stamped when an event is created or when its `start` is recorded? The `origin` validation depends on the answer.
- Should `source` also apply to group `purpose` and `culture`? These are often inferred from few observations, but a per-field source adds a sibling field for scalars. An entry-level marker, or routing group facts through `knowledge`, may be cleaner.
- Should element relationships be tracked at all, or restricted to important elements? The soak shows 260 updates and 22% revisited paths for the lowest-value relationship type.
- Do character and group relationships need `source`? The soak has too few updates (34 on two characters) to measure. Decide once the change log provides data.