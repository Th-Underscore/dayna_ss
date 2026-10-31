# DSS Quality Recommendations

Scope: state quality for LLM roleplay (current goal), kept general enough for other memory-system uses later.
Evidence base: the `scifi_heist` seed-524 soak (100 turns, `compressed` cadence) and the v0.2.0 source. Performance work is tracked separately.

## Summary

| # | Item | Type | Extra LLM cost | Priority |
|---|------|------|----------------|----------|
| 1 | ~~Fix `description` semantics (stable vs. current status)~~ **DONE** `4efa3063` | Schema fix | Negative (fewer tokens) | ✅ |
| 2 | Epistemics (who knows what) | New field/subject | 1 call per scene | High |
| 3 | Threads (ongoing/recurring/open items) | New subject + `StoryEvent` fields | 0–1 call per scene | High |
| 4 | Change history (per-field diffs) | Engine hook | 0 | High, cheap |
| 5 | Goals / intent | New field | Folded into scene-boundary update | Medium |
| 6 | Spatial and physical state | Mostly code-derived | 0 | Medium |
| 7 | Event provenance and milestone types | Schema fields + dedup guard | ~1–3 tokens per item | High (provenance), Medium (types) |
| 8 | Other clarification enums | Small schema fields | ~1–3 tokens per item | Low–Medium |
| 9 | Values for characters (`red_lines` for ideological groups) | New field, set at creation | ~100–200 tokens per entry | Medium |
| 10 | Shared `source` convention and precedence rule | Enum + code guard | ~1 token per item | Medium |
| 11 | Threads as a layer over goals, values and arcs | Positioning + read-time view | 0 | High (with item 3) |
| 12 | World rules as deviations from real life (`general_info.rules`) | Seed pass, keyed rules, scoped patches | 1 call per story, then net negative | High |
| 13 | Skills and aptitudes (discrete techniques vs continuous proficiencies) | `skills` field; keyed `attributes` | ~1–2 tokens per entry plus a short `can_do` | Decide after benchmark baseline |
| 14 | Anchored scales for comparable capabilities | Shared ladders defined once, plus code validation | Net small; defined in the seed pass | Decide with item 13 |

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

See section 11 for how threads relate to goals, values and arcs.

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

## 11. Threads as a layer over goals, values and arcs

### Position

Threads should be primary for **narrative state** (what is pending, overdue or paying off). Motivations (values, goals, fears) stay primary for **character behavior**. They answer different questions. A motivation explains why a character decides something, so it belongs to the character. A thread is something the story owes the reader. It can have several participants and a lifecycle, and it can exist with no character wanting anything, such as a ticking clock or an unsolved mystery.

### One source of truth per concept

The risk of making threads primary is double bookkeeping. Avoid it by giving every concept exactly one home and deriving the rest.

| Concept | Source of truth | Role of threads |
|---------|-----------------|-----------------|
| Values, fears | Character | None |
| Personal goal | `Character.goals` (section 5) | Derived view when `horizon=arc` or `blocked_by` is set |
| Party objective | `Objective`, or a thread of kind `objective` | The thread itself |
| Promise, debt, mystery, foreshadowing, schedule | Thread | Primary |
| Held secret | Knowledge (`secret_from`, section 2) | Derived "exposure risk" view |
| Ongoing event | Event with `status=open` | Derived view |

The `threads` subject stores only what has no other home. A read-time union in code supplies the ranked "open threads, stalest first" list. This keeps the cost at zero extra LLM calls.

### Writer-facing use

- Inject only the top 3–5 threads, ranked by importance and staleness plus anything due. Cap at roughly 150 tokens.
- Keep character motivation in the character sections. Do not mix the two in one block.
- Apply the section 10 precedence rule. An `inferred` thread cannot create an obligation, so a hallucinated thread cannot silently become something the story must resolve.

### Evidence that a layer is missing

- **`Arc.plot_threads`** is a `list[str]` generated at arc level. Arcs were empty at turn 99, so nothing captured open threads between the scene and chapter timescales.
- **`Objective`** (`{description, importance}`) stayed at its 2 seed entries for all 100 turns, yet it still runs a gate, a patch and an importance update on every scene boundary.

### Recommendations

- Derive arc-level `plot_threads` as a roll-up of the thread subject instead of generating it separately.
- Consider folding `Objective` into threads as `kind=objective`, with `main_objective` kept as a pointer to the top open objective. This is optional and can wait until threads exist.

## 12. World rules as deviations from real life

### Evidence

- `general_info.rules` never changed after turn 0. All 3 entries are seed values, and two are genre constraints rather than world rules ("No fantasy elements", "No soft-science hand-waving").
- The story did contain world mechanics, scattered across other subjects: the Vault "requires specialized credentials", and a failed handshake triggers a sector override, then a secondary relay, then patrol redirects. A regex scan of the turn-99 state found 58 rule-like sentences outside `rules`. Most are narrative noise, and a handful are genuine mechanics. A heist story will not produce a few dozen. That volume has to come from world-building.
- `general_info` took 1,624 s (5.2% of DSS time) in the soak, 953 s of it decode.

### Causes

- **Wrong input.** The run's `GeneralInfo` fires both a path patch and a full-JSON rewrite on every scene boundary. The rewrite's only story input is the last scene summary (1–3 sentences), which compresses away world mechanics.
- **Nothing prompts for rules.** The patch call's only nudge is to fill empty fields, and `rules` is not empty.
- **The placeholder pulls the wrong way.** Its examples are fantasy-style ("mana crystals") and it says "nothing character-specific", so a heist story rarely triggers it.
- **Schema drift.** The run's schema fires `on_new_scene: query_branch_for_changes (skip_query) + perform_update`. The repo example fires only `always: query_branch_for_changes (skip_query)`. Confirm which is intended.
- **The seed prompt skips rules.** The repo's `initial_population` prompt mentions only `writing_style`, `tone` and `themes`. The soak state carries `_populated_from_first_scene`, so check which prompt produced the 3 seed rules.

### Shape

Key rules by slug instead of list position. Index-addressed lists flip-flopped in the churn data (section 10), and slugs also give stable change-log paths and natural dedup.

```
rules: dict[slug, Rule]
Rule:
  text:   str    # at most ~20 words; a deviation from real-world expectations
  domain: physics | technology | biology | economy | law_institutions
          | social | travel_comms | environment | knowledge | meta
  scope:  global | <location or faction>
  source: stated | demonstrated | inferred   # section 10
```

Use `domain: meta` for genre and tone constraints. They constrain the narrator and are not world facts.

### Recommendations

1. **Seed pass at creation.** One call walks the domain checklist and asks, for each domain, how this world differs from reality. Cap it at about 3 per domain, which gives 20–30 terse deviations. This is the main lever for "a few dozen" rules.
2. **Frame rules as deviations.** Use the test "would a reader who knows the real world get this wrong without being told?" and drop the fantasy examples from the placeholder. Real life is the baseline, so record only the differences.
3. **Replace the full rewrite with scoped patches** for `synopsis`, `main_objective` and rules. The rewrite regenerates roughly 500 tokens to change a sentence or two, and its input is too thin to find rules anyway. It also risks silently rewording or dropping seed rules.
4. **Harvest from verbatim exchanges.** Use a regex prefilter ("requires", "only if", "triggers", "cannot", "must") to decide whether the harvest call runs, rather than an LLM gate. Run it every N scenes or at chapter boundaries, over the exchanges and not the summaries.
5. **Promote candidates.** Hold newly harvested rules as `inferred` candidates. Promote them on a second sighting or an explicit statement.
6. **Retrieve by scope.** At about 12–20 tokens each, 40 rules cost roughly 500–800 tokens, so include them all until the list grows. Beyond that, always include `meta` rules and retrieve the rest by scope and relevance to the current scene.

## 13. Skills and aptitudes (candidate)

This item comes from the benchmark discussion below. Decide on it after the baseline run, since it may turn out not to be needed. It depends on the shared ladders in section 14.

### The distinction: discrete versus continuous

The useful split is not "quantitative versus qualitative". Both kinds of field need to be quantified at the same resolution, otherwise one of them stays impossible to compare. The real distinction is:

| Field | What it holds | Examples |
|-------|---------------|----------|
| `skills` | Discrete, nameable techniques that are used on demand, with costs and limits | "Sunbeam" pierces obsidian; "Meteor Drive" blasts through mountains |
| `attributes` | Continuous aptitudes with no activation | Raw strength; a strategist's speed or thoroughness |

### Evidence

Learned skills currently live in `attributes: list[str]`. The placeholder describes "abilities, skills, or quantifiable characteristics", and the entries are free prose (for example "Engineering: proficient with improvised repairs…"). There is no tier, no source, no limits and no record of who knows about them. The list is index-addressed, so a skill that levels up means rewriting a string, which has the churn problem shown in section 10.

### Shape (sketch)

```
skills: dict[name, Skill]            # keyed, not index-addressed
Skill:
  can_do:        str      # 1–2 concrete sentences using the ladder's yardsticks (section 14)
  tier:          int      # rung on the magnitude ladder
  limits:        str|null # cost, cooldown, requirements
  acquired_via:  trained | innate | item | granted
  learned_scene: str
  evidence:      event name | null   # the demonstrated feat; null = stated or inferred
  visibility:    public | secret     # links to knowledge (section 2)
  source:        stated | demonstrated | inferred   # section 10

attributes: dict[name, Aptitude]     # migrate from list[str]
Aptitude:
  can_do:   str      # 1–2 concrete sentences; name the axis (speed, thoroughness, breadth)
  tier:     int      # rung on the proficiency ladder
  source:   stated | demonstrated | inferred
```

- **Progression** is recorded through the change log (section 4) and milestone types `feat` and `transformation` (section 7), not a separate progression log.
- **Keep keys stable.** Keying by name gives stable change-log paths and dedup, the same reason as for `rules` in section 12.
- **Migration:** existing `attributes: list[str]` entries can be converted once, in code, by keying on the text before the first colon.

## 14. Anchored scales for comparable capabilities

### Problem

Qualitative capability text cannot be compared across entries. One entry pierces obsidian, another outruns bullets, a third blasts mountains, and nothing orders them. The same problem applies to other fields. In the soak, 1 of 178 element attributes contained a number, and 0 of 179 element properties did. Values are ordinal words such as "minimal", "low" and "High-tier corporate encryption". That is fine for a heist, but it would not survive a power-scaling story.

A prompt that only asks for "a couple of sentences about what it can do" is a good first step, but it will produce sentences measured against different yardsticks. The fix is to anchor everything to a shared, story-defined ladder. The existing universal Importance Scale is the precedent: it already solves comparability for importance.

### Two ladders

| Ladder | Used by | Tier 0 |
|--------|---------|--------|
| **Magnitude** (output, durability, impact) | Skills, element durability, group force, event scale | A real-world human or object |
| **Proficiency** | Aptitudes (`attributes`) | Real-world percentiles: competent professional, top 1%, then rungs above that defined by the story |

Tier 0 and the lowest rungs are anchored to the real world, which matches the "real life is the baseline, rules are the deviations" approach in section 12. Rungs above the human range are story-specific and are defined by the seed pass.

### Rung structure

```
Rung:
  tier:       int
  yardstick:  a coarse reference (a vehicle door, a building wall, a hillside, a mountain)
  exemplars:  up to 2 established feats or abilities from this story
```

- **Define the ladders once** in `rules` (domain `physics` or `technology`) or the system context, the way the Importance Scale is defined. Keep per-field placeholders to a single line. About 8–10 rungs is enough.
- **Make the exemplars living.** The seed pass writes the first ones. Demonstrated feats add more over time ("T4: Sunbeam pierces obsidian"). Anchoring to the story's own exemplars keeps the model from importing its inflated genre priors.
- **Use coarse yardsticks, not invented numbers.** "Pierces a building wall" is safer than "42 cm of concrete". False precision becomes canon in the writer's output and gets contradicted later. If the story states a number, keep it verbatim with its unit.
- **Multi-axis capabilities** (a strategist's speed versus thoroughness) live in the `can_do` text, which names the axis. Add structured axes only for comparison-heavy stories, otherwise cost grows quickly.

### Code validation

These rules are cheap and target the main failure mode in this genre, power creep:

- A tier can increase only on a demonstrated feat (`evidence` set).
- Cap increases at one rung per scene unless a `transformation` milestone is recorded.
- A decrease needs a stated reason (drain, sealing, injury).
- Stated claims ("the strongest in the universe") cannot raise a tier. This is the `source` precedence rule from section 10.
- Coerce unknown tiers to the nearest valid rung. Do not retry.

### Check calibration before committing

- **Agreement test:** ask the model to tier 10 abilities twice. If the results are not within ±1 tier at least ~90% of the time, tighten the exemplars.
- **Consistency metric:** check that contest outcomes in events agree with tier ordering. Flag exceptions (plot armor, narrative reasons) instead of auto-fixing them.
- **Power-creep metric:** compute tier increases per character per 100 turns from the change log (section 4).

### Where else this applies

Apply anchored scaling only where something will compare: a contest, a threshold or an escalation. This is the same rule as for `source` in section 10.

| Field | Apply? | Consumer |
|-------|--------|----------|
| Skills and aptitudes (section 13) | Yes | Ordering and contests |
| Element durability, capacity, security level | Yes, on the magnitude ladder | Threshold checks ("can this break that?") become a tier comparison |
| Group force and reach | Yes, as scale enums (individual, squad, organization, city, nation, planet) | Faction-versus-faction outcomes |
| Event scale of impact | Yes, as a scale enum | Escalation and consistency across fights |
| Injury severity (section 6) | Yes, as a severity enum | Continuity across turns |
| Travel times and distances | Yes, as multiples of a real-world reference | Fast-travel stories |
| Relationship intensity | Low priority | Only if the story tracks it |
| Traits, voice, quirks, preferences, descriptions | No | Descriptive; nothing compares them |

## Rollout order

0. **Record baselines** on both benchmark stories (see "Benchmark stories" below) before implementing the next item.
1. ~~**Item 1** (`description` split)~~ **DONE** — committed `4efa3063`.
2. **Item 4** (change history). Pure engine hook, no schema risk, and it gives you data to judge everything else.
3. **Item 12, seed pass and deviation framing.** One call per story, no per-turn cost, and it addresses the sparse `rules` directly. For power-scaled stories, include the section 14 ladders in the same pass.
4. **Item 3, deterministic parts** (`end.*` from `current_scene`, stale/due computation).
5. **Item 6, code-derived parts** (`last_seen`).
6. **Item 7 guardrails** (cross-tier dedup, `origin` validation). Deterministic and independent of the other items.
7. **Item 12, scoped patches and regex-gated harvest.** Replaces the `general_info` full rewrite.
8. **Items 2, 3 (creation/resolution) and 5** behind one batched scene-boundary call.
9. **Item 7 milestone types and item 8 enums**, after milestones are actually being populated.
10. **Item 9 values**, set at creation first. Add revision on `transformation`, `betrayal` and `oath` milestones once milestones populate.
11. **Item 10 precedence rule**, then `source` on goals, threads, rules and character/group relationships, after knowledge and values exist.
12. **Item 11 roll-up** of arc `plot_threads` from threads, once threads exist and arcs close during a run.
13. **Items 13 and 14** (skills, aptitudes, anchored ladders), only if the benchmark baseline shows capability tracking is a real gap. Start with the prompt-only `can_do` step plus the ladder definitions, then add the keyed fields and code validation.

## How to measure it

The soak already has needle probes (for example `n01`…). Add probe types for the new fields:

- **Leak probes:** plant a fact with `secret_from: X`, then check that X's replies never use it.
- **Thread probes:** plant a promise or scheduled event, then check it resurfaces when due or stale.
- **Drift probes:** track `description` size and item count per character over 100 turns; both should stay roughly flat after item 1.
- **Provenance probes:** count `past` entries whose message node falls inside the story range, and count events duplicated across tiers by start node. Both should be zero after the dedup guard.
- **Value probes:** plant a dilemma and check that replies follow the character's stated tension. Count value edits per 100 turns; it should stay near zero outside trigger milestones.
- **Source probes:** check that `inferred` items are never asserted as canon in writer output, and that no `inferred` patch overwrites an `observed` value.
- **Rule probes:** count deviation rules after the seed pass (target 20–30), and check that no `meta` rule is asserted as a world fact. Plant a rule stated once in-story and check that it is harvested within N scenes.
- **Thread view probes:** check that every open event, objective and arc-horizon goal appears in the derived thread list exactly once.
- **Skill probes:** plant a learned technique and a tier-up, then check that the tier history is recorded, that no silent downgrade occurs, and that a stated claim does not raise a tier.
- **Ladder probes:** run the two-pass agreement test (within ±1 tier at least ~90% of the time), the contest-versus-tier consistency check, and the tier-increase rate per character.
- **Cost guard:** per-scene-turn call count and seconds should not rise after each addition. If one does, that addition is wrong-shaped.

### Benchmark stories

Keep the heist as the performance and regression baseline, since it is already measured. Add a second, deliberately different story to cover what the heist cannot.

**An original shounen or manhwa-style story is a good second benchmark.** It covers things the heist never exercised:

| Coverage gap in the heist run | What the second story exercises |
|-------------------------------|---------------------------------|
| 2 characters. `select_entries_min_size` is 4, so character selection never ran | A large, growing cast: selection, roster compaction, knowledge fan-out |
| Rules stayed at 3 seed entries | A power system with real deviations from baseline (section 12) |
| 1–2 chapters, empty arcs | Sagas: arc and chapter roll-ups (section 11) |
| Few long-running commitments | Tournaments, rivalries, training, prophecies: scheduled and recurring threads (section 3) |
| One character hiding something | Hidden power levels and secret identities (section 2) |
| Skills as prose in `attributes` | Learned techniques, tier-ups and comparable capabilities (sections 13 and 14) |
| Mostly continuous time | Time skips and travel between realms (sections 6 and 8) |

Design notes:

- **Use original characters and an invented power system.** A story built on a famous canon lets the model fill gaps from pretraining, so memory failures pass probes for the wrong reason, and it may contradict the roleplay from canon details.
- **Script planted facts and score recall.** Plant a gold list per item: who knows a secret, a promise with a due scene, a technique learned at a known turn, a rule stated once. Score the state against it. The harness already has an auditor and beat tracking; extend it with one probe family per item.
- **Replay for state quality, run live for writer behavior.** Record the transcript once and replay it through DSS only, so every build sees identical input. Without this, each item changes the story itself, because the writer reads DSS state. Use live runs only for probes that depend on the writer, such as leak probes.
- **Control noise.** Local calls run at temperature 1.0, so run at least 2 seeds for headline metrics or lower the DSS temperature. Compare call counts and token counts across builds. Wall-clock time is confounded by server stalls, such as the 50–142 s TTFT stalls in turn 38 of the soak.
- **Reach arc level.** Set the cadence so at least one arc closes within the run. Otherwise the roll-up in section 11 is untested.
- **Shorten the loop.** A full 100-turn run took about 10 hours. Use shorter targeted replays between items and one full run per milestone.

Expect this genre to stress `status` over `description` (power-ups and injuries are short-lived) and the change log (fast state changes), which also makes it a good check on item 1.

## Open questions

- Should `knowledge` be a shared subject or live on each entity? Shared is easier to keep consistent. Per-entity is simpler to retrieve. The doc assumes shared with derived views.
- Should threads and events merge into one subject long-term? Events with `status` and `cadence` cover a lot. Keep them separate until the overlap proves too large.
- How much history should the change log keep per field before compaction? Start at 20 entries and tune.
- Should milestones remain a stored list or become a derived view over event participants? The derived view is single-source and needs no per-character call, but it needs backstory events for pre-story milestones.
- Is `_message_node` stamped when an event is created or when its `start` is recorded? The `origin` validation depends on the answer.
- Should `source` also apply to group `purpose` and `culture`? These are often inferred from few observations, but a per-field source adds a sibling field for scalars. An entry-level marker, or routing group facts through `knowledge`, may be cleaner.
- Should element relationships be tracked at all, or restricted to important elements? The soak shows 260 updates and 22% revisited paths for the lowest-value relationship type.
- Do character and group relationships need `source`? The soak has too few updates (34 on two characters) to measure. Decide once the change log provides data.
- Which `GeneralInfo` trigger is intended: the run's `on_new_scene` patch plus rewrite, or the repo example's `always` patch only?
- Which prompt produced the 3 seed rules, given that `initial_population` mentions only `writing_style`, `tone` and `themes`?
- Should `Objective` fold into threads, or stay as the structured form of `main_objective`?
- Should rule `scope` be free text or a reference to an existing location or group entry? References are validatable, but they couple rules to the entity subjects.
- Should the ladders live in `rules` (simplest, defined by the seed pass) or in their own small subject? The doc assumes `rules`.
- How should existing `attributes: list[str]` data migrate to a keyed structure? Keying on the text before the first colon works for the sample data, but other stories may not follow that pattern.
- Which stories need the ladders at all? A power-scaled flag set at creation would avoid carrying the ladder cost in a heist or slice-of-life story.
- Is `yardstick` best expressed as a free-text reference or as a small fixed list of reference objects? A fixed list is easier to validate but less flexible.