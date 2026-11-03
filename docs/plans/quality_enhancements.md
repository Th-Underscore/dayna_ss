# DSS Quality Plan

Scope: state quality for LLM roleplay (current goal), general enough for other memory-system uses later.
Evidence: the `scifi_heist` seed-524 soak (100 turns, `compressed` cadence) and the v0.2.0 source. Performance work is tracked separately.
Section numbers differ from the earlier item numbering. A map is at the end.

## 1. Summary

| Section | Item | Extra LLM cost | Priority |
|---------|------|----------------|----------|
| 5.1 | ~~`description` split (stable vs. current status)~~ **DONE** `4efa3063` | Negative (fewer tokens) | ✅ |
| 3.1 | ~~Change history (per-field diffs)~~ **DONE** `78dd784b` | 0 | ✅ |
| 3.2 | Shared `source` convention and precedence rule | ~1 token per item | Medium |
| 4.1 | World rules as deviations from real life | 1 call per scene boundary (rules query only) | High (partial: harvest trigger done, shape migration pending) |
| 4.2 | Epistemics (who knows what) | 1 call per scene | High |
| 5.2 | Values for characters | Set at creation | Medium |
| 5.3 | Goals / intent | Folded into the scene-boundary call | Medium |
| 5.4 | Spatial and physical state | Mostly code-derived | Medium |
| 6 | Skills, aptitudes and anchored scales | Small, plus a short `can_do` per entry | Decide after the benchmark baseline |
| 7.1 | Event provenance and milestone types | ~1–3 tokens per item | High (provenance), Medium (types) |
| 7.2 | Threads | 0–1 call per scene | High |

Relationship asymmetry is already handled. Bidirectionality exists for entity-graph retrieval, not for the relationship data itself, so there is nothing to add.

## 2. Principles

Every addition should respect what the timing data showed, otherwise quality is paid for in minutes per turn.

- **No new per-entry LLM calls.** Entry-level patch calls were ~57% of DSS time. Ride on existing calls or on one batched call per scene boundary (`on_new_scene`).
- **Prefer code-derived over LLM-derived.** If a value can be computed from `current_scene`, event participants or the applied-update log, compute it.
- **No hardcoded subject structures in code.** Deterministic logic must be self-contained and convention-driven (e.g. "any `TimePoint` field named `end` syncs from `current_scene.now.when`"), not per-subject field-name lists. The `scene_archive_template` pattern (Jinja2 template in schema defaults) is the model: the "which fields, from where" knowledge lives in the schema, the code renders generically. A small rule table (name pattern + type → action) is acceptable; a hardcoded list of `StoryEvents.end.when = ...` is not.
- **Keyed, bounded, patch-style fields.** The path/value patch format works, with `NO_UPDATES_REQUIRED` as the cheap exit. Index-addressed lists flip-flop (3.2), so key new collections by name or slug and cap anything log-like.
- **One source of truth per concept.** Derive views instead of generating the same fact twice.
- **Add a field only where something consumes it:** writer context, code validation, capping or a probe. This applies to `source` (3.2) and to scales (6.2).
- **Coerce enums in code and never retry.** Retries were 7.6% of DSS time. Map unknown values to the nearest valid one or a default.
- **Define each taxonomy once, in the system context,** the way the universal Importance Scale is. An enum's real cost is the placeholder text and example JSON it adds to every prompt, which currently sits in the uncached part. Keep per-field placeholders to one line.

## 3. Foundations

### 3.1 Change history

Goal: a git-diff-style record per field, so the system can answer "what changed, and when" without rescanning scenes.

The update-apply path already computes `old_value` for every validated update (the soak log prints `Applying update: <path> == <old value>`). Persist it:

```
_history: list of { turn, scene, path, old, new, reason? }
```

- Store it in a private field (leading underscore) that the traversal skips. There is no per-turn prompt cost and no extra LLM call. `reason` stays empty unless the model supplied one.
- Retrieve on demand (for example via `dss_get_info`), not by default. Filter at read time: skip `importance`-only and time-bump (`end.*`) changes.
- Compact when a field's history exceeds a cap (start at 20), collapsing old entries into one line per scene.
- It makes other items enforceable: a field that changes every scene becomes visible (5.1), and it supplies the tier-increase rate (6.2).

### 3.2 Source and precedence

`source` records how a fact is known. It earns its place only where something consumes it:

1. **Writer context:** hedge or omit `inferred` items.
2. **Precedence, enforced in code:** a weaker source cannot overwrite a stronger one. Upgrades are allowed. Conflicts go to the change log.
3. **Capping:** drop `inferred` items first.
4. **Probes:** check that inferred items are not asserted as canon.

Vocabulary: one base enum with refinements. Precedence is `observed` > `told` > `inferred`.

| Base | Refinements | Used by |
|------|-------------|---------|
| `observed` | `witnessed`, `stated`, `demonstrated` | Knowledge (`witnessed`); values, goals and skills (`stated`, `demonstrated`) |
| `told` | `overheard` | Knowledge |
| `inferred` | none | Everywhere |

Values, goals and skills keep `stated / demonstrated / inferred`, because self-professed versus shown through action matters for them.

Where to apply it:

| Subject / field | Decision | Notes |
|-----------------|----------|-------|
| Knowledge (4.2), values (5.2), goals (5.3), skills and aptitudes (6.1), threads (7.2), rules (4.1) | Yes | An inferred thread cannot create an obligation |
| Relationships on characters and groups | Trial | Measure the revisit rate before and after |
| Event `origin` (7.1) | Already covered | `reported` is the event-level analog of `told` |
| `traits`, `quirks`, `preferences`, `fears` (`list[str]`) | No | Structuring plain string lists costs more than it returns. Use the change log |
| Hidden element properties | Route elsewhere | Use `knowledge` with `known_by` |
| A character's self-told backstory | Route elsewhere | Use `knowledge` with `status`; do not assert it as true |

Evidence for the relationship decision: per-path churn in the soak's applied updates. "Revisited" means a later patch's old value equals an earlier one, so the field returned to a previous state.

| Subject | Field | Paths | Updates | Updated 2+ times | Revisited |
|---------|-------|-------|---------|------------------|-----------|
| elements | relationships | 134 | 260 | 51 | 30 |
| elements | description | 76 | 193 | 36 | 19 |
| groups | resources | 42 | 111 | 23 | 18 |
| groups | aliases | 22 | 94 | 20 | 12 |
| characters | relationships | 14 | 34 | 3 | 1 |

- **Index-addressed lists inflate the table.** `resources`, `aliases` and `description` are addressed by position (`resources.2`), and positions shift. Use keyed or deduplicated lists there, not `source`.
- **Element relationships are the cleaner signal.** They are dict-keyed, and 22% (30 of 134) revisited an earlier value.
- **The churn is in the lowest-value place.** Element relationships link props and locations. Consider restricting them to important elements, or dropping them, before adding `source` there. Characters and groups have too few relationship updates to judge yet.

## 4. World knowledge

### 4.1 Rules as deviations from real life

Evidence:

- `general_info.rules` never changed after turn 0. All 3 entries are seed values, and two are genre constraints rather than world rules ("No fantasy elements", "No soft-science hand-waving").
- The story did contain world mechanics, scattered across other subjects: the Vault "requires specialized credentials", and a failed handshake triggers a sector override, then a secondary relay, then patrol redirects. A regex scan of the turn-99 state found 58 rule-like sentences outside `rules`. Most are narrative noise and a handful are genuine. A heist will not produce a few dozen, so that volume has to come from world-building.
- `general_info` took 1,624 s (5.2% of DSS time), 953 s of it decode.

Causes:

- **Wrong input.** The run's `GeneralInfo` fires a path patch and a full-JSON rewrite on every scene boundary. The rewrite's only story input is the last scene summary (1–3 sentences), which compresses away mechanics. *(Since replaced: the full-rewrite `perform_update` trigger was removed; only `query_branch_for_changes` entries remain.)*
- **Nothing prompts for rules.** The patch call only nudges toward empty fields, and `rules` is not empty.
- **The placeholder pulls the wrong way.** Its examples are fantasy-style ("mana crystals") and it says "nothing character-specific".
- **Seed and schema drift.** The repo's `initial_population` prompt mentions only `writing_style`, `tone` and `themes`, and the run's schema fires `on_new_scene` (patch plus rewrite) where the repo example fires `always` (patch only). See the open questions.

Shape: key rules by slug, not list position.

```
rules: dict[slug, Rule]
Rule:
  text:   str    # at most ~20 words; a deviation from real-world expectations
  domain: physics | technology | biology | economy | law_institutions
          | social | travel_comms | environment | knowledge | meta
  scope:  global | <location or faction>
  source: stated | demonstrated | inferred   # 3.2
```

Use `domain: meta` for genre and tone constraints. They constrain the narrator and are not world facts.

Recommendations:

1. ~~Seed pass at creation.~~ **DONE** `9e93e070`. A dedicated `on_new_scene` trigger with a rules-focused `query_branch_for_changes` (see `rules_query_prompt_template` in `GeneralInfo`) now harvests world rules on every scene boundary. No one-shot seed pass needed — the first scene boundary populates the initial rules, and subsequent boundaries add new ones. Rides on existing trigger + action infrastructure (zero code). **Shape migration done:** `rules` is now `dict[str, WorldRule]` with structured fields (`text`, `domain`, `scope`, `source`).
2. **Frame rules as deviations.** Use the test "would a reader who knows the real world get this wrong without being told?" and drop the fantasy examples. Real life is the baseline.
3. **Replace the full rewrite with scoped patches** for `synopsis`, `main_objective` and rules. The rewrite regenerates ~500 tokens to change a sentence or two, and it risks silently rewording or dropping seed rules.
4. **Harvest from verbatim exchanges, not summaries.** A regex prefilter ("requires", "only if", "triggers", "cannot", "must") decides whether the harvest call runs, instead of an LLM gate. Run it every N scenes or at chapter boundaries.
5. **Promote candidates.** Hold harvested rules as `inferred`. Promote on a second sighting or an explicit statement.
6. **Retrieve by scope.** At ~12–20 tokens each, 40 rules cost ~500–800 tokens, so include them all until the list grows. After that, always include `meta` rules and retrieve the rest by scope and scene relevance.

### 4.2 Epistemics

Characters acting on things they shouldn't know are the most visible roleplay memory bugs, and nothing in the schema models knowledge.

```
Knowledge item:                      # one shared `knowledge` subject
  fact:          str                 # one atomic statement
  known_by:      list[str]           # characters and groups
  secret_from:   list[str]           # explicit "must not know"
  status:        true | false | suspected | rumor
  source:        witnessed | told | overheard | inferred   # 3.2
  learned_scene: str
```

- **One source of truth.** Per-character and per-group views are derived, not separately generated, so `known_by` stays consistent by construction.
- **Populate with one call at the scene boundary** over the scene recap: "Who learned or concealed what in this scene? Return only new or changed items." Code fans the result out to the views. That is 1 call per scene, not N entry calls.
- **Retrieval:** include a character's view only when they are in the current scene or about to speak. Always include `secret_from` constraints for characters in the scene, since those prevent leaks.
- **Memories (deferred):** a per-character "Memories" layer (what they *remember experiencing*, the prose/experiential view) is not an independent generation target. It falls out as a derived prompt-build projection from knowledge items (this section) + milestone participations (7.1) + `_history` entries (3.1): path-scope under the entity → collapse per-path runs → rank by recency × inverse path-frequency → render compact lines. No extra LLM call. The structured data is the source of truth; prose is a view.

## 5. Characters

### 5.1 `description` split (done)

Evidence: 517 of 576 applied character updates (90%) targeted `description.*`. `Dex.description` reached 71 items (10.9k chars) by turn 99 and `Sable.description` 20 items (4.8k chars). The items were paraphrased situation snapshots appended each scene, not exact duplicates, so hygiene did not catch them. Character calls were the decode hogs: 101 patch calls, ~31 s decode each, 6% no-ops.

Cause: the placeholder asked for "current state, appearance, and how they present themselves", which mixes stable appearance, current status and a history log.

Fix (implemented): `description` holds stable appearance and demeanor, rewrite-only with a soft cap. A separate `status` holds the current situation as a single overwritten string. History lives in events and the change log (3.1).

Follow-up: the same log pattern appears in the `events` lists on elements (`modified keypad`: 49 items, 7k chars) and groups (`Security signatures`: 31 items). Give those a hard cap and let hygiene drop or merge the oldest.

### 5.2 Values

Characters have `traits`, `attributes`, `voice`, `quirks`, `fears` and `preferences`, but nothing that explains why they act. A value with a built-in tension ("the longer you survive, the longer you can help" versus heroic instinct) tells the writer how the character resolves a dilemma.

```
Value:                               # 2–4 per character
  maxim:           str               # in their own words
  drives:          str               # the behavior it produces
  in_tension_with: str | null        # another value, fear or instinct it fights
  source:          stated | demonstrated | inferred   # 3.2
  coping:          str | null        # optional: how they justify breaking it
```

- **Populate at creation** in the existing `add_new` call.
- **Revise rarely,** overwrite-only with a cap, when a `transformation`, `betrayal` or `oath` milestone is created (7.1). Otherwise it becomes another `description`-style log. The change log then shows value drift, which is character development.
- **`coping` is optional.** It feeds inner monologue and dissonance and is the easiest part to over-generate.
- **Writer context:** one line, only for characters in the current scene.
- **Cost:** ~100–200 tokens per entry, small next to a ~3k-token character entry.

How it fits with nearby fields:

| Field | Lifetime | Role |
|-------|----------|------|
| Values | Stable | Why they act |
| Goals (5.3) | Short-term | What they want now |
| `fears` | Stable | Negative driver |
| Knowledge and secrets (4.2) | State | What they know and hide |
| `voice`, `quirks` | Stable | How they sound; a literal catchphrase belongs here |

Groups: skip a motto field. `purpose` and `culture` already cover it, and in the sample the three corporate groups have near-identical text. For factions with real ideology, add `red_lines` (what they will always or never do), which predicts reactions better than a motto.

### 5.3 Goals

Short-term intent separates a character that reacts from one that acts.

```
Character.goals:                     # 1–3 per character, overwritten
  - want:       str
    hiding:     str | null           # links to knowledge.secret_from
    blocked_by: str | null
    horizon:    scene | arc
    source:     stated | demonstrated | inferred   # 3.2; inferred goals are often wrong
```

Fold the update into the scene-boundary call, not per-entry patches. `hiding` and `secret_from` should reference the same facts.

### 5.4 Spatial and physical state

`current_scene` tracks where characters are now. The gap is where they are when not in the scene, and the system must not hallucinate movement.

- **`last_seen` in code,** updated from `current_scene.now.who` each turn: `{scene, location, turn}`. Zero LLM calls.
- **Surface "last seen at X, N scenes ago," not "is at X."** Optionally downgrade confidence as scenes pass. Only an explicit statement ("Dex left for the docks") sets a new location, and that comes from the scene-boundary call.
- **Physical state:** a small struct on characters and groups (`held_items`, `injuries`, `attire`), overwritten like `status`. Injuries use the severity enum from 6.2. Pick one source of truth for who holds what, either `held_by` on elements or `held_items` on characters.
- **Small enums:** element `condition` (intact, damaged, lost, consumed). Character `lifecycle` (active, absent, deceased, unknown) only if not already tracked; I did not check.

## 6. Capabilities

### 6.1 Skills and aptitudes (candidate)

Decide after the benchmark baseline, since it may not be needed. It depends on the ladders in 6.2.

The useful split is not "quantitative versus qualitative". Both kinds need quantifying at the same resolution, otherwise one stays impossible to compare. The real split is discrete versus continuous:

| Field | What it holds | Examples |
|-------|---------------|----------|
| `skills` | Discrete, named techniques used on demand, with costs and limits | "Sunbeam" pierces obsidian; "Meteor Drive" blasts through mountains |
| `attributes` | Continuous aptitudes with no activation | Raw strength; a strategist's speed or thoroughness |

Evidence: skills currently live in `attributes: list[str]` as free prose ("Engineering: proficient with improvised repairs…"), with no tier, source, limits or record of who knows about them. The list is index-addressed, so a level-up means rewriting a string, which has the churn problem in 3.2.

```
skills: dict[name, Skill]
Skill:
  can_do:        str       # 1–2 concrete sentences using the ladder's yardsticks (6.2)
  tier:          int       # rung on the magnitude ladder
  limits:        str|null  # cost, cooldown, requirements
  acquired_via:  trained | innate | item | granted
  learned_scene: str
  evidence:      event name | null   # the demonstrated feat; null = stated or inferred
  visibility:    public | secret     # links to knowledge (4.2)
  source:        stated | demonstrated | inferred

attributes: dict[name, Aptitude]       # migrate from list[str]
Aptitude:
  can_do:  str      # 1–2 concrete sentences; name the axis (speed, thoroughness, breadth)
  tier:    int      # rung on the proficiency ladder
  source:  stated | demonstrated | inferred
```

- **Progression** goes through the change log (3.1) and milestone types `feat` and `transformation` (7.1), not a separate log.
- **Migration:** convert existing `attributes` once, in code, keying on the text before the first colon.

### 6.2 Anchored scales

Problem: capability text cannot be compared across entries. One entry pierces obsidian, another outruns bullets, a third blasts mountains, and nothing orders them. In the soak, 1 of 178 element attributes contained a number and 0 of 179 element properties did. Values are ordinal words ("minimal", "low", "High-tier corporate encryption"). That is fine for a heist but would not survive a power-scaling story.

A prompt that asks for "a couple of sentences about what it can do" is a good first step, but it yields sentences measured against different yardsticks. Anchor everything to a shared, story-defined ladder. The universal Importance Scale is the precedent: it already solves comparability for importance.

| Ladder | Used by | Tier 0 |
|--------|---------|--------|
| **Magnitude** (output, durability, impact) | Skills, element durability, group force, event scale | A real-world human or object |
| **Proficiency** | Aptitudes | Real-world percentiles: competent professional, top 1%, then rungs above that defined by the story |

The low rungs are anchored to the real world, matching 4.1's "real life is the baseline." Rungs above the human range are story-specific and come from the rules harvest (first scene boundary populates, subsequent boundaries add).

```
Rung:
  tier:       int
  yardstick:  a coarse reference (a vehicle door, a building wall, a hillside, a mountain)
  exemplars:  up to 2 established feats or abilities from this story
```

- **Define the ladders once** in `rules` (domain `physics` or `technology`) or the system context. About 8–10 rungs is enough.
- **Keep the exemplars living.** The rules harvest writes the first ones at the first scene boundary, and demonstrated feats add more ("T4: Sunbeam pierces obsidian"). Anchoring to the story's own exemplars keeps the model from importing inflated genre priors.
- **Use coarse yardsticks, not invented numbers.** "Pierces a building wall" is safer than "42 cm of concrete", because false precision becomes canon in the writer's output and gets contradicted later. If the story states a number, keep it verbatim with its unit.
- **Multi-axis capabilities** (speed versus thoroughness) live in the `can_do` text, which names the axis. Add structured axes only for comparison-heavy stories.

Code validation (cheap, and aimed at power creep, the main failure mode in this genre):

- A tier can increase only on a demonstrated feat (`evidence` set).
- Cap increases at one rung per scene unless a `transformation` milestone is recorded.
- A decrease needs a stated reason (drain, sealing, injury).
- Stated claims ("the strongest in the universe") cannot raise a tier (3.2).
- Coerce unknown tiers to the nearest valid rung. Do not retry.

Check calibration before committing: ask the model to tier 10 abilities twice, and tighten the exemplars if the results are not within ±1 tier at least ~90% of the time. Also check that contest outcomes in events agree with tier ordering, flagging exceptions instead of auto-fixing them.

Where else this applies. Use it only where something will compare: a contest, a threshold or an escalation.

| Field | Apply? | Consumer |
|-------|--------|----------|
| Skills and aptitudes (6.1) | Yes | Ordering and contests |
| Element durability, capacity, security level | Yes, magnitude ladder | Threshold checks ("can this break that?") become a tier comparison |
| Group force and reach | Yes, scale enum (individual, squad, organization, city, nation, planet) | Faction-versus-faction outcomes |
| Event scale of impact | Yes, scale enum | Escalation and consistency across fights |
| Injury severity (5.4) | Yes, severity enum | Continuity across turns |
| Travel times and distances | Yes, as multiples of a real-world reference | Fast-travel stories |
| Relationship intensity | Low priority | Only if the story tracks it |
| Traits, voice, quirks, preferences, descriptions | No | Descriptive; nothing compares them |

## 7. Story structure

### 7.1 Event provenance and milestones

Evidence:

- **Milestones are never populated.** Both characters had zero at every one of the 100 turns, and the log shows no milestone-related validation failures. A type enum alone does nothing until something prompts creation.
- **Milestones cannot link to events.** `CharacterMilestone` is `{title, scenes: list[int], importance}`, while events are keyed by name. As far as I can see, nothing resolves a scene number to an event.
- **The container is a weak backstory classifier.** The prompt says `past` is BACKSTORY ONLY, yet all 28 `past` entries carry an in-story message node (for example "The Severed Security Cable"). The same occurrence is also stored across tiers: 14 of 17 `past` entries share a start node with a scene or beat, and 20 of 35 beats share one with a scene.

Put each attribute at the level where it is true by definition, and derive the rest:

| Attribute | Lives on | Why |
|-----------|----------|-----|
| `origin` (backstory, on_page, off_screen, reported) | The event | A property of the occurrence. `off_screen` and `reported` also feed epistemics |
| Milestone `type` | The event participant (`EventParticipant`) | One event can be a feat for one character, a loss for another and a revelation for a third |
| `character.milestones` | Derived view | Built in code from participants above an importance threshold |

The event's `add_new` call already emits `participants`, so a type per participant adds a few output tokens and no new call, and fixes the empty-milestones problem without a per-character call. If you prefer a stored list, give `CharacterMilestone` an `event` key and a `type` and drop `scenes: list[int]`. Characters with pre-story milestones can point at a `past` event.

Milestone types: 6–8 mutually exclusive values with a one-line decision rule each, for example `feat` (achievement through action), `setback` (loss or failure), `revelation`, `bond` / `betrayal`, `transformation` (change in identity, ability, allegiance or status) and `oath` (links to threads, 7.2). Derive revelation milestones from `knowledge` items whose `source` is learned, instead of generating the same fact twice.

Event enums:

- `outcome_status` (success, partial, failure, ongoing, unresolved) next to the free-text `outcome`. It lets code drive thread `status` and replaces part of the `outcome` rewrites, which were 24% of events updates.
- `seq`, an integer assigned in code from message-node order. Dates were `Unknown` or `N/A` in the sampled entries, so ordering is unreliable without it.

Guardrails:

- **Cross-tier dedup in code.** Reject a `past` add whose start node matches an existing scene or beat. This fixes the misclassification and saves `add_new` calls (events `add_new` was 2,673 s in the soak).
- **Validate `origin` against the node.** If a `past` event's message node falls inside the story's message range, flag or re-tag it `on_page`. First check whether `_message_node` is stamped at creation or at event start, because stamping at creation cannot distinguish a backstory event referenced at that message.

### 7.2 Threads

Threads cover promises, debts, mysteries, foreshadowing, schedules and recurring events.

**Position.** Threads are primary for narrative state (what is pending, overdue or paying off). Motivations (values, goals, fears) stay primary for character behavior. A motivation explains why a character decides something, so it belongs to the character. A thread is something the story owes the reader. It can have several participants and a lifecycle, and it can exist with no character wanting anything, such as a ticking clock.

**One home per concept.** The risk of making threads primary is double bookkeeping, so each concept has one source of truth and the rest are derived views:

| Concept | Source of truth | Role of threads |
|---------|-----------------|-----------------|
| Values, fears | Character | None |
| Personal goal | `Character.goals` (5.3) | Derived view when `horizon=arc` or `blocked_by` is set |
| Party objective | `Objective`, or a thread of kind `objective` | The thread itself |
| Promise, debt, mystery, foreshadowing, schedule | Thread | Primary |
| Held secret | Knowledge (`secret_from`, 4.2) | Derived "exposure risk" view |
| Ongoing event | Event with `status=open` | Derived view |

Some of this already exists implicitly. In the soak, 44% of applied event updates (310 of 700) were `end.*` time bumps and 24% (165) were `outcome` rewrites on events still in progress. Make it explicit.

Add to `StoryEvent`: `status` (open, resolved, dormant) and an optional `cadence` ("nightly patrol", "every third scene"). Add a thin `threads` subject only for what has no other home:

```
Thread:
  kind:               promise | debt | mystery | foreshadowing | schedule | recurring | objective
  owner:              str
  summary:            str
  due:                str | null
  cadence:            str | null
  status:             open | resolved | dormant
  source:             stated | demonstrated | inferred   # 3.2
  last_touched_scene: str
```

A read-time union in code supplies the ranked "open threads, stalest first" list, at zero extra LLM cost.

Make most of it deterministic:

- While an event is `open` and overlaps the current scene, set its `end.*` from `current_scene.now.when` in code. This removes close to half of the events patch output. Implement via the convention rule (section 2): any `TimePoint` field named `end` syncs from `current_scene.now.when` — no per-subject hardcoding. The `_message_node` stamp is the existing precedent for this pattern.
- Compute "stale" and "due" in code, and surface stale threads as reminders.
- Only creation and resolution need an LLM. They share the scene-boundary call from 4.2 ("new or resolved threads").
- An `inferred` thread cannot create an obligation (3.2), so a hallucinated thread cannot silently become something the story must resolve.

Writer-facing use: inject only the top 3–5 threads, ranked by importance and staleness plus anything due, capped at ~150 tokens. Keep character motivation in the character sections, not in the same block.

Arcs and objectives. This layer is missing today:

- `Arc.plot_threads` is a `list[str]` generated at arc level, and arcs were empty at turn 99, so nothing captured open threads between the scene and chapter timescales. Derive it as a roll-up of the thread subject instead of generating it separately.
- `Objective` (`{description, importance}`) stayed at its 2 seed entries for all 100 turns, yet it still runs a gate, a patch and an importance update on every scene boundary. Consider folding it into threads as `kind=objective`, with `main_objective` kept as a pointer to the top open objective. This can wait until threads exist.

## 7.5 Speculative: hierarchical format templates

The current `format_templates.json` renders all subjects as flat `Field --- value` lines with inline `<<<<<<<<<<<< path` annotations. A hierarchical Markdown format (`## Subject` → `### Sub-section` → bullets) would mirror the structure LLMs see in training data and may improve field targeting for deeply nested entries (e.g., character → relationships → person → status).

**Unproven.** No observed parsing failure motivates the change; the flat format produces correct path-patches today. The tradeoff:

- **Pro:** Hierarchical Markdown is the dominant structure in LLM training corpora. Sub-sections give the model a scanning skeleton for nested data.
- **Con:** Token overhead. Every `###` header, blank line, and bullet marker is a prefill token. At 30K context, 3–5% overhead is 900–1500 tokens per call — working against the entity-capping and description-split savings.
- **Con:** The `<<<<<<<<<<<<` path annotation contract is line-oriented. A hierarchical format requires deciding where annotations attach (header? each bullet?) and updating the parse side accordingly.

**If pursued:** scope to one subject (e.g., `general_info`) as an A/B smoke, measure patch correctness and field targeting against the flat baseline. Roll out only if quality measurably improves enough to offset the token cost.

## 8. Rollout order

0. **Record baselines** on both benchmark stories (section 9) before implementing the next item.
1. ~~**5.1** `description` split~~ **DONE** `4efa3063`.
2. ~~**3.1** Change history~~ **DONE** `78dd784b`. Per-field `{path, old, new}` on a private `_history` list, capped at 20 per path. Turn/scene/reason deferred to v2.
3. **4.1** Rules harvest. **PARTIAL** `9e93e070` (harvest trigger) + shape migration (`dict[str, WorldRule]`). Remaining: deviation framing in prompt, scoped patches, regex-gated harvest (step 7).
4. **7.2** Deterministic parts (`end.*` from `current_scene`, stale and due computation).
5. **5.4** Code-derived `last_seen`.
6. **7.1** Guardrails (cross-tier dedup, `origin` validation). Deterministic and independent of the rest.
7. **4.1** Scoped patches and regex-gated harvest, replacing the `general_info` rewrite.
8. **4.2, 7.2 (creation and resolution) and 5.3** behind one batched scene-boundary call.
9. **7.1** Milestone types and enums, once milestones are actually being populated.
10. **5.2** Values. Set at creation first, then add revision on `transformation`, `betrayal` and `oath` milestones once milestones populate.
11. **3.2** Precedence rule, then `source` on goals, threads, rules and character and group relationships, after knowledge and values exist.
12. **7.2** Roll-up of arc `plot_threads` from threads, once threads exist and arcs close during a run.
13. **6** Skills, aptitudes and ladders, only if the baseline shows capability tracking is a real gap. Start with the prompt-only `can_do` step plus the ladder definitions, then add keyed fields and code validation.

## 9. Measurement

The soak already has needle probes (for example `n01`…). Add one probe family per item:

| Probe | Section | Pass criterion |
|-------|---------|----------------|
| Drift | 5.1 | `description` size and item count per character stay roughly flat over 100 turns |
| Leak | 4.2 | A planted fact with `secret_from: X` never appears in X's replies |
| Thread | 7.2 | A planted promise or schedule resurfaces when due or stale. Every open event, objective and arc-horizon goal appears in the derived list exactly once |
| Provenance | 7.1 | Zero `past` entries with in-story nodes. Zero events duplicated across tiers |
| Value | 5.2 | Replies follow the stated tension in a planted dilemma. Value edits stay near zero per 100 turns outside trigger milestones |
| Source | 3.2 | No `inferred` item is asserted as canon. No `inferred` patch overwrites an `observed` value |
| Rule | 4.1 | 20–30 deviation rules after the rules harvest populates them. No `meta` rule asserted as a world fact. A rule stated once in-story is harvested within N scenes |
| Skill | 6.1 | A planted technique and tier-up are recorded in history, with no silent downgrade. A stated claim does not raise a tier |
| Ladder | 6.2 | Two-pass agreement within ±1 tier at least ~90% of the time. Contest outcomes consistent with tier ordering. Tier-increase rate per character |
| Cost guard | all | Per-scene-turn calls and tokens do not rise after an addition. If they do, it is wrong-shaped |

### Benchmark stories

Keep the heist as the performance and regression baseline, since it is already measured. Add a second, deliberately different story for what the heist cannot cover. An original shounen or manhwa-style story is a good fit:

| Coverage gap in the heist run | What the second story exercises |
|-------------------------------|---------------------------------|
| 2 characters; `select_entries_min_size` is 4, so character selection never ran | A large, growing cast: selection, roster compaction, knowledge fan-out |
| Rules stayed at 3 seed entries | A power system with real deviations from baseline (4.1) |
| 1–2 chapters, empty arcs | Sagas: arc and chapter roll-ups (7.2) |
| Few long-running commitments | Tournaments, rivalries, training, prophecies: scheduled and recurring threads |
| One character hiding something | Hidden power levels and secret identities (4.2) |
| Skills as prose in `attributes` | Learned techniques, tier-ups and comparable capabilities (6) |
| Mostly continuous time | Time skips and travel between realms (5.4) |

Design notes:

- **Use original characters and an invented power system.** A famous canon lets the model fill gaps from pretraining, so memory failures pass probes for the wrong reason.
- **Script planted facts and score recall.** Keep a gold list per item: who knows a secret, a promise with a due scene, a technique learned at a known turn, a rule stated once. The harness already has an auditor and beat tracking to extend.
- **Replay for state quality, run live for writer behavior.** Record the transcript once and replay it through DSS only, so every build sees identical input. Otherwise each item changes the story itself, since the writer reads DSS state. Use live runs only for probes that depend on the writer, such as leaks.
- **Control noise.** Local calls run at temperature 1.0, so use at least 2 seeds for headline metrics or lower the DSS temperature. Compare call and token counts across builds. Wall-clock time is confounded by server stalls (turn 38 of the soak had 50–142 s TTFT stalls).
- **Reach arc level.** Set the cadence so at least one arc closes within the run, otherwise the 7.2 roll-up is untested.
- **Shorten the loop.** A full 100-turn run took about 10 hours. Use targeted replays between items and one full run per milestone.

Expect this genre to stress `status` over `description` (power-ups and injuries are short-lived) and the change log, which also makes it a good check on 5.1.

## 10. Open questions

Data model:

- Should `knowledge` be a shared subject or live on each entity? The plan assumes shared with derived views.
- Should milestones stay a stored list or become a derived view over event participants? The derived view needs no per-character call but needs backstory events for pre-story milestones.
- Should threads and events merge long-term? Keep them separate until the overlap proves too large. Should `Objective` fold into threads or stay as the structured form of `main_objective`?
- Should rule `scope` be free text or a reference to a location or group entry? References are validatable but couple rules to the entity subjects.
- Should the ladders live in `rules` (assumed) or their own small subject? Is `yardstick` free text or a fixed list of reference objects (easier to validate, less flexible)? Should a power-scaled flag set at creation avoid carrying the ladders in a heist or slice-of-life story?
- How should existing `attributes: list[str]` data migrate to a keyed structure? Keying on the text before the first colon works for the sample data, but other stories may not follow that pattern.
- How much change history per field before compaction? Start at 20 entries and tune.

To verify in code:

- Is `_message_node` stamped when an event is created or when its `start` is recorded? The `origin` validation depends on it.
- Which `GeneralInfo` trigger is intended: the run's `on_new_scene` patch plus rewrite, or the repo example's `always` patch only?
- Which prompt produced the 3 seed rules, given that `initial_population` mentions only `writing_style`, `tone` and `themes`?

Scope decisions that need data first:

- Should `source` apply to group `purpose` and `culture`? These are often inferred from few observations, but a per-field source adds a sibling field for scalars. An entry-level marker, or routing group facts through `knowledge`, may be cleaner.
- Do character and group relationships need `source`? The soak has too few updates (34 on two characters) to measure. Decide once the change log provides data.
- Should element relationships be tracked at all, or restricted to important elements? The soak shows 260 updates and 22% revisited paths for the lowest-value relationship type.

## Map from the earlier numbering

| Earlier | Now |
|---------|-----|
| 1 `description` | 5.1 |
| 2 Epistemics | 4.2 |
| 3 and 11 Threads | 7.2 |
| 4 Change history | 3.1 |
| 5 Goals | 5.3 |
| 6 Spatial and physical | 5.4 |
| 7 Provenance and milestones | 7.1 |
| 8 Clarification enums | Split: event enums in 7.1, `condition` and `lifecycle` in 5.4, `source` in 3.2, cost note in section 2 |
| 9 Values | 5.2 |
| 10 `source` | 3.2 |
| 12 Rules | 4.1 |
| 13 Skills and aptitudes | 6.1 |
| 14 Anchored scales | 6.2 |