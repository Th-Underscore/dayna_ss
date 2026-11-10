# DSS Quality Plan

Scope: state quality for LLM roleplay (current goal), general enough for other memory-system uses later.
Evidence: the `scifi_heist` seed-524 soak (100 turns, `compressed` cadence) and the v0.2.0 source. Performance work is tracked separately (`soak_optimization_plan.md`). Related plans: read-side ranking in `retrieval_plan.md`, the engine and UI boundary in `headless_engine_ui_plan.md`, and how DSS compares with Smart-Memory, TunnelVision and Talemate in `memory_comparisons.md`.
Section numbers differ from the earlier item numbering. A map is at the end.

## 1. Summary

| Section | Item | Extra LLM cost | Priority |
|---------|------|----------------|----------|
| 5.1 | ~~`description` split (stable vs. current status)~~ **DONE** `4efa3063` | Negative (fewer tokens) | ✅ |
| 3.1 | ~~Change history (per-field diffs)~~ **DONE** `78dd784b` | 0 | ✅ |
| 3.2 | Shared `source` convention and precedence rule (with a `user` rung) | ~1 token per item | Medium |
| 3.3 | Entity resolution shared by write and read paths | 0 | High |
| 3.4 | Rewind safety for swipes, edits and deletes | 0 | High |
| 4.1 | ~~World rules as deviations from real life~~ **DONE** | 1 call per scene boundary (rules query only) | ✅ |
| 4.2 | ~~Epistemics (who knows what)~~ **DONE** — `Knowledge` subject + `KnowledgeItem` dataclass + `on_new_scene` trigger + format template; writer-side rendering fixed, scene-cast stances proposed in 4.2 Consumption | 1 call per scene | ✅ |
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

Follow-up (not yet implemented): add `node` (the message node of the turn that produced the patch) and `actor` (`llm` | `user` | `system`) to each entry. Both are free at write time. `node` is what makes rewind possible (3.4) and `actor` is what makes user-edit precedence enforceable (3.2). A read of state as of an earlier turn is already possible in principle by loading that turn's snapshot directory (3.4 Evidence); `node` adds a turn-to-directory index and a cheaper replay of `_history`. Graphiti's bi-temporal edges are the reference (`memory_comparisons.md`, criterion 17).

### 3.2 Source and precedence

`source` records how a fact is known. It earns its place only where something consumes it:

1. **Writer context:** hedge or omit `inferred` items.
2. **Precedence, enforced in code:** a weaker source cannot overwrite a stronger one. Upgrades are allowed. Conflicts go to the change log.
3. **Capping:** drop `inferred` items first.
4. **Probes:** check that inferred items are not asserted as canon.

Vocabulary: one base enum with refinements. Precedence is `user` > `observed` > `told` > `inferred`.

`user` is never emitted by the model. Code stamps it on any edit made through the UI, or through an LLM-assisted edit the user approved (`headless_engine_ui_plan.md` P5). A path stamped `user` is locked against LLM patches until the user unlocks it, and the attempted overwrite goes to the change log. It costs no tokens and it is what makes direct editing safe. Precedent: TunnelVision never lets background automation rewrite authored (`constant`) lorebook entries (`memory_comparisons.md`, criterion 7).

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

### 3.3 Entity resolution (write and read paths)

Goal: one definition of "which entity does this string refer to", used wherever the engine turns text into a key.

Evidence (static read of `a3c64b4`):

- **The write side is already good.** `agents/data_summarizer/parsing.py` has `_normalize_entry_name` (alphanumerics only), `_resolve_dict_key` (exact, then normalized, then alias match) and `_retarget_dict_key` (re-addresses a lookalike key such as `The_Second_Runner` onto the canonical `Second Runner`). `_collect_entry_aliases` walks any nested `aliases` array, schema-agnostic.
- **The read side is separate and weaker.** `rag/structured_rag/context_retriever.py` does its own matching:
  - `_create_character_patterns` and `_create_element_patterns` compile `(full|first|last)` case-insensitive regexes with no word boundary and no stopword filter. A multi-word name that starts with an article ("The Vault") adds `The` as a standalone alternative, which matches almost any sentence. A short name matches inside other words (`Ray` in `array`, `Kit` in `kitchen`).
  - `_get_relevant_elements` and `_get_relevant_groups` call `re.search(name, context)` with the name **unescaped**. A name containing `(`, `+`, `.` or `?` is read as a pattern, so it can raise `re.error` or match the wrong text. `_safe_subject` catches the error, prints a traceback and substitutes an empty subject, so the whole elements or groups section silently disappears from that turn's context.
  - Aliases are consumed on the read side only for groups. Characters and elements have no top-level `aliases` field, so a nickname or title never resolves to them.
- **Why it matters.** Matches become `initial_entities`, which seed the graph traversal. A false positive pulls unrelated state in through decay and inflates context against the 40k standing rule (`soak_optimization_plan.md` §35). A false negative loses the seed.
- The "Entity references" problem in section 10 (the model paraphrasing a slug) is the same problem on the write side.

Design:

- **One resolver module** used by the retriever, `data_summarizer` and the section 10 validator. Reuse the `parsing.py` functions; do not write a second implementation.
- **Index per subject:** normalized name to key, normalized alias to key, partial token to a list of keys. Build it on load and update it when an entry changes. Per section 2, no per-subject field-name lists: aliases come from any `aliases` list `_collect_entry_aliases` finds, and a schema may also declare an `aliases` field on characters and elements.
- **Matching rules:** `re.escape` everywhere. Word boundaries (`(?<!\w)` and `(?!\w)`). Strip a leading article before deriving partial tokens. Ignore partial tokens under 3 characters or in a small stopword list. A partial token that maps to more than one entity is ambiguous and does not seed on its own.
- **Match tiers set the seed weight** in the graph traversal: full name 1.0, alias 0.9, unique partial 0.5. These are starting values, to be tuned with the Retrieval probe in section 9.
- **Aliases cost almost nothing to create:** ask for them in the character and element `add_new` output (a few tokens). Keep them as a deduplicated, append-only, capped set (start at 8), not an index-addressed list, to avoid the churn shown in the 3.2 table.

Additions from reading Graphiti and OpenVault (`memory_comparisons.md`, criterion 3):

- **Stages, cheapest first.** Exact normalized name, then alias, then a fuzzy match, and only then (optionally) an LLM. Graphiti's deterministic stage compares character-shingle sets (Jaccard at least 0.9) and has an entropy guard: short or low-information names are not fuzzy-matched, because they collide, and move on to the next stage. Use the same guard here.
- **Merges leave redirects.** When two entries are merged or a key is retargeted, keep an old-key to canonical-key map (OpenVault's `_mergeRedirects`, followed through chains with a cycle guard), so a later paraphrase of the old name resolves immediately. This persists what `_retarget_dict_key` decides per call today.
- **Type-aware thresholds.** Per its data-schema doc, OpenVault merges people on high embedding similarity but requires token overlap for other entity types. Characters and elements here likely need different rules.
- **Identity fields.** Graphiti lets a type declare identity properties that match before any fuzzy step. A schema could declare `identity: [...]` per subject (default: the name), which keeps the resolver schema-driven.

Cost: 0 LLM calls. Retriever-side details are in `retrieval_plan.md` R2. Verify with the Resolution and Retrieval probes (9).

### 3.4 Rewind safety

Problem: swipes, regenerations, edits and deletes happen every session. DSS applies patches after each turn. Regenerating or swiping the latest reply is already safe, but editing or deleting an earlier message loses state (see Evidence).

Evidence (read in the code at `a3c64b4`, not run): state is stored in one directory per turn, keyed by a hash of the full history (`retrieve_history_path` in `context_engine.py`). Each turn loads the previous turn's directory and writes a new one, so these are content-addressed snapshots, and the message index lives inside each one. A swipe or regeneration restarts from the same pre-exchange directory, so the latest turn rewinds correctly by construction. Discarded branches leave orphaned directories behind. An edit or deletion earlier in the chat changes every later hash, no matching directory exists, and `prepare_context` creates an empty directory holding only the schema files, so state appears to be lost. `backtrack_history` in `core.py` is the hook meant for this, and it is currently a placeholder that does nothing. For comparison, TunnelVision snapshots each write and `revertInvalidSnapshots` undoes any whose message was swiped, edited or deleted, keyed on message identity so it needs no host event arguments. Smart-Memory subscribes to swipe and delete events.

First step (small): implement `backtrack_history` by nearest-prefix snapshot.

- Hash progressively shorter prefixes of the history (the same `hash_key` over `history[:k]`) and find the longest prefix whose directory exists.
- Copy that directory (state and message index) forward as the starting state.
- Replay the exchanges after the prefix through `summarize_latest_state`, at full cost, or accept the gap and mark the state stale until a replay is run (open question in section 10).
- Garbage-collect orphaned directories (swiped replies, abandoned branches) by age, keeping the last N branches.

Cost: 0 LLM calls to restore, plus N turns of DSS calls if the missing turns are replayed.

Finer-grained rewind, building on 3.1 (only if the first step is not enough):

- Every `_history` entry carries `node` and `actor`.
- `rewind(to_node)` applies the inverse of every `llm` entry with `node` greater than `to_node`, newest first. The `old` values are already stored. `user` entries are never reverted automatically; they are listed for the user instead (3.2).
- The 3.1 compaction cap can discard the old values rewind needs. Keep the last N turns (start at 10) uncompacted and compact only older entries.
- State that is not a field patch needs its own inverse: record creates as `{path, old: <absent>}`, keep the previous version of regenerated digests (`synopsis`), and refuse to rewind past a chapter or arc archival boundary at first.
- Detect a rewind by comparing the chat's message-node list with the last applied node, so no host event is required.
- The message-chunk index has the same problem: chunks for removed messages must be dropped. Check whether `message_chunker` already does this.

Alternatives found in other systems (`memory_comparisons.md`, criterion 8):

- **Commit log keyed by message (Luker).** State is the replay of every commit whose (message, swipe) still exists in the live chat, so edits, swipes and deletes need no event handler and no inverse patches. Luker's log once grew to about 19 MB for about 1 MB of data until commits became incremental, so store diffs, not snapshots. DSS's `_history` is already per-field diffs. The decision is whether it becomes the source of truth (`headless_engine_ui_plan.md` P1) or stays an audit trail with inverse patches applied on rewind.
- **Commit lag for slow subjects (OpenVault, Summaryception).** Both avoid rollback by processing only turns that have aged out of a recent window, so a swipe never reaches memory. That cannot work for `current_scene`, because the writer needs this turn's state, but it suits subjects that change slowly and are consumed later: chapters, arcs, `knowledge` and the canon synopsis. Apply those N turns late, and rewind only has to cover the fast subjects.
- **Provenance counting (Graphiti).** Each edge records its source episodes, and `remove_episode` deletes only what no other episode supports. It does not matter for tail rewinds, but it would if mid-chat edits are ever supported.

Cost: 0 LLM calls. Verify with the Rewind probe (9).

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
KnowledgeItem:                       # one shared `knowledge` subject
  fact:          str                 # one atomic statement
  type:          fact | secret | suspicion | rumor | lie | revelation
  status:        active | confirmed | debunked | resolved
  source:        Source              # witnessed | stated | demonstrated | overheard | inferred
  importance:    Importance          # score (0-100) + reason + faction

Cross-entity links (on the connecting side):
  Character.knows:  dict[str, EpistemicStance]   # key=item slug, value=stance
  Group.knows:      dict[str, EpistemicStance]
  Scene.learned:    dict[str, EventKnowledgeRole]  # key=item slug, value=scene's role

EpistemicStance:    knows | believes | suspects | spreading | sees_through
EventKnowledgeRole: established | reinforced | challenged | debunked
```

- **Item is self-contained.** `type` (what kind of knowledge) and `status` (narrative lifecycle) are orthogonal axes. A suspicion that gets confirmed is `type: suspicion, status: confirmed`. A secret that gets exposed is `type: secret → revelation, status: resolved`.
- **Links live on the connecting side.** Character/Group carry `knows` (who knows what, and with what stance); Scene carries `learned` (what the scene established/challenged). The item doesn't point outward. New characters just add their `knows` entries without retroactively editing item lists.
- **Stance is typed.** `EpistemicStance` is a `field_options`-constrained alias — the LLM gets a hard `enum` in the prompt. `believes` vs `suspects` vs `sees_through` distinguishes epistemic posture toward the same item.
- **Memories (deferred):** a per-character "Memories" layer (what they *remember experiencing*, the prose/experiential view) is not an independent generation target. It falls out as a derived prompt-build projection from `knows` edges + milestone participations (7.1) + `_history` entries (3.1). No extra LLM call. The structured data is the source of truth; prose is a view.

**Consumption.** The `Knowledge` subject was generated but never rendered for the writer: `RetrievalContext` had no `knowledge` field, `retrieve_context` never populated one, and `context_engine` skips any `_context_order` type whose attribute is `None`. That is fixed. The structural cause (a new subject is silently dropped unless the retriever has bespoke code for it) is addressed in `retrieval_plan.md` R1. What remains is the shape of the rendering:

- **Show the stance of every character in the current scene,** computed in code: `knows: A, B | suspects: C | unaware: D`, where `unaware` is the scene cast with no edge to the item. The writer then sees who must not act on a fact, which is the Leak probe's pass condition. Smart-Memory tags `unaware` explicitly per character; deriving it from a missing edge avoids a generated field.
- **Select items** first by an edge to a scene character, then by importance and status (`active` and `confirmed` before `debunked` and `resolved`), inside a budget set by the shared scorer (`retrieval_plan.md` R8).
- **Omit** items with no edge to any scene character, rather than showing them unfiltered.

Verify with the Leak and Knowledge render probes (9).

**Implementation:** `Knowledge` wrapper with `KnowledgeMap` (`dict[str, KnowledgeItem]`). Triggers: `on_new_scene` → `[add_new, perform_gate_check]` on `KnowledgeMap` (creation), `query_branch_for_changes` on `Knowledge` (modification). `Source` alias with `field_options` (5 values). `EpistemicStance` alias with `field_options` (5 values). `type` and `status` use `field_options` in KnowledgeItem defaults. `score_range: [0,100]` on Importance. `knows`/`learned` declared in `relationship_format` on wrapper classes (`Characters`, `Groups`, `StoryEvents`) for entity graph edge extraction — the string-value branch in `_process_relationship_dict` handles `dict[str, str]` edge labels. Format template renders entries with type/status/fact/importance/source. Schema-driven subject discovery — no code changes needed for the subject itself.

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
- **Place state (candidate):** Smart-Memory's state cards track `hazards`, `accessibility`, `damage` and `occupants` for places, which suits a heist (a sealed door, a tripped alarm, a collapsed corridor). Keep `hazards`, `accessibility` and `damage` as short overwritten strings on location elements. Derive `occupants` in code from `last_seen` instead of storing it. Add it only if the writer context will use it.
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

### 7.1a Tune PastEvents prompting + rename to `offscreen` (deferred)

The `PastEvents` alias was refactored from inheriting `StoryEvents` to `dict[str, StoryEvent]` with its own triggers and tightened prompts (commit `ae9f04cf`). The query prompt now asks specifically "did a character REVEAL/MENTION/REFERENCE a past event?" with concrete qualifying examples, and the creation prompt states the event happened before the story. Result: 39 in-narrative entries → 1 (a 97% reduction). The remaining 1 is a borderline case (an in-narrative "discovery" moment the model latched onto).

**Why the name `past` is wrong**: The real distinguishing axis is **what the reader witnessed**, not **when it happened on the timeline**. "Past" implies a temporal boundary (before now), which over-constrains: in stories with timeskips (e.g. The Last of Us), an event from 20 years ago is backstory to a character introduced in the present, even though it's "during the story's timeline." Late-introduced characters compound this — their backstory can overlap with earlier acts from the reader's perspective. **`offscreen`** captures the actual semantics: events the reader did not see depicted, regardless of temporal position. It also aligns with the `origin` enum value `off_screen` (7.1).

**Rename scope** (when done alongside the prompt tuning):
- Schema key: `events.past` → `events.offscreen`
- Schema definition: `PastEvents` → `OffscreenEvents`
- Format template section header
- Spreadsheet beats: `subject: "past"` → `"offscreen"`
- Any code references to `events.past`

**Prompt tuning direction**: The `origin` enum (7.1) is the proper long-term solution — `backstory` vs `on_page` vs `off_screen` vs `reported` per event, validated in code against the message node. The prompting should encode the *intent* ("this is an offscreen event the character references, not something the reader witnessed") without over-constraining the temporal boundary. The cross-tier dedup guardrail (7.1) would catch the residual edge case (an in-narrative event the model misclassifies as offscreen).

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

Statuses: `open | resolved | dormant` are declared by the model. Add a computed `stalled` (open and untouched for N scenes, from `last_touched_scene`), so staleness is never generated. TunnelVision's arc tracker distinguishes active, stalled, resolved and abandoned, and keeps a just-resolved arc in the prompt for a few turns so the payoff gets written. Do the same: include threads resolved within the last 2 to 3 scenes in the writer block, labelled resolved.

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
2a. **3.4** Rewind safety. First implement `backtrack_history` by nearest-prefix snapshot (tail swipes already work). Then add `node` and `actor` to `_history` if finer rollback is wanted.
2b. **3.3** Entity resolution. One shared resolver, then move the retriever onto it (`retrieval_plan.md` R2). Needs the Resolution probe first.
2c. **3.2 (user rung)** Stamp `user` and lock paths in code. Needed before the UI allows editing (`headless_engine_ui_plan.md` P1).
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
| Resolution | 3.3 | No mention in a gold set (`The Vault`, short names, titles, nicknames, names with punctuation) resolves to the wrong entity. Zero `re.error` from entity names. |
| Retrieval | 3.3, `retrieval_plan.md` R0 | Replay only, no LLM calls. Planted facts appear in the injected context (recall@k). Report the share of injected tokens that duplicate the live window and the false-seed rate. A beat served at consecutive checkpoints with no new mentions is not re-served. |
| Knowledge render | 4.2 | For each scene, an item whose edges exclude a scene character renders that character as `unaware`. No item without an edge to the scene cast renders unfiltered. |
| Rewind | 3.4 | Swipe, edit or delete at turn N: state equals the pre-N state (fields, new entries, history). `user`-stamped edits are untouched and reported. |
| User edit | 3.2 | No LLM patch overwrites a `user`-stamped path, and the attempt is logged. |
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
- **Retrieval needs no LLM at all.** Replay the recorded transcript and state snapshots through the retriever only, so a retrieval change is checked in minutes (`retrieval_plan.md` R0).

Expect this genre to stress `status` over `description` (power-ups and injuries are short-lived) and the change log, which also makes it a good check on 5.1.

## 10. Open questions

Entity references (deferred):

- The LLM sometimes paraphrases entity names (e.g. writing "a location in the building" instead of the slug `conduit_junction`) when populating cross-reference fields (`knows`, `relationships`, `learned`, `participants`). Two complementary mitigations considered:
  - **Prompt-side:** a `_references: "<subject>"` convention in schema defaults that generates a comment hint ("Keys must be exact slugs from the <subject> subject") in the example JSON. Low effort — effectively a template wording change. Can be done incrementally per field as the need arises.
  - **Validation-side:** post-apply set-membership check. For a field with `_references: "knowledge"`, validate that every key (dict field) or element (list field) exists in `keys(data["knowledge"])`. Type-aware: dict → validate keys, list → validate elements. CurrentScene should be excluded (it legitimately references unidentified entities before they're added to their subjects). More complex — needs the data tree at validation time and per-type logic.
  - **Shared resolver (3.3):** both mitigations can reuse one resolver. `_resolve_dict_key` and `_retarget_dict_key` already handle lookalike spellings, but no string matching fixes a paraphrase such as "a location in the building", so keep the validation-side check for those.

Data model:

- Should `knowledge` be a shared subject or live on each entity? The plan assumes shared with derived views.
- Should characters and elements get a declared `aliases` field, or keep deriving aliases from relationship entries (3.3)?
- Should milestones stay a stored list or become a derived view over event participants? The derived view needs no per-character call but needs backstory events for pre-story milestones.
- Should threads and events merge long-term? Keep them separate until the overlap proves too large. Should `Objective` fold into threads or stay as the structured form of `main_objective`?
- Should rule `scope` be free text or a reference to a location or group entry? References are validatable but couple rules to the entity subjects.
- Should the ladders live in `rules` (assumed) or their own small subject? Is `yardstick` free text or a fixed list of reference objects (easier to validate, less flexible)? Should a power-scaled flag set at creation avoid carrying the ladders in a heist or slice-of-life story?
- How should existing `attributes: list[str]` data migrate to a keyed structure? Keying on the text before the first colon works for the sample data, but other stories may not follow that pattern.
- How much change history per field before compaction? Start at 20 entries and tune.

To verify in code:

- After `backtrack_history` restores the nearest earlier snapshot (3.4), how are the turns between that snapshot and the current history handled: replay through DSS at full cost, or accept the gap? Where is `delete_message_chunks` called from, if anywhere?
- Confirmed: `_safe_subject` hides failures from `_get_relevant_elements` and `_get_relevant_groups` (3.3). Should it surface a visible warning in the UI and the soak log?
- How much of `last_x_messages` and the rolling summaries overlaps the RAG-selected messages (`retrieval_plan.md` R3)?
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

Sections added or extended 2026-11-05: new 3.3 and 3.4, 4.2 Consumption; extended 3.1, 3.2, 5.4, 7.2, 8 (steps 2a to 2c), 9 and 10.
Added 2026-11-06 after reading Luker, OpenVault, Graphiti and Summaryception: 3.1 (as-of reads), 3.3 (staged resolver, redirects, identity fields), 3.4 (commit-log and commit-lag alternatives).
Revised 2026-11-06 after tracing the history-hash directories in the code: 3.4 now starts with `backtrack_history`, and the as-of note in 3.1 changed.
