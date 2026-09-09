# DAYNA Story Summarizer

---

## Realtime UI (v0.1.0)

- [x] SSE server for real-time UI updates
- [x] PhaseManager for progress tracking
- [x] Streaming UI panel with collapsible phases
- [ ] **BUG FIX: Buffer-length polling stops when ring buffer fills** (sse_server.py:171)
- [ ] Improve error handling for child phases in _update_recursive

## Schema editor UI (v0.2.0)

---

## IMMEDIATE TODO

- [x] Handle start of chat
    - [x] Create subjects off of schema
        - [x] Pass schema refs
        - [x] Generate example with descriptions of branches and values
        - [ ] Handle new chars/groups
    - [x] Create current\_scene
        - [x] Pass schema refs
        - [x] Generate example with descriptions of branches and values
            - [x] Maybe only generate “start” and copy to “now”? For now, keep as is

- [x] Properly update subjects and add new ones when mentioned
    - [x] Update current scene “now”, keeping “start” in its original state
    - [x] Give full schema and example like “Handle start of chat”
        - [x] Common gen prompt for all fields (general, unspecific) until last line to save context + time + consistency (static-first/dynamic-last template ordering shipped)
    - [x] Only update when new scene? If last\_x > scene messages (i.e. when relevant messages being truncated), add to data before truncation
- [x] Add general info
    - [x] writing style (~~editable user+assistant tendencies (e.g. third-person)~~)
        - [ ] Scene length
        - [ ] Sentence length
        - [x] Paragraph length
        - [x] Writing perspective
        - [ ] Writing _style_
    - [x] synopsis
    - [x] objectives (main objective)
    - [x] themes+tone
    - [x] custom\_state\['context'\]:
        - [x] Always have original state\['context'\] inside
    - [x] sum of last scene
    - [ ] ~~sum of current scene~~
- [ ] Finalize message node format (1-indexed, bot vs user, etc.)
- [ ] More schema/generation flow stuff:
    - [ ] For most cases, keep all data the same per current\_scene to save tons of time (prompt eval, data summarization, more prompt eval)
        - [ ] Weigh when extremely important updates need to happen? 

    - [ ] Optionally keep context info (`history_path`) the same until new scene (for prompt eval)

- [x] no\_update “trigger” (SceneStart uses it; enforced in data_summarizer)

<br>

## Current TODO

- [ ] Add general info
    - [x] Always have original state\['context'\] in custom\_state\['context'\]
    - [ ] sum of current\_scene
    - [x] editable user+assistant tendencies (e.g. third-person)
- [x] Handle start of chat
- [x] Create new scene
    - [x] ~~Default to create when new chat~~ <— “Handle start of chat”
    - [x] End scene button or user-input prefix
    - [x] Use current\_scene data to generate a new key for “scenes” in Events
        - [ ] Summarize in same history line (keep the previous response in context during summarization)
    - [x] Generate new current\_scene
        - [x] Refactor update\_when + perform\_update format to single perform\_update\_when property
    - [X] Auto-detect scene end
        - [ ] Decide whether to count user\_input as new scene, or start from output
    - [ ] Detect important events at end of scene?
- [ ] last\_x is all messages in current scene
    - [ ] Could be what the LLM decides is relevant from the last scene (e.g. +2 msg context)?
    - [ ] Could just be specific conditions? When to use full messages vs summarizations
- [ ] User-defined example message format
- [x] More sum prompt options (gate check → “YES”, “NO”, “SPECIFIC\_FIELD\_UPDATE” aka query branch fields to change instead of modifications all in one prompt)
- [ ] Subject data UI (tree)
    - [ ] Update realtime!!
    - [ ] Reset history\_path on save
- [ ] Get the last time a character was in a scene
    - [ ] For any subject?
    - [ ] Mentions?
- [ ] Persist custom\_state for specific branch defaults?
- [ ] Add # of attempts during failure
- [ ] Expand keys to full data when getting relevant info for context retrieval (e.g. “events” in relationships)
- [ ] Two context modes:
    - [ ] DAYNA mode - Ask questions then instruct
        - [ ] Also tool calling mode
    - [ ] Character mode (persona) - Provide as`"context"`  then place in spot
    - [ ] Also parsing/imitation for both modes
- [ ] Short-term goal for this scene/event (`general_info`?)
- [ ] Allow the user to put instructions via "\[\[NOTE HERE\]\]" within the message. Whether to preserve this internally in history or remove it is unclear
    - [ ] Also disable sum gen and/or give specific keywords to direct generation? (e.g. “suzie dead by pure accident”)
- [x] Update “importance” values throughout (universal salience×weight rubric, format_templates.json importance_scale)
- [ ] Temporary importance offset (`temp_offset` sibling beside `Importance.score`, default 0): captures moment-to-moment urgency (e.g. +40 “saving their life is my utmost priority right now”, -10 after an argument) WITHOUT inflating the durable long-term `score`. Applies ONLY to the effective value used for context ordering / top-of-mind, never to the stored `score` or the roster threshold (below 50 → one-line). Optional/unset = ignored. Deferred until the universal Importance Scale (format_templates.json) settles.
- [ ] Separate updates into “categories” i.e. “major”, “minor”, “side”
- [x] Current scene should include current directive and should persist through to the next scene, maybe character motivations would be in characters.json and plot goals would be in current\_scenes+scenes, or perhaps that’s what general\_summarization will be for (general\_context?)
- [ ] event\_ids + scene\_ids instead of event\_id + scene\_id
- [ ] When generating instruction, include possible `user_instr` given by the user (for `instr` only, if in a “\[\[NOTE HERE\]\]" format exclude it from the actual message)
- [ ] give user instructions for regenerate, maybe separate extension (“Regenerate with instructions” vs “Regenerate with user feedback (explicit):”)
- [ ] “The user’s input is the highest priority; if anything said or done by a character doesn’t match its personality trait in the existing knowledge base, consider whether this should be changed in the knowledge base, or was done intentionally.”
- [ ] General info for each subject category (i.e. characters, groups, events)
- [ ] RAG for each character’s individual memory (most memorable moments for specific scene/event)
    - [ ] Also each relationship (essentially what`"events":`  is for, but better)
    - [ ] Maybe new schema type for “Memory”?
- [ ] Detect edits and compare original vs new to determine what to change in history\_str
- [ ] Add instructions.json (`instr`) toggle
- [x] Refactor `_update_recursive` into more standard recursive style
- [ ] Save message word count?
- [ ] What specific branches should be updated then looping those recursively (rather than a gate check)
    - [ ] Essentially function calling
- [ ] Message RAG: Include the “speaker” (`char1` vs `char2`) per message (not just per node necessarily)
- [x] Triggers with specific defaults (prompt templates), rather than defaults and triggers separate?
- [ ] `on_new_scene_or_truncate` 
- [ ] Include `msg_` or first few characters of message when creating `history_str` 
- [ ] More meaningful “importance”:
    - [ ] Depending on level, character-specific "thoughts" summary on specific events/relationships
    - [ ] `100` = Always on their mind
`80` = Frequently on their mind
`50` = Friend
`30` = Remembers when spoken of
`20` = Hardly remember when spoken of
`10` = Struggle to remember
`5` = Familiar, tip of the tongue
`0` = Stranger
    - [X] “favour” level for negative/positive relationship
- [ ] Entity aggregation decay
- [ ] Instead of expanding lists (modifying the actual data structure), just add comments for each element e.g.

```jsonc
[
  "foo", // 0
  "bar", // 1
  "baz", // 2
]
```

- [ ] "Memories" - notes for each character for each scene and event — summary of specific things that happened that are memorable for this specific character
- [ ] Analyze "user intent" before generating instructions
- [ ] Track changes in-scene then actually apply them at the end
- [ ] Persist DataSummarizer change-diffs into message_index (MessageChunker)
    - `_apply_branch_updates` (data_summarizer.py:2423) already builds `{"path", "value", "old_value"}` per applied update and pushes it through pm.done_step — collect these per turn into ONE turn-level diff instead of leaving them transient in the phase log
    - Attach to stored chunks via `update_node_metadata_by_message_idx` (context_retriever.py:1913) as `data_diff` metadata (one diff per message\_idx, shared across that turn's para/sentence nodes)
    - Two channels: scene turns ALWAYS carry what data changed in message\_index (structured diff); the text message summary keeps carrying general (important) updates in prose
    - Feeds: message-edit reversion (apply inverse diff), per-character "Memories", frozen-loop debugging
    - Design (2026-08-20): store the FULL audit unfiltered, keyed by resolved entity path + message\_idx; filtering is a READ-side projection only
    - "General history" projection (per-entity, computed at prompt-build): path-scope under the entity incl. sub-parts (relationships/importance children; relationship updates union both entities) -> drop schema-classified volatile fields (\_recent\_state, now.\*) -> collapse per-path runs to last K transitions -> rank by recency x inverse path-frequency (churn sinks, quiet-but-moved rises) -> render compact "turn N: path: old -> new" lines, token-capped
    - Injection: tail of the per-entry context block branch\_update already renders (dynamic-last, prefix-cache safe); audit data traced automatically when an entity is being updated
    - Metadata alone is NOT embedded — RAG-retrievable mirror comes from the distilled tier below, not raw metadata
    - Optional quality tier: at scene-archive, distill each entity's turn-diffs into 2–3 prose changelog lines stored engine-managed on the entry (`_change_log`, same reserved-field precedent as \_recent\_state) — old scenes stay queryable without replaying raw audits
- [ ] Include simulated "character thoughts" when summarizing (or even message_index?)
- [ ] Format_messages shouldn't include index `10. <message>` to avoid re-evaluating; instead, use normal format but at the end give summaries specifying which message is which
- [ ] More detailed relationship info (temporary status etc.)
- [ ] `"extra_info"` fields?
- [ ] "Entities" subject types
    - [ ] Locations
    - [ ] Items/resources
    - [ ] Creatures
    - [ ] Goals?
- [ ] 5x5 grid to place characters in a scene (LOW PRIORITY, parked 2026-08-20; schema-driven, no hardcoded engine dispatch)
    - New current\_scene field (e.g. `positions`: cell -> entity, or entity -> \[row, col\]) riding the normal update machinery with its own prompt template
    - Open: update cadence — every-turn branch\_update (costly; current\_scene.now is already a full-branch update every turn) vs new-scene rewrite + movement-gated query
    - Open: render as an ASCII grid into the reply context so the writer model reasons about adjacency/earshot/line-of-sight (who can slip away unnoticed)
    - Open: absolute room map vs relative-to-speaker positions; multi-room scenes
    - Small-model risk: positional churn / teleporting — may need a stability nudge ("only move cells the text implies")
    - Related: Entities > Locations
- [ ] Granular Events for more precise RAG? Basically summarized messages
- [ ] Only perform initial population after first scene?
- [x] Execute DataSummarizer in parallel - stopping if interrupted - only truncating/updating RetrievalContext after a _full_ run (max_subject_workers, schema-order reassembly)
- [ ] Let DSS run constantly (Historian), giving long windows to summarize

<br>

## Scale & aggregation units

- [ ] Cadence profiles — config-driven chapter/arc bounds; `compressed` (current 4-8 scenes/10 hard, 3-6 chapters/12 hard) vs `campaign` (~15-40 scenes/chapter, ~8-24 chapters/arc); schema defaults stay compressed
- [ ] Canon synopsis subject — ~300-token narrative digest written at chapter archival
- [ ] Renderer demotion rule — closed-arc entities render as roster lines; full detail reserved for open arcs
- [ ] Resolved-state export / sequel import — chain sessions like D&D sessions (each soak run = one session); notes span sessions
- [ ] Campaign-scale spreadsheets + chained-session pilot (Tier 3 test ladder)
- [ ] Arc `_active`/`_resolved` flags with render-first ordering for open threads

## Memory quality

- [ ] Uncertainty-preservation directives in character templates — record claims as claims; never resolve identities/aliases the story holds open
- [ ] Discovery similarity guard — mpnet cosine on add_new candidates (>~0.85 → merge instead of create)
- [ ] Background merge/consolidation pass for duplicate entities (buyer/vendor class)
- [ ] Continuity contradiction probe — proposed-vs-current branch diff between parse and commit (schema-flag gated)
- [ ] Rolling-overview findings → instruction-generation feedback loop
- [ ] Entry-selection overlap rule — entities mentioned in the latest exchange always whitelisted for revision

## Epistemics

- [ ] Secrets subject — normalized rows {summary, stance: knows|suspects|believes_falsely|unaware|hiding, owner, target?, about?, status, importance}
- [ ] Deception maintenance — reveals flip status/stance on a single small branch under mark_field
- [ ] POV injection — group rows by stance over owner ∈ current_scene.now.who

## Steering & planning

- [ ] `[[...]]` inline steering directives — strip from prose; inject into instruction generation + summarization; guide-side usage for beat-steering
- [ ] Engine-native planning/"spreadsheet" system — PARKED until good 100-turn results

## Production reliability

- [ ] Handle `_continue` — debug how TGWUI passes `_continue`, `user_input`, and `state["history"]` during "Continue"
- [ ] Handle message edits — detect changes, re-summarize only affected fields (pairs with the change-diff persistence item above)
- [ ] Stop/cancel mid-generation — restore original seed, clean partial summaries
- [ ] Backtrack history when no history_path exists
- [ ] Graceful child-phase failure in _update_recursive
- [ ] Schema editor UI — tree view + JSON import/export + immutable default fallback
- [ ] Fix banned_prefixes str/list parse edge case (script.py)
- [ ] Cache current_scene "now" — skip summarization when unchanged

## Model slate

- [ ] Magnum-v4-12B live test (Mistral alias patch applied)
- [ ] Gemma4 via llama.cpp or C++ port (docs/plans/lmdeploy_gemma4_pr_plan.md)
- [ ] Small-model JSON compliance sweep → decide tagged-text fallback profile

## Tooling & UI extras

- [ ] Dry-run turns — compute updates, stream over SSE, approve/discard before commit
- [ ] Context telemetry — per-block token counts over the SSE channel
- [ ] Entity timeline view from recorded old_values (supersession history)
- [ ] Entity graph canvas visualization (served off the SSE host)
- [ ] Away recap modal ("previously on…") from rolling summaries
- [ ] Generation-lifecycle hooks — swipe-abort, delete/edit re-summarization
- [ ] Dyad canonicalization pass — one canonical edge per relationship pair at graph build time
- [ ] Adaptive budget allocation — fewer memory tokens on action-heavy beats, more on dialogue
- [ ] MCP server wrapper over the import-clean core
