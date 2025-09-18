# DAYNA Story Summarizer

chunk messages `#chunk_messages()`:

- [x] chunk messages by paragraph by line
- [x] include message summary in database
- [ ] check if any gaps in history

<br>

ui:

- [x] make copies of each original tgwui util/module method
- [ ] integrate story datetime into metadata (~~integrate time into boogaPlus using shared.message\_data timestamps~~)

<br>

<br>

**IMMEDIATE TODO**:

- [x] Handle start of chat
    - [x] Create subjects off of schema
        - [x] Pass schema refs
        - [x] Generate example with descriptions of branches and values
        - [x] Handle new chars/groups
    - [x] Create current\_scene
        - [x] Pass schema refs
        - [x] Generate example with descriptions of branches and values
            - [x] Maybe only generate “start” and copy to “now”? For now, keep as is

- [x] Properly update subjects and add new ones when mentioned
    - [x] Update current scene “now”, keeping “start” in its original state
    - [x] Give full schema and example like “Handle start of chat”
        - [ ] Common gen prompt for all fields (general, unspecific) until last line to save context + time + consistency?
    - [x] Only update when new scene? If last\_x > scene messages (i.e. when relevant messages being truncated), add to data before truncation
- [x] Create new scene
    - [x] ~~Default to create when new chat~~ <— “Handle start of chat”
    - [x] End scene button or user-input prefix
    - [x] Use current\_scene data to generate a new key for “scenes” in Events
        - [ ] Summarize in same history line (keep the previous response in context during summarization)
    - [x] Generate new current\_scene
        - [x] Refactor update\_when + perform\_update format to single perform\_update\_when property
    - [ ] Auto-detect scene end
        - [ ] Decide whether to count user\_input as new scene, or start from output
    - [ ] Detect important events at end of scene?
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

<br>

Current TODO:

- [ ] Add general info
    - [x] Always have original state\['context'\] in custom\_state\['context'\]
    - [ ] sum of current\_scene
    - [x] editable user+assistant tendencies (e.g. third-person)
- [x] Handle start of chat
- [ ] Create new scene
    - [ ] End scene button or user-input prefix
    - [ ] Use current\_scene data to generate a key for “scenes” in Events
        - [ ] Summarize in same history line (keep the previous response in context during summarization)
    - [ ] Auto-detect scene end
        - [ ] Also detect events

- [ ] last\_x is all messages in current scene
    - [ ] Could be what the LLM decides is relevant from the last scene (e.g. +2 msg context)?
    - [ ] Could just be specific conditions? When to use full messages vs summarizations
- [ ] User-defined example message format
- [x] More sum prompt options (gate check → “YES”, “NO”, “SPECIFIC\_FIELD\_UPDATE” aka query branch fields to change instead of modifications all in one prompt)
- [ ] Subject data UI (tree)
    - [ ] Update realtime!!
    - [ ] Reset history\_path on save
- [ ] Get the last time a character was in a scene
- [ ] Persist custom\_state for specific branch defaults?
- [ ] Add # of attempts during failure
- [ ] Expand keys to full data when getting relevant info for context retrieval (e.g. “events” in relationships)
- [ ] Two context modes:
    - [ ] DAYNA mode - Ask questions then instruct
        - [ ] Also tool calling mode
    - [ ] Character mode (persona) - Provide as`"context"`  then place in spot
    - [ ] Also parsing/imitation for both modes
- [ ] Short-term goal for this scene/event
- [ ] Allow the user to put instructions via "\[\[NOTE HERE\]\]" within the message. Whether to persist this internally in history or remove it is unclear
    - [ ] Also disable sum gen and/or give specific keywords to direct generation? (e.g. “suzie dead by pure accident”)
- [ ] Update “importance” values throughout
- [ ] Separate updates into “categories” i.e. “major”, “minor”, “side”
- [x] Current scene should include current directive and should persist through to the next scene, maybe character motivations would be in characters.json and plot goals would be in current\_scenes+scenes, or perhaps that’s what general\_summarization will be for (general\_context?)
- [ ] event\_ids + scene\_ids instead of event\_id + scene\_id
- [ ] When generating instruction, include possible`user_instr` given by the user
- [ ] give user instructions for regenerate, maybe separate extension (“Regenerate with instructions” vs “Regenerate with feedback (explicit)”)
- [ ] “The user’s input is the highest priority; if anything said or done by a character doesn’t match its personality trait in the existing knowledge base, consider whether this should be changed in the knowledge base, or was done intentionally.”
- [ ] General info for each subject category (i.e. characters, groups, events)
- [ ] RAG for each character’s individual memory (most memorable moments for specific scene/event)
    - [ ] Also each relationship (essentially what`"events":`  is for, but better)
- [ ] Detect edits and compare original vs new to determine what to change in history\_str
- [ ] Add instructions.json (`instr`) toggle
- [ ] Refactor `_update_recursive` into more standard recursive style

<br>

Far TODO:

1. Schema UI (flow or tree?)
    - Gray out children when update\_prompt or branch\_query\_prompt is checked
    - Have immutable “Default” in case a custom schema breaks
    - Upload JSON as schema
    - Download schema JSON
    - Convert JSON-schema format to subjects\_schema.json
2. Save full UI settings (like tgwui “Session”)
3. Sync tgwui generation parameters to dss
4. Add ability to change specific names/keywords in message\_index/subject data
5. User-defined story structure/objectives (specify future scenes/events)
    - Include “lasts until” or time range during current\_scene (mainly for general DAYNA)
6. Optionally sync state seed (state → custom state)
7. Add locations schema?
8. Send to Notebook
9. Logs UI (show field updates and creations)
    - Also track updates to show LLM?
10. Force “YES” gate check for CurrentScene?
11. Update character IDs when necessary
12. Generate subject data with input (manually ask DAYNA to generate something based off of current context)
13. Show summarization progress
    - Include ETA estimation / progress %
14. Different prompts depending on always, next scene, first message, etc.
15. Generate initial subject data based off of general character context
16. Note edits to bot replies and include in general\_info instructions
17. Delete current history\_str (self.last) - for manual edits to data
18. Update history-context in realtime with DataSummarizer? Will potentially hurt eval time
19. For events: additional context from the future
20. Trigger archive at end of scene (to archive values that aren’t needed in the main subjects files but may still be needed for extra details in the future)?

  

## OLD TODO (Handle message `#handle_input_output()`)

on pre-gen `#custom_generate_chat_prompt()`:

- [x] get history\_path (current context)
    - [x] history\_path = character\_path / has⁠h(og\_internal\_history)
        - [ ] TODO: retrieve history\_path using hashes in map stored on character\_path (order history\_path by index i.e. “10\_1”, “10\_2”, etc.)
    - [ ] TODO: if not history\_path.exists():;;
        - [ ] backtrack history until existing path is found
        - [ ] generate summaries from last existing path (`#handle_input_output()`)

- [x] `#retrieve_context()` 
- [x] retrieve instr\_prompt
    - [x] if persisted instr\_prompt does not exist:
        - [ ] generate instr\_prompt
            - [ ] ADD NEW INSTR FORMATTING, MAYBE PURE INSTR OVER CHAT-INSTR
        - [x] persist instr\_prompt using user\_input
    - [x] else:
        - [x] retrieve persisted instr\_prompt

on output `#handle_output()`:

- [x] get history\_path (current context)
    - [x] history\_path = character\_path / hash(og\_internal\_history)
    - [ ] TODO: if not history\_path.exists():
        - [ ] backtrack history until existing path is found
        - [ ] generate summaries from last existing path (`#handle_input_output()`)

- [x] `#retrieve_context()`
- [x] get history\_path (new context)
    - [x] history\_path = character\_path / hash(new\_internal\_history)
- [ ] generate and store summaries
    - [ ] (group together unknown character data? i.e. “traits & status --- unknown”)
    - [ ] recursive generation formatting
        - [ ] start from lowest level possible (`recurse → gen`  instead of `gen → recurse` )
        - [ ] indicate final gen + specify prompt template
            - [ ] tuple?
            - [ ] `_attr` ?
            - [ ] eval vs str.format
        - [ ] push to`keys` and set current `data` 
    - [x] ~~characters, groups, events/scenes, current scene~~
    - [x] ~~messages~~
        - [ ] ~~if id f"{message\_idxs\[i\]}\_summary" not exist~~ 
        - [x] ~~summary generation draft~~
    - [x] user-defined info (describe which subjects to add to?)
        - [x] retrieve info
        - [x] summary generation draft

retrieve context `#retrieve_context()`:

- [x] user-defined info
    - [x] characters
    - [x] groups
    - [x] events/scenes
    - [x] current scene
- [x] messages
    - [x] RAG draft
    - [ ] Advanced RAG
    - [ ] Add scene summarization and surrounding messages
    - \[ \]
- [ ] TODO: general summarization
- [ ] TODO: All lines spoken to, from, or about specific subject depending on importance threshold (>10/100?)