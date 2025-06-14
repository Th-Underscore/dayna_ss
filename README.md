# DAYNA Story Summarizer
<sub>formerly [Super Story Summarizer](https://github.com/Th-Underscore/super_story_summarizer)</sub>

A text-generation-webui extension that dynamically manages and summarizes your story context, because who wants to manually keep track of all that stuff?

## Current Features
- **Dynamic Story Data Management:**
    - Schema-driven creation and updates for characters, groups, scenes, and events.
    - LLM-powered summarization and dynamic updates to structured story data.
    - Initial data population for new story elements using LLM.
- **Context Retrieval (RAG):**
    - Enhanced message chunking with metadata (timestamps, speakers, entities).
    - Retrieval of relevant context for LLM prompts.
- **User Interface (Gradio-based):**
    - Management of characters, instruction templates, and generation parameters.
    - File operations for saving and loading story data.
- **Configuration:**
    - Schema configuration for characters, groups, scenes, and events per character and chat.
- **Performance:**
    - Background importing for heavy libraries (e.g., `llama_index`, `nltk`, `spacy`) to improve startup time.
- **Scene Management:**
    - Creation of new scenes and initial population of current scene data.
    - Updates to the "now" state of the current scene while preserving the "start" state.

## Installation
Drop this bad boy into your text-generation-webui/extensions folder, then through `cmd_windows.bat` (or `cmd_linux.sh` etc.) run:
```sh
pip -r extensions/dayna_ss/requirements.txt
```

## Planned Features
- **Advanced Subject Management:**
    - Comprehensive updates for subjects (characters, groups) when mentioned.
    - UI for real-time viewing and editing of subject data.
- **Enhanced Scene and Event Handling:**
    - Manual and automatic detection for scene endings (currently based on "NEXT SCENE:" prefix).
    - Generation of new scenes and linking them to events.
    - Summarization of scenes within the same history context.
- **General Summarization Improvements:**
    - Contextual summarization of current and past scenes.
    - Editable user/assistant tendencies for summarization style.
- **Improved LLM Interaction:**
    - More granular prompt options for data updates (e.g., gate checks, specific field queries).
    - User-defined example message formats and instructions.
- **Memory and Context:**
    - `last_x` to include all messages in the current scene.
    - Tracking last and relevant appearances of subjects in scenes.
    - Expansion of keys to full data for context retrieval.
    - Dual context modes (instruction-based and character-persona-based). Could also imitate \<think> tags.
- **User Experience & Tooling:**
    - Schema management UI (view, edit, import/export).
    - Saving/loading of full UI settings.
    - UI for logs and summarization progress.
    - Ability to insert in-message notes/instructions.
- **Data Integrity and Control:**
    - Tracking update attempts and failures.
    - Persisting custom state for branch defaults.
    - Updating "importance" values for relative subjects.

## Configuration
- Active (iterative) vs Total (comprehensive) summarization modes
- Customizable summary weights and event breaks
- Timestamp options
- Data schema and prompt templates
- Subject mapping configuration
- Memory management settings

## Usage
Will be updated as the project matures.

## State of this Project
This project is still pre-alpha, not yet fit for use. Though I've been working on it sporadically, I've had enough time the past couple weeks to rewrite most of the logic, now making it "functional" — pages of dumps and debugs included — but still very, _very_ much in the works.
