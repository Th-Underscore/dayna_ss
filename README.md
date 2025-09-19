# DAYNA Story Summarizer
<sub>formerly [Super Story Summarizer](https://github.com/Th-Underscore/super_story_summarizer)</sub>

An autonomous agent that acts as your story's intelligent knowledge keeper.

### The Problem: The Forgetful AI

Long-term memory is the biggest challenge in narrative AI. Language models are powerful, but they live in the moment, constrained by a finite context window. Characters forget crucial plot points, mix up relationships, and lose track of the timeline. Existing solutions often require tedious manual updates to text files or use simple retrieval systems that fail to grasp the complex, evolving state of a story.

### The Solution: An Agent That Manages Itself

DAYNA Story Summarizer (DSS) is not just another memory system; it's an **autonomous agent** that actively curates and manages a structured "world model" of your story.

Instead of passively storing data, DSS uses the Language Model (LLM) itself to reason about the narrative after every major story beat. The key innovation is a **procedural schema**—a living blueprint that tells the agent *how*, *when*, and *why* to maintain its own knowledge base.

This transforms the story's memory from a static file into a dynamic, intelligent system that can:
*   **Reason** about which information is important and needs updating.
*   **Proactively discover** new characters and events as they are mentioned.
*   **Consolidate** memories at the end of a scene, learning from what just happened.
*   **Bootstrap** the entire story's initial state from a single character greeting.

### How It Works: The Scene-Based Memory Cycle

DSS treats the story as a sequence of scenes. This narrative structure is the key to its memory management loop:

1.  **During the Scene:** As you and the AI exchange messages, the agent continuously updates a temporary "working memory" of the current scene—who is present, what is happening, and where.
2.  **Scene Ends:** The user signals the end of the scene (e.g., with a command like `NEXT SCENE:`). This is the primary trigger for the agent's main cognitive process.
3.  **The Agent Learns:** The agent analyzes its working memory of the completed scene. It asks the LLM to summarize the key events, outcomes, and character developments.
4.  **Memory Consolidation:** This summary is archived as a permanent "memory" in the agent's long-term knowledge base. The agent then intelligently updates its understanding of characters, relationships, and the overarching plot based on what it just learned. A new, empty "working memory" is created for the next scene.

This cycle ensures that the agent's knowledge is always growing, relevant, and structured, allowing it to maintain perfect continuity from the first message to the thousandth.

---

### Current Features
- **Agentic Knowledge Management:** The agent intelligently updates characters, groups, the current scene, and events without manual intervention.
- **Schema-Driven Logic:** The agent's update behavior is defined by a powerful and customizable JSON schema that includes triggers, actions, and prompt templates.
- **Scene-Based Memory Consolidation:** Uses the end of a scene as a trigger to summarize and archive plot points into long-term memory, creating a robust story timeline.
- **Proactive Entity Detection:** The system identifies new characters and groups as they appear in the narrative and adds them to its knowledge base.
- **Versioned State History:** Each turn in the conversation creates a unique, hashed snapshot of the story state, enabling stable history.
- **Dynamic Context Retrieval (RAG):** Retrieves relevant messages, character data, and scene information to build a rich context for the LLM's next response.

### Installation
1.  Place the `dayna_ss` folder into your `text-generation-webui/extensions` directory.
2.  Run the following command to install the required dependencies:
    ```sh
    pip install -r extensions/dayna_ss/requirements.txt
    ```

---

### Project Status
⚠️ **Pre-Alpha:** This project is under heavy development. The core scene-based update cycle is now functional, but the system is still undergoing significant changes. Expect bugs, an exorbitant amount of debug logs, and a rapidly evolving feature set.

### General Roadmap
This is a high-level overview of planned features, simplified from the detailed [TODO list](./todo.md).

#### **Core Agent & Memory**
-   **Automated Event & Scene Detection:** Enhance the agent's ability to automatically detect when a scene has ended or a significant plot event has occurred.
-   **Advanced RAG:** Implement more sophisticated retrieval, including character-specific memories (e.g., "what does this character remember about this event?") and relationship-specific context.
-   **Importance Tracking:** Develop a system for the agent to weigh the "importance" of information, allowing it to prioritize major plot points over minor details during updates.
-   **Performance Optimizations:** Introduce logic to skip redundant updates during a scene, saving significant processing time.

#### **UI & User Experience (UI/UX)**
-   **Interactive Knowledge Base UI:** Create a real-time interface to view and manually edit all story data (characters, events, etc.) in a user-friendly tree or flow chart.
-   **Schema Editor:** Develop a dedicated UI for creating and customizing the `subjects_schema.json`, allowing users to easily define their own story structures.
-   **Logs & Progress UI:** Add UI panels for viewing the agent's update history and showing summarization progress in real-time.

#### **Prompting & Context**
-   **Dual Context Modes:** Allow the user to switch between **"DAYNA Mode"** (an analytical Q&A to load context) and **"Character Mode"** (direct context injection) to best suit the loaded model's strengths.
-   **In-Message Directives:** Enable users to give direct instructions to the agent within their message (e.g., `[[NOTE: Ensure this character is secretly angry.]]`), influencing the memory update without appearing in the story.

#### **Customization & Control**
-   **User-Defined Objectives:** Allow users to specify high-level story goals or future plot points for the agent to track.
-   **Fine-Grained UI Controls:** Add more UI options for customizing prompt templates, agent behaviors, and generation parameters.
-   **Manual Triggers:** Enable the user to manually trigger agent actions, such as generating a new character entry from a piece of text.