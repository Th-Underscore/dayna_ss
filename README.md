# DAYNA Story Summarizer

An autonomous agent that manages your story's long-term memory so you don't have to.

## The Problem

Language models are great at generating text, but they have one fatal flaw: they forget. Every character detail, plot twist, and relationship nuance you mentioned thirty messages ago? Gone. The model only sees what's in the current context window, and once it scrolls off, it might as well have never existed.

## The Solution

DAYNA doesn't just store your story data—it actively manages it. After each scene, the agent analyzes what happened, updates its knowledge base, and builds a structured "world model" that persists across the entire conversation.

The key is the schema. It's a JSON file that tells the agent *how* to reason about your story, *when* to update specific fields, and *why* certain changes matter. You can customize it to match any story structure you want (in the future, several examples will also be provided).

## How It Works

1. You chat with the AI normally
2. When a scene ends (currently triggered by a `NEXT SCENE:` prefix), the agent kicks into gear
3. It analyzes the scene, extracts important events, updates characters and relationships
4. A summary gets archived; a fresh "current scene" starts for the next beat
5. On the next response, the agent pulls relevant context from its knowledge base

The cycle repeats. Over time, the agent builds a rich, interconnected graph of your story's world.

## Features

- Scene-based memory consolidation with automatic archiving
- JSON schema-driven update logic you can customize
- Proactive detection of new characters and groups
- Hashed state snapshots for stable history retrieval
- Dynamic context retrieval (RAG) for relevant memory on demand
- Live progress UI showing each summarization phase in real-time

## Installation

```bash
cd /path/to/text-generation-webui/extensions
git clone https://github.com/Th-Underscore/dayna_ss.git
pip install -r dayna_ss/requirements.txt # in your text-generation-webui Python environment
```

## Status

Pre-alpha. The core cycle works, but this is very much a project in progress. Expect bugs, an exorbitant amount of debug logs, and a rapidly evolving feature set.

## Roadmap

- Auto-detect scene boundaries instead of manual triggers
- Character-specific memory retrieval ("what does Alice remember about this?")
- Importance weighting to skip trivial updates
- Interactive knowledge base UI for manual edits
- Schema editor for visual schema building
