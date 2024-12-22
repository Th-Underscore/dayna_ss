# DAYNA Story Summarizer
<sub>formerly [Super Story Summarizer](https://github.com/Th-Underscore/super_story_summarizer)</sub>

A text-generation-webui extension that dynamically manages and summarizes your story context, because who wants to manually keep track of all that stuff?

## Current features
- Chunking and retrieval of specific messages based on relevance (untuned RAG)
- Rudimentary understanding of character relationships
- Primitive character details and traits
- Generate new summaries based on existing context (BUT NOT SAVE THEM LMAO (NOT ITERATIVE !!!))
- Unimplemented KV-cache memory management for swapping between context prefixes

## Installation
Just drop this bad boy into your text-generation-webui/extensions folder and a whole load of nothing is good to go! 100x of pointless added generation time at your fingertips! Have fun with that!

## Planned Features (more details in the old repo)
- Character detail mapping (each character gets their own list of traits and history)
- Real-time event tracking and short-term memory
- Context style detection and summarization
- Custom user instructions inserted anywhere along the line
- Full chat history management with vector search
- Dynamic subject updates and memory management
- Multi-model agentic system (far, FAR future)

## Configuration
- Active (iterative) vs Total (comprehensive) summarization modes
- Customizable summary weights and event breaks
- Timestamp options
- Memory management settings
- Subject mapping configuration

## State of this project
This commit is a pre-pre-pre-alpha commit! I haven't even verified if this version works, because there's essentially nothing implemented yet. I just wanna get this out there before the new year.

But, unlike last time, I have a full vision of how I want this agent system to work, and soon-to-be-loads of time on my hands. Also a cool 4x V100 SXM2 server.