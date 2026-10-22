# Context Injection Evaluation Plan

## Background

We've been testing the `highlight_relevant` (entity graph) and `Scene Focus` (duplicated `current_scene`) context injection blocks against a Greek mythology retelling soak run (Qwen3.8-27B finetuned, 141 turns). The initial hypothesis was that the papyrus — the object Leander is actively interacting with — was being "missed" by the model because it didn't appear in the injected top-10 entities.

## Key Finding: The "Papyrus Gap" Was a False Positive

The reasoning traces from the 6-mode test (turn 141) show the model **deliberately** excluding the papyrus:

> "The papyrus is gone. The passage bends. **What's new?**"

This is correct narrative flow. At turn 141, the papyrus was destroyed in the *previous* beat (turn 140). The model is writing the *next* beat, where the chisel is the new focus. The original soak run confirms this:

| Turn | Scene | Papyrus in NOW? | Original reply focus | Papyrus mentions |
|------|-------|:---------------:|---------------------|:----------------:|
| 139 | Following the papyrus designation | ✓ (held, unfolded) | Papyrus (pulled out, read, tucked) | 1 (central) |
| 140 | Destroys the papyrus in condensation | ✓ (dissolving) | Papyrus (fished out, dropped, watched) | 1 (central) |
| 141 | Drives chisel into obsidian floor | ✗ (gone) | Chisel (pulled out, driven, listened) | 0 (correct) |

The model was tracking narrative flow correctly all along. The "relevance gap" was my error in expecting a resolved plot object to dominate the next beat.

## What We Actually Need to Measure

The real question is: **does the context injection system (graph relevance, Scene Focus, highlight blocks) improve reply quality vs. baseline, when measured against the original run's output at the same turn?**

The baseline IS the original soak run. The test harness replays the same turn with modified context injection and we compare the new reply against the original.

## Proposed Test Matrix

### Primary test turn: **Turn 139**

Why 139:
- Papyrus is the **active** focus (held, being read)
- The original reply is papyrus-centric (the whole reply is about unfolding and reading it)
- If the context system works, the reply should maintain papyrus focus
- If the context system *hurts* (e.g., injects globally-important-but-absent entities that distract), the reply will drift

### Secondary test turns:
- **Turn 130** — papyrus in NOW, scene is "descends through slag cleft away from smithy" (transition beat, papyrus is the guiding object)
- **Turn 140** — papyrus destruction (the object is being actively destroyed; high emotional weight)
- **Turn 141** — chisel scene (control: papyrus should NOT appear; chisel should dominate)

### Modes to compare (per turn):
1. **Baseline** — no injection (current behavior, matches original run conditions)
2. **Graph top-12** — `highlight_relevant` with entity graph scores
3. **Scene Focus only** — the `current_scene2` block (deterministic, from `current_scene.json`)
4. **Graph + Scene Focus** — both blocks

### Comparison metrics:
- **Focus entity presence**: Does the reply mention the entity that `current_scene.now.who.elements` says is active? (binary, per turn)
- **Original similarity**: Qualitative — does the new reply cover the same beats as the original? (manual read)
- **Drift**: Does the reply introduce entities/actions NOT in the scene? (count of off-scene references)
- **Length ratio**: new_reply_len / original_reply_len (proxy for verbosity/terse-ness shift)

## Model Choice

**Current**: Qwen3.8-27B (agentic programming finetune) — creative writing is a secondary capability.

**Proposed**: Gemma-4-12B-StyleTune.i1-Q6_K.gguf — purpose-tuned for creative/roleplay writing. Better test subject for measuring whether the *context system* helps creative output, because the model's creative prior is strong enough that context injection is the limiting factor.

Sampling params for Gemma-4 (per TurboMind defaults for creative models):
- temperature: 0.7–0.8
- top_p: 0.9
- top_k: 40–50
- max_tokens: 1024

## Results: Turn 139 (Qwen3.8-27B)

Turn 139 is the **lowest-scoring turn in the entire run** (quality=2.5, style=3.0). The judge's verdict: "a near-verbatim transcription of the instructional block, failing to transform."

| Mode | chars | papyrus | Beat match to original |
|------|:-----:|:-------:|------------------------|
| **Original** | 783 | **1** | — (reference) |
| baseline (F,F,F) | 744 | 0 | Drifts to ember + counting drips |
| baseline+think (F,F,T) | 712 | 0 | Drifts to ember + static |
| highlight (F,T,F) | 664 | 0 | **Drifts to tile** (previous scene's object) |
| highlight+think (F,T,T) | 719 | 0 | Drifts to ember + counting |
| **director (T,F,F)** | 775 | **1** | **Near beat-for-beat match** |
| **director+highlight (T,T,F)** | 758 | **1** | **Near beat-for-beat match** |

**Findings:**

1. **`do_instr` (director mode) is the decisive factor.** 2/2 hit the papyrus with it, 0/4 without. The director instruction prompt gives the model explicit beat-by-beat guidance that the context injection blocks were supposed to provide.

2. **`highlight_relevant` (graph top-10) appears to hurt.** The highlight-only mode (F,T,F) focuses on the **tile** — the clay tile from the *previous* scene (turn 138). The graph ranks it high by global importance, but it's stale. Injecting it steers the model away from the current focus.

3. **The Scene Focus block's contribution is confounded** — it's always present in the context string, so we can't isolate its effect from this run. A control run (director mode with Scene Focus removed) would be needed.

4. **The original run's weakness at this turn is mechanical transcription**, not entity tracking. The model *does* track the papyrus correctly when given proper instruction. The quality problem is that the prose is a copy of the instruction block rather than a transformation of it.

## Results: Turn 39 (Gemma-4-12B-StyleTune)

Turn 39 is the worst **style** score in the run (2.0): "severe lack of originality, lifting almost entire sentences verbatim."

First, answering the question "is the judge too harsh?" — **no.** The original Qwen reply is a near-verbatim copy of the instruction block. Quantified via 5-gram overlap with the `INSTRUCTIONS TO FOLLOW` section:

| Reply | chars | 5-gram overlap vs instruction |
|-------|:-----:|:-----------------------------:|
| **Original (Qwen3.8-27B)** | 1269 | **78.0%** |
| baseline (F,F,F) | 699 | **0.0%** |
| baseline+think (F,F,T) | 757 | 0.0% |
| highlight (F,T,F) | 738 | 0.0% |
| highlight+think (F,T,T) | 615 | 0.0% |
| director (T,F,F) | 796 | 0.0% |
| director+highlight (T,T,F) | 726 | 0.0% |

**Finding 1: The verbatim-transcription failure is model-specific, not systemic.** All six Gemma-4-12B-StyleTune replies are 0% overlap — fully original prose that transforms the instruction beats. The creative-tuned model respects the "never quote or reuse any distinctive phrase verbatim" steering note that Qwen3.8-27B's agentic finetune ignores. This closes the open question: context injection does **not** fix the transcription problem (it wasn't a retrieval problem), but **model choice does**.

**Finding 2: At turn 39 the instruction block and the user input disagree about the action.**
- User input (Calliope): the fishmonger "jerked his chin toward a **stack of heavy crates**… a command silent and absolute."
- Instruction block (director): "grasp the nearest strip of raw **fish**… lift the fish to eye level…"

Director mode (T,*,*) follows the **instruction** (fish); the four non-director modes (F,*,*) follow the **user narrative** (crates). This is the same pattern as turn 139: `do_instr` is the switch that makes the model track the director's planned beat rather than improvising on Calliope's setup. Both readings are coherent, original prose — they just track different sources of truth.

| Mode | Follows | Sample line (original prose) |
|------|---------|------------------------------|
| baseline / +think | crates (user input) | "the ice coating the exterior melts instantly against your skin" |
| highlight / +think | crates (user input) | "a clean, mortal pain that anchors you more firmly than any divine blessing" |
| **director / +highlight** | fish (instruction) | "a far cry from the eternal, searing heat of the divine flame" |

**Conclusion for the model question:** Gemma-4-12B-StyleTune is the correct test subject for measuring the context system in isolation. Its creative prior is strong enough that the transcription failure disappears, so any remaining quality differences between modes are attributable to the context/instruction variables, not to model capability. All subsequent turns should be run with Gemma.

### Reasoning traces (turn 39)

Saved verbatim to `turn_039/gemma_reasoning_traces.md` (raw `reasoning_content` from the API, untransformed). Only the two thinking-enabled modes carried reasoning: baseline+think (8,888 chars) and highlight+think (15,822 chars); the director modes ran with thinking off, so there is no trace for the fish-following path.

Three findings, all visible only in the traces:

1. **The 0% overlap is *ignoring* the instruction block, not transforming it.** Both non-director traces build the draft from the **user input (crates)**, never engaging the director's **fish** instruction. baseline+think opens its scene-state with "A fishmonger has signaled him to move heavy crates" and drafts entirely around crates. The instruction block (fish) is simply not in the model's working set when `do_instr=False`. So "0% verbatim" overstates originality — the model isn't rewording the instruction, it's bypassing it.
2. **The StyleTune finetune runs an explicit constraint-satisfaction loop.** baseline+think shows a full draft → critique → revise cycle with a hard **forbidden-phrase self-check** ("something shifted", "a weight settled", "quietly [verb]", "It wasn't X, it was Y", em-dash overuse, negative parallelism). This is the mechanism that keeps the prose from the cliché patterns the judge penalizes — and it's a model property, not a context-system property.
3. **The `highlight_relevant` block is visible in the reasoning.** highlight+think lists "Key entities to include: … maybe a subtle nod to the **volcano's throat/caldera**" — `caldera` is a graph-ranked entity from the highlight block. So the block does enter the model's planning, but as a soft "maybe nod to" nudge, not a directive. The reply's "divine fire you carried to the volcano's throat" line is the (subtle) realization of that nudge.

**Trace quality note:** the traces are high-signal artifacts. From here on, each test turn records its reasoning traces (file path + load-bearing lines + quality assessment) in this doc.

## Results: Turns 20, 64, 79 (Gemma-4-12B-StyleTune)

All three confirm the same finding as turn 141: the judge's "prop continuity" criticisms are **false positives**.

| Turn | Judge flags | In `current_scene.who.elements`? | In entity graph? | In reply? | In reasoning? |
|------|-----------|:-------------------------------:|:----------------:|:---------:|:-------------:|
| 20 | Styx "silver and thick as spilled milk" | N/A (no current_scene.json) | Yes | No | **Yes** |
| 64 | borrowed sword present in market | **No** | Yes (imp 45) | No | **Yes** |
| 79 | borrowed sword not mentioned | **No** | Yes (imp 45) | No | **Yes** |

Key facts:

1. **"Borrowed sword" is a graph-tracked element, NOT a current-scene element.** Absent from `current_scene.who.elements` at turns 64 and 79. Scene Focus (which renders `current_scene`) never surfaces it. The model has no reason to write about a prop not in the active scene.

2. **The model explicitly acknowledges the props in reasoning and excludes them deliberately.** Turn 64: "the borrowed sword is present in the market but not being used." Turn 79: "I'm not mentioning the borrowed sword, the three-eyed crow, or the iron stand, as these are not part of this beat." Same "resolved-beat exclusion" pattern as turn 141 (papyrus).

3. **Turn 20's Styx imagery IS in the reasoning.** The trace writes "the silver river Styx, thick as spilled milk" — the exact line the judge says is missing. The model considered it and chose not to use it.

4. **The Scene Focus block was NOT in these runs.** Only turns 139/141 have `current_scene2` in their templates. These results do not measure Scene Focus's effect.

5. **Empty-reply defect (2/3 turns).** Turn 20 `highlight+think` (F,T,T) and turn 79 `baseline+think` (F,F,T) returned 0-char replies despite 15-16k chars of reasoning. Known `last_content` empty-content issue.

**Per-turn detail:** `turn_020/gemma_eval.md`, `turn_064/gemma_eval.md`, `turn_079/gemma_eval.md`.
**Reasoning traces:** `turn_020/gemma_reasoning_traces.md`, `turn_064/gemma_reasoning_traces.md`, `turn_079/gemma_reasoning_traces.md`.

**Assessment:** The "prop continuity" weakness in `final_overview.json` is a **judge artifact**, not a generation defect. The model tracks the props in the graph, acknowledges them in reasoning, and correctly excludes them from the reply. The judge, which only sees the final prose, penalizes the omission. Third instance of the same pattern (after turn 141's papyrus).

## Recommended Test Turns (from `final_overview.json` judge records)

The judge identified three recurring weakness patterns across the run. Each maps to a different context injection hypothesis:

### Pattern 1: Prop continuity (dropped items)

The borrowed sword, clay pot (hairline crack), and red sash are repeatedly dropped from Leander's physical inventory. These are exactly the entities that `current_scene.now.who.elements` tracks.

| Turn | Quality | Missing prop | Why test |
|------|:-------:|-------------|---------|
| **64** | 4.0 | borrowed sword | "not mentioned in this paragraph, though it was present" — sword keeps vanishing |
| **79** | 4.0 | bronze sword | Same sword-continuity issue in a different scene |
| **34** | 4.0 | clay pot | "not present in this scene (already returned the fire...)" — ambiguous: is the pot *supposed* to be here? |

**Hypothesis:** Scene Focus (which renders `current_scene.now.who.elements`) should make dropped props visible to the model, reducing the "frequently dropping established items" weakness.

### Pattern 2: Mechanical / verbatim transcription

| Turn | Quality | Style | Issue |
|------|:-------:|:-----:|-------|
| **139** | 2.5 | 3.0 | "near-verbatim transcription of the instructional block" (already tested) |
| **39** | 3.5 | **2.0** | "severe lack of originality, lifting almost entire sentences verbatim" |
| **10** | 3.0 | 3.0 | "follows the physical actions requested in the prompt" — clay pot planting missed |

**Hypothesis:** These turns fail because the instruction block is too prescriptive and the model copies it. Context injection won't fix this — it's a model/prompt-engineering problem, not a retrieval problem. Testing these turns would confirm that the context system is not the bottleneck for this failure mode.

### Pattern 3: Missed sensory detail

| Turn | Quality | Missed detail |
|------|:-------:|--------------|
| **20** | 3.5 | "Styx's water is silver and thick as spilled milk; a drop of it remembers every foot" |
| **35** | 3.5 | "black-sailed boat is not mentioned or echoed" |

**Hypothesis:** These are world-building details that should be in the entity graph (as elements) with high importance. If they're in the graph but not in the top-10, the `highlight_relevant` block is too narrow. If they're not in the graph at all, the extraction pipeline is missing them.

### Recommended test order

1. **Turn 39** — worst style score (2.0), confirms whether the transcription problem is fixable by context injection (expected: no, it's a prompt problem)
2. **Turn 64** — sword continuity, tests whether Scene Focus recovers dropped props
3. **Turn 20** — Styx water sensory detail, tests whether the graph captures and surfaces world-building elements
4. **Turn 79** — second sword-continuity data point (confirm turn 64 finding generalizes)

## Implementation Steps

1. ~~Verify turn 139 works~~ ✅
2. ~~Run 4-mode comparison at turn 139~~ ✅ (results above)
3. **Run turns 39, 64, 20, 79** with Qwen3.8-27B (6-mode matrix)
4. **Swap to Gemma-4-12B-StyleTune** and re-run the same turns
5. **Compare against original replies** (saved in `turn_NNN/reply.txt`)
6. **Document results** in this file

## Open Questions

- Does `Scene Focus` recover dropped props (sword, sash, clay pot)? The turn 64/79 tests will answer this.
- Does the graph's `highlight_relevant` block *hurt* by injecting stale, globally-important entities? Turn 139 suggests yes (tile displaced papyrus).
- Is the CE (cross-encoder) worth keeping in the injection path? Its semantic-similarity signal doesn't align with scene presence.
- Can context injection fix the verbatim-transcription problem (turns 39, 139)? Expected answer: no — that's a prompt/model problem, not a retrieval problem.

## File References

- Test harness: `extensions/dayna_ss/tests/test_context_injection.py`
- Engine: `extensions/dayna_ss/agents/summarizer/context_engine.py`
- Formatted data: `extensions/dayna_ss/agents/formatted_data.py`
- Run data: `extensions/dayna_ss/tests/runs/greek_mythology_retelling__2e7575de/`
- Original replies: `turn_NNN/reply.txt`
- Original user input: `turn_NNN/user.txt`
- Scene state: `turn_NNN/state_snapshot/current_scene.json`
- Format templates: `turn_NNN/state_snapshot/format_templates.json`