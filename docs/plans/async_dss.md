# Async DSS: Decoupling State Updates from Response Latency

## Problem

In the current synchronous model, every turn blocks on DSS before the Story Master responds. A non-scene turn takes 55–118s (growing with run length), which is the entire user-facing latency. But most of that time is spent on DSS work the user doesn't need to wait for:

- Gate checks (has anything changed?)
- Branch updates (patch entry fields)
- Event tracking
- Element/character state maintenance

The LLM (Story Master) only needs the *current* DSS state to write a good response. It does not need DSS to have *processed* the latest exchange first.

## Core Insight

The LLM already understands message progression. Given the conversation history + current DSS state, it can write a coherent next turn even if DSS hasn't caught up. DSS is a *maintenance* system, not a *prerequisite* for generation.

This makes `CurrentScene.now` partially redundant in fast exchanges: the LLM can infer "the character is now at the door" from the last two messages without needing DSS to have updated `current_scene.now.participants`.

## Architecture

### Current (synchronous)

```
User message
  → DSS processes turn (55–118s)
    → DSS state updated
  → Story Master generates response (2–5s)
  → User sees response
```

Total user-facing latency: 57–123s

### Proposed (asynchronous)

```
User message
  → DSS state read (snapshot, <1ms)
  → Story Master generates response (2–5s)
  → User sees response
  → DSS processes turn in background (55–118s)
    → DSS state updated for next turn
```

Total user-facing latency: 2–5s

DSS runs as a background worker. Each turn's DSS processing is queued and executed sequentially (one at a time, preserving the current `--max-subject-workers 1` constraint).

## Queue vs Skip

Two strategies for when DSS falls behind:

### Strategy A: Queue (backlog)

DSS processes every turn, just asynchronously. If the user sends 3 messages while DSS is still working on turn N, turns N+1 and N+2 queue up. DSS works through the backlog sequentially.

- **Pros:** No data loss. Every exchange is captured. State is eventually consistent.
- **Cons:** Backlog grows if the user outpaces DSS. After 5 rapid messages, DSS is 5 turns behind. The Story Master is writing against increasingly stale state.
- **Mitigation:** If backlog exceeds a threshold (e.g., 3 turns), skip the oldest queued turns and process only the latest. The intermediate turns' state changes are likely subsumed by the latest turn.

### Strategy B: Skip (coalesce)

If a new user message arrives before DSS finishes the current turn, **discard** the in-progress DSS work and start fresh with the latest state. Only the most recent turn is processed.

- **Pros:** DSS is never more than 1 turn behind. State is always near-fresh. Simpler to reason about.
- **Cons:** Intermediate state changes are lost. If turn N established "character picked up the key" and turn N+2 is "character uses the key", the key-pickup is lost if turn N+1 was skipped.
- **Mitigation:** The LLM's conversation history carries the intermediate state. DSS can re-derive it from the full history on the next processing pass. This is less precise than turn-by-turn capture but functionally adequate for most cases.

### Recommendation

**Start with Strategy B (skip/coalesce).** It's simpler, keeps DSS close to real-time, and the conversation history provides a safety net. Add Strategy A (queue with threshold) later if analysis shows meaningful data loss from skipped turns.

Implementation: a single-slot queue. When a new turn arrives:
1. If DSS is idle → process immediately.
2. If DSS is busy → discard the current work, start fresh with the new turn's state.

The "discard" is cheap: DSS work is a sequence of LLM calls. Aborting the sequence and restarting from the latest state costs nothing (no partial writes are committed until the full turn completes).

## `CurrentScene.now` Redesign

In the synchronous model, `CurrentScene.now` is the authoritative "where is everyone right now" state. In the async model, it becomes a **last-confirmed** state: "where DSS last confirmed everyone was."

Two options:

1. **Keep as-is.** `CurrentScene.now` reflects the last DSS-processed state. The LLM sees it as "last confirmed position" and uses the conversation history for the current moment. This is the minimal change.

2. **Split into `now` (LLM-inferred) and `confirmed` (DSS-verified).** The LLM's response implicitly updates `now` (parsed from the response text in code, not LLM). DSS confirms or corrects it on its next pass. This is more work but gives a fresher position state.

Start with option 1. The conversation history is the source of truth for the current moment; `CurrentScene.now` is the source of truth for the last *verified* state.

## DSS Turn Selection

Not every turn needs full DSS processing. In the async model, we can be more aggressive about skipping:

| Turn type | DSS work needed? | Rationale |
|-----------|-----------------|-----------|
| New scene | **Yes** (full) | Scene boundaries are when state changes are large |
| Non-scene, first in sequence | **Yes** (light) | Establishes the exchange context |
| Non-scene, continuation | **Skip** | The LLM is continuing the same exchange; no new state to capture |
| Non-scene, after gap | **Yes** (light) | A gap implies time passed; state may have drifted |

"Light" processing = only `CurrentScene` + `SceneCharacter` updates (the 2 `always` triggers). No gate checks, no element/character branch updates. Those are deferred to the next scene boundary.

This would reduce non-scene DSS work from ~10 calls to ~2 calls, putting async DSS at ~10–15s per non-scene turn. Even if DSS falls behind by 1 turn, it catches up within a single user message interval.

## Queued Updates

On non-scene turns, queue general "which subject and why" as suggestions to the real update. Keep suggestions general (not too specific on which entity exactly), or give plaintext entity list (e.g. "characters: John, Mona, Leo"), though JSON would work as well (prefill is the main issue, not decode).

## Implementation Sketch

```python
# In the soak/production loop:

class AsyncDSS:
    def __init__(self, dss_engine, model):
        self.engine = dss_engine
        self.model = model
        self.current_state = snapshot()  # read-only for the LLM
        self.work_queue = None           # at most 1 item
        self.worker = None

    def on_user_message(self, message) -> str:
        """Called synchronously. Returns the Story Master's response."""
        # 1. Generate response with CURRENT (possibly stale) DSS state
        context = self.current_state.format_for_llm()
        response = self.model.generate(context + message)

        # 2. Queue DSS processing for this turn (async)
        self._enqueue_dss_work(message, response)

        return response

    def _enqueue_dss_work(self, message, response):
        """Replace any in-progress DSS work with the latest turn."""
        if self.worker and self.worker.active:
            self.worker.abort()  # discard in-progress work
        self.worker = Thread(
            target=self._process_dss_turn,
            args=(message, response)
        )
        self.worker.start()

    def _process_dss_turn(self, message, response):
        """Runs in background. Updates self.current_state on completion."""
        new_state = self.engine.process_turn(message, response)
        self.current_state = new_state  # atomic swap for next turn
```

Key properties:
- `self.current_state` is a snapshot. The LLM always reads the latest committed state.
- DSS work is aborted and restarted on each new message (Strategy B).
- The state swap is atomic: either the old state or the new state, never a partial update.
- DSS processing uses the same `--max-subject-workers 1` constraint (sequential LLM calls).

## What This Unlocks

1. **Fast RP styles.** D&D-style one-two-sentence exchanges become viable. User-facing latency drops from 55–118s to 2–5s.
2. **Longer soaks.** The 12h/150-turn wall disappears. DSS runs in the background; the soak pace is set by the LLM's response time, not DSS.
3. **Selective DSS.** Skip DSS for continuation turns. Only process scene boundaries and exchange starts. This compounds with the trigger reorg to cut DSS work by 60–80% on non-scene turns.
4. **Future: parallel DSS.** Once the async model is stable, DSS could run on a separate GPU or a smaller model, further decoupling it from the response path.

## Risks

- **Stale state drift.** If DSS is consistently 1–2 turns behind, the LLM writes against slightly stale state. Mitigated by: (a) the conversation history carrying the current context, (b) `CurrentScene.now` being "last confirmed" not "current", (c) scene boundaries forcing a full DSS pass.
- **Lost intermediate state.** Skipped turns lose their individual state captures. Mitigated by: (a) the conversation history as a recovery source, (b) scene-boundary DSS re-deriving state from the full history.
- **Complexity.** The async worker adds a thread, an abort mechanism, and a state-swap protocol. This is manageable but must be done carefully to avoid race conditions on `current_state`.

## Rollout

1. **Implement the async wrapper** around the existing DSS engine. No changes to the engine itself — just the call pattern (synchronous → background thread).
2. **Add the skip/coalesce logic.** Single-slot queue, abort-on-new-message.
3. **Reduce non-scene DSS work.** Light processing (2 calls) for continuation turns, full processing for scene boundaries.
4. **Test with a short soak.** 30 turns, compare state quality (judge scores, needle probes) against the synchronous baseline.
5. **If quality holds, run a long soak.** 300+ turns at the async pace.