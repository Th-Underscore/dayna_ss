# `--max-update-history` vs. Engine Message History Management

**Investigation date:** 2026-07-19
**Scope:** `extensions/dayna_ss/tests/long_horizon_soak.py` × `extensions/dayna_ss/agents/summarizer/context_engine.py`

---

## 1. What `--max-update-history` does in the soak

### Definition

`long_horizon_soak.py` line 212:

```python
p.add_argument("--max-update-history", type=int, default=8,
               help="max engine internal-history exchanges sent per DataSummarizer call (0 = unlimited); "
                    "composes with the rolling window as sent = history[-min(rolling_window, this):]")
```

### Storage

Stored on `LocalModel` as `self.max_update_history_exchanges` (line 344/357).

### Where it is applied

`LocalModel._complete()` (line 427):

```python
bound = self.max_update_history_exchanges if bound_history else 0
internal_msgs = self._engine_internal_messages(state, max_exchanges=bound)
```

`bound_history` is `True` for DataSummarizer call paths, `False` for reply generation:

| Method | `bound_history` | Context |
|---|---|---|
| `generate_with_sse` (line 566) | `True` | DataSummarizer per-entry updates |
| `generate_using_tgwui` (line 575) | `True` | DataSummarizer archives, discovery, digest |
| `generate_with_streaming` (line 546) | `False` (default) | Reply generation |
| `complete` (line 585) | `False` (explicit) | Reply generation |

So the cap **only applies to DataSummarizer calls**, never to the reply-generation path.

### The bounding logic — `_engine_internal_messages` (line 377–419)

Reads `state['history']['internal']` (the engine's already-formatted artificial history) and, when `max_exchanges > 0`:

1. **Preserves** every entry whose first element starts with `"Summaries of earlier messages"` — these are the sticky-roll summary pairs the engine injects.
2. **Bounds** all other entries to the last `max_exchanges`:

```python
internal = summaries + rest[-max_exchanges:]
```

The docstring (line 388–394) states the rationale explicitly:

> The DataSummarizer's per-entry update prompts only reference "the latest exchange(s)", so on the summarization/update call paths the old exchanges are redundant with the engine's own scene-bounded window — trimming them keeps the per-call prompt (and thus prefill cost) from growing with the whole story, which is the O(N²) blowup in long runs.

---

## 2. What the engine does for message history management

All logic lives in `agents/summarizer/context_engine.py`.

### 2a. The engine builds its own artificial history

`retrieve_and_format_context()` (line 228) produces `custom_state["history"]["internal"]` — a **synthetic** history (DAYNA Mode) that replaces the raw chat history. This is what the soak's `LocalModel` reads.

The content of this artificial history depends on `message_mode`:

#### Enumerated mode (default, `message_mode="enumerated"`)

For each subject in `context_order`, the engine appends a Q&A pair:
```
[formatted prompt for subject, formatted subject data]
```
Then one exchanges pair (line 415–416):
```
["What were the last {N} exchanges (pairs of messages)?", formatted_dialogue]
```
Then one analysis marker (line 423–428):
```
["Analyze all of the above information. Confirm when your analysis is complete.", "Analysis complete."]
```

The exchanges pair's `N` is `last_x`, computed by `_scene_dialogue_window()`.

#### Rolling mode (`message_mode="rolling"`)

Raw user/assistant pairs from the scene-bounded window (line 411–413):
```python
for exchange in history[-last_x:]:
    custom_history.append([str(exchange[0] or ""), str(exchange[1] or "")])
```

Plus, when `rolling_summaries > 0`, a sticky-roll summaries pair prepended ahead of the raw pairs (line 403–408):
```
["Summaries of earlier messages (messages {start}-{end-1}, before the most recent window):", formatted_summaries]
```

### 2b. The scene-bounded window — `_scene_dialogue_window` (line 811–866)

```python
last_x_max = int(kwargs.get("last_x_max", 8))   # default 8 (enumerated)
last_x_min = int(kwargs.get("last_x_min", 2))   # default 2
```

In rolling mode, `last_x_max` is overridden by `rolling_window` (line 464–465, default 6):
```python
if self.config.get("message_mode", "enumerated") == "rolling":
    kwargs.setdefault("last_x_max", int(self.config.get("rolling_window", 6) or 6))
```

The window is **scene-bounded**: it cannot cross a scene boundary. If the current scene started at exchange `scene_start_exchange`, the window is `len(history) - scene_start_exchange`, clamped to `[last_x_min, last_x_max]`.

### 2c. The sticky-roll summaries (rolling mode only)

When `rolling_summaries > 0`, `_rolling_summary_window()` (line 717–749) computes how far back to reach for message summaries:

- `end` = where the raw rolling window begins (message index)
- `start` = the further of:
  - `floor` messages back (`floor = rolling_summaries` config, in message exchanges)
  - the start of the last `ROLLING_SUMMARY_SCENE_ROLL = 5` scenes
- The computed `start` is **locked** to the current scene key and only recomputed when the scene changes (sticky lock for prefix caching).

Summaries are loaded from the chunker's index (`_load_summary_nodes`, line 751–779) and formatted by `_format_rolling_summaries` (line 781–810), which caps each entry at `rolling_summary_max_chars` (default 280 chars) at a word boundary.

---

## 3. Do they override, duplicate, or conflict?

**They are independent, stacked layers.** The engine windows first; the soak's cap is a second, coarser cut applied to the engine's already-windowed output.

### The composition

```
raw chat history
    │
    ▼  engine: _scene_dialogue_window()
scene-bounded window (≤ last_x_max, ≤ rolling_window in rolling mode)
    │
    ▼  engine: retrieve_and_format_context()
custom_state["history"]["internal"]  (artificial history)
    │
    ▼  soak: _engine_internal_messages(max_exchanges=8)
bounded internal history  (DataSummarizer calls only)
    │
    ▼
LocalModel._complete()  →  LLM call
```

### When the soak's cap actually binds

| Mode | Engine window size | Soak cap (default 8) | Cap binds? |
|---|---|---|---|
| Enumerated | 1 Q&A pair per subject + 1 exchanges pair + 1 marker | 8 entries | **Yes** — when context_order has > 6 subjects, the total exceeds 8 and the oldest context pairs are cut |
| Rolling, `rolling_summaries=0` (default) | ≤ `rolling_window` (default 6) raw pairs | 8 entries | **No** — 6 < 8, cap is a no-op |
| Rolling, `rolling_summaries>0` | ≤ 6 raw pairs + 1 summaries pair | 8 entries | **No** — summaries pair is always preserved; 6 raw < 8 |
| Rolling, `rolling_window=10` | ≤ 10 raw pairs | 8 entries | **Yes** — 2 oldest raw pairs are cut |

### The critical design invariant

The "Summaries of earlier messages" pair is **never cut** by the soak's cap. This is the engine's mechanism for carrying older context in summarized form. When the cap does cut raw pairs, the summaries pair (if present) preserves the information that the raw pairs carried.

---

## 4. Practical impact — is information lost?

### Case A: Default config (enumerated mode, `max_update_history=8`)

The engine produces one Q&A pair per subject in `context_order` (characters, groups, elements, events, chapters, arcs, lines — up to 7) + 1 exchanges pair + 1 marker = up to 9 entries. The cap of 8 cuts the oldest context pair.

**Information lost:** The oldest subject's formatted data (e.g., the first subject in `context_order`) is absent from the DataSummarizer's per-entry update prompt. In practice this is low-impact because:
- The DataSummarizer's update prompts are per-subject: each call only needs the data for the subject being updated, not all subjects.
- The engine's retrieval layer (`StoryContextRetriever`) already selects which subjects are relevant to the current turn; the cap only affects the prompt framing, not the retrieval.
- The reply-generation path (`bound_history=False`) is unaffected.

### Case B: Rolling mode, `rolling_summaries=0` (default)

The engine drops out-of-window messages entirely (no summaries pair). The soak's cap of 8 is a no-op because `rolling_window=6 < 8`.

**Information lost:** Messages outside the 6-exchange rolling window are gone from the DataSummarizer's prompt. This is **by design** — the docstring says the update prompts only need "the latest exchange(s)". The engine's subject-level summarization (DataSummarizer's per-subject `generate` calls) is the mechanism that persists older message content into the subject JSONs, so the message-level detail is not needed in the update prompt.

### Case C: Rolling mode, `rolling_summaries>0`

The engine injects the sticky-roll summaries pair. The soak's cap preserves this pair and bounds the raw pairs to 8.

**Information lost:** Raw pairs beyond the 8-exchange cap are cut, but the summaries pair carries their summarized content. The per-entry cap of 280 chars per summary means very long messages get truncated in the summaries, but this is a deliberate tradeoff (the docstring at line 784–787 explains: "the grounding beats survive a per-entry cap because entity current-state lines already carry the detail").

### Case D: Rolling mode, `rolling_window > 8`

The engine's window exceeds the soak's cap. The oldest raw pairs are cut.

**Information lost:** Same as Case C if `rolling_summaries > 0` (summaries pair carries the cut content). If `rolling_summaries=0`, the cut raw pairs' content is **genuinely lost** from the DataSummarizer's prompt — no summary carries it. This is the one configuration where the soak's cap causes net information loss that the engine's own mechanisms don't compensate for.

### Summary table

| Config | Soak cap binds? | Net info loss? |
|---|---|---|
| Enumerated, default | Yes (oldest context pair) | Minimal — per-subject updates don't need all subjects |
| Rolling, `rolling_summaries=0`, `rolling_window ≤ 8` | No | Engine's own windowing loss (by design) |
| Rolling, `rolling_summaries>0`, `rolling_window ≤ 8` | No | Engine's summaries carry the older context |
| Rolling, `rolling_summaries=0`, `rolling_window > 8` | **Yes** | **Genuine loss** — cut raw pairs have no summary substitute |
| Rolling, `rolling_summaries>0`, `rolling_window > 8` | Yes | Summaries pair carries the cut content (280-char cap per entry) |

---

## 5. Residual risks

1. **`rolling_summaries=0` + `rolling_window > 8`** is the only configuration where the soak's cap causes uncompensated information loss. The default config (`rolling_window=6`) avoids this. If a user raises `rolling_window` above 8 without also enabling `rolling_summaries`, the cap silently drops raw pairs whose content has no summary substitute.

2. **The cap is applied uniformly** to all DataSummarizer call paths (updates, discovery, archives, digest). A discovery call that needs to see more context to decide "is this a new entity?" would benefit from a larger window than a per-entry update call that only needs the latest exchange. The current design doesn't differentiate.

3. **The enumerated-mode cap cuts context pairs, not message pairs.** In enumerated mode, the "exchanges" are bundled into a single Q&A pair, so the cap's effect is on the number of subject context pairs visible, not on message history depth. This is a different kind of cut than in rolling mode and is less obviously safe.