# Soak Run Analysis Method

How to analyze a completed long-horizon soak run. Replicable for any run.

## Inputs

- **Log file:** `benchmarks/runs/logs/<run_name>.log` (e.g. `scifi_heist_gemma4_s200.log`)
- **Run directory:** `benchmarks/runs/<spreadsheet>__<hash>/` (e.g. `sci_fi_heist__c05349c6/`)
  - Verify latest via ``ls -ltd runs/*__*/ | head -n 3; ls -ltd runs/logs/ | head -n 3``
  - `turn_NNN/reply.txt` — model reply per turn
  - `turn_NNN/instr_prompt.txt` — instruction block per turn
  - `turn_NNN/result.json` — judge verdict (score, notes, probes)
  - `turn_NNN/state_snapshot/` — entity state (characters.json, elements.json, events.json, knowledge.json, etc.)
  - `turn_NNN/state_snapshot/current_scene.json` — scene state snapshot
  - `rolling_overview_NNN.json` — rolling summary every N turns
  - `final_overview.json` — final LLM-generated overview
- **Manifest:** `benchmarks/runs/<spreadsheet>__<hash>/manifest.json`

## Analysis Scripts

Two scripts automate the bulk of the analysis. Run both before deep-diving.

### `deep_timing.py`

Per-subject, per-action timing from `dss_calls.jsonl`.

```bash
python3 benchmarks/deep_timing.py <run_dir>            # full report
python3 benchmarks/deep_timing.py <run_dir> --slices 5 # + per-slice breakdowns
python3 benchmarks/deep_timing.py <run_dir> --turns 1,22  # turn range filter
```

`--slices N` divides the turn range into N equal slices and prints per-slice
subject and action breakdowns (calls, time, %, avg/call, avg in-tok) plus
overall DSS/writer stats per slice. Replaces the old 6-command `--turns` approach.

### `analyze_run.py`

State evolution + prose quality from `turn_NNN/*` files. Writes a Markdown report.

```bash
python3 benchmarks/analyze_run.py <run_dir>                          # both sections
python3 benchmarks/analyze_run.py <run_dir> --sections state         # state only
python3 benchmarks/analyze_run.py <run_dir> --sections prose         # prose only
python3 benchmarks/analyze_run.py <run_dir> --out PATH --sample 15   # custom output
```

**State section:** per-turn tracking table, turns-per-scene, entity growth curves,
character entry/exit, scene transitions, rolling overviews, synopsis growth, anomalies.

**Prose section:** reply length distribution, 5-gram overlap with outlier detection,
flagged turns (lowest judge scores), voice consistency, note adherence.

## Workflow

### Step 1: Parent quick triage

Read `final_overview.json` and sample a few `result.json` files to get:
- Final score and the overview's stated weaknesses
- Score distribution (mean, floor, % at 4.5+)
- Which turns are flagged lowest
- Obvious anomalies (empty replies, score jumps)

Write an **interim findings doc** with these observations. This gives the user
an early read while the deeper analysis runs.

### Step 2: Run analysis scripts + targeted deep-dives

Run both scripts (they produce the bulk of the analysis):

```bash
python3 benchmarks/deep_timing.py <run_dir> --slices 5
python3 benchmarks/analyze_run.py <run_dir> --out benchmarks/docs/<run_id>_analysis.md
```

Then the parent reads the script outputs and does **targeted deep-dives** on
what they flag. Typically 2–4 file reads per axis:

**Timing:**
- Read `deep_timing.py` output — identify which subjects/actions are growing
- If cloud failures are present: grep the log for `FAILED`, `HTTP 503`, gap patterns
- Compare per-slice token growth to identify which subjects are bloating

**State:**
- Read `analyze_run.py` state section — check the anomaly flags
- For each flagged item: read the specific `turn_NNN/state_snapshot/<subject>.json`
  to verify (entity drops, empty files, missing fields)
- Cross-reference scene transitions against `current_scene.json`

**Prose:**
- Read `analyze_run.py` prose section — check outliers and flagged turns
- For 1–2 outlier turns: read the `turn_NNN/reply.txt` to verify the n-gram
  overlap is real transcription vs. coincidental phrasing
- Check voice consistency anomalies: is the density shift a real voice drift?

**Deliverable:** `<run_id>_analysis.md` (from `analyze_run.py`) + deep-dive
findings appended to the interim doc.

### Step 3: Consolidate

Merge findings into the interim doc and produce the final assessment with:
- What's working well
- What's broken (systemic vs run-specific)
- Judge artifacts (false positives)
- Priority-ordered fix list

### Optional: Subagents for very long runs

For runs > 100 turns, the script outputs are large enough that dispatching
parallel subagents (one per axis) can save wall time and keep the parent's
context clean. The critical constraint still applies:
> Never read a large file into LLM context in one operation. Write a Python
> script that does the mechanical I/O and writes the deliverable `.md` to disk.
> The subagent's context should only see the script's small stdout.

## Timing Interpretation

### What the numbers mean

`deep_timing.py` reads `dss_calls.jsonl` — one record per LLM call. Each record has:
- `turn` — the soak turn number
- `subject` — the phase_id path (e.g. `events.scenes`, `elements.entries.foo`)
- `action` — the DSS action (`apply_updates`, `add_new`, `perform_update`, etc.)
- `total` — wall time for that single call (seconds)
- `ttft` — time to first token (prefill phase, seconds)
- `in` — input tokens for that call
- `out` — output tokens for that call
- `cached` — cached tokens (prefix cache hits; 0 if not logged)

### DSS vs Writer

- **DSS** (DataSummarizer) = the local 26B model doing structured JSON summarization. This IS the "engine" — it produces the state data (characters, elements, events, etc.) that feeds the writer.
- **Writer** = the local gemma-4 model generating the in-character reply. Separate model, separate calls.

The "DSS share" percentage tells you how much of total local LLM time goes to summarization vs. response generation.

### Per-call vs per-turn

- **`Avg in`** in the subject/action tables is the mean input tokens **per call**, not per turn. A turn typically has 5-30 DSS calls, so per-turn input is the sum across those calls.
- **`Avg/call`** is mean wall time per individual LLM call (typically 3-30s).
- **Per-turn total** = sum of all DSS + writer call times in that turn. See the "PER-TURN DETAIL" section for individual turn totals.

### Slices

`--slices N` divides the turn range into N equal groups. The per-slice tables show **cumulative** totals across all turns in that slice — NOT per-turn values. A slice showing "12,918s" for events across 36 turns means ~358s/turn for that subject in that slice, not 12,918s per turn.

### What does NOT appear in dss_calls.jsonl

- **Rolling overviews** — cloud-side (gemini), not local. Never in the DSS call log.
- **Judge verdicts** — cloud-side. Not in the DSS call log.
- **Guide/replan calls** — cloud-side. Not in the DSS call log.
- **Cloud failures/cooldowns** — not LLM calls at all; they appear only in the soak log.

The DSS call log captures ONLY local model invocations (DSS + writer). Cloud-side latency is visible only in the soak log's `done: turn N in Xs (cloud Ys, local Zs)` lines.

### Prefix cache

If `cached` is 0 across all records, the run's vLLM server did not report `cached_tokens` in its responses (older server version or config). The TTFT-based estimation fallback is then used. A healthy prefix-cache setup shows 40-70% cached on repeated subjects.

### Scene turns vs non-scene turns

Scene turns (>20 DSS calls) are turns where a scene transition was detected — the DSS processes all subjects (events, elements, characters, groups, etc.). Non-scene turns (≤20 calls) typically only touch `current_scene`. This is why the subject breakdown differs dramatically between the two groups.

## Prompt-Level Timing Analysis

When a run's scene turns are too slow, the next question is: **which prompts are causing the delay?**
This section documents the method for answering that from `dss_calls.jsonl` + schema templates.

### Step 1: Identify worst turns

Use `deep_timing.py --turns N,N` for specific turns to see the per-call breakdown. The worst
turns are typically the ones with the most DSS calls (scene turns that trigger cascades):

```bash
python3 benchmarks/deep_timing.py <run_dir> --turns 60,60
python3 benchmarks/deep_timing.py <run_dir> --turns 84,84
```

This shows which subjects/actions fired and their individual call times.

### Step 2: Per-subject:action time accumulation

Parse `dss_calls.jsonl` to get total time per `subject:action` pair across the whole run.
This reveals which prompt templates are the biggest cumulative cost drivers:

```python
# Pseudocode: group by subject:action, sum total time
# Sort descending, take top 10
# For each: show call count, total time, avg time, avg input tokens
```

Key columns to track: `calls`, `total_s`, `avg_s`, `avg_in` (mean input tokens).

### Step 3: Input token growth per subject:action

For the top offenders, track `in` (input tokens) across turns. This reveals the growth
mechanism. Two patterns:

- **Linear growth** (e.g., `events.beats:apply_updates`: 14K → 53K over 90 turns):
  The prompt includes a full-data dump of a growing branch. Map to the template variable:
  `branch_update_prompt_template` has `{{ value | tojsonc }}` which serializes the entire
  branch's entries. At ~940 tokens/entry, 42 beats = ~39K tokens just for the data section.

- **Sublinear growth** (e.g., `events.beats:add_new`: 14K → 30K):
  The prompt includes a name list (`{{ branch_list }}`) that grows with entry count but
  carries no per-entry data. Slower growth, same root cause (branch expansion).

- **Flat** (e.g., `message_summary:summarize_msg_1`: 13K → 29K):
  The prompt is bounded by the writer's conversation context, not by entity state.

### Step 4: Map to schema templates

For each growing subject:action, find the corresponding prompt template in the schema:

1. Open `user_data/example/schemas/subjects_schema_noschema.json` (the run's actual schema)
2. Navigate to `definitions.<SubjectType>.defaults.<action>_prompt_template`
3. Identify which template variable is the growth driver (usually `{{ value | tojsonc }}`
   for `apply_updates`, `{{ branch_list }}` for `add_new_query`)
4. Check the static template size vs the dynamic data size to quantify the signal-to-noise ratio

### Step 5: Quantify the fan-out inefficiency

The core metric: **prompt size / useful signal size**.

- `apply_updates` sends the full branch (O(N) entries) but the LLM only needs to diff
  against 1-2 entries affected by the latest exchange (O(1) useful signal).
- As N grows, the model spends most of its prefill tokens reading data it will ignore.
- The fix direction is to narrow the prompt to only the entries that could be affected
  (e.g., entries whose participants overlap with the current scene's cast).

### Quick one-liner for the top offenders

```bash
# Top 10 subject:action by total time, with input growth
python3 -c "
import json; from collections import defaultdict
calls = [json.loads(l) for l in open('dss_calls.jsonl')]
sa = defaultdict(lambda: {'t':0,'n':0,'in_first':0,'in_last':0,'turns':[]})
for c in sorted(calls, key=lambda x: x['turn']):
    k = f\"{c['subject']}:{c['action']}\"
    sa[k]['t'] += c['total']; sa[k]['n'] += 1
    if not sa[k]['turns']: sa[k]['in_first'] = c['in']
    sa[k]['in_last'] = c['in']; sa[k]['turns'].append(c['turn'])
for k, v in sorted(sa.items(), key=lambda x: x[1]['t'], reverse=True)[:10]:
    print(f\"{k:55} {v['n']:3d} {v['t']:7.0f}s {v['in_first']:6d}→{v['in_last']:6d}\")
"
```

## Log Format Reference

Key log line patterns (regex):
- Turn boundary: `=== TURN (\d+) ===`
- Local LLM call: `local-call.*role=(\w+).*total=([\d.]+)s.*in=(\d+).*out=(\d+)`
- Cloud LLM call: `cloud-call.*role=(\w+).*total=([\d.]+)s`
- Cloud failure: `cloud-call.*role=(\w+).*FAILED`
- DSS tool call: `dss_tool.*tool=(\w+)`
- Scene transition: `scene_number.*(\d+)`

Adjust regexes to match the actual log format — check a sample of lines first.

## File Naming Convention

All deliverables go to `benchmarks/docs/`:
- `<run_id>_analysis.md` (from `analyze_run.py` — state + prose tables)
- `<run_id>_interim.md` (consolidated: triage + script findings + deep-dives + final assessment)

Where `<run_id>` = `<spreadsheet>__<hash>` (e.g. `scifi_heist_c05349c6`).

For subagent-dispatched runs (100+ turns), each subagent writes its own
`<run_id>_<axis>.md` (e.g. `_timing`, `_context`, `_prose`) as before.

## Mid-Run Reviews

Checks to perform at natural pause points (reboot, timeout kill, user-initiated stop) to verify the run is healthy before resuming.

### Data structure integrity

- **`events.json` top-level keys** — must be exactly `['past', 'scenes', 'chapters', 'beats']`. Any other key (especially `'events'`) indicates the LLM echoed the branch name as a parent wrapper (the bug the prompt fixes target).
- **No phantom nesting** — no entry should contain a nested object with the same name as its parent key.

### Subject population

- **Characters** — count and per-character populated field count. Expect 5+ named characters from the spreadsheet, each with 7+ populated fields (description, biography, traits, attributes, voice, quirks, fears, preferences, importance, etc.).
- **Elements** — total count. Track growth across checkpoints (e.g. 72 at turn 20 → 123 at turn 188 is ~0.3/turn). Flag if growth stalls (possible LLM stopping from creating new elements) or spikes (possible duplicate creation).
- **Groups** — count. Usually small (1-6).
- **Beats/Scenes/Chapters/Past** — counts in `events.json`. Beats should grow monotonically. Scenes should be in a small window (3-7 active). Chapters increment at archival boundaries. Past accumulates archived chapters.
- **Chapters** — counts in `events.json`. They should correlate 1-to-1 with chapter turns.
- **Arcs** — counts in `arcs.json`. They should correlate 1-to-1 with arc turns.

### Narrative coherence

- **Rolling overviews** — `rolling_overview_NNN.json` at the run root (every `--overview-every` turns). Check the latest: `overall_score`, `trajectory` (should be `stable` or `improving`), `verdict`, `memory_fidelity`. A declining trajectory or dropping score across consecutive overviews is a warning.
- **Synopsis** — `general_info.json` → `synopsis`. Should grow and stay coherent. Check length and that it references recent story developments, not just the opening.
- **Current scene** — `current_scene.json`. Scene number should be monotonically increasing. `now.who` should list characters that exist in `characters.json`.

### Judge/audit health

- **Judge scores** — from `result.json` → `judge` field (only present on judged turns, every `--judge-every` turns). Track `style_score`, `quality_score`, `memory_fidelity`. Flag: any score < 4.0, a declining trend, or a spike in low scores.
- **Cloud call counts** — sum `cloud_usage` + `judge_usage` + `auditor_usage` across all `result.json` files. All three should be non-zero. If judge or auditor calls are 0, the background workers may not be running.

### Error check

- **Log file** — grep for `Traceback`, `FAILED`, `Error` in the run log. A few HTTP 502s from the cloud endpoint are normal (transient). Repeated failures or `TypeError`/`AttributeError` tracebacks indicate code bugs.
- **All errors including validation** — also grep for `validation`, `ValidationError`, `Failed to parse`, `no matching step`, and ANSI-coded error lines (lines containing `\x1b[31m` red or `\x1b[33m` yellow that start with an error indicator). Validation errors are often the most important signal (schema drift, LLM producing wrong shapes) and are easy to miss if you only grep for "Error".
- **Count and categorize** — don't just flag that errors exist; count them by type and identify which turns/subjects they affect. A pattern (e.g. 12 `no matching step` errors on `current_scene.now.why`) is more actionable than a single traceback.

### Resume readiness

Before re-launching after a mid-run stop:
1. Confirm the last turn's `result.json` is complete (has `reply`, `judge` if expected, `cloud_usage`).
2. Confirm no partial state snapshot (all JSON files parse cleanly).
3. Remove any temporary tasks from the launch script (e.g. parallel quantization jobs).
4. Re-launch via `gpu-quiesce`.

## Temporary Checks

Ephemeral, run-specific checks tied to what a particular run is testing. Unlike the
mid-run review (general health) or the 3-axis subagent analysis (general quality),
these verify a specific feature landed correctly. Remove the section once the run attains
excellent quality for this standard.

### Validated checks

These checks were added for specific features and are now confirmed working.
Keep them for regression monitoring but they are no longer the primary focus.

- **Knowledge population** — 29 entries in s562. Clean vocab. ✅
- **CHECK_ARCHIVE trigger system** — Scenes 1:1 with scene turns, chapters
  increment at boundaries, synopsis updates. ✅
- **Format template restructure** — No `_history`/`_message_node` in rendered
  prompts, no bare `N/A` artifacts, path markers stripped. ✅
- **`stamp_end_message_nodes` guard** — 15/16 unique `_message_node` values
  (1 shared due to placeholder scene). ✅
- **PATH SCOPE integrity** — Clean top-level keys, no corruption. ✅
- **Schema-agnostic retrieval** — `subjects` dict, 0 errors, all subjects
  populated. ✅
- **Chapter LLM gate check** — Signal-detection framing (s563: 11 chapters,
  proper titles). ✅

### Offscreen events (renamed from PastEvents)

The `events.past` field is now `events.offscreen`. Verify:

- **`events.offscreen` key present** in `events.json` (not `events.past`).
- **Offscreen entries appear** — at least 1 entry by mid-run (backstory
  events from late-introduced characters, timeskips).
- **Offscreen nodes in entity graph** — `offscreen:` prefix nodes exist.

### Beat nodes in entity graph

Beats were previously absent from the entity graph. Verify:

- **`beat:` nodes** — at least 1 beat node in `entity_graph.json` by turn 5+.
- **Beat count correlates** — graph beat nodes ≈ `events.beats` entry count.

### Polymorphic target_type

`CharacterMilestone.target_type` is now a list (e.g.
`["offscreen","beat","scene","chapter","arc"]`). Verify:

- **No crashes** — 0 `AttributeError` in the run log (the list-vs-string
  `.lower()`/`.capitalize()`/`+ "s"` crashes are the regression risk).
- **No `no matching step`** errors related to milestone/relationship fields.

### Graph-relevance filtering in apply_updates

`_create_update_prompt` now filters `current_value` to only entries in the
entity graph's relevance set. Verify:

- **Filter activates** — in a long run (50+ turns), the input token count for
  `apply_updates` on growing branches (beats, scenes) should plateau or grow
  sublinearly, NOT linearly with entry count. Compare `avg_in` across slices:
  if it stays roughly flat while entry count grows, the filter is working.
- **No data loss** — filtered entries should still be present in the state
  files (the filter only narrows the LLM prompt, not the data).

### `select_entries_to_update` filtering

Events branches (StoryEvents, OffscreenEvents, Scenes, Chapters) now use
`select_entries_to_update` to cap per-entry `apply_updates` to the 2-3 entries
the LLM selects, instead of recursing into all entries.

- **Select calls present** — `select_entries` action appears in `dss_calls.jsonl`
  for events branches (beats, scenes, offscreen, chapters).
- **Filter reduces apply_updates** — compare `apply_updates` call count for
  events branches against a pre-select baseline (s573: ~78 chapter calls,
  ~75 full-branch calls). Expect 70-80% reduction.
- **No empty selections on active branches** — if the LLM selects 0 entries
  for a branch that clearly has active content, the filter may be too aggressive.

### Hierarchical archival triggers (`on_{unit}_archived`)

Arcs now trigger on `on_chapter_archived` instead of `on_new_scene`, making
arc archival dependent on a chapter having archived in the same turn.

- **Arcs only fire after chapters** — in the log, `on_chapter_archived` trigger
  should appear on the same turn as a chapter archive, never on a scene-only turn.
- **No independent arc archives** — an arc archive on a turn where no chapter
  archived indicates the hierarchy is broken.
- **`_turn_start_numbers` snapshot** — the trigger map for arcs should be
  `{'on_chapter_archived': [...]}` only when `_chapter_number` increased.

### Attribute/Skill dataclass restructure

`Character.attributes` and `Element.attributes` changed from `list[str]`
to `dict[str, Attribute]`; `Character.skills: dict[str, Skill]` added.

- **Attributes are dicts** — in `characters.json`, each entry's `attributes`
  should be a dict of `{name: {can_do, tier, source}}`, not a list of strings.
- **Skills populated** — at least one character should have a non-empty `skills`
  dict by mid-run.
- **Template renders correctly** — in the log's rendered prompts, attributes
  should appear as `- name: can_do (tier N)` lines, not raw `{'can_do': ...}` dicts.

### Underscore structural fields + archive path restoration

`_starting_scene`/`_ending_scene`/`_starting_chapter`/`_ending_chapter` are
underscore-prefixed (invisible to LLM in prompts) and restored from the
programmatic archive template if the LLM nulls them during refinement.

- **All `_ending_scene` values non-None** — every chapter entry in
  `events.json` must have a non-None `_ending_scene`.
- **All `_ending_chapter` values non-None** — every arc entry in
  `arcs.json` must have a non-None `_ending_chapter`.
- **`_chapter_start_scene` / `_arc_start_chapter` present** — in
  `current_scene.json`, these tracking fields should be set after the first
  chapter/arc archive.
- **Cadence gate functional** — chapters should NOT archive on every scene turn.
  Expected: ~1 chapter per 8-10 scenes, not 1 per 2-3 scenes.

## Lessons Learned

1. **Check the most recent run directory**. Don't assume what you have in memory is the actual latest run - always do a quick check on the actual run directories first. Hashes aren't sorted either.
2. **Subagents time out on large log files** if they try to read them into context.
   The Python-script approach is mandatory, not optional.
3. **`live_mode_*.json` files are overwritten each run.** Only the last run's JSON
   survives. For multi-turn analysis, the log file is the durable record.
4. **Log files truncate replies at ~303 chars.** Full replies are in `turn_NNN/reply.txt`.
5. **The `total=Xs` field in local-call lines includes non-GPU overhead.** The sum
   of call durations exceeds wall time by ~86%. Use per-turn wall time as ground truth.
6. **Write the interim doc before dispatching subagents.** It gives them context
   and gives the user an early read while the subagents run.
7. **Check the schema files, not just the plan docs.** The optimization plan claimed
   a fix was "applied and verified" but it was never actually in the schema files.
   Always verify against the actual files the run uses.