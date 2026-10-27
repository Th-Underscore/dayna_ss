# Soak Run Analysis Method

How to analyze a completed long-horizon soak run. Replicable for any run.Alr

## Inputs

- **Log file:** `tests/runs/logs/<run_name>.log` (e.g. `scifi_heist_gemma4_s200.log`)
- **Run directory:** `tests/runs/<spreadsheet>__<hash>/` (e.g. `sci_fi_heist__c05349c6/`)
  - `turn_NNN/reply.txt` — model reply per turn
  - `turn_NNN/instr_prompt.txt` — instruction block per turn
  - `turn_NNN/current_scene.json` — scene state snapshot
  - `turn_NNN/characters.json`, `elements.json`, `arcs.json`, `events.json` — entity state
  - `turn_NNN/result.json` — judge verdict (score, notes, probes)
  - `rolling_overview_NNN.json` — rolling summary every N turns
  - `final_overview.json` — final LLM-generated overview
- **Manifest:** `tests/runs/<spreadsheet>__<hash>/manifest.json`

## Workflow

### Step 1: Parent quick triage (no subagents)

Read `final_overview.json` and sample a few `result.json` files to get:
- Final score and the overview's stated weaknesses
- Score distribution (mean, floor, % at 4.5+)
- Which turns are flagged lowest
- Obvious anomalies (empty replies, score jumps)

Write an **interim findings doc** with these observations. This gives the subagents
context for what to look for and gives the user an early read.

### Step 2: Dispatch 3 parallel subagents

One subagent per analysis axis. All three run in parallel (single workflow call - see `skill:parallel-subagents` for more details).

**Critical constraint (baked into every subagent prompt):**
> Never read a large file into LLM context in one operation. For any file > ~50 lines,
> write a Python script (stdlib only) that does the mechanical I/O (regex, JSON parse,
> aggregation) and writes the deliverable `.md` directly to disk. The subagent's context
> should only see the script's small stdout (match counts, file sizes, key numbers),
> never the raw file content. Process one source at a time. Write the deliverable
> incrementally (append per completed section) so progress survives a kill.

#### Subagent A: Timing

**Task:** Parse the log file for per-turn and per-category LLM call timing.

**Script approach:**
1. Parse `local-call` and `cloud-call` log lines (regex for timestamp, role, duration, tokens)
2. Group by turn (turn boundaries are `=== TURN N ===` markers in the log)
3. Compute: per-turn wall time, local/cloud split, call counts, token counts
4. Compute: per-category stats (writer vs DSS vs cloud guide/judge/auditor) — mean, median, p95, max
5. Identify: slowest turns, outlier turns, failed cloud calls
6. Compute: total LLM time vs wall time (the ratio reveals non-GPU overhead)

**Deliverable:** `<run_id>_timing.md` with tables.

#### Subagent B: Context/State Evolution

**Task:** Track how DSS state evolves across all turns.

**Script approach:**
1. For each `turn_NNN/` directory, read the JSON state files
2. Track: scene number, character count, element count, arc count, event count
3. Detect: scene transitions (scene_number changes), character entry/exit, entity growth
4. Compute: turns-per-scene distribution, entity growth curves, JSON size growth
5. Check: rolling overview file count and size growth
6. Flag: anomalies (scene number resets, entity count drops, empty state files)

**Deliverable:** `<run_id>_context.md` with tables and trajectory samples.

#### Subagent C: Prose Quality + Instruction Fidelity

**Task:** Assess prose quality, instruction fidelity, and judge reliability.

**Script approach:**
1. Read all `reply.txt` files → length distribution (mean, median, p95, min, max, shortest/longest 10)
2. For 20 sampled turns (evenly spaced): compute 5-gram overlap between `instr_prompt.txt` and `reply.txt`
3. For flagged turns (lowest scores from `result.json`): extract judge summary, note statuses, reply text
4. Cross-check: do the judge's "transcription" claims match the n-gram data?
5. Check voice consistency: first-person markers across sampled turns
6. Aggregate note adherence: for each note ID, how many turns has it "present" vs "missed"

**Deliverable:** `<run_id>_prose.md` with tables and flagged-turn deep-dives.

### Step 3: Consolidate

Parent reads the three subagent reports, merges findings into the interim doc,
and produces the final consolidated assessment with:
- What's working well
- What's broken (systemic vs run-specific)
- Judge artifacts (false positives)
- Priority-ordered fix list

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

All deliverables go to `tests/docs/`:
- `<run_id>_timing.md`
- `<run_id>_context.md`
- `<run_id>_prose.md`
- `<run_id>_interim.md` (consolidated, starts as interim findings, updated with subagent results)

Where `<run_id>` = `<spreadsheet>__<hash>` (e.g. `scifi_heist_c05349c6`).

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

### Trigger reorg + no-schema-descriptions performance (s522)

This run tests the trigger reorg (non-scene turns: 13-14 → 2 `always` calls) and
`--no-schema-descriptions` (55% schema token reduction). Verify:

- **Non-scene turn call count** — should be ~2-5 local calls (down from 17-20).
  Parse the log for per-turn local-call counts on non-scene turns.
- **Non-scene turn wall time** — should be ~15-35s (down from ~90-110s).
- **Scene turn wall time** — expect modest improvement (~25-30% from schema reduction),
  still dominated by call count.
- **No parse errors** — the slim schema (no descriptions) should not cause the LLM
  to produce malformed JSON. Check for `Failed to parse` / `Failed to extract` in log.

### Call timing comparison (s522 vs s521)

- **Per-turn timing via `parse_timing.py`** — compare s522 (trigger reorg + slim schema)
  against s521 baseline (17-20 calls/turn, 86-108s non-scene).
- **Per-category breakdown** — DSS calls should show reduced input tokens (slim schema)
  and reduced call count (trigger reorg).
- **Prefill vs decode split** — confirm prefix caching is still effective (effective
  prefill rate should be ~2-3K tok/s with LCP).

### Chapter + arc archival (1:1 turns vs logging)

- **1:1 chapter logging** — chapter count in `events.json` should match the number of
  distinct chapter-boundary turns observed in the log. Each chapter archival event
  should produce exactly one entry.
- **1:1 arc logging** — same for arcs. Arc count should match arc-boundary turns.
- **LLM refined titles** — no entry should still have `"title": "CHAPTER #N"` or
  `"title": "ARC #N"`.
- **Scene/chapter span correctness** — consecutive integers, no gaps or overlaps.
- **Narrative fields filled** — `summary`, `key_changes`, `character_arcs` non-empty.

### Data comparison vs s521 (f42ea382)

- **Character count and field population** — compare per-character populated field
  counts. The gap-check + build_state fixes should maintain the improvement from s521.
- **Element count growth rate** — compare elements/turn between the two runs.
  Expect similar or slightly lower (trigger reorg gates element discovery to scene turns).
- **Events structure** — `past` should remain clean (no in-story beats leaking in,
  per the PastEvents discovery prompt fix).
- **Judge scores** — style/quality/memory should remain at s521 levels (4.9/4.8/5.0).

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