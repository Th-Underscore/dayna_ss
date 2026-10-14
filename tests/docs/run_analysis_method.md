# Soak Run Analysis Method

How to analyze a completed long-horizon soak run. Replicable for any run.

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

## Lessons Learned

1. **Subagents time out on large log files** if they try to read them into context.
   The Python-script approach is mandatory, not optional.
2. **`live_mode_*.json` files are overwritten each run.** Only the last run's JSON
   survives. For multi-turn analysis, the log file is the durable record.
3. **Log files truncate replies at ~303 chars.** Full replies are in `turn_NNN/reply.txt`.
4. **The `total=Xs` field in local-call lines includes non-GPU overhead.** The sum
   of call durations exceeds wall time by ~86%. Use per-turn wall time as ground truth.
5. **Write the interim doc before dispatching subagents.** It gives them context
   and gives the user an early read while the subagents run.
6. **Check the schema files, not just the plan docs.** The optimization plan claimed
   a fix was "applied and verified" but it was never actually in the schema files.
   Always verify against the actual files the run uses.