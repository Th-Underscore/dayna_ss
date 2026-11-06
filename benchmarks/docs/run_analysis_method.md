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

### Knowledge "YES" echo fix (s546+)

The branch query→update handoff previously appended the query's `"YES"` response
to the conversation history, causing the LLM to echo `"YES"` instead of generating
JSON in the `apply_updates` call. Fixed by removing the history append
(`6123ec9e`). Verify:

- **No `Failed to parse LLM response as JSON ... Raw response: YES`** in the log.
- **`knowledge.json` entries** — `apply_updates` calls should produce valid JSON
  path-patch arrays, not bare `"YES"` tokens.

### Knowledge in writer prompts (s546+)

`RetrievalContext` now carries a `knowledge` field populated from `knowledge.json`,
so writer (gemma-4) prompts include knowledge entries (`15587c51`). Verify via
oaiproxy:

- **Writer prompt** — knowledge entries (e.g. `black_key_cipher_function`) should
  appear in the system context of gemma-4 calls, not just in DSS calls.

### Cloud/local time split (s546+)

Turn completion lines now show the split: `done: turn N in Xs (cloud Ys, local Zs)`
(`514575dc`). Verify:

- **Log format** — every `done: turn` line includes `(cloud ...s, local ...s)`.
- **Cloud share** — `cloud_s / dt_s` ratio per turn. A high ratio (>50%) on
  non-scene turns indicates guide hangs, not DSS slowness.

### Knowledge subject population (ongoing)

Knowledge `add_new_query` was rewritten with concrete criteria, JSON example,
`branch_list`, and scene grounding (`96be4dd4`). Verify after next run:

- **Knowledge population** — `knowledge.json` should have entries by mid-run
  (s545 had 2 entries; expect 5+ with the grounded prompt).
- **`knows` on Characters** — `dict[str, EpistemicStance]` with in-vocab values
  only. Flag any out-of-vocab values (e.g. `"confirmed"`).
- **`learned` on StoryEvents** — `dict[str, EventKnowledgeRole]` with in-vocab
  values only.

### Scene archival cascade (known broken)

The `add_new_query` prompt for `events.scenes` is vague (same failure mode as
Knowledge before its rewrite). Scenes never get archived, which blocks the entire
cascade: scenes → chapters → canon digest → synopsis updates. Verify it's still
broken (confirming the fix is needed) or fixed if the prompt was updated:

- **`events.json` → `scenes`** — should have entries. If always empty, the cascade
  is still blocked.
- **`events.json` → `chapters`** — should increment at archival boundaries.
- **`general_info.json` → `synopsis`** — should update at chapter/arc archival,
  not just once at the end.

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