# Interim Findings: sci_fi_heist__f66f99cb (150 turns, seed 514)

## Run Config

- 150 turns, seed 514, Gemma-4-26B-A4B-StyleTune-V2 (local)
- Cloud: gemini-3.5-flash-lite (guide, judge, auditor)
- `do_instr: false` + `enable_thinking: true` + `reasoning_effort: low` (Option D)
- Judge memory limits: 50k per entry, 200k total (bumped from 160/2800)
- Character attributes added to Sable and Dex in spreadsheet

## Triage Summary

| Metric | This Run (f66f99cb) | Baseline (d9b155be, 300t) | Delta |
|---|---|---|---|
| Style | 4.81 | 4.93 | -0.12 |
| Quality | 4.57 | 4.62 | -0.05 |
| Memory | **4.83** | 4.26 | **+0.57** |
| Combined | **4.74** | 4.60 | **+0.14** |
| Turns ≥4.5 | 49/55 (89%) | — | — |
| Turns <4.0 | 2/55 (3.6%) | — | — |

- Final overview: 4.5, "publication-quality precision"
- 55 judged turns (every 5th turn × ~150 turns minus gaps)
- Cloud calls: guide 21, judge 20, auditor 40

## Key Observations

1. **Memory fidelity jumped +0.57** — the 50k/200k limit bump is the clear driver. The judge no longer sees truncated state. This was the weakest axis in the baseline (4.26) and is now near-ceiling (4.83).

2. **Style/Quality slightly down** (-0.12/-0.05). Could be:
   - Thinking tokens consuming output budget (shorter replies?)
   - Sampling variance (different seed)
   - `do_instr: false` removing the instruction prompt's pacing/structure guidance
   - Need prose subagent to check reply length distribution

3. **Two outlier turns** (84, 104) with style=3.0:
   - Turn 84: voice + memory slip (narrating drainage pipe scene while stored state shows skimmer in flight)
   - Turn 104: judge claims third-person slip ("I lift my legs") — but "I" IS first-person. Possible judge error.

4. **Early-turn memory dips** (turns 4, 19, 24 at 3.0): judge notes "empty character entries" — expected early in the run before DSS has populated all characters.

## Anomalies to Investigate

- Turn 84: genuine continuity break? (scene mismatch between reply and stored state)
- Turn 104: judge false positive? ("I lift my legs" is first-person)
- Reply length distribution vs baseline (267 words mean)
- Proper noun overlap vs baseline (~3 mean)