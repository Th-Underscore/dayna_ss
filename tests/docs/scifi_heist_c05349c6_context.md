# DSS State Evolution: scifi_heist c05349c6 (300 turns)

Run directory: `extensions/dayna_ss/tests/runs/sci_fi_heist__c05349c6`

Total turns: 300 (turn_000 through turn_299)

---

## 1. Scene Transitions

- **Distinct scenes**: 114
- **Total scene transitions**: 113 (every scene change is exactly +1)
- **Avg turns per scene**: 2.6
- **Transition frequency by phase**:
  - Early (turns 1–100): ~37 transitions
  - Mid (turns 101–200): ~38 transitions
  - Late (turns 201–300): ~38 transitions

### Turns-per-scene distribution

| Turns in scene | # of scenes |
|---|---|
| 1 | 2 |
| 2 | 96 |
| 3 | 12 |
| 4 | 2 |
| 5 | 1 |
| 7 | 1 |

Most scenes last 2 turns (96/114 = 84%). A few are 1-turn (2) or 3+ turns (16).

### Anomalies

None detected — scene numbers progress monotonically by +1. No jumps or resets across the full 300 turns.

### Scene number trajectory (sampled)

| Turn | Scene # | Chapter | Arc |
|---|---|---|---|
| 1 | 1 | 1 | 1 |
| 26 | 11 | 1 | 1 |
| 51 | 22 | 1 | 1 |
| 76 | 33 | 1 | 1 |
| 101 | 39 | 1 | 1 |
| 126 | 50 | 1 | 1 |
| 151 | 60 | 1 | 1 |
| 176 | 70 | 1 | 1 |
| 201 | 78 | 1 | 1 |
| 226 | 88 | 1 | 1 |
| 251 | 97 | 1 | 1 |
| 276 | 106 | 1 | 1 |
| 300 | 114 | 1 | 1 |

**Note**: The entire 300-turn run stays in Chapter 1, Arc 1. The scene counter is the only progression axis.

---

## 2. Character Tracking

- **Roster**: 2 characters throughout (Sable, Dex)
- **Min characters**: 2 (constant)
- **Max characters**: 2 (constant)
- **Total entries**: 0 (both present from turn 1)
- **Total exits**: 0

### Character detail growth

| Turn | Chars | Total fields | JSON size |
|---|---|---|---|
| 1 | 2 | 24 | 1,549 B |
| 50 | 2 | 24 | 1,549 B |
| 100 | 2 | 24 | 1,549 B |
| 150 | 2 | 24 | 1,549 B |
| 200 | 2 | 24 | 1,549 B |
| 250 | 2 | 24 | 2,895 B |
| 300 | 2 | 24 | 4,623 B |

The character *schema* (field count) is static at 24 fields per character. The JSON size grows ~3× by turn 300 (1.5KB → 4.6KB), indicating field *values* are accumulating text (e.g. descriptions, history) but no new fields are added.

### Entry/exit log

None — both characters present from turn 1 through turn 300.

---

## 3. Entity Graph Growth

| Turn | Arcs | Elements | Groups | Events |
|---|---|---|---|---|
| 1 | 0 | 19 | 0 | 4 |
| 50 | 0 | 107 | 0 | 4 |
| 100 | 0 | 120 | 0 | 4 |
| 150 | 0 | 120 | 1 | 4 |
| 200 | 0 | 120 | 1 | 4 |
| 250 | 0 | 120 | 1 | 4 |
| 300 | 0 | 120 | 1 | 4 |

### Growth summary

| Type | Initial (turn 1) | Final (turn 300) | Growth |
|---|---|---|---|
| arcs | 0 | 0 | +0 |
| elements | 19 | 120 | +101 |
| groups | 0 | 1 | +1 |
| events | 4 | 4 | +0 |

**Key observation**: Element growth is front-loaded — 101 of 120 total elements (84%) appear in the first 100 turns. The count plateaus at 120 by turn 100 and stays flat for the remaining 200 turns. Groups get 1 entry around turn 150. Arcs and events remain static throughout.

---

## 4. Current Scene Content Depth

| Turn | JSON size | Top-level keys | `_scene_number` | `_scene_start_message` |
|---|---|---|---|---|
| 1 | 982 B | 7 | 1 | 2 |
| 50 | 1,110 B | 7 | 22 | 100 |
| 100 | 1,335 B | 7 | 39 | 198 |
| 150 | 1,744 B | 7 | 60 | 300 |
| 200 | 1,562 B | 7 | 78 | 396 |
| 250 | 1,476 B | 7 | 97 | 500 |
| 300 | 1,363 B | 7 | 114 | 596 |

### Structure (stable across all turns)

Top-level keys: `what`, `start`, `now`, `_scene_number`, `_chapter_number`, `_arc_number`, `_scene_start_message`

- `start` and `now` are both dicts with 4 sub-keys each (constant)
- `what` is a free-text scene summary string
- The scene is **stable in structure** — no growth in field count, no new keys
- JSON size oscillates between ~1KB and 1.7KB (content varies with scene content, not structural growth)
- `_scene_start_message` tracks the cumulative message offset (grows linearly with turns)

**Conclusion**: Scene state is a fixed-shape record that gets *replaced* at each scene transition. It does not accumulate depth over time.

---

## 5. Rolling Summaries

- **Count**: 37 files
- **Interval**: every 8 turns (turns 8, 16, 24, ..., 296)
- **Smallest**: 3,242 B (turn 8)
- **Largest**: 50,078 B (turn 296)
- **Final overview** (final_overview.json): 50,356 B

### Size growth

| Turn | Size (bytes) |
|---|---|
| 8 | 3,242 |
| 48 | ~6,500 |
| 88 | ~9,000 |
| 128 | ~14,000 |
| 168 | ~22,000 |
| 208 | ~32,000 |
| 248 | ~44,000 |
| 288 | 49,569 |
| 296 | 50,078 |

### Structure (stable keys)

Keys present: `overall_score`, `trajectory`, `verdict`, `arc_progression`, `style_consistency`, `memory_fidelity` (latter two appear from turn 32+).

**Growth pattern**: The rolling summaries grow roughly linearly from 3KB to 50KB over the run — a ~15× increase. This is the primary *growing* state artifact in the run. The content accumulates narrative history that the model must carry forward.

---

## Summary of State Evolution

| Artifact | Growth pattern | Final size |
|---|---|---|
| current_scene.json | Stable (replaced each scene) | ~1.4 KB |
| characters.json | Slow value growth (static schema) | 4.6 KB |
| elements.json | Front-loaded, plateaus at 120 | 120 items |
| arcs.json | Static (0) | 0 |
| groups.json | Single addition at ~turn 150 | 1 |
| events.json | Static (4) | 4 |
| rolling_overview_*.json | **Linear growth 3KB → 50KB** | 50 KB (37 files) |

### Key findings

1. **Scene progression is the primary state axis**: 114 scenes over 300 turns (~2.6 turns/scene), perfectly monotonic, no anomalies.
2. **Character roster is static**: 2 characters, no entry/exit events. Growth is only in field values (text accumulation), not structure.
3. **Entity graph is front-loaded**: Elements reach their 120-item plateau by turn 100 and never grow again. Arcs and groups barely change.
4. **Rolling summaries are the main growth vector**: 15× size increase over the run (3KB → 50KB). This is the artifact that will dominate context budget at scale.
5. **Scene state is a fixed-shape record**: No structural accumulation. Each scene transition replaces the scene wholesale.
6. **Single-chapter, single-arc run**: All 114 scenes occur within Chapter 1 / Arc 1. The chapter/arc counters never advance.