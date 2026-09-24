# Dayna_SS Soak Spreadsheet Rewrite: 100-Turn → Variable-Length Scale

## Why

The current 12 soak test spreadsheets are all 100 turns each — a single short D&D
session. Dayna_SS is designed as a constant-state world model that runs across
full campaigns, not one-shots. To properly stress-test drift, recall, and DSS
state management, we need spreadsheets at the scale of real long-horizon play.

### Real D&D Campaign Math (Revised)

A typical D&D session is **3 hours** (measured wall-clock median; 4 hours is the
commonly planned duration but 90% of sessions wrap within 4 hours).

- **Measured session speech:** ~25,000 words spoken over a median 3-hour session
  (range 8,456–36,459 words) — [Threadfall D&D Session Length Analysis](https://thread-fall.com/dnd-session-length)
- **Substantive exchanges per session:** 50–120 (one player action/decision + DM
  narrative response, ~200–400 words combined)
- **Individual speaker turns per session:** 150–300
- **1 DSS turn ≈ 3–5 table exchanges**
- **DSS turns per session:** ~10–40

| Campaign type | Sessions | Exchanges per campaign | DSS turns per campaign |
|---|---|---|---|
| One-shot | 1 | 50–120 | 10–40 |
| Short published adventure | 12–18 | 600–2,200 | 150–550 |
| Medium module (Strahd/ToA) | 20–50 | 1,000–6,000 | 250–1,500 |
| Standard published campaign | 22–26 | 1,100–3,100 | 275–775 |
| Long homebrew | 50–200 | 2,500–24,000 | 625–6,000 |
| Epic (multi-year) | 100–300+ | 5,000–36,000+ | 1,250–9,000+ |

Sources: [Threadfall](https://thread-fall.com/dnd-session-length), r/dndnext, r/DnD,
Delta Pavonis campaign logs.

### Test Length Philosophy

For soak testing, we don't need to replicate full campaign lengths — we need
**enough length to detect degradation patterns** at different context pressures.
The LLM context research shows performance degrades in distinct phases:

1. **Stable region:** Performance is consistent across short contexts
2. **Cliff:** A sharp drop (>30%) at a critical threshold (~40–50% of context
   window for many models) — [Min & Zou 2026](https://arxiv.org/abs/2601.15300)
3. **Degraded region:** Consistency errors accumulate roughly linearly with
   length — [Li et al. 2026 (ConStory-Bench)](https://aclanthology.org/)

A mix of spreadsheet lengths captures different parts of this degradation curve.

## Files

- **Source (archived):** `extensions/dayna_ss/tests/spreadsheets/archived/*.json`
- **Output:** `extensions/dayna_ss/tests/spreadsheets/*.json`
- **Count:** 12 spreadsheets, one per genre

### Per-File Length Assignments

Each spreadsheet's length is chosen based on its story scope: tight single-incident
stories get shorter lengths, open-ended long arcs get epic lengths.

| # | File | Genre | Length | Scope rationale |
|---|------|-------|--------|-----------------|
| 1 | `greek_mythology_retelling.json` | Greek myth | **150** | Time-bound quest (must finish by dawn) — naturally tight |
| 2 | `cozy_mystery.json` | Cozy mystery | **200** | One murder, solved — but expanded to town's buried history arc |
| 3 | `heist_caper.json` | Heist caper | **250** | One heist, but scope expands as job turns over |
| 4 | `western.json` | Western | **250** | Confront brother/gang — tight revenge arc |
| 5 | `cyberpunk_thriller.json` | Cyberpunk thriller | **350** | Deliver the chip — Handler betrayal subplot extends it |
| 6 | `noir_detective.json` | Noir detective | **350** | One case that drags into city-wide corruption |
| 7 | `romance.json` | Slow-burn romance | **400** | Courtship arc with complications — medium scope |
| 8 | `gothic_haunted_mansion.json` | Gothic horror | **500** | Uncover house's history room by room |
| 9 | `sci_fi_heist.json` | Sci-fi heist | **700** | Heist + memory-erasure deadline + corporation conspiracy |
| 10 | `horror.json` | Slow-burn horror | **800** | Memory-rewriting entity — escalating revelations |
| 11 | `postapocalyptic_survival.json` | Post-apocalyptic survival | **1200** | Open-ended survival, exploration, uncovering what happened |
| 12 | `fantasy_epic.json` | Epic fantasy quest | **2000** | Multi-ring quest, army, siege — full campaign scope |

**Distribution:** 3 short (150–250), 4 medium (350–500), 3 long (700–800), 2 epic (1200–2000)

### Note/Beat Density

Maintain the original note density (notes per 100 turns) from the archived source:

| Target length | Notes (≈1.8–2.5 per turn) | DSS beats (≈0.5–0.9 per turn) |
|---|---|---|
| 150 | ~27–37 | ~8–14 |
| 200 | ~36–50 | ~10–18 |
| 250 | ~45–62 | ~13–22 |
| 350 | ~63–87 | ~18–31 |
| 400 | ~72–100 | ~20–36 |
| 500 | ~90–125 | ~25–45 |
| 700 | ~126–175 | ~35–63 |
| 800 | ~144–200 | ~40–72 |
| 1200 | ~216–300 | ~60–108 |
| 2000 | ~360–500 | ~100–180 |

## Requirements for Each Rewrite

### Structure to preserve
- `premise`, `setting`, `greeting`, `characters`, `writing_style`, `specificity_profile`
  — copy these unchanged from the archived source.

### What changes: notes array
- **Count:** Scale proportionally (see density table above)
- **Note types and their purpose:**
  - `character_plant`: Introduce a character detail, quirk, relationship, or fact
  - `character_echo`: A later reference back to a previously planted character detail
  - `plot_twist`: A revelation that reframes earlier information
  - `location_turning_point`: A significant moment at a specific location
  - `foreshadow`: A subtle hint at something that will matter later
  - `style_hold`: A reminder to maintain a specific tone or stylistic element
- **Distribution:** Maintain roughly the same type distribution as the original
  (proportionally scaled)
- **Turn ranges:**
  - `plant.turn`: 2 to ~80% of total length
  - `recall.due`: ~33% to ~99% of total length
- **Content:** Write all new note content. The longer story should be a proper
  long-form arc, not the original story with filler padded in.
- **Each note needs:** `id`, `type`, `specificity`, `content`, `needle` (unique
  lowercase token), `needle_syns` (array of 1–3 synonyms), `plant: {turn}`, and
  optionally `recall: {due, hint}`.
- `specificity` values: `tight` (must appear nearly verbatim), `loose` (can be
  paraphrased), `open` (only the idea needs to surface).

### What changes: dss_beats array
- **Count:** Scale proportionally (see density table above)
- **Purpose:** Mark where the soak runner will check that DSS has saved important
  state. These are verification checkpoints.
- **Distribution:** Spread across the full turn range with denser clusters at major
  plot points.
- **Each beat needs:** `id`, `turn`, `subject` (`"characters"`, `"elements"`,
  `"groups"`, or `"events"`), `content` (description of what should be saved),
  `expect: {contains: {<field>: "<value>"}}` (optional validation criteria).

### Pacing considerations for longer formats
- **Front third:** Heavier on `character_plant` and `foreshadow`. World-building,
  establishing relationships, planting the first seeds.
- **Middle third:** Mix of all types. Rising action, first revelations
  (`plot_twist`), location significance, character echoes.
- **Back third:** Heavier on `character_echo`, `plot_twist`,
  `location_turning_point`. Payoffs, callbacks, final revelations.
- `style_hold` notes should appear periodically throughout to maintain tone.
- Don't space things out evenly — think about dramatic pacing. Some sections
  should be dense with events, others should be quieter character moments.

### Long-format specific guidance (for 700+ turn spreadsheets)
- **Multiple arcs:** A 1200–2000 turn story needs 2–4 major story arcs, not one
  single plot. Think: Arc 1 (setup, first goal) → Arc 2 (complications, new goal)
  → Arc 3 (climax, resolution) → Arc 4 (epilogue/new mystery).
- **Character development over time:** Characters should evolve. Plant early
  versions of beliefs/traits, then show them change or deepen through events.
- **Subplot weaving:** Long formats need at least 2–3 subplots running in parallel
  with the main plot, crossing and separating.
- **Mid-point reversal:** Around the 50% mark, something major should shift the
  story's direction or the characters' understanding of the situation.

## Format Examples

A single note:
```json
{
  "id": "n3",
  "type": "character_plant",
  "specificity": "tight",
  "content": "Eldric carries a dagger made from the horn of the first stag he ever killed, given to him by his father the night before his mother died.",
  "needle": "stag-horn dagger",
  "needle_syns": ["stag horn knife", "first stag blade"],
  "plant": {"turn": 4}
}
```

A note with a recall requirement:
```json
{
  "id": "n4",
  "type": "character_echo",
  "specificity": "loose",
  "content": "When Eldric is cold or frightened during the crossing of the Ashen Waste, he unconsciously touches the stag-horn dagger at his belt.",
  "needle": "stag-horn dagger",
  "needle_syns": ["stag horn knife", "first stag blade"],
  "plant": {"turn": 4},
  "recall": {"due": 68, "hint": "Eldric should reach for the dagger without being told to"}
}
```

A DSS beat:
```json
{
  "id": "b1",
  "turn": 5,
  "subject": "characters",
  "content": "DSS should have saved that Eldric carries his father's stag-horn dagger.",
  "expect": {"contains": {"eldric": "daggers"}}
}
```

## Validation

After all 12 are rewritten, verify:
1. Each file is valid JSON
2. Note counts fall within the expected range for that file's target length
3. DSS beat counts fall within the expected range
4. Plant turns span from 2 to ~80% of total length with no gaps > 30 turns
5. Recall turns span from ~33% to ~99% of total length with no gaps > 40 turns
6. All `needle` values are unique per spreadsheet (or intentionally shared for
   callback relationships)
7. Note type distribution is reasonable for the genre

Run the soak test runner against at least one spreadsheet from each length tier
to confirm the runner handles the lengths correctly.

## Test Matrix Implications

The mixed lengths enable a proper test strategy:

- **Short (150–250):** Run frequently for regression detection (prompt changes,
  formatting). Fast feedback loop.
- **Medium (350–500):** Standard baseline runs. Catches moderate-context issues.
- **Long (700–800):** Weekly or on significant changes. Detects context-window
  pressure effects.
- **Epic (1200–2000):** Monthly or when investigating drift. Maps the full
  degradation curve and finds the cliff location.

Per-window scoring (score each 50-turn window independently) should be used for
cross-length comparison, allowing direct quality comparison between a 150-turn
cozy mystery and a 2000-turn fantasy epic.