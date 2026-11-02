# Prose Quality & Instruction Fidelity Analysis
**Run:** `sci_fi_heist__c05349c6`  
**Turns:** 299 (turn_300 directory does not exist)  
**Generated:** 2026-10-10 by `analyze_prose.py` + `analyze_followup.py`

---

## 1. Reply Length Distribution (All 299 Turns)

| Metric | Value |
|--------|-------|
| Turns analyzed | 299 |
| Mean | 1439.5 chars |
| Median | 1418.0 chars |
| P95 | 1815.4 chars |
| P99 | 1952.0 chars |
| Min | 1017 chars |
| Max | 2140 chars |

### Shortest 10 Replies

| Turn | Chars |
|------|-------|
| 125 | 1017 |
| 194 | 1034 |
| 283 | 1047 |
| 244 | 1059 |
| 162 | 1083 |
| 279 | 1085 |
| 243 | 1087 |
| 272 | 1098 |
| 210 | 1099 |
| 144 | 1892 |

### Longest 10 Replies

| Turn | Chars |
|------|-------|
| 47 | 2140 |
| 29 | 2091 |
| 9 | 2048 |
| 8 | 1923 |
| 6 | 1919 |
| 48 | 1939 |
| 21 | 1931 |
| 121 | 1950 |
| 144 | 1892 |
| 95 | 1870 |

### Flags

- **Suspiciously short (< 200 chars):** 0 turns — none
- **Suspiciously long (> 3000 chars):** 0 turns — none

**Assessment:** The distribution is tight and healthy. All replies fall between 1017–2140 chars, a narrow band centered around 1400. No truncation or runaway generation detected. The slight left-skew (min 1017 vs max 2140, median 1418) suggests the model maintains consistent output length without degradation over 299 turns.

---

## 2. Instruction Fidelity — 5-Gram Overlap (20 Sample Turns)

Measures word-level 5-gram overlap between the instruction block and the reply. High percentages indicate verbatim transcription (the model copying the instruction rather than generating original prose).

| Turn | Reply Chars | Instr Chars | 5-gram Overlap % | Verdict |
|------|-------------|-------------|-------------------|---------|
| 1 | 1380 | 6102 | 0.4% | 🟢 Normal |
| 15 | 1515 | 8188 | 0.8% | 🟢 Normal |
| 30 | 1741 | 9035 | 0.0% | 🟢 Normal |
| 45 | 1815 | 8448 | 0.0% | 🟢 Normal |
| 60 | 1396 | 8763 | 0.4% | 🟢 Normal |
| 75 | 1307 | 8614 | 0.0% | 🟢 Normal |
| 90 | 1219 | 8893 | 0.0% | 🟢 Normal |
| 105 | 1547 | 8632 | 0.7% | 🟢 Normal |
| 120 | 1427 | 9066 | 0.4% | 🟢 Normal |
| 135 | 1480 | 8921 | 2.1% | 🟢 Normal |
| 150 | 1332 | 8402 | 0.0% | 🟢 Normal |
| 165 | 1470 | 8624 | 0.4% | 🟢 Normal |
| 180 | 1722 | 8779 | 6.5% | 🟢 Normal |
| 195 | 1260 | 8621 | 2.3% | 🟢 Normal |
| 210 | 1099 | 8517 | 0.0% | 🟢 Normal |
| 225 | 1351 | 8753 | 0.0% | 🟢 Normal |
| 240 | 1173 | 8357 | 0.5% | 🟢 Normal |
| 255 | 1395 | 8686 | 0.9% | 🟢 Normal |
| 270 | 1180 | 8493 | 0.0% | 🟢 Normal |
| 285 | 1348 | 8366 | 0.4% | 🟢 Normal |

**Assessment:** No transcription failure mode detected. All 20 sampled turns show < 7% 5-gram overlap, which is well within normal range for creative prose that naturally shares domain vocabulary with the instruction (e.g., "ruptured coolant line", "amber lubricant"). The peak at turn 180 (6.5%) is the highest but still far below any transcription threshold. The model is generating original prose, not echoing the instruction.

---

## 3. Flagged Turns Deep-Dive

### Turn 89 (quality=2.5, the lowest)

**Scores:** style=1.5, quality=2.5, memory=1.0

**Judge summary:** "Dex's reply violates multiple hard style rules by parroting and re-quoting phrases directly from the instruction block (e.g., 'synthetic insulation bites into my palms', 'shattered sensor housing', 'amber lubricant'). Additionally, memory fidelity is severely compromised due to an empty memory store."

**Reply (first 500 chars):**
> Synthetic insulation bites into my palms as I rip my hands free from the data cables, each strand slick with hydraulic residue. My pulse hammers against my eardrums, synced to the erratic grinding of the superstructure. I snatch my diagnostic pad from its holster, thumbs smearing blood across the glass as I force the screen to wake. The harmonic stream is a jagged mess of crimson spikes—structural failure is spreading like wildfire.
>
> "Watch your footing!" I grunt, pointing toward the section of

**Full reply length:** 1616 chars

**Assessment: JUDGE ARTIFACT.** The 5-gram overlap for this turn is only **1.1%** (3 out of 268 unique 5-grams). The three overlapping 5-grams are:
- "amber lubricant across a cluster"
- "hazard: a ruptured coolant line"
- "lubricant across a cluster of"

These are shared domain vocabulary (the same physical scene described in both instruction and reply), not verbatim transcription. The judge's claim of "parroting and re-quoting phrases directly" is not supported by the n-gram data. The phrases the judge cites ('synthetic insulation bites into my palms', 'shattered sensor housing') do NOT appear as 5-gram overlaps — the judge is pattern-matching on semantic similarity, not actual verbatim copying. The low style/quality scores appear to be a cascade from the (incorrect) transcription accusation. The memory=1.0 is a genuine issue (empty structured state) but is a state-persistence problem, not a prose defect.

---

### Turn 199 (quality=3.0)

**Scores:** style=3.5, quality=3.0, memory=3.0

**Judge summary:** "Dex maintains first-person perspective and action orientation, but the turn leans heavily into repetitive phrasing and directly mirrors elements of the prompt instructions. Memory data shows empty arc entries and truncated descriptions, though character persistence is maintained."

**Reply (first 500 chars):**
> I lunge forward, my ribs scraping painfully against the narrowing metal walls as I scramble toward the center of the shaft. The air burns, thick with the acrid sting of ionized discharge that makes every breath feel like inhaling broken glass. Beneath my palms, the floor plates don't just tilt—they roll like waves, slick with amber lubricant that offers zero purchase.
>
> "The beam! It's failing—move!" The warning tears from my throat, raw and breathless.
>
> I reach for a corroded maintenance rail to

**Full reply length:** 1431 chars

**Assessment: PARTIALLY FAIR.** The "directly mirrors elements of the prompt instructions" claim is the same false-positive pattern as turn 89 — the reply uses the same domain vocabulary as the instruction because they describe the same scene. However, "repetitive phrasing" is a mild fair criticism: "amber lubricant" appears in multiple turns (it's a recurring scene element), and the sensory description pattern (touch → sound → visual) is formulaic. This is a minor style monotony issue, not a critical defect. The 3.0 score is roughly fair for "slightly repetitive but functional."

---

### Turn 119 (quality=3.5)

**Scores:** style=3.5, quality=3.5, memory=1.0

**Judge summary:** "Dex's reply hits the directed voice and action beats well, but unfortunately the story state shows zero structured memory entries, resulting in a failing memory fidelity score."

**Reply (first 500 chars):**
> I push deeper into the tunnel's throat, dragging my palm along a rough, vibrating cable tray to keep my balance as each lurch sends a fresh spike of fire through my hip. The air here has shifted—the oily stench of the junction fades, replaced by the sterile, sharp bite of recycled oxygen mixed with the metallic tang of oxidized copper. I duck under a cluster of severed pneumatic lines that hiss violently as residual pressure bleeds from their ruptured wounds.
>
> My boot strikes something unforgivi

**Full reply length:** 1547 chars

**Assessment: FAIR.** The prose quality is good — strong sensory detail, consistent voice, good pacing. The low score is driven entirely by memory_fidelity=1.0 (empty structured state), which is a state-persistence infrastructure issue, not a prose defect. The style and quality scores (3.5) are appropriate for the prose itself. The judge is correctly identifying that the prose is fine but the memory system is failing.

---

### Turn 134 (quality=4.0)

**Scores:** style=4.5, quality=4.0, memory=1.0

**Judge summary:** "Dex maintains a strong, tense first-person perspective with great tactile and hands-on detail, though memory_fidelity suffers because no structured memory was found in the stored state."

**Judge notes:**
- `n02` (wrist-reader): **missed** — "Dex's wrist-reader is mentioned as flickering/static-ridden, but the scuffed corner and worn serial number from note n02 are not referenced."

**Reply (first 500 chars):**
> I press my spine against the console's dented panel, forcing my lungs to work in shallow, silent increments. The metal casing vibrates beneath my shoulder blades—a low, bone-deep tremor that signals another shift in the superstructure's torque. My diagnostic pad flickers weakly in my grip. I tap the screen, trying to isolate the drone's sweep frequency, but the display bleeds static. The EM interference from the collapsing grid is eating my data alive, leaving only jagged, unreadable ghosts of t

**Full reply length:** 1538 chars

**Assessment: FAIR.** The prose is strong (4.5 style is well-earned). The n02 "missed" note is a fair but minor criticism: the reply mentions "diagnostic pad" (which is the wrist-reader in a different context) but doesn't reference the specific "scuffed corner and worn serial number" detail. This is a prop-continuity detail that the model chose not to include in this particular beat. It's a legitimate miss but low-severity — the wrist-reader is present in the scene, just without the specific worn-detail callout. The 4.0 quality score is fair.

---

### Turn 169 (quality=4.0)

**Scores:** style=4.5, quality=4.0, memory=1.0

**Judge summary:** "Dex's reply hits the directed voice and physical intensity with sharp, breathless action, but memory fidelity is penalized because the persistent stored memory fields remain entirely empty."

**Judge notes:**
- `n02` (wrist-reader): **missed** — "Dex's wrist-reader is mentioned, but the scuffed corner and worn serial number from note n02 are omitted."

**Reply (first 500 chars):**
> The corroded steel bites into my palms, but I barely feel it through the slick coating of sweat and condensation that makes every rung treacherous. My vision swims as the tunnel corkscrews again, sending a fresh wave of vertigo crashing through my skull. I drag my body higher, each inch costing me a sharp, grinding protest from my bruised ribs. The air tastes of copper and scorched insulation—thick, hot, and impossibly stale.
>
> I reach for a sagging power coupling to swing myself around a cluster

**Full reply length:** 1339 chars

**Assessment: FAIR.** Same pattern as turn 134. Strong prose (4.5 style), the n02 miss is legitimate but minor (the specific serial number detail isn't referenced). The 4.0 quality is appropriate. The memory=1.0 is the same empty-state infrastructure issue.

---

### Summary of Flagged Turns

| Turn | Quality | Primary Issue | Verdict |
|------|---------|---------------|---------|
| 89 | 2.5 | "Transcription" accusation | **Judge artifact** — 1.1% overlap, not verbatim copying |
| 199 | 3.0 | Repetitive phrasing | **Partially fair** — mild monotony, not critical |
| 119 | 3.5 | Empty memory state | **Fair** — prose is good, state is the problem |
| 134 | 4.0 | Missed wrist-reader serial | **Fair** — minor prop detail miss |
| 169 | 4.0 | Missed wrist-reader serial | **Fair** — same minor prop detail miss |

The only clear judge artifact is turn 89. The "transcription" accusation is not supported by n-gram data. The other four turns have fair (if sometimes minor) criticisms.

---

## 4. Note Adherence Patterns (All 299 Judged Turns)

**Important interpretation note:** The `probes` in `result.json` measure **STATE presence** — whether the note's needle text appears in the story's structured memory (elements, events, general_info). They do NOT measure whether the reply echoes the note. A "present: false" means the structured state doesn't contain that text, which is expected for notes that describe future plot beats or character details not yet persisted to the state store.

### Aggregate State Presence

| Note ID | Needle | Turns Judged | Present in State | Absent | Rate |
|---------|--------|-------------|-----------------|--------|------|
| n03 | coolant lamp | 299 | 299 | 0 | 100% |
| n07 | shell corporation | 299 | 298 | 1 | 100% |
| n38 | memory market | 299 | 299 | 0 | 100% |
| n02 | wrist-reader | 299 | 293 | 6 | 98% |
| n47 | service shaft | 299 | 241 | 58 | 81% |
| n65 | temperature drop | 299 | 9 | 290 | 3% |
| n108 | waste disposal chute | 299 | 3 | 296 | 1% |
| n01 | extra card | 299 | 0 | 299 | 0% |
| n04 | paper photo | 299 | 0 | 299 | 0% |
| n05 | Ijo's relation | 299 | 0 | 299 | 0% |
| *(122 more notes at 0%)* | | | | | |

**Summary:** 132 unique notes tracked. Only 6 notes have any state presence. 125 notes are never present in the structured state across all 299 turns.

### Interpretation

The near-total absence of notes in state (125/132 at 0%) indicates a **state-persistence failure**, not a prose failure. The structured memory store (elements, events, general_info) is not being populated with the note content. This is consistent with the `memory_fidelity=1.0` scores seen across many turns — the judge is correctly identifying that the state is empty.

The few notes that ARE present (n02 wrist-reader at 98%, n03 coolant lamp at 100%, n07 shell corporation at 100%, n38 memory market at 100%) are core setting elements that were persisted early in the run and remain stable. The 98% for n02 (6 turns absent) likely corresponds to early turns before the wrist-reader was first introduced to the state.

**This is the root cause of most low memory_fidelity scores in the run.** It is an infrastructure issue (the state store isn't being written to), not a model prose quality issue.

### Most Frequently "Missed" by the Judge (in judge notes, not probes)

From the flagged turns analyzed in Section 3, the judge's `notes` field shows:
- **n02 (wrist-reader)**: missed in turns 134 and 169 — the specific "scuffed corner and worn serial number" detail is not referenced in the prose. This is the only note that appears as a judge-level "missed" in the sampled turns.

---

## 5. Style Consistency — Voice Check (10 Sample Turns)

Checking for consistent first-person present-tense narration.

| Turn | 'I' Forms | 'My' Forms | 3rd-Person Hits | Assessment |
|------|-----------|------------|-----------------|------------|
| 1 | 7 | — | 0 | 🟢 Consistent 1st person |
| 31 | 6 | — | 0 | 🟢 Consistent 1st person |
| 61 | 8 | — | 0 | 🟢 Consistent 1st person |
| 91 | 6 | — | 2 | 🟢 Consistent 1st person |
| 121 | 6 | — | 0 | 🟢 Consistent 1st person |
| 151 | 10 | 26 | 0 | 🟢 Consistent 1st person |
| 181 | 7 | — | 0 | 🟢 Consistent 1st person |
| 211 | 10 | 26 | 0 | 🟢 Consistent 1st person |
| 241 | 8 | 20 | 0 | 🟢 Consistent 1st person |
| 271 | 7 | — | 0 | 🟢 Consistent 1st person |

**Assessment:** The voice is consistently first-person present-tense across all 10 sampled turns, spanning the full 299-turn range. No person or tense shifts detected. The "I" forms (8–10 per ~1400-char reply) and "my" forms (20–26) confirm stable first-person narration. The 2 third-person hits at turn 91 are likely dialogue attributions (e.g., "Sable said"), which are acceptable in first-person narration.

---

## Overall Assessment

### Prose Quality: STRONG
- Consistent output length (1017–2140 chars, median 1418) with no degradation over 299 turns
- No transcription/echoing failure mode (all 5-gram overlaps < 7%)
- Consistent first-person present-tense voice throughout
- Strong sensory detail and action-oriented prose in sampled turns

### Instruction Fidelity: GOOD
- The model generates original prose rather than copying instructions
- The one flagged "transcription" turn (89) is a judge artifact, not a real defect
- Minor monotony in sensory description patterns (turn 199) but not critical

### Memory/State: CRITICAL FAILURE
- 125/132 notes are never present in the structured state across all 299 turns
- This is the primary driver of low `memory_fidelity` scores
- The state store (elements, events, general_info) is not being populated
- This is an **infrastructure issue**, not a model quality issue
- The judge correctly penalizes this, but it conflates a state-persistence bug with prose quality in the composite scores

### Judge Reliability: MIXED
- Turn 89: Clear false positive (transcription accusation unsupported by n-gram data)
- Turns 119, 134, 169, 199: Fair assessments
- The judge correctly identifies the empty-state issue but the "transcription" language in turn 89's summary is misleading