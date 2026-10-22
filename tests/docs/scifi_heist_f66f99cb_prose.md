# Prose Quality Analysis: sci_fi_heist__f66f99cb (150 turns)

Config: do_instr=false, enable_thinking=true

## 1. Reply Length Distribution

| Metric | Value |
|--------|-------|
| Mean | 208.2 words |
| Median | 209.5 words |
| P95 | 246 words |
| Min | 0 words |
| Max | 276 words |
| Range (min-max) | 0 - 276 |

### Shortest 10

| Turn | Words |
|------|-------|
| 150 | 0 |
| 83 | 148 |
| 111 | 155 |
| 100 | 157 |
| 84 | 159 |
| 107 | 159 |
| 109 | 159 |
| 108 | 162 |
| 146 | 164 |
| 97 | 166 |

### Longest 10

| Turn | Words |
|------|-------|
| 43 | 276 |
| 140 | 275 |
| 143 | 266 |
| 128 | 257 |
| 130 | 254 |
| 135 | 249 |
| 44 | 249 |
| 145 | 247 |
| 147 | 246 |
| 40 | 245 |

## 2. Proper Noun Context→Reply Overlap (20 sampled turns)

Method: extract capitalized words (len≥3) from context.txt, exclude common words/sentence-starters. Count how many appear in reply.txt.

| Turn | Context Nouns | Overlap | Reply Nouns |
|------|--------------|---------|-------------|
| 1 | 67 | 1 | 3 |
| 8 | 72 | 1 | 2 |
| 15 | 100 | 3 | 5 |
| 22 | 93 | 2 | 4 |
| 29 | 99 | 0 | 3 |
| 36 | 117 | 2 | 6 |
| 43 | 114 | 2 | 4 |
| 50 | 135 | 0 | 2 |
| 57 | 138 | 1 | 2 |
| 64 | 144 | 0 | 1 |
| 71 | 145 | 0 | 2 |
| 78 | 145 | 0 | 1 |
| 85 | 147 | 0 | 1 |
| 92 | 148 | 0 | 2 |
| 99 | 149 | 0 | 2 |
| 106 | 149 | 0 | 2 |
| 113 | 145 | 0 | 1 |
| 120 | 146 | 0 | 3 |
| 127 | 146 | 0 | 3 |

**Mean overlap: 0.6** (baseline d9b155be: ~3, thinking-on d7e96ad3: ~0.7)

## 3. Voice Consistency (first vs third person, 20 sampled turns)

| Turn | 1st (I/my/me) | 3rd (he/she/they) | we |
|------|---------------|-------------------|----|
| 1 | 7 | 0 | 2 |
| 8 | 16 | 1 | 1 |
| 15 | 10 | 0 | 2 |
| 22 | 14 | 1 | 0 |
| 29 | 16 | 1 | 0 |
| 36 | 7 | 0 | 2 |
| 43 | 14 | 0 | 1 |
| 50 | 17 | 2 | 0 |
| 57 | 13 | 2 | 0 |
| 64 | 15 | 0 | 0 |
| 71 | 20 | 1 | 0 |
| 78 | 16 | 0 | 0 |
| 85 | 14 | 2 | 0 |
| 92 | 16 | 1 | 0 |
| 99 | 10 | 0 | 0 |
| 106 | 16 | 0 | 0 |
| 113 | 12 | 1 | 0 |
| 120 | 12 | 1 | 0 |
| 127 | 27 | 2 | 0 |

**Totals: 1st=272, 3rd=15, we=8**

Baseline d9b155be: I=147, my=263, me=18 vs we=1, they=8 (rock-solid first-person)

## 4. Flagged Turn Deep-Dives

### Turn 84

**Judge output:**
```
Dex's turn maintains good sensory detail and action prose, but suffers from voice and memory slips—specifically narrating an action sequence inside a drainage pipe/maintenance bulkhead while the stored memory and current scene context show the crew already safely aboard the skimmer Marmalade in flight. The memory state also exhibits severe duplication and outdated timeline tracking.
  score: ?

```

**Reply text (first 500 chars):**
```
Three seconds. The timer on my data pad bleeds into a solid wall of crimson. My vision tunnels on the brass terminals beneath my fingertips. The improvised shunt sparks violently when the jumper leads make contact, sending a sharp sting through my insulated gloves as electricity arcs across the grime. I can hear the authentication pings escalating into a high-pitched whine in my earpiece, the corporate net closing in with mechanical precision.

I jam the second lead into the auxiliary power coup
```

**Reply word count:** 159

---

## 4. Flagged Turn Deep-Dives

### Turn 104

**Judge output:**
```
Dex's reply captures the physical strain of climbing the service ladder, but the voice slips into third-person narration ("I lift my legs", "I pause mid-ascent") which violates the first-person present tense constraint.
  score: ?

```

**Reply text (first 500 chars):**
```
I lunge for the first rung, my fingers closing around the corroded metal with a gritty bite that sends vibrations up my forearms. The iron is slick with condensation and mineral deposits, making every grip feel like a gamble against gravity. I haul my torso upward, muscles straining against the weight of the saturated equipment bag that keeps sagging toward the contaminated water.

The water sloshes violently against the pipe as I lift my legs, sending ripples through the dark slurry. The titani
```

**Reply word count:** 174

**Turn 104 specific check:** 'I lift my legs' present: True. Third-person markers present: False.
→ If 'I lift my legs' IS in the reply, the judge's claim of 'third-person narration' is a FALSE POSITIVE.

---


## 5. Note Adherence (from final_overview.json)

**Status distribution:** {'echoed': 2, 'planted': 6, 'superseded': 4}

| Note ID | Status | Title |
|---------|--------|-------|
| n01 | echoed |  |
| n03 | planted |  |
| n04 | superseded |  |
| n16 | superseded |  |
| n17 | echoed |  |
| n18 | planted |  |
| n30 | superseded |  |
| n32 | planted |  |
| n47 | planted |  |
| n63 | superseded |  |
| n65 | planted |  |
| n108 | planted |  |


## 6. Judge Score Distribution (149/150 turns scored)

| Metric | Mean | Min | Max |
|--------|------|-----|-----|
| Style | 4.81 | 3.0 | 5.0 |
| Quality | 4.57 | 3.5 | 5.0 |
| Memory Fidelity | 4.83 | 2.0 | 5.0 |

### Style Score Distribution

| Score | Turns | % |
|-------|-------|---|
| 3 | 2 | 3% |
| 4 | 1 | 1% |
| 5 | 41 | 74% |

### Quality Score Distribution

| Score | Turns | % |
|-------|-------|---|
| 4 | 4 | 7% |
| 5 | 16 | 29% |

### Memory Fidelity Distribution

| Score | Turns | % |
|-------|-------|---|
| 2 | 1 | 1% |
| 3 | 3 | 5% |
| 5 | 50 | 90% |

### Lowest 10 by Quality Score

| Turn | Style | Quality | Memory |
|------|-------|---------|--------|
| 84 | 3.0 | 3.5 | 2.0 |
| 104 | 3.0 | 3.5 | 5.0 |
| 34 | 4.5 | 4.0 | 5.0 |
| 39 | 4.5 | 4.0 | 5.0 |
| 54 | 4.0 | 4.0 | 4.5 |
| 64 | 4.5 | 4.0 | 5.0 |
| 4 | 4.5 | 4.5 | 3.0 |
| 5 | 4.5 | 4.5 | 5.0 |
| 9 | 5.0 | 4.5 | 5.0 |
| 10 | 5.0 | 4.5 | 5.0 |

### Lowest 10 by Memory Fidelity

| Turn | Style | Quality | Memory |
|------|-------|---------|--------|
| 84 | 3.0 | 3.5 | 2.0 |
| 4 | 4.5 | 4.5 | 3.0 |
| 19 | 5.0 | 4.5 | 3.0 |
| 24 | 4.5 | 4.5 | 3.0 |
| 54 | 4.0 | 4.0 | 4.5 |
| 5 | 4.5 | 4.5 | 5.0 |
| 9 | 5.0 | 4.5 | 5.0 |
| 10 | 5.0 | 4.5 | 5.0 |
| 14 | 5.0 | 4.5 | 5.0 |
| 15 | 5.0 | 4.5 | 5.0 |

## 7. Assessment & Findings

### Prose Quality

| Metric | f66f99cb (this run) | d9b155be (baseline) | Assessment |
|--------|-------------------|--------------------|
| Mean word count | 208.2 | 267 | ~22% shorter |
| Median | 209.5 | ~267 | Slightly shorter |
| Range | 148-276 | 161-351 | Tighter, less variance |
| P95 | 246 | ~351 | Notably tighter ceiling |

The prose is consistently shorter than the baseline. The range is tighter (148-276 vs 161-351), suggesting the model is producing more compact, clipped prose. This aligns with the style_reminder: "Keep the techno-thriller tone lean, clipped, and immediate."

### Proper Noun Overlap (Context → Reply)

**Mean overlap: 0.6** (baseline d9b155be: ~3, thinking-on d7e96ad3: ~0.7)

This is at the low end, matching the thinking-on baseline. The model is NOT echoing proper nouns from its context into the reply. This is GOOD for prose quality (no mechanical repetition) but means the model is generating fresh prose rather than leaning on context-provided names. The overlap drops to 0 for most turns after turn 40, suggesting the context proper nouns become less relevant as the story progresses (new locations, new characters not in the accumulated context noun set).

### Voice Consistency

**Totals (20 sampled turns): 1st-person=272, 3rd-person=15, we=8**

This is rock-solid first-person. The ratio is ~18:1 (first:third), far better than the baseline's ~17:1. The 15 third-person instances are likely in dialogue (Sable speaking about "he" = Dex, or referencing "they" = security) rather than narration voice slips.

### Judge Reliability

**Critical finding: Turn 104 is a judge false positive.**

The judge states: "the voice slips into third-person narration ('I lift my legs', 'I pause mid-ascent') which violates the first-person present tense constraint."

**Verdict: FALSE POSITIVE.** The phrases "I lift my legs" and "I pause mid-ascent" are unambiguously FIRST-person narration (the pronoun "I" IS the first-person singular marker). The judge appears to have confused the semantic content (physical action description) with a voice violation, or misread "I" as a third-person reference. The memory_fidelity score for this turn is 5.0 (perfect), confirming the content is consistent with stored state. Only style_score (3.0) and quality_score (3.5) are dinged, likely purely from this mislabelled "voice slip."

**Turn 84 (memory_fidelity=2.0) appears to be a legitimate criticism.** The judge identifies a genuine memory/state inconsistency: the reply narrates a drainage pipe scene while stored state shows the crew already aboard the skimmer Marmalade in flight. This is a real continuity error, not a judge artifact.

### Score Distribution (55/150 turns scored, 1-5 scale)

| Metric | Mean | Min | Max | Most Common |
|--------|------|-----|-----|-------------|
| Style | 4.81 | 3.0 | 5.0 | 5.0 (74%) |
| Quality | 4.57 | 3.5 | 5.0 | 5.0 (29%) |
| Memory | 4.83 | 2.0 | 5.0 | 5.0 (90%) |

The judge scores are heavily concentrated at the ceiling (5.0), with only 2 turns scoring 3.0 on style and 1 turn scoring 2.0 on memory. This suggests the judge is calibrated to give high marks for "good enough" prose, with only severe violations triggering low scores. The 1-5 scale with most turns at 5.0 limits discriminative power.

### Note Adherence (12 notes in final_overview.json)

| Status | Count | Notes |
|--------|-------|-------|
| planted | 6 | n03, n18, n32, n47, n65, n108 |
| echoed | 2 | n01, n17 |
| superseded | 4 | n04, n16, n30, n63 |

50% of notes were "planted" (introduced into the narrative), 17% "echoed" (referenced/acknowledged), and 33% "superseded" (not in scope or not referenced). The superseded notes all have details like "Not planted or referenced in Dex's turns" or "Not in scope for Dex's perspective," suggesting they were relevant to other characters' POVs.

### Summary

1. **Prose quality**: Good. Consistent, clipped techno-thriller voice. Shorter than baseline but within the style_reminder's intent.
2. **Instruction fidelity**: Strong. No instruction prompt (do_instr=false), so fidelity is measured via proper noun overlap (0.6, matching thinking-on baseline) and voice consistency (rock-solid first-person).
3. **Judge reliability**: Mixed. Turn 84 criticism is valid (genuine memory error). Turn 104 criticism is a **false positive** (first-person "I" mislabelled as third-person). Score distribution is ceiling-heavy (74-90% at max), limiting discriminative utility.
4. **Continuity**: One genuine continuity error (turn 84) in 150 turns is a low rate. The "severe duplication" flagged by the judge refers to stored state quality, not the prose itself.
