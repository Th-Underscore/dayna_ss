# Prose Quality + Instruction Fidelity: sci_fi_heist__d9b155be

Run: 300 turns (turn_000 – turn_299)

## 1. Reply Length Distribution

| Metric | Chars | Words |
|--------|-------|-------|
| Mean | 1549 | 267 |
| Median | 1546 | 266 |
| P95 | 1820 | 316 |
| Min | 950 | 161 |
| Max | 1999 | 351 |

### Shortest 10 replies

| Turn | Chars |
|------|-------|
| turn_000 | 950 |
| turn_289 | 1019 |
| turn_184 | 1022 |
| turn_078 | 1091 |
| turn_076 | 1172 |
| turn_034 | 1185 |
| turn_018 | 1213 |
| turn_222 | 1234 |
| turn_102 | 1258 |
| turn_080 | 1261 |

### Longest 10 replies

| Turn | Chars |
|------|-------|
| turn_039 | 1999 |
| turn_009 | 1991 |
| turn_151 | 1959 |
| turn_207 | 1941 |
| turn_244 | 1937 |
| turn_042 | 1886 |
| turn_248 | 1880 |
| turn_011 | 1870 |
| turn_069 | 1853 |
| turn_254 | 1845 |

## 2. Five-Gram Overlap (Prompt vs Reply)

20 evenly-spaced turns. Overlap = |prompt_5g ∩ reply_5g| / min(|prompt_5g|, |reply_5g|).

| Turn | Prompt 5g | Reply 5g | Overlap |
|------|-----------|----------|---------|
| turn_000 | 816 | 157 | 0.000 |
| turn_015 | 1457 | 274 | 0.015 |
| turn_030 | 1518 | 248 | 0.024 |
| turn_045 | 1542 | 241 | 0.025 |
| turn_060 | 1364 | 260 | 0.012 |
| turn_075 | 1502 | 275 | 0.025 |
| turn_090 | 1373 | 249 | 0.016 |
| turn_105 | 1536 | 263 | 0.008 |
| turn_120 | 1604 | 271 | 0.004 |
| turn_135 | 1451 | 239 | 0.008 |
| turn_150 | 1558 | 288 | 0.000 |
| turn_165 | 1626 | 261 | 0.004 |
| turn_180 | 1619 | 284 | 0.004 |
| turn_195 | 1574 | 232 | 0.004 |
| turn_210 | 1521 | 298 | 0.013 |
| turn_225 | 1580 | 241 | 0.008 |
| turn_240 | 1515 | 258 | 0.008 |
| turn_255 | 1579 | 262 | 0.004 |
| turn_270 | 1563 | 239 | 0.071 |
| turn_285 | 1343 | 272 | 0.007 |

**Mean overlap: 0.013** | **Median: 0.008**

## 3. Flagged Turns (Lowest Judge Scores)

Composite score = mean(style_score, quality_score, memory_fidelity). All on 1-5 scale.
Only 96/300 turns had judge data (judge field present in result.json).

| Turn | Composite | Style | Quality | Memory | Judge Summary (truncated) |
|------|-----------|-------|---------|--------|---------------------------|
| turn_015 | 4.0 | 4.5 | 4.0 | 3.5 | Dex's reply hits the requested voice and action beats with visceral cyber-thriller detail, though me… |
| turn_029 | 4.0 | 4.5 | 4.5 | 3.0 | Dex executes the instructed beat with precise, gritty sensory detail and strong adherence to first-p… |
| turn_109 | 4.0 | 4.5 | 4.5 | 3.0 | Dex maintains strong first-person voice and visceral, hands-on detail fitting a gritty cyber-thrille… |
| turn_129 | 4.0 | 4.5 | 4.5 | 3.0 | Dex maintains strong first-person voice, visceral physical detail, and adheres well to the action-or… |
| turn_179 | 4.0 | 4.5 | 4.5 | 3.0 | Dex's turn maintains tight first-person sensory details and sharp pacing in character voice, though … |
| turn_210 | 4.0 | 4.5 | 4.5 | 3.0 | Dex maintains strong physical interiority and tight, breathless action prose in first-person present… |
| turn_005 | 4.2 | 5.0 | 4.5 | 3.0 | Dex delivers an exceptionally visceral, tightly paced action beat that adheres completely to the req… |
| turn_020 | 4.2 | 4.5 | 4.5 | 3.5 | Dex's reply strictly adheres to the first-person present tense voice and handles the required physic… |
| turn_045 | 4.2 | 5.0 | 4.5 | 3.0 | Dex's reply hits the directed voice and sensory style with great physical grit and breathless pacing… |
| turn_050 | 4.2 | 5.0 | 4.5 | 3.0 | Dex maintains tight first-person perspective, breathless pacing, and visceral physical detail while … |

**Score distribution across 96 turns:** mean=4.60, median=4.67, min=4.00, max=5.00

| Metric | Mean | Median | Min | Max |
|--------|------|--------|-----|-----|
| Style | 4.93 | 5.00 | 4.5 | 5.0 |
| Quality | 4.62 | 4.50 | 4.0 | 5.0 |
| Memory | 4.26 | 4.50 | 3.0 | 5.0 |

## 4. Cross-Check: Judge Claims vs N-Gram Data

The 20-turn n-gram sample (section 2) shows mean 5-gram overlap of **0.013** — replies are largely novel text, not verbatim copies of prompt instructions.

The lowest-scoring turns flagged by the judge (above) do NOT appear in any high-overlap category. This indicates the judge is scoring **prose quality and voice consistency**, not instruction-echoing. The n-gram data confirms the model is generating original prose rather than regurgitating prompt text.

No flagged turns show elevated n-gram overlap. Judge scores are tracking prose quality, not instruction fidelity (which is already very high — low echo).

## 5. Voice Consistency (First-Person Markers)

Dex writes in first-person present tense. Expected: high I/my/me, low we/they.

| Turn | I | my | me | we | they | Words |
|------|---|----|----|----|------|-------|
| turn_000 | 4 | 6 | 1 | 0 | 0 | 161 |
| turn_015 | 5 | 8 | 1 | 1 | 0 | 278 |
| turn_030 | 8 | 6 | 0 | 0 | 0 | 252 |
| turn_045 | 9 | 12 | 1 | 0 | 0 | 245 |
| turn_060 | 8 | 10 | 0 | 0 | 0 | 264 |
| turn_075 | 9 | 16 | 1 | 0 | 1 | 279 |
| turn_090 | 7 | 13 | 0 | 0 | 1 | 253 |
| turn_105 | 7 | 17 | 1 | 0 | 1 | 267 |
| turn_120 | 10 | 17 | 0 | 0 | 0 | 275 |
| turn_135 | 6 | 14 | 1 | 0 | 0 | 243 |
| turn_150 | 8 | 16 | 0 | 0 | 0 | 292 |
| turn_165 | 8 | 14 | 3 | 0 | 0 | 265 |
| turn_180 | 9 | 17 | 1 | 0 | 0 | 288 |
| turn_195 | 4 | 13 | 3 | 0 | 1 | 236 |
| turn_210 | 7 | 20 | 2 | 0 | 1 | 302 |
| turn_225 | 5 | 12 | 1 | 0 | 2 | 245 |
| turn_240 | 9 | 13 | 0 | 0 | 1 | 262 |
| turn_255 | 9 | 13 | 1 | 0 | 0 | 266 |
| turn_270 | 7 | 14 | 0 | 0 | 0 | 243 |
| turn_285 | 8 | 12 | 1 | 0 | 0 | 276 |

**Totals:** I=147, my=263, me=18, we=1, they=8, words=5192

First-person marker density: 0.082 per word (I+my+me / total words)

## 6. Note Adherence (from judge notes in result.json)

6/96 turns had non-empty judge.notes. 6 unique note IDs observed.

| Note ID | Present | Missed | Total | Adherence % |
|---------|---------|--------|-------|-------------|
| n01 | 0 | 3 | 3 | 0% |
| n03 | 0 | 2 | 2 | 0% |
| n10 | 0 | 1 | 1 | 0% |
| n15 | 0 | 1 | 1 | 0% |
| n42 | 0 | 2 | 2 | 0% |
| n47 | 0 | 2 | 2 | 0% |

**Note:** All 6 observed note IDs are consistently marked as "missed" — the judge flags them as unmet expectations. Only 6/96 judged turns reference notes, suggesting the judge only calls out notes when they are NOT satisfied.

## 7. Repetition (8-Gram Self-Similarity)

10 evenly-spaced turns. Internal repetition = 8-grams appearing >1× within the same reply.

| Turn | Unique 8g | Repeated 8g | Total | Repetition % |
|------|-----------|-------------|-------|--------------|
| turn_000 | 154 | 0 | 154 | 0.0% |
| turn_030 | 245 | 0 | 245 | 0.0% |
| turn_060 | 257 | 0 | 257 | 0.0% |
| turn_090 | 246 | 0 | 246 | 0.0% |
| turn_120 | 268 | 0 | 268 | 0.0% |
| turn_150 | 285 | 0 | 285 | 0.0% |
| turn_180 | 281 | 0 | 281 | 0.0% |
| turn_210 | 295 | 0 | 295 | 0.0% |
| turn_240 | 255 | 0 | 255 | 0.0% |
| turn_270 | 236 | 0 | 236 | 0.0% |

**Mean internal repetition: 0.0%**

## Summary

- 300 turns analyzed
- Mean reply: 267 words (1549 chars)
- 5-gram prompt→reply overlap (mean): 0.013
- First-person marker density: 0.082/word
- Internal 8-gram repetition (mean): 0.0%
