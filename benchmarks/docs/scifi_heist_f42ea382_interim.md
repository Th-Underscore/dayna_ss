# Sci-Fi Heist f42ea382 — Interim Findings

**Run:** `sci_fi_heist__f42ea382` (seed 518, 150 turns, Gemma-4-26B IQ4_XS, compressed cadence, no arc force)
**Config:** `do_instr: false` + thinking enabled (`enable_thinking: true`, `reasoning_effort: low`)
**Wall time:** ~13h (09:28 → 00:26 local, including ~1h server startup)

## Quality Summary

| Metric | Value |
|--------|-------|
| Final overview score | 4.5/5.0 |
| Judge style | 4.95 mean (min 4.5) |
| Judge quality | 4.74 mean (min 4.0) |
| Judge memory | 5.00 mean (min 5.0) |
| Turns with quality < 4.0 | 0 |
| Cloud calls | guide 150, judge 51, auditor 150 |

## Temporary Check Results

### Character attribute population
- **Turn 0 initial population:** ✅ Sable (1 attr), Dex (1 attr), Corporate circuit-riders (2 attrs) — all seeded from spreadsheet
- **Turn 149 final state:** ✅ 12 characters, 0 empty attributes
- **N/A placeholders:** ⚠️ 6 characters still have "N/A" in biography or voice (non-POV characters introduced mid-run)

### Chapter archival
- **Count:** ⚠️ 6 chapters in `events.json` vs chapter #12 in `current_scene` — significant gap
- **Titles:** ✅ All narrative (no "CHAPTER #N" placeholders)
- **Spans:** ⚠️ Gaps (scenes 10-14, 25-28 unarchived) and overlap (chapters 2 and 4 both cover scenes 33-42)
- **Narrative fields:** ✅ All have summary + key_changes

### Arc archival (natural, no force)
- **Count:** ✅ 1 arc ("The Ascension Extraction")
- **Chapter span:** ✅ Consecutive [1, 2, 3, 4]
- **Narrative fields:** ✅ All populated (summary 711 chars, character_arcs 328, relationship_shifts 172, plot_threads 3)

## Known Issues (carried forward)

1. **Chapter archival gap** — `events.json` only has 6 chapters while `current_scene` tracks chapter #12. The archival mechanism is dropping chapters or the scene-to-chapter mapping is inconsistent.
2. **Chapter overlap** — Two chapters cover the same scene range (33-42).
3. **Fence parsing warnings** — 77 instances of "Failed to extract new entry names" due to `strip_response` not stripping whitespace before the `[]` stopping-string check. Fixed in `discovery.py` (`.strip()` added).