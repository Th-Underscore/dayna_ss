# Long-Horizon Soak Test v2: Design Decisions

Decisions for the next round of DSS long-horizon soak tests. Resolves spreadsheet
length, cloud model selection, and local model testing order.

## Decision 1: Spreadsheet length (varies by genre)

Lengths range from **150 to 2000 turns** across the 12 sheets, deliberately varied to
capture the context degradation curve at different lengths.

Math: A typical 3-hour D&D session has 50-120 substantive exchanges (150-300 speaker
turns). One dayna_ss turn covers ~8-12 exchanges, so one session = ~7-12 DSS turns.

Real campaign lengths (per Reddit r/dndnext, 229 DMs surveyed):
- One-shot: 50-120 exchanges
- Short published adventure: 600-2,200 exchanges
- Medium module (Strahd/ToA): 1,000-6,000 exchanges
- Standard published campaign: 1,100-3,100 exchanges
- Long homebrew: 2,500-24,000 exchanges
- Epic: 5,000-36,000 exchanges

These are for real campaigns; our test lengths are shorter but still represent full
campaigns at DSS's higher exchange density per turn.

### Per-sheet length assignments

| Sheet | Genre | Turns | Rationale |
|-------|-------|-------|-----------|
| greek_mythology_retelling | Quest | **150** | Time-bound, ends at known goal. Tightest pacing. |
| cozy_mystery | Mystery | **200** | Single investigation, clear resolution endpoint |
| heist_caper | Heist | **250** | Planning → execution → fallout arc |
| western | Western | **250** | Town arrival → conflict → showdown |
| cyberpunk_thriller | Cyberpunk | **350** | Tech investigation with corporate escalation |
| noir_detective | Mystery/noir | **350** | Layered conspiracy reveal |
| romance | Romance | **400** | Slow-burn emotional arc |
| gothic_haunted_mansion | Gothic horror | **500** | Discovery → revelation → confrontation |
| sci_fi_heist | Sci-fi heist | **700** | Multi-crew, multi-system operation |
| horror | Cosmic horror | **800** | Progressive revelation + sanity degradation |
| postapocalyptic_survival | Survival | **1200** | Long-term rebuilding, faction politics |
| fantasy_epic | Epic fantasy | **2000** | Multiple arcs, character development, war |

Distribution: 3 short (150-200), 4 medium (250-400), 3 long (500-800), 2 epic (1200-2000).

**Note/beat density scaling:**
- Short (150-200): ~25% of original notes
- Medium (250-400): ~40% of original
- Long (500-800): ~65% of original
- Epic (1200-2000): ~100% of original

Same note-type distribution as original 100-turn sheets (character_plant, foreshadow,
supersession, style_hold, etc.). Plant turns distributed across the full range, not
bunched early. No duplicate needles.

## Decision 2: Cloud model for guide/judge/auditor

### Primary: Mistral La Plateforme (Experiment tier)

**Mistral Medium 3.5-128B** for guide role (creative writing).

**Why Mistral won:** The binding constraint for long-horizon soak testing is monthly
token volume, not RPM or TPM. Our test matrix requires ~15M tokens total (6,000 turns
across 12 spreadsheets × ~2,466 tokens/turn). No other free provider comes close.

| Provider | Free Tier Volume | Models Available | Credit Card | Best For |
|----------|-----------------|------------------|-------------|----------|
| **Mistral La Plateforme** | **~1B tokens/month** | All models incl. Large 3, Medium 3.5, Small 4 | No (SMS) | **Primary: guide role (creative)** |
| Cloudflare Workers AI | 10,000 neurons/day (~13K turns/mo) | Qwen 3.8 27B, GLM-5.3-Flash, Llama variants | No | Supplement: high-quality creative |
| Gemini 3.5 Flash Lite | 500 RPD (~15K turns/mo) | Gemini 3.5 Flash Lite only (2026) | No | Supplement: judge/auditor roles |
| OpenRouter Free | Very low RPM/TPM | Ling 3.0 Flash Fin, Laguna S 2.1, Nemotron 3 Ultra | No | Backup only |

**Mistral model lineup (La Plateforme):**

| Model | Parameters | Released | Notes |
|-------|-----------|----------|-------|
| Mistral Medium 3.5 | 128B dense | Apr 2026 | Merged instruct/reason/coding. FP8, fits 1× H200. **Recommended for guide.** |
| Mistral Small 4 | 119B total / 6.5B active (MoE) | Mar 2026 | Fast, capable. Good for judge/auditor. |
| Mistral Large 3 | 675B total / 41B active (MoE) | Dec 2025 | Top-tier quality. 256K context. |
| Ministral 3 8B | 8B dense | Dec 2025 | Edge-optimized. Budget option. |

All models share the same 1B token/month budget on the free tier. Mistral uses an
OpenAI-compatible API at `https://api.mistral.ai/v1/`, so adding it to `cloud_client.py`
is straightforward.

### Secondary: Cloudflare Workers AI (free tier)

**Qwen 3.8 27B** — 27B dense, released Aug 17 2026, free tier, no credit card.

Why: Fresh (1 month old), dense 27B (not a distilled small model), 262K context window.
Same architecture class as the user's local 27B models. Free tier is 10,000 neurons/day
which translates to ~13K turns/month at our average 2,466 tokens/turn — enough for the
short and medium spreadsheets but not the epics.

Caveat: Several newer Cloudflare models (GLM-5.3-Flash, Kimi K2.7, DeepSeek V4)
require paid billing. Only older/baseline models are on the free tier.

### Supplemental: Gemini (for judge/auditor)

**Gemini 3.5 Flash Lite** — the only 3.x model with meaningful free-tier limits.

From the user's rate-limit research (see `gemini_api_rate_limits.md`):
- 500 RPD, 600K TPM, 1K concurrent, 1M context
- The only Gemini model with enough RPD for soak testing (others cap at 20-100 RPD)
- Known to produce high-quality structured evaluation output (proven in prior runs)
- Binding constraint: 500 requests/day limits us to ~1-2 spreadsheets/day for guide role

### Why not use local model for scoring?

The user's local 27B could theoretically score itself, but the cloud judge receives
different context (outlines and overviews, not the full dialogue), making it a
genuinely independent evaluation. Using a separate cloud model avoids any
self-evaluation bias.

### Recommended allocation for soak test

| Role | Primary | Fallback |
|------|---------|----------|
| Guide (creative writing) | Mistral Medium 3.5-128B | Cloudflare Qwen 3.8 27B |
| Judge (per-turn evaluation) | Mistral Small 4 | Gemini 3.5 Flash Lite |
| Auditor (periodic audit) | Mistral Small 4 | Gemini 3.5 Flash Lite |
| Final overview | Mistral Large 3 | Gemini 3.5 Flash Lite |

This keeps all roles within the single 1B token/month Mistral budget while
preserving evaluation independence. The total ~15M tokens for the full matrix
is ~1.5% of the monthly budget.

## Decision 3: Local model testing order

**Test Qwen3.8-27B (Queen variant) first, then drop to smaller models.**

Order:
1. Qwen3.8-Queen-27B-W4A16-AWQ (frontier medium, already on server)
2. Qwen3.5-9B-W4A16-AWQ (current default)
3. Gemma-4-12B-StyleTune.i1-Q6_K (alternative architecture)

Reasoning: Establish the system's upper-bound quality with the best model before
measuring the degradation delta with weaker models. If the 27B degrades at turn 300,
that's a DSS system issue. If it holds to 500, we can attribute mid-campaign decay
in smaller models to model capability rather than system drift.

## Current state (for reference)

- 12 rewritten sheets: greek_mythology_retelling (150), cozy_mystery (200),
  heist_caper (250), western (250), cyberpunk_thriller (350), noir_detective (350),
  romance (400), gothic_haunted_mansion (500), sci_fi_heist (700), horror (800),
  postapocalyptic_survival (1200), fantasy_epic (2000)
- Total: 1,456 notes, 427 beats across all sheets
- Original 100-turn sheets archived in spreadsheets/archived/
- Best soak model proven: Qwen3.6-35B-A3B (4.0/5 runs at 100 turns)
- Current server: Qwen3.8-Queen-27B-W4A16-AWQ (tp=2, port 5000)
- 100-turn run duration: ~100 min (1.2 min/turn)
- Estimated durations: 200-turn sheet ~3.5h, 500-turn ~8h, 2000-turn ~32h

## Next steps

1. ~~Pick one sheet, write 500-turn version~~ ✓ Done — all 12 sheets rewritten
   (see `spreadsheet_rewrite.md` and validation results)
2. Add Mistral support to `cloud_client.py` (OpenAI-compatible API at
   `https://api.mistral.ai/v1/`)
3. Create Mistral La Plateforme account (SMS verification, no credit card),
   get API key
4. Update `long_horizon_soak.py` defaults: guide=Mistral Medium 3.5, judge/auditor=
   Mistral Small 4, or add `--guide-model`/`--judge-model`/`--auditor-model` flags
5. Run one medium-length sheet (e.g. cozy_mystery, 200 turns) to validate the
   Mistral integration end-to-end
6. Run the full 12-sheet matrix
7. Analyze retention curves, identify degradation points
8. Repeat with smaller local models to measure capability delta

## Sources

- D&D campaign length: Threadfall campaign data (median 7 sessions, main 30-60)
- D&D per-session exchange density: Reddit r/DnD encounter data, Dice Dragons
  pacing guides
- Mistral free tier (La Plateforme): docs.mistral.ai/models, console.mistral.ai
  (verified September 2026)
- Mistral Medium 3.5: huggingface.co/mistralai/Mistral-Medium-3.5-128B
- Mistral Small 4: docs.mistral.ai/models/mistral-small-4-0-26-03
- Cloudflare Workers AI: developers.cloudflare.com/workers-ai/models/
  (verified September 2026)
- Qwen 3.8 27B on Cloudflare: developers.cloudflare.com/changelog/post/2026-08-17-
  qwen-3.8-27b-workers-ai/ (released Aug 17, 2026)
- OpenRouter free models: openrouter.ai/collections/free-models (verified
  September 2026)
- Nemotron 3 Ultra: build.nvidia.com/nvidia/nemotron-3-ultra-550b-a55b (June 2026)
- Laguna S 2.1: huggingface.co/poolside/Laguna-S-2.1 (July 2026)
- Ling 3.0 Flash Fin: huggingface.co/inclusionAI/Ling-3.0-flash-Fin (Sept 3, 2026)
- Gemini rate limits: Google AI Studio docs (September 2026)
- Groq rate limits: console.groq.com/docs/rate-limits (verified July 2026)