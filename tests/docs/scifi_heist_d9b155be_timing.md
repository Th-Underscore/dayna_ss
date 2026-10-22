# Soak Run Timing Analysis: `sci_fi_heist__d9b155be`

**Log:** `scifi_heist_gemma4_s209.log` (312 MB, 1,269,167 lines)
**Run window:** 2026-10-15 05:58:07 → 2026-10-16 05:38:20 (~23h 40m)
**Turns:** 300 (turns 0–300)
**Local model:** Gemma-4-26B-A4B-StyleTune-V2.i1-Q5_K_M.gguf
**Cloud model:** google3/gemini-3.5-flash-lite

## Summary

| Metric | Value |
|--------|-------|
| Total wall time | 84228s (23.4h) |
| Total LLM time (all categories) | 170631s (47.4h) |
| LLM / wall ratio | 202.6% |
| Non-LLM overhead | -86403s (-102.6%) |
| Local calls | 7932 |
| Guide replan calls | 60 |
| Guide user-turn calls | 302 |
| Judge calls | 96 |
| Auditor entries | 344 |
| DSS summarizations | 299 |
| Cloud call failures | 1 |

## Time Breakdown by Category

| Category | Total Time | % of Wall | % of LLM |
|----------|-----------|-----------|----------|
| Local (writer/DSS gen) | 160472s (44.58h) | 190.5% | 94.0% |
| Cloud guide (replan+user) | 3484s (58.1m) | 4.1% | 2.0% |
| Cloud judge | 793s (13.2m) | 0.9% | 0.5% |
| Cloud auditor | 5881s (98.0m) | 7.0% | 3.4% |
| DSS summarization *(subset of local)* | 59846s (997.4m) | 71.1% | — |
| **Non-LLM overhead** | **-86403s** | **-102.6%** | — |

## Per-Category Call Duration Stats

| Category | N | Mean (s) | Median (s) | P95 (s) | Max (s) | Min (s) |
|----------|---|----------|------------|---------|---------|----------|
| Local call (total) | 7932 | 20.29 | 13.19 | 66.88 | 327.49 | 0.53 |
| Local call (TTFT) | 7932 | 17.79 | 11.82 | 63.48 | 326.04 | 0.11 |
| Guide replan | 59 | 15.70 | 10.10 | 47.20 | 68.70 | 8.50 |
| Guide user-turn | 301 | 8.50 | 6.70 | 15.20 | 40.30 | 5.70 |
| Judge verdict | 96 | 8.26 | 6.50 | 16.20 | 49.00 | 5.90 |
| Auditor | 329 | 17.88 | 0.00 | 104.70 | 545.90 | 0.00 |
| DSS summarization | 299 | 200.15 | 101.20 | 520.00 | 808.60 | 45.40 |

## Per-Turn Wall Time

| Stat | Value |
|------|-------|
| Mean | 280.8s |
| Median | 167.0s |
| P95 | 598.0s |
| Max | 6537.0s |
| Min | 0.0s |

## Slowest 15 Turns

| Turn | Wall (s) | Local (s) | Guide (s) | Judge (s) | Auditor (s) | DSS (s) | Local N | LLM (s) |
|------|----------|-----------|-----------|-----------|-------------|---------|---------|---------|
| 187 | 6537 | 68 | 6.6 | 0.0 | 93.0 | 0 | 5 | 168 |
| 260 | 874 | 2233 | 21.4 | 0.0 | 293.5 | 809 | 53 | 2548 |
| 139 | 840 | 2098 | 7.4 | 0.0 | 84.3 | 778 | 42 | 2190 |
| 181 | 764 | 2083 | 6.4 | 6.2 | 93.1 | 696 | 42 | 2189 |
| 170 | 750 | 1821 | 16.0 | 0.0 | 0.0 | 671 | 41 | 1837 |
| 6 | 699 | 1073 | 6.1 | 6.2 | 0.0 | 660 | 34 | 1086 |
| 184 | 668 | 1718 | 6.4 | 0.0 | 0.0 | 607 | 39 | 1724 |
| 85 | 667 | 1723 | 15.9 | 113.5 | 0.0 | 622 | 43 | 1852 |
| 254 | 647 | 1721 | 13.8 | 0.0 | 174.4 | 585 | 45 | 1909 |
| 76 | 643 | 1750 | 6.6 | 10.0 | 0.0 | 595 | 51 | 1766 |
| 227 | 641 | 1672 | 6.0 | 0.0 | 0.0 | 596 | 51 | 1678 |
| 239 | 634 | 1642 | 6.6 | 0.0 | 0.0 | 582 | 45 | 1649 |
| 142 | 625 | 1634 | 6.1 | 0.0 | 0.0 | 569 | 34 | 1640 |
| 130 | 621 | 1477 | 49.8 | 0.0 | 0.0 | 517 | 36 | 1527 |
| 290 | 601 | 1077 | 12.9 | 0.0 | 291.3 | 388 | 34 | 1382 |

## Outlier Turns (wall > 2× median = 334s)

**93 turns** exceed the threshold.

| Turn | Wall (s) | Local (s) | DSS (s) | Guide (s) | Local N |
|------|----------|-----------|---------|-----------|--------|
| 187 | 6537 | 68 | 0 | 6.6 | 5 |
| 260 | 874 | 2233 | 809 | 21.4 | 53 |
| 139 | 840 | 2098 | 778 | 7.4 | 42 |
| 181 | 764 | 2083 | 696 | 6.4 | 42 |
| 170 | 750 | 1821 | 671 | 16.0 | 41 |
| 6 | 699 | 1073 | 660 | 6.1 | 34 |
| 184 | 668 | 1718 | 607 | 6.4 | 39 |
| 85 | 667 | 1723 | 622 | 15.9 | 43 |
| 254 | 647 | 1721 | 585 | 13.8 | 45 |
| 76 | 643 | 1750 | 595 | 6.6 | 51 |
| 227 | 641 | 1672 | 596 | 6.0 | 51 |
| 239 | 634 | 1642 | 582 | 6.6 | 45 |
| 142 | 625 | 1634 | 569 | 6.1 | 34 |
| 130 | 621 | 1477 | 517 | 49.8 | 36 |
| 290 | 601 | 1077 | 388 | 12.9 | 34 |
| 224 | 598 | 1476 | 548 | 6.7 | 55 |
| 264 | 596 | 1452 | 547 | 8.2 | 44 |
| 285 | 588 | 1315 | 462 | 48.0 | 35 |
| 167 | 587 | 1555 | 523 | 7.2 | 37 |
| 223 | 570 | 1502 | 519 | 6.3 | 44 |

## Cloud Call Failures

**1 turn(s)** had cloud call failures:

- Turn 290: 1 failure(s), wall=601s

Total cloud failures in log: **1**

## LLM Time vs Wall Time

The ratio reveals non-GPU overhead (network, orchestration, I/O, sleep).
**Warning:** Sum of individual call `total` durations is 203% of wall time (>100%).
This indicates concurrent LLM calls: the per-call `total` field includes wall-clock time for each call,
and when calls overlap (parallel requests to the local server), their sum exceeds actual elapsed time.
Effective concurrency factor: **2.03x**

```
Total wall:         84228s  (23.40h)
Total LLM:         170631s  (47.40h)  [local+cloud, no DSS double-count]
Ratio:             202.6%
Overhead:          -86403s  (-102.6%)
DSS spans:          59846s  (16.62h)  [subset of local, informational]
```

## Local Call Token Stats

| Metric | Value |
|--------|-------|
| Total local calls | 7932 |
| Total prompt tokens | 185,463,750 |
| Mean prompt tokens/call | 23382 |
| Median prompt tokens/call | 22838 |
| Max prompt tokens/call | 50,829 |

## Log-End Summary (from log)

```
local calls: 3028 tokens p=67264595 c=0
judge cloud: {'calls': 59, 'prompt_tokens': 3144740, 'completion_tokens': 20158, 'total_tokens': 3164898, 'model': 'google3/gemini-3.5-flash-lite'}
auditor cloud: {'calls': 336, 'prompt_tokens': 1200982, 'completion_tokens': 21429, 'total_tokens': 1222411, 'model': 'google3/gemini-3.5-flash-lite'}
```

## Notes

- `out=0` on all local calls: the log records `out=0` for all local-call lines; completion tokens are not tracked per-call in this log format.
- DSS summarization time includes multiple local LLM calls (summarizer + message chunks); it is NOT a single call.
- Guide user-turn calls (302) are the per-turn cloud calls for the guide's user message; guide replan calls (60) are the less-frequent replan triggers.
- Auditor entries with 0 calls (e.g. `auditor: 0 verdicts in 0.0s (0 calls)`) are included in the count but contribute 0s to timing.
- 1 cloud call failure at line 1216552 (2026-10-16 04:53:46): `replan: cloud call FAILED (unreachable); keeping the existing plan`.
