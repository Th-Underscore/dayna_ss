# DSS Update Redundancy: sci_fi_heist__f42ea382

**Log:** `scifi_heist_gemma4_s518.log`

## 1. Branch Update Generation Time Distribution

Generation time = `total - ttft` (proxy for response size since `out=0` is a logging bug)

| Stat | Value (s) |
|------|-----------|
| N | 1925 |
| Mean | 2.4 |
| Median | 0.5 |
| P25 | 0.1 |
| P75 | 3.6 |
| P95 | 8.8 |
| Max | 41.2 |

## 2. Size Buckets

| Bucket | Count | % |
|--------|-------|---|
| <2s | 1229 | 63.8% |
| 2-5s | 372 | 19.3% |
| 5-10s | 265 | 13.8% |
| 10-20s | 44 | 2.3% |
| 20-50s | 15 | 0.8% |
| >50s | 0 | 0.0% |

## 3. Trivial Updates

- **<5s generation time:** 1601/1925 = 83.2%
- These are updates where the LLM generated very little new content