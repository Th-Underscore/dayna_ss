# DSS Call Profile: sci_fi_heist__f42ea382

**Log:** `scifi_heist_gemma4_s518.log` (951K lines)

**Total local calls:** 4406

**Total LLM time:** 85749s (23.8h)

## 1. Call Category Breakdown

| Category | N | Mean (s) | Median (s) | P95 (s) | Max (s) | Total (s) | % Time |
|----------|---|----------|------------|---------|---------|-----------|--------|
| dss_other | 1926 | 16.3 | 11.8 | 45.6 | 118.7 | 31454 | 36.7% |
| dss_element | 465 | 23.9 | 21.2 | 51.7 | 126.3 | 11106 | 13.0% |
| gate_check | 404 | 26.8 | 20.9 | 56.4 | 64.2 | 10836 | 12.6% |
| dss_character | 158 | 36.3 | 33.0 | 59.0 | 119.7 | 5741 | 6.7% |
| dss_storyevents | 181 | 29.8 | 29.8 | 55.6 | 91.8 | 5388 | 6.3% |
| dss_scenecharacter | 184 | 20.5 | 22.5 | 41.8 | 77.2 | 3777 | 4.4% |
| dss_currentscene | 61 | 43.0 | 50.9 | 58.2 | 63.4 | 2621 | 3.1% |
| dss_generalinfo | 138 | 17.6 | 10.6 | 53.8 | 88.0 | 2434 | 2.8% |
| writer | 151 | 15.6 | 15.9 | 18.2 | 45.8 | 2359 | 2.8% |
| dss_importance | 171 | 12.3 | 8.6 | 39.2 | 70.1 | 2104 | 2.5% |
| dss_arcs | 93 | 21.5 | 23.6 | 36.5 | 48.2 | 2001 | 2.3% |
| dss_scenewhen | 102 | 10.6 | 8.4 | 30.8 | 43.6 | 1080 | 1.3% |
| dss_arc | 57 | 18.3 | 13.4 | 43.3 | 100.4 | 1046 | 1.2% |
| dss_scenewhy | 101 | 10.0 | 2.3 | 30.4 | 65.7 | 1010 | 1.2% |
| dss_scenegroup | 84 | 11.6 | 7.4 | 32.8 | 56.3 | 971 | 1.1% |
| dss_sceneelement | 47 | 10.3 | 1.4 | 34.4 | 47.2 | 486 | 0.6% |
| dss_elementmap | 32 | 15.0 | 17.7 | 19.2 | 19.9 | 479 | 0.6% |
| dss_groupmap | 23 | 13.4 | 15.6 | 19.1 | 19.9 | 308 | 0.4% |
| dss_characterrelationships | 10 | 22.1 | 20.3 | 45.5 | 45.5 | 221 | 0.3% |
| dss_charactermap | 13 | 12.5 | 11.3 | 19.7 | 19.7 | 162 | 0.2% |
| dss_group | 4 | 38.5 | 43.6 | 56.0 | 56.0 | 154 | 0.2% |
| dss_timepoint | 1 | 10.4 | 10.4 | 10.4 | 10.4 | 10 | 0.0% |

## 2. DSS Per-Subject Breakdown

| Subject | N | Mean (s) | Total (s) | % of DSS |
|---------|---|----------|-----------|----------|
| other | 1926 | 16.3 | 31454 | 37.7% |
| element | 465 | 23.9 | 11106 | 13.3% |
| character | 158 | 36.3 | 5741 | 6.9% |
| storyevents | 181 | 29.8 | 5388 | 6.5% |
| scenecharacter | 184 | 20.5 | 3777 | 4.5% |
| currentscene | 61 | 43.0 | 2621 | 3.1% |
| generalinfo | 138 | 17.6 | 2434 | 2.9% |
| importance | 171 | 12.3 | 2104 | 2.5% |
| arcs | 93 | 21.5 | 2001 | 2.4% |
| scenewhen | 102 | 10.6 | 1080 | 1.3% |
| arc | 57 | 18.3 | 1046 | 1.3% |
| scenewhy | 101 | 10.0 | 1010 | 1.2% |
| scenegroup | 84 | 11.6 | 971 | 1.2% |
| sceneelement | 47 | 10.3 | 486 | 0.6% |
| elementmap | 32 | 15.0 | 479 | 0.6% |
| groupmap | 23 | 13.4 | 308 | 0.4% |
| characterrelationships | 10 | 22.1 | 221 | 0.3% |
| charactermap | 13 | 12.5 | 162 | 0.2% |
| group | 4 | 38.5 | 154 | 0.2% |
| timepoint | 1 | 10.4 | 10 | 0.0% |

## 3. Gate Check Verdicts

| Subject | YES | NO | Retry | YES Rate |
|---------|-----|----|-------|----------|
| elements.entries | 124 | 27 | 0 | 82.1% |
| general_info.objectives[0] | 34 | 116 | 0 | 22.7% |
| characters.entries | 45 | 16 | 0 | 73.8% |
| groups.entries | 4 | 57 | 0 | 6.6% |

## 4. Discovery Results

| Subject | Empty | Found | Empty Rate |
|---------|-------|-------|------------|
| unknown | 101 | 0 | 100.0% |

## 5. Per-Turn Call Counts

| Stat | Value |
|------|-------|
| Turns | 150 |
| Mean | 29.4 |
| Median | 21 |
| P95 | 57 |
| Max | 144 |

## 6. Top 10 Slowest Calls

| # | Turn | Category | Duration (s) | Prompt Tokens |
|---|------|----------|-------------|---------------|
| 1 | 110 | dss_element | 126.3 | 16473 |
| 2 | 110 | dss_character | 119.7 | 20758 |
| 3 | 110 | dss_other | 118.7 | 25210 |
| 4 | 110 | dss_other | 101.4 | 23216 |
| 5 | 147 | dss_arc | 100.4 | 24099 |
| 6 | 147 | dss_other | 97.7 | 40308 |
| 7 | 129 | dss_element | 96.6 | 34777 |
| 8 | 110 | dss_character | 93.8 | 23751 |
| 9 | 144 | dss_other | 92.8 | 36505 |
| 10 | 144 | dss_element | 92.0 | 28368 |

## 7. Key Findings

- DSS is 97% of total LLM time (83389s of 85749s)
- Writer is 151 calls totaling 2359s (2.8%)
- Gate checks: 423 total, 216 returned NO/unchanged (51% wasted)