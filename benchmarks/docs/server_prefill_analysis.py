#!/usr/bin/env python3
"""Extract llama-server prompt processing stats to measure prefix cache hit rate.

For each completed prompt (progress=1.00), records n_tokens and time.
The ratio of n_tokens to the DSS log's `in=` token count tells us
how much of the prompt was NOT cached (had to be re-prefilled).
"""
import re
from collections import defaultdict

LOG = "extensions/dayna_ss/tests/runs/logs/server_gm4_stv2.log"

# llama-server: "X.XX.XXX.XXX I slot print_timing: id 0 | task N | prompt processing, n_tokens = NNNNN, progress = 1.00, t = XX.XX s / XXXX.XX tokens per second"
pat = re.compile(
    r'^(\d+)\.(\d+)\.(\d+)\.(\d+)\s+I slot print_timing: id\s+\d+\s+\|\s+task\s+(\d+)\s+\|\s+prompt processing,\s+n_tokens\s*=\s*(\d+),\s+progress\s*=\s*1\.00,\s+t\s*=\s*([\d.]+)\s*s\s*/\s*([\d.]+)\s*tokens per second'
)

calls = []
with open(LOG, 'r', errors='replace') as f:
    for line in f:
        m = pat.match(line)
        if m:
            rel_min = int(m.group(1))
            rel_sec = int(m.group(2))
            rel_ms = int(m.group(3))
            rel_us = int(m.group(4))
            task_id = int(m.group(5))
            n_tokens = int(m.group(6))
            t_sec = float(m.group(7))
            tok_per_s = float(m.group(8))
            # Convert relative time to absolute (server started at 00:15:55)
            total_rel_sec = rel_min * 60 + rel_sec + rel_ms / 1000.0 + rel_us / 1000000.0
            abs_sec = 15*60 + 55 + total_rel_sec
            h = int(abs_sec // 3600)
            mnt = int((abs_sec % 3600) // 60)
            s = abs_sec % 60
            abs_ts = f"{h:02d}:{mnt:02d}:{s:05.2f}"
            calls.append({
                'abs': abs_ts,
                'task': task_id,
                'n_tokens': n_tokens,
                't': t_sec,
                'tok_s': tok_per_s,
            })

print(f"Total completed prompts: {len(calls)}")
print()

# Distribution of n_tokens
buckets = [
    (0, 1000, "<1K"),
    (1000, 2000, "1-2K"),
    (2000, 3000, "2-3K"),
    (3000, 5000, "3-5K"),
    (5000, 8000, "5-8K"),
    (8000, 12000, "8-12K"),
    (12000, 20000, "12-20K"),
    (20000, 40000, "20-40K"),
    (40000, 999999, ">40K"),
]

print(f"{'n_tokens range':<12} {'Count':>6} {'%':>6} {'Avg time(s)':>12} {'Avg tok/s':>12}")
print("-" * 55)
for lo, hi, label in buckets:
    in_bucket = [c for c in calls if lo <= c['n_tokens'] < hi]
    if not in_bucket:
        continue
    n = len(in_bucket)
    avg_t = sum(c['t'] for c in in_bucket) / n
    avg_s = sum(c['tok_s'] for c in in_bucket) / n
    pct = 100 * n / len(calls)
    print(f"{label:<12} {n:>6} {pct:>5.1f}% {avg_t:>12.2f} {avg_s:>12.0f}")

# Summary stats
total_prefill_time = sum(c['t'] for c in calls)
total_tokens = sum(c['n_tokens'] for c in calls)
print(f"\nTotal prefill tokens: {total_tokens:,}")
print(f"Total prefill time: {total_prefill_time:.1f}s")
print(f"Overall prefill rate: {total_tokens/total_prefill_time:.0f} tok/s")

# Compare: if ALL prompts were ~10K tokens (the DSS in-tok average),
# total would be len(calls) * 10000. The ratio shows cache hit.
dss_in_avg = 9676  # from s522 analysis
hypothetical_no_cache = len(calls) * dss_in_avg
cache_savings = 100 * (1 - total_tokens / hypothetical_no_cache)
print(f"\nHypothetical no-cache total: {hypothetical_no_cache:,} tok")
print(f"Actual prefill total: {total_tokens:,} tok")
print(f"Prefix cache savings: {cache_savings:.0f}%")