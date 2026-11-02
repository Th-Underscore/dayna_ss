#!/usr/bin/env python3
"""Correlate LCP f_keep with print_timing n_tokens to measure actual cache effectiveness.

Key question: is n_tokens in print_timing the FULL prompt or the POST-CACHE count?
If post-cache: n_tokens should be small when f_keep is high.
If full prompt: n_tokens should be constant regardless of f_keep.
"""
import re
from collections import defaultdict

LOG = "extensions/dayna_ss/tests/runs/logs/server_gm4_stv2.log"

lcp_pat = re.compile(
    r'^(\d+)\.(\d+)\.(\d+)\.(\d+)\s+I slot get_availabl:.*f_keep = ([\d.]+)'
)
done_pat = re.compile(
    r'^(\d+)\.(\d+)\.(\d+)\.(\d+)\s+I slot print_timing:.*n_tokens\s*=\s*(\d+),\s+progress\s*=\s*1\.00,\s+t\s*=\s*([\d.]+)\s*s'
)

# Collect all events with relative timestamps
events = []  # (rel_time_sec, type, data)
with open(LOG, 'r', errors='replace') as f:
    for line in f:
        m = lcp_pat.match(line)
        if m:
            t = int(m.group(1))*60 + int(m.group(2)) + int(m.group(3))/1000 + int(m.group(4))/1000000
            events.append((t, 'lcp', float(m.group(5))))
            continue
        m = done_pat.match(line)
        if m:
            t = int(m.group(1))*60 + int(m.group(2)) + int(m.group(3))/1000 + int(m.group(4))/1000000
            events.append((t, 'done', (int(m.group(5)), float(m.group(6)))))

events.sort(key=lambda x: x[0])

# Pair each LCP with the next DONE event (same slot, sequential)
pairs = []
pending_lcp = None
for t, typ, data in events:
    if typ == 'lcp':
        pending_lcp = (t, data)
    elif typ == 'done' and pending_lcp is not None:
        lcp_t, f_keep = pending_lcp
        n_tokens, t_sec = data
        # Only pair if the done event is within 120s of the LCP (same call)
        if t - lcp_t < 120:
            pairs.append({'f_keep': f_keep, 'n_tokens': n_tokens, 't': t_sec})
        pending_lcp = None

print(f"Paired LCP+DONE events: {len(pairs)}")
if not pairs:
    exit()

# Bucket by f_keep and show avg n_tokens
buckets = [
    (0.0, 0.2, "0-20%"),
    (0.2, 0.4, "20-40%"),
    (0.4, 0.6, "40-60%"),
    (0.6, 0.8, "60-80%"),
    (0.8, 0.9, "80-90%"),
    (0.9, 0.95, "90-95%"),
    (0.95, 1.01, "95-100%"),
]

print(f"\n{'f_keep range':<12} {'Count':>6} {'Avg n_tokens':>14} {'Avg time(s)':>12}")
print("-" * 50)
for lo, hi, label in buckets:
    in_b = [p for p in pairs if lo <= p['f_keep'] < hi]
    if not in_b:
        continue
    n = len(in_b)
    avg_tok = sum(p['n_tokens'] for p in in_b) / n
    avg_t = sum(p['t'] for p in in_b) / n
    print(f"{label:<12} {n:>6} {avg_tok:>14,.0f} {avg_t:>12.2f}")

# The critical question: does n_tokens scale with (1 - f_keep)?
# If yes → n_tokens is post-cache. If no → n_tokens is full prompt.
import statistics
high_cache = [p for p in pairs if p['f_keep'] >= 0.9]
low_cache = [p for p in pairs if p['f_keep'] < 0.3]

if high_cache and low_cache:
    avg_high = sum(p['n_tokens'] for p in high_cache) / len(high_cache)
    avg_low = sum(p['n_tokens'] for p in low_cache) / len(low_cache)
    print(f"\nHigh cache (≥90%): avg n_tokens = {avg_high:,.0f}  (n={len(high_cache)})")
    print(f"Low cache (<30%):  avg n_tokens = {avg_low:,.0f}  (n={len(low_cache)})")
    ratio = avg_high / avg_low if avg_low else 0
    print(f"Ratio high/low: {ratio:.2f}")
    if ratio < 0.5:
        print("→ n_tokens is POST-CACHE (scales with cache hit)")
    else:
        print("→ n_tokens is FULL PROMPT (does NOT scale with cache hit)")