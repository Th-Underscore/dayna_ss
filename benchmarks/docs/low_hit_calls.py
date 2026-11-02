#!/usr/bin/env python3
"""Find <10% actual cache-hit calls with timestamps, and the call before each.

Uses the absolute-timestamp server log to pair prompt_eval + release events,
computes actual cache hit, and outputs timestamps for oaiproxy lookup.
"""
import re

LOG = "extensions/dayna_ss/tests/runs/logs/server_gm4_stv2_abs.log"

prompt_eval_pat = re.compile(
    r'^\[(\d{2}:\d{2}:\d{2}\.\d{3})\].*prompt eval time =\s+([\d.]+)\s+ms\s*/\s*(\d+)\s+tokens'
)
release_pat = re.compile(
    r'^\[(\d{2}:\d{2}:\d{2}\.\d{3})\].*release:.*n_tokens\s*=\s*(\d+)'
)
lcp_pat = re.compile(
    r'^\[(\d{2}:\d{2}:\d{2}\.\d{3})\].*f_keep = ([\d.]+)'
)

def ts_to_sec(ts):
    h, m, s = ts.split(':')
    return int(h)*3600 + int(m)*60 + float(s)

# Collect events
pending_eval = None
calls = []
lcp_events = []

with open(LOG, 'r', errors='replace') as f:
    for line in f:
        m = lcp_pat.match(line)
        if m:
            lcp_events.append((m.group(1), float(m.group(2))))
            continue
        m = prompt_eval_pat.search(line)
        if m:
            pending_eval = {'ts': m.group(1), 'n_tokens': int(m.group(3))}
            continue
        m = release_pat.search(line)
        if m and pending_eval:
            full = int(m.group(2))
            uncached = pending_eval['n_tokens']
            hit = 1 - (uncached / full) if full > 0 else 0
            calls.append({
                'ts': pending_eval['ts'],
                'full': full,
                'uncached': uncached,
                'hit': hit,
            })
            pending_eval = None

# Find <10% hits
low_hits = [c for c in calls if c['hit'] < 0.10]
print(f"Total calls: {len(calls)}")
print(f"<10% cache hit: {len(low_hits)}")
print()

# For each low hit, find the call immediately before it
print(f"{'#':>3} {'Time':<14} {'Full':>8} {'Uncached':>9} {'Hit%':>6} | {'Prev time':<14} {'Prev full':>8} {'Prev hit%':>9}")
print("-" * 85)

for i, c in enumerate(low_hits[:20]):
    # Find the call just before this one
    idx = calls.index(c)
    if idx > 0:
        prev = calls[idx - 1]
        print(f"{i:>3} {c['ts']:<14} {c['full']:>8,} {c['uncached']:>9,} {c['hit']*100:>5.1f}% | {prev['ts']:<14} {prev['full']:>8,} {prev['hit']*100:>8.1f}%")
    else:
        print(f"{i:>3} {c['ts']:<14} {c['full']:>8,} {c['uncached']:>9,} {c['hit']*100:>5.1f}% | (no prev)")

# Also: for the first 5 low hits, find the nearest LCP event
print()
print("Nearest LCP f_keep for first 5 low-hit calls:")
for i, c in enumerate(low_hits[:5]):
    c_sec = ts_to_sec(c['ts'])
    # Find closest LCP event
    best = None
    best_diff = float('inf')
    for lts, fk in lcp_events:
        diff = abs(ts_to_sec(lts) - c_sec)
        if diff < best_diff:
            best_diff = diff
            best = (lts, fk)
    if best:
        print(f"  {i}: call at {c['ts']}, LCP at {best[0]} (f_keep={best[1]:.3f}, gap={best_diff:.1f}s)")