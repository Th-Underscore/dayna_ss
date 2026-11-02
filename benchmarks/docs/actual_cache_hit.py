#!/usr/bin/env python3
"""Parse llama-server logs for ACTUAL prefix cache hit rate.

Uses:
  - `prompt eval time = X ms / N tokens` → actual uncached (processed) tokens
  - `release: n_tokens = N` → full prompt size
  - `n_past was set to N` → cached prefix length (when present)

Cache hit = 1 - (prompt_eval_tokens / release_n_tokens)
"""
import re

LOG = "extensions/dayna_ss/tests/runs/logs/server_gm4_stv2.log"

prompt_eval_pat = re.compile(
    r'prompt eval time =\s+([\d.]+)\s+ms\s*/\s*(\d+)\s+tokens'
)
release_pat = re.compile(
    r'release:.*n_tokens\s*=\s*(\d+)'
)
npast_pat = re.compile(
    r'n_past was set to\s+(\d+)'
)

# We need to pair prompt_eval with the NEXT release (same task)
# Track: pending prompt_eval, then match with release
calls = []
pending_eval = None
pending_npast = None

with open(LOG, 'r', errors='replace') as f:
    for line in f:
        m = npast_pat.search(line)
        if m:
            pending_npast = int(m.group(1))
            continue

        m = prompt_eval_pat.search(line)
        if m:
            t_ms = float(m.group(1))
            n_tokens = int(m.group(2))
            pending_eval = {'t_ms': t_ms, 'n_tokens': n_tokens, 'npast': pending_npast}
            pending_npast = None
            continue

        m = release_pat.search(line)
        if m and pending_eval is not None:
            full_prompt = int(m.group(1))
            pe = pending_eval
            uncached = pe['n_tokens']
            cached = full_prompt - uncached
            hit = cached / full_prompt if full_prompt > 0 else 0
            calls.append({
                'full': full_prompt,
                'uncached': uncached,
                'cached': cached,
                'hit': hit,
                't_ms': pe['t_ms'],
                'npast': pe['npast'],
            })
            pending_eval = None
            continue

print(f"Paired calls (prompt_eval + release): {len(calls)}")
if not calls:
    exit()

# Distribution of actual cache hit rate
buckets = [
    (-0.01, 0.1, "<10%"),
    (0.1, 0.3, "10-30%"),
    (0.3, 0.5, "30-50%"),
    (0.5, 0.7, "50-70%"),
    (0.7, 0.9, "70-90%"),
    (0.9, 0.95, "90-95%"),
    (0.95, 1.01, "95-100%"),
]

print(f"\n{'Cache hit':<12} {'Count':>6} {'%':>6} {'Avg full':>10} {'Avg uncached':>13} {'Avg time(s)':>12}")
print("-" * 65)
for lo, hi, label in buckets:
    in_b = [c for c in calls if lo <= c['hit'] < hi]
    if not in_b:
        continue
    n = len(in_b)
    avg_full = sum(c['full'] for c in in_b) / n
    avg_unc = sum(c['uncached'] for c in in_b) / n
    avg_t = sum(c['t_ms'] for c in in_b) / n / 1000
    pct = 100 * n / len(calls)
    print(f"{label:<12} {n:>6} {pct:>5.1f}% {avg_full:>10,.0f} {avg_unc:>13,.0f} {avg_t:>12.2f}")

# Summary
total_full = sum(c['full'] for c in calls)
total_uncached = sum(c['uncached'] for c in calls)
total_cached = sum(c['cached'] for c in calls)
total_time = sum(c['t_ms'] for c in calls) / 1000

print(f"\nFull prompt tokens:  {total_full:>12,}")
print(f"Uncached (prefill):  {total_uncached:>12,}  ({100*total_uncached/total_full:.1f}%)")
print(f"Cached (skipped):    {total_cached:>12,}  ({100*total_cached/total_full:.1f}%)")
print(f"Total prefill time:  {total_time:>12.1f}s")
print(f"Effective prefill rate (uncached only): {total_uncached/total_time:.0f} tok/s")

# Cross-check with f_keep
# How many calls have n_past data?
with_npast = [c for c in calls if c['npast'] is not None]
if with_npast:
    print(f"\nCalls with explicit n_past: {len(with_npast)}")
    avg_npast = sum(c['npast'] for c in with_npast) / len(with_npast)
    avg_full_np = sum(c['full'] for c in with_npast) / len(with_npast)
    print(f"Avg n_past: {avg_npast:,.0f}  Avg full: {avg_full_np:,.0f}  Hit: {100*avg_npast/avg_full_np:.1f}%")