#!/usr/bin/env python3
"""Correlate LLM calls to turns. Check if <10% cache hits only happen at turn boundaries."""
import re
from datetime import datetime

SOAK_LOG = "extensions/dayna_ss/tests/runs/logs/scifi_heist_gemma4_s522.log"
SERVER_LOG = "extensions/dayna_ss/tests/runs/logs/server_gm4_stv2_abs.log"

def ts_to_sec(ts):
    h, m, s = ts.split(':')
    return int(h)*3600 + int(m)*60 + float(s)

# 1. Get turn boundaries from soak log
turn_pat = re.compile(r'^\[(\d{2}:\d{2}:\d{2}\.\d{3})\].*--- turn (\d+)/\d+ ---')
turn_starts = []  # (sec, turn_number)
with open(SOAK_LOG, 'r', errors='replace') as f:
    for line in f:
        m = turn_pat.match(line)
        if m:
            turn_starts.append((ts_to_sec(m.group(1)), int(m.group(2))))

print(f"Turn boundaries found: {len(turn_starts)}")
print(f"First turn start: {turn_starts[0][1]} at {datetime.fromtimestamp(turn_starts[0][0]+datetime.timestamp(datetime(2026,10,23)))}")
print()

# 2. Get all server calls with cache hit rates
prompt_eval_pat = re.compile(
    r'^\[(\d{2}:\d{2}:\d{2}\.\d{3})\].*prompt eval time =\s+([\d.]+)\s+ms\s*/\s*(\d+)\s+tokens'
)
release_pat = re.compile(
    r'^\[(\d{2}:\d{2}:\d{2}\.\d{3})\].*release:.*n_tokens\s*=\s*(\d+)'
)

pending_eval = None
calls = []  # (sec, full, uncached, hit)
with open(SERVER_LOG, 'r', errors='replace') as f:
    for line in f:
        m = prompt_eval_pat.search(line)
        if m:
            pending_eval = (ts_to_sec(m.group(1)), int(m.group(3)))
            continue
        m = release_pat.search(line)
        if m and pending_eval:
            t_sec, uncached = pending_eval
            full = int(m.group(2))
            hit = 1 - (uncached / full) if full > 0 else 0
            calls.append((t_sec, full, uncached, hit))
            pending_eval = None

print(f"Server calls: {len(calls)}")

# 3. For each call, determine which turn it belongs to
# A call belongs to the turn whose start is immediately before it
def get_turn(t_sec):
    # Find the last turn start that is <= t_sec
    turn = 0
    for ts, tn in turn_starts:
        if ts <= t_sec:
            turn = tn
        else:
            break
    return turn

# 4. Analyze: for each turn, what's the cache hit pattern?
# Focus on: is the FIRST call of each turn the low-hit one?
turn_data = {}  # turn_num -> list of (call_idx_in_turn, hit)
for i, (t_sec, full, uncached, hit) in enumerate(calls):
    tn = get_turn(t_sec)
    if tn not in turn_data:
        turn_data[tn] = []
    turn_data[tn].append((len(turn_data[tn]), hit, full, uncached))

# 5. Report: for each turn, show the hit rate of each call
# Focus on turns with 3+ calls to see the pattern
print(f"{'Turn':>4} {'#Calls':>6} {'1st':>6} {'2nd':>6} {'3rd':>6} {'4th':>6} {'5th':>6} {'6th':>6} {'Mean':>6} {'Min':>6}")
print("-" * 75)

low_hit_first = 0
low_hit_nonfirst = 0
total_first = 0
total_nonfirst = 0

for tn in sorted(turn_data.keys()):
    calls_in_turn = turn_data[tn]
    if len(calls_in_turn) < 2:
        continue
    
    hits = [c[1] for c in calls_in_turn]
    n = len(hits)
    
    # Show first 6 calls
    parts = []
    for i in range(min(6, n)):
        parts.append(f"{hits[i]*100:>5.0f}%")
    while len(parts) < 6:
        parts.append("     ")
    
    mean = sum(hits)/n
    mn = min(hits)
    
    print(f"{tn:>4} {n:>6} {' '.join(parts)} {mean*100:>5.0f}% {mn*100:>5.0f}%")
    
    # Track: is the first call the low one?
    total_first += 1
    total_nonfirst += n - 1
    if hits[0] < 0.10:
        low_hit_first += 1
    for h in hits[1:]:
        if h < 0.10:
            low_hit_nonfirst += 1

print()
print(f"First calls per turn: {total_first}, of which <10% hit: {low_hit_first} ({100*low_hit_first/total_first:.0f}%)")
print(f"Non-first calls: {total_nonfirst}, of which <10% hit: {low_hit_nonfirst} ({100*low_hit_nonfirst/total_nonfirst:.0f}%)")