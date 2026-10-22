#!/usr/bin/env python3
"""Measure per-subject processing time from timestamped soak log.

Uses 'Processing Schema Node' lines as markers. Duration of a node =
time until the next 'Processing Schema Node' line (or turn boundary).
Aggregates hierarchically: total time per top-level subject.
"""
import re
import sys
from collections import defaultdict
from datetime import datetime

def parse_ts(line):
    """Extract timestamp from a log line. Returns datetime or None."""
    # logtime format: [HH:MM:SS.mmm] at start
    m = re.match(r'^\[(\d{2}):(\d{2}):(\d{2})(?:\.\d+)?\]', line)
    if m:
        return datetime(2026, 1, 1, int(m.group(1)), int(m.group(2)), int(m.group(3)))
    # soak's own format: [2026-10-22 HH:MM:SS]
    m = re.search(r'\[\d{4}-\d{2}-\d{2} (\d{2}):(\d{2}):(\d{2})\]', line)
    if m:
        return datetime(2026, 1, 1, int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return None

def main():
    log_path = sys.argv[1] if len(sys.argv) > 1 else "runs/logs/scifi_heist_gemma4_s521.log"
    
    turn_pat = re.compile(r'--- turn (\d+)/\d+ ---')
    node_pat = re.compile(r"Processing Schema Node: \S+ \(\w+\) at '([^']+)'")
    call_pat = re.compile(r'local-call \S+/generate.*?ttft=([\d.]+)s total=([\d.]+)s in=(\d+) out=(\d+)')
    
    # Per turn: list of (timestamp, node_path)
    turns = defaultdict(list)
    # Per turn: list of (ttft, total, in_tok, out_tok) for each LLM call
    calls = defaultdict(list)
    current_turn = -1
    last_ts = None
    
    with open(log_path, 'r', errors='replace') as f:
        for line in f:
            ts = parse_ts(line)
            if ts:
                last_ts = ts
            
            m = turn_pat.search(line)
            if m:
                current_turn = int(m.group(1))
                continue
            
            if current_turn < 0:
                continue
            
            m = node_pat.search(line)
            if m and last_ts:
                turns[current_turn].append((last_ts, m.group(1)))
                continue
            
            m = call_pat.search(line)
            if m:
                calls[current_turn].append({
                    'ttft': float(m.group(1)),
                    'total': float(m.group(2)),
                    'in': int(m.group(3)),
                    'out': int(m.group(4))
                })
    
    if not turns:
        print("No turns found. Check log path.")
        return
    
    # Calculate durations: each node's duration = next node's start - this node's start
    # (for the last node in a turn, we can't measure — mark as None)
    
    # Aggregate: for each turn, sum durations by top-level subject
    # Top-level = first component of the path
    results = {}  # turn_num -> {subject: total_seconds}
    
    for tn, nodes in turns.items():
        if len(nodes) < 2:
            continue
        subj_time = defaultdict(float)
        for i in range(len(nodes) - 1):
            t_start, path_start = nodes[i]
            t_end, _ = nodes[i + 1]
            dur = (t_end - t_start).total_seconds()
            if dur < 0:
                dur = 0  # clock anomaly
            top = path_start.split('.')[0]
            subj_time[top] += dur
        results[tn] = subj_time
    
    # Summary: average time per subject across all turns
    all_subjects = defaultdict(list)
    for tn, subj_time in results.items():
        for subj, dur in subj_time.items():
            all_subjects[subj].append((tn, dur))
    
    print(f"{'Subject':<20} {'Turns':>6} {'Avg (s)':>9} {'Total (s)':>10} {'% of turn':>10}")
    print("-" * 60)
    
    # Total time per turn (sum of all subjects)
    turn_totals = {tn: sum(st.values()) for tn, st in results.items()}
    grand_total = sum(turn_totals.values())
    
    for subj in sorted(all_subjects.keys(), key=lambda s: -sum(d for _, d in all_subjects[s])):
        entries = all_subjects[subj]
        avg = sum(d for _, d in entries) / len(entries)
        total = sum(d for _, d in entries)
        pct = (total / grand_total * 100) if grand_total > 0 else 0
        print(f"{subj:<20} {len(entries):>6} {avg:>9.2f} {total:>10.1f} {pct:>9.1f}%")
    
    print(f"\nGrand total (measured): {grand_total:.0f}s across {len(results)} turns")
    print(f"Avg turn processing time: {grand_total/len(results):.1f}s")
    
    # Per-turn detail: show first 10 turns
    print(f"\n\n=== PER-TURN BREAKDOWN (first 12 turns) ===")
    print(f"{'Turn':>5} {'Total':>8} ", end="")
    # Get consistent subject order
    top_subjects = sorted(all_subjects.keys(), key=lambda s: -sum(d for _, d in all_subjects[s]))
    for s in top_subjects:
        print(f" {s[:8]:>9}", end="")
    print()
    print("-" * (8 + 5 + 10 * len(top_subjects)))
    
    for tn in sorted(results.keys())[:12]:
        st = results[tn]
        total = sum(st.values())
        print(f"{tn:>5} {total:>7.1f}s ", end="")
        for s in top_subjects:
            v = st.get(s, 0)
            print(f" {v:>8.1f}", end="")
        print()
    
    # Prefill vs decode breakdown
    print(f"\n\n=== PREFILL vs DECODE BREAKDOWN ===")
    total_prefill = 0
    total_decode = 0
    total_in = 0
    total_out = 0
    total_calls = 0
    per_turn_prefill = defaultdict(float)
    per_turn_decode = defaultdict(float)
    for tn, clist in calls.items():
        for c in clist:
            prefill = c['ttft']
            decode = c['total'] - c['ttft']
            total_prefill += prefill
            total_decode += decode
            total_in += c['in']
            total_out += c['out']
            total_calls += 1
            per_turn_prefill[tn] += prefill
            per_turn_decode[tn] += decode
    
    grand_call_time = total_prefill + total_decode
    print(f"Total LLM calls: {total_calls}")
    print(f"Total in-tokens: {total_in:,}")
    print(f"Total out-tokens: {total_out:,}")
    print(f"Total prefill (ttft): {total_prefill:.0f}s ({total_prefill/grand_call_time*100:.0f}%)")
    print(f"Total decode: {total_decode:.0f}s ({total_decode/grand_call_time*100:.0f}%)")
    print(f"Total LLM time: {grand_call_time:.0f}s")
    print(f"Avg prefill/call: {total_prefill/total_calls:.2f}s")
    print(f"Avg decode/call: {total_decode/total_calls:.2f}s")
    print(f"Avg in-tok/call: {total_in/total_calls:,.0f}")
    print(f"Avg out-tok/call: {total_out/total_calls:,.0f}")
    
    # Per-turn prefill/decode for first 12 turns
    print(f"\n{'Turn':>5} {'Calls':>6} {'Prefill':>9} {'Decode':>9} {'In-tok':>8} {'Out-tok':>8}")
    print("-" * 50)
    for tn in sorted(calls.keys())[:12]:
        clist = calls[tn]
        p = sum(c['ttft'] for c in clist)
        d = sum(c['total'] - c['ttft'] for c in clist)
        i_t = sum(c['in'] for c in clist)
        o_t = sum(c['out'] for c in clist)
        print(f"{tn:>5} {len(clist):>6} {p:>8.1f}s {d:>8.1f}s {i_t:>8,} {o_t:>8,}")
    
    # Show a detailed tree for one turn
    if len(sys.argv) > 2:
        detail_turn = int(sys.argv[2])
        if detail_turn in turns:
            print(f"\n\n=== DETAILED NODE TIMELINE: TURN {detail_turn} ===")
            nodes = turns[detail_turn]
            for i in range(len(nodes) - 1):
                t_start, path = nodes[i]
                t_end, _ = nodes[i + 1]
                dur = (t_end - t_start).total_seconds()
                print(f"  {t_start.strftime('%H:%M:%S')}  {dur:>7.2f}s  {path}")
            # Last node
            t_start, path = nodes[-1]
            print(f"  {t_start.strftime('%H:%M:%S')}       ??  {path}  (last, unmeasured)")

if __name__ == '__main__':
    main()