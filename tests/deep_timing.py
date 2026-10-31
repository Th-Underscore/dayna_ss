#!/usr/bin/env python3
"""Deep per-subject, per-action timing analysis for soak runs.

Primary input: <run_dir>/dss_calls.jsonl (one JSON line per LLM call).
Falls back to plaintext log parsing if no JSONL is found.

Usage:
  python3 deep_timing.py <run_dir_or_log_path>

  If the path contains a dss_calls.jsonl, uses that (preferred).
  If the path is a .log file, uses the old plaintext parser.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# Subjects that are NOT DSS (writer / setup)
WRITER_SUBJECTS = {'reply', 'instr_prompt'}
SETUP_PREFIXES = ('message_summary', 'scene_detection', 'legacy', 'first_scene_population')


def is_dss(subject: str) -> bool:
    if subject in WRITER_SUBJECTS:
        return False
    for p in SETUP_PREFIXES:
        if subject.startswith(p):
            return False
    return True


def top_subject(subject: str) -> str:
    """Extract the top-level subject from a phase_id path.

    Examples:
      current_scene.now.who.characters[0]. -> current_scene
      elements.entries.emergency_access_panel -> elements
      characters.entries.dex -> characters
      events.beats -> events
    """
    # Strip trailing dots
    s = subject.rstrip('.')
    # Handle [N] indices in the path
    # Take the first segment before any dot or bracket
    m = re.match(r'^([a-z_]+)', s)
    if m:
        return m.group(1)
    return s.split('.')[0]


def load_jsonl(path: Path) -> list[dict]:
    """Load dss_calls.jsonl into a list of records."""
    records = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def load_from_log(log_path: str) -> list[dict]:
    """Fallback: parse the plaintext log (old format). Returns same record shape."""
    re_call = re.compile(
        r'local-call (\S+)/(\S+):\s+ttft=([\d.]+|—)s\s+total=([\d.]+)s\s+in=(\d+)\s+out=(\d+)(?:\s+cached=(\d+))?'
    )
    re_turn = re.compile(r'--- turn (\d+)/(\d+) ---')

    records = []
    current_turn = -1
    with open(log_path, 'r', errors='replace') as f:
        for line in f:
            m = re_turn.search(line)
            if m:
                current_turn = int(m.group(1))
                continue
            m = re_call.search(line)
            if m and current_turn >= 0:
                ttft_str = m.group(3)
                records.append({
                    'turn': current_turn,
                    'subject': m.group(1),
                    'action': m.group(2),
                    'ttft': float(ttft_str) if ttft_str != '—' else None,
                    'total': float(m.group(4)),
                    'in': int(m.group(5)),
                    'out': int(m.group(6)),
                    'cached': int(m.group(7) or 0),
                })
    return records


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <run_dir_or_log_path>")
        sys.exit(1)

    path = Path(sys.argv[1])

    # Determine input source
    if path.is_dir():
        jsonl = path / 'dss_calls.jsonl'
        if jsonl.exists():
            records = load_jsonl(jsonl)
            source = str(jsonl)
        else:
            # Try to find a log file
            log_dir = path / 'logs'
            logs = list(log_dir.glob('*.log')) if log_dir.exists() else []
            if not logs:
                print(f"No dss_calls.jsonl or .log file found in {path}")
                sys.exit(1)
            records = load_from_log(str(logs[0]))
            source = str(logs[0])
    elif path.suffix == '.jsonl':
        records = load_jsonl(path)
        source = str(path)
    else:
        # Assume .log file
        records = load_from_log(str(path))
        source = str(path)

    if not records:
        print("No records found.")
        return

    # Separate DSS from writer
    dss_records = [r for r in records if is_dss(r['subject'])]
    writer_records = [r for r in records if not is_dss(r['subject'])]

    # Add top-level subject classification
    for r in dss_records:
        r['top'] = top_subject(r['subject'])
    for r in writer_records:
        r['top'] = r['subject']

    # Turn numbers
    all_turns = sorted(set(r['turn'] for r in records))
    dss_turns = sorted(set(r['turn'] for r in dss_records))

    # Scene detection: >20 DSS calls in a turn
    dss_per_turn = defaultdict(int)
    for r in dss_records:
        dss_per_turn[r['turn']] += 1
    scene_turns = {t for t, n in dss_per_turn.items() if n > 20}
    nonscene_turns = {t for t, n in dss_per_turn.items() if n <= 20}

    # === SUMMARY ===
    print(f"\n{'=' * 100}")
    print(f"  DEEP TIMING ANALYSIS: {source}")
    print(f"  {len(all_turns)} turns total")
    print(f"{'=' * 100}")

    dss_time = sum(r['total'] for r in dss_records)
    writer_time = sum(r['total'] for r in writer_records)
    total_time = dss_time + writer_time

    print(f"\n  DSS calls: {len(dss_records)}  |  Writer/other calls: {len(writer_records)}")
    print(f"  DSS time: {dss_time:.0f}s  |  Writer time: {writer_time:.0f}s  |  Total: {total_time:.0f}s")
    if total_time:
        print(f"  DSS share: {dss_time/total_time*100:.1f}%")

    # Cached tokens summary
    total_cached = sum(r.get('cached', 0) for r in dss_records)
    total_in = sum(r['in'] for r in dss_records)
    if total_in:
        print(f"  Prefix cache: {total_cached:,} / {total_in:,} tokens cached ({total_cached/total_in*100:.1f}%)")

    # === PER-SUBJECT AGGREGATION ===
    for label, tset in [("ALL TURNS", set(dss_turns)),
                        ("NON-SCENE TURNS", nonscene_turns),
                        ("SCENE TURNS", scene_turns)]:
        if not tset:
            continue

        subj_stats = defaultdict(lambda: {'calls': 0, 'time': 0, 'prefill': 0, 'decode': 0, 'in_tok': 0})
        action_stats = defaultdict(lambda: {'calls': 0, 'time': 0, 'prefill': 0, 'decode': 0, 'in_tok': 0})
        subj_action = defaultdict(lambda: defaultdict(lambda: {'calls': 0, 'time': 0}))

        for r in dss_records:
            if r['turn'] not in tset:
                continue
            s = r['top']
            a = r['action']
            t = r['total']
            p = r['ttft'] if r['ttft'] is not None else 0
            d = t - p
            i = r['in']

            subj_stats[s]['calls'] += 1
            subj_stats[s]['time'] += t
            subj_stats[s]['prefill'] += p
            subj_stats[s]['decode'] += d
            subj_stats[s]['in_tok'] += i

            action_stats[a]['calls'] += 1
            action_stats[a]['time'] += t
            action_stats[a]['prefill'] += p
            action_stats[a]['decode'] += d
            action_stats[a]['in_tok'] += i

            subj_action[s][a]['calls'] += 1
            subj_action[s][a]['time'] += t

        group_total = sum(v['time'] for v in subj_stats.values())

        print(f"\n{'─' * 100}")
        print(f"  {label} ({len(tset)} turns)")
        print(f"{'─' * 100}")

        # Subject table
        print(f"\n  {'Subject':<18} {'Calls':>6} {'Time (s)':>10} {'%':>6} {'Avg/call':>9} {'Prefill':>9} {'Decode':>9} {'Avg in':>8}")
        print(f"  {'-' * 82}")
        for s in sorted(subj_stats.keys(), key=lambda x: -subj_stats[x]['time']):
            v = subj_stats[s]
            pct = v['time'] / group_total * 100 if group_total else 0
            avg = v['time'] / v['calls'] if v['calls'] else 0
            avg_in = v['in_tok'] // v['calls'] if v['calls'] else 0
            print(f"  {s:<18} {v['calls']:>6} {v['time']:>10.1f} {pct:>5.1f}% {avg:>9.2f} "
                  f"{v['prefill']:>9.1f} {v['decode']:>9.1f} {avg_in:>8,}")

        # Action table
        print(f"\n  {'Action':<18} {'Calls':>6} {'Time (s)':>10} {'%':>6} {'Avg/call':>9} {'Avg in':>8}")
        print(f"  {'-' * 62}")
        for a in sorted(action_stats.keys(), key=lambda x: -action_stats[x]['time']):
            v = action_stats[a]
            pct = v['time'] / group_total * 100 if group_total else 0
            avg = v['time'] / v['calls'] if v['calls'] else 0
            avg_in = v['in_tok'] // v['calls'] if v['calls'] else 0
            print(f"  {a:<18} {v['calls']:>6} {v['time']:>10.1f} {pct:>5.1f}% {avg:>9.2f} {avg_in:>8,}")

        # Subject × Action matrix (top subjects only)
        top_subjects = sorted(subj_stats.keys(), key=lambda x: -subj_stats[x]['time'])[:8]
        all_actions = sorted(action_stats.keys(), key=lambda x: -action_stats[x]['time'])

        print(f"\n  Subject × Action (time in s):")
        header = f"  {'':>18}" + "".join(f" {a[:12]:>13}" for a in all_actions) + f" {'TOTAL':>10}"
        print(header)
        print(f"  {'-' * (18 + 14 * len(all_actions) + 10)}")
        for s in top_subjects:
            row = f"  {s:>18}"
            for a in all_actions:
                v = subj_action[s][a]
                if v['calls'] > 0:
                    row += f" {v['time']:>13.1f}"
                else:
                    row += f" {'·':>13}"
            total_s = subj_stats[s]['time']
            row += f" {total_s:>10.1f}"
            print(row)

    # Per-turn detail
    print(f"\n{'=' * 100}")
    print(f"  PER-TURN DETAIL")
    print(f"{'=' * 100}")
    print(f"\n  {'Turn':>5} {'Scn':>4} {'Calls':>6} {'LLM (s)':>9} {'Prefill':>9} {'Decode':>9} "
          f"{'In-tok':>10} {'Cached':>8} {'Top subject':>18} {'Top action':>14}")
    print(f"  {'-' * 96}")

    for tn in all_turns:
        dss = [r for r in dss_records if r['turn'] == tn]
        if not dss:
            continue
        n = len(dss)
        t = sum(r['total'] for r in dss)
        p = sum(r['ttft'] for r in dss if r['ttft'] is not None)
        d = t - p
        i = sum(r['in'] for r in dss)
        cached = sum(r.get('cached', 0) for r in dss)
        scn = 'Y' if tn in scene_turns else ''

        st = defaultdict(float)
        for r in dss:
            st[r['top']] += r['total']
        top_s = max(st, key=lambda k: st[k]) if st else '?'

        at = defaultdict(float)
        for r in dss:
            at[r['action']] += r['total']
        top_a = max(at, key=lambda k: at[k]) if at else '?'

        print(f"  {tn:>5} {scn:>4} {n:>6} {t:>9.1f} {p:>9.1f} {d:>9.1f} {i:>10,} {cached:>8,} {top_s:>18} {top_a:>14}")

    # === PREFIX CACHE ANALYSIS ===
    print(f"\n{'=' * 100}")
    print(f"  PREFIX CACHE ANALYSIS (DSS calls only)")
    print(f"{'=' * 100}")

    # If we have actual cached_tokens data, use it
    has_cached = any(r.get('cached', 0) > 0 for r in dss_records)

    if has_cached:
        print(f"\n  Using actual cached_tokens from vLLM:")
        total_cached = sum(r.get('cached', 0) for r in dss_records)
        total_in = sum(r['in'] for r in dss_records)
        print(f"  Overall: {total_cached:,} / {total_in:,} = {total_cached/total_in*100:.1f}%")

        # TTFT by prompt size (with actual cache data)
        print(f"\n  TTFT by prompt size (actual cache):")
        print(f"  {'In-tok range':<16} {'N':>5} {'TTFT p50':>10} {'Avg cached%':>12} {'Avg TTFT':>10}")
        print(f"  {'-' * 60}")
        for lo, hi in [(3000, 6000), (6000, 8000), (8000, 10000), (10000, 12000),
                        (12000, 15000), (15000, 20000), (20000, 30000), (30000, 50000)]:
            subset = [r for r in dss_records if lo <= r['in'] < hi]
            if not subset:
                continue
            n = len(subset)
            ttfts = sorted(r['ttft'] for r in subset if r['ttft'] is not None)
            p50 = ttfts[int(n * 0.5)] if ttfts else 0
            avg_cached_pct = sum(r.get('cached', 0) / r['in'] * 100 for r in subset if r['in'] > 0) / n
            avg_ttft = sum(r['ttft'] for r in subset if r['ttft'] is not None) / n if ttfts else 0
            print(f"  {lo // 1000}-{hi // 1000}K{'':<10} {n:>5} {p50:>10.2f} {avg_cached_pct:>10.1f}% {avg_ttft:>10.2f}")

        # Cache hit by turn
        print(f"\n  Cache hit by turn:")
        print(f"  {'Turn':>6} {'Calls':>6} {'Avg cached%':>12} {'Avg TTFT':>10}")
        print(f"  {'-' * 40}")
        for tn in all_turns:
            dss = [r for r in dss_records if r['turn'] == tn and r['in'] > 0]
            if not dss:
                continue
            avg_pct = sum(r.get('cached', 0) / r['in'] * 100 for r in dss) / len(dss)
            avg_ttft = sum(r['ttft'] for r in dss if r['ttft'] is not None) / len(dss)
            print(f"  {tn:>6} {len(dss):>6} {avg_pct:>10.1f}% {avg_ttft:>10.2f}")
    else:
        # Fallback: estimate from TTFT
        print(f"\n  No cached_tokens data; estimating from TTFT (COLD_RATE=2000 tok/s)")
        COLD_RATE = 2000
        print(f"\n  TTFT by prompt size:")
        print(f"  {'In-tok range':<16} {'N':>5} {'TTFT p50':>10} {'Implied hit p50':>16}")
        print(f"  {'-' * 55}")
        for lo, hi in [(3000, 6000), (6000, 8000), (8000, 10000), (10000, 12000),
                        (12000, 15000), (15000, 20000), (20000, 30000), (30000, 50000)]:
            subset = sorted(r['ttft'] for r in dss_records if lo <= r['in'] < hi and r['ttft'] is not None)
            if not subset:
                continue
            n = len(subset)
            p50 = subset[int(n * 0.5)]
            avg_in = (lo + hi) // 2
            hit_p50 = max(0, 1 - (p50 * COLD_RATE) / avg_in) * 100
            print(f"  {lo // 1000}-{hi // 1000}K{'':<10} {n:>5} {p50:>10.2f} {hit_p50:>14.1f}%")


if __name__ == '__main__':
    main()