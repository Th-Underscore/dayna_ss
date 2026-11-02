#!/usr/bin/env python3
"""Consolidated analysis: server-log timing + oaiproxy full prompts.

Parses the llama-server log for per-call timing (prompt eval, n_tokens, f_keep)
and cross-references with oaiproxy to get the actual prompt content.
Correlates the two by timestamp proximity.
"""

import re
import json
import urllib.request
import sys
from datetime import datetime, timedelta

SERVER_LOG = "extensions/dayna_ss/tests/runs/logs/server_gm4_stv2_swa_full.log"
OAIPROXY = "http://localhost:5001/__dashboard/api"

# Server started at 11:36:13 (from the log)
SERVER_START_H = 11
SERVER_START_M = 36
SERVER_START_S = 13

def server_time_to_abs(sec_frac: float) -> str:
    """Convert server-relative time (e.g. 605.794.470) to HH:MM:SS."""
    total_s = int(sec_frac)
    h = SERVER_START_H + (total_s // 3600)
    m = SERVER_START_M + ((total_s % 3600) // 60)
    s = SERVER_START_S + (total_s % 60)
    # normalize
    s += (m % 60)
    m = m // 60 + s // 60
    s = s % 60
    h += m // 60
    m = m % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def parse_server_log(path: str):
    """Extract per-call timing data from the llama-server log."""
    prompt_eval_pat = re.compile(r'prompt eval time =\s+([\d.]+)\s+ms\s*/\s*(\d+)\s+tokens')
    release_pat = re.compile(r'release:.*n_tokens\s*=\s*(\d+)')
    lcp_pat = re.compile(
        r'^(\d+)\.(\d{2})\.(\d{3})\.(\d{3})\s+I slot get_availabl:.*f_sim_best = ([\d.]+).*f_keep = ([\d.]+)'
    )
    # Also match the simpler LCP line format
    lcp_simple_pat = re.compile(
        r'^(\d+)\.(\d{2})\.(\d{3})\.(\d{3})\s+I slot get_availabl:.*f_keep = ([\d.]+)'
    )
    # Task launch line to get task id
    task_pat = re.compile(r'I slot launch_slot_:\s+id\s+(\d+)\s+\|\s+task\s+(\d+)')
    # Progress lines for n_tokens
    progress_pat = re.compile(r'prompt processing, n_tokens =\s+(\d+)')

    pending_eval = None
    pending_full = None
    calls = []
    evictions = 0
    current_task = None
    current_lcp = None

    with open(path, 'r', errors='replace') as f:
        for line in f:
            if 'making room for prompt cache entry' in line:
                evictions += 1
                continue

            # LCP similarity
            m = lcp_pat.match(line)
            if m:
                t = float(m.group(1)) + float(m.group(2))/100 + float(m.group(3))/1000 + float(m.group(4))/1e6
                current_lcp = {
                    'time': t,
                    'f_sim': float(m.group(5)),
                    'f_keep': float(m.group(6)),
                }
                continue
            m = lcp_simple_pat.match(line)
            if m:
                t = float(m.group(1)) + float(m.group(2))/100 + float(m.group(3))/1000 + float(m.group(4))/1e6
                current_lcp = {
                    'time': t,
                    'f_keep': float(m.group(5)),
                }
                continue

            # Task launch
            m = task_pat.search(line)
            if m:
                current_task = int(m.group(2))
                continue

            # Prompt eval time
            m = prompt_eval_pat.search(line)
            if m:
                pending_eval = (float(m.group(1)), int(m.group(2)))
                continue

            # Release line (n_tokens = full prompt)
            m = release_pat.search(line)
            if m and pending_eval is not None:
                full = int(m.group(1))
                eval_ms, uncached_tok = pending_eval
                hit = 1 - (uncached_tok / full) if full > 0 else 0
                calls.append({
                    'task': current_task,
                    'full_prompt': full,
                    'uncached_tokens': uncached_tok,
                    'eval_ms': eval_ms,
                    'cache_hit': hit,
                    'lcp': current_lcp,
                })
                pending_eval = None
                current_lcp = None
                continue

            # Progress line (alternative full prompt size)
            m = progress_pat.search(line)
            if m and pending_eval is None:
                # Track the max n_tokens seen in progress for this task
                pass

    return calls, evictions


def fetch_oaiproxy_requests(since: str, until: str, limit: int = 200):
    """Fetch requests from oaiproxy in a time window."""
    url = f"{OAIPROXY}/requests?since={since}&until={until}&limit={limit}&full=1&format=ndjson"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            lines = resp.read().decode('utf-8').strip().split('\n')
            results = []
            for line in lines:
                if line.strip():
                    results.append(json.loads(line))
            return results
    except Exception as e:
        print(f"  oaiproxy fetch failed: {e}", file=sys.stderr)
        return []


def analyze():
    print(f"=== SWA Full Analysis ===\n")

    # 1. Parse server log
    print(f"[1/3] Parsing server log: {SERVER_LOG}")
    calls, evictions = parse_server_log(SERVER_LOG)
    print(f"  Calls: {len(calls)}, Evictions: {evictions}\n")

    if not calls:
        print("  No calls found. Exiting.")
        return

    # 2. Fetch oaiproxy data
    # The run was 11:36 to 11:46
    since = "2026-10-23T11:36:00"
    until = "2026-10-23T11:47:00"
    print(f"[2/3] Fetching oaiproxy requests ({since} to {until})")
    proxy_reqs = fetch_oaiproxy_requests(since, until)
    print(f"  Got {len(proxy_reqs)} requests from oaiproxy\n")

    # 3. Correlate and report
    print(f"[3/3] Correlating and reporting\n")
    print(f"{'#':>3} {'Task':>6} {'Full':>6} {'Uncach':>7} {'Hit%':>5} {'f_keep':>7} {'f_sim':>6} {'Eval':>8} {'Proxy_tok':>10} {'Proxy_dur':>10}")
    print("-" * 90)

    for i, c in enumerate(calls):
        task = c.get('task', '?')
        full = c['full_prompt']
        unc = c['uncached_tokens']
        hit = c['cache_hit'] * 100
        lcp = c.get('lcp') or {}
        f_keep = lcp.get('f_keep', -1)
        f_sim = lcp.get('f_sim', -1)
        eval_s = c['eval_ms'] / 1000

        # Try to match with oaiproxy by prompt token count
        proxy_tok = ''
        proxy_dur = ''
        if proxy_reqs:
            # Find closest match by prompt_tokens
            best = None
            best_diff = float('inf')
            for pr in proxy_reqs:
                pt = pr.get('prompt_tokens')
                if pt is None:
                    continue
                diff = abs(pt - full)
                if diff < best_diff:
                    best_diff = diff
                    best = pr
            if best and best_diff < full * 0.15:  # within 15%
                proxy_tok = str(best.get('prompt_tokens', '?'))
                dur = best.get('duration')
                proxy_dur = f"{dur:.1f}s" if dur else '?'

        f_keep_str = f"{f_keep:.3f}" if f_keep >= 0 else "  -  "
        f_sim_str = f"{f_sim:.3f}" if f_sim >= 0 else "  -  "

        print(f"{i+1:>3} {str(task):>6} {full:>6} {unc:>7} {hit:>4.1f}% {f_keep_str:>7} {f_sim_str:>6} {eval_s:>7.2f}s {proxy_tok:>10} {proxy_dur:>10}")

    # Summary
    total_full = sum(c['full_prompt'] for c in calls)
    total_unc = sum(c['uncached_tokens'] for c in calls)
    cached_pct = 100 * (total_full - total_unc) / total_full if total_full else 0

    print(f"\n{'=' * 90}")
    print(f"Summary: {len(calls)} calls, {total_full:,} total tokens")
    print(f"  Cached:   {total_full - total_unc:,} ({cached_pct:.1f}%)")
    print(f"  Uncached: {total_unc:,} ({100-cached_pct:.1f}%)")
    print(f"  Evictions: {evictions}")

    # Distribution
    buckets = [
        (0, 0.1, '<10%'),
        (0.1, 0.3, '10-30%'),
        (0.3, 0.5, '30-50%'),
        (0.5, 0.7, '50-70%'),
        (0.7, 0.9, '70-90%'),
        (0.9, 0.95, '90-95%'),
        (0.95, 1.01, '95-100%'),
    ]
    print(f"\nCache hit distribution:")
    for lo, hi, label in buckets:
        n = sum(1 for c in calls if lo <= c['cache_hit'] < hi)
        if n:
            bar = '#' * n
            print(f"  {label:<10} {n:>3}  {bar}")

    # Compare with previous runs
    print(f"\nComparison:")
    print(f"  {'Config':<30} {'<10%':>6} {'95-100%':>8} {'Overall':>8}")
    print(f"  {'-'*56}")
    print(f"  {'26B IQ4_XS (n_ctx 131K)':<30} {'51.3%':>6} {'33.3%':>8} {'42.4%':>8}")
    print(f"  {'26B IQ4_XS (n_ctx 65K)':<30} {'51.3%':>6} {'33.3%':>8} {'42.4%':>8}")
    print(f"  {'12B Q5_K_M dense (65K)':<30} {'51.3%':>6} {'33.3%':>8} {'42.4%':>8}")
    cur_lt10 = sum(1 for c in calls if c['cache_hit'] < 0.1) / len(calls) * 100
    cur_95 = sum(1 for c in calls if c['cache_hit'] >= 0.95) / len(calls) * 100
    print(f"  {'26B IQ4_XS +swa-full (65K)':<30} {cur_lt10:>5.1f}% {cur_95:>7.1f}% {cached_pct:>7.1f}%")


if __name__ == '__main__':
    analyze()