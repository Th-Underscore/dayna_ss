#!/usr/bin/env python3
"""Full consolidation: ground-truth LCP from raw prompt tokens vs server prefill timing.

For each consecutive pair of prompts, computes the exact token-level LCP.
Compares against what llama-server's timing implies it actually cached.
"""

import re
import json
import os
import sys

PROMPT_DIR = "/tmp/llama-server-prompts"
SERVER_LOG = "extensions/dayna_ss/tests/runs/logs/server_gm4_stv_swa_cms0.log"

def load_prompts(directory):
    """Load all prompt files as token lists, sorted by timestamp filename."""
    prompts = []
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith('.txt'):
            continue
        path = os.path.join(directory, fname)
        with open(path, 'r') as f:
            tokens = json.load(f)
        # Extract timestamp from filename (microseconds)
        ts = int(fname.replace('.txt', ''))
        prompts.append((ts, tokens))
    prompts.sort(key=lambda x: x[0])
    return prompts

def compute_lcp(tokens_a, tokens_b):
    """Compute exact longest common prefix length between two token lists."""
    lcp = 0
    min_len = min(len(tokens_a), len(tokens_b))
    for i in range(min_len):
        if tokens_a[i] == tokens_b[i]:
            lcp = i + 1
        else:
            break
    return lcp

def compute_lcs_approx(tokens_a, tokens_b, max_check=2000):
    """Approximate: find the longest common substring in the first N tokens of each.
    This tells us how much of prompt B appears contiguously in prompt A (not just prefix)."""
    # For efficiency, only check a window
    a_set = set()
    # Build a set of 8-gram tuples from A for quick lookup
    a_grams = {}
    for i in range(len(tokens_a) - 7):
        gram = tuple(tokens_a[i:i+8])
        if gram not in a_grams:
            a_grams[gram] = i
    # Scan B for the longest match
    best = 0
    i = 0
    while i < len(tokens_b) - 7:
        gram = tuple(tokens_b[i:i+8])
        if gram in a_grams:
            a_pos = a_grams[gram]
            # Extend the match
            length = 8
            while (a_pos + length < len(tokens_a) and 
                   i + length < len(tokens_b) and
                   tokens_a[a_pos + length] == tokens_b[i + length]):
                length += 1
            if length > best:
                best = length
            i += length  # skip past this match
        else:
            i += 1
    return best

def parse_server_log(path):
    """Extract per-call timing from the server log."""
    prompt_eval_pat = re.compile(r'prompt eval time =\s+([\d.]+)\s+ms\s*/\s*(\d+)\s+tokens')
    release_pat = re.compile(r'release:.*n_tokens\s*=\s*(\d+)')
    lcp_full_pat = re.compile(r'f_sim_best = ([\d.]+).*f_keep = ([\d.]+)')
    task_pat = re.compile(r'I slot launch_slot_:\s+id\s+(\d+)\s+\|\s+task\s+(\d+)')

    pending_eval = None
    calls = []
    evictions = 0
    current_task = None
    last_fsim = None
    last_fkeep = None

    with open(path, 'r', errors='replace') as f:
        for line in f:
            if 'making room for prompt cache entry' in line:
                evictions += 1
            m = lcp_full_pat.search(line)
            if m:
                last_fsim = float(m.group(1))
                last_fkeep = float(m.group(2))
                continue
            m = task_pat.search(line)
            if m:
                current_task = int(m.group(2))
                continue
            m = prompt_eval_pat.search(line)
            if m:
                pending_eval = int(m.group(2))
                continue
            m = release_pat.search(line)
            if m and pending_eval is not None:
                full = int(m.group(1))
                uncached = pending_eval
                hit = 1 - (uncached / full) if full > 0 else 0
                calls.append({
                    'task': current_task,
                    'full': full,
                    'uncached': uncached,
                    'hit': hit,
                    'fsim': last_fsim,
                    'fkeep': last_fkeep,
                })
                pending_eval = None
                last_fsim = None
                last_fkeep = None
                continue
    return calls, evictions

def main():
    print("=" * 100)
    print("GROUND TRUTH: Actual prompt LCP vs llama-server prefill")
    print("=" * 100)

    # 1. Load prompts
    print(f"\n[1] Loading prompts from {PROMPT_DIR}")
    prompts = load_prompts(PROMPT_DIR)
    print(f"    {len(prompts)} prompts loaded")
    for i, (ts, tokens) in enumerate(prompts):
        print(f"    #{i+1:2d}  ts={ts}  tokens={len(tokens)}")

    # 2. Parse server log
    print(f"\n[2] Parsing server log: {SERVER_LOG}")
    calls, evictions = parse_server_log(SERVER_LOG)
    print(f"    {len(calls)} calls, {evictions} evictions")

    # 3. Compute LCP between consecutive prompts
    print(f"\n[3] Computing exact token-level LCP between consecutive prompts")
    print(f"\n{'#':>3} {'Tokens':>7} {'LCP vs prev':>12} {'LCP%':>6} {'Server hit':>11} {'Delta':>7} {'fsim':>6} {'fkeep':>6} {'Verdict':>12}")
    print("-" * 100)

    lcp_values = []
    for i in range(1, len(prompts)):
        ts, tokens = prompts[i]
        prev_tokens = prompts[i-1][1]
        
        lcp = compute_lcp(tokens, prev_tokens)
        lcp_pct = lcp / len(tokens) * 100 if tokens else 0
        lcp_values.append(lcp)

        # Match with server log call (by index, assuming 1:1)
        if i < len(calls):
            call = calls[i]
            server_hit = call['hit'] * 100
            # Server's implied cached tokens
            server_cached = call['full'] - call['uncached']
            # Delta: actual LCP vs what server cached
            delta = lcp - server_cached
            fsim = f"{call['fsim']:.3f}" if call['fsim'] is not None else "  -  "
            fkeep = f"{call['fkeep']:.3f}" if call['fkeep'] is not None else "  -  "
            
            # Verdict
            if abs(delta) < lcp * 0.15 and lcp > 100:
                verdict = "MATCH"
            elif server_hit > 90 and lcp_pct < 50:
                verdict = "OVER-CACHE"
            elif server_hit < 10 and lcp_pct > 50:
                verdict = "UNDER-CACHE"
            elif lcp < 100:
                verdict = "NO-SHARE"
            else:
                verdict = f"gap={delta:+d}"
        else:
            server_hit = -1
            fsim = "  -  "
            fkeep = "  -  "
            delta = 0
            verdict = "no-call"

        print(f"{i+1:>3} {len(tokens):>7} {lcp:>12} {lcp_pct:>5.1f}% {server_hit:>10.1f}% {delta:>+7} {fsim:>6} {fkeep:>6} {verdict:>12}")

    # 4. Summary
    print(f"\n{'=' * 100}")
    print("SUMMARY")
    print(f"{'=' * 100}")
    
    if lcp_values:
        avg_lcp = sum(lcp_values) / len(lcp_values)
        min_lcp = min(lcp_values)
        max_lcp = max(lcp_values)
        print(f"  Actual LCP: avg={avg_lcp:.0f}, min={min_lcp}, max={max_lcp}")
        print(f"  LCP distribution:")
        buckets = [
            (0, 100, '<100 (no shared prefix)'),
            (100, 1000, '100-1K'),
            (1000, 5000, '1K-5K'),
            (5000, 10000, '5K-10K'),
            (10000, 20000, '10K-20K'),
            (20000, 999999, '>20K'),
        ]
        for lo, hi, label in buckets:
            n = sum(1 for v in lcp_values if lo <= v < hi)
            if n:
                print(f"    {label:<25} {n:>3}")

    # 5. Where do prompts diverge?
    print(f"\n[4] Divergence analysis (first N tokens where prompts differ):")
    # For the worst cases (low LCP), show what's at the divergence point
    worst = [(i, lcp_values[i-1]) for i in range(1, len(prompts)) if lcp_values[i-1] < 1000]
    worst.sort(key=lambda x: x[1])
    
    for idx, lcp in worst[:5]:
        tokens = prompts[idx][1]
        prev = prompts[idx-1][1]
        print(f"\n  Call #{idx+1} (LCP={lcp}):")
        print(f"    Prev prompt [{lcp}:{lcp+20}]: {prev[lcp:lcp+20]}")
        print(f"    Curr prompt [{lcp}:{lcp+20}]: {tokens[lcp:lcp+20]}")
        if lcp < len(tokens):
            print(f"    (divergence at token {lcp} of {len(tokens)} total)")

if __name__ == '__main__':
    main()