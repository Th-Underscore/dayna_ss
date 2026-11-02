#!/usr/bin/env python3
"""Break down per-subject LLM time into prefill vs decode for s522."""
import re
from collections import defaultdict
from datetime import datetime

LOG = "extensions/dayna_ss/tests/runs/logs/scifi_heist_gemma4_s522.log"

turn_pat = re.compile(r'--- turn (\d+)/\d+ ---')
node_pat = re.compile(r"Processing Schema Node: \S+ \(\w+\) at '([^']+)'")
call_pat = re.compile(r'local-call \S+/generate.*?ttft=([\d.]+)s total=([\d.]+)s in=(\d+) out=(\d+)')

subject_calls = defaultdict(lambda: {'count':0, 'ttft_sum':0.0, 'decode_sum':0.0, 'in_sum':0})
current_subject = None

with open(LOG, 'r', errors='replace') as f:
    for line in f:
        m = node_pat.search(line)
        if m:
            path = m.group(1)
            top = path.split('.')[0] if '.' in path else path
            current_subject = top
            continue
        m = call_pat.search(line)
        if m and current_subject:
            ttft = float(m.group(1))
            total = float(m.group(2))
            in_tok = int(m.group(3))
            decode = max(total - ttft, 0)
            d = subject_calls[current_subject]
            d['count'] += 1
            d['ttft_sum'] += ttft
            d['decode_sum'] += decode
            d['in_sum'] += in_tok

hdr = f'{"Subject":<20} {"Calls":>6} {"Prefill(s)":>12} {"Decode(s)":>12} {"Prefill%":>9} {"In-tok":>10}'
print(hdr)
print('-' * len(hdr))
for subj in sorted(subject_calls.keys(), key=lambda x: subject_calls[x]['ttft_sum']+subject_calls[x]['decode_sum'], reverse=True):
    d = subject_calls[subj]
    total_t = d['ttft_sum'] + d['decode_sum']
    if total_t == 0:
        continue
    ppct = 100 * d['ttft_sum'] / total_t
    print(f'{subj:<20} {d["count"]:>6} {d["ttft_sum"]:>12.1f} {d["decode_sum"]:>12.1f} {ppct:>8.0f}% {d["in_sum"]:>10,}')