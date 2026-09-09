"""Analyze dayna_ss schema prompt templates for shared-prefix / prefill-cache
potential across per-entry LLM calls.

For every prompt-template key family (new_entry_query, new_entry, gate_check,
branch_query, branch_update, select_entries, update, importance_query, etc.):

  - membership (which definitions carry it)
  - raw template length
  - longest common prefix (raw bytes) across the family
  - longest common prefix with template variables ({{ ... }}) collapsed to a
    single sentinel token — this is the achievable shared prefix if the branch
    variables are moved to the tail
  - the first divergence point (variable position) in each member
  - shared text highlighted vs per-member divergent text
"""

import json
import re

SCHEMA = "/mnt/c/Users/there/Downloads/Projects/Programming/Python/textgen/extensions/dayna_ss/user_data/example/schemas/subjects_schema_sceneagg.json"

VAR_RE = re.compile(r"\{\{[^{}]+\}\}")

with open(SCHEMA, encoding="utf-8") as f:
    schema = json.load(f)

# Collect templates: key -> [(def_name, template_str)]
families = {}
defs = schema["definitions"]
for def_name, definition in defs.items():
    defaults = definition.get("defaults", {}) or {}
    for key, value in defaults.items():
        if key.endswith("_prompt_template") and isinstance(value, str):
            families.setdefault(key, []).append((def_name, value))
# Also initial_population prompts
for def_name, definition in defs.items():
    ip = (definition.get("defaults", {}) or {}).get("initial_population")
    if isinstance(ip, dict):
        for ipkey in ("prompt_template", "identification_prompt", "population_prompt"):
            v = ip.get(ipkey)
            if isinstance(v, str):
                families.setdefault(f"initial_population[{ipkey}]", []).append((def_name, v))


def lcp(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def strip_vars(s):
    # collapse each {{ ... }} to a single sentinel byte so shared static text
    # around different variables can be measured
    return VAR_RE.sub("\x00", s)


def var_first_raw(s):
    m = VAR_RE.search(s)
    return (m.start(), m.group(0)) if m else (len(s), None)


def var_first_collapsed(collapsed):
    return collapsed.find("\x00")


def describe(raw_list):
    """Return (shared_static_prefix_len_collapsed, shared_static_prefix_len_raw)."""
    st = [strip_vars(t) for t in raw_list]
    n = min(len(x) for x in st)
    i = 0
    while i < n and all(x[i] == "\x00" or x[i] == st[0][i] for x in st) and len({x[i] for x in st}) == 1:
        i += 1
    return i


def first_var_positions(raw):
    """For each member: collapsed offset of first var = collapsed prefix length before divergence via vars."""
    return [(VAR_RE.search(strip_vars(t)).start() if VAR_RE.search(t) else len(t))
            for t in raw]


def highlight(s, shared_len):
    return s[:shared_len] + " ⟦" + s[shared_len:] + "⟧"


def raw_index_at(t, collapsed_pos):
    """Map a position in the var-collapsed string to the raw string index."""
    count = 0
    i = 0
    while i < len(t) and count < collapsed_pos:
        m = VAR_RE.match(t, i)
        if m:
            i = m.end()
        else:
            i += 1
            count += 1
    return i


print("=" * 78)
print("PROMPT TEMPLATE SHARED-PREFIX ANALYSIS — subjects_schema_sceneagg.json")
print("=" * 78)

summary = []
for key, members in sorted(families.items()):
    if len(members) < 2:
        continue
    summary.append((key, members))

# Sort families by shared-prefix potential (descending)
for key, members in sorted(summary, key=lambda kv: -max(
    lcp(strip_vars(a), strip_vars(b)) for i, (_, a) in enumerate(kv[1]) for _, b in kv[1][i + 1:]
) if len(kv[1]) > 1 else 0):
    print("\n" + "#" * 78)
    print(f"FAMILY: {key}   ({len(members)} members)")
    print("#" * 78)

    raw = [t for _, t in members]
    st_r = [strip_vars(t) for t in raw]

    # pairwise stats
    raw_lens = [len(t) for t in raw]
    min_raw = min(raw_lens)
    max_pair_raw = max(lcp(a, b) for i, a in enumerate(raw) for b in raw[i + 1:])
    max_pair_std = max(lcp(a, b) for i, a in enumerate(st_r) for b in st_r[i + 1:])

    # common prefix across ALL members (raw + var-stripped)
    cp_all_raw = len(raw[0])
    for t in raw[1:]:
        cp_all_raw = min(cp_all_raw, lcp(raw[0], t))
    cp_all_std = len(st_r[0])
    for t in st_r[1:]:
        cp_all_std = min(cp_all_std, lcp(st_r[0], t))

    print(f"  min/max raw length : {min_raw} / {max(raw_lens)}")
    print(f"  longest shared prefix (raw, any pair)     : {max_pair_raw}")
    print(f"  longest shared prefix (vars collapsed)    : {max_pair_std}")
    print(f"  common to ALL (raw)                       : {cp_all_raw}")
    print(f"  common to ALL (vars collapsed)            : {cp_all_std}")

    # The best common prefix after collapsing vars, over all members:
    # find it via the pair with the largest collapsed LCP
    best_pair = max(
        ((i, j, lcp(st_r[i], st_r[j])) for i in range(len(raw)) for j in range(i + 1, len(raw))),
        key=lambda ijl: ijl[2],
    )
    bi, bj, best_std = best_pair
    print(f"  best matched pair  : [{members[bi][0]}] × [{members[bj][0]}]  (collapsed LCP {best_std})")

    print(f"\n  --- shared static prefix (vars collapsed, best pair) ---")
    if best_std:
        ridx = raw_index_at(raw[bi], best_std)
        print("  " + highlight(raw[bi][:ridx], best_std))
    else:
        print("  (none — even the first byte diverges per member)")

    print(f"\n  --- per-member divergence ---")
    for def_name, t in members:
        vpos, v = var_first_raw(t)
        tag = f"first var @raw {vpos} {v}" if v else "no vars"
        print(f"  * [{def_name}] len={len(t)}  {tag}")
        lead = t[:100].replace("\n", "\\n")
        print(f"      head: {lead}")
        if best_std:
            ridx = raw_index_at(t, best_std)
            tail = t[ridx:]
            print(f"      tail (from best-pair shared boundary): {tail[:160].replace(chr(10), chr(32))}...")

    # identical members?
    idents = {}
    for def_name, t in members:
        idents.setdefault(t, []).append(def_name)
    for t, names in idents.items():
        if len(names) > 1:
            print(f"\n  IDENTICAL template across: {names}")

    # archetype grouping: group by first-48-chars (rough archetype)
    arch = {}
    for def_name, t in members:
        a = t[:48]
        arch.setdefault(a, []).append(def_name)
    if len(arch) > 1:
        print(f"\n  TEXT ARCHETYPES ({len(arch)}):")
        for a, names in arch.items():
            print(f"    - {names}: {a!r}...")