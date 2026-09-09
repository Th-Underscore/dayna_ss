#!/usr/bin/env python3
"""
Same-Referent Split audit for the dayna_ss subject store.

READ-ONLY, idempotent, CPU + stdlib only. Re-implements the two dedup
primitives inline (does NOT import anything from extensions.dayna_ss):
  - norm(s)                 == _normalize_entry_name  (lowercase + [^a-z0-9] strip)
  - collect_aliases(node)   == _collect_entry_aliases  (bounded recursive walk)

Targets the cyberpunk_thriller__4dbf1435 run: the final checkpoint (primary)
plus per-turn state_snapshot time-series. Writes full verbatim evidence to
split_node_audit_report.md and prints a compact verdict summary to stdout.

Re-runnable: every run reloads from disk and regenerates the report.
"""
import json
import os
import re
from collections import Counter, defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(BASE, "runs", "cyberpunk_thriller__4dbf1435")
FIN = os.path.join(
    RUN, "sandbox", "user_data", "history",
    "soak_cyberpunk_thriller", "soak_100", "d4d4bd6164b1f0a03c2b9d9e")
OUT = os.path.join(BASE, "split_node_audit_report.md")

SUBJECTS = ["characters.json", "groups.json", "elements.json"]
# Denser sampling than the 4-5 suggested: needed to pin first-observation
# of split pairs (rot-sizing: past-event vs ongoing).
SAMPLE_TURNS = [
    "turn_000", "turn_008", "turn_016", "turn_024", "turn_032", "turn_040",
    "turn_048", "turn_056", "turn_064", "turn_072", "turn_080", "turn_084",
]
STOPS = {"the", "a", "an"}

# ---------------------------------------------------------------- primitives
def norm(s):
    """Inline re-implementation of _normalize_entry_name."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

def tokenize_words(s):
    """Tokenize on original word boundaries: strip punctuation, keep spaces."""
    return re.sub(r"[^a-z0-9\s]", "", str(s).lower()).split()

def toks_minus_stops(s):
    """norm-of(stopword-stripped word tokens) -> compare for 'the X' vs 'X'."""
    toks = [t for t in tokenize_words(s) if t not in STOPS]
    return "".join(toks)

def collect_aliases(node, _depth=0, _cap=200):
    """Inline re-implementation of _collect_entry_aliases.
    Bounded recursive walk (depth<=5, list cap) collecting every 'aliases'
    value that is a list of strings."""
    out = []
    if _depth > 5 or len(out) >= _cap:
        return out
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "aliases" and isinstance(v, list):
                for item in v:
                    if isinstance(item, str):
                        out.append(item)
                    if len(out) >= _cap:
                        break
            else:
                out.extend(collect_aliases(v, _depth + 1, _cap))
            if len(out) >= _cap:
                break
    elif isinstance(node, list):
        for item in node:
            out.extend(collect_aliases(item, _depth + 1, _cap))
            if len(out) >= _cap:
                break
    return out

def entries_of(data):
    if not isinstance(data, dict):
        return {}
    e = data.get("entries")
    return e if isinstance(e, dict) else {}

def load_subject(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def node_map(subject):
    return entries_of(subject)

def alias_sets(nodes):
    """key -> frozenset of alias strings (raw + normalized) for a subject map."""
    raw = {k: collect_aliases(n) for k, n in nodes.items()}
    return raw

def partner_set(node):
    rel = node.get("relationships") if isinstance(node, dict) else None
    if not isinstance(rel, dict):
        return set()
    return {p for p in rel.keys()}

def importance_of(partner_val):
    """Extract a numeric importance score from a relationship value."""
    if isinstance(partner_val, dict):
        imp = partner_val.get("importance")
        if isinstance(imp, dict) and isinstance(imp.get("score"), (int, float)):
            return imp["score"]
        return None
    if isinstance(partner_val, list):
        scores = []
        for item in partner_val:
            if isinstance(item, dict):
                imp = item.get("importance")
                if isinstance(imp, dict) and isinstance(imp.get("score"), (int, float)):
                    scores.append(imp["score"])
        if scores:
            return max(scores)
        return None
    return None

def sent_list(node):
    """All description + biography sentences as a list of raw strings."""
    out = []
    for field in ("description", "biography"):
        v = node.get(field) if isinstance(node, dict) else None
        if isinstance(v, list):
            out.extend([x for x in v if isinstance(x, str)])
        elif isinstance(v, str):
            out.append(v)
    return out

def norm_sent(s):
    return re.sub(r"\s+", " ", str(s).casefold().strip())

# ------------------------------------------------------------- pair analysis
def candidate_reasons(A, B, anorm, bnorm, astops, bstops, an_alias, bn_alias):
    reasons = []
    if anorm == bnorm:
        reasons.append("dedup-bypass(norm-equal)")
    if astops == bstops and astops:
        reasons.append("stopword-reframe(the/a/an)")
    if anorm and bnorm and (anorm in bnorm or bnorm in anorm):
        reasons.append("containment")
    if an_alias & bn_alias:
        reasons.append("alias-overlap")
    # alias of one == other's key (normalized)
    if an_alias & {bnorm} or bn_alias & {anorm}:
        reasons.append("alias==key")
    return reasons

def evidence_and_classify(A, B, nA, nB, an_alias, bn_alias):
    """Return (is_split, evidence_dict, distinct_evidence_bool)."""
    ev = {}
    # 1. sentence overlap (normalized)
    sA = {norm_sent(s) for s in sent_list(nA) if norm_sent(s)}
    sB = {norm_sent(s) for s in sent_list(nB) if norm_sent(s)}
    shared = sorted(sA & sB, key=len)
    ev["shared_sentences"] = shared
    ev["shared_sentence_count"] = len(shared)
    ev["nA_desc_sentences"] = len(sA)
    ev["nB_desc_sentences"] = len(sB)
    # 2. shared partner keys (raw + norm)
    pA_raw, pB_raw = partner_set(nA), partner_set(nB)
    pA_norm = {norm(p) for p in pA_raw}
    pB_norm = {norm(p) for p in pB_raw}
    shared_raw = pA_raw & pB_raw
    shared_norm_only = (pA_norm & pB_norm)  # includes raw
    ev["shared_partners_raw"] = sorted(shared_raw)
    ev["shared_partners_norm"] = sorted({p for p in pA_raw for q in pB_raw
                                          if norm(p) == norm(q)})
    # per-partner importance near-identity
    relA = nA.get("relationships") if isinstance(nA, dict) else None
    relB = nB.get("relationships") if isinstance(nB, dict) else None
    if not isinstance(relA, dict):
        relA = {}
    if not isinstance(relB, dict):
        relB = {}
    imp_match = []
    for pA in pA_raw:
        for pB in pB_raw:
            if norm(pA) == norm(pB):
                iA, iB = importance_of(relA.get(pA)), importance_of(relB.get(pB))
                if iA is not None and iB is not None and abs(iA - iB) <= 5:
                    imp_match.append((pA, pB, iA, iB))
    ev["importance_aligned"] = imp_match
    # 3. mutual cross-reference (A rels -> B key, B rels -> A key)
    an_norm = {norm(a) for a in an_alias}
    bn_norm = {norm(b) for b in bn_alias}
    ev["A_refs_B"] = any(norm(p) == norm(B) or norm(p) in bn_norm
                         for p in pA_raw)
    ev["B_refs_A"] = any(norm(p) == norm(A) or norm(p) in an_norm
                         for p in pB_raw)
    # 4. node's own top-level importance near-identity
    oA = nA.get("importance") if isinstance(nA, dict) else None
    oB = nB.get("importance") if isinstance(nB, dict) else None
    oAs = oA.get("score") if isinstance(oA, dict) else None
    oBs = oB.get("score") if isinstance(oB, dict) else None
    ev["own_importance_A"] = oAs
    ev["own_importance_B"] = oBs
    # aliases each node claims (raw strings)
    ev["A_aliases"] = sorted(set(an_alias))
    ev["B_aliases"] = sorted(set(bn_alias))

    # ---- classification ----
    split_signals = []
    if ev["shared_sentence_count"] > 0:
        split_signals.append("sentence-overlap")
    if ev["importance_aligned"]:
        split_signals.append("importance-aligned-on-shared-partner")
    if ev["A_refs_B"] and ev["B_refs_A"]:
        split_signals.append("mutual-cross-ref")
    elif ev["A_refs_B"] or ev["B_refs_A"]:
        split_signals.append("unilateral-cross-ref")
    if ev["shared_partners_raw"]:
        split_signals.append("shared-partners")
    is_split = len(split_signals) >= 1 and (
        ev["shared_sentence_count"] > 0
        or ev["importance_aligned"]
        or (ev["A_refs_B"] and ev["B_refs_A"]))
    # shared partners + shared sentences also strong
    if ev["shared_sentence_count"] > 0 and ev["shared_partners_raw"]:
        is_split = True
    return is_split, ev, split_signals

def analyze_subject(subject, subj_name, nodes=None):
    if nodes is None:
        nodes = node_map(subject)
    keys = list(nodes.keys())
    norm_map = {k: norm(k) for k in keys}
    stop_map = {k: toks_minus_stops(k) for k in keys}
    raw_alias = {k: collect_aliases(nodes[k]) for k in keys}
    alias_norm = {k: {norm(a) for a in raw_alias[k]} for k in keys}
    alias_norm_own = {k: alias_norm[k] | {norm_map[k]} for k in keys}

    out = {
        "subject": subj_name,
        "node_count": len(keys),
        "dedup_bypasses": [],   # norm-equal (should be 0)
        "candidates": [],       # {A,B,reasons,is_split,ev,signals}
        "splits": [],
        "distinguishable": [],
        "dangles": [],
    }
    # alias resolution index for dangles
    all_alias_norm = set()
    for k in keys:
        all_alias_norm |= alias_norm[k]

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            A, B = keys[i], keys[j]
            reasons = candidate_reasons(
                A, B, norm_map[A], norm_map[B],
                stop_map[A], stop_map[B],
                alias_norm_own[A], alias_norm_own[B])
            if norm_map[A] == norm_map[B]:
                out["dedup_bypasses"].append([A, B])
                continue
            if not reasons:
                continue
            is_split, ev, signals = evidence_and_classify(
                A, B, nodes[A], nodes[B], raw_alias[A], raw_alias[B])
            cand = {"A": A, "B": B, "reasons": reasons,
                    "is_split": is_split, "signals": signals, "ev": ev}
            out["candidates"].append(cand)
            if is_split:
                out["splits"].append(cand)
            else:
                out["distinguishable"].append(cand)
    # dangles
    for k in keys:
        for p in partner_set(nodes[k]):
            if p in keys:
                continue  # exact
            if norm(p) in {norm_map[x] for x in keys}:
                continue  # normalized
            if norm(p) in all_alias_norm:
                continue  # alias
            out["dangles"].append({"node": k, "partner": p})
    return out

def eg_name_collisions(eg):
    nodes = eg.get("nodes") if isinstance(eg, dict) else None
    if not isinstance(nodes, dict):
        return [], 0
    by_name = defaultdict(list)
    for nid, nd in nodes.items():
        nm = nd.get("name") if isinstance(nd, dict) else None
        tp = nd.get("type") if isinstance(nd, dict) else None
        if nm is not None:
            by_name[nm].append((nid, tp))
    collisions = [(nm, len(v), v) for nm, v in sorted(by_name.items()) if len(v) > 1]
    return collisions, len(nodes)

# --------------------------------------------------------------- main driver
def main():
    L = []
    ap = L.append
    ap("# Same-Referent Split Audit — dayna_ss subject store\n")
    ap("Run: `cyberpunk_thriller__4dbf1435`  ")
    ap("Primary: final checkpoint (soak_100/d4d4bd6164b1f0a03c2b9d9e)  ")
    ap("Time-series: per-turn `state_snapshot/` over sampled turns.\n")
    ap("> Read-only, idempotent, CPU + stdlib. Dedup primitives re-implemented")
    ap("> inline. Full verbatim evidence below.\n")

    # ---------- 1. FINAL CHECKPOINT ----------
    ap("## 1. FINAL CHECKPOINT\n")
    final_analyses = {}
    all_splits = {}  # (subject, frozenset({normA,normB})) -> cand
    total_dedup = []
    total_dangles = {}
    for subj in SUBJECTS:
        data = load_subject(os.path.join(FIN, subj))
        an = analyze_subject(data, subj)
        final_analyses[subj] = an
        total_dedup += [[a, b] for a, b in an["dedup_bypasses"]]
        total_dangles[subj] = an["dangles"]
        ap("### %s  (node count: %d)\n" % (subj, an["node_count"]))
        if an["dedup_bypasses"]:
            ap("**DEDUP BYPASS (norm-equal keys — should be 0):**")
            for a, b in an["dedup_bypasses"]:
                ap("- `%s` == `%s` (normalized identical)" % (a, b))
            ap("")
        if not an["candidates"]:
            ap("_No candidate pairs generated._\n")
            continue
        ap("#### Candidate pairs (split candidates)\n")
        for cand in an["candidates"]:
            A, B = cand["A"], cand["B"]
            verdict = "SAME-REFERENT-SPLIT" if cand["is_split"] else "distinct/ambiguous"
            ap("##### `%s`  <>  `%s`  → **%s**" % (A, B, verdict))
            ap("- candidate trigger(s): %s" % ", ".join(cand["reasons"]))
            if cand["is_split"]:
                ap("- split signal(s): %s" % ", ".join(cand["signals"]))
            ev = cand["ev"]
            ap("- shared sentences: **%d**  (A has %d, B has %d description/bio sentences)"
               % (ev["shared_sentence_count"], ev["nA_desc_sentences"], ev["nB_desc_sentences"]))
            for s in ev["shared_sentences"][:12]:
                ap("  - SHARED: `%s`" % s)
            if len(ev["shared_sentences"]) > 12:
                ap("  - … %d more shared sentences" % (len(ev["shared_sentences"]) - 12))
            ap("- shared partners (raw): %s" % (ev["shared_partners_raw"] or "[]"))
            ap("- shared partners (normalized): %s" % (ev["shared_partners_norm"] or "[]"))
            if ev["importance_aligned"]:
                ap("- importance-aligned on shared partner(s):")
                for pA, pB, iA, iB in ev["importance_aligned"]:
                    ap("  - A rel `%s` score=%s  ==  B rel `%s` score=%s" % (pA, iA, pB, iB))
            ap("- A refs B key: %s   |   B refs A key: %s" % (ev["A_refs_B"], ev["B_refs_A"]))
            ap("- own importance: A=%s  B=%s" % (ev["own_importance_A"], ev["own_importance_B"]))
            ap("- aliases(A): %s" % (ev["A_aliases"] or "[]"))
            ap("- aliases(B): %s" % (ev["B_aliases"] or "[]"))
            ap("")
        # splits summary
        if an["splits"]:
            ap("**SPLIT verdicts in %s:**" % subj)
            for cand in an["splits"]:
                key = (subj, frozenset({norm(cand["A"]), norm(cand["B"])}))
                all_splits[key] = cand
                ap("- `%s` / `%s` — signals: %s"
                   % (cand["A"], cand["B"], ", ".join(cand["signals"])))
            ap("")
        if an["distinguishable"]:
            ap("**Distinct/ambiguous candidates in %s (no strong split signal):**" % subj)
            for cand in an["distinguishable"]:
                ap("- `%s` / `%s` — triggers %s; shared sentences %d; shared partners %s; xref A→B=%s B→A=%s"
                   % (cand["A"], cand["B"], ", ".join(cand["reasons"]),
                      cand["ev"]["shared_sentence_count"],
                      cand["ev"]["shared_partners_raw"],
                      cand["ev"]["A_refs_B"], cand["ev"]["B_refs_A"]))
            ap("")
        if an["dangles"]:
            ap("**Dangling relationship edges in %s (partner resolves to NO node):**" % subj)
            for d in sorted(an["dangles"], key=lambda x: (x["node"], x["partner"])):
                ap("- node `%s` → partner `%s`" % (d["node"], d["partner"]))
            ap("")
        ap("")

    # ---------- 2. DANGLES summary ----------
    ap("## 2. DANGLING RELATIONSHIP EDGES (final checkpoint)\n")
    partner_freq = Counter()
    for subj in SUBJECTS:
        for d in total_dangles[subj]:
            partner_freq[(subj, d["partner"])] += 1
    ap("Total dangling edges: **%d**  (top-10 partners by frequency)\n"
       % sum(partner_freq.values()))
    ap("| subject | partner | count | lookalike? |")
    ap("|---|---|---|---|")
    # lookalike if norm(partner) is a near-miss of some node key
    for subj in SUBJECTS:
        nodes = node_map(load_subject(os.path.join(FIN, subj)))
        normkeys = {norm(k) for k in nodes}
        for (s, p), c in sorted(partner_freq.items()):
            if s != subj:
                continue
            is_look = "LOOKALIKE" if any(
                p.lower().replace("_", "").replace(" ", "") == nk
                or norm(p) in nk or nk in norm(p) for nk in normkeys) else "abstract/foreign"
            ap("| %s | `%s` | %d | %s |" % (s, p, c, is_look))
    ap("")

    # ---------- 3. ENTITY GRAPH ----------
    ap("## 3. ENTITY-GRAPH DISPLAY-NAME COLLISIONS\n")
    eg = load_subject(os.path.join(FIN, "entity_graph.json"))
    collisions, eg_node_total = eg_name_collisions(eg)
    ap("Total graph nodes: **%d**  |  distinct display names in collision: **%d**\n"
       % (eg_node_total, len(collisions)))
    if not collisions:
        ap("_No display-name collisions._\n")
    for nm, cnt, v in collisions:
        ap("### `%s`  — %d graph nodes\n" % (nm, cnt))
        ap("| node id | type |")
        ap("|---|---|")
        for nid, tp in sorted(v):
            ap("| `%s` | %s |" % (nid, tp))
        ap("")

    # ---------- 4. TIME SERIES ----------
    ap("## 4. TIME-SERIES GROWTH\n")
    ap("Sampled turns: %s\n" % ", ".join(SAMPLE_TURNS))
    header = "| turn | " + " | ".join(SUBJECTS) + " | split-pairs present | dangling edges |\n"
    ap(header)
    ap("|---|" + "---|" * (len(SUBJECTS) + 2))
    ts_rows = []
    per_turn = {}  # turn -> {subj: {node_count, splits:set, dangles:int}}
    for tn in SAMPLE_TURNS:
        snap = os.path.join(RUN, tn, "state_snapshot")
        if not os.path.isdir(snap):
            ap("| %s | MISSING | %s | MISSING | %s |"
               % (tn, " | ".join(["MISSING"] * len(SUBJECTS)), "MISSING"))
            continue
        row = [tn]
        turn_split_pairs = set()
        turn_dangles = 0
        per_turn[tn] = {}
        for subj in SUBJECTS:
            data = load_subject(os.path.join(snap, subj))
            an = analyze_subject(data, subj)
            per_turn[tn][subj] = {
                "nodes": an["node_count"],
                "split_normkeys": {frozenset({norm(c["A"]), norm(c["B"])})
                                     for c in an["splits"]},
                "dangles": len(an["dangles"]),
                "dangle_list": an["dangles"],
                "keyset": {norm(k) for k in node_map(data)},
            }
            turn_split_pairs |= per_turn[tn][subj]["split_normkeys"]
            turn_dangles += len(an["dangles"])
            row.append(str(an["node_count"]))
        row.append(str(len(turn_split_pairs)))
        row.append(str(turn_dangles))
        ts_rows.append((tn, row))
    for tn, row in ts_rows:
        ap("| %s | %s |" % (tn, " | ".join(row)))
    ap("")

    # first-observation per final split pair
    ap("### First-observation of final-checkpoint split pairs\n")
    if not all_splits:
        ap("_No split pairs in final checkpoint._\n")
    for (subj, normkey), cand in sorted(all_splits.items(), key=lambda kv: kv[0][0]):
        # find earliest sampled turn that already contains BOTH normalized keys
        first = None
        alone = {"A": None, "B": None}
        Akey = [k for k in normkey if k == norm(cand["A"])][0]
        Bkey = [k for k in normkey if k == norm(cand["B"])][0]
        for tn in SAMPLE_TURNS:
            if subj not in per_turn.get(tn, {}):
                continue
            ks = per_turn[tn][subj]["keyset"]
            aIn, bIn = Akey in ks, Bkey in ks
            if aIn and bIn and first is None:
                first = tn
            if aIn and alone["A"] is None:
                alone["A"] = tn
            if bIn and alone["B"] is None:
                alone["B"] = tn
        ap("- `%s` / `%s`  (%s):" % (cand["A"], cand["B"], subj))
        ap("  - earliest sampled turn with BOTH nodes: **%s**" % (first or "none in sampled window"))
        ap("  - first solo appearance: `%s` → **%s**,  `%s` → **%s**"
           % (cand["A"], alone["A"] or "—", cand["B"], alone["B"] or "—"))
    ap("")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

    # ------------------------------------------------------------- stdout
    n_split = len(all_splits)
    n_dang = sum(partner_freq.values())
    print("WROTE", OUT)
    print("==== SPLIT PAIRS: %d ====" % n_split)
    for (subj, nk), cand in sorted(all_splits.items(), key=lambda kv: kv[0][0]):
        print("  [SPLIT] %s: %r / %r  signals=%s"
              % (subj, cand["A"], cand["B"], cand["signals"]))
    print("==== DEDUP BYPASSES (norm-equal): %d ====" % len(total_dedup))
    for a, b in total_dedup:
        print("  [DEDUP-BYPASS] %r / %r" % (a, b))
    print("==== DANGLING EDGES: %d ====" % n_dang)
    for (s, p), c in partner_freq.most_common(10):
        print("  [%s] %r x%d" % (s, p, c))
    print("==== EG NAME COLLISIONS: %d ====" % len(collisions))
    for nm, cnt, v in collisions:
        print("  %r x%d -> %s" % (nm, cnt, ", ".join(sorted(x[0] for x in v))))
    print("==== DENSE SAMPLE (12 pts) growth ====")
    for tn, row in ts_rows:
        print("  %s: %s" % (tn, " ".join(row)))
    print("REPORT LINES:", len(L))

if __name__ == "__main__":
    main()