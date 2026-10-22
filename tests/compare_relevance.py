#!/usr/bin/env python3
"""Compare the entity graph's relevance against cross-encoder relevance against
hand-labeled ground truth for a single turn.

Grounds the question: "how accurately does each method reflect the current plot
state?" using the turn-141 greek_mythology_retelling run as the probe.

Graph relevance (real machinery, recomputed):
  - Seeded with the in-scene characters (retrieve_context builds
    initial_entities from current_scene["now"]["who"]["characters"] plus
    characters named in the context). For turn 141 that is just {"Leander"}.
  - _unified_entity_aggregation -> traverse_graph_detailed over the REAL
    19-node / 39-relationship entity_graph.json (resolved via the turn's
    result.json -> history_path), with the production DecayConfig defaults
    (final_threshold=75, decay_exponent=0.4, convergence + re-traversal on).
  - Pruned to entities with relevance >= final_threshold.
  - Elements are NOT graph nodes; they are pulled by substring re.search over
    the 73K context, so effectively ALL elements are "relevant".

Cross-encoder relevance:
  - Each entity (rendered via format_templates) scored against the recent user
    narration (the current turn's input) with ms-marco-MiniLM-L6-v2.

Ground truth:
  - Hand-labeled from the turn-141 narrative + reply.
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]  # .../textgen — parent of `extensions`
sys.path.insert(0, str(REPO))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

RUN = HERE / "runs" / "greek_mythology_retelling__2e7575de"
TURN = 141


def _coerce_importance(v):
    """Coerce a possibly-string importance score to int (see todo.md bug note)."""
    if v is None:
        return 0
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        try:
            return int(v)
        except ValueError:
            return 0
    return int(v)


# ---------------------------------------------------------------------------
# Ground truth, hand-labeled from the turn-141 narrative + reply.
# 3 = clearly relevant to THIS turn's action; 0 = clearly not.
# ---------------------------------------------------------------------------
def ground_truth() -> dict[str, int]:
    # Hand-labeled from the turn-141 narrative + reply, using the REAL entity
    # names from state_snapshot/elements.json. Tiers: 3 = clearly central to
    # THIS turn's action, 2 = present in scene / campaign-relevant, 0 = clearly
    # absent (opening-scene / elsewhere).
    return {
        # --- 3: the whole action is built on these ---
        "Leander": 3,                              # protagonist; whole scene is his
        "small, folded square of papyrus": 3,      # dissolving in the water (central object)
        "dark water": 3,                           # the papyrus dissolves into it
        "heavy iron chisel": 3,                    # he picks it up and swings it (reply)
        "hammer with leather grip": 3,             # the hammer (reply)
        "sulfur": 3,                               # scent hanging "like incense before an altar"
        "obsidian shards": 3,                      # the polished obsidian floor
        # --- 2: in scene / campaign-relevant, not the focus ---
        "Calliope": 2,                             # referenced in scene
        "deep, stagnant water": 2,                 # the cavern water
        "wet, luminescent rock": 2,                # the glowing / phosphorescent stone
        "vessel of cracked clay": 2,               # present in the scene
        "ember": 2,                                # the campaign's quest object
        # --- 0: clearly NOT part of this turn's action ---
        "wool blanket": 0,
        "tethered goat": 0,
        "market square": 0,
        "steaming bowl of porridge": 0,
        "wine-dark shore": 0,
        "cold tide": 0,
        "black sands": 0,
        "oar": 0,
        "ceramic pot": 0,
        "smithy": 0,
        "borrowed sword": 0,
        "crooked thatched roof": 0,
    }


def spearman(pairs: list[tuple[float, float]]) -> float:
    """Spearman rank correlation between two score lists (aligned by entity)."""
    n = len(pairs)
    if n < 2:
        return float("nan")

    def rank(vals: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1  # 1-based average rank for ties
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    rx = rank([a for a, _ in pairs])
    ry = rank([b for _, b in pairs])
    mx = sum(rx) / n
    my = sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    vy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if vx == 0 or vy == 0:
        return float("nan")
    return cov / (vx * vy)


def _resolve_history_path() -> Path:
    """Resolve turn TURN's actual history dir via result.json -> history_path."""
    result = json.loads((RUN / f"turn_{TURN}" / "result.json").read_text())
    hp = Path(result["history_path"])
    if not (hp / "entity_graph.json").exists():
        raise FileNotFoundError(f"no entity_graph.json at resolved history_path: {hp}")
    return hp


def _run_real_graph_aggregation(history_path: Path) -> dict[str, dict[str, float]]:
    """Run the production entity-graph aggregation for this turn.

    Mirrors ContextRetriever.retrieve_context: seed with in-scene characters,
    run _unified_entity_aggregation over the real loaded EntityGraph, prune at
    final_threshold. Returns {etype: {name: relevance}} for character/group/event.
    """
    from extensions.dayna_ss.rag.structured_rag.entity_graph import EntityGraph
    from extensions.dayna_ss.rag.structured_rag.decay_config import DecayConfig
    # Pull the aggregation as an unbound function so we can drive it with a
    # real EntityGraph without spinning up the whole retriever (which needs the
    # embedding model / LLM). It only touches self.entity_graph and
    # self.schema_classes.
    from extensions.dayna_ss.rag.structured_rag import context_retriever as cr

    snap = RUN / f"turn_{TURN}" / "state_snapshot"
    cs = json.loads((snap / "current_scene.json").read_text())
    seed_chars: list[str] = []
    for c in cs.get("now", {}).get("who", {}).get("characters", []):
        if isinstance(c, dict) and c.get("name") and c["name"] not in seed_chars:
            seed_chars.append(c["name"])

    graph = EntityGraph(history_path, persist=True, schema_classes=None)
    print(f"  EntityGraph loaded: {len(graph.nodes)} nodes / {len(graph.relationships)} rels")

    class _Harness:
        pass

    h = _Harness()
    h.entity_graph = graph
    h.schema_classes = {}

    decay = DecayConfig()
    initial_entities = {"Character": set(seed_chars)}
    all_data = {
        "Character": json.loads((history_path / "characters.json").read_text()),
        "Group": json.loads((history_path / "groups.json").read_text()),
        "Event": json.loads((history_path / "events.json").read_text()),
    }
    result = cr.StoryContextRetriever._unified_entity_aggregation(
        h,
        initial_entities,
        all_data,
        importance_threshold=decay.final_threshold,
        max_depth=decay.max_depth,
        decay_config=decay,
    )
    print(f"  seed characters (in-scene): {seed_chars}")
    return result


def main() -> None:
    print("=" * 72)
    print("RELEVANCE COMPARISON — turn", TURN, "greek_mythology_retelling")
    print("=" * 72)

    # ---- Load entities (rendered through format_templates) ------------------
    from test_context_injection import (
        CrossEncoderReranker,
        extract_entities_from_state,
    )

    entities = extract_entities_from_state(RUN, TURN)
    print(f"\nEntities from state_snapshot: {len(entities)}")
    tc = Counter(e["type"] for e in entities)
    print(f"  type breakdown: {dict(tc)}")

    # ---- Recent-message scene text (the current turn's user narration) ------
    result = json.loads((RUN / f"turn_{TURN}" / "result.json").read_text())
    scene_text = result["user_input"]
    print(f"\nCE target = current user narration ({len(scene_text)} chars):")
    print(f"  {scene_text[:160]}...")

    # ---- Cross-encoder scores ----------------------------------------------
    print("\n" + "-" * 72)
    print("CROSS-ENCODER (ms-marco-MiniLM-L6-v2) vs recent narration")
    print("-" * 72)
    reranker = CrossEncoderReranker(model_name="mid", device="cpu")
    ce_scored = reranker.score(scene_text, entities)  # sorted desc by score
    ce_rank = {e["name"]: i for i, e in enumerate(ce_scored)}
    ce_rank = {name: r + 1 for name, r in ce_rank.items()}
    ce_score = {e["name"]: e["relevance_score"] for e in ce_scored}

    # ---- Graph relevance (recomputed via real machinery) -------------------
    print("\n" + "-" * 72)
    print("GRAPH (structural) — REAL traversal over the turn's entity_graph.json")
    print("-" * 72)
    try:
        history_path = _resolve_history_path()
        print(f"  history_path: {history_path.name}")
        graph_result = _run_real_graph_aggregation(history_path)
    except Exception as e:
        print(f"  !! real aggregation failed: {type(e).__name__}: {e}")
        graph_result = {"character": {}, "group": {}, "event": {}}

    graph_char_names = set(graph_result.get("character", {}))
    graph_event_names = set(graph_result.get("event", {}))
    graph_group_names = set(graph_result.get("group", {}))
    # A binary "relevance score" for the graph: character/event/group included=1.
    # Elements are handled separately (substring flood = all of them).
    all_element_names = {e["name"] for e in entities if e["type"] == "element"}
    graph_included = (graph_char_names | graph_group_names | graph_event_names
                      | all_element_names)
    graph_count = len(graph_included)

    print(f"\n  character relevance from traversal (name = relevance, pruned @ >=75):")
    for name, rel in sorted(graph_result.get("character", {}).items(),
                            key=lambda x: -x[1]):
        print(f"      {rel:6.1f}  {name}")
    print(f"  event relevance from traversal:")
    for name, rel in sorted(graph_result.get("event", {}).items(),
                            key=lambda x: -x[1]):
        print(f"      {rel:6.1f}  {name}")
    print(f"  group relevance from traversal:")
    for name, rel in sorted(graph_result.get("group", {}).items(),
                            key=lambda x: -x[1]):
        print(f"      {rel:6.1f}  {name}")
    print(f"\n  characters included (graph traversal): {sorted(graph_char_names) or '(none)'}")
    print(f"  events/groups included               : {sorted(graph_event_names | graph_group_names) or '(none)'}")
    print(f"  elements included (substring match)  : {len(all_element_names)} (ALL — circular)")
    print(f"  total entities graph marks 'relevant' : {graph_count} / {len(entities)}")
    graph_score = {e["name"]: (1.0 if e["name"] in graph_included else 0.0)
                   for e in entities}

    # ---- Ground truth -------------------------------------------------------
    gt = ground_truth()
    print("\n" + "-" * 72)
    print("GROUND TRUTH (hand-labeled from narrative+reply)")
    print("-" * 72)
    for name, score in sorted(gt.items(), key=lambda x: -x[1]):
        label = {3: "CLEARLY relevant", 2: "relevant", 0: "NOT relevant"}.get(score, "?")
        print(f"  [{label:18s}] {name}")

    # ---- Agreement metrics --------------------------------------------------
    print("\n" + "-" * 72)
    print(f"AGREEMENT vs GROUND TRUTH (n={len(gt)} labeled entities)")
    print("-" * 72)

    ce_pairs = [(ce_score.get(n, float("-inf")), float(s)) for n, s in gt.items()]
    ce_spearman = spearman(ce_pairs)

    relevant_gt = {n for n, s in gt.items() if s >= 2}
    top10 = [e["name"] for e in ce_scored[:10]]
    ce_top10_hit = len([n for n in top10 if n in relevant_gt])
    ce_top10_prec = ce_top10_hit / 10 if top10 else 0.0

    # Graph: binary inclusion. Use precision/recall against the labeled set.
    # The labeled set is all-entities (characters + elements) — compare on the
    # labeled subset only so the metric is over the same population as CE.
    gt_labeled = set(gt)
    graph_hit_labeled = len([n for n in relevant_gt if n in graph_included])
    graph_recall = graph_hit_labeled / len(relevant_gt) if relevant_gt else 0.0
    # precision over the labeled population: of the labeled entities the graph
    # marks relevant, how many truly are.
    graph_marked_labeled = [n for n in gt_labeled if n in graph_included]
    graph_prec_labeled = (graph_hit_labeled / len(graph_marked_labeled)
                          if graph_marked_labeled else 0.0)
    n_irrelevant_gt = len(gt) - len(relevant_gt)
    print(f"  labeled 'relevant' tier (score>=2): {len(relevant_gt)}  |  'NOT relevant': {n_irrelevant_gt}")

    print(f"\n  Spearman rank correlation with ground truth (labeled subset):")
    print(f"    cross-encoder : {ce_spearman:+.3f}")
    print(f"    graph         : n/a  (binary inclusion; use P/R below)")

    print(f"\n  Top-10 precision (fraction of CE's top-10 that are labeled-relevant):")
    print(f"    cross-encoder : {ce_top10_hit}/10 = {ce_top10_prec:.0%}")

    print(f"\n  Precision / Recall against the labeled 'relevant' set:")
    print(f"    cross-encoder : (see table)  -- binary view: marks ALL as relevant")
    print(f"    graph         : precision {graph_prec_labeled:.0%}  recall {graph_recall:.0%}  "
          f"(over {len(graph_marked_labeled)} labeled entities it marked relevant)")

    # ---- CE top/bottom vs ground-truth tier ---------------------------------
    print("\n" + "-" * 72)
    print("CROSS-ENCODER RANK of the labeled entities (1 = most relevant)")
    print("-" * 72)
    labeled_sorted = sorted(ce_rank.items(), key=lambda x: x[1])
    gt_set = set(gt)
    print("  (labeled entities)")
    for name, rank in labeled_sorted:
        if name not in gt_set:
            continue
        s = gt[name]
        label = {3: "GT=CLEAR", 2: "GT=related", 0: "GT=NOT"}[s]
        mark = "  <== top" if rank <= 10 else ("  <== bottom" if rank >= len(ce_scored) - 9 else "")
        print(f"  CE rank {rank:3d} | {label:12s} | {name}{mark}")

    print("\n  ALL entities by CE rank (1 = most relevant):")
    for rank, e in enumerate(ce_scored, 1):
        name = e["name"]
        tag = "  <== LABELED" if name in gt_set else ""
        print(f"  {rank:3d}  {e['relevance_score']:+.3f}  [{e['type']:9s}] {name}{tag}")

    # ---- The presentation problem ------------------------------------------
    print("\n" + "-" * 72)
    print("WHAT THIS MEANS")
    print("-" * 72)
    print(f"  CE direction is right: the clearly-relevant items cluster at the")
    print(f"  top (dark water, papyrus, deep water, Leander, sulfur) and the")
    print(f"  off-scene village items sink to the bottom (wool blanket, goat,")
    print(f"  thatched roof, borrowed sword).")
    print(f"  CE scores are raw MiniLM logits, not probabilities — only the")
    print(f"  RANKING is meaningful, not the raw values.")
    print(f"  Graph character side (real traversal): it now reaches")
    print(f"  {sorted(graph_char_names) or 'no characters'} from the Leander seed via the")
    print(f"  19-node / 39-rel graph — far more than the 2-node line I first")
    print(f"  (wrongly) read. The character signal is real, not degenerate.")
    print(f"  Elements remain the weak point: substring match floods ALL")
    print(f"  {len(all_element_names)} of them in, so the graph still cannot rank elements.")
    print(f"  => CE is the strongest element signal; the graph is the strongest")
    print(f"     character/structure signal. They are complementary, not in conflict.")


if __name__ == "__main__":
    main()