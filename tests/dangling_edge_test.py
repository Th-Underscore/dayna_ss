"""Hermetic gate: entity-graph dangling-edge audit (retrieval-oracle integrity).

Root-cause evidence: turn-84 replay on the live 4dbf1435 run (176 nodes /
525 edges) proved the relevance oracle runs on a graph with a data-integrity
gap. 6 of the 7 top-RANKED events in the scene-29 seed were PHANTOM labels --
``Cadence Protocol Established``, ``Grid Audit Completion``, ``Identity
Revelation``, ``Imposter Detection``, ``Broker of the Double-Sold Contract``,
``Sensor Repair and Navigation`` -- each with ZERO occurrences in raw
``events.json`` and NO backing graph node, yet ranked #1-#4 at scores 85-87.
18.1% of all edges (95/525) dangled; all 95 were target-missing, split across
``milestones`` (38) + ``relationships`` (48) + ``characters`` (9). Three were
pure type mismatches (edge says ``event:X`` but the node is stored
``scene:X``) -- the split-node / lookalike class we already root-caused for
``Second Runner`` -> ``The_Second_Runner``.

The consumer path (context_retriever.py:1540) silently drops non-stored event
names, so the reply text stays clean -- but the ``relevant_entities`` scoring
pool (line 1486) carries the 6 ghosts at the top of the budget. Phase 4 (hybrid
scoring + injection merger) ranks and packs from exactly that pool, so it would
amplify the contamination rather than the signal. This audit is the standing
regression that catches the write-time re-naming BEFORE it corrupts the oracle.

Two failure classes are distinguished (different fix levers):
  * PHANTOM       -- the referent's bare name exists NOWHERE. A dangling
                     referent minted at write-time. Caught by the pre-mutation
                     referent-resolution gate (docs/plans/referent_resolution_gate.md).
  * TYPE MISMATCH -- the bare name EXISTS but under a different type prefix
                     (``event:X`` edge vs a ``scene:X`` node). The split-node /
                     alias-mis-addressing class; caught by the same gate's
                     canonical-key constraint + value retargeting.

Read-only, stdlib, model-free, no disk/WSL contact. The audit is a pure
function over ``graph.nodes`` + ``graph.relationships`` and is exercised
against synthetic fixtures that mirror the turn-84 shapes in miniature, so the
gate is hermetic (isolated) and the report is full-verbatim (no truncation).
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

TEST_DIR = Path(__file__).parent
REPO_ROOT = TEST_DIR.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from extensions.dayna_ss.rag.structured_rag.entity_graph import (  # noqa: E402
    EntityNode,
    Relationship,
)


def audit_dangling_edges(graph: SimpleNamespace) -> dict:
    """Pure audit over a graph's in-memory ``nodes`` + ``relationships``.

    No side effects. Reads only ``graph.nodes`` (a mapping keyed by node id)
    and ``graph.relationships`` (a sequence of ``Relationship``). Classifies
    every edge:

      * clean         -- both endpoints have a backing node.
      * dangling      -- at least one endpoint has NO backing node.

    Every dangling endpoint is further classified:
      * "phantom"       -- the bare name exists nowhere in the graph.
      * "type_mismatch" -- the bare name exists, but under a different type
                           prefix (the split-node / lookalike class).

    Returns a full-verbatim report dict (never truncated).
    """
    total = len(graph.relationships)
    clean = 0
    dangling_edges = 0
    endpoints = []  # full-verbatim per-dangling-endpoint records
    by_side = Counter()
    by_field = Counter()
    by_class = Counter()

    for r in graph.relationships:
        src_ok = r.source_id in graph.nodes
        tgt_ok = r.target_id in graph.nodes
        if src_ok and tgt_ok:
            clean += 1
            continue
        dangling_edges += 1
        for side, endpoint, ok in (
            ("source", r.source_id, src_ok),
            ("target", r.target_id, tgt_ok),
        ):
            if ok:
                continue
            _, _, bare = endpoint.partition(":")
            # A backing node under the SAME bare name but any other prefix.
            existing_as = [nid for nid in graph.nodes
                            if bare and nid.endswith(":" + bare)]
            cls = "type_mismatch" if existing_as else "phantom"
            by_side[side] += 1
            by_field[r.field_name] += 1
            by_class[cls] += 1
            endpoints.append({
                "edge": (r.source_id, r.target_id, r.field_name, r.importance),
                "endpoint": endpoint,
                "side": side,
                "class": cls,
                "existing_as": existing_as,
            })

    return {
        "total_edges": total,
        "clean_edges": clean,
        "dangling_edges": dangling_edges,
        "dangling_rate": (100.0 * dangling_edges / total) if total else 0.0,
        "by_side": dict(by_side),
        "by_field": dict(by_field),
        "by_class": dict(by_class),
        "endpoints": endpoints,
    }


def _node(node_id: str, node_type: str) -> EntityNode:
    name = node_id.partition(":")[2]
    return EntityNode(id=node_id, type=node_type, name=name, data={})


def _build_fixture() -> SimpleNamespace:
    """A minimal graph mirroring the turn-84 dangling shapes in miniature.

    Node set (backing):
      character:Juno, character:Second Runner, character:The Handler
      event:Second Runner Identification      (real -- has backing)
      scene:Handler's Relay Switch            (stored scene:, NOT event:)
    Missing (no backing anywhere):
      event:Cadence Protocol Established      (phantom target)
      character:Ghost                         (phantom source)

    Edge set (5):
      1 CLEAN          Juno -> event:Second Runner Identification
      2 CLEAN          Juno -> character:The Handler
      3 PHANTOM        Juno -> event:Cadence Protocol Established (target absent)
      4 TYPE MISMATCH  The Handler -> event:Handler's Relay Switch
                       (edge says event:, backing node is scene:)
      5 PHANTOM        character:Ghost -> Juno  (source absent)
    """
    nodes = {
        "character:Juno": _node("character:Juno", "character"),
        "character:Second Runner": _node("character:Second Runner", "character"),
        "character:The Handler": _node("character:The Handler", "character"),
        "event:Second Runner Identification": _node("event:Second Runner Identification", "event"),
        "scene:Handler's Relay Switch": _node("scene:Handler's Relay Switch", "scene"),
    }
    relationships = [
        # 1 CLEAN
        Relationship("character:Juno", "event:Second Runner Identification",
                     "milestone", "", field_name="milestones", importance=85),
        # 2 CLEAN
        Relationship("character:Juno", "character:The Handler",
                     "debtor", "unpaid", field_name="relationships", importance=50),
        # 3 PHANTOM (target has no backing node anywhere)
        Relationship("character:Juno", "event:Cadence Protocol Established",
                     "milestone", "", field_name="milestones", importance=90),
        # 4 TYPE MISMATCH (edge says event:X, node stored scene:X)
        Relationship("character:The Handler", "event:Handler's Relay Switch",
                     "milestone", "", field_name="milestones", importance=85),
        # 5 PHANTOM (source has no backing node)
        Relationship("character:Ghost", "character:Juno",
                     "debtor", "unpaid", field_name="relationships", importance=40),
    ]
    return SimpleNamespace(nodes=nodes, relationships=relationships)


def _build_clean_graph() -> SimpleNamespace:
    """A fully-resolved graph (every edge's both endpoints have backing) --
    used to prove the audit raises ZERO false positives."""
    nodes = {
        "character:A": _node("character:A", "character"),
        "character:B": _node("character:B", "character"),
        "event:X": _node("event:X", "event"),
    }
    relationships = [
        Relationship("character:A", "character:B", "ally", "", "relationships", 60),
        Relationship("character:B", "event:X", "milestone", "", "milestones", 70),
        Relationship("character:A", "event:X", "milestone", "", "milestones", 75),
    ]
    return SimpleNamespace(nodes=nodes, relationships=relationships)


def run_dangling_edge_test() -> tuple[bool, list[str]]:
    msgs: list[str] = []
    checks: list[tuple[bool, str]] = []
    report = audit_dangling_edges(_build_fixture())
    endpoints = {e["endpoint"]: e for e in report["endpoints"]}

    # --- C1: the audit is a pure no-op on a fully-resolved graph (zero false
    #        positives) -- a clean graph must yield dangling=0, clean=total.
    clean_report = audit_dangling_edges(_build_clean_graph())
    checks.append(
        (clean_report["dangling_edges"] == 0
         and clean_report["clean_edges"] == 3
         and clean_report["endpoints"] == [],
         "C1: audit is a no-op on a fully-resolved graph (0 false positives)"),
    )

    # --- C2: a phantom endpoint (bare name exists NOWHERE) is classified
    #        'phantom', not silently passed over.
    c2 = (
        endpoints.get("event:Cadence Protocol Established", {}).get("class") == "phantom"
        and endpoints.get("character:Ghost", {}).get("class") == "phantom"
    )
    checks.append((c2, "C2: dangling referent with no backing node classified 'phantom'"))

    # --- C3: a type-mismatch endpoint (edge says event:X, node stored scene:X)
    #        is classified 'type_mismatch' AND names the alternate backing node
    #        -- the split-node/lookalike class, distinct from a phantom.
    c3 = (
        endpoints.get("event:Handler's Relay Switch", {}).get("class") == "type_mismatch"
        and endpoints.get("event:Handler's Relay Switch", {}).get("existing_as")
        == ["scene:Handler's Relay Switch"]
    )
    checks.append(
        (c3,
         "C3: event:/scene: type-mismatch classified 'type_mismatch' + names the real backing node"),
    )

    # --- C4: a source-side dangle (source has no backing node) is detected on
    #        the SOURCE side, not only the target side.
    checks.append(
        (report["by_side"].get("source") == 1,
         "C4: source-side dangling edge detected (source has no backing node)"),
    )

    # --- C5: exact per-field / per-side / per-class tallies match the fixture
    #        -- proof the breakdown is precise, not approximate.
    c5 = (
        report["total_edges"] == 5
        and report["clean_edges"] == 2
        and report["dangling_edges"] == 3
        and report["by_side"] == {"target": 2, "source": 1}
        and report["by_field"] == {"milestones": 2, "relationships": 1}
        and report["by_class"] == {"phantom": 2, "type_mismatch": 1}
    )
    checks.append((c5, "C5: per-field / per-side / per-class tallies exact"))

    # --- C6: the dangling rate is computed correctly (3/5 = 60.0%).
    checks.append(
        (abs(report["dangling_rate"] - 60.0) < 1e-9,
         "C6: dangling rate computed correctly (60.0%)"),
    )

    ok = all(passed for passed, _ in checks)
    for passed, name in checks:
        msgs.append(("PASS  " if passed else "FAIL  ") + name)

    # Full-verbatim audit report (no truncation) -- the analysis surface the
    # user needs to confirm the classification is sound, edge by edge.
    msgs.append("[dangling_edge] audit report (turn-84-shape fixture):")
    msgs.append(
        "  total=%d clean=%d dangling=%d rate=%.1f%%"
        % (report["total_edges"], report["clean_edges"],
           report["dangling_edges"], report["dangling_rate"])
    )
    msgs.append("  by_side=%s" % report["by_side"])
    msgs.append("  by_field=%s" % report["by_field"])
    msgs.append("  by_class=%s" % report["by_class"])
    for e in sorted(report["endpoints"], key=lambda x: x["edge"][1]):
        src, tgt, fname, imp = e["edge"]
        msgs.append(
            "  %-28s -> %-32s f=%-13s i=%-3d [%-12s] existing_as=%s"
            % (src, tgt, fname, imp, e["class"], e["existing_as"] or "-")
        )
    msgs.append(
        "[dangling_edge] %s -- %d/%d checks" % ("PASS" if ok else "FAIL",
                                                 sum(p for p, _ in checks),
                                                 len(checks))
    )
    return ok, msgs


if __name__ == "__main__":
    ok, msgs = run_dangling_edge_test()
    for m in msgs:
        print(m)
    sys.exit(0 if ok else 1)