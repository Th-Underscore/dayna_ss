#!/usr/bin/env python3
"""Memory-hygiene regression tests (data_summarizer.hygiene).

Covers the live failure family from cyberpunk_thriller__4dbf1435:
sentence restatement inside entities, cross-entity injection of profile
sentences, relationship-row alias storms / concept-spam / junk rows / caps,
plus shape-generality (no hardcoded per-type dispatchers) and idempotency.
"""

import copy
import json
import os
import sys

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_TEST_DIR, "..", "..", "..")))

from extensions.dayna_ss.agents.data_summarizer.hygiene import (  # noqa: E402
    _entity_hygiene_pass,
    canon_name,
    clean_store,
)

SENT_A = ("She prioritizes precise data over broad assumptions, refusing to "
          "give away her routes or codes until the final moment of delivery.")
SHORT = "Locket on desk."
UNIQ = ("Vell once walked the Neon Market district personally to verify the "
        "markers carved into the sluice plates beside the canal.")

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def _ent(desc=None, bio=None, extra=None):
    d = {}
    if desc is not None:
        d["description"] = [desc] if isinstance(desc, str) else desc
    if bio is not None:
        d["biography"] = [bio] if isinstance(bio, str) else bio
    if extra:
        d.update(extra)
    return d


def _stores(*entities):
    return {"characters": {"entries": {f"e{i}": e for i, e in enumerate(entities)}}}


def _n(stores):
    c = 0
    for s in stores.values():
        entries = s.get("entries") or {}
        for v in entries.values():
            if isinstance(v, dict):
                c += json.dumps(v, ensure_ascii=False).count(SENT_A[:60])
    return c


def _run(name, fn):
    try:
        fn()
        print(f"  {PASS}  {name}")
        return True
    except AssertionError as exc:
        print(f"  {FAIL}  {name}: {exc}")
        return False


tests = []


def test(fn):
    tests.append(fn)
    return fn


@test
def intra_entity_keep_first():
    stores = _stores(_ent(desc=SENT_A, bio=f"{SENT_A} {SENT_A}"))
    dropped = _entity_hygiene_pass(
        [(f"characters\x00e{i}", e)
         for i, e in enumerate(stores["characters"]["entries"].values())],
        60, 0.995)
    assert dropped == 2, f"expected 2 drops, got {dropped}"
    assert _n(stores) == 1, f"expected exactly 1 survivor, got {_n(stores)}"


@test
def cross_entity_majority_wins():
    stores = _stores(_ent(bio=SENT_A),
                     _ent(bio=f"{SENT_A} {SENT_A} {SENT_A}"),
                     _ent(desc=SENT_A))
    _entity_hygiene_pass(
        [(f"characters\x00e{i}", e)
         for i, e in enumerate(stores["characters"]["entries"].values())],
        60, 0.995)
    hits = [k for k, v in stores["characters"]["entries"].items()
            if SENT_A[:60] in json.dumps(v)]
    assert hits == ["e1"], f"expected majority holder e1 only, got {hits}"


@test
def tie_break_lexicographic():
    stores = _stores(_ent(bio=SENT_A), _ent(desc=SENT_A))
    _entity_hygiene_pass(
        [(f"characters\x00e{i}", e)
         for i, e in enumerate(stores["characters"]["entries"].values())],
        60, 0.995)
    hits = [k for k, v in stores["characters"]["entries"].items()
            if SENT_A[:60] in json.dumps(v)]
    assert hits == ["e0"], f"expected lexicographic winner e0, got {hits}"


@test
def unique_and_short_sentences_untouched():
    stores = _stores(_ent(desc=UNIQ, bio=f"{SHORT} {SHORT} {SHORT}"))
    before_uniq = UNIQ[:50]
    _entity_hygiene_pass(
        [(f"characters\x00e{i}", e)
         for i, e in enumerate(stores["characters"]["entries"].values())],
        60, 0.995)
    d = stores["characters"]["entries"]["e0"]
    assert before_uniq in json.dumps(d["description"]), "unique sentence was dropped"
    assert json.dumps(d["biography"]).count(SHORT) == 3, \
        "short sentences must be exempt from dedup"


@test
def cosmetic_variants_collapse():
    variant = SENT_A.replace("broad assumptions", "broad assumptions,").replace("  ", " ")
    variant = variant[:-1] + "."  # ensure still >= 60 chars & near-identical
    stores = _stores(_ent(desc=SENT_A, bio=variant))
    _entity_hygiene_pass(
        [(f"characters\x00e{i}", e)
         for i, e in enumerate(stores["characters"]["entries"].values())],
        60, 0.995)
    assert _n(stores) == 1, f"cosmetic variant should collapse, kept {_n(stores)}"


@test
def paraphrase_layers_survive():
    legit = ("She prioritizes precise data above sweeping guesses and never "
             "reveals her routes or codes before the final delivery instant.")
    stores = _stores(_ent(desc=SENT_A, bio=legit))
    _entity_hygiene_pass(
        [(f"characters\x00e{i}", e)
         for i, e in enumerate(stores["characters"]["entries"].values())],
        60, 0.995)
    assert legit[:50] in json.dumps(stores["characters"]["entries"]["e0"]), \
        "paraphrase (not cosmetic clone) must survive 0.995 threshold"


@test
def rel_alias_merge_richest_payload():
    stores = {"elements": {"entries": {"datachip": {
        "relationships": {
            "The Handler": {"relation": "monitors_as_witness", "status": "active",
                            "importance": {"score": 85, "reason": "rich", "faction": ""}},
            "Handler": {"relation": "monitors_as_witness", "status": "",
                        "importance": {"score": 65, "reason": "", "faction": ""}},
            "The Client": {"relation": "held_by", "status": "", "importance": {}},
            "Client": {"relation": "held_by", "status": "delivered",
                       "importance": {"score": 75, "reason": "custody", "faction": ""}},
        }}}}}
    stats = clean_store(stores, summarizer=None, peer_scope=False)
    chip = stores["elements"]["entries"]["datachip"]["relationships"]
    # No characters store here, so no subject titles exist: the shortest
    # spelling wins per display rule (underscore-free preference, then len).
    assert set(chip) == {"Handler", "Client"}, f"alias groups wrong: {sorted(chip)}"
    assert stats["rel_rows_merged"] == 2, f"merged={stats['rel_rows_merged']}"
    assert chip["Handler"]["importance"]["score"] == 85
    assert chip["Client"]["status"] == "delivered"


@test
def junk_rows_pruned():
    stores = {"elements": {"entries": {"x": {"custom_links": {
        "File": {"relation": "that", "status": "", "importance": {}},
        "Real_one": {"relation": "guards", "status": "active",
                     "importance": {"score": 70, "reason": "", "faction": ""}},
    }}}}}
    stats = clean_store(stores, summarizer=None, peer_scope=False)
    links = stores["elements"]["entries"]["x"]["custom_links"]
    assert set(links) == {"Real one"}, f"junk not pruned: {sorted(links)}"
    assert stats["rel_rows_pruned"] == 1


@test
def cap_enforced_known_first():
    rows = {}
    for i in range(30):
        name = f"unknown_partner_{i}"
        imp = {"score": 90 if i % 5 == 0 else 10, "reason": "", "faction": ""}
        rows[name] = {"relation": f"watches_{i}", "status": "active",
                      "importance": imp}
    # Two known partners with modest scores must beat unknowns regardless.
    rows["Juno"] = {"relation": "carries", "status": "active",
                    "importance": {"score": 20, "reason": "", "faction": ""}}
    stores = {"elements": {"entries": {"chip": {"relationships": rows}}}}
    import types

    cfg_stub = types.SimpleNamespace(
        config={"max_relationships_per_entity": 10,
                "dedup_min_sentence_chars": 60})
    clean_store(stores, summarizer=cfg_stub, peer_scope=False)
    kept = stores["elements"]["entries"]["chip"]["relationships"]
    assert len(kept) <= 10, f"cap violated: {len(kept)}"
    assert "Juno" in kept, "known partner must never lose to unknown ones"
    high_scores = [v["importance"]["score"] for k, v in kept.items() if k != "Juno"]
    assert 90 in high_scores, "highest-importance unknowns should win the rest"


@test
def shape_generality_no_type_dispatch():
    # A relationships-shaped collection under ANY field name, ANY store,
    # ANY depth is sanitized — proves there is no per-data-type key map.
    stores = {"widgets": {"entries": {"gizmo": {"nested": [
        {"links": {
            "the_widget_core": {"relation": "powers", "status": "on",
                                "importance": {"score": 50}},
            "Widget_Core": {"relation": "powers", "status": "on",
                            "importance": {}},
        }},
    ]}}}}
    clean_store(stores, summarizer=None, peer_scope=False)
    got = stores["widgets"]["entries"]["gizmo"]["nested"][0]["links"]
    # deterministic spelling: underscore-free shortest variant wins
    assert set(got) == {"Widget Core"}, f"shape-dispatch failed: {sorted(got)}"


@test
def sectioned_event_stores_supported():
    stores = {"events": {"past": {
        "Chapter One": {"summary": f"{SENT_A} Extra unique chapter context here."},
        "Chapter Two": {"summary": f"{SENT_A} Totally different second part."},
    }}}
    _entity_hygiene_pass(
        [(hid, ref) for hid, ref in _ents(stores)], 60, 0.995)
    ch1 = stores["events"]["past"]["Chapter One"]["summary"]
    ch2 = stores["events"]["past"]["Chapter Two"]["summary"]
    total = (ch1.count(SENT_A[:60]) + ch2.count(SENT_A[:60]))
    assert total == 1, f"shared event-summary sentence should keep 1, got {total}"
    assert "Extra unique" in ch1 and "Totally different" in ch2


@test
def full_clean_store_idempotent():
    stores = _stores(_ent(desc=SENT_A, bio=f"{SENT_A} {SENT_A}"),
                     _ent(bio=SENT_A))
    s1 = clean_store(stores, summarizer=None)
    s2 = clean_store(stores, summarizer=None)
    assert s1["dropped_sentences"] > 0, "first pass should drop"
    assert all(v == 0 for v in s2.values()), f"second pass not idempotent: {s2}"


# ---------------------------------------------------------------------------

def _ents(all_subjects_data):
    from extensions.dayna_ss.agents.data_summarizer.hygiene import _entities
    return _entities(all_subjects_data)


def run_hygiene_test():
    print("[hygiene_test]")
    ok = True
    for t in tests:
        ok = _run(t.__name__, t) and ok
    ok = _run(
        "canon_name strips articles/punct/case",
        lambda: (
            (lambda: (_ for _ in ()).throw(AssertionError(canon_name("The_Handler")))
             if canon_name("The_Handler") != canon_name("handler") else None),
            (lambda: (_ for _ in ()).throw(AssertionError(canon_name("Grey Coats")))
             if canon_name("Grey Coats") != canon_name("grey-coats") else None),
        ) and None,
    )
    print(f"[hygiene_test] {'PASS' if ok else 'FAIL'}")
    return ok, ([t.__name__ for t in tests])


if __name__ == "__main__":
    ok, _ = run_hygiene_test()
    sys.exit(0 if ok else 1)
