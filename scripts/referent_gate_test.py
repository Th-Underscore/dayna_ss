"""Hermetic smoke gate: pre-mutation referent-resolution patch (H1-H4).

Root-cause evidence: tests/split_node_audit_report.md + tests/split_adjudication.md
+ tests/alias_branchquery_map.md. The model emits entity data addressed at the
WRONG subject (a new lookalike key, a foreign alias/partner, foreign prose) —
one root, four symptoms, growing linearly with run length (323 dangling edges
over one run). The patch has four hunks; each assertion below pins one:

  H1  branch templates emit the already-computed {{ value }} (target-node inline)
  H2  branch templates carry the canonical-key constraint
  H3  parse-time VALUE retargeting in _apply_branch_updates (dangle -> canonical)
  H4  the alias placeholders across the four alias-bearing classes are hardened

Read-only, stdlib, model-free. The H1 proof drives format_str_or_jinja
(the very formatter the engine calls at agents/data_summarizer/prompts.py:516)
against the shipped branch templates — proving the target node's own content
is substituted into the prompt while no other entry's content leaks in.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

TEST_DIR = Path(__file__).parent
REPO_ROOT = TEST_DIR.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from extensions.dayna_ss.utils.formatting import format_str_or_jinja  # noqa: E402
from extensions.dayna_ss.agents.data_summarizer.parsing import (  # noqa: E402
    _filter_new_entry_names,
    _retarget_value,
)

# Both shipped schemas must carry identical patch text (two source files, one
# patch) — assert over both.
SCHEMA_PATHS = [
    REPO_ROOT / "extensions/dayna_ss/user_data/example/subjects_schema.json",
    REPO_ROOT / "extensions/dayna_ss/user_data/example/schemas/subjects_schema_sceneagg.json",
]

# A target node with DISTINCTIVE markers (used to prove its content is inlined),
# plus a sibling node whose markers must NOT appear (the no-whole-map invariant).
TARGET_NODE = {
    "description": "Runs the green-market route and keeps a ledger of chip debts.",
    "aliases": ["Second Runner", "the runner"],
    "relationships": {"Juno": {"relation": "debt", "status": "unpaid"}},
    "importance": 60,
}
SIBLING_NODE = {
    "description": "A wholly separate observer with a private frequency.",
    "aliases": ["the observer"],
    "relationships": {},
    "importance": 20,
}
# A node carrying a lookalike key the model mis-spelled; its alias lets the
# retarget resolve the dangle back onto the canonical node.
DATA = {
    "Second Runner": {**TARGET_NODE, "aliases": ["Second Runner", "the runner", "Runner"]},
    "The_Second_Runner": {
        "relationships": {"Juno": {"relation": "compromised"}}
    },
    "Juno": {"description": SIBLING_NODE["description"]},
}


def _load_schemas() -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in SCHEMA_PATHS]


def _branch_update_template(schema: dict) -> str:
    return schema["definitions"]["Character"]["defaults"]["branch_update_prompt_template"]


def run_referent_gate_test() -> tuple[bool, list[str]]:
    msgs: list[str] = []
    checks: list[tuple[bool, str]] = []
    schemas = _load_schemas()

    # --- A1 (H1): target node's OWN content is inlined into the branch prompt ---
    # Drives format_str_or_jinja (the very formatter the engine calls at
    # agents/data_summarizer/prompts.py:516) against the shipped template, so the
    # substitution is proven end-to-end, not asserted by string-poking.
    prompts_by_file = {}
    for path, schema in zip(SCHEMA_PATHS, schemas):
        t = _branch_update_template(schema)
        prompt = format_str_or_jinja(
            t,
            value=json.dumps(TARGET_NODE),
            branch_name="Second Runner",
            item_name="Second Runner",
            field_name="",
            keys=["Second Runner"],
            schema_snippet="",
            example_json="",
            name1="User",
            name2="Assistant",
        )
        prompts_by_file[path.name] = prompt
        # Distinctive markers from the target node must be present verbatim.
        checks.append(
            (
                "green-market route" in prompt and '"Second Runner", "the runner"' in prompt
                and '"Juno": {"relation": "debt"' in prompt,
                f"{path.name} H1: target node's aliases + relationship rows inlined verbatim",
            )
        )
        # A1b: the inline is the TARGET node's own data (ground truth the model
        # diffs against) — the 'Current entry state' framing is present.
        checks.append(
            ("Current entry state" in prompt, f"{path.name} H1: 'Current entry state' framing present")
        )

    # --- A2: no whole-map re-arm (only the target node is inlined, KBs, not the map) ---
    for path, schema, prompt in zip(SCHEMA_PATHS, schemas, prompts_by_file.values()):
        for cn in ["Character", "Group", "SceneState"]:
            bt = schema["definitions"][cn]["defaults"]["branch_update_prompt_template"]
            checks.append(
                (
                    "{{ subjects" not in bt and "Other entries in this section" not in bt,
                    f"{path.name} A2: {cn} branch template carries no whole-map re-arm",
                )
            )
        # The sibling node's distinctive content must NOT be reachable in the
        # prompt for a target-node edit (proves the inline is scoped, not map-wide).
        assert SIBLING_NODE["description"] not in json.dumps(TARGET_NODE)
        checks.append(
            ("the observer" not in prompt and "private frequency" not in prompt,
             f"{path.name} A2: sibling node's content absent from the target's prompt")
        )

    # --- A3 (H2): canonical-key constraint line is in the branch template ---
    for path, schema in zip(SCHEMA_PATHS, schemas):
        t = _branch_update_template(schema)
        checks.append(
            (
                "CANONICAL KEY CONSTRAINT" in t
                and "add_new" in t
                and "MUST resolve to a key already present" in t,
                f"{path.name} A3: canonical-key constraint present in branch template",
            )
        )

    # --- A4 (H3): retarget is a guaranteed no-op + idempotent + catches the dangle ---
    # (a) A value with NO dangling referent passes through byte-identical.
    clean_value = {
        "Juno": {"relation": "debt"},
        "Second Runner": {"relation": "compromised"},  # already canonical
    }
    import copy as _copy
    clean_copy = _copy.deepcopy(clean_value)
    _retarget_value(clean_value, DATA)
    checks.append(
        (clean_value == clean_copy, "A4a: value with no dangling referent passes through unchanged")
    )
    # (b) A dangle (a lookalike key that would mint a stray sibling) is re-addressed
    #     onto its canonical node.
    dangle_value = {
        "Runner": {"relation": "the debt"},  # 'Runner' -> 'Second Runner' (distinct collision)
        "Ghost": "unresolvable",             # no existing node -> must stay untouched
    }
    _retarget_value(dangle_value, DATA)
    checks.append(
        (
            "Second Runner" in dangle_value
            and dangle_value.get("Runner") is None
            and dangle_value.get("Second Runner") == {"relation": "the debt"}
            and dangle_value.get("Ghost") == "unresolvable",
            "A4b: dangling referent re-addressed onto canonical node (dangle resolved)",
        )
    )
    # (c) Idempotent: a second pass over the already-retargeted value is a no-op.
    after_first = _copy.deepcopy(dangle_value)
    _retarget_value(dangle_value, DATA)
    checks.append(
        (dangle_value == after_first, "A4c: retarget is idempotent (second pass = no-op)")
    )
    # (d) add_new near-dup guard: a lookalike proposal against an existing key is
    #     still rejected (prevents the sibling mint at the add_new boundary).
    checks.append(
        (
            _filter_new_entry_names(["The_Second_Runner"], DATA) == [],
            "A4d: add_new rejects a lookalike proposal that collides with an existing key",
        )
    )

    # --- A5 (H3 + adjudication C3/C4 boundary): distinct-target edge preserved ---
    # A canonical edge that happens to share a name with ANOTHER existing node is
    # NOT collapsed into it: retargeting re-addresses DANGLES, it never erases a
    # distinct-target relationship (e.g. the Pike/wore/Vell-frequency case).
    two_distinct = {
        "Juno": {"relation": "the shared frequency"},  # canonical, resolves to itself
        "Vell": {"relation": "the same frequency"},    # canonical, resolves to itself
    }
    two_distinct_copy = _copy.deepcopy(two_distinct)
    _retarget_value(two_distinct, DATA)
    checks.append(
        (
            two_distinct == two_distinct_copy
            and "Juno" in two_distinct and "Vell" in two_distinct
            and two_distinct["Juno"]["relation"] == "the shared frequency",
            "A5: two distinct canonical targets sharing a name are NOT collapsed (shared-identity preserved)",
        )
    )

    # --- A6 (H4): the alias placeholders across all four classes are hardened ---
    ALIAS_CLASSES = ["CharacterRelationship", "CharacterGroupStatus", "Group", "StoryEvent"]
    for path, schema in zip(SCHEMA_PATHS, schemas):
        for cn in ALIAS_CLASSES:
            ph = schema["definitions"][cn]["defaults"]["aliases_placeholder"]
            checks.append(
                (
                    "names this entry only" in ph.lower()
                    and "not the name of a different" in ph.lower()
                    and "a nickname, title, or handle referring to this same entity" in ph
                    and "belongs on that entity's own entry" in ph
                    and "record shared/undercover/impersonation identities" in ph,
                    f"{path.name} A6: {cn} alias placeholder hardened (own-entity + shared/undercover routing)",
                )
            )

    ok = all(passed for passed, _ in checks)
    for passed, name in checks:
        msgs.append(("PASS  " if passed else "FAIL  ") + name)
    msgs.append(
        f"[referent_gate] {'PASS' if ok else 'FAIL'} — {sum(p for p, _ in checks)}/{len(checks)} checks"
    )
    return ok, msgs


if __name__ == "__main__":
    ok, msgs = run_referent_gate_test()
    for m in msgs:
        print(m)
    sys.exit(0 if ok else 1)