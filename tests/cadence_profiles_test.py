"""Hermetic regression test for unit-cadence profile resolution.

Decision ledger #3/#37: chapter/arc numeric bounds are calibration, not truth.
They stay in the schema (compressed numbers) and move via named config
profiles ("campaign" = true-scale) instead of editing schemas or duplicating
numbers in code. _resolve_cadence is the single precedence point:
profile overlay > schema defaults > code fallback.

Asserts:

- No profile: schema defaults pass through untouched (both units).
- Campaign overlay replaces bounds without touching other defaults keys.
- Unknown / empty profile name is inert.
- Missing schema defaults fall back to code constants, still overlaid.
- Garbage values in defaults fall back safely instead of crashing the gate.
"""

from __future__ import annotations

import sys

TEST_DIR = __file__.rsplit("/", 1)[0]
REPO_ROOT = TEST_DIR + "/../../.."
sys.path.insert(0, REPO_ROOT)

from extensions.dayna_ss.agents.data_summarizer.archives import (  # noqa: E402
    UNIT_CADENCE_PROFILES,
    _resolve_cadence,
)

SCENEAGG_CHAPTER_DEFAULTS = {
    "suggested_min_scenes": 4,
    "suggested_max_scenes": 8,
    "max_scenes_before_required": 10,
}
SCENEAGG_ARC_DEFAULTS = {
    "suggested_min_chapters": 2,
    "suggested_max_chapters": 5,
    "max_chapters_before_required": 8,
}


def run_cadence_profile_test() -> tuple[bool, list[str]]:
    msgs: list[str] = []
    checks: list[tuple[bool, str]] = []

    # 1. No profile -> schema defaults pass through.
    checks += [
        (_resolve_cadence(SCENEAGG_CHAPTER_DEFAULTS, "chapters", "scenes", (4, 8, 10), None)
         == (4, 8, 10), "no profile keeps chapter schema defaults"),
        (_resolve_cadence(SCENEAGG_ARC_DEFAULTS, "arcs", "chapters", (2, 5, 8), None)
         == (2, 5, 8), "no profile keeps arc schema defaults"),
    ]

    # 2. Explicit compressed profile == shipped schema numbers.
    checks += [
        (_resolve_cadence(SCENEAGG_CHAPTER_DEFAULTS, "chapters", "scenes", (4, 8, 10), "compressed")
         == (4, 8, 10), "compressed overlay matches shipped chapter numbers"),
    ]

    # 3. Campaign overlay wins over schema defaults; arc defaults untouched keys.
    ch = _resolve_cadence(SCENEAGG_CHAPTER_DEFAULTS, "chapters", "scenes", (4, 8, 10), "campaign")
    ar = _resolve_cadence(SCENEAGG_ARC_DEFAULTS, "arcs", "chapters", (2, 5, 8), "campaign")
    checks += [
        (ch == tuple(UNIT_CADENCE_PROFILES["campaign"]["chapters"][k] for k in
                     ("suggested_min_scenes", "suggested_max_scenes", "max_scenes_before_required")),
         f"campaign overlays chapter bounds (got {ch})"),
        (ar == tuple(UNIT_CADENCE_PROFILES["campaign"]["arcs"][k] for k in
                     ("suggested_min_chapters", "suggested_max_chapters", "max_chapters_before_required")),
         f"campaign overlays arc bounds (got {ar})"),
        (ch[1] > ch[0], "campaign chapter bounds stay ordered min < max"),
        (ar[1] > ar[0], "campaign arc bounds stay ordered min < max"),
    ]

    # 4. Unknown profile name is inert.
    checks += [
        (_resolve_cadence(SCENEAGG_CHAPTER_DEFAULTS, "chapters", "scenes", (4, 8, 10), "nope")
         == (4, 8, 10), "unknown profile ignored"),
    ]

    # 5. Missing schema defaults: code fallback, then campaign over it.
    checks += [
        (_resolve_cadence(None, "chapters", "scenes", (4, 8, 10), None) == (4, 8, 10),
         "missing defaults use code fallback"),
        (_resolve_cadence(None, "chapters", "scenes", (4, 8, 10), "campaign")
         == tuple(UNIT_CADENCE_PROFILES["campaign"]["chapters"][k] for k in
                  ("suggested_min_scenes", "suggested_max_scenes", "max_scenes_before_required")),
         "campaign overlays even without schema defaults"),
    ]

    # 6. Garbage values degrade to fallback instead of raising inside the gate.
    garbage = {"suggested_min_scenes": "x", "suggested_max_scenes": None}
    checks += [
        (_resolve_cadence(garbage, "chapters", "scenes", (4, 8, 10), None) == (4, 8, 10),
         "garbage defaults fall back safely"),
    ]

    ok = all(passed for passed, _ in checks)
    for passed, name in checks:
        msgs.append(("PASS  " if passed else "FAIL  ") + name)
    if ok:
        msgs.append("[cadence_profiles] PASS")
    return ok, msgs


if __name__ == "__main__":
    ok, msgs = run_cadence_profile_test()
    for m in msgs:
        print(m)
    sys.exit(0 if ok else 1)
