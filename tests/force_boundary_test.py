"""Hermetic regression test for forced chapter/arc boundary staging.

Root cause being guarded (found while building the aggregation-fix live smoke):
``NEXT CHAPTER:`` / ``NEXT ARC:`` set ``summarizer.last.force_next_*`` in
production's chat_input_modifier, but ``get_retrieval_context`` rebuilds
``SummarizationContextCache`` for every exchange (the history-path hash changes),
so the flags were dropped before ``_run_boundary_checks`` ever read them. The fix
stages the intent in ``runtime.persistent_ui_state``; ``prepare_context``
consumes it onto the fresh cache via ``_consume_forced_unit_boundaries``.

Drives ``Summarizer.prepare_context`` with a stubbed
``retrieve_and_format_context`` (no LLM, no retrieval stack) to assert:

- Staged ``force_next_chapter`` / ``force_next_arc`` land on the CURRENT cache,
  flag a manual scene turn, and are cleared after consumption.
- An empty UI state consumes nothing.
- Consumption composes with the NEXT SCENE: prefix path (both flag scene turn).
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

TEST_DIR = __file__.rsplit("/", 1)[0]
REPO_ROOT = TEST_DIR + "/../../.."
sys.path.insert(0, REPO_ROOT)

from extensions.dayna_ss.agents.summarizer import Summarizer  # noqa: E402
from extensions.dayna_ss.runtime import runtime  # noqa: E402


def _make_summarizer(ui_state: dict) -> Summarizer:
    s = Summarizer()
    s.retrieve_and_format_context = lambda state, history, **kw: {"stub": True}
    runtime.configure(
        persistent_ui_state_provider=lambda: ui_state,
    )
    return s


def _fresh_last() -> SimpleNamespace:
    return SimpleNamespace(
        is_new_scene_turn=False,
        is_new_scene_auto_detected=True,
        new_scene_start_node=None,
        force_next_chapter=False,
        force_next_arc=False,
    )


def run_force_boundary_test() -> tuple[bool, list[str]]:
    msgs: list[str] = []
    checks: list[tuple[bool, str]] = []

    # 1. Chapter force staged -> consumed onto fresh cache, cleared afterwards.
    ui = {"force_next_chapter": True}
    s = _make_summarizer(ui)
    s.last = _fresh_last()
    out_input, _cs = s.prepare_context("hello", {}, [])
    checks += [
        (s.last.force_next_chapter is True, "staged force_next_chapter lands on current cache"),
        (s.last.is_new_scene_turn is True, "forced chapter flags a manual scene turn"),
        (s.last.is_new_scene_auto_detected is False, "forced chapter marks manual trigger"),
        (ui.get("force_next_chapter") is False, "staged force_next_chapter cleared after consumption"),
        (s.last.force_next_arc is False, "unrelated force stays off"),
        (out_input == "hello", "user input passes through unmodified"),
    ]

    # 2. Arc force staged -> same contract.
    ui = {"force_next_arc": True}
    s = _make_summarizer(ui)
    s.last = _fresh_last()
    s.prepare_context("hi", {}, [])
    checks += [
        (s.last.force_next_arc is True, "staged force_next_arc lands on current cache"),
        (s.last.is_new_scene_turn is True, "forced arc flags a manual scene turn"),
        (ui.get("force_next_arc") is False, "staged force_next_arc cleared after consumption"),
        (s.last.force_next_chapter is False, "unrelated force stays off"),
    ]

    # 3. Empty UI state -> nothing consumed, scene turn untouched.
    ui: dict = {}
    s = _make_summarizer(ui)
    s.last = _fresh_last()
    s.prepare_context("hi", {}, [])
    checks += [
        (s.last.is_new_scene_turn is False, "no staged intent leaves scene turn alone"),
        (s.last.force_next_chapter is False and s.last.force_next_arc is False,
         "no staged intent leaves unit forces off"),
    ]

    # 4. Composes with the NEXT SCENE: prefix (both mechanisms on one exchange).
    ui = {"force_next_chapter": True}
    s = _make_summarizer(ui)
    s.last = _fresh_last()
    out_input, _cs = s.prepare_context("NEXT SCENE: at the docks", {}, [])
    checks += [
        (out_input == "at the docks", "NEXT SCENE: prefix stripped from user input"),
        (s.last.is_new_scene_turn is True and s.last.force_next_chapter is True,
         "prefix + staged chapter force compose on one cache"),
    ]

    ok = all(passed for passed, _ in checks)
    for passed, name in checks:
        msgs.append(("PASS  " if passed else "FAIL  ") + name)
    if ok:
        msgs.append("[forced_unit_boundaries] PASS")
    return ok, msgs


if __name__ == "__main__":
    ok, msgs = run_force_boundary_test()
    for m in msgs:
        print(m)
    sys.exit(0 if ok else 1)
