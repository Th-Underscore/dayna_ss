"""Hermetic regression test for the scene-bounded dialogue window.

Drives ``Summarizer._scene_dialogue_window`` directly (no LLM, no llama_index,
no torch) to assert the Q5 prompt-management fix:

- The recent-dialogue window is capped to ``last_x_max`` and never reaches back
  across a scene boundary.
- A flat fallback (current behavior: ``last 6``) is used when no scene boundary
  is resolvable (first scene / missing ``_message_node``).
- The ``last_x_min`` floor prevents an empty window at the start of a scene.

Scenario used throughout: a story in which each exchange produces a message at
index ``2*turn`` (user) / ``2*turn+1`` (assistant), and a scene's start is
recorded as ``_message_node`` = ``"{message_idx}_1_1"``.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

TEST_DIR = __file__.rsplit("/", 1)[0]
REPO_ROOT = TEST_DIR + "/../../.."
sys.path.insert(0, REPO_ROOT)

from extensions.dayna_ss.agents.summarizer import Summarizer  # noqa: E402


class FakeContextRetriever:
    """Returns a canned current_scene from the caller."""

    def __init__(self, current_scene: dict):
        self._scene = current_scene

    def get_current_scene(self):
        return self._scene


def _make_summarizer() -> Summarizer:
    s = Summarizer()
    s.last = SimpleNamespace(new_scene_start_node=None)
    return s


def _scene_at(exchange_idx: int) -> dict:
    """A current_scene whose start node is at the given exchange (message idx 2*N)."""
    return {"start": {"when": {"_message_node": f"{exchange_idx * 2}_1_1"}}}


def _history(n: int) -> list:
    return [[f"u{i}", f"a{i}"] for i in range(n)]


def _make_current_scene(start_node: str) -> dict:
    return {"start": {"when": {"_message_node": start_node}}}


def run_scene_window_test() -> tuple[bool, list[str]]:
    msgs: list[str] = []
    checks: list[tuple[bool, str]] = []

    s = _make_summarizer()

    # 1. Scene-bounded: scene started at exchange 70, history has 100 -> window = 30,
    #    clamped to last_x_max (8).
    w1 = s._scene_dialogue_window(_history(100), FakeContextRetriever(_scene_at(70)), {})
    checks.append((w1 == 8, f"scene-bounded window = {w1}, expected 8 (clamped to max)"))

    # 2. Scene-bounded, non-clamped: scene started at exchange 96, history 100 -> window 4.
    w2 = s._scene_dialogue_window(_history(100), FakeContextRetriever(_scene_at(96)), {})
    checks.append((w2 == 4, f"scene-bounded short window = {w2}, expected 4"))

    # 3. Fresh scene (start == current history length, boundary at/after len):
    #    nothing to bound by yet -> whole history capped at last_x_max.
    w3 = s._scene_dialogue_window(_history(100), FakeContextRetriever(_scene_at(100)), {})
    checks.append((w3 == 8, f"fresh-scene window = {w3}, expected 8 (whole history, capped)"))

    # 3b. Scene one exchange old (boundary inside history, tiny window): last_x_min floor.
    w3b = s._scene_dialogue_window(_history(100), FakeContextRetriever(_scene_at(99)), {})
    checks.append((w3b == 2, f"1-exchange-old scene = {w3b}, expected 2 (last_x_min floor)"))

    # 4. Flat fallback when no current_scene is resolvable.
    w4 = s._scene_dialogue_window(_history(100), FakeContextRetriever({}), {})
    checks.append((w4 == 6, f"flat fallback = {w4}, expected 6"))

    # 5. Flat fallback when _message_node is missing from the scene dict.
    w5 = s._scene_dialogue_window(_history(100), FakeContextRetriever({"start": {"when": {}}}), {})
    checks.append((w5 == 6, f"missing-node fallback = {w5}, expected 6"))

    # 6. Never exceeds history length.
    w6 = s._scene_dialogue_window(_history(3), FakeContextRetriever(_scene_at(96)), {})
    checks.append((w6 == 3, f"short-history window = {w6}, expected 3 (min(history_len, max))"))

    # 7. new_scene_start_node fallback when current_scene has no boundary.
    s2 = _make_summarizer()
    s2.last.new_scene_start_node = "200_1_1"  # exchange 100
    w7 = s2._scene_dialogue_window(_history(150), FakeContextRetriever({}), {})
    checks.append((w7 == 8, f"new_scene_start_node fallback = {w7}, expected 8"))

    # 8. Explicit last_x_max override is honored.
    w8 = s._scene_dialogue_window(
        _history(200), FakeContextRetriever(_scene_at(70)), {"last_x_max": 4, "last_x_min": 1}
    )
    checks.append((w8 == 4, f"last_x_max override = {w8}, expected 4"))

    passed = True
    for ok, msg in checks:
        msgs.append(("PASS" if ok else "FAIL") + ": " + msg)
        passed = passed and ok
    return passed, msgs


if __name__ == "__main__":
    ok, msgs = run_scene_window_test()
    for m in msgs:
        print(m)
    sys.exit(0 if ok else 1)
