"""Hermetic director-pass cost test.

Drives the real ``Summarizer.generate_instr_prompt`` with a counting fake model
and a stubbed retrieval boundary, then asserts the director-pass cost contract:

- ``do_instr=True`` makes exactly one extra LLM call per turn vs. ``do_instr=False``.
- The generated instruction prompt embeds the director's instructions when on,
  and does not when off.
- A second turn with the same seed is served from the ``instructions.json``
  cache (zero extra calls).

The test is fully hermetic: no llama_index, torch, gradio, or network.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Generator

TEST_DIR = Path(__file__).parent
REPO_ROOT = TEST_DIR.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from extensions.dayna_ss.agents.summarizer import Summarizer  # noqa: E402
from extensions.dayna_ss.runtime import runtime  # noqa: E402


class FakeModel:
    """Minimal model backend: counts generate_with_streaming calls, yields scripted text."""

    def __init__(self, scripted_text: str = "DIRECTOR SAYS: analyze the mood; write in a low, gravelly voice."):
        self.scripted_text = scripted_text
        self.call_count = 0

    def generate_with_streaming(self, encoded_prompt: Any, state: dict) -> Generator[str, Any, None]:
        self.call_count += 1
        yield self.scripted_text


def _make_summarizer(fake_model: FakeModel, history_path: Path) -> Summarizer:
    """Build a Summarizer bound to the fake model, with retrieval stubbed out."""
    runtime.configure(
        model_provider=lambda: fake_model,
        stop_provider=lambda: False,
        prompt_builder=lambda prompt, state, **kw: prompt,
        encoder=lambda text, add_bos_token=True: text,
        persistent_ui_state_provider=lambda: {},
        settings_provider=lambda: {},
    )
    s = Summarizer()

    # Stub the retrieval boundary (prepare_context -> retrieve_and_format_context)
    def _stub_retrieve(state, history, **kwargs):
        return {"history": {"internal": []}, "name1": state.get("name1", "User"), "name2": state.get("name2", "Assistant")}

    s.retrieve_and_format_context = _stub_retrieve  # type: ignore[method-assign]
    s.last = SimpleNamespace(
        history_path=history_path,
        context=None,
        schema_parser=None,
        is_new_scene_turn=False,
        force_next_chapter=False,
        force_next_arc=False,
    )
    return s


def run_director_test() -> tuple[bool, list[str]]:
    """Return (passed, [messages])."""
    msgs: list[str] = []
    workdir = tempfile.mkdtemp(prefix="dss_director_")
    history_path = Path(workdir) / "history"
    history_path.mkdir(parents=True, exist_ok=True)

    fake = FakeModel()
    s = _make_summarizer(fake, history_path)

    state = {"name1": "User", "name2": "Assistant", "seed": 42, "context": "Some world context."}
    history = [["User", "The cave mouth looms."], ["Assistant", "She squints into the dark."]]  # __len__ works

    # --- do_instr=False: no director call ---
    fake.call_count = 0
    prompt_off, _, path_off, _ = s.generate_instr_prompt("The cave mouth looms.", state, history, do_instr=False)
    calls_off = fake.call_count
    msgs.append(f"do_instr=False -> {calls_off} LLM calls")

    # --- do_instr=True: one extra director call ---
    fake.call_count = 0
    prompt_on, _, path_on, _ = s.generate_instr_prompt("The cave mouth looms.", state, history, do_instr=True)
    calls_on = fake.call_count
    msgs.append(f"do_instr=True  -> {calls_on} LLM calls")

    # --- same seed again: served from cache ---
    fake.call_count = 0
    prompt_cached, _, path_cached, _ = s.generate_instr_prompt("The cave mouth looms.", state, history, do_instr=True)
    calls_cached = fake.call_count
    msgs.append(f"do_instr=True (cached seed) -> {calls_cached} LLM calls")

    passed = True
    checks = [
        (calls_off == 0, f"do_instr=False should make 0 LLM calls, got {calls_off}"),
        (calls_on == 1, f"do_instr=True should make exactly 1 extra LLM call, got {calls_on}"),
        (calls_cached == 0, f"cached seed should make 0 LLM calls, got {calls_cached}"),
        ("DIRECTOR SAYS" in str(prompt_on), "do_instr=True prompt should embed the director's instructions"),
        ("DIRECTOR SAYS" not in str(prompt_off), "do_instr=False prompt should not embed director instructions"),
        (str(prompt_on) != str(prompt_off), "prompts should differ between director on/off"),
        (str(path_on) == str(path_off) == str(history_path), "history_path should be preserved"),
    ]
    for ok, msg in checks:
        msgs.append(("PASS" if ok else "FAIL") + ": " + msg)
        passed = passed and ok

    # instructions.json cache written
    instr_path = history_path / "instructions.json"
    if instr_path.exists():
        cached = json.loads(instr_path.read_text(encoding="utf-8"))
        msgs.append(f"instructions.json cached {len(cached)} key(s)")
    else:
        msgs.append("FAIL: instructions.json was not written")
        passed = False

    return passed, msgs


if __name__ == "__main__":
    ok, msgs = run_director_test()
    for m in msgs:
        print(m)
    sys.exit(0 if ok else 1)
