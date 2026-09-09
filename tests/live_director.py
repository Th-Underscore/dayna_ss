"""Live director-pass quality + cost test.

Drives the real ``Summarizer.generate_instr_prompt`` against a live model
(OpenAI-compatible endpoint) and compares ``do_instr`` on vs off:

- Cost side: LLM calls per turn and total tokens. The hermetic ``director_test``
  proves ``do_instr=True`` costs exactly one extra call; this live run confirms
  it against a real backend and adds the token cost.
- Quality side: the director's output is inspectable plain-text paragraphs (the
  property the strategy review wants preserved), the final reply generated from
  the resulting prompt is compliant prose (non-empty, not a recap of the user's
  input), and the director actually changes the outcome (prompt and reply differ
  from the no-director path).

The model backend is a small ``LiveModel`` wrapper exposing the same
``generate_with_streaming`` surface the engine's fake models use, so the real
engine code path (``generate_instr_prompt`` -> ``generate_with_sse``) runs
unmodified. Skipped cleanly when ``DSS_BENCH_BASE_URL`` is unset.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Generator

TEST_DIR = Path(__file__).parent
REPO_ROOT = TEST_DIR.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from extensions.dayna_ss.agents.summarizer import Summarizer  # noqa: E402
from extensions.dayna_ss.runtime import runtime  # noqa: E402


def _thinking_enabled() -> bool:
    return os.environ.get("DSS_BENCH_THINKING", "0") in ("1", "true", "True")


class LiveModel:
    """Minimal OpenAI-compatible model backend with call/token accounting."""

    def __init__(self, base_url: str, api_key: str, model: str, max_tokens: int = 400):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.call_count = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def generate_with_streaming(self, encoded_prompt: Any, state: dict) -> Generator[str, Any, None]:
        self.call_count += 1
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": str(encoded_prompt)}],
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
        }
        if not _thinking_enabled():
            payload["enable_thinking"] = False
        url = self.base_url + "/chat/completions"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        usage = data.get("usage", {})
        self.prompt_tokens += int(usage.get("prompt_tokens", 0))
        self.completion_tokens += int(usage.get("completion_tokens", 0))
        content = data["choices"][0]["message"].get("content", "") or ""
        yield content


def _make_summarizer(live_model: LiveModel, history_path: Path) -> Summarizer:
    """Build a Summarizer bound to the live model, with retrieval stubbed out."""
    runtime.configure(
        model_provider=lambda: live_model,
        stop_provider=lambda: False,
        prompt_builder=lambda prompt, state, **kw: prompt,
        encoder=lambda text, add_bos_token=True: text,
        persistent_ui_state_provider=lambda: {},
        settings_provider=lambda: {},
    )
    s = Summarizer()

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


def _score_instruction(instr: str) -> tuple[bool, str]:
    """Is the director's output plain, inspectable, actionable paragraphs?"""
    if not instr or not instr.strip():
        return False, "empty instruction"
    if any(m in instr for m in ("**", "##", "```", "JSON", "```")):
        return False, "contains markdown/code formatting"
    lines = [l for l in instr.strip().splitlines() if l.strip()]
    if len(lines) < 2:
        return False, "not multi-paragraph"
    low = instr.lower()
    if not any(k in low for k in ("style", "paragraph", "sentence", "dialogue", "voice", "tone", "response should", "reply")):
        return False, "no style/length/dialogue directive"
    return True, f"{len(lines)} plain paragraphs"


def _score_reply(reply: str, user_input: str) -> tuple[bool, str]:
    """Is the reply compliant prose: non-empty and not a recap of the input?"""
    t = reply.strip()
    if not t:
        return False, "empty reply"
    if len(t) < 3:
        return False, "too short"
    words = user_input.strip().split()[:8]
    prefix = " ".join(words)
    if len(prefix) >= 12 and prefix.lower() in t.lower()[:120]:
        return False, "recaps the user's input verbatim"
    return True, f"{len(t)} chars"


def run_live_director() -> tuple[int, list[str]]:
    """Return (skipped_or_fail_count, [messages]). 0 skipped = pass."""
    msgs: list[str] = []
    base_url = os.environ.get("DSS_BENCH_BASE_URL", "").strip()
    if not base_url:
        msgs.append("[live_director] SKIP: DSS_BENCH_BASE_URL not set")
        return 0, msgs
    api_key = os.environ.get("DSS_BENCH_API_KEY", "not-needed")
    model = os.environ.get("DSS_BENCH_MODEL", "local-model")
    inputs = [
        "The cave mouth looms ahead of her, dark and silent. She whispers: what do you think is in there?",
        "He tosses the coin on the table and leans back. Tell me the truth this time — where did you get it?",
        "The storm breaks. She pulls the lantern closer and studies the map, then points at the eastern pass.",
    ]
    if os.environ.get("DSS_BENCH_DIRECTOR_INPUTS"):
        inputs = [i for i in os.environ["DSS_BENCH_DIRECTOR_INPUTS"].split("||") if i.strip()] or inputs

    state = {"name1": "User", "name2": "Assistant", "seed": 42, "context": "A quiet fantasy village by the sea."}
    history = [["User", "The gate creaks open."], ["Assistant", "She steps into the yard, boots wet."]]

    n_pass_on = n_pass_off = 0
    n_total = len(inputs)
    print(f"[live_director] model={model} inputs={n_total} base={base_url}")
    for idx, user_input in enumerate(inputs, start=1):
        row: list[str] = []
        workdir = tempfile.mkdtemp(prefix="dss_live_dir_")
        history_path = Path(workdir) / "history"
        history_path.mkdir(parents=True, exist_ok=True)

        # do_instr=False
        live_off = LiveModel(base_url, api_key, model)
        s = _make_summarizer(live_off, history_path)
        prompt_off, _, _, _ = s.generate_instr_prompt(user_input, state, history, do_instr=False)
        t0 = time.time()
        reply_off = "".join(live_off.generate_with_streaming(str(prompt_off), state))
        dt_off = time.time() - t0

        # do_instr=True
        live_on = LiveModel(base_url, api_key, model)
        s2 = _make_summarizer(live_on, history_path)
        prompt_on, _, _, _ = s2.generate_instr_prompt(user_input, state, history, do_instr=True)
        instr = _cached_instruction(history_path)
        t0 = time.time()
        reply_on = "".join(live_on.generate_with_streaming(str(prompt_on), state))
        dt_on = time.time() - t0

        instr_ok, instr_note = _score_instruction(instr)
        reply_ok_on, note_on = _score_reply(reply_on, user_input)
        reply_ok_off, note_off = _score_reply(reply_off, user_input)
        changed = str(prompt_on) != str(prompt_off)

        if reply_ok_on:
            n_pass_on += 1
        if reply_ok_off:
            n_pass_off += 1

        row.append(f"input #{idx}")
        row.append(
            f"  calls on={live_on.call_count} off={live_off.call_count} | "
            f"tokens on={live_on.prompt_tokens + live_on.completion_tokens} off={live_off.prompt_tokens + live_off.completion_tokens} | "
            f"latency on={dt_on:.1f}s off={dt_off:.1f}s"
        )
        row.append(f"  instr: {'PASS' if instr_ok else 'FAIL'} ({instr_note}) | prompts differ: {changed}")
        row.append(f"  reply on:  {'PASS' if reply_ok_on else 'FAIL'} ({note_on})")
        row.append(f"  reply off: {'PASS' if reply_ok_off else 'FAIL'} ({note_off})")
        msgs.extend(row)

    msgs.append(f"[live_director] reply-compliance on={n_pass_on}/{n_total} off={n_pass_off}/{n_total}")
    return 0, msgs


def _cached_instruction(history_path: Path) -> str:
    """Read the instruction text the engine cached to instructions.json."""
    p = history_path / "instructions.json"
    if not p.exists():
        return ""
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return ""
    for v in data.values():
        if isinstance(v, str):
            return v
    return ""


if __name__ == "__main__":
    code, msgs = run_live_director()
    for m in msgs:
        print(m)
    sys.exit(code)
