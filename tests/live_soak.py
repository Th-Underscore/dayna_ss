"""Live soak harness: detail retention against a real model.

Extends the hermetic ``soak_conversation`` concept to a live backend. The
scripted model is replaced with a ``LiveSoakModel`` that answers the engine's
gate-check / branch-query / field-update prompts through an OpenAI-compatible
endpoint, and the planted needles are injected into the *conversation text* at
their planted turn (the model must decide to write them into the structured
per-subject state and keep them there).

This measures the actual retention behavior the hermetic soak can only
approximate: whether a real (typically small, local) model encodes narrative
details into the schema state and preserves them over many turns.

The report is informational (like the JSON-compliance benchmark) — live models
are non-deterministic, so retention rates are reported rather than asserted.
Skipped cleanly when ``DSS_BENCH_BASE_URL`` is unset.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

TEST_DIR = Path(__file__).parent
REPO_ROOT = TEST_DIR.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from extensions.dayna_ss.agents.data_summarizer import DataSummarizer  # noqa: E402
from extensions.dayna_ss.ui.phase_manager import PhaseManager  # noqa: E402
from extensions.dayna_ss.utils.schema_parser import SchemaParser  # noqa: E402
from harness import FIXTURES_DIR  # noqa: E402
from soak_test import SoakScenario  # noqa: E402


def _thinking_enabled() -> bool:
    return os.environ.get("DSS_BENCH_THINKING", "0") in ("1", "true", "True")


class LiveSoakModel:
    """ScriptedModel-compatible live backend: answers engine prompts via the API."""

    def __init__(self, base_url: str, api_key: str, model: str, max_tokens: int = 400):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.calls: list[dict] = []
        self.turn: int = 0
        self.last: Any = None
        self.exchanges: list[tuple[str, str]] = []

    def _complete(self, prompt: str) -> str:
        # Mirror production: the engine's chat prompt wraps the raw prompt with
        # the recent dialogue (runtime.generate_chat_prompt + state history).
        # Without this the "latest exchange" the templates reference is absent.
        history_str = "\n".join(f"{u}: {o}" for u, o in self.exchanges[-6:])
        full_prompt = f"{history_str}\n\n{prompt}" if history_str else prompt
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": full_prompt}],
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
        }
        if not _thinking_enabled():
            payload["enable_thinking"] = False
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"].get("content", "") or ""

    @staticmethod
    def _stop(text: str, stopping_strings: list[str] | None, match_prefix_only: bool) -> tuple[str, str]:
        stopping_strings = stopping_strings or []
        if match_prefix_only:
            for ss in stopping_strings:
                if text.lstrip().startswith(ss):
                    return text, ss
        else:
            for ss in stopping_strings:
                if ss in text:
                    return text, ss
        return text, ""

    def generate_with_sse(self, prompt: str, state: dict | None = None, phase_id: str = "",
                          step_id: str = "", history_path: str | Path | None = None,
                          stopping_strings: list[str] | None = None,
                          match_prefix_only: bool = True, **kwargs) -> tuple[str, str]:
        text = self._complete(prompt)
        self.calls.append({"kind": "generate_with_sse", "step_id": step_id, "prompt": prompt, "text": text})
        return self._stop(text, stopping_strings, match_prefix_only)

    def generate_using_tgwui(self, prompt: str, state: dict | None = None,
                             history_path: str | Path | None = None,
                             stopping_strings: list[str] | None = None,
                             match_prefix_only: bool = True, **kwargs) -> tuple[str, str]:
        text = self._complete(prompt)
        self.calls.append({"kind": "generate_using_tgwui", "prompt": prompt, "text": text})
        return self._stop(text, stopping_strings, match_prefix_only)

    def format_dialogue(self, state: dict | None, history: list) -> str:
        return "\n".join(f"{u}: {o}" for u, o in (history or []))


def _inject_needles(exchanges: list[dict], facts: list[dict]) -> list[dict]:
    """Append each planted needle to the user_input at its planted turn, so the
    live model sees the detail in conversation context."""
    out = [dict(e) for e in exchanges]
    for fact in facts:
        idx = min(fact.get("planted_at", 0), len(out) - 1)
        out[idx]["user_input"] = f"{out[idx]['user_input']} ({fact['needle']})"
    return out


def _find_needle(needle: str, node, path: str = "") -> list[str]:
    """Return all dot-paths where the needle appears (case-insensitive)."""
    hits = []
    if isinstance(node, dict):
        for k, v in node.items():
            hits += _find_needle(needle, v, f"{path}.{k}" if path else k)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            hits += _find_needle(needle, v, f"{path}[{i}]")
    elif isinstance(node, str) and needle.lower() in node.lower():
        hits.append(path)
    return hits


def run_live_soak() -> tuple[int, list[str]]:
    """Return (skip_or_fail, [messages]). Skip (0) when no base URL; informational otherwise."""
    msgs: list[str] = []
    base_url = os.environ.get("DSS_BENCH_BASE_URL", "").strip()
    if not base_url:
        msgs.append("[live_soak] SKIP: DSS_BENCH_BASE_URL not set")
        return 0, msgs
    api_key = os.environ.get("DSS_BENCH_API_KEY", "not-needed")
    model = os.environ.get("DSS_BENCH_MODEL", "local-model")
    name = os.environ.get("DSS_BENCH_SOAK_FIXTURE", "soak_conversation")
    fixture_dir = FIXTURES_DIR / name
    if not (fixture_dir / "scenario.json").exists():
        msgs.append(f"[live_soak] fixture not found: {fixture_dir}")
        return 0, msgs

    scenario = SoakScenario(name, fixture_dir)
    meta = scenario.meta
    schema_path = scenario.resolve_ref(meta.get("schema", "user_data/example/subjects_schema.json"))
    templates_path = scenario.resolve_ref(meta.get("format_templates", "user_data/example/format_templates.json"))

    out_dir = Path(tempfile.mkdtemp(prefix="dss_live_soak_"))
    history_path = out_dir / "history"
    history_path.mkdir(parents=True, exist_ok=True)
    shutil.copy2(schema_path, history_path / "subjects_schema.json")
    shutil.copy2(templates_path, history_path / "format_templates.json")
    for subject, data in meta.get("initial_state", {}).items():
        (history_path / f"{subject}.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

    schema_parser = SchemaParser(schema_path)
    phase_manager = PhaseManager()
    model_obj = LiveSoakModel(base_url, api_key, model)

    custom_state = dict(meta.get("custom_state", {}))
    custom_state.setdefault("history", {"internal": []})
    custom_state.setdefault("name1", "User")
    custom_state.setdefault("name2", "Assistant")
    model_obj.last = type(
        "L",
        (),
        {
            "history_path": history_path,
            "schema_parser": schema_parser,
            "state": custom_state,
            "is_new_scene_turn": False,
            "force_next_chapter": False,
            "force_next_arc": False,
        },
    )

    all_subjects_data = dict(meta.get("initial_state", {}))
    exchanges = _inject_needles(meta["exchanges"], meta.get("facts", []))
    subjects = meta.get("subjects", list(all_subjects_data.keys()))
    turns = len(exchanges)

    print(f"[live_soak] model={model} fixture={name} turns={turns} base={base_url}")
    for turn_idx, exchange in enumerate(exchanges):
        model_obj.turn = turn_idx
        model_obj.exchanges.append((exchange["user_input"], exchange["output"]))
        summarizer = DataSummarizer(
            summarizer=model_obj,
            exchange=(exchange["user_input"], exchange["output"]),
            custom_state=custom_state,
            history_path=history_path,
            schema_parser=schema_parser,
            all_subjects_data=all_subjects_data,
            phase_manager=phase_manager,
        )
        for subject in subjects:
            data = all_subjects_data.get(subject)
            if data is None:
                continue
            schema_class = schema_parser.get_subject_class(subject)
            all_subjects_data[subject] = summarizer.generate(subject, data, schema_class)

    # Probe retention across the whole subject state (the live model decides
    # which fields to write; the hermetic fixture's fixed-path probes do not
    # apply to non-deterministic model output).
    retained = 0
    for fact in meta.get("facts", []):
        subject = fact.get("subject", subjects[0] if subjects else "characters")
        state_path = history_path / f"{subject}.json"
        if not state_path.exists():
            msgs.append(f"  [-] {fact['id']} {fact['needle']!r} no subject state file")
            continue
        state = json.loads(state_path.read_text(encoding="utf-8"))
        hits = _find_needle(fact["needle"], state)
        if hits:
            retained += 1
            msgs.append(f"  [RETAINED] {fact['id']} {fact['needle']!r} -> {hits[0]}")
        else:
            msgs.append(f"  [LOST] {fact['id']} {fact['needle']!r} not found in state")
    rate = retained / max(1, len(meta.get("facts", []))) * 100
    msgs.append(f"[live_soak] {retained}/{len(meta.get('facts', []))} needles retained ({rate:.0f}%), {len(model_obj.calls)} LLM calls")

    return 0, msgs


if __name__ == "__main__":
    code, msgs = run_live_soak()
    for m in msgs:
        print(m)
    sys.exit(code)
