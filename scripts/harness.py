"""Golden-fixture test harness for the dayna_ss summarization engine.

Drives the engine hermetically (no LLM, no llama_index, no torch, no gradio)
by substituting a scripted model for the LLM calls. The scripted model answers
gate checks, branch queries, and branch updates from a declarative rule table
in the fixture, so the schema-state updates are fully deterministic.

Layout:

    tests/fixtures/<scenario>/
        scenario.json          -- exchanges, subjects, initial state, script, schema refs
    tests/golden/<scenario>/
        after_turn_<n>/<subject>.json   -- expected per-subject state after each exchange

Run with:

    python tests/run_tests.py                # run all scenarios, compare vs golden
    python tests/run_tests.py -s <name>      # run one scenario
    python tests/run_tests.py --update-golden  # regenerate golden files
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

TEST_DIR = Path(__file__).parent
FIXTURES_DIR = TEST_DIR / "fixtures"
GOLDEN_DIR = TEST_DIR / "golden"

# Import the engine package (relies on llama_index/torch being lazily imported).
REPO_ROOT = TEST_DIR.parent.parent.parent  # <repo>/extensions/dayna_ss/tests -> <repo>
sys.path.insert(0, str(REPO_ROOT))

from extensions.dayna_ss.utils.schema_parser import SchemaParser  # noqa: E402
from extensions.dayna_ss.agents.data_summarizer import DataSummarizer  # noqa: E402
from extensions.dayna_ss.ui.phase_manager import PhaseManager  # noqa: E402


class ScriptedModel:
    """Fake summarizer answering generate_with_sse / generate_using_tgwui.

    The `script` is a list of rules::

        {"when_prompt_contains": [..substrings..], "respond": "text"}

    The first rule whose substrings all appear in the prompt wins. A rule with
    no `when_prompt_contains` acts as the default. Emulates the real generator's
    stopping-string semantics: if the response text starts with a stopping
    string (prefix match), that string becomes the stop_reason.
    """

    def __init__(self, parser: SchemaParser, state: dict, history_path: Path, script: list[dict]):
        self.script = script or []
        self.calls: list[dict] = []
        self.turn: int = 0
        self.last = SimpleNamespace(
            schema_parser=parser,
            state=state,
            is_new_scene_turn=False,
            force_next_chapter=False,
            force_next_arc=False,
            history_path=history_path,
        )

    def _respond(self, prompt: str) -> str:
        for rule in self.script:
            if "turn" in rule and rule["turn"] != self.turn:
                continue
            markers = rule.get("when_prompt_contains")
            if markers is None:
                return rule.get("respond", "NO")
            if all(m in prompt for m in markers):
                return rule.get("respond", "NO")
        return "NO"

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
        text = self._respond(prompt)
        self.calls.append({"kind": "generate_with_sse", "step_id": step_id, "prompt": prompt, "text": text})
        return self._stop(text, stopping_strings, match_prefix_only)

    def generate_using_tgwui(self, prompt: str, state: dict | None = None,
                             history_path: str | Path | None = None,
                             stopping_strings: list[str] | None = None,
                             match_prefix_only: bool = True, **kwargs) -> tuple[str, str]:
        text = self._respond(prompt)
        self.calls.append({"kind": "generate_using_tgwui", "prompt": prompt, "text": text})
        return self._stop(text, stopping_strings, match_prefix_only)

    def format_dialogue(self, state: dict | None, history: list) -> str:
        return "\n".join(f"{u}: {o}" for u, o in (history or []))


class Scenario:
    """A hermetic scenario: a scripted transcript against a schema pack."""

    def __init__(self, name: str, fixture_dir: Path):
        self.name = name
        self.fixture_dir = fixture_dir
        self.meta = json.loads((fixture_dir / "scenario.json").read_text(encoding="utf-8"))

    @property
    def golden_dir(self) -> Path:
        return GOLDEN_DIR / self.name

    def resolve_ref(self, ref: str) -> Path:
        """Resolve a file ref relative to the extension root or the fixture dir."""
        root = TEST_DIR.parent
        candidate = (self.fixture_dir / ref) if not (root / ref).exists() else (root / ref)
        if not candidate.exists():
            raise FileNotFoundError(f"scenario '{self.name}': cannot resolve ref '{ref}'")
        return candidate

    # ---------------------------------------------------------------- run

    def run(self) -> tuple[Path, ScriptedModel]:
        """Execute the scenario in a temp workspace; return (out_dir, model)."""
        schema_path = self.resolve_ref(self.meta.get("schema", "user_data/example/subjects_schema.json"))
        templates_path = self.resolve_ref(self.meta.get("format_templates", "user_data/example/format_templates.json"))

        out_dir = tempfile.mkdtemp(prefix=f"dss_{self.name}_")
        history_path = Path(out_dir) / "history"
        history_path.mkdir(parents=True, exist_ok=True)
        shutil.copy2(schema_path, history_path / "subjects_schema.json")
        shutil.copy2(templates_path, history_path / "format_templates.json")

        # Initial per-subject state
        initial = self.meta.get("initial_state", {})
        for subject, data in initial.items():
            (history_path / f"{subject}.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

        schema_parser = SchemaParser(schema_path)
        phase_manager = PhaseManager()
        model = ScriptedModel(schema_parser, self._custom_state(), history_path, self.meta.get("script"))

        all_subjects_data = dict(initial)
        exchanges = self.meta["exchanges"]

        for turn_idx, exchange in enumerate(exchanges):
            model.turn = turn_idx
            user_input, output = exchange["user_input"], exchange["output"]
            summarizer = DataSummarizer(
                summarizer=model,
                exchange=(user_input, output),
                custom_state=self._custom_state(),
                history_path=history_path,
                schema_parser=schema_parser,
                all_subjects_data=all_subjects_data,
                phase_manager=phase_manager,
            )
            for subject in self.meta.get("subjects", list(all_subjects_data.keys())):
                data = all_subjects_data.get(subject)
                if data is None:
                    continue
                schema_class = schema_parser.get_subject_class(subject)
                all_subjects_data[subject] = summarizer.generate(subject, data, schema_class)

            # Snapshot state after this turn (canonical form as persisted by the engine)
            turn_dir = Path(out_dir) / f"after_turn_{turn_idx}"
            turn_dir.mkdir(parents=True, exist_ok=True)
            for subject in self.meta.get("subjects", list(all_subjects_data.keys())):
                src = history_path / f"{subject}.json"
                if src.exists():
                    shutil.copy2(src, turn_dir / f"{subject}.json")

        return Path(out_dir), model

    def _custom_state(self) -> dict:
        default = {"history": {"internal": []}, "name1": "User", "name2": "Assistant"}
        state = dict(self.meta.get("custom_state", {}))
        for k, v in default.items():
            state.setdefault(k, v)
        return state

    # ---------------------------------------------------------------- golden

    def compare(self, out_dir: Path) -> tuple[list[str], list[str]]:
        """Return (missing, mismatches) relative to golden."""
        if not self.golden_dir.exists():
            return [f"no golden dir: {self.golden_dir}"], []
        mismatches, missing = [], []
        for turn_dir in sorted(self.golden_dir.glob("after_turn_*")):
            rel = turn_dir.relative_to(self.golden_dir)
            out_turn = out_dir / rel
            if not out_turn.exists():
                missing.append(f"{rel}: no output turn dir")
                continue
            for gold_file in sorted(turn_dir.glob("*.json")):
                name = gold_file.name
                out_file = out_turn / name
                if not out_file.exists():
                    missing.append(f"{rel}/{name}: no output file")
                    continue
                got = json.loads(out_file.read_text(encoding="utf-8"))
                want = json.loads(gold_file.read_text(encoding="utf-8"))
                if got != want:
                    mismatches.append(f"{rel}/{name}")
        return missing, mismatches

    def write_golden(self, out_dir: Path) -> None:
        if self.golden_dir.exists():
            shutil.rmtree(self.golden_dir)
        self.golden_dir.mkdir(parents=True, exist_ok=True)
        for turn_dir in sorted(out_dir.glob("after_turn_*")):
            shutil.copytree(turn_dir, self.golden_dir / turn_dir.name)
