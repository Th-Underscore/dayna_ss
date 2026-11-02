"""Soak harness: long-horizon conversation retention test.

Drives the engine over dozens of scripted turns (hermetic — no LLM,
no llama_index, no torch), planting details into the per-subject state at
specific turns and then measuring whether those details survive to the end.

Two things are tested:

1. Faithful state application — the engine must apply every scripted
   gate-check / branch-query / field-update across all turns without
   corrupting or dropping earlier state. Any unexpected loss fails the test.

2. Retention behavior — the report shows, per planted fact, when it was
   planted, the last turn it was observed, and whether it survived. Facts the
   fixture marks ``expected_to_survive`` must still be present in the final
   per-subject JSON; facts marked otherwise must be gone (e.g. an overwrite
   turn that deliberately rewrites a field, modeling in-place supersession).

The writing-style dimension is approximated hermetically by planting
distinctive phrasing strings (e.g. a voice/description line) and asserting
they persist across many turns; true style-retention against a live model is
covered by the director on/off + live benchmark tests in the same suite.

Fixture format (tests/fixtures/<name>/scenario.json)::

    {
      "name": "soak_conversation",
      "schema": "user_data/example/subjects_schema.json",
      "format_templates": "user_data/example/format_templates.json",
      "subjects": ["characters"],
      "initial_state": {"characters": {...}},
      "exchanges": [{"user_input": "...", "output": "..."}, ...],   # N turns
      "script": [...],        # turn-aware rules (quiet turns need none;
                              #   a bare default "NO" handles them)
      "facts": [
        {"id": "f1", "needle": "crimson scarf",
         "probe": "entries.John Jones.description",   # dot-path into subject state
         "planted_at": 0, "expected_to_survive": false},
        ...
      ]
    }

Run with ``python tests/run_tests.py`` (soak runs after golden + director).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from harness import FIXTURES_DIR, Scenario


class SoakScenario(Scenario):
    """A long scripted conversation with planted facts and retention probes."""

    def __init__(self, name: str, fixture_dir: Path):
        super().__init__(name, fixture_dir)
        self.facts = self.meta.get("facts", [])

    # ------------------------------------------------------------ probing

    @staticmethod
    def _resolve(path: str, state: dict):
        node = state
        for part in path.split("."):
            if isinstance(node, dict):
                node = node.get(part)
            elif isinstance(node, list) and part.isdigit():
                idx = int(part)
                node = node[idx] if idx < len(node) else None
            else:
                return None
            if node is None:
                return None
        return node

    def _needle_present(self, subject: str, probe: str, needle: str, state_dir: Path) -> bool:
        path = state_dir / f"{subject}.json"
        if not path.exists():
            return False
        state = json.loads(path.read_text(encoding="utf-8"))
        value = self._resolve(probe, state)
        if value is None:
            return False
        return needle in json.dumps(value, ensure_ascii=False)

    # ---------------------------------------------------------------- run

    def run(self) -> tuple[dict, Any, Path]:
        """Run the soak conversation; return (report, model, out_dir)."""
        out_dir, model = super().run()

        # Presence of each fact per turn (after each exchange).
        turns = len(self.meta["exchanges"])
        subjects = self.meta.get("subjects", [])
        presence = {f["id"]: [False] * turns for f in self.facts}
        for turn_idx in range(turns):
            turn_dir = out_dir / f"after_turn_{turn_idx}"
            for fact in self.facts:
                subject = fact.get("subject", subjects[0] if subjects else "characters")
                presence[fact["id"]][turn_idx] = self._needle_present(
                    subject, fact["probe"], fact["needle"], turn_dir
                )

        # Build report.
        rows = []
        for fact in self.facts:
            fid, needle = fact["id"], fact["needle"]
            planted = fact.get("planted_at", 0)
            seq = presence[fid]
            observed = [i for i, ok in enumerate(seq) if ok]
            last_seen = observed[-1] if observed else None
            survived = bool(observed and last_seen == turns - 1)
            expected = fact.get("expected_to_survive", True)
            loss_turn = None
            if observed:
                for i in range(planted + 1, turns):
                    if not seq[i]:
                        loss_turn = i
                        break
            rows.append({
                "id": fid,
                "needle": needle,
                "planted_at": planted,
                "last_seen": last_seen,
                "survived": survived,
                "expected_to_survive": expected,
                "loss_turn": loss_turn,
                "ok": survived == expected,
            })

        retained = sum(1 for r in rows if r["survived"])
        report = {
            "turns": turns,
            "facts_planted": len(rows),
            "facts_retained": retained,
            "rows": rows,
            "passed": all(r["ok"] for r in rows),
        }
        return report, model, out_dir


def render_report(report: dict) -> list[str]:
    lines = []
    lines.append(f"Soak: {report['turns']} turns, {report['facts_planted']} facts, "
                 f"{report['facts_retained']} retained ({report['facts_retained'] / max(1, report['facts_planted']) * 100:.0f}%)")
    for r in report["rows"]:
        status = "RETAINED" if r["survived"] else "LOST"
        flag = "PASS" if r["ok"] else "FAIL"
        loss = f", lost turn {r['loss_turn']}" if r["loss_turn"] is not None else ""
        last = f"last_seen {r['last_seen']}" if r["last_seen"] is not None else "never seen"
        lines.append(f"  [{flag}] {status:8s} {r['id']} {r['needle']!r} planted@{r['planted_at']} {last}{loss}"
                     f" (expected {'survive' if r['expected_to_survive'] else 'lost'})")
    return lines


def run_soak(name: str = "soak_conversation") -> tuple[bool, list[str], Path | None]:
    """Run the soak; return (passed, full result lines, per-turn out_dir).

    The ``out_dir`` (third element) is the ephemeral per-turn state tree
    (``after_turn_*`` + result.json) — preserved by the suite dump so the full
    per-turn state is analyzable, not just the summary. ``None`` if the run
    could not start (e.g. fixture missing).
    """
    fixture_dir = FIXTURES_DIR / name
    if not (fixture_dir / "scenario.json").exists():
        return False, [f"soak fixture not found: {fixture_dir}"], None
    scenario = SoakScenario(name, fixture_dir)
    report, model, out_dir = scenario.run()
    lines = render_report(report)
    lines.append(f"[soak:{name}] PASS ({len(model.calls)} LLM calls scripted)"
                 if report["passed"] else f"[soak:{name}] FAIL")
    return report["passed"], lines, out_dir
