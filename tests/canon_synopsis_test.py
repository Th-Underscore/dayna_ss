"""Hermetic tests for Canon Synopsis P0 (docs/plans/canon_synopsis_p0.md).

Covers both halves that only work together:
1. Digest  — regenerated from compact inputs at chapter/arc archival into
   general_info.synopsis; fail-open keeps the previous digest on garbage.
2. Demotion — render-time staleness set (compute_stale_entities): cast
   presence and recent participant credits keep an entity fresh; re-entry
   instantly reverses staleness. Roster-line rendering asserted through
   FormattedData with the real templates.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

TEST_DIR = Path(__file__).parent
REPO_ROOT = TEST_DIR.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from extensions.dayna_ss.utils.schema_parser import SchemaParser  # noqa: E402
from extensions.dayna_ss.agents.data_summarizer import DataSummarizer  # noqa: E402
from extensions.dayna_ss.agents.formatted_data import FormattedData  # noqa: E402
from extensions.dayna_ss.agents.summarizer.context_engine import compute_stale_entities  # noqa: E402
from extensions.dayna_ss.ui.phase_manager import PhaseManager  # noqa: E402

SCHEMA_PATH = REPO_ROOT / "extensions/dayna_ss/user_data/example/schemas/subjects_schema_sceneagg.json"

DIGEST_1 = (
    "Juno aborted the Exchange floor drop after spotting unbadged corporate suits and rerouted "
    "through Graft toward the tower's North Face service entrance. The Handler directed the new "
    "plan while a third presence on comms raised the stakes. Open: the dawn delivery, the buyer's "
    "identity, and the pursuer who followed her into the service throat."
)

CHAPTER_JSON = json.dumps({
    "title": "The Test Episode",
    "starting_scene": 1,
    "ending_scene": 4,
    "scenes": [1, 2, 3, 4],
    "summary": "A complete dramatic movement.",
    "key_changes": [{"description": "Status quo shattered", "scene": "Scene 4"}],
    "status": "concluded",
})


class _ScriptedModel:
    """Pops canned responses in call order; records every prompt."""

    def __init__(self, payloads: list[str]):
        self.last = SimpleNamespace(force_next_chapter=False, force_next_arc=False)
        self.config: dict = {}
        self._payloads = list(payloads)
        self.prompts: list[str] = []

    def generate_using_tgwui(self, prompt="", state=None, history_path=None, **kw):
        self.prompts.append(prompt)
        return self._payloads.pop(0), None


def _make_ds(history_path: Path, all_subjects_data: dict, model) -> DataSummarizer:
    return DataSummarizer(
        summarizer=model,
        exchange=("u", "a"),
        custom_state={"history": {"internal": []}, "name1": "User", "name2": "Assistant"},
        history_path=history_path,
        schema_parser=SchemaParser(SCHEMA_PATH),
        all_subjects_data=all_subjects_data,
        phase_manager=PhaseManager(),
    )


def _seed_events(scenes: int) -> dict:
    return {
        "past": {},
        "events": {},
        "chapters": [],
        "scenes": {
            f"Scene {i}": {"summary": f"Scene {i} happened", "importance": {"score": 50}}
            for i in range(1, scenes + 1)
        },
    }


def _seed_general_info() -> dict:
    return {"main_objective": "Deliver the chip before dawn.", "synopsis": ""}


def run_canon_synopsis_test() -> tuple[bool, list[str]]:
    checks: list[tuple[bool, str]] = []

    # --- Digest written at chapter archival ---
    tmp = Path(tempfile.mkdtemp(prefix="dss_canon_"))
    model = _ScriptedModel(["YES", CHAPTER_JSON, DIGEST_1])
    ds = _make_ds(tmp, {"events": _seed_events(4), "current_scene": {"_chapter_number": 1, "_arc_number": 1},
                        "general_info": _seed_general_info()}, model)
    ds.check_and_archive_chapter()
    gi = json.loads((tmp / "general_info.json").read_text(encoding="utf-8"))
    checks += [
        (gi.get("synopsis") == DIGEST_1, "digest persisted to general_info.json on disk"),
        ((tmp / "canon_history.json").exists(), "canon_history.json appended"),
        (json.loads((tmp / "canon_history.json").read_text(encoding="utf-8"))[0]["trigger"] == "chapter_1",
         "canon_history entry carries the trigger"),
        ("The Test Episode" in model.prompts[-1], "digest prompt includes the archived chapter"),
        ("Deliver the chip before dawn." in model.prompts[-1], "digest prompt anchors on main_objective"),
    ]

    # --- Fail-open: garbage response keeps the previous digest ---
    tmp2 = Path(tempfile.mkdtemp(prefix="dss_canon_"))
    gi_seed = _seed_general_info()
    gi_seed["synopsis"] = "Previous good digest."
    (tmp2 / "general_info.json").write_text(json.dumps(gi_seed), encoding="utf-8")
    model2 = _ScriptedModel(["YES", CHAPTER_JSON, '{"not": "prose"}'])
    ds2 = _make_ds(tmp2, {"events": _seed_events(4), "current_scene": {"_chapter_number": 1, "_arc_number": 1},
                          "general_info": gi_seed}, model2)
    ds2.check_and_archive_chapter()
    gi2 = json.loads((tmp2 / "general_info.json").read_text(encoding="utf-8"))
    checks += [
        (gi2.get("synopsis") == "Previous good digest.", "garbage digest response leaves previous synopsis intact"),
    ]

    # --- Arc archival also regenerates (arcs read from arcs.json) ---
    tmp3 = Path(tempfile.mkdtemp(prefix="dss_canon_"))
    second_chapter = json.loads(CHAPTER_JSON)
    second_chapter.update({"title": "Second Movement", "ending_scene": 6})
    events3 = _seed_events(6)
    events3["chapters"] = [json.loads(CHAPTER_JSON), second_chapter]
    (tmp3 / "arcs.json").write_text(json.dumps({"Season One": {
        "title": "Season One", "summary": "The long-range conflict resolved."}}), encoding="utf-8")
    model3 = _ScriptedModel(["YES", json.dumps({
        "title": "Season One", "summary": "The long-range conflict resolved."}), DIGEST_1])
    ds3 = _make_ds(tmp3, {"events": events3, "current_scene": {"_chapter_number": 3, "_arc_number": 1},
                          "general_info": _seed_general_info()}, model3)
    ds3.check_and_archive_arc()
    gi3 = json.loads((tmp3 / "general_info.json").read_text(encoding="utf-8"))
    checks += [
        (gi3.get("synopsis") == DIGEST_1, "arc archival regenerates the digest"),
        ("Season One" in model3.prompts[-1], "arc digest prompt includes arcs from arcs.json"),
    ]

    # --- Staleness math ---
    scenes = {
        f"S{i}": {"participants": {f"Char{i}": {}, }}
        for i in range(1, 9)  # S1..S8
    }
    stale = compute_stale_entities(scenes, {}, window=6)
    checks += [
        ({"Char1", "Char2"}.issubset(stale)
         and not {"Char3", "Char4", "Char5", "Char6", "Char7", "Char8"} & stale,
         "window: only participants older than the last 6 scenes go stale"),
    ]
    # Cast presence overrides recency even for an ancient participant.
    fresh_cast = compute_stale_entities(scenes, {"characters": [{"name": "Char1"}], "groups": [], "elements": []}, window=6)
    checks.append(("Char1" not in fresh_cast, "cast credit in now.who keeps entity fresh"))
    # Re-entry reversal: add Char1 as participant of the newest scene -> fresh.
    scenes_rev = json.loads(json.dumps(scenes))
    scenes_rev["S8"]["participants"]["Char1"] = {}
    reversed_set = compute_stale_entities(scenes_rev, {}, window=6)
    checks.append(("Char1" not in reversed_set, "re-entry into the newest scene reverses staleness"))
    # List-shaped participants are honored too (scene credited a list of names).
    scenes_list = {
        "Old": {"participants": [{"name": "Listy"}, {"name": "Juno"}]},
        "Newer": {"participants": {"Juno": {}, "DictKey": {}}},
        "Newest": {"participants": [{"name": "Juno"}]},
    }
    stale_list = compute_stale_entities(scenes_list, {"characters": [{"name": "Juno"}]}, window=2)
    checks.append(("Listy" in stale_list and "DictKey" not in stale_list and "Juno" not in stale_list,
                   "list- and dict-shaped participants handled across the window"))

    # --- Demoted roster render through the real templates ---
    parser = SchemaParser(SCHEMA_PATH)
    chars = {"entries": {
        "Juno": {"description": {"main": "Courier."}, "importance": {"score": 90}, "biography": "",
                 "traits": [], "voice": "", "quirks": [], "fears": [], "relationships": {},
                 "_recent_state": "active"},
        "Old Contact": {"description": ["A fixer from a past life."], "importance": {"score": 75},
                        "biography": "", "traits": [], "voice": "", "quirks": [], "fears": [],
                        "relationships": {}, "_recent_state": ""},
    }}
    fd = FormattedData(chars, "characters", parser=parser,
                       extra_context={"_stale_entities": ["Old Contact"]})
    out = fd.st
    checks += [
        ("Old Contact [past chapters]" in out, "stale entity collapses to a roster line"),
        ("Description -- A fixer" not in out, "stale entity no longer renders its full profile"),
        ("Character --- Juno" in out and "Courier." in out, "fresh entity keeps its full profile"),
    ]

    msgs = [f"{'PASS' if ok else 'FAIL'}  {msg}" for ok, msg in checks]
    return all(ok for ok, _ in checks), msgs


if __name__ == "__main__":
    ok, messages = run_canon_synopsis_test()
    print("\n".join(messages))
    print("[canon_synopsis] PASS" if ok else "[canon_synopsis] FAIL")
    sys.exit(0 if ok else 1)
