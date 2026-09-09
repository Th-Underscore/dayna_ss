"""Hermetic regression test: boundary-check mutations must reach DISK.

Found live (aggregation-fix validation smoke, cyberpunk_thriller seed 7,
2026-08-25): check_and_archive_chapter logged "Archiving chapter 1" at turn 2,
but every events.json on disk still had chapters=[] — per-subject files are
saved inside DataSummarizer.generate BEFORE the post-loop boundary checks run,
so the chapter append and the _chapter_number stamp existed only in memory and
were lost when the next turn loaded from disk. The arc path saved arcs.json but
not the _arc_number stamp on current_scene.

Drives check_and_archive_chapter / check_and_archive_arc directly against a
temp history dir with a scripted gate/generation model, then asserts the ON-DISK
files — not the in-memory dicts — carry the archive.
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
from extensions.dayna_ss.ui.phase_manager import PhaseManager  # noqa: E402

SCHEMA_PATH = REPO_ROOT / "extensions/dayna_ss/user_data/example/schemas/subjects_schema_sceneagg.json"


class _ScriptedBoundaryModel:
    """Answers the chapter/arc gate ("YES") then hands back canned JSON payloads."""

    def __init__(self, payloads: list[str]):
        self.last = SimpleNamespace(force_next_chapter=False, force_next_arc=False)
        self.config: dict = {}
        self._payloads = list(payloads)
        self.calls: list[str] = []

    def generate_using_tgwui(self, prompt="", state=None, history_path=None, **kw):
        self.calls.append(prompt[:60])
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


CHAPTER_JSON = json.dumps({
    "title": "The Test Episode",
    "starting_scene": 1,
    "ending_scene": 4,
    "scenes": [1, 2, 3, 4],
    "summary": "A complete dramatic movement.",
    "key_changes": [{"description": "Status quo shattered", "scene": "Scene 4"}],
    "status": "concluded",
})

ARC_JSON = json.dumps({
    "title": "Season One",
    "starting_chapter": 1,
    "ending_chapter": 2,
    "chapters": [1, 2],
    "summary": "The long-range conflict resolved.",
})


def run_boundary_persistence_test() -> tuple[bool, list[str]]:
    msgs: list[str] = []
    checks: list[tuple[bool, str]] = []

    # --- Chapter archival persists events.json + current_scene.json ---
    tmp = Path(tempfile.mkdtemp(prefix="dss_boundpersist_"))
    ds = _make_ds(tmp, {"events": _seed_events(4), "current_scene": {"_chapter_number": 1, "_arc_number": 1}},
                  _ScriptedBoundaryModel(["YES", CHAPTER_JSON]))
    ds.check_and_archive_chapter()

    ev_disk = json.loads((tmp / "events.json").read_text(encoding="utf-8"))
    cs_disk = json.loads((tmp / "current_scene.json").read_text(encoding="utf-8"))
    ch_list = ev_disk.get("chapters")
    checks += [
        (isinstance(ch_list, list) and len(ch_list) == 1, "events.json on disk holds the archived chapter"),
        (isinstance(ch_list, list) and ch_list
         and ch_list[0].get("ending_scene") == 4 and ch_list[0].get("title") == "The Test Episode",
         "archived chapter span data intact on disk"),
        (cs_disk.get("_chapter_number") == 2, "current_scene.json on disk carries the bumped chapter number"),
    ]

    # --- Arc archival persists the _arc_number stamp (arcs.json already did) ---
    tmp2 = Path(tempfile.mkdtemp(prefix="dss_boundpersist_"))
    second_chapter = json.loads(CHAPTER_JSON)
    second_chapter.update({"title": "Chapter 2", "ending_scene": 8, "scenes": [5, 6, 7, 8]})
    subjects = {
        "events": {
            "past": {}, "events": {},
            # Two well-formed archived chapters: the arc gate skips below its
            # suggested_min_chapters (2 in both shipped schemas), and malformed
            # entries are dropped by the sanitizer rather than counted.
            "chapters": [json.loads(CHAPTER_JSON), second_chapter],
            "scenes": {},
        },
        "current_scene": {"_chapter_number": 3, "_arc_number": 1},
    }
    (tmp2 / "arcs.json").write_text("{}", encoding="utf-8")
    ds2 = _make_ds(tmp2, subjects, _ScriptedBoundaryModel(["YES", ARC_JSON]))
    ds2.check_and_archive_arc()

    cs2 = json.loads((tmp2 / "current_scene.json").read_text(encoding="utf-8"))
    arcs_disk = json.loads((tmp2 / "arcs.json").read_text(encoding="utf-8"))
    checks += [
        (len(arcs_disk) == 1, "arcs.json on disk holds the archived arc"),
        (cs2.get("_arc_number") == 2, "current_scene.json on disk carries the bumped arc number"),
        (list(arcs_disk.values())[0].get("chapters") == [1, 2]
         and list(arcs_disk.values())[0].get("ending_chapter") == 2,
         "arc chapter spans are deterministic (model values overridden)"),
    ]

    # --- Corrupted chapters array is sanitized, not propagated ---
    # Live-found shape: a mis-heal unwrapped legal [chapter_dict] to a bare
    # dict; _entries_as_list then flattened it via dict.values() into
    # positional scalars, and archival appended onto the monolith.
    tmp3 = Path(tempfile.mkdtemp(prefix="dss_boundpersist_"))
    survivor = {"title": "The Ascent and the Beacon", "starting_scene": 1,
                "ending_scene": 2, "scenes": [1, 2], "status": "concluded"}
    corrupted = ["Shadow Side Run", 1, 2, [1, 2], "A summary.",
                 [{"description": "x", "scene": "Scene 2"}], "concluded",
                 survivor]
    ds3 = _make_ds(tmp3, {"events": {**_seed_events(6), "chapters": corrupted},
                          "current_scene": {"_chapter_number": 2, "_arc_number": 1}},
                   _ScriptedBoundaryModel(["YES", CHAPTER_JSON]))
    ds3.check_and_archive_chapter()
    ev3 = json.loads((tmp3 / "events.json").read_text(encoding="utf-8"))
    ch3 = ev3.get("chapters")
    checks += [
        (all(isinstance(c, dict) for c in ch3),
         f"sanitized chapters list holds only dicts ({[type(c).__name__ for c in ch3]})"),
        (len(ch3) == 2 and {c.get('title') for c in ch3} == {"The Ascent and the Beacon", "The Test Episode"},
         "surviving well-formed chapters kept + new archive appended"),
        (ch3[-1].get("ending_scene") == 6 and ch3[-1].get("scenes") == [3, 4, 5, 6],
         f"new archive spans scenes since survivor's boundary (got {ch3[-1].get('scenes')})"),
    ]

    # --- Model-hallucinated spans are overridden with computed ones ---
    bogus = json.loads(CHAPTER_JSON)
    bogus.update({"starting_scene": 12, "ending_scene": 14, "scenes": [12, 13, 14]})
    tmp4 = Path(tempfile.mkdtemp(prefix="dss_boundpersist_"))
    ds4 = _make_ds(tmp4, {"events": _seed_events(5), "current_scene": {"_chapter_number": 1, "_arc_number": 1}},
                   _ScriptedBoundaryModel(["YES", json.dumps(bogus)]))
    ds4.check_and_archive_chapter()
    ev4 = json.loads((tmp4 / "events.json").read_text(encoding="utf-8"))
    c4 = ev4["chapters"][0]
    checks += [
        (c4.get("starting_scene") == 1 and c4.get("ending_scene") == 5 and c4.get("scenes") == [1, 2, 3, 4, 5],
         f"hallucinated span replaced by computed span (got {c4.get('scenes')})"),
    ]

    ok = all(passed for passed, _ in checks)
    for passed, name in checks:
        msgs.append(("PASS  " if passed else "FAIL  ") + name)
    if ok:
        msgs.append("[boundary_persistence] PASS")
    return ok, msgs


if __name__ == "__main__":
    ok, msgs = run_boundary_persistence_test()
    for m in msgs:
        print(m)
    sys.exit(0 if ok else 1)
