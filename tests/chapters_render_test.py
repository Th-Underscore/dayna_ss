"""Regression: chapters/arcs must render as full blocks, not key-scalar mush.

Live smoke (2026-08-25) rendered an archived chapter as "Chapter [1] --- 0":
FormattedData.__init__ ran expand_lists_in_data_for_llm with schema_type=None
(no hint branch for 'chapters'), which dict-expanded [chapter_dict] into
{"0": chapter_dict}; the list-iterating template then saw stringified keys.
Untyped object lists now stay lists.
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from extensions.dayna_ss.agents.formatted_data import FormattedData  # noqa: E402
from extensions.dayna_ss.utils.schema_parser import SchemaParser  # noqa: E402

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "user_data/example/schemas/subjects_schema_sceneagg.json"


CHAPTER = {
    "title": "Route Adjustment",
    "starting_scene": 1,
    "ending_scene": 3,
    "scenes": [1, 2, 3],
    "summary": "Juno reroutes the drop after spotting corporate suits.",
    "key_changes": [{"description": "Drop aborted", "scene": "Scene 2"}],
    "status": "active",
}


def run_chapters_render_test() -> tuple[bool, list[str]]:
    checks: list[tuple[bool, str]] = []

    # Real parser: exercises the alias-hint path (chapters -> Chapters).
    parser = SchemaParser(SCHEMA_PATH)
    fd = FormattedData([CHAPTER], "chapters", parser=parser)
    checks.append((isinstance(fd.data, list), f"alias-hinted chapters stay a list (got {type(fd.data).__name__})"))
    out = fd.st
    checks += [
        ("Route Adjustment" in out, "chapter title rendered"),
        ("Summary ---" in out, "chapter summary rendered"),
        ("Scenes --- 1, 2, 3" in out, "chapter scene span rendered"),
        (not out.strip().startswith("Chapter [1] --- 0"), "no key-scalar mush line"),
    ]

    # Scalar lists still expand for LLM readability (existing behavior kept).
    class _NoHintParser:
        def get_subject_class(self, name):
            return None

        definitions: dict = {}

        @property
        def defaults(self):
            return {}

    fd2 = FormattedData(["alpha", "beta"], "generic_tags", parser=_NoHintParser())
    checks.append((isinstance(fd2.data, dict), "scalar lists still expand to indexed dicts"))

    msgs = [f"{'PASS' if ok else 'FAIL'}  {msg}" for ok, msg in checks]
    return all(ok for ok, _ in checks), msgs


if __name__ == "__main__":
    ok, messages = run_chapters_render_test()
    print("\n".join(messages))
    print("[chapters_render] PASS" if ok else "[chapters_render] FAIL")
    sys.exit(0 if ok else 1)
