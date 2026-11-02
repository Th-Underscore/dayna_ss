"""Spreadsheet validator for the long-horizon soak.

Validates a story spreadsheet against the note schema defined in
``docs/plans/long_horizon_soak_plan.md`` §5. Checks:

- Top-level structure (id, genre, premise, setting, characters, writing_style,
  specificity_profile, notes).
- Every note has a valid ``type`` and ``specificity``, a non-empty ``content``,
  and a probeable needle where the note type requires one.
- ``plant``/``recall`` turns are sane (0 <= turn <= total_turns, plant before
  recall when both exist) and recall hints are present for recall-bearing notes.
- ``needle_syns`` is a list of non-empty strings.
- Character names are present and distinct.

Exit code 0 = valid; prints human-readable problems otherwise.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

NOTE_TYPES = {
    "character_plant",   # needs: needle
    "character_echo",    # needs: needle + recall
    "plot_twist",        # optional needle; recall window optional
    "location_turning_point",  # needs: needle + recall
    "supersession",      # needs: needle + recall
    "foreshadow",        # needs: needle + recall
    "style_hold",        # recall only; needle intentionally empty
}

SPECIFICITIES = {"loose", "mixed", "exact"}
PROFILES = {"loose", "mixed", "exact"}

REQUIRES_RECALL = {"character_echo", "location_turning_point", "supersession", "foreshadow", "style_hold"}


def validate_spreadsheet(path: Path, total_turns: int = 100) -> list[str]:
    """Return a list of problems; empty means the spreadsheet is valid."""
    problems: list[str] = []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return [f"{path.name}: cannot parse JSON: {e}"]

    name = path.stem
    if data.get("id") != name:
        problems.append(f"{name}: spreadsheet 'id' ({data.get('id')!r}) does not match filename '{name}'")
    for field in ("genre", "premise", "setting"):
        if not data.get(field):
            problems.append(f"{name}: missing required field '{field}'")
    if not data.get("greeting"):
        problems.append(f"{name}: missing required field 'greeting' (name2's opening line, in the dss voice — "
                        f"production parity, chat.py:1801)")

    chars = data.get("characters", {})
    for role in ("name1", "name2"):
        c = chars.get(role) or {}
        if not c.get("name"):
            problems.append(f"{name}: characters.{role}.name required")
    if chars.get("name1", {}).get("name") == chars.get("name2", {}).get("name"):
        problems.append(f"{name}: name1 and name2 must be distinct")

    ws = data.get("writing_style", {})
    if not ws.get("directive"):
        problems.append(f"{name}: writing_style.directive required")
    if not ws.get("guide_directive"):
        problems.append(f"{name}: writing_style.guide_directive required (cloud model's voice for name1)")
    if not ws.get("dss_directive"):
        problems.append(f"{name}: writing_style.dss_directive required (desired DSS output voice for name2)")

    profile = data.get("specificity_profile")
    if profile not in PROFILES:
        problems.append(f"{name}: specificity_profile must be one of {sorted(PROFILES)}, got {profile!r}")

    notes = data.get("notes")
    if not isinstance(notes, list) or not notes:
        problems.append(f"{name}: 'notes' must be a non-empty list")
        return problems

    seen_ids = set()
    for i, note in enumerate(notes):
        tag = f"{name}:notes[{i}]"
        nid = note.get("id")
        if not nid:
            problems.append(f"{tag}: missing 'id'")
        elif nid in seen_ids:
            problems.append(f"{tag}: duplicate note id '{nid}'")
        seen_ids.add(nid)

        ntype = note.get("type")
        if ntype not in NOTE_TYPES:
            problems.append(f"{tag}: unknown type {ntype!r} (expected one of {sorted(NOTE_TYPES)})")

        spec = note.get("specificity")
        if spec not in SPECIFICITIES:
            problems.append(f"{tag}: specificity must be one of {sorted(SPECIFICITIES)}, got {spec!r}")

        if not note.get("content"):
            problems.append(f"{tag}: missing 'content'")

        syns = note.get("needle_syns", [])
        if not isinstance(syns, list) or any(not isinstance(s, str) or not s for s in syns):
            problems.append(f"{tag}: needle_syns must be a list of non-empty strings")

        needle = note.get("needle", "")
        if ntype in {"character_plant", "character_echo", "location_turning_point", "supersession", "foreshadow"}:
            if not needle:
                problems.append(f"{tag}: type '{ntype}' requires a non-empty 'needle'")
            elif needle not in syns and not any(needle.lower() in s.lower() or s.lower() in needle.lower() for s in syns):
                problems.append(f"{tag}: 'needle' should also appear in (or overlap) needle_syns")

        plant = note.get("plant", {})
        recall = note.get("recall", {})
        pturn = plant.get("turn") if isinstance(plant, dict) else None
        rturn = recall.get("due") if isinstance(recall, dict) else None

        if pturn is not None and not (0 <= int(pturn) < total_turns):
            problems.append(f"{tag}: plant.turn {pturn!r} out of range [0,{total_turns})")
        if rturn is not None and not (0 <= int(rturn) < total_turns):
            problems.append(f"{tag}: recall.due {rturn!r} out of range [0,{total_turns})")
        if pturn is not None and rturn is not None and int(pturn) > int(rturn):
            problems.append(f"{tag}: plant.turn ({pturn}) must be <= recall.due ({rturn})")

        if ntype in REQUIRES_RECALL and not recall:
            problems.append(f"{tag}: type '{ntype}' requires a 'recall' block")
        if recall and not recall.get("hint") and ntype != "style_hold":
            problems.append(f"{tag}: recall block requires a 'hint'")

        if ntype == "character_echo" and not rturn:
            problems.append(f"{tag}: character_echo requires recall.due")

    dss_beats = data.get("dss_beats")
    if dss_beats is not None:
        if not isinstance(dss_beats, list):
            problems.append(f"{name}: 'dss_beats' must be a list")
        else:
            beat_ids = set()
            for i, beat in enumerate(dss_beats):
                tag = f"{name}:dss_beats[{i}]"
                bid = beat.get("id")
                if not bid:
                    problems.append(f"{tag}: missing 'id'")
                elif bid in beat_ids:
                    problems.append(f"{tag}: duplicate beat id '{bid}'")
                beat_ids.add(bid)
                bturn = beat.get("turn")
                if bturn is None or not (0 <= int(bturn) < total_turns):
                    problems.append(f"{tag}: 'turn' required and in range [0,{total_turns}), got {bturn!r}")
                if not beat.get("content") and not beat.get("expectation"):
                    problems.append(f"{tag}: missing 'content' (the beat's expectation)")

    return problems


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    total = 100
    if len(sys.argv) > 2 and sys.argv[2].isdigit():
        total = int(sys.argv[2])
    failed = 0
    for arg in sys.argv[1:]:
        if arg.startswith("-") or arg.isdigit():
            continue
        path = Path(arg)
        problems = validate_spreadsheet(path, total)
        if problems:
            failed += 1
            for p in problems:
                print(f"  {p}")
        else:
            print(f"[validate_spreadsheet] {path.name}: OK ({total} turns)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
