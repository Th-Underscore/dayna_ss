"""Report aggregation for the long-horizon soak.

Reads ``run_dir/turn_*/result.json`` (probes, recalls, judge) plus the
``manifest.json`` and spreadsheet, and writes ``report.json`` + ``report.md``.

Metrics produced (plan §6):

- Retention rate per needle (planted -> last_seen -> survived), split by type.
- Recall success for ``character_echo`` notes at their due turn.
- Supersession correctness for ``supersession`` notes.
- Style-consistency curve and memory-fidelity curve from judge scores.
- Attribution: guide-failure (never planted) vs DSS-retention-loss vs supersession.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

NOTE_TYPES = {"character_plant", "character_echo", "plot_twist", "location_turning_point",
              "supersession", "foreshadow", "style_hold"}
TYPE_LABEL = {
    "character_plant": "plant",
    "character_echo": "echo",
    "plot_twist": "twist",
    "location_turning_point": "location",
    "supersession": "supersede",
    "foreshadow": "foreshadow",
    "style_hold": "style",
}


def _load_results(run_dir: Path) -> list[dict]:
    out = []
    for d in sorted(run_dir.glob("turn_*")):
        p = d / "result.json"
        if p.exists():
            try:
                out.append(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                continue
    return out


def _retention(notes: list[dict], results: list[dict]) -> list[dict]:
    """Per-note retention timeline from probe presence across turns."""
    rows = []
    turns = len(results)
    story_text = "\n".join(r.get("user_input", "") for r in results).lower()
    for note in notes:
        nid = note["id"]
        present = [False] * turns
        for i, r in enumerate(results):
            probe = (r.get("probes") or {}).get(nid)
            present[i] = bool(probe and probe.get("present"))
        planted = (note.get("plant") or {}).get("turn")
        observed = [i for i, ok in enumerate(present) if ok]
        last_seen = observed[-1] if observed else None
        survived = bool(observed and last_seen == turns - 1)
        expected_survive = note["type"] != "supersession"
        superseded_at = (note.get("recall") or {}).get("due")
        superseded_ok = None
        if note["type"] == "supersession" and superseded_at is not None:
            # Supersession pass = present before due, absent after.
            after = present[int(superseded_at) + 1:] if int(superseded_at) + 1 <= turns else []
            superseded_ok = (not observed) or (bool(observed) and not any(after))
        rows.append({
            "id": nid,
            "type": note["type"],
            "specificity": note.get("specificity"),
            "needle": note.get("needle", ""),
            "planted_at": planted,
            "last_seen": last_seen,
            "survived": survived,
            "expected_to_survive": expected_survive,
            "superseded_ok": superseded_ok,
            "attribution": _attribute(note, observed, present, turns, story_text),
        })
    return rows


def _needle_in_story(note: dict, story_text: str) -> bool:
    """Did the note's needle ever appear in the guide's story text (user turns)?"""
    needle = (note.get("needle") or "").strip().lower()
    return bool(needle) and needle in story_text


def _attribute(note: dict, observed: list[int], present: list[bool], turns: int,
               story_text: str = "") -> str:
    """guide_failure | dss_retention_loss | supersession | survived | not_planted.

    ``guide_failure`` means the note was scheduled to be planted but its needle
    never appeared in the guide's story text. ``dss_retention_loss`` means the
    story planted the needle but DSS never stored/retained it in its memory.
    """
    in_story = _needle_in_story(note, story_text)
    if note["type"] == "supersession" and not observed:
        return "dss_retention_loss" if in_story else "guide_failure"
    if not observed:
        if in_story:
            return "dss_retention_loss"  # guide wrote it; DSS never stored it
        return "guide_failure" if (note.get("plant") or {}).get("turn") is not None else "not_planted"
    if note["type"] == "supersession":
        due = (note.get("recall") or {}).get("due")
        if due is not None and int(due) + 1 <= turns:
            after = present[int(due) + 1:]
            if not any(after):
                return "supersession"
        return "dss_retention_loss"
    if observed[-1] != turns - 1:
        return "dss_retention_loss"
    return "survived"


def _recalls(notes: list[dict], results: list[dict]) -> list[dict]:
    rows = []
    for note in notes:
        due = (note.get("recall") or {}).get("due")
        if due is None:
            continue
        hits = [r.get("recalls") or {} for r in results]
        entry = {}
        for h in hits:
            if note["id"] in h:
                entry = h[note["id"]]
        rows.append({
            "id": note["id"],
            "due": int(due),
            "reached": int(due) < len(results),
            "echoed": entry.get("echoed", False),
            "echo_by_text_only": entry.get("echo_by_text_only", False),
            "matched_syns": entry.get("matched_syns", []),
        })
    return rows


def _curves(notes: list[dict], results: list[dict]) -> dict:
    style = []
    fidelity = []
    quality = []
    for r in results:
        j = r.get("judge")
        if not j or "error" in j:
            continue
        style.append((r["turn"], j.get("style_score")))
        fidelity.append((r["turn"], j.get("memory_fidelity")))
        quality.append((r["turn"], j.get("quality_score")))
    return {
        "style_curve": style,
        "fidelity_curve": fidelity,
        "quality_curve": quality,
    }


def _audits(beats: list[dict], results: list[dict]) -> dict:
    """Aggregate DSS-save audit verdicts (per dss_beat) from turn results."""
    per_beat = {}
    for r in results:
        audits = r.get("audit")
        if not isinstance(audits, list):
            continue
        for a in audits:
            if not isinstance(a, dict):
                continue
            bid = a.get("id") or "?"
            per_beat.setdefault(bid, {"id": bid, "statuses": [], "details": []})
            per_beat[bid]["statuses"].append((r["turn"], a.get("status"), a.get("detail", "")))
    rows = []
    for bid, info in per_beat.items():
        statuses = info["statuses"]
        last_turn, last_status, last_detail = statuses[-1]
        rows.append({
            "id": bid,
            "turn": last_turn,
            "status": last_status,
            "detail": last_detail,
            "statuses": statuses,
        })
    rows.sort(key=lambda x: x["turn"])
    pass_count = sum(1 for r in rows if r["status"] == "saved")
    return {
        "rows": rows,
        "total": len(rows),
        "saved": pass_count,
        "partial": sum(1 for r in rows if r["status"] == "partial"),
        "missing": sum(1 for r in rows if r["status"] == "missing"),
        "wrong": sum(1 for r in rows if r["status"] == "wrong"),
        "unassessable": sum(1 for r in rows if r["status"] == "unassessable"),
        "error": sum(1 for r in rows if r["status"] == "error"),
    }


def build_report(run_dir: Path, spreadsheet: dict, args=None) -> dict:
    results = _load_results(run_dir)
    notes = spreadsheet.get("notes", [])
    retention = _retention(notes, results)
    recalls = _recalls(notes, results)
    curves = _curves(notes, results)
    audits = _audits(spreadsheet.get("dss_beats", []), results)

    by_type = defaultdict(list)
    for row in retention:
        by_type[row["type"]].append(row)

    type_stats = {}
    for t, rows in by_type.items():
        n = len(rows)
        survived = sum(1 for r in rows if r["survived"] and r["attribution"] != "supersession")
        guide_fail = sum(1 for r in rows if r["attribution"] == "guide_failure")
        dss_loss = sum(1 for r in rows if r["attribution"] == "dss_retention_loss")
        superseded = sum(1 for r in rows if r["attribution"] == "supersession")
        type_stats[t] = {
            "count": n,
            "survived": survived,
            "guide_failure": guide_fail,
            "dss_retention_loss": dss_loss,
            "superseded": superseded,
            "retention_rate": (survived / n * 100.0) if n else 0.0,
        }

    # B2: only count recalls whose due turn was actually reached — a note with
    # due beyond the run never fired a probe, so it's not a DSS failure.
    reached_recalls = [r for r in recalls if r.get("reached")]
    echo_notes = [r for r in reached_recalls if r.get("echoed")]
    total_echo = len(reached_recalls)
    supersession_rows = [r for r in retention if r["type"] == "supersession" and r.get("superseded_ok") is not None]
    supersession_ok = sum(1 for r in supersession_rows if r["superseded_ok"])

    style_scores = [s for _, s in curves["style_curve"]]
    fidelity_scores = [f for _, f in curves["fidelity_curve"]]

    final_overview = None
    fo_path = run_dir / "final_overview.json"
    if fo_path.exists():
        try:
            final_overview = json.loads(fo_path.read_text(encoding="utf-8"))
        except Exception:
            final_overview = None

    rolling_overviews = []
    for ov_path in sorted(run_dir.glob("rolling_overview_*.json")):
        try:
            ov = json.loads(ov_path.read_text(encoding="utf-8"))
            ov["_file"] = ov_path.name
            rolling_overviews.append(ov)
        except Exception:
            pass

    return {
        "spreadsheet": spreadsheet["id"],
        "genre": spreadsheet.get("genre"),
        "turns_run": len(results),
        "notes_total": len(notes),
        "retention": retention,
        "recalls": recalls,
        "curves": curves,
        "by_type": type_stats,
        "audits": audits,
        "final_overview": final_overview,
        "rolling_overviews": rolling_overviews,
        "summary": {
            "overall_retention": sum(1 for r in retention if r["attribution"] == "survived") / max(1, len(notes)) * 100.0,
            "recall_success": len(echo_notes) / max(1, total_echo) * 100.0,
            "supersession_ok": supersession_ok,
            "supersession_total": len(supersession_rows),
            "guide_failures": sum(1 for r in retention if r["attribution"] == "guide_failure"),
            "dss_losses": sum(1 for r in retention if r["attribution"] == "dss_retention_loss"),
            "avg_style": (sum(style_scores) / len(style_scores)) if style_scores else None,
            "avg_fidelity": (sum(fidelity_scores) / len(fidelity_scores)) if fidelity_scores else None,
            "style_curve": curves["style_curve"],
            "fidelity_curve": curves["fidelity_curve"],
        },
    }


def _md(report: dict) -> str:
    s = report["summary"]
    lines = [
        f"# Long-Horizon Soak Report — {report['spreadsheet']} ({report['genre']})",
        "",
        f"Turns run: {report['turns_run']} | Notes: {report['notes_total']}",
        "",
        "## Summary",
        "",
        f"- Overall retention: **{s['overall_retention']:.0f}%**",
        f"- Recall success: **{s['recall_success']:.0f}%**",
        f"- Supersession: {s['supersession_ok']}/{s['supersession_total']} correct",
        f"- Guide failures: {s['guide_failures']} | DSS retention losses: {s['dss_losses']}",
        f"- Avg style: {s['avg_style'] if s['avg_style'] is not None else 'n/a'} | Avg fidelity: {s['avg_fidelity'] if s['avg_fidelity'] is not None else 'n/a'}",
        "",
    ]
    rollings = report.get("rolling_overviews") or []
    if rollings:
        lines += ["## Rolling overviews (mid-run judge)", ""]
        lines += ["| checkpoint | overall | trajectory | generated |", "|---|---|---|---|"]
        for ov in rollings:
            lines.append(
                f"| {ov.get('turns', '?')} turns | {ov.get('overall_score', 'n/a')}/5 | "
                f"{ov.get('trajectory', 'n/a')} | {ov.get('generated_at', ov.get('_file', ''))} |"
            )
        lines += [""]
        for ov in rollings:
            lines += [
                f"### Rolling overview at {ov.get('turns', '?')} turns",
                "",
                f"**Verdict:** {ov.get('verdict', '')}",
                "",
            ]
            if ov.get("summary"):
                lines += [ov["summary"], ""]
        lines += ["---", ""]
    fo = report.get("final_overview")
    if fo and "error" not in fo:
        lines += [
            "## Final overview (end-of-run judge)",
            "",
            f"- Overall score: **{fo.get('overall_score', 'n/a')}/5**",
            f"- Verdict: {fo.get('verdict', '')}",
            "",
        ]
        for key, title in (("arc_progression", "Arc progression"),
                           ("style_consistency", "Style consistency"),
                           ("memory_fidelity", "Memory fidelity")):
            val = fo.get(key)
            if val:
                lines += [f"**{title}:** {val}", ""]
        if fo.get("strengths"):
            lines += ["**Strengths:**"]
            lines += [f"- {x}" for x in fo["strengths"]]
            lines += [""]
        if fo.get("weaknesses"):
            lines += ["**Weaknesses:**"]
            lines += [f"- {x}" for x in fo["weaknesses"]]
            lines += [""]
        fo_notes = fo.get("notes")
        if fo_notes:
            lines += ["Final note adherence:"]
            for n in fo_notes:
                if isinstance(n, dict):
                    lines.append(f"- {n.get('id')} ({n.get('status')}): {n.get('detail', '')}")
                else:
                    lines.append(f"- {n}")
            lines += [""]
        if fo.get("summary"):
            lines += [fo["summary"], ""]
        lines += ["---", ""]
    lines += [
        "## Retention by note type",
        "",
        "| type | count | survived | guide-fail | dss-loss | superseded | rate |",
        "|---|---|---|---|---|---|---|",
    ]
    for t, st in sorted(report["by_type"].items()):
        lines.append(
            f"| {t} | {st['count']} | {st['survived']} | {st['guide_failure']} | "
            f"{st['dss_retention_loss']} | {st['superseded']} | {st['retention_rate']:.0f}% |"
        )
    lines += ["", "## Per-note retention", "", "| id | type | needle | planted | last_seen | survived | attribution |", "|---|---|---|---|---|---|---|"]
    for r in report["retention"]:
        lines.append(
            f"| {r['id']} | {TYPE_LABEL.get(r['type'], r['type'])} | {r['needle']!r} | {r['planted_at']} | "
            f"{r['last_seen'] if r['last_seen'] is not None else 'never'} | {r['survived']} | {r['attribution']} |"
        )
    if report["recalls"]:
        lines += ["", "## Echo/recall probes", "", "| id | due | reached | echoed | text-only | matched |", "|---|---|---|---|---|---|"]
        for r in report["recalls"]:
            lines.append(
                f"| {r['id']} | {r['due']} | {'yes' if r.get('reached') else 'no (past run end)'} | "
                f"{r['echoed']} | {r['echo_by_text_only']} | {', '.join(r['matched_syns']) or '-'} |"
            )
    lines += ["", "## Curves", "", "| turn | style | fidelity | quality |", "|---|---|---|---|"]
    style_map = dict(report["curves"]["style_curve"])
    fid_map = dict(report["curves"]["fidelity_curve"])
    qual_map = dict(report["curves"]["quality_curve"])
    for t in sorted(set(style_map) | set(fid_map) | set(qual_map)):
        lines.append(f"| {t} | {style_map.get(t, '-')} | {fid_map.get(t, '-')} | {qual_map.get(t, '-')} |")
    audits = report.get("audits") or {}
    if audits.get("rows"):
        lines += ["", "## DSS save-beat audits", ""]
        lines.append(
            f"- Beats: **{audits['total']}** | saved: **{audits['saved']}** | partial: **{audits['partial']}** | "
            f"missing: **{audits['missing']}** | wrong: **{audits['wrong']}** | "
            f"unassessable: **{audits['unassessable']}** | error: **{audits['error']}**"
        )
        lines += ["", "| id | turn | status | detail |", "|---|---|---|---|"]
        for a in audits["rows"]:
            lines.append(f"| {a['id']} | {a['turn']} | {a['status']} | {a['detail']} |")
    lines.append("")
    return "\n".join(lines)


def write_report(run_dir: Path, spreadsheet: dict, args=None) -> Path:
    report = build_report(run_dir, spreadsheet, args)
    (run_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = _md(report)
    (run_dir / "report.md").write_text(md, encoding="utf-8")
    return run_dir / "report.md"


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python report.py <run_dir> [spreadsheet.json]")
        sys.exit(1)
    run_dir = Path(sys.argv[1])
    spreadsheet_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    if spreadsheet_path is None:
        man = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        from pathlib import Path as _P
        sp = _P(__file__).parent / "spreadsheets" / f"{man.get('spreadsheet')}.json"
        spreadsheet_path = sp if sp.exists() else None
    if spreadsheet_path is None:
        print("cannot locate spreadsheet; pass it explicitly")
        sys.exit(1)
    spreadsheet = json.loads(spreadsheet_path.read_text(encoding="utf-8"))
    path = write_report(run_dir, spreadsheet)
    print(f"report written: {path}")
