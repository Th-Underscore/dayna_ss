#!/usr/bin/env python3
"""Run-level analysis for soak runs: state evolution + prose quality.

Reads per-turn files (turn_NNN/state_snapshot/*.json, turn_NNN/reply.txt,
turn_NNN/instr_prompt.txt, turn_NNN/result.json) and writes a Markdown report.

Usage:
  python3 analyze_run.py <run_dir> [--sections state,prose] [--out PATH]
                         [--sample N] [--outlier-x F]

  --sections  Comma-separated: state, prose (default: both)
  --out       Output .md path (default: <run_dir>/analysis.md)
  --sample    Number of turns to sample for prose analysis (default: 12)
  --outlier-x Multiplier for 5-gram overlap outlier threshold (default: 4.0)
"""
import json
import os
import re
import statistics
import sys
from collections import Counter, OrderedDict
from pathlib import Path

STATE_FILES = [
    "characters.json", "elements.json", "arcs.json", "events.json",
    "groups.json", "knowledge.json", "current_scene.json", "general_info.json",
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def load_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return None


def entries_count(d):
    if d is None:
        return -1
    if isinstance(d, dict):
        if "entries" in d:
            e = d["entries"]
            return len(e) if isinstance(e, (dict, list)) else 0
        return 0
    if isinstance(d, list):
        return len(d)
    return 0


def who_names(scene, phase):
    out = []
    try:
        who = scene[phase]["who"]
        for c in who.get("characters", []) or []:
            if isinstance(c, dict) and c.get("name"):
                out.append(c["name"])
        for g in who.get("groups", []) or []:
            if isinstance(g, dict) and g.get("name"):
                out.append(g["name"])
    except Exception:
        pass
    return out


def ngrams(text, n=5):
    words = re.findall(r"[a-z']+", text.lower())
    return {" ".join(words[i:i+n]) for i in range(len(words) - n + 1)}


def ngram_overlap(a, b, n=5):
    ga, gb = ngrams(a, n), ngrams(b, n)
    shared = ga & gb
    return {
        "count": len(shared),
        "ratio_in_reply": len(shared) / len(ga) if ga else 0.0,
        "ratio_in_instr": len(shared) / len(gb) if gb else 0.0,
        "shared": shared,
    }


def fp_markers(text) -> dict[str, float]:
    tl = text.lower()
    return {
        "I": float(len(re.findall(r"\bI\b", text))),
        "my": float(len(re.findall(r"\bmy\b", tl))),
        "me": float(len(re.findall(r"\bme\b", tl))),
        "contr": float(len(re.findall(r"\bI['\u2019](?:m|ve|ll|d|s|re)\b", tl))),
    }


def percentile(data, p):
    s = sorted(data)
    if not s:
        return 0
    k = (len(s) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def get_turn_dirs(run_dir: Path) -> list[Path]:
    return sorted(
        d for d in run_dir.iterdir()
        if d.name.startswith("turn_") and d.is_dir()
    )


def read_turn(tdir: Path):
    reply = (tdir / "reply.txt").read_text(encoding="utf-8", errors="replace")
    instr = (tdir / "instr_prompt.txt").read_text(encoding="utf-8", errors="replace")
    result = load_json(tdir / "result.json") or {}
    return reply, instr, result


# ---------------------------------------------------------------------------
# State evolution section
# ---------------------------------------------------------------------------

def scan_turn_state(turn_dir: Path) -> dict | None:
    ss = turn_dir / "state_snapshot"
    if not ss.is_dir():
        return None
    row: dict = {"dir": turn_dir.name}
    sizes = {}
    for f in STATE_FILES:
        p = ss / f
        sizes[f] = p.stat().st_size if p.exists() else -1
    row["sizes"] = sizes
    row["total_bytes"] = sum(v for v in sizes.values() if v >= 0)

    chars = load_json(ss / "characters.json")
    elems = load_json(ss / "elements.json")
    arcs = load_json(ss / "arcs.json")
    ev = load_json(ss / "events.json")
    groups = load_json(ss / "groups.json")
    know = load_json(ss / "knowledge.json")
    scene = load_json(ss / "current_scene.json")
    gen = load_json(ss / "general_info.json")

    row["characters"] = entries_count(chars)
    row["elements"] = entries_count(elems)
    row["arcs"] = entries_count(arcs)
    row["groups"] = entries_count(groups)
    row["knowledge"] = entries_count(know)

    if isinstance(chars, dict) and isinstance(chars.get("entries"), dict):
        row["char_names"] = set(chars["entries"].keys())
    else:
        row["char_names"] = set()

    def ev_count(key):
        if not isinstance(ev, dict):
            return 0
        v = ev.get(key)
        if v is None:
            return 0
        return len(v) if isinstance(v, (dict, list)) else 0
    row["ev_past"] = ev_count("past")
    row["ev_scenes"] = ev_count("scenes")
    row["ev_chapters"] = ev_count("chapters")
    row["ev_beats"] = ev_count("beats")

    if isinstance(scene, dict):
        row["scene_no"] = scene.get("_scene_number")
        row["chapter_no"] = scene.get("_chapter_number")
        row["arc_no"] = scene.get("_arc_number")
        row["where"] = (scene.get("now") or {}).get("where")
        row["who_now"] = who_names(scene, "now")
        row["who_start"] = who_names(scene, "start")
        row["history_len"] = len(scene.get("_history", []) or [])
    else:
        row.update({"scene_no": None, "chapter_no": None, "arc_no": None,
                     "where": None, "who_now": [], "who_start": [], "history_len": 0})

    if isinstance(gen, dict):
        row["synopsis_len"] = len(gen.get("synopsis", "") or "")
        row["synopsis"] = gen.get("synopsis", "") or ""
    else:
        row["synopsis_len"] = 0
        row["synopsis"] = ""
    return row


def write_state_section(L, run_dir: Path, turn_dirs: list[Path]):
    rows, missing = [], []
    for td in turn_dirs:
        r = scan_turn_state(td)
        if r is None:
            missing.append(td.name)
        else:
            rows.append(r)

    L.append(f"# State Evolution: {run_dir.name}\n")
    L.append(f"**Turns with snapshots:** {len(rows)} of {len(turn_dirs)}\n")
    if missing:
        L.append(f"**Without snapshots:** {', '.join(missing)}\n")
    L.append("")

    if not rows:
        L.append("No state snapshots found.\n")
        return

    # 1. Per-turn tracking
    L.append("## 1. Per-Turn State\n")
    L.append("| Turn | Sc# | Ch# | Ar# | Chars | Elems | Groups | Know | Ev Past | Ev Sc | Ev Ch | Ev Bt | Total B |\n")
    L.append("|------|-----|-----|-----|-------|-------|--------|------|---------|-------|-------|-------|---------|\n")
    for r in rows:
        L.append(f"| {r['dir']} | {r['scene_no']} | {r['chapter_no']} | {r['arc_no']} | "
                 f"{r['characters']} | {r['elements']} | {r['groups']} | {r['knowledge']} | "
                 f"{r['ev_past']} | {r['ev_scenes']} | {r['ev_chapters']} | {r['ev_beats']} | "
                 f"{r['total_bytes']:,} |\n")
    L.append("")

    # 2. Turns per scene
    L.append("## 2. Turns per Scene\n")
    scene_turns = OrderedDict()
    for r in rows:
        scene_turns.setdefault(r["scene_no"], []).append(r["dir"])
    L.append("| Scene# | Turns | Count |\n|--------|-------|-------|\n")
    for sn, ts in scene_turns.items():
        L.append(f"| {sn} | {ts[0]}..{ts[-1]} | {len(ts)} |\n")
    L.append("")

    # 3. Entity growth
    L.append("## 3. Entity Growth (value, \u0394)\n")
    L.append("| Turn | Chars | Elems | Know | Groups |\n")
    L.append("|------|-------|-------|------|--------|\n")
    prev = {}
    for r in rows:
        def fmt(key):
            v = r[key]  # type: ignore[index]
            d = v - prev.get(key, v) if key in prev else 0
            prev[key] = v
            return f"{v} ({d:+d})"
        L.append(f"| {r['dir']} | {fmt('characters')} | {fmt('elements')} | {fmt('knowledge')} | {fmt('groups')} |\n")
    L.append("")

    # 4. Character entry/exit
    L.append("## 4. Character Entry/Exit\n")
    prev_names = set()
    any_change = False
    for r in rows:
        cur = r["char_names"]
        added = sorted(cur - prev_names)
        removed = sorted(prev_names - cur)
        if added or removed:
            any_change = True
            L.append(f"- **{r['dir']}**: +{added if added else '\u2205'} "
                     f"\u2212{removed if removed else '\u2205'}  (now {sorted(cur)})\n")
        prev_names = cur
    if not any_change:
        L.append("- No character entry/exit detected.\n")
    if rows:
        L.append(f"- **Final cast:** {sorted(prev_names)}\n")
    L.append("")

    # 5. Scene transitions
    L.append("## 5. Scene Transitions\n")
    prev_sn, prev_where, trans = None, None, 0
    for r in rows:
        sn = r["scene_no"]
        if prev_sn is not None and sn != prev_sn:
            trans += 1
            extra = f"  ({prev_where} \u2192 {r['where']})" if (prev_where and r["where"] and prev_where != r["where"]) else ""
            L.append(f"- **{r['dir']}**: scene {prev_sn} \u2192 {sn}{extra}\n")
        prev_sn, prev_where = sn, r["where"]
    L.append(f"- **Total transitions:** {trans}\n")
    L.append("")

    # 6. Rolling overviews
    ro_files = sorted(f for f in os.listdir(run_dir) if f.startswith("rolling_overview_") and f.endswith(".json"))
    if ro_files:
        L.append("## 6. Rolling Overviews\n")
        L.append("| File | Size | Score | Trajectory | Turns |\n")
        L.append("|------|------|-------|------------|-------|\n")
        for f in ro_files:
            d = load_json(run_dir / f)
            sz = (run_dir / f).stat().st_size
            if isinstance(d, dict):
                L.append(f"| {f} | {sz} | {d.get('overall_score')} | {d.get('trajectory')} | {d.get('turns')} |\n")
            else:
                L.append(f"| {f} | {sz} | (unreadable) | | |\n")
        L.append("")

    # 7. Synopsis growth
    L.append("## 7. Synopsis Growth\n")
    L.append("| Turn | Chars |\n|------|-------|\n")
    for r in rows:
        L.append(f"| {r['dir']} | {r['synopsis_len']} |\n")
    L.append("")

    # 8. Anomalies
    L.append("## 8. Anomalies\n")
    flags = []
    prev_counts = {}
    for r in rows:
        t = r["dir"]
        sn = r["scene_no"]
        if sn is not None and "scene_no" in prev_counts and sn < prev_counts["scene_no"]:
            flags.append(f"- **{t}**: scene number RESET {prev_counts['scene_no']} \u2192 {sn}")
        prev_counts["scene_no"] = sn
        for key, label in [("characters", "characters"), ("elements", "elements"),
                            ("knowledge", "knowledge"), ("groups", "groups")]:
            v = r[key]
            if v < 0:
                flags.append(f"- **{t}**: {label} UNREADABLE")
            elif key in prev_counts and v < prev_counts[key]:
                flags.append(f"- **{t}**: {label} DROP {prev_counts[key]} \u2192 {v}")
            prev_counts[key] = v
        for f, sz in r["sizes"].items():
            if sz == 0:
                flags.append(f"- **{t}**: EMPTY {f}")
            elif sz == -1:
                flags.append(f"- **{t}**: MISSING {f}")
    L.append("\n".join(flags) + "\n" if flags else "- No anomalies detected.\n")
    L.append("")


# ---------------------------------------------------------------------------
# Prose quality section
# ---------------------------------------------------------------------------

def write_prose_section(L, run_dir: Path, turn_dirs: list[Path],
                        n_sample: int = 12, outlier_x: float = 4.0):
    total = len(turn_dirs)
    if total == 0:
        L.append("No turn directories found.\n")
        return

    # Evenly spaced sample
    if n_sample >= total:
        sample_turns = list(range(total))
    else:
        step = total // n_sample
        sample_turns = list(range(0, total, step))[:n_sample]
        if sample_turns[-1] != total - 1:
            sample_turns[-1] = total - 1
        sample_turns = sorted(set(sample_turns))

    # 1. Length distribution
    lengths = []
    for td in turn_dirs:
        try:
            reply = (td / "reply.txt").read_text(encoding="utf-8", errors="replace")
            lengths.append(len(reply))
        except Exception:
            lengths.append(0)

    L.append(f"# Prose Quality: {run_dir.name}\n")
    L.append(f"**Turns:** {total}  |  **Sampled:** {len(sample_turns)}\n")

    L.append("## 1. Reply Length Distribution\n")
    L.append("| Metric | Value |\n|--------|-------|\n")
    L.append(f"| Mean | {statistics.mean(lengths):.0f} chars |\n")
    L.append(f"| Median | {statistics.median(lengths):.0f} chars |\n")
    L.append(f"| P95 | {percentile(lengths, 95):.0f} chars |\n")
    L.append(f"| Min | {min(lengths)} chars |\n")
    L.append(f"| Max | {max(lengths)} chars |\n")
    if len(lengths) > 1:
        L.append(f"| Std dev | {statistics.stdev(lengths):.0f} chars |\n")
    L.append("")

    # 2. 5-gram overlap
    L.append("## 2. 5-Gram Overlap (instr \u2192 reply)\n")
    L.append("| Turn | Reply ch | Shared | % in reply |\n")
    L.append("|------|----------|--------|------------|\n")
    ov = {}
    for i, td in enumerate(turn_dirs):
        if i not in sample_turns:
            continue
        try:
            reply, instr, _ = read_turn(td)
        except Exception:
            continue
        d = ngram_overlap(reply, instr)
        d["reply_len"] = len(reply)
        ov[i] = d
        L.append(f"| {i} | {d['reply_len']} | {d['count']} | {d['ratio_in_reply']*100:.1f}% |\n")
    L.append("")

    if ov:
        ratios = [d["ratio_in_reply"] for d in ov.values()]
        baseline = statistics.median(ratios)
        L.append(f"**Baseline (median): {baseline*100:.1f}%**\n")
        outliers = sorted(t for t, d in ov.items() if d["ratio_in_reply"] >= outlier_x * baseline)
        if outliers:
            ot = ", ".join(f"turn {t} ({ov[t]['ratio_in_reply']*100:.1f}%)" for t in outliers)
            L.append(f"**Outliers (\u2265 {outlier_x:.0f}\u00d7 baseline):** {ot}\n")
        L.append("")

    # 3. Flagged turns (lowest judge scores)
    L.append("## 3. Flagged Turns (Lowest Judge Scores)\n")
    judged = []
    for i, td in enumerate(turn_dirs):
        try:
            _, _, result = read_turn(td)
        except Exception:
            continue
        j = result.get("judge")
        if j and j.get("summary"):
            judged.append({
                "turn": i,
                "style": j.get("style_score"), "quality": j.get("quality_score"),
                "memory": j.get("memory_fidelity"),
                "avg": (j.get("style_score", 0) + j.get("quality_score", 0) + j.get("memory_fidelity", 0)) / 3,
                "summary": j["summary"],
                "notes": j.get("notes", []),
            })
    if judged:
        judged.sort(key=lambda x: x["avg"])
        L.append("| Turn | Style | Quality | Memory | Avg | Summary |\n")
        L.append("|------|-------|---------|--------|-----|---------|\n")
        for ft in judged[:5]:
            L.append(f"| {ft['turn']} | {ft['style']} | {ft['quality']} | {ft['memory']} | {ft['avg']:.2f} | {ft['summary'][:70]}\u2026 |\n")
        L.append("")
    else:
        L.append("No judge data found.\n")

    # 4. Voice consistency
    L.append("## 4. Voice Consistency (first-person density /1000 chars)\n")
    L.append("| Turn | I | my | me | contr | total | density |\n")
    L.append("|------|---|----|----|-------|-------|---------|\n")
    voice = {}
    for i, td in enumerate(turn_dirs):
        if i not in sample_turns:
            continue
        try:
            reply, _, _ = read_turn(td)
        except Exception:
            continue
        m = fp_markers(reply)
        m["total"] = sum(m.values())
        m["density"] = m["total"] / max(len(reply), 1) * 1000
        voice[i] = m
        L.append(f"| {i} | {m['I']} | {m['my']} | {m['me']} | {m['contr']} | {m['total']} | {m['density']:.1f} |\n")
    L.append("")

    # 5. Note adherence
    L.append("## 5. Note Adherence\n")
    note_stats = {}
    for i, td in enumerate(turn_dirs):
        try:
            _, _, result = read_turn(td)
        except Exception:
            continue
        j = result.get("judge")
        if not j:
            continue
        for n in j.get("notes", []):
            note_stats.setdefault(n.get("id", "?"), Counter())[n.get("status", "?")] += 1
    if note_stats:
        L.append("| Note | missed | superseded | n/a | total |\n")
        L.append("|------|--------|------------|-----|-------|\n")
        for nid in sorted(note_stats):
            s = note_stats[nid]
            L.append(f"| {nid} | {s.get('missed',0)} | {s.get('superseded',0)} | {s.get('n/a',0)} | {sum(s.values())} |\n")
    else:
        L.append("No notes tracked.\n")
    L.append("")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <run_dir> [--sections state,prose] [--out PATH] [--sample N] [--outlier-x F]")
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    if not run_dir.is_dir():
        print(f"Not a directory: {run_dir}")
        sys.exit(1)

    sections = {"state", "prose"}
    out_path = run_dir / "analysis.md"
    n_sample = 12
    outlier_x = 4.0

    args = sys.argv[2:]
    for i, arg in enumerate(args):
        if arg == "--sections" and i + 1 < len(args):
            sections = set(args[i + 1].split(","))
        elif arg == "--out" and i + 1 < len(args):
            out_path = Path(args[i + 1])
        elif arg == "--sample" and i + 1 < len(args):
            n_sample = int(args[i + 1])
        elif arg == "--outlier-x" and i + 1 < len(args):
            outlier_x = float(args[i + 1])

    turn_dirs = get_turn_dirs(run_dir)
    if not turn_dirs:
        print("No turn_NNN directories found.")
        sys.exit(1)

    print(f"Run: {run_dir.name}  |  {len(turn_dirs)} turns  |  sections: {sorted(sections)}")

    L = []
    if "state" in sections:
        print("  State evolution...")
        write_state_section(L, run_dir, turn_dirs)
    if "prose" in sections:
        print("  Prose quality...")
        write_prose_section(L, run_dir, turn_dirs, n_sample, outlier_x)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(L), encoding="utf-8")
    print(f"Wrote {len(L)} lines \u2192 {out_path}")


if __name__ == "__main__":
    main()