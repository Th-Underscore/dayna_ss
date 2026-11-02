#!/usr/bin/env python3
"""Analyze s522 run: timing, trigger reorg effectiveness, chapter/arc archival, data comparison vs s521."""

import json
import re
import statistics
from pathlib import Path
from collections import defaultdict

BASE = Path("extensions/dayna_ss/tests/runs")
LOG522 = BASE / "logs" / "scifi_heist_gemma4_s522.log"
RUN522 = BASE / "sci_fi_heist__cce5eba2"
RUN521 = BASE / "sci_fi_heist__f42ea382"  # s521 run dir
OUT = BASE / "docs" / "s522_analysis.md"

# --- Regex patterns (matching the log format) ---
# logtime format: [HH:MM:SS.mmm] [YYYY-MM-DD HH:MM:SS] <content>
TS = r"\[\d{2}:\d{2}:\d{2}\.\d{3}\] \[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]"
TURN_START = re.compile(rf"{TS} --- turn (\d+)/(\d+) ---")
TURN_DONE = re.compile(rf"{TS}\s+done: turn (\d+) in ([\d.]+)s")
LOCAL_CALL = re.compile(rf"{TS}\s+local-call (\S+)/generate: ttft=([\d.]+)s total=([\d.]+)s in=(\d+) out=(\d+) tok")
SCENE_MARKER = re.compile(r"scene_number.*?(\d+)")

def parse_log():
    turns = {}
    current_turn = None
    local_calls = []
    scene_turns = set()
    
    with open(LOG522, "r", errors="replace") as f:
        for line in f:
            m = TURN_START.search(line)
            if m:
                current_turn = int(m.group(1))
                turns.setdefault(current_turn, {"local": []})
                continue
            m = TURN_DONE.search(line)
            if m:
                t = int(m.group(1))
                turns.setdefault(t, {"local": []})
                turns[t]["wall_s"] = float(m.group(2))
                continue
            m = LOCAL_CALL.search(line)
            if m and current_turn is not None:
                kind = m.group(1)
                ttft = float(m.group(2))
                total = float(m.group(3))
                in_tok = int(m.group(4))
                out_tok = int(m.group(5))
                local_calls.append((current_turn, ttft, total, in_tok, out_tok))
                turns[current_turn]["local"].append((ttft, total, in_tok, out_tok))
    
    return turns, local_calls

def get_scene_turns():
    """Identify scene turns from the run directory (current_scene.json _scene_number changes)."""
    scene_turns = set()
    last_scene = -1
    turn_dirs = sorted(RUN522.glob("turn_*"))
    for td in turn_dirs:
        cs_path = td / "state_snapshot" / "current_scene.json"
        if cs_path.exists():
            try:
                cs = json.loads(cs_path.read_text())
                sn = cs.get("_scene_number", -1)
                if sn != last_scene:
                    if last_scene >= 0:  # skip turn 0
                        scene_turns.add(int(td.name.split("_")[1]))
                    last_scene = sn
            except (json.JSONDecodeError, KeyError):
                pass
    return scene_turns

def get_event_counts(run_dir):
    """Get chapter/arc/scene/beats counts from the latest turn's events.json."""
    turn_dirs = sorted(run_dir.glob("turn_*"))
    if not turn_dirs:
        return {}
    last = turn_dirs[-1]
    snap = last / "state_snapshot"
    result = {"turn": last.name}
    events_path = snap / "events.json"
    if events_path.exists():
        try:
            ev = json.loads(events_path.read_text())
            result["chapters"] = len(ev.get("chapters", []))
            result["scenes"] = len(ev.get("scenes", []))
            result["beats"] = len(ev.get("beats", []))
            result["past"] = len(ev.get("past", []))
        except json.JSONDecodeError:
            pass
    # arcs.json
    arcs_path = snap / "arcs.json"
    if arcs_path.exists():
        try:
            arcs = json.loads(arcs_path.read_text())
            if isinstance(arcs, list):
                result["arcs"] = len(arcs)
            elif isinstance(arcs, dict):
                result["arcs"] = len(arcs.get("entries", arcs))
        except json.JSONDecodeError:
            pass
    # characters
    chars_path = snap / "characters.json"
    if chars_path.exists():
        try:
            chars = json.loads(chars_path.read_text())
            if isinstance(chars, list):
                result["characters"] = len(chars)
            elif isinstance(chars, dict):
                result["characters"] = len(chars.get("entries", chars))
        except json.JSONDecodeError:
            pass
    # elements
    elem_path = snap / "elements.json"
    if elem_path.exists():
        try:
            elems = json.loads(elem_path.read_text())
            if isinstance(elems, list):
                result["elements"] = len(elems)
            elif isinstance(elems, dict):
                result["elements"] = len(elems.get("entries", elems))
        except json.JSONDecodeError:
            pass
    return result

def percentile(data, p):
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)

def main():
    print(f"Parsing {LOG522}...")
    turns, local_calls = parse_log()
    print(f"  {len(turns)} turns, {len(local_calls)} local calls")
    
    scene_turns = get_scene_turns()
    print(f"  {len(scene_turns)} scene turns identified")
    
    # --- Separate scene vs non-scene ---
    non_scene = {}
    scene = {}
    for t, data in turns.items():
        if "wall_s" not in data:
            continue
        if t in scene_turns:
            scene[t] = data
        else:
            non_scene[t] = data
    
    # --- Timing stats ---
    def timing_stats(turn_dict, label):
        walls = [d["wall_s"] for d in turn_dict.values()]
        call_counts = [len(d["local"]) for d in turn_dict.values()]
        all_calls = [c for d in turn_dict.values() for c in [x[1] for x in d["local"]]]
        all_in = [c for d in turn_dict.values() for c in [x[2] for x in d["local"]]]
        
        lines = []
        lines.append(f"### {label} ({len(turn_dict)} turns)")
        lines.append("")
        lines.append("| Metric | Mean | Median | P95 | Min | Max |")
        lines.append("|--------|------|--------|-----|-----|-----|")
        if walls:
            lines.append(f"| Wall time | {statistics.mean(walls):.1f}s | {statistics.median(walls):.1f}s | {percentile(walls,95):.1f}s | {min(walls):.1f}s | {max(walls):.1f}s |")
        if call_counts:
            lines.append(f"| Local calls | {statistics.mean(call_counts):.1f} | {statistics.median(call_counts):.0f} | {percentile(call_counts,95):.0f} | {min(call_counts)} | {max(call_counts)} |")
        if all_calls:
            lines.append(f"| Call total time | {statistics.mean(all_calls):.1f}s | {statistics.median(all_calls):.1f}s | {percentile(all_calls,95):.1f}s | {min(all_calls):.1f}s | {max(all_calls):.1f}s |")
        if all_in:
            lines.append(f"| Input tokens/call | {statistics.mean(all_in):.0f} | {statistics.median(all_in):.0f} | {percentile(all_in,95):.0f} | {min(all_in)} | {max(all_in)} |")
        lines.append("")
        return lines
    
    # --- Prefill/decode split ---
    def prefill_decode(turn_dict):
        ttfts = [c[0] for d in turn_dict.values() for c in d["local"]]
        totals = [c[1] for d in turn_dict.values() for c in d["local"]]
        prefills = sum(ttfts)
        decodes = sum(t - ttf for ttf, t in zip(ttfts, totals) if t > ttf)
        return prefills, decodes, ttfts, totals
    
    # --- Build report ---
    lines = []
    lines.append("# s522 Analysis: Trigger Reorg + No-Schema-Descriptions")
    lines.append("")
    lines.append(f"Run: `sci_fi_heist__cce5eba2` (seed 522, 88 turns completed)")
    lines.append(f"Log: `{LOG522.name}`")
    lines.append(f"Config: `--max-subject-workers 1 --no-schema-descriptions` + trigger reorg (2 always calls)")
    lines.append("")
    
    # Section 1: Timing
    lines.append("## 1. Timing: Non-Scene vs Scene Turns")
    lines.append("")
    lines.extend(timing_stats(non_scene, "Non-Scene Turns"))
    lines.extend(timing_stats(scene, "Scene Turns"))
    
    # Prefill/decode
    ns_p, ns_d, ns_ttf, ns_tot = prefill_decode(non_scene)
    s_p, s_d, s_ttf, s_tot = prefill_decode(scene)
    lines.append("### Prefill vs Decode")
    lines.append("")
    lines.append("| Category | Prefill (s) | Decode (s) | Prefill % |")
    lines.append("|----------|-------------|------------|-----------|")
    ns_total = ns_p + ns_d
    s_total = s_p + s_d
    if ns_total > 0:
        lines.append(f"| Non-scene | {ns_p:.0f} | {ns_d:.0f} | {100*ns_p/ns_total:.0f}% |")
    if s_total > 0:
        lines.append(f"| Scene | {s_p:.0f} | {s_d:.0f} | {100*s_p/s_total:.0f}% |")
    lines.append("")
    
    # Effective prefill rate
    ns_in_total = sum(c[2] for d in non_scene.values() for c in d["local"])
    if ns_p > 0:
        lines.append(f"Effective prefill rate (non-scene): {ns_in_total/ns_p:,.0f} tok/s")
    s_in_total = sum(c[2] for d in scene.values() for c in d["local"])
    if s_p > 0:
        lines.append(f"Effective prefill rate (scene): {s_in_total/s_p:,.0f} tok/s")
    lines.append("")
    
    # Section 2: Call count distribution
    lines.append("## 2. Local Call Count Distribution (Non-Scene)")
    lines.append("")
    lines.append("| Calls | Turn Count | Turns |")
    lines.append("|-------|-----------|-------|")
    call_dist = defaultdict(list)
    for t, d in sorted(non_scene.items()):
        n = len(d["local"])
        call_dist[n].append(t)
    for n in sorted(call_dist.keys()):
        tlist = call_dist[n]
        turns_str = ", ".join(str(t) for t in tlist[:10])
        if len(tlist) > 10:
            turns_str += f" (+{len(tlist)-10} more)"
        lines.append(f"| {n} | {len(tlist)} | {turns_str} |")
    lines.append("")
    
    # Section 3: Comparison vs s521
    lines.append("## 3. Comparison vs s521 Baseline")
    lines.append("")
    lines.append("| Metric | s521 (baseline) | s522 (trigger reorg + slim) | Change |")
    lines.append("|--------|----------------|---------------------------|--------|")
    
    # s521 baseline from prior analysis: 17-20 calls non-scene, 86-108s
    ns_mean_calls = statistics.mean(len(d["local"]) for d in non_scene.values()) if non_scene else 0
    ns_mean_wall = statistics.mean(d["wall_s"] for d in non_scene.values()) if non_scene else 0
    lines.append(f"| Non-scene calls/turn | 17-20 | {ns_mean_calls:.1f} | {100*(1-ns_mean_calls/18):.0f}% reduction |")
    lines.append(f"| Non-scene wall time | 86-108s | {ns_mean_wall:.1f}s | {100*(1-ns_mean_wall/97):.0f}% reduction |")
    
    s_mean_calls = statistics.mean(len(d["local"]) for d in scene.values()) if scene else 0
    s_mean_wall = statistics.mean(d["wall_s"] for d in scene.values()) if scene else 0
    lines.append(f"| Scene calls/turn | 40-70 | {s_mean_calls:.1f} | — |")
    lines.append(f"| Scene wall time | 200-300s | {s_mean_wall:.1f}s | — |")
    
    ns_mean_in = statistics.mean(c[2] for d in non_scene.values() for c in d["local"]) if non_scene else 0
    lines.append(f"| Non-scene in-tok/call | ~10,000 | {ns_mean_in:,.0f} | {100*(1-ns_mean_in/10000):.0f}% reduction |")
    lines.append("")
    
    # Section 4: Chapter/Arc archival
    lines.append("## 4. Chapter + Arc Archival")
    lines.append("")
    ev522 = get_event_counts(RUN522)
    lines.append(f"Latest turn: {ev522.get('turn', '?')}")
    lines.append(f"| Entity | Count |")
    lines.append(f"|--------|-------|")
    for key in ["chapters", "scenes", "beats", "past", "arcs", "characters", "elements"]:
        if key in ev522:
            lines.append(f"| {key} | {ev522[key]} |")
    lines.append("")
    
    # Check 1:1 - count chapter boundary turns from log
    # (chapters should match the number of distinct chapter archival events)
    # For now just report the counts; the 1:1 check needs the log's chapter events
    lines.append("### 1:1 Check")
    lines.append("")
    # Parse chapter/arc archival events from log
    chapter_events = []
    arc_events = []
    with open(LOG522, "r", errors="replace") as f:
        for line in f:
            if "archiving chapter" in line.lower() or "chapter.*archiv" in line.lower():
                m = TURN_START.search(line)
                chapter_events.append(line.strip()[:120])
            if "archiving arc" in line.lower() or "arc.*archiv" in line.lower():
                arc_events.append(line.strip()[:120])
    
    if chapter_events:
        lines.append(f"Chapter archival events in log: {len(chapter_events)}")
        for e in chapter_events[:5]:
            lines.append(f"  - {e}")
    else:
        lines.append("No explicit chapter archival log lines found (checking events.json count as proxy).")
    lines.append("")
    if arc_events:
        lines.append(f"Arc archival events in log: {len(arc_events)}")
        for e in arc_events[:5]:
            lines.append(f"  - {e}")
    else:
        lines.append("No explicit arc archival log lines found.")
    lines.append("")
    
    # Chapter title check
    turn_dirs = sorted(RUN522.glob("turn_*"))
    if turn_dirs:
        last = turn_dirs[-1]
        snap = last / "state_snapshot"
        ev_path = snap / "events.json"
        if ev_path.exists():
            try:
                ev = json.loads(ev_path.read_text())
                chapters = ev.get("chapters", [])
                bad_titles = [c for c in chapters if "CHAPTER #" in str(c.get("title", ""))]
                lines.append(f"Chapters with placeholder titles: {len(bad_titles)}/{len(chapters)}")
                if chapters:
                    lines.append(f"  First: {chapters[0].get('title', '?')}")
                    lines.append(f"  Last: {chapters[-1].get('title', '?')}")
            except json.JSONDecodeError:
                pass
        arcs_path = last / "arcs.json"
        if arcs_path.exists():
            try:
                arcs = json.loads(arcs_path.read_text())
                if isinstance(arcs, dict):
                    arcs = arcs.get("entries", [])
                bad_arcs = [a for a in arcs if "ARC #" in str(a.get("title", ""))]
                lines.append(f"Arcs with placeholder titles: {len(bad_arcs)}/{len(arcs)}")
                if arcs:
                    lines.append(f"  First: {arcs[0].get('title', '?')}")
                    lines.append(f"  Last: {arcs[-1].get('title', '?')}")
            except json.JSONDecodeError:
                pass
    lines.append("")
    
    # Section 5: Data comparison vs s521
    lines.append("## 5. Data Comparison vs s521 (f42ea382)")
    lines.append("")
    ev521 = get_event_counts(RUN521)
    lines.append("| Entity | s521 (turn {}) | s522 (turn {}) |".format(ev521.get('turn', '?'), ev522.get('turn', '?')))
    lines.append("|--------|--------------|--------------|")
    for key in ["chapters", "scenes", "beats", "past", "arcs", "characters", "elements"]:
        v521 = ev521.get(key, "—")
        v522 = ev522.get(key, "—")
        lines.append(f"| {key} | {v521} | {v522} |")
    lines.append("")
    
    # Section 6: Parse errors
    lines.append("## 6. Parse Errors")
    lines.append("")
    parse_errors = 0
    with open(LOG522, "r", errors="replace") as f:
        for line in f:
            if "Failed to parse" in line or "Failed to extract" in line:
                parse_errors += 1
    lines.append(f"Total parse/extract failures: {parse_errors}")
    lines.append("")
    if parse_errors == 0:
        lines.append("No parse errors — slim schema is working cleanly.")
    lines.append("")
    
    # Write report
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to {OUT} ({len(lines)} lines)")

if __name__ == "__main__":
    main()