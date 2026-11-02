#!/usr/bin/env python3
"""Parse the 300-turn soak run log and produce a timing report."""

import re
import statistics
from pathlib import Path

LOG = Path("extensions/dayna_ss/tests/runs/logs/scifi_heist_gemma4_s200.log")
OUT = Path("extensions/dayna_ss/docs/tests/scifi_heist_c05349c6_timing.md")

# --- Regex patterns ---
TURN_START = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] --- turn (\d+)/(\d+) ---")
TURN_DONE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s+done: turn (\d+) in ([\d.]+)s")
LOCAL_CALL = re.compile(
    r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s+local-call (\S+)/generate: ttft=([\d.]+)s total=([\d.]+)s in=(\d+) out=(\d+) tok"
)
CLOUD_GUIDE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s+guide: cloud user turn in ([\d.]+)s")
CLOUD_REPLAN = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s+replan: cloud response in ([\d.]+)s \(guide calls=(\d+)\)")
CLOUD_AUDITOR = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s+auditor: (\d+) verdicts in ([\d.]+)s \((\d+) calls\)")
CLOUD_JUDGE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s+judge: calling cloud")
CLOUD_JUDGE_DONE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s+judge:.*in ([\d.]+)s")
CLOUD_FAILED = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s+(guide|judge|auditor):.*cloud call FAILED")


def parse_log():
    """Stream-parse the log and build per-turn data structures."""
    turns = {}  # turn_num -> dict
    current_turn = None
    total_lines = 0
    local_calls = []  # (turn, kind, ttft, total, in_tok, out_tok)
    cloud_calls = []  # (turn, kind, duration)
    turn_starts = {}  # turn_num -> timestamp string
    turn_dones = {}  # turn_num -> (timestamp, duration_s)
    failed_calls = []  # (turn, role, timestamp)

    with open(LOG, "r", errors="replace") as f:
        for line in f:
            total_lines += 1
            line_stripped = line.rstrip("\n")

            # Turn boundaries
            m = TURN_START.match(line_stripped)
            if m:
                current_turn = int(m.group(2))
                turn_starts[current_turn] = m.group(1)
                turns.setdefault(current_turn, {"local": [], "cloud": []})
                continue

            m = TURN_DONE.match(line_stripped)
            if m:
                turn_num = int(m.group(2))
                turn_dones[turn_num] = (m.group(1), float(m.group(3)))
                turns.setdefault(turn_num, {"local": [], "cloud": []})
                continue

            # Local LLM calls
            m = LOCAL_CALL.match(line_stripped)
            if m:
                kind = m.group(2)  # "reply" for writer, "?" for DSS
                ttft = float(m.group(3))
                total = float(m.group(4))
                in_tok = int(m.group(5))
                out_tok = int(m.group(6))
                local_calls.append((current_turn, kind, ttft, total, in_tok, out_tok))
                if current_turn is not None:
                    turns[current_turn]["local"].append((kind, total, in_tok, out_tok))
                continue

            # Cloud calls
            m = CLOUD_GUIDE.match(line_stripped)
            if m:
                dur = float(m.group(2))
                cloud_calls.append((current_turn, "guide", dur))
                if current_turn is not None:
                    turns[current_turn]["cloud"].append(("guide", dur))
                continue

            m = CLOUD_REPLAN.match(line_stripped)
            if m:
                dur = float(m.group(2))
                cloud_calls.append((current_turn, "replan", dur))
                if current_turn is not None:
                    turns[current_turn]["cloud"].append(("replan", dur))
                continue

            m = CLOUD_AUDITOR.match(line_stripped)
            if m:
                dur = float(m.group(3))
                cloud_calls.append((current_turn, "auditor", dur))
                if current_turn is not None:
                    turns[current_turn]["cloud"].append(("auditor", dur))
                continue

            m = CLOUD_FAILED.match(line_stripped)
            if m:
                failed_calls.append((current_turn, m.group(2), m.group(1)))
                continue

    return {
        "total_lines": total_lines,
        "turns": turns,
        "local_calls": local_calls,
        "cloud_calls": cloud_calls,
        "turn_starts": turn_starts,
        "turn_dones": turn_dones,
        "failed_calls": failed_calls,
    }


def percentile(data, p):
    """Compute p-th percentile (0-100) of a list."""
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def stats_block(values, label):
    """Produce a stats line: mean/median/p95."""
    if not values:
        return f"| {label} | 0 | — | — | — |"
    mean = statistics.mean(values)
    med = statistics.median(values)
    p95 = percentile(values, 95)
    return f"| {label} | {len(values)} | {mean:.1f}s | {med:.1f}s | {p95:.1f}s |"


def main():
    print(f"Parsing {LOG} ...")
    data = parse_log()
    print(f"  {data['total_lines']} lines parsed")
    print(f"  {len(data['turns'])} turns identified")
    print(f"  {len(data['local_calls'])} local LLM calls")
    print(f"  {len(data['cloud_calls'])} cloud LLM calls")
    print(f"  {len(data['failed_calls'])} failed cloud calls")

    # --- Build report ---
    lines = []
    lines.append("# Timing Analysis: scifi_heist_c05349c6 (300-turn soak)")
    lines.append("")
    lines.append(f"Log: `tests/runs/logs/scifi_heist_gemma4_s200.log` ({data['total_lines']:,} lines)")
    lines.append(f"Run dir: `tests/runs/sci_fi_heist__c05349c6`")
    lines.append("")

    # === Section 1: Overall summary ===
    lines.append("## 1. Overall Summary")
    lines.append("")
    done_durations = [d for _, d in data["turn_dones"].values()]
    total_run_s = sum(done_durations)
    total_run_min = total_run_s / 60.0
    lines.append(f"- **Turns completed:** {len(done_durations)}/300")
    lines.append(f"- **Total wall time (sum of turn durations):** {total_run_s:.0f}s ({total_run_min:.1f} min)")
    if data["turn_starts"] and data["turn_dones"]:
        first_start = min(data["turn_starts"].values())
        last_done = max(ts for ts, _ in data["turn_dones"].values())
        lines.append(f"- **First turn start:** {first_start}")
        lines.append(f"- **Last turn done:** {last_done}")
    lines.append(f"- **Local LLM calls:** {len(data['local_calls'])}")
    lines.append(f"- **Cloud LLM calls:** {len(data['cloud_calls'])}")
    lines.append(f"- **Failed cloud calls:** {len(data['failed_calls'])}")
    lines.append("")

    # === Section 2: Per-category timing ===
    lines.append("## 2. Per-Category LLM Call Timing")
    lines.append("")
    lines.append("| Category | Count | Mean | Median | P95 |")
    lines.append("|----------|-------|------|--------|-----|")

    # Local writer (reply)
    writer_totals = [t for _, kind, _, t, _, _ in data["local_calls"] if kind == "reply"]
    lines.append(stats_block(writer_totals, "Local writer (reply/generate)"))

    # Local DSS (non-reply local calls)
    dss_totals = [t for _, kind, _, t, _, _ in data["local_calls"] if kind != "reply"]
    lines.append(stats_block(dss_totals, "Local DSS/summarizer (?/generate)"))

    # All local
    all_local = [t for _, _, _, t, _, _ in data["local_calls"]]
    lines.append(stats_block(all_local, "All local calls"))

    # Cloud guide
    guide_durs = [d for _, k, d in data["cloud_calls"] if k == "guide"]
    lines.append(stats_block(guide_durs, "Cloud guide"))

    # Cloud replan
    replan_durs = [d for _, k, d in data["cloud_calls"] if k == "replan"]
    lines.append(stats_block(replan_durs, "Cloud replan"))

    # Cloud auditor
    auditor_durs = [d for _, k, d in data["cloud_calls"] if k == "auditor"]
    lines.append(stats_block(auditor_durs, "Cloud auditor"))

    lines.append("")

    # === Section 3: Per-turn breakdown ===
    lines.append("## 3. Per-Turn Timing")
    lines.append("")
    lines.append("| Turn | Wall Time | Local (s) | Cloud (s) | Local Calls | Cloud Calls |")
    lines.append("|------|-----------|-----------|-----------|-------------|-------------|")

    turn_rows = []
    for turn_num in sorted(data["turn_dones"].keys()):
        wall = data["turn_dones"][turn_num][1]
        tdata = data["turns"].get(turn_num, {"local": [], "cloud": []})
        local_s = sum(x[1] for x in tdata["local"])
        cloud_s = sum(x[1] for x in tdata["cloud"])
        n_local = len(tdata["local"])
        n_cloud = len(tdata["cloud"])
        turn_rows.append((turn_num, wall, local_s, cloud_s, n_local, n_cloud))

    for row in turn_rows:
        lines.append(f"| {row[0]} | {row[1]:.1f}s | {row[2]:.1f}s | {row[3]:.1f}s | {row[4]} | {row[5]} |")
    lines.append("")

    # === Section 4: LLM vs Overhead ratio ===
    lines.append("## 4. LLM Time vs Overhead")
    lines.append("")
    total_local = sum(all_local)
    total_cloud = sum(d for _, _, d in data["cloud_calls"])
    total_llm = total_local + total_cloud
    overhead = total_run_s - total_llm
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total wall time (turns) | {total_run_s:.0f}s |")
    lines.append(f"| Total local LLM time | {total_local:.0f}s ({100*total_local/total_run_s:.1f}%) |")
    lines.append(f"| Total cloud LLM time | {total_cloud:.0f}s ({100*total_cloud/total_run_s:.1f}%) |")
    lines.append(f"| Total LLM time | {total_llm:.0f}s ({100*total_llm/total_run_s:.1f}%) |")
    lines.append(f"| Overhead (non-LLM) | {overhead:.0f}s ({100*max(overhead,0)/total_run_s:.1f}%) |")
    lines.append("")
    lines.append("Overhead includes: context retrieval/building, DSS tool execution, prompt assembly,")
    lines.append("streaming/rendering, disk I/O, and inter-step coordination.")
    lines.append("")

    # === Section 5: Slowest turns ===
    lines.append("## 5. Slowest Turns (Top 20)")
    lines.append("")
    lines.append("| Turn | Wall Time | Local (s) | Cloud (s) | Overhead (s) |")
    lines.append("|------|-----------|-----------|-----------|--------------|")
    sorted_turns = sorted(turn_rows, key=lambda r: r[1], reverse=True)[:20]
    for row in sorted_turns:
        ovh = max(row[1] - row[2] - row[3], 0)
        lines.append(f"| {row[0]} | {row[1]:.1f}s | {row[2]:.1f}s | {row[3]:.1f}s | {ovh:.1f}s |")
    lines.append("")

    # === Section 6: Local call detail ===
    lines.append("## 6. Local Model Call Detail")
    lines.append("")
    lines.append("### Writer (reply/generate)")
    lines.append("")
    lines.append("| Stat | Value |")
    lines.append("|------|-------|")
    if writer_totals:
        lines.append(f"| Count | {len(writer_totals)} |")
        lines.append(f"| Mean total | {statistics.mean(writer_totals):.2f}s |")
        lines.append(f"| Median total | {statistics.median(writer_totals):.2f}s |")
        lines.append(f"| P95 total | {percentile(writer_totals, 95):.2f}s |")
        lines.append(f"| Max total | {max(writer_totals):.2f}s |")
        lines.append(f"| Mean TTFT | {statistics.mean(t for _, k, t, _, _, _ in data['local_calls'] if k == 'reply'):.2f}s |")
        lines.append(f"| Mean input tokens | {statistics.mean(i for _, k, _, _, i, _ in data['local_calls'] if k == 'reply'):.0f} |")
    lines.append("")

    lines.append("### DSS/Summarizer (?/generate)")
    lines.append("")
    lines.append("| Stat | Value |")
    lines.append("|------|-------|")
    dss_calls = [(t, tt, i, o) for _, k, tt, t, i, o in data["local_calls"] if k != "reply"]
    if dss_calls:
        lines.append(f"| Count | {len(dss_calls)} |")
        lines.append(f"| Mean total | {statistics.mean(t for _, _, t, _, _, _ in data['local_calls'] if _ != 'reply'):.2f}s |")
        lines.append(f"| Median total | {statistics.median(d[0] for d in dss_calls):.2f}s |")
        lines.append(f"| P95 total | {percentile([d[0] for d in dss_calls], 95):.2f}s |")
        lines.append(f"| Max total | {max(d[0] for d in dss_calls):.2f}s |")
        lines.append(f"| Mean input tokens | {statistics.mean(d[2] for d in dss_calls):.0f} |")
    lines.append("")

    # === Section 7: Failed cloud calls ===
    if data["failed_calls"]:
        lines.append("## 7. Failed Cloud Calls")
        lines.append("")
        lines.append(f"Total failed: {len(data['failed_calls'])}")
        lines.append("")
        # Group by role
        from collections import Counter
        role_counts = Counter(r for _, r, _ in data["failed_calls"])
        lines.append("| Role | Count |")
        lines.append("|------|-------|")
        for role, cnt in role_counts.most_common():
            lines.append(f"| {role} | {cnt} |")
        lines.append("")
        lines.append("First 10 failures:")
        lines.append("")
        lines.append("| Turn | Role | Timestamp |")
        lines.append("|------|------|-----------|")
        for turn, role, ts in data["failed_calls"][:10]:
            lines.append(f"| {turn} | {role} | {ts} |")
        lines.append("")

    # === Section 8: Token throughput ===
    lines.append("## 8. Token Throughput (Local Model)")
    lines.append("")
    # Compute effective tok/s for writer calls
    writer_in = [i for _, k, _, _, i, _ in data["local_calls"] if k == "reply"]
    writer_out = [o for _, k, _, _, _, o in data["local_calls"] if k == "reply"]
    dss_in = [i for _, k, _, _, i, _ in data["local_calls"] if k != "reply"]
    dss_out = [o for _, k, _, _, _, o in data["local_calls"] if k != "reply"]
    lines.append(f"| Category | Total input tokens | Total output tokens | Mean in/call | Mean out/call |")
    lines.append(f"|----------|-------------------|--------------------|--------------|---------------|")
    if writer_in:
        lines.append(f"| Writer | {sum(writer_in):,} | {sum(writer_out):,} | {sum(writer_in)/len(writer_in):.0f} | {sum(writer_out)/len(writer_out):.0f} |")
    if dss_in:
        lines.append(f"| DSS | {sum(dss_in):,} | {sum(dss_out):,} | {sum(dss_in)/len(dss_in):.0f} | {sum(dss_out)/len(dss_out):.0f} |")
    lines.append("")
    lines.append("> Note: `out=0` in the log likely means output token count wasn't captured at log time")
    lines.append("> (the call was still streaming or the counter wasn't flushed). Input tokens are reliable.")
    lines.append("")

    # Write report
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to {OUT} ({len(lines)} lines)")


if __name__ == "__main__":
    main()