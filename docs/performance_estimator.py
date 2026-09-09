#!/usr/bin/env python3
"""
performance_estimator.py — per-turn latency model for DAYNA Story Summarizer (DSS)
vs Smart-Memory (SM).

Estimates, for a single chat turn under a given local-model speed profile:

  * MAIN-GEN PREFILL   — tokens the memory system injects into the story prompt,
                         divided by the model's prefill speed.
  * MAIN-GEN DECODE    — reply length / decode speed (memory systems barely
                         affect this; included so totals are complete).
  * BACKGROUND WORK    — the memory system's own LLM calls (prefill + decode),
                         amortized per turn. This is where SM and DSS differ most.

All token counts are configurable estimates grounded in source constants:
  - SM: budgets/cadences read from settings.js defaults + constants.js
        (chars/4 token heuristic everywhere: estimateTokens = len/4).
  - DSS: injected block sizes estimated from format_templates.json rendering,
        engine call counts derived from subjects_schema_sceneagg.json triggers.
Measured anchors used for sanity-checking are cited near the presets.

Usage:
  python3 performance_estimator.py                     # all scenarios, default speed
  python3 performance_estimator.py --speed desktop-35b-moe
  python3 performance_estimator.py --speed custom --prefill 400 --decode 30
  python3 performance_estimator.py --scenario dss-balanced,dss-light --json
  python3 performance_estimator.py --dss-cast 10 --sm-extract-every 1  # overrides
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, replace

# ─────────────────────────────────────────────────────────────────────────────
# Speed profiles (tokens/sec). Replace with numbers measured on your hardware:
#   prefill ≈ prompt-processing tks/s, decode ≈ generation tks/s.
# Anchors from the DSS project: Qwen3.6-35B-A3B (MoE) on a local server measured
# ~23 serial engine calls in ~115 s of mostly-prefill work (~450–600 tk/s effective
# prefill incl. overhead); decode on that class of card runs ~40–60 tk/s.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Speed:
    name: str
    prefill_tps: float
    decode_tps: float
    note: str = ""


SPEED_PRESETS: dict[str, Speed] = {
    "laptop-8b":       Speed("laptop-8b",        80,  12, "8B dense, gaming laptop / 8 GB VRAM"),
    "desktop-35b-moe": Speed("desktop-35b-moe", 500,  50, "35B-A3B MoE class, RTX 4090-class"),
    "desktop-70b":     Speed("desktop-70b",     180,   9, "70B dense q4, 2×16 GB or 24 GB+"),
    "cloud-fast":      Speed("cloud-fast",     2000,  90, "hosted endpoint, high throughput"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Shared result type
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TurnEstimate:
    scenario: str
    system: str
    injected_tokens: int          # into the main story prompt (prefill cost/turn)
    bg_calls_per_turn: float      # background LLM calls, amortized
    bg_in_per_turn: float         # background prefill tokens, amortized
    bg_out_per_turn: float        # background decode tokens, amortized
    main_reply_out: int           # decode tokens of the visible reply
    prefix_hit_frac: float        # fraction of engine-call prefill served from cache
    speed: Speed

    def seconds(self) -> dict[str, float]:
        s = self.speed
        cached = self.bg_in_per_turn * self.prefix_hit_frac
        uncached_in = max(self.bg_in_per_turn - cached, 0.0)
        # cached prefill typically runs ~5-10x faster; model as 8x
        bg_prefill_s = uncached_in / s.prefill_tps + cached / (s.prefill_tps * 8)
        return {
            "main_prefill_s": self.injected_tokens / s.prefill_tps,
            "main_decode_s": self.main_reply_out / s.decode_tps,
            "bg_s": bg_prefill_s + self.bg_out_per_turn / s.decode_tps,
            "total_s": (
                self.injected_tokens / s.prefill_tps
                + self.main_reply_out / s.decode_tps
                + bg_prefill_s
                + self.bg_out_per_turn / s.decode_tps
            ),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Smart-Memory scenarios
# Constants from settings.js defaults (extract_every=3, continuity_auto_check on
# Profile B, budgets 2000+800+400+500+700+400+250+300+200) and prompts.js
# template sizes (LT extract ≈853 tok static, session ≈769, arcs ≈559,
# relationship delta ≈538, profiles ≈386, continuity ≈140).
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SMConfig:
    label: str
    note: str
    profile_b: bool = True             # hosted/main-API profile (auto-continuity etc.)
    extract_every: int = 3             # min(session_extract_every, longterm_extract_every)
    injected_steady: int = 4200        # realistic fill of the 3750 core + summary partial
    injected_max: int = 6250           # every tier pinned at budget incl. triggered dup
    lt_window_msgs: int = 20
    session_window_msgs: int = 40
    arc_window_msgs: int = 100
    msg_tokens: float = 45.0           # avg tokens per chat message
    continuity_on: bool | None = None  # None → follow profile_b
    scene_break_every: float = 12.0    # heuristic breaks per N turns
    compaction_every: float = 60.0     # threshold-triggered; rough steady-state period
    consolidation_extra_per_pass: float = 1.5   # probabilistic trigger-gen + confirms + merges
    reply_out: int = 350               # typical visible reply tokens


def sm_estimate(cfg: SMConfig, speed: Speed, inj_override: int | None = None) -> TurnEstimate:
    cont_on = cfg.continuity_on if cfg.continuity_on is not None else cfg.profile_b

    pass_calls = [
        ("session_extract",  769 + cfg.session_window_msgs * cfg.msg_tokens + 300, 220),
        ("lt_extract",       853 + cfg.lt_window_msgs * cfg.msg_tokens + 750,   260),
        ("relationship",     538 + cfg.lt_window_msgs * cfg.msg_tokens + 250,   120),
        ("arcs_extract",     559 + cfg.arc_window_msgs * cfg.msg_tokens + 300,  160),
        ("profiles_regen",   386 + 900,                                            240),
    ]
    # continuity fires EVERY turn (not on the extraction cadence)
    cont_calls, cont_in, cont_out = (1.0, 2740.0, 130.0) if cont_on else (0.0, 0.0, 0.0)

    pass_only = pass_calls
    pass_in = sum(i for _, i, _ in pass_only)
    pass_out = sum(o for _, _, o in pass_only)

    cadence = cfg.extract_every
    amortized_in = (pass_in + cfg.consolidation_extra_per_pass * 900) / cadence
    amortized_out = (pass_out + cfg.consolidation_extra_per_pass * 90) / cadence
    amortized_calls = (len(pass_only) + cfg.consolidation_extra_per_pass) / cadence

    # scene-break extras: epistemic (≈927 tok template + buffer) + scene summary
    sb_rate = 1.0 / cfg.scene_break_every
    amortized_in += sb_rate * (927 + 1500 + 130)
    amortized_out += sb_rate * (400 * 0.5 + 200 * 0.7)
    amortized_calls += sb_rate * 2

    # compaction (progressive): big input, big output, rare
    comp_rate = 1.0 / cfg.compaction_every
    amortized_in += comp_rate * (625 + 2000 + 1500)
    amortized_out += comp_rate * 2000 * 0.8
    amortized_calls += comp_rate

    injected = inj_override if inj_override is not None else cfg.injected_steady
    return TurnEstimate(
        scenario=cfg.label, system="Smart-Memory",
        injected_tokens=injected,
        bg_calls_per_turn=amortized_calls + cont_calls,
        bg_in_per_turn=amortized_in + cont_in,
        bg_out_per_turn=amortized_out + cont_out,
        main_reply_out=cfg.reply_out,
        prefix_hit_frac=0.0,   # ST slots churn between tiers each turn → low cache reuse
        speed=speed,
    )


SM_SCENARIOS: dict[str, SMConfig] = {
    "sm-default-b": SMConfig(
        "sm-default-b",
        "SM defaults, Profile B (main API): auto-continuity ON, extraction every 3 msgs"),
    "sm-default-a": SMConfig(
        "sm-default-a",
        "SM defaults, Profile A (Ollama/WebLLM): no auto-continuity, cap 2 new/type, sequential",
        profile_b=False, continuity_on=False),
    "sm-minimal": SMConfig(
        "sm-minimal",
        "SM trimmed: no continuity, extraction every 5, fewer consolidation extras",
        profile_b=False, continuity_on=False, extract_every=5,
        consolidation_extra_per_pass=0.5, injected_steady=2800),
    "sm-aggressive": SMConfig(
        "sm-aggressive",
        "SM high-frequency: extraction every message, AI scene detection, auto-repair",
        extract_every=1, scene_break_every=6, consolidation_extra_per_pass=3.0,
        injected_steady=5500),
}


# ─────────────────────────────────────────────────────────────────────────────
# DSS scenarios
# Engine call counts derived from subjects_schema_sceneagg.json triggers.
# Measured anchor (project memory #67): 2-turn L2 smoke — serial 46 calls/230 s,
# 4 workers 26 calls/128 s ⇒ ~22–23 engine-side calls/turn balanced config.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DSSConfig:
    label: str
    note: str
    cast: int = 6                # characters with entries
    groups_n: int = 3
    elements_n: int = 8
    events_map_gate: bool = True
    current_scene_skip: bool = False      # False = two-step query+update; True = single call
    two_step_branch_query: bool = False   # False = skip_query single-call variant
    select_entries_on: bool = True        # whitelist selection when map ≥ 4 entries
    gates_on: bool = True
    add_new_discovery: bool = True
    avg_new_entries: float = 0.4          # discovered per turn across maps
    importance_refresh_when_gate_fail: bool = True
    gate_fail_rate: float = 0.25          # share of turns a map gate reports "nothing changed"
    new_scene_every: float = 14.0         # turns per scene transition
    director_call: bool = True
    msg_summary: bool = True
    scene_probe_llm: bool = True
    parallel_workers: int = 0             # 0 = serial (wall-clock unchanged here; counts same)
    reply_out: int = 380
    # injected blocks (rendered tokens, adjustable to your world size)
    inj_general_info: int = 500
    inj_current_scene: int = 250
    inj_arcs_chapters: int = 350
    inj_lists: int = 150                  # character/group/element one-line rosters
    inj_groups: int = 250
    inj_elements: int = 250
    inj_events: int = 350
    inj_characters: int = 3200            # full profiles × cast above threshold
    inj_lines_messages: int = 300
    inj_director: int = 180               # instruction block in the final prompt
    engine_ctx_restate: int = 600         # avg per-call branch context re-injection


def dss_engine_calls(cfg: DSSConfig) -> tuple[float, float]:
    """Return (calls, input_tokens) for one turn's subject-engine work."""
    calls = 0.0
    tin = 0.0
    STATIC_TOK = 850            # rendered template body shared by engine calls
    TAIL_TOK = 350              # recent-window + instructions tail (uncached side)
    restate = cfg.engine_ctx_restate

    def entry_calls(n: float, gate: bool, two_step: bool, select: bool, sel_min: int = 4) -> tuple[float, float]:
        c = 0.0
        t = 0.0
        if gate:
            c += 1
            t += STATIC_TOK + TAIL_TOK
        if select and n >= sel_min:
            c += 1
            t += STATIC_TOK + TAIL_TOK
        per_entry = 2 if two_step else 1
        c += n * per_entry
        t += n * per_entry * (STATIC_TOK + TAIL_TOK + restate)
        return c, t

    # general_info: ALWAYS, skip_query single update call
    calls += 1
    tin += STATIC_TOK + TAIL_TOK + restate
    # current_scene: ALWAYS — two-step query+update, or skip_query single call
    cs_calls = 1 if cfg.current_scene_skip else 2
    calls += cs_calls
    tin += cs_calls * (STATIC_TOK + TAIL_TOK + restate)

    # characters
    c, t = entry_calls(cfg.cast, cfg.gates_on, cfg.two_step_branch_query, cfg.select_entries_on)
    fail_extra = cfg.importance_refresh_when_gate_fail * cfg.gate_fail_rate * cfg.cast
    calls += c + fail_extra
    tin += t + fail_extra * (STATIC_TOK * 0.7 + TAIL_TOK)

    # groups / elements: gate + discovery (+ pops below)
    for n in (cfg.groups_n, cfg.elements_n):
        if cfg.gates_on:
            calls += 1
            tin += STATIC_TOK + TAIL_TOK
    # events: gate only (+ conditional refresh on gate-fail)
    if cfg.events_map_gate and cfg.gates_on:
        calls += 1 + (0.5 if cfg.importance_refresh_when_gate_fail else 0)
        tin += 1.5 * (STATIC_TOK + TAIL_TOK)

    # ADD_NEW discovery queries (characters/groups/elements)
    if cfg.add_new_discovery:
        calls += 3
        tin += 3 * (STATIC_TOK + TAIL_TOK + 250)
        pops = cfg.avg_new_entries * 1.15          # retries included
        calls += pops
        tin += pops * (STATIC_TOK + TAIL_TOK + 400)

    # arcs/chapters archival checks fire on new-scene turns only
    ns_rate = 1.0 / cfg.new_scene_every
    calls += ns_rate * 2
    tin += ns_rate * 2 * (STATIC_TOK + TAIL_TOK + 500)

    return calls, tin


def dss_estimate(cfg: DSSConfig, speed: Speed, inj_override: int | None = None) -> TurnEstimate:
    calls, tin = dss_engine_calls(cfg)
    if cfg.director_call:
        calls += 1
        tin += 1100 + 900          # recap + requirements + last window
    if cfg.msg_summary:
        calls += 1
        tin += 700                 # last exchange pair
    probe_rate = (1.0 / cfg.new_scene_every) if cfg.scene_probe_llm else 0.0
    calls += probe_rate
    tin += probe_rate * (1600 + 900)

    injected = inj_override if inj_override is not None else (
        cfg.inj_general_info + cfg.inj_current_scene + cfg.inj_arcs_chapters
        + cfg.inj_lists + cfg.inj_groups + cfg.inj_elements + cfg.inj_events
        + cfg.inj_characters + cfg.inj_lines_messages + cfg.inj_director
    )
    return TurnEstimate(
        scenario=cfg.label, system="DSS",
        injected_tokens=injected,
        bg_calls_per_turn=calls,
        bg_in_per_turn=tin,
        bg_out_per_turn=calls * 210,     # JSON updates are short; ~200-token outputs dominate
        main_reply_out=cfg.reply_out,
        prefix_hit_frac=0.65,            # static-first templates share prefixes across per-entry calls
        speed=speed,
    )


DSS_SCENARIOS: dict[str, DSSConfig] = {
    "dss-light": DSSConfig(
        "dss-light",
        "DSS lean schema: skip_query everywhere, gates only, discovery off-season, small cast",
        cast=4, groups_n=2, elements_n=4, add_new_discovery=False, select_entries_on=False,
        avg_new_entries=0.1, engine_ctx_restate=400),
    "dss-balanced": DSSConfig(
        "dss-balanced",
        "DSS shipped defaults (sceneagg): gates + selection + skip_query entries, cast 6"),
    "dss-thorough": DSSConfig(
        "dss-thorough",
        "DSS heavy: two-step branch queries per entry, discovery always on, big cast/world",
        cast=10, groups_n=4, elements_n=12, two_step_branch_query=True,
        avg_new_entries=1.0, engine_ctx_restate=800, inj_characters=5200),
    "dss-currentscene-skip": DSSConfig(
        "dss-currentscene-skip",
        "Balanced but current_scene switched to skip_query (one call instead of two)",
        current_scene_skip=True),
}


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────


def fmt_row(est: TurnEstimate) -> str:
    t = est.seconds()
    return (
        f"{est.scenario:<24} {est.injected_tokens:>6,} "
        f"{est.bg_calls_per_turn:>6.1f} {est.bg_in_per_turn:>8,.0f} "
        f"{t['main_prefill_s']:>6.1f} {t['main_decode_s']:>6.1f} "
        f"{t['bg_s']:>7.1f} {t['total_s']:>7.1f}"
    )


HEADER = (
    f"{'scenario':<24} {'inj tok':>7} {'calls':>6} {'bg in':>8} "
    f"{'pre s':>6} {'dec s':>6} {'bg s':>7} {'TOTAL':>7}"
)


def run(speed: Speed, scenarios: list[str], overrides: dict) -> list[TurnEstimate]:
    results: list[TurnEstimate] = []
    for key in scenarios:
        if key.startswith("sm-"):
            base = SM_SCENARIOS[key]
            cfg = replace(
                base,
                extract_every=overrides.get("sm_extract_every", base.extract_every),
                injected_steady=overrides.get("injected", base.injected_steady),
            )
            results.append(sm_estimate(cfg, speed))
        elif key.startswith("dss-"):
            base = DSS_SCENARIOS[key]
            cfg = replace(
                base,
                cast=overrides.get("dss_cast", base.cast),
                elements_n=overrides.get("dss_elements", base.elements_n),
            )
            est = dss_estimate(cfg, speed, inj_override=overrides.get("injected"))
            if "injected" in overrides:
                est = replace(est, injected_tokens=overrides["injected"])
            if "dss_prefix_hit" in overrides:
                est = replace(est, prefix_hit_frac=max(min(overrides["dss_prefix_hit"], 1.0), 0.0))
            results.append(est)
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speed", default="desktop-35b-moe", choices=[*SPEED_PRESETS, "custom"])
    ap.add_argument("--prefill", type=float, help="custom prefill tks/s (with --speed custom)")
    ap.add_argument("--decode", type=float, help="custom decode tks/s (with --speed custom)")
    ap.add_argument("--scenario", default="", help="comma list (default: all)")
    ap.add_argument("--injected", type=int, help="override injected-tokens for every scenario")
    ap.add_argument("--sm-extract-every", type=int)
    ap.add_argument("--dss-cast", type=int)
    ap.add_argument("--dss-elements", type=int)
    ap.add_argument("--dss-prefix-hit", type=float,
                    help="override DSS engine-call prefix-cache hit fraction (default 0.65; "
                         "set 0 when running --disable-prefix-caching or 4+ subject workers)")
    ap.add_argument("--reply-out", type=int, help="override visible-reply tokens")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.speed == "custom":
        if not args.prefill or not args.decode:
            ap.error("--speed custom requires --prefill and --decode")
        speed = Speed("custom", args.prefill, args.decode)
    else:
        speed = SPEED_PRESETS[args.speed]

    keys = list(SM_SCENARIOS) + list(DSS_SCENARIOS)
    if args.scenario:
        keys = [k.strip() for k in args.scenario.split(",")]
        for k in keys:
            if k not in SM_SCENARIOS and k not in DSS_SCENARIOS:
                ap.error(f"unknown scenario: {k}")

    overrides: dict = {}
    if args.injected is not None:
        overrides["injected"] = args.injected
    if args.sm_extract_every:
        overrides["sm_extract_every"] = args.sm_extract_every
    if args.dss_cast is not None:
        overrides["dss_cast"] = args.dss_cast
    if args.dss_elements is not None:
        overrides["dss_elements"] = args.dss_elements
    if args.dss_prefix_hit is not None:
        overrides["dss_prefix_hit"] = args.dss_prefix_hit

    results = run(speed, keys, overrides)
    if args.reply_out:
        results = [replace(r, main_reply_out=args.reply_out) for r in results]

    if args.json:
        print(json.dumps([
            {"scenario": r.scenario, "system": r.system, **r.seconds(),
             "injected": r.injected_tokens, "calls": round(r.bg_calls_per_turn, 2)}
            for r in results
        ], indent=2))
        return 0

    print(f"\nSpeed profile: {speed.name} — prefill {speed.prefill_tps:.0f} tk/s, "
          f"decode {speed.decode_tps:.0f} tk/s  ({speed.note})\n")
    print(HEADER)
    print("-" * len(HEADER))
    for r in sorted(results, key=lambda x: (x.system, x.seconds()["total_s"])):
        print(fmt_row(r))
    print()
    print("columns: inj tok = memory blocks injected into the story prompt ·")
    print("         calls/bg in = background LLM calls & their prefill tokens, per-turn average ·")
    print("         pre s / dec s = main-generation prefill & decode · bg s = background work ·")
    print("         TOTAL ≈ wall-clock added by the memory stack per turn (serial assumption)")
    print("\nScenario notes:")
    for k in keys:
        src = SM_SCENARIOS.get(k) or DSS_SCENARIOS.get(k)
        print(f"  {k:<24} {src.note}")
    print("\nAnchors: DSS serial smoke measured ~46 calls/230 s over 2 turns (balanced);")
    print("         SM docs claim 'several seconds per Ollama call', no published benchmarks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
