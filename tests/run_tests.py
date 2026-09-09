"""CLI for the dayna_ss golden-fixture test harness.

Usage:
    python tests/run_tests.py                 # run all scenarios + director test
    python tests/run_tests.py -s <name>       # run one golden scenario
    python tests/run_tests.py --update-golden # regenerate golden files
    python tests/run_tests.py --live          # also run live tests (JSON-compliance
                                              # sweep, director on/off, live soak;
                                              # needs DSS_BENCH_BASE_URL)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from director_test import run_director_test
from boundary_persistence_test import run_boundary_persistence_test
from cadence_profiles_test import run_cadence_profile_test
from force_boundary_test import run_force_boundary_test
from harness import FIXTURES_DIR, Scenario
from scene_window_test import run_scene_window_test
from soak_test import run_soak


def list_scenarios() -> list[str]:
    return sorted(p.name for p in FIXTURES_DIR.iterdir() if (p / "scenario.json").exists())


def main() -> int:
    parser = argparse.ArgumentParser(description="dayna_ss golden-fixture harness")
    parser.add_argument("-s", "--scenario", help="run only this scenario")
    parser.add_argument("-u", "--update-golden", action="store_true", help="regenerate golden files")
    parser.add_argument("--live", action="store_true", help="also run the live JSON-compliance benchmark (needs DSS_BENCH_BASE_URL)")
    parser.add_argument("--long-soak", metavar="SPREADSHEET", default=None,
                        help="run the long-horizon soak against a spreadsheet in tests/spreadsheets/ "
                             "(opt-in; needs DSS_BENCH_BASE_URL + cloud key)")
    parser.add_argument("--smoke", type=int, default=None,
                        help="limit the long-horizon soak to N turns (fast validation)")
    parser.add_argument("--level", type=int, default=None, choices=[1, 2],
                        help="long-horizon soak level: 1=fast per-subject, 2=full production path")
    parser.add_argument("--resume", metavar="RUNDIR", default=None,
                        help="resume a prior long-horizon soak run dir")
    parser.add_argument("--plan-every", type=int, default=None,
                        help="guide live-plan replan cadence (0 disables the live plan)")
    parser.add_argument("--audit-every", type=int, default=None,
                        help="long-horizon soak DSS save-beat audit cadence (0 = beats only)")
    parser.add_argument("--abort-after", type=int, default=None,
                        help="long-horizon soak early-stop threshold for extreme DSS failures (0 disables)")
    parser.add_argument("--guide-max-attempts", type=int, default=None,
                        help="long-horizon soak cloud guide attempts per turn before giving up")
    parser.add_argument("--cloud-empty-retries", type=int, default=None,
                        help="long-horizon soak max re-rolls per cloud call on empty content")
    parser.add_argument("--cloud-empty-cooldown", type=float, default=None,
                        help="long-horizon soak cooldown (seconds) before each empty-content re-roll")
    parser.add_argument("--overview-every", type=int, default=None,
                        help="long-horizon soak rolling-overview cadence (0 disables)")
    parser.add_argument("--schema-type", type=int, default=None, choices=[1, 2, 3],
                        help="long-horizon soak subjects schema type (1=incremental, 2=scene-aggregation, 3=hybrid)")
    parser.add_argument("--synthetic-chat", action="store_true",
                        help="long-horizon soak isolation mode: cloud writes both sides, DSS runs only DataSummarizer")
    parser.add_argument("--judge-model", default=None, help="cloud judge model")
    parser.add_argument("--auditor-model", default=None, help="cloud auditor model")
    parser.add_argument("--max-update-history", type=int, default=None,
                        help="max engine internal-history exchanges per DataSummarizer call (0 = unlimited)")
    parser.add_argument("--message-mode", default=None, choices=["enumerated", "rolling"],
                        help="long-horizon soak Mode B message mode (rolling = raw pairs, no enumeration)")
    parser.add_argument("--retrieval-placement", default=None, choices=["prompt_start", "system", "inline"],
                        help="long-horizon soak Mode B retrieval placement (system = shared context block)")
    parser.add_argument("--rolling-window", type=int, default=None,
                        help="long-horizon soak Mode B recent-dialogue window size")
    parser.add_argument("--rolling-summaries", type=int, default=None,
                        help="long-horizon soak Mode B sticky-roll message summaries (0 = off)")
    parser.add_argument("--restate-map-context", default=None, choices=["auto", "always", "never"],
                        help="long-horizon soak per-entry whole-subject context re-statement gating")
    parser.add_argument("--restate-map-threshold", type=int, default=None,
                        help="long-horizon soak re-statement size gate (chars, auto mode)")
    parser.add_argument("--force-chapter-turn", type=int, default=None,
                        help="long-horizon soak: stage a forced chapter boundary at turn N (0 = off)")
    parser.add_argument("--cadence-profile", choices=["compressed", "campaign"], default=None,
                        help="long-horizon soak unit-cadence overlay (compressed/campaign)")
    args = parser.parse_args()

    names = [args.scenario] if args.scenario else list_scenarios()
    if not names:
        print("No scenarios found in fixtures/")
        return 1

    failures = 0
    for name in names:
        scenario = Scenario(name, FIXTURES_DIR / name)
        if scenario.meta.get("kind") == "soak":
            if args.scenario and not args.update_golden:
                soak_ok, soak_msgs = run_soak(name)
                for m in soak_msgs:
                    print("  " + m)
                if not soak_ok:
                    failures += 1
            continue  # soak scenarios run below for the full suite
        out_dir, model = scenario.run()
        if args.update_golden:
            scenario.write_golden(out_dir)
            print(f"[{name}] golden regenerated ({len(model.calls)} LLM calls scripted)")
            continue

        missing, mismatches = scenario.compare(out_dir)
        if not missing and not mismatches:
            print(f"[{name}] PASS ({len(model.calls)} LLM calls scripted)")
        else:
            failures += 1
            print(f"[{name}] FAIL")
            for m in missing:
                print(f"  MISSING: {m}")
            for m in mismatches:
                print(f"  MISMATCH: {m}")

    if not args.scenario and not args.update_golden:
        director_ok, director_msgs = run_director_test()
        for m in director_msgs:
            print("  " + m)
        if not director_ok:
            failures += 1
            print("[director_pass_on_off] FAIL")
        else:
            print("[director_pass_on_off] PASS")

        scene_ok, scene_msgs = run_scene_window_test()
        for m in scene_msgs:
            print("  " + m)
        if not scene_ok:
            failures += 1
            print("[scene_bounded_window] FAIL")
        else:
            print("[scene_bounded_window] PASS")

        force_ok, force_msgs = run_force_boundary_test()
        for m in force_msgs:
            print("  " + m)
        if not force_ok:
            failures += 1
            print("[forced_unit_boundaries] FAIL")
        else:
            print("[forced_unit_boundaries] PASS")

        cadence_ok, cadence_msgs = run_cadence_profile_test()
        for m in cadence_msgs:
            print("  " + m)
        if not cadence_ok:
            failures += 1
            print("[cadence_profiles] FAIL")
        else:
            print("[cadence_profiles] PASS")

        persist_ok, persist_msgs = run_boundary_persistence_test()
        for m in persist_msgs:
            print("  " + m)
        if not persist_ok:
            failures += 1
            print("[boundary_persistence] FAIL")
        else:
            print("[boundary_persistence] PASS")

        from chapters_render_test import run_chapters_render_test
        render_ok, render_msgs = run_chapters_render_test()
        for m in render_msgs:
            print("  " + m)
        if not render_ok:
            failures += 1
            print("[chapters_render] FAIL")
        else:
            print("[chapters_render] PASS")

        from canon_synopsis_test import run_canon_synopsis_test
        canon_ok, canon_msgs = run_canon_synopsis_test()
        for m in canon_msgs:
            print("  " + m)
        if not canon_ok:
            failures += 1
            print("[canon_synopsis] FAIL")
        else:
            print("[canon_synopsis] PASS")

        from hygiene_test import run_hygiene_test
        hyg_ok, hyg_msgs = run_hygiene_test()
        for m in hyg_msgs:
            print("  " + m)
        if not hyg_ok:
            failures += 1
            print("[hygiene] FAIL")
        else:
            print("[hygiene] PASS")

        soak_ok, soak_msgs = run_soak()
        for m in soak_msgs:
            print("  " + m)
        if not soak_ok:
            failures += 1

        if args.live:
            from live_benchmark import run_live_benchmark
            from live_director import run_live_director
            from live_soak import run_live_soak
            code = run_live_benchmark()
            if code != 0:
                failures += 1
            skip, dir_msgs = run_live_director()
            for m in dir_msgs:
                print("  " + m)
            if skip:
                failures += 1
            skip, soak_msgs = run_live_soak()
            for m in soak_msgs:
                print("  " + m)
            if skip:
                failures += 1

    if args.update_golden:
        return 0

    if args.long_soak:
        import subprocess
        spreadsheet = args.long_soak
        if not spreadsheet.endswith(".json"):
            spreadsheet = f"spreadsheets/{spreadsheet}.json"
        cmd = [sys.executable, str(Path(__file__).parent / "long_horizon_soak.py"),
               "--spreadsheet", spreadsheet]
        if getattr(args, "smoke", None):
            cmd += ["--smoke", str(args.smoke)]
        if getattr(args, "level", None):
            cmd += ["--level", str(args.level)]
        if getattr(args, "resume", None):
            cmd += ["--resume", args.resume]
        if getattr(args, "plan_every", None) is not None:
            cmd += ["--plan-every", str(args.plan_every)]
        if getattr(args, "audit_every", None) is not None:
            cmd += ["--audit-every", str(args.audit_every)]
        if getattr(args, "abort_after", None) is not None:
            cmd += ["--abort-after", str(args.abort_after)]
        if getattr(args, "guide_max_attempts", None) is not None:
            cmd += ["--guide-max-attempts", str(args.guide_max_attempts)]
        if getattr(args, "cloud_empty_retries", None) is not None:
            cmd += ["--cloud-empty-retries", str(args.cloud_empty_retries)]
        if getattr(args, "cloud_empty_cooldown", None) is not None:
            cmd += ["--cloud-empty-cooldown", str(args.cloud_empty_cooldown)]
        if getattr(args, "overview_every", None) is not None:
            cmd += ["--overview-every", str(args.overview_every)]
        if getattr(args, "schema_type", None) is not None:
            cmd += ["--schema-type", str(args.schema_type)]
        if getattr(args, "synthetic_chat", False):
            cmd += ["--synthetic-chat"]
        if getattr(args, "judge_model", None):
            cmd += ["--judge-model", args.judge_model]
        if getattr(args, "auditor_model", None):
            cmd += ["--auditor-model", args.auditor_model]
        if getattr(args, "max_update_history", None) is not None:
            cmd += ["--max-update-history", str(args.max_update_history)]
        if getattr(args, "message_mode", None):
            cmd += ["--message-mode", args.message_mode]
        if getattr(args, "retrieval_placement", None):
            cmd += ["--retrieval-placement", args.retrieval_placement]
        if getattr(args, "rolling_window", None) is not None:
            cmd += ["--rolling-window", str(args.rolling_window)]
        if getattr(args, "rolling_summaries", None) is not None:
            cmd += ["--rolling-summaries", str(args.rolling_summaries)]
        if getattr(args, "restate_map_context", None):
            cmd += ["--restate-map-context", args.restate_map_context]
        if getattr(args, "restate_map_threshold", None) is not None:
            cmd += ["--restate-map-threshold", str(args.restate_map_threshold)]
        if getattr(args, "force_chapter_turn", None) is not None:
            cmd += ["--force-chapter-turn", str(args.force_chapter_turn)]
        if getattr(args, "cadence_profile", None) is not None:
            cmd += ["--cadence-profile", args.cadence_profile]
        print(f"[long_soak] running: {' '.join(cmd)}")
        code = subprocess.call(cmd)
        if code != 0:
            failures += 1

    if failures:
        print(f"\n{failures} test(s) failed")
        return 1
    print("\nAll tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
