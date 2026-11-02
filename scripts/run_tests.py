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
import datetime as _dt
import json as _json
import shutil as _shutil
import subprocess as _subprocess
import sys
from pathlib import Path

from director_test import run_director_test
from boundary_persistence_test import run_boundary_persistence_test
from cadence_profiles_test import run_cadence_profile_test
from force_boundary_test import run_force_boundary_test
from harness import FIXTURES_DIR, Scenario
from scene_window_test import run_scene_window_test
from soak_test import run_soak


class DumpWriter:
    """Consolidated, minimal-truncation result dump for a test run.

    Every hermetic test already returns its complete, un-truncated result
    strings (the ``msgs`` list); the soak and golden scenarios additionally
    produce an ephemeral per-turn ``out_dir``. This writer persists both so the
    user can analyze the full results offline: a single verbatim ``results.txt``
    (header + every test's full ``msgs`` in run order) plus a copy of each
    preserved ``out_dir`` (per-turn state + result.json).

    The dump dir is ``tests/_dumps/<UTC-timestamp>/`` (already git-ignored via
    the ``/extensions`` pattern). Writing is best-effort: a dump failure must
    never mask a test failure, so ``flush`` swallows and reports its own errors
    into the results file / stdout rather than raising.
    """

    def __init__(self) -> None:
        ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.dir = Path(__file__).parent / "_dumps" / ts
        self.dir.mkdir(parents=True, exist_ok=True)
        self._results: list[tuple[str, bool, list[str]]] = []
        self._outdirs: list[tuple[str, Path]] = []
        self._git_head = self._git_head()

    @staticmethod
    def _git_head() -> str:
        try:
            return _subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip() or "unknown"
        except Exception:
            return "unknown"

    def record(self, name: str, ok: bool, msgs: list[str]) -> None:
        """Record one test's full result (name, pass/fail, complete msgs)."""
        self._results.append((name, ok, list(msgs)))

    def record_outdir(self, name: str, out_dir: Path) -> None:
        """Queue an ephemeral per-turn out_dir for preservation into the dump."""
        self._outdirs.append((name, Path(out_dir)))

    def flush(self) -> Path | None:
        """Write results.txt + preserve all queued out_dirs; return the dump dir.

        Best-effort: any per-item failure is recorded inline (in results.txt /
        stdout) and skipped, never raised, so a dump problem cannot mask or
        alter the test verdict.
        """
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"[dump] cannot create dump dir: {e}")
            return None

        # --- preserve out_dirs (per-turn state) first, so a results.txt write
        #     failure below does not also lose the preserved state ---
        preserved: list[str] = []
        for name, out_dir in self._outdirs:
            dest = self.dir / "out_dirs" / _slug(name)
            try:
                if out_dir.exists():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if dest.exists():
                        _shutil.rmtree(dest, ignore_errors=True)
                    _shutil.copytree(out_dir, dest, symlinks=True,
                                      ignore=_shutil.ignore_patterns(".*"))
                    preserved.append(f"{_slug(name)} <- {out_dir}")
            except Exception as e:
                preserved.append(f"{_slug(name)}: COPY FAILED ({e})")

        # --- consolidated verbatim results.txt ---
        lines: list[str] = []
        lines.append("# dayna_ss test run dump")
        lines.append(f"timestamp: {self.dir.name}")
        lines.append(f"git HEAD: {self._git_head}")
        n_pass = sum(1 for _, ok, _ in self._results if ok)
        n_fail = len(self._results) - n_pass
        overall = "PASS" if n_fail == 0 else f"FAIL ({n_fail} test(s) failed)"
        lines.append(f"overall: {overall}")
        lines.append(f"tests: {len(self._results)} | pass: {n_pass} | fail: {n_fail}")
        if preserved:
            lines.append(f"preserved out_dirs: {len(preserved)}")
            for p in preserved:
                lines.append(f"  {p}")
        lines.append("")
        for name, ok, msgs in self._results:
            lines.append(f"=== {name} [{'PASS' if ok else 'FAIL'}] ===")
            for m in msgs:
                lines.append(m)
            lines.append("")
        try:
            (self.dir / "results.txt").write_text("\n".join(lines), encoding="utf-8")
        except Exception as e:
            print(f"[dump] failed to write results.txt: {e}")
            return None
        print(f"[dump] results written: {self.dir / 'results.txt'} "
              f"({len(self._results)} tests, {len(preserved)} out_dirs preserved)")
        return self.dir


def _slug(name: str) -> str:
    """Filesystem-safe slug for a test/scenario name (dump subdir naming)."""
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in name) or "unnamed"


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
    # Consolidated result dump (verbatim msgs + preserved per-turn out_dirs).
    # Only for real runs — golden regeneration (-u) and single-scenario runs
    # (-s) do not produce a suite dump.
    dump = DumpWriter() if (not args.update_golden and not args.scenario) else None
    for name in names:
        scenario = Scenario(name, FIXTURES_DIR / name)
        if scenario.meta.get("kind") == "soak":
            if args.scenario and not args.update_golden:
                soak_ok, soak_msgs, _ = run_soak(name)
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
            if dump:
                dump.record(name, True, [f"[{name}] PASS ({len(model.calls)} LLM calls scripted)"])
                dump.record_outdir(name, out_dir)
        else:
            failures += 1
            print(f"[{name}] FAIL")
            for m in missing:
                print(f"  MISSING: {m}")
            for m in mismatches:
                print(f"  MISMATCH: {m}")
            if dump:
                dump.record(name, False,
                             [f"[{name}] FAIL"] + [f"  MISSING: {m}" for m in missing]
                             + [f"  MISMATCH: {m}" for m in mismatches])
                dump.record_outdir(name, out_dir)

    if not args.scenario and not args.update_golden:
        director_ok, director_msgs = run_director_test()
        for m in director_msgs:
            print("  " + m)
        if not director_ok:
            failures += 1
            print("[director_pass_on_off] FAIL")
        else:
            print("[director_pass_on_off] PASS")
        if dump:
            dump.record("director_pass_on_off", director_ok, director_msgs)

        scene_ok, scene_msgs = run_scene_window_test()
        for m in scene_msgs:
            print("  " + m)
        if not scene_ok:
            failures += 1
            print("[scene_bounded_window] FAIL")
        else:
            print("[scene_bounded_window] PASS")
        if dump:
            dump.record("scene_bounded_window", scene_ok, scene_msgs)

        force_ok, force_msgs = run_force_boundary_test()
        for m in force_msgs:
            print("  " + m)
        if not force_ok:
            failures += 1
            print("[forced_unit_boundaries] FAIL")
        else:
            print("[forced_unit_boundaries] PASS")
        if dump:
            dump.record("forced_unit_boundaries", force_ok, force_msgs)

        cadence_ok, cadence_msgs = run_cadence_profile_test()
        for m in cadence_msgs:
            print("  " + m)
        if not cadence_ok:
            failures += 1
            print("[cadence_profiles] FAIL")
        else:
            print("[cadence_profiles] PASS")
        if dump:
            dump.record("cadence_profiles", cadence_ok, cadence_msgs)

        persist_ok, persist_msgs = run_boundary_persistence_test()
        for m in persist_msgs:
            print("  " + m)
        if not persist_ok:
            failures += 1
            print("[boundary_persistence] FAIL")
        else:
            print("[boundary_persistence] PASS")
        if dump:
            dump.record("boundary_persistence", persist_ok, persist_msgs)

        from chapters_render_test import run_chapters_render_test
        render_ok, render_msgs = run_chapters_render_test()
        for m in render_msgs:
            print("  " + m)
        if not render_ok:
            failures += 1
            print("[chapters_render] FAIL")
        else:
            print("[chapters_render] PASS")
        if dump:
            dump.record("chapters_render", render_ok, render_msgs)

        from canon_synopsis_test import run_canon_synopsis_test
        canon_ok, canon_msgs = run_canon_synopsis_test()
        for m in canon_msgs:
            print("  " + m)
        if not canon_ok:
            failures += 1
            print("[canon_synopsis] FAIL")
        else:
            print("[canon_synopsis] PASS")
        if dump:
            dump.record("canon_synopsis", canon_ok, canon_msgs)

        from hygiene_test import run_hygiene_test
        hyg_ok, hyg_msgs = run_hygiene_test()
        for m in hyg_msgs:
            print("  " + m)
        if not hyg_ok:
            failures += 1
            print("[hygiene] FAIL")
        else:
            print("[hygiene] PASS")
        if dump:
            dump.record("hygiene", hyg_ok, hyg_msgs)

        from referent_gate_test import run_referent_gate_test
        gate_ok, gate_msgs = run_referent_gate_test()
        for m in gate_msgs:
            print("  " + m)
        if not gate_ok:
            failures += 1
            print("[referent_gate] FAIL")
        else:
            print("[referent_gate] PASS")
        if dump:
            dump.record("referent_gate", gate_ok, gate_msgs)

        from dangling_edge_test import run_dangling_edge_test
        dedge_ok, dedge_msgs = run_dangling_edge_test()
        for m in dedge_msgs:
            print("  " + m)
        if not dedge_ok:
            failures += 1
            print("[dangling_edge] FAIL")
        else:
            print("[dangling_edge] PASS")
        if dump:
            dump.record("dangling_edge", dedge_ok, dedge_msgs)

        soak_ok, soak_msgs, soak_outdir = run_soak()
        for m in soak_msgs:
            print("  " + m)
        if not soak_ok:
            failures += 1
        if dump:
            dump.record("soak", soak_ok, soak_msgs)
            if soak_outdir is not None:
                dump.record_outdir("soak", soak_outdir)

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

    if dump is not None:
        dump.flush()

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
