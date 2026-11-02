#!/usr/bin/env python3
"""Dump/debug tooling for long_horizon_soak run directories.

All commands take a run dir (e.g. tests/runs/cyberpunk_thriller__4dbf1435).

Commands:
  turns   <run_dir>                        one-line table of every turn
  show    <run_dir> <turn>                 full dump of one turn (prompt/context/reply/judge/state)
  history <run_dir> [--upto N]             assembled user/reply transcript
  trace   <run_dir> <file.path.to.field>   field value across every snapshot (staleness hunter);
                                           file is a snapshot stem: current_scene characters groups
                                           elements events general_info past  e.g.
                                             trace RUN current_scene.now.what
                                             trace RUN characters.entries.Juno.locket_ritual
  compare <run_dir> <a> <b>                coarse snapshot diff between two turns
  bundle  <run_dir> <turns...> [-o DIR]    compile self-contained markdown bundles for manual review
  prompts <run_dir> <turns...>             REPLAY the DataSummarizer update passes offline and
                                           write every LLM prompt the engine builds per turn to
                                           <run>/_prompts/turn_NNN/ (no GPU/server needed).
  prompts-dump <run_dir>                   split the engine's own (last-turn-overwritten)
                                           dump.txt diagnostics into labeled files.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

SNAP_FILES = ["current_scene", "characters", "groups", "elements", "events",
              "general_info", "past"]


def _load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _turn_dirs(run_dir: Path) -> list[int]:
    return sorted(int(p.name.split("_")[1]) for p in run_dir.glob("turn_*")
                  if p.is_dir() and p.name.split("_")[1].isdigit())


def _result(run_dir: Path, t: int) -> dict | None:
    return _load(run_dir / f"turn_{t:03d}" / "result.json")


def _snap(run_dir: Path, t: int) -> dict:
    """Snapshot files of a turn as {stem: parsed_json_or_None}."""
    d = run_dir / f"turn_{t:03d}" / "state_snapshot"
    out = {}
    if d.is_dir():
        for p in sorted(d.glob("*.json")):
            out[p.stem] = _load(p)
    return out


def _walk(obj, parts):
    cur = obj
    for p in parts:
        if isinstance(cur, dict):
            if p not in cur:
                return None, f"key '{p}' missing (have: {list(cur)[:12]})"
            cur = cur[p]
        elif isinstance(cur, list):
            try:
                cur = cur[int(p)]
            except (ValueError, IndexError):
                return None, f"'{p}' not a valid index (len={len(cur)})"
        else:
            return None, f"cannot descend '{p}' into {type(cur).__name__}"
    return cur, None


def _truncate(v, n=160):
    s = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v
    s = " ".join(s.split())
    return s[:n] + ("…" if len(s) > n else "")


def cmd_turns(a):
    rows = []
    for t in _turn_dirs(Path(a.run_dir)):
        r = _result(Path(a.run_dir), t) or {}
        j = r.get("judge") or {}
        scores = (f"{j.get('style_score')}/{j.get('quality_score')}/{j.get('memory_fidelity')}"
                  if j else "-")
        dt = r.get("dt_s") or 0
        ctx = len(r.get("context") or "")
        reply = _truncate(r.get("reply") or "", 70)
        rows.append(f"t{t:03d}  dt={dt:>6.0f}s  ctx={ctx:>6}c  judge(s/q/m)={scores:>9}  | {reply}")
    print("\n".join(rows))


def cmd_show(a):
    rd, t = Path(a.run_dir), a.turn
    r = _result(rd, t)
    if not r:
        sys.exit(f"no result.json for turn {t}")
    line = "=" * 78
    print(f"{line}\nTURN {t}   dt={r.get('dt_s'):.0f}s\n{line}")
    for label in ("user_input", "context", "instr_prompt", "reply"):
        print(f"\n----- {label} ({len(r.get(label) or '')} chars) -----")
        print(r.get(label) or "(none)")
    j = r.get("judge")
    if j:
        print(f"\n----- judge (style={j.get('style_score')} quality={j.get('quality_score')} "
              f"mem={j.get('memory_fidelity')} abort={j.get('abort')}) -----")
        for n in j.get("notes") or []:
            print(f"  [{n.get('id')}] {n.get('status')}: {_truncate(n.get('detail'), 300)}")
        print("  summary:", _truncate(j.get("summary"), 600))
    print("\n----- state summary -----")
    snaps = _snap(rd, t)
    cs = snaps.get("current_scene") or {}
    gi = snaps.get("general_info") or {}
    ev = snaps.get("events") or {}
    print(f"  scene=_scene_number={cs.get('_scene_number')} chapter=_chapter_number="
          f"{cs.get('_chapter_number')} arc={cs.get('_arc_number')}")
    print(f"  now.who={cs.get('now', {}).get('who')}  now.what={_truncate(cs.get('now', {}).get('what'), 120)}")
    print(f"  synopsis={len(gi.get('synopsis') or '')}c  "
          f"chars/groups/elements={len(snaps.get('characters', {}).get('entries') or {})}/"
          f"{len((snaps.get('groups') or {}).get('entries') or {})}/"
          f"{len((snaps.get('elements') or {}).get('entries') or {})}  "
          f"scenes={len(ev.get('scenes') or {})} chapters={len(ev.get('chapters') or [])}")
    for stem, data in snaps.items():
        p = rd / f"turn_{t:03d}" / "state_snapshot" / f"{stem}.json"
        print(f"\n----- snapshot {stem}.json ({p.stat().st_size}b) -----")
        print(json.dumps(data, indent=2, ensure_ascii=False)[:a.max_snap * 1000])


def cmd_history(a):
    rd = Path(a.run_dir)
    upto = a.upto or max(_turn_dirs(rd), default=0)
    for t in range(1, upto + 1):
        r = _result(rd, t)
        if not r:
            continue
        print(f"\n===== [{t}] USER =====\n{(r.get('user_input') or '').strip()}")
        print(f"\n----- [{t}] JUNO -----\n{(r.get('reply') or '').strip()}")


def cmd_trace(a):
    rd = Path(a.run_dir)
    stem, _, rest = a.field.partition(".")
    if stem not in SNAP_FILES:
        sys.exit(f"first path element must be a snapshot file, one of: {SNAP_FILES}")
    parts = rest.split(".") if rest else []
    prev = object()
    for t in _turn_dirs(rd):
        snaps = _snap(rd, t)
        val, err = _walk(snaps.get(stem), parts)
        rendered = _truncate(val, a.width)
        marker = ""
        if err:
            rendered = f"<{err}>"
        elif val == prev:
            marker = "  (=)"
        elif not isinstance(prev, object) or prev is None:
            pass
        print(f"t{t:03d}: {rendered}{marker}")
        prev = val


def cmd_compare(a):
    rd = Path(a.run_dir)
    sa, sb = _snap(rd, a.a), _snap(rd, a.b)

    def flat(o, prefix="", depth=0):
        items = {}
        if depth >= a.depth or not isinstance(o, (dict, list)):
            items[prefix] = o
            return items
        if isinstance(o, dict):
            for k, v in o.items():
                items.update(flat(v, f"{prefix}.{k}" if prefix else str(k), depth + 1))
        elif isinstance(o, list):
            items[f"{prefix}#len"] = len(o)
        return items

    fa, fb = flat(sa), flat(sb)
    changed = [k for k in fa if k in fb and fa[k] != fb[k]]
    added = [k for k in fb if k not in fa]
    dropped = [k for k in fa if k not in fb]
    same = sum(1 for k in fa if k in fb and fa[k] == fb[k])
    print(f"# {a.a} -> {a.b}: {len(changed)} changed, {len(added)} added, "
          f"{len(dropped)} dropped, {same} identical leaves (depth={a.depth})")
    if a.all:
        for k in sorted(changed):
            print(f"~ {k}:\n   @{a.a}: {_truncate(fa[k], 200)}\n   @{a.b}: {_truncate(fb[k], 200)}")
    else:
        print("changed roots:", ", ".join(sorted({k.split('.')[0] for k in changed})))
        print("added roots:  ", ", ".join(sorted({k.split('.')[0] for k in added})))
        print("dropped roots:", ", ".join(sorted({k.split('.')[0] for k in dropped})) or "-")
        print("(use --all for leaf-level diffs)")


def cmd_bundle(a):
    rd = Path(a.run_dir)
    out_dir = Path(a.out) if a.out else rd / "_bundles"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifests = []
    for t in a.turns:
        r = _result(rd, t)
        if not r:
            print(f"skip t{t:03d}: no result.json", file=sys.stderr)
            continue
        snaps = _snap(rd, t)
        md = [f"# Soak bundle — {rd.name} — turn {t}", ""]
        j = r.get("judge") or {}
        md.append(f"- dt: {r.get('dt_s'):.0f}s | ctx chars: {len(r.get('context') or '')} | "
                  f"judge: style={j.get('style_score')} quality={j.get('quality_score')} "
                  f"mem={j.get('memory_fidelity')} abort={j.get('abort')}")
        probes = r.get("probes") or {}
        recalls = r.get("recalls") or {}
        md.append(f"- probes fired: {json.dumps(probes, ensure_ascii=False)}")
        md.append(f"- recalls hit: {json.dumps(recalls, ensure_ascii=False)}")
        cs = (snaps.get("current_scene") or {})
        md.append(f"- state: scene={cs.get('_scene_number')} chapter={cs.get('_chapter_number')} "
                  f"arc={cs.get('_arc_number')}")
        for label in ("user_input", "instr_prompt", "context", "reply"):
            md += ["", f"## {label}", "", "```", str(r.get(label) or ""), "```"]
        if j:
            md += ["", "## judge", "", "```json",
                   json.dumps(j, indent=2, ensure_ascii=False), "```"]
        for stem, data in snaps.items():
            md += ["", f"## snapshot: {stem}.json", "", "```json",
                   json.dumps(data, indent=2, ensure_ascii=False), "```"]
        dest = out_dir / f"turn_{t:03d}_bundle.md"
        dest.write_text("\n".join(md), encoding="utf-8")
        manifests.append(dest)
        print(f"wrote {dest} ({dest.stat().st_size // 1024} KiB)")
    idx = out_dir / "INDEX.md"
    idx.write_text("\n".join(f"- {p.name}" for p in manifests) + "\n", encoding="utf-8")


# --------------------------------------------------------------- prompts ---

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_history_pairs(run_dir: Path, upto: int) -> list[list[str]]:
    pairs = []
    for t in range(1, upto + 1):
        r = _result(run_dir, t)
        if not r:
            sys.exit(f"turn {t} has no result.json — cannot build history")
        pairs.append([r.get("user_input") or "", r.get("reply") or ""])
    return pairs


class _RecordingModel:
    """Offline stand-in plugged in as ``runtime.model``.

    The summarizer's LLM client funnels every call through
    ``model.generate_with_streaming`` after the FINAL prompt has been assembled,
    so one generator records the exact production string and returns a canned
    response that keeps the pipeline flowing (stopping-string logic then runs in
    the client, same as production).
    """

    CANNED = {
        "perform_gate_check": "YES",
        "query_changes": "YES",
        "apply_updates": "NO_UPDATES_REQUIRED",
    }
    DEFAULT_CANNED = "UNCHANGED"

    def __init__(self):
        self.calls: list[dict] = []
        self._label: dict | None = None      # injected by patched generate_with_sse

    def set_label(self, label: dict) -> None:
        self._label = label

    def generate_with_streaming(self, encoded_prompt, state):
        text = self.CANNED.get((self._label or {}).get("step_id", ""),
                               self.DEFAULT_CANNED)
        self.calls.append({
            "phase_id": (self._label or {}).get("phase_id", "unknown"),
            "step_id": (self._label or {}).get("step_id", "bypass"),
            "prompt": str(encoded_prompt),
            "canned": text,
        })
        yield text

    def stats(self):
        return {"calls": len(self.calls)}


def _write_prompt_files(calls, out_dir):
    index = ["| # | phase_id | step_id | chars | first line |", "|---|---|---|---|---|"]
    for i, c in enumerate(calls, 1):
        slug = (c["prompt"].strip().splitlines() or [""])[0][:60].strip()
        slug = "".join(ch if ch.isalnum() or ch in "-_ " else "_" for ch in slug)[:60]
        name = f"{i:03d}_{c['phase_id']}__{c['step_id']}.txt"
        header = (f"record #{i}\nphase_id = {c['phase_id']}\nstep_id  = {c['step_id']}\n"
                  f"chars    = {len(c['prompt'])}\n"
                  + "-" * 70 + "\n")
        (out_dir / name).write_text(header + c["prompt"], encoding="utf-8")
        index.append(f"| {i} | {c['phase_id']} | {c['step_id']} | {len(c['prompt'])} | {slug[:48]} |")
    (out_dir / "INDEX.md").write_text("\n".join(index) + "\n", encoding="utf-8")


def cmd_prompts(a):
    """Offline replay of DataSummarizer passes; no GPU / server / network.

    For each requested turn N:
      * copy run sandbox to a temp dir (all engine writes land there),
      * seed turn N-1's snapshot into the hashed state dir turn N would read,
      * rebuild the exchange history from result.json files,
      * plug a recording model into runtime, with step-aware canned replies,
      * run the REAL summarize_latest_state for exchange N.
    """
    repo = _repo_root()
    root_str = str(repo)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    rd = Path(a.run_dir)
    manifest = _load(rd / "manifest.json") or {}
    margs = SimpleNamespace(**manifest.get("args", {}))
    margs.max_subject_workers = 0

    # FIRST THING: make the engine's lazy importer synchronous. DaynaSS spawns
    # one background thread per (module, attr) on first request; several of
    # those race against each other on python's per-module locks, hit CPython's
    # "deadlock detected" guard, and poison their cache keys for the whole
    # process. Inline imports are strictly slower but always correct.
    import shutil as _shutil
    import tempfile

    from extensions.dayna_ss.utils import background_importer as _bi
    import threading as _threading

    def _sync_start(module_name, attribute_name=None):
        key = _bi._get_key(module_name, attribute_name)
        _bi._import_locks.setdefault(key, _threading.Lock())
        _bi._import_done_events.setdefault(key, _threading.Event())
        if not _bi._import_started_flags.get(key, False):
            _bi._import_started_flags[key] = True
            _bi._perform_import(module_name, attribute_name)

    _bi.start_background_import = _sync_start

    spread_path = (Path(a.spreadsheet) if a.spreadsheet else
                   repo / "extensions/dayna_ss/tests" / getattr(margs, "spreadsheet", ""))
    from extensions.dayna_ss.tests.long_horizon_soak import load_spreadsheet
    spreadsheet = load_spreadsheet(spread_path)
    greeting = spreadsheet.get("greeting", "")
    char2 = spreadsheet["characters"]["name2"]["name"]

    from extensions.dayna_ss.runtime import runtime
    from extensions.dayna_ss.agents.summarizer import Summarizer
    from extensions.dayna_ss.tests.long_horizon_soak import build_state

    for turn in a.turns:
        result_n = _result(rd, turn)
        if not result_n:
            print(f"skip turn {turn}: no result.json", file=sys.stderr)
            continue
        print(f"== replaying turn {turn}")

        internal = ([["<|BEGIN-VISIBLE-CHAT|>", greeting]] if greeting else []) \
            + _load_history_pairs(rd, turn)
        user_input, reply = internal[-1]

        with tempfile.TemporaryDirectory(prefix="dss_prompts_") as tmp_raw:
            tmp = Path(tmp_raw) / "sandbox"
            _shutil.copytree(rd / "sandbox", tmp,
                             ignore=_shutil.ignore_patterns("__pycache__", "*.log"))

            hp = result_n.get("history_path") or ""
            rel = hp.split("sandbox/", 1)[-1] if "sandbox/" in hp else None
            if rel:
                dst = tmp / rel
                dst.mkdir(parents=True, exist_ok=True)
                src_seed = rd / f"turn_{max(turn - 1, 0):03d}" / "state_snapshot"
                if src_seed.is_dir():
                    for p in sorted(src_seed.glob("*.json")):
                        _shutil.copy2(p, dst / p.name)

            current_character = {"name": f"soak_{spreadsheet['id']}"}
            recorder = _RecordingModel()
            orig_sse = Summarizer.generate_with_sse

            def _labeled_sse(self, prompt, state, *args, **kw):
                phase = kw.get("phase_id") or (args[0] if len(args) > 0 else "")
                step = kw.get("step_id") or (args[1] if len(args) > 1 else "")
                recorder.set_label({"phase_id": phase, "step_id": step})
                return orig_sse(self, prompt, state, *args, **kw)

            runtime.configure(
                model_provider=lambda: recorder,
                stop_provider=lambda: False,
                prompt_builder=lambda prompt, state, **kw: prompt,
                encoder=lambda text, add_bos_token=True: text,
                persistent_ui_state_provider=lambda: {},
                current_character_provider=lambda: current_character["name"],
                settings_provider=lambda: {},
                update_config_fn=lambda state: False,
                register_tool_executors_fn=lambda _: None,
                extension_dir=tmp,
            )

            summarizer = Summarizer()
            Summarizer.generate_with_sse = _labeled_sse

            # Seed prior-turn state into WHATEVER hashed dir this replay
            # resolves (the hash depends on build_state internals, so capture it
            # at load time instead of precomputing).
            seeded = {"done": False}
            orig_load = summarizer._load_all_subjects_data

            def _seeded_load(last_history_path, *args2, **kw2):
                if not seeded["done"]:
                    last_history_path.mkdir(parents=True, exist_ok=True)
                    src_seed = rd / f"turn_{max(turn - 1, 0):03d}" / "state_snapshot"
                    if src_seed.is_dir():
                        for p in sorted(src_seed.glob("*.json")):
                            _shutil.copy2(p, last_history_path / p.name)
                        print(f"  seeded {last_history_path.name} <- turn {turn-1} snapshot")
                    seeded["done"] = True
                return orig_load(last_history_path, *args2, **kw2)

            summarizer._load_all_subjects_data = _seeded_load
            summarizer.config.update({
                "max_subject_workers": 0,
                "cadence_profile": getattr(margs, "cadence_profile", None),
                "max_scene_part_messages": max(0, getattr(margs, "max_scene_messages", 8)),
                "message_mode": getattr(margs, "message_mode", "rolling"),
                "retrieval_placement": getattr(margs, "retrieval_placement", "system"),
                "rolling_window": max(0, getattr(margs, "rolling_window", 0)),
                "rolling_summaries": max(0, getattr(margs, "rolling_summaries", 0)),
                "restate_map_context": getattr(margs, "restate_map_context", True),
                "restate_map_threshold_chars":
                    max(0, getattr(margs, "restate_map_threshold", 3000)),
            })

            # production parity: build_state builds the full raw TGWUI-shaped
            # state (unique_id, context, etc.) exactly like the harness's per-turn
            # init; then the same overrides applied at summarize time.
            engine_state = build_state(spreadsheet, getattr(margs, "seed", 0),
                                       f"soak_{getattr(margs, 'seed', 0)}",
                                       history=internal)
            engine_state["history"]["internal"] = [list(e) for e in internal]
            engine_state["context"] = result_n.get("context") or ""

            out_dir = rd / "_prompts" / f"turn_{turn:03d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            try:
                summarizer.summarize_latest_state(reply, user_input, engine_state, internal)
            except Exception:
                traceback.print_exc()
                print(f"  replay aborted mid-pass (partial prompts kept)")
            finally:
                Summarizer.generate_with_sse = orig_sse
                recorder.set_label(None)

            _write_prompt_files(recorder.calls, out_dir)
            print(f"  captured {len(recorder.calls)} prompts -> {out_dir}")
            steps = {}
            for c in recorder.calls:
                steps[c["step_id"]] = steps.get(c["step_id"], 0) + 1
            print("  by step:", ", ".join(f"{k}x{v}" for k, v in sorted(steps.items())))


def cmd_prompts_dump(a):
    """Split the engine's own dump.txt diagnostics into labeled files.

    NOTE: instruction_generator rewrites soak dump.txt every turn ("w" mode), so
    only the final turn's prompts survive here — use `prompts` (replay) for any
    other turn.
    """
    rd = Path(a.run_dir)
    d = rd / "sandbox" / "user_data" / "history"
    files = sorted(d.rglob("dump.txt"))
    if not files:
        sys.exit("no dump.txt found under sandbox/user_data/history")
    out_root = rd / "_prompts" / "engine_dumps"
    out_root.mkdir(parents=True, exist_ok=True)
    import re
    SEP_PROMPT = "========================== NEW PROMPT\n"
    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        tag = f.parent.name or "root"
        out_dir = out_root / tag.replace("/", "_")
        out_dir.mkdir(parents=True, exist_ok=True)
        chunks = text.split(SEP_PROMPT)[1:]
        index = ["| # | chars | first line |", "|---|---|---|"]
        for i, chunk in enumerate(chunks, 1):
            # layout: PROMPT ... "\n\n====\n" {next record's kwargs+internal};
            # the real prompt stops at the first long '=' rule line.
            m = re.search(r"\n={15,}[ \t]*\n", chunk)
            body = (chunk[:m.start()] if m else chunk).strip("\n")
            first = (body.splitlines() or [""])[0][:60]
            name = f"{i:03d}.txt"
            header = f"source={f.name} ({tag}) record #{i}\n" + "-" * 70 + "\n"
            (out_dir / name).write_text(header + body, encoding="utf-8")
            index.append(f"| {i} | {len(body)} | {first} |")
        (out_dir / "INDEX.md").write_text("\n".join(index) + "\n", encoding="utf-8")
        print(f"split {len(chunks)} records -> {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def base(name, help_):
        p = sub.add_parser(name, help=help_)
        p.add_argument("run_dir")
        return p

    base("turns", "table of every turn").set_defaults(func=cmd_turns)

    p = base("show", "full dump of one turn")
    p.add_argument("turn", type=int)
    p.add_argument("--max-snap", type=int, default=2000,
                   help="max KB per printed snapshot json")
    p.set_defaults(func=cmd_show)

    p = base("history", "user/reply transcript")
    p.add_argument("--upto", type=int)
    p.set_defaults(func=cmd_history)

    p = base("trace", "field across snapshots (staleness hunter)")
    p.add_argument("field")
    p.add_argument("--width", type=int, default=160)
    p.set_defaults(func=cmd_trace)

    p = base("compare", "snapshot diff between two turns")
    p.add_argument("a", type=int)
    p.add_argument("b", type=int)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--all", action="store_true")
    p.set_defaults(func=cmd_compare)

    p = base("bundle", "compile markdown bundles for manual review")
    p.add_argument("turns", nargs="+", type=int)
    p.add_argument("-o", "--out")
    p.set_defaults(func=cmd_bundle)

    p = base("prompts", "replay DSS update passes offline; write every prompt per turn")
    p.add_argument("turns", nargs="+", type=int)
    p.add_argument("--spreadsheet",
                   help="override spreadsheet path (default: manifest args, repo-relative)")
    p.set_defaults(func=cmd_prompts)

    base("prompts-dump", "split engine dump.txt diagnostics into labeled files"
        ).set_defaults(func=cmd_prompts_dump)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
