"""Standalone dashboard for the dayna_ss long-horizon soak.

Inspects the ``tests/runs/<config_hash>/`` directories produced by
``long_horizon_soak.py``. Top level is computed stats; every exact payload
(user input, reply, instruction prompt, judge result, probes, state snapshot)
is one click away and fetched on demand.

Run::

    python soak_dashboard.py [--runs-dir tests/runs] [--port 7861] [--host 0.0.0.0]

Stdlib only. Run directories are indexed incrementally: each ``result.json`` is
parsed once and re-parsed only if it changes, so refreshes are O(new turns).
The page polls /api/poll once a second and refetches only what changed.

API:
  GET /api/runs                              run list (+ latest-active run)
  GET /api/poll?run=<id>                     live stats, plus revs to detect changes
  GET /api/runs/<id>/overview                aggregates, per-turn rows, overviews
  GET /api/runs/<id>/turns/<n>               full detail for one turn
  GET /api/runs/<id>/turns/<n>/file/<name>   one state snapshot file (or context.txt)
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import secrets
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

TEST_DIR = Path(__file__).parent
REPO_ROOT = TEST_DIR.parent.parent.parent
DEFAULT_RUNS_DIR = TEST_DIR / "runs"
DEFAULT_LOCAL_BASE = "http://0.0.0.0:5001/v1"

BOOT = secrets.token_hex(3)
ACTIVE_WINDOW = 300
FRESH_CLOUD_CALL = 45

# Cloud model metadata: max context tokens, and input/output cost in USD per
# million tokens (from models.dev, where known).
MODEL_INFO = {
    "deepseek-v4-flash":   {"ctx": 1048576, "cost_in": None,  "cost_out": None},
    "deepseek-v4-pro":     {"ctx": 1048576, "cost_in": 1.73,  "cost_out": 3.46},
    "glm-5.2":             {"ctx": 1048576, "cost_in": 1.20,  "cost_out": 4.20},
    "glm-5.1":             {"ctx": 202752,  "cost_in": 1.384, "cost_out": 4.348},
    "gpt-5.6-luna":        {"ctx": 1050000, "cost_in": 1.10,  "cost_out": 6.599},
    "kimi-k3":             {"ctx": 1048576, "cost_in": 3.00,  "cost_out": 14.999},
    "kimi-k2.7-code":      {"ctx": 262144,  "cost_in": 0.75,  "cost_out": 3.50},
    "kimi-k2.6":           {"ctx": 262144,  "cost_in": 0.773, "cost_out": 3.38},
    "mimo-v2.5":           {"ctx": 1048576, "cost_in": 0.0,   "cost_out": 0.0},
    "mimo-v2.5-pro":       {"ctx": 1048576, "cost_in": 0.0,   "cost_out": 0.0},
    "minimax-m3":          {"ctx": 1048576, "cost_in": 0.395, "cost_out": 1.977},
    "minimax-m2.7":        {"ctx": 196608,  "cost_in": 0.668, "cost_out": 2.674},
    "qwen3.7-max":         {"ctx": 1000000, "cost_in": 0.0,   "cost_out": 0.0},
    "qwen3.6-plus":        {"ctx": 1000000, "cost_in": 0.0,   "cost_out": 0.0},
    "hy3":                 {"ctx": 256000,  "cost_in": 0.0,   "cost_out": 0.0},
}

_BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
               "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")

# Plain-text help for (i) tooltips, defined once and shared with the page.
HELP = {
    "retention": "Note retention: did DSS keep remembering a planted story detail "
                 "(needle) across turns, and did it survive supersession/archiving?",
    "guide_failure": "A note was planted by the guide but never resurfaced because "
                     "the guide failed to mention it again in its turns.",
    "dss_retention_loss": "DSS lost the note — it was planted and later recalled by "
                          "the guide, but DSS no longer has it in its private memory.",
    "superseded": "A note was explicitly superseded by a later note (e.g. a plan "
                  "changed), so it is no longer expected to be remembered.",
    "style": "Style score (0-5): how well DSS's reply matched the writing-style "
             "directive (third-person, register, no verbatim echo of the guide).",
    "fidelity": "Memory fidelity (0-5): how accurately DSS reproduced its stored "
                "memory (characters, items, events, the scene) in the reply.",
    "quality": "Quality score (0-5): overall prose quality — coherence, scene "
               "consistency, plot progression for DSS's reply alone.",
    "scene_part": "Scene-part budget split: when a scene exceeds max_scene_messages "
                  "(default 12), the engine forces a new scene part so on-new-scene "
                  "triggers (e.g. new-character discovery) fire on a regular cadence.",
    "recall": "Recall probe: on a scheduled turn, the harness checks whether a "
              "needle phrase from a planted note appears in DSS's reply.",
    "unassessable": "Audit verdict: the beat could NOT have been satisfied (the "
                    "entity never appeared in-story, or the subject was gated that "
                    "turn) — not a DSS failure.",
    "probe": "A probe checks whether a planted needle phrase is present in DSS's "
             "private memory state at that turn.",
    "audit_statuses": "Statuses: saved / partial / wrong / missing / unassessable.",
    "local_server": "The LMDeploy server DSS talks to. Max ctx from the server flags; "
                    "'last turn ~N tok sent' is the estimated prompt size of the last "
                    "turn, 'unbound history ~N tok' is the whole accumulated "
                    "conversation (untruncated by --max-update-history) estimate.",
    "cloud_models": "The guide, judge, and auditor cloud models. Max context per "
                    "model; 'last turn ~N tok sent' is that turn's prompt size.",
    "progress": "Turns completed of the configured total, plus the last checkpoint "
                "turn and run start time.",
    "usage_totals": "Cumulative LLM usage for the whole run (per-turn deltas, resume-aware).",
    "local_ctx": "Total prompt tokens DSS sent to the local model (server reports 0, "
                 "so this is an estimate).",
}

TRANSITION_KINDS = (("scene", "_scene_number"), ("chapter", "_chapter_number"), ("arc", "_arc_number"))


# --------------------------------------------------------------------------- #
# file helpers
# --------------------------------------------------------------------------- #


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, ValueError):
        return ""


_json_cache: dict[Path, tuple[tuple[int, int], object]] = {}


def cached_json(path: Path):
    """read_json keyed on (mtime, size); failed parses are never cached so a
    half-written file is retried on the next call."""
    try:
        st = path.stat()
    except OSError:
        _json_cache.pop(path, None)
        return None
    key = (st.st_mtime_ns, st.st_size)
    hit = _json_cache.get(path)
    if hit and hit[0] == key:
        return hit[1]
    data = read_json(path)
    if data is not None:
        _json_cache[path] = (key, data)
    return data


def parse_created(man: dict, fallback: float) -> float:
    ts = man.get("created_at") or ""
    try:
        return dt.datetime.fromisoformat(ts).timestamp()
    except ValueError:
        return fallback


def hms(iso: str) -> str:
    if not iso:
        return ""
    try:
        return dt.datetime.fromisoformat(iso).strftime("%H:%M:%S")
    except ValueError:
        return iso[-8:]


# --------------------------------------------------------------------------- #
# local system probes (run in a sampler thread, never on a request)
# --------------------------------------------------------------------------- #


def local_api_key() -> str:
    key = os.environ.get("DSS_BENCH_API_KEY", "").strip()
    if key:
        return key
    auth = read_json(Path.home() / ".local" / "share" / "opencode" / "auth.json")
    if isinstance(auth, dict):
        for name in ("localhost", "textgen", "llama"):
            e = auth.get(name)
            if isinstance(e, dict) and e.get("key"):
                return e["key"]
            if isinstance(e, str):
                return e
    return "not-needed"


def local_ctx_from_flags() -> int:
    m = re.search(r"--n_ctx\s+(\d+)", read_text(REPO_ROOT / "CMD_FLAGS_m.txt"))
    return int(m.group(1)) if m else 65536


def local_server_status(base: str) -> dict:
    out = {"up": False, "model": None, "loader": None, "error": None}
    key = local_api_key()
    for path, into in (("models", "model"), ("internal/model/info", "loader")):
        try:
            req = urllib.request.Request(
                base.rstrip("/") + "/" + path,
                headers={"Authorization": f"Bearer {key}", "User-Agent": _BROWSER_UA},
            )
            data = json.loads(urllib.request.urlopen(req, timeout=4).read().decode("utf-8"))
            if into == "model":
                out["up"] = True
                out["model"] = data["data"][0].get("id") if data.get("data") else None
            else:
                out["loader"] = data.get("loader")
        except Exception as e:
            if into == "model":
                out["error"] = str(e)
    return out


def gpu_stats() -> list[dict]:
    try:
        lines = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=6,
        ).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in lines:
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 5:
            rows.append({"index": p[0], "name": p[1], "util": p[2], "mem_used": p[3], "mem_total": p[4]})
    return rows


class Sampler(threading.Thread):
    """Samples GPU and local-server state on its own cadence, only while a
    client is polling."""

    IDLE_AFTER = 30

    def __init__(self):
        super().__init__(daemon=True)
        self.lock = threading.Lock()
        self.local_base = DEFAULT_LOCAL_BASE
        self.last_touch = 0.0
        self.force_server = False
        self.state = {"server": None, "ctx": local_ctx_from_flags(), "gpus": [], "gpu_at": None}

    def touch(self, local_base: str | None):
        self.last_touch = time.time()
        if local_base and local_base != self.local_base:
            self.local_base = local_base
            self.force_server = True

    def snapshot(self) -> dict:
        with self.lock:
            return dict(self.state)

    def run(self):
        next_gpu = next_server = 0.0
        while True:
            time.sleep(0.5)
            now = time.time()
            if now - self.last_touch > self.IDLE_AFTER:
                continue
            if now >= next_gpu:
                rows = gpu_stats()
                with self.lock:
                    self.state["gpus"] = rows
                    self.state["gpu_at"] = dt.datetime.now().strftime("%H:%M:%S")
                next_gpu = now + (3 if rows else 60)
            if now >= next_server or self.force_server:
                self.force_server = False
                status = local_server_status(self.local_base)
                status["at"] = dt.datetime.now().strftime("%H:%M:%S")
                with self.lock:
                    self.state["server"] = status
                    self.state["ctx"] = local_ctx_from_flags()
                next_server = time.time() + 10


# --------------------------------------------------------------------------- #
# per-run index
# --------------------------------------------------------------------------- #


def read_scene(turn_dir: Path):
    data = read_json(turn_dir / "state_snapshot" / "current_scene.json")
    if not isinstance(data, dict):
        return None
    return {k: (data[key] if isinstance(data.get(key), int) else None) for k, key in TRANSITION_KINDS}


def _merge_cloud_usage(r: dict) -> dict:
    """Merge guide + judge + auditor cloud usage into one dict."""
    cu = dict(r.get("cloud_usage") or {})
    for key in ("judge_usage", "auditor_usage"):
        sub = r.get(key)
        if isinstance(sub, dict):
            for field in ("calls", "prompt_tokens", "completion_tokens", "total_tokens"):
                cu[field] = cu.get(field, 0) + sub.get(field, 0)
    return cu


def summarize(turn_dir: Path, r: dict, st: os.stat_result) -> dict:
    """The small slice of a result.json that aggregates need. Bulky fields
    (prompts, replies, context) stay on disk until a turn is expanded."""
    j = r.get("judge")
    if isinstance(j, dict) and j:
        judge = ({"error": str(j["error"])} if "error" in j else
                 {"style": j.get("style_score"), "quality": j.get("quality_score"),
                  "fidelity": j.get("memory_fidelity")})
    else:
        judge = None
    wall = hms(r.get("wall_ts") or "") or dt.datetime.fromtimestamp(st.st_mtime).strftime("%H:%M:%S")
    return {
        "dir": turn_dir.name,
        "v": st.st_mtime_ns // 1000,
        "dt_s": float(r.get("dt_s") or 0),
        "wall": wall,
        "judge": judge,
        "present": frozenset(k for k, p in (r.get("probes") or {}).items()
                             if isinstance(p, dict) and p.get("present")),
        "audit": [a for a in (r.get("audit") or []) if isinstance(a, dict)],
        "lu": r.get("local_usage"),
        "cu": _merge_cloud_usage(r),
        "local_calls": r.get("local_calls"),
        "instr_len": len((r.get("instr_prompt") or "").strip()),
        "scene": read_scene(turn_dir),
    }


def usage_series(turns: list[dict]) -> list[dict]:
    """Per-turn model usage. New runs carry per-turn deltas in local_usage /
    cloud_usage. Older runs carried cumulative counters that reset on resume, so
    deltas are reconstructed and a decrease is treated as a process restart."""
    series = []
    prev_l = prev_cc = prev_c = 0
    for s in turns:
        lu, cu = s["lu"], s["cu"]
        if lu is not None:
            series.append({
                "local_calls": int(lu.get("calls") or 0),
                "local_prompt": int(lu.get("prompt_tokens") or 0) or s["instr_len"] // 4,
                "cloud_calls": int((cu or {}).get("calls") or 0),
                "cloud_prompt": int((cu or {}).get("prompt_tokens") or 0),
            })
            prev_l = prev_cc = prev_c = 0
            continue
        l = int(s["local_calls"] or 0)
        cc = int((cu or {}).get("calls") or 0)
        c = int((cu or {}).get("prompt_tokens") or 0)
        reset = (prev_cc and cc < prev_cc) or (prev_l and l < prev_l)
        series.append({
            "local_calls": l if reset else l - prev_l,
            "local_prompt": s["instr_len"] // 4,
            "cloud_calls": cc if reset else cc - prev_cc,
            "cloud_prompt": c if reset else c - prev_c,
        })
        prev_l, prev_cc, prev_c = l, cc, c
    return series


def note_timeline(turns: list[tuple[int, dict]], notes: list[dict], wall: dict) -> list[dict]:
    last_turn = turns[-1][0] if turns else None
    rows = []
    for note in notes:
        nid = note["id"]
        observed = [n for n, s in turns if nid in s["present"]]
        last_seen = observed[-1] if observed else None
        planted = (note.get("plant") or {}).get("turn")
        if not observed:
            attribution = "guide_failure" if (note["type"] == "supersession" or planted is not None) else "not_planted"
        elif note["type"] == "supersession":
            attribution = "supersession"
        elif last_seen != last_turn:
            attribution = "dss_retention_loss"
        else:
            attribution = "survived"
        rows.append({
            "id": nid,
            "type": note["type"],
            "specificity": note.get("specificity", ""),
            "needle": note.get("needle", ""),
            "planted_at": planted,
            "planted_t": wall.get(planted, "") if planted is not None else "",
            "last_seen": last_seen,
            "last_seen_t": wall.get(last_seen, "") if last_seen is not None else "",
            "survived": bool(observed and last_seen == last_turn),
            "attribution": attribution,
        })
    return rows


def type_table(rows: list[dict]) -> list[list]:
    typed: dict[str, dict] = {}
    for r in rows:
        t = typed.setdefault(r["type"], {"count": 0, "survived": 0, "guide_failure": 0, "dss": 0, "superseded": 0})
        t["count"] += 1
        t["survived"] += r["attribution"] == "survived"
        t["guide_failure"] += r["attribution"] == "guide_failure"
        t["dss"] += r["attribution"] == "dss_retention_loss"
        t["superseded"] += r["attribution"] == "supersession"
    return [[k, t["count"], t["survived"], t["guide_failure"], t["dss"], t["superseded"],
             t["survived"] / t["count"] * 100] for k, t in sorted(typed.items())]


def find_transitions(turns: list[tuple[int, dict]]) -> dict[int, dict]:
    prev: dict[str, int | None] = {k: None for k, _ in TRANSITION_KINDS}
    out = {}
    for n, s in turns:
        nos = s["scene"] or {k: None for k, _ in TRANSITION_KINDS}
        moved = {k: [prev[k], nos[k]] for k, _ in TRANSITION_KINDS
                 if prev[k] is not None and nos[k] is not None and nos[k] != prev[k]}
        if moved:
            out[n] = moved
        for k, _ in TRANSITION_KINDS:
            if nos[k] is not None:
                prev[k] = nos[k]
    return out


def split_models(args: dict) -> list[dict]:
    """Unique cloud models across roles. Role values may be comma-separated
    endpoint pools ('prov1/m,prov2/m'); pools of the same model collapse."""
    counts: dict[str, int] = {}
    for role in ("guide_model", "judge_model", "auditor_model"):
        for part in str(args.get(role) or "").split(","):
            part = part.strip()
            if part:
                base = part.rsplit("/", 1)[-1]
                counts[base] = counts.get(base, 0) + 1
    return [{"name": m, "ctx": (MODEL_INFO.get(m) or {}).get("ctx"), "n": n} for m, n in counts.items()]


class Run:
    MIN_REFRESH = 0.5

    def __init__(self, run_id: str, path: Path):
        self.id = run_id
        self.dir = path
        self.lock = threading.Lock()
        self.checked = -1e9
        self.turns: dict[int, dict] = {}
        self.sigs: dict[int, tuple[int, int]] = {}
        self.plans: dict[int, dict] = {}
        self.aux: tuple = ()
        self.version = 0
        self.rev = ""
        self.overview_json = b"{}"
        self.manifest: dict = {}

    def refresh(self):
        if time.monotonic() - self.checked < self.MIN_REFRESH:
            return
        with self.lock:
            if time.monotonic() - self.checked < self.MIN_REFRESH:
                return
            changed = self._scan()
            self.checked = time.monotonic()
            if changed or self.version == 0:
                self._rebuild()

    def _scan(self) -> bool:
        changed = False
        aux = []
        seen = set()
        try:
            entries = list(os.scandir(self.dir))
        except OSError:
            entries = []
        for e in entries:
            name = e.name
            if e.is_file() and (name == "manifest.json" or name == "final_overview.json"
                                or name.startswith("rolling_overview_")):
                try:
                    aux.append((name, e.stat().st_mtime_ns))
                except OSError:
                    pass
                continue
            if not (e.is_dir() and name.startswith("turn_")):
                continue
            try:
                n = int(name.split("_", 1)[1])
            except ValueError:
                continue
            seen.add(n)
            rp = Path(e.path) / "result.json"
            try:
                st = rp.stat()
            except OSError:
                continue
            sig = (st.st_mtime_ns, st.st_size)
            if self.sigs.get(n) != sig:
                r = read_json(rp)
                if not isinstance(r, dict) or not r:
                    continue
                self.turns[n] = summarize(Path(e.path), r, st)
                self.sigs[n] = sig
                plan = r.get("plan") or read_json(Path(e.path) / "plan.json")
                if isinstance(plan, dict) and isinstance(plan.get("created_turn"), int) and plan["created_turn"] >= 0:
                    self.plans[plan["created_turn"]] = plan
                changed = True
            elif self.turns[n]["scene"] is None and time.time() - st.st_mtime < 120:
                scene = read_scene(Path(e.path))
                if scene:
                    self.turns[n]["scene"] = scene
                    changed = True
        for n in set(self.turns) - seen:
            del self.turns[n]
            self.sigs.pop(n, None)
            changed = True
        aux_t = tuple(sorted(aux))
        if aux_t != self.aux:
            self.aux = aux_t
            changed = True
        return changed

    def _spreadsheet(self, man: dict) -> dict:
        sid = man.get("spreadsheet", "")
        if not sid:
            return {}
        for p in (self.dir / f"{sid}.json", TEST_DIR / "spreadsheets" / f"{sid}.json",
                  self.dir.parent.parent / "spreadsheets" / f"{sid}.json"):
            data = cached_json(p)
            if isinstance(data, dict) and data:
                return data
        return {}

    def _rebuild(self):
        man = cached_json(self.dir / "manifest.json") or {}
        self.manifest = man
        args = man.get("args") or {}
        sheet = self._spreadsheet(man)
        turns = sorted(self.turns.items())
        summaries = [s for _, s in turns]
        wall = {n: s["wall"] for n, s in turns}
        last_wall = turns[-1][1]["wall"] if turns else ""

        notes = note_timeline(turns, sheet.get("notes", []), wall) if sheet else []
        survived = sum(r["attribution"] == "survived" for r in notes)
        count = lambda a: sum(r["attribution"] == a for r in notes)
        stats = {
            "notes_total": len(notes),
            "overall_retention": survived / max(1, len(notes)) * 100.0,
            "guide_failure": count("guide_failure"),
            "dss_retention_loss": count("dss_retention_loss"),
            "supersession": count("supersession"),
        }

        series = usage_series(summaries)
        cost = {
            "dt_s": sum(s["dt_s"] for s in summaries),
            "local_calls": sum(u["local_calls"] for u in series),
            "local_prompt": sum(u["local_prompt"] for u in series),
            "cloud_calls": sum(u["cloud_calls"] for u in series),
            "cloud_prompt": sum(u["cloud_prompt"] for u in series),
        }
        trans = find_transitions(turns)

        turn_rows = [{"turn": n, "v": s["v"], "dt_s": s["dt_s"], "wall": s["wall"], "judge": s["judge"],
                      "trans": trans.get(n)} for n, s in turns]
        curves = [{"turn": n, "wall": s["wall"], **s["judge"]} for n, s in turns
                  if s["judge"] and "error" not in s["judge"]]
        audits = [[a.get("id", "?"), n, wall.get(n, "") if isinstance(n, int) else "",
                   a.get("status", "?"), a.get("detail", "")]
                  for tn, s in turns for a in s["audit"] for n in [a.get("turn", tn)]]

        rollings = []
        for name, _ in sorted(a for a in self.aux if a[0].startswith("rolling_overview_")):
            ov = cached_json(self.dir / name)
            if isinstance(ov, dict) and ov:
                rollings.append({k: v for k, v in ov.items() if k != "judge_records"})
        final = cached_json(self.dir / "final_overview.json")
        if isinstance(final, dict):
            final = {k: v for k, v in final.items() if k != "judge_records"}
        else:
            final = None

        total = args.get("turns") or args.get("smoke") or 0
        last_usage = series[-1] if series else {}
        last_lu = (turns[-1][1]["lu"] or {}) if turns else {}
        live = {
            "progress": {"turns_n": len(turns), "total": total, "last_turn": man.get("last_turn"),
                         "created_at": man.get("created_at")},
            "local_sent": int(last_usage.get("local_prompt") or 0),
            "last_fh": int(last_lu.get("full_history_tokens") or 0),
            "cloud_sent": int(last_usage.get("cloud_prompt") or 0),
            "cloud_models": split_models(args),
            "totals": cost,
        }

        self.version += 1
        self.rev = f"{BOOT}.{self.version}"
        self.overview_json = json.dumps({
            "rev": self.rev,
            "id": self.id,
            "manifest": {k: v for k, v in man.items() if k != "args"},
            "args": args,
            "stats": stats,
            "cost": cost,
            "last_wall": last_wall,
            "live": live,
            "turns": turn_rows,
            "transitions": [{"turn": n, "moves": m} for n, m in sorted(trans.items())],
            "notes": notes,
            "types": type_table(notes),
            "curves": curves,
            "audits": audits,
            "plans": [{"created_turn": k, **v} for k, v in sorted(self.plans.items())],
            "rollings": rollings,
            "final": final,
        }).encode()

    def cloud_activity(self) -> list[dict]:
        live_dir = self.dir / "live"
        rows = []
        try:
            files = sorted(live_dir.glob("cloud_*.json"))
        except OSError:
            return rows
        now = time.time()
        for f in files:
            st = read_json(f)
            if not isinstance(st, dict):
                continue
            elapsed = float(st.get("elapsed_s", 0) or 0)
            n_c = int(st.get("content_chars", 0) or 0)
            n_r = int(st.get("reasoning_chars", 0) or 0)
            rows.append({
                "role": str(st.get("role", "?")),
                "model": str(st.get("model", "?")),
                "phase": str(st.get("phase", "generating")),
                "stale": now - float(st.get("updated_at", 0) or 0) > FRESH_CLOUD_CALL,
                "elapsed": elapsed,
                "rate": (n_c + n_r) / elapsed if elapsed > 1 else None,
                "content_chars": n_c,
                "reasoning_chars": n_r,
                "note": str(st.get("note", "")),
                "head": str(st.get("head", ""))[-90:],
                "tail": str(st.get("tail", ""))[:90],
            })
        return rows

    def turn_detail(self, n: int) -> dict | None:
        s = self.turns.get(n)
        if s is None:
            return None
        d = self.dir / s["dir"]
        r = read_json(d / "result.json") or {}
        lu, cu = r.get("local_usage") or {}, _merge_cloud_usage(r)
        plan = r.get("plan") or read_json(d / "plan.json") or {}

        j = r.get("judge")
        if not j:
            judge = []
        elif "error" in j:
            judge = [["error", str(j["error"])]]
        else:
            judge = [["style_score", j.get("style_score")], ["quality_score", j.get("quality_score")],
                     ["memory_fidelity", j.get("memory_fidelity")], ["summary", j.get("summary", "")]]
            judge += [[f"note:{x.get('id')}", f"{x.get('status')} — {x.get('detail')}"] for x in j.get("notes", [])]

        probes = []
        for nid, p in sorted((r.get("probes") or {}).items()):
            paths = []
            for entry in p.get("paths", []):
                paths.append("/".join(str(seg) for seg in entry if seg) if isinstance(entry, (list, tuple)) else str(entry))
            probes.append([nid, p.get("needle", ""), "YES" if p.get("present") else "no", "; ".join(paths) or "—"])

        recalls = [[nid, "YES" if i.get("echoed") else "no", "text-only" if i.get("echo_by_text_only") else "—",
                    ", ".join(i.get("matched_syns", [])) or "—"] for nid, i in sorted((r.get("recalls") or {}).items())]
        audit = [[a.get("id"), a.get("turn"), a.get("status", "?"), a.get("detail", "")] for a in (r.get("audit") or [])]

        state = []
        snap = d / "state_snapshot"
        if snap.is_dir():
            for f in sorted(snap.glob("*.json")):
                data = read_json(f)
                state.append([f.name, len(data) if isinstance(data, dict) else 1])
        if (d / "context.txt").is_file():
            state.append(["context.txt", "text"])

        return {
            "turn": n,
            "ts": r.get("ts", "—"),
            "wall": s["wall"],
            "local": {"calls": lu.get("calls", r.get("local_calls", 0)),
                      "prompt_tokens": lu.get("prompt_tokens", 0),
                      "peak": int(lu.get("max_prompt_tokens") or 0),
                      "ctx": int(lu.get("ctx_size") or 0),
                      "history": int(lu.get("full_history_tokens") or 0)},
            "cloud": {"calls": cu.get("calls", "?"), "prompt_tokens": cu.get("prompt_tokens", 0)},
            "user": read_text(d / "user.txt") or r.get("user_input", ""),
            "reply": read_text(d / "reply.txt") or r.get("reply", ""),
            "instr_prompt": r.get("instr_prompt", ""),
            "plan": plan,
            "judge": judge,
            "probes": probes,
            "recalls": recalls,
            "audit": audit,
            "state": state,
        }

    def state_file(self, n: int, name: str) -> str | None:
        s = self.turns.get(n)
        if s is None or not re.fullmatch(r"[\w.\-]+\.(json|txt)", name):
            return None
        d = (self.dir / s["dir"]).resolve()
        p = (d / name) if name == "context.txt" else (d / "state_snapshot" / name)
        p = p.resolve()
        if not p.is_file() or not p.is_relative_to(d):
            return None
        raw = p.read_text(encoding="utf-8", errors="ignore")
        try:
            return json.dumps(json.loads(raw), indent=2, ensure_ascii=False)
        except ValueError:
            return raw


# --------------------------------------------------------------------------- #
# run registry
# --------------------------------------------------------------------------- #


class Registry:
    LIST_TTL = 1.0

    def __init__(self, runs_dir: Path):
        self.dir = runs_dir
        self.runs: dict[str, Run] = {}
        self.lock = threading.Lock()
        self.listed = -1e9
        self.cache: dict = {"runs": [], "latest": None, "sig": ""}

    def listing(self) -> dict:
        with self.lock:
            if time.monotonic() - self.listed < self.LIST_TTL:
                return self.cache
            items = []
            try:
                entries = list(os.scandir(self.dir))
            except OSError:
                entries = []
            now = time.time()
            for e in entries:
                if not e.is_dir() or e.name.startswith("turn"):
                    continue
                mpath = Path(e.path) / "manifest.json"
                man = cached_json(mpath)
                if not isinstance(man, dict):
                    continue
                try:
                    stamps = [mpath.stat().st_mtime, e.stat().st_mtime]
                    live = Path(e.path) / "live"
                    if live.is_dir():
                        stamps.append(live.stat().st_mtime)
                except OSError:
                    continue
                args = man.get("args") or {}
                total = args.get("turns") or args.get("smoke") or 0
                last = man.get("last_turn")
                done = last + 1 if isinstance(last, int) and last >= 0 else 0
                activity = max(stamps)
                items.append({
                    "id": e.name,
                    "spreadsheet": man.get("spreadsheet", e.name),
                    "genre": man.get("genre", "?"),
                    "created_at": man.get("created_at", "?"),
                    "created": parse_created(man, stamps[0]),
                    "turns_done": done,
                    "total": total,
                    "activity": activity,
                    "active": now - activity < ACTIVE_WINDOW and not (total and done >= total),
                })
            items.sort(key=lambda i: i["created"], reverse=True)
            latest = max(items, key=lambda i: i["activity"])["id"] if items else None
            sig = hashlib.md5(json.dumps([{k: v for k, v in i.items() if k != "activity"} for i in items]).encode()).hexdigest()[:12]
            self.cache = {"runs": items, "latest": latest, "sig": f"{BOOT}.{sig}", "runs_dir": str(self.dir)}
            self.listed = time.monotonic()
            return self.cache

    def get(self, run_id: str) -> Run | None:
        if not run_id or "/" in run_id or run_id.startswith(".") or run_id.startswith("turn"):
            return None
        path = self.dir / run_id
        if not (path / "manifest.json").is_file():
            return None
        with self.lock:
            run = self.runs.get(run_id)
            if run is None:
                run = self.runs[run_id] = Run(run_id, path)
        return run


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #

REGISTRY: Registry
SAMPLER: Sampler


def poll_payload(run_id: str | None) -> dict:
    listing = REGISTRY.listing()
    run = REGISTRY.get(run_id) if run_id else None
    rev = None
    local_base = None
    if run:
        run.refresh()
        rev = run.rev
        local_base = (run.manifest.get("args") or {}).get("local_base")
    SAMPLER.touch(local_base)
    snap = SAMPLER.snapshot()
    return {
        "runs_sig": listing["sig"],
        "latest": listing["latest"],
        "rev": rev,
        "live": {
            "now": time.time(),
            "server": snap["server"],
            "ctx": snap["ctx"],
            "gpus": snap["gpus"],
            "gpu_at": snap["gpu_at"],
            "cloud": run.cloud_activity() if run else [],
        },
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def send_body(self, status: int, ctype: str, body: bytes):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, status: int, obj):
        self.send_body(status, "application/json", obj if isinstance(obj, bytes) else json.dumps(obj).encode())

    def do_GET(self):
        try:
            self.route()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            traceback.print_exc()
            self.send_json(500, {"error": "internal error"})

    def route(self):
        url = urlsplit(self.path)
        params = {k: v[-1] for k, v in parse_qs(url.query).items()}
        parts = [unquote(p) for p in url.path.strip("/").split("/")] if url.path.strip("/") else []

        if not parts:
            self.send_body(200, "text/html; charset=utf-8", PAGE.encode())
        elif parts == ["api", "runs"]:
            self.send_json(200, REGISTRY.listing())
        elif parts == ["api", "poll"]:
            self.send_json(200, poll_payload(params.get("run")))
        elif len(parts) >= 4 and parts[:2] == ["api", "runs"]:
            run = REGISTRY.get(parts[2])
            if run is None:
                return self.send_json(404, {"error": "unknown run"})
            run.refresh()
            self.run_route(run, parts[3:])
        else:
            self.send_json(404, {"error": "not found"})

    def run_route(self, run: Run, rest: list[str]):
        if rest == ["overview"]:
            return self.send_json(200, run.overview_json)
        if len(rest) >= 2 and rest[0] == "turns" and rest[1].isdigit():
            n = int(rest[1])
            if len(rest) == 2:
                detail = run.turn_detail(n)
                return self.send_json(200, detail) if detail else self.send_json(404, {"error": "no such turn"})
            if len(rest) == 4 and rest[2] == "file":
                text = run.state_file(n, rest[3])
                if text is None:
                    return self.send_json(404, {"error": "no such file"})
                return self.send_body(200, "text/plain; charset=utf-8", text.encode())
        self.send_json(404, {"error": "not found"})


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


# --------------------------------------------------------------------------- #
# page
# --------------------------------------------------------------------------- #

PAGE_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Soak dashboard</title>
<script>document.documentElement.dataset.theme = localStorage.getItem('soak-theme') || 'dark';</script>
<style>
  :root { --bg:#0f1419; --panel:#1a232c; --panel2:#161e26; --border:#2b3a47; --text:#dbe1e8; --muted:#8b98a5;
          --faint:#6b7a88; --accent:#6cb2eb; --ok:#2a9d5f; --warn:#e67e22; --bad:#c0392b; --purple:#8e44ad; color-scheme:dark; }
  :root[data-theme=light] { --bg:#f5f6f8; --panel:#ffffff; --panel2:#eef0f3; --border:#d6dce2; --text:#1d2630;
          --muted:#5d6b78; --faint:#8a97a3; --accent:#1a6fc0; color-scheme:light; }
  [hidden] { display:none !important; }
  * { box-sizing:border-box; }
  body { margin:0; padding:16px 20px 60px; background:var(--bg); color:var(--text);
         font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif; }
  h1 { font-size:20px; margin:0; }
  header { display:flex; flex-wrap:wrap; gap:10px 14px; align-items:center; margin-bottom:12px; }
  header .grow { flex:1; }
  select, button, input[type=checkbox] { font:inherit; }
  select, button { background:var(--panel); color:var(--text); border:1px solid var(--border); border-radius:8px; padding:6px 10px; }
  button { cursor:pointer; } button:hover { background:var(--panel2); }
  select { min-width:280px; max-width:100%; }
  label.follow { display:flex; gap:6px; align-items:center; color:var(--muted); cursor:pointer; }
  #conn { font-size:12px; color:var(--muted); }
  #conn.bad { color:var(--bad); }
  #banner { border:1px solid var(--warn); background:var(--panel); border-radius:8px; padding:8px 12px; margin-bottom:10px;
            display:flex; gap:10px; align-items:center; }
  .cards { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:8px; }
  .card { border:1px solid var(--border); border-top:3px solid var(--border); border-radius:8px; padding:10px 14px;
          min-width:130px; max-width:360px; text-align:center; background:var(--panel); }
  .card .cl { font-size:12px; color:var(--muted); }
  .card .cv { font-size:22px; font-weight:700; word-break:break-word; }
  .card .cs { font-size:11px; color:var(--faint); }
  .muted { color:var(--muted); font-size:12px; }
  .help { font-size:12px; color:var(--muted); margin:2px 0 6px; }
  .help span { cursor:help; text-decoration:underline dotted; }
  details.acc { border:1px solid var(--border); border-radius:8px; margin:6px 0; background:var(--panel); }
  details.acc > summary { padding:8px 12px; cursor:pointer; font-weight:600; list-style:none; }
  details.acc > summary::-webkit-details-marker { display:none; }
  details.acc > summary::before { content:'▸'; display:inline-block; width:1.2em; color:var(--muted); }
  details.acc[open] > summary::before { content:'▾'; }
  .acc-body { padding:8px 12px; border-top:1px solid var(--border); }
  .acc-body details.acc { background:var(--panel2); }
  .tw { overflow-x:auto; }
  table { border-collapse:collapse; width:100%; font-size:13px; }
  th, td { border:1px solid var(--border); padding:4px 8px; text-align:left; vertical-align:top; }
  th { background:var(--panel2); white-space:nowrap; }
  td.wrap { white-space:pre-wrap; }
  pre { white-space:pre-wrap; word-break:break-word; margin:0; padding:8px; border-radius:6px; background:var(--panel2);
        font:12.5px ui-monospace,Menlo,Consolas,monospace; max-height:60vh; overflow:auto; }
  .kv { display:grid; grid-template-columns:max-content 1fr; gap:2px 14px; font-size:13px; }
  .kv b { font-weight:600; } .kv code { font:12.5px ui-monospace,Menlo,Consolas,monospace; word-break:break-all; }
  p.para { margin:4px 0 8px; white-space:pre-wrap; }
  #turns { display:flex; flex-direction:column; }
  #turns.newest { flex-direction:column-reverse; }
  .toolbar { display:flex; gap:10px; align-items:center; margin:2px 0 8px; }
  .meta { font-size:12.5px; color:var(--muted); margin-bottom:6px; }
  table.act td { border:none; border-bottom:1px solid var(--border); }
  .live-title { font-size:12px; color:var(--muted); font-weight:700; margin:6px 2px 2px; }
  .dot-on { color:var(--ok); }
</style>
</head>
<body>
<header>
  <h1>Long-Horizon Soak Dashboard</h1>
  <select id="run" title="Run (config hash)"></select>
  <label class="follow" title="Switch automatically to the most recently active run"><input type="checkbox" id="follow"> follow latest</label>
  <span id="conn">connecting…</span>
  <span class="grow"></span>
  <button id="theme">🌙 / ☀️</button>
</header>
<div id="banner" hidden></div>
<div id="empty" class="muted" hidden></div>
<div id="live"><div class="cards" id="live-cards"></div><div id="cloud"></div></div>
<div id="stats"><div class="cards" id="stat-cards"></div><div class="cards" id="cost-cards"></div></div>
<div id="trans" class="help"></div>
<div id="sections"></div>
<script>
const HELP = __HELP__;
const API = '/api';
const S = { runs: [], runsSig: null, latest: null, baseLatest: null, run: null, rev: null, ov: null, live: null,
            follow: true, gen: 0, timer: null, openState: new Map(), turnEls: new Map(), sections: [], pendingTurn: null };

const h = (tag, props, ...kids) => {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (k === 'class') el.className = v;
    else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
    else if (v !== false && v !== null && v !== undefined) el.setAttribute(k, v === true ? '' : v);
  }
  el.append(...kids.flat(Infinity).filter(k => k !== null && k !== undefined && k !== false));
  return el;
};
const $ = id => document.getElementById(id);
const enc = encodeURIComponent;
const fmtDt = s => s < 60 ? `${Math.round(s)}s` : `${Math.floor(s / 60)}m ${Math.floor(s % 60)}s`;
const fmtTok = n => { n = +n || 0; return !n ? '0' : n >= 1e6 ? (n / 1e6).toFixed(1) + 'M' : n >= 1e3 ? Math.round(n / 1e3) + 'K' : String(Math.round(n)); };
const fmtTs = iso => iso ? String(iso).replace('T', ' ').slice(0, 19) : '—';
const nz = v => v === null || v === undefined ? '—' : v;
const pre = text => h('pre', null, text);

async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${url}`);
  return r.json();
}

function helpNote(caption, text) {
  return h('div', { class: 'help' }, h('span', { title: text }, `ℹ ${caption}`));
}

function table(headers, rows, wrapCols = []) {
  if (!rows.length) return h('div', { class: 'muted' }, '(none)');
  return h('div', { class: 'tw' }, h('table', null,
    h('thead', null, h('tr', null, headers.map(x => h('th', null, x)))),
    h('tbody', null, rows.map(r => h('tr', null, r.map((c, i) => h('td', { class: wrapCols.includes(i) ? 'wrap' : '' }, nz(c))))))));
}

function isOpen(key, def) { return S.openState.has(key) ? S.openState.get(key) : !!def; }

// Accordion whose body is built on first open and rebuilt when marked dirty.
function acc(key, title, build, def = false) {
  const sum = h('summary', null, title);
  const body = h('div', { class: 'acc-body' });
  const el = h('details', { class: 'acc' }, sum, body);
  const a = { el, sum, body, dirty: true };
  const render = () => {
    if (!a.dirty) return;
    a.dirty = false;
    body.replaceChildren(...[build()].flat(Infinity).filter(x => x !== null && x !== undefined && x !== false));
  };
  a.refresh = () => { a.dirty = true; if (el.open) render(); };
  el.addEventListener('toggle', () => { S.openState.set(key, el.open); if (el.open) render(); });
  el.open = isOpen(key, def);
  if (el.open) render();
  return a;
}

// ---- cards ---------------------------------------------------------------
const cards = new Map();
function setCard(parent, id, o) {
  let c = cards.get(id);
  if (!c) {
    c = { el: h('div', { class: 'card' }), l: h('div', { class: 'cl' }), v: h('div', { class: 'cv' }), s: h('div', { class: 'cs' }) };
    c.el.append(c.l, c.v, c.s);
    parent.append(c.el);
    cards.set(id, c);
  }
  c.l.textContent = o.label;
  c.v.textContent = o.value;
  c.s.textContent = o.sub || '';
  c.el.title = o.tip || '';
  c.el.style.borderTopColor = o.accent || '';
}

function renderLive() {
  const L = S.live, V = S.ov && S.ov.live;
  const parent = $('live-cards');
  if (!L) return;
  const sv = L.server;
  const sent = V ? V.local_sent : 0;
  if (!sv) {
    setCard(parent, 'local', { label: 'Local model (server)', value: '…', sub: 'probing', tip: HELP.local_server });
  } else if (sv.up) {
    const fh = V && V.last_fh ? ` · unbound history ~${fmtTok(V.last_fh)} tok` : '';
    setCard(parent, 'local', { label: 'Local model (server)', value: sv.model || 'online', accent: 'var(--ok)', tip: HELP.local_server,
      sub: `max ctx ${L.ctx} · last turn ~${fmtTok(sent)} tok sent${fh} · ${sv.loader || '?'} · ${sv.at}` });
  } else {
    setCard(parent, 'local', { label: 'Local model (server)', value: 'OFFLINE', accent: 'var(--bad)', tip: HELP.local_server,
      sub: `max ctx ${L.ctx} · ${sv.error || 'no response'} · ${sv.at}` });
  }

  if (V) {
    const models = V.cloud_models;
    setCard(parent, 'cloud', { label: 'Cloud model(s)', tip: HELP.cloud_models,
      value: models.map(m => m.name + (m.n > 1 ? ` ×${m.n}` : '')).join(' · ') || '—',
      sub: 'max ctx ' + models.map(m => `${m.name} ${m.ctx ?? '?'}`).join(' · ') + ` · last turn ~${fmtTok(V.cloud_sent)} tok sent` });

    const p = V.progress;
    const started = `started ${fmtTs(p.created_at)}`;
    if (p.turns_n > 0 && Number.isInteger(p.last_turn) && p.last_turn >= 0) {
      setCard(parent, 'progress', { label: 'Run progress', tip: HELP.progress, accent: 'var(--ok)',
        value: `turn ${Math.min(p.turns_n, p.total || p.turns_n)}/${p.total}`, sub: `last checkpoint t${p.last_turn} · ${started}` });
    } else {
      setCard(parent, 'progress', { label: 'Run progress', tip: HELP.progress, accent: 'var(--warn)',
        value: `0 / ${p.total}`, sub: `no turns yet · ${started}` });
    }
    const t = V.totals;
    setCard(parent, 'totals', { label: 'Usage totals', tip: HELP.usage_totals,
      value: `${t.cloud_calls} cloud / ${t.local_calls} local calls`,
      sub: `${fmtTok(t.cloud_prompt)} cloud · ${fmtTok(t.local_prompt)} local ctx tok` });
  }

  if (L.gpus.length) {
    setCard(parent, 'gpu', { label: 'GPU', value: `${L.gpus.length} device(s)`, accent: 'var(--purple)',
      sub: L.gpus.map(g => `GPU${g.index} ${g.util}% ${(g.mem_used / 1024).toFixed(1)}/${(g.mem_total / 1024).toFixed(1)} GB`).join(' · ') + ` · ${L.gpu_at}` });
  } else {
    setCard(parent, 'gpu', { label: 'GPU', value: 'n/a', sub: 'nvidia-smi unavailable' });
  }
  renderCloud(L);
}

function renderCloud(L) {
  const box = $('cloud');
  const at = new Date(L.now * 1000).toLocaleTimeString([], { hour12: false });
  if (!L.cloud.length) {
    box.replaceChildren(h('div', { class: 'muted', style: 'margin:6px 4px' }, `Cloud activity: no call in flight · ${at}`));
    return;
  }
  const color = { generating: 'var(--ok)', cooldown: 'var(--warn)', backoff: 'var(--bad)' };
  const rows = L.cloud.map(c => h('tr', null,
    h('td', null, h('b', { style: `color:${color[c.phase] || 'var(--muted)'}` }, c.role), h('br'),
      h('span', { class: 'muted' }, c.phase + (c.stale ? ' ⚠ stale?' : ''))),
    h('td', null, c.model),
    h('td', null, `${Math.round(c.elapsed)}s`, h('br'), h('span', { class: 'muted' }, c.rate ? `${Math.round(c.rate)} ch/s` : '')),
    h('td', null, `${c.content_chars.toLocaleString()} content · ${c.reasoning_chars.toLocaleString()} reasoning ch`),
    h('td', { class: 'muted' }, (c.note ? c.note + ' · ' : '') + (c.head && c.tail ? `${c.head} … ${c.tail}` : (c.head || c.tail || '—')))));
  box.replaceChildren(h('div', { class: 'live-title' }, `Cloud activity (live) · ${at}`), h('div', { class: 'tw' }, h('table', { class: 'act' }, h('tbody', null, rows))));
}

// ---- overview sections ---------------------------------------------------
function renderStats(ov) {
  const s = ov.stats, c = ov.cost, w = ov.last_wall;
  const sub = w ? `as of ${w}` : '';
  const P = $('stat-cards'), C = $('cost-cards');
  setCard(P, 's-ret', { label: 'Overall retention', value: `${Math.round(s.overall_retention)}%`, sub, tip: HELP.retention });
  setCard(P, 's-turns', { label: 'Turns run', value: String(ov.turns.length), sub });
  setCard(P, 's-notes', { label: 'Notes', value: String(s.notes_total), sub });
  setCard(P, 's-gf', { label: 'Guide failures', value: String(s.guide_failure), sub, tip: HELP.guide_failure });
  setCard(P, 's-dl', { label: 'DSS retention loss', value: String(s.dss_retention_loss), sub, tip: HELP.dss_retention_loss });
  setCard(P, 's-sup', { label: 'Superseded', value: String(s.supersession), sub, tip: HELP.superseded });
  setCard(C, 'c-lc', { label: 'Local LLM calls', value: String(c.local_calls), sub });
  setCard(C, 'c-lt', { label: 'Local ctx (prompt tok)', value: fmtTok(c.local_prompt), sub, tip: HELP.local_ctx });
  setCard(C, 'c-cc', { label: 'Cloud calls (guide+judge+audit)', value: String(c.cloud_calls), sub });
  setCard(C, 'c-ct', { label: 'Cloud ctx (prompt tok)', value: fmtTok(c.cloud_prompt), sub });
  setCard(C, 'c-wt', { label: 'Total wall time', value: fmtDt(c.dt_s), sub });

  const moves = [];
  for (const t of ov.transitions) for (const k of ['scene', 'chapter', 'arc']) if (t.moves[k]) moves.push(`turn ${t.turn}: ${k} ${t.moves[k][0]}→${t.moves[k][1]}`);
  $('trans').replaceChildren(...(moves.length ? [h('b', null, 'Scene / chapter / arc turns'), ` (scene-part budget splits): ${moves.join(', ')} `, h('span', { title: HELP.scene_part }, 'ℹ')] : []));
}

const para = (label, text) => text ? h('p', { class: 'para' }, label ? h('b', null, label + ' ') : null, String(text)) : null;
const list = (label, items) => items && items.length ? [h('b', null, label), h('ul', null, items.map(x => h('li', null, String(x))))] : null;

function planBlock(p, withRecalled) {
  const beats = (p.beats || []).map(b => withRecalled
    ? [b.turn, b.action || '', (b.note_ids || []).join(', ') || '—', b.recalled || '—']
    : [b.turn, b.action || '', (b.note_ids || []).join(', ') || '—']);
  return [para('Intent:', p.intent || '—'), para('Style reminder:', p.style_reminder),
          table(withRecalled ? ['+turn', 'action', 'note_ids', 'recalled'] : ['+turn', 'action', 'note_ids'], beats, [1])];
}

function buildSections() {
  const defs = [
    ['cfg', 'Run configuration', false, ov => h('div', { class: 'kv' }, Object.entries(ov.args).sort().flatMap(([k, v]) =>
      [h('b', null, k), h('code', null, typeof v === 'object' && v !== null ? JSON.stringify(v) : String(v))]))],
    ['plan', 'Live plan', false, ov => ov.plans.length ? [
      h('div', { class: 'muted' }, "The guide's own short-term plan, refreshed on replan turns and edited from the spreadsheet outline. Expand a replan to see the beats it set."),
      ov.plans.map(p => acc(`plan:${p.created_turn}`, `Replan @ turn ${p.created_turn}`, () => planBlock(p, true)).el)]
      : 'No live plan in this run (replan disabled or no replan turns).'],
    ['roll', 'Rolling overviews (mid-run judge)', false, ov => ov.rollings.length ? ov.rollings.map(o =>
      acc(`roll:${o.turns}`, `${o.turns ?? '?'} turns — ${o.overall_score ?? 'n/a'}/5 (${o.trajectory ?? 'n/a'}) · ${(o.generated_at || '').slice(11, 19)}`, () => [
        para('Verdict:', o.verdict || '—'), para('', o.summary), para('Memory fidelity:', o.memory_fidelity), para('Style consistency:', o.style_consistency)]).el)
      : 'No rolling overviews in this run (opt-in via --overview-every N; 0 disables).'],
    ['final', 'Final overview', false, ov => {
      const f = ov.final;
      if (!f) return 'No final overview in this run (end-of-run judge pass missing — re-run with --final-overview <run_dir> to backfill).';
      if (f.error) return `Final overview failed to generate — ${f.error}. Re-run with --final-overview <run_dir> to retry.`;
      const notes = (f.notes || []).map(n => typeof n === 'object' ? [n.id, n.status, n.detail || ''] : [n, '', '']);
      return [para('', `Overall score: ${f.overall_score ?? 'n/a'}/5 — Verdict: ${f.verdict || '—'} · written at ${(f.generated_at || '').slice(11, 19) || '—'}`),
        para('Arc progression:', f.arc_progression), para('Style consistency:', f.style_consistency), para('Memory fidelity:', f.memory_fidelity),
        list('Strengths:', f.strengths), list('Weaknesses:', f.weaknesses), notes.length ? table(['id', 'status', 'detail'], notes, [2]) : null, para('', f.summary)];
    }],
    ['audits', 'DSS save-beat audits', false, ov => ov.audits.length ? [
      helpNote('audit statuses', `${HELP.audit_statuses} ${HELP.unassessable}`), table(['beat', 'turn', 'at', 'status', 'detail'], ov.audits, [4])]
      : 'No DSS save-beat audits in this run (spreadsheet has no dss_beats, or the auditor never ran).'],
    ['curves', 'Judge curves', true, ov => [
      helpNote('style / fidelity / quality', `${HELP.style}\n${HELP.fidelity}\n${HELP.quality}`),
      table(['turn', 'at', 'style', 'fidelity', 'quality'], ov.curves.map(c => [c.turn, c.wall, c.style, c.fidelity, c.quality]))]],
    ['types', 'Retention by type', true, ov => [
      h('div', { class: 'muted' }, `Computed at ${ov.last_wall || '—'} (last checkpoint).`),
      helpNote('retention terms', `${HELP.retention}\nguide-fail = ${HELP.guide_failure}\ndss-loss = ${HELP.dss_retention_loss}\nsuperseded = ${HELP.superseded}`),
      table(['type', 'count', 'survived', 'guide-fail', 'dss-loss', 'superseded', 'rate %'], ov.types.map(r => [...r.slice(0, 6), r[6].toFixed(1)]))]],
    ['notes', 'Per-note retention', true, ov => [
      h('div', { class: 'muted' }, `Planted / last-seen turn@time. Computed at ${ov.last_wall || '—'}.`),
      helpNote('retention per note', HELP.retention),
      table(['id', 'type', 'specificity', 'needle', 'planted (turn@time)', 'last_seen (turn@time)', 'survived', 'attribution'],
        ov.notes.map(r => [r.id, r.type, r.specificity, r.needle,
          `${nz(r.planted_at)}${r.planted_t ? ' @' + r.planted_t : ''}`, `${nz(r.last_seen)}${r.last_seen_t ? ' @' + r.last_seen_t : ''}`,
          r.survived ? 'Y' : 'n', r.attribution]))]],
  ];
  const root = $('sections');
  for (const [key, title, def, build] of defs) {
    const a = acc(`sec:${key}`, title, () => S.ov ? build(S.ov) : [], def);
    S.sections.push(a);
    root.append(a.el);
  }
  const toolbar = h('div', { class: 'toolbar' },
    h('span', { class: 'muted' }, 'Expand any turn for the exact user input, reply, instruction prompt, judge result, probes, recalls and state snapshot.'),
    h('span', { style: 'flex:1' }),
    h('button', { id: 'order', onclick: ev => { const on = $('turns').classList.toggle('newest'); localStorage.setItem('soak-newest', on ? '1' : '0'); ev.target.textContent = on ? 'Newest first' : 'Oldest first'; } }, 'Oldest first'));
  S.turnsAcc = acc('sec:turns', 'Per-turn detail', () => [toolbar, h('div', { id: 'turns' })], true);
  root.append(S.turnsAcc.el);
  S.turnsAcc.refresh = () => {};
}

// ---- per-turn list -------------------------------------------------------
function turnTitle(r) {
  let s = `Turn ${r.turn} · ${fmtDt(r.dt_s)}`;
  if (r.judge && !r.judge.error) s += ` · style ${nz(r.judge.style)} fid ${nz(r.judge.fidelity)}`;
  for (const k of ['scene', 'chapter', 'arc']) if (r.trans && r.trans[k]) s += ` · ⛨ ${k.toUpperCase()} ${r.trans[k][0]}→${r.trans[k][1]}`;
  return s;
}

function makeTurn(row) {
  const t = { row, detailV: null, body: h('div', { class: 'acc-body' }), sum: h('summary') };
  t.el = h('details', { class: 'acc', id: `turn-${row.turn}` }, t.sum, t.body);
  t.el.addEventListener('toggle', () => {
    if (t.el.open) { loadTurn(t); writeHash(t.row.turn); }
    else if (S.openTurn === t.row.turn) writeHash(null);
  });
  return t;
}

function updateTurn(t, row) {
  t.row = row;
  t.sum.textContent = turnTitle(row);
  if (t.el.open && t.detailV !== null && t.detailV !== row.v) loadTurn(t);
}

async function loadTurn(t) {
  const gen = S.gen, v = t.row.v;
  if (t.detailV === v) return;
  const d = await getJSON(`${API}/runs/${enc(S.run)}/turns/${t.row.turn}`);
  if (gen !== S.gen) return;
  t.detailV = v;
  t.body.replaceChildren(...renderDetail(d));
}

function renderDetail(d) {
  const K = name => `t${d.turn}:${name}`;
  const lo = d.local, cl = d.cloud;
  const peak = lo.peak && lo.ctx ? ` · peak ctx ${fmtTok(lo.peak)}/${lo.ctx} (${Math.round(100 * lo.peak / lo.ctx)}%)` : lo.peak ? ` · peak prompt ${fmtTok(lo.peak)}` : '';
  const hist = lo.history ? ` · history (unbound) ${fmtTok(lo.history)}${lo.ctx ? `/${lo.ctx} (${Math.round(100 * lo.history / lo.ctx)}%)` : ''}` : ' · history (unbound) —';
  const stateFiles = d.state.map(r => r[0]);
  return [
    h('div', { class: 'meta' }, `ts ${d.ts} · at ${d.wall || '—'} · local ${lo.calls} calls (${fmtTok(lo.prompt_tokens)} tok)${peak}${hist} · cloud ${cl.calls} calls (${fmtTok(cl.prompt_tokens)} tok)`),
    acc(K('user'), 'User input', () => pre(d.user)).el,
    acc(K('reply'), 'Assistant reply', () => pre(d.reply)).el,
    acc(K('instr'), 'Instruction prompt', () => pre(d.instr_prompt)).el,
    acc(K('plan'), 'Live plan snapshot', () => d.plan && Object.keys(d.plan).length ? planBlock(d.plan, false) : '(no plan at this turn)').el,
    acc(K('judge'), 'Judge result', () => d.judge.length ? [helpNote('judge scores', `style = ${HELP.style}\nfidelity = ${HELP.fidelity}\nquality = ${HELP.quality}`), table(['field', 'value'], d.judge, [1])] : 'No judge run for this turn').el,
    acc(K('probes'), 'Probes', () => d.probes.length ? [helpNote('probe', HELP.probe), table(['note', 'needle', 'present', 'matched path'], d.probes)] : 'No probes for this turn').el,
    acc(K('recalls'), 'Recalls', () => d.recalls.length ? [helpNote('recall', HELP.recall), table(['note', 'echoed', 'text-only', 'matched syns'], d.recalls)] : 'No recalls due this turn').el,
    acc(K('audit'), 'DSS save-beat audit', () => d.audit.length ? [helpNote('audit verdict', HELP.unassessable), table(['beat', 'turn', 'status', 'detail'], d.audit, [3])] : 'No DSS save-beat audit this turn').el,
    acc(K('state'), 'State snapshot', () => {
      if (!d.state.length) return 'No state snapshot for this turn';
      const out = h('div');
      const view = h('div');
      const sel = h('select', { onchange: async () => {
        if (!sel.value) return view.replaceChildren();
        const r = await fetch(`${API}/runs/${enc(S.run)}/turns/${d.turn}/file/${enc(sel.value)}`);
        view.replaceChildren(pre(r.ok ? await r.text() : 'file not found'));
      } }, h('option', { value: '' }, '(choose a file to view)'), stateFiles.map(f => h('option', { value: f }, f)));
      out.append(table(['file', 'entries'], d.state), h('div', { style: 'margin:8px 0' }, sel), view);
      return out;
    }).el,
  ];
}

function renderTurns(ov) {
  const list = $('turns');
  if (!list) return;
  const seen = new Set();
  let prev = null;
  for (const row of ov.turns) {
    seen.add(row.turn);
    let t = S.turnEls.get(row.turn);
    if (!t) {
      t = makeTurn(row);
      S.turnEls.set(row.turn, t);
      if (prev) prev.el.after(t.el); else list.prepend(t.el);
      if (S.pendingTurn === row.turn) { t.el.open = true; S.pendingTurn = null; t.el.scrollIntoView(); }
    }
    updateTurn(t, row);
    prev = t;
  }
  for (const [n, t] of S.turnEls) if (!seen.has(n)) { t.el.remove(); S.turnEls.delete(n); }
}

function renderOverview() {
  const ov = S.ov;
  renderStats(ov);
  for (const a of S.sections) a.refresh();
  renderTurns(ov);
  renderLive();
  const p = ov.live.progress;
  document.title = `Soak · ${ov.manifest.spreadsheet || ov.id} · ${p.turns_n}/${p.total}`;
}

// ---- runs, selection, polling -------------------------------------------
function runLabel(r) {
  return `${r.active ? '● ' : ''}${r.spreadsheet} — ${r.genre} · ${fmtTs(r.created_at)} · ${r.turns_done}/${r.total}`;
}

function renderRunMenu() {
  const sel = $('run');
  sel.replaceChildren(...S.runs.map(r => h('option', { value: r.id }, runLabel(r))));
  if (S.run) sel.value = S.run;
}

async function loadRuns() {
  const d = await getJSON(`${API}/runs`);
  S.runs = d.runs; S.runsSig = d.sig; S.latest = d.latest; S.runsDir = d.runs_dir;
  renderRunMenu();
  const empty = $('empty');
  empty.hidden = d.runs.length > 0;
  empty.textContent = `No run directories found under ${d.runs_dir} yet. Start the soak; this page will pick it up.`;
}

function renderBanner() {
  const b = $('banner');
  const show = !S.follow && S.latest && S.latest !== S.run && S.latest !== S.baseLatest;
  b.hidden = !show;
  if (!show) return;
  const r = S.runs.find(x => x.id === S.latest);
  b.replaceChildren(h('span', null, `A newer run is active: ${r ? r.spreadsheet + ' · ' + fmtTs(r.created_at) : S.latest}`),
    h('button', { onclick: () => selectRun(S.latest) }, 'View'),
    h('button', { onclick: () => { S.baseLatest = S.latest; renderBanner(); } }, 'Dismiss'));
}

function writeHash(turn) {
  S.openTurn = turn;
  const p = new URLSearchParams();
  if (S.run && !S.follow) p.set('run', S.run);
  if (turn !== null && turn !== undefined) p.set('turn', turn);
  history.replaceState(null, '', p.toString() ? '#' + p : location.pathname);
}

function resetView() {
  S.gen++;
  S.rev = null; S.ov = null; S.openTurn = null;
  S.turnEls.clear();
  for (const k of [...S.openState.keys()]) if (!k.startsWith('sec:')) S.openState.delete(k);
  const list = $('turns');
  if (list) list.replaceChildren();
}

async function selectRun(id) {
  resetView();
  S.run = id;
  S.baseLatest = S.latest;
  $('run').value = id;
  writeHash(null);
  renderBanner();
  await loadOverview();
}

async function loadOverview() {
  const gen = S.gen;
  const ov = await getJSON(`${API}/runs/${enc(S.run)}/overview`);
  if (gen !== S.gen) return;
  S.ov = ov; S.rev = ov.rev;
  renderOverview();
}

function setConn(ok) {
  const c = $('conn');
  c.textContent = ok ? '● live' : '✕ server unreachable';
  c.className = ok ? 'dot-on' : 'bad';
}

async function pickInitial() {
  const id = S.wantRun && S.runs.some(r => r.id === S.wantRun) ? S.wantRun : S.latest;
  if (id) await selectRun(id);
}

async function poll() {
  clearTimeout(S.timer);
  if (!document.hidden) {
    try {
      const d = await getJSON(`${API}/poll` + (S.run ? `?run=${enc(S.run)}` : ''));
      setConn(true);
      S.latest = d.latest;
      if (d.runs_sig !== S.runsSig) await loadRuns();
      if (S.follow && S.latest && S.latest !== S.run) {
        await selectRun(S.latest);
      } else if (!S.run) {
        await pickInitial();
      } else if (S.run && d.rev === null && S.latest) {
        await selectRun(S.latest);
      } else {
        S.live = d.live;
        renderLive();
        renderBanner();
        if (S.run && d.rev !== null && d.rev !== S.rev) await loadOverview();
      }
    } catch (e) {
      setConn(false);
    }
  }
  S.timer = setTimeout(poll, 1000);
}

async function boot() {
  const hp = new URLSearchParams(location.hash.slice(1));
  S.wantRun = hp.get('run');
  S.follow = !S.wantRun;
  S.pendingTurn = hp.get('turn') !== null ? Number(hp.get('turn')) : null;
  $('follow').checked = S.follow;
  $('follow').addEventListener('change', ev => {
    S.follow = ev.target.checked;
    writeHash(S.openTurn);
    if (S.follow && S.latest && S.latest !== S.run) selectRun(S.latest);
  });
  $('run').addEventListener('change', ev => {
    S.follow = false;
    $('follow').checked = false;
    selectRun(ev.target.value);
  });
  $('theme').addEventListener('click', () => {
    const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    localStorage.setItem('soak-theme', next);
  });
  document.addEventListener('visibilitychange', () => { if (!document.hidden) poll(); });

  buildSections();
  if (localStorage.getItem('soak-newest') !== '0') { $('turns').classList.add('newest'); $('order').textContent = 'Newest first'; }
  try {
    await loadRuns();
    await pickInitial();
  } catch (e) { setConn(false); }
  poll();
}
boot();
</script>
</body>
</html>
"""

PAGE = PAGE_TEMPLATE.replace("__HELP__", json.dumps(HELP))


def main() -> int:
    global REGISTRY, SAMPLER
    ap = argparse.ArgumentParser(description="DSS long-horizon soak dashboard")
    ap.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR,
                    help="directory holding run outputs (default: tests/runs)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7861)
    args = ap.parse_args()

    REGISTRY = Registry(args.runs_dir)
    SAMPLER = Sampler()
    SAMPLER.start()
    server = Server((args.host, args.port), Handler)
    print(f"soak dashboard on http://{args.host}:{args.port}  (runs: {args.runs_dir})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
