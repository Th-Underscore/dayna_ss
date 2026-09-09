"""Standalone Gradio dashboard for the dayna_ss long-horizon soak.

Launches a local web UI that inspects the ``tests/runs/<config_hash>/`` run
directories produced by ``long_horizon_soak.py``. Raw responses stay hidden
behind accordions — the top-level view is computed stats, and every exact
payload (user input, reply, instruction prompt, judge result, probes, state
snapshot) is one click away without flooding the screen.

Run::

    python soak_dashboard.py [--runs-dir tests/runs] [--port 7861]

Requires: gradio (already a text-generation-webui dependency).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import html
import json
import re
import subprocess
import time
import urllib.request
from pathlib import Path

import gradio as gr

TEST_DIR = Path(__file__).parent
REPO_ROOT = TEST_DIR.parent.parent.parent
DEFAULT_RUNS_DIR = TEST_DIR / "runs"
DEFAULT_LOCAL_BASE = "http://0.0.0.0:5000/v1"

# Cloud model metadata for the live-stats panel: max context tokens, and
# input/output cost in USD per million tokens (from models.dev, where known).
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

_CLOUD_BASE = "https://opencode.ai/zen/go/v1"
_BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
               "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")

# Plain-text help for (i) tooltips. Kept as one dict so every term is defined
# in one place; cards render it as a native title attr, tables as info=.
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
}

_DARK_CSS = """
body.tg-dark {
  --body-background-fill: #0f1419;
  --body-text-color: #dbe1e8;
  --body-text-color-subdued: #8b98a5;
  --background-fill-primary: #0f1419;
  --background-fill-secondary: #1a232c;
  --block-background-fill: #1a232c;
  --block-border-color: #2b3a47;
  --block-title-text-color: #e7edf3;
  --block-title-background-fill: #1a232c;
  --block-label-text-color: #b9c6d2;
  --block-label-background-fill: #1a232c;
  --block-info-text-color: #8b98a5;
  --panel-background-fill: #1a232c;
  --panel-border-color: #2b3a47;
  --input-background-fill: #131a21;
  --input-background-fill-focus: #1a232c;
  --input-background-fill-hover: #131a21;
  --input-border-color: #2b3a47;
  --input-border-color-focus: #3b4c5e;
  --input-border-color-hover: #2b3a47;
  --input-placeholder-color: #6b7a88;
  --table-even-background-fill: #1a232c;
  --table-odd-background-fill: #161e26;
  --table-text-color: #dbe1e8;
  --table-border-color: #2b3a47;
  --table-row-focus: #22303d;
  --button-primary-background-fill: #2f6f4f;
  --button-primary-background-fill-hover: #3a8a62;
  --button-primary-text-color: #eef4f0;
  --button-primary-text-color-hover: #ffffff;
  --button-secondary-background-fill: #1a232c;
  --button-secondary-background-fill-hover: #22303d;
  --button-secondary-border-color: #2b3a47;
  --button-secondary-border-color-hover: #3b4c5e;
  --button-secondary-text-color: #dbe1e8;
  --button-secondary-text-color-hover: #ffffff;
  --border-color-primary: #2b3a47;
  --link-text-color: #6cb2eb;
  --link-text-color-hover: #8ec9f5;
  --link-text-color-visited: #6cb2eb;
  --checkbox-background-color: #131a21;
  --checkbox-background-color-selected: #2f6f4f;
  --checkbox-border-color: #2b3a47;
  --checkbox-label-background-fill: #1a232c;
  --checkbox-label-background-fill-hover: #22303d;
  --checkbox-label-text-color: #dbe1e8;
  --checkbox-label-text-color-selected: #e7edf3;
  --code-background-fill: #161e26;
  --color-accent: #6cb2eb;
  --color-accent-soft: #22303d;
  --accordion-text-color: #e7edf3;
  color-scheme: dark;
}
body.tg-dark .label-wrap,
body.tg-dark .label-wrap span,
body.tg-dark button.label-wrap {
  color: #e7edf3 !important;
}
body.tg-dark .label-wrap:hover,
body.tg-dark .label-wrap:hover span {
  color: #ffffff !important;
}
body.tg-dark .soak-card { border-color: #2b3a47 !important; }
body.tg-dark .soak-card div { color: inherit; }
body.tg-dark .soak-card div[style*="color:#888"] { color: #8b98a5 !important; }
body.tg-dark .soak-card div[style*="color:#aaa"] { color: #6b7a88 !important; }
body.tg-dark .soak-card div[style*="font-size:22px"] { color: #e7edf3 !important; }
body.tg-dark .soak-card div[style*="font-size:26px"] { color: #e7edf3 !important; }
body.tg-dark .soak-help { color: #8b98a5; }
body.tg-dark #tg-dark-toggle { border-color: #2b3a47 !important; }
"""


def _now() -> str:
    return _dt.datetime.now().strftime("%H:%M:%S")


def _full_ts(iso: str) -> str:
    """Render a manifest ISO timestamp like '2026-08-10T14:50:37' as a friendly string."""
    if not iso:
        return "—"
    try:
        return _dt.datetime.fromisoformat(iso).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return iso


def _local_api_key() -> str:
    """Resolve the local server API key (same pattern as the soak harness)."""
    import os
    key = os.environ.get("DSS_BENCH_API_KEY", "").strip()
    if key:
        return key
    auth_path = Path.home() / ".local" / "share" / "opencode" / "auth.json"
    if auth_path.exists():
        try:
            auth = json.loads(auth_path.read_text(encoding="utf-8"))
            for name in ("localhost", "textgen", "llama"):
                e = auth.get(name)
                if isinstance(e, dict) and e.get("key"):
                    return e["key"]
                if isinstance(e, str):
                    return e
        except Exception:
            pass
    return "not-needed"


def _local_ctx_from_flags() -> int:
    """Read --n_ctx from the webui's CMD_FLAGS_m.txt (authoritative server ctx)."""
    for p in (REPO_ROOT / "CMD_FLAGS_m.txt",):
        try:
            m = re.search(r"--n_ctx\s+(\d+)", p.read_text(encoding="utf-8"))
            if m:
                return int(m.group(1))
        except Exception:
            pass
    return 65536


def _local_server_status(base: str) -> dict:
    """Probe the local model server: up/down, model id, loader."""
    out = {"up": False, "model": None, "loader": None, "error": None}
    key = _local_api_key()
    for path, into in (("models", "model"), ("internal/model/info", "loader")):
        try:
            req = urllib.request.Request(
                base.rstrip("/") + "/" + path,
                headers={"Authorization": f"Bearer {key}", "User-Agent": _BROWSER_UA},
            )
            data = json.loads(urllib.request.urlopen(req, timeout=4).read().decode("utf-8"))
            if into == "model":
                out["up"] = True
                out["model"] = data.get("data", [{}])[0].get("id") if data.get("data") else None
            else:
                out["loader"] = data.get("loader")
        except Exception as e:
            if into == "model":
                out["error"] = str(e)
    return out


def _gpu_stats() -> list[dict]:
    """nvidia-smi one-liner per GPU (util%, mem used/total). Empty on failure."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=6,
        ).stdout.strip().splitlines()
        rows = []
        for line in out:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                rows.append({"index": parts[0], "name": parts[1], "util": parts[2],
                             "mem_used": parts[3], "mem_total": parts[4]})
        return rows
    except Exception:
        return []

# --------------------------------------------------------------------------- #
# data loading
# --------------------------------------------------------------------------- #


def _json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def list_runs(runs_dir: Path) -> list[tuple[str, str]]:
    """Return (run_id, label) for every run directory, newest created first."""
    out = []
    if not runs_dir.exists():
        return out
    dirs = [d for d in runs_dir.iterdir() if d.is_dir()
            and (d / "manifest.json").exists() and not d.name.startswith("turn")]

    def created(d: Path) -> float:
        man = _json(d / "manifest.json")
        ts = man.get("created_at") or ""
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
            try:
                return _dt.strptime(ts, fmt).timestamp()
            except Exception:
                continue
        try:
            return d.stat().st_mtime
        except Exception:
            return 0.0

    for d in sorted(dirs, key=created, reverse=True):
        man = _json(d / "manifest.json")
        label = f"{man.get('spreadsheet', d.name)} — {man.get('genre', '?')} · {man.get('created_at', '?')}"
        out.append((d.name, label))
    return out


def _turns(run_dir: Path) -> list[dict]:
    """Load and sort all turn_XXX result dicts (with their run_dir)."""
    turns = []
    for d in sorted(run_dir.glob("turn_*")):
        if not d.is_dir():
            continue
        r = _json(d / "result.json")
        if not r:
            continue
        r["_dir"] = str(d)
        turns.append(r)
    turns.sort(key=lambda r: r.get("turn", 0))
    return turns


def _probe_present(r: dict, note_id: str) -> bool:
    probe = (r.get("probes") or {}).get(note_id)
    return bool(probe and probe.get("present"))


def _per_note_timeline(run_dir: Path, spreadsheet: dict) -> list[dict]:
    """Recompute per-note retention from raw turn results (report-agnostic)."""
    turns = _turns(run_dir)
    n = len(turns)
    notes = spreadsheet.get("notes", [])
    wall = _turn_wall_map(turns)
    rows = []
    for note in notes:
        nid = note["id"]
        present = [_probe_present(r, nid) for r in turns]
        observed = [i for i, ok in enumerate(present) if ok]
        last_seen = observed[-1] if observed else None
        survived = bool(observed and last_seen == n - 1)
        planted = (note.get("plant") or {}).get("turn")
        if note["type"] == "supersession" and not observed:
            attribution = "guide_failure"
        elif not observed:
            attribution = "guide_failure" if planted is not None else "not_planted"
        elif note["type"] == "supersession":
            attribution = "supersession"
        elif observed[-1] != n - 1:
            attribution = "dss_retention_loss"
        else:
            attribution = "survived"
        planted_t = wall.get(planted, "") if planted is not None else ""
        last_t = wall.get(last_seen, "") if last_seen is not None else ""
        rows.append({
            "id": nid,
            "type": note["type"],
            "specificity": note.get("specificity", ""),
            "needle": note.get("needle", ""),
            "planted_at": planted if planted is not None else "—",
            "planted_t": planted_t,
            "last_seen": last_seen if last_seen is not None else "—",
            "last_seen_t": last_t,
            "survived": survived,
            "attribution": attribution,
            "present": present,
        })
    return rows


def _summary_stats(rows: list[dict]) -> dict:
    notes = len(rows)
    survived = sum(1 for r in rows if r["attribution"] == "survived")
    attribution = {
        "guide_failure": sum(1 for r in rows if r["attribution"] == "guide_failure"),
        "dss_retention_loss": sum(1 for r in rows if r["attribution"] == "dss_retention_loss"),
        "supersession": sum(1 for r in rows if r["attribution"] == "supersession"),
        "not_planted": sum(1 for r in rows if r["attribution"] == "not_planted"),
    }
    return {
        "notes_total": notes,
        "overall_retention": survived / max(1, notes) * 100.0,
        **attribution,
    }


def _judge_curves(run_dir: Path) -> dict:
    style, qual, fid = [], [], []
    for r in _turns(run_dir):
        j = r.get("judge")
        if not j or "error" in j:
            continue
        t = r["turn"]
        if j.get("style_score") is not None:
            style.append((t, j.get("style_score")))
        if j.get("quality_score") is not None:
            qual.append((t, j.get("quality_score")))
        if j.get("memory_fidelity") is not None:
            fid.append((t, j.get("memory_fidelity")))
    return {"style": style, "quality": qual, "fidelity": fid}


def _usage_series(turns: list[dict]) -> list[dict]:
    """Per-turn model usage, handling both result formats.

    New runs (since the dashboard live-stats work) carry per-turn deltas in
    ``local_usage`` / ``cloud_usage`` / ``judge_usage`` / ``auditor_usage``.
    Older runs carried cumulative counters that reset on resume (so naive
    summation overcounts badly) — reconstruct per-turn deltas from the
    cumulative series, treating a decrease as a process restart.
    """
    series = []
    prev_l = prev_cc = prev_c = 0
    for r in turns:
        lu = r.get("local_usage")
        cu = r.get("cloud_usage")
        if lu is not None:
            lp = int(lu.get("prompt_tokens", 0) or 0)
            if not lp:
                lp = int((r.get("instr_prompt") or "").strip().__len__() / 4)
            series.append({
                "local_calls": int(lu.get("calls", 0) or 0),
                "local_prompt": lp,
                "cloud_calls": int((cu or {}).get("calls", 0) or 0),
                "cloud_prompt": int((cu or {}).get("prompt_tokens", 0) or 0),
            })
            prev_l = prev_c = 0
            continue
        # legacy: cumulative counters that reset on resume. The reset is visible
        # in the calls counter (it can decrease); apply the boundary to all counters.
        l = int(r.get("local_calls", 0) or 0)
        cc = int((r.get("cloud_usage") or {}).get("calls", 0) or 0)
        c = int((r.get("cloud_usage") or {}).get("prompt_tokens", 0) or 0)
        reset = (prev_cc and cc < prev_cc) or (prev_l and l < prev_l)
        dl = l if reset else (l - prev_l)
        dcc = cc if reset else (cc - prev_cc)
        dc = c if reset else (c - prev_c)
        series.append({
            "local_calls": dl,
            "local_prompt": int((r.get("instr_prompt") or "").strip().__len__() / 4),
            "cloud_calls": dcc,
            "cloud_prompt": dc,
        })
        prev_l, prev_cc, prev_c = l, cc, c
    return series


def _turn_wall_map(turns: list[dict]) -> dict:
    """turn number -> wall-clock HH:MM:SS (from wall_ts, else file mtime)."""
    out = {}
    for r in turns:
        t = r.get("turn")
        wall = r.get("wall_ts") or ""
        if wall:
            try:
                out[t] = _dt.datetime.fromisoformat(wall).strftime("%H:%M:%S")
            except Exception:
                out[t] = wall[-8:]
        else:
            d = Path(r.get("_dir") or "")
            try:
                out[t] = _dt.datetime.fromtimestamp(d.stat().st_mtime).strftime("%H:%M:%S")
            except Exception:
                out[t] = ""
    return out


def _cost(run_dir: Path) -> dict:
    total = {"local_calls": 0, "local_prompt": 0, "dt_s": 0.0, "cloud_calls": 0,
             "prompt": 0, "completion": 0, "total_tokens": 0}
    for r in _turns(run_dir):
        total["dt_s"] += r.get("dt_s", 0) or 0
    for u in _usage_series(_turns(run_dir)):
        total["local_calls"] += u["local_calls"]
        total["local_prompt"] += u["local_prompt"]
        total["cloud_calls"] += u["cloud_calls"]
        total["cloud_prompt"] = total.get("cloud_prompt", 0) + u["cloud_prompt"]
    return total


def _fmt_dt(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    return f"{int(seconds // 60)}m {int(seconds % 60)}s"


def _scalar_table(obj: dict) -> list[list]:
    return [[k, str(v)] for k, v in obj.items() if not isinstance(v, (dict, list))]


def _render_html(text: str) -> str:
    return f"<pre style='white-space:pre-wrap;font-family:monospace'>{html.escape(text)}</pre>"


def _help_html(caption: str, title_text: str) -> str:
    """A muted (i) marker with a native hover tooltip — gr.Dataframe has no info= in
    gradio 4.37, so tables get their term explanations this way."""
    safe_title = html.escape(title_text)
    safe_caption = html.escape(caption)
    print (f"<div class='soak-help' style='font-size:12px;color:#888;margin:2px 0 4px;white-space:pre-line;'>"
            f"<span style='cursor:help;text-decoration:underline dotted' "
            f"title='{safe_title}'>ℹ {safe_caption}</span></div>")
    return (f"<div class='soak-help' style='font-size:12px;color:#888;margin:2px 0 4px;white-space:pre-line;'>"
            f"<span style='cursor:help;text-decoration:underline dotted' "
            f"title='{safe_title}'>ℹ {safe_caption}</span></div>")



# --------------------------------------------------------------------------- #
# per-turn accordions
# --------------------------------------------------------------------------- #


def _judge_table(j: dict) -> list[list]:
    if not j:
        return [["No judge run for this turn"]]
    if "error" in j:
        return [["error", str(j["error"])]]
    rows = [
        ["style_score", j.get("style_score")],
        ["quality_score", j.get("quality_score")],
        ["memory_fidelity", j.get("memory_fidelity")],
        ["summary", j.get("summary", "")],
    ]
    for n in j.get("notes", []):
        rows.append([f"note:{n.get('id')}", f"{n.get('status')} — {n.get('detail')}"])
    return rows


def _probe_table(probes: dict) -> list[list]:
    if not probes:
        return [["No probes for this turn"]]
    out = []
    for nid, p in sorted(probes.items()):
        needle = p.get("needle", "")
        paths = p.get("paths", [])
        rendered = []
        for entry in paths:
            if isinstance(entry, (list, tuple)):
                rendered.append("/".join(str(seg) for seg in entry if seg))
            elif isinstance(entry, str):
                rendered.append(entry)
        target = "; ".join(rendered) if rendered else "—"
        out.append([nid, needle, "YES" if p.get("present") else "no", target])
    return out


def _recalls_table(recalls: dict) -> list[list]:
    if not recalls:
        return [["No recalls due this turn"]]
    out = []
    for nid, info in sorted(recalls.items()):
        out.append([nid,
                    "YES" if info.get("echoed") else "no",
                    "text-only" if info.get("echo_by_text_only") else "—",
                    ", ".join(info.get("matched_syns", [])) or "—"])
    return out


def _audit_table(audits: list) -> list[list]:
    if not audits:
        return [["No DSS save-beat audit this turn"]]
    out = []
    for a in audits:
        out.append([a.get("id"), a.get("turn"), a.get("status", "?"), a.get("detail", "")])
    return out


def _state_table(run_dir_str: str) -> tuple[list[list], list[str]]:
    """Return (state file table, list of available file paths)."""
    d = Path(run_dir_str) / "state_snapshot"
    if not d.exists():
        return [["No state snapshot for this turn"]], []
    rows, files = [], []
    for f in sorted(d.glob("*.json")):
        data = _json(f)
        n_keys = len(data) if isinstance(data, dict) else 1
        rows.append([f.name, n_keys])
        files.append(str(f))
    ctx = Path(run_dir_str) / "context.txt"
    if ctx.exists():
        rows.append(["context.txt", "text"])
        files.append(str(ctx))
    return rows, files


_TRANSITION_KINDS = (("scene", "_scene_number"), ("chapter", "_chapter_number"), ("arc", "_arc_number"))
_TRANSITION_LABELS = {"scene": "SCENE", "chapter": "CHAPTER", "arc": "ARC"}


def _transitions(run_dir: Path) -> dict[int, dict[str, tuple[int, int]]]:
    """turn_no -> {kind: (from, to)} for scene/chapter/arc transition turns,
    read from each turn's state_snapshot/current_scene.json `_*_number`."""
    trans: dict[int, dict[str, tuple[int, int]]] = {}
    prev: dict[str, int | None] = {kind: None for kind, _ in _TRANSITION_KINDS}
    for d in sorted(run_dir.glob("turn_*")):
        if not d.is_dir():
            continue
        nos: dict[str, int | None] = {kind: None for kind, _ in _TRANSITION_KINDS}
        p = d / "state_snapshot" / "current_scene.json"
        if p.exists():
            data = _json(p)
            if isinstance(data, dict):
                for kind, key in _TRANSITION_KINDS:
                    v = data.get(key)
                    nos[kind] = v if isinstance(v, int) else None
        try:
            t = int(d.name.split("_")[1])
        except Exception:
            continue
        moved: dict[str, tuple[int, int]] = {}
        for kind, _ in _TRANSITION_KINDS:
            if prev[kind] is not None and nos[kind] is not None and nos[kind] != prev[kind]:
                moved[kind] = (prev[kind], nos[kind])
        if moved:
            trans[t] = moved
        for kind, _ in _TRANSITION_KINDS:
            if nos[kind] is not None:
                prev[kind] = nos[kind]
    return trans


def _build_turn_accordion(r: dict, transitions: dict[int, dict[str, tuple[int, int]]] | None = None) -> gr.Accordion:
    dir_str = r.get("_dir", "")
    turn_no = r.get("turn", "?")
    style_fid = ""
    j = r.get("judge")
    if j and "error" not in j:
        style_fid = f"  ·  style {j.get('style_score')}  fid {j.get('memory_fidelity')}"
    wall = r.get("wall_ts") or ""
    try:
        wall = _dt.datetime.fromisoformat(wall).strftime("%H:%M:%S") if wall else ""
    except Exception:
        wall = ""
    trans_tag = ""
    try:
        mv = transitions.get(int(turn_no)) if transitions else None
    except Exception:
        mv = None
    if mv:
        for kind, _ in _TRANSITION_KINDS:
            v = mv.get(kind)
            if v:
                trans_tag += f"  ·  ⛨ {_TRANSITION_LABELS[kind]} {v[0]}→{v[1]}"
    with gr.Accordion(f"Turn {turn_no}  ·  {_fmt_dt(r.get('dt_s', 0) or 0)}{style_fid}{trans_tag}",
                      open=False, elem_id=f"soak-turn-{turn_no}"):
        with gr.Row():
            local_u = (r.get("local_usage") or {})
            cloud_u = (r.get("cloud_usage") or {})
            peak = int(local_u.get("max_prompt_tokens", 0) or 0)
            ctx = int(local_u.get("ctx_size", 0) or 0)
            ctx_txt = ""
            if peak and ctx:
                ctx_txt = f"  ·  **peak ctx** {_fmt_tokens(peak)}/{ctx} ({100.0 * peak / ctx:.0f}%)"
            elif peak:
                ctx_txt = f"  ·  **peak prompt** {_fmt_tokens(peak)}"
            fh = int(local_u.get("full_history_tokens", 0) or 0)
            if fh:
                if ctx:
                    hist_txt = (f"  ·  **history (unbound)** {_fmt_tokens(fh)}/{ctx} "
                                f"({100.0 * fh / ctx:.0f}%)")
                else:
                    hist_txt = f"  ·  **history (unbound)** {_fmt_tokens(fh)}"
            else:
                hist_txt = "  ·  **history (unbound)** —"
            gr.Markdown(f"**ts** {r.get('ts', '—')}  ·  **at** {wall or '—'}  ·  "
                        f"**local** {local_u.get('calls', r.get('local_calls', 0))} calls "
                        f"({_fmt_tokens(local_u.get('prompt_tokens', 0))} tok){ctx_txt}{hist_txt}  ·  "
                        f"**cloud** {cloud_u.get('calls', '?')} calls "
                        f"({_fmt_tokens(cloud_u.get('prompt_tokens', 0))} tok)")
        with gr.Accordion("User input", open=False):
            gr.HTML(value=_render_html(_text(Path(dir_str) / "user.txt") or r.get("user_input", "")))
        with gr.Accordion("Assistant reply", open=False):
            gr.HTML(value=_render_html(_text(Path(dir_str) / "reply.txt") or r.get("reply", "")))
        with gr.Accordion("Instruction prompt", open=False):
            gr.HTML(value=_render_html(r.get("instr_prompt", "")))
        with gr.Accordion("Live plan snapshot", open=False):
            plan = r.get("plan") or _json(Path(dir_str) / "plan.json") or {}
            if plan:
                gr.Markdown(f"**intent:** {plan.get('intent') or '—'}")
                if plan.get("style_reminder"):
                    gr.Markdown(f"**style_reminder:** {plan['style_reminder']}")
                beat_rows = []
                for b in plan.get("beats", []):
                    note = ", ".join(b.get("note_ids", [])) or "—"
                    beat_rows.append([b.get("turn"), b.get("action", ""), note])
                gr.Dataframe(value=beat_rows or [["(no beats)"]],
                             headers=["+turn", "action", "note_ids"],
                             type="array", interactive=False)
            else:
                gr.Markdown("(no plan at this turn)")
        with gr.Accordion("Judge result", open=False):
            gr.HTML(value=_help_html("judge scores",
                                     f"style = {HELP['style']}\n"
                                     f"fidelity = {HELP['fidelity']}\n"
                                     f"quality = {HELP['quality']}"))
            gr.Dataframe(value=_judge_table(j), headers=["field", "value"],
                         type="array", interactive=False)
        with gr.Accordion("Probes", open=False):
            gr.HTML(value=_help_html("probe",
                                     "A probe checks whether a planted needle phrase is "
                                     "present in DSS's private memory state at that turn."))
            gr.Dataframe(value=_probe_table(r.get("probes", {})),
                         headers=["note", "needle", "present", "matched path"],
                         type="array", interactive=False)
        with gr.Accordion("Recalls", open=False):
            gr.HTML(value=_help_html("recall", HELP["recall"]))
            gr.Dataframe(value=_recalls_table(r.get("recalls", {})),
                         headers=["note", "echoed", "text-only", "matched syns"],
                         type="array", interactive=False)
        with gr.Accordion("DSS save-beat audit", open=False):
            gr.HTML(value=_help_html("audit verdict", HELP["unassessable"]))
            gr.Dataframe(value=_audit_table(r.get("audit", []) or []),
                         headers=["beat", "turn", "status", "detail"],
                         type="array", interactive=False)
        with gr.Accordion("State snapshot", open=False):
            state_rows, state_files = _state_table(dir_str)
            gr.Dataframe(value=state_rows, headers=["file", "entries"],
                         type="array", interactive=False)
            if state_files:
                state_dd = gr.Dropdown(choices=state_files, label="View state file",
                                       value=state_files[0] if state_files else None)
                state_out = gr.HTML()
                state_dd.change(fn=_load_state_file, inputs=state_dd,
                                outputs=state_out)
    return None


def _plan_timeline(run_dir: Path) -> list[tuple[int, dict]]:
    """Replan turn -> plan dict, from turn snapshots (created_turn marks replans)."""
    out = []
    for d in sorted(run_dir.glob("turn_*")):
        if not d.is_dir():
            continue
        p = _json(d / "plan.json")
        if not p:
            continue
        created = p.get("created_turn")
        if isinstance(created, int) and created >= 0:
            if not out or out[-1][0] != created:
                out.append((created, p))
    return out


def _load_state_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return "<i>file not found</i>"
    raw = p.read_text(encoding="utf-8", errors="ignore")
    try:
        parsed = json.loads(raw)
    except Exception:
        return _render_html(raw)
    return _render_html(json.dumps(parsed, indent=2))


def _run_audits(run_dir: Path) -> list[list]:
    """Aggregate audit verdicts across all turns (the full audit log, newest last)."""
    rows: list[list] = []
    for d in sorted(run_dir.glob("turn_*")):
        r = _json(d / "result.json")
        for a in r.get("audit", []) or []:
            rows.append([a.get("id", "?"), a.get("turn", "?"), a.get("status", "?"),
                         a.get("detail", "")])
    return rows


# --------------------------------------------------------------------------- #
# live system stats
# --------------------------------------------------------------------------- #


def _fmt_tokens(n) -> str:
    if not n:
        return "0"
    n = float(n)
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.0f}K"
    return f"{int(n)}"


def _live_stats_html(run_id: str | None, runs_dir: Path) -> str:
    """Build the top-of-page live system status card block (HTML string).

    Shown is: local model server (up/down, model, ctx max + last-turn ctx sent),
    the run's cloud models + their ctx max + last-turn ctx sent, run progress
    (turn x/y, elapsed), GPU load, and token/cost totals — each with a
    timestamp of when it was measured.
    """
    now = _now()
    cards: list[str] = []

    def _card(label: str, value: str, sub: str = "", accent: str = "#222", tip: str = "") -> str:
        title = f" title='{tip}'" if tip else ""
        return (f"<div class='soak-card'{title} style='border:1px solid #ddd;border-radius:8px;"
                "padding:10px 14px;"
                "margin:4px;min-width:150px;text-align:center;border-top:3px solid " + accent + "'>"
                f"<div style='font-size:11px;color:#888'>{label}</div>"
                f"<div style='font-size:22px;font-weight:700'>{value}</div>"
                f"<div style='font-size:11px;color:#aaa'>{sub}</div></div>")

    # ---- per-turn usage (for "current ctx sent" = last turn's prompt tokens) ----
    man = _json(runs_dir / run_id / "manifest.json") if run_id else {}
    turns = _turns(runs_dir / run_id) if run_id else []
    usage = _usage_series(turns)
    last_usage = usage[-1] if usage else {}

    # ---- local model ------------------------------------------------------
    local_base = (man.get("args") or {}).get("local_base") or DEFAULT_LOCAL_BASE
    ls = _local_server_status(local_base)
    local_ctx = _local_ctx_from_flags()
    local_sent = int(last_usage.get("local_prompt", 0) or 0)
    last_fh = int((turns[-1].get("local_usage") or {}).get("full_history_tokens", 0) or 0) if turns else 0
    if ls["up"]:
        local_val = ls["model"] or "online"
        local_accent = "#2a9d5f"
        fh_sub = f" · unbound history ~{_fmt_tokens(last_fh)} tok" if last_fh else ""
        local_sub = (f"max ctx {local_ctx} · last turn ~{_fmt_tokens(local_sent)} tok sent"
                     f"{fh_sub} · {ls['loader'] or '?'} · {now}")
    else:
        local_val = "OFFLINE"
        local_accent = "#c0392b"
        local_sub = f"max ctx {local_ctx} · {ls['error'] or 'no response'} · {now}"
    cards.append(_card("Local model (server)", local_val, local_sub, local_accent,
                       "The LMDeploy server DSS talks to. Max ctx from the server flags; "
                       "'last turn ~N tok sent' is the estimated prompt size of the last turn, "
                       "'unbound history ~N tok' is the whole accumulated conversation "
                       "(untruncated by --max-update-history) estimate."))

    # ---- cloud models -----------------------------------------------------
    args = man.get("args") or {}
    cloud_models = [m for m in (args.get("guide_model"), args.get("judge_model"),
                                args.get("auditor_model")) if m]
    cloud_sent = int(last_usage.get("cloud_prompt", 0) or 0)
    unique = dict.fromkeys(cloud_models)
    cloud_val = " · ".join(unique) or "—"
    cloud_sub = (f"max ctx " + " · ".join(f"{m} {MODEL_INFO.get(m, {}).get('ctx', '?')}" for m in unique)
                 + f" · last turn ~{_fmt_tokens(cloud_sent)} tok sent · {len(unique)} role(s) · {now}")
    cards.append(_card("Cloud model(s)", cloud_val, cloud_sub, tip=(
        "The guide, judge, and auditor cloud models (roles in parentheses below). "
        "Max context per model; 'last turn ~N tok sent' is that turn's prompt size.")))

    # ---- run progress -----------------------------------------------------
    if run_id:
        rd = runs_dir / run_id
        turns_n = len(turns)
        total = args.get("turns") or (args.get("smoke") or 0)
        last_turn = man.get("last_turn")
        if isinstance(last_turn, int) and last_turn >= 0 and turns_n > 0:
            prog = f"turn {min(turns_n, total)}/{total}"
            sub = f"last checkpoint t{last_turn} · started {_full_ts(man.get('created_at'))} · {now}"
            accent = "#2a9d5f"
        else:
            prog = "0 / " + str(total)
            sub = f"no turns yet · started {_full_ts(man.get('created_at'))} · {now}"
            accent = "#e67e22"
        cards.append(_card("Run progress", prog, sub, accent,
                           "Turns completed of the configured total, plus the last "
                           "checkpoint turn and run start time."))
    else:
        cards.append(_card("Run progress", "—", "select a run · " + now, "#888"))

    # ---- total cloud usage (per-turn deltas summed, legacy-reset aware) ----
    tot = _cost(runs_dir / run_id) if run_id else {"cloud_calls": 0, "cloud_prompt": 0,
                                                   "local_calls": 0, "local_prompt": 0}
    cards.append(_card("Usage totals",
                       f"{tot['cloud_calls']} cloud / {tot['local_calls']} local calls",
                       f"{_fmt_tokens(tot['cloud_prompt'])} cloud · {_fmt_tokens(tot['local_prompt'])} local ctx tok · {now}",
                       tip="Cumulative LLM usage for the whole run (per-turn deltas, resume-aware)."))

    # ---- GPU ---------------------------------------------------------------
    gpus = _gpu_stats()
    if gpus:
        gpu_lines = " · ".join(
            f"GPU{g['index']} {g['util']}% {_fmt_tokens(int(g['mem_used']) * 1e6)}/{_fmt_tokens(int(g['mem_total']) * 1e6)}"
            for g in gpus)
        cards.append(_card("GPU", f"{len(gpus)} device(s)", gpu_lines + " · " + now, "#8e44ad"))
    else:
        cards.append(_card("GPU", "n/a", "nvidia-smi unavailable · " + now, "#888"))

    return ("<div style='display:flex;flex-wrap:wrap'>" + "".join(cards) + "</div>"
            + _cloud_activity_html(run_id, runs_dir))


def _cloud_activity_html(run_id: str | None, runs_dir: Path) -> str:
    """Live in-flight cloud-call table (reads {run_dir}/live/cloud_*.json).

    The soak harness streams every cloud call when status reporting is enabled
    and keeps one small JSON per in-flight call (role, phase, elapsed, chars
    streamed, head/tail snippet); the file is removed when the call finishes.
    """
    if not run_id:
        return ""
    live_dir = runs_dir / run_id / "live"
    if not live_dir.is_dir():
        return ""
    now = time.time()
    rows = []
    try:
        files = sorted(live_dir.glob("cloud_*.json"))
    except Exception:
        files = []
    for f in files:
        try:
            st = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        role = html.escape(str(st.get("role", "?")))
        model = html.escape(str(st.get("model", "?")))
        phase = str(st.get("phase", "generating"))
        age = now - float(st.get("updated_at", 0))
        stale = age > 45
        elapsed = float(st.get("elapsed_s", 0) or 0)
        n_c = int(st.get("content_chars", 0) or 0)
        n_r = int(st.get("reasoning_chars", 0) or 0)
        rate = f"{(n_c + n_r) / elapsed:.0f} ch/s" if elapsed > 1 else ""
        note = html.escape(str(st.get("note", "")))
        head = html.escape(str(st.get("head", ""))[-90:])
        tail = html.escape(str(st.get("tail", ""))[:90])
        snippet = (head + " … " + tail) if head and tail else (head or tail or "—")
        phase_color = {"generating": "#2a9d5f", "cooldown": "#e67e22",
                       "backoff": "#c0392b"}.get(phase, "#888")
        stale_tag = " ⚠ stale?" if stale else ""
        sub = f"{note} · " if note else ""
        rows.append(
            f"<tr>"
            f"<td style='padding:3px 10px'><b style='color:{phase_color}'>{role}</b>"
            f"<br><span style='font-size:10px;color:#888'>{phase}{stale_tag}</span></td>"
            f"<td style='padding:3px 10px'>{model}</td>"
            f"<td style='padding:3px 10px'>{elapsed:.0f}s<br>"
            f"<span style='font-size:10px;color:#888'>{rate}</span></td>"
            f"<td style='padding:3px 10px'>{n_c:,} content · {n_r:,} reasoning ch</td>"
            f"<td style='padding:3px 6px;font-size:11px;color:#aaa;max-width:520px;"
            f"overflow:hidden;text-overflow:ellipsis;white-space:nowrap'>"
            f"{sub}{snippet}</td></tr>")
    if not rows:
        return ("<div style='font-size:11px;color:#888;margin:6px 4px'>"
                "Cloud activity: no call in flight · " + _now() + "</div>")
    table = ("<table style='border-collapse:collapse;width:100%;font-size:13px'>"
             + "".join(rows) + "</table>")
    return ("<div style='margin-top:8px'>"
            "<div style='font-size:12px;color:#888;font-weight:700;margin:2px 4px'>"
            f"Cloud activity (live) · {_now()}</div>{table}</div>")


# --------------------------------------------------------------------------- #
# main UI builder
# --------------------------------------------------------------------------- #


def _run_display(run_id: str, runs_dir: Path):
    """Render the full dashboard for one run id."""
    run_dir = runs_dir / run_id
    man = _json(run_dir / "manifest.json")
    spreadsheet_id = man.get("spreadsheet", "")
    spreadsheet = _json(TEST_DIR / "spreadsheets" / f"{spreadsheet_id}.json")
    if not spreadsheet:
        spreadsheet = _json(run_dir / ".." / ".." / "spreadsheets" / f"{spreadsheet_id}.json")
        if not spreadsheet:
            spreadsheet = _json(TEST_DIR / "spreadsheets" / f"{run_dir.parent.name}.json")

    rows = _per_note_timeline(run_dir, spreadsheet) if spreadsheet else []
    summary = _summary_stats(rows) if rows else {"notes_total": 0, "overall_retention": 0.0,
                                                  "guide_failure": 0, "dss_retention_loss": 0,
                                                  "supersession": 0, "not_planted": 0}
    curves = _judge_curves(run_dir)
    turns = _turns(run_dir)
    scene_turns = _transitions(run_dir)
    cost = _cost(run_dir)
    wall = _turn_wall_map(turns)
    last_wall = wall.get(max(turns, key=lambda r: r.get("turn", 0)).get("turn"), "") if turns else ""

    args_html = "<br>".join(f"<b>{k}</b>: <code>{v}</code>"
                            for k, v in sorted((man.get("args") or {}).items()))
    args_box = gr.HTML(value=args_html)

    def _timed_card(label: str, value: str, at: str = "", tip: str = "") -> str:
        title = f" title='{tip}'" if tip else ""
        return (f"<div class='soak-card'{title} style='border:1px solid #ddd;border-radius:8px;"
                "padding:10px 14px;"
                "margin:4px;min-width:120px;text-align:center'>"
                f"<div style='font-size:12px;color:#888'>{label}</div>"
                f"<div style='font-size:26px;font-weight:700'>{value}</div>"
                f"<div style='font-size:11px;color:#aaa'>{at}</div></div>")

    stat_boxes = [
        ("Overall retention", f"{summary['overall_retention']:.0f}%", last_wall, HELP["retention"]),
        ("Turns run", f"{len(turns)}", last_wall, ""),
        ("Notes", f"{summary['notes_total']}", last_wall, ""),
        ("Guide failures", f"{summary.get('guide_failure', 0)}", last_wall, HELP["guide_failure"]),
        ("DSS retention loss", f"{summary.get('dss_retention_loss', 0)}", last_wall, HELP["dss_retention_loss"]),
        ("Superseded", f"{summary.get('supersession', 0)}", last_wall, HELP["superseded"]),
    ]
    cost_boxes = [
        ("Local LLM calls", f"{cost['local_calls']}", last_wall, ""),
        ("Local ctx (prompt tok)", _fmt_tokens(cost.get("local_prompt", 0)), last_wall,
         "Total prompt tokens DSS sent to the local model (server reports 0, so this is an estimate)."),
        ("Cloud guide calls", f"{cost['cloud_calls']}", last_wall, ""),
        ("Cloud ctx (prompt tok)", _fmt_tokens(cost.get("cloud_prompt", 0)), last_wall, ""),
        ("Total wall time", _fmt_dt(cost["dt_s"]), last_wall, ""),
    ]

    def _card(label: str, value: str) -> str:
        return ("<div style='border:1px solid #ddd;border-radius:8px;padding:10px 14px;"
                "margin:4px;min-width:120px;text-align:center'>"
                f"<div style='font-size:12px;color:#888'>{label}</div>"
                f"<div style='font-size:26px;font-weight:700'>{value}</div></div>")

    gr.HTML(value="<div style='display:flex;flex-wrap:wrap'>"
                  + "".join(_timed_card(l, v, at, tip) for l, v, at, tip in stat_boxes) + "</div>")
    gr.HTML(value="<div style='display:flex;flex-wrap:wrap'>"
                  + "".join(_timed_card(l, v, at, tip) for l, v, at, tip in cost_boxes) + "</div>")

    if scene_turns:
        parts = []
        for t, mv in sorted(scene_turns.items()):
            for kind, _ in _TRANSITION_KINDS:
                if kind in mv:
                    a, b = mv[kind]
                    parts.append(f"turn {t}: {kind} {a}→{b}")
        gr.Markdown(f"**Scene / chapter / arc turns** (scene-part budget splits): {', '.join(parts)}  \n"
                    f"<span style='color:#888;font-size:12px'>{HELP['scene_part']}</span>")

    with gr.Accordion("Run configuration", open=False):
        args_box

    plan_tl = _plan_timeline(run_dir)
    with gr.Accordion("Live plan", open=False):
        if not plan_tl:
            gr.Markdown("No live plan in this run (replan disabled or no replan turns).")
        else:
            gr.Markdown("The guide's own short-term plan, refreshed on replan turns "
                        "and edited from the spreadsheet outline. Expand a replan to "
                        "see the beats it set for the next ~15 exchanges.")
            for created_turn, p in plan_tl:
                with gr.Accordion(f"Replan @ turn {created_turn}", open=False):
                    gr.Markdown(f"**Intent:** {p.get('intent') or '—'}")
                    if p.get("style_reminder"):
                        gr.Markdown(f"**Style reminder:** {p['style_reminder']}")
                    beat_rows = []
                    for b in p.get("beats", []):
                        note = ", ".join(b.get("note_ids", [])) or "—"
                        recalled = b.get("recalled") or "—"
                        beat_rows.append([b.get("turn"), b.get("action", ""), note, recalled])
                    gr.Dataframe(value=beat_rows or [["(no beats)"]],
                                 headers=["+turn", "action", "note_ids", "recalled"],
                                 type="array", interactive=False)

    with gr.Accordion("Rolling overviews (mid-run judge)", open=False):
        rollings = sorted(run_dir.glob("rolling_overview_*.json"))
        if not rollings:
            gr.Markdown("No rolling overviews in this run (opt-in via `--overview-every N`; "
                        "0 disables).")
        else:
            for ov_path in rollings:
                ov = _json(ov_path)
                if not ov:
                    continue
                gen_at = ""
                if ov.get("generated_at"):
                    try:
                        gen_at = _dt.datetime.fromisoformat(ov["generated_at"]).strftime("%H:%M:%S")
                    except Exception:
                        gen_at = str(ov["generated_at"])
                with gr.Accordion(f"{ov.get('turns', '?')} turns — "
                                  f"{ov.get('overall_score', 'n/a')}/5 "
                                  f"({ov.get('trajectory', 'n/a')})  ·  {gen_at}",
                                  open=False):
                    gr.Markdown(f"**Verdict:** {ov.get('verdict') or '—'}")
                    if ov.get("summary"):
                        gr.Markdown(ov["summary"])
                    if ov.get("memory_fidelity"):
                        gr.Markdown(f"**Memory fidelity:** {ov['memory_fidelity']}")
                    if ov.get("style_consistency"):
                        gr.Markdown(f"**Style consistency:** {ov['style_consistency']}")

    with gr.Accordion("Final overview", open=False):
        fo = _json(run_dir / "final_overview.json")
        if not fo:
            gr.Markdown("No final overview in this run (end-of-run judge pass missing — "
                        "re-run with `--final-overview <run_dir>` to backfill).")
        elif "error" in fo:
            gr.Markdown(f"**Final overview failed to generate** — {fo['error']}\n\n"
                        "Re-run with `--final-overview <run_dir>` to retry (now uses a 6000-token "
                        "budget with reasoning disabled).")
        else:
            gen_at = ""
            if fo.get("generated_at"):
                try:
                    gen_at = _dt.datetime.fromisoformat(fo["generated_at"]).strftime("%H:%M:%S")
                except Exception:
                    gen_at = str(fo["generated_at"])
            gr.Markdown(f"**Overall score:** {fo.get('overall_score', 'n/a')}/5 — "
                        f"**Verdict:** {fo.get('verdict') or '—'}  ·  "
                        f"**written at** {gen_at or '—'}")
            for key, title in (("arc_progression", "Arc progression"),
                               ("style_consistency", "Style consistency"),
                               ("memory_fidelity", "Memory fidelity")):
                if fo.get(key):
                    gr.Markdown(f"**{title}:** {fo[key]}")
            if fo.get("strengths"):
                gr.Markdown("**Strengths:**\n" + "\n".join(f"- {x}" for x in fo["strengths"]))
            if fo.get("weaknesses"):
                gr.Markdown("**Weaknesses:**\n" + "\n".join(f"- {x}" for x in fo["weaknesses"]))
            fo_notes = fo.get("notes")
            if fo_notes:
                fo_note_rows = [[n.get("id"), n.get("status"), n.get("detail", "")] if isinstance(n, dict)
                                else [n, "", ""] for n in fo_notes]
                gr.Dataframe(value=fo_note_rows, headers=["id", "status", "detail"],
                             type="array", interactive=False)
            if fo.get("summary"):
                gr.Markdown(fo["summary"])

    with gr.Accordion("DSS save-beat audits", open=False):
        audit_rows = _run_audits(run_dir)
        if not audit_rows:
            gr.Markdown("No DSS save-beat audits in this run (spreadsheet has no `dss_beats`, "
                        "or the auditor never ran).")
        else:
            audit_rows_out = []
            for a in audit_rows:
                t = a[1]
                aout = [a[0], t] + list(a[2:])
                if isinstance(t, int):
                    aout.insert(2, wall.get(t, ""))
                else:
                    aout.insert(2, "")
                audit_rows_out.append(aout)
            gr.HTML(value=_help_html("audit statuses",
                                     "Statuses: saved / partial / wrong / missing / unassessable. "
                                     + HELP["unassessable"]))
            gr.Dataframe(value=audit_rows_out,
                         headers=["beat", "turn", "at", "status", "detail"],
                         type="array", interactive=False)

    with gr.Accordion("Judge curves", open=True):
        curve_rows = []
        max_t = max([t for t, _ in curves["style"]] + [t for t, _ in curves["fidelity"]]
                    + [t for t, _ in curves["quality"]] or [0])
        for t in range(max_t + 1):
            s = dict(curves["style"]).get(t, "—")
            q = dict(curves["quality"]).get(t, "—")
            f = dict(curves["fidelity"]).get(t, "—")
            curve_rows.append([t, wall.get(t, ""), s, f, q])
        gr.HTML(value=_help_html("style / fidelity / quality",
                                 f"{HELP['style']}\n{HELP['fidelity']}\n{HELP['quality']}"))
        gr.Dataframe(value=curve_rows, headers=["turn", "at", "style", "fidelity", "quality"],
                     type="array", interactive=False)

    with gr.Accordion("Retention by type", open=True):
        gr.Markdown(f"Computed at **{last_wall or '—'}** (last checkpoint).")
        typed = {}
        for r in rows:
            typed.setdefault(r["type"], {"count": 0, "survived": 0, "guide_failure": 0,
                                         "dss": 0, "superseded": 0})
            t = typed[r["type"]]
            t["count"] += 1
            t["survived"] += int(r["attribution"] == "survived")
            t["guide_failure"] += int(r["attribution"] == "guide_failure")
            t["dss"] += int(r["attribution"] == "dss_retention_loss")
            t["superseded"] += int(r["attribution"] == "supersession")
        type_rows = [[rt, t["count"], t["survived"], t["guide_failure"], t["dss"], t["superseded"],
                      (t["survived"] / t["count"] * 100 if t["count"] else 0)]
                     for rt, t in sorted(typed.items())]
        gr.HTML(value=_help_html("retention terms",
                                 f"{HELP['retention']}\n"
                                 f"guide-fail = {HELP['guide_failure']}\n"
                                 f"dss-loss = {HELP['dss_retention_loss']}\n"
                                 f"superseded = {HELP['superseded']}"))
        gr.Dataframe(value=type_rows,
                     headers=["type", "count", "survived", "guide-fail", "dss-loss", "superseded", "rate %"],
                     type="array", interactive=False)

    with gr.Accordion("Per-note retention", open=True):
        gr.Markdown(f"Planted / last-seen turn@time. Computed at **{last_wall or '—'}**.")
        note_rows = [[r["id"], r["type"], r["specificity"], r["needle"],
                      f"{r['planted_at']}" + (f" @{r['planted_t']}" if r["planted_t"] else ""),
                      f"{r['last_seen']}" + (f" @{r['last_seen_t']}" if r["last_seen_t"] else ""),
                      "Y" if r["survived"] else "n", r["attribution"]]
                     for r in rows]
        gr.HTML(value=_help_html("retention per note", HELP["retention"]))
        gr.Dataframe(value=note_rows,
                     headers=["id", "type", "specificity", "needle", "planted (turn@time)",
                              "last_seen (turn@time)", "survived", "attribution"],
                     type="array", interactive=False)

    with gr.Accordion("Per-turn detail", open=True):
        gr.Markdown("Expand any turn to view the exact user input, assistant reply, "
                    "instruction prompt, judge result, probes, recalls, and the full state snapshot.")
        for r in turns:
            _build_turn_accordion(r, scene_turns)

    return None


def build_dashboard(runs_dir: Path) -> gr.Blocks:
    runs = list_runs(runs_dir)
    choices = [rid for rid, _ in runs]

    with gr.Blocks(title="DSS Long-Horizon Soak Dashboard", css=_DARK_CSS) as demo:
        with gr.Row():
            gr.Markdown("# Long-Horizon Soak Dashboard")
            gr.HTML(value=(
                "<button id='tg-dark-toggle' onclick='tgToggle()' "
                "style='border:1px solid #ddd;border-radius:8px;padding:6px 12px;"
                "background:transparent;cursor:pointer;margin:10px'>🌙 Dark mode</button>"))

        # Live system stats (top, refreshed on demand / on an interval)
        live_html = gr.HTML(value="", elem_id="soak-live-stats")

        with gr.Row():
            run_menu = gr.Dropdown(choices=choices, label="Run (config hash)",
                                   value=choices[0] if choices else None, scale=4)
            refresh_btn = gr.Button("Refresh", scale=1)
        live_refresh_btn = gr.Button("↻", elem_id="soak-live-refresh",
                                     visible=False, min_width=1)

        def refresh_runs():
            runs_now = list_runs(runs_dir)
            return gr.update(choices=[rid for rid, _ in runs_now],
                             value=[rid for rid, _ in runs_now][0] if runs_now else None)

        def update_live(run_id):
            return _live_stats_html(run_id, runs_dir)

        # live stats update quietly (no "loading" spinner overlay)
        refresh_btn.click(fn=refresh_runs, inputs=[], outputs=[run_menu], show_progress="hidden")
        refresh_btn.click(fn=update_live, inputs=[run_menu], outputs=[live_html], show_progress="hidden")
        run_menu.change(fn=update_live, inputs=[run_menu], outputs=[live_html], show_progress="hidden")
        live_refresh_btn.click(fn=update_live, inputs=[run_menu], outputs=[live_html], show_progress="hidden")
        demo.load(fn=update_live, inputs=[run_menu], outputs=[live_html], show_progress="hidden")

        @gr.render(inputs=[run_menu])
        def render_run(run_id):
            # gr.render re-invokes this on run_menu change, so selecting a
            # different run truly rebuilds the dashboard below.
            if not run_id:
                gr.Markdown("No run directories found under "
                            f"`{runs_dir}` yet. Run the long-horizon soak first, "
                            "then press **Refresh**.")
                return
            _run_display(run_id, runs_dir)

        # Auto-refresh the live stats every 10s via a tiny injected interval.
        demo.load(fn=None, js="""
        function() {
          setInterval(function() {
            const el = document.getElementById('soak-live-refresh');
            if (el) el.click();
          }, 10000);
          return [];
        }
        """)

        # Dark-mode toggle: persist in localStorage, apply on load.
        demo.load(fn=None, js="""
        function() {
          function tgApply() {
            if (localStorage.getItem('tg-dark') === '1') {
              document.body.classList.add('tg-dark');
            } else {
              document.body.classList.remove('tg-dark');
            }
          }
          function tgToggle() {
            document.body.classList.toggle('tg-dark');
            localStorage.setItem('tg-dark',
              document.body.classList.contains('tg-dark') ? '1' : '0');
          }
          window.tgToggle = tgToggle;
          tgApply();
          return [];
        }
        """)

    return demo


def main() -> int:
    ap = argparse.ArgumentParser(description="DSS long-horizon soak dashboard")
    ap.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR,
                    help="directory holding run outputs (default: tests/runs)")
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    demo = build_dashboard(args.runs_dir)
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=args.port,
                share=args.share, inbrowser=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
