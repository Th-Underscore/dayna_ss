"""Long-horizon soak orchestrator.

Runs one continuous multi-genre conversation through the real dayna_ss engine on
a live local model, while a cloud model (OpenCode Go) guides the story and judges
memory fidelity. See ``docs/plans/long_horizon_soak_plan.md`` §5-§8 for the design.

Usage::

    python long_horizon_soak.py --spreadsheet spreadsheets/noir_detective.json
        --turns 100 --guide-style implicit --specificity-profile mixed
        [--smoke 5] [--resume runs/<dir>] [--level 2] [--director on]

Environment (see plan §0):
    DSS_BENCH_BASE_URL   local OpenAI-compatible server (default http://127.0.0.1:5001/v1)
    DSS_BENCH_API_KEY    local server key
    DSS_BENCH_MODEL      local model id
    OPENCODE_GO_API_KEY  cloud key (or ~/.local/share/opencode/auth.json)

Levels:
    1 — per-subject ``DataSummarizer.generate`` (fast; smoke path, no retrieval)
    2 — full production ``summarize_latest_state`` (retrieval + chunking; default)

Exit codes: 0 clean finish (or clean checkpoint), 1 any failure.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import queue
import re
import shutil
import signal
import sys
import tempfile
import threading
import time
import traceback
import urllib.request
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any


# ----------------------------------------------------------------- logging ---

_LOGFILE: Path | None = None
_LOGFILE_FH = None


def _open_logfile(path: Path) -> None:
    """Open the dedicated soak logfile (append)."""
    global _LOGFILE, _LOGFILE_FH
    _LOGFILE = path
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _LOGFILE_FH = open(path, "a", encoding="utf-8")
    except Exception as e:
        print(f"[soak] WARN: cannot open logfile {path}: {e}")
        _LOGFILE_FH = None


def soak_log(msg: str, ts: bool = True) -> None:
    """Line to stdout AND the dedicated logfile, both timestamped. Thread-safe."""
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S") if ts else ""
    line = f"[{stamp}] {msg}" if stamp else msg
    with _LOG_LOCK:
        try:
            print(line, flush=True)
        except Exception:
            pass
        if _LOGFILE_FH is not None:
            try:
                _LOGFILE_FH.write(line + "\n")
                _LOGFILE_FH.flush()
            except Exception:
                pass


_LOG_LOCK = threading.Lock()


_SOAK_TURN = 0
_SOAK_JSONL_PATH: str | None = os.environ.get("SOAK_JSONL_PATH")
_SOAK_JSONL_LOCK = threading.Lock()


def _soak_log_call(phase_id: str, step_id: str, ttft: float, total: float, pt: int, ct: int, cached_pt: int = 0) -> None:
    """Per-local-call latency line: real TTFT + full round-trip + token counts.

    ``ttft`` is time-to-first-token as observed by the harness: the interval from
    request issue until the first streamed delta carrying content/reasoning
    (``_complete`` requests ``stream: true`` and parses the SSE chunks). Under
    concurrent workers it includes server-side queue time, not just prefill.
    ``total`` is the full round-trip (prefill + decode + transport + queue) from
    request issue to the [DONE] frame. Maps soak calls to the LMDeploy server log
    on the ``input_tokens`` (approximate prefill rate = pt/total).
    ``cached_pt`` is the number of prompt tokens served from the prefix cache
    (vLLM ``prompt_tokens_details.cached_tokens``); 0 when unavailable.
    """
    label = f"{phase_id or '?'}/{step_id or 'generate'}"
    cache_str = f" cached={cached_pt}" if cached_pt else " "
    if ttft is not None:
        soak_log(f"  local-call {label}: ttft={ttft:.2f}s total={total:.2f}s "
                 f"in={pt} out={ct} tok{cache_str} ({(pt / total / 1000):.1f}K tok/s)")
    else:
        soak_log(f"  local-call {label}: ttft=— total={total:.2f}s "
                 f"in={pt} out={ct} tok{cache_str} ({(pt / total / 1000):.1f}K tok/s)")
    # Structured JSONL sidecar (opt-in via SOAK_JSONL_PATH env var)
    if _SOAK_JSONL_PATH:
        rec = {"ts": round(time.time(), 3),
               "turn": _SOAK_TURN, "subject": phase_id, "action": step_id,
               "ttft": round(ttft, 3) if ttft is not None else None,
               "total": round(total, 3), "in": pt, "out": ct, "cached": cached_pt}
        with _SOAK_JSONL_LOCK:
            with open(_SOAK_JSONL_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, separators=(",", ":")) + "\n")


def _elapsed(t0: float) -> str:
    return f"{time.time() - t0:.1f}s"


def _delta_stats(before: dict, after: dict) -> dict:
    """Per-task usage delta of a CloudModel's cumulative counters."""
    return {k: (after.get(k, 0) or 0) - (before.get(k, 0) or 0)
            for k in ("calls", "prompt_tokens", "completion_tokens", "total_tokens")}

TEST_DIR = Path(__file__).parent
REPO_ROOT = TEST_DIR.parent.parent.parent
EXTENSION_DIR = TEST_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from cloud_client import (CloudModel, CloudError, normalize_endpoint,  # noqa: E402
                          resolve_api_key_for_endpoint, resolve_local_api_key,
                          resolve_model_id)


# ----------------------------------------------------------------- config ---

DEFAULT_LOCAL_BASE = "http://127.0.0.1:5001/v1"


def _arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Long-horizon soak for dayna_ss")
    p.add_argument("--spreadsheet", default=None, help="path to spreadsheet JSON (not needed with --final-overview)")
    p.add_argument("--turns", type=int, default=100, help="total turns (default 100)")
    p.add_argument("--guide-style", choices=["explicit", "implicit"], default="implicit")
    p.add_argument("--specificity-profile", choices=["loose", "mixed", "exact"], default=None)
    p.add_argument("--director", choices=["on", "off"], default=None,
                    help="force do_instr on/off (default: use dss_config.json)")
    p.add_argument("--arc-breaks", choices=["none", "3_arcs"], default="none")
    p.add_argument("--judge-every", type=int, default=10, help="judge cadence (default 10)")
    p.add_argument("--plan-every", type=int, default=5,
                   help="replan cadence for the guide's live short-term plan "
                        "(0 disables the live plan; default 5)")
    p.add_argument("--smoke", type=int, default=0, help="run N turns then stop (validation)")
    p.add_argument("--level", type=int, choices=[1, 2], default=2)
    p.add_argument("--resume", default=None, help="run dir to resume")
    p.add_argument("--runs-dir", default=None,
                   help="root directory for run outputs (default: tests/runs). "
                        "Each run lands in <runs-dir>/<spreadsheet>__<hash>/")
    p.add_argument("--final-overview", metavar="RUN_DIR", default=None,
                   help="standalone: write a final overview for an existing run dir, then exit")
    p.add_argument("--backfill", action="store_true",
                    help="standalone: backfill missing judge/audit/overview for completed turns "
                         "(requires --resume); no new guide turns are run")
    p.add_argument("--clear-from", type=int, default=0,
                    help="standalone: clear judge/audit/overview data from turn N (0-indexed) onward "
                         "in an existing run (requires --resume); makes way for --backfill with a "
                         "new cloud model. Composes with --backfill to clear-then-re-backfill in one call")
    p.add_argument("--backfill-rolling-overview", nargs=2, metavar=("RUN_DIR", "N"),
                   action="append", default=None,
                   help="standalone: re-write the rolling overview checkpoint N (turn count, e.g. 32 "
                        "writes rolling_overview_032.json over the first 32 exchanges) for a run dir "
                        "whose checkpoint failed/errored; repeatable")
    p.add_argument("--overview-every", type=int, default=0,
                   help="opt-in rolling overview: every N turns, the judge cloud model writes a "
                        "MID-RUN overview of {name2}'s performance so far "
                        "(run_dir/rolling_overview_XXX.json). 0 disables (default 0).")
    p.add_argument("--embed-device", default="auto",
                   help="embedding device: auto|cpu|cuda:N (level 2 only)")
    p.add_argument("--max-subject-workers", type=int, default=3,
                   help="concurrent subjects processed by the engine's DataSummarizer per turn "
                        "(1 = serial; on the LMDeploy fork keep <=3 with prefix caching on, higher "
                        "with --disable-prefix-caching)")
    p.add_argument("--max-scene-messages", type=int, default=12,
                   help="hard per-scene-part message budget: force a scene boundary (archive + "
                        "on_new_scene triggers) once a scene part exceeds this many messages; "
                         "0 = only natural/auto-detected transitions (default 12)")
    p.add_argument("--force-chapter-turn", type=int, default=0,
                   help="turn N: stage a manual scene transition + forced chapter archival "
                        "(persistent_ui_state channel, same as production NEXT CHAPTER:); 0 = off")
    p.add_argument("--cadence-profile", choices=["compressed", "campaign"], default=None,
                   help="unit-cadence overlay over the schema's chapter/arc bounds "
                        "(compressed = shipped numbers, campaign = true-scale); default None = schema numbers")
    p.add_argument("--schema-type", type=int, choices=[1, 2, 3], default=1,
                   help="subjects schema type: 1=incremental-per-turn (default), 2=scene-aggregation, "
                        "3=hybrid (see schema_types_and_test_matrix.md)")
    p.add_argument("--schema", default=None,
                   help="explicit subjects_schema.json path (overrides --schema-type resolution)")
    p.add_argument("--no-schema-descriptions", action="store_true",
                   help="strip field descriptions from generated JSON schemas in DSS prompts (saves ~50% schema tokens)")
    p.add_argument("--embed-full-schema", action="store_true",
                   help="embed the full merged JSON schema in every DSS prompt (stable prefix-cache anchor; ~11K tokens baseline)")
    p.add_argument("--synthetic-chat", action="store_true",
                   help="ISOLATION MODE: the cloud model writes BOTH sides of the conversation; "
                        "DSS runs only the DataSummarizer (no local reply generation). Isolates "
                        "memory/summarization flaws from the local writer's prose.")
    p.add_argument("--guide-model", default="deepseek-v4-flash")
    p.add_argument("--judge-model", default="deepseek-v4-flash")
    p.add_argument("--auditor-model", default="deepseek-v4-flash",
                   help="cloud model that audits DSS's DataSummarizer saves against spreadsheet dss_beats")
    p.add_argument("--audit-every", type=int, default=0,
                   help="audit cadence in turns; 0 = audit only on beats' turn (default 0)")
    p.add_argument("--abort-after", type=int, default=2,
                   help="stop early if judge/auditor flags EXTREME DSS failure on N consecutive check turns "
                        "(0 disables; see the abort clause in the judge/auditor prompts)")
    p.add_argument("--guide-max-attempts", type=int, default=5,
                   help="cloud guide attempts per turn before giving up (default 5; robustness knobs are NOT "
                        "folded into the run-id hash, so a resume picks them up without changing the run dir)")
    p.add_argument("--cloud-empty-retries", type=int, default=3,
                   help="max re-rolls per cloud call when the model returns an empty content block (default 3)")
    p.add_argument("--cloud-empty-cooldown", type=float, default=300.0,
                    help="cooldown in seconds before each empty-content re-roll (default 300; a reasoning-only "
                         "draw is often time-correlated, so waiting lets the endpoint recover)")
    p.add_argument("--request-interval", type=float, default=5.0,
                    help="seconds to wait between consecutive cloud requests (default 5.0; "
                         "prevents rate-limiting on shared endpoints; 0 disables)")
    p.add_argument("--log-file", default=None,
                   help="path to a dedicated soak logfile; default writes to runs/logs/<spreadsheet>_<hash>.log")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--local-base", default=DEFAULT_LOCAL_BASE)
    p.add_argument("--local-model", default=None, help="model name for the local server (overrides DSS_BENCH_MODEL env var)")
    p.add_argument("--cloud-endpoint", default=None, help="common self-hosted OpenAI-compatible base URL for all cloud models (e.g. http://localhost:9931). Overrides default cloud endpoint.")
    p.add_argument("--cloud-provider", default=None, help="auth.json entry name to use as the API key for --cloud-endpoint (e.g. 'llama' or 'textgen'). When omitted, resolve_local_api_key() picks from its default order.")
    p.add_argument("--judge-window", type=int, default=12, help="exchanges shown to the judge")
    p.add_argument("--guide-memory", choices=("full", "names", "off"), default="full",
                   help="how much of DSS's memory the cloud guide/planner sees: 'full' (default, unchanged), "
                        "'names' (who and what exists, no detail), or 'off'. The guide restating details it read in "
                        "the memory keeps refreshing DSS, which flatters retention; 'names' or 'off' removes that loop")
    p.add_argument("--audit-retry-every", type=int, default=10,
                   help="every N turns, re-audit beats whose latest verdict is not 'saved' (default 10; 0 = off). "
                        "The auditor accepts late saves, so a beat audited once at its scheduled turn is judged too early")
    p.add_argument("--memory-grace", type=int, default=MEMORY_GRACE_DEFAULT,
                   help="exchanges a planted note/beat gets before a missing memory save counts against DSS "
                        f"(default {MEMORY_GRACE_DEFAULT}; scene-level subjects only update at scene boundaries)")
    p.add_argument("--max-update-history", type=int, default=0,
                   help="max engine internal-history exchanges sent per DataSummarizer call (0 = unlimited); "
                        "composes with the rolling window as sent = history[-min(rolling_window, this):]")
    p.add_argument("--temperature", type=float, default=0.7,
                   help="sampling temperature for the local model (Qwen recommended 0.7; default 0.7)")
    p.add_argument("--top-p", type=float, default=0.80,
                   help="nucleus sampling cutoff for the local model (Qwen recommended 0.80)")
    p.add_argument("--top-k", type=int, default=20,
                   help="top-k cutoff for the local model (Qwen recommended 20)")
    p.add_argument("--min-p", type=float, default=0.0,
                   help="min-p cutoff for the local model (Qwen recommended 0.0)")
    p.add_argument("--repetition-penalty", type=float, default=1.1,
                   help="repetition penalty for the local model (R4: default 1.1 to damp cross-turn "
                        "verbatim re-use; lmdeploy 0.15 has no presence_penalty, so this is the only "
                        "repetition knob and 1.0 is neutral)")
    p.add_argument("--message-mode", choices=["enumerated", "rolling"], default="enumerated",
                   help="Mode B: 'rolling' renders the recent dialogue as raw user/assistant pairs "
                        "(scene-bounded window) instead of the enumerated 'N. name >> ...' re-statement; "
                        "drops the 'Analyze all' marker. Default enumerated = baseline (see "
                        "schema_types_and_test_matrix.md §3)")
    p.add_argument("--retrieval-placement", choices=["prompt_start", "system", "inline"], default="prompt_start",
                   help="Mode B: where the non-to_context retrieved subjects (characters/groups/elements/"
                        "events/chapters/arcs/lines) render. prompt_start = baseline simulated Q&A turns; "
                        "system = ONE 'CURRENT CONTEXT (retrieved subjects)' block appended to the shared "
                        "system prefix (primary Mode B design — visible to reply AND every DataSummarizer "
                        "call); inline = same block as the last history pair at the generation boundary")
    p.add_argument("--rolling-window", type=int, default=6,
                   help="Mode B: scene-bounded recent-dialogue window size (rolling mode; default 6)")
    p.add_argument("--rolling-summaries", type=int, default=0,
                   help="Mode B (engine upgrade): when >0, inject the accumulated message summaries for "
                         "messages BEFORE the rolling window (sticky roll = max(N, last 5 scenes), locked "
                         "until the next scene turn for prefix caching) as a single pair ahead of the raw "
                         "recent pairs. 0 = disabled (default; the rolling window drops out-of-window "
                         "messages entirely)")
    p.add_argument("--restate-map-context", choices=["auto", "always", "never"], default="auto",
                   help="Per-entry whole-subject context re-statement in DataSummarizer calls "
                        "(docs/plans/long_horizon_soak_plan.md §23). auto = keep the full rendered subject map near "
                        "the generation boundary while it is under --restate-map-threshold chars (cheap "
                        "accentuation insurance for small maps), else drop it for a compact sibling "
                        "roster — fixes the O(map)-per-call prefill + context-window pressure. "
                        "always = quality-first (restate every call); never = perf-first.")
    p.add_argument("--restate-map-threshold", type=int, default=10000,
                   help="chars threshold for --restate-map-context auto (default 10000)")
    return p


# ------------------------------------------------------------- local model ---

def _thinking_enabled() -> bool:
    return os.environ.get("DSS_BENCH_THINKING", "0") in ("1", "true", "True")


def resolve_local_key() -> str:
    """Local server API key: DSS_BENCH_API_KEY, else auth.json 'localhost', else 'not-needed'."""
    key = os.environ.get("DSS_BENCH_API_KEY", "").strip()
    if key:
        return key
    auth_path = Path.home() / ".local" / "share" / "opencode" / "auth.json"
    if auth_path.exists():
        try:
            auth = json.loads(auth_path.read_text(encoding="utf-8"))
            for name in ("localhost", "textgen", "llama"):
                entry = auth.get(name)
                if isinstance(entry, dict) and entry.get("key"):
                    return entry["key"]
                if isinstance(entry, str):
                    return entry
        except Exception:
            pass
    return "not-needed"


# ---- Reply-side anti-transcription / anti-self-anchor guard (harness) ----
# The repetition loop's core is DIALOGUE transcription: the reply transcribes
# the instruction block nearly verbatim (0.86-0.99 ratio in the tail of the
# 40add80fa4 run) and self-anchors on its own prior sentences (rolling mode
# serves the model's own last replies next to the generation boundary).
# Production reply text is generated OUTSIDE the engine (TGWUI chat.py), so
# this deterministic post-check can only live in the soak harness; it fires a
# single bounded regeneration with an explicit anti-repeat directive, then
# keeps the less-colliding result. Detection is cheap (normalized difflib).
_REPLY_TX_SIM_THRESHOLD = 0.70   # reply vs the instruction block that steered it
_REPLY_SELF_SIM_THRESHOLD = 0.80 # reply vs the previous reply (self-anchor)
_REPLY_SELF_MIN_CHARS = 200       # self-anchor needs a substantial reply to matter
_REPLY_ANTI_REPEAT_DIRECTIVE = (
    "\n\nThe draft reply above repeats the instruction block or your own previous "
    "reply (same sentences, opening gestures, props, or imagery). Rewrite it FRESH "
    "for the CURRENT moment: different opening gesture, different concrete actions "
    "and objects, and do not echo the instruction block's wording — execute it, do "
    "not transcribe it. Keep the same voice and writing style."
)


def _reply_transcribes_prompt(reply: str, prompt: str) -> bool:
    """Deterministic check: does the reply near-verbatim transcribe the prompt
    (the instruction block / reply direction)?"""
    if not reply or not prompt:
        return False
    nr = re.sub(r"[^a-z0-9 ]", "", reply.lower()).strip()
    np_ = re.sub(r"[^a-z0-9 ]", "", prompt.lower()).strip()
    if len(nr) < 80 or len(np_) < 80:
        return False
    return difflib.SequenceMatcher(None, nr, np_, autojunk=False).ratio() > _REPLY_TX_SIM_THRESHOLD


def _reply_self_anchors(reply: str, prev_reply: str) -> bool:
    """Deterministic check: does the reply re-anchor on the previous reply
    (same sentences/gestures — the rolling-window self-anchor)?"""
    if not reply or not prev_reply:
        return False
    if len(reply) < _REPLY_SELF_MIN_CHARS or len(prev_reply) < _REPLY_SELF_MIN_CHARS:
        return False
    nr = re.sub(r"[^a-z0-9 ]", "", reply.lower()).strip()
    np_ = re.sub(r"[^a-z0-9 ]", "", prev_reply.lower()).strip()
    if len(nr) < 80 or len(np_) < 80:
        return False
    return difflib.SequenceMatcher(None, nr, np_, autojunk=False).ratio() > _REPLY_SELF_SIM_THRESHOLD


class LocalModel:
    """Live local model: OpenAI-compatible chat/completions, streamed to the engine.

    Implements ``generate_with_streaming(encoded_prompt, state)`` as a generator
    so the real engine path runs unmodified. Wraps every prompt with the recent
    dialogue (mirrors production ``generate_chat_prompt``), because the engine's
    gate-check/update/message-summary templates reference the latest exchange.
    """

    def __init__(self, base_url: str, api_key: str, model: str, max_tokens: int = 2048,
                 temperature: float = 0.7, top_p: float = 0.80, top_k: int = 20,
                 min_p: float = 0.0, repetition_penalty: float = 1.0,
                 history_provider: Any = None,
                  max_update_history_exchanges: int = 0,
                 reasoning_effort: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.repetition_penalty = repetition_penalty
        self.reasoning_effort = reasoning_effort
        self.history_provider = history_provider  # callable() -> list of [u, r]
        self.max_update_history_exchanges = max_update_history_exchanges
        self.call_count = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.max_prompt_tokens = 0
        self.ctx_size = 65536
        self.last = None
        self.last_content = ""
        self.last_reasoning = ""

    def _wrap(self, prompt: str, state: dict) -> str:
        name1 = state.get("name1", "User")
        name2 = state.get("name2", "Assistant")
        if self.history_provider is not None:
            hist = self.history_provider() or []
            recent = "\n".join(f"{name1}: {u}\n{name2}: {r}" for u, r in hist[-6:])
            return f"{recent}\n\n{prompt}" if recent else prompt
        return prompt

    @staticmethod
    def _engine_internal_messages(state: dict, max_exchanges: int = 0) -> list[dict]:
        """Render ``state['history']['internal']`` as user/assistant pairs.

        Mirrors ``modules.chat.generate_chat_prompt`` (production renders the
        engine's artificial internal history for every call). The harness used to
        bypass this with ``_wrap`` (last-6 raw exchanges bolted onto the prompt),
        which made the model's own previous reply the last text before the
        instruction block — a greedy-copy anchor. Rendering the engine's internal
        history (retrieval Q&A, scene-bounded last-X window, analysis marker) keeps
        the         ordering production uses instead.

        ``max_exchanges`` bounds how much of the internal history is sent. The
        DataSummarizer's per-entry update prompts only reference "the latest
        exchange(s)", so on the summarization/update call paths the old exchanges
        are redundant with the engine's own scene-bounded window — trimming them
        keeps the per-call prompt (and thus prefill cost) from growing with the
        whole story, which is the O(N²) blowup in long runs. 0/None = unlimited.
        """
        internal = ((state or {}).get("history") or {}).get("internal") or []
        if max_exchanges:
            # The engine's "Summaries of earlier messages" pair is the ONE thing
            # the bound must not cut: it exists precisely to restore the older
            # context the bound drops (the sticky-roll window). Preserve it and
            # bound only the raw rolling pairs behind it.
            summaries = []
            rest = []
            for entry in internal:
                if isinstance(entry, (list, tuple)) and len(entry) and str(entry[0] or "").startswith("Summaries of earlier messages"):
                    summaries.append(entry)
                else:
                    rest.append(entry)
            internal = summaries + rest[-max_exchanges:]
        msgs = []
        for entry in internal:
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                continue
            user_msg = str(entry[0] or "").strip()
            assistant_msg = str(entry[1] or "").strip()
            if user_msg not in ("", "<|BEGIN-VISIBLE-CHAT|>"):
                msgs.append({"role": "user", "content": user_msg})
            if assistant_msg:
                msgs.append({"role": "assistant", "content": assistant_msg})
        return msgs

    def _complete(self, prompt: str, state: dict, bound_history: bool = False,
                  phase_id: str = "", step_id: str = "") -> str:
        messages = []
        context = (state or {}).get("context", "").strip()
        if context:
            messages.append({"role": "system", "content": context})
        bound = self.max_update_history_exchanges if bound_history else 0
        internal_msgs = self._engine_internal_messages(state, max_exchanges=bound)
        if internal_msgs:
            messages.extend(internal_msgs)
            messages.append({"role": "user", "content": prompt})
        else:
            messages.append({"role": "user", "content": self._wrap(prompt, state)})
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            # Qwen's recommended sampling: top_p 0.80, top_k 20, min_p 0.0,
            # repetition_penalty 1.0. presence_penalty is NOT sent: lmdeploy
            # 0.15's GenerationConfig has no such field, so it is a no-op on
            # this server (and would only apply to the cloud models, which are
            # deepseek, not Qwen). These are sent explicitly so the soak is
            # server-independent of the running instance's top_p/top_k defaults.
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "repetition_penalty": self.repetition_penalty,
            # Stream so the harness can measure real time-to-first-token.
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if _thinking_enabled():
            payload["enable_thinking"] = True
            if self.reasoning_effort:
                payload["reasoning_effort"] = self.reasoning_effort
        else:
            payload["enable_thinking"] = False
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        t0 = time.perf_counter()
        # The lmdeploy v0.15 engine occasionally returns an empty completion
        # (content="" and reasoning_content="", finish="stop") under concurrent
        # load, and can stall past the socket timeout when the GPU is busy.
        # Retry a few times so the DataSummarizer never sees a blank and the
        # run never dies on a transient transport error.
        for attempt in range(3):
            try:
                ttft, total, content, reasoning, pt, ct, cached_pt = self._stream_post(req, t0)
            except (TimeoutError, ConnectionError, urllib.error.URLError) as e:
                if attempt >= 2:
                    raise
                print(f"local: _complete attempt {attempt + 1} failed ({e}); retrying")
                time.sleep(1.0 * (attempt + 1))
                continue
            if not pt:
                # lmdeploy v0.15 returns prompt_tokens=0 in usage; estimate from
                # the actual payload so ctx-size tracking is meaningful.
                pt = sum(len((m.get("content") or "")) for m in messages) // 4
            self.prompt_tokens += pt
            self.completion_tokens += ct
            self.max_prompt_tokens = max(self.max_prompt_tokens, pt)
            if content or reasoning:
                _soak_log_call(phase_id, step_id, ttft, total, pt, ct, cached_pt)
                self.last_content = content
                self.last_reasoning = reasoning
                return content or reasoning
            if attempt < 2:
                time.sleep(0.5 * (attempt + 1))
        _soak_log_call(phase_id, step_id, ttft, total, pt, ct, cached_pt)
        return ""

    @staticmethod
    def _stream_post(req: urllib.request.Request, t0: float):
        """POST with stream=true, parse SSE, return (ttft, total, content, reasoning, pt, ct).

        ``ttft`` is the wall time until the first streamed delta carries any
        content (or reasoning when thinking is enabled); ``total`` is time to the
        ``[DONE]`` frame. ``pt``/``ct`` come from the include_usage chunk (may be
        0 on lmdeploy v0.15, the caller estimates pt from the prompt length).
        """
        ttft = None
        content = ""
        reasoning = ""
        pt = ct = cached_pt = 0
        with urllib.request.urlopen(req, timeout=300) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if ttft is None:
                    try:
                        delta = chunk["choices"][0].get("delta") or {}
                    except (IndexError, KeyError):
                        delta = {}
                    if delta.get("content") or delta.get("reasoning_content"):
                        ttft = time.perf_counter() - t0
                if "usage" in chunk:
                    u = chunk.get("usage") or {}
                    pt, ct = int(u.get("prompt_tokens", 0) or 0), int(u.get("completion_tokens", 0) or 0)
                    details = u.get("prompt_tokens_details") or {}
                    cached_pt = int(details.get("cached_tokens", 0) or 0)
                fr = None
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        content += delta["content"]
                    if delta.get("reasoning_content"):
                        reasoning += delta["reasoning_content"]
                    if choice.get("finish_reason"):
                        fr = choice["finish_reason"]
                if fr:
                    break
        return ttft, time.perf_counter() - t0, content, reasoning, pt, ct, cached_pt

    def generate_with_streaming(self, encoded_prompt: Any, state: dict):
        self.call_count += 1
        phase_id = state.get('_dss_phase_id', '')
        step_id = state.get('_dss_step_id', '')
        text = self._complete(str(encoded_prompt), state, phase_id=phase_id, step_id=step_id)
        yield text

    @staticmethod
    def _stop(text: str, stopping_strings: list[str] | None, match_prefix_only: bool) -> tuple[str, str]:
        stopping_strings = stopping_strings or []
        if match_prefix_only:
            for ss in stopping_strings:
                if text.lstrip().startswith(ss):
                    return text, ss
        else:
            for ss in stopping_strings:
                if ss in text:
                    return text, ss
        return text, ""

    def generate_with_sse(self, prompt: str, state: dict | None = None, phase_id: str = "",
                          step_id: str = "", history_path: str | Path | None = None,
                          stopping_strings: list[str] | None = None,
                          match_prefix_only: bool = True, **kwargs) -> tuple[str, str]:
        text = self._complete(prompt, state or {}, bound_history=True,
                              phase_id=phase_id, step_id=step_id)
        self.call_count += 1
        return self._stop(text, stopping_strings, match_prefix_only)

    def generate_using_tgwui(self, prompt: str, state: dict | None = None,
                             history_path: str | Path | None = None,
                             stopping_strings: list[str] | None = None,
                             match_prefix_only: bool = True, **kwargs) -> tuple[str, str]:
        text = self._complete(prompt, state or {}, bound_history=True,
                              phase_id=kwargs.get("phase_id", ""), step_id=kwargs.get("step_id", ""))
        self.call_count += 1
        return self._stop(text, stopping_strings, match_prefix_only)

    def format_dialogue(self, state: dict | None, history: list) -> str:
        return "\n".join(f"{u}: {o}" for u, o in (history or []))

    def complete(self, prompt: str, state: dict) -> str:
        self.call_count += 1
        return self._complete(prompt, state, bound_history=False, phase_id="reply", step_id="generate")

    def stats(self) -> dict:
        max_prompt = self.max_prompt_tokens
        self.max_prompt_tokens = 0
        return {
            "calls": self.call_count,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "max_prompt_tokens": max_prompt,
            "ctx_size": self.ctx_size,
            "model": self.model,
        }


def health_check(base_url: str, api_key: str, model: str, tries: int = 3) -> bool:
    for i in range(tries):
        try:
            req = urllib.request.Request(
                base_url + "/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status == 200
        except Exception as e:
            if i == tries - 1:
                print(f"[soak] health check failed ({tries} tries): {e}")
                return False
            time.sleep(5)
    return False


# ----------------------------------------------------------- spreadsheet ---

def load_spreadsheet(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_spreadsheet(raw: str) -> dict:
    """Resolve a spreadsheet path, falling back to tests/spreadsheets/<name>."""
    p = Path(raw)
    if not p.exists():
        alt = TEST_DIR / "spreadsheets" / p.name
        if alt.exists():
            p = alt
    return load_spreadsheet(p)


def note_window_start(note: dict) -> int:
    plant = note.get("plant", {}).get("turn")
    if plant is not None:
        return int(plant)
    recall = note.get("recall", {}).get("due")
    return int(recall) if recall is not None else 0


def note_window_end(note: dict, spread: int = 8) -> int:
    recall = note.get("recall", {}).get("due")
    if recall is not None:
        return int(recall) + spread
    plant = note.get("plant", {}).get("turn")
    return (int(plant) + spread) if plant is not None else 0


def notes_visible_at(notes: list[dict], turn: int, guide_style: str, spread: int = 8) -> list[dict]:
    """Windowed notes for the guide at ``turn`` (plan §5/§6: never full outline)."""
    out = []
    for note in notes:
        start = note_window_start(note)
        end = note_window_end(note, spread)
        if start <= turn <= end:
            out.append(note)
    return out


# ------------------------------------------------------------- engine p1 ---

def _warm_imports() -> None:
    """Import heavy deps synchronously to avoid the background-importer deadlock.

    ``context_retriever`` kicks off background threads for llama_index/spacy/nltk;
    if the main thread then asks for a *different* key (e.g. ``llama_index.core``
    vs ``llama_index.core.StorageContext``) a second thread starts while the first
    holds the module lock and Python reports an import deadlock. Pre-importing
    everything up front makes every background lookup a cache hit.
    """
    import llama_index.core  # noqa: F401
    import llama_index.core.node_parser  # noqa: F401
    import llama_index.core.indices.loading  # noqa: F401
    import llama_index.core.settings  # noqa: F401
    import llama_index.core.schema  # noqa: F401
    import llama_index.embeddings.huggingface  # noqa: F401
    import nltk  # noqa: F401
    import spacy  # noqa: F401


def _apply_embed_device(device: str) -> None:
    """Set CUDA_VISIBLE_DEVICES before torch/llama_index load, so the embedding
    model lands on the requested GPU (or CPU). Called before engine import."""
    if device == "auto":
        return
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif device.startswith("cuda:"):
        idx = device.split(":", 1)[1]
        os.environ["CUDA_VISIBLE_DEVICES"] = idx


# --------------------------------------------------- history stripping ----

def _strip_history_keys(node: Any) -> Any:
    """Recursively remove any dict key ending in ``_history`` from a JSON-loaded
    structure. These are engine-internal bookkeeping fields (per-field update
    history) that bloat cloud prompts without being relevant to the guide,
    judge, or auditor."""
    if isinstance(node, dict):
        return {k: _strip_history_keys(v)
                for k, v in node.items() if not k == "_history"}
    if isinstance(node, list):
        return [_strip_history_keys(item) for item in node]
    return node


# ------------------------------------------------------------- needles -----

def _find_needle(needle: str, node, path: str = "") -> list[str]:
    """Return all dot-paths where the needle appears (case-insensitive)."""
    hits = []
    if isinstance(node, dict):
        for k, v in node.items():
            hits += _find_needle(needle, v, f"{path}.{k}" if path else k)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            hits += _find_needle(needle, v, f"{path}[{i}]")
    elif isinstance(node, str) and needle.lower() in node.lower():
        hits.append(path)
    return hits


def probe_needles(notes: list[dict], state_dir: Path) -> dict:
    """Deterministic per-note presence timeline over the subject JSONs."""
    result = {}
    if not state_dir.exists():
        return result
    subjects = {}
    for f in sorted(state_dir.glob("*.json")):
        if f.name in ("subjects_schema.json", "format_templates.json", "entity_graph.json"):
            continue
        try:
            subjects[f.stem] = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
    for note in notes:
        nid = note["id"]
        needle = note.get("needle", "")
        if not needle:
            result[nid] = {"needle": needle, "present": False, "paths": []}
            continue
        hits = []
        for subj, data in subjects.items():
            hits += [(subj, p) for p in _find_needle(needle, data)]
        result[nid] = {"needle": needle, "present": bool(hits), "paths": hits[:5]}
    return result


def recall_probe(note: dict, reply: str, state_dir: Path) -> dict:
    """Two-tier recall: exact (needle_syns in reply) + state-backed (Q3)."""
    text = (reply or "").lower()
    syns = note.get("needle_syns", [])
    matched = [s for s in syns if s and s.lower() in text]
    # False-positive guard: state must still hold the detail (Q3a).
    state_holds = False
    if state_dir.exists():
        probe = probe_needles([note], state_dir)
        state_holds = bool(probe.get(note["id"], {}).get("present"))
    return {
        "id": note["id"],
        "matched_syns": matched,
        "echoed": bool(matched) and state_holds,
        "echo_by_text_only": bool(matched) and not state_holds,
    }


# ------------------------------------------------------------ prompts -------

GUIDE_SYSTEM_TMPL = """You are the user in a collaborative fiction.
You write {name1}'s turn in the story: {name1} is a {role1}. The other character, {name2} ({role2}), is written by another writer and maintains a private memory of the story.

Premise: {premise}
Setting: {setting}
Your voice ({name1}, guide turns): {guide_directive}
{taboos}

Continue the story with {name1}'s turn. Advance the plot naturally. You have a private outline; below are the notes relevant to THIS turn only — interpret them loosely and naturally, never quote them verbatim.

{steering_note}

HARD OUTPUT RULES:
- Output ONLY this turn as fiction — {name1}'s words and actions, narrated in the person, tense, and register your voice directive above specifies (it may be first, second, or third person). Never meta-commentary, never narration about the story, never planning language.
- Write strictly in {name1}'s voice as the style directive above specifies — person, tense, register — one scene beat, roughly the length the scene calls for.
- Never reveal that you are following notes or that an outline exists."""

GUIDE_STEERING = {
    "explicit": "Work the notes' specifics into what you say or do this turn — make the planted details explicit in the scene.",
    "implicit": "Advance the scene naturally and let the other writer pick up the planted details on their own. Do not force them.",
}

_META_PREFIXES = (
    "the user wants", "i need to", "we need to", "let me", "okay,", "alright,",
    "as marlowe", "i'm playing", "i am playing", "my turn", "the scene:",
    "first,", "so,", "now i", "continuing", "response:", "as {",
)


def clean_guide_output(text: str) -> str:
    """Strip the meta/planning preamble some cloud checkpoints emit before the
    in-fiction turn (deepseek-v4-flash occasionally leaks reasoning into content)."""
    text = (text or "").strip()
    if not text:
        return text
    lines = [l for l in text.splitlines() if l.strip()]
    keep_from = 0
    for i, line in enumerate(lines):
        low = line.strip().lower()
        if low.startswith(_META_PREFIXES) or low.startswith("the user wants me"):
            keep_from = i + 1
        else:
            break
    kept = lines[keep_from:] if keep_from else lines
    return "\n".join(kept).strip() or text


def guide_turn_with_retry(guide: "CloudModel", spreadsheet: dict, msgs: list[dict],
                          max_tokens: int = 8000, max_attempts: int = 5,
                          cooldown: float = 300.0,
                          label: str = "user turn") -> str:
    """Call the cloud guide, clean the output, and validate it (non-empty, no
    verbatim note text). Retries with a harness note on unusable output, resting
    on a cooldown between attempts for empty/truncated draws (the reasoning-only
    variance is time-correlated, so waiting lets the endpoint recover); only
    raises after every attempt fails, so a transient outage can no longer kill
    the run."""
    OUTAGE_REASONS = ("returned no usable text", "returned a truncated/barely-started turn")
    for attempt in range(max_attempts):
        t0 = time.time()
        try:
            raw = guide.complete(msgs, max_tokens=max_tokens)
        except CloudError as e:
            # Transport-level failure (HTTP 4xx/5xx after backoff, network death).
            # Free/tiered endpoints reject long prompts with HTTP 400 and hiccup
            # under load — treat it as an empty draw so the cooldown loop below
            # absorbs it instead of the run dying on the spot. Auth failures are
            # permanent: fail fast rather than burn five cooldown cycles.
            if "HTTP 401" in str(e) or "HTTP 403" in str(e):
                raise RuntimeError(
                    f"guide {label}: cloud endpoint rejected auth ({e}); "
                    f"check the API key for {guide.endpoint}") from e
            soak_log(f"  guide: {label} cloud call FAILED ({e}); treating as empty draw")
            raw = ""
        text = clean_guide_output(raw)
        soak_log(f"  guide: cloud {label} in {_elapsed(t0)} -> {text[:80]!r}...")
        reason = None
        if not text.strip():
            reason = "returned no usable text"
        elif len(text.strip()) < 80:
            reason = "returned a truncated/barely-started turn"
        else:
            for note in spreadsheet["notes"]:
                content = (note.get("content") or "").strip()
                if content and content in text:
                    reason = (f"quoted private outline note '{note['id']}' verbatim "
                              f"(never repeat outline text)")
                    break
        if reason is None:
            return text
        if attempt >= max_attempts - 1:
            raise RuntimeError(
                f"guide produced unusable {label} {max_attempts}x ({reason}); "
                f"aborting to protect DSS prompts"
            )
        soak_log(f"  guide: {label} {reason}; retrying ({attempt + 1}/{max_attempts}) ...")
        if reason in OUTAGE_REASONS and cooldown > 0:
            soak_log(f"  guide: empty/truncated draw -> cooling down {cooldown:.0f}s before next attempt ...")
            time.sleep(cooldown)
        msgs = [dict(m) for m in msgs]
        msgs[-1] = {**msgs[-1], "content": msgs[-1]["content"] + (
            f"\n\n[HARNESS NOTE] Your previous attempt {reason}. "
            f"Write ONLY this turn as in-fiction prose — no planning, "
            f"no meta-commentary, no outline text.")}
    raise RuntimeError(f"guide produced unusable {label}")  # unreachable

# ---------------------------------------------------------------------------
# Memory fidelity v2: answer key, evidence-audited verification, code-computed
# score. The judge never writes a memory number; it verifies items and cites
# evidence, and the harness turns verdicts into a score (see score_memory_checks).
# ---------------------------------------------------------------------------
import math as _math

_NON_SUBJECT_JSON = ("subjects_schema.json", "format_templates.json", "entity_graph.json")
MEMORY_GRACE_DEFAULT = 3  # exchanges a detail gets before a missing save counts against DSS

_NOTE_CREDIT = {"retained": 1.0, "partial": 0.5, "missing": 0.0, "contradicted": 0.0}
_BEAT_CREDIT = {"saved": 1.0, "partial": 0.5, "wrong": 0.0, "missing": 0.0}
_FACT_CREDIT = {"retained": 1.0, "partial": 0.5, "stale": 0.25, "missing": 0.0, "contradicted": 0.0}
_FACT_WEIGHT = {"high": 3.0, "medium": 2.0, "low": 1.0}
_MEMORY_WEIGHTS = {"notes": 0.35, "beats": 0.25, "facts": 0.40}
_MEMORY_MIN_ITEMS_FOR_TOP = 6  # fewer checked items than this cannot earn above 4.5
_CLAIM_PENALTY = 0.5
_CLAIM_PENALTY_CAP = 1.5
_DEMOTE = {"retained": "partial", "saved": "partial"}


def _text_tokens(text: str) -> set[str]:
    cleaned = (text or "").lower().replace("'", "").replace("\u2019", "")
    return set(re.findall(r"[a-z0-9]{3,}", cleaned))


def _quote_supported(quote: str, haystack_tokens: set[str], min_ratio: float = 0.75) -> bool:
    """True when most content words of ``quote`` occur in the haystack (tolerates light paraphrase)."""
    q = _text_tokens(quote)
    if not q:
        return False
    return len(q & haystack_tokens) / len(q) >= min_ratio


def _first_seen(terms: list[str], history: list[list[str]]) -> int | None:
    terms = [t.strip().lower() for t in terms if t and t.strip()]
    if not terms:
        return None
    for i, (u, r) in enumerate(history):
        blob = f"{u or ''}\n{r or ''}".lower()
        if any(t in blob for t in terms):
            return i
    return None


def memory_answer_key(spreadsheet: dict, history: list[list[str]], grace: int = MEMORY_GRACE_DEFAULT,
                      in_beat_ids: set[str] | None = None) -> dict:
    """What the memory is expected to hold right now, derived in code from the outline + transcript.

    A planted note (needle seen in the story) or an expected beat (entity seen in the story) becomes
    checkable ``grace`` exchanges after it first appeared; earlier than that it is listed as pending so
    DSS is not charged for a save it has not had time to make.
    """
    last = len(history) - 1
    in_beat_ids = set(in_beat_ids or ())
    notes, pending_notes = [], []
    for n in spreadsheet.get("notes", []):
        needle = (n.get("needle") or "").strip()
        if not needle or n.get("type") == "style_hold":
            continue
        first = _first_seen([needle], history)
        if first is None:
            continue
        age = last - first
        if age < grace:
            pending_notes.append(n["id"])
        else:
            lm = _last_seen([needle], history)
            notes.append({"id": n["id"], "type": n.get("type"), "content": n.get("content"),
                          "age_exchanges": age, "silent_exchanges": last - lm if lm is not None else age,
                          "in_beat": n["id"] in in_beat_ids})
    beats, pending_beats = [], []
    for b in spreadsheet.get("dss_beats", []) or []:
        if int(b.get("turn", 0) or 0) > last:
            continue
        name = ((b.get("expect") or {}).get("name") or "").strip()
        first = _first_seen([name], history) if name else None
        if name and first is None:
            continue  # the entity has not appeared in the story yet
        if first is not None and last - first < grace:
            pending_beats.append(b["id"])
            continue
        beats.append({"id": b["id"], "subject": b.get("subject"), "content": b.get("content"),
                      "expect": b.get("expect")})
    return {"notes": notes, "beats": beats, "pending_notes": pending_notes, "pending_beats": pending_beats}


def score_memory_checks(checks, key: dict, state_text: str, window_text: str = "",
                        wrong_character: bool = False, notes_by_id: dict | None = None):
    """Turn the judge's verification checklist into (score 0-5 | None, breakdown).

    Safeguards against leniency, all deterministic:
    - retained/saved/partial/stale must carry a state_quote that really occurs in the state, else the
      verdict is demoted one level;
    - a note judged retained whose needle (and synonyms) is absent from the whole state is demoted to partial;
    - story facts whose source quote is not in the transcript are dropped (the judge invented them);
    - a contradicted claim without a source quote that exists in the transcript/notes is treated as unverifiable;
    - fewer than _MEMORY_MIN_ITEMS_FOR_TOP checked items cannot reach 5; the score is floored to 0.5.
    """
    if not isinstance(checks, dict):
        return None, {"error": "memory_checks missing or not an object"}
    notes_by_id = notes_by_id or {}
    state_tokens = _text_tokens(state_text)
    state_low = (state_text or "").lower()
    source_tokens = _text_tokens(window_text)
    for n in notes_by_id.values():
        source_tokens |= _text_tokens(n.get("content", ""))
    flags: list[str] = []
    failures: list[dict] = []

    def items(name):
        v = checks.get(name)
        return [x for x in v if isinstance(x, dict)] if isinstance(v, list) else []

    def verify(it, status, valid):
        """Apply the quote requirement; returns the audited status."""
        status = str(status or "").strip().lower()
        if status not in valid:
            return None
        if status in _DEMOTE and not _quote_supported(str(it.get("state_quote", "")), state_tokens):
            flags.append(f"quote_not_in_state:{it.get('id') or str(it.get('fact', ''))[:30]}")
            status = _DEMOTE[status]
        it["status_audited"] = status
        return status

    # ---- A. planted notes
    expected_notes = {n["id"] for n in key.get("notes", [])}
    seen: dict[str, str] = {}
    for it in items("note_retention"):
        nid = str(it.get("id", ""))
        if nid not in expected_notes or nid in seen:
            continue
        st = verify(it, it.get("status"), _NOTE_CREDIT)
        if st is None:
            continue
        spec = notes_by_id.get(nid, {})
        terms = [t for t in [spec.get("needle", "")] + list(spec.get("needle_syns") or []) if t]
        if st == "retained" and terms and not any(t.lower() in state_low for t in terms):
            flags.append(f"needle_absent:{nid}")
            st = it["status_audited"] = "partial"
        seen[nid] = st
        if st != "retained":
            failures.append({"kind": "note", "id": nid, "status": st})
    note_credits = [(_NOTE_CREDIT[s], 1.0) for s in seen.values()]

    # ---- B. memory beats
    expected_beats = {b["id"] for b in key.get("beats", [])}
    bseen: dict[str, str] = {}
    for it in items("beat_checks"):
        bid = str(it.get("id", ""))
        if bid not in expected_beats or bid in bseen:
            continue
        st = verify(it, it.get("status"), _BEAT_CREDIT)
        if st is None:
            continue
        bseen[bid] = st
        if st != "saved":
            failures.append({"kind": "beat", "id": bid, "status": st})
    beat_credits = [(_BEAT_CREDIT[s], 1.0) for s in bseen.values()]

    # ---- C. story facts
    fact_credits, fact_rows = [], []
    basis = {"explicit": 0, "implicit": 0}
    for it in items("story_facts")[:12]:
        if not _quote_supported(str(it.get("source", "")), source_tokens):
            flags.append(f"fact_source_not_in_story:{str(it.get('fact', ''))[:30]}")
            continue
        st = verify(it, it.get("status"), _FACT_CREDIT)
        if st is None:
            continue
        w = _FACT_WEIGHT.get(str(it.get("importance", "medium")).lower(), 2.0)
        fact_credits.append((_FACT_CREDIT[st], w))
        b = str(it.get("basis", "")).lower()
        if b in basis:
            basis[b] += 1
        fact_rows.append((st, str(it.get("era", "")).lower(), _FACT_CREDIT[st], w))
        if st != "retained":
            failures.append({"kind": "fact", "fact": str(it.get("fact", ""))[:90], "status": st,
                             "importance": str(it.get("importance", "medium")).lower()})

    # ---- precision audit
    claims = items("state_claims")[:8]
    contradicted = 0
    for it in claims:
        st = str(it.get("status", "")).strip().lower()
        if st == "contradicted":
            if _quote_supported(str(it.get("source", "")), source_tokens):
                contradicted += 1
                failures.append({"kind": "claim", "claim": str(it.get("claim", ""))[:90], "status": "contradicted"})
            else:
                flags.append(f"contradiction_without_source:{str(it.get('claim', ''))[:30]}")
                it["status_audited"] = "unverifiable"
    penalty = min(_CLAIM_PENALTY_CAP, _CLAIM_PENALTY * contradicted)

    # ---- combine
    def cat(credits):
        if not credits:
            return None
        return sum(c * w for c, w in credits) / sum(w for _, w in credits)

    cats = {"notes": cat(note_credits), "beats": cat(beat_credits), "facts": cat(fact_credits)}
    present = {k: v for k, v in cats.items() if v is not None}
    n_items = len(note_credits) + len(beat_credits) + len(fact_credits)
    if not present:
        return None, {"error": "no checkable memory items in the judge output", "flags": flags[:20]}
    wsum = sum(_MEMORY_WEIGHTS[k] for k in present)
    raw = sum(_MEMORY_WEIGHTS[k] * v for k, v in present.items()) / wsum
    score = raw * 5.0 - penalty
    caps = []
    if n_items < _MEMORY_MIN_ITEMS_FOR_TOP:
        score = min(score, 4.5)
        caps.append(f"thin_evidence(<{_MEMORY_MIN_ITEMS_FOR_TOP}_items)")
    if wrong_character:
        score = min(score, 2.0)
        caps.append("wrong_character")
    score = max(0.0, min(5.0, _math.floor(score * 2.0 + 1e-9) / 2.0))

    def counts(vals):
        out: dict[str, int] = {}
        for v in vals:
            out[v] = out.get(v, 0) + 1
        return out

    by_era = {}
    for era in ("early", "middle", "late"):
        rows = [(c, w) for _, e, c, w in fact_rows if e == era]
        if rows:
            by_era[era] = round(sum(c * w for c, w in rows) / sum(w for _, w in rows), 2)
    breakdown = {
        "score": score, "raw": round(raw, 3), "n_items": n_items,
        "categories": {
            "notes": {"n": len(note_credits), "score": None if cats["notes"] is None else round(cats["notes"], 3),
                      "counts": counts(seen.values()), "coverage": f"{len(seen)}/{len(expected_notes)}"},
            "beats": {"n": len(beat_credits), "score": None if cats["beats"] is None else round(cats["beats"], 3),
                      "counts": counts(bseen.values()), "coverage": f"{len(bseen)}/{len(expected_beats)}"},
            "facts": {"n": len(fact_credits), "score": None if cats["facts"] is None else round(cats["facts"], 3),
                      "counts": counts(r[0] for r in fact_rows), "basis": basis},
        },
        "claims": {"checked": len(claims), "contradicted": contradicted, "penalty": penalty},
        "caps": caps, "flags": flags[:20], "failures": failures[:12],
    }
    # retention split by silence: a note the story keeps repeating is refreshed, not remembered
    sil = {n["id"]: n.get("silent_exchanges") for n in key.get("notes", [])}
    split = {"refreshed": [], "silent": []}
    for nid, st in seen.items():
        bucket = "silent" if (sil.get(nid) or 0) >= 5 else "refreshed"
        split[bucket].append(_NOTE_CREDIT[st])
    breakdown["notes_by_silence"] = {k: {"n": len(v), "score": round(sum(v) / len(v), 3) if v else None}
                                     for k, v in split.items()}
    if by_era:
        breakdown["facts_by_era"] = by_era
    return score, breakdown


def compact_judge_record(rec: dict) -> dict:
    """Judge record minus the bulky checklist, for prompts that carry every record (overviews)."""
    out = {k: rec[k] for k in ("turn", "style_score", "quality_score", "memory_fidelity", "notes", "abort", "summary")
           if k in rec}
    bd = rec.get("memory_breakdown")
    if isinstance(bd, dict):
        out["memory_breakdown"] = {k: bd[k] for k in ("n_items", "categories", "claims", "caps", "failures") if k in bd}
    return out


def memory_ledger(judge_records: list[dict]) -> str:
    """Computed per-turn memory scores as text, so overviews anchor on numbers, not impressions."""
    rows = [(r.get("turn"), r.get("memory_fidelity")) for r in judge_records
            if isinstance(r, dict) and isinstance(r.get("memory_fidelity"), (int, float))]
    if not rows:
        return "(no per-turn memory checks were recorded)"
    rows.sort(key=lambda x: (x[0] is None, x[0]))
    vals = [v for _, v in rows]
    legacy = not any(isinstance(r.get("memory_breakdown"), dict) for r in judge_records if isinstance(r, dict))
    lines = [f"Judged turns: {len(rows)}. Computed memory score per judged turn: "
             + ", ".join(f"t{t}={v:.1f}" for t, v in rows),
             f"Mean {sum(vals) / len(vals):.2f}; first-third mean "
             f"{sum(vals[:max(1, len(vals) // 3)]) / max(1, len(vals) // 3):.2f}; last-third mean "
             f"{sum(vals[-max(1, len(vals) // 3):]) / max(1, len(vals) // 3):.2f}."]
    if legacy:
        lines.append("NOTE: these per-turn scores are holistic impressions from the older rubric, not computed checks; "
                     "weigh them less than your own verification.")
    tally: dict[str, int] = {}
    for r in judge_records:
        for f in ((r.get("memory_breakdown") or {}).get("failures") or []):
            label = f.get("id") or f.get("fact") or f.get("claim") or "?"
            label = f"{f.get('kind')}:{label}:{f.get('status')}"
            tally[label] = tally.get(label, 0) + 1
    if tally:
        top = sorted(tally.items(), key=lambda kv: -kv[1])[:10]
        lines.append("Most repeated non-retained items (judged turns in which each failed): "
                     + "; ".join(f"{k} x{v}" for k, v in top))
    return "\n".join(lines)


def overview_memory_inputs(spreadsheet: dict, history: list[list[str]], judge_records: list[dict],
                           grace: int = MEMORY_GRACE_DEFAULT):
    return memory_answer_key(spreadsheet, history, grace), memory_ledger(judge_records)


def finalize_overview_memory(result, spreadsheet: dict, history: list[list[str]], state_summary: str, key: dict):
    """Compute memory_fidelity_score / memory_breakdown on an overview result (keeps the text field intact)."""
    if not isinstance(result, dict) or "error" in result:
        return result
    window_text = "\n".join(f"{u}\n{r}" for u, r in history)
    notes_by_id = {n["id"]: n for n in spreadsheet.get("notes", [])}
    score, breakdown = score_memory_checks(result.get("memory_checks"), key, state_summary, window_text,
                                           wrong_character=bool(result.get("wrong_character")),
                                           notes_by_id=notes_by_id)
    result["memory_fidelity_score"] = score
    result["memory_breakdown"] = breakdown
    return result


def render_memory_summary(state_dir: Path | None) -> str:
    """Bounded DSS-state inventory for guide/judge/auditor/overview (single shared renderer).

    Every character, group, element, event and arc is listed with per-entry detail truncated so the
    list is never cut short; ``_history`` keys are stripped. Any other subject file in the state dir
    (for example ``knowledge.json``) is rendered generically, so the judge sees every subject the
    engine saves, not a hard-coded subset.
    """
    if not state_dir or not state_dir.exists():
        return "(no structured memory yet)"
    parts = []

    def compact_entries(data: dict, per_entry: int = 50000, max_total: int = 200000) -> str:
        entries = data.get("entries", data) if isinstance(data, dict) else {}
        if not isinstance(entries, dict) or not entries:
            return ""
        lines = []
        for name, entry in entries.items():
            try:
                blob = json.dumps(entry, ensure_ascii=False, indent=2)[:per_entry]
            except Exception:
                blob = ""
            lines.append(f"- {name}: {blob}")
        total = "\n".join(lines)
        if len(total) > max_total:
            total = "\n".join(f"- {n}" for n in entries.keys())
        return total[:max_total]

    for fname in ("characters.json", "groups.json", "elements.json"):
        p = state_dir / fname
        if p.exists():
            try:
                data = _strip_history_keys(json.loads(p.read_text(encoding="utf-8")))
                summary = compact_entries(data)
                if summary:
                    parts.append(f"== {fname.removesuffix('.json')} ==\n" + summary)
                else:
                    parts.append(f"== {fname.removesuffix('.json')} == (EMPTY — no entries saved)")
            except Exception:
                continue
        else:
            parts.append(f"== {fname.removesuffix('.json')} == (EMPTY — no entries saved)")

    for fname in ("events.json", "arcs.json"):
        p = state_dir / fname
        if p.exists():
            try:
                data = _strip_history_keys(json.loads(p.read_text(encoding="utf-8")))
                if not data:
                    parts.append(f"== {fname.removesuffix('.json')} == (EMPTY — nothing saved)")
                else:
                    parts.append(f"== {fname.removesuffix('.json')} ==\n" + json.dumps(data, ensure_ascii=False)[:50000])
            except Exception:
                continue
        else:
            parts.append(f"== {fname.removesuffix('.json')} == (EMPTY — nothing saved)")

    for fname in ("general_info.json", "current_scene.json"):
        p = state_dir / fname
        if p.exists():
            try:
                data = _strip_history_keys(json.loads(p.read_text(encoding="utf-8")))
                parts.append(f"== {fname} ==\n" + json.dumps(data, indent=1)[:50000])
            except Exception:
                continue
        else:
            parts.append(f"== {fname} == (EMPTY — no data saved)")

    rendered = {"characters.json", "groups.json", "elements.json", "events.json", "arcs.json",
                "general_info.json", "current_scene.json", *_NON_SUBJECT_JSON}
    for p in sorted(state_dir.glob("*.json")):
        if p.name in rendered:
            continue
        try:
            data = _strip_history_keys(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
        stem = p.name.removesuffix(".json")
        if not data:
            parts.append(f"== {stem} == (EMPTY — nothing saved)")
            continue
        summary = compact_entries(data) if isinstance(data, dict) and isinstance(data.get("entries", data), dict) else ""
        parts.append(f"== {stem} ==\n" + (summary or json.dumps(data, ensure_ascii=False)[:50000]))
    return "\n".join(parts) if parts else "(no structured memory yet)"



# ---------------------------------------------------------------------------
# Retrieval precision (deterministic, no LLM): how accurately DSS's memory was
# actually INSERTED into the writer's context this turn. Pure function of the
# rendered payload + the state the retriever saw + the story so far, so it can
# be recomputed offline from turn_NNN/retrieval_payload.json.
#
# The payload is the engine's artificial history ([prompt, text] pairs) plus
# custom_state["context"]. Entity blocks are recognised by the format-template
# path markers ("<<<<<<<<<<<< .Name.field"), which every rendered entity line
# carries, so the detail each entity received is measured, not guessed. The
# subject comes from the section the block sits in (the marker path may or may
# not repeat it).
# ---------------------------------------------------------------------------
_MARKER_RE = re.compile(r"<{8,}\s*(?P<path>[^\n]*?)\s*$")
_ENTITY_SUBJECTS = ("characters", "groups", "elements")
_LINE_ITEM_RE = re.compile(r"^\s*\[\d+\]\s+(?P<label>[^:\n]{0,90}):\s*(?P<text>.+?)\s*$", re.M)
_LEAK_RE = re.compile(r"\{\{|\}\}|\{%|%\}|\|\s*join\b|\bd\.description\b|\(\(\(")
_TIER_OF_TYPE = {
    "current_scene": "short_term", "current_scene2": "short_term", "recent_dialogue": "short_term",
    "general_info": "history", "events": "history", "chapters": "history", "arcs": "history",
    "message_summaries": "history", "canon": "history",
    "lines": "long_term_messages",
    "characters": "entities", "character_list": "entities", "groups": "entities",
    "elements": "entities", "knowledge": "entities",
}
RELEVANCE_WINDOW = 10      # exchanges of story that count toward an entity's relevance
RELEVANCE_DECAY = 0.8      # weight of each exchange further back
RELEVANCE_HIGH = 0.8       # >= this: mentioned in the current or previous exchange
RELEVANCE_JUSTIFIED = 0.2  # >= this: mentioned within roughly the last 6 exchanges
RECENT_NOTE_SILENCE = 8    # note silent for fewer exchanges than this = "recent", else "long-term"


def _norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def _term_regex(term: str):
    t = re.escape(term.strip())
    return re.compile(rf"(?<![A-Za-z0-9]){t}(?![A-Za-z0-9])", re.I) if t else None


def _last_seen(terms: list[str], history: list[list[str]]) -> int | None:
    terms = [t.strip().lower() for t in terms if t and t.strip()]
    if not terms:
        return None
    for i in range(len(history) - 1, -1, -1):
        blob = f"{history[i][0] or ''}\n{history[i][1] or ''}".lower()
        if any(t in blob for t in terms):
            return i
    return None


def build_entity_index(state_dir: Path | None) -> dict:
    """{(subject, name): {terms, importance, text, _res}} for characters/groups/elements in a state dir."""
    index: dict[tuple[str, str], dict] = {}
    if not state_dir or not Path(state_dir).exists():
        return index
    for subject in _ENTITY_SUBJECTS:
        p = Path(state_dir) / f"{subject}.json"
        if not p.exists():
            continue
        try:
            data = _strip_history_keys(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
        entries = data.get("entries", data) if isinstance(data, dict) else {}
        if not isinstance(entries, dict):
            continue
        for name, entry in entries.items():
            if not isinstance(name, str) or not name.strip():
                continue
            terms = [name]
            aliases = entry.get("aliases") if isinstance(entry, dict) else None
            if isinstance(aliases, dict):
                terms += [str(v) for v in aliases.values()]
            elif isinstance(aliases, list):
                terms += [str(v) for v in aliases]
            imp = None
            if isinstance(entry, dict) and isinstance(entry.get("importance"), dict):
                try:
                    imp = int(entry["importance"].get("score"))
                except Exception:
                    imp = None
            index[(subject, name)] = {"terms": [t for t in terms if len(t.strip()) >= 3], "importance": imp,
                                      "text": json.dumps(entry, ensure_ascii=False).lower()}
    # first/last-name partials for people, only when unique across the whole universe
    counts: dict[str, int] = {}
    for (subject, name) in index:
        for tok in set(_norm_name(name).split()):
            counts[tok] = counts.get(tok, 0) + 1
    for (subject, name), info in index.items():
        if subject != "characters":
            continue
        for tok in set(_norm_name(name).split()):
            if len(tok) >= 4 and counts.get(tok) == 1 and tok not in {t.lower() for t in info["terms"]}:
                info["terms"].append(tok)
    for info in index.values():
        info["_res"] = [r for r in (_term_regex(t) for t in info["terms"]) if r]
    return index


def _mentioned(text: str, index: dict) -> set:
    return {k for k, info in index.items() if any(r.search(text) for r in info["_res"])}


def entity_relevance(index: dict, history: list[list[str]], user_input: str) -> dict:
    """Story-derived relevance of each entity right now (independent of DSS's own importance score)."""
    exchanges = [user_input or ""] + [f"{u}\n{r}" for u, r in reversed(history)]
    rel = {k: 0.0 for k in index}
    for i, text in enumerate(exchanges[:RELEVANCE_WINDOW]):
        w = RELEVANCE_DECAY ** i
        for k in _mentioned(text, index):
            rel[k] += w
    return rel


def parse_context_payload(pairs: list, section_types: dict | None = None) -> list[dict]:
    """Split the payload into sections: {type, tier, text}. ``pairs`` = [[prompt, text], ...]."""
    section_types = {re.sub(r"\s+", " ", (k or "")).strip(): v for k, v in (section_types or {}).items()}
    out = []
    for item in pairs or []:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        prompt, text = str(item[0] or ""), str(item[1] or "")
        key = re.sub(r"\s+", " ", prompt).strip()
        low = key.lower()
        if key == "__context__":
            typ = "context"
        elif key in section_types:
            typ = section_types[key]
        elif low.startswith("summaries of earlier messages"):
            typ = "message_summaries"
        elif low.startswith("analyze all of the above"):
            typ = "framing"
        elif low.startswith("current context"):
            typ = "retrieved_block"
        else:
            typ = "recent_dialogue"  # rolling mode: raw user/assistant pairs are the recent window
            text = f"{prompt}\n{text}"
        out.append({"type": typ, "tier": _TIER_OF_TYPE.get(typ, "overhead"), "text": text})
    return out


def _parse_markers(text: str, names_by_subject: dict[str, list[str]], subject: str | None) -> dict:
    """Per-entity rendered detail from marker lines: {(subject, name): {chars, segments, first}}."""
    all_names = [(s, n) for s, ns in names_by_subject.items() for n in ns]
    blocks: dict[tuple[str, str], dict] = {}
    buf: list[str] = []
    for line in text.splitlines():
        m = _MARKER_RE.search(line)
        if not m:
            buf.append(line)
            continue
        segment = "\n".join(buf + [line[:m.start()]]).strip()
        buf = []
        rest = m.group("path").split(", ")[0].lstrip(".")
        subj = subject
        pm = re.match(r"^(characters|groups|elements)\.(.*)$", rest)
        if pm:
            rest = pm.group(2)
            if not subj:  # the section the block sits in wins; the path prefix is only a fallback
                subj = pm.group(1)
        name = None
        if rest.startswith("["):
            end = rest.find("]")
            name = rest[1:end] if end > 0 else None
        if name is None:
            pool = [(subj, n) for n in names_by_subject.get(subj, [])] if subj else all_names
            for s_, cand in sorted(pool, key=lambda sn: len(sn[1]), reverse=True):
                if rest == cand or rest.startswith(cand + "."):
                    name, subj = cand, s_
                    break
        if name is None:
            name = rest.split(".")[0]
        if not subj:
            continue
        b = blocks.setdefault((subj, name), {"chars": 0, "segments": 0, "first": ""})
        b["chars"] += len(segment)
        b["segments"] += 1
        if not b["first"]:
            b["first"] = segment
    return blocks


def _form(block: dict) -> str:
    first = block.get("first", "")
    if "[past chapters]" in first:
        return "stale"
    if "dormant" in first.lower() and block.get("segments", 0) <= 1:
        return "dormant"
    return "detailed" if block.get("segments", 0) >= 3 else "brief"


_FORM_RANK = {"stale": 0, "dormant": 0, "brief": 1, "detailed": 2}


def retrieval_metrics(pairs: list, state_dir: Path | None, history: list[list[str]], user_input: str,
                      notes: list[dict] | None = None, due_note_ids: set[str] | None = None,
                      section_types: dict | None = None) -> dict:
    """Deterministic retrieval-precision report for one turn (see module comment).

    ``history`` is the story BEFORE this turn's exchange; ``state_dir`` is the state the retriever saw.
    """
    sections = parse_context_payload(pairs, section_types)
    if not sections:
        return {"available": False, "reason": "no payload captured"}
    index = build_entity_index(state_dir)
    all_text = "\n".join(s["text"] for s in sections)
    tier_chars: dict[str, int] = {}
    for s in sections:
        tier_chars[s["tier"]] = tier_chars.get(s["tier"], 0) + len(s["text"])
    total_chars = max(1, sum(tier_chars.values()))
    names_by_subject: dict[str, list[str]] = {}
    for (subject, name) in index:
        names_by_subject.setdefault(subject, []).append(name)
    blocks: dict[tuple[str, str], dict] = {}
    roster: set[tuple[str, str]] = set()
    for s in sections:
        default_subject = s["type"] if s["type"] in _ENTITY_SUBJECTS else ("characters" if s["type"] == "character_list" else None)
        for k, b in _parse_markers(s["text"], names_by_subject, default_subject).items():
            if s["type"] == "character_list":
                roster.add(k)  # a roster line is not a full entry; tracked separately
                continue
            cur = blocks.setdefault(k, {"chars": 0, "segments": 0, "first": ""})
            cur["chars"] += b["chars"]
            cur["segments"] += b["segments"]
            cur["first"] = cur["first"] or b["first"]
    report: dict = {"available": True, "n_entities_in_state": len(index), "n_entities_included": len(blocks),
                    "n_roster": len(roster),
                    "tiers": {"tokens": {t: c // 4 for t, c in tier_chars.items()},
                              "share": {t: round(c / total_chars, 3) for t, c in tier_chars.items()}},
                    "template_leak_chars": sum(len(l) for l in all_text.splitlines() if _LEAK_RE.search(l))}
    components: dict[str, float] = {}
    rel: dict = {}
    high: set = set()
    if index:
        rel = entity_relevance(index, history, user_input)
        high = {k for k, v in rel.items() if v >= RELEVANCE_HIGH}
    if index and blocks:
        adj_to_high = set()
        for h in high:
            h_text = index[h]["text"]
            for k, info in index.items():
                if k != h and (any(r.search(h_text) for r in info["_res"]) or any(r.search(info["text"]) for r in index[h]["_res"])):
                    adj_to_high.add(k)
        included = {k for k in blocks if k in index}
        justified = {k for k in included if rel[k] >= RELEVANCE_JUSTIFIED or k in adj_to_high}
        forms = {k: _form(blocks[k]) for k in included}
        report["included_by_subject"] = {s: sum(1 for k in included if k[0] == s) for s in _ENTITY_SUBJECTS}
        report["recall_high"] = round(len(high & included) / len(high), 3) if high else None
        report["n_high"] = len(high)
        report["missing_high"] = sorted(f"{k[0]}:{k[1]}" for k in high - included)[:8]
        mentioned_now = _mentioned(user_input or "", index)
        report["recall_mentioned_now"] = round(len(mentioned_now & included) / len(mentioned_now), 3) if mentioned_now else None
        report["precision_entities"] = round(len(justified) / len(included), 3) if included else None
        inc_chars = sum(blocks[k]["chars"] for k in included) or 1
        report["token_waste"] = round(sum(blocks[k]["chars"] for k in included - justified) / inc_chars, 3)
        report["unjustified"] = sorted(f"{k[0]}:{k[1]}" for k in included - justified)[:8]
        # detail calibration: relevant entities should get more rendered detail than irrelevant ones
        hi = [k for k in included if rel[k] >= RELEVANCE_HIGH]
        lo = [k for k in included if rel[k] < RELEVANCE_JUSTIFIED]
        # compared within a subject only: a character entry is not comparable in length to an element entry
        pairs = [(a, b) for a in hi for b in lo if a[0] == b[0]]
        if pairs:
            wins = sum(1.0 if blocks[a]["chars"] > blocks[b]["chars"] else 0.5 if blocks[a]["chars"] == blocks[b]["chars"] else 0.0
                       for a, b in pairs)
            report["detail_concordance"] = round(wins / len(pairs), 3)
            report["detail_high_tokens"] = round(sum(blocks[k]["chars"] for k in hi) / len(hi) / 4)
            report["detail_low_tokens"] = round(sum(blocks[k]["chars"] for k in lo) / len(lo) / 4)
        elif hi and lo:
            report["detail_concordance"] = None  # relevant and irrelevant entities never share a subject
        else:
            report["detail_concordance"] = None
        report["under_detailed_high"] = sorted(f"{k[0]}:{k[1]}" for k in hi if _FORM_RANK[forms[k]] < 2)[:8]
        report["over_detailed_low"] = sorted(f"{k[0]}:{k[1]}" for k in lo if _FORM_RANK[forms[k]] == 2)[:8]
        report["forms"] = {f: sum(1 for v in forms.values() if v == f) for f in ("detailed", "brief", "stale", "dormant")}
        if report["recall_high"] is not None:
            components["recall_high"] = report["recall_high"]
        if report["precision_entities"] is not None:
            components["precision_entities"] = report["precision_entities"]
        if report.get("detail_concordance") is not None:
            components["detail_concordance"] = report["detail_concordance"]
    elif index and high:
        # entities exist and are relevant, yet no entity block was recognised: that is itself a recall failure
        # (unless the payload simply has no markers, which `n_roster`/tiers will show)
        report["recall_high"] = 0.0 if any(s["tier"] == "entities" for s in sections) else None
        report["n_high"] = len(high)
        if report["recall_high"] is not None:
            components["recall_high"] = 0.0
    # retrieved earlier messages ("lines"): are they relevant, and not just repeating the recent window?
    line_items = [m.group("text") for s in sections if s["type"] == "lines" for m in _LINE_ITEM_RE.finditer(s["text"])]
    if line_items:
        recent_norm = _norm_name("\n".join(f"{u}\n{r}" for u, r in history[-4:]))
        notes_terms: list[str] = []
        for n in notes or []:
            notes_terms += [t.lower() for t in [n.get("needle", "")] + list(n.get("needle_syns") or []) if t]
        high_res = [r for k in high for r in index[k]["_res"]] if index else []
        user_words = set(re.findall(r"[a-z]{5,}", (user_input or "").lower()))
        relevant = dup = 0
        for item in line_items:
            low = item.lower()
            if any(r.search(item) for r in high_res) or any(t in low for t in notes_terms) \
                    or len(user_words & set(re.findall(r"[a-z]{5,}", low))) >= 2:
                relevant += 1
            head = _norm_name(item)[:80]
            if head and head in recent_norm:
                dup += 1
        report["lines"] = {"n": len(line_items), "precision": round(relevant / len(line_items), 3),
                           "dup_recent": round(dup / len(line_items), 3)}
        components["lines_precision"] = report["lines"]["precision"]
    # planted notes that are due now: is the detail actually in the context?
    due = [n for n in (notes or []) if due_note_ids and n.get("id") in due_note_ids and (n.get("needle") or "").strip()]
    if due:
        low_all = all_text.lower()
        rows = []
        for n in due:
            terms = [t.lower() for t in [n["needle"]] + list(n.get("needle_syns") or []) if t]
            if _first_seen([n["needle"]], history) is None:
                continue  # not in the story yet, nothing to retrieve
            last = _last_seen([n["needle"]], history)
            rows.append({"id": n["id"], "present": any(t in low_all for t in terms),
                         "silent": (len(history) - 1 - last) if last is not None else None})
        if rows:
            rec = [r for r in rows if r["silent"] is not None and r["silent"] < RECENT_NOTE_SILENCE]
            lng = [r for r in rows if r["silent"] is None or r["silent"] >= RECENT_NOTE_SILENCE]
            report["notes"] = {"due": len(rows), "recall": round(sum(r["present"] for r in rows) / len(rows), 3),
                               "recall_recent": round(sum(r["present"] for r in rec) / len(rec), 3) if rec else None,
                               "recall_long_term": round(sum(r["present"] for r in lng) / len(lng), 3) if lng else None,
                               "missing": [r["id"] for r in rows if not r["present"]][:8]}
            components["note_recall"] = report["notes"]["recall"]
    # short-term: the most recent exchange must actually be in the prompt
    if history:
        recent_secs = [s["text"] for s in sections if s["tier"] == "short_term"]
        probe = _norm_name(history[-1][1])[:60]
        ok = (bool(probe) and any(probe in _norm_name(t) for t in recent_secs)) if recent_secs else None
        report["short_term_last_present"] = ok
        if ok is not None:
            components["short_term"] = 1.0 if ok else 0.0
    weights = {"recall_high": 0.30, "precision_entities": 0.20, "detail_concordance": 0.15,
               "lines_precision": 0.10, "note_recall": 0.15, "short_term": 0.10}
    if components:
        wsum = sum(weights[k] for k in components)
        report["score"] = round(sum(weights[k] * v for k, v in components.items()) / wsum, 3)
    else:
        report["score"] = None
    report["components"] = {k: round(v, 3) for k, v in components.items()}
    return report


def section_types_from_templates(path: Path | None) -> dict:
    """{prompt text -> section type} from the sandbox's format_templates.json (_context_order)."""
    if not path or not Path(path).exists():
        return {}
    try:
        order = json.loads(Path(path).read_text(encoding="utf-8")).get("_context_order") or []
    except Exception:
        return {}
    out = {}
    for it in order:
        if isinstance(it, dict):
            p, t = it.get("prompt"), it.get("type")
        elif isinstance(it, (list, tuple)) and len(it) >= 2:
            t, p = it[0], it[1]
        else:
            continue
        if p and t:
            out[str(p)] = str(t)
    return out


# ---------------------------------------------------------------------------
# Beat audit helpers: decide what code can decide, evidence-check the rest.
# ---------------------------------------------------------------------------

def _state_entries(state_dir: Path | None) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not state_dir or not Path(state_dir).exists():
        return out
    for p in sorted(Path(state_dir).glob("*.json")):
        if p.name in _NON_SUBJECT_JSON:
            continue
        try:
            data = _strip_history_keys(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
        entries = data.get("entries", data) if isinstance(data, dict) else {}
        if isinstance(entries, dict):
            out[p.name.removesuffix(".json")] = entries
    return out


def beat_precheck(beat: dict, state_dir: Path | None, history: list[list[str]]) -> dict | None:
    """Code-side verdict for a beat with a structured expectation, or None if not applicable.

    Returns {"decisive": bool, "status": ..., "detail": ..., "facts": text}. Decisive verdicts:
    unassessable (the entity never appeared in the story), missing (no entry and the name is absent from
    the whole state), saved (an entry exists in the target subject and covers every expected attribute).
    Anything else is left to the LLM auditor, which is given the facts.
    """
    expect = beat.get("expect")
    if not isinstance(expect, dict) or not str(expect.get("name", "")).strip():
        return None
    name = str(expect["name"]).strip()
    if _first_seen([name], history) is None:
        return {"decisive": True, "status": "unassessable", "facts": "",
                "detail": f"Deterministic: '{name}' never appeared in the story, so DSS could not have saved it."}
    entries = _state_entries(state_dir)
    target = beat.get("subject")
    nn = _norm_name(name)
    found = None
    for subj in ([target] if target in entries else []) + [s for s in entries if s != target]:
        for key, entry in entries[subj].items():
            k = _norm_name(str(key))
            alias_hit = False
            if isinstance(entry, dict) and isinstance(entry.get("aliases"), (dict, list)):
                vals = entry["aliases"].values() if isinstance(entry["aliases"], dict) else entry["aliases"]
                alias_hit = any(_norm_name(str(a)) == nn for a in vals)
            if k == nn or alias_hit or (len(nn) >= 5 and (nn in k or (len(k) >= 5 and k in nn))):
                found = (subj, key, entry)
                break
        if found:
            break
    name_tokens = [t for t in nn.split() if len(t) >= 4] or nn.split()
    whole = " ".join(json.dumps(e, ensure_ascii=False).lower() for ents in entries.values() for e in ents.values())
    whole += " " + " ".join(str(k).lower() for ents in entries.values() for k in ents)
    if not found:
        if name_tokens and not any(t in whole for t in name_tokens):
            return {"decisive": True, "status": "missing", "facts": "",
                    "detail": f"Deterministic: '{name}' appeared in the story but nothing resembling it exists anywhere in DSS's memory."}
        return {"decisive": False, "status": None,
                "facts": f"- Deterministic pre-check: no entry keyed '{name}' was found, but parts of the name occur elsewhere in the state; it may be stored under a different key."}
    subj, key, entry = found
    text = json.dumps(entry, ensure_ascii=False).lower()
    cov, lines = [], []
    for attr, val in expect.items():
        if attr == "name":
            continue
        toks = re.findall(r"[a-z0-9]{4,}", str(val).lower())
        if not toks:
            continue
        hit = sum(1 for t in toks if t in text) / len(toks)
        cov.append(hit)
        lines.append(f"{attr}: {hit:.0%} of its words appear in the entry")
    facts = (f"- Deterministic pre-check: an entry '{key}' exists in the '{subj}' subject. Expected-attribute coverage: "
             + ("; ".join(lines) if lines else "(no attributes to compare)") + ".")
    if subj == target and cov and min(cov) >= 0.6:
        return {"decisive": True, "status": "saved", "facts": facts,
                "detail": f"Deterministic: entry '{key}' in '{subj}' covers every expected attribute ({'; '.join(lines)})."}
    if subj != target:
        facts += f" It is stored under '{subj}', not the beat's '{target}'."
    return {"decisive": False, "status": None, "facts": facts}


def verify_audit_verdict(verdict: dict, state_summary: str) -> dict:
    """Demote an LLM 'saved' that carries no quote actually found in the memory (leniency guard)."""
    if not isinstance(verdict, dict):
        return verdict
    status = str(verdict.get("status", "")).strip().lower()
    verdict["status_raw"] = verdict.get("status")
    if status == "saved":
        ev = str(verdict.get("evidence", ""))
        if not _quote_supported(ev, _text_tokens(state_summary)):
            verdict["status"] = "partial"
            verdict["audit_flag"] = "saved_without_verifiable_evidence"
    return verdict


def render_memory_names(state_dir: Path | None) -> str:
    """Names only: who and what exists in DSS's memory, without any detail (guide-memory 'names' mode)."""
    entries = _state_entries(state_dir)
    lines = [f"== {s} == " + ", ".join(str(k) for k in ents) for s, ents in entries.items() if ents]
    return "\n".join(lines) or "(no structured memory yet)"



_SCORE_ANCHORS = """SCORE CALIBRATION for prose-based scores (style_score, quality_score, overall_score). Memory fidelity is NOT scored on this ladder: it is verified item by item and computed from the checklist. Use the FULL scale; anchor each prose-based score to these meanings:
- 5.0 — Exceptional. A sentence or passage that would stand out in a published anthology: original imagery, precise diction, emotional resonance that goes beyond the expected. Zero flaws AND at least one moment of genuine surprise. Most turns will NOT reach this.
- 4.5 — Distinctive. Clearly above the norm of competent AI prose: a fresh angle, an unexpected detail, or a line of dialogue that reveals character in a way the instructions didn't mandate. The reader notices something specific and appreciable beyond "this is well-written."
- 4.0 — Solid. Competent, coherent, no flaws that call for revision. Reads well but does not surprise; the kind of prose that fills a page without distinguishing itself.
- 3.5 — Good. Readable with real craft, but carrying minor flaws (repetition, thin beats, generic imagery, a line that could be in any story) a careful reader would feel.
- 3.0 — Functional. Passable but clearly flawed; multiple notable issues that accumulate.
- 2.5 — Weak. Several significant flaws across dimensions.
- 2.0 — Poor. Fails most dimensions.
- 1.0–1.5 — Failing. Fundamentally broken output.
Calibration rules:
- A 3.5 is the DEFAULT for competent output. Do not inflate to 4.0+ for prose that is "well-written" in a generic sense.
- To score 4.0+, you must identify a SPECIFIC line, image, or choice that is distinctive — name it in your summary. If you cannot point to a concrete moment of distinction, the score is 3.5 or below.
- To score 4.5+, the distinction must be clear and appreciable: a fresh angle, an unexpected detail, or character-revealing dialogue that the instructions did not mandate.
- To score 5.0, the turn must contain something genuinely surprising or original that a human reader would highlight. If you are not certain, score 4.5.
- Compare this turn to the OTHER turns in the window: if it is comparable in quality to the surrounding turns, it is NOT extraordinary — score it relative to the best turn in the window, not in absolute terms.
- The MANDATORY FLOORS still cap scores regardless of this ladder."""


JUDGE_SYSTEM_TMPL = """You are a quality auditor grading ONE writer — {name2} ({role2}), the DSS-driven writer — on a single turn of a collaborative story. {name1} ({role1}) writes the OTHER writer's turns; those turns are provided ONLY as CONTEXT (what {name2} was responding to). Do not credit or penalize {name2} for anything {name1} wrote, and never score {name1}'s prose or the joint plot. The verdict measures {name2} alone.

You grade three things about {name2}:
(1) style_score: did {name2}'s reply hold THEIR directed voice ({dss_directive}) and its hard rules;
(2) quality_score: the narrative quality of {name2}'s reply itself (not the story's plot);
(3) memory fidelity: how faithfully {name2}'s private memory (the STATE, last in the user message) captured what the story was MEANT to establish and what it ACTUALLY established. You do NOT give a number for this. You verify specific items against the state and report evidence in "memory_checks"; the harness computes the score from your verdicts.
You also fill the per-note list for the reply (reply adherence, IN-BEAT notes only).

NOTES HAVE TWO SEPARATE JOBS:
- Reply adherence (the "notes" list): IN-BEAT NOTES are the notes whose prop/concept is part of THIS turn's active beat (the harness instruction text for this turn names it). Mark 'echoed' when the reply handles one, 'planted' when it introduces one, and 'missed' only when the reply drops an IN-BEAT note with no handling. A story that has moved on from a prop is not dropping it: NEVER mark a note that is not IN-BEAT as 'missed' in this list, and never let it lower style_score or quality_score.
- Memory retention (inside "memory_checks"): EVERY planted note in the MEMORY ANSWER KEY, in-beat or not, is checked against the STATE. Whether the reply mentions the note is irrelevant for this job; the only question is whether the memory recorded it.

{anchors}

SCORING RUBRIC for style_score and quality_score (be strict — a mediocre turn must NOT score 4+):
style_score:
- 5: the whole turn holds {name2}'s directed voice and hard rules with zero violations.
- 4: a single minor slip (one stray word in the wrong person — e.g. one 'I' when the directive demands third person, or one 'he/she' when it demands first — one slightly repetitive phrase) that does not break voice.
- 3: TWO or more slips, or ONE near-verbatim re-quote of {name1}, or a repeated closing line from an earlier turn.
- 2: recurring wrong-person narration throughout (e.g. first-person 'I' when {dss_directive} demands third person, or third person when it demands first, or a second-person 'you' that drifts to 'I'/'he'), or repeated verbatim re-quoting of {name1}, or a wholesale tonal break.
- 1: the reply is barely in character at all.
quality_score:
- 5: advances the scene with concrete action or observation; fresh details; no repetition, no continuity break.
- 4: competent but leans on imagery/atmosphere more than it moves the scene, or references few of the IN-BEAT notes (do not charge for PAST notes — those are continuity context, not a per-turn mandate).
- 3: repetitive imagery or metaphor, OR a scene inconsistency (a character present who is established absent, a location mismatch), OR the reply is largely atmospheric with no concrete action, dialogue, or new detail.
- 2: multiple scene inconsistencies or heavy self-repetition, or it ignores nearly all IN-BEAT notes.
- 1: incoherent, off-topic, or content-free.
MEMORY FIDELITY — VERIFY EACH ITEM, DO NOT JUDGE BY IMPRESSION
You never write a memory score. You fill in "memory_checks" and the harness computes the score. Three kinds of item are checked against the STATE:
A. PLANTED NOTES: details the outline meant to establish (MEMORY ANSWER KEY, part 1). Did the memory record each one?
B. MEMORY BEATS: specific saves the outline expected DSS to make, in a named subject, with expected data (MEMORY ANSWER KEY, part 2).
C. STORY FACTS: details the transcript itself established, whether EXPLICIT or IMPLICIT. YOU choose these (rules below).
Items listed as EXCLUDED appeared too recently for DSS to have saved them: ignore them, and do not add your own "too recent" excuses.

VERDICTS
- Notes (A) and story facts (C): retained | partial | missing | contradicted, and for facts also stale.
  retained = the state records the substance accurately. partial = the gist is there but a specific that a later scene would need is missing, vague or generalized (e.g. "a key" where the story says "the vestry key"; the right object in the wrong place). stale = the state holds an earlier version that the story has since changed. missing = nothing in the state records it. contradicted = the state says something incompatible with it.
- Beats (B): saved | partial | wrong | missing. saved = the expected entry exists with the expected data; partial = some of it; wrong = an entry exists but its data contradicts the expectation; missing = no such entry.

EVIDENCE RULES (these decide whether a verdict counts)
- Search the WHOLE state, every subject, not just the obvious one. A detail stored under another subject, key or wording still counts. Judge substance, not formatting or phrasing.
- "retained", "saved", "partial" and "stale" REQUIRE "state_quote": a short verbatim excerpt (15 words or fewer) copied from the state. The harness checks that the excerpt really appears in the state, and a quote that is not there downgrades the verdict. No quote means not retained.
- "missing" REQUIRES "looked_in": the subjects you searched.
- Absence is absence. Never assume the state "probably" holds something, and never credit it for something that only the transcript or the reply says. Consistency between the reply and the state is expected by construction and earns nothing.
- Never invent a contradiction. Use "contradicted" only when the state says something incompatible, and quote it in "state_quote".

CHOOSING STORY FACTS (C): select them from the transcript FIRST, and only then look in the state. Do not pick a fact because you can already see it in the state.
- List 6 to 8 facts that later scenes would most need to get right.
- At least 3 must be IMPLICIT: not stated in so many words but necessarily true given what was written (someone who hands over a key no longer holds it; someone who leaves through the door is no longer in the room; a reaction that shows a character now knows the secret; a motive the scene makes plain; the order in which things happened).
- At least 3 must come from the OLDER half of the window, so forgetting becomes visible.
- Cover different kinds: who is where and who is present; who holds or controls which object; relationships and attitudes that shifted; what each character knows, suspects or hides; promises, deadlines and goals; physical or emotional condition; the order and timing of events.
- Include a fact even if the harness instructions mandated it, as long as it appears in the story text. Skip trivia the story could drop without consequence.
- For each fact, "source" is a short verbatim quote (12 words or fewer) from the transcript that establishes it (for an implicit fact, the quote it follows from). "importance" is high (a later scene breaks if it is wrong), medium (it affects continuity) or low (color). The harness drops any fact whose source quote it cannot find in the transcript.

PRECISION AUDIT ("state_claims"): pick 4 to 6 entries or fields in the state about characters and elements active in the window, preferring fields that changed recently, and test each against the transcript and the notes. supported = the transcript or notes back it. contradicted = the transcript or notes show it is wrong or already superseded (quote that source in "source"). unverifiable = neither confirms nor refutes it, for example because it comes from before the window: this is NEUTRAL and never penalized. Never mark a claim contradicted without a source quote.

MANDATORY FLOORS (apply regardless of how pretty the prose is):
- If {name2} casts themself as the wrong character (e.g. as {name1}'s identity/role), set wrong_character to true: the harness caps memory fidelity at 2, and overall cannot reach 4. (Exception: if the premise establishes that {name1} and {name2} share one identity — a 'same-perspective' story — shared-identity narration is CORRECT.)
- If the turn re-quotes {name1} verbatim or repeats an earlier turn's closing line, style_score is at most 3.
- If a scene inconsistency exists, quality_score is at most 3.
- Missed notes lower quality only through the anchors above; a craft-strong reply (concrete action, new detail, fresh imagery) is NOT capped for missing notes. The per-note statuses carry the note-adherence verdict — never double-penalize the same missed note in both quality_score and the per-note list.

Style directives (for reference only — {name1}'s voice is not scored):
- {name1}'s voice ({role1}, guide turns): {guide_directive}
- {name2}'s voice ({role2}, DSS turns): {dss_directive}
{taboos}

ABORT CLAUSE: set "abort" to true ONLY if {name2}'s output is catastrophically broken — e.g. gibberish or
incoherent text, outright refusal or meta-commentary instead of an in-character reply, or a wholesale violation
of the hard style rules (wrong-person narration throughout PLUS re-quoting {name1} verbatim PLUS heavy repetition) that
persists across MULTIPLE consecutive turns. A single style slip, an occasional wrong-person word, a repetitive
phrase, or a missed note is NEVER enough — grade it with the scores instead and set abort=false. When in doubt,
set abort=false.

Return STRICT JSON only, exactly this shape. Inside each memory_checks item, fill the fields in the order shown (evidence before verdict). Use [] for a list with nothing to check:
{{
  "style_score": 0-5, may use 0.5 steps ({name2}'s voice adherence to {dss_directive}),
  "quality_score": 0-5, may use 0.5 steps (quality of {name2}'s reply),
  "notes": [{{"id": "<in-beat note id>", "status": "planted|echoed|missed|superseded|n/a", "detail": "one sentence on what {name2}'s turn did with this note"}}],
  "memory_checks": {{
    "story_facts": [{{"fact": "<one sentence>", "basis": "explicit|implicit", "importance": "high|medium|low", "source": "<12 words or fewer, quoted from the transcript>", "state_quote": "<15 words or fewer, quoted from the state, or empty>", "looked_in": "<subjects searched; required when missing>", "status": "retained|partial|stale|missing|contradicted"}}],
    "note_retention": [{{"id": "<answer-key note id>", "state_quote": "<15 words or fewer, quoted from the state, or empty>", "looked_in": "<subjects searched; required when missing>", "status": "retained|partial|missing|contradicted"}}],
    "beat_checks": [{{"id": "<answer-key beat id>", "state_quote": "<15 words or fewer, quoted from the state, or empty>", "looked_in": "<subjects searched; required when missing>", "status": "saved|partial|wrong|missing"}}],
    "state_claims": [{{"claim": "<what the state says>", "path": "<subject/key/field>", "source": "<quote from the transcript or notes; required when contradicted>", "status": "supported|contradicted|unverifiable"}}]
  }},
  "wrong_character": <true|false>,
  "abort": <true|false>,
  "summary": "two-sentence editorial note on {name2}'s turn only"
}}"""


def build_guide_messages(spreadsheet: dict, active_notes: list[dict], recent: list[list[str]],
                         guide_style: str, memory_summary: str,
                         plan_view: str = "", greeting: str = "") -> list[dict]:
    chars = spreadsheet["characters"]
    ws = spreadsheet["writing_style"]
    steering = GUIDE_STEERING[guide_style]
    taboos = "\n".join(f"- {t}" for t in ws.get("taboos", [])) or "- none"
    system = GUIDE_SYSTEM_TMPL.format(
        name1=chars["name1"]["name"], role1=chars["name1"].get("role", "the user"),
        name2=chars["name2"]["name"], role2=chars["name2"].get("role", "the other character"),
        premise=spreadsheet["premise"], setting=spreadsheet["setting"],
        guide_directive=ws["guide_directive"], taboos=taboos, steering_note=steering,
    )
    user_parts = []
    # Production parity: the character greeting opens the chat BEFORE the user's
    # first input (chat.py:1801), so on the very first turn {name1} responds to it
    # in the same scene. Cloud roles otherwise see only [user, reply] exchanges.
    if greeting.strip() and not recent:
        user_parts.append(
            f"The scene opens with {chars['name2']['name']}'s greeting:\n{greeting.strip()}\n\n"
            f"{chars['name1']['name']}'s first turn is the direct response to that greeting — "
            f"same scene, same moment, continuing where it left off."
        )
    user_parts.append(f"Private outline notes for THIS turn: {json.dumps([{'id': n['id'], 'content': n['content']} for n in active_notes])}"
                      if active_notes else "No private outline notes for this turn.")
    if plan_view:
        user_parts.append(f"Your current SHORT-TERM PLAN (your own working agenda for the next ~15 exchanges — follow it):\n{plan_view}")
    if memory_summary:
        user_parts.append(f"What the other writer currently believes (private memory summary):\n{memory_summary}")
    user_parts.append("Recent exchanges:")
    for u, r in recent[-8:]:
        user_parts.append(f"{chars['name1']['name']}: {u}")
        user_parts.append(f"{chars['name2']['name']}: {r}")
    user_parts.append("Write {name1}'s next turn now (one scene beat, in character).".format(name1=chars["name1"]["name"]))
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(user_parts)},
    ]


SYNTHETIC_REPLY_SYSTEM_TMPL = """You are writing {name2}'s ({role2}) reply to {name1}'s ({role1}) latest turn, as a HIGH-QUALITY substitute for the small local writer being tested. Write in strict {name2}-voice per the DSS writing-style directive — the person, tense, and register it specifies. One scene beat, 2-4 paragraphs, no meta-commentary, no recapping {name1}'s input verbatim."""


def build_synthetic_reply_messages(spreadsheet: dict, recent: list[list[str]], user_input: str) -> list[dict]:
    chars = spreadsheet["characters"]
    ws = spreadsheet["writing_style"]
    n1, n2 = chars["name1"], chars["name2"]
    system = SYNTHETIC_REPLY_SYSTEM_TMPL.format(
        name1=n1["name"], role1=n1.get("role", "the other character"),
        name2=n2["name"], role2=n2.get("role", "the protagonist"),
    )
    parts = [
        f"Writing-style directive for {n2['name']}: {ws['dss_directive']}",
        "Recent exchanges:",
    ]
    for u, r in recent[-8:]:
        parts.append(f"{n1['name']}: {u}")
        parts.append(f"{n2['name']}: {r}")
    parts.append(f"{n1['name']} just said: {user_input}")
    parts.append(f"Write {n2['name']}'s reply now.")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(parts)},
    ]



def build_judge_messages(spreadsheet: dict, in_beat_notes: list[dict], past_notes: list[dict],
                         window: list[list[str]],
                         state_summary: str, reply: str, synthetic: bool = False,
                         unplanted_ids: list[str] | None = None,
                         instructions: str = "",
                         memory_key: dict | None = None) -> list[dict]:
    chars = spreadsheet["characters"]
    ws = spreadsheet["writing_style"]
    taboos = "\n".join(f"- {t}" for t in ws.get("taboos", [])) or "- none"
    system = JUDGE_SYSTEM_TMPL.format(
        name1=chars["name1"]["name"], role1=chars["name1"].get("role", "the user"),
        name2=chars["name2"]["name"], role2=chars["name2"].get("role", "the other character"),
        guide_directive=ws["guide_directive"], dss_directive=ws["dss_directive"], taboos=taboos,
        anchors=_SCORE_ANCHORS,
    )
    name1 = chars["name1"]["name"]
    role1 = chars["name1"].get("role", "the user")
    name2 = chars["name2"]["name"]
    role2 = chars["name2"].get("role", "the other character")
    # Order matters: what to grade and the answer key come first, the transcript next, the STATE last,
    # so the judge chooses story facts from the transcript before it can anchor on what the state contains.
    user_parts = [
        f"Story premise (context): {spreadsheet['premise']}",
        "IN-BEAT NOTES (part of this turn's beat — grade {name2}'s reply adherence to these only, not {name1}'s)".format(name1=name1, name2=name2),
        json.dumps([{"id": n["id"], "type": n["type"], "content": n["content"]} for n in in_beat_notes], indent=1) if in_beat_notes else "(none — no planted note is part of this turn's beat)",
    ]
    if unplanted_ids:
        user_parts.append(
            f"{len(unplanted_ids)} outline note(s) never appeared in the story text and are NOT in "
            f"scope for this grade: {sorted(unplanted_ids)}. Never mark them 'missed' — the guide "
            f"never planted them, so {name2} could not have honored them."
        )
    _structure = spreadsheet.get("structure")
    if _structure:
        _notes_by_id = {n["id"]: n for n in spreadsheet.get("notes", [])}
        _lines = ["STORY STRUCTURE (expected chapter/arc boundaries, for context):"]
        for ch in _structure.get("chapters", []):
            _ref = _notes_by_id.get(ch.get("after_note", ""), {})
            _turn = _ref.get("plant", {}).get("turn", "?")
            _lines.append(
                f"- Chapter '{ch['label']}': expected around turn {_turn} "
                f"(±{ch.get('margin_scenes', 3)} scenes)"
            )
        for arc in _structure.get("arcs", []):
            _ref = _notes_by_id.get(arc.get("after_note", ""), {})
            _turn = _ref.get("plant", {}).get("turn", "?")
            _lines.append(
                f"- Arc '{arc['label']}': expected around turn {_turn} "
                f"(±{arc.get('margin_chapters', 1)} chapters)"
            )
        user_parts.append("\n".join(_lines))
    if instructions:
        user_parts.append(
            f"INSTRUCTIONS {name2} WAS GIVEN for this reply (the harness-generated instruction block):\n{instructions}\n"
            "ATTRIBUTION RULE: if a detail or phrasing in the reply comes VERBATIM from these instructions, that is "
            f"poor EXECUTION on {name2}'s part — a good writer performs an instruction in prose, it does not copy "
            f"the instruction's words. But do NOT penalize {name2} for content the instructions MANDATED (a prop, an "
            f"animal, a specified ending) as if it were {name2}'s own creative choice, and do NOT score it as a "
            f"memory error or identity confusion — {name2} was told to write it."
        )
    if memory_key is not None:
        user_parts.append(
            "MEMORY ANSWER KEY — part 1: PLANTED NOTES. Every note below needs one verdict in "
            "memory_checks.note_retention, checked against the STATE (in-beat or not; whether the reply "
            "mentions it is irrelevant). \"age_exchanges\" is how long ago the note first appeared in the story; \"silent_exchanges\" is how long since the story last mentioned it (a note that has been silent for many exchanges is the real test of memory)."
        )
        user_parts.append(json.dumps(memory_key.get("notes", []), indent=1) if memory_key.get("notes") else "(none)")
        user_parts.append(
            "MEMORY ANSWER KEY — part 2: MEMORY BEATS DSS was expected to have saved by now (the named "
            "subject, with the expected data). Every beat below needs one verdict in memory_checks.beat_checks."
        )
        user_parts.append(json.dumps(memory_key.get("beats", []), indent=1) if memory_key.get("beats") else "(none)")
        excluded = sorted(memory_key.get("pending_notes", []) + memory_key.get("pending_beats", []))
        if excluded:
            user_parts.append(f"EXCLUDED (appeared too recently for DSS to have saved them; do not check): {excluded}")
    elif past_notes:
        user_parts.append("PAST NOTES (planted earlier, NOT in this turn's beat — continuity context only):")
        user_parts.append(json.dumps([{"id": n["id"], "type": n["type"], "content": n["content"]} for n in past_notes], indent=1))
    user_parts.append(
        "TRANSCRIPT WINDOW — [CONTEXT] is {name1}'s guide turn (NOT scored); [EVALUATED] is {name2}'s DSS output "
        "(the object of grading). Choose the story facts from here.".format(name1=name1, name2=name2)
    )
    for u, r in window:
        user_parts.append(f"[CONTEXT — {name1} ({role1}), NOT scored]: {u}")
        user_parts.append(f"[EVALUATED — {name2} ({role2})]: {r}")
    if synthetic:
        user_parts.append(
            f"SYNTHETIC-CHAT ISOLATION MODE: the [EVALUATED] {name2} reply was written by the "
            f"CLOUD model, NOT by DSS. Fill in memory_checks (DSS's private memory is what is being "
            f"verified). Set style_score and quality_score to a neutral pass — DSS "
            f"did not write the reply this turn."
        )
    user_parts.append(f"[EVALUATED] Latest {name2} reply: {reply}")
    user_parts.append(f"[EVALUATED] {name2}'s private memory of the story so far (the STATE — verify against this, last):\n{state_summary}")
    user_parts.append(
        "Produce the JSON for {name2}'s turn now. Work in this order: (1) choose the story facts from the "
        "transcript, (2) give a verdict for EVERY answer-key note and beat, (3) audit state claims, "
        "(4) then style_score and quality_score.".format(name2=name2)
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(user_parts)},
    ]



FINAL_OVERVIEW_SYSTEM_TMPL = """You are a quality auditor evaluating ONE writer only: {name2} ({role2}) — the DSS-driven writer. {name1} ({role1}) wrote the OTHER writer's turns. This overview measures {name2}'s performance ALONE, never the story as a whole and never {name1}.

The story was written as an exchange: {name1} writes a turn, then {name2} replies. {name1}'s turns are provided ONLY as CONTEXT — they are the stimuli {name2} was responding to. Do not credit {name2} for ideas {name1} introduced, and do not penalize {name2} for plot or voice elements that came from {name1}'s turns. {name1}'s turns are context only; do not credit or penalize {name1} in any dimension. The verdict measures {name2} alone.

What is evaluated is {name2}'s craft as a writer and as the DSS agent:
- their adherence to their directed voice ({dss_directive}) and its hard rules;
- their use of their private memory of the story;
- how faithfully they honored the outline notes in their turns;
- how well their turns advanced or landed the intended beats.

SCORING RUBRIC (be strict — a mediocre run must NOT score 4+):
- overall_score is NOT the joint story quality and NOT a straight average of the sub-scores. Weigh:
  voice adherence ~30%, memory fidelity ~25%, note/clue usage ~25%, scene progression ~20%.
- voice: 5 = zero voice violations across the run; 4 = an occasional single slip; 3 = recurring slips or repeated verbatim re-quotes of {name1} or repeated closing lines; 2 = wholesale wrong-person drift (first person where the directive demands third, or vice versa, or a second-person 'you' slipping into 'I'/'he') or habitual re-quoting; 1 = never in character.
- memory fidelity (how faithfully the private memory captured what the outline meant to establish AND what the story actually established, explicit or implicit): this is VERIFIED item by item in memory_checks, not judged by impression and not by counting entries; the harness computes memory_fidelity_score from your verdicts. For overall_score take the memory component from that verification and from the MEMORY LEDGER (the computed per-turn scores). A tidy, well-populated state earns nothing by itself, and excellent prose must not lift it. If your checklist and the ledger's recent mean differ by more than 0.5, say why in the memory_fidelity text. Consistency between the replies and the state is expected by construction and earns nothing. (Exception: if the premise establishes that {name1} and {name2} share one identity — a 'same-perspective' story — shared-identity narration is CORRECT.)
- notes: grade ONLY the notes marked in scope (the unplanted ones are excluded — {name2} cannot honor a note the other writer never planted). 5 = most in-scope notes planted/echoed at the right turns; 4 = most notes used but some fumbled; 3 = roughly half the in-scope notes missed; 2 = the clear majority missed; 1 = notes essentially ignored.
- progression: 5 = their turns consistently advanced the beats; 4 = mostly advancing but some beats fumbled; 3 = imagery over action, several beats missed; 2 = rarely advanced anything; 1 = static or regressive.

{anchors}

MEMORY VERIFICATION (fill "memory_checks"; the harness computes the score). Check the END-OF-RUN memory against what the outline intended and what the story actually established, with the same evidence rules as the per-turn judge:
- note_retention: a verdict (retained | partial | missing | contradicted) for EVERY note id in the MEMORY ANSWER KEY. Search the whole state; judge substance, not wording.
- beat_checks: a verdict (saved | partial | wrong | missing) for every beat in the answer key.
- story_facts: 9 to 12 facts the story established, EXPLICIT or IMPLICIT, chosen from the transcript BEFORE you look at the state: at least 3 from the first third of the story ("era": "early"), 3 from the middle ("middle") and 3 from the last third ("late"), and at least 4 implicit ones (necessarily true though not stated outright: who now holds an object, who is where, who knows what, what a reaction reveals, the order of events). The eras exist to expose forgetting over the run: early items that are gone by the end are the main long-horizon failure.
- state_claims: 6 to 8 entries from the final state, tested against the transcript. contradicted needs a quoted source; unverifiable is neutral.
- retained, saved, partial and stale require "state_quote" (a verbatim excerpt of 15 words or fewer from the final state; the harness verifies it and downgrades a quote it cannot find). missing requires "looked_in". Each story fact needs a "source" quote (12 words or fewer) from the transcript; the harness drops a fact whose source it cannot find.
- A subject shown as "(EMPTY ...)" is a real memory failure: every item that belongs in it is missing. Never invent a contradiction, and never credit the state for something only the transcript says.
- Attribution: {name2} receives per-turn instructions generated by the harness; if a recurring flaw in their turns was explicitly mandated by those instructions (a specified prop, animal, or ending), weigh it as weak EXECUTION, not as {name2}'s own creative choice or a memory error.

MANDATORY FLOORS (apply regardless of how pretty the prose is):
- Recurring identity confusion ({name2} as the wrong character) caps overall_score at 3, and memory-fidelity discussion must call it out explicitly.
- More than half of the IN-SCOPE (planted) notes missed caps overall_score at 3.
- Any wholesale voice break (narration in the wrong person throughout, habitual verbatim re-quoting) caps overall_score at 3.
- A 5 overall requires near-zero flaws in all four dimensions; a 4 overall requires at most ONE moderately flawed dimension; anything with multiple flawed dimensions is 3 or below.

Be consistent with the per-turn judge records you are given: if the judge records show repeated low memory_fidelity or a high missed-note rate, your overall_score must reflect that. Do not let excellent prose inflate the overall score when memory or note usage failed.

Do NOT score the joint plot's inherent quality, do NOT score {name1}'s prose, and do NOT review the story "as a whole". Ignore how good or bad the resulting story happens to be; focus purely on how well {name2} executed their part.

Style directives (for reference only — {name1}'s voice is not scored):
- {name1}'s voice: {guide_directive}
- {name2}'s voice: {dss_directive}
{taboos}

You are given: the premise, the full outline notes, the entire transcript (each line labeled
CONTEXT or EVALUATED), the per-turn judge records produced during the run (compacted), the MEMORY
ANSWER KEY, the MEMORY LEDGER, and {name2}'s private memory at the end.

Return STRICT JSON only, exactly this shape:
{{
  "overall_score": 0-5, may use 0.5 steps — score of how well {name2} performed as the DSS writer per the SCORING RUBRIC above (voice adherence ~30%, memory fidelity ~25%, note/clue usage ~25%, scene progression ~20%; apply the MANDATORY FLOORS) — NOT joint story quality,
  "verdict": "one-sentence verdict on {name2}'s performance alone",
  "arc_progression": "2-3 sentences on how well {name2}'s turns advanced or landed the intended beats (what the DSS contributed), not how good the plot is",
  "style_consistency": "2-3 sentences on whether {name2} held THEIR directed voice ({dss_directive}) and its hard rules; {name1}'s voice is not scored",
  "memory_fidelity": "2-3 sentences summarizing the memory checklist: what was retained, the most important misses, and whether early-story items decayed",
  "memory_checks": {{
    "story_facts": [{{"fact": "<one sentence>", "era": "early|middle|late", "basis": "explicit|implicit", "importance": "high|medium|low", "source": "<12 words or fewer, quoted from the transcript>", "state_quote": "<15 words or fewer, quoted from the final state, or empty>", "looked_in": "<subjects searched; required when missing>", "status": "retained|partial|stale|missing|contradicted"}}],
    "note_retention": [{{"id": "<answer-key note id>", "state_quote": "<...>", "looked_in": "<...>", "status": "retained|partial|missing|contradicted"}}],
    "beat_checks": [{{"id": "<answer-key beat id>", "state_quote": "<...>", "looked_in": "<...>", "status": "saved|partial|wrong|missing"}}],
    "state_claims": [{{"claim": "<what the state says>", "path": "<subject/key/field>", "source": "<quote; required when contradicted>", "status": "supported|contradicted|unverifiable"}}]
  }},
  "wrong_character": <true|false>,
  "notes": [{{"id": "<note id>", "status": "planted|echoed|missed|superseded", "detail": "one sentence on what {name2}'s turns did with this note"}}],
  "strengths": ["2-4 strengths specific to {name2}'s turns"],
  "weaknesses": ["2-4 weaknesses specific to {name2}'s turns"],
  "summary": "3-4 sentence final editorial summary of {name2}'s performance alone; do not evaluate {name1} or the joint plot"
}}"""


def build_final_overview_messages(spreadsheet: dict, notes: list[dict], history: list[list[str]],
                                  judge_records: list[dict], state_summary: str,
                                  scope: str = "final", unplanted_ids: list[str] | None = None,
                                  memory_key: dict | None = None, ledger_text: str = "") -> list[dict]:
    chars = spreadsheet["characters"]
    ws = spreadsheet["writing_style"]
    taboos = "\n".join(f"- {t}" for t in ws.get("taboos", [])) or "- none"
    system = FINAL_OVERVIEW_SYSTEM_TMPL.format(
        name1=chars["name1"]["name"], role1=chars["name1"].get("role", "the user"),
        name2=chars["name2"]["name"], role2=chars["name2"].get("role", "the other character"),
        guide_directive=ws["guide_directive"], dss_directive=ws["dss_directive"], taboos=taboos,
        anchors=_SCORE_ANCHORS,
    )
    if scope == "rolling":
        system = (
            "MID-RUN ROLLING OVERVIEW — this is NOT the final verdict. The story is still in "
            "progress; the transcript below stops partway through the intended arc. Review only the "
            "turns that have happened so far: score {name2}'s performance up to this checkpoint, "
            "report the trajectory (improving/stable/declining) vs what a longer run would need, and "
            "flag any pattern that must be corrected before it hardens. Keep every field to the same "
            "JSON shape (overall_score 0-5 with 0.5 steps per the rubric).\n\n".format(name2=chars["name2"]["name"])
            + system
        )
    name1 = chars["name1"]["name"]
    role1 = chars["name1"].get("role", "the user")
    name2 = chars["name2"]["name"]
    role2 = chars["name2"].get("role", "the other character")
    user_parts = [f"Story premise (context): {spreadsheet['premise']}"]
    user_parts.append("Outline notes (the intended arc; the in-scope ones are also the MEMORY ANSWER KEY part 1 — see below):")
    user_parts.append(json.dumps([{"id": n["id"], "type": n["type"], "content": n["content"]} for n in notes], indent=1))
    if unplanted_ids:
        user_parts.append(
            f"The following outline note(s) NEVER appeared in the story text — the other writer "
            f"never planted them — so they are NOT in scope and must be marked 'superseded', never "
            f"'missed': {sorted(unplanted_ids)}"
        )
    if memory_key is not None:
        user_parts.append(
            "MEMORY ANSWER KEY — part 1: give a note_retention verdict for each of these note ids (their "
            "content is in the outline notes above; style-hold notes are not memory items): "
            f"{[n['id'] for n in memory_key.get('notes', [])]}"
        )
        user_parts.append(
            "MEMORY ANSWER KEY — part 2: MEMORY BEATS DSS was expected to have saved (give each a beat_checks verdict):"
        )
        user_parts.append(json.dumps(memory_key.get("beats", []), indent=1) if memory_key.get("beats") else "(none)")
        _excl = sorted(memory_key.get("pending_notes", []) + memory_key.get("pending_beats", []))
        if _excl:
            user_parts.append(f"EXCLUDED (appeared too recently for DSS to have saved them; do not check): {_excl}")
        if ledger_text:
            user_parts.append("MEMORY LEDGER (computed from the per-turn judge checks during the run):\n" + ledger_text)
    _structure = spreadsheet.get("structure")
    if _structure:
        _notes_by_id = {n["id"]: n for n in spreadsheet.get("notes", [])}
        _lines = ["STORY STRUCTURE (expected chapter/arc boundaries — assess whether DSS archived at the right times):"]
        for ch in _structure.get("chapters", []):
            _ref = _notes_by_id.get(ch.get("after_note", ""), {})
            _turn = _ref.get("plant", {}).get("turn", "?")
            _lines.append(
                f"- Chapter '{ch['label']}': expected around turn {_turn} "
                f"(±{ch.get('margin_scenes', 3)} scenes)"
            )
        for arc in _structure.get("arcs", []):
            _ref = _notes_by_id.get(arc.get("after_note", ""), {})
            _turn = _ref.get("plant", {}).get("turn", "?")
            _lines.append(
                f"- Arc '{arc['label']}': expected around turn {_turn} "
                f"(±{arc.get('margin_chapters', 1)} chapters)"
            )
        user_parts.append("\n".join(_lines))
    user_parts.append(
        "Full transcript — [CONTEXT] lines are {name1}'s guide turns (read them to understand what {name2} was "
        "responding to, but do NOT score them); [EVALUATED] lines are {name2}'s DSS output, the object of this review."
        .format(name1=name1, name2=name2)
    )
    for u, r in history:
        user_parts.append(f"[CONTEXT — {name1} ({role1}), NOT scored]: {u}")
        user_parts.append(f"[EVALUATED — {name2} ({role2})]: {r}")
    user_parts.append(f"[EVALUATED] {name2}'s private memory at the end (verify the answer key against this):\n{state_summary}")
    if judge_records:
        user_parts.append("Per-turn judge records (from during the run; these already scored {name2} only):".format(name2=name2))
        for rec in judge_records:
            user_parts.append(json.dumps(compact_judge_record(rec) if memory_key is not None else rec, indent=1))
    if scope == "rolling":
        user_parts.append(
            "Write a MID-RUN rolling overview of {name2}'s performance up to this checkpoint "
            "(DSS-only verdict, NOT the conclusion). Also include a \"trajectory\" field "
            "(\"improving\" | \"stable\" | \"declining\").".format(name2=name2)
        )
    else:
        user_parts.append("Write the final overview of {name2}'s performance now (DSS-only verdict).".format(name2=name2))
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(user_parts)},
    ]


PLAN_SYSTEM_TMPL = """You are the author-director of a collaborative fiction written by two writers.You write {name1}'s turns; the other writer ({name2}) writes {name2}'s turns and keeps a private memory of the story.

You keep a SHORT-TERM PLAN for the next ~15 exchanges of the story. The plan is your working agenda:
which beats to land, which outline notes to pull in, and in what order. You revise it on every replan.

Your private reference is the FULL story outline below — use as much or as little of it as you like.
Anything you want active over the next ~15 exchanges must be copied into the plan (it is the ONLY thing
that follows you between replans; the outline is not re-sent on steering turns).

Full story outline:
{outline}

Writing style of the piece: {directive}
{taboos}

Previous plan (may be empty on first replan):
{prev_plan}

DSS's private memory of the story so far (the other writer's view):
{memory}

Recent exchanges:
{recent}

Now revise the short-term plan for the next ~15 exchanges. Produce STRICT JSON only, exactly this shape:
{{
  "intent": "one-sentence near-future direction",
  "beats": [
    {{
      "turn": <turn offset 0..14 from the next exchange>,
      "action": "what {name1} should do/say/plant this exchange",
      "note_ids": ["outline note ids this beat uses"],
      "recalled": "concrete details pulled from the outline to work in here"
    }}
  ],
  "style_reminder": "one-line voice note for the window"
}}
Return nothing but the JSON object."""


def build_plan_messages(spreadsheet: dict, prev_plan: dict, recent: list[list[str]],
                        memory: str) -> list[dict]:
    """Messages for the replan call: the guide gets the FULL outline + plan context."""
    chars = spreadsheet["characters"]
    ws = spreadsheet["writing_style"]
    taboos = "\n".join(f"- {t}" for t in ws.get("taboos", [])) or "- none"
    outline = json.dumps(spreadsheet["notes"], indent=1)
    recent_txt = "\n".join(f"{chars['name1']['name']}: {u}\n{chars['name2']['name']}: {r}"
                           for u, r in recent[-8:]) or "(story not started)"
    prev_txt = json.dumps(prev_plan, indent=1) if prev_plan else "(no previous plan)"
    system = PLAN_SYSTEM_TMPL.format(
        name1=chars["name1"]["name"], name2=chars["name2"]["name"],
        outline=outline, directive=ws["directive"], taboos=taboos,
        prev_plan=prev_txt, memory=memory or "(no structured memory yet)",
        recent=recent_txt,
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "Revise the short-term plan now (next ~15 exchanges)."},
    ]


def _repair_truncated_json(raw: str) -> str:
    """Best-effort repair of a JSON string truncated mid-document.

    Drops the final incomplete value (everything after the last ``,\n`` or
    ``{``/``[`` that never got a closer), then closes all open brackets.
    Returns the original text if no repair is possible.
    """
    text = raw.strip()
    if not text:
        return text
    # Trim trailing garbage that isn't a plausible JSON document end.
    # If the tail ends mid-string (inside an unclosed "..."), the fallback
    # loop below will find a valid prefix, so only strip obvious non-JSON.
    while text and text[-1] not in "}]":
        # Don't strip a closing quote if it closes a string; the fallback handles it.
        if text[-1] == '"':
            break
        text = text[:-1]
    pairs = {")": "(", "]": "[", "}": "{"}
    closers = set(pairs)
    close_for = {"{": "}", "[": "]", "(": ")"}
    openers = set(close_for)
    stack = []
    for ch in text:
        if ch in openers:
            stack.append(ch)
        elif ch in closers and stack and stack[-1] == pairs[ch]:
            stack.pop()
        elif ch in closers and stack:
            break  # mismatched closer -> drop everything from here
    repair = "".join(close_for[c] for c in reversed(stack))
    try:
        json.loads(text + repair)
        return text + repair
    except Exception:
        # Fallback: keep the longest valid prefix by progressively closing.
        for cut in range(len(text) - 1, 0, -1):
            candidate = text[:cut]
            stack = []
            for ch in candidate:
                if ch in openers:
                    stack.append(ch)
                elif ch in closers and stack and stack[-1] == pairs[ch]:
                    stack.pop()
            candidate = candidate + "".join(close_for[c] for c in reversed(stack))
            try:
                json.loads(candidate)
                return candidate
            except Exception:
                continue
    return text


def parse_plan(raw: str, prev_plan: dict) -> dict:
    """Parse the replan response into a validated plan; fall back to prev on error."""
    if not raw:
        return prev_plan
    try:
        start = raw.index("{")
        end = raw.rindex("}")
        obj = json.loads(raw[start:end + 1])
    except Exception:
        # Truncation repair: close open brackets/parens/strings iteratively.
        repaired = _repair_truncated_json(raw)
        try:
            obj = json.loads(repaired)
        except Exception:
            return prev_plan
    if not isinstance(obj, dict):
        return prev_plan
    beats = obj.get("beats")
    if not isinstance(beats, list):
        obj["beats"] = []
    else:
        clean = []
        for b in beats:
            if isinstance(b, dict) and "action" in b:
                clean.append({
                    "turn": int(b.get("turn", 0)),
                    "action": str(b.get("action", "")),
                    "note_ids": b.get("note_ids", []) if isinstance(b.get("note_ids"), list) else [],
                    "recalled": str(b.get("recalled", "")),
                })
        obj["beats"] = clean
    obj["intent"] = str(obj.get("intent", ""))
    obj["style_reminder"] = str(obj.get("style_reminder", ""))
    return obj


AUDITOR_SYSTEM_TMPL = """You are a MEMORY AUDITOR for a collaborative story. Another system (DSS) maintains
a private structured memory of the story as JSON subject files (characters, groups, elements,
events, current_scene, general_info, and any other subject shown, such as knowledge).

The spreadsheet scheduled a memory beat with a target turn. The scheduled turn is the EARLIEST
intended save, NOT a deadline: the beat is satisfied whenever the expected entry is saved — once
the entity has appeared in the story and the target subject is writable — regardless of which turn
the save lands on. You are shown the beat's expectation, a DSS CAPABILITY CONTEXT (deterministic
facts about what DSS could have saved by the EVALUATION turn shown in the message), and DSS's actual
private memory at the evaluation point. Judge whether DSS did what the beat asked against the
EVALUATION turn, not the scheduled turn.

ABORT CLAUSE: set "abort" to true ONLY if DSS's ENTIRE private memory is catastrophically broken —
completely empty, gibberish, or wholly unrelated to the story across all subjects. A single missed
beat or an imperfect entry is NEVER enough; grade it with "status"/"detail" instead and set abort=false.
When in doubt, set abort=false.

Return STRICT JSON only, exactly this shape:
{{
  "id": "<beat id>",
  "evidence": "15 words or fewer quoted VERBATIM from DSS's memory that show the expected data, or empty if none",
  "looked_in": "the subjects you searched (required when the status is missing)",
  "status": "saved|partial|wrong|missing|unassessable",
  "abort": <true|false>,
  "detail": "one-two sentences: what DSS actually saved (or failed to save), and what is wrong if anything"
}}

EVIDENCE RULES: search the WHOLE memory, every subject, not only the beat's subject; judge substance, not
wording. "saved" and "partial" REQUIRE "evidence": a short verbatim excerpt copied from DSS's memory. The
harness checks that the excerpt really appears there, and a "saved" without a verifiable quote is downgraded.
Absence is absence: never assume the memory "probably" holds something, and never credit it for what only
the story says. If a DETERMINISTIC PRE-CHECK is given, treat its facts as ground truth about what is present.

- "saved": the expected entry exists and contains the expected data.
- "partial": something of the expected data exists but is incomplete or mis-fitted.
- "wrong": an entry exists but the data contradicts the beat's expectation.
- "missing": no such entry was saved.
- "unassessable": the beat could not have been satisfied — the entity never appeared in the
  story by the EVALUATION turn, or the subject was frozen / could not be updated by then (per
  the DSS CAPABILITY CONTEXT you are given). Do NOT call this a DSS failure. If the expected
  entry IS present in the subject at the evaluation turn, mark "saved"/"partial" instead — a
  save that landed later than the scheduled turn still satisfies the beat."""


def build_audit_messages(spreadsheet: dict, beat: dict, state_summary: str,
                         capability_context: str = "", turn: int = 0, precheck_facts: str = "") -> list[dict]:
    """Messages for the auditor: the beat expectation + DSS's actual memory."""
    system = AUDITOR_SYSTEM_TMPL
    user_parts = [
        f"Story: {spreadsheet['premise']}",
        f"Memory beat {beat.get('id', '?')} (scheduled at turn {beat.get('turn', '?')} — "
        f"EVALUATED at turn {turn}; the scheduled turn is the earliest intended save, "
        f"not a deadline):",
        beat.get("content", beat.get("expectation", "")),
    ]
    if beat.get("subject"):
        user_parts.append(f"Beat targets the '{beat['subject']}' subject.")
    if beat.get("expect"):
        user_parts.append(f"Expected save: {json.dumps(beat['expect'], indent=1)}")
    if capability_context:
        user_parts.append(capability_context)
    if precheck_facts:
        user_parts.append("DETERMINISTIC PRE-CHECK (computed in code from DSS's memory):\n" + precheck_facts)
    user_parts.append(f"DSS's actual private memory at this point:\n{state_summary or '(no structured memory yet)'}")
    user_parts.append("Return your verdict as STRICT JSON now.")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(user_parts)},
    ]


# ---------------------------------------------------------------- run ------

def make_run_id(spreadsheet_id: str, args: argparse.Namespace) -> str:
    h = hashlib.sha1(json.dumps({
        "guide_style": args.guide_style,
        "specificity": args.specificity_profile,
        "director": args.director,
        "arc_breaks": args.arc_breaks,
        "force_chapter_turn": getattr(args, "force_chapter_turn", 0),
        "cadence_profile": getattr(args, "cadence_profile", None),
        "level": args.level,
        "seed": args.seed,
        "schema_type": getattr(args, "schema_type", 1),
        "synthetic_chat": bool(getattr(args, "synthetic_chat", False)),
        "message_mode": getattr(args, "message_mode", "enumerated"),
        "retrieval_placement": getattr(args, "retrieval_placement", "prompt_start"),
        "rolling_window": getattr(args, "rolling_window", 6),
        "rolling_summaries": getattr(args, "rolling_summaries", 0),
        # Sampling params are part of the run identity: a different temperature/
        # top_p/top_k would otherwise silently auto-resume a completed run with
        # the same seed and produce apples-to-oranges outputs.
        "temperature": getattr(args, "temperature", 0.7),
        "top_p": getattr(args, "top_p", 0.80),
        "top_k": getattr(args, "top_k", 20),
        "min_p": getattr(args, "min_p", 0.0),
        "repetition_penalty": getattr(args, "repetition_penalty", 1.0),
        "restate_map_context": getattr(args, "restate_map_context", "auto"),
        "restate_map_threshold": getattr(args, "restate_map_threshold", 10000),
    }, sort_keys=True).encode()).hexdigest()[:8]
    return f"{spreadsheet_id}__{h}"


def resolve_subjects_schema(args: argparse.Namespace) -> str | None:
    """Resolve the subjects_schema.json path (relative to the extension dir) per
    schema_types_and_test_matrix.md: explicit --schema, then per-genre+type,
    per-type, per-genre, then the legacy default. Returns None for the default.
    """
    if args.schema:
        return str(args.schema)
    genre = getattr(args, "spreadsheet", None)
    genre = genre.replace(".json", "") if genre else None
    stype = getattr(args, "schema_type", 1)
    if stype in (1, 3):
        return None
    names = []
    if stype == 2:
        if genre:
            names.append(f"user_data/example/schemas/subjects_schema_{genre}_sceneagg.json")
        names.append("user_data/example/schemas/subjects_schema_sceneagg.json")
    return names[0] if names and (EXTENSION_DIR / names[0]).exists() else (
        names[1] if len(names) > 1 and (EXTENSION_DIR / names[1]).exists() else None
    )


def build_state(spreadsheet: dict, seed: int, unique_id: str, history: list | None = None) -> dict:
    chars = spreadsheet["characters"]
    ws = spreadsheet["writing_style"]
    taboos = "\n".join(f"- {t}" for t in ws.get("taboos", [])) or "- none"
    n1, n2 = chars["name1"], chars["name2"]
    def _char_block(c: dict) -> str:
        parts = []
        for k, v in c.items():
            if isinstance(v, list):
                v = '; '.join(v)
            parts.append(f"{str(k).title()}: {v}")
        return "\n".join(parts)

    context = (
        f"Premise: {spreadsheet['premise']}\n"
        f"Setting: {spreadsheet['setting']}\n"
        f"Canonical cast: {n1['name']} = {n1.get('role', 'the other character')} "
        f"(NOT the protagonist; the other character). {n2['name']} = "
        f"{n2.get('role', 'the protagonist')} (the protagonist being roleplayed).\n"
        f"Your voice ({n2['name']}, DSS turns): {ws['dss_directive']}\n"
        f"Taboos: {taboos}\n"
        f"Characters:\n{_char_block(n1)}\n{_char_block(n2)}"
    )
    extra = spreadsheet.get("extra_context", "").strip()
    if extra:
        context += f"\n{extra}"
    greeting = spreadsheet.get("greeting", "").strip()
    if not greeting:
        greeting = f"You walk into {spreadsheet['setting'].split(',')[0].strip() or 'the scene'}."
    # Production shape (chat.py:1801): state["history"]["internal"] is TGWUI's
    # FULL internal history — the character greeting pair at index 0 (assistant
    # side, with the visible-chat marker in the user slot) followed by every
    # exchange. The engine reads [0][1] for the greeting (get_retrieval_context,
    # _populate_from_first_scene) and iterates the tail for DSS's own recent
    # replies (generate_instr_prompt's R1 negative-exemplar block). The harness
    # is a pure session emulator: it must not hand the engine a greeting-only
    # internal history, or R1 silently sees nothing.
    internal = [["<|BEGIN-VISIBLE-CHAT|>", greeting]]
    if history:
        internal += [list(e) for e in history]
    return {
        "context": context,
        "name1": chars["name1"]["name"],
        "name2": chars["name2"]["name"],
        "seed": seed,
        "unique_id": unique_id,
        "character_menu": f"soak_{spreadsheet['id']}",
        "greeting": greeting,
        "user_bio": n1.get("description", ""),
        "chat_template_str": "",
        "history": {"internal": internal},
        "truncation_length": 65536,
    }


class SoakRun:
    """Checkpointable soak run. Owns run_dir, engine binding, and the loop."""

    def __init__(self, spreadsheet: dict, args: argparse.Namespace, run_dir: Path,
                 local: LocalModel, guide: CloudModel, judge: CloudModel,
                 auditor: CloudModel | None = None):
        self.spreadsheet = spreadsheet
        self.args = args
        self.run_dir = run_dir
        self.local = local
        self.guide = guide
        self.judge = judge
        self.auditor = auditor
        self.history: list[list[str]] = []
        self.summarizer = None
        self.custom_state: dict | None = None
        self._stop = False
        self.abort_flags = 0
        self.plan: dict = {}
        self.judge_records: list[dict] = []
        self.audit_records: list[dict] = []
        # Background cloud workers. Per-turn judge/audit and rolling-overview
        # calls are pure side effects (reports, records, abort) — they don't
        # gate the next turn — so they run on dedicated worker threads, one per
        # CloudModel instance (serialized per role: each role's tasks run in
        # dispatch order on a single thread), and merge their verdicts into the
        # checkpointed result.json when they land. The guide stays on the main
        # thread (its output is the next turn's input).
        self._cloud_lock = threading.RLock()
        self._judge_q: "queue.Queue" = queue.Queue()
        self._audit_q: "queue.Queue" = queue.Queue()
        self._pending_judge: list = []
        self._pending_audit: list = []
        self._workers_started = False
        self._workers_joined = False
        self._clamp_note_windows()
        # Reply-side anti-transcription / anti-self-anchor guard state.
        self._last_reply: str | None = None
        # Backing dict for runtime.persistent_ui_state (forced unit boundaries).
        self._ui_state: dict = {}

    # --------------------------------------------------------- reply guard ----

    def _generate_reply(self, prompt: str, state: dict) -> str:
        """Reply generation with the transcription/self-anchor post-check.

        Deterministic, bounded: the first reply is checked against the steering
        prompt (transcription) and the previous reply (self-anchor); a collision
        triggers ONE regeneration appending the anti-repeat directive, and the
        less-colliding result is kept. Harness-side by necessity — production
        reply text is emitted by TGWUI outside the engine.
        """
        reply = self.local.complete(prompt, state).strip()
        reason = ""
        if _reply_transcribes_prompt(reply, prompt):
            reason = "transcribed the instruction block"
        elif _reply_self_anchors(reply, self._last_reply or ""):
            reason = "re-anchored on its own previous reply"
        if reason:
            soak_log(f"  reply post-check: draft {reason}; regenerating once ...")
            reply2 = self.local.complete(prompt + _REPLY_ANTI_REPEAT_DIRECTIVE, state).strip()
            if reply2:
                # Prefer the anti-repeat draft: even a draft that still collides
                # breaks the byte-anchoring loop better than the verbatim original.
                reply = reply2
        return reply

    # -------------------------------------------------- background cloud I/O ---

    def _start_cloud_workers(self) -> None:
        if self._workers_started:
            return
        self._workers_started = True
        threading.Thread(target=self._judge_worker, daemon=True,
                         name="soak-judge-worker").start()
        threading.Thread(target=self._audit_worker, daemon=True,
                         name="soak-audit-worker").start()

    def _judge_worker(self) -> None:
        while True:
            task = self._judge_q.get()
            if task is None:
                self._judge_q.task_done()
                return
            try:
                task()
            except Exception as e:
                soak_log(f"  judge worker error: {e}")
                traceback.print_exc()
            finally:
                self._judge_q.task_done()

    def _audit_worker(self) -> None:
        while True:
            task = self._audit_q.get()
            if task is None:
                self._audit_q.task_done()
                return
            try:
                task()
            except Exception as e:
                soak_log(f"  auditor worker error: {e}")
                traceback.print_exc()
            finally:
                self._audit_q.task_done()

    def flush_pending_tasks(self) -> None:
        """Enqueue judge/audit tasks queued by the last run_turn (call AFTER the
        turn's result.json is checkpointed so worker merges never race the write)."""
        for t in self._pending_judge:
            self._judge_q.put(t)
        self._pending_judge = []
        for t in self._pending_audit:
            self._audit_q.put(t)
        self._pending_audit = []

    def _join_cloud_workers(self) -> None:
        if not self._workers_started or self._workers_joined:
            return
        self._workers_joined = True
        soak_log("  joining background judge/audit workers ...")
        self._judge_q.put(None)
        self._audit_q.put(None)
        self._judge_q.join()
        self._audit_q.join()

    def _merge_into_result(self, turn: int, partial: dict) -> None:
        """Atomically merge a partial (judge/audit/usage) into the turn's result.json."""
        p = self.run_dir / f"turn_{turn:03d}" / "result.json"
        try:
            data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        except Exception:
            data = {}
        data.update(partial)
        tmp = p.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            tmp.replace(p)
        except Exception as e:
            soak_log(f"  cloud worker: could not merge into {p}: {e}")

    def _note_abort(self, aborted: bool) -> None:
        """Consecutive-extreme-failure early stop, fed by judge/audit verdicts
        landing asynchronously. Delayed by up to a turn vs the old inline check
        (the worker sets _stop; the main loop notices at the next loop top)."""
        limit = self.args.abort_after
        if not limit:
            return
        with self._cloud_lock:
            if aborted:
                self.abort_flags += 1
                soak_log(f"  [abort] flag {self.abort_flags}/{limit}: judge/auditor reported extreme DSS failure")
                if self.abort_flags >= limit:
                    self._stop = True
                    soak_log(f"  [abort] TRIGGERED after {limit} consecutive extreme-failure flags — will stop")
            else:
                self.abort_flags = 0

    def _make_judge_task(self, turn: int, active_notes: list[dict], reply: str,
                         state_dir: Path | None):
        window = list(self.history[-self.args.judge_window:])
        full_history = list(self.history)
        planted = set(self._planted_note_ids())
        judge = self.judge

        def task() -> None:
            t0 = time.time()
            soak_log(f"  judge: calling cloud ({judge.model}) json_mode ...")
            base = judge.stats()
            try:
                verdict = self._run_judge(turn, active_notes, reply, state_dir,
                                          window=window, planted=planted, full_history=full_history)
            except Exception as e:
                soak_log(f"  judge: task failed at turn {turn}: {e}")
                traceback.print_exc()
                verdict = {"error": str(e)}
            usage = _delta_stats(base, judge.stats())
            with self._cloud_lock:
                if isinstance(verdict, dict) and "error" not in verdict:
                    self.judge_records.append(verdict)
                    self._note_abort(bool(verdict.get("abort")))
                partial = {"judge": verdict}
                if usage:
                    partial["judge_usage"] = usage
                self._merge_into_result(turn, partial)
            soak_log(f"  judge: verdict in {_elapsed(t0)} -> {str(verdict)[:120]}... "
                     f"({usage.get('calls', 0)} calls)")

        return task

    def _make_audit_task(self, turn: int, state_dir: Path | None):
        auditor = self.auditor
        if auditor is None:
            return None

        def task() -> None:
            t0 = time.time()
            soak_log(f"  auditor: auditing DSS saves for turn {turn} ({auditor.model}) ...")
            base = auditor.stats()
            try:
                verdicts = self._run_audit(turn, state_dir) or []
            except Exception as e:
                soak_log(f"  auditor: task failed at turn {turn}: {e}")
                traceback.print_exc()
                verdicts = [{"id": "?", "status": "error", "detail": str(e), "turn": turn}]
            usage = _delta_stats(base, auditor.stats())
            with self._cloud_lock:
                self.audit_records.extend(
                    v for v in verdicts if isinstance(v, dict) and "error" not in v)
                self._note_abort(any(isinstance(v, dict) and v.get("abort") for v in verdicts))
                partial = {"audit": verdicts}
                if usage:
                    partial["auditor_usage"] = usage
                self._merge_into_result(turn, partial)
            soak_log(f"  auditor: {len(verdicts)} verdicts in {_elapsed(t0)} "
                     f"({usage.get('calls', 0)} calls)")

        return task

    def _make_rolling_overview_task(self, turn: int):
        state_dir = self._current_state_dir()
        history_snap = list(self.history)
        planted_snap = set(self._planted_note_ids())
        judge = self.judge

        def task() -> None:
            t0 = time.time()
            soak_log(f"  judge: rolling overview after {turn + 1} turns ({judge.model}) ...")
            base = judge.stats()
            with self._cloud_lock:
                records_snap = list(self.judge_records)
            try:
                rover = self._run_overview(state_dir, scope="rolling",
                                           history=history_snap, planted=planted_snap,
                                           judge_records=records_snap)
            except Exception as e:
                soak_log(f"  rolling overview skipped at turn {turn}: {e}")
                traceback.print_exc()
                return
            usage = _delta_stats(base, judge.stats())
            if rover:
                rover["turns"] = turn + 1
                rover["judge_records"] = records_snap
                rover["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                ov_path = self.run_dir / f"rolling_overview_{turn + 1:03d}.json"
                try:
                    ov_path.write_text(json.dumps(rover, indent=2), encoding="utf-8")
                    soak_log(f"rolling overview ({turn + 1} turns) written to {ov_path.name} "
                             f"({usage.get('calls', 0)} calls)")
                except Exception as e:
                    soak_log(f"rolling overview write failed: {e}")

        return task

    def backfill_post_turn_events(self, resume_from: int) -> None:
        """Backfill missing post-turn events for completed turns before resuming.

        When a soak run is killed mid-flight, in-flight judge/audit/overview
        tasks never complete. On resume we reconstruct and re-enqueue them
        for every completed turn, using that turn's point-in-time data.
        """
        print(f"[soak] backfilling post-turn events for turns 0-{resume_from-1}")
        saved_history = self.history
        self.history = []
        for t in range(resume_from):
            d = self.run_dir / f"turn_{t:03d}"
            result_path = d / "result.json"
            reply_path = d / "reply.txt"
            if not result_path.exists() or not reply_path.exists():
                continue
            result = json.loads(result_path.read_text(encoding="utf-8"))
            reply_text = reply_path.read_text(encoding="utf-8")
            # Live parity: a live run appends the exchange to self.history BEFORE it dispatches the
            # judge/overview, so the window includes the exchange being judged. Do the same here.
            user_path = d / "user.txt"
            if user_path.exists():
                self.history.append([user_path.read_text(encoding="utf-8"), reply_text])
            # Point-in-time state: the per-turn snapshot written at checkpoint, not the newest dir on disk
            # (which, for a finished run, is the FINAL state and would flatter every earlier turn).
            snap = d / "state_snapshot"
            point_in_time = snap if snap.exists() else None
            # Judge: missing or error (respect judge_every cadence)
            judge = result.get("judge")
            judge_ok = judge is not None and not (isinstance(judge, dict) and judge.get("error"))
            if self.args.judge_every and ((t + 1) % self.args.judge_every == 0) and not judge_ok:
                dss_state = point_in_time or self._current_state_dir()
                task = self._make_judge_task(t, [], reply_text, dss_state)
                if task:
                    self._judge_q.put(task)
            # Audit: missing (respect audit_every cadence)
            if self.args.audit_every and ((t + 1) % self.args.audit_every == 0) and not result.get("audit"):
                task = self._make_audit_task(t, point_in_time or d)
                if task:
                    self._audit_q.put(task)
            # Rolling overview: missing (respect overview_every cadence)
            ov_path = self.run_dir / f"rolling_overview_{t+1:03d}.json"
            if self.args.overview_every and ((t + 1) % self.args.overview_every == 0) and not ov_path.exists():
                self._judge_q.put(self._make_rolling_overview_task(t))
        self.history = saved_history
        print(f"[soak] backfill complete")

    def _clamp_note_windows(self) -> None:
        """B1: clamp recall due / plant turns inside the run horizon.

        Spreadsheet notes were authored for longer runs (due up to ~78); a short
        run would never reach them, so the recall probe never fires and report.py
        counts the note as failed. Normalize in-memory so every plant/recall can
        actually fire within ``args.turns`` (last probe turn = turns - 2).
        """
        horizon = max(1, self.args.turns - 2)
        clamped = 0
        for note in self.spreadsheet.get("notes", []):
            r = note.get("recall") or {}
            if r.get("due") is not None and int(r["due"]) > horizon:
                r["due"] = horizon
                clamped += 1
            p = note.get("plant") or {}
            if p.get("turn") is not None and int(p["turn"]) > horizon:
                p["turn"] = horizon
                clamped += 1
        if clamped:
            soak_log(f"  dss: clamped {clamped} plant/recall turns into run horizon ({horizon})")

    def _unresolved_beats(self, turn: int) -> list[dict]:
        """Beats scheduled at or before ``turn`` whose latest recorded audit verdict is not 'saved'."""
        latest: dict[str, str] = {}
        for v in sorted((r for r in self.audit_records if isinstance(r, dict)), key=lambda r: r.get("turn", 0)):
            if v.get("id") and v.get("status") != "error":
                latest[v["id"]] = v.get("status")
        return [b for b in self.spreadsheet.get("dss_beats", [])
                if b.get("turn") is not None and int(b["turn"]) <= turn and latest.get(b["id"]) != "saved"]

    def _retry_audit_due(self, turn: int) -> bool:
        every = int(getattr(self.args, "audit_retry_every", 0) or 0)
        return bool(every and turn > 0 and turn % every == 0 and self._unresolved_beats(turn))

    def _beats_due(self, turn: int) -> bool:
        """True if any spreadsheet dss_beat is scheduled for this turn."""
        for beat in self.spreadsheet.get("dss_beats", []):
            due = beat.get("turn")
            if due is not None and int(due) == turn:
                return True
        return self._retry_audit_due(turn)

    def _seed_general_info_style(self, history_path: Path | None) -> None:
        """Pin ``general_info.writing_style`` to the spreadsheet's ``dss_directive``.

        Production carries the DSS voice through ``general_info.writing_style``:
        the ``general_info`` format template ("Writing Style --- {{writing_style}}")
        is ``to_context`` and rendered into the system context every turn, and
        ``_format_general_info_static`` re-injects it once scenes archive. Seeding
        the exact directive (rather than the engine's lossy LLM paraphrase from
        ``state["context"]``) keeps the real voice flowing through the production
        path from turn 1 onward. Idempotent; re-runs each turn.
        """
        if not history_path:
            return
        dss = (self.spreadsheet.get("writing_style", {}).get("dss_directive") or "").strip()
        if not dss:
            return
        gi_path = history_path / "general_info.json"
        try:
            gi = json.loads(gi_path.read_text(encoding="utf-8")) if gi_path.exists() else {}
            changed = False
            if gi.get("writing_style") != dss:
                gi["writing_style"] = dss
                changed = True
            roster = {}
            for side in ("name1", "name2"):
                c = self.spreadsheet.get("characters", {}).get(side)
                if c:
                    roster[side] = {
                        "name": c.get("name"),
                        "role": c.get("role", ""),
                        "description": c.get("description", ""),
                    }
            if roster and gi.get("_cast") != roster:
                gi["_cast"] = roster
                changed = True
            if changed:
                gi_path.write_text(json.dumps(gi, indent=2, ensure_ascii=False), encoding="utf-8")
                soak_log("  dss: general_info.writing_style + _cast seeded from spreadsheet")
        except Exception as e:
            soak_log(f"  dss: general_info style seed skipped: {e}")

    def _audit_capability_context(self, beat: dict, turn: int) -> str:
        """Deterministic facts about what DSS *could* have done by this turn, so the
        auditor does not flag 'missing' for beats the engine's gating made impossible.

        Facts computed from the run dir (state_snapshots + user/reply files):
        - which turns each subject's JSON actually changed (update/add_new ran);
        - which turns were scene transitions (characters add_new only fires then);
        - whether the beat's expected entity ever appeared in the story by `turn`.
        """
        facts = []

        def subject_changed_turns(subject: str) -> list[int]:
            changed = []
            prev_blob = None
            for t in range(turn + 1):
                p = self.run_dir / f"turn_{t:03d}" / "state_snapshot" / f"{subject}.json"
                if not p.exists():
                    prev_blob = None
                    continue
                try:
                    blob = p.read_text(encoding="utf-8")
                except Exception:
                    continue
                if prev_blob is not None and blob != prev_blob:
                    changed.append(t)
                prev_blob = blob
            return changed

        subject = beat.get("subject")
        name = (beat.get("expect") or {}).get("name", "")
        hit_turns: list[int] = []
        if name:
            needle = name.lower()
            for t in range(turn + 1):
                found = False
                for fname in ("user.txt", "reply.txt"):
                    p = self.run_dir / f"turn_{t:03d}" / fname
                    if p.exists() and needle in p.read_text(encoding="utf-8", errors="ignore").lower():
                        found = True
                        break
                if found:
                    hit_turns.append(t)

        if subject:
            changed = subject_changed_turns(subject)
            facts.append(
                f"- The '{subject}' subject file changed at turns "
                f"{changed if changed else '(never)'} up to {turn} — only those turns "
                f"could have written {subject}."
            )
            if name:
                facts.append(
                    f"- The expected entity '{name}' appears in the story at turns "
                    f"{hit_turns if hit_turns else '(never)'} up to {turn}."
                )

        # characters add_new is gated behind scene transitions in this engine.
        if subject == "characters":
            trans = []
            prev_no = None
            for t in range(turn + 1):
                p = self.run_dir / f"turn_{t:03d}" / "state_snapshot" / "current_scene.json"
                if not p.exists():
                    continue
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    continue
                no = data.get("_scene_number")
                if prev_no is not None and no != prev_no:
                    trans.append(t)
                prev_no = no
            facts.append(
                f"- Scene transitions happened at turns {trans if trans else '(none)'} up to {turn}; "
                "characters add_new only runs on scene-transition turns, so a NEW character "
                "could only be saved there."
            )
            # explicit determinable conclusion: was there a transition AFTER the entity appeared?
            if name and trans and hit_turns:
                savable = [t for t in trans if t >= min(hit_turns)]
                if savable:
                    facts.append(
                        f"- DSS WAS able to save '{name}': a scene transition occurred at "
                        f"turn(s) {savable}, on or after its first in-story appearance "
                        f"(turn {min(hit_turns)}). The beat is ASSESSABLE — if '{name}' is not "
                        "in the saved subject by the latest such turn, mark it 'missing', "
                        "not 'unassessable'."
                    )

        if not facts:
            return ""
        guide = (
            "DSS CAPABILITY CONTEXT (deterministic, from the run data — use it to judge whether "
            "this beat was even satisfiable):\n"
            + "\n".join(facts)
            + f"\nIf the expected entity never appeared in the story by the evaluation turn {turn}, "
              "or the subject could not have been updated by then (frozen / gating), mark the beat "
              "'unassessable' instead of 'missing'. If the expected entry IS present in the subject "
              "at the evaluation turn, the beat is satisfied — mark 'saved'/'partial', do NOT mark "
              "'unassessable' merely because the save landed after the scheduled turn. Only mark "
              "'missing' when the entity appeared AND the subject was updatable by the evaluation "
              "turn but DSS still failed to save it."
        )
        return guide

    def _run_audit(self, turn: int, state_dir: Path | None) -> list[dict] | None:
        """Audit DSS's saves against spreadsheet dss_beats.

        Runs on beats scheduled at this turn (or every audit_every turns when
        set). The auditor cloud model gets the beat expectation + the actual
        subject JSON files and reports pass/fail per beat.
        """
        if not self.auditor:
            return None
        beats = self.spreadsheet.get("dss_beats", [])
        if not beats:
            return None
        due = []
        if self.args.audit_every:
            if turn > 0 and turn % self.args.audit_every == 0:
                due = [b for b in beats
                       if b.get("turn") is None or int(b["turn"]) <= turn]
        else:
            due = [b for b in beats if b.get("turn") is not None and int(b["turn"]) == turn]
        if self._retry_audit_due(turn):
            have = {b["id"] for b in due}
            due += [b for b in self._unresolved_beats(turn) if b["id"] not in have]
        if not due:
            return None
        state_summary = self._memory_summary(state_dir) if state_dir else "(no state)"
        results = []
        for beat in due:
            capability = self._audit_capability_context(beat, turn)
            pre = beat_precheck(beat, state_dir, list(self.history))
            if pre and pre.get("decisive"):
                verdict = {"id": beat["id"], "status": pre["status"], "abort": False,
                           "detail": pre["detail"], "source": "deterministic",
                           "turn": turn, "wall_ts": datetime.now().isoformat()}
                results.append(verdict)
                soak_log(f"  auditor: beat {beat['id']} -> {verdict['status']} (deterministic)")
                continue
            msgs = build_audit_messages(self.spreadsheet, beat, state_summary, capability, turn=turn,
                                        precheck_facts=(pre or {}).get("facts", ""))
            t0 = time.time()
            soak_log(f"  auditor: beat {beat['id']} vs DSS state ({self.auditor.model}) ...")
            try:
                verdict = self.auditor.json_complete(msgs, max_tokens=1600, json_mode=True)
            except CloudError as e:
                soak_log(f"  auditor: beat {beat['id']} call failed: {e}")
                verdict = {"id": beat["id"], "status": "error", "detail": str(e)}
            verdict.setdefault("id", beat["id"])
            verdict = verify_audit_verdict(verdict, state_summary)
            verdict["turn"] = turn
            verdict["wall_ts"] = datetime.now().isoformat()
            results.append(verdict)
            soak_log(f"  auditor: beat {beat['id']} -> {verdict.get('status')} in {_elapsed(t0)}")
        return results

    # ----------------------------------------------------------- runtime ---

    def _configure_runtime(self, sandbox: Path) -> None:
        from extensions.dayna_ss.runtime import runtime

        current_character = {"name": f"soak_{self.spreadsheet['id']}"}
        runtime.configure(
            model_provider=lambda: self.local,
            stop_provider=lambda: self._stop,
            prompt_builder=lambda prompt, state, **kw: prompt,
            encoder=lambda text, add_bos_token=True: text,
            persistent_ui_state_provider=lambda: self._ui_state,
            current_character_provider=lambda: current_character["name"],
            settings_provider=lambda: {},
            update_config_fn=lambda state: (current_character.__setitem__(
                "name", state.get("character_menu", current_character["name"])) or False),
            register_tool_executors_fn=lambda _: None,
            extension_dir=sandbox,
        )

    def _make_summarizer(self, sandbox: Path):
        from extensions.dayna_ss.agents.summarizer import Summarizer
        self._configure_runtime(sandbox)
        self.summarizer = Summarizer()
        self.summarizer.config["max_subject_workers"] = self.args.max_subject_workers
        # Unit-cadence profile (compressed/campaign) — overlays the schema's
        # chapter/arc gate bounds; None keeps schema numbers untouched.
        self.summarizer.config["cadence_profile"] = getattr(self.args, "cadence_profile", None)
        self.summarizer.config["max_scene_part_messages"] = max(0, self.args.max_scene_messages)
        self.summarizer.config["message_mode"] = self.args.message_mode
        self.summarizer.config["retrieval_placement"] = self.args.retrieval_placement
        self.summarizer.config["rolling_window"] = max(0, self.args.rolling_window)
        self.summarizer.config["rolling_summaries"] = max(0, self.args.rolling_summaries)
        # Per-entry whole-subject context re-statement gating (see §23).
        self.summarizer.config["restate_map_context"] = self.args.restate_map_context
        self.summarizer.config["restate_map_threshold_chars"] = max(0, self.args.restate_map_threshold)
        self.summarizer.config["embed_full_schema"] = self.args.embed_full_schema
        # The selected schema is copied into the sandbox at the default location
        # (see main()'s sandbox setup), so the engine resolves
        # `extension_dir / user_data/example/subjects_schema.json` to the variant.
        self.local.config = self.summarizer.config
        return self.summarizer

    # ------------------------------------------------------ state summary ---

    @staticmethod
    def _retrieval_payload(custom_state: dict) -> list[list[str]]:
        """The memory the engine actually inserted: artificial-history [prompt, text] pairs + context."""
        pairs: list[list[str]] = []
        if not isinstance(custom_state, dict):
            return pairs
        for item in ((custom_state.get("history") or {}).get("internal") or []):
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                pairs.append([str(item[0]), str(item[1])])
        if custom_state.get("context"):
            pairs.append(["__context__", str(custom_state["context"])])
        return pairs

    def _due_note_ids(self, turn: int, active_notes: list[dict]) -> set[str]:
        ids = {n["id"] for n in (active_notes or []) if isinstance(n, dict) and n.get("id")}
        for n in self.spreadsheet.get("notes", []):
            if (n.get("recall") or {}).get("due") == turn:
                ids.add(n["id"])
        return ids

    def _section_types(self) -> dict:
        cached = getattr(self, "_section_types_cache", None)
        if cached is not None:
            return cached
        found: dict = {}
        try:
            from extensions.dayna_ss.runtime import runtime
            for rel in ("user_data/example/format_templates.json", "user_data/format_templates.json"):
                p = Path(runtime.extension_dir) / rel
                if p.exists():
                    found = section_types_from_templates(p)
                    break
        except Exception:
            found = {}
        self._section_types_cache = found
        return found

    def _planted_now(self, user_input: str, reply: str) -> list[str]:
        blob = f"{user_input}\n{reply}".lower()
        return [n["id"] for n in self.spreadsheet.get("notes", [])
                if (n.get("needle") or "").strip() and n["needle"].lower() in blob]

    def _guide_memory_view(self, state_dir: Path | None) -> str:
        """What the cloud guide/planner may see of DSS's memory (--guide-memory full|names|off)."""
        mode = getattr(self.args, "guide_memory", "full")
        if mode == "off":
            return "(DSS's private memory is not shown to the guide in this run)"
        if mode == "names":
            return render_memory_names(state_dir)
        return self._memory_summary(state_dir)

    def _memory_summary(self, state_dir: Path | None) -> str:
        """Bounded DSS-state inventory for guide/judge/auditor/overview (see render_memory_summary)."""
        return render_memory_summary(state_dir)

    # ------------------------------------------------------------ needles ---

    def _current_state_dir(self) -> Path | None:
        """The engine's latest persisted state dir for the full history.

        ``retrieve_history_path`` hashes the full history, so each turn creates a
        new directory — but the per-subject state files are written by
        ``summarize_latest_state`` into the dir for the *completed* exchange and
        (during the first scene) into the fresh-chat dir. We therefore scan the
        run's history root for the newest directory that actually holds subject
        JSONs, falling back to the engine's own ``last.history_path``.
        """
        if self.args.level == 1:
            d = self.run_dir / "state_l1"
            return d if d.exists() else None
        root = self._history_root()
        if root and root.exists():
            subject_files = {"characters.json", "current_scene.json", "events.json",
                             "general_info.json", "groups.json"}
            candidates = []
            for d in root.iterdir():
                if not d.is_dir():
                    continue
                has_subject = any((d / f).exists() for f in subject_files)
                if has_subject:
                    candidates.append(d)
            if candidates:
                return max(candidates, key=lambda d: d.stat().st_mtime)
        try:
            if self.summarizer and self.summarizer.last and self.summarizer.last.history_path:
                p = Path(str(self.summarizer.last.history_path))
                if p.exists():
                    return p
        except Exception:
            pass
        return None

    def _history_root(self) -> Path | None:
        """The engine's per-run history dir (parent of the hashed per-turn dirs)."""
        try:
            hp = self.summarizer.retrieve_history_path(self._engine_state(), [])
            return Path(str(hp)).parent if hp else None
        except Exception:
            return None

    def _engine_state(self) -> dict:
        """Return a state dict shaped for the engine (with unique_id etc)."""
        return {
            "unique_id": f"soak_{self.args.seed}",
            "character_menu": f"soak_{self.spreadsheet['id']}",
            "seed": self.args.seed,
            "name1": self.spreadsheet["characters"]["name1"]["name"],
            "name2": self.spreadsheet["characters"]["name2"]["name"],
            "context": build_state(self.spreadsheet, self.args.seed, "").get("context", ""),
        }

    # ------------------------------------------------------------- turn -----

    def _latest_history_path(self) -> Path | None:
        if self.summarizer is None:
            return None
        try:
            return Path(str(self.summarizer.retrieve_history_path(self._engine_state(), self.history)))
        except Exception:
            return None

    # ------------------------------------------------------------- plan -----

    def _maybe_replan(self, turn: int, state_dir: Path | None) -> dict | None:
        """Refresh the guide's live short-term plan on plan-every cadence.

        Returns the fresh plan (and persists it), or None if unchanged/disabled.
        """
        plan_every = self.args.plan_every
        if not plan_every or turn % plan_every != 0:
            return None
        memory = self._guide_memory_view(state_dir)
        msgs = build_plan_messages(self.spreadsheet, self.plan, self.history, memory)
        t0 = time.time()
        soak_log(f"  replan: calling cloud guide ({self.guide.model}) json_mode ...", ts=True)
        try:
            raw = self.guide.complete(msgs, max_tokens=4000, json_mode=True,
                                      reasoning_effort="low")
        except CloudError as e:
            # Replan is optional steering — an endpoint failure here keeps the
            # previous plan and the run continues on the next turn.
            soak_log(f"  replan: cloud call FAILED ({e}); keeping the existing plan")
            return None
        soak_log(f"  replan: cloud response in {_elapsed(t0)} (guide calls={self.guide.calls})")
        plan = parse_plan(raw, self.plan)
        plan["created_turn"] = turn
        plan["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.plan = plan
        plan_path = self.run_dir / "plan.json"
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        soak_log(f"  replan: saved plan with {len(plan.get('beats', []))} beats")
        return plan

    def _plan_view(self) -> str:
        """Compact one-line-per-beat plan text for the steering prompt."""
        if not self.plan:
            return ""
        lines = []
        if self.plan.get("intent"):
            lines.append(f"Near-future intent: {self.plan['intent']}")
        for b in self.plan.get("beats", [])[:6]:
            note = f" [notes: {', '.join(b['note_ids'])}]" if b.get("note_ids") else ""
            lines.append(f"- +{b.get('turn', 0)}: {b.get('action', '')}{note}")
        if self.plan.get("style_reminder"):
            lines.append(f"Style reminder: {self.plan['style_reminder']}")
        return "\n".join(lines)

    def run_turn(self, turn: int) -> dict:
        from extensions.dayna_ss.agents.summarizer import Summarizer  # noqa: F401

        args = self.args
        spread = self.spreadsheet

        if args.level == 1:
            if args.synthetic_chat:
                raise RuntimeError("--synthetic-chat is a Level-2 isolation mode; it is not supported at Level 1")
            return self._run_turn_level1(turn)
        
        soak_log(1)

        # 1. guide writes the user turn
        active_notes = notes_visible_at(spread["notes"], turn, args.guide_style)
        state_dir = self._current_state_dir()
        _cloud_t0 = time.time()
        replanned = self._maybe_replan(turn, state_dir)
        memory = self._guide_memory_view(state_dir)
        soak_log(1.5)
        guide_msgs = build_guide_messages(spread, active_notes, self.history, args.guide_style,
                                          memory, self._plan_view(),
                                          greeting=spread.get("greeting", ""))
        user_input = guide_turn_with_retry(
            self.guide, spread, guide_msgs,
            max_attempts=self.args.guide_max_attempts,
            cooldown=self.args.cloud_empty_cooldown)
        cloud_s = time.time() - _cloud_t0
        soak_log(1.75)
        if args.arc_breaks == "3_arcs" and turn in (25, 50, 75):
            # Stage via persistent_ui_state — the same channel production's
            # NEXT ARC: prefix feeds. Prepending the literal prefix here does
            # nothing: the harness never routes through TGWUI's
            # chat_input_modifier, and the engine only parsed prefixes there.
            from extensions.dayna_ss.runtime import runtime
            runtime.persistent_ui_state["force_next_arc"] = True
        soak_log(1.875)
        if getattr(args, "force_chapter_turn", 0) and turn == args.force_chapter_turn:
            from extensions.dayna_ss.runtime import runtime
            runtime.persistent_ui_state["force_next_chapter"] = True
        soak_log(1.9875)

        soak_log(2)
        # 2. engine reply prompt
        # The raw state handed to the engine carries the full production-shaped
        # internal history (greeting pair at [0] + every exchange so far) — a pure
        # session emulator. The engine reads it for the greeting ([0][1]) and for
        # DSS's own recent replies (generate_instr_prompt's R1 negative-exemplar
        # block); custom_state's internal (built from `history_internal`) is the
        # separate artificial retrieval context for the model payloads.
        engine_state = build_state(spread, args.seed, f"soak_{args.seed}", history=self.history)
        greeting = engine_state.get("greeting", "").strip()
        # DEBUG: hash trace for resume debugging
        import hashlib as _hl
        _hi = engine_state["history"]["internal"]
        _hh = _hl.sha1(str(_hi).encode("utf-8")).hexdigest()[:24]
        _hp = self._history_root() / _hh if self._history_root() else None
        soak_log(f"  [hash-debug] turn={turn} hist_len={len(_hi)} greeting={greeting[:60]!r} hash={_hh} path_exists={_hp.exists() if _hp else 'no-root'}")
        # Production view: the conversation INCLUDES the character greeting pair at
        # index 0 (assistant side), so message indices / rolling pairs / format_dialogue
        # all match production. Cloud roles (guide/judge/synthetic) keep seeing only the
        # [user, reply] exchanges — the greeting is name2's own line, not their concern.
        history_internal = ([["<|BEGIN-VISIBLE-CHAT|>", greeting]] if greeting else []) + [
            list(e) for e in self.history
        ]
        t0 = time.time()
        soak_log("  engine: building instruction prompt (retrieval + cache warm) ...")
        _instr_kwargs = {"do_instr": args.director == "on"} if args.director is not None else {}
        instr_prompt, custom_state, history_path, ts = self.summarizer.generate_instr_prompt(
            user_input, engine_state, history_internal, **_instr_kwargs
        )
        soak_log(f"  engine: instr prompt built in {_elapsed(t0)}")
        # Retrieval precision: how accurately DSS's memory was inserted into this turn's context.
        # Computed from the state the retriever saw and the story BEFORE this exchange (history is
        # appended after the reply). Never allowed to break a turn.
        try:
            retrieval_payload = self._retrieval_payload(custom_state)
            retrieval = retrieval_metrics(
                retrieval_payload, state_dir, list(self.history), user_input,
                notes=self.spreadsheet.get("notes", []),
                due_note_ids=self._due_note_ids(turn, active_notes),
                section_types=self._section_types())
        except Exception as e:
            soak_log(f"  retrieval metrics failed: {e}")
            retrieval_payload, retrieval = [], {"available": False, "reason": f"error: {e}"}
        self._seed_general_info_style(history_path)

        # contamination guard (§6/§7): raw note text must never reach DSS prompts
        instr_text = str(instr_prompt)
        for note in spread["notes"]:
            content = note.get("content", "")
            if content and content.strip() in instr_text:
                raise RuntimeError(
                    f"CONTAMINATION: note '{note['id']}' content leaked into instr_prompt at turn {turn}"
                )
            if content and content.strip() in user_input:
                raise RuntimeError(
                    f"CONTAMINATION: note '{note['id']}' content leaked into user_input at turn {turn}"
                )

        soak_log(3)
        # 3. reply: local model, OR (synthetic-chat isolation) the cloud model
        t0 = time.time()
        if args.synthetic_chat:
            soak_log(f"  synthetic: cloud ({self.guide.model}) writes {spread['characters']['name2']['name']}'s reply ...")
            synth_msgs = build_synthetic_reply_messages(spread, self.history, user_input)
            reply = guide_turn_with_retry(self.guide, spread, synth_msgs, label="synthetic reply",
                                  max_attempts=self.args.guide_max_attempts,
                                  cooldown=self.args.cloud_empty_cooldown).strip()
            soak_log(f"  synthetic: cloud reply in {_elapsed(t0)} -> {reply[:80]!r}...")
        else:
            soak_log(f"  local: generating reply ({self.local.model}) ...")
            reply = self._generate_reply(instr_text, custom_state)
            soak_log(f"  local: reply in {_elapsed(t0)} -> {reply[:80]!r}...")

        self._last_reply = reply

        # 4. engine summarization (real memory update)
        self.history.append([user_input, reply])
        # Production parity: by summarize time the reply is in TGWUI state, so the
        # raw state's internal history now carries the just-finished exchange too.
        engine_state["history"]["internal"] = ([["<|BEGIN-VISIBLE-CHAT|>", greeting]] if greeting else []) + [
            list(e) for e in self.history
        ]
        t0 = time.time()
        soak_log("  dss: summarize_latest_state (DataSummarizer + message chunks) ...")
        self.summarizer.summarize_latest_state(
            reply, user_input, engine_state,
            ([["<|BEGIN-VISIBLE-CHAT|>", greeting]] if greeting else []) + self.history,
        )
        soak_log(f"  dss: summarization done in {_elapsed(t0)}")

        # 5. deterministic needle probes
        cur = self._current_state_dir()
        probes = probe_needles(spread["notes"], cur) if cur else {}

        # 6. echo/recall probes on due turns
        recalls = {}
        for note in spread["notes"]:
            due = (note.get("recall") or {}).get("due")
            if due is not None and int(due) == turn:
                recalls[note["id"]] = recall_probe(note, reply, cur) if cur else {
                    "id": note["id"], "matched_syns": [], "echoed": False, "echo_by_text_only": False}

        # 7. judge pass (background worker — merged into result.json when it lands)
        if turn > 0 and args.judge_every and turn % args.judge_every == 0:
            self._pending_judge.append(self._make_judge_task(turn, active_notes, reply, cur))

        # 8. DSS-save audit (dss_beats, background worker)
        if args.audit_every or self._beats_due(turn):
            task = self._make_audit_task(turn, cur)
            if task is not None:
                self._pending_audit.append(task)

        return {
            "turn": turn,
            "user_input": user_input,
            "reply": reply,
            "instr_prompt": instr_text,
            "context": custom_state.get("context", ""),
            "retrieval": retrieval,
            "retrieval_payload": retrieval_payload,
            "planted_now": self._planted_now(user_input, reply),
            "history_path": str(history_path) if history_path else None,
            "probes": probes,
            "recalls": recalls,
            "judge": None,
            "audit": None,
            "plan": self.plan,
            "ts": ts,
            "local_calls": self.local.call_count,
            "cloud_usage": self.guide.stats(),
            "cloud_s": round(cloud_s, 1),
            "synthetic_chat": bool(getattr(args, "synthetic_chat", False)),
        }

    def _run_turn_level1(self, turn: int) -> dict:
        """Fast smoke path (plan §Q1): per-subject ``DataSummarizer.generate``.

        No retrieval / chunking / scene machinery — validates spreadsheet + config
        quickly before committing to Level 2.
        """
        from extensions.dayna_ss.agents.data_summarizer import DataSummarizer
        from extensions.dayna_ss.ui.phase_manager import PhaseManager
        from extensions.dayna_ss.utils.schema_parser import SchemaParser

        spread = self.spreadsheet
        state_dir = self._current_state_dir()
        schema_path = self.run_dir / "sandbox" / "user_data" / "example" / "subjects_schema.json"
        schema_parser = SchemaParser(schema_path, include_descriptions=not self.args.no_schema_descriptions)

        # guide writes the turn
        active_notes = notes_visible_at(spread["notes"], turn, self.args.guide_style)
        memory = self._guide_memory_view(state_dir)
        replanned = self._maybe_replan(turn, state_dir)
        guide_msgs = build_guide_messages(spread, active_notes, self.history, self.args.guide_style,
                                          memory, self._plan_view(),
                                          greeting=spread.get("greeting", ""))
        user_input = guide_turn_with_retry(
            self.guide, spread, guide_msgs,
            max_attempts=self.args.guide_max_attempts,
            cooldown=self.args.cloud_empty_cooldown)

        # simple reply prompt (no retrieval in smoke)
        name1, name2 = spread["characters"]["name1"]["name"], spread["characters"]["name2"]["name"]
        dss_voice = spread["writing_style"]["dss_directive"]
        greeting = spread.get("greeting", "").strip()
        base = ([["<|BEGIN-VISIBLE-CHAT|>", greeting]] if greeting else []) + self.history
        recent = "\n".join(f"{name1}: {u}\n{name2}: {r}" for u, r in base[-4:])
        prompt = (f"{recent}\n\n{user_input}\n\n"
                  f"Write {name2}'s next reply, in character, following {name2}'s voice:\n"
                  f"{dss_voice}\n"
                  f"Draw naturally on the characters, items, and details already established in the conversation above; "
                  f"reference them concretely where they fit rather than leaving remembered details unused.")
        t0 = time.time()
        soak_log(f"  local: generating reply ({self.local.model}) ...")
        reply = self._generate_reply(prompt, {"name1": name1, "name2": name2})
        soak_log(f"  local: reply in {_elapsed(t0)} -> {reply[:80]!r}...")
        self._last_reply = reply

        # per-subject summarization
        all_subjects_data = {}
        if state_dir and state_dir.exists():
            for f in state_dir.glob("*.json"):
                if f.name in ("subjects_schema.json", "format_templates.json", "entity_graph.json"):
                    continue
                try:
                    all_subjects_data[f.stem] = json.loads(f.read_text(encoding="utf-8"))
                except Exception:
                    continue
        if not all_subjects_data:
            for subject in schema_parser.subjects:
                all_subjects_data[subject] = {}

        state_out = self.run_dir / "state_l1"
        state_out.mkdir(parents=True, exist_ok=True)
        # Pure session emulator: the internal history carries the greeting pair +
        # every completed exchange (production TGWUI shape), so DSS's own recent
        # replies reach the engine's R1 negative-exemplar block and the
        # DataSummarizer's message-derived values see the real story.
        custom_state = {
            "history": {"internal": ([["<|BEGIN-VISIBLE-CHAT|>", greeting]] if greeting else []) + [list(e) for e in self.history]},
            "name1": name1, "name2": name2,
        }
        pm = PhaseManager()
        self.local.last = SimpleNamespace(
            schema_parser=schema_parser,
            state=custom_state,
            history_path=state_out,
            is_new_scene_turn=False,
            force_next_chapter=False,
            force_next_arc=False,
        )
        # real_history mirrors what summarize_latest_state would receive: greeting
        # + completed exchanges + the exchange being summarized.
        real_history = ([["<|BEGIN-VISIBLE-CHAT|>", greeting]] if greeting else []) + [list(e) for e in self.history] + [[user_input, reply]]
        ds = DataSummarizer(
            summarizer=self.local,
            exchange=(user_input, reply),
            custom_state=custom_state,
            history_path=state_out,
            schema_parser=schema_parser,
            all_subjects_data=all_subjects_data,
            phase_manager=pm,
            real_history=real_history,
        )
        t0 = time.time()
        soak_log("  dss: per-subject DataSummarizer.generate ...")
        for subject in schema_parser.subjects:
            if subject not in all_subjects_data:
                continue
            schema_class = schema_parser.get_subject_class(subject)
            all_subjects_data[subject] = ds.generate(subject, all_subjects_data[subject], schema_class)
        soak_log(f"  dss: DataSummarizer done in {_elapsed(t0)}")

        self.history.append([user_input, reply])
        probes = probe_needles(spread["notes"], state_out)

        # judge + audit run on background workers (merged into result.json when they land)
        if turn > 0 and self.args.judge_every and turn % self.args.judge_every == 0:
            self._pending_judge.append(self._make_judge_task(turn, active_notes, reply, state_out))
        if self.args.audit_every or self._beats_due(turn):
            task = self._make_audit_task(turn, state_out)
            if task is not None:
                self._pending_audit.append(task)

        return {
            "turn": turn,
            "user_input": user_input,
            "reply": reply,
            "instr_prompt": prompt,
            "context": "",
            "history_path": str(state_out),
            "probes": probes,
            "recalls": {},
            "judge": None,
            "audit": None,
            "plan": self.plan,
            "ts": None,
            "local_calls": self.local.call_count,
            "cloud_usage": self.guide.stats(),
        }

    def _planted_note_ids(self) -> set[str]:
        """Note ids whose needle actually appeared in the story text (guide or DSS turn).

        Style-hold notes (empty needle) are always in scope. Notes the guide never
        planted cannot be charged against DSS — they are excluded from the in-scope
        set so a 'missed' verdict can only mean a genuine DSS failure.
        """
        story_text = "\n".join(u + "\n" + r for u, r in self.history).lower()
        planted: set[str] = set()
        for n in self.spreadsheet["notes"]:
            needle = (n.get("needle") or "").strip()
            if not needle or needle.lower() in story_text:
                planted.add(n["id"])
        return planted

    def _run_judge(self, turn: int, active_notes: list[dict], reply: str, state_dir: Path | None,
                   window: list[list[str]] | None = None, planted: set[str] | None = None,
                   full_history: list[list[str]] | None = None) -> dict | None:
        spread = self.spreadsheet
        if window is None:
            window = self.history[-self.args.judge_window:]
        all_notes = active_notes or spread["notes"]
        if planted is None:
            planted = self._planted_note_ids()
        notes_in_scope = [n for n in all_notes if n["id"] in planted]
        unplanted_ids = [n["id"] for n in all_notes if n["id"] not in planted]
        state_summary = self._memory_summary(state_dir)
        instr_text = ""
        instr_path = self.run_dir / f"turn_{turn:03d}" / "instr_prompt.txt"
        if instr_path.exists():
            full = instr_path.read_text(encoding="utf-8", errors="ignore")
            m = re.search(r"INSTRUCTIONS TO FOLLOW:\s*\"\"\"\s*(.*?)\s*\"\"\"", full, re.S)
            if m:
                instr_text = m.group(1)
        # Split planted notes into in-beat (needle in instruction) vs. past (all others)
        instr_lower = instr_text.lower()
        in_beat_notes = []
        past_notes = []
        for n in notes_in_scope:
            needle = (n.get("needle") or n.get("content") or "").strip().lower()
            if needle and needle in instr_lower:
                in_beat_notes.append(n)
            else:
                past_notes.append(n)
        grace = int(getattr(self.args, "memory_grace", MEMORY_GRACE_DEFAULT))
        memory_key = memory_answer_key(spread, full_history if full_history is not None else self.history,
                                       grace=grace, in_beat_ids={n["id"] for n in in_beat_notes})
        msgs = build_judge_messages(spread, in_beat_notes, past_notes, window, state_summary, reply,
                                    unplanted_ids=unplanted_ids, instructions=instr_text,
                                    synthetic=bool(getattr(self.args, "synthetic_chat", False)),
                                    memory_key=memory_key)
        try:
            result = self.judge.json_complete(msgs, max_tokens=8000, json_mode=True)
        except CloudError as e:
            print(f"[soak] judge call failed at turn {turn}: {e}")
            return {"error": str(e)}
        if result and "error" not in result:
            result["turn"] = turn
            window_text = "\n".join(f"{u}\n{r}" for u, r in window) + "\n" + reply
            notes_by_id = {n["id"]: n for n in spread.get("notes", [])}
            result.pop("memory_fidelity", None)  # the harness computes it; never trust a judge-written number
            score, breakdown = score_memory_checks(
                result.get("memory_checks"), memory_key, state_summary, window_text,
                wrong_character=bool(result.get("wrong_character")), notes_by_id=notes_by_id)
            if score is None:
                return {"error": "judge memory_checks unusable: " + str(breakdown.get("error", "unknown"))}
            result["memory_fidelity"] = score
            result["memory_breakdown"] = breakdown
        return result

    def _run_final_overview(self, state_dir: Path | None) -> dict | None:
        """End-of-run judge pass: final editorial overview of the whole chat."""
        return self._run_overview(state_dir, scope="final")

    def _run_overview(self, state_dir: Path | None, scope: str = "final",
                      history: list[list[str]] | None = None,
                      planted: set[str] | None = None,
                      judge_records: list[dict] | None = None) -> dict | None:
        """Shared judge pass: final or rolling overview of {name2}'s performance.

        ``history``/``planted``/``judge_records`` snapshot the run data as of the
        dispatch turn when called from a background worker (the live lists keep
        growing on the main thread); final-overview calls omit them to read the
        full post-run state.
        """
        state_summary = self._memory_summary(state_dir)
        if history is None:
            history = self.history
        if planted is None:
            planted = self._planted_note_ids()
        if judge_records is None:
            judge_records = self.judge_records
        all_notes = self.spreadsheet["notes"]
        in_scope_notes = [n for n in all_notes if n["id"] in planted]
        unplanted_ids = [n["id"] for n in all_notes if n["id"] not in planted]
        grace = int(getattr(self.args, "memory_grace", MEMORY_GRACE_DEFAULT))
        memory_key, ledger_text = overview_memory_inputs(self.spreadsheet, history, judge_records, grace)
        msgs = build_final_overview_messages(
            self.spreadsheet, in_scope_notes, history, judge_records,
            state_summary, scope=scope, unplanted_ids=unplanted_ids,
            memory_key=memory_key, ledger_text=ledger_text,
        )
        try:
            result = self.judge.json_complete(
                msgs, max_tokens=16000, json_mode=True, reasoning_effort="low",
            )
        except CloudError as e:
            print(f"[soak] {scope} overview call failed: {e}")
            return {"error": str(e)}
        return finalize_overview_memory(result, self.spreadsheet, history, state_summary, memory_key)


def _clear_judge_audit_overview(run_dir: Path, from_turn: int) -> None:
    """Clear judge/audit/overview data from turn ``from_turn`` onward.

    Removes ``judge``, ``judge_usage``, ``audit``, ``auditor_usage`` keys from
    each ``turn_NNN/result.json`` (turn N >= from_turn), and deletes
    ``rolling_overview_NNN.json`` and ``final_overview.json``. The result.json
    files themselves are preserved (reply/user/instr_prompt/local_usage etc.
    remain intact), making way for a subsequent ``--backfill`` with a new model.
    """
    run_dir = Path(run_dir)
    cleared = 0
    for d in sorted(run_dir.glob("turn_*")):
        if not d.is_dir():
            continue
        try:
            t = int(d.name.split("_")[-1])
        except (ValueError, IndexError):
            continue
        if t < from_turn:
            continue
        result_path = d / "result.json"
        if result_path.exists():
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
                removed = [k for k in ("judge", "judge_usage", "audit", "auditor_usage") if k in result]
                if removed:
                    for k in removed:
                        del result[k]
                    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
                    cleared += 1
            except Exception:
                pass
    # Rolling overviews: rolling_overview_NNN.json where NNN-1 >= from_turn
    for ov in sorted(run_dir.glob("rolling_overview_*.json")):
        try:
            n = int(ov.stem.split("_")[-1])
        except (ValueError, IndexError):
            continue
        if (n - 1) >= from_turn:
            ov.unlink()
    # Final overview is derived from all judge records; stale after a partial clear.
    final_ov = run_dir / "final_overview.json"
    if final_ov.exists():
        final_ov.unlink()
    print(f"[soak] cleared judge/audit/overview from turn {from_turn} onward "
          f"({cleared} result.json files updated)")


def _backfill_overview(run_dir: Path, scope: str = "final", checkpoint: int | None = None,
                       model: str | None = None, endpoint: str | None = None,
                       provider: str | None = None) -> int:
    """Standalone: (re-)write a final or rolling overview for an existing run dir.

    ``scope="final"`` writes ``final_overview.json`` over ALL turns;
    ``scope="rolling"`` with ``checkpoint=N`` writes ``rolling_overview_NNN.json``
    reviewing the first ``N`` exchanges only (missing/errored rolling checkpoints
    can be recovered this way).
    """
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"[soak] ERROR: no manifest at {run_dir}")
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    m_args = manifest.get("args", {})
    raw_sheet = m_args.get("spreadsheet") or ""
    sheet_path = Path(raw_sheet)
    if not sheet_path.exists():
        sheet_name = str(sheet_path.name if raw_sheet else (manifest.get("spreadsheet") or "")).rstrip(".json")
        sheet_path = TEST_DIR / "spreadsheets" / f"{sheet_name}.json"
    if not sheet_path.exists():
        print(f"[soak] ERROR: cannot locate spreadsheet for {run_dir} (tried {sheet_path})")
        return 1
    spreadsheet = load_spreadsheet(sheet_path)

    history: list[list[str]] = []
    judge_records: list[dict] = []
    last_state_dir: Path | None = None
    for d in sorted(run_dir.glob("turn_*")):
        turn_no = int(d.name.split("_")[1])
        if checkpoint is not None and turn_no >= checkpoint:
            continue
        if (d / "user.txt").exists() and (d / "reply.txt").exists():
            history.append([
                (d / "user.txt").read_text(encoding="utf-8"),
                (d / "reply.txt").read_text(encoding="utf-8"),
            ])
        rp = d / "result.json"
        if rp.exists():
            try:
                r = json.loads(rp.read_text(encoding="utf-8"))
                j = r.get("judge")
                if j and "error" not in j:
                    j["turn"] = r.get("turn")
                    judge_records.append(j)
            except Exception:
                pass
        snap = d / "state_snapshot"
        if snap.exists():
            last_state_dir = snap

    live_dir = run_dir / "live"
    try:
        live_dir.mkdir(exist_ok=True)
    except OSError:
        live_dir = None
    m_args = manifest.get("args", {})
    judge_endpoint = endpoint or m_args.get("cloud_endpoint") or None
    judge_provider = provider or m_args.get("cloud_provider") or None
    judge_model = model or m_args.get("judge_model") or "deepseek-v4-flash"
    judge = CloudModel(endpoint=judge_endpoint, model=judge_model,
                       max_tokens=8000, timeout=300,
                       reasoning_effort="low", fallback_reasoning=False,
                       request_interval=m_args.get("request_interval", 5.0),
                       status_dir=str(live_dir) if live_dir else None, role="judge",
                       provider=judge_provider)
    if not judge.api_key:
        print("[soak] ERROR: no cloud API key (set OPENCODE_GO_API_KEY)")
        return 1
    state_summary = render_memory_summary(last_state_dir)
    story_text = "\n".join(u + "\n" + r for u, r in history).lower()
    planted = {n["id"] for n in spreadsheet["notes"]
               if not (n.get("needle") or "").strip() or (n.get("needle") or "").strip().lower() in story_text}
    in_scope = [n for n in spreadsheet["notes"] if n["id"] in planted]
    unplanted_ids = [n["id"] for n in spreadsheet["notes"] if n["id"] not in planted]
    grace = int(m_args.get("memory_grace", MEMORY_GRACE_DEFAULT))
    memory_key, ledger_text = overview_memory_inputs(spreadsheet, history, judge_records, grace)
    msgs = build_final_overview_messages(spreadsheet, in_scope, history, judge_records,
                                         state_summary, unplanted_ids=unplanted_ids, scope=scope,
                                         memory_key=memory_key, ledger_text=ledger_text)
    try:
        result = judge.json_complete(
            msgs, max_tokens=16000, json_mode=True, reasoning_effort="low",
        )
    except CloudError as e:
        print(f"[soak] {scope} overview call failed: {e}")
        return 1
    if not result:
        result = {"error": "empty response"}
    result = finalize_overview_memory(result, spreadsheet, history, state_summary, memory_key)
    result["turns"] = len(history)
    result["judge_records"] = judge_records
    result["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    if scope == "rolling":
        out_path = run_dir / f"rolling_overview_{checkpoint:03d}.json"
    else:
        out_path = run_dir / "final_overview.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[soak] {scope} overview written to {out_path}")
    print(json.dumps({k: result[k] for k in result if k != "judge_records"}, indent=2)[:1500])
    return 0


def main() -> int:
    args = _arg_parser().parse_args()

    if args.final_overview:
        return _backfill_overview(args.final_overview, scope="final",
                                  model=args.judge_model,
                                  endpoint=args.cloud_endpoint,
                                  provider=args.cloud_provider)

    if args.backfill_rolling_overview:
        code = 0
        for run_dir, n in args.backfill_rolling_overview:
            code = max(code, _backfill_overview(run_dir, scope="rolling", checkpoint=int(n),
                                                model=args.judge_model,
                                                endpoint=args.cloud_endpoint,
                                                provider=args.cloud_provider))
        return code

    _apply_embed_device(args.embed_device)
    if args.level == 2:
        _warm_imports()

    spreadsheet = _resolve_spreadsheet(args.spreadsheet)
    if args.specificity_profile and spreadsheet.get("specificity_profile") != args.specificity_profile:
        print(f"[soak] note: spreadsheet profile is {spreadsheet.get('specificity_profile')!r}, "
              f"flag says {args.specificity_profile!r}")

    run_id = make_run_id(spreadsheet["id"], args)
    if args.resume:
        run_dir = Path(args.resume)
        runs_root = run_dir.parent
    else:
        runs_root = Path(args.runs_dir) if args.runs_dir else TEST_DIR / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)
        run_dir = runs_root / run_id

    # dedicated soak logfile
    if args.log_file:
        log_path = Path(args.log_file)
    else:
        log_path = runs_root / "logs" / f"{run_id}.log"
    _open_logfile(log_path)
    soak_log(f"=== long-horizon soak start: spreadsheet={spreadsheet['id']} run_dir={run_dir} ===")

    # manifest
    manifest_path = run_dir / "manifest.json"
    if run_dir.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        resume_from = manifest.get("last_turn", -1) + 1
        # A resumed run inherits its predecessor's recorded args verbatim, so a
        # retargeted launch (glm-5.3-flash -> dots-3-note-preview:free) kept
        # writing the OLD model ids to the manifest while every cloud call used
        # the new ones. Overwrite args with the LIVE invocation.
        manifest["args"] = vars(args)
        manifest["resumed_from_run_id"] = manifest.get("config_hash")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        resume_from = 0
        manifest = {
            "spreadsheet": spreadsheet["id"],
            "genre": spreadsheet.get("genre"),
            "args": vars(args),
            "config_hash": run_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "last_turn": -1,
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    total = args.smoke if args.smoke else args.turns
    print(f"[soak] spreadsheet={spreadsheet['id']} turns={total} resume_from={resume_from} "
          f"run_dir={run_dir} level={args.level}")

    # Default the JSONL sidecar to <run_dir>/dss_calls.jsonl unless explicitly
    # overridden (empty string disables).
    if "SOAK_JSONL_PATH" not in os.environ:
        global _SOAK_JSONL_PATH
        _SOAK_JSONL_PATH = str(run_dir / "dss_calls.jsonl")

    # sandbox for engine extension_dir isolation
    sandbox = run_dir / "sandbox"
    sandbox.mkdir(parents=True, exist_ok=True)
    (sandbox / "user_data" / "example").mkdir(parents=True, exist_ok=True)
    ext = REPO_ROOT / "extensions" / "dayna_ss"
    for f in ("subjects_schema.json", "format_templates.json"):
        src = ext / "user_data" / "example" / f
        if src.exists():
            shutil.copy2(src, sandbox / "user_data" / "example" / f)
    # custom schema (--schema-type / --schema): copy the resolved schema over the
    # sandbox default so the engine's `extension_dir / user_data/example/subjects_schema.json`
    # resolves to the selected variant instead of the base schema.
    schema_rel = resolve_subjects_schema(args)
    if schema_rel:
        schema_src = Path(schema_rel) if Path(schema_rel).is_absolute() else ext / schema_rel
        if schema_src.exists():
            shutil.copy2(schema_src, sandbox / "user_data" / "example" / "subjects_schema.json")
    # Refresh schema/template copies already inside session history dirs. The
    # engine parses `history_path/subjects_schema.json` per turn (summarizer.py
    # "Summarizer.schema_parser loaded"), and on resume those per-session copies
    # are stale from the original launch — schema edits would never reach the
    # running DataSummarizer. Overwrite every existing session copy (and the
    # initial world cache) so resumed runs pick up schema changes. The per-turn
    # copy-forward then propagates the fresh schema to all future dirs.
    fresh_schema = sandbox / "user_data" / "example" / "subjects_schema.json"
    fresh_fmt = sandbox / "user_data" / "example" / "format_templates.json"
    for existing in sorted((sandbox / "user_data" / "history").rglob("subjects_schema.json")):
        shutil.copy2(fresh_schema, existing)
    for existing in sorted((sandbox / "user_data" / "history").rglob("format_templates.json")):
        if fresh_fmt.exists():
            shutil.copy2(fresh_fmt, existing)
    for f in ("dss_config.json",):
        src = ext / f
        if src.exists() and not (sandbox / f).exists():
            shutil.copy2(src, sandbox / f)

    # Local model
    base_url = args.local_base
    api_key = resolve_local_key()
    model = (args.local_model or os.environ.get("DSS_BENCH_MODEL", "")).strip()
    if not model:
        # Resolve the id from the live server (the source of truth) so usage
        # records/reports name the actual model instead of a stale default.
        try:
            req = urllib.request.Request(
                base_url + "/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                ids = [m.get("id") for m in json.loads(resp.read().decode()).get("data", [])]
            if ids:
                model = ids[0]
                soak_log(f"resolved local model id from server: {model}")
        except Exception as e:
            soak_log(f"could not resolve local model id from server: {e}")
    if not model:
        model = "Huihui-Qwen3.5-9B-abliterated-AWQ-4bit"
    local = LocalModel(
        base_url, api_key, model,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
        max_update_history_exchanges=args.max_update_history,
    )

    # Determine cloud endpoint: explicit --cloud-endpoint wins over DEFAULT_ENDPOINT
    cloud_endpoint = args.cloud_endpoint or DEFAULT_ENDPOINT

    # Live progress directory for streaming status file
    live_dir = run_dir / "live"
    live_dir.mkdir(exist_ok=True)
    for stale in Path(live_dir).glob("cloud_*.json"):
        try:
            stale.unlink()
        except OSError:
            pass

    guide = CloudModel(endpoint=cloud_endpoint, model=args.guide_model, max_tokens=8000,
                       reasoning_effort="low", fallback_reasoning=False,
                       max_empty_retries=args.cloud_empty_retries,
                       empty_cooldown=args.cloud_empty_cooldown,
                       request_interval=args.request_interval,
                       status_dir=live_dir, role="guide", provider=args.cloud_provider)
    judge = CloudModel(endpoint=cloud_endpoint, model=args.judge_model, max_tokens=8000,
                       reasoning_effort="low", fallback_reasoning=False, timeout=300,
                       max_empty_retries=args.cloud_empty_retries,
                       empty_cooldown=args.cloud_empty_cooldown,
                       request_interval=args.request_interval,
                       status_dir=live_dir, role="judge", provider=args.cloud_provider)
    auditor = CloudModel(endpoint=cloud_endpoint, model=args.auditor_model, max_tokens=1600, timeout=300,
                          max_empty_retries=args.cloud_empty_retries,
                          empty_cooldown=args.cloud_empty_cooldown,
                          request_interval=args.request_interval,
                          status_dir=live_dir, role="auditor", provider=args.cloud_provider)
    if not guide.api_key:
        print("[soak] ERROR: no cloud API key (set OPENCODE_GO_API_KEY)")
        return 1

    soak = SoakRun(spreadsheet, args, run_dir, local, guide, judge, auditor)
    soak._start_cloud_workers()
    local.history_provider = lambda: soak.history
    if args.level == 2:
        soak._make_summarizer(sandbox)

    # resume: reconstruct history from saved turn dirs so history-hash paths line up
    if resume_from > 0:
        for t in range(resume_from):
            d = run_dir / f"turn_{t:03d}"
            if (d / "user.txt").exists() and (d / "reply.txt").exists():
                soak.history.append([
                    (d / "user.txt").read_text(encoding="utf-8"),
                    (d / "reply.txt").read_text(encoding="utf-8"),
                ])
            # Reload judge/audit records from turn results
            result_path = d / "result.json"
            if result_path.exists():
                try:
                    r = json.loads(result_path.read_text(encoding="utf-8"))
                    j = r.get("judge")
                    if j and "error" not in j:
                        j["turn"] = r.get("turn", t)
                        soak.judge_records.append(j)
                    a = r.get("audit")
                    if isinstance(a, list):
                        for av in a:
                            if isinstance(av, dict) and "error" not in av:
                                av["turn"] = r.get("turn", t)
                                soak.audit_records.append(av)
                except Exception:
                    pass
        plan_path = run_dir / "plan.json"
        if plan_path.exists():
            try:
                soak.plan = json.loads(plan_path.read_text(encoding="utf-8"))
            except Exception:
                soak.plan = {}
        print(f"[soak] resumed {len(soak.history)} prior exchanges "
              f"({len(soak.judge_records)} judge records, {len(soak.audit_records)} audit records)")

        if args.clear_from:
            _clear_judge_audit_overview(run_dir, args.clear_from)
            # Also drop the in-memory records that were loaded from the now-cleared
            # disk files, so the report/final-overview don't mix stale + fresh data.
            if args.clear_from < resume_from:
                soak.judge_records = [r for r in soak.judge_records
                                       if r.get("turn", 0) < args.clear_from]
                soak.audit_records = [r for r in soak.audit_records
                                       if r.get("turn", 0) < args.clear_from]
            if not args.backfill:
                soak_log(f"--clear-from: done. run_dir={run_dir}")
                return 0

        soak.backfill_post_turn_events(resume_from)

        if args.backfill:
            soak_log("--backfill: joining cloud workers to let backfilled tasks complete ...")
            soak._join_cloud_workers()
            from report import write_report
            write_report(run_dir, spreadsheet, args)
            soak_log(f"--backfill: done. run_dir={run_dir}")
            return 0

    # graceful stop
    def _sig(signum, frame):
        print(f"\n[soak] signal {signum}; checkpointing after current turn")
        soak._stop = True
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    # health check before the loop
    soak_log("health check: local model ...")
    if not health_check(base_url, api_key, model):
        soak_log("ERROR: local model not healthy at {}/models".format(base_url))
        return 1

    try:
        for turn in range(resume_from, total):
            if soak._stop:
                break
            global _SOAK_TURN
            _SOAK_TURN = turn
            soak_log(f"--- turn {turn}/{total} ---")
            t0 = time.time()
            local_before = local.stats()
            guide_before = soak.guide.stats()
            result = soak.run_turn(turn)
            dt = time.time() - t0
            result["dt_s"] = round(dt, 1)

            # per-turn usage (deltas of the cumulative counters) + wall-clock ts.
            # judge/auditor usage is attributed per-task by the background workers
            # (they merge judge_usage/auditor_usage into result.json directly).
            local_after = local.stats()
            guide_after = soak.guide.stats()
            result["local_usage"] = _delta_stats(local_before, local_after)
            result["local_usage"]["model"] = local_after.get("model")
            result["local_usage"]["max_prompt_tokens"] = local_after.get("max_prompt_tokens", 0)
            result["local_usage"]["ctx_size"] = local_after.get("ctx_size", 0)
            # Full (untruncated) accumulated history estimate: what the whole
            # conversation costs if NOT bounded by --max-update-history, per the
            # existing //4 chars->tokens convention (self.history already holds
            # this turn's [user, reply] pair when the checkpoint runs).
            full_chars = sum(len(u) + len(r) for u, r in soak.history)
            result["local_usage"]["full_history_tokens"] = full_chars // 4
            result["local_usage"]["full_history_chars"] = full_chars
            result["cloud_usage"] = _delta_stats(guide_before, guide_after)
            result["cloud_usage"]["model"] = guide_after.get("model")
            result["wall_ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            cloud_s = result.get("cloud_s", 0)
            soak_log(f"  done: turn {turn} in {dt:.1f}s (cloud {cloud_s:.0f}s, local {dt - cloud_s:.0f}s)")
            soak_log(f"  user: {result['user_input'][:100]!r}")
            soak_log(f"  reply: {result['reply'][:100]!r}")
            soak_log(f"  local_calls={result['local_usage']['calls']} "
                     f"cloud={result['cloud_usage']['calls']}")

            # checkpoint (atomic)
            turn_dir = run_dir / f"turn_{turn:03d}"
            tmp = turn_dir.with_suffix(".tmp")
            tmp.mkdir(parents=True, exist_ok=True)
            (tmp / "user.txt").write_text(result["user_input"], encoding="utf-8")
            (tmp / "reply.txt").write_text(result["reply"], encoding="utf-8")
            (tmp / "instr_prompt.txt").write_text(result["instr_prompt"], encoding="utf-8")
            if result.get("context"):
                (tmp / "context.txt").write_text(result["context"], encoding="utf-8")
            payload = result.pop("retrieval_payload", None)
            if payload:
                (tmp / "retrieval_payload.json").write_text(json.dumps(payload), encoding="utf-8")
            (tmp / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            if soak.plan:
                (tmp / "plan.json").write_text(json.dumps(soak.plan, indent=2), encoding="utf-8")
            cur = soak._current_state_dir()
            if cur and cur.exists():
                (tmp / "state_snapshot").mkdir(exist_ok=True)
                for f in cur.glob("*.json"):
                    shutil.copy2(f, tmp / "state_snapshot" / f.name)
            if turn_dir.exists():
                shutil.rmtree(turn_dir)
            tmp.rename(turn_dir)

            manifest["last_turn"] = turn
            (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

            # dispatch this turn's judge/audit to the background workers NOW that
            # result.json is checkpointed — they merge their verdicts in-place.
            soak.flush_pending_tasks()

            # rolling overview (opt-in): the judge writes a mid-run overview every
            # --overview-every turns, on the judge worker (serialized after the
            # per-turn judge calls so judge_records is complete).
            if args.overview_every and ((turn + 1) % args.overview_every == 0):
                soak._judge_q.put(soak._make_rolling_overview_task(turn))

            if soak._stop:
                soak_log("graceful stop; checkpointed")
                break
    except Exception as e:
        soak_log(f"ERROR at turn {soak.history and len(soak.history) - 1}: {e}")
        traceback.print_exc()
        soak._join_cloud_workers()
        return 1

    # let in-flight judge/audit/rolling-overview verdicts land before report/final overview
    soak._join_cloud_workers()

    # report
    from report import write_report
    write_report(run_dir, spreadsheet, args)

    # final overview: end-of-run judge pass over the whole chat.
    # Skip when no new turns were run (e.g. a resume that only backfilled
    # judge/audit gaps) — the original final overview is authoritative and
    # regenerating it from a mixed old+new verdict set produces a misleading
    # narrative (e.g. "absence of structured state" from backfilled "missed" notes).
    if resume_from < total or not (run_dir / "final_overview.json").exists():
        try:
            cur = soak._current_state_dir()
            overview = soak._run_final_overview(cur)
            if overview:
                overview["turns"] = len(soak.history)
                overview["judge_records"] = soak.judge_records
                overview["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                (run_dir / "final_overview.json").write_text(json.dumps(overview, indent=2), encoding="utf-8")
                soak_log(f"final overview written to {run_dir / 'final_overview.json'}")
        except Exception as e:
            soak_log(f"final overview skipped: {e}")
    else:
        soak_log(f"final overview skipped (no new turns; keeping existing)")

    soak_log(f"done. run_dir={run_dir}")
    soak_log(f"guide cloud: {soak.guide.stats()}")
    soak_log(f"judge cloud: {soak.judge.stats()}")
    soak_log(f"auditor cloud: {soak.auditor.stats() if soak.auditor else 'n/a'}")
    soak_log(f"local calls: {local.call_count} tokens p={local.prompt_tokens} c={local.completion_tokens}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
