"""Long-horizon soak orchestrator.

Runs one continuous multi-genre conversation through the real dayna_ss engine on
a live local model, while a cloud model (OpenCode Go) guides the story and judges
memory fidelity. See ``docs/plans/long_horizon_soak_plan.md`` §5-§8 for the design.

Usage::

    python long_horizon_soak.py --spreadsheet spreadsheets/noir_detective.json
        --turns 100 --guide-style implicit --specificity-profile mixed
        [--smoke 5] [--resume runs/<dir>] [--level 2] [--director on]

Environment (see plan §0):
    DSS_BENCH_BASE_URL   local OpenAI-compatible server (default http://127.0.0.1:5000/v1)
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


def _soak_log_call(phase_id: str, step_id: str, ttft: float, total: float, pt: int, ct: int) -> None:
    """Per-local-call latency line: real TTFT + full round-trip + token counts.

    ``ttft`` is time-to-first-token as observed by the harness: the interval from
    request issue until the first streamed delta carrying content/reasoning
    (``_complete`` requests ``stream: true`` and parses the SSE chunks). Under
    concurrent workers it includes server-side queue time, not just prefill.
    ``total`` is the full round-trip (prefill + decode + transport + queue) from
    request issue to the [DONE] frame. Maps soak calls to the LMDeploy server log
    on the ``input_tokens`` (approximate prefill rate = pt/total).
    """
    label = f"{phase_id or '?'}/{step_id or 'generate'}"
    if ttft is not None:
        soak_log(f"  local-call {label}: ttft={ttft:.2f}s total={total:.2f}s "
                 f"in={pt} out={ct} tok ({(pt / total / 1000):.1f}K tok/s)")
    else:
        soak_log(f"  local-call {label}: ttft=— total={total:.2f}s "
                 f"in={pt} out={ct} tok ({(pt / total / 1000):.1f}K tok/s)")


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

DEFAULT_LOCAL_BASE = "http://127.0.0.1:5000/v1"


def _arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Long-horizon soak for dayna_ss")
    p.add_argument("--spreadsheet", default=None, help="path to spreadsheet JSON (not needed with --final-overview)")
    p.add_argument("--turns", type=int, default=100, help="total turns (default 100)")
    p.add_argument("--guide-style", choices=["explicit", "implicit"], default="implicit")
    p.add_argument("--specificity-profile", choices=["loose", "mixed", "exact"], default=None)
    p.add_argument("--director", choices=["on", "off"], default="on")
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
    p.add_argument("--log-file", default=None,
                   help="path to a dedicated soak logfile; default writes to runs/logs/<spreadsheet>_<hash>.log")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--local-base", default=DEFAULT_LOCAL_BASE)
    p.add_argument("--cloud-endpoint", default=None, help="common self-hosted OpenAI-compatible base URL for all cloud models (e.g. http://localhost:9931). Overrides default cloud endpoint.")
    p.add_argument("--cloud-provider", default=None, help="auth.json entry name to use as the API key for --cloud-endpoint (e.g. 'llama' or 'textgen'). When omitted, resolve_local_api_key() picks from its default order.")
    p.add_argument("--judge-window", type=int, default=12, help="exchanges shown to the judge")
    p.add_argument("--max-update-history", type=int, default=8,
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
                 max_update_history_exchanges: int = 8):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.repetition_penalty = repetition_penalty
        self.history_provider = history_provider  # callable() -> list of [u, r]
        self.max_update_history_exchanges = max_update_history_exchanges
        self.call_count = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.max_prompt_tokens = 0
        self.ctx_size = 65536
        self.last = None

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
        if not _thinking_enabled():
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
                ttft, total, content, reasoning, pt, ct = self._stream_post(req, t0)
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
                _soak_log_call(phase_id, step_id, ttft, total, pt, ct)
                return content or reasoning
            if attempt < 2:
                time.sleep(0.5 * (attempt + 1))
        _soak_log_call(phase_id, step_id, ttft, total, pt, ct)
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
        pt = ct = 0
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
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        content += delta["content"]
                    if delta.get("reasoning_content"):
                        reasoning += delta["reasoning_content"]
        return ttft, time.perf_counter() - t0, content, reasoning, pt, ct

    def generate_with_streaming(self, encoded_prompt: Any, state: dict):
        self.call_count += 1
        text = self._complete(str(encoded_prompt), state)
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

_SCORE_ANCHORS = """SCORE CALIBRATION — use the FULL scale; anchor every numeric score you return to these meanings:
- 5.0 — Perfect. Masterfully written and composed: publication-quality craft, zero flaws in any scored dimension. Reserved for genuinely exceptional output; most good work is NOT a 5.
- 4.5 — Extraordinary. The expected quality of an experienced professional human writer on a good day: controlled, vivid, precise, essentially flawless apart from trivial nitpicks.
- 4.0 — Excellent. Impressive and production-ready — unambiguously "good enough" by any professional standard; strong craft whose minor imperfections do not call for revision.
- 3.5 — Good. Solid and readable with real craft, but carrying noticeable flaws (some repetition, thin beats, minor slips) a careful reader would feel.
- 3.0 — Functional. Passable but clearly flawed; multiple notable issues that accumulate.
- 2.5 — Weak. Several significant flaws across dimensions.
- 2.0 — Poor. Fails most dimensions.
- 1.0–1.5 — Failing. Fundamentally broken output.
Calibration rule: competent-but-flawed output lands 2.5–3.5; a 4.0 must be EARNED, not granted for effort; do NOT cluster scores at 3–4 out of caution when the work is plainly better or worse than that. The MANDATORY FLOORS still cap scores regardless of this ladder."""


JUDGE_SYSTEM_TMPL = """You are a quality auditor grading ONE writer — {name2} ({role2}), the DSS-driven writer — on a single turn of a collaborative story. {name1} ({role1}) writes the OTHER writer's turns; those turns are provided ONLY as CONTEXT (what {name2} was responding to). Do not credit or penalize {name2} for anything {name1} wrote, and never score {name1}'s prose or the joint plot. The verdict measures {name2} alone.

Score four things, all about {name2}'s turn:
(1) style_score: did {name2}'s reply hold HER directed voice ({dss_directive}) and its hard rules;
(2) quality_score: the narrative quality of {name2}'s reply itself (not the story's plot);
(3) memory_fidelity: does {name2}'s private memory of the story so far reflect what actually happened;
(4) per-note adherence: for the notes in scope this window, did {name2}'s turn plant/echo/honor them.

{anchors}

SCORING RUBRIC (be strict — a mediocre turn must NOT score 4+):
style_score:
- 5: the whole turn holds {name2}'s directed voice and hard rules with zero violations.
- 4: a single minor slip (one stray word in the wrong person — e.g. one 'I' when the directive demands third person, or one 'he/she' when it demands first — one slightly repetitive phrase) that does not break voice.
- 3: TWO or more slips, or ONE near-verbatim re-quote of {name1}, or a repeated closing line from an earlier turn.
- 2: recurring wrong-person narration throughout (e.g. first-person 'I' when {dss_directive} demands third person, or third person when it demands first, or a second-person 'you' that drifts to 'I'/'he'), or repeated verbatim re-quoting of {name1}, or a wholesale tonal break.
- 1: the reply is barely in character at all.
quality_score:
- 5: advances the scene with concrete action or observation; fresh details; no repetition, no continuity break.
- 4: competent but leans on imagery/atmosphere more than it moves the scene, or references few of the in-scope notes despite them being relevant this turn.
- 3: repetitive imagery or metaphor, OR a scene inconsistency (a character present who is established absent, a location mismatch), OR the reply is largely atmospheric with no concrete action, dialogue, or new detail.
- 2: multiple scene inconsistencies or heavy self-repetition, or it ignores nearly all in-scope notes entirely.
- 1: incoherent, off-topic, or content-free.
memory_fidelity:
- 5: {name2}'s memory matches the story — roles, relationships, and facts all consistent with the EXACT stored fields shown in {name2}'s private memory below.
- 4: one minor misattribution or a small factual slip.
- 3: TWO or more minor errors, or one role/identity confusion (a character's role or relationship misstated).
- 2: recurring identity confusion — e.g. {name2} treating HERSELF as another character, the suspect, or the protagonist of the other writer's arc — or state that contradicts the premise outright.
- 1: memory is wholesale wrong or contradicts the premise.

GROUND MEMORY SCORES IN THE STORED STATE: judge {name2}'s memory strictly against the exact fields listed under "[EVALUATED] {name2}'s private memory" below. Do not invent contradictions that are not visible in the stored state, and do not infer a fact the memory does not contain. If a subject is shown as "(EMPTY)" (e.g. '== characters == (EMPTY — no entries saved)'), that emptiness IS a memory failure — score it down accordingly and say so explicitly.

MANDATORY FLOORS (apply regardless of how pretty the prose is):
- If {name2} casts herself as the wrong character (e.g. as {name1}'s identity/role), memory_fidelity is at most 2 and overall cannot reach 4. (Exception: if the premise explicitly establishes that {name1} and {name2} share one identity — a 'same-perspective' story — shared-identity narration is CORRECT, not a violation.)
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

Return STRICT JSON only, exactly this shape:
{{
  "style_score": 0-5, may use 0.5 steps ({name2}'s voice adherence to {dss_directive}),
  "quality_score": 0-5, may use 0.5 steps (quality of {name2}'s reply),
  "memory_fidelity": 0-5, may use 0.5 steps,
  "notes": [{{"id": "<note id>", "status": "planted|echoed|missed|superseded|n/a", "detail": "one sentence on what {name2}'s turn did with this note"}}],
  "abort": <true|false>,
  "summary": "two-sentence editorial note on {name2}'s turn only"
}}"""


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


def build_judge_messages(spreadsheet: dict, notes_in_scope: list[dict], window: list[list[str]],
                         state_summary: str, reply: str, synthetic: bool = False,
                         unplanted_ids: list[str] | None = None,
                         instructions: str = "") -> list[dict]:
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
    user_parts = [
        f"Story premise (context): {spreadsheet['premise']}",
        "Notes in scope this window (grade {name2}'s adherence to these, not {name1}'s)".format(name1=name1, name2=name2),
        json.dumps([{"id": n["id"], "type": n["type"], "content": n["content"]} for n in notes_in_scope], indent=1),
        "Recent window — [CONTEXT] is {name1}'s guide turn (NOT scored); [EVALUATED] is {name2}'s DSS output (the object of grading).".format(name1=name1, name2=name2),
    ]
    if unplanted_ids:
        user_parts.append(
            f"{len(unplanted_ids)} outline note(s) never appeared in the story text and are NOT in "
            f"scope for this grade: {sorted(unplanted_ids)}. Never mark them 'missed' — the guide "
            f"never planted them, so {name2} could not have honored them."
        )
    if instructions:
        user_parts.append(
            f"INSTRUCTIONS {name2} WAS GIVEN for this reply (the harness-generated instruction block):\n{instructions}\n"
            "ATTRIBUTION RULE: if a detail or phrasing in the reply comes VERBATIM from these instructions, that is "
            f"poor EXECUTION on {name2}'s part — a good writer performs an instruction in prose, it does not copy "
            f"the instruction's words. But do NOT penalize {name2} for content the instructions MANDATED (a prop, an "
            f"animal, a specified ending) as if it were {name2}'s own creative choice, and do NOT score it as a "
            f"memory error or identity confusion — {name2} was told to write it."
        )
    for u, r in window:
        user_parts.append(f"[CONTEXT — {name1} ({role1}), NOT scored]: {u}")
        user_parts.append(f"[EVALUATED — {name2} ({role2})]: {r}")
    if synthetic:
        user_parts.append(
            f"SYNTHETIC-CHAT ISOLATION MODE: the [EVALUATED] {name2} reply was written by the "
            f"CLOUD model, NOT by DSS. Score ONLY DSS's MEMORY FIDELITY (did its private memory track "
            f"what happened) and NOTE RETENTION. Set voice/style/progression to a neutral pass — DSS "
            f"did not write the reply this turn."
        )
    user_parts.append(f"[EVALUATED] Latest {name2} reply: {reply}")
    user_parts.append(f"[EVALUATED] {name2}'s private memory of the story so far:\n{state_summary}")
    user_parts.append("Score the four dimensions for {name2}'s turn now.".format(name2=name2))
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(user_parts)},
    ]


FINAL_OVERVIEW_SYSTEM_TMPL = """You are a quality auditor evaluating ONE writer only: {name2} ({role2}) — the DSS-driven writer. {name1} ({role1}) wrote the OTHER writer's turns. This overview measures {name2}'s performance ALONE, never the story as a whole and never {name1}.

The story was written as an exchange: {name1} writes a turn, then {name2} replies. {name1}'s turns are provided ONLY as CONTEXT — they are the stimuli {name2} was responding to. Do not credit {name2} for ideas {name1} introduced, and do not penalize {name2} for plot or voice elements that came from {name1}'s turns. {name1}'s turns are context only; do not credit or penalize {name1} in any dimension. The verdict measures {name2} alone.

What is evaluated is {name2}'s craft as a writer and as the DSS agent:
- her adherence to her directed voice ({dss_directive}) and its hard rules;
- her use of her private memory of the story;
- how faithfully she honored the outline notes in her turns;
- how well her turns advanced or landed the intended beats.

SCORING RUBRIC (be strict — a mediocre run must NOT score 4+):
- overall_score is NOT the joint story quality and NOT a straight average of the sub-scores. Weigh:
  voice adherence ~30%, memory fidelity ~25%, note/clue usage ~25%, scene progression ~20%.
- voice: 5 = zero voice violations across the run; 4 = an occasional single slip; 3 = recurring slips or repeated verbatim re-quotes of {name1} or repeated closing lines; 2 = wholesale wrong-person drift (first person where the directive demands third, or vice versa, or a second-person 'you' slipping into 'I'/'he') or habitual re-quoting; 1 = never in character.
- memory: 5 = roles/relationships/facts tracked correctly throughout; 4 = a few minor misattributions; 3 = an identity or role confusion that recurs; 2 = {name2} repeatedly casts herself as the wrong character (e.g. as {name1}'s identity or as the suspect) or contradicts the premise; 1 = wholesale wrong. (Exception: if the premise establishes that {name1} and {name2} share one identity — a 'same-perspective' story — shared-identity narration is CORRECT.)
- notes: grade ONLY the notes marked in scope (the unplanted ones are excluded — {name2} cannot honor a note the other writer never planted). 5 = most in-scope notes planted/echoed at the right turns; 4 = most notes used but some fumbled; 3 = roughly half the in-scope notes missed; 2 = the clear majority missed; 1 = notes essentially ignored.
- progression: 5 = her turns consistently advanced the beats; 4 = mostly advancing but some beats fumbled; 3 = imagery over action, several beats missed; 2 = rarely advanced anything; 1 = static or regressive.

{anchors}

GROUND MEMORY SCORES IN THE STORED STATE: judge {name2}'s memory strictly against the exact fields in "{name2}'s private memory at the end" given below. Do not invent contradictions not visible in the stored state, and do not infer facts the memory does not contain. A subject shown as "(EMPTY)" (e.g. '== characters == (EMPTY — no entries saved)') IS a memory failure — say so and score it down. Attribution: {name2} receives per-turn instructions generated by the harness; if a recurring flaw in her turns was explicitly mandated by those instructions (a specified prop, animal, or ending), weigh it as weak EXECUTION, not as {name2}'s own creative choice or a memory error.

MANDATORY FLOORS (apply regardless of how pretty the prose is):
- Recurring identity confusion ({name2} as the wrong character) caps overall_score at 3, and memory-fidelity discussion must call it out explicitly.
- More than half of the IN-SCOPE (planted) notes missed caps overall_score at 3.
- Any wholesale voice break (narration in the wrong person throughout, habitual verbatim re-quoting) caps overall_score at 3.
- A 5 overall requires near-zero flaws in all four dimensions; a 4 overall requires at most ONE moderately flawed dimension; anything with multiple flawed dimensions is 3 or below.

Be consistent with the per-turn judge records you are given: if the judge records show repeated low memory_fidelity or a high missed-note rate, your overall_score must reflect that. Do not let excellent prose inflate the overall score when memory or note usage failed.

Do NOT score the joint plot's inherent quality, do NOT score {name1}'s prose, and do NOT review the story "as a whole". Ignore how good or bad the resulting story happens to be; focus purely on how well {name2} executed her part.

Style directives (for reference only — {name1}'s voice is not scored):
- {name1}'s voice: {guide_directive}
- {name2}'s voice: {dss_directive}
{taboos}

You are given: the premise, the full outline notes, the entire transcript (each line labeled
CONTEXT or EVALUATED), the per-turn judge records produced during the run, and {name2}'s
private memory at the end.

Return STRICT JSON only, exactly this shape:
{{
  "overall_score": 0-5, may use 0.5 steps — score of how well {name2} performed as the DSS writer per the SCORING RUBRIC above (voice adherence ~30%, memory fidelity ~25%, note/clue usage ~25%, scene progression ~20%; apply the MANDATORY FLOORS) — NOT joint story quality,
  "verdict": "one-sentence verdict on {name2}'s performance alone",
  "arc_progression": "2-3 sentences on how well {name2}'s turns advanced or landed the intended beats (what the DSS contributed), not how good the plot is",
  "style_consistency": "2-3 sentences on whether {name2} held HER directed voice ({dss_directive}) and its hard rules; {name1}'s voice is not scored",
  "memory_fidelity": "2-3 sentences on whether {name2}'s private memory tracked the actual story",
  "notes": [{{"id": "<note id>", "status": "planted|echoed|missed|superseded", "detail": "one sentence on what {name2}'s turns did with this note"}}],
  "strengths": ["2-4 strengths specific to {name2}'s turns"],
  "weaknesses": ["2-4 weaknesses specific to {name2}'s turns"],
  "summary": "3-4 sentence final editorial summary of {name2}'s performance alone; do not evaluate {name1} or the joint plot"
}}"""


def build_final_overview_messages(spreadsheet: dict, notes: list[dict], history: list[list[str]],
                                  judge_records: list[dict], state_summary: str,
                                  scope: str = "final", unplanted_ids: list[str] | None = None) -> list[dict]:
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
    user_parts.append("Outline notes (the intended arc):")
    user_parts.append(json.dumps([{"id": n["id"], "type": n["type"], "content": n["content"]} for n in notes], indent=1))
    if unplanted_ids:
        user_parts.append(
            f"The following outline note(s) NEVER appeared in the story text — the other writer "
            f"never planted them — so they are NOT in scope and must be marked 'superseded', never "
            f"'missed': {sorted(unplanted_ids)}"
        )
    user_parts.append(
        "Full transcript — [CONTEXT] lines are {name1}'s guide turns (read them to understand what {name2} was "
        "responding to, but do NOT score them); [EVALUATED] lines are {name2}'s DSS output, the object of this review."
        .format(name1=name1, name2=name2)
    )
    for u, r in history:
        user_parts.append(f"[CONTEXT — {name1} ({role1}), NOT scored]: {u}")
        user_parts.append(f"[EVALUATED — {name2} ({role2})]: {r}")
    user_parts.append(f"[EVALUATED] {name2}'s private memory at the end (score its fidelity):\n{state_summary}")
    if judge_records:
        user_parts.append("Per-turn judge records (from during the run; these already scored {name2} only):".format(name2=name2))
        for rec in judge_records:
            user_parts.append(json.dumps(rec, indent=1))
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
events, current_scene, general_info).

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
  "status": "saved|partial|wrong|missing|unassessable",
  "abort": <true|false>,
  "detail": "one-two sentences: what DSS actually saved (or failed to save), and what is wrong if anything"
}}

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
                         capability_context: str = "", turn: int = 0) -> list[dict]:
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
    context = (
        f"Premise: {spreadsheet['premise']}\n"
        f"Setting: {spreadsheet['setting']}\n"
        f"Canonical cast: {n1['name']} = {n1.get('role', 'the other character')} "
        f"(NOT the protagonist; the other character). {n2['name']} = "
        f"{n2.get('role', 'the protagonist')} (the protagonist being roleplayed).\n"
        f"Your voice ({n2['name']}, DSS turns): {ws['dss_directive']}\n"
        f"Taboos: {taboos}\n"
        f"Characters:\n- {n1['name']}: {n1.get('description', n1.get('role', ''))}\n"
        f"- {n2['name']}: {n2.get('description', n2.get('role', ''))}"
    )
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
        planted = set(self._planted_note_ids())
        judge = self.judge

        def task() -> None:
            t0 = time.time()
            soak_log(f"  judge: calling cloud ({judge.model}) json_mode ...")
            base = judge.stats()
            try:
                verdict = self._run_judge(turn, active_notes, reply, state_dir,
                                          window=window, planted=planted)
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

    def _beats_due(self, turn: int) -> bool:
        """True if any spreadsheet dss_beat is scheduled for this turn."""
        for beat in self.spreadsheet.get("dss_beats", []):
            due = beat.get("turn")
            if due is not None and int(due) == turn:
                return True
        return False

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
        if not due:
            return None
        state_summary = self._memory_summary(state_dir) if state_dir else "(no state)"
        results = []
        for beat in due:
            capability = self._audit_capability_context(beat, turn)
            msgs = build_audit_messages(self.spreadsheet, beat, state_summary, capability, turn=turn)
            t0 = time.time()
            soak_log(f"  auditor: beat {beat['id']} vs DSS state ({self.auditor.model}) ...")
            try:
                verdict = self.auditor.json_complete(msgs, max_tokens=1600, json_mode=True)
            except CloudError as e:
                soak_log(f"  auditor: beat {beat['id']} call failed: {e}")
                verdict = {"id": beat["id"], "status": "error", "detail": str(e)}
            verdict.setdefault("id", beat["id"])
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
        # The selected schema is copied into the sandbox at the default location
        # (see main()'s sandbox setup), so the engine resolves
        # `extension_dir / user_data/example/subjects_schema.json` to the variant.
        return self.summarizer

    # ------------------------------------------------------ state summary ---

    def _memory_summary(self, state_dir: Path | None) -> str:
        """Bounded DSS-state inventory for guide/judge/auditor.

        Renders the FULL inventory of what DSS has saved (every character, group,
        element, event, arc — not just a raw-JSON prefix), with per-entry detail
        truncated so the list is never cut short. The old implementation dumped
        ``json.dumps(entries)[:2500]``, which hid every entry after the first two
        characters and omitted groups/elements/events/arcs entirely — so the
        auditor judged "missing" on state that was actually saved.
        """
        if not state_dir or not state_dir.exists():
            return "(no structured memory yet)"
        parts = []

        def compact_entries(data: dict, per_entry: int = 160, max_total: int = 2800) -> str:
            entries = data.get("entries", data) if isinstance(data, dict) else {}
            if not isinstance(entries, dict) or not entries:
                return ""
            lines = []
            for name, entry in entries.items():
                try:
                    blob = json.dumps(entry, ensure_ascii=False, indent=0)[:per_entry]
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
                    data = json.loads(p.read_text(encoding="utf-8"))
                    summary = compact_entries(data)
                    if summary:
                        parts.append(f"== {fname.replace('.json', '')} ==\n" + summary)
                    else:
                        parts.append(f"== {fname.replace('.json', '')} == (EMPTY — no entries saved)")
                except Exception:
                    continue

        for fname in ("events.json", "arcs.json"):
            p = state_dir / fname
            if p.exists():
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    if not data:
                        parts.append(f"== {fname.replace('.json', '')} == (EMPTY — nothing saved)")
                    else:
                        parts.append(f"== {fname.replace('.json', '')} ==\n" + json.dumps(data, ensure_ascii=False)[:1500])
                except Exception:
                    continue

        for fname in ("general_info.json", "current_scene.json"):
            p = state_dir / fname
            if p.exists():
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    parts.append(f"== {fname} ==\n" + json.dumps(data, indent=1)[:1500])
                except Exception:
                    continue
        return "\n".join(parts) if parts else "(no structured memory yet)"

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
        memory = self._memory_summary(state_dir)
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

        # 1. guide writes the user turn
        active_notes = notes_visible_at(spread["notes"], turn, args.guide_style)
        state_dir = self._current_state_dir()
        replanned = self._maybe_replan(turn, state_dir)
        memory = self._memory_summary(state_dir)
        guide_msgs = build_guide_messages(spread, active_notes, self.history, args.guide_style,
                                          memory, self._plan_view(),
                                          greeting=spread.get("greeting", ""))
        user_input = guide_turn_with_retry(
            self.guide, spread, guide_msgs,
            max_attempts=self.args.guide_max_attempts,
            cooldown=self.args.cloud_empty_cooldown)
        if args.arc_breaks == "3_arcs" and turn in (25, 50, 75):
            # Stage via persistent_ui_state — the same channel production's
            # NEXT ARC: prefix feeds. Prepending the literal prefix here does
            # nothing: the harness never routes through TGWUI's
            # chat_input_modifier, and the engine only parsed prefixes there.
            from extensions.dayna_ss.runtime import runtime
            runtime.persistent_ui_state["force_next_arc"] = True
        if getattr(args, "force_chapter_turn", 0) and turn == args.force_chapter_turn:
            from extensions.dayna_ss.runtime import runtime
            runtime.persistent_ui_state["force_next_chapter"] = True

        # 2. engine reply prompt
        # The raw state handed to the engine carries the full production-shaped
        # internal history (greeting pair at [0] + every exchange so far) — a pure
        # session emulator. The engine reads it for the greeting ([0][1]) and for
        # DSS's own recent replies (generate_instr_prompt's R1 negative-exemplar
        # block); custom_state's internal (built from `history_internal`) is the
        # separate artificial retrieval context for the model payloads.
        engine_state = build_state(spread, args.seed, f"soak_{args.seed}", history=self.history)
        greeting = engine_state.get("greeting", "").strip()
        # Production view: the conversation INCLUDES the character greeting pair at
        # index 0 (assistant side), so message indices / rolling pairs / format_dialogue
        # all match production. Cloud roles (guide/judge/synthetic) keep seeing only the
        # [user, reply] exchanges — the greeting is name2's own line, not their concern.
        history_internal = ([["<|BEGIN-VISIBLE-CHAT|>", greeting]] if greeting else []) + [
            list(e) for e in self.history
        ]
        t0 = time.time()
        soak_log("  engine: building instruction prompt (retrieval + cache warm) ...")
        instr_prompt, custom_state, history_path, ts = self.summarizer.generate_instr_prompt(
            user_input, engine_state, history_internal, do_instr=(args.director == "on")
        )
        soak_log(f"  engine: instr prompt built in {_elapsed(t0)}")
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
            "history_path": str(history_path) if history_path else None,
            "probes": probes,
            "recalls": recalls,
            "judge": None,
            "audit": None,
            "plan": self.plan,
            "ts": ts,
            "local_calls": self.local.call_count,
            "cloud_usage": self.guide.stats(),
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
        schema_parser = SchemaParser(schema_path)

        # guide writes the turn
        active_notes = notes_visible_at(spread["notes"], turn, self.args.guide_style)
        memory = self._memory_summary(state_dir)
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
                   window: list[list[str]] | None = None, planted: set[str] | None = None) -> dict | None:
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
        msgs = build_judge_messages(spread, notes_in_scope, window, state_summary, reply,
                                    unplanted_ids=unplanted_ids, instructions=instr_text,
                                    synthetic=bool(getattr(self.args, "synthetic_chat", False)))
        try:
            result = self.judge.json_complete(msgs, max_tokens=8000, json_mode=True)
        except CloudError as e:
            print(f"[soak] judge call failed at turn {turn}: {e}")
            return {"error": str(e)}
        if result and "error" not in result:
            result["turn"] = turn
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
        msgs = build_final_overview_messages(
            self.spreadsheet, in_scope_notes, history, judge_records,
            state_summary, scope=scope, unplanted_ids=unplanted_ids,
        )
        try:
            result = self.judge.json_complete(
                msgs, max_tokens=16000, json_mode=True, reasoning_effort="low",
            )
        except CloudError as e:
            print(f"[soak] {scope} overview call failed: {e}")
            return {"error": str(e)}
        return result


def _backfill_overview(run_dir: Path, scope: str = "final", checkpoint: int | None = None) -> int:
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

    # state summary from the last snapshot (best-effort reconstruction)
    def _mem(snap: Path | None) -> str:
        if not snap or not snap.exists():
            return "(no structured memory yet)"
        parts = []
        for fname in ("general_info.json", "current_scene.json"):
            p = snap / fname
            if p.exists():
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    parts.append(f"== {fname} ==\n" + json.dumps(data, indent=1)[:1500])
                except Exception:
                    continue
        chars_p = snap / "characters.json"
        if chars_p.exists():
            try:
                chars = json.loads(chars_p.read_text(encoding="utf-8"))
                parts.append("== characters ==\n" + json.dumps(chars.get("entries", {}), indent=1)[:2500])
            except Exception:
                pass
        return "\n".join(parts) if parts else "(no structured memory yet)"

    live_dir = run_dir / "live"
    try:
        live_dir.mkdir(exist_ok=True)
    except OSError:
        live_dir = None
    judge = CloudModel(model=manifest.get("args", {}).get("judge_model") or "deepseek-v4-flash",
                       max_tokens=8000, timeout=300,
                       status_dir=str(live_dir) if live_dir else None, role="judge")
    if not judge.api_key:
        print("[soak] ERROR: no cloud API key (set OPENCODE_GO_API_KEY)")
        return 1
    state_summary = _mem(last_state_dir)
    story_text = "\n".join(u + "\n" + r for u, r in history).lower()
    planted = {n["id"] for n in spreadsheet["notes"]
               if not (n.get("needle") or "").strip() or (n.get("needle") or "").strip().lower() in story_text}
    in_scope = [n for n in spreadsheet["notes"] if n["id"] in planted]
    unplanted_ids = [n["id"] for n in spreadsheet["notes"] if n["id"] not in planted]
    msgs = build_final_overview_messages(spreadsheet, in_scope, history, judge_records,
                                         state_summary, unplanted_ids=unplanted_ids, scope=scope)
    try:
        result = judge.json_complete(
            msgs, max_tokens=16000, json_mode=True, reasoning_effort="low",
        )
    except CloudError as e:
        print(f"[soak] {scope} overview call failed: {e}")
        return 1
    if not result:
        result = {"error": "empty response"}
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
        return _backfill_overview(args.final_overview, scope="final")

    if args.backfill_rolling_overview:
        code = 0
        for run_dir, n in args.backfill_rolling_overview:
            code = max(code, _backfill_overview(run_dir, scope="rolling", checkpoint=int(n)))
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
    model = os.environ.get("DSS_BENCH_MODEL", "").strip()
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
                       status_dir=live_dir, role="guide", provider=args.cloud_provider)
    judge = CloudModel(endpoint=cloud_endpoint, model=args.judge_model, max_tokens=8000,
                       reasoning_effort="low", fallback_reasoning=False, timeout=300,
                       max_empty_retries=args.cloud_empty_retries,
                       empty_cooldown=args.cloud_empty_cooldown,
                       status_dir=live_dir, role="judge", provider=args.cloud_provider)
    auditor = CloudModel(endpoint=cloud_endpoint, model=args.auditor_model, max_tokens=1600, timeout=300,
                         max_empty_retries=args.cloud_empty_retries,
                         empty_cooldown=args.cloud_empty_cooldown,
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
        plan_path = run_dir / "plan.json"
        if plan_path.exists():
            try:
                soak.plan = json.loads(plan_path.read_text(encoding="utf-8"))
            except Exception:
                soak.plan = {}
        print(f"[soak] resumed {len(soak.history)} prior exchanges")

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
            soak_log(f"  done: turn {turn} in {dt:.1f}s")
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

    # final overview: end-of-run judge pass over the whole chat
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

    soak_log(f"done. run_dir={run_dir}")
    soak_log(f"guide cloud: {soak.guide.stats()}")
    soak_log(f"judge cloud: {soak.judge.stats()}")
    soak_log(f"auditor cloud: {soak.auditor.stats() if soak.auditor else 'n/a'}")
    soak_log(f"local calls: {local.call_count} tokens p={local.prompt_tokens} c={local.completion_tokens}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
