"""Thin OpenAI-compatible client for the cloud guide/judge models (Route A).

Talks to OpenCode Go (https://opencode.ai/zen/go/v1) over standard Bearer auth,
exactly like ``live_benchmark.py`` already calls the local server. Key is read
from ``~/.local/share/opencode/auth.json`` (the ``opencode-go`` entry) or the
``OPENCODE_GO_API_KEY`` env var.

Behavioral notes discovered while wiring this up:

- Cloudflare blocks the default urllib User-Agent (error 1010) — a browser-ish
  UA header must be sent.
- ``ox-alpha`` (and other Go models) are *thinking* models, and the
  ``reasoning``/``reasoning_content`` field is now emitted at EVERY
  ``reasoning_effort`` level — including ``"none"`` (measured 2026-08-16: none
  produced 2-8K chars of reasoning and was the source of empty-content variance;
  ``"low"`` was content-consistent in probes — 6/6 draws vs 3/4 for none, with
  reasoning still only ~2K chars). ``reasoning_effort`` is preferred over
  ``enable_thinking: false`` (which the endpoint largely ignores for this model).
``max_tokens`` must leave headroom above the reasoning block so real content
   lands (guide/judge use 8000, overviews 16000); a reasoning-only draw is handled
   by a bounded cooldown re-roll loop in ``complete`` (default: 3 re-rolls, 300s
   between them), so a transient reasoning-only draw cannot kill an overnight run.
- The endpoint returns ``usage`` (prompt/completion tokens) and ``cost``.
"""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

# Endpoint migrated to OpenRouter (2026-08-24): the stealth/ox-alpha trio runs
# there; the legacy OpenCode Go endpoint is retired (rejects the OpenRouter key).
DEFAULT_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
#DEFAULT_ENDPOINT = "https://opencode.ai/zen/go/v1/chat/completions"  # retired
#DEFAULT_ENDPOINT = "https://opencode.ai/zen/v1/chat/completions"
#DEFAULT_ENDPOINT = "http://localhost:5015/v1/chat/completions"
DEFAULT_MODEL = "glm-5.3-flash"
_BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)


def _log(msg: str) -> None:
    print(f"[cloud] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def resolve_api_key() -> str:
    """Return an API key matching DEFAULT_ENDPOINT.

    The endpoint migrated to OpenRouter, so OpenRouter credentials are tried
    first (env OPENROUTER_API_KEY, then auth.json 'openrouter'); the legacy
    OpenCode Go sources (OPENCODE_GO_API_KEY / auth.json 'opencode-go') remain
    as fallbacks so the old endpoint still works if re-enabled.
    """
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key:
        return key
    auth_path = Path.home() / ".local" / "share" / "opencode" / "auth.json"
    if auth_path.exists():
        try:
            auth = json.loads(auth_path.read_text(encoding="utf-8"))
            for name in ("openrouter", "opencode-go"):
                entry = auth.get(name)
                if isinstance(entry, dict):
                    entry = entry.get("key", "")
                if isinstance(entry, str) and entry.strip():
                    return entry
        except Exception:
            pass
    return os.environ.get("OPENCODE_GO_API_KEY", "").strip()


# Endpoint routing (2026-08-25): ox-alpha is served from TWO pools — OpenRouter
# as `stealth/ox-alpha` and the re-enabled OpenCode Go endpoint as
# `ox-alpha-free`. Route by model shape: provider-prefixed ids ("/") go to
# OpenRouter; bare ids (ox-alpha-free, deepseek-v4-flash) go to zen/go. This
# spreads load across two free tiers instead of sharing one.
ZEN_GO_ENDPOINT = "https://opencode.ai/zen/go/v1/chat/completions"


def endpoint_for_model(model: str) -> str:
    return ZEN_GO_ENDPOINT if "/" not in (model or "") else DEFAULT_ENDPOINT


def normalize_endpoint(url: str) -> str:
    """Accept either a full chat-completions URL or a bare server base URL.

    ``http://localhost:9931`` and ``http://localhost:9931/v1`` both mean
    ``http://localhost:9931/v1/chat/completions`` here; anything already carrying
    a chat-completions path is passed through untouched.
    """
    url = (url or "").strip().rstrip("/")
    if not url:
        return url
    if url.endswith("/chat/completions"):
        return url
    parsed = urllib.parse.urlsplit(url)
    if not parsed.path or parsed.path == "/":
        return url + "/v1/chat/completions"
    return url + "/chat/completions"


def is_local_endpoint(endpoint: str) -> bool:
    host = (urllib.parse.urlsplit(endpoint).hostname or "").lower()
    return host in ("localhost", "127.0.0.1", "::1", "0.0.0.0")


def resolve_local_api_key(provider: str | None = None) -> str:
    """Key for a self-hosted OpenAI-compatible server (llama-server/TGWUI):
    DSS_BENCH_API_KEY, else auth.json 'llama'/'localhost'/'tgw'/'textgen', else 'not-needed'.

    ``provider`` selects which auth.json entry to read when several are present
    (e.g. llama.cpp at http://localhost:9931 vs TGWUI at http://127.0.0.1:5000);
    the explicit name wins over the default order.
    """
    key = os.environ.get("DSS_BENCH_API_KEY", "").strip()
    if key:
        return key
    auth_path = Path.home() / ".local" / "share" / "opencode" / "auth.json"
    if auth_path.exists():
        try:
            auth = json.loads(auth_path.read_text(encoding="utf-8"))
            names = [provider] if provider else []
            names += ["llama", "localhost", "tgw", "text-generation-webui"]
            for name in names:
                if not name:
                    continue
                entry = auth.get(name)
                if isinstance(entry, dict) and entry.get("key"):
                    return entry["key"]
                if isinstance(entry, str):
                    return entry
        except Exception:
            pass
    return "not-needed"


def resolve_model_id(base_url: str, api_key: str) -> str:
    """Ask a server what it actually serves (the source of truth) so a run can be
    launched with only --cloud-endpoint and still record the real model id.
    Returns "" when the server is unreachable/anonymous."""
    url = normalize_endpoint(base_url).replace("/chat/completions", "/models")
    try:
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        ids = [m.get("id") or m.get("model") for m in data.get("data", [])]
        ids = [i for i in ids if i]
        return ids[0] if ids else ""
    except Exception:
        return ""


def resolve_api_key_for_endpoint(endpoint: str, provider: str | None = None) -> str:
    """API key matching a specific endpoint (zen/go must NOT get the OpenRouter
    key — that mismatch was the 401 cause when the legacy endpoint retired)."""
    if is_local_endpoint(endpoint):
        return resolve_local_api_key(provider)
    if endpoint == ZEN_GO_ENDPOINT:
        key = os.environ.get("OPENCODE_GO_API_KEY", "").strip()
        if key:
            return key
        auth_path = Path.home() / ".local" / "share" / "opencode" / "auth.json"
        if auth_path.exists():
            try:
                auth = json.loads(auth_path.read_text(encoding="utf-8"))
                for name in ("opencode-go", "opencode-zen"):
                    entry = auth.get(name)
                    if isinstance(entry, dict):
                        entry = entry.get("key", "")
                    if isinstance(entry, str) and entry.strip():
                        return entry
            except Exception:
                pass
        # Last resort: any key at all (some proxies accept anything).
        return resolve_api_key()
    if endpoint == DEFAULT_ENDPOINT:
        return resolve_api_key()
    return os.environ.get("OPENCODE_GO_API_KEY", "").strip() or resolve_api_key()


class CloudError(Exception):
    pass


class CloudModel:
    """OpenAI-compatible chat client with retry/backoff + token accounting."""

    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 1024,
        temperature: float = 0.8,
        max_retries: int = 4,
        base_delay: float = 90.0,
        timeout: int = 120,
        thinking: bool = False,
        reasoning_effort: str | None = None,
        fallback_reasoning: bool = True,
        max_retry_budget: int = 32768,
        max_empty_retries: int = 3,
        empty_cooldown: float = 300.0,
        status_dir: str | None = None,
        role: str = "cloud",
        # Sampler passthrough (all optional: None = not sent, so hosted providers
        # keep their own defaults byte-identically). Self-hosted endpoints
        # (llama-server) accept the llama.cpp sampler set — top_k/min_p/
        # repetition_penalty/preserve_thinking — which is how a run reproduces a
        # sampler profile like opencode.jsonc's qwen3.8-27b block.
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        repetition_penalty: float | None = None,
        presence_penalty: float | None = None,
        preserve_thinking: bool | None = None,
        enable_thinking: bool | None = None,
        provider: str | None = None,
    ):
        # Endpoint/key auto-routing: explicit endpoint wins; otherwise route by
        # model shape (see endpoint_for_model) and pick the matching key.
        if endpoint is None:
            endpoint = endpoint_for_model(model)
        self.endpoint = normalize_endpoint(endpoint)
        if api_key is None:
            api_key = resolve_api_key_for_endpoint(self.endpoint, provider)
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.timeout = timeout
        self.thinking = thinking
        self.reasoning_effort = reasoning_effort
        self.fallback_reasoning = fallback_reasoning
        self.max_retry_budget = max_retry_budget
        self.max_empty_retries = max_empty_retries
        self.empty_cooldown = empty_cooldown
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.repetition_penalty = repetition_penalty
        self.presence_penalty = presence_penalty
        self.preserve_thinking = preserve_thinking
        self.enable_thinking = enable_thinking
        # Explicit passthrough wins over the thinking/effort defaults below.
        self._sampler_passthrough = {
            k: v for k, v in (
                ("top_p", top_p), ("top_k", top_k), ("min_p", min_p),
                ("repetition_penalty", repetition_penalty),
                ("presence_penalty", presence_penalty),
                ("preserve_thinking", preserve_thinking),
                ("enable_thinking", enable_thinking),
            ) if v is not None
        }
        # Live progress reporting (soak only): when status_dir is set, calls are
        # STREAMED and a small JSON file in status_dir tracks the in-flight call
        # (role, elapsed, chars streamed, phase) for the soak dashboard; a sparse
        # heartbeat line also lands on stdout every _HB_EVERY seconds so a slow
        # cloud model is distinguishable from a hang in the tee'd log.
        self.status_dir = status_dir
        self.role = role
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0
        self.requests_log: list[dict] = []

    # ------------------------------------------------------------ transport

    _HB_EVERY = 60.0          # stdout heartbeat cadence while generating
    _PROGRESS_MIN_GAP = 1.0   # min seconds between live-status file writes

    def _status_path(self) -> Path | None:
        if not self.status_dir:
            return None
        return Path(self.status_dir) / f"cloud_{self.role}_{os.getpid()}.json"

    def _heartbeat(self, text: str) -> None:
        now = time.time()
        if now - getattr(self, "_hb_last", 0.0) >= self._HB_EVERY:
            self._hb_last = now
            print(f"[cloud {self.role}] {text}", flush=True)

    def _progress_write(self, state: dict) -> None:
        """Throttled write of the in-flight call state for the dashboard."""
        path = self._status_path()
        if path is None:
            return
        now = time.time()
        if now - state.get("_last_write", 0.0) < self._PROGRESS_MIN_GAP and not state.get("force"):
            return
        state["_last_write"] = now
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            out = {k: v for k, v in state.items() if not k.startswith("_")}
            out["updated_at"] = now
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(out), encoding="utf-8")
            os.replace(tmp, path)
        except Exception:
            pass  # progress reporting must never kill a cloud call

    def _progress_end(self) -> None:
        path = self._status_path()
        if path is None:
            return
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass

    def _post(self, payload: dict) -> dict:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": _BROWSER_UA,
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _post_stream(self, payload: dict) -> dict:
        """Streaming variant of ``_post`` with the same normalized return shape.

        Accumulates SSE deltas into ``choices[0].message`` (content +
        reasoning_content), captures the final usage chunk when the endpoint
        sends one, and feeds the live-status file + heartbeat while generating.
        Raises the same exceptions as ``_post`` so retry semantics are shared.
        """
        payload = dict(payload)
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": _BROWSER_UA,
                "Accept": "text/event-stream",
            },
        )
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        usage: dict = {}
        finish_reason: str | None = None
        started = time.time()
        state: dict = {"role": self.role, "model": self.model, "phase": "generating",
                       "started_at": started}
        self._progress_write({**state, "force": True})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue  # SSE comments / keep-alives (": OPENROUTER PROCESSING")
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except Exception:
                    continue
                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                c = delta.get("content")
                r = delta.get("reasoning_content") or delta.get("reasoning")
                if c:
                    content_parts.append(c)
                if r:
                    reasoning_parts.append(r)
                if choices[0].get("finish_reason"):
                    finish_reason = choices[0]["finish_reason"]
                elapsed = time.time() - started
                n_content = sum(len(p) for p in content_parts)
                n_reason = sum(len(p) for p in reasoning_parts)
                joined = "".join(content_parts)
                state.update({
                    "elapsed_s": round(elapsed, 1),
                    "content_chars": n_content,
                    "reasoning_chars": n_reason,
                    "head": joined[:160],
                    "tail": joined[-160:],
                })
                self._progress_write(state)
                self._heartbeat(f"{self.role}: generating ... {elapsed:.0f}s, "
                                f"{n_content + n_reason} chars streamed")
        data = {
            "choices": [{
                "message": {
                    "content": "".join(content_parts),
                    "reasoning_content": "".join(reasoning_parts),
                },
                "finish_reason": finish_reason,
            }],
            "usage": usage,
            "streamed": True,
        }
        # Some providers omit stream usage; fall back to a char estimate so
        # accounting keeps working (prompt_tokens stays unknown -> 0).
        if not usage:
            est = (len("".join(content_parts)) + len("".join(reasoning_parts))) // 4
            data["usage"] = {"prompt_tokens": 0, "completion_tokens": est,
                             "total_tokens": est}
        return data

    def _retry_post(self, payload: dict) -> dict:
        post = self._post_stream if self.status_dir else self._post
        delay = self.base_delay
        for attempt in range(self.max_retries + 1):
            try:
                return post(payload)
            except urllib.error.HTTPError as e:
                if e.code in (429,) or e.code >= 500:
                    if attempt >= self.max_retries:
                        raise CloudError(f"HTTP {e.code} after {self.max_retries} retries: {e}") from e
                    self._progress_write({"role": self.role, "model": self.model,
                                          "phase": "backoff", "force": True,
                                          "note": f"HTTP {e.code}; retrying in {delay:.0f}s"})
                    self._heartbeat(f"{self.role}: HTTP {e.code}; backing off {delay:.0f}s "
                                    f"(attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise CloudError(
                    f"HTTP {e.code}: {e}"
                    + (" (401/403: invalid or missing API key for this endpoint; "
                       "4xx otherwise often means the prompt exceeds this model's "
                       "context window or the model id is wrong)"
                       if e.code in (401, 403) else
                       " (4xx often means the prompt exceeds this model's context "
                       "window or the model id is wrong — call sites retry with cooldown)"
                       if 400 <= e.code < 500 else "")
                ) from e
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                if attempt >= self.max_retries:
                    raise CloudError(f"network error after {self.max_retries} retries: {e}") from e
                self._progress_write({"role": self.role, "model": self.model,
                                      "phase": "backoff", "force": True,
                                      "note": f"network error; retrying in {delay:.0f}s"})
                time.sleep(delay)
                delay *= 2
        raise CloudError("unreachable")

    # --------------------------------------------------------------- public

    def complete(self, messages: list[dict], max_tokens: int | None = None,
                 temperature: float | None = None, json_mode: bool = False,
                 reasoning_effort: str | None = None) -> str:
        """One chat completion; returns the assistant's text content."""
        if not self.api_key:
            raise CloudError("no API key: set OPENROUTER_API_KEY or the auth.json 'openrouter' entry")
        budget = max_tokens or self.max_tokens
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": budget,
            "temperature": self.temperature if temperature is None else temperature,
        }
        effort = reasoning_effort if reasoning_effort is not None else self.reasoning_effort
        if effort is not None:
            # "none"|"low"|"medium"|"high" — supported by the Go endpoint; "none"
            # suppresses reasoning entirely, "low" keeps a small budget.
            payload["reasoning_effort"] = effort
        elif not self.thinking:
            payload["enable_thinking"] = False
        # Explicit sampler passthrough (set via CloudModel kwargs) overrides the
        # defaults above — this is how a self-hosted llama-server endpoint gets
        # the opencode.jsonc sampler profile (top_p/top_k/min_p/rep_penalty/
        # preserve_thinking). Only explicitly-set keys are sent.
        payload.update(self._sampler_passthrough)
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            data = self._retry_post(payload)
        except CloudError as e:
            # Some providers reject response_format outright (ling-3.0-flash-fin:
            # HTTP 400 on EVERY prompt size), which silently broke the judge and
            # the whole auditor every night. Fall back to plain text and let the
            # caller's own JSON parser decide — a bad parse is recoverable, a
            # hard 400 is not. The downgraded attempt does NOT get the empty-content
            # rollover loop either: a provider that rejects the format is also the
            # one most likely to draw empty content, and 3 x 300s stalls a run.
            if json_mode and "400" in str(e) and "response_format" not in str(e):
                _log(f"{self.model}: response_format rejected ({str(e)[:60]}); "
                     f"retrying without it")
                payload.pop("response_format", None)
                data = self._retry_post(payload)
            else:
                self._progress_end()
                raise
        self.calls += 1
        usage = data.get("usage", {})
        self.prompt_tokens += int(usage.get("prompt_tokens", 0))
        self.completion_tokens += int(usage.get("completion_tokens", 0))
        self.requests_log.append({
            "model": self.model,
            "messages": messages,
            "response": data,
            "ts": time.time(),
        })
        msg = data["choices"][0]["message"]
        content = msg.get("content")
        # deepseek-v4-flash still spends budget on reasoning_content/reasoning even
        # with thinking disabled and reasoning_effort="none"; on large prompts the
        # reasoning block alone can eat the whole output budget, leaving content
        # empty (finish=length). Handle this by re-rolling after a cooldown so a
        # transient reasoning-only draw cannot kill the run. The retry budget is
        # tiered: SMALL budgets get doubled (truncated JSON is their failure
        # mode — more room lets the real content land), LARGE budgets just get a
        # small headroom bump (doubling them invites proportionally longer
        # reasoning with no content and costs minutes per attempt).
        empty_hits = 0
        budget_bumped = False
        while (not content or not content.strip()) and not self.thinking and empty_hits < self.max_empty_retries:
            empty_hits += 1
            if not budget_bumped:
                if budget < 4096:
                    payload["max_tokens"] = min(budget * 2, self.max_retry_budget)
                else:
                    payload["max_tokens"] = min(budget + 2048, self.max_retry_budget)
                budget_bumped = True
            _log(f"{self.model}: empty-content draw {empty_hits}/{self.max_empty_retries} "
                 f"(budget {payload['max_tokens']}); cooling down "
                 f"{self.empty_cooldown:.0f}s before re-rolling ...")
            self._progress_write({"role": self.role, "model": self.model,
                                  "phase": "cooldown", "force": True,
                                  "note": f"empty draw {empty_hits}; re-roll in "
                                          f"{self.empty_cooldown:.0f}s"})
            self._heartbeat(f"{self.role}: empty draw {empty_hits}/{self.max_empty_retries}; "
                            f"cooling down {self.empty_cooldown:.0f}s")
            time.sleep(self.empty_cooldown)
            try:
                data = self._retry_post(payload)
            except Exception:
                self._progress_end()
                raise
            self.calls += 1
            usage = data.get("usage", {})
            self.prompt_tokens += int(usage.get("prompt_tokens", 0))
            self.completion_tokens += int(usage.get("completion_tokens", 0))
            self.requests_log.append({"model": self.model, "messages": messages,
                                      "response": data, "ts": time.time()})
            msg = data["choices"][0]["message"]
            content = msg.get("content")
        if not content:
            # Thinking left on: surface the reasoning as the fallback text.
            # With fallback_reasoning=False (the story guide), an empty content
            # is a FAILURE — the reasoning block is planning, never a usable turn.
            if self.fallback_reasoning:
                content = msg.get("reasoning_content")
        self._progress_end()
        return content or ""

    def json_complete(self, messages: list[dict], **kwargs) -> dict:
        """Like ``complete`` but parses the response as JSON (judge output).

        On a parse failure the request is retried once with a doubled token
        budget (truncated JSON is the common cause — deepseek-v4-flash spends
        budget on reasoning even with thinking disabled, cutting the real
        content mid-document). Returns the parsed dict or raises CloudError.
        """
        text, err = self._complete_and_parse(messages, kwargs)
        if text is not None:
            return text
        budget = kwargs.get("max_tokens") or self.max_tokens
        if budget < 8192:
            kwargs["max_tokens"] = min(budget * 2, self.max_retry_budget)
            text, err = self._complete_and_parse(messages, kwargs)
            if text is not None:
                return text
        if err:
            # Final attempt with the actual parse error fed back to the model —
            # re-sending the identical prompt is rarely the right retry.
            feedback = [
                *messages,
                {
                    "role": "user",
                    "content": (
                        "Your previous output could not be parsed as JSON. "
                        f"Parse error: {err}\n"
                        "Return ONLY a single valid JSON object (no fenced block, "
                        "no surrounding text), exactly matching the schema asked for above."
                    ),
                },
            ]
            text, _ = self._complete_and_parse(feedback, kwargs)
            if text is not None:
                return text
        text = self.complete(messages, **kwargs).strip()
        raise CloudError(f"judge output was not JSON:\n{text[:400]}")

    def _complete_and_parse(self, messages: list[dict], kwargs: dict) -> tuple[dict | None, str | None]:
        """One attempt: complete + fence-strip + balanced-brace fallback.

        Returns (parsed dict | None, parse-error message | None).
        """
        text = self.complete(messages, **kwargs)
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
        try:
            return json.loads(text), None
        except Exception as e:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1]), None
                except Exception:
                    pass
            return None, str(e)

    def stats(self) -> dict:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "model": self.model,
        }


if __name__ == "__main__":
    c = CloudModel()
    print(c.stats())
    out = c.complete([{"role": "user", "content": "Reply with exactly the word PONG."}], max_tokens=50)
    print(f"reply: {out!r}")
    print(c.stats())
