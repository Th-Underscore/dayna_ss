"""Live JSON-compliance benchmark for the dayna_ss engine.

Measures how reliably a real (typically small, local) model obeys the engine's
output contracts, scored against the engine's OWN parsing rules — not a
hand-written JSON checker.

Prompt types exercised (built from the real example schema templates):
    gate_check   -- must answer YES/NO/UNCHANGED (prefix-stopped by the engine)
    branch_query -- must answer YES/NO/UNCHANGED
    field_update -- must produce a JSON update list, single object, or
                    NO_UPDATES_REQUIRED (parsed by `_parse_llm_field_updates`)
    new_entry    -- must produce a JSON list of names or NO/[]

Configuration via environment:
    DSS_BENCH_BASE_URL   OpenAI-compatible base (e.g. http://127.0.0.1:1234/v1)
    DSS_BENCH_API_KEY    API key (default: "not-needed")
    DSS_BENCH_MODEL      model id (default: "local-model")
    DSS_BENCH_RUNS       samples per prompt (default: 10)
    DSS_BENCH_PROMPTS    comma-separated prompt types (default: all)
    DSS_BENCH_THINKING   "1"/"true" to keep the model's thinking enabled
                         (default: "0" — thinking is disabled because modern
                         thinking models spend their whole budget on
                         reasoning_content, leaving content empty)

If the base URL is unset, the benchmark prints a skip notice and exits 0.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any

TEST_DIR = Path(__file__).parent
REPO_ROOT = TEST_DIR.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _prompt_suite() -> dict[str, list[str]]:
    """Build realistic prompts from the example schema templates + sample data."""
    from extensions.dayna_ss.utils.schema_parser import SchemaParser

    ext_root = REPO_ROOT / "extensions" / "dayna_ss"
    example_dir = ext_root / "user_data" / "example"
    schema_path = example_dir / "subjects_schema.json"
    parser = SchemaParser(schema_path)

    chars = json.loads((example_dir / "characters.json").read_text(encoding="utf-8"))
    sample = chars["entries"]["John Jones"]
    sample_json = json.dumps(sample, indent=2)[:2500]

    t = lambda name: parser.defaults.get("CharacterMap", {}).get(name) or parser.defaults.get("Character", {}).get(name)

    gate = t("gate_check_prompt_template")
    bq = t("branch_query_prompt_template")
    bu = t("branch_update_prompt_template")

    branch_ctx = f"Current context for 'characters.entries':\n{sample_json[:800]}"

    gate_prompt = f"{branch_ctx}\n\n{gate}".replace("{{ branch_name }}", "characters.entries")
    bq_prompt = f"{branch_ctx}\n\n{bq}".replace("{{ branch_name }}", "John Jones")
    bu_prompt = f"{branch_ctx}\n\n{bu}".replace("{{ branch_name }}", "John Jones")

    schema_snippet = parser.get_relevant_json_schema_definitions(parser.get_subject_class("characters"))
    bu_prompt = bu_prompt.replace("{{ schema_snippet }}", json.dumps(schema_snippet, indent=2)[:2000])
    bu_prompt = bu_prompt.replace("{{ example_json }}", json.dumps({"entries": {"John Jones": sample}}, indent=2)[:1500])

    new_query = t("new_entry_query_prompt_template")
    new_prompt = f"{branch_ctx}\n\n{new_query}".replace("{{ branch_name }}", "characters.entries")

    return {
        "gate_check": [gate_prompt],
        "branch_query": [bq_prompt],
        "field_update": [bu_prompt],
        "new_entry": [new_prompt],
    }


def _chat_complete(prompt: str, base_url: str, api_key: str, model: str) -> str:
    """Call an OpenAI-compatible chat/completions endpoint; return the text."""
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 400,
    }
    # Modern thinking models (Qwen3.5-era) spend their whole budget on
    # reasoning_content by default; the engine's contracts require plain text,
    # so disable thinking unless the operator explicitly opts in.
    if os.environ.get("DSS_BENCH_THINKING", "0") not in ("1", "true", "True"):
        payload["enable_thinking"] = False
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


class EngineScorer:
    """Scores model responses using the engine's own parsing contracts."""

    def __init__(self):
        from extensions.dayna_ss.agents.data_summarizer import DataSummarizer  # noqa: F401
        from extensions.dayna_ss.agents.summarizer import strip_response, strip_thinking  # noqa: F401

        self.strip_response = strip_response
        self.strip_thinking = strip_thinking
        # Reuse the engine's real field-update parser by binding it to a throwaway instance.
        self._ds = DataSummarizer.__new__(DataSummarizer)

    def gate_check(self, text: str) -> bool:
        t = self.strip_thinking(text).strip().upper()
        return any(t.startswith(k) for k in ("YES", "NO", "UNCHANGED"))

    def branch_query(self, text: str) -> bool:
        return self.gate_check(text)

    def field_update(self, text: str) -> bool:
        t = self.strip_thinking(text).strip()
        if t in ("NO_UPDATES_REQUIRED", "NO"):
            return True
        parsed = self._ds._parse_llm_field_updates(t, "bench")
        return len(parsed) > 0

    def new_entry(self, text: str) -> bool:
        t = self.strip_response(self.strip_thinking(text))
        if t.strip() in ("NO", "[]"):
            return True
        try:
            import jsonc
            val = jsonc.loads(t)
            return isinstance(val, list) and all(isinstance(n, str) for n in val)
        except Exception:
            return False


def run_live_benchmark() -> int:
    base_url = os.environ.get("DSS_BENCH_BASE_URL", "").strip()
    if not base_url:
        print("[live_benchmark] SKIP: DSS_BENCH_BASE_URL not set (e.g. http://127.0.0.1:1234/v1)")
        return 0
    api_key = os.environ.get("DSS_BENCH_API_KEY", "not-needed")
    model = os.environ.get("DSS_BENCH_MODEL", "local-model")
    runs = int(os.environ.get("DSS_BENCH_RUNS", "10"))
    want = {p.strip() for p in os.environ.get("DSS_BENCH_PROMPTS", "").split(",") if p.strip()}

    suite = _prompt_suite()
    scorer = EngineScorer()

    print(f"[live_benchmark] model={model} runs/prompt={runs} base={base_url}")
    overall_total = overall_ok = 0
    for ptype, prompts in suite.items():
        if want and ptype not in want:
            continue
        ok = 0
        total = 0
        scorer_fn = getattr(scorer, ptype)
        for i in range(runs):
            prompt = prompts[i % len(prompts)]
            try:
                resp = _chat_complete(prompt, base_url, api_key, model)
            except Exception as e:
                print(f"  [{ptype}] request {i}: ERROR {e}")
                total += 1
                continue
            total += 1
            if scorer_fn(resp):
                ok += 1
            else:
                snippet = " ".join(resp.split())[:80]
                print(f"  [{ptype}] request {i}: NONCOMPLIANT -> {snippet!r}")
        rate = (ok / total * 100.0) if total else 0.0
        print(f"[live_benchmark] {ptype}: {ok}/{total} compliant ({rate:.1f}%)")
        overall_ok += ok
        overall_total += total

    overall = (overall_ok / overall_total * 100.0) if overall_total else 0.0
    print(f"[live_benchmark] OVERALL: {overall_ok}/{overall_total} ({overall:.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(run_live_benchmark())
