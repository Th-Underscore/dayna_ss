"""Quick test: 3 parallel 40k-token prompts (30k shared prefix) against the local server.

Verifies that llama.cpp can handle 3 concurrent sequences with a shared prefix
within the KV-cache budget. Uses LocalModel from long_horizon_soak for
auth/endpoint resolution.
"""

import sys
import time
import threading
import json

# Reuse the soak's LocalModel and key resolution
from long_horizon_soak import LocalModel, DEFAULT_LOCAL_BASE
from cloud_client import resolve_local_api_key, resolve_model_id


def build_prompt(shared_prefix: str, suffix: str) -> str:
    return shared_prefix + suffix


def main():
    # --- Config ---
    SHARED_TOKENS = 30_000
    UNIQUE_TOKENS = 10_000
    N_REQUESTS = 3
    MAX_TOKENS = 1  # prefill only

    # --- Resolve server ---
    base_url = DEFAULT_LOCAL_BASE
    api_key = resolve_local_api_key("textgen")
    model = resolve_model_id(base_url, api_key)
    if not model:
        print(f"ERROR: could not resolve model from {base_url}/models")
        sys.exit(1)
    print(f"Server: {base_url}")
    print(f"Model:  {model}")
    print(f"Key:    {api_key[:8]}... ({len(api_key)} chars)")
    print()

    # --- Build prompts ---
    # ~4 chars per token for repetitive English
    chars_per_token = 4
    shared_prefix = "The quick brown fox jumps over the lazy dog. " * (SHARED_TOKENS // 11)
    # Each suffix is unique
    suffixes = []
    for i in range(N_REQUESTS):
        suffixes.append(
            f" [Block {i}] Now analyze variant alpha-{i} in detail. " * (UNIQUE_TOKENS // 40)
        )

    print(f"Shared prefix: {len(shared_prefix)} chars (~{len(shared_prefix)//chars_per_token} tokens)")
    for i, s in enumerate(suffixes):
        print(f"Suffix {i}:      {len(s)} chars (~{len(s)//chars_per_token} tokens)")
    total_chars = len(shared_prefix) + len(suffixes[0])
    print(f"Total per req: {total_chars} chars (~{total_chars//chars_per_token} tokens)")
    print()

    # --- Create LocalModel ---
    local = LocalModel(
        base_url, api_key, model,
        max_tokens=MAX_TOKENS,
        temperature=0.0,
        top_p=1.0,
        top_k=1,
    )

    # --- Send in parallel ---
    results = []
    errors = []
    lock = threading.Lock()

    def send(idx: int, prompt: str):
        try:
            t0 = time.time()
            # Use the raw HTTP path directly (LocalModel.generate_with_streaming
            # expects engine state dict; we just need a plain completion)
            import urllib.request
            url = base_url + "/chat/completions"
            payload = json.dumps({
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": MAX_TOKENS,
                "temperature": 0.0,
            }).encode()
            req = urllib.request.Request(
                url, data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
            )
            with urllib.request.urlopen(req, timeout=600) as resp:
                body = json.loads(resp.read().decode())
            elapsed = time.time() - t0
            with lock:
                results.append((idx, elapsed, resp.status))
        except Exception as e:
            elapsed = time.time() - t0
            with lock:
                errors.append((idx, elapsed, str(e)[:300]))

    # --- Warm-up: send the shared prefix alone to populate KV cache ---
    print(f"Warm-up: sending shared prefix alone ({len(shared_prefix)} chars)...")
    t_warm = time.time()
    send(99, shared_prefix + " Continue.")
    warm_elapsed = time.time() - t_warm
    # Remove the warm-up from results so it doesn't pollute the report
    results = [(i, e, s) for i, e, s in results if i != 99]
    errors = [(i, e, err) for i, e, err in errors if i != 99]
    print(f"Warm-up done in {warm_elapsed:.1f}s")
    print()

    print(f"Sending {N_REQUESTS} parallel requests...")
    t0 = time.time()
    threads = []
    for i in range(N_REQUESTS):
        prompt = build_prompt(shared_prefix, suffixes[i])
        t = threading.Thread(target=send, args=(i, prompt))
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    total = time.time() - t0

    # --- Report ---
    print(f"\nTotal wall time: {total:.1f}s")
    print(f"{'Req':<5}{'Status':<10}{'Time':<10}{'Note'}")
    print("-" * 50)
    for idx, elapsed, status in sorted(results):
        print(f"{idx:<5}{status:<10}{elapsed:.1f}s")
    for idx, elapsed, err in sorted(errors):
        print(f"{idx:<5}{'ERROR':<10}{elapsed:.1f}s  {err}")

    if len(results) == N_REQUESTS:
        print(f"\nALL {N_REQUESTS} OK — parallel sequences fit.")
    elif results:
        print(f"\n{len(results)}/{N_REQUESTS} OK, {len(errors)} failed.")
    else:
        print(f"\nALL {N_REQUESTS} FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()