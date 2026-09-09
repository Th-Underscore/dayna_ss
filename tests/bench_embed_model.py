#!/usr/bin/env python3
"""Benchmark the structured-RAG embedding model against REAL run data.

Reads real persisted indexes (message_index/docstore.json + default__vector_store.json)
from existing soaks and answers the question "is all-mpnet-base-v2 as fast and accurate
as it needs to be for its job?" — measuring on real data, not synthetic.

It does NOT modify any on-disk store. All embedding work happens in-memory.

The model under test loads OFFLINE from the local llama_index HF cache
(~/.cache/llama_index/models--<name>/snapshots/<hash>/) and is forced to
device='cpu' (the GPUs are pinned by the user's LMDeploy/tgwui stack; a
default-device load would OOM).

Run:
  python3 bench_embed_model.py                     # default: mpnet on the default run
  python3 bench_embed_model.py --list-runs         # list candidate runs (index sizes)
  python3 bench_embed_model.py --run cyberpunk_thriller__4dbf1435
  python3 bench_embed_model.py --turns 42,62,82  # context blobs (query side) for recall
  python3 bench_embed_model.py --model all-MiniLM-L6-v2   # compare against a second model
  python3 bench_embed_model.py --no-gpu            # (always cpu here; kept for clarity)
"""
import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np

EXT = Path(__file__).resolve().parent  # extensions/dayna_ss/tests
RUNS = EXT / "runs"
HOME = Path.home()

# Models we know about: repo id -> (expected dim). Weights, if present, live in the
# llama_index HF cache at ~/.cache/llama_index/models--<repo>--<name>/...
KNOWN = {
    "all-mpnet-base-v2": 768,
    "all-MiniLM-L6-v2": 384,
}
DEFAULT_REPO = "sentence-transformers"


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------- #
# Model loading (offline, CPU)
# --------------------------------------------------------------------------- #
def find_local_model(repo: str, name: str) -> Path | None:
    """Locate a fully-cached SentenceTransformer weights dir under the llama_index
    HF cache. A model is usable iff its snapshot resolves model.safetensors (or
    pytorch_model.bin) to a real >100MB blob (a 110M-param fp32 model is ~440MB;
    a git-blob pointer file is tens of bytes)."""
    base = HOME / ".cache" / "llama_index" / f"models--{repo}--{name}"
    snap_dir = base / "snapshots"
    if not snap_dir.exists():
        return None
    for snap in snap_dir.iterdir():
        for wt in ("model.safetensors", "pytorch_model.bin"):
            f = snap / wt
            if f.is_symlink() or f.is_file():
                try:
                    real = f.resolve()
                    if real.exists() and real.stat().st_size > 100 * 1024 * 1024:
                        return snap
                except OSError:
                    continue
    return None


def load_st(repo: str, name: str, device: str = "cpu"):
    """Offline SentenceTransformer from the local cache. None if not cached."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    torch.set_num_threads(os.cpu_count())
    from sentence_transformers import SentenceTransformer
    local = find_local_model(repo, name)
    if local is None:
        log(f"  [skip] {name}: not fully cached locally (would trigger a download); skipping")
        return None
    t0 = time.time()
    model = SentenceTransformer(str(local), device=device)
    dim_fn = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    log(f"  loaded {name}: max_seq_length={model.max_seq_length} dim={dim_fn()} "
        f"load={time.time() - t0:.1f}s device={device}")
    return model


# --------------------------------------------------------------------------- #
# Corpus / query data (real, from disk)
# --------------------------------------------------------------------------- #
def _node_count(doc: dict) -> int:
    """Node count from a persisted docstore. llama_index persists a FLAT dict with
    slash-keys ('docstore/data', 'docstore/metadata'), not a nested structure — so
    we read the flat key, with a nested fallback for older serializations."""
    if "docstore/data" in doc:
        return len(doc["docstore/data"])
    return len(doc.get("docstore", {}).get("data", {}))


def _docstore_data(doc: dict) -> dict:
    """The node-id -> node mapping, flat or nested."""
    if "docstore/data" in doc:
        return doc["docstore/data"]
    return doc.get("docstore", {}).get("data", {})


def list_runs() -> list[tuple[int, Path]]:
    """(total_nodes, run_path) for every run with persisted message_index, sorted desc."""
    out = []
    for run in RUNS.iterdir():
        if not run.is_dir():
            continue
        total = 0
        for f in run.glob("sandbox/**/message_index/docstore.json"):
            try:
                total += _node_count(json.load(f.open()))
            except Exception:
                continue
        if total:
            out.append((total, run))
    out.sort(reverse=True)
    return out


def largest_index(run: Path) -> Path | None:
    """The most-complete (largest-node) message_index docstore under this run."""
    best, best_n = None, -1
    for f in run.glob("sandbox/**/message_index/docstore.json"):
        try:
            n = _node_count(json.load(f.open()))
        except Exception:
            continue
        if n > best_n:
            best, best_n = f, n
    return best


def load_corpus(docstore_path: Path) -> list[dict]:
    """Real sentence-level chunks: [{'id','text','meta'}]. The index stores only
    embeddings + metadata; the text lives in the docstore, so we read it from there."""
    data = _docstore_data(json.load(docstore_path.open()))
    chunks = []
    for nid, node in data.items():
        dd = node.get("__data__", {})
        text = dd.get("text") or ""
        if not text.strip():
            continue
        chunks.append({"id": nid, "text": text, "meta": dd.get("metadata", {})})
    return chunks


def turn_context(run: Path, turn: str) -> str | None:
    f = run / turn / "context.txt"
    return f.read_text() if f.exists() else None


# --------------------------------------------------------------------------- #
# Measurements
# --------------------------------------------------------------------------- #
def corpus_report(model, name: str, chunks: list[dict]) -> dict:
    """Corpus (index) side: is anything being truncated? (per-sentence nodes)"""
    import torch
    texts = [c["text"] for c in chunks]
    maxlen = model.max_seq_length
    # token lengths in one tokenizer pass (no model forward)
    enc = model.tokenizer(texts, truncation=False)
    lens = [len(x) for x in enc["input_ids"]]
    arr = np.array(lens)
    over1000 = int((arr > 1000).sum())
    over2000 = int((arr > 2000).sum())
    overcap = int((arr > maxlen).sum())
    # throughput: time N real forward encodes (the job's cost)
    n = min(len(texts), 500)
    t0 = time.time()
    model.encode(texts[:n], batch_size=32, show_progress_bar=False, normalize_embeddings=False)
    dt = time.time() - t0
    return {
        "model": name, "n_nodes": len(chunks), "cap_tokens": maxlen,
        "tok_min": int(arr.min()), "tok_p50": int(np.percentile(arr, 50)),
        "tok_p95": int(np.percentile(arr, 95)), "tok_p99": int(np.percentile(arr, 99)),
        "tok_max": int(arr.max()),
        "over_cap": overcap, "over_1000": over1000, "over_2000": over2000,
        "encode_n": n, "encode_ms": dt * 1000, "per_1k_ms": dt * 1000 / n * 1000,
    }


def query_report(model, name: str, texts: dict[str, str]) -> dict:
    """Query (retrieval) side: does truncation collapse the context blob to static
    boilerplate, and does that collapse make per-turn query vectors ~identical
    (i.e. the semantic ranking degenerates)?"""
    maxlen = model.max_seq_length
    q0 = next(iter(texts.values()))
    full = model.tokenizer([q0], truncation=False)["input_ids"][0]
    kept = model.tokenizer([q0], truncation=True, max_length=maxlen)["input_ids"][0]
    dropped = 100.0 * (1 - len(kept) / max(1, len(full)))
    # Where does real story content sit? measure token position of first story markers
    def tokpos(sub: str) -> int | None:
        c = q0.find(sub)
        if c < 0:
            return None
        return len(model.tokenizer([q0[:c]], truncation=False)["input_ids"][0])
    markers = {}
    for mk in ["You are DAYNA", "Handler", "Juno", "chip", "grey coat", "ledger", "Exchange"]:
        p = tokpos(mk)
        markers[mk] = p
    # Cross-turn collapse: encode each (truncated) blob, pairwise cosine
    vecs = {t: model.encode(x, normalize_embeddings=True) for t, x in texts.items()}
    ts = list(texts)
    pairs = [float(np.dot(vecs[a], vecs[b])) for i, a in enumerate(ts) for b in ts[i + 1:]]
    collapse = float(st.mean(pairs)) if pairs else float("nan")
    # Does the kept window contain ANY story marker?
    kept_ids = model.tokenizer([q0], truncation=True, max_length=maxlen)["input_ids"][0]
    kept_text = model.tokenizer.decode(kept_ids)
    story_in_window = [mk for mk, p in markers.items()
                       if p is not None and p < maxlen and mk in kept_text]
    return {
        "model": name, "n_queries": len(texts), "cap_tokens": maxlen,
        "q0_total_tokens": len(full), "q0_kept_tokens": len(kept),
        "q0_dropped_pct": dropped,
        "story_markers_in_blob": {mk: (p is not None) for mk, p in markers.items()},
        "story_marker_tokpos": markers,
        "story_markers_surviving_trunc": story_in_window,
        "mean_cross_turn_query_cosine": collapse,
        "max_seq_length_note": "384 (not 512) — this model truncates earlier than assumed",
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="run dir name under runs/ (default: largest index)")
    ap.add_argument("--list-runs", action="store_true")
    ap.add_argument("--model", default="all-mpnet-base-v2",
                    help="repo name (default all-mpnet-base-v2)")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--turns", default="", help="comma-separated turn_XXX names for the query side")
    ap.add_argument("--compare", action="store_true",
                    help="also run the known-alternative (MiniLM) if cached, for A/B")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    if args.list_runs:
        for n, r in list_runs():
            log(f"  {n:6d} nodes   {r.name}")
        return

    if args.run:
        run = RUNS / args.run
        if not run.exists():
            log(f"no such run: {args.run}"); return
    else:
        lr = list_runs()
        if not lr:
            log("no runs with persisted message_index found under runs/")
            return
        _, run = lr[0]
    log(f"run: {run.name}")

    doc = largest_index(run)
    if doc is None:
        log("no message_index docstore found in this run"); return
    log(f"corpus: {doc.parent}")
    chunks = load_corpus(doc)
    log(f"corpus nodes loaded: {len(chunks)}")
    if not chunks:
        return

    turns = [t.strip() for t in args.turns.split(",") if t.strip()] or None
    model = load_st(args.repo, args.model, args.device)
    if model is None:
        return

    # --- corpus side ---
    log("\n=== CORPUS (index) side — is anything truncated? ===")
    cr = corpus_report(model, args.model, chunks)
    log(f"  {cr['n_nodes']} nodes, cap={cr['cap_tokens']} tok")
    log(f"  token len  min={cr['tok_min']} p50={cr['tok_p50']} "
        f"p95={cr['tok_p95']} p99={cr['tok_p99']} max={cr['tok_max']}")
    log(f"  nodes over cap: {cr['over_cap']}  | over 1000 tok: {cr['over_1000']} "
        f"| over 2000 tok: {cr['over_2000']}")
    log(f"  throughput: {cr['encode_n']} encodes in {cr['encode_ms']:.0f} ms "
        f"= {cr['per_1k_ms']:.0f} ms per 1k (the job's per-write cost)")

    # --- query side ---
    qtexts = {}
    if turns:
        for t in turns:
            x = turn_context(run, t)
            if x:
                qtexts[t] = x
    else:
        # auto: sample 4 turns spanning the run
        for cand in sorted(run.glob("turn_*/context.txt")):
            qtexts[cand.parent.name] = cand.read_text()
        items = sorted(qtexts)
        if len(items) > 4:
            pick = [items[0], items[len(items) // 3], items[2 * len(items) // 3], items[-1]]
            qtexts = {t: qtexts[t] for t in pick}
    log(f"\n=== QUERY (retrieval) side — {len(qtexts)} context blobs ===")
    qr = query_report(model, args.model, qtexts)
    log(f"  blob {qr['q0_total_tokens']} tokens -> {qr['q0_kept_tokens']} kept "
        f"({qr['q0_dropped_pct']:.1f}% silently dropped)")
    log(f"  NOTE: this model's cap is {qr['cap_tokens']} tokens, not the 512 assumed")
    log("  story markers present in blob -> token pos:")
    for mk, present in qr["story_markers_in_blob"].items():
        p = qr["story_marker_tokpos"].get(mk)
        log(f"    {mk!r:18} present={present}  tokpos={p}")
    log(f"  markers SURVIVING truncation: {qr['story_markers_surviving_trunc']}")
    log(f"  mean cross-turn query-vector cosine: {qr['mean_cross_turn_query_cosine']:.5f}"
        f"   (1.00000 == total collapse)")

    # --- A/B compare ---
    if args.compare:
        alt = "all-MiniLM-L6-v2" if args.model == "all-mpnet-base-v2" else "all-mpnet-base-v2"
        am = load_st(args.repo, alt, args.device)
        if am is not None:
            log(f"\n=== A/B: {alt} on the SAME real data ===")
            acr = corpus_report(am, alt, chunks)
            log(f"  corpus over-cap: {acr['over_cap']} | per-1k ms: {acr['per_1k_ms']:.0f}")
            aqr = query_report(am, alt, qtexts)
            log(f"  query collapse cosine: {aqr['mean_cross_turn_query_cosine']:.5f} "
                f"| surviving markers: {aqr['story_markers_surviving_trunc']}")


if __name__ == "__main__":
    main()