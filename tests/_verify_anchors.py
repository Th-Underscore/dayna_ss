#!/usr/bin/env python3
"""CPU-only verification of the RAG fan-out anchor builder on REAL run data.

Builds a StoryContextRetriever over a real turn's state_snapshot (a complete
history_path) and exercises _build_retrieval_anchors against that turn's actual
rendered context + recent dialogue. Proves the anchors are (a) short (inside the
384-token window), (b) discriminative (not all-identical), and (c) boilerplate-free
(story markers present, static system-prompt text absent).

Read-only: loads the model offline from the local llama_index HF cache, forces
device='cpu'. Modifies nothing on disk.
"""
import json
import os
import sys
import time
from pathlib import Path

EXT = Path(__file__).resolve().parent  # extensions/dayna_ss/tests
DAYNA = EXT.parent
REPO_ROOT = DAYNA.parent.parent  # .../textgen
sys.path.insert(0, str(REPO_ROOT))

RUN = EXT / "runs" / "cyberpunk_thriller__4dbf1435"
TURN = "turn_084"
SNAP = RUN / TURN / "state_snapshot"
CONTEXT = RUN / TURN / "context.txt"

# Story markers we expect the run to contain (from the collapse analysis).
MARKERS = ["Handler", "Juno", "chip", "grey coat", "ledger", "Exchange"]
# Static system-prompt boilerplate that must NOT dominate an anchor.
BOILERPLATE = ["You are DAYNA", "writing style", "Do not break", "system prompt"]


def main():
    # --- load the model offline, CPU ---
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    torch.set_num_threads(os.cpu_count())
    from sentence_transformers import SentenceTransformer

    # Reuse bench's local-cache locator.
    from bench_embed_model import find_local_model
    snap = find_local_model("sentence-transformers", "all-mpnet-base-v2")
    if snap is None:
        print("model not cached; aborting")
        return 1
    model = SentenceTransformer(str(snap), device="cpu")
    cap = model.max_seq_length
    print(f"model cap (max_seq_length) = {cap} tokens\n")

    # --- build the retriever over the real snapshot ---
    # Pre-import heavy deps synchronously to avoid the background-importer deadlock
    # (same _warm_imports() as long_horizon_soak.py).
    import llama_index.core  # noqa: F401
    import llama_index.core.node_parser  # noqa: F401
    import llama_index.core.indices.loading  # noqa: F401
    import llama_index.core.settings  # noqa: F401
    import llama_index.core.schema  # noqa: F401
    import nltk  # noqa: F401
    import spacy  # noqa: F401
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import extensions.dayna_ss.rag.structured_rag.context_retriever as _cr
    from extensions.dayna_ss.rag.structured_rag.context_retriever import StoryContextRetriever

    # The anchor builder (_build_retrieval_anchors + helpers) is purely DATA-DRIVEN —
    # it never touches self.chunker or any embed model (verified). The ONLY network
    # side-effect in __init__ is constructing the real MessageChunker (which loads an
    # HuggingFaceEmbedding from the network). We are verifying anchor TEXT, not
    # retrieval, so stub the chunker out: a no-op class with a dead query_similar.
    # Encoding below is done directly with the local `model`, so nothing is lost.
    class _ChunkerStub:
        def __init__(self, *a, **k):
            self.nlp = None
            self.history_path = None
            self.storage_dir = None
        def query_similar(self, *a, **k):
            return []
    _cr.MessageChunker = _ChunkerStub  # patches the name used at line 166

    t0 = time.time()
    ret = StoryContextRetriever(SNAP, schema_classes=None, summarizer=None, rag_anchor_count=5)
    print(f"retriever built in {time.time() - t0:.1f}s")
    print(f"  events buckets: past={len(ret.events.get('past', {}))} "
          f"scenes={len(ret.events.get('scenes', {}))} "
          f"events={len(ret.events.get('events', {}))} "
          f"chapters={len(ret.events.get('chapters', {}))}")
    print(f"  characters: {len(ret.characters.get('entries', ret.characters))}")

    # --- real context + a recent-dialogue stand-in ---
    context = CONTEXT.read_text()
    print(f"\ncontext.txt = {len(context)} chars")
    # Recent dialogue: use the turn's own user message (the freshest real exchange).
    user_msg = (RUN / TURN / "user.txt").read_text() if (RUN / TURN / "user.txt").exists() else ""
    last_x = [user_msg] if user_msg.strip() else []
    print(f"recent dialogue (user.txt) = {len(user_msg)} chars")

    # --- build anchors ---
    anchors = ret._build_retrieval_anchors(context, last_x, ret.current_scene, 5)
    print(f"\n=== {len(anchors)} anchors built ===")
    for i, a in enumerate(anchors, 1):
        ntok = len(model.tokenizer([a], truncation=False)["input_ids"][0])
        flag = "OK " if ntok <= cap else "OVER"
        print(f"  [{i}] {flag} {ntok:3d} tok | {len(a):3d} ch | {a[:110]}")

    # --- metrics ---
    print("\n=== anchor metrics ===")
    lens = [len(model.tokenizer([a], truncation=False)["input_ids"][0]) for a in anchors]
    over = sum(1 for l in lens if l > cap)
    print(f"  within {cap}-tok window: {len(lens) - over}/{len(lens)}  (over: {over})")

    # discriminative: encode, pairwise cosine
    import numpy as np
    import statistics as st
    if len(anchors) >= 2:
        vecs = [model.encode(a, normalize_embeddings=True) for a in anchors]
        pairs = [float(np.dot(vecs[i], vecs[j])) for i in range(len(vecs)) for j in range(i + 1, len(vecs))]
        print(f"  mean pairwise anchor cosine: {st.mean(pairs):.5f}  (1.00000 == collapse; want < 0.95)")
    else:
        print("  (fewer than 2 anchors; discriminativity n/a)")

    # boilerplate-free: do anchors carry story markers and avoid boilerplate?
    joined = " || ".join(anchors)
    hit_markers = [m for m in MARKERS if m.lower() in joined.lower()]
    hit_boiler = [b for b in BOILERPLATE if b.lower() in joined.lower()]
    print(f"  story markers present in anchors: {hit_markers}")
    print(f"  boilerplate present in anchors:   {hit_boiler}")

    # compare to the OLD single-blob behavior: does the raw context's kept window carry markers?
    kept_ids = model.tokenizer([context], truncation=True, max_length=cap)["input_ids"][0]
    kept_text = model.tokenizer.decode(kept_ids)
    blob_markers = [m for m in MARKERS if m in kept_text]
    print(f"\n  (contrast) OLD single-blob kept-window markers: {blob_markers}")
    print(f"  (contrast) OLD single-blob kept-window boilerplate: "
          f"{[b for b in BOILERPLATE if b.lower() in kept_text.lower()]}")

    # --- verdict ---
    ok = (over == 0) and (not hit_boiler) and (len(hit_markers) > 0)
    if len(anchors) >= 2:
        ok = ok and (st.mean(pairs) < 0.95)
    print(f"\nVERDICT: {'GREEN' if ok else 'RED'} — anchors are "
          f"{'short + discriminative + boilerplate-free' if ok else 'NOT yet green'}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())