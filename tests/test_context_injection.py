#!/usr/bin/env python3
"""
test_context_injection.py

Tests entity relevance aggregation and instruction generation on an existing
soak run's turn. Two modes of operation:

  --offline (default): Analyze saved turn data, reconstruct all 6 injection
    modes, show diffs. No model calls.

  --live: Actually call the local model to generate a new reply for each
    injection mode, enabling direct A/B comparison of how different
    context/instruction strategies affect the output.

Matrix (all combinations of instr/advisor/reasoning):
  instr    advisor    reasoning    description
  ---------------------------------------------------------------------------
  0        0          0            No instr (baseline)
  0        0          1            No instr w/ reasoning (show what WOULD be generated)
  0        1          0            Advisor-only: inject [[...]] as steering requirements
  0        1          1            Advisor-only w/ reasoning
  1        0          0            Director on (baseline with instructions)
  1        1          0            Director + advisor: [[...]] injected as steering

Invalid: instr=1 + reasoning=1 (reasoning only applies without director)

Entity relevance:
  a. Graph-only (existing retrieval)
  b. Graph + cross-encoder rerank (optional, --rerank)

Cross-encoder reranking (--rerank): Extracts entity mentions from the saved
context, scores each against the current scene text (user input) using a
cross-encoder, and reports which entities are most/least relevant to the
immediate scene. This validates the second-pass relevance architecture:
graph retrieval generates the candidate set, cross-encoder scores semantic
relevance for the current moment.

Usage:
  # Offline analysis (no model calls)
  python test_context_injection.py --run greek_mythology_retelling__2e7575de --turn 141

  # Live generation: generate a new reply for each mode and compare
  python test_context_injection.py --run greek_mythology_retelling__2e7575de --turn 141 --live

  # With cross-encoder relevance scoring
  python test_context_injection.py --run greek_mythology_retelling__2e7575de --turn 141 --rerank

  # Inject synthetic advisor instructions
  python test_context_injection.py --run greek_mythology_retelling__2e7575de --turn 141 --inject "Mention the sword" "Describe the anvil"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

# Add parent so we can import from extensions/dayna_ss
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

# Add tests dir so we can import LongHorizonSoakTest and related
TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR))

# Import model classes (not duplicated — single source of truth)
from long_horizon_soak import LocalModel, resolve_local_key, soak_log

# Real DSS formatting engine — renders entities exactly as the summarizer does
from extensions.dayna_ss.agents.formatted_data import FormattedData
from extensions.dayna_ss.utils.text_parsing import strip_thinking


def parse_inline_instructions(text: str) -> tuple[str, list[str]]:
    """
    Parse [[...]] authorial directive blocks from user narration.

    Returns (cleaned_narrative, [steering_requirements]).
    """
    instructions = []

    def replace_match(match):
        directive = match.group(1).strip()
        instructions.append(directive)
        return ""

    clean = re.sub(r"\[\[(.+?)\]\]", replace_match, text)
    # Clean up extra whitespace left by directive removal
    clean = re.sub(r"[ \t]{2,}", " ", clean).strip()
    return clean, instructions


class CrossEncoderReranker:
    """Scores entity relevance to a scene using a sentence-transformers cross-encoder.

    Primary model: cross-encoder/ms-marco-MiniLM-L6-v2 (~83ms for 150 entities on GPU)
    Alternatives: ms-marco-MiniLM-L2-v2 (faster, ~17ms), bge-reranker-v2-m3 (higher quality, ~357ms)
    """

    MODELS = {
        "mini": "cross-encoder/ms-marco-MiniLM-L2-v2",
        "mid": "cross-encoder/ms-marco-MiniLM-L6-v2",  # default: quality/latency balance
        "bge": "BAAI/bge-reranker-v2-m3",
    }

    def __init__(self, model_name: str = "mid", device: str = "cpu"):
        # Default to CPU: GPUs are occupied by the LMDeploy LLM server.
        # A MiniLM cross-encoder is small (~80MB) and fast enough on CPU
        # for this use case (~100-300ms for 150 entity pairs).
        from sentence_transformers import CrossEncoder
        if device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    # Check if there's actual free memory (>1GB)
                    free, _ = torch.cuda.mem_get_info()
                    if free > 1073741824:  # 1GB free
                        device = "cuda"
                    else:
                        print("[rerank] GPU detected but <1GB free; using CPU")
                        device = "cpu"
                else:
                    device = "cpu"
            except ImportError:
                device = "cpu"
        resolved = self.MODELS.get(model_name, model_name)
        print(f"[rerank] loading cross-encoder: {resolved} (device={device})")
        t0 = time.time()
        self.model = CrossEncoder(resolved, device=device)
        print(f"[rerank] loaded in {time.time() - t0:.1f}s")

    def score(self, scene_text: str, entities: list[dict]) -> list[dict]:
        """Score each entity against the current scene.

        Args:
            scene_text: The current scene text (user input) to score relevance against
            entities: List of dicts with 'name' and optional 'description'/'recent' keys

        Returns: List of dicts with entity fields + 'relevance_score' (0-1 scale),
                 sorted highest first.
        """
        if not entities:
            return []

        pairs = []
        for ent in entities:
            # Build the candidate text: name + key description fragments
            candidate = ent["name"]
            if ent.get("description"):
                candidate += f": {ent['description'][:100]}"
            pairs.append((scene_text, candidate))

        t0 = time.time()
        scores = self.model.predict(pairs)
        elapsed = time.time() - t0
        print(f"[rerank] scored {len(entities)} entities in {elapsed:.1f}s")

        results = []
        for ent, score in zip(entities, scores):
            result = dict(ent)
            result["relevance_score"] = float(score)
            results.append(result)

        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results


_ENTITY_TYPES = [
    ("characters.json", "characters", "character"),
    ("elements.json", "elements", "element"),
    ("groups.json", "groups", "group"),
    ("events.json", "event", "event"),
]


def extract_entities_from_state(
    run_dir: Path,
    turn: int,
    templates_path: Path | None = None,
) -> list[dict]:
    """Load entities from the state_snapshot directory for a given turn.

    Each entity is rendered through the REAL DSS formatting pipeline
    (FormattedData.format_retrieval_data + the run's format_templates.json),
    exactly as the summarizer builds context. This produces the precise text
    the cross-encoder should score — not a naive list-join of raw fields.

    Returns a flat list of entity dicts: {name, type, description}, where
    description is the marker-stripped, template-rendered text for that entity.
    """
    snap_dir = run_dir / f"turn_{turn}" / "state_snapshot"

    # Default to the run's own templates; allow a custom override (--templates).
    if templates_path is None:
        templates_path = snap_dir / "format_templates.json"

    session_id = f"{run_dir.name}_{turn}"
    FormattedData.set_session_templates_path(session_id, templates_path)

    # Minimal stub satisfying the attributes format_retrieval_data touches:
    # context_cache.schema_parser (None = no schema defaults) and
    # context_cache.history_path (drives session_id for template lookup).
    class _StubCache:
        schema_parser = None
        history_path = session_id

    entities: list[dict] = []
    for fname, data_type, ent_type in _ENTITY_TYPES:
        path = snap_dir / fname
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            print(f"Warning: failed to load {fname}: {e}", file=sys.stderr)
            continue

        for name, ent_data in data.get("entries", {}).items():
            # Render this single entity through the real template pipeline.
            rendered = FormattedData.format_retrieval_data(
                {"entries": {name: ent_data}},
                data_type,
                prefix="",
                context_cache=_StubCache(),
            )
            # Strip the path markers (<<<<<< ...rest-of-line) the template emits —
            # they are bookkeeping, not part of the semantic text the encoder sees.
            description = re.sub(r" <<<<<<<<<<<< [^\n]*", "", rendered).strip()
            entities.append({
                "name": name,
                "type": ent_type,
                "description": description,
            })

    return entities


class ContextInjectionTest:
    def __init__(self, run_name: str, turn: int, rerank: bool = False,
                 inject_instructions: list[str] | None = None, live: bool = False,
                 rerank_model: str = "mid",
                 thinking: bool = False, reasoning_effort: str | None = None):
        self.run_dir = Path(__file__).resolve().parent / "runs" / run_name
        self.turn = turn
        self.rerank = rerank
        self.live = live
        self.injected_instructions = inject_instructions or []
        self.reranker = None

        # Load turn data
        turn_dir = self.run_dir / f"turn_{turn:03d}"
        self.result = json.loads((turn_dir / "result.json").read_text())
        self.user_input = self.result["user_input"]
        self.reply = self.result["reply"]
        self.instr_prompt = self.result["instr_prompt"]
        self.saved_context = self.result.get("context", "")

        # Load prior exchanges (up to this turn)
        self.history = self._load_history()
        print(f"Loaded {len(self.history)} history entries for turn {turn}")

        # Build user input with advisor instructions if any
        if self.injected_instructions:
            self.user_input_with_advisor = self.user_input
            for instr in self.injected_instructions:
                self.user_input_with_advisor += f" [[{instr}]]"
            # Parse: narrative stays clean, instructions are extracted
            self.clean_input, self.parsed_instructions = parse_inline_instructions(
                self.user_input_with_advisor
            )
            # Ensure narrative is the original (parsing should have removed the [[...]] blocks)
            self.clean_input = self.user_input
        else:
            self.user_input_with_advisor = self.user_input
            self.clean_input, self.parsed_instructions = parse_inline_instructions(self.user_input)

        if self.parsed_instructions:
            print(f"Found {len(self.parsed_instructions)} inline instruction(s):")
            for i, instr in enumerate(self.parsed_instructions):
                print(f"  [{i}] {instr[:100]}")
        else:
            print("No inline instructions in user input")

        # Initialize cross-encoder if reranking requested
        if self.rerank:
            self.reranker = CrossEncoderReranker(model_name=rerank_model)

        # Initialize local model + DSS engine if live generation requested
        self.local_model = None
        self.summarizer = None
        if self.live:
            if thinking:
                os.environ["DSS_BENCH_THINKING"] = "1"
            base_url = os.environ.get("DSS_BENCH_BASE_URL", "http://127.0.0.1:5001/v1")
            api_key = resolve_local_key()
            model_id = os.environ.get("DSS_BENCH_MODEL", "local-model")
            print(f"[live] local model: {model_id} @ {base_url}")
            if thinking:
                print(f"[live] thinking=on reasoning_effort={reasoning_effort or 'default'}")
            self.local_model = LocalModel(
                base_url=base_url,
                api_key=api_key,
                model=model_id,
                max_tokens=16384,
                reasoning_effort=reasoning_effort,
            )
            # Configure runtime + Summarizer so generate_instr_prompt can run
            # the full DSS pipeline (retrieval, graph traversal, instruction gen).
            self._setup_engine()

    def _setup_engine(self) -> None:
        """Configure the DSS runtime + Summarizer so generate_instr_prompt can run.

        Points the runtime at the soak run's sandbox directory (which contains
        the session state, schema, and format templates) and creates a
        Summarizer instance — the same setup the soak does in
        _make_summarizer / _configure_runtime.
        """
        from extensions.dayna_ss.runtime import runtime
        from extensions.dayna_ss.agents.summarizer import Summarizer

        # RAG (MessageChunker) requires llama_index which deadlocks on this WSL
        # environment. The context_retriever now handles this gracefully — the
        # MessageChunker constructor is wrapped in try/except, so RAG is simply
        # disabled while entity-graph traversal remains fully functional.

        # Load the manifest to get the seed and spreadsheet path
        manifest_path = self.run_dir / "manifest.json"
        if not manifest_path.exists():
            raise RuntimeError(f"No manifest.json in {self.run_dir}")
        manifest = json.loads(manifest_path.read_text())
        self._seed = manifest["args"].get("seed", 0)
        self._spreadsheet_path = manifest["args"].get("spreadsheet", "")
        self._unique_id = f"soak_{self._seed}"

        # Load the spreadsheet (path in manifest is relative to tests/)
        tests_dir = Path(__file__).resolve().parent  # extensions/dayna_ss/tests/
        spreadsheet_file = tests_dir / self._spreadsheet_path
        if not spreadsheet_file.exists():
            raise RuntimeError(f"Spreadsheet not found: {spreadsheet_file}")
        self._spreadsheet = json.loads(spreadsheet_file.read_text())

        # Configure runtime to point at the soak's sandbox
        sandbox = self.run_dir / "sandbox"
        if not sandbox.exists():
            raise RuntimeError(f"Sandbox not found: {sandbox}")
        current_character = {"name": f"soak_{self._spreadsheet['id']}"}
        runtime.configure(
            model_provider=lambda: self.local_model,
            stop_provider=lambda: False,
            prompt_builder=lambda prompt, state, **kw: prompt,
            encoder=lambda text, add_bos_token=True: text,
            persistent_ui_state_provider=lambda: {},
            current_character_provider=lambda: current_character["name"],
            settings_provider=lambda: {},
            update_config_fn=lambda state: (current_character.__setitem__(
                "name", state.get("character_menu", current_character["name"])) or False),
            register_tool_executors_fn=lambda _: None,
            extension_dir=sandbox,
        )

        # Copy the turn-specific format_templates.json into the sandbox history
        # path so the engine's session-local template lookup finds it (including
        # any test-specific entries like current_scene2).
        import shutil, glob as _glob
        snap_templates = self.run_dir / f"turn_{self.turn}" / "state_snapshot" / "format_templates.json"
        # The history dir uses the spreadsheet id, not the numeric seed.
        # The engine's history_path is nested: soak_<id>/<seed>/<hash>/
        history_base = sandbox / "user_data" / "history" / f"soak_{self._spreadsheet['id']}"
        if snap_templates.exists() and history_base.exists():
            # Find the nested hash dir(s) under the history base
            nested_dirs = list(history_base.glob("*/*"))  # seed/hash
            if not nested_dirs:
                nested_dirs = [history_base]  # fallback: no nesting
            for nd in nested_dirs:
                if nd.is_dir():
                    shutil.copy2(snap_templates, nd / "format_templates.json")
                    print(f"[engine] Copied turn_{self.turn} format_templates.json -> {nd}")

        self.summarizer = Summarizer()
        # Match the production soak's config so the test's context matches
        # what the model saw in the original run.
        self.summarizer.config["retrieval_placement"] = "system"
        self.summarizer.config["rolling_summaries"] = max(
            0, manifest["args"].get("rolling_summaries", 0)
        )
        self.summarizer.config["message_mode"] = manifest["args"].get(
            "message_mode", "enumerated"
        )
        print(f"[engine] Summarizer ready, sandbox={sandbox}, "
              f"rolling_summaries={self.summarizer.config['rolling_summaries']}, "
              f"message_mode={self.summarizer.config['message_mode']}")

    def _build_engine_state(self) -> dict:
        """Build the TGWUI-shaped engine state, matching the soak's build_state."""
        spread = self._spreadsheet
        chars = spread["characters"]
        ws = spread["writing_style"]
        taboos = "\n".join(f"- {t}" for t in ws.get("taboos", [])) or "- none"
        n1, n2 = chars["name1"], chars["name2"]
        context = (
            f"Premise: {spread['premise']}\n"
            f"Setting: {spread['setting']}\n"
            f"Canonical cast: {n1['name']} = {n1.get('role', 'the other character')} "
            f"(NOT the protagonist; the other character). {n2['name']} = "
            f"{n2.get('role', 'the protagonist')} (the protagonist being roleplayed).\n"
            f"Your voice ({n2['name']}, DSS turns): {ws['dss_directive']}\n"
            f"Taboos: {taboos}\n"
            f"Characters:\n- {n1['name']}: {n1.get('description', n1.get('role', ''))}\n"
            f"- {n2['name']}: {n2.get('description', n2.get('role', ''))}"
        )
        greeting = spread.get("greeting", "").strip()
        if not greeting:
            setting_first = spread['setting'].split(',')[0].strip() or 'the scene'
            greeting = f"You walk into {setting_first}."

        # Production shape: greeting pair at [0] + every prior exchange
        internal = [["<|BEGIN-VISIBLE-CHAT|>", greeting]]
        for user_msg, assistant_msg in self.history:
            internal.append([user_msg, assistant_msg])

        return {
            "context": context,
            "name1": n1["name"],
            "name2": n2["name"],
            "seed": self._seed,
            "unique_id": self._unique_id,
            "character_menu": f"soak_{spread['id']}",
            "greeting": greeting,
            "user_bio": n1.get("description", ""),
            "chat_template_str": "",
            "history": {"internal": internal},
            "truncation_length": 65536,
        }

    def _load_history(self) -> list[tuple[str, str]]:
        """Load all exchanges prior to this turn."""
        history = []
        # Scan all turn dirs (handles zero-padded names like turn_001)
        turn_dirs = sorted(
            d for d in self.run_dir.iterdir()
            if d.is_dir() and d.name.startswith("turn_")
        )
        for turn_dir in turn_dirs:
            result_path = turn_dir / "result.json"
            if not result_path.exists():
                continue
            try:
                result = json.loads(result_path.read_text())
                if result["turn"] < self.turn:
                    history.append((result["user_input"], result["reply"]))
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: failed to load {turn_dir.name}: {e}", file=sys.stderr)
        return history

    def build_baseline_context(self) -> dict:
        """Reconstruct what DSS actually saw (no advisor, director as configured)."""
        return {
            "user_input": self.user_input,
            "instr_prompt": self.instr_prompt,
            "context": self.saved_context,
        }

    def build_advisor_context(self, use_director: bool) -> dict:
        """Build context with [[...]] parsed and injected as steering."""
        clean_input, steering = parse_inline_instructions(self.user_input_with_advisor)

        if use_director:
            # Director mode: steering requirements appended to the saved instruction prompt
            if steering:
                steering_block = "\n\n[ADVISOR STEERING — weave these naturally into the reply:]\n"
                for s in steering:
                    steering_block += f"- {s}\n"
                modified_instr = self.instr_prompt + steering_block
            else:
                modified_instr = self.instr_prompt
        else:
            # No director: steering becomes direct reply requirements
            if steering:
                steering_block = "\n\n[REQUIRED — incorporate these elements into your reply:]\n"
                for s in steering:
                    steering_block += f"- {s}\n"
                modified_instr = steering_block
            else:
                modified_instr = ""

        return {
            "user_input": clean_input,
            "instr_prompt": modified_instr,
            "context": self.saved_context,
            "steering_requirements": steering,
        }

    def build_no_instr_context(self, include_reasoning: bool = False) -> dict:
        """Build context with no instruction generation (director off)."""
        clean_input, steering = parse_inline_instructions(self.user_input_with_advisor)

        if include_reasoning:
            # Reasoning mode: show what the instruction would have been
            reasoning = f"[WOULD HAVE GENERATED (director off): {len(self.instr_prompt)} chars of instruction]"
            if steering:
                reasoning += f"\n[WOULD ALSO HAVE STEERED WITH: {len(steering)} advisor requirement(s)]"
            return {
                "user_input": clean_input,
                "instr_prompt": "",
                "reasoning": reasoning,
                "context": self.saved_context,
            }
        else:
            return {
                "user_input": clean_input,
                "instr_prompt": "",
                "context": self.saved_context,
            }

    def analyze_context(self, ctx: dict, label: str) -> None:
        """Analyze and print details about a constructed context."""
        print(f"\n{'=' * 70}")
        print(f"{label}")
        print(f"{'=' * 70}")

        # User input
        ui = ctx.get("user_input", "")
        print(f"\n[User Input] ({len(ui)} chars)")
        print(f"  {ui[:200]}{'...' if len(ui) > 200 else ''}")

        # Instruction prompt
        ip = ctx.get("instr_prompt", "")
        if ip:
            print(f"\n[Instruction Prompt] ({len(ip)} chars)")
            print(f"  {ip[:300]}{'...' if len(ip) > 300 else ''}")
        else:
            print("\n[Instruction Prompt] (none)")

        # Steering requirements
        if "steering_requirements" in ctx:
            print(f"\n[Steering Requirements] ({len(ctx['steering_requirements'])})")
            for i, s in enumerate(ctx["steering_requirements"]):
                print(f"  [{i}] {s[:100]}{'...' if len(s) > 100 else ''}")

        # Reasoning
        if "reasoning" in ctx:
            print(f"\n[Reasoning]")
            print(f"  {ctx['reasoning']}")

        # Saved context (what DSS saw)
        saved_ctx = ctx.get("context", "")
        print(f"\n[Saved Context] ({len(saved_ctx)} chars)")
        if len(saved_ctx) > 500:
            print(f"  (showing first 500 chars)")
            print(f"  {saved_ctx[:500]}...")
        else:
            print(f"  {saved_ctx}")

    def diff_contexts(self, base: dict, modified: dict, label: str) -> None:
        """Show what changed between baseline and modified context."""
        print(f"\n{'-' * 70}")
        print(f"DIFF: {label}")
        print(f"{'-' * 70}")

        # User input diff
        if base.get("user_input") != modified.get("user_input"):
            print(f"[User Input] CHANGED: {len(base['user_input'])} → {len(modified['user_input'])} chars")
        else:
            print(f"[User Input] unchanged")

        # Instr prompt diff
        if base.get("instr_prompt") != modified.get("instr_prompt"):
            print(f"[Instr Prompt] CHANGED: {len(base.get('instr_prompt', ''))} → {len(modified.get('instr_prompt', ''))} chars")
            # Show added content
            base_ip = base.get("instr_prompt", "")
            mod_ip = modified.get("instr_prompt", "")
            if mod_ip.startswith(base_ip):
                added = mod_ip[len(base_ip):]
                print(f"  Added: {added[:200]}{'...' if len(added) > 200 else ''}")
            else:
                print(f"  (content differs entirely)")
        else:
            print(f"[Instr Prompt] unchanged")



    def test_relevance(self) -> None:
        """Test entity relevance: load entities from state_snapshot, score
        each against the current scene text using the cross-encoder.

        When graph relevance scores are available (from a prior --live engine
        call), prints a one-to-one comparison: CE rank/score vs graph rank/score
        for every entity, so the two relevance signals can be evaluated side-by-side.
        """
        print(f"\n{'=' * 70}")
        print(f"TEST: Entity Relevance (cross-encoder)")
        print(f"{'=' * 70}")

        # Load entities from the state snapshot
        entities = extract_entities_from_state(self.run_dir, self.turn)
        print(f"Loaded {len(entities)} entities from state_snapshot")

        if not entities:
            print("No entities found; skipping relevance scoring.")
            return

        # Show entity type breakdown
        type_counts = {}
        for e in entities:
            t = e.get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
        print(f"Breakdown: {type_counts}")

        # Score against the current scene (user input)
        scene_text = self.user_input
        print(f"\nScene text: {scene_text[:120]}{'...' if len(scene_text) > 120 else ''}")
        scored = self.reranker.score(scene_text, entities)

        # Build graph relevance lookup (from the last live engine call)
        graph_scores = getattr(self, "_last_graph_relevance", {})

        if graph_scores:
            # One-to-one comparison table
            self._print_relevance_comparison(scored, graph_scores)
        else:
            # No graph scores available — print CE-only (original behavior)
            print(f"\nTop 10 most relevant entities (CE only, no graph comparison):")
            for i, ent in enumerate(scored[:10]):
                print(f"  {i+1:2d}. [{ent['type']:10s}] {ent['name']:30s} CE={ent['relevance_score']:.3f}")

            if len(scored) > 10:
                print(f"\nBottom 10 least relevant entities:")
                for i, ent in enumerate(scored[-10:]):
                    print(f"  {len(scored)-9+i:2d}. [{ent['type']:10s}] {ent['name']:30s} CE={ent['relevance_score']:.3f}")

    def _print_relevance_comparison(self, ce_scored: list[dict],
                                     graph_scores: dict[str, tuple[str, float]]) -> None:
        """Print a one-to-one CE vs graph relevance comparison table.

        Args:
            ce_scored: List of entity dicts sorted by CE score (descending),
                        each with 'name', 'type', 'relevance_score'.
            graph_scores: Dict mapping entity name -> (type, graph_score).
        """
        # Build graph rank lookup (sorted by graph score descending)
        graph_sorted = sorted(graph_scores.items(), key=lambda x: x[1][1], reverse=True)
        graph_rank = {name: i + 1 for i, (name, _) in enumerate(graph_sorted)}

        # Count overlap: entities in both CE and graph
        ce_names = {e["name"] for e in ce_scored}
        overlap = ce_names & set(graph_scores.keys())

        print(f"\n{'=' * 70}")
        print(f"RELEVANCE COMPARISON: Cross-Encoder vs Entity Graph")
        print(f"{'=' * 70}")
        print(f"  CE entities: {len(ce_scored)}  |  Graph entities: {len(graph_scores)}  |  Overlap: {len(overlap)}")
        print(f"  CE scores: raw MiniLM logits (rank is meaningful, not absolute value)")
        print(f"  Graph scores: 0-100 traversal importance (final_threshold=75 gate)")

        # Header
        hdr = (f"{'CE rank':>7} {'CE score':>9} | "
               f"{'Gr rank':>7} {'Gr score':>8} | "
               f"{'type':<10} {'name'}")
        print(f"\n{hdr}")
        print("-" * len(hdr))

        # One row per CE-scored entity, showing its graph score (or '---' if absent)
        for i, ent in enumerate(ce_scored):
            ce_rank = i + 1
            ce_score = ent["relevance_score"]
            name = ent["name"]
            etype = ent["type"]

            if name in graph_scores:
                gr_rank = graph_rank[name]
                gr_score = graph_scores[name][1]
                gr_col = f"{gr_rank:>7} {gr_score:>8.1f}"
            else:
                gr_col = f"{'—':>7} {'—':>8}"

            print(f"{ce_rank:>7} {ce_score:>9.3f} | {gr_col} | {etype:<10} {name}")

        # Show graph entities NOT in the CE set (in graph but not in state_snapshot)
        only_graph = set(graph_scores.keys()) - ce_names
        if only_graph:
            print(f"\n  Graph-only entities (not in state_snapshot / CE set): {len(only_graph)}")
            for name in sorted(only_graph, key=lambda n: graph_scores[n][1], reverse=True)[:10]:
                etype, score = graph_scores[name]
                print(f"         [{etype:<10}] {name:30s} graph={score:.1f}")

        # Summary: top-10 overlap analysis
        print(f"\n{'-' * 70}")
        print(f"TOP-10 OVERLAP ANALYSIS")
        print(f"{'-' * 70}")
        ce_top10 = {e["name"] for e in ce_scored[:10]}
        gr_top10 = {name for name, _ in graph_sorted[:10]}
        overlap_top10 = ce_top10 & gr_top10
        print(f"  CE top-10:    {sorted(ce_top10)}")
        print(f"  Graph top-10: {sorted(gr_top10)}")
        print(f"  Overlap:      {len(overlap_top10)}/10  {sorted(overlap_top10) if overlap_top10 else '(none)'}")

        # Spearman rank correlation — re-rank within the overlap set only
        if len(overlap) >= 3:
            overlap_ce = sorted(overlap, key=lambda n: next(e["score"] for e in ce_scored if e["name"] == n))
            overlap_gr = sorted(overlap, key=lambda n: graph_scores[n][1], reverse=True)
            ce_rank_in_overlap = {n: i + 1 for i, n in enumerate(overlap_ce)}
            gr_rank_in_overlap = {n: i + 1 for i, n in enumerate(overlap_gr)}
            pairs = [(ce_rank_in_overlap[n], gr_rank_in_overlap[n]) for n in overlap]
            n = len(pairs)
            d_sq_sum = sum((a - b) ** 2 for a, b in pairs)
            spearman = 1 - (6 * d_sq_sum) / (n * (n ** 2 - 1))
            print(f"\n  Spearman rank correlation (n={n} overlapping, re-ranked within overlap): {spearman:+.3f}")

    def generate_reply_live(self, label: str, do_instr: bool = False,
                            highlight_relevant: bool = False,
                            thinking: bool | None = None) -> str:
        """Generate a reply through the DSS engine's full pipeline.

        Calls generate_instr_prompt (retrieval + graph traversal + instruction
        generation) then passes the returned custom_state to the local model —
        the same path the soak uses in production.

        ``thinking`` overrides the DSS_BENCH_THINKING env var for this call
        only (used by the A/B reasoning modes). None = use the env var as-is.
        """
        if self.local_model is None or self.summarizer is None:
            raise RuntimeError("--live flag required to generate replies")

        engine_state = self._build_engine_state()
        history_internal = engine_state["history"]["internal"]

        print(f"\n{'-' * 70}")
        print(f"[LIVE] {label}")
        print(f"  do_instr={do_instr}  highlight_relevant={highlight_relevant}  thinking={thinking}")
        print(f"  history_internal: {len(history_internal)} entries")
        print(f"{'-' * 70}")

        # Full DSS pipeline: retrieval, graph traversal, instruction generation
        t0 = time.time()
        instr_prompt, custom_state, history_path, ts = self.summarizer.generate_instr_prompt(
            self.user_input, engine_state, history_internal,
            do_instr=do_instr,
            highlight_relevant=highlight_relevant,
        )
        print(f"[engine] instr_prompt built in {time.time() - t0:.1f}s "
              f"({len(str(instr_prompt))} chars)")

        # Capture graph relevance scores for the one-to-one CE comparison
        self._last_graph_relevance = {}
        try:
            _rc = self.summarizer.last.context[0] if self.summarizer.last else None
            if _rc and getattr(_rc, "relevant_entities", None):
                for etype, name_score in _rc.relevant_entities.items():
                    for name, score in name_score.items():
                        self._last_graph_relevance[name] = (etype, score)
        except (IndexError, AttributeError):
            pass

        # Context completeness diagnostic
        ctx = custom_state.get("context", "") if custom_state else ""
        ctx_len = len(str(ctx))
        hist_len = len(history_internal)
        print(f"[engine] context: {ctx_len} chars, history: {hist_len} entries")
        if ctx_len < 20000:
            print(f"[engine] WARNING: context is {ctx_len} chars — may be incomplete")

        # Override thinking for this call if requested (A/B reasoning modes)
        _saved_env = os.environ.get("DSS_BENCH_THINKING")
        if thinking is not None:
            os.environ["DSS_BENCH_THINKING"] = "1" if thinking else "0"
        try:
            # Model call with the engine-built state (has proper context + history)
            t0 = time.time()
            self.local_model.complete(str(instr_prompt), custom_state)
            elapsed = time.time() - t0
        finally:
            # Restore env var
            if _saved_env is not None:
                os.environ["DSS_BENCH_THINKING"] = _saved_env
            elif "DSS_BENCH_THINKING" in os.environ:
                del os.environ["DSS_BENCH_THINKING"]

        # Use last_content (not the complete() return) so thinking never leaks
        # into the reply when content is empty. Strip any inline thinking tags
        # as a safety net.
        reply = strip_thinking(self.local_model.last_content)

        # Surface raw thinking verbatim when thinking is on.
        reasoning = getattr(self.local_model, "last_reasoning", "")
        if reasoning:
            print(f"\n[THINKING] ({len(reasoning)} chars) — raw reasoning_content, verbatim")
            print("-" * 70)
            print(reasoning)
            print("-" * 70)

        print(f"\nGenerated in {elapsed:.1f}s ({len(reply)} chars)")
        print(f"Reply: {reply[:300]}{'...' if len(reply) > 300 else ''}")

        # Save full response to file for inspection
        out_path = self.run_dir / f"live_mode_{do_instr}_{highlight_relevant}_{thinking}.json"
        out_path.write_text(json.dumps({
            "mode": "chat-instruct",
            "do_instr": do_instr,
            "highlight_relevant": highlight_relevant,
            "thinking": thinking,
            "reply": reply,
            "reasoning": reasoning,
            "instr_prompt": str(instr_prompt),
            "context_chars": len(str(custom_state.get("context", ""))),
            "elapsed_s": elapsed,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  [saved] {out_path.name}")

        return reply

    def test_all_modes(self) -> None:
        """Run all valid test modes and compare."""
        print(f"\n{'=' * 70}")
        print(f"TEST SUITE: Context Injection — {self.run_dir.name} turn {self.turn}")
        print(f"{'=' * 70}")

        results = {}

        # Mode 1: No instr, no advisor, no reasoning (baseline)
        ctx_000 = self.build_no_instr_context(include_reasoning=False)
        self.analyze_context(ctx_000, "MODE 1: No instr, no advisor (baseline)")
        if self.live:
            results["000"] = self.generate_reply_live("Mode 1: No instr, no advisor", do_instr=False, thinking=False)

        # Mode 2: No instr, no advisor, WITH reasoning (A/B vs Mode 1)
        ctx_001 = self.build_no_instr_context(include_reasoning=True)
        self.analyze_context(ctx_001, "MODE 2: No instr w/ reasoning")
        self.diff_contexts(ctx_000, ctx_001, "Mode 2 vs Mode 1")
        if self.live:
            results["001"] = self.generate_reply_live("Mode 2: No instr w/ reasoning", do_instr=False, thinking=True)

        # Mode 3: No instr, WITH advisor (highlight_relevant), no reasoning
        ctx_010 = self.build_advisor_context(use_director=False)
        self.analyze_context(ctx_010, "MODE 3: No instr + advisor")
        self.diff_contexts(ctx_000, ctx_010, "Mode 3 vs Mode 1")
        if self.live:
            results["010"] = self.generate_reply_live("Mode 3: No instr + highlight", do_instr=False, highlight_relevant=True, thinking=False)

        # Mode 4: No instr, WITH advisor, WITH reasoning (A/B vs Mode 3)
        ctx_011 = self.build_advisor_context(use_director=False)
        steering = ctx_011.get("steering_requirements", [])
        ctx_011["reasoning"] = (
            f"[WOULD HAVE STEERED WITH {len(steering)} advisor requirement(s). "
            f"Advisor-only mode does not generate a full instruction block — "
            f"steering requirements are appended as reply constraints.]"
        )
        self.analyze_context(ctx_011, "MODE 4: No instr + advisor w/ reasoning")
        self.diff_contexts(ctx_010, ctx_011, "Mode 4 vs Mode 3")
        if self.live:
            results["011"] = self.generate_reply_live("Mode 4: No instr + highlight w/ reasoning", do_instr=False, highlight_relevant=True, thinking=True)

        # Mode 5: Director on, no advisor
        ctx_100 = self.build_baseline_context()
        self.analyze_context(ctx_100, "MODE 5: Director on (saved baseline)")
        if self.live:
            results["100"] = self.generate_reply_live("Mode 5: Director on", do_instr=True, thinking=False)

        # Mode 6: Director on + advisor (highlight_relevant)
        ctx_110 = self.build_advisor_context(use_director=True)
        self.analyze_context(ctx_110, "MODE 6: Director on + advisor")
        self.diff_contexts(ctx_100, ctx_110, "Mode 6 vs Mode 5")
        if self.live:
            results["110"] = self.generate_reply_live("Mode 6: Director + highlight", do_instr=True, highlight_relevant=True, thinking=False)

        # Entity relevance (if rerank requested)
        if self.rerank:
            self.test_relevance()

        # Summary comparison (live mode only)
        if self.live and len(results) >= 2:
            print(f"\n{'=' * 70}")
            print(f"LIVE GENERATION COMPARISON")
            print(f"{'=' * 70}")
            for mode_id, reply in results.items():
                print(f"\n[Mode {mode_id}] ({len(reply)} chars)")
                # Show first 200 chars of each
                print(f"{reply[:200]}{'...' if len(reply) > 200 else ''}")



def main():
    parser = argparse.ArgumentParser(
        description="Test context injection on an existing soak turn. "
                    "Offline (default): analyze saved data only. "
                    "--live: generate new replies. --rerank: cross-encoder relevance.")
    parser.add_argument("--run", required=True, help="Run name (e.g., greek_mythology_retelling__2e7575de)")
    parser.add_argument("--turn", type=int, required=True, help="Turn number")
    parser.add_argument("--live", action="store_true", help="Generate new replies using the local model for each mode")
    parser.add_argument("--thinking", action="store_true", help="Enable thinking (enable_thinking=true) for live generation")
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default=None,
                        help="Reasoning effort level (requires --thinking)")
    parser.add_argument("--rerank", action="store_true", help="Cross-encoder entity relevance scoring")
    parser.add_argument("--rerank-model", choices=["mini", "mid", "bge"], default="mid",
                        help="Cross-encoder model: mini (L2, fastest), mid (L6, default), bge (highest quality)")
    parser.add_argument("--inject", nargs="+", help="Inject synthetic [[...]] instructions for testing")
    args = parser.parse_args()

    test = ContextInjectionTest(
        args.run, args.turn,
        rerank=args.rerank,
        live=args.live,
        rerank_model=args.rerank_model,
        inject_instructions=args.inject,
        thinking=args.thinking,
        reasoning_effort=args.reasoning_effort,
    )
    test.test_all_modes()


if __name__ == "__main__":
    main()