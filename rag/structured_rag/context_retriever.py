from __future__ import annotations
from typing import TYPE_CHECKING, Any

import copy
import jsonc
import logging
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
import traceback
from os import PathLike

from ...utils.schema_parser import SchemaWrapper


if TYPE_CHECKING:
    import nltk
    import spacy
    from spacy.tokens import Doc
    from llama_index.core import VectorStoreIndex, StorageContext
    from llama_index.core.node_parser import SimpleNodeParser
    from llama_index.core.indices.loading import load_index_from_storage
    from llama_index.core.settings import Settings
    from llama_index.core.schema import TextNode
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from ...agents.summarizer import Summarizer
else:
    nltk = None
    spacy = None
    Doc = None
    HuggingFaceEmbedding = None
    VectorStoreIndex = None
    StorageContext = None
    SimpleNodeParser = None
    load_index_from_storage = None
    Settings = None
    TextNode = None

from ...utils.helpers import (
    _ERROR,
    _SUCCESS,
    _INPUT,
    _GRAY,
    _HILITE,
    _BOLD,
    _RESET,
    _DEBUG,
    _WARNING,
)


from ...utils.background_importer import (
    start_background_import,
    get_imported_attribute,
)

from .entity_graph import EntityGraph

start_background_import("nltk")
start_background_import("spacy")
start_background_import("llama_index.core")
start_background_import("llama_index.core.node_parser", "SimpleNodeParser")
start_background_import("llama_index.core.indices.loading", "load_index_from_storage")
start_background_import("llama_index.core.settings", "Settings")
start_background_import("llama_index.core.schema", "TextNode")
start_background_import("llama_index.embeddings.huggingface", "HuggingFaceEmbedding")


from .decay_config import DecayConfig


@dataclass
class RetrievalContext:
    context: str = ""
    current_scene: dict[str, dict] = field(default_factory=dict)
    # Subject data keyed by subject name ("characters", "groups", etc.)
    # New subjects need a schema entry, not a dataclass field.
    subjects: dict[str, dict] = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)
    messages_metadata: list[dict] = field(default_factory=list)
    character_status: dict[str, dict] = field(default_factory=dict)
    character_milestones: dict[str, list[dict]] = field(default_factory=dict)
    relevant_entities: dict[str, dict[str, float]] = field(default_factory=dict)
    # Full stored events (unfiltered) as a separate view from the capped
    # ``events`` render block, so scene-boundary/scene-key consumers never lose
    # far-away archived scenes to the selection cap.
    events_full: dict[str, dict] = field(default_factory=dict)


# Recency window (in message-index units) promoted ahead of pure-similarity RAG
# ranking: messages within this of the newest message outrank older ones, which
# stops the retriever re-serving STALE beats (measured: the pact-hall beat was
# [1] at consecutive checkpoints) underneath the physically-recent beats.
_RAG_RECENCY_MESSAGES = 15
_RAG_ANCHOR_COUNT_DEFAULT = 5


class StoryContextRetriever:
    def __init__(self, history_path: PathLike, schema_classes: dict | None = None, summarizer: 'Summarizer' | None = None, rag_anchor_count: int | None = None, retrieval_order: list[str] | None = None):
        """Initialize the context retriever with a history path.

        Args:
            history_path: Path to the history directory.
            schema_classes: Optional dict of ParsedSchemaClass objects for schema-driven entity graph.
            summarizer: Optional Summarizer instance for LLM-based speaker extraction.
        """
        history_path = Path(history_path)
        if not history_path.exists():
            raise ValueError(f"History path does not exist: {history_path}")
        self.history_path = history_path

        # Load static data
        self.characters_path = history_path / "characters.json"
        self.events_path = history_path / "events.json"
        self.groups_path = history_path / "groups.json"
        self.elements_path = history_path / "elements.json"
        self.general_info_path = history_path / "general_info.json"
        self.current_scene_path = history_path / "current_scene.json"
        self.arcs_path = history_path / "arcs.json"
        self.knowledge_path = history_path / "knowledge.json"

        self.characters = self._load_json(self.characters_path)
        self.groups = self._load_json(self.groups_path)
        self.elements = self._load_json(self.elements_path)
        self.events = self._load_json(self.events_path)
        self.general_info = self._load_json(self.general_info_path)
        self.current_scene = self._load_json(self.current_scene_path)
        self.arcs = self._load_json(self.arcs_path)
        self.knowledge = self._load_json(self.knowledge_path)

        print(f"{_DEBUG}StoryContextRetriever loaded:")
        print(f"  history_path: {history_path}")
        print(f"  characters keys: {list(self.characters.keys()) if self.characters else 'empty'}")
        print(f"  groups keys: {list(self.groups.keys()) if self.groups else 'empty'}")
        print(f"  elements keys: {list(self.elements.keys()) if self.elements else 'empty'}")
        print(f"  events keys: {list(self.events.keys()) if self.events else 'empty'}")
        print(f"  general_info keys: {list(self.general_info.keys()) if self.general_info else 'empty'}")
        print(f"  current_scene keys: {list(self.current_scene.keys()) if self.current_scene else 'empty'}")
        print(f"  arcs count: {len(self.arcs) if self.arcs else 0}")
        print(f"  knowledge count: {len(self.knowledge.get('entries', self.knowledge)) if self.knowledge else 0}{_RESET}")

        # Initialize entity graph for relationship tracking
        self.schema_classes = schema_classes or {}
        self.retrieval_order = retrieval_order or []
        self.schema_wrapper = SchemaWrapper(self.schema_classes) if self.schema_classes else None
        try:
            self.entity_graph = EntityGraph(history_path, persist=True, schema_classes=schema_classes)
            print(f"{_DEBUG}EntityGraph initialized with {len(self.entity_graph.nodes)} nodes{_RESET}")
        except Exception as e:
            print(f"{_ERROR}Failed to initialize EntityGraph: {e}{_RESET}")
            self.entity_graph = None

        # Store summarizer and pass to chunker for LLM-based speaker extraction
        self.summarizer = summarizer
        # TODO: Make this configurable via UI toggle
        self.use_llm_for_speakers = True  # Toggle: True to always use LLM, False to use regex/spaCy
        characters_map = self.characters.get("entries", self.characters)
        groups_map = self.groups.get("entries", self.groups)
        try:
            self.chunker = MessageChunker(history_path, characters_map, groups_map, self.elements, self.events, self.current_scene, summarizer=summarizer, use_llm_for_speakers=self.use_llm_for_speakers)
        except Exception as e:
            print(f"[RAG] MessageChunker unavailable ({e.__class__.__name__}): {e}")
            print("[RAG] RAG similarity search disabled; entity graph traversal still active.")
            self.chunker = self._build_docstore_shim(history_path)

        # Create entity name patterns for recognition (from both JSON and graph)
        self._entity_patterns: dict[str, dict[str, re.Pattern]] = {}
        self._entity_alias_maps: dict[str, dict[str, str]] = {}
        for _st in self.retrieval_order:
            _st_cap = _st.capitalize()
            self._entity_patterns[_st_cap], self._entity_alias_maps[_st_cap] = self._create_entity_patterns(_st_cap)

        self.rag_anchor_count = int(rag_anchor_count) if rag_anchor_count else _RAG_ANCHOR_COUNT_DEFAULT

    @staticmethod
    def _build_docstore_shim(history_path: PathLike):
        """Build a minimal chunker shim that exposes index.docstore.docs.

        When MessageChunker cannot be constructed (e.g. llama_index import
        deadlock on WSL), DSS rolling summaries are still available from the
        persisted docstore.json. This shim provides just enough interface for
        ``_load_summary_nodes`` to read them.
        """
        docstore_path = Path(history_path) / "message_index" / "docstore.json"
        if not docstore_path.exists():
            return None

        class _ShimNode:
            def __init__(self, data: dict):
                self.metadata = data.get("metadata", {})
                self.text = data.get("text", "")

        class _ShimDocstore:
            def __init__(self, docs: dict):
                self.docs = docs

        class _ShimIndex:
            def __init__(self, docstore):
                self.docstore = docstore

        class _ShimChunker:
            def __init__(self, index):
                self.index = index

        try:
            with open(docstore_path) as f:
                raw = jsonc.load(f)
            data_map = raw.get("docstore/data", {})
            docs = {}
            for node_id, entry in data_map.items():
                # Each entry is {"__data__": {text, metadata, ...}, "__type__": "1"}
                node_data = entry.get("__data__", {}) if isinstance(entry, dict) else {}
                text = node_data.get("text", "")
                meta = dict(node_data.get("metadata", {}))
                docs[node_id] = _ShimNode({"metadata": meta, "text": text})
            if not docs:
                return None
            return _ShimChunker(_ShimIndex(_ShimDocstore(docs)))
        except Exception as e:
            print(f"[RAG] docstore shim failed: {e}")
            return None

    # Subjects whose entries are referenced by name in prose and therefore
    # need compiled name patterns for text extraction. Adding a new subject
    def _create_entity_patterns(self, subject_type: str) -> tuple[dict[str, re.Pattern], dict[str, str]]:
        """Create regex name patterns for a subject type (generic).

        Collects names from JSON entries and (if available) entity-graph
        nodes of the matching type. Builds whole-word patterns (\b
        boundaries) over the full name, first token, last token, and any
        declared aliases. Returns (patterns, alias_map) where alias_map
        maps lowercase alias → canonical name.
        """
        patterns: dict[str, re.Pattern] = {}
        alias_map: dict[str, str] = {}

        data = getattr(self, subject_type.lower(), {})
        entries = self._get_entries(data, subject_type)

        # Names: JSON entries first, then any graph-only names
        names = list(entries.keys())
        if self.entity_graph:
            graph_nodes = self.entity_graph.get_nodes_by_type(subject_type.lower())
            for node in graph_nodes:
                if node.name not in names:
                    names.append(node.name)

        for name in names:
            tokens = name.split()
            alternatives = [re.escape(name)]
            if len(tokens) > 1:
                alternatives.append(re.escape(tokens[0]))
                alternatives.append(re.escape(tokens[-1]))
            entry = entries.get(name, {})
            if not isinstance(entry, dict):
                entry = {}
            for alias in self._get_field_value(entry, subject_type, "aliases", []):
                if alias and alias != name:
                    alternatives.append(re.escape(alias))
                    alias_map[alias.lower()] = name
            patterns[name] = re.compile(
                "(" + "|".join(f"\\b{a}\\b" for a in alternatives) + ")",
                flags=re.IGNORECASE,
            )
        return patterns, alias_map

    def _extract_entity_names(
        self,
        patterns: dict[str, re.Pattern],
        alias_map: dict[str, str],
        text: str,
    ) -> list[str]:
        """Extract canonical entity names from text (generic).

        Uses pre-compiled patterns for primary matching, then resolves
        any alias hits to their canonical name.
        """
        found: set[str] = set()
        for name, pattern in patterns.items():
            if pattern.search(text):
                found.add(name)
        for alias, canonical in alias_map.items():
            if re.search(r'\b' + re.escape(alias) + r'\b', text, flags=re.IGNORECASE):
                found.add(canonical)
        return list(found)

    # Cosine threshold for semantic entity matching. High on purpose:
    # this is a fallback for names the regex stage missed, so precision
    # matters more than recall. 0.85 ≈ same entity referred to
    # descriptively; near-names ("Kael" vs "Kaelan") typically score
    # 0.75-0.82 and are correctly rejected.
    _SEMANTIC_ENTITY_THRESHOLD = 0.85

    def _semantic_entity_fallback(
        self, text: str, known_names: set[str], all_names: list[str],
    ) -> list[str]:
        """Semantic fallback for entity resolution.

        Encodes the search text and each unmatched entity name with the
        shared embedding model, then returns names whose cosine similarity
        exceeds _SEMANTIC_ENTITY_THRESHOLD. Only considers names NOT
        already in known_names (the regex stage's results).

        Returns an empty list if the embed model is unavailable.
        """
        unmatched = [n for n in all_names if n not in known_names]
        if not unmatched:
            return []

        try:
            from .message_chunker import MessageChunker
            embed_model = MessageChunker._embed_model
            if embed_model is None:
                return []
            # HuggingFaceEmbedding wraps a SentenceTransformer in ._model
            sf_model = embed_model._model
            import numpy as np
            text_emb = sf_model.encode(text, normalize_embeddings=True)
            name_embs = sf_model.encode(unmatched, normalize_embeddings=True)
            # Cosine similarity (vectors are already normalized → dot product)
            sims = name_embs @ text_emb
            matched = []
            for name, sim in zip(unmatched, sims):
                if sim >= self._SEMANTIC_ENTITY_THRESHOLD:
                    matched.append(name)
            return matched
        except Exception:
            return []

    # ── Schema-driven relevance dispatcher ──────────────────────────────────

    def _get_relevance_config(self, subject_type: str) -> list:
        """Read the relevance signal list from the schema for a subject type.

        Each entry is a bare string ("text", "topology", "relationship", "semantic")
        or a dict with a "signal" key plus params (e.g. {"signal": "structural", "field": "scenes"}).
        """
        if not self.schema_wrapper:
            return []
        fields = self.schema_wrapper.get_entity_fields(subject_type.capitalize())
        rel = fields.get("relevance", {}).get("default", [])
        return rel if isinstance(rel, list) else []

    def _signal_text(self, subject_type: str, context: str, entries: dict) -> set[str]:
        """Text co-occurrence: entry name/alias appears in the prose."""
        patterns, alias_map = self._create_entity_patterns(subject_type)
        found = self._extract_entity_names(patterns, alias_map, context)
        return set(n for n in found if n in entries)

    def _signal_relationship(self, subject_type: str, anchors: dict[str, set[str]], entries: dict) -> set[str]:
        """Relationship reference: an entry's own data references an already-relevant entity.

        Reads relationship_format from the schema to discover which fields
        hold references and what target type they point to.
        """
        if not self.schema_wrapper:
            return set()
        rel_fields = self.schema_wrapper.get_relationship_fields(subject_type.capitalize())
        matched: set[str] = set()
        for field_name, config in rel_fields.items():
            target_type = config.get("target_type", "").capitalize()
            anchor_names = anchors.get(target_type, set())
            if not anchor_names:
                continue
            for entry_name, entry_data in entries.items():
                if not isinstance(entry_data, dict):
                    continue
                field_val = entry_data.get(field_name, {})
                if isinstance(field_val, dict):
                    if any(a in field_val for a in anchor_names):
                        matched.add(entry_name)
        return matched

    def _signal_topology(self, subject_type: str, anchors: dict[str, set[str]], entries: dict) -> set[str]:
        """Topological proximity: the entity graph has a qualifying edge
        from an already-relevant entity to this subject's entry."""
        if not self.entity_graph:
            return set()
        matched: set[str] = set()
        target_lower = subject_type.lower()
        for anchor_type, anchor_names in anchors.items():
            if not anchor_names:
                continue
            for name in anchor_names:
                try:
                    neighbors = self.entity_graph.get_neighbors(
                        name, anchor_type.lower(), target_type=target_lower,
                    )
                    matched.update(n for n in neighbors if n in entries)
                except Exception:
                    pass
        return matched

    def _signal_structural(self, field: str, context: str, entries: dict) -> set[str]:
        """Structural membership: entry is a named child of a container field.

        Navigates into entries[field] to get the actual sub-entries, then
        checks if any sub-entry name appears in the context.
        """
        sub = entries.get(field, {})
        if not isinstance(sub, dict):
            return set()
        matched: set[str] = set()
        for name in sub:
            if re.search(r'\b' + re.escape(name) + r'\b', context, flags=re.IGNORECASE):
                matched.add(name)
        return matched

    def _signal_semantic(self, subject_type: str, context: str, entries: dict, already: set[str]) -> set[str]:
        """Semantic fallback: embedding similarity for names the other signals missed."""
        all_names = list(entries.keys())
        return set(self._semantic_entity_fallback(context, already, all_names))

    def _get_relevant_entries(
        self,
        subject_type: str,
        anchors: dict[str, set[str]],
        context: str,
    ) -> dict[str, dict]:
        """Schema-driven relevance dispatcher.

        Runs each signal from the subject's schema-declared relevance list,
        unioning the results. Returns {name: entry_data}.
        """
        entries = self._get_entries(
            getattr(self, subject_type.lower(), {}),
            subject_type,
        )
        if not entries:
            return {}

        config = self._get_relevance_config(subject_type)
        if not config:
            return {}

        matched: set[str] = set()
        for signal in config:
            if isinstance(signal, str):
                sig = signal
                params = {}
            else:
                sig = signal.get("signal", "")
                params = signal

            if sig == "text":
                matched |= self._signal_text(subject_type, context, entries)
            elif sig == "relationship":
                matched |= self._signal_relationship(subject_type, anchors, entries)
            elif sig == "topology":
                matched |= self._signal_topology(subject_type, anchors, entries)
            elif sig == "structural":
                field = params.get("field", "")
                if field:
                    matched |= self._signal_structural(field, context, entries)
            elif sig == "semantic":
                matched |= self._signal_semantic(subject_type, context, entries, matched)

        return {n: entries[n] for n in matched if n in entries}

    def _select_relevant_events(
        self,
        context: str,
        characters: list[str],
        groups: dict[str, dict],
        current_scene: dict | None,
        max_events: int = 8,
    ) -> dict[str, dict | list]:
        """Data-driven events selection for the retrieval context.

        This is the READ-side counterpart to the write-side events storage. The
        old path could only surface events that happened to exist as entity-graph
        milestone edges AND returned them in an ``{"entries": ...}`` shape the
        format template never reads — so the events block rendered ZERO sections
        even when events.json was full. Selection is: (1) name/alias match
        against the query context, (2) events of the CURRENT scene (start at/after
        the scene's opening message node), (3) recency fill (freshest remaining)
        up to max_events. Returns the FULL category shape the events format
        template renders (``{past, scenes, events, chapters, entries}``).
        """
        by_cat: dict[str, dict] = {}
        for bucket in ("past", "scenes", "events", "chapters"):
            bd = self._get_field_value(self.events, "Event", bucket, {}) or {}
            by_cat[bucket] = {k: v for k, v in bd.items() if isinstance(v, dict)} if isinstance(bd, dict) else {}

        def _node_idx(data: dict) -> int:
            try:
                node = (data.get("start") or {}).get("_message_node", "")
                return int(str(node).split("_")[0]) if node else -1
            except Exception:
                return -1

        # flattened name -> (category, data)
        flat: list[tuple[str, str, dict]] = []
        for cat, bucket in by_cat.items():
            flat.extend((n, cat, d) for n, d in bucket.items())

        ctx_n = context.casefold()
        picked: list[tuple[str, str, dict]] = []
        picked_keys: set[str] = set()
        # 1) name/alias match against the query context
        for name, cat, data in flat:
            if name in picked_keys:
                continue
            aliases = data.get("aliases")
            if not isinstance(aliases, (list, tuple)):
                aliases = []
            hay = " ".join(
                str(x).casefold()
                for x in [name, data.get("formal_name"), *aliases]
                if x
            )
            if hay and (hay in ctx_n or name.casefold() in ctx_n):
                picked.append((name, cat, data))
                picked_keys.add(name)
        # 2) current scene's own events (started at/after its opening)
        cs_start = -1
        if isinstance(current_scene, dict):
            node = (current_scene.get("start") or {}).get("_message_node", "") or current_scene.get("_message_node", "")
            if node:
                try:
                    cs_start = int(str(node).split("_")[0])
                except Exception:
                    cs_start = -1
        if cs_start >= 0:
            for name, cat, data in flat:
                if name not in picked_keys and _node_idx(data) >= cs_start:
                    picked.append((name, cat, data))
                    picked_keys.add(name)
        # 3) recency fill (freshest first)
        if len(picked) < max_events:
            rest = [(n, c, d) for n, c, d in flat if n not in picked_keys]
            rest.sort(key=lambda t: _node_idx(t[2]), reverse=True)
            for name, cat, data in rest:
                if len(picked) >= max_events:
                    break
                picked.append((name, cat, data))
                picked_keys.add(name)

        result: dict[str, dict | list] = {}
        for bucket, bd in by_cat.items():
            result[bucket] = {n: d for n, c, d in picked if c == bucket}
            if bucket == "chapters":
                # chapters is a LIST[Chapter] in the schema; keep the original
                # list shape (the template iterates it as a sequence).
                result[bucket] = [d for n, c, d in picked if c == bucket]
        result["entries"] = {n: d for n, c, d in picked}
        return result

    def _load_message_summaries(self) -> list[tuple[int, str]]:
        """[(message_idx, text)] from the chunker's summary nodes, ascending,
        deduped by message_idx (newest write wins). None-safe.
        """
        try:
            chunker = getattr(self, "chunker", None)
            index = getattr(chunker, "index", None)
            docs = getattr(index, "docstore", None)
            if docs is None:
                return []
            by_idx: dict[int, str] = {}
            for node in getattr(docs, "docs", {}).values():
                meta = getattr(node, "metadata", None) or {}
                if not meta.get("is_summary"):
                    continue
                idx = meta.get("message_idx")
                if not isinstance(idx, int) or idx < 0:
                    continue
                text = getattr(node, "text", None) or ""
                if isinstance(text, str) and text.strip():
                    by_idx[idx] = text.strip()
            return sorted(by_idx.items())
        except Exception:
            return []

    def _recent_state_map(self, names: list[str]) -> dict[str, str]:
        """Freshest story fragment mentioning each name -> {name: text}.

        This is the per-entity 'latest state' surface: cards whose stored
        description went stale (the blade stayed 'oilcloth-wrapped' 10 turns
        after baring) still show what the STORY actually did with them most
        recently, from the message summaries (dense, recent) with a stored
        events fallback. Text is truncated to a compact line.
        """
        result: dict[str, str] = {}
        summaries = self._load_message_summaries()
        tokenized: list[tuple[int, set[str], str]] = []
        for idx, text in summaries:
            words = set(re.sub(r"[^a-z0-9 ]", " ", text.casefold()).split())
            tokenized.append((idx, words, text))
        for name in names:
            if not name:
                continue
            nf = name.casefold()
            hit = None
            for idx, words, text in reversed(tokenized):  # newest first
                if " " in name:
                    m = nf in text.casefold()
                else:
                    m = nf in words
                if m:
                    hit = (idx, text.strip())
                    break
            if hit:
                result[name] = "[msg %d] %s" % (hit[0], hit[1][:220])
                continue
            # events fallback: freshest event/scene whose text mentions the entity
            combined: dict[str, dict] = {}
            for bucket in ("past", "scenes", "events"):
                bd = self._get_field_value(self.events, "Event", bucket, {}) or {}
                if isinstance(bd, dict):
                    combined.update({k: v for k, v in bd.items() if isinstance(v, dict)})

            def _node(v: dict) -> int:
                try:
                    node = (v.get("start") or {}).get("_message_node", "")
                    return int(str(node).split("_")[0]) if node else -1
                except Exception:
                    return -1

            best = None
            best_idx = -1
            for k, v in combined.items():
                blob = " ".join(str(x) for x in [k, v.get("summary"), v.get("catalyst"), v.get("outcome")] if x)
                if nf in blob.casefold():
                    idx = _node(v)
                    if idx > best_idx:
                        best_idx = idx
                        best = (k, v)
            if best:
                k, v = best
                result[name] = "[%s] %s" % (k, str(v.get("summary") or v.get("catalyst") or "")[:220])
        return result

    def _get_message_chunks(self, scene_name: str = None) -> list[str]:
        """Retrieve relevant message chunks based on scene or context."""
        messages = []

        if self.chunker is None:
            return messages

        # Get message indices from current scene
        current_scene = self.get_current_scene()
        if current_scene and "messages" in current_scene:
            for msg_range in current_scene["messages"]:
                start, end = msg_range
                # Adjust indices for 1-based indexing in query
                query = f"message_idx:[{start + 1} TO {end + 1}]"
                results = self.chunker.query_similar(query)
                if results and results["documents"]:
                    messages.extend(results["documents"])

        return messages

    # ------------------------------------------------------------------ #
    # RAG fan-out (rag_redesign_p1.md, Decisions 1 + 2)
    # ------------------------------------------------------------------ #
    # The old path fed the ENTIRE rendered context (~14.7k tokens) as ONE embedding
    # query. The 384-token-cap embedder kept only the first 384 tokens = 100% static
    # system-prompt boilerplate; every story marker sat past the cut, so per-turn query
    # vectors were ~identical (measured cosine 1.00000) and the ranking was noise.
    # The fix is query CONSTRUCTION, not the model: fan out over N short POINTED anchor
    # queries, each built deterministically from already-computed signals (cast,
    # mentioned entities, event anchors, stale roster) and each inside the 384-token
    # window, so each is a genuinely discriminative vector. No new model, ~0.3s, 0 VRAM.
    _ANCHOR_CHAR_BUDGET = 240  # chars per anchor (well inside the 384-token window)

    def _event_importance(self, data: dict) -> int:
        """importance.score of an event dict (0 when absent/non-int)."""
        imp = data.get("importance")
        if isinstance(imp, dict):
            s = imp.get("score")
            if isinstance(s, (int, float)):
                return int(s)
        return 0

    def _event_participants(self, data: dict) -> list[str]:
        """Event participant names, highest per-participant importance first."""
        parts = data.get("participants")
        if not isinstance(parts, dict):
            return []
        def _pimp(v: Any) -> int:
            if isinstance(v, dict):
                s = v.get("importance")
                if isinstance(s, dict):
                    sc = s.get("score")
                    if isinstance(sc, (int, float)):
                        return int(sc)
            return 0
        return sorted((n for n in parts if isinstance(n, str)), key=lambda n: _pimp(parts.get(n)), reverse=True)

    def _build_retrieval_anchors(
        self,
        context: str,
        last_x_messages: list[str],
        current_scene: dict | None,
        n: int,
    ) -> list[str]:
        """Build up to ``n`` short, POINTED, delta-framed retrieval anchor strings.

        Every input is already computed deterministically each turn, so this needs no
        model. Each anchor is a high-relevance subset (an event + its scoped cast, a
        mentioned entity + its recent state, the current-scene cast, or the stale
        roster) framed as "what is new/changing within <scope>" — the DELTA, never a
        re-surfacing of state already carried by general_info, the chapter/arc digests,
        and the entity graph. A bare entity name is too broad; an entity plus its event
        bounds is the narrowest honest unit. Each string is capped at
        _ANCHOR_CHAR_BUDGET so it stays inside the embedder's 384-token window.
        """
        anchors: list[str] = []
        seen: set[str] = set()

        def _push(text: str) -> None:
            t = " ".join(str(text).split())
            if not t:
                return
            key = t.casefold()
            if key in seen:
                return
            seen.add(key)
            anchors.append(t[: self._ANCHOR_CHAR_BUDGET])

        # 1) Event anchors: high-importance events first (the user's event-bounded
        #    instinct, in embedding form). Each = event title + its scoped cast +
        #    "what is new/developing within it".
        ev_flat: list[tuple[str, dict]] = []
        for bucket in ("past", "scenes", "events", "chapters"):
            bd = self._get_field_value(self.events, "Event", bucket, {}) or {}
            if isinstance(bd, dict):
                ev_flat.extend((nm, d) for nm, d in bd.items() if isinstance(d, dict))
        ev_flat.sort(key=lambda t: self._event_importance(t[1]), reverse=True)
        for _nm, data in ev_flat:
            if len(anchors) >= n:
                break
            title = data.get("name") or _nm
            cast = self._event_participants(data)[:5]
            if cast:
                _push(f"{title} — {', '.join(cast)}: what is new or developing within this event")
            else:
                _push(f"{title}: what is new or developing within this event")

        # 2) Mentioned-entity anchors: entities actually mentioned in the recent
        #    dialogue, each paired with its freshest recent state (the delta).
        recent_text = "\n".join(last_x_messages or [])
        mentioned = self._extract_entity_names(self._entity_patterns.get("Character", {}), self._entity_alias_maps.get("Character", {}), recent_text)
        if not mentioned:
            mentioned = self._extract_entity_names(self._entity_patterns.get("Character", {}), self._entity_alias_maps.get("Character", {}), context)
        recent_state = self._recent_state_map(mentioned)
        for name in mentioned:
            if len(anchors) >= n:
                break
            rs = recent_state.get(name)
            if rs:
                _push(f"{name}: what has changed or is developing relative to — {rs}")
            else:
                _push(f"{name}: what is new or changing")

        # 3) Current-scene cast anchor: who is present now + what is shifting between
        #    them (the live scene, the highest-salience delta).
        if isinstance(current_scene, dict):
            who = (current_scene.get("now") or {}).get("who") or {}
            _raw_names: list[str] = []
            chars = who.get("characters")
            if isinstance(chars, dict):
                _raw_names = [c.get("name") for c in chars.values() if isinstance(c, dict) and c.get("name")]
            elif isinstance(chars, list):
                _raw_names = [c.get("name") for c in chars if isinstance(c, dict) and c.get("name")]
            # Collapse alias-duplicate display names (multiple distinct graph nodes can
            # share a display name) so the anchor isn't padded with repeats. Order is
            # preserved and nothing in the store is mutated — only the anchor string.
            cast_names: list[str] = []
            _seen_names: set[str] = set()
            for c in _raw_names:
                if c and c not in _seen_names:
                    _seen_names.add(c)
                    cast_names.append(c)
            if cast_names:
                _push(f"current scene: {', '.join(cast_names[:6])} — what is shifting or developing among them")

        # 4) Stale-roster anchor: entities that have gone quiet — what has changed for
        #    them since they left the active cast (catches off-screen developments).
        stale = (getattr(self, "_last_stale_entities", None) or set())
        for name in sorted(stale):
            if len(anchors) >= n:
                break
            _push(f"{name}: what has changed since they were last active")

        # 5) Recency fallback: if we still have room, the freshest real exchange
        #    verbatim (never boilerplate) — a guaranteed non-collapsed anchor.
        if len(anchors) < n:
            for msg in reversed(last_x_messages or []):
                if len(anchors) >= n:
                    break
                _push(str(msg))

        return anchors

    def _fanout_query_messages(
        self,
        anchors: list[str],
        n_results: int = 5,
        current_context: str = "",
        last_x_messages: list[str] | None = None,
    ) -> tuple[list[str], list[dict]]:
        """Fan out over N short anchor queries, pool, dedup, and recency re-rank.

        Replaces the single collapsed-blob embed. Each anchor is queried independently
        (each a discriminative vector within the 384-token window); the pooled
        candidates are deduped by message_idx (summary node preferred) and re-ranked
        by the SAME recency banding as ``query_messages`` (recent band first, summary
        preferred, similarity as tiebreak). If no anchors are available, falls back to
        the legacy single-blob path so we never do worse than before.
        """
        anchors = [a for a in (anchors or []) if a and a.strip()]
        if not anchors:
            # No deterministic anchors available — legacy single-blob fallback.
            if current_context is not None:
                return self.query_messages(current_context, n_results=n_results)
            return [], []

        # Pool candidates across all anchors (per-anchor recency re-rank, then pool).
        per_anchor: list[tuple[str, dict]] = []
        for anchor in anchors:
            try:
                docs, metas = self.query_messages(anchor, n_results=n_results)
            except Exception:
                continue
            per_anchor.extend(zip(docs, metas))

        if not per_anchor:
            return [], []

        # Global dedup by message_idx (summary node preferred) + recency re-rank.
        def _idx(meta: dict) -> int:
            v = (meta or {}).get("message_idx")
            return v if isinstance(v, int) and v >= 0 else -1

        max_idx = max((_idx(m) for _d, m in per_anchor), default=-1)
        # Preserve first-seen (similarity) order as the final tiebreak.
        final: list[tuple[int, int, int, int, str, dict]] = []
        for pos, (doc, meta) in enumerate(per_anchor):
            idx = _idx(meta)
            if idx < 0:
                final.append((2, 0, 0, pos, str(doc), meta or {}))
            elif max_idx - idx <= _RAG_RECENCY_MESSAGES:
                final.append((0, -idx, -1 if (meta or {}).get("is_summary") else 0, pos, str(doc), meta or {}))
            else:
                final.append((1, -idx, -1 if (meta or {}).get("is_summary") else 0, pos, str(doc), meta or {}))
        final.sort(key=lambda t: (t[0], t[1], t[2], t[3]))

        chosen_messages: list[str] = []
        chosen_metadata: list[dict] = []
        seen_idx: set[int] = set()
        for _b, _neg, _sp, _p, doc, meta in final:
            idx = _idx(meta)
            if idx >= 0 and idx in seen_idx:
                continue
            if idx >= 0:
                seen_idx.add(idx)
            chosen_messages.append(doc)
            chosen_metadata.append(meta)
            if len(chosen_messages) >= n_results:
                break
        return chosen_messages, chosen_metadata

    def query_messages(self, query: str, n_results: int = 5) -> tuple[list[str], list[dict]]:
        """Query message chunks with semantic search, re-ranked for RECENCY.

        Raw similarity over a long history re-serves STALE beats (measured: the
        pact-hall beat stayed a top hit at consecutive checkpoints because the
        query re-describes the same subject matter). This fetches more candidates
        and re-ranks by how fresh each message is: messages within
        _RAG_RECENCY_MESSAGES of the newest outrank anything older; within a
        recency band, the raw similarity order is kept; summary nodes are
        preferred per message (they carry the compressed 'what happened').

        Returns:
            tuple of (messages, metadata)
        """
        if self.chunker is None:
            return [], []
        try:
            candidate_n = max(n_results * 6, 30)
            results = self.chunker.query_similar(query, n_results=candidate_n)
            docs = results.get("documents", []) or []
            metas = results.get("metadatas", []) or []
            if not docs:
                return [], []

            def _idx(meta: dict) -> int:
                v = meta.get("message_idx")
                return v if isinstance(v, int) and v >= 0 else -1

            max_idx = max((_idx(m) for m in metas), default=-1)
            # Similarity order preserved as a tiebreak (position-based, so no
            # score-polarity assumption across retriever backends).
            # Sort key: band (0 recent / 1 older / 2 unknown) -> newest idx
            # first -> summary-node preference -> original similarity position.
            final: list[tuple[int, int, int, int, str, dict]] = []
            for pos, (doc, meta) in enumerate(zip(docs, metas)):
                idx = _idx(meta)
                if idx < 0:
                    final.append((2, 0, 0, pos, str(doc), meta or {}))
                elif max_idx - idx <= _RAG_RECENCY_MESSAGES:
                    final.append((0, -idx, -1 if meta.get("is_summary") else 0, pos, str(doc), meta or {}))
                else:
                    final.append((1, -idx, -1 if meta.get("is_summary") else 0, pos, str(doc), meta or {}))
            final.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
            # Dedupe by message_idx, preferring the summary node for each index.
            chosen_messages: list[str] = []
            chosen_metadata: list[dict] = []
            seen_idx: set[int] = set()
            for _b, _neg, _sp, _p, doc, meta in final:
                idx = _idx(meta)
                if idx >= 0 and idx in seen_idx:
                    continue
                if idx >= 0:
                    seen_idx.add(idx)
                chosen_messages.append(doc)
                chosen_metadata.append(meta)
                if len(chosen_messages) >= n_results:
                    break
            return chosen_messages, chosen_metadata
        except Exception as e:
            print(f"{_WARNING}query_messages re-rank failed ({e}); falling back to raw similarity{_RESET}")
            try:
                results = self.chunker.query_similar(query, n_results=n_results)
                return (results.get("documents", []) or []), (results.get("metadatas", []) or [])
            except Exception:
                return [], []

    def _load_json(self, path: Path) -> dict:
        """Load and parse a JSON file."""
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return jsonc.load(f)
            except Exception as e:
                print(f"{_ERROR}Failed to load JSON file {path}: {e}{_RESET}")
                traceback.print_exc()
        return {}

    def get_current_scene(self) -> dict:
        """Get the current scene data."""
        return self._load_json(self.current_scene_path)

    def _get_character_important_relationships(self, char_name: str, importance_threshold: int = 75) -> dict[str, list[dict]]:
        """Get a character's important relationships."""
        characters_data = self._get_entries(self.characters, "Character")
        if char_name not in characters_data:
            return {}

        char_data = characters_data[char_name]
        rels = {}

        char_rels = self._get_field_value(char_data, "Character", "relationships")
        if char_rels:
            print(f"{_GRAY}relationships{_RESET}: {char_rels}")
            for related_char, rel_list in char_rels.items():
                # The dotted-name split bug (no [brackets] around "Mrs.
                # Arbuthnot") left rel_list as a dict keyed by the second part
                # ({"Arbuthnot": [...]}) or a bare string. Normalize those so one
                # malformed entry can never abort the whole subject scan.
                if isinstance(rel_list, dict):
                    rel_list = [r for v in rel_list.values() for r in (v if isinstance(v, list) else [v])]
                elif not isinstance(rel_list, list):
                    continue
                important_rels = [
                    rel for rel in rel_list
                    if isinstance(rel, dict)
                    and self._get_importance(rel, "Character", "relationships") >= importance_threshold
                ]
                if important_rels:
                    rels[related_char] = important_rels

        return rels

    def _get_character_scene_relationships(
        self, char1: str, char2: str, correlation_threshold: int = 0
    ) -> dict[str, list[dict]]:
        """Get relationships between two characters in the same scene, regardless of importance."""
        characters_data = self._get_entries(self.characters, "Character")

        # Try using graph first for bidirectional relationship lookup
        if hasattr(self, 'entity_graph') and self.entity_graph:
            bidir = self.entity_graph.get_bidirectional_relationship(char1, char2)

            rels = {}
            # Check forward relationship (char1 -> char2)
            if bidir["forward"]:
                rel_list = [{
                    "relation": bidir["forward"].relation,
                    "status": bidir["forward"].status,
                    "aliases": bidir["forward"].aliases,
                    "events": bidir["forward"].events,
                    "importance": {
                        "score": bidir["forward"].importance,
                        "reason": bidir["forward"].importance_reason,
                        "faction": bidir["forward"].faction
                    }
                }]
                if bidir["forward"].importance >= correlation_threshold:
                    rels[char2] = rel_list

            if rels:
                return rels

        # Fallback to JSON-based lookup
        if char1 not in characters_data or char2 not in characters_data:
            return {}

        char_data = characters_data[char1]
        rels = {}

        char_rels = self._get_field_value(char_data, "Character", "relationships")
        if char_rels and char2 in char_rels:
            rel_list = char_rels[char2]
            scene_rels = [rel for rel in rel_list if self._get_importance(rel, "Character", "relationships") >= correlation_threshold]
            if scene_rels:
                rels[char2] = scene_rels

        return rels

    def _get_all_relevant_character_relationships(
        self,
        scene_characters: list[str],
        importance_threshold: int = 75,
        correlation_threshold: int = 0,
    ) -> dict[str, dict]:
        """Get all relevant relationships for characters, including both important and scene-based relationships."""
        result = {}
        processed_chars = []
        characters_data = self._get_entries(self.characters, "Character")

        # First pass: Get important relationships for all characters
        for char_name in scene_characters:
            if char_name in characters_data:
                char_data = copy.deepcopy(characters_data[char_name])  # Deep copy to avoid modifying original

                # Get important relationships
                if hasattr(self, 'entity_graph') and self.entity_graph:
                    graph_rels = self.entity_graph.get_important_relationships(char_name, importance_threshold)
                    # Milestone rows (field_name='milestones') are EVENT links,
                    # not person-to-person relationships; rendering them under a
                    # "Relationships ---" label mislabels scene titles as bonds.
                    graph_rels = [r for r in (graph_rels or [])
                                  if getattr(r, 'field_name', 'relationships') == 'relationships']
                    if graph_rels:
                        important_rels = {}
                        for rel in graph_rels:
                            target = rel.target_id
                            normalized_target = target.split(":", 1)[1] if ":" in target else target
                            if normalized_target not in important_rels:
                                important_rels[normalized_target] = []
                            important_rels[normalized_target].append({
                                "relation": rel.relation,
                                "status": rel.status,
                                "aliases": rel.aliases,
                                "events": rel.events,
                                "importance": {
                                    "score": rel.importance,
                                    "reason": rel.importance_reason,
                                    "faction": rel.faction
                                }
                            })
                    else:
                        important_rels = self._get_character_important_relationships(char_name, importance_threshold)
                else:
                    important_rels = self._get_character_important_relationships(char_name, importance_threshold)

                char_rels = self._get_field_value(char_data, "Character", "relationships")
                if not char_rels:
                    char_rels = {}
                # Create new dict instead of updating in place
                new_char_rels = dict(char_rels)
                new_char_rels.update(important_rels)
                char_data["relationships"] = new_char_rels

                # Attach the milestone/event links that were just filtered out
                # of relationships as their OWN field so character_list can
                # render them under a truthful label.
                if not char_data.get("relevant_milestones"):
                    ms: list[dict] = []
                    if hasattr(self, 'entity_graph') and self.entity_graph:
                        try:
                            ms = self.entity_graph.get_character_milestones(
                                char_name, min_importance=importance_threshold
                            ) or []
                        except Exception:
                            ms = []
                    char_data["relevant_milestones"] = ms[:6]
                result[char_name] = char_data

                for related_char in important_rels:
                    if related_char in characters_data and related_char not in scene_characters:
                        scene_characters.append(related_char)

        for char1 in scene_characters:
            for char2 in scene_characters:
                if char1 != char2 and char1 in result:
                    scene_rels = self._get_character_scene_relationships(char1, char2, correlation_threshold)

                    result_rels = self._get_field_value(result[char1], "Character", "relationships")
                    if scene_rels and char2 not in (result_rels or {}):
                        if not result_rels:
                            result_rels = {}
                        result_rels.update(scene_rels)
                        result[char1]["relationships"] = result_rels

        return {"entries": result}

    def _unified_entity_aggregation(
        self,
        initial_entities: dict[str, set[str]],
        all_data: dict[str, dict],
        importance_threshold: int = 75,
        max_depth: int = 10,
        decay_config: DecayConfig | None = None,
    ) -> dict[str, dict[str, float]]:
        if not getattr(self, 'entity_graph', None):
            fallback: dict[str, dict[str, float]] = {"character": {}, "group": {}, "event": {}}
            for etype, names in initial_entities.items():
                key = etype.lower()
                if key in fallback:
                    for name in names:
                        fallback[key][name] = float(importance_threshold)
            return fallback

        if decay_config is None:
            decay_config = DecayConfig()

        field_map = self.entity_graph.get_schema_relationship_map()
        if not field_map:
            field_map = {
                "character": {
                    "relationships": [{"target_type": "character", "direction": "both"}],
                    "group_status": [{"target_type": "group", "direction": "outgoing"}],
                    "milestones": [{"target_type": "event", "direction": "outgoing"}],
                },
                "group": {
                    "characters": [{"target_type": "character", "direction": "outgoing"}],
                    "relationships": [{"target_type": "group", "direction": "both"}],
                },
                "element": {
                    "relationships": [{"target_type": "character", "direction": "both"}],
                    "events": [{"target_type": "event", "direction": "outgoing"}],
                },
            }

        graph_initial: dict[str, set[str]] = {}
        for etype, names in initial_entities.items():
            if etype.lower() in ("character", "group", "event"):
                graph_initial[etype.lower()] = names

        initial_scene_entities: set[str] = set()
        for etype, names in graph_initial.items():
            for name in names:
                initial_scene_entities.add(f"{etype}:{name}")

        # ---- Pass 1: Broad Discovery ----
        init_rel: dict[str, float] = {}
        for etype, names in graph_initial.items():
            for name in names:
                init_rel[f"{etype}:{name}"] = float(decay_config.final_threshold)
        scoring, path_records = self.entity_graph.traverse_graph_detailed(
            graph_initial,
            initial_relevance=init_rel,
            field_map=field_map,
            decay_config=decay_config,
        )

        entity_relevance: dict[str, dict[str, float]] = {
            etype: {} for etype in scoring
        }

        def _apply_convergence(source_map: dict[str, dict[str, float]]):
            source_relevance: dict[str, float] = {}
            for etype, entities in source_map.items():
                for name, rel in entities.items():
                    source_relevance[f"{etype}:{name}"] = rel

            for etype, entities in source_map.items():
                for name, base_rel in entities.items():
                    record = path_records.get(f"{etype}:{name}")
                    if record is None:
                        entity_relevance[etype][name] = base_rel
                        continue

                    n_paths = len(record.source_ids)
                    if n_paths <= 1:
                        entity_relevance[etype][name] = base_rel
                        continue

                    total_vouch = 0.0
                    weighted_sum = 0.0
                    for source_id in record.source_ids:
                        src_rel = source_relevance.get(source_id, 50.0)
                        edge = self.entity_graph.get_edge(source_id, f"{etype}:{name}")
                        if edge is None:
                            continue
                        vouch = (src_rel / 100.0) * (edge.importance / 100.0)
                        total_vouch += vouch
                        weighted_sum += src_rel * vouch

                    if total_vouch == 0:
                        entity_relevance[etype][name] = base_rel
                        continue

                    pull_target = weighted_sum / total_vouch
                    pull_strength = min(1.0, total_vouch * decay_config.pull_rate)

                    relevance = base_rel + (pull_target - base_rel) * pull_strength
                    entity_relevance[etype][name] = relevance

        # ---- Pass 2: Convergence Scoring ----
        if decay_config.convergence_enabled:
            _apply_convergence(scoring)
        else:
            for etype, entities in scoring.items():
                for name, base_rel in entities.items():
                    entity_relevance[etype][name] = base_rel

        # ---- Pass 3: Re-traversal (optional) ----
        if decay_config.re_traverse_enabled:
            new_seeds: dict[str, float] = {}
            for etype, entities in entity_relevance.items():
                for name, rel in entities.items():
                    named_id = f"{etype}:{name}"
                    if rel >= decay_config.re_traverse_threshold and named_id not in initial_scene_entities:
                        new_seeds[named_id] = rel

            for seed_id, seed_rel in new_seeds.items():
                seed_type = seed_id.split(":")[0]
                seed_name = seed_id.split(":", 1)[1]
                for entries in field_map.get(seed_type, {}).values():
                    for entry in entries:
                        target_type = entry["target_type"]
                        nbrs = self.entity_graph.get_neighbors_decayed(
                            seed_name, seed_type,
                            source_relevance=seed_rel,
                            target_type=target_type,
                            decay_config=decay_config,
                            direction=entry["direction"],
                        )
                        for nbr in nbrs:
                            nbr_name = nbr["name"]
                            edge_imp = nbr["importance"]
                            if edge_imp < decay_config.re_traverse_min_importance:
                                continue
                            if nbr_name not in entity_relevance.get(target_type, {}):
                                if target_type not in entity_relevance:
                                    entity_relevance[target_type] = {}
                                path_min = min(seed_rel, float(edge_imp))
                                entity_relevance[target_type][nbr_name] = path_min

            # Note: _apply_convergence here uses the original path_records from Pass 1,
            # which don't include paths discovered during re-traversal. This means
            # convergence only reflects pre-re-traversal paths — intentional since
            # re-traversal is a secondary discovery pass using single-edge P4 decay.
            # If path_records were updated, re-run convergence would strengthen
            # entities with newly discovered paths.
            if decay_config.convergence_enabled:
                _apply_convergence(entity_relevance)

        # ---- Pass 4: Pruning ----
        final_threshold = decay_config.final_threshold
        pruned: dict[str, dict[str, float]] = {}
        for etype, entities in entity_relevance.items():
            scored = {
                name: rel
                for name, rel in entities.items()
                if rel >= final_threshold
            }
            if scored:
                pruned[etype] = scored

        return pruned

    def _get_character_group_status(
        self,
        char_name: str,
        current_scene: dict,
        importance_threshold: int = 75,
    ) -> dict[str, dict]:
        """Get a character's group statuses meeting Condition A (high importance) OR Condition B (in current scene).

        Args:
            char_name: Name of the character
            current_scene: Current scene data for scene matching
            importance_threshold: Minimum importance score for Condition A

        Returns:
            Dict of status_name -> status_data
        """
        characters_data = self._get_entries(self.characters, "Character")
        if char_name not in characters_data:
            return {}

        char_data = characters_data[char_name]
        group_status = self._get_field_value(char_data, "Character", "group_status")
        if not group_status:
            fallback_status = self._get_field_value(char_data, "Character", "status")
            group_status = fallback_status if fallback_status else {}

        if not group_status:
            return {}

        result = {}
        current_scene_what = current_scene.get("what", "") if current_scene else ""
        current_scene_characters = []
        if current_scene and "who" in current_scene.get("now", {}):
            current_scene_characters = [
                c["name"] for c in current_scene["now"]["who"].get("characters", [])
            ]

        for status_name, status_data in group_status.items():
            include_status = False

            importance_score = self._get_importance(status_data, "Character", "group_status")

            if importance_score >= importance_threshold:
                include_status = True
            elif status_name in current_scene_characters or re.search(status_name, current_scene_what, re.IGNORECASE):
                include_status = True
            else:
                events = self._get_field_value(status_data, "Character", "events", [])
                if events:
                    for event_name in events:
                        if re.search(event_name, current_scene_what, re.IGNORECASE):
                            include_status = True
                            break

            if include_status:
                result[status_name] = status_data

        return result

    def _get_character_milestones(
        self,
        char_name: str,
        current_scene: dict,
        importance_threshold: int = 75,
    ) -> list[dict]:
        """Get a character's milestones meeting Condition A (high importance) OR Condition B (in current scene).

        Args:
            char_name: Name of the character
            current_scene: Current scene data for scene matching
            importance_threshold: Minimum importance score for Condition A

        Returns:
            List of milestone dicts
        """
        characters_data = self._get_entries(self.characters, "Character")
        if char_name not in characters_data:
            return []

        char_data = characters_data[char_name]
        milestones = self._get_field_value(char_data, "Character", "milestones", [])

        if not milestones:
            return []

        result = []
        current_scene_number = current_scene.get("_scene_number") if current_scene else None
        current_scene_what = current_scene.get("what", "") if current_scene else ""

        for milestone in milestones:
            include_milestone = False

            importance_score = self._get_importance(milestone, "Character", "milestones")

            if importance_score >= importance_threshold:
                include_milestone = True
            elif current_scene_number:
                milestone_scenes = self._get_field_value(milestone, "Character", "scenes", [])
                if current_scene_number in milestone_scenes:
                    include_milestone = True

            if not include_milestone:
                milestone_title = self._get_field_value(milestone, "Character", "title", "")
                if re.search(milestone_title, current_scene_what, re.IGNORECASE):
                    include_milestone = True

            if include_milestone:
                result.append(milestone)

        return result

    def _get_character_milestones_from_graph(
        self,
        char_name: str,
        importance_threshold: int = 75,
        current_scene: int | None = None
    ) -> list[dict]:
        """Get character milestones from entity graph with schema-driven filtering.

        This method uses the entity graph instead of hardcoded JSON access,
        leveraging the schema's relationship_format for field resolution.

        Args:
            char_name: Name of the character
            importance_threshold: Minimum importance score
            current_scene: Optional scene number for filtering

        Returns:
            List of milestone dicts from graph
        """
        if not hasattr(self, 'entity_graph') or not self.entity_graph:
            return []

        return self.entity_graph.get_character_milestones(
            char_name,
            importance_threshold=importance_threshold,
            current_scene=current_scene
        )

    def _get_character_group_status_from_graph(
        self,
        char_name: str,
        importance_threshold: int = 75,
        current_scene: int | None = None
    ) -> dict:
        """Get character group status from entity graph with schema-driven filtering.

        Args:
            char_name: Name of the character
            importance_threshold: Minimum importance score
            current_scene: Optional scene number for filtering

        Returns:
            Dict mapping group name to status data from graph
        """
        if not hasattr(self, 'entity_graph') or not self.entity_graph:
            return {}

        relationships = self.entity_graph.get_relationships_by_field(
            char_name,
            field_name="group_status",
            min_importance=importance_threshold,
            current_scene=current_scene
        )

        result = {}
        for r in relationships:
            group_name = r.target_id.split(":", 1)[1] if ":" in r.target_id else r.target_id
            result[group_name] = {
                "position": [r.relation],
                "importance": {"score": r.importance, "reason": r.importance_reason, "faction": r.faction},
            }

        return result

    def _get_entries(self, data: dict, subject_type: str) -> dict:
        """Get entries from data dict using schema-driven key resolution.

        Args:
            data: The data dict (e.g., self.characters)
            subject_type: Subject type (e.g., "Character", "Group")

        Returns:
            The entries dict or the data itself if no entries wrapper
        """
        if self.schema_wrapper:
            entity_type = subject_type.capitalize()
            all_fields = self.schema_wrapper.get_entity_fields(entity_type)
            if "entries" in all_fields or "entries" in data:
                return data.get("entries", data)
        return data.get("entries", data)

    def _get_field_value(self, entity_data: dict, entity_type: str, field_name: str, default=None):
        """Get a field value from entity data using schema-driven resolution.

        Args:
            entity_data: The entity's data dict
            entity_type: Entity type (e.g., "Character", "Group")
            field_name: Field name
            default: Default value

        Returns:
            Field value or default
        """
        if self.schema_wrapper:
            return self.schema_wrapper.get_field_value(entity_data, entity_type, field_name, default)
        return entity_data.get(field_name, default)

    def _get_nested_field_value(self, entity_data: dict, entity_type: str, path: str, default=None):
        """Get a nested field value using schema-driven dot notation resolution.

        Args:
            entity_data: The entity's data dict
            entity_type: Entity type (e.g., "Character")
            path: Dot-separated path (e.g., "group_status.Rebel Force.importance.score")
            default: Default value

        Returns:
            Value at path or default
        """
        if self.schema_wrapper:
            return self.schema_wrapper.get_nested_field_value(entity_data, entity_type, path, default)
        from ...utils.helpers import split_keys_to_list
        parts = split_keys_to_list(path)
        current = entity_data
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return default
        return current if current is not None else default

    def _get_importance(self, item_data: dict, item_type: str, field_name: str) -> int:
        """Get importance score for an item using schema-driven path resolution.

        Args:
            item_data: The item's data dict (e.g., milestone, status)
            item_type: Parent entity type (e.g., "Character")
            field_name: Field name (e.g., "milestones", "group_status")

        Returns:
            Importance score (0-100) or 0 if not found
        """
        if not isinstance(item_data, dict):
            return 0
        if self.schema_wrapper:
            score = self._get_nested_field_value(item_data, item_type, "importance.score", None)
            if isinstance(score, int):
                return score
            if isinstance(score, dict):
                return score.get("score", 0)
            if score is None:
                path = f"{field_name}.importance.score"
                score = self._get_nested_field_value(item_data, item_type, path, None)
                if isinstance(score, int):
                    return score
                if isinstance(score, dict):
                    return score.get("score", 0)
        importance = item_data.get("importance", {})
        if isinstance(importance, dict):
            return importance.get("score", 0)
        return importance if isinstance(importance, int) else 0

    def _get_all_relevant_status_and_milestones(
        self,
        initial_characters: list[str],
        all_groups: dict[str, dict],
        all_characters: dict[str, dict],
        all_events: dict[str, dict],
        current_scene: dict,
        importance_threshold: int = 75,
        max_depth: int = 10,
    ) -> tuple[dict[str, dict], dict[str, list[dict]]]:
        """Get all relevant statuses and milestones using unified entity aggregation.

        Uses _unified_entity_aggregation to get the complete pool of relevant
        characters and groups, then extracts statuses and milestones from that pool.

        Args:
            initial_characters: Initial list of characters in the scene
            all_groups: Full groups data dict
            all_characters: Full characters data dict
            all_events: Full events data dict
            current_scene: Current scene data
            importance_threshold: Minimum importance score
            max_depth: Maximum recursion depth

        Returns:
            Tuple of (character_status dict, character_milestones dict)
        """
        initial_entities = {"Character": set(initial_characters)}
        all_data = {
            "Character": all_characters,
            "Group": all_groups,
            "Event": all_events,
        }

        relevant = self._unified_entity_aggregation(
            initial_entities, all_data,
            importance_threshold=importance_threshold,
            max_depth=max_depth,
        )

        char_key = "character"
        relevant_chars = relevant.get(char_key, {}).keys()
        characters_data = self._get_entries(all_characters, "Character")

        result_status = {}
        result_milestones = {}

        for char_name in relevant_chars:
            if char_name not in characters_data:
                continue

            current_scene_number = current_scene.get("_scene_number") if current_scene else None
            group_status = self._get_character_group_status_from_graph(
                char_name,
                importance_threshold=importance_threshold,
                current_scene=current_scene_number
            )
            if not group_status:
                group_status = self._get_character_group_status(char_name, current_scene, importance_threshold)
            if group_status:
                result_status[char_name] = group_status

            milestones = self._get_character_milestones_from_graph(
                char_name,
                importance_threshold=importance_threshold,
                current_scene=current_scene_number
            )
            if not milestones:
                milestones = self._get_character_milestones(char_name, current_scene, importance_threshold)
            if milestones:
                result_milestones[char_name] = milestones

        return result_status, result_milestones

    def retrieve_context(self, current_context: str, last_x_messages: list[str]) -> RetrievalContext:
        """Main method to retrieve all relevant context based on current state."""
        result = RetrievalContext(subjects={"general_info": self.general_info})
        current_scene = self.get_current_scene()
        result.current_scene = current_scene

        # Get characters from current scene and context
        scene_characters = []
        if current_scene and "who" in current_scene.get("now", {}):
            for char in current_scene["now"]["who"].get("characters", []):
                if not isinstance(char, dict) or not char.get("name"):
                    continue
                print(char["name"], "-", scene_characters)
                if char["name"] not in scene_characters:
                    scene_characters.append(char["name"])

        # Add characters mentioned in context and last messages
        context_to_search = current_context + "\n" + "\n".join(last_x_messages)
        mentioned_characters = self._extract_entity_names(self._entity_patterns.get("Character", {}), self._entity_alias_maps.get("Character", {}), context_to_search)
        for char in mentioned_characters:
            if char not in scene_characters:
                scene_characters.append(char)

        # Semantic fallback: catch characters referred to descriptively
        # that the regex stage missed (e.g. "the security captain" → "Captain Kael")
        _all_char_names = list(self._get_entries(self.characters, "Character").keys())
        _semantic_chars = self._semantic_entity_fallback(
            context_to_search, set(scene_characters), _all_char_names
        )
        for char in _semantic_chars:
            if char not in scene_characters:
                scene_characters.append(char)
        if _semantic_chars:
            print(f"{_DEBUG}semantic entity fallback: +{len(_semantic_chars)} characters {_semantic_chars}{_RESET}")

        # Get elements from current scene
        scene_elements = []
        if current_scene and "who" in current_scene.get("now", {}):
            for element in current_scene["now"]["who"].get("elements", []):
                if isinstance(element, dict) and element.get("name") and element["name"] not in scene_elements:
                    scene_elements.append(element["name"])

        # Add elements mentioned in context and last messages
        mentioned_elements = self._extract_entity_names(self._entity_patterns.get("Element", {}), self._entity_alias_maps.get("Element", {}), context_to_search)
        for element in mentioned_elements:
            if element not in scene_elements:
                scene_elements.append(element)

        # Semantic fallback for elements (same rationale as characters)
        _all_elem_names = list(self.elements.get("entries", self.elements).keys())
        _semantic_elems = self._semantic_entity_fallback(
            context_to_search, set(scene_elements), _all_elem_names
        )
        for element in _semantic_elems:
            if element not in scene_elements:
                scene_elements.append(element)
        if _semantic_elems:
            print(f"{_DEBUG}semantic entity fallback: +{len(_semantic_elems)} elements {_semantic_elems}{_RESET}")

        # Get groups from current scene
        scene_groups = []
        if current_scene and "who" in current_scene.get("now", {}):
            for grp in current_scene["now"]["who"].get("groups", []):
                if isinstance(grp, dict) and grp.get("name") and grp["name"] not in scene_groups:
                    scene_groups.append(grp["name"])
                elif isinstance(grp, str) and grp and grp not in scene_groups:
                    scene_groups.append(grp)

        # Dedup guards make this a no-op for subjects already fully covered
        # (Character: text+semantic); it adds new signals for the rest.
        if self.retrieval_order:
            _anchor_map: dict[str, set[str]] = {
                "Character": set(scene_characters),
                "Element": set(scene_elements),
                "Group": set(scene_groups),
            }
            for _st in self.retrieval_order:
                _st_cap = _st.capitalize()
                _rel = self._get_relevant_entries(_st_cap, _anchor_map, context_to_search)
                if not _rel:
                    continue
                _names = set(_rel.keys())
                if _st_cap == "Character":
                    for n in _names:
                        if n not in scene_characters:
                            scene_characters.append(n)
                    _anchor_map["Character"] = set(scene_characters)
                elif _st_cap == "Element":
                    for n in _names:
                        if n not in scene_elements:
                            scene_elements.append(n)
                    _anchor_map["Element"] = set(scene_elements)
                elif _st_cap == "Group":
                    for n in _names:
                        if n not in scene_groups:
                            scene_groups.append(n)
                    _anchor_map["Group"] = set(scene_groups)
                if _names:
                    print(f"{_DEBUG}relevance dispatcher: {_st_cap} +{len(_names)} {sorted(_names)}{_RESET}")

        try:
            print(f"{_DEBUG}retrieve_context try block starting. general_info type: {type(result.subjects.get('general_info', {}))}, is empty: {not result.subjects.get('general_info', {})}{_RESET}")
            print(f"{_DEBUG}scene_characters to look up: {scene_characters}{_RESET}")
            print(f"{_DEBUG}scene_elements to look up: {scene_elements}{_RESET}")
            print(f"{_DEBUG}self.characters keys: {list(self.characters.keys()) if self.characters else 'empty'}{_RESET}")

            initial_entities = {"Character": set(scene_characters), "Element": set(scene_elements)}
            all_data = {
                "Character": self.characters,
                "Group": self.groups,
                "Event": self.events,
                "Element": self.elements,
            }
            decay_config = DecayConfig()
            unified_result = self._unified_entity_aggregation(
                initial_entities, all_data,
                importance_threshold=decay_config.final_threshold,
                max_depth=decay_config.max_depth,
                decay_config=decay_config,
            )
            result.relevant_entities = unified_result
            char_key = "character"
            group_key = "group"
            event_key = "event"
            element_key = "element"
            unified_chars = list(unified_result.get(char_key, {}).keys())
            unified_groups = list(unified_result.get(group_key, {}).keys())
            unified_events = list(unified_result.get(event_key, {}).keys())
            unified_elements = unified_result.get(element_key, {})
            print(f"{_DEBUG}unified aggregation: {len(unified_chars)} chars, {len(unified_groups)} groups, {len(unified_events)} events, {len(unified_elements)} elements{_RESET}")

            def _safe_subject(label: str, fn):
                try:
                    return fn()
                except Exception as e:
                    print(f"{_ERROR}retrieve_context: {label} extraction failed ({type(e).__name__}: {e}); continuing with partial subject.{_RESET}")
                    traceback.print_exc()
                    return None

            # Scene characters are always included; graph adds extras
            base_chars = [c for c in scene_characters if c]
            extra_chars = [c for c in unified_chars if c not in base_chars]
            all_chars = base_chars + extra_chars
            result.subjects["characters"] = _safe_subject(
                "characters",
                lambda: self._get_all_relevant_character_relationships(all_chars),
            ) or {}
            print(f"{_DEBUG}characters retrieved: type={type(result.subjects.get('characters', {})).__name__}, count: {len(result.subjects.get('characters', {}).get('entries', {})) if result.subjects.get('characters') else 0}{_RESET}")
            # Scene groups are always included; graph adds extras
            base_groups = [g for g in scene_groups if g]
            extra_groups = [g for g in unified_groups if g not in base_groups]
            all_groups = base_groups + extra_groups
            groups_entries = self._get_entries(self.groups, "Group")
            result.subjects["groups"] = _safe_subject(
                "groups",
                lambda: {"entries": {g: groups_entries.get(g, {}) for g in all_groups}},
            ) or {}
            print(f"{_DEBUG}groups retrieved: type={type(result.subjects.get('groups', {})).__name__}, count: {len(result.subjects.get('groups', {})) if result.subjects.get('groups') else 0}{_RESET}")
            _MAX_CONTEXT_ELEMENTS = 25
            def _graph_scored_elements():
                entries = self.elements.get("entries", {})
                # Scene elements are always included (they're directly present)
                base = {n for n in scene_elements if n in entries}
                # Graph-scored elements add relevance beyond the scene
                extras = []
                if unified_elements:
                    ranked = sorted(unified_elements.items(), key=lambda x: x[1], reverse=True)
                    extras = [n for n, _ in ranked if n not in base and n in entries]
                # Scene elements first, then graph-scored extras, capped
                selected = list(base) + extras[:_MAX_CONTEXT_ELEMENTS - len(base)]
                if not selected:
                    return {"entries": {}}
                return {"entries": {n: entries[n] for n in selected}}
            result.subjects["elements"] = _safe_subject(
                "elements",
                _graph_scored_elements,
            ) or {}
            print(f"{_DEBUG}elements retrieved: type={type(result.subjects.get('elements', {})).__name__}, count: {len(result.subjects.get('elements', {}).get('entries', {})) if result.subjects.get('elements') else 0}{_RESET}")

            scenes = self._get_field_value(self.events, "Event", "scenes", {}) or {}
            events = self._get_field_value(self.events, "Event", "events", {}) or {}
            past = self._get_field_value(self.events, "Event", "past", {}) or {}
            events_dict = {**scenes, **events, **past}
            events_selected = _safe_subject(
                "events",
                lambda: self._select_relevant_events(
                    context_to_search, scene_characters,
                    result.subjects.get('groups', {}), current_scene,
                ),
            ) or {}
            # Union with the graph-aggregation picks so edges never drop a
            # relevant stored event, but the data-driven selection is primary.
            if isinstance(events_selected, dict):
                cat_of: dict[str, str] = {}
                for _cat, _src in (("scenes", scenes), ("events", events), ("past", past)):
                    if isinstance(_src, dict):
                        for k in _src:
                            cat_of[k] = _cat
                for e in unified_events:
                    if e in events_selected.get("entries", {}) or e not in events_dict or not isinstance(events_dict[e], dict):
                        continue
                    _cat = cat_of.get(e)
                    if not _cat:
                        continue
                    events_selected.setdefault(_cat, {})[e] = events_dict[e]
                    events_selected["entries"][e] = events_dict[e]
            result.subjects["events"] = events_selected
            # Uncapped bucketed view for boundary/scene-key consumers (the capped
            # ``events`` block must not lose far-away archived scenes to them).
            full_chapters = self._get_field_value(self.events, "Event", "chapters", {}) or {}
            result.events_full = {
                "past": past, "scenes": scenes, "events": events,
                "chapters": full_chapters if isinstance(full_chapters, list) else (list(full_chapters.values()) if isinstance(full_chapters, dict) else []),
            }
            print(f"{_DEBUG}events retrieved: selected={len(events_selected.get('entries', {}))}/{len(events_dict)} ({list(events_selected.get('entries', {}))[:3]}...){_RESET}")

            if self.arcs:
                result.subjects["arcs"] = _safe_subject("arcs", lambda: self.arcs) or {}
                print(f"{_DEBUG}arcs retrieved: type={type(result.subjects.get('arcs', {})).__name__}, count: {len(result.subjects.get('arcs', {})) if result.subjects.get('arcs') else 0}{_RESET}")
                  
            if self.knowledge:
                result.subjects["knowledge"] = _safe_subject("knowledge", lambda: self.knowledge) or {}
                print(f"{_DEBUG}knowledge retrieved: type={type(result.subjects.get('knowledge', {})).__name__}, count: {len(result.subjects.get('knowledge', {}).get('entries', result.subjects.get('knowledge', {}))) if result.subjects.get('knowledge') else 0}{_RESET}")

            chapters_data = self._get_field_value(self.events, "Event", "chapters", {})
            if chapters_data:
                result.subjects["chapters"] = _safe_subject("chapters", lambda: chapters_data) or {}
                print(f"{_DEBUG}chapters retrieved: type={type(result.subjects.get('chapters', {})).__name__}, count: {len(result.subjects.get('chapters', {})) if result.subjects.get('chapters') else 0}{_RESET}")

            # 3c: per-entity 'current state' annotation. Entries are copied so the
            # stored JSON is never polluted; each entry gains a `_recent_state`
            # line (rendered by the format templates) = the freshest story
            # fragment mentioning the entity.
            try:
                recent_names = [str(n) for n in scene_characters + scene_elements]
                recent_map = self._recent_state_map(recent_names)
                if recent_map:
                    for subject_key in ("characters", "elements", "groups"):
                        subj = result.subjects.get(subject_key, {})
                        entries = subj.get("entries") if isinstance(subj, dict) else None
                        if not isinstance(entries, dict):
                            continue
                        for ename, edata in list(entries.items()):
                            rs = recent_map.get(str(ename))
                            if rs and isinstance(edata, dict) and "_recent_state" not in edata:
                                edata = dict(edata)
                                edata["_recent_state"] = rs
                                entries[ename] = edata
            except Exception as _e:
                print(f"{_ERROR}recent-state annotation failed: {_e}{_RESET}")
                traceback.print_exc()

            result.character_status, result.character_milestones = self._get_all_relevant_status_and_milestones(
                scene_characters, self.groups, self.characters, self.events, current_scene
            )
            print(f"{_DEBUG}character_status retrieved: {type(result.character_status)}, count: {len(result.character_status) if result.character_status else 0}{_RESET}")
            print(f"{_DEBUG}character_milestones retrieved: {type(result.character_milestones)}, count: {len(result.character_milestones) if result.character_milestones else 0}{_RESET}")

            # Get messages using both retrieval methods
            # scene_messages = self._get_message_chunks()  # Index-based retrieval
            _anchors = self._build_retrieval_anchors(
                context_to_search, last_x_messages, current_scene, self.rag_anchor_count
            )
            semantic_messages, semantic_metadata = self._fanout_query_messages(
                _anchors, n_results=5, current_context=current_context, last_x_messages=last_x_messages
            )  # Semantic search (fan-out)

            # Combine and deduplicate messages
            all_messages = []
            all_metadata = []

            # # First add scene messages to maintain chronological order
            # for msg in scene_messages:
            #     if msg not in all_messages:
            #         all_messages.append(msg)

            # Then add semantically relevant messages
            for i, msg in enumerate(semantic_messages):
                if msg not in all_messages:
                    all_messages.append(msg)
                    all_metadata.append(semantic_metadata[i] if i < len(semantic_metadata) else {})

            result.messages = all_messages
            result.messages_metadata = all_metadata

        except Exception as e:
            print(f"{_ERROR}EXCEPTION in retrieve_context: {str(e)}{_RESET}")
            print(f"{_ERROR}general_info is: {result.subjects.get('general_info', {})}{_RESET}")
            traceback.print_exc()

        return result


from .message_chunker import MessageChunker  # re-export (summarizer imports it from here)
