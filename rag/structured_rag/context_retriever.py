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
    characters: dict[str, dict] = field(default_factory=dict)
    groups: dict[str, dict] = field(default_factory=dict)
    elements: dict[str, dict] = field(default_factory=dict)
    events: dict[str, dict] = field(default_factory=dict)
    chapters: dict[str, dict] = field(default_factory=dict)
    arcs: dict[str, dict] = field(default_factory=dict)
    general_info: dict[str, dict] = field(default_factory=dict)
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
    def __init__(self, history_path: PathLike, schema_classes: dict | None = None, summarizer: 'Summarizer' | None = None, rag_anchor_count: int | None = None):
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

        self.characters = self._load_json(self.characters_path)
        self.groups = self._load_json(self.groups_path)
        self.elements = self._load_json(self.elements_path)
        self.events = self._load_json(self.events_path)
        self.general_info = self._load_json(self.general_info_path)
        self.current_scene = self._load_json(self.current_scene_path)
        self.arcs = self._load_json(self.arcs_path)

        print(f"{_DEBUG}StoryContextRetriever loaded:")
        print(f"  history_path: {history_path}")
        print(f"  characters keys: {list(self.characters.keys()) if self.characters else 'empty'}")
        print(f"  groups keys: {list(self.groups.keys()) if self.groups else 'empty'}")
        print(f"  elements keys: {list(self.elements.keys()) if self.elements else 'empty'}")
        print(f"  events keys: {list(self.events.keys()) if self.events else 'empty'}")
        print(f"  general_info keys: {list(self.general_info.keys()) if self.general_info else 'empty'}")
        print(f"  current_scene keys: {list(self.current_scene.keys()) if self.current_scene else 'empty'}")
        print(f"  arcs count: {len(self.arcs) if self.arcs else 0}{_RESET}")

        # Initialize entity graph for relationship tracking
        self.schema_classes = schema_classes or {}
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
        self.chunker = MessageChunker(history_path, characters_map, groups_map, self.elements, self.events, self.current_scene, summarizer=summarizer, use_llm_for_speakers=self.use_llm_for_speakers)

        # Create character name patterns for recognition (from both JSON and graph)
        self.character_patterns = self._create_character_patterns()
        # Create element name patterns for recognition
        self.element_patterns = self._create_element_patterns()

        self.rag_anchor_count = int(rag_anchor_count) if rag_anchor_count else _RAG_ANCHOR_COUNT_DEFAULT

    def _create_character_patterns(self) -> dict[str, re.Pattern]:
        """Create regex patterns for character name recognition."""
        patterns = {}

        # Get characters from graph (primary source)
        char_names = []
        if hasattr(self, 'entity_graph') and self.entity_graph:
            graph_char_nodes = self.entity_graph.get_nodes_by_type("character")
            char_names = [node.name for node in graph_char_nodes]

        # Fallback to JSON if no graph characters
        if not char_names:
            characters_data = self._get_entries(self.characters, "Character")
            char_names = list(characters_data.keys())

        for char_name in char_names:
            # Create pattern that matches full name and possible first/last name only
            names = char_name.split()
            safe_full = re.escape(char_name)
            safe_first = re.escape(names[0])
            safe_last = re.escape(names[-1])
            pattern = f"({safe_full}"
            if len(names) > 1:
                pattern += f"|{safe_first}|{safe_last}"
            pattern += ")"
            patterns[char_name] = re.compile(pattern, flags=re.IGNORECASE)
        return patterns

    def _extract_character_names(self, text: str) -> list[str]:
        """Extract character names from text using regex patterns."""
        found_names = []
        for char_name, pattern in self.character_patterns.items():
            if pattern.search(text):
                if char_name not in found_names:
                    found_names.append(char_name)
        return found_names

    def _get_relevant_groups(self, characters: list[str], context: str) -> dict[str, dict]:
        """Get groups relevant to the current context and characters."""
        relevant_groups = {}
        groups_data = self._get_entries(self.groups, "Group")

        if hasattr(self, 'entity_graph') and self.entity_graph:
            graph_groups = self.entity_graph.get_relevant_groups(characters, context)
            for group_name in graph_groups:
                if group_name in groups_data:
                    relevant_groups[group_name] = groups_data[group_name]
        else:
            for group_name, group_data in groups_data.items():
                for char in characters:
                    group_chars = self._get_field_value(group_data, "Group", "characters", {})
                    if char in group_chars:
                        relevant_groups[group_name] = group_data
                        break

        for group_name, group_data in groups_data.items():
            if group_name in relevant_groups:
                continue
            group_aliases = self._get_field_value(group_data, "Group", "aliases", [])
            if re.search(group_name, context, flags=re.IGNORECASE) or any(
                alias for alias in group_aliases if re.search(alias, context, flags=re.IGNORECASE)
            ):
                relevant_groups[group_name] = group_data

        return {"entries": relevant_groups}

    def _create_element_patterns(self) -> dict[str, re.Pattern]:
        """Create regex patterns for element name recognition."""
        patterns = {}
        elements_data = self.elements.get("entries", self.elements)
        for element_name in elements_data:
            names = element_name.split()
            pattern = f"({re.escape(element_name)}"
            if len(names) > 1:
                pattern += f"|{re.escape(names[0])}|{re.escape(names[-1])}"
            pattern += ")"
            patterns[element_name] = re.compile(pattern, flags=re.IGNORECASE)
        return patterns

    def _extract_element_names(self, text: str) -> list[str]:
        """Extract element names from text using regex patterns."""
        found_names = []
        for element_name, pattern in self.element_patterns.items():
            if pattern.search(text):
                if element_name not in found_names:
                    found_names.append(element_name)
        return found_names

    def _get_relevant_elements(self, characters: list[str], context: str) -> dict[str, dict]:
        """Get elements relevant to the current context and characters."""
        relevant_elements = {}
        elements_data = self.elements.get("entries", {})

        for element_name, element_data in elements_data.items():
            # Check if element is mentioned in context
            if re.search(element_name, context, flags=re.IGNORECASE):
                relevant_elements[element_name] = element_data
                continue

            # Check if element is referenced via relationship to any character
            for char in characters:
                if "relationships" in element_data and isinstance(element_data["relationships"], dict):
                    if char in element_data["relationships"]:
                        relevant_elements[element_name] = element_data
                        break

        return {"entries": relevant_elements}

    def _get_relevant_events(self, characters: list[str], groups: dict[str, dict], context: str) -> dict[str, dict]:
        """Get events relevant to the current context and groups."""
        relevant_events = {}

        scenes = self._get_field_value(self.events, "Event", "scenes", {})
        if scenes:
            for scene_name, scene in scenes.items():
                if re.search(scene_name, context, flags=re.IGNORECASE):
                    relevant_events[scene_name] = scene
                    continue

                for group_data in groups.values():
                    group_events = self._get_field_value(group_data, "Group", "events", [])
                    if scene_name in group_events:
                        relevant_events[scene_name] = scene
                        break

        return relevant_events

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
        mentioned = self._extract_character_names(recent_text)
        if not mentioned:
            mentioned = self._extract_character_names(context)
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
                "character": {"relationships": "character", "group_status": "group", "milestones": "event"},
                "group": {"characters": "character", "relationships": "group", "events": "event"},
                "event": {"participants": "character"},
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
                for target_type in field_map.get(seed_type, {}).values():
                    nbrs = self.entity_graph.get_neighbors_decayed(
                        seed_name, seed_type,
                        source_relevance=seed_rel,
                        target_type=target_type,
                        decay_config=decay_config,
                        direction="outgoing",
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
        result = RetrievalContext(general_info=self.general_info)
        current_scene = self.get_current_scene()
        result.current_scene = current_scene

        # Get characters from current scene and context
        scene_characters = []
        if current_scene and "who" in current_scene.get("now", {}):
            for char in current_scene["now"]["who"].get("characters", []):
                print(char["name"], "-", scene_characters)
                if char["name"] not in scene_characters:
                    scene_characters.append(char["name"])

        # Add characters mentioned in context and last messages
        context_to_search = current_context + "\n" + "\n".join(last_x_messages)
        mentioned_characters = self._extract_character_names(context_to_search)
        for char in mentioned_characters:
            if char not in scene_characters:
                scene_characters.append(char)

        # Get elements from current scene
        scene_elements = []
        if current_scene and "who" in current_scene.get("now", {}):
            for element in current_scene["now"]["who"].get("elements", []):
                if isinstance(element, dict) and element.get("name") and element["name"] not in scene_elements:
                    scene_elements.append(element["name"])

        # Add elements mentioned in context and last messages
        mentioned_elements = self._extract_element_names(context_to_search)
        for element in mentioned_elements:
            if element not in scene_elements:
                scene_elements.append(element)

        try:
            print(f"{_DEBUG}retrieve_context try block starting. general_info type: {type(result.general_info)}, is empty: {not result.general_info}{_RESET}")
            print(f"{_DEBUG}scene_characters to look up: {scene_characters}{_RESET}")
            print(f"{_DEBUG}scene_elements to look up: {scene_elements}{_RESET}")
            print(f"{_DEBUG}self.characters keys: {list(self.characters.keys()) if self.characters else 'empty'}{_RESET}")

            initial_entities = {"Character": set(scene_characters)}
            all_data = {
                "Character": self.characters,
                "Group": self.groups,
                "Event": self.events,
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
            unified_chars = list(unified_result.get(char_key, {}).keys())
            unified_groups = list(unified_result.get(group_key, {}).keys())
            unified_events = list(unified_result.get(event_key, {}).keys())
            print(f"{_DEBUG}unified aggregation: {len(unified_chars)} chars, {len(unified_groups)} groups, {len(unified_events)} events{_RESET}")

            def _safe_subject(label: str, fn):
                try:
                    return fn()
                except Exception as e:
                    print(f"{_ERROR}retrieve_context: {label} extraction failed ({type(e).__name__}: {e}); continuing with partial subject.{_RESET}")
                    traceback.print_exc()
                    return None

            result.characters = _safe_subject(
                "characters",
                lambda: self._get_all_relevant_character_relationships(list(unified_chars)),
            ) or {}
            print(f"{_DEBUG}characters retrieved: type={type(result.characters).__name__}, count: {len(result.characters) if result.characters else 0}{_RESET}")
            groups_entries = self._get_entries(self.groups, "Group")
            result.groups = _safe_subject(
                "groups",
                lambda: {"entries": {g: groups_entries.get(g, {}) for g in unified_groups}},
            ) or {}
            print(f"{_DEBUG}groups retrieved: type={type(result.groups).__name__}, count: {len(result.groups) if result.groups else 0}{_RESET}")
            result.elements = _safe_subject(
                "elements",
                lambda: self._get_relevant_elements(scene_characters, context_to_search),
            ) or {}
            print(f"{_DEBUG}elements retrieved: type={type(result.elements).__name__}, count: {len(result.elements) if result.elements else 0}{_RESET}")

            scenes = self._get_field_value(self.events, "Event", "scenes", {}) or {}
            events = self._get_field_value(self.events, "Event", "events", {}) or {}
            past = self._get_field_value(self.events, "Event", "past", {}) or {}
            events_dict = {**scenes, **events, **past}
            events_selected = _safe_subject(
                "events",
                lambda: self._select_relevant_events(
                    context_to_search, scene_characters,
                    result.groups, current_scene,
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
            result.events = events_selected
            # Uncapped bucketed view for boundary/scene-key consumers (the capped
            # ``events`` block must not lose far-away archived scenes to them).
            full_chapters = self._get_field_value(self.events, "Event", "chapters", {}) or {}
            result.events_full = {
                "past": past, "scenes": scenes, "events": events,
                "chapters": full_chapters if isinstance(full_chapters, list) else (list(full_chapters.values()) if isinstance(full_chapters, dict) else []),
            }
            print(f"{_DEBUG}events retrieved: selected={len(events_selected.get('entries', {}))}/{len(events_dict)} ({list(events_selected.get('entries', {}))[:3]}...){_RESET}")

            if self.arcs:
                result.arcs = _safe_subject("arcs", lambda: self.arcs) or {}
                print(f"{_DEBUG}arcs retrieved: type={type(result.arcs).__name__}, count: {len(result.arcs) if result.arcs else 0}{_RESET}")

            chapters_data = self._get_field_value(self.events, "Event", "chapters", {})
            if chapters_data:
                result.chapters = _safe_subject("chapters", lambda: chapters_data) or {}
                print(f"{_DEBUG}chapters retrieved: type={type(result.chapters).__name__}, count: {len(result.chapters) if result.chapters else 0}{_RESET}")

            # 3c: per-entity 'current state' annotation. Entries are copied so the
            # stored JSON is never polluted; each entry gains a `_recent_state`
            # line (rendered by the format templates) = the freshest story
            # fragment mentioning the entity.
            try:
                recent_names = [str(n) for n in scene_characters + scene_elements]
                recent_map = self._recent_state_map(recent_names)
                if recent_map:
                    for subject_key in ("characters", "elements", "groups"):
                        subj = getattr(result, subject_key, None) or {}
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
            print(f"{_ERROR}general_info is: {result.general_info}{_RESET}")
            traceback.print_exc()

        return result


from .message_chunker import MessageChunker  # re-export (summarizer imports it from here)
