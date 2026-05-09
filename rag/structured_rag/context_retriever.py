from __future__ import annotations
from typing import TYPE_CHECKING, Any

import jsonc
import logging
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
import traceback
from os import PathLike

from ...utils.schema_parser import SchemaWrapper

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.indices.loading import load_index_from_storage

# HuggingFaceEmbedding will be background imported
from llama_index.core.settings import Settings
from llama_index.core.schema import TextNode


if TYPE_CHECKING:
    import nltk
    import spacy
    from spacy.tokens import Doc
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from extensions.dayna_ss_graph.agents.summarizer import Summarizer
else:
    nltk = None
    spacy = None
    Doc = None
    HuggingFaceEmbedding = None

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
start_background_import("llama_index.embeddings.huggingface", "HuggingFaceEmbedding")


@dataclass
class RetrievalContext:
    context: str = ""
    current_scene: dict[str, dict] = field(default_factory=dict)
    characters: dict[str, dict] = field(default_factory=dict)
    groups: dict[str, dict] = field(default_factory=dict)
    events: dict[str, dict] = field(default_factory=dict)
    chapters: dict[str, dict] = field(default_factory=dict)
    arcs: dict[str, dict] = field(default_factory=dict)
    general_info: dict[str, dict] = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)
    messages_metadata: list[dict] = field(default_factory=list)
    character_status: dict[str, dict] = field(default_factory=dict)
    character_milestones: dict[str, list[dict]] = field(default_factory=dict)
    relevant_entities: dict[str, set[str]] = field(default_factory=dict)


class StoryContextRetriever:
    def __init__(self, history_path: PathLike, schema_classes: dict | None = None, summarizer: 'Summarizer' | None = None):
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
        self.general_info_path = history_path / "general_info.json"
        self.current_scene_path = history_path / "current_scene.json"
        self.arcs_path = history_path / "arcs.json"

        self.characters = self._load_json(self.characters_path)
        self.groups = self._load_json(self.groups_path)
        self.events = self._load_json(self.events_path)
        self.general_info = self._load_json(self.general_info_path)
        self.current_scene = self._load_json(self.current_scene_path)
        self.arcs = self._load_json(self.arcs_path)

        print(f"{_DEBUG}StoryContextRetriever loaded:")
        print(f"  history_path: {history_path}")
        print(f"  characters keys: {list(self.characters.keys()) if self.characters else 'empty'}")
        print(f"  groups keys: {list(self.groups.keys()) if self.groups else 'empty'}")
        print(f"  events keys: {list(self.events.keys()) if self.events else 'empty'}")
        print(f"  general_info keys: {list(self.general_info.keys()) if self.general_info else 'empty'}")
        print(f"  current_scene keys: {list(self.current_scene.keys()) if self.current_scene else 'empty'}")
        print(f"  arcs count: {len(self.arcs) if self.arcs else 0}{_RESET}")

        # Initialize entity graph for relationship tracking
        self.schema_classes = schema_classes or {}
        self.schema_wrapper = SchemaWrapper(self.schema_classes) if self.schema_classes else None
        self.entity_graph = EntityGraph(history_path, persist=True, schema_classes=schema_classes)
        print(f"{_DEBUG}EntityGraph initialized with {len(self.entity_graph.nodes)} nodes{_RESET}")

        # Store summarizer and pass to chunker for LLM-based speaker extraction
        self.summarizer = summarizer
        # TODO: Make this configurable via UI toggle
        self.use_llm_for_speakers = True  # Toggle: True to always use LLM, False to use regex/spaCy
        self.chunker = MessageChunker(history_path, self.characters, self.groups, self.events, self.current_scene, summarizer=summarizer, use_llm_for_speakers=self.use_llm_for_speakers)

        # Create character name patterns for recognition (from both JSON and graph)
        self.character_patterns = self._create_character_patterns()

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
            pattern = f"({char_name}"
            if len(names) > 1:
                pattern += f"|{names[0]}|{names[-1]}"
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

    def query_messages(self, query: str, n_results: int = 5) -> tuple[list[str], list[dict]]:
        """Query messages using semantic search.

        Returns:
            tuple of (messages, metadata)
        """
        results = self.chunker.query_similar(query, n_results=n_results)
        messages = results.get("documents", []) if results else []
        metadata = results.get("metadatas", []) if results else []
        return messages, metadata

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
                important_rels = [rel for rel in rel_list if self._get_importance(rel, "Character", "relationships") >= importance_threshold]
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
                char_data = characters_data[char_name].copy()  # Copy to avoid modifying original

                # Get important relationships
                if hasattr(self, 'entity_graph') and self.entity_graph:
                    graph_rels = self.entity_graph.get_important_relationships(char_name, importance_threshold)
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
                char_rels.update(important_rels)
                char_data["relationships"] = char_rels
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
    ) -> dict[str, set[str]]:
        """Unified dynamic entity aggregation using EntityGraph.traverse_graph().

        Delegates all relationship traversal to the entity graph for O(1) neighbor lookups
        and bidirectional traversal. Falls back to legacy logic only if entity_graph is unavailable.

        Args:
            initial_entities: Dict of entity_type -> initial set of entity names
                e.g., {"Character": {"John", "Amy"}, "Group": {"Rebel Force"}}
            all_data: Dict of entity_type -> entity data dict (unused when using entity graph)
            importance_threshold: Minimum importance score for filtering
            max_depth: Maximum recursion depth

        Returns:
            Dict of entity_type -> aggregated set of entity names
        """
        if not hasattr(self, 'entity_graph') or not self.entity_graph:
            return {"character": set(), "group": set(), "event": set()}

        field_map = self.entity_graph.get_schema_relationship_map()
        if not field_map:
            field_map = {
                "character": {"relationships": "character", "group_status": "group", "milestones": "event"},
                "group": {"characters": "character", "relationships": "group", "events": "event"},
                "event": {"participants": "character"},
            }

        graph_initial = {}
        for etype, names in initial_entities.items():
            if etype.lower() in ("character", "group", "event"):
                graph_initial[etype.lower()] = names

        result = self.entity_graph.traverse_graph(
            graph_initial,
            field_map=field_map,
            min_importance=importance_threshold,
            max_depth=max_depth,
        )

        return result

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
        parts = path.split(".")
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
        if self.schema_wrapper:
            path = f"{field_name}.importance.score"
            score = self._get_nested_field_value(item_data, item_type, path, 0)
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
            initial_entities, all_data, importance_threshold=importance_threshold, max_depth=max_depth
        )

        char_key = "character"
        relevant_chars = relevant.get(char_key, set())
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

        try:
            print(f"{_DEBUG}retrieve_context try block starting. general_info type: {type(result.general_info)}, is empty: {not result.general_info}{_RESET}")
            print(f"{_DEBUG}scene_characters to look up: {scene_characters}{_RESET}")
            print(f"{_DEBUG}self.characters keys: {list(self.characters.keys()) if self.characters else 'empty'}{_RESET}")

            initial_entities = {"Character": set(scene_characters)}
            all_data = {
                "Character": self.characters,
                "Group": self.groups,
                "Event": self.events,
            }
            unified_result = self._unified_entity_aggregation(
                initial_entities, all_data, importance_threshold=75, max_depth=10
            )
            result.relevant_entities = unified_result
            char_key = "character"
            group_key = "group"
            event_key = "event"
            unified_chars = unified_result.get(char_key, set())
            unified_groups = unified_result.get(group_key, set())
            unified_events = unified_result.get(event_key, set())
            print(f"{_DEBUG}unified aggregation: {len(unified_chars)} chars, {len(unified_groups)} groups, {len(unified_events)} events{_RESET}")

            result.characters = self._get_all_relevant_character_relationships(list(unified_chars))
            print(f"{_DEBUG}characters retrieved: {type(result.characters)}, count: {len(result.characters) if result.characters else 0}{_RESET}")
            groups_entries = self._get_entries(self.groups, "Group")
            result.groups = {"entries": {g: groups_entries.get(g, {}) for g in unified_groups}}
            print(f"{_DEBUG}groups retrieved: {type(result.groups)}, count: {len(result.groups) if result.groups else 0}{_RESET}")

            scenes = self._get_field_value(self.events, "Event", "scenes", {})
            events = self._get_field_value(self.events, "Event", "events", {})
            past = self._get_field_value(self.events, "Event", "past", {})
            events_dict = {**scenes, **events, **past}
            result.events = {"entries": {e: events_dict.get(e, {}) for e in unified_events}}
            print(f"{_DEBUG}events retrieved: {type(result.events)}, count: {len(result.events) if result.events else 0}{_RESET}")

            if self.arcs:
                result.arcs = self.arcs
                print(f"{_DEBUG}arcs retrieved: {type(result.arcs)}, count: {len(result.arcs) if result.arcs else 0}{_RESET}")

            chapters_data = self._get_field_value(self.events, "Event", "chapters", {})
            if chapters_data:
                result.chapters = chapters_data
                print(f"{_DEBUG}chapters retrieved: {type(result.chapters)}, count: {len(result.chapters) if result.chapters else 0}{_RESET}")

            result.character_status, result.character_milestones = self._get_all_relevant_status_and_milestones(
                scene_characters, self.groups, self.characters, self.events, current_scene
            )
            print(f"{_DEBUG}character_status retrieved: {type(result.character_status)}, count: {len(result.character_status) if result.character_status else 0}{_RESET}")
            print(f"{_DEBUG}character_milestones retrieved: {type(result.character_milestones)}, count: {len(result.character_milestones) if result.character_milestones else 0}{_RESET}")

            # Get messages using both retrieval methods
            # scene_messages = self._get_message_chunks()  # Index-based retrieval
            semantic_messages, semantic_metadata = self.query_messages(context_to_search, n_results=5)  # Semantic search

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


from typing import Any


class MessageChunker:
    # Class-level singletons to avoid reloading heavy resources
    _embed_model = None
    _nlp = None
    _nltk_downloaded = False
    _spacy_model_downloaded = False
    _initialized = False
    _warning_suppressed = False

    @classmethod
    def _init_shared_resources(cls):
        """Initialize shared resources (embed model, spaCy, NLTK) only once."""
        if cls._initialized:
            return

        # Suppress MPNet warning
        if not cls._warning_suppressed:
            warnings.filterwarnings("ignore", message=".*position_ids.*")
            logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
            logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
            cls._warning_suppressed = True

        # Get background imported modules
        if not TYPE_CHECKING:
            global nltk, spacy, HuggingFaceEmbedding
            if any((nltk is None, spacy is None, HuggingFaceEmbedding is None)):
                nltk = get_imported_attribute("nltk")
                spacy = get_imported_attribute("spacy")
                HuggingFaceEmbedding = get_imported_attribute("llama_index.embeddings.huggingface", "HuggingFaceEmbedding")

        # Download NLTK data and load model once
        if not cls._nltk_downloaded:
            nltk_data_path = Path("user_data/nltk_data")
            nltk.data.path.append(nltk_data_path.resolve())
            nltk.download("punkt", download_dir=nltk_data_path, quiet=True)
            nltk.download("punkt_tab", download_dir=nltk_data_path, quiet=True)
            cls._nltk_downloaded = True

        if cls._embed_model is None:
            cls._embed_model = HuggingFaceEmbedding(
                model_name="sentence-transformers/all-mpnet-base-v2"
                # model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            Settings.embed_model = cls._embed_model

        if cls._nlp is None:
            try:
                cls._nlp = spacy.load("en_core_web_sm")
            except OSError:
                print(f"{_BOLD}Downloading spaCy model...{_RESET}")
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
                cls._nlp = spacy.load("en_core_web_sm")

        cls._initialized = True

    def __init__(
        self,
        history_path: PathLike,
        characters_data: dict[str, Any],
        groups_data: dict[str, Any],
        events_data: dict[str, Any],
        current_scene_data: dict[str, Any],
        summarizer: 'Summarizer' | None = None,
        use_llm_for_speakers: bool = True,
    ):
        print(f"{_BOLD}Initializing MessageChunker...{_RESET}")

        MessageChunker._init_shared_resources()

        # Use class-level shared resources
        self.nlp = MessageChunker._nlp
        self.summarizer = summarizer
        # TODO: Make configurable via UI toggle
        self.use_llm_for_speakers = use_llm_for_speakers

        self.history_path = Path(history_path)
        self.storage_dir = self.history_path / "message_index"
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Store provided data
        self.characters_data = characters_data
        self.groups_data = groups_data
        self.events_data = events_data
        self.current_scene_data = current_scene_data

        # Initialize or load existing index
        try:
            self.storage_context = StorageContext.from_defaults(persist_dir=str(self.storage_dir))
            self.index = load_index_from_storage(
                storage_context=self.storage_context,
            )
        except Exception:
            self.index = VectorStoreIndex([])
            self.index.storage_context.persist(persist_dir=str(self.storage_dir))

        self.parser = SimpleNodeParser.from_defaults()

        # Load character patterns for pronoun resolution
        self.pronoun_character_patterns = self._load_pronoun_character_patterns()

        # Create simpler name/alias patterns for direct entity matching
        self.character_name_patterns = self._create_name_alias_patterns(self.characters_data, main_name_key_is_dict_key=True)
        self.group_name_patterns = self._create_name_alias_patterns(self.groups_data, main_name_key_is_dict_key=True)
        self.event_name_patterns = self._create_event_name_patterns(self.events_data)

    DIALOGUE_VERBS = {
        "say",
        "tell",
        "ask",
        "reply",
        "shout",
        "whisper",
        "exclaim",
        "mutter",
        "state",
        "declare",
        "respond",
        "add",
        "continue",
        "begin",
        "murmur",
        "interject",
        "question",
        "answer",
        "stammer",
        "insist",
        "suggest",
        "warn",
    }

    def _create_name_alias_patterns(
        self,
        entity_data: dict[str, dict[str, Any]],
        main_name_key_is_dict_key: bool = True,
    ) -> dict[str, re.Pattern]:
        """Creates regex patterns for entity names and their aliases."""
        patterns = {}
        if not entity_data:
            return patterns
        for main_name, data in entity_data.items():
            names_to_match = [main_name]
            if isinstance(data, dict) and "aliases" in data:
                aliases = data.get("aliases", [])
                if isinstance(aliases, list):
                    names_to_match.extend(aliases)

            # Filter out empty strings and ensure uniqueness
            unique_names = sorted(list(set(filter(None, names_to_match))), key=len, reverse=True)
            if unique_names:
                # Pattern to match whole words, case-insensitive
                pattern_str = r"\b(" + "|".join(re.escape(name) for name in unique_names) + r")\b"
                patterns[main_name] = re.compile(pattern_str, flags=re.IGNORECASE)
        return patterns

    def _create_event_name_patterns(self, events_data: dict[str, list[dict[str, Any]]]) -> dict[str, re.Pattern]:
        """Creates regex patterns for event names."""
        patterns = {}
        if not events_data:
            return patterns

        event_names = []
        for event_list_key in [
            "past",
            "scenes",
            "events",
        ]:  # Iterate through different event categories
            for event_item in events_data.get(event_list_key, []):
                if isinstance(event_item, dict) and "name" in event_item:
                    event_names.append(event_item["name"])

        unique_event_names = sorted(list(set(filter(None, event_names))), key=len, reverse=True)
        if unique_event_names:
            for name in unique_event_names:  # Create a pattern for each unique event name
                # Pattern to match whole words, case-insensitive
                pattern_str = r"\b(" + re.escape(name) + r")\b"
                patterns[name] = re.compile(pattern_str, flags=re.IGNORECASE)
        return patterns

    def _extract_entities(self, text: str, entity_patterns: dict[str, re.Pattern]) -> list[str]:
        """Extract unique entity names from text using provided patterns."""
        found_entities = set()
        for entity_name, pattern in entity_patterns.items():
            if pattern.search(text):
                found_entities.add(entity_name)
        return list(found_entities)

    def _determine_speakers(self, paragraph_text: str) -> list[str]:
        """Determine speakers from text using LLM (primary) with regex "Name:" as quick pre-filter."""
        speakers = set()
        doc = self.nlp(paragraph_text)

        # 1. Check for "Name: Dialogue" format line by line
        lines = paragraph_text.split("\n")
        char_patterns_for_speakers = {
            name: pattern for name, pattern in self.character_name_patterns.items() if isinstance(pattern, re.Pattern)
        }
        group_patterns_for_speakers = {
            name: pattern for name, pattern in self.group_name_patterns.items() if isinstance(pattern, re.Pattern)
        }
        # Primarily, characters are speakers. Groups might be if they have a collective voice represented.
        speaker_name_patterns = {
            **char_patterns_for_speakers,
            **group_patterns_for_speakers,
        }

        for line in lines:
            stripped_line = line.strip()
            if not stripped_line:
                continue

            for name, pattern_obj in speaker_name_patterns.items():
                match = pattern_obj.match(stripped_line)
                if match and match.start() == 0:  # Pattern matches at the beginning of the line
                    # Check if the character(s) immediately following the match is a colon
                    if stripped_line[match.end() :].strip().startswith(":"):
                        speakers.add(name)
                        break  # Found speaker for this line by "Name:" pattern

        # Use LLM for speaker extraction if enabled (more accurate than regex/spaCy)
        # TODO: Make configurable via UI toggle
        if self.use_llm_for_speakers and self.summarizer:
            try:
                prompt = f'''Analyze the following text in context and identify the names of the character(s) who are speaking or being addressed.

Respond with a JSON array of character names:
["Character1", "Character2", ...]

Text:
```
{paragraph_text}
```

Do not include generic terms like "you", "someone", "they". Only include characters that are explicitly or implicitly mentioned as speaking.'''
                response_text, _ = self.summarizer.generate_with_sse(prompt, self.summarizer.last.custom_state, "determine_speakers", "speakers_llm", None)
                if response_text:
                    try:
                        llm_speakers = jsonc.loads(response_text.strip())
                        if isinstance(llm_speakers, list):
                            speakers.update(llm_speakers)
                    except jsonc.JSONDecodeError:
                        pass
            except Exception:
                pass

        if not speakers:
            # 2. Fall back to spaCy-based analysis for quoted speech and other dialogue indicators within sentences
            for sent in doc.sents:
                # Basic check for quotes. More sophisticated quote detection might be needed for complex cases.
                has_quote = (
                    '"' in sent.text
                    or "'" in sent.text
                    or "“" in sent.text
                    or "”" in sent.text
                    or "‘" in sent.text
                    or "’" in sent.text
                )

                for token in sent:
                    # Check for dialogue verbs
                    if token.lemma_.lower() in self.DIALOGUE_VERBS and token.pos_ == "VERB":
                        # Find subject of the verb (potential speaker)
                        subject_token = None
                        for child in token.children:
                            if child.dep_ == "nsubj":
                                subject_token = child
                                break

                        if subject_token:
                            # Extract text of the subject (could be a single name or a phrase)
                            # We can check the subject token itself or its subtree for more complex subjects.
                            subject_text = subject_token.text
                            potential_speakers_from_subject = self._extract_entities(subject_text, self.character_name_patterns)
                            for speaker_name in potential_speakers_from_subject:
                                if has_quote:
                                    speakers.add(speaker_name)

                        # Additionally, check for character names directly preceding/following quotes if not caught by subject-verb
                        # This part can be expanded with more rules.
                        # For example, if token is a quote, check previous/next tokens for names.

                if has_quote and not speakers.intersection(self._extract_entities(sent.text, self.character_name_patterns)):
                    chars_in_sentence_with_quote = self._extract_entities(sent.text, self.character_name_patterns)
                    for char_name in chars_in_sentence_with_quote:
                        # A more robust check would analyze proximity to quote marks.
                        speakers.add(char_name)  # This might over-generate, needs refinement or context.

        return list(speakers)

    def _load_pronoun_character_patterns(self) -> dict[str, dict]:
        """Load character patterns for pronoun resolution from self.characters_data."""
        if not self.characters_data:
            return {}

        patterns = {}
        for char_name, char_data in self.characters_data.items():
            names = [char_name]
            if isinstance(char_data, dict) and "aliases" in char_data:
                aliases = char_data.get("aliases", [])
                if isinstance(aliases, list):
                    names.extend(aliases)

            sex = char_data.get("sex") if isinstance(char_data, dict) else None
            pronouns = []
            if sex == "male":
                pronouns = ["he", "him", "his", "himself"]
            elif sex == "female":
                pronouns = ["she", "her", "hers", "herself"]
            else:
                pronouns = ["they", "them", "their", "theirs", "themself", "themselves"]
            patterns[char_name] = {
                "names": list(set(filter(None, names))),  # Ensure unique and non-empty
                "pronouns": pronouns,
            }
        return patterns

    def _detect_character_references(self, text: str, doc: "Doc" = None) -> list[tuple[str, list[str]]]:
        """Detect character references in text, including pronouns."""
        if doc is None:
            doc = self.nlp(text)

        references = []

        # Track the last mentioned character for pronoun resolution
        last_character = None
        possible_characters = set()

        for token in doc:
            # Direct name matches
            matched_char = None
            for char_name, char_data in self.pronoun_character_patterns.items():
                if any(name.lower() in token.text.lower() for name in char_data["names"]):
                    matched_char = char_name
                    last_character = char_name
                    possible_characters = {char_name}
                    break

            # Pronoun handling
            if token.pos_ == "PRON" or (token.pos_ == "DET" and token.dep_ == "poss"):  # Include possessive determiners
                pron = token.text.lower()
                matching_chars = []

                # If we have a recent character mention and the pronoun matches
                if last_character:
                    char_data = self.pronoun_character_patterns.get(last_character, {})
                    if pron in char_data.get("pronouns", []):
                        matching_chars = [last_character]

                # If no match with recent character, find all possible matches
                if not matching_chars:
                    for char_name, char_data in self.pronoun_character_patterns.items():
                        if pron in char_data["pronouns"]:
                            matching_chars.append(char_name)

                if matching_chars:
                    # For reflexive pronouns (himself/herself/themselves), strongly prefer the last character
                    if pron.endswith("self") and last_character in matching_chars:
                        matching_chars = [last_character]

                    # Update possible characters for this reference
                    if len(matching_chars) == 1:
                        possible_characters = {matching_chars[0]}
                        last_character = matching_chars[0]
                    else:
                        possible_characters.update(matching_chars)

                    references.append((token.text, list(possible_characters)))

            elif matched_char:
                references.append((token.text, [matched_char]))

        return references

    def _tag_character_references(self, text: str) -> str:
        """Tag character references in text with possible character names."""
        doc = self.nlp(text)
        references = self._detect_character_references(text, doc)

        # Sort references by position (longest matches first to avoid nested replacements)
        references.sort(key=lambda x: len(x[0]), reverse=True)

        # Replace references with tagged versions
        tagged_text = text
        for ref_text, possible_chars in references:
            if len(possible_chars) == 1:
                replacement = f"{ref_text} [{possible_chars[0]}]"
            elif len(possible_chars) > 1:
                chars_str = "/".join(possible_chars)
                replacement = f"{ref_text} [{chars_str}]"
            tagged_text = tagged_text.replace(ref_text, replacement)

        return tagged_text

    def chunk_message(self, message: str, message_idx: int, current_timestamp: str, do_determine_speakers: bool = True) -> list:
        """Split message into chunks at different granularities, enrich with metadata."""
        chunks = []

        # Determine characters present in the current scene once
        scene_active_characters = []
        if (
            self.current_scene_data
            and isinstance(self.current_scene_data.get("now"), dict)
            and isinstance(self.current_scene_data["now"].get("who"), dict)
            and isinstance(self.current_scene_data["now"]["who"].get("characters"), dict)
        ):
            for char_info in self.current_scene_data["now"]["who"]["characters"].values():
                if isinstance(char_info, dict) and "name" in char_info:
                    scene_active_characters.append(char_info["name"])

        # Split into paragraphs
        paragraphs = [p.strip() for p in message.split("\n\n") if p.strip()]

        for para_idx, paragraph in enumerate(paragraphs, start=1):
            # Split paragraph into sentences
            sentences = nltk.sent_tokenize(paragraph)

            if do_determine_speakers:
                paragraph_speakers = self._determine_speakers(paragraph)
            else:
                paragraph_speakers = [None]  # Placeholder

            for sent_idx, sentence_text in enumerate(sentences, start=1):
                chunk_id = f"{message_idx}_{para_idx}_{sent_idx}"
                speakers = paragraph_speakers

                # Extract entities directly mentioned in the current sentence
                characters_mentioned_in_sentence = self._extract_entities(sentence_text, self.character_name_patterns)
                groups_referenced_in_sentence = self._extract_entities(sentence_text, self.group_name_patterns)
                events_referenced_in_sentence = self._extract_entities(sentence_text, self.event_name_patterns)

                subjects_referenced = {
                    "characters": characters_mentioned_in_sentence,
                    "groups": groups_referenced_in_sentence,
                    "events": events_referenced_in_sentence,
                }

                chunks.append(
                    {
                        "id": chunk_id,
                        "text": sentence_text,
                        "indices": [message_idx, para_idx, sent_idx],
                        "timestamp": current_timestamp,
                        "speakers": speakers,
                        "characters_present": scene_active_characters,
                        "subjects_referenced": subjects_referenced,
                        "scene_id": None,  # To be filled later
                        "scene_number": self.current_scene_data.get("_scene_number"),
                        "chapter_number": self.current_scene_data.get("_chapter_number"),
                        "event_id": None,  # To be filled later
                    }
                )

        return chunks

    def query_similar(self, query: str, n_results: int = 5):
        """Query similar chunks using LlamaIndex."""
        retriever = self.index.as_retriever(similarity_top_k=n_results)
        nodes = retriever.retrieve(query)

        ids, documents, metadatas, distances = [], [], [], []
        for node in nodes:
            ids.append(node.node.id_)
            documents.append(node.node.text)
            metadatas.append(node.node.metadata)
            distances.append(node.score)
        results = {
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
            "distances": distances,
        }
        return results

    def delete_message_chunks(self, message_idx: int):
        """Delete all chunks for a given message index."""
        # Get all nodes
        all_nodes = self.index.docstore.docs

        # Find nodes to delete
        nodes_to_delete = []
        for node_id, node in all_nodes.items():
            if node.metadata["message_idx"] == message_idx:
                nodes_to_delete.append(node_id)

        # Delete nodes
        for node_id in nodes_to_delete:
            del self.index.docstore.docs[node_id]

        # Persist changes
        self.index.storage_context.persist(persist_dir=str(self.storage_dir))

    def update_node_metadata_by_message_idx(
        self, message_idx: int, metadata_updates: dict[str, Any], persist_dir: PathLike | None = None
    ):
        """Update metadata for all nodes associated with a message_idx."""
        nodes_to_update = []
        # node_ids_to_delete_for_update = [] # Not strictly needed if insert_nodes handles updates by ID

        for node_id, node in self.index.docstore.docs.items():
            if node.metadata.get("message_idx") == message_idx:
                new_metadata = node.metadata.copy()
                new_metadata.update(metadata_updates)

                updated_node = TextNode(
                    text=node.text,
                    id_=node.id_,
                    metadata=new_metadata,
                    # relationships=node.relationships # Preserve relationships if any
                )
                nodes_to_update.append(updated_node)
                # node_ids_to_delete_for_update.append(node_id)

        if nodes_to_update:
            self.index.insert_nodes(nodes_to_update)
            self.index.storage_context.persist(persist_dir=str(persist_dir or self.storage_dir))
            try:
                print(f"{_SUCCESS}Updated metadata for {len(nodes_to_update)} nodes for message_idx {message_idx}{_RESET}")
            except Exception as e:
                print(f"{_ERROR}Error during post-update operations for message_idx {message_idx}: {e}{_RESET}")
        else:
            print(f"{_HILITE}No nodes found for message_idx {message_idx} to update metadata.{_RESET}")

    def process_message(self, message: str, message_idx: int, current_timestamp: str, do_determine_speakers: bool = True) -> list:
        """Process and store a new message. Overwrites existing chunks if message_idx exists."""
        # Delete existing chunks for this message if any
        self.delete_message_chunks(message_idx)

        # Create and store new chunks
        chunks = self.chunk_message(message, message_idx, current_timestamp, do_determine_speakers=do_determine_speakers)
        self.store_chunks(chunks)
        return chunks

    def update_message_speakers(self, message_idx: int) -> bool:
        """Update speakers for existing chunks of a message using current state.

        Uses the current stored message text to re-determine speakers via LLM,
        then updates the metadata for all chunks with that message_idx.

        Args:
            message_idx: The message index to update speakers for.

        Returns:
            bool: True if update succeeded, False otherwise.
        """
        try:
            all_nodes = self.index.docstore.docs
            message_chunks = []
            for node_id, node in all_nodes.items():
                if node.metadata.get("message_idx") == message_idx:
                    message_chunks.append(node)

            if not message_chunks:
                print(f"{_WARNING}No chunks found for message_idx {message_idx} to update speakers.{_RESET}")
                return False

            first_chunk = message_chunks[0]
            message_chunks_sorted = sorted(message_chunks, key=lambda n: n.metadata.get("indices", [0, 0, 0]))
            full_message_text = "\n\n".join(n.text for n in message_chunks_sorted if n.text)

            paragraphs = [p.strip() for p in full_message_text.split("\n\n") if p.strip()]
            if not paragraphs:
                print(f"{_WARNING}No text found in chunks for message_idx {message_idx}.{_RESET}")
                return False

            full_message_text = "\n\n".join(paragraphs)
            speakers = self._determine_speakers(full_message_text)

            if not speakers:
                print(f"{_DEBUG}No speakers determined for message_idx {message_idx}.{_RESET}")

            self.update_node_metadata_by_message_idx(message_idx, {"speakers": speakers})
            return True

        except Exception as e:
            print(f"{_ERROR}Error updating speakers for message_idx {message_idx}: {str(e)}{_RESET}")
            traceback.print_exc()
            return False

    def store_chunks(self, chunks: list, persist_dir: PathLike | None = None):
        """Store chunks using LlamaIndex."""

        nodes = []
        for chunk in chunks:
            metadata = {
                "message_idx": chunk["indices"][0],
                "paragraph_idx": chunk["indices"][1],
                "sentence_idx": chunk["indices"][2],
                "timestamp": chunk.get("timestamp"),
                "speakers": chunk.get("speakers", []),
                "characters_present": chunk.get("characters_present", []),
                "subjects_referenced": chunk.get("subjects_referenced", {}),
                "scene_id": chunk.get("scene_id"),  # Will be None initially
                "scene_number": chunk.get("scene_number"),  # Will be None initially
                "chapter_number": chunk.get("chapter_number"),  # Will be None initially
                "event_id": chunk.get("event_id"),  # Will be None initially
                "is_summary": chunk.get("is_summary", False),
            }

            node = TextNode(text=chunk["text"], id_=chunk["id"], metadata=metadata)
            nodes.append(node)

        if nodes:  # Only insert if there are nodes to avoid errors with empty list
            self.index.insert_nodes(nodes)
            self.index.storage_context.persist(persist_dir=str(persist_dir or self.storage_dir))
            # try:
            #     import shutil

            #     shutil.copytree(
            #         self.storage_dir,
            #         self.history_path / "message_index",
            #         dirs_exist_ok=True,
            #     )
            # except Exception as e:
            #     print(f"{_ERROR}Error copying message_index after storing chunks: {e}{_RESET}")