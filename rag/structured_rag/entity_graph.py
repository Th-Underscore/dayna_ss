"""
Entity Graph for structured relationship tracking.

Provides a graph-based representation of story entities (characters, groups, events, etc.)
and their relationships. Designed to be extensible for schema-driven configuration.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...utils.helpers import load_json, save_json, _DEBUG, _SUCCESS, _ERROR, _RESET, _BOLD


@dataclass
class EntityNode:
    """Represents a single entity in the graph."""
    id: str
    type: str  # e.g., "character", "group", "event", "scene", "chapter", "arc"
    name: str
    data: dict = field(default_factory=dict)
    relationships: dict[str, list[dict]] = field(default_factory=dict)  # target_id -> list of relationship dicts


@dataclass
class Relationship:
    """Represents a single relationship between two entities."""
    source_id: str
    target_id: str
    relation: str  # e.g., "brother", "enemy", "member"
    status: str  # e.g., "Family", "Rebel Force"
    field_name: str = ""
    importance: int = 0
    importance_reason: str = ""
    faction: str = "neutral"  # "positive", "negative", "neutral"
    bidirectional: bool = False
    aliases: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    scenes: list[int] = field(default_factory=list)


class EntityGraph:
    """
    Graph-based representation of story entities and their relationships.

    Uses schema-driven configuration when schema_classes are provided,
    otherwise falls back to hardcoded parsing.
    """

    def __init__(self, history_path: Path, persist: bool = True, schema_classes: dict | None = None):
        """
        Initialize the entity graph.

        Args:
            history_path: Path to the history directory containing subject JSON files.
            persist: Whether to save/load graph from disk for debugging.
            schema_classes: Optional dict of ParsedSchemaClass objects for schema-driven building.
        """
        self.history_path = Path(history_path)
        self.persist = persist
        self.schema_classes = schema_classes or {}

        self.nodes: dict[str, EntityNode] = {}
        self.relationships: list[Relationship] = []

        self.adjacency: dict[str, dict[str, set[str]]] = {}  # source_type -> target_type -> {target_ids}
        self.adjacency_built: bool = False

        self.source_fingerprint: dict = {}
        self._load_or_build()

    def _resolve_participant_id(self, participant: str) -> tuple[str, str]:
        """Resolve a participant name to (node_id, entity_type).

        Checks self.nodes for existing keys (including suffix matches for raw names),
        explicit prefixes, then defaults to character.
        """
        if participant in self.nodes:
            return participant, self.nodes[participant].type
        for node_id in self.nodes:
            if node_id.endswith(f":{participant}"):
                return node_id, self.nodes[node_id].type
        if ":" in participant:
            prefix = participant.split(":", 1)[0]
            return participant, prefix
        return f"character:{participant}", "character"

    def _attach_participants_to_event(self, node_id: str, participants: list | dict) -> None:
        """Create participant nodes and relationship edges for an event/scene node.

        Handles both list and dict participant entries, extracting importance
        from dict shapes.
        """
        if not participants:
            return
        if isinstance(participants, dict):
            iterable = participants.items()
        else:
            iterable = enumerate(participants)

        for entry in iterable:
            if isinstance(entry, tuple) and len(entry) == 2:
                if isinstance(entry[0], str):
                    participant_name = entry[0]
                    participant_val = entry[1]
                else:
                    participant_name = None
                    participant_val = entry[1]
            else:
                participant_name = None
                participant_val = entry

            if isinstance(participant_val, str):
                participant_id, ptype = self._resolve_participant_id(participant_val)
                importance_val = 50
            elif isinstance(participant_val, dict):
                name = (
                    participant_name
                    or participant_val.get("name", "")
                    or participant_val.get("id", "")
                )
                if not name:
                    continue
                participant_id, ptype = self._resolve_participant_id(name)
                imp = participant_val.get("importance", {})
                if isinstance(imp, dict):
                    importance_val = imp.get("score", 50)
                elif isinstance(imp, int):
                    importance_val = imp
                else:
                    importance_val = 50
            else:
                continue
            if participant_id not in self.nodes:
                pname = participant_id.split(":", 1)[1] if ":" in participant_id else participant_id
                self.nodes[participant_id] = EntityNode(
                    id=participant_id,
                    type=ptype,
                    name=pname,
                    data={}
                )
            self.relationships.append(Relationship(
                source_id=node_id,
                target_id=participant_id,
                relation="participant",
                status="event_participation",
                field_name="participants",
                importance=importance_val
            ))

    def _load_or_build(self):
        """Load from disk or build from source files."""
        graph_path = self.history_path / "entity_graph.json"

        if self.persist and graph_path.exists():
            # Check if source files have changed
            self.source_fingerprint = self._compute_source_fingerprint()
            data = load_json(graph_path)
            saved_fingerprint = data.get("source_fingerprint", {})

            if self.source_fingerprint == saved_fingerprint:
                self._load_from_disk(graph_path)
            else:
                print(f"{_DEBUG}Source files have changed, rebuilding entity graph...{_RESET}")
                self._build_from_source_files()
                self._save_to_disk(graph_path)
        else:
            self._build_from_source_files()
            self.source_fingerprint = self._compute_source_fingerprint()
            if self.persist:
                self._save_to_disk(graph_path)

    def _compute_source_fingerprint(self) -> dict:
        """Compute fingerprint of source files based on modification times."""
        fingerprint = {}
        source_files = ["characters.json", "groups.json", "events.json", "arcs.json"]

        for filename in source_files:
            file_path = self.history_path / filename
            if file_path.exists():
                fingerprint[filename] = file_path.stat().st_mtime
            else:
                fingerprint[filename] = None

        return fingerprint

    def _load_from_disk(self, graph_path: Path):
        """Load graph from JSON file."""
        data = load_json(graph_path)

        self.nodes = {
            node_id: EntityNode(
                id=node_data["id"],
                type=node_data["type"],
                name=node_data["name"],
                data=node_data.get("data", {}),
                relationships=node_data.get("relationships", {})
            )
            for node_id, node_data in data.get("nodes", {}).items()
        }

        self.relationships = [
            Relationship(
                source_id=r["source_id"],
                target_id=r["target_id"],
                relation=r["relation"],
                status=r["status"],
                importance=r.get("importance", 0),
                importance_reason=r.get("importance_reason", ""),
                faction=r.get("faction", "neutral"),
                bidirectional=r.get("bidirectional", False),
                aliases=r.get("aliases", []),
                events=r.get("events", []),
                field_name=r.get("field_name", ""),
                scenes=r.get("scenes", [])
            )
            for r in data.get("relationships", [])
        ]

        print(f"{_SUCCESS}Loaded entity graph with {len(self.nodes)} nodes and {len(self.relationships)} relationships{_RESET}")

    def _save_to_disk(self, graph_path: Path):
        """Save graph to JSON file for debugging."""
        data = {
            "source_fingerprint": self._compute_source_fingerprint(),
            "nodes": {
                node_id: {
                    "id": node.id,
                    "type": node.type,
                    "name": node.name,
                    "data": node.data,
                    "relationships": node.relationships
                }
                for node_id, node in self.nodes.items()
            },
            "relationships": [
                {
                    "source_id": r.source_id,
                    "target_id": r.target_id,
                    "relation": r.relation,
                    "status": r.status,
                    "importance": r.importance,
                    "importance_reason": r.importance_reason,
                    "faction": r.faction,
                    "bidirectional": r.bidirectional,
                    "aliases": r.aliases,
                    "events": r.events,
                    "field_name": r.field_name,
                    "scenes": r.scenes
                }
                for r in self.relationships
            ]
        }

        save_json(data, graph_path)
        print(f"{_DEBUG}Saved entity graph to {graph_path}{_RESET}")

    def _build_from_source_files(self):
        """Build graph from source JSON files using schema-driven approach."""
        print(f"{_BOLD}Building entity graph from source files...{_RESET}")

        if self.schema_classes:
            self._build_from_schema()
        else:
            self._build_characters()
            self._build_groups()
            self._build_events()
            self._build_arcs()

        print(f"{_SUCCESS}Built entity graph with {len(self.nodes)} nodes and {len(self.relationships)} relationships{_RESET}")

        self.build_adjacency()

    def _build_from_schema(self):
        """Build graph using schema-defined relationship formats."""
        # Map subject names to entity types and JSON files
        subject_mapping = {
            "characters": ("character", "characters.json"),
            "groups": ("group", "groups.json"),
            "events": ("event", "events.json"),
            "arcs": ("arc", "arcs.json"),
        }

        for subject_name, (entity_type, filename) in subject_mapping.items():
            # Get schema class for this subject (e.g., Character, Group)
            schema_class = self.schema_classes.get(subject_name.capitalize())
            if not schema_class:
                continue

            rel_fields = schema_class.get_relationship_fields()

            # Use specialized builders for event-like schemas that have nested structure
            if subject_name == "events":
                self._build_events()
            elif subject_name == "arcs":
                self._build_arcs(schema_rel_fields=rel_fields)
            else:
                self._build_entities_from_schema(entity_type, filename, rel_fields)

        # Apply bidirectional rules based on schema config
        self._apply_bidirectional_rules_from_schema()

    def _build_entities_from_schema(self, entity_type: str, filename: str, rel_fields: dict[str, dict]):
        """Build entities of a specific type using schema relationship config."""
        file_path = self.history_path / filename
        if not file_path.exists():
            return

        data = load_json(file_path)
        entries = data.get("entries", data)

        for entity_name, entity_data in entries.items():
            node_id = f"{entity_type}:{entity_name}"
            node = EntityNode(
                id=node_id,
                type=entity_type,
                name=entity_name,
                data=entity_data
            )

            if rel_fields:
                self._process_relationship_fields(node, entity_data, entity_type, entity_name, node_id, rel_fields)
            self.nodes[node_id] = node

    def _process_relationship_fields(
        self,
        node: EntityNode,
        entity_data: dict,
        entity_type: str,
        entity_name: str,
        node_id: str,
        rel_fields: dict[str, dict],
    ):
        """Process relationship fields defined in schema for a given entity node."""
        for field_name, rel_config in rel_fields.items():
            if field_name not in entity_data:
                continue

            field_data = entity_data[field_name]
            target_type = rel_config.get("target_type", "unknown")
            is_bidirectional = rel_config.get("bidirectional", False)
            relationship_def = rel_config.get("relationship_definition", "")

            # Store original relationships in node
            node.relationships[field_name] = field_data

            # Build relationships based on field type
            if isinstance(field_data, dict) and field_name == "characters" and entity_type == "group":
                self._process_group_membership(node_id, field_data, field_name)
            elif isinstance(field_data, dict):
                self._process_relationship_dict(
                    node_id, entity_name, target_type, field_data,
                    relationship_def, is_bidirectional, field_name
                )
            elif isinstance(field_data, list):
                self._process_list_relationships(
                    node_id, entity_name, target_type, field_data, field_name
                )

    def _process_relationship_dict(
        self,
        source_id: str,
        source_name: str,
        target_type: str,
        rel_dict: dict,
        relationship_def: str,
        bidirectional: bool,
        field_name: str = ""
    ):
        """Process a dictionary of relationships (target -> list of relationship objects)."""
        for target_name, rel_list in rel_dict.items():
            if not isinstance(rel_list, list):
                continue

            target_id = f"{target_type}:{target_name}"

            for rel in rel_list:
                if not isinstance(rel, dict):
                    continue

                # Extract importance data
                importance_data = rel.get("importance", {})
                if isinstance(importance_data, dict):
                    importance = importance_data.get("score", 0)
                    importance_reason = importance_data.get("reason", "")
                    faction = importance_data.get("faction", "neutral")
                else:
                    importance = importance_data if isinstance(importance_data, int) else 0
                    importance_reason = ""
                    faction = "neutral"

                relationship = Relationship(
                    source_id=source_id,
                    target_id=target_id,
                    relation=rel.get("relation", ""),
                    status=rel.get("status", ""),
                    field_name=field_name,
                    importance=importance,
                    importance_reason=importance_reason,
                    faction=faction,
                    aliases=rel.get("aliases", []),
                    events=rel.get("events", []),
                    scenes=rel.get("scenes", [])
                )
                self.relationships.append(relationship)

    def _process_group_membership(self, group_id: str, members: dict, field_name: str = "characters"):
        """Process group membership (character -> GroupCharacterInfo)."""
        for char_name, char_info in members.items():
            if not isinstance(char_info, dict):
                continue

            relationship = Relationship(
                source_id=group_id,
                target_id=f"character:{char_name}",
                relation="member",
                status="group_membership",
                field_name=field_name,
                importance=50,
                events=char_info.get("events", []),
                scenes=char_info.get("scenes", [])
            )
            self.relationships.append(relationship)

    def _process_list_relationships(
        self,
        source_id: str,
        source_name: str,
        target_type: str,
        rel_list: list,
        field_name: str
    ):
        """Process a list of relationship targets (like milestones -> events, participants -> characters)."""
        for rel_item in rel_list:
            scenes = []
            if isinstance(rel_item, dict):
                target_name = rel_item.get("title", rel_item.get("name", ""))
                importance_data = rel_item.get("importance", {})
                if isinstance(importance_data, dict):
                    importance = importance_data.get("score", 0)
                    importance_reason = importance_data.get("reason", "")
                    faction = importance_data.get("faction", "neutral")
                else:
                    importance = importance_data if isinstance(importance_data, int) else 0
                    importance_reason = ""
                    faction = "neutral"
                scenes = rel_item.get("scenes", [])
            elif isinstance(rel_item, str):
                target_name = rel_item
                importance = 0
                importance_reason = ""
                faction = "neutral"
            else:
                continue

            if not target_name:
                continue

            target_id = f"{target_type}:{target_name}"
            relationship = Relationship(
                source_id=source_id,
                target_id=target_id,
                relation="participant",
                status=field_name,
                field_name=field_name,
                importance=importance,
                importance_reason=importance_reason,
                faction=faction,
                scenes=scenes
            )
            self.relationships.append(relationship)

    @staticmethod
    def _singularize(name: str) -> str:
        """Singularize an entity type name, matching subject_mapping convention."""
        subject_mapping = {
            "characters": "character", "groups": "group",
            "events": "event", "arcs": "arc",
        }
        lower = name.lower()
        result = subject_mapping.get(lower)
        if result is not None:
            return result
        if lower.endswith("ies"):
            return lower[:-3] + "y"
        if lower.endswith("s"):
            return lower[:-1]
        return lower

    def _apply_bidirectional_rules_from_schema(self):
        """Apply bidirectional relationship rules based on schema configuration."""
        # Get bidirectional rules from schema
        bidir_rules = []

        for schema_name, schema_class in self.schema_classes.items():
            rel_fields = schema_class.get_relationship_fields()
            for config in rel_fields.values():
                if config.get("bidirectional", False):
                    source_type = self._singularize(schema_name)
                    target_type = self._singularize(config.get("target_type", "unknown"))
                    bidir_rules.append((source_type, target_type))

        # Mark relationships as bidirectional in metadata
        for rel in self.relationships:
            source_type = rel.source_id.split(":")[0]
            target_type = rel.target_id.split(":")[0]

            for src, tgt in bidir_rules:
                if source_type == src and target_type == tgt:
                    rel.bidirectional = True
                    break

    def _build_characters(self):
        """Parse characters.json to build character nodes and their relationships."""
        characters_path = self.history_path / "characters.json"
        if not characters_path.exists():
            return

        characters_data = load_json(characters_path)
        entries = characters_data.get("entries", characters_data)

        for char_name, char_data in entries.items():
            node_id = f"character:{char_name}"
            node = EntityNode(
                id=node_id,
                type="character",
                name=char_name,
                data=char_data
            )

            # Store relationships
            relationships = char_data.get("relationships", {})
            node.relationships = relationships
            self.nodes[node_id] = node

            # Add relationships to graph
            for target_name, rel_list in relationships.items():
                if not isinstance(rel_list, list):
                    continue

                for rel in rel_list:
                    if not isinstance(rel, dict):
                        continue

                    importance_data = rel.get("importance", {})
                    if isinstance(importance_data, dict):
                        importance = importance_data.get("score", 0)
                        importance_reason = importance_data.get("reason", "")
                        faction = importance_data.get("faction", "neutral")
                    else:
                        importance = importance_data if isinstance(importance_data, int) else 0
                        importance_reason = ""
                        faction = "neutral"

                    relationship = Relationship(
                        source_id=node_id,
                        target_id=f"character:{target_name}",
                        relation=rel.get("relation", ""),
                        status=rel.get("status", ""),
                        importance=importance,
                        importance_reason=importance_reason,
                        faction=faction,
                        aliases=rel.get("aliases", []),
                        events=rel.get("events", [])
                    )
                    self.relationships.append(relationship)

    def _build_groups(self):
        """Parse groups.json to build group nodes and their relationships."""
        groups_path = self.history_path / "groups.json"
        if not groups_path.exists():
            return

        groups_data = load_json(groups_path)
        entries = groups_data.get("entries", groups_data)

        for group_name, group_data in entries.items():
            node_id = f"group:{group_name}"
            node = EntityNode(
                id=node_id,
                type="group",
                name=group_name,
                data=group_data
            )

            # Store character membership
            characters = group_data.get("characters", {})
            node.relationships = {"members": list(characters.keys())}

            # Add group->character relationships
            for char_name, char_info in characters.items():
                relationship = Relationship(
                    source_id=node_id,
                    target_id=f"character:{char_name}",
                    relation="member",
                    status="group_membership",
                    importance=50,  # Default importance for group membership
                    events=char_info.get("events", []) if isinstance(char_info, dict) else []
                )
                self.relationships.append(relationship)

            # Add group->group relationships
            group_rels = group_data.get("relationships", {})
            for target_group, rel_data in group_rels.items():
                relationship = Relationship(
                    source_id=node_id,
                    target_id=f"group:{target_group}",
                    relation="related",
                    status=rel_data.get("position", [""])[0] if isinstance(rel_data, dict) else "",
                    importance=50
                )
                self.relationships.append(relationship)

            self.nodes[node_id] = node

    def _build_events(self):
        """Parse events.json to build event/scene nodes."""
        events_path = self.history_path / "events.json"
        if not events_path.exists():
            return

        events_data = load_json(events_path)

        # Build scenes
        scenes = events_data.get("scenes", {})
        for scene_name, scene_data in scenes.items():
            node_id = f"scene:{scene_name}"
            node = EntityNode(
                id=node_id,
                type="scene",
                name=scene_name,
                data=scene_data
            )

            participants = scene_data.get("participants", [])
            if participants:
                node.data["participants"] = participants
            self._attach_participants_to_event(node_id, participants)
            self.nodes[node_id] = node

        # Build past events
        past = events_data.get("past", {})
        for event_name, event_data in past.items():
            node_id = f"event:{event_name}"
            node = EntityNode(
                id=node_id,
                type="event",
                name=event_name,
                data=event_data
            )

            participants = event_data.get("participants", [])
            if participants:
                node.data["participants"] = participants
            self._attach_participants_to_event(node_id, participants)
            self.nodes[node_id] = node

        # Build crucial events
        crucial = events_data.get("events", {})
        for event_name, event_data in crucial.items():
            node_id = f"event:{event_name}"
            node = EntityNode(
                id=node_id,
                type="event",
                name=event_name,
                data=event_data
            )

            participants = event_data.get("participants", [])
            if participants:
                node.data["participants"] = participants
            self._attach_participants_to_event(node_id, participants)
            self.nodes[node_id] = node

        # Build chapters (handle both dict and list)
        chapters = events_data.get("chapters", {})
        if isinstance(chapters, dict):
            for chapter_name, chapter_data in chapters.items():
                node_id = f"chapter:{chapter_name}"
                node = EntityNode(
                    id=node_id,
                    type="chapter",
                    name=chapter_name,
                    data=chapter_data
                )
                self.nodes[node_id] = node
        elif isinstance(chapters, list):
            for idx, chapter in enumerate(chapters):
                # Use stable identifier: id, name, or index
                chapter_id = chapter.get("id") or chapter.get("name") or f"chapter_{idx}"
                chapter_name = chapter.get("name") or chapter.get("title") or f"Chapter {idx + 1}"
                node_id = f"chapter:{chapter_id}"
                node = EntityNode(
                    id=node_id,
                    type="chapter",
                    name=chapter_name,
                    data=chapter
                )
                self.nodes[node_id] = node

    def _build_arcs(self, schema_rel_fields: dict[str, dict] | None = None):
        """Parse arcs.json to build arc nodes.

        Args:
            schema_rel_fields: Optional relationship field config from schema,
                               passed when called from schema mode.
        """
        arcs_path = self.history_path / "arcs.json"
        if not arcs_path.exists():
            return

        arcs_data = load_json(arcs_path)

        if isinstance(arcs_data, list):
            arcs_list = arcs_data
        elif isinstance(arcs_data, dict):
            arcs_list = arcs_data.get("entries", arcs_data.get("arcs", []))
            if isinstance(arcs_list, dict):
                arcs_list = [{"title": k, **(v if isinstance(v, dict) else {})} for k, v in arcs_list.items()]
        else:
            return

        for arc_data in arcs_list:
            if not isinstance(arc_data, dict):
                continue

            arc_title = arc_data.get("title", "")
            if not arc_title:
                continue

            node_id = f"arc:{arc_title}"
            node = EntityNode(
                id=node_id,
                type="arc",
                name=arc_title,
                data=arc_data
            )
            self.nodes[node_id] = node

            if schema_rel_fields:
                self._process_relationship_fields(node, arc_data, "arc", arc_title, node_id, schema_rel_fields)

    def get_node(self, node_id: str) -> EntityNode | None:
        """Get a node by ID."""
        return self.nodes.get(node_id)

    def get_nodes_by_type(self, node_type: str) -> list[EntityNode]:
        """Get all nodes of a specific type."""
        return [node for node in self.nodes.values() if node.type == node_type]

    def get_character_relationships(self, character_name: str, min_importance: int = 0) -> list[Relationship]:
        """
        Get all relationships for a character.

        Args:
            character_name: Name of the character
            min_importance: Minimum importance score to include (default: 0)

        Returns:
            List of Relationship objects
        """
        character_id = f"character:{character_name}"
        return [
            r for r in self.relationships
            if r.source_id == character_id and r.importance >= min_importance
        ]

    def get_important_relationships(self, character_name: str, threshold: int = 75) -> list[Relationship]:
        """Get important relationships for a character (importance >= threshold)."""
        return self.get_character_relationships(character_name, min_importance=threshold)

    def get_relationships_by_field(
        self,
        character_name: str,
        field_name: str,
        min_importance: int = 0,
        current_scene: int | None = None
    ) -> list[Relationship]:
        """Get relationships for a character filtered by field_name (e.g., 'group_status', 'relationships').

        Args:
            character_name: Name of the character
            field_name: Field name to filter by (e.g., "group_status", "relationships", "milestones")
            min_importance: Minimum importance score
            current_scene: Optional scene number for filtering

        Returns:
            List of Relationship objects matching the criteria
        """
        character_id = f"character:{character_name}"
        results = []

        for r in self.relationships:
            if r.source_id != character_id:
                continue
            if r.field_name != field_name:
                continue
            if r.importance < min_importance:
                continue
            if current_scene is not None and r.scenes:
                if current_scene not in r.scenes:
                    continue
            results.append(r)

        return results

    def get_scene_characters(self, scene_name: str) -> list[str]:
        """Get all characters in a scene."""
        return self.get_entities_involved_in_event(scene_name)

    def get_groups_for_character(self, character_name: str) -> list[str]:
        """
        Get all groups a character belongs to.

        Args:
            character_name: Name of the character

        Returns:
            List of group names the character is a member of
        """
        character_id = f"character:{character_name}"
        groups = []
        for r in self.relationships:
            if r.target_id == character_id and r.relation == "member":
                # Extract display name from namespaced ID
                group_name = r.source_id.split(":", 1)[1] if ":" in r.source_id else r.source_id
                groups.append(group_name)
        return groups

    def get_groups_for_characters(self, character_names: list[str]) -> dict[str, list[str]]:
        """
        Get groups for multiple characters.

        Args:
            character_names: List of character names

        Returns:
            Dict mapping character names to their groups
        """
        result = {}
        for char_name in character_names:
            groups = self.get_groups_for_character(char_name)
            if groups:
                result[char_name] = groups
        return result

    def get_relevant_groups(self, character_names: list[str], context: str = "") -> list[str]:
        """
        Get groups relevant to the given characters.

        Args:
            character_names: List of character names in the scene
            context: Optional context string (for future text-based matching)

        Returns:
            List of relevant group names
        """
        relevant = set()
        for char_name in character_names:
            groups = self.get_groups_for_character(char_name)
            relevant.update(groups)
        return list(relevant)

    def get_events_for_character(self, character_name: str) -> list[str]:
        """
        Get all events a character is involved in.

        Args:
            character_name: Name of the character

        Returns:
            List of event names the character is involved in
        """
        character_id = f"character:{character_name}"
        events = set()
        # Get events from relationship metadata
        for r in self.relationships:
            if r.source_id == character_id:
                events.update(r.events)
        # Get events where character is a participant
        for r in self.relationships:
            if r.target_id == character_id and r.relation == "participant":
                # Extract display name from namespaced ID
                event_name = r.source_id.split(":", 1)[1] if ":" in r.source_id else r.source_id
                events.add(event_name)
        return list(events)

    def get_character_milestones(
        self,
        character_name: str,
        importance_threshold: int = 0,
        current_scene: int | None = None
    ) -> list[dict]:
        """
        Get milestones for a character filtered by importance and/or scene.

        Args:
            character_name: Name of the character
            importance_threshold: Minimum importance score (0-100)
            current_scene: Optional scene number to filter by

        Returns:
            List of milestone dicts with name, importance, scenes
        """
        character_id = f"character:{character_name}"
        milestones = []

        for r in self.relationships:
            if r.source_id == character_id and r.field_name == "milestones":
                if r.importance < importance_threshold:
                    continue
                if current_scene is not None and r.scenes:
                    if current_scene not in r.scenes:
                        continue
                target_name = r.target_id.split(":", 1)[1] if ":" in r.target_id else r.target_id
                milestones.append({
                    "title": target_name,
                    "importance": r.importance,
                    "importance_reason": r.importance_reason,
                    "faction": r.faction,
                    "scenes": r.scenes
                })

        return milestones

    def get_relevant_events(
        self,
        character_names: list[str],
        group_names: list[str] | None = None,
        context: str = "",
        current_scene: int | None = None
    ) -> list[str]:
        """
        Get events relevant to the given characters and groups.

        Args:
            character_names: List of character names
            group_names: Optional list of group names
            context: Optional context string (for future text-based matching)
            current_scene: Optional scene number to filter by

        Returns:
            List of relevant event/scene names
        """
        relevant = set()

        for char_name in character_names:
            character_id = f"character:{char_name}"
            for r in self.relationships:
                if r.source_id == character_id or r.target_id == character_id:
                    if current_scene is not None and r.scenes:
                        if current_scene not in r.scenes:
                            continue
                    relevant.update(r.events)
                    for rel_id in (r.source_id, r.target_id):
                        if rel_id != character_id and (rel_id.startswith("event:") or rel_id.startswith("scene:")):
                            event_or_scene_name = rel_id.split(":", 1)[1] if ":" in rel_id else rel_id
                            relevant.add(event_or_scene_name)

        if group_names:
            for group_name in group_names:
                group_id = f"group:{group_name}"
                node = self.get_node(group_id)
                if node and node.type == "group":
                    events = node.data.get("events", [])
                    relevant.update(events)

        return list(relevant)

    def get_scene_for_characters(self, character_names: list[str]) -> str | None:
        """
        Try to determine which scene the given characters are in.

        This is a placeholder - would need scene->character mapping in node data.

        Args:
            character_names: List of character names

        Returns:
            Scene name if determinable, None otherwise
        """
        # Could be enhanced to track which characters are in which scene
        # For now, return None
        return None

    def get_group_members(self, group_name: str) -> list[str]:
        """Get all members of a group."""
        group_id = f"group:{group_name}"
        node = self.get_node(group_id)
        if node and node.type == "group":
            members = node.relationships.get("members", [])
            if not members:
                members = list(node.relationships.get("characters", {}).keys())
            return members
        return []

    def get_bidirectional_relationship(self, entity1: str, entity2: str) -> dict[str, Relationship]:
        """
        Get relationships between two entities in both directions.

        Returns:
            Dict with "forward" and "backward" Relationship objects
        """
        # Convert to namespaced IDs (assume characters for now)
        entity1_id = f"character:{entity1}"
        entity2_id = f"character:{entity2}"

        result = {"forward": None, "backward": None}

        for r in self.relationships:
            if r.source_id == entity1_id and r.target_id == entity2_id:
                result["forward"] = r
            elif r.source_id == entity2_id and r.target_id == entity1_id:
                result["backward"] = r

        return result

    def get_bidirectional_importance(self, entity1: str, entity2: str) -> int:
        """
        Get combined importance score for bidirectional relationship.

        If both entities know each other, returns the sum of both importance scores.
        If only one knows the other, returns that single score.

        Args:
            entity1: First entity ID
            entity2: Second entity ID

        Returns:
            Combined importance score (0-200), or single direction importance if unidirectional
        """
        bidir = self.get_bidirectional_relationship(entity1, entity2)

        forward_imp = bidir["forward"].importance if bidir["forward"] else 0
        backward_imp = bidir["backward"].importance if bidir["backward"] else 0

        if forward_imp > 0 and backward_imp > 0:
            return forward_imp + backward_imp
        elif forward_imp > 0:
            return forward_imp
        elif backward_imp > 0:
            return backward_imp
        return 0

    def get_entities_involved_in_event(self, event_name: str) -> list[str]:
        """Get all entities involved in an event."""
        for prefix in ["event:", "scene:"]:
            event_id = f"{prefix}{event_name}"
            node = self.get_node(event_id)
            if node:
                participants = node.data.get("participants", [])
                if participants:
                    if isinstance(participants, dict):
                        return list(participants.keys())
                    elif isinstance(participants, list):
                        return participants
                participant_ids = []
                for r in self.relationships:
                    if r.source_id == event_id and r.relation == "participant":
                        participant_name = r.target_id.split(":", 1)[1] if ":" in r.target_id else r.target_id
                        participant_ids.append(participant_name)
                return participant_ids
        return []

    def rebuild(self):
        """Rebuild the graph from source files."""
        self.nodes = {}
        self.relationships = []
        self._build_from_source_files()

        if self.persist:
            graph_path = self.history_path / "entity_graph.json"
            self._save_to_disk(graph_path)

    def rebuild_if_needed(self, force: bool = False):
        """
        Rebuild the graph if source files have been modified.

        Args:
            force: If True, always rebuild. If False, only rebuild if sources changed.
        """
        if force:
            self.rebuild()
            self.source_fingerprint = self._compute_source_fingerprint()
            return
        current = self._compute_source_fingerprint()
        if current == self.source_fingerprint:
            return
        self.rebuild()
        self.source_fingerprint = current

    def get_relationship_summary(self, character_name: str, max_relationships: int = 5, threshold: int = 50) -> list[dict]:
        """
        Get a brief summary of a character's important relationships for context inclusion.

        Returns a simple list of relationships suitable for always including in context
        without granular details.

        Args:
            character_name: Name of the character
            max_relationships: Maximum number of relationships to include (default: 5)
            threshold: Minimum importance score (default: 50)

        Returns:
            List of dicts with basic relationship info: {target, relation, importance}
        """
        relationships = self.get_important_relationships(character_name, threshold=threshold)

        # Sort by importance and take top N
        sorted_rels = sorted(relationships, key=lambda r: r.importance, reverse=True)[:max_relationships]

        return [
            {
                "target": r.target_id.split(":", 1)[1] if ":" in r.target_id else r.target_id,
                "relation": r.relation,
                "status": r.status,
                "importance": r.importance
            }
            for r in sorted_rels
        ]

    def get_all_important_relationships_for_characters(self, character_names: list[str], threshold: int = 50) -> dict[str, list[dict]]:
        """
        Get important relationships for multiple characters.

        Args:
            character_names: List of character names
            threshold: Minimum importance score (default: 50)

        Returns:
            Dict mapping character names to their important relationships
        """
        result = {}
        for char_name in character_names:
            char_id = f"character:{char_name}"
            if char_id in self.nodes:
                result[char_name] = self.get_relationship_summary(char_name, threshold=threshold)
        return result

    def build_adjacency(self):
        """Build adjacency dict from relationships list for O(1) traversal."""
        self.adjacency = {}
        for rel in self.relationships:
            source_type = rel.source_id.split(":")[0] if ":" in rel.source_id else "unknown"
            target_type = rel.target_id.split(":")[0] if ":" in rel.target_id else "unknown"

            if source_type not in self.adjacency:
                self.adjacency[source_type] = {}
            if target_type not in self.adjacency[source_type]:
                self.adjacency[source_type][target_type] = set()
            self.adjacency[source_type][target_type].add(rel.target_id)

        self.adjacency_built = True
        print(f"{_DEBUG}Built adjacency: {len(self.adjacency)} source types{_RESET}")

    def get_neighbors(
        self,
        entity_name: str,
        entity_type: str,
        target_type: str | None = None,
        field_name: str | None = None,
        min_importance: int = 0,
        direction: str = "outgoing"
    ):
        """Get neighboring entity names through relationships.

        Args:
            entity_name: Name of the entity
            entity_type: Type of entity (character, group, event)
            target_type: Optional target type to filter by
            field_name: Optional field name to filter by (relationships, group_status, etc.)
            min_importance: Minimum importance score
            direction: "outgoing", "incoming", or "both"

        Returns:
            List of neighboring entity names (without type prefix)
        """
        entity_id = f"{entity_type}:{entity_name}"
        result = []

        for rel in self.relationships:
            include = False
            matched_from_outgoing = False

            if direction in ("outgoing", "both"):
                if rel.source_id == entity_id:
                    include = True
                    matched_from_outgoing = True
                    if target_type and not rel.target_id.startswith(target_type + ":"):
                        include = False
                    if field_name and rel.field_name != field_name:
                        include = False
                    if min_importance and rel.importance < min_importance:
                        include = False

            if direction in ("incoming", "both") and not include:
                if rel.target_id == entity_id:
                    include = True
                    matched_from_outgoing = False
                    if target_type and not rel.source_id.startswith(target_type + ":"):
                        include = False
                    if field_name and rel.field_name != field_name:
                        include = False
                    if min_importance and rel.importance < min_importance:
                        include = False

            if include:
                target = rel.target_id if matched_from_outgoing else rel.source_id
                target_clean = target.split(":", 1)[1] if ":" in target else target
                if target_clean not in result:
                    result.append(target_clean)

        return result

    def query(
        self,
        source_entity: str | None = None,
        source_type: str | None = None,
        target_type: str | None = None,
        field_name: str | None = None,
        min_importance: int = 0,
        max_importance: int = 100,
        current_scene: int | None = None,
        direction: str = "outgoing",
    ) -> list[Relationship]:
        """Unified query method for graph traversal and filtering.

        This method provides a generic interface to query relationships with
        various filters, replacing multiple specialized get_* methods.

        Args:
            source_entity: Name of source entity (e.g., "John Jones")
            source_type: Type of source entity ("character", "group", "event", "scene")
            target_type: Optional target type to filter by ("character", "group", "event")
            field_name: Optional field name to filter by ("relationships", "group_status", "milestones")
            min_importance: Minimum importance score (inclusive)
            max_importance: Maximum importance score (inclusive)
            current_scene: Optional scene number to filter relationships by scene presence
            direction: "outgoing", "incoming", or "both"

        Returns:
            List of Relationship objects matching all criteria

        Examples:
            # Get all character milestones with importance >= 75
            graph.query(source_entity="John", source_type="character", field_name="milestones", min_importance=75)

            # Get group status for character in scene 5
            graph.query(source_entity="John", source_type="character", target_type="group", field_name="group_status", current_scene=5)

            # Get all events a character is involved in
            graph.query(source_entity="John", source_type="character", target_type="event")
        """
        results = []

        for rel in self.relationships:
            source_id = rel.source_id
            target_id = rel.target_id

            source_parts = source_id.split(":", 1) if ":" in source_id else ["", source_id]
            source_prefix = source_parts[0]
            source_name = source_parts[1] if len(source_parts) > 1 else source_id

            target_parts = target_id.split(":", 1) if ":" in target_id else ["", target_id]
            target_prefix = target_parts[0]
            target_name = target_parts[1] if len(target_parts) > 1 else target_id

            field_match = True
            if field_name:
                field_match = rel.field_name == field_name

            importance_match = min_importance <= rel.importance <= max_importance

            scene_match = True
            if current_scene is not None and rel.scenes:
                scene_match = current_scene in rel.scenes

            if direction == "outgoing":
                source_match = source_name == source_entity if source_entity else True
                type_match = source_prefix == source_type.lower() if source_type else True
                target_type_match = target_prefix == target_type.lower() if target_type else True
            elif direction == "incoming":
                source_match = target_name == source_entity if source_entity else True
                type_match = target_prefix == source_type.lower() if source_type else True
                target_type_match = source_prefix == target_type.lower() if target_type else True
            else:
                source_match = (source_name == source_entity if source_entity else True) or (target_name == source_entity if source_entity else True)
                type_match = (source_prefix == source_type.lower() if source_type else True) or (target_prefix == source_type.lower() if source_type else True)
                target_type_match = (target_prefix == target_type.lower() if target_type else True) or (source_prefix == target_type.lower() if target_type else True)

            if source_match and type_match and target_type_match and field_match and importance_match and scene_match:
                results.append(rel)

        return results

    def query_names(
        self,
        source_entity: str | None = None,
        source_type: str | None = None,
        target_type: str | None = None,
        field_name: str | None = None,
        min_importance: int = 0,
        max_importance: int = 100,
        current_scene: int | None = None,
        direction: str = "outgoing",
    ) -> list[str]:  # Unused
        """Query and return just target entity names (convenience wrapper).

        Args:
            Same as query()

        Returns:
            List of target entity names (without type prefix)
        """
        relationships = self.query(
            source_entity=source_entity,
            source_type=source_type,
            target_type=target_type,
            field_name=field_name,
            min_importance=min_importance,
            max_importance=max_importance,
            current_scene=current_scene,
            direction=direction,
        )

        names = []
        for rel in relationships:
            target_id = rel.target_id
            if ":" in target_id:
                target_name = target_id.split(":", 1)[1]
            else:
                target_name = target_id
            if target_name not in names:
                names.append(target_name)

        return names

    def traverse_graph(
        self,
        initial_entities: dict[str, set[str]],
        field_map: dict[str, dict[str, str]] | None = None,
        min_importance: int = 75,
        max_depth: int = 10,
    ) -> dict[str, set[str]]:
        """Traverse graph from initial entities, returning all discovered entities by type.

        Args:
            initial_entities: Dict of entity_type -> set of entity names
                e.g., {"character": {"John", "Amy"}, "group": {"Rebel Force"}}
            field_map: Optional relationship map (source_type -> {field_name: target_type})
                If None, derived from schema_classes
            min_importance: Minimum importance score for filtering
            max_depth: Maximum traversal depth

        Returns:
            Dict of entity_type -> set of discovered entity names
        """
        if field_map is None:
            field_map = self.get_schema_relationship_map()

        # Initialize relevant with all entity types from field_map and initial_entities
        all_entity_types = set(field_map.keys()) | set(initial_entities.keys())
        relevant: dict[str, set[str]] = {etype: set() for etype in all_entity_types}

        for etype, names in initial_entities.items():
            relevant[etype].update(names)

        visited: dict[str, set[str]] = {etype: set() for etype in all_entity_types}

        for depth in range(max_depth):
            new_entities = False

            for source_type in list(relevant.keys()):
                source_names = relevant.get(source_type, set())
                if not source_names:
                    continue

                field_config = field_map.get(source_type, {})
                if not field_config:
                    continue

                for source_name in list(source_names):
                    if source_name in visited.get(source_type, set()):
                        continue
                    visited[source_type].add(source_name)

                    for field_name, target_type in field_config.items():
                        neighbors = self.get_neighbors(
                            source_name, source_type,
                            target_type=target_type,
                            field_name=field_name if field_name != "_self" else None,
                            min_importance=min_importance,
                            direction="outgoing"
                        )

                        for neighbor in neighbors:
                            # Ensure target_type exists in relevant dict
                            if target_type not in relevant:
                                relevant[target_type] = set()
                            if target_type not in visited:
                                visited[target_type] = set()

                            if neighbor not in relevant[target_type]:
                                relevant[target_type].add(neighbor)
                                new_entities = True

            if not new_entities and depth > 0:
                break

        return relevant

    def get_schema_relationship_map(self) -> dict[str, dict[str, str]]:
        """Load relationship map from schema_classes.

        Returns:
            Dict of source_type -> {field_name: target_type}
            e.g., {"character": {"relationships": "character", "group_status": "group", ...}}
        """
        field_map: dict[str, dict[str, str]] = {}

        for schema_name, schema_class in self.schema_classes.items():
            source_type = self._singularize(schema_name)
            rel_fields = schema_class.get_relationship_fields()
            field_map[source_type] = {}

            for field_name, config in rel_fields.items():
                target_type = self._singularize(config.get("target_type", "unknown"))
                field_map[source_type][field_name] = target_type

        return field_map