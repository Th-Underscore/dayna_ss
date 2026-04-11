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
    importance: int = 0
    importance_reason: str = ""
    faction: str = "neutral"  # "positive", "negative", "neutral"
    aliases: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)


class EntityGraph:
    """
    Graph-based representation of story entities and their relationships.
    
    Currently hardcodes parsing for the existing schema (characters.json, groups.json, etc.)
    but is designed to accept schema-driven configuration in the future.
    """
    
    def __init__(self, history_path: Path, persist: bool = True):
        """
        Initialize the entity graph.
        
        Args:
            history_path: Path to the history directory containing subject JSON files.
            persist: Whether to save/load graph from disk for debugging.
        """
        self.history_path = Path(history_path)
        self.persist = persist
        
        self.nodes: dict[str, EntityNode] = {}
        self.relationships: list[Relationship] = []
        
        self._load_or_build()
    
    def _load_or_build(self):
        """Load from disk or build from source files."""
        graph_path = self.history_path / "entity_graph.json"

        if self.persist and graph_path.exists():
            # Check if source files have changed
            current_fingerprint = self._compute_source_fingerprint()
            data = load_json(graph_path)
            saved_fingerprint = data.get("source_fingerprint", {})

            if current_fingerprint == saved_fingerprint:
                self._load_from_disk(graph_path)
            else:
                print(f"{_DEBUG}Source files have changed, rebuilding entity graph...{_RESET}")
                self._build_from_source_files()
                self._save_to_disk(graph_path)
        else:
            self._build_from_source_files()
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
                aliases=r.get("aliases", []),
                events=r.get("events", [])
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
                    "aliases": r.aliases,
                    "events": r.events
                }
                for r in self.relationships
            ]
        }

        save_json(data, graph_path)
        print(f"{_DEBUG}Saved entity graph to {graph_path}{_RESET}")
    
    def _build_from_source_files(self):
        """Build graph from source JSON files."""
        print(f"{_BOLD}Building entity graph from source files...{_RESET}")
        
        self._build_characters()
        self._build_groups()
        self._build_events()
        self._build_arcs()
        
        print(f"{_SUCCESS}Built entity graph with {len(self.nodes)} nodes and {len(self.relationships)} relationships{_RESET}")
    
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

            # Extract and store participants
            participants = scene_data.get("participants", [])
            if participants:
                node.data["participants"] = participants
                # Create edges to participants
                for participant in participants:
                    # Ensure participant node exists (create if missing)
                    participant_id = f"character:{participant}"
                    if participant_id not in self.nodes:
                        # Create a minimal node for this participant
                        self.nodes[participant_id] = EntityNode(
                            id=participant_id,
                            type="character",
                            name=participant,
                            data={}
                        )

                    # Add edge from event to participant
                    relationship = Relationship(
                        source_id=node_id,
                        target_id=participant_id,
                        relation="participant",
                        status="event_participation",
                        importance=50
                    )
                    self.relationships.append(relationship)

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

            # Extract and store participants
            participants = event_data.get("participants", [])
            if participants:
                node.data["participants"] = participants
                # Create edges to participants
                for participant in participants:
                    # Ensure participant node exists (create if missing)
                    participant_id = f"character:{participant}"
                    if participant_id not in self.nodes:
                        # Create a minimal node for this participant
                        self.nodes[participant_id] = EntityNode(
                            id=participant_id,
                            type="character",
                            name=participant,
                            data={}
                        )

                    # Add edge from event to participant
                    relationship = Relationship(
                        source_id=node_id,
                        target_id=participant_id,
                        relation="participant",
                        status="event_participation",
                        importance=50
                    )
                    self.relationships.append(relationship)

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

            # Extract and store participants
            participants = event_data.get("participants", [])
            if participants:
                node.data["participants"] = participants
                # Create edges to participants
                for participant in participants:
                    # Ensure participant node exists (create if missing)
                    participant_id = f"character:{participant}"
                    if participant_id not in self.nodes:
                        # Create a minimal node for this participant
                        self.nodes[participant_id] = EntityNode(
                            id=participant_id,
                            type="character",
                            name=participant,
                            data={}
                        )

                    # Add edge from event to participant
                    relationship = Relationship(
                        source_id=node_id,
                        target_id=participant_id,
                        relation="participant",
                        status="event_participation",
                        importance=50
                    )
                    self.relationships.append(relationship)

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
    
    def _build_arcs(self):
        """Parse arcs.json to build arc nodes."""
        arcs_path = self.history_path / "arcs.json"
        if not arcs_path.exists():
            return
        
        arcs_data = load_json(arcs_path)

        if isinstance(arcs_data, list):
            arcs_list = arcs_data
        elif isinstance(arcs_data, dict):
            arcs_list = arcs_data.get("entries", arcs_data.get("arcs", []))
            # If arcs_list is a dict (e.g., {"entries": {...}}), convert to iterable of arc dicts
            if isinstance(arcs_list, dict):
                arcs_list = list(arcs_list.values())
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
    
    def get_scene_characters(self, scene_name: str) -> list[str]:
        """Get all characters in a scene."""
        # This would need to be enhanced with scene->character mapping
        # For now, return empty list - can be expanded later
        return []
    
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
    
    def get_relevant_events(self, character_names: list[str], group_names: list[str] | None = None, context: str = "") -> list[str]:
        """
        Get events relevant to the given characters and groups.
        
        Args:
            character_names: List of character names
            group_names: Optional list of group names
            context: Optional context string (for future text-based matching)
        
        Returns:
            List of relevant event/scene names
        """
        relevant = set()

        # Get events from character relationships
        for char_name in character_names:
            character_id = f"character:{char_name}"
            for r in self.relationships:
                if r.source_id == character_id or r.target_id == character_id:
                    relevant.update(r.events)

        # Get events from groups
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
            return node.relationships.get("members", [])
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
        # Try event and scene namespaces
        for prefix in ["event:", "scene:"]:
            event_id = f"{prefix}{event_name}"
            node = self.get_node(event_id)
            if node:
                # Return participants from node data
                participants = node.data.get("participants", [])
                if participants:
                    return participants
                # Or follow edges
                participant_ids = []
                for r in self.relationships:
                    if r.source_id == event_id and r.relation == "participant":
                        # Extract display name from namespaced ID
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
        
        Currently always rebuilds since we don't track file modification times.
        Can be enhanced later to compare file timestamps.
        
        Args:
            force: If True, always rebuild. If False, currently always rebuilds.
        """
        self.rebuild()
    
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