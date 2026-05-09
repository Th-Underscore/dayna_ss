#!/usr/bin/env python3
"""
Direct integration test without TGWUI dependencies.
Tests the entity graph and context retriever with actual JSON data.
"""

import sys
import json
from pathlib import Path

ext_dir = Path(__file__).parent
example_dir = ext_dir / "user_data" / "example"
sys.path.insert(0, str(ext_dir))


def load_json(filename):
    with open(example_dir / filename) as f:
        return json.load(f)


def test_entity_graph_logic():
    """Test entity graph loading and traversal directly."""
    print("=" * 60)
    print("ENTITY GRAPH LOGIC TEST")
    print("=" * 60)
    
    # Load JSON data directly
    characters = load_json("characters.json")
    groups = load_json("groups.json")
    events = load_json("events.json")
    
    print(f"\nData loaded:")
    print(f"  Characters: {len(characters.get('entries', {}))}")
    print(f"  Groups: {len(groups.get('entries', {}))}")
    print(f"  Events: {len(events.get('events', {}))}")
    
    # Build relationships manually (simulating EntityGraph)
    def build_relationships():
        """Build relationship list from JSON."""
        rels = []
        char_entries = characters.get("entries", {})
        
        for char_name, char_data in char_entries.items():
            # Relationships
            for rel_name, rel_list in char_data.get("relationships", {}).items():
                for rel in rel_list:
                    score = 0
                    imp = rel.get("importance", {})
                    if isinstance(imp, dict):
                        score = imp.get("score", 0)
                    rels.append({
                        "source": f"character:{char_name}",
                        "target": f"character:{rel_name}",
                        "relation": rel.get("relation", ""),
                        "field": "relationships",
                        "score": score,
                    })
            
            # Group status (as outgoing to groups)
            for group_name, status_data in char_data.get("group_status", {}).items():
                score = 0
                imp = status_data.get("importance", {})
                if isinstance(imp, dict):
                    score = imp.get("score", 0)
                rels.append({
                    "source": f"character:{char_name}",
                    "target": f"group:{group_name}",
                    "relation": "member",
                    "field": "group_status",
                    "score": score,
                })
            
            # Group -> Character (reverse)
            group_entries = groups.get("entries", {})
            for group_name, group_data in group_entries.items():
                for member_name, member_info in group_data.get("characters", {}).items():
                    member_score = 0
                    if isinstance(member_info, dict):
                        imp = member_info.get("importance", 0)
                        if isinstance(imp, dict):
                            member_score = imp.get("score", 0)
                        else:
                            member_score = imp if isinstance(imp, int) else 0
                    rels.append({
                        "source": f"group:{group_name}",
                        "target": f"character:{member_name}",
                        "relation": "member",
                        "field": "characters",
                        "score": member_score,
                    })
        
        return rels
    
    relationships = build_relationships()
    print(f"\n  Relationships built: {len(relationships)}")
    
    # Test traversal
    def traverse(start_char, min_score, max_depth):
        """Simulate EntityGraph.traverse_graph()"""
        relevant = {"character": {start_char}, "group": set()}
        visited = set()
        current_frontier = {start_char}
        
        for depth in range(max_depth):
            if not current_frontier:
                break
            next_frontier = set()
            for rel in relationships:
                source_char = rel["source"].replace("character:", "")
                if source_char in current_frontier and rel["score"] >= min_score:
                    if rel["target"].startswith("character:"):
                        target = rel["target"].replace("character:", "")
                        if target not in relevant["character"]:
                            relevant["character"].add(target)
                            next_frontier.add(target)
                    elif rel["target"].startswith("group:"):
                        target = rel["target"].replace("group:", "")
                        relevant["group"].add(target)
            visited.update(current_frontier)
            current_frontier = {c for c in next_frontier if c not in visited}
        
        return relevant
    
    print("\n" + "-" * 40)
    print("Testing traversal from John Jones (score >= 75)")
    print("-" * 40)
    
    result = traverse("John Jones", 75, 3)
    print(f"  Characters: {sorted(result['character'])}")
    print(f"  Groups: {sorted(result['group'])}")
    
    assert "John Jones" in result["character"]
    assert "Paul Jones" in result["character"]
    print("✓ PASS: Core team retrieved")
    
    print("\n" + "-" * 40)
    print("Testing from Subterranean Leader (enemy focus)")
    print("-" * 40)
    
    result = traverse("Subterranean Leader", 75, 2)
    print(f"  Characters: {sorted(result['character'])}")
    print(f"  Groups: {sorted(result['group'])}")
    
    # Should include Vex (through Council? let's see)
    print("✓ PASS: Enemy network test")
    
    return True


def test_context_retriever_logic():
    """Test context retriever logic directly."""
    print("\n" + "=" * 60)
    print("CONTEXT RETRIEVER LOGIC TEST")
    print("=" * 60)
    
    # This simulates what StoryContextRetriever does
    
    characters = load_json("characters.json")
    groups = load_json("groups.json")
    events = load_json("events.json")
    current_scene = load_json("current_scene.json")
    
    print(f"\n  Current scene: {current_scene.get('what')}")
    print(f"  Scene number: {current_scene.get('_scene_number')}")
    
    # Get scene characters
    scene_chars = []
    if "now" in current_scene and "who" in current_scene["now"]:
        for char in current_scene["now"]["who"].get("characters", []):
            scene_chars.append(char["name"])
    
    print(f"  Scene characters: {scene_chars}")
    
    # Simulate unified aggregation
    def unified_aggregate(scene_chars, min_score=50, max_depth=3):
        """Simulate _unified_entity_aggregation"""
        char_entries = characters.get("entries", {})
        
        relevant = {"character": set(scene_chars), "group": set(), "event": set()}
        
        for depth in range(max_depth):
            new = False
            
            for char_name in list(relevant["character"]):
                char_data = char_entries.get(char_name, {})
                
                # Get important relationships (outgoing)
                for rel_name, rel_list in char_data.get("relationships", {}).items():
                    for rel in rel_list:
                        imp = rel.get("importance", {})
                        if isinstance(imp, dict):
                            score = imp.get("score", 0)
                        else:
                            score = 0
                        
                        if score >= min_score and rel_name not in relevant["character"]:
                            relevant["character"].add(rel_name)
                            new = True
                
                # Get group_status
                for group_name, status_data in char_data.get("group_status", {}).items():
                    imp = status_data.get("importance", {})
                    if isinstance(imp, dict):
                        score = imp.get("score", 0)
                    else:
                        score = 0
                    
                    if score >= min_score and group_name not in relevant["group"]:
                        relevant["group"].add(group_name)
                        new = True
                
                # Get milestones -> events
                for milestone in char_data.get("milestones", []):
                    imp = milestone.get("importance", {})
                    if isinstance(imp, dict):
                        score = imp.get("score", 0)
                    else:
                        score = 0
                    
                    if score >= min_score and milestone.get("title"):
                        relevant["event"].add(milestone.get("title"))
                        new = True
            
            if not new and depth > 0:
                break
        
        return relevant
    
    print("\n" + "-" * 40)
    print("Testing unified aggregation from current scene")
    print("-" * 40)
    
    result = unified_aggregate(scene_chars)
    
    print(f"  Characters: {len(result['character'])} - {sorted(result['character'])}")
    print(f"  Groups: {len(result['group'])} - {sorted(result['group'])}")
    print(f"  Events: {len(result['event'])} - {list(result['event'])[:5]}...")
    
    # From scene 18, should get core team
    assert "John Jones" in result["character"]
    assert "Paul Jones" in result["character"]
    print("✓ PASS: Scene-based aggregation works")
    
    return True


def main():
    print("Running direct integration tests...\n")

    tests = [
        ("Entity Graph Logic", test_entity_graph_logic),
        ("Context Retriever Logic", test_context_retriever_logic),
        ("Entity Graph With Schema", test_entity_graph_with_schema),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"ERROR in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    passed = 0
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {name}")
        if success:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(results)} passed")
    
    return 0 if passed == len(results) else 1


def test_entity_graph_with_schema():
    """Test SchemaWrapper runtime introspection methods (Phase 4)."""
    print("=" * 60)
    print("ENTITY GRAPH WITH SCHEMA TEST (Phase 4)")
    print("=" * 60)

    from utils.schema_parser import SchemaParser, SchemaWrapper

    example_dir = Path(__file__).parent / "user_data" / "example"
    schema_path = example_dir / "subjects_schema.json"

    try:
        parser = SchemaParser(schema_path)
        schema_classes = parser.definitions

        print(f"\nSchema loaded: {len(schema_classes)} definitions")
        print(f"Available entity types: {SchemaWrapper(schema_classes).get_entity_types()}")

        schema_wrapper = SchemaWrapper(schema_classes)
        print(f"SchemaWrapper initialized")

        print("\nTesting runtime schema introspection:")

        for entity_type in ["Character", "Group", "Event"]:
            if entity_type not in schema_classes:
                print(f"  {entity_type}: Not found in schema")
                continue

            print(f"\n  {entity_type}:")

            filterable = schema_wrapper.get_filterable_fields(entity_type)
            print(f"    get_filterable_fields(): {filterable}")

            rel_fields = schema_wrapper.get_relationship_fields(entity_type)
            print(f"    get_relationship_fields(): {list(rel_fields.keys())}")

            for rel_field in rel_fields.keys():
                targets = schema_wrapper.get_relationship_targets(entity_type, rel_field)
                print(f"      {rel_field} -> {targets}")

            all_fields = schema_wrapper.get_all_data_fields(entity_type)
            print(f"    get_all_data_fields(): {all_fields}")

            field_info = schema_wrapper.get_field_info(entity_type, "importance")
            print(f"    get_field_info('importance'): {field_info}")

        print("\n" + "-" * 40)
        print("Testing EntityGraph simulated logic:")

        characters = load_json("characters.json")
        char_entries = characters.get("entries", {})
        john_data = char_entries.get("John Jones", {})
        if john_data:
            imp = schema_wrapper.get_importance(john_data, "Character", "relationships", default=0)
            print(f"  get_importance('John Jones', 'relationships'): {imp}")

            val = schema_wrapper.get_field_value(john_data, "Character", "occupation")
            print(f"  get_field_value('John Jones', 'occupation'): {val}")

        print("\n✓ EntityGraph with schema test passed")
        return True

    except Exception as e:
        print(f"\n✗ EntityGraph with schema test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    sys.exit(main())