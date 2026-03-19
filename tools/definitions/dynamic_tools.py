from typing import Any, Callable, TYPE_CHECKING
import json

if TYPE_CHECKING:
    from extensions.dayna_ss.agents.summarizer import Summarizer


def create_dss_tool_definitions() -> list[dict]:
    """Create OpenAI-compatible tool definitions for DSS tools."""
    return [
        {
            "type": "function",
            "function": {
                "name": "dss_get_info",
                "description": """Retrieve information from the story knowledge base using a dot-notation path.
            
Use this when you need specific information about characters, groups, scenes, events, or general story info.
The path follows the schema structure: category.subcategory.field

Examples:
  - "characters.John Jones" - Get all info about John Jones
  - "characters.John Jones.relationships" - Get John's relationships
  - "characters.John Jones.relationships.Mary" - Get John's relationship with Mary
  - "current_scene.now.where" - Get current location
  - "current_scene.now.who.characters" - Get characters in current scene
  - "events.scenes" - Get all scenes
  - "events.scenes.Battle of Endor" - Get specific scene
  - "general_info.writing_style" - Get writing style
  - "groups.The Rebellion" - Get Rebellion group info

Available top-level categories: characters, groups, current_scene, events, general_info""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Dot-notation path to the information (e.g., 'characters.John.relationships' or 'current_scene.now.where')"
                        }
                    },
                    "required": ["path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "dss_search_info",
                "description": """Search the story knowledge base using a natural language query.
            
Use this when you're not sure of the exact path, or when you want to find information related to a concept.
The search will look through names, descriptions, summaries, and other text fields.

Examples:
  - "Who is John's brother?" - Find John's brother relationship
  - "current battle scene" - Find current battle-related scene
  - "relationship between characters" - Find character relationship info
  - "main character's objectives" - Find character objectives
  - "recent events" - Find recent story events
  - "the villain's plan" - Find information about villain plans""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language search query (e.g., 'Who is John?', 'current location', 'recent events')"
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "dss_list_paths",
                "description": """List available information paths in the knowledge base.
            
Use this to discover what information is available when you don't know the exact structure.

Examples:
  - "" or omit parameter - List top-level categories
  - "characters" - List paths under characters
  - "characters.John Jones" - List paths for John Jones""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Parent path to list children of (leave empty for top-level)",
                            "default": ""
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "dss_set_info",
                "description": """Update information in the story knowledge base using a dot-notation path.
            
Use this to record changes to characters, scenes, events, or other story elements.

Examples:
  - path="characters.John Jones.status.current_mood", value="angry" - Update John's mood
  - path="current_scene.now.where", value="The Dark Forest" - Update location
  - path="events.scenes.Battle Now.ending", value="The heroes won" - Update scene ending

CAUTION: This modifies the knowledge base. Use dss_get_info first to understand the current state.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Dot-notation path to the field to update"
                        },
                        "value": {
                            "description": "The new value (string, number, boolean, or object depending on field type)"
                        }
                    },
                    "required": ["path", "value"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "dss_close_arc",
                "description": """Close an active narrative arc, marking it as concluded.

Use this when a narrative arc has reached its natural conclusion and should no longer be considered active.
An arc's ending scene will be set to the current scene index.

Examples:
  - arc_title="The Betrayal" - Close the arc titled "The Betrayal"
  - arc_title="The Hunt", conclusion="The heroes found the artifact" - Close with summary""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "arc_title": {
                            "type": "string",
                            "description": "Title of the arc to close (will use fuzzy matching)"
                        },
                        "conclusion_summary": {
                            "type": "string",
                            "description": "Optional summary of how this arc concluded"
                        }
                    },
                    "required": ["arc_title"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "dss_list_arcs",
                "description": """List all narrative arcs in the story.

Use this to see existing arcs, their status, and which scenes they're active in.

Examples:
  - "" or no arguments - List all arcs""",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        }
    ]


def create_dss_tool_executors(summarizer: "Summarizer") -> dict[str, Callable]:
    """Create executor functions for DSS tools."""
    
    def dss_get_info(arguments: dict) -> str:
        path = arguments.get("path", "")
        result = _get_info(summarizer, path)
        return json.dumps(result, indent=2, ensure_ascii=False)
    
    def dss_search_info(arguments: dict) -> str:
        query = arguments.get("query", "")
        result = _search_info(summarizer, query)
        return json.dumps(result, indent=2, ensure_ascii=False)
    
    def dss_list_paths(arguments: dict) -> str:
        path = arguments.get("path", "")
        result = _list_paths(summarizer, path)
        return json.dumps(result, indent=2, ensure_ascii=False)
    
    def dss_set_info(arguments: dict) -> str:
        path = arguments.get("path", "")
        value = arguments.get("value")
        result = _set_info(summarizer, path, value)
        return json.dumps(result, indent=2, ensure_ascii=False)
    
    def dss_close_arc(arguments: dict) -> str:
        arc_title = arguments.get("arc_title", "")
        conclusion_summary = arguments.get("conclusion_summary", "")
        result = _close_arc(summarizer, arc_title, conclusion_summary)
        return json.dumps(result, indent=2, ensure_ascii=False)
    
    def dss_list_arcs(arguments: dict) -> str:
        result = _list_arcs(summarizer)
        return json.dumps(result, indent=2, ensure_ascii=False)
    
    return {
        "dss_get_info": dss_get_info,
        "dss_search_info": dss_search_info,
        "dss_list_paths": dss_list_paths,
        "dss_set_info": dss_set_info,
        "dss_close_arc": dss_close_arc,
        "dss_list_arcs": dss_list_arcs,
    }


def _get_info(summarizer: "Summarizer", path: str) -> dict:
    if not summarizer.last or not summarizer.last.context:
        return {"error": "No context available. Start a conversation first."}

    retrieval_context = summarizer.last.context[0]
    parts = path.split(".")

    try:
        result = _navigate_path(retrieval_context, parts)
        if result is None:
            return {
                "error": f"Path '{path}' not found",
                "suggestion": "Use dss_list_paths('') to see available top-level categories."
            }
        return {"path": path, "result": result}
    except Exception as e:
        return {"error": f"Error accessing path '{path}': {str(e)}"}


def _navigate_path(data: Any, parts: list[str]) -> Any:
    """Navigate through nested dict/list structure."""
    if not parts:
        return data

    current = data
    for part in parts:
        if current is None:
            return None

        if isinstance(current, dict):
            if part in current:
                current = current[part]
            else:
                for key, value in current.items():
                    if isinstance(key, str):
                        key_lower = key.lower()
                        part_lower = part.lower()
                        if part_lower in key_lower or key_lower in part_lower:
                            current = value
                            break
                else:
                    return None
        elif isinstance(current, (list, tuple)) and part.isdigit():
            idx = int(part)
            if 0 <= idx < len(current):
                current = current[idx]
            else:
                return None
        else:
            return None

    return current


def _search_info(summarizer: "Summarizer", query: str) -> dict:
    if not summarizer.last or not summarizer.last.context:
        return {"error": "No context available. Start a conversation first.", "results": []}

    retrieval_context = summarizer.last.context[0]
    query_lower = query.lower()
    results = []

    for attr_name in ["characters", "groups", "events", "general_info", "current_scene", "arcs"]:
        data = getattr(retrieval_context, attr_name, None)
        if data is not None:
            results.extend(_search_in_dict(data, attr_name, query_lower))

    results = _deduplicate_results(results)

    if not results:
        return {
            "query": query,
            "results": [],
            "suggestion": "Try being more specific or use dss_list_paths('') to explore available data."
        }

    return {
        "query": query,
        "results": results[:10],
        "total_found": len(results)
    }


def _search_in_dict(data: Any, category: str, query: str) -> list[dict]:
    """Recursively search through dict/list for matching text."""
    results = []

    if isinstance(data, dict):
        for key, value in data.items():
            key_str = str(key).lower()
            value_str = str(value).lower()

            if query in key_str or query in value_str:
                results.append({
                    "category": category,
                    "path": f"{category}.{key}",
                    "matched_on": key_str if query in key_str else None,
                    "preview": _truncate_preview(value)
                })

            if isinstance(value, (dict, list)):
                nested = _search_in_dict(value, f"{category}.{key}", query)
                results.extend(nested)

    elif isinstance(data, (list, tuple)):
        for idx, item in enumerate(data):
            if isinstance(item, (dict, list)):
                nested = _search_in_dict(item, category, query)
                results.extend(nested)
            else:
                item_str = str(item).lower()
                if query in item_str:
                    results.append({
                        "category": category,
                        "matched_on": item_str,
                        "preview": _truncate_preview(item)
                    })

    return results


def _truncate_preview(value: Any, max_len: int = 200) -> str:
    """Create a preview of a value, truncated for display."""
    if isinstance(value, str):
        preview = value.strip()
    else:
        preview = json.dumps(value, indent=None) if isinstance(value, (dict, list)) else str(value)

    if len(preview) > max_len:
        preview = preview[:max_len] + "..."
    return preview


def _deduplicate_results(results: list[dict]) -> list[dict]:
    """Remove duplicate results based on path."""
    seen = set()
    deduped = []

    for result in results:
        path = result.get("path", result.get("matched_on", ""))
        if path not in seen:
            seen.add(path)
            deduped.append(result)

    return deduped


def _list_paths(summarizer: "Summarizer", path: str) -> dict:
    if not summarizer.last or not summarizer.last.context:
        return {"error": "No context available. Start a conversation first.", "paths": []}

    retrieval_context = summarizer.last.context[0]

    if not path:
        return {
            "path": "",
            "paths": ["characters", "groups", "current_scene", "events", "general_info", "arcs"],
            "description": "Top-level categories in the knowledge base"
        }

    parts = path.split(".")
    try:
        data = _navigate_path(retrieval_context, parts)
    except Exception:
        data = None

    if data is None:
        return {
            "path": path,
            "error": f"Path '{path}' not found",
        }

    if isinstance(data, dict):
        keys = list(data.keys())
        return {
            "path": path,
            "paths": keys,
            "count": len(keys),
            "hint": f"Access with {path}.<name>"
        }
    elif isinstance(data, (list, tuple)):
        return {
            "path": path,
            "count": len(data),
            "hint": f"{path} is a list with {len(data)} items"
        }
    else:
        return {
            "path": path,
            "value": data,
            "type": type(data).__name__,
        }


def _set_info(summarizer: "Summarizer", path: str, value: Any) -> dict:
    if not summarizer.last or not summarizer.last.context:
        return {"error": "No context available. Start a conversation first."}

    retrieval_context = summarizer.last.context[0]
    parts = path.split(".")

    if len(parts) < 2:
        return {
            "error": f"Path '{path}' is too short",
            "hint": "Paths should be at least 2 levels (e.g., 'characters.John' or 'current_scene.now.where')"
        }

    try:
        _set_path_value(retrieval_context, parts, value)
        return {
            "success": True,
            "path": path,
            "new_value": value,
            "message": f"Updated {path}"
        }
    except Exception as e:
        return {"error": f"Failed to set path '{path}': {str(e)}"}


def _set_path_value(data: dict, parts: list[str], value: Any) -> None:
    """Set a value at a specific path in a nested dict."""
    current = data

    for i, part in enumerate(parts[:-1]):
        if part not in current:
            current[part] = {}
        current = current[part]

        if not isinstance(current, dict):
            raise ValueError(f"Cannot traverse non-dict at path segment '{part}'")

    final_key = parts[-1]
    if isinstance(current, dict):
        current[final_key] = value
    else:
        raise ValueError(f"Cannot set value on non-dict container at '{final_key}'")


def _close_arc(summarizer: "Summarizer", arc_title: str, conclusion_summary: str = "") -> dict:
    """Close an active narrative arc.
    
    Args:
        summarizer: The Summarizer instance
        arc_title: Title of the arc to close (fuzzy matched)
        conclusion_summary: Optional summary of how the arc concluded
        
    Returns:
        Result dict with success status and details
    """
    if not summarizer.last or not summarizer.last.history_path:
        return {"error": "No history path available. Start a conversation first."}
    
    arcs_path = summarizer.last.history_path / "arcs.json"
    
    try:
        # Load arcs data
        if arcs_path.exists():
            with open(arcs_path, "r", encoding="utf-8") as f:
                arcs_data = json.load(f)
        else:
            arcs_data = []
        
        # Find arc by fuzzy matching title
        arc_title_lower = arc_title.lower()
        arc_index = None
        matched_title = None
        
        for i, arc in enumerate(arcs_data):
            arc_title_in_arc = arc.get("title", "").lower()
            if arc_title_lower in arc_title_in_arc or arc_title_in_arc in arc_title_lower:
                arc_index = i
                matched_title = arc.get("title")
                break
        
        if arc_index is None:
            return {
                "error": f"Arc '{arc_title}' not found",
                "available_arcs": [arc.get("title") for arc in arcs_data if arc.get("status") == "active"],
                "hint": "Use dss_list_arcs() to see available arcs"
            }
        
        # Get current scene index from events
        current_scene_idx = 0
        events_path = summarizer.last.history_path / "events.json"
        if events_path.exists():
            with open(events_path, "r", encoding="utf-8") as f:
                events_data = json.load(f)
                scenes = events_data.get("scenes", {})
                current_scene_idx = len(scenes) - 1 if scenes else 0
        
        # Close the arc
        arcs_data[arc_index]["ending_scene"] = current_scene_idx
        arcs_data[arc_index]["status"] = "concluded"
        
        if conclusion_summary:
            arcs_data[arc_index]["summary"] = (
                arcs_data[arc_index].get("summary", "") + 
                f"\n\nConclusion: {conclusion_summary}"
            )
        
        # Save arcs data
        with open(arcs_path, "w", encoding="utf-8") as f:
            json.dump(arcs_data, f, indent=2)
        
        return {
            "success": True,
            "arc_title": matched_title,
            "ending_scene": current_scene_idx,
            "status": "concluded",
            "message": f"Arc '{matched_title}' has been closed"
        }
        
    except Exception as e:
        return {"error": f"Failed to close arc: {str(e)}"}


def _list_arcs(summarizer: "Summarizer") -> dict:
    """List all narrative arcs.
    
    Args:
        summarizer: The Summarizer instance
        
    Returns:
        Dict with list of arcs and their details
    """
    if not summarizer.last or not summarizer.last.history_path:
        return {"error": "No history path available. Start a conversation first.", "arcs": []}
    
    arcs_path = summarizer.last.history_path / "arcs.json"
    
    try:
        if not arcs_path.exists():
            return {
                "arcs": [],
                "message": "No arcs have been created yet"
            }
        
        with open(arcs_path, "r", encoding="utf-8") as f:
            arcs_data = json.load(f)
        
        # Format arcs for display
        formatted_arcs = []
        for arc in arcs_data:
            formatted_arcs.append({
                "title": arc.get("title", "Untitled"),
                "status": arc.get("status", "unknown"),
                "starting_scene": arc.get("starting_scene"),
                "ending_scene": arc.get("ending_scene"),
                "scenes": arc.get("scenes", []),
                "summary": _truncate_preview(arc.get("summary", ""), max_len=300)
            })
        
        # Group by status
        active = [a for a in formatted_arcs if a["status"] == "active"]
        concluded = [a for a in formatted_arcs if a["status"] == "concluded"]
        
        return {
            "total": len(formatted_arcs),
            "active_count": len(active),
            "concluded_count": len(concluded),
            "active_arcs": active,
            "concluded_arcs": concluded,
            "all_arcs": formatted_arcs
        }
        
    except Exception as e:
        return {"error": f"Failed to list arcs: {str(e)}", "arcs": []}
