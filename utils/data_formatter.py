import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

# Placeholder for a more sophisticated template engine if needed
# For now, we'll implement a simple regex-based one.

class DataFormatter:
    """
    Formats data into strings using templates defined in a JSON configuration file.
    """

    def __init__(self, format_config_path: Union[str, Path]):
        """
        Initializes the DataFormatter.

        Args:
            format_config_path: Path to the JSON file containing format templates.
        """
        self.format_config_path = Path(format_config_path)
        self.templates_config = self._load_config()
        self.helpers = self._register_helpers()

    def _load_config(self) -> Dict[str, Any]:
        """Loads the formatting configuration JSON file."""
        try:
            with open(self.format_config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            # Basic validation
            if "format_templates" not in config and "type_string_overrides" not in config:
                raise ValueError(
                    "Configuration must contain 'format_templates' or 'type_string_overrides'"
                )
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Format configuration file not found: {self.format_config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding format configuration JSON: {e}")

    def _register_helpers(self) -> Dict[str, Callable]:
        """Registers built-in helper functions for templates."""
        return {
            "get_values": self._helper_get_values,
            "json_dump": self._helper_json_dump,
            "nl": lambda: "\n",
            # enumerate_list could be added if data isn't pre-processed
        }

    def _helper_get_values(self, data_obj: Any) -> List[Any]:
        """Template helper to get values from a dict or return a list as is."""
        if isinstance(data_obj, dict):
            return list(data_obj.values())
        if isinstance(data_obj, list):
            return data_obj
        return []

    def _helper_json_dump(self, data_obj: Any) -> str:
        """Template helper to dump an object to a JSON string."""
        try:
            return json.dumps(data_obj)
        except TypeError:
            return str(data_obj) # Fallback

    def _get_value_from_path(self, data_context: Any, relative_path_str: str, base_path_for_tracking: str, default: Optional[Any] = None) -> Tuple[Any, Optional[str]]:
        """
        Retrieves a value from the context using a dot-separated path.
        Returns the value and its fully constructed path string for tracking.
        Supports basic dictionary and list access (e.g., "key1.list_name.0.item_key").
        """
        if not relative_path_str: # Cannot resolve empty path
            return default, None

        parts = relative_path_str.split('.')
        current_val = data_context
        current_path_trail: List[str] = []

        try:
            for i, part in enumerate(parts):
                is_last_part = (i == len(parts) - 1)
                if isinstance(current_val, dict):
                    if part not in current_val:
                        return default, None
                    current_val = current_val[part]
                    current_path_trail.append(part)
                elif isinstance(current_val, list):
                    try:
                        idx = int(part)
                        # Check bounds only if not potentially slicing or if it's the last part
                        if not (0 <= idx < len(current_val)):
                             return default, None
                        current_val = current_val[idx]
                        current_path_trail.append(f"[{idx}]")
                    except ValueError: # Not an integer, maybe a method on a list?
                        if hasattr(current_val, part) and callable(getattr(current_val, part)):
                            # This case is complex: calling methods on lists during path resolution
                            # For now, we don't support method calls in paths.
                            return default, None
                        return default, None # Not an index or known attribute
                    except IndexError:
                        return default, None
                elif hasattr(current_val, part):
                    current_val = getattr(current_val, part)
                    current_path_trail.append(part)
                else:
                    return default, None
            
            final_path_str = base_path_for_tracking
            if not current_path_trail: # Should not happen if relative_path_str is not empty
                return current_val, base_path_for_tracking

            for p_idx, p_part in enumerate(current_path_trail):
                if p_part.startswith("["): # List index
                    final_path_str += p_part
                else: # Dict key or attribute
                    if final_path_str and not final_path_str.endswith(".") and p_idx > 0 and not current_path_trail[p_idx-1].startswith("["):
                        final_path_str += "." + p_part
                    elif final_path_str and not final_path_str.endswith("."):
                         final_path_str += "." + p_part
                    elif not final_path_str:
                         final_path_str = p_part
                    else: # base_path_for_tracking already ends with dot or is empty
                        final_path_str += p_part
            
            # Ensure base_path_for_tracking is prepended if it exists and trail is not empty
            if base_path_for_tracking and current_path_trail:
                constructed_trail = ""
                for p_idx, p_part in enumerate(current_path_trail):
                    if p_part.startswith("["):
                        constructed_trail += p_part
                    else:
                        if constructed_trail and not constructed_trail.endswith("."):
                             constructed_trail += "."
                        constructed_trail += p_part
                
                if base_path_for_tracking.endswith(".") and constructed_trail.startswith("."):
                    final_path_str = base_path_for_tracking + constructed_trail[1:]
                elif not base_path_for_tracking.endswith(".") and not constructed_trail.startswith(".") and base_path_for_tracking and constructed_trail:
                    final_path_str = base_path_for_tracking + "." + constructed_trail
                else:
                    final_path_str = base_path_for_tracking + constructed_trail

            elif not base_path_for_tracking and current_path_trail: # No base, just the trail
                constructed_trail = ""
                for p_idx, p_part in enumerate(current_path_trail):
                    if p_part.startswith("["):
                        constructed_trail += p_part
                    else:
                        if constructed_trail and not constructed_trail.endswith("."):
                             constructed_trail += "."
                        constructed_trail += p_part
                final_path_str = constructed_trail


            return current_val, final_path_str

        except Exception: # Broad catch for unexpected issues during path resolution
            return default, None

    def _render_simple_variables(self, template_str: str, data_context: Any, base_path_for_tracking: str, tracked_paths_list: List[str]) -> str:
        """
        Renders simple {{ variable }} and {{ path.to.variable 'default' }} placeholders.
        Tracks resolved paths.
        """
        def replace_variable(match):
            var_path = match.group(1).strip()
            default_val_str = match.group(3)
            default_val = None

            if default_val_str is not None:
                try:
                    default_val = json.loads(default_val_str)
                except json.JSONDecodeError:
                    default_val = default_val_str.strip("'\"")
            
            value, full_path = self._get_value_from_path(data_context, var_path, base_path_for_tracking, default_val)
            
            if full_path:
                tracked_paths_list.append(full_path)
            
            # If value was resolved to default due to path error, but a default was provided,
            # we still might want to track the attempted path if it was partially valid.
            # For now, only track if full_path is valid.

            return str(value) if value is not None else ""

        variable_pattern = re.compile(r"{{\s*([\w\.]+)\s*(?:(['\"](.*?)['\"]))?\s*}}")
        return variable_pattern.sub(replace_variable, template_str)

    _PATH_TRACK_CONTEXT_PATTERN = re.compile(r"{{\s*@track_context\s*}}")
    _PATH_TRACK_KEY_LITERAL_PATTERN = re.compile(r"{{\s*@track_key_literal\s+(['\"])(.*?)\1\s*}}")
    _PATH_OUTPUT_PATTERN = re.compile(r"{{\s*@paths\s*}}")

    def _render(self, template_str: str, data_context: Any, base_path_for_tracking: str) -> str:
        """
        Renders the given template string with the provided data and base path for tracking.
        """
        tracked_paths_this_scope: List[str] = []
        processed_template = template_str

        # Pass 1: Process {{@track_context}}
        def replace_track_context(match):
            if base_path_for_tracking:
                tracked_paths_this_scope.append(base_path_for_tracking)
            return ""
        processed_template = self._PATH_TRACK_CONTEXT_PATTERN.sub(replace_track_context, processed_template)

        # Pass 2: Process {{@track_key_literal "key"}}
        def replace_track_key_literal(match):
            key_literal = match.group(2)
            if base_path_for_tracking:
                tracked_paths_this_scope.append(f"{base_path_for_tracking}.{key_literal} (key)")
            else: # Should ideally have a base path for key tracking
                tracked_paths_this_scope.append(f"{key_literal} (key)")
            return ""
        processed_template = self._PATH_TRACK_KEY_LITERAL_PATTERN.sub(replace_track_key_literal, processed_template)

        # Pass 3: Process helpers {{ (helper_name args...) }}
        # Note: Arguments to helpers are resolved, but their paths are not automatically tracked yet.
        def replace_helper(match):
            full_match = match.group(0)
            expression = match.group(1).strip()
            parts = expression.split()
            helper_name = parts[0]
            args_str = parts[1:]
            
            resolved_args = []
            for arg_s in args_str:
                # Resolve argument, but don't add its path to tracked_paths_this_scope here.
                # Path tracking for helper arguments could be a future enhancement.
                val, _ = self._get_value_from_path(data_context, arg_s, base_path_for_tracking, arg_s)
                resolved_args.append(val)

            if helper_name in self.helpers:
                try:
                    return str(self.helpers[helper_name](*resolved_args))
                except Exception as e:
                    return f"[Error in helper {helper_name}: {e}]"
            return full_match
        
        helper_pattern = re.compile(r"{{\s*\(([\w\s\.\'\"]+)\)\s*}}") # Basic: {{ (helper arg) }}
        processed_template = helper_pattern.sub(replace_helper, processed_template)
        
        # Pass 4: Render simple variables {{ variable }} and collect their paths
        processed_template = self._render_simple_variables(processed_template, data_context, base_path_for_tracking, tracked_paths_this_scope)

        # Pass 5: Process {{@paths}}
        def replace_paths_output(match):
            if not tracked_paths_this_scope:
                return ""
            # Unique, sorted paths
            unique_paths = sorted(list(set(tracked_paths_this_scope)))
            paths_str = ",".join(unique_paths)
            tracked_paths_this_scope.clear() # Clear after use for this instance
            return f" <<<<<<<<<<<< {paths_str}"
        
        rendered_str = self._PATH_OUTPUT_PATTERN.sub(replace_paths_output, processed_template)
        
        # If there are multiple {{@paths}} directives, subsequent ones will be empty unless new paths are collected.
        # This simple multi-pass regex approach might need refinement for complex nested scopes with multiple {{@paths}}.
        # For now, {{@paths}} consumes all paths collected up to that point in its _render call.

        return rendered_str

    def format(self, data: Any, schema_class_name: Optional[str] = None, type_string_override: Optional[str] = None) -> str:
        """
        Formats the given data using the appropriate template.

        Args:
            data: The data object to format.
            schema_class_name: The name of the schema class for the data (e.g., "Character").
            type_string_override: A specific type string to look up in "type_string_overrides"
                                  (e.g., "character_list"). Takes precedence.

        Returns:
            The formatted string.
        """
        template_str = None
        template_source_name = None
        initial_base_path = ""

        if type_string_override:
            if type_string_override in self.templates_config.get("type_string_overrides", {}):
                template_str = self.templates_config["type_string_overrides"][type_string_override].get("template")
                template_source_name = f"type_string_override: {type_string_override}"
                initial_base_path = type_string_override # e.g., "character_list"
                 # Adjust if specific base needed, e.g. "characters" for "character_list"
                if type_string_override == "character_list": initial_base_path = "characters"
                if type_string_override == "characters_list": initial_base_path = "characters"
                if type_string_override == "scene" or type_string_override == "event": initial_base_path = type_string_override
                if type_string_override == "events" or type_string_override == "events_list": initial_base_path = "events"


        elif schema_class_name:
            if schema_class_name in self.templates_config.get("format_templates", {}):
                template_str = self.templates_config["format_templates"][schema_class_name].get("template")
                template_source_name = f"format_template: {schema_class_name}"
                initial_base_path = schema_class_name.lower() # e.g., "currentscene" -> "current_scene"
                # Convention: "current_scene" for "CurrentScene"
                if schema_class_name == "CurrentScene": initial_base_path = "current_scene"
                elif schema_class_name == "StoryEvent" and not type_string_override : initial_base_path = "event" # if not overridden by type_string

        if template_str is None:
            if type_string_override:
                return f"[No template found for type_string_override: {type_string_override}]"
            if schema_class_name:
                return f"[No template found for schema_class_name: {schema_class_name}]"
            return str(data)

        try:
            return self._render(template_str, data, initial_base_path)
        except Exception as e:
            # Log error or handle more gracefully
            return f"[Error rendering template '{template_source_name}': {e}]"

if __name__ == '__main__':
    # Example Usage (assuming formatted_class_strings.json is in the same directory or correct path)
    try:
        format_config_path = Path("formatted_class_strings.json") #Path(__file__).parent / "formatted_class_strings.json"
        dummy_config_path = Path("dummy_formatted_class_strings.json")
        if not format_config_path.exists() and not dummy_config_path.exists():
            dummy_content = {
                "format_templates": {
                    "User": {"template": "User: {{ name }} ({{ id }})"}
                },
                "type_string_overrides": {
                    "user_greeting": {"template": "Hello, {{ user.name }}! Your ID is {{ user.id }}."},
                    "user_list": {"template": "{{#each users as |user|}}User: {{user.name}}\n{{/each}}"}
                }
            }
            with open(dummy_config_path, "w") as f:
                json.dump(dummy_content, f, indent=2)
            config_to_use = dummy_config_path
            print(f"Using dummy config: {dummy_config_path}")
        else:
            config_to_use = format_config_path if format_config_path.exists() else dummy_config_path
            print(f"Using config: {config_to_use}")


        formatter = DataFormatter(config_to_use)

        # Test 1: Simple format_template
        user_data_schema = {"name": "Alice", "id": 123}
        print(f"\nTest 1 (User schema): {formatter.format(user_data_schema, schema_class_name='User')}")

        # Test 2: type_string_override
        user_data_type = {"user": {"name": "Bob", "id": 456}}
        print(f"\nTest 2 (user_greeting type): {formatter.format(user_data_type, type_string_override='user_greeting')}")

        # Test 3: Non-existent template
        print(f"\nTest 3 (Non-existent): {formatter.format({'data': 'test'}, schema_class_name='NonExistent')}")

        # Test 4: Default value in placeholder
        test_data_defaults = {"name": "Charlie"}
        default_template_config = {
            "format_templates": {"WithDefault": {"template": "Name: {{ name }}, Age: {{ age '30' }}, City: {{ city 'Unknown' }} {{@paths}}"}}
        }
        with open("temp_defaults_config.json", "w") as f:
            json.dump(default_template_config, f)
        formatter_defaults = DataFormatter("temp_defaults_config.json")
        print(f"\nTest 4 (Defaults): {formatter_defaults.format(test_data_defaults, schema_class_name='WithDefault')}")
        Path("temp_defaults_config.json").unlink()


        # Test 5: Helpers
        helper_test_data = {"items_dict": {"a":1, "b":2}, "items_list": [10,20]}
        helper_template_config = {
             "type_string_overrides": {
                "helper_test": {"template": "DictVals: {{ (get_values items_dict) }}\\nListVals: {{ (get_values items_list) }}\\nNL: {{ (nl) }}EndNL\\nJSON: {{ (json_dump items_dict) }} {{@paths}}"}
            }
        }
        with open("temp_helper_config.json", "w") as f:
            json.dump(helper_template_config, f)
        formatter_helpers = DataFormatter("temp_helper_config.json")
        print(f"\nTest 5 (Helpers): {formatter_helpers.format(helper_test_data, type_string_override='helper_test')}")
        Path("temp_helper_config.json").unlink()


        # Test 6: Path tracking
        path_test_data = {"user": {"name": "PathUser", "details": {"id": 789, "active": True}}, "items": ["item1", "item2"]}
        path_template_config = {
            "type_string_overrides": {
                "path_test_simple": {
                    "template": "User: {{user.name}} {{@paths}}. ID: {{user.details.id}} {{@paths}}. Active: {{user.details.active}} {{@paths}}"
                },
                "path_test_context": {
                    "template": "{{@track_context}}Item 0: {{items.0}} {{@paths}}"
                },
                "path_test_key_literal": {
                    "template": "{{@track_context}}{{@track_key_literal \"name\"}}Name: {{user.name}} {{@paths}}"
                },
                 "path_test_combined": {
                    "template": "{{@track_context}}User: {{user.name}}, ID: {{user.details.id}}. Items: {{items.0}}, {{items.1}}. {{@paths}}"
                }
            }
        }
        with open("temp_path_config.json", "w") as f:
            json.dump(path_template_config, f, indent=2)
        
        formatter_paths = DataFormatter("temp_path_config.json")
        print(f"\nTest 6.1 (Path Simple): {formatter_paths.format(path_test_data, type_string_override='path_test_simple')}")
        # Expected: User: PathUser <<<<<<<<<<<< path_test_simple.user.name. ID: 789 <<<<<<<<<<<< path_test_simple.user.details.id. Active: True <<<<<<<<<<<< path_test_simple.user.details.active
        
        print(f"\nTest 6.2 (Path Context): {formatter_paths.format(path_test_data, type_string_override='path_test_context')}")
        # Expected: Item 0: item1 <<<<<<<<<<<< path_test_context,path_test_context.items[0]

        # For path_test_key_literal, base path is "path_test_key_literal"
        # {{@track_context}} adds "path_test_key_literal"
        # {{@track_key_literal "name"}} adds "path_test_key_literal.name (key)"
        # {{user.name}} adds "path_test_key_literal.user.name"
        print(f"\nTest 6.3 (Path Key Literal): {formatter_paths.format(path_test_data, type_string_override='path_test_key_literal')}")
        # Expected: Name: PathUser <<<<<<<<<<<< path_test_key_literal,path_test_key_literal.name (key),path_test_key_literal.user.name
        
        print(f"\nTest 6.4 (Path Combined): {formatter_paths.format(path_test_data, type_string_override='path_test_combined')}")
        # Expected: User: PathUser, ID: 789. Items: item1, item2. <<<<<<<<<<<< path_test_combined,path_test_combined.items[0],path_test_combined.items[1],path_test_combined.user.details.id,path_test_combined.user.name

        Path("temp_path_config.json").unlink()


        if dummy_config_path.exists() and config_to_use == dummy_config_path: # type: ignore
            dummy_config_path.unlink() # type: ignore
            print(f"Cleaned up dummy config: {dummy_config_path}") # type: ignore

    except Exception as e:
        print(f"An error occurred during example usage: {e}")
        import traceback
        traceback.print_exc()