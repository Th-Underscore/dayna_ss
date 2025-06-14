import copy
import json
import re
from pathlib import Path
from typing import Any, Union, get_origin, get_args

# Colour codes
_ERROR = "\033[1;31m"
_SUCCESS = "\033[1;32m"
_INPUT = "\033[0;33m"
_GRAY = "\033[0;30m"
_HILITE = "\033[0;36m"
_BOLD = "\033[1;37m"
_RESET = "\033[0m"

_DEBUG = "\033[1;31m||\033[0;32m"

# Basic type mapping
TYPE_MAP = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "any": Any,
}


class ParsedSchemaField:
    """Represents a field parsed from the schema."""

    def __init__(self, name: str, type_hint: Any, default: Any = None):
        self.name = name
        self._type = type_hint
        self.type: ParsedSchemaClass | type | None = None  # Will be resolved after parsing
        self.default = default
        self.no_update: bool = False


class ParsedSchemaClass:
    """Represents a class structure parsed from the schema."""

    def __init__(
        self,
        name: str,
        definition_type: str,
        fields: list[ParsedSchemaField] | None = None,
        field: ParsedSchemaField | None = None,
        defaults: dict[str, Any] | None = None,
        events: dict[str, list[str]] | None = None,
    ):
        self.name = name
        self.definition_type = definition_type
        self.defaults = defaults or {}
        self.event_map: dict[str, list[str]] = events or {}

        self._fields_dict: dict[str, ParsedSchemaField] | None = None  # Parsed fields for "dataclass" type
        self._field: ParsedSchemaField | None = None  # Parsed type for "field" type

        if self.definition_type == "dataclass":
            if not fields:
                raise ValueError(f"ParsedSchemaClass '{name}' of type 'dataclass' must have fields defined.")
            self._fields_dict = {f.name: f for f in fields}
        elif self.definition_type == "field":
            if not field:
                raise ValueError(f"ParsedSchemaClass '{name}' of type 'field' must have a field defined.")
            self._field = field

        self.no_update: bool = False
        self.gate_check_prompt_template: str | None = None
        self.branch_query_prompt_template: str | None = None
        self.update_prompt_template: str | None = None
        self.branch_update_prompt_template: str | None = None
        self.new_field_query_prompt_template: str | None = None
        self.new_field_entry_prompt_template: str | None = None
        self.do_expand_into_dict: bool = True

        # Add default values to fields where applicable and parse specific flags/templates from defaults
        for field_name_or_flag, value in self.defaults.items():
            if field_name_or_flag == "no_update":
                self.no_update = bool(value)
            elif field_name_or_flag == "gate_check_prompt_template":
                self.gate_check_prompt_template = str(value)
            elif field_name_or_flag == "branch_query_prompt_template":
                self.branch_query_prompt_template = str(value)
            elif field_name_or_flag == "branch_update_prompt_template":
                self.branch_update_prompt_template = str(value)
            elif field_name_or_flag == "new_field_query_prompt_template":
                self.new_field_query_prompt_template = str(value)
            elif field_name_or_flag == "new_field_entry_prompt_template":
                self.new_field_entry_prompt_template = str(value)
            elif field_name_or_flag == "update_prompt_template":
                self.update_prompt_template = str(value)
            elif field_name_or_flag == "do_expand_into_dict":
                self.do_expand_into_dict = bool(value)
            # Field-specific defaults (like descriptions)
            elif self.definition_type == "dataclass" and fields:
                for field_obj in fields:
                    if field_obj.name == field_name_or_flag:
                        field_obj.default = value
                        break

    def get_fields(self) -> list[ParsedSchemaField]:
        """Mimics dataclasses.fields() for 'dataclass' type. Returns empty list for 'field' type."""
        if self.definition_type == "dataclass":
            return list(self._fields_dict.values())
        return [self._field]

    def get_field(self, name: str | None = None) -> ParsedSchemaField:
        """Get a specific field by name for 'dataclass' type. Returns self._field for 'field' type."""
        if self.definition_type == "dataclass":
            return self._fields_dict.get(name)
        return self._field

    def _type_to_json_schema_dict(self, type_hint: Any, all_definitions_map: dict) -> dict:
        """Converts a Python type hint (potentially a ParsedSchemaClass) to a JSON schema dict."""
        if isinstance(type_hint, ParsedSchemaClass):
            # This is a reference to another defined type
            if type_hint.name in all_definitions_map:
                return {"$ref": f"#/definitions/{type_hint.name}"}
            else:
                # Fallback if somehow not in all_definitions_map, convert it directly
                # This case should ideally not be hit if all_definitions_map is comprehensive
                return type_hint.to_json_schema_dict(all_definitions_map)
        elif type_hint is str:
            return {"type": "string"}
        elif type_hint is int:
            return {"type": "integer"}
        elif type_hint is float:
            return {"type": "number"}
        elif type_hint is bool:
            return {"type": "boolean"}
        elif hasattr(type_hint, "__origin__"):
            origin = type_hint.__origin__
            args = getattr(type_hint, "__args__", tuple())
            if origin is list and args:
                return {
                    "type": "array",
                    "items": self._type_to_json_schema_dict(args[0], all_definitions_map),
                }
            elif origin is dict and len(args) == 2:
                # JSON schema for dicts is typically an object with string keys.
                # For 'additionalProperties', we describe the type of the values.
                return {
                    "type": "object",
                    "additionalProperties": self._type_to_json_schema_dict(args[1], all_definitions_map),
                }
        elif type_hint is Any or type_hint is None:
            return {}  # Any type or can be null

        # Fallback for unknown types: treat as string or object? For now, an empty schema.
        # Consider raising an error or logging a warning for unhandled types.
        print(f"{_ERROR}Warning: Unhandled type '{type_hint}' in _type_to_json_schema_dict. Returning empty schema.{_RESET}")
        return {}

    def to_json_schema_dict(self, all_definitions_map: dict) -> dict:
        """Converts this ParsedSchemaClass instance to a JSON schema dictionary."""
        if self.definition_type == "dataclass":
            properties = {}
            required_fields = []  # Assuming all fields are required unless a default is present or explicitly marked optional

            for field_obj in self.get_fields():
                properties[field_obj.name] = self._type_to_json_schema_dict(field_obj.type, all_definitions_map)
                if field_obj.default is None:  # Basic check for required; could be more sophisticated
                    # Check if default is None because it was not set, vs. explicitly set to None
                    is_explicitly_optional = False  # Placeholder for more advanced optionality check
                    if not is_explicitly_optional:
                        required_fields.append(field_obj.name)

            schema_dict = {"type": "object", "properties": properties}
            if required_fields:
                schema_dict["required"] = required_fields
            return schema_dict

        elif self.definition_type == "field":
            # For a 'field' type, the ParsedSchemaClass itself represents the type.
            # Its 'name' is the definition name, and its '_field.type' is the actual type.
            if self._field and self._field.type:
                return self._type_to_json_schema_dict(self._field.type, all_definitions_map)
            else:
                # Should not happen if parser is correct
                print(
                    f"{_ERROR}Warning: 'field' type ParsedSchemaClass '{self.name}' has no _field.type. Returning empty schema.{_RESET}"
                )
                return {}

        print(
            f"{_ERROR}Warning: Unknown definition_type '{self.definition_type}' for '{self.name}'. Returning empty schema.{_RESET}"
        )
        return {}

    def generate_example_json(self, all_definitions_map: dict, depth: int = 0, max_depth: int = 5) -> Any:
        """Generate an example JSON-like structure based on this schema class."""
        if depth > max_depth:
            return f"<max_depth_reached_for_{self.name}>"

        if self.definition_type == "dataclass":
            example_obj = {}
            for field_obj in self.get_fields():
                if field_obj.default is not None:
                    pass
                    # example_obj[field_obj.name] = copy.copy(field_obj.default)
                else:
                    field_type = field_obj.type
                    field_name = field_obj.name
                    # Try to get description from parent class's defaults to use as example value
                    field_description_key = f"{field_name}_desc"
                    field_description = self.defaults.get(field_description_key)

                    if field_description is not None:
                        if field_type is str or field_type is int or field_type is float or field_type is bool:
                            example_obj[field_name] = str(field_description)
                        elif isinstance(field_type, ParsedSchemaClass):  # Nested class, recurse
                            example_obj[field_name] = field_type.generate_example_json(
                                all_definitions_map, depth + 1, max_depth
                            )
                        elif hasattr(field_type, "__origin__"):  # Handle generics like list/dict
                            pass
                        else:  # Fallback for other types if description not used
                            example_obj[field_name] = f"<{field_description}>"  # Use description as placeholder

                    # If description wasn't used or applicable for the primitive type, or if it's a complex type
                    if field_name not in example_obj:
                        if isinstance(field_type, ParsedSchemaClass):
                            example_obj[field_name] = field_type.generate_example_json(
                                all_definitions_map, depth + 1, max_depth
                            )
                        elif field_type is str:
                            example_obj[field_name] = f"{field_name}_example_string"
                        elif field_type is int:
                            example_obj[field_name] = 0
                        elif field_type is float:
                            example_obj[field_name] = 0.0
                        elif field_type is bool:
                            example_obj[field_name] = True

                # This part handles generics if not already handled by description logic
                if field_name not in example_obj:
                    if hasattr(field_type, "__origin__"):
                        origin = field_type.__origin__
                        args = getattr(field_type, "__args__", tuple())
                        if origin is list and args:
                            item_type = args[0]
                            if isinstance(item_type, ParsedSchemaClass):
                                example_obj[field_name] = [
                                    item_type.generate_example_json(all_definitions_map, depth + 1, max_depth)
                                ]
                            elif item_type is str:
                                example_obj[field_name] = ["string_example_in_list"]
                            elif item_type is int:
                                example_obj[field_name] = [0]
                            elif item_type is float:
                                example_obj[field_name] = [0.0]
                            elif item_type is bool:
                                example_obj[field_name] = [True]
                            elif item_type is Any:
                                example_obj[field_name] = ["any_value_in_list"]
                            else:
                                example_obj[field_name] = [f"<example_for_{str(item_type)}>"]
                        elif origin is dict and len(args) == 2:
                            # Determine the example key using the field's description if available
                            field_description_for_key_lookup = f"{field_name}_desc"
                            field_description_as_key = self.defaults.get(field_description_for_key_lookup)
                            key_example = (
                                str(field_description_as_key)
                                if field_description_as_key is not None
                                else f"{field_name}_key_example"
                            )

                            value_type = args[1]
                            if isinstance(value_type, ParsedSchemaClass):
                                example_obj[field_name] = {
                                    key_example: value_type.generate_example_json(all_definitions_map, depth + 1, max_depth)
                                }
                            elif value_type is str:
                                example_obj[field_name] = {key_example: "value_string_example"}
                            elif value_type is int:
                                example_obj[field_name] = {key_example: 0}
                            elif value_type is float:
                                example_obj[field_name] = {key_example: 0.0}
                            elif value_type is bool:
                                example_obj[field_name] = {key_example: True}
                            elif value_type is Any:
                                example_obj[field_name] = {key_example: "any_value_in_dict"}
                            else:
                                example_obj[field_name] = {key_example: f"<example_for_{str(value_type)}>"}
                        else:
                            example_obj[field_name] = f"<unhandled_generic_type_{str(field_type)}>"
                    elif field_type is Any:
                        example_obj[field_name] = "any_value_example"
                    elif field_type is None:
                        example_obj[field_name] = None
                    else:
                        example_obj[field_name] = f"<unknown_type_{str(field_type)}>"
            return example_obj

        elif self.definition_type == "field":
            if self._field:
                field_obj = self._field
                field_type = field_obj.type
                field_description = self.defaults.get("field_desc")

                if field_description is not None:
                    if field_type is str or field_type is int or field_type is float or field_type is bool:
                        return str(field_description)
                    elif isinstance(field_type, ParsedSchemaClass):  # Nested class, recurse
                        return field_type.generate_example_json(all_definitions_map, depth + 1, max_depth)
                    elif hasattr(field_type, "__origin__"):  # Handle generics like list/dict
                        pass
                    else:  # Fallback for other types if description not used
                        return f"<{field_description}>"
                if field_obj.default is not None:
                    return json.loads(json.dumps(field_obj.default))
                if isinstance(field_type, ParsedSchemaClass):
                    return field_type.generate_example_json(all_definitions_map, depth + 1, max_depth)
                elif field_type is str:
                    return "string_example"
                elif field_type is int:
                    return 1
                elif field_type is float:
                    return 1.0
                elif field_type is bool:
                    return False
                elif hasattr(field_type, "__origin__"):
                    origin = field_type.__origin__
                    args = getattr(field_type, "__args__", tuple())
                    if origin is list and args:
                        item_type = args[0]
                        if isinstance(item_type, ParsedSchemaClass):
                            return [item_type.generate_example_json(all_definitions_map, depth + 1, max_depth)]
                        elif item_type is str:
                            return ["string_example_in_list"]
                        elif item_type is int:
                            return [1]
                        elif item_type is float:
                            return [1.0]
                        elif item_type is bool:
                            return [False]
                        elif item_type is Any:
                            return ["any_value_in_list"]
                        else:
                            return [f"<example_for_{str(item_type)}_in_list>"]
                    elif origin is dict and len(args) == 2:
                        key_example = "key_example"
                        value_type = args[1]
                        if isinstance(value_type, ParsedSchemaClass):
                            return {key_example: value_type.generate_example_json(all_definitions_map, depth + 1, max_depth)}
                        elif value_type is str:
                            return {key_example: "value_string_example"}
                        elif value_type is int:
                            return {key_example: 1}
                        elif value_type is float:
                            return {key_example: 1.0}
                        elif value_type is bool:
                            return {key_example: False}
                        elif value_type is Any:
                            return {key_example: "any_value_in_dict"}
                        else:
                            return {key_example: f"<example_for_{str(value_type)}_in_dict>"}
                    else:
                        return f"<unhandled_generic_type_{str(field_type)}>"
                elif field_type is Any:
                    return "any_value_example"
                else:
                    return f"<unknown_type_{str(field_type)}>"
            return f"<empty_definition_{self.name}>"

        return f"<unknown_definition_type_{self.definition_type}_for_{self.name}>"

    def __repr__(self):
        if self.definition_type == "field":
            return f"ParsedSchemaClass(name='{self.name}', type='field', field_type='{self._field}')"
        return f"ParsedSchemaClass(name='{self.name}', type='dataclass')"


class SchemaParser:
    """Loads and parses the subjects JSON schema."""

    def __init__(self, schema_path: Union[str, Path]):
        self.schema_path = Path(schema_path)
        self.schema = self._load_schema()
        self.definitions: dict[str, ParsedSchemaClass] = {}
        self.subjects: dict[str, ParsedSchemaClass | type] = {}
        self._parse_definitions()
        self._parse_subjects()

    def _load_schema(self) -> dict:
        """Load the JSON schema file."""
        try:
            with open(self.schema_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Schema file not found: {self.schema_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON schema file: {e}")

    def _parse_type_string(self, type_str: str):
        """Parse a type string (e.g., 'list[string]', 'dict[string, CharacterStatus]')"""
        type_str = type_str.strip()

        # Basic types
        if type_str in TYPE_MAP:
            return TYPE_MAP[type_str]

        # List types
        list_match = re.match(r"list\[(.+)\]", type_str)
        if list_match:
            inner_type_str = list_match.group(1).strip()
            inner_type = self._parse_type_string(inner_type_str)
            return list[inner_type]

        # dict types
        dict_match = re.match(r"dict\[(.+),\s*(.+)\]", type_str)
        if dict_match:
            key_type_str = dict_match.group(1).strip()
            value_type_str = dict_match.group(2).strip()
            key_type = self._parse_type_string(key_type_str)
            value_type = self._parse_type_string(value_type_str)
            return dict[key_type, value_type]

        # Custom defined types (references to other definitions)
        if type_str in self.definitions:
            return self.definitions[type_str]  # ParsedSchemaClass

        # If it's a definition we haven't parsed yet, return the name as a string placeholder
        if type_str in self.schema.get("definitions", {}):
            return type_str  # Placeholder, will be resolved later

        raise ValueError(f"Unknown type string: {type_str}")

    def _resolve_type_placeholders(self, type_hint: ParsedSchemaClass | type):
        """Recursively resolve string placeholders in type hints."""
        if isinstance(type_hint, str) and type_hint in self.definitions:
            return self.definitions[type_hint]
        elif hasattr(type_hint, "__origin__") and hasattr(type_hint, "__args__"):
            # Handle generics like List, dict
            origin = type_hint.__origin__
            args = tuple(self._resolve_type_placeholders(arg) for arg in type_hint.__args__)
            if origin is list:
                return list[args[0]]
            if origin is dict:
                return dict[args[0], args[1]]
        return type_hint

    def _parse_definitions(self):
        """Parse the 'definitions' section of the schema."""
        schema_definitions = self.schema.get("definitions", {})

        # First pass: Create ParsedSchemaClass objects with string placeholders for types
        for name, definition in schema_definitions.items():
            def_type = definition.get("type")
            defaults = definition.get("defaults")
            events_data = definition.get("events")

            if def_type == "dataclass":
                fields_data = definition.get("fields", {})
                if not fields_data:
                    pass  # Allow empty dataclasses

                fields = []
                for field_name, type_str in fields_data.items():
                    fields.append(ParsedSchemaField(field_name, type_str))  # type_str is placeholder
                self.definitions[name] = ParsedSchemaClass(
                    name, definition_type="dataclass", fields=fields, defaults=defaults, events=events_data
                )
            elif def_type == "field":
                field_type_str = definition.get("field")
                if not field_type_str:
                    raise ValueError(f"Definition '{name}' of type 'field' is missing the 'field' attribute.")
                self.definitions[name] = ParsedSchemaClass(
                    name,
                    definition_type="field",
                    field=ParsedSchemaField(name, field_type_str),
                    defaults=defaults,
                    events=events_data,
                )
            # Add handling for other types if needed (e.g., enums)

        # Second pass: Resolve type placeholders
        for name, parsed_class_obj in self.definitions.items():
            if parsed_class_obj.definition_type == "dataclass":
                for field_obj in parsed_class_obj.get_fields():
                    if isinstance(field_obj._type, str):
                        field_obj.type = self._parse_type_string(field_obj._type)
                        field_obj.type = self._resolve_type_placeholders(field_obj.type)
                    # Propagate no_update from type to field
                    if isinstance(field_obj.type, ParsedSchemaClass):
                        field_obj.no_update = field_obj.type.no_update
            elif parsed_class_obj.definition_type == "field":
                field_obj = parsed_class_obj._field
                if isinstance(field_obj._type, str):
                    field_obj.type = self._parse_type_string(field_obj._type)
                    field_obj.type = self._resolve_type_placeholders(field_obj.type)
                if isinstance(field_obj.type, ParsedSchemaClass):
                    field_obj.no_update = field_obj.type.no_update

    def get_definitions_as_json_schema(self) -> dict:
        """Convert all parsed definitions into a JSON schema 'definitions' block."""
        json_schema_definitions = {}
        for name, parsed_class_obj in self.definitions.items():
            json_schema_definitions[name] = parsed_class_obj.to_json_schema_dict(self.definitions)
        return json_schema_definitions

    def _collect_refs_from_schema_part(
        self, schema_part: Any, queue: list[str], visited: set[str], all_definition_keys: set[str]
    ):
        """
        Recursively scans a part of a JSON schema, identifies $ref links,
        and adds the referenced definition names to the queue if they are valid and not yet visited.
        """
        if isinstance(schema_part, dict):
            for key, value in schema_part.items():
                if key == "$ref" and isinstance(value, str):
                    # Extract name from "#/definitions/Name"
                    try:
                        ref_name = value.split("/")[-1]
                        if ref_name in all_definition_keys and ref_name not in visited and ref_name not in queue:
                            queue.append(ref_name)
                    except IndexError:
                        # Handle potential malformed $ref, though unlikely with internal generation
                        print(f"{_ERROR}Warning: Malformed $ref encountered: {value}{_RESET}")
                else:
                    self._collect_refs_from_schema_part(value, queue, visited, all_definition_keys)
        elif isinstance(schema_part, list):
            for item in schema_part:
                self._collect_refs_from_schema_part(item, queue, visited, all_definition_keys)

    def get_relevant_definitions_json(self, root_definition_name: str) -> dict:
        """
        Collects the JSON schema for the given root_definition_name and all
        recursively referenced definitions.
        Returns a dictionary structured for JSON schema "definitions" section.
        """
        relevant_schemas: dict[str, dict] = {}
        queue: list[str] = []
        visited: set[str] = set()
        all_definition_keys = set(self.definitions.keys())

        if root_definition_name in all_definition_keys:
            queue.append(root_definition_name)
        else:
            print(f"{_ERROR}Error: Root definition name '{root_definition_name}' not found in schema definitions.{_RESET}")
            return {"definitions": {}}

        while queue:
            current_name = queue.pop(0)
            if current_name in visited:
                continue
            visited.add(current_name)

            parsed_class_obj = self.definitions.get(current_name)
            if not parsed_class_obj:
                print(
                    f"{_ERROR}Warning: Definition '{current_name}' not found during recursive collection, though it was queued.{_RESET}"
                )
                continue

            # Generate the JSON schema for the current definition
            current_schema_json = parsed_class_obj.to_json_schema_dict(self.definitions)
            relevant_schemas[current_name] = current_schema_json

            # Scan current_schema_json for new $refs to add to the queue
            self._collect_refs_from_schema_part(current_schema_json, queue, visited, all_definition_keys)

        return {"definitions": relevant_schemas}

    def _parse_subjects(self):
        """Parse the 'subjects' section, resolving types."""
        schema_subjects = self.schema.get("subjects", {})
        for name, type_str in schema_subjects.items():
            parsed_type = self._parse_type_string(type_str)
            resolved_type = self._resolve_type_placeholders(parsed_type)
            self.subjects[name] = resolved_type

    def get_subject_class(self, subject_name: str) -> ParsedSchemaClass | None:
        """Get the parsed class definition for a top-level subject."""
        subject_type = self.subjects.get(subject_name)
        if isinstance(subject_type, ParsedSchemaClass):
            return subject_type
        if hasattr(subject_type, "__origin__"):
            args = getattr(subject_type, "__args__", tuple())
            if subject_type.__origin__ is dict and isinstance(args[1], ParsedSchemaClass):
                return args[1]
            elif subject_type.__origin__ is list and isinstance(args[0], ParsedSchemaClass):
                return args[0]
        return None

    def validate_data(self, data_instance: Any, schema_name: str, path: str = "") -> list[str]:
        """
        Recursively validates the data_instance against the schema definition.
        Returns a list of human-readable error messages.
        """
        errors: list[str] = []
        schema_definition = self.definitions.get(schema_name)

        if not schema_definition:
            errors.append(f"Schema definition for '{schema_name}' not found at path '{path}'.")
            return errors

        current_path = path if path else schema_name

        if schema_definition.definition_type == "dataclass":
            if not isinstance(data_instance, dict):
                errors.append(f"Invalid type at '{current_path}': Expected a dictionary, got {type(data_instance).__name__}.")
                return errors

            # Check for required fields
            for field_def in schema_definition.get_fields():
                if field_def.default is None and field_def.name not in data_instance:
                    errors.append(f"Missing required field '{field_def.name}' at '{current_path}'.")

            # Check types of existing fields
            for key, value in data_instance.items():
                field_def = schema_definition.get_field(key)
                field_path = f"{current_path}.{key}"
                if not field_def:
                    errors.append(f"Unexpected field '{key}' found at '{current_path}'.")
                    continue

                expected_type = field_def.type
                errors.extend(self._validate_value_type(value, expected_type, field_path))

        elif schema_definition.definition_type == "field":
            expected_type = schema_definition.get_field().type
            errors.extend(self._validate_value_type(data_instance, expected_type, current_path))

        else:
            errors.append(
                f"Unknown schema definition type '{schema_definition.definition_type}' for '{schema_name}' at '{current_path}'."
            )

        return errors

    def _validate_value_type(self, value: Any, expected_type: Any, current_path: str) -> list[str]:
        """Helper function to validate the type of a single value."""
        errors: list[str] = []

        if expected_type is Any:
            return errors

        origin_type = get_origin(expected_type)
        args_type = get_args(expected_type)

        if isinstance(expected_type, ParsedSchemaClass):
            # Recursively validate against the nested schema definition
            errors.extend(self.validate_data(value, expected_type.name, path=current_path))
        elif origin_type is list:
            if not isinstance(value, list):
                errors.append(f"Invalid type at '{current_path}': Expected a list, got {type(value).__name__}.")
                return errors
            if args_type:
                item_type = args_type[0]
                for i, item in enumerate(value):
                    item_path = f"{current_path}[{i}]"
                    errors.extend(self._validate_value_type(item, item_type, item_path))
        elif origin_type is dict:
            if not isinstance(value, dict):
                errors.append(f"Invalid type at '{current_path}': Expected a dictionary, got {type(value).__name__}.")
                return errors
            if args_type and len(args_type) == 2:
                key_type, val_type = args_type
                for k, v in value.items():
                    # Validate key type (usually str for JSON, but schema might specify)
                    if not isinstance(k, key_type):
                        errors.append(
                            f"Invalid key type at '{current_path}': Expected {key_type.__name__} for key, got {type(k).__name__} ('{k}')."
                        )
                    val_path = f"{current_path}['{k}']"
                    errors.extend(self._validate_value_type(v, val_type, val_path))
        elif expected_type in TYPE_MAP.values():
            if not isinstance(value, expected_type):
                if isinstance(value, str) and (expected_type is bool or expected_type is float or expected_type is int):
                    errors.append(
                        f"Invalid type at '{current_path}': Expected {expected_type.__name__}, got {type(value).__name__} ('{value}')."
                    )
        else:
            if not isinstance(value, expected_type):
                errors.append(
                    f"Type mismatch at '{current_path}': Expected type compatible with '{expected_type}', got {type(value).__name__} ('{value}')."
                )

        return errors


# Example usage (for testing)
if __name__ == "__main__":
    try:
        schema_file_path = Path(__file__).parent / "subjects_schema.json"
        if not schema_file_path.exists():
            schema_file_path = Path("extensions/dayna_ss/utils/subjects_schema.json")

        parser = SchemaParser(schema_file_path)
        print("Schema loaded and parsed successfully.")
        print("\nDefinitions:")
        for name, definition_obj in parser.definitions.items():
            print(f"- {name} (type: {definition_obj.definition_type}):")
            if definition_obj.definition_type == "dataclass":
                for field in definition_obj.get_fields():
                    print(f"  - {field.name}: {field.type} (default: {field.default})")
            elif definition_obj.definition_type == "field":
                print(f"  - Field Type: {definition_obj._field}")
            if definition_obj.defaults:
                print(f"  - Defaults: {definition_obj.defaults}")

        print("\nSubjects:")
        for name, subject_type in parser.subjects.items():
            print(f"- {name}: {subject_type}")

        # Test getting subject class
        print("\n--- Testing get_subject_class ---")
        characters_subject_def = parser.get_subject_class("characters")
        print(f"\nSubject Definition for 'characters': {characters_subject_def}")
        if characters_subject_def:
            print(f"  Name: {characters_subject_def.name}")
            print(f"  Definition Type: {characters_subject_def.definition_type}")
            if characters_subject_def.definition_type == "field":
                print(f"  Parsed Field Type: {characters_subject_def._field}")
            elif characters_subject_def.definition_type == "dataclass":
                print(f"  Fields: {[f.name for f in characters_subject_def.get_fields()]}")

        groups_subject_def = parser.get_subject_class("groups")
        print(f"\nSubject Definition for 'groups': {groups_subject_def}")
        if groups_subject_def:
            print(f"  Name: {groups_subject_def.name}")
            print(f"  Definition Type: {groups_subject_def.definition_type}")
            if groups_subject_def.definition_type == "field":
                print(f"  Parsed Field Type: {groups_subject_def._field}")

        current_scene_subject_def = parser.get_subject_class("current_scene")
        print(f"\nSubject Definition for 'current_scene': {current_scene_subject_def}")
        if current_scene_subject_def:
            print(f"  Name: {current_scene_subject_def.name}")
            print(f"  Definition Type: {current_scene_subject_def.definition_type}")
            if current_scene_subject_def.definition_type == "dataclass":
                print(f"  Fields: {[f.name for f in current_scene_subject_def.get_fields()]}")
                # Example of accessing a field within a dataclass subject
                what_field = current_scene_subject_def.get_field("what")
                if what_field:
                    print(f"    'what' field type: {what_field.type}")

        events_subject_def = parser.get_subject_class("events")
        print(f"\nSubject Definition for 'events': {events_subject_def}")
        if events_subject_def:  # 'Events' is a dataclass
            print(f"  Name: {events_subject_def.name}")
            print(f"  Definition Type: {events_subject_def.definition_type}")
            if events_subject_def.definition_type == "dataclass":
                print(f"  Fields: {[f.name for f in events_subject_def.get_fields()]}")
                past_field = events_subject_def.get_field("past")
                if past_field:
                    print(
                        f"    'past' field type: {past_field.type}"
                    )  # Should be dict[str, ParsedSchemaClass(name='StoryEvent')]
                    # Accessing the type of StoryEvent
                    if hasattr(past_field.type, "__args__"):
                        story_event_class = past_field.type.__args__[1]
                        if isinstance(story_event_class, ParsedSchemaClass):
                            print(f"      StoryEvent fields: {[f.name for f in story_event_class.get_fields()]}")

    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
