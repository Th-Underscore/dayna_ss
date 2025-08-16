import re
from types import FunctionType
from typing import Iterable, Any
from pathlib import Path
import json
import jsonc
import traceback
from functools import reduce

# Colour codes
_ERROR = "\033[1;31m"
_SUCCESS = "\033[1;32m"
_INPUT = "\033[0;33m"
_GRAY = "\033[0;30m"
_HILITE = "\033[0;36m"
_BILITE = "\033[1;36m"
_BOLD = "\033[1;37m"
_RESET = "\033[0m"

_DEBUG = "\033[1;31m||\033[0;32m"

# Helper types
History = list[list[str]]
Histories = dict[str, History]


def load_json(file_path: str | Path, verbose: bool = False) -> dict:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return jsonc.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        if verbose:
            print(f"Warning: Could not load {file_path}, returning empty dict")
        return {}


def save_json(data: dict, file_path: str | Path) -> bool:
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        print(f"{_ERROR}Error saving {file_path}: {str(e)}{_RESET}")
        return False


def validate_path(path: str | Path) -> Path:
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        return False
    return path


def recursive_get(data: dict | Iterable, keyList: Iterable[Any], default: Any = None):
    """Iterate through the keys of a Python object tree / map.

    Strings are accepted as indices of lists.\n

    Equivalent to:
    ```
    data[keyList[0]][keyList[1]][...][keyList[-1]]
    ```

    Example:
    ```
    my_dict = {"key1": {"key2": {"key3": "my_value"}}
    value = recursive_get(my_dict, ["key1", "key2", "key3"])
    # value is "my_value"
    ```

    Args:
        data (Iterable): The dictionary or Iterable to search.
        keyList (Iterable): The list of keys to traverse in the data.
        default (Any): The default value to return if any keys are not found.
    Returns:
        out (Any): The value found at the end of the key path, or the default value if not found.
    """
    if not keyList:
        return data
    try:
        current_level = data
        length = len(keyList)
        for i, key in enumerate(keyList):
            if isinstance(current_level, dict):
                if key not in current_level:
                    key = str(key)
            elif isinstance(current_level, Iterable):
                if isinstance(key, str) and key.isdigit():
                    key = int(key)
            if i == length - 1:
                return current_level[key]
            current_level = current_level[key]
    except (KeyError, IndexError, TypeError):
        return default
    except Exception as e:
        print(f"{_ERROR}Error getting value:{_RESET} {e}")
        traceback.print_exc()
        return None


def recursive_set(data: dict | list, keyList: Iterable, value: Any) -> None:
    """Set a value in a nested dictionary based on a list of keys.

    Strings are accepted as indices of lists.\n
    Creates nested dictionaries if they don't exist along the path.\n
    Does not accept default values with keys.

    Equivalent to:
    ```
    data[keyList[0]][keyList[1]][...][keyList[-1]] = value
    recursive_get(data, keyList[:-2])[keyList[-1] = value
    ```

    Example:
    ```
    my_dict = {}
    recursive_set(my_dict, ["key1", "key2", "key3"], "my_value")
    # my_dict is {"key1": {"key2": {"key3": "my_value"}}}
    ```

    Args:
        data (dict | list): The dictionary or list to modify.
        keyList (Iterable): The list of keys to traverse in the data.
        value (Any): The value to set at the end of the path.
    """
    current_level = data
    length = len(keyList)
    for i, key in enumerate(keyList):
        if i == length - 1:
            current_level[key] = value
        elif isinstance(current_level, dict):
            current_level = current_level.setdefault(key, {})
        elif isinstance(current_level, list):
            if key.isdigit():
                index = int(key)
                while len(current_level) <= index:
                    current_level.append(None)  # Ensure the list is long enough
                current_level = current_level[index]
            else:
                raise TypeError(f"Cannot use non-integer key '{key}' on a list.")


def split_keys_to_list(keys: str | Iterable[str]) -> list[str]:
    """Split a string of keys into a list, handling both dot separators and square brackets.

    Does not accept default values with keys.

    Args:
        keys (str | Iterable[str]): A string of keys separated by dots or square brackets, or an iterable of strings.
    Returns:
        out (list[str]): The list of keys.
    """
    if isinstance(keys, str):
        keys = re.sub(r"\[([^\]]+)\]", r".\1", keys).replace('"', "").split(".")
    return [key.strip() for key in keys if key.strip()]


from typing import Any
from extensions.dayna_ss.utils.schema_parser import ParsedSchemaClass, SchemaParser, TYPE_MAP


def expand_lists_in_data_for_llm(data: dict | list, schema_type: ParsedSchemaClass | None, parser: SchemaParser) -> Any:
    """Recursively traverse data and expand lists into dictionaries
    with stringified integer keys if the corresponding schema indicates to do so.

    This is for creating a JSON-like structure that is clearer for an LLM.
    This does not mutate the original data.

    Args:
        data (dict | list): The data to process (dict, list, or primitive).
        schema_type (ParsedSchemaClass): The type for the current data segment, if available.
        parser (SchemaParser): The SchemaParser instance for resolving type definitions.

    Returns:
        out (Any): A new data structure with specified lists expanded.
    """
    if schema_type in TYPE_MAP.values() or data is None:
        return data

    if isinstance(schema_type, ParsedSchemaClass):
        parsed_class_obj = schema_type

        if parsed_class_obj.definition_type == "dataclass" and isinstance(data, dict):
            data_copy = {}
            for k, v in data.items():
                field_schema = parsed_class_obj.get_field(k)
                if field_schema:
                    item_schema_type = field_schema.type if field_schema.type else None
                    data_copy[k] = expand_lists_in_data_for_llm(v, item_schema_type, parser)
            return data_copy
        elif parsed_class_obj.definition_type == "field":
            field_type = parsed_class_obj.get_field().type
            item_schema_type = None

            if isinstance(data, dict):
                if (
                    hasattr(field_type, "__origin__")
                    and field_type.__origin__ is dict
                    and hasattr(field_type, "__args__")
                    and field_type.__args__
                ):
                    item_schema_type = field_type.__args__[1]
                return {k: expand_lists_in_data_for_llm(v, item_schema_type, parser) for k, v in data.items()}

            if parsed_class_obj.do_expand_into_dict and isinstance(data, list):
                if (
                    hasattr(field_type, "__origin__")
                    and field_type.__origin__ is list
                    and hasattr(field_type, "__args__")
                    and field_type.__args__
                ):
                    item_schema_type = field_type.__args__[0]
                return {str(i): expand_lists_in_data_for_llm(item, item_schema_type, parser) for i, item in enumerate(data)}

            print(f"Returning '{schema_type.name}' as is: {_BOLD}{data} {_DEBUG} {field_type}{_RESET}")
            return [expand_lists_in_data_for_llm(item, field_type, parser) for item in data] if isinstance(data, list) else data

    if isinstance(data, dict):
        effective_type = schema_type.__args__[1] if hasattr(schema_type, "__args__") else schema_type
        return {k: expand_lists_in_data_for_llm(v, effective_type, parser) for k, v in data.items()}  # data_copy

    if isinstance(data, list):
        effective_type = schema_type.__args__[0] if hasattr(schema_type, "__args__") else schema_type
        return {str(i): expand_lists_in_data_for_llm(item, effective_type, parser) for i, item in enumerate(data)}  # data_copy

    return data


def expand_list(data: list) -> dict:
    return {str(i): item for i, item in enumerate(data)}


def _is_dict_expandable_to_list(data: dict) -> bool:
    """Check if a dictionary's keys are stringified sequential integers (e.g., "0", "1", "2", ...)."""
    if not data:
        return True
    int_keys = []
    for k in data.keys():
        if not isinstance(k, str) or not k.isdigit():
            return False
        int_keys.append(int(k))
    if not int_keys:
        return True
    int_keys.sort()
    return all(int_keys[i] == i for i in range(len(int_keys)))


def unexpand_lists_in_data_from_llm(data: Any, schema_type: ParsedSchemaClass | None, parser: SchemaParser) -> Any:
    """
    Recursively traverse data and convert dictionaries with stringified integer keys
    back into lists if the corresponding schema indicates to do so, or if no schema is provided
    and the dictionary keys suggest it was an expanded list.

    This is the inverse of `expand_lists_in_data_for_llm`.
    This does not mutate the original data.

    Args:
        data (Any): The data to process (dict, list, or primitive).
        schema_type (ParsedSchemaClass | None): The type for the current data segment, if available.
        parser (SchemaParser): The SchemaParser instance for resolving type definitions.

    Returns:
        out (Any): A new data structure with specified dictionaries converted to lists.
    """
    if schema_type in TYPE_MAP.values() or data is None:
        return data

    if isinstance(schema_type, ParsedSchemaClass):
        parsed_class_obj = schema_type

        if parsed_class_obj.definition_type == "dataclass" and isinstance(data, dict):
            data_copy = {}
            for k, v in data.items():
                field_schema = parsed_class_obj.get_field(k)
                current_item_schema_type = field_schema.type if field_schema and field_schema.type else None
                data_copy[k] = unexpand_lists_in_data_from_llm(v, current_item_schema_type, parser)
            return data_copy
        elif parsed_class_obj.definition_type == "field":
            field_type = parsed_class_obj.get_field().type
            if parsed_class_obj.do_expand_into_dict and isinstance(data, dict):
                if _is_dict_expandable_to_list(data):
                    list_item_schema_type = None
                    if (
                        field_type
                        and hasattr(field_type, "__origin__")
                        and field_type.__origin__ is list
                        and hasattr(field_type, "__args__")
                        and field_type.__args__
                    ):
                        list_item_schema_type = field_type.__args__[0]

                    # Ensure keys are sorted numerically for correct list order
                    sorted_items = sorted(data.items(), key=lambda x: int(x[0]))
                    return [
                        unexpand_lists_in_data_from_llm(item_data, list_item_schema_type, parser)
                        for _, item_data in sorted_items
                    ]  # data_copy

            if isinstance(data, dict):
                dict_value_schema_type = None
                if (
                    field_type
                    and hasattr(field_type, "__origin__")
                    and field_type.__origin__ is dict
                    and hasattr(field_type, "__args__")
                    and len(field_type.__args__) > 1
                ):
                    dict_value_schema_type = field_type.__args__[1]
                return {
                    k: unexpand_lists_in_data_from_llm(v, dict_value_schema_type, parser) for k, v in data.items()
                }  # data_copy

            if isinstance(data, list):
                list_item_schema_type = None
                if (
                    field_type
                    and hasattr(field_type, "__origin__")
                    and field_type.__origin__ is list
                    and hasattr(field_type, "__args__")
                    and field_type.__args__
                ):
                    list_item_schema_type = field_type.__args__[0]
                elif field_type:  # If it's a non-generic type for list items
                    list_item_schema_type = field_type

                return [unexpand_lists_in_data_from_llm(item, list_item_schema_type, parser) for item in data]  # data_copy

            return data

    if isinstance(data, dict):
        if _is_dict_expandable_to_list(data):
            sorted_items = sorted(data.items(), key=lambda x: int(x[0]))
            return [unexpand_lists_in_data_from_llm(item_data, None, parser) for _, item_data in sorted_items]
        return {k: unexpand_lists_in_data_from_llm(v, None, parser) for k, v in data.items()}

    if isinstance(data, list):
        return [unexpand_lists_in_data_from_llm(item, None, parser) for item in data]

    return data


def get_values(data: dict | list):
    """Get the values of a potentially expanded list."""
    print(type(data))
    if isinstance(data, dict):
        print("is dict")
        return data.values()
    print("is list")
    return data


def enumerate_list(data: dict | list):
    """Enumerate a potentially expanded list."""
    if isinstance(data, dict):
        return enumerate(data.values())  # data.items() for expanded lists
    return enumerate(data)


def format_str(string: str, **kwargs) -> str:
    """Format a string with keyword arguments, calling functions as needed.

    Args:
        string (str): The string to format. It should contain placeholders like {key}.

    Returns:
        str: The formatted string.

    Examples:
        >>> format_str("Hello, {name}!", name="John Doe")
        'Hello, John Doe!'
        >>> format_str("The result is {result}.", result=lambda: print(42))
        42  # Only printed if {result} is present.
        # In a real scenario, the lambda would call a function that returns a value.
        'The result is None.'
    """
    for key, value in kwargs.items():
        if f"{{{key}}}" in string:
            if isinstance(value, FunctionType):
                value = value()
            string = string.replace(f"{{{key}}}", str(value))
    return string


# --- LLM response parsing ---
patterns = (
    r'^[ \t]*(?:```(?:\S*)?|""")(.*?)(?:```|""")',  # Multi-line JSON object with code blocks
    r"^(?![\s\t])(?:.*?})({.*})",  # Single-line JSON object
    r"({.*?^})",  # Multi-line JSON object without code blocks
)
# json_regex = re.compile(patterns, flags=(re.MULTILINE + re.DOTALL))
print(f"{_HILITE}JSON regex: {_RESET}{patterns}")


def strip_json_response(output: str) -> str:
    """Remove any markdown formatting or extra text from a JSON response."""
    for pattern in patterns:
        match = re.search(pattern, output, flags=(re.MULTILINE + re.DOTALL))
        if match:
            return match.group(1)
    return output


def strip_thinking(output: str) -> str:
    """
    Exclude the "\\<think> ... \\</think>" tags and their content from a response.
    If \\<think> is present but \\</think> is not, returns an empty string.
    If \\</think> is present but \\<think> is not, returns everything after the first \\</think>.
    """
    if "<think>" in output and "</think>" not in output:
        return ""

    cleaned_output = re.sub(r"(?:<think>.*?)?<\/think>", "", output, flags=(re.MULTILINE + re.DOTALL))
    return cleaned_output.lstrip()


def strip_response(output: str) -> str:
    """Remove the response section from a response."""
    return strip_json_response(strip_thinking(output))


def extract_meaningful_paragraphs(text: str) -> str:
    """
    Extracts meaningful paragraphs from text, attempting to strip out common
    LLM-generated conversational filler, titles, and formatting.
    """
    if not text:
        return ""

    # 1. Strip leading/trailing quotes that might wrap the whole response
    processed_text = text.strip()
    if processed_text.startswith('"') and processed_text.endswith('"'):
        processed_text = processed_text[1:-1].strip()

    # 2. Remove common conversational intros
    intro_patterns = [  # TODO: Make these patterns configurable in the UI, including placeholders
        r"^(Here's|Here is) my response as \"\w+\":\s*\n*",
        r"^(Of course|Certainly|Here is the response|Here's the story|Here is your story), as requested:\s*\n*",
    ]
    for pattern in intro_patterns:
        processed_text = re.sub(pattern, "", processed_text, flags=re.IGNORECASE)

    # 3. Split into blocks and process each one
    blocks = re.split(r"\n\s*\n+", processed_text.strip())
    meaningful_blocks = []

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # 4. Strip list markers and bolded titles (e.g., "* **Tactical Movement**:")
        # This handles cases where the title is on the same line as the paragraph start.
        block = re.sub(r"^\s*[\*\-]\s*\*\*(.*?)\*\*:\s*", "", block)

        # 5. Identify and discard standalone titles/headings
        is_likely_title = False
        if block.startswith("**") and block.endswith("**"):
            content_inside = block[2:-2].strip()
            if len(content_inside.split()) < 8 or content_inside.endswith(":"):
                is_likely_title = True

        if not is_likely_title and len(block.split()) < 8 and block.endswith(":"):
            is_likely_title = True

        if not is_likely_title:
            meaningful_blocks.append(block)

    return "\n\n".join(meaningful_blocks)
