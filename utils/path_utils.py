import traceback
from typing import Any, Iterable

from .console import _ERROR, _RESET, TypedKey


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
        if isinstance(key, TypedKey):  # Handle typed keys
            key = key.key

        if isinstance(current_level, dict):
            is_value_exists = key in current_level
        if isinstance(current_level, list):
            if isinstance(key, str) and key.isdigit():
                key = int(key)
            elif not isinstance(key, int):
                raise TypeError(f"Cannot use non-integer key '{key}' on a list.")
            
            if len(current_level) <= key:
                while len(current_level) <= key:
                    current_level.append(None)  # Ensure the list is long enough
                is_value_exists = False

        if i == length - 1:  # Set the final value
            current_level[key] = value
        else:  # Create nested data structures
            if not is_value_exists:
                next_key = keyList[i + 1]
                if isinstance(next_key, TypedKey):
                    current_level[key] = next_key.type()
                elif isinstance(next_key, str) and next_key.isdigit():
                    current_level[key] = []
                else:
                    current_level[key] = {}
            current_level = current_level[key]
    return data


def strip_underscore_keys(data: Any) -> Any:
    """Recursively remove all underscore-prefixed keys from dicts.

    Used to hide internal fields (_message_node, _history, etc.) from LLM prompts
    and to sanitize LLM output before applying updates.
    """
    if isinstance(data, dict):
        return {k: strip_underscore_keys(v) for k, v in data.items() if not (isinstance(k, str) and k.startswith("_"))}
    elif isinstance(data, list):
        return [strip_underscore_keys(item) for item in data]
    return data


def merge_underscore_preserve(new_data: Any, old_data: Any) -> Any:
    """Strip underscore-prefixed keys from new_data, then restore any that exist in old_data.

    Preserves internal fields (_message_node, _history, etc.) across full-rewrite updates
    so the LLM cannot corrupt them.
    """
    if isinstance(new_data, dict):
        old_dict = old_data if isinstance(old_data, dict) else {}
        result = {}
        for k, v in new_data.items():
            if isinstance(k, str) and k.startswith("_"):
                continue  # discard LLM-written internal fields
            result[k] = merge_underscore_preserve(v, old_dict.get(k))
        # Restore internal fields from old data
        for k, v in old_dict.items():
            if isinstance(k, str) and k.startswith("_"):
                result[k] = v
        return result
    elif isinstance(new_data, list):
        old_list = old_data if isinstance(old_data, list) else []
        return [merge_underscore_preserve(n, o) if i < len(old_list) else strip_underscore_keys(n)
                for i, (n, o) in enumerate(zip(new_data, old_list))]
    return new_data


def stamp_end_message_nodes(data: Any, message_node: str) -> None:
    """Recursively stamp _message_node on all 'end' TimePoint fields.

    Convention: any dict-valued field named 'end' gets its _message_node
    set to the given value. This is the type+name convention — name 'end'
    + dict shape (TimePoint) — with no per-subject hardcoding.
    Mutates data in place.
    """
    if isinstance(data, dict):
        for k, v in data.items():
            if k == "end" and isinstance(v, dict):
                if "_message_node" not in v:
                    v["_message_node"] = message_node
            elif isinstance(v, (dict, list)):
                stamp_end_message_nodes(v, message_node)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                stamp_end_message_nodes(item, message_node)


def split_keys_to_list(keys: str | Iterable[str]) -> list[str]:
    """Split a string of keys into a list, handling both dot separators and square brackets.

    Square bracket notation (e.g., "[Mrs. Patterson]") treats the bracketed content
    as a single key regardless of dots inside. This is useful for key names that
    contain literal dots.

    Does not accept default values with keys.

    Args:
        keys (str | Iterable[str]): A string of keys separated by dots or square brackets, or an iterable of strings.
    Returns:
        out (list[str]): The list of keys.
    """
    if isinstance(keys, str):
        keys = keys.replace('"', "").replace("'", "")
        result = []
        i = 0
        while i < len(keys):
            if keys[i] == '[':
                end = keys.find(']', i)
                if end != -1:
                    part = keys[i+1:end].strip()
                    if part:
                        result.append(part)
                    i = end + 1
                    if i < len(keys) and keys[i] == '.':
                        i += 1
                else:
                    result.append(keys[i])
                    i += 1
            elif keys[i] == '.':
                i += 1
            else:
                j = i
                while j < len(keys) and keys[j] != '.' and keys[j] != '[':
                    j += 1
                part = keys[i:j].strip()
                if part:
                    result.append(part)
                i = j
        return result
    return [key.strip() for key in keys if key.strip()]
