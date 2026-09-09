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
                if len(current_level) <= key:
                    while len(current_level) <= key:
                        current_level.append(None)  # Ensure the list is long enough
                    is_value_exists = False
            else:
                raise TypeError(f"Cannot use non-integer key '{key}' on a list.")

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
