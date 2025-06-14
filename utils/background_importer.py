import threading
import importlib
import time
from typing import Any

# Stores imported attributes, e.g., {"torch.no_grad": <function no_grad>}
_imported_attributes = {}
_import_locks = {}  # module_name.attribute_name -> Lock
_import_done_events = {}  # module_name.attribute_name -> Event
_import_exceptions = {}  # module_name.attribute_name -> Exception
_import_started_flags = {}  # module_name.attribute_name -> bool

_DEBUG = "\033[94m"
_ERROR = "\033[91m"
_SUCCESS = "\033[92m"
_RESET = "\033[0m"


def _get_key(module_name: str, attribute_name: str | None) -> str:
    return f"{module_name}{'.' + attribute_name if attribute_name else ''}"


def _perform_import(module_name: str, attribute_name: str | None = None):
    """
    Perform the actual import of a module or a specific attribute from a module.
    This function is intended to be run in a background thread.
    """
    key = _get_key(module_name, attribute_name)
    global _imported_attributes, _import_exceptions

    try:
        # print(f"[Importer] {_DEBUG}Background import of '{key}' started...{_RESET}")
        start_time = time.time()
        imported_module = importlib.import_module(module_name)
        if attribute_name:
            attribute_value = getattr(imported_module, attribute_name)
            _imported_attributes[key] = attribute_value
        else:
            _imported_attributes[key] = imported_module  # Store the whole module
        print(f"[Importer] {_SUCCESS}'{key}' imported successfully in {time.time() - start_time:.2f} seconds.{_RESET}")
    except Exception as e:
        _import_exceptions[key] = e
        print(f"[Importer] {_ERROR}Failed to import '{key}': {e}{_RESET}")
    finally:
        if key in _import_done_events:
            _import_done_events[key].set()


def start_background_import(module_name: str, attribute_name: str | None = None):
    """
    Start the background thread to import a module/attribute if it hasn't started already.
    This function is safe to call multiple times for the same module/attribute.
    """
    # print(f"[Importer] {_DEBUG}Starting background import for '{module_name}'{_RESET}")
    key = _get_key(module_name, attribute_name)
    global _import_started_flags, _import_locks, _import_done_events

    if not _import_started_flags.get(key, False):
        # Initialize lock and event for this specific import if not present
        _import_locks.setdefault(key, threading.Lock())
        _import_done_events.setdefault(key, threading.Event())

        with _import_locks[key]:
            if not _import_started_flags.get(key, False):  # Double-check inside lock
                # print(f"[Importer] {_DEBUG}Initiating background import thread for '{key}'.{_RESET}")
                thread = threading.Thread(target=_perform_import, args=(module_name, attribute_name), daemon=True)
                thread.start()
                _import_started_flags[key] = True


def get_imported_attribute(module_name: str, attribute_name: str | None = None, timeout: float | None = None) -> Any:
    """Retrieve the imported module or attribute.
    Blocks until the background import is complete or timeout is reached.

    Args:
        module_name (str): The name of the module to import (e.g., 'torch').
        attribute_name (str, optional): The name of the attribute to get from the module (e.g., 'no_grad').
                                     If None, the whole module is returned.
        timeout (float, optional): Optional timeout in seconds to wait for the import.
                                If None, waits indefinitely.

    Returns:
        any: The imported attribute or module.

    Raises:
        RuntimeError: If the import failed.
        ImportError: If the attribute/module is not available after import.
        TimeoutError: If the import doesn't complete within the specified timeout.
    """
    key = _get_key(module_name, attribute_name)
    global _imported_attributes, _import_exceptions, _import_started_flags

    if not _import_started_flags.get(key, False):
        # print(
        #     f"[Importer] {_DEBUG}Background import for '{key}' not explicitly started. Triggering now.{_RESET}"
        # )
        start_background_import(module_name, attribute_name)

    done_event: threading.Event = _import_done_events.get(key)
    if not done_event:
        raise RuntimeError(f"[Importer] Import event for '{key}' not found. This is unexpected.")

    if not done_event.wait(timeout=timeout):
        raise TimeoutError(f"[Importer] Timed out waiting for '{key}' import after {timeout} seconds.")

    if key in _import_exceptions:
        raise RuntimeError(f"[Importer] Failed to import '{key}' in background thread.") from _import_exceptions[key]

    if key not in _imported_attributes:
        raise ImportError(f"[Importer] '{key}' not available after background import, and no exception was recorded.")

    return _imported_attributes[key]
