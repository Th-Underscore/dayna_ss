"""
Integration module for DSS tools with TGWUI's native tool calling system.

This module patches TGWUI's tool_use module to add DSS tool executors,
allowing DSS tools to be used alongside native TGWUI tools.
"""
from typing import Callable
import json

from modules import tool_use

_original_execute_tool = None
_dss_tool_executors: dict[str, Callable] = {}
_dss_tool_definitions: list[dict] = []
_dss_enabled_check: Callable | None = None


def set_dss_enabled_check(check_fn: Callable) -> None:
    """Set a function to check if DSS is currently enabled.

    The function should return True if DSS tools should be available.
    """
    global _dss_enabled_check
    _dss_enabled_check = check_fn


def register_dss_tool_executors(executors: dict[str, Callable]) -> None:
    """Register DSS tool executors with the TGWUI tool system."""
    global _dss_tool_executors, _dss_tool_definitions
    _dss_tool_executors.update(executors)

    for name in executors:
        if not _original_execute_tool:
            _patch_tool_execution()


def _patch_tool_execution() -> None:
    """Patch TGWUI's execute_tool to handle DSS tools."""
    global _original_execute_tool

    if _original_execute_tool is not None:
        return

    _original_execute_tool = tool_use.execute_tool

    def patched_execute_tool(
        tool_name: str,
        tool_input: dict | str,
        tool_handlers: dict | None = None,
    ) -> str:
        if tool_name in _dss_tool_executors:
            if isinstance(tool_input, str):
                try:
                    tool_input = json.loads(tool_input)
                except json.JSONDecodeError:
                    tool_input = {}

            result = _dss_tool_executors[tool_name](tool_input)
            return result

        return _original_execute_tool(tool_name, tool_input, tool_handlers)

    tool_use.execute_tool = patched_execute_tool


def add_dss_tools_to_state(state: dict, tool_definitions: list[dict]) -> bool:
    """Add DSS tool definitions to state for TGWUI to use.

    Only adds tools if DSS is enabled (checked via the registered enabled check function).

    Args:
        state: The TGWUI state dict to add tools to
        tool_definitions: List of OpenAI-format tool definitions

    Returns:
        True if tools were added, False otherwise
    """
    if _dss_enabled_check and not _dss_enabled_check():
        return False

    if 'tools' not in state:
        state['tools'] = []

    existing_names = {
        t.get('function', {}).get('name')
        for t in state['tools']
        if 'function' in t
    }

    added = False
    for tool_def in tool_definitions:
        print("tool:", json.dumps(tool_def))
        tool_name = tool_def.get('function', {}).get('name', '')
        if tool_name and tool_name not in existing_names:
            state['tools'].append(tool_def)
            added = True

    print("Added DSS tools to state:", added)

    return added
