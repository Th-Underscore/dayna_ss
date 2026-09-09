"""Themed helper modules.

`helpers.py` was split into cohesive modules (2026-08-23):
  * console         - colour codes, History/Histories, TypedKey
  * json_utils      - load_json / save_json / validate_path
  * path_utils      - recursive_get / recursive_set / split_keys_to_list
  * data_expansion  - LLM list expansion/coercion round-trip
  * formatting      - format_str + Jinja rendering (shared cached env)
  * text_parsing    - LLM response stripping / paragraph extraction

This module remains as a re-export shim so every historical import site keeps working.
NOTE: keep this shim free of module-level `schema_parser` imports — schema_parser
imports helpers at load time, and the expansion helpers' schema imports must stay
function-local to avoid a circular import.
"""

from .console import (
    _BILITE,
    _BOLD,
    _DEBUG,
    _ERROR,
    _GRAY,
    _HILITE,
    _INPUT,
    _RESET,
    _SUCCESS,
    _WARNING,
    Histories,
    History,
    TypedKey,
)
from .json_utils import load_json, save_json, validate_path
from .path_utils import recursive_get, recursive_set, split_keys_to_list
from .data_expansion import (
    _COLLECTION_KEY_CANDIDATES,
    _infer_collection_key,
    _is_dict_expandable_to_list,
    coerce_container_types,
    enumerate_list,
    expand_list,
    expand_lists_in_data_for_llm,
    get_values,
    unexpand_lists_in_data_from_llm,
)
from .formatting import (
    _bracket_dots,
    _get_jinja_env,
    _get_param,
    _last_item,
    _scene_messages,
    format_str,
    format_str_or_jinja,
    render_jinja_template,
)
from .text_parsing import (
    extract_meaningful_paragraphs,
    patterns,
    strip_json_response,
    strip_response,
    strip_thinking,
)

__all__ = [
    # console
    "_BILITE", "_BOLD", "_DEBUG", "_ERROR", "_GRAY", "_HILITE", "_INPUT", "_RESET",
    "_SUCCESS", "_WARNING", "Histories", "History", "TypedKey",
    # json_utils
    "load_json", "save_json", "validate_path",
    # path_utils
    "recursive_get", "recursive_set", "split_keys_to_list",
    # data_expansion
    "_COLLECTION_KEY_CANDIDATES", "_infer_collection_key", "_is_dict_expandable_to_list",
    "coerce_container_types", "enumerate_list", "expand_list",
    "expand_lists_in_data_for_llm", "get_values", "unexpand_lists_in_data_from_llm",
    # formatting
    "_bracket_dots", "_get_jinja_env", "_get_param", "_last_item", "_scene_messages",
    "format_str", "format_str_or_jinja", "render_jinja_template",
    # text_parsing
    "extract_meaningful_paragraphs", "patterns", "strip_json_response",
    "strip_response", "strip_thinking",
]
