"""Schema traversal and field initialization (T4 extraction). _update_recursive, _traverse_structure, _process_field, _initialize_field.
"""
from __future__ import annotations

import copy
import json
import jsonc
import re
import time
import traceback
from typing import Any

from ...utils.helpers import (
    _ERROR,
    _SUCCESS,
    _INPUT,
    _GRAY,
    _HILITE,
    _BOLD,
    _RESET,
    _DEBUG,
    _WARNING,
    load_json,
    save_json,
    split_keys_to_list,
    recursive_set,
    recursive_get,
    expand_lists_in_data_for_llm,
    unexpand_lists_in_data_from_llm,
    coerce_container_types,
    strip_thinking,
    strip_response,
    format_str_or_jinja,
)

from ...runtime import runtime

from ...utils.schema_parser import (
    SchemaParser,
    ParsedSchemaClass,
    ParsedSchemaField,
    Action,
    Trigger,
)

from ..formatted_data import FormattedData

from .parsing import (
    _entries_as_list,
    _safe_int,
    _normalize_entry_name,
    _collect_entry_aliases,
    _resolve_dict_key,
    _resolve_path_keys,
    _is_negative_verdict,
    _tolerant_json_loads,
    _is_schema_echo,
    _filter_new_entry_names,
    _extract_entry_names,

    defaults_to_inherit,
)
if False:  # forward refs only
    from .core import DataSummarizer


class DSTraversalMixin:
    def _update_recursive(
        self,
        item_name_prefix: str,
        data: dict,
        formatted_data: FormattedData,
        unexpanded_formatted_data: FormattedData,
        target_schema_class: ParsedSchemaClass,
        keys: list = [],
        trigger_update: bool = False,
    ) -> dict:
        """
        Entry point for processing a specific Schema Class node against a data object.
        Handles Triggers (Add New, Gate Check, Branch Query) before drilling down.

        Args:
            item_name_prefix: Path prefix for the item.
            data: Data to update.
            formatted_data: Formatted data for prompts.
            unexpanded_formatted_data: Unexpanded formatted data.
            target_schema_class: Schema class to process.
            keys: Path keys to the data location.
            trigger_update: If True, allow LLM calls for leaf fields. Defaults to False (opt-in).
        """
        if runtime.stop_everything:
            return data

        if not isinstance(target_schema_class, ParsedSchemaClass):
            print(f"{_ERROR}Target schema is not a ParsedSchemaClass: {type(target_schema_class)}{_RESET}")
            return data

        current_event_triggers = self._get_current_event_triggers(target_schema_class)
        branch_name = item_name_prefix or target_schema_class.name

        print(f"{_BOLD}Processing Schema Node: {target_schema_class.name} ({target_schema_class.definition_type}) at '{branch_name}'{_RESET}")

        # Collect all triggered configs in order (respecting schema array order)
        all_triggered_configs = []
        print(f"trigger map: {_GRAY}{target_schema_class.trigger_map}{_RESET}")
        for trigger in current_event_triggers:
            if trigger in target_schema_class.trigger_map:
                for action, config in target_schema_class.trigger_map[trigger]:
                    all_triggered_configs.append((action, config))

        # Execute actions in schema order
        i = 0
        while i < len(all_triggered_configs):
            action, config = all_triggered_configs[i]
            when_condition = config.get("when") if config else None

            # Skip conditional actions until their condition is met
            if when_condition == "gate_check_fail":
                i += 1
                continue

            stop_processing, gate_failed, skip_children = self._execute_action(
                action, config, branch_name, data,
                formatted_data, unexpanded_formatted_data,
                target_schema_class, keys
            )

            if gate_failed:
                # Execute remaining conditional actions for gate_check_fail
                for j in range(i + 1, len(all_triggered_configs)):
                    cond_action, cond_config = all_triggered_configs[j]
                    if cond_config and cond_config.get("when") == "gate_check_fail":
                        self._execute_action(
                            cond_action, cond_config, branch_name, data,
                            formatted_data, unexpanded_formatted_data,
                            target_schema_class, keys
                        )
                return data

            if skip_children:
                return data

            if stop_processing:
                return data

            i += 1

        # Drill Down / Recursion
        self._traverse_structure(
            branch_name,
            data,
            formatted_data,
            unexpanded_formatted_data,
            target_schema_class,
            keys,
            trigger_update=trigger_update
        )

        return data

    def _traverse_structure(
        self,
        item_name_prefix: str,
        data: dict,
        formatted_data: FormattedData,
        unexpanded_formatted_data: FormattedData,
        schema_class: ParsedSchemaClass,
        keys: list,
        trigger_update: bool = False,
    ):
        """
        Traverse and update `data` according to `schema_class`, recursing into nested schema classes, lists, and dicts.
        
        This mutates `data` and `formatted_data` in place: it descends dataclass fields, unwraps alias/field wrappers, recurses into nested ParsedSchemaClass instances, and applies per-item or per-value updates for lists and dicts (using `_update_recursive` and `_update_field`). For list/dict elements that are schema classes this starts and completes sub-phases via the phase manager to track progress.
        
        Parameters:
            item_name_prefix (str): Human-readable path prefix used for prompts and phase ids (e.g., "chapter.scenes" or "characters[0]").
            data (dict): The current branch of data to traverse and potentially update.
            formatted_data (FormattedData): Formatted view of the data used to render prompts and to update contextual markers; this will be kept in sync with changes.
            unexpanded_formatted_data (FormattedData): Unexpanded formatted view used when creating prompts that require original/unexpanded values.
            schema_class (ParsedSchemaClass): Schema description that determines traversal behavior (dataclass, alias/field, wrapped list/dict, or nested schema).
            keys (list): List of keys/indices representing the path within `formatted_data.data` corresponding to `data`.
            trigger_update (bool): If True, allow LLM calls for leaf fields. Defaults to False (opt-in).
        """

        # --- Case A: Dataclass (object with defined fields) ---
        if schema_class.definition_type == "dataclass":
            for field in schema_class.get_fields():
                self._process_field(
                    parent_path=item_name_prefix,
                    parent_data=data,
                    field_def=field,
                    formatted_data=formatted_data,
                    unexpanded_formatted_data=unexpanded_formatted_data,
                    parent_schema_class=schema_class,
                    parent_keys=keys,
                    trigger_update=trigger_update
                )

        # --- Case B: Alias (wrapper around a type) ---
        elif schema_class.definition_type == "alias" or schema_class.definition_type == "field":
            wrapped_type = schema_class._field.type

            # 1. Direct nested Schema Class (Alias -> Alias || Alias -> Dataclass)
            if isinstance(wrapped_type, ParsedSchemaClass):
                effective_child_schema = self._inherit_defaults_from_parent(
                    wrapped_type,
                    schema_class,
                    defaults_to_inherit
                )

                # Pass the exact same 'data' and 'keys'
                self._update_recursive(
                    item_name_prefix,
                    data,
                    formatted_data,
                    unexpanded_formatted_data,
                    effective_child_schema,
                    keys,
                    trigger_update=trigger_update
                )
                return

            # 3. Dict container
            elif hasattr(wrapped_type, "__origin__") and wrapped_type.__origin__ is dict:
                # dict[key_type, value_type]
                value_type = wrapped_type.__args__[1]

                if isinstance(data, dict):
                    # 3a. Dict of Schema Classes (Recurse)
                    if isinstance(value_type, ParsedSchemaClass):
                        pm = self._phase_manager
                        # Entry-selection whitelist (set by select_entries_to_update
                        # at the map level): None/absent = run all. Entries added
                        # by add_new this pass are always run (they need initial
                        # population even if the selection call didn't name them —
                        # add_new may fire after selection in the trigger order).
                        whitelist = self._entry_whitelists.get(item_name_prefix)
                        new_this_pass = self._new_entry_names.get(item_name_prefix, set())
                        for key, val_data in data.items():
                            # Heal array-wrapped class instances BEFORE the
                            # whitelist check so a model-written [{...}] shape
                            # is repaired (and saved healed) rather than
                            # crashing the subject's traversal.
                            normalized = self._normalize_schema_instance(val_data)
                            if normalized is not val_data:
                                print(f"{_GRAY}Normalized array-wrapped class instance "
                                      f"at '{item_name_prefix}.{key}' "
                                      f"(model wrote the object as a list).{_RESET}")
                                data[key] = normalized
                                val_data = normalized
                            if whitelist is not None and key not in whitelist and key not in new_this_pass:
                                print(
                                    f"{_GRAY}select_entries_to_update: skipping '{key}' "
                                    f"(not selected for update).{_RESET}"
                                )
                                continue
                            effective_val_schema = self._inherit_defaults_from_parent(
                                value_type,
                                schema_class,
                                defaults_to_inherit
                            )

                            sub_phase_id = f"{item_name_prefix}.{key}".lower().replace(" ", "_")
                            pm.start_phase(sub_phase_id, f"{item_name_prefix}.{key}")

                            new_keys = [*keys, key]
                            self._update_recursive(
                                f"{item_name_prefix}.{key}",
                                val_data,
                                formatted_data,
                                unexpanded_formatted_data,
                                effective_val_schema,
                                new_keys,
                                trigger_update=trigger_update
                            )
                            recursive_set(formatted_data.data, new_keys, val_data)
                            pm.done_phase(sub_phase_id)

                    # 3b. Dict of Primitives (Leaf Update)
                    else:
                        # Create a dummy field definition to represent the dict value
                        dummy_field = copy.copy(schema_class._field)
                        dummy_field.type = value_type

                        for key, val_data in data.items():
                            self._update_field(
                                parent_item_name_prefix=item_name_prefix,  # Path up to dict
                                parent_data_object=data,                   # The dict itself
                                field_name=key,
                                field_value=val_data,
                                formatted_data=formatted_data,
                                parent_schema_class=schema_class,          # Alias as parent for prompts
                                field=dummy_field,
                                keys=keys,
                                trigger_update=trigger_update
                            )

    @staticmethod
    def _normalize_schema_instance(value: Any) -> Any:
        """Normalize a schema-class instance the model wrapped in an array.

        LLM update responses sometimes write a class object as ``[{...}]``
        (single-element array) or split its fields across several array items
        (e.g. ``"relationships": {"Juno": [{...}, {...}]}``). Entry-time
        coercion cannot rebuild these (the items carry no name field), and
        traversing them as if they were the class dict crashes with
        "list indices must be integers or slices, not str". Unwrap to a plain
        dict so traversal can process them; non-repairable values pass through.
        """
        if isinstance(value, list):
            dicts = [x for x in value if isinstance(x, dict)]
            if dicts and len(dicts) == len(value):
                if len(dicts) == 1:
                    return dicts[0]
                merged: dict = {}
                for d in dicts:
                    merged.update(d)
                return merged
        return value

    def _process_field(
        self,
        parent_path: str,
        parent_data: dict,
        field_def: ParsedSchemaField,
        formatted_data: FormattedData,
        unexpanded_formatted_data: FormattedData,
        parent_schema_class: ParsedSchemaClass,
        parent_keys: list,
        trigger_update: bool = False,
    ):
        """
        Process and update a single field within a parent data object according to its schema definition.
        
        This function:
        - Initializes the field when missing.
        - Skips fields that are internal (names starting with "_") or marked `no_update`.
        - If the field's type is a ParsedSchemaClass, recursively updates its nested structure (inheriting defaults from the parent).
        - If the field is a list or dict whose element/value type is a ParsedSchemaClass, iterates each element/value and recursively updates each item with its own phase.
        - Otherwise treats the field as a leaf (primitive or container of primitives) and delegates to `_update_field` to request/apply updates.
        
        Parameters:
            parent_path (str): Dot-separated path of the parent object (empty for top-level).
            parent_data (dict): The parent data object containing the field.
            field_def (ParsedSchemaField): Schema definition for the field to process.
            formatted_data (FormattedData): Formatted representation of the current data used for prompt generation and marking.
            unexpanded_formatted_data (FormattedData): Unexpanded formatted data used when generating prompts that require raw content.
            parent_schema_class (ParsedSchemaClass): Schema class of the parent, used for inheriting defaults for nested schemas.
            parent_keys (list): List of keys representing the path to the parent within the formatted data structure.
            trigger_update (bool): If True, allow LLM calls for leaf fields. Defaults to False (opt-in).
        """
        field_name = field_def.name
        full_path = f"{parent_path}.{field_name}" if parent_path else field_name
        current_keys = [*parent_keys, field_name]

        # 1. Skip if internal
        if field_name.startswith("_"):
            return

        # 2. Check no_update flag
        if field_def.no_update:
            return

        # 3. Initialize missing data
        if field_name not in parent_data:
            self._initialize_field(parent_data, field_name, field_def)

        current_value = parent_data[field_name]
        field_type = field_def.type

        # Determine if this field should trigger updates (from parent schema defaults)
        field_update_key = f"{field_name}_update"
        field_trigger_update = parent_schema_class.defaults.get(field_update_key, trigger_update)

        # 4. Handle Recursion for nested Schema Classes
        if isinstance(field_type, ParsedSchemaClass):
            # Create effective schema for the child (inheriting prompt templates from parent)
            effective_child_schema = self._inherit_defaults_from_parent(
                field_type,
                parent_schema_class,
                defaults_to_inherit
            )
            # A list-alias field's natural shape IS a list of instances
            # (events.chapters = list[Chapter]): unwrapping a legal [{...}]
            # into a bare dict corrupts the archive downstream
            # (_entries_as_list flattens a bare dict via dict.values()).
            # Heal only fields whose type is not list-originated.
            _ft_origin = getattr(getattr(getattr(field_type, "_field", None), "type", None), "__origin__", None)
            if _ft_origin is not list:
                normalized = self._normalize_schema_instance(current_value)
                if normalized is not current_value:
                    print(f"{_GRAY}Normalized array-wrapped class instance at "
                          f"'{full_path}' (model wrote the object as a list).{_RESET}")
                    parent_data[field_name] = normalized
                    current_value = normalized

            sub_phase_id = full_path.lower().replace(" ", "_")
            pm = self._phase_manager
            pm.start_phase(sub_phase_id, full_path)
            self._update_recursive(
                full_path,
                current_value,
                formatted_data,
                unexpanded_formatted_data,
                effective_child_schema,
                current_keys,
                trigger_update=field_trigger_update
            )
            recursive_set(formatted_data.data, current_keys, current_value)
            pm.done_phase(sub_phase_id)
            return

        # 5. Handle List/Dict of Schema Classes (Generics defined directly on a field, not via Alias)
        # e.g. fields = { "my_list": "list[MyClass]" }
        origin = getattr(field_type, "__origin__", None)
        args = getattr(field_type, "__args__", tuple())

        if origin is list and args and isinstance(args[0], ParsedSchemaClass):
            # List of objects
            item_schema = args[0]
            pm = self._phase_manager
            if isinstance(current_value, list):
                for i, item in enumerate(current_value):
                    effective_schema = self._inherit_defaults_from_parent(item_schema, parent_schema_class, defaults_to_inherit)
                    sub_phase_id = f"{full_path}[{i}]".lower().replace(" ", "_")
                    pm.start_phase(sub_phase_id, f"{full_path}[{i}]")
                    self._update_recursive(
                        f"{full_path}[{i}]",
                        item,
                        formatted_data,
                        unexpanded_formatted_data,
                        effective_schema,
                        [*current_keys, i],
                        trigger_update=field_trigger_update
                    )
                    recursive_set(formatted_data.data, [*current_keys, i], item)
                    pm.done_phase(sub_phase_id)
            return

        elif origin is dict and len(args) > 1 and isinstance(args[1], ParsedSchemaClass):
            # Dict of objects
            val_schema = args[1]
            pm = self._phase_manager
            if isinstance(current_value, dict):
                for k, v in current_value.items():
                    effective_schema = self._inherit_defaults_from_parent(val_schema, parent_schema_class, defaults_to_inherit)
                    sub_phase_id = f"{full_path}.{k}".lower().replace(" ", "_")
                    pm.start_phase(sub_phase_id, f"{full_path}.{k}")
                    self._update_recursive(
                        f"{full_path}.{k}",
                        v,
                        formatted_data,
                        unexpanded_formatted_data,
                        effective_schema,
                        [*current_keys, k],
                        trigger_update=field_trigger_update
                    )
                    recursive_set(formatted_data.data, [*current_keys, k], v)
                    pm.done_phase(sub_phase_id)
            return

        # 6. Handle Leaf Nodes (Primitives, or Lists/Dicts of Primitives)
        self._update_field(
            parent_path,
            parent_data,
            field_name,
            current_value,
            formatted_data,
            parent_schema_class,
            field_def,
            parent_keys,
            trigger_update=field_trigger_update
        )

    def _initialize_field(self, data: dict, field_name: str, field_def: ParsedSchemaField):
        """Initializes a missing field with a safe default."""
        if not isinstance(data, dict):
            # Fail open: a malformed container (e.g. model wrote a list where
            # the schema declares an object) must never abort the whole subject.
            print(f"{_GRAY}Skipping init of '{field_name}': parent container is "
                  f"{type(data).__name__}, not dict.{_RESET}")
            return
        field_type = field_def.type

        # Check origin for generics (list, dict)
        origin = getattr(field_type, "__origin__", None)

        if origin is list:
            data[field_name] = []
        elif origin is dict:
            data[field_name] = {}
        elif field_def.default is not None:
             data[field_name] = copy.deepcopy(field_def.default)
        elif isinstance(field_type, ParsedSchemaClass):
            # TODO: If it's a class, instantiate a minimal dict for it
            data[field_name] = {}
        else:
            # Primitives
            if field_type is str: data[field_name] = ""
            elif field_type is int: data[field_name] = 0
            elif field_type is float: data[field_name] = 0.0
            elif field_type is bool: data[field_name] = False
            else: data[field_name] = None

        print(f"{_GRAY}Initialized missing field '{field_name}'{_RESET}")
