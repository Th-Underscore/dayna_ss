from pathlib import Path
import copy
import json
import jsonc
import traceback
import re
import time
from typing import Any

import modules.shared as shared

from extensions.dayna_ss.utils.helpers import (
    _ERROR,
    _SUCCESS,
    _INPUT,
    _GRAY,
    _HILITE,
    _BOLD,
    _RESET,
    _DEBUG,
    save_json,
    split_keys_to_list,
    recursive_set,
    recursive_get,
    expand_lists_in_data_for_llm,
    unexpand_lists_in_data_from_llm,
    strip_thinking,
    strip_response,
    format_str,
)
from extensions.dayna_ss.utils.schema_parser import (
    SchemaParser,
    ParsedSchemaClass,
    ParsedSchemaField,
    Action,
    Trigger,
)
from extensions.dayna_ss.agents.summarizer import Summarizer, FormattedData


defaults_to_inherit = [
    "gate_check_prompt_template",
    "branch_query_prompt_template",
    "branch_update_prompt_template",
    "update_prompt_template",
]


class DataSummarizer:
    def __init__(
        self,
        summarizer: Summarizer,
        exchange: tuple[str, str],
        custom_state: dict,
        history_path: Path,
        schema_parser: SchemaParser,
        all_subjects_data: dict,
    ):
        """Initialize DataSummarizer.

        Args:
            summarizer (Summarizer): The main Summarizer instance.
            history_path (Path): Path to the current history data.
            schema_parser (SchemaParser): Parser for data schemas.
        """
        self.summarizer = summarizer
        self.user_input = exchange[0]
        self.output = exchange[1]
        self.custom_state = custom_state
        print(
            f"DataSummarizer initialized with history: {_DEBUG} {json.dumps(self.custom_state['history']['internal'], indent=2)}{_RESET}"
        )
        self.history_path = history_path
        self.schema_parser = schema_parser
        self.all_subjects_data = all_subjects_data

    def _get_effective_setting(
        self,
        data: dict,
        schema_definition: ParsedSchemaClass,
        base_setting_name: str,
        field_name_context: str | None = None,
    ):
        """
        Gets an effective setting value, primarily for prompt templates.

        Priority:
        1. Direct override for base_setting_name in data._overrides.
        2. For templates, field-specific override in data._overrides (e.g., description_prompt_template).
        3. Schema attribute on schema_definition (e.g., schema_definition.gate_check_prompt_template).
        4. For templates, field-specific template from schema_definition.defaults.
        5. General base_setting_name from schema_definition.defaults (if not already an attribute) or schema attribute.
        6. Fallback to None for templates.

        Args:
            data (dict): The actual data dictionary that might contain _overrides.
            schema_definition (ParsedSchemaClass): The schema definition for this data.
            base_setting_name (str): The name of the setting to get (e.g., "update_prompt_template").
            field_name_context (str, optional): For field-specific templates, e.g., "description". Defaults to None.
        """
        overrides = data.get("_overrides", {})

        # 1. Direct override for the base_setting_name
        if base_setting_name in overrides:
            return overrides[base_setting_name]

        # 2. For prompt templates, check for field-specific override
        if field_name_context and base_setting_name.endswith("_prompt_template"):
            field_specific_override_key = f"{field_name_context}_prompt_template"
            if field_specific_override_key in overrides:
                return overrides[field_specific_override_key]

        # 3. Get from schema_definition attributes (e.g., schema_definition.gate_check_prompt_template)
        schema_attr_value = getattr(schema_definition, base_setting_name, None)

        # 4. For prompt templates, field-specific template from schema_definition.defaults
        if field_name_context and base_setting_name.endswith("_prompt_template"):
            field_specific_schema_key = f"{field_name_context}_prompt_template"
            if field_specific_schema_key in schema_definition.defaults:
                return schema_definition.defaults[field_specific_schema_key]
            if schema_attr_value is not None:
                return schema_attr_value
            if base_setting_name in schema_definition.defaults:
                return schema_definition.defaults[base_setting_name]

        # 5. If not a field-specific template context, or if it was but not found:
        if schema_attr_value is not None:
            return schema_attr_value
        if base_setting_name in schema_definition.defaults:
            return schema_definition.defaults[base_setting_name]

        # 6. Fallback default (primarily for templates)
        if base_setting_name.endswith("_prompt_template"):
            return None

        # Boolean flags for actions are now handled by trigger_map.
        return None

    def _get_current_event_triggers(self):
        """Get the current event triggers based on the last turn, which can be used to determine whether an action is triggered."""
        current_event_triggers = [Trigger.ALWAYS]
        if self.summarizer.last and self.summarizer.last.is_new_scene_turn:
            current_event_triggers.append(Trigger.ON_NEW_SCENE)
        else:
            current_event_triggers.append(Trigger.ON_EXISTING_SCENE)
        return current_event_triggers

    def _is_action_triggered(  # TODO?: Also accept overrides?
        self,
        schema_class: ParsedSchemaClass,
        action: Action,
        event_triggers: list[Trigger] = [],
    ) -> bool:
        """Whether a given action is triggered by any of the current event conditions. Use `_should_update_subject` for a general check."""
        if not schema_class or not schema_class.trigger_map:
            return False
        for trigger in event_triggers or self._get_current_event_triggers():
            if action in schema_class.trigger_map.get(trigger, []):
                return True
        return False

    def _should_update_subject(self, schema_class: ParsedSchemaClass, event_triggers: list[Trigger] = []) -> bool:
        """Whether the subject should be updated at all based on the schema_class's trigger_map. Use `_is_action_triggered` for more fine-grained control."""
        if not schema_class or not schema_class.trigger_map:
            return False
        for trigger in event_triggers or self._get_current_event_triggers():
            if schema_class.trigger_map.get(trigger):
                return True
        return False

    def generate(self, data_type: str, data: dict, target_schema_class: ParsedSchemaClass) -> dict:
        """Dynamically generate summaries for data based on its class structure.

        Args:
            data_type (str): Type of data being summarized (e.g., 'characters', 'groups').
            data (dict): Data to summarize.
            target_schema_class (ParsedSchemaClass): The schema class for the target data structure.

        Returns:
            out (dict): Updated data with summaries, modified in place.
        """
        try:
            print(f"{_BOLD}Summarizing {data_type}{_RESET}")
            start = time.time()

            if not (data or isinstance(data, dict)):
                print(f"{_ERROR}No {data_type} data to summarize{_RESET}")
                return data

            subject_type = self.schema_parser.subjects.get(data_type)
            print(f"{_HILITE}Subject type for '{data_type}':{_RESET} {subject_type}")
            unexpanded_formatted_data = FormattedData(data, data_type)
            formatted_data = FormattedData(data, data_type, self.schema_parser)
            if not isinstance(subject_type, ParsedSchemaClass):
                raise ValueError(f"Subject '{data_type}' should be a ParsedSchemaClass: {subject_type}")

            self._update_recursive(data_type, data, formatted_data, unexpanded_formatted_data, target_schema_class)

            print(f"{_HILITE}Summary for {data_type} completed in {time.time() - start:.2f} seconds.{_RESET}")
            save_json(
                unexpand_lists_in_data_from_llm(data, target_schema_class, self.schema_parser),
                self.history_path / f"{data_type}.json",
            )
            return data

        except Exception as e:
            print(f"{_ERROR}Error in summarize_{data_type}: {e}{_RESET}")
            traceback.print_exc()
            return data

    def _detect_and_add_new_entries_to_branch(
        self,
        branch_name: str,
        data: dict,
        formatted_data: FormattedData,
        branch_schema_class: ParsedSchemaClass,
        new_query_template: str | None = None,
        new_entry_template: str | None = None,
        keys: list = [],
    ):
        """Handles querying for and adding new entries to a dictionary-like branch."""
        if not (
            branch_schema_class.definition_type == "alias"
            and hasattr(branch_schema_class._field.type, "__origin__")
            and branch_schema_class._field.type.__origin__ is dict
        ):
            # This logic is for dictionary fields like "Characters: dict[str, Character]"
            return

        new_query_template = new_query_template or self._get_effective_setting(
            data, branch_schema_class, "new_entry_query_prompt_template"
        )
        new_entry_template = new_entry_template or self._get_effective_setting(
            data, branch_schema_class, "new_entry_prompt_template"
        )

        if not new_query_template or not new_entry_template:
            print(
                f"{_GRAY}Missing new query or entry prompt template for '{branch_name}'. Skipping new entry addition.{_RESET}"
            )
            return

        print(f"{_INPUT}Querying for new entries to add to '{branch_name}'...{_RESET}")

        # Step 1: Query for names of new entries
        query_prompt = self._create_update_prompt(
            item_name=branch_name,
            field_name="",  # Not specific to a field
            formatted_data=formatted_data,
            prompt_template_str=new_query_template,
            target_schema_or_type=branch_schema_class,
            keys=keys,
        )

        query_stopping_strings = ["NO", "[]"]
        llm_query_response, stop_reason = self.summarizer.generate_using_tgwui(
            prompt=query_prompt,
            state=self.custom_state or {},
            history_path=self.history_path,
            stopping_strings=query_stopping_strings,
            match_prefix_only=True,
        )

        if shared.stop_everything:
            return
        stripped_response = strip_response(llm_query_response)
        if (stop_reason and stop_reason in query_stopping_strings) or stripped_response in query_stopping_strings:
            print(f"{_GRAY}LLM indicates no new entries for '{branch_name}'.{_RESET}")
            return

        new_entry_names = []
        try:
            parsed_names = jsonc.loads(stripped_response)
            if not parsed_names:
                print(f"{_INPUT}LLM response for new entry names for '{branch_name}' was empty: {llm_query_response}{_RESET}")
                return
            elif isinstance(parsed_names, list) and all(isinstance(name, str) for name in parsed_names):
                new_entry_names = parsed_names
                print(f"{_SUCCESS}Identified potential new entries for '{branch_name}': {new_entry_names}{_RESET}")
            else:
                print(
                    f"{_ERROR}LLM response for new entry names for '{branch_name}' was not a list of strings: {llm_query_response}{_RESET}"
                )
                return
        except json.JSONDecodeError:
            print(
                f"{_ERROR}Failed to parse LLM response for new entry names for '{branch_name}' as JSON: {llm_query_response}{_RESET}"
            )
            return

        if not new_entry_names:
            return

        # Step 2: Generate data for each new entry
        value_schema_class = None
        if hasattr(branch_schema_class._field.type, "__args__") and len(branch_schema_class._field.type.__args__) == 2:
            value_type_candidate = branch_schema_class._field.type.__args__[1]
            if isinstance(value_type_candidate, ParsedSchemaClass):
                value_schema_class = value_type_candidate

        if not value_schema_class:
            print(f"{_ERROR}Could not determine value schema for new entries in '{branch_name}'. Skipping generation.{_RESET}")
            return

        for entry_name in new_entry_names:
            if shared.stop_everything:
                return
            print(f"{_INPUT}Generating data for new entry '{entry_name}' in '{branch_name}'...{_RESET}")

            entry_generation_prompt = self._create_update_prompt(
                item_name=branch_name,
                field_name="",  # Not updating a sub-field of the new entry yet
                formatted_data=formatted_data,
                prompt_template_str=new_entry_template,
                target_schema_or_type=value_schema_class,
                entry_name=entry_name,
                keys=keys,
            )

            llm_entry_response, _ = self.summarizer.generate_using_tgwui(
                prompt=entry_generation_prompt,
                state=self.custom_state or {},
                history_path=self.history_path,
                # No specific stopping strings, expect full JSON
            )
            if shared.stop_everything:
                return

            try:
                stripped_entry_json = strip_response(llm_entry_response)
                new_entry_data = jsonc.loads(stripped_entry_json)
                if isinstance(new_entry_data, dict):
                    data[entry_name] = new_entry_data
                    expanded_data: dict = recursive_get(formatted_data.data, keys, default=None)
                    expanded_data[entry_name] = new_entry_data
                    print(f"\r{_SUCCESS}Successfully added new entry '{entry_name}' to '{branch_name}'.{_RESET}")
                else:
                    print(
                        f"{_ERROR}LLM response for new entry '{entry_name}' in '{branch_name}' was not a JSON object: {llm_entry_response}{_RESET}"
                    )
            except json.JSONDecodeError:
                print(
                    f"{_ERROR}Failed to parse LLM response for new entry '{entry_name}' in '{branch_name}' as JSON: {llm_entry_response}{_RESET}"
                )

        recursive_set(formatted_data.data, keys, data)
        return data

    # --- Helper methods for parsing and applying LLM updates ---
    def _parse_llm_field_updates(self, llm_response_text: str, branch_name_for_log: str) -> list[dict[str, Any]]:
        """Parses LLM response for field updates.

        Expected LLM response formats:
        - "NO" ("NO_UPDATES_REQUIRED")
        - JSON list: `[{"path": "...", "value": ...}, ...]`
        - Single JSON object: `{"path": "...", "value": ...}`
        - Line-by-line: `path: path.to.field, value: new_value`
        An optional "END_OF_UPDATES" marker can be appended.

        Args:
            llm_response_text (str): The raw text response from the LLM.
            branch_name_for_log (str): Name of the data branch for logging purposes.

        Returns:
            out (list[dict[str, Any]]): A list of update dictionaries (`{"path": str, "value": Any}`), or an empty list if no valid updates are found.
        """
        response_text = llm_response_text.strip()

        if response_text.endswith("END_OF_UPDATES"):
            response_text = response_text[: -len("END_OF_UPDATES")].strip()

        if not response_text:
            print(f"{_GRAY}[{branch_name_for_log}] LLM response empty after stripping END_OF_UPDATES.{_RESET}")
            return []

        if response_text == "NO":
            print(f"{_INPUT}[{branch_name_for_log}] LLM indicates no updates required.{_RESET}")
            return []

        original_response_text = response_text
        response_text = strip_response(response_text)

        updates: list[dict[str, Any]] = []
        try:
            # Attempt to parse as JSON list
            parsed_json = jsonc.loads(response_text)
            if isinstance(parsed_json, list):
                # Validate structure of each item
                for item in parsed_json:
                    if isinstance(item, dict) and "path" in item and "value" in item:
                        updates.append(item)
                    else:
                        print(f"{_ERROR}[{branch_name_for_log}] Invalid item in JSON list: {item}. Skipping.{_RESET}")
            elif isinstance(parsed_json, dict) and "path" in parsed_json and "value" in parsed_json:
                updates = [parsed_json]  # Handle single update object
                print(f"{_GRAY}[{branch_name_for_log}] Treated single JSON object as a list of one update.{_RESET}")
            else:
                print(
                    f"{_ERROR}[{branch_name_for_log}] LLM response parsed as JSON but is not a list of updates or a single update object: {type(parsed_json)}{_RESET}\nRaw response: {original_response_text}\n{_DEBUG}Parsed response: {response_text}{_RESET}"
                )
                return []
        except json.JSONDecodeError:
            print(
                f"{_GRAY}[{branch_name_for_log}] LLM response not valid JSON, trying line-by-line parsing. Response: '{response_text[:100]}...' {_RESET}"
            )
            line_updates_temp = []
            line_pattern = re.compile(r"^\s*path\s*:\s*(?P<path>[^,]+?)\s*,\s*value\s*:\s*(?P<value>.+)\s*$", re.MULTILINE)
            for match in line_pattern.finditer(response_text):
                path = match.group("path").strip()
                value_str = match.group("value").strip()
                value: Any
                try:  # Try to interpret value as JSON primitive/object/array first
                    value = jsonc.loads(value_str)
                except json.JSONDecodeError:
                    val_lower = value_str.lower()
                    if val_lower == "true":
                        value = True
                    elif val_lower == "false":
                        value = False
                    elif val_lower == "null":
                        value = None
                    else:
                        try:
                            value = int(value_str)
                        except ValueError:
                            try:
                                value = float(value_str)
                            except ValueError:
                                value = value_str  # Default to string
                line_updates_temp.append({"path": path, "value": value})

            if line_updates_temp:
                updates = line_updates_temp
                print(f"{_GRAY}[{branch_name_for_log}] Parsed {len(updates)} updates from line-by-line format.{_RESET}")
            else:
                print(
                    f"{_ERROR}[{branch_name_for_log}] Failed to parse LLM response as JSON or line-by-line updates. Raw response: {response_text}{_RESET}"
                )
                return []

        # Final validation of updates list
        valid_updates = []
        for i, update_item in enumerate(updates):
            if not isinstance(update_item, dict) or "path" not in update_item or "value" not in update_item:
                print(
                    f"{_ERROR}[{branch_name_for_log}] Invalid update item format at index {i} after parsing: {update_item}{_RESET}"
                )
                continue
            if not isinstance(update_item["path"], str):
                print(
                    f"{_ERROR}[{branch_name_for_log}] Path in update item is not a string: {update_item['path']}. Skipping update.{_RESET}"
                )
                continue
            valid_updates.append(update_item)

        if not valid_updates and updates:  # Some items were filtered
            print(f"{_GRAY}[{branch_name_for_log}] Some parsed update items were invalid.{_RESET}")
        elif not updates:  # No updates parsed at all
            print(f"{_GRAY}[{branch_name_for_log}] No updates found in the parsed LLM response.{_RESET}")

        return valid_updates

    def _update_recursive(
        self,
        item_name_prefix: str,
        data: dict,
        formatted_data: FormattedData,
        unexpanded_formatted_data: FormattedData,
        target_schema_class: ParsedSchemaClass,
        keys: list = [],
    ) -> dict:
        """
        Entry point for processing a specific Schema Class node against a data object.
        Handles Triggers (Add New, Gate Check, Branch Query) before drilling down.
        """
        if shared.stop_everything:
            return data

        if not isinstance(target_schema_class, ParsedSchemaClass):
            print(f"{_ERROR}Target schema is not a ParsedSchemaClass: {type(target_schema_class)}{_RESET}")
            return data

        current_event_triggers = self._get_current_event_triggers()
        branch_name = item_name_prefix or target_schema_class.name

        print(f"{_BOLD}Processing Schema Node: {target_schema_class.name} ({target_schema_class.definition_type}) at '{branch_name}'{_RESET}")

        # 1. TRIGGER: Add New Entries (only valid for Alias types wrapping a dict)
        if self._is_action_triggered(target_schema_class, Action.ADD_NEW, current_event_triggers):
            if target_schema_class.definition_type == "alias":
                field_type = target_schema_class._field.type
                is_dict = hasattr(field_type, "__origin__") and field_type.__origin__ is dict

                if is_dict:
                    self._detect_and_add_new_entries_to_branch(
                        branch_name=branch_name,
                        data=data,
                        formatted_data=unexpanded_formatted_data,
                        branch_schema_class=target_schema_class,
                        keys=keys
                    )

        # 2. TRIGGER: Gate Check
        gate_passed = True
        gate_template = self._get_effective_setting(data, target_schema_class, "gate_check_prompt_template")

        if self._is_action_triggered(target_schema_class, Action.PERFORM_GATE_CHECK, current_event_triggers) and gate_template:
            gate_passed = self._perform_gate_check(
                branch_name,
                gate_template,
                target_schema_class,
                formatted_data,
                keys
            )

        if not gate_passed:
            return data

        # 3. TRIGGER: Full Update ("Overwrite")
        update_template = self._get_effective_setting(data, target_schema_class, "update_prompt_template")
        did_full_update = False

        if self._is_action_triggered(target_schema_class, Action.PERFORM_UPDATE, current_event_triggers) and update_template:
            # Perform update and return early if successful
            did_full_update = self._perform_full_branch_update(
                branch_name,
                data,
                update_template,
                formatted_data,
                target_schema_class,
                keys
            )
            if did_full_update:
                return data

        # 4. TRIGGER: Branch Query ("Diff")
        # If gate check passed (or wasn't needed), we proceed.
        bq_template = self._get_effective_setting(data, target_schema_class, "branch_query_prompt_template")
        bu_template = self._get_effective_setting(data, target_schema_class, "branch_update_prompt_template")

        branch_query_handled = False
        if (
            not did_full_update  # Skip if already did a full update
            and self._is_action_triggered(target_schema_class, Action.QUERY_BRANCH_FOR_CHANGES, current_event_triggers)
            and bq_template
            and bu_template
        ):
            self._perform_branch_query(
                branch_name,
                data,
                formatted_data,
                bq_template,
                bu_template,
                target_schema_class,
                keys
            )
            branch_query_handled = True

        # 6. Drill Down / Recursion
        if not branch_query_handled:
            self._traverse_structure(
                branch_name,
                data,
                formatted_data,
                unexpanded_formatted_data,
                target_schema_class,
                keys
            )

        return data

    def _traverse_structure(
        self,
        item_name_prefix: str,
        data: dict,
        formatted_data: FormattedData,
        unexpanded_formatted_data: FormattedData,
        schema_class: ParsedSchemaClass,
        keys: list
    ):
        """
        Handles the structural iteration based on definition_type.
        Unwraps 'alias' types to find the real structure underneath.
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
                    parent_keys=keys
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
                    keys
                )
                return

            # 2. List container
            if hasattr(wrapped_type, "__origin__") and wrapped_type.__origin__ is list:
                item_type = wrapped_type.__args__[0]

                if isinstance(data, list):
                    # 2a. List of Schema Classes (Recurse)
                    if isinstance(item_type, ParsedSchemaClass):
                        for i, item_data in enumerate(data):
                            effective_item_schema = self._inherit_defaults_from_parent(
                                item_type,
                                schema_class,
                                defaults_to_inherit
                            )
                            # Sync formatted data to keep context fresh
                            new_keys = [*keys, i]
                            self._update_recursive(
                                f"{item_name_prefix}[{i}]",
                                item_data,
                                formatted_data,
                                unexpanded_formatted_data,
                                effective_item_schema,
                                new_keys
                            )
                            recursive_set(formatted_data.data, new_keys, item_data)

                    # 2b. List of Primitives (Leaf update)
                    else:
                        # Create a dummy field definition to represent the list item
                        dummy_field = copy.copy(schema_class._field)
                        dummy_field.type = item_type

                        for i, item_val in enumerate(data):
                            self._update_field(
                                parent_item_name_prefix=item_name_prefix,  # Path up to list
                                parent_data_object=data,                   # The list itself
                                field_name=i,
                                field_value=item_val,
                                formatted_data=formatted_data,
                                parent_schema_class=schema_class,          # Alias as parent for prompts
                                field=dummy_field,
                                keys=keys
                            )

            # 3. Dict container
            elif hasattr(wrapped_type, "__origin__") and wrapped_type.__origin__ is dict:
                # dict[key_type, value_type]
                value_type = wrapped_type.__args__[1]

                if isinstance(data, dict):
                    # 3a. Dict of Schema Classes (Recurse)
                    if isinstance(value_type, ParsedSchemaClass):
                        for key, val_data in data.items():
                            effective_val_schema = self._inherit_defaults_from_parent(
                                value_type,
                                schema_class,
                                defaults_to_inherit
                            )

                            new_keys = [*keys, key]
                            self._update_recursive(
                                f"{item_name_prefix}.{key}",
                                val_data,
                                formatted_data,
                                unexpanded_formatted_data,
                                effective_val_schema,
                                new_keys
                            )
                            recursive_set(formatted_data.data, new_keys, val_data)

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
                                keys=keys
                            )

    def _process_field(
        self,
        parent_path: str,
        parent_data: dict,
        field_def: ParsedSchemaField,
        formatted_data: FormattedData,
        unexpanded_formatted_data: FormattedData,
        parent_schema_class: ParsedSchemaClass,
        parent_keys: list
    ):
        """
        Decides how to handle a specific field within a parent data object.
        1. Initializes missing data.
        2. Checks 'no_update'.
        3. Branches: Update Leaf (Primitive) OR Recurse (SchemaClass/Container).
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

        # 4. Handle Recursion for nested Schema Classes
        if isinstance(field_type, ParsedSchemaClass):
            # Create effective schema for the child (inheriting prompt templates from parent)
            effective_child_schema = self._inherit_defaults_from_parent(
                field_type,
                parent_schema_class,
                defaults_to_inherit
            )

            self._update_recursive(
                full_path,
                current_value,
                formatted_data,
                unexpanded_formatted_data,
                effective_child_schema,
                current_keys
            )
            recursive_set(formatted_data.data, current_keys, current_value)
            return

        # 5. Handle List/Dict of Schema Classes (Generics defined directly on a field, not via Alias)
        # e.g. fields = { "my_list": "list[MyClass]" }
        origin = getattr(field_type, "__origin__", None)
        args = getattr(field_type, "__args__", tuple())

        if origin is list and args and isinstance(args[0], ParsedSchemaClass):
            # List of objects
            item_schema = args[0]
            if isinstance(current_value, list):
                for i, item in enumerate(current_value):
                    effective_schema = self._inherit_defaults_from_parent(item_schema, parent_schema_class, defaults_to_inherit)
                    self._update_recursive(
                        f"{full_path}[{i}]",
                        item,
                        formatted_data,
                        unexpanded_formatted_data,
                        effective_schema,
                        [*current_keys, i]
                    )
                    recursive_set(formatted_data.data, [*current_keys, i], item)
            return

        elif origin is dict and len(args) > 1 and isinstance(args[1], ParsedSchemaClass):
            # Dict of objects
            val_schema = args[1]
            if isinstance(current_value, dict):
                for k, v in current_value.items():
                    effective_schema = self._inherit_defaults_from_parent(val_schema, parent_schema_class, defaults_to_inherit)
                    self._update_recursive(
                        f"{full_path}.{k}",
                        v,
                        formatted_data,
                        unexpanded_formatted_data,
                        effective_schema,
                        [*current_keys, k]
                    )
                    recursive_set(formatted_data.data, [*current_keys, k], v)
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
            parent_keys
        )

    def _initialize_field(self, data: dict, field_name: str, field_def: ParsedSchemaField):
        """Initializes a missing field with a safe default."""
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

    def _perform_gate_check(
        self,
        branch_name: str,
        template: str,
        schema_class: ParsedSchemaClass,
        formatted_data: FormattedData,
        keys: list
    ) -> bool:
        """
        Asks the LLM a Yes/No question to determine if this branch needs processing.
        Returns:
            out (bool): `True` if the branch should be processed, `False` if the branch should be skipped.
        """
        gate_check_prompt = self._create_update_prompt(
            item_name=branch_name,
            field_name="",  # Not a sub-field, but the branch itself
            formatted_data=formatted_data,
            prompt_template_str=template,
            target_schema_or_type=schema_class,
            keys=keys,
        )

        if not gate_check_prompt:
            return True

        print(f"{_GRAY}Performing gate check for '{branch_name}'...{_RESET}")

        # Contextualize the prompt
        gate_check_full_prompt = (
            f"Current context for '{branch_name}':\n"
            f"{formatted_data.mark_field(branch_name)}\n\n"
            f"{gate_check_prompt}"
        )

        current_custom_state = self.custom_state or {}
        stopping_strings = ["NO", "UNCHANGED", "YES"]

        llm_response_text, stop_reason = self.summarizer.generate_using_tgwui(
            prompt=gate_check_full_prompt,
            state=current_custom_state,
            history_path=self.history_path,
            stopping_strings=stopping_strings,
            match_prefix_only=True,
        )

        if shared.stop_everything:
            return False

        llm_response_text = strip_thinking(llm_response_text).strip()
        stop_reason = stop_reason.upper() if stop_reason else ""

        print(f"{_GRAY}Gate check response for '{branch_name}': '{llm_response_text}'. Stop: '{stop_reason}'{_RESET}")

        # Logic to determine YES vs NO
        # 1. Check stopping strings logic
        if stop_reason in ["NO", "UNCHANGED"]:
            print(f"{_INPUT}Gate check for '{branch_name}' returned {stop_reason}. Skipping branch.{_RESET}")
            return False

        # 2. Check text content logic
        is_negative = (
            llm_response_text.upper().startswith("NO") or
            "UNCHANGED" in llm_response_text.upper()
        )

        if is_negative:
            print(f"{_INPUT}Gate check for '{branch_name}' returned NO/UNCHANGED. Skipping branch.{_RESET}")
            return False

        # 3. Default to YES (Process the branch)
        print(f"{_INPUT}Gate check for '{branch_name}' returned YES. Proceeding.{_RESET}")
        return True

    def _perform_full_branch_update(
        self,
        branch_name: str,
        data: dict,
        template: str,
        formatted_data: FormattedData,
        schema_class: ParsedSchemaClass,
        keys: list
    ) -> bool:
        """
        Asks the LLM to rewrite the entire JSON object for the branch.
        Returns:
            out (bool): `True` if the branch was updated, `False` if not.
        """
        print(f"{_INPUT}Attempting direct update for branch '{branch_name}'...{_RESET}")

        # We expect the LLM to return the entire dictionary/object
        updated_branch_data = self._generate_field_update(
            item_name_prefix=branch_name,
            field_name="",
            current_value=data,
            formatted_data=formatted_data,
            prompt_template_str=template,
            expected_type=dict,  # Full update expects a dict structure
            target_schema_or_type=schema_class,
            keys=keys,
            context_marker_path_override=branch_name,
        )

        if shared.stop_everything:
            return False

        if (
            updated_branch_data is not None
            and isinstance(updated_branch_data, dict)
            and updated_branch_data is not data
        ):
            # Simple equality check to see if it actually changed (set a flag instead?)
            if updated_branch_data != data:
                print(f"{_SUCCESS}Applying direct branch update to '{branch_name}'.{_RESET}")
                data.update(updated_branch_data)
                return True
            else:
                print(f"{_GRAY}Direct branch update for '{branch_name}' resulted in identical data.{_RESET}")
                return True  # Returned valid data, just no change needed. Stop recursion.

        elif updated_branch_data is None:
            print(f"{_ERROR}Direct branch update for '{branch_name}' returned None.{_RESET}")
            return False  # Failed, maybe try recursion?

        return False

    def _perform_branch_query(
        self,
        branch_name: str,
        data: dict,
        formatted_data: FormattedData,
        query_template: str,
        update_template: str,
        schema_class: ParsedSchemaClass,
        keys: list
    ):
        """
        Performs the "Diff" strategy:
        1. Asks "Are there changes?"
        2. If YES, asks "List the changes (path, value)"
        3. Applies changes relative to this branch
        """
        # --- Step 1: Query ---
        branch_query_prompt = self._create_update_prompt(
            item_name=branch_name,
            field_name="",
            formatted_data=formatted_data,
            prompt_template_str=query_template,
            target_schema_or_type=schema_class,
            keys=keys,
        )

        if not branch_query_prompt:
            return

        print(f"{_GRAY}Querying LLM: Does branch '{branch_name}' need updates?{_RESET}")

        branch_query_full_prompt = (
            f"Current context for '{branch_name}':\n"
            f"{formatted_data.mark_field(branch_name)}\n\n"
            f"{branch_query_prompt}"
        )

        current_custom_state = self.custom_state or {}
        query_stopping_strings = ["NO", "UNCHANGED", "YES"]

        llm_response_text, stop_reason = self.summarizer.generate_using_tgwui(
            prompt=branch_query_full_prompt,
            state=current_custom_state,
            history_path=self.history_path,
            stopping_strings=query_stopping_strings,
            match_prefix_only=True,
        )

        if shared.stop_everything:
            return

        llm_response_text = strip_thinking(llm_response_text).strip()
        stop_reason = stop_reason.upper() if stop_reason else ""

        print(f"{_GRAY}Query branch '{branch_name}' response: '{llm_response_text}'. Stop: '{stop_reason}'{_RESET}")

        # Check if negative response
        if stop_reason in ["NO", "UNCHANGED"] or llm_response_text.upper() in ["NO", "UNCHANGED"]:
            print(f"{_INPUT}Skipping updates for branch '{branch_name}' (query returned NO).{_RESET}")
            return

        # --- Step 2: Request List of Changes ---
        # Simulate a conversation history so the LLM knows it just said "YES"
        temp_state = copy.deepcopy(current_custom_state)
        temp_state["history"]["internal"].append([branch_query_full_prompt, llm_response_text])

        branch_update_prompt = self._create_update_prompt(
            item_name=branch_name,
            field_name="",
            formatted_data=formatted_data,
            prompt_template_str=update_template,
            target_schema_or_type=schema_class,
            keys=keys,
        )

        print(f"{_INPUT}Query for '{branch_name}' suggests changes. Requesting field updates list...{_RESET}")

        branch_update_list_full_prompt = (
            f"Current context for '{branch_name}':\n"
            f"{formatted_data.mark_field(branch_name)}\n\n"
            f"{branch_update_prompt}"
        )

        update_stopping_strings = ["NO", "END"]

        llm_update_response_text, _ = self.summarizer.generate_using_tgwui(
            prompt=branch_update_list_full_prompt,
            state=temp_state,  # Use the temp state with history
            history_path=self.history_path,
            stopping_strings=update_stopping_strings,
            match_prefix_only=True,
        )

        if shared.stop_everything:
            return

        # --- Step 3: Parse and Apply ---
        parsed_updates = self._parse_llm_field_updates(llm_update_response_text, branch_name)

        if parsed_updates:
            print(f"{_INPUT}Applying {len(parsed_updates)} field update(s) to '{branch_name}'...{_RESET}")
            for update_item in parsed_updates:
                path_str = update_item["path"]
                value = update_item["value"]

                # Convert path string to key list (e.g. "status.0.name" -> ["status", 0, "name"])
                keyList_relative_to_branch = split_keys_to_list(path_str)

                try:
                    print(f"{_GRAY}[{branch_name}] Applying update: {path_str} == {json.dumps(recursive_get(data, keyList_relative_to_branch), indent=None)}{_RESET}")
                    recursive_set(data, keyList_relative_to_branch, value)  # Modifies 'data' in place
                    print(f"{_GRAY}[{branch_name}] Applied update: {path_str} = {json.dumps(value, indent=None)}{_RESET}")

                    # Also keep the 'formatted_data' in sync if needed by future prompts in this run
                    # (This depends on if your recursive_set handles the wrapper object or just dicts)
                    # recursive_set(formatted_data.data, [*keys, *keyList_relative_to_branch], value)

                except Exception as e:
                    print(f"{_ERROR}[{branch_name}] Failed to apply update {path_str} = {repr(value)}: {e}{_RESET}")
                    # traceback.print_exc()
        else:
            print(f"{_GRAY}No specific field updates applied to '{branch_name}' from branch query response.{_RESET}")

    def _retrieve_nested_field(
        self,
        target_schema_class: ParsedSchemaClass,
        defaults_to_inherit: list[str] | None = None,
        do_inherit_triggers: bool = False,
        depth: int = -1,
    ):
        """Recursively retrieve nested fields in a parent field, ending on the field holding the final schema field class.

        Example:
            ParsedSchemaClass(type='alias') -> ParsedSchemaClass(type='alias') -> ParsedSchemaClass(type='dataclass') => ParsedSchemaField(type=ParsedSchemaClass(type='dataclass'))
        """
        effective_child_schema = target_schema_class
        i = 0
        while (
            effective_child_schema.definition_type == "alias"
            and isinstance(effective_child_schema._field.type, ParsedSchemaClass)
            and effective_child_schema._field.type.definition_type == "alias"
        ):
            i += 1
            effective_child_schema = self._inherit_defaults_from_parent(
                effective_child_schema._field.type, effective_child_schema, defaults_to_inherit, do_inherit_triggers
            )
            if i == depth:
                break

        return effective_child_schema._field

    def _retrieve_nested_dataclass(
        self,
        target_schema_class: ParsedSchemaClass,
        defaults_to_inherit: list[str] | None = None,
        do_inherit_triggers: bool = False,
        depth: int = -1,
    ):
        """Recursively retrieve nested fields in a parent field, ending on the final schema field class.

        Example:
            ParsedSchemaClass(type='alias') -> ParsedSchemaClass(type='alias') -> ParsedSchemaClass(type='dataclass') => ParsedSchemaClass(type='dataclass')
        """
        effective_child_schema = target_schema_class
        i = 0
        while effective_child_schema.definition_type == "alias" and isinstance(
            effective_child_schema._field.type, ParsedSchemaClass
        ):
            i += 1
            effective_child_schema = self._inherit_defaults_from_parent(
                effective_child_schema._field.type, effective_child_schema, defaults_to_inherit, do_inherit_triggers
            )
            if i == depth:
                break

        return effective_child_schema

    def _inherit_defaults_from_parent(
        self,
        child_schema_class: ParsedSchemaClass,
        parent_schema_class: ParsedSchemaClass,
        defaults_to_inherit: list[str] | None = None,
        do_inherit_triggers: bool = False,
    ):
        """Inherit default values from parent schema class to child schema class."""
        effective_child_schema = copy.copy(child_schema_class)
        effective_child_schema.defaults = copy.copy(child_schema_class.defaults)

        if do_inherit_triggers:
            effective_child_schema.trigger_map = copy.copy(child_schema_class.trigger_map)
            effective_child_schema.trigger_map.update(parent_schema_class.trigger_map)

        if defaults_to_inherit:
            for attr_name in defaults_to_inherit:
                if getattr(effective_child_schema, attr_name, None) is None:
                    parent_template_value = getattr(parent_schema_class, attr_name, None)
                    if parent_template_value is not None:
                        setattr(effective_child_schema, attr_name, parent_template_value)
                        effective_child_schema.defaults[attr_name] = parent_template_value
        else:
            effective_child_schema.defaults.update(parent_schema_class.defaults)

        return effective_child_schema

    def _generate_field_update(
        self,
        item_name_prefix: str,
        field_name: str,
        current_value: Any,
        formatted_data: FormattedData,
        prompt_template_str: str,
        expected_type: type,
        target_schema_or_type: ParsedSchemaClass | None = None,
        keys: list = [],
        context_marker_path_override: str | None = None,
        entry_name_for_prompt: str | None = None,
    ) -> Any:
        """Core helper to get an updated value from the LLM for a given field or branch.

        Args:
            item_name_prefix (str): Prefix for the item name (e.g., "CharacterName").
            field_name (str): Name of the field being updated.
            current_value (Any): The current value of the field.
            formatted_data (FormattedData): Formatted data for LLM context.
            prompt_template_str (str): The prompt template string to use.
            expected_type (type): The expected Python type of the updated value.
            target_schema_or_type (ParsedSchemaClass, optional): Schema for snippet/example.
            keys (list, optional): Path keys to the current data level. Defaults to [].
            context_marker_path_override (str, optional): Override path for context marking. Defaults to None.
            entry_name_for_prompt (str, optional): Name of the new entry if generating one.

        Returns:
            out: The potentially updated value, or the original value if no update or an error.
        """
        try:
            prompt = self._create_update_prompt(
                item_name=item_name_prefix,
                field_name=field_name,
                formatted_data=formatted_data,
                prompt_template_str=prompt_template_str,
                target_schema_or_type=target_schema_or_type,
                entry_name=entry_name_for_prompt,
                keys=keys,
                indent=2,
            )

            current_custom_state = self.custom_state or {}

            context_path_for_marker = context_marker_path_override
            if context_path_for_marker is None:
                if isinstance(current_value, (dict, list)):
                    context_path_for_marker = item_name_prefix
                else:
                    context_path_for_marker = f"{item_name_prefix}.{field_name}"

            llm_interaction_prompt = f"Context for '{item_name_prefix}':\n{formatted_data.mark_field(item_name_prefix, context_path_for_marker)}\n\n{prompt}"

            text, stop = self.summarizer.generate_using_tgwui(
                llm_interaction_prompt,
                current_custom_state,
                self.history_path,
                match_prefix_only=False,
            )

            if shared.stop_everything:
                print(
                    f"{_HILITE}Stop signal received during LLM value update generation for '{context_path_for_marker}'.{_RESET}"
                )
                return current_value

            if not stop:
                try:
                    if expected_type == int:
                        return int(text)
                    if expected_type == float:
                        return float(text)
                    if expected_type == bool:
                        return text.strip().lower() in ["true", "yes", "1"]
                    stripped_text = ""
                    if expected_type == list or (hasattr(expected_type, "__origin__") and expected_type.__origin__ is list):
                        try:
                            stripped_text = strip_response(text)
                            parsed_list = jsonc.loads(stripped_text)
                            if isinstance(parsed_list, list):
                                return parsed_list
                            else:
                                print(
                                    f"{_ERROR}LLM response for list field {context_path_for_marker} was valid JSON but not a list: '{text}'\n\nStripped: '{stripped_text}'{_RESET}"
                                )
                                return current_value  # Fallback to original
                        except json.JSONDecodeError:
                            # Only try comma-separated for lists of strings as a fallback
                            is_list_of_str = False
                            if hasattr(expected_type, "__args__") and len(expected_type.__args__) == 1:
                                if expected_type.__args__[0] == str:
                                    is_list_of_str = True
                            if is_list_of_str:
                                return [item.strip() for item in text.split(",")]
                            print(
                                f"{_ERROR}LLM response for list field {context_path_for_marker} is not valid JSON and type is not list[str]: '{text}'\n\nStripped: '{stripped_text}'{_RESET}"
                            )
                            return current_value
                    if expected_type == dict or (hasattr(expected_type, "__origin__") and expected_type.__origin__ is dict):
                        try:
                            stripped_text = strip_response(text)
                            parsed_dict = jsonc.loads(stripped_text)
                            if isinstance(parsed_dict, dict):
                                return parsed_dict
                            else:
                                print(
                                    f"{_ERROR}LLM response for dict field {context_path_for_marker} was valid JSON but not a dict: '{text}'\n\nStripped: '{stripped_text}'{_RESET}"
                                )
                                return current_value
                        except json.JSONDecodeError:
                            print(
                                f"{_ERROR}LLM response for dict field {context_path_for_marker} is not valid JSON: '{text}'\n\nStripped: '{stripped_text}'{_RESET}"
                            )
                            return current_value
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        return text
                except (ValueError, TypeError) as e:
                    print(
                        f"{_ERROR}Could not convert LLM response '{text}' to type {expected_type} for {context_path_for_marker}: {e}{_RESET}"
                    )
                    return current_value
            else:
                return current_value
        except Exception as e:
            print(f"{_ERROR}Error in _generate_field_update for {item_name_prefix}.{field_name}: {e}{_RESET}")
            traceback.print_exc()
            return current_value

    def _update_field(
        self,
        parent_item_name_prefix: str,
        parent_data_object: dict,
        field_name: str,
        field_value: Any,
        formatted_data: FormattedData,
        parent_schema_class: ParsedSchemaClass,
        field: ParsedSchemaField,
        keys: list = [],
    ) -> Any:
        """Update a single field using the _generate_field_update method.
        Checks for overrides in parent_data_object for prompt templates.
        Determines the schema context for the prompt.

        Args:
            parent_item_name_prefix (str): Path to the parent object (e.g., "CharacterName").
            parent_data_object (dict): Actual data of the parent object.
            field_name (str): Name of the field to update (e.g., "description").
            field_value (Any): Current value of the field.
            formatted_data (FormattedData): Formatted data for LLM context.
            parent_schema_class (ParsedSchemaClass): Schema of the parent object.
            field (ParsedSchemaField): The field in question.
            keys (list, optional): Path keys to the parent object. Defaults to [].
        """
        prompt_template: str | None = self._get_effective_setting(
            parent_data_object, parent_schema_class, "update_prompt_template", field_name_context=field_name
        )

        if not prompt_template:
            if prompt_template == "":
                print(f"{_INPUT}Skipping update for {parent_item_name_prefix}.{field_name}{_RESET}")
                return field_value
            prompt_template = (
                f"Based on the most recent exchange, is the field '{{field_name}}' for item '{{item_name}}' inaccurate or incomplete?\n\n"
                f"Current value:\n```\n{{value}}\n```\n\n"
                f"Relevant Schema for '{{field_name}}':\n```json\n{{schema_snippet}}\n```\n\n"
                f"Example JSON structure for '{{field_name}}':\n```json\n{{example_json}}\n```\n\n"
                f"If yes, respond with the updated value for '{{field_name}}'.\n"
                f'If no, respond "UNCHANGED".\n'
                f'If unsure, respond "UNCHANGED".\n\n'
                f'REMEMBER: Respond *only* with the updated value or the word "UNCHANGED". Do not add explanations.'
            )
            print(
                f"{_INPUT}Using default structured prompt template with schema/example for {parent_item_name_prefix}.{field_name}{_RESET}"
            )

        updated_value = self._generate_field_update(
            item_name_prefix=parent_item_name_prefix,
            field_name=field_name,
            current_value=field_value,
            formatted_data=formatted_data,
            prompt_template_str=prompt_template,
            expected_type=field.type,
            target_schema_or_type=field.type,
            keys=keys,
        )

        print(f"{_INPUT}Updated {parent_item_name_prefix}.{field_name} from '''\n{_RESET}{field_value}{_INPUT}\n''' to '''\n{_BOLD}{updated_value}{_INPUT}\n'''{_RESET}")
        parent_data_object.update({ field_name: updated_value })

    def _create_update_prompt(
        self,
        item_name: str,
        field_name: str,
        formatted_data: FormattedData,
        prompt_template_str: str,
        target_schema_or_type: ParsedSchemaClass | type | None = None,
        entry_name: str | None = None,
        keys: list = [],
        indent: int | str | None = None,
        **kwargs: str,
    ) -> str:
        """Create a prompt for updating or generating data using a specific template string.

        Args:
            item_name (str): Name of the item/branch being processed (e.g., "CharacterName", "characters").
            field_name (str): Name of the specific field if applicable.
            formatted_data (FormattedData): Formatted data object for context.
            prompt_template_str (str): The prompt template string.
            target_schema_or_type (ParsedSchemaClass | type, optional): The schema definition or type hint relevant to what the prompt is asking to generate or update.
            entry_name (str, optional): The name of a new entry being generated (e.g. a new character's name).
            keys (list, optional): Path keys to the current data.
            indent (int | str, optional): Indentation for JSON stringification. Defaults to None.
            kwargs (str, optional): Additional keyword arguments to include when formatting.

        Returns:
            out (str): The generated prompt string.
        """
        current_value = None
        path_to_value = []
        if keys:
            path_to_value.extend(keys)
        if field_name and not field_name.startswith("{"):  # Avoid treating placeholder as key
            path_to_value.append(field_name)

        if path_to_value:
            current_value = recursive_get(formatted_data.data, path_to_value, default=None)

        value_str = ""
        if current_value is not None:
            if isinstance(current_value, (dict, list)):
                value_str = json.dumps(current_value, indent=indent)
            else:
                value_str = str(current_value)

        schema_snippet_str = ""
        example_json_str = ""
        branch_list_str = ""

        if target_schema_or_type:
            target_name = getattr(target_schema_or_type, "name", str(target_schema_or_type))
            print(f"{_INPUT}Generating schema snippet and example JSON for {target_name}{_RESET}")
            try:
                schema_snippet_str = lambda: json.dumps(
                    self.schema_parser.get_relevant_json_schema_definitions(target_schema_or_type), indent=2
                )
                example_json_str = lambda: json.dumps(self.schema_parser.generate_example_json(target_schema_or_type), indent=2)
            except Exception as e:
                print(f"{_ERROR}Error generating schema snippet or example JSON for {target_name}: {e}{_RESET}")
                traceback.print_exc()

        if formatted_data:
            branch_list_str = (
                lambda: FormattedData(formatted_data.data, f"{item_name}_list").st or "The list is empty! Maybe add some items?"
            )

        state = self.summarizer.last.state
        format_kwargs = {
            "branch_name": item_name,
            "item_name": item_name,
            "field_name": field_name,
            "value": value_str,
            "keys": keys or [],
            "schema_snippet": schema_snippet_str,
            "example_json": example_json_str,
            "branch_list": branch_list_str,
            "user_input": self.user_input,
            "output": self.output,
            "exchange": lambda: self.summarizer.format_dialogue(self.custom_state, [[self.user_input, self.output]]),
            "{user}": state["name1"],
            "{char}": state["name2"],
            **kwargs,
        }
        if entry_name is not None:
            format_kwargs["entry_name"] = entry_name

        try:
            return format_str(prompt_template_str, **format_kwargs)
        except KeyError as e:
            print(
                f"{_ERROR}Missing key in prompt template formatting: {e}. Template: '{prompt_template_str}', Args: {format_kwargs}{_RESET}"
            )
            return f"Update field '{field_name}' for item '{item_name}'. Current value: {value_str}"
