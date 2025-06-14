from pathlib import Path
import copy
import json
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
    strip_response,
)
from extensions.dayna_ss.utils.schema_parser import (
    SchemaParser,
    ParsedSchemaClass,
    ParsedSchemaField,
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
        self.schema_parser = schema_parser  # Store the schema parser instance

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
            base_setting_name (str): The name of the setting to get (e.g., "do_perform_gate_check", "update_prompt_template").
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
            # If field-specific not in defaults, and schema_attr_value (general template) exists, return it
            if schema_attr_value is not None:  # General template from attribute
                return schema_attr_value
            # If general template also not an attribute, check defaults for general template
            if base_setting_name in schema_definition.defaults:  # General template from defaults
                return schema_definition.defaults[base_setting_name]

        # 5. If not a field-specific template context, or if it was but not found:
        #    Return the schema attribute value (if any)
        if schema_attr_value is not None:
            return schema_attr_value
        #    Else, check the defaults dictionary for the base setting name.
        if base_setting_name in schema_definition.defaults:
            return schema_definition.defaults[base_setting_name]

        # 6. Fallback default (primarily for templates)
        if base_setting_name.endswith("_prompt_template"):
            return None

        # For other non-template settings not found, None is a safe default.
        # Boolean flags for actions are now handled by event_map.
        return None

    def _is_action_triggered(
        self,
        schema_definition: ParsedSchemaClass,
        action_name: str,
        current_event_triggers: list[str],
    ) -> bool:
        """Checks if a given action is triggered by any of the current event conditions."""
        if not schema_definition or not hasattr(schema_definition, "event_map"):
            return False
        for trigger in current_event_triggers:
            if action_name in schema_definition.event_map.get(trigger, []):
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
            original_data = FormattedData(data, data_type)
            formatted_data = FormattedData(data, data_type, self.schema_parser)
            if hasattr(subject_type, "__origin__") and subject_type.__origin__ is dict:
                # e.g., "characters": "dict[str, Character]"
                for name, item_data in data.items():
                    if not item_data:
                        print(f"{_ERROR}Empty data for {name}, skipping{_RESET}")
                        continue
                    updated_fields = self._update_recursive(name, item_data, formatted_data, original_data, target_schema_class)
                    if updated_fields:
                        item_data.update(updated_fields)
            else:  # e.g., "current_scene": "CurrentScene"
                updated_fields = self._update_recursive(data_type, data, formatted_data, original_data, target_schema_class)
                if updated_fields:
                    data.update(updated_fields)

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
        current_data_dict: dict,
        formatted_data: FormattedData,
        branch_schema_class: ParsedSchemaClass,
        keys: list = [],
    ):
        """Handles querying for and adding new entries to a dictionary-like branch."""
        if not (
            branch_schema_class.definition_type == "field"
            and hasattr(branch_schema_class._field.type, "__origin__")
            and branch_schema_class._field.type.__origin__ is dict
        ):
            # This logic is for dictionary fields like "Characters: dict[str, Character]"
            return

        new_query_template = branch_schema_class.new_field_query_prompt_template
        new_entry_template = branch_schema_class.new_field_entry_prompt_template

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
            schema_for_prompt_context=branch_schema_class,
            keys=keys,
        )

        query_stopping_strings = ["NO", "[]"]
        llm_query_response, stop_reason = self.summarizer.generate_using_tgwui(
            prompt=query_prompt,
            state=self.custom_state or {},
            history_path=self.history_path,
            stopping_strings=query_stopping_strings,
        )

        if shared.stop_everything:
            return
        stripped_response = strip_response(llm_query_response)
        if (stop_reason and stop_reason in query_stopping_strings) or stripped_response in query_stopping_strings:
            print(f"{_GRAY}LLM indicates no new entries for '{branch_name}'.{_RESET}")
            return

        new_entry_names = []
        try:
            parsed_names = json.loads(stripped_response)
            if parsed_names and isinstance(parsed_names, list) and all(isinstance(name, str) for name in parsed_names):
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

            # Create a temporary, minimal state for generating this single new entry
            # This helps the LLM focus on generating just this entry from scratch.
            # We pass the overall formatted_data for context, but the prompt targets the new entry.

            entry_generation_prompt = self._create_update_prompt(
                item_name=branch_name,
                field_name="",  # Not updating a sub-field of the new entry yet
                formatted_data=formatted_data,
                prompt_template_str=new_entry_template,
                schema_for_prompt_context=value_schema_class,
                entry_name=entry_name,
                keys=keys,
            )

            # For generating a new entry, we don't mark a specific field in the existing data.
            llm_entry_response, _ = self.summarizer.generate_using_tgwui(
                prompt=entry_generation_prompt,
                state=self.custom_state or {},  # Use current custom state
                history_path=self.history_path,
                # No specific stopping strings here, expect full JSON
            )
            if shared.stop_everything:
                return

            try:
                stripped_entry_json = strip_response(llm_entry_response)
                new_entry_data = json.loads(stripped_entry_json)
                if isinstance(new_entry_data, dict):
                    current_data_dict[entry_name] = new_entry_data
                    # TODO: Also update formatted_data?
                    print(f"\r{_SUCCESS}Successfully added new entry '{entry_name}' to '{branch_name}'.{_RESET}")
                else:
                    print(
                        f"{_ERROR}LLM response for new entry '{entry_name}' in '{branch_name}' was not a JSON object: {llm_entry_response}{_RESET}"
                    )
            except json.JSONDecodeError:
                print(
                    f"{_ERROR}Failed to parse LLM response for new entry '{entry_name}' in '{branch_name}' as JSON: {llm_entry_response}{_RESET}"
                )

        recursive_set(formatted_data.data, keys, current_data_dict)
        recursive_set(formatted_data.data, keys, current_data_dict)

    # --- Helper methods for parsing and applying LLM updates ---
    def _parse_llm_field_updates(self, llm_response_text: str, branch_name_for_log: str) -> list[dict[str, Any]]:
        """Parses LLM response for field updates.

        Expected LLM response formats:
        - "NO_UPDATES_REQUIRED"
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

        if response_text == "NO_UPDATES_REQUIRED":
            print(f"{_INPUT}[{branch_name_for_log}] LLM indicates no updates required.{_RESET}")
            return []

        updates: list[dict[str, Any]] = []
        try:
            # Attempt to parse as JSON list
            parsed_json = json.loads(response_text)
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
                    f"{_ERROR}[{branch_name_for_log}] LLM response parsed as JSON but is not a list of updates or a single update object: {type(parsed_json)}{_RESET}"
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
                    value = json.loads(value_str)
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

    def _should_update_subject(self, schema_class: ParsedSchemaClass) -> bool:
        if not schema_class:
            return False

        current_event_triggers = ["always"]
        if self.summarizer.last and self.summarizer.last.is_new_scene_turn:
            current_event_triggers.append("on_new_scene")

        actions_that_imply_update_check = [
            "perform_update",
            "perform_gate_check",
            "query_branch_for_changes",
            "add_new",
        ]
        for action in actions_that_imply_update_check:
            if self._is_action_triggered(schema_class, action, current_event_triggers):
                return True
        return False

    def _update_recursive(
        self,
        item_name_prefix: str,
        data: dict,
        formatted_data: FormattedData,
        original_data: FormattedData,
        target_schema_class: ParsedSchemaClass,
        parent_schema_class: ParsedSchemaClass | None = None,
        keys: list = [],
    ) -> dict:
        """Recursively update fields based on parsed schema structure.

        Handles gate checks, full branch updates, branch queries, and individual field updates based on the target_schema_class's event_map.

        Args:
            item_name_prefix (str): Prefix for the current item's name (e.g., "CharacterName.inventory").
            data (dict): The data dictionary to update.
            formatted_data (FormattedData): Formatted data for LLM context.
            target_schema_class (ParsedSchemaClass): Schema for the current data level.
            parent_schema_class (ParsedSchemaClass, optional): Schema of the parent. Defaults to None.
            keys (list, optional): Path keys to the current data level. Defaults to [].

        Returns:
            out (dict): A dictionary of fields that were updated at the current recursion level.
        """
        updated_fields = {}
        keys = keys or []
        skip_current_schema_branch_query = False
        global defaults_to_inherit

        if not isinstance(target_schema_class, ParsedSchemaClass):
            print(f"{_ERROR}Target class {target_schema_class} is not a ParsedSchemaClass{_RESET}")
            return updated_fields

        if shared.stop_everything:
            return updated_fields

        print(
            f"{_BOLD}Updating {target_schema_class.name} (type: {target_schema_class.definition_type}) at path: {item_name_prefix or target_schema_class.name}{_RESET}"
        )

        # Determine current event triggers
        current_event_triggers = ["always"]
        if self.summarizer.last and self.summarizer.last.is_new_scene_turn:
            current_event_triggers.append("on_new_scene")

        # --- START: Add New Entries to Dictionary Logic ---
        if target_schema_class.definition_type == "field" and self._is_action_triggered(
            target_schema_class, "add_new", current_event_triggers
        ):

            new_field_query_prompt_template = self._get_effective_setting(
                data, target_schema_class, "new_field_query_prompt_template"
            )
            new_field_entry_prompt_template = self._get_effective_setting(
                data, target_schema_class, "new_field_entry_prompt_template"
            )

            if (
                hasattr(target_schema_class._field.type, "__origin__")
                and target_schema_class._field.type.__origin__ is dict
                and new_field_query_prompt_template
                and new_field_entry_prompt_template
            ):
                self._detect_and_add_new_entries_to_branch(
                    branch_name=item_name_prefix,
                    current_data_dict=data,
                    formatted_data=original_data,
                    branch_schema_class=target_schema_class,
                )
        # --- END: Add New Entries to Dictionary Logic ---

        # --- START: Gate Check Logic ---
        gate_check_prompt_template = self._get_effective_setting(data, target_schema_class, "gate_check_prompt_template")
        if (
            self._is_action_triggered(target_schema_class, "perform_gate_check", current_event_triggers)
            and gate_check_prompt_template
        ):

            gate_check_branch_name = item_name_prefix or target_schema_class.name or "current data category"

            try:
                gate_check_prompt = gate_check_prompt_template.format(branch_name=gate_check_branch_name)
            except KeyError as e:
                print(
                    f"{_ERROR}Gate check prompt for '{gate_check_branch_name}' missing key: {e}. Template: '{gate_check_prompt_template}'{_RESET}"
                )
                traceback.print_exc()
                gate_check_prompt = None

            if gate_check_prompt:
                print(f"{_GRAY}Performing gate check for '{gate_check_branch_name}'...{_RESET}")
                gate_check_full_prompt = f"Current context for '{item_name_prefix}':\n{formatted_data.mark_field(item_name_prefix)}\n\n{gate_check_prompt}"
                current_custom_state = self.custom_state or {}
                stopping_strings = ["NO", "UNCHANGED"]

                llm_response_text, stop_reason = self.summarizer.generate_using_tgwui(
                    prompt=gate_check_full_prompt,
                    state=current_custom_state,
                    history_path=self.history_path,
                    stopping_strings=[*stopping_strings, "YES"],
                )
                if shared.stop_everything:
                    return updated_fields
                print(
                    f"{_GRAY}Gate check response for '{gate_check_branch_name}': '{llm_response_text}'. Stop: '{stop_reason}'{_RESET}"
                )

                if stop_reason and stop_reason in stopping_strings:
                    print(f"{_INPUT}Gate check for '{gate_check_branch_name}' returned NO. Skipping branch.{_RESET}")
                    return updated_fields
                elif llm_response_text.strip().upper() == "YES" or (stop_reason and stop_reason.upper() == "YES"):
                    print(f"{_INPUT}Gate check for '{gate_check_branch_name}' returned YES. Proceeding.{_RESET}")
                    skip_current_schema_branch_query = True
                    # NOTE: Could also persist the gate check response in custom state to track the decision, but won't for now
                else:
                    print(
                        f"{_INPUT}Gate check for '{gate_check_branch_name}' unclear. Assuming NO. Response: '{llm_response_text}'{_RESET}"
                    )
                    return updated_fields
        # --- END: Gate Check Logic ---

        # --- START: Full Branch Update (perform_update at current level) ---
        update_prompt_for_branch = self._get_effective_setting(
            data, target_schema_class, "update_prompt_template"  # General update template for the branch
        )
        if (
            self._is_action_triggered(target_schema_class, "perform_update", current_event_triggers)
            and update_prompt_for_branch
            and target_schema_class.definition_type == "dataclass"
        ):

            branch_name_for_prompt = item_name_prefix or target_schema_class.name or "current data section"
            print(
                f"{_INPUT}Attempting direct update for branch '{branch_name_for_prompt}' as 'perform_update' is triggered.{_RESET}"
            )

            try:
                # The update_prompt_template might use {item_name} or {branch_name}
                # For a full branch update, item_name and branch_name are the same.
                current_prompt_template_str = update_prompt_for_branch
                if "{branch_name}" in current_prompt_template_str and "{item_name}" not in current_prompt_template_str:
                    current_prompt_template_str = current_prompt_template_str.replace("{branch_name}", "{item_name}")

            except KeyError as e:
                print(
                    f"{_ERROR}Full update prompt for '{branch_name_for_prompt}' missing key: {e}. Template: '{update_prompt_for_branch}'{_RESET}"
                )
                current_prompt_template_str = None

            if current_prompt_template_str:
                # We expect the LLM to return the entire dictionary for 'data'
                updated_branch_data = self._get_llm_update_for_value(
                    item_name_prefix=item_name_prefix,
                    field_name="",  # Not a sub-field, but the branch itself
                    current_value=data,
                    formatted_data=formatted_data,
                    prompt_template_str=current_prompt_template_str,
                    expected_type=dict,
                    schema_for_prompt_context=target_schema_class,
                    keys=keys,
                    context_marker_path_override=item_name_prefix,
                )

                if updated_branch_data is not None and isinstance(updated_branch_data, dict) and not updated_branch_data is data:
                    print(f"{_SUCCESS}Applying direct branch update to '{branch_name_for_prompt}'.{_RESET}")
                    data.clear()
                    data.update(updated_branch_data)
                elif updated_branch_data == data or updated_branch_data is None:
                    print(f"{_GRAY}Direct branch update for '{branch_name_for_prompt}' resulted in no changes.{_RESET}")
                else:
                    print(
                        f"{_ERROR}Direct branch update for '{branch_name_for_prompt}' unexpected type: {type(updated_branch_data)}.{_RESET}"
                    )
                return updated_fields
        # --- END: Full Branch Update ---

        # --- START: Branch Query Logic (query_branch_for_changes) ---
        branch_query_prompt_template = self._get_effective_setting(data, target_schema_class, "branch_query_prompt_template")
        branch_update_prompt_template = self._get_effective_setting(data, target_schema_class, "branch_update_prompt_template")

        if (
            not skip_current_schema_branch_query
            and self._is_action_triggered(target_schema_class, "query_branch_for_changes", current_event_triggers)
            and branch_query_prompt_template
            and branch_update_prompt_template
        ):
            branch_name_for_prompt = item_name_prefix or target_schema_class.name or "current data section"

            try:
                branch_query_prompt = branch_query_prompt_template.format(branch_name=branch_name_for_prompt)
                branch_update_prompt = branch_update_prompt_template.format(branch_name=branch_name_for_prompt)
            except KeyError as e:
                print(
                    f"{_ERROR}Branch query/update prompts for '{branch_name_for_prompt}' missing key: {e}. QueryT: '{branch_query_prompt_template}', UpdateT: '{branch_update_prompt_template}'{_RESET}"
                )
                branch_query_prompt = None

            if branch_query_prompt:
                print(f"{_GRAY}Querying LLM: Does branch '{branch_name_for_prompt}' need updates?{_RESET}")
                branch_query_full_prompt = f"Current context for '{item_name_prefix}':\n{formatted_data.mark_field(item_name_prefix)}\n\n{branch_query_prompt}"
                current_custom_state = self.custom_state or {}
                query_stopping_strings = ["NO", "UNCHANGED"]
                llm_response_text, stop_reason = self.summarizer.generate_using_tgwui(
                    prompt=branch_query_full_prompt,
                    state=current_custom_state,
                    history_path=self.history_path,
                    stopping_strings=[*query_stopping_strings, "YES"],
                )
                if shared.stop_everything:
                    return updated_fields
                print(
                    f"{_GRAY}Query branch '{branch_name_for_prompt}' response: '{llm_response_text}'. Stop: '{stop_reason}'{_RESET}"
                )

                if stop_reason and stop_reason in query_stopping_strings:
                    print(f"{_INPUT}Skipping updates for branch '{branch_name_for_prompt}' (query returned NO).{_RESET}")
                    return updated_fields

                current_custom_state = copy.deepcopy(current_custom_state)
                current_custom_state["history"]["internal"].append([branch_query_full_prompt, llm_response_text])

                print(
                    f"{_INPUT}Query for '{branch_name_for_prompt}' suggests changes. Requesting field updates list...{_RESET}"
                )
                branch_update_list_full_prompt = f"Current context for '{item_name_prefix}':\n{formatted_data.mark_field(item_name_prefix)}\n\n{branch_update_prompt}"
                update_stopping_strings = ["NO_UPDATES_REQUIRED", "END_OF_UPDATES"]
                llm_update_response_text, _ = self.summarizer.generate_using_tgwui(
                    prompt=branch_update_list_full_prompt,
                    state=current_custom_state,
                    history_path=self.history_path,
                    stopping_strings=update_stopping_strings,
                )
                if shared.stop_everything:
                    return updated_fields

                parsed_updates = self._parse_llm_field_updates(llm_update_response_text, branch_name_for_prompt)
                if parsed_updates:
                    print(f"{_INPUT}Applying {len(parsed_updates)} field update(s) to '{branch_name_for_prompt}'...{_RESET}")
                    for update_item in parsed_updates:
                        path_str = update_item["path"]
                        value = update_item["value"]
                        keyList_relative_to_branch = split_keys_to_list(path_str)
                        try:
                            recursive_set(data, keyList_relative_to_branch, value)
                            print(f"{_GRAY}[{branch_name_for_prompt}] Applied update: {path_str} = {repr(value)}{_RESET}")
                        except Exception as e:
                            print(
                                f"{_ERROR}[{branch_name_for_prompt}] Failed to apply update {path_str} = {repr(value)}: {e}{_RESET}"
                            )
                            traceback.print_exc()
                else:
                    print(
                        f"{_GRAY}No specific field updates applied to '{branch_name_for_prompt}' from branch query response.{_RESET}"
                    )
                return updated_fields  # Branch querying handles the whole branch, no further recursion needed for its fields
        # --- END: Branch Query Logic ---

        fields_to_iterate = []
        if target_schema_class.definition_type == "dataclass":
            fields_to_iterate = target_schema_class.get_fields()
        elif target_schema_class.definition_type == "field" and isinstance(target_schema_class._field.type, ParsedSchemaClass):
            if target_schema_class._field.type.definition_type == "dataclass":
                print(f"{_HILITE}{target_schema_class.name}{_GRAY}: {_RESET}{updated_fields}")
                effective_child_schema = copy.copy(target_schema_class._field.type)
                for attr_name in defaults_to_inherit:
                    if getattr(effective_child_schema, attr_name, None) is None:
                        parent_template_value = getattr(target_schema_class, attr_name, None)
                        if parent_template_value is not None:
                            setattr(effective_child_schema, attr_name, parent_template_value)
                            effective_child_schema.defaults[attr_name] = parent_template_value
                updated_fields.update(
                    self._update_recursive(
                        item_name_prefix,
                        data,
                        formatted_data,
                        original_data,
                        effective_child_schema,
                        target_schema_class,
                        keys,
                    )
                )

        if not isinstance(data, dict):
            return updated_fields

        for field in fields_to_iterate:
            field_name = field.name
            field_type = field.type

            current_item_name = f"{item_name_prefix}.{field_name}"

            if field.no_update:
                print(f"{_INPUT}Skipping update for field '{current_item_name}' and its children as no_update is True.{_RESET}")
                continue

            if field_name.startswith("_"):  # Skip internal fields if any
                continue

            # Initialize missing fields based on type hint (basic version)
            if field_name not in data:
                if hasattr(field_type, "__origin__") and field_type.__origin__ is list:
                    data[field_name] = []
                elif hasattr(field_type, "__origin__") and field_type.__origin__ is dict:
                    data[field_name] = {}
                else:
                    default = field.default
                    while isinstance(field.type, ParsedSchemaClass) and field.type.definition_type == "field":
                        field: ParsedSchemaField = field.type._field
                        default = field.default or default
                    field_type = field.type
                    print(f"{_RESET}Initializing missing field '{field_name}' (type: {field_type}) {_DEBUG} {data}{_RESET}")
                    data[field_name] = (
                        default or (hasattr(field_type, "__origin__") and field_type.__origin__()) or field_type()
                    )
                continue

            field_value = data.get(field_name)

            if field_value is None:
                continue

            # --- Handle different field types based on parsed schema ---
            field_origin = getattr(field_type, "__origin__", None)
            field_args = getattr(field_type, "__args__", tuple())

            if field_origin is dict:
                # Dictionary field
                value_type = field_args[1] if len(field_args) > 1 else Any
                if isinstance(value_type, ParsedSchemaClass) and isinstance(field_value, dict):
                    # Dictionary of nested schema classes
                    updated_dict_items = {}
                    for k, v_dict in field_value.items():
                        if isinstance(v_dict, dict):
                            # Prepare child schema with inherited templates
                            effective_child_schema = copy.copy(value_type)

                            for attr_name in defaults_to_inherit:
                                if getattr(effective_child_schema, attr_name, None) is None:
                                    parent_template_value = getattr(target_schema_class, attr_name, None)
                                    if parent_template_value is not None:
                                        setattr(effective_child_schema, attr_name, parent_template_value)
                                        effective_child_schema.defaults[attr_name] = parent_template_value

                            updated_item = self._update_recursive(
                                f"{current_item_name}.{k}",
                                v_dict,
                                formatted_data,
                                original_data,
                                effective_child_schema,
                                target_schema_class,
                                keys=[*keys, field_name, k],
                            )
                            if updated_item:
                                v_dict.update(updated_item)  # Update nested dict in place
                                updated_dict_items[k] = v_dict  # Track changes if needed, though modification is in-place
                        else:
                            print(f"{_INPUT}Skipping non-dict value in dict field {current_item_name}: {k}={v_dict}{_RESET}")
                else:
                    # Regular dictionary (or dict where value is not a schema class) - update as a whole field
                    updated_value = self._update_field(
                        item_name_prefix,  # parent_item_name_prefix for _update_field
                        data,  # parent_data_object for _update_field
                        field_name,
                        field_value,
                        formatted_data,
                        target_schema_class,  # parent_schema_class for _update_field
                        field,  # field_schema for _update_field
                        keys,
                    )
                    if updated_value is not None and updated_value != field_value:  # Check for actual change
                        updated_fields[field_name] = updated_value

            elif field_origin is list:
                # List field
                item_type = field_args[0] if field_args else Any
                if isinstance(item_type, ParsedSchemaClass) and isinstance(field_value, list):
                    # List of nested schema classes
                    updated_list_items = []
                    for i, item_dict in enumerate(field_value):
                        if isinstance(item_dict, dict):
                            # Prepare child schema with inherited templates
                            effective_child_schema = copy.copy(item_type)

                            for attr_name in defaults_to_inherit:
                                if getattr(effective_child_schema, attr_name, None) is None:
                                    parent_template_value = getattr(target_schema_class, attr_name, None)
                                    if parent_template_value is not None:
                                        setattr(effective_child_schema, attr_name, parent_template_value)
                                        effective_child_schema.defaults[attr_name] = parent_template_value

                            updated_item = self._update_recursive(
                                f"{current_item_name}[{i}]",  # Pass the full path as prefix
                                item_dict,
                                formatted_data,
                                original_data,
                                effective_child_schema,
                                target_schema_class,
                                keys=[*keys, field_name, i],
                            )
                            if updated_item:
                                item_dict.update(updated_item)  # Update item dict in place
                                updated_list_items.append(item_dict)  # Track changes if needed
                        else:
                            print(
                                f"{_INPUT}Skipping non-dict item in list field {current_item_name}: index {i}={item_dict}{_RESET}"
                            )
                    # Similar to dict, in-place modification assumed sufficient.
                else:
                    # Regular list (or list where item is not a schema class) - update as a whole field
                    updated_value = self._update_field(
                        item_name_prefix,  # parent_item_name_prefix for _update_field
                        data,  # parent_data_object for _update_field
                        field_name,
                        field_value,
                        formatted_data,
                        target_schema_class,  # parent_schema_class for _update_field
                        field,  # field_schema for _update_field
                        keys,
                    )
                    if updated_value is not None and updated_value != field_value:
                        updated_fields[field_name] = updated_value

            elif isinstance(field_type, ParsedSchemaClass) and isinstance(field_value, dict):
                # Nested single schema class

                # Prepare child schema with inherited templates
                original_child_schema = field_type
                effective_child_schema = copy.deepcopy(original_child_schema)

                for attr_name in defaults_to_inherit:
                    if getattr(effective_child_schema, attr_name, None) is None:
                        parent_template_value = getattr(target_schema_class, attr_name, None)
                        if parent_template_value is not None:
                            setattr(effective_child_schema, attr_name, parent_template_value)
                            effective_child_schema.defaults[attr_name] = parent_template_value

                updated_nested_fields = self._update_recursive(
                    current_item_name,
                    field_value,
                    formatted_data,
                    original_data,
                    effective_child_schema,
                    target_schema_class,
                    keys=[*keys, field_name],
                )
                if updated_nested_fields:
                    field_value.update(updated_nested_fields)

            else:
                # Simple field (str, int, etc.)
                updated_value = self._update_field(
                    item_name_prefix,
                    data,
                    field_name,
                    field_value,
                    formatted_data,
                    target_schema_class,
                    field,
                    keys,
                )
                # Check if value actually changed to avoid unnecessary updates
                if updated_value is not None and updated_value != field_value:
                    updated_fields[field_name] = updated_value

        return updated_fields  # Return dict of fields that were actually updated at this level

    def _get_llm_update_for_value(
        self,
        item_name_prefix: str,
        field_name: str,
        current_value: Any,
        formatted_data: FormattedData,
        prompt_template_str: str,
        expected_type: type,
        schema_for_prompt_context: ParsedSchemaClass | None = None,
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
            schema_for_prompt_context (ParsedSchemaClass, optional): Schema for snippet/example.
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
                schema_for_prompt_context=schema_for_prompt_context,
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
                    if expected_type == list or (hasattr(expected_type, "__origin__") and expected_type.__origin__ is list):
                        try:
                            stripped_text = strip_response(text)
                            parsed_list = json.loads(stripped_text)
                            if isinstance(parsed_list, list):
                                return parsed_list
                            else:
                                print(
                                    f"{_ERROR}LLM response for list field {context_path_for_marker} was valid JSON but not a list: '{text}'{_RESET}"
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
                                f"{_ERROR}LLM response for list field {context_path_for_marker} is not valid JSON and type is not list[str]: '{text}'{_RESET}"
                            )
                            return current_value
                    if expected_type == dict or (hasattr(expected_type, "__origin__") and expected_type.__origin__ is dict):
                        try:
                            stripped_text = strip_response(text)
                            parsed_dict = json.loads(stripped_text)
                            if isinstance(parsed_dict, dict):
                                return parsed_dict
                            else:
                                print(
                                    f"{_ERROR}LLM response for dict field {context_path_for_marker} was valid JSON but not a dict: '{text}'{_RESET}"
                                )
                                return current_value
                        except json.JSONDecodeError:
                            print(
                                f"{_ERROR}LLM response for dict field {context_path_for_marker} is not valid JSON: '{text}'{_RESET}"
                            )
                            return current_value
                    return text
                except (ValueError, TypeError) as e:
                    print(
                        f"{_ERROR}Could not convert LLM response '{text}' to type {expected_type} for {context_path_for_marker}: {e}{_RESET}"
                    )
                    return current_value
            else:
                return current_value
        except Exception as e:
            print(f"{_ERROR}Error in _get_llm_update_for_value for {item_name_prefix}.{field_name}: {e}{_RESET}")
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
        field_schema: ParsedSchemaField,
        keys: list = [],
    ) -> Any:
        """Update a single field using the _get_llm_update_for_value method.
        Checks for overrides in parent_data_object for prompt templates.
        Determines the schema context for the prompt.

        Args:
            parent_item_name_prefix (str): Path to the parent object (e.g., "CharacterName").
            parent_data_object (dict): Actual data of the parent object.
            field_name (str): Name of the field to update (e.g., "description").
            field_value (Any): Current value of the field.
            formatted_data (FormattedData): Formatted data for LLM context.
            parent_schema_class (ParsedSchemaClass): Schema of the parent object.
            field_schema (ParsedSchemaField): Schema of the field itself.
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
                f"Relevant Schema for '{{field_name}}':\n```json\n{{schema_snippet}}\n```\n"
                f"Example JSON structure for '{{field_name}}':\n```json\n{{example_json}}\n```\n"
                f"If yes, respond with the updated value for '{{field_name}}'.\n"
                f'If no, respond "unchanged".\n'
                f'If unsure, respond "unchanged".\n\n'
                f'REMEMBER: Respond *only* with the updated value or the word "unchanged". Do not add explanations.'
            )
            print(
                f"{_INPUT}Using default structured prompt template with schema/example for {parent_item_name_prefix}.{field_name}{_RESET}"
            )

        schema_for_context = None
        if isinstance(field_schema.type, ParsedSchemaClass):
            schema_for_context = field_schema.type

        return self._get_llm_update_for_value(
            item_name_prefix=parent_item_name_prefix,
            field_name=field_name,
            current_value=field_value,
            formatted_data=formatted_data,
            prompt_template_str=prompt_template,
            expected_type=field_schema.type,
            schema_for_prompt_context=schema_for_context,
            keys=keys,
        )

    def _create_update_prompt(
        self,
        item_name: str,
        field_name: str,
        formatted_data: FormattedData,
        prompt_template_str: str,
        schema_for_prompt_context: ParsedSchemaClass | None = None,
        entry_name: str | None = None,
        keys: list = [],
        indent: int | str | None = None,
    ) -> str:
        """Create a prompt for updating or generating data using a specific template string.

        Args:
            item_name (str): Name of the item/branch being processed (e.g., "CharacterName", "characters").
            field_name (str): Name of the specific field if applicable.
            formatted_data (FormattedData): Formatted data object for context.
            prompt_template_str (str): The prompt template string.
            schema_for_prompt_context (ParsedSchemaClass, optional): The schema definition relevant
                to what the prompt is asking to generate or update (e.g., schema for 'Character' if
                generating a new character, or schema for 'Character.description').
            entry_name (str, optional): The name of a new entry being generated (e.g. a new character's name).
            keys (list, optional): Path keys to the current data.
            indent (int | str, optional): Indentation for JSON stringification. Defaults to None.

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

        if schema_for_prompt_context:
            try:
                # Ensure all_definitions_map is correctly passed or accessed if needed by these methods
                # Assuming self.schema_parser.definitions is the comprehensive map
                schema_snippet_str = json.dumps(
                    self.schema_parser.get_relevant_definitions_json(schema_for_prompt_context.name), indent=2
                )
                example_json_str = json.dumps(
                    schema_for_prompt_context.generate_example_json(self.schema_parser.definitions), indent=2
                )
            except Exception as e:
                print(
                    f"{_ERROR}Error generating schema snippet or example JSON for {schema_for_prompt_context.name}: {e}{_RESET}"
                )
                # Keep them as empty strings if generation fails

        if formatted_data:
            branch_list_str = FormattedData(formatted_data.data, f"{item_name}_list").st or "The list is empty! Maybe add some items?"

        format_kwargs = {
            "branch_name": item_name,  # 'item_name' often serves as branch_name in templates
            "item_name": item_name,  # For field-level prompts
            "field_name": field_name,
            "value": value_str,
            "keys": keys or [],
            "schema_snippet": schema_snippet_str,
            "example_json": example_json_str,
            "branch_list": branch_list_str,
            "user_input": self.user_input,
            "output": self.output,
            "exchange": (
                self.summarizer.format_dialogue(self.custom_state, [[self.user_input, self.output]])
                if "{exchange}" in prompt_template_str
                else ""
            ),
        }
        if entry_name is not None:
            format_kwargs["entry_name"] = entry_name

        try:
            # Find all placeholders in the template string
            placeholders = re.findall(r"\{(\w+)\}", prompt_template_str)

            final_format_kwargs = {k: v for k, v in format_kwargs.items() if k in placeholders}

            # Ensure all placeholders found are in format_kwargs, or provide a default/error
            for ph in placeholders:
                if ph not in final_format_kwargs:
                    print(
                        f"{_INPUT}Placeholder '{{{ph}}}' in template but not in provided format arguments. Using empty string.{_RESET}"
                    )
                    final_format_kwargs[ph] = ""

            return prompt_template_str.format(**final_format_kwargs)
        except KeyError as e:
            print(
                f"{_ERROR}Missing key in prompt template formatting: {e}. Template: '{prompt_template_str}', Args: {final_format_kwargs}{_RESET}"
            )
            return f"Update field '{field_name}' for item '{item_name}'. Current value: {value_str}"
