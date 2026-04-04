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
    load_json,
    save_json,
    split_keys_to_list,
    recursive_set,
    recursive_get,
    expand_lists_in_data_for_llm,
    unexpand_lists_in_data_from_llm,
    strip_thinking,
    strip_response,
    format_str,
    render_jinja_template,
    format_str_or_jinja,
)
from extensions.dayna_ss.utils.schema_parser import (
    SchemaParser,
    ParsedSchemaClass,
    ParsedSchemaField,
    Action,
    Trigger,
)
from extensions.dayna_ss.agents.summarizer import Summarizer, FormattedData
from extensions.dayna_ss.ui import PhaseManager


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
        phase_manager: "PhaseManager",
    ):
        """Initialize DataSummarizer.

        Args:
            summarizer (Summarizer): The main Summarizer instance.
            exchange (tuple[str, str]): (user_input, output) pair being summarized.
            custom_state (dict): Custom state dict for generation.
            history_path (Path): Path to the current history data.
            schema_parser (SchemaParser): Parser for data schemas.
            all_subjects_data (dict): Data for all subjects to be summarized.
            phase_manager (PhaseManager): PhaseManager instance for tracking progress.
        """
        self.summarizer = summarizer
        self._phase_manager = phase_manager
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
        override_config: dict | None = None,
    ):
        """
        Gets an effective setting value, primarily for prompt templates.

        Priority:
        1. Override config from trigger entry (highest priority).
        2. Direct override for base_setting_name in data._overrides.
        3. For templates, field-specific override in data._overrides (e.g., description_prompt_template).
        4. Schema attribute on schema_definition (e.g., schema_definition.gate_check_prompt_template).
        5. For templates, field-specific template from schema_definition.defaults.
        6. General base_setting_name from schema_definition.defaults (if not already an attribute) or schema attribute.
        7. Fallback to None for templates.

        Args:
            data (dict): The actual data dictionary that might contain _overrides.
            schema_definition (ParsedSchemaClass): The schema definition for this data.
            base_setting_name (str): The name of the setting to get (e.g., "update_prompt_template").
            field_name_context (str, optional): For field-specific templates, e.g., "description". Defaults to None.
            override_config (dict, optional): Action-specific overrides from trigger entry. Defaults to None.
        """

        # 1. Check override_config
        if override_config and base_setting_name in override_config:
            return override_config[base_setting_name]

        overrides = data.get("_overrides", {})

        # 2. Direct override for the base_setting_name
        if base_setting_name in overrides:
            return overrides[base_setting_name]

        # 3. For prompt templates, check for field-specific override
        if field_name_context and base_setting_name.endswith("_prompt_template"):
            field_specific_override_key = f"{field_name_context}_prompt_template"
            if field_specific_override_key in overrides:
                return overrides[field_specific_override_key]

        # 4. Get from schema_definition attributes (e.g., schema_definition.gate_check_prompt_template)
        schema_attr_value = getattr(schema_definition, base_setting_name, None)

        # 5. For prompt templates, field-specific template from schema_definition.defaults
        if field_name_context and base_setting_name.endswith("_prompt_template"):
            field_specific_schema_key = f"{field_name_context}_prompt_template"
            if field_specific_schema_key in schema_definition.defaults:
                return schema_definition.defaults[field_specific_schema_key]
            if schema_attr_value is not None:
                return schema_attr_value
            if base_setting_name in schema_definition.defaults:
                return schema_definition.defaults[base_setting_name]

        # 6. If not a field-specific template context, or if it was but not found:
        if schema_attr_value is not None:
            return schema_attr_value
        if base_setting_name in schema_definition.defaults:
            return schema_definition.defaults[base_setting_name]

        # 7. Fallback default (primarily for templates)
        if base_setting_name.endswith("_prompt_template"):
            return None

        # Boolean flags for actions are now handled by trigger_map.
        return None

    def _get_current_event_triggers(self, schema_class: ParsedSchemaClass | None = None) -> list[Trigger]:
        """Get the current event triggers based on the last turn, which can be used to determine whether an action is triggered.
        """
        current_event_triggers = [Trigger.ALWAYS]
        if self.summarizer.last and self.summarizer.last.is_new_scene_turn:
            current_event_triggers.append(Trigger.ON_NEW_SCENE)
        else:
            current_event_triggers.append(Trigger.ON_EXISTING_SCENE)

        return current_event_triggers

    def _get_triggered_configs(
        self,
        schema_class: ParsedSchemaClass,
        action: Action,
        event_triggers: list[Trigger] = [],
    ) -> list[tuple[Action, dict | None]]:
        """Get all matching trigger configs for a given action.

        Returns a list of (action, override_config) tuples where override_config
        may contain action-specific overrides like 'prompt_template'.
        """
        matching = []
        if not schema_class or not schema_class.trigger_map:
            return matching
        for trigger in event_triggers or self._get_current_event_triggers(schema_class):
            for trigger_action, config in schema_class.trigger_map.get(trigger, []):
                if trigger_action == action:
                    matching.append((trigger_action, config))
        return matching

    def _is_action_triggered(
        self,
        schema_class: ParsedSchemaClass,
        action: Action,
        event_triggers: list[Trigger] = [],
    ) -> bool:
        """Whether a given action is triggered by any of the current event conditions. Use `_should_update_subject` for a general check."""
        return len(self._get_triggered_configs(schema_class, action, event_triggers)) > 0

    def _execute_action(
        self,
        action: Action,
        config: dict | None,
        branch_name: str,
        data: dict,
        formatted_data: FormattedData,
        unexpanded_formatted_data: FormattedData,
        target_schema_class: ParsedSchemaClass,
        keys: list,
    ) -> tuple[bool, bool]:
        """
        Execute a single configured action for a schema branch and apply its effects to the provided data.
        
        Parameters:
            action (Action): The action to perform.
            config (dict | None): Optional per-action configuration overrides.
            branch_name (str): Logical name of the branch being processed.
            data (dict): The subject data object to modify in-place.
            formatted_data (FormattedData): Schema-aware view of `data` used for LLM prompts and context.
            unexpanded_formatted_data (FormattedData): FormattedData instance without schema expansions, used for some prompts.
            target_schema_class (ParsedSchemaClass): Schema class describing the branch being acted on.
            keys (list): Path keys that locate the branch within `formatted_data`/`data`.
        
        Returns:
            tuple[bool, bool]: (stop_processing, gate_failed)
                - `stop_processing`: `True` if this action completed a full update and subsequent actions should be skipped.
                - `gate_failed`: `True` if a gate check failed and the branch should be skipped.
        """
        if shared.stop_everything:
            return (True, False)

        pm = self._phase_manager
        phase_id = branch_name.lower().replace(" ", "_")
        action_name = action.name.lower()

        if action == Action.ADD_NEW:
            pm.start_step(phase_id, action_name, f"Checking for new {branch_name} entries", {"branch_name": branch_name})
            if target_schema_class.definition_type == "alias":
                field_type = target_schema_class._field.type
                is_dict = hasattr(field_type, "__origin__") and field_type.__origin__ is dict
                if is_dict:
                    new_query_template = config.get("new_entry_query_prompt_template") if config else None
                    new_entry_template = config.get("new_entry_prompt_template") if config else None
                    self._detect_and_add_new_entries_to_branch(
                        branch_name=branch_name,
                        data=data,
                        formatted_data=unexpanded_formatted_data,
                        branch_schema_class=target_schema_class,
                        new_query_template=new_query_template,
                        new_entry_template=new_entry_template,
                        keys=keys
                    )
            pm.done_step(phase_id, action_name)

        elif action == Action.PERFORM_GATE_CHECK:
            pm.start_step(phase_id, action_name, f"Gate check for {branch_name}", {"branch_name": branch_name})
            gate_template = self._get_effective_setting(
                data, target_schema_class, "gate_check_prompt_template", override_config=config
            )
            if gate_template:
                if not self._perform_gate_check(
                    branch_name, gate_template, target_schema_class, formatted_data, keys
                ):
                    return (False, True)  # Gate check failed
            else:
                pm.done_step(phase_id, action_name, "Gate: No template")

        elif action == Action.PERFORM_UPDATE:
            pm.start_step(phase_id, action_name, f"Updating {branch_name}", {"branch_name": branch_name})
            update_template = self._get_effective_setting(
                data, target_schema_class, "update_prompt_template", override_config=config
            )
            if update_template:
                if self._perform_full_branch_update(
                    branch_name, data, update_template, formatted_data, target_schema_class, keys
                ):
                    # done_step already called by generate_with_sse via step_update with complete:true
                    return (True, False)  # Full update succeeded, stop processing
            pm.done_step(phase_id, action_name, "Update: Skipped")

        elif action == Action.QUERY_BRANCH_FOR_CHANGES:
            pm.start_step(phase_id, action_name, f"Querying {branch_name} for changes", {"branch_name": branch_name})
            update_template_key = "branch_update_prompt_template"
            if config and "prompt_template" in config:
                update_template_key = config["prompt_template"]

            # Derive query template name from update template name
            if "_update" in update_template_key:
                query_template_key = update_template_key.replace("_update", "_query")
            else:
                query_template_key = "branch_query_prompt_template"

            bq_template = self._get_effective_setting(
                data, target_schema_class, query_template_key, override_config=config
            )
            bu_template = self._get_effective_setting(
                data, target_schema_class, update_template_key, override_config=config
            )
            if bq_template and bu_template:
                self._perform_branch_query(
                    branch_name, data, formatted_data, bq_template, bu_template, target_schema_class, keys
                )
            pm.done_step(phase_id, action_name)

        return (False, False)  # Continue processing

    def _should_update_subject(self, schema_class: ParsedSchemaClass, event_triggers: list[Trigger] = []) -> bool:
        """Whether the subject should be updated at all based on the schema_class's trigger_map. Use `_is_action_triggered` for more fine-grained control."""
        if not schema_class or not schema_class.trigger_map:
            return False
        for trigger in event_triggers or self._get_current_event_triggers(schema_class):
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
                    history_internal = self.custom_state.get("history", {}).get("internal", [])
                    current_message_node = f"{len(history_internal) * 2}_1_1"
                    if "start" in new_entry_data:
                        new_entry_data["start"]["_message_node"] = current_message_node
                    print(f"{_DEBUG}Auto-set '_message_node' to '{current_message_node}' for new entry '{entry_name}'.{_RESET}")

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

    def check_and_archive_chapter(self):
        """Check if the current chapter should be archived and create a new one if needed.

        This is called after scene archival to determine if a chapter boundary
        should be crossed based on LLM-driven decision making.

        Can be forced via summarizer.last.force_next_chapter = True
        """
        if shared.stop_everything:
            return

        events_data = self.all_subjects_data.get("events", {})
        if not events_data:
            print(f"{_GRAY}No events data available. Skipping chapter check.{_RESET}")
            return

        # Check for forced chapter transition
        force_chapter = self.summarizer.last and self.summarizer.last.force_next_chapter
        if force_chapter:
            print(f"{_BOLD}Force next chapter requested. Skipping LLM check.{_RESET}")

        # Get current chapter info
        current_scene_data = self.all_subjects_data.get("current_scene", {})
        current_chapter_number = current_scene_data.get("_chapter_number", 1)

        # Count scenes in current chapter
        scenes = events_data.get("scenes", {})
        scene_names = list(scenes.keys())
        total_scenes = len(scene_names)
        scenes_in_chapter = total_scenes  # For simplicity's sake

        # Get chapter configuration from schema
        chapters_schema = self.schema_parser.get_subject_class("chapters")
        if not chapters_schema:
            chapters_schema = self.schema_parser.definitions.get("Chapters")

        suggested_min = 4
        suggested_max = 8
        max_hard = 10
        gate_check_prompt = None

        if chapters_schema:
            defaults = chapters_schema.defaults if hasattr(chapters_schema, 'defaults') else {}
            suggested_min = defaults.get("suggested_min_scenes", 4)
            suggested_max = defaults.get("suggested_max_scenes", 8)
            max_hard = defaults.get("max_scenes_before_required", 10)
            gate_check_prompt = chapters_schema.gate_check_prompt_template

        if not force_chapter and scenes_in_chapter < suggested_min:
            print(f"{_GRAY}Chapter has {scenes_in_chapter} scenes (min suggested: {suggested_min}). Skipping chapter check.{_RESET}")
            return

        if force_chapter:
            should_archive = True
            print(f"{_BOLD}Forcing chapter transition.{_RESET}")
        elif gate_check_prompt:
            # Use the schema template
            scene_in_chapter = scenes_in_chapter
            chapters_count = len(events_data.get("chapters", {}))

            gate_prompt = render_jinja_template(
                gate_check_prompt,
                scenes_count=total_scenes,
                chapters_count=chapters_count,
                current_scene_number=current_scene_data.get("_scene_number", 1),
                current_chapter_number=current_chapter_number,
                current_arc_number=current_scene_data.get("_arc_number", 1),
                scenes_in_chapter=scenes_in_chapter,
                scene_in_chapter=scene_in_chapter,
                chapters_in_arc=chapters_count,
                chapter_in_arc=current_chapter_number,
                suggested_min=suggested_min,
                suggested_max=suggested_max,
                max_scenes_before_required=max_hard,
            )

            if shared.stop_everything:
                return

            print(f"{_INPUT}Checking if chapter should be archived (scenes: {scenes_in_chapter})...{_RESET}")

            current_custom_state = self.custom_state or {}
            llm_response, stop_reason = self.summarizer.generate_using_tgwui(
                prompt=gate_prompt,
                state=current_custom_state,
                history_path=self.history_path,
                stopping_strings=["YES", "NO"],
                match_prefix_only=True,
            )

            if shared.stop_everything:
                return

            response_upper = strip_response(llm_response).upper().strip()
            should_archive = (
                "YES" in response_upper
                or scenes_in_chapter >= max_hard
                or (stop_reason and "YES" in stop_reason.upper())
            )

            if not should_archive:
                print(f"{_GRAY}Chapter continues (response: {llm_response[:50]}...).{_RESET}")
                return
        else:
            print(f"{_ERROR}No gate_check_prompt_template for chapters. Skipping.{_RESET}")
            return

        if force_chapter and self.summarizer.last:
            self.summarizer.last.force_next_chapter = False

        recent_scene_name = scene_names[-1] if scene_names else "Unknown"
        recent_scene_summary = scenes[recent_scene_name].get("summary", "No summary") if recent_scene_name in scenes else ""

        print(f"{_SUCCESS}Archiving chapter {current_chapter_number} and creating new one.{_RESET}")

        # Generate chapter data
        chapter_title = f"Chapter {current_chapter_number}"
        chapter_generation_prompt = f"""Based on all the scenes in the current chapter, generate the full data for the chapter named '{chapter_title}'.

Recent Scene: "{recent_scene_name}"
Summary: {recent_scene_summary}

Characters involved:
{json.dumps(current_scene_data.get("now", {}).get("who", {}).get("characters", []), indent=2) if current_scene_data.get("now", {}).get("who", {}).get("characters") else "No characters available"}

Generate chapter data that includes:
- A title (you can rename from "Chapter {current_chapter_number}" to something more evocative)
- A concise summary of what happened in this chapter
- Key changes that occurred, each with a description and the scene where it occurred

Schema for Chapter:
{{
  "title": "str",
  "starting_scene": "int (scene index where this chapter began, starting from 1)",
  "ending_scene": "int (scene index where this chapter concluded)",
  "scenes": "list[int] (list of scene indices in this chapter)",
  "summary": "str",
  "key_changes": [
    {{
      "description": "str (what changed)",
      "scene": "str (scene name where this occurred)"
    }}
  ],
  "status": "str ('active', 'concluded', or 'suspended')"
}}

Respond with ONLY the JSON object for this chapter."""

        chapter_response, _ = self.summarizer.generate_using_tgwui(
            prompt=chapter_generation_prompt,
            state=current_custom_state,
            history_path=self.history_path,
        )

        if shared.stop_everything:
            return

        try:
            chapter_data = jsonc.loads(strip_response(chapter_response))
            if isinstance(chapter_data, dict):
                # Ensure required fields
                if "title" not in chapter_data:
                    chapter_data["title"] = chapter_title
                if "status" not in chapter_data:
                    chapter_data["status"] = "concluded"
                if "starting_scene" not in chapter_data:
                    chapter_data["starting_scene"] = 1
                if "ending_scene" not in chapter_data:
                    chapter_data["ending_scene"] = total_scenes
                if "scenes" not in chapter_data:
                    chapter_data["scenes"] = list(range(1, total_scenes + 1))
                if "key_changes" not in chapter_data:
                    chapter_data["key_changes"] = []

                # Add chapter to events (dict format, keyed by title)
                if "chapters" not in events_data:
                    events_data["chapters"] = {}

                events_data["chapters"][chapter_data["title"]] = chapter_data

                new_chapter_number = current_chapter_number + 1
                current_scene_data["_chapter_number"] = new_chapter_number
                print(f"{_SUCCESS}Archived chapter {current_chapter_number} and set new chapter number to {new_chapter_number}.{_RESET}")

        except json.JSONDecodeError as e:
            print(f"{_ERROR}Failed to parse chapter data: {e}. Chapter response: {chapter_response[:200]}...{_RESET}")

    def check_and_archive_arc(self):
        """Check if the current arc should be archived and create a new one if needed.

        This is called after chapter archival to determine if an arc boundary
        should be crossed based on LLM-driven decision making.

        Can be forced via summarizer.last.force_next_arc = True
        """
        if shared.stop_everything:
            return

        events_data = self.all_subjects_data.get("events", {})
        if not events_data:
            print(f"{_GRAY}No events data available. Skipping arc check.{_RESET}")
            return

        # Check for forced arc transition
        force_arc = self.summarizer.last and self.summarizer.last.force_next_arc
        if force_arc:
            print(f"{_BOLD}Force next arc requested. Skipping LLM check.{_RESET}")

        # Get current arc info
        current_scene_data = self.all_subjects_data.get("current_scene", {})
        current_chapter_number = current_scene_data.get("_chapter_number", 1)
        current_arc_number = current_scene_data.get("_arc_number", 1)

        # Load arcs data
        arcs_path = self.history_path / "arcs.json"
        arcs_data = load_json(arcs_path) or []

        # Count chapters in current arc
        chapters_dict = events_data.get("chapters", {})
        total_chapters = len(chapters_dict) if chapters_dict else 0
        chapters_in_arc = total_chapters  # TODO: Calculate

        # Get arc configuration from schema
        arcs_schema = self.schema_parser.get_subject_class("arcs")
        if not arcs_schema:
            arcs_schema = self.schema_parser.definitions.get("Arcs")

        suggested_min = 3
        suggested_max = 6
        max_hard = 12
        gate_check_prompt = None

        if arcs_schema:
            defaults = arcs_schema.defaults if hasattr(arcs_schema, 'defaults') else {}
            suggested_min = defaults.get("suggested_min_chapters", 3)
            suggested_max = defaults.get("suggested_max_chapters", 6)
            max_hard = defaults.get("max_chapters_before_required", 12)
            gate_check_prompt = arcs_schema.gate_check_prompt_template

        if not force_arc and chapters_in_arc < suggested_min:
            print(f"{_GRAY}Arc has {chapters_in_arc} chapters (min suggested: {suggested_min}). Skipping arc check.{_RESET}")
            return

        if force_arc:
            should_archive = True
            print(f"{_BOLD}Forcing arc transition.{_RESET}")
        elif gate_check_prompt:
            # Use the schema template
            chapter_in_arc = chapters_in_arc
            scenes_count = len(events_data.get("scenes", {}))

            gate_prompt = render_jinja_template(
                gate_check_prompt,
                scenes_count=scenes_count,
                chapters_count=total_chapters,
                current_scene_number=current_scene_data.get("_scene_number", 1),
                current_chapter_number=current_chapter_number,
                current_arc_number=current_arc_number,
                scenes_in_chapter=scenes_count,
                scene_in_chapter=current_scene_data.get("_scene_number", 1),
                chapters_in_arc=chapters_in_arc,
                chapter_in_arc=chapter_in_arc,
                suggested_min=suggested_min,
                suggested_max=suggested_max,
                max_chapters_before_required=max_hard,
            )

            if shared.stop_everything:
                return

            print(f"{_INPUT}Checking if arc should be archived (chapters: {chapters_in_arc})...{_RESET}")

            current_custom_state = self.custom_state or {}
            llm_response, stop_reason = self.summarizer.generate_using_tgwui(
                prompt=gate_prompt,
                state=current_custom_state,
                history_path=self.history_path,
                stopping_strings=["YES", "NO"],
                match_prefix_only=True,
            )

            if shared.stop_everything:
                return

            response_upper = strip_response(llm_response).upper().strip()
            should_archive = (
                "YES" in response_upper
                or chapters_in_arc >= max_hard
                or (stop_reason and "YES" in stop_reason.upper())
            )

            if not should_archive:
                print(f"{_GRAY}Arc continues (response: {llm_response[:50]}...).{_RESET}")
                return
        else:
            print(f"{_ERROR}No gate_check_prompt_template for arcs. Skipping.{_RESET}")
            return

        if force_arc and self.summarizer.last:
            self.summarizer.last.force_next_arc = False

        recent_chapter_name = list(chapters_dict.keys())[-1] if chapters_dict else "Unknown"
        recent_chapter_summary = chapters_dict[recent_chapter_name].get("summary", "No summary") if recent_chapter_name in chapters_dict else ""

        print(f"{_SUCCESS}Archiving arc {current_arc_number} and creating new one.{_RESET}")

        # Generate arc data
        arc_title = f"Arc {current_arc_number}"
        arc_generation_prompt = f"""Based on all the chapters in the current arc, generate the full data for the arc named '{arc_title}'.

Recent Chapter: "{recent_chapter_name}"
Summary: {recent_chapter_summary}

Chapters in this arc:
{json.dumps(list(chapters_dict.values()), indent=2) if chapters_dict else "No chapters available"}

Generate arc data that includes:
- A title (you can rename from "Arc {current_arc_number}" to something more evocative)
- A concise summary of what happened in this arc
- Character arc progressions and key relationship shifts
- Plot threads introduced or resolved
- Status: 'active', 'concluded', or 'suspended'

Schema for Arc:
{{
  "title": "str",
  "starting_chapter": "int (chapter index where this arc began, starting from 1)",
  "ending_chapter": "int (chapter index where this arc concluded)",
  "chapters": "list[int] (list of chapter indices in this arc)",
  "summary": "str",
  "character_arcs": "str (how characters changed)",
  "relationship_shifts": "str (how relationships evolved)",
  "plot_threads": ["str (plot threads in this arc)"],
  "status": "str ('active', 'concluded', or 'suspended')"
}}

Respond with ONLY the JSON object for this arc."""

        arc_response, _ = self.summarizer.generate_using_tgwui(
            prompt=arc_generation_prompt,
            state=current_custom_state,
            history_path=self.history_path,
        )

        if shared.stop_everything:
            return

        try:
            arc_data = jsonc.loads(strip_response(arc_response))
            if isinstance(arc_data, dict):
                # Ensure required fields
                if "title" not in arc_data:
                    arc_data["title"] = arc_title
                if "status" not in arc_data:
                    arc_data["status"] = "concluded"
                if "starting_chapter" not in arc_data:
                    arc_data["starting_chapter"] = 1
                if "ending_chapter" not in arc_data:
                    arc_data["ending_chapter"] = total_chapters
                if "chapters" not in arc_data:
                    arc_data["chapters"] = list(range(1, total_chapters + 1))
                if "character_arcs" not in arc_data:
                    arc_data["character_arcs"] = ""
                if "relationship_shifts" not in arc_data:
                    arc_data["relationship_shifts"] = ""
                if "plot_threads" not in arc_data:
                    arc_data["plot_threads"] = []
                if "summary" not in arc_data:
                    arc_data["summary"] = recent_chapter_summary

                # Add arc to list
                arcs_data.append(arc_data)
                save_json(arcs_data, arcs_path)

                new_arc_number = current_arc_number + 1
                current_scene_data["_arc_number"] = new_arc_number
                print(f"{_SUCCESS}Archived arc {current_arc_number} and set new arc number to {new_arc_number}.{_RESET}")

        except json.JSONDecodeError as e:
            print(f"{_ERROR}Failed to parse arc data: {e}. Arc response: {arc_response[:200]}...{_RESET}")

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

            stop_processing, gate_failed = self._execute_action(
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
        Traverse and update `data` according to `schema_class`, recursing into nested schema classes, lists, and dicts.
        
        This mutates `data` and `formatted_data` in place: it descends dataclass fields, unwraps alias/field wrappers, recurses into nested ParsedSchemaClass instances, and applies per-item or per-value updates for lists and dicts (using `_update_recursive` and `_update_field`). For list/dict elements that are schema classes this starts and completes sub-phases via the phase manager to track progress.
        
        Parameters:
            item_name_prefix (str): Human-readable path prefix used for prompts and phase ids (e.g., "chapter.scenes" or "characters[0]").
            data (dict | list): The current branch of data to traverse and potentially update.
            formatted_data (FormattedData): Formatted view of the data used to render prompts and to update contextual markers; this will be kept in sync with changes.
            unexpanded_formatted_data (FormattedData): Unexpanded formatted view used when creating prompts that require original/unexpanded values.
            schema_class (ParsedSchemaClass): Schema description that determines traversal behavior (dataclass, alias/field, wrapped list/dict, or nested schema).
            keys (list): List of keys/indices representing the path within `formatted_data.data` corresponding to `data`.
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
                        pm = self._phase_manager
                        for i, item_data in enumerate(data):
                            effective_item_schema = self._inherit_defaults_from_parent(
                                item_type,
                                schema_class,
                                defaults_to_inherit
                            )
                            # Sync formatted data to keep context fresh
                            new_keys = [*keys, i]
                            sub_phase_id = f"{item_name_prefix}[{i}]".lower().replace(" ", "_")
                            pm.start_phase(sub_phase_id, f"{item_name_prefix}[{i}]")
                            try:
                                self._update_recursive(
                                    f"{item_name_prefix}[{i}]",
                                    item_data,
                                    formatted_data,
                                    unexpanded_formatted_data,
                                    effective_item_schema,
                                    new_keys
                                )
                                recursive_set(formatted_data.data, new_keys, item_data)
                            finally:
                                pm.done_phase(sub_phase_id)

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
                        pm = self._phase_manager
                        for key, val_data in data.items():
                            effective_val_schema = self._inherit_defaults_from_parent(
                                value_type,
                                schema_class,
                                defaults_to_inherit
                            )

                            sub_phase_id = f"{item_name_prefix}.{key}".lower().replace(" ", "_")
                            pm.start_phase(sub_phase_id, f"{item_name_prefix}.{key}")

                            new_keys = [*keys, key]
                            try:
                                self._update_recursive(
                                    f"{item_name_prefix}.{key}",
                                    val_data,
                                    formatted_data,
                                    unexpanded_formatted_data,
                                    effective_val_schema,
                                    new_keys
                                )
                                recursive_set(formatted_data.data, new_keys, val_data)
                            finally:
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

            sub_phase_id = full_path.lower().replace(" ", "_")
            pm = self._phase_manager
            pm.start_phase(sub_phase_id, full_path)
            try:
                self._update_recursive(
                    full_path,
                    current_value,
                    formatted_data,
                    unexpanded_formatted_data,
                    effective_child_schema,
                    current_keys
                )
                recursive_set(formatted_data.data, current_keys, current_value)
            finally:
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
                    try:
                        self._update_recursive(
                            f"{full_path}[{i}]",
                            item,
                            formatted_data,
                            unexpanded_formatted_data,
                            effective_schema,
                            [*current_keys, i]
                        )
                        recursive_set(formatted_data.data, [*current_keys, i], item)
                    finally:
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
                    try:
                        self._update_recursive(
                            f"{full_path}.{k}",
                            v,
                            formatted_data,
                            unexpanded_formatted_data,
                            effective_schema,
                            [*current_keys, k]
                        )
                        recursive_set(formatted_data.data, [*current_keys, k], v)
                    finally:
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
        Decides whether a branch should be processed by asking the LLM a gate-check question.
        
        If the rendered gate-check prompt is empty or missing, this function defaults to allowing processing. It sends the prompt to the LLM and treats a stop reason of `"NO"` or `"UNCHANGED"`, a response that begins with `"NO"`, or any response containing `"UNCHANGED"` as a decision to skip the branch; all other responses are treated as a decision to process the branch.
        
        Returns:
            `true` if the branch should be processed, `false` otherwise.
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

        pm = self._phase_manager
        phase_id = branch_name.lower().replace(" ", "_")

        # Contextualize the prompt
        gate_check_full_prompt = (
            f"Current context for '{branch_name}':\n"
            f"{formatted_data.mark_field(branch_name)}\n\n"
            f"{gate_check_prompt}"
        )

        current_custom_state = self.custom_state or {}
        stopping_strings = ["NO", "UNCHANGED", "YES"]

        llm_response_text, stop_reason = self.summarizer.generate_with_sse(
            prompt=gate_check_full_prompt,
            state=current_custom_state,
            phase_id=phase_id,
            step_id="perform_gate_check",
            history_path=self.history_path,
            stopping_strings=stopping_strings,
            match_prefix_only=True,
        )

        if shared.stop_everything:
            return False

        llm_response_text = strip_thinking(llm_response_text).strip()
        stop_reason = stop_reason.upper() if stop_reason else ""

        print(f"{_GRAY}Gate check response for '{branch_name}': '{llm_response_text}'. Stop: '{stop_reason}'{_RESET}")

        pm.update_step(phase_id, "perform_gate_check", f"LLM response: {llm_response_text}")

        # Logic to determine YES vs NO
        # 1. Check stopping strings logic
        if stop_reason in ["NO", "UNCHANGED"]:
            print(f"{_INPUT}Gate check for '{branch_name}' returned {stop_reason}. Skipping branch.{_RESET}")
            pm.done_step(phase_id, "perform_gate_check", f"Gate: FAILED ({stop_reason})\nResponse: {llm_response_text}")
            return False

        # 2. Check text content logic
        is_negative = (
            llm_response_text.upper().startswith("NO") or
            "UNCHANGED" in llm_response_text.upper()
        )

        if is_negative:
            print(f"{_INPUT}Gate check for '{branch_name}' returned NO/UNCHANGED. Skipping branch.{_RESET}")
            pm.done_step(phase_id, "perform_gate_check", f"Gate: FAILED\nResponse: {llm_response_text}")
            return False

        # 3. Default to YES (Process the branch)
        print(f"{_INPUT}Gate check for '{branch_name}' returned YES. Proceeding.{_RESET}")
        pm.done_step(phase_id, "perform_gate_check", f"Gate: PASSED\nResponse: {llm_response_text}")
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
        Request the LLM to produce a complete replacement for the branch's dictionary and apply it when the result differs.
        
        Returns:
            True if the branch was updated or a valid identical dictionary was returned, False otherwise.
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
        Query a branch to determine whether it needs updates and, if so, request a list of field changes from the LLM and apply those updates to the branch data.
        
        This function performs three high-level steps:
        1. Ask the LLM (using query_template) whether the branch contains changes.
        2. If changes are indicated, request a list of updates (path, value) using update_template.
        3. Parse the LLM response and apply each update to `data` relative to this branch.
        
        Parameters:
            branch_name (str): Logical name of the branch being queried (used in prompts and phase IDs).
            data (dict): The branch's data structure to be modified in-place by applied updates.
            formatted_data (FormattedData): Contextual, pre-formatted representation of the branch used to build prompts.
            query_template (str): Prompt template used to ask whether changes exist for this branch.
            update_template (str): Prompt template used to request the list of field updates when changes are detected.
            schema_class (ParsedSchemaClass): Schema metadata used to build and validate prompts and examples.
            keys (list): Path keys identifying this branch within the larger data structure (used when constructing prompts).
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

        phase_id = branch_name.lower().replace(" ", "_")
        pm = self._phase_manager

        print(f"{_GRAY}Querying LLM: Does branch '{branch_name}' need updates?{_RESET}")
        pm.start_step(phase_id, "query_changes", "Checking for changes...")

        branch_query_full_prompt = (
            f"Current context for '{branch_name}':\n"
            f"{formatted_data.mark_field(branch_name)}\n\n"
            f"{branch_query_prompt}"
        )

        current_custom_state = self.custom_state or {}
        query_stopping_strings = ["NO", "UNCHANGED", "YES"]

        llm_response_text, stop_reason = self.summarizer.generate_with_sse(
            prompt=branch_query_full_prompt,
            state=current_custom_state,
            phase_id=phase_id,
            step_id="query_changes",
            history_path=self.history_path,
            stopping_strings=query_stopping_strings,
            match_prefix_only=True,
        )

        if shared.stop_everything:
            return

        llm_response_text = strip_thinking(llm_response_text).strip()
        stop_reason = stop_reason.upper() if stop_reason else ""

        print(f"{_GRAY}Query branch '{branch_name}' response: '{llm_response_text}'. Stop: '{stop_reason}'{_RESET}")

        pm.done_step(phase_id, "query_changes", f"Response: {llm_response_text}")

        # Check if negative response
        if stop_reason in ["NO", "UNCHANGED"] or llm_response_text.upper() in ["NO", "UNCHANGED"]:
            pm.update_step(phase_id, "query_branch_for_changes", "No changes needed")
            print(f"{_INPUT}Skipping updates for branch '{branch_name}' (query returned NO).{_RESET}")
            return

        pm.update_step(phase_id, "query_changes", "Changes detected, requesting details...")
        pm.start_step(phase_id, "apply_updates", "Applying updates...")

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

        llm_update_response_text, _ = self.summarizer.generate_with_sse(
            prompt=branch_update_list_full_prompt,
            state=temp_state,  # Use the temp state with history
            phase_id=phase_id,
            step_id="apply_updates",
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
            pm.update_step(phase_id, "apply_updates", f"Applying {len(parsed_updates)} update(s)...")
            formatted_updates = []
            for update_item in parsed_updates:
                path_str = update_item["path"]
                value = update_item["value"]

                # Convert path string to key list (e.g. "status.0.name" -> ["status", 0, "name"])
                keyList_relative_to_branch = split_keys_to_list(path_str)

                try:
                    old_value = recursive_get(data, keyList_relative_to_branch)
                    print(f"{_GRAY}[{branch_name}] Applying update: {path_str} == {json.dumps(old_value, indent=None)}{_RESET}")
                    recursive_set(data, keyList_relative_to_branch, value)  # Modifies 'data' in place
                    print(f"{_GRAY}[{branch_name}] Applied update: {path_str} = {json.dumps(value, indent=None)}{_RESET}")
                    formatted_updates.append({"path": path_str, "value": value, "old_value": old_value})
                except Exception as e:
                    print(f"{_ERROR}[{branch_name}] Failed to apply update {path_str} = {repr(value)}: {e}{_RESET}")
            pm.done_step(phase_id, "apply_updates", f"Applied {len(parsed_updates)} update(s)", {"updates": formatted_updates, "raw_json": json.dumps(parsed_updates, indent=2)})
        else:
            pm.done_step(phase_id, "apply_updates", "No updates to apply")
            print(f"{_GRAY}No specific field updates applied to '{branch_name}' from branch query response.{_RESET}")

    def _retrieve_nested_field(
        self,
        target_schema_class: ParsedSchemaClass,
        defaults_to_inherit: list[str] | None = None,
        do_inherit_triggers: bool = False,
        depth: int = -1,
    ):
        """
        Traverse alias-wrapped schema layers to locate and return the innermost schema field that holds the final (non-alias) schema type.
        
        Parameters:
            target_schema_class (ParsedSchemaClass): Starting schema class to traverse. Traversal proceeds through alias wrappers whose `. _field.type` is another `ParsedSchemaClass`.
            defaults_to_inherit (list[str] | None): Names of default attributes to inherit from parent schemas when creating effective intermediate schema copies; passed to `_inherit_defaults_from_parent`.
            do_inherit_triggers (bool): If True, merge trigger mappings from parent schemas into intermediate effective schema copies.
            depth (int): Maximum alias-wrapping levels to traverse. A value of -1 means no limit; a non-negative integer stops traversal after that many alias hops.
        
        Returns:
            ParsedSchemaField: The schema field object (`_field`) from the deepest inspected schema class (the field that holds the final schema type).
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
        """
        Request an updated value for a field from the LLM, parse and validate the response, and return the updated value (with retries on parse/validation failures).
        
        Attempts to assemble a prompt from `prompt_template_str`, call the LLM, and convert or validate the response to `expected_type`. Retries on parsing or schema validation errors up to a small limit; returns the original `current_value` if the update is aborted, a stop signal is received, or retries are exhausted.
        
        Parameters:
            item_name_prefix (str): Identifier or path prefix for the item being updated (used for prompt/context).
            field_name (str): Name of the specific field to update (empty when updating a whole branch).
            current_value (Any): Current value present in the data; used as a fallback and for context selection.
            formatted_data (FormattedData): Context wrapper used to mark and format the current data for the prompt.
            prompt_template_str (str): Template used to build the LLM prompt (may be augmented with validation feedback on retries).
            expected_type (type): The Python type the returned value should conform to (e.g., int, dict, list, or typing hints).
            target_schema_or_type (ParsedSchemaClass | None): Optional schema used to validate complex structured responses.
            keys (list): Path keys to the current data location (used when assembling the prompt).
            context_marker_path_override (str | None): Optional override path used for marking context instead of the default.
            entry_name_for_prompt (str | None): Optional entry name included in the prompt when generating new dictionary entries.
        
        Returns:
            The parsed and validated updated value (converted to the requested type when possible). If no valid update is produced, or on stop/error conditions, returns the original `current_value`.
        """
        try:
            context_path_for_marker = context_marker_path_override
            if context_path_for_marker is None:
                if isinstance(current_value, (dict, list)):
                    context_path_for_marker = item_name_prefix
                else:
                    context_path_for_marker = f"{item_name_prefix}.{field_name}"

            base_prompt = prompt_template_str
            max_retries = 2
            last_error = None
            phase_id = item_name_prefix.lower().replace(" ", "_")
            pm = self._phase_manager
            field_phase_id = f"{phase_id}.{field_name}".lower() if field_name is not None else phase_id
            field_phase_name = f"{item_name_prefix}.{field_name}" if field_name is not None else item_name_prefix

            def _done_field_phase(msg=None):
                if field_name is not None:
                    pm.done_phase(field_phase_id, msg)

            if field_name is not None:
                pm.start_phase(field_phase_id, field_phase_name)

            for attempt in range(max_retries + 1):
                if attempt > 0 and last_error:
                    error_feedback = f"\n\nThe previous attempt failed validation: {last_error}\nPlease correct the response."
                    effective_prompt = base_prompt + error_feedback
                    pm.update_step(field_phase_id, "perform_update", f"Retry {attempt}/2: {last_error}")
                else:
                    effective_prompt = base_prompt
                    pm.update_step(field_phase_id, "perform_update", "Assembling prompt...")

                prompt = self._create_update_prompt(
                    item_name=item_name_prefix,
                    field_name=field_name,
                    formatted_data=formatted_data,
                    prompt_template_str=effective_prompt,
                    target_schema_or_type=target_schema_or_type,
                    entry_name=entry_name_for_prompt,
                    keys=keys,
                    indent=2,
                )

                current_custom_state = self.custom_state or {}
                llm_interaction_prompt = f"Context for '{item_name_prefix}':\n{formatted_data.mark_field(item_name_prefix, context_path_for_marker)}\n\n{prompt}"

                text, stop = self.summarizer.generate_with_sse(
                    llm_interaction_prompt,
                    current_custom_state,
                    phase_id=field_phase_id,
                    step_id="perform_update",
                    history_path=self.history_path,
                    match_prefix_only=False,
                )

                if shared.stop_everything:
                    print(
                        f"{_HILITE}Stop signal received during LLM value update generation for '{context_path_for_marker}'.{_RESET}"
                    )
                    _done_field_phase()
                    return current_value

                if stop:
                    _done_field_phase()
                    return current_value

                try:
                    if expected_type == int:
                        _done_field_phase()
                        return int(text)
                    if expected_type == float:
                        _done_field_phase()
                        return float(text)
                    if expected_type == bool:
                        _done_field_phase()
                        return text.strip().lower() in ["true", "yes", "1"]

                    if expected_type == list or (hasattr(expected_type, "__origin__") and expected_type.__origin__ is list):
                        stripped_text = strip_response(text)
                        try:
                            parsed_list = jsonc.loads(stripped_text)
                            if isinstance(parsed_list, list):
                                _done_field_phase()
                                return parsed_list
                            else:
                                last_error = f"Response was valid JSON but not a list: '{stripped_text}'"
                                print(f"{_ERROR}LLM response for list field {context_path_for_marker}: {last_error}{_RESET}")
                                pm.update_step(field_phase_id, "perform_update", f"Parse error: {last_error}")
                                continue
                        except json.JSONDecodeError:
                            is_list_of_str = False
                            if hasattr(expected_type, "__args__") and len(expected_type.__args__) == 1:
                                if expected_type.__args__[0] == str:
                                    is_list_of_str = True
                            if is_list_of_str:
                                _done_field_phase()
                                return [item.strip() for item in text.split(",")]
                            last_error = f"Response is not valid JSON and type is not list[str]: '{stripped_text}'"
                            print(f"{_ERROR}LLM response for list field {context_path_for_marker}: {last_error}{_RESET}")
                            pm.update_step(field_phase_id, "perform_update", f"Parse error: {last_error}")
                            continue

                    if expected_type == dict or (hasattr(expected_type, "__origin__") and expected_type.__origin__ is dict):
                        stripped_text = strip_response(text)
                        try:
                            parsed_dict = jsonc.loads(stripped_text)
                            if isinstance(parsed_dict, dict):
                                _done_field_phase()
                                return parsed_dict
                            else:
                                last_error = f"Response was valid JSON but not a dict: '{stripped_text}'"
                                print(f"{_ERROR}LLM response for dict field {context_path_for_marker}: {last_error}{_RESET}")
                                pm.update_step(field_phase_id, "perform_update", f"Parse error: {last_error}")
                                continue
                        except json.JSONDecodeError:
                            last_error = f"Response is not valid JSON: '{stripped_text}'"
                            print(f"{_ERROR}LLM response for dict field {context_path_for_marker}: {last_error}{_RESET}")
                            pm.update_step(field_phase_id, "perform_update", f"Parse error: {last_error}")
                            continue

                    # Try to parse as JSON for complex types
                    stripped_text = strip_response(text)
                    try:
                        parsed_value = jsonc.loads(stripped_text)

                        if isinstance(target_schema_or_type, ParsedSchemaClass):
                            validation_errors = self.summarizer.last.schema_parser.validate_data(parsed_value, target_schema_or_type.name)
                            if validation_errors:
                                error_msg = "; ".join(validation_errors[:3])
                                last_error = f"Validation failed: {error_msg}"
                                print(f"{_ERROR}Validation errors for {context_path_for_marker}: {validation_errors}{_RESET}")
                                pm.update_step(field_phase_id, "perform_update", f"Validation error: {error_msg}")
                                if attempt < max_retries:
                                    continue
                                else:
                                    print(f"{_ERROR}Max retries reached, accepting value with warnings.{_RESET}")
                                    _done_field_phase()
                                    return parsed_value
                        _done_field_phase()
                        return parsed_value
                    except json.JSONDecodeError:
                        _done_field_phase()
                        return text

                except (ValueError, TypeError) as e:
                    last_error = f"Could not convert response to type {expected_type}: {e}"
                    print(f"{_ERROR}Could not convert LLM response '{text}' to type {expected_type} for {context_path_for_marker}: {e}{_RESET}")
                    pm.update_step(field_phase_id, "perform_update", f"Type error: {last_error}")
                    if attempt >= max_retries:
                        _done_field_phase()
                        return current_value

            _done_field_phase()
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

    def _resolve_cross_branch_reference(self, reference: str) -> str:
        """Resolve a {subjects.X.Y.Z} style reference to a formatted string.

        Syntax:
            {subjects.characters.John.description}      → Formatted
            {subjects.characters.John.description:raw} → Raw JSON
            {subjects.events.scenes[-1]}              → Formatted, last scene
            {subjects.arcs[-1].summary}                 → Formatted, last arc

        Args:
            reference: The reference string (including curly braces)

        Returns:
            Formatted or raw string value, or error message if not found.
        """
        # Remove curly braces
        ref_content = reference.strip("{}")

        # Check for :raw suffix
        raw_format = False
        if ref_content.endswith(":raw"):
            raw_format = True
            ref_content = ref_content[:-5]

        # Parse path parts
        parts = ref_content.split(".")

        if not parts or parts[0] != "subjects":
            return f"[Invalid reference: {reference}]"

        # Get subject name (e.g., "characters", "arcs", "events")
        if len(parts) < 2:
            return f"[Invalid reference path: {reference}]"

        subject_name = parts[1]

        # Load subject data from history_path
        subject_data = {}
        subject_path = self.history_path / f"{subject_name}.json"
        if subject_path.exists():
            try:
                subject_data = load_json(subject_path) or {}
            except Exception as e:
                print(f"{_ERROR}Error loading subject data for '{subject_name}': {e}{_RESET}")
                return f"[Error loading {subject_name}]"
        else:
            # Subject file doesn't exist yet
            return f"[{subject_name} not yet initialized]"

        # Navigate the remaining path
        value = self._resolve_fuzzy_path(subject_data, parts[2:])

        if value is None:
            return f"[{subject_name}: path not found]"

        # Format the result
        if raw_format:
            return json.dumps(value, indent=2)

        return self._format_for_llm(value)

    def _resolve_fuzzy_path(self, data: dict | list, path_parts: list[str]) -> Any:
        """Navigate nested structure with fuzzy name matching.

        Args:
            data: The data to navigate (dict or list)
            path_parts: List of path components (e.g., ["John", "description"] or ["-1", "summary"])

        Returns:
            The value at the resolved path, or None if not found.
        """
        if not path_parts:
            return data

        current = data

        for part in path_parts:
            if isinstance(current, dict):
                # Fuzzy match against keys
                matched_key = self._fuzzy_match(list(current.keys()), part)
                if matched_key:
                    current = current[matched_key]
                else:
                    return None
            elif isinstance(current, list):
                # Handle list indices (including negative)
                try:
                    idx = int(part)
                    current = current[idx]
                except (ValueError, IndexError):
                    return None
            else:
                return None

        return current

    def _fuzzy_match(self, keys: list[str], pattern: str) -> str | None:
        """Match pattern against keys (case-insensitive, partial match).

        "John" matches "John Smith", "john_doe", "Johnny"
        Returns first match or original pattern if exact match found.

        Args:
            keys: List of available keys
            pattern: Pattern to match

        Returns:
            Matched key or None
        """
        pattern_lower = pattern.lower()

        # First try exact match
        for key in keys:
            if key.lower() == pattern_lower:
                return key

        # Then try partial match
        for key in keys:
            if pattern_lower in key.lower():
                return key

        # Return original pattern as fallback (might work for direct access)
        return pattern if pattern in keys else None

    def _format_for_llm(self, value: Any, max_length: int = 2000) -> str:
        """Format a value for inclusion in LLM prompts.

        Args:
            value: The value to format
            max_length: Maximum length before truncation

        Returns:
            Formatted string representation
        """
        if value is None:
            return "N/A"

        if isinstance(value, dict):
            lines = []
            for k, v in value.items():
                if isinstance(v, (dict, list)):
                    v_str = json.dumps(v, indent=2)[:200]
                else:
                    v_str = str(v)
                lines.append(f"- {k}: {v_str}")
            result = "\n".join(lines) if lines else "Empty"
        elif isinstance(value, list):
            if not value:
                return "Empty"
            # Format list items
            lines = []
            for i, item in enumerate(value[:20]):  # Limit to 20 items
                if isinstance(item, dict):
                    lines.append(f"[{i}]: {json.dumps(item, indent=2)[:150]}...")
                else:
                    lines.append(f"[{i}]: {str(item)}")
            result = "\n".join(lines)
            if len(value) > 20:
                result += f"\n... and {len(value) - 20} more items"
        else:
            result = str(value)

        # Truncate if too long
        if len(result) > max_length:
            result = result[:max_length] + "..."

        return result

    def _extract_cross_branch_references(self, template: str) -> list[str]:
        """Extract all {subjects.X.Y.Z} references from a template string.

        Args:
            template: The template string to scan

        Returns:
            List of reference strings found (including curly braces)
        """
        import re
        pattern = r'\{subjects\.[^}]+\}'
        return re.findall(pattern, template)

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
        history_internal = self.custom_state.get("history", {}).get("internal", [])
        current_message_node = f"{len(history_internal) * 2}_1_1"

        # Calculate counts for context
        events_data = self.all_subjects_data.get("events", {})
        scenes_dict = events_data.get("scenes", {}) if events_data else {}
        chapters_dict = events_data.get("chapters", {}) if events_data else {}

        scenes_count = len(scenes_dict)  # Total scenes in story
        chapters_count = len(chapters_dict)  # Total chapters in story

        current_scene_data = self.all_subjects_data.get("current_scene", {})
        current_scene_number = current_scene_data.get("_scene_number", 1)
        current_chapter_number = current_scene_data.get("_chapter_number", 1)
        current_arc_number = current_scene_data.get("_arc_number", 1)

        # Calculate relative counts (within current chapter/arc)
        # For scenes in current chapter: count scenes since last chapter transition
        scenes_in_chapter = 1
        if chapters_dict:
            chapters_list = list(chapters_dict.values()) if isinstance(chapters_dict, dict) else chapters_dict
            if chapters_list:
                last_chapter = chapters_list[-1] if chapters_list else {}
                last_chapter_ending_scene = last_chapter.get("ending_scene", 0)
                scenes_in_chapter = max(1, scenes_count - last_chapter_ending_scene)

        scene_in_chapter = scenes_in_chapter  # Current scene position within chapter

        # For chapters in current arc: count chapters since last arc transition
        chapters_in_arc = chapters_count
        arcs_data = self.all_subjects_data.get("arcs", [])
        if isinstance(arcs_data, list) and arcs_data:
            last_arc = arcs_data[-1] if arcs_data else {}
            last_arc_ending_chapter = last_arc.get("ending_chapter", 0)
            chapters_in_arc = max(1, chapters_count - last_arc_ending_chapter)

        chapter_in_arc = chapters_in_arc  # Current chapter position within arc

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
            "current_message_node": current_message_node,
            # Absolute counts
            "scenes_count": scenes_count,
            "chapters_count": chapters_count,
            "current_scene_number": current_scene_number,
            "current_chapter_number": current_chapter_number,
            "current_arc_number": current_arc_number,
            # Relative counts (within current chapter/arc)
            "scenes_in_chapter": scenes_in_chapter,
            "scene_in_chapter": scene_in_chapter,
            "chapters_in_arc": chapters_in_arc,
            "chapter_in_arc": chapter_in_arc,
            **kwargs,
        }

        # Resolve cross-branch references ({subjects.X.Y.Z})
        cross_refs = self._extract_cross_branch_references(prompt_template_str)
        for ref in cross_refs:
            resolved_value = self._resolve_cross_branch_reference(ref)
            ref_key = ref.strip("{}")
            format_kwargs[ref_key] = resolved_value
            ref_key_compat = ref_key.replace(".", "_")
            format_kwargs[ref_key_compat] = resolved_value

        if entry_name is not None:
            format_kwargs["entry_name"] = entry_name

        try:
            return format_str_or_jinja(prompt_template_str, **format_kwargs)
        except KeyError as e:
            print(
                f"{_ERROR}Missing key in prompt template formatting: {e}. Template: '{prompt_template_str}', Args: {format_kwargs}{_RESET}"
            )
            return f"Update field '{field_name}' for item '{item_name}'. Current value: {value_str}"