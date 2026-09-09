"""Subject population engine (T4 extraction).

First-scene and identify-based population of subject data, direct population,
internal field stamping. Methods live on PopulationMixin mixed into Summarizer.
"""
from __future__ import annotations

import copy
import json
import jsonc
import traceback
from pathlib import Path
from typing import Any

from ...runtime import runtime

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
    format_str_or_jinja,
    History,
    load_json,
    save_json,
    strip_thinking,
    strip_response,
)

from ...utils.schema_parser import SchemaParser, ParsedSchemaClass
from .context_engine import base_state

if False:  # forward refs only
    from .core import Summarizer


class PopulationMixin:
    def _set_last_exchange(self, user_input: str, output: str):
        history: History = self.last.custom_state["history"]["internal"]
        current_history_length = len(history)
        if current_history_length < self.last.history_length:
            raise ValueError(f"History length ({len(history)}) is less than expected ({self.last.history_length})")
        for i in range(len(history) - 1, self.last.history_length - 1, -1):
            history.pop(i)
        history.append([user_input, output])

    def _set_internal_fields(self, data: Any, message_node: str = "1_1_1") -> None:
        """Recursively set internal fields (prefixed with _) in data structure.

        Args:
            data: The data structure to process (dict, list, or other)
            message_node: The message_node value to set for _message_node fields
        """
        if isinstance(data, dict):
            for key, value in data.items():
                if key == "_message_node":
                    data[key] = message_node
                else:
                    self._set_internal_fields(value, message_node)
        elif isinstance(data, list):
            for item in data:
                self._set_internal_fields(item, message_node)

    def _populate_subject_direct(
        self,
        initial_world_data_path: Path,
        schema_parser: SchemaParser,
        state: dict,
        subject_name: str,
        population_config: dict,
        phase_id: str | None = None,
        step_id: str | None = None,
    ) -> str | None:
        """
        Populates a single subject using direct LLM interaction.

        Args:
            initial_world_data_path: Path to save the populated data
            schema_parser: Schema parser instance
            state: Current state
            subject_name: Name of the subject to populate
            population_config: Configuration dict with 'target_file' and 'prompt_template'
        """

        target_file = population_config.get("target_file", f"{subject_name}.json")
        prompt_template = population_config.get("prompt_template", "")

        print(f"{_DEBUG}Attempting to populate '{subject_name}' for {initial_world_data_path}{_RESET}")
        target_path = initial_world_data_path / target_file

        try:
            schema_def = schema_parser.get_subject_class(subject_name)
            if not schema_def:
                print(f"{_ERROR}Could not retrieve schema definition for '{subject_name}'. Aborting.{_RESET}")
                save_json({}, target_path)
                return

            schema_class_name = schema_def.name
            example_json = schema_def.generate_example_json(all_definitions_map=schema_parser.definitions)
            example_json_str = json.dumps(example_json, indent=2)

            all_definitions = schema_parser.get_relevant_json_schema_definitions(schema_class_name)
            all_definitions_str = json.dumps(all_definitions, indent=2)

            char_context = state.get("context", "")
            char_context_str = f'Character Context:\n"""\n{char_context}\n"""\n\n' if char_context else ""
            char_greeting = state["history"]["internal"][0][1]
            char_greeting_str = f'Initial Greeting:\n"""\n{char_greeting}\n"""\n\n'

            # Format the prompt template with context variables using Jinja
            prompt = format_str_or_jinja(
                prompt_template,
                char_context=char_context,
                char_greeting=char_greeting,
                char_context_str=char_context_str,
                char_greeting_str=char_greeting_str,
                all_relevant_definitions_json_str=all_definitions_str,
                example_json=example_json_str,
                schema_definition_json=all_definitions_str,
                retry_feedback_placeholder="",
            )

            custom_state = copy.deepcopy(state)
            custom_state.update(copy.deepcopy(base_state))

            max_retries = 2
            populated_data = None
            pm = self._phase_manager

            for attempt in range(max_retries + 1):
                print(f"{_DEBUG}Attempt {attempt + 1}/{max_retries + 1} to generate '{subject_name}' data...{_RESET}")
                if attempt > 0:
                    print(f"{_INPUT}Retrying LLM prompt for '{subject_name}' with validation feedback.{_RESET}")
                    if phase_id and step_id:
                        pm.warn_step(phase_id, step_id, f"Retry {attempt}/{max_retries}")

                response_text, _ = self.generate_with_sse(
                    prompt=prompt,
                    state=custom_state,
                    phase_id=phase_id or "legacy",
                    step_id=step_id or "generate",
                    history_path=initial_world_data_path,
                    match_prefix_only=False,
                )

                if runtime.stop_everything:
                    print(f"{_HILITE}Stop signal received during '{subject_name}' generation.{_RESET}")
                    save_json({}, target_path)
                    return

                print(f"{_DEBUG}LLM response for '{subject_name}' (Attempt {attempt + 1}): {response_text[:300]}...{_RESET}")
                cleaned_response = strip_response(response_text)

                try:
                    current_data = jsonc.loads(cleaned_response)
                    validation_errors = schema_parser.validate_data(current_data, schema_class_name)

                    if not validation_errors:
                        self._set_internal_fields(current_data, message_node="1_1_1")
                        populated_data = current_data
                        print(f"{_SUCCESS}'{subject_name}' data validated successfully on attempt {attempt + 1}.{_RESET}")
                        break
                    else:
                        print(f"{_ERROR}Validation errors for '{subject_name}' on attempt {attempt + 1}:{_RESET}")
                        for err in validation_errors:
                            print(f"{_ERROR}- {err}{_RESET}")

                        if attempt < max_retries:
                            error_feedback = f"The previous attempt to generate JSON for '{subject_name}' failed with validation errors:\n"
                            for err in validation_errors:
                                error_feedback += f"- {err}\n"
                            prompt = format_str_or_jinja(
                                prompt_template,
                                char_context=char_context,
                                char_greeting=char_greeting,
                                char_context_str=char_context_str,
                                char_greeting_str=char_greeting_str,
                                all_relevant_definitions_json_str=all_definitions_str,
                                example_json=example_json_str,
                                schema_definition_json=all_definitions_str,
                                retry_feedback_placeholder=error_feedback + "\n",
                            )
                except json.JSONDecodeError as e:
                    print(f"{_ERROR}Failed to parse LLM response for '{subject_name}' as JSON: {e}{_RESET}")
                    print(f"{_ERROR}LLM Raw Response was: {_GRAY}{cleaned_response}{_RESET}")
                    if attempt < max_retries:
                        error_feedback = f"The previous JSON was invalid. Ensure valid JSON output.\n"
                        prompt = format_str_or_jinja(
                            prompt_template,
                            char_context=char_context,
                            char_greeting=char_greeting,
                            char_context_str=char_context_str,
                            char_greeting_str=char_greeting_str,
                            all_relevant_definitions_json_str=all_definitions_str,
                            example_json=example_json_str,
                            schema_definition_json=all_definitions_str,
                            retry_feedback_placeholder=error_feedback,
                        )
                except Exception as e:
                    print(f"{_ERROR}Unexpected error during '{subject_name}' processing on attempt {attempt + 1}: {e}{_RESET}")
                    traceback.print_exc()

            if populated_data:
                save_json(populated_data, target_path)
                print(f"{_SUCCESS}Successfully populated and saved '{target_file}' at {target_path}{_RESET}")
                if phase_id and step_id:
                    pm.done_step(phase_id, step_id, f"Populated {subject_name}: {response_text}")
                return response_text
            else:
                print(f"{_ERROR}Failed to populate '{subject_name}' after all retries. Saving empty JSON.{_RESET}")
                save_json({}, target_path)
                return None

        except Exception as e:
            print(f"{_ERROR}Error in _populate_subject_direct for '{subject_name}': {e}{_RESET}")
            traceback.print_exc()
            save_json({}, target_path)
            return None

    def _populate_subject_identify(
        self,
        initial_world_data_path: Path,
        schema_parser: SchemaParser,
        state: dict,
        subject_name: str,
        population_config: dict,
        phase_id: str | None = None,
        step_id: str | None = None,
    ) -> str | None:
        """
        Populates multiple subjects using identify-then-populate pattern.

        Args:
            initial_world_data_path: Path to save the populated data
            schema_parser: Schema parser instance
            state: Current state
            subject_name: Name of the subject (used as the primary, e.g., "Characters")
            population_config: Configuration dict with 'identification_prompt', 'population_prompt', etc.
        """

        target_files = population_config.get("target_files", [])
        wrapper_key = population_config.get("target_key", "entries")  # Key to wrap entity data (e.g., "entries")
        type_mapping = population_config.get("type_mapping", {})  # Maps LLM type -> target_key (e.g., {"element": "elements"})
        identification_prompt_template = population_config.get("identification_prompt", "")
        population_prompt_template = population_config.get("population_prompt", "")

        print(f"{_DEBUG}Attempting to populate entities via identification for '{subject_name}'{_RESET}")

        # TODO: Pass in more context variables (example json, schema definitions, etc.) to the prompts
        char_context = state.get("context", "")
        char_context_str = f'Character Context:\n"""\n{char_context}\n"""\n\n' if char_context else ""
        char_greeting = state["history"]["internal"][0][1]
        char_greeting_str = f'Initial Greeting:\n"""\n{char_greeting}\n"""\n\n'

        custom_state = copy.deepcopy(state)
        custom_state.update(copy.deepcopy(base_state))
        pm = self._phase_manager

        # Step 1: Identification
        identification_prompt = format_str_or_jinja(
            identification_prompt_template,
            char_context=char_context,
            char_greeting=char_greeting,
            char_context_str=char_context_str,
            char_greeting_str=char_greeting_str,
            name1=state.get("name1"),
            name2=state.get("name2"),
        )

        # Phase 1: Exclude already-populated subject names from identification
        existing_names = {}
        for json_path in sorted(initial_world_data_path.glob("*.json")):
            if json_path.name in target_files:
                continue
            exclude_data = load_json(json_path)
            exclude_entries = exclude_data.get("entries", exclude_data)
            if isinstance(exclude_entries, dict):
                existing_names[json_path.stem] = list(exclude_entries.keys())

        if any(existing_names.values()):
            exclusion_lines = []
            for category, names in existing_names.items():
                exclusion_lines.append(f"- {category}: {', '.join(names)}")
            identification_prompt += (
                "\n\nThe following names are already tracked under other categories.\n"
                "Do NOT include them as elements:\n"
                + "\n".join(exclusion_lines)
            )

        print(f"{_DEBUG}Prompting LLM for entity identification...{_RESET}")
        identification_response_text, _ = self.generate_with_sse(
            prompt=identification_prompt,
            state=custom_state,
            phase_id=phase_id or "legacy",
            step_id=step_id or "identify",
            history_path=initial_world_data_path,
            match_prefix_only=False,
        )

        if runtime.stop_everything:
            print(f"{_HILITE}Stop signal received during entity identification.{_RESET}")
            for tf in target_files:
                save_json({}, initial_world_data_path / tf)
            return

        print(f"{_DEBUG}LLM response for entity identification: {identification_response_text[:300]}...{_RESET}")

        identified_entities = []
        try:
            cleaned_id_response = strip_response(identification_response_text)
            parsed_entities = jsonc.loads(cleaned_id_response)
            if isinstance(parsed_entities, list):
                identified_entities = [
                    e for e in parsed_entities if isinstance(e, dict) and "type" in e and "name" in e and "descriptor" in e
                ]
            else:
                print(f"{_ERROR}LLM response for entity identification was not a list.{_RESET}")
        except json.JSONDecodeError as e:
            print(f"{_ERROR}Failed to parse LLM response for entity identification as JSON: {e}{_RESET}")
            print(f"{_ERROR}LLM Raw Response was: {_GRAY}{cleaned_id_response}{_RESET}")
        except Exception as e:
            print(f"{_ERROR}An unexpected error occurred during entity identification parsing: {e}{_RESET}")
            traceback.print_exc()

        # Post-LLM filter: remove entities matching already-populated subjects
        all_excluded = set()
        for names in existing_names.values():
            all_excluded.update(n.lower() for n in names)
        filtered = [
            e for e in identified_entities
            if e.get("name", "").lower() not in all_excluded
        ]
        if len(filtered) < len(identified_entities):
            print(f"{_DEBUG}Filtered out {len(identified_entities) - len(filtered)} entities matching already-populated subjects.{_RESET}")
        identified_entities = filtered

        if not identified_entities:
            print(f"{_INPUT}No entities identified by LLM or parsing failed. Saving empty files.{_RESET}")
            for tf in target_files:
                save_json({}, initial_world_data_path / tf)
            return

        print(f"{_SUCCESS}Identified {len(identified_entities)} entities. Proceeding to detail extraction.{_RESET}")

        # Step 2: Populate each entity
        entity_data = {tf.replace(".json", ""): {} for tf in target_files}

        # Phase 2: Load cross-reference data from schema's relationship_format
        individual_schema_name = subject_name.rstrip("s") if subject_name.endswith("s") else subject_name
        individual_schema = schema_parser.definitions.get(individual_schema_name)
        cross_refs = {}
        if individual_schema and isinstance(individual_schema, ParsedSchemaClass):
            ref_format = individual_schema.defaults.get("relationship_format", {})
            for field_name, ref_cfg in ref_format.items():
                target_type = ref_cfg.get("target_type")
                if not target_type:
                    continue
                target_subject = target_type + "s"
                ref_path = initial_world_data_path / f"{target_subject}.json"
                ref_data = {}
                if ref_path.exists():
                    loaded = load_json(ref_path)
                    entries = loaded.get("entries", loaded)
                    if isinstance(entries, dict):
                        ref_data = entries
                field_type = individual_schema.fields.get(field_name, "str")
                is_dict = field_type.startswith("dict[")
                cross_refs[field_name] = {
                    "target_subject": target_subject,
                    "entries": ref_data,
                    "is_dict": is_dict,
                }

        for entity in identified_entities:
            entity_name: str = entity["name"]
            entity_type: str = entity["type"]
            entity_descriptor: str = entity["descriptor"]

            # Determine which target file to use
            target_key = type_mapping.get(entity_type, entity_type + "s")  # "character" -> "characters", "entity" -> "entities"
            if target_key not in entity_data:
                print(f"{_ERROR}Unknown entity type '{entity_type}' for '{entity_name}'. Skipping.{_RESET}")
                continue

            # Use plural schema name for validation (e.g., "Groups" not "Group")
            schema_name = target_key.capitalize()  # "groups" -> "Groups"
            try:
                # Get the schema from definitions (e.g., "Groups" which has "entries" field)
                schema_to_use = schema_parser.definitions.get(schema_name)

                if schema_to_use and isinstance(schema_to_use, ParsedSchemaClass):
                    example_json = schema_to_use.generate_example_json(all_definitions_map=schema_parser.definitions)
                    example_json_str = json.dumps(example_json, indent=2)
                    schema_definition_json = json.dumps(schema_parser.get_relevant_json_schema_definitions(schema_name), indent=2)

                    population_prompt = format_str_or_jinja(
                        population_prompt_template,
                        entity_type=entity_type,
                        entity_name=entity_name,
                        descriptor=entity_descriptor,
                        schema_definition_json=schema_definition_json,
                        example_json=example_json_str,
                        char_context=char_context,
                        char_greeting=char_greeting,
                        char_context_str=char_context_str,
                        char_greeting_str=char_greeting_str,
                        name1=state.get("name1"),
                        name2=state.get("name2"),
                    )

                    # Inject cross-reference targets into population prompt
                    ref_lines = []
                    for field_name, ref_info in cross_refs.items():
                        if ref_info["entries"]:
                            names = ", ".join(ref_info["entries"].keys())
                            if ref_info["is_dict"]:
                                ref_lines.append(
                                    f"Existing {ref_info['target_subject']} you may reference as keys in '{field_name}': {names}"
                                )
                            else:
                                ref_lines.append(
                                    f"Existing {ref_info['target_subject']} you may reference in the '{field_name}' field: {names}"
                                )
                    if ref_lines:
                        population_prompt += "\n\n" + "\n".join(ref_lines)

                    custom_state_detail = copy.deepcopy(custom_state)
                    max_retries = 2
                    entity_data_validated = None
                    detail_response_text = ""

                    for attempt in range(max_retries + 1):
                        print(f"{_DEBUG}Attempt {attempt + 1}/{max_retries + 1} to generate details for {entity_type} '{entity_name}'...{_RESET}")
                        if attempt > 0:
                            print(f"{_INPUT}Retrying LLM prompt for '{entity_name}' with validation feedback.{_RESET}")
                            if phase_id and step_id:
                                pm.warn_step(phase_id, step_id, f"Retry {attempt}/{max_retries}")

                        detail_response_text, _ = self.generate_with_sse(
                            prompt=population_prompt,
                            state=custom_state_detail,
                            phase_id=phase_id or "legacy",
                            step_id=step_id or "populate",
                            history_path=initial_world_data_path,
                            match_prefix_only=False,
                        )

                        if runtime.stop_everything:
                            print(f"{_HILITE}Stop signal received during detail extraction for '{entity_name}'.{_RESET}")
                            break

                        print(f"{_DEBUG}LLM response for '{entity_name}' (Attempt {attempt+1}): {detail_response_text[:300]}...{_RESET}")
                        cleaned_detail_response = strip_response(detail_response_text)

                        try:
                            current_entity_data = jsonc.loads(cleaned_detail_response)

                            # # Handle case where LLM returns full structure with "entries" wrapper
                            # if isinstance(current_entity_data, dict) and "entries" in current_entity_data:
                            #     entries = current_entity_data["entries"]
                            #     if isinstance(entries, dict) and entity_name in entries:
                            #         current_entity_data = entries[entity_name]
                            #         print(f"{_DEBUG}Unwrapped 'entries' wrapper for '{entity_name}'.{_RESET}")

                            validation_errors = schema_parser.validate_data(current_entity_data, schema_name)

                            # Cross-reference validation from relationship_format
                            if isinstance(current_entity_data, dict):
                                for field_name, ref_info in cross_refs.items():
                                    field_value = current_entity_data.get(field_name)
                                    if field_value is None:
                                        continue
                                    target_label = ref_info["target_subject"]
                                    ref_entries = ref_info["entries"]
                                    if ref_info["is_dict"] and isinstance(field_value, dict):
                                        for key in field_value:
                                            if key and key not in ref_entries:
                                                validation_errors.append(
                                                    f"'{field_name}' key '{key}' references non-existent {target_label}"
                                                )
                                    elif isinstance(field_value, list):
                                        for item in field_value:
                                            if item and item not in ref_entries:
                                                validation_errors.append(
                                                    f"'{field_name}' references '{item}' which does not exist in {target_label}"
                                                )

                            if not validation_errors:
                                self._set_internal_fields(current_entity_data, message_node="1_1_1")
                                entity_data_validated = current_entity_data
                                print(f"{_SUCCESS}Details for {entity_type} '{entity_name}' validated successfully on attempt {attempt + 1}.{_RESET}")
                                break
                            else:
                                print(f"{_ERROR}Validation errors for '{entity_name}' on attempt {attempt + 1}:{_RESET}")
                                for err in validation_errors:
                                    print(f"{_ERROR}- {err}{_RESET}")

                                if attempt < max_retries:
                                    error_feedback = f"The previous attempt to generate JSON for '{entity_name}' failed validation. Correct these issues:\n"
                                    for err in validation_errors:
                                        error_feedback += f"- {err}\n"
                                    population_prompt = format_str_or_jinja(
                                        population_prompt_template,
                                        entity_type=entity_type,
                                        entity_name=entity_name,
                                        descriptor=entity_descriptor,
                                        schema_definition_json=schema_definition_json,
                                        example_json=example_json_str,
                                        char_context=char_context,
                                        char_greeting=char_greeting,
                                        char_context_str=char_context_str,
                                        char_greeting_str=char_greeting_str,
                                        name1=state.get("name1"),
                                        name2=state.get("name2"),
                                    ) + "\n" + error_feedback
                                    if ref_lines:
                                        population_prompt += "\n\n" + "\n".join(ref_lines)
                        except json.JSONDecodeError as e:
                            print(f"{_ERROR}Failed to parse LLM response for '{entity_name}' as JSON: {e}{_RESET}")
                            print(f"{_ERROR}LLM Raw Response was: {_GRAY}{cleaned_detail_response}{_RESET}")
                            if attempt < max_retries:
                                population_prompt = format_str_or_jinja(
                                    population_prompt_template,
                                    entity_type=entity_type,
                                    entity_name=entity_name,
                                    descriptor=entity_descriptor,
                                    schema_definition_json=schema_definition_json,
                                    example_json=example_json_str,
                                    char_context=char_context,
                                    char_greeting=char_greeting,
                                    char_context_str=char_context_str,
                                    char_greeting_str=char_greeting_str,
                                    name1=state.get("name1"),
                                    name2=state.get("name2"),
                                ) + "\nThe previous JSON was invalid. Ensure valid JSON output.\n"
                                if ref_lines:
                                    population_prompt += "\n\n" + "\n".join(ref_lines)
                        except Exception as e:
                            print(f"{_ERROR}Unexpected error processing '{entity_name}' on attempt {attempt + 1}: {e}{_RESET}")
                            traceback.print_exc()

                    if entity_data_validated:
                        if isinstance(entity_data_validated, dict) and wrapper_key in entity_data_validated:
                            entity_data[target_key].update(entity_data_validated[wrapper_key])
                        else:
                            entity_data[target_key][entity_name] = entity_data_validated
                        print(f"{_SUCCESS}Successfully populated and stored details for {entity_type} '{entity_name}'.{_RESET}")
                        if phase_id and step_id:
                            pm.done_step(phase_id, step_id, f"Populated {entity_name}: {detail_response_text}")
                    else:
                        print(f"{_ERROR}Failed to populate valid details for {entity_type} '{entity_name}' after all retries.{_RESET}")

                if runtime.stop_everything:
                    print(f"{_HILITE}Stop signal received after processing for '{entity_name}'. Aborting.{_RESET}")
                    break
            except Exception as e:
                print(f"{_ERROR}Error populating entity '{entity_name}': {e}{_RESET}")
                traceback.print_exc()

        # Save all populated data with wrapper_key wrapper
        for tf in target_files:
            key = tf.replace(".json", "")
            raw_data = entity_data.get(key, {})
            wrapped_data = {wrapper_key: raw_data} if wrapper_key else raw_data
            save_json(wrapped_data, initial_world_data_path / tf)
            print(f"{_SUCCESS}Saved '{tf}' at {initial_world_data_path}{_RESET}")

        return detail_response_text if detail_response_text else None

    def _populate_from_schema(  # Dead code, but keeping for reference
        self,
        initial_world_data_path: Path,
        schema_parser: SchemaParser,
        state: dict,
    ) -> None:
        """
        Dynamically populates initial data based on schema definitions.

        Iterates through all schema classes and populates those with
        'initial_population' defined in their defaults.
        """

        print(f"{_DEBUG}Starting schema-driven initial population...{_RESET}")

        pm = self._phase_manager
        subject_names = [name for name, schema_def in schema_parser.get_subject_classes().items() if schema_def.defaults.get("initial_population")]

        pm.start_turn("Initial Population")

        pm._phases.append({"id": "initial_population", "name": "Initial Population", "weight": 1})
        pm.start_phase("initial_population", "Initial Population")

        for subject_name, schema_def in schema_parser.get_subject_classes().items():
            population_config = schema_def.defaults.get("initial_population")
            if not population_config:
                continue

            mode = population_config.get("mode", "direct")
            phase_id = f"initial_population.{subject_name.lower().replace(' ', '_')}"

            print(f"{_DEBUG}Populating '{subject_name}' using mode '{mode}'...{_RESET}")

            pm.start_phase(phase_id, f"Populate {subject_name}")
            pm.start_step(phase_id, "populate", f"Populating {subject_name}...")

            try:
                response = None
                if mode == "direct":
                    response = self._populate_subject_direct(
                        initial_world_data_path,
                        schema_parser,
                        state,
                        subject_name,
                        population_config,
                        phase_id=phase_id,
                        step_id="populate",
                    )
                elif mode == "identify":
                    response = self._populate_subject_identify(
                        initial_world_data_path,
                        schema_parser,
                        state,
                        subject_name,
                        population_config,
                        phase_id=phase_id,
                        step_id="populate",
                    )
                else:
                    print(f"{_WARNING}Unknown population mode '{mode}' for '{subject_name}'. Skipping.{_RESET}")
                    pm.skip_phase(phase_id, f"Unknown mode: {mode}")

                if response is not None:
                    pm.done_step(phase_id, "populate", f"Populated {subject_name}: {response}")
                else:
                    pm.done_step(phase_id, "populate", f"Populated {subject_name}")
                pm.done_phase(phase_id)
            except Exception as e:
                pm.error_phase(phase_id, str(e))
                print(f"{_ERROR}Error populating '{subject_name}': {e}{_RESET}")

        pm.done_phase("initial_population")
        pm.end_turn()
        pm.end_session(publish=False)

        print(f"{_SUCCESS}Schema-driven initial population complete.{_RESET}")

    def _populate_from_first_scene(
        self,
        user_input: str,
        output: str,
        state: dict,
        history_path: Path,
    ) -> None:
        """
        Populate all subjects using the full first-scene text (character card,
        greeting, user input, and bot output) as the context, then delegates
        to the same schema-driven dispatch logic as _populate_from_schema.

        Uses PhaseManager for realtime UI progress tracking, matching the
        pattern established by _populate_from_schema.
        """
        schema_parser = self.last.schema_parser
        char_context = state.get("context", "")
        greeting = state["history"]["internal"][0][1] if state.get("history", {}).get("internal") else ""
        enriched_context = (
            f"{char_context}\n\n"
            f"Initial Greeting:\n{greeting}\n\n"
            f"First User Input:\n{user_input}\n\n"
            f"First Response:\n{output}"
        )
        enriched_state = copy.deepcopy(state)
        enriched_state["context"] = enriched_context

        pm = self._phase_manager
        pm.start_turn("First Scene Population")

        pm._phases.append({"id": "first_scene_population", "name": "First Scene Population", "weight": 1})
        pm.start_phase("first_scene_population", "First Scene Population")

        for subject_name, schema_def in schema_parser.get_subject_classes().items():
            population_config = schema_def.defaults.get("initial_population")
            if not population_config:
                continue

            mode = population_config.get("mode", "direct")
            phase_id = f"first_scene_population.{subject_name.lower().replace(' ', '_')}"

            pm.start_phase(phase_id, f"Populate {subject_name}")
            pm.start_step(phase_id, "populate", f"Populating {subject_name}...")

            try:
                response = None
                if mode == "direct":
                    response = self._populate_subject_direct(
                        history_path,
                        schema_parser,
                        enriched_state,
                        subject_name,
                        population_config,
                        phase_id=phase_id,
                        step_id="populate",
                    )
                elif mode == "identify":
                    response = self._populate_subject_identify(
                        history_path,
                        schema_parser,
                        enriched_state,
                        subject_name,
                        population_config,
                        phase_id=phase_id,
                        step_id="populate",
                    )
                else:
                    print(f"{_WARNING}Unknown population mode '{mode}' for '{subject_name}'. Skipping.{_RESET}")
                    pm.skip_phase(phase_id, f"Unknown mode: {mode}")

                if response is not None:
                    pm.done_step(phase_id, "populate", f"Populated {subject_name}: {response}")
                else:
                    pm.done_step(phase_id, "populate", f"Populated {subject_name}")
                pm.done_phase(phase_id)
            except Exception as e:
                pm.error_phase(phase_id, str(e))
                print(f"{_ERROR}Error in first-scene population for '{subject_name}': {e}{_RESET}")

        pm.done_phase("first_scene_population")
        pm.end_turn()
        pm.end_session(publish=False)

        print(f"{_SUCCESS}First-scene population complete.{_RESET}")


from ..formatted_data import FormattedData, MessageSummarizer  # re-export (data_summarizer + tests import from here)
