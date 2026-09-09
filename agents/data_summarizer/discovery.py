"""New-entry discovery and entry selection (T4 extraction). _detect_and_add_new_entries_to_branch and _select_entries_to_update.
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
)

if False:  # forward refs only
    from .core import DataSummarizer


ADD_NEW_MAX_TOTAL_ENTRIES = 120


class DSDiscoveryMixin:
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
        """Handles querying for and adding new entries to a collection branch.

        Supports both dict-keyed branches ("Characters: dict[str, Character]")
        and list-typed branches ("Chapters: list[Chapter]"); list entries are
        deduped against their title-ish field values and appended in order.
        """
        field_origin = (
            getattr(branch_schema_class._field.type, "__origin__", None)
            if branch_schema_class.definition_type == "alias"
            else None
        )
        if field_origin not in (dict, list):
            return
        is_list_branch = field_origin is list

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
        subject_name = branch_name.split(".")[0]
        routing_guide = self._build_subject_routing_guide(subject_name)
        query_prompt = self._create_update_prompt(
            item_name=branch_name,
            field_name="",  # Not specific to a field
            formatted_data=formatted_data,
            prompt_template_str=new_query_template,
            target_schema_or_type=branch_schema_class,
            keys=keys,
        )
        query_prompt = f"{routing_guide}\n\n{query_prompt}"

        query_stopping_strings = ["NO", "[]"]
        llm_query_response, stop_reason = self.summarizer.generate_using_tgwui(
            prompt=query_prompt,
            state=self.custom_state or {},
            history_path=self.history_path,
            stopping_strings=query_stopping_strings,
            match_prefix_only=True,
            phase_id=str(branch_name).lower().replace(" ", "_"),
            step_id="add_new_query",
        )

        if runtime.stop_everything:
            return
        stripped_response = strip_response(llm_query_response)
        if (stop_reason and stop_reason in query_stopping_strings) or stripped_response in query_stopping_strings:
            print(f"{_GRAY}LLM indicates no new entries for '{branch_name}'.{_RESET}")
            return

        new_entry_names = _extract_entry_names(stripped_response)
        if not new_entry_names:
            print(
                f"{_ERROR}Failed to extract new entry names for '{branch_name}' from LLM response "
                f"(expected a JSON array of names): {llm_query_response}{_RESET}"
            )
            return
        print(f"{_SUCCESS}Identified potential new entries for '{branch_name}': {new_entry_names}{_RESET}")

        if not new_entry_names:
            return

        # Deterministic dedupe + growth cap guard against fabricated micro-entries
        if len(data) >= ADD_NEW_MAX_TOTAL_ENTRIES:
            print(
                f"{_GRAY}add_new: '{branch_name}' already has {len(data)} entries (>= {ADD_NEW_MAX_TOTAL_ENTRIES}); skipping.{_RESET}"
            )
            return
        if is_list_branch:
            # _filter_new_entry_names reads dict keys/aliases; list branches
            # dedupe against each entry's title-ish field instead.
            data_for_dedupe = {}
            for entry in data:
                if isinstance(entry, dict):
                    for name_key in ("title", "name", "formal_name", "id"):
                        val = entry.get(name_key)
                        if isinstance(val, str) and val.strip():
                            data_for_dedupe[val.strip()] = entry
                            break
        else:
            data_for_dedupe = data
        filtered_names = _filter_new_entry_names(new_entry_names, data_for_dedupe)
        if not filtered_names:
            print(f"{_GRAY}add_new: all {len(new_entry_names)} proposed entries for '{branch_name}' were duplicates; skipping.{_RESET}")
            return
        if len(filtered_names) < len(new_entry_names):
            print(
                f"{_GRAY}add_new: dedupe kept {len(filtered_names)}/{len(new_entry_names)} proposed entries for '{branch_name}'.{_RESET}"
            )
        new_entry_names = filtered_names

        # Step 2: Generate data for each new entry
        value_schema_class = None
        field_args = getattr(branch_schema_class._field.type, "__args__", ()) or ()
        value_schema_class = next(
            (arg for arg in field_args if isinstance(arg, ParsedSchemaClass)), None
        )

        if not value_schema_class:
            print(f"{_ERROR}Could not determine value schema for new entries in '{branch_name}'. Skipping generation.{_RESET}")
            return

        for entry_name in new_entry_names:
            if runtime.stop_everything:
                return
            print(f"{_INPUT}Generating data for new entry '{entry_name}' in '{branch_name}'...{_RESET}")

            max_retries = 2
            last_error = None
            for attempt in range(max_retries + 1):
                if attempt > 0 and last_error:
                    error_feedback = (
                        f"\n\nYour previous attempt was rejected: {last_error}\n"
                        f"Do NOT reproduce the schema, the example, or any other text from this prompt. "
                        f"Respond with ONLY the new entry's JSON object."
                    )
                    generation_template = new_entry_template + error_feedback
                else:
                    generation_template = new_entry_template

                entry_generation_prompt = self._create_update_prompt(
                    item_name=branch_name,
                    field_name="",  # Not updating a sub-field of the new entry yet
                    formatted_data=formatted_data,
                    prompt_template_str=generation_template,
                    target_schema_or_type=value_schema_class,
                    entry_name=entry_name,
                    keys=keys,
                )
                # Universal anti-echo guard: the schema/example blocks sit right
                # before the generation boundary, and a small-active model can
                # latch onto them and reproduce the JSONSchema instead of the
                # entry. Applies to every new-entry call regardless of subject.
                entry_generation_prompt += (
                    "\n\nIMPORTANT: Output ONLY the new entry's JSON data — a fresh object "
                    f"for '{entry_name}'. Never reproduce the schema definition, the example "
                    "JSON, or any other text from this prompt."
                )

                llm_entry_response, _ = self.summarizer.generate_using_tgwui(
                    prompt=entry_generation_prompt,
                    state=self.custom_state or {},
                    history_path=self.history_path,
                    # No specific stopping strings, expect full JSON
                    phase_id=str(branch_name).lower().replace(" ", "_"),
                    step_id="add_new",
                )
                if runtime.stop_everything:
                    return

                stripped_entry_json = strip_response(llm_entry_response)
                try:
                    new_entry_data = _tolerant_json_loads(stripped_entry_json)
                except json.JSONDecodeError:
                    snippet = stripped_entry_json[:120].replace("\n", " ")
                    last_error = f"the response was not valid JSON (started with: {snippet}...)"
                    print(
                        f"{_ERROR}Failed to parse LLM response for new entry '{entry_name}' in "
                        f"'{branch_name}' as JSON: {llm_entry_response}{_RESET}"
                    )
                    continue
                if not isinstance(new_entry_data, dict):
                    last_error = f"the response parsed to {type(new_entry_data).__name__}, not a JSON object"
                    print(
                        f"{_ERROR}LLM response for new entry '{entry_name}' in '{branch_name}' "
                        f"was not a JSON object: {llm_entry_response}{_RESET}"
                    )
                    continue
                # json_repair turns even 'not json {{' into {}, and a truly-empty
                # entry would render as a stub missing every required field.
                if not new_entry_data:
                    last_error = "the response was empty or contained no entry fields"
                    print(
                        f"{_ERROR}LLM response for new entry '{entry_name}' in '{branch_name}' "
                        f"was empty after parsing: {llm_entry_response[:200]}{_RESET}"
                    )
                    continue
                if _is_schema_echo(new_entry_data):
                    last_error = "the response reproduced this prompt's schema instead of describing the entry"
                    print(
                        f"{_ERROR}LLM response for new entry '{entry_name}' in '{branch_name}' "
                        f"echoed the schema: {llm_entry_response[:200]}{_RESET}"
                    )
                    continue

                current_message_node = self._resolve_current_message_node()
                if "start" in new_entry_data:
                    new_entry_data["start"]["_message_node"] = current_message_node
                if is_list_branch and not any(
                    isinstance(new_entry_data.get(k), str) and new_entry_data.get(k, "").strip()
                    for k in ("title", "name", "formal_name", "id")
                ):
                    # Keep list entries addressable by their proposed title even
                    # if the model omitted the field.
                    new_entry_data["title"] = entry_name
                print(f"{_DEBUG}Auto-set '_message_node' to '{current_message_node}' for new entry '{entry_name}'.{_RESET}")

                if is_list_branch:
                    data.append(new_entry_data)
                    expanded_data = recursive_get(formatted_data.data, keys, default=None)
                    # Guard aliasing: formatted_data may share the list object,
                    # and unlike dict assignment a mirrored append duplicates.
                    if isinstance(expanded_data, list) and expanded_data is not data:
                        expanded_data.append(new_entry_data)
                else:
                    data[entry_name] = new_entry_data
                    expanded_data: dict = recursive_get(formatted_data.data, keys, default=None)
                    expanded_data[entry_name] = new_entry_data
                self._new_entry_names.setdefault(branch_name, set()).add(entry_name)
                print(f"\r{_SUCCESS}Successfully added new entry '{entry_name}' to '{branch_name}'.{_RESET}")
                break
            else:
                print(
                    f"{_ERROR}add_new: all {max_retries + 1} attempts failed for entry '{entry_name}' "
                    f"in '{branch_name}'; skipping entry (last error: {last_error}).{_RESET}"
                )

        recursive_set(formatted_data.data, keys, data)
        return data

    def _select_entries_to_update(
        self,
        branch_name: str,
        data: dict,
        formatted_data: FormattedData,
        map_schema_class: ParsedSchemaClass,
        config: dict | None,
        keys: list,
    ):
        """Ask the map which of its existing entries plausibly need updating this pass.

        Filters the per-entry loop in `_traverse_structure`: only whitelisted
        entries run their triggers. Fail-open: any error, a map below the
        min-size guard, or a missing template leaves the whitelist None = run
        all. Entries added by `add_new` this pass are always included (they
        need initial population). An empty selection = skip all per-entry
        updates (the whole-map "nothing changed" win case).
        """
        if not isinstance(data, dict):
            self._entry_whitelists[branch_name] = None
            return

        min_size = int((map_schema_class.defaults or {}).get("select_entries_min_size", 4) or 4)
        if len(data) < min_size:
            print(
                f"{_GRAY}select_entries_to_update: '{branch_name}' has {len(data)} entries "
                f"(< {min_size}); running all.{_RESET}"
            )
            self._entry_whitelists[branch_name] = None
            return

        sel_template = self._get_effective_setting(
            data, map_schema_class, "select_entries_to_update_prompt_template", override_config=config
        )
        if not sel_template:
            print(f"{_GRAY}select_entries_to_update: no template for '{branch_name}'; running all.{_RESET}")
            self._entry_whitelists[branch_name] = None
            return

        selection_prompt = self._create_update_prompt(
            item_name=branch_name,
            field_name="",
            formatted_data=formatted_data,
            prompt_template_str=sel_template,
            target_schema_or_type=map_schema_class,
            keys=keys,
        )

        llm_response, _ = self.summarizer.generate_with_sse(
            prompt=selection_prompt,
            state=self.custom_state or {},
            phase_id=branch_name.lower().replace(" ", "_"),
            step_id="select_entries",
            history_path=self.history_path,
        )
        if runtime.stop_everything:
            self._entry_whitelists[branch_name] = None
            return

        stripped = strip_response(llm_response)
        # The template demands a bare JSON array. Require the raw response to
        # look like one AND to parse as a flat string list, else fail open —
        # json_repair can "repair" garbage into nested structures (e.g.
        # '{{{{ NOT JSON ]]]' -> [[[['NOT JSON']]]]) that must NOT be treated
        # as an (empty) selection.
        if not stripped.lstrip().startswith("["):
            print(
                f"{_WARNING}select_entries_to_update: response for '{branch_name}' is not a JSON "
                f"array ({llm_response[:120]!r}); running all.{_RESET}"
            )
            self._entry_whitelists[branch_name] = None
            return
        try:
            parsed = _tolerant_json_loads(stripped)
        except json.JSONDecodeError:
            parsed = None
        if not isinstance(parsed, list) or not all(isinstance(n, str) for n in parsed):
            print(
                f"{_WARNING}select_entries_to_update: unparseable response for '{branch_name}' "
                f"({llm_response[:120]!r}); running all.{_RESET}"
            )
            self._entry_whitelists[branch_name] = None
            return

        resolved: set[str] = set()
        for name in parsed:
            key = _resolve_dict_key(data, name)
            if key is not None:
                resolved.add(key)
        self._entry_whitelists[branch_name] = resolved
        shown = sorted(resolved)[:8]
        print(
            f"{_SUCCESS}select_entries_to_update '{branch_name}': {len(resolved)}/{len(data)} entries selected "
            f"({shown}{'...' if len(resolved) > 8 else ''}).{_RESET}"
        )
