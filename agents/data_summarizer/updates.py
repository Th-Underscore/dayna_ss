"""LLM update parsing and application (T4 extraction). Gate checks, branch queries/full updates, field-update generation and application.
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
    _retarget_dict_key,
    _retarget_value,
    _is_negative_verdict,
    _tolerant_json_loads,
    _is_schema_echo,
    _filter_new_entry_names,
    _extract_entry_names,
)

if False:  # forward refs only
    from .core import DataSummarizer


class DSUpdatesMixin:
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
        gate_check_full_prompt = gate_check_prompt + self._context_restatement(formatted_data, branch_name, keys)

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

        if runtime.stop_everything:
            pm.done_step(phase_id, "perform_gate_check", "Stopped")
            return False

        llm_response_text = strip_thinking(llm_response_text).strip()
        stop_reason = stop_reason.upper() if stop_reason else ""

        print(f"{_GRAY}Gate check response for '{branch_name}': '{llm_response_text}'. Stop: '{stop_reason}'{_RESET}")

        pm.update_step(phase_id, "perform_gate_check", f"LLM response: {llm_response_text}")

        def _is_negative(text: str, stop: str) -> bool:
            return stop in ("NO", "UNCHANGED") or text.upper().startswith("NO") or "UNCHANGED" in text.upper()

        def _is_affirmative(text: str, stop: str) -> bool:
            return stop == "YES" or text.upper().startswith("YES")

        def _record(result: bool) -> bool:
            if result:
                print(f"{_INPUT}Gate check for '{branch_name}' returned YES. Proceeding.{_RESET}")
                pm.done_step(phase_id, "perform_gate_check", f"Gate: PASSED\nResponse: {llm_response_text}")
            else:
                print(f"{_INPUT}Gate check for '{branch_name}' returned NO/UNCHANGED. Skipping branch.{_RESET}")
                pm.done_step(phase_id, "perform_gate_check", f"Gate: FAILED\nResponse: {llm_response_text}")
            return result

        # 1. Clear verdict
        if _is_negative(llm_response_text, stop_reason):
            return _record(False)
        if _is_affirmative(llm_response_text, stop_reason):
            return _record(True)

        # 2. Non-verdict (e.g. story continuation instead of an answer): one
        #    bounded retry with a strict re-frame before falling back to the
        #    fail-open default (matches the pre-existing YES-on-anything default,
        #    so the A2 no-freeze property is preserved).
        print(f"{_GRAY}Gate check for '{branch_name}' returned a non-verdict; retrying once.{_RESET}")
        pm.update_step(phase_id, "perform_gate_check", "Non-verdict; retrying with strict re-frame")
        retry_prompt = (
            gate_check_full_prompt
            + "\n\nYour previous response was not a valid verdict.\n"
              "This is a memory-management decision, not a story continuation. "
              "Do NOT write fiction. Re-read the question above and respond with exactly YES or NO."
        )
        retry_text, retry_reason = self.summarizer.generate_with_sse(
            prompt=retry_prompt,
            state=current_custom_state,
            phase_id=phase_id,
            step_id="perform_gate_check",
            history_path=self.history_path,
            stopping_strings=stopping_strings,
            match_prefix_only=True,
        )
        if runtime.stop_everything:
            pm.done_step(phase_id, "perform_gate_check", "Stopped")
            return False
        retry_text = strip_thinking(retry_text).strip()
        retry_reason = retry_reason.upper() if retry_reason else ""
        print(f"{_GRAY}Gate check retry for '{branch_name}': '{retry_text}'. Stop: '{retry_reason}'{_RESET}")
        if _is_negative(retry_text, retry_reason):
            return _record(False)

        # 3. Default to YES (process the branch)
        return _record(True)

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

        if runtime.stop_everything:
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
        keys: list,
        skip_query: bool = False
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
            return True

        phase_id = branch_name.lower().replace(" ", "_")
        pm = self._phase_manager

        if skip_query:
            # Update-only: skip the separate "does it need updates?" pre-query and go
            # straight to the self-detecting update call. The branch update template
            # already supports NO_UPDATES_REQUIRED, so this halves the per-entry cost
            # (query + update -> one call) with no behavior loss — the pre-query
            # returns YES almost every turn in a live story, so it is pure overhead.
            return self._branch_update_only(
                branch_name, data, formatted_data, update_template, schema_class, keys, phase_id, pm
            )

        print(f"{_GRAY}Querying LLM: Does branch '{branch_name}' need updates?{_RESET}")
        pm.start_step(phase_id, "query_changes", "Checking for changes...")

        branch_query_full_prompt = branch_query_prompt + self._context_restatement(formatted_data, branch_name, keys)

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

        if runtime.stop_everything:
            pm.done_step(phase_id, "query_changes", "Stopped")
            return True

        llm_response_text = strip_thinking(llm_response_text).strip()
        stop_reason = stop_reason.upper() if stop_reason else ""

        print(f"{_GRAY}Query branch '{branch_name}' response: '{llm_response_text}'. Stop: '{stop_reason}'{_RESET}")

        pm.done_step(phase_id, "query_changes", f"Response: {llm_response_text}")

        # Check if negative response (NO / UNCHANGED / NO_UPDATES_REQUIRED / bare "n").
        # The templates demand variants of these tokens; treating them all as a
        # negative skips the wasted apply_updates call.
        neg = llm_response_text.upper().strip().rstrip(".!")
        negative = (
            not neg
            or stop_reason in ["NO", "UNCHANGED"]
            or neg in ["NO", "UNCHANGED", "N"]
            or neg.startswith("NO")
            or "NO_UPDATES_REQUIRED" in neg
        )
        if negative:
            pm.update_step(phase_id, "query_changes", "No changes needed")
            print(f"{_INPUT}Skipping updates for branch '{branch_name}' (query returned negative).{_RESET}")
            return False

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

        branch_update_list_full_prompt = branch_update_prompt + self._context_restatement(formatted_data, branch_name, keys)

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

        if runtime.stop_everything:
            return True

        # --- Step 3: Parse and Apply ---
        return self._apply_branch_updates(branch_name, data, llm_update_response_text, phase_id, pm)

    def _branch_update_only(
        self,
        branch_name: str,
        data: dict,
        formatted_data: FormattedData,
        update_template: str,
        schema_class: ParsedSchemaClass,
        keys: list,
        phase_id: str,
        pm,
    ) -> bool:
        """Update-only branch pass (config['skip_query']): run the self-detecting
        branch update template directly, without the pre-query pre-filter."""
        branch_update_prompt = self._create_update_prompt(
            item_name=branch_name,
            field_name="",
            formatted_data=formatted_data,
            prompt_template_str=update_template,
            target_schema_or_type=schema_class,
            keys=keys,
        )
        if not branch_update_prompt:
            return True
        print(f"{_GRAY}Update-only: checking '{branch_name}' for updates.{_RESET}")
        pm.start_step(phase_id, "apply_updates", "Applying updates...")
        branch_update_list_full_prompt = branch_update_prompt + self._context_restatement(formatted_data, branch_name, keys)
        llm_update_response_text, _ = self.summarizer.generate_with_sse(
            prompt=branch_update_list_full_prompt,
            state=self.custom_state or {},
            phase_id=phase_id,
            step_id="apply_updates",
            history_path=self.history_path,
            stopping_strings=["NO", "END"],
            match_prefix_only=True,
        )
        if runtime.stop_everything:
            return True
        llm_update_response_text = strip_thinking(llm_update_response_text).strip()
        neg = llm_update_response_text.upper().strip().rstrip(".!")
        if not llm_update_response_text or neg.startswith("NO") or "NO_UPDATES_REQUIRED" in neg:
            pm.done_step(phase_id, "apply_updates", "No updates needed")
            print(f"{_INPUT}No updates needed for '{branch_name}' (update-only).{_RESET}")
            return False
        return self._apply_branch_updates(branch_name, data, llm_update_response_text, phase_id, pm)

    def _apply_branch_updates(
        self,
        branch_name: str,
        data: dict,
        llm_update_response_text: str,
        phase_id: str,
        pm,
    ) -> bool:
        """Parse and apply a branch update response (path/value list)."""
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

                # Canonicalize: exact/dot-join/normalized/alias resolution against
                # the live structure, so dotted names written without [brackets]
                # ("Mr" + "Peters") and nickname/alias references ("The Judge") land
                # on the real key instead of creating stray nested dicts.
                keyList_relative_to_branch = _resolve_path_keys(data, keyList_relative_to_branch)

                # H3 (referent-resolution gate): symmetric VALUE-side retarget. The
                # path above is already canonicalized; the value (relationship
                # partner keys, row keys, alias entries) was not. Re-address any
                # dangling referent (a lookalike key that would mint a stray sibling
                # or dangling edge) onto its canonical same-branch node. Fires only on
                # a DISTINCT collision and is a guaranteed no-op otherwise, so
                # legitimate shared/undercover identities (one canonical key) and
                # cross-branch referents are never rewritten.
                value = _retarget_value(value, data)

                try:
                    old_value = recursive_get(data, keyList_relative_to_branch)
                    print(f"{_GRAY}[{branch_name}] Applying update: {path_str} == {json.dumps(old_value, indent=None)}{_RESET}")
                    recursive_set(data, keyList_relative_to_branch, value)  # Modifies 'data' in place
                    # Null hygiene: list-typed targets must never carry explicit
                    # null/empty/None-string items minted by partial LLM updates
                    # (the biography-null leak). Filtering is shape-only and
                    # type-agnostic — no per-field keys involved.
                    stored = recursive_get(data, keyList_relative_to_branch)
                    if isinstance(stored, list):
                        cleaned_list = [
                            v for v in stored
                            if v is not None and v != "" and str(v).strip().lower() not in ("null", "none")
                        ]
                        if len(cleaned_list) != len(stored):
                            recursive_set(data, keyList_relative_to_branch, cleaned_list)
                            print(f"{_GRAY}[{branch_name}] Null scrub on {path_str}: "
                                  f"{len(stored)} -> {len(cleaned_list)} items{_RESET}")
                            value = cleaned_list
                    print(f"{_GRAY}[{branch_name}] Applied update: {path_str} = {json.dumps(value, indent=None)}{_RESET}")
                    formatted_updates.append({"path": path_str, "value": value, "old_value": old_value})
                except Exception as e:
                    print(f"{_ERROR}[{branch_name}] Failed to apply update {path_str} = {repr(value)}: {e}{_RESET}")
            pm.done_step(phase_id, "apply_updates", f"Applied {len(parsed_updates)} update(s)", {"updates": formatted_updates, "raw_json": json.dumps(parsed_updates, indent=2)})
        else:
            pm.done_step(phase_id, "apply_updates", "No updates to apply")
            print(f"{_GRAY}No specific field updates applied to '{branch_name}' from branch query response.{_RESET}")

        return True

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
                    pm.warn_step(field_phase_id, "perform_update", f"Retry {attempt}/2: {last_error}")
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
                llm_interaction_prompt = prompt + self._context_restatement(
                    formatted_data, item_name_prefix, keys, marker_paths=(item_name_prefix, context_path_for_marker)
                )

                text, stop = self.summarizer.generate_with_sse(
                    llm_interaction_prompt,
                    current_custom_state,
                    phase_id=field_phase_id,
                    step_id="perform_update",
                    history_path=self.history_path,
                    match_prefix_only=False,
                )

                if runtime.stop_everything:
                    print(
                        f"{_HILITE}Stop signal received during LLM value update generation for '{context_path_for_marker}'.{_RESET}"
                    )
                    _done_field_phase()
                    return current_value

                if stop:
                    _done_field_phase()
                    return current_value

                if not text.strip():
                    last_error = "The model returned an empty response."
                    pm.warn_step(field_phase_id, "perform_update", f"Retry {attempt}/2: {last_error}")
                    continue

                if _is_negative_verdict(text):
                    pm.warn_step(field_phase_id, "perform_update", "No update required (negative verdict).")
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
                            parsed_list = _tolerant_json_loads(stripped_text)
                            if isinstance(parsed_list, list):
                                _done_field_phase()
                                return parsed_list
                            elif isinstance(parsed_list, dict):
                                return unexpand_lists_in_data_from_llm(parsed_list, target_schema_or_type, self.schema_parser)
                            last_error = f"Response was valid JSON but not a list: '{stripped_text}'"
                            print(f"{_ERROR}LLM response for list field {context_path_for_marker}: {last_error}{_RESET}")
                            pm.warn_step(field_phase_id, "perform_update", f"Parse error: {last_error}")
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
                            pm.warn_step(field_phase_id, "perform_update", f"Parse error: {last_error}")
                            continue

                    if expected_type == dict or (hasattr(expected_type, "__origin__") and expected_type.__origin__ is dict):
                        stripped_text = strip_response(text)
                        try:
                            parsed_dict = _tolerant_json_loads(stripped_text)
                            if isinstance(parsed_dict, dict):
                                _done_field_phase()
                                return parsed_dict
                            else:
                                last_error = f"Response was valid JSON but not a dict: '{stripped_text}'"
                                print(f"{_ERROR}LLM response for dict field {context_path_for_marker}: {last_error}{_RESET}")
                                pm.warn_step(field_phase_id, "perform_update", f"Parse error: {last_error}")
                                continue
                        except json.JSONDecodeError:
                            last_error = f"Response is not valid JSON: '{stripped_text}'"
                            print(f"{_ERROR}LLM response for dict field {context_path_for_marker}: {last_error}{_RESET}")
                            pm.warn_step(field_phase_id, "perform_update", f"Parse error: {last_error}")
                            continue

                    # Try to parse as JSON for complex types
                    stripped_text = strip_response(text)
                    try:
                        parsed_value = _tolerant_json_loads(stripped_text)

                        if isinstance(target_schema_or_type, ParsedSchemaClass):
                            validation_errors = self.summarizer.last.schema_parser.validate_data(parsed_value, target_schema_or_type.name)
                            if validation_errors:
                                error_msg = "; ".join(validation_errors[:3])
                                last_error = f"Validation failed: {error_msg}"
                                print(f"{_ERROR}Validation errors for {context_path_for_marker}: {validation_errors}{_RESET}")
                                pm.warn_step(field_phase_id, "perform_update", f"Validation error: {error_msg}")
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
                    pm.warn_step(field_phase_id, "perform_update", f"Type error: {last_error}")
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
        trigger_update: bool = False,
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
            trigger_update (bool, optional): If True, allow LLM call. Defaults to False (opt-in).
        """
        if not trigger_update:
            print(f"{_GRAY}Skipping update for {parent_item_name_prefix}.{field_name} (trigger_update=False){_RESET}")
            return field_value
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
        parent_data_object[field_name] = updated_value
        return updated_value
