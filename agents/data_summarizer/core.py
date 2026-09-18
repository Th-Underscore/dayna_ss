from pathlib import Path
import copy
import json
import jsonc
import traceback
import re
import time
from typing import Any

try:
    import json_repair
    _HAS_JSON_REPAIR = True
except ImportError:
    _HAS_JSON_REPAIR = False

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
    format_str,
    render_jinja_template,
    format_str_or_jinja,
)
from ...utils.schema_parser import (
    SchemaParser,
    ParsedSchemaClass,
    ParsedSchemaField,
    Action,
    Trigger,
)
from ..formatted_data import FormattedData
from ..summarizer import Summarizer
from ...ui import PhaseManager

RESTATE_MAP_THRESHOLD_CHARS = 10000
ADD_NEW_MAX_TOTAL_ENTRIES = 120

from .parsing import (
    defaults_to_inherit,
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




from .prompts import DSPromptsMixin
from .discovery import DSDiscoveryMixin
from .updates import DSUpdatesMixin
from .traversal import DSTraversalMixin
from .archives import DSArchivesMixin


class DataSummarizer(DSPromptsMixin, DSDiscoveryMixin, DSUpdatesMixin, DSTraversalMixin, DSArchivesMixin):
    def __init__(
        self,
        summarizer: Summarizer,
        exchange: tuple[str, str],
        custom_state: dict,
        history_path: Path,
        schema_parser: SchemaParser,
        all_subjects_data: dict,
        phase_manager: "PhaseManager",
        real_history: list | None = None,
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
            real_history (list, optional): The REAL message history (the same
                list passed to summarize_latest_state — greeting pair + every
                exchange, INCLUDING the one being summarized). Message-derived
                values (_message_node, scene recap) must be computed from THIS,
                never from custom_state's artificial internal history.
        """
        self.summarizer = summarizer
        self._phase_manager = phase_manager
        self.user_input = exchange[0]
        self.output = exchange[1]
        self.custom_state = custom_state
        self.real_history = real_history
        print(
            f"DataSummarizer initialized with history: {_DEBUG} {json.dumps(self.custom_state['history']['internal'], indent=2)}{_RESET}"
        )
        self.history_path = history_path
        self.schema_parser = schema_parser
        self.all_subjects_data = all_subjects_data
        # Entry-selection whitelists (map branch_name -> set of entry keys that
        # must be updated this pass). Keyed per branch; consumed by the dict-of-
        # schema-classes loop in _traverse_structure. None/absent = run all.
        self._entry_whitelists: dict[str, set[str] | None] = {}
        # Names added by add_new this pass, per map branch_name. Always updated
        # (a fresh entry needs initial population even if the selection call
        # did not name it).
        self._new_entry_names: dict[str, set[str]] = {}

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

        if not isinstance(data, dict):
            overrides = {}
        else:
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
    ) -> tuple[bool, bool, bool]:
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
            tuple[bool, bool, bool]: (stop_processing, gate_failed, skip_children)
                - `stop_processing`: `True` if this action completed a full update and subsequent actions should be skipped.
                - `gate_failed`: `True` if a gate check failed and the branch should be skipped.
                - `skip_children`: `True` if a branch query returned a negative ("no changes")
                  verdict and drilling into this branch's sub-fields would only produce
                  wasted LLM calls (they'd answer NO too).
        """
        if runtime.stop_everything:
            return (True, False, False)

        pm = self._phase_manager
        phase_id = branch_name.lower().replace(" ", "_")
        action_name = action.name.lower()

        if action == Action.ADD_NEW:
            pm.start_step(phase_id, action_name, f"Checking for new {branch_name} entries", {"branch_name": branch_name})
            if target_schema_class.definition_type == "alias":
                field_origin = getattr(target_schema_class._field.type, "__origin__", None)
                if field_origin in (dict, list):
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
                    return (False, True, False)  # Gate check failed
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
                    pm.done_step(phase_id, action_name, "Full update: Complete")
                    return (True, False, False)  # Full update succeeded, stop processing
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
                applied_changes = self._perform_branch_query(
                    branch_name, data, formatted_data, bq_template, bu_template, target_schema_class, keys,
                    skip_query=bool(config and config.get("skip_query")),
                )
                if applied_changes is False:
                    # Negative branch query (no changes): don't waste calls drilling
                    # into sub-branches that would also answer NO.
                    pm.done_step(phase_id, action_name, "No changes — skipping sub-branches")
                    return (False, False, True)
            pm.done_step(phase_id, action_name)

        elif action == Action.SELECT_ENTRIES_TO_UPDATE:
            pm.start_step(phase_id, action_name, f"Selecting {branch_name} entries to update", {"branch_name": branch_name})
            try:
                self._select_entries_to_update(
                    branch_name, data, formatted_data, target_schema_class, config, keys
                )
            except Exception as e:
                # Fail-open: never let a selection error skip per-entry updates.
                print(f"{_ERROR}select_entries_to_update failed ({e}); running all entries.{_RESET}")
                self._entry_whitelists[branch_name] = None
            pm.done_step(phase_id, action_name)

        return (False, False, False)  # Continue processing

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

            self._entry_whitelists.clear()
            self._new_entry_names.clear()

            subject_type = self.schema_parser.subjects.get(data_type)
            print(f"{_HILITE}Subject type for '{data_type}':{_RESET} {subject_type}")

            # Repair LLM-written container shapes against the schema BEFORE any
            # update resolution runs. A model writing an empty list where the
            # schema declares dict[str, ...] (e.g. Character.relationships /
            # group_status, observed in cozy_mystery__3fbf1a99) made later
            # name-keyed update paths crash in recursive_set with "Cannot use
            # non-integer key 'Glove' on a list" — the coerced dict lets them
            # resolve and keeps rendering schema-consistent.
            data = coerce_container_types(data, target_schema_class, self.schema_parser)

            unexpanded_formatted_data = FormattedData(data, data_type)
            formatted_data = FormattedData(data, data_type, self.schema_parser)
            if not isinstance(subject_type, ParsedSchemaClass):
                raise ValueError(f"Subject '{data_type}' should be a ParsedSchemaClass: {subject_type}")

            self._update_recursive(data_type, data, formatted_data, unexpanded_formatted_data, target_schema_class)

            # Scene boundary: copy the model's fresh start → now so both render
            # identically at the boundary (the model reliably rewrites start but
            # only copies to now ~31% of the time; see scene_start_to_now_copy.md).
            # Delta-check variant: only copy if start actually changed; skip when
            # start is the empty NO_UPDATES shell and now is already populated.
            if data_type == "current_scene" and self.summarizer.last and self.summarizer.last.is_new_scene_turn:
                if data.get("start") is not None:
                    data["now"] = copy.deepcopy(data["start"])

            # Memory hygiene: collapse restated sentences (intra-entity), strip
            # verbatim copies injected into peer entities, canonicalize/prune/cap
            # relationship rows — all shape-driven, before persistence. Note:
            # under parallel subject workers the peer stores here are the
            # worker's private deep-copy snapshot, so cross-entity pruning
            # persists for THIS subject's own file regardless; peers get their
            # identical treatment from their own workers.
            try:
                from .hygiene import clean_store
                peers = dict(self.all_subjects_data) if isinstance(self.all_subjects_data, dict) else {}
                peers[data_type] = data
                clean_store(
                    peers,
                    summarizer=self.summarizer,
                    log=lambda m: print(f"{_WARNING}{m}{_RESET}"),
                )
            except Exception as e:
                print(f"{_ERROR}Memory hygiene pass failed (non-fatal): {e}{_RESET}")

            print(f"{_HILITE}Summary for {data_type} completed in {time.time() - start:.2f} seconds.{_RESET}")
            normalized = unexpand_lists_in_data_from_llm(data, target_schema_class, self.schema_parser)
            save_json(normalized, self.history_path / f"{data_type}.json")
            if data:
                _lkg = getattr(self.summarizer, "_last_good_subjects", None)
                if _lkg is not None:
                    _lkg[data_type] = copy.deepcopy(data)
            return data

        except Exception as e:
            print(f"{_ERROR}Error in summarize_{data_type}: {e}{_RESET}")
            traceback.print_exc()
            # Persist the in-memory data despite the update error. `_update_recursive`
            # mutates `data` in place, so it still holds the prior entries — dropping
            # the write would leave this history dir WITHOUT the subject file, and the
            # next turn's load/copy-forward would silently propagate an EMPTY subject
            # for every later turn (the t13 character-map wipe in cozy_mystery__dded7733:
            # 10 chars -> missing -> {} for 15+ turns). Prefer the normalized form,
            # fall back to the raw data if normalization itself is what failed.
            try:
                save_json(
                    unexpand_lists_in_data_from_llm(data, target_schema_class, self.schema_parser),
                    self.history_path / f"{data_type}.json",
                )
                print(f"{_WARNING}Preserved prior {data_type} state despite update error.{_RESET}")
            except Exception as e2:
                print(f"{_ERROR}Normalized preservation failed ({e2}); saving raw data.{_RESET}")
                try:
                    save_json(data, self.history_path / f"{data_type}.json")
                    print(f"{_WARNING}Preserved raw {data_type} state despite update error.{_RESET}")
                except Exception as e3:
                    print(f"{_ERROR}Could not preserve {data_type} state at all: {e3}{_RESET}")
            if data:
                _lkg = getattr(self.summarizer, "_last_good_subjects", None)
                if _lkg is not None:
                    _lkg[data_type] = copy.deepcopy(data)
            return data

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

        if _is_negative_verdict(response_text):
            print(f"{_INPUT}[{branch_name_for_log}] LLM indicates no updates required ({response_text.strip()[:40]}).{_RESET}")
            return []

        original_response_text = response_text
        response_text = strip_response(response_text)

        updates: list[dict[str, Any]] = []
        try:
            # Attempt to parse as JSON list (tolerant: json_repair catches the
            # token-level syntax slips small models make)
            parsed_json = _tolerant_json_loads(response_text)
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
        parts = split_keys_to_list(ref_content)

        if not parts or parts[0] != "subjects":
            return f"[Invalid reference: {reference}]"

        # Get subject name (e.g., "characters", "arcs", "events")
        if len(parts) < 2:
            return f"[Invalid reference path: {reference}]"

        subject_name = parts[1]

        # Resolve from the in-memory snapshot first (deterministic under parallel
        # subject workers, where other subjects' files may be mid-write); fall back
        # to reading the file from history_path for subjects not in memory.
        subject_data = {}
        in_memory = self.all_subjects_data.get(subject_name)
        if isinstance(in_memory, dict):
            subject_data = in_memory
        else:
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
                # Resolve via exact/normalized/alias matching (nickname/title).
                matched_key = _resolve_dict_key(current, part) or self._fuzzy_match(list(current.keys()), part)
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
