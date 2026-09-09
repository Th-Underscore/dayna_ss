"""Prompt assembly for DataSummarizer (T4 extraction). Subject routing guide, context restatement, nested-field retrievers, LLM value formatting, scene recap/events text, and _create_update_prompt.
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


RESTATE_MAP_THRESHOLD_CHARS = 10000


class DSPromptsMixin:
    def _build_subject_routing_guide(self, subject_name: str) -> str:
        """Compose a dynamic routing block listing every subject and what belongs in each.

        Preprended to `new_entry_query_prompt_template` so the model routes new entities to
        the correct subject instead of dumping named people/beings into `elements`.
        Derived from the schema (`subjects` + `subject_routing`) — no hardcoded dispatch.
        """
        routing = getattr(getattr(self, "schema_parser", None), "subject_routing", None) or {}
        subjects = list((getattr(getattr(self, "schema_parser", None), "subjects", None) or {}).keys())
        lines = ["SUBJECT ROUTING — which kinds of new entries belong in which subject:"]
        for name in subjects:
            desc = routing.get(name)
            if not desc:
                continue
            marker = "  <<< YOU ARE NOW POPULATING THIS SUBJECT >>>" if name == subject_name else ""
            lines.append(f"- {name.upper()}: {desc} {marker}".rstrip())
        lines.append(
            "Route every new entity to its correct subject. In particular, never put a named "
            "person or a speaking/recurring being in elements — that belongs in characters — "
            "and never put an organization, committee, or crowd in elements — that belongs in groups."
        )
        return "\n".join(lines)

    def _context_restatement(
        self,
        formatted_data: FormattedData,
        branch_name: str,
        keys: list,
        marker_paths: tuple = (),
    ) -> str:
        """Whole-subject context block for a per-entry call, gated by config.

        The engine re-states the ENTIRE rendered subject map near the generation
        boundary so a small-active model knows exactly which data is in question
        (the same map already sits at the start of the prompt in the retrieval
        context — the re-statement is boundary-accentuation insurance against
        the lost-in-the-middle effect). But the block is O(whole map) and sits
        AFTER the per-call entry reference, so it is re-encoded on every call
        (never prefix-cached) and it pressures the context window, forcing
        history truncation in long runs.

        Config keys (summarizer.config):
          restate_map_context            "auto" (default) | "always" | "never"
          restate_map_threshold_chars    int, default RESTATE_MAP_THRESHOLD_CHARS

        auto: keep the full map while it is under the threshold (cheap
        boundary-accentuation insurance for small maps), else drop it and return
        a compact sibling roster (names only) so per-entry calls keep
        boundary-near cross-reference names for relationships.
        """
        config = getattr(self.summarizer, "config", None) or {}
        mode = config.get("restate_map_context", "auto")
        map_text = formatted_data.mark_field(*marker_paths) if marker_paths else formatted_data.mark_field(branch_name)

        if mode == "always":
            return f"\n\nCurrent context for '{branch_name}':\n{map_text}"
        if mode != "auto":
            return ""  # never: no boundary re-statement at all
        threshold = int(config.get("restate_map_threshold_chars", RESTATE_MAP_THRESHOLD_CHARS))
        if len(map_text) <= threshold:
            return f"\n\nCurrent context for '{branch_name}':\n{map_text}"
        # Dropped (map outgrew the budget): compact sibling roster. keys[:-1] is
        # the parent map path for per-entry calls (e.g. ["entries", "Evelyn"] ->
        # entries map), so the model keeps boundary-near cross-reference names.
        try:
            if len(keys) >= 2:
                parent = recursive_get(formatted_data.data, keys[:-1], default=None)
                if isinstance(parent, dict):
                    names = [str(k) for k in parent.keys() if not str(k).startswith("_")]
                    if names:
                        return "\n\nOther entries in this section:\n" + "\n".join(f"- {n}" for n in names)
        except Exception:
            pass
        return ""

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

    def _resolve_current_message_node(self) -> str:
        """Message node of the just-finished exchange, from the REAL history.

        Message nodes are 1-indexed; the engine's canonical "current message
        index" is ``len(real_history) * 2`` (same convention as
        ``new_scene_start_node`` in prepare_context and ``_scene_start_message``
        in summarize_latest_state), so message-node comparisons stay consistent.
        The old code derived the node from ``custom_state['history']['internal']``
        — the ARTIFICIAL retrieval-context list (bounded to the rolling window,
        ~6-8 pairs), which capped message nodes at ~16 regardless of the true
        story length. Falls back to that legacy derivation only for construction
        sites that don't pass ``real_history``.
        """
        if getattr(self, "real_history", None):
            return f"{len(self.real_history) * 2}_1_1"
        internal = self.custom_state.get("history", {}).get("internal", []) or []
        return f"{len(internal) * 2}_1_1"

    def _scene_recap_text(self, max_msgs: int = 12) -> str:
        """Plain numbered listing of the most recent exchanges — the aggregated
        'what happened this scene' recap used by scene-aggregation (Type 2)
        schema prompts. Only rendered when a template references {{ scene_recap }}.
        """
        state = self.custom_state or {}
        name1 = state.get("name1", "user")
        name2 = state.get("name2", "assistant")
        # The REAL message history is the truth for a story recap (each entry is
        # a [user, assistant] pair). The custom_state internal history is the
        # artificial retrieval-context list — bounded to the rolling window and
        # (in enumerated mode) laced with "What are the relevant characters?"
        # Q&A pairs — so it only serves as a legacy fallback for construction
        # sites that don't pass real_history.
        real = getattr(self, "real_history", None)
        if real:
            lines = []
            shown = 0
            for pair in real[-max_msgs:]:
                if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                    continue
                u, a = str(pair[0] or "").strip(), str(pair[1] or "").strip()
                if u in ("", "<|BEGIN-VISIBLE-CHAT|>"):
                    continue
                shown += 1
                lines.append(f"{shown}. {name1}: {u}")
                if a:
                    lines.append(f"   {name2}: {a}")
            return "\n".join(lines) if lines else "(no prior messages yet)"
        internal = (state.get("history", {}) or {}).get("internal", []) or []
        if not internal:
            return "(no prior messages yet)"
        pairs = []
        i = 0
        while i < len(internal):
            u = internal[i]
            a = internal[i + 1] if i + 1 < len(internal) else ""
            def _txt(m):
                if isinstance(m, (list, tuple)) and len(m) >= 2:
                    return str(m[1])
                return str(m)
            pairs.append((_txt(u), _txt(a)))
            i += 2
        lines = []
        for idx, (u, a) in enumerate(pairs[-max_msgs:]):
            lines.append(f"{idx + 1}. {name1}: {u}")
            if a.strip():
                lines.append(f"   {name2}: {a}")
        return "\n".join(lines)

    def _scene_events_text(self, max_entries: int = 12) -> str:
        """Compact rendering of the events store (scenes/crucial events/past
        events/chapters) for scene-aggregation (Type 2) schema prompts.
        Only rendered when a template references {{ scene_events }}.
        """
        events = self.all_subjects_data.get("events", {})
        if not isinstance(events, dict):
            return "(no events recorded yet)"
        sections = []
        for key in ("past", "scenes", "events", "chapters"):
            d = events.get(key)
            if not isinstance(d, dict) or not d:
                continue
            entries = []
            for name, ev in list(d.items())[:max_entries]:
                if isinstance(ev, dict):
                    summary = str(ev.get("summary", "") or "")
                    entries.append(f"- {name}: {summary}" if summary else f"- {name}")
                # else:
                #     entries.append(f"- {name}")
            if entries:
                sections.append(f"=== {key} ===\n" + "\n".join(entries))
        return "\n\n".join(sections) if sections else "(no events recorded yet)"

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
            branch_data = recursive_get(formatted_data.data, keys, default=None)
            existing_keys = list(branch_data.keys()) if isinstance(branch_data, dict) else []
            if existing_keys:
                branch_list_str = lambda: "\n".join(f"- {k}" for k in existing_keys)
            else:
                list_template_key = f"{item_name}_list"
                if target_schema_or_type and hasattr(target_schema_or_type, "defaults"):
                    declared = target_schema_or_type.defaults.get("list_template")
                    if declared:
                        list_template_key = declared
                branch_list_str = lambda: FormattedData(formatted_data.data, list_template_key).st or "The list is empty! Maybe add some items?"

        state = self.summarizer.last.state
        current_message_node = self._resolve_current_message_node()

        # Calculate counts for context
        events_data = self.all_subjects_data.get("events", {})
        scenes_dict = events_data.get("scenes", {}) if events_data else {}
        chapters = _entries_as_list(events_data.get("chapters")) if events_data else []

        scenes_count = len(scenes_dict)  # Total scenes in story
        chapters_count = len(chapters)  # Total chapters in story

        current_scene_data = self.all_subjects_data.get("current_scene", {})
        current_scene_number = current_scene_data.get("_scene_number", 1)
        current_chapter_number = current_scene_data.get("_chapter_number", 1)
        current_arc_number = current_scene_data.get("_arc_number", 1)

        # Calculate relative counts (within current chapter/arc)
        # For scenes in current chapter: count scenes since last chapter transition
        scenes_in_chapter = 1
        if chapters:
            last_chapter = chapters[-1] if isinstance(chapters[-1], dict) else {}
            last_chapter_ending_scene = _safe_int(last_chapter.get("ending_scene"))
            scenes_in_chapter = max(1, scenes_count - last_chapter_ending_scene)

        scene_in_chapter = scenes_in_chapter  # Current scene position within chapter

        # For chapters in current arc: count chapters since last arc transition
        chapters_in_arc = chapters_count
        arcs_data = self.all_subjects_data.get("arcs", {})
        if isinstance(arcs_data, dict) and arcs_data:
            last_arc = max(
                (a for a in arcs_data.values() if isinstance(a, dict)),
                key=lambda a: _safe_int(a.get("ending_chapter")),
                default=None,
            )
            if last_arc is not None:
                last_arc_ending_chapter = _safe_int(last_arc.get("ending_chapter"))
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
            "scene_recap": lambda: self._scene_recap_text(),
            "scene_events": lambda: self._scene_events_text(),
            "{user}": state["name1"],
            "{char}": state["name2"],
            "name1": state["name1"],
            "name2": state["name2"],
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
            "subjects": self.all_subjects_data,
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
