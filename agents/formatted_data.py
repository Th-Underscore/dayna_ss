"""FormattedData: rendered retrieval-context container and jinja formatting
engine; MessageSummarizer: per-exchange message chunk generation.

Extracted verbatim from agents/summarizer.py (T4 refactor). Shared by both
engine halves (summarizer + data_summarizer) and the soak harness.
"""
from __future__ import annotations

import json
import jsonc
import logging
import re
import traceback
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..runtime import runtime

from ..rag.structured_rag.context_retriever import MessageChunker
from ..utils.schema_parser import SchemaParser
from ..utils.helpers import (
    _BOLD,
    _DEBUG,
    _ERROR,
    _GRAY,
    _HILITE,
    _RESET,
    _SUCCESS,
    _WARNING,
    expand_lists_in_data_for_llm,
    load_json,
    save_json,
    strip_thinking,
    format_str_or_jinja,
    _get_jinja_env,
)

if TYPE_CHECKING:  # forward refs only (annotations are lazy)
    from .summarizer import Summarizer, SummarizationContextCache  # noqa: F401


class MessageSummarizer:
    def __init__(self, summarizer: Summarizer, history_path: Path, current_timestamp: str):
        """Initialize MessageSummarizer with a Summarizer instance and session details."""
        self.summarizer = summarizer
        self.custom_state = summarizer.last.custom_state
        if not summarizer.last or not summarizer.last.context:
            raise RuntimeError("Summarizer.last.context not available for MessageSummarizer initialization.")
        self.chunker: MessageChunker = summarizer.last.context[1].chunker
        self.history_path = history_path
        self.current_timestamp = current_timestamp

    def generate(self, exchange: tuple[str, str], message_idxs: tuple[int, int]) -> None:
        """Summarize messages and store in vector database with metadata.

        Note: Speakers are extracted per-node in context_retriever.py (chunk_message).
        This generates summaries with subjects_referenced at the summary level.
        """
        print(f"{_BOLD}Summarizing messages for indices {message_idxs}{_RESET}")

        pm = self.summarizer._phase_manager

        for i, message_content in enumerate(exchange):
            current_message_idx = message_idxs[i]
            prompt = f'''Analyze the provided message and generate a concise summary of the key events, interactions, and developments.

REMEMBER: Do not add anything else to the response. Only respond with the summary.

Here is the message: """\n{message_content.strip()}\n"""'''
            try:
                summary_text, _ = self.summarizer.generate_with_sse(
                    prompt, self.custom_state, "message_summary", f"summarize_msg_{i}", self.history_path
                )
                if runtime.stop_everything:
                    print(
                        f"{_HILITE}Stop signal received in MessageSummarizer after generating summary for message_idx {current_message_idx}.{_RESET}"
                    )
                    pm.done_step("message_summary", f"summarize_msg_{i}", "Stopped")
                    return
                summary_text = strip_thinking(summary_text)

                summary_chars_present = self.chunker._extract_entities(summary_text, self.chunker.character_name_patterns)
                summary_groups_ref = self.chunker._extract_entities(summary_text, self.chunker.group_name_patterns)
                summary_events_ref = self.chunker._extract_entities(summary_text, self.chunker.event_name_patterns)

                summary_subjects_referenced = {
                    "characters": summary_chars_present,
                    "groups": summary_groups_ref,
                    "events": summary_events_ref,
                }

                summary_chunk_data = {
                    "id": f"{current_message_idx}_summary",
                    "text": summary_text,
                    "indices": [
                        current_message_idx,
                        0,
                        0,
                    ],
                    "timestamp": self.current_timestamp,
                    "speakers": [],
                    "characters_present": summary_chars_present,
                    "subjects_referenced": summary_subjects_referenced,
                    "scene_id": None,
                    "event_id": None,
                    "is_summary": True,
                }
                self.chunker.store_chunks([summary_chunk_data], persist_dir=(self.history_path / "message_index"))
                print(f"{_SUCCESS}Stored summary for message_idx {current_message_idx}{_RESET}")
            except Exception as e:
                print(f"{_ERROR}Error generating message summary for message_idx {current_message_idx}: {e}{_RESET}")
                traceback.print_exc()


class FormattedData:
    def __init__(
        self,
        data: Any,
        data_type: str,
        parser: SchemaParser | None = None,
        context_cache: SummarizationContextCache | None = None,
        all_subjects_data: dict | None = None,
        extra_context: dict | None = None,
    ):
        """Initialize and process data for LLM formatting.

        Prepares a string representation (`self.st`) for LLM prompts.

        If schema parser is provided, expands lists to dicts with schema indicates.

        Args:
            data (Any): Data to process (dict, list, primitive).
            data_type (str): Type hint (e.g., "current_scene", "characters").
            parser (SchemaParser, optional): For schema-based list expansion.
            context_cache: SummarizationContextCache for schema access.
            all_subjects_data: Additional subject data.
            extra_context: Extra context to pass to templates (e.g., scene_names).
        """
        self.original_data = data
        self.data_type = data_type
        self.parser = parser
        self.context_cache = context_cache
        self.last = self.context_cache
        self.all_subjects_data = all_subjects_data
        self.extra_context = extra_context or {}

        # Template-driven shape normalization (no key maps): a subject whose
        # template LOOPs the data sequence (`{% for x in data %}`) must receive
        # a sequence. Dict-keyed row stores ({title: row}) would otherwise have
        # their string keys rendered as entries. Applies to any such template,
        # so new subjects inherit it without a per-type branch.
        if isinstance(data, dict):
            try:
                tpl = FormattedData._load_format_templates(
                    session_id=getattr(context_cache, "session_id", "")) or {}
                tpl_str = tpl.get(data_type, "")
                if tpl_str and "{% for " in tpl_str and " in data %}" in tpl_str:
                    vals = list(data.values())
                    if vals and all(isinstance(v, (dict, str, int, float)) for v in vals):
                        data = vals
            except Exception:
                pass

        if self.parser:
            # TODO: Make dynamic
            actual_data_schema_hint = None
            if data_type == "current_scene":
                actual_data_schema_hint = self.parser.get_subject_class("current_scene")
            elif data_type == "character_list" or data_type == "characters":
                actual_data_schema_hint = self.parser.get_subject_class("characters")
            elif data_type == "groups":
                actual_data_schema_hint = self.parser.get_subject_class("groups")
            elif data_type == "events":
                actual_data_schema_hint = self.parser.get_subject_class("events")
            elif data_type == "scene" or data_type == "event":
                actual_data_schema_hint = self.parser.definitions.get("StoryEvent")
            elif data_type == "general_info":
                actual_data_schema_hint = self.parser.get_subject_class("general_info")
            elif not self.parser.get_subject_class(data_type):
                # Render targets backed by an ALIAS definition (chapters ->
                # Chapters, arcs -> Arcs): hand the expander the alias class so
                # the list shape survives (list-iterating templates must see
                # entries, not stringified {"0": ...} keys).
                alias_def = self.parser.definitions.get(data_type.capitalize())
                if alias_def is not None and getattr(alias_def, "definition_type", "") == "alias":
                    aliased_type = None
                    try:
                        f = alias_def.get_field() if hasattr(alias_def, "get_field") else None
                        aliased_type = f.type if f is not None else None
                    except Exception:
                        aliased_type = None
                    is_list_alias = (
                        hasattr(aliased_type, "__origin__")
                        and aliased_type.__origin__ is list
                    )
                    if isinstance(data, list):
                        actual_data_schema_hint = alias_def
                    elif is_list_alias and isinstance(data, dict):
                        # Dict-keyed row store ({title: row}) feeding a
                        # list-iterating template would render the KEYS.
                        data = list(data.values())

            self.data = expand_lists_in_data_for_llm(data, actual_data_schema_hint, self.parser)
        else:
            self.data = data

        self._str = FormattedData.format_retrieval_data(
            self.data, self.data_type, context_cache=self.context_cache, all_subjects_data=self.all_subjects_data, extra_context=self.extra_context
        )

    def __getitem__(self, index):
        """Allow dictionary-like access to the (potentially expanded) data."""
        return self.data[index]

    _format_templates_caches: dict[str, dict] = {}
    _session_templates_paths: dict[str, Path] = {}

    @staticmethod
    def set_session_templates_path(session_id: str, path: Path):
        """Set the session-local templates path and invalidate only that session's cache entry."""
        path_str = str(path)
        FormattedData._session_templates_paths[session_id] = path
        FormattedData._format_templates_caches.pop(path_str, None)

    @staticmethod
    def _load_format_templates(session_id: str = "") -> dict:
        """Load format templates, preferring session-local over global.

        If session_id is provided and a path is registered for it, loads from that path.
        Otherwise falls back to the default global templates.
        """
        template_path = None
        dss_dir = Path(__file__).parent.parent

        if session_id and session_id in FormattedData._session_templates_paths:
            candidate = FormattedData._session_templates_paths[session_id]
            if candidate.exists():
                template_path = candidate

        if template_path is None:
            template_path = dss_dir / "user_data" / "example" / "format_templates.json"

        cache_key = str(template_path)
        # if cache_key in FormattedData._format_templates_caches:
        #     return FormattedData._format_templates_caches[cache_key]

        try:
            templates = load_json(template_path) or {}
            # FormattedData._format_templates_caches[cache_key] = templates
            print(f"{_DEBUG}Loaded {len(templates)} format templates from {cache_key}{_RESET}")
            return templates
        except Exception as e:
            print(f"{_WARNING}Failed to load format templates: {e}{_RESET}")
            return {}

    @staticmethod
    def get_context_order(session_id: str = "") -> list[dict]:
        """Get the context order configuration from templates."""
        templates = FormattedData._load_format_templates(session_id=session_id)
        return templates.get("_context_order", [])

    @staticmethod
    def _render_jinja_template(
        template_str: str,
        data: dict | list,
        data_type: str,
        path_prefix: str = "",
        all_subjects_data: dict | None = None,
        parser: SchemaParser | None = None,
        extra_context: dict | None = None,
    ) -> str:
        """Render a Jinja2 template with the given data.

        Args:
            template_str: The Jinja2 template string
            data: The data to render
            data_type: The data type (used to determine schema class)
            path_prefix: The prefix for path markers
            all_subjects_data: Full dict of all subjects
            parser: SchemaParser for getting schema defaults
            extra_context: Additional context to pass to the template

        Returns:
            Rendered string or empty string if rendering fails
        """
        if not template_str:
            return ""

        try:
            jinja_env = _get_jinja_env()
            template = jinja_env.from_string(template_str)

            context = {
                "data": data,
                "path": path_prefix,
                "subjects": all_subjects_data,
            }

            if parser:
                context["defaults"] = parser.defaults

            if extra_context:
                context.update(extra_context)
                if "metadata" in extra_context:
                    context["metadata"] = extra_context["metadata"]

            rendered = template.render(**context)
            return rendered.strip()

        except Exception as e:
            print(f"{_WARNING}Jinja template rendering failed for {data_type}: {e}{_RESET}")
            # Fall back to a raw, compact dump so the subject is never silently
            # dropped from the model's context (e.g. when the LLM wrote a list
            # where a mapping was expected and the template called .items()).
            try:
                fallback = json.dumps(data, ensure_ascii=False)[:2000]
            except Exception:
                fallback = ""
            if fallback:
                print(f"{_WARNING}Falling back to raw JSON dump for {data_type} ({len(fallback)} chars).{_RESET}")
                return fallback
            traceback.print_exc()
            return ""

    @staticmethod
    def format_retrieval_data(
        data: dict | list,
        data_type: str,
        prefix: str = "",
        context_cache: SummarizationContextCache | None = None,
        all_subjects_data: dict | None = None,
        extra_context: dict | None = None,
    ) -> str:
        """Format retrieved data based on its type."""
        if not data:
            return ""

        try:
            parser = context_cache.schema_parser if context_cache else None

            user_template = runtime.settings.get(f"template_{data_type}")
            if user_template:
                rendered = FormattedData._render_jinja_template(
                    user_template, data, data_type, prefix, all_subjects_data, parser, extra_context
                )
                if rendered:
                    print(f"{_DEBUG}Using user template for {data_type}{_RESET}")
                    return rendered

            session_id = str(context_cache.history_path) if context_cache and context_cache.history_path else ""
            templates = FormattedData._load_format_templates(session_id=session_id)
            template_str = templates.get(data_type)
            if template_str:
                rendered = FormattedData._render_jinja_template(
                    template_str, data, data_type, prefix, all_subjects_data, parser, extra_context
                )
                if rendered:
                    print(f"{_DEBUG}Using file template for {data_type}{_RESET}")
                    return rendered

            return "<EMPTY>"  # str(data)

        except Exception as e:
            print(f"{_ERROR}Error formatting data: {e}{_RESET}")
            traceback.print_exc()
            print(f"{_BOLD}{json.dumps(data)}{_RESET}")
            return "<ERROR>"

    @property
    def st(self):
        """Return the marker-stripped string representation of the data."""
        return self.strip_markers()

    def __str__(self):
        """Return the marker-stripped string representation of the data."""
        return self.strip_markers()

    def clean(self):
        """Remove all path markers from `self._str` and return the cleaned string."""
        cleaned_string = re.sub(r" <<<<<<<< [^\n]*", "", self._str)
        self._str = cleaned_string
        return cleaned_string

    def mark_field(self, *paths: str):
        """Mark specific fields in the formatted string to prevent them from being removed by `strip_markers`.

        This method searches for lines containing the long marker " <<<<<<<<<<<< ".
        If the provided `path` argument matches one of the comma-separated identifiers
        that follow this long marker on a line, the long marker (" <<<<<<<<<<<< ")
        is replaced with a short marker (" <<<<<<<< ") for that specific instance.

        `strip_markers` only removes long markers and their associated paths.
        By changing a long marker to a short one, this method ensures that the
        field and its path information are preserved when `strip_markers` is called.
        This is useful for highlighting or retaining specific paths in the output.

        Args:
            *paths (str): One or more path identifiers to mark.
        """
        if not paths:
            return self._str

        print(f"{_GRAY}Finding {paths} to mark...{_RESET}")

        def replacer_logic(match_obj: re.Match[str]):
            path_identifiers_blob = match_obj.group(1)

            # Split the blob into individual path identifiers
            individual_identifiers_on_line = [
                identifier.strip() for identifier in path_identifiers_blob.split(",") if identifier.strip() in paths
            ]

            if individual_identifiers_on_line:
                print(f"Marking {individual_identifiers_on_line}")
                return f"  <<<<<<<< {individual_identifiers_on_line}"
            else:
                return ""  # Strip other markers

        return re.sub(r" <<<<<<<<<<<< ([^\n]*)", replacer_logic, self._str)

    def strip_markers(self, string: str | None = None):
        """Remove all <<<<<<<<<<<< markers from the formatted string.

        Args:
            string (str, optional): The formatted string to remove markers from. Defaults to `self._str`.
        Returns:
            out (str): A new string with markers removed.
        """
        cleaned_string = re.sub(r" <<<<<<<<<<<< [^\n]*", "", string or self._str)
        return cleaned_string
