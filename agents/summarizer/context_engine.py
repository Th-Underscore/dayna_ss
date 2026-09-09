"""Context retrieval and formatting engine (T4 extraction).

prepare_context, retrieve_and_format_context, get_retrieval_context, the
scene-boundary / rolling-summary window helpers, history-path lookup and
scene-transition detection. Methods live on ContextEngineMixin mixed into
Summarizer.
"""
from __future__ import annotations

import copy
import hashlib
import json
import jsonc
import random
import re
import shutil
import time
import traceback
from os import PathLike
from pathlib import Path
from typing import Any, Iterator

from ...runtime import runtime

from ...utils.helpers import (
    _ERROR,
    _INPUT,
    _SUCCESS,
    _WARNING,
    _GRAY,
    _HILITE,
    _BOLD,
    _RESET,
    _DEBUG,
    History,
    load_json,
    save_json,
    format_str_or_jinja,
)

from ...rag.structured_rag.context_retriever import RetrievalContext, StoryContextRetriever
from ...utils.schema_parser import SchemaParser
from ..formatted_data import FormattedData

if False:  # forward refs only
    from .core import Summarizer


# TODO: Get base_state gen params from config (ui_parameters)
base_state = {
    "name1": "SYSTEM",
    "name2": "DAYNA",
    "mode": "instruct",
    "chat-instruct_command": 'Continue the chat dialogue below. Write a single reply for the character "DAYNA". Answer questions flawlessly. Follow instructions to a T.\n\n<|prompt|>',
    "context": (
        "You are DAYNA, an advanced AI assistant integrated into a comprehensive story-writing and world-building environment. Your primary function is to act as a collaborative partner, generating responses that continue a narrative based on a rich, structured context.\n\n"
        "This context is provided in several parts:\n\n"
        "1.  **General Info:** An overview of the story's world, plot, and writing style.\n"
        "2.  **Current Scene:** Detailed information about the immediate setting, characters present, time, and circumstances. This is the most immediate and relevant context for your next response.\n"
        "3.  **Relevant Characters & Groups:** Detailed descriptions, relationships, and statuses of characters and groups pertinent to the current interaction.\n"
        "4.  **Relevant Events:** Summaries of past or ongoing events that influence the current situation.\n"
        "5.  **Relevant Messages:** Specific dialogue snippets from earlier in the story that have been identified as relevant.\n"
        "6.  **Recent Dialogue:** The last few exchanges in the conversation to ensure continuity.\n\n"
        "Your instructions are delivered by the SYSTEM. You must follow them precisely. Your goal is to generate a natural, in-character response for your designated persona that seamlessly continues the story, respecting all the provided context and instructions. You are creative, adaptable, and capable of writing in diverse styles and tones.\n\n"
        "CRITICAL STYLE CONSTRAINTS:\n"
        "- AVOID: 'It wasn't X, it was Y' structures (negative parallelism).\n"
        "- AVOID: 'hidden part,' 'something shifted,' 'a weight settled,' 'quietly [verb]' phrases.\n"
        "- AVOID: Ending scenes with sudden emotional realizations or 'newfound' understanding.\n"
        "- AVOID: Excessive em dashes (—) or colons (:) linking clauses.\n"
        "- AVOID: 'Here's the thing,' 'Here's the kicker,' 'Here's where it gets interesting,' 'At the end of the day.'\n"
        "- AVOID: False suspense transitions like 'Not X. Not Y. Just Z.' or rhetorical questions 'The result? Devastating.'\n"
        "- AVOID: 'In conclusion,' 'To sum up,' 'In summary' (signposted conclusions).\n"
        "- AVOID: Repetitive metaphors, similes, or wordplay that get beaten into the ground across a scene.\n"
        "- AVOID: 'Game changer,' 'double-edged sword,' 'tip of the iceberg,' 'perfect storm' (clichéd idioms).\n"
        "- AVOID: Cliffhangers or artificially dramatic endings to individual messages. Each response should flow naturally into the next, feeling like part of a continuous scene, not a self-contained one-shot.\n"
        "- PREFER: Rough edges, unexpected word choices, varied sentence lengths (very short mixed with long), sensory details (sound, smell, touch).\n"
        "- TONE: Show the scene through direct action and specific sensory experience, not through narrator reflection or summary."
    ),
    "auto_max_new_tokens": True,
    "temperature": 0.3,
    "truncation_length": 65536,
    "history": {"internal": [["<|BEGIN-VISIBLE-CHAT|>", "I am ready to receive instructions!"]]},
}


def compute_stale_entities(
    events_scenes: dict,
    current_scene_who: dict,
    window: int = 6,
) -> set[str]:
    """Names of entities eligible for roster-line demotion (Canon Synopsis P0).

    An entity is stale when ALL hold:
    - it has no cast credit in the live scene (current_scene.now.who), and
    - it has no participant credit in any of the last `window` archived scenes.

    Pure render-time staleness: computed fresh each turn from stored shapes,
    nothing persisted, instantly reversed by re-entry (design doc decision 3).
    Importance is deliberately NOT consulted — staleness forces the floor tier
    regardless of score; importance still ranks entities inside the fresh set.
    """
    fresh = _collect_fresh_names(current_scene_who)
    if isinstance(events_scenes, dict):
        scenes = [s for s in events_scenes.values() if isinstance(s, dict)]
        for scene in scenes[-max(0, window):]:
            fresh.update(_iter_entity_names(scene.get("participants")))
    else:
        scenes = []

    stale: set[str] = set()
    for scene in scenes:
        for name in _iter_entity_names(scene.get("participants")):
            if name not in fresh:
                stale.add(name)
    return stale


def _collect_fresh_names(current_scene_who) -> set[str]:
    """Cast names from current_scene.now.who ({characters/groups/elements: [{name}]})."""
    fresh: set[str] = set()
    if isinstance(current_scene_who, dict):
        for group in current_scene_who.values():
            if isinstance(group, dict):
                for member_list in group.values():
                    fresh.update(_iter_entity_names(member_list))
            elif isinstance(group, list):
                fresh.update(_iter_entity_names(group))
    return fresh


def _iter_entity_names(participants) -> Iterator[str]:
    """Yield every entity name credited in one scene's participants shape."""
    if isinstance(participants, dict):
        for k in participants.keys():
            yield str(k)
    elif isinstance(participants, list):
        for p in participants:
            if isinstance(p, dict) and p.get("name"):
                yield str(p["name"])
            elif isinstance(p, str):
                yield p


class ContextEngineMixin:
    def prepare_context(self, user_input: str, state: dict, history: History, **kwargs):
        """Retrieve and format context for the prompt, as well as detecting a new scene turn.

        Returns:
            out (tuple[str, dict]): A tuple of (user_input, custom_state)
        """
        print(f"{_BOLD}prepare_context{_RESET}")
        custom_state = self.retrieve_and_format_context(state, history, **kwargs)
        if not self.last:
            print(f"{_ERROR}Summarizer.last not available in prepare_context.{_RESET}")
            raise RuntimeError("Summarizer.last not available in prepare_context.")

        if runtime.persistent_ui_state.get("next_scene", False):
            self.last.is_new_scene_turn = True
        is_new_scene_auto = False

        next_scene_prefix = "NEXT SCENE:"  # NEW SCENE:
        if user_input.startswith(next_scene_prefix):
            print(f"{_DEBUG}Found '{next_scene_prefix}' in user input in prepare_context.{_RESET}")
            user_input = user_input[len(next_scene_prefix) :].lstrip()
            self.last.is_new_scene_turn = True
            self.last.is_new_scene_auto_detected = False  # Manual trigger

        self._consume_forced_unit_boundaries()

        # --- Auto Scene Detection ---
        if not self.last.is_new_scene_turn and not is_new_scene_auto:
            pass  # Defer to post-processing
            # NOTE: Doesn't update Gradio checkbox

        if self.last.is_new_scene_turn:
            # Message nodes are 1-indexed and user messages are at turn_idx * 2:
            self.last.new_scene_start_node = f"{len(history) * 2}_1_1"
            print(f"{_DEBUG}New scene turn flagged. Start message node for new scene: {self.last.new_scene_start_node}{_RESET}")

        return user_input, custom_state

    def _consume_forced_unit_boundaries(self) -> None:
        """Stage-in forced chapter/arc boundaries onto the current turn's cache.

        chat_input_modifier (production) and the soak harness stage the intent in
        persistent_ui_state instead of setting summarizer.last.* directly:
        get_retrieval_context rebuilds the cache for every exchange (the
        history-path hash changes), which silently dropped flags set on the
        previous object. Consumed here — after recreation, before reply
        generation — the flags survive to _run_boundary_checks because
        summarize_latest_state calls prepare_context with history[:-1] and so
        reuses this same cache object.
        """
        ui_state = runtime.persistent_ui_state
        for key in ("force_next_chapter", "force_next_arc"):
            if ui_state.get(key, False):
                setattr(self.last, key, True)
                self.last.is_new_scene_turn = True
                self.last.is_new_scene_auto_detected = False  # Manual trigger
                ui_state[key] = False
                print(f"{_DEBUG}Forced unit boundary consumed: {key} (scene turn flagged).{_RESET}")


    def _format_general_info_static(self, gi_data: dict) -> str:
        """Format general_info static fields into a concise prose block.

        Renders all non-empty, non-dict fields from the general_info dict as
        label: value lines. This replaces raw ``state[\"context\"]`` in the
        prompt after the first scene completes.
        """
        parts = []
        for key, value in gi_data.items():
            if not value:
                continue
            if isinstance(value, dict):
                continue
            if isinstance(value, list):
                items = [str(v) for v in value if v]
                if not items:
                    continue
                value_str = ", ".join(items)
            else:
                value_str = str(value)
            label = key.replace("_", " ").title()
            parts.append(f"{label}: {value_str}")
        return "\n".join(parts)

    def retrieve_and_format_context(self, state: dict, history: History, **kwargs) -> dict:
        """Retrieve and format context for instructing model based on history.

        This method generates `custom_state` with an artificial history (DAYNA Mode).

        Args:
            state (dict): Original (TGWUI) state

        Returns:
            custom_state (dict): custom_state
        """

        current_context = state["context"]
        custom_state, (retrieval_context, context_retriever, last_x, last_x_messages) = self.get_retrieval_context(
            state, history, current_context, **kwargs
        )
        if not self.last.history_length is None:
            return custom_state

        custom_state.update(copy.deepcopy(base_state))
        custom_history: History = custom_state["history"]["internal"]

        # Check whether any scene has been archived (on-disk truth of first-scene completion).
        # Before the first scene completes, inject raw state["context"] as fallback;
        # after that, the structured general_info fields come from the format template
        # in the context_order loop below — the static label:value render is now only a
        # FALLBACK for runs whose template yields nothing (previously BOTH rendered,
        # duplicating the full synopsis + premise meta twice per prompt).
        has_archived_scenes = bool(
            (getattr(retrieval_context, "events_full", None) or retrieval_context.events).get("scenes", {})
        )
        gi_fallback_render = ""
        if has_archived_scenes:
            gi_fallback_render = self._format_general_info_static(retrieval_context.general_info)
        else:
            custom_state["context"] += f"\n\n{current_context}"

        formatted_last_x = self.format_number(last_x)

        # TODO: Summary of last scene
        # last_scene = context_retriever.get_scene(-1)
        # if last_scene:
        #     formatted_last_scene = FormattedData(last_scene, 'scene').st
        #     custom_history.append(["What happened in the last scene?", formatted_last_scene])

        if not self.last.schema_parser:
            raise RuntimeError("Schema parser not initialized in retrieve_and_format_context.")

        print(f"{_BOLD}Retrieving context for {formatted_last_x} messages:", retrieval_context, _RESET)
        print(f"{_HILITE}RetrievalContext attributes:")
        print(f"  general_info: {retrieval_context.general_info}")
        print(f"  current_scene: {retrieval_context.current_scene}")
        print(f"  characters: {retrieval_context.characters}")
        print(f"  groups: {retrieval_context.groups}")
        print(f"  events: {retrieval_context.events}")
        print(f"  messages: {retrieval_context.messages}")
        print(f"  messages_metadata: {retrieval_context.messages_metadata}{_RESET}")

        session_id = str(self.last.history_path.resolve()) if self.last and self.last.history_path else ""
        context_order = FormattedData.get_context_order(session_id=session_id)
        context_attr_map = {
            "general_info": "general_info",
            "current_scene": "current_scene",
            "character_list": "characters",
            "characters": "characters",
            "groups": "groups",
            "elements": "elements",
            "events": "events",
            "chapters": "chapters",
            "arcs": "arcs",
            "lines": "messages",
        }

        # Mode B (rolling + inline/system retrieval): the four header blocks
        # ([CONTEXT]/[GENERAL_INFO]/[CURRENT_SCENE]/[OTHER RETRIEVED SUBJECTS]) form
        # a single shared system prefix per turn, and the recent dialogue is raw
        # user/assistant pairs instead of an enumerated re-statement. Config keys:
        #   message_mode        "enumerated" (baseline) | "rolling"
        #   retrieval_placement "prompt_start" (baseline) | "system" | "inline"
        use_rolling = self.config.get("message_mode", "enumerated") == "rolling"
        placement = self.config.get("retrieval_placement", "prompt_start")
        block_parts: list[str] = []
        gi_template_rendered = False

        # Canon Synopsis P0: roster-demotion set, computed once per turn and
        # injected for every subject render (templates consult _stale_entities).
        _demote_window = int(self.config.get("demote_after_scenes", 6) or 6)
        _events_for_stale = getattr(retrieval_context, "events_full", None) or retrieval_context.events
        stale_entities = compute_stale_entities(
            (_events_for_stale or {}).get("scenes", {}),
            (retrieval_context.current_scene or {}).get("now", {}).get("who", {}),
            window=_demote_window,
        )

        for item in context_order:
            data_type = item.get("type")
            prompt = item.get("prompt", "")
            to_context = item.get("to_context", False)
            no_prompt = item.get("no_prompt", False)

            attr_name = context_attr_map.get(data_type)
            data = getattr(retrieval_context, attr_name, None) if attr_name else None

            if no_prompt:
                formatted = FormattedData(data if data is not None else {}, data_type, parser=None, context_cache=self.last).st
                if to_context and formatted:
                    custom_state["context"] += f"\n\n{formatted}"
                continue

            if not attr_name:
                continue

            if data is None:
                continue

            messages_metadata = getattr(retrieval_context, "messages_metadata", [])

            if data_type == "lines":
                lines_data = {
                    "messages": data,
                    "metadata": messages_metadata,
                }
                scene_names = getattr(
                    getattr(retrieval_context, "events_full", None) or retrieval_context.events,
                    "get", lambda k, d={}: d.get(k, "Unknown"),
                )("scenes", {})
                scene_name_map = {name.lower(): name for name in scene_names.keys()} if isinstance(scene_names, dict) else {}
                extra_context = {"scene_names": scene_name_map, "metadata": messages_metadata}
                formatted = FormattedData(lines_data, data_type, parser=None, context_cache=self.last, extra_context=extra_context).st
            else:
                extra_context = {"metadata": messages_metadata, "_stale_entities": stale_entities}
                formatted = FormattedData(data, data_type, self.last.schema_parser, context_cache=self.last, extra_context=extra_context).st

            # Empty subjects render as a bare "<EMPTY>" placeholder — sending that
            # verbatim into the prompt is noise (and once anchored a 60K-char
            # misread of the section map). Skip empty renders entirely.
            if formatted and formatted.strip() == "<EMPTY>":
                formatted = ""

            if data_type == "general_info" and formatted:
                gi_template_rendered = True

            if to_context:
                custom_state["context"] += f"\n\n{formatted}"

            if prompt and formatted:
                if placement == "prompt_start":
                    custom_history.append([prompt, formatted])
                elif not to_context:
                    block_parts.append(formatted)

        # Static-label fallback: only when the format template produced no
        # general_info render (keeps a single synopsis/premise copy per prompt).
        if has_archived_scenes and not gi_template_rendered and gi_fallback_render:
            custom_state["context"] += f"\n\n{gi_fallback_render}"

        # The OTHER RETRIEVED SUBJECTS block: system placement joins the shared
        # context prefix (visible to reply AND every DataSummarizer call);
        # inline placement lands as the last history pair, right before the
        # per-call instruction (the generation boundary).
        if placement == "system" and block_parts:
            custom_state["context"] += "\n\nCURRENT CONTEXT (retrieved subjects):\n\n" + "\n\n".join(block_parts)

        # Recent dialogue: rolling = raw pairs from the same scene-bounded window
        # that the enumerated block used; enumerated = the numbered re-statement.
        if use_rolling:
            summary_floor = int(self.config.get("rolling_summaries", 0) or 0)
            if summary_floor > 0:
                # Sticky-roll message summaries: messages BEFORE the rolling
                # window are otherwise dropped entirely. Inject the last N
                # summaries (N = max(floor, last-ROLLING_SUMMARY_SCENE_ROLL
                # scenes), sticky-locked until the next scene turn) as a single
                # pair ahead of the raw recent pairs, oldest first.
                try:
                    retriever = self.last.context[1] if (self.last and self.last.context and len(self.last.context) > 1) else None
                    summaries = self._load_summary_nodes(getattr(retriever, "chunker", None))
                    if summaries:
                        bounds = self._collect_scene_boundaries(retrieval_context)
                        scene_key = self._current_scene_key(retrieval_context)
                        start, end = self._rolling_summary_window(len(history), last_x, bounds, summary_floor, scene_key)
                        if end > start:
                            in_range = [s for s in summaries if start <= s[0] < end]
                            if in_range:
                                custom_history.append(
                                    [
                                        f"Summaries of earlier messages (messages {start}-{end - 1}, before the most recent window):",
                                        self._format_rolling_summaries(in_range),
                                    ]
                                )
                except Exception as e:
                    print(f"{_WARNING}Rolling message summaries unavailable: {e}{_RESET}")
            for exchange in history[-last_x:]:
                if isinstance(exchange, (list, tuple)) and len(exchange) >= 2:
                    custom_history.append([str(exchange[0] or ""), str(exchange[1] or "")])
        else:
            if last_x_messages:
                custom_history.append([f"What were the last {formatted_last_x} exchanges (pairs of messages)?", last_x_messages])

        if placement == "inline" and block_parts:
            custom_history.append(["CURRENT CONTEXT (retrieved subjects):", "\n\n".join(block_parts)])

        # Analysis complete marker closes the Q&A framing; only meaningful when
        # both knobs are at their baseline values.
        if not use_rolling and placement == "prompt_start":
            custom_history.append(
                [
                    "Analyze all of the above information. Confirm when your analysis is complete.",
                    "Analysis complete.",
                ]
            )

        print(f"{_HILITE}FORMATTED CONTEXT {_SUCCESS}{json.dumps(custom_history, indent=2)}{_RESET}")
        self.last.history_length = len(custom_history)
        return custom_state

    def get_retrieval_context(
        self, state: dict, history: History, current_context: str, **kwargs
    ) -> tuple[dict, tuple[RetrievalContext, StoryContextRetriever, int, str]]:
        """Retrieve and initialize context for the current turn.

        Handles new chats, loads initial world data (from cache or generation),
        sets up session history paths, and loads schemas.

        This method generates `self.last` if it is a fresh input/output.

        Args:
            state (dict): Current application state.
            history (History): Chat history.
            current_context (str): General summarization context.
            **kwargs: Additional arguments (e.g., `last_x` for message count).

        Returns:
            tuple: (custom_state, retrieval_context_obj, story_context_retriever,
                    num_last_messages, formatted_last_messages_str).
        """
        # TODO: Make this path configurable or discoverable
        GLOBAL_SUBJECTS_SCHEMA_TEMPLATE_PATH = runtime.extension_dir / self.config.get(
            "subjects_schema", "user_data/example/subjects_schema.json"
        )
        GLOBAL_FORMAT_TEMPLATES_PATH = runtime.extension_dir / "user_data" / "example" / "format_templates.json"

        # Mode B rolling: the recent-dialogue window is capped by the
        # `rolling_window` config (default 6) instead of the enumerated-mode
        # `last_x_max` (8). Still scene-bounded via _scene_dialogue_window.
        if self.config.get("message_mode", "enumerated") == "rolling":
            kwargs.setdefault("last_x_max", int(self.config.get("rolling_window", 6) or 6))

        history_path = self.retrieve_history_path(state, history)
        initial_schema_parser = None
        is_new_scene = False

        if len(history) < 2 and not history_path.exists():  # New chat
            GLOBAL_SUBJECTS_SCHEMA_TEMPLATE_PATH = runtime.extension_dir / self.config.get(
                "subjects_schema", "user_data/example/subjects_schema.json"
            )
            GLOBAL_FORMAT_TEMPLATES_TEMPLATE_PATH = runtime.extension_dir / "user_data" / "example" / "format_templates.json"
            GLOBAL_SCHEMA_PARSER = SchemaParser(GLOBAL_SUBJECTS_SCHEMA_TEMPLATE_PATH)

            print(f"{_BOLD}Fresh chat detected. Initializing...{_RESET}")

            # Phase 0: Determine Initial World Data Path & Check Cache
            char_context = state.get("context", "")
            char_greeting = state["history"]["internal"][0][1]  # state.get("greeting", "")
            user_bio = state.get("user_bio", "")
            cache_content_key_string = char_context + char_greeting + user_bio
            world_data_cache_hash = self.hash_key(cache_content_key_string, precision=24)

            # Ensure dss_shared.current_character is available
            if not runtime.current_character:
                runtime.update_config(state)
                if not runtime.current_character:
                    raise ValueError("runtime.current_character is not set. Cannot determine cache path.")
            initial_world_data_path = runtime.extension_dir / "user_data" / "history" / runtime.current_character / "initial_world_cache" / world_data_cache_hash
            print(f"{_DEBUG}Initial world data cache path: {initial_world_data_path}{_RESET}")

            required_cache_files = [
                "subjects_schema.json",
                "format_templates.json",
                *[f"{subject}.json" for subject in GLOBAL_SCHEMA_PARSER.subjects.keys()],
            ]
            cache_hit = initial_world_data_path.exists() and all(
                (initial_world_data_path / f).exists() for f in required_cache_files
            )

            if cache_hit:
                print(f"{_SUCCESS}Cache hit for initial world data at {initial_world_data_path}{_RESET}")
                # A fresh chat must STILL run first-scene population. The cache only
                # ever holds empty placeholders + seeded general_info (population writes
                # to the session path, never back to the cache), so a cache hit — e.g.
                # crash-then-restart within the same run dir — otherwise skips
                # _populate_from_first_scene and the world stays empty all run.
                is_new_scene = True
            else:
                print(f"{_INPUT}Cache miss for initial world data. Populating cache at {initial_world_data_path}...{_RESET}")
                initial_world_data_path.mkdir(parents=True, exist_ok=True)

                try:
                    with open(initial_world_data_path.parent / "dump.txt", "w", encoding="utf-8") as f:
                        dump_str = str(json.dumps(kwargs, indent=2))
                        dump_str += "\n\n========================== ORIGINAL STATE\n\n"
                        dump_str += str(json.dumps(state, indent=2))
                        dump_str += "\n\n==========================\n"
                        f.write(dump_str)
                        f.close()
                except Exception as e:
                    print(f"{_ERROR}Error writing dump.txt: {str(e)}{_RESET}")
                    traceback.print_exc()

                # Copy global subjects_schema.json to initial_world_data_path
                schema_cache_path = initial_world_data_path / "subjects_schema.json"
                if not GLOBAL_SUBJECTS_SCHEMA_TEMPLATE_PATH.exists():
                    raise FileNotFoundError(
                        f"Global subjects schema template not found at {GLOBAL_SUBJECTS_SCHEMA_TEMPLATE_PATH}"
                    )
                shutil.copy(GLOBAL_SUBJECTS_SCHEMA_TEMPLATE_PATH, schema_cache_path)
                print(f"{_SUCCESS}Copied global schema to {schema_cache_path}{_RESET}")

                format_templates_cache_path = initial_world_data_path / "format_templates.json"
                if not GLOBAL_FORMAT_TEMPLATES_TEMPLATE_PATH.exists():
                    raise FileNotFoundError(
                        f"Global format templates template not found at {GLOBAL_FORMAT_TEMPLATES_TEMPLATE_PATH}"
                    )
                shutil.copy(GLOBAL_FORMAT_TEMPLATES_TEMPLATE_PATH, format_templates_cache_path)

                try:
                    initial_schema_parser = SchemaParser(schema_cache_path)
                except Exception as e:
                    print(f"{_ERROR}Failed to load SchemaParser for initial_world_data_path: {e}{_RESET}")
                    raise

                is_new_scene = True

                # Write empty placeholder files for all subjects (data_summarizer's generate handles empty dicts)
                for subject_name in GLOBAL_SCHEMA_PARSER.subjects:
                    subject_file = f"{subject_name}.json"
                    if subject_name == "general_info":
                        continue  # general_info gets seeded via LLM below
                    save_json({}, initial_world_data_path / subject_file)

                # Seed general_info static fields from state["context"] via LLM
                general_info_config = initial_schema_parser.get_subject_class("general_info").defaults.get("initial_population", {})
                if general_info_config.get("mode") == "direct":
                    self._populate_subject_direct(
                        initial_world_data_path,
                        initial_schema_parser,
                        state,
                        "general_info",
                        general_info_config,
                    )
                print(f"{_SUCCESS}Initial world data (empty placeholders + seeded general_info) written to cache: {initial_world_data_path}{_RESET}")

            # Phase 1: Session history_path Generation & Creation
            history_path.mkdir(parents=True)
            print(f"{_DEBUG}Session specific path for new chat: {history_path}{_RESET}")

            # Phase 2: Initial File Population in the Session history_path
            # Copy from cache (initial_world_data_path) to session_specific_path
            for file_name in required_cache_files:
                if (initial_world_data_path / file_name).exists():
                    shutil.copy(initial_world_data_path / file_name, history_path / file_name)
                else:
                    save_json({}, history_path / file_name)
                    print(f"{_SUCCESS}Created initial {file_name} in {history_path}{_RESET}")

            # Also copy format_templates.json to session history_path
            if GLOBAL_FORMAT_TEMPLATES_PATH.exists():
                shutil.copy(GLOBAL_FORMAT_TEMPLATES_PATH, history_path / "format_templates.json")

            print(f"{_SUCCESS}Copied initial data from cache to session path {history_path}{_RESET}")

        if not history_path.exists():
            history_path.mkdir(parents=True)

        # Ensure essential schema/template files exist in the history path
        if not (history_path / "subjects_schema.json").exists():
            shutil.copy(GLOBAL_SUBJECTS_SCHEMA_TEMPLATE_PATH, history_path / "subjects_schema.json")
            print(f"{_SUCCESS}Copied fallback schema to {history_path / 'subjects_schema.json'}{_RESET}")
        if not (history_path / "format_templates.json").exists() and GLOBAL_FORMAT_TEMPLATES_PATH.exists():
            shutil.copy(GLOBAL_FORMAT_TEMPLATES_PATH, history_path / "format_templates.json")
            print(f"{_SUCCESS}Copied fallback format templates to {history_path / 'format_templates.json'}{_RESET}")

        last = getattr(self, "last", None)
        if not last or (history_path and history_path != last.history_path):
            # Point format templates loader to session-local path
            FormattedData.set_session_templates_path(str(history_path), history_path / "format_templates.json")
            custom_state = copy.deepcopy(state)
            # Initialize schema parser first to get schema classes for entity graph
            schema_parser = initial_schema_parser or SchemaParser(history_path / "subjects_schema.json")
            print(f"{_SUCCESS}Summarizer.schema_parser loaded for {history_path}{_RESET}")

            # Get schema classes for subjects (Characters, Groups, etc.) to pass to EntityGraph
            schema_classes = {
                "Characters": schema_parser.definitions.get("Characters"),
                "Groups": schema_parser.definitions.get("Groups"),
                "Events": schema_parser.definitions.get("Events"),
                "Arcs": schema_parser.definitions.get("Arcs"),
            }
            schema_classes = {k: v for k, v in schema_classes.items() if v}

            context_retriever = StoryContextRetriever(history_path, schema_classes=schema_classes, summarizer=self)

            # Retrieve last x messages (scene-bounded: the recent-dialogue window
            # is capped to the current scene so it cannot reach across a scene
            # boundary, per the TODO below; flat fallback when no boundary).
            last_x = self._scene_dialogue_window(history, context_retriever, kwargs)
            last_x_messages = self.format_dialogue(state, history[-last_x:])

            retrieval_context = context_retriever.retrieve_context(current_context, last_x_messages)

            # Randomize seed before passing to text-generation-webui
            original_seed = state["seed"]
            if original_seed == -1:
                state["seed"] = random.randint(1, 2**31)
                print(f"{_BOLD}New seed for session{_RESET}: {state['seed']}")

            # Deferred: SummarizationContextCache lives in .core, which imports
            # this module for ContextEngineMixin — a module-level import here
            # would be circular.
            from .core import SummarizationContextCache

            self.last = SummarizationContextCache(
                context=(retrieval_context, context_retriever, last_x, last_x_messages),
                state=state,
                custom_state=custom_state,
                history_path=history_path,
                original_seed=original_seed,
                schema_parser=schema_parser,
                is_new_scene_turn=is_new_scene,
            )

        return self.last.custom_state, self.last.context

    # ---- Rolling message summaries (sticky roll) ----
    # Rolling mode drops messages outside the recent window entirely, which
    # strands older story context. When `rolling_summaries` is configured,
    # the LAST N message summaries (from the accumulated message_index store)
    # are injected before the raw rolling pairs, with a sticky roll: the
    # window reaches back at least `floor` messages OR to the start of the
    # last ROLLING_SUMMARY_SCENE_ROLL scenes — whichever reaches further — and
    # the computed start is then LOCKED until the next scene turn. The lock
    # keeps the summaries block byte-stable across turns within a scene so the
    # shared prefix survives for prefix caching (encode once, reuse for every
    # call in the turn AND across turns in the scene). In-memory only: a
    # process restart legitimately recomputes, since the prefix cache is gone
    # with it anyway.
    ROLLING_SUMMARY_SCENE_ROLL = 5

    def _collect_scene_boundaries(self, retrieval_ctx: Any) -> list[int]:
        """Message-idx of each scene's start (archived scenes + current), ascending."""
        if retrieval_ctx is None:
            return []
        bounds: list[int] = []
        try:
            ev = getattr(retrieval_ctx, "events_full", None) or getattr(retrieval_ctx, "events", None) or {}
            scenes = (ev or {}).get("scenes", {})
            if isinstance(scenes, dict):
                for s in scenes.values():
                    if not isinstance(s, dict):
                        continue
                    node = (s.get("start") or {}).get("_message_node", "")
                    if isinstance(node, str) and node:
                        idx = int(str(node).split("_")[0])
                        if idx >= 0:
                            bounds.append(idx)
            cur = getattr(retrieval_ctx, "current_scene", None) or {}
            node = (cur.get("start") or {}).get("_message_node", "") or cur.get("_message_node", "")
            if isinstance(node, str) and node:
                idx = int(str(node).split("_")[0])
                if idx >= 0:
                    bounds.append(idx)
        except Exception:
            return []
        return sorted(set(bounds))

    def _current_scene_key(self, retrieval_ctx: Any) -> str | None:
        """Identity of the current scene (its start message-node, else _scene_number).

        Used to lock the rolling-summaries window until the next scene turn.
        None when no identity is resolvable — the window then recomputes per
        turn instead of locking.
        """
        if retrieval_ctx is None:
            return None
        try:
            cur = getattr(retrieval_ctx, "current_scene", None) or {}
            node = (cur.get("start") or {}).get("_message_node", "") or cur.get("_message_node", "")
            if isinstance(node, str) and node:
                return node
            num = cur.get("_scene_number")
            if num is not None:
                return f"scene-{num}"
        except Exception:
            pass
        return None

    def _rolling_summary_window(self, history_len: int, last_x: int, scene_bounds: list[int],
                                floor: int, scene_key: str | None = None) -> tuple[int, int]:
        """Message-idx range [start, end) of summaries to include.

        ``end`` is where the raw rolling window begins; ``start`` reaches back
        at least ``floor`` messages (``floor*2`` message indices), or to the
        start of the last ROLLING_SUMMARY_SCENE_ROLL scenes — whichever is
        further back. ``start`` is recomputed only when ``scene_key`` changes
        (a new scene turn) and otherwise STICKS, so the summaries block stays
        byte-stable within a scene for prefix caching. With ``scene_key=None``
        (no scene identity resolvable) it recomputes per turn. Returns (0, 0)
        when there is nothing to summarize.
        """
        window_start_msg = 2 * max(0, history_len - last_x)
        if window_start_msg <= 0:
            return (0, 0)
        if scene_key is not None and getattr(self, "_locked_summary_scene", None) == scene_key:
            start = getattr(self, "_locked_summary_start", 0)
        else:
            floor_start = 2 * max(0, history_len - max(1, floor))
            roll_back = (
                scene_bounds[-self.ROLLING_SUMMARY_SCENE_ROLL]
                if len(scene_bounds) >= self.ROLLING_SUMMARY_SCENE_ROLL
                else (scene_bounds[0] if scene_bounds else None)
            )
            start = floor_start if roll_back is None else min(floor_start, roll_back)
            start = max(0, start)
            if scene_key is not None:
                self._locked_summary_start = start
                self._locked_summary_scene = scene_key
        if start >= window_start_msg:
            return (0, 0)
        return (start, window_start_msg)

    def _load_summary_nodes(self, chunker: Any) -> list[tuple[int, str]]:
        """[(message_idx, text)] for the is_summary nodes in the chunker's index,
        ascending, deduped by message_idx (last write wins — the newest summary
        for a given message). None-safe.
        """
        try:
            if chunker is None:
                return []
            index = getattr(chunker, "index", None)
            if index is None:
                return []
            docs = getattr(index, "docstore", None)
            if docs is None:
                return []
            by_idx: dict[int, str] = {}
            for node in getattr(docs, "docs", {}).values():
                meta = getattr(node, "metadata", None) or {}
                if not meta.get("is_summary"):
                    continue
                idx = meta.get("message_idx")
                if not isinstance(idx, int) or idx < 0:
                    continue
                text = getattr(node, "text", None) or ""
                if not isinstance(text, str) or not text.strip():
                    continue
                by_idx[idx] = text.strip()
            return sorted(by_idx.items())
        except Exception:
            return []

    def _format_rolling_summaries(self, summaries: list[tuple[int, str]]) -> str:
        """Render the summaries block: one entry per message, oldest first.

        Each summary is capped at ``rolling_summary_max_chars`` (default 280) at a
        word boundary: 32 near-full paragraphs of ping-pong restatement was ~19.6K
        chars (~4.9K tokens) riding on EVERY call; the grounding beats survive a
        per-entry cap because entity current-state lines already carry the detail.
        """
        if not summaries:
            return ""
        try:
            max_chars = int(self.config.get("rolling_summary_max_chars", 280) or 280)
        except Exception:
            max_chars = 280

        def _cap(text: str) -> str:
            text = " ".join(str(text).split())
            if len(text) <= max_chars:
                return text
            cut = text[:max_chars]
            # back off to the last sentence end inside the cap when one exists
            for sep in (". ", "! ", "? "):
                pos = cut.rfind(sep)
                if pos >= max_chars // 2:
                    return cut[: pos + 1]
            return cut.rsplit(" ", 1)[0] + " \u2026"

        lines = [f"- [message {idx}] {_cap(text)}" for idx, text in summaries]
        return "\n".join(lines)

    def _scene_dialogue_window(self, history: History, context_retriever: Any, kwargs: dict) -> int:
        """Compute the recent-dialogue window (``last_x``) bounded to the current scene.

        Scenes persist their start as a ``_message_node`` (e.g. ``"140_1_1"``); the
        first integer is the 0-based message index of the exchange that opened the
        scene (user messages at even indices ``2*turn``, assistant at ``2*turn+1``),
        so ``message_idx // 2`` is the exchange index where the scene began. The
        window therefore reaches back only to the current scene's start, never
        across a scene boundary. Falls back to the flat default when no boundary
        is resolvable (first scene, or ``_message_node`` absent).

        Returns:
            int: Number of exchanges to include in the recent-dialogue window.
        """
        flat = min(len(history), kwargs.get("last_x", 6))
        scene_start_exchange = None
        try:
            current_scene = context_retriever.get_current_scene()
            start_node = ""
            if current_scene and isinstance(current_scene, dict):
                start_node = (
                    current_scene.get("start", {})
                    .get("when", {})
                    .get("_message_node", "")
                ) or (current_scene.get("start", {}) or {}).get("_message_node", "")
            elif self.last and self.last.new_scene_start_node:
                start_node = str(self.last.new_scene_start_node)
            if start_node:
                message_idx = int(str(start_node).split("_")[0])
                scene_start_exchange = message_idx // 2
        except Exception as e:
            print(f"{_ERROR}Could not resolve scene dialogue window: {str(e)}{_RESET}")
            traceback.print_exc()

        if scene_start_exchange is None:
            print(f"{_DEBUG}No scene boundary found; using flat last_x={flat}{_RESET}")
            return flat

        last_x_max = int(kwargs.get("last_x_max", 8))
        last_x_min = int(kwargs.get("last_x_min", 2))
        if scene_start_exchange >= len(history):
            # Stale/inconsistent boundary at or beyond the current position:
            # nothing meaningful to bound by, so use the whole history (capped).
            last_x = min(len(history), last_x_max)
            print(
                f"{_DEBUG}Scene boundary ({scene_start_exchange}) at/after history len ({len(history)}); "
                f"using last_x={last_x}{_RESET}"
            )
            return last_x
        window = max(last_x_min, len(history) - scene_start_exchange)
        last_x = min(window, last_x_max, len(history))
        print(
            f"{_DEBUG}Scene-bounded last_x: scene_start_exchange={scene_start_exchange} "
            f"history_len={len(history)} -> last_x={last_x}{_RESET}"
        )
        return last_x

    def retrieve_history_path(self, state: dict, history: History) -> Path:
        """Generate a unique history data path based on character, session ID, and history hash."""
        runtime.update_config(state)
        character_path = runtime.extension_dir / "user_data" / "history" / runtime.current_character
        hashed_history_str = self.hash_key(history, precision=24)
        history_path: Path = character_path / state["unique_id"] / hashed_history_str
        print(f'{_HILITE}history_path{_RESET}: "{history_path}"')
        print(f"{_HILITE}history_str {hashed_history_str}{_RESET} {hashed_history_str}")
        return history_path

    def hash_key(self, key: Any, precision: int = 24) -> str:
        """Deterministically hash a key to a hex string of a given precision using the SHA-1 algorithm.

        Args:
            key (Any): The key to hash. This is converted to str() before hashing.
            precision (int, optional): The number of characters to return. Defaults to 24.

        Returns:
            out (str): The hashed string.
        """
        return hashlib.sha1(str(key).encode("utf-8")).hexdigest()[:precision]

    def get_current_scene(self, state: dict):
        """Get the current scene data; initializes if not present."""
        try:
            if not self.current_scene:
                self.current_scene = ""
                return self.current_scene
            return self.current_scene
        except Exception as e:
            print(f"{_ERROR}Error getting current scene: {str(e)}{_RESET}")
            return None

    def _check_scene_transition(
        self,
        user_input: str,
        output: str,
        recent_history: History,
        custom_state: dict,
    ) -> bool:
        """
        Check if a scene transition occurred in the recent messages.

        Asks the LLM to analyze the recent exchange and determine if:
        1. A new scene should begin (setting, time, or location change)
        2. The current scene has ended

        Parameters:
            user_input: The latest user message
            output: The latest bot response
            recent_history: Recent message history for context
            custom_state: The custom state for LLM calls

        Returns:
            bool: True if a scene transition was detected, False otherwise
        """
        # Format recent history for context
        history_str = ""
        for i, msg in enumerate(recent_history):
            role = "User" if i % 2 == 0 else "Assistant"
            content = msg[1] if len(msg) > 1 else ""
            if content:
                history_str += f"{role}: {content[:500]}\n"  # Truncate for prompt efficiency

        prompt = f"""Analyze the following conversation exchange and determine if a SCENE TRANSITION has occurred.

A scene is one continuous stretch of story. A scene transition has occurred ONLY when at least one of these is clearly true from the text:
- The characters have moved to a different, concretely named location (e.g. leaving a building, entering a new area)
- A clearly stated time skip or passage of time has happened (e.g. "the next day", "later that week", a named time jump)
- A major story event has fully concluded and a distinctly new one has begun
- The dramatic action enters a new phase: a new objective is taken up, or the situation's core tension resolves and a different one takes over
- The active character presence changes materially: a principal arrives or departs and the interaction reorganizes around that change

A scene transition has NOT occurred when the story merely continues in the same place, time, cast, and objective. Passing tonal color or mood shifts within the same beat are not transitions; neither is ordinary conversational turn-taking.

Recent conversation (last 2 exchanges):
{history_str}

Latest exchange:
User: {user_input[:500]}
Assistant: {output[:500]}

Respond with ONLY one of these exact responses:
- YES_SCENE_TRANSITION: If a new scene has clearly begun
- NO_SCENE_TRANSITION: If the scene continues normally

When in doubt, respond NO_SCENE_TRANSITION. Only answer YES when a real, textually-evident transition is present."""

        print(f"{_DEBUG}Checking for scene transition...{_RESET}")

        try:
            response_text, _ = self.generate_with_sse(
                prompt,
                custom_state,
                phase_id="scene_detection",
                step_id="scene_check",
                history_path=getattr(self.last, 'history_path', None),
                stopping_strings=["YES_SCENE_TRANSITION", "NO_SCENE_TRANSITION"],
                match_prefix_only=True,
            )

            response_str = str(response_text) if response_text is not None else ""
            response_upper = response_str.strip().upper() if response_str else ""

            # Check the response
            if "YES_SCENE_TRANSITION" in response_upper:
                return True
            elif "NO_SCENE_TRANSITION" in response_upper:
                return False
            else:
                print(f"{_DEBUG}Ambiguous scene detection response: {response_upper[:100]}, defaulting to NO{_RESET}")
                return False

        except Exception as e:
            print(f"{_ERROR}Error in scene transition detection: {e}{_RESET}")
            traceback.print_exc()
            return False
