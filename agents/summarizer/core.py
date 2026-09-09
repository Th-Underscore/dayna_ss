"""Summarizer engine core (T4 extraction).

Summarizer composes the extracted mixins (LLM client, instruction generator;
context engine and population follow) and retains orchestration:
summarization turn phases, subject processing, scene-transition bookkeeping,
chunk saving, and the shared support classes below.
"""
from typing import Any, Callable, Generator, TextIO
from os import PathLike
import copy
from collections import deque
import threading
import traceback
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from ...runtime import runtime

# from ...utils.memory_management import VRAMManager
from ...ui import get_update_queue, PhaseManager
from ...rag.structured_rag.context_retriever import (
    RetrievalContext,
    StoryContextRetriever,
)

from ...utils.helpers import (
    _ERROR,
    _SUCCESS,
    _INPUT,
    _GRAY,
    _HILITE,
    _BOLD,
    _WARNING,
    _RESET,
    _DEBUG,
    History,
    load_json,
    save_json,
    expand_lists_in_data_for_llm,
    get_values,
    strip_thinking,
    strip_response,
    format_str_or_jinja,
    _get_jinja_env,
)

from ...utils.schema_parser import SchemaParser, ParsedSchemaClass

from ...tools.definitions.dynamic_tools import create_dss_tool_executors, create_dss_tool_definitions
from ...tools.tool_registry import Tool, ToolRegistry


# Instruction-block engine (repetition guard, rings, context blocks) lives in
# ./instruction_blocks.py (T3 extraction; pure functions).
from ..instruction_blocks import (  # noqa: E402
    _SENT_SPLIT_RE,
    _INSTR_RING_MAXLEN,
    _INSTR_ANTI_REPEAT_DIRECTIVE,
    _block_collides,
    _ring_blocks,
    _extract_props,
    _recently_used_block,
    _collect_known_names,
    _own_replies_block,
    _current_scene_recap,
)


class DualStream:
    def __init__(self, primary: TextIO, secondary: TextIO):
        self.primary = primary
        self.secondary = secondary

    def write(self, data):
        self.primary.write(data)
        self.secondary.write(data)

    def flush(self):
        self.primary.flush()
        self.secondary.flush()


@dataclass
class SummarizationContextCache:
    history_path: Path
    state: dict
    custom_state: dict
    context: tuple[RetrievalContext, StoryContextRetriever, int, str]
    original_seed: int
    schema_parser: SchemaParser
    history_length: int | None = None
    is_new_scene_turn: bool = False
    is_new_scene_auto_detected: bool = False
    new_scene_start_node: int | None = None
    detected_new_entities: list | None = None
    force_next_chapter: bool = False
    force_next_arc: bool = False




class _LockedPhaseManager:
    """One shared PhaseManager with every mutating call serialized under one RLock.

    The DataSummarizer subject loop can run concurrently (max_subject_workers > 1).
    PhaseManager is not thread-safe (single-slot _active_phase/_active_step, unlocked
    step appends), so all mutations go through this wrapper. The UI still shows every
    subject phase independently because the update queue tracks phases by id.
    """

    _MUTATING = {
        "start_phase", "done_phase", "error_phase", "skip_phase",
        "start_step", "update_step", "warn_step", "done_step",
        "start_turn", "end_turn", "end_session", "separator",
    }

    def __init__(self, pm: PhaseManager):
        self._pm = pm
        self._lock = threading.RLock()

    def _locked(self, name):
        def wrapper(*args, **kwargs):
            with self._lock:
                return getattr(self._pm, name)(*args, **kwargs)
        return wrapper

    def __getattr__(self, name):
        if name in self._MUTATING:
            return self._locked(name)
        return getattr(self._pm, name)


def _build_worker_all_subjects(all_subjects_data: dict, subject_name: str) -> dict:
    """Frozen snapshot of OTHER subjects; this subject's dict stays LIVE.

    The worker mutates only its own subject's dict in place, so the original
    all_subjects_data (also the main DataSummarizer's) reflects every subject's final
    state after the barrier — post-loop chapter/arc checks need no merging, and the
    worker's own reads match serial exactly. Copying the other subjects removes the
    concurrent dict-mutation/iteration crash risk.
    """
    snap = copy.deepcopy(all_subjects_data)
    snap[subject_name] = all_subjects_data[subject_name]
    return snap


from .llm_client import LLMClientMixin
from .instruction_generator import InstructionGeneratorMixin
from .context_engine import ContextEngineMixin, base_state  # noqa: F401 (re-export)
from .population import PopulationMixin
from ..formatted_data import FormattedData, MessageSummarizer  # noqa: F401 (re-export)


class Summarizer(LLMClientMixin, InstructionGeneratorMixin, ContextEngineMixin, PopulationMixin):
    def __init__(self, config_path: PathLike | None = None, phase_manager: PhaseManager | None = None):
        """
        Create a Summarizer and initialize its configuration, tool registry, and UI integration.

        If `config_path` is provided, load configuration from that path; otherwise load the default
        `dss_config.json` from the extension root. Initializes internal state used by the summarizer:
        - `self.last` (context cache),
        - tool executors and the tool registry,
        - real-time UI update queue and phase manager.

        Parameters:
            config_path (PathLike | None): Path to a JSON configuration file. When `None`, the
                default configuration at the extension root (`runtime.extension_dir / "dss_config.json"`)
                is loaded.
        """
        dss_dir = runtime.extension_dir  # Root directory of the extension
        self.config = self._load_config(config_path or dss_dir / "dss_config.json")

        # self.vram_manager = VRAMManager()
        # # Initialize RAG system
        # self.story_rag = StoryRAG(
        #     collection_prefix="story_summary",
        #     persist_directory=EXTENSION_DIR / "storage" / "vectors"
        # )
        self.last: SummarizationContextCache | None = None
        # Previous turn's instruction block (cross-turn repetition guard in
        # generate_instr_prompt). Lives on the long-lived singleton so the
        # comparison survives across turns in both production and the soak.
        self._prev_instruction: str | None = None
        # Ring of the last few instruction blocks' extracted props (P1/P2:
        # recently-used-imagery list + semantic overlap check). Tuples of
        # (label, props). Single-instance like _prev_instruction so it survives
        # across turns in both production and the soak.
        self._recent_instr_blocks: deque[tuple[str, set[str], str]] = deque(maxlen=_INSTR_RING_MAXLEN)
        self._instr_seq = 0
        # Last successfully-saved non-empty subject state, keyed by subject name.
        # Load-time fallback so a wiped/missing subject file (an update exception
        # or crashed write) can never silently empty a subject for every later turn.
        self._last_good_subjects: dict[str, dict] = {}
        self.dss_tool_executors: dict[str, Callable] = {}
        self.tool_registry = ToolRegistry()
        self._init_tool_registry()

        # Real-time UI update system
        self._update_queue = get_update_queue()
        self._phase_manager = phase_manager or PhaseManager(queue=self._update_queue)

    def _init_tool_registry(self) -> None:
        """Initialize DSS tool executors for TGWUI's native tool system."""
        self.dss_tool_executors = create_dss_tool_executors(self)
        runtime.register_tool_executors(self.dss_tool_executors)

        tool_defs = create_dss_tool_definitions()
        for tool_def in tool_defs:
            func_def = tool_def.get("function", {})
            tool = Tool(
                name=func_def.get("name", ""),
                description=func_def.get("description", ""),
                parameters=[],
                handler=self.dss_tool_executors.get(func_def.get("name")),
            )
            self.tool_registry.register(tool)

        self.tool_registry.set_callbacks(
            on_tool_call=self._on_tool_call,
            on_tool_result=self._on_tool_result,
        )

        print(f"{_SUCCESS}Initialized DSS tool executors: {list(self.dss_tool_executors.keys())}{_RESET}")

    def _on_tool_call(self, tool_name: str, arguments: dict) -> None:
        """Callback when a tool is called."""
        self.log_activity("Tool Call", f"{tool_name}({arguments})", "info")
        print(f"{_DEBUG}Tool call: {tool_name} with args {arguments}{_RESET}")

    def _on_tool_result(self, tool_name: str, result: Any, error: str | None) -> None:
        """Callback when a tool result is ready."""
        if error:
            self.log_activity("Tool Error", f"{tool_name}: {error}", "error")
            print(f"{_ERROR}Tool error: {tool_name}: {error}{_RESET}")
        else:
            result_preview = str(result)[:100] + "..." if len(str(result)) > 100 else str(result)
            self.log_activity("Tool Result", f"{tool_name}: {result_preview}", "success")
            print(f"{_DEBUG}Tool result: {tool_name}: {result_preview}{_RESET}")

    def log_activity(self, event: str, details: str = "", level: str = "info") -> None:
        """Log an activity to the activity logger.

        Args:
            event: Short name of the event
            details: Additional details
            level: Log level - "info", "success", "warning", "error"
        """
        runtime.activity_logger.log(event, details, level)

    def _load_config(self, config_path: PathLike) -> dict:
        """Load summarizer configuration from a JSON file at `config_path`."""
        config = load_json(config_path) or {}
        defaults = {
            "retrieval_mode": "passive",
            "max_tool_calls_per_turn": 5,
            "tool_call_stopping_strings": ["UNCHANGED", "NO_UPDATE"],
            "default_summarization_params": {"max_length": 150},
            "max_subject_workers": 3,
            "max_scene_part_messages": 12,
            "min_scene_part_messages": 4,
        }
        for key, value in defaults.items():
            if key not in config:
                config[key] = value
        return config

    @property
    def retrieval_mode(self) -> str:
        """Get current retrieval mode: 'passive' or 'active'."""
        return self.config.get("retrieval_mode", "passive")

    @retrieval_mode.setter
    def retrieval_mode(self, mode: str) -> None:
        """Set retrieval mode."""
        if mode not in ("passive", "active"):
            raise ValueError(f"Invalid retrieval mode: {mode}. Must be 'passive' or 'active'.")
        self.config["retrieval_mode"] = mode
        self.log_activity("Retrieval Mode", f"Switched to {mode} mode", "info")
        print(f"{_HILITE}Retrieval mode set to: {mode}{_RESET}")

    def save_message_chunks(self, message: str, index: int, current_timestamp: str, path: Path | None = None, do_determine_speakers: bool = True) -> None:
        """Save message chunks to the history path with timestamp."""
        print(f"{_BOLD}save_message_chunks{_RESET} Path: {path}, Index: {index}, Timestamp: {current_timestamp}")
        if not path:
            if not self.last or not self.last.history_path:
                print(f"{_ERROR}History path not set in save_message_chunks{_RESET}")
                raise ValueError("History path not set")
            path = self.last.history_path

        try:
            if not self.last or not self.last.context:
                print(f"{_ERROR}Summarizer.last.context not available for MessageChunker init in save_message_chunks.{_RESET}")
                raise RuntimeError("Summarizer.last.context not available for MessageChunker initialization.")

            context_retriever = self.last.context[1]
            if not isinstance(context_retriever, StoryContextRetriever):
                raise TypeError(f"Expected StoryContextRetriever, got {type(context_retriever)}")

            chunker = context_retriever.chunker
            chunks = chunker.process_message(message, index, current_timestamp, do_determine_speakers=do_determine_speakers)
            print(f"{_SUCCESS}Stored {len(chunks)} message chunks for index {index}{_RESET}")
        except Exception as e:
            print(f"{_ERROR}Error processing message chunks for index {index}: {str(e)}{_RESET}")
            traceback.print_exc()

    def update_previous_message_speakers(self, current_message_idx: int) -> bool:
        """Update speakers for the previous message (current_message_idx - 1) using current state.

        Args:
            current_message_idx: The current message index (e.g., from output). The previous message index
                is calculated as current_message_idx - 1.

        Returns:
            bool: True if update succeeded, False otherwise.
        """
        previous_idx = current_message_idx - 1
        print(f"{_BOLD}update_previous_message_speakers{_RESET} Previous index: {previous_idx}")

        try:
            if not self.last or not self.last.context:
                print(f"{_ERROR}Summarizer.last.context not available for update_previous_message_speakers.{_RESET}")
                return False

            context_retriever = self.last.context[1]
            if not isinstance(context_retriever, StoryContextRetriever):
                raise TypeError(f"Expected StoryContextRetriever, got {type(context_retriever)}")

            chunker = context_retriever.chunker
            success = chunker.update_message_speakers(previous_idx)
            if success:
                print(f"{_SUCCESS}Updated speakers for previous message index {previous_idx}{_RESET}")
            else:
                print(f"{_DEBUG}Could not update speakers for previous message index {previous_idx}.{_RESET}")
            return success
        except Exception as e:
            print(f"{_ERROR}Error updating speakers for previous message index {previous_idx}: {str(e)}{_RESET}")
            traceback.print_exc()
            return False

    def _ensure_summarization_phases(self, subject_names: list[str]) -> None:
        """Build the summarization phase list, register unseen phases, and start the turn."""
        pm = self._phase_manager
        summarization_phases = [
            {"id": "context", "name": "Context Preparation", "weight": 1},
        ]
        for subject in subject_names:
            summarization_phases.append({"id": subject.lower().replace(" ", "_"), "name": subject, "weight": 2})
        summarization_phases.extend([
            {"id": "chapter_check", "name": "Chapter Boundary Check", "weight": 1},
            {"id": "arc_check", "name": "Arc Boundary Check", "weight": 1},
            {"id": "message_summary", "name": "Message Summarization", "weight": 1},
            {"id": "chunking", "name": "Message Chunking", "weight": 1},
        ])

        pm.start_turn("Summarization")
        for p in summarization_phases:
            if p["id"] not in pm._phase_lookup:
                phase = {"id": p["id"], "name": p["name"], "weight": p.get("weight", 1)}
                pm._phases.append(phase)
                pm._phase_lookup[p["id"]] = phase
                pm._phase_steps[p["id"]] = []
                pm._total_weight += phase["weight"]

    def _load_all_subjects_data(self, last_history_path: Path) -> tuple[dict, list[str]]:
        """Load every schema subject's data from disk, restoring wiped files from the last known good state."""
        all_subjects_data = {}
        missing_schemas = []

        # Step 1: Dynamically load the data for every subject defined in the schema.
        for subject_name in self.last.schema_parser.subjects:
            # Check if the schema for this subject exists before proceeding
            if not self.last.schema_parser.get_subject_class(subject_name):
                missing_schemas.append(subject_name)
                continue

            subject_path = last_history_path / f"{subject_name}.json"
            all_subjects_data[subject_name] = load_json(subject_path) or {}
            # Wipe guard: if the file is missing/empty but a previous turn
            # saved non-empty data for this subject, restore it instead of
            # letting an empty subject propagate through the copy-forward for
            # the rest of the run (see the t13 characters wipe in
            # cozy_mystery__dded7733: 10 chars -> {} for 15+ turns).
            if not all_subjects_data[subject_name] and self._last_good_subjects.get(subject_name):
                all_subjects_data[subject_name] = copy.deepcopy(self._last_good_subjects[subject_name])
                print(
                    f"{_WARNING}RESTORED {subject_name} from last known good state "
                    f"({len(all_subjects_data[subject_name])} entries) — subject file was missing or empty."
                )
        print(f"{_DEBUG}All subjects data: {all_subjects_data.keys()}{_RESET}")
        return all_subjects_data, missing_schemas

    def _auto_detect_scene_transition(self, user_input: str, output: str, history: History, custom_state: dict, all_subjects_data: dict) -> None:
        """Auto-detect a scene transition on non-scene turns (budget split / cooldown floor / LLM check)."""
        pm = self._phase_manager
        if self.last and not self.last.is_new_scene_turn:
            pm.start_phase("scene_detection", "Scene Transition Detection")
            self.log_activity("Scene Detection", "Checking for scene transition...", "info")

            # Hard message-budget split: if the current scene part has run long
            # enough (max_scene_part_messages), force a scene boundary so the
            # on_new_scene triggers (characters add_new, scene archive, etc.)
            # fire on a regular cadence instead of only on natural transitions.
            max_part_msgs = int(self.config.get("max_scene_part_messages", 0) or 0)
            min_part_msgs = int(self.config.get("min_scene_part_messages", 0) or 0)
            cs_data = all_subjects_data.get("current_scene") or {}
            part_start_msg = int(cs_data.get("_scene_start_message") or 0)
            msgs_in_part = (len(history) * 2) - part_start_msg
            if max_part_msgs and msgs_in_part >= max_part_msgs:
                self.last.is_new_scene_turn = True
                self.last.is_new_scene_auto_detected = True
                part_no = int(cs_data.get("_scene_number") or 1)
                self.log_activity(
                    "Scene Detection",
                    f"Budget split: {msgs_in_part} msgs >= {max_part_msgs}; archiving scene part {part_no}",
                    "warning",
                )
                print(
                    f"{_DEBUG}Budget scene split: {msgs_in_part} messages in current part "
                    f"(>= {max_part_msgs}); forcing new scene part.{_RESET}"
                )
            elif min_part_msgs and msgs_in_part < min_part_msgs:
                # Cooldown floor: never archive more than once per
                # min_scene_part_messages messages, so a trigger-happy
                # transition check can't fire the heavy on_new_scene pass
                # on back-to-back turns.
                self.log_activity(
                    "Scene Detection",
                    f"Cooldown: {msgs_in_part} msgs < {min_part_msgs}; skipping LLM transition check",
                    "info",
                )
            else:
                # Get recent messages for context (last 4 messages = 2 exchanges)
                recent_history = history[-4:] if len(history) >= 2 else history

                # Ask the LLM if a scene transition occurred
                scene_transition_detected = self._check_scene_transition(
                    user_input, output, recent_history, custom_state
                )

                if scene_transition_detected:
                    self.last.is_new_scene_turn = True
                    self.last.is_new_scene_auto_detected = True
                    self.log_activity("Scene Detection", "Auto-detected new scene transition!", "success")
                    print(f"{_DEBUG}Auto-detected scene transition in last exchange(s){_RESET}")
                else:
                    self.log_activity("Scene Detection", "No scene transition detected", "info")

            pm.done_phase("scene_detection")


    def _stamp_new_scene_meta(self, all_subjects_data: dict, history: History) -> None:
        """On a scene turn, advance scene numbering and stamp scene-start/chapter/arc metadata."""
        if self.last and self.last.is_new_scene_turn:
            if "current_scene" in all_subjects_data:
                events_data = all_subjects_data.get("events", {})
                scenes_count = len(events_data.get("scenes", {})) if events_data else 0
                new_scene_number = scenes_count + 1
                current_scene_number = all_subjects_data["current_scene"].get("_scene_number")
                if current_scene_number is not None:
                    new_scene_number = current_scene_number + 1
                all_subjects_data["current_scene"]["_scene_number"] = new_scene_number
                print(f"{_DEBUG}Setting '_scene_number' to {new_scene_number} for new scene.{_RESET}")

                # Track the message index where this scene part begins so the
                # hard message-budget split can count messages per part.
                all_subjects_data["current_scene"]["_scene_start_message"] = max(
                    0, (len(history) * 2) - 2
                )

                # Defensive: stamp the new scene's opening message node on
                # start.when so the scene-bounded dialogue window and the
                # rolling-summaries scene-key always resolve, even if the
                # LLM's scene-start rewrite (scene_start_update_prompt_template)
                # answers NO_UPDATES_REQUIRED. Same value as new_scene_start_node
                # computed in prepare_context (history there excludes the current
                # exchange, so len(history[:-1])*2 == len(history)*2 - 2 here).
                _cs_data = all_subjects_data["current_scene"]
                _start_block = _cs_data.get("start")
                if not isinstance(_start_block, dict):
                    _start_block = {}
                    _cs_data["start"] = _start_block
                _start_when = _start_block.get("when")
                if not isinstance(_start_when, dict):
                    _start_when = {}
                    _start_block["when"] = _start_when
                _start_when["_message_node"] = f"{max(0, (len(history) * 2) - 2)}_1_1"
                print(f"{_DEBUG}Stamped current_scene.start.when._message_node = {_start_when['_message_node']}{_RESET}")

                # Initialize _chapter_number if not set
                if "_chapter_number" not in all_subjects_data["current_scene"]:
                    all_subjects_data["current_scene"]["_chapter_number"] = 1
                    print(f"{_DEBUG}Setting initial '_chapter_number' to 1.{_RESET}")

                # Initialize _arc_number if not set
                if "_arc_number" not in all_subjects_data["current_scene"]:
                    all_subjects_data["current_scene"]["_arc_number"] = 1
                    print(f"{_DEBUG}Setting initial '_arc_number' to 1.{_RESET}")


    def _maybe_populate_first_scene(self, user_input: str, output: str, state: dict, last_history_path: Path, new_history_path: Path, all_subjects_data: dict, has_archived_scenes: bool) -> dict:
        """Run guarded first-scene initial population exactly once per session; returns the (possibly reloaded) subject data."""
        first_scene_marker = ".populated_from_first_scene"
        populated_before = (
            (last_history_path / first_scene_marker).exists()
            or (new_history_path / first_scene_marker).exists()
            or bool((all_subjects_data.get("general_info") or {}).get("_populated_from_first_scene"))
        )
        if (
            self.last
            and self.last.is_new_scene_turn
            and not has_archived_scenes
            and not populated_before
        ):
            # Preserve scene/chapter/arc numbers set by the scene turn handler above,
            # since population will reload all_subjects_data from disk.
            if "current_scene" in all_subjects_data:
                preserved_scene_meta = {
                    k: all_subjects_data["current_scene"][k]
                    for k in ("_scene_number", "_chapter_number", "_arc_number", "_scene_start_message")
                    if k in all_subjects_data["current_scene"]
                }
            else:
                preserved_scene_meta = {}

            self._populate_from_first_scene(user_input, output, state, last_history_path)
            try:
                (last_history_path / first_scene_marker).write_text("")
                (new_history_path / first_scene_marker).write_text("")
            except Exception as e:
                print(f"{_WARNING}Could not write first-scene marker: {e}{_RESET}")
            all_subjects_data = {}
            for subject_name in self.last.schema_parser.subjects:  # TODO: Use in-memory data from population instead of reloading from disk
                subject_path = last_history_path / f"{subject_name}.json"
                all_subjects_data[subject_name] = load_json(subject_path) or {}
            if preserved_scene_meta and "current_scene" in all_subjects_data:
                all_subjects_data["current_scene"].update(preserved_scene_meta)

            # Initialize the scene-part start marker on the population turn
            # (population creates current_scene without it).
            cs_data = all_subjects_data.get("current_scene")
            if cs_data is not None and "_scene_start_message" not in cs_data:
                cs_data["_scene_start_message"] = max(0, (len(state["history"]["internal"]) * 2) - 2)

            # Persist the population marker inside general_info.json so it
            # survives history-dir rotation (the marker file itself is not
            # copied forward each turn). The DataSummarizer's GeneralInfo
            # processing carries this key into every subsequent history dir.
            gi_data = all_subjects_data.get("general_info")
            if gi_data is not None:
                gi_data["_populated_from_first_scene"] = True
                try:
                    save_json(gi_data, last_history_path / "general_info.json")
                except Exception as e:
                    print(f"{_WARNING}Could not persist population marker in general_info: {e}{_RESET}")

        return all_subjects_data

    def _copy_static_session_files(self, last_history_path: Path, new_history_path: Path) -> None:
        """Copy static schema/template files forward to the new history path."""
        save_json(
            load_json(last_history_path / "subjects_schema.json"),
            new_history_path / "subjects_schema.json",
        )
        save_json(
            load_json(last_history_path / "format_templates.json"),
            new_history_path / "format_templates.json",
        )


    def _process_all_subjects(self, data_summarizer, all_subjects_data: dict, custom_state: dict, new_history_path: Path, exchange: tuple[str, str], history: History) -> tuple[dict | None, bool]:
        """Process every subject serially or in parallel; returns (processed_data, stopped)."""
        pm = self._phase_manager
        user_input, output = exchange
        def process_subject_update(subject_name: str, data: dict, schema_class: ParsedSchemaClass) -> dict:
            if runtime.stop_everything:
                print(f"{_HILITE}Stop signal received during subject update for {subject_name}.{_RESET}")
                save_json(data, new_history_path / f"{subject_name}.json")
                return data

            return data_summarizer.generate(subject_name, data, schema_class)  # Triggers are handled by DataSummarizer... probably
            # print(f"{_BOLD}{subject_name}{_RESET} {data_summarizer._should_update_subject(schema_class)}")
            # if data_summarizer._should_update_subject(schema_class):
            #     return data_summarizer.generate(subject_name, data, schema_class)
            # else:
            #     save_json(data, new_history_path / f"{subject_name}.json")
            #     return data

        # Step 2: Dynamically process each subject.
        processed_subjects_data = {}
        total_subjects = len(all_subjects_data)
        max_workers = max(0, int(self.config.get("max_subject_workers", 1)))

        # Build the run list in schema order, skipping missing schemas.
        subjects_to_run = []
        for subject_name, subject_data in all_subjects_data.items():
            schema_class = self.last.schema_parser.get_subject_class(subject_name)
            print(f"{_BOLD}Processing subject: {subject_name}{_RESET} {schema_class}")
            if not schema_class:  # Redundant but good for safety
                pm.skip_phase(subject_name.lower().replace(" ", "_"), "Schema not found", subject_name)
                continue
            subjects_to_run.append((subject_name, subject_data, schema_class))

        if max_workers > 1 and len(subjects_to_run) > 1:
            # --- Parallel subject processing ---
            # Each subject is independent (mutates only its own dict, writes only
            # its own {subject}.json), so subjects run concurrently when the backend
            # accepts concurrent requests. NOTE (LMDeploy fork): concurrency 4+
            # segfaults the Turbomind BlockTrie when prefix caching is enabled;
            # keep workers <= 3 with prefix caching, or higher with
            # --disable-prefix-caching.
            locked_pm = _LockedPhaseManager(pm)
            for name, _, _ in subjects_to_run:
                locked_pm.start_phase(name.lower().replace(" ", "_"), name)
            try:
                with ThreadPoolExecutor(max_workers=min(max_workers, len(subjects_to_run))) as pool:
                    futures = {}
                    for name, data, schema in subjects_to_run:
                        futures[pool.submit(
                            self._process_subject_parallel,
                            name, data, schema,
                            _build_worker_all_subjects(all_subjects_data, name),
                            copy.deepcopy(custom_state),
                            new_history_path,
                            (user_input, output),
                            locked_pm,
                            history,
                        )] = name
                    for fut in as_completed(futures):
                        name = futures[fut]
                        processed_subjects_data[name] = fut.result()
                        locked_pm.done_phase(name.lower().replace(" ", "_"), name)
                        self.log_activity("Subject Updated", name, "success")
            except Exception as e:
                for name, _, _ in subjects_to_run:
                    try:
                        pm.error_phase(name.lower().replace(" ", "_"), str(e), name)
                    except Exception:
                        pass
                raise
            # Deterministic key order: schema order, not completion order.
            processed_subjects_data = {
                n: processed_subjects_data[n] for n, _, _ in subjects_to_run
                if n in processed_subjects_data
            }
            if runtime.stop_everything:
                pm.end_turn()
                pm.end_session(publish=False)
                return None
        else:
            # --- Serial subject processing (original behavior) ---
            for idx, (subject_name, subject_data, schema_class) in enumerate(subjects_to_run, start=1):
                phase_id = subject_name.lower().replace(" ", "_")
                pm.start_phase(phase_id, subject_name)
                print(f"{subject_name} exists")

                self.log_activity("Update Subject", f"{subject_name} ({idx}/{total_subjects})", "info")
                try:
                    updated_data = process_subject_update(subject_name, subject_data, schema_class)
                    processed_subjects_data[subject_name] = updated_data
                    pm.done_phase(phase_id, subject_name)
                except Exception as e:
                    pm.error_phase(phase_id, str(e), subject_name)
                    raise

                self.log_activity("Subject Updated", subject_name, "success")

                if runtime.stop_everything:
                    pm.end_turn()
                    pm.end_session(publish=False)
                    return None

        return processed_subjects_data, False

    def _run_boundary_checks(self, data_summarizer) -> bool:
        """Run chapter/arc boundary checks on a scene turn; returns True when the run was stopped."""
        pm = self._phase_manager
        if self.last and self.last.is_new_scene_turn:
            pm.start_phase("chapter_check", "Chapter Boundary Check")
            self.log_activity("Chapter Check", "Checking chapter boundary", "info")
            data_summarizer.check_and_archive_chapter()
            if runtime.stop_everything:
                pm.done_phase("chapter_check", "Stopped")
                pm.end_turn()
                pm.end_session(publish=False)
                return None
            self.log_activity("Chapter Check", "Complete", "success")
            pm.done_phase("chapter_check")

            pm.start_phase("arc_check", "Arc Boundary Check")
            self.log_activity("Arc Check", "Checking arc boundary", "info")
            data_summarizer.check_and_archive_arc()
            if runtime.stop_everything:
                pm.done_phase("arc_check", "Stopped")
                pm.end_turn()
                pm.end_session(publish=False)
                return None
            self.log_activity("Arc Check", "Complete", "success")
            pm.done_phase("arc_check")
        else:
            pm.skip_phase("chapter_check", "Not a scene transition")
            pm.skip_phase("arc_check", "Not a scene transition")

        return False

    def _update_chunk_scene_event_metadata(self, processed_subjects_data: dict) -> None:
        """Update scene_id/event_id metadata on stored message chunks from processed events data."""
        pm = self._phase_manager
        pm.start_phase("chunking", "Message Chunking")
        pm.start_step("chunking", "update_metadata", "Updating chunk metadata...")
        if self.last and self.last.context:
            context_retriever = self.last.context[1]
            persist_dir = context_retriever.history_path / "message_index"
            chunker_instance = context_retriever.chunker
            events_data = processed_subjects_data.get("events", {})

            # Process scenes from the processed events_data
            for scene_info in get_values(events_data.get("scenes", {})):
                scene_id = scene_info.get("name")
                scene_start_node_str = scene_info.get("start", {}).get("_message_node", "")
                scene_end_node_str = scene_info.get("end", {}).get("_message_node", "")

                if scene_id and scene_start_node_str and scene_end_node_str:
                    try:
                        start_msg_idx = int(scene_start_node_str.split("_")[0])
                        end_msg_idx = int(scene_end_node_str.split("_")[0])

                        for msg_idx_to_update in range(start_msg_idx, end_msg_idx + 1):
                            chunker_instance.update_node_metadata_by_message_idx(
                                msg_idx_to_update, {"scene_id": scene_id}, persist_dir=persist_dir
                            )
                    except (ValueError, IndexError) as e:
                        print(f"{_ERROR}Could not parse message nodes for scene '{scene_id}': {e}{_RESET}")

            # Process events from the processed events_data
            for event_info in get_values(events_data.get("events", {})):
                event_id = event_info.get("name")
                event_start_node_str = event_info.get("start", {}).get("_message_node", "")
                event_end_node_str = event_info.get("end", {}).get("_message_node", "")

                if event_id and event_start_node_str and event_end_node_str:
                    try:
                        start_msg_idx = int(event_start_node_str.split("_")[0])
                        end_msg_idx = int(event_end_node_str.split("_")[0])

                        for msg_idx_to_update in range(start_msg_idx, end_msg_idx + 1):
                            chunker_instance.update_node_metadata_by_message_idx(
                                msg_idx_to_update, {"event_id": event_id}, persist_dir=persist_dir
                            )
                    except (ValueError, IndexError) as e:
                        print(f"{_ERROR}Could not parse message nodes for event '{event_id}': {e}{_RESET}")

        else:
            print(f"{_ERROR}Cannot update scene/event IDs for chunks: Summarizer.last.context not available.{_RESET}")

        pm.done_step("chunking", "update_metadata", "Metadata updated")
        pm.done_phase("chunking")

    def summarize_latest_state(self, output: str, user_input: str, state: dict, history: History) -> str:  # After output
        """
        Summarizes the latest user/assistant exchange into the session's structured subject data and message chunks.

        Prepares retrieval context, runs subject-level summarization and any chapter/arc boundary checks needed for a new scene, generates a concise message summary saved as one or more message chunks, and updates chunk metadata (scene_id and event_id) based on processed events. The method manages PhaseManager phases for each major step and ends the phase session on completion, early stop, or error.

        Parameters:
            output (str): The assistant's text output to be summarized.
            user_input (str): The user's input corresponding to the output.
            state (dict): Current session/generation state used for context and persistence.
            history (History): Conversation history (list-like of exchanges); the last entry is the exchange being summarized.

        Returns:
            str or None: ISO-8601 timestamp string associated with the processed message (derived from scene time when available) on success, or `None` if summarization was aborted or failed.
        """
        print(f"{_HILITE}summarize_message{_RESET}")
        self.log_activity("Summarizing", "Processing latest exchange", "info")

        pm = self._phase_manager

        try:
            subject_names = list(self.last.schema_parser.subjects.keys()) if self.last and self.last.schema_parser else []

            self._ensure_summarization_phases(subject_names)

            pm.start_phase("context", "Context Preparation")
            user_input, custom_state_ref = self.prepare_context(user_input, state, history[:-1])
            history[-1][0] = user_input  # TODO: Persist next_scene state for this history_path
            if runtime.stop_everything:
                print(f"{_HILITE}Stop signal received after retrieve_and_format_context in summarize_latest_state.{_RESET}")
                pm.done_phase("context", "Stopped")
                pm.end_turn()
                pm.end_session(publish=False)
                return None

            if not self.last or not self.last.history_path:
                print(f"{_ERROR}Summarizer.last.history_path not available after prepare_context{_RESET}")
                pm.done_phase("context", "Missing history path")
                pm.end_turn()
                pm.end_session(publish=False)
                return None
            last_history_path = self.last.history_path
            new_history_path = self.retrieve_history_path(state, history)
            if not new_history_path.exists():
                new_history_path.mkdir(parents=True)
            self.backtrack_history(history, new_history_path)

            from ..data_summarizer import DataSummarizer

            output = strip_thinking(output)

            custom_state = copy.deepcopy(custom_state_ref)
            custom_state["history"]["internal"].append(
                [f"What was the very last exchange?", f"{self.format_dialogue(state, [[user_input, output]])}"]
            )

            events_data = (
                self.last.context[0].events_full
                if self.last and self.last.context and getattr(self.last.context[0], "events_full", None)
                else (self.last.context[0].events if self.last and self.last.context else {})
            )
            has_archived_scenes = bool(events_data.get("scenes", {}))

            all_subjects_data, missing_schemas = self._load_all_subjects_data(last_history_path)

            if missing_schemas:
                print(
                    f"{_ERROR}Could not find required schema definitions for: {missing_schemas}. Aborting summarization.{_RESET}"
                )
                pm.done_phase("context", "Missing schemas")
                pm.end_turn()
                pm.end_session(publish=False)
                return None

            self._auto_detect_scene_transition(user_input, output, history, custom_state, all_subjects_data)

            self._stamp_new_scene_meta(all_subjects_data, history)

            all_subjects_data = self._maybe_populate_first_scene(
                user_input, output, state, last_history_path, new_history_path, all_subjects_data, has_archived_scenes
            )

            data_summarizer = DataSummarizer(
                self, (user_input, output), custom_state, new_history_path, self.last.schema_parser, all_subjects_data, pm,
                real_history=history,
            )

            pm.done_phase("context")

            print(f"{_BOLD}Dynamically summarizing data for all subjects using DataSummarizer...{_RESET}")

            self._copy_static_session_files(last_history_path, new_history_path)

            processed_subjects_data, stopped = self._process_all_subjects(
                data_summarizer, all_subjects_data, custom_state, new_history_path, (user_input, output), history
            )
            if stopped:
                return None

            if self._run_boundary_checks(data_summarizer):
                return None

            # --- Summarize New Messages ---
            pm.start_phase("message_summary", "Message Summarization")
            pm.start_step("message_summary", "summarize", "Preparing message summary...")
            message_idx = len(history) * 2 - 1  # history was passed as history[:-1] to retrieve_and_format_context

            # Determine current timestamp from the processed current_scene data
            current_timestamp_str = datetime.now().isoformat()
            current_scene_data = processed_subjects_data.get("current_scene", {})
            if current_scene_data:
                scene_time_data = current_scene_data.get("now", {}).get("when", {})
                if scene_time_data.get("specific_time") and scene_time_data.get("date"):
                    current_timestamp_str = f"{scene_time_data['date']}T{scene_time_data['specific_time']}"
                elif scene_time_data.get("date"):
                    current_timestamp_str = scene_time_data["date"]

            self.log_activity("Summarize Messages", f"Message index: {message_idx}", "info")
            msg_summarizer = MessageSummarizer(self, new_history_path, current_timestamp_str)
            msg_summarizer.generate((user_input, output), (message_idx - 1, message_idx))
            if runtime.stop_everything:
                pm.done_step("message_summary", "summarize", "Stopped")
                pm.done_phase("message_summary", "Stopped")
                pm.end_turn()
                pm.end_session(publish=False)
                return None
            pm.done_step("message_summary", "summarize", "Message chunks saved")
            self.log_activity("Messages Summarized", "Message chunks saved", "success")
            pm.done_phase("message_summary")

            self._update_chunk_scene_event_metadata(processed_subjects_data)

            self.log_activity("Summarization Complete", f"Scene saved at {new_history_path.name}", "success")
            pm.end_session(publish=True)
            return current_timestamp_str

        except Exception as e:
            print(f"{_ERROR}Error during summarization or metadata update: {str(e)}{_RESET}")
            self.log_activity("Summarization Failed", str(e), "error")
            traceback.print_exc()
            if pm.active_phase:
                pm.error_phase(pm.active_phase, str(e))
            pm.end_turn()
            pm.end_session(publish=False)
            return None

    def _process_subject_parallel(
        self,
        subject_name: str,
        data: dict,
        schema_class: ParsedSchemaClass,
        worker_state: dict,
        worker_custom_state: dict,
        history_path: Path,
        exchange: tuple[str, str],
        locked_pm: "_LockedPhaseManager",
        real_history: list | None = None,
    ) -> dict:
        """Parallel subject worker: one DataSummarizer per thread.

        Mirrors process_subject_update's stop semantics; the LLM-call path and the
        DataSummarizer both early-return on runtime.stop_everything. The worker's own
        subject dict in ``worker_state`` is the live object (see
        ``_build_worker_all_subjects``), so its writes land in the shared state for the
        post-loop chapter/arc checks.
        """
        if runtime.stop_everything:
            save_json(data, history_path / f"{subject_name}.json")
            return data
        # Deferred import: data_summarizer imports Summarizer at module level
        # (circular), so DataSummarizer is imported locally like in
        # summarize_latest_state.
        from ..data_summarizer import DataSummarizer
        ds = DataSummarizer(
            self,
            exchange,
            worker_custom_state,
            history_path,
            self.last.schema_parser,
            worker_state,
            locked_pm,
            real_history=real_history,
        )
        return ds.generate(subject_name, data, schema_class)

    def backtrack_history(self, history: History, history_path: Path) -> bool:
        """
        Attempt to reconcile and repair a session's history on disk by backtracking through prior state and restoring or copying missing/inconsistent history files.

        This is a placeholder hook intended to:
        - detect discrepancies between the in-memory `history` and files under `history_path`,
        - create or restore any missing subject/state files, and
        - return whether any changes were made.

        Parameters:
            history (History): In-memory history list of exchanges for the session.
            history_path (Path): Path to the session's history directory on disk.

        Returns:
            bool: `True` if backtracking made or persisted any changes to disk, `False` if no changes were necessary.

        Note:
            The current implementation is a no-op and should be implemented to perform the reconciliation described above.
        """
        pass

    @staticmethod
    def format_number(num: int):
        """Convert an integer to a pretty string. Currently, it just stringifies the number."""
        return str(num)

    def format_dialogue(self, state: dict, partial_history: History):
        """Format `partial_history` into a dialogue string using the model's Jinja template from `state`."""
        # # Copied from modules.chat
        # from functools import partial
        # from jinja2.sandbox import ImmutableSandboxedEnvironment

        # jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)

        # chat_template = jinja_env.from_string(state["chat_template_str"])
        # chat_renderer = partial(
        #     chat_template.render,
        #     add_generation_prompt=False,
        #     name1=state["name1"],
        #     name2=state["name2"],
        # )
        # messages = []
        # for exchange in partial_history:
        #     if exchange[0] and exchange[0] != "<|BEGIN-VISIBLE-CHAT|>":
        #         messages.append({"role": "user", "content": exchange[0]})
        #     if exchange[1]:
        #         messages.append({"role": "assistant", "content": exchange[1]})
        # return chat_renderer(messages=messages)

        name1 = state["name1"]
        name2 = state["name2"]
        messages = []
        length = len(partial_history)
        i = -1
        [["<|BEGIN-VISIBLE-CHAT|>", "5"], ["4", "3"], ["2", "1"]]
        for exchange in partial_history:
            if exchange[0]:
                i += 1
                if exchange[0] != "<|BEGIN-VISIBLE-CHAT|>":
                    messages.append(f"{length*2-i}. '{name1}' >> {exchange[0]}")
            if exchange[1]:
                i += 1
                messages.append(f"{length*2-i}. '{name2}' >> {exchange[1]}")
        return "\n\n".join(messages)

