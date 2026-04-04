from typing import Any, Callable, Generator, TextIO, TYPE_CHECKING
from os import PathLike
import copy
import hashlib
import io
import json
import jsonc
import random
import re
import shutil
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from jinja2.sandbox import ImmutableSandboxedEnvironment

if TYPE_CHECKING:
    from torch import no_grad

from modules import shared
from modules.chat import generate_chat_prompt
from modules.llama_cpp_server import LlamaServer
from modules.text_generation import encode

import extensions.dayna_ss.shared as dss_shared

# from extensions.dayna_ss.utils.memory_management import VRAMManager
from extensions.dayna_ss.ui import get_update_queue, PhaseManager
from extensions.dayna_ss.rag.structured_rag.context_retriever import (
    RetrievalContext,
    StoryContextRetriever,
    MessageChunker,
)

from extensions.dayna_ss.utils.helpers import (
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
    Histories,
    load_json,
    save_json,
    recursive_get,
    expand_lists_in_data_for_llm,
    get_values,
    enumerate_list,
    strip_thinking,
    strip_response,
    format_str_or_jinja,
    _get_jinja_env,
)

from extensions.dayna_ss.utils.schema_parser import SchemaParser, ParsedSchemaClass

from extensions.dayna_ss.utils.background_importer import (
    start_background_import,
    get_imported_attribute,
)

from extensions.dayna_ss.tools.definitions.dynamic_tools import create_dss_tool_executors, create_dss_tool_definitions
from extensions.dayna_ss.tools.tgwui_integration import register_dss_tool_executors
from extensions.dayna_ss.tools.tool_registry import Tool, ToolRegistry

start_background_import("torch", "no_grad")


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
    new_scene_start_node: int | None = None
    detected_new_entities: list | None = None
    force_next_chapter: bool = False
    force_next_arc: bool = False


# TODO: Get base_state gen params from config (ui_parameters)
base_state = {
    "name1": "SYSTEM",
    "name2": "DAYNA",
    "mode": "instruct",
    "chat-instruct_command": 'Continue the chat dialogue below. Write a single reply for the character "DAYNA". Answer questions flawlessly. Follow instructions to a T.\n\n<|prompt|>',
    "enable_thinking": True,
    "context": (
        "You are DAYNA, an advanced AI assistant integrated into a comprehensive story-writing and world-building environment. Your primary function is to act as a collaborative partner, generating responses that continue a narrative based on a rich, structured context.\n\n"
        "This context is provided in several parts:\n\n"
        "1.  **General Info:** An overview of the story's world, plot, and writing style.\n"
        "2.  **Current Scene:** Detailed information about the immediate setting, characters present, time, and circumstances. This is the most immediate and relevant context for your next response.\n"
        "3.  **Relevant Characters & Groups:** Detailed descriptions, relationships, and statuses of characters and groups pertinent to the current interaction.\n"
        "4.  **Relevant Events:** Summaries of past or ongoing events that influence the current situation.\n"
        "5.  **Relevant Messages:** Specific dialogue snippets from earlier in the story that have been identified as relevant.\n"
        "6.  **Recent Dialogue:** The last few exchanges in the conversation to ensure continuity.\n\n"
        "Your instructions are delivered by the SYSTEM. You must follow them precisely. Your goal is to generate a natural, in-character response for your designated persona that seamlessly continues the story, respecting all the provided context and instructions. You are creative, adaptable, and capable of writing in diverse styles and tones."
    ),
    "auto_max_new_tokens": True,
    "temperature": 0.3,
    "truncation_length": 16384,
    "history": {"internal": [["<|BEGIN-VISIBLE-CHAT|>", "I am ready to receive instructions!"]]},
}


class Summarizer:
    def __init__(self, config_path: PathLike | None = None):
        """
        Create a Summarizer and initialize its configuration, tool registry, and UI integration.
        
        If `config_path` is provided, load configuration from that path; otherwise load the default
        `dss_config.json` from the extension root. Initializes internal state used by the summarizer:
        - `self.last` (context cache),
        - tool executors and the tool registry,
        - real-time UI update queue and phase manager.
        
        Parameters:
            config_path (PathLike | None): Path to a JSON configuration file. When `None`, the
                default configuration at the extension root (`extensions/dayna_ss/dss_config.json`)
                is loaded.
        """
        dss_dir = Path(__file__).parent.parent  # Root directory of the extension
        self.config = self._load_config(config_path or dss_dir / "dss_config.json")

        # self.vram_manager = VRAMManager()
        # # Initialize RAG system
        # self.story_rag = StoryRAG(
        #     collection_prefix="story_summary",
        #     persist_directory="extensions/dayna_ss/storage/vectors"
        # )
        self.last: SummarizationContextCache | None = None
        self.dss_tool_executors: dict[str, Callable] = {}
        self.tool_registry = ToolRegistry()
        self._init_tool_registry()

        # Real-time UI update system
        self._update_queue = get_update_queue()
        self._phase_manager = PhaseManager(queue=self._update_queue)

    def _init_tool_registry(self) -> None:
        """Initialize DSS tool executors for TGWUI's native tool system."""
        self.dss_tool_executors = create_dss_tool_executors(self)
        register_dss_tool_executors(self.dss_tool_executors)

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
        dss_shared.activity_logger.log(event, details, level)

    def _load_config(self, config_path: PathLike) -> dict:
        """Load summarizer configuration from a JSON file at `config_path`."""
        config = load_json(config_path) or {}
        defaults = {
            "retrieval_mode": "passive",
            "max_tool_calls_per_turn": 5,
            "tool_call_stopping_strings": ["UNCHANGED", "NO_UPDATE"],
            "default_summarization_params": {"max_length": 150},
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

    def generate_using_tgwui(
        self,
        prompt: str,
        state: dict,
        history_path: Path | None = None,
        stopping_strings: list[str] | None = ["UNCHANGED", "unchanged", "NO_UPDATE", "no_update", '"UNCHANGED"', '"unchanged"', '"NO_UPDATE"', '"no_update"'],
def generate_using_tgwui(
        self,
        prompt: str,
        state: dict,
        history_path: Path | None = None,
        stopping_strings: list[str] | None = ["UNCHANGED", "unchanged", "NO_UPDATE", "no_update", '"UNCHANGED"', '"unchanged"', '"NO_UPDATE"', '"no_update"'],
        match_prefix_only: bool = True,
        **kwargs,
    ) -> tuple[str, str]:
        """
        Generate a response from the configured TGWUI model, using the active tool loop or passive summarization stream based on retrieval mode.
        
        This captures model stderr output to a temporary buffer and appends an internal debug dump (context, history, prompt, final text, and trailing stderr) to a dump.txt file adjacent to the provided history path when available. Generation stops when a configured stopping string is emitted, a tool call handoff occurs, the tool-call limit is reached, or global cancellation is requested.
        
        Parameters:
            prompt (str): The prompt to give to the LLM.
            state (dict): The state dictionary to generate context with.
            history_path (Path | None): Directory used for writing debug dump files; defaults to the last known history path.
            stopping_strings (list[str] | None): Strings that, when produced by the model, signal generation should stop; comparisons may be prefix-only depending on match_prefix_only.
            match_prefix_only (bool): If True, stopping strings are matched only against the start of the generated text (after left-stripping); if False, stopping strings are searched anywhere in the text.
            **kwargs: Additional arguments forwarded to the underlying generation routine.
        
        Returns:
            tuple[str, str]: (response_text, stop_reason) where `response_text` is the final trimmed generated text, and `stop_reason` is the stopping token that terminated generation or an empty string if none.
        """
        **kwargs,
    ) -> tuple[str, str]:
        """
        Generate a response from the configured TGWUI model, using the active tool loop or passive summarization stream based on retrieval mode.
        
        This captures model stderr output to a temporary buffer and appends an internal debug dump (context, history, prompt, final text, and trailing stderr) to a dump.txt file adjacent to the provided history path when available. Generation stops when a configured stopping string is emitted, a tool call handoff occurs, the tool-call limit is reached, or global cancellation is requested.
        
        Parameters:
        	history_path (Path | None): Directory used for writing debug dump files; defaults to the last known history path.
        	stopping_strings (list[str] | None): Strings that, when produced by the model, signal generation should stop; comparisons may be prefix-only depending on match_prefix_only.
        	match_prefix_only (bool): If True, stopping strings are matched only against the start of the generated text (after left-stripping); if False, stopping strings are searched anywhere in the text.
        
        Returns:
        	(response_text, stop_reason): response_text is the final trimmed generated text; stop_reason is the stopping token that terminated generation or an empty string if none.
        """
        if not history_path:
            history_path = self.last.history_path
        try:
            dump_str = (
                f"\n\n==========================\n"
                f"==========================\n"
                f"========================== INTERNAL CONTEXT ({self.hash_key(state['context'])})\n\n"
                f"{state['context']}"
                f"\n\n==========================\n"
                f"========================== INTERNAL HISTORY ({self.hash_key(state['history']['internal'])})\n\n"
                f"{json.dumps(state['history']['internal'], indent=2)}"
                f"\n\n==========================\n"
                f"========================== NEW PROMPT\n\n"
                f"{prompt}"
            )
            # print(f"{_GRAY}{dump_str}{_RESET}")
            with open(history_path.parent / "dump.txt", "a", encoding="utf-8") as f:
                f.write(dump_str)
                f.close()
        except Exception as e:
            print(f"{_ERROR}Error writing to history file: {str(e)}{_RESET}")
            traceback.print_exc()
        text = ""
        stop = ""
        if shared.stop_everything:
            return "", ""

        # Capture the output
        capture_buffer = io.StringIO()
        use_tool_loop = self.retrieval_mode == "active"

        with redirect_stderr(DualStream(sys.stderr, capture_buffer)):  # redirect_stdout(DualStream(sys.stdout, capture_buffer))
            if use_tool_loop:
                for t, s in self.generate_with_tool_loop(
                    prompt,
                    state,
                    stopping_strings,
                    match_prefix_only=match_prefix_only,
                    **kwargs,
                ):
                    if shared.stop_everything:
                        return text, stop
                    text = t
                    stop = s
                    if s and s not in ("tool_call", ""):
                        break
            else:
                for t, s in self.generate_summary_with_streaming(
                    prompt,
                    state,
                    stopping_strings,
                    match_prefix_only=match_prefix_only,
                    **kwargs,
                ):
                    if shared.stop_everything:
                        return text, stop
                    text = t
                    stop = s
        if shared.stop_everything:
            return text, stop
        text = text.strip()
        try:
            string = capture_buffer.getvalue()
            string = string[string.rfind("\r") :].strip()
            dump_str = f"\n\n==========================\n\n{text}\n\n" f"==========================\n\n{string}"
            with open(history_path.parent / "dump.txt", "a", encoding="utf-8") as f:
                f.write(dump_str)
                f.close()
        except Exception as e:
            print(f"{_ERROR}Error writing to history file: {str(e)}{_RESET}")
            traceback.print_exc()
        return text, stop

    def generate_with_sse(
        self,
        prompt: str,
        state: dict,
        phase_id: str,
        step_id: str,
        history_path: Path | None = None,
        stopping_strings: list[str] | None = ["UNCHANGED", "unchanged", "NO_UPDATE", "no_update", '"UNCHANGED"', '"unchanged"', '"NO_UPDATE"', '"no_update"'],
        match_prefix_only: bool = True,
        **kwargs,
    ) -> tuple[str, str]:
        """
        Stream model output as token-level step updates to the UI update queue.
        
        Publishes step lifecycle events (prompt assembly, generation start, incremental token snippets, errors, and completion) to self._update_queue while driving generation either through the tool loop or standard streaming generator. Throttles token emissions to avoid flooding the UI and returns the final aggregated output and the stopping reason.
        
        Parameters:
            prompt (str): The assembled prompt passed to the model.
            state (dict): Runtime state used for generation and context.
            phase_id (str): Identifier for the current PhaseManager phase sent with updates.
            step_id (str): Identifier for the current step sent with updates.
            history_path (Path | None): Optional history path for logging/debug dumps; when None, uses the cached last history path.
            stopping_strings (list[str] | None): Strings that, when produced by the model, are treated as a stop condition.
            match_prefix_only (bool): If true, only matches stopping strings against the start of the generated text.
            **kwargs: Additional arguments forwarded to the underlying generation routine.
        
        Returns:
            tuple[str, str]: `response_text` — the final generated text (trimmed); `stop_reason` — the stopping string that ended generation or an empty string if none.
        """
        if not history_path:
            history_path = self.last.history_path

        text = ""
        stop = ""
        if shared.stop_everything:
            return "", ""

        use_tool_loop = self.retrieval_mode == "active"
        last_emit_len = 0
        emit_threshold = 50  # Emit every ~50 chars to avoid flooding SSE
        last_emit_time = 0
        emit_interval = 0.3  # Also limit to ~3 updates/sec

        # Emit prompt assembly step
        self._update_queue.publish({
            "type": "step_update",
            "phase": {"id": phase_id},
            "step": {"id": step_id, "message": f"Assembling prompt ({len(prompt)} chars)..."},
        })

        try:
            if use_tool_loop:
                gen = self.generate_with_tool_loop(
                    prompt, state, stopping_strings,
                    match_prefix_only=match_prefix_only, **kwargs,
                )
            else:
                gen = self.generate_summary_with_streaming(
                    prompt, state, stopping_strings,
                    match_prefix_only=match_prefix_only, **kwargs,
                )

            # Emit LLM generation started
            self._update_queue.publish({
                "type": "step_update",
                "phase": {"id": phase_id},
                "step": {"id": step_id, "message": "Generating response..."},
            })

            for t, s in gen:
                if shared.stop_everything:
                    return text, stop
                text = t
                stop = s

                # Stream tokens to SSE (throttled)
                import time as _time
                now = _time.time()
                if len(text) - last_emit_len >= emit_threshold and now - last_emit_time >= emit_interval:
                    snippet = text[last_emit_len:]
                    if snippet:
                        self._update_queue.publish({
                            "type": "step_update",
                            "phase": {"id": phase_id},
                            "step": {"id": step_id, "token": snippet, "full_text_len": len(text)},
                            "token": snippet,
                        })
                    last_emit_len = len(text)
                    last_emit_time = now

                if s and s not in ("tool_call", ""):
                    break

        except Exception as e:
            print(f"{_ERROR}Error in generate_with_sse: {str(e)}{_RESET}")
            traceback.print_exc()
            self._update_queue.publish({
                "type": "step_update",
                "phase": {"id": phase_id},
                "step": {"id": step_id, "message": f"Error: {str(e)}"},
            })

        # Emit any remaining text
        if len(text) > last_emit_len:
            snippet = text[last_emit_len:]
            self._update_queue.publish({
                "type": "step_update",
                "phase": {"id": phase_id},
                "step": {"id": step_id, "token": snippet, "full_text_len": len(text)},
                "token": snippet,
            })

        # Emit full response text on completion
        final_text = text.strip()
        self._update_queue.publish({
            "type": "step_update",
            "phase": {"id": phase_id},
            "step": {"id": step_id, "complete": True, "message": final_text},
        })

        return text.strip(), stop

    def generate_summary_with_streaming(
        self,
        prompt: str,
        state: dict,
        stopping_strings: list[str] | None = ["UNCHANGED"],
        match_prefix_only: bool = True,
        **kwargs,
    ) -> Generator[tuple[str, str], Any, None]:
        """
        Stream partial generated text chunks from the configured model and signal when a configured stopping marker is reached.
        
        Yields incremental (text, stop_reason) tuples as the model produces output; when a stopping marker is detected the generator yields the text containing the marker and the matching stopping string as stop_reason and then stops. The generator may yield empty stop_reason for intermediate partial outputs.
        
        Parameters:
            stopping_strings (list[str] | None): List of marker strings that, when detected in the generated text, cause the generator to stop and return that marker as the stop_reason. If None or empty, no automatic stopping based on markers is performed.
            match_prefix_only (bool): If True, a stopping marker is considered matched only when it appears at the start of the generated text after left-stripping whitespace; if False, the marker is matched anywhere in the generated text.
        
        Returns:
            tuple[str, str]: Streamed tuples where the first element is the current generated text chunk and the second element is the stop reason — the matching stopping string when generation ended, or an empty string for ongoing partial outputs.
        """
        # if stopping_strings:
        #     quoted_tokens = [f'"{token}"' for token in stopping_strings]
        #     custom_state['custom_token_bans'] = ', '.join(quoted_tokens) if custom_state['custom_token_bans'] else ', '.join(quoted_tokens)
        try:
            model: LlamaServer = shared.model
            if model is not None:
                if not TYPE_CHECKING:
                    no_grad = get_imported_attribute("torch", "no_grad")
                with no_grad():
                    if shared.stop_everything:
                        return
                    instr_prompt = generate_chat_prompt(prompt, state, **kwargs)
                    encoded_instr_prompt = (
                        encode(instr_prompt, add_bos_token=True) if model.__class__.__name__ != "LlamaServer" else instr_prompt
                    )
                    text = ""
                    token_count = 0
                    if shared.stop_everything:
                        yield text, ""
                        return
                    for text in model.generate_with_streaming(encoded_instr_prompt, state):
                        token_count += 1
                        if shared.stop_everything:
                            yield text, ""
                            return
                        if stopping_strings:
                            text = strip_thinking(text)
                            for stopping_string in stopping_strings:
                                if match_prefix_only:
                                    if text.lstrip().startswith(stopping_string):
                                        yield text, stopping_string
                                        return
                                else:
                                    if stopping_string in text:
                                        yield text, stopping_string
                                        return
                        yield text, ""
                    if shared.stop_everything:
                        return
                    print(f"{_GRAY}Generated summary length: {token_count} ({len(text)}){_RESET}")
        except Exception as e:
            print(f"{_ERROR}Error generating summary: {str(e)}{_RESET}")
            traceback.print_exc()

    def generate_with_tool_loop(
        self,
        prompt: str,
        state: dict,
        stopping_strings: list[str] | None = ["UNCHANGED"],
        match_prefix_only: bool = True,
        max_tool_calls: int | None = None,
        **kwargs,
    ) -> Generator[tuple[str, str], Any, None]:
        """Generate with active tool calling loop.

        In active retrieval mode, the model can call DSS tools to retrieve information
        before generating its response. This method handles the loop of:
        1. Generate text
        2. Check for tool calls
        3. Execute tools and append results
        4. Continue generation

        Args:
            prompt: The initial prompt
            state: The state dictionary for context
            stopping_strings: Strings that stop generation
            match_prefix_only: Only match prefix for stopping strings
            max_tool_calls: Maximum tool calls per turn (default from config)
            **kwargs: Additional arguments passed to streaming generation

        Yields:
            Tuples of (text, stop_reason)
        """
        if max_tool_calls is None:
            max_tool_calls = self.config.get("max_tool_calls_per_turn", 5)

        self.log_activity("Active Retrieval", "Starting tool call loop", "info")
        print(f"{_HILITE}Starting active retrieval mode with max {max_tool_calls} tool calls{_RESET}")

        tool_call_stopping_strings = self.config.get("tool_call_stopping_strings", ["UNCHANGED", "NO_UPDATE"])

        full_response = ""
        tool_call_count = 0
        prompt_history = [prompt]

        while tool_call_count < max_tool_calls:
            if shared.stop_everything:
                yield full_response, ""
                return

            self.log_activity("Generation", f"Turn {tool_call_count + 1}", "info")
            print(f"{_DEBUG}Tool call loop turn {tool_call_count + 1}{_RESET}")

            accumulated_text = ""
            found_tool_call = False

            for text, stop_reason in self.generate_summary_with_streaming(
                prompt_history[-1] if len(prompt_history) > 1 else prompt,
                state,
                stopping_strings=tool_call_stopping_strings,
                match_prefix_only=match_prefix_only,
                **kwargs,
            ):
                accumulated_text = text

                result = self.tool_registry.parse_tool_calls(text)
                if result.status.value in ("complete", "error"):
                    found_tool_call = True
                    tool_call_result = self.execute_tool_result(result)

                    tool_response = self.tool_registry.format_tool_response(tool_call_result)

                    prompt_history.append(
                        f"{text}\n\n{self.tool_registry.TOOL_RESPONSE_OPEN}\n{tool_response}\n{self.tool_registry.TOOL_RESPONSE_CLOSE}"
                    )

                    tool_call_count += 1
                    full_response += text + "\n"
                    yield text, "tool_call"

                    self.log_activity(
                        "Tool Call Complete",
                        f"Call #{tool_call_count}: {result.call.tool_name if result.call else 'unknown'}",
                        "success"
                    )
                    break

                if stop_reason:
                    if stop_reason in (tool_call_stopping_strings or []):
                        found_tool_call = False
                        full_response = text
                        yield text, stop_reason
                        return
                    else:
                        full_response += text
                        yield text, stop_reason
                        return

                yield text, ""

            if not found_tool_call:
                if accumulated_text:
                    full_response = accumulated_text
                break

        if tool_call_count >= max_tool_calls:
            self.log_activity("Tool Limit", f"Reached max tool calls ({max_tool_calls})", "warning")
            print(f"{_WARNING}Reached max tool calls ({max_tool_calls}), continuing without more tool calls{_RESET}")

        full_response = full_response or accumulated_text
        yield full_response, "tool_limit"

    def execute_tool_result(self, result) -> Any:
        """Execute a parsed tool call and return the result.

        Args:
            result: ToolCallResult from parse_tool_calls

        Returns:
            ToolCallResult with output or error
        """
        if result.status.value == "error":
            return result

        if result.call is None:
            return result

        tool_result = self.tool_registry.execute_tool_call(result.call)
        return tool_result

    def save_message_chunks(self, message: str, index: int, current_timestamp: str, path: Path | None = None) -> None:
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
            chunks = chunker.process_message(message, index, current_timestamp)
            print(f"{_SUCCESS}Stored {len(chunks)} message chunks for index {index}{_RESET}")
        except Exception as e:
            print(f"{_ERROR}Error processing message chunks for index {index}: {str(e)}{_RESET}")
            traceback.print_exc()

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

        self.last.is_new_scene_turn = dss_shared.persistent_ui_state.get("next_scene", False)
        next_scene_prefix = "NEXT SCENE:"  # NEW SCENE:
        if user_input.startswith(next_scene_prefix):
            print(f"{_DEBUG}Found '{next_scene_prefix}' in user input in prepare_context.{_RESET}")
            user_input = user_input[len(next_scene_prefix) :].lstrip()
            self.last.is_new_scene_turn = True
            # NOTE: Doesn't update Gradio checkbox

        if self.last.is_new_scene_turn:
            # Message nodes are 1-indexed and user messages are at turn_idx * 2:
            self.last.new_scene_start_node = f"{len(history) * 2}_1_1"
            print(f"{_DEBUG}New scene turn flagged. Start message node for new scene: {self.last.new_scene_start_node}{_RESET}")

        return user_input, custom_state

    def generate_instr_prompt(
        self, user_input: str, state: dict, history: History, **kwargs
    ) -> tuple[str, dict, Path, str]:  # After input
        """
        Builds the instruction prompt used to steer the model's character response and returns that prompt plus a snapshot of state, the history path, and a scene timestamp.
        
        The function prepares retrieval context, optionally generates or loads a cached set of plain-text instruction paragraphs (when `do_instr` is true), composes a final prompt for the character, and encodes it when required by the model. It also records a deep-copied custom state and derives a current-scene timestamp to be used for subsequent summarization and chunking.
        
        Parameters:
            user_input (str): The latest user message to be incorporated into the instruction prompt.
            state (dict): The current session state/configuration (may be mutated internally for seed handling).
            history (History): Conversation history used to compute message indices and the history path.
            **kwargs: Optional flags and generation options. Recognized keys include:
                do_instr (bool): If true, generate detailed instruction paragraphs and persist them to instructions.json.
        
        Returns:
            tuple[str, dict, Path, str]:
                instr_prompt — The instruction prompt ready for model consumption; encoded string when required by the backend, or the original `user_input` on early stop/failure.
                custom_state — A deep-copied snapshot of the state used for generating the instruction prompt.
                history_path — Path to the session's history directory containing cached artifacts.
                current_timestamp_str — ISO-8601 timestamp string for the current scene (derived from retrieval context when available); `None` when generation was stopped or failed.
        
        Side effects:
            - May persist generated instruction text to <history_path>/instructions.json.
            - Writes a diagnostic dump file (dump.txt) next to the history directory.
            - Logs activity and updates phase/step tracking for UI telemetry.
        """
        print(f"{_HILITE}generate_instr_prompt{_RESET} {kwargs}")
        self.log_activity("Generating Instructions", "Preparing context", "info")

        pm = self._phase_manager
        pm.start_phase("instr_prompt", "Instruction Generation")

        pm.start_step("instr_prompt", "context", "Preparing context...")
        user_input, custom_state_ref = self.prepare_context(user_input, state, history, **kwargs)
        pm.done_step("instr_prompt", "context", "Context prepared")
        custom_state = copy.deepcopy(custom_state_ref)
        history_path = self.last.history_path

        if shared.stop_everything:
            print(f"{_HILITE}Stop signal received after prepare_context in generate_instr_prompt.{_RESET}")
            return user_input, state, history_path, None

        # Get current timestamp for saving message chunks
        current_timestamp_str = datetime.now().isoformat()  # Default timestamp
        if self.last and self.last.context:
            retrieval_ctx: RetrievalContext = self.last.context[0]
            if retrieval_ctx and retrieval_ctx.current_scene:
                scene_time_data = retrieval_ctx.current_scene.get("now", retrieval_ctx.current_scene.get("start", {})).get(
                    "when", {}
                )
                if scene_time_data.get("specific_time") and scene_time_data.get("date"):
                    current_timestamp_str = f"{scene_time_data['date']}T{scene_time_data['specific_time']}"
                elif scene_time_data.get("date"):
                    current_timestamp_str: str = scene_time_data["date"]

        user_input_message_idx = len(history) * 2

        if history_path:
            """Get the current state of the story summary."""
            try:
                # # Check if there is a cached KV state for this prompt
                # cached_kv = self.vram_manager.get_context_cache()
                # if cached_kv is not None:
                #     print(f"{_SUCCESS}Found cached KV state{_RESET}")
                #     print(f"{_SUCCESS}Cache ready for current position{_RESET}")

                user_input_prompt = f'This is the latest user input:\n\n"""\n{user_input}\n"""\n\n'
                name1 = state["name1"] or "User"
                name2 = state["name2"] or "Assistant"

                instr_path = history_path / "instructions.json"
                instructions: dict[str, str] = load_json(instr_path)

                # Get shared model (LlamaServer)
                model: LlamaServer = shared.model

                # Use existing instruction prompt
                original_seed = state["seed"]  # Randomize seed before passing to text-generation-webui
                if original_seed == -1:
                    state["seed"] = random.randint(1, 2**31)
                    print(f"{_BOLD}New seed{_RESET}: {state['seed']}")
                self.last.original_seed = original_seed
                input_key = str(state["seed"])

                print(f"{_HILITE}input_key{_RESET}: {input_key}")
                instr = ""
                # input_key = self.hash_key(user_input + shared.model_name)
                if input_key in instructions:
                    instr: str = instructions[input_key]
                    print(f"{_SUCCESS}Found cached instruction prompt{_RESET}")
                    pm.start_step("instr_prompt", "cache_hit", "Loaded cached instructions")
                    pm.done_step("instr_prompt", "cache_hit", f"Loaded {len(instr)} chars from cache")
                else:
                    # Generate prompt using LLM
                    try:
                        if model is not None:
                            # Create custom state for summary generation
                            print(f"{_SUCCESS}State set{_RESET}")

                            if kwargs.get("do_instr", False):
                                # Generate instruction
                                prompt = (
                                    f"{user_input_prompt}\n\n"
                                    f"You are to generate instructions for {name2}'s response to '{name1}'. These instructions will be given directly to {name2}.\n"
                                    f"The instructions must guide {name2} on what to say or do, following the tone of the latest messages, and should be detailed and specific.\n\n"
                                    f"FORMATTING REQUIREMENTS:\n"
                                    f"- Present the instructions as a series of plain text paragraphs.\n"
                                    f"- Each paragraph should represent a distinct part of the response plan.\n"
                                    # f"- CRITICAL: Do NOT use any bold formatting, titles, or headings for these paragraphs. Only the paragraph text itself.\n"
                                    f"Example of desired output structure (imagine these are the instructions):\n"
                                    f"  First, analyze {name1}'s query to understand their core need. Then, formulate a concise opening statement that acknowledges their input.\n"
                                    f"  Next, provide the main information or answer, breaking it down into logical points if necessary. Ensure clarity and accuracy in this section.\n"
                                    f"Remember: The above is an example. In a narrative context, explicitly acknowledging {name1}'s input would break immersion. Additionally, the length of the response should match the established writing style.\n\n"
                                    f"INSTRUCTION CONTENT:\n"
                                    f"1. Explain in detail, step-by-step, in imperative mood, what {name2} should include in their response.\n"
                                    f"2. Be specific, detailing each step.\n"
                                    f"3. You are providing instructions FOR the response, not writing the response itself.\n"
                                    f"4. Address the instructions directly to {name2} (e.g., 'Start by...', 'Then, explain...'). Do not refer to {name2} in the third person (e.g., '{name2} should...').\n"
                                    f"5. Specify the desired length of {name2}'s actual final response (e.g., 'The final response should be one paragraph', 'Aim for two short paragraphs', 'Keep it to three sentences').\n"
                                    f"6. Instruct on the use of dialogue: specify when it is appropriate for characters to speak, which characters should speak, and when narration should be used instead of dialogue.\n"
                                    f"7. IMPORTANT: Remind {name2} not to recap {name1}'s input in the response. Even if necessary to clarify {name1}'s intent, remember: Show, don't tell.\n"
                                    f"8. CRITICAL: Explicitly include an additional instruction on the \"Writing Style\" of the response (taken from e.g. '{name2}: 3rd-person prose, one paragraph per response, no more than one paragraph', '{name2}: 1st-person dialogue as \"{name2}\" with actions in asterisks', '{name2}: 2nd-person prose (speaking to \"{name1}\"), the same number of paragraphs as {name1}').\n\n"
                                    f"REMEMBER: Your entire output must ONLY consist of the instructional paragraphs, adhering strictly to the no-bolding, no-titles format. No extra text, greetings, or sign-offs."
                                )

                                instr, _ = self.generate_with_sse(
                                    prompt=prompt,
                                    state=custom_state,
                                    phase_id="instr_prompt",
                                    step_id="generate_instructions",
                                    history_path=history_path,
                                    match_prefix_only=False,
                                )
                                if shared.stop_everything:
                                    print(f"{_HILITE}Stop signal received after instruction generation.{_RESET}")
                                    pm.done_phase("instr_prompt", "Stopped")
                                    return user_input, state, history_path, None

                                instructions[input_key] = instr
                                print(f"{_HILITE}Instruction:{_RESET} {instr}")
                                save_json(instructions, instr_path)  # Persist instruction prompt for regenerations

                            # # Save the KV cache for this instruction generation if not already saved
                            # self.vram_manager.save_context_cache()
                            # self.vram_manager.increment_position()
                    except Exception as e:
                        print(f"{_ERROR}Error generating instruction: {str(e)}{_RESET}")
                        traceback.print_exc()
                        return user_input, state, history_path, None

                # TODO: Get output gen params from config (ui_parameters)
                # Generate instruction prompt
                instr_prompt = ""
                if kwargs.get("do_instr", False):
                    # TODO: Include additional user instructions from UI blocks
                    full_instr = instr
                    instr_prompt = (
                        f"{user_input_prompt}\n\n"
                        f'You are to write a reply in character as "{name2}".\n'
                        f"The following instructions, presented as plain text paragraphs, outline how you should construct your response:\n\n"
                        f'INSTRUCTIONS TO FOLLOW:\n"""\n{full_instr}\n"""\n\n'
                        f"Adhere loosely to these instructions. Maintain the style and tone consistent with recent messages from both {name1} and {name2}.\n"
                        f"Your reply must be natural-sounding prose.\n\n" #ABSOLUTELY CRITICAL: Do NOT use any formatting whatsoever. This includes, but is not limited to, Markdown, bold text, asterisks for emphasis, headings, or titles. The entire response must be plain, unformatted text, unless it's an organic part of {name2}'s typical speech pattern or dialogue.\n\n"
                        f'REMEMBER: You are "{name2}" replying to "{name1}". Write from {name2}\'s perspective.'
                    )
                else:
                    pm.start_step("instr_prompt", "skip", "Instruction generation disabled")
                    pm.done_step("instr_prompt", "skip", "Skipped")
                    instr_prompt = (
                        f"{user_input_prompt}\n\n"
                        f'You are to write a reply in character as "{name2}".\n'
                        f"Your reply must be natural-sounding prose.\n\n" #ABSOLUTELY CRITICAL: Do NOT use any formatting whatsoever. This includes, but is not limited to, Markdown, bold text, asterisks for emphasis, headings, or titles. The entire response must be plain, unformatted text, unless it's an organic part of {name2}'s typical speech pattern or dialogue.\n\n"
                        f'REMEMBER: You are "{name2}" replying to "{name1}". Write from {name2}\'s perspective.'
                    )
                encoded_instr_prompt = (
                    encode(instr_prompt, add_bos_token=True) if model.__class__.__name__ != "LlamaServer" else instr_prompt
                )
                print(
                    f"{_SUCCESS}Encoded instruct prompt: {True if model.__class__.__name__ != 'LlamaServer' else False}{_RESET}"
                )

                print(f"{_SUCCESS}State set{_RESET}")

                try:
                    with open(history_path.parent / "dump.txt", "w", encoding="utf-8") as f:
                        dump_str = str(json.dumps(kwargs, indent=2))
                        dump_str += "\n\n========================== CUSTOM STATE\n\n"
                        dump_str += str(json.dumps(custom_state, indent=2))
                        dump_str += "\n\n========================== ORIGINAL STATE\n\n"
                        dump_str += str(json.dumps(state, indent=2))
                        dump_str += "\n\n==========================\n\n"
                        dump_str += str(instr)
                        dump_str += "\n\n==========================\n\n"
                        dump_str += str(instr_prompt)
                        dump_str += "\n\n==========================\n"
                        f.write(dump_str)
                        f.close()
                except Exception as e:
                    print(f"{_ERROR}Error writing dump.txt: {str(e)}{_RESET}")
                    traceback.print_exc()

                print(f"{_SUCCESS}Generated instruction prompt{_RESET}")
                self.log_activity("Instructions Ready", f"Path: {history_path.name}", "success")
                pm.done_phase("instr_prompt", "Instructions ready")
                return (
                    encoded_instr_prompt,
                    custom_state,
                    history_path,
                    current_timestamp_str,
                )
            except Exception as e:
                print(f"{_ERROR}Error in get_summary_state: {str(e)}{_RESET}")
                self.log_activity("Instruction Gen Failed", str(e), "error")
                traceback.print_exc()
                return user_input, state, history_path, None

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
        subject_names = list(self.last.schema_parser.subjects.keys()) if self.last and self.last.schema_parser else []
        print(f"{_DEBUG}[DSS] Starting PhaseManager session with subjects: {subject_names}{_RESET}")
        pm.start_session(subject_names=subject_names)
        print(f"{_DEBUG}[DSS] PhaseManager session started. Queue subscribers: {len(self._update_queue._subscribers)}{_RESET}")

        try:
            if shared.stop_everything:
                print(f"{_HILITE}Stop signal received before retrieve_and_format_context in summarize_latest_state.{_RESET}")
                pm.end_session()
                return None

            pm.start_phase("context", "Context Preparation")
            user_input, custom_state_ref = self.prepare_context(user_input, state, history[:-1])
            history[-1][0] = user_input  # TODO: Persist next_scene state for this history_path
            if shared.stop_everything:
                print(f"{_HILITE}Stop signal received after retrieve_and_format_context in summarize_latest_state.{_RESET}")
                pm.end_session()
                return None

            # self.last should be set by prepare_context
            last_history_path = self.last.history_path
            new_history_path = self.retrieve_history_path(state, history)
            if not new_history_path.exists():
                new_history_path.mkdir(parents=True)
            self.backtrack_history(history, new_history_path)

            from extensions.dayna_ss.agents.data_summarizer import DataSummarizer

            output = strip_thinking(output)

            custom_state = copy.deepcopy(custom_state_ref)
            custom_state["history"]["internal"].append(
                [f"What was the very last exchange?", f"{self.format_dialogue(state, [[user_input, output]])}"]
            )

            all_subjects_data = {}
            for subject_name in self.last.schema_parser.subjects:
                subject_path = last_history_path / f"{subject_name}.json"
                all_subjects_data[subject_name] = load_json(subject_path) or {}

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
            print(f"{_DEBUG}All subjects data: {all_subjects_data.keys()}{_RESET}")

            data_summarizer = DataSummarizer(
                self, (user_input, output), custom_state, new_history_path, self.last.schema_parser, all_subjects_data
            )

            if missing_schemas:
                print(
                    f"{_ERROR}Could not find required schema definitions for: {missing_schemas}. Aborting summarization.{_RESET}"
                )
                pm.end_session()
                return None

            # Handle special case for new scene turn before processing
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

                    # Initialize _chapter_number if not set
                    if "_chapter_number" not in all_subjects_data["current_scene"]:
                        all_subjects_data["current_scene"]["_chapter_number"] = 1
                        print(f"{_DEBUG}Setting initial '_chapter_number' to 1.{_RESET}")

                    # Initialize _arc_number if not set
                    if "_arc_number" not in all_subjects_data["current_scene"]:
                        all_subjects_data["current_scene"]["_arc_number"] = 1
                        print(f"{_DEBUG}Setting initial '_arc_number' to 1.{_RESET}")

            pm.done_phase("context")

            print(f"{_BOLD}Dynamically summarizing data for all subjects using DataSummarizer...{_RESET}")

            # Copy static data to the new history path
            save_json(
                load_json(last_history_path / "subjects_schema.json"),
                new_history_path / "subjects_schema.json",
            )
            save_json(
                load_json(last_history_path / "format_templates.json"),
                new_history_path / "format_templates.json",
            )

            def process_subject_update(subject_name: str, data: dict, schema_class: ParsedSchemaClass) -> dict:
                if shared.stop_everything:
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
            for idx, (subject_name, subject_data) in enumerate(all_subjects_data.items(), start=1):
                schema_class = self.last.schema_parser.get_subject_class(subject_name)
                print(f"{_BOLD}Processing subject: {subject_name}{_RESET} {schema_class}")
                if not schema_class:  # Redundant but good for safety
                    pm.skip_phase(subject_name.lower().replace(" ", "_"), "Schema not found", subject_name)
                    continue

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

                if shared.stop_everything:
                    pm.end_session()
                    return None

            if self.last and self.last.is_new_scene_turn:
                pm.start_phase("chapter_check", "Chapter Boundary Check")
                self.log_activity("Chapter Check", "Checking chapter boundary", "info")
                data_summarizer.check_and_archive_chapter()
                self.log_activity("Chapter Check", "Complete", "success")
                pm.done_phase("chapter_check")

                pm.start_phase("arc_check", "Arc Boundary Check")
                self.log_activity("Arc Check", "Checking arc boundary", "info")
                data_summarizer.check_and_archive_arc()
                self.log_activity("Arc Check", "Complete", "success")
                pm.done_phase("arc_check")
            else:
                pm.skip_phase("chapter_check", "Not a scene transition")
                pm.skip_phase("arc_check", "Not a scene transition")

            # --- Summarize New Messages ---
            pm.start_phase("message_summary", "Message Summarization")
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
            self.log_activity("Messages Summarized", "Message chunks saved", "success")
            pm.done_phase("message_summary")

            # --- Update scene_id and event_id for message chunks ---
            pm.start_phase("chunking", "Message Chunking")
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

            pm.done_phase("chunking")
            self.log_activity("Summarization Complete", f"Scene saved at {new_history_path.name}", "success")
            pm.end_session()
            return current_timestamp_str

        except Exception as e:
            print(f"{_ERROR}Error during summarization or metadata update: {str(e)}{_RESET}")
            self.log_activity("Summarization Failed", str(e), "error")
            traceback.print_exc()
            pm.end_session()
            return None

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

    def retrieve_and_format_context(self, state: dict, history: History, **kwargs) -> dict:
        """Retrieve and format context for instructing model based on history.

        This method generates `custom_state` with an artificial history (DAYNA Mode).

        Args:
            state (dict): Original (TGWUI) state

        Returns:
            custom_state (dict): custom_state
        """
        global base_state

        current_context = state["context"]
        custom_state, (retrieval_context, context_retriever, last_x, last_x_messages) = self.get_retrieval_context(
            state, history, current_context, **kwargs
        )
        if not self.last.history_length is None:
            return custom_state

        custom_state.update(copy.deepcopy(base_state))
        custom_history: History = custom_state["history"]["internal"]

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

        context_order = FormattedData.get_context_order()
        context_attr_map = {
            "general_info": "general_info",
            "current_scene": "current_scene",
            "character_list": "characters",
            "characters": "characters",
            "groups": "groups",
            "events": "events",
            "chapters": "chapters",
            "arcs": "arcs",
            "lines": "messages",
        }

        for item in context_order:
            data_type = item.get("type")
            prompt = item.get("prompt", "")
            to_context = item.get("to_context", False)

            attr_name = context_attr_map.get(data_type)
            if not attr_name:
                continue

            data = getattr(retrieval_context, attr_name, None)
            if data is None:
                continue

            if data_type == "lines":
                lines_data = {
                    "messages": data,
                    "metadata": getattr(retrieval_context, "messages_metadata", []),
                }
                scene_names = getattr(retrieval_context.events, "get", lambda k, d={}: d.get(k, "Unknown"))("scenes", {})
                scene_name_map = {name.lower(): name for name in scene_names.keys()} if isinstance(scene_names, dict) else {}
                extra_context = {"scene_names": scene_name_map}
                formatted = FormattedData(lines_data, data_type, parser=None, extra_context=extra_context).st
            else:
                formatted = FormattedData(data, data_type, self.last.schema_parser).st

            if to_context:
                custom_state["context"] += f"\n\n{formatted}"

            if prompt and formatted:
                custom_history.append([prompt, formatted])

        # Append last x messages
        if last_x_messages:
            custom_history.append([f"What were the last {formatted_last_x} exchanges (pairs of messages)?", last_x_messages])

        # Analysis complete marker TODO: Get this SYSTEM prompt from config
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

        history_path = self.retrieve_history_path(state, history)
        initial_schema_parser = None
        is_new_scene = False

        if len(history) < 2 and not history_path.exists():  # New chat
            GLOBAL_SUBJECTS_SCHEMA_TEMPLATE_PATH = Path("extensions/dayna_ss/user_data/example/subjects_schema.json")
            GLOBAL_SCHEMA_PARSER = SchemaParser(GLOBAL_SUBJECTS_SCHEMA_TEMPLATE_PATH)

            print(f"{_BOLD}Fresh chat detected. Initializing...{_RESET}")

            # Phase 0: Determine Initial World Data Path & Check Cache
            char_context = state.get("context", "")
            char_greeting = state["history"]["internal"][0][1]  # state.get("greeting", "")
            user_bio = state.get("user_bio", "")
            cache_content_key_string = char_context + char_greeting + user_bio
            world_data_cache_hash = self.hash_key(cache_content_key_string, precision=24)

            # Ensure dss_shared.current_character is available
            if not dss_shared.current_character:
                dss_shared.update_config(state)
                if not dss_shared.current_character:
                    raise ValueError("dss_shared.current_character is not set. Cannot determine cache path.")

            initial_world_data_path = Path(
                "extensions/dayna_ss/user_data/history",
                dss_shared.current_character,
                "initial_world_cache",
                world_data_cache_hash,
            )
            print(f"{_DEBUG}Initial world data cache path: {initial_world_data_path}{_RESET}")

            required_cache_files = [
                "subjects_schema.json",
                *[f"{subject}.json" for subject in GLOBAL_SCHEMA_PARSER.subjects.keys()],
            ]
            cache_hit = initial_world_data_path.exists() and all(
                (initial_world_data_path / f).exists() for f in required_cache_files
            )

            if cache_hit:
                print(f"{_SUCCESS}Cache hit for initial world data at {initial_world_data_path}{_RESET}")
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

                try:
                    initial_schema_parser = SchemaParser(schema_cache_path)
                except Exception as e:
                    print(f"{_ERROR}Failed to load SchemaParser for initial_world_data_path: {e}{_RESET}")
                    raise

                is_new_scene = True
                # Populate initial data using schema-driven approach
                self._populate_from_schema(initial_world_data_path, initial_schema_parser, state)
                print(f"{_SUCCESS}Initial world data populated in cache: {initial_world_data_path}{_RESET}")

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

            print(f"{_SUCCESS}Copied initial data from cache to session path {history_path}{_RESET}")

        if not history_path.exists():
            history_path.mkdir(parents=True)

        if not self.last or (history_path and history_path != self.last.history_path):
            custom_state = copy.deepcopy(state)
            context_retriever = StoryContextRetriever(history_path)

            # Retrieve last x messages
            last_x = min(len(history), kwargs.get("last_x", 6))  # TODO: Get all in current scene
            last_x_messages = self.format_dialogue(state, history[-last_x:])

            retrieval_context = context_retriever.retrieve_context(current_context, last_x_messages)

            # Randomize seed before passing to text-generation-webui
            original_seed = state["seed"]
            if original_seed == -1:
                state["seed"] = random.randint(1, 2**31)
                print(f"{_BOLD}New seed for session{_RESET}: {state['seed']}")

            # Initialize self.last.schema_parser if it's not already set for this history_path
            schema_parser = initial_schema_parser or SchemaParser(history_path / "subjects_schema.json")
            print(f"{_SUCCESS}Summarizer.schema_parser loaded for {history_path}{_RESET}")

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

    def retrieve_history_path(self, state: dict, history: History) -> Path:
        """Generate a unique history data path based on character, session ID, and history hash."""
        dss_shared.update_config(state)
        character_path = Path("extensions/dayna_ss/user_data/history", dss_shared.current_character)
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

    def format_number(self, num: int):
        """Convert an integer to a pretty string. Currently, it just stringifies the number."""
        return str(num)

    def format_dialogue(self, state: dict, partial_history: History):
        """Format `partial_history` into a dialogue string."""
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
    ) -> None:
        """
        Populates a single subject using direct LLM interaction.

        Args:
            initial_world_data_path: Path to save the populated data
            schema_parser: Schema parser instance
            state: Current state
            subject_name: Name of the subject to populate
            population_config: Configuration dict with 'target_file' and 'prompt_template'
        """
        global base_state

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
                char_context_str=char_context_str,
                char_greeting_str=char_greeting_str,
                all_relevant_definitions_json_str=all_definitions_str,
                example_json=example_json_str,
                retry_feedback_placeholder="",
            )

            custom_state = copy.deepcopy(state)
            custom_state.update(copy.deepcopy(base_state))

            max_retries = 2
            populated_data = None

            for attempt in range(max_retries + 1):
                print(f"{_DEBUG}Attempt {attempt + 1}/{max_retries + 1} to generate '{subject_name}' data...{_RESET}")
                if attempt > 0:
                    print(f"{_INPUT}Retrying LLM prompt for '{subject_name}' with validation feedback.{_RESET}")

                response_text, _ = self.generate_using_tgwui(
                    prompt=prompt,
                    state=custom_state,
                    history_path=initial_world_data_path,
                    match_prefix_only=False,
                )

                if shared.stop_everything:
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
                                char_context_str=char_context_str,
                                char_greeting_str=char_greeting_str,
                                all_relevant_definitions_json_str=all_definitions_str,
                                example_json=example_json_str,
                                retry_feedback_placeholder=error_feedback + "\n",
                            )
                except json.JSONDecodeError as e:
                    print(f"{_ERROR}Failed to parse LLM response for '{subject_name}' as JSON: {e}{_RESET}")
                    print(f"{_ERROR}LLM Raw Response was: {_GRAY}{cleaned_response}{_RESET}")
                    if attempt < max_retries:
                        error_feedback = f"The previous JSON was invalid. Ensure valid JSON output.\n"
                        prompt = format_str_or_jinja(
                            prompt_template,
                            char_context_str=char_context_str,
                            char_greeting_str=char_greeting_str,
                            all_relevant_definitions_json_str=all_definitions_str,
                            example_json=example_json_str,
                            retry_feedback_placeholder=error_feedback,
                        )
                except Exception as e:
                    print(f"{_ERROR}Unexpected error during '{subject_name}' processing on attempt {attempt + 1}: {e}{_RESET}")
                    traceback.print_exc()

            if populated_data:
                save_json(populated_data, target_path)
                print(f"{_SUCCESS}Successfully populated and saved '{target_file}' at {target_path}{_RESET}")
            else:
                print(f"{_ERROR}Failed to populate '{subject_name}' after all retries. Saving empty JSON.{_RESET}")
                save_json({}, target_path)

        except Exception as e:
            print(f"{_ERROR}Error in _populate_subject_direct for '{subject_name}': {e}{_RESET}")
            traceback.print_exc()
            save_json({}, target_path)

    def _populate_subject_identify(
        self,
        initial_world_data_path: Path,
        schema_parser: SchemaParser,
        state: dict,
        subject_name: str,
        population_config: dict,
    ) -> None:
        """
        Populates multiple subjects using identify-then-populate pattern.

        Args:
            initial_world_data_path: Path to save the populated data
            schema_parser: Schema parser instance
            state: Current state
            subject_name: Name of the subject (used as the primary, e.g., "Characters")
            population_config: Configuration dict with 'identification_prompt', 'population_prompt', etc.
        """
        global base_state

        target_files = population_config.get("target_files", [])
        wrapper_key = population_config.get("target_key", "entries")  # Key to wrap entity data (e.g., "entries")
        identification_prompt_template = population_config.get("identification_prompt", "")
        population_prompt_template = population_config.get("population_prompt", "")

        print(f"{_DEBUG}Attempting to populate entities via identification for '{subject_name}'{_RESET}")

        char_context = state.get("context", "")
        char_greeting = state["history"]["internal"][0][1]

        custom_state = copy.deepcopy(state)
        custom_state.update(copy.deepcopy(base_state))

        # Step 1: Identification
        identification_prompt = format_str_or_jinja(
            identification_prompt_template,
            char_context=char_context,
            char_greeting=char_greeting,
        )

        print(f"{_DEBUG}Prompting LLM for entity identification...{_RESET}")
        identification_response_text, _ = self.generate_using_tgwui(
            prompt=identification_prompt,
            state=custom_state,
            history_path=initial_world_data_path,
            match_prefix_only=False,
        )

        if shared.stop_everything:
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

        if not identified_entities:
            print(f"{_INPUT}No entities identified by LLM or parsing failed. Saving empty files.{_RESET}")
            for tf in target_files:
                save_json({}, initial_world_data_path / tf)
            return

        print(f"{_SUCCESS}Identified {len(identified_entities)} entities. Proceeding to detail extraction.{_RESET}")

        # Step 2: Populate each entity
        entity_data = {tf.replace(".json", ""): {} for tf in target_files}

        for entity in identified_entities:
            entity_name: str = entity["name"]
            entity_type: str = entity["type"]
            entity_descriptor: str = entity["descriptor"]

            # Determine which target file to use
            target_key = entity_type + "s"  # "character" -> "characters", "group" -> "groups"
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
                    )

                    custom_state_detail = copy.deepcopy(custom_state)
                    max_retries = 2
                    entity_data_validated = None

                    for attempt in range(max_retries + 1):
                        print(f"{_DEBUG}Attempt {attempt + 1}/{max_retries + 1} to generate details for {entity_type} '{entity_name}'...{_RESET}")
                        if attempt > 0:
                            print(f"{_INPUT}Retrying LLM prompt for '{entity_name}' with validation feedback.{_RESET}")

                        detail_response_text, _ = self.generate_using_tgwui(
                            prompt=population_prompt,
                            state=custom_state_detail,
                            history_path=initial_world_data_path,
                            match_prefix_only=False,
                        )

                        if shared.stop_everything:
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
                                    ) + "\n" + error_feedback
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
                                ) + "\nThe previous JSON was invalid. Ensure valid JSON output.\n"
                        except Exception as e:
                            print(f"{_ERROR}Unexpected error processing '{entity_name}' on attempt {attempt + 1}: {e}{_RESET}")
                            traceback.print_exc()

                    if entity_data_validated:
                        if isinstance(entity_data_validated, dict) and wrapper_key in entity_data_validated:
                            entity_data[target_key].update(entity_data_validated[wrapper_key])
                        else:
                            entity_data[target_key][entity_name] = entity_data_validated
                        print(f"{_SUCCESS}Successfully populated and stored details for {entity_type} '{entity_name}'.{_RESET}")
                    else:
                        print(f"{_ERROR}Failed to populate valid details for {entity_type} '{entity_name}' after all retries.{_RESET}")

                if shared.stop_everything:
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

    def _populate_from_schema(
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
        global base_state

        print(f"{_DEBUG}Starting schema-driven initial population...{_RESET}")

        for subject_name, schema_def in schema_parser.get_subject_classes().items():
            population_config = schema_def.defaults.get("initial_population")
            if not population_config:
                continue

            mode = population_config.get("mode", "direct")
            print(f"{_DEBUG}Populating '{subject_name}' using mode '{mode}'...{_RESET}")

            if mode == "direct":
                self._populate_subject_direct(
                    initial_world_data_path,
                    schema_parser,
                    state,
                    subject_name,
                    population_config,
                )
            elif mode == "identify":
                self._populate_subject_identify(
                    initial_world_data_path,
                    schema_parser,
                    state,
                    subject_name,
                    population_config,
                )
            else:
                print(f"{_WARNING}Unknown population mode '{mode}' for '{subject_name}'. Skipping.{_RESET}")

        print(f"{_SUCCESS}Schema-driven initial population complete.{_RESET}")


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
        """Summarize messages and store in vector database with metadata."""
        print(f"{_BOLD}Summarizing messages for indices {message_idxs}{_RESET}")

        for i, message_content in enumerate(exchange):
            current_message_idx = message_idxs[i]
            prompt = f'''Analyze the provided message and generate a concise summary of the key events, interactions, and developments.

REMEMBER: Do not add anything else to the response. Only respond with the summary.

Here is the message: """\n{message_content.strip()}\n"""'''
            try:
                summary_text, _ = self.summarizer.generate_using_tgwui(prompt, self.custom_state, self.history_path)
                if shared.stop_everything:
                    print(
                        f"{_HILITE}Stop signal received in MessageSummarizer after generating summary for message_idx {current_message_idx}.{_RESET}"
                    )
                    return
                summary_text = strip_thinking(summary_text)

                summary_speakers = ["System"]  # TODO: Derive from context
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
                    ],  # Use 0,0 to indicate this is a summary
                    "timestamp": self.current_timestamp,
                    "speakers": summary_speakers,
                    "characters_present": summary_chars_present,
                    "subjects_referenced": summary_subjects_referenced,
                    "scene_id": None,  # Scene not yet determined
                    "event_id": None,  # Same as scene_id
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

            self.data = expand_lists_in_data_for_llm(data, actual_data_schema_hint, self.parser)
        else:
            self.data = data

        self._str = FormattedData.format_retrieval_data(
            self.data, self.data_type, context_cache=self.context_cache, all_subjects_data=self.all_subjects_data, extra_context=self.extra_context
        )

    def __getitem__(self, index):
        """Allow dictionary-like access to the (potentially expanded) data."""
        return self.data[index]

    _format_templates_cache: dict | None = None

    @staticmethod
    def _load_format_templates() -> dict:
        """Load format templates from the templates JSON file."""
        if FormattedData._format_templates_cache is not None:
            return FormattedData._format_templates_cache

        try:
            dss_dir = Path(__file__).parent.parent
            template_path = dss_dir / "user_data" / "example" / "format_templates.json"
            templates = load_json(template_path) or {}
            FormattedData._format_templates_cache = templates
            print(f"{_DEBUG}Loaded {len(templates)} format templates{_RESET}")
            return templates
        except Exception as e:
            print(f"{_WARNING}Failed to load format templates: {e}{_RESET}")
            return {}

    @staticmethod
    def get_context_order() -> list[dict]:
        """Get the context order configuration from templates."""
        templates = FormattedData._load_format_templates()
        return templates.get("_context_order", [])

    @staticmethod
    def _render_jinja_template(
        template_str: str,
        data: dict | list,
        data_type: str,
        path_prefix: str = "",
        parser=None,
        extra_context: dict | None = None,
    ) -> str:
        """Render a Jinja2 template with the given data.

        Args:
            template_str: The Jinja2 template string
            data: The data to render
            data_type: The data type (used to determine schema class)
            path_prefix: The prefix for path markers
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
            }

            if parser:
                context["defaults"] = parser.defaults

            if extra_context:
                context.update(extra_context)

            rendered = template.render(**context)
            return rendered.strip()

        except Exception as e:
            print(f"{_WARNING}Jinja template rendering failed for {data_type}: {e}{_RESET}")
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
            import extensions.dayna_ss.shared as dss_shared

            user_template = dss_shared.settings.get(f"template_{data_type}")
            if user_template:
                rendered = FormattedData._render_jinja_template(
                    user_template, data, data_type, prefix, parser, extra_context
                )
                if rendered:
                    print(f"{_DEBUG}Using user template for {data_type}{_RESET}")
                    return rendered

            templates = FormattedData._load_format_templates()
            template_str = templates.get(data_type)
            if template_str:
                rendered = FormattedData._render_jinja_template(
                    template_str, data, data_type, prefix, parser, extra_context
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
