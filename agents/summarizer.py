from typing import Any, Generator, TextIO, TYPE_CHECKING
import hashlib
import io
import json
import jsonc
from contextlib import redirect_stdout, redirect_stderr
import copy
from datetime import datetime
from pathlib import Path
import random
import re
import shutil
import sys
import traceback
from dataclasses import dataclass

if TYPE_CHECKING:
    from torch import no_grad

from modules import shared
from modules.chat import generate_chat_prompt
from modules.llama_cpp_server import LlamaServer
from modules.text_generation import encode

import extensions.dayna_ss.shared as dss_shared

# from extensions.dayna_ss.utils.memory_management import VRAMManager
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
    _RESET,
    _DEBUG,
    History,
    Histories,
    load_json,
    save_json,
    expand_lists_in_data_for_llm,
    get_values,
    enumerate_list,
    strip_thinking,
    strip_response,
)

from extensions.dayna_ss.utils.schema_parser import SchemaParser, ParsedSchemaClass

from extensions.dayna_ss.utils.background_importer import (
    start_background_import,
    get_imported_attribute,
)

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


# TODO: Get base_state from config
base_state = {
    "name1": "SYSTEM",
    "name2": "DAYNA",
    "mode": "instruct",
    "chat-instruct_command": 'Continue the chat dialogue below. Write a single reply for the character "DAYNA". Answer questions flawlessly. Follow instructions to a T.\n\n<|prompt|>',
    "enable_thinking": True,
    "context": (
        "You are DAYNA, an advanced AI assistant integrated into a comprehensive story-writing and world-building environment. Your primary function is to act as a collaborative partner, generating responses that continue a narrative based on a rich, structured context.\n\n"
        "This context is provided in several parts:\n\n"
        "1.  **General Summary:** An overview of the story's world and plot.\n"
        "2.  **Current Scene:** Detailed information about the immediate setting, characters present, time, and circumstances. This is the most immediate and relevant context for your next response.\n"
        "3.  **Relevant Characters & Groups:** Detailed descriptions, relationships, and statuses of characters and groups pertinent to the current interaction.\n"
        "4.  **Relevant Events:** Summaries of past or ongoing events that influence the current situation.\n"
        "5.  **Relevant Messages:** Specific dialogue snippets from earlier in the story that have been identified as relevant.\n"
        "6.  **Recent Dialogue:** The last few exchanges in the conversation to ensure continuity.\n\n"
        "Your instructions are delivered by the SYSTEM. You must follow them precisely. Your goal is to generate a natural, in-character response for your designated persona that seamlessly continues the story, respecting all the provided context and instructions. You are creative, adaptable, and capable of writing in diverse styles and tones."
    ),
    "auto_max_new_tokens": True,
    "temperature": 0.3,
    "history": {"internal": [["<|BEGIN-VISIBLE-CHAT|>", "I am ready to receive instructions!"]]},
}


class Summarizer:
    def __init__(self, config_path: str | None = None):
        """Initialize Summarizer, optionally loading config from `config_path`."""
        dss_dir = Path(__file__).parent.parent  # Root directory of the extension
        self.config = self._load_config(config_path or dss_dir / "dss_config.json")

        # self.vram_manager = VRAMManager()
        # # Initialize RAG system
        # self.story_rag = StoryRAG(
        #     collection_prefix="story_summary",
        #     persist_directory="extensions/dayna_ss/storage/vectors"
        # )
        self.last: SummarizationContextCache | None = None

    def _load_config(self, config_path: str | Path) -> dict:
        """Load summarizer configuration from a JSON file at `config_path`."""
        return load_json(config_path) or {"default_summarization_params": {"max_length": 150}}

    def generate_using_tgwui(
        self,
        prompt: str,
        state: dict,
        history_path: Path | None = None,
        stopping_strings: list[str] | None = ["Unchanged", "unchanged"],
        match_prefix_only: bool = True,
        **kwargs,
    ) -> tuple[str, str]:
        """Generate a message using TGWUI's shared model.

        Args:
            prompt (str): The prompt to give to the LLM.
            state (dict): The state dictionary to generate context with.
            history_path (Path, optional): The history path.
            stopping_strings (list[str], optional): The optional stopping strings.
            match_prefix_only (bool, optional): Whether to match prefix only for stopping strings.

        Returns:
            out (tuple[str, str]): A tuple of (response_text, stop_reason)
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
        with redirect_stderr(DualStream(sys.stderr, capture_buffer)):  # redirect_stdout(DualStream(sys.stdout, capture_buffer))
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

    def generate_summary_with_streaming(
        self,
        prompt: str,
        state: dict,
        stopping_strings: list[str] | None = ["Unchanged", "unchanged"],
        match_prefix_only: bool = True,
        **kwargs,
    ) -> Generator[tuple[str, str], Any, None]:
        """Generate a summary using the loaded TGWUI model with custom stopping strings."""
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

            context_retriever: StoryContextRetriever = self.last.context[1]
            if not isinstance(context_retriever, StoryContextRetriever):
                raise TypeError(f"Expected StoryContextRetriever, got {type(context_retriever)}")

            chunker = MessageChunker(
                history_path=path,
                characters_data=context_retriever.characters,
                groups_data=context_retriever.groups,
                events_data=context_retriever.events,
                current_scene_data=context_retriever.current_scene,
            )
            chunks = chunker.process_message(message, index, current_timestamp)
            print(f"{_SUCCESS}Stored {len(chunks)} message chunks for index {index}{_RESET}")
        except Exception as e:
            print(f"{_ERROR}Error processing message chunks for index {index}: {str(e)}{_RESET}")
            traceback.print_exc()

    def generate_summary_instr_prompt(
        self, user_input: str, state: dict, history: History, **kwargs
    ) -> tuple[str, dict, Path, str]:  # Input method
        print(f"{_HILITE}generate_summary_instr_prompt{_RESET} {kwargs}")
        custom_state = self.retrieve_and_format_context(state, history, **kwargs)
        history_path = self.last.history_path

        next_scene_prefix = "NEXT SCENE:"

        if self.last:
            if user_input.startswith(next_scene_prefix):
                print(f"{_DEBUG}Found '{next_scene_prefix}' in user input in generate_summary_instr_prompt.{_RESET}")
                user_input = user_input[len(next_scene_prefix) :].lstrip()
                self.last.is_new_scene_turn = True
            if self.last.is_new_scene_turn:
                # Message nodes are 0-indexed and user messages are at turn_idx * 2:
                self.last.new_scene_start_node = f"{len(history) * 2}_1_1"
                print(
                    f"{_DEBUG}New scene turn flagged. Start message node for new scene: {self.last.new_scene_start_node}{_RESET}"
                )

        if shared.stop_everything:
            print(f"{_HILITE}Stop signal received after retrieve_and_format_context in generate_summary_instr_prompt.{_RESET}")
            # history_path is already correctly set from self.last.history_path
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
                # input_key = self.hash_key(user_input + shared.model_name)
                if input_key in instructions:
                    instr: str = instructions[input_key]
                    print(f"{_SUCCESS}Found instruction prompt{_RESET} {instr}")
                else:
                    # Generate prompt using LLM
                    try:
                        if model is not None:
                            # Create custom state for summary generation
                            print(f"{_SUCCESS}State set{_RESET}")

                            # Generate instruction
                            prompt = (
                                f"{user_input_prompt}\n\n"
                                f"You are to generate instructions for {name2}'s response. These instructions will be given directly to {name2}.\n"
                                f"The instructions must guide {name2} on what to say or do, following the tone of the latest messages, and should be detailed and specific.\n\n"
                                f"FORMATTING REQUIREMENTS:\n"
                                f"- Present the instructions as a series of plain text paragraphs.\n"
                                f"- Each paragraph should represent a distinct part of the response plan.\n"
                                f"- CRITICAL: Do NOT use any bold formatting, titles, or headings for these paragraphs. Only the paragraph text itself.\n"
                                f"Example of desired output structure (imagine these are the instructions):\n"
                                f"  First, analyze the user's query to understand their core need. Then, formulate a concise opening statement that acknowledges their input.\n"
                                f"  Next, provide the main information or answer, breaking it down into logical points if necessary. Ensure clarity and accuracy in this section.\n"
                                f"Remember: The above is an example. In a narrative context, acknowledging the user's input would break immersion. Additionally, the length of the response should match the established writing style.\n\n"
                                f"INSTRUCTION CONTENT:\n"
                                f"1. Explain in detail, step-by-step, in imperative mood, what {name2} should include in their response.\n"
                                f"2. Be extremely specific, detailing each step.\n"
                                f"3. You are providing instructions FOR the response, not writing the response itself.\n"
                                f"4. Address the instructions directly to {name2} (e.g., 'Start by...', 'Then, explain...'). Do not refer to {name2} in the third person (e.g., '{name2} should...').\n"
                                f"5. Specify the desired length of {name2}'s actual final response (e.g., 'The final response should be one paragraph', 'Aim for two short paragraphs', 'Keep it to three sentences').\n"
                                f"6. Instruct on the use of dialogue: specify when it is appropriate for characters to speak, which characters should speak, and when narration should be used instead of dialogue.\n"
                                f"7. CRITICAL: Explicitly include an instruction that the final response MUST NOT contain any formatting (no bold, no italics, no markdown, no bullet points, no headings, etc.). It must be plain text prose.\n\n"
                                f"REMEMBER: Your entire output must ONLY consist of the instructional paragraphs, adhering strictly to the no-bolding, no-titles format. No extra text, greetings, or sign-offs."
                            )

                            # TODO: Get these from config (ui_parameters)
                            custom_state["auto_max_new_tokens"] = True
                            custom_state["truncation_length"] = 16384
                            custom_state["temperature"] = 0.3
                            instr, _ = self.generate_using_tgwui(
                                prompt, custom_state
                            )  # TODO: Force start of response via "continue"
                            if shared.stop_everything:
                                print(f"{_HILITE}Stop signal received after instruction generation.{_RESET}")
                                return user_input, state, history_path, None

                            # instr += (
                            #     f"\n\n"
                            #     f"Style: You are speaking in third person to {name1}"
                            # )  # TODO: Derive from UI block
                            instructions[input_key] = instr
                            print(f"{_HILITE}Instruction:{_RESET} {instr}")
                            save_json(instructions, instr_path)  # Persist instruction prompt for regenerations

                            # # Save the KV cache for this instruction generation if not already saved
                            # self.vram_manager.save_context_cache()
                            # # Increment position for next generation
                            # self.vram_manager.increment_position()
                    except Exception as e:
                        print(f"{_ERROR}Error generating instruction: {str(e)}{_RESET}")
                        traceback.print_exc()
                        return user_input, state, history_path, None

                # Generate instruction prompt
                # TODO: Include additional user instructions from UI blocks
                full_instr = instr
                # TODO: Get gen params from UI parameters
                instr_prompt = (
                    f"{user_input_prompt}\n\n"
                    f'You are to write a reply in character as "{name2}".\n'
                    f"The following instructions, presented as plain text paragraphs, outline how you should construct your response:\n\n"
                    f'INSTRUCTIONS TO FOLLOW:\n"""\n{full_instr}\n"""\n\n'
                    f"Adhere strictly to these instructions. Maintain the style and tone consistent with recent messages from both {name1} and {name2}.\n"
                    f"Your reply must be natural-sounding prose. ABSOLUTELY CRITICAL: Do NOT use any formatting whatsoever. This includes, but is not limited to, Markdown, bold text, asterisks for emphasis, headings, or titles. The entire response must be plain, unformatted text, unless it's an organic part of {name2}'s typical speech pattern or dialogue.\n\n"
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
                return (
                    encoded_instr_prompt,
                    custom_state,
                    history_path,
                    current_timestamp_str,
                )
            except Exception as e:
                print(f"{_ERROR}Error in get_summary_state: {str(e)}{_RESET}")
                traceback.print_exc()
                return user_input, state, history_path, None

    def summarize_latest_state(self, output: str, user_input: str, state: dict, history: History) -> str:  # Output method
        """Summarize a single message with its context."""
        print(f"{_HILITE}summarize_message{_RESET}")

        try:
            if shared.stop_everything:
                print(f"{_HILITE}Stop signal received before retrieve_and_format_context in summarize_latest_state.{_RESET}")
                return None
            self.retrieve_and_format_context(state, history[:-1])
            if shared.stop_everything:
                print(f"{_HILITE}Stop signal received after retrieve_and_format_context in summarize_latest_state.{_RESET}")
                return None

            # self.last should be set by retrieve_and_format_context
            last_history_path = self.last.history_path
            new_history_path = self.retrieve_history_path(state, history)
            if not new_history_path.exists():
                new_history_path.mkdir(parents=True)
            self.backtrack_history(history, new_history_path)

            from extensions.dayna_ss.agents.data_summarizer import DataSummarizer

            custom_state = copy.deepcopy(self.last.custom_state)
            custom_state["history"]["internal"].append(
                [f"What was the very last exchange?", f"{self.format_dialogue(state, [[user_input, output]])}"]
            )

            data_summarizer = DataSummarizer(
                self, (user_input, output), custom_state, new_history_path, self.last.schema_parser
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

            if missing_schemas:
                print(
                    f"{_ERROR}Could not find required schema definitions for: {missing_schemas}. Aborting summarization.{_RESET}"
                )
                return None

            # Handle special case for new scene turn before processing
            if self.last and self.last.is_new_scene_turn and self.last.new_scene_start_node is not None:
                if "current_scene" in all_subjects_data:
                    print(
                        f"{_DEBUG}Populating '_start_message_node' for new scene with start node: {self.last.new_scene_start_node}{_RESET}"
                    )
                    all_subjects_data["current_scene"]["_start_message_node"] = str(self.last.new_scene_start_node)

            print(f"{_BOLD}Dynamically summarizing data for all subjects using DataSummarizer...{_RESET}")

            # Copy subjects_schema.json to the new history path
            save_json(
                load_json(last_history_path / "subjects_schema.json"),
                new_history_path / "subjects_schema.json",
            )

            def process_subject_update(subject_name: str, data: dict, schema_class: ParsedSchemaClass) -> dict:
                if shared.stop_everything:
                    print(f"{_HILITE}Stop signal received during subject update for {subject_name}.{_RESET}")
                    save_json(data, new_history_path / f"{subject_name}.json")
                    return data

                print(f"{_BOLD}{subject_name}{_RESET} {data_summarizer._should_update_subject(schema_class)}")
                if data_summarizer._should_update_subject(schema_class):
                    return data_summarizer.generate(subject_name, data, schema_class)
                else:
                    save_json(data, new_history_path / f"{subject_name}.json")
                    return data

            # Step 2: Dynamically process each subject.
            processed_subjects_data = {}
            for subject_name, subject_data in all_subjects_data.items():
                schema_class = self.last.schema_parser.get_subject_class(subject_name)
                print(f"{_BOLD}Processing subject: {subject_name}{_RESET} {schema_class}")
                if not schema_class:  # Redundant but good for safety
                    continue
                print(f"{subject_name} exists")

                updated_data = process_subject_update(subject_name, subject_data, schema_class)
                processed_subjects_data[subject_name] = updated_data

                if shared.stop_everything:
                    return None

            # --- Summarize New Messages ---
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

            msg_summarizer = MessageSummarizer(self, new_history_path, current_timestamp_str)
            msg_summarizer.generate((user_input, output), (message_idx - 1, message_idx))

            # --- Update scene_id and event_id for message chunks ---
            if self.last and self.last.context:
                context_retriever = self.last.context[1]
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
                                chunker_instance.update_node_metadata_by_message_idx(msg_idx_to_update, {"scene_id": scene_id})
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
                                chunker_instance.update_node_metadata_by_message_idx(msg_idx_to_update, {"event_id": event_id})
                        except (ValueError, IndexError) as e:
                            print(f"{_ERROR}Could not parse message nodes for event '{event_id}': {e}{_RESET}")

            else:
                print(f"{_ERROR}Cannot update scene/event IDs for chunks: Summarizer.last.context not available.{_RESET}")

            return current_timestamp_str

        except Exception as e:
            print(f"{_ERROR}Error during summarization or metadata update: {str(e)}{_RESET}")
            traceback.print_exc()
            return None

    def backtrack_history(self, history: History, history_path: Path) -> bool:
        """Placeholder for history backtracking logic. Currently does nothing."""
        pass

    def retrieve_and_format_context(self, state: dict, history: History, **kwargs) -> dict:
        """Retrieve and format context for instructing model based on history.

        Args:
            state (dict): Original (TGWUI) state

        Returns:
            out (dict): _description_
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

        # TODO: Add this formatting to subjects_schema.json? Make it dynamic
        # General info
        if retrieval_context.general_info:
            formatted_general_info = FormattedData(
                retrieval_context.general_info, "general_info", self.last.schema_parser
            ).st
            custom_state["context"] += f"\n\n{formatted_general_info}"
            custom_history.append(["Describe the story in broad strokes.", formatted_general_info])

        # Current scene
        if retrieval_context.current_scene:
            formatted_current_scene = FormattedData(
                retrieval_context.current_scene, "current_scene", self.last.schema_parser
            ).st
            custom_history.append(["What is the current scene in the story?", formatted_current_scene])

        # Format and append retrieved information
        if retrieval_context.characters:
            formatted_character_list = FormattedData(retrieval_context.characters, "character_list", self.last.schema_parser).st
            custom_history.append(
                [
                    "What are the relevant details? Start with a list of relevant characters.",
                    formatted_character_list,
                ]
            )

        if retrieval_context.groups:
            formatted_groups = FormattedData(retrieval_context.groups, "groups", self.last.schema_parser).st
            custom_history.append(
                [
                    "Now, relevant groups.",
                    formatted_groups,
                ]
            )

        if retrieval_context.events:
            formatted_events = FormattedData(retrieval_context.events, "events", self.last.schema_parser).st
            custom_history.append(
                [
                    "Now, relevant events.",
                    formatted_events,
                ]
            )

        if retrieval_context.characters:
            formatted_characters = FormattedData(retrieval_context.characters, "characters", self.last.schema_parser).st
            custom_history.append(
                [
                    "Now, describe each of the relevant characters.",
                    formatted_characters,
                ]
            )

        if retrieval_context.messages:
            formatted_lines = FormattedData(retrieval_context.messages, "lines", self.last.schema_parser).st
            custom_history.append(
                [
                    "Now, retrieve specific lines earlier in the story that might be relevant:",
                    formatted_lines,
                ]
            )

        # Repeat current scene for context
        if retrieval_context.current_scene:  # formatted_current_scene is already defined
            custom_history.append(["Repeat the details of the current scene.", formatted_current_scene])

        # Append last x messages
        if last_x_messages:
            custom_history.append([f"What were the last {formatted_last_x} messages?", last_x_messages])

        # Analysis complete marker TODO: Get this SYSTEM prompt from config
        custom_history.append(
            [
                "Analyze all of the above information. Confirm when your analysis is complete.",
                "Analysis complete.",
            ]
        )

        print(
            f"{_HILITE}FORMATTED CONTEXT {_SUCCESS}{json.dumps(custom_history, indent=2)} {_GRAY}{json.dumps(custom_state['history']['internal'], indent=2)}{_RESET}"
        )
        self.last.history_length = len(custom_history)
        return custom_state

    def get_retrieval_context(
        self, state: dict, history: History, current_context: str, **kwargs
    ) -> tuple[dict, tuple[RetrievalContext, StoryContextRetriever, int, str]]:
        """Retrieve and initialize context for the current turn.

        Handles new chats, loads initial world data (from cache or generation),
        sets up session history paths, and loads schemas.

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
                # Populate current_scene.json, characters.json, groups.json using LLM (stubs for now)
                self._populate_initial_current_scene_llm(initial_world_data_path, initial_schema_parser, state)
                self._populate_initial_entities_llm(initial_world_data_path, initial_schema_parser, state)
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
            key (Any): The key to hash.
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

    def _populate_initial_current_scene_llm(
        self, initial_world_data_path: Path, schema_parser: SchemaParser, state: dict
    ) -> None:
        """
        Uses LLM interaction to populate current_scene.json based on state['context'] and state['greeting'].
        Saves to initial_world_data_path / "current_scene.json".
        """
        global base_state

        print(f"{_DEBUG}Attempting to populate initial current_scene.json for {initial_world_data_path}{_RESET}")
        current_scene_target_path = initial_world_data_path / "current_scene.json"

        try:
            current_scene_schema_def = schema_parser.get_subject_class("current_scene")
            if not current_scene_schema_def:
                print(f"{_ERROR}Could not retrieve schema definition for 'current_scene'. Aborting population.{_RESET}")
                save_json({}, current_scene_target_path)  # Save empty as fallback
                return

            example_current_scene = current_scene_schema_def.generate_example_json(
                all_definitions_map=schema_parser.definitions
            )
            example_current_scene_json_str = json.dumps(example_current_scene, indent=2)

            # Get all relevant schema definitions recursively for "CurrentScene"
            all_relevant_definitions = schema_parser.get_relevant_definitions_json("CurrentScene")
            all_relevant_definitions_json_str = json.dumps(all_relevant_definitions, indent=2)

            char_context = state.get("context")
            char_context_str = f'Character Context:\n"""\n{char_context}\n"""\n\n' if char_context else ""
            user_bio = state.get("user_bio")
            # user_bio_str = f'User Bio:\n"""\n{user_bio}\n"""\n\n' if user_bio else ""
            char_greeting = state["history"]["internal"][0][1]  # state.get("greeting")
            char_greeting_str = f'Initial Greeting:\n"""\n{char_greeting}\n"""\n\n'

            base_prompt_template = (
                f"You are an expert data extractor. Based on the provided character context and initial greeting, "
                f"populate the fields for an initial CurrentScene object. "
                f"You MUST strictly adhere to the provided JSON schemas. The primary object to generate is 'CurrentScene'. "
                f"All necessary schema definitions are provided under the 'definitions' key.\n\n"
                f"Focus on the 'start' state. The 'greeting' often sets the immediate scene. "
                f"If information for a specific field is not present, use sensible defaults (empty string, empty list, null) "
                f"or omit the field if optional, ensuring the output strictly adheres to the schema.\n\n"
                f"{char_context_str}"
                # f"{user_bio_str}"
                f"{char_greeting_str}"
                f"Complete JSON Schema Definitions (including CurrentScene and all its dependencies):\n```json\n{all_relevant_definitions_json_str}\n```\n\n"
                f"Example of the expected JSON structure for CurrentScene:\n```json\n{example_current_scene_json_str}\n```\n\n"
                f"{{retry_feedback_placeholder}}"
                f"Your output must be a single valid JSON object for 'CurrentScene' matching the schema and mirroring the structure of the example.\n\n"
                f"REMEMBER: Do not add anything else to the response. Only respond with the JSON object."
            )

            prompt = base_prompt_template.replace("{retry_feedback_placeholder}", "")

            # Prepare a minimal state for generate_using_tgwui
            # It needs 'seed', and other generation params can be default or from self.config
            custom_state = copy.deepcopy(state)
            custom_state.update(copy.deepcopy(base_state))
            # custom_state = {
            #     "seed": state.get("seed", -1), # Use existing seed or default
            #     "max_new_tokens": 1024, # Sensible default for JSON generation
            #     "temperature": 0.5, # Moderate temperature for factual extraction
            #     # Add other necessary default parameters from shared.settings or self.config if needed
            # }
            # Ensure seed is randomized if -1, similar to other generation calls

            max_retries = 2
            populated_data = None

            for attempt in range(max_retries + 1):
                print(f"{_DEBUG}Attempt {attempt + 1}/{max_retries + 1} to generate CurrentScene data...{_RESET}")
                if attempt > 0:  # This is a retry
                    print(f"{_INPUT}Retrying LLM prompt for CurrentScene with validation feedback.{_RESET}")

                llm_response_text, _ = self.generate_using_tgwui(
                    prompt=prompt,
                    state=custom_state,
                    history_path=initial_world_data_path,  # For dump.txt logging
                    match_prefix_only=False,
                )

                if shared.stop_everything:
                    print(f"{_HILITE}Stop signal received during CurrentScene LLM generation.{_RESET}")
                    save_json({}, current_scene_target_path)
                    return

                print(f"{_DEBUG}LLM response for CurrentScene (Attempt {attempt + 1}): {llm_response_text[:300]}...{_RESET}")
                cleaned_response_text = strip_response(llm_response_text)

                try:
                    current_populated_data = jsonc.loads(cleaned_response_text)
                    validation_errors = schema_parser.validate_data(current_populated_data, "CurrentScene")

                    if not validation_errors:
                        populated_data = current_populated_data
                        if "_start_message_node" not in populated_data or not populated_data.get("_start_message_node"):
                            populated_data["_start_message_node"] = "1_1_1"
                            print(f"{_DEBUG}Set '_start_message_node' to '1_1_1' for initial CurrentScene.{_RESET}")
                        print(f"{_SUCCESS}CurrentScene data validated successfully on attempt {attempt + 1}.{_RESET}")
                        break  # Successful validation
                    else:
                        print(f"{_ERROR}Validation errors for CurrentScene on attempt {attempt + 1}:{_RESET}")
                        for error_msg in validation_errors:
                            print(f"{_ERROR}- {error_msg}{_RESET}")

                        if attempt < max_retries:
                            error_feedback = "The previous attempt to generate the JSON failed with the following validation errors. Please correct these issues in your new response:\n"
                            for err in validation_errors:
                                error_feedback += f"- {err}\n"
                            prompt = base_prompt_template.replace("{retry_feedback_placeholder}", error_feedback + "\n\n")
                        else:
                            print(f"{_ERROR}Maximum retries reached for CurrentScene. Persistent validation failure.{_RESET}")

                except json.JSONDecodeError as e:
                    print(
                        f"{_ERROR}Failed to parse LLM response for CurrentScene as JSON on attempt {attempt + 1}: {e}{_RESET}"
                    )
                    print(f"{_ERROR}LLM Raw Response was: {_GRAY}{cleaned_response_text}{_RESET}")
                    if attempt < max_retries:
                        error_feedback = "The previous attempt to generate the JSON was not valid JSON. Please ensure your output is a single, valid JSON object.\n"
                        prompt = base_prompt_template.replace("{retry_feedback_placeholder}", error_feedback + "\n\n")
                    else:
                        print(f"{_ERROR}Maximum retries reached. Failed to parse JSON for CurrentScene.{_RESET}")
                except Exception as e:
                    print(
                        f"{_ERROR}An unexpected error occurred during CurrentScene processing on attempt {attempt + 1}: {e}{_RESET}"
                    )
                    traceback.print_exc()
                    if attempt == max_retries:
                        print(f"{_ERROR}Maximum retries reached. Unexpected error for CurrentScene.{_RESET}")
                    # No specific prompt update here unless we want to tell LLM about generic error

            # After loop, save whatever data we have (valid or last attempt if all failed)
            if populated_data:
                if "_start_message_node" not in populated_data or not populated_data.get("_start_message_node"):
                    populated_data["_start_message_node"] = "1_1_1"
                    print(f"{_DEBUG}Ensured '_start_message_node' is '1_1_1' before saving initial CurrentScene.{_RESET}")
                save_json(populated_data, current_scene_target_path)
                print(f"{_SUCCESS}Successfully populated and saved current_scene.json at {current_scene_target_path}{_RESET}")
            else:
                print(f"{_ERROR}Failed to populate CurrentScene data after all retries. Saving empty JSON.{_RESET}")
                save_json({}, current_scene_target_path)

        except Exception as e:
            print(f"{_ERROR}Error in _populate_initial_current_scene_llm: {e}{_RESET}")
            traceback.print_exc()
            save_json({}, current_scene_target_path)

    def _populate_initial_entities_llm(self, initial_world_data_path: Path, schema_parser: SchemaParser, state: dict) -> None:
        """
        Uses LLM interaction to populate characters.json and groups.json based on state['context'] and state['greeting'].
        Saves to initial_world_data_path / "characters.json" and "groups.json".
        """
        global base_state

        print(f"{_DEBUG}Attempting to populate initial entities (characters & groups) for {initial_world_data_path}{_RESET}")
        characters_target_path = initial_world_data_path / "characters.json"
        groups_target_path = initial_world_data_path / "groups.json"

        char_context = state.get("context", "")
        char_greeting = state["history"]["internal"][0][1]  # state.get("greeting", "")

        custom_state = copy.deepcopy(state)
        custom_state.update(copy.deepcopy(base_state))

        # --- Step A: Identification ---
        identification_prompt = (
            f"From the provided character context and greeting, identify and list all distinct character names and group names. "
            f"For each, provide a brief one-sentence descriptor if available in the text. "
            f"Output as a JSON list of objects, each with 'type' ('character' or 'group'), 'name' (string), and 'descriptor' (string).\n\n"
            f'Character Context:\n"""\n{char_context}\n"""\n\n'
            f'Initial Greeting:\n"""\n{char_greeting}\n"""\n\n'
            f"Example Output:\n"
            f"```json\n"
            f"[\n"
            f'  {{"type": "character", "name": "Elara", "descriptor": "A young sorceress searching for ancient artifacts."}},\n'
            f'  {{"type": "group", "name": "The Silver Hand", "descriptor": "A secretive guild of mages."}}\n'
            f"]\n"
            f"```\n\n"
            f"Identify and list all distinct character names and group names. For each, provide a brief one-sentence descriptor if available in the text. Output as a JSON list of objects, each with 'type' ('character' or 'group'), 'name' (string), and 'descriptor' (string).\n\n"
            f"REMEMBER: Do not add anything else to the response. Only respond with the JSON list."
        )

        print(f"{_DEBUG}Prompting LLM for entity identification...{_RESET}")
        identification_response_text, _ = self.generate_using_tgwui(
            prompt=identification_prompt,
            state=custom_state,
            history_path=initial_world_data_path,  # For dump.txt
            match_prefix_only=False,
        )

        if shared.stop_everything:
            print(f"{_HILITE}Stop signal received during entity identification.{_RESET}")
            save_json({}, characters_target_path)
            save_json({}, groups_target_path)
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
            save_json({}, characters_target_path)
            save_json({}, groups_target_path)
            return

        print(f"{_SUCCESS}Identified {len(identified_entities)} entities. Proceeding to detail extraction.{_RESET}")

        all_characters_data, all_groups_data = self._extract_entity_details_llm(
            initial_world_data_path,
            schema_parser,
            custom_state,
            identified_entities,
            char_context,
            char_greeting,
        )

        # Save the accumulated data
        save_json(all_characters_data, characters_target_path)
        save_json(all_groups_data, groups_target_path)
        print(f"{_SUCCESS}Saved populated characters.json and groups.json at {initial_world_data_path}{_RESET}")

    def _extract_entity_details_llm(
        self,
        initial_world_data_path: Path,
        schema_parser: SchemaParser,
        custom_state: dict,
        identified_entities: list[dict],
        char_context: str,
        char_greeting: str,
    ) -> tuple[dict, dict]:
        """Extracts detailed information for each identified entity using LLM interaction."""
        all_characters_data = {}
        all_groups_data = {}

        for entity in identified_entities:
            entity_name = entity["name"]
            entity_type = entity["type"]
            entity_descriptor = entity["descriptor"]

            schema_to_use = None
            data_dict_to_update = None
            schema_name_for_prompt = ""

            if entity_type == "character":
                schema_class_definition = schema_parser.get_subject_class("characters")
                if (
                    schema_class_definition
                    and hasattr(schema_class_definition, "__args__")
                    and len(schema_class_definition.__args__) == 2
                ):
                    schema_to_use = schema_class_definition.__args__[1]
                data_dict_to_update = all_characters_data
                schema_name_for_prompt = "Character"
            elif entity_type == "group":
                schema_class_definition = schema_parser.get_subject_class("groups")
                if (
                    schema_class_definition
                    and hasattr(schema_class_definition, "__args__")
                    and len(schema_class_definition.__args__) == 2
                ):
                    schema_to_use = schema_class_definition.__args__[1]
                data_dict_to_update = all_groups_data
                schema_name_for_prompt = "Group"
            else:
                print(f"{_ERROR}Unknown entity type '{entity_type}' for '{entity_name}'. Skipping.{_RESET}")
                continue

            if not schema_to_use or not isinstance(schema_to_use, ParsedSchemaClass):
                print(
                    f"{_ERROR}Could not retrieve valid schema definition for {entity_type} '{entity_name}'. Skipping. {_GRAY}{_RESET}"
                )
                continue

            example_entity_data = schema_to_use.generate_example_json(all_definitions_map=schema_parser.definitions)
            example_entity_json_str = json.dumps(example_entity_data, indent=2)

            all_relevant_entity_definitions = schema_parser.get_relevant_definitions_json(schema_name_for_prompt)
            all_relevant_entity_definitions_json_str = json.dumps(all_relevant_entity_definitions, indent=2)

            base_detail_prompt_template = (
                f"Based on the character context, greeting, and focusing on the {entity_type} '{entity_name}' (described as: '{entity_descriptor}'), "
                f"populate the fields for a {schema_name_for_prompt} object. "
                f"You MUST strictly adhere to the provided JSON schemas. The primary object to generate is '{schema_name_for_prompt}'. "
                f"All necessary schema definitions are provided under the 'definitions' key.\n\n"
                f"Prioritize fields like 'description' (as a list of strings), and 'traits' (for characters). "
                f"If information is missing for a field, use sensible defaults (empty list, empty string, null) or omit if optional, adhering to the schema. "
                f"Output a single valid JSON object for this {entity_type}.\n\n"
                f'Character Context:\n"""\n{char_context}\n"""\n\n'
                f'Initial Greeting:\n"""\n{char_greeting}\n"""\n\n'
                f"Complete JSON Schema Definitions (including {schema_name_for_prompt} and all its dependencies):\n```json\n{all_relevant_entity_definitions_json_str}\n```\n\n"
                f"Example of the expected JSON structure for {schema_name_for_prompt} ('{entity_name}'):\n```json\n{example_entity_json_str}\n```\n\n"
                f"{{retry_feedback_placeholder}}"  # Placeholder for validation error feedback
                f"Your output must be a single valid JSON object for '{schema_name_for_prompt}' matching the schema and mirroring the structure of the example."
            )

            detail_prompt = base_detail_prompt_template.replace("{retry_feedback_placeholder}", "")
            custom_state_detail = copy.deepcopy(custom_state)
            max_retries = 2
            entity_data_validated = None

            for attempt in range(max_retries + 1):
                print(
                    f"{_DEBUG}Attempt {attempt + 1}/{max_retries + 1} to generate details for {entity_type} '{entity_name}'...{_RESET}"
                )
                if attempt > 0:
                    print(f"{_INPUT}Retrying LLM prompt for '{entity_name}' with validation feedback.{_RESET}")

                detail_response_text, _ = self.generate_using_tgwui(
                    prompt=detail_prompt,
                    state=custom_state_detail,
                    history_path=initial_world_data_path,
                    match_prefix_only=False,
                )

                if shared.stop_everything:
                    print(f"{_HILITE}Stop signal received during detail extraction for '{entity_name}'.{_RESET}")
                    break

                print(
                    f"{_DEBUG}LLM response for '{entity_name}' (Attempt {attempt+1}): {detail_response_text[:300]}...{_RESET}"
                )
                cleaned_detail_response = strip_response(detail_response_text)

                try:
                    current_entity_data = jsonc.loads(cleaned_detail_response)
                    validation_errors = schema_parser.validate_data(current_entity_data, schema_name_for_prompt)

                    if not validation_errors:
                        entity_data_validated = current_entity_data
                        print(
                            f"{_SUCCESS}Details for {entity_type} '{entity_name}' validated successfully on attempt {attempt + 1}.{_RESET}"
                        )
                        break  # Successful validation
                    else:
                        print(f"{_ERROR}Validation errors for '{entity_name}' on attempt {attempt + 1}:{_RESET}")
                        for error_msg in validation_errors:
                            print(f"{_ERROR}- {error_msg}{_RESET}")

                        if attempt < max_retries:
                            error_feedback = f"The previous attempt to generate JSON for '{entity_name}' failed validation. Correct these issues:\n"
                            for err in validation_errors:
                                error_feedback += f"- {err}\n"
                            detail_prompt = base_detail_prompt_template.replace(
                                "{retry_feedback_placeholder}", error_feedback + "\n\n"
                            )
                        else:
                            print(
                                f"{_ERROR}Maximum retries reached for '{entity_name}'. Persistent validation failure.{_RESET}"
                            )

                except json.JSONDecodeError as e:
                    print(
                        f"{_ERROR}Failed to parse LLM response for '{entity_name}' as JSON on attempt {attempt + 1}: {e}{_RESET}"
                    )
                    print(f"{_ERROR}LLM Raw Response was: {_GRAY}{cleaned_detail_response}{_RESET}")
                    if attempt < max_retries:
                        error_feedback = f"The previous JSON for '{entity_name}' was invalid. Ensure valid JSON output.\n"
                        detail_prompt = base_detail_prompt_template.replace(
                            "{retry_feedback_placeholder}", error_feedback + "\n\n"
                        )
                    else:
                        print(f"{_ERROR}Maximum retries reached. Failed to parse JSON for '{entity_name}'.{_RESET}")
                except Exception as e:
                    print(f"{_ERROR}Unexpected error processing '{entity_name}' on attempt {attempt + 1}: {e}{_RESET}")
                    traceback.print_exc()
                    if attempt == max_retries:
                        print(f"{_ERROR}Maximum retries reached. Unexpected error for '{entity_name}'.{_RESET}")

            if entity_data_validated:
                data_dict_to_update[entity_name] = entity_data_validated
                print(f"{_SUCCESS}Successfully populated and stored details for {entity_type} '{entity_name}'.{_RESET}")
            else:
                print(
                    f"{_ERROR}Failed to populate valid details for {entity_type} '{entity_name}' after all retries. It will be omitted.{_RESET}"
                )

            if shared.stop_everything:  # Check again in case stop happened during the last attempt's processing
                print(
                    f"{_HILITE}Stop signal received after processing for '{entity_name}'. Aborting further entity processing.{_RESET}"
                )
                break

        return all_characters_data, all_groups_data


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
                self.chunker.store_chunks([summary_chunk_data])
                print(f"{_SUCCESS}Stored summary for message_idx {current_message_idx}{_RESET}")
            except Exception as e:
                print(f"{_ERROR}Error generating message summary for message_idx {current_message_idx}: {e}{_RESET}")
                traceback.print_exc()


class FormattedData:
    def __init__(self, data: Any, data_type: str, parser: SchemaParser = None):
        """Initialize and process data for LLM formatting.

        Prepares a string representation (`self.st`) for LLM prompts.

        If schema parser is provided, expands lists to dicts with string keys if schema indicates.

        Args:
            data (Any): Data to process (dict, list, primitive).
            data_type (str): Type hint (e.g., "current_scene", "characters").
            parser (SchemaParser, optional): For schema-based list expansion.
        """
        self.original_data = data
        self.data_type = data_type
        self.parser = parser
        self.data = data

        if self.parser:
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

            self.data = expand_lists_in_data_for_llm(self.data, actual_data_schema_hint, self.parser)

        self._str = FormattedData.format_retrieval_data(self.data, self.data_type)

    def __getitem__(self, index):
        """Allow dictionary-like access to the (potentially expanded) data."""
        return self.data[index]

    @staticmethod
    def format_retrieval_data(data: dict | list, data_type: str) -> str:
        """Format retrieved data based on its type."""
        if not data:
            return ""

        nl = "\n"

        if data_type == "current_scene":
            if (data.get("start") or data.get("now")) is None:
                print(f"{_ERROR}No current scene data available.{_RESET}")
                return "<EMPTY>"

            formatted_str = []
            for start_or_now in ["start", "now"]:
                who = data[start_or_now].get("who", {})
                characters = who.get("characters", {})
                groups = who.get("groups", {})
                when = data[start_or_now].get("when", {})
                where = data[start_or_now].get("where", "Unknown")
                why = data[start_or_now].get("why", {})

                # Format character entries
                char_entries = [
                    f"{char_data.get('name', 'Unknown')} @ {char_data.get('location', 'Unknown')} <<<<<<<<<<<< current_scene.{start_or_now}.who.characters[{i}], current_scene.{start_or_now}.who.characters[{i}].name, current_scene.{start_or_now}.who.characters[{i}].location"
                    for i, char_data in enumerate_list(characters)
                ]
                formatted_chars = f"\n- ".join(char_entries) if char_entries else "None"

                # Format group entries
                group_entries = [
                    f"{group_data.get('name', 'Unknown')} @ {group_data.get('location', 'Unknown')} <<<<<<<<<<<< current_scene.{start_or_now}.who.groups[{i}], current_scene.{start_or_now}.who.groups[{i}].name, current_scene.{start_or_now}.who.groups[{i}].location"
                    for i, group_data in enumerate_list(groups)
                ]
                formatted_groups = f"\n- ".join(group_entries) if group_entries else "None"

                formatted_when = f"{when.get('date', 'Unknown')} - {when.get('time', 'Unknown')} ({when.get('specific_time', 'Unknown')}) <<<<<<<<<<<< current_scene.{start_or_now}.when, current_scene.{start_or_now}.when.date, current_scene.{start_or_now}.when.time, current_scene.{start_or_now}.when.specific_time"

                # Format why entries
                why_entries = [
                    f"{reason_data.get('name', 'Unknown')} -- {reason_data.get('details', 'Unknown')} <<<<<<<<<<<< current_scene.{start_or_now}.why[{i}]"
                    for i, reason_data in enumerate_list(why)
                ]
                formatted_why = f"\n- ".join(why_entries) if why_entries else "Unknown"

                formatted_str.append(
                    f"Scene -- {data.get('what', 'Unknown')} <<<<<<<<<<<< current_scene.{start_or_now}, current_scene.what\n\n"
                    f"Characters -- <<<<<<<<<<<< current_scene.{start_or_now}.who.characters\n- {formatted_chars}\n\n"
                    f"Groups -- <<<<<<<<<<<< current_scene.{start_or_now}.who.groups\n- {formatted_groups}\n\n"
                    f"When -- {formatted_when}\n\n"
                    f"Where -- {where} <<<<<<<<<<<< current_scene.{start_or_now}.where\n\n"
                    f"Why -- <<<<<<<<<<<< current_scene.{start_or_now}.why\n- {formatted_why}"
                )

            return (
                f"Current Scene Start --- <<<<<<<<<<<< current_scene.start\n{formatted_str[0]}\n\n"
                f"Current Scene Now --- <<<<<<<<<<<< current_scene.now\n{formatted_str[1]}"
            )

        if data_type == "character_list":
            return "\n".join(
                f"Character --- {char} <<<<<<<<<<<< characters.{char} (key)\n"
                f"Relationships --- {', '.join(char_data.get('relationships', {}).keys())} <<<<<<<<<<<< characters.{char}.relationships\n"
                for char, char_data in data.items()
            )

        if data_type == "characters_list":
            return f"[{json.dumps(list(data.keys()), indent=None)}] <<<<<<<<<<<< characters"

        if data_type == "characters":
            return "\n\n".join(
                f"Character --- {char} <<<<<<<<<<<< characters.{char} (key)\n"
                f"Description --- {nl.join(char_data.get('description', []))} <<<<<<<<<<<< characters.{char}.description\n"
                f"Relationships --- <<<<<<<<<<<< characters.{char}.relationships\n- {f'{nl}- '.join(f'{rel_char}: {json.dumps(rel_data, indent=None)} <<<<<<<<<<<< characters.{char}.{rel_char} (key), characters.{char}.{rel_char}' for rel_char, rel_data in char_data.get('relationships', {}).items())}\n"
                f"Group Status --- <<<<<<<<<<<< characters.{char}.status\n- {f'{nl}- '.join(f'{group}: {json.dumps(status, indent=None)} <<<<<<<<<<<< characters.{char}.status.{group} (key), characters.{char}.status.{group}' for group, status in char_data.get('status', {}).items())}"
                for char, char_data in data.items()
            )

        if data_type == "groups" or data_type == "groups_list":
            return "\n\n".join(
                f"Group --- {group} ({', '.join(get_values(group_data.get('aliases', {})))}) <<<<<<<<<<<< groups.{group} (key), groups.{group}.aliases\n"
                f"Description --- {nl.join(group_data.get('description', []))} <<<<<<<<<<<< groups.{group}.description\n"
                f"Specific Events --- {', '.join(get_values(group_data.get('events', {})))} <<<<<<<<<<<< groups.{group}.events\n"
                for group, group_data in data.items()
            )

        if data_type == "scene" or data_type == "event":  # StoryEvent
            start = data.get("start", {})
            end = data.get("end", {})
            return (
                f"Scene --- {data.get('name', 'Unknown')} <<<<<<<<<<<< {data_type}.name\n\n"
                f"Start --- <<<<<<<<<<<< {data_type}.start\n"
                f"- Date: {start.get('date', 'Unknown')} <<<<<<<<<<<< {data_type}.start.date\n"
                f"- Time: {start.get('time', 'Unknown')} ({start.get('specific_time', 'no specific time')}) <<<<<<<<<<<< {data_type}.start.time, {data_type}.start.specific_time\n\n"
                f"End --- <<<<<<<<<<<< {data_type}.end\n"
                f"- Date: {end.get('date', 'Unknown')} <<<<<<<<<<<< {data_type}.end.date\n"
                f"- Time: {end.get('time', 'Unknown')} ({end.get('specific_time', 'no specific time')}) <<<<<<<<<<<< {data_type}.end.time, {data_type}.end.specific_time\n\n"
                f"Summary --- {data.get('summary', 'No summary available')} <<<<<<<<<<<< {data_type}.summary"
            )

        if data_type == "events" or data_type == "events_list":
            formatted_str = []
            formatted_str.append(
                "\n\n".join(
                    f"Event -- {event_data.get('name')} <<<<<<<<<<<< events.past[{i}].name\n"
                    f"When -- {event_data.get('start', {}).get('date', 'Unknown')} <<<<<<<<<<<< events.past[{i}].start.date\n"
                    f"Summary -- {event_data.get('summary', 'No summary available')} <<<<<<<<<<<< events.past[{i}].summary"
                    for i, event_data in enumerate_list(data.get("past", []))
                )
                or "Empty"
            )

            formatted_str.append(
                "\n\n".join(
                    f"Scene -- {event_data.get('name')} <<<<<<<<<<<< events.scenes[{i}].name\n"
                    f"When -- {event_data.get('start', {}).get('date', 'Unknown')} <<<<<<<<<<<< events.scenes[{i}].start.date\n"
                    f"Summary -- {event_data.get('summary', 'No summary available')} <<<<<<<<<<<< events.scenes[{i}].summary"
                    for i, event_data in enumerate_list(data.get("scenes", []))
                )
                or "Empty"
            )

            formatted_str.append(
                "\n\n".join(
                    f"Event -- {event_data.get('name')} <<<<<<<<<<<< events.events[{i}].name\n"
                    f"When -- {event_data.get('start', {}).get('date', 'Unknown')} <<<<<<<<<<<< events.events[{i}].start.date\n"
                    f"Summary -- {event_data.get('summary', 'No summary available')} <<<<<<<<<<<< events.events[{i}].summary"
                    for i, event_data in enumerate_list(data.get("events", []))
                )
                or "Empty"
            )

            return (
                f"Past Events --- <<<<<<<<<<<< events.past\n{formatted_str[0]}\n\n"
                f"Scenes --- <<<<<<<<<<<< events.scenes\n{formatted_str[1]}\n\n"
                f"Events --- <<<<<<<<<<<< events.events\n{formatted_str[2]}"
            )
        
        if data_type == "general_info":
            return (
                f"Synopsis --- {data.get('synopsis', 'No synopsis available')} <<<<<<<<<<<< general_info.synopsis\n\n"
                f"Main Objective --- {data.get('main_objective', 'No main objective specified')} <<<<<<<<<<<< general_info.main_objective\n\n"
                f"Themes --- {', '.join(get_values(data.get('themes', []))) or 'No themes specified'} <<<<<<<<<<<< general_info.themes\n\n"
                f"Tone --- {data.get('tone', 'No tone specified')} <<<<<<<<<<<< general_info.tone\n\n"
                f"Writing Style --- {data.get('writing_style', 'No writing style specified')} <<<<<<<<<<<< general_info.writing_style"
            )

        if data_type == "lines":
            return "\n".join(data.values())

        return "<EMPTY>"  # str(data)

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

        def replacer_logic(match_obj):
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
