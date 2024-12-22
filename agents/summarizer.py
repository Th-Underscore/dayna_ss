from cgi import test
from typing import Dict, List, Optional, Any, Generator
import json
import copy
from datetime import datetime
from pathlib import Path
import traceback
from dataclasses import dataclass, field

from modules.chat import generate_chat_prompt
from modules.llamacpp_model import LlamaCppModel
from modules.text_generation import encode
import extensions.dayna_story_summarizer.config as config
from extensions.dayna_story_summarizer.utils.memory_management import VRAMManager
from extensions.dayna_story_summarizer.rag.structured_rag.context_retriever import RetrievalContext, StoryContextRetriever, MessageChunker
import torch

# Color codes
_ERROR = "\033[1;31m"
_SUCCESS = "\033[1;32m"
_INPUT = "\033[0;33m"
_GRAY = "\033[0;30m"
_HILITE = "\033[0;36m"
_BOLD = "\033[1;37m"
_RESET = "\033[0m"

@dataclass
class SummarizationData:
    history_path: Path
    custom_state: Dict
    context: tuple[RetrievalContext, StoryContextRetriever, int, str]

class Summarizer:
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path or Path(__file__).parent / "dss_config.json")
        self.vram_manager = VRAMManager()
        # Initialize RAG system
        # self.story_rag = StoryRAG(
        #     collection_prefix="story_summary",
        #     persist_directory="extensions/dayna_story_summarizer/storage/vectors"
        # )
        self.last: Optional[SummarizationData] = None
        self.last_user_input: Optional[str] = None

    def _load_config(self, config_path: str | Path) -> Dict:
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"Warning: Could not load config from {config_path}, using defaults")
            return {"default_summarization_params": {"max_length": 150}}

    def load_json_data(self, filename: str) -> Dict:
        """Load JSON data from the history path."""
        try:
            filepath = Path(self.last.history_path) / filename
            if filepath.exists():
                with open(filepath, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            print(f"{_ERROR}Error loading {filename}: {str(e)}{_RESET}")
            return {}

    def save_json_data(self, data: Dict, filename: str) -> bool:
        """Save JSON data to the history path."""
        try:
            filepath = Path(self.last.history_path) / filename
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            print(f"{_ERROR}Error saving {filename}: {str(e)}{_RESET}")
            return False
        
    def generate_summary(self, prompt: str, state: Dict, **kwargs) -> tuple[str, bool | str]:
        """Generate a summary using the shared model."""
        if not self.last.history_path:
            self.last.history_path = self.last.history_path
        text = ""
        stop = False
        for t, s in self.generate_summary_with_streaming(prompt, state, **kwargs):
            text = t
            stop = s
        try:
            with open(self.last.history_path / "dump.txt", "a") as f:
                dump_str =  "\n\n==========================\n"
                dump_str += "==========================\n"
                dump_str += "==========================\n"
                dump_str += "==========================\n"
                dump_str += "==========================\n\n"
                # dump_str += json.dumps(state, indent=2)
                dump_str += prompt
                dump_str += "\n\n==========================\n\n"
                dump_str += text
                f.write(dump_str)
                f.close()
        except Exception as e:
            print(f"{_ERROR}Error writing to history file: {str(e)}{_RESET}")
            traceback.print_exc()
        return text.strip(), stop

    def generate_summary_with_streaming(self, prompt: str, state: Dict, **kwargs) -> Generator[tuple[str, bool | str], Any, None]:
        """Generate a summary using the shared model."""
        custom_state = copy.deepcopy(state)
        stopping_strings = kwargs.get('stopping_strings', ["Unchanged", "unchanged"])
        # if stopping_strings:
        #     quoted_tokens = [f'"{token}"' for token in stopping_strings]
        #     custom_state['custom_token_bans'] = ', '.join(quoted_tokens) if custom_state['custom_token_bans'] else ', '.join(quoted_tokens)
        try:
            from modules.shared import model
            model: LlamaCppModel
            if model is not None:
                with torch.no_grad():
                    instr_prompt = generate_chat_prompt(prompt, custom_state, **kwargs)
                    encoded_instr_prompt = encode(instr_prompt, add_bos_token=True) if model.__class__.__name__ != 'LlamaCppModel' else instr_prompt
                    stopped = False
                    for text in model.generate_with_streaming(encoded_instr_prompt, custom_state):
                        if stopping_strings:
                            for stopping_string in stopping_strings:
                                if stopping_string in text:
                                    yield text, stopping_string
                                    stopped = True
                                    break
                        if stopped: break
                        yield text, False
                    print(f"{_RESET}Generated summary length: {len(text)}{_RESET}")
                    #return text, stopped
        except Exception as e:
            print(f"{_ERROR}Error generating summary: {str(e)}{_RESET}")
            traceback.print_exc()
    
    def save_message_chunks(self, message: str, index: int, **kwargs) -> None:
        """Save message chunks to the history path."""
        print(f'{_BOLD}save_message_chunks{_RESET}')
        if not self.last or not self.last.history_path:
            raise ValueError(f"History path not set")
        try:
            chunker = MessageChunker(self.last.history_path)
            chunks = chunker.process_message(message, index)
            print(f"{_SUCCESS}Stored {len(chunks)} message chunks{_RESET}")
        except Exception as e:
            print(f"{_ERROR}Error processing message chunks: {str(e)}{_RESET}")
            traceback.print_exc()

    def summarize_message(self, output: str, state: Dict, **kwargs) -> str:
        """Summarize a single message with its context."""
        print(f'{_HILITE}summarize_message{_RESET}')
        
        try:
            self.retrieve_and_format_context(state, **kwargs)

            # Load and summarize all data types
            CharacterSummarizer(self).generate()
            # events = EventSummarizer(self, custom_state).generate()
            GroupSummarizer(self).generate()
            # current_scene = CurrentSceneSummarizer(self, custom_state).generate()
            user_input = state['history']['internal'][-1][0]
            message_idx = len(state['history']['internal']) * 2
            MessageSummarizer(self).generate([user_input, output], [message_idx - 1, message_idx])

        except Exception as e:
            print(f"{_ERROR}Error during summarization: {str(e)}{_RESET}")
            traceback.print_exc()
            return None

    def get_summary_prompt(self, user_input: str, state: Dict, **kwargs) -> Dict:
        print(f"{_HILITE}get_summary_prompt{_RESET} {kwargs}")
        history = kwargs.pop('history')
        _continue = kwargs.get('_continue', False)
        impersonate = kwargs.pop('impersonate', False)
        regenerate = state['history']['internal'][-1][0] == user_input
        if regenerate or _continue:
            old_state = state
            print(f"{_SUCCESS}Detected regenerate/continue{_RESET} {user_input}\n{_INPUT}{state['history']['internal'][-1]}")
            state = copy.deepcopy(state)
            state['history']['internal'].pop(-1)
        else:
            old_state = None
        custom_state = self.retrieve_and_format_context(state, **kwargs)
        history_path = self.last.history_path
        if history_path:
            """Get the current state of the story summary."""
            try:
                # # Check if we have a cached KV state for this prompt
                # cached_kv = self.vram_manager.get_context_cache()
                # if cached_kv is not None:
                #     print(f"{_SUCCESS}Found cached KV state{_RESET}")
                #     print(f"{_SUCCESS}Cache ready for current position{_RESET}")

                # Generate summary using direct model inference
                try:
                    # Get shared model
                    from modules.shared import model
                    from modules.llamacpp_model import LlamaCppModel
                    model: LlamaCppModel
                    if model is not None:
                        user_input_prompt = f"This is the latest user input:\n\n\"\"\"{user_input}\"\"\"\n\n"
                        # Create custom state for summary generation
                        with torch.no_grad():
                            name1 = state['name1'] or 'User'
                            name2 = state['name2'] or 'Assistant'
                            print(f"{_SUCCESS}State set{_RESET}")

                            # Generate instruction
                            prompt = f"{user_input_prompt}\n\nNow, this is your instruction: Explain what {name2} should respond with. These instructions will be given to {name2} directly. Follow the tone of the latest messages. Explain in detail, step-by-step, in imperative mood. Be extremely specific, detailing each step of the response. Only give instructions, not the actual response. Remember, list each instruction step by step in imperative form as if speaking to {name2}. Do not be vague."
                            
                            # TODO: Get these from config
                            custom_state['max_new_tokens'] = 512
                            custom_state['truncation_length'] = 16384
                            custom_state['temperature'] = 0.3
                            
                            instr, _ = self.generate_summary(prompt, custom_state)

                        # # Save the KV cache for this summary generation if not already saved
                        # self.vram_manager.save_context_cache()
                        # # Increment position for next summary
                        # self.vram_manager.increment_position()

                        # Generate instruction prompt TODO: Include additional user instructions from UI
                        instr_prompt = f"{user_input_prompt}\n\nNow, write a reply as \"{name2}\". Imitating {name2}, follow these instructions:\n\n{instr}\n\nFollow each and every one of these instructions to a T. Follow the style and tone of the latest messages as a reference.\n\nRemember, write as if you are {name2}. Follow the same style of writing as {name1} and {name2}. Don't add any unnecessary formatting."
                        encoded_instr_prompt = encode(instr_prompt, add_bos_token=True) if model.__class__.__name__ != 'LlamaCppModel' else instr_prompt
                        print(f"{_SUCCESS}Encoded instruct prompt: {True if model.__class__.__name__ != 'LlamaCppModel' else False}{_RESET}")
                        print(f"{_SUCCESS}State set{_RESET}")

                        try:
                            if not history_path.exists():
                                history_path.mkdir(parents=True)
                            with open(history_path / "dump.txt", "w") as f:
                                dump_str = str(json.dumps(kwargs, indent=2))
                                dump_str += "\n\n==========================\n\n"
                                dump_str += str(json.dumps(custom_state, indent=2))
                                dump_str += "\n\n==========================\n\n"
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
                        return encoded_instr_prompt, custom_state, history_path
                    print(f"{_ERROR}No model found{_RESET}")
                    return user_input, state, history_path
                except Exception as e:
                    print(f"{_ERROR}Error during custom state generation: {str(e)}{_RESET}")
                    traceback.print_exc()
                    return user_input, state, history_path
            except Exception as e:
                print(f"{_ERROR}Error in get_summary_state: {str(e)}{_RESET}")
                traceback.print_exc()
                return user_input, state, history_path

    def retrieve_and_format_context(self, state: Dict, **kwargs) -> Dict:
        """Retrieve and format context for instructing model based on history.

        Args:
            state (Dict): Original (base) state

        Returns:
            Dict: _description_
        """
        try:
            old_custom_state = None
            if self.last:
                old_custom_state = self.last.custom_state
            current_context = self.get_general_summarization(state) or state['context']
            custom_state, retrieval_context, context_retriever, last_x, last_x_messages = self.get_retrieval_context(state, current_context, **kwargs)
            if custom_state is old_custom_state:
                return custom_state
            
            # TODO: Get these all from config
            custom_state['name1'] = "SYSTEM"
            custom_state['name2'] = "DAYNA"
            custom_state['chat-instruct_command'] = "Continue the chat dialogue below. Write a single reply for the character \"DAYNA\". Answer questions flawlessly. Follow instructions to a T.\n\n<|prompt|>"
            custom_state['context'] = "The following is a conversation with an AI Large Language Model agent, DAYNA. DAYNA has been trained to answer questions, assist with storywriting, and help with decision making. DAYNA follows system (SYSTEM) requests. DAYNA specializes writing in various styles and tones. DAYNA thinks outside the box."
            custom_state['history']['internal'] = [["<|BEGIN-VISIBLE-CHAT|>", "I am ready to receive instructions!"]]
            internal_history = custom_state['history']['internal']
            
            custom_state['context'] += f"\n\n{current_context}"
            
            formatted_last_x = self.format_number(last_x)
            current_scene = context_retriever.get_current_scene()
            formatted_current_scene = self.format_retrieval_data(current_scene, 'current_scene')

            # Format and append retrieved information
            if retrieval_context.characters:
                formatted_char_list = self.format_retrieval_data(retrieval_context.characters, 'character_list')
                internal_history.append(["What are the relevant details? Start with a list of relevant characters.", formatted_char_list])

            if retrieval_context.groups:
                formatted_groups = self.format_retrieval_data(retrieval_context.groups, 'groups')
                internal_history.append(["Now, relevant groups.", formatted_groups])

            if retrieval_context.events:
                formatted_events = self.format_retrieval_data(retrieval_context.events, 'events')
                internal_history.append(["Now, relevant events.", formatted_events])

            if retrieval_context.characters:
                formatted_chars = self.format_retrieval_data(retrieval_context.characters, 'characters')
                internal_history.append(["Now, describe each of the relevant characters.", formatted_chars])

            if retrieval_context.messages:
                formatted_lines = self.format_retrieval_data(retrieval_context.messages, 'lines')
                internal_history.append(["Now, retrieve specific lines earlier in the story that might be relevant:", formatted_lines])

            # Repeat current scene for context
            if current_scene:
                internal_history.append(["Repeat the details of the current scene.", formatted_current_scene])

            # Append last x messages
            if last_x_messages:
                internal_history.append([f"What were the last {formatted_last_x} messages?", last_x_messages])

            # Analysis complete marker TODO: Get this SYSTEM prompt from config
            internal_history.append(["Analyze all of the above information. Confirm when your analysis is complete.", "Analysis complete."])

            print(f"{_HILITE}FORMATTED CONTEXT {_SUCCESS}{json.dumps(internal_history, indent=2)} {_GRAY}{json.dumps(custom_state['history']['internal'], indent=2)}{_RESET}")
            return custom_state
        except Exception as e:
            print(f"{_ERROR}Error retrieving and formatting context: {str(e)}{_RESET}")
            traceback.print_exc()
            return None

    def get_retrieval_context(self, state: Dict, current_context: str, **kwargs) -> tuple[Dict, RetrievalContext, StoryContextRetriever, int, str]:
        config.update_config(state)
        character_path = Path("extensions/dayna_story_summarizer/user_data/history", config.current_character)
        history_path = character_path / "20241207-12-53-25"

        if not self.last or (history_path and history_path != self.last.history_path):
            custom_state = copy.deepcopy(state)
            context_retriever = StoryContextRetriever(history_path)

            # RAG
            internal_history = custom_state['history']['internal']

            # TODO: Summary of last scene
            # last_scene = context_retriever.get_scene(-1)
            # if last_scene:
            #     formatted_last_scene = self.format_retrieval_data(last_scene, 'scene')
            #     current_context += f"\n\nThe last scene was: {formatted_last_scene}"
            
            # Current scene
            current_scene = context_retriever.get_current_scene()
            if current_scene:
                formatted_current_scene = self.format_retrieval_data(current_scene, 'scene')
                current_context += f"\n\nThe current scene is: {formatted_current_scene}"
                internal_history.append(["What is the current scene in the story?", formatted_current_scene])

            # Retrieve last x messages
            og_internal_history = state['history']['internal']
            last_x = kwargs.get('last_x', 2)
            last_x = min(len(og_internal_history), last_x)
            last_x_messages = self.format_dialogue(state, og_internal_history[-last_x:])

            retrieval_context = context_retriever.retrieve_context(
                current_context=current_context,
                last_x_messages=last_x_messages
            )

            self.last = SummarizationData(
                context=(retrieval_context, context_retriever, last_x, last_x_messages),
                custom_state=custom_state,
                history_path=history_path
            )

        return self.last.custom_state, *self.last.context

    def format_retrieval_data(self, data: Dict, data_type: str) -> str:
        """Format retrieved data based on its type."""
        if not data:
            return "<EMPTY>"

        nl = "\n"
        if data_type == 'current_scene':
            who = data.get("now", data.get("start", {})).get("who", {})
            characters = who.get("characters", [])
            groups = who.get("groups", [])
            when = data.get("now", data.get("start", {})).get("when", {})
            where = data.get("now", data.get("start", {})).get("where", "Unknown")
            why = data.get("now", data.get("start", {})).get("why", [])

            # Format character entries
            char_entries = []
            for char in characters:
                char_entry = f"{char.get('name', 'Unknown')} @ {char.get('location', 'Unknown')}"
                char_entries.append(char_entry)
            formatted_chars = f"{nl}- ".join(char_entries)

            # Format group entries
            group_entries = []
            for group in groups:
                group_entry = f"{group.get('name', 'Unknown')} @ {group.get('location', 'Unknown')}"
                group_entries.append(group_entry)
            formatted_groups = f"{nl}- ".join(group_entries)

            why_entries = [
                f"{reason.get('name', 'Unknown')} --- {reason.get('details', 'Unknown')}"
                for reason in why
            ]
            # Format why entries
            formatted_why = f"{nl}- ".join(why_entries) if why_entries else 'Unknown'

            # Build the final string
            scene_header = f"Scene --- {data.get('what', 'Unknown')}"
            chars_section = f"Characters ---\n- {formatted_chars}"
            groups_section = f"Groups ---\n- {formatted_groups}"
            when_section = f"When --- {when.get('date', 'Unknown')}; {when.get('time', 'Unknown')} ({when.get('specific time', 'Unknown')})"
            where_section = f"Where --- {where}"
            why_section = f"Why ---\n- {formatted_why}"

            return (
                f"{scene_header}\n\n" +
                f"{chars_section}\n\n" +
                f"{groups_section}\n\n" +
                f"{when_section}\n\n" +
                f"{where_section}\n\n" +
                f"{why_section}"
            )

        if data_type == 'character_list':
            return "\n".join(
                f"Character --- {name}\n" +
                f"Relationships --- {', '.join(char_data.get('relationships', {}).keys())}\n"
                for name, char_data in data.items()
            )

        if data_type == 'characters':
            return "\n\n".join(
                f"Character --- {name}\n" +
                f"Description --- {nl.join(char_data.get('description', []))}\n" +
                f"Relationships ---\n- {f'{nl}- '.join(char_data.get('relationships', {}).keys())}\n" +
                f"Status ---\n- {f'{nl}- '.join(f'{group}: {json.dumps(status, indent=None)}' for group, status in char_data.get('status', {}).items())}"
                for name, char_data in data.items()
            )

        if data_type == 'groups':
            return "\n\n".join(
                f"Group --- {name}\n" +
                f"Description --- {' '.join(group_data.get('description', []))}\n" +
                f"Events --- {', '.join(group_data.get('events', []))}"
                for name, group_data in data.items()
            )

        if data_type == 'scene':
            start = data.get("start", {})
            end = data.get("end", {})

            return (
                f"Scene --- {data.get('name', 'Unknown')}\n\n" +
                f"Start ---\n" +
                f"- Date: {start.get('date', 'Unknown')}\n" +
                f"- Time: {start.get('time', 'Unknown')} ({start.get('specific time', 'Unknown')})\n\n" +
                f"End ---\n" +
                f"- Date: {end.get('date', 'Unknown')}\n" +
                f"- Time: {end.get('time', 'Unknown')} ({end.get('specific time', 'Unknown')})\n\n" +
                f"Summary --- {data.get('summary', 'No summary available')}"
            )

        if data_type == 'events':
            return "\n\n".join(
                f"Event --- {name}\n" +
                f"When --- {event_data.get('start', {}).get('date', 'Unknown')}\n" +
                f"Summary --- {event_data.get('summary', 'No summary available')}"
                for name, event_data in data.items()
            )

        if data_type == 'lines':
            return "\n".join(data)

        return "<EMPTY>"#str(data)

    def _create_summary_prompt(self, message: str, context: Dict):
        """Create a prompt for summarization based on message and context"""
        return f"""Summarize the following message while preserving key information:
Context: {json.dumps(context)}
Message: {message}

Summary:"""

    def get_general_summarization(self, state: Dict):
        return

    def get_current_scene(self, state: Dict):
        try:
            if not self.current_scene:
                self.current_scene = ""
                return self.current_scene
            return self.current_scene
        except Exception as e:
            print(f"{_ERROR}Error getting current scene: {str(e)}{_RESET}")
            return None

    def format_number(self, num: int):
        return str(num)

    def format_dialogue(self, state: Dict, internal_history: list[list]):
        # Copied from modules.chat
        from functools import partial
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)

        chat_template = jinja_env.from_string(state['chat_template_str'])
        chat_renderer = partial(
            chat_template.render,
            add_generation_prompt=False,
            name1=state['name1'],
            name2=state['name2']
        )
        messages = []
        for message in internal_history:
            if message[0] and message[0] != "<|BEGIN-VISIBLE-CHAT|>": messages.append({ 'role': 'user', 'content': message[0] })
            if message[1]: messages.append({ 'role': 'assistant', 'content': message[1] })
        return chat_renderer(messages=messages)

class CharacterSummarizer:
    def __init__(self, summarizer: Summarizer, custom_state: Dict):
        self.summarizer = summarizer
        self.data = summarizer.last
        self.custom_state = custom_state
    
    def generate(self) -> Dict:
        """Update character data."""
        try:
            print(f"{_BOLD}Summarizing characters{_RESET}")
            retrieval_context = self.data.context[0]
            characters = retrieval_context.characters

            for char_name, char_data in characters.items():
                formatted_data = self.summarizer.format_retrieval_data({ char_name: char_data }, 'characters')

                print(f"{_HILITE}Summarizing character description for {char_name}: {_GRAY}{json.dumps(char_data)}{_RESET}")

                self.update_description(char_name, char_data, formatted_data)
                self.update_traits(char_name, char_data, formatted_data)
                self.update_attributes(char_name, char_data, formatted_data)
                self.update_status(char_name, char_data, formatted_data)
                self.update_relationships(char_name, char_data, formatted_data)

                print(f"{_HILITE}Updated character description for {char_name}: {_SUCCESS}{json.dumps(char_data)}{_RESET}")
                with open(self.data.history_path / "dump.txt", "a") as f:
                    dump_str =  "\n\n==========================\n"
                    dump_str += "==========================\n"
                    dump_str += "==========================\n"
                    dump_str += "==========================\n"
                    dump_str += "==========================\n\n"
                    dump_str += json.dumps(char_data, indent=2)
                    f.write(dump_str)
                    f.close()

            return characters
        except Exception as e:
            print(f"{_ERROR}Error in summarize_characters: {e}{_RESET}")
            traceback.print_exc()
            return {}

    def update_description(self, char_name: str, char_data: Dict, formatted_data: str):
        """Update a character's description."""
        try:
            desc = '\n'.join(char_data['description'])
            new_prompt = (
f'''Is {char_name}'s description inaccurate or incomplete?

If yes, respond with the updated description for {char_name}.
If no, respond "unchanged".
If unsure, respond "unchanged".

REMEMBER: Do not add anything else to the response. Only respond with the updated or unchanged description.

Here is the description: """{desc}"""'''
            )
            text, stop = self.summarizer.generate_summary(
                f"This is {char_name}'s data:\n{formatted_data}\n\n{new_prompt}",
                self.custom_state
            )
            if not stop:
                char_data['description'] = '\n'.split(text)
        except Exception as e:
            print(f"{_ERROR}Error generating description for {char_name}: {e}{_RESET}")
            return ""

    def update_traits(self, char_name: str, char_data: Dict, formatted_data: str):
        """Update a character's traits."""
        prompt = f"Is {char_name}'s traits inaccurate or incomplete?"
        try:
            for i in range(len(char_data['traits'])):
                trait = char_data['traits'][i]
                new_prompt = (
f'''{prompt} If yes, respond with the updated trait for {char_name}.
If no, respond "unchanged".
If unsure, respond "unchanged".

REMEMBER: Do not add anything else to the response. Only respond with the updated or unchanged trait.

Specifically, determine if """{trait}""" is inaccurate or incomplete.'''
                )
                text, stop = self.summarizer.generate_summary(
                    f"This is {char_name}'s data:\n{formatted_data}\n\n{new_prompt}",
                    self.custom_state
                )
                if not stop:
                    char_data['traits'][i] = text
        except Exception as e:
            print(f"{_ERROR}Error generating traits for {char_name}: {e}{_RESET}")
            traceback.print_exc()
        return char_data['traits']

    def update_attributes(self, char_name: str, char_data: Dict, formatted_data: str):
        """Update a character's attributes."""
        prompt = f"Is {char_name}'s attributes inaccurate or incomplete?"
        try:
            for i in range(len(char_data['attributes'])):
                attribute = char_data['attributes'][i]
                new_prompt = (
f'''{prompt} If yes, respond with the updated attribute for {char_name}.
If no, respond "unchanged".
If unsure, respond "unchanged".

REMEMBER: Do not add anything else to the response. Only respond with the updated or unchanged attribute.

Specifically, determine if """{attribute}""" is inaccurate or incomplete.'''
                )
                text, stop = self.summarizer.generate_summary(
                    f"This is {char_name}'s data:\n{formatted_data}\n\n{new_prompt}",
                    self.custom_state
                )
                if not stop:
                    char_data['attributes'][i] = text
        except Exception as e:
            print(f"{_ERROR}Error generating attributes for {char_name}: {e}{_RESET}")
            traceback.print_exc()
        return char_data['attributes']

    def update_status(self, char_name: str, char_data: Dict, formatted_data: str):
        """Update a character's status in various organizations."""
        if 'status' not in char_data:
            return
        
        prompt = f"Is {char_name}'s status inaccurate or incomplete?"
        try:
            for org in char_data['status']:
                status = char_data['status'][org]
                new_prompt = (
f'''{prompt} If yes, respond with the updated status for {char_name}.
If no, respond "unchanged".
If unsure, respond "unchanged".

REMEMBER: Do not add anything else to the response. Only respond with the updated or unchanged status.

Specifically, determine if ```json\n{json.dumps(status, indent=2)}\n``` is inaccurate or incomplete.'''
                )
                text, stop = self.summarizer.generate_summary(
                    f"This is {char_name}'s data:\n{formatted_data}\n\n{new_prompt}",
                    self.custom_state
                )
                if not stop:
                    char_data['status'][org] = text
        except Exception as e:
            print(f"{_ERROR}Error generating status for {char_name}: {e}{_RESET}")
            traceback.print_exc()
        return char_data['status']

    def update_relationships(self, char_name: str, char_data: Dict, formatted_data: str):
        """Update a character's relationships."""
        if 'relationships' not in char_data:
            return
        
        prompt = f"Is {char_name}'s relationships inaccurate or incomplete?"
        try:
            for other_char in char_data['relationships']:
                rel = char_data['relationships'][other_char]
                new_prompt = (
f'''{prompt} If yes, respond with the updated relationship for {char_name}.
If no, respond "unchanged".
If unsure, respond "unchanged".

REMEMBER: Do not add anything else to the response. Only respond with the updated or unchanged relationship.

Specifically, determine if ```json\n{json.dumps(rel, indent=2)}\n``` is inaccurate or incomplete.'''
                )
                text, stop = self.summarizer.generate_summary(
                    f"This is {char_name}'s data:\n{formatted_data}\n\n{new_prompt}",
                    self.custom_state
                )
                if not stop:
                    char_data['relationships'][other_char] = text
        except Exception as e:
            print(f"{_ERROR}Error generating relationships for {char_name}: {e}{_RESET}")
            traceback.print_exc()
        return char_data['relationships']

class GroupSummarizer:
    def __init__(self, summarizer: Summarizer, custom_state: Dict):
        self.summarizer = summarizer
        self.data = summarizer.last
        self.custom_state = custom_state

    def generate(self) -> Dict:
        """Summarize group data."""
        print(f"{_BOLD}Summarizing groups{_RESET}")
        groups = self.summarizer.load_json_data('groups.json')
        for group_name, group_data in groups.items():
            formatted_data = self.summarizer.format_retrieval_data({ group_name: group_data }, 'groups')
            self.update_description(group_name, group_data, formatted_data)
            self.update_aliases(group_name, group_data, formatted_data)
            self.update_events(group_name, group_data, formatted_data)
            self.update_relationships(group_name, group_data, formatted_data)

        self.summarizer.save_json_data(groups, 'groups_summary.json')
        return groups

    def update_description(self, group_name: str, group_data: Dict, formatted_data: str):
        """Update a group's description."""
        prompt = f"Is {group_name}'s description inaccurate or incomplete?"
        try:
            desc = '\n'.join(group_data['description'])
            new_prompt = (
f'''{prompt} If yes, respond with the updated description for {group_name}.
If no, respond "unchanged".
If unsure, respond "unchanged".

REMEMBER: Do not add anything else to the response. Only respond with the updated or unchanged description.

Here is the description: """{desc}"""'''
            )
            text, stop = self.summarizer.generate_summary(
                f"This is {group_name}'s data:\n{formatted_data}\n\n{new_prompt}",
                self.custom_state
            )
            if not stop:
                group_data['description'] = '\n'.split(text)
        except Exception as e:
            print(f"{_ERROR}Error generating description for {group_name}: {e}{_RESET}")
            traceback.print_exc()
        return group_data['description']

    def update_aliases(self, group_name: str, group_data: Dict, formatted_data: str):
        """Update a group's aliases."""
        prompt = f"Is {group_name}'s aliases inaccurate or incomplete?"
        try:
            for i in range(len(group_data['aliases'])):
                alias = group_data['aliases'][i]
                new_prompt = (
f'''{prompt} If yes, respond with the updated alias for {group_name}.
If no, respond "unchanged".
If unsure, respond "unchanged".

REMEMBER: Do not add anything else to the response. Only respond with the updated or unchanged alias.

Specifically, determine if """{alias}""" is inaccurate or incomplete.'''
                )
                text, stop = self.summarizer.generate_summary(
                    f"This is {group_name}'s data:\n{formatted_data}\n\n{new_prompt}",
                    self.custom_state
                )
                if not stop:
                    group_data['aliases'][i] = text
        except Exception as e:
            print(f"{_ERROR}Error generating aliases for {group_name}: {e}{_RESET}")
            traceback.print_exc()
        return group_data['aliases']

    def update_events(self, group_name: str, group_data: Dict, formatted_data: str):
        """Update a group's events."""
        prompt = f"Is {group_name}'s events inaccurate or incomplete?"
        try:
            for i in range(len(group_data['events'])):
                event = group_data['events'][i]
                new_prompt = (
f'''{prompt} If yes, respond with the updated event for {group_name}.
If no, respond "unchanged".
If unsure, respond "unchanged".

REMEMBER: Do not add anything else to the response. Only respond with the updated or unchanged event.

Specifically, determine if """{event}""" is inaccurate or incomplete.'''
                )
                text, stop = self.summarizer.generate_summary(
                    f"This is {group_name}'s data:\n{formatted_data}\n\n{new_prompt}",
                    self.custom_state
                )
                if not stop:
                    group_data['events'][i] = text
        except Exception as e:
            print(f"{_ERROR}Error generating events for {group_name}: {e}{_RESET}")
            traceback.print_exc()
        return group_data['events']

    def update_relationships(self, group_name: str, group_data: Dict, formatted_data: str):
        """Update a group's relationships."""
        if 'relationships' not in group_data:
            return
        
        prompt = f"Is {group_name}'s relationships inaccurate or incomplete?"
        try:
            for other_group in group_data['relationships']:
                rel_data = group_data['relationships'][other_group]
                for field in ['position']:
                    if field in rel_data:
                        for i in range(len(rel_data[field])):
                            value = rel_data[field][i]
                            new_prompt = (
f'''{prompt} If yes, respond with the updated {field} for {group_name}'s relationship with {other_group}.
If no, respond "unchanged".
If unsure, respond "unchanged".

REMEMBER: Do not add anything else to the response. Only respond with the updated or unchanged {field}.

Specifically, determine if """{value}""" is inaccurate or incomplete.'''
                            )
                            text, stop = self.summarizer.generate_summary(
                                f"This is {group_name}'s data:\n{formatted_data}\n\n{new_prompt}",
                                self.custom_state
                            )
                            if not stop:
                                rel_data[field][i] = text
        except Exception as e:
            print(f"{_ERROR}Error generating relationships for {group_name}: {e}{_RESET}")
            traceback.print_exc()
        return group_data['relationships']

class MessageSummarizer:
    def __init__(self, summarizer: Summarizer):
        self.summarizer = summarizer
        self.custom_state = summarizer.last.custom_state
        self.chunker = summarizer.last.context[1].chunker
    
    def generate(self, messages: List[str], message_idxs: List[int]) -> Dict:
        """Summarize message data and store in vector database."""
        print(f"{_BOLD}Summarizing messages{_RESET}")

        for i, message in enumerate(messages):
            prompt = (
f'''Analyze the provided message and generate a concise summary of the key events, interactions, and developments.

REMEMBER: Do not add anything else to the response. Only respond with the summary.

Here is the message: """{message}"""'''
            )
            try:
                summary, _ = self.summarizer.generate_summary(prompt, self.custom_state)
                
                # Create and store a summary chunk
                summary_chunk = {
                    "id": f"{message_idxs[i]}_summary",
                    "text": summary,
                    "indices": [message_idxs[i], 0, 0],  # Use 0,0 to indicate this is a summary
                    "is_summary": True
                }
                self.chunker.store_chunks([summary_chunk])
            except Exception as e:
                print(f"{_ERROR}Error generating message summary: {e}{_RESET}")
                traceback.print_exc()