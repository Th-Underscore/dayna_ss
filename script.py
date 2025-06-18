import re
from subprocess import Popen
import ast
import html
from pathlib import Path
import asyncio
import threading
import time
from typing import Coroutine
import traceback

import modules.shared as shared

import extensions.dayna_ss.shared as dss_shared
from extensions.dayna_ss.agents.summarizer import Summarizer

from extensions.dayna_ss.utils.helpers import (
    _ERROR,
    _SUCCESS,
    _INPUT,
    _GRAY,
    _HILITE,
    _BILITE,
    _BOLD,
    _RESET,
    _DEBUG,
    History,
    Histories,
    extract_meaningful_paragraphs,
)

params = {"display_name": "DSS", "is_tab": True}

# === Internal constants (don't change these without good reason) ===
_CONFIG_PATH = "extensions/dayna_ss/dss_config.json"

# Global event loop for background tasks
_background_loop = None
_loop_thread = None

_last_generation = {}


# def input_modifier(user_input: str, state: dict, is_chat=False):
#     next_scene_prefix = "NEXT SCENE:"

#     if summarizer and not summarizer.last:
#         current_context = summarizer.get_general_summarization(state)
#         summarizer.get_retrieval_context(state, state["history"]["internal"], current_context)

#     if user_input.startswith(next_scene_prefix) and summarizer and summarizer.last:
#         print(f"{_DEBUG}Found '{next_scene_prefix}' in user input.{_RESET}")
#         summarizer.last.is_new_scene_turn = True
#         user_input = user_input[len(next_scene_prefix) :].lstrip()

#     # TODO: Wait until summarization over
#     return user_input


def chat_input_modifier(text: str, visible_text: str, state: dict):
    if not dss_shared.persistent_ui_state.get("dss_toggle", True):
        return text, visible_text
    
    next_scene_prefix = "NEXT SCENE:"

    if summarizer and not summarizer.last:
        current_context = summarizer.get_general_summarization(state)
        summarizer.get_retrieval_context(state, state["history"]["internal"], current_context)

    if text.startswith(next_scene_prefix) and summarizer and summarizer.last:
        print(f"{_DEBUG}Found '{next_scene_prefix}' in user input.{_RESET}")
        summarizer.last.is_new_scene_turn = True
        text = text[len(next_scene_prefix) :].lstrip()
        visible_text = visible_text[len(next_scene_prefix) :].lstrip()  # Assuming visible_text prefix is the same

    # TODO: Wait until summarization over
    return text, visible_text


def custom_generate_chat_prompt(user_input: str, state: dict, history: Histories, **kwargs):
    global summarizer, _last_generation
    from modules.chat import generate_chat_prompt

    if not shared.model:
        return generate_chat_prompt(user_input, state, **kwargs)
    if not dss_shared.persistent_ui_state.get("dss_toggle", True):
        return generate_chat_prompt(user_input, state, **kwargs)

    if not summarizer:
        summarizer = Summarizer(_CONFIG_PATH)

    handle_input(user_input, state, history)

    try:
        start = time.time()
        impersonate = kwargs.get("impersonate")
        result = dss_shared.update_config(state)
        print(
            f"{_BILITE}custom_generate_chat_prompt{_RESET}",
            "impersonate",
            impersonate,
            "user_input",
            user_input,
        )

        index = len(history["internal"]) * 2  # User input index ([:-1].__len__() + 2 - 2)

        instr_prompt, custom_state, history_path, timestamp_str = summarizer.generate_summary_instr_prompt(
            user_input, state, history["internal"], **kwargs
        )
        if timestamp_str:
            # TODO: Only rewrite if not _continue or regenerate
            summarizer.save_message_chunks(user_input, index, timestamp_str)

        kwargs.pop("_continue", False)
        prompt = generate_chat_prompt(instr_prompt, custom_state, **kwargs)
        with open(history_path.parent / "dump.txt", "a") as f:
            dump_str = "==========================\n"
            dump_str += "==========================\n"
            dump_str += "==========================\n"
            dump_str += "==========================\n\n"
            dump_str += prompt
            f.write(dump_str)
            f.close()
        print(f"{_BILITE}Prompt generated in {time.time() - start:.2f} seconds.{_RESET}")
        dss_shared.custom_state.value = custom_state
        return prompt
    except Exception as e:
        print(f"{_ERROR}Error generating custom summarization prompt:{_RESET} {str(e)}")
        traceback.print_exc()
        if summarizer.last and summarizer.last.original_seed:
            state["seed"] = summarizer.last.original_seed
        return generate_chat_prompt(user_input, state, **kwargs)


def handle_input(user_input: str, state: dict, history: Histories):
    global summarizer, story_rag
    if not summarizer or not story_rag:
        return

    print("handle_input")

    if summarizer and not summarizer.last:
        print(f"handle_input{_BOLD}no last summarization{_RESET}")
        current_context = summarizer.get_general_summarization(state)
        summarizer.get_retrieval_context(state, history["internal"], current_context)


def handle_output(output: str, state: dict, history: Histories):
    """Generate new history data using input and output after generation."""
    global summarizer, story_rag, _last_generation

    if not dss_shared.persistent_ui_state.get("dss_toggle", True) or not summarizer or not story_rag or not shared.model:
        return

    print("handle_output")

    async def process_summarizations():
        start = time.time()
        index = len(history["internal"]) * 2 - 1  # Output index
        try:
            print(f"{_BILITE}process_summarizations{_RESET}")
            # Reset sequence before processing new batch of prompts
            # summarizer.vram_manager.reset_sequence()

            timestamp_str = summarizer.summarize_latest_state(output, history["internal"][-1][0], state, history["internal"])
            if timestamp_str:
                summarizer.save_message_chunks(output, index, timestamp_str)

        except Exception as e:
            print(f"{_ERROR}Error during summarization:{_RESET} {str(e)}")
            traceback.print_exc()

        state["seed"] = summarizer.last.original_seed

        print(f"{_BILITE}Summarization completed in {time.time() - start:.2f} seconds.{_RESET}")

    # Create task and let it run in background
    run_async(process_summarizations())

    return


def strip_prefix(text: str, banned_prefixes: list) -> str:
    """Strip user-defined prefixes from the text."""
    import re

    for prefix in banned_prefixes:
        # prefix_match = re.match(rf'^{re.escape(prefix)}\s*', text, flags=re.IGNORECASE)
        if text.startswith(prefix):
            text = text.removeprefix(prefix).lstrip()
            return text
        text = re.sub(
            rf"^{re.escape(prefix)}\s*", "", text, count=1, flags=re.IGNORECASE
        )  # Doesn't work for some reason? re.match does though
    return text


def ensure_background_loop():
    """Ensure a background event loop is running in a separate thread"""
    global _background_loop, _loop_thread

    def run_event_loop(loop: asyncio.AbstractEventLoop):
        asyncio.set_event_loop(loop)
        loop.run_forever()

    if _background_loop is None:
        _background_loop = asyncio.new_event_loop()
        _loop_thread = threading.Thread(target=run_event_loop, args=(_background_loop,), daemon=True)
        _loop_thread.start()


def setup():
    """Initialize the extension"""
    global summarizer, story_rag
    print("Loaded DAYNA Story Summarizer!")

    # Initialize summarizer
    summarizer = Summarizer(_CONFIG_PATH)
    story_rag = True
    # story_rag = summarizer.story_rag


def run_async(coro: Coroutine) -> asyncio.Future | None:
    """Run an async function in the current thread.

    Args:
        coro (Coroutine): An async coroutine to be run.

    Returns:
        out (Future, optional): The result of the coroutine if successful, otherwise None.
    """
    print(f"{_BOLD}Running async:{_RESET} {coro}")
    try:
        ensure_background_loop()
        future = asyncio.run_coroutine_threadsafe(coro, _background_loop)
        return future
    except Exception as e:
        print(f"{_ERROR}Error in async execution:{_RESET} {str(e)}")
        return None


# // Gradio UI // #

import gradio as gr

from extensions.dayna_ss.ui import ui_chat, ui_file_saving, ui_parameters, utils

params["is_tab"] = True
print(f"{_BOLD}Initial params: {_RESET}{params['is_tab']} ({params})")
tab_created = False


def ui():
    global tab_created, params

    dss_shared.input_elements = utils.list_interface_input_elements()

    print(f"{_BOLD}Creating UI: {_RESET}{tab_created} - {params['is_tab']} ({params})")

    if not tab_created:
        params["is_tab"] = True
        dss_shared.gradio["interface_state"] = gr.State({k: None for k in dss_shared.input_elements})

        ui_file_saving.create_ui()
        ui_chat.create_ui()
        ui_parameters.create_ui(dss_shared.settings["preset"])

        params["is_tab"] = False
        tab_created = True
    else:
        params["is_tab"] = False
        ui_chat.create_block_ui()
        params["is_tab"] = True
        tab_created = False

        ui_chat.create_event_handlers()
        ui_file_saving.create_event_handlers()
        ui_parameters.create_event_handlers()


is_final_output = False


def output_modifier(string, state, is_chat=False):
    global is_final_output
    is_final_output = True
    return string


# // TGWUI Monkey Patches // #


"""generate_chat_reply_wrapper"""
import modules.chat as chat

_generate_chat_reply = chat.generate_chat_reply


def generate_chat_reply(text, state, regenerate=False, _continue=False, loading_message=True, for_ui=False):
    """
    Same as above but returns HTML for the UI
    """
    global is_final_output
    history = state["history"]

    if not dss_shared.persistent_ui_state.get("dss_toggle", True):
        for history in _generate_chat_reply(
            text,
            state,
            regenerate=regenerate,
            _continue=_continue,
            loading_message=loading_message,
            for_ui=for_ui,
        ):
            yield history
        is_final_output = False
        return

    banned_prefixes = dss_shared.persistent_ui_state.get("banned_prefixes", dss_shared.settings["banned_prefixes"])
    print(f"banned_prefixes {_DEBUG} {banned_prefixes}{_RESET}")
    if type(banned_prefixes) is str:
        banned_prefixes = ast.literal_eval(f"[{banned_prefixes}]")
    elif not (type(banned_prefixes) is list and len(banned_prefixes) > 0):
        print(f"{_ERROR}Invalid prefix format in banned_prefixes:{_RESET} {banned_prefixes}")
        banned_prefixes = ""
    banned_prefixes = [chat.replace_character_names(prefix, state["name1"], state["name2"]) for prefix in banned_prefixes]

    # --- Injection --- #
    for history in _generate_chat_reply(
        text,
        state,
        regenerate=regenerate,
        _continue=_continue,
        loading_message=loading_message,
        for_ui=for_ui,
    ):
        current_reply_internal: str = history["internal"][-1][1]

        if current_reply_internal and not is_final_output:
            processed_reply = extract_meaningful_paragraphs(current_reply_internal)

            for prefix_to_strip in banned_prefixes:
                processed_reply = re.sub(f"^{re.escape(prefix_to_strip)}\s*", "", processed_reply, count=1, flags=re.IGNORECASE)

            history["internal"][-1][1] = processed_reply

        if history["internal"][-1][1] is not None:
            history["visible"][-1][1] = html.escape(history["internal"][-1][1])
        else:
            history["visible"][-1][1] = ""

        yield history
    is_final_output = False

    handle_output(history["internal"][-1][1], state, history)  # Internal output


chat.generate_chat_reply = generate_chat_reply

#  //  //  #


def debug(msg: str):
    print(f"\033[0;{30}m" + msg)


print("----------")
print("DSS CONFIG")
print("----------")
print("change these values in dss_config.json")
# pprint.pprint(_CONFIG)
print("----------")
print()
