import re
import ast
import html
import asyncio
import socket
import concurrent.futures
import threading
import time
from typing import Coroutine
import traceback

import modules.shared as shared

from . import shared as dss_shared
from .agents.summarizer import Summarizer
from .tools import tgwui_integration
from .tools.definitions.dynamic_tools import create_dss_tool_definitions
from .ui.sse_server import start_sse_server, stop_sse_server

from .utils.helpers import (
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
    strip_thinking,
)

params = {
    "display_name": "DSS",
    "is_tab": True,
    "sse_internal_port": 7880,
    "sse_external_path": ":7880",
}


def _find_available_port(start_port, max_attempts=10):
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            continue
    return -1

# === Internal constants (don't change these without good reason) ===
from .shared import EXTENSION_DIR
_CONFIG_PATH = EXTENSION_DIR / "dss_config.json"

# Global event loop for background tasks
_background_loop = None
_loop_thread = None

_last_generation = {}


# def input_modifier(user_input: str, state: dict, is_chat=False):
#     next_scene_prefix = "NEXT SCENE:"

#     if summarizer and not summarizer.last:
#         current_context = state["context"]
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
    force_chapter_prefix = "NEXT CHAPTER:"
    force_arc_prefix = "NEXT ARC:"

    if summarizer and not summarizer.last:
        current_context = state["context"]
        summarizer.get_retrieval_context(state, state["history"]["internal"], current_context)

    if text.startswith(next_scene_prefix) and summarizer and summarizer.last:
        print(f"{_DEBUG}Found '{next_scene_prefix}' in user input.{_RESET}")
        summarizer.last.is_new_scene_turn = True
        text = text[len(next_scene_prefix) :].lstrip()
        visible_text = visible_text[len(next_scene_prefix) :].lstrip()

    if text.startswith(force_chapter_prefix) and summarizer and summarizer.last:
        print(f"{_DEBUG}Found '{force_chapter_prefix}' in user input.{_RESET}")
        summarizer.last.is_new_scene_turn = True
        summarizer.last.force_next_chapter = True
        text = text[len(force_chapter_prefix) :].lstrip()
        visible_text = visible_text[len(force_chapter_prefix) :].lstrip()

    if text.startswith(force_arc_prefix) and summarizer and summarizer.last:
        print(f"{_DEBUG}Found '{force_arc_prefix}' in user input.{_RESET}")
        summarizer.last.is_new_scene_turn = True
        summarizer.last.force_next_arc = True
        text = text[len(force_arc_prefix) :].lstrip()
        visible_text = visible_text[len(force_arc_prefix) :].lstrip()

    return text, visible_text


def custom_generate_chat_prompt(user_input: str, state: dict, history: Histories, **kwargs):
    global summarizer, _last_generation
    from modules.chat import generate_chat_prompt

    if not shared.model or not dss_shared.persistent_ui_state.get("dss_toggle", True):
        return generate_chat_prompt(user_input, state, **kwargs)

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
        
        kwargs.setdefault("do_instr", dss_shared.persistent_ui_state.get("do_instr", True))

        index = len(history["internal"]) * 2  # User input index ([:-1].__len__() + 2 - 2)

        instr_prompt, custom_state, history_path, timestamp_str = summarizer.generate_instr_prompt(
            user_input, state, history["internal"], **kwargs
        )
        if shared.stop_everything:
            print(f"{_HILITE}Stop signal received after instruction prompt generation.{_RESET}")
            shared.stop_everything = False
            return ""

        tgwui_integration.add_dss_tools_to_state(custom_state, tgwui_integration._dss_tool_definitions)
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
    # from .utils.model_loader import load_secondary_model
    # model_2, tokenizer_2 = load_secondary_model("Lucy-128k-Q6_K.gguf", {})
    # print(_HILITE, model_2, tokenizer_2, _RESET)

    if summarizer and not summarizer.last:
        print(f"handle_input{_BOLD}no last summarization{_RESET}")
        current_context = state["context"]
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
            if shared.stop_everything:
                state["seed"] = summarizer.last.original_seed
                print(f"{_HILITE}Stop signal received after summarization.{_RESET}")
                shared.stop_everything = False
                return ""
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
    """
    Start a background asyncio event loop in a dedicated daemon thread if one is not already running.

    If no background loop exists, create a new asyncio event loop and start it on a daemon thread; the created loop and thread are stored in the module-level globals `_background_loop` and `_loop_thread`.
    """
    global _background_loop, _loop_thread

    def run_event_loop(loop: asyncio.AbstractEventLoop):
        asyncio.set_event_loop(loop)
        loop.run_forever()

    if _background_loop is None:
        _background_loop = asyncio.new_event_loop()
        _loop_thread = threading.Thread(target=run_event_loop, args=(_background_loop,), daemon=True)
        _loop_thread.start()


_sse_port = 7880  # Default SSE port


def setup():
    """
    Initialize and configure the DAYNA Story Summarizer extension.

    Creates the Summarizer instance and enables story retrieval, registers tool
    definitions and the DSS-enabled callback with TGWUI integration, starts the
    internal SSE server on the configured port, and prints the startup status.

    Modifies globals:
    - summarizer: set to a new Summarizer instance
    - story_rag: set to True
    - _sse_port: set to the port returned by start_sse_server

    Side effects:
    - Registers tool definitions in tgwui_integration._dss_tool_definitions
    - Registers a DSS enabled-check callback via tgwui_integration.set_dss_enabled_check
    - Attempts to start an SSE server and prints success or failure to stdout
    """
    global summarizer, story_rag, _sse_port
    print("Loaded DAYNA Story Summarizer!")

    summarizer = Summarizer(_CONFIG_PATH)
    story_rag = True

    tool_defs = create_dss_tool_definitions()
    tgwui_integration._dss_tool_definitions = tool_defs

    def dss_enabled_check():
        """
        Check whether the Dynamic Story Summarizer (DSS) feature is enabled in the persistent UI state.

        Returns:
            True if the DSS toggle (`dss_toggle`) in persistent UI state is enabled, False otherwise.
        """
        return dss_shared.persistent_ui_state.get("dss_toggle", True)
    tgwui_integration.set_dss_enabled_check(dss_enabled_check)

    _sse_port = start_sse_server(host="127.0.0.1", port=params["sse_internal_port"])
    if _sse_port > 0:
        params["sse_internal_port"] = _sse_port
        print(f"{_SUCCESS}DSS SSE server started on port {_sse_port}{_RESET}")
    else:
        print(f"{_ERROR}Failed to start DSS SSE server{_RESET}")


def run_async(coro: Coroutine) -> concurrent.futures.Future | None:
    """
    Schedule a coroutine to run on the module's background asyncio event loop.

    Parameters:
        coro (Coroutine): The coroutine to schedule.

    Returns:
        future (concurrent.futures.Future | None): A future representing the scheduled coroutine if scheduling succeeded, or `None` on failure.
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

from .ui import ui_chat, ui_file_saving, ui_parameters, ui_templates, utils

# --- HTML + JS for the real-time SSE status panel --- #
_SSE_PANEL_HTML = """<div id="dss-status-panel" style="font-family: monospace; font-size: 13px; background: #0d1117; color: #c9d1d9; border-radius: 8px; padding: 16px; min-height: 200px; max-height: 600px; overflow-y: auto;">
    <div id="dss-debug" style="color: #ff4444; font-size: 11px; margin-bottom: 8px;">JS NOT LOADED</div>
    <div id="dss-session-info" style="margin-bottom: 12px; color: #8b949e; font-size: 12px;">
        Waiting for summarization...
    </div>
    <div id="dss-progress-wrap" style="margin-bottom: 12px;">
        <div style="background: #21262d; border-radius: 4px; height: 8px; overflow: hidden;">
            <div id="dss-progress-bar" style="background: linear-gradient(90deg, #238636, #2ea043); height: 100%; width: 0%; transition: width 0.3s;"></div>
        </div>
        <span id="dss-progress-text" style="font-size: 11px; color: #8b949e;">0%</span>
    </div>
    <div id="dss-queue-section" style="margin-bottom: 8px;">
        <div style="color: #8b949e; font-size: 11px; text-transform: uppercase; margin-bottom: 4px;">Queue</div>
        <div id="dss-queue-list"></div>
    </div>
    <div id="dss-phases"></div>
</div>"""


def custom_js():
    """
    Render and return the extension's SSE client JavaScript.

    Loads the ui/ui.js.j2 Jinja2 template and renders it with the configured SSE external path from params.

    Returns:
        str: Rendered JavaScript; an empty string if the template cannot be loaded or rendered.
    """
    import os
    from jinja2 import Environment, FileSystemLoader, select_autoescape, TemplateError
    js_path = os.path.join(os.path.dirname(__file__), "ui")
    try:
        env = Environment(
            loader=FileSystemLoader(js_path),
            autoescape=select_autoescape(['js', 'j2'])
        )
        template = env.get_template("ui.js.j2")
        return template.render(sse_external_path=params.get("sse_external_path", ":7880"))
    except TemplateError as e:
        print(f"{_ERROR}Failed to load ui.js.j2: {e}{_RESET}")
        return ""

params["is_tab"] = True
print(f"{_BOLD}Initial params: {_RESET}{params['is_tab']} ({params})")
tab_created = False


def ui():
    """
    Builds and registers the extension's Gradio user interface and associated event handlers.

    When called the first time, creates the extension as a tab (including file saving, chat, parameters, templates UIs and the DSS real-time status panel) and initializes shared interface state. On subsequent calls, creates the UI as a block under the main chat, registers UI event handlers, and updates module-level flags and `params["is_tab"]` to reflect the current layout.
    """
    global tab_created, params

    dss_shared.input_elements = utils.list_interface_input_elements()

    print(f"{_BOLD}Creating UI: {_RESET}{tab_created} - {params['is_tab']} ({params})")

    if not tab_created:  # Tab
        params["is_tab"] = True
        dss_shared.gradio["interface_state"] = gr.State({k: None for k in dss_shared.input_elements})

        ui_file_saving.create_ui()
        ui_chat.create_ui()
        ui_parameters.create_ui(dss_shared.settings["preset"])
        ui_templates.create_ui()

        params["is_tab"] = False
        tab_created = True
    else:  # Block under chat
        params["is_tab"] = False

        ui_chat.create_block_ui()
        with gr.Accordion("DSS Real-time Status", open=True):
            gr.HTML(value=_SSE_PANEL_HTML, elem_id="dss-sse-panel")

        params["is_tab"] = True
        tab_created = False



        ui_chat.create_event_handlers()
        ui_file_saving.create_event_handlers()
        ui_parameters.create_event_handlers()
        ui_templates.create_event_handlers()


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
            processed_reply = strip_thinking(current_reply_internal) or current_reply_internal
            processed_reply = extract_meaningful_paragraphs(processed_reply)

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