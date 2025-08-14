import json
import jsonc
import os
from typing import TYPE_CHECKING

import gradio as gr

# import modules.shared as shared
# from modules.chat import generate_chat_prompt
# from modules.html_generator import fix_newlines

if TYPE_CHECKING:
    from extensions.dayna_ss.agents.summarizer import Summarizer
import modules.shared as shared

global current_character
current_character: str = shared.settings["character"]


def load_json_with_fallback(file_path: str) -> dict:
    """Load a JSON file with fallback to default configuration if the file is empty or doesn't exist.

    Args:
        file_path (str): Path to the JSON file to load

    Returns:
        out (dict): The loaded configuration, either from the file or the default
    """
    try:
        with open(file_path, "rt") as handle:
            content = jsonc.load(handle) or _load_and_save_default(file_path)
            print(f"Loaded config from {file_path}")
            return content
    except FileNotFoundError:
        print(f"Warning: Could not load config from {file_path}, using defaults")
        return _load_and_save_default(file_path)


def _load_and_save_default(file_path: str) -> dict:
    """Internal function to load default config and save it to the specified path.

    Args:
        file_path (str): Path where to save the default configuration

    Returns:
        out (dict): The default configuration
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # Load default config
    with open("./extensions/dayna_ss/default_config.json", "rt") as default_handle:
        default_config = jsonc.load(default_handle)
    # Save the default config
    with open(file_path, "wt") as handle:
        json.dump(default_config, handle, indent=4)
    return default_config


def update_config(state: dict) -> bool:
    """Update the current config state based on the current selected generation character.

    Returns:
        bool: True if updated, false if not.
    """
    global current_character
    print(state["character_menu"])
    print(current_character)
    if state["character_menu"] != current_character:
        current_character = state["character_menu"]
        return True
    return False


global summarizer
summarizer: "Summarizer"

global gradio
gradio = {}
settings = {
    "show_controls": True,
    "start_with": "",
    "mode": "instruct",
    "chat_style": "cai-chat",
    "chat-instruct_command": 'Continue the chat dialogue below. Write a single reply for the character "<|character|>".\n\n<|prompt|>',
    "prompt-default": "QA",
    "prompt-notebook": "QA",
    "character": "Assistant",
    "name1": "You",
    "user_bio": "",
    "custom_system_message": "",
    "preset": "min_p",
    "max_new_tokens": 4096,
    "max_new_tokens_min": 1,
    "max_new_tokens_max": 8192,
    "prompt_lookup_num_tokens": 0,
    "max_tokens_second": 0,
    "max_updates_second": 12,
    "auto_max_new_tokens": True,
    "ban_eos_token": False,
    "add_bos_token": True,
    "enable_thinking": True,
    "skip_special_tokens": True,
    "stream": True,
    "static_cache": False,
    "truncation_length": 8192,
    "seed": -1,
    "custom_stopping_strings": "",
    "custom_token_bans": "",
    "negative_prompt": "",
    "dark_theme": True,
    "default_extensions": [],
    "instruction_template_str": "{%- set ns = namespace(found=false) -%}\n{%- for message in messages -%}\n    {%- if message['role'] == 'system' -%}\n        {%- set ns.found = true -%}\n    {%- endif -%}\n{%- endfor -%}\n{%- if not ns.found -%}\n    {{- '' + 'Below is an instruction that describes a task. Write a response that appropriately completes the request.' + '\\n\\n' -}}\n{%- endif %}\n{%- for message in messages %}\n    {%- if message['role'] == 'system' -%}\n        {{- '' + message['content'] + '\\n\\n' -}}\n    {%- else -%}\n        {%- if message['role'] == 'user' -%}\n            {{-'### Instruction:\\n' + message['content'] + '\\n\\n'-}}\n        {%- else -%}\n            {{-'### Response:\\n' + message['content'] + '\\n\\n' -}}\n        {%- endif -%}\n    {%- endif -%}\n{%- endfor -%}\n{%- if add_generation_prompt -%}\n    {{-'### Response:\\n'-}}\n{%- endif -%}",
    "chat_template_str": "{%- for message in messages %}\n    {%- if message['role'] == 'system' -%}\n        {%- if message['content'] -%}\n            {{- message['content'] + '\\n\\n' -}}\n        {%- endif -%}\n        {%- if user_bio -%}\n            {{- user_bio + '\\n\\n' -}}\n        {%- endif -%}\n    {%- else -%}\n        {%- if message['role'] == 'user' -%}\n            {{- name1 + ': ' + message['content'] + '\\n'-}}\n        {%- else -%}\n            {{- name2 + ': ' + message['content'] + '\\n' -}}\n        {%- endif -%}\n    {%- endif -%}\n{%- endfor -%}",
    "banned_prefixes": '"{{char}}:", "(as {{char}})"',
}

persistent_ui_state = {}
custom_state = gr.State()
banned_prefixes = [
    '"{{char}}:", "{{char}} >>", "(as {{char}})", "({{char}})"',
    '"{{user}}:", "{{user}} >>", "(as {{user}})", "({{user}})"',
]

# settings = {
#     # Curve shape
#     "temperature": 0.7,
#     "dynatemp_low": 1.0,
#     "dynatemp_high": 1.0,
#     "dynatemp_exponent": 1.0,
#     "smoothing_factor": 0.0,
#     "smoothing_curve": 1.0,

#     # Curve cutoff
#     "preset": 'min_p',
#     "min_p": 0.0,
#     "top_n_sigma": 0.0,
#     "top_p": 0.9,
#     "top_k": 20,
#     "typical_p": 1.0,
#     "xtc_threshold": 0.05,
#     "xtc_probability": 0.0,
#     "epsilon_cutoff": 0,
#     "eta_cutoff": 0,
#     "tfs": 1.0,
#     "top_a": 0.0,

#     # Repetition suppression
#     "dry_multiplier": 1.0,
#     "dry_allowed_length": 0,
#     "dry_base": 0.5,
#     "repetition_penalty": 1.15,
#     "frequency_penalty": 0.0,
#     "presence_penalty": 0.0,
#     "encoder_repetition_penalty": 1.0,
#     "no_repeat_ngram_size": 0,
#     "repetition_penalty_range": 0,

#     # Alternative sampling methods
#     "penalty_alpha": 0.0,
#     "mirostat_mode": 0,
#     "mirostat_tau": 5.0,
#     "mirostat_eta": 0.1,

#     # Other options
#     "max_new_tokens": 512,
#     "max_new_tokens_min": 1,
#     "max_new_tokens_max": 4096,
#     "prompt_lookup_num_tokens": 0,
#     "max_tokens_second": 0,
#     "max_updates_second": 2,
#     "do_sample": True,
#     "dynamic_temperature": False,
#     "temperature_last": False,
#     "auto_max_new_tokens": False,
#     "ban_eos_token": False,
#     "add_bos_token": False,
#     "enable_thinking": False,
#     "skip_special_tokens": True,
#     "stream": True,
#     "static_cache": False,
#     "truncation_length": 2048,
#     "seed": -1,
#     "sampler_priority": "repetition_penalty\npresence_penalty\nfrequency_penalty\ndry\ntop_n_sigma\ntemperature\ndynamic_temperature\nquadratic_sampling\ntop_k\ntop_p\ntypical_p\nepsilon_cutoff\neta_cutoff\ntfs\ntop_a\nmin_p\nmirostat\nxtc\nencoder_repetition_penalty\nno_repeat_ngram",
#     "custom_stopping_strings": "",
#     "custom_token_bans": "",
#     "negative_prompt": "", # Also part of CFG with guidance_scale
#     "dry_sequence_breakers": "",
#     "grammar_file": "",
#     "grammar_string": "",

#     # Instruction and Chat Templates
#     "instruction_template": "Alpaca",
#     "instruction_template_str": "",
#     "instruction_templates": {},
#     "custom_system_message": "",
#     "chat_template_str": "",

#     # Character and User settings
#     "character_menu": "None",
#     "name1": "You",
#     "your_name": "You",
#     "user_bio": "",
#     "name2": "Assistant",
#     "context": "This is a conversation with an AI assistant.",
#     "greeting": "",

#     # Mode and Guidance
#     "mode": "chat", # Default mode (chat, instruct, chat-instruct)
#     "guidance_scale": 1.0, # For CFG (Classifier-Free Guidance)
# }
