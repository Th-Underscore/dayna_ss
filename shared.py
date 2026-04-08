import json
import jsonc
import os
from collections import deque
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any

import gradio as gr

EXTENSION_DIR = Path(__file__).parent

# import modules.shared as shared
# from modules.chat import generate_chat_prompt
# from modules.html_generator import fix_newlines

if TYPE_CHECKING:
    from .agents.summarizer import Summarizer
import modules.shared as shared

from .utils.helpers import (
    _ERROR,
    _SUCCESS,
    _INPUT,
    _GRAY,
    _HILITE,
    _BOLD,
    _WARNING,
    _RESET,
    _DEBUG,
)

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
    with open(EXTENSION_DIR / "default_config.json", "rt") as default_handle:
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

def _init_template_settings():
    """Initialize template settings from format_templates.json."""
    try:
        from .utils.helpers import load_json
        template_path = EXTENSION_DIR / "user_data" / "example" / "format_templates.json"
        templates = load_json(template_path) or {}
        for key in templates:
            settings[f"template_{key}"] = ""
        print(f"{_DEBUG}Initialized {len(templates)} template settings{_RESET}")
    except Exception as e:
        print(f"{_WARNING}Failed to initialize template settings: {e}{_RESET}")

_init_template_settings()

persistent_ui_state = {}
custom_state = gr.State()
banned_prefixes = [
    '"{{char}}:", "{{char}} >>", "(as {{char}})", "({{char}})"',
    '"{{user}}:", "{{user}} >>", "(as {{user}})", "({{user}})"',
]


class ActivityLogger:
    """Thread-safe activity logger for tracking DSS agent operations in real-time."""
    
    MAX_ENTRIES = 100
    
    def __init__(self, max_entries: int = MAX_ENTRIES):
        self._entries: deque[dict[str, Any]] = deque(maxlen=max_entries)
        self._lock = Lock()
    
    def log(self, event: str, details: str = "", level: str = "info") -> None:
        """Add an activity entry.
        
        Args:
            event: Short name of the event (e.g., "Summarizing", "Tool Call")
            details: Additional details about the event
            level: Log level - "info", "success", "warning", "error"
        """
        with self._lock:
            self._entries.append({
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "event": event,
                "details": details,
                "level": level,
            })
    
    def get_entries(self) -> list[dict[str, Any]]:
        """Get all log entries as a list."""
        with self._lock:
            return list(self._entries)
    
    def clear(self) -> None:
        """Clear all log entries."""
        with self._lock:
            self._entries.clear()
    
    def render_html(self, max_display: int = 50) -> str:
        """Render entries as HTML for display in Gradio.
        
        Args:
            max_display: Maximum number of entries to display
            
        Returns:
            HTML string for display
        """
        entries = self.get_entries()
        if not entries:
            return '<div class="dss-activity-empty" style="font-family: monospace; font-size: 12px; color: #888; padding: 10px;">Activity log is empty</div>'
        
        entries = entries[-max_display:]
        
        lines = [
            '<div class="dss-activity-feed" style="font-family: monospace; font-size: 12px; background: #1a1a2e; border-radius: 8px; padding: 10px; max-height: 300px; overflow-y: auto; color: #eee;">'
        ]
        for entry in entries:
            lines.append(self._render_entry_html(entry))
        lines.append('</div>')
        
        return '\n'.join(lines)
    
    def _render_entry_html(self, entry: dict[str, Any]) -> str:
        """Render a single entry as HTML."""
        timestamp = entry["timestamp"].split("T")[1] if "T" in entry["timestamp"] else entry["timestamp"]
        
        level_colors = {
            "info": "#00bfff",
            "success": "#00ff00",
            "warning": "#ffaa00",
            "error": "#ff4444",
        }
        level_names = {
            "info": "INFO",
            "success": "OK",
            "warning": "WARN",
            "error": "ERR",
        }
        
        color = level_colors.get(entry["level"], "#ffffff")
        level_name = level_names.get(entry["level"], "INFO")
        
        details_html = ""
        if entry["details"]:
            details_html = f'<span class="dss-activity-details" style="color: #aaa;"> - {entry["details"]}</span>'
        
        return (
            f'<div class="dss-activity-entry" data-level="{entry["level"]}" style="margin: 4px 0; padding: 2px 0; border-bottom: 1px solid #333;">'
            f'<span class="dss-activity-time" style="color: #666;">[{timestamp}]</span> '
            f'<span class="dss-activity-level" style="color: {color}; font-weight: bold;">[DSS {level_name}]</span> '
            f'<span class="dss-activity-event">{entry["event"]}</span>'
            f'{details_html}'
            f'</div>'
        )


activity_logger = ActivityLogger()

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
