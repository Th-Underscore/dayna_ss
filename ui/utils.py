import os
from pathlib import Path

from modules.logging_colors import logger

import extensions.dayna_ss.shared as dss_shared

# Helper function to get multiple values from dss_shared.gradio


def gradio(*keys):
    if len(keys) == 1 and type(keys[0]) in [list, tuple]:
        keys = keys[0]

    return [dss_shared.gradio[k] for k in keys]


from modules.utils import (
    save_file,
    delete_file,
    get_available_presets,
    get_available_prompts,
    get_available_characters,
    get_available_instruction_templates,
    get_available_grammars,
)


# --- UI Elements ---


def list_model_elements():
    elements = [  # TODO: Multi-model support (far future)
        # 'filter_by_loader',
        # 'loader',
        # 'cpu_memory',
        # 'gpu_layers',
        # 'threads',
        # 'threads_batch',
        # 'batch_size',
        # 'ctx_size',
        # 'cache_type',
        # 'tensor_split',
        # 'extra_flags',
        # 'streaming_llm',
        # 'gpu_split',
        # 'alpha_value',
        # 'rope_freq_base',
        # 'compress_pos_emb',
        # 'compute_dtype',
        # 'quant_type',
        # 'num_experts_per_token',
        # 'load_in_8bit',
        # 'load_in_4bit',
        # 'torch_compile',
        # 'flash_attn',
        # 'use_flash_attention_2',
        # 'cpu',
        # 'disk',
        # 'row_split',
        # 'no_kv_offload',
        # 'no_mmap',
        # 'mlock',
        # 'numa',
        # 'use_double_quant',
        # 'use_eager_attention',
        # 'bf16',
        # 'autosplit',
        # 'enable_tp',
        # 'no_flash_attn',
        # 'no_xformers',
        # 'no_sdpa',
        # 'cfg_cache',
        # 'cpp_runner',
        # 'trust_remote_code',
        # 'no_use_fast',
        # 'model_draft',
        # 'draft_max',
        # 'gpu_layers_draft',
        # 'device_draft',
        # 'ctx_size_draft',
    ]

    return elements


def list_interface_input_elements():
    elements = [
        "temperature",
        "dynatemp_low",
        "dynatemp_high",
        "dynatemp_exponent",
        "smoothing_factor",
        "smoothing_curve",
        "min_p",
        "top_p",
        "top_k",
        "typical_p",
        "xtc_threshold",
        "xtc_probability",
        "epsilon_cutoff",
        "eta_cutoff",
        "tfs",
        "top_a",
        "top_n_sigma",
        "dry_multiplier",
        "dry_allowed_length",
        "dry_base",
        "repetition_penalty",
        "frequency_penalty",
        "presence_penalty",
        "encoder_repetition_penalty",
        "no_repeat_ngram_size",
        "repetition_penalty_range",
        "penalty_alpha",
        "guidance_scale",
        "mirostat_mode",
        "mirostat_tau",
        "mirostat_eta",
        "max_new_tokens",
        "prompt_lookup_num_tokens",
        "max_tokens_second",
        "max_updates_second",
        "do_sample",
        "dynamic_temperature",
        "temperature_last",
        "auto_max_new_tokens",
        "ban_eos_token",
        "add_bos_token",
        "enable_thinking",
        "skip_special_tokens",
        "stream",
        "static_cache",
        "truncation_length",
        "seed",
        "sampler_priority",
        "custom_stopping_strings",
        "custom_token_bans",
        "negative_prompt",
        "dry_sequence_breakers",
        "grammar_string",
    ]

    # Chat elements
    elements += [
        "unique_id",
        "start_with",
        "mode",
        "chat-instruct_command",
        "character_menu",
        "name2",
        "context",
        "greeting",
        "name1",
        "user_bio",
        "custom_system_message",
        "instruction_template_str",
        "chat_template_str",
        "dss_toggle",
        "dss_instr_prompt_template",
    ]

    # Model elements
    elements += list_model_elements()

    return elements


def gather_interface_values(*args):
    interface_elements = list_interface_input_elements()

    output = {}
    for element, value in zip(interface_elements, args):
        output[element] = value
        dss_shared.persistent_ui_state[element] = value

    return output
