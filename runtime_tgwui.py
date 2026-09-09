"""
TextGen adapter for the DSS runtime seam.

Binds the framework-agnostic ``runtime`` singleton to TextGen's own modules
(``modules.shared``, ``modules.chat``, ``modules.text_generation``) and the
extension's ``shared`` (dss_shared) config. This is the only core-adjacent
module that is allowed to import ``modules.*`` / ``gradio``.
"""
from __future__ import annotations

import torch

import modules.chat as chat
import modules.shared as shared
from modules.text_generation import encode

from . import shared as dss_shared
from .runtime import runtime
from .tools import tgwui_integration


def configure_runtime() -> None:
    """Bind the DSS runtime to TextGen's live model, prompt builder, and state."""
    runtime.configure(
        model_provider=lambda: shared.model,
        stop_provider=lambda: shared.stop_everything,
        prompt_builder=chat.generate_chat_prompt,
        encoder=encode,
        activity_logger=dss_shared.activity_logger,
        persistent_ui_state_provider=lambda: dss_shared.persistent_ui_state,
        current_character_provider=lambda: dss_shared.current_character,
        settings_provider=lambda: dss_shared.settings,
        update_config_fn=dss_shared.update_config,
        register_tool_executors_fn=tgwui_integration.register_dss_tool_executors,
        extension_dir=dss_shared.EXTENSION_DIR,
        no_grad_cm=torch.no_grad,
    )
