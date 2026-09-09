"""
Framework-agnostic runtime seam for the DSS core engine.

The core engine (``schema_parser``, ``entity_graph``, ``context_retriever``,
``data_summarizer``, ``summarizer``) must never import from a specific chat
frontend (``modules.*``, ``gradio``). All host-provided capabilities flow
through the module-level ``runtime`` singleton, which a host adapter
(e.g. ``runtime_tgwui``) configures at startup.

Unconfigured, every accessor returns a safe inert default so the core can be
imported and unit-tested standalone.
"""
from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, Callable, ContextManager


class _NullLogger:
    """No-op stand-in for the host's activity logger."""

    def log(self, event: str, details: str = "", level: str = "info") -> None:
        pass


class Runtime:
    """Host-provided capabilities consumed by the DSS core engine."""

    def __init__(self) -> None:
        self._model_provider: Callable[[], Any] | None = None
        self._stop_provider: Callable[[], bool] | None = None
        self._prompt_builder: Callable[..., str] | None = None
        self._encoder: Callable[..., Any] | None = None
        self._activity_logger: Any = _NullLogger()
        self._persistent_ui_state_provider: Callable[[], dict] | None = None
        self._current_character_provider: Callable[[], str] | None = None
        self._settings_provider: Callable[[], dict] | None = None
        self._update_config_fn: Callable[[dict], bool] | None = None
        self._register_tool_executors_fn: Callable[[dict], None] | None = None
        self._extension_dir: Path | None = None
        self._no_grad_cm: Callable[[], ContextManager] | None = None

    # --- model / generation ---

    @property
    def model(self) -> Any:
        """The current generation model object, or None."""
        if self._model_provider is not None:
            return self._model_provider()
        return None

    @property
    def stop_everything(self) -> bool:
        """Live "stop generation" flag from the host."""
        if self._stop_provider is not None:
            return bool(self._stop_provider())
        return False

    def generate_chat_prompt(self, prompt: str, state: dict, **kwargs) -> str:
        """Build a host-formatted chat prompt for `prompt` and `state`."""
        if self._prompt_builder is not None:
            return self._prompt_builder(prompt, state, **kwargs)
        return prompt

    def encode(self, text: str, add_bos_token: bool = False) -> Any:
        """Tokenize `text` through the host's tokenizer when one is required."""
        if self._encoder is not None:
            return self._encoder(text, add_bos_token=add_bos_token)
        return text

    def no_grad(self) -> ContextManager:
        """Return a host-provided inference context (e.g. ``torch.no_grad``).

        Unconfigured, falls back to a no-op context so the core runs standalone.
        """
        if self._no_grad_cm is not None:
            return self._no_grad_cm()
        return contextlib.nullcontext()

    # --- extension state ---

    @property
    def activity_logger(self) -> Any:
        """Logger exposing ``log(event, details, level)``."""
        return self._activity_logger

    @property
    def persistent_ui_state(self) -> dict:
        if self._persistent_ui_state_provider is not None:
            return self._persistent_ui_state_provider()
        return {}

    @property
    def current_character(self) -> str:
        if self._current_character_provider is not None:
            return self._current_character_provider()
        return ""

    @property
    def settings(self) -> dict:
        if self._settings_provider is not None:
            return self._settings_provider()
        return {}

    def update_config(self, state: dict) -> bool:
        """Sync host config state for the given chat state; returns True if changed."""
        if self._update_config_fn is not None:
            return self._update_config_fn(state)
        return False

    @property
    def extension_dir(self) -> Path:
        if self._extension_dir is not None:
            return self._extension_dir
        return Path(__file__).resolve().parent

    def register_tool_executors(self, executors: dict) -> None:
        """Register DSS tool executors with the host's tool system."""
        if self._register_tool_executors_fn is not None:
            self._register_tool_executors_fn(executors)

    # --- configuration ---

    def configure(
        self,
        *,
        model_provider: Callable[[], Any] | None = None,
        stop_provider: Callable[[], bool] | None = None,
        prompt_builder: Callable[..., str] | None = None,
        encoder: Callable[..., Any] | None = None,
        activity_logger: Any = None,
        persistent_ui_state_provider: Callable[[], dict] | None = None,
        current_character_provider: Callable[[], str] | None = None,
        settings_provider: Callable[[], dict] | None = None,
        update_config_fn: Callable[[dict], bool] | None = None,
        register_tool_executors_fn: Callable[[dict], None] | None = None,
        extension_dir: Path | None = None,
        no_grad_cm: Callable[[], ContextManager] | None = None,
    ) -> None:
        """Bind host providers. Unconfigured entries keep their defaults."""
        if model_provider is not None:
            self._model_provider = model_provider
        if stop_provider is not None:
            self._stop_provider = stop_provider
        if prompt_builder is not None:
            self._prompt_builder = prompt_builder
        if encoder is not None:
            self._encoder = encoder
        if activity_logger is not None:
            self._activity_logger = activity_logger
        if persistent_ui_state_provider is not None:
            self._persistent_ui_state_provider = persistent_ui_state_provider
        if current_character_provider is not None:
            self._current_character_provider = current_character_provider
        if settings_provider is not None:
            self._settings_provider = settings_provider
        if update_config_fn is not None:
            self._update_config_fn = update_config_fn
        if register_tool_executors_fn is not None:
            self._register_tool_executors_fn = register_tool_executors_fn
        if extension_dir is not None:
            self._extension_dir = extension_dir
        if no_grad_cm is not None:
            self._no_grad_cm = no_grad_cm


runtime = Runtime()
