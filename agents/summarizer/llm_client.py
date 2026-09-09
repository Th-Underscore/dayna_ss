"""LLM generation wrappers for the Summarizer engine (T4 extraction).

SSE streaming, TGWUI completion, summary streaming, and the tool loop.
Methods live on LLMClientMixin and are mixed into Summarizer; all state is
accessed via self (duck-typed).
"""
from __future__ import annotations

import json
import jsonc
from pathlib import Path
import time
import traceback
from typing import Any, Generator

from ...runtime import runtime

from ...utils.helpers import (
    _ERROR,
    _SUCCESS,
    _GRAY,
    _HILITE,
    _BOLD,
    _RESET,
    _DEBUG,
    _WARNING,
    History,
    strip_thinking,
    strip_response,
)

if False:  # forward refs only
    from .core import Summarizer


class LLMClientMixin:
    def generate_using_tgwui(
        self,
        prompt: str,
        state: dict,
        history_path: Path | None = None,
        stopping_strings: list[str] | None = ["UNCHANGED", "unchanged", "NO_UPDATE", "no_update", '"UNCHANGED"', '"unchanged"', '"NO_UPDATE"', '"no_update"'],
        match_prefix_only: bool = True,
        **kwargs,
    ) -> tuple[str, str]:
        """Deprecated. Use generate_with_sse for real-time UI updates."""
        default_phase = kwargs.pop("phase_id", "legacy")
        default_step = kwargs.pop("step_id", "generate")
        return self.generate_with_sse(prompt, state, default_phase, default_step, history_path, stopping_strings, match_prefix_only, **kwargs)

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
            last = getattr(self, "last", None)
            history_path = last.history_path if last else None

        text = ""
        stop = ""
        if runtime.stop_everything:
            return "", ""

        internal_history = state.get("history", {}).get("internal", [])
        history_str = self.hash_key(internal_history, precision=24)
        try:
            with open(history_path.parent / "dump.txt", "a", encoding="utf-8") as f:
                dump_str = str(json.dumps(kwargs, indent=2))
                dump_str += f"\n\n========================== INTERNAL CONTEXT ({history_str})\n\n"
                dump_str += str(json.dumps(internal_history, indent=2))
                dump_str += "\n\n=========================="
                dump_str += "\n========================== NEW PROMPT\n\n"
                dump_str += str(prompt)
                dump_str += "\n\n==========================\n"
                f.write(dump_str)
                f.close()
        except Exception as e:
            print(f"{_ERROR}Error writing dump.txt: {str(e)}{_RESET}")
            traceback.print_exc()

        use_tool_loop = self.retrieval_mode == "active"
        last_emit_len = 0
        emit_threshold = 50  # Emit every ~50 chars to avoid flooding SSE
        last_emit_time = 0
        emit_interval = 0.3  # Also limit to ~3 updates/sec

        eval_start_time = None
        first_token_time = None

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
            eval_start_time = time.time()
            self._update_queue.publish({
                "type": "step_update",
                "phase": {"id": phase_id},
                "step": {"id": step_id, "message": "Generating response..."},
            })

            for t, s in gen:
                if first_token_time is None:
                    first_token_time = time.time()
                    eval_time_ms = (first_token_time - eval_start_time) * 1000
                    self._update_queue.publish({
                        "type": "step_update",
                        "phase": {"id": phase_id},
                        "step": {"id": step_id, "message": f"First token ({eval_time_ms:.0f}ms eval time)"},
                    })
                if runtime.stop_everything:
                    self._update_queue.publish({
                        "type": "step_done",
                        "phase": {"id": phase_id},
                        "step": {"id": step_id, "message": "Stopped", "status": "done"},
                    })
                    return text, stop
                text = t
                stop = s

                # Stream tokens to SSE (throttled)
                now = time.time()
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

                    try:
                        with open(history_path.parent / "dump.txt", "a", encoding="utf-8") as f:
                            f.write(snippet)
                            f.close()
                    except Exception as e:
                        print(f"{_ERROR}Error appending to dump.txt: {str(e)}{_RESET}")
                        traceback.print_exc()

                if s and s not in ("tool_call", ""):
                    break

        except Exception as e:
            error_msg = f"Error in generate_with_sse: {str(e)}"
            print(f"{_ERROR}{error_msg}{_RESET}")
            traceback.print_exc()
            self._update_queue.publish({
                "type": "step_update",
                "phase": {"id": phase_id},
                "step": {"id": step_id, "message": f"Error: {str(e)}"},
            })
            return "", "error"

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

        try:
            with open(history_path.parent / "dump.txt", "a", encoding="utf-8") as f:
                f.write("\n\n========================== OUTPUT\n\n")
                f.write(text.strip())
                f.write("\n\n==========================\n")
                f.close()
        except Exception as e:
            print(f"{_ERROR}Error writing closing marker to dump.txt: {str(e)}{_RESET}")
            traceback.print_exc()

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
            model = runtime.model
            if model is not None:
                with runtime.no_grad():
                    if runtime.stop_everything:
                        return
                    instr_prompt = runtime.generate_chat_prompt(prompt, state, **kwargs)
                    encoded_instr_prompt = (
                        runtime.encode(instr_prompt, add_bos_token=True) if model.__class__.__name__ not in ["LlamaServer", "LMDeployModel"] else instr_prompt
                    )
                    text = ""
                    token_count = 0
                    if runtime.stop_everything:
                        yield text, ""
                        return
                    for text in model.generate_with_streaming(encoded_instr_prompt, state):
                        token_count += 1
                        if runtime.stop_everything:
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
                    if runtime.stop_everything:
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
            if runtime.stop_everything:
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
