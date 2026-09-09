"""Instruction-prompt generation and post-generation cleaning (T4 extraction).

generate_instr_prompt builds the per-turn instruction block using the
instruction-block engine in ../instruction_blocks.py; _clean_generated_instructions
post-checks the generated instructions against the latest exchange.
Methods live on InstructionGeneratorMixin mixed into Summarizer.
"""
from __future__ import annotations

import copy
import difflib
import json
import random
import re
import traceback
from datetime import datetime
from pathlib import Path

from ...rag.structured_rag.context_retriever import RetrievalContext
from ...runtime import runtime

from ...utils.helpers import (
    _ERROR,
    _SUCCESS,
    _INPUT,
    _GRAY,
    _HILITE,
    _BOLD,
    _RESET,
    _DEBUG,
    _WARNING,
    History,
    load_json,
    save_json,
    strip_thinking,
    strip_response,
)

from ..instruction_blocks import (
    _SENT_SPLIT_RE,
    _INSTR_REGEN_MAX,
    _INSTR_RING_MAXLEN,
    _INSTR_ANTI_REPEAT_DIRECTIVE,
    _block_collides,
    _ring_blocks,
    _extract_props,
    _recently_used_block,
    _collect_known_names,
    _own_replies_block,
    _current_scene_recap,
)

if False:  # forward refs only
    from .core import Summarizer


class InstructionGeneratorMixin:
    def _append_instr_block(self, block: str, known_names: list[str]) -> None:
        """Record an accepted instruction block (fresh or cache-hit) into the
        recently-used ring for P1/P2. The label tracks order only.
        """
        if not block:
            return
        seq = self._instr_seq
        self._instr_seq += 1
        self._recent_instr_blocks.append((f"recent {seq}", _extract_props(block, known_names), block))

    def generate_instr_prompt(
        self, user_input: str, state: dict, history: History, **kwargs
    ) -> tuple[str, dict, Path, str]:  # After input
        """
        Builds the instruction prompt used to steer the model's character response and returns that prompt plus a snapshot of state, the history path, and a scene timestamp.

        The function prepares retrieval context, optionally generates or loads a cached set of plain-text instruction paragraphs (when `do_instr` is true), composes a final prompt for the character, and encodes it when required by the model. It also records a deep-copied custom state and derives a current-scene timestamp to be used for subsequent summarization and chunking.

        Parameters:
            user_input (str): The latest user message to be incorporated into the instruction prompt.
            state (dict): The current session state/configuration (may be mutated internally for seed handling).
            history (History): Conversation history used to compute message indices and the history path.
            **kwargs: Optional flags and generation options. Recognized keys include:
                do_instr (bool): If true, generate detailed instruction paragraphs and persist them to instructions.json.

        Returns:
            tuple[str, dict, Path, str]:
                instr_prompt — The instruction prompt ready for model consumption; encoded string when required by the backend, or the original `user_input` on early stop/failure.
                custom_state — A deep-copied snapshot of the state used for generating the instruction prompt.
                history_path — Path to the session's history directory containing cached artifacts.
                current_timestamp_str — ISO-8601 timestamp string for the current scene (derived from retrieval context when available); `None` when generation was stopped or failed.

        Side effects:
            - May persist generated instruction text to <history_path>/instructions.json.
            - Writes a diagnostic dump file (dump.txt) next to the history directory.
            - Logs activity and updates phase/step tracking for UI telemetry.
        """
        print(f"{_HILITE}generate_instr_prompt{_RESET} {kwargs}")
        self.log_activity("Generating Instructions", "Preparing context", "info")

        pm = self._phase_manager
        pm.start_turn("Instruction Generation")
        if "instr_prompt" not in pm._phase_lookup:
            phase = {"id": "instr_prompt", "name": "Instruction Generation", "weight": 1}
            pm._phases.append(phase)
            pm._phase_lookup["instr_prompt"] = phase
            pm._phase_steps["instr_prompt"] = []
            pm._total_weight += 1

        pm.start_phase("instr_prompt", "Instruction Generation")

        try:
            pm.start_step("instr_prompt", "context", "Preparing context...")
            user_input, custom_state_ref = self.prepare_context(user_input, state, history, **kwargs)
            pm.done_step("instr_prompt", "context", "Context prepared")
            custom_state = copy.deepcopy(custom_state_ref)
            last = getattr(self, "last", None)
            history_path = last.history_path if last else None

            if runtime.stop_everything:
                print(f"{_HILITE}Stop signal received after prepare_context in generate_instr_prompt.{_RESET}")
                pm.done_phase("instr_prompt", "Stopped")
                return user_input, state, history_path, None

            current_timestamp_str = datetime.now().isoformat()
            retrieval_ctx = None
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
                try:
                    user_input_prompt = f'This is the latest user input:\n\n"""\n{user_input}\n"""\n\n---'
                    name1 = state["name1"] or "User"
                    name2 = state["name2"] or "Assistant"

                    # P1: surface the recently-used imagery to the generator so
                    # it can satisfy req 9 + req 14 (it has no other way to know
                    # which stored objects prior replies already used).
                    known_names = _collect_known_names(retrieval_ctx)
                    recent_props_block = _recently_used_block(list(self._recent_instr_blocks))
                    # R1: DSS's own last replies as negative exemplars for the
                    # reply prompt (self-anchoring is a transcription driver).
                    # Items carry their ABSOLUTE assistant-message index
                    # (2*pair_position + 1 over the full internal history,
                    # greeting pair included) so the numbering is stable
                    # regardless of how many replies are quoted.
                    own_replies: list[tuple[int, str]] = []
                    try:
                        internal = (state.get("history") or {}).get("internal") or []
                        total_pairs = len(internal)
                        for offset, pair in enumerate(reversed(internal)):
                            if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                                continue
                            reply = pair[1] if isinstance(pair[1], str) else ""
                            if len(reply.strip()) >= 80:
                                pair_pos = total_pairs - 1 - offset
                                own_replies.append((2 * pair_pos + 1, reply.strip()))
                                if len(own_replies) >= 3:
                                    break
                        own_replies.reverse()
                    except Exception:
                        own_replies = []
                    own_replies_block = _own_replies_block(own_replies)

                    instr_path = history_path / "instructions.json"
                    instructions: dict[str, str] = load_json(instr_path) or {}

                    model = runtime.model

                    original_seed = state["seed"]
                    if original_seed == -1:
                        state["seed"] = random.randint(1, 2**31)
                        print(f"{_BOLD}New seed{_RESET}: {state['seed']}")
                    self.last.original_seed = original_seed
                    input_key = str(state["seed"])

                    print(f"{_HILITE}input_key{_RESET}: {input_key}")
                    instr = ""
                    prompt = ""
                    if input_key in instructions:
                        instr: str = instructions[input_key]
                        print(f"{_SUCCESS}Found cached instruction prompt{_RESET}")
                        self._prev_instruction = instr
                        self._append_instr_block(instr, known_names)
                        pm.start_step("instr_prompt", "cache_hit", "Loaded cached instructions")
                        pm.done_step("instr_prompt", "cache_hit", f"Loaded {len(instr)} chars from cache")
                    else:
                        try:
                            if model is not None:
                                print(f"{_SUCCESS}State set{_RESET}")

                                if kwargs.get("do_instr", False):
                                    # Ground the plan in the LIVE scene (authoritative,
                                    # current) rather than the stable top of the subject
                                    # block — the source of the prop-recycling loop.
                                    scene_recap = ""
                                    if self.last and self.last.context and self.last.context[0]:
                                        scene_recap = _current_scene_recap(
                                            getattr(self.last.context[0], "current_scene", None)
                                        )
                                    scene_recap_block = (
                                        f"Current scene (authoritative \u2014 plan the response around it):\n{scene_recap}\n"
                                        if scene_recap
                                        else ""
                                    )
                                    prompt = (
                                        f"{scene_recap_block}{recent_props_block}{user_input_prompt}\n\n"
                                        f"You are to generate instructions for {name2}'s response to '{name1}'. These instructions will be given directly to {name2}.\n"
                                        f"The instructions must guide {name2} on what to say or do, in {name2}'s own voice and writing style as defined by the writing-style directive (general_info.writing_style in the system context), and should be detailed and specific. Do NOT instruct {name2} to imitate {name1}'s voice, register, or turns of phrase, and do not instruct {name2} to quote or re-use {name1}'s lines.\n\n"
                                        f"FORMATTING REQUIREMENTS:\n"
                                        f"- Present the instructions as a series of plain text paragraphs.\n"
                                        f"- Each paragraph should represent a distinct part of the response plan.\n"
                                        f"Example of desired output structure (imagine these are the instructions):\n"
                                        f"  First, analyze {name1}'s query to understand their core need. Then, formulate a concise opening statement that acknowledges their input.\n"
                                        f"  Next, provide the main information or answer, breaking it down into logical points if necessary. Ensure clarity and accuracy in this section.\n"
                                        f"Remember: The above is an example. In a narrative context, explicitly acknowledging {name1}'s input would break immersion. Additionally, the length of the response should match the established writing style.\n\n"
                                        f"INSTRUCTION CONTENT:\n"
                                        f"1. Explain in detail, step-by-step, in imperative mood, what {name2} should include in their response.\n"
                                        f"2. Be specific, detailing each step.\n"
                                        f"3. You are providing instructions FOR the response, not writing the response itself.\n"
                                        f"4. Address the instructions directly to {name2} (e.g., 'Start by...', 'Then, explain...'). Do not refer to {name2} in the third person (e.g., '{name2} should...').\n"
                                         f"5. Specify the desired length of {name2}'s actual final response by referencing the length guidance in the Writing Style directive (general_info.writing_style) \u2014 e.g., 'Match the directive's paragraph count'. Do not write self-referential 'The final response should be...' phrasings.\n"
                                         f"6. Instruct on the use of dialogue: specify when it is appropriate for characters to speak, which characters should speak, and when narration should be used instead of dialogue.\n"
                                         f"7. IMPORTANT: Remind {name2} not to recap {name1}'s input in the response. Even if necessary to clarify {name1}'s intent, remember: Show, don't tell.\n"
                                         f"8. CRITICAL: Explicitly include an additional instruction on the \"Writing Style\" of the response, taken from the Writing Style directive (general_info.writing_style) in the system context \u2014 the authoritative source for {name2}'s voice. The style MUST stay {name2}'s own voice; it must NOT drift toward {name1}'s voice, and {name2} must not narrate from {name1}'s perspective.\n"
                                         f"9. Instruct {name2} to weave in specific characters, items, and past events already established in the system context (General Info, Current Scene, Characters, Events sections) where they fit naturally, so the reply reflects what {name2} remembers about the story rather than leaving remembered details unused.\n"
                                         f"10. Instruct {name2} to advance the scene with concrete action, dialogue, or a specific new detail \u2014 the reply must not be purely atmospheric or observational. Prohibit premature conclusions: {name2} must not declare the mystery resolved, name a culprit, or treat a suspicion as fact until the story has actually revealed it. Prohibit repeating imagery already used in earlier replies. Never instruct {name2} to end the reply by passing the turn back to {name1} \u2014 no 'leaving the next move in {name1}'s hands', no 'step back to let {name1}'. The final beat must be an action {name2} performs herself.\n"
                                         f"11. ROLE BINDING: {name2} is the AI character being roleplayed \u2014 the story's protagonist, who writes this reply. {name1} is the other principal character (the user's character), whose turn just ended. Never swap these roles, and never assign {name1}'s identity, occupation, or biography to {name2}.\n"
                                         f"12. Phrase every instruction as a concrete visible action in imperative mood addressed to {name2} (e.g. 'Open the minute book', 'Ask Mrs. Arbuthnot about the map', 'Examine the tin on the shelf'). Never include meta-instructions about how the reply itself should be constructed \u2014 no 'ensure your response', 'your response should', 'focusing on', 'as a prop to emphasize', 'end on a specific action', 'rather than a summary', or 'advance the scene with concrete action by'.\n"
                                         f"13. Keep the instruction block SHORT: at most three concrete beats. Do not pile up more than three instructions in a single block.\n"
                                         f"14. STRICT IMAGERY RULE: Never instruct {name2} to reuse imagery, gestures, props, or beats already used in earlier replies, and never re-anchor on the same stored objects from the system-context memory \u2014 the RECENTLY USED IMAGERY block above exhaustively lists the props the last {len(self._recent_instr_blocks)} blocks anchored on, and you must pick the response's concrete objects, locations, and actions from stored items NOT on that list (or introduce a new detail). UNLESS returning to one is a deliberate, pointed stylistic callback (poignant or comedic) \u2014 then use it at most once and vary the action around it. Routine repetition is forbidden: every reply must introduce at least one fresh concrete detail this scene has not yet used, and must not open with a gesture already performed in an earlier reply.\n\n"
                                        f"REMEMBER: Your entire output must ONLY consist of the instructional paragraphs, adhering strictly to the no-bolding, no-titles format. No extra text, greetings, or sign-offs."
                                    )

                                    instr, status = self.generate_with_sse(
                                        prompt=prompt,
                                        state=custom_state,
                                        phase_id="instr_prompt",
                                        step_id="generate_instructions",
                                        history_path=history_path,
                                        match_prefix_only=False,
                                    )
                                    if runtime.stop_everything:
                                        print(f"{_HILITE}Stop signal received after instruction generation.{_RESET}")
                                        pm.done_phase("instr_prompt", "Stopped")
                                        return user_input, state, history_path, None

                                    if status == "error" or not instr:
                                        print(f"{_ERROR}Instruction generation failed or returned empty result{_RESET}")
                                        pm.error_phase("instr_prompt", "Failed to generate instructions")
                                        return user_input, state, history_path, None

                                    cleaned_instr = self._clean_generated_instructions(instr, user_input)
                                    if cleaned_instr != instr:
                                        print(f"{_WARNING}Instruction post-check: removed degenerate repetition/echo ({len(instr) - len(cleaned_instr)} chars){_RESET}")
                                    # Cross-turn repetition guard: full-ring scan.
                                    # The repetition loop re-draws props/blocks on a
                                    # 5-6 turn cadence that sits positions 3-8 of the
                                    # ring, so the guard checks EVERY recent block
                                    # (byte-ratio, LCS, and a 2-prop re-anchor cluster)
                                    # and names the actual offending props in the
                                    # regeneration directive (the t28 lesson).
                                    prev_instr = getattr(self, "_prev_instruction", None)
                                    full_ring = _ring_blocks(self._recent_instr_blocks, prev_instr)
                                    collides, offending_props = _block_collides(
                                        cleaned_instr, full_ring, known_names
                                    )
                                    if collides:
                                        print(f"{_WARNING}Instruction block repeats imagery/props from the recent ring; regenerating...{_RESET}")
                                        directive = _INSTR_ANTI_REPEAT_DIRECTIVE
                                        if offending_props:
                                            directive += (
                                                "\nSpecifically, do NOT re-anchor on these props that the last instruction blocks already used: "
                                                + ", ".join(sorted(offending_props)[:8])
                                                + ". Choose different concrete objects, locations, or actions \u2014 or introduce a new detail."
                                            )
                                        # Hard-reject acceptance: try up to GRD max
                                        # attempts, keep the LEAST-colliding candidate
                                        # (a name-less single retry cannot de-loop).
                                        best = (cleaned_instr, offending_props)
                                        best_score = len(offending_props)
                                        accepted = False
                                        for _attempt in range(_INSTR_REGEN_MAX):
                                            instr_n, status_n = self.generate_with_sse(
                                                prompt=prompt + directive,
                                                state=custom_state,
                                                phase_id="instr_prompt",
                                                step_id="generate_instructions",
                                                history_path=history_path,
                                                match_prefix_only=False,
                                            )
                                            if status_n == "error" or not instr_n:
                                                break
                                            cleaned_n = self._clean_generated_instructions(instr_n, user_input)
                                            n_collides, n_off = _block_collides(cleaned_n, full_ring, known_names)
                                            if not n_collides and not n_off:
                                                cleaned_instr = cleaned_n
                                                accepted = True
                                                break
                                            if len(n_off) < best_score:
                                                best = (cleaned_n, n_off)
                                                best_score = len(n_off)
                                        if not accepted:
                                            print(f"{_WARNING}Regenerated blocks still collide; keeping the least-colliding attempt.{_RESET}")
                                            if best_score < len(offending_props):
                                                cleaned_instr = best[0]
                                        else:
                                            print(f"{_WARNING}Regenerated instruction block accepted (clean vs the full ring).{_RESET}")
                                    # The cleaned block is what the reply prompt actually uses; the
                                    # previous code cached the cleaned text but fed the RAW block to
                                    # the model on the generating turn, making both the cleaner and
                                    # this guard dead for the current prompt.
                                    instr = cleaned_instr
                                    instructions[input_key] = instr
                                    self._prev_instruction = instr
                                    self._append_instr_block(instr, known_names)
                                    print(f"{_HILITE}Instruction:{_RESET} {instr}")
                                    save_json(instructions, instr_path)

                        except Exception as e:
                            print(f"{_ERROR}Error generating instruction: {str(e)}{_RESET}")
                            traceback.print_exc()
                            pm.error_phase("instr_prompt", str(e))
                            return user_input, state, history_path, None

                    instr_prompt = ""
                    writing_style = ""
                    try:
                        gi = (retrieval_ctx.general_info or {}) if retrieval_ctx else {}
                        writing_style = (gi.get("writing_style") or "").strip()
                    except Exception:
                        writing_style = ""
                    if not writing_style:
                        writing_style = "see the writing-style directive (general_info.writing_style) in your system context"
                    voice_frame = (
                        f'\n[VOICE: The message above was written by {name1}, NOT by you. Do not continue it, do not '
                        f'adopt its voice or perspective, and do not quote or re-use its words. You are {name2}. '
                        f'Write a fresh reply strictly in {name2}\'s own voice per the Writing Style directive below.]\n'
                    )
                    if kwargs.get("do_instr", False):
                        full_instr = (
                            instr
                            + "\n\n[STEERING NOTES: These directives are private off-stage guidance for you as the AUTHOR of "
                              f"{name2}'s reply — never material to print. Transform each beat into concrete scene action in {name2}'s own "
                              "voice; never quote or reuse any distinctive phrase, wording, or image from this block verbatim in the reply itself.]"
                        )
                        instr_prompt = (
                            f"{user_input_prompt}\n\n"
                            f"{voice_frame}\n"
                            f'You are to write a reply in character as "{name2}".\n'
                            f"The following instructions, presented as plain text paragraphs, outline how you should construct your response:\n\n"
                            f'INSTRUCTIONS TO FOLLOW:\n"""\n{full_instr}\n"""\n\n'
                            f"Adhere loosely to these instructions.\n"
                            f"WRITING STYLE DIRECTIVE (you MUST follow it):\n{writing_style}\n"
                            f"Write strictly in {name2}'s own voice as defined by the directive above; do NOT imitate {name1}'s voice, register, or turns of phrase, and never quote or re-use {name1}'s lines.\n"
                            f"Your reply must be natural-sounding prose.\n"
                            f"Draw naturally on the characters, items, and past events established in your system context (General Info, Current Scene, Characters, Events sections); reference them concretely where they fit rather than leaving remembered details unused.\n"
                             f"Do not let the reply be purely atmospheric or observational \u2014 include concrete action, dialogue, or a specific new detail. Do not declare the mystery resolved, name a culprit, or treat a suspicion as fact until the story has revealed it. Do not repeat imagery or a closing sentence/ending gesture you have already used in earlier replies.\n"
                             f"End the reply with {name2} performing a concrete action of her own \u2014 never end with a sentence that hands the next move to {name1} ('leaving the next move', 'step back to let'). Never narrate the reply's own construction (no 'she advanced the scene', 'her response remained', 'she avoided declaring', 'as a prop to emphasize') \u2014 perform the action directly.{own_replies_block}\n\n"
                            f'REMEMBER: You are "{name2}" replying to "{name1}". Narrate {name2} from outside — and do not repeat or re-speak {name1}\'s lines.'
                        )
                    else:
                        pm.start_step("instr_prompt", "skip", "Instruction generation disabled")
                        pm.done_step("instr_prompt", "skip", "Skipped")
                        instr_prompt = (
                            f"{user_input_prompt}\n\n"
                            f"{voice_frame}\n"
                            f'You are to write a reply in character as "{name2}".\n'
                            f"WRITING STYLE DIRECTIVE (you MUST follow it):\n{writing_style}\n"
                            f"Write strictly in {name2}'s own voice as defined by the directive above; do NOT imitate {name1}'s voice, register, or turns of phrase, and never quote or re-use {name1}'s lines.\n"
                            f"Your reply must be natural-sounding prose.\n"
                            f"Draw naturally on the characters, items, and past events established in your system context (General Info, Current Scene, Characters, Events sections); reference them concretely where they fit rather than leaving remembered details unused.\n"
                             f"Do not let the reply be purely atmospheric or observational \u2014 include concrete action, dialogue, or a specific new detail. Do not declare the mystery resolved, name a culprit, or treat a suspicion as fact until the story has revealed it. Do not repeat imagery or a closing sentence/ending gesture you have already used in earlier replies.\n"
                             f"End the reply with {name2} performing a concrete action of her own \u2014 never end with a sentence that hands the next move to {name1} ('leaving the next move', 'step back to let'). Never narrate the reply's own construction (no 'she advanced the scene', 'her response remained', 'she avoided declaring', 'as a prop to emphasize') \u2014 perform the action directly.{own_replies_block}\n\n"
                            f'REMEMBER: You are "{name2}" replying to "{name1}". Narrate {name2} in the THIRD PERSON from outside — never from {name2}\'s own "I" (no "I", "my", "me") — and do not repeat or re-speak {name1}\'s lines.'
                        )
                    encoded_instr_prompt = (
                        runtime.encode(instr_prompt, add_bos_token=True) if do_enc(model) else instr_prompt
                    )
                    print(
                        f"{_SUCCESS}Encoded instruct prompt: {True if do_enc(model) else False}{_RESET}"
                    )

                    print(f"{_SUCCESS}State set{_RESET}")

                    try:
                        with open(history_path.parent / "dump.txt", "w", encoding="utf-8") as f:
                            dump_str = str(json.dumps(kwargs, indent=2))
                            dump_str += "\n\n========================== INSTRUCTION GENERATION PROMPT\n\n"
                            dump_str += str(prompt)
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
                    self.log_activity("Instructions Ready", f"Path: {history_path.name}", "success")
                    return (
                        encoded_instr_prompt,
                        custom_state,
                        history_path,
                        current_timestamp_str,
                    )
                except Exception as e:
                    print(f"{_ERROR}Error in get_summary_state: {str(e)}{_RESET}")
                    self.log_activity("Instruction Gen Failed", str(e), "error")
                    traceback.print_exc()
                    pm.error_phase("instr_prompt", str(e))
                    return user_input, state, history_path, None

            print(f"{_ERROR}generate_instr_prompt reached its end without a result (no history path).{_RESET}")
            pm.error_phase("instr_prompt", "No history path")
            return user_input, state, history_path, None
        finally:
            if pm.active_phase == "instr_prompt":
                pm.done_phase("instr_prompt", "Completed")
                pm.end_turn()


    def _clean_generated_instructions(self, instr: str, user_input: str) -> str:
        """Post-check the LLM-generated instruction block (deterministic, no extra inference).

        Three deterministic passes:
        1. Drop whole sentences that describe the REPLY's own construction
           ("Ensure your response remains...", "Your response should end on...") —
           the small model transcribes these verbatim as narration (the t27
           meta-commentary collapse in cozy_mystery__dded7733).
        2. Strip construction/deferral clauses from within kept sentences
           ("leaving the next move in Prudence's hands", "as a prop to
           emphasize", "end on ... rather than a summary") while keeping the
           sentence's actionable content.
        3. Remove near-duplicate sentences (degenerate repetition / template
           lock-in) and sentences that near-verbatim echo the user input.
        Returns the cleaned text, or the original if nothing clearly degenerate.
        """
        if not instr or not isinstance(instr, str) or not user_input:
            return instr
        sentences = [s.strip() for s in re.split(_SENT_SPLIT_RE, instr) if s.strip()]
        if len(sentences) < 3:
            return instr

        def norm(s: str) -> str:
            return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()

        def ratio(a: str, b: str) -> float:
            na, nb = norm(a), norm(b)
            if not na or not nb:
                return 0.0
            return difflib.SequenceMatcher(None, na, nb).ratio()

        meta_sentence_pats = [
            r"ensure your response",
            r"ensure that you narrate",
            r"ensure your voice remains",
            r"your response remains",
            r"your reply remains",
            r"your response should end",
            r"your final response should",
            r"the final response should",
            r"keep your response to",
            r"keep the response to",
            r"end on a specific action or observation rather than a summary",
            r"rather than narrating",
        ]
        # Clauses stripped from within kept sentences (the actionable part survives).
        meta_clause_pats = [
            r",?\s*leaving the next move in \w+['\u2019]?s hands\b",
            r",?\s*step(?:ping)? back to let \w+[^,.!?]*",
            r",?\s*hand(?:ing)? the next move to \w+[^,.!?]*",
            r",?\s*(?:and\s+)?end(?:ing)? on a concrete action or observation rather than a summary\b",
            r",?\s*rather than a summary\b",
            r",?\s*as a prop to emphasize[^,.!?]*",
            r",?\s*focusing on \w+['\u2019]?s observations and movements\b",
            r",?\s*ensure that your dialogue remains[^.!?]*",
            r"\bAvoid declaring the mystery resolved or naming the culprit definitively;\s*(?:instead,?\s*)?",
            r"\bDo not declare the mystery resolved or name [^;]{0,60}culprit definitively;\s*(?:instead,?\s*)?",
            r"\bAdvance the scene with concrete action by\s+",
        ]
        clause_re = [(re.compile(p, re.IGNORECASE)) for p in meta_clause_pats]

        user_sents = [s.strip() for s in re.split(_SENT_SPLIT_RE, user_input) if s.strip()]
        kept: list[str] = []
        dropped = 0
        for s in sentences:
            if any(re.search(p, s, re.IGNORECASE) for p in meta_sentence_pats):
                dropped += 1
                continue
            for cr in clause_re:
                s = cr.sub("", s).strip()
                if not s:
                    dropped += 1
                    break
            if not s:
                continue
            if any(ratio(s, k) > 0.9 for k in kept):
                dropped += 1
                continue
            if any(ratio(s, us) > 0.85 for us in user_sents if len(norm(us)) > 20):
                dropped += 1
                continue
            kept.append(s)
        if dropped == 0:
            return instr
        out = " ".join(kept)
        return out if out.strip() else instr



def do_enc(model):
    return model.__class__.__name__ not in ["LlamaServer", "LMDeployModel"]
