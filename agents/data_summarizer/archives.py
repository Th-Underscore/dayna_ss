"""Chapter and arc boundary checks/archiving (T4 extraction). check_and_archive_chapter and check_and_archive_arc.
"""
from __future__ import annotations

import copy
import json
import jsonc
import re
import time
import traceback
from typing import Any

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
    load_json,
    save_json,
    split_keys_to_list,
    recursive_set,
    recursive_get,
    expand_lists_in_data_for_llm,
    unexpand_lists_in_data_from_llm,
    coerce_container_types,
    strip_thinking,
    strip_response,
    format_str,
    render_jinja_template,
    format_str_or_jinja,
)

from ...runtime import runtime

from ...utils.schema_parser import (
    SchemaParser,
    ParsedSchemaClass,
    ParsedSchemaField,
    Action,
    Trigger,
)

from ..formatted_data import FormattedData

from .parsing import (
    _entries_as_list,
    _safe_int,
    _normalize_entry_name,
    _collect_entry_aliases,
    _resolve_dict_key,
    _resolve_path_keys,
    _is_negative_verdict,
    _tolerant_json_loads,
    _is_schema_echo,
    _filter_new_entry_names,
    _extract_entry_names,
)

if False:  # forward refs only
    from .core import DataSummarizer


# Unit-cadence profiles (decisions ledger #3/#37): calibration overlays applied
# over whatever the session's schema declares. Unit length is goal-relative —
# scene/chapter/arc sizing depends on the campaign's intended scope — so the
# numbers are declared config, not truth. "compressed" restates the shipped
# schema defaults (exercises the machinery in 20-40-turn tests); "campaign" is
# true-scale cadence for long/chained deployments. Pure plumbing: same engine
# and templates, only the gate guardrails move.
UNIT_CADENCE_PROFILES: dict[str, dict[str, dict[str, int]]] = {
    "compressed": {
        "chapters": {
            "suggested_min_scenes": 4,
            "suggested_max_scenes": 8,
            "max_scenes_before_required": 10,
        },
        "arcs": {
            "suggested_min_chapters": 2,
            "suggested_max_chapters": 5,
            "max_chapters_before_required": 8,
        },
    },
    "campaign": {
        "chapters": {
            "suggested_min_scenes": 15,
            "suggested_max_scenes": 40,
            "max_scenes_before_required": 50,
        },
        "arcs": {
            "suggested_min_chapters": 8,
            "suggested_max_chapters": 24,
            "max_chapters_before_required": 30,
        },
    },
}


def _resolve_cadence(
    schema_defaults: dict | None,
    unit: str,
    child_unit: str,
    fallback: tuple[int, int, int],
    profile_name: str | None,
) -> tuple[int, int, int]:
    """Resolve (suggested_min, suggested_max, max_hard) for an aggregation unit.

    Precedence: named-profile overlay > schema defaults > code fallback. The
    gate templates render these values as guardrails, so a profile change
    flows into prompts without any template duplication.
    """
    overlay = UNIT_CADENCE_PROFILES.get(profile_name or "", {}).get(unit, {})
    defaults = schema_defaults or {}
    picked = []
    for key, fb in (
        (f"suggested_min_{child_unit}", fallback[0]),
        (f"suggested_max_{child_unit}", fallback[1]),
        (f"max_{child_unit}_before_required", fallback[2]),
    ):
        try:
            picked.append(int(overlay.get(key, defaults.get(key, fb))))
        except (TypeError, ValueError):
            picked.append(fb)
    return tuple(picked)  # type: ignore[return-value]


# Canon Synopsis P0 (docs/plans/canon_synopsis_p0.md): the digest prompt keeps
# STATIC directive content first and renders the compact inputs last (#131
# prefix-ordering principle). Objective plot-level prose only — stances are
# stated as facts-of-the-record; private interiority stays out (epistemics
# composition, memory #212).
_CANON_DIGEST_PROMPT = """You are maintaining the running canon of an interactive story.
Write a narrative digest of the story so far, in third person, as flowing prose (not a list).

Rules:
- At most 300 tokens (~1200 characters). Cover the whole span, weighting recent events most.
- State what happened, who drove it, and what remains open or unresolved.
- Plot-level and objective: character beliefs/suspicions may be stated only as facts the scenes established.
- Do not enumerate scenes or repeat raw dialogue; the scene archive already does that.
- This digest replaces the previous one entirely; do not refer to "previously".

Story anchors:
Main objective: {main_objective}

Archived arcs:
{arcs_block}

Archived chapters:
{chapters_block}

Respond with ONLY the digest prose."""


class DSArchivesMixin:

    # Canon Synopsis P0: bounded cap for the regenerated digest. The model is
    # asked for ~300 tokens; the hard ceiling guards against runaway responses.
    _CANON_DIGEST_MAX_CHARS = 3000

    def _regenerate_canon_digest(self, trigger: str):
        """Regenerate the story-so-far digest into general_info.synopsis.

        Called after a successful chapter/arc archival. ONE call regenerates
        the WHOLE digest from compact inputs (never append-to-digest: drift
        would accumulate across rewrites). Fail-open — on empty/garbage
        response or any exception the previous digest stays in place; the
        digest is a cache of its inputs, never load-bearing state.

        Persists general_info.json explicitly (per-subject files were saved
        before boundary checks ran) and appends an entry to canon_history.json
        beside it for forensics.
        """
        try:
            events_data = self.all_subjects_data.get("events", {})
            general_info = self.all_subjects_data.get("general_info")
            if not isinstance(general_info, dict):
                return

            chapters = [c for c in _entries_as_list(events_data.get("chapters")) if isinstance(c, dict)]
            # Arcs are their own subject persisted to arcs.json (dict keyed by
            # title); read them from disk so an arc archived earlier in this
            # same boundary pass is already visible.
            arcs_source = load_json(self.history_path / "arcs.json")
            if not isinstance(arcs_source, (dict, list)) or not arcs_source:
                arcs_source = self.all_subjects_data.get("arcs", {})
            arcs = [a for a in _entries_as_list(arcs_source) if isinstance(a, dict)]
            if not chapters and not arcs:
                return  # nothing archived yet; leave the field alone

            chapters_block = "\n".join(
                f"- {c.get('title', '?')} [{c.get('starting_scene', '?')}-{c.get('ending_scene', '?')}]: "
                f"{str(c.get('summary', ''))[:400]}"
                + (f" Key changes: {'; '.join(str(k.get('description', ''))[:100] for k in c.get('key_changes', [])[:5])}"
                   if c.get("key_changes") else "")
                for c in chapters
            ) or "- (none yet)"
            arcs_block = "\n".join(
                f"- {a.get('title', '?')}: {str(a.get('summary', ''))[:300]}"
                for a in arcs
            ) or "- (none yet)"

            prompt = _CANON_DIGEST_PROMPT.format(
                main_objective=str(general_info.get("main_objective", ""))[:200] or "(not set)",
                arcs_block=arcs_block,
                chapters_block=chapters_block,
            )

            current_custom_state = self.custom_state or {}
            print(f"{_INPUT}Regenerating canon synopsis ({trigger})...{_RESET}")
            digest_response, _ = self.summarizer.generate_using_tgwui(
                prompt=prompt,
                state=current_custom_state,
                history_path=self.history_path,
                phase_id='canon',
                step_id='digest',
            )
            if runtime.stop_everything:
                return

            digest = strip_response(digest_response or "").strip()
            # Drop wrapping quotes some models add around prose.
            digest = digest.strip('"').strip()
            if len(digest) < 40:
                print(f"{_WARNING}Canon synopsis regeneration produced nothing usable; keeping previous digest.{_RESET}")
                return
            if len(digest) > self._CANON_DIGEST_MAX_CHARS:
                cut = digest.rfind(".", 0, self._CANON_DIGEST_MAX_CHARS)
                digest = digest[: cut + 1 if cut > 0 else self._CANON_DIGEST_MAX_CHARS]

            previous = str(general_info.get("synopsis", "") or "")
            general_info["synopsis"] = digest
            save_json(general_info, self.history_path / "general_info.json")

            current_scene_data = self.all_subjects_data.get("current_scene", {})
            history_entry = {
                "trigger": trigger,
                "scene": current_scene_data.get("_scene_number"),
                "chapter": current_scene_data.get("_chapter_number"),
                "arc": current_scene_data.get("_arc_number"),
                "chars": len(digest),
                "replaced_chars": len(previous),
                "digest": digest,
            }
            history_path = self.history_path / "canon_history.json"
            canon_history = []
            if history_path.exists():
                try:
                    loaded = json.loads(history_path.read_text(encoding="utf-8"))
                    if isinstance(loaded, list):
                        canon_history = loaded
                except Exception:
                    canon_history = []
            canon_history.append(history_entry)
            save_json(canon_history, history_path)
            print(f"{_SUCCESS}Canon synopsis updated ({len(previous)} -> {len(digest)} chars).{_RESET}")
        except Exception as e:
            print(f"{_ERROR}Canon synopsis regeneration failed ({type(e).__name__}: {e}); "
                  f"keeping previous digest.{_RESET}")

    def check_and_archive_chapter(self):
        """Check if the current chapter should be archived and create a new one if needed.

        This is called after scene archival to determine if a chapter boundary
        should be crossed based on LLM-driven decision making.

        Can be forced via summarizer.last.force_next_chapter = True
        """
        if runtime.stop_everything:
            return

        events_data = self.all_subjects_data.get("events", {})
        if not events_data:
            print(f"{_GRAY}No events data available. Skipping chapter check.{_RESET}")
            return

        current_custom_state = self.custom_state or {}

        # Check for forced chapter transition
        force_chapter = self.summarizer.last and self.summarizer.last.force_next_chapter
        if force_chapter:
            print(f"{_BOLD}Force next chapter requested. Skipping LLM check.{_RESET}")

        # Get current chapter info
        current_scene_data = self.all_subjects_data.get("current_scene", {})
        current_chapter_number = current_scene_data.get("_chapter_number", 1)

        # Count scenes in the CURRENT chapter: archived scenes after the last
        # chapter's ending_scene boundary. Counting all scenes ever would leave
        # the gate permanently above its minimum after the first archive.
        scenes = events_data.get("scenes", {})
        # Skip reserved field-name keys: a misdirected add_new can nest a
        # whole {"scenes": {...}} wrapper under a key like 'events', which
        # would inflate the span math by one phantom scene.
        _RESERVED_SCENE_KEYS = {"events", "past", "chapters", "arcs"}
        scene_names = [k for k in scenes.keys() if k not in _RESERVED_SCENE_KEYS]
        total_scenes = len(scene_names)
        raw_chapters = _entries_as_list(events_data.get("chapters"))
        # Sanitize: only well-formed dicts count as chapters. A corrupted
        # array (flattened field scalars from a mis-shaped round-trip) must
        # not feed ending_scene scans or grow via appends.
        chapters_list = [c for c in raw_chapters if isinstance(c, dict)]
        if len(chapters_list) != len(raw_chapters):
            print(f"{_WARNING}Dropping {len(raw_chapters) - len(chapters_list)} malformed "
                  f"chapter entr(y/ies) from events.chapters.{_RESET}")
            events_data["chapters"] = chapters_list
        last_ending_scene = 0
        for ch_entry in reversed(chapters_list):
            if isinstance(ch_entry, dict):
                val = ch_entry.get("ending_scene")
                if isinstance(val, int) and not isinstance(val, bool) and val > last_ending_scene:
                    last_ending_scene = val
                break
        scenes_in_chapter = total_scenes - last_ending_scene
        if scenes_in_chapter < 0:
            scenes_in_chapter = 0

        # Get chapter configuration from schema, overlaid by the configured
        # cadence profile (summarizer.config["cadence_profile"], default None
        # = schema numbers untouched).
        chapters_schema = self.schema_parser.get_subject_class("chapters")
        if not chapters_schema:
            chapters_schema = self.schema_parser.definitions.get("Chapters")

        schema_defaults = {}
        if chapters_schema:
            schema_defaults = chapters_schema.defaults if hasattr(chapters_schema, 'defaults') else {}
        suggested_min, suggested_max, max_hard = _resolve_cadence(
            schema_defaults, "chapters", "scenes", (4, 8, 10),
            (self.summarizer.config or {}).get("cadence_profile"),
        )
        gate_check_prompt = None

        if chapters_schema:
            gate_check_prompt = chapters_schema.gate_check_prompt_template

        if not force_chapter and scenes_in_chapter < suggested_min:
            print(f"{_GRAY}Chapter has {scenes_in_chapter} scenes (min suggested: {suggested_min}). Skipping chapter check.{_RESET}")
            return

        if force_chapter:
            should_archive = True
            print(f"{_BOLD}Forcing chapter transition.{_RESET}")
        elif gate_check_prompt:
            # Use the schema template
            scene_in_chapter = scenes_in_chapter
            chapters_count = len(chapters_list)

            # Compact digest of the scenes inside the current chapter span so
            # the gate judges story content, not bare counters.
            span_start = max(0, total_scenes - scenes_in_chapter)
            span_lines = []
            for idx, name in enumerate(scene_names[span_start:], start=span_start + 1):
                entry = scenes.get(name) or {}
                summary = str(entry.get("summary", ""))[:160]
                span_lines.append(f"Scene {idx}: {name} - {summary}")
            span_digest = "\n".join(span_lines) if span_lines else "(no archived scenes yet)"

            gate_prompt = render_jinja_template(
                gate_check_prompt,
                scenes_count=total_scenes,
                chapters_count=chapters_count,
                current_scene_number=current_scene_data.get("_scene_number", 1),
                current_chapter_number=current_chapter_number,
                current_arc_number=current_scene_data.get("_arc_number", 1),
                scenes_in_chapter=scenes_in_chapter,
                scene_in_chapter=scene_in_chapter,
                chapters_in_arc=chapters_count,
                chapter_in_arc=current_chapter_number,
                suggested_min=suggested_min,
                suggested_max=suggested_max,
                max_scenes_before_required=max_hard,
                span_digest=span_digest,
            )

            if runtime.stop_everything:
                return

            print(f"{_INPUT}Checking if chapter should be archived (scenes: {scenes_in_chapter})...{_RESET}")

            current_custom_state = self.custom_state or {}
            llm_response, stop_reason = self.summarizer.generate_using_tgwui(
                prompt=gate_prompt,
                state=current_custom_state,
                history_path=self.history_path,
                stopping_strings=["YES", "NO"],
                match_prefix_only=True,
                                phase_id='chapters',
                    step_id='check_archive',
)

            if runtime.stop_everything:
                return

            response_upper = strip_response(llm_response).upper().strip()
            should_archive = (
                "YES" in response_upper
                or scenes_in_chapter >= max_hard
                or (stop_reason and "YES" in stop_reason.upper())
            )

            if not should_archive:
                print(f"{_GRAY}Chapter continues (response: {llm_response[:50]}...).{_RESET}")
                return
        else:
            print(f"{_ERROR}No gate_check_prompt_template for chapters. Skipping.{_RESET}")
            return

        if force_chapter and self.summarizer.last:
            self.summarizer.last.force_next_chapter = False

        recent_scene_name = scene_names[-1] if scene_names else "Unknown"
        recent_scene_summary = scenes[recent_scene_name].get("summary", "No summary") if recent_scene_name in scenes else ""

        # Scene inventory of the chapter span (name + summary) so the generated
        # chapter aggregates the whole episode, not just the newest scene.
        span_start = max(0, total_scenes - max(scenes_in_chapter, 1))
        chapter_scenes_block = "\n".join(
            f"- {name}: {str((scenes.get(name) or {}).get('summary', ''))[:200]}"
            for name in scene_names[span_start:]
        ) or "- (no archived scenes)"

        print(f"{_SUCCESS}Archiving chapter {current_chapter_number} and creating new one.{_RESET}")

        # Generate chapter data
        chapter_title = f"Chapter {current_chapter_number}"
        chapter_generation_prompt = f"""Based on all the scenes in the current chapter, generate the full data for the chapter named '{chapter_title}'.

Scenes in this chapter ({len(scene_names[span_start:])}):
{chapter_scenes_block}

Most recent Scene: "{recent_scene_name}"
Summary: {recent_scene_summary}

Characters involved:
{json.dumps(current_scene_data.get("now", {}).get("who", {}).get("characters", []), indent=2) if current_scene_data.get("now", {}).get("who", {}).get("characters") else "No characters available"}

Generate chapter data that includes:
- A title (you can rename from "Chapter {current_chapter_number}" to something more evocative)
- A concise summary of what happened in this chapter
- Key changes that occurred, each with a description and the scene where it occurred

Schema for Chapter:
{{
  "title": "str",
  "starting_scene": "int (scene index where this chapter began, starting from 1)",
  "ending_scene": "int (scene index where this chapter concluded)",
  "scenes": "list[int] (list of scene indices in this chapter)",
  "summary": "str",
  "key_changes": [
    {{
      "description": "str (what changed)",
      "scene": "str (scene name where this occurred)"
    }}
  ],
  "status": "str ('active', 'concluded', or 'suspended')"
}}

Respond with ONLY the JSON object for this chapter."""

        chapter_response, _ = self.summarizer.generate_using_tgwui(
            prompt=chapter_generation_prompt,
            state=current_custom_state,
            history_path=self.history_path,
                            phase_id='chapters',
                    step_id='archive',
)

        if runtime.stop_everything:
            return

        try:
            chapter_data = jsonc.loads(strip_response(chapter_response))
            if isinstance(chapter_data, dict):
                # Ensure required fields
                if "title" not in chapter_data:
                    chapter_data["title"] = chapter_title
                if "status" not in chapter_data:
                    chapter_data["status"] = "concluded"
                # Span fields are deterministic truth (scene indices computed
                # from the archive chain); the model contributes title,
                # summary, key_changes, status. Overriding unconditionally
                # prevents hallucinated indices (seen live: end=14 with 5
                # scenes on disk).
                span_indices = list(range(last_ending_scene + 1, total_scenes + 1)) or [total_scenes]
                chapter_data["starting_scene"] = span_indices[0]
                chapter_data["ending_scene"] = total_scenes
                chapter_data["scenes"] = span_indices
                if "key_changes" not in chapter_data:
                    chapter_data["key_changes"] = []

                # Add chapter to events (schema shape: list[Chapter]; tolerate
                # legacy dict-keyed-by-title from old archive code)
                chapters = _entries_as_list(events_data.get("chapters"))
                for i, existing in enumerate(chapters):
                    if isinstance(existing, dict) and existing.get("title") == chapter_data["title"]:
                        chapters[i] = chapter_data
                        break
                else:
                    chapters.append(chapter_data)
                events_data["chapters"] = chapters

                new_chapter_number = current_chapter_number + 1
                current_scene_data["_chapter_number"] = new_chapter_number
                print(f"{_SUCCESS}Archived chapter {current_chapter_number} and set new chapter number to {new_chapter_number}.{_RESET}")
                # Persist the mutation: per-subject files were already saved
                # inside DataSummarizer.generate BEFORE the boundary checks ran,
                # so without an explicit save the chapter + numbering exist only
                # in memory and are lost when the next turn loads from disk
                # (found live: chapter 1 archived at t2, every events.json on
                # disk still chapters=[]). The arc path saves its own file; the
                # scene-numbering stamps ride current_scene.json.
                save_json(events_data, self.history_path / "events.json")
                if isinstance(current_scene_data, dict) and current_scene_data:
                    save_json(current_scene_data, self.history_path / "current_scene.json")
                # Canon Synopsis P0: refresh the story-so-far digest now that
                # the new chapter is part of the archive chain.
                self._regenerate_canon_digest(f"chapter_{current_chapter_number}")

        except json.JSONDecodeError as e:
            print(f"{_ERROR}Failed to parse chapter data: {e}. Chapter response: {chapter_response[:200]}...{_RESET}")

    def check_and_archive_arc(self):
        """Check if the current arc should be archived and create a new one if needed.

        This is called after chapter archival to determine if an arc boundary
        should be crossed based on LLM-driven decision making.

        Can be forced via summarizer.last.force_next_arc = True
        """
        if runtime.stop_everything:
            return

        events_data = self.all_subjects_data.get("events", {})
        if not events_data:
            print(f"{_GRAY}No events data available. Skipping arc check.{_RESET}")
            return

        current_custom_state = self.custom_state or {}

        # Check for forced arc transition
        force_arc = self.summarizer.last and self.summarizer.last.force_next_arc
        if force_arc:
            print(f"{_BOLD}Force next arc requested. Skipping LLM check.{_RESET}")

        # Get current arc info
        current_scene_data = self.all_subjects_data.get("current_scene", {})
        current_chapter_number = current_scene_data.get("_chapter_number", 1)
        current_arc_number = current_scene_data.get("_arc_number", 1)

        # Load arcs data
        arcs_path = self.history_path / "arcs.json"
        arcs_data = load_json(arcs_path) or {}

        # Count chapters in the CURRENT arc: archived chapters after the last
        # arc's ending_chapter boundary (index-based; 0 when no arc exists yet).
        raw_chapter_entries = _entries_as_list(events_data.get("chapters"))
        chapters = [c for c in raw_chapter_entries if isinstance(c, dict)]
        if len(chapters) != len(raw_chapter_entries):
            print(f"{_WARNING}Arc check: ignoring {len(raw_chapter_entries) - len(chapters)} "
                  f"malformed entries in events.chapters.{_RESET}")
        total_chapters = len(chapters)
        raw_arcs = _entries_as_list(arcs_data) if isinstance(arcs_data, (dict, list)) else []
        arcs_list = [a for a in raw_arcs if isinstance(a, dict)]
        if len(arcs_list) != len(raw_arcs):
            print(f"{_WARNING}Dropping {len(raw_arcs) - len(arcs_list)} malformed "
                  f"arc entr(y/ies) from arcs.json.{_RESET}")
            arcs_data = arcs_list
            save_json(arcs_data, arcs_path)
        last_ending_chapter = 0
        for arc_entry in reversed(arcs_list):
            if isinstance(arc_entry, dict):
                val = arc_entry.get("ending_chapter")
                if isinstance(val, int) and not isinstance(val, bool) and val > last_ending_chapter:
                    last_ending_chapter = val
                break
        chapters_in_arc = total_chapters - last_ending_chapter
        if chapters_in_arc < 0:
            chapters_in_arc = 0

        # Get arc configuration from schema, overlaid by the configured
        # cadence profile (see check_and_archive_chapter).
        arcs_schema = self.schema_parser.get_subject_class("arcs")
        if not arcs_schema:
            arcs_schema = self.schema_parser.definitions.get("Arcs")

        schema_defaults = {}
        if arcs_schema:
            schema_defaults = arcs_schema.defaults if hasattr(arcs_schema, 'defaults') else {}
        suggested_min, suggested_max, max_hard = _resolve_cadence(
            schema_defaults, "arcs", "chapters", (2, 5, 8),
            (self.summarizer.config or {}).get("cadence_profile"),
        )
        gate_check_prompt = None

        if arcs_schema:
            gate_check_prompt = arcs_schema.gate_check_prompt_template

        if not force_arc and chapters_in_arc < suggested_min:
            print(f"{_GRAY}Arc has {chapters_in_arc} chapters (min suggested: {suggested_min}). Skipping arc check.{_RESET}")
            return

        if force_arc:
            should_archive = True
            print(f"{_BOLD}Forcing arc transition.{_RESET}")
        elif gate_check_prompt:
            # Use the schema template
            chapter_in_arc = chapters_in_arc
            scenes_count = len(events_data.get("scenes", {}))

            # Compact digest of the chapters inside the current arc span.
            arc_span_start = max(0, total_chapters - chapters_in_arc)
            arc_span_lines = []
            for idx, ch_entry in enumerate(chapters[arc_span_start:], start=arc_span_start + 1):
                if isinstance(ch_entry, dict):
                    title = str(ch_entry.get("title", f"Chapter {idx}"))[:80]
                    summary = str(ch_entry.get("summary", ""))[:160]
                    arc_span_lines.append(f"Chapter {idx}: {title} - {summary}")
            chapters_digest = "\n".join(arc_span_lines) if arc_span_lines else "(no archived chapters yet)"

            gate_prompt = render_jinja_template(
                gate_check_prompt,
                scenes_count=scenes_count,
                chapters_count=total_chapters,
                current_scene_number=current_scene_data.get("_scene_number", 1),
                current_chapter_number=current_chapter_number,
                current_arc_number=current_arc_number,
                scenes_in_chapter=scenes_count,
                scene_in_chapter=current_scene_data.get("_scene_number", 1),
                chapters_in_arc=chapters_in_arc,
                chapter_in_arc=chapter_in_arc,
                suggested_min=suggested_min,
                suggested_max=suggested_max,
                max_chapters_before_required=max_hard,
                chapters_digest=chapters_digest,
            )

            if runtime.stop_everything:
                return

            print(f"{_INPUT}Checking if arc should be archived (chapters: {chapters_in_arc})...{_RESET}")

            current_custom_state = self.custom_state or {}
            llm_response, stop_reason = self.summarizer.generate_using_tgwui(
                prompt=gate_prompt,
                state=current_custom_state,
                history_path=self.history_path,
                stopping_strings=["YES", "NO"],
                match_prefix_only=True,
                                phase_id='arcs',
                    step_id='check_archive',
)

            if runtime.stop_everything:
                return

            response_upper = strip_response(llm_response).upper().strip()
            should_archive = (
                "YES" in response_upper
                or chapters_in_arc >= max_hard
                or (stop_reason and "YES" in stop_reason.upper())
            )

            if not should_archive:
                print(f"{_GRAY}Arc continues (response: {llm_response[:50]}...).{_RESET}")
                return
        else:
            print(f"{_ERROR}No gate_check_prompt_template for arcs. Skipping.{_RESET}")
            return

        if force_arc and self.summarizer.last:
            self.summarizer.last.force_next_arc = False

        recent_chapter = chapters[-1] if chapters else {}
        recent_chapter_name = recent_chapter.get("title", "Unknown") if chapters else "Unknown"
        recent_chapter_summary = recent_chapter.get("summary", "No summary") if chapters else ""

        print(f"{_SUCCESS}Archiving arc {current_arc_number} and creating new one.{_RESET}")

        # Generate arc data
        arc_title = f"Arc {current_arc_number}"
        arc_generation_prompt = f"""Based on all the chapters in the current arc, generate the full data for the arc named '{arc_title}'.

Recent Chapter: "{recent_chapter_name}"
Summary: {recent_chapter_summary}

Chapters in this arc:
{json.dumps(chapters, indent=2) if chapters else "No chapters available"}

Generate arc data that includes:
- A title (you can rename from "Arc {current_arc_number}" to something more evocative)
- A concise summary of what happened in this arc
- Character arc progressions and key relationship shifts
- Plot threads introduced or resolved
- Status: 'active', 'concluded', or 'suspended'

Schema for Arc:
{{
  "title": "str",
  "starting_chapter": "int (chapter index where this arc began, starting from 1)",
  "ending_chapter": "int (chapter index where this arc concluded)",
  "chapters": "list[int] (list of chapter indices in this arc)",
  "summary": "str",
  "character_arcs": "str (how characters changed)",
  "relationship_shifts": "str (how relationships evolved)",
  "plot_threads": ["str (plot threads in this arc)"],
  "status": "str ('active', 'concluded', or 'suspended')"
}}

Respond with ONLY the JSON object for this arc."""

        arc_response, _ = self.summarizer.generate_using_tgwui(
            prompt=arc_generation_prompt,
            state=current_custom_state,
            history_path=self.history_path,
                            phase_id='arcs',
                    step_id='archive',
)

        if runtime.stop_everything:
            return

        try:
            arc_data = jsonc.loads(strip_response(arc_response))
            if isinstance(arc_data, dict):
                # Ensure required fields
                if "title" not in arc_data:
                    arc_data["title"] = arc_title
                if "status" not in arc_data:
                    arc_data["status"] = "concluded"
                # Chapter spans are deterministic truth (see chapter path).
                arc_data["starting_chapter"] = last_ending_chapter + 1
                arc_data["ending_chapter"] = total_chapters
                arc_data["chapters"] = list(range(last_ending_chapter + 1, total_chapters + 1))
                if "character_arcs" not in arc_data:
                    arc_data["character_arcs"] = ""
                if "relationship_shifts" not in arc_data:
                    arc_data["relationship_shifts"] = ""
                if "plot_threads" not in arc_data:
                    arc_data["plot_threads"] = []
                if "summary" not in arc_data:
                    arc_data["summary"] = recent_chapter_summary

                # Add arc to dict keyed by title
                arc_key = arc_data.get("title", arc_title)
                arcs_data[arc_key] = arc_data
                save_json(arcs_data, arcs_path)

                new_arc_number = current_arc_number + 1
                current_scene_data["_arc_number"] = new_arc_number
                print(f"{_SUCCESS}Archived arc {current_arc_number} and set new arc number to {new_arc_number}.{_RESET}")
                # arcs.json was saved above; the _arc_number stamp lives in
                # current_scene, which generate() saved before this check ran.
                if isinstance(current_scene_data, dict) and current_scene_data:
                    save_json(current_scene_data, self.history_path / "current_scene.json")
                # Canon Synopsis P0: refresh the digest with the arc's
                # long-range framing folded in.
                self._regenerate_canon_digest(f"arc_{current_arc_number}")

        except json.JSONDecodeError as e:
            print(f"{_ERROR}Failed to parse arc data: {e}. Arc response: {arc_response[:200]}...{_RESET}")

    # --- Helper methods for parsing and applying LLM updates ---
