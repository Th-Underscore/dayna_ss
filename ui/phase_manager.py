"""Phase manager for tracking summarization progress and emitting UI events."""

import time
from typing import Any

from .update_queue import get_update_queue


# Phase definitions with weights for progress calculation
DEFAULT_PHASES = [
    {"id": "init", "name": "Initialization", "weight": 1},
    {"id": "context", "name": "Context Preparation", "weight": 1},
    {"id": "current_scene", "name": "CurrentScene", "weight": 1},
    {"id": "general_info", "name": "GeneralInfo", "weight": 1},
    {"id": "characters", "name": "Characters", "weight": 2},
    {"id": "groups", "name": "Groups", "weight": 1},
    {"id": "events", "name": "Events", "weight": 2},
    {"id": "scene_state", "name": "SceneState", "weight": 1},
    {"id": "arcs", "name": "Arcs", "weight": 2},
    {"id": "chapter_check", "name": "Chapter Boundary Check", "weight": 1},
    {"id": "arc_check", "name": "Arc Boundary Check", "weight": 1},
    {"id": "message_summary", "name": "Message Summarization", "weight": 1},
    {"id": "chunking", "name": "Message Chunking", "weight": 1},
]

# Action titles for step display
ACTION_TITLES = {
    "perform_gate_check": "Gate Check",
    "query_branch_for_changes": "Querying Changes",
    "perform_update": "Applying Updates",
    "add_new": "Detecting New Entries",
    "archive": "Archiving",
    "save": "Saving",
}

# Sub-step descriptions for actions
STEP_DESCRIPTIONS = {
    "perform_gate_check": "Checking if updates are needed...",
    "query_branch_for_changes": "Scanning for field changes...",
    "perform_update": "Updating data with LLM...",
    "add_new": "Querying LLM for new entries...",
    "save": "Writing to disk...",
}


class PhaseManager:
    """Manages phase tracking and emits real-time UI events.

    Usage:
        pm = PhaseManager()
        pm.start_session(phases=[...])

        pm.start_phase("characters")
        pm.start_step("characters", "perform_gate_check")
        pm.update_step("characters", "perform_gate_check", "LLM responded YES")
        pm.done_step("characters", "perform_gate_check")
        pm.done_phase("characters")

        pm.end_session()
    """

    def __init__(self, queue=None):
        self._queue = queue or get_update_queue()
        self._phases: list[dict] = []
        self._phase_lookup: dict[str, dict] = {}
        self._active_phase: str | None = None
        self._active_step: str | None = None
        self._completed_phases: set[str] = set()
        self._phase_steps: dict[str, list[dict]] = {}
        self._total_weight: int = 0
        self._completed_weight: float = 0.0
        self._session_start: float = 0
        self._phase_start_times: dict[str, float] = {}

    def start_session(self, phases: list[dict] = None, subject_names: list[str] = None) -> None:
        """Start a new summarization session.

        Args:
            phases: Custom phase list, or None for defaults
            subject_names: Dynamic subject names from schema (e.g., ["Characters", "Groups"])
        """
        print(f"[DSS Phase] Starting session with subjects: {subject_names}")
        self._phases = []
        self._phase_lookup = {}
        self._active_phase = None
        self._active_step = None
        self._completed_phases = set()
        self._phase_steps = {}
        self._phase_start_times = {}
        self._total_weight = 0
        self._completed_weight = 0.0
        self._session_start = time.time()

        if phases:
            base_phases = phases
        elif subject_names:
            # Build phases from subject names
            base_phases = [
                {"id": "init", "name": "Initialization", "weight": 1},
                {"id": "context", "name": "Context Preparation", "weight": 1},
            ]
            for name in subject_names:
                pid = name.lower().replace(" ", "_")
                base_phases.append({"id": pid, "name": name, "weight": 2})
            base_phases.extend([
                {"id": "chapter_check", "name": "Chapter Boundary Check", "weight": 1},
                {"id": "arc_check", "name": "Arc Boundary Check", "weight": 1},
                {"id": "message_summary", "name": "Message Summarization", "weight": 1},
                {"id": "chunking", "name": "Message Chunking", "weight": 1},
            ])
        else:
            base_phases = list(DEFAULT_PHASES)

        for p in base_phases:
            phase = {"id": p["id"], "name": p["name"], "weight": p.get("weight", 1)}
            self._phases.append(phase)
            self._phase_lookup[p["id"]] = phase
            self._phase_steps[p["id"]] = []
            self._total_weight += phase["weight"]

        self._queue.publish({
            "type": "session_start",
            "phase": {"id": "_session", "name": "Summarization"},
            "queue": [{"id": p["id"], "name": p["name"], "status": "pending"} for p in self._phases],
            "progress": self._get_progress(),
        })

    def start_phase(self, phase_id: str, name: str = None) -> None:
        """Mark a phase as started."""
        self._active_phase = phase_id
        self._active_step = None
        self._phase_start_times[phase_id] = time.time()

        if phase_id not in self._phase_steps:
            self._phase_steps[phase_id] = []

        phase = self._get_phase_info(phase_id, name)

        self._queue.publish({
            "type": "phase_start",
            "phase": phase,
            "progress": self._get_progress(),
            "queue": self._get_pending(),
        })

    def done_phase(self, phase_id: str, name: str = None) -> None:
        """Mark a phase as completed."""
        self._completed_phases.add(phase_id)
        phase = self._get_phase_info(phase_id, name)
        phase["steps"] = self._phase_steps.get(phase_id, [])
        phase["elapsed"] = self._get_phase_elapsed(phase_id)

        if phase_id in self._phase_lookup:
            self._completed_weight += self._phase_lookup[phase_id].get("weight", 1)

        self._active_phase = None
        self._active_step = None

        self._queue.publish({
            "type": "phase_done",
            "phase": phase,
            "progress": self._get_progress(),
            "queue": self._get_pending(),
        })

    def error_phase(self, phase_id: str, error: str, name: str = None) -> None:
        """Mark a phase as errored."""
        phase = self._get_phase_info(phase_id, name)
        phase["error"] = error
        phase["steps"] = self._phase_steps.get(phase_id, [])

        self._completed_phases.add(phase_id)
        self._active_phase = None
        self._active_step = None

        self._queue.publish({
            "type": "phase_error",
            "phase": phase,
            "progress": self._get_progress(),
            "queue": self._get_pending(),
        })

    def start_step(self, phase_id: str, step_id: str, message: str = None, data: dict = None) -> None:
        """Start a sub-step within a phase."""
        step = {
            "id": step_id,
            "title": self._generate_step_title(phase_id, step_id, data),
            "message": message or STEP_DESCRIPTIONS.get(step_id, "Processing..."),
            "status": "processing",
            "start_time": time.time(),
            "updates": [],
        }

        self._active_step = step_id
        if phase_id not in self._phase_steps:
            self._phase_steps[phase_id] = []
        self._phase_steps[phase_id].append(step)

        self._queue.publish({
            "type": "step_start",
            "phase": self._get_phase_info(phase_id),
            "step": step,
            "progress": self._get_progress(),
        })

    def update_step(self, phase_id: str, step_id: str, message: str, data: dict = None) -> None:
        """Update a running sub-step with new information."""
        steps = self._phase_steps.get(phase_id, [])
        step = None
        for s in reversed(steps):
            if s["id"] == step_id:
                step = s
                break

        if step:
            step["updates"].append({"message": message, "time": time.time()})
            step["message"] = message

            self._queue.publish({
                "type": "step_update",
                "phase": self._get_phase_info(phase_id),
                "step": step,
                "progress": self._get_progress(),
                "data": data,
            })

    def done_step(self, phase_id: str, step_id: str, message: str = None, data: dict = None) -> None:
        """Mark a sub-step as completed."""
        steps = self._phase_steps.get(phase_id, [])
        step = None
        for s in reversed(steps):
            if s["id"] == step_id:
                step = s
                break

        if step:
            step["status"] = "done"
            step["end_time"] = time.time()
            step["elapsed"] = step["end_time"] - step["start_time"]
            if message:
                step["message"] = message

            self._queue.publish({
                "type": "step_done",
                "phase": self._get_phase_info(phase_id),
                "step": step,
                "progress": self._get_progress(),
                "data": data,
            })

    def end_session(self) -> None:
        """End the summarization session."""
        elapsed = time.time() - self._session_start

        self._queue.publish({
            "type": "session_end",
            "phase": {"id": "_session", "name": "Summarization"},
            "progress": {
                "completed": len(self._phases),
                "total": len(self._phases),
                "percent": 100.0,
                "elapsed": elapsed,
            },
        })

        self._active_phase = None
        self._active_step = None

    def skip_phase(self, phase_id: str, reason: str = "Not applicable", name: str = None) -> None:
        """Skip a phase without processing."""
        self._completed_phases.add(phase_id)
        phase = self._get_phase_info(phase_id, name)
        phase["skipped"] = True
        phase["skip_reason"] = reason
        phase["steps"] = []

        if phase_id in self._phase_lookup:
            self._completed_weight += self._phase_lookup[phase_id].get("weight", 1)

        self._queue.publish({
            "type": "phase_done",
            "phase": phase,
            "progress": self._get_progress(),
            "queue": self._get_pending(),
        })

    def _get_phase_info(self, phase_id: str, name: str = None) -> dict:
        """Get phase info dict."""
        if phase_id in self._phase_lookup:
            return dict(self._phase_lookup[phase_id])
        return {"id": phase_id, "name": name or phase_id.title(), "weight": 1}

    def _get_progress(self) -> dict:
        """Calculate current progress."""
        total = len(self._phases)
        completed = len(self._completed_phases)
        percent = (self._completed_weight / self._total_weight * 100) if self._total_weight > 0 else 0

        return {
            "completed": completed,
            "total": total,
            "percent": round(percent, 1),
            "weighted_completed": round(self._completed_weight, 1),
            "weighted_total": self._total_weight,
        }

    def _get_pending(self) -> list[dict]:
        """Get list of pending phases."""
        pending = []
        for p in self._phases:
            if p["id"] not in self._completed_phases and p["id"] != self._active_phase:
                pending.append({"id": p["id"], "name": p["name"], "status": "pending"})
        return pending

    def _get_phase_elapsed(self, phase_id: str) -> float:
        """Get elapsed time for a phase (from start_phase call)."""
        start_time = self._phase_start_times.get(phase_id)
        if start_time:
            return time.time() - start_time
        return 0.0

    def _generate_step_title(self, phase_id: str, step_id: str, data: dict = None) -> str:
        """Generate a human-readable title for a step."""
        phase_name = self._get_phase_info(phase_id).get("name", phase_id)
        action_name = ACTION_TITLES.get(step_id, step_id.replace("_", " ").title())

        if data:
            entry_name = data.get("entry_name", data.get("branch_name", ""))
            if entry_name:
                return f"{phase_name}: {action_name} ({entry_name})"

        return f"{phase_name}: {action_name}"

    @property
    def active_phase(self) -> str | None:
        return self._active_phase

    @property
    def completed_count(self) -> int:
        return len(self._completed_phases)

    @property
    def total_phases(self) -> int:
        return len(self._phases)
