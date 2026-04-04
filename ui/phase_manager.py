"""Phase manager for tracking summarization progress and emitting UI events."""

import logging
import time

logger = logging.getLogger(__name__)

from .update_queue import get_update_queue


# Phase definitions with weights for progress calculation
DEFAULT_PHASES = [
    {"id": "init", "name": "Initialization", "weight": 1},
    {"id": "instr_prompt", "name": "Instruction Generation", "weight": 1},
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
        """
        Initialize the PhaseManager and set up its internal tracking state and update queue.
        
        Parameters:
            queue (optional): An injected update queue to publish UI events; if omitted, the module default `get_update_queue()` is used.
        
        Description:
            Creates and initializes internal structures used to track phases, per-phase steps, active phase/step identifiers, completed phases and weights, session start time, and per-phase start times.
        """
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

    def start_session(self, phases: list[dict] | None = None, subject_names: list[str] | None = None) -> None:
        """
        Initialize and open a new summarization session, resetting internal tracking and publishing a `session_start` event.
        
        Resets internal phase/step state, clears completed phase tracking, resets weights and timestamps, and records the session start time. Builds the session phase list from, in order of precedence:
        - the provided `phases` list if present;
        - otherwise a generated list based on `subject_names` (includes fixed `init` and `context` phases, one phase per subject with weight 2, and fixed boundary/chunking phases);
        - otherwise the module `DEFAULT_PHASES`.
        
        Each phase is normalized to a dict with keys `id`, `name`, and `weight`; per-phase step lists and a phase lookup are created and the total weight is computed. Finally, publishes a `session_start` event containing a synthetic session phase (`Summarization`), a pending queue snapshot for all phases, and the initial progress payload.
        
        Parameters:
            phases (list[dict] | None): Optional custom list of phase definitions (each must include `id` and `name`; `weight` is optional and defaults to 1).
            subject_names (list[str] | None): Optional list of subject names to generate phase entries (used only when `phases` is not provided).
        """
        logger.info("Starting session with subjects: %s", subject_names)
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

    def start_phase(self, phase_id: str, name: str | None = None) -> None:
        """
        Begin tracking a phase and publish a `phase_start` update.
        
        Sets the given phase as active, clears any active step, records the phase start time, and ensures a step list exists for the phase. Publishes a `phase_start` event containing the phase payload, current progress, and the pending phase queue.
        
        Parameters:
            phase_id (str): Identifier of the phase to start.
            name (str, optional): Optional display name to override the phase's stored name.
        """
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

    def done_phase(self, phase_id: str, name: str | None = None) -> None:
        """
        Mark the given phase as completed and publish a `phase_done` update to the queue.
        
        Parameters:
            phase_id (str): Identifier of the phase to mark completed.
            name (str, optional): Optional display name to use if the phase is not known.
        
        Description:
            Records the phase as completed, attaches its recorded steps and elapsed time, increments
            the completed weight if the phase is defined in the session, clears the active phase/step,
            and publishes a `phase_done` event containing the phase payload, current progress, and the pending queue.
        """
        if phase_id not in self._completed_phases:
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
        else:
            phase = self._get_phase_info(phase_id, name)
            phase["steps"] = self._phase_steps.get(phase_id, [])
            phase["elapsed"] = self._get_phase_elapsed(phase_id)
            self._queue.publish({
                "type": "phase_done",
                "phase": phase,
                "progress": self._get_progress(),
                "queue": self._get_pending(),
            })

    def error_phase(self, phase_id: str, error: str, name: str | None = None) -> None:
        """
        Record that a phase failed, mark it completed, clear active tracking, and publish a `phase_error` event.
        
        Parameters:
            phase_id (str): Identifier of the phase that encountered an error.
            error (str): Human-readable error message to attach to the phase.
            name (str, optional): Fallback display name used when the phase id is not present in the current phase lookup.
        """
        phase = self._get_phase_info(phase_id, name)
        phase["error"] = error
        phase["steps"] = self._phase_steps.get(phase_id, [])

        if phase_id not in self._completed_phases:
            self._completed_phases.add(phase_id)
            if phase_id in self._phase_lookup:
                self._completed_weight += self._phase_lookup[phase_id].get("weight", 1)
        self._active_phase = None
        self._active_step = None

        self._queue.publish({
            "type": "phase_error",
            "phase": phase,
            "progress": self._get_progress(),
            "queue": self._get_pending(),
        })

    def start_step(self, phase_id: str, step_id: str, message: str | None = None, data: dict | None = None) -> None:
        """
        Begin tracking a sub-step (step) within the specified phase and publish a `step_start` update to the manager's queue.
        
        Parameters:
            phase_id (str): Identifier of the phase this step belongs to.
            step_id (str): Identifier for the sub-step; used to generate the step title.
            message (str, optional): Initial human-readable message for the step; if omitted a default description is used.
            data (dict, optional): Optional metadata used when generating the step title (e.g., may include `entry_name` or `branch_name`).
        
        Side effects:
            - Records the step in the manager's internal step list and sets it as the active step.
            - Publishes a `step_start` event containing the phase info, step payload, and current progress.
        """
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

    def update_step(self, phase_id: str, step_id: str, message: str, data: dict | None = None) -> None:
        """
        Record an update for an active sub-step and publish a 'step_update' event to the update queue.
        
        Parameters:
            phase_id (str): Identifier of the phase that contains the step.
            step_id (str): Identifier of the step to update.
            message (str): New message to append to the step and set as its current message.
            data (dict, optional): Optional additional payload to include with the published event.
        """
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
        else:
            logger.warning("update_step: no matching step found for phase_id='%s', step_id='%s'", phase_id, step_id)

    def done_step(self, phase_id: str, step_id: str, message: str | None = None, data: dict | None = None) -> None:
        """
        Mark a phase sub-step as completed and publish a `step_done` event.
        
        Sets the step's status to "done", records its end time and elapsed duration, optionally updates its message, and publishes a `step_done` event containing the phase info, the completed step, current progress, and the provided `data`.
        
        Parameters:
            phase_id (str): Identifier of the phase that contains the step.
            step_id (str): Identifier of the step to mark as completed.
            message (str, optional): Message to set on the step; if omitted the existing message is kept.
            data (dict, optional): Additional payload to include in the published event.
        """
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
        """
        End the current summarization session and publish a final session event.
        
        Publishes a `session_end` event containing the synthetic session phase (`id: "_session", name: "Summarization"`), progress summary (`completed` and `total` set to the total number of phases, `percent` set to 100.0, and `elapsed` time), and clears the active phase and active step state.
        """
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

    def skip_phase(self, phase_id: str, reason: str = "Not applicable", name: str | None = None) -> None:
        """
        Mark a phase as skipped, record the skip reason, and publish a `phase_done` event.
        
        Parameters:
            phase_id (str): Identifier of the phase to skip.
            reason (str): Human-readable reason why the phase was skipped. Defaults to "Not applicable".
            name (str | None): Optional display name to use if the phase is not known locally.
        
        Description:
            Adds the phase to the completed set, records it as skipped with the provided reason, increments
            completed weighted progress if the phase is known, and publishes the updated phase and progress.
        """
        if phase_id not in self._completed_phases:
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
        else:
            phase = self._get_phase_info(phase_id, name)
            phase["skipped"] = True
            phase["skip_reason"] = reason
            phase["steps"] = []
            self._queue.publish({
                "type": "phase_done",
                "phase": phase,
                "progress": self._get_progress(),
                "queue": self._get_pending(),
            })

    def _get_phase_info(self, phase_id: str, name: str | None = None) -> dict:
        """
        Return a phase info mapping for the given phase identifier.
        
        Parameters:
            phase_id (str): The phase identifier to look up.
            name (str, optional): Fallback display name to use when the phase is not known.
        
        Returns:
            dict: A mapping with keys `id`, `name`, and `weight`. If `phase_id` is known in the manager's phase lookup, returns a shallow copy of that phase's definition; otherwise returns a minimal phase dict using `phase_id`, `name` (or title-cased `phase_id`), and a default weight of 1.
        """
        if phase_id in self._phase_lookup:
            return dict(self._phase_lookup[phase_id])
        return {"id": phase_id, "name": name or phase_id.title(), "weight": 1}

    def _get_progress(self) -> dict:
        """
        Compute the session's phase completion progress and weighted progress values.

        Returns:
            progress (dict): Progress snapshot containing:
                - completed (int): Number of phases marked completed.
                - total (int): Total number of phases in the session.
                - percent (float): Overall completion percentage based on weights, rounded to 1 decimal (0-100).
                - weighted_completed (float): Sum of completed phase weights, rounded to 1 decimal.
                - weighted_total (float): Sum of all phase weights.
        """
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
        """
        List phases that are pending (not completed and not currently active).
        
        Returns:
            list[dict]: A list of phase entries, each a dict with keys:
                - `id` (str): phase identifier
                - `name` (str): phase display name
                - `status` (str): set to `"pending"`
        """
        pending = []
        for p in self._phases:
            if p["id"] not in self._completed_phases and p["id"] != self._active_phase:
                pending.append({"id": p["id"], "name": p["name"], "status": "pending"})
        return pending

    def _get_phase_elapsed(self, phase_id: str) -> float:
        """
        Return the seconds elapsed since the given phase was started.
        
        Parameters:
            phase_id (str): Identifier of the phase.
        
        Returns:
            elapsed (float): Seconds elapsed since the phase's start, or 0.0 if the phase has no recorded start time.
        """
        start_time = self._phase_start_times.get(phase_id)
        if start_time:
            return time.time() - start_time
        return 0.0

    def _generate_step_title(self, phase_id: str, step_id: str, data: dict | None = None) -> str:
        """
        Constructs a human-readable title for a step within a phase.
        
        Parameters:
            phase_id (str): Identifier of the phase; used to look up the phase name.
            step_id (str): Identifier of the step; mapped to a readable action title.
            data (dict, optional): Optional metadata; if it contains `entry_name` or `branch_name`,
                that value is appended to the title in parentheses.
        
        Returns:
            title (str): Formatted title in the form "Phase Name: Action Name" or
                "Phase Name: Action Name (EntryName)" when an entry/branch name is present.
        """
        phase_name = self._get_phase_info(phase_id).get("name", phase_id)
        action_name = ACTION_TITLES.get(step_id, step_id.replace("_", " ").title())

        if data:
            entry_name = data.get("entry_name", data.get("branch_name", ""))
            if entry_name:
                return f"{phase_name}: {action_name} ({entry_name})"

        return f"{phase_name}: {action_name}"

    @property
    def active_phase(self) -> str | None:
        """
        Get the identifier of the currently active phase.
        
        Returns:
            The active phase id as a `str`, or `None` if no phase is active.
        """
        return self._active_phase

    @property
    def completed_count(self) -> int:
        """
        Number of phases that have been completed in the current session.
        
        Returns:
            int: Count of completed phases.
        """
        return len(self._completed_phases)

    @property
    def total_phases(self) -> int:
        """
        Total number of phases in the current session.
        
        Returns:
            int: Number of phases configured for the current session.
        """
        return len(self._phases)