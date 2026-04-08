"""Thread-safe event queue for real-time UI updates via SSE."""

import json
import queue
import threading
import time
import uuid
from collections import defaultdict
from typing import Any


class UpdateQueue:
    """Thread-safe event queue with multi-subscriber support for SSE.

    Design:
    - Events are published by the Summarizer process (any thread)
    - Each SSE connection subscribes and receives events via its own queue.Queue
    - Recent events are buffered for new subscribers to catch up
    - Events are serialized to JSON for SSE transmission
    """

    def __init__(self, max_buffer: int = 20000):
        """
        Create a thread-safe UpdateQueue with an optional bounded event buffer.
        
        Parameters:
            max_buffer (int): Maximum number of past events to retain in the internal buffer; older events are discarded when the buffer exceeds this size. Default is 20000.
        """
        self._subscribers: dict[str, queue.Queue] = {}
        self._buffer: list[dict] = []
        self._max_buffer = max_buffer
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {
            "active_phases": [],
            "completed_phases": [],
            "pending_phases": [],
            "progress": {"completed": 0, "total": 0, "percent": 0},
            "running": False,
        }

    def publish(self, event: dict) -> None:
        """
        Publish an event to the queue, update the shared state snapshot, and notify all subscribers.
        
        Ensures the event has an `id` and `timestamp` (they are added if missing), appends the event to the internal bounded buffer, updates the derived `_state` based on the event `type`, and enqueues the JSON-serialized event to each subscriber's asyncio queue without blocking; subscriber queues that are full are skipped.
        
        Parameters:
            event (dict): Event dictionary. Must include a `"type"` key and typically a `"phase"` mapping; if `"id"` or `"timestamp"` are absent they will be generated.
        """
        event["id"] = event.get("id", str(uuid.uuid4())[:8])
        event["timestamp"] = event.get("timestamp", time.time())

        serialized = json.dumps(event, default=str)
        print(f"[DSS Queue] Publishing: {event.get('type')} - {event.get('phase', {}).get('id', '?')}")

        with self._lock:
            self._buffer.append(event)
            if len(self._buffer) > self._max_buffer:
                self._buffer = self._buffer[-self._max_buffer:]

            # Update state snapshot
            self._update_state(event)

            # Notify all subscribers (thread-safe — queue.Queue is inherently thread-safe)
            for sid, q in list(self._subscribers.items()):
                try:
                    q.put_nowait(serialized)
                except queue.Full:
                    print(f"[DSS Queue] Subscriber {sid} queue full, skipping")

        print(f"[DSS Queue] Published. Buffer: {len(self._buffer)}, Subscribers: {len(self._subscribers)}")

    def subscribe(self, subscriber_id: str = None) -> "SubscriberContext":
        """
        Register a new subscriber and return an async context manager that yields the subscriber's serialized events.
        
        Parameters:
            subscriber_id (str, optional): Custom identifier for the subscriber. If omitted, an 8-character UUID suffix is generated.
        
        Returns:
            SubscriberContext: An async context manager and async iterator that yields serialized JSON event strings for the subscriber and automatically unregisters the subscriber when exited or cancelled.
        """
        sub_id = subscriber_id or str(uuid.uuid4())[:8]
        q: queue.Queue = queue.Queue(maxsize=500)

        with self._lock:
            self._subscribers[sub_id] = q

        return SubscriberContext(self, sub_id, q)

    def unsubscribe(self, subscriber_id: str) -> None:
        """
        Unregisters a subscriber so it no longer receives published events.
        
        Parameters:
            subscriber_id (str): The subscriber identifier assigned when subscribing; if the id is not registered this call has no effect.
        """
        with self._lock:
            self._subscribers.pop(subscriber_id, None)

    def get_buffered_events(self) -> list[str]:
        """
        Return a snapshot of the internal event buffer as JSON-serialized strings.
        
        Returns:
            list[str]: JSON-serialized representations of the buffered event dictionaries, in the same order they appear in the buffer.
        """
        with self._lock:
            return [json.dumps(e, default=str) for e in self._buffer]

    def get_state(self) -> dict:
        """
        Return a thread-safe deep copy of the queue's shared UI state snapshot.
        
        The returned dictionary contains the current values for:
        - `active_phases` (list): phases currently in progress
        - `completed_phases` (list): finished phases (errors may be marked on phase dicts)
        - `pending_phases` (list): phases queued to run
        - `progress` (number or mapping): progress counters or metrics
        - `running` (bool): whether a session is active
        
        Returns:
            dict: A deep-copied snapshot of the internal state.
        """
        with self._lock:
            return json.loads(json.dumps(self._state, default=str))

    def clear(self) -> None:
        """
        Clear all buffered events and reset the shared state to its initial empty defaults.
        
        This operation removes every event from the internal buffer and resets `active_phases`, `completed_phases`,
        `pending_phases`, `progress` (to zeros), and `running` (to False). The reset is performed under the instance lock.
        """
        with self._lock:
            print("[DSS Queue] Clearing queue...")
            self._buffer.clear()
            print("[DSS Queue] Buffer cleared:", len(self._buffer))
            self._state = {
                "active_phases": [],
                "completed_phases": [],
                "pending_phases": [],
                "progress": {"completed": 0, "total": 0, "percent": 0},
                "running": False,
            }

    def _update_state(self, event: dict) -> None:
        """
        Update the queue's shared UI state snapshot based on an incoming event.
        
        Modifies the instance's internal `_state` (keys: `running`, `active_phases`, `completed_phases`,
        `pending_phases`, `progress`) according to `event["type"]` and `event.get("phase", {}).get("id")`:
        
        - `session_start`: sets `running` to True, clears `active_phases` and `completed_phases`,
          sets `pending_phases` from `event["queue"]` (if present), and updates `progress`.
        - `phase_start`: ensures the phase (by id) is not duplicated in `active_phases`, appends
          `event["phase"]` to `active_phases`, and removes the phase from `pending_phases`.
        - `phase_done`: removes the phase from `active_phases`, appends `event["phase"]` to
          `completed_phases`, and updates `progress`.
        - `phase_error`: marks the provided phase dict with `error = True`, removes it from
          `active_phases`, and appends it to `completed_phases`.
        - `session_end`: sets `running` to False and updates `progress`.
        
        Parameters:
            event (dict): Event dictionary expected to include a `"type"` key and, for phase-related
                events, a `"phase"` dict with an `"id"`. Optional keys: `"queue"` and `"progress"`.
        """
        etype = event.get("type", "")
        phase_id = event.get("phase", {}).get("id", "")

        if etype == "session_start":
            self._state["running"] = True
            self._state["active_phases"] = []
            self._state["completed_phases"] = []
            self._state["pending_phases"] = event.get("queue", [])
            self._state["progress"] = event.get("progress", self._state["progress"])

        elif etype == "phase_start":
            self._state["active_phases"] = [
                p for p in self._state["active_phases"]
                if p.get("id") != phase_id
            ]
            self._state["active_phases"].append(event.get("phase", {}))
            self._state["pending_phases"] = [
                p for p in self._state["pending_phases"]
                if p.get("id") != phase_id
            ]

        elif etype == "phase_done":
            self._state["active_phases"] = [
                p for p in self._state["active_phases"]
                if p.get("id") != phase_id
            ]
            self._state["completed_phases"].append(event.get("phase", {}))
            self._state["progress"] = event.get("progress", self._state["progress"])

        elif etype == "phase_error":
            phase = event.get("phase", {})
            phase["error"] = True
            self._state["active_phases"] = [
                p for p in self._state["active_phases"]
                if p.get("id") != phase_id
            ]
            self._state["completed_phases"].append(phase)

        elif etype == "session_end":
            self._state["running"] = False
            self._state["progress"] = event.get("progress", self._state["progress"])


class SubscriberContext:
    """Sync context manager for SSE subscribers."""

    def __init__(self, q: UpdateQueue, sub_id: str, event_queue: queue.Queue):
        """
        Initialize the SubscriberContext that wraps a single subscriber's event queue.
        
        Parameters:
            q (UpdateQueue): Parent UpdateQueue used to unregister the subscriber on exit.
            sub_id (str): Unique identifier for the subscriber.
            event_queue (queue.Queue): Per-subscriber queue.Queue that yields serialized event strings.
        """
        self._queue = q
        self._sub_id = sub_id
        self._event_queue = event_queue

    def __enter__(self):
        """
        Enter the context and return the subscriber context for consumption.
        
        Returns:
            self (SubscriberContext): The context manager instance used to iterate and receive events.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Unregister the subscriber from the parent UpdateQueue when exiting the context.
        
        This method always removes the subscriber identified by `self._sub_id` from the parent queue; it does not suppress or modify any exception raised within the context.
        """
        self._queue.unsubscribe(self._sub_id)

    def __iter__(self):
        return self

    def __next__(self):
        """
        Return the next serialized event string from the subscriber's internal queue.
        
        Returns:
            event (str): The next JSON-serialized event string from the subscriber queue.
        
        Raises:
            StopIteration: Raised when the subscriber is unsubscribed to terminate iteration.
        """
        try:
            return self._event_queue.get(timeout=30)
        except queue.Empty:
            raise StopIteration


# Singleton instance
_update_queue: UpdateQueue | None = None


def get_update_queue() -> UpdateQueue:
    """
    Return the module-level singleton UpdateQueue, creating it if necessary.
    
    Lazily instantiates the singleton on first call and returns the same instance on subsequent calls.
    
    Returns:
        UpdateQueue: The shared UpdateQueue instance.
    """
    global _update_queue
    if _update_queue is None:
        _update_queue = UpdateQueue()
    return _update_queue


def reset_update_queue() -> None:
    """
    Reset the module-level UpdateQueue singleton.
    
    If a singleton instance exists, this calls its `clear()` method and removes the module reference so that a subsequent `get_update_queue()` will create a new instance. Intended for testing and re-initialization.
    """
    global _update_queue
    if _update_queue:
        _update_queue.clear()
    _update_queue = None
