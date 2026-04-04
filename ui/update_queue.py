"""Thread-safe event queue for real-time UI updates via SSE."""

import asyncio
import json
import threading
import time
import uuid
from collections import defaultdict
from typing import Any


class UpdateQueue:
    """Thread-safe event queue with multi-subscriber support for SSE.

    Design:
    - Events are published by the Summarizer process (any thread)
    - Each SSE connection subscribes and receives events via its own asyncio.Queue
    - Recent events are buffered for new subscribers to catch up
    - Events are serialized to JSON for SSE transmission
    """

    def __init__(self, max_buffer: int = 1000):
        self._subscribers: dict[str, asyncio.Queue] = {}
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
        """Publish an event from any thread.

        Args:
            event: Event dict with at least 'type' and 'phase' keys
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

            # Notify all subscribers (non-blocking)
            for sid, queue in list(self._subscribers.items()):
                try:
                    queue.put_nowait(serialized)
                except asyncio.QueueFull:
                    print(f"[DSS Queue] Subscriber {sid} queue full, skipping")

        print(f"[DSS Queue] Published. Buffer: {len(self._buffer)}, Subscribers: {len(self._subscribers)}")

    def subscribe(self, subscriber_id: str = None) -> "SubscriberContext":
        """Subscribe to events. Returns a context manager.

        Usage:
            async with queue.subscribe() as sub:
                async for event_json in sub:
                    # process event
        """
        sub_id = subscriber_id or str(uuid.uuid4())[:8]
        queue: asyncio.Queue = asyncio.Queue(maxsize=500)

        with self._lock:
            self._subscribers[sub_id] = queue

        return SubscriberContext(self, sub_id, queue)

    def unsubscribe(self, subscriber_id: str) -> None:
        """Remove a subscriber."""
        with self._lock:
            self._subscribers.pop(subscriber_id, None)

    def get_buffered_events(self) -> list[str]:
        """Get all buffered events as serialized JSON strings."""
        with self._lock:
            return [json.dumps(e, default=str) for e in self._buffer]

    def get_state(self) -> dict:
        """Get current state snapshot."""
        with self._lock:
            return json.loads(json.dumps(self._state, default=str))

    def clear(self) -> None:
        """Clear all buffered events and reset state."""
        with self._lock:
            self._buffer.clear()
            self._state = {
                "active_phases": [],
                "completed_phases": [],
                "pending_phases": [],
                "progress": {"completed": 0, "total": 0, "percent": 0},
                "running": False,
            }

    def _update_state(self, event: dict) -> None:
        """Update internal state based on event type."""
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
    """Async context manager for SSE subscribers."""

    def __init__(self, queue: UpdateQueue, sub_id: str, event_queue: asyncio.Queue):
        self._queue = queue
        self._sub_id = sub_id
        self._event_queue = event_queue

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._queue.unsubscribe(self._sub_id)

    async def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return await self._event_queue.get()
        except asyncio.CancelledError:
            self._queue.unsubscribe(self._sub_id)
            raise StopAsyncIteration


# Singleton instance
_update_queue: UpdateQueue | None = None


def get_update_queue() -> UpdateQueue:
    """Get the singleton UpdateQueue instance."""
    global _update_queue
    if _update_queue is None:
        _update_queue = UpdateQueue()
    return _update_queue


def reset_update_queue() -> None:
    """Reset the singleton (for testing or re-initialization)."""
    global _update_queue
    if _update_queue:
        _update_queue.clear()
    _update_queue = None
