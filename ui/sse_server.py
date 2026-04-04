"""Lightweight SSE server using stdlib http.server.

Runs alongside Gradio in a background thread to push real-time
summarization updates to the UI via Server-Sent Events.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from .update_queue import get_update_queue


class SSEServer:
    """Thread-safe SSE server for pushing real-time updates.

    Endpoints:
        GET /sse/dss-status  - SSE event stream
        GET /sse/dss-state   - JSON state snapshot
        GET /sse/dss-health  - Health check
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 7861, queue=None):
        self.host = host
        self.port = port
        self._queue = queue or get_update_queue()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> int:
        """Start the SSE server. Returns the actual port used."""
        if self._running:
            return self.port

        parent = self

        class SSEHandler(BaseHTTPRequestHandler):
            """HTTP handler with SSE streaming support."""

            def do_GET(self):
                path = urlparse(self.path).path
                try:
                    if path == "/sse/dss-status":
                        self._handle_sse()
                    elif path == "/sse/dss-state":
                        self._handle_state()
                    elif path == "/sse/dss-health":
                        self._handle_health()
                    else:
                        self.send_error(404, "Not Found")
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                    pass  # Client disconnected, normal

            def _handle_sse(self):
                """Stream SSE events to the client."""
                print(f"[DSS SSE] Client connected")
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

                # Send buffered events first (catch-up)
                buffered = parent._queue.get_buffered_events()
                print(f"[DSS SSE] Sending {len(buffered)} buffered events")
                for event_json in buffered:
                    if not self._send_event(event_json):
                        print(f"[DSS SSE] Client disconnected during catch-up")
                        return

                # Stream live events via polling
                last_event_count = len(parent._queue.get_buffered_events())
                last_heartbeat = time.time()
                heartbeat_interval = 5.0  # Send comment every 5s to keep connection alive

                while parent._running:
                    events = parent._queue.get_buffered_events()
                    if len(events) > last_event_count:
                        for event_json in events[last_event_count:]:
                            if not self._send_event(event_json):
                                print(f"[DSS SSE] Client disconnected")
                                return
                        last_event_count = len(events)
                        last_heartbeat = time.time()
                    else:
                        # Heartbeat: send SSE comment to prevent timeout
                        now = time.time()
                        if now - last_heartbeat >= heartbeat_interval:
                            if not self._send_heartbeat():
                                return
                            last_heartbeat = now

                    time.sleep(0.2)

            def _send_event(self, data: str) -> bool:
                """Send a single SSE event. Returns False if connection is dead."""
                try:
                    message = f"data: {data}\n\n"
                    self.wfile.write(message.encode("utf-8"))
                    self.wfile.flush()
                    return True
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                    return False

            def _send_heartbeat(self) -> bool:
                """Send SSE comment (heartbeat) to keep connection alive."""
                try:
                    self.wfile.write(b": heartbeat\n\n")
                    self.wfile.flush()
                    return True
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                    return False

            def _handle_state(self):
                """Return JSON state snapshot."""
                state = parent._queue.get_state()
                data = json.dumps(state, default=str).encode("utf-8")

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(data)

            def _handle_health(self):
                """Health check endpoint."""
                data = json.dumps({"status": "ok", "port": parent.port}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, format, *args):
                """Suppress default logging."""
                pass

            def handle(self):
                """Override to suppress connection errors during request handling."""
                try:
                    super().handle()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                    pass

            def handle_error(self, request, client_address):
                """Suppress socket error tracebacks."""
                pass

        try:
            self._server = ThreadingHTTPServer((self.host, self.port), SSEHandler)
            self._running = True
            self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
            self._thread.start()
            return self.port
        except OSError as e:
            print(f"[DSS SSE] Failed to start on port {self.port}: {e}")
            self._running = False
            return -1

    def stop(self):
        """Stop the SSE server."""
        self._running = False
        if self._server:
            self._server.shutdown()
            self._server = None
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

    @property
    def is_running(self) -> bool:
        return self._running and self._thread is not None and self._thread.is_alive()

    @property
    def status_url(self) -> str:
        return f"http://{self.host}:{self.port}/sse/dss-status"

    @property
    def state_url(self) -> str:
        return f"http://{self.host}:{self.port}/sse/dss-state"


# Singleton instance
_sse_server: SSEServer | None = None


def get_sse_server(host: str = "127.0.0.1", port: int = 7861) -> SSEServer:
    """Get or create the singleton SSE server."""
    global _sse_server
    if _sse_server is None or not _sse_server.is_running:
        _sse_server = SSEServer(host=host, port=port)
    return _sse_server


def start_sse_server(host: str = "127.0.0.1", port: int = 7861) -> int:
    """Start the SSE server. Returns the port used, or -1 on failure."""
    server = get_sse_server(host, port)
    return server.start()


def stop_sse_server():
    """Stop the SSE server."""
    global _sse_server
    if _sse_server:
        _sse_server.stop()
        _sse_server = None
