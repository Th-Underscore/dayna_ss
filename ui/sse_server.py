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
        """
        Initialize the SSEServer with network binding and an update queue.
        
        Parameters:
        	host (str): IP address to bind the server to (default "127.0.0.1").
        	port (int): TCP port to listen on (default 7861).
        	queue: Optional queue-like object supplying updates; if omitted, obtains the module default via `get_update_queue()`.
        
        Notes:
        	Initializes internal server and thread references to `None` and sets the running flag to `False`.
        """
        self.host = host
        self.port = port
        self._queue = queue or get_update_queue()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> int:
        """
        Start the SSE server and run it in a background daemon thread.
        
        If the server is already running, this returns the configured port without starting a new server.
        
        Returns:
            int: The port the server is listening on, or `-1` if startup failed.
        """
        if self._running:
            return self.port

        parent = self

        class SSEHandler(BaseHTTPRequestHandler):
            """HTTP handler with SSE streaming support."""

            def do_GET(self):
                """
                Route HTTP GET requests to the appropriate SSE or JSON handlers based on the request path.
                
                Dispatches:
                - /sse/dss-status -> _handle_sse()
                - /sse/dss-state  -> _handle_state()
                - /sse/dss-health -> _handle_health()
                
                Sends a 404 response for any other path. Suppresses client-disconnect and socket-related exceptions (e.g., BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError).
                """
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
                """
                Stream server-sent events (SSE) to the connected client.
                
                Sends any buffered events from the server queue to the client first, then continuously streams newly buffered events as they arrive. Periodically emits SSE comment heartbeats to keep the connection alive and stops streaming if the client disconnects or the server stops running.
                """
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
                """
                Send a single Server-Sent Events (SSE) data frame to the client.
                
                Writes an SSE `data:` frame for the given string and flushes the output stream.
                
                Returns:
                    True if the frame was written and flushed successfully, False if the connection is closed or a socket error occurred.
                """
                try:
                    message = f"data: {data}\n\n"
                    self.wfile.write(message.encode("utf-8"))
                    self.wfile.flush()
                    return True
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                    return False

            def _send_heartbeat(self) -> bool:
                """
                Send an SSE comment heartbeat to keep the client connection alive.
                
                Returns:
                    bool: `True` if the heartbeat was written and flushed successfully, `False` if the connection is closed or a socket error occurred.
                """
                try:
                    self.wfile.write(b": heartbeat\n\n")
                    self.wfile.flush()
                    return True
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                    return False

            def _handle_state(self):
                """
                Send a JSON snapshot of the current update queue state to the client.
                
                The queue state is serialized with json.dumps (using str() for non-serializable objects) and written with Content-Type `application/json`, a matching Content-Length, and an `Access-Control-Allow-Origin: *` header.
                """
                state = parent._queue.get_state()
                data = json.dumps(state, default=str).encode("utf-8")

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(data)

            def _handle_health(self):
                """
                Return a small JSON health payload for the SSE server.
                
                Responds with HTTP 200 and a JSON body `{"status": "ok", "port": <port>}` and sets `Content-Type`, `Content-Length`, and `Access-Control-Allow-Origin` headers.
                """
                data = json.dumps({"status": "ok", "port": parent.port}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, format, *args):
                """
                Override to suppress HTTP request logging emitted by BaseHTTPRequestHandler.
                
                This method intentionally does nothing so that request-level log messages are not written to stderr.
                """
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
        """
        Stop the SSE server and release its resources.
        
        Sets the running flag to False, shuts down the underlying HTTP server if present, and joins the background thread (waiting up to 2 seconds) before clearing internal references.
        """
        self._running = False
        if self._server:
            self._server.shutdown()
            self._server = None
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

    @property
    def is_running(self) -> bool:
        """
        Indicates whether the server is currently running and its background thread is alive.
        
        Returns:
            bool: `True` if the server's `_running` flag is set and the background thread exists and is alive, `False` otherwise.
        """
        return self._running and self._thread is not None and self._thread.is_alive()

    @property
    def status_url(self) -> str:
        """
        Builds the full URL for the server's SSE status endpoint.
        
        Returns:
            The HTTP URL pointing to the `/sse/dss-status` endpoint, including the server's host and port.
        """
        return f"http://{self.host}:{self.port}/sse/dss-status"

    @property
    def state_url(self) -> str:
        """
        Return the full HTTP URL for the server's state JSON endpoint.
        
        Returns:
            url (str): The URL to the `/sse/dss-state` endpoint using the server's configured host and port.
        """
        return f"http://{self.host}:{self.port}/sse/dss-state"


# Singleton instance
_sse_server: SSEServer | None = None


def get_sse_server(host: str = "127.0.0.1", port: int = 7861) -> SSEServer:
    """
    Return the module-level singleton SSEServer, creating a new instance if none exists or the existing one is not running.
    
    Returns:
        SSEServer: The singleton SSEServer instance configured with the provided host and port.
    """
    global _sse_server
    if _sse_server is None or not _sse_server.is_running:
        _sse_server = SSEServer(host=host, port=port)
    return _sse_server


def start_sse_server(host: str = "127.0.0.1", port: int = 7861) -> int:
    """
    Start and return the configured SSE server's listening port.
    
    Parameters:
        host (str): Host address to bind the server to (default "127.0.0.1").
        port (int): Preferred port to bind the server to (default 7861).
    
    Returns:
        int: The port the server is listening on, or `-1` if startup failed.
    """
    server = get_sse_server(host, port)
    return server.start()


def stop_sse_server():
    """Stop the SSE server."""
    global _sse_server
    if _sse_server:
        _sse_server.stop()
        _sse_server = None
