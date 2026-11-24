"""SecondSelf — Static Preview Server Lifecycle Manager (Task 3.3).

Provides a robust, zero-orphan static HTTP server with readiness health-checking,
automatic port discovery/fallback, clean context management, and proxy-bypassing loopback client.
"""

from __future__ import annotations

import functools
import http.server
import logging
import os
import socket
import socketserver
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger("secondself.preview.server")


class UnifiedPreviewHandler(http.server.SimpleHTTPRequestHandler):
    """Unified HTTP handler serving static public/ assets while routing /api/ask to the serverless function."""

    def _set_cors_headers(self) -> None:
        from api.ask import handler as AskHandler
        AskHandler._set_cors_headers(self)

    def _set_security_headers(self) -> None:
        from api.ask import handler as AskHandler
        AskHandler._set_security_headers(self)

    def _send_json(self, status_code: int, payload: dict[str, Any], extra_headers: dict[str, str] | None = None) -> None:
        from api.ask import handler as AskHandler
        AskHandler._send_json(self, status_code, payload, extra_headers)

    def _handle_method_not_allowed(self) -> None:
        from api.ask import handler as AskHandler
        AskHandler._handle_method_not_allowed(self)

    def do_OPTIONS(self) -> None:
        if self.path.startswith("/api/"):
            from api.ask import handler as AskHandler
            AskHandler.do_OPTIONS(self)
        else:
            self.send_response(204)
            self.end_headers()

    def do_POST(self) -> None:
        if self.path.startswith("/api/"):
            from api.ask import handler as AskHandler
            AskHandler.do_POST(self)
        else:
            self.send_error(405, "Method Not Allowed")

    def do_GET(self) -> None:
        if self.path.startswith("/api/"):
            from api.ask import handler as AskHandler
            AskHandler.do_GET(self)
        else:
            super().do_GET()

    def log_message(self, format: str, *args: Any) -> None:
        """Direct standard HTTP access logs to debug level rather than stdout."""
        logger.debug("%s - - [%s] %s", self.client_address[0] if self.client_address else "127.0.0.1", self.log_date_time_string(), format % args)


class ServerReadinessTimeout(Exception):
    """Raised when the preview server fails to respond within the readiness timeout ceiling."""


class StaticPreviewServer:
    """Manages the background execution, health checking, and clean termination of a static server."""

    def __init__(
        self,
        directory: Path,
        host: str = "127.0.0.1",
        port: int = 8000,
        readiness_timeout_seconds: float = 5.0,
        poll_interval_seconds: float = 0.05,
    ) -> None:
        self.directory = Path(directory).resolve()
        self.host = host
        self.requested_port = port
        self.port = port
        self.readiness_timeout = readiness_timeout_seconds
        self.poll_interval = poll_interval_seconds

        self._server: socketserver.TCPServer | None = None
        self._thread: threading.Thread | None = None
        self._is_running = False

    @property
    def url(self) -> str:
        """Returns the base HTTP URL of the server."""
        return f"http://{self.host}:{self.port}"

    @property
    def is_running(self) -> bool:
        """Returns True if the server thread is active."""
        return self._is_running and self._thread is not None and self._thread.is_alive()

    def _find_available_port(self, start_port: int, max_attempts: int = 50) -> int:
        """Finds an open TCP port starting from start_port."""
        for p in range(start_port, start_port + max_attempts):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    sock.bind((self.host, p))
                    return p
                except OSError:
                    continue
        raise OSError(f"Could not find an available port on {self.host} in range {start_port}..{start_port + max_attempts}")

    def start(self) -> str:
        """Starts the static file server in a background thread and waits for readiness.

        Returns:
            The healthy server URL.
        """
        if self._is_running:
            return self.url

        if not self.directory.exists() or not self.directory.is_dir():
            raise FileNotFoundError(f"Static directory to serve does not exist: {self.directory}")

        # Ensure NO_PROXY is set for local loopback so system proxies do not intercept health checks
        for key in ("NO_PROXY", "no_proxy"):
            current = os.environ.get(key, "")
            if "127.0.0.1" not in current:
                os.environ[key] = f"{current},127.0.0.1,localhost".strip(",")

        # Bind TCPServer with port fallback
        handler_factory = functools.partial(UnifiedPreviewHandler, directory=str(self.directory))

        try:
            self._server = socketserver.TCPServer((self.host, self.requested_port), handler_factory)
            self.port = self.requested_port
        except OSError as err:
            logger.warning("Port %d unavailable (%s), searching for next free port...", self.requested_port, err)
            self.port = self._find_available_port(self.requested_port + 1)
            self._server = socketserver.TCPServer((self.host, self.port), handler_factory)

        self._server.daemon_threads = True

        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name=f"SecondSelfPreviewServer-{self.port}",
            daemon=True,
        )
        self._thread.start()
        self._is_running = True

        logger.info("Started static server on %s serving %s", self.url, self.directory)

        # Health check readiness polling
        self.wait_until_ready()
        return self.url

    def wait_until_ready(self) -> None:
        """Polls server URL until healthy or timeout expires."""
        start_time = time.monotonic()
        test_url = f"{self.url}/index.html"
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

        attempts = 0
        while (time.monotonic() - start_time) < self.readiness_timeout:
            attempts += 1
            try:
                req = urllib.request.Request(test_url, method="GET")
                with opener.open(req, timeout=1.0) as resp:
                    if resp.status == 200:
                        logger.debug("Server ready on attempt %d (took %.3fs)", attempts, time.monotonic() - start_time)
                        return
            except (urllib.error.URLError, ConnectionError, OSError, TimeoutError):
                time.sleep(self.poll_interval)

        # Timed out
        self.stop()
        raise ServerReadinessTimeout(
            f"Preview server at {self.url} failed to respond within {self.readiness_timeout}s timeout ({attempts} attempts)."
        )

    def stop(self) -> None:
        """Cleanly shuts down the server, releases all sockets, and joins thread."""
        if self._server is not None:
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception as err:
                logger.debug("Error during server shutdown: %s", err)
            finally:
                self._server = None

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            self._thread = None

        self._is_running = False
        logger.info("Preview server on %s stopped cleanly.", self.url)

    def __enter__(self) -> StaticPreviewServer:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()
