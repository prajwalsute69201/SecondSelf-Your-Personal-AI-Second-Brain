#!/usr/bin/env python3
"""SecondSelf — Vercel Serverless Function Handler (Task 4.2).

Exposes the SecondSelf retrieval-augmented synthesis engine (ask.py) as a stateless
serverless HTTP microservice for Vercel Python runtime:
- Wraps core ask() logic without reimplementing retrieval or synthesis.
- Enforces strict input validation and payload bounds (max 8 KB, question <= 1000 chars).
- Sets comprehensive CORS preflight (OPTIONS) and security response headers.
- Handles internal exceptions cleanly without exposing sensitive system internals.

Endpoint:
  POST /api/ask
  OPTIONS /api/ask
"""

from __future__ import annotations

import json
import logging
import os
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

# Ensure workspace root is in sys.path for absolute imports
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("secondself.api.ask")

# Constraints
MAX_PAYLOAD_BYTES = 8192      # 8 KB ceiling on incoming JSON body
MAX_QUESTION_LENGTH = 1000    # Maximum question character length
MIN_CONTEXT_CHARS = 500       # Minimum character budget
MAX_CONTEXT_CHARS = 32000     # Maximum character budget


def _resolve_cors_origin(incoming_origin: str | None) -> str:
    """Determines appropriate Access-Control-Allow-Origin header."""
    try:
        from config import get_settings
        allowed = get_settings().serverless.cors_origins
    except Exception:
        raw = os.getenv("CORS_ORIGINS", "*")
        allowed = [o.strip() for o in raw.split(",") if o.strip()]

    if "*" in allowed or not incoming_origin:
        return "*"
    if incoming_origin in allowed:
        return incoming_origin
    return allowed[0] if allowed else "*"


class handler(BaseHTTPRequestHandler):
    """Vercel-compatible BaseHTTPRequestHandler for question answering."""

    def _set_cors_headers(self) -> None:
        """Injects Cross-Origin Resource Sharing (CORS) headers."""
        origin = self.headers.get("Origin")
        allow_origin = _resolve_cors_origin(origin)
        self.send_header("Access-Control-Allow-Origin", allow_origin)
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With")
        self.send_header("Access-Control-Max-Age", "86400")

    def _set_security_headers(self) -> None:
        """Injects defensive HTTP security headers."""
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "strict-origin-when-cross-origin")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, proxy-revalidate")

    def _send_json(
        self,
        status_code: int,
        payload: dict[str, Any],
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        """Sends formatted JSON response with status code and standard headers."""
        try:
            body = json.dumps(payload, indent=None, ensure_ascii=False).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self._set_cors_headers()
            self._set_security_headers()
            if extra_headers:
                for k, v in extra_headers.items():
                    self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError) as exc:
            logger.warning("Client disconnected before response could be delivered: %s", exc)

    def log_message(self, format: str, *args: Any) -> None:
        """Routes HTTP access logs through Python standard logging."""
        client_ip = self.client_address[0] if self.client_address else "127.0.0.1"
        logger.info("%s - - [%s] %s", client_ip, self.log_date_time_string(), format % args)

    # --------------------------------------------------------------------------
    # CORS Preflight
    # --------------------------------------------------------------------------
    def do_OPTIONS(self) -> None:
        """Handles CORS preflight OPTIONS requests."""
        try:
            self.send_response(204)
            self._set_cors_headers()
            self._set_security_headers()
            self.end_headers()
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError) as exc:
            logger.warning("Client disconnected during OPTIONS preflight: %s", exc)

    # --------------------------------------------------------------------------
    # Method Not Allowed Handlers (GET, PUT, DELETE, etc.)
    # --------------------------------------------------------------------------
    def _handle_method_not_allowed(self) -> None:
        """Rejects non-POST requests with 405 Method Not Allowed."""
        self._send_json(
            status_code=405,
            payload={"ok": False, "error": "Method Not Allowed. Use POST."},
            extra_headers={"Allow": "POST, OPTIONS"},
        )

    def do_GET(self) -> None:
        self._handle_method_not_allowed()

    def do_PUT(self) -> None:
        self._handle_method_not_allowed()

    def do_DELETE(self) -> None:
        self._handle_method_not_allowed()

    def do_PATCH(self) -> None:
        self._handle_method_not_allowed()

    def do_HEAD(self) -> None:
        self._handle_method_not_allowed()

    # --------------------------------------------------------------------------
    # Main POST Handler
    # --------------------------------------------------------------------------
    def do_POST(self) -> None:
        """Processes query request, calls ask(), and returns structured answer."""
        # 1. Content-Length Validation
        content_length_raw = self.headers.get("Content-Length")
        if not content_length_raw:
            self._send_json(400, {"ok": False, "error": "Missing Content-Length header"})
            return

        try:
            content_length = int(content_length_raw)
            if content_length < 0:
                raise ValueError("Negative Content-Length")
        except (ValueError, TypeError):
            self._send_json(400, {"ok": False, "error": "Invalid Content-Length header"})
            return

        if content_length > MAX_PAYLOAD_BYTES:
            self._send_json(
                413,
                {
                    "ok": False,
                    "error": f"Payload size ({content_length} bytes) exceeds maximum limit of {MAX_PAYLOAD_BYTES} bytes",
                },
            )
            return

        # 2. Content-Type Validation
        content_type = self.headers.get("Content-Type", "")
        if content_type and "application/json" not in content_type.lower():
            self._send_json(400, {"ok": False, "error": "Content-Type must be 'application/json'"})
            return

        # 3. Read & Decode Body
        try:
            raw_body = self.rfile.read(content_length)
        except Exception as exc:
            logger.error("Failed reading request stream: %s", exc)
            self._send_json(400, {"ok": False, "error": "Failed to read request body"})
            return

        try:
            data = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"ok": False, "error": "Malformed JSON payload"})
            return

        if not isinstance(data, dict):
            self._send_json(400, {"ok": False, "error": "Request body must be a JSON object"})
            return

        # 4. Field Validation
        # 4a. Question
        question = data.get("question")
        if question is None:
            self._send_json(400, {"ok": False, "error": "Missing required field: 'question'"})
            return

        if not isinstance(question, str) or not question.strip():
            self._send_json(400, {"ok": False, "error": "Field 'question' must be a non-empty string"})
            return

        question_clean = question.strip()
        if len(question_clean) > MAX_QUESTION_LENGTH:
            self._send_json(
                400,
                {
                    "ok": False,
                    "error": f"Question length ({len(question_clean)}) exceeds maximum limit of {MAX_QUESTION_LENGTH} characters",
                },
            )
            return

        # 4b. Optional top_k (1 to 20)
        top_k = data.get("top_k")
        if top_k is not None:
            if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1 or top_k > 20:
                self._send_json(400, {"ok": False, "error": "Parameter 'top_k' must be an integer between 1 and 20"})
                return

        # 4c. Optional min_score (0.0 to 1.0)
        min_score = data.get("min_score")
        if min_score is not None:
            if (
                not isinstance(min_score, (int, float))
                or isinstance(min_score, bool)
                or not (0.0 <= float(min_score) <= 1.0)
            ):
                self._send_json(
                    400,
                    {"ok": False, "error": "Parameter 'min_score' must be a numeric float between 0.0 and 1.0"},
                )
                return
            min_score = float(min_score)

        # 4d. Optional booleans
        no_llm = data.get("no_llm", False)
        if not isinstance(no_llm, bool):
            self._send_json(400, {"ok": False, "error": "Parameter 'no_llm' must be a boolean"})
            return

        expand_graph = data.get("expand_graph", False)
        if not isinstance(expand_graph, bool):
            self._send_json(400, {"ok": False, "error": "Parameter 'expand_graph' must be a boolean"})
            return

        # 4e. Optional max_context_chars (500 to 32000)
        max_context_chars = data.get("max_context_chars")
        if max_context_chars is not None:
            if (
                not isinstance(max_context_chars, int)
                or isinstance(max_context_chars, bool)
                or max_context_chars < MIN_CONTEXT_CHARS
                or max_context_chars > MAX_CONTEXT_CHARS
            ):
                self._send_json(
                    400,
                    {
                        "ok": False,
                        "error": f"Parameter 'max_context_chars' must be an integer between {MIN_CONTEXT_CHARS} and {MAX_CONTEXT_CHARS}",
                    },
                )
                return

        # 5. Core Engine Execution
        try:
            # Lazy import to minimize cold-start penalty
            from ask.cli import ask

            # Resolve paths for precomputed index & graph artifacts
            index_path = REPO_ROOT / "public" / "index.json"
            graph_path = REPO_ROOT / "public" / "graph.json"

            answer = ask(
                question=question_clean,
                top_k=top_k,
                min_score=min_score,
                use_llm=not no_llm,
                expand_graph=expand_graph,
                max_context_chars=max_context_chars,
                index_file=index_path if index_path.exists() else None,
                graph_file=graph_path if graph_path.exists() else None,
            )

            response_payload = {
                "ok": True,
                "answer": answer.text,
                "citations": [c.to_dict() for c in answer.citations],
                "retrieved": [r.to_dict() for r in answer.retrieved],
                "mode": answer.mode,
                "model": answer.model,
                "prompt_version": answer.prompt_version,
                "latency_ms": round(answer.latency_ms, 2),
                "is_grounded": answer.is_grounded,
                "uncited_claims_flag": answer.uncited_claims_flag,
            }
            self._send_json(200, response_payload)

        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError) as exc:
            logger.warning("Client connection closed before response could be sent: %s", exc)
        except Exception as exc:
            logger.exception("Internal error executing question synthesis: %s", exc)
            self._send_json(
                500,
                {
                    "ok": False,
                    "error": "Internal server error processing query",
                },
            )
