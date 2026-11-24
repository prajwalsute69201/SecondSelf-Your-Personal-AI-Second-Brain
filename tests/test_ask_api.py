"""SecondSelf — Vercel Serverless Function Unit & Contract Tests (Task 4.2).

Validates api/ask.py serverless handler:
- HTTP 200 OK with Answer schema on valid queries
- HTTP 400 Bad Request on missing/empty questions, invalid parameters, and malformed JSON
- HTTP 413 Payload Too Large on oversized bodies (> 8 KB)
- HTTP 405 Method Not Allowed on non-POST methods (GET, PUT, DELETE)
- HTTP 204 No Content with CORS preflight headers on OPTIONS
- HTTP 500 Internal Server Error with safe, sanitized error message on engine exceptions
"""

from __future__ import annotations

import http.client
import json
import threading
import time
from http.server import HTTPServer
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from api.ask import handler
from ask.models import Answer, Citation, RetrievedItem


class EphemeralTestServer:
    """Manages an ephemeral local HTTP server bound to 127.0.0.1:0 running handler."""

    def __init__(self) -> None:
        self.server = HTTPServer(("127.0.0.1", 0), handler)
        self.host, self.port = self.server.server_address
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> EphemeralTestServer:
        self.thread.start()
        time.sleep(0.02)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture(scope="module")
def api_server() -> EphemeralTestServer:
    """Fixture providing an ephemeral running HTTP server for the module."""
    with EphemeralTestServer() as srv:
        yield srv


def request_api(
    server: EphemeralTestServer,
    method: str,
    path: str = "/api/ask",
    body: Any = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], dict[str, Any] | None, str]:
    """Helper sending HTTP request to ephemeral server via direct http.client (proxy-safe)."""
    conn = http.client.HTTPConnection(server.host, server.port, timeout=10.0)
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)

    payload_bytes: bytes | None = None
    if body is not None:
        if isinstance(body, bytes):
            payload_bytes = body
        elif isinstance(body, str):
            payload_bytes = body.encode("utf-8")
        else:
            payload_bytes = json.dumps(body).encode("utf-8")

    if payload_bytes is not None and "Content-Length" not in req_headers:
        req_headers["Content-Length"] = str(len(payload_bytes))

    try:
        conn.request(method, path, body=payload_bytes, headers=req_headers)
        resp = conn.getresponse()
        resp_headers = {k.lower(): v for k, v in resp.getheaders()}
        raw_text = resp.read().decode("utf-8", errors="replace")
        try:
            parsed_json = json.loads(raw_text) if raw_text else None
        except Exception:
            parsed_json = None
        return resp.status, resp_headers, parsed_json, raw_text
    finally:
        conn.close()


# ------------------------------------------------------------------------------
# Test Scenarios
# ------------------------------------------------------------------------------

def test_valid_post_success(api_server: EphemeralTestServer) -> None:
    """Verifies that a valid POST returns 200 OK with the full Answer schema contract."""
    mock_answer = Answer(
        question="What is force layout?",
        text="Force-directed layout simulates nodes as charged particles with springs.",
        citations=[
            Citation(
                item_id="item:note-1",
                title="Force-directed Associative Layout",
                snippet="Graph layout using spring-electrical forces.",
                score=0.92,
                source="raw/20261123_test.json",
            )
        ],
        retrieved=[
            RetrievedItem(
                id="item:note-1",
                title="Force-directed Associative Layout",
                snippet="Graph layout using spring-electrical forces.",
                content="Full text of note-1",
                score=0.92,
                category="Resources",
            )
        ],
        mode="retrieval_only",
        model=None,
        prompt_version="1.0.0",
        latency_ms=15.4,
        is_grounded=True,
        uncited_claims_flag=False,
    )

    with patch("ask.cli.ask", return_value=mock_answer) as mock_ask:
        status, headers, data, _ = request_api(
            api_server,
            "POST",
            body={
                "question": "What is force layout?",
                "top_k": 3,
                "min_score": 0.35,
                "no_llm": True,
                "expand_graph": False,
            },
        )

        assert status == 200
        assert "application/json" in headers.get("content-type", "")
        assert headers.get("access-control-allow-origin") == "*"
        assert headers.get("x-content-type-options") == "nosniff"

        assert data is not None
        assert data["ok"] is True
        assert data["answer"] == mock_answer.text
        assert len(data["citations"]) == 1
        assert data["citations"][0]["item_id"] == "item:note-1"
        assert len(data["retrieved"]) == 1
        assert data["mode"] == "retrieval_only"
        assert data["model"] is None
        assert data["latency_ms"] == 15.4
        assert data["is_grounded"] is True
        assert data["uncited_claims_flag"] is False

        mock_ask.assert_called_once()
        _, kwargs = mock_ask.call_args
        assert kwargs["question"] == "What is force layout?"
        assert kwargs["top_k"] == 3
        assert kwargs["use_llm"] is False


def test_missing_question_returns_400(api_server: EphemeralTestServer) -> None:
    """Verifies that missing 'question' yields HTTP 400 Bad Request."""
    status, _, data, _ = request_api(api_server, "POST", body={"top_k": 5})
    assert status == 400
    assert data is not None
    assert data["ok"] is False
    assert "question" in data["error"].lower()


def test_empty_question_returns_400(api_server: EphemeralTestServer) -> None:
    """Verifies that empty/whitespace 'question' yields HTTP 400 Bad Request."""
    status, _, data, _ = request_api(api_server, "POST", body={"question": "   \n\t  "})
    assert status == 400
    assert data is not None
    assert data["ok"] is False
    assert "non-empty" in data["error"].lower()


def test_question_exceeding_length_returns_400(api_server: EphemeralTestServer) -> None:
    """Verifies that questions longer than 1000 characters yield HTTP 400 Bad Request."""
    status, _, data, _ = request_api(api_server, "POST", body={"question": "X" * 1001})
    assert status == 400
    assert data is not None
    assert data["ok"] is False
    assert "maximum limit" in data["error"].lower()


def test_invalid_json_body_returns_400(api_server: EphemeralTestServer) -> None:
    """Verifies that invalid or malformed JSON yields HTTP 400 Bad Request."""
    status, _, data, _ = request_api(
        api_server,
        "POST",
        body=b'{"question": "unfinished json payload...',
    )
    assert status == 400
    assert data is not None
    assert data["ok"] is False
    assert "malformed" in data["error"].lower()


def test_oversized_payload_returns_413(api_server: EphemeralTestServer) -> None:
    """Verifies that payloads larger than 8 KB yield HTTP 413 Payload Too Large."""
    huge_padding = "x" * 9000
    status, _, data, _ = request_api(
        api_server,
        "POST",
        body={"question": "What is life?", "padding": huge_padding},
    )
    assert status == 413
    assert data is not None
    assert data["ok"] is False
    assert "exceeds maximum limit" in data["error"].lower()


@pytest.mark.parametrize(
    "invalid_payload,error_fragment",
    [
        ({"question": "Q", "top_k": 0}, "top_k"),
        ({"question": "Q", "top_k": 25}, "top_k"),
        ({"question": "Q", "top_k": "5"}, "top_k"),
        ({"question": "Q", "min_score": -0.1}, "min_score"),
        ({"question": "Q", "min_score": 1.2}, "min_score"),
        ({"question": "Q", "no_llm": "yes"}, "no_llm"),
        ({"question": "Q", "expand_graph": 1}, "expand_graph"),
        ({"question": "Q", "max_context_chars": 200}, "max_context_chars"),
        ({"question": "Q", "max_context_chars": 50000}, "max_context_chars"),
    ],
)
def test_invalid_parameters_return_400(
    api_server: EphemeralTestServer,
    invalid_payload: dict[str, Any],
    error_fragment: str,
) -> None:
    """Verifies boundary checks on top_k, min_score, and flags yield HTTP 400."""
    status, _, data, _ = request_api(api_server, "POST", body=invalid_payload)
    assert status == 400
    assert data is not None
    assert data["ok"] is False
    assert error_fragment in data["error"].lower()


@pytest.mark.parametrize("method", ["GET", "PUT", "DELETE", "PATCH", "HEAD"])
def test_non_post_methods_return_405(api_server: EphemeralTestServer, method: str) -> None:
    """Verifies non-POST methods yield HTTP 405 Method Not Allowed with Allow header."""
    status, headers, data, _ = request_api(api_server, method)
    assert status == 405
    assert "allow" in headers
    assert "POST" in headers["allow"]
    if method != "HEAD":
        assert data is not None
        assert data["ok"] is False
        assert "method not allowed" in data["error"].lower()


def test_options_preflight_returns_204(api_server: EphemeralTestServer) -> None:
    """Verifies OPTIONS preflight returns HTTP 204 No Content with CORS headers."""
    status, headers, _, _ = request_api(
        api_server,
        "OPTIONS",
        headers={"Origin": "https://myapp.vercel.app"},
    )
    assert status == 204
    assert "access-control-allow-origin" in headers
    assert "POST, OPTIONS" in headers.get("access-control-allow-methods", "")
    assert "content-type" in headers.get("access-control-allow-headers", "").lower()


def test_internal_exception_returns_500_safe_error(api_server: EphemeralTestServer) -> None:
    """Verifies that internal unhandled exceptions yield HTTP 500 without leaking secrets."""
    with patch("ask.cli.ask", side_effect=RuntimeError("Internal Secret DB Password Error!")):
        status, _, data, raw_text = request_api(
            api_server,
            "POST",
            body={"question": "Trigger secret explosion"},
        )
        assert status == 500
        assert data is not None
        assert data["ok"] is False
        assert data["error"] == "Internal server error processing query"
        # Guarantee no stack trace or secret string leaked
        assert "Password" not in raw_text
        assert "RuntimeError" not in raw_text
