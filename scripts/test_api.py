#!/usr/bin/env python3
"""SecondSelf — Local API Test Harness & Contract Verifier (Task 4.2).

Spins up an ephemeral HTTP server running api/ask.py (or targets a live Vercel URL),
exercises all contract conditions (success, CORS preflight, parameter validation,
size limits, and error codes), and validates responses against the schema.

Usage:
  python scripts/test_api.py
  python scripts/test_api.py --live https://your-deployment.vercel.app/api/ask
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import sys
import threading
import time
from http.server import HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# Ensure workspace root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Color formatting helpers for terminal
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


class EphemeralServerContext:
    """Manages an ephemeral in-process HTTP server binding api/ask.handler to port 0."""

    def __init__(self) -> None:
        from api.ask import handler

        self.server = HTTPServer(("127.0.0.1", 0), handler)
        self.host, self.port = self.server.server_address
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> EphemeralServerContext:
        self.thread.start()
        # Brief pause to ensure loop is active
        time.sleep(0.05)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.server.shutdown()
        self.server.server_close()


def send_http_request(
    host: str,
    port: int,
    method: str,
    path: str,
    body: bytes | str | None = None,
    headers: dict[str, str] | None = None,
    use_ssl: bool = False,
    timeout: float = 45.0,
) -> tuple[int, dict[str, str], bytes]:
    """Sends raw HTTP request using standard library http.client (bypasses system proxy)."""
    if use_ssl:
        conn = http.client.HTTPSConnection(host, port, timeout=timeout)
    else:
        conn = http.client.HTTPConnection(host, port, timeout=timeout)

    req_headers = headers.copy() if headers else {}
    req_body = body.encode("utf-8") if isinstance(body, str) else body

    if req_body is not None and "Content-Length" not in req_headers:
        req_headers["Content-Length"] = str(len(req_body))

    try:
        conn.request(method, path, body=req_body, headers=req_headers)
        resp = conn.getresponse()
        resp_headers = {k: v for k, v in resp.getheaders()}
        resp_body = resp.read()
        return resp.status, resp_headers, resp_body
    finally:
        conn.close()


def run_test_suite(target_url: str | None = None) -> bool:
    """Executes the test suite against either an ephemeral server or live URL."""
    print(f"\n{BOLD}{CYAN}{'='*72}{RESET}")
    print(f"{BOLD}{CYAN}  SECONDSELF API CONTRACT VERIFIER (Task 4.2){RESET}")
    print(f"{BOLD}{CYAN}{'='*72}{RESET}\n")

    if target_url:
        parsed = urlparse(target_url)
        use_ssl = parsed.scheme == "https"
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if use_ssl else 80)
        path = parsed.path or "/api/ask"
        print(f"Targeting LIVE endpoint: {BOLD}{target_url}{RESET} (host={host}, port={port})\n")
        return _execute_cases(host, port, path, use_ssl)
    else:
        print(f"Spawning EPHEMERAL in-process server for {BOLD}api/ask.py{RESET}...")
        with EphemeralServerContext() as ctx:
            print(f"Server ready on {BOLD}http://{ctx.host}:{ctx.port}/api/ask{RESET}\n")
            return _execute_cases(ctx.host, ctx.port, "/api/ask", False)


def _execute_cases(host: str, port: int, path: str, use_ssl: bool) -> bool:
    """Runs all test scenarios and outputs results."""
    tests_run = 0
    tests_passed = 0

    def assert_case(
        title: str,
        method: str,
        body: Any = None,
        headers: dict[str, str] | None = None,
        expected_status: int = 200,
        validator: Any = None,
    ) -> bool:
        nonlocal tests_run, tests_passed
        tests_run += 1
        print(f"[{tests_run:02d}] {title:<48} ... ", end="", flush=True)

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

        start_time = time.perf_counter()
        try:
            status, resp_headers, resp_bytes = send_http_request(
                host=host,
                port=port,
                method=method,
                path=path,
                body=payload_bytes,
                headers=req_headers,
                use_ssl=use_ssl,
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # 1. Status Code Check
            if status != expected_status:
                print(f"{RED}FAILED{RESET} (Expected HTTP {expected_status}, got {status})")
                print(f"     Response: {resp_bytes.decode('utf-8', errors='replace')[:200]}")
                return False

            # 2. JSON & Validator Check
            if status != 204:
                try:
                    data = json.loads(resp_bytes.decode("utf-8"))
                except Exception as exc:
                    print(f"{RED}FAILED{RESET} (Invalid JSON response: {exc})")
                    return False

                if validator:
                    valid, reason = validator(data, resp_headers)
                    if not valid:
                        print(f"{RED}FAILED{RESET} ({reason})")
                        return False

            print(f"{GREEN}PASSED{RESET} ({elapsed_ms:.1f}ms)")
            tests_passed += 1
            return True

        except Exception as exc:
            print(f"{RED}ERROR{RESET} ({exc})")
            return False

    # --------------------------------------------------------------------------
    # Test Cases
    # --------------------------------------------------------------------------

    # 1. OPTIONS Preflight
    def validate_options(_data: Any, headers: dict[str, str]) -> tuple[bool, str]:
        if "access-control-allow-methods" not in {k.lower(): v for k, v in headers.items()}:
            return False, "Missing Access-Control-Allow-Methods header"
        return True, ""

    assert_case(
        "OPTIONS CORS preflight request",
        method="OPTIONS",
        expected_status=204,
        validator=validate_options,
    )

    # 2. Valid POST (Retrieval-Only mode for speed and testability)
    def validate_valid_post(data: dict[str, Any], headers: dict[str, str]) -> tuple[bool, str]:
        if not data.get("ok"):
            return False, f"Response 'ok' is not True: {data}"
        if "answer" not in data or not data["answer"]:
            return False, "Missing 'answer' in response"
        if "citations" not in data or not isinstance(data["citations"], list):
            return False, "Missing or invalid 'citations' list"
        if "retrieved" not in data or not isinstance(data["retrieved"], list):
            return False, "Missing or invalid 'retrieved' list"
        if data.get("mode") != "retrieval_only":
            return False, f"Expected mode 'retrieval_only', got {data.get('mode')}"
        if "latency_ms" not in data:
            return False, "Missing latency_ms"
        return True, ""

    assert_case(
        "POST valid query (no_llm=true, top_k=3)",
        method="POST",
        body={"question": "What is vector linking?", "no_llm": True, "top_k": 3},
        expected_status=200,
        validator=validate_valid_post,
    )

    # 3. Missing 'question' field
    def validate_error_response(data: dict[str, Any], _h: Any) -> tuple[bool, str]:
        if data.get("ok") is not False:
            return False, "Expected ok=False"
        if "error" not in data or not data["error"]:
            return False, "Missing error message"
        return True, ""

    assert_case(
        "POST missing 'question' parameter",
        method="POST",
        body={"top_k": 5},
        expected_status=400,
        validator=validate_error_response,
    )

    # 4. Empty/whitespace question
    assert_case(
        "POST empty whitespace 'question'",
        method="POST",
        body={"question": "    "},
        expected_status=400,
        validator=validate_error_response,
    )

    # 5. Question exceeding length limit (> 1000 characters)
    assert_case(
        "POST question exceeding 1000 characters",
        method="POST",
        body={"question": "A" * 1005},
        expected_status=400,
        validator=validate_error_response,
    )

    # 6. Invalid top_k parameter
    assert_case(
        "POST invalid top_k out of bounds (top_k=99)",
        method="POST",
        body={"question": "Test query", "top_k": 99},
        expected_status=400,
        validator=validate_error_response,
    )

    # 7. Invalid min_score parameter
    assert_case(
        "POST invalid min_score out of bounds (min_score=1.5)",
        method="POST",
        body={"question": "Test query", "min_score": 1.5},
        expected_status=400,
        validator=validate_error_response,
    )

    # 8. Malformed JSON body
    assert_case(
        "POST malformed unparseable JSON",
        method="POST",
        body=b'{"question": "broken json...',
        expected_status=400,
        validator=validate_error_response,
    )

    # 9. Oversized payload (> 8 KB limit)
    assert_case(
        "POST oversized payload (> 8192 bytes)",
        method="POST",
        body={"question": "Huge payload", "padding": "x" * 9000},
        expected_status=413,
        validator=validate_error_response,
    )

    # 10. Non-POST method (GET)
    def validate_method_not_allowed(data: dict[str, Any], headers: dict[str, str]) -> tuple[bool, str]:
        headers_lower = {k.lower(): v for k, v in headers.items()}
        if "allow" not in headers_lower or "POST" not in headers_lower["allow"]:
            return False, "Missing or incorrect Allow header in 405 response"
        if data.get("ok") is not False:
            return False, "Expected ok=False"
        return True, ""

    assert_case(
        "GET request (Method Not Allowed)",
        method="GET",
        expected_status=405,
        validator=validate_method_not_allowed,
    )

    # 11. Non-POST method (PUT)
    assert_case(
        "PUT request (Method Not Allowed)",
        method="PUT",
        body={"question": "Test"},
        expected_status=405,
        validator=validate_method_not_allowed,
    )

    # 12. Non-POST method (DELETE)
    assert_case(
        "DELETE request (Method Not Allowed)",
        method="DELETE",
        expected_status=405,
        validator=validate_method_not_allowed,
    )

    # --------------------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------------------
    print(f"\n{'-'*72}")
    if tests_passed == tests_run:
        print(f"{BOLD}{GREEN}ALL {tests_passed}/{tests_run} API CONTRACT TESTS PASSED!{RESET}")
        return True
    else:
        print(f"{BOLD}{RED}{tests_passed}/{tests_run} TESTS PASSED ({tests_run - tests_passed} FAILED){RESET}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="SecondSelf Vercel Serverless API Test Suite")
    parser.add_argument(
        "--live",
        type=str,
        default=None,
        help="Optional live URL to test (e.g., https://secondself.vercel.app/api/ask)",
    )
    args = parser.parse_args()

    success = run_test_suite(target_url=args.live)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
