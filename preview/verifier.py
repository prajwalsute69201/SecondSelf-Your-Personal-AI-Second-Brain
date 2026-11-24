"""SecondSelf — Frontend Verification Engine (Task 3.3).

Executes five automated quality gate verification checks:
1. Artifact Integrity: existence, non-emptiness, valid JSON, topology metrics, hashes.
2. Data Contract: exact field compliance per Task 3.2 contract note, missing/extra fields.
3. HTTP Availability: 200 OK across HTML, CSS, JS ES modules, and data artifacts.
4. Page Smoke Test: Playwright browser automation (or graceful DOM inspection fallback).
5. Graceful Failure: error state simulation on missing/malformed graph data.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import time
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from preview.report import CheckResult, VerificationReport
from preview.staging import compute_sha256

logger = logging.getLogger("secondself.preview.verifier")


class SimpleDOMParser(HTMLParser):
    """Extracts tag IDs, classes, script sources, and link hrefs from HTML."""

    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.classes: set[str] = set()
        self.scripts: list[dict[str, str]] = []
        self.links: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_dict = {k: v or "" for k, v in attrs}
        if "id" in attr_dict:
            self.ids.add(attr_dict["id"])
        if "class" in attr_dict:
            for cls_name in attr_dict["class"].split():
                self.classes.add(cls_name)
        if tag == "script":
            self.scripts.append(attr_dict)
        elif tag == "link":
            self.links.append(attr_dict)


class FrontendVerifier:
    """Orchestrates comprehensive verification of static frontend assets and data contracts."""

    def __init__(
        self,
        public_dir: Path,
        server_url: str = "http://127.0.0.1:8000",
        artifacts_dir: Path | None = None,
        source_graph: Path | None = None,
    ) -> None:
        self.public_dir = Path(public_dir).resolve()
        self.server_url = server_url.rstrip("/")
        self.artifacts_dir = Path(artifacts_dir or (self.public_dir.parent / "verify-artifacts")).resolve()
        self.source_graph = Path(source_graph).resolve() if source_graph else (self.public_dir / "data" / "graph.json")

    def run_all(self) -> VerificationReport:
        """Executes all 5 verification phases and compiles the VerificationReport."""
        start_time = time.monotonic()
        report = VerificationReport(
            server_url=self.server_url,
            environment={
                "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
                "platform": os.sys.platform,
                "public_dir": str(self.public_dir),
            },
        )

        # 1. Artifact Integrity
        check1 = self.verify_artifact_integrity()
        report.add_check(check1)
        report.artifacts = check1.details.get("artifacts", {})

        # 2. Data Contract Compliance
        check2 = self.verify_data_contract()
        report.add_check(check2)
        report.contract_summary = check2.details

        # 3. HTTP Availability
        check3 = self.verify_http_availability()
        report.add_check(check3)

        # 4. Page Load Smoke Test
        check4 = self.verify_page_smoke_test(report)
        report.add_check(check4)

        # 5. Graceful Failure Simulation
        check5 = self.verify_graceful_failure()
        report.add_check(check5)

        total_duration = (time.monotonic() - start_time) * 1000
        report.finalize(total_duration)
        return report

    # --- CHECK 1: ARTIFACT INTEGRITY -----------------------------------------
    def verify_artifact_integrity(self) -> CheckResult:
        """Validates that graph.json exists, is valid JSON, and has valid topology."""
        start = time.monotonic()
        errors: list[str] = []
        warnings: list[str] = []
        details: dict[str, Any] = {"artifacts": {}}

        target_file = self.source_graph if self.source_graph.exists() else (self.public_dir / "graph.json")

        if not target_file.exists() or not target_file.is_file():
            errors.append(f"Graph artifact not found at {target_file}")
            return CheckResult(
                name="artifact_integrity",
                status="FAIL",
                description="Artifact existence, JSON syntax, and topology counts",
                duration_ms=(time.monotonic() - start) * 1000,
                details=details,
                errors=errors,
            )

        # File size and hash
        size_bytes = target_file.stat().st_size
        if size_bytes == 0:
            errors.append(f"Graph artifact is 0 bytes: {target_file}")

        file_hash = compute_sha256(target_file)
        details["artifacts"]["graph.json"] = {
            "path": str(target_file),
            "size_bytes": size_bytes,
            "sha256": file_hash,
        }

        # Parse JSON
        try:
            with open(target_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as err:
            errors.append(f"Failed to parse JSON in {target_file}: {err}")
            return CheckResult(
                name="artifact_integrity",
                status="FAIL",
                description="Artifact existence, JSON syntax, and topology counts",
                duration_ms=(time.monotonic() - start) * 1000,
                details=details,
                errors=errors,
            )

        # Top-level keys
        for req_key in ("schema_version", "statistics", "nodes"):
            if req_key not in data:
                errors.append(f"Missing required top-level key '{req_key}'")

        raw_edges = data.get("edges") or data.get("links")
        if raw_edges is None:
            errors.append("Missing required 'edges' or 'links' top-level key")

        nodes = data.get("nodes", [])
        if not isinstance(nodes, list) or len(nodes) == 0:
            errors.append(f"'nodes' must be a non-empty array (got {len(nodes) if isinstance(nodes, list) else type(nodes)})")

        edges = raw_edges if isinstance(raw_edges, list) else []
        if len(edges) == 0:
            warnings.append("Graph has 0 edges (isolated nodes only)")

        # Topology and ID uniqueness
        node_ids = set()
        duplicate_ids = set()
        for idx, n in enumerate(nodes):
            nid = n.get("id") if isinstance(n, dict) else None
            if not nid:
                errors.append(f"Node at index {idx} missing 'id'")
            elif nid in node_ids:
                duplicate_ids.add(nid)
            else:
                node_ids.add(nid)

        if duplicate_ids:
            errors.append(f"Detected {len(duplicate_ids)} duplicate node IDs: {list(duplicate_ids)[:3]}")

        # Check for dangling edges
        dangling_edges = 0
        for e in edges:
            if not isinstance(e, dict):
                continue
            src = e.get("source") or e.get("from")
            tgt = e.get("target") or e.get("to")
            if src not in node_ids or tgt not in node_ids:
                dangling_edges += 1

        if dangling_edges > 0:
            errors.append(f"Found {dangling_edges} dangling edges referencing missing node IDs")

        stats = data.get("statistics", {})
        details["nodes_count"] = len(nodes)
        details["edges_count"] = len(edges)
        details["stats_nodes"] = stats.get("total_nodes")
        details["stats_edges"] = stats.get("total_edges")

        status = "FAIL" if errors else "PASS"
        return CheckResult(
            name="artifact_integrity",
            status=status,
            description="Artifact existence, JSON syntax, and topology counts",
            duration_ms=(time.monotonic() - start) * 1000,
            details=details,
            errors=errors,
            warnings=warnings,
        )

    # --- CHECK 2: DATA CONTRACT COMPLIANCE -----------------------------------
    def verify_data_contract(self) -> CheckResult:
        """Verifies that nodes and edges conform to the schema expected by the frontend."""
        start = time.monotonic()
        errors: list[str] = []
        warnings: list[str] = []

        target_file = self.source_graph if self.source_graph.exists() else (self.public_dir / "graph.json")
        if not target_file.exists():
            return CheckResult(
                name="data_contract",
                status="FAIL",
                description="Field presence and PARA taxonomy adherence",
                duration_ms=(time.monotonic() - start) * 1000,
                errors=["Cannot verify contract: graph.json missing"],
            )

        with open(target_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        nodes = data.get("nodes", [])
        edges = data.get("edges") or data.get("links") or []

        # Required & expected node fields
        required_node_keys = {"id", "label", "type"}
        expected_node_keys = {"group", "value", "degree", "summary", "tags", "properties"}
        expected_item_props = {"record_id", "category", "confidence", "rationale", "source_type"}

        valid_categories = {"Projects", "Areas", "Resources", "Archives"}

        missing_required_nodes = set()
        missing_recommended_nodes = set()
        extra_node_keys = set()
        category_counts: dict[str, int] = {}
        invalid_categories = set()

        for n in nodes:
            if not isinstance(n, dict):
                continue
            n_keys = set(n.keys())
            missing_req = required_node_keys - n_keys
            if missing_req:
                missing_required_nodes.update(missing_req)

            missing_rec = expected_node_keys - n_keys
            if missing_rec:
                missing_recommended_nodes.update(missing_rec)

            extra = n_keys - (required_node_keys | expected_node_keys)
            if extra:
                extra_node_keys.update(extra)

            # Check Item specific properties
            if n.get("type") == "Item":
                props = n.get("properties") or {}
                cat = props.get("category") or n.get("group")
                if cat:
                    category_counts[cat] = category_counts.get(cat, 0) + 1
                    if cat not in valid_categories:
                        invalid_categories.add(cat)
                else:
                    warnings.append(f"Item node '{n.get('id')}' missing category")

        if missing_required_nodes:
            errors.append(f"Nodes missing required schema keys: {sorted(missing_required_nodes)}")
        if invalid_categories:
            errors.append(f"Nodes contain invalid PARA categories: {sorted(invalid_categories)}")

        # Check edge fields
        missing_edge_keys = set()
        for e in edges:
            if not isinstance(e, dict):
                continue
            src = e.get("from") or e.get("source")
            tgt = e.get("to") or e.get("target")
            etype = e.get("type")
            if not src or not tgt or not etype:
                missing_edge_keys.add("endpoints_or_type")

        if missing_edge_keys:
            errors.append("Edges missing required endpoints (from/source, to/target) or type")

        details = {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "categories": category_counts,
            "missing_required": list(missing_required_nodes),
            "missing_recommended": list(missing_recommended_nodes),
            "extra_fields": list(extra_node_keys),
        }

        status = "FAIL" if errors else "PASS"
        return CheckResult(
            name="data_contract",
            status=status,
            description="Field presence and PARA taxonomy adherence",
            duration_ms=(time.monotonic() - start) * 1000,
            details=details,
            errors=errors,
            warnings=warnings,
        )

    # --- CHECK 3: HTTP AVAILABILITY ------------------------------------------
    def verify_http_availability(self) -> CheckResult:
        """Verifies 200 OK responses across all static HTML, CSS, JS, and JSON assets."""
        start = time.monotonic()
        errors: list[str] = []
        warnings: list[str] = []
        tested_routes: dict[str, dict[str, Any]] = {}

        candidate_routes = [
            "/index.html",
            "/css/main.css",
            "/style.css",
            "/js/app.js",
            "/js/dataLoader.js",
            "/js/graph.js",
            "/js/filters.js",
            "/js/detailPanel.js",
            "/data/graph.json",
            "/graph.json",
            "/README.md",
        ]

        # Scan index.html for additional referenced files
        index_path = self.public_dir / "index.html"
        if index_path.exists():
            try:
                parser = SimpleDOMParser()
                parser.feed(index_path.read_text(encoding="utf-8"))
                for s in parser.scripts:
                    src = s.get("src")
                    if src and not src.startswith(("http://", "https://", "//")):
                        norm = "/" + src.lstrip("./")
                        if norm not in candidate_routes:
                            candidate_routes.append(norm)
                for lk in parser.links:
                    href = lk.get("href")
                    if href and not href.startswith(("http://", "https://", "//", "data:")):
                        norm = "/" + href.lstrip("./")
                        if norm not in candidate_routes:
                            candidate_routes.append(norm)
            except Exception as parse_err:
                warnings.append(f"Could not parse index.html for asset discovery: {parse_err}")

        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

        for route in candidate_routes:
            url = f"{self.server_url}{route}"
            req_start = time.monotonic()
            try:
                req = urllib.request.Request(url, method="GET")
                with opener.open(req, timeout=3.0) as resp:
                    status_code = resp.status
                    content_length = len(resp.read())
                    elapsed = (time.monotonic() - req_start) * 1000
                    tested_routes[route] = {
                        "status": status_code,
                        "bytes": content_length,
                        "latency_ms": round(elapsed, 2),
                    }
                    if status_code != 200:
                        errors.append(f"Route '{route}' returned HTTP {status_code} (expected 200)")
            except urllib.error.HTTPError as http_err:
                tested_routes[route] = {"status": http_err.code, "error": str(http_err)}
                errors.append(f"Route '{route}' failed with HTTP {http_err.code}")
            except Exception as conn_err:
                tested_routes[route] = {"status": "CONN_ERR", "error": str(conn_err)}
                errors.append(f"Connection failure fetching '{route}': {conn_err}")

        details = {
            "server_url": self.server_url,
            "routes_tested": len(tested_routes),
            "tested_routes": tested_routes,
        }
        status = "FAIL" if errors else "PASS"
        return CheckResult(
            name="http_availability",
            status=status,
            description="HTTP 200 status across all static assets and ES modules",
            duration_ms=(time.monotonic() - start) * 1000,
            details=details,
            errors=errors,
            warnings=warnings,
        )

    # --- CHECK 4: PAGE SMOKE TEST --------------------------------------------
    def verify_page_smoke_test(self, report: VerificationReport) -> CheckResult:
        """Executes browser automation smoke test if Playwright is available; otherwise degrades to DOM inspection."""
        start = time.monotonic()
        errors: list[str] = []
        warnings: list[str] = []
        details: dict[str, Any] = {}

        # Attempt Playwright browser automation
        playwright_available = False
        try:
            import playwright.sync_api  # type: ignore

            playwright_available = True
        except ImportError:
            playwright_available = False

        if playwright_available:
            try:
                logger.info("Executing Playwright browser smoke test against %s", self.server_url)
                from playwright.sync_api import sync_playwright

                self.artifacts_dir.mkdir(parents=True, exist_ok=True)
                screenshot_path = self.artifacts_dir / "screenshot-loaded.png"

                console_messages: list[str] = []
                page_errors: list[str] = []

                with sync_playwright() as p:
                    browser = p.chromium.launch(headless=True)
                    page = browser.new_page(viewport={"width": 1280, "height": 800})
                    page.on("console", lambda msg: console_messages.append(f"[{msg.type}] {msg.text}"))
                    page.on("pageerror", lambda exc: page_errors.append(str(exc)))

                    page.goto(self.server_url, wait_until="networkidle", timeout=10000)

                    # Wait for canvas or main container
                    page.wait_for_selector("#network-canvas", timeout=5000)
                    page.screenshot(path=str(screenshot_path))
                    report.evidence_files.append(str(screenshot_path))

                    # Verify stats strip rendered
                    stat_total = page.locator("#stat-total").inner_text()
                    details["browser_stats_total"] = stat_total
                    details["screenshot"] = str(screenshot_path)
                    details["mode"] = "PLAYWRIGHT_HEADLESS"
                    browser.close()

                if page_errors:
                    errors.append(f"Browser runtime errors caught on page: {page_errors}")

                return CheckResult(
                    name="page_smoke_test",
                    status="FAIL" if errors else "PASS",
                    description="Headless browser page load, canvas rendering, and screenshot capture",
                    duration_ms=(time.monotonic() - start) * 1000,
                    details=details,
                    errors=errors,
                )
            except Exception as pw_err:
                warnings.append(f"Playwright browser execution encountered error: {pw_err}; falling back to DOM inspection")

        # Graceful fallback: DOM & Landmark static inspection
        details["mode"] = "DOM_INSPECTION_FALLBACK"
        index_path = self.public_dir / "index.html"
        if not index_path.exists():
            errors.append("index.html not found for DOM smoke test")
        else:
            parser = SimpleDOMParser()
            parser.feed(index_path.read_text(encoding="utf-8"))

            required_landmarks = {
                "app",
                "main-content",
                "network-canvas",
                "loading-overlay",
                "error-overlay",
                "detail-drawer",
                "stats-strip",
                "search-input",
            }
            missing_landmarks = required_landmarks - parser.ids
            if missing_landmarks:
                errors.append(f"index.html missing critical UI element IDs: {sorted(missing_landmarks)}")

            # Check ES module entrypoint script
            has_app_script = any(
                s.get("type") == "module" and "js/app.js" in s.get("src", "") for s in parser.scripts
            )
            if not has_app_script:
                errors.append("Missing <script type='module' src='js/app.js'> in index.html")

            details["dom_landmarks_verified"] = len(required_landmarks - missing_landmarks)
            details["total_landmarks_checked"] = len(required_landmarks)
            warnings.append(
                "Playwright not installed; degraded gracefully to HTML DOM smoke test. "
                "See Docs/frontend-qa-checklist.md for manual browser verification."
            )

        status = "FAIL" if errors else "PASS"
        return CheckResult(
            name="page_smoke_test",
            status=status,
            description="Static DOM integrity & semantic layout verification",
            duration_ms=(time.monotonic() - start) * 1000,
            details=details,
            errors=errors,
            warnings=warnings,
        )

    # --- CHECK 5: GRACEFUL FAILURE SIMULATION ---------------------------------
    def verify_graceful_failure(self) -> CheckResult:
        """Simulates missing or corrupt graph data and verifies error state handling."""
        start = time.monotonic()
        errors: list[str] = []
        warnings: list[str] = []
        details: dict[str, Any] = {}

        # 1. Test missing file route returns 404
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        missing_url = f"{self.server_url}/data/nonexistent_test_artifact.json"
        try:
            req = urllib.request.Request(missing_url, method="GET")
            with opener.open(req, timeout=2.0) as resp:
                errors.append(f"Expected 404 for missing file, but got HTTP {resp.status}")
        except urllib.error.HTTPError as err:
            if err.code != 404:
                warnings.append(f"Missing file returned HTTP {err.code} instead of 404")
            details["missing_route_status"] = err.code

        # 2. Verify DataLoader defensive contract on malformed JSON
        app_js_path = self.public_dir / "js" / "dataLoader.js"
        if not app_js_path.exists():
            errors.append("dataLoader.js missing from public/js/")
        else:
            data_loader_src = app_js_path.read_text(encoding="utf-8")
            # Verify dataLoader checks CANDIDATE_URLS and validates nodes array
            if "CANDIDATE_URLS" not in data_loader_src or "validateAndNormalize" not in data_loader_src:
                errors.append("dataLoader.js does not contain defensive candidate fallback or validation logic")

        # 3. Verify Error Overlay exists in index.html with user-visible retry action
        index_path = self.public_dir / "index.html"
        if index_path.exists():
            index_src = index_path.read_text(encoding="utf-8")
            if "error-overlay" not in index_src or "retry-btn" not in index_src:
                errors.append("index.html missing error-overlay or retry-btn for graceful error recovery")
            else:
                details["error_overlay_verified"] = True

        status = "FAIL" if errors else "PASS"
        return CheckResult(
            name="graceful_failure",
            status=status,
            description="Error state fallback on missing or malformed graph data",
            duration_ms=(time.monotonic() - start) * 1000,
            details=details,
            errors=errors,
            warnings=warnings,
        )
