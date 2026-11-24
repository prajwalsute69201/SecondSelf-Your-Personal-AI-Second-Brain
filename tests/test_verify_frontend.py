"""SecondSelf — Frontend Verification & Preview Test Suite (Task 3.3).

Verifies the reliability, clean process lifecycle, data contract validation,
and failure detection mechanisms of the static frontend preview and verification harness.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from preview.report import CheckResult, VerificationReport
from preview.server import ServerReadinessTimeout, StaticPreviewServer
from preview.staging import ArtifactStagingManager, StagingError, compute_sha256
from preview.verifier import FrontendVerifier


@pytest.fixture
def sample_graph_data() -> dict:
    """Fixture providing a minimal compliant graph payload."""
    return {
        "schema_version": "1.0.0",
        "projection_version": "1.0.0",
        "generated_at": "2026-11-23T12:00:00Z",
        "statistics": {
            "total_nodes": 2,
            "total_edges": 1,
            "node_counts_by_type": {"Item": 1, "PARA_Category": 1},
            "edge_counts_by_type": {"BELONGS_TO": 1},
        },
        "nodes": [
            {
                "id": "item:note-1",
                "label": "Test Note 1",
                "type": "Item",
                "group": "Projects",
                "value": 14.0,
                "degree": 1,
                "summary": "Sample summary text",
                "tags": ["testing"],
                "properties": {
                    "record_id": "note-1",
                    "title": "Test Note 1",
                    "category": "Projects",
                    "confidence": 0.85,
                    "rationale": "Active effort with goal.",
                    "source_type": "markdown",
                    "created_at": "2026-11-23T10:00:00Z",
                },
            },
            {
                "id": "para:projects",
                "label": "Projects",
                "type": "PARA_Category",
                "group": "Projects",
                "value": 28.0,
                "degree": 1,
                "summary": "Projects taxonomy hub",
                "tags": [],
                "properties": {"category": "Projects"},
            },
        ],
        "edges": [
            {
                "from": "item:note-1",
                "to": "para:projects",
                "source": "item:note-1",
                "target": "para:projects",
                "type": "BELONGS_TO",
                "value": 1.0,
                "title": "BELONGS_TO",
                "properties": {"category": "Projects"},
            }
        ],
    }


def test_artifact_staging_copies_safely(tmp_path: Path, sample_graph_data: dict) -> None:
    """Verifies that ArtifactStagingManager safely stages artifacts without mutating source."""
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_graph = source_dir / "graph.json"
    source_graph.write_text(json.dumps(sample_graph_data), encoding="utf-8")
    original_hash = compute_sha256(source_graph)

    public_dir = tmp_path / "public"
    public_dir.mkdir()

    staging = ArtifactStagingManager(
        source_graph=source_graph,
        public_dir=public_dir,
    )
    manifest = staging.stage()

    assert "graph.json" in manifest
    staged_file = public_dir / "data" / "graph.json"
    assert staged_file.is_file()
    assert manifest["graph.json"]["sha256"] == original_hash
    # Source file remains untouched
    assert compute_sha256(source_graph) == original_hash


def test_staging_missing_source_raises_staging_error(tmp_path: Path) -> None:
    """Verifies that pointing to a nonexistent source raises StagingError."""
    staging = ArtifactStagingManager(
        source_graph=tmp_path / "nonexistent.json",
        public_dir=tmp_path / "public",
    )
    with pytest.raises(StagingError, match="Source graph artifact not found"):
        staging.stage()


def test_static_preview_server_lifecycle_and_clean_shutdown(tmp_path: Path) -> None:
    """Verifies that the static preview server starts, responds, and terminates cleanly."""
    pub_dir = tmp_path / "pub"
    pub_dir.mkdir()
    (pub_dir / "index.html").write_text("<!DOCTYPE html><html><body>Test</body></html>", encoding="utf-8")

    server = StaticPreviewServer(
        directory=pub_dir,
        host="127.0.0.1",
        port=8988,
        readiness_timeout_seconds=3.0,
    )

    with server:
        assert server.is_running
        assert server.port == 8988
        assert "http://127.0.0.1:8988" in server.url

    # Server must be stopped outside context
    assert not server.is_running


def test_verifier_passes_on_real_corpus() -> None:
    """Verifies that the entire real public/ corpus passes all 5 quality gate checks."""
    public_dir = Path("public")
    assert public_dir.is_dir()

    with StaticPreviewServer(directory=public_dir, host="127.0.0.1", port=8987) as server:
        verifier = FrontendVerifier(
            public_dir=public_dir,
            server_url=server.url,
            source_graph=public_dir / "data" / "graph.json",
        )
        report: VerificationReport = verifier.run_all()

        assert report.overall_status == "PASS"
        assert report.exit_code == 0
        assert len(report.checks) == 5
        assert report.checks["artifact_integrity"].status == "PASS"
        assert report.checks["data_contract"].status == "PASS"
        assert report.checks["http_availability"].status == "PASS"
        assert report.checks["page_smoke_test"].status == "PASS"
        assert report.checks["graceful_failure"].status == "PASS"

        # Check console summary renders without error
        summary = report.render_console_summary()
        assert "SECONDSELF STATIC FRONTEND VERIFICATION — PASS" in summary
        assert "[PASS]" in summary


def test_verifier_catches_missing_artifact(tmp_path: Path) -> None:
    """Verifies that missing graph.json fails the integrity check."""
    fake_public = tmp_path / "public"
    fake_public.mkdir()
    (fake_public / "index.html").write_text("<html></html>", encoding="utf-8")

    verifier = FrontendVerifier(
        public_dir=fake_public,
        server_url="http://127.0.0.1:8987",
        source_graph=fake_public / "data" / "nonexistent.json",
    )
    result = verifier.verify_artifact_integrity()
    assert result.status == "FAIL"
    assert any("not found" in err.lower() for err in result.errors)


def test_verifier_catches_malformed_json(tmp_path: Path) -> None:
    """Verifies that corrupt JSON fails the integrity check."""
    fake_public = tmp_path / "public"
    fake_public.mkdir()
    corrupt_file = fake_public / "corrupt.json"
    corrupt_file.write_text("{ this is malformed json : not valid", encoding="utf-8")

    verifier = FrontendVerifier(
        public_dir=fake_public,
        server_url="http://127.0.0.1:8987",
        source_graph=corrupt_file,
    )
    result = verifier.verify_artifact_integrity()
    assert result.status == "FAIL"
    assert any("Failed to parse JSON" in err for err in result.errors)


def test_verifier_catches_missing_required_contract_field(tmp_path: Path, sample_graph_data: dict) -> None:
    """Verifies that nodes missing required schema fields fail data contract verification."""
    # Delete 'label' from the item node
    del sample_graph_data["nodes"][0]["label"]

    graph_file = tmp_path / "graph.json"
    graph_file.write_text(json.dumps(sample_graph_data), encoding="utf-8")

    verifier = FrontendVerifier(
        public_dir=tmp_path,
        server_url="http://127.0.0.1:8987",
        source_graph=graph_file,
    )
    result = verifier.verify_data_contract()
    assert result.status == "FAIL"
    assert any("missing required schema keys" in err.lower() for err in result.errors)


def test_verifier_catches_invalid_para_category(tmp_path: Path, sample_graph_data: dict) -> None:
    """Verifies that an unknown PARA category fails data contract verification."""
    sample_graph_data["nodes"][0]["properties"]["category"] = "InvalidCustomCategory"

    graph_file = tmp_path / "graph.json"
    graph_file.write_text(json.dumps(sample_graph_data), encoding="utf-8")

    verifier = FrontendVerifier(
        public_dir=tmp_path,
        server_url="http://127.0.0.1:8987",
        source_graph=graph_file,
    )
    result = verifier.verify_data_contract()
    assert result.status == "FAIL"
    assert any("invalid PARA categories" in err for err in result.errors)


def test_verification_report_serialization_and_save(tmp_path: Path) -> None:
    """Verifies JSON serialization and round-trip fidelity of VerificationReport."""
    report = VerificationReport(
        overall_status="PASS",
        exit_code=0,
        server_url="http://127.0.0.1:8000",
    )
    report.add_check(
        CheckResult(
            name="test_check",
            status="PASS",
            description="Sample check description",
            duration_ms=12.5,
        )
    )
    report.finalize(12.5)

    report_path = tmp_path / "verify-report.json"
    saved = report.save(report_path)
    assert saved.is_file()

    loaded = json.loads(report_path.read_text(encoding="utf-8"))
    assert loaded["overall_status"] == "PASS"
    assert loaded["exit_code"] == 0
    assert "test_check" in loaded["checks"]
    assert loaded["checks"]["test_check"]["status"] == "PASS"
