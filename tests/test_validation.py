"""Unit and integration tests for SecondSelf Validation & Quality Gates (validation/)."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from storage import RawStore
from validation import (
    AcceptanceCriteriaValidator,
    CompletenessValidator,
    IntegrityFailure,
    IntegrityValidator,
    SchemaValidator,
    ValidationResult,
    VerificationOrchestrator,
    VerificationReport,
)


@pytest.fixture
def populated_store(tmp_path: Path) -> RawStore:
    """Fixture providing a valid RawStore with 3 distinct items."""
    store = RawStore(root_dir=tmp_path)
    store.write("First text thought on graph associative memory.", source_type="text")
    store.write("Second note with a web URL origin.", source_type="url", source_uri="https://example.com/pkm")
    store.write("Third note from a local markdown document.", source_type="file", source_uri="/docs/plan.md")
    return store


# ------------------------------------------------------------------------------
# 1. SchemaValidator Tests
# ------------------------------------------------------------------------------

def test_schema_validator_valid_record(populated_store: RawStore) -> None:
    validator = SchemaValidator()
    record = populated_store.read(populated_store.list_records()[0].content_hash)
    result = validator.validate(record)
    assert result.passed is True
    assert result.checks_run >= 6
    assert len(result.errors) == 0


def test_schema_validator_invalid_source_type(populated_store: RawStore) -> None:
    validator = SchemaValidator()
    record = populated_store.read(populated_store.list_records()[0].content_hash)
    # Corrupt source_type
    record.metadata.source_type = "unsupported_device_stream"
    result = validator.validate(record)
    assert result.passed is False
    assert any("Unknown source_type" in err for err in result.errors)


# ------------------------------------------------------------------------------
# 2. IntegrityValidator Tests
# ------------------------------------------------------------------------------

def test_integrity_validator_clean(populated_store: RawStore) -> None:
    validator = IntegrityValidator()
    result = validator.validate(populated_store)
    assert result.passed is True
    assert len(result.errors) == 0


def test_integrity_validator_tampered_payload(populated_store: RawStore) -> None:
    validator = IntegrityValidator()
    record = populated_store.read(populated_store.list_records()[0].content_hash)
    # Intentionally corrupt the payload on the record object
    record.payload = "Tampered unauthorized text insertion."

    result = validator.validate(record)
    assert result.passed is False
    assert any("TAMPER DETECTED" in err for err in result.errors)


# ------------------------------------------------------------------------------
# 3. CompletenessValidator Tests
# ------------------------------------------------------------------------------

def test_completeness_validator_clean(populated_store: RawStore) -> None:
    validator = CompletenessValidator()
    result = validator.validate(populated_store)
    assert result.passed is True
    assert result.details["total_manifest_entries"] == 3
    assert result.details["orphan_count"] == 0
    assert result.details["missing_count"] == 0


def test_completeness_validator_missing_file(populated_store: RawStore) -> None:
    validator = CompletenessValidator()
    first_entry = populated_store.list_records()[0]
    target_file = populated_store.root_dir / first_entry.file_rel_path
    target_file.unlink()  # Delete real file to simulate missing asset

    result = validator.validate(populated_store)
    assert result.passed is False
    assert any("Missing record file" in err for err in result.errors)


def test_completeness_validator_orphaned_file(populated_store: RawStore) -> None:
    validator = CompletenessValidator()
    orphan = populated_store.objects_dir / "ff" / "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff.json"
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_text("{}", encoding="utf-8")

    result = validator.validate(populated_store)
    assert result.passed is False
    assert any("Orphaned object file" in err for err in result.errors)


# ------------------------------------------------------------------------------
# 4. AcceptanceCriteriaValidator Tests
# ------------------------------------------------------------------------------

def test_acceptance_criteria_validator_pass(populated_store: RawStore) -> None:
    validator = AcceptanceCriteriaValidator()
    result = validator.validate(populated_store, context={"min_records": 3})
    assert result.passed is True
    assert result.details["total_records"] == 3


def test_acceptance_criteria_validator_below_threshold(populated_store: RawStore) -> None:
    validator = AcceptanceCriteriaValidator()
    result = validator.validate(populated_store, context={"min_records": 10})
    assert result.passed is False
    assert any("minimum required for acceptance is 10" in err for err in result.errors)


# ------------------------------------------------------------------------------
# 5. Orchestrator & Report Tests
# ------------------------------------------------------------------------------

def test_orchestrator_all_pass(populated_store: RawStore) -> None:
    orchestrator = VerificationOrchestrator()
    report = orchestrator.run(populated_store, min_records=2)

    assert report.overall_status == "PASSED"
    assert report.exit_code == 0
    assert report.summary.failed_checks == 0
    assert report.summary.passed_checks > 0
    assert len(report.results) == 4


def test_orchestrator_integrity_compromised(populated_store: RawStore) -> None:
    # Corrupt a record file on disk
    first_entry = populated_store.list_records()[0]
    file_path = populated_store.root_dir / first_entry.file_rel_path
    raw_data = json.loads(file_path.read_text(encoding="utf-8"))
    raw_data["raw_content"] = "Corrupted text byte"
    file_path.write_text(json.dumps(raw_data), encoding="utf-8")

    orchestrator = VerificationOrchestrator()
    report = orchestrator.run(populated_store)

    assert report.overall_status == "INTEGRITY_COMPROMISED"
    assert report.exit_code == 2
    assert report.summary.failed_checks > 0


def test_orchestrator_deterministic_json_report(populated_store: RawStore, tmp_path: Path) -> None:
    orchestrator = VerificationOrchestrator()
    report = orchestrator.run(populated_store, min_records=1)

    json_output = report.to_json(indent=2)
    assert '"overall_status": "PASSED"' in json_output

    saved_file = tmp_path / "verification_report.json"
    report.save(saved_file)
    assert saved_file.exists()
    assert json.loads(saved_file.read_text(encoding="utf-8"))["run_id"] == report.run_id
