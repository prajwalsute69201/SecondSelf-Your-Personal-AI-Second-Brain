"""SecondSelf — Data Validation & Verification Models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal
from pydantic import BaseModel, Field


class ValidationResult(BaseModel):
    """Encapsulates the outcome of a single validator run."""

    validator_name: str = Field(..., description="Name of the executing validator")
    passed: bool = Field(default=True, description="Whether all checks succeeded")
    checks_run: int = Field(default=0, description="Total number of discrete checks performed")
    errors: list[str] = Field(default_factory=list, description="Descriptions of failed assertions")
    warnings: list[str] = Field(default_factory=list, description="Non-blocking diagnostic warnings")
    details: dict[str, Any] = Field(default_factory=dict, description="Diagnostic data or counts")

    def record_check(self, success: bool, error_message: str | None = None) -> None:
        """Records a single boolean check, updating counters and errors."""
        self.checks_run += 1
        if not success:
            self.passed = False
            if error_message:
                self.errors.append(error_message)

    def add_warning(self, message: str) -> None:
        """Adds a non-fatal warning."""
        self.warnings.append(message)


class VerificationSummary(BaseModel):
    """High-level statistical overview of the verification pipeline."""

    total_checks: int = Field(default=0, description="Sum of all checks executed across validators")
    passed_checks: int = Field(default=0, description="Number of passing checks")
    failed_checks: int = Field(default=0, description="Number of failing checks")
    warning_count: int = Field(default=0, description="Number of warnings generated")
    total_records_evaluated: int = Field(default=0, description="Total raw store records evaluated")


class VerificationReport(BaseModel):
    """Machine-readable Phase 1 quality gate verification report."""

    run_id: str = Field(..., description="Unique UUIDv4 execution ID")
    timestamp: str = Field(..., description="ISO-8601 UTC timestamp of report generation")
    phase: str = Field(default="Phase 1: The Archivist", description="Verification scope")
    overall_status: Literal["PASSED", "FAILED", "INTEGRITY_COMPROMISED"] = Field(
        default="PASSED",
        description="Top-level pipeline gate status",
    )
    exit_code: int = Field(
        default=0,
        description="CLI/CI exit code: 0=Pass, 1=Validation Failure, 2=Integrity Compromised",
    )
    summary: VerificationSummary = Field(default_factory=VerificationSummary)
    results: dict[str, ValidationResult] = Field(default_factory=dict)

    def to_json(self, indent: int = 2) -> str:
        """Returns deterministic, canonical JSON with sorted keys."""
        return json.dumps(self.model_dump(), indent=indent, sort_keys=True, ensure_ascii=False)

    def save(self, output_path: Path | str) -> Path:
        """Persists the machine-readable verification report to disk."""
        target = Path(output_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(indent=2), encoding="utf-8")
        return target
