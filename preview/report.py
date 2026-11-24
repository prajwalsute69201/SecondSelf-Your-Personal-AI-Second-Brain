"""SecondSelf — Verification Report & Console Summary (Task 3.3).

Generates structured JSON evidence reports and formatted console summaries
representing the quality gate status of the static frontend before deployment.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class CheckResult:
    """Represents the outcome of an individual verification check."""
    name: str
    status: str  # "PASS", "FAIL", "WARN", "SKIP"
    description: str
    duration_ms: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.status == "PASS"


@dataclass
class VerificationReport:
    """Complete machine-readable and auditable Phase 3 quality gate report."""
    run_timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    overall_status: str = "FAIL"  # "PASS" or "FAIL"
    exit_code: int = 1  # 0: Pass, 1: Check failure, 2: Setup error
    duration_ms: float = 0.0
    server_url: str = ""
    environment: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    checks: dict[str, CheckResult] = field(default_factory=dict)
    evidence_files: list[str] = field(default_factory=list)
    contract_summary: dict[str, Any] = field(default_factory=dict)

    def add_check(self, check: CheckResult) -> None:
        """Appends a completed check and updates overall status."""
        self.checks[check.name] = check
        if check.status == "FAIL":
            self.overall_status = "FAIL"
            self.exit_code = 1

    def finalize(self, total_duration_ms: float) -> None:
        """Calculates final status and exit code across all checks."""
        self.duration_ms = round(total_duration_ms, 2)
        has_failures = any(c.status == "FAIL" for c in self.checks.values())
        if has_failures:
            self.overall_status = "FAIL"
            self.exit_code = 1
        else:
            self.overall_status = "PASS"
            self.exit_code = 0

    def to_dict(self) -> dict[str, Any]:
        """Serializes the report to a Python dictionary."""
        data = asdict(self)
        # Convert checks to nested dicts
        data["checks"] = {k: asdict(v) for k, v in self.checks.items()}
        return data

    def to_json(self, indent: int = 2) -> str:
        """Serializes report to deterministic JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, output_path: Path) -> Path:
        """Persists the JSON report to disk."""
        target = Path(output_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(), encoding="utf-8")
        if str(target) not in self.evidence_files:
            self.evidence_files.append(str(target))
        return target

    def render_console_summary(self) -> str:
        """Renders an attractive, readable console verification summary."""
        lines = [
            "",
            "=" * 78,
            f"  SECONDSELF STATIC FRONTEND VERIFICATION — {self.overall_status}",
            "=" * 78,
            f"  Run Timestamp : {self.run_timestamp}",
            f"  Server URL    : {self.server_url or 'N/A'}",
            f"  Duration      : {self.duration_ms:.1f} ms",
            f"  Exit Code     : {self.exit_code}",
            "-" * 78,
            "  VERIFICATION CHECKS:",
        ]

        for name, check in self.checks.items():
            badge = f"[{check.status}]"
            padding = " " * (8 - len(badge))
            lines.append(f"    {badge}{padding} {name:28} : {check.description} ({check.duration_ms:.1f}ms)")
            for err in check.errors:
                lines.append(f"             ERROR: {err}")
            for warn in check.warnings:
                lines.append(f"             WARN : {warn}")

        if self.contract_summary:
            lines.extend([
                "-" * 78,
                "  DATA CONTRACT SUMMARY:",
                f"    Total Nodes    : {self.contract_summary.get('total_nodes', 'N/A')}",
                f"    Total Edges    : {self.contract_summary.get('total_edges', 'N/A')}",
                f"    Categories     : {self.contract_summary.get('categories', {})}",
                f"    Missing Req    : {len(self.contract_summary.get('missing_required', []))}",
                f"    Missing Recom  : {len(self.contract_summary.get('missing_recommended', []))}",
                f"    Extra Fields   : {len(self.contract_summary.get('extra_fields', []))}",
            ])

        if self.evidence_files:
            lines.extend([
                "-" * 78,
                "  GENERATED EVIDENCE ARTIFACTS:",
            ])
            for ef in self.evidence_files:
                lines.append(f"    - {ef}")

        lines.extend([
            "=" * 78,
            f"  OVERALL RESULT: {self.overall_status} (exit code: {self.exit_code})",
            "=" * 78,
            "",
        ])
        return "\n".join(lines)
