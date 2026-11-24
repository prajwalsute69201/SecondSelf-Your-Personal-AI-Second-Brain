"""SecondSelf — Quality Gate Verification Orchestrator."""

from __future__ import annotations

import argparse
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import get_settings
from storage.store import RawStore
from validation.acceptance import AcceptanceCriteriaValidator
from validation.base import BaseValidator, IntegrityFailure
from validation.completeness import CompletenessValidator
from validation.integrity import IntegrityValidator
from validation.models import VerificationReport, VerificationSummary
from validation.report import format_terminal_report
from validation.schema import SchemaValidator


class VerificationOrchestrator:
    """Orchestrates quality gate execution across all deliverable validators."""

    def __init__(self, validators: list[BaseValidator] | None = None) -> None:
        if validators is not None:
            self.validators = list(validators)
        else:
            self.validators = [
                SchemaValidator(),
                IntegrityValidator(),
                CompletenessValidator(),
                AcceptanceCriteriaValidator(),
            ]

    def add_validator(self, validator: BaseValidator) -> None:
        """Appends a new validator to the pipeline."""
        self.validators.append(validator)

    def run(
        self,
        target: Any,
        *,
        fail_fast: bool = False,
        min_records: int = 1,
        context: dict[str, Any] | None = None,
    ) -> VerificationReport:
        """Executes all registered quality gate validators against the target deliverable.

        Args:
            target: Deliverable to verify (typically RawStore or list of RawRecords).
            fail_fast: If True, halts execution on the first validator failure.
            min_records: Minimum records required by the acceptance criteria validator.
            context: Additional contextual flags or environment overrides.

        Returns:
            Consolidated, machine-readable VerificationReport.
        """
        run_id = str(uuid.uuid4())
        now_iso = datetime.now(timezone.utc).isoformat()
        ctx = {"fail_fast": fail_fast, "min_records": min_records, **(context or {})}

        results: dict[str, Any] = {}
        total_checks = 0
        failed_checks = 0
        warning_count = 0
        records_evaluated = 0

        if isinstance(target, RawStore):
            records_evaluated = target.manifest.count()
        elif isinstance(target, list):
            records_evaluated = len(target)

        has_integrity_failure = False

        for validator in self.validators:
            try:
                res = validator.validate(target, context=ctx)
            except IntegrityFailure as exc:
                has_integrity_failure = True
                res = validator.validate(target, context={**ctx, "fail_fast": False})
                res.passed = False
                res.errors.append(f"IntegrityFailure: {exc}")
                results[validator.name] = res
                total_checks += res.checks_run
                failed_checks += len(res.errors)
                if fail_fast:
                    break
                continue
            except Exception as exc:
                # Catch unexpected validator errors gracefully
                from validation.models import ValidationResult
                res = ValidationResult(validator_name=validator.name)
                res.record_check(False, f"Validator crash: {exc}")

            results[validator.name] = res
            total_checks += res.checks_run
            failed_checks += len(res.errors)
            warning_count += len(res.warnings)

            if not res.passed and validator.name == "integrity_validator":
                has_integrity_failure = True

            if not res.passed and fail_fast:
                break

        passed_checks = max(0, total_checks - failed_checks)

        # Determine overall status and standard CI exit codes:
        # Exit 0 = All Passed, Exit 1 = Validation Failure, Exit 2 = Integrity Compromised
        if has_integrity_failure:
            overall_status = "INTEGRITY_COMPROMISED"
            exit_code = 2
        elif failed_checks > 0:
            overall_status = "FAILED"
            exit_code = 1
        else:
            overall_status = "PASSED"
            exit_code = 0

        summary = VerificationSummary(
            total_checks=total_checks,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            warning_count=warning_count,
            total_records_evaluated=records_evaluated,
        )

        return VerificationReport(
            run_id=run_id,
            timestamp=now_iso,
            phase="Phase 1: The Archivist",
            overall_status=overall_status,
            exit_code=exit_code,
            summary=summary,
            results=results,
        )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for running the Phase 1 quality gate."""
    settings = get_settings()

    parser = argparse.ArgumentParser(
        prog="validate_phase1",
        description="SecondSelf Phase 1 Quality Gate & Verification Runner",
    )
    parser.add_argument(
        "--store-dir",
        type=Path,
        default=settings.paths.raw_dir,
        help=f"Target raw storage directory (default: {settings.paths.raw_dir})",
    )
    parser.add_argument(
        "--min-records",
        type=int,
        default=1,
        help="Minimum records required to pass acceptance criteria (default: 1)",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Halt validation on first failed check",
    )
    parser.add_argument(
        "--report-file",
        type=Path,
        default=None,
        help="Optional path to write the JSON verification report",
    )
    parser.add_argument(
        "--format",
        choices=["terminal", "json"],
        default="terminal",
        help="Output format: 'terminal' (human readable) or 'json' (machine readable)",
    )

    args = parser.parse_args(argv)
    store = RawStore(root_dir=args.store_dir)
    orchestrator = VerificationOrchestrator()

    report = orchestrator.run(
        store,
        fail_fast=args.fail_fast,
        min_records=args.min_records,
    )

    if args.report_file:
        saved_path = report.save(args.report_file)
        print(f"Verification report saved to: {saved_path}", file=sys.stderr)

    if args.format == "json":
        print(report.to_json(indent=2))
    else:
        print(format_terminal_report(report))

    return report.exit_code


if __name__ == "__main__":
    sys.exit(main())
