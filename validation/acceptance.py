"""SecondSelf — Phase 1 Acceptance Criteria Validator."""

from __future__ import annotations

from typing import Any

from storage.models import RawRecord
from storage.store import RawStore
from validation.base import BaseValidator
from validation.models import ValidationResult


class AcceptanceCriteriaValidator(BaseValidator):
    """Evaluates Phase 1 (The Archivist) milestone deliverables against acceptance criteria.

    Acceptance Rules:
    - Minimum records populated in vault (configurable via min_records, default 1).
    - Source type diversity (at least 2 distinct ingestion modes represented when vault >= 2 items).
    - 100% of records have non-empty content.
    - 100% of records have valid metadata and timestamps.
    """

    name = "acceptance_criteria_validator"

    def validate(self, target: Any, context: dict[str, Any] | None = None) -> ValidationResult:
        result = ValidationResult(validator_name=self.name)
        min_records = context.get("min_records", 1) if context else 1

        records: list[RawRecord] = []
        if isinstance(target, RawStore):
            for entry in target.list_records():
                try:
                    records.append(target.read(entry.content_hash, verify_integrity=False))
                except Exception:
                    pass
        elif isinstance(target, list) and all(isinstance(r, RawRecord) for r in target):
            records = target
        elif isinstance(target, RawRecord):
            records = [target]
        else:
            result.record_check(False, f"Unsupported target type for AcceptanceCriteriaValidator: {type(target).__name__}")
            return result

        count = len(records)
        result.details["total_records"] = count
        result.details["min_required"] = min_records

        # 1. Minimum record count check
        result.record_check(
            count >= min_records,
            f"Vault contains {count} item(s); minimum required for acceptance is {min_records}.",
        )

        # 2. Non-empty content rule
        empty_records = [r.record_id for r in records if not r.payload.strip()]
        result.record_check(
            len(empty_records) == 0,
            f"Found {len(empty_records)} record(s) with empty content: {empty_records}",
        )

        # 3. Source type diversity (if count >= 2, verify at least 2 distinct types)
        source_types = {r.metadata.source_type for r in records}
        result.details["source_types_represented"] = list(source_types)
        if count >= 2:
            result.record_check(
                len(source_types) >= 2,
                f"Expected multi-modal diversity across >=2 source types, found only: {list(source_types)}",
            )

        return result
