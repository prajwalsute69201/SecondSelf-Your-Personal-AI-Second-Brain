"""SecondSelf — Cryptographic Integrity Validator."""

from __future__ import annotations

from typing import Any

from storage.exceptions import IntegrityError
from storage.models import RawRecord
from storage.serializer import compute_sha256
from storage.store import RawStore
from validation.base import BaseValidator, IntegrityFailure
from validation.models import ValidationResult


class IntegrityValidator(BaseValidator):
    """Verifies cryptographic tamper-evidence and SHA-256 hash fidelity."""

    name = "integrity_validator"

    def validate(self, target: Any, context: dict[str, Any] | None = None) -> ValidationResult:
        result = ValidationResult(validator_name=self.name)
        fail_fast = bool(context and context.get("fail_fast", False))

        if isinstance(target, RawRecord):
            records = [target]
        elif isinstance(target, RawStore):
            try:
                audit = target.verify_all()
                result.details = audit
                result.record_check(
                    len(audit["corrupted_records"]) == 0,
                    f"Integrity check failed: {len(audit['corrupted_records'])} corrupted record(s) detected: {audit['corrupted_records']}",
                )
                if audit["corrupted_records"] and fail_fast:
                    raise IntegrityFailure(f"Immediate fail-fast on corrupted records: {audit['corrupted_records']}")
                return result
            except IntegrityError as exc:
                result.record_check(False, str(exc))
                if fail_fast:
                    raise IntegrityFailure(str(exc)) from exc
                return result
        elif isinstance(target, list) and all(isinstance(r, RawRecord) for r in target):
            records = target
        else:
            result.record_check(False, f"Unsupported target type for IntegrityValidator: {type(target).__name__}")
            return result

        result.details["records_checked"] = len(records)

        for rec in records:
            computed_hash = compute_sha256(rec.payload)
            expected_hash = rec.metadata.content_hash
            is_intact = computed_hash == expected_hash

            result.record_check(
                is_intact,
                f"TAMPER DETECTED in record '{rec.record_id}': Computed hash {computed_hash} != metadata hash {expected_hash}.",
            )
            if not is_intact and fail_fast:
                raise IntegrityFailure(f"Immediate fail-fast: Record '{rec.record_id}' has been tampered with or corrupted.")

        return result
