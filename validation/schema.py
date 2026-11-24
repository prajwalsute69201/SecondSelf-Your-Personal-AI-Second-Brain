"""SecondSelf — Schema Conformance Validator."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from storage.models import RawRecord
from storage.store import RawStore
from validation.base import BaseValidator
from validation.models import ValidationResult

VALID_SOURCE_TYPES = {"text", "url", "file", "stdin"}
SUPPORTED_SCHEMA_VERSIONS = {"1.0.0"}


class SchemaValidator(BaseValidator):
    """Verifies that records and metadata adhere strictly to data contracts."""

    name = "schema_validator"

    def validate(self, target: Any, context: dict[str, Any] | None = None) -> ValidationResult:
        result = ValidationResult(validator_name=self.name)
        records: list[RawRecord] = []

        if isinstance(target, RawStore):
            for entry in target.list_records():
                try:
                    rec = target.read(entry.content_hash, verify_integrity=False)
                    records.append(rec)
                except Exception as exc:
                    result.record_check(False, f"Cannot deserialize record '{entry.record_id}': {exc}")
        elif isinstance(target, RawRecord):
            records = [target]
        elif isinstance(target, list) and all(isinstance(r, RawRecord) for r in target):
            records = target
        else:
            result.record_check(False, f"Unsupported target type for SchemaValidator: {type(target).__name__}")
            return result

        result.details["records_evaluated"] = len(records)

        for rec in records:
            # 1. Non-empty record ID
            result.record_check(bool(rec.record_id.strip()), f"Record {rec.record_id}: record_id cannot be blank.")

            # 2. Supported schema version
            result.record_check(
                rec.metadata.schema_version in SUPPORTED_SCHEMA_VERSIONS,
                f"Record {rec.record_id}: Unsupported schema_version '{rec.metadata.schema_version}'. Expected: {SUPPORTED_SCHEMA_VERSIONS}",
            )

            # 3. Recognized source type
            result.record_check(
                rec.metadata.source_type in VALID_SOURCE_TYPES,
                f"Record {rec.record_id}: Unknown source_type '{rec.metadata.source_type}'. Allowed: {VALID_SOURCE_TYPES}",
            )

            # 4. Valid ISO-8601 UTC timestamp
            try:
                datetime.fromisoformat(rec.metadata.created_at)
                result.record_check(True)
            except Exception:
                result.record_check(False, f"Record {rec.record_id}: Invalid ISO-8601 timestamp '{rec.metadata.created_at}'.")

            # 5. SHA-256 Hex Format (64 chars lowercase)
            is_valid_hash = bool(re.match(r"^[0-9a-f]{64}$", rec.metadata.content_hash.lower()))
            result.record_check(is_valid_hash, f"Record {rec.record_id}: Malformed SHA-256 content_hash '{rec.metadata.content_hash}'.")

            # 6. Character and byte counts match payload
            expected_chars = len(rec.payload)
            expected_bytes = len(rec.payload.encode("utf-8"))
            result.record_check(
                rec.metadata.char_count == expected_chars,
                f"Record {rec.record_id}: char_count mismatch (metadata: {rec.metadata.char_count}, actual: {expected_chars}).",
            )
            result.record_check(
                rec.metadata.byte_size == expected_bytes,
                f"Record {rec.record_id}: byte_size mismatch (metadata: {rec.metadata.byte_size}, actual: {expected_bytes}).",
            )

        return result
