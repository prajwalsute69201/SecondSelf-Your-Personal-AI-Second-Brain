"""SecondSelf — Repository Completeness & Orphan Detection Validator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from storage.store import RawStore
from validation.base import BaseValidator
from validation.models import ValidationResult


class CompletenessValidator(BaseValidator):
    """Ensures manifest parity: zero orphaned objects and zero missing records."""

    name = "completeness_validator"

    def validate(self, target: Any, context: dict[str, Any] | None = None) -> ValidationResult:
        result = ValidationResult(validator_name=self.name)

        if not isinstance(target, RawStore):
            result.record_check(False, f"CompletenessValidator expects RawStore, received: {type(target).__name__}")
            return result

        manifest_entries = target.list_records()
        manifest_hashes = {e.content_hash.lower() for e in manifest_entries}
        result.details["total_manifest_entries"] = len(manifest_entries)

        # 1. Verify every manifest entry points to an existing file
        missing_files: list[str] = []
        for entry in manifest_entries:
            expected_file = target.root_dir / entry.file_rel_path
            if not expected_file.exists():
                missing_files.append(f"{entry.record_id} ({entry.file_rel_path})")

        result.record_check(
            len(missing_files) == 0,
            f"Missing record file(s) logged in manifest: {missing_files}",
        )

        # 2. Check for orphaned objects on disk not referenced in manifest
        orphan_files: list[str] = []
        if target.objects_dir.exists():
            for obj_file in target.objects_dir.glob("*/*.json"):
                file_hash = obj_file.stem.lower()
                if file_hash not in manifest_hashes:
                    orphan_files.append(str(obj_file.relative_to(target.root_dir)))

        result.record_check(
            len(orphan_files) == 0,
            f"Orphaned object file(s) found on disk without manifest log: {orphan_files}",
        )
        result.details["orphan_count"] = len(orphan_files)
        result.details["missing_count"] = len(missing_files)

        return result
