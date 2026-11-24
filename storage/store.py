"""SecondSelf — Content-Addressed Immutable Raw Storage Store."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import get_settings
from storage.exceptions import IntegrityError, RecordNotFoundError
from storage.manifest import Manifest
from storage.models import ManifestEntry, RawRecord, RecordMetadata
from storage.serializer import compute_sha256, deserialize_record, serialize_record


class RawStore:
    """Append-only, content-addressed storage repository with tamper verification.

    Guarantees:
    - Immutability: Once written, records are never mutated in place.
    - Idempotency & Deduplication: Identical payloads yield identical hashes and deduplicate.
    - Atomicity: File writes use temporary files + atomic rename (os.replace).
    - Auditability: Every write is permanently logged in an append-only manifest.jsonl.
    """

    def __init__(self, root_dir: Path | str | None = None) -> None:
        if root_dir is not None:
            self.root_dir = Path(root_dir).resolve()
        else:
            self.root_dir = get_settings().paths.raw_dir.resolve()

        self.objects_dir = self.root_dir / "objects"
        self.manifest_file = self.root_dir / "manifest.jsonl"
        self.manifest = Manifest(self.manifest_file)

        # Ensure base directories exist
        self.objects_dir.mkdir(parents=True, exist_ok=True)

    def _get_rel_path_for_hash(self, content_hash: str) -> Path:
        """Derives two-level sharded relative path: objects/3f/3f8a...json."""
        prefix = content_hash[:2]
        return Path("objects") / prefix / f"{content_hash}.json"

    def _get_abs_path_for_hash(self, content_hash: str) -> Path:
        """Returns full absolute path for a content hash."""
        return self.root_dir / self._get_rel_path_for_hash(content_hash)

    def exists(self, content_hash_or_id: str) -> bool:
        """Checks if a record with the specified hash or ID exists in the store."""
        target = content_hash_or_id.strip().lower()
        if len(target) == 64:
            if self._get_abs_path_for_hash(target).exists():
                return True
        return self.manifest.find_by_id(content_hash_or_id) is not None

    def write(
        self,
        content: str | bytes,
        *,
        source_type: str = "text",
        source_uri: str | None = None,
        content_type: str = "text/plain",
        adapter_name: str | None = None,
        record_id: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> tuple[RawRecord, bool]:
        """Atomically stores raw content with integrity hash and sidecar metadata.

        If the content already exists, it idempotently returns the existing record
        without creating a duplicate (deduplication).

        Args:
            content: Raw text string or byte sequence.
            source_type: Ingestion origin type ('text', 'url', 'file', 'stdin').
            source_uri: Origin URL or local file path.
            content_type: MIME type of the payload.
            adapter_name: Extraction adapter used.
            record_id: Optional explicit unique record ID (defaults to content_hash).
            extra: Additional sidecar metadata.

        Returns:
            A tuple of (RawRecord, is_new: bool).
        """
        if isinstance(content, bytes):
            payload_str = content.decode("utf-8", errors="replace")
        elif isinstance(content, str):
            payload_str = content
        else:
            raise TypeError(f"Content must be str or bytes, got {type(content).__name__}")

        content_hash = compute_sha256(payload_str)
        target_path = self._get_abs_path_for_hash(content_hash)
        rel_path = self._get_rel_path_for_hash(content_hash)

        # 1. Deduplication: Check if identical content hash already exists
        if target_path.exists():
            existing_record = self.read(content_hash, verify_integrity=True)
            return existing_record, False

        # 2. Build metadata sidecar
        now_iso = datetime.now(timezone.utc).isoformat()
        resolved_id = record_id or content_hash
        payload_bytes = payload_str.encode("utf-8")

        metadata = RecordMetadata(
            schema_version="1.0.0",
            created_at=now_iso,
            source_type=source_type,
            source_uri=source_uri,
            content_type=content_type,
            char_count=len(payload_str),
            byte_size=len(payload_bytes),
            content_hash=content_hash,
            adapter_name=adapter_name,
            extra=extra or {},
        )

        record = RawRecord(
            record_id=resolved_id,
            payload=payload_str,
            metadata=metadata,
        )

        # 3. Atomic Write: Write to temp file in root_dir then atomic os.replace()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        serialized_json = serialize_record(record)

        temp_fd, temp_path_str = tempfile.mkstemp(
            prefix="tmp_raw_",
            suffix=".json",
            dir=str(self.root_dir),
        )
        try:
            with os.fdopen(temp_fd, "w", encoding="utf-8") as temp_file:
                temp_file.write(serialized_json)
                temp_file.flush()
                os.fsync(temp_file.fileno())

            # Atomic rename (guaranteed on same filesystem mount)
            os.replace(temp_path_str, str(target_path))
        except Exception:
            if os.path.exists(temp_path_str):
                os.remove(temp_path_str)
            raise

        # 4. Append to Manifest audit log
        manifest_entry = ManifestEntry(
            record_id=resolved_id,
            content_hash=content_hash,
            timestamp=now_iso,
            source_type=source_type,
            source_uri=source_uri,
            file_rel_path=str(rel_path).replace("\\", "/"),
            byte_size=len(payload_bytes),
            schema_version="1.0.0",
        )
        self.manifest.append(manifest_entry)

        return record, True

    def read(self, record_id_or_hash: str, *, verify_integrity: bool = True) -> RawRecord:
        """Reads a stored record and optionally verifies its cryptographic integrity.

        Args:
            record_id_or_hash: SHA-256 hash or unique record ID.
            verify_integrity: If True, recomputes the payload hash and raises
                              IntegrityError if tampering or mismatch is detected.

        Returns:
            The loaded RawRecord instance.

        Raises:
            RecordNotFoundError: If record file does not exist.
            IntegrityError: If stored payload does not match its SHA-256 hash.
        """
        target = record_id_or_hash.strip()
        target_path: Path | None = None

        # A: Direct hash lookup
        if len(target) == 64:
            direct_path = self._get_abs_path_for_hash(target.lower())
            if direct_path.exists():
                target_path = direct_path

        # B: Manifest lookup by ID or hash
        if target_path is None:
            entry = self.manifest.find_by_id(target) or self.manifest.find_by_hash(target)
            if entry:
                cand_path = self.root_dir / entry.file_rel_path
                if cand_path.exists():
                    target_path = cand_path

        if target_path is None or not target_path.exists():
            raise RecordNotFoundError(f"Record '{record_id_or_hash}' not found in storage.")

        raw_json = target_path.read_text(encoding="utf-8")
        record = deserialize_record(raw_json)

        # Cryptographic verification
        if verify_integrity:
            computed_hash = record.compute_hash()
            if computed_hash != record.metadata.content_hash:
                raise IntegrityError(
                    f"TAMPER DETECTED in record '{record_id_or_hash}'!\n"
                    f"  Computed payload hash: {computed_hash}\n"
                    f"  Expected metadata hash: {record.metadata.content_hash}"
                )

        return record

    def verify(self, record_id_or_hash: str) -> tuple[bool, str]:
        """Explicitly audits and verifies a record's cryptographic integrity.

        Returns:
            Tuple of (True, "OK").

        Raises:
            IntegrityError: If corrupted or tampered with.
            RecordNotFoundError: If missing.
        """
        record = self.read(record_id_or_hash, verify_integrity=True)
        return True, f"Verified OK (Hash: {record.content_hash})"

    def verify_all(self) -> dict[str, Any]:
        """Audits the entire store, verifying all records against manifest logs."""
        results: dict[str, Any] = {
            "total_manifest_entries": 0,
            "verified_valid": 0,
            "corrupted_records": [],
            "missing_files": [],
        }

        for entry in self.manifest.iter_entries():
            results["total_manifest_entries"] += 1
            file_path = self.root_dir / entry.file_rel_path
            if not file_path.exists():
                results["missing_files"].append(entry.record_id)
                continue

            try:
                self.read(entry.content_hash, verify_integrity=True)
                results["verified_valid"] += 1
            except IntegrityError as exc:
                results["corrupted_records"].append({"id": entry.record_id, "error": str(exc)})
            except Exception as exc:
                results["corrupted_records"].append({"id": entry.record_id, "error": str(exc)})

        return results

    def list_records(self) -> list[ManifestEntry]:
        """Returns all logged records from the manifest audit log."""
        return self.manifest.read_all()
