"""Unit and integration tests for SecondSelf Immutable Storage Layer (storage/)."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from storage import (
    CorruptRecordError,
    IntegrityError,
    Manifest,
    ManifestEntry,
    RawRecord,
    RawStore,
    RecordMetadata,
    RecordNotFoundError,
    compute_sha256,
    deserialize_record,
    serialize_record,
)


def test_compute_sha256_deterministic() -> None:
    data = "SecondSelf content-addressed storage"
    h1 = compute_sha256(data)
    h2 = compute_sha256(data.encode("utf-8"))
    assert len(h1) == 64
    assert h1 == h2
    assert h1 != compute_sha256("Different text")


def test_deterministic_serialization() -> None:
    meta = RecordMetadata(
        created_at="2026-11-23T10:00:00Z",
        source_type="text",
        content_hash="abc123",
        extra={"b_key": 2, "a_key": 1},
    )
    rec = RawRecord(record_id="rec_1", payload="test payload", metadata=meta)
    json_1 = serialize_record(rec)
    json_2 = serialize_record(rec)
    assert json_1 == json_2
    # Verify sorted keys
    assert json_1.index('"a_key"') < json_1.index('"b_key"')


def test_write_and_read_roundtrip(tmp_path: Path) -> None:
    store = RawStore(root_dir=tmp_path)
    content = "Immutable associative memory note."

    record, is_new = store.write(
        content,
        source_type="text",
        source_uri="note://1",
        adapter_name="text",
        extra={"tags": ["ai", "memory"]},
    )

    assert is_new is True
    assert record.payload == content
    assert record.metadata.char_count == len(content)
    assert record.metadata.source_type == "text"
    assert record.metadata.source_uri == "note://1"
    assert record.metadata.extra["tags"] == ["ai", "memory"]

    # Read back by content_hash
    loaded = store.read(record.content_hash, verify_integrity=True)
    assert loaded.payload == content
    assert loaded.content_hash == record.content_hash
    assert loaded.is_valid() is True


def test_idempotent_deduplication(tmp_path: Path) -> None:
    store = RawStore(root_dir=tmp_path)
    content = "Identical content for deduplication test."

    record_1, is_new_1 = store.write(content, source_type="text")
    record_2, is_new_2 = store.write(content, source_type="text")

    assert is_new_1 is True
    assert is_new_2 is False
    assert record_1.content_hash == record_2.content_hash

    # Verify only one file and one manifest entry exist
    manifest_entries = store.list_records()
    assert len(manifest_entries) == 1
    assert manifest_entries[0].content_hash == record_1.content_hash


def test_tamper_detection_raises_integrity_error(tmp_path: Path) -> None:
    store = RawStore(root_dir=tmp_path)
    content = "Original authentic document text."

    record, _ = store.write(content, source_type="text")
    file_path = store._get_abs_path_for_hash(record.content_hash)
    assert file_path.exists()

    # Intentionally corrupt the stored record on disk (simulate tampering)
    raw_data = json.loads(file_path.read_text(encoding="utf-8"))
    key_to_tamper = "raw_content" if "raw_content" in raw_data else "payload"
    raw_data[key_to_tamper] = "Tampered fraudulent text inserted."
    file_path.write_text(json.dumps(raw_data), encoding="utf-8")

    # Read with verification should raise IntegrityError
    with pytest.raises(IntegrityError, match="TAMPER DETECTED"):
        store.read(record.content_hash, verify_integrity=True)

    # store.verify() should also raise IntegrityError
    with pytest.raises(IntegrityError):
        store.verify(record.content_hash)


def test_read_missing_record_raises_record_not_found(tmp_path: Path) -> None:
    store = RawStore(root_dir=tmp_path)
    with pytest.raises(RecordNotFoundError, match="not found in storage"):
        store.read("0000000000000000000000000000000000000000000000000000000000000000")


def test_manifest_audit_consistency(tmp_path: Path) -> None:
    store = RawStore(root_dir=tmp_path)
    doc_a = "Document Alpha"
    doc_b = "Document Beta"
    doc_c = "Document Gamma"

    rec_a, _ = store.write(doc_a, source_type="text")
    rec_b, _ = store.write(doc_b, source_type="url", source_uri="https://example.com/b")
    rec_c, _ = store.write(doc_c, source_type="file", source_uri="/tmp/c.txt")

    entries = store.list_records()
    assert len(entries) == 3
    hashes = {e.content_hash for e in entries}
    assert rec_a.content_hash in hashes
    assert rec_b.content_hash in hashes
    assert rec_c.content_hash in hashes

    # Verify audit results
    audit = store.verify_all()
    assert audit["total_manifest_entries"] == 3
    assert audit["verified_valid"] == 3
    assert len(audit["corrupted_records"]) == 0
    assert len(audit["missing_files"]) == 0
