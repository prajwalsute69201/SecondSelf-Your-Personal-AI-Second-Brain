"""SecondSelf — Immutable Raw Storage Layer Public API."""

from __future__ import annotations

from storage.exceptions import (
    CorruptRecordError,
    IntegrityError,
    RecordNotFoundError,
    StorageError,
)
from storage.manifest import Manifest
from storage.models import ManifestEntry, RawRecord, RecordMetadata
from storage.serializer import compute_sha256, deserialize_record, serialize_record
from storage.store import RawStore

__all__ = [
    "CorruptRecordError",
    "IntegrityError",
    "Manifest",
    "ManifestEntry",
    "RawRecord",
    "RawStore",
    "RecordMetadata",
    "RecordNotFoundError",
    "StorageError",
    "compute_sha256",
    "deserialize_record",
    "serialize_record",
]
