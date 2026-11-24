"""SecondSelf — Deterministic Serialization & Hashing Utilities."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from storage.exceptions import CorruptRecordError
from storage.models import RawRecord


def compute_sha256(content: str | bytes) -> str:
    """Computes standard deterministic SHA-256 digest of string or bytes.

    Args:
        content: String or raw bytes to hash.

    Returns:
        Hexadecimal 64-character SHA-256 hash.
    """
    if isinstance(content, str):
        raw_bytes = content.encode("utf-8")
    elif isinstance(content, bytes):
        raw_bytes = content
    else:
        raise TypeError(f"Expected str or bytes, got {type(content).__name__}")

    return hashlib.sha256(raw_bytes).hexdigest()


def canonicalize_json(data: dict[str, Any], indent: int = 2) -> str:
    """Serializes a dictionary to deterministic, canonical JSON with sorted keys.

    Args:
        data: Dictionary data.
        indent: Indentation spaces.

    Returns:
        Formatted canonical JSON string.
    """
    return json.dumps(data, indent=indent, sort_keys=True, ensure_ascii=False)


def serialize_record(record: RawRecord) -> str:
    """Serializes a RawRecord to canonical JSON string format."""
    return canonicalize_json(record.model_dump(by_alias=True))


def deserialize_record(raw_json: str) -> RawRecord:
    """Parses and validates a canonical JSON string into a RawRecord.

    Raises:
        CorruptRecordError: If JSON syntax is invalid or schema fails validation.
    """
    try:
        data = json.loads(raw_json)
        return RawRecord.model_validate(data)
    except Exception as exc:
        raise CorruptRecordError(f"Failed to parse stored record: {exc}") from exc
