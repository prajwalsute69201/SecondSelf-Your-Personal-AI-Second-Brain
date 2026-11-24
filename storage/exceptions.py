"""SecondSelf — Storage Exceptions."""

from __future__ import annotations


class StorageError(Exception):
    """Base exception for all storage and persistence operations."""


class RecordNotFoundError(StorageError, FileNotFoundError):
    """Raised when a requested record hash or ID does not exist in the store."""


class IntegrityError(StorageError, ValueError):
    """Raised when data tampering, corruption, or hash mismatch is detected."""


class CorruptRecordError(StorageError, ValueError):
    """Raised when stored JSON data is unparseable or violates schema constraints."""
