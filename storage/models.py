"""SecondSelf — Immutable Raw Storage Data Contracts."""

from __future__ import annotations

import hashlib
from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class RecordMetadata(BaseModel):
    """Immutable sidecar metadata describing the raw capture event."""

    schema_version: str = Field(default="1.0.0", description="Data contract schema version")
    created_at: str = Field(..., description="ISO-8601 UTC timestamp of write")
    source_type: str = Field(..., description="Ingestion mode ('text', 'url', 'file', 'stdin')")
    source_uri: str | None = Field(default=None, description="Original source URI or filepath")
    content_type: str = Field(default="text/plain", description="MIME content type")
    char_count: int = Field(default=0, description="Payload character count")
    byte_size: int = Field(default=0, description="Payload UTF-8 byte size")
    content_hash: str = Field(..., description="Deterministic SHA-256 digest of payload")
    adapter_name: str | None = Field(default=None, description="Extraction adapter used")
    extra: dict[str, Any] = Field(default_factory=dict, description="Custom diagnostic attributes")

    model_config = ConfigDict(populate_by_name=True)


class RawRecord(BaseModel):
    """Complete content-addressed raw record packaging payload and metadata."""

    record_id: str = Field(..., alias="id", description="Unique record identifier (defaults to content_hash)")
    payload: str = Field(..., alias="raw_content", description="The unmodified raw content text")
    metadata: RecordMetadata = Field(..., description="Immutable sidecar metadata")

    model_config = ConfigDict(populate_by_name=True)

    @property
    def id(self) -> str:
        return self.record_id

    @property
    def raw_content(self) -> str:
        return self.payload

    @property
    def content_hash(self) -> str:
        return self.metadata.content_hash

    def compute_hash(self) -> str:
        """Recomputes SHA-256 digest directly from the payload."""
        return hashlib.sha256(self.payload.encode("utf-8")).hexdigest()

    def is_valid(self) -> bool:
        """Returns True if the stored content hash matches the recomputed payload hash."""
        return self.compute_hash() == self.metadata.content_hash


class ManifestEntry(BaseModel):
    """Index entry stored in the append-only manifest.jsonl audit log."""

    record_id: str = Field(..., alias="id", description="Record unique ID")
    content_hash: str = Field(..., description="SHA-256 digest")
    timestamp: str = Field(..., description="ISO-8601 UTC timestamp")
    source_type: str = Field(..., description="Ingestion mode")
    source_uri: str | None = Field(default=None, description="Source URI or path")
    file_rel_path: str = Field(..., description="Relative storage path from raw root")
    byte_size: int = Field(default=0, description="Payload size in bytes")
    schema_version: str = Field(default="1.0.0", description="Record schema version")

    model_config = ConfigDict(populate_by_name=True)
