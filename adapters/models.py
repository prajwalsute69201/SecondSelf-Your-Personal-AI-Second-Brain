"""SecondSelf — Normalized Extracted Content Data Contract."""

from __future__ import annotations

import hashlib
from typing import Any
from pydantic import BaseModel, Field


class ExtractedContent(BaseModel):
    """Standardized output model returned by all extraction adapters."""

    text: str = Field(..., description="Clean, extracted plain text or markdown content")
    title: str = Field(default="Untitled", description="Derived or extracted title of the document")
    source_type: str = Field(..., description="Adapter identifier (e.g. 'text', 'markdown', 'html', 'pdf')")
    source_uri: str | None = Field(default=None, description="Original source URI, URL, or file path")
    char_count: int = Field(default=0, description="Total character count of the normalized text")
    word_count: int = Field(default=0, description="Total word count of the normalized text")
    content_hash: str = Field(default="", description="SHA-256 digest of normalized text")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Adapter-specific metadata")
    warnings: list[str] = Field(default_factory=list, description="Non-fatal warnings during extraction")

    def model_post_init(self, __context: Any) -> None:
        """Auto-calculate derived metrics if not explicitly provided."""
        if not self.char_count:
            self.char_count = len(self.text)
        if not self.word_count:
            self.word_count = len(self.text.split())
        if not self.content_hash and self.text:
            self.content_hash = hashlib.sha256(self.text.strip().encode("utf-8")).hexdigest()
