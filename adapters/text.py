"""SecondSelf — Plain Text Content Extractor."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from adapters.base import BaseExtractor, MalformedContentError
from adapters.models import ExtractedContent


class TextExtractor(BaseExtractor):
    """Adapter for raw thoughts, plain strings, and .txt files.

    Example:
        >>> extractor = TextExtractor()
        >>> content = extractor.extract("Fast associative memory idea")
        >>> content.title
        'Fast associative memory idea'
        >>> content.source_type
        'text'
    """

    name = "text"
    supported_mimes = ("text/plain",)
    supported_extensions = (".txt", ".text", ".log")

    def can_handle(
        self,
        raw_input: str | bytes | Path,
        *,
        mime_type: str | None = None,
        filename: str | None = None,
    ) -> bool:
        if super().can_handle(raw_input, mime_type=mime_type, filename=filename):
            return True
        if isinstance(raw_input, str):
            s = raw_input.strip()
            # Let specialized extractors handle URLs, HTML tags, or markdown headers
            if (
                s.startswith("http://")
                or s.startswith("https://")
                or s.startswith("#")
                or s.startswith("---")
                or bool(re.search(r"<\w+[^>]*>", s))
            ):
                return False
            return bool(s)
        return False

    def extract(
        self,
        raw_input: str | bytes | Path,
        *,
        source_uri: str | None = None,
        **kwargs: Any,
    ) -> ExtractedContent:
        origin_uri = source_uri

        if isinstance(raw_input, Path):
            origin_uri = origin_uri or str(raw_input)
            try:
                text = raw_input.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = raw_input.read_text(encoding="latin-1", errors="replace")
        elif isinstance(raw_input, bytes):
            try:
                text = raw_input.decode("utf-8")
            except UnicodeDecodeError:
                text = raw_input.decode("latin-1", errors="replace")
        elif isinstance(raw_input, str):
            text = raw_input
        else:
            raise MalformedContentError(f"Unsupported text input type: {type(raw_input)}")

        clean_text = text.strip()
        if not clean_text:
            raise MalformedContentError("Text content cannot be empty or contains only whitespace.")

        # Extract title from the first non-empty line and strip markdown # symbols
        first_line = clean_text.splitlines()[0].strip()
        title = re.sub(r"^#+\s*", "", first_line)
        title = title[:60] + "..." if len(title) > 60 else title

        return ExtractedContent(
            text=clean_text,
            title=title or "Quick Note",
            source_type=self.name,
            source_uri=origin_uri,
            metadata={"format": "plain_text"},
        )
