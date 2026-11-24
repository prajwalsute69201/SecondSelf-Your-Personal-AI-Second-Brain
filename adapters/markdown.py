"""SecondSelf — Markdown Document Extractor."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from adapters.base import BaseExtractor, MalformedContentError
from adapters.models import ExtractedContent


class MarkdownExtractor(BaseExtractor):
    """Adapter for Markdown files with YAML frontmatter awareness.

    Example:
        >>> extractor = MarkdownExtractor()
        >>> content = extractor.extract("# Graph Engine\\nPhysics-based canvas.")
        >>> content.title
        'Graph Engine'
        >>> content.source_type
        'markdown'
    """

    name = "markdown"
    supported_mimes = ("text/markdown", "text/x-markdown")
    supported_extensions = (".md", ".markdown", ".mdown")

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
            # Detect markdown headings or frontmatter
            return s.startswith("---") or bool(re.match(r"^#{1,6}\s+\S+", s))
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
            raise MalformedContentError(f"Unsupported markdown input type: {type(raw_input)}")

        clean_text = text.strip()
        if not clean_text:
            raise MalformedContentError("Markdown content is empty.")

        metadata: dict[str, Any] = {"format": "markdown"}
        title = "Untitled Markdown"

        # Check for YAML Frontmatter: --- \n ... \n ---
        fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", clean_text, re.DOTALL)
        if fm_match:
            frontmatter_raw, body_raw = fm_match.groups()
            clean_text = body_raw.strip()
            # Basic key-value parsing for frontmatter
            for line in frontmatter_raw.splitlines():
                if ":" in line:
                    key, val = line.split(":", 1)
                    k = key.strip()
                    v = val.strip().strip("'\"")
                    metadata[k] = v
                    if k.lower() == "title" and v:
                        title = v

        # If title wasn't found in frontmatter, find first # Heading
        if title == "Untitled Markdown":
            for line in clean_text.splitlines():
                line_s = line.strip()
                if line_s.startswith("#"):
                    title = re.sub(r"^#+\s*", "", line_s)
                    break
            else:
                first_line = clean_text.splitlines()[0].strip() if clean_text else "Quick Note"
                title = first_line[:60] + "..." if len(first_line) > 60 else first_line

        return ExtractedContent(
            text=clean_text,
            title=title,
            source_type=self.name,
            source_uri=origin_uri,
            metadata=metadata,
        )
