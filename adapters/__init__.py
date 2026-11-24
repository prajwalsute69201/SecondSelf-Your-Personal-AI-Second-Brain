"""SecondSelf — Content Extraction Adapters Public API.

Exposes the standardized adapter registry, base contracts, normalized
models, and concrete extractors for Text, Markdown, HTML, and PDF.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from adapters.base import (
    BaseExtractor,
    ExtractionError,
    MalformedContentError,
    MissingDependencyError,
    UnsupportedContentError,
)
from adapters.html import HTMLExtractor
from adapters.markdown import MarkdownExtractor
from adapters.models import ExtractedContent
from adapters.pdf import PDFExtractor
from adapters.registry import ExtractorRegistry
from adapters.text import TextExtractor

# Default global singleton registry
_GLOBAL_REGISTRY = ExtractorRegistry()

# Register standard adapters in resolution order (Specialized first, fallback Text last)
_GLOBAL_REGISTRY.register(TextExtractor())
_GLOBAL_REGISTRY.register(MarkdownExtractor())
_GLOBAL_REGISTRY.register(PDFExtractor())
_GLOBAL_REGISTRY.register(HTMLExtractor())


def get_registry() -> ExtractorRegistry:
    """Returns the default global ExtractorRegistry instance."""
    return _GLOBAL_REGISTRY


def extract_content(
    raw_input: str | bytes | Path,
    *,
    adapter_name: str | None = None,
    mime_type: str | None = None,
    filename: str | None = None,
    source_uri: str | None = None,
    **kwargs: Any,
) -> ExtractedContent:
    """Convenience function to extract content via the global registry.

    Args:
        raw_input: Raw string, byte buffer, or Path.
        adapter_name: Optional explicit adapter name ('text', 'markdown', 'html', 'pdf').
        mime_type: Optional explicit MIME type.
        filename: Optional filename with extension.
        source_uri: Optional source URI.
        **kwargs: Extra parameters passed to the extractor.

    Returns:
        ExtractedContent normalized model.

    Example:
        >>> from adapters import extract_content
        >>> result = extract_content("A quick thought for my second brain.")
        >>> result.source_type
        'text'
        >>> result.title
        'A quick thought for my second brain.'
    """
    return _GLOBAL_REGISTRY.extract(
        raw_input,
        adapter_name=adapter_name,
        mime_type=mime_type,
        filename=filename,
        source_uri=source_uri,
        **kwargs,
    )


__all__ = [
    "BaseExtractor",
    "ExtractedContent",
    "ExtractionError",
    "ExtractorRegistry",
    "HTMLExtractor",
    "MalformedContentError",
    "MarkdownExtractor",
    "MissingDependencyError",
    "PDFExtractor",
    "TextExtractor",
    "UnsupportedContentError",
    "extract_content",
    "get_registry",
]
