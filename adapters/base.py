"""SecondSelf — Base Content Extractor Contract and Exceptions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from adapters.models import ExtractedContent


class ExtractionError(Exception):
    """Base exception for all extraction failures."""


class UnsupportedContentError(ValueError, ExtractionError):
    """Raised when no registered adapter can process the given input."""


class MissingDependencyError(ImportError, ExtractionError):
    """Raised when an optional third-party library is not installed."""


class MalformedContentError(ValueError, ExtractionError):
    """Raised when input content is corrupted, empty, or unparseable."""


class BaseExtractor(ABC):
    """Abstract base class defining the content extraction contract."""

    name: str = "base"
    supported_mimes: tuple[str, ...] = ()
    supported_extensions: tuple[str, ...] = ()

    @abstractmethod
    def extract(
        self,
        raw_input: str | bytes | Path,
        *,
        source_uri: str | None = None,
        **kwargs: Any,
    ) -> ExtractedContent:
        """Extracts normalized text and metadata from the given input.

        Args:
            raw_input: Raw string content, byte stream, or filesystem Path.
            source_uri: Optional original origin (URL, filepath, or label).
            **kwargs: Adapter-specific parameters (e.g. timeout, headers).

        Returns:
            ExtractedContent instance with clean text and metadata.

        Raises:
            MalformedContentError: If input is empty or invalid.
            ExtractionError: If extraction fails.
        """

    def can_handle(
        self,
        raw_input: str | bytes | Path,
        *,
        mime_type: str | None = None,
        filename: str | None = None,
    ) -> bool:
        """Determines whether this adapter can process the given input.

        Args:
            raw_input: The input object or content.
            mime_type: Optional explicit MIME type (e.g. 'text/html').
            filename: Optional filename or path with extension.

        Returns:
            True if this adapter can process the input, False otherwise.
        """
        if mime_type and mime_type.lower() in self.supported_mimes:
            return True
        if filename:
            ext = Path(filename).suffix.lower()
            if ext in self.supported_extensions:
                return True
        if isinstance(raw_input, Path):
            return raw_input.suffix.lower() in self.supported_extensions
        return False
