"""SecondSelf — Content Extraction Adapter Registry & Factory."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, TypeVar

from adapters.base import BaseExtractor, UnsupportedContentError
from adapters.models import ExtractedContent

T = TypeVar("T", bound=BaseExtractor)


class ExtractorRegistry:
    """Registry and factory for pluggable content extraction adapters."""

    def __init__(self) -> None:
        self._adapters: dict[str, BaseExtractor] = {}
        self._order: list[BaseExtractor] = []

    def register(self, adapter_or_cls: BaseExtractor | type[BaseExtractor]) -> BaseExtractor:
        """Registers a new adapter instance or class. Can be used as a decorator.

        Example:
            >>> registry = ExtractorRegistry()
            >>> @registry.register
            ... class CustomExtractor(BaseExtractor):
            ...     name = "custom"
            ...     def extract(self, raw_input, **kwargs): ...
        """
        instance = adapter_or_cls() if isinstance(adapter_or_cls, type) else adapter_or_cls
        name = instance.name.lower()
        self._adapters[name] = instance
        # Maintain resolution priority (most recently registered checked first)
        if instance in self._order:
            self._order.remove(instance)
        self._order.insert(0, instance)
        return instance

    def get(self, name: str) -> BaseExtractor:
        """Retrieves an adapter by explicit name.

        Args:
            name: Case-insensitive adapter identifier (e.g. 'html', 'pdf', 'text').

        Returns:
            Registered BaseExtractor instance.

        Raises:
            KeyError: If no adapter with the specified name is registered.
        """
        key = name.strip().lower()
        if key not in self._adapters:
            available = ", ".join(repr(k) for k in self._adapters.keys())
            raise KeyError(f"No adapter registered under name '{name}'. Available adapters: [{available}]")
        return self._adapters[key]

    def list_adapters(self) -> list[str]:
        """Returns the names of all registered adapters."""
        return list(self._adapters.keys())

    def resolve(
        self,
        raw_input: str | bytes | Path,
        *,
        mime_type: str | None = None,
        filename: str | None = None,
    ) -> BaseExtractor:
        """Automatically resolves the appropriate adapter for the given input.

        Args:
            raw_input: Raw string content, byte buffer, or Path.
            mime_type: Optional explicit MIME type (e.g. 'application/pdf').
            filename: Optional filename used for extension matching.

        Returns:
            The first matching BaseExtractor instance.

        Raises:
            UnsupportedContentError: If no adapter can handle the input.
        """
        # 1. Ask registered adapters in priority order
        for adapter in self._order:
            if adapter.can_handle(raw_input, mime_type=mime_type, filename=filename):
                return adapter

        # 2. Heuristic fallback based on input properties
        target_name = filename or (str(raw_input) if isinstance(raw_input, Path) else None)
        if target_name:
            ext = Path(target_name).suffix.lower()
            if ext == ".pdf" and "pdf" in self._adapters:
                return self._adapters["pdf"]
            if ext in (".md", ".markdown") and "markdown" in self._adapters:
                return self._adapters["markdown"]
            if ext in (".html", ".htm") and "html" in self._adapters:
                return self._adapters["html"]
            if ext in (".txt", ".text") and "text" in self._adapters:
                return self._adapters["text"]

        # Default string fallback to text adapter
        if isinstance(raw_input, str) and "text" in self._adapters:
            return self._adapters["text"]

        raise UnsupportedContentError(
            f"No suitable extraction adapter found for input of type '{type(raw_input).__name__}' "
            f"(MIME: {mime_type}, Filename: {filename})."
        )

    def extract(
        self,
        raw_input: str | bytes | Path,
        *,
        adapter_name: str | None = None,
        mime_type: str | None = None,
        filename: str | None = None,
        source_uri: str | None = None,
        **kwargs: Any,
    ) -> ExtractedContent:
        """Extracts content using either an explicitly named or auto-resolved adapter.

        Args:
            raw_input: Raw input data (string, bytes, or Path).
            adapter_name: Optional explicit adapter name (bypasses auto-detection).
            mime_type: Optional explicit MIME type for auto-detection.
            filename: Optional filename for extension auto-detection.
            source_uri: Optional source URI or file path recorded on output.
            **kwargs: Extra parameters passed into the concrete extractor.

        Returns:
            Normalized ExtractedContent model.
        """
        if adapter_name:
            adapter = self.get(adapter_name)
        else:
            adapter = self.resolve(raw_input, mime_type=mime_type, filename=filename)

        return adapter.extract(raw_input, source_uri=source_uri, **kwargs)
