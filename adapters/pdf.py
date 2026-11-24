"""SecondSelf — PDF Document Content Extractor."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from adapters.base import BaseExtractor, ExtractionError, MalformedContentError, MissingDependencyError
from adapters.models import ExtractedContent


class PDFExtractor(BaseExtractor):
    """Adapter for multi-page PDF documents.

    Lazy-loads pypdf to preserve fast startup and serverless portability.
    Extracts text page by page, tracking page counts and warnings for blank pages.

    Example:
        >>> extractor = PDFExtractor()
        >>> content = extractor.extract(Path("paper.pdf"))
        >>> content.metadata['page_count'] > 0
        True
    """

    name = "pdf"
    supported_mimes = ("application/pdf",)
    supported_extensions = (".pdf",)

    def extract(
        self,
        raw_input: str | bytes | Path,
        *,
        source_uri: str | None = None,
        **kwargs: Any,
    ) -> ExtractedContent:
        # Lazy import pypdf to avoid loading heavy dependencies on module import
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise MissingDependencyError(
                "pypdf is required to extract PDF files. "
                "Install via: pip install -r requirements/pipeline.txt"
            ) from exc

        origin_uri = source_uri
        stream: io.BytesIO | Path

        if isinstance(raw_input, Path):
            if not raw_input.exists():
                raise FileNotFoundError(f"PDF file does not exist: {raw_input}")
            origin_uri = origin_uri or str(raw_input)
            stream = raw_input
            title = raw_input.stem.replace("_", " ").replace("-", " ").title()
        elif isinstance(raw_input, bytes):
            if not raw_input.startswith(b"%PDF-"):
                raise MalformedContentError("Byte sequence does not have a valid %PDF- magic header.")
            stream = io.BytesIO(raw_input)
            title = "Extracted PDF Document"
        elif isinstance(raw_input, str):
            path = Path(raw_input)
            if path.exists():
                origin_uri = origin_uri or str(path)
                stream = path
                title = path.stem.replace("_", " ").replace("-", " ").title()
            else:
                raise FileNotFoundError(f"PDF path does not exist: {raw_input}")
        else:
            raise MalformedContentError(f"Unsupported PDF input type: {type(raw_input)}")

        try:
            reader = PdfReader(str(stream) if isinstance(stream, Path) else stream)
            total_pages = len(reader.pages)
        except Exception as exc:
            raise ExtractionError(f"Could not parse PDF document: {exc}") from exc

        if total_pages == 0:
            raise MalformedContentError("PDF contains 0 pages.")

        pages_text: list[str] = []
        warnings: list[str] = []

        for idx, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text() or ""
                page_clean = page_text.strip()
                if page_clean:
                    pages_text.append(f"--- Page {idx + 1} ---\n{page_clean}")
                else:
                    warnings.append(f"Page {idx + 1} has no extractable text (may be scanned or an image).")
            except Exception as exc:
                warnings.append(f"Failed to extract page {idx + 1}: {exc}")

        combined_text = "\n\n".join(pages_text).strip()
        if not combined_text:
            raise MalformedContentError(
                "No readable text could be extracted from PDF (document may be purely scanned images)."
            )

        # Attempt to extract title from PDF metadata
        if reader.metadata and reader.metadata.title:
            meta_title = reader.metadata.title.strip()
            if meta_title:
                title = meta_title

        metadata: dict[str, Any] = {
            "format": "pdf",
            "page_count": total_pages,
            "extracted_pages": len(pages_text),
        }
        if reader.metadata:
            for k in ("author", "creator", "producer", "subject"):
                val = getattr(reader.metadata, k, None)
                if val:
                    metadata[k] = str(val)

        return ExtractedContent(
            text=combined_text,
            title=title,
            source_type=self.name,
            source_uri=origin_uri,
            metadata=metadata,
            warnings=warnings,
        )
