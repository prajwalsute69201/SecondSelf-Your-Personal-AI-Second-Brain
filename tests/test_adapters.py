"""Unit and integration tests for SecondSelf Content Extraction Adapters."""

from __future__ import annotations

import io
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch

from adapters import (
    BaseExtractor,
    ExtractedContent,
    ExtractorRegistry,
    HTMLExtractor,
    MalformedContentError,
    MarkdownExtractor,
    MissingDependencyError,
    PDFExtractor,
    TextExtractor,
    UnsupportedContentError,
    extract_content,
    get_registry,
)


# ------------------------------------------------------------------------------
# 1. TextExtractor Tests
# ------------------------------------------------------------------------------

def test_text_extractor_direct_string() -> None:
    extractor = TextExtractor()
    content = extractor.extract("Idea: Use cosine similarity for associative memory.")
    assert content.source_type == "text"
    assert content.title == "Idea: Use cosine similarity for associative memory."
    assert content.char_count == len("Idea: Use cosine similarity for associative memory.")
    assert content.word_count == 7
    assert len(content.content_hash) == 64


def test_text_extractor_empty_raises() -> None:
    extractor = TextExtractor()
    with pytest.raises(MalformedContentError, match="empty or contains only whitespace"):
        extractor.extract("   \n\t  ")


# ------------------------------------------------------------------------------
# 2. MarkdownExtractor Tests
# ------------------------------------------------------------------------------

def test_markdown_extractor_with_heading() -> None:
    extractor = MarkdownExtractor()
    md = "# Vector Indexes\n\nExploring HNSW and flat vector indexes."
    content = extractor.extract(md)
    assert content.source_type == "markdown"
    assert content.title == "Vector Indexes"
    assert "Exploring HNSW" in content.text


def test_markdown_extractor_with_yaml_frontmatter() -> None:
    extractor = MarkdownExtractor()
    md = """---
title: PARA Classification Architecture
author: SecondSelf
category: Resources
---
# The Architecture
Detailed notes on knowledge vaults.
"""
    content = extractor.extract(md)
    assert content.title == "PARA Classification Architecture"
    assert content.metadata.get("category") == "Resources"
    assert content.metadata.get("author") == "SecondSelf"
    assert "Detailed notes on knowledge vaults." in content.text


# ------------------------------------------------------------------------------
# 3. HTMLExtractor Tests
# ------------------------------------------------------------------------------

def test_html_extractor_raw_html_string() -> None:
    extractor = HTMLExtractor()
    html = "<html><head><title>Test Article</title></head><body><h1>Main Title</h1><p>This is clean article content.</p></body></html>"
    content = extractor.extract(html)
    assert content.source_type == "html"
    assert "This is clean article content." in content.text
    assert content.title in ("Test Article", "Main Title")


def test_html_extractor_web_url_mocked() -> None:
    extractor = HTMLExtractor()
    fake_html = "<html><head><title>Mocked Page</title></head><body><p>Mocked web text for testing.</p></body></html>"

    mock_resp = MagicMock()
    mock_resp.text = fake_html
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp):
        content = extractor.extract("https://example.com/mock-article")
        assert content.source_type == "html"
        assert content.source_uri == "https://example.com/mock-article"
        assert "Mocked web text" in content.text
        assert content.metadata.get("http_status") == 200


# ------------------------------------------------------------------------------
# 4. PDFExtractor Tests
# ------------------------------------------------------------------------------

def test_pdf_extractor_in_memory_bytes() -> None:
    # Build a minimal valid PDF in memory using pypdf
    from pypdf import PdfWriter
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    # Note: blank page will trigger warning or malformed if no text
    buf = io.BytesIO()
    writer.write(buf)
    pdf_bytes = buf.getvalue()

    extractor = PDFExtractor()
    # Blank PDF has no text, so should raise MalformedContentError
    with pytest.raises(MalformedContentError, match="No readable text"):
        extractor.extract(pdf_bytes)


def test_pdf_extractor_missing_file_raises() -> None:
    extractor = PDFExtractor()
    with pytest.raises(FileNotFoundError):
        extractor.extract(Path("non_existent_paper.pdf"))


# ------------------------------------------------------------------------------
# 5. Registry & Dispatch Tests
# ------------------------------------------------------------------------------

def test_registry_lists_standard_adapters() -> None:
    registry = get_registry()
    adapters = registry.list_adapters()
    for expected in ("text", "markdown", "html", "pdf"):
        assert expected in adapters


def test_registry_auto_resolve_by_type() -> None:
    registry = get_registry()

    # URL -> HTML
    assert registry.resolve("https://news.ycombinator.com").name == "html"

    # Raw HTML string -> HTML
    assert registry.resolve("<p>Some paragraph</p>").name == "html"

    # Markdown Path -> Markdown
    assert registry.resolve(Path("notes.md")).name == "markdown"

    # PDF Path -> PDF
    assert registry.resolve(Path("paper.pdf")).name == "pdf"

    # Plain text -> Text
    assert registry.resolve("Just a thought").name == "text"


def test_registry_explicit_adapter_selection() -> None:
    registry = get_registry()
    # Explicitly force 'text' adapter on an HTML string
    content = registry.extract("<h1>Title</h1>", adapter_name="text")
    assert content.source_type == "text"
    assert "<h1>Title</h1>" in content.text


def test_registry_unknown_adapter_name_raises() -> None:
    registry = get_registry()
    with pytest.raises(KeyError, match="No adapter registered under name 'unknown'"):
        registry.get("unknown")


def test_extract_content_convenience_function() -> None:
    result = extract_content("Testing the global convenience extractor.")
    assert isinstance(result, ExtractedContent)
    assert result.source_type == "text"
    assert result.word_count == 5


def test_one_line_custom_adapter_registration() -> None:
    custom_reg = ExtractorRegistry()

    @custom_reg.register
    class CsvExtractor(BaseExtractor):
        name = "csv"
        supported_extensions = (".csv",)

        def extract(self, raw_input, **kwargs):
            text = str(raw_input).strip()
            return ExtractedContent(
                text=text,
                title="CSV Data",
                source_type="csv",
            )

    assert "csv" in custom_reg.list_adapters()
    content = custom_reg.extract("col1,col2\nval1,val2", filename="test.csv")
    assert content.source_type == "csv"
    assert content.title == "CSV Data"
