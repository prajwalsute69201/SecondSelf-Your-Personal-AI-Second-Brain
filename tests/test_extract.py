"""Unit tests for lib/extract.py text extraction helpers (Task 2.1.3).

Validates:
- Plain text and Markdown note extraction (including UTF-8 and UTF-8 BOM).
- PDF extraction using pypdf in original page order.
- PDF fallback to filename on malformed or empty/scanned PDFs.
- Link extraction with HTML stripping and graceful URL string fallback.
- Safe binary file handling without corrupted text decoding.
- Empty files and directory error handling.
- Missing file error handling (explicitly raises FileNotFoundError).
- Unified extract_text dispatcher for raw captures and CaptureMeta instances.
"""

import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Ensure secondself is in sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import lib.extract as extract
from lib.models import CaptureMeta
import lib.storage as storage
import pypdf
import requests


def _create_minimal_pdf_bytes(pages_text: list[str]) -> bytes:
    """Generate minimal valid PDF byte stream with the specified text per page."""
    # Build PDF with pypdf writer and content streams
    writer = pypdf.PdfWriter()
    for text in pages_text:
        page = writer.add_blank_page(width=612, height=792)
        # In pypdf, we can attach a stream or create simple text
    # Or write raw PDF objects
    objects = []
    # Simple direct construction of valid PDF with extractable text
    # Page object format with text streams
    out = io.BytesIO()
    # Use pypdf to assemble
    writer = pypdf.PdfWriter()
    for t in pages_text:
        # Create page with text
        page = writer.add_blank_page(width=612, height=792)
    # Alternatively write raw PDF text stream
    return _build_raw_pdf(pages_text)


def _build_raw_pdf(pages_text: list[str]) -> bytes:
    """Build a deterministic raw multi-page PDF byte string with extractable text."""
    # Simple PDF syntax with Type1 font and text streams
    objs: list[bytes] = []
    page_obj_ids = []

    # Object 1: Catalog
    objs.append(b"1 0 obj\n<</Type/Catalog/Pages 2 0 R>>\nendobj\n")
    # Object 2 will be Pages (filled later)
    # Font object ID will be 3
    font_id = 3
    objs.append(b"3 0 obj\n<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>\nendobj\n")

    current_id = 4
    for text in pages_text:
        page_id = current_id
        content_id = current_id + 1
        current_id += 2

        # Sanitize text for PDF string literal
        clean_text = text.replace("(", "\\(").replace(")", "\\)")
        stream_data = f"BT /F1 14 Tf 50 700 Td ({clean_text}) Tj ET".encode("latin-1")
        stream_len = len(stream_data)

        page_obj = (
            f"{page_id} 0 obj\n"
            f"<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents {content_id} 0 R"
            f"/Resources<</Font<</F1 {font_id} 0 R>>>>>>\nendobj\n"
        ).encode("ascii")

        content_obj = (
            f"{content_id} 0 obj\n<</Length {stream_len}>>\nstream\n"
        ).encode("ascii") + stream_data + b"\nendstream\nendobj\n"

        objs.append(page_obj)
        objs.append(content_obj)
        page_obj_ids.append(page_id)

    # Now build Pages object (ID 2)
    kids = " ".join(f"{pid} 0 R" for pid in page_obj_ids)
    pages_obj = (
        f"2 0 obj\n<</Type/Pages/Kids[{kids}]/Count {len(page_obj_ids)}>>\nendobj\n"
    ).encode("ascii")
    # Insert object 2 at position 1
    objs.insert(1, pages_obj)

    # Assemble file
    body = io.BytesIO()
    body.write(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objs:
        offsets.append(body.tell())
        body.write(obj)

    xref_offset = body.tell()
    body.write(f"xref\n0 {len(offsets)}\n".encode("ascii"))
    body.write(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        body.write(f"{off:010d} 00000 n \n".encode("ascii"))

    trailer = (
        f"trailer\n<</Size {len(offsets)}/Root 1 0 R>>\nstartxref\n{xref_offset}\n%%EOF\n"
    ).encode("ascii")
    body.write(trailer)
    return body.getvalue()


class TestExtractModule(unittest.TestCase):
    """Test suite for lib/extract.py."""

    def setUp(self) -> None:
        """Create an isolated temporary directory for test fixtures."""
        self.test_dir = tempfile.mkdtemp(prefix="secondself_extract_test_")
        self.test_path = Path(self.test_dir)
        storage.set_project_root(self.test_path)

    def tearDown(self) -> None:
        """Clean up temporary directory and reset storage root."""
        storage.set_project_root(None)
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_01_plain_text_and_markdown_extraction(self) -> None:
        """Extract text from plain text and Markdown files."""
        # Plain text
        txt_file = self.test_path / "sample.txt"
        txt_content = "Simple note without formatting."
        txt_file.write_text(txt_content, encoding="utf-8")

        extracted_txt = extract.extract_text_from_note(txt_file)
        self.assertEqual(extracted_txt, txt_content)

        # Markdown
        md_file = self.test_path / "content.md"
        md_content = "# Project Title\n\n- Task 1: Scaffolding\n- Task 2: Extraction\n\n```python\nprint('hello')\n```"
        md_file.write_text(md_content, encoding="utf-8")

        extracted_md = extract.extract_text_from_note(md_file)
        self.assertEqual(extracted_md, md_content)

    def test_02_unicode_with_and_without_bom(self) -> None:
        """Handle UTF-8 with and without BOM correctly."""
        # With BOM (\xef\xbb\xbf)
        bom_file = self.test_path / "bom.md"
        unicode_text = "# Unicode Note: Café, Zürich, 日本語, 🚀"
        bom_bytes = b"\xef\xbb\xbf" + unicode_text.encode("utf-8")
        bom_file.write_bytes(bom_bytes)

        extracted = extract.extract_text_from_note(bom_file)
        # BOM must be stripped
        self.assertFalse(extracted.startswith("\ufeff"))
        self.assertEqual(extracted, unicode_text)

        # Without BOM
        nobom_file = self.test_path / "nobom.md"
        nobom_file.write_text(unicode_text, encoding="utf-8")
        self.assertEqual(extract.extract_text_from_note(nobom_file), unicode_text)

    def test_03_pdf_extraction_page_order(self) -> None:
        """Extract text from PDF pages in their original order using pypdf."""
        pdf_file = self.test_path / "document.pdf"
        pages = [
            "Page 1: Introduction to SecondSelf architecture",
            "Page 2: Vector embeddings and graph visualization",
        ]
        pdf_bytes = _build_raw_pdf(pages)
        pdf_file.write_bytes(pdf_bytes)

        extracted = extract.extract_text_from_pdf(pdf_file)
        self.assertIn("Page 1: Introduction to SecondSelf architecture", extracted)
        self.assertIn("Page 2: Vector embeddings and graph visualization", extracted)

        # Confirm Page 1 text appears before Page 2 text
        idx1 = extracted.index("Page 1")
        idx2 = extracted.index("Page 2")
        self.assertLess(idx1, idx2)

    def test_04_pdf_scanned_or_blank_falls_back_to_filename(self) -> None:
        """PDF with no extractable text falls back to filename."""
        blank_pdf_file = self.test_path / "scanned_invoice.pdf"
        writer = pypdf.PdfWriter()
        writer.add_blank_page(width=72, height=72)
        with blank_pdf_file.open("wb") as f:
            writer.write(f)

        extracted = extract.extract_text_from_pdf(
            blank_pdf_file,
            fallback_filename="original_invoice.pdf",
        )
        self.assertEqual(extracted, "original_invoice.pdf")

    def test_05_malformed_pdf_falls_back_to_filename(self) -> None:
        """Malformed or corrupted PDF safely falls back to filename without crashing."""
        corrupted_file = self.test_path / "corrupted.pdf"
        corrupted_file.write_bytes(b"%PDF-1.4\nCorrupted binary junk \x00\xff\xfe\xdeadbeef")

        extracted = extract.extract_text_from_pdf(corrupted_file)
        self.assertEqual(extracted, "corrupted.pdf")

    def test_06_empty_file_returns_empty_string(self) -> None:
        """Empty text file returns empty string."""
        empty_file = self.test_path / "empty.txt"
        empty_file.write_bytes(b"")

        extracted = extract.extract_text_from_file(empty_file)
        self.assertEqual(extracted, "")

    def test_07_unsupported_binary_file_falls_back_to_filename(self) -> None:
        """Arbitrary binary file falls back to filename without corrupted text decoding."""
        bin_file = self.test_path / "system_firmware.bin"
        # Binary data with null bytes and invalid UTF-8 sequences
        bin_file.write_bytes(b"\x00\x01\x02\xff\xfe\x80\xaa\x55binarypayload\x00")

        extracted = extract.extract_text_from_file(
            bin_file,
            original_filename="firmware_update.bin",
        )
        self.assertEqual(extracted, "firmware_update.bin")

    def test_08_missing_files_raise_filenotfound(self) -> None:
        """Missing files explicitly raise FileNotFoundError (no silent empty string substitution)."""
        missing = self.test_path / "nonexistent_file.md"

        with self.assertRaises(FileNotFoundError):
            extract.extract_text_from_note(missing)

        with self.assertRaises(FileNotFoundError):
            extract.extract_text_from_pdf(missing)

        with self.assertRaises(FileNotFoundError):
            extract.extract_text_from_file(missing)

    def test_09_directory_path_raises_value_error(self) -> None:
        """Directory paths passed to file extractors raise ValueError."""
        sub_dir = self.test_path / "sub_directory"
        sub_dir.mkdir()

        with self.assertRaises(ValueError):
            extract.extract_text_from_note(sub_dir)

        with self.assertRaises(ValueError):
            extract.extract_text_from_pdf(sub_dir)

        with self.assertRaises(ValueError):
            extract.extract_text_from_file(sub_dir)

    def test_10_link_extraction_html_stripping(self) -> None:
        """Extract web link content by stripping HTML tags with BeautifulSoup."""
        html_markup = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Graph Embeddings Paper</title>
            <style>body { background: black; }</style>
            <script>console.log("tracking code");</script>
        </head>
        <body>
            <header><nav><a href="/">Home</a></nav></header>
            <main>
                <h1>Knowledge Graph Embeddings</h1>
                <p>This paper presents a novel approach to force-directed layouts.</p>
                <p>Nodes are colored by PARA category.</p>
            </main>
            <footer>Copyright 2026</footer>
        </body>
        </html>
        """
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = html_markup
        mock_response.headers = {"content-type": "text/html; charset=utf-8"}
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response):
            text = extract.extract_text_from_link("https://arxiv.org/abs/2301.00001")

        self.assertIn("Knowledge Graph Embeddings", text)
        self.assertIn("novel approach to force-directed layouts", text)
        self.assertNotIn("tracking code", text)
        self.assertNotIn("Copyright 2026", text)

    def test_11_link_extraction_network_failure_fallback_to_url(self) -> None:
        """Link extraction falls back to URL string on network failure, timeout, or HTTP error."""
        # Connection error
        with patch("requests.get", side_effect=requests.ConnectionError("DNS failure")):
            fallback = extract.extract_text_from_link("https://example.com/unreachable")
            self.assertEqual(fallback, "https://example.com/unreachable")

        # Timeout
        with patch("requests.get", side_effect=requests.Timeout("Timed out")):
            fallback = extract.extract_text_from_link("https://example.com/slow")
            self.assertEqual(fallback, "https://example.com/slow")

        # HTTP 404
        mock_err_response = MagicMock()
        mock_err_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
        with patch("requests.get", return_value=mock_err_response):
            fallback = extract.extract_text_from_link("https://example.com/missing")
            self.assertEqual(fallback, "https://example.com/missing")

    def test_12_link_extraction_from_content_file_with_notes(self) -> None:
        """Extract link from raw content.txt preserving user notes on success and fallback."""
        link_content_file = self.test_path / "content.txt"
        link_content_file.write_text(
            "https://arxiv.org/abs/2301.00001\n\nMust-read reference for embeddings.",
            encoding="utf-8",
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html><body><h1>Embeddings Paper</h1><p>Body text</p></body></html>"
        mock_response.headers = {"content-type": "text/html"}
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response):
            extracted = extract.extract_text_from_link(link_content_file)

        self.assertIn("Embeddings Paper", extracted)
        self.assertIn("Must-read reference for embeddings.", extracted)

        # Test fallback on content file when fetch fails
        with patch("requests.get", side_effect=requests.ConnectionError("Offline")):
            fallback = extract.extract_text_from_link(link_content_file)
            self.assertIn("https://arxiv.org/abs/2301.00001", fallback)
            self.assertIn("Must-read reference for embeddings.", fallback)

    def test_13_unified_extract_text_dispatcher(self) -> None:
        """Test unified extract_text helper for note, link, and file raw captures."""
        raw_dir = storage.get_raw_dir(self.test_path)

        # 1. Note capture
        note_id = "2026-11-08_note0001"
        note_dir = raw_dir / note_id
        note_dir.mkdir(parents=True)
        (note_dir / "meta.json").write_text(
            json.dumps({"id": note_id, "timestamp": "...", "type": "note", "source": "cli", "original_filename": None, "content_hash": "..."}),
            encoding="utf-8",
        )
        (note_dir / "content.md").write_text("# Note Title\nNote body content.", encoding="utf-8")

        meta_note = CaptureMeta(note_id, "...", "note", "cli", None, "...")
        text_note = extract.extract_text(meta_note, root=self.test_path)
        self.assertEqual(text_note, "# Note Title\nNote body content.")

        # Test directory path dispatch
        text_note_dir = extract.extract_text(note_dir)
        self.assertEqual(text_note_dir, "# Note Title\nNote body content.")

        # 2. File capture (PDF)
        file_id = "2026-11-08_file0001"
        file_dir = raw_dir / file_id
        file_dir.mkdir(parents=True)
        (file_dir / "meta.json").write_text(
            json.dumps({"id": file_id, "timestamp": "...", "type": "file", "source": "path", "original_filename": "paper.pdf", "content_hash": "..."}),
            encoding="utf-8",
        )
        pdf_bytes = _build_raw_pdf(["Research PDF Content"])
        (file_dir / "content.pdf").write_bytes(pdf_bytes)

        meta_file = CaptureMeta(file_id, "...", "file", "path", "paper.pdf", "...")
        text_file = extract.extract_text(meta_file, root=self.test_path)
        self.assertIn("Research PDF Content", text_file)

        # 3. Link capture
        link_id = "2026-11-08_link0001"
        link_dir = raw_dir / link_id
        link_dir.mkdir(parents=True)
        (link_dir / "meta.json").write_text(
            json.dumps({"id": link_id, "timestamp": "...", "type": "link", "source": "https://example.com/info", "original_filename": None, "content_hash": "..."}),
            encoding="utf-8",
        )
        (link_dir / "content.txt").write_text("https://example.com/info", encoding="utf-8")

        meta_link = CaptureMeta(link_id, "...", "link", "https://example.com/info", None, "...")
        with patch("requests.get", side_effect=requests.ConnectionError("Offline")):
            text_link = extract.extract_text(meta_link, root=self.test_path)
            self.assertEqual(text_link, "https://example.com/info")


if __name__ == "__main__":
    unittest.main()

