"""Unit and integration tests for capture.py core capture functions.

Validates that capture_note, capture_link, and capture_file adhere to data models,
filesystem contracts, hashing, ID generation, timestamping, and output formatting,
while strictly executing in isolated temporary directories.
"""

from contextlib import redirect_stdout
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

# Ensure secondself is in sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import capture
from lib.models import CaptureMeta, CaptureResult
import lib.storage as storage


class TestCaptureFunctions(unittest.TestCase):
    """Test suite for capture.py core functions."""

    def setUp(self) -> None:
        """Create an isolated temporary directory for storage testing."""
        self.test_dir = tempfile.mkdtemp(prefix="secondself_test_")
        self.test_path = Path(self.test_dir)
        # Point lib.storage to the isolated temporary directory
        storage.set_project_root(self.test_path)

    def tearDown(self) -> None:
        """Restore storage project root and clean up temporary files."""
        storage.set_project_root(None)
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_07_module_imports_successfully(self) -> None:
        """Requirement 7: The module imports successfully and exports required functions."""
        self.assertTrue(hasattr(capture, "capture_note"))
        self.assertTrue(hasattr(capture, "capture_link"))
        self.assertTrue(hasattr(capture, "capture_file"))
        self.assertEqual(
            capture.__all__,
            ["capture_note", "capture_link", "capture_file"],
        )

        import inspect

        sig_note = inspect.signature(capture.capture_note)
        self.assertEqual(list(sig_note.parameters.keys()), ["text"])

        sig_link = inspect.signature(capture.capture_link)
        self.assertEqual(list(sig_link.parameters.keys()), ["url", "notes"])
        self.assertEqual(sig_link.parameters["notes"].default, "")

        sig_file = inspect.signature(capture.capture_file)
        self.assertEqual(list(sig_file.parameters.keys()), ["path"])

    def test_01_capture_note_creates_metadata_and_content(self) -> None:
        """Requirement 1: capture_note returns a CaptureResult and creates expected files."""
        note_text = "# Project Roadmap\n\n- Phase 1: Core Capture\n- Phase 2: Classification"
        stdout_buf = io.StringIO()

        with redirect_stdout(stdout_buf):
            result = capture.capture_note(note_text)

        self.assertIsInstance(result, CaptureResult)
        self.assertEqual(result.type, "note")
        self.assertTrue(result.id)

        capture_dir = Path(result.path)
        self.assertTrue(capture_dir.is_dir())
        self.assertEqual(capture_dir.name, result.id)

        # Verify meta.json
        meta_file = capture_dir / "meta.json"
        self.assertTrue(meta_file.is_file())
        with meta_file.open("r", encoding="utf-8") as f:
            meta_data = json.load(f)

        self.assertEqual(meta_data["id"], result.id)
        self.assertEqual(meta_data["type"], "note")
        self.assertEqual(meta_data["source"], "cli")
        self.assertIsNone(meta_data["original_filename"])
        expected_hash = hashlib.sha256(note_text.encode("utf-8")).hexdigest()
        self.assertEqual(meta_data["content_hash"], expected_hash)

        # Verify content.md
        content_file = capture_dir / "content.md"
        self.assertTrue(content_file.is_file())
        with content_file.open("r", encoding="utf-8") as f:
            stored_content = f.read()
        self.assertEqual(stored_content, note_text)

    def test_02_capture_link_records_url_and_preserves_notes(self) -> None:
        """Requirement 2: capture_link records the URL in metadata and preserves optional notes."""
        url = "https://arxiv.org/abs/2301.00001"
        notes = "Key reference for vector embeddings in knowledge graphs."
        stdout_buf = io.StringIO()

        with redirect_stdout(stdout_buf):
            result = capture.capture_link(url, notes=notes)

        self.assertIsInstance(result, CaptureResult)
        self.assertEqual(result.type, "link")

        capture_dir = Path(result.path)
        self.assertTrue(capture_dir.is_dir())

        # Verify meta.json: URL must be in source for Phase 2 web extraction
        meta_file = capture_dir / "meta.json"
        with meta_file.open("r", encoding="utf-8") as f:
            meta_data = json.load(f)

        self.assertEqual(meta_data["id"], result.id)
        self.assertEqual(meta_data["type"], "link")
        self.assertEqual(meta_data["source"], url)
        self.assertIsNone(meta_data["original_filename"])

        # Verify content.txt: Predictable text content file containing the notes
        content_file = capture_dir / "content.txt"
        self.assertTrue(content_file.is_file())
        with content_file.open("r", encoding="utf-8") as f:
            stored_content = f.read()

        self.assertIn(notes, stored_content)
        self.assertIn(url, stored_content)

        # Verify capture_link without notes (default notes="")
        stdout_buf2 = io.StringIO()
        with redirect_stdout(stdout_buf2):
            result_no_notes = capture.capture_link("https://example.com/doc")

        capture_dir_no_notes = Path(result_no_notes.path)
        content_file_no_notes = capture_dir_no_notes / "content.txt"
        self.assertTrue(content_file_no_notes.is_file())
        with content_file_no_notes.open("r", encoding="utf-8") as f:
            self.assertEqual(f.read(), "https://example.com/doc")

    def test_03_capture_file_preserves_binary_bytes_and_records_filename(self) -> None:
        """Requirement 3: capture_file preserves binary bytes exactly and records original filename."""
        # Create a sample binary file with null bytes, arbitrary byte values, and high bytes
        raw_binary_data = b"\x00\x01\x02\x03\xff\xfe\xfd\x80\xaa\x55%PDF-1.4\r\n\x00\xff"
        src_file = self.test_path / "sample_document.pdf"
        src_file.write_bytes(raw_binary_data)

        stdout_buf = io.StringIO()
        with redirect_stdout(stdout_buf):
            result = capture.capture_file(src_file)

        self.assertIsInstance(result, CaptureResult)
        self.assertEqual(result.type, "file")

        capture_dir = Path(result.path)
        self.assertTrue(capture_dir.is_dir())

        # Verify meta.json: original_filename recorded and source documented
        meta_file = capture_dir / "meta.json"
        with meta_file.open("r", encoding="utf-8") as f:
            meta_data = json.load(f)

        self.assertEqual(meta_data["id"], result.id)
        self.assertEqual(meta_data["type"], "file")
        self.assertEqual(meta_data["original_filename"], "sample_document.pdf")
        self.assertEqual(meta_data["source"], str(src_file))
        expected_hash = hashlib.sha256(raw_binary_data).hexdigest()
        self.assertEqual(meta_data["content_hash"], expected_hash)

        # Verify content.pdf preserves bytes byte-for-byte
        content_file = capture_dir / "content.pdf"
        self.assertTrue(content_file.is_file())
        stored_bytes = content_file.read_bytes()
        self.assertEqual(stored_bytes, raw_binary_data)

        # Test string path variant
        src_file_txt = self.test_path / "notes_file.txt"
        src_file_txt.write_text("Hello from a text file!", encoding="utf-8")

        stdout_buf2 = io.StringIO()
        with redirect_stdout(stdout_buf2):
            result_txt = capture.capture_file(str(src_file_txt))

        content_txt_file = Path(result_txt.path) / "content.txt"
        self.assertTrue(content_txt_file.is_file())
        self.assertEqual(content_txt_file.read_text(encoding="utf-8"), "Hello from a text file!")

    def test_04_unique_ids_and_valid_iso_timestamps(self) -> None:
        """Requirement 4: Each function generates a unique ID and a valid ISO timestamp."""
        results = []
        stdout_buf = io.StringIO()

        with redirect_stdout(stdout_buf):
            results.append(capture.capture_note("Note A"))
            results.append(capture.capture_note("Note B"))
            results.append(capture.capture_link("https://example.com/1"))
            results.append(capture.capture_link("https://example.com/2"))

            temp_file = self.test_path / "temp.dat"
            temp_file.write_bytes(b"data1")
            results.append(capture.capture_file(temp_file))

            temp_file2 = self.test_path / "temp2.dat"
            temp_file2.write_bytes(b"data2")
            results.append(capture.capture_file(temp_file2))

        # Check unique IDs
        ids = [r.id for r in results]
        self.assertEqual(len(ids), len(set(ids)), "Capture IDs must be unique across all captures")

        # Validate ID format (YYYY-MM-DD_{8_hex})
        for cid in ids:
            parts = cid.split("_")
            self.assertEqual(len(parts), 2, f"ID {cid} does not match YYYY-MM-DD_{{uuid8}}")
            date_part, uuid_part = parts
            datetime.strptime(date_part, "%Y-%m-%d")
            self.assertEqual(len(uuid_part), 8)

        # Validate timestamps in meta.json
        for r in results:
            meta_path = Path(r.path) / "meta.json"
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)

            ts_str = meta["timestamp"]
            # fromisoformat must successfully parse it
            dt = datetime.fromisoformat(ts_str)
            self.assertIsNotNone(dt.tzinfo, "Timestamp must be timezone-aware (UTC)")

    def test_05_stores_correct_capture_type_and_content_hash(self) -> None:
        """Requirement 5: Each function stores the correct capture type and content hash."""
        stdout_buf = io.StringIO()

        # Note
        with redirect_stdout(stdout_buf):
            res_note = capture.capture_note("Hash testing text")
        meta_note = json.loads((Path(res_note.path) / "meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta_note["type"], "note")
        self.assertEqual(
            meta_note["content_hash"],
            hashlib.sha256("Hash testing text".encode("utf-8")).hexdigest(),
        )

        # Link
        url = "https://test.domain/path"
        notes = "Notes for hash verification"
        with redirect_stdout(stdout_buf):
            res_link = capture.capture_link(url, notes=notes)
        meta_link = json.loads((Path(res_link.path) / "meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta_link["type"], "link")
        expected_link_content = f"{url}\n\n{notes}"
        self.assertEqual(
            meta_link["content_hash"],
            hashlib.sha256(expected_link_content.encode("utf-8")).hexdigest(),
        )

        # File
        data_bytes = b"Special binary payload \xde\xad\xbe\xef"
        test_file = self.test_path / "payload.bin"
        test_file.write_bytes(data_bytes)
        with redirect_stdout(stdout_buf):
            res_file = capture.capture_file(test_file)
        meta_file = json.loads((Path(res_file.path) / "meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta_file["type"], "file")
        self.assertEqual(
            meta_file["content_hash"],
            hashlib.sha256(data_bytes).hexdigest(),
        )

    def test_06_printed_confirmation_format(self) -> None:
        """Requirement 6: The printed confirmation contains the generated raw capture path."""
        stdout_buf = io.StringIO()

        with redirect_stdout(stdout_buf):
            res = capture.capture_note("Hello confirmation")

        output = stdout_buf.getvalue().strip()
        expected = f"Captured → raw/{res.id}"
        self.assertEqual(output, expected)

        # Also test link confirmation
        stdout_buf2 = io.StringIO()
        with redirect_stdout(stdout_buf2):
            res_link = capture.capture_link("https://example.com")
        self.assertEqual(stdout_buf2.getvalue().strip(), f"Captured → raw/{res_link.id}")

        # Also test file confirmation
        test_file = self.test_path / "hello.txt"
        test_file.write_text("hello", encoding="utf-8")
        stdout_buf3 = io.StringIO()
        with redirect_stdout(stdout_buf3):
            res_file = capture.capture_file(test_file)
        self.assertEqual(stdout_buf3.getvalue().strip(), f"Captured → raw/{res_file.id}")

    def test_compatibility_with_storage_read_raw_captures(self) -> None:
        """Ensure written captures can be read by lib.storage.read_raw_captures."""
        stdout_buf = io.StringIO()
        with redirect_stdout(stdout_buf):
            res1 = capture.capture_note("Note for read test")
            res2 = capture.capture_link("https://example.com/read", notes="Read link")
            f = self.test_path / "read_test.pdf"
            f.write_bytes(b"%PDF dummy")
            res3 = capture.capture_file(f)

        captures = storage.read_raw_captures()
        capture_ids = {c.id for c in captures}
        self.assertIn(res1.id, capture_ids)
        self.assertIn(res2.id, capture_ids)
        self.assertIn(res3.id, capture_ids)

        for c in captures:
            self.assertIsInstance(c, CaptureMeta)
            self.assertIn(c.type, {"note", "link", "file"})
            self.assertTrue(c.content_hash)


if __name__ == "__main__":
    unittest.main()

