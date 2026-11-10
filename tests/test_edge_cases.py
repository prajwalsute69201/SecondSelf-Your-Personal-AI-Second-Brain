"""Unit tests for Task 1.3 edge cases in SecondSelf capture.py.

Tests:
- Missing file returns exit code 1 without creating partial captures.
- Directory path rejected without creating raw folders.
- Unreadable file handled gracefully.
- Empty and whitespace-only notes rejected inside capture_note() and CLI.
- Meaningful whitespace in notes preserved exactly without stripping.
- Binary files copied byte-for-byte with original_filename and extension preserved.
- Duplicate detection for notes, links, and files (warns on stderr, saves both).
- Duplicate detection works even when earlier capture is marked processed in index.json.
- Distinct content produces no duplicate warnings.
"""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

# Ensure secondself is in sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import capture
from lib.models import CaptureMeta, CaptureResult
import lib.storage as storage


class TestCaptureEdgeCases(unittest.TestCase):
    """Test suite covering edge cases for capture.py (Task 1.3)."""

    def setUp(self) -> None:
        """Create an isolated temporary directory for raw storage and index."""
        self.test_dir = tempfile.mkdtemp(prefix="secondself_edge_test_")
        self.test_path = Path(self.test_dir)
        storage.set_project_root(self.test_path)

    def tearDown(self) -> None:
        """Restore storage project root and clean up temporary directory."""
        storage.set_project_root(None)
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _get_raw_capture_folders(self) -> list[Path]:
        """Return list of capture subdirectories under the isolated raw/ directory."""
        raw_dir = storage.get_raw_dir()
        if not raw_dir.exists():
            return []
        return [
            item
            for item in raw_dir.iterdir()
            if item.is_dir() and not item.name.startswith(".")
        ]

    # --- Edge Case 1: Missing and invalid files ---

    def test_missing_file_rejected_and_cli_exit_code_1(self) -> None:
        """Verify missing file raises FileNotFoundError and CLI returns exit code 1."""
        missing_path = self.test_path / "does_not_exist.pdf"

        # Direct function call must raise FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            capture.capture_file(missing_path)

        self.assertEqual(len(self._get_raw_capture_folders()), 0)

        # CLI call must print error to stderr and return 1
        stderr_buf = io.StringIO()
        with redirect_stderr(stderr_buf):
            rc = capture.main(["file", str(missing_path)])

        self.assertEqual(rc, 1)
        err = stderr_buf.getvalue()
        self.assertIn("Error:", err)
        self.assertIn("File not found", err)
        self.assertIn(str(missing_path), err)
        self.assertEqual(len(self._get_raw_capture_folders()), 0)

    def test_directory_path_rejected_without_creating_capture(self) -> None:
        """Verify passing a directory path is rejected and creates no capture."""
        sub_dir = self.test_path / "a_directory"
        sub_dir.mkdir(parents=True, exist_ok=True)

        with self.assertRaises(ValueError):
            capture.capture_file(sub_dir)

        self.assertEqual(len(self._get_raw_capture_folders()), 0)

        stderr_buf = io.StringIO()
        with redirect_stderr(stderr_buf):
            rc = capture.main(["file", str(sub_dir)])

        self.assertEqual(rc, 1)
        err = stderr_buf.getvalue()
        self.assertIn("Error:", err)
        self.assertIn("not a regular file", err)
        self.assertEqual(len(self._get_raw_capture_folders()), 0)

    def test_unreadable_file_handled_gracefully(self) -> None:
        """Verify unreadable file raises OSError and CLI exits with code 1 without partial capture."""
        dummy_file = self.test_path / "unreadable.txt"
        dummy_file.write_text("content", encoding="utf-8")

        with patch.object(Path, "read_bytes", side_effect=PermissionError("Permission denied")):
            with self.assertRaises(OSError):
                capture.capture_file(dummy_file)

            stderr_buf = io.StringIO()
            with redirect_stderr(stderr_buf):
                rc = capture.main(["file", str(dummy_file)])

            self.assertEqual(rc, 1)
            self.assertIn("Cannot read file", stderr_buf.getvalue())
            self.assertEqual(len(self._get_raw_capture_folders()), 0)

    # --- Edge Case 2: Empty notes ---

    def test_empty_note_rejected_in_function_and_cli(self) -> None:
        """Verify empty string is rejected inside capture_note() and in CLI."""
        with self.assertRaises(ValueError):
            capture.capture_note("")

        self.assertEqual(len(self._get_raw_capture_folders()), 0)

        stderr_buf = io.StringIO()
        with redirect_stderr(stderr_buf):
            rc = capture.main(["note", ""])

        self.assertEqual(rc, 1)
        self.assertIn("Note text cannot be empty", stderr_buf.getvalue())
        self.assertEqual(len(self._get_raw_capture_folders()), 0)

    def test_whitespace_only_note_rejected_without_creating_capture(self) -> None:
        """Verify whitespace-only strings (spaces, tabs, newlines) are rejected."""
        whitespace_cases = ["   ", "\t", "\n\n  \t\n", "   \r\n   "]
        for ws in whitespace_cases:
            with self.assertRaises(ValueError):
                capture.capture_note(ws)
            self.assertEqual(len(self._get_raw_capture_folders()), 0)

            stderr_buf = io.StringIO()
            with redirect_stderr(stderr_buf):
                rc = capture.main(["note", ws])
            self.assertEqual(rc, 1)
            self.assertIn("Note text cannot be empty", stderr_buf.getvalue())
            self.assertEqual(len(self._get_raw_capture_folders()), 0)

    def test_empty_stdin_rejected_in_cli(self) -> None:
        """Verify empty and whitespace-only stdin input is rejected in CLI."""
        stdin_stream = io.StringIO("")
        stdin_stream.isatty = lambda: False  # type: ignore[assignment]

        stderr_buf = io.StringIO()
        with patch("sys.stdin", stdin_stream), redirect_stderr(stderr_buf):
            rc = capture.main([])

        self.assertEqual(rc, 1)
        self.assertIn("Note text cannot be empty", stderr_buf.getvalue())
        self.assertEqual(len(self._get_raw_capture_folders()), 0)

    def test_valid_note_preserves_meaningful_whitespace(self) -> None:
        """Verify valid note content preserves leading/trailing meaningful whitespace."""
        indented_note = "    def hello():\n        return 'world'\n"
        stdout_buf = io.StringIO()
        with redirect_stdout(stdout_buf):
            result = capture.capture_note(indented_note)

        capture_dir = Path(result.path)
        content_file = capture_dir / "content.md"
        self.assertEqual(content_file.read_text(encoding="utf-8"), indented_note)

    # --- Edge Case 3: Binary files ---

    def test_binary_file_preserved_byte_for_byte(self) -> None:
        """Verify arbitrary binary bytes are preserved byte-for-byte with original_filename and extension."""
        # Non-UTF8 payload with null bytes, high bytes, and PDF header
        binary_payload = b"%PDF-1.7\x00\xff\xfe\x01\x02\x80\xaa\x55\xef\xbb\xbf\x00\x00\xff"
        src_file = self.test_path / "document.PDF"
        src_file.write_bytes(binary_payload)

        stdout_buf = io.StringIO()
        with redirect_stdout(stdout_buf):
            result = capture.capture_file(src_file)

        self.assertEqual(result.type, "file")
        capture_dir = Path(result.path)

        # Content file should preserve lowercase .pdf extension
        stored_content_file = capture_dir / "content.pdf"
        self.assertTrue(stored_content_file.is_file())
        self.assertEqual(stored_content_file.read_bytes(), binary_payload)

        # Meta file must record original basename
        meta_file = capture_dir / "meta.json"
        meta_data = json.loads(meta_file.read_text(encoding="utf-8"))
        self.assertEqual(meta_data["original_filename"], "document.PDF")
        self.assertEqual(meta_data["type"], "file")
        self.assertEqual(meta_data["content_hash"], storage.content_hash(binary_payload))

    # --- Edge Case 4: Duplicate detection ---

    def test_duplicate_notes_warn_and_both_saved(self) -> None:
        """Verify duplicate note prints warning to stderr, but saves both captures with distinct IDs."""
        note_text = "Important note to be duplicated"

        stdout1, stderr1 = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout1), redirect_stderr(stderr1):
            res1 = capture.capture_note(note_text)

        # First capture should produce no warning
        self.assertEqual(stderr1.getvalue(), "")
        self.assertIn(f"Captured → raw/{res1.id}", stdout1.getvalue())

        stdout2, stderr2 = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout2), redirect_stderr(stderr2):
            res2 = capture.capture_note(note_text)

        # Second capture must print warning to stderr
        err_out = stderr2.getvalue()
        self.assertIn("Warning:", err_out)
        self.assertIn(f"raw/{res1.id}", err_out)
        self.assertIn(storage.content_hash(note_text), err_out)

        # Both captures must exist with distinct IDs
        self.assertNotEqual(res1.id, res2.id)
        self.assertTrue(Path(res1.path).is_dir())
        self.assertTrue(Path(res2.path).is_dir())
        self.assertEqual(len(self._get_raw_capture_folders()), 2)

    def test_duplicate_links_warn_and_both_saved(self) -> None:
        """Verify duplicate link with identical notes warns but creates new capture."""
        url = "https://example.com/duplicate_link"
        notes = "User notes for link duplicate test"

        stdout1, stderr1 = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout1), redirect_stderr(stderr1):
            res1 = capture.capture_link(url, notes=notes)

        self.assertEqual(stderr1.getvalue(), "")

        stdout2, stderr2 = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout2), redirect_stderr(stderr2):
            res2 = capture.capture_link(url, notes=notes)

        # Second capture warns
        err_out = stderr2.getvalue()
        self.assertIn("Warning:", err_out)
        self.assertIn(f"raw/{res1.id}", err_out)

        self.assertNotEqual(res1.id, res2.id)
        self.assertTrue(Path(res1.path).is_dir())
        self.assertTrue(Path(res2.path).is_dir())

    def test_duplicate_files_warn_and_both_saved(self) -> None:
        """Verify duplicate files (same bytes, even different names) warn and save both."""
        payload = b"Binary payload for file duplicate check"
        f1 = self.test_path / "file1.bin"
        f2 = self.test_path / "file2.bin"
        f1.write_bytes(payload)
        f2.write_bytes(payload)

        stdout1, stderr1 = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout1), redirect_stderr(stderr1):
            res1 = capture.capture_file(f1)
        self.assertEqual(stderr1.getvalue(), "")

        stdout2, stderr2 = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout2), redirect_stderr(stderr2):
            res2 = capture.capture_file(f2)

        err_out = stderr2.getvalue()
        self.assertIn("Warning:", err_out)
        self.assertIn(f"raw/{res1.id}", err_out)
        self.assertNotEqual(res1.id, res2.id)
        self.assertTrue(Path(res1.path).is_dir())
        self.assertTrue(Path(res2.path).is_dir())

    def test_duplicate_detected_when_earlier_capture_is_processed(self) -> None:
        """Verify duplicate detection finds earlier capture even if marked processed in index.json."""
        text = "Capture that gets processed before duplicate attempt"

        stdout1, stderr1 = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout1), redirect_stderr(stderr1):
            res1 = capture.capture_note(text)

        # Mark res1 as processed in data/index.json
        index = storage.load_index()
        meta1 = json.loads((Path(res1.path) / "meta.json").read_text(encoding="utf-8"))
        index["raw_processed"][res1.id] = {
            "wiki_id": "test_wiki_id",
            "classified_at": "2026-11-08T12:00:00Z",
            "content_hash": meta1["content_hash"],
        }
        storage.save_index(index)

        # Verify that read_raw_captures() now excludes res1 (proves it filters processed items)
        unprocessed = storage.read_raw_captures()
        self.assertNotIn(res1.id, [c.id for c in unprocessed])

        # Attempt second capture of the same text
        stdout2, stderr2 = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout2), redirect_stderr(stderr2):
            res2 = capture.capture_note(text)

        # Duplicate must STILL be detected and warned!
        err_out = stderr2.getvalue()
        self.assertIn("Warning:", err_out)
        self.assertIn(f"raw/{res1.id}", err_out)

        # Both captures exist
        self.assertNotEqual(res1.id, res2.id)
        self.assertTrue(Path(res1.path).is_dir())
        self.assertTrue(Path(res2.path).is_dir())

    def test_distinct_content_produces_no_warning(self) -> None:
        """Verify distinct note content does not emit any duplicate warning."""
        stdout1, stderr1 = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout1), redirect_stderr(stderr1):
            capture.capture_note("First unique note")

        stdout2, stderr2 = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout2), redirect_stderr(stderr2):
            capture.capture_note("Second completely different note")

        self.assertEqual(stderr1.getvalue(), "")
        self.assertEqual(stderr2.getvalue(), "")


if __name__ == "__main__":
    unittest.main()

