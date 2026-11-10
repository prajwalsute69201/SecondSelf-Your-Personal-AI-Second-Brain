"""Unit and integration tests for capture.py CLI (Task 1.2).

Verifies the argparse CLI subcommands (note, link, file), interactive and piped
stdin modes, help generation, error handling, and exit codes.
"""

from contextlib import redirect_stderr, redirect_stdout
import importlib
import io
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Ensure secondself is in sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import capture
import lib.storage as storage


class TestCaptureCLI(unittest.TestCase):
    """Test suite for capture.py argparse CLI."""

    def setUp(self) -> None:
        """Create an isolated temporary directory for storage testing."""
        self.test_dir = tempfile.mkdtemp(prefix="secondself_cli_test_")
        self.test_path = Path(self.test_dir)
        storage.set_project_root(self.test_path)

    def tearDown(self) -> None:
        """Restore storage root and clean up temporary directory."""
        storage.set_project_root(None)
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_note_command_calls_capture_note(self) -> None:
        """Test note command parses text and calls capture_note with exact text."""
        with patch.object(capture, "capture_note") as mock_note:
            mock_note.return_value = MagicMock()
            rc = capture.main(["note", "Remember to review embeddings paper"])

            self.assertEqual(rc, 0)
            mock_note.assert_called_once_with("Remember to review embeddings paper")

    def test_link_command_calls_capture_link(self) -> None:
        """Test link command parses URL and calls capture_link with exact URL."""
        with patch.object(capture, "capture_link") as mock_link:
            mock_link.return_value = MagicMock()
            rc = capture.main(["link", "https://arxiv.org/abs/2301.00001"])

            self.assertEqual(rc, 0)
            mock_link.assert_called_once_with("https://arxiv.org/abs/2301.00001")

    def test_file_command_calls_capture_file(self) -> None:
        """Test file command parses path and calls capture_file with supplied path."""
        with patch.object(capture, "capture_file") as mock_file:
            mock_file.return_value = MagicMock()
            rc = capture.main(["file", "./documents/resume.pdf"])

            self.assertEqual(rc, 0)
            mock_file.assert_called_once_with("./documents/resume.pdf")

    def test_no_arguments_interactive_stdin(self) -> None:
        """Test running without arguments displays prompt and calls capture_note."""
        stdin_stream = io.StringIO("Interactive note text")
        stdin_stream.isatty = lambda: True  # type: ignore[assignment]

        stdout_buf = io.StringIO()
        with patch("sys.stdin", stdin_stream), patch.object(capture, "capture_note") as mock_note:
            mock_note.return_value = MagicMock()
            with redirect_stdout(stdout_buf):
                rc = capture.main([])

            self.assertEqual(rc, 0)
            mock_note.assert_called_once_with("Interactive note text")

        output = stdout_buf.getvalue()
        self.assertIn("Enter note content", output)
        self.assertIn("Ctrl+Z", output)
        self.assertIn("Ctrl+D", output)

    def test_no_arguments_piped_stdin(self) -> None:
        """Test running with piped stdin skips interactive prompt and captures input."""
        stdin_stream = io.StringIO("An idea piped into SecondSelf\n")
        stdin_stream.isatty = lambda: False  # type: ignore[assignment]

        stdout_buf = io.StringIO()
        with patch("sys.stdin", stdin_stream), patch.object(capture, "capture_note") as mock_note:
            mock_note.return_value = MagicMock()
            with redirect_stdout(stdout_buf):
                rc = capture.main([])

            self.assertEqual(rc, 0)
            mock_note.assert_called_once_with("An idea piped into SecondSelf\n")

        # In piped mode, the interactive prompt should not be displayed
        self.assertNotIn("Enter note content", stdout_buf.getvalue())

    def test_multiline_stdin_preserved(self) -> None:
        """Test multiline stdin content is preserved verbatim."""
        multiline_text = (
            "# SecondSelf Brainstorm\n\n"
            "- Item 1: RAG retrieval\n"
            "- Item 2: Force-directed graph\n"
            "- Item 3: Local storage\n"
        )
        stdin_stream = io.StringIO(multiline_text)
        stdin_stream.isatty = lambda: False  # type: ignore[assignment]

        with patch("sys.stdin", stdin_stream), patch.object(capture, "capture_note") as mock_note:
            mock_note.return_value = MagicMock()
            rc = capture.main([])

            self.assertEqual(rc, 0)
            mock_note.assert_called_once_with(multiline_text)

    def test_help_messages(self) -> None:
        """Test top-level and subcommand --help output."""
        # Top-level --help
        stdout_buf = io.StringIO()
        with redirect_stdout(stdout_buf), self.assertRaises(SystemExit) as cm:
            capture.main(["--help"])
        self.assertEqual(cm.exception.code, 0)
        top_help = stdout_buf.getvalue()
        self.assertIn("note", top_help)
        self.assertIn("link", top_help)
        self.assertIn("file", top_help)
        self.assertIn("Ctrl+Z", top_help)

        # note --help
        stdout_buf = io.StringIO()
        with redirect_stdout(stdout_buf), self.assertRaises(SystemExit) as cm:
            capture.main(["note", "--help"])
        self.assertEqual(cm.exception.code, 0)
        self.assertIn("text", stdout_buf.getvalue())

        # link --help
        stdout_buf = io.StringIO()
        with redirect_stdout(stdout_buf), self.assertRaises(SystemExit) as cm:
            capture.main(["link", "--help"])
        self.assertEqual(cm.exception.code, 0)
        self.assertIn("url", stdout_buf.getvalue())

        # file --help
        stdout_buf = io.StringIO()
        with redirect_stdout(stdout_buf), self.assertRaises(SystemExit) as cm:
            capture.main(["file", "--help"])
        self.assertEqual(cm.exception.code, 0)
        self.assertIn("path", stdout_buf.getvalue())

    def test_capture_failure_outputs_stderr_and_nonzero_exit(self) -> None:
        """Test that exceptions during capture output to stderr and return code 1."""
        stderr_buf = io.StringIO()
        with patch.object(capture, "capture_file", side_effect=FileNotFoundError("File not found: missing.pdf")):
            with redirect_stderr(stderr_buf):
                rc = capture.main(["file", "missing.pdf"])

            self.assertEqual(rc, 1)
            self.assertIn("missing.pdf", stderr_buf.getvalue())

    def test_keyboard_interrupt_handled(self) -> None:
        """Test that KeyboardInterrupt during stdin returns exit code 130."""
        stderr_buf = io.StringIO()
        with patch.object(capture, "capture_note", side_effect=KeyboardInterrupt):
            with redirect_stderr(stderr_buf):
                rc = capture.main(["note", "interrupted"])

            self.assertEqual(rc, 130)
            self.assertIn("cancelled", stderr_buf.getvalue().lower())

    def test_import_does_not_execute_cli(self) -> None:
        """Test that importing capture does not execute the CLI entry point."""
        with patch.object(capture, "main") as mock_main:
            importlib.reload(capture)
            mock_main.assert_not_called()

    def test_subprocess_execution(self) -> None:
        """End-to-end integration test executing capture.py via subprocess."""
        python_exe = sys.executable

        # Run --help
        res = subprocess.run(
            [python_exe, "capture.py", "--help"],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0)
        self.assertIn("SecondSelf capture CLI", res.stdout)
        self.assertIn("note", res.stdout)
        self.assertIn("link", res.stdout)
        self.assertIn("file", res.stdout)


if __name__ == "__main__":
    unittest.main()

