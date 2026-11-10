"""Unit tests for classify.py and classification workflow (Task 2.1.4).

Validates:
- Structured PARA classification via LLM for all four categories:
  'Projects', 'Areas', 'Resources', 'Archives'.
- Parsing of clean JSON, fenced markdown JSON, and case-normalized categories.
- Rejection of invalid PARA categories, malformed JSON, and missing fields.
- Rejection of empty content input.
- End-to-end capture classification for notes, links, and files.
- Frontmatter integrity and body preservation in wiki notes.
- Updating and persistence of data/index.json raw_processed tracking.
- Preservation of raw capture content without modification or deletion.
- Skipping of already-processed captures (idempotency).
- Failed classifications do NOT write wiki notes or mark index.json as processed.
- CLI argument parsing, help output, dry-run mode, and error handling.
"""

from datetime import datetime, timezone
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import classify
from lib.extract import extract_text
import lib.llm as llm
from lib.models import CaptureMeta, WikiNote
import lib.storage as storage
import pypdf


def _build_test_pdf(text: str) -> bytes:
    """Build a valid minimal PDF byte stream containing the specified text."""
    clean_text = text.replace("(", "\\(").replace(")", "\\)")
    stream_data = f"BT /F1 12 Tf 50 700 Td ({clean_text}) Tj ET".encode("latin-1")
    stream_len = len(stream_data)

    body = io.BytesIO()
    body.write(b"%PDF-1.4\n")
    offsets = [0]

    def _write_obj(num: int, content: bytes) -> None:
        offsets.append(body.tell())
        body.write(f"{num} 0 obj\n".encode("ascii") + content + b"\nendobj\n")

    _write_obj(1, b"<</Type/Catalog/Pages 2 0 R>>")
    _write_obj(2, b"<</Type/Pages/Kids[4 0 R]/Count 1>>")
    _write_obj(3, b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>")
    _write_obj(
        4,
        b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 5 0 R"
        b"/Resources<</Font<</F1 3 0 R>>>>>>",
    )
    _write_obj(5, f"<</Length {stream_len}>>\nstream\n".encode("ascii") + stream_data + b"\nendstream")

    xref_offset = body.tell()
    body.write(f"xref\n0 {len(offsets)}\n".encode("ascii"))
    body.write(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        body.write(f"{off:010d} 00000 n \n".encode("ascii"))

    body.write(f"trailer\n<</Size {len(offsets)}/Root 1 0 R>>\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii"))
    return body.getvalue()


class TestClassifyModule(unittest.TestCase):
    """Test suite for classify.py and classification logic."""

    def setUp(self) -> None:
        """Create isolated temporary test workspace with raw/, wiki/, and data/."""
        self.test_dir = tempfile.mkdtemp(prefix="secondself_classify_test_")
        self.test_path = Path(self.test_dir)
        storage.set_project_root(self.test_path)

        # Initialize folders
        (self.test_path / "raw").mkdir(parents=True, exist_ok=True)
        for cat in ("Projects", "Areas", "Resources", "Archives"):
            (self.test_path / "wiki" / cat).mkdir(parents=True, exist_ok=True)
        (self.test_path / "data").mkdir(parents=True, exist_ok=True)

        # Initialize default index.json
        storage.save_index(
            {
                "raw_processed": {},
                "embeddings_version": "all-MiniLM-L6-v2",
                "last_graph_build": None,
            },
            root=self.test_path,
        )

    def tearDown(self) -> None:
        """Clean up temporary test workspace."""
        storage.set_project_root(None)
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # 1. classify_content and JSON Schema Validation Tests
    # -------------------------------------------------------------------------

    def test_01_classify_content_all_para_categories(self) -> None:
        """classify_content correctly handles all 4 valid PARA categories."""
        categories = ["Projects", "Areas", "Resources", "Archives"]

        for cat in categories:
            mock_json = json.dumps({
                "para": cat,
                "tags": [cat.lower(), "test"],
                "summary": f"Summary for {cat}",
            })
            with patch("lib.llm.call_llm", return_value=mock_json):
                result = llm.classify_content(f"Sample note content for {cat}")
                self.assertEqual(result["para"], cat)
                self.assertEqual(result["tags"], [cat.lower(), "test"])
                self.assertEqual(result["summary"], f"Summary for {cat}")

    def test_02_classify_content_handles_markdown_code_fences(self) -> None:
        """classify_content successfully parses JSON enclosed in markdown code fences."""
        mock_output = (
            "Here is the classification:\n```json\n"
            "{\n  \"para\": \"Resources\",\n  \"tags\": [\"docs\", \"guide\"],\n"
            "  \"summary\": \"API reference guide\"\n}\n```"
        )
        with patch("lib.llm.call_llm", return_value=mock_output):
            result = llm.classify_content("Extracted content text")
            self.assertEqual(result["para"], "Resources")
            self.assertEqual(result["tags"], ["docs", "guide"])
            self.assertEqual(result["summary"], "API reference guide")

    def test_03_classify_content_normalizes_category_casing(self) -> None:
        """classify_content normalizes lowercase or mixed-case PARA category names."""
        mock_json = json.dumps({
            "para": "projects",
            "tags": ["alpha"],
            "summary": "Project summary",
        })
        with patch("lib.llm.call_llm", return_value=mock_json):
            result = llm.classify_content("Sample project text")
            self.assertEqual(result["para"], "Projects")

    def test_04_classify_content_rejects_invalid_para_category(self) -> None:
        """classify_content rejects invalid categories without silently defaulting."""
        mock_invalid = json.dumps({
            "para": "WorkRelated",
            "tags": ["work"],
            "summary": "Invalid category test",
        })
        with patch("lib.llm.call_llm", return_value=mock_invalid):
            with self.assertRaises(ValueError) as cm:
                llm.classify_content("Sample text", max_retries=1)
            self.assertIn("Invalid PARA category", str(cm.exception))

    def test_05_classify_content_rejects_malformed_json(self) -> None:
        """classify_content raises ValueError when LLM outputs unparseable text."""
        with patch("lib.llm.call_llm", return_value="This is not JSON at all."):
            with self.assertRaises(ValueError) as cm:
                llm.classify_content("Sample text", max_retries=1)
            self.assertIn("Malformed JSON", str(cm.exception))

    def test_06_classify_content_rejects_missing_required_fields(self) -> None:
        """classify_content raises ValueError if required fields are missing."""
        # Missing summary
        mock_no_summary = json.dumps({"para": "Projects", "tags": ["code"]})
        with patch("lib.llm.call_llm", return_value=mock_no_summary):
            with self.assertRaises(ValueError) as cm:
                llm.classify_content("Sample text", max_retries=1)
            self.assertIn("missing required 'summary' field", str(cm.exception))

        # Missing para
        mock_no_para = json.dumps({"tags": ["code"], "summary": "Note summary"})
        with patch("lib.llm.call_llm", return_value=mock_no_para):
            with self.assertRaises(ValueError) as cm:
                llm.classify_content("Sample text", max_retries=1)
            self.assertIn("missing required 'para' field", str(cm.exception))

    def test_07_classify_content_rejects_empty_input_text(self) -> None:
        """classify_content raises ValueError on empty or whitespace input."""
        with self.assertRaises(ValueError):
            llm.classify_content("")

        with self.assertRaises(ValueError):
            llm.classify_content("   \n\t  ")

    def test_08_classify_content_truncates_long_input(self) -> None:
        """classify_content truncates long input text to prevent exceeding token limits."""
        huge_text = "Word " * 10000  # ~50,000 characters
        mock_json = json.dumps({
            "para": "Resources",
            "tags": ["bigdata"],
            "summary": "Large document",
        })
        with patch("lib.llm.call_llm", return_value=mock_json) as mock_call:
            result = llm.classify_content(huge_text)
            self.assertEqual(result["para"], "Resources")
            # Verify the prompt sent to call_llm was truncated
            sent_prompt = mock_call.call_args[1]["prompt"]
            self.assertLess(len(sent_prompt), 20000)
            self.assertIn("...[content truncated for classification]...", sent_prompt)

    # -------------------------------------------------------------------------
    # 2. classify_capture End-to-End Workflow Tests
    # -------------------------------------------------------------------------

    def test_09_classify_note_capture_creates_wiki_and_updates_index(self) -> None:
        """Classify a raw note capture, verify wiki file written and index updated."""
        # Setup raw capture
        cap_id = "2026-11-08_a1b2c3d4"
        cap_dir = self.test_path / "raw" / cap_id
        cap_dir.mkdir(parents=True)
        meta = CaptureMeta(
            id=cap_id,
            timestamp="2026-11-08T10:00:00Z",
            type="note",
            source="cli",
            original_filename=None,
            content_hash="sha256:abc123hash",
        )
        (cap_dir / "meta.json").write_text(json.dumps(meta.__dict__), encoding="utf-8")
        note_text = "# Machine Learning Pivot\nPlan to learn PyTorch by Q2."
        (cap_dir / "content.md").write_text(note_text, encoding="utf-8")

        mock_llm_json = json.dumps({
            "para": "Projects",
            "tags": ["ml", "career"],
            "summary": "ML engineering learning plan",
        })

        with patch("lib.llm.call_llm", return_value=mock_llm_json):
            note = classify.classify_capture(meta, root=self.test_path)

        # 1. Verify returned WikiNote
        self.assertEqual(note.id, "a1b2c3d4")
        self.assertEqual(note.raw_id, cap_id)
        self.assertEqual(note.para, "Projects")
        self.assertEqual(note.tags, ["ml", "career"])
        self.assertEqual(note.summary, "ML engineering learning plan")
        self.assertEqual(note.links, [])
        self.assertEqual(note.body, note_text)

        # 2. Verify wiki file written on disk
        wiki_file = self.test_path / "wiki" / "Projects" / "a1b2c3d4.md"
        self.assertTrue(wiki_file.exists())
        wiki_content = wiki_file.read_text(encoding="utf-8")
        self.assertIn("id: a1b2c3d4", wiki_content)
        self.assertIn("raw_id: 2026-11-08_a1b2c3d4", wiki_content)
        self.assertIn("para: Projects", wiki_content)
        self.assertIn("summary: ML engineering learning plan", wiki_content)
        self.assertIn(note_text, wiki_content)

        # 3. Verify index.json updated
        index = storage.load_index(root=self.test_path)
        self.assertIn(cap_id, index["raw_processed"])
        proc_data = index["raw_processed"][cap_id]
        self.assertEqual(proc_data["wiki_id"], "a1b2c3d4")
        self.assertEqual(proc_data["para"], "Projects")
        self.assertEqual(proc_data["content_hash"], "sha256:abc123hash")
        self.assertIn("classified_at", proc_data)

        # 4. Verify raw capture files were NOT modified or deleted
        self.assertTrue((cap_dir / "meta.json").exists())
        self.assertTrue((cap_dir / "content.md").exists())
        self.assertEqual((cap_dir / "content.md").read_text(encoding="utf-8"), note_text)

    def test_10_classify_link_capture_with_html_content(self) -> None:
        """Classify a raw link capture with web text and user notes."""
        cap_id = "2026-11-08_link0001"
        cap_dir = self.test_path / "raw" / cap_id
        cap_dir.mkdir(parents=True)
        meta = CaptureMeta(
            id=cap_id,
            timestamp="2026-11-08T11:00:00Z",
            type="link",
            source="https://example.com/guide",
            original_filename=None,
            content_hash="sha256:linkhash123",
        )
        (cap_dir / "meta.json").write_text(json.dumps(meta.__dict__), encoding="utf-8")
        (cap_dir / "content.txt").write_text(
            "https://example.com/guide\n\nMust check out section 4.",
            encoding="utf-8",
        )

        mock_html = (
            "<html><head><title>Guide</title></head>"
            "<body><h1>Documentation Guide</h1><p>Important reference material.</p></body></html>"
        )
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = mock_html
        mock_response.headers = {"content-type": "text/html"}
        mock_response.raise_for_status = MagicMock()

        mock_llm_json = json.dumps({
            "para": "Resources",
            "tags": ["docs", "reference"],
            "summary": "Online documentation guide",
        })

        with patch("requests.get", return_value=mock_response), \
             patch("lib.llm.call_llm", return_value=mock_llm_json):
            note = classify.classify_capture(meta, root=self.test_path)

        self.assertEqual(note.para, "Resources")
        self.assertEqual(note.id, "link0001")
        self.assertIn("Documentation Guide", note.body)
        self.assertIn("Must check out section 4.", note.body)

        wiki_file = self.test_path / "wiki" / "Resources" / "link0001.md"
        self.assertTrue(wiki_file.exists())

    def test_11_classify_pdf_file_capture(self) -> None:
        """Classify a raw PDF file capture."""
        cap_id = "2026-11-08_file0001"
        cap_dir = self.test_path / "raw" / cap_id
        cap_dir.mkdir(parents=True)
        meta = CaptureMeta(
            id=cap_id,
            timestamp="2026-11-08T12:00:00Z",
            type="file",
            source="documents/resume.pdf",
            original_filename="resume.pdf",
            content_hash="sha256:pdfhash123",
        )
        (cap_dir / "meta.json").write_text(json.dumps(meta.__dict__), encoding="utf-8")

        pdf_bytes = _build_test_pdf("John Doe Software Engineer Resume and Work Experience")
        (cap_dir / "content.pdf").write_bytes(pdf_bytes)

        mock_llm_json = json.dumps({
            "para": "Areas",
            "tags": ["resume", "career"],
            "summary": "John Doe professional resume",
        })

        with patch("lib.llm.call_llm", return_value=mock_llm_json):
            note = classify.classify_capture(meta, root=self.test_path)

        self.assertEqual(note.para, "Areas")
        self.assertEqual(note.id, "file0001")
        wiki_file = self.test_path / "wiki" / "Areas" / "file0001.md"
        self.assertTrue(wiki_file.exists())

    def test_12_already_processed_capture_raises_and_is_skipped(self) -> None:
        """Already processed captures are skipped and reject redundant classification."""
        cap_id = "2026-11-08_done0001"
        index = storage.load_index(root=self.test_path)
        index["raw_processed"][cap_id] = {
            "wiki_id": "done0001",
            "para": "Projects",
            "classified_at": "2026-11-08T09:00:00Z",
            "content_hash": "sha256:donehash",
        }
        storage.save_index(index, root=self.test_path)

        meta = CaptureMeta(
            id=cap_id,
            timestamp="2026-11-08T08:00:00Z",
            type="note",
            source="cli",
            original_filename=None,
            content_hash="sha256:donehash",
        )

        with self.assertRaises(ValueError) as cm:
            classify.classify_capture(meta, root=self.test_path)
        self.assertIn("already been processed", str(cm.exception))

    def test_13_failed_classification_does_not_update_index_or_wiki(self) -> None:
        """If classification fails, wiki file is not created and index is NOT marked processed."""
        cap_id = "2026-11-08_fail0001"
        cap_dir = self.test_path / "raw" / cap_id
        cap_dir.mkdir(parents=True)
        meta = CaptureMeta(
            id=cap_id,
            timestamp="2026-11-08T13:00:00Z",
            type="note",
            source="cli",
            original_filename=None,
            content_hash="sha256:failhash",
        )
        (cap_dir / "meta.json").write_text(json.dumps(meta.__dict__), encoding="utf-8")
        (cap_dir / "content.md").write_text("Valid note text.", encoding="utf-8")

        # Mock LLM API failure
        with patch("lib.llm.call_llm", side_effect=RuntimeError("Groq API rate limit exceeded")):
            with self.assertRaises(RuntimeError):
                classify.classify_capture(meta, root=self.test_path)

        # Confirm NOT in raw_processed
        index = storage.load_index(root=self.test_path)
        self.assertNotIn(cap_id, index["raw_processed"])

        # Confirm no wiki note was written anywhere
        wiki_notes = list((self.test_path / "wiki").rglob("*.md"))
        self.assertEqual(len(wiki_notes), 0)

    def test_14_dry_run_does_not_write_files_or_update_index(self) -> None:
        """Dry-run returns WikiNote without writing wiki files or modifying index.json."""
        cap_id = "2026-11-08_dry0001"
        cap_dir = self.test_path / "raw" / cap_id
        cap_dir.mkdir(parents=True)
        meta = CaptureMeta(
            id=cap_id,
            timestamp="2026-11-08T14:00:00Z",
            type="note",
            source="cli",
            original_filename=None,
            content_hash="sha256:dryhash",
        )
        (cap_dir / "meta.json").write_text(json.dumps(meta.__dict__), encoding="utf-8")
        (cap_dir / "content.md").write_text("Dry run note.", encoding="utf-8")

        mock_llm_json = json.dumps({
            "para": "Resources",
            "tags": ["dry"],
            "summary": "Dry run summary",
        })

        with patch("lib.llm.call_llm", return_value=mock_llm_json):
            note = classify.classify_capture(meta, root=self.test_path, dry_run=True)

        self.assertEqual(note.id, "dry0001")
        self.assertEqual(note.para, "Resources")

        # Verify no wiki files created
        wiki_notes = list((self.test_path / "wiki").rglob("*.md"))
        self.assertEqual(len(wiki_notes), 0)

        # Verify index.json was not updated
        index = storage.load_index(root=self.test_path)
        self.assertNotIn(cap_id, index["raw_processed"])

    # -------------------------------------------------------------------------
    # 3. classify_all and Batch Processing Tests
    # -------------------------------------------------------------------------

    def test_15_classify_all_handles_batch_with_partial_failures(self) -> None:
        """classify_all processes valid items and reports failures without crashing."""
        # Setup 2 captures: 1 valid, 1 failing
        c1 = "2026-11-08_good0001"
        c2 = "2026-11-08_bad0002"

        d1 = self.test_path / "raw" / c1
        d1.mkdir(parents=True)
        m1 = CaptureMeta(c1, "2026-11-08T15:00:00Z", "note", "cli", None, "sha256:h1")
        (d1 / "meta.json").write_text(json.dumps(m1.__dict__), encoding="utf-8")
        (d1 / "content.md").write_text("Good note text", encoding="utf-8")

        d2 = self.test_path / "raw" / c2
        d2.mkdir(parents=True)
        m2 = CaptureMeta(c2, "2026-11-08T15:01:00Z", "note", "cli", None, "sha256:h2")
        (d2 / "meta.json").write_text(json.dumps(m2.__dict__), encoding="utf-8")
        (d2 / "content.md").write_text("Bad note text", encoding="utf-8")

        def mock_llm_responder(prompt: str, **kwargs: object) -> str:
            if "Good note text" in prompt:
                return json.dumps({"para": "Projects", "tags": ["good"], "summary": "Good summary"})
            raise RuntimeError("API failure for bad item")

        with patch("lib.llm.call_llm", side_effect=mock_llm_responder):
            successful, failed = classify.classify_all(root=self.test_path)

        self.assertEqual(len(successful), 1)
        self.assertEqual(successful[0].raw_id, c1)

        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0][0], c2)

        # Verify c1 is indexed, c2 is NOT indexed
        index = storage.load_index(root=self.test_path)
        self.assertIn(c1, index["raw_processed"])
        self.assertNotIn(c2, index["raw_processed"])

    def test_16_classify_all_no_pending_captures_returns_empty(self) -> None:
        """classify_all returns empty lists when raw/ has no unprocessed items."""
        successful, failed = classify.classify_all(root=self.test_path)
        self.assertEqual(successful, [])
        self.assertEqual(failed, [])

    # -------------------------------------------------------------------------
    # 4. CLI Execution Tests
    # -------------------------------------------------------------------------

    def test_17_cli_help_command(self) -> None:
        """classify.py --help prints usage instructions and exits code 0."""
        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            code = classify.main(["--help"])
        self.assertEqual(code, 0)
        self.assertIn("Classify unprocessed SecondSelf raw captures", mock_out.getvalue())

    def test_18_cli_no_pending_captures_exits_zero(self) -> None:
        """classify.py with no pending captures prints up-to-date message and exits 0."""
        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            code = classify.main(["--root", str(self.test_path)])
        self.assertEqual(code, 0)
        self.assertIn("No unprocessed captures found in raw/. Everything is up to date.", mock_out.getvalue())

    def test_19_cli_successful_run(self) -> None:
        """classify.py executes batch successfully and returns exit code 0."""
        cap_id = "2026-11-08_cli0001"
        cap_dir = self.test_path / "raw" / cap_id
        cap_dir.mkdir(parents=True)
        meta = CaptureMeta(cap_id, "2026-11-08T16:00:00Z", "note", "cli", None, "sha256:clihash")
        (cap_dir / "meta.json").write_text(json.dumps(meta.__dict__), encoding="utf-8")
        (cap_dir / "content.md").write_text("CLI test note", encoding="utf-8")

        mock_llm_json = json.dumps({
            "para": "Archives",
            "tags": ["archive"],
            "summary": "Archived note",
        })

        with patch("lib.llm.call_llm", return_value=mock_llm_json), \
             patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            code = classify.main(["--root", str(self.test_path)])

        self.assertEqual(code, 0)
        out = mock_out.getvalue()
        self.assertIn("Classified raw/2026-11-08_cli0001 → wiki/Archives/cli0001.md", out)
        self.assertIn("Classification complete: 1 processed, 0 failed.", out)

    def test_20_cli_batch_with_errors_exits_one(self) -> None:
        """classify.py with errors prints to stderr and returns exit code 1."""
        cap_id = "2026-11-08_err0001"
        cap_dir = self.test_path / "raw" / cap_id
        cap_dir.mkdir(parents=True)
        meta = CaptureMeta(cap_id, "2026-11-08T17:00:00Z", "note", "cli", None, "sha256:errhash")
        (cap_dir / "meta.json").write_text(json.dumps(meta.__dict__), encoding="utf-8")
        (cap_dir / "content.md").write_text("Error note", encoding="utf-8")

        with patch("lib.llm.call_llm", side_effect=RuntimeError("Groq quota exceeded")), \
             patch("sys.stdout", new_callable=io.StringIO), \
             patch("sys.stderr", new_callable=io.StringIO) as mock_err:
            code = classify.main(["--root", str(self.test_path)])

        self.assertEqual(code, 1)
        err_out = mock_err.getvalue()
        self.assertIn("Error classifying raw/2026-11-08_err0001", err_out)
        self.assertIn("0 processed, 1 failed", err_out)


if __name__ == "__main__":
    unittest.main()

