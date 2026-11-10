"""Unit and integration tests for pipeline.py orchestrator (Task 2.2.4).

Validates:
- End-to-end pipeline execution with multiple pending raw captures.
- Empty raw input directory handling.
- Already-processed capture skipping (idempotency).
- Multi-stage coordination: classification followed by auto-linking.
- Handling and isolation of stage failures:
  - Text extraction failure.
  - LLM classification failure.
  - Sentence-transformers embedding failure.
  - Pairwise linking failure.
  - Partial batch failure (successful items proceed, failing items isolated).
- Idempotency on repeated runs (no duplicate wiki notes, links, or index corruption).
- Atomic processing state updates in data/index.json.
- Preservation of original raw capture files and existing wiki notes.
- Preservation of existing embeddings and unrelated links.
- Command-line interface: 'process', 'classify', 'link', '--dry-run', and '--help'.
"""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Ensure secondself is in sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import lib.embeddings as emb
from lib.models import CaptureMeta, WikiNote
import lib.storage as storage
import pipeline
import validate_graph


def _make_unit_vector(angle_rad: float, dim: int = 384) -> np.ndarray:
    """Generate a 384-dimensional unit vector in a 2D subspace for exact cosine similarities."""
    vec = np.zeros(dim, dtype=np.float32)
    vec[0] = np.cos(angle_rad)
    vec[1] = np.sin(angle_rad)
    return vec


def _create_mock_encoder(mapping: dict[str, np.ndarray] | None = None, dim: int = 384) -> MagicMock:
    """Create a mock SentenceTransformer that returns deterministic vectors."""
    mock = MagicMock()
    vec_map = mapping or {}

    def _encode(text, convert_to_numpy=True, show_progress_bar=False):
        if isinstance(text, str):
            for key, vec in vec_map.items():
                if key in text:
                    return np.asarray(vec, dtype=np.float32)
            rng = np.random.RandomState(abs(hash(text)) % (2**31))
            v = rng.randn(dim).astype(np.float32)
            norm = np.linalg.norm(v)
            return v / norm if norm > 0 else v
        elif isinstance(text, (list, tuple)):
            rows = []
            for t in text:
                matched = False
                for key, vec in vec_map.items():
                    if key in t:
                        rows.append(np.asarray(vec, dtype=np.float32))
                        matched = True
                        break
                if not matched:
                    rng = np.random.RandomState(abs(hash(t)) % (2**31))
                    v = rng.randn(dim).astype(np.float32)
                    norm = np.linalg.norm(v)
                    rows.append(v / norm if norm > 0 else v)
            return np.array(rows, dtype=np.float32)
        else:
            raise TypeError("Unsupported text type")

    mock.encode.side_effect = _encode
    return mock


class TestPipelineOrchestrator(unittest.TestCase):
    """Test suite for pipeline.py orchestrator."""

    def setUp(self) -> None:
        """Create isolated test directory and initialize standard structure."""
        self.test_dir = tempfile.mkdtemp(prefix="secondself_pipeline_test_")
        self.test_path = Path(self.test_dir)
        storage.set_project_root(self.test_path)

        # Create standard folders
        (self.test_path / "raw").mkdir(parents=True, exist_ok=True)
        (self.test_path / "data").mkdir(parents=True, exist_ok=True)
        (self.test_path / "wiki" / "Projects").mkdir(parents=True, exist_ok=True)
        (self.test_path / "wiki" / "Areas").mkdir(parents=True, exist_ok=True)
        (self.test_path / "wiki" / "Resources").mkdir(parents=True, exist_ok=True)
        (self.test_path / "wiki" / "Archives").mkdir(parents=True, exist_ok=True)

        storage.save_index(
            {
                "raw_processed": {},
                "embeddings_version": "all-MiniLM-L6-v2",
                "last_graph_build": None,
            },
            root=self.test_path,
        )
        emb.reset_model()

    def tearDown(self) -> None:
        """Clean up temporary directory and reset state."""
        emb.reset_model()
        storage.set_project_root(None)
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_raw_capture(
        self,
        cap_id: str,
        content: str,
        cap_type: str = "note",
        filename: str | None = None,
    ) -> CaptureMeta:
        """Helper to create a raw capture directory with meta.json and content file."""
        meta = CaptureMeta(
            id=cap_id,
            timestamp="2026-11-08T12:00:00Z",
            type=cap_type,
            source="cli",
            original_filename=filename,
            content_hash=storage.content_hash(content),
        )
        storage.write_raw_capture(meta, content, root=self.test_path)
        return meta

    # -------------------------------------------------------------------------
    # 1. Pipeline Execution with Multiple Pending Captures
    # -------------------------------------------------------------------------

    def test_01_pipeline_multiple_pending_captures_successful(self) -> None:
        """Pipeline process command successfully classifies and links multiple captures."""
        # Setup 2 captures that share high semantic similarity
        self._create_raw_capture("2026-11-08_cap001", "Plan for building SecondSelf knowledge graph.")
        self._create_raw_capture("2026-11-08_cap002", "Study guide on vector embeddings and knowledge brains.")

        # Unit vectors with cos = 0.85
        theta = float(np.arccos(0.85))
        v1 = _make_unit_vector(0.0)
        v2 = _make_unit_vector(theta)

        mock_llm_responses = {
            "SecondSelf": {"para": "Projects", "tags": ["ai", "graph"], "summary": "SecondSelf build plan"},
            "embeddings": {"para": "Resources", "tags": ["ai", "embeddings"], "summary": "Vector embedding study guide"},
        }

        def _mock_classify(text, model=None):
            for k, val in mock_llm_responses.items():
                if k.lower() in text.lower():
                    return val
            return {"para": "Resources", "tags": [], "summary": "Generic summary"}

        mock_encoder = _create_mock_encoder({
            "SecondSelf": v1,
            "embeddings": v2,
        })

        with patch("classify.classify_content", side_effect=_mock_classify):
            report = pipeline.process(
                root=self.test_path,
                model=mock_encoder,
                threshold=0.65,
            )

        self.assertTrue(report.success)
        self.assertEqual(report.captures_discovered, 2)
        self.assertEqual(report.captures_classified, 2)
        self.assertEqual(report.captures_failed, 0)
        self.assertEqual(report.notes_examined, 2)
        self.assertEqual(report.links_created, 2)

        # Check wiki notes created
        notes = {n.id: n for n in storage.read_wiki_notes(root=self.test_path)}
        self.assertIn("cap001", notes)
        self.assertIn("cap002", notes)
        self.assertEqual(notes["cap001"].para, "Projects")
        self.assertEqual(notes["cap002"].para, "Resources")

        # Check links created
        self.assertIn("cap002", notes["cap001"].links)
        self.assertIn("[[cap002]]", notes["cap001"].body)
        self.assertIn("cap001", notes["cap002"].links)
        self.assertIn("[[cap001]]", notes["cap002"].body)

        # Check index.json updated
        index = storage.load_index(root=self.test_path)
        self.assertIn("2026-11-08_cap001", index["raw_processed"])
        self.assertIn("2026-11-08_cap002", index["raw_processed"])

    # -------------------------------------------------------------------------
    # 2. Empty Input Directory
    # -------------------------------------------------------------------------

    def test_02_empty_input_directory(self) -> None:
        """Pipeline handles empty raw/ directory gracefully without errors."""
        mock_encoder = _create_mock_encoder()

        report = pipeline.process(root=self.test_path, model=mock_encoder)

        self.assertTrue(report.success)
        self.assertEqual(report.captures_discovered, 0)
        self.assertEqual(report.captures_classified, 0)
        self.assertEqual(report.notes_examined, 0)
        self.assertEqual(report.links_created, 0)
        self.assertEqual(len(report.failures), 0)

    # -------------------------------------------------------------------------
    # 3. Already-Processed Captures Skipped
    # -------------------------------------------------------------------------

    def test_03_already_processed_captures_skipped(self) -> None:
        """Captures present in raw_processed are skipped and not re-classified."""
        self._create_raw_capture("2026-11-08_done001", "Already processed content")

        # Mark processed in index.json
        index = storage.load_index(root=self.test_path)
        index["raw_processed"]["2026-11-08_done001"] = {
            "wiki_id": "done001",
            "para": "Projects",
            "classified_at": "2026-11-08T12:00:00Z",
            "content_hash": "dummy_hash",
        }
        storage.save_index(index, root=self.test_path)

        with patch("classify.classify_content") as mock_classify:
            report = pipeline.classify_only(root=self.test_path)

        self.assertEqual(report.captures_discovered, 0)
        self.assertEqual(report.captures_classified, 0)
        mock_classify.assert_not_called()

    # -------------------------------------------------------------------------
    # 4. Successful Processing Through Subcommands (classify vs link)
    # -------------------------------------------------------------------------

    def test_04_classify_only_and_link_only_subcommands(self) -> None:
        """classify_only and link_only execute only their respective stages."""
        self._create_raw_capture("2026-11-08_sub001", "Study transformer architecture.")

        mock_classify_res = {"para": "Resources", "tags": ["ml"], "summary": "Transformers guide"}
        mock_encoder = _create_mock_encoder()

        # Step 1: classify_only
        with patch("classify.classify_content", return_value=mock_classify_res):
            c_report = pipeline.classify_only(root=self.test_path)

        self.assertEqual(c_report.command, "classify")
        self.assertEqual(c_report.captures_classified, 1)
        self.assertEqual(c_report.notes_examined, 0)  # Linking did not run

        # Step 2: link_only
        l_report = pipeline.link_only(root=self.test_path, model=mock_encoder)

        self.assertEqual(l_report.command, "link")
        self.assertEqual(l_report.captures_discovered, 0)  # Classify did not run
        self.assertEqual(l_report.notes_examined, 1)

    # -------------------------------------------------------------------------
    # 5. Extraction Failure
    # -------------------------------------------------------------------------

    def test_05_extraction_failure_handled_safely(self) -> None:
        """Capture extraction failure is reported in failures and does not update index."""
        cap_id = "2026-11-08_fail_ext"
        self._create_raw_capture(cap_id, "Some text")

        with patch("classify.extract_text", side_effect=ValueError("Corrupted document")):
            report = pipeline.process(root=self.test_path)

        self.assertFalse(report.success)
        self.assertEqual(report.captures_failed, 1)
        self.assertEqual(len(report.failures), 1)

        failure = report.failures[0]
        self.assertEqual(failure.item_id, cap_id)
        self.assertEqual(failure.stage, "classify")
        self.assertEqual(failure.error_category, "ExtractionError")
        self.assertFalse(failure.retryable)

        # Verify not marked in index.json
        index = storage.load_index(root=self.test_path)
        self.assertNotIn(cap_id, index["raw_processed"])

    # -------------------------------------------------------------------------
    # 6. Classification Failure (LLM Failure)
    # -------------------------------------------------------------------------

    def test_06_classification_failure_handled_safely(self) -> None:
        """LLM error during classification is captured with retryable=True and index kept clean."""
        cap_id = "2026-11-08_fail_llm"
        self._create_raw_capture(cap_id, "Plan content")

        with patch("classify.classify_content", side_effect=RuntimeError("Groq API rate limit 429")):
            report = pipeline.process(root=self.test_path)

        self.assertFalse(report.success)
        self.assertEqual(report.captures_failed, 1)

        failure = report.failures[0]
        self.assertEqual(failure.item_id, cap_id)
        self.assertEqual(failure.stage, "classify")
        self.assertEqual(failure.error_category, "LLMNetworkError")
        self.assertTrue(failure.retryable)

        index = storage.load_index(root=self.test_path)
        self.assertNotIn(cap_id, index["raw_processed"])

    # -------------------------------------------------------------------------
    # 7. Embedding Failure
    # -------------------------------------------------------------------------

    def test_07_embedding_failure_handled_safely(self) -> None:
        """Embedding generation failure in link stage is reported without crashing."""
        note = WikiNote(
            id="bad_emb",
            raw_id="2026-11-08_bad_emb",
            para="Projects",
            tags=[],
            summary="Bad note",
            created="2026-11-08T12:00:00Z",
            links=[],
            body="Text for note",
        )
        storage.write_wiki_note(note, root=self.test_path)

        with patch("lib.embeddings.embed_text", side_effect=ValueError("Non-finite values")):
            report = pipeline.link_only(root=self.test_path)

        self.assertFalse(report.success)
        self.assertGreaterEqual(len(report.failures), 1)
        failure = report.failures[0]
        self.assertEqual(failure.stage, "link")
        self.assertEqual(failure.error_category, "EmbeddingValidationError")

    # -------------------------------------------------------------------------
    # 8. Linking Failure (Fatal Error in link stage)
    # -------------------------------------------------------------------------

    def test_08_linking_stage_fatal_failure_handled(self) -> None:
        """Fatal error during link stage returns failure record and non-zero exit."""
        with patch("link.link_notes", side_effect=RuntimeError("Unexpected storage failure")):
            report = pipeline.link_only(root=self.test_path)

        self.assertFalse(report.success)
        self.assertEqual(len(report.failures), 1)
        self.assertEqual(report.failures[0].stage, "link")
        self.assertIn("Unexpected storage failure", report.failures[0].error_message)

    # -------------------------------------------------------------------------
    # 9. Partial Batch Failure
    # -------------------------------------------------------------------------

    def test_09_partial_batch_failures_isolated(self) -> None:
        """In a mixed batch, valid items succeed and link, while broken items fail cleanly."""
        self._create_raw_capture("2026-11-08_good001", "Good capture 1 about data analysis")
        self._create_raw_capture("2026-11-08_broken", "Broken capture")
        self._create_raw_capture("2026-11-08_good002", "Good capture 2 about data analysis")

        def _mock_classify(text, model=None):
            if "broken" in text.lower():
                raise ValueError("Invalid schema returned by LLM")
            return {"para": "Projects", "tags": ["data"], "summary": "Data analysis note"}

        mock_encoder = _create_mock_encoder()

        with patch("classify.classify_content", side_effect=_mock_classify):
            report = pipeline.process(root=self.test_path, model=mock_encoder)

        self.assertFalse(report.success)
        self.assertEqual(report.captures_discovered, 3)
        self.assertEqual(report.captures_classified, 2)
        self.assertEqual(report.captures_failed, 1)

        # Successful notes were written to wiki
        notes = {n.id: n for n in storage.read_wiki_notes(root=self.test_path)}
        self.assertIn("good001", notes)
        self.assertIn("good002", notes)
        self.assertNotIn("broken", notes)

        # Index updated only for good ones
        index = storage.load_index(root=self.test_path)
        self.assertIn("2026-11-08_good001", index["raw_processed"])
        self.assertIn("2026-11-08_good002", index["raw_processed"])
        self.assertNotIn("2026-11-08_broken", index["raw_processed"])

    # -------------------------------------------------------------------------
    # 10. Repeated Pipeline Execution (Idempotency)
    # -------------------------------------------------------------------------

    def test_10_repeated_pipeline_execution_idempotency(self) -> None:
        """Re-running the pipeline on already-processed data performs 0 redundant work."""
        self._create_raw_capture("2026-11-08_rep001", "Study machine learning")
        self._create_raw_capture("2026-11-08_rep002", "Machine learning study guide")

        theta = float(np.arccos(0.85))
        v1 = _make_unit_vector(0.0)
        v2 = _make_unit_vector(theta)

        mock_encoder = _create_mock_encoder({
            "Study machine learning": v1,
            "Machine learning study": v2,
        })
        mock_classify_res = {"para": "Projects", "tags": ["ml"], "summary": "ML study"}

        # Run 1
        with patch("classify.classify_content", return_value=mock_classify_res):
            rep1 = pipeline.process(root=self.test_path, model=mock_encoder, threshold=0.65)

        self.assertEqual(rep1.captures_classified, 2)
        self.assertEqual(rep1.links_created, 2)

        # Snapshot files
        f1 = self.test_path / "wiki" / "Projects" / "rep001.md"
        f2 = self.test_path / "wiki" / "Projects" / "rep002.md"
        content1_run1 = f1.read_text(encoding="utf-8")
        content2_run1 = f2.read_text(encoding="utf-8")

        # Run 2: Re-run process
        with patch("classify.classify_content", return_value=mock_classify_res) as mock_cls:
            rep2 = pipeline.process(root=self.test_path, model=mock_encoder, threshold=0.65)

        self.assertEqual(rep2.captures_discovered, 0)
        self.assertEqual(rep2.captures_classified, 0)
        self.assertEqual(rep2.links_created, 0)
        self.assertEqual(rep2.links_preserved, 2)
        mock_cls.assert_not_called()

        # Files remain identical
        self.assertEqual(content1_run1, f1.read_text(encoding="utf-8"))
        self.assertEqual(content2_run1, f2.read_text(encoding="utf-8"))

    # -------------------------------------------------------------------------
    # 11. Preservation of Original Raw Files
    # -------------------------------------------------------------------------

    def test_11_preservation_of_original_raw_files(self) -> None:
        """Original raw/ files remain completely unmodified throughout pipeline execution."""
        cap_id = "2026-11-08_raw_pres"
        content = "Exact original raw note content."
        self._create_raw_capture(cap_id, content)

        raw_meta_file = self.test_path / "raw" / cap_id / "meta.json"
        raw_content_file = self.test_path / "raw" / cap_id / "content.md"

        meta_before = raw_meta_file.read_bytes()
        content_before = raw_content_file.read_bytes()

        mock_encoder = _create_mock_encoder()
        mock_classify_res = {"para": "Projects", "tags": [], "summary": "Raw pres test"}

        with patch("classify.classify_content", return_value=mock_classify_res):
            pipeline.process(root=self.test_path, model=mock_encoder)

        # Exact byte comparison
        self.assertEqual(meta_before, raw_meta_file.read_bytes())
        self.assertEqual(content_before, raw_content_file.read_bytes())

    # -------------------------------------------------------------------------
    # 12. Preservation of Existing Notes, Embeddings, and Unrelated Links
    # -------------------------------------------------------------------------

    def test_12_preservation_of_existing_notes_and_links(self) -> None:
        """Existing wiki notes and unrelated manual links are retained when new notes link."""
        # Pre-existing note with manual link to 'external_manual_link'
        existing_note = WikiNote(
            id="exist01",
            raw_id="2026-11-08_exist01",
            para="Areas",
            tags=["finance"],
            summary="Pre-existing financial note",
            created="2026-11-08T10:00:00Z",
            links=["external_manual_link"],
            body="Existing financial content. See [[external_manual_link]].",
        )
        storage.write_wiki_note(existing_note, root=self.test_path)

        # Mark as processed in index
        index = storage.load_index(root=self.test_path)
        index["raw_processed"]["2026-11-08_exist01"] = {
            "wiki_id": "exist01",
            "para": "Areas",
            "classified_at": "2026-11-08T10:00:00Z",
            "content_hash": "hash_exist01",
        }
        storage.save_index(index, root=self.test_path)

        # Capture a new note similar to exist01
        self._create_raw_capture("2026-11-08_new001", "Financial tax investment note")

        theta = float(np.arccos(0.90))
        v1 = _make_unit_vector(0.0)
        v2 = _make_unit_vector(theta)

        mock_encoder = _create_mock_encoder({
            "financial": v1,
            "Financial": v2,
        })
        mock_classify_res = {"para": "Areas", "tags": ["finance"], "summary": "New financial note"}

        with patch("classify.classify_content", return_value=mock_classify_res):
            pipeline.process(root=self.test_path, model=mock_encoder, threshold=0.65)

        notes = {n.id: n for n in storage.read_wiki_notes(root=self.test_path)}
        n_exist = notes["exist01"]

        # Manual link preserved + new link appended
        self.assertIn("external_manual_link", n_exist.links)
        self.assertIn("new001", n_exist.links)
        self.assertIn("[[external_manual_link]]", n_exist.body)
        self.assertIn("[[new001]]", n_exist.body)

    # -------------------------------------------------------------------------
    # 13. CLI Interface & Exit Codes
    # -------------------------------------------------------------------------

    def test_13_cli_help_flag(self) -> None:
        """pipeline.py --help prints usage and subcommands cleanly."""
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = pipeline.main(["--help"])
        self.assertEqual(code, 0)
        out = buf.getvalue()
        self.assertIn("pipeline.py", out)
        self.assertIn("process", out)
        self.assertIn("classify", out)
        self.assertIn("link", out)

    def test_14_cli_dry_run_flag(self) -> None:
        """pipeline.py --dry-run previews stages without modifying filesystem or index."""
        self._create_raw_capture("2026-11-08_dry001", "Dry run capture")

        mock_encoder = _create_mock_encoder()
        mock_classify_res = {"para": "Projects", "tags": [], "summary": "Dry summary"}

        stdout_buf = io.StringIO()
        with patch("classify.classify_content", return_value=mock_classify_res), \
             patch("lib.embeddings.load_model", return_value=mock_encoder), \
             redirect_stdout(stdout_buf):
            code = pipeline.main([
                "process",
                "--root", str(self.test_path),
                "--dry-run",
            ])

        self.assertEqual(code, 0)
        out = stdout_buf.getvalue()
        self.assertIn("[dry-run]", out)
        self.assertIn("Would write", out)

        # Confirm no wiki notes written
        self.assertEqual(len(storage.read_wiki_notes(root=self.test_path)), 0)

        # Confirm index.json not modified
        index = storage.load_index(root=self.test_path)
        self.assertNotIn("2026-11-08_dry001", index["raw_processed"])

    def test_15_cli_failure_exit_code(self) -> None:
        """pipeline CLI returns exit code 1 when stage failures occur."""
        self._create_raw_capture("2026-11-08_err001", "Failing capture")

        stderr_buf = io.StringIO()
        with patch("classify.classify_content", side_effect=RuntimeError("API error")), \
             redirect_stderr(stderr_buf):
            code = pipeline.main([
                "classify",
                "--root", str(self.test_path),
            ])

        self.assertEqual(code, 1)
        err = stderr_buf.getvalue()
        self.assertIn("Failures Encountered", err)
        self.assertIn("2026-11-08_err001", err)

    # -------------------------------------------------------------------------
    # Task 3.2.5: Graph Integration Tests
    # -------------------------------------------------------------------------

    def test_16_process_invokes_graph_builder_and_produces_graph_json(self) -> None:
        """pipeline process rebuilds data/graph.json upon successful linking."""
        self._create_raw_capture("2026-11-08_cap101", "Building SecondSelf graph orchestrator.")
        self._create_raw_capture("2026-11-08_cap102", "Knowledge brain graph visualization with vis-network.")

        theta = float(np.arccos(0.85))
        v1 = _make_unit_vector(0.0)
        v2 = _make_unit_vector(theta)

        mock_classify_responses = {
            "orchestrator": {"para": "Projects", "tags": ["graph"], "summary": "Pipeline graph builder"},
            "vis-network": {"para": "Resources", "tags": ["ui"], "summary": "Vis-network interactive UI"},
        }

        def _mock_classify(text, model=None):
            for k, val in mock_classify_responses.items():
                if k.lower() in text.lower():
                    return val
            return {"para": "Resources", "tags": [], "summary": "Generic"}

        mock_encoder = _create_mock_encoder({
            "orchestrator": v1,
            "vis-network": v2,
        })

        with patch("classify.classify_content", side_effect=_mock_classify):
            report = pipeline.process(
                root=self.test_path,
                model=mock_encoder,
                threshold=0.65,
            )

        self.assertTrue(report.success)
        self.assertTrue(report.graph_rebuilt)
        self.assertEqual(report.graph_node_count, 2)
        self.assertEqual(report.graph_edge_count, 1)

        # Confirm data/graph.json exists and is valid
        graph_path = self.test_path / "data" / "graph.json"
        self.assertTrue(graph_path.exists())
        with open(graph_path, "r", encoding="utf-8") as f:
            graph_data = json.load(f)

        self.assertEqual(len(graph_data["nodes"]), 2)
        self.assertEqual(len(graph_data["edges"]), 1)
        self.assertEqual(graph_data["metadata"]["node_count"], 2)
        self.assertEqual(graph_data["metadata"]["edge_count"], 1)

        # Confirm Task 3.1.2 schema validation passes
        val_report = validate_graph.validate_graph_file(graph_path)
        self.assertTrue(val_report.valid)
        self.assertEqual(val_report.error_count, 0)

        # Confirm index.json last_graph_build matches
        index = storage.load_index(root=self.test_path)
        self.assertEqual(index["last_graph_build"], graph_data["metadata"]["generated_at"])

    def test_17_execution_order_strict_pipeline_sequence(self) -> None:
        """Pipeline enforces strict execution sequence: classify -> link -> graph builder."""
        self._create_raw_capture("2026-11-08_order01", "Sequence verification capture.")

        call_sequence = []

        def _mock_classify_all(*args, **kwargs):
            call_sequence.append("classify")
            note = WikiNote(
                id="order01",
                raw_id="2026-11-08_order01",
                para="Projects",
                tags=["test"],
                summary="Order note",
                created="2026-11-08T12:00:00Z",
                links=[],
                body="Body",
            )
            return [note], []

        def _mock_link_notes(*args, **kwargs):
            call_sequence.append("link")
            rep = pipeline.LinkReport(notes_examined=1, links_created=0)
            return rep

        def _mock_build_graph(*args, **kwargs):
            call_sequence.append("graph")
            return {"metadata": {"node_count": 1, "edge_count": 0}}, []

        with patch("classify.classify_all", side_effect=_mock_classify_all), \
             patch("link.link_notes", side_effect=_mock_link_notes), \
             patch("build_graph.build_and_save_graph", side_effect=_mock_build_graph):
            report = pipeline.process(root=self.test_path)

        self.assertTrue(report.success)
        self.assertEqual(call_sequence, ["classify", "link", "graph"])

    def test_18_linking_failure_prevents_graph_building(self) -> None:
        """If linking stage fails, graph rebuilding is skipped and pipeline reports failure."""
        self._create_raw_capture("2026-11-08_linkfail01", "Capture before link failure.")

        mock_classify_res = {"para": "Projects", "tags": [], "summary": "Note before fail"}

        with patch("classify.classify_content", return_value=mock_classify_res), \
             patch("link.link_notes", side_effect=RuntimeError("Embedding model crashed")), \
             patch("build_graph.build_and_save_graph") as mock_builder:
            report = pipeline.process(root=self.test_path)

        self.assertFalse(report.success)
        self.assertFalse(report.graph_rebuilt)
        # Graph builder should NOT have been invoked
        mock_builder.assert_not_called()

        # Failure recorded under link stage
        link_failures = [f for f in report.failures if f.stage == "link"]
        self.assertGreater(len(link_failures), 0)
        self.assertIn("Embedding model crashed", link_failures[0].error_message)

    def test_19_graph_building_failure_reported_and_wiki_preserved(self) -> None:
        """If graph building fails, failure is reported without rolling back wiki notes."""
        self._create_raw_capture("2026-11-08_graphfail01", "Valid capture before graph fail.")

        mock_classify_res = {"para": "Projects", "tags": ["graph"], "summary": "Valid wiki summary"}
        mock_encoder = _create_mock_encoder()

        with patch("classify.classify_content", return_value=mock_classify_res), \
             patch("build_graph.save_graph", side_effect=OSError("Disk full")):
            report = pipeline.process(
                root=self.test_path,
                model=mock_encoder,
            )

        self.assertFalse(report.success)
        self.assertFalse(report.graph_rebuilt)

        # Graph failure recorded
        graph_failures = [f for f in report.failures if f.stage == "graph"]
        self.assertEqual(len(graph_failures), 1)
        self.assertIn("Disk full", graph_failures[0].error_message)

        # Wiki notes created by Stage 1 must NOT be deleted or rolled back
        notes = storage.read_wiki_notes(root=self.test_path)
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0].id, "graphfail01")

    def test_20_repeated_pipeline_idempotency_no_duplicate_nodes_or_edges(self) -> None:
        """Repeated pipeline runs maintain clean graph without duplicate nodes or edges."""
        self._create_raw_capture("2026-11-08_idem001", "First note for idempotency test.")
        self._create_raw_capture("2026-11-08_idem002", "Second note for idempotency test.")

        theta = float(np.arccos(0.80))
        mock_encoder = _create_mock_encoder({
            "First": _make_unit_vector(0.0),
            "Second": _make_unit_vector(theta),
        })

        mock_classify_res = {"para": "Areas", "tags": ["work"], "summary": "Idempotent note"}

        with patch("classify.classify_content", return_value=mock_classify_res):
            # First execution
            rep1 = pipeline.process(root=self.test_path, model=mock_encoder, threshold=0.65)
            self.assertTrue(rep1.success)
            self.assertEqual(rep1.graph_node_count, 2)
            self.assertEqual(rep1.graph_edge_count, 1)

            # Second execution immediately after
            rep2 = pipeline.process(root=self.test_path, model=mock_encoder, threshold=0.65)
            self.assertTrue(rep2.success)
            self.assertEqual(rep2.captures_classified, 0)  # 0 new captures
            self.assertEqual(rep2.graph_node_count, 2)     # still 2 nodes, no duplicates
            self.assertEqual(rep2.graph_edge_count, 1)     # still 1 edge, no duplicates

        # Validate graph schema on disk
        graph_file = self.test_path / "data" / "graph.json"
        val = validate_graph.validate_graph_file(graph_file)
        self.assertTrue(val.valid)
        self.assertEqual(val.node_count, 2)
        self.assertEqual(val.edge_count, 1)

    def test_21_cli_process_reports_graph_rebuilt(self) -> None:
        """pipeline CLI 'process' reports Graph Stage in console output."""
        self._create_raw_capture("2026-11-08_cliproc01", "CLI process capture")

        mock_encoder = _create_mock_encoder()
        mock_classify_res = {"para": "Resources", "tags": [], "summary": "CLI summary"}

        stdout_buf = io.StringIO()
        with patch("classify.classify_content", return_value=mock_classify_res), \
             patch("lib.embeddings.load_model", return_value=mock_encoder), \
             redirect_stdout(stdout_buf):
            code = pipeline.main([
                "process",
                "--root", str(self.test_path),
            ])

        self.assertEqual(code, 0)
        out = stdout_buf.getvalue()
        self.assertIn("Graph Stage:", out)
        self.assertIn("Built knowledge graph: 1 nodes", out)
        self.assertIn("Graph Rebuilt:         Yes", out)

    def test_22_classify_only_and_link_only_do_not_rebuild_graph(self) -> None:
        """'classify' and 'link' isolated commands do not trigger graph rebuilding."""
        self._create_raw_capture("2026-11-08_subcmd01", "Subcommand capture")
        mock_classify_res = {"para": "Projects", "tags": [], "summary": "Subcmd summary"}
        mock_encoder = _create_mock_encoder()

        with patch("classify.classify_content", return_value=mock_classify_res):
            rep_class = pipeline.classify_only(root=self.test_path)
        self.assertTrue(rep_class.success)
        self.assertFalse(rep_class.graph_rebuilt)

        rep_link = pipeline.link_only(root=self.test_path, model=mock_encoder)
        self.assertTrue(rep_link.success)
        self.assertFalse(rep_link.graph_rebuilt)


if __name__ == "__main__":
    unittest.main()

