"""Unit and integration tests for SecondSelf Streamlit web application (Tasks 4.2.1 and 4.2.2).

Validates:
- Layout and component presence matching Task 4.2.1/4.2.2 specifications:
  Header with Refresh Graph, Ask section, Interactive Graph, and Sidebar (Capture | Process | Stats).
- Helper functions: error message sanitization, graph data loader, graph HTML inlining, and brain stats.
- Component Wiring (Task 4.2.2):
  1. Ask bar -> ask.ask(question)
  2. Graph -> components.html() embedding vis-network
  3. Capture form -> capture.capture_note(text)
  4. Process button -> pipeline.process() -> rebuild graph
  5. Stats sidebar -> count nodes/edges from graph.json
  6. Refresh button -> re-run build_graph.build_and_save_graph(), reload component
- Application States:
  - Initial state
  - Empty question submission (rejected with warning)
  - Successful answer with multiple sources
  - Answer with no sources
  - Insufficient knowledge / no relevant results
  - Missing or invalid API configuration (handled safely without leaking credentials)
  - RAG errors (handled gracefully without crashing application)
  - Rerun persistence (no duplicate API calls)
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from streamlit.testing.v1 import AppTest

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import app
import ask
from lib.models import AskResult, CaptureResult, WikiNote
import lib.storage as storage
from pipeline import PipelineReport


class TestAppHelpers(unittest.TestCase):
    """Test suite for app.py helper functions."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

        # Initialize directories
        self.data_dir = self.root / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.wiki_dir = self.root / "wiki"
        self.wiki_dir.mkdir(parents=True, exist_ok=True)
        for cat in ("Projects", "Areas", "Resources", "Archives"):
            (self.wiki_dir / cat).mkdir(exist_ok=True)
        self.raw_dir = self.root / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.static_dir = self.root / "static"
        self.static_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_sanitize_error_message_redacts_api_key(self) -> None:
        raw_error = "Groq request failed with key gsk_abcdef1234567890abcdef1234567890"
        sanitized = app.sanitize_error_message(raw_error)
        self.assertNotIn("gsk_abcdef1234567890abcdef1234567890", sanitized)
        self.assertIn("authentication", sanitized.lower())

    def test_sanitize_error_message_redacts_paths(self) -> None:
        raw_error = "Failed to load D:\\New folder (2)\\secondself\\secret.txt"
        sanitized = app.sanitize_error_message(raw_error)
        self.assertNotIn("D:\\New folder (2)", sanitized)
        self.assertIn("[local path]", sanitized)

    def test_sanitize_error_message_rate_limit(self) -> None:
        raw_error = "Rate limit reached: error 429 too many requests"
        sanitized = app.sanitize_error_message(raw_error)
        self.assertIn("rate limit", sanitized.lower())

    def test_sanitize_error_message_network_connection(self) -> None:
        raw_error = "Connection timeout while connecting to groq host"
        sanitized = app.sanitize_error_message(raw_error)
        self.assertIn("connection", sanitized.lower())

    def test_sanitize_error_message_missing_key(self) -> None:
        raw_error = "ValueError: GROQ_API_KEY not found or empty. Please set GROQ_API_KEY in your .env file"
        sanitized = app.sanitize_error_message(raw_error)
        self.assertIn("authentication", sanitized.lower())

    def test_load_graph_data_missing_file_returns_default(self) -> None:
        graph = app.load_graph_data(root=str(self.root))
        self.assertEqual(graph["nodes"], [])
        self.assertEqual(graph["edges"], [])
        self.assertEqual(graph["metadata"]["node_count"], 0)

    def test_load_graph_data_existing_file(self) -> None:
        graph_file = self.data_dir / "graph.json"
        test_data = {
            "nodes": [{"id": "n1", "label": "Node 1"}],
            "edges": [],
            "metadata": {"node_count": 1, "edge_count": 0},
        }
        graph_file.write_text(json.dumps(test_data), encoding="utf-8")

        # Clear cache to ensure fresh read
        app.load_graph_data.clear()
        graph = app.load_graph_data(root=str(self.root))
        self.assertEqual(len(graph["nodes"]), 1)
        self.assertEqual(graph["nodes"][0]["id"], "n1")

    def test_get_graph_html_inlines_data(self) -> None:
        html_file = self.static_dir / "graph.html"
        html_file.write_text(
            "<html><head><title>Test</title></head><body>"
            "const response = await fetch(\"../data/graph.json\", { cache: \"no-store\" });\n"
            "          if (!response.ok) {\n"
            "            throw new Error(`HTTP ${response.status} (${response.statusText})`);\n"
            "          }\n\n"
            "          const graph = await response.json();"
            "</body></html>",
            encoding="utf-8",
        )

        test_data = {"nodes": [{"id": "n1"}], "edges": []}
        result_html = app.get_graph_html(test_data, root=self.root)
        self.assertIn("window.__INLINED_GRAPH_DATA__", result_html)
        self.assertIn('"id": "n1"', result_html)

    def test_get_graph_html_missing_template(self) -> None:
        missing_root = self.root / "nonexistent"
        result_html = app.get_graph_html({}, root=missing_root)
        self.assertIn("Knowledge graph template not found", result_html)

    def test_get_brain_stats(self) -> None:
        # Create sample wiki note
        note = WikiNote(
            id="test01",
            raw_id="raw01",
            para="Projects",
            tags=["ai"],
            summary="Project note",
            created="2026-07-06T12:00:00Z",
            links=[],
            body="Content of note",
        )
        storage.write_wiki_note(note, root=self.root)

        # Write graph.json
        graph_file = self.data_dir / "graph.json"
        graph_file.write_text(
            json.dumps({"nodes": [{"id": "test01"}], "edges": [], "metadata": {"node_count": 1, "edge_count": 0}}),
            encoding="utf-8",
        )

        app.load_graph_data.clear()
        stats = app.get_brain_stats(root=self.root)
        self.assertEqual(stats["total_notes"], 1)
        self.assertEqual(stats["para_breakdown"]["Projects"], 1)
        self.assertEqual(stats["graph_nodes"], 1)
        self.assertEqual(stats["graph_edges"], 0)

    def test_capture_uploaded_file_saves_to_raw(self) -> None:
        """Verify capture_uploaded_file correctly extracts content and writes metadata."""
        mock_file = MagicMock()
        mock_file.name = "sample_document.pdf"
        mock_file.getvalue.return_value = b"%PDF-1.4 sample content"

        cap_res = app.capture_uploaded_file(mock_file, root=self.root)
        self.assertIsInstance(cap_res, CaptureResult)
        self.assertEqual(cap_res.type, "file")

        # Verify created folder and files
        cap_dir = self.root / "raw" / cap_res.id
        self.assertTrue(cap_dir.exists())
        self.assertTrue((cap_dir / "meta.json").exists())
        self.assertTrue((cap_dir / "content.pdf").exists())

        with open(cap_dir / "meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.assertEqual(meta["original_filename"], "sample_document.pdf")
        self.assertEqual(meta["type"], "file")


class TestAppLayoutAndInteractions(unittest.TestCase):
    """Integration test suite using streamlit.testing.v1.AppTest."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app_path = str((_REPO_ROOT / "app.py").resolve())

    def setUp(self) -> None:
        # Clear Streamlit caches before each integration test
        app.load_graph.clear()
        app.load_embeddings.clear()
        self.mock_model = MagicMock()
        self.mock_model_patcher = patch("lib.embeddings.load_model", return_value=self.mock_model)
        self.mock_model_patcher.start()

    def tearDown(self) -> None:
        self.mock_model_patcher.stop()

    def test_initial_page_layout_elements(self) -> None:
        """Verify all planned layout elements are present on initial render."""
        at = AppTest.from_file(self.app_path).run()

        # 1. Header with application title and Refresh Graph button
        title_values = [t.value for t in at.title]
        self.assertTrue(any("🧠 SecondSelf" in t for t in title_values))

        button_labels = [b.label for b in at.button]
        self.assertIn("🔄 Refresh Graph", button_labels)

        # 2. Ask section
        text_input_labels = [ti.label for ti in at.text_input]
        self.assertIn("Ask your brain:", text_input_labels)
        self.assertIn("Ask", button_labels)

        # 3. Interactive Knowledge Graph section
        subheader_values = [sh.value for sh in at.subheader]
        self.assertIn("Interactive Knowledge Graph", subheader_values)

        # 4. Sidebar: Capture, Process, Stats
        sidebar_subheaders = [sh.value for sh in at.sidebar.subheader]
        self.assertTrue(any("Capture" in sh for sh in sidebar_subheaders))
        self.assertTrue(any("Process" in sh for sh in sidebar_subheaders))
        self.assertTrue(any("Stats" in sh for sh in sidebar_subheaders))

        self.assertIn("Capture Note", [b.label for b in at.sidebar.button])
        self.assertIn("🚀 Process New Captures", [b.label for b in at.sidebar.button])

        # Verify no answers or errors displayed initially
        self.assertEqual(len(at.error), 0)

    def test_blank_question_does_not_call_backend(self) -> None:
        """Verify that submitting empty or whitespace question shows warning and does not call ask()."""
        at = AppTest.from_file(self.app_path).run()

        with patch("ask.ask") as mock_ask:
            # Set input to whitespace
            at.text_input[0].input("   ")
            # Click the Ask submit button
            at.button[1].click().run()

            mock_ask.assert_not_called()
            self.assertTrue(any("Please enter a question" in w.value for w in at.warning))

    def test_submitted_question_calls_backend_and_displays_results(self) -> None:
        """Verify question submission calls ask() once and renders answer and sources."""
        at = AppTest.from_file(self.app_path).run()

        mock_sources = [
            {
                "id": "note001",
                "para": "Areas",
                "summary": "Career milestones and goals",
                "relevance_score": 0.8842,
            },
            {
                "id": "note002",
                "para": "Resources",
                "summary": "ML architecture notes",
                "relevance_score": 0.7651,
            },
        ]
        mock_result = AskResult(
            answer="Your primary career milestone is transitioning to ML engineering.",
            sources=mock_sources,
        )

        with patch("ask.ask", return_value=mock_result) as mock_ask:
            at.text_input[0].input("What are my career goals?")
            at.button[1].click().run()

            self.assertEqual(mock_ask.call_count, 1)
            self.assertEqual(mock_ask.call_args[0][0], "What are my career goals?")

            # Verify Answer is displayed
            markdown_texts = [m.value for m in at.markdown]
            self.assertTrue(any("transitioning to ML engineering" in m for m in markdown_texts))

            # Verify Sources are displayed
            self.assertTrue(any("[note001]" in m for m in markdown_texts))
            self.assertTrue(any("Career milestones and goals" in m for m in markdown_texts))
            self.assertTrue(any("[note002]" in m for m in markdown_texts))
            self.assertTrue(any("0.8842" in m for m in markdown_texts))

    def test_answer_with_no_sources_display(self) -> None:
        """Verify that an answer returned with no sources displays the appropriate caption."""
        at = AppTest.from_file(self.app_path).run()

        mock_result = AskResult(
            answer="This is a direct answer with no specific source citations.",
            sources=[],
        )

        with patch("ask.ask", return_value=mock_result):
            at.text_input[0].input("Direct question")
            at.button[1].click().run()

            markdown_texts = [m.value for m in at.markdown]
            self.assertTrue(any("direct answer with no specific source citations" in m for m in markdown_texts))

            caption_texts = [c.value for c in at.caption]
            self.assertTrue(any("No sources were cited" in c for c in caption_texts))

    def test_rerun_preserves_answer_without_duplicate_call(self) -> None:
        """Verify that a subsequent rerun preserves displayed answer and does not call backend again."""
        at = AppTest.from_file(self.app_path).run()

        mock_result = AskResult(
            answer="Answer to be preserved across reruns.",
            sources=[{"id": "note001", "para": "Areas", "summary": "Sample summary", "relevance_score": 0.85}],
        )

        with patch("ask.ask", return_value=mock_result) as mock_ask:
            at.text_input[0].input("Preserve this answer")
            at.button[1].click().run()
            self.assertEqual(mock_ask.call_count, 1)

            # Ordinary rerun (e.g. without re-submitting ask form)
            at.run()
            # Backend should NOT be called again
            self.assertEqual(mock_ask.call_count, 1)

            # Content should still be visible
            markdown_texts = [m.value for m in at.markdown]
            self.assertTrue(any("Answer to be preserved across reruns." in m for m in markdown_texts))
            self.assertTrue(any("[note001]" in m for m in markdown_texts))

    def test_backend_error_handled_without_exposing_secrets(self) -> None:
        """Verify backend failures display sanitized error and hide API keys and stack traces."""
        at = AppTest.from_file(self.app_path).run()

        sensitive_error = Exception(
            "Groq API call failed: gsk_testsecretkey99999999999999999999 "
            "Traceback (most recent call last): File D:\\New folder (2)\\secondself\\ask.py"
        )

        with patch("ask.ask", side_effect=sensitive_error):
            at.text_input[0].input("Question causing error")
            at.button[1].click().run()

            error_values = [e.value for e in at.error]
            self.assertGreater(len(error_values), 0)

            combined_error_text = " ".join(error_values)
            self.assertNotIn("gsk_testsecretkey", combined_error_text)
            self.assertNotIn("D:\\New folder (2)", combined_error_text)
            self.assertNotIn("Traceback", combined_error_text)

    def test_insufficient_information_response_handling(self) -> None:
        """Verify informative message when backend returns no-notes answer."""
        at = AppTest.from_file(self.app_path).run()

        mock_result = AskResult(answer=ask.NO_NOTES_ANSWER, sources=[])

        with patch("ask.ask", return_value=mock_result):
            at.text_input[0].input("What is my secret recipe?")
            at.button[1].click().run()

            info_values = [i.value for i in at.info]
            self.assertTrue(any("I don't have notes about that." in i for i in info_values))

    def test_sidebar_capture_wires_to_capture_note(self) -> None:
        """Verify sidebar capture form invokes capture.capture_note(text) and reports success."""
        at = AppTest.from_file(self.app_path).run()

        mock_capture_result = CaptureResult(
            id="2026-11-09_abcd1234",
            path="raw/2026-11-09_abcd1234",
            type="note",
        )

        with patch("capture.capture_note", return_value=mock_capture_result) as mock_cap:
            at.sidebar.text_area[0].input("Project idea: LLM agents for code review")
            at.sidebar.button[0].click().run()

            mock_cap.assert_called_once_with("Project idea: LLM agents for code review")
            sidebar_success = [s.value for s in at.sidebar.success]
            self.assertTrue(any("2026-11-09_abcd1234" in s for s in sidebar_success))

    def test_sidebar_capture_empty_shows_warning(self) -> None:
        """Verify sidebar capture with empty text does not call backend and shows warning."""
        at = AppTest.from_file(self.app_path).run()

        with patch("capture.capture_note") as mock_cap:
            at.sidebar.text_area[0].input("   ")
            at.sidebar.button[0].click().run()

            mock_cap.assert_not_called()
            sidebar_warnings = [w.value for w in at.sidebar.warning]
            self.assertTrue(any("Note text cannot be empty" in w for w in sidebar_warnings))

    def test_sidebar_process_wires_to_pipeline_process(self) -> None:
        """Verify sidebar Process button invokes pipeline.process() and reports metrics."""
        at = AppTest.from_file(self.app_path).run()

        mock_report = PipelineReport(
            command="process",
            captures_classified=3,
            links_created=5,
            graph_rebuilt=True,
            graph_node_count=10,
            graph_edge_count=8,
        )

        with patch("pipeline.process", return_value=mock_report) as mock_proc:
            proc_btn = next((b for b in at.sidebar.button if "Process" in b.label), at.sidebar.button[-1])
            proc_btn.click().run()

            mock_proc.assert_called_once()
            sidebar_success = [s.value for s in at.sidebar.success]
            self.assertTrue(any("3 classified" in s for s in sidebar_success))
            self.assertTrue(any("5 linked" in s for s in sidebar_success))
            self.assertTrue(any("10" in s for s in sidebar_success))

    def test_header_refresh_wires_to_build_graph(self) -> None:
        """Verify header Refresh Graph button invokes build_graph.build_and_save_graph()."""
        at = AppTest.from_file(self.app_path).run()

        mock_graph_data = {"nodes": [], "edges": [], "metadata": {"node_count": 0, "edge_count": 0}}

        with patch("build_graph.build_and_save_graph", return_value=(mock_graph_data, [])) as mock_build:
            at.button[0].click().run()

            mock_build.assert_called_once()
            success_messages = [s.value for s in at.success]
            self.assertTrue(any("Graph refreshed successfully" in s for s in success_messages))


if __name__ == "__main__":
    unittest.main()
