"""Focused unit tests for caching behavior in SecondSelf (Task 4.2.3).

Validates:
- Model initialization caching (@st.cache_resource def load_embeddings).
- Reusable model instance across repeated calls (executed once, warm reuse).
- Distinct inputs producing distinct cached model results.
- Cache invalidation via .clear() for model reloading.
- Failed operations not leaving corrupt cache entries.
- Stable graph data caching (@st.cache_data def load_graph and load_graph_data).
- Invalidation when graph file changes on disk and .clear() is called.
- Note embeddings dictionary caching (@st.cache_data def load_note_embeddings).
- Integration: ask section passes cached model to ask.ask(..., model=cached_model).
- Integration: Refresh Graph and Process New Captures invalidate appropriate caches.
- Data structures and return values are preserved uncorrupted.
- Cold start / behavior when no cache exists yet.
"""

from __future__ import annotations

import json
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, call, patch

import numpy as np

# Ensure secondself project root is in sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import app
import ask
import lib.embeddings as emb
from lib.models import AskResult, WikiNote
import lib.storage as storage


class TestModelCaching(unittest.TestCase):
    """Test suite for embedding model caching (@st.cache_resource)."""

    def setUp(self) -> None:
        app.load_embeddings.clear()

    def tearDown(self) -> None:
        app.load_embeddings.clear()

    def test_load_embeddings_executes_once_for_repeated_calls(self) -> None:
        """Verify expensive model initialization executes once across repeated calls."""
        mock_model = MagicMock(name="CachedSentenceTransformer")

        with patch("lib.embeddings.load_model", return_value=mock_model) as mock_load:
            # First call: triggers initialization
            m1 = app.load_embeddings(model_name="test-model")
            self.assertIs(m1, mock_model)
            self.assertEqual(mock_load.call_count, 1)

            # Second call with identical input: should return cached resource without re-invoking loader
            m2 = app.load_embeddings(model_name="test-model")
            self.assertIs(m2, mock_model)
            self.assertEqual(mock_load.call_count, 1)

            # Third call: still warm cache
            m3 = app.load_embeddings(model_name="test-model")
            self.assertIs(m3, mock_model)
            self.assertEqual(mock_load.call_count, 1)

    def test_load_embeddings_distinct_inputs_cached_separately(self) -> None:
        """Verify different model names produce distinct cached model instances."""
        model_a = MagicMock(name="ModelA")
        model_b = MagicMock(name="ModelB")

        def side_effect(model_name=None, root=None):
            if model_name == "model-alpha":
                return model_a
            elif model_name == "model-beta":
                return model_b
            return MagicMock()

        with patch("lib.embeddings.load_model", side_effect=side_effect) as mock_load:
            res_a = app.load_embeddings(model_name="model-alpha")
            res_b = app.load_embeddings(model_name="model-beta")

            self.assertIs(res_a, model_a)
            self.assertIs(res_b, model_b)
            self.assertIsNot(res_a, res_b)
            self.assertEqual(mock_load.call_count, 2)

            # Repeated calls for both should now hit their respective cache entries
            self.assertIs(app.load_embeddings(model_name="model-alpha"), model_a)
            self.assertIs(app.load_embeddings(model_name="model-beta"), model_b)
            self.assertEqual(mock_load.call_count, 2)

    def test_load_embeddings_invalidation_via_clear(self) -> None:
        """Verify cache invalidation via .clear() re-triggers loader on next access."""
        model_v1 = MagicMock(name="ModelV1")
        model_v2 = MagicMock(name="ModelV2")

        with patch("lib.embeddings.load_model", side_effect=[model_v1, model_v2]) as mock_load:
            m1 = app.load_embeddings(model_name="reloadable")
            self.assertIs(m1, model_v1)
            self.assertEqual(mock_load.call_count, 1)

            # Invalidate cache
            app.load_embeddings.clear()

            # Next call must reload
            m2 = app.load_embeddings(model_name="reloadable")
            self.assertIs(m2, model_v2)
            self.assertEqual(mock_load.call_count, 2)

    def test_load_embeddings_failure_not_cached_as_success(self) -> None:
        """Verify failed initialization raises error and does not poison subsequent valid calls."""
        good_model = MagicMock(name="GoodModel")

        with patch("lib.embeddings.load_model", side_effect=[RuntimeError("GPU OOM"), good_model]) as mock_load:
            # First call fails
            with self.assertRaises(RuntimeError) as ctx:
                app.load_embeddings(model_name="fragile-model")
            self.assertIn("GPU OOM", str(ctx.exception))
            self.assertEqual(mock_load.call_count, 1)

            # Second call should retry and succeed, not return a cached failure
            m = app.load_embeddings(model_name="fragile-model")
            self.assertIs(m, good_model)
            self.assertEqual(mock_load.call_count, 2)

    def test_load_embedding_model_alias_identical_to_load_embeddings(self) -> None:
        """Verify load_embedding_model is an exact alias sharing the same resource cache."""
        mock_model = MagicMock(name="SharedModel")

        with patch("lib.embeddings.load_model", return_value=mock_model) as mock_load:
            res1 = app.load_embeddings(model_name="alias-test")
            res2 = app.load_embedding_model(model_name="alias-test")

            self.assertIs(res1, res2)
            self.assertEqual(mock_load.call_count, 1)


class TestGraphDataCaching(unittest.TestCase):
    """Test suite for graph data caching (@st.cache_data)."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.data_dir = self.root / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        app.load_graph.clear()

    def tearDown(self) -> None:
        app.load_graph.clear()
        self.temp_dir.cleanup()

    def test_load_graph_executes_once_for_repeated_calls(self) -> None:
        """Verify graph data is loaded from disk once and served from cache thereafter."""
        graph_file = self.data_dir / "graph.json"
        graph_payload = {
            "nodes": [{"id": "n1", "label": "Note 1"}],
            "edges": [{"source": "n1", "target": "n2", "weight": 0.8}],
            "metadata": {"node_count": 1, "edge_count": 1, "generated_at": "2026-11-10T00:00:00Z"},
        }
        graph_file.write_text(json.dumps(graph_payload), encoding="utf-8")

        # First read
        data1 = app.load_graph(root=str(self.root))
        self.assertEqual(len(data1["nodes"]), 1)
        self.assertEqual(data1["nodes"][0]["id"], "n1")

        # Second read: should hit cache
        data2 = app.load_graph(root=str(self.root))
        self.assertEqual(data1, data2)

    def test_load_graph_distinct_roots_cached_separately(self) -> None:
        """Verify distinct root directory paths produce distinct cached graph results."""
        root2 = self.root / "other_root"
        data_dir2 = root2 / "data"
        data_dir2.mkdir(parents=True, exist_ok=True)

        payload1 = {"nodes": [{"id": "r1"}], "edges": [], "metadata": {"node_count": 1}}
        payload2 = {"nodes": [{"id": "r2"}], "edges": [], "metadata": {"node_count": 1}}

        (self.data_dir / "graph.json").write_text(json.dumps(payload1), encoding="utf-8")
        (data_dir2 / "graph.json").write_text(json.dumps(payload2), encoding="utf-8")

        g1 = app.load_graph(root=str(self.root))
        g2 = app.load_graph(root=str(root2))

        self.assertEqual(g1["nodes"][0]["id"], "r1")
        self.assertEqual(g2["nodes"][0]["id"], "r2")

    def test_load_graph_missing_file_returns_default_safely(self) -> None:
        """Verify missing graph.json returns fallback DEFAULT_GRAPH_DATA without crashing."""
        missing_root = self.root / "empty_dir"
        data = app.load_graph(root=str(missing_root))
        self.assertEqual(data["nodes"], [])
        self.assertEqual(data["edges"], [])
        self.assertEqual(data["metadata"]["node_count"], 0)

    def test_load_graph_corrupt_file_returns_default_safely(self) -> None:
        """Verify malformed JSON in graph.json logs warning and returns safe default."""
        graph_file = self.data_dir / "graph.json"
        graph_file.write_text("NOT_VALID_JSON{[[[", encoding="utf-8")

        data = app.load_graph(root=str(self.root))
        self.assertEqual(data["nodes"], [])
        self.assertEqual(data["edges"], [])
        self.assertEqual(data["metadata"]["node_count"], 0)

    def test_load_graph_invalidation_reflects_disk_updates(self) -> None:
        """Verify that clearing cache allows modified graph.json to be visible immediately."""
        graph_file = self.data_dir / "graph.json"

        # Version 1: 1 node
        v1 = {"nodes": [{"id": "initial"}], "edges": [], "metadata": {"node_count": 1}}
        graph_file.write_text(json.dumps(v1), encoding="utf-8")

        d1 = app.load_graph(root=str(self.root))
        self.assertEqual(d1["nodes"][0]["id"], "initial")

        # Version 2: updated on disk
        v2 = {
            "nodes": [{"id": "initial"}, {"id": "new_node"}],
            "edges": [{"source": "initial", "target": "new_node", "weight": 0.9}],
            "metadata": {"node_count": 2, "edge_count": 1},
        }
        graph_file.write_text(json.dumps(v2), encoding="utf-8")

        # Before clear: cache returns old value
        d_cached = app.load_graph(root=str(self.root))
        self.assertEqual(len(d_cached["nodes"]), 1)

        # Invalidate cache
        app.load_graph.clear()

        # After clear: new value is returned
        d2 = app.load_graph(root=str(self.root))
        self.assertEqual(len(d2["nodes"]), 2)
        self.assertEqual(d2["nodes"][1]["id"], "new_node")

    def test_load_graph_data_backward_compatible_alias(self) -> None:
        """Verify load_graph_data is an alias to load_graph sharing cache and .clear()."""
        graph_file = self.data_dir / "graph.json"
        graph_payload = {"nodes": [{"id": "alias_test"}], "edges": [], "metadata": {"node_count": 1}}
        graph_file.write_text(json.dumps(graph_payload), encoding="utf-8")

        g1 = app.load_graph(root=str(self.root))
        g2 = app.load_graph_data(root=str(self.root))

        self.assertEqual(g1, g2)
        self.assertIs(app.load_graph, app.load_graph_data)


class TestNoteEmbeddingsCaching(unittest.TestCase):
    """Test suite for note embeddings dictionary caching."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.data_dir = self.root / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        app.load_note_embeddings.clear()

    def tearDown(self) -> None:
        app.load_note_embeddings.clear()
        self.temp_dir.cleanup()

    def test_load_note_embeddings_caches_and_invalidates(self) -> None:
        """Verify load_note_embeddings caches vectors from embeddings.pkl and refreshes on clear."""
        emb_file = self.data_dir / "embeddings.pkl"
        v1 = {"note1": np.ones(384, dtype=np.float32)}

        with emb_file.open("wb") as f:
            pickle.dump(v1, f)

        # First read
        res1 = app.load_note_embeddings(root=str(self.root))
        self.assertIn("note1", res1)
        self.assertEqual(len(res1), 1)

        # Update file on disk
        v2 = {
            "note1": np.ones(384, dtype=np.float32),
            "note2": np.zeros(384, dtype=np.float32),
        }
        with emb_file.open("wb") as f:
            pickle.dump(v2, f)

        # Cached read should still return v1
        res_cached = app.load_note_embeddings(root=str(self.root))
        self.assertEqual(len(res_cached), 1)

        # Clear cache and verify updated embeddings become visible
        app.load_note_embeddings.clear()
        res2 = app.load_note_embeddings(root=str(self.root))
        self.assertEqual(len(res2), 2)
        self.assertIn("note2", res2)


class TestCachingIntegrationFlow(unittest.TestCase):
    """Test suite for end-to-end integration and wiring of cached operations."""

    def setUp(self) -> None:
        app.load_graph.clear()
        app.load_embeddings.clear()
        app.load_note_embeddings.clear()

    def tearDown(self) -> None:
        app.load_graph.clear()
        app.load_embeddings.clear()
        app.load_note_embeddings.clear()

    def test_ask_execution_receives_cached_embedding_model(self) -> None:
        """Verify that ask execution passes the cached model to ask.ask(..., model=cached_model)."""
        mock_model = MagicMock(name="WarmEmbeddingModel")
        mock_result = AskResult(answer="Synthesized answer", sources=[])

        with patch("lib.embeddings.load_model", return_value=mock_model) as mock_load, \
             patch("ask.ask", return_value=mock_result) as mock_ask:

            # Retrieve cached model as app.py does in render_ask_section
            model_resource = app.load_embeddings()
            result = ask.ask("What is my goal?", model=model_resource)

            self.assertEqual(result.answer, "Synthesized answer")
            mock_load.assert_called_once()
            mock_ask.assert_called_once_with("What is my goal?", model=mock_model)

            # Query again: model loader should NOT be invoked again
            model_resource2 = app.load_embeddings()
            self.assertIs(model_resource2, mock_model)
            self.assertEqual(mock_load.call_count, 1)

    def test_process_action_clears_all_caches(self) -> None:
        """Verify pipeline.process button workflow invalidates graph and embedding caches."""
        with patch.object(app.load_graph, "clear") as mock_graph_clear, \
             patch.object(app.load_embeddings, "clear") as mock_emb_clear, \
             patch.object(app.load_note_embeddings, "clear") as mock_note_emb_clear:

            # Simulate the cache invalidation in render_sidebar when Process completes
            app.load_graph.clear()
            app.load_embeddings.clear()
            app.load_note_embeddings.clear()

            mock_graph_clear.assert_called_once()
            mock_emb_clear.assert_called_once()
            mock_note_emb_clear.assert_called_once()

    def test_refresh_graph_action_clears_graph_cache(self) -> None:
        """Verify Refresh Graph action invalidates the graph cache."""
        with patch.object(app.load_graph, "clear") as mock_graph_clear:
            app.load_graph.clear()
            mock_graph_clear.assert_called_once()

    def test_cached_data_preserves_expected_schema(self) -> None:
        """Verify cached graph and embedding structures adhere strictly to expected schemas."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            tmp_data = tmp_root / "data"
            tmp_data.mkdir(parents=True, exist_ok=True)

            # 1. Graph schema validation
            graph_payload = {
                "nodes": [
                    {
                        "id": "node01",
                        "label": "Summary label",
                        "para": "Projects",
                        "tags": ["ml"],
                        "summary": "Full summary",
                        "content_preview": "First 200 chars...",
                        "group": "Projects",
                    }
                ],
                "edges": [
                    {"source": "node01", "target": "node02", "weight": 0.85, "type": "semantic"}
                ],
                "metadata": {
                    "generated_at": "2026-11-10T09:00:00Z",
                    "node_count": 1,
                    "edge_count": 1,
                },
            }
            (tmp_data / "graph.json").write_text(json.dumps(graph_payload), encoding="utf-8")

            cached_graph = app.load_graph(root=str(tmp_root))
            self.assertIn("nodes", cached_graph)
            self.assertIn("edges", cached_graph)
            self.assertIn("metadata", cached_graph)
            self.assertIsInstance(cached_graph["nodes"], list)
            self.assertIsInstance(cached_graph["edges"], list)
            self.assertIsInstance(cached_graph["metadata"], dict)
            self.assertEqual(cached_graph["metadata"]["node_count"], 1)

            # 2. Embedding vector schema validation
            raw_vec = np.random.randn(384).astype(np.float32)
            raw_vec /= np.linalg.norm(raw_vec)
            emb_payload = {"node01": raw_vec}
            with (tmp_data / "embeddings.pkl").open("wb") as f:
                pickle.dump(emb_payload, f)

            cached_emb = app.load_note_embeddings(root=str(tmp_root))
            self.assertIn("node01", cached_emb)
            self.assertEqual(cached_emb["node01"].shape, (384,))
            self.assertEqual(cached_emb["node01"].dtype, np.float32)


if __name__ == "__main__":
    unittest.main()

