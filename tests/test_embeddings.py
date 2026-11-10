"""Unit tests for lib/embeddings.py embedding module (Task 2.2.1).

Validates:
- Embedding model loading, caching, and configuration consistency.
- Single-text embedding generation (shape (384,), float32, finite values).
- Batch multi-text embedding generation (shape (N, 384)).
- Empty and invalid input rejection (empty string, whitespace, empty list, bad types).
- Vector dimension validation and non-finite (NaN/Inf) rejection.
- Cosine similarity calculation, including zero-norm division-by-zero protection.
- Atomic persistence in data/embeddings.pkl (saving, loading, corruption handling).
- Isolated testing with mocked model to avoid unnecessary network calls.
"""

from pathlib import Path
import pickle
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
import lib.storage as storage


def _create_mock_encoder(dim: int = 384) -> MagicMock:
    """Create a mock SentenceTransformer that returns deterministic vectors."""
    mock_model = MagicMock()

    def _mock_encode(text, convert_to_numpy=True, show_progress_bar=False):
        if isinstance(text, str):
            # Deterministic vector based on string hash
            rng = np.random.RandomState(abs(hash(text)) % (2**31))
            vec = rng.randn(dim).astype(np.float32)
            # Normalize for convenience
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            return vec
        elif isinstance(text, (list, tuple)):
            rows = []
            for t in text:
                rng = np.random.RandomState(abs(hash(t)) % (2**31))
                vec = rng.randn(dim).astype(np.float32)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                rows.append(vec)
            return np.array(rows, dtype=np.float32)
        else:
            raise TypeError("Unsupported text type")

    mock_model.encode.side_effect = _mock_encode
    return mock_model


class TestEmbeddingsModule(unittest.TestCase):
    """Test suite for lib/embeddings.py."""

    def setUp(self) -> None:
        """Create isolated temporary directory for test storage."""
        self.test_dir = tempfile.mkdtemp(prefix="secondself_embeddings_test_")
        self.test_path = Path(self.test_dir)
        storage.set_project_root(self.test_path)
        (self.test_path / "data").mkdir(parents=True, exist_ok=True)
        emb.reset_model()

    def tearDown(self) -> None:
        """Clean up temporary directory and reset cached model."""
        emb.reset_model()
        storage.set_project_root(None)
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # 1. Model Configuration & Loading Tests
    # -------------------------------------------------------------------------

    def test_01_model_configuration_consistency(self) -> None:
        """Configured model name and dimension match project specifications."""
        self.assertEqual(emb.DEFAULT_MODEL_NAME, "all-MiniLM-L6-v2")
        self.assertEqual(emb.EMBEDDING_DIMENSION, 384)

        # Reads version from index.json when configured
        storage.save_index({"embeddings_version": "custom-model-v1"}, root=self.test_path)
        self.assertEqual(emb.get_configured_model_name(root=self.test_path), "custom-model-v1")

    def test_02_load_model_caching_and_singleton(self) -> None:
        """load_model caches the model instance and avoids redundant initialization."""
        mock_instance = _create_mock_encoder(384)

        with patch("sentence_transformers.SentenceTransformer", return_value=mock_instance) as mock_cls:
            model1 = emb.load_model()
            model2 = emb.load_model()

            # Verify singleton behavior
            self.assertIs(model1, model2)
            # Constructor was only called once
            self.assertEqual(mock_cls.call_count, 1)

            # Test force_reload re-initializes
            model3 = emb.load_model(force_reload=True)
            self.assertEqual(mock_cls.call_count, 2)

    def test_03_load_model_failure_handling(self) -> None:
        """load_model raises clear RuntimeError on initialization failure."""
        with patch("sentence_transformers.SentenceTransformer", side_effect=OSError("Network unreachable")):
            with self.assertRaises(RuntimeError) as cm:
                emb.load_model()
            self.assertIn("Failed to load sentence-transformers model", str(cm.exception))

    # -------------------------------------------------------------------------
    # 2. Embedding Generation Tests
    # -------------------------------------------------------------------------

    def test_04_embed_single_text_shape_and_dtype(self) -> None:
        """embed_text returns 1D float32 numpy array of shape (384,) for single text."""
        mock_model = _create_mock_encoder(384)

        text = "SecondSelf: Autonomous personal knowledge brain."
        vec = emb.embed_text(text, model=mock_model)

        self.assertIsInstance(vec, np.ndarray)
        self.assertEqual(vec.shape, (384,))
        self.assertEqual(vec.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(vec)))

    def test_05_embed_batch_texts_shape_and_dtype(self) -> None:
        """embed_text returns 2D float32 numpy array of shape (N, 384) for batch sequence."""
        mock_model = _create_mock_encoder(384)

        texts = [
            "First note about knowledge graph embeddings.",
            "Second note about vector database indexing.",
            "Third note about PARA organization.",
        ]
        vecs = emb.embed_text(texts, model=mock_model)

        self.assertIsInstance(vecs, np.ndarray)
        self.assertEqual(vecs.shape, (3, 384))
        self.assertEqual(vecs.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(vecs)))

    def test_06_embed_text_empty_input_rejected(self) -> None:
        """embed_text rejects empty string, whitespace string, and empty batch list."""
        mock_model = _create_mock_encoder(384)

        # Empty string
        with self.assertRaises(ValueError) as cm:
            emb.embed_text("", model=mock_model)
        self.assertIn("empty or whitespace", str(cm.exception))

        # Whitespace-only string
        with self.assertRaises(ValueError) as cm:
            emb.embed_text("   \n\t  ", model=mock_model)
        self.assertIn("empty or whitespace", str(cm.exception))

        # Empty batch list
        with self.assertRaises(ValueError) as cm:
            emb.embed_text([], model=mock_model)
        self.assertIn("empty batch list", str(cm.exception))

        # Batch containing whitespace string
        with self.assertRaises(ValueError) as cm:
            emb.embed_text(["valid note", "   "], model=mock_model)
        self.assertIn("non-empty", str(cm.exception))

        # Invalid type (e.g. int, dict)
        with self.assertRaises(TypeError):
            emb.embed_text(12345, model=mock_model)  # type: ignore[arg-type]

    def test_07_embed_text_dimension_mismatch_validation(self) -> None:
        """embed_text raises ValueError if model produces wrong vector dimension."""
        mock_model_768 = _create_mock_encoder(dim=768)

        with self.assertRaises(ValueError) as cm:
            emb.embed_text("Sample text", model=mock_model_768)
        self.assertIn("Expected embedding dimension (384,)", str(cm.exception))

    def test_08_embed_text_non_finite_values_rejected(self) -> None:
        """embed_text raises ValueError if model output contains NaN or Inf."""
        bad_model = MagicMock()
        nan_vec = np.zeros(384, dtype=np.float32)
        nan_vec[10] = np.nan
        bad_model.encode.return_value = nan_vec

        with self.assertRaises(ValueError) as cm:
            emb.embed_text("Sample note", model=bad_model)
        self.assertIn("non-finite values", str(cm.exception))

    # -------------------------------------------------------------------------
    # 3. Cosine Similarity Tests
    # -------------------------------------------------------------------------

    def test_09_cosine_similarity_identical_orthogonal_opposite(self) -> None:
        """cosine_similarity correctly evaluates standard angle geometries."""
        # Identical vectors -> 1.0
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([2.0, 0.0, 0.0], dtype=np.float32)
        self.assertAlmostEqual(emb.cosine_similarity(v1, v2), 1.0, places=5)

        # Orthogonal vectors -> 0.0
        v_orth = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.assertAlmostEqual(emb.cosine_similarity(v1, v_orth), 0.0, places=5)

        # Opposite vectors -> -1.0
        v_opp = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        self.assertAlmostEqual(emb.cosine_similarity(v1, v_opp), -1.0, places=5)

    def test_10_cosine_similarity_zero_norm_division_by_zero_safe(self) -> None:
        """cosine_similarity returns 0.0 for all-zero vectors without division by zero error."""
        v_zero = np.zeros(384, dtype=np.float32)
        v_valid = np.ones(384, dtype=np.float32)

        # Left zero
        self.assertEqual(emb.cosine_similarity(v_zero, v_valid), 0.0)
        # Right zero
        self.assertEqual(emb.cosine_similarity(v_valid, v_zero), 0.0)
        # Both zero
        self.assertEqual(emb.cosine_similarity(v_zero, v_zero), 0.0)

    def test_11_cosine_similarity_mismatched_shapes_and_nans_rejected(self) -> None:
        """cosine_similarity rejects mismatched dimensions and non-finite numbers."""
        v3 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        v4 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        with self.assertRaises(ValueError) as cm:
            emb.cosine_similarity(v3, v4)
        self.assertIn("Embedding shapes must match", str(cm.exception))

        v_nan = np.array([1.0, np.nan, 3.0], dtype=np.float32)
        with self.assertRaises(ValueError) as cm:
            emb.cosine_similarity(v3, v_nan)
        self.assertIn("non-finite values", str(cm.exception))

    # -------------------------------------------------------------------------
    # 4. Persistence Tests (data/embeddings.pkl)
    # -------------------------------------------------------------------------

    def test_12_save_and_load_embeddings_roundtrip(self) -> None:
        """save_embeddings and load_embeddings serialize and restore vectors accurately."""
        # Create test embeddings
        rng = np.random.RandomState(42)
        vec1 = rng.randn(384).astype(np.float32)
        vec2 = rng.randn(384).astype(np.float32)

        data = {
            "note001": vec1,
            "note002": vec2,
        }

        saved_path = emb.save_embeddings(data, root=self.test_path)
        self.assertTrue(saved_path.exists())
        self.assertEqual(saved_path, self.test_path / "data" / "embeddings.pkl")

        # Load back
        loaded = emb.load_embeddings(root=self.test_path)
        self.assertEqual(set(loaded.keys()), {"note001", "note002"})
        np.testing.assert_allclose(loaded["note001"], vec1, rtol=1e-5)
        np.testing.assert_allclose(loaded["note002"], vec2, rtol=1e-5)

    def test_13_load_embeddings_missing_file_returns_empty_dict(self) -> None:
        """load_embeddings returns empty dict when embeddings.pkl does not exist."""
        loaded = emb.load_embeddings(root=self.test_path)
        self.assertEqual(loaded, {})

    def test_14_save_embeddings_validates_dimensions_and_keys(self) -> None:
        """save_embeddings rejects invalid types, wrong dimensions, or non-finite values."""
        # Wrong dimension
        bad_dim_data = {"bad_note": np.zeros(100, dtype=np.float32)}
        with self.assertRaises(ValueError) as cm:
            emb.save_embeddings(bad_dim_data, root=self.test_path)
        self.assertIn("invalid shape", str(cm.exception))

        # Non-finite values
        nan_vec = np.zeros(384, dtype=np.float32)
        nan_vec[0] = np.nan
        with self.assertRaises(ValueError) as cm:
            emb.save_embeddings({"nan_note": nan_vec}, root=self.test_path)
        self.assertIn("non-finite values", str(cm.exception))

        # Non-dict input
        with self.assertRaises(ValueError):
            emb.save_embeddings(["not", "a", "dict"], root=self.test_path)  # type: ignore[arg-type]

    def test_15_load_embeddings_corrupted_file_handling(self) -> None:
        """load_embeddings raises clear ValueError on corrupted pickle data."""
        pkl_path = emb.get_embeddings_path(root=self.test_path)
        pkl_path.write_bytes(b"CORRUPTED_NON_PICKLE_BYTES\x00\xff\xfe")

        with self.assertRaises(ValueError) as cm:
            emb.load_embeddings(root=self.test_path)
        self.assertIn("Failed to load embeddings", str(cm.exception))


if __name__ == "__main__":
    unittest.main()

