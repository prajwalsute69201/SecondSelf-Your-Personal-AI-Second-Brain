"""Embeddings generation and similarity module for SecondSelf (Task 2.2.1).

Wraps sentence-transformers to convert knowledge note text into 384-dimensional
semantic embedding vectors using all-MiniLM-L6-v2, computes vector cosine similarity,
and manages persistence in data/embeddings.pkl.
"""

from __future__ import annotations

import os
from pathlib import Path
import pickle
import sys
import tempfile
from typing import Any, Sequence

import numpy as np

# Ensure repository root is on sys.path so 'lib' is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    import lib.storage as storage
except ImportError:
    from . import storage  # type: ignore[no-redef]

# Centralized model configuration
DEFAULT_MODEL_NAME: str = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION: int = 384

# Module-level model cache
_cached_model: Any = None
_cached_model_name: str | None = None

__all__ = [
    "DEFAULT_MODEL_NAME",
    "EMBEDDING_DIMENSION",
    "get_configured_model_name",
    "load_model",
    "reset_model",
    "embed_text",
    "cosine_similarity",
    "get_embeddings_path",
    "load_embeddings",
    "save_embeddings",
]


def get_configured_model_name(root: Path | str | None = None) -> str:
    """Return configured model identifier from data/index.json or default."""
    try:
        index = storage.load_index(root=root)
        version = index.get("embeddings_version")
        if version and isinstance(version, str) and version.strip():
            return version.strip()
    except Exception:
        pass
    return DEFAULT_MODEL_NAME


def reset_model() -> None:
    """Clear cached model instance (useful in tests)."""
    global _cached_model, _cached_model_name
    _cached_model = None
    _cached_model_name = None


def load_model(
    model_name: str | None = None,
    force_reload: bool = False,
    root: Path | str | None = None,
) -> Any:
    """Load and cache the sentence-transformers embedding model.

    Args:
        model_name: Optional model identifier override (defaults to configured/default model).
        force_reload: If True, ignores cached model instance and re-initializes.
        root: Optional project root path override for retrieving configured version.

    Returns:
        SentenceTransformer: Initialized sentence-transformers model instance.

    Raises:
        RuntimeError: If sentence-transformers is not installed or model loading fails.
    """
    global _cached_model, _cached_model_name

    target_name = model_name or get_configured_model_name(root=root)

    if _cached_model is not None and not force_reload and _cached_model_name == target_name:
        return _cached_model

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise RuntimeError(
            "The 'sentence-transformers' package is required for embedding generation. "
            "Please ensure it is installed in your virtual environment."
        ) from e

    try:
        model = SentenceTransformer(target_name)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load sentence-transformers model {target_name!r}: {e}"
        ) from e

    _cached_model = model
    _cached_model_name = target_name
    return model


def embed_text(
    text: str | Sequence[str],
    model: Any | None = None,
    root: Path | str | None = None,
) -> np.ndarray:
    """Generate 384-dimensional numerical embedding vector(s) for input text.

    Args:
        text: Single text string, or a sequence/list of text strings for batch encoding.
        model: Optional pre-loaded SentenceTransformer instance.
        root: Optional project root path override.

    Returns:
        np.ndarray:
            - If text is str: 1D array of shape (384,) and dtype float32.
            - If text is sequence of str: 2D array of shape (N, 384) and dtype float32.

    Raises:
        ValueError: If input is empty, whitespace-only, contains non-finite values,
            or produces an unexpected vector dimension.
        TypeError: If input text is neither a str nor a sequence of strings.
        RuntimeError: If model fails to encode.
    """
    # 1. Single string input
    if isinstance(text, str):
        if not text or not text.strip():
            raise ValueError("Cannot generate embedding for empty or whitespace-only text.")

        active_model = model or load_model(root=root)
        try:
            raw_vec = active_model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as e:
            raise RuntimeError(f"Embedding encoding failed: {e}") from e

        vec = np.asarray(raw_vec, dtype=np.float32)

        # Handle models returning 2D array of shape (1, 384)
        if vec.ndim == 2 and vec.shape[0] == 1:
            vec = vec[0]

        if vec.shape != (EMBEDDING_DIMENSION,):
            raise ValueError(
                f"Expected embedding dimension ({EMBEDDING_DIMENSION},), got {vec.shape}"
            )

        if not np.all(np.isfinite(vec)):
            raise ValueError("Generated embedding contains non-finite values (NaN or Inf).")

        return vec

    # 2. Batch sequence input
    elif isinstance(text, (list, tuple)):
        if len(text) == 0:
            raise ValueError("Cannot generate embeddings for an empty batch list.")

        for i, item in enumerate(text):
            if not isinstance(item, str) or not item.strip():
                raise ValueError(
                    f"Batch item at index {i} must be a non-empty, non-whitespace string."
                )

        active_model = model or load_model(root=root)
        try:
            raw_vecs = active_model.encode(
                list(text),
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as e:
            raise RuntimeError(f"Batch embedding encoding failed: {e}") from e

        vecs = np.asarray(raw_vecs, dtype=np.float32)

        if vecs.ndim != 2 or vecs.shape[1] != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Expected batch embedding shape (N, {EMBEDDING_DIMENSION}), got {vecs.shape}"
            )

        if not np.all(np.isfinite(vecs)):
            raise ValueError("Generated batch embeddings contain non-finite values (NaN or Inf).")

        return vecs

    else:
        raise TypeError(
            f"Input text must be a string or sequence of strings, got {type(text).__name__}"
        )


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1D embedding vectors.

    Formula: (a . b) / (||a|| * ||b||).
    Handles zero-norm edge case safely by returning 0.0 to prevent division by zero.

    Args:
        a: First embedding vector (1D array or array-like).
        b: Second embedding vector (1D array or array-like).

    Returns:
        float: Cosine similarity score in range [-1.0, 1.0].

    Raises:
        ValueError: If vectors have mismatched lengths, invalid dimensions, or contain NaN/Inf.
    """
    arr_a = np.asarray(a, dtype=np.float32).ravel()
    arr_b = np.asarray(b, dtype=np.float32).ravel()

    if arr_a.shape != arr_b.shape:
        raise ValueError(
            f"Embedding shapes must match: {arr_a.shape} vs {arr_b.shape}"
        )

    if not np.all(np.isfinite(arr_a)) or not np.all(np.isfinite(arr_b)):
        raise ValueError("Input vectors contain non-finite values (NaN or Inf).")

    norm_a = float(np.linalg.norm(arr_a))
    norm_b = float(np.linalg.norm(arr_b))

    # Guard against division by zero (edge case 3.3)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    dot = float(np.dot(arr_a, arr_b))
    score = dot / (norm_a * norm_b)

    # Clamp floating-point inaccuracies into valid [-1.0, 1.0] interval
    return float(np.clip(score, -1.0, 1.0))


def get_embeddings_path(root: Path | str | None = None) -> Path:
    """Return filesystem path to data/embeddings.pkl."""
    return storage.get_data_dir(root) / "embeddings.pkl"


def load_embeddings(root: Path | str | None = None) -> dict[str, np.ndarray]:
    """Load note embeddings dictionary from data/embeddings.pkl.

    Args:
        root: Optional project root path override.

    Returns:
        dict[str, np.ndarray]: Mapping of note ID to 1D embedding vector.
            Returns empty dict if data/embeddings.pkl does not exist.

    Raises:
        ValueError: If file is corrupted or contains invalid embedding data.
    """
    path = get_embeddings_path(root)
    if not path.exists():
        return {}

    try:
        with path.open("rb") as f:
            data = pickle.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load embeddings from {path}: {e}") from e

    if not isinstance(data, dict):
        raise ValueError(
            f"Embeddings file at {path} must contain a dict, got {type(data).__name__}"
        )

    validated: dict[str, np.ndarray] = {}
    for key, val in data.items():
        if not isinstance(key, str):
            raise ValueError(f"Embedding key must be str, got {type(key).__name__}")
        arr = np.asarray(val, dtype=np.float32)
        if arr.shape != (EMBEDDING_DIMENSION,):
            raise ValueError(
                f"Embedding for note {key!r} has invalid shape {arr.shape}, expected ({EMBEDDING_DIMENSION},)"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Embedding for note {key!r} contains non-finite values (NaN or Inf)")
        validated[key] = arr

    return validated


def save_embeddings(
    embeddings: dict[str, np.ndarray],
    root: Path | str | None = None,
) -> Path:
    """Save note embeddings dictionary atomically to data/embeddings.pkl.

    Args:
        embeddings: Mapping of note ID to 1D numpy array.
        root: Optional project root path override.

    Returns:
        Path: Path to saved data/embeddings.pkl.

    Raises:
        ValueError: If embeddings is not a dict or contains invalid vectors.
        OSError: If filesystem write fails.
    """
    if not isinstance(embeddings, dict):
        raise ValueError(f"embeddings must be a dict, got {type(embeddings).__name__}")

    # Validate all entries before writing
    to_save: dict[str, np.ndarray] = {}
    for key, val in embeddings.items():
        if not isinstance(key, str):
            raise ValueError(f"Embedding key must be str, got {type(key).__name__}")
        arr = np.asarray(val, dtype=np.float32)
        if arr.shape != (EMBEDDING_DIMENSION,):
            raise ValueError(
                f"Embedding for note {key!r} has invalid shape {arr.shape}, expected ({EMBEDDING_DIMENSION},)"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Embedding for note {key!r} contains non-finite values (NaN or Inf)")
        to_save[key] = arr

    target_path = get_embeddings_path(root)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Atomic write using NamedTemporaryFile in the same directory
    temp_file: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target_path.parent,
            delete=False,
            suffix=".tmp",
        ) as f:
            temp_file = Path(f.name)
            pickle.dump(to_save, f, protocol=pickle.HIGHEST_PROTOCOL)

        temp_file.replace(target_path)
    except Exception as e:
        if temp_file and temp_file.exists():
            temp_file.unlink(missing_ok=True)
        raise OSError(f"Failed to write embeddings to {target_path}: {e}") from e

    return target_path

