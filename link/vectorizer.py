"""SecondSelf — Semantic Vectorization Engine with Provider Abstraction.

Implements batched, cached, deterministic dense embedding generation with
graceful degradation and pluggable provider backends.
"""

from __future__ import annotations

import abc
import hashlib
import logging
import time
from typing import Any
import numpy as np

from config import get_settings
from config.base import VectorConfig
from link.models import Embedding

logger = logging.getLogger("secondself.link.vectorizer")


class EmbeddingProvider(abc.ABC):
    """Abstract interface for dense semantic embedding backends."""

    @abc.abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Transforms a batch of raw strings into normalized dense float vectors."""
        pass


class SentenceTransformerProvider(EmbeddingProvider):
    """Local embedding provider utilizing HuggingFace sentence-transformers."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model: Any = None

    def _load_model(self) -> Any:
        """Lazily imports and instantiates the SentenceTransformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                logger.info("Loading sentence-transformers model: %s", self.model_name)
                self._model = SentenceTransformer(self.model_name)
            except Exception as exc:
                logger.error("Failed to load SentenceTransformer: %s", exc)
                raise RuntimeError(f"Could not initialize model '{self.model_name}': {exc}") from exc
        return self._model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Encodes texts into L2-normalized float lists."""
        if not texts:
            return []
        model = self._load_model()
        vectors = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [vec.tolist() for vec in vectors]


class MockEmbeddingProvider(EmbeddingProvider):
    """Deterministic synthetic provider for testing and offline environments.

    Generates reproducible unit-normalized vectors derived from the SHA-256
    digest of the input content. Never invokes GPU or external networks.
    """

    def __init__(self, dimensions: int = 384, should_fail: bool = False) -> None:
        self.dimensions = dimensions
        self.should_fail = should_fail

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Produces deterministic pseudo-random unit-normalized vectors."""
        if self.should_fail:
            raise ConnectionError("Simulated embedding provider outage / timeout.")

        results: list[list[float]] = []
        for text in texts:
            # Seed standard pseudo-random generator with content hash integer
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            seed_int = int.from_bytes(digest[:8], byteorder="big")
            rng = np.random.default_rng(seed_int)

            raw = rng.standard_normal(self.dimensions, dtype=np.float32)
            norm = np.linalg.norm(raw)
            normalized = (raw / (norm if norm > 0 else 1.0)).tolist()
            results.append(normalized)

        return results


class Vectorizer:
    """Orchestrator for dense text vectorization with caching and retry logic."""

    def __init__(
        self,
        config: VectorConfig | None = None,
        provider: EmbeddingProvider | None = None,
    ) -> None:
        self.config = config or get_settings().vector
        self.provider = provider or SentenceTransformerProvider(model_name=self.config.model_name)
        # In-memory cache keyed by "content_hash:model_name:dimensions"
        self._cache: dict[str, Embedding] = {}

    def _compute_content_hash(self, text: str) -> str:
        """Computes deterministic SHA-256 hex digest of stripped UTF-8 string."""
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

    def _cache_key(self, content_hash: str) -> str:
        """Derives unique cache key incorporating model identity and dimensionality."""
        return f"{content_hash}:{self.config.model_name}:{self.config.dimensions}"

    def embed(self, content: str, record_id: str | None = None) -> Embedding:
        """Encodes single text into an Embedding model with caching and retry logic.

        Args:
            content: Raw text or composite document string.
            record_id: Optional record ID for diagnostic tracing.

        Returns:
            Normalized Embedding model instance.
        """
        content_hash = self._compute_content_hash(content)
        cache_key = self._cache_key(content_hash)

        if self.config.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]

        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                vectors = self.provider.embed_texts([content])
                if not vectors or len(vectors[0]) != self.config.dimensions:
                    actual_dim = len(vectors[0]) if vectors else 0
                    raise ValueError(
                        f"Provider returned vector of dimension {actual_dim}, "
                        f"expected {self.config.dimensions}"
                    )

                vec = vectors[0]
                # Ensure unit normalization
                arr = np.asarray(vec, dtype=np.float32)
                norm = float(np.linalg.norm(arr))
                if norm > 0 and abs(norm - 1.0) > 1e-4:
                    arr = arr / norm
                    vec = arr.tolist()

                result = Embedding(
                    vector=vec,
                    model_name=self.config.model_name,
                    dimensions=self.config.dimensions,
                    content_hash=content_hash,
                    status="success",
                )

                if self.config.cache_enabled:
                    self._cache[cache_key] = result
                return result

            except Exception as exc:
                last_error = exc
                wait_seconds = (2 ** attempt) * 0.5
                logger.warning(
                    "Embedding attempt %d failed for record '%s': %s. Retrying in %.1fs...",
                    attempt + 1,
                    record_id or content_hash[:8],
                    exc,
                    wait_seconds,
                )
                time.sleep(wait_seconds)

        # Graceful Degradation: Log failure and return marked failed embedding without crashing
        logger.error(
            "Vectorization permanently failed after %d retries for record '%s': %s",
            self.config.max_retries,
            record_id or content_hash[:8],
            last_error,
        )
        failed_embedding = Embedding(
            vector=[],
            model_name=self.config.model_name,
            dimensions=self.config.dimensions,
            content_hash=content_hash,
            status="failed",
            error_message=str(last_error),
        )
        return failed_embedding

    def embed_batch(self, items: list[tuple[str, str]]) -> list[Embedding]:
        """Encodes a batch of (record_id, content) tuples preserving original input order.

        Args:
            items: Ordered list of tuples containing (record_id, text_content).

        Returns:
            List of Embedding instances corresponding 1:1 with input items.
        """
        if not items:
            return []

        results: list[Embedding | None] = [None] * len(items)
        missing_indices: list[int] = []
        missing_texts: list[str] = []

        # 1. Check cache for all items
        for idx, (rec_id, text) in enumerate(items):
            content_hash = self._compute_content_hash(text)
            cache_key = self._cache_key(content_hash)

            if self.config.cache_enabled and cache_key in self._cache:
                results[idx] = self._cache[cache_key]
            else:
                missing_indices.append(idx)
                missing_texts.append(text)

        # 2. Batch-encode cache misses in chunks of batch_size
        batch_size = max(1, self.config.batch_size)
        for chunk_start in range(0, len(missing_texts), batch_size):
            chunk_end = chunk_start + batch_size
            chunk_texts = missing_texts[chunk_start:chunk_end]
            chunk_indices = missing_indices[chunk_start:chunk_end]

            try:
                vectors = self.provider.embed_texts(chunk_texts)
                for c_idx, vec in zip(chunk_indices, vectors):
                    rec_id, text = items[c_idx]
                    content_hash = self._compute_content_hash(text)

                    arr = np.asarray(vec, dtype=np.float32)
                    norm = float(np.linalg.norm(arr))
                    if norm > 0 and abs(norm - 1.0) > 1e-4:
                        arr = arr / norm
                        vec = arr.tolist()

                    emb = Embedding(
                        vector=vec,
                        model_name=self.config.model_name,
                        dimensions=self.config.dimensions,
                        content_hash=content_hash,
                        status="success",
                    )
                    results[c_idx] = emb
                    if self.config.cache_enabled:
                        self._cache[self._cache_key(content_hash)] = emb

            except Exception as exc:
                logger.error("Batch embedding chunk failed: %s. Falling back to item-by-item retry.", exc)
                for c_idx in chunk_indices:
                    rec_id, text = items[c_idx]
                    results[c_idx] = self.embed(text, record_id=rec_id)

        # 3. Verify all slots populated
        final_embeddings: list[Embedding] = []
        for idx, res in enumerate(results):
            if res is None:
                rec_id, text = items[idx]
                final_embeddings.append(self.embed(text, record_id=rec_id))
            else:
                final_embeddings.append(res)

        return final_embeddings
