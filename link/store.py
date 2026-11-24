"""SecondSelf — Persistent Append-Only Embedding Store and Similarity Search.

Provides immutable, content-addressed storage for dense vector representations,
with sub-millisecond in-memory NumPy cosine similarity querying and export to public/index.json.
"""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Any
import numpy as np

from config import get_settings
from config.base import PathsConfig, VectorConfig
from link.models import Embedding

logger = logging.getLogger("secondself.link.store")


class EmbeddingStore:
    """Content-addressed, append-only embedding repository with vectorized similarity search."""

    def __init__(
        self,
        index_file: Path | str | None = None,
        config: VectorConfig | None = None,
        auto_load: bool = True,
    ) -> None:
        self.config = config or get_settings().vector
        if index_file is not None:
            self.index_file = Path(index_file).resolve()
        else:
            self.index_file = get_settings().paths.index_file.resolve()

        # In-memory index structures
        self._ids: list[str] = []
        self._embeddings: dict[str, Embedding] = {}  # keyed by record_id
        self._hash_to_id: dict[str, str] = {}        # content_hash -> record_id
        self._metadata: dict[str, dict[str, Any]] = {}
        self._matrix: np.ndarray | None = None       # shape: (N, dimensions)

        # Attempt to load existing precomputed index
        if auto_load and self.index_file.exists():
            try:
                self.load_index(self.index_file)
            except Exception as exc:
                logger.warning("Could not pre-load index from '%s': %s", self.index_file, exc)

    def __len__(self) -> int:
        return len(self._ids)

    def contains_hash(self, content_hash: str) -> bool:
        """Returns True if an embedding with this content hash already exists."""
        return content_hash in self._hash_to_id

    def contains_id(self, record_id: str) -> bool:
        """Returns True if an embedding for this record_id already exists."""
        return record_id in self._embeddings

    def get_by_hash(self, content_hash: str) -> Embedding | None:
        """Retrieves embedding by content hash if model matches."""
        rec_id = self._hash_to_id.get(content_hash)
        if rec_id and rec_id in self._embeddings:
            emb = self._embeddings[rec_id]
            if emb.model_name == self.config.model_name and emb.dimensions == self.config.dimensions:
                return emb
        return None

    def get_by_id(self, record_id: str) -> Embedding | None:
        """Retrieves embedding by record ID."""
        return self._embeddings.get(record_id)

    def get_metadata(self, record_id: str) -> dict[str, Any]:
        """Retrieves note metadata associated with record ID."""
        return self._metadata.get(record_id, {})

    def add(
        self,
        record_id: str,
        embedding: Embedding,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Adds or updates an embedding entry in the store.

        Returns:
            True if newly added or updated, False if skipped as unchanged.
        """
        if embedding.status != "success" or not embedding.vector:
            logger.warning("Skipping addition of failed/empty embedding for '%s'", record_id)
            return False

        meta = dict(metadata or {})
        existing = self._embeddings.get(record_id)

        # Idempotency check: Skip if exact same content hash and model already stored
        if existing and existing.content_hash == embedding.content_hash and existing.model_name == embedding.model_name:
            self._metadata[record_id] = meta
            return False

        if record_id not in self._ids:
            self._ids.append(record_id)

        self._embeddings[record_id] = embedding
        self._hash_to_id[embedding.content_hash] = record_id
        self._metadata[record_id] = meta
        self._matrix = None  # Invalidate cached matrix
        return True

    def add_batch(
        self,
        entries: list[tuple[str, Embedding, dict[str, Any]]],
    ) -> int:
        """Adds a batch of (record_id, embedding, metadata) tuples. Returns added count."""
        added_count = 0
        for rec_id, emb, meta in entries:
            if self.add(rec_id, emb, meta):
                added_count += 1
        return added_count

    def _ensure_matrix(self) -> np.ndarray:
        """Constructs or retrieves the stacked 2D float32 vector matrix (N, dimensions)."""
        if self._matrix is not None and len(self._matrix) == len(self._ids):
            return self._matrix

        if not self._ids:
            self._matrix = np.empty((0, self.config.dimensions), dtype=np.float32)
            return self._matrix

        vectors = [self._embeddings[rec_id].to_numpy() for rec_id in self._ids]
        self._matrix = np.vstack(vectors).astype(np.float32)
        return self._matrix

    def query(
        self,
        vector: list[float] | np.ndarray,
        top_k: int = 5,
        min_score: float = 0.0,
        exclude_id: str | None = None,
    ) -> list[tuple[str, float]]:
        """Performs exact nearest-neighbor cosine similarity search using NumPy dot products.

        Args:
            vector: Target normalized query vector (1D array of floats).
            top_k: Maximum number of closest items to return.
            min_score: Minimum cosine similarity threshold (e.g. 0.65).
            exclude_id: Optional ID to exclude from results (e.g. self-note).

        Returns:
            List of (record_id, similarity_score) sorted deterministically by score descending,
            with stable alphabetical tie-breaking by ID.
        """
        if len(self._ids) == 0:
            return []

        matrix = self._ensure_matrix()
        q_vec = np.asarray(vector, dtype=np.float32)

        # Normalize query vector if not already unit length
        q_norm = float(np.linalg.norm(q_vec))
        if q_norm > 0 and abs(q_norm - 1.0) > 1e-4:
            q_vec = q_vec / q_norm

        # Vectorized dot products: S = M · q
        scores = np.dot(matrix, q_vec)

        candidates: list[tuple[str, float]] = []
        for rec_id, score in zip(self._ids, scores):
            score_float = float(score)
            if exclude_id and rec_id == exclude_id:
                continue
            if score_float >= min_score:
                candidates.append((rec_id, score_float))

        # Deterministic sorting: highest score first, ties broken by record_id ascending
        candidates.sort(key=lambda item: (-item[1], item[0]))
        return candidates[:top_k]

    def export_index(self, output_path: Path | str | None = None) -> Path:
        """Exports the in-memory vectors and note metadata into portable public/index.json.

        Adheres strictly to the schema specified in ADR-003 and implementation plan:
        {
          "model": "all-MiniLM-L6-v2",
          "dimensions": 384,
          "generated_at": "ISO-TIMESTAMP",
          "notes": [ ... ]
        }
        """
        dest = Path(output_path or self.index_file).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)

        notes_payload: list[dict[str, Any]] = []
        for rec_id in sorted(self._ids):
            emb = self._embeddings[rec_id]
            meta = self._metadata.get(rec_id, {})
            notes_payload.append({
                "id": rec_id,
                "title": meta.get("title", "Untitled"),
                "category": meta.get("category", "Resources"),
                "summary": meta.get("summary", ""),
                "tags": meta.get("tags", []),
                "content": meta.get("content", ""),
                "content_hash": emb.content_hash,
                "embedding": [round(float(v), 5) for v in emb.vector],
            })

        document = {
            "model": self.config.model_name,
            "dimensions": self.config.dimensions,
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "notes": notes_payload,
        }

        dest.write_text(json.dumps(document, indent=2), encoding="utf-8")
        logger.info("Exported semantic index with %d notes to '%s'", len(notes_payload), dest)
        return dest

    def load_index(self, input_path: Path | str | None = None) -> int:
        """Loads precomputed vectors and metadata from an existing public/index.json file.

        Returns:
            Number of successfully loaded notes.
        """
        src = Path(input_path or self.index_file).resolve()
        if not src.exists():
            return 0

        data = json.loads(src.read_text(encoding="utf-8"))
        file_dim = data.get("dimensions")
        if file_dim is not None and file_dim != self.config.dimensions:
            logger.warning(
                "Index file dimensions (%d) do not match store config (%d); skipping load from '%s'",
                file_dim,
                self.config.dimensions,
                src,
            )
            return 0

        model_name = data.get("model", self.config.model_name)
        dimensions = self.config.dimensions
        notes = data.get("notes", [])

        loaded_count = 0
        for n in notes:
            rec_id = n.get("id")
            raw_vec = n.get("embedding", [])
            c_hash = n.get("content_hash", "")
            if not rec_id or not raw_vec or len(raw_vec) != dimensions:
                continue

            emb = Embedding(
                vector=raw_vec,
                model_name=model_name,
                dimensions=dimensions,
                content_hash=c_hash or rec_id,
                status="success",
            )
            meta = {
                "title": n.get("title", "Untitled"),
                "category": n.get("category", "Resources"),
                "summary": n.get("summary", ""),
                "tags": n.get("tags", []),
                "content": n.get("content", ""),
            }
            if self.add(rec_id, emb, meta):
                loaded_count += 1

        self._ensure_matrix()
        logger.info("Loaded %d notes from index file '%s'", loaded_count, src)
        return loaded_count
