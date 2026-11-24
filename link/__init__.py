"""SecondSelf — Semantic Vectorization & Auto-Linking Module.

Exposes the unified API for semantic embeddings, append-only vector storage,
cosine similarity querying, PARA centroid alignment, and autonomous Wikilink generation.
"""

from __future__ import annotations

from config.base import VectorConfig
from link.centroids import PARACentroidCalculator
from link.cli import main, scan_wiki_vault
from link.linker import AutoLinker
from link.models import Embedding, Link, LinkResult, NoteItem
from link.store import EmbeddingStore
from link.vectorizer import (
    EmbeddingProvider,
    MockEmbeddingProvider,
    SentenceTransformerProvider,
    Vectorizer,
)

_GLOBAL_LINKER: AutoLinker | None = None


def embed(content: str, config: VectorConfig | None = None) -> Embedding:
    """Convenience helper to compute a normalized embedding for text."""
    vectorizer = Vectorizer(config=config)
    return vectorizer.embed(content)


def link_item(
    item: NoteItem,
    embedding: Embedding,
    config: VectorConfig | None = None,
    store: EmbeddingStore | None = None,
) -> LinkResult:
    """Convenience helper to link a single note against the store."""
    linker = AutoLinker(config=config, store=store)
    return linker.link_item(item, embedding)


def link_all(
    notes: list[NoteItem],
    config: VectorConfig | None = None,
    dry_run: bool = False,
) -> list[LinkResult]:
    """Convenience helper to batch link an entire list of notes."""
    linker = AutoLinker(config=config)
    return linker.link_all(notes, dry_run=dry_run)


__all__ = [
    "Embedding",
    "Link",
    "LinkResult",
    "NoteItem",
    "EmbeddingProvider",
    "SentenceTransformerProvider",
    "MockEmbeddingProvider",
    "Vectorizer",
    "EmbeddingStore",
    "PARACentroidCalculator",
    "AutoLinker",
    "embed",
    "link_item",
    "link_all",
    "scan_wiki_vault",
    "main",
]
