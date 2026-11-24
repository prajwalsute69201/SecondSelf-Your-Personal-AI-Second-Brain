#!/usr/bin/env python3
"""SecondSelf — Semantic Vectorization & Auto-Linking CLI & Service.

The Librarian (Milestone 2 - Task 2.2):
Traverses curated markdown notes in wiki/Projects, wiki/Areas, wiki/Resources,
and wiki/Archives; vectorizes note contents using sentence-transformers; computes
per-PARA category centroids; auto-links related notes based on cosine similarity;
injects reciprocal Wikilinks into markdown notes; and exports the precomputed
index to public/index.json.

Usage:
    python link.py --batch
    python link.py --dry-run
    python link.py --threshold 0.70 --top-k 3
    python link.py --item-id <note_uuid>
"""

from __future__ import annotations

import sys

from link import (
    AutoLinker,
    Embedding,
    EmbeddingProvider,
    EmbeddingStore,
    Link,
    LinkResult,
    MockEmbeddingProvider,
    NoteItem,
    PARACentroidCalculator,
    SentenceTransformerProvider,
    Vectorizer,
    embed,
    link_all,
    link_item,
    main,
    scan_wiki_vault,
)
from link.cli import (
    EXIT_GENERAL_ERROR,
    EXIT_IO_ERROR,
    EXIT_SUCCESS,
    EXIT_USAGE_ERROR,
    build_arg_parser,
)

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
    "build_arg_parser",
    "main",
    "EXIT_SUCCESS",
    "EXIT_GENERAL_ERROR",
    "EXIT_USAGE_ERROR",
    "EXIT_IO_ERROR",
]


if __name__ == "__main__":
    sys.exit(main())
