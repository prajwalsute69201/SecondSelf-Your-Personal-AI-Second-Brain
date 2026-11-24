"""SecondSelf — Semantic Auto-Linking CLI & Vault Traversal.

Scans the curated wiki notes vault, builds dense embeddings, computes PARA
centroids, injects reciprocal Wikilinks, and exports the public vector index.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from config import get_settings
from link.centroids import PARACentroidCalculator
from link.linker import AutoLinker
from link.models import LinkResult, NoteItem
from link.store import EmbeddingStore
from link.vectorizer import MockEmbeddingProvider, SentenceTransformerProvider, Vectorizer

logger = logging.getLogger("secondself.link")

EXIT_SUCCESS = 0
EXIT_GENERAL_ERROR = 1
EXIT_USAGE_ERROR = 2
EXIT_IO_ERROR = 3


def scan_wiki_vault(wiki_dir: Path) -> list[NoteItem]:
    """Traverses wiki/Projects, wiki/Areas, wiki/Resources, wiki/Archives for markdown notes.

    Args:
        wiki_dir: Root directory of wiki vault.

    Returns:
        List of parsed NoteItem instances sorted deterministically by ID.
    """
    notes: list[NoteItem] = []
    if not wiki_dir.exists():
        return notes

    for cat_dir in sorted(wiki_dir.glob("*")):
        if cat_dir.is_dir():
            for md_file in sorted(cat_dir.glob("*.md")):
                try:
                    note = NoteItem.from_markdown_file(md_file)
                    notes.append(note)
                except Exception as exc:
                    logger.warning("Could not parse markdown note '%s': %s", md_file, exc)

    notes.sort(key=lambda n: n.id)
    return notes


def build_arg_parser() -> argparse.ArgumentParser:
    """Builds the CLI argument parser for link.py."""
    parser = argparse.ArgumentParser(
        prog="secondself-link",
        description="Autonomous Semantic Vectorization & Auto-Linking Engine.",
    )
    parser.add_argument(
        "-b", "--batch",
        dest="batch",
        action="store_true",
        help="Process and auto-link the entire wiki vault (default)",
    )
    parser.add_argument(
        "-i", "--item-id",
        dest="item_id",
        type=str,
        default=None,
        help="Specific note ID to evaluate and link",
    )
    parser.add_argument(
        "--threshold",
        dest="similarity_threshold",
        type=float,
        default=None,
        help="Minimum cosine similarity threshold (0.0 to 1.0)",
    )
    parser.add_argument(
        "--top-k",
        dest="top_k_links",
        type=int,
        default=None,
        help="Maximum reciprocal links per note (default: 5)",
    )
    parser.add_argument(
        "--provider",
        dest="provider",
        choices=["sentence-transformers", "mock"],
        default=None,
        help="Embedding provider override (sentence-transformers or mock)",
    )
    parser.add_argument(
        "--wiki-dir",
        dest="wiki_dir",
        type=Path,
        default=None,
        help="Root directory of markdown wiki vault",
    )
    parser.add_argument(
        "--index-file",
        dest="index_file",
        type=Path,
        default=None,
        help="Precomputed index JSON destination override",
    )
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Calculate embeddings and links without mutating notes on disk",
    )
    parser.add_argument(
        "--format",
        dest="format",
        choices=["text", "json"],
        default="text",
        help="Output serialization format (default: text)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI execution entrypoint."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    settings = get_settings()
    cfg_kwargs: dict[str, Any] = {}
    if args.similarity_threshold is not None:
        cfg_kwargs["similarity_threshold"] = args.similarity_threshold
    if args.top_k_links is not None:
        cfg_kwargs["top_k_links"] = args.top_k_links

    vec_config = (
        settings.vector.model_copy(update=cfg_kwargs)
        if cfg_kwargs
        else settings.vector
    )

    # Provider selection
    provider = None
    if args.provider == "mock":
        provider = MockEmbeddingProvider(dimensions=vec_config.dimensions)
    elif args.provider == "sentence-transformers":
        provider = SentenceTransformerProvider(model_name=vec_config.model_name)

    vectorizer = Vectorizer(config=vec_config, provider=provider)
    index_file = args.index_file or settings.paths.index_file
    store = EmbeddingStore(index_file=index_file, config=vec_config)
    linker = AutoLinker(config=vec_config, vectorizer=vectorizer, store=store)

    wiki_dir = args.wiki_dir or settings.paths.wiki_dir
    notes = scan_wiki_vault(wiki_dir)

    if not notes:
        if args.format == "json":
            sys.stdout.write(json.dumps({"status": "empty", "message": "Wiki vault contains zero notes.", "notes": []}, indent=2) + "\n")
        else:
            sys.stdout.write("Wiki vault is empty. Ingest and classify notes first using capture.py & classify.py.\n")
        return EXIT_SUCCESS

    # Single item link mode
    if args.item_id:
        target_note = next((n for n in notes if n.id == args.item_id), None)
        if not target_note:
            sys.stderr.write(f"Error: Note ID '{args.item_id}' not found in wiki vault.\n")
            return EXIT_IO_ERROR

        # Ensure index populated
        linker.link_all(notes=notes, dry_run=True)
        emb = vectorizer.embed(target_note.composite_text, record_id=target_note.id)
        result = linker.link_item(item=target_note, embedding=emb)

        if args.format == "json":
            sys.stdout.write(json.dumps(result.to_dict(), indent=2) + "\n")
        else:
            sys.stdout.write(f"Note:       {target_note.title} ({target_note.id})\n")
            sys.stdout.write(f"Category:   {target_note.category}\n")
            sys.stdout.write(f"Centroid:   {result.nearest_centroid} (score: {result.centroid_score:.2f}, aligned: {result.category_alignment})\n")
            sys.stdout.write(f"Links ({len(result.links)}):\n")
            for link in result.links:
                sys.stdout.write(f"  - [[{link.target_id}|{link.target_title}]] — {link.score:.1%} ({link.reason})\n")
        return EXIT_SUCCESS

    # Vault batch mode
    results = linker.link_all(notes=notes, dry_run=args.dry_run)

    if args.format == "json":
        payload = {
            "model": vec_config.model_name,
            "threshold": vec_config.similarity_threshold,
            "top_k": vec_config.top_k_links,
            "dry_run": args.dry_run,
            "total_notes": len(notes),
            "results": [r.to_dict() for r in results],
            "centroids": linker.centroid_calc.centroids if linker.centroid_calc else {},
        }
        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
    else:
        sys.stdout.write(
            f"Autonomous Semantic Auto-Linking Complete:\n"
            f"Processed: {len(results)} notes (dry_run={args.dry_run}, threshold={vec_config.similarity_threshold:.2f}, top_k={vec_config.top_k_links})\n"
        )
        total_links = sum(len(r.links) for r in results)
        sys.stdout.write(f"Total Discovered Links: {total_links}\n")
        for r in results:
            note_title = r.metadata.get("title", r.item_id)
            c_info = f"[Centroid: {r.nearest_centroid} {r.centroid_score:.0%}]" if r.nearest_centroid else ""
            sys.stdout.write(f" - {note_title} ({r.item_id[:8]}) -> {len(r.links)} links {c_info}\n")

    return EXIT_SUCCESS
