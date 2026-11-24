"""SecondSelf — CLI Entrypoint for Graph Projection (Task 3.1).

Scans curated markdown vault (wiki/), discovers semantic relationships,
projects entities into a canonical Graph, and serializes to JSON, Cypher, or GraphML.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from config import get_settings
from graph.projector import GraphProjector
from graph.serializers import (
    CypherSerializer,
    GraphMLSerializer,
    NodeLinkJSONSerializer,
)
from link.models import NoteItem

logger = logging.getLogger("secondself.graph.cli")


def build_arg_parser() -> argparse.ArgumentParser:
    """Builds the CLI argument parser for graph projection."""
    settings = get_settings()
    parser = argparse.ArgumentParser(
        prog="secondself-build-graph",
        description="Project SecondSelf notes, PARA categories, tags, and links into a canonical graph.",
    )
    parser.add_argument(
        "--wiki-dir",
        type=Path,
        default=settings.paths.wiki_dir,
        help="Path to wiki vault directory containing PARA folders (default: wiki/)",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=settings.paths.raw_dir,
        help="Path to raw captures directory for provenance metadata (default: raw/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.paths.graph_file,
        help="Primary output file path (default: public/graph.json)",
    )
    parser.add_argument(
        "--format",
        choices=["json", "cypher", "graphml", "all"],
        default="json",
        help="Output serialization format (default: json)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=settings.vector.similarity_threshold,
        help="Minimum similarity score for semantic link edges (default: 0.65)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=settings.vector.top_k_links,
        help="Maximum semantic link edges emitted per note (default: 5)",
    )
    parser.add_argument(
        "--no-tags",
        action="store_true",
        help="Exclude Tag nodes and TAGGED_WITH edges from projection",
    )
    parser.add_argument(
        "--no-sources",
        action="store_true",
        help="Exclude Source nodes and EXTRACTED_FROM edges from projection",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print detailed graph topology statistics to stdout",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute in-memory projection without writing output files to disk",
    )
    return parser


def load_raw_metadata_map(raw_dir: Path) -> dict[str, dict[str, Any]]:
    """Loads provenance metadata from raw capture JSON files."""
    meta_map: dict[str, dict[str, Any]] = {}
    if not raw_dir.exists():
        return meta_map

    for json_file in raw_dir.glob("*.json"):
        if json_file.name in ("manifest.json", "index.json", "graph.json"):
            continue
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            rec_id = data.get("record_id") or data.get("id") or json_file.stem
            meta = data.get("metadata", {})
            meta_map[rec_id] = {
                "source_type": meta.get("source_type") or data.get("source_type", "text"),
                "source_uri": meta.get("source_uri") or data.get("source_uri", ""),
                "char_count": meta.get("char_count") or len(data.get("raw_content", "")),
                "timestamp": data.get("timestamp", ""),
            }
        except Exception as exc:
            logger.debug("Skipping unreadable raw file '%s': %s", json_file.name, exc)

    return meta_map


def scan_wiki_notes(wiki_dir: Path) -> list[NoteItem]:
    """Finds and parses all Markdown notes across the PARA vault folders."""
    notes: list[NoteItem] = []
    if not wiki_dir.exists():
        return notes

    for md_file in sorted(wiki_dir.glob("*/*.md")):
        if md_file.name.startswith("."):
            continue
        try:
            note = NoteItem.from_markdown_file(md_file)
            notes.append(note)
        except Exception as exc:
            logger.warning("Failed to parse note '%s': %s", md_file, exc)

    return notes


def main(argv: list[str] | None = None) -> int:
    """CLI execution entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    wiki_dir = args.wiki_dir.resolve()
    raw_dir = args.raw_dir.resolve()
    output_path = args.output.resolve()

    logger.info("Scanning wiki vault at '%s'...", wiki_dir)
    notes = scan_wiki_notes(wiki_dir)
    logger.info("Discovered %d curated Markdown notes.", len(notes))

    if not notes:
        logger.warning("No notes found in '%s'. Graph projection aborted.", wiki_dir)
        return 0

    raw_meta_map = load_raw_metadata_map(raw_dir)

    projector = GraphProjector(
        include_categories=True,
        include_tags=not args.no_tags,
        include_sources=not args.no_sources,
        similarity_threshold=args.threshold,
        top_k_links=args.top_k,
    )

    graph = projector.project_corpus(notes=notes, raw_metadata_map=raw_meta_map)
    stats = graph.statistics

    logger.info(
        "Projected graph: %d nodes, %d edges (Density: %.4f, Avg degree: %.2f)",
        stats.total_nodes,
        stats.total_edges,
        stats.graph_density,
        stats.avg_degree,
    )

    if not args.dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if args.format in ("json", "all"):
            json_ser = NodeLinkJSONSerializer()
            json_payload = json_ser.serialize(graph)
            target_json = output_path if output_path.suffix == ".json" else output_path.with_suffix(".json")
            target_json.write_text(json_payload, encoding="utf-8")
            logger.info("Saved node-link JSON to '%s'", target_json)

        if args.format in ("cypher", "all"):
            cypher_ser = CypherSerializer()
            cypher_payload = cypher_ser.serialize(graph)
            target_cypher = output_path.with_suffix(".cypher")
            target_cypher.write_text(cypher_payload, encoding="utf-8")
            logger.info("Saved Cypher script to '%s'", target_cypher)

        if args.format in ("graphml", "all"):
            graphml_ser = GraphMLSerializer()
            graphml_payload = graphml_ser.serialize(graph)
            target_graphml = output_path.with_suffix(".graphml")
            target_graphml.write_text(graphml_payload, encoding="utf-8")
            logger.info("Saved GraphML XML to '%s'", target_graphml)

    if args.stats or args.dry_run:
        print("\n=== SecondSelf Knowledge Graph Topology Summary ===")
        print(f"Total Nodes: {stats.total_nodes}")
        for n_type, count in sorted(stats.node_counts_by_type.items()):
            print(f"  - {n_type}: {count}")
        print(f"Total Edges: {stats.total_edges}")
        for e_type, count in sorted(stats.edge_counts_by_type.items()):
            print(f"  - {e_type}: {count}")
        print(f"Average Degree: {stats.avg_degree:.2f}")
        print(f"Degree Range: Min = {stats.min_degree}, Max = {stats.max_degree}")
        print(f"Island Nodes: {stats.island_nodes_count} ({stats.island_ratio * 100:.1f}%)")
        print(f"Graph Density: {stats.graph_density:.4f}")
        print(f"Duplicates Prevented: Nodes={stats.duplicates_prevented_nodes}, Edges={stats.duplicates_prevented_edges}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
