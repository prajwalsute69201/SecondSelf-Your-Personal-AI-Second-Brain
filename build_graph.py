"""Knowledge graph builder and serializer for SecondSelf (Task 3.1.1).

Parses classified and linked wiki notes from `wiki/`, enriches them into
`GraphNode` records, extracts and deduplicates relationships from frontmatter
`links[]` and body `[[target-id]]` wikilinks into `GraphEdge` records, validates
graph integrity, and serializes the result to `data/graph.json` with metadata.

Usage:
    python build_graph.py           # Build and save data/graph.json
    python build_graph.py --dry-run # Preview graph stats without writing files
    python build_graph.py --help    # Show CLI options
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Sequence

# Ensure repository root is on sys.path so 'lib' is importable
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import lib.embeddings as emb
from lib.models import GraphEdge, GraphNode, WikiNote
import lib.storage as storage
import link

logger = logging.getLogger(__name__)

VALID_PARA_CATEGORIES: set[str] = {"Projects", "Areas", "Resources", "Archives"}

__all__ = [
    "VALID_PARA_CATEGORIES",
    "get_graph_path",
    "extract_node_preview",
    "extract_wikilinks",
    "build_graph_nodes",
    "build_graph_edges",
    "validate_graph",
    "build_graph_data",
    "save_graph",
    "build_and_save_graph",
    "build_parser",
    "main",
]


def get_graph_path(
    root: Path | str | None = None,
    user_id: str | None = None,
) -> Path:
    """Return the filesystem path to data/graph.json, optionally scoped to user_id.

    Args:
        root: Optional project root path override.
        user_id: Optional user ID to scope the graph file.

    Returns:
        Path: Absolute or resolved path to data/graph.json or data/{user_id}/graph.json.
    """
    base_data = storage.get_data_dir(root)
    if user_id and user_id != "00000000-0000-0000-0000-000000000001":
        scoped_dir = base_data / user_id
        scoped_dir.mkdir(parents=True, exist_ok=True)
        return scoped_dir / "graph.json"
    return base_data / "graph.json"


def extract_node_preview(body: str, max_chars: int = 200) -> str:
    """Extract a concise content preview (first ~200 characters) from note body.

    Strips trailing auto-generated wikilinks and normalizes internal whitespace
    so that preview represents the actual prose content.

    Args:
        body: Markdown body string.
        max_chars: Maximum character limit for the preview (default 200).

    Returns:
        str: Cleaned preview string of up to max_chars characters.
    """
    if not body:
        return ""

    # Strip auto-generated trailing [[...]] wikilinks first
    cleaned = link.strip_appended_wikilinks(body).strip()
    if not cleaned:
        cleaned = body.strip()

    # Normalize excessive newlines and whitespace
    normalized = " ".join(cleaned.split())
    if len(normalized) <= max_chars:
        return normalized

    return normalized[:max_chars].rstrip()


def extract_wikilinks(body: str) -> list[str]:
    """Extract all [[target-id]] wikilink targets from a note body.

    Args:
        body: Markdown body text string.

    Returns:
        list of extracted target note identifier strings.
    """
    if not body:
        return []
    return re.findall(r"\[\[([a-zA-Z0-9_\-]+)\]\]", body)


def build_graph_nodes(notes: Sequence[WikiNote]) -> tuple[list[GraphNode], list[str]]:
    """Convert WikiNote instances into enriched GraphNode records.

    Mapping per Implementationplan.md Task 3.1.1:
    - id: Unique note ID.
    - label: Human-readable note summary (or ID fallback).
    - para: PARA category ('Projects', 'Areas', 'Resources', 'Archives').
    - tags: Semantic topic tags list.
    - summary: Note summary string.
    - content_preview: First ~200 characters of note body.
    - group: Grouping identifier matching the PARA category for visual clustering.

    Args:
        notes: Sequence of parsed WikiNote instances.

    Returns:
        tuple: (list of validated GraphNode instances sorted by ID, list of validation warnings).
    """
    nodes: list[GraphNode] = []
    warnings: list[str] = []
    seen_ids: set[str] = set()

    for note in notes:
        nid = str(note.id).strip()
        if not nid:
            warnings.append(f"Encountered note with empty ID; skipping node.")
            continue

        if nid in seen_ids:
            warnings.append(f"Duplicate note ID detected: '{nid}'; skipping duplicate.")
            continue
        seen_ids.add(nid)

        para = str(note.para).strip()
        if para not in VALID_PARA_CATEGORIES:
            warnings.append(
                f"Note '{nid}' has non-standard PARA category '{para}'; grouping under '{para}'."
            )

        summary = str(note.summary or "").strip()
        label = summary if summary else nid
        tags = [str(t).strip() for t in (note.tags or []) if str(t).strip()]
        preview = extract_node_preview(note.body or "", 200)

        node = GraphNode(
            id=nid,
            label=label,
            para=para,
            tags=tags,
            summary=summary,
            content_preview=preview,
            group=para,
        )
        nodes.append(node)

    nodes.sort(key=lambda n: n.id)
    return nodes, warnings


def build_graph_edges(
    notes: Sequence[WikiNote],
    valid_node_ids: set[str],
    embeddings: dict[str, Any] | None = None,
) -> tuple[list[GraphEdge], list[str]]:
    """Extract and deduplicate graph edges from note links and body wikilinks.

    Deduplication per Implementationplan.md Task 3.1.1:
    Edge key = (min(source, target), max(source, target)).
    Preserves relationship weights using embedding cosine similarity when available,
    defaulting to 1.0 for explicit links.

    Args:
        notes: Sequence of WikiNote instances.
        valid_node_ids: Set of known, existing node IDs.
        embeddings: Optional dictionary mapping note IDs to embedding vectors.

    Returns:
        tuple: (list of GraphEdge instances sorted by (source, target), list of warnings).
    """
    edges: list[GraphEdge] = []
    warnings: list[str] = []
    seen_edge_keys: set[tuple[str, str]] = set()

    for note in notes:
        source_id = str(note.id).strip()
        if source_id not in valid_node_ids:
            continue

        # Gather target candidates from frontmatter links[] and inline [[id]] wikilinks
        body_targets = extract_wikilinks(note.body or "")
        fm_targets = [str(l).strip() for l in (note.links or []) if str(l).strip()]
        all_targets = list(dict.fromkeys(fm_targets + body_targets))

        for target_id in all_targets:
            # 1. Prevent self-links (Edge Case 3.6)
            if target_id == source_id:
                warnings.append(f"Ignored self-referencing link on note '{source_id}'.")
                continue

            # 2. Skip edges pointing to non-existent nodes (Edge Case 3.8)
            if target_id not in valid_node_ids:
                warnings.append(
                    f"Skipping edge from '{source_id}' to non-existent node '{target_id}'."
                )
                continue

            # 3. Deduplicate undirected edge pair (min, max)
            edge_key = (min(source_id, target_id), max(source_id, target_id))
            if edge_key in seen_edge_keys:
                continue
            seen_edge_keys.add(edge_key)

            src, tgt = edge_key

            # 4. Determine weight and type
            weight = 1.0
            edge_type = "explicit"

            if embeddings and src in embeddings and tgt in embeddings:
                try:
                    sim = emb.cosine_similarity(embeddings[src], embeddings[tgt])
                    weight = round(float(sim), 4)
                    edge_type = "semantic"
                except Exception as e:
                    logger.warning(
                        f"Failed to calculate similarity weight between {src} and {tgt}: {e}"
                    )
                    weight = 1.0
                    edge_type = "explicit"

            edge = GraphEdge(
                source=src,
                target=tgt,
                weight=weight,
                type=edge_type,
            )
            edges.append(edge)

    edges.sort(key=lambda e: (e.source, e.target))
    return edges, warnings


def validate_graph(nodes: Sequence[GraphNode], edges: Sequence[GraphEdge]) -> list[str]:
    """Validate graph invariants before serialization.

    Checks:
    - Node ID uniqueness and required fields.
    - PARA category validity.
    - Every edge references existing source and target nodes.
    - No self-referencing edges.
    - No duplicate edges.
    - Numerical validity of edge weights.

    Args:
        nodes: Sequence of GraphNode instances.
        edges: Sequence of GraphEdge instances.

    Returns:
        list of validation error strings (empty list indicates valid graph).
    """
    errors: list[str] = []
    node_ids: set[str] = set()

    for i, node in enumerate(nodes):
        if not node.id:
            errors.append(f"Node at index {i} has empty ID.")
        elif node.id in node_ids:
            errors.append(f"Duplicate node ID '{node.id}' found in nodes list.")
        else:
            node_ids.add(node.id)

        if node.para not in VALID_PARA_CATEGORIES:
            errors.append(
                f"Node '{node.id}' has invalid PARA category '{node.para}'. "
                f"Must be one of {sorted(VALID_PARA_CATEGORIES)}."
            )

    seen_edges: set[tuple[str, str]] = set()
    for j, edge in enumerate(edges):
        if edge.source == edge.target:
            errors.append(
                f"Edge at index {j} is self-referencing ({edge.source} -> {edge.target})."
            )

        if edge.source not in node_ids:
            errors.append(
                f"Edge at index {j} references non-existent source node '{edge.source}'."
            )

        if edge.target not in node_ids:
            errors.append(
                f"Edge at index {j} references non-existent target node '{edge.target}'."
            )

        pair_key = (edge.source, edge.target)
        if pair_key in seen_edges:
            errors.append(f"Duplicate edge detected: {pair_key}.")
        seen_edges.add(pair_key)

        if not isinstance(edge.weight, (int, float)) or not (
            -1.0 <= edge.weight <= 1.0 or edge.weight == 1.0
        ):
            errors.append(
                f"Edge {pair_key} has invalid numerical weight {edge.weight!r}."
            )

    return errors


def build_graph_data(
    notes: Sequence[WikiNote],
    embeddings: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Construct full dictionary {nodes, edges, metadata} from notes and embeddings.

    Schema per Implementationplan.md Task 3.1.2:
    {
      "nodes": [{ "id", "label", "para", "tags", "summary", "content_preview", "group" }],
      "edges": [{ "source", "target", "weight", "type" }],
      "metadata": { "generated_at", "node_count", "edge_count" }
    }

    Args:
        notes: Sequence of WikiNote instances.
        embeddings: Optional embeddings dictionary for weight annotation.

    Returns:
        tuple: (graph_data_dict, validation_and_warning_messages).
    """
    nodes, node_warnings = build_graph_nodes(notes)
    valid_ids = {n.id for n in nodes}

    edges, edge_warnings = build_graph_edges(
        notes=notes,
        valid_node_ids=valid_ids,
        embeddings=embeddings,
    )

    all_warnings = node_warnings + edge_warnings

    # Validate graph invariants
    validation_errors = validate_graph(nodes, edges)
    if validation_errors:
        all_warnings.extend(validation_errors)

    now_iso = datetime.now(timezone.utc).isoformat()

    def _clean_node(n: GraphNode) -> dict[str, Any]:
        d = asdict(n)
        d.pop("user_id", None)
        return d

    def _clean_edge(e: GraphEdge) -> dict[str, Any]:
        d = asdict(e)
        d.pop("user_id", None)
        return d

    graph_data: dict[str, Any] = {
        "nodes": [_clean_node(n) for n in nodes],
        "edges": [_clean_edge(e) for e in edges],
        "metadata": {
            "generated_at": now_iso,
            "node_count": len(nodes),
            "edge_count": len(edges),
        },
    }

    return graph_data, all_warnings


def save_graph(
    graph_data: dict[str, Any],
    root: Path | str | None = None,
    user_id: str | None = None,
) -> Path:
    """Atomically write graph JSON to data/graph.json (or user scoped) and update index.json.

    Args:
        graph_data: Dictionary structure matching graph.json schema.
        root: Optional project root path override.
        user_id: Optional user ID to scope the graph persistence.

    Returns:
        Path: Path to the written graph.json file.

    Raises:
        OSError: If filesystem operations fail.
    """
    target_path = get_graph_path(root, user_id=user_id)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    temp_file: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target_path.parent,
            delete=False,
            suffix=".tmp",
        ) as f:
            temp_file = Path(f.name)
            json.dump(graph_data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())

        temp_file.replace(target_path)
    except Exception as e:
        if temp_file and temp_file.exists():
            temp_file.unlink(missing_ok=True)
        raise OSError(f"Failed to write graph data to {target_path}: {e}") from e

    # Update data/index.json last_graph_build timestamp
    try:
        index_data = storage.load_index(root=root, user_id=user_id)
        index_data["last_graph_build"] = graph_data["metadata"]["generated_at"]
        storage.save_index(index_data, root=root, user_id=user_id)
    except Exception as e:
        logger.warning(f"Failed to update last_graph_build in index.json: {e}")

    # Persist graph nodes, edges, and metadata in SQLite database for durable restoration
    try:
        from lib import database as db
        db.save_graph_records(
            nodes=graph_data.get("nodes", []),
            edges=graph_data.get("edges", []),
            metadata=graph_data.get("metadata", {}),
            user_id=user_id,
            root=root,
        )
    except Exception as e:
        logger.warning(f"Failed to persist graph records to SQLite: {e}")

    return target_path


def build_and_save_graph(
    root: Path | str | None = None,
    dry_run: bool = False,
    user_id: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Discover wiki notes, build knowledge graph, validate, and persist.

    Args:
        root: Optional project root path override.
        dry_run: If True, builds graph without writing files or updating index.
        user_id: Optional user ID to isolate the knowledge graph.

    Returns:
        tuple: (graph_data_dict, messages_and_warnings_list).
    """
    # 1. Load notes safely scoped to user
    notes, read_errors = link.safe_read_wiki_notes(root=root, user_id=user_id)

    # 2. Load embeddings if available (optional, for edge weights)
    embeddings = {}
    try:
        embeddings = emb.load_embeddings(root=root)
    except Exception as e:
        logger.warning(f"Could not load embeddings for graph edge weighting: {e}")

    # 3. Build graph structure
    graph_data, warnings = build_graph_data(notes, embeddings=embeddings)
    all_messages = read_errors + warnings

    # 4. Save if not dry-run
    if not dry_run:
        save_graph(graph_data, root=root, user_id=user_id)

    return graph_data, all_messages


def build_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser for build_graph.py."""
    parser = argparse.ArgumentParser(
        prog="build_graph.py",
        description="Convert SecondSelf classified wiki notes into data/graph.json.",
    )
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Preview graph metrics without writing data/graph.json or modifying index.json.",
    )
    parser.add_argument(
        "--root",
        dest="root",
        metavar="PATH",
        help=argparse.SUPPRESS,  # For internal test use
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for build_graph.py.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        int: Exit status code (0 for success, 1 for fatal error).
    """
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 0

    root = Path(args.root).resolve() if args.root else None
    dry_run = args.dry_run

    prefix = "[dry-run] " if dry_run else ""
    print(f"{prefix}Building SecondSelf knowledge graph from wiki notes...")

    try:
        graph_data, messages = build_and_save_graph(root=root, dry_run=dry_run)
    except Exception as e:
        print(f"Error building knowledge graph: {e}", file=sys.stderr)
        return 1

    node_count = graph_data["metadata"]["node_count"]
    edge_count = graph_data["metadata"]["edge_count"]

    if dry_run:
        print(f"Dry-run preview complete: {node_count} nodes, {edge_count} edges.")
    else:
        out_path = get_graph_path(root)
        print(f"Graph written to {out_path} ({node_count} nodes, {edge_count} edges).")

    if messages:
        print(f"\nGraph Diagnostics & Warnings ({len(messages)}):")
        for msg in messages:
            print(f"  - {msg}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

