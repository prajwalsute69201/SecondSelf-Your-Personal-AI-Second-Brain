"""SecondSelf — Deterministic Knowledge Graph Projector.

Transforms verified markdown notes, PARA classifications, semantic links,
and raw capture metadata into a canonical, schema-validated knowledge graph.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

from classify.taxonomy import PARACategory, get_criteria
from graph.builder import GraphBuilder
from graph.models import Edge, Graph, Node
from graph.schema import (
    EdgeType,
    NodeType,
    make_item_id,
    make_para_id,
    make_source_id,
    make_tag_id,
    slugify,
)
from link.models import NoteItem

logger = logging.getLogger("secondself.graph.projector")

# Regular expression matching Wikilinks in '## Related Notes'
# Example: - [[89275e83-e2d8-455d-9fa9-68a4251e5654|Metric Selection]] — Match: 69%
RE_RELATED_WIKILINK = re.compile(
    r"\[\[(?P<target_id>[a-zA-Z0-9_\-]+)(?:\|(?P<title>[^\]]+))?\]\]\s*(?:[—\-–]\s*Match:\s*(?P<pct>\d+)%)?",
    flags=re.IGNORECASE,
)


class GraphProjector:
    """Projects ingested and classified notes into an in-memory knowledge graph."""

    def __init__(
        self,
        include_categories: bool = True,
        include_tags: bool = True,
        include_sources: bool = True,
        similarity_threshold: float = 0.65,
        top_k_links: int = 5,
    ) -> None:
        self.include_categories = include_categories
        self.include_tags = include_tags
        self.include_sources = include_sources
        self.similarity_threshold = similarity_threshold
        self.top_k_links = top_k_links

    @staticmethod
    def _parse_related_links_from_markdown(markdown_content: str) -> list[tuple[str, str, float]]:
        """Extracts (target_id, target_title, score) from the '## Related Notes' section."""
        matches: list[tuple[str, str, float]] = []
        rel_match = re.search(r"##\s+Related Notes\s*\n(.*?)(?=\n##|\Z)", markdown_content, flags=re.DOTALL)
        if not rel_match:
            return matches

        related_body = rel_match.group(1)
        for line in related_body.splitlines():
            m = RE_RELATED_WIKILINK.search(line)
            if m:
                target_id = m.group("target_id").strip()
                title = (m.group("title") or target_id).strip()
                pct_str = m.group("pct")
                score = (float(pct_str) / 100.0) if pct_str else 0.70
                matches.append((target_id, title, score))

        return matches

    def _emit_para_categories(self, builder: GraphBuilder) -> None:
        """Registers the 4 fixed canonical PARA quadrant nodes."""
        category_metadata = [
            (
                PARACategory.PROJECT,
                "Projects",
                "Active, time-bound efforts with a definitive goal and deadline.",
                "#F59E0B",
            ),
            (
                PARACategory.AREA,
                "Areas",
                "Ongoing spheres of activity and standards of performance to maintain indefinitely.",
                "#3B82F6",
            ),
            (
                PARACategory.RESOURCE,
                "Resources",
                "Reference materials, topic guides, and conceptual resources for future lookup.",
                "#10B981",
            ),
            (
                PARACategory.ARCHIVE,
                "Archives",
                "Completed projects, inactive areas, and superseded historical records.",
                "#6B7280",
            ),
        ]

        for cat_enum, display_name, desc, color_hex in category_metadata:
            crit = get_criteria(cat_enum)
            node_id = make_para_id(cat_enum.value)
            node = Node(
                id=node_id,
                node_type=NodeType.PARA_CATEGORY,
                label=display_name,
                properties={
                    "category": cat_enum.value,
                    "display_name": display_name,
                    "target_folder": crit.target_folder,
                    "description": desc,
                    "color_hex": color_hex,
                },
                visual_value=32.0,
            )
            builder.add_node(node, overwrite=True)

    def project_note(
        self,
        note: NoteItem,
        builder: GraphBuilder,
        raw_meta: dict[str, Any] | None = None,
    ) -> Node:
        """Projects a single NoteItem into an Item node plus Category, Tag, and Source edges."""
        item_node_id = make_item_id(note.id)
        raw_info = raw_meta or {}

        # 1. Register Item Node
        char_count = raw_info.get("char_count") or len(note.content)
        word_count = raw_info.get("word_count") or len(note.content.split())
        source_type = note.metadata.get("source_type") or raw_info.get("source_type", "text")
        confidence = float(note.metadata.get("confidence", 1.0))
        needs_review = bool(note.metadata.get("needs_review", False))

        item_node = Node(
            id=item_node_id,
            node_type=NodeType.ITEM,
            label=note.title,
            properties={
                "record_id": note.id,
                "title": note.title,
                "category": note.category,
                "confidence": round(confidence, 4),
                "summary": note.summary or "",
                "tags": note.tags,
                "source_type": source_type,
                "created_at": note.metadata.get("classified_at", ""),
                "content_hash": note.metadata.get("content_hash", ""),
                "char_count": char_count,
                "word_count": word_count,
                "needs_review": needs_review,
                "file_path": str(note.file_path) if note.file_path else "",
            },
        )
        builder.add_node(item_node, overwrite=True)

        # 2. Emit BELONGS_TO Edge
        if self.include_categories and note.category:
            cat_target_id = make_para_id(note.category)
            belongs_edge = Edge(
                source_id=item_node_id,
                target_id=cat_target_id,
                edge_type=EdgeType.BELONGS_TO,
                properties={
                    "category": note.category,
                    "confidence": round(confidence, 4),
                    "method": note.metadata.get("method", "hybrid"),
                    "rationale": note.metadata.get("rationale", ""),
                    "assigned_at": note.metadata.get("classified_at", ""),
                },
            )
            builder.add_edge(belongs_edge)

        # 3. Emit Tag Nodes and TAGGED_WITH Edges
        if self.include_tags and note.tags:
            for tag in note.tags:
                clean_tag = tag.strip().lstrip("#")
                if not clean_tag:
                    continue
                tag_node_id = make_tag_id(clean_tag)
                tag_node = Node(
                    id=tag_node_id,
                    node_type=NodeType.TAG,
                    label=f"#{clean_tag}",
                    properties={
                        "name": clean_tag,
                        "slug": slugify(clean_tag),
                    },
                    visual_value=14.0,
                )
                builder.add_node(tag_node, overwrite=False)

                tagged_edge = Edge(
                    source_id=item_node_id,
                    target_id=tag_node_id,
                    edge_type=EdgeType.TAGGED_WITH,
                    properties={
                        "tag_slug": slugify(clean_tag),
                        "tag_name": clean_tag,
                    },
                )
                builder.add_edge(tagged_edge)

        # 4. Emit Source Node and EXTRACTED_FROM Edge
        if self.include_sources and source_type:
            src_node_id = make_source_id(source_type)
            src_node = Node(
                id=src_node_id,
                node_type=NodeType.SOURCE,
                label=source_type.upper(),
                properties={
                    "source_type": source_type,
                    "display_name": source_type.title(),
                    "source_uri": raw_info.get("source_uri") or "",
                },
                visual_value=16.0,
            )
            builder.add_node(src_node, overwrite=False)

            extracted_edge = Edge(
                source_id=item_node_id,
                target_id=src_node_id,
                edge_type=EdgeType.EXTRACTED_FROM,
                properties={
                    "source_type": source_type,
                    "source_uri": raw_info.get("source_uri") or "",
                    "char_count": char_count,
                    "captured_at": raw_info.get("timestamp") or "",
                },
            )
            builder.add_edge(extracted_edge)

        return item_node

    def project_corpus(
        self,
        notes: list[NoteItem],
        raw_metadata_map: dict[str, dict[str, Any]] | None = None,
        builder: GraphBuilder | None = None,
    ) -> Graph:
        """Projects a collection of notes into a canonical Graph.

        Executes in two passes:
        1. Emits all Category, Item, Tag, and Source nodes with structural edges.
        2. Discovers and emits SEMANTIC_LINK edges strictly between active Item nodes.
        """
        active_builder = builder or GraphBuilder()
        raw_map = raw_metadata_map or {}

        # 1. Compute corpus hash for provenance and reproducibility
        hasher = hashlib.sha256()
        for note in sorted(notes, key=lambda n: n.id):
            hasher.update(f"{note.id}:{note.title}:{note.category}".encode("utf-8"))
        corpus_hash = hasher.hexdigest()

        # 2. Emit fixed category nodes
        if self.include_categories:
            self._emit_para_categories(active_builder)

        # 3. Pass 1: Emit all Item nodes and structural edges
        for note in notes:
            raw_meta = raw_map.get(note.id)
            self.project_note(note, active_builder, raw_meta)

        # 4. Pass 2: Emit SEMANTIC_LINK edges between existing Item nodes
        for note in notes:
            source_node_id = make_item_id(note.id)
            related_candidates = self._parse_related_links_from_markdown(note.content)

            # Sort candidates by score descending and apply top_k_links cap
            sorted_candidates = sorted(related_candidates, key=lambda c: c[2], reverse=True)[: self.top_k_links]

            for target_id, target_title, score in sorted_candidates:
                target_node_id = make_item_id(target_id)

                # Skip if target node does not exist in graph (prevents dangling edges)
                if not active_builder.contains_node(target_node_id):
                    continue

                # Skip if below similarity threshold
                if score < self.similarity_threshold:
                    continue

                pct = int(round(score * 100))
                semantic_edge = Edge(
                    source_id=source_node_id,
                    target_id=target_node_id,
                    edge_type=EdgeType.SEMANTIC_LINK,
                    properties={
                        "score": round(score, 4),
                        "similarity_pct": pct,
                        "method": "item_to_item",
                        "target_title": target_title,
                        "reason": f"Semantic similarity {pct}% with '{target_title}'",
                    },
                )
                try:
                    active_builder.add_edge(semantic_edge)
                except Exception as exc:
                    logger.debug("Skipped edge (%s -> %s): %s", source_node_id, target_node_id, exc)

        return active_builder.build(corpus_hash=corpus_hash)

    def project_incremental(
        self,
        existing_graph: Graph,
        new_or_updated_notes: list[NoteItem],
        raw_metadata_map: dict[str, dict[str, Any]] | None = None,
    ) -> Graph:
        """Incrementally adds or updates notes on an existing Graph.

        Respects immutability: Re-seeds a new GraphBuilder from existing graph nodes
        and edges, then applies the projection for new notes.
        """
        builder = GraphBuilder(
            schema_version=existing_graph.metadata.graph_schema_version,
            projection_version=existing_graph.metadata.projection_version,
        )

        # Re-populate existing nodes and edges
        for node in existing_graph.nodes.values():
            # Clone node
            builder.add_node(Node(**node.model_dump()), overwrite=True)

        for edge in existing_graph.edges:
            builder.add_edge(Edge(**edge.model_dump()))

        # Pass 1: Project new or updated notes
        raw_map = raw_metadata_map or {}
        for note in new_or_updated_notes:
            raw_meta = raw_map.get(note.id)
            self.project_note(note, builder, raw_meta)

        # Pass 2: Project semantic links for updated notes
        for note in new_or_updated_notes:
            source_node_id = make_item_id(note.id)
            related_candidates = self._parse_related_links_from_markdown(note.content)
            sorted_candidates = sorted(related_candidates, key=lambda c: c[2], reverse=True)[: self.top_k_links]

            for target_id, target_title, score in sorted_candidates:
                target_node_id = make_item_id(target_id)
                if not builder.contains_node(target_node_id) or score < self.similarity_threshold:
                    continue

                pct = int(round(score * 100))
                edge = Edge(
                    source_id=source_node_id,
                    target_id=target_node_id,
                    edge_type=EdgeType.SEMANTIC_LINK,
                    properties={
                        "score": round(score, 4),
                        "similarity_pct": pct,
                        "method": "item_to_item",
                        "target_title": target_title,
                        "reason": f"Semantic similarity {pct}% with '{target_title}'",
                    },
                )
                try:
                    builder.add_edge(edge)
                except Exception as exc:
                    logger.debug("Incremental edge skipped: %s", exc)

        return builder.build()
