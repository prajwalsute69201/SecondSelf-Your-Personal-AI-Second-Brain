"""SecondSelf — Deterministic GraphBuilder & Validation Engine.

Provides an incremental, deduplicating builder that validates node and edge
schemas, rejects dangling edges and self-links, computes centrality metrics,
and produces canonical, sorted Graph instances.
"""

from __future__ import annotations

import collections
import logging
from typing import Any

from graph.models import (
    DanglingEdgeError,
    Edge,
    Graph,
    GraphMetadata,
    GraphStatistics,
    GraphValidationError,
    Node,
    SelfLinkError,
)
from graph.schema import (
    EDGE_SCHEMAS,
    GRAPH_SCHEMA_VERSION,
    NODE_SCHEMAS,
    PROJECTION_VERSION,
    EdgeType,
    NodeType,
)

logger = logging.getLogger("secondself.graph.builder")


class GraphBuilder:
    """Incremental builder for assembling, validating, and finalizing graphs."""

    def __init__(
        self,
        schema_version: str = GRAPH_SCHEMA_VERSION,
        projection_version: str = PROJECTION_VERSION,
    ) -> None:
        self.schema_version = schema_version
        self.projection_version = projection_version

        self._nodes: dict[str, Node] = collections.OrderedDict()
        self._edges: list[Edge] = []
        self._edge_keys: set[tuple[str, str, str]] = set()

        self._duplicates_prevented_nodes: int = 0
        self._duplicates_prevented_edges: int = 0

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    def contains_node(self, node_id: str) -> bool:
        """Returns True if the node is already registered in the builder."""
        return node_id in self._nodes

    def contains_edge(self, source_id: str, target_id: str, edge_type: EdgeType) -> bool:
        """Returns True if the edge key is already registered in the builder."""
        return (source_id, target_id, edge_type.value) in self._edge_keys

    def add_node(self, node: Node, overwrite: bool = False) -> bool:
        """Adds a node with schema validation and idempotent deduplication.

        Args:
            node: Node instance to register.
            overwrite: If True, updates properties for existing node ID.

        Returns:
            True if newly added or updated; False if skipped as duplicate.

        Raises:
            GraphValidationError: If required properties for node_type are missing.
        """
        # 1. Validate required properties against schema contract
        schema = NODE_SCHEMAS.get(node.node_type)
        if schema:
            missing = [prop for prop in schema.required_properties if prop not in node.properties]
            if missing:
                raise GraphValidationError(
                    f"Node '{node.id}' of type '{node.node_type.value}' is missing required properties: {missing}"
                )

        # 2. Check for existing node
        if node.id in self._nodes:
            self._duplicates_prevented_nodes += 1
            if overwrite:
                existing = self._nodes[node.id]
                existing.properties.update(node.properties)
                if node.label:
                    existing.label = node.label
                return True
            return False

        self._nodes[node.id] = node
        return True

    def add_edge(
        self,
        edge: Edge,
        allow_self_links: bool = False,
        allow_dangling: bool = False,
    ) -> bool:
        """Adds a typed edge with endpoint verification, self-loop prevention, and deduplication.

        Args:
            edge: Edge instance to add.
            allow_self_links: If False, raises SelfLinkError on source_id == target_id.
            allow_dangling: If False, raises DanglingEdgeError on missing endpoints.

        Returns:
            True if added; False if skipped as duplicate.

        Raises:
            SelfLinkError: When source_id equals target_id and allow_self_links is False.
            DanglingEdgeError: When endpoints do not exist in builder and allow_dangling is False.
            GraphValidationError: When required edge properties are missing.
        """
        # 1. Check self-links
        if not allow_self_links and edge.source_id == edge.target_id:
            raise SelfLinkError(
                f"Self-link detected: source '{edge.source_id}' equals target '{edge.target_id}' for edge type '{edge.edge_type.value}'."
            )

        # 2. Check endpoint presence (Dangling edge prevention)
        if not allow_dangling:
            if edge.source_id not in self._nodes:
                raise DanglingEdgeError(
                    f"Dangling edge: source node '{edge.source_id}' does not exist in graph."
                )
            if edge.target_id not in self._nodes:
                raise DanglingEdgeError(
                    f"Dangling edge: target node '{edge.target_id}' does not exist in graph."
                )

        # 3. Validate required properties
        schema = EDGE_SCHEMAS.get(edge.edge_type)
        if schema:
            missing = [prop for prop in schema.required_properties if prop not in edge.properties]
            if missing:
                raise GraphValidationError(
                    f"Edge ({edge.source_id} -> {edge.target_id}) of type '{edge.edge_type.value}' is missing required properties: {missing}"
                )

        # 4. Check for duplicate edge
        if edge.key in self._edge_keys:
            self._duplicates_prevented_edges += 1
            return False

        self._edges.append(edge)
        self._edge_keys.add(edge.key)
        return True

    def validate(self) -> list[str]:
        """Scans the builder state and returns a list of integrity violation descriptions."""
        violations: list[str] = []
        for edge in self._edges:
            if edge.source_id not in self._nodes:
                violations.append(f"Dangling edge source: '{edge.source_id}' missing.")
            if edge.target_id not in self._nodes:
                violations.append(f"Dangling edge target: '{edge.target_id}' missing.")
            if edge.source_id == edge.target_id:
                violations.append(f"Invalid self-loop on node: '{edge.source_id}'.")
        return violations

    def build(self, corpus_hash: str = "", strict: bool = True) -> Graph:
        """Finalizes graph assembly, calculates degrees and visual sizes, and sorts deterministically.

        Args:
            corpus_hash: Optional SHA-256 digest of input dataset for provenance tracking.
            strict: If True, raises GraphValidationError if any integrity violations exist.

        Returns:
            Fully populated, canonical, immutable Graph instance.
        """
        violations = self.validate()
        if violations and strict:
            raise GraphValidationError(
                f"Graph validation failed with {len(violations)} errors: {violations[:3]}"
            )

        # 1. Compute node degree centralities
        degrees: dict[str, int] = {nid: 0 for nid in self._nodes}
        for e in self._edges:
            if e.source_id in degrees:
                degrees[e.source_id] += 1
            if e.target_id in degrees:
                degrees[e.target_id] += 1

        for nid, node in self._nodes.items():
            node.degree = degrees.get(nid, 0)

        # 2. Scale visual sizes for Canvas / frontend layout (min: 12.0, max: 36.0)
        min_deg = min(degrees.values()) if degrees else 0
        max_deg = max(degrees.values()) if degrees else 0

        for node in self._nodes.values():
            if node.node_type == NodeType.PARA_CATEGORY:
                # Category hub nodes are given prominent fixed sizing
                node.visual_value = 32.0
            elif node.node_type == NodeType.SOURCE:
                node.visual_value = 16.0
            elif node.node_type == NodeType.TAG:
                node.visual_value = 14.0
            else:
                # Scale Item nodes proportionally to degree centrality
                if max_deg > min_deg:
                    ratio = (node.degree - min_deg) / (max_deg - min_deg)
                    node.visual_value = round(12.0 + (ratio * 24.0), 2)
                else:
                    node.visual_value = 18.0

        # 3. Compute structural statistics
        total_nodes = len(self._nodes)
        total_edges = len(self._edges)

        node_counts: dict[str, int] = collections.defaultdict(int)
        for n in self._nodes.values():
            node_counts[n.node_type.value] += 1

        edge_counts: dict[str, int] = collections.defaultdict(int)
        for e in self._edges:
            edge_counts[e.edge_type.value] += 1

        avg_deg = (2.0 * total_edges) / total_nodes if total_nodes > 0 else 0.0
        island_count = sum(1 for d in degrees.values() if d == 0)
        island_ratio = island_count / total_nodes if total_nodes > 0 else 0.0

        if total_nodes > 1:
            density = (2.0 * total_edges) / (total_nodes * (total_nodes - 1))
        else:
            density = 0.0

        statistics = GraphStatistics(
            total_nodes=total_nodes,
            total_edges=total_edges,
            node_counts_by_type=dict(node_counts),
            edge_counts_by_type=dict(edge_counts),
            avg_degree=round(avg_deg, 2),
            max_degree=max_deg,
            min_degree=min_deg,
            island_nodes_count=island_count,
            island_ratio=round(island_ratio, 4),
            graph_density=round(density, 4),
            duplicates_prevented_nodes=self._duplicates_prevented_nodes,
            duplicates_prevented_edges=self._duplicates_prevented_edges,
        )

        metadata = GraphMetadata(
            graph_schema_version=self.schema_version,
            projection_version=self.projection_version,
            corpus_hash=corpus_hash,
        )

        # 4. Enforce strict deterministic sort ordering:
        # - Nodes sorted by (node_type.value, id)
        # - Edges sorted by (edge_type.value, source_id, target_id)
        sorted_nodes = collections.OrderedDict(
            sorted(self._nodes.items(), key=lambda item: (item[1].node_type.value, item[0]))
        )
        sorted_edges = sorted(
            self._edges,
            key=lambda e: (e.edge_type.value, e.source_id, e.target_id),
        )

        return Graph(
            metadata=metadata,
            nodes=sorted_nodes,
            edges=sorted_edges,
            statistics=statistics,
        )
