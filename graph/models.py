"""SecondSelf — Graph Data Contracts & Model Definitions.

Defines Node, Edge, Graph, and GraphStatistics models for holding in-memory
knowledge graphs prior to serialization into JSON, Cypher, or GraphML formats.
"""

from __future__ import annotations

import datetime
from typing import Any
from pydantic import BaseModel, ConfigDict, Field

from graph.schema import (
    GRAPH_SCHEMA_VERSION,
    PROJECTION_VERSION,
    EdgeType,
    NodeType,
)


class GraphValidationError(Exception):
    """Raised when node or edge properties violate schema contracts."""
    pass


class DanglingEdgeError(GraphValidationError):
    """Raised when an edge references a non-existent source or target node ID."""
    pass


class SelfLinkError(GraphValidationError):
    """Raised when an edge has identical source and target IDs."""
    pass


class Node(BaseModel):
    """A discrete knowledge entity in the SecondSelf graph."""

    id: str = Field(..., description="Stable, unique canonical node identifier")
    node_type: NodeType = Field(..., description="Categorical classification of the node")
    label: str = Field(..., description="Primary human-readable display label")
    properties: dict[str, Any] = Field(default_factory=dict, description="Typed attributes")
    degree: int = Field(default=0, description="Total node degree (incoming + outgoing)")
    visual_value: float = Field(default=15.0, description="Visual display size for canvas physics")

    model_config = ConfigDict(populate_by_name=True)

    @property
    def group(self) -> str:
        """Visual clustering group for frontend styling."""
        if self.node_type == NodeType.ITEM:
            return str(self.properties.get("category", "Resources"))
        return self.node_type.value

    def get_property(self, name: str, default: Any = None) -> Any:
        """Retrieves a specific property value with fallback default."""
        return self.properties.get(name, default)

    def to_dict(self) -> dict[str, Any]:
        """Serializes node to standard dictionary representation."""
        return {
            "id": self.id,
            "type": self.node_type.value,
            "label": self.label,
            "group": self.group,
            "degree": self.degree,
            "visual_value": round(self.visual_value, 2),
            "properties": self.properties,
        }


class Edge(BaseModel):
    """A directed, typed semantic association connecting two nodes."""

    source_id: str = Field(..., description="Identifier of the origin node")
    target_id: str = Field(..., description="Identifier of the destination node")
    edge_type: EdgeType = Field(..., description="Semantic relationship classification")
    properties: dict[str, Any] = Field(default_factory=dict, description="Edge attributes")

    model_config = ConfigDict(populate_by_name=True)

    @property
    def key(self) -> tuple[str, str, str]:
        """Unique compound key for edge deduplication: (source, target, type)."""
        return (self.source_id, self.target_id, self.edge_type.value)

    def get_property(self, name: str, default: Any = None) -> Any:
        """Retrieves an edge attribute with fallback default."""
        return self.properties.get(name, default)

    def to_dict(self) -> dict[str, Any]:
        """Serializes edge to standard dictionary representation."""
        return {
            "source": self.source_id,
            "target": self.target_id,
            "type": self.edge_type.value,
            "properties": self.properties,
        }


class GraphStatistics(BaseModel):
    """Aggregate topological metrics and health indicators of the projected graph."""

    total_nodes: int = 0
    total_edges: int = 0
    node_counts_by_type: dict[str, int] = Field(default_factory=dict)
    edge_counts_by_type: dict[str, int] = Field(default_factory=dict)
    avg_degree: float = 0.0
    max_degree: int = 0
    min_degree: int = 0
    island_nodes_count: int = 0
    island_ratio: float = 0.0
    graph_density: float = 0.0
    duplicates_prevented_nodes: int = 0
    duplicates_prevented_edges: int = 0

    model_config = ConfigDict(populate_by_name=True)


class GraphMetadata(BaseModel):
    """Provenance and schema metadata describing the projected graph."""

    graph_schema_version: str = GRAPH_SCHEMA_VERSION
    projection_version: str = PROJECTION_VERSION
    generated_at: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    corpus_hash: str = Field(default="", description="Deterministic SHA-256 digest of input corpus")

    model_config = ConfigDict(populate_by_name=True)


class Graph(BaseModel):
    """Canonical, immutable knowledge graph container holding nodes, edges, and statistics."""

    metadata: GraphMetadata = Field(default_factory=GraphMetadata)
    nodes: dict[str, Node] = Field(default_factory=dict)
    edges: list[Edge] = Field(default_factory=list)
    statistics: GraphStatistics = Field(default_factory=GraphStatistics)

    model_config = ConfigDict(populate_by_name=True)

    def get_node(self, node_id: str) -> Node | None:
        """Looks up a node by its canonical ID."""
        return self.nodes.get(node_id)

    def get_edges(
        self,
        source_id: str | None = None,
        target_id: str | None = None,
        edge_type: EdgeType | None = None,
    ) -> list[Edge]:
        """Filters edges by source ID, target ID, and/or EdgeType."""
        filtered: list[Edge] = []
        for e in self.edges:
            if source_id is not None and e.source_id != source_id:
                continue
            if target_id is not None and e.target_id != target_id:
                continue
            if edge_type is not None and e.edge_type != edge_type:
                continue
            filtered.append(e)
        return filtered

    def to_dict(self) -> dict[str, Any]:
        """Serializes the entire graph to a structured dictionary."""
        return {
            "metadata": self.metadata.model_dump(),
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
            "statistics": self.statistics.model_dump(),
        }
