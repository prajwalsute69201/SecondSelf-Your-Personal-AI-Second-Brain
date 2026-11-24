"""SecondSelf — Graph Data Modeling and Projection Subsystem (Phase 3).

Exposes the typed graph schema, models, GraphBuilder, GraphProjector, and serializers.
"""

from graph.builder import GraphBuilder
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
from graph.projector import GraphProjector
from graph.schema import (
    EDGE_SCHEMAS,
    GRAPH_SCHEMA_VERSION,
    NODE_SCHEMAS,
    PROJECTION_VERSION,
    EdgeType,
    NodeType,
    PropertySchema,
    make_item_id,
    make_para_id,
    make_source_id,
    make_tag_id,
    slugify,
)
from graph.serializers import (
    CypherSerializer,
    GraphMLSerializer,
    GraphSerializer,
    NodeLinkJSONSerializer,
)

__all__ = [
    "GRAPH_SCHEMA_VERSION",
    "PROJECTION_VERSION",
    "NodeType",
    "EdgeType",
    "PropertySchema",
    "NODE_SCHEMAS",
    "EDGE_SCHEMAS",
    "make_item_id",
    "make_para_id",
    "make_tag_id",
    "make_source_id",
    "slugify",
    "Node",
    "Edge",
    "Graph",
    "GraphMetadata",
    "GraphStatistics",
    "GraphValidationError",
    "DanglingEdgeError",
    "SelfLinkError",
    "GraphBuilder",
    "GraphProjector",
    "GraphSerializer",
    "NodeLinkJSONSerializer",
    "CypherSerializer",
    "GraphMLSerializer",
]
