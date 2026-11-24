"""SecondSelf — Node-Link JSON Graph Serializer.

Emits dual-purpose JSON compatible with both the D3/NetworkX node-link specification
and frontend force-directed canvas rendering engines (Vis.js / Sigma.js).
"""

from __future__ import annotations

import json
from typing import Any
from graph.models import Graph, NodeType
from graph.serializers.base import GraphSerializer


class NodeLinkJSONSerializer(GraphSerializer):
    """Serializes Graph to structured node-link JSON format."""

    @property
    def extension(self) -> str:
        return ".json"

    def serialize(self, graph: Graph, indent: int = 2) -> str:
        """Serializes Graph to formatted JSON string."""
        payload = self.to_dict(graph)
        return json.dumps(payload, indent=indent, sort_keys=False)

    def to_dict(self, graph: Graph) -> dict[str, Any]:
        """Builds dictionary payload for JSON serialization."""
        nodes_payload: list[dict[str, Any]] = []
        for n in graph.nodes.values():
            node_dict: dict[str, Any] = {
                "id": n.id,
                "label": n.label,
                "type": n.node_type.value,
                "group": n.group,
                "value": n.visual_value,
                "degree": n.degree,
                "summary": n.get_property("summary", ""),
                "tags": n.get_property("tags", []),
                "properties": n.properties,
            }
            # Special visual color mapping for PARA Categories if present
            color_hex = n.get_property("color_hex")
            if color_hex:
                node_dict["color"] = {"background": color_hex, "border": color_hex}

            nodes_payload.append(node_dict)

        edges_payload: list[dict[str, Any]] = []
        for e in graph.edges:
            weight = float(e.get_property("score", 1.0))
            sim_pct = e.get_property("similarity_pct")
            title_tooltip = f"{e.edge_type.value} (Match: {sim_pct}%)" if sim_pct else e.edge_type.value

            edge_dict: dict[str, Any] = {
                "source": e.source_id,
                "target": e.target_id,
                "from": e.source_id,  # Vis.js compatibility
                "to": e.target_id,    # Vis.js compatibility
                "type": e.edge_type.value,
                "value": weight,
                "title": title_tooltip,
                "properties": e.properties,
            }
            edges_payload.append(edge_dict)

        return {
            "schema_version": graph.metadata.graph_schema_version,
            "projection_version": graph.metadata.projection_version,
            "generated_at": graph.metadata.generated_at,
            "corpus_hash": graph.metadata.corpus_hash,
            "statistics": graph.statistics.model_dump(),
            "nodes": nodes_payload,
            "edges": edges_payload,
            "links": edges_payload,  # D3.js node-link standard alias
        }
