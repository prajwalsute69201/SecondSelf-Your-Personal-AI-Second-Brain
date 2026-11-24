"""SecondSelf — XML GraphML Serializer.

Emits standard XML GraphML specification files for interoperability with Gephi,
Cytoscape, NetworkX, and yEd Graph Editor.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from xml.dom import minidom
from graph.models import Graph
from graph.serializers.base import GraphSerializer


class GraphMLSerializer(GraphSerializer):
    """Serializes Graph to standard XML GraphML representation."""

    @property
    def extension(self) -> str:
        return ".graphml"

    def serialize(self, graph: Graph) -> str:
        """Generates formatted, indented GraphML XML string."""
        root = ET.Element(
            "graphml",
            {
                "xmlns": "http://graphml.graphdrawing.org/xmlns",
                "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
                "xsi:schemaLocation": "http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd",
            },
        )

        # 1. Attribute Key Declarations
        node_keys = [
            ("label", "string", "node"),
            ("type", "string", "node"),
            ("group", "string", "node"),
            ("degree", "int", "node"),
            ("visual_value", "double", "node"),
            ("summary", "string", "node"),
            ("confidence", "double", "node"),
        ]
        edge_keys = [
            ("type", "string", "edge"),
            ("score", "double", "edge"),
            ("similarity_pct", "int", "edge"),
            ("method", "string", "edge"),
            ("reason", "string", "edge"),
        ]

        for k_id, k_type, k_for in node_keys:
            ET.SubElement(root, "key", {"id": f"d_{k_id}", "for": k_for, "attr.name": k_id, "attr.type": k_type})

        for k_id, k_type, k_for in edge_keys:
            ET.SubElement(root, "key", {"id": f"e_{k_id}", "for": k_for, "attr.name": k_id, "attr.type": k_type})

        # 2. Graph Element
        graph_elem = ET.SubElement(root, "graph", {"id": "G", "edgedefault": "directed"})

        # 3. Node Elements
        for n in graph.nodes.values():
            node_elem = ET.SubElement(graph_elem, "node", {"id": n.id})
            self._add_data(node_elem, "d_label", n.label)
            self._add_data(node_elem, "d_type", n.node_type.value)
            self._add_data(node_elem, "d_group", n.group)
            self._add_data(node_elem, "d_degree", str(n.degree))
            self._add_data(node_elem, "d_visual_value", str(n.visual_value))
            if "summary" in n.properties:
                self._add_data(node_elem, "d_summary", str(n.properties["summary"]))
            if "confidence" in n.properties:
                self._add_data(node_elem, "d_confidence", str(n.properties["confidence"]))

        # 4. Edge Elements
        for idx, e in enumerate(graph.edges):
            edge_id = f"e{idx}"
            edge_elem = ET.SubElement(graph_elem, "edge", {"id": edge_id, "source": e.source_id, "target": e.target_id})
            self._add_data(edge_elem, "e_type", e.edge_type.value)
            if "score" in e.properties:
                self._add_data(edge_elem, "e_score", str(e.properties["score"]))
            if "similarity_pct" in e.properties:
                self._add_data(edge_elem, "e_similarity_pct", str(e.properties["similarity_pct"]))
            if "method" in e.properties:
                self._add_data(edge_elem, "e_method", str(e.properties["method"]))
            if "reason" in e.properties:
                self._add_data(edge_elem, "e_reason", str(e.properties["reason"]))

        # Pretty-print XML
        raw_xml = ET.tostring(root, encoding="utf-8")
        parsed = minidom.parseString(raw_xml)
        return parsed.toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")

    @staticmethod
    def _add_data(parent: ET.Element, key: str, value: str) -> None:
        if value is not None and value != "":
            d = ET.SubElement(parent, "data", {"key": key})
            d.text = str(value)
