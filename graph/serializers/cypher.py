"""SecondSelf — Neo4j & Memgraph Cypher Statement Serializer.

Emits deterministic, idempotent Cypher DDL (constraints, indexes) and DML (MERGE)
statements ready for loading into Neo4j or Memgraph graph databases.
"""

from __future__ import annotations

import json
from typing import Any
from graph.models import Graph, NodeType
from graph.serializers.base import GraphSerializer


def _escape_cypher_string(value: str) -> str:
    """Escapes string literal for safe Cypher query inclusion."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r")


def _format_cypher_literal(value: Any) -> str:
    """Formats Python primitive or collection into Cypher literal expression."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        items = [_format_cypher_literal(v) for v in value]
        return f"[{', '.join(items)}]"
    return f'"{_escape_cypher_string(str(value))}"'


class CypherSerializer(GraphSerializer):
    """Generates executable Cypher scripts containing constraints, nodes, and relationships."""

    @property
    def extension(self) -> str:
        return ".cypher"

    def serialize(self, graph: Graph) -> str:
        """Renders complete Cypher script with schema constraints, nodes, and edges."""
        lines: list[str] = [
            f"// ============================================================================",
            f"// SecondSelf Knowledge Graph — Neo4j / Memgraph Cypher Import Script",
            f"// Schema Version: {graph.metadata.graph_schema_version} | Projection Version: {graph.metadata.projection_version}",
            f"// Generated At: {graph.metadata.generated_at}",
            f"// Corpus Hash: {graph.metadata.corpus_hash}",
            f"// Total Nodes: {graph.statistics.total_nodes} | Total Edges: {graph.statistics.total_edges}",
            f"// ============================================================================",
            f"",
            f"// 1. Schema Constraints & Uniqueness Indexes",
            f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:Item) REQUIRE n.id IS UNIQUE;",
            f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:PARA_Category) REQUIRE n.id IS UNIQUE;",
            f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:Tag) REQUIRE n.id IS UNIQUE;",
            f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:Source) REQUIRE n.id IS UNIQUE;",
            f"",
            f"// 2. Node Insertion & Property Setting (Idempotent MERGE)",
        ]

        # 2. Emit Nodes
        for node in graph.nodes.values():
            lbl = node.node_type.value
            node_id_lit = _format_cypher_literal(node.id)
            label_lit = _format_cypher_literal(node.label)

            set_clauses = [f"n.label = {label_lit}", f"n.degree = {node.degree}"]
            for prop_key, prop_val in sorted(node.properties.items()):
                val_lit = _format_cypher_literal(prop_val)
                set_clauses.append(f"n.{prop_key} = {val_lit}")

            set_stmt = ", ".join(set_clauses)
            lines.append(f"MERGE (n:{lbl} {{id: {node_id_lit}}}) SET {set_stmt};")

        lines.extend([
            f"",
            f"// 3. Relationship Insertion (Idempotent MERGE)",
        ])

        # 3. Emit Edges
        for edge in graph.edges:
            src_lit = _format_cypher_literal(edge.source_id)
            tgt_lit = _format_cypher_literal(edge.target_id)
            rel_type = edge.edge_type.value

            set_clauses: list[str] = []
            for prop_key, prop_val in sorted(edge.properties.items()):
                val_lit = _format_cypher_literal(prop_val)
                set_clauses.append(f"r.{prop_key} = {val_lit}")

            if set_clauses:
                set_stmt = " SET " + ", ".join(set_clauses)
            else:
                set_stmt = ""

            lines.append(
                f"MATCH (s {{id: {src_lit}}}), (t {{id: {tgt_lit}}}) "
                f"MERGE (s)-[r:{rel_type}]->(t){set_stmt};"
            )

        return "\n".join(lines) + "\n"
