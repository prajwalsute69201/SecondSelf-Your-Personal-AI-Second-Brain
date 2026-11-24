"""SecondSelf — Unit Test Suite for Graph Projection (Task 3.1).

Validates schema contracts, stable ID determinism, node/edge deduplication,
integrity validation (dangling edge and self-link prevention), incremental projection,
and multi-format serializers (node-link JSON, Cypher, GraphML).
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
import pytest

from graph.builder import GraphBuilder
from graph.models import (
    DanglingEdgeError,
    Edge,
    Graph,
    GraphValidationError,
    Node,
    SelfLinkError,
)
from graph.projector import GraphProjector
from graph.schema import (
    GRAPH_SCHEMA_VERSION,
    PROJECTION_VERSION,
    EdgeType,
    NodeType,
    make_item_id,
    make_para_id,
    make_source_id,
    make_tag_id,
    slugify,
)
from graph.serializers import (
    CypherSerializer,
    GraphMLSerializer,
    NodeLinkJSONSerializer,
)
from link.models import NoteItem


# ==============================================================================
# 1. Deterministic Identifier & Schema Contract Tests
# ==============================================================================

def test_deterministic_id_generators() -> None:
    # Stable Item IDs
    assert make_item_id("rec-123") == "item:rec-123"
    assert make_item_id("  rec-123  ") == "item:rec-123"

    # Stable PARA Category IDs
    assert make_para_id("Projects") == "para:projects"
    assert make_para_id("AREAS") == "para:areas"

    # Stable Tag IDs with slugification
    assert make_tag_id("Vector Search") == "tag:vector-search"
    assert make_tag_id("#ai-agents!") == "tag:ai-agents"

    # Stable Source IDs
    assert make_source_id("URL") == "source:url"
    assert make_source_id("pdf") == "source:pdf"


def test_slugify_normalization() -> None:
    assert slugify("Machine Learning 101") == "machine-learning-101"
    assert slugify("---special@#characters---") == "specialcharacters"
    assert slugify("deep_dive_rag") == "deep_dive_rag"


# ==============================================================================
# 2. GraphBuilder Deduplication & Integrity Tests
# ==============================================================================

def test_builder_node_deduplication_and_property_merging() -> None:
    builder = GraphBuilder()
    node1 = Node(
        id="item:doc-1",
        node_type=NodeType.ITEM,
        label="Doc 1",
        properties={"record_id": "doc-1", "title": "Doc 1", "category": "Projects", "char_count": 100},
    )
    added1 = builder.add_node(node1)
    assert added1 is True
    assert builder.node_count == 1

    # Add duplicate without overwrite -> should skip and increment duplicate counter
    node2_dup = Node(
        id="item:doc-1",
        node_type=NodeType.ITEM,
        label="Doc 1 Duplicate",
        properties={"record_id": "doc-1", "title": "Doc 1 Duplicate", "category": "Projects", "char_count": 200},
    )
    added2 = builder.add_node(node2_dup, overwrite=False)
    assert added2 is False
    assert builder.node_count == 1
    assert builder._nodes["item:doc-1"].properties["char_count"] == 100

    # Add duplicate with overwrite -> should update properties
    added3 = builder.add_node(node2_dup, overwrite=True)
    assert added3 is True
    assert builder.node_count == 1
    assert builder._nodes["item:doc-1"].properties["char_count"] == 200
    assert builder._duplicates_prevented_nodes == 2


def test_builder_edge_deduplication() -> None:
    builder = GraphBuilder()
    builder.add_node(Node(id="item:1", node_type=NodeType.ITEM, label="N1", properties={"record_id": "1", "title": "N1", "category": "Projects"}))
    builder.add_node(Node(id="item:2", node_type=NodeType.ITEM, label="N2", properties={"record_id": "2", "title": "N2", "category": "Projects"}))

    edge1 = Edge(
        source_id="item:1",
        target_id="item:2",
        edge_type=EdgeType.SEMANTIC_LINK,
        properties={"score": 0.85},
    )
    edge2_dup = Edge(
        source_id="item:1",
        target_id="item:2",
        edge_type=EdgeType.SEMANTIC_LINK,
        properties={"score": 0.90},
    )

    assert builder.add_edge(edge1) is True
    assert builder.edge_count == 1

    # Exact duplicate (source, target, type) should be prevented
    assert builder.add_edge(edge2_dup) is False
    assert builder.edge_count == 1
    assert builder._duplicates_prevented_edges == 1


def test_builder_rejects_dangling_edge() -> None:
    builder = GraphBuilder()
    builder.add_node(Node(id="item:1", node_type=NodeType.ITEM, label="N1", properties={"record_id": "1", "title": "N1", "category": "Projects"}))

    dangling_edge = Edge(
        source_id="item:1",
        target_id="item:missing-node",
        edge_type=EdgeType.SEMANTIC_LINK,
        properties={"score": 0.80},
    )

    with pytest.raises(DanglingEdgeError) as exc_info:
        builder.add_edge(dangling_edge, allow_dangling=False)
    assert "target node 'item:missing-node' does not exist" in str(exc_info.value)


def test_builder_rejects_self_links() -> None:
    builder = GraphBuilder()
    builder.add_node(Node(id="item:1", node_type=NodeType.ITEM, label="N1", properties={"record_id": "1", "title": "N1", "category": "Projects"}))

    self_edge = Edge(
        source_id="item:1",
        target_id="item:1",
        edge_type=EdgeType.SEMANTIC_LINK,
        properties={"score": 1.0},
    )

    with pytest.raises(SelfLinkError) as exc_info:
        builder.add_edge(self_edge, allow_self_links=False)
    assert "Self-link detected" in str(exc_info.value)


def test_builder_enforces_required_properties() -> None:
    builder = GraphBuilder()
    # Missing required 'category' property for Item node
    invalid_node = Node(
        id="item:bad",
        node_type=NodeType.ITEM,
        label="Bad",
        properties={"record_id": "bad", "title": "Bad"},
    )
    with pytest.raises(GraphValidationError) as exc_info:
        builder.add_node(invalid_node)
    assert "missing required properties: ['category']" in str(exc_info.value)


# ==============================================================================
# 3. GraphProjector End-to-End & Incremental Tests
# ==============================================================================

def test_projector_end_to_end_corpus() -> None:
    notes = [
        NoteItem(
            id="note-1",
            title="Note One",
            category="Projects",
            summary="First project note",
            tags=["vector", "search"],
            content="# Note One\n\nSome text.\n\n## Related Notes\n- [[note-2|Note Two]] — Match: 80%\n- [[missing-note|Ghost]] — Match: 95%\n",
            metadata={"source_type": "url", "confidence": 0.95},
        ),
        NoteItem(
            id="note-2",
            title="Note Two",
            category="Resources",
            summary="Reference guide",
            tags=["search"],
            content="# Note Two\n\nReference content.\n\n## Related Notes\n- [[note-1|Note One]] — Match: 80%\n",
            metadata={"source_type": "pdf", "confidence": 0.88},
        ),
    ]

    projector = GraphProjector(
        include_categories=True,
        include_tags=True,
        include_sources=True,
        similarity_threshold=0.65,
        top_k_links=5,
    )

    graph = projector.project_corpus(notes)
    stats = graph.statistics

    # 4 Category nodes + 2 Item nodes + 2 Tag nodes ('vector', 'search') + 2 Source nodes ('url', 'pdf') = 10 nodes
    assert stats.total_nodes == 10
    assert stats.node_counts_by_type["PARA_Category"] == 4
    assert stats.node_counts_by_type["Item"] == 2
    assert stats.node_counts_by_type["Tag"] == 2
    assert stats.node_counts_by_type["Source"] == 2

    # Edges:
    # 2 BELONGS_TO
    # 3 TAGGED_WITH (note-1 has vector & search, note-2 has search)
    # 2 EXTRACTED_FROM
    # 2 SEMANTIC_LINK (note-1 -> note-2, note-2 -> note-1). Note: missing-note was rejected!
    assert stats.edge_counts_by_type["BELONGS_TO"] == 2
    assert stats.edge_counts_by_type["TAGGED_WITH"] == 3
    assert stats.edge_counts_by_type["EXTRACTED_FROM"] == 2
    assert stats.edge_counts_by_type["SEMANTIC_LINK"] == 2
    assert stats.total_edges == 9

    # Verify dangling target 'missing-note' was excluded
    semantic_edges = graph.get_edges(edge_type=EdgeType.SEMANTIC_LINK)
    assert len(semantic_edges) == 2
    assert all("missing-note" not in (e.source_id, e.target_id) for e in semantic_edges)


def test_projector_threshold_and_top_k_cap() -> None:
    # Note with 3 links: 85% (keep), 70% (keep), 50% (below 65% threshold -> drop)
    notes = [
        NoteItem(
            id="a",
            title="A",
            category="Projects",
            content="# A\n## Related Notes\n- [[b|B]] — Match: 85%\n- [[c|C]] — Match: 70%\n- [[d|D]] — Match: 50%\n",
        ),
        NoteItem(id="b", title="B", category="Projects", content="# B"),
        NoteItem(id="c", title="C", category="Projects", content="# C"),
        NoteItem(id="d", title="D", category="Projects", content="# D"),
    ]

    projector = GraphProjector(
        similarity_threshold=0.65,
        top_k_links=2,  # Cap at top 2
    )

    graph = projector.project_corpus(notes)
    links_from_a = graph.get_edges(source_id="item:a", edge_type=EdgeType.SEMANTIC_LINK)

    assert len(links_from_a) == 2
    target_ids = {e.target_id for e in links_from_a}
    assert target_ids == {"item:b", "item:c"}
    assert "item:d" not in target_ids  # Below threshold


def test_incremental_projection_adds_only_new_items() -> None:
    initial_notes = [
        NoteItem(id="n1", title="N1", category="Projects", content="# N1"),
    ]
    projector = GraphProjector()
    base_graph = projector.project_corpus(initial_notes)
    assert base_graph.statistics.total_nodes >= 5  # 4 PARA + 1 Item

    # Incremental add of second note linking to first
    new_notes = [
        NoteItem(id="n2", title="N2", category="Resources", content="# N2\n## Related Notes\n- [[n1|N1]] — Match: 82%\n"),
    ]
    updated_graph = projector.project_incremental(base_graph, new_notes)

    assert updated_graph.statistics.total_nodes == base_graph.statistics.total_nodes + 1
    assert updated_graph.get_node("item:n1") is not None
    assert updated_graph.get_node("item:n2") is not None

    semantic_links = updated_graph.get_edges(edge_type=EdgeType.SEMANTIC_LINK)
    assert len(semantic_links) == 1
    assert semantic_links[0].source_id == "item:n2"
    assert semantic_links[0].target_id == "item:n1"


# ==============================================================================
# 4. Multi-Format Serializers Tests
# ==============================================================================

def test_node_link_json_serializer() -> None:
    builder = GraphBuilder()
    builder.add_node(Node(id="item:1", node_type=NodeType.ITEM, label="Doc 1", properties={"record_id": "1", "title": "Doc 1", "category": "Projects", "summary": "Sum"}))
    builder.add_node(Node(id="para:projects", node_type=NodeType.PARA_CATEGORY, label="Projects", properties={"category": "Projects", "display_name": "Projects", "color_hex": "#F59E0B"}))
    builder.add_edge(Edge(source_id="item:1", target_id="para:projects", edge_type=EdgeType.BELONGS_TO, properties={"category": "Projects", "confidence": 1.0}))
    graph = builder.build()

    serializer = NodeLinkJSONSerializer()
    json_str = serializer.serialize(graph)
    parsed = json.loads(json_str)

    assert parsed["schema_version"] == GRAPH_SCHEMA_VERSION
    assert len(parsed["nodes"]) == 2
    assert len(parsed["edges"]) == 1
    assert parsed["edges"][0]["from"] == "item:1"
    assert parsed["edges"][0]["to"] == "para:projects"
    assert parsed["nodes"][1]["color"]["background"] == "#F59E0B"


def test_cypher_serializer_syntax() -> None:
    builder = GraphBuilder()
    builder.add_node(Node(id="item:1", node_type=NodeType.ITEM, label="Doc 1", properties={"record_id": "1", "title": "Doc 1", "category": "Projects"}))
    builder.add_node(Node(id="para:projects", node_type=NodeType.PARA_CATEGORY, label="Projects", properties={"category": "Projects", "display_name": "Projects"}))
    builder.add_edge(Edge(source_id="item:1", target_id="para:projects", edge_type=EdgeType.BELONGS_TO, properties={"category": "Projects"}))
    graph = builder.build()

    serializer = CypherSerializer()
    cypher_text = serializer.serialize(graph)

    assert "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Item) REQUIRE n.id IS UNIQUE;" in cypher_text
    assert "MERGE (n:Item {id: \"item:1\"})" in cypher_text
    assert "MERGE (n:PARA_Category {id: \"para:projects\"})" in cypher_text
    assert "MERGE (s)-[r:BELONGS_TO]->(t)" in cypher_text


def test_graphml_serializer_xml_validity() -> None:
    builder = GraphBuilder()
    builder.add_node(Node(id="item:1", node_type=NodeType.ITEM, label="Doc 1", properties={"record_id": "1", "title": "Doc 1", "category": "Projects"}))
    builder.add_node(Node(id="para:projects", node_type=NodeType.PARA_CATEGORY, label="Projects", properties={"category": "Projects", "display_name": "Projects"}))
    builder.add_edge(Edge(source_id="item:1", target_id="para:projects", edge_type=EdgeType.BELONGS_TO, properties={"category": "Projects"}))
    graph = builder.build()

    serializer = GraphMLSerializer()
    xml_text = serializer.serialize(graph)

    # Parse XML to verify syntax validity
    root = ET.fromstring(xml_text)
    assert root.tag.endswith("graphml")
    nodes = [elem for elem in root.iter() if elem.tag.endswith("node")]
    edges = [elem for elem in root.iter() if elem.tag.endswith("edge")]
    assert len(nodes) == 2
    assert len(edges) == 1
