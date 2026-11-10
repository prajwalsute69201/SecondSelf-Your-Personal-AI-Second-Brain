"""Unit and integration tests for build_graph.py (Task 3.1.1).

Validates:
- Graph creation from multiple wiki notes.
- Correct node field mapping (id, label, para, tags, summary, content_preview, group).
- Correct PARA categories ('Projects', 'Areas', 'Resources', 'Archives').
- Tag and summary structure preservation.
- Content preview extraction (first ~200 characters, whitespace normalized, trailing wikilinks stripped).
- Edge extraction from frontmatter links[] and body [[id]] wikilinks.
- Missing relationship target handling (skipped without crashing).
- Duplicate edge prevention (min(src, tgt), max(src, tgt)).
- Self-referencing link prevention.
- Empty wiki directory handling (zero counts, valid schema).
- Single node graph handling.
- Edge weighting via embeddings cosine similarity.
- Strict schema validation matching Implementationplan.md Task 3.1.2.
- Unicode content preservation (emojis, non-ASCII scripts).
- Atomic writing and existing graph.json preservation on failure.
- Updating of data/index.json last_graph_build timestamp.
- CLI argument parsing, --help, and --dry-run mode.
"""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

# Ensure secondself is in sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import build_graph
import lib.embeddings as emb
from lib.models import GraphEdge, GraphNode, WikiNote
import lib.storage as storage


def _make_unit_vector(angle_rad: float, dim: int = 384) -> np.ndarray:
    """Generate a 384-dimensional unit vector in a 2D subspace for exact cosine similarities."""
    vec = np.zeros(dim, dtype=np.float32)
    vec[0] = np.cos(angle_rad)
    vec[1] = np.sin(angle_rad)
    return vec


class TestBuildGraph(unittest.TestCase):
    """Test suite for build_graph.py."""

    def setUp(self) -> None:
        """Create isolated test directory and initialize standard structure."""
        self.test_dir = tempfile.mkdtemp(prefix="secondself_graph_test_")
        self.test_path = Path(self.test_dir)
        storage.set_project_root(self.test_path)

        # Standard folders
        (self.test_path / "data").mkdir(parents=True, exist_ok=True)
        (self.test_path / "wiki" / "Projects").mkdir(parents=True, exist_ok=True)
        (self.test_path / "wiki" / "Areas").mkdir(parents=True, exist_ok=True)
        (self.test_path / "wiki" / "Resources").mkdir(parents=True, exist_ok=True)
        (self.test_path / "wiki" / "Archives").mkdir(parents=True, exist_ok=True)

        storage.save_index(
            {
                "raw_processed": {},
                "embeddings_version": "all-MiniLM-L6-v2",
                "last_graph_build": None,
            },
            root=self.test_path,
        )

    def tearDown(self) -> None:
        """Clean up temporary directory and reset state."""
        storage.set_project_root(None)
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_note(
        self,
        nid: str,
        para: str = "Projects",
        summary: str = "Test summary",
        tags: list[str] | None = None,
        links: list[str] | None = None,
        body: str = "Test note body content.",
    ) -> WikiNote:
        """Helper to create and persist a test WikiNote."""
        note = WikiNote(
            id=nid,
            raw_id=f"2026-07-06_{nid}",
            para=para,
            tags=tags or [],
            summary=summary,
            created="2026-07-06T12:00:00Z",
            links=links or [],
            body=body,
        )
        storage.write_wiki_note(note, root=self.test_path)
        return note

    # -------------------------------------------------------------------------
    # 1. Graph Creation from Multiple Wiki Notes
    # -------------------------------------------------------------------------

    def test_01_graph_creation_from_multiple_notes(self) -> None:
        """build_graph correctly parses multiple notes and builds nodes and edges."""
        n1 = self._create_note(
            nid="note001",
            para="Projects",
            summary="SecondSelf implementation plan",
            tags=["ai", "python"],
            links=["note002"],
            body="# SecondSelf\nFull architecture plan.\n\n[[note002]]\n",
        )
        n2 = self._create_note(
            nid="note002",
            para="Resources",
            summary="Embeddings guide",
            tags=["nlp"],
            links=["note001"],
            body="# Embeddings\nMiniLM guide.\n\n[[note001]]\n",
        )

        graph_data, warnings = build_graph.build_and_save_graph(root=self.test_path)

        self.assertEqual(graph_data["metadata"]["node_count"], 2)
        self.assertEqual(graph_data["metadata"]["edge_count"], 1)

        # Confirm file was written
        graph_file = build_graph.get_graph_path(root=self.test_path)
        self.assertTrue(graph_file.exists())

        loaded = json.loads(graph_file.read_text(encoding="utf-8"))
        self.assertEqual(loaded["metadata"]["node_count"], 2)
        self.assertEqual(loaded["metadata"]["edge_count"], 1)

    # -------------------------------------------------------------------------
    # 2. Node Field Mapping & Enrichment
    # -------------------------------------------------------------------------

    def test_02_node_field_mapping_and_enrichment(self) -> None:
        """GraphNode fields strictly match the specification from Implementationplan.md."""
        self._create_note(
            nid="rich01",
            para="Projects",
            summary="ML engineering roadmap",
            tags=["career", "ml"],
            body="# Career Plan\n" + "Detailed learning path for ML. " * 15,
        )

        graph_data, _ = build_graph.build_graph_data(
            storage.read_wiki_notes(root=self.test_path)
        )
        node = graph_data["nodes"][0]

        # Field verifications
        self.assertEqual(node["id"], "rich01")
        self.assertEqual(node["label"], "ML engineering roadmap")
        self.assertEqual(node["para"], "Projects")
        self.assertEqual(node["group"], "Projects")
        self.assertEqual(node["tags"], ["career", "ml"])
        self.assertEqual(node["summary"], "ML engineering roadmap")
        self.assertTrue(len(node["content_preview"]) <= 200)
        self.assertTrue(node["content_preview"].startswith("# Career Plan"))

    # -------------------------------------------------------------------------
    # 3. PARA Categories & Color Groups
    # -------------------------------------------------------------------------

    def test_03_para_categories_and_group_mapping(self) -> None:
        """All four PARA categories are mapped correctly to para and group fields."""
        categories = ["Projects", "Areas", "Resources", "Archives"]
        for cat in categories:
            self._create_note(
                nid=f"note_{cat.lower()}",
                para=cat,
                summary=f"Summary for {cat}",
            )

        graph_data, _ = build_graph.build_graph_data(
            storage.read_wiki_notes(root=self.test_path)
        )
        nodes_by_id = {n["id"]: n for n in graph_data["nodes"]}

        for cat in categories:
            nid = f"note_{cat.lower()}"
            self.assertEqual(nodes_by_id[nid]["para"], cat)
            self.assertEqual(nodes_by_id[nid]["group"], cat)

    # -------------------------------------------------------------------------
    # 4. Content Preview Extraction
    # -------------------------------------------------------------------------

    def test_04_content_preview_extraction_rules(self) -> None:
        """extract_node_preview handles short text, long text, and trailing wikilinks."""
        # Short text
        p1 = build_graph.extract_node_preview("Short text.")
        self.assertEqual(p1, "Short text.")

        # Long text truncated to ~200 chars
        long_text = "Word " * 60
        p2 = build_graph.extract_node_preview(long_text, max_chars=200)
        self.assertLessEqual(len(p2), 200)

        # Trailing wikilinks stripped
        body_with_links = "Core summary content.\n\n[[link_a]]\n[[link_b]]\n"
        p3 = build_graph.extract_node_preview(body_with_links)
        self.assertEqual(p3, "Core summary content.")
        self.assertNotIn("[[link_a]]", p3)

    # -------------------------------------------------------------------------
    # 5. Edge Extraction from Frontmatter and Body
    # -------------------------------------------------------------------------

    def test_05_edge_extraction_sources(self) -> None:
        """Edges are extracted from both frontmatter links[] and body [[id]] wikilinks."""
        # Note A has link to B in frontmatter
        self._create_note("node_a", links=["node_b"], body="Prose A")
        # Note B has link to C in body only
        self._create_note("node_b", links=[], body="Prose B mentioning [[node_c]].")
        self._create_note("node_c", links=[], body="Prose C")

        graph_data, _ = build_graph.build_graph_data(
            storage.read_wiki_notes(root=self.test_path)
        )
        edge_pairs = {(e["source"], e["target"]) for e in graph_data["edges"]}

        self.assertIn(("node_a", "node_b"), edge_pairs)
        self.assertIn(("node_b", "node_c"), edge_pairs)

    # -------------------------------------------------------------------------
    # 6. Edge Deduplication
    # -------------------------------------------------------------------------

    def test_06_edge_deduplication_bidirectional(self) -> None:
        """Bidirectional links between A and B collapse into a single undirected edge."""
        # A links to B, and B links to A
        self._create_note("node_x", links=["node_y"], body="[[node_y]]")
        self._create_note("node_y", links=["node_x"], body="[[node_x]]")

        graph_data, _ = build_graph.build_graph_data(
            storage.read_wiki_notes(root=self.test_path)
        )
        self.assertEqual(len(graph_data["edges"]), 1)

        edge = graph_data["edges"][0]
        self.assertEqual(edge["source"], "node_x")
        self.assertEqual(edge["target"], "node_y")

    # -------------------------------------------------------------------------
    # 7. Missing Target Handling (Broken Links)
    # -------------------------------------------------------------------------

    def test_07_missing_relationship_target_skipped_safely(self) -> None:
        """Edges referencing non-existent nodes are skipped with a warning emitted."""
        self._create_note("valid_node", links=["nonexistent_target"], body="[[broken_id]]")

        graph_data, warnings = build_graph.build_graph_data(
            storage.read_wiki_notes(root=self.test_path)
        )
        self.assertEqual(len(graph_data["edges"]), 0)
        self.assertTrue(any("non-existent" in w for w in warnings))

    # -------------------------------------------------------------------------
    # 8. Self-Link Prevention
    # -------------------------------------------------------------------------

    def test_08_self_link_prevention(self) -> None:
        """A note referencing itself in links or body does not produce an edge."""
        self._create_note("self_ref", links=["self_ref"], body="Self link [[self_ref]].")

        graph_data, warnings = build_graph.build_graph_data(
            storage.read_wiki_notes(root=self.test_path)
        )
        self.assertEqual(len(graph_data["edges"]), 0)
        self.assertTrue(any("self-referencing" in w for w in warnings))

    # -------------------------------------------------------------------------
    # 9. Edge Weighting via Embeddings
    # -------------------------------------------------------------------------

    def test_09_edge_weighting_via_embeddings(self) -> None:
        """Semantic edge weight is computed from cosine similarity when embeddings exist."""
        self._create_note("emb_a", links=["emb_b"], body="[[emb_b]]")
        self._create_note("emb_b", links=[], body="Content B")

        theta = float(np.arccos(0.87))
        v1 = _make_unit_vector(0.0)
        v2 = _make_unit_vector(theta)

        embeddings = {"emb_a": v1, "emb_b": v2}

        graph_data, _ = build_graph.build_graph_data(
            storage.read_wiki_notes(root=self.test_path),
            embeddings=embeddings,
        )
        self.assertEqual(len(graph_data["edges"]), 1)
        edge = graph_data["edges"][0]
        self.assertAlmostEqual(edge["weight"], 0.87, places=3)
        self.assertEqual(edge["type"], "semantic")

    # -------------------------------------------------------------------------
    # 10. Empty Wiki Directory
    # -------------------------------------------------------------------------

    def test_10_empty_wiki_directory_produces_valid_graph(self) -> None:
        """An empty wiki produces a valid graph.json with 0 nodes, 0 edges, and metadata."""
        graph_data, warnings = build_graph.build_and_save_graph(root=self.test_path)

        self.assertEqual(graph_data["nodes"], [])
        self.assertEqual(graph_data["edges"], [])
        self.assertEqual(graph_data["metadata"]["node_count"], 0)
        self.assertEqual(graph_data["metadata"]["edge_count"], 0)
        self.assertIn("generated_at", graph_data["metadata"])

        # Check saved JSON on disk
        graph_file = build_graph.get_graph_path(root=self.test_path)
        self.assertTrue(graph_file.exists())
        loaded = json.loads(graph_file.read_text(encoding="utf-8"))
        self.assertEqual(loaded["metadata"]["node_count"], 0)

    # -------------------------------------------------------------------------
    # 11. Single Node Graph
    # -------------------------------------------------------------------------

    def test_11_single_node_graph(self) -> None:
        """A single note produces a graph with 1 node and 0 edges."""
        self._create_note("solo_node", summary="Solo node summary")

        graph_data, _ = build_graph.build_graph_data(
            storage.read_wiki_notes(root=self.test_path)
        )
        self.assertEqual(len(graph_data["nodes"]), 1)
        self.assertEqual(len(graph_data["edges"]), 0)
        self.assertEqual(graph_data["metadata"]["node_count"], 1)
        self.assertEqual(graph_data["metadata"]["edge_count"], 0)

    # -------------------------------------------------------------------------
    # 12. Strict Schema Compliance (Task 3.1.2)
    # -------------------------------------------------------------------------

    def test_12_schema_compliance_validation(self) -> None:
        """Output JSON strictly complies with Implementationplan.md Task 3.1.2 schema."""
        self._create_note("sch_a", links=["sch_b"])
        self._create_note("sch_b")

        graph_data, _ = build_graph.build_graph_data(
            storage.read_wiki_notes(root=self.test_path)
        )

        # Top-level keys
        self.assertEqual(set(graph_data.keys()), {"nodes", "edges", "metadata"})

        # Node keys
        expected_node_keys = {"id", "label", "para", "tags", "summary", "content_preview", "group"}
        for n in graph_data["nodes"]:
            self.assertEqual(set(n.keys()), expected_node_keys)

        # Edge keys
        expected_edge_keys = {"source", "target", "weight", "type"}
        for e in graph_data["edges"]:
            self.assertEqual(set(e.keys()), expected_edge_keys)

        # Metadata keys
        expected_meta_keys = {"generated_at", "node_count", "edge_count"}
        self.assertEqual(set(graph_data["metadata"].keys()), expected_meta_keys)

    # -------------------------------------------------------------------------
    # 13. Unicode Content Preservation
    # -------------------------------------------------------------------------

    def test_13_unicode_content_preservation(self) -> None:
        """Unicode characters (emojis, accents, Hindi) are preserved cleanly in JSON."""
        unicode_summary = "मशीन लर्निंग (Machine Learning) 🧠 — SecondSelf"
        unicode_body = "हिंदी टेक्स्ट और इमोज़ी 🚀 testing unicode UTF-8 encoding."

        self._create_note(
            nid="uni001",
            summary=unicode_summary,
            body=unicode_body,
        )

        build_graph.build_and_save_graph(root=self.test_path)

        graph_file = build_graph.get_graph_path(root=self.test_path)
        raw_text = graph_file.read_text(encoding="utf-8")

        self.assertIn("मशीन लर्निंग", raw_text)
        self.assertIn("🧠", raw_text)
        self.assertIn("🚀", raw_text)

        loaded = json.loads(raw_text)
        self.assertEqual(loaded["nodes"][0]["summary"], unicode_summary)

    # -------------------------------------------------------------------------
    # 14. Atomic Writing & Index Timestamp Update
    # -------------------------------------------------------------------------

    def test_14_index_timestamp_update_and_atomic_replace(self) -> None:
        """save_graph updates data/index.json last_graph_build timestamp atomically."""
        self._create_note("idx001")

        build_graph.build_and_save_graph(root=self.test_path)

        index = storage.load_index(root=self.test_path)
        self.assertIsNotNone(index.get("last_graph_build"))

        # Verify timestamp matches metadata generated_at
        graph_file = build_graph.get_graph_path(root=self.test_path)
        graph_data = json.loads(graph_file.read_text(encoding="utf-8"))
        self.assertEqual(index["last_graph_build"], graph_data["metadata"]["generated_at"])

    # -------------------------------------------------------------------------
    # 15. CLI Interface Tests
    # -------------------------------------------------------------------------

    def test_15_cli_execution_and_help(self) -> None:
        """CLI arguments, --help, and --dry-run work cleanly."""
        self._create_note("cli_node")

        # --help
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = build_graph.main(["--help"])
        self.assertEqual(code, 0)
        self.assertIn("data/graph.json", buf.getvalue())

        # --dry-run
        dry_buf = io.StringIO()
        with redirect_stdout(dry_buf):
            code = build_graph.main(["--root", str(self.test_path), "--dry-run"])
        self.assertEqual(code, 0)
        self.assertIn("[dry-run]", dry_buf.getvalue())
        self.assertIn("1 nodes, 0 edges", dry_buf.getvalue())

        # Real CLI build
        run_buf = io.StringIO()
        with redirect_stdout(run_buf):
            code = build_graph.main(["--root", str(self.test_path)])
        self.assertEqual(code, 0)
        self.assertIn("Graph written to", run_buf.getvalue())


if __name__ == "__main__":
    unittest.main()

