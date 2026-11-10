"""Unit and integration tests for validate_graph.py (Task 3.1.2).

Validates:
- Valid populated graph passes schema and consistency checks.
- Valid empty graph (0 nodes, 0 edges, valid metadata) passes.
- Invalid JSON syntax and non-standard tokens (NaN, Infinity) fail cleanly.
- Top-level structure violations (missing keys, wrong types) fail.
- Metadata validation (missing fields, wrong types, node/edge count mismatches).
- Node attribute validation (missing fields, blank IDs, duplicate IDs, invalid PARA, wrong tag types).
- Edge attribute validation (missing fields, blank endpoints, non-existent endpoints).
- Graph invariants (self-referencing edges, duplicate undirected edges).
- Edge weight and type validation (non-finite numbers, booleans, invalid types).
- Non-existent file error handling.
- CLI execution and exit codes (0 for PASS, 1 for FAIL).
"""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

# Ensure secondself is in sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import validate_graph as vg


def _create_minimal_valid_graph_dict() -> dict:
    """Return a deep copy of a minimal valid graph dictionary with 2 nodes and 1 edge."""
    return {
        "nodes": [
            {
                "id": "note-1",
                "label": "Note 1",
                "para": "Projects",
                "tags": ["tag1", "tag2"],
                "summary": "First note summary",
                "content_preview": "Preview of note 1 content.",
                "group": "Projects",
            },
            {
                "id": "note-2",
                "label": "Note 2",
                "para": "Resources",
                "tags": ["reference"],
                "summary": "Second note summary",
                "content_preview": "Preview of note 2 content.",
                "group": "Resources",
            },
        ],
        "edges": [
            {
                "source": "note-1",
                "target": "note-2",
                "weight": 0.85,
                "type": "semantic",
            }
        ],
        "metadata": {
            "generated_at": "2026-11-09T10:00:00Z",
            "node_count": 2,
            "edge_count": 1,
        },
    }


class TestValidateGraphValid(unittest.TestCase):
    """Tests for graphs that should successfully PASS validation."""

    def test_valid_populated_graph(self) -> None:
        """A properly formatted graph with nodes and edges passes all checks."""
        data = _create_minimal_valid_graph_dict()
        report = vg.validate_graph_dict(data)

        self.assertTrue(report.valid)
        self.assertTrue(report.json_valid)
        self.assertTrue(report.top_level_valid)
        self.assertTrue(report.nodes_valid)
        self.assertTrue(report.edges_valid)
        self.assertTrue(report.consistency_valid)
        self.assertEqual(report.node_count, 2)
        self.assertEqual(report.edge_count, 1)
        self.assertEqual(report.error_count, 0)
        self.assertEqual(len(report.issues), 0)

    def test_valid_empty_graph(self) -> None:
        """An empty graph with 0 nodes and 0 edges is valid according to schema."""
        data = {
            "nodes": [],
            "edges": [],
            "metadata": {
                "generated_at": "2026-11-09T00:00:00Z",
                "node_count": 0,
                "edge_count": 0,
            },
        }
        report = vg.validate_graph_dict(data)

        self.assertTrue(report.valid)
        self.assertEqual(report.node_count, 0)
        self.assertEqual(report.edge_count, 0)
        self.assertEqual(report.error_count, 0)

    def test_all_para_categories_accepted(self) -> None:
        """Nodes belonging to any valid PARA category are accepted."""
        categories = ["Projects", "Areas", "Resources", "Archives"]
        nodes = []
        for idx, cat in enumerate(categories):
            nodes.append(
                {
                    "id": f"node-{idx}",
                    "label": f"Note {cat}",
                    "para": cat,
                    "tags": [cat.lower()],
                    "summary": f"Summary for {cat}",
                    "content_preview": f"Preview for {cat}",
                    "group": cat,
                }
            )

        data = {
            "nodes": nodes,
            "edges": [],
            "metadata": {
                "generated_at": "2026-11-09T00:00:00Z",
                "node_count": 4,
                "edge_count": 0,
            },
        }
        report = vg.validate_graph_dict(data)
        self.assertTrue(report.valid)
        self.assertTrue(report.nodes_valid)

    def test_both_edge_types_accepted(self) -> None:
        """Both 'semantic' and 'explicit' edge types are accepted."""
        data = _create_minimal_valid_graph_dict()
        # Add a 3rd node and an explicit edge
        data["nodes"].append(
            {
                "id": "note-3",
                "label": "Note 3",
                "para": "Areas",
                "tags": [],
                "summary": "Third note",
                "content_preview": "Content preview 3",
                "group": "Areas",
            }
        )
        data["edges"].append(
            {
                "source": "note-2",
                "target": "note-3",
                "weight": 1.0,
                "type": "explicit",
            }
        )
        data["metadata"]["node_count"] = 3
        data["metadata"]["edge_count"] = 2

        report = vg.validate_graph_dict(data)
        self.assertTrue(report.valid)
        self.assertTrue(report.edges_valid)
        self.assertEqual(report.edge_count, 2)


class TestValidateGraphJSONParsing(unittest.TestCase):
    """Tests for JSON syntax and raw string parsing."""

    def test_corrupted_json_syntax(self) -> None:
        """Malformed JSON string fails parsing cleanly without crash."""
        bad_json = '{"nodes": [{"id": "incomplete"'
        report = vg.validate_graph_json_string(bad_json)

        self.assertFalse(report.valid)
        self.assertFalse(report.json_valid)
        self.assertGreater(report.error_count, 0)
        self.assertTrue(any("JSON parsing failed" in issue.message for issue in report.issues))

    def test_reject_nan_constant(self) -> None:
        """JSON with NaN token is rejected."""
        json_with_nan = """
        {
            "nodes": [],
            "edges": [],
            "metadata": {"generated_at": "now", "node_count": NaN, "edge_count": 0}
        }
        """
        report = vg.validate_graph_json_string(json_with_nan)
        self.assertFalse(report.valid)
        self.assertFalse(report.json_valid)

    def test_reject_infinity_constant(self) -> None:
        """JSON with Infinity token is rejected."""
        json_with_inf = """
        {
            "nodes": [{"id": "n1", "label": "L", "para": "Projects", "tags": [], "summary": "", "content_preview": "", "group": "Projects"}],
            "edges": [{"source": "n1", "target": "n1", "weight": Infinity, "type": "semantic"}],
            "metadata": {"generated_at": "now", "node_count": 1, "edge_count": 1}
        }
        """
        report = vg.validate_graph_json_string(json_with_inf)
        self.assertFalse(report.valid)
        self.assertFalse(report.json_valid)


class TestValidateGraphTopLevel(unittest.TestCase):
    """Tests for top-level schema validation."""

    def test_root_not_a_dict(self) -> None:
        """A root JSON array or primitive fails top-level validation."""
        report = vg.validate_graph_dict(["not", "a", "dict"])
        self.assertFalse(report.valid)
        self.assertFalse(report.top_level_valid)
        self.assertTrue(any("must be a JSON object" in i.message for i in report.issues))

    def test_missing_top_level_keys(self) -> None:
        """Missing 'nodes', 'edges', or 'metadata' fails."""
        data = {"nodes": [], "edges": []}  # missing metadata
        report = vg.validate_graph_dict(data)
        self.assertFalse(report.valid)
        self.assertFalse(report.top_level_valid)
        self.assertTrue(any("Missing required top-level keys" in i.message for i in report.issues))

    def test_nodes_not_a_list(self) -> None:
        """nodes must be a list."""
        data = {"nodes": {}, "edges": [], "metadata": {}}
        report = vg.validate_graph_dict(data)
        self.assertFalse(report.valid)
        self.assertTrue(any("'nodes' must be a JSON array" in i.message for i in report.issues))

    def test_edges_not_a_list(self) -> None:
        """edges must be a list."""
        data = {"nodes": [], "edges": "not-a-list", "metadata": {}}
        report = vg.validate_graph_dict(data)
        self.assertFalse(report.valid)
        self.assertTrue(any("'edges' must be a JSON array" in i.message for i in report.issues))

    def test_metadata_not_a_dict(self) -> None:
        """metadata must be a dictionary."""
        data = {"nodes": [], "edges": [], "metadata": []}
        report = vg.validate_graph_dict(data)
        self.assertFalse(report.valid)
        self.assertTrue(any("'metadata' must be a JSON object" in i.message for i in report.issues))


class TestValidateGraphMetadata(unittest.TestCase):
    """Tests for metadata validation."""

    def test_missing_metadata_keys(self) -> None:
        """Missing required metadata fields fails."""
        data = _create_minimal_valid_graph_dict()
        del data["metadata"]["generated_at"]
        report = vg.validate_graph_dict(data)
        self.assertFalse(report.valid)
        self.assertTrue(any("Missing required metadata fields" in i.message for i in report.issues))

    def test_node_count_mismatch(self) -> None:
        """metadata.node_count mismatch with actual node array length fails."""
        data = _create_minimal_valid_graph_dict()
        data["metadata"]["node_count"] = 99
        report = vg.validate_graph_dict(data)
        self.assertFalse(report.valid)
        self.assertFalse(report.consistency_valid)
        self.assertTrue(any("node_count" in i.path for i in report.issues))

    def test_edge_count_mismatch(self) -> None:
        """metadata.edge_count mismatch with actual edge array length fails."""
        data = _create_minimal_valid_graph_dict()
        data["metadata"]["edge_count"] = 5
        report = vg.validate_graph_dict(data)
        self.assertFalse(report.valid)
        self.assertFalse(report.consistency_valid)
        self.assertTrue(any("edge_count" in i.path for i in report.issues))

    def test_metadata_boolean_counts_rejected(self) -> None:
        """Boolean values for node_count/edge_count are rejected."""
        data = _create_minimal_valid_graph_dict()
        data["metadata"]["node_count"] = True
        report = vg.validate_graph_dict(data)
        self.assertFalse(report.valid)
        self.assertTrue(any("'node_count' must be an integer" in i.message for i in report.issues))


class TestValidateGraphNodes(unittest.TestCase):
    """Tests for node schema and uniqueness validation."""

    def test_missing_node_required_field(self) -> None:
        """Missing required field on a node fails."""
        data = _create_minimal_valid_graph_dict()
        del data["nodes"][0]["summary"]
        report = vg.validate_graph_dict(data)
        self.assertFalse(report.valid)
        self.assertFalse(report.nodes_valid)
        self.assertTrue(any("Node missing required fields" in i.message for i in report.issues))

    def test_node_blank_id(self) -> None:
        """Blank or whitespace-only node ID fails."""
        data = _create_minimal_valid_graph_dict()
        data["nodes"][0]["id"] = "   "
        report = vg.validate_graph_dict(data)
        self.assertFalse(report.valid)
        self.assertFalse(report.nodes_valid)
        self.assertTrue(any("Node 'id' must be a non-empty string" in i.message for i in report.issues))

    def test_node_non_string_id(self) -> None:
        """Non-string node ID fails."""
        data = _create_minimal_valid_graph_dict()
        data["nodes"][0]["id"] = 12345
        report = vg.validate_graph_dict(data)
        self.assertFalse(report.valid)
        self.assertFalse(report.nodes_valid)

    def test_duplicate_node_ids(self) -> None:
        """Duplicate node IDs across nodes fail."""
        data = _create_minimal_valid_graph_dict()
        data["nodes"][1]["id"] = "note-1"  # same as node 0
        report = vg.validate_graph_dict(data)
        self.assertFalse(report.valid)
        self.assertFalse(report.nodes_valid)
        self.assertFalse(report.consistency_valid)
        self.assertTrue(any("Duplicate node ID detected" in i.message for i in report.issues))

    def test_invalid_para_category(self) -> None:
        """Invalid PARA category string fails."""
        data = _create_minimal_valid_graph_dict()
        data["nodes"][0]["para"] = "Tasks"  # invalid
        report = vg.validate_graph_dict(data)
        self.assertFalse(report.valid)
        self.assertFalse(report.nodes_valid)
        self.assertTrue(any("must be one of" in i.message for i in report.issues))

    def test_tags_not_list(self) -> None:
        """Tags not being a list fails."""
        data = _create_minimal_valid_graph_dict()
        data["nodes"][0]["tags"] = "single-tag-string"
        report = vg.validate_graph_dict(data)
        self.assertFalse(report.valid)
        self.assertFalse(report.nodes_valid)
        self.assertTrue(any("must be a list of strings" in i.message for i in report.issues))

    def test_tags_with_non_string_element(self) -> None:
        """Tags list containing numbers or booleans fails."""
        data = _create_minimal_valid_graph_dict()
        data["nodes"][0]["tags"] = ["valid", 123]
        report = vg.validate_graph_dict(data)
        self.assertFalse(report.valid)
        self.assertFalse(report.nodes_valid)
        self.assertTrue(any("Tag must be a string" in i.message for i in report.issues))


class TestValidateGraphEdges(unittest.TestCase):
    """Tests for edge schema and graph invariants."""

    def test_missing_edge_required_field(self) -> None:
        """Missing required edge field fails."""
        data = _create_minimal_valid_graph_dict()
        del data["edges"][0]["weight"]
        report = vg.validate_graph_dict(data)
        self.assertFalse(report.valid)
        self.assertFalse(report.edges_valid)
        self.assertTrue(any("Edge missing required fields" in i.message for i in report.issues))

    def test_broken_edge_source(self) -> None:
        """Edge source referencing non-existent node ID fails referential integrity."""
        data = _create_minimal_valid_graph_dict()
        data["edges"][0]["source"] = "ghost-node"
        report = vg.validate_graph_dict(data)
        self.assertFalse(report.valid)
        self.assertFalse(report.edges_valid)
        self.assertFalse(report.consistency_valid)
        self.assertTrue(any("does not reference any existing node ID" in i.message for i in report.issues))

    def test_broken_edge_target(self) -> None:
        """Edge target referencing non-existent node ID fails referential integrity."""
        data = _create_minimal_valid_graph_dict()
        data["edges"][0]["target"] = "ghost-node"
        report = vg.validate_graph_dict(data)
        self.assertFalse(report.valid)
        self.assertFalse(report.edges_valid)
        self.assertFalse(report.consistency_valid)
        self.assertTrue(any("does not reference any existing node ID" in i.message for i in report.issues))

    def test_self_referencing_edge(self) -> None:
        """Edge connecting a node to itself fails."""
        data = _create_minimal_valid_graph_dict()
        data["edges"][0]["target"] = "note-1"  # source is also note-1
        report = vg.validate_graph_dict(data)
        self.assertFalse(report.valid)
        self.assertFalse(report.edges_valid)
        self.assertFalse(report.consistency_valid)
        self.assertTrue(any("Self-referencing edge detected" in i.message for i in report.issues))

    def test_duplicate_undirected_edges(self) -> None:
        """Duplicate edges between the same pair of nodes in reverse order fail."""
        data = _create_minimal_valid_graph_dict()
        # Add reverse edge
        data["edges"].append(
            {
                "source": "note-2",
                "target": "note-1",
                "weight": 0.85,
                "type": "semantic",
            }
        )
        data["metadata"]["edge_count"] = 2
        report = vg.validate_graph_dict(data)
        self.assertFalse(report.valid)
        self.assertFalse(report.edges_valid)
        self.assertFalse(report.consistency_valid)
        self.assertTrue(any("Duplicate edge detected" in i.message for i in report.issues))

    def test_invalid_edge_weight(self) -> None:
        """Non-numeric edge weight fails."""
        data = _create_minimal_valid_graph_dict()
        data["edges"][0]["weight"] = "heavy"
        report = vg.validate_graph_dict(data)
        self.assertFalse(report.valid)
        self.assertFalse(report.edges_valid)
        self.assertTrue(any("weight" in i.path for i in report.issues))

    def test_boolean_edge_weight_rejected(self) -> None:
        """Boolean True/False is rejected as weight."""
        data = _create_minimal_valid_graph_dict()
        data["edges"][0]["weight"] = True
        report = vg.validate_graph_dict(data)
        self.assertFalse(report.valid)
        self.assertFalse(report.edges_valid)

    def test_invalid_edge_type(self) -> None:
        """Edge type other than 'semantic' or 'explicit' fails."""
        data = _create_minimal_valid_graph_dict()
        data["edges"][0]["type"] = "association"
        report = vg.validate_graph_dict(data)
        self.assertFalse(report.valid)
        self.assertFalse(report.edges_valid)
        self.assertTrue(any("Edge 'type' must be one of" in i.message for i in report.issues))


class TestValidateGraphFileAndCLI(unittest.TestCase):
    """Tests for file loading and CLI entrypoint."""

    def test_nonexistent_file(self) -> None:
        """Attempting to validate a missing file fails cleanly."""
        missing_path = Path("/tmp/does_not_exist_graph_12345.json")
        report = vg.validate_graph_file(missing_path)
        self.assertFalse(report.valid)
        self.assertFalse(report.json_valid)
        self.assertTrue(any("does not exist" in i.message for i in report.issues))

    def test_validate_graph_file_success(self) -> None:
        """Validating a valid graph file on disk succeeds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "graph.json"
            data = _create_minimal_valid_graph_dict()
            file_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

            report = vg.validate_graph_file(file_path)
            self.assertTrue(report.valid)
            self.assertEqual(report.node_count, 2)
            self.assertEqual(report.edge_count, 1)

    def test_main_cli_success(self) -> None:
        """CLI main returns 0 for a valid graph."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "graph.json"
            data = _create_minimal_valid_graph_dict()
            file_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

            buf = io.StringIO()
            with redirect_stdout(buf):
                exit_code = vg.main([str(file_path)])

            self.assertEqual(exit_code, 0)
            self.assertIn("SecondSelf Graph Schema Validation: PASS", buf.getvalue())

    def test_main_cli_failure(self) -> None:
        """CLI main returns 1 for an invalid graph."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "bad_graph.json"
            data = _create_minimal_valid_graph_dict()
            data["edges"][0]["target"] = "missing-target"
            file_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

            buf = io.StringIO()
            with redirect_stdout(buf):
                exit_code = vg.main([str(file_path)])

            self.assertEqual(exit_code, 1)
            self.assertIn("SecondSelf Graph Schema Validation: FAIL", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
