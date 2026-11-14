"""Schema and consistency validator for SecondSelf knowledge graph (Task 3.1.2).

Validates that `data/graph.json` conforms to the schema defined in
`Implementationplan.md` Task 3.1.2 and the data models in `lib/models.py`.

Checks:
- Valid UTF-8 JSON encoding without non-standard NaN or Infinity tokens.
- Top-level structure: 'nodes', 'edges', 'metadata'.
- Node attributes: 'id', 'label', 'para', 'tags', 'summary', 'content_preview', 'group'.
- Edge attributes: 'source', 'target', 'weight', 'type'.
- Metadata attributes: 'generated_at', 'node_count', 'edge_count'.
- Referential integrity: all edge endpoints must reference valid, existing nodes.
- Graph invariants: no self-referencing edges, no duplicate edges, no duplicate node IDs.
- Metadata count consistency: node_count and edge_count must match array lengths.

Usage:
    python validate_graph.py                    # Validate data/graph.json
    python validate_graph.py path/to/graph.json # Validate a specific graph file
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Sequence

# Ensure repository root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import lib.storage as storage

try:
    from lib.constants import VALID_CATEGORIES
except ImportError:
    from .constants import VALID_CATEGORIES

VALID_PARA_CATEGORIES: set[str] = set(VALID_CATEGORIES)
VALID_CATEGORIES_SET: set[str] = set(VALID_CATEGORIES)
VALID_EDGE_TYPES: set[str] = {"semantic", "explicit"}

REQUIRED_TOP_LEVEL_KEYS: set[str] = {"nodes", "edges", "metadata"}
REQUIRED_NODE_KEYS: set[str] = {
    "id",
    "label",
    "para",
    "tags",
    "summary",
    "content_preview",
    "group",
}
REQUIRED_EDGE_KEYS: set[str] = {"source", "target", "weight", "type"}
REQUIRED_METADATA_KEYS: set[str] = {"generated_at", "node_count", "edge_count"}

__all__ = [
    "VALID_PARA_CATEGORIES",
    "VALID_EDGE_TYPES",
    "ValidationIssue",
    "ValidationReport",
    "validate_graph_dict",
    "validate_graph_json_string",
    "validate_graph_file",
    "main",
]


@dataclass
class ValidationIssue:
    """Individual issue found during graph validation.

    Attributes:
        severity: 'ERROR' for schema or invariant violations, 'WARNING' for non-fatal concerns.
        path: Path to the affected element (e.g. 'nodes[0].para', 'edges[2]', 'metadata.node_count').
        message: Detailed explanation of the validation failure.
    """

    severity: str
    path: str
    message: str


@dataclass
class ValidationReport:
    """Consolidated summary of graph validation results.

    Attributes:
        valid: True if there are zero ERROR-level issues.
        json_valid: True if JSON parsed successfully without syntax errors.
        top_level_valid: True if top-level structure and keys are correct.
        node_count: Number of nodes present in the graph.
        edge_count: Number of edges present in the graph.
        nodes_valid: True if all nodes conform to schema.
        edges_valid: True if all edges conform to schema.
        consistency_valid: True if graph-wide invariants (referential integrity, counts) hold.
        issues: List of ValidationIssue records describing all detected problems.
    """

    valid: bool = True
    json_valid: bool = True
    top_level_valid: bool = True
    node_count: int = 0
    edge_count: int = 0
    nodes_valid: bool = True
    edges_valid: bool = True
    consistency_valid: bool = True
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        """Count of ERROR-level issues."""
        return sum(1 for issue in self.issues if issue.severity == "ERROR")

    @property
    def warning_count(self) -> int:
        """Count of WARNING-level issues."""
        return sum(1 for issue in self.issues if issue.severity == "WARNING")


def _reject_constant(val: str) -> None:
    """Callback for json.loads to reject NaN and Infinity constants."""
    raise ValueError(f"Invalid non-standard JSON numerical constant: {val!r}")


def validate_graph_dict(data: Any) -> ValidationReport:
    """Validate a Python dictionary representation of graph.json.

    Args:
        data: Deserialized JSON data to validate.

    Returns:
        ValidationReport: Validation findings and overall status.
    """
    report = ValidationReport(valid=True, json_valid=True)

    # 1. Top-level structure validation
    if not isinstance(data, dict):
        report.valid = False
        report.top_level_valid = False
        report.issues.append(
            ValidationIssue(
                severity="ERROR",
                path="$",
                message=f"Root structure must be a JSON object, got {type(data).__name__}",
            )
        )
        return report

    missing_top_keys = REQUIRED_TOP_LEVEL_KEYS - set(data.keys())
    if missing_top_keys:
        report.valid = False
        report.top_level_valid = False
        report.issues.append(
            ValidationIssue(
                severity="ERROR",
                path="$",
                message=f"Missing required top-level keys: {sorted(missing_top_keys)}",
            )
        )

    # Validate top-level types
    nodes_raw = data.get("nodes")
    edges_raw = data.get("edges")
    metadata_raw = data.get("metadata")

    if nodes_raw is not None and not isinstance(nodes_raw, list):
        report.valid = False
        report.top_level_valid = False
        report.issues.append(
            ValidationIssue(
                severity="ERROR",
                path="nodes",
                message=f"'nodes' must be a JSON array, got {type(nodes_raw).__name__}",
            )
        )

    if edges_raw is not None and not isinstance(edges_raw, list):
        report.valid = False
        report.top_level_valid = False
        report.issues.append(
            ValidationIssue(
                severity="ERROR",
                path="edges",
                message=f"'edges' must be a JSON array, got {type(edges_raw).__name__}",
            )
        )

    if metadata_raw is not None and not isinstance(metadata_raw, dict):
        report.valid = False
        report.top_level_valid = False
        report.issues.append(
            ValidationIssue(
                severity="ERROR",
                path="metadata",
                message=f"'metadata' must be a JSON object, got {type(metadata_raw).__name__}",
            )
        )

    # Stop early if top-level types are fundamentally invalid
    if not report.top_level_valid:
        return report

    nodes_list = nodes_raw if isinstance(nodes_raw, list) else []
    edges_list = edges_raw if isinstance(edges_raw, list) else []
    metadata_dict = metadata_raw if isinstance(metadata_raw, dict) else {}

    report.node_count = len(nodes_list)
    report.edge_count = len(edges_list)

    # 2. Metadata Validation
    if isinstance(metadata_raw, dict):
        missing_meta = REQUIRED_METADATA_KEYS - set(metadata_dict.keys())
        if missing_meta:
            report.valid = False
            report.issues.append(
                ValidationIssue(
                    severity="ERROR",
                    path="metadata",
                    message=f"Missing required metadata fields: {sorted(missing_meta)}",
                )
            )

        gen_at = metadata_dict.get("generated_at")
        if gen_at is not None and not isinstance(gen_at, str):
            report.valid = False
            report.issues.append(
                ValidationIssue(
                    severity="ERROR",
                    path="metadata.generated_at",
                    message=f"'generated_at' must be a string, got {type(gen_at).__name__}",
                )
            )

        meta_node_count = metadata_dict.get("node_count")
        if meta_node_count is not None:
            if not isinstance(meta_node_count, int) or isinstance(meta_node_count, bool):
                report.valid = False
                report.issues.append(
                    ValidationIssue(
                        severity="ERROR",
                        path="metadata.node_count",
                        message=f"'node_count' must be an integer, got {type(meta_node_count).__name__}",
                    )
                )
            elif meta_node_count != len(nodes_list):
                report.valid = False
                report.consistency_valid = False
                report.issues.append(
                    ValidationIssue(
                        severity="ERROR",
                        path="metadata.node_count",
                        message=(
                            f"Metadata node_count ({meta_node_count}) does not match "
                            f"actual nodes array length ({len(nodes_list)})"
                        ),
                    )
                )

        meta_edge_count = metadata_dict.get("edge_count")
        if meta_edge_count is not None:
            if not isinstance(meta_edge_count, int) or isinstance(meta_edge_count, bool):
                report.valid = False
                report.issues.append(
                    ValidationIssue(
                        severity="ERROR",
                        path="metadata.edge_count",
                        message=f"'edge_count' must be an integer, got {type(meta_edge_count).__name__}",
                    )
                )
            elif meta_edge_count != len(edges_list):
                report.valid = False
                report.consistency_valid = False
                report.issues.append(
                    ValidationIssue(
                        severity="ERROR",
                        path="metadata.edge_count",
                        message=(
                            f"Metadata edge_count ({meta_edge_count}) does not match "
                            f"actual edges array length ({len(edges_list)})"
                        ),
                    )
                )

    # 3. Node Validation
    known_node_ids: set[str] = set()

    for idx, node in enumerate(nodes_list):
        path_prefix = f"nodes[{idx}]"
        if not isinstance(node, dict):
            report.valid = False
            report.nodes_valid = False
            report.issues.append(
                ValidationIssue(
                    severity="ERROR",
                    path=path_prefix,
                    message=f"Node must be a JSON object, got {type(node).__name__}",
                )
            )
            continue

        missing_fields = REQUIRED_NODE_KEYS - set(node.keys())
        if missing_fields:
            report.valid = False
            report.nodes_valid = False
            report.issues.append(
                ValidationIssue(
                    severity="ERROR",
                    path=path_prefix,
                    message=f"Node missing required fields: {sorted(missing_fields)}",
                )
            )

        # Validate node 'id'
        nid = node.get("id")
        if nid is None or not isinstance(nid, str) or not nid.strip():
            report.valid = False
            report.nodes_valid = False
            report.issues.append(
                ValidationIssue(
                    severity="ERROR",
                    path=f"{path_prefix}.id",
                    message=f"Node 'id' must be a non-empty string, got {nid!r}",
                )
            )
        else:
            clean_id = nid.strip()
            if clean_id in known_node_ids:
                report.valid = False
                report.nodes_valid = False
                report.consistency_valid = False
                report.issues.append(
                    ValidationIssue(
                        severity="ERROR",
                        path=f"{path_prefix}.id",
                        message=f"Duplicate node ID detected: '{clean_id}'",
                    )
                )
            else:
                known_node_ids.add(clean_id)

        # Validate node 'label'
        label = node.get("label")
        if label is not None and not isinstance(label, str):
            report.valid = False
            report.nodes_valid = False
            report.issues.append(
                ValidationIssue(
                    severity="ERROR",
                    path=f"{path_prefix}.label",
                    message=f"Node 'label' must be a string, got {type(label).__name__}",
                )
            )

        # Validate node 'para'
        para = node.get("para")
        if para is None or not isinstance(para, str) or para not in VALID_PARA_CATEGORIES:
            report.valid = False
            report.nodes_valid = False
            report.issues.append(
                ValidationIssue(
                    severity="ERROR",
                    path=f"{path_prefix}.para",
                    message=(
                        f"Node 'para' must be one of {sorted(VALID_PARA_CATEGORIES)}, got {para!r}"
                    ),
                )
            )

        # Validate node 'group'
        group = node.get("group")
        if group is not None and not isinstance(group, str):
            report.valid = False
            report.nodes_valid = False
            report.issues.append(
                ValidationIssue(
                    severity="ERROR",
                    path=f"{path_prefix}.group",
                    message=f"Node 'group' must be a string, got {type(group).__name__}",
                )
            )

        # Validate node 'tags'
        tags = node.get("tags")
        if tags is None or not isinstance(tags, list):
            report.valid = False
            report.nodes_valid = False
            report.issues.append(
                ValidationIssue(
                    severity="ERROR",
                    path=f"{path_prefix}.tags",
                    message=f"Node 'tags' must be a list of strings, got {type(tags).__name__}",
                )
            )
        else:
            for t_idx, tag_item in enumerate(tags):
                if not isinstance(tag_item, str):
                    report.valid = False
                    report.nodes_valid = False
                    report.issues.append(
                        ValidationIssue(
                            severity="ERROR",
                            path=f"{path_prefix}.tags[{t_idx}]",
                            message=f"Tag must be a string, got {type(tag_item).__name__}",
                        )
                    )

        # Validate node 'summary'
        summary = node.get("summary")
        if summary is not None and not isinstance(summary, str):
            report.valid = False
            report.nodes_valid = False
            report.issues.append(
                ValidationIssue(
                    severity="ERROR",
                    path=f"{path_prefix}.summary",
                    message=f"Node 'summary' must be a string, got {type(summary).__name__}",
                )
            )

        # Validate node 'content_preview'
        preview = node.get("content_preview")
        if preview is not None and not isinstance(preview, str):
            report.valid = False
            report.nodes_valid = False
            report.issues.append(
                ValidationIssue(
                    severity="ERROR",
                    path=f"{path_prefix}.content_preview",
                    message=f"Node 'content_preview' must be a string, got {type(preview).__name__}",
                )
            )

    # 4. Edge Validation
    seen_undirected_edges: set[tuple[str, str]] = set()

    for idx, edge in enumerate(edges_list):
        path_prefix = f"edges[{idx}]"
        if not isinstance(edge, dict):
            report.valid = False
            report.edges_valid = False
            report.issues.append(
                ValidationIssue(
                    severity="ERROR",
                    path=path_prefix,
                    message=f"Edge must be a JSON object, got {type(edge).__name__}",
                )
            )
            continue

        missing_fields = REQUIRED_EDGE_KEYS - set(edge.keys())
        if missing_fields:
            report.valid = False
            report.edges_valid = False
            report.issues.append(
                ValidationIssue(
                    severity="ERROR",
                    path=path_prefix,
                    message=f"Edge missing required fields: {sorted(missing_fields)}",
                )
            )

        # Validate endpoints
        src = edge.get("source")
        tgt = edge.get("target")

        src_valid = isinstance(src, str) and bool(src.strip())
        tgt_valid = isinstance(tgt, str) and bool(tgt.strip())

        if not src_valid:
            report.valid = False
            report.edges_valid = False
            report.issues.append(
                ValidationIssue(
                    severity="ERROR",
                    path=f"{path_prefix}.source",
                    message=f"Edge 'source' must be a non-empty string, got {src!r}",
                )
            )

        if not tgt_valid:
            report.valid = False
            report.edges_valid = False
            report.issues.append(
                ValidationIssue(
                    severity="ERROR",
                    path=f"{path_prefix}.target",
                    message=f"Edge 'target' must be a non-empty string, got {tgt!r}",
                )
            )

        # Referential integrity check (endpoints must exist in nodes)
        if src_valid and src not in known_node_ids:
            report.valid = False
            report.edges_valid = False
            report.consistency_valid = False
            report.issues.append(
                ValidationIssue(
                    severity="ERROR",
                    path=f"{path_prefix}.source",
                    message=f"Edge source '{src}' does not reference any existing node ID",
                )
            )

        if tgt_valid and tgt not in known_node_ids:
            report.valid = False
            report.edges_valid = False
            report.consistency_valid = False
            report.issues.append(
                ValidationIssue(
                    severity="ERROR",
                    path=f"{path_prefix}.target",
                    message=f"Edge target '{tgt}' does not reference any existing node ID",
                )
            )

        # Self-referencing edge check
        if src_valid and tgt_valid and src == tgt:
            report.valid = False
            report.edges_valid = False
            report.consistency_valid = False
            report.issues.append(
                ValidationIssue(
                    severity="ERROR",
                    path=path_prefix,
                    message=f"Self-referencing edge detected ({src} -> {tgt})",
                )
            )

        # Duplicate edge check (undirected pair key)
        if src_valid and tgt_valid and src != tgt:
            undirected_key = (min(src, tgt), max(src, tgt))
            if undirected_key in seen_undirected_edges:
                report.valid = False
                report.edges_valid = False
                report.consistency_valid = False
                report.issues.append(
                    ValidationIssue(
                        severity="ERROR",
                        path=path_prefix,
                        message=f"Duplicate edge detected between nodes '{src}' and '{tgt}'",
                    )
                )
            else:
                seen_undirected_edges.add(undirected_key)

        # Validate weight
        weight = edge.get("weight")
        if (
            weight is None
            or not isinstance(weight, (int, float))
            or isinstance(weight, bool)
            or not math.isfinite(weight)
        ):
            report.valid = False
            report.edges_valid = False
            report.issues.append(
                ValidationIssue(
                    severity="ERROR",
                    path=f"{path_prefix}.weight",
                    message=f"Edge 'weight' must be a finite numerical float or int, got {weight!r}",
                )
            )

        # Validate type
        edge_type = edge.get("type")
        if edge_type is None or not isinstance(edge_type, str) or edge_type not in VALID_EDGE_TYPES:
            report.valid = False
            report.edges_valid = False
            report.issues.append(
                ValidationIssue(
                    severity="ERROR",
                    path=f"{path_prefix}.type",
                    message=f"Edge 'type' must be one of {sorted(VALID_EDGE_TYPES)}, got {edge_type!r}",
                )
            )

    return report


def validate_graph_json_string(json_text: str) -> ValidationReport:
    """Validate graph from raw JSON string, strictly rejecting non-standard constants.

    Args:
        json_text: Raw string containing JSON content.

    Returns:
        ValidationReport: Validation findings and overall status.
    """
    report = ValidationReport(valid=True)

    try:
        data = json.loads(json_text, parse_constant=_reject_constant)
    except Exception as e:
        report.valid = False
        report.json_valid = False
        report.top_level_valid = False
        report.nodes_valid = False
        report.edges_valid = False
        report.consistency_valid = False
        report.issues.append(
            ValidationIssue(
                severity="ERROR",
                path="$",
                message=f"JSON parsing failed: {e}",
            )
        )
        return report

    return validate_graph_dict(data)


def validate_graph_file(path: Path | str) -> ValidationReport:
    """Validate a graph.json file on disk.

    Args:
        path: Path to the target graph.json file.

    Returns:
        ValidationReport: Validation findings.
    """
    target = Path(path)
    report = ValidationReport(valid=True)

    if not target.exists():
        report.valid = False
        report.json_valid = False
        report.top_level_valid = False
        report.issues.append(
            ValidationIssue(
                severity="ERROR",
                path=str(target),
                message=f"Graph file does not exist: {target}",
            )
        )
        return report

    try:
        text = target.read_text(encoding="utf-8")
    except Exception as e:
        report.valid = False
        report.json_valid = False
        report.top_level_valid = False
        report.issues.append(
            ValidationIssue(
                severity="ERROR",
                path=str(target),
                message=f"Failed to read file as UTF-8: {e}",
            )
        )
        return report

    return validate_graph_json_string(text)


def build_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser for validate_graph.py."""
    parser = argparse.ArgumentParser(
        prog="validate_graph.py",
        description="Validate data/graph.json against the SecondSelf schema specification.",
    )
    parser.add_argument(
        "graph_path",
        nargs="?",
        default=None,
        metavar="PATH",
        help="Path to graph.json (defaults to data/graph.json in repository).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for validate_graph.py.

    Args:
        argv: Command-line argument list.

    Returns:
        int: 0 if validation succeeds (PASS), 1 if validation fails (FAIL).
    """
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 0

    if args.graph_path:
        target_path = Path(args.graph_path).resolve()
    else:
        target_path = storage.get_data_dir() / "graph.json"

    print(f"Validating SecondSelf knowledge graph: {target_path}")
    report = validate_graph_file(target_path)

    status_str = "PASS" if report.valid else "FAIL"
    print("\n" + "=" * 60)
    print(f"SecondSelf Graph Schema Validation: {status_str}")
    print("=" * 60)
    print(f"JSON Syntax:         {'PASS' if report.json_valid else 'FAIL'}")
    print(f"Top-level Schema:    {'PASS' if report.top_level_valid else 'FAIL'}")
    print(f"Node Definitions:    {'PASS' if report.nodes_valid else 'FAIL'} ({report.node_count} nodes)")
    print(f"Edge Definitions:    {'PASS' if report.edges_valid else 'FAIL'} ({report.edge_count} edges)")
    print(f"Graph Consistency:   {'PASS' if report.consistency_valid else 'FAIL'}")
    print(f"Total Errors:        {report.error_count}")
    print(f"Total Warnings:      {report.warning_count}")
    print("=" * 60)

    if report.issues:
        print("\nValidation Issues Detected:")
        for issue in report.issues:
            print(f"  [{issue.severity}] {issue.path}: {issue.message}")

    return 0 if report.valid else 1


if __name__ == "__main__":
    sys.exit(main())

