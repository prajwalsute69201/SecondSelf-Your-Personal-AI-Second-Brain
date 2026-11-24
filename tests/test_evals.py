"""SecondSelf — Unit Test Suite for evals Framework (Task 2.3).

Validates pure mathematical metric calculators, failure analyzers,
Markdown and JSON report serializers, and isolated harness execution.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from evals.failures import FailureAnalyzer, GraphAnomaly, MisclassificationFailure
from evals.harness import VerificationHarness
from evals.metrics import (
    ClassificationMetrics,
    DistributionMetrics,
    GraphTopologyMetrics,
    RobustnessMetrics,
    compute_classification_metrics,
    compute_confusion_matrix,
    compute_distribution,
    compute_graph_topology_metrics,
)
from evals.report import (
    ExecutionProfile,
    SanityCheckResult,
    VerificationReport,
    generate_markdown_summary,
)


# ==============================================================================
# 1. Pure Mathematical Metric Computation Tests
# ==============================================================================

def test_compute_distribution_known_values() -> None:
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    dist = compute_distribution(values)

    assert dist.count == 5
    assert dist.min_value == 10.0
    assert dist.max_value == 50.0
    assert dist.mean == 30.0
    assert dist.median == 30.0
    assert dist.p25 == 20.0
    assert dist.p75 == 40.0
    assert dist.std_dev > 0.0


def test_compute_distribution_empty() -> None:
    dist = compute_distribution([])
    assert dist.count == 0
    assert dist.mean == 0.0
    assert dist.median == 0.0


def test_compute_confusion_matrix_n_by_n() -> None:
    y_true = ["Projects", "Projects", "Areas", "Resources", "Archives"]
    y_pred = ["Projects", "Areas", "Areas", "Resources", "Archives"]
    categories = ["Projects", "Areas", "Resources", "Archives"]

    matrix = compute_confusion_matrix(y_true, y_pred, categories)

    assert matrix["Projects"]["Projects"] == 1
    assert matrix["Projects"]["Areas"] == 1
    assert matrix["Projects"]["Resources"] == 0
    assert matrix["Areas"]["Areas"] == 1
    assert matrix["Resources"]["Resources"] == 1
    assert matrix["Archives"]["Archives"] == 1


def test_compute_classification_metrics_perfect() -> None:
    y_true = ["Projects", "Areas", "Resources", "Archives"]
    y_pred = ["Projects", "Areas", "Resources", "Archives"]
    categories = ["Projects", "Areas", "Resources", "Archives"]

    res = compute_classification_metrics(y_true, y_pred, categories)

    assert res.total_samples == 4
    assert res.accuracy == 1.0
    assert res.macro_precision == 1.0
    assert res.macro_recall == 1.0
    assert res.macro_f1 == 1.0
    assert res.weighted_f1 == 1.0
    assert all(m.f1_score == 1.0 for m in res.per_category.values())


def test_compute_classification_metrics_with_misclassifications() -> None:
    # 2 Projects (1 correct, 1 predicted as Resource)
    # 1 Area (correct)
    # 1 Resource (correct)
    y_true = ["Projects", "Projects", "Areas", "Resources"]
    y_pred = ["Projects", "Resources", "Areas", "Resources"]
    categories = ["Projects", "Areas", "Resources", "Archives"]

    res = compute_classification_metrics(y_true, y_pred, categories)

    assert res.total_samples == 4
    assert res.accuracy == 0.75  # 3 of 4 correct

    # Project: TP=1, FP=0, FN=1 -> precision=1.0, recall=0.5, f1=2/3 = 0.6667
    proj = res.per_category["Projects"]
    assert proj.true_positive == 1
    assert proj.false_negative == 1
    assert proj.precision == 1.0
    assert proj.recall == 0.5
    assert pytest.approx(proj.f1_score, abs=1e-3) == 0.6667

    # Resource: TP=1, FP=1, FN=0 -> precision=0.5, recall=1.0, f1=0.6667
    res_cat = res.per_category["Resources"]
    assert res_cat.true_positive == 1
    assert res_cat.false_positive == 1
    assert res_cat.precision == 0.5
    assert res_cat.recall == 1.0


def test_compute_graph_topology_metrics_structure() -> None:
    # 3 nodes: A, B, C. Edges: A <-> B (weight 0.8), node C is island
    node_ids = ["A", "B", "C"]
    edges = [("A", "B", 0.8), ("B", "A", 0.8)]
    expected = {"A": {"B"}, "B": {"A"}}

    topo = compute_graph_topology_metrics(
        node_ids=node_ids,
        edges=edges,
        expected_links=expected,
        k_precision=5,
    )

    assert topo.total_nodes == 3
    assert topo.total_edges == 2
    # avg_degree = 2 * 2 / 3 = 1.33
    assert pytest.approx(topo.avg_degree, abs=0.1) == 1.33
    assert topo.island_nodes_count == 1  # Node C has 0 outgoing links
    assert pytest.approx(topo.island_ratio, abs=0.01) == 0.3333
    assert topo.link_precision_at_k == 1.0
    assert topo.score_distribution.mean == 0.8


# ==============================================================================
# 2. Failure Analysis & Anomaly Detection Tests
# ==============================================================================

def test_extract_misclassifications() -> None:
    items = [
        {
            "id": "1",
            "title": "Doc 1",
            "category": "Resources",
            "expected_category": "Projects",
            "confidence": 0.55,
            "needs_review": True,
            "rationale": "Weak signals",
            "signals": [{"description": "Found reference"}],
        },
        {
            "id": "2",
            "title": "Doc 2",
            "category": "Areas",
            "expected_category": "Areas",
            "confidence": 0.90,
            "needs_review": False,
            "rationale": "Ongoing task",
            "signals": [],
        },
    ]

    misclass = FailureAnalyzer.extract_misclassifications(items)
    assert len(misclass) == 1
    assert misclass[0].item_id == "1"
    assert misclass[0].expected_category == "Projects"
    assert misclass[0].predicted_category == "Resources"
    assert "Weak/ambiguous signals" in misclass[0].root_cause


def test_detect_graph_anomalies_self_loops_and_dangling() -> None:
    all_nodes = {"A", "B", "C"}
    edges = [
        ("A", "A", 0.95),          # Self-loop!
        ("A", "D", 0.80),          # Dangling target! (D is not in nodes)
        ("A", "B", 0.45),          # Sub-threshold! (0.45 < 0.65)
    ]

    anomalies = FailureAnalyzer.detect_graph_anomalies(all_nodes, edges, min_threshold=0.65)

    types = [a.anomaly_type for a in anomalies]
    assert "self_loop" in types
    assert "dangling_target" in types
    assert "sub_threshold_link" in types


def test_extract_representative_case_studies() -> None:
    items = [
        {
            "id": "p1",
            "title": "Active Sprint",
            "category": "Projects",
            "expected_category": "Projects",
            "confidence": 0.95,
            "needs_review": False,
            "rationale": "Deadline next week",
            "signals": [{"description": "deadline"}],
        },
        {
            "id": "a1",
            "title": "SOP Protocol",
            "category": "Areas",
            "expected_category": "Areas",
            "confidence": 0.85,
            "needs_review": False,
            "rationale": "Routine upkeep",
            "signals": [{"description": "recurring"}],
        },
        {
            "id": "r1",
            "title": "API Docs",
            "category": "Resources",
            "expected_category": "Resources",
            "confidence": 0.75,
            "needs_review": False,
            "rationale": "Topic guide",
            "signals": [],
        },
        {
            "id": "ar1",
            "title": "Completed Project",
            "category": "Archives",
            "expected_category": "Archives",
            "confidence": 0.90,
            "needs_review": False,
            "rationale": "Status DONE",
            "signals": [{"description": "done"}],
        },
        {
            "id": "rev1",
            "title": "Draft Note",
            "category": "Resources",
            "confidence": 0.40,
            "needs_review": True,
            "rationale": "Ambiguous",
            "signals": [],
        },
    ]

    cases = FailureAnalyzer.extract_representative_case_studies(items)
    assert len(cases) == 5
    scenarios = [c.scenario_type for c in cases]
    assert "clear_success" in scenarios
    assert "stale_archive" in scenarios
    assert "ambiguous_review" in scenarios


# ==============================================================================
# 3. Report Generation & Serialization Tests
# ==============================================================================

def test_report_json_and_markdown_serialization() -> None:
    profile = ExecutionProfile(
        dataset_name="test_dataset.json",
        dataset_hash="abcdef1234567890",
        sample_size=10,
        total_items_processed=10,
        duration_seconds=1.23,
        classifier_model="Hybrid",
        embedding_model="all-MiniLM-L6-v2",
        seed=42,
        timestamp="2026-11-23T12:00:00Z",
    )
    class_metrics = ClassificationMetrics(
        total_samples=10,
        accuracy=0.90,
        macro_f1=0.88,
    )
    topo_metrics = GraphTopologyMetrics(
        total_nodes=10,
        total_edges=8,
        avg_degree=1.6,
    )
    robust_metrics = RobustnessMetrics(
        total_items=10,
        determinism_score=1.0,
    )
    sanity = [
        SanityCheckResult(check_name="Item Coverage", passed=True, details="All items covered"),
    ]

    report = VerificationReport(
        profile=profile,
        classification_metrics=class_metrics,
        graph_topology_metrics=topo_metrics,
        robustness_metrics=robust_metrics,
        sanity_checks=sanity,
        overall_pass=True,
    )

    # Test JSON serialization
    json_str = report.to_json()
    parsed = json.loads(json_str)
    assert parsed["profile"]["sample_size"] == 10
    assert parsed["overall_pass"] is True

    # Test Markdown generation
    md_str = report.to_markdown()
    assert "# SecondSelf — Phase 2 Quality Verification Report" in md_str
    assert "🟢 PASS" in md_str
    assert "90.0%" in md_str
    assert "Item Coverage" in md_str


# ==============================================================================
# 4. End-to-End Verification Harness Tests
# ==============================================================================

def test_verification_harness_runs_on_golden_sample(tmp_path: Path) -> None:
    harness = VerificationHarness()
    dataset_file = Path("evals/datasets/para_golden.json")

    # Run verification with a sample of 5 items
    report = harness.run_verification(
        dataset_path=dataset_file,
        sample_size=5,
        seed=42,
    )

    assert report.profile.total_items_processed == 5
    assert len(report.sanity_checks) >= 4
    assert all(chk.passed for chk in report.sanity_checks)
    assert report.robustness_metrics.determinism_score == 1.0
    assert report.classification_metrics.total_samples == 5
