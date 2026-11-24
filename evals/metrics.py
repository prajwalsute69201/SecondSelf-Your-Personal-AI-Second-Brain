"""SecondSelf — Pure Mathematical Metric Computation Module for Phase 2 Verification.

Computes classification metrics (Accuracy, Macro-F1, Confusion Matrix, Precision/Recall),
graph topology metrics (density, average degree, island ratio, Precision@K),
and robustness/confidence distributions. Operates strictly in-memory without I/O.
"""

from __future__ import annotations

import math
from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class CategoryMetric(BaseModel):
    """Detailed evaluation statistics for a single PARA category."""

    category: str
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    support: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0

    model_config = ConfigDict(populate_by_name=True)


class ClassificationMetrics(BaseModel):
    """Multi-class classification evaluation results."""

    total_samples: int = 0
    accuracy: float = 0.0
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    macro_f1: float = 0.0
    weighted_f1: float = 0.0
    per_category: dict[str, CategoryMetric] = Field(default_factory=dict)
    confusion_matrix: dict[str, dict[str, int]] = Field(default_factory=dict)

    model_config = ConfigDict(populate_by_name=True)


class DistributionMetrics(BaseModel):
    """Summary statistical distribution for numerical confidence or similarity scores."""

    count: int = 0
    mean: float = 0.0
    std_dev: float = 0.0
    min_value: float = 0.0
    max_value: float = 0.0
    median: float = 0.0
    p25: float = 0.0
    p75: float = 0.0

    model_config = ConfigDict(populate_by_name=True)


class GraphTopologyMetrics(BaseModel):
    """Structural health and linkage metrics of the semantic knowledge graph."""

    total_nodes: int = 0
    total_edges: int = 0
    avg_degree: float = 0.0
    max_degree: int = 0
    min_degree: int = 0
    island_nodes_count: int = 0
    island_ratio: float = 0.0
    graph_density: float = 0.0
    link_precision_at_k: float = 0.0
    score_distribution: DistributionMetrics = Field(default_factory=DistributionMetrics)

    model_config = ConfigDict(populate_by_name=True)


class RobustnessMetrics(BaseModel):
    """Robustness, fallback, and error resilience metrics."""

    total_items: int = 0
    failed_embeddings_count: int = 0
    failed_embeddings_ratio: float = 0.0
    low_confidence_count: int = 0
    low_confidence_ratio: float = 0.0
    needs_review_count: int = 0
    needs_review_ratio: float = 0.0
    error_count: int = 0
    determinism_score: float = 1.0  # 1.0 = 100% reproducible

    model_config = ConfigDict(populate_by_name=True)


# ==============================================================================
# Pure Math Calculation Functions
# ==============================================================================

def compute_distribution(values: list[float]) -> DistributionMetrics:
    """Computes summary statistics (mean, std, quantiles) for a list of floats."""
    if not values:
        return DistributionMetrics()

    n = len(values)
    mean_val = sum(values) / n
    variance = sum((x - mean_val) ** 2 for x in values) / n if n > 1 else 0.0
    std_val = math.sqrt(variance)

    sorted_vals = sorted(values)
    min_val = sorted_vals[0]
    max_val = sorted_vals[-1]

    def quantile(p: float) -> float:
        idx = int(round(p * (n - 1)))
        return sorted_vals[max(0, min(n - 1, idx))]

    return DistributionMetrics(
        count=n,
        mean=round(mean_val, 4),
        std_dev=round(std_val, 4),
        min_value=round(min_val, 4),
        max_value=round(max_val, 4),
        median=round(quantile(0.50), 4),
        p25=round(quantile(0.25), 4),
        p75=round(quantile(0.75), 4),
    )


def compute_confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    categories: list[str],
) -> dict[str, dict[str, int]]:
    """Generates an N x N confusion matrix dictionary: matrix[true_cat][pred_cat] = count."""
    matrix = {c_true: {c_pred: 0 for c_pred in categories} for c_true in categories}
    for yt, yp in zip(y_true, y_pred):
        if yt in matrix and yp in matrix[yt]:
            matrix[yt][yp] += 1
        elif yt in matrix:
            # Fallback bucket if predicted an unexpected category
            matrix[yt].setdefault(yp, 0)
            matrix[yt][yp] += 1
    return matrix


def compute_classification_metrics(
    y_true: list[str],
    y_pred: list[str],
    categories: list[str],
) -> ClassificationMetrics:
    """Calculates multi-class Accuracy, Per-Category Precision/Recall/F1, and Macro-F1."""
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return ClassificationMetrics()

    n = len(y_true)
    matrix = compute_confusion_matrix(y_true, y_pred, categories)

    per_category: dict[str, CategoryMetric] = {}
    total_correct = 0

    for cat in categories:
        tp = matrix.get(cat, {}).get(cat, 0)
        total_correct += tp

        # False positives: others predicted as cat
        fp = sum(matrix.get(other_true, {}).get(cat, 0) for other_true in categories if other_true != cat)
        # False negatives: cat predicted as other
        fn = sum(matrix.get(cat, {}).get(other_pred, 0) for other_pred in matrix.get(cat, {}) if other_pred != cat)
        support = sum(matrix.get(cat, {}).values())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        per_category[cat] = CategoryMetric(
            category=cat,
            true_positive=tp,
            false_positive=fp,
            false_negative=fn,
            support=support,
            precision=round(prec, 4),
            recall=round(rec, 4),
            f1_score=round(f1, 4),
        )

    accuracy = total_correct / n if n > 0 else 0.0
    macro_prec = sum(m.precision for m in per_category.values()) / len(categories) if categories else 0.0
    macro_rec = sum(m.recall for m in per_category.values()) / len(categories) if categories else 0.0
    macro_f1 = sum(m.f1_score for m in per_category.values()) / len(categories) if categories else 0.0

    # Weighted F1 based on support
    total_support = sum(m.support for m in per_category.values())
    weighted_f1 = (
        sum(m.f1_score * m.support for m in per_category.values()) / total_support
        if total_support > 0
        else 0.0
    )

    return ClassificationMetrics(
        total_samples=n,
        accuracy=round(accuracy, 4),
        macro_precision=round(macro_prec, 4),
        macro_recall=round(macro_rec, 4),
        macro_f1=round(macro_f1, 4),
        weighted_f1=round(weighted_f1, 4),
        per_category=per_category,
        confusion_matrix=matrix,
    )


def compute_graph_topology_metrics(
    node_ids: list[str],
    edges: list[tuple[str, str, float]],
    expected_links: dict[str, set[str]] | None = None,
    k_precision: int = 5,
) -> GraphTopologyMetrics:
    """Computes structural metrics for the generated semantic graph.

    Args:
        node_ids: All active node identifiers.
        edges: List of (source_id, target_id, similarity_score).
        expected_links: Optional ground-truth dict mapping note_id -> set of expected target IDs.
        k_precision: Cutoff for link precision@K.

    Returns:
        GraphTopologyMetrics model instance.
    """
    total_nodes = len(node_ids)
    if total_nodes == 0:
        return GraphTopologyMetrics()

    # Track degree per node
    degrees: dict[str, int] = {nid: 0 for nid in node_ids}
    scores: list[float] = []

    for src, tgt, score in edges:
        if src in degrees:
            degrees[src] += 1
        scores.append(score)

    total_edges = len(edges)
    avg_degree = (2.0 * total_edges) / total_nodes if total_nodes > 0 else 0.0
    max_deg = max(degrees.values()) if degrees else 0
    min_deg = min(degrees.values()) if degrees else 0

    island_count = sum(1 for deg in degrees.values() if deg == 0)
    island_ratio = island_count / total_nodes if total_nodes > 0 else 0.0

    # Graph density: D = 2|E| / (|V|(|V|-1))
    if total_nodes > 1:
        max_possible_edges = total_nodes * (total_nodes - 1)
        density = (2.0 * total_edges) / max_possible_edges
    else:
        density = 0.0

    # Link precision@K evaluation if ground-truth benchmark links are provided
    link_prec = 0.0
    if expected_links:
        precisions: list[float] = []
        # Group edges by source
        src_to_targets: dict[str, list[str]] = {}
        for src, tgt, _ in edges:
            src_to_targets.setdefault(src, []).append(tgt)

        for src_id, exp_set in expected_links.items():
            if not exp_set:
                continue
            actual_top_k = src_to_targets.get(src_id, [])[:k_precision]
            if actual_top_k:
                hits = sum(1 for tgt in actual_top_k if tgt in exp_set)
                precisions.append(hits / len(actual_top_k))

        if precisions:
            link_prec = sum(precisions) / len(precisions)

    score_dist = compute_distribution(scores)

    return GraphTopologyMetrics(
        total_nodes=total_nodes,
        total_edges=total_edges,
        avg_degree=round(avg_degree, 2),
        max_degree=max_deg,
        min_degree=min_deg,
        island_nodes_count=island_count,
        island_ratio=round(island_ratio, 4),
        graph_density=round(density, 4),
        link_precision_at_k=round(link_prec, 4),
        score_distribution=score_dist,
    )
