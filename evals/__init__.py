"""SecondSelf — Verification & Quality Evaluation Framework for Phase 2.

Exposes pure mathematical metric calculators, failure analysis engines,
verification harness orchestrators, and dual-format report serializers.
"""

from evals.failures import (
    CaseStudy,
    FailureAnalyzer,
    GraphAnomaly,
    MisclassificationFailure,
)
from evals.harness import VerificationHarness
from evals.metrics import (
    CategoryMetric,
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

__all__ = [
    "CategoryMetric",
    "ClassificationMetrics",
    "DistributionMetrics",
    "GraphTopologyMetrics",
    "RobustnessMetrics",
    "compute_classification_metrics",
    "compute_confusion_matrix",
    "compute_distribution",
    "compute_graph_topology_metrics",
    "CaseStudy",
    "FailureAnalyzer",
    "GraphAnomaly",
    "MisclassificationFailure",
    "ExecutionProfile",
    "SanityCheckResult",
    "VerificationReport",
    "generate_markdown_summary",
    "VerificationHarness",
]
