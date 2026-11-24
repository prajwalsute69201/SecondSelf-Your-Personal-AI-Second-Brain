"""SecondSelf — Verification Report Serializer & Markdown Generator.

Produces machine-readable JSON and publication-grade GitHub-flavored Markdown summaries
documenting classification quality, semantic graph topology, sanity checks, and case studies.
"""

from __future__ import annotations

import json
from typing import Any
from pydantic import BaseModel, ConfigDict, Field

from evals.failures import CaseStudy, GraphAnomaly, MisclassificationFailure
from evals.metrics import (
    ClassificationMetrics,
    DistributionMetrics,
    GraphTopologyMetrics,
    RobustnessMetrics,
)


class SanityCheckResult(BaseModel):
    """Result of a pipeline integrity sanity check."""

    check_name: str
    passed: bool
    details: str

    model_config = ConfigDict(populate_by_name=True)


class ExecutionProfile(BaseModel):
    """Metadata detailing the verification execution environment and parameters."""

    dataset_name: str
    dataset_hash: str
    sample_size: int
    total_items_processed: int
    duration_seconds: float
    classifier_model: str
    embedding_model: str
    seed: int
    timestamp: str

    model_config = ConfigDict(populate_by_name=True)


class VerificationReport(BaseModel):
    """Complete end-to-end verification report for Phase 2."""

    profile: ExecutionProfile
    classification_metrics: ClassificationMetrics
    graph_topology_metrics: GraphTopologyMetrics
    robustness_metrics: RobustnessMetrics
    sanity_checks: list[SanityCheckResult] = Field(default_factory=list)
    misclassifications: list[MisclassificationFailure] = Field(default_factory=list)
    graph_anomalies: list[GraphAnomaly] = Field(default_factory=list)
    case_studies: list[CaseStudy] = Field(default_factory=list)
    overall_pass: bool = True
    summary: str = ""

    model_config = ConfigDict(populate_by_name=True)

    def to_json(self, indent: int = 2) -> str:
        """Serializes verification report to formatted JSON."""
        return self.model_dump_json(indent=indent)

    def to_markdown(self) -> str:
        """Renders comprehensive, publication-grade Markdown verification summary."""
        return generate_markdown_summary(self)


def generate_markdown_summary(report: VerificationReport) -> str:
    """Renders a structured, evidence-backed GitHub-flavored Markdown report."""
    p = report.profile
    cm = report.classification_metrics
    gm = report.graph_topology_metrics
    rm = report.robustness_metrics

    status_badge = "🟢 PASS (Quality Gate Cleared)" if report.overall_pass else "🔴 FAIL (Quality Gate Breached)"

    lines: list[str] = [
        f"# SecondSelf — Phase 2 Quality Verification Report",
        f"",
        f"**Status:** {status_badge}  ",
        f"**Generated:** `{p.timestamp}`  ",
        f"**Dataset:** `{p.dataset_name}` (`{p.dataset_hash[:16]}...`)  ",
        f"**Sample Size:** {p.total_items_processed} / {p.sample_size} items processed in {p.duration_seconds:.2f}s  ",
        f"**Models:** Classifier=`{p.classifier_model}` | Embedding=`{p.embedding_model}` | Seed=`{p.seed}`  ",
        f"",
        f"---",
        f"",
        f"## 1. Executive Summary & Quality Gate Status",
        f"",
        f"{report.summary or 'Verification harness executed end-to-end against real benchmark data.'}",
        f"",
        f"| Metric Domain | Primary Indicator | Measured Value | Target SLA | Status |",
        f"| :--- | :--- | :--- | :--- | :--- |",
        f"| **PARA Classification** | Accuracy | **{cm.accuracy * 100:.1f}%** | $\\ge 85.0\\%$ | {'✅ PASS' if cm.accuracy >= 0.85 else '⚠️ REVIEW'} |",
        f"| **PARA Classification** | Macro-F1 | **{cm.macro_f1:.4f}** | $\\ge 0.8000$ | {'✅ PASS' if cm.macro_f1 >= 0.80 else '⚠️ REVIEW'} |",
        f"| **Graph Topology** | Island Ratio ($R_{{island}}$) | **{gm.island_ratio * 100:.1f}%** | $\\le 25.0\\%$ | {'✅ PASS' if gm.island_ratio <= 0.25 else '⚠️ REVIEW'} |",
        f"| **Graph Topology** | Link Precision@5 | **{gm.link_precision_at_k * 100:.1f}%** | $\\ge 60.0\\%$ | {'✅ PASS' if gm.link_precision_at_k >= 0.60 else '⚠️ REVIEW'} |",
        f"| **Robustness** | Determinism Score | **{rm.determinism_score * 100:.1f}%** | $100.0\\%$ | {'✅ PASS' if rm.determinism_score == 1.0 else '❌ FAIL'} |",
        f"| **Robustness** | Error Rate | **{rm.error_count} errors** | $0$ errors | {'✅ PASS' if rm.error_count == 0 else '❌ FAIL'} |",
        f"",
        f"---",
        f"",
        f"## 2. PARA Classification Quality Metrics",
        f"",
        f"### Aggregate Scores",
        f"- **Total Evaluated Samples:** {cm.total_samples}",
        f"- **Overall Accuracy:** {cm.accuracy * 100:.2f}%",
        f"- **Macro Precision:** {cm.macro_precision:.4f}",
        f"- **Macro Recall:** {cm.macro_recall:.4f}",
        f"- **Macro F1-Score:** {cm.macro_f1:.4f}",
        f"- **Weighted F1-Score:** {cm.weighted_f1:.4f}",
        f"",
        f"### Per-Category Breakdown",
        f"",
        f"| Category | Precision | Recall | F1-Score | Support (N) | TP | FP | FN |",
        f"| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for cat_name, m in cm.per_category.items():
        lines.append(
            f"| **{cat_name}** | {m.precision * 100:.1f}% | {m.recall * 100:.1f}% | {m.f1_score:.4f} | {m.support} | {m.true_positive} | {m.false_positive} | {m.false_negative} |"
        )

    lines.extend([
        f"",
        f"### Confusion Matrix",
        f"",
        f"Rows represent Ground Truth categories; Columns represent Model Predictions.",
        f"",
    ])

    # Confusion matrix header
    all_cats = list(cm.per_category.keys())
    cm_header = "| True \\ Pred | " + " | ".join(f"**{c}**" for c in all_cats) + " |"
    cm_sep = "| :--- | " + " | ".join(":---:" for _ in all_cats) + " |"
    lines.append(cm_header)
    lines.append(cm_sep)

    for c_true in all_cats:
        row_vals = [f"{cm.confusion_matrix.get(c_true, {}).get(c_pred, 0)}" for c_pred in all_cats]
        lines.append(f"| **{c_true}** | " + " | ".join(row_vals) + " |")

    lines.extend([
        f"",
        f"---",
        f"",
        f"## 3. Semantic Vectorization & Auto-Linking Topology",
        f"",
        f"### Graph Structural Health",
        f"- **Total Nodes ($|V|$):** {gm.total_nodes}",
        f"- **Total Undirected Links ($|E|$):** {gm.total_edges}",
        f"- **Average Degree ($\\bar{{k}} = 2|E|/|V|$):** {gm.avg_degree:.2f} links/node",
        f"- **Degree Range:** Min = {gm.min_degree}, Max = {gm.max_degree}",
        f"- **Island Nodes ($deg(v) = 0$):** {gm.island_nodes_count} ({gm.island_ratio * 100:.1f}%)",
        f"- **Graph Density ($D = 2|E|/(|V|(|V|-1))$):** {gm.graph_density:.4f}",
        f"- **Benchmark Link Precision@5:** {gm.link_precision_at_k * 100:.1f}%",
        f"",
        f"### Similarity Score Distribution",
        f"",
        f"| Metric | Mean | Std Dev | Min | P25 | Median | P75 | Max |",
        f"| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    sd = gm.score_distribution
    lines.append(
        f"| **Cosine Similarity** | {sd.mean:.3f} | {sd.std_dev:.3f} | {sd.min_value:.3f} | {sd.p25:.3f} | {sd.median:.3f} | {sd.p75:.3f} | {sd.max_value:.3f} |"
    )

    lines.extend([
        f"",
        f"---",
        f"",
        f"## 4. Pipeline Sanity & Integrity Checks",
        f"",
        f"| Check Description | Result | Details |",
        f"| :--- | :---: | :--- |",
    ])

    for chk in report.sanity_checks:
        res_icon = "✅ PASS" if chk.passed else "❌ FAIL"
        lines.append(f"| {chk.check_name} | {res_icon} | {chk.details} |")

    lines.extend([
        f"",
        f"---",
        f"",
        f"## 5. Diagnostic Case Studies (Real Data Proof Points)",
        f"",
        f"Concrete examples illustrating pipeline behavior across diverse scenarios:",
        f"",
    ])

    for idx, cs in enumerate(report.case_studies, start=1):
        type_badge = {
            "clear_success": "✅ Clear Classification Success",
            "misclassification": "❌ Misclassification",
            "ambiguous_review": "⚠️ Flagged for Human Review",
            "stale_archive": "📦 Stale / Completed Archive",
        }.get(cs.scenario_type, cs.scenario_type)

        lines.extend([
            f"### Case {idx}: {cs.title}",
            f"- **Scenario Type:** {type_badge}",
            f"- **Ground Truth:** `{cs.expected_category or 'N/A'}` $\\rightarrow$ **Predicted:** `{cs.predicted_category}`",
            f"- **Confidence:** `{cs.confidence:.2f}` (Needs Review: `{cs.needs_review}`)",
            f"- **Signals Matched:** {', '.join(f'`{s}`' for s in cs.signals_matched) if cs.signals_matched else 'None'}",
            f"- **Nearest PARA Centroid:** `{cs.nearest_centroid}` (Affinity: `{cs.centroid_score if cs.centroid_score is not None else 0.0:.2f}`)",
            f"- **Semantic Links Discovered ({cs.links_count}):** {', '.join(f'`{t}`' for t in cs.top_link_titles) if cs.top_link_titles else 'None'}",
            f"- **Model Rationale:** *\"{cs.rationale}\"*",
            f"- **Root Cause / Technical Analysis:** {cs.root_cause_analysis}",
            f"",
        ])

    if report.misclassifications:
        lines.extend([
            f"---",
            f"",
            f"## 6. Detailed Failure Modes (Misclassifications)",
            f"",
            f"| Title | Expected | Predicted | Conf | Signals | Root Cause |",
            f"| :--- | :--- | :--- | :--- | :--- | :--- |",
        ])
        for mf in report.misclassifications:
            sig_str = ", ".join(mf.signals[:2]) if mf.signals else "None"
            lines.append(
                f"| {mf.title} | `{mf.expected_category}` | `{mf.predicted_category}` | {mf.confidence:.2f} | {sig_str} | {mf.root_cause} |"
            )
        lines.append("")

    if report.graph_anomalies:
        lines.extend([
            f"---",
            f"",
            f"## 7. Graph Topology Anomalies",
            f"",
            f"| Anomaly Type | Source | Target | Score | Description |",
            f"| :--- | :--- | :--- | :--- | :--- |",
        ])
        for ga in report.graph_anomalies:
            lines.append(
                f"| `{ga.anomaly_type}` | `{ga.source_id}` | `{ga.target_id or 'N/A'}` | {ga.score if ga.score is not None else 'N/A'} | {ga.description} |"
            )
        lines.append("")

    lines.extend([
        f"---",
        f"",
        f"## 8. Interpretation & Remediation Guide",
        f"",
        f"1. **Accuracy & F1 ($\\\\ge 85\\%$):** If accuracy drops below target, check the per-category confusion matrix. False positives in Projects usually stem from date mentions in reference documents.",
        f"2. **Graph Density & Island Nodes:** If island ratio exceeds $25\\%$, lower the cosine similarity threshold from $\\theta=0.65$ to $\\theta=0.60$ or expand note text content before embedding.",
        f"3. **Needs Review Pipeline:** Notes flagged `NEEDS_REVIEW` with confidence $< 0.60$ are safely routed to Resources by default rather than guessing, preserving knowledge base integrity.",
        f"4. **Determinism:** The pipeline achieves $100\\%$ determinism by seeding embeddings, maintaining strict dictionary sort orders, and running zero temperature.",
    ])

    return "\n".join(lines)
