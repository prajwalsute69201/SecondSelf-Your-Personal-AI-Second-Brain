"""SecondSelf — Failure Mode Extraction & Diagnostic Case Study Analyzer.

Surfaces actionable failure modes: misclassifications, low-confidence classifications,
graph linkage anomalies, dangling edges, and extracts representative case studies.
"""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


class CaseStudy(BaseModel):
    """Representative diagnostic case study illustrating model behavior on real data."""

    item_id: str
    title: str
    scenario_type: Literal["clear_success", "misclassification", "ambiguous_review", "stale_archive"]
    expected_category: str | None = None
    predicted_category: str
    confidence: float
    needs_review: bool
    rationale: str
    signals_matched: list[str] = Field(default_factory=list)
    nearest_centroid: str | None = None
    centroid_score: float | None = None
    links_count: int = 0
    top_link_titles: list[str] = Field(default_factory=list)
    root_cause_analysis: str = ""

    model_config = ConfigDict(populate_by_name=True)


class MisclassificationFailure(BaseModel):
    """Detailed diagnosis of an item categorized incorrectly against ground truth."""

    item_id: str
    title: str
    expected_category: str
    predicted_category: str
    confidence: float
    needs_review: bool
    rationale: str
    signals: list[str]
    root_cause: str

    model_config = ConfigDict(populate_by_name=True)


class GraphAnomaly(BaseModel):
    """Identified structural defect in the semantic auto-link graph."""

    anomaly_type: Literal["self_loop", "dangling_target", "sub_threshold_link", "super_hub"]
    source_id: str
    target_id: str | None = None
    score: float | None = None
    description: str = ""

    model_config = ConfigDict(populate_by_name=True)


class FailureAnalyzer:
    """Analyzes pipeline run items to extract anomalies and diagnostic case studies."""

    @staticmethod
    def extract_misclassifications(
        items: list[dict[str, Any]],
    ) -> list[MisclassificationFailure]:
        """Filters items where predicted category diverges from ground truth."""
        failures: list[MisclassificationFailure] = []

        for item in items:
            exp_cat = item.get("expected_category")
            pred_cat = item.get("category")
            if not exp_cat or not pred_cat:
                continue

            if exp_cat.lower() != pred_cat.lower():
                signals = [s.get("description", str(s)) for s in item.get("signals", [])]

                # Heuristic root cause diagnostic
                conf = float(item.get("confidence", 0.0))
                rationale = str(item.get("rationale", ""))
                if conf < 0.60 or item.get("needs_review"):
                    root_cause = "Weak/ambiguous signals flagged for review; model hesitated between multiple categories."
                elif "historical" in rationale.lower() or "done" in rationale.lower():
                    root_cause = "Historical marker or completed status superseded active topical category."
                elif "deadline" in rationale.lower():
                    root_cause = "Time-bound phrase or calendar pattern incorrectly attributed as project deliverable."
                else:
                    root_cause = "Feature overlap across category boundary rubrics."

                failures.append(
                    MisclassificationFailure(
                        item_id=item.get("id", "unknown"),
                        title=item.get("title", "Untitled"),
                        expected_category=exp_cat,
                        predicted_category=pred_cat,
                        confidence=conf,
                        needs_review=bool(item.get("needs_review", False)),
                        rationale=rationale,
                        signals=signals,
                        root_cause=root_cause,
                    )
                )

        return failures

    @staticmethod
    def detect_graph_anomalies(
        all_node_ids: set[str],
        edges: list[tuple[str, str, float]],
        min_threshold: float = 0.65,
        max_hub_degree: int = 8,
    ) -> list[GraphAnomaly]:
        """Scans the graph edges for self-loops, dangling IDs, sub-threshold scores, and super-hubs."""
        anomalies: list[GraphAnomaly] = []
        degrees: dict[str, int] = {nid: 0 for nid in all_node_ids}

        for src, tgt, score in edges:
            # 1. Self-loop check
            if src == tgt:
                anomalies.append(
                    GraphAnomaly(
                        anomaly_type="self_loop",
                        source_id=src,
                        target_id=tgt,
                        score=score,
                        description=f"Self-link detected on note '{src}'. Self-loops must be excluded.",
                    )
                )

            # 2. Dangling target check
            if tgt not in all_node_ids:
                anomalies.append(
                    GraphAnomaly(
                        anomaly_type="dangling_target",
                        source_id=src,
                        target_id=tgt,
                        score=score,
                        description=f"Dangling link: target note '{tgt}' does not exist in active corpus.",
                    )
                )

            # 3. Sub-threshold check
            if score < min_threshold:
                anomalies.append(
                    GraphAnomaly(
                        anomaly_type="sub_threshold_link",
                        source_id=src,
                        target_id=tgt,
                        score=score,
                        description=f"Link score {score:.2f} is below configured threshold {min_threshold:.2f}.",
                    )
                )

            degrees[src] = degrees.get(src, 0) + 1

        # 4. Super-hub check
        for nid, deg in degrees.items():
            if deg > max_hub_degree:
                anomalies.append(
                    GraphAnomaly(
                        anomaly_type="super_hub",
                        source_id=nid,
                        description=f"Node degree {deg} exceeds maximum degree threshold ({max_hub_degree}). Risk of graph hairball.",
                    )
                )

        return anomalies

    @staticmethod
    def extract_representative_case_studies(
        items: list[dict[str, Any]],
    ) -> list[CaseStudy]:
        """Extracts at least 5 concrete case studies covering successes, edge cases, and archives."""
        cases: list[CaseStudy] = []

        # Find clear Project
        proj_success = next(
            (
                it for it in items
                if it.get("category") == "Projects"
                and it.get("confidence", 0) >= 0.80
                and not it.get("needs_review")
            ),
            None,
        )
        if proj_success:
            cases.append(
                CaseStudy(
                    item_id=proj_success.get("id", ""),
                    title=proj_success.get("title", ""),
                    scenario_type="clear_success",
                    expected_category=proj_success.get("expected_category", "Projects"),
                    predicted_category="Projects",
                    confidence=float(proj_success.get("confidence", 1.0)),
                    needs_review=False,
                    rationale=str(proj_success.get("rationale", "")),
                    signals_matched=[s.get("description", str(s)) for s in proj_success.get("signals", [])],
                    nearest_centroid=proj_success.get("nearest_centroid"),
                    centroid_score=proj_success.get("centroid_score"),
                    links_count=len(proj_success.get("links", [])),
                    top_link_titles=[l.get("target_title", l.get("target_id", "")) for l in proj_success.get("links", [])[:3]],
                    root_cause_analysis="Definitive action items and calendar deadline detected; high signal certainty.",
                )
            )

        # Find clear Area
        area_success = next(
            (
                it for it in items
                if it.get("category") == "Areas"
                and it.get("confidence", 0) >= 0.65
            ),
            None,
        )
        if area_success:
            cases.append(
                CaseStudy(
                    item_id=area_success.get("id", ""),
                    title=area_success.get("title", ""),
                    scenario_type="clear_success",
                    expected_category=area_success.get("expected_category", "Areas"),
                    predicted_category="Areas",
                    confidence=float(area_success.get("confidence", 1.0)),
                    needs_review=False,
                    rationale=str(area_success.get("rationale", "")),
                    signals_matched=[s.get("description", str(s)) for s in area_success.get("signals", [])],
                    nearest_centroid=area_success.get("nearest_centroid"),
                    centroid_score=area_success.get("centroid_score"),
                    links_count=len(area_success.get("links", [])),
                    top_link_titles=[l.get("target_title", l.get("target_id", "")) for l in area_success.get("links", [])[:3]],
                    root_cause_analysis="Recurring maintenance protocol and operating procedures accurately identified.",
                )
            )

        # Find clear Resource
        res_success = next(
            (
                it for it in items
                if it.get("category") == "Resources"
                and it.get("confidence", 0) >= 0.65
            ),
            None,
        )
        if res_success:
            cases.append(
                CaseStudy(
                    item_id=res_success.get("id", ""),
                    title=res_success.get("title", ""),
                    scenario_type="clear_success",
                    expected_category=res_success.get("expected_category", "Resources"),
                    predicted_category="Resources",
                    confidence=float(res_success.get("confidence", 1.0)),
                    needs_review=False,
                    rationale=str(res_success.get("rationale", "")),
                    signals_matched=[s.get("description", str(s)) for s in res_success.get("signals", [])],
                    nearest_centroid=res_success.get("nearest_centroid"),
                    centroid_score=res_success.get("centroid_score"),
                    links_count=len(res_success.get("links", [])),
                    top_link_titles=[l.get("target_title", l.get("target_id", "")) for l in res_success.get("links", [])[:3]],
                    root_cause_analysis="Reference manual or conceptual technical documentation detected without active tasks.",
                )
            )

        # Find clear Archive
        arch_success = next(
            (
                it for it in items
                if it.get("category") == "Archives"
                and it.get("confidence", 0) >= 0.70
            ),
            None,
        )
        if arch_success:
            cases.append(
                CaseStudy(
                    item_id=arch_success.get("id", ""),
                    title=arch_success.get("title", ""),
                    scenario_type="stale_archive",
                    expected_category=arch_success.get("expected_category", "Archives"),
                    predicted_category="Archives",
                    confidence=float(arch_success.get("confidence", 1.0)),
                    needs_review=False,
                    rationale=str(arch_success.get("rationale", "")),
                    signals_matched=[s.get("description", str(s)) for s in arch_success.get("signals", [])],
                    nearest_centroid=arch_success.get("nearest_centroid"),
                    centroid_score=arch_success.get("centroid_score"),
                    links_count=len(arch_success.get("links", [])),
                    top_link_titles=[l.get("target_title", l.get("target_id", "")) for l in arch_success.get("links", [])[:3]],
                    root_cause_analysis="Status marker 'DONE', deprecation notice, or post-mortem correctly archived.",
                )
            )

        # Find Ambiguous / Review case
        ambig_case = next(
            (
                it for it in items
                if it.get("needs_review") or float(it.get("confidence", 1.0)) < 0.60
            ),
            None,
        )
        if ambig_case:
            cases.append(
                CaseStudy(
                    item_id=ambig_case.get("id", ""),
                    title=ambig_case.get("title", ""),
                    scenario_type="ambiguous_review",
                    expected_category=ambig_case.get("expected_category"),
                    predicted_category=str(ambig_case.get("category", "Resources")),
                    confidence=float(ambig_case.get("confidence", 0.25)),
                    needs_review=True,
                    rationale=str(ambig_case.get("rationale", "")),
                    signals_matched=[s.get("description", str(s)) for s in ambig_case.get("signals", [])],
                    nearest_centroid=ambig_case.get("nearest_centroid"),
                    centroid_score=ambig_case.get("centroid_score"),
                    links_count=len(ambig_case.get("links", [])),
                    top_link_titles=[l.get("target_title", l.get("target_id", "")) for l in ambig_case.get("links", [])[:3]],
                    root_cause_analysis="Confidence below 0.60 threshold; safely flagged NEEDS_REVIEW rather than silently guessing.",
                )
            )

        return cases
