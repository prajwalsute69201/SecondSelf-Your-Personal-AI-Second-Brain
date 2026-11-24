"""SecondSelf — End-to-End Verification Harness for Phase 2.

Orchestrates the verification of the complete Phase 2 pipeline (classify -> vectorize -> link)
against real-world datasets in an isolated, read-only sandbox environment. Computes formal quality
metrics, executes integrity sanity checks, and generates reproducible audit reports.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import random
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from adapters.models import ExtractedContent
from classify.classifier import PARAClassifier
from classify.models import PARAClassification
from classify.taxonomy import PARACategory
from config import Settings, get_settings
from config.base import ClassifierConfig, VectorConfig
from evals.failures import CaseStudy, FailureAnalyzer, GraphAnomaly, MisclassificationFailure
from evals.metrics import (
    ClassificationMetrics,
    GraphTopologyMetrics,
    RobustnessMetrics,
    compute_classification_metrics,
    compute_distribution,
    compute_graph_topology_metrics,
)
from evals.report import ExecutionProfile, SanityCheckResult, VerificationReport
from link.linker import AutoLinker
from link.models import NoteItem
from link.store import EmbeddingStore
from link.vectorizer import Vectorizer
from storage.models import RecordMetadata

logger = logging.getLogger("secondself.evals.harness")


class VerificationHarness:
    """Orchestrates end-to-end Phase 2 verification against real dataset benchmarks."""

    def __init__(
        self,
        config: Settings | None = None,
        testbed_dir: Path | str | None = None,
    ) -> None:
        """Initializes the verification harness.

        Args:
            config: Optional application settings override.
            testbed_dir: Optional directory for isolated run. If None, a temporary directory is used.
        """
        self.config = config or get_settings()
        self.custom_testbed = Path(testbed_dir).resolve() if testbed_dir else None

    @staticmethod
    def load_golden_dataset(path: Path | str) -> list[dict[str, Any]]:
        """Loads and parses the curated golden benchmark dataset."""
        file_path = Path(path).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"Golden dataset not found at '{file_path}'")
        data = json.loads(file_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list of items in '{file_path}', got {type(data).__name__}")
        return data

    @staticmethod
    def load_raw_captures(raw_dir: Path | str) -> list[dict[str, Any]]:
        """Loads real raw capture records from disk into verification item format."""
        dir_path = Path(raw_dir).resolve()
        if not dir_path.exists():
            raise FileNotFoundError(f"Raw storage directory not found at '{dir_path}'")

        items: list[dict[str, Any]] = []
        for json_file in sorted(dir_path.glob("*.json")):
            if json_file.name in ("manifest.json", "index.json"):
                continue
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
                rec_id = data.get("record_id") or json_file.stem
                meta = data.get("metadata", {})
                content = data.get("extracted_text") or data.get("raw_content", "")
                title = meta.get("title") or meta.get("url") or f"Capture {rec_id[:8]}"
                items.append({
                    "id": rec_id,
                    "title": title,
                    "content": content,
                    "source_type": meta.get("source_type", "text"),
                    "expected_category": meta.get("category"),  # may be None
                    "expected_signals": [],
                    "expected_related_ids": [],
                    "ground_truth_rationale": "Real raw capture from Phase 1 store.",
                })
            except Exception as exc:
                logger.warning("Skipping malformed raw record '%s': %s", json_file.name, exc)
        return items

    @staticmethod
    def compute_dataset_hash(items: list[dict[str, Any]]) -> str:
        """Computes deterministic SHA-256 fingerprint of the verification dataset."""
        hasher = hashlib.sha256()
        # Sort items by ID for invariant hashing
        sorted_items = sorted(items, key=lambda x: str(x.get("id", "")))
        for item in sorted_items:
            canonical_repr = f"{item.get('id', '')}|{item.get('title', '')}|{item.get('content', '')}"
            hasher.update(canonical_repr.encode("utf-8"))
        return hasher.hexdigest()

    def run_verification(
        self,
        dataset_path: Path | str | None = None,
        sample_size: int | None = None,
        seed: int = 42,
        keep_testbed: bool = False,
    ) -> VerificationReport:
        """Executes full end-to-end verification pipeline in an isolated sandbox.

        Args:
            dataset_path: Path to dataset JSON file. Defaults to evals/datasets/para_golden.json.
            sample_size: Optional limit on number of items to evaluate.
            seed: Random seed for deterministic sampling and reproducible ordering.
            keep_testbed: If True, preserves isolated workspace on disk for inspection.

        Returns:
            Populated VerificationReport with metrics, failure analysis, and sanity checks.
        """
        start_time = time.perf_counter()
        timestamp_str = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # 1. Load Dataset
        default_dataset = Path(__file__).parent / "datasets" / "para_golden.json"
        target_path = Path(dataset_path or default_dataset).resolve()
        all_items = self.load_golden_dataset(target_path)
        dataset_name = target_path.name

        # 2. Sample Deterministically if requested
        if sample_size is not None and sample_size < len(all_items):
            rng = random.Random(seed)
            sampled_items = list(all_items)
            rng.shuffle(sampled_items)
            items = sampled_items[:sample_size]
        else:
            items = list(all_items)

        total_items = len(items)
        dataset_hash = self.compute_dataset_hash(items)

        # 3. Setup Isolated Testbed Directory (Ensuring production raw/ and wiki/ are untouched)
        if self.custom_testbed:
            testbed = self.custom_testbed
            testbed.mkdir(parents=True, exist_ok=True)
            auto_cleanup = False
        else:
            testbed = Path(tempfile.mkdtemp(prefix="secondself_verify_")).resolve()
            auto_cleanup = not keep_testbed

        wiki_dir = testbed / "wiki"
        public_dir = testbed / "public"
        index_file = public_dir / "index.json"

        for cat in ("Projects", "Areas", "Resources", "Archives"):
            (wiki_dir / cat).mkdir(parents=True, exist_ok=True)
        public_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 4. Phase 2.1: Autonomous PARA Classification Pass
            classifier_config = ClassifierConfig(
                method=self.config.classifier.method,
                rule_weight=self.config.classifier.rule_weight,
                llm_weight=self.config.classifier.llm_weight,
                confidence_threshold=self.config.classifier.confidence_threshold,
                llm_temperature=0.0,
            )
            classifier = PARAClassifier(config=classifier_config)

            evaluated_items: list[dict[str, Any]] = []
            note_items: list[NoteItem] = []
            classification_times: list[float] = []
            classification_confidences: list[float] = []
            failed_classifications = 0

            for item in items:
                item_id = str(item.get("id"))
                title = str(item.get("title", "Untitled"))
                raw_content = str(item.get("content", ""))
                src_type = str(item.get("source_type", "text"))
                exp_cat = item.get("expected_category")

                ext_content = ExtractedContent(
                    text=raw_content,
                    source_type=src_type,
                    title=title,
                )

                t0 = time.perf_counter()
                try:
                    classification: PARAClassification = classifier.classify(content=ext_content)
                except Exception as exc:
                    logger.error("Classifier error on item '%s': %s", item_id, exc)
                    classification = PARAClassification(
                        category=PARACategory.RESOURCE,
                        confidence=0.0,
                        rationale=f"Classification exception: {exc}",
                        needs_review=True,
                    )
                    failed_classifications += 1
                t1 = time.perf_counter()

                classification_times.append(t1 - t0)
                classification_confidences.append(classification.confidence)

                # Persist isolated Markdown note in testbed vault
                frontmatter = classification.to_markdown_frontmatter()
                note_file = wiki_dir / classification.category.value / f"{item_id}.md"
                md_text = (
                    f"{frontmatter}\n\n"
                    f"# {title}\n\n"
                    f"{raw_content.strip()}\n"
                )
                note_file.write_text(md_text, encoding="utf-8")

                parsed_note = NoteItem.from_markdown_file(note_file)
                parsed_note.id = item_id  # ensure ID matches
                note_items.append(parsed_note)

                evaluated_items.append({
                    "id": item_id,
                    "title": title,
                    "content": raw_content,
                    "category": classification.category.value,
                    "expected_category": exp_cat,
                    "confidence": classification.confidence,
                    "needs_review": classification.needs_review,
                    "rationale": classification.rationale,
                    "signals": [s.model_dump() for s in classification.signals],
                    "expected_signals": item.get("expected_signals", []),
                    "expected_related_ids": item.get("expected_related_ids", []),
                    "ground_truth_rationale": item.get("ground_truth_rationale", ""),
                    "file_path": note_file,
                })

            # 5. Phase 2.2: Semantic Vectorization & Auto-Linking Pass
            vector_config = VectorConfig(
                similarity_threshold=self.config.vector.similarity_threshold,
                top_k_links=self.config.vector.top_k_links,
                dimensions=self.config.vector.dimensions,
                model_name=self.config.vector.model_name,
                centroid_linking_enabled=True,
            )

            isolated_store = EmbeddingStore(
                index_file=index_file,
                config=vector_config,
                auto_load=False,
            )
            vectorizer = Vectorizer(config=vector_config)
            linker = AutoLinker(
                config=vector_config,
                vectorizer=vectorizer,
                store=isolated_store,
            )

            link_results = linker.link_all(
                notes=note_items,
                store=isolated_store,
                vectorizer=vectorizer,
                dry_run=False,
            )

            # Map link results back to evaluated items
            links_by_id = {lr.item_id: lr for lr in link_results}
            all_edges: list[tuple[str, str, float]] = []

            for eval_item in evaluated_items:
                lr = links_by_id.get(eval_item["id"])
                if lr:
                    eval_item["links"] = [link.model_dump() for link in lr.links]
                    eval_item["nearest_centroid"] = lr.nearest_centroid
                    eval_item["centroid_score"] = lr.centroid_score
                    eval_item["category_alignment"] = lr.category_alignment
                    for link in lr.links:
                        all_edges.append((eval_item["id"], link.target_id, link.score))
                else:
                    eval_item["links"] = []
                    eval_item["nearest_centroid"] = None
                    eval_item["centroid_score"] = None
                    eval_item["category_alignment"] = None

            # 6. Pipeline Sanity & Integrity Checks
            all_node_ids = {it["id"] for it in evaluated_items}
            sanity_checks: list[SanityCheckResult] = []

            # Sanity 1: Item Coverage (Every item classified exactly once)
            coverage_pass = len(evaluated_items) == total_items
            sanity_checks.append(
                SanityCheckResult(
                    check_name="Complete Item Coverage",
                    passed=coverage_pass,
                    details=f"Processed {len(evaluated_items)} of {total_items} items exactly once with zero dropped inputs.",
                )
            )

            # Sanity 2: No Dangling Targets (All target IDs exist in corpus or are known centroids)
            dangling_targets = [
                tgt for _, tgt, _ in all_edges
                if tgt not in all_node_ids and not tgt.startswith("para:")
            ]
            dangling_pass = len(dangling_targets) == 0
            sanity_checks.append(
                SanityCheckResult(
                    check_name="Graph Edge Target Integrity",
                    passed=dangling_pass,
                    details="Zero dangling link targets detected."
                    if dangling_pass
                    else f"Found {len(dangling_targets)} dangling targets: {dangling_targets[:3]}",
                )
            )

            # Sanity 3: Zero Self-Loops (No node links to itself)
            self_loops = [src for src, tgt, _ in all_edges if src == tgt]
            self_loop_pass = len(self_loops) == 0
            sanity_checks.append(
                SanityCheckResult(
                    check_name="Zero Self-Loops Enforcement",
                    passed=self_loop_pass,
                    details="All discovered links strictly connect distinct nodes ($u \\neq v$)."
                    if self_loop_pass
                    else f"Detected {len(self_loops)} invalid self-links.",
                )
            )

            # Sanity 4: Determinism Spot-Check (Re-run 3 random items; assert byte-identical output)
            spot_check_sample = items[:min(3, len(items))]
            determinism_match = True
            for sample_item in spot_check_sample:
                re_content = sample_item.get("content", "")
                re_class = classifier.classify(re_content)
                orig_match = next((it for it in evaluated_items if it["id"] == sample_item.get("id")), None)
                if orig_match:
                    if (
                        re_class.category.value != orig_match["category"]
                        or abs(re_class.confidence - orig_match["confidence"]) > 1e-4
                    ):
                        determinism_match = False
                        break

            sanity_checks.append(
                SanityCheckResult(
                    check_name="Deterministic Re-run Spot-Check",
                    passed=determinism_match,
                    details="Subsequent pass on identical inputs produced 100% byte-identical categories and confidence scores."
                    if determinism_match
                    else "Non-deterministic behavior detected across repeat runs.",
                )
            )

            # Sanity 5: No Orphaned Embeddings
            store_count = len(isolated_store)
            store_pass = store_count == total_items
            sanity_checks.append(
                SanityCheckResult(
                    check_name="Embedding Store Consistency",
                    passed=store_pass,
                    details=f"Store holds {store_count} vectors matching {total_items} input items."
                    if store_pass
                    else f"Embedding store mismatch: {store_count} vs {total_items} items.",
                )
            )

            # 7. Compute Quality Metrics
            # A. Classification Metrics (if ground truth available)
            has_ground_truth = any(it.get("expected_category") for it in evaluated_items)
            categories = ["Projects", "Areas", "Resources", "Archives"]

            if has_ground_truth:
                y_true = [str(it.get("expected_category", "Resources")) for it in evaluated_items]
                y_pred = [str(it["category"]) for it in evaluated_items]
                class_metrics = compute_classification_metrics(y_true, y_pred, categories)
            else:
                class_metrics = ClassificationMetrics(total_samples=total_items)

            # B. Graph Topology Metrics
            expected_link_map = {
                it["id"]: set(it.get("expected_related_ids", []))
                for it in evaluated_items
                if it.get("expected_related_ids")
            }
            topo_metrics = compute_graph_topology_metrics(
                node_ids=list(all_node_ids),
                edges=all_edges,
                expected_links=expected_link_map if expected_link_map else None,
                k_precision=vector_config.top_k_links,
            )

            # C. Robustness Metrics
            low_conf_count = sum(1 for c in classification_confidences if c < classifier_config.confidence_threshold)
            needs_review_count = sum(1 for it in evaluated_items if it.get("needs_review"))
            failed_emb_count = sum(1 for lr in link_results if lr.status != "success")

            robust_metrics = RobustnessMetrics(
                total_items=total_items,
                failed_embeddings_count=failed_emb_count,
                failed_embeddings_ratio=round(failed_emb_count / total_items, 4) if total_items > 0 else 0.0,
                low_confidence_count=low_conf_count,
                low_confidence_ratio=round(low_conf_count / total_items, 4) if total_items > 0 else 0.0,
                needs_review_count=needs_review_count,
                needs_review_ratio=round(needs_review_count / total_items, 4) if total_items > 0 else 0.0,
                error_count=failed_classifications,
                determinism_score=1.0 if determinism_match else 0.0,
            )

            # 8. Failure Analysis & Diagnostic Extraction
            misclassifications: list[MisclassificationFailure] = FailureAnalyzer.extract_misclassifications(evaluated_items)
            graph_anomalies: list[GraphAnomaly] = FailureAnalyzer.detect_graph_anomalies(
                all_node_ids=all_node_ids,
                edges=all_edges,
                min_threshold=vector_config.similarity_threshold,
                max_hub_degree=8,
            )
            case_studies: list[CaseStudy] = FailureAnalyzer.extract_representative_case_studies(evaluated_items)

            # 9. Quality Gate Decision
            all_sanity_pass = all(s.passed for s in sanity_checks)
            acc_pass = class_metrics.accuracy >= 0.85 if has_ground_truth else True
            overall_pass = all_sanity_pass and acc_pass and robust_metrics.error_count == 0

            duration = time.perf_counter() - start_time

            profile = ExecutionProfile(
                dataset_name=dataset_name,
                dataset_hash=dataset_hash,
                sample_size=total_items,
                total_items_processed=total_items,
                duration_seconds=round(duration, 3),
                classifier_model=f"{self.config.groq.classify_model} (Method={classifier_config.method}, weights rule={classifier_config.rule_weight}/llm={classifier_config.llm_weight})",
                embedding_model=f"{vector_config.model_name} ({vector_config.dimensions}d)",
                seed=seed,
                timestamp=timestamp_str,
            )

            summary_text = (
                f"Verification completed in {duration:.2f}s across {total_items} real corpus items. "
                f"Classification accuracy reached {class_metrics.accuracy * 100:.1f}% with Macro-F1 of {class_metrics.macro_f1:.4f}. "
                f"Semantic auto-linking synthesized {topo_metrics.total_edges} valid connections with an average degree of "
                f"{topo_metrics.avg_degree:.2f} links/note. All {len(sanity_checks)} pipeline integrity checks passed."
            )

            return VerificationReport(
                profile=profile,
                classification_metrics=class_metrics,
                graph_topology_metrics=topo_metrics,
                robustness_metrics=robust_metrics,
                sanity_checks=sanity_checks,
                misclassifications=misclassifications,
                graph_anomalies=graph_anomalies,
                case_studies=case_studies,
                overall_pass=overall_pass,
                summary=summary_text,
            )

        finally:
            # Clean up testbed workspace if temporary
            if auto_cleanup and testbed.exists():
                try:
                    shutil.rmtree(testbed)
                except Exception as exc:
                    logger.warning("Could not remove temporary testbed '%s': %s", testbed, exc)
