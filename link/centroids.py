"""SecondSelf — PARA Category Centroids and Semantic Alignment Calculator.

Calculates normalized centroids for each of the four PARA categories (Projects, Areas,
Resources, Archives), computes nearest-centroid affinities, and evaluates classification alignment.
"""

from __future__ import annotations

import logging
from typing import Any
import numpy as np

from classify.taxonomy import PARACategory
from link.models import Embedding, Link

logger = logging.getLogger("secondself.link.centroids")


class PARACentroidCalculator:
    """Calculates geometric category centers and evaluates semantic cluster alignment."""

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions
        # Centroid vectors: category_name -> 1D float32 normalized NumPy array
        self._centroids: dict[str, np.ndarray] = {}
        self._counts: dict[str, int] = {cat.value: 0 for cat in PARACategory}

    @property
    def centroids(self) -> dict[str, list[float]]:
        """Returns dictionary of category centroids as plain float lists."""
        return {cat: vec.tolist() for cat, vec in self._centroids.items()}

    def compute_centroids(
        self,
        items: list[tuple[str, str, Embedding]],
    ) -> dict[str, list[float]]:
        """Computes and normalizes centroid vectors for each PARA category.

        Args:
            items: List of (item_id, category_name, Embedding) tuples.

        Returns:
            Dictionary mapping category name to normalized centroid vector.
        """
        category_sums: dict[str, np.ndarray] = {}
        category_counts: dict[str, int] = {}

        for _, cat, emb in items:
            if emb.status != "success" or not emb.vector:
                continue

            canonical_cat = PARACategory.from_string(cat).value
            vec = emb.to_numpy()

            if canonical_cat not in category_sums:
                category_sums[canonical_cat] = np.zeros(self.dimensions, dtype=np.float32)
                category_counts[canonical_cat] = 0

            category_sums[canonical_cat] += vec
            category_counts[canonical_cat] += 1

        new_centroids: dict[str, np.ndarray] = {}
        for cat_name, sum_vec in category_sums.items():
            count = category_counts[cat_name]
            if count > 0:
                mean_vec = sum_vec / count
                norm = float(np.linalg.norm(mean_vec))
                if norm > 0:
                    normalized_vec = mean_vec / norm
                else:
                    normalized_vec = mean_vec
                new_centroids[cat_name] = normalized_vec.astype(np.float32)

        self._centroids = new_centroids
        self._counts = category_counts
        return self.centroids

    def nearest_category(
        self,
        embedding: Embedding | list[float] | np.ndarray,
    ) -> tuple[str, float, dict[str, float]]:
        """Calculates cosine similarity to all populated category centroids and finds the nearest.

        Args:
            embedding: Target vector.

        Returns:
            Tuple of (nearest_category_name, highest_score, all_scores_dict).
        """
        if not self._centroids:
            return ("Resources", 0.0, {})

        if isinstance(embedding, Embedding):
            q_vec = embedding.to_numpy()
        else:
            q_vec = np.asarray(embedding, dtype=np.float32)

        q_norm = float(np.linalg.norm(q_vec))
        if q_norm > 0 and abs(q_norm - 1.0) > 1e-4:
            q_vec = q_vec / q_norm

        scores: dict[str, float] = {}
        for cat_name, centroid_vec in self._centroids.items():
            sim = float(np.dot(centroid_vec, q_vec))
            scores[cat_name] = sim

        # Deterministic winner selection: highest similarity, stable alphabetical tie-break
        sorted_cats = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        nearest_cat, best_score = sorted_cats[0]
        return nearest_cat, best_score, scores

    def evaluate_alignment(
        self,
        assigned_category: str,
        embedding: Embedding,
    ) -> tuple[bool, str, float, str]:
        """Evaluates whether the assigned category matches the nearest geometric centroid.

        Args:
            assigned_category: The PARA category previously assigned by classifier.
            embedding: Document embedding vector.

        Returns:
            Tuple of (is_aligned, nearest_category, similarity_score, reason).
        """
        canonical_assigned = PARACategory.from_string(assigned_category).value
        nearest_cat, score, scores = self.nearest_category(embedding)

        is_aligned = canonical_assigned == nearest_cat
        assigned_score = scores.get(canonical_assigned, 0.0)

        if is_aligned:
            reason = (
                f"Semantically aligned with assigned '{canonical_assigned}' category centroid "
                f"(cosine {score:.2f})."
            )
        else:
            reason = (
                f"Semantic divergence detected: assigned to '{canonical_assigned}' ({assigned_score:.2f}), "
                f"but closest to '{nearest_cat}' centroid ({score:.2f})."
            )

        return is_aligned, nearest_cat, score, reason

    def create_centroid_link(
        self,
        item_id: str,
        assigned_category: str,
        embedding: Embedding,
    ) -> Link | None:
        """Constructs an explainable Link model connecting the item to its nearest PARA centroid."""
        if not self._centroids:
            return None

        is_aligned, nearest_cat, score, reason = self.evaluate_alignment(
            assigned_category=assigned_category,
            embedding=embedding,
        )

        return Link(
            source_id=item_id,
            target_id=f"para:{nearest_cat}",
            score=score,
            reason=reason,
            method="item_to_centroid",
            target_title=f"{nearest_cat} Centroid",
            target_category=nearest_cat,
        )
