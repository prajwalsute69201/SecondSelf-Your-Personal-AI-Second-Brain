"""SecondSelf — Autonomous Semantic Auto-Linking Engine.

Discovers meaningful reciprocal connections between notes, applies configurable
cosine similarity thresholds and top-K caps, computes deterministic rankings,
and injects explainable Wikilinks into markdown notes.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from config import get_settings
from config.base import VectorConfig
from link.centroids import PARACentroidCalculator
from link.models import Embedding, Link, LinkResult, NoteItem
from link.store import EmbeddingStore
from link.vectorizer import Vectorizer

logger = logging.getLogger("secondself.link.linker")


class AutoLinker:
    """Discovers semantic relationships, injects markdown backlinks, and exports graph links."""

    def __init__(
        self,
        config: VectorConfig | None = None,
        vectorizer: Vectorizer | None = None,
        store: EmbeddingStore | None = None,
    ) -> None:
        self.config = config or get_settings().vector
        self.vectorizer = vectorizer if vectorizer is not None else Vectorizer(config=self.config)
        self.store = store if store is not None else EmbeddingStore(config=self.config)
        self.centroid_calc = (
            PARACentroidCalculator(dimensions=self.config.dimensions)
            if self.config.centroid_linking_enabled
            else None
        )

    def link_item(
        self,
        item: NoteItem,
        embedding: Embedding,
        store: EmbeddingStore | None = None,
        centroid_calc: PARACentroidCalculator | None = None,
    ) -> LinkResult:
        """Computes thresholded semantic links for a single note against the store.

        Args:
            item: The source NoteItem.
            embedding: The precomputed Embedding for the item.
            store: Optional EmbeddingStore override.
            centroid_calc: Optional PARACentroidCalculator override.

        Returns:
            Structured LinkResult with discovered links and centroid alignment.
        """
        active_store = store if store is not None else self.store
        active_centroids = centroid_calc if centroid_calc is not None else self.centroid_calc

        if embedding.status != "success" or not embedding.vector:
            logger.warning("Cannot link item '%s' with failed embedding", item.id)
            return LinkResult(
                item_id=item.id,
                links=[],
                embedding_ref=embedding.content_hash,
                status="failed",
                metadata={"error": embedding.error_message or "Invalid embedding vector"},
            )

        # 1. Query nearest neighbors with strict threshold and top-K cap
        candidates = active_store.query(
            vector=embedding.vector,
            top_k=self.config.top_k_links,
            min_score=self.config.similarity_threshold,
            exclude_id=item.id,
        )

        item_links: list[Link] = []
        for target_id, score in candidates:
            target_meta = active_store.get_metadata(target_id)
            target_title = target_meta.get("title") or target_id
            target_cat = target_meta.get("category")

            pct = int(round(score * 100))
            reason = f"Cosine similarity {score:.2f} (semantic affinity with '{target_title}')"

            item_links.append(
                Link(
                    source_id=item.id,
                    target_id=target_id,
                    score=score,
                    reason=reason,
                    method="item_to_item",
                    target_title=target_title,
                    target_category=target_cat,
                )
            )

        # 2. Evaluate secondary PARA centroid connection if enabled
        nearest_centroid: str | None = None
        centroid_score: float | None = None
        category_alignment: bool | None = None
        centroid_link: Link | None = None

        if active_centroids is not None and active_centroids._centroids:
            is_aligned, nearest_cat, c_score, c_reason = active_centroids.evaluate_alignment(
                assigned_category=item.category,
                embedding=embedding,
            )
            nearest_centroid = nearest_cat
            centroid_score = c_score
            category_alignment = is_aligned
            centroid_link = Link(
                source_id=item.id,
                target_id=f"para:{nearest_cat}",
                score=c_score,
                reason=c_reason,
                method="item_to_centroid",
                target_title=f"{nearest_cat} Centroid",
                target_category=nearest_cat,
            )

        return LinkResult(
            item_id=item.id,
            links=item_links,
            embedding_ref=embedding.content_hash,
            nearest_centroid=nearest_centroid,
            centroid_score=centroid_score,
            category_alignment=category_alignment,
            centroid_link=centroid_link,
            status="success",
            metadata={
                "title": item.title,
                "category": item.category,
                "link_count": len(item_links),
            },
        )

    def inject_wikilinks_into_markdown(
        self,
        markdown_content: str,
        links: list[Link],
    ) -> str:
        """Injects or cleanly replaces the '## Related Notes' section with reciprocal Wikilinks.

        Args:
            markdown_content: Raw note markdown content.
            links: Discovered semantic item-to-item links.

        Returns:
            Updated markdown content string.
        """
        # Strip existing Related Notes section
        cleaned = re.sub(
            r"\n##\s+Related Notes.*?(?=\n##|\Z)",
            "",
            markdown_content,
            flags=re.DOTALL,
        ).rstrip()

        if not links:
            return cleaned + "\n"

        # Build formatted related notes block
        wikilink_lines = [link.to_wikilink() for link in links]
        related_section = "## Related Notes\n" + "\n".join(wikilink_lines)

        return f"{cleaned}\n\n{related_section}\n"

    def link_all(
        self,
        notes: list[NoteItem],
        store: EmbeddingStore | None = None,
        vectorizer: Vectorizer | None = None,
        centroid_calc: PARACentroidCalculator | None = None,
        dry_run: bool = False,
    ) -> list[LinkResult]:
        """Processes an entire corpus of notes: generates embeddings, computes centroids,
        discovers reciprocal links, injects markdown backlinks, and exports the vector index.

        Args:
            notes: List of NoteItem objects representing the vault.
            store: Optional EmbeddingStore override.
            vectorizer: Optional Vectorizer override.
            centroid_calc: Optional PARACentroidCalculator override.
            dry_run: If True, executes calculations without writing to disk.

        Returns:
            List of LinkResult objects for all processed notes.
        """
        active_store = store if store is not None else self.store
        active_vec = vectorizer if vectorizer is not None else self.vectorizer
        active_centroids = centroid_calc if centroid_calc is not None else self.centroid_calc

        if not notes:
            logger.info("Corpus is empty. Zero notes to link.")
            return []

        # 1. Prepare batch items and generate embeddings
        batch_inputs: list[tuple[str, str]] = [(n.id, n.composite_text) for n in notes]
        embeddings = active_vec.embed_batch(batch_inputs)

        # 2. Populate store with all valid embeddings and note metadata
        for note, emb in zip(notes, embeddings):
            if emb.status == "success" and emb.vector:
                active_store.add(
                    record_id=note.id,
                    embedding=emb,
                    metadata={
                        "title": note.title,
                        "category": note.category,
                        "summary": note.summary,
                        "tags": note.tags,
                        "content": note.content,
                    },
                )

        # 3. Calculate PARA centroids across valid corpus embeddings
        if active_centroids is not None:
            centroid_items = [
                (n.id, n.category, emb)
                for n, emb in zip(notes, embeddings)
                if emb.status == "success" and emb.vector
            ]
            active_centroids.compute_centroids(centroid_items)

        # 4. Link each note against the populated index
        results: list[LinkResult] = []
        for note, emb in zip(notes, embeddings):
            link_res = self.link_item(
                item=note,
                embedding=emb,
                store=active_store,
                centroid_calc=active_centroids,
            )
            results.append(link_res)

            # 5. Inject markdown backlinks into file if not dry_run
            if not dry_run and note.file_path and note.file_path.exists():
                try:
                    raw_md = note.file_path.read_text(encoding="utf-8")
                    updated_md = self.inject_wikilinks_into_markdown(raw_md, link_res.links)
                    note.file_path.write_text(updated_md, encoding="utf-8")
                except Exception as exc:
                    logger.warning("Could not inject Wikilinks into '%s': %s", note.file_path, exc)

        # 6. Export index to public/index.json if not dry_run
        if not dry_run:
            try:
                active_store.export_index()
            except Exception as exc:
                logger.error("Failed to export index.json: %s", exc)

        return results
