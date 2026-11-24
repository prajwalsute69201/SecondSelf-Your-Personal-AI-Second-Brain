"""SecondSelf — Hybrid Semantic & Graph Retriever (Task 4.1).

Combines dense vector similarity querying with deterministic ranking,
thresholding, and optional 1st-degree graph neighbor expansion via graph.json.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from config import get_settings
from config.base import AskConfig
from link.models import Embedding
from link.store import EmbeddingStore
from link.vectorizer import Vectorizer
from ask.models import RetrievedItem

logger = logging.getLogger("secondself.ask.retriever")


class HybridRetriever:
    """Retrieves relevant corpus items via vector search and graph expansion."""

    def __init__(
        self,
        store: EmbeddingStore | None = None,
        vectorizer: Vectorizer | None = None,
        graph_path: Path | str | None = None,
        config: AskConfig | None = None,
    ) -> None:
        settings = get_settings()
        self.config = config or settings.ask
        self.store = store if store is not None else EmbeddingStore()
        self.vectorizer = vectorizer if vectorizer is not None else Vectorizer()

        self.graph_path = Path(graph_path).resolve() if graph_path else settings.paths.graph_file.resolve()
        self._graph_data: dict[str, Any] | None = None

    def _load_graph(self) -> dict[str, Any] | None:
        """Lazily loads and caches the precomputed graph projection."""
        if self._graph_data is not None:
            return self._graph_data

        candidate_paths = [
            self.graph_path,
            self.graph_path.parent / "data" / "graph.json",
            Path("public/data/graph.json").resolve(),
            Path("public/graph.json").resolve(),
        ]

        for p in candidate_paths:
            if p.exists() and p.is_file():
                try:
                    self._graph_data = json.loads(p.read_text(encoding="utf-8"))
                    logger.debug("Loaded graph projection from %s", p)
                    return self._graph_data
                except Exception as exc:
                    logger.warning("Could not parse graph from '%s': %s", p, exc)

        return None

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float | None = None,
        expand_graph: bool | None = None,
    ) -> list[RetrievedItem]:
        """Retrieves and ranks the most semantically relevant items for the given query.

        Args:
            query: Natural language query string.
            top_k: Optional cap on maximum items returned (defaults to config).
            min_score: Minimum similarity score threshold (defaults to config).
            expand_graph: Whether to expand 1st-degree neighbors in graph (defaults to config).

        Returns:
            Deterministically sorted list of RetrievedItem instances.
        """
        k = top_k if top_k is not None else self.config.top_k
        threshold = min_score if min_score is not None else self.config.min_score
        should_expand = expand_graph if expand_graph is not None else self.config.expand_graph

        clean_query = query.strip()
        if not clean_query:
            return []

        # 1. Embed Query
        emb_result: Embedding = self.vectorizer.embed(clean_query)
        if not emb_result.vector:
            logger.warning("Query vectorization returned empty vector for query: %s", clean_query)
            return []

        # 2. Dense Vector Similarity Search
        # Retrieve more candidates initially to allow graph expansion & deduplication
        raw_candidates = self.store.query(
            vector=emb_result.vector,
            top_k=max(k * 3, 10),
            min_score=threshold,
        )

        item_map: dict[str, RetrievedItem] = {}

        for rec_id, score in raw_candidates:
            meta = self.store.get_metadata(rec_id)
            title = meta.get("title") or rec_id
            snippet = meta.get("summary") or meta.get("content", "")[:250]
            content = meta.get("content") or snippet

            canonical_id = f"item:{rec_id}" if not rec_id.startswith("item:") else rec_id

            item = RetrievedItem(
                id=canonical_id,
                title=title,
                snippet=snippet,
                content=content,
                score=score,
                category=meta.get("category", "Resources"),
                source_type=meta.get("source_type", "markdown"),
                retrieval_method="vector",
                tags=meta.get("tags", []),
                file_path=meta.get("file_path"),
            )
            item_map[canonical_id] = item

        # 3. Optional Graph Neighbor Expansion
        if should_expand and item_map:
            graph = self._load_graph()
            if graph and "edges" in graph:
                discount = self.config.graph_discount_factor
                edges = graph.get("edges") or graph.get("links") or []
                nodes_by_id = {n["id"]: n for n in graph.get("nodes", []) if isinstance(n, dict) and "id" in n}

                # Find 1st-degree neighbors of initial vector hits
                initial_ids = list(item_map.keys())
                for seed_id in initial_ids:
                    seed_score = item_map[seed_id].score

                    for edge in edges:
                        from_id = edge.get("from") or edge.get("source")
                        to_id = edge.get("to") or edge.get("target")
                        edge_type = edge.get("type", "")

                        # Focus on semantic or topical associations
                        if edge_type not in ("SEMANTIC_LINK", "BELONGS_TO", "TAGGED_WITH"):
                            continue

                        neighbor_id = None
                        if from_id == seed_id and to_id:
                            neighbor_id = to_id
                        elif to_id == seed_id and from_id:
                            neighbor_id = from_id

                        if not neighbor_id or not neighbor_id.startswith("item:"):
                            continue

                        edge_val = float(edge.get("value", 1.0))
                        neighbor_score = round(seed_score * edge_val * discount, 4)

                        if neighbor_score < threshold:
                            continue

                        if neighbor_id in item_map:
                            # If already present, keep the higher score
                            if neighbor_score > item_map[neighbor_id].score:
                                item_map[neighbor_id].score = neighbor_score
                                item_map[neighbor_id].retrieval_method = "hybrid_graph"
                        else:
                            # Add newly discovered neighbor
                            node_data = nodes_by_id.get(neighbor_id, {})
                            props = node_data.get("properties", {})
                            n_title = props.get("title") or node_data.get("label") or neighbor_id
                            n_summary = node_data.get("summary") or props.get("summary", "")
                            n_cat = props.get("category") or node_data.get("group", "Resources")

                            item_map[neighbor_id] = RetrievedItem(
                                id=neighbor_id,
                                title=n_title,
                                snippet=n_summary[:250],
                                content=n_summary,
                                score=neighbor_score,
                                category=n_cat,
                                source_type=props.get("source_type", "markdown"),
                                retrieval_method="graph_neighbor",
                                tags=node_data.get("tags", []),
                                file_path=props.get("file_path"),
                            )

        # 4. Deterministic Ranking & Tie-Breaking
        # Sort descending by score, ties broken alphabetically by item ID
        ranked_items = sorted(item_map.values(), key=lambda it: (-it.score, it.id))

        final_items = ranked_items[:k]
        logger.debug(
            "Query '%s' retrieved %d items (top score: %.3f)",
            clean_query,
            len(final_items),
            final_items[0].score if final_items else 0.0,
        )
        return final_items
