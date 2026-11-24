"""SecondSelf — CLI Entrypoint for Question Answering Engine (Task 4.1).

Provides command-line interface for natural-language corpus retrieval and synthesis:
  ask.py "How does vector linking work?" [--top-k N] [--no-llm] [--json] [--expand-graph]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from config import get_settings
from ask.context import ContextBuilder
from ask.models import Answer
from ask.retriever import HybridRetriever
from ask.synthesizer import Synthesizer

logger = logging.getLogger("secondself.ask.cli")


def build_arg_parser() -> argparse.ArgumentParser:
    """Builds CLI argument parser for question answering."""
    settings = get_settings()
    parser = argparse.ArgumentParser(
        prog="secondself-ask",
        description="Ask questions over your SecondSelf personal knowledge corpus with grounded citations.",
    )
    parser.add_argument(
        "question",
        type=str,
        help="Natural language question to answer from corpus",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=settings.ask.top_k,
        help=f"Number of top corpus items to retrieve (default: {settings.ask.top_k})",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=settings.ask.min_score,
        help=f"Minimum cosine similarity score threshold (default: {settings.ask.min_score})",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Force pure retrieval-only mode (bypasses LLM synthesis)",
    )
    parser.add_argument(
        "--expand-graph",
        action="store_true",
        default=settings.ask.expand_graph,
        help="Enable hybrid 1st-degree graph neighbor expansion via graph.json",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=settings.ask.max_context_chars,
        help=f"Maximum character budget for assembled context (default: {settings.ask.max_context_chars})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output response as formatted machine-readable JSON",
    )
    parser.add_argument(
        "--index-file",
        type=Path,
        default=settings.paths.index_file,
        help="Path to precomputed vector embeddings index (default: public/index.json)",
    )
    parser.add_argument(
        "--graph-file",
        type=Path,
        default=settings.paths.graph_file,
        help="Path to precomputed knowledge graph artifact (default: public/graph.json)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable detailed debug logging to stderr",
    )
    return parser


def ask(
    question: str,
    top_k: int | None = None,
    min_score: float | None = None,
    use_llm: bool = True,
    expand_graph: bool = False,
    max_context_chars: int | None = None,
    index_file: Path | str | None = None,
    graph_file: Path | str | None = None,
    groq_client: Any | None = None,
) -> Answer:
    """High-level programmatic API for asking questions over the SecondSelf corpus.

    Args:
        question: Natural language question.
        top_k: Optional cap on retrieved items.
        min_score: Optional similarity score threshold.
        use_llm: True for LLM synthesis; False for retrieval-only summary.
        expand_graph: Whether to expand 1st-degree neighbors via graph.json.
        max_context_chars: Character ceiling for assembled context.
        index_file: Optional path to index.json.
        graph_file: Optional path to graph.json.
        groq_client: Optional injected Groq client for mocking/custom backends.

    Returns:
        Answer dataclass containing answer text, citations, retrieved items, and latency.
    """
    settings = get_settings()
    k = top_k if top_k is not None else settings.ask.top_k
    threshold = min_score if min_score is not None else settings.ask.min_score
    should_expand = expand_graph if expand_graph is not None else settings.ask.expand_graph

    # 1. Initialize Retriever
    from link.store import EmbeddingStore
    from link.vectorizer import Vectorizer

    idx_path = Path(index_file).resolve() if index_file else settings.paths.index_file.resolve()
    store = EmbeddingStore(index_file=idx_path)
    vectorizer = Vectorizer()

    retriever = HybridRetriever(
        store=store,
        vectorizer=vectorizer,
        graph_path=graph_file,
    )

    # 2. Retrieve Items
    retrieved_items = retriever.retrieve(
        query=question,
        top_k=k,
        min_score=threshold,
        expand_graph=should_expand,
    )

    # 3. Synthesize Answer
    context_builder = ContextBuilder(max_chars=max_context_chars)
    synthesizer = Synthesizer(
        groq_client=groq_client,
        context_builder=context_builder,
    )

    return synthesizer.synthesize(
        question=question,
        retrieved_items=retrieved_items,
        use_llm=use_llm,
        max_context_chars=max_context_chars,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI execution entrypoint."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    answer = ask(
        question=args.question,
        top_k=args.top_k,
        min_score=args.min_score,
        use_llm=not args.no_llm,
        expand_graph=args.expand_graph,
        max_context_chars=args.max_context_chars,
        index_file=args.index_file,
        graph_file=args.graph_file,
    )

    if args.json:
        print(answer.to_json())
    else:
        print(answer.render_console())

    return 0


if __name__ == "__main__":
    sys.exit(main())
