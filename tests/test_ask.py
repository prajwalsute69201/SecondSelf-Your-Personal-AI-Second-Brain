"""SecondSelf — Question Answering & Retrieval-Augmented Generation Tests (Task 4.1).

Tests semantic retrieval, graph expansion, context assembly, LLM synthesis,
citation extraction, groundedness validation, and retrieval-only fallback.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ask.cli import ask, main
from ask.context import ContextBuilder
from ask.models import Answer, Citation, RetrievedItem
from ask.prompts import NOT_FOUND_TOKEN
from ask.retriever import HybridRetriever
from ask.synthesizer import Synthesizer
from link.models import Embedding
from link.store import EmbeddingStore
from link.vectorizer import Vectorizer


@pytest.fixture
def mock_store() -> EmbeddingStore:
    """Fixture providing an in-memory EmbeddingStore with deterministic dummy vectors."""
    store = EmbeddingStore(auto_load=False)

    # 3 dummy items with known 3D unit vectors
    items = [
        (
            "note-1",
            [1.0, 0.0, 0.0],
            {
                "title": "Force-directed Associative Layout",
                "category": "Resources",
                "summary": "Graph layout using spring-electrical forces and physics simulation.",
                "tags": ["graph", "layout", "physics"],
            },
        ),
        (
            "note-2",
            [0.8, 0.6, 0.0],
            {
                "title": "Vector Embeddings & Semantic Search",
                "category": "Resources",
                "summary": "Dense vector representations for nearest-neighbor similarity querying.",
                "tags": ["vector", "embeddings", "search"],
            },
        ),
        (
            "note-3",
            [0.0, 1.0, 0.0],
            {
                "title": "Autonomous PARA Classification",
                "category": "Projects",
                "summary": "Categorizing items into Projects, Areas, Resources, and Archives.",
                "tags": ["para", "classification"],
            },
        ),
    ]

    for rec_id, vec, meta in items:
        emb = Embedding(
            vector=vec,
            model_name="mock-model",
            dimensions=3,
            content_hash=f"hash-{rec_id}",
            status="success",
        )
        store.add(rec_id, emb, meta)

    return store


@pytest.fixture
def mock_vectorizer() -> Vectorizer:
    """Fixture providing a mock Vectorizer returning predictable vectors."""
    vec = MagicMock(spec=Vectorizer)

    def fake_embed(query: str) -> Embedding:
        q = query.lower()
        if "force" in q or "layout" in q or "physics" in q:
            vector = [1.0, 0.0, 0.0]
        elif "vector" in q or "search" in q or "embedding" in q:
            vector = [0.8, 0.6, 0.0]
        elif "para" in q or "classif" in q:
            vector = [0.0, 1.0, 0.0]
        else:
            vector = [0.577, 0.577, 0.577]

        return Embedding(
            vector=vector,
            model_name="mock-model",
            dimensions=3,
            content_hash=f"hash-{query}",
            status="success",
        )

    vec.embed.side_effect = fake_embed
    return vec


def test_retriever_known_question_retrieves_top_item(
    mock_store: EmbeddingStore, mock_vectorizer: Vectorizer
) -> None:
    """Verifies that asking for 'force layout' retrieves note-1 with highest similarity."""
    retriever = HybridRetriever(store=mock_store, vectorizer=mock_vectorizer)
    results = retriever.retrieve("Explain force-directed layout", top_k=2)

    assert len(results) > 0
    top = results[0]
    assert top.id == "item:note-1"
    assert "Force-directed" in top.title
    assert pytest.approx(top.score, rel=1e-3) == 1.0


def test_retriever_threshold_filters_weak_matches(
    mock_store: EmbeddingStore, mock_vectorizer: Vectorizer
) -> None:
    """Verifies that items with score below min_score are excluded."""
    retriever = HybridRetriever(store=mock_store, vectorizer=mock_vectorizer)
    # Target vector [1.0, 0.0, 0.0]. note-3 is [0.0, 1.0, 0.0] -> dot product 0.0
    results = retriever.retrieve("force-directed layout", top_k=5, min_score=0.5)

    retrieved_ids = [r.id for r in results]
    assert "item:note-1" in retrieved_ids  # score 1.0
    assert "item:note-2" in retrieved_ids  # score 0.8
    assert "item:note-3" not in retrieved_ids  # score 0.0 < 0.5


def test_retriever_determinism_identical_runs(
    mock_store: EmbeddingStore, mock_vectorizer: Vectorizer
) -> None:
    """Verifies that identical queries against the same store yield identical results and rankings."""
    retriever = HybridRetriever(store=mock_store, vectorizer=mock_vectorizer)

    run1 = retriever.retrieve("vector search", top_k=3)
    run2 = retriever.retrieve("vector search", top_k=3)

    assert len(run1) == len(run2)
    for it1, it2 in zip(run1, run2):
        assert it1.id == it2.id
        assert it1.score == it2.score


def test_retriever_graph_expansion(
    mock_store: EmbeddingStore, mock_vectorizer: Vectorizer, tmp_path: Path
) -> None:
    """Verifies that graph expansion discovers semantically connected neighbor nodes."""
    # Create a small graph.json where note-1 connects to note-3
    graph_payload = {
        "nodes": [
            {"id": "item:note-1", "label": "Force-directed", "properties": {"title": "Force-directed"}},
            {"id": "item:note-3", "label": "PARA Classification", "properties": {"title": "PARA Classification"}},
        ],
        "edges": [
            {
                "from": "item:note-1",
                "to": "item:note-3",
                "type": "SEMANTIC_LINK",
                "value": 0.9,
            }
        ],
    }
    graph_file = tmp_path / "graph.json"
    graph_file.write_text(json.dumps(graph_payload), encoding="utf-8")

    retriever = HybridRetriever(
        store=mock_store,
        vectorizer=mock_vectorizer,
        graph_path=graph_file,
    )

    # With min_score=0.5, note-3 has vector score 0.0, but graph expansion should inject it
    # neighbor_score = parent_score (1.0) * edge (0.9) * discount (0.85) = 0.765 >= 0.5
    results = retriever.retrieve("force layout", top_k=5, min_score=0.5, expand_graph=True)
    retrieved_ids = [r.id for r in results]

    assert "item:note-1" in retrieved_ids
    assert "item:note-3" in retrieved_ids
    neighbor_item = next(it for it in results if it.id == "item:note-3")
    assert neighbor_item.retrieval_method == "graph_neighbor"
    assert pytest.approx(neighbor_item.score, rel=1e-2) == 0.765


def test_context_builder_bounds_and_truncates() -> None:
    """Verifies that ContextBuilder respects character budget and drops overflow items."""
    builder = ContextBuilder(max_chars=350)
    items = [
        RetrievedItem(id="item:1", title="First", snippet="A" * 150, content="A" * 150, score=0.9),
        RetrievedItem(id="item:2", title="Second", snippet="B" * 150, content="B" * 150, score=0.8),
        RetrievedItem(id="item:3", title="Third", snippet="C" * 150, content="C" * 150, score=0.7),
    ]

    bundle = builder.build(items)
    assert len(bundle.items) == 1  # 2nd item exceeds 350 char budget
    assert bundle.dropped_items_count == 2
    assert "--- START SOURCE ITEM [item:1] ---" in bundle.formatted_context


def test_synthesizer_llm_valid_citations() -> None:
    """Verifies that valid citations are parsed and mapped to Citation objects."""
    items = [
        RetrievedItem(id="item:note-1", title="Layout Note", snippet="Spring forces", content="Spring forces", score=0.95),
        RetrievedItem(id="item:note-2", title="Vector Note", snippet="Embeddings", content="Embeddings", score=0.80),
    ]

    mock_groq = MagicMock()
    mock_groq.chat.completions.create.return_value = MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(
                    content="SecondSelf organizes items using physical force simulation [Note: item:note-1] and dense embeddings [item:note-2]."
                )
            )
        ]
    )

    synthesizer = Synthesizer(groq_client=mock_groq)
    answer = synthesizer.synthesize("How does SecondSelf organize notes?", items)

    assert answer.mode == "llm_synthesis"
    assert len(answer.citations) == 2
    assert answer.is_grounded is True
    assert answer.uncited_claims_flag is False
    assert answer.citations[0].item_id == "item:note-1"
    assert answer.citations[1].item_id == "item:note-2"


def test_synthesizer_flags_uncited_claims() -> None:
    """Verifies that an answer making claims without any citations is flagged."""
    items = [
        RetrievedItem(id="item:note-1", title="Layout Note", snippet="Spring forces", content="Spring forces", score=0.95),
    ]

    mock_groq = MagicMock()
    mock_groq.chat.completions.create.return_value = MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(
                    content="SecondSelf is a personal AI system that runs entirely on local static files."
                )
            )
        ]
    )

    synthesizer = Synthesizer(groq_client=mock_groq)
    answer = synthesizer.synthesize("What is SecondSelf?", items)

    assert len(answer.citations) == 0
    assert answer.uncited_claims_flag is True


def test_synthesizer_detects_unverified_hallucinated_citation() -> None:
    """Verifies that citing a nonexistent item ID sets is_grounded = False."""
    items = [
        RetrievedItem(id="item:note-1", title="Layout Note", snippet="Spring forces", content="Spring forces", score=0.95),
    ]

    mock_groq = MagicMock()
    mock_groq.chat.completions.create.return_value = MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(
                    content="SecondSelf uses quantum computing [Note: item:hallucinated-id-999]."
                )
            )
        ]
    )

    synthesizer = Synthesizer(groq_client=mock_groq)
    answer = synthesizer.synthesize("What hardware is used?", items)

    assert answer.is_grounded is False
    # Hallucinated ID is discarded from citations list
    assert len(answer.citations) == 0


def test_synthesizer_retrieval_only_fallback() -> None:
    """Verifies that use_llm=False generates a structured summary with auto-citations."""
    items = [
        RetrievedItem(id="item:note-1", title="Layout Note", snippet="Spring physics", content="Spring physics", score=0.95),
    ]

    synthesizer = Synthesizer(groq_client=None)
    answer = synthesizer.synthesize("Explain layout", items, use_llm=False)

    assert answer.mode == "retrieval_only"
    assert answer.model is None
    assert "Layout Note" in answer.text
    assert len(answer.citations) == 1
    assert answer.citations[0].item_id == "item:note-1"
    assert answer.is_grounded is True


def test_synthesizer_empty_retrieval_returns_not_found() -> None:
    """Verifies that an empty retrieved list returns the standard not found message."""
    synthesizer = Synthesizer(groq_client=None)
    answer = synthesizer.synthesize("Unknown topic", [])

    assert NOT_FOUND_TOKEN in answer.text
    assert len(answer.citations) == 0
    assert len(answer.retrieved) == 0


def test_ask_cli_dry_run_no_llm_json(capsys: pytest.CaptureFixture) -> None:
    """Verifies running the ask CLI with --no-llm and --json produces valid JSON output."""
    exit_code = main(["What is SecondSelf?", "--no-llm", "--json"])
    assert exit_code == 0

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["question"] == "What is SecondSelf?"
    assert data["mode"] == "retrieval_only"
    assert "text" in data
    assert "citations" in data
    assert "retrieved" in data
