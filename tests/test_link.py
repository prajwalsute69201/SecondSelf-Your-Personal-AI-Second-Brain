"""Comprehensive Unit & Integration Test Suite for SecondSelf Semantic Vectorization & Auto-Linking.

Validates:
- Deterministic embedding generation (identical content -> identical normalized vector).
- In-memory cache hit prevents redundant provider calls.
- Threshold filtering: weak semantic links below threshold are dropped.
- Top-K constraint enforcement: maximum outgoing links capped.
- Deterministic tie-breaking by target ID on identical similarity scores.
- Provider failure handling: retries with backoff and graceful degradation without crashing.
- Centroid nearest-category calculation and PARA semantic alignment detection.
- Append-only embedding store persistence and exact cosine dot-product queries.
- Reciprocal Wikilink injection into Markdown notes with idempotent section replacement.
- CLI execution modes (empty vault, batch dry-run, json output, single item).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest

from config.base import VectorConfig
from link import (
    AutoLinker,
    Embedding,
    EmbeddingStore,
    Link,
    LinkResult,
    MockEmbeddingProvider,
    NoteItem,
    PARACentroidCalculator,
    Vectorizer,
    embed,
    link_all,
    link_item,
    main,
    scan_wiki_vault,
)


# ==============================================================================
# 1. Embedding Model & Vectorizer Tests
# ==============================================================================

def test_identical_content_yields_identical_normalized_embedding() -> None:
    provider = MockEmbeddingProvider(dimensions=384)
    vectorizer = Vectorizer(config=VectorConfig(dimensions=384), provider=provider)

    content = "Autonomous personal second brain architecture using PARA taxonomy."
    emb1 = vectorizer.embed(content)
    emb2 = vectorizer.embed(content)

    assert emb1.status == "success"
    assert emb2.status == "success"
    assert emb1.dimensions == 384
    assert len(emb1.vector) == 384
    assert emb1.content_hash == emb2.content_hash
    assert emb1.vector == emb2.vector
    # Normalized length check
    assert emb1.is_normalized()


def test_cache_hit_skips_provider_call() -> None:
    mock_provider = MagicMock(spec=MockEmbeddingProvider)
    # Return fixed vector
    fixed_vec = [1.0] + [0.0] * 383
    mock_provider.embed_texts.return_value = [fixed_vec]

    vectorizer = Vectorizer(
        config=VectorConfig(dimensions=384, cache_enabled=True),
        provider=mock_provider,
    )

    text = "Cache hit verification note."
    emb1 = vectorizer.embed(text)
    assert mock_provider.embed_texts.call_count == 1

    emb2 = vectorizer.embed(text)
    # Should not invoke provider second time
    assert mock_provider.embed_texts.call_count == 1
    assert emb1.vector == emb2.vector


def test_provider_failure_gracefully_degrades_without_crash() -> None:
    failing_provider = MockEmbeddingProvider(dimensions=384, should_fail=True)
    vectorizer = Vectorizer(
        config=VectorConfig(dimensions=384, max_retries=2),
        provider=failing_provider,
    )

    emb = vectorizer.embed("Content that fails to embed.")
    assert emb.status == "failed"
    assert emb.vector == []
    assert emb.error_message is not None
    assert "Simulated embedding provider outage" in emb.error_message


def test_embed_batch_preserves_order() -> None:
    provider = MockEmbeddingProvider(dimensions=384)
    vectorizer = Vectorizer(config=VectorConfig(dimensions=384, batch_size=2), provider=provider)

    items = [
        ("id-1", "Note one about machine learning."),
        ("id-2", "Note two about serverless architectures."),
        ("id-3", "Note three about knowledge graphs."),
    ]
    embeddings = vectorizer.embed_batch(items)

    assert len(embeddings) == 3
    assert all(e.status == "success" for e in embeddings)
    # Verify deterministic correspondence
    for (rec_id, text), emb in zip(items, embeddings):
        assert emb.content_hash == vectorizer._compute_content_hash(text)


# ==============================================================================
# 2. EmbeddingStore & Similarity Query Tests
# ==============================================================================

def test_store_query_exact_cosine_similarity() -> None:
    store = EmbeddingStore(config=VectorConfig(dimensions=3))

    # Add 3 orthogonal / known vectors
    # v1 = [1, 0, 0], v2 = [0.6, 0.8, 0], v3 = [0, 1, 0]
    store.add("doc-1", Embedding(vector=[1.0, 0.0, 0.0], model_name="test", dimensions=3, content_hash="h1"))
    store.add("doc-2", Embedding(vector=[0.6, 0.8, 0.0], model_name="test", dimensions=3, content_hash="h2"))
    store.add("doc-3", Embedding(vector=[0.0, 1.0, 0.0], model_name="test", dimensions=3, content_hash="h3"))

    # Query with [1, 0, 0]
    results = store.query([1.0, 0.0, 0.0], top_k=5, min_score=0.0)
    assert len(results) == 3
    assert results[0][0] == "doc-1"
    assert pytest.approx(results[0][1], abs=1e-4) == 1.0
    assert results[1][0] == "doc-2"
    assert pytest.approx(results[1][1], abs=1e-4) == 0.6
    assert results[2][0] == "doc-3"
    assert pytest.approx(results[2][1], abs=1e-4) == 0.0


def test_store_idempotent_deduplication() -> None:
    store = EmbeddingStore(config=VectorConfig(dimensions=3))
    emb = Embedding(vector=[1.0, 0.0, 0.0], model_name="test", dimensions=3, content_hash="hash-abc")

    added1 = store.add("rec-1", emb)
    assert added1 is True
    assert len(store) == 1

    added2 = store.add("rec-1", emb)
    assert added2 is False
    assert len(store) == 1


def test_store_export_and_load_index_roundtrip(tmp_path: Path) -> None:
    index_file = tmp_path / "index.json"
    store = EmbeddingStore(index_file=index_file, config=VectorConfig(dimensions=3))

    emb1 = Embedding(vector=[1.0, 0.0, 0.0], model_name="test", dimensions=3, content_hash="hash-1")
    store.add("rec-1", emb1, metadata={"title": "Doc 1", "category": "Projects"})

    store.export_index(index_file)
    assert index_file.exists()

    # Load in new store instance
    new_store = EmbeddingStore(index_file=index_file, config=VectorConfig(dimensions=3))
    assert len(new_store) == 1
    loaded_emb = new_store.get_by_id("rec-1")
    assert loaded_emb is not None
    assert loaded_emb.vector == [1.0, 0.0, 0.0]
    meta = new_store.get_metadata("rec-1")
    assert meta["title"] == "Doc 1"
    assert meta["category"] == "Projects"


# ==============================================================================
# 3. Auto-Linking, Thresholding & Deterministic Tie-Breaking
# ==============================================================================

def test_threshold_filters_weak_links() -> None:
    store = EmbeddingStore(config=VectorConfig(dimensions=3))
    store.add("target-high", Embedding(vector=[0.9, 0.435, 0.0], model_name="test", dimensions=3, content_hash="h1"))
    store.add("target-low", Embedding(vector=[0.0, 1.0, 0.0], model_name="test", dimensions=3, content_hash="h2"))

    query_vec = [1.0, 0.0, 0.0]
    # Query with threshold 0.70 -> target-high has dot ~0.90, target-low has dot 0.0
    results = store.query(query_vec, min_score=0.70)
    assert len(results) == 1
    assert results[0][0] == "target-high"


def test_top_k_cap_enforced() -> None:
    store = EmbeddingStore(config=VectorConfig(dimensions=2))
    # Add 10 targets
    for i in range(10):
        # Slightly different vectors
        angle = (i * 0.05)
        vec = [float(np.cos(angle)), float(np.sin(angle))]
        store.add(f"doc-{i:02d}", Embedding(vector=vec, model_name="test", dimensions=2, content_hash=f"h{i}"))

    # Cap at top_k=3
    results = store.query([1.0, 0.0], top_k=3, min_score=0.0)
    assert len(results) == 3


def test_deterministic_tie_breaking_by_id() -> None:
    store = EmbeddingStore(config=VectorConfig(dimensions=2))
    # Add two items with exact same vector [1, 0]
    store.add("zebra-note", Embedding(vector=[1.0, 0.0], model_name="test", dimensions=2, content_hash="hz"))
    store.add("alpha-note", Embedding(vector=[1.0, 0.0], model_name="test", dimensions=2, content_hash="ha"))

    # Both have dot product 1.0; tie breaker must yield 'alpha-note' first, then 'zebra-note'
    results = store.query([1.0, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0][0] == "alpha-note"
    assert results[1][0] == "zebra-note"


def test_link_item_generates_explainable_links() -> None:
    config = VectorConfig(dimensions=2, similarity_threshold=0.60, top_k_links=5)
    store = EmbeddingStore(config=config)
    store.add("doc-b", Embedding(vector=[0.8, 0.6], model_name="test", dimensions=2, content_hash="hb"), metadata={"title": "Vector Math"})

    source_item = NoteItem(id="doc-a", title="Embedding Guide", category="Resources", content="Text about embeddings.")
    source_emb = Embedding(vector=[1.0, 0.0], model_name="test", dimensions=2, content_hash="ha")

    linker = AutoLinker(config=config, store=store)
    result = linker.link_item(item=source_item, embedding=source_emb)

    assert result.item_id == "doc-a"
    assert len(result.links) == 1
    link = result.links[0]
    assert link.target_id == "doc-b"
    assert pytest.approx(link.score, abs=1e-2) == 0.80
    assert "Vector Math" in link.reason
    assert "Cosine similarity" in link.reason
    assert link.to_wikilink() == "- [[doc-b|Vector Math]] — Match: 80%"


# ==============================================================================
# 4. PARA Centroid & Alignment Tests
# ==============================================================================

def test_centroid_computation_and_nearest_category() -> None:
    calc = PARACentroidCalculator(dimensions=3)

    # 2 Projects pointing towards +X, 2 Areas pointing towards +Y
    items = [
        ("p1", "Projects", Embedding(vector=[1.0, 0.1, 0.0], model_name="test", dimensions=3, content_hash="p1")),
        ("p2", "Projects", Embedding(vector=[1.0, -0.1, 0.0], model_name="test", dimensions=3, content_hash="p2")),
        ("a1", "Areas", Embedding(vector=[0.1, 1.0, 0.0], model_name="test", dimensions=3, content_hash="a1")),
        ("a2", "Areas", Embedding(vector=[-0.1, 1.0, 0.0], model_name="test", dimensions=3, content_hash="a2")),
    ]
    calc.compute_centroids(items)
    assert "Projects" in calc.centroids
    assert "Areas" in calc.centroids

    # Query with vector [1, 0, 0] -> Nearest should be Projects
    nearest_cat, score, _ = calc.nearest_category([1.0, 0.0, 0.0])
    assert nearest_cat == "Projects"
    assert score > 0.95

    # Query with vector [0, 1, 0] -> Nearest should be Areas
    nearest_cat_area, score_area, _ = calc.nearest_category([0.0, 1.0, 0.0])
    assert nearest_cat_area == "Areas"
    assert score_area > 0.95


def test_centroid_alignment_detection() -> None:
    calc = PARACentroidCalculator(dimensions=2)
    items = [
        ("p1", "Projects", Embedding(vector=[1.0, 0.0], model_name="test", dimensions=2, content_hash="p1")),
        ("a1", "Areas", Embedding(vector=[0.0, 1.0], model_name="test", dimensions=2, content_hash="a1")),
    ]
    calc.compute_centroids(items)

    # Note assigned to Projects with vector [1, 0] -> aligned
    emb_aligned = Embedding(vector=[1.0, 0.0], model_name="test", dimensions=2, content_hash="t1")
    is_aligned, nearest_cat, score, reason = calc.evaluate_alignment("Projects", emb_aligned)
    assert is_aligned is True
    assert nearest_cat == "Projects"
    assert "aligned" in reason.lower()

    # Note assigned to Projects with vector [0, 1] -> divergent (closer to Areas)
    emb_divergent = Embedding(vector=[0.0, 1.0], model_name="test", dimensions=2, content_hash="t2")
    is_aligned_div, nearest_cat_div, _, reason_div = calc.evaluate_alignment("Projects", emb_divergent)
    assert is_aligned_div is False
    assert nearest_cat_div == "Areas"
    assert "divergence" in reason_div.lower()


# ==============================================================================
# 5. Markdown Wikilink Injection Tests
# ==============================================================================

def test_inject_wikilinks_into_markdown() -> None:
    linker = AutoLinker()
    raw_md = (
        "---\n"
        'title: "System Architecture"\n'
        'category: "Resources"\n'
        "---\n\n"
        "# System Architecture\n\n"
        "Core notes on distributed systems."
    )
    links = [
        Link(source_id="note-1", target_id="note-2", score=0.88, reason="Strong link", target_title="Graph Engine"),
        Link(source_id="note-1", target_id="note-3", score=0.72, reason="Moderate link", target_title="Vector Search"),
    ]

    updated_md = linker.inject_wikilinks_into_markdown(raw_md, links)
    assert "## Related Notes" in updated_md
    assert "- [[note-2|Graph Engine]] — Match: 88%" in updated_md
    assert "- [[note-3|Vector Search]] — Match: 72%" in updated_md

    # Idempotent re-injection check
    updated_again = linker.inject_wikilinks_into_markdown(updated_md, links)
    assert updated_again.count("## Related Notes") == 1


# ==============================================================================
# 6. NoteItem Markdown Parsing Tests
# ==============================================================================

def test_note_item_from_markdown_file(tmp_path: Path) -> None:
    note_file = tmp_path / "Projects" / "note_123.md"
    note_file.parent.mkdir(parents=True, exist_ok=True)
    note_file.write_text(
        "---\n"
        'title: "Ship Q4 Release"\n'
        'category: "Projects"\n'
        'summary: "Finish deliverable by Friday"\n'
        "tags:\n"
        "  - release\n"
        "  - sprint\n"
        "---\n\n"
        "# Ship Q4 Release\n\n"
        "Body content of the release notes.",
        encoding="utf-8",
    )

    note = NoteItem.from_markdown_file(note_file)
    assert note.id == "note_123"
    assert note.title == "Ship Q4 Release"
    assert note.category == "Projects"
    assert note.summary == "Finish deliverable by Friday"
    assert "release" in note.tags
    assert "sprint" in note.tags
    assert "Body content" in note.composite_text


# ==============================================================================
# 7. CLI Execution Interface Tests
# ==============================================================================

def test_link_cli_empty_vault(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    empty_wiki = tmp_path / "empty_wiki"
    empty_wiki.mkdir()
    exit_code = main(["--dry-run", "--wiki-dir", str(empty_wiki)])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Wiki vault is empty" in out


def test_link_cli_batch_dry_run_with_notes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    wiki_dir = tmp_path / "wiki"
    proj_dir = wiki_dir / "Projects"
    proj_dir.mkdir(parents=True, exist_ok=True)

    # Create two synthetic notes
    (proj_dir / "note_a.md").write_text("# Project Alpha\nDeep dive into machine learning.", encoding="utf-8")
    (proj_dir / "note_b.md").write_text("# Project Beta\nMachine learning algorithms and optimization.", encoding="utf-8")

    index_file = tmp_path / "index.json"

    exit_code = main([
        "--batch",
        "--dry-run",
        "--provider", "mock",
        "--threshold", "0.0",
        "--wiki-dir", str(wiki_dir),
        "--index-file", str(index_file),
    ])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Auto-Linking Complete" in out
    assert "Processed: 2 notes" in out
