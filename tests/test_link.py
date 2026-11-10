"""Unit and integration tests for link.py semantic linker (Task 2.2.2).

Validates:
- Discovery of semantic relationships using cosine similarity against threshold (0.75).
- High similarity note linking (reciprocal/bidirectional relationships created).
- Unrelated notes (no links created when similarity is below threshold).
- Boundary similarity tests (exactly at threshold 0.75 vs below threshold 0.7499).
- Prevention of self-referential linking (source == target).
- Duplicate relationship prevention in both frontmatter links[] and body [[id]] wikilinks.
- Preservation of existing note content, metadata, and unrelated pre-existing links.
- Handling of missing, empty, or invalid notes and non-finite embedding vectors.
- Empty collection handling (0 notes and 1 isolated note).
- Repeated execution idempotency (no link duplication or file corruption on repeated runs).
- Correct disk persistence of YAML frontmatter and body wikilinks.
- CLI argument parsing, dry-run mode, and error reporting.
"""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import pickle
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Ensure secondself is in sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import lib.embeddings as emb
from lib.models import WikiNote
import lib.storage as storage
import link


def _make_unit_vector(angle_rad: float, dim: int = 384) -> np.ndarray:
    """Generate a 384-dimensional unit vector in a 2D subspace for exact cosine similarities."""
    vec = np.zeros(dim, dtype=np.float32)
    vec[0] = np.cos(angle_rad)
    vec[1] = np.sin(angle_rad)
    return vec


def _make_mock_model_with_vectors(mapping: dict[str, np.ndarray], dim: int = 384) -> MagicMock:
    """Create a mock SentenceTransformer that returns pre-determined vectors based on text substring."""
    mock = MagicMock()

    def _encode(text, convert_to_numpy=True, show_progress_bar=False):
        if isinstance(text, str):
            for key, vec in mapping.items():
                if key in text:
                    return np.asarray(vec, dtype=np.float32)
            # Default fallback: deterministic random unit vector
            rng = np.random.RandomState(abs(hash(text)) % (2**31))
            v = rng.randn(dim).astype(np.float32)
            norm = np.linalg.norm(v)
            return v / norm if norm > 0 else v
        elif isinstance(text, (list, tuple)):
            rows = []
            for t in text:
                matched = False
                for key, vec in mapping.items():
                    if key in t:
                        rows.append(np.asarray(vec, dtype=np.float32))
                        matched = True
                        break
                if not matched:
                    rng = np.random.RandomState(abs(hash(t)) % (2**31))
                    v = rng.randn(dim).astype(np.float32)
                    norm = np.linalg.norm(v)
                    rows.append(v / norm if norm > 0 else v)
            return np.array(rows, dtype=np.float32)
        else:
            raise TypeError("Unsupported text type")

    mock.encode.side_effect = _encode
    return mock


class TestLinkModule(unittest.TestCase):
    """Test suite for SecondSelf Task 2.2.2 (link.py)."""

    def setUp(self) -> None:
        """Create isolated test directory and configure storage project root."""
        self.test_dir = tempfile.mkdtemp(prefix="secondself_link_test_")
        self.test_path = Path(self.test_dir)
        storage.set_project_root(self.test_path)

        # Initialize standard directory structure
        (self.test_path / "data").mkdir(parents=True, exist_ok=True)
        (self.test_path / "wiki" / "Projects").mkdir(parents=True, exist_ok=True)
        (self.test_path / "wiki" / "Areas").mkdir(parents=True, exist_ok=True)
        (self.test_path / "wiki" / "Resources").mkdir(parents=True, exist_ok=True)
        (self.test_path / "wiki" / "Archives").mkdir(parents=True, exist_ok=True)

        storage.save_index(
            {
                "raw_processed": {},
                "embeddings_version": "all-MiniLM-L6-v2",
                "last_graph_build": None,
            },
            root=self.test_path,
        )
        emb.reset_model()

    def tearDown(self) -> None:
        """Clean up temporary directory and reset state."""
        emb.reset_model()
        storage.set_project_root(None)
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # 1. High Similarity Notes (>= 0.75)
    # -------------------------------------------------------------------------

    def test_01_two_highly_similar_notes_linked(self) -> None:
        """Two notes with cosine similarity 0.95 are linked reciprocally."""
        # Angle with cos(theta) = 0.95 -> theta = arccos(0.95)
        theta = float(np.arccos(0.95))
        v1 = _make_unit_vector(0.0)
        v2 = _make_unit_vector(theta)

        note1 = WikiNote(
            id="note0001",
            raw_id="2026-07-06_note0001",
            para="Projects",
            tags=["ml", "deep-learning"],
            summary="Study transformer architectures and attention",
            created="2026-07-06T10:00:00Z",
            links=[],
            body="# Transformers\nUnderstanding multi-head self-attention mechanisms.",
        )
        note2 = WikiNote(
            id="note0002",
            raw_id="2026-07-06_note0002",
            para="Resources",
            tags=["ml", "nlp"],
            summary="Attention Is All You Need paper notes",
            created="2026-07-06T11:00:00Z",
            links=[],
            body="# Attention Paper\nKey architectural breakdown of attention layers.",
        )

        storage.write_wiki_note(note1, root=self.test_path)
        storage.write_wiki_note(note2, root=self.test_path)

        mock_model = _make_mock_model_with_vectors({
            "Transformers": v1,
            "Attention Paper": v2,
        })

        report = link.link_notes(
            root=self.test_path,
            model=mock_model,
            threshold=0.75,
        )

        # Verification of execution metrics
        self.assertEqual(report.notes_examined, 2)
        self.assertEqual(report.comparisons_performed, 1)
        self.assertEqual(report.links_created, 2)  # note1 -> note2 and note2 -> note1
        self.assertEqual(report.notes_skipped, 0)
        self.assertEqual(len(report.errors), 0)

        # Verify disk persistence
        notes_on_disk = {n.id: n for n in storage.read_wiki_notes(root=self.test_path)}
        self.assertIn("note0002", notes_on_disk["note0001"].links)
        self.assertIn("[[note0002]]", notes_on_disk["note0001"].body)

        self.assertIn("note0001", notes_on_disk["note0002"].links)
        self.assertIn("[[note0001]]", notes_on_disk["note0002"].body)

        # Verify embeddings were saved to embeddings.pkl
        saved_emb = emb.load_embeddings(root=self.test_path)
        self.assertIn("note0001", saved_emb)
        self.assertIn("note0002", saved_emb)

    # -------------------------------------------------------------------------
    # 2. Unrelated Notes (< 0.75)
    # -------------------------------------------------------------------------

    def test_02_two_unrelated_notes_not_linked(self) -> None:
        """Two orthogonal notes (cosine similarity 0.0) are not linked."""
        v_ml = _make_unit_vector(0.0)
        v_cooking = _make_unit_vector(float(np.pi / 2))  # orthogonal: cos = 0.0

        note1 = WikiNote(
            id="note_ml",
            raw_id="2026-07-06_note_ml",
            para="Projects",
            tags=["ml"],
            summary="Machine learning models",
            created="2026-07-06T10:00:00Z",
            links=[],
            body="# Machine Learning\nTraining gradient boosted decision trees.",
        )
        note2 = WikiNote(
            id="note_cook",
            raw_id="2026-07-06_note_cook",
            para="Areas",
            tags=["recipes"],
            summary="Italian pasta recipe",
            created="2026-07-06T11:00:00Z",
            links=[],
            body="# Carbonara Recipe\nEgg yolks, pecorino cheese, guanciale.",
        )

        storage.write_wiki_note(note1, root=self.test_path)
        storage.write_wiki_note(note2, root=self.test_path)

        mock_model = _make_mock_model_with_vectors({
            "Machine Learning": v_ml,
            "Carbonara Recipe": v_cooking,
        })

        report = link.link_notes(
            root=self.test_path,
            model=mock_model,
            threshold=0.75,
        )

        self.assertEqual(report.notes_examined, 2)
        self.assertEqual(report.comparisons_performed, 1)
        self.assertEqual(report.links_created, 0)
        self.assertEqual(len(report.errors), 0)

        # Verify no links created on disk
        notes_on_disk = {n.id: n for n in storage.read_wiki_notes(root=self.test_path)}
        self.assertEqual(notes_on_disk["note_ml"].links, [])
        self.assertNotIn("[[note_cook]]", notes_on_disk["note_ml"].body)
        self.assertEqual(notes_on_disk["note_cook"].links, [])
        self.assertNotIn("[[note_ml]]", notes_on_disk["note_cook"].body)

    # -------------------------------------------------------------------------
    # 3. Exact Threshold (Boundary Test: == 0.75)
    # -------------------------------------------------------------------------

    def test_03_similarity_exactly_at_threshold_linked(self) -> None:
        """Notes with similarity exactly 0.7500 MUST be linked (inclusive threshold >= 0.75)."""
        theta = float(np.arccos(0.75))
        v1 = _make_unit_vector(0.0)
        v2 = _make_unit_vector(theta)

        # Verify exact mathematical cosine similarity
        sim = emb.cosine_similarity(v1, v2)
        self.assertAlmostEqual(sim, 0.75, places=5)
        self.assertGreaterEqual(sim, 0.75)

        note1 = WikiNote(
            id="bound_a",
            raw_id="2026-07-06_bound_a",
            para="Projects",
            tags=["boundary"],
            summary="Note A at threshold",
            created="2026-07-06T10:00:00Z",
            links=[],
            body="Content for Note A boundary test.",
        )
        note2 = WikiNote(
            id="bound_b",
            raw_id="2026-07-06_bound_b",
            para="Projects",
            tags=["boundary"],
            summary="Note B at threshold",
            created="2026-07-06T11:00:00Z",
            links=[],
            body="Content for Note B boundary test.",
        )

        storage.write_wiki_note(note1, root=self.test_path)
        storage.write_wiki_note(note2, root=self.test_path)

        mock_model = _make_mock_model_with_vectors({
            "Note A": v1,
            "Note B": v2,
        })

        report = link.link_notes(
            root=self.test_path,
            model=mock_model,
            threshold=0.75,
        )

        self.assertEqual(report.links_created, 2)
        notes = {n.id: n for n in storage.read_wiki_notes(root=self.test_path)}
        self.assertIn("bound_b", notes["bound_a"].links)
        self.assertIn("bound_a", notes["bound_b"].links)

    # -------------------------------------------------------------------------
    # 4. Just Below Threshold (Boundary Test: < 0.75)
    # -------------------------------------------------------------------------

    def test_04_similarity_below_threshold_not_linked(self) -> None:
        """Notes with similarity 0.7499 must NOT be linked."""
        theta = float(np.arccos(0.7499))
        v1 = _make_unit_vector(0.0)
        v2 = _make_unit_vector(theta)

        sim = emb.cosine_similarity(v1, v2)
        self.assertLess(sim, 0.75)

        note1 = WikiNote(
            id="sub_a",
            raw_id="2026-07-06_sub_a",
            para="Resources",
            tags=["boundary"],
            summary="Sub-threshold Note A",
            created="2026-07-06T10:00:00Z",
            links=[],
            body="Content A just below threshold.",
        )
        note2 = WikiNote(
            id="sub_b",
            raw_id="2026-07-06_sub_b",
            para="Resources",
            tags=["boundary"],
            summary="Sub-threshold Note B",
            created="2026-07-06T11:00:00Z",
            links=[],
            body="Content B just below threshold.",
        )

        storage.write_wiki_note(note1, root=self.test_path)
        storage.write_wiki_note(note2, root=self.test_path)

        mock_model = _make_mock_model_with_vectors({
            "Sub-threshold Note A": v1,
            "Sub-threshold Note B": v2,
        })

        report = link.link_notes(
            root=self.test_path,
            model=mock_model,
            threshold=0.75,
        )

        self.assertEqual(report.links_created, 0)
        notes = {n.id: n for n in storage.read_wiki_notes(root=self.test_path)}
        self.assertEqual(notes["sub_a"].links, [])
        self.assertEqual(notes["sub_b"].links, [])

    # -------------------------------------------------------------------------
    # 5. Self-Comparison Prevention
    # -------------------------------------------------------------------------

    def test_05_note_compared_with_itself_prevented(self) -> None:
        """A note is never compared with or linked to itself."""
        v = _make_unit_vector(0.0)

        note = WikiNote(
            id="self001",
            raw_id="2026-07-06_self001",
            para="Projects",
            tags=["solo"],
            summary="Single note alone",
            created="2026-07-06T10:00:00Z",
            links=[],
            body="# Solo Note\nI am the only note.",
        )
        storage.write_wiki_note(note, root=self.test_path)

        mock_model = _make_mock_model_with_vectors({"Solo Note": v})

        report = link.link_notes(
            root=self.test_path,
            model=mock_model,
            threshold=0.75,
        )

        self.assertEqual(report.notes_examined, 1)
        self.assertEqual(report.comparisons_performed, 0)
        self.assertEqual(report.links_created, 0)

        # Direct link_pair check
        self.assertFalse(link.link_pair(note, "self001"))
        self.assertEqual(note.links, [])
        self.assertNotIn("[[self001]]", note.body)

    # -------------------------------------------------------------------------
    # 6. Duplicate Relationship Prevention
    # -------------------------------------------------------------------------

    def test_06_duplicate_relationship_prevention(self) -> None:
        """Existing relationships are not duplicated in frontmatter or body."""
        theta = float(np.arccos(0.90))
        v1 = _make_unit_vector(0.0)
        v2 = _make_unit_vector(theta)

        # Note1 already links to Note2 in frontmatter and body
        note1 = WikiNote(
            id="dup001",
            raw_id="2026-07-06_dup001",
            para="Projects",
            tags=["test"],
            summary="Duplicate test note 1",
            created="2026-07-06T10:00:00Z",
            links=["dup002"],
            body="Existing body text.\n\n[[dup002]]\n",
        )
        note2 = WikiNote(
            id="dup002",
            raw_id="2026-07-06_dup002",
            para="Projects",
            tags=["test"],
            summary="Duplicate test note 2",
            created="2026-07-06T11:00:00Z",
            links=["dup001"],
            body="Existing body text.\n\n[[dup001]]\n",
        )

        storage.write_wiki_note(note1, root=self.test_path)
        storage.write_wiki_note(note2, root=self.test_path)

        mock_model = _make_mock_model_with_vectors({
            "Duplicate test note 1": v1,
            "Duplicate test note 2": v2,
        })

        report = link.link_notes(
            root=self.test_path,
            model=mock_model,
            threshold=0.75,
        )

        self.assertEqual(report.links_created, 0)
        self.assertEqual(report.links_preserved, 2)

        notes = {n.id: n for n in storage.read_wiki_notes(root=self.test_path)}
        # Must still contain exactly one entry
        self.assertEqual(notes["dup001"].links, ["dup002"])
        self.assertEqual(notes["dup001"].body.count("[[dup002]]"), 1)

        self.assertEqual(notes["dup002"].links, ["dup001"])
        self.assertEqual(notes["dup002"].body.count("[[dup001]]"), 1)

    # -------------------------------------------------------------------------
    # 7. Existing Relationship Preservation
    # -------------------------------------------------------------------------

    def test_07_existing_unrelated_links_and_metadata_preserved(self) -> None:
        """Existing unrelated manual links and note metadata are strictly preserved."""
        theta = float(np.arccos(0.90))
        v1 = _make_unit_vector(0.0)
        v2 = _make_unit_vector(theta)

        # Note 1 already has manual links to 'manual_x' and 'manual_y'
        note1 = WikiNote(
            id="pres001",
            raw_id="2026-07-06_pres001",
            para="Areas",
            tags=["finance", "tax"],
            summary="Financial planning 2026",
            created="2026-07-06T10:00:00Z",
            links=["manual_x", "manual_y"],
            body="Planning investments. See also [[manual_x]].",
        )
        note2 = WikiNote(
            id="pres002",
            raw_id="2026-07-06_pres002",
            para="Areas",
            tags=["finance", "savings"],
            summary="Tax efficient savings strategy",
            created="2026-07-06T11:00:00Z",
            links=[],
            body="Maximizing retirement account contributions.",
        )

        storage.write_wiki_note(note1, root=self.test_path)
        storage.write_wiki_note(note2, root=self.test_path)

        mock_model = _make_mock_model_with_vectors({
            "Financial planning 2026": v1,
            "Tax efficient savings": v2,
        })

        report = link.link_notes(
            root=self.test_path,
            model=mock_model,
            threshold=0.75,
        )

        self.assertGreaterEqual(report.links_created, 1)

        notes = {n.id: n for n in storage.read_wiki_notes(root=self.test_path)}
        n1 = notes["pres001"]

        # Metadata preserved
        self.assertEqual(n1.id, "pres001")
        self.assertEqual(n1.raw_id, "2026-07-06_pres001")
        self.assertEqual(n1.para, "Areas")
        self.assertEqual(n1.tags, ["finance", "tax"])
        self.assertEqual(n1.summary, "Financial planning 2026")
        self.assertEqual(n1.created, "2026-07-06T10:00:00Z")

        # Unrelated links preserved + new link appended
        self.assertIn("manual_x", n1.links)
        self.assertIn("manual_y", n1.links)
        self.assertIn("pres002", n1.links)
        self.assertIn("[[manual_x]]", n1.body)
        self.assertIn("[[pres002]]", n1.body)

    # -------------------------------------------------------------------------
    # 8. Missing or Invalid Embeddings Handling
    # -------------------------------------------------------------------------

    def test_08_empty_note_content_skipped_safely(self) -> None:
        """A note with empty summary and empty body is skipped with error recorded."""
        empty_note = WikiNote(
            id="empty001",
            raw_id="2026-07-06_empty001",
            para="Resources",
            tags=[],
            summary="",
            created="2026-07-06T10:00:00Z",
            links=[],
            body="   \n\t  ",
        )
        storage.write_wiki_note(empty_note, root=self.test_path)

        report = link.link_notes(root=self.test_path, threshold=0.75)
        self.assertEqual(report.notes_examined, 1)
        self.assertEqual(report.notes_skipped, 1)
        self.assertEqual(report.links_created, 0)
        self.assertTrue(any("empty content" in e.lower() for e in report.errors))

    def test_09_non_finite_embedding_values_handled(self) -> None:
        """Embedding generation producing NaN/Inf is skipped safely."""
        bad_note = WikiNote(
            id="nan001",
            raw_id="2026-07-06_nan001",
            para="Projects",
            tags=["nan"],
            summary="Note causing NaN embedding",
            created="2026-07-06T10:00:00Z",
            links=[],
            body="Causes model to return NaN.",
        )
        storage.write_wiki_note(bad_note, root=self.test_path)

        bad_model = MagicMock()
        nan_vec = np.zeros(384, dtype=np.float32)
        nan_vec[0] = np.nan
        bad_model.encode.return_value = nan_vec

        report = link.link_notes(root=self.test_path, model=bad_model, threshold=0.75)
        self.assertEqual(report.notes_skipped, 1)
        self.assertEqual(report.links_created, 0)
        self.assertTrue(any("non-finite" in e.lower() for e in report.errors))

    def test_10_corrupted_embeddings_file_recovers_gracefully(self) -> None:
        """Corrupted embeddings.pkl does not crash linker; cache resets and recomputes."""
        theta = float(np.arccos(0.90))
        v1 = _make_unit_vector(0.0)
        v2 = _make_unit_vector(theta)

        note1 = WikiNote(
            id="rec001",
            raw_id="2026-07-06_rec001",
            para="Projects",
            tags=["recovery"],
            summary="Recovery test note 1",
            created="2026-07-06T10:00:00Z",
            links=[],
            body="# Recovery 1\nImportant notes for recovery.",
        )
        note2 = WikiNote(
            id="rec002",
            raw_id="2026-07-06_rec002",
            para="Projects",
            tags=["recovery"],
            summary="Recovery test note 2",
            created="2026-07-06T11:00:00Z",
            links=[],
            body="# Recovery 2\nImportant notes for recovery.",
        )
        storage.write_wiki_note(note1, root=self.test_path)
        storage.write_wiki_note(note2, root=self.test_path)

        # Write corrupted garbage bytes to data/embeddings.pkl
        pkl_path = emb.get_embeddings_path(root=self.test_path)
        pkl_path.write_bytes(b"CORRUPTED_BYTES_NOT_PICKLE")

        mock_model = _make_mock_model_with_vectors({
            "Recovery 1": v1,
            "Recovery 2": v2,
        })

        report = link.link_notes(root=self.test_path, model=mock_model, threshold=0.75)

        # Successful recovery: notes were examined, recomputed, and linked
        self.assertEqual(report.notes_examined, 2)
        self.assertEqual(report.links_created, 2)
        self.assertTrue(any("corrupted" in e.lower() for e in report.errors))

        # embeddings.pkl is now valid again
        valid_loaded = emb.load_embeddings(root=self.test_path)
        self.assertIn("rec001", valid_loaded)
        self.assertIn("rec002", valid_loaded)

    # -------------------------------------------------------------------------
    # 9. Empty Note Collections
    # -------------------------------------------------------------------------

    def test_11_empty_note_collection_returns_clean_report(self) -> None:
        """Linking with no wiki notes returns empty report with zero errors."""
        report = link.link_notes(root=self.test_path, threshold=0.75)

        self.assertEqual(report.notes_examined, 0)
        self.assertEqual(report.comparisons_performed, 0)
        self.assertEqual(report.links_created, 0)
        self.assertEqual(report.links_preserved, 0)
        self.assertEqual(report.notes_skipped, 0)
        self.assertEqual(report.errors, [])

    # -------------------------------------------------------------------------
    # 10. Repeated Execution (Idempotency)
    # -------------------------------------------------------------------------

    def test_12_repeated_execution_is_strictly_idempotent(self) -> None:
        """Running linker multiple times does not duplicate links or mutate text."""
        theta = float(np.arccos(0.85))
        v1 = _make_unit_vector(0.0)
        v2 = _make_unit_vector(theta)

        note1 = WikiNote(
            id="idem001",
            raw_id="2026-07-06_idem001",
            para="Projects",
            tags=["idempotency"],
            summary="First idempotency test note",
            created="2026-07-06T10:00:00Z",
            links=[],
            body="# Idempotency Test 1\nEnsuring consistency across runs.",
        )
        note2 = WikiNote(
            id="idem002",
            raw_id="2026-07-06_idem002",
            para="Projects",
            tags=["idempotency"],
            summary="Second idempotency test note",
            created="2026-07-06T11:00:00Z",
            links=[],
            body="# Idempotency Test 2\nEnsuring consistency across runs.",
        )

        storage.write_wiki_note(note1, root=self.test_path)
        storage.write_wiki_note(note2, root=self.test_path)

        mock_model = _make_mock_model_with_vectors({
            "Idempotency Test 1": v1,
            "Idempotency Test 2": v2,
        })

        # Run 1: Links created
        report1 = link.link_notes(root=self.test_path, model=mock_model, threshold=0.75)
        self.assertEqual(report1.links_created, 2)
        self.assertEqual(report1.links_preserved, 0)

        # Snapshot note files on disk
        file1_path = self.test_path / "wiki" / "Projects" / "idem001.md"
        file2_path = self.test_path / "wiki" / "Projects" / "idem002.md"
        content1_run1 = file1_path.read_text(encoding="utf-8")
        content2_run1 = file2_path.read_text(encoding="utf-8")

        # Run 2: Exact same notes, must create 0 new links and preserve 2
        report2 = link.link_notes(root=self.test_path, model=mock_model, threshold=0.75)
        self.assertEqual(report2.links_created, 0)
        self.assertEqual(report2.links_preserved, 2)

        content1_run2 = file1_path.read_text(encoding="utf-8")
        content2_run2 = file2_path.read_text(encoding="utf-8")

        # Byte-for-byte identical content
        self.assertEqual(content1_run1, content1_run2)
        self.assertEqual(content2_run1, content2_run2)

        # Run 3: Dry run also produces identical preservation
        report3 = link.link_notes(
            root=self.test_path, model=mock_model, threshold=0.75, dry_run=True
        )
        self.assertEqual(report3.links_created, 0)
        self.assertEqual(report3.links_preserved, 2)

    # -------------------------------------------------------------------------
    # 11. Persistence & Multi-Note Topologies
    # -------------------------------------------------------------------------

    def test_13_multi_note_cluster_linking(self) -> None:
        """Three related notes all link to each other (triangle topology)."""
        # Angles such that all mutual similarities are >= 0.85
        # cos(0.2) ≈ 0.98, cos(0.4) ≈ 0.92
        v1 = _make_unit_vector(0.0)
        v2 = _make_unit_vector(0.2)
        v3 = _make_unit_vector(0.4)

        for i, (nid, v) in enumerate([("tri1", v1), ("tri2", v2), ("tri3", v3)], 1):
            n = WikiNote(
                id=nid,
                raw_id=f"2026-07-06_{nid}",
                para="Projects",
                tags=["cluster"],
                summary=f"Cluster note {i}",
                created="2026-07-06T10:00:00Z",
                links=[],
                body=f"# Cluster {i}\nContent for note {i}.",
            )
            storage.write_wiki_note(n, root=self.test_path)

        mock_model = _make_mock_model_with_vectors({
            "Cluster 1": v1,
            "Cluster 2": v2,
            "Cluster 3": v3,
        })

        report = link.link_notes(root=self.test_path, model=mock_model, threshold=0.75)

        self.assertEqual(report.notes_examined, 3)
        self.assertEqual(report.comparisons_performed, 3)  # C(3, 2) = 3
        self.assertEqual(report.links_created, 6)  # 3 pairs * 2 directions = 6

        notes = {n.id: n for n in storage.read_wiki_notes(root=self.test_path)}
        self.assertEqual(set(notes["tri1"].links), {"tri2", "tri3"})
        self.assertEqual(set(notes["tri2"].links), {"tri1", "tri3"})
        self.assertEqual(set(notes["tri3"].links), {"tri1", "tri2"})

    # -------------------------------------------------------------------------
    # 12. Single-Note Focused Linking (--id)
    # -------------------------------------------------------------------------

    def test_14_link_specific_note_by_id(self) -> None:
        """Targeting a single note via note_id links only that note's relationships."""
        theta = float(np.arccos(0.90))
        v1 = _make_unit_vector(0.0)
        v2 = _make_unit_vector(theta)

        note1 = WikiNote(
            id="focus1",
            raw_id="2026-07-06_focus1",
            para="Projects",
            tags=["focus"],
            summary="Focus note 1",
            created="2026-07-06T10:00:00Z",
            links=[],
            body="# Focus 1\nPrimary target.",
        )
        note2 = WikiNote(
            id="focus2",
            raw_id="2026-07-06_focus2",
            para="Projects",
            tags=["focus"],
            summary="Focus note 2",
            created="2026-07-06T11:00:00Z",
            links=[],
            body="# Focus 2\nSecondary note.",
        )
        storage.write_wiki_note(note1, root=self.test_path)
        storage.write_wiki_note(note2, root=self.test_path)

        mock_model = _make_mock_model_with_vectors({
            "Focus 1": v1,
            "Focus 2": v2,
        })

        # Link focusing specifically on focus1
        report = link.link_notes(
            root=self.test_path,
            model=mock_model,
            threshold=0.75,
            note_id="focus1",
        )

        self.assertEqual(report.notes_examined, 1)
        self.assertEqual(report.comparisons_performed, 1)
        self.assertIn("focus2", report.new_links[0])

        # Non-existent ID returns error
        err_report = link.link_notes(
            root=self.test_path,
            model=mock_model,
            note_id="nonexistent_id",
        )
        self.assertTrue(any("not found" in e for e in err_report.errors))

    # -------------------------------------------------------------------------
    # 13. Text Helpers & Formatting Tests
    # -------------------------------------------------------------------------

    def test_15_append_wikilinks_formatting_and_deduplication(self) -> None:
        """append_wikilink_to_body and append_wikilinks_to_body format cleanly."""
        # Empty body
        body0 = ""
        res0 = link.append_wikilink_to_body(body0, "note01")
        self.assertEqual(res0, "[[note01]]\n")

        # Non-empty body without trailing wikilink
        body1 = "Intro text."
        res1 = link.append_wikilink_to_body(body1, "note01")
        self.assertEqual(res1, "Intro text.\n\n[[note01]]\n")

        # Body ending with existing wikilink
        res2 = link.append_wikilink_to_body(res1, "note02")
        self.assertEqual(res2, "Intro text.\n\n[[note01]]\n[[note02]]\n")

        # Already present -> no modification
        res3 = link.append_wikilink_to_body(res2, "note01")
        self.assertEqual(res3, res2)

        # Batch append
        batch_res = link.append_wikilinks_to_body("Heading", ["id1", "id2"])
        self.assertEqual(batch_res, "Heading\n\n[[id1]]\n[[id2]]\n")

    def test_16_strip_appended_wikilinks_preserves_core_content(self) -> None:
        """strip_appended_wikilinks removes trailing auto-links while keeping core body."""
        body = "# Header\nSome text with [[mid_link]] inside.\n\n[[auto_1]]\n[[auto_2]]\n"
        stripped = link.strip_appended_wikilinks(body)
        self.assertEqual(stripped, "# Header\nSome text with [[mid_link]] inside.")

    # -------------------------------------------------------------------------
    # 14. CLI Tests
    # -------------------------------------------------------------------------

    def test_17_cli_help_flag(self) -> None:
        """link.py --help prints usage and exits cleanly with code 0."""
        stdout_buf = io.StringIO()
        with redirect_stdout(stdout_buf):
            exit_code = link.main(["--help"])
        self.assertEqual(exit_code, 0)
        self.assertIn("Auto-link semantically related", stdout_buf.getvalue())
        self.assertIn("--threshold", stdout_buf.getvalue())

    def test_18_cli_dry_run_flag(self) -> None:
        """link.py --dry-run prints preview without writing files to disk."""
        theta = float(np.arccos(0.90))
        v1 = _make_unit_vector(0.0)
        v2 = _make_unit_vector(theta)

        note1 = WikiNote(
            id="cli001",
            raw_id="2026-07-06_cli001",
            para="Projects",
            tags=["cli"],
            summary="CLI note 1",
            created="2026-07-06T10:00:00Z",
            links=[],
            body="# CLI 1\nTesting dry run CLI.",
        )
        note2 = WikiNote(
            id="cli002",
            raw_id="2026-07-06_cli002",
            para="Projects",
            tags=["cli"],
            summary="CLI note 2",
            created="2026-07-06T11:00:00Z",
            links=[],
            body="# CLI 2\nTesting dry run CLI.",
        )
        storage.write_wiki_note(note1, root=self.test_path)
        storage.write_wiki_note(note2, root=self.test_path)

        mock_model = _make_mock_model_with_vectors({
            "CLI 1": v1,
            "CLI 2": v2,
        })

        stdout_buf = io.StringIO()
        with patch("lib.embeddings.load_model", return_value=mock_model):
            with redirect_stdout(stdout_buf):
                exit_code = link.main([
                    "--root", str(self.test_path),
                    "--dry-run",
                ])

        self.assertEqual(exit_code, 0)
        out = stdout_buf.getvalue()
        self.assertIn("[dry-run]", out)
        self.assertIn("cli001 <-> cli002", out)

        # Confirm files on disk were NOT modified
        notes = {n.id: n for n in storage.read_wiki_notes(root=self.test_path)}
        self.assertEqual(notes["cli001"].links, [])
        self.assertEqual(notes["cli002"].links, [])

    # -------------------------------------------------------------------------
    # 15. Tuned Threshold (0.65) Calibration Tests (Task 2.2.3)
    # -------------------------------------------------------------------------

    def test_19_tuned_default_threshold_is_0_65(self) -> None:
        """DEFAULT_SIMILARITY_THRESHOLD is configured to 0.65 per Task 2.2.3 tuning."""
        self.assertEqual(link.DEFAULT_SIMILARITY_THRESHOLD, 0.65)

    def test_20_similarity_at_tuned_threshold_0_65_boundary(self) -> None:
        """Default threshold (0.65) links notes with similarity exactly 0.6500."""
        theta = float(np.arccos(0.65))
        v1 = _make_unit_vector(0.0)
        v2 = _make_unit_vector(theta)

        sim = emb.cosine_similarity(v1, v2)
        self.assertAlmostEqual(sim, 0.65, places=5)
        self.assertGreaterEqual(sim, 0.65)

        note1 = WikiNote(
            id="tuned_a",
            raw_id="2026-07-06_tuned_a",
            para="Projects",
            tags=["tuning"],
            summary="Tuned note A at 0.65 threshold",
            created="2026-07-06T10:00:00Z",
            links=[],
            body="Content for tuned note A.",
        )
        note2 = WikiNote(
            id="tuned_b",
            raw_id="2026-07-06_tuned_b",
            para="Projects",
            tags=["tuning"],
            summary="Tuned note B at 0.65 threshold",
            created="2026-07-06T11:00:00Z",
            links=[],
            body="Content for tuned note B.",
        )
        storage.write_wiki_note(note1, root=self.test_path)
        storage.write_wiki_note(note2, root=self.test_path)

        mock_model = _make_mock_model_with_vectors({
            "Tuned note A": v1,
            "Tuned note B": v2,
        })

        # Run with default threshold (no explicit threshold passed)
        report = link.link_notes(root=self.test_path, model=mock_model)
        self.assertEqual(report.links_created, 2)

        notes = {n.id: n for n in storage.read_wiki_notes(root=self.test_path)}
        self.assertIn("tuned_b", notes["tuned_a"].links)
        self.assertIn("tuned_a", notes["tuned_b"].links)

    def test_21_similarity_below_tuned_threshold_0_65_rejected(self) -> None:
        """Default threshold (0.65) rejects notes with similarity 0.6499."""
        theta = float(np.arccos(0.6499))
        v1 = _make_unit_vector(0.0)
        v2 = _make_unit_vector(theta)

        sim = emb.cosine_similarity(v1, v2)
        self.assertLess(sim, 0.65)

        note1 = WikiNote(
            id="sub65_a",
            raw_id="2026-07-06_sub65_a",
            para="Projects",
            tags=["tuning"],
            summary="Sub-0.65 note A",
            created="2026-07-06T10:00:00Z",
            links=[],
            body="Content for sub-0.65 note A.",
        )
        note2 = WikiNote(
            id="sub65_b",
            raw_id="2026-07-06_sub65_b",
            para="Projects",
            tags=["tuning"],
            summary="Sub-0.65 note B",
            created="2026-07-06T11:00:00Z",
            links=[],
            body="Content for sub-0.65 note B.",
        )
        storage.write_wiki_note(note1, root=self.test_path)
        storage.write_wiki_note(note2, root=self.test_path)

        mock_model = _make_mock_model_with_vectors({
            "Sub-0.65 note A": v1,
            "Sub-0.65 note B": v2,
        })

        # Run with default threshold
        report = link.link_notes(root=self.test_path, model=mock_model)
        self.assertEqual(report.links_created, 0)

        notes = {n.id: n for n in storage.read_wiki_notes(root=self.test_path)}
        self.assertEqual(notes["sub65_a"].links, [])
        self.assertEqual(notes["sub65_b"].links, [])

    def test_22_candidate_thresholds_calibration_comparison(self) -> None:
        """Compare candidate thresholds 0.65, 0.75, and 0.80 across calibrated test pairs."""
        # Pair with sim = 0.70 (linked at 0.65, not linked at 0.75 or 0.80)
        theta_70 = float(np.arccos(0.70))
        v_base = _make_unit_vector(0.0)
        v_70 = _make_unit_vector(theta_70)

        note1 = WikiNote(
            id="calib1",
            raw_id="2026-07-06_calib1",
            para="Projects",
            tags=["calib"],
            summary="Calibration Note 1",
            created="2026-07-06T10:00:00Z",
            links=[],
            body="Base content for calibration.",
        )
        note2 = WikiNote(
            id="calib2",
            raw_id="2026-07-06_calib2",
            para="Projects",
            tags=["calib"],
            summary="Calibration Note 2",
            created="2026-07-06T11:00:00Z",
            links=[],
            body="Related content with sim 0.70.",
        )
        storage.write_wiki_note(note1, root=self.test_path)
        storage.write_wiki_note(note2, root=self.test_path)

        mock_model = _make_mock_model_with_vectors({
            "Calibration Note 1": v_base,
            "Calibration Note 2": v_70,
        })

        # At threshold 0.80 -> 0 links
        rep_80 = link.link_notes(root=self.test_path, model=mock_model, threshold=0.80, dry_run=True)
        self.assertEqual(rep_80.links_created, 0)

        # At threshold 0.75 -> 0 links
        rep_75 = link.link_notes(root=self.test_path, model=mock_model, threshold=0.75, dry_run=True)
        self.assertEqual(rep_75.links_created, 0)

        # At tuned threshold 0.65 -> 2 links
        rep_65 = link.link_notes(root=self.test_path, model=mock_model, threshold=0.65, dry_run=True)
        self.assertEqual(rep_65.links_created, 2)


if __name__ == "__main__":
    unittest.main()

