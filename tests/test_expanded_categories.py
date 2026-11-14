"""Unit and integration tests for expanded knowledge graph categories in SecondSelf."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import app
import build_graph
from lib.constants import (
    CATEGORY_DEFINITIONS,
    DEFAULT_USER_ID,
    VALID_CATEGORIES,
    VALID_PARA_CATEGORIES,
)
from lib.models import WikiNote
import lib.storage as storage
import validate_graph


class TestExpandedCategoriesConstants(unittest.TestCase):
    """Test category constants and metadata configuration."""

    def test_all_14_categories_present(self) -> None:
        expected = {
            "Projects",
            "Areas",
            "Resources",
            "Archives",
            "Work",
            "Payments",
            "Finance",
            "Personal",
            "Learning",
            "Health",
            "People",
            "Ideas",
            "Events",
            "Shopping",
        }
        self.assertEqual(set(VALID_CATEGORIES), expected)
        self.assertEqual(set(VALID_PARA_CATEGORIES), expected)
        self.assertEqual(set(CATEGORY_DEFINITIONS.keys()), expected)

    def test_category_definitions_have_required_keys(self) -> None:
        colors_seen = set()
        for cat, meta in CATEGORY_DEFINITIONS.items():
            self.assertIn("icon", meta, f"Category {cat} missing 'icon'")
            self.assertIn("description", meta, f"Category {cat} missing 'description'")
            self.assertIn("color", meta, f"Category {cat} missing 'color'")
            self.assertIn("border", meta, f"Category {cat} missing 'border'")
            self.assertIn("highlight", meta, f"Category {cat} missing 'highlight'")
            self.assertTrue(meta["color"].startswith("#"), f"Category {cat} color must be hex")
            self.assertNotIn(meta["color"].lower(), colors_seen, f"Color collision for {cat}")
            colors_seen.add(meta["color"].lower())


class TestExpandedCategoriesBuildGraph(unittest.TestCase):
    """Test knowledge graph builder and validator with new categories."""

    def setUp(self) -> None:
        self.test_dir = Path(tempfile.mkdtemp(prefix="secondself_cat_test_"))
        self.wiki_dir = self.test_dir / "wiki"
        self.data_dir = self.test_dir / "data"
        self.wiki_dir.mkdir(parents=True)
        self.data_dir.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_build_nodes_with_new_categories(self) -> None:
        notes = [
            WikiNote(
                id="w01",
                raw_id="r01",
                para="Work",
                tags=["tasks"],
                summary="Sprint Planning",
                created="2026-11-14T10:00:00Z",
                links=[],
                body="Sprint planning notes",
            ),
            WikiNote(
                id="f01",
                raw_id="r02",
                para="Finance",
                tags=["budget"],
                summary="Annual Budget",
                created="2026-11-14T10:01:00Z",
                links=["w01"],
                body="Budget notes referencing [[w01]]",
            ),
            WikiNote(
                id="h01",
                raw_id="r03",
                para="Health",
                tags=["workout"],
                summary="Marathon Training",
                created="2026-11-14T10:02:00Z",
                links=[],
                body="10k run plan",
            ),
        ]

        nodes, warnings = build_graph.build_graph_nodes(notes)
        self.assertEqual(len(nodes), 3)
        self.assertEqual(len(warnings), 0)

        nodes_by_id = {n.id: n for n in nodes}
        self.assertEqual(nodes_by_id["w01"].group, "Work")
        self.assertEqual(nodes_by_id["f01"].group, "Finance")
        self.assertEqual(nodes_by_id["h01"].group, "Health")

        # Validate graph schema accepts all nodes without error
        edges, _ = build_graph.build_graph_edges(notes, nodes)
        errors = build_graph.validate_graph(nodes, edges)
        self.assertEqual(len(errors), 0, f"Graph validation errors: {errors}")

    def test_save_edited_note_reclassifies_to_new_category(self) -> None:
        note = WikiNote(
            id="edit01",
            raw_id="r99",
            para="Projects",
            tags=["old"],
            summary="Old Title",
            created="2026-11-14T09:00:00Z",
            links=[],
            body="Initial note body",
            user_id=DEFAULT_USER_ID,
        )
        storage.write_wiki_note(note, root=self.test_dir)
        old_path = self.wiki_dir / "Projects" / "edit01.md"
        self.assertTrue(old_path.exists())

        # Update note to "Payments"
        with patch("build_graph.build_and_save_graph"):
            with patch("lib.embeddings.embed_text", return_value=[0.1] * 384):
                with patch("lib.embeddings.load_embeddings", return_value={}):
                    with patch("lib.embeddings.save_embeddings"):
                        updated = app.save_edited_note(
                            note_id="edit01",
                            new_para="Payments",
                            new_summary="Subscription Payment",
                            new_tags=["billing"],
                            new_body="Monthly cloud hosting invoice",
                            root=self.test_dir,
                            user_id=DEFAULT_USER_ID,
                        )

        self.assertEqual(updated.para, "Payments")
        self.assertFalse(old_path.exists(), "Old file in Projects should be removed")
        new_path = self.wiki_dir / "Payments" / "edit01.md"
        self.assertTrue(new_path.exists(), "New file in Payments should exist")

        # Verify persisted note in new category
        reloaded = storage.read_wiki_notes(root=self.test_dir, user_id=DEFAULT_USER_ID)
        self.assertEqual(len(reloaded), 1)
        self.assertEqual(reloaded[0].para, "Payments")
        self.assertEqual(reloaded[0].summary, "Subscription Payment")


class TestGraphHtmlTemplate(unittest.TestCase):
    """Test static/graph.html template contains all 14 categories and interactive filtering."""

    def test_graph_html_contains_all_14_categories(self) -> None:
        html_file = _REPO_ROOT / "static" / "graph.html"
        self.assertTrue(html_file.exists())
        content = html_file.read_text(encoding="utf-8")

        for cat in VALID_CATEGORIES:
            self.assertIn(cat, content, f"Category '{cat}' missing from static/graph.html")

        # Verify filter controls and DataView filtering are present
        self.assertIn("btn-filter-all", content)
        self.assertIn("btn-filter-none", content)
        self.assertIn("activeCategories", content)
        self.assertIn("vis.DataView", content)
        self.assertIn("legend-items", content)


if __name__ == "__main__":
    unittest.main()
