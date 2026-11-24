"""SecondSelf — Frontend Artifact & Contract Verification Tests (Task 3.2).

Ensures that the static frontend scaffold in public/ conforms to all architectural,
accessibility, and schema requirements without external dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def public_dir() -> Path:
    """Fixture providing path to public/ directory."""
    path = Path("public")
    assert path.is_dir(), "public/ directory must exist"
    return path


def test_public_directory_structure(public_dir: Path) -> None:
    """Verifies that all required files and subdirectories exist in public/."""
    expected_files = [
        public_dir / "index.html",
        public_dir / "css" / "main.css",
        public_dir / "js" / "app.js",
        public_dir / "js" / "dataLoader.js",
        public_dir / "js" / "graph.js",
        public_dir / "js" / "filters.js",
        public_dir / "js" / "detailPanel.js",
        public_dir / "data" / "graph.json",
        public_dir / "graph.json",
        public_dir / "README.md",
    ]

    for f in expected_files:
        assert f.is_file(), f"Expected static frontend asset missing: {f}"


def test_index_html_semantic_structure(public_dir: Path) -> None:
    """Verifies index.html semantic landmarks, accessibility attributes, and ES module script."""
    html_content = (public_dir / "index.html").read_text(encoding="utf-8")

    # Semantic landmarks
    assert "<!DOCTYPE html>" in html_content
    assert '<html lang="en">' in html_content
    assert "<header" in html_content
    assert "<main" in html_content
    assert "<aside" in html_content

    # Accessibility & ARIA landmarks
    assert 'aria-label=' in html_content
    assert 'role="banner"' in html_content
    assert 'role="dialog"' in html_content or 'role="region"' in html_content
    assert 'aria-live="polite"' in html_content
    assert 'aria-pressed=' in html_content

    # ES Module script inclusion
    assert '<script type="module" src="js/app.js"></script>' in html_content

    # CSS inclusion
    assert 'href="css/main.css"' in html_content


def test_css_design_tokens_and_categories(public_dir: Path) -> None:
    """Verifies CSS tokens for PARA categories and accessibility features."""
    css_content = (public_dir / "css" / "main.css").read_text(encoding="utf-8")

    # PARA color tokens
    assert "--color-project:" in css_content
    assert "--color-area:" in css_content
    assert "--color-resource:" in css_content
    assert "--color-archive:" in css_content

    # Accessible focus rings and screen reader utilities
    assert ":focus-visible" in css_content
    assert ".sr-only" in css_content

    # Responsive media queries
    assert "@media (max-width: 768px)" in css_content
    assert "@media (max-width: 1024px)" in css_content


def test_data_contract_graph_json(public_dir: Path) -> None:
    """Verifies that public/data/graph.json complies with the Task 3.1 node-link schema."""
    graph_path = public_dir / "data" / "graph.json"
    assert graph_path.is_file()

    with open(graph_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Root metadata
    assert "schema_version" in data
    assert "statistics" in data
    assert "nodes" in data
    assert "edges" in data

    # Statistics
    stats = data["statistics"]
    assert stats["total_nodes"] > 0
    assert stats["total_edges"] > 0

    # Nodes verification
    nodes = data["nodes"]
    assert len(nodes) == stats["total_nodes"]
    node_ids = set()

    for node in nodes:
        assert "id" in node
        assert "label" in node
        assert "type" in node
        assert "properties" in node
        node_ids.add(node["id"])

    # Edges verification
    edges = data["edges"]
    assert len(edges) == stats["total_edges"]
    for edge in edges:
        from_id = edge.get("from") or edge.get("source")
        to_id = edge.get("to") or edge.get("target")
        assert from_id in node_ids, f"Edge references missing endpoint {from_id}"
        assert to_id in node_ids, f"Edge references missing endpoint {to_id}"
        assert "type" in edge


def test_readme_documentation_completeness(public_dir: Path) -> None:
    """Verifies that README.md documents serving commands, QA checklists, and deployment."""
    readme_content = (public_dir / "README.md").read_text(encoding="utf-8")

    assert "python -m http.server 8000 --directory public" in readme_content
    assert "Data Contract" in readme_content
    assert "Accessibility (WCAG 2.1 AA)" in readme_content
    assert "Manual QA & Testing Checklist" in readme_content
    assert "GitHub Pages" in readme_content


def test_qa_search_and_citations_assets(public_dir: Path) -> None:
    """Verifies Task 4.3 conversational Q&A search and citation highlighting assets."""
    # Check modular JavaScript files
    assert (public_dir / "js" / "search.js").is_file()
    assert (public_dir / "js" / "citations.js").is_file()
    assert (public_dir / "js" / "highlight.js").is_file()

    # Check HTML markup elements
    html_content = (public_dir / "index.html").read_text(encoding="utf-8")
    assert 'id="qa-search-form"' in html_content
    assert 'id="qa-search-input"' in html_content
    assert 'id="qa-submit-btn"' in html_content
    assert 'id="qa-status-live"' in html_content
    assert 'id="qa-error-box"' in html_content
    assert 'id="qa-answer-container"' in html_content

    # Check CSS class styling
    css_content = (public_dir / "css" / "main.css").read_text(encoding="utf-8")
    assert ".qa-search-section" in css_content
    assert ".qa-answer-container" in css_content
    assert ".inline-citation" in css_content
    assert ".citation-card" in css_content

