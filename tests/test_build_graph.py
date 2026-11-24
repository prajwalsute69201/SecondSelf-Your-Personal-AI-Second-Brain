"""SecondSelf — CLI Integration Tests for build_graph.py."""

from __future__ import annotations

from pathlib import Path
from build_graph import main


def test_build_graph_cli_dry_run_with_sample_vault(tmp_path: Path) -> None:
    wiki_dir = tmp_path / "wiki"
    (wiki_dir / "Projects").mkdir(parents=True, exist_ok=True)
    (wiki_dir / "Resources").mkdir(parents=True, exist_ok=True)

    note1 = wiki_dir / "Projects" / "p1.md"
    note1.write_text(
        "---\ntitle: Project One\ncategory: Projects\nconfidence: 1.0\n---\n# Project One\nContent\n",
        encoding="utf-8",
    )

    out_file = tmp_path / "graph.json"

    exit_code = main([
        "--wiki-dir", str(wiki_dir),
        "--output", str(out_file),
        "--format", "all",
        "--stats",
    ])

    assert exit_code == 0
    assert out_file.exists()
    assert (tmp_path / "graph.cypher").exists()
    assert (tmp_path / "graph.graphml").exists()
