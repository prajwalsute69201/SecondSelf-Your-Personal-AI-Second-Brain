"""Unit and integration tests for SecondSelf knowledge capture CLI (capture.py)."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from capture import (
    EXIT_IO_ERROR,
    EXIT_SUCCESS,
    EXIT_USAGE_ERROR,
    capture,
    compute_content_hash,
    extract_file,
    extract_text,
    main,
)


def test_compute_content_hash() -> None:
    text = "SecondSelf associative memory test"
    digest = compute_content_hash(text)
    assert len(digest) == 64
    assert digest == compute_content_hash(text)
    assert digest != compute_content_hash("Different text")


def test_extract_text_valid() -> None:
    text = "# My Research Notes\nExploring force-directed graph canvas."
    clean, meta = extract_text(text)
    assert clean == text
    assert meta["title"] == "My Research Notes"
    assert meta["char_count"] == len(text)


def test_extract_text_empty_raises() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        extract_text("   \n\t  ")


def test_extract_file_markdown(tmp_path: Path) -> None:
    sample_file = tmp_path / "notes_sample.md"
    sample_file.write_text("# Knowledge Architecture\nDeep dive into P.A.R.A.", encoding="utf-8")

    content, meta = extract_file(sample_file)
    assert "Knowledge Architecture" in content
    assert meta["title"] in ("Knowledge Architecture", "Notes Sample")
    assert meta["char_count"] > 0


def test_extract_file_missing_raises() -> None:
    non_existent = Path("does_not_exist_xyz123.md")
    with pytest.raises(FileNotFoundError):
        extract_file(non_existent)


def test_capture_text_dry_run() -> None:
    payload, saved_path = capture(
        text="Quick architectural thought for vector embeddings.",
        dry_run=True,
    )
    assert payload.source_type == "text"
    assert payload.source_uri is None
    assert saved_path is None
    assert payload.metadata["char_count"] > 0
    assert "content_hash" in payload.metadata


def test_capture_text_persists_to_disk(tmp_path: Path) -> None:
    payload, saved_path = capture(
        text="Idea: Compute reciprocal wikilinks using cosine dot-product.",
        output_dir=tmp_path,
        dry_run=False,
    )
    assert saved_path is not None
    assert saved_path.exists()
    assert saved_path.suffix == ".json"

    # Verify JSON file structure
    data = json.loads(saved_path.read_text(encoding="utf-8"))
    assert data["id"] == payload.id
    assert data["source_type"] == "text"
    assert "cosine dot-product" in data["raw_content"]
    assert data["metadata"]["content_hash"] == payload.metadata["content_hash"]


def test_capture_conflicting_inputs_raises() -> None:
    with pytest.raises(ValueError, match="Conflicting inputs"):
        capture(text="Idea A", url="https://example.com")


def test_capture_no_inputs_raises() -> None:
    with pytest.raises(ValueError, match="No input source provided"):
        capture()


def test_cli_main_success_with_text(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["-t", "CLI test note", "-o", str(tmp_path), "--format", "json"])
    assert exit_code == EXIT_SUCCESS

    captured = capsys.readouterr().out
    data = json.loads(captured)
    assert data["source_type"] == "text"
    assert data["raw_content"] == "CLI test note"


def test_cli_main_missing_input(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main([])
    assert exit_code == EXIT_USAGE_ERROR
    err = capsys.readouterr().err
    assert "No input specified" in err


def test_cli_main_file_not_found(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["-f", "missing_file_xyz.txt"])
    assert exit_code == EXIT_IO_ERROR
