"""Comprehensive Unit & Integration Test Suite for SecondSelf PARA Classifier.

Validates:
- Pure rule-based signal extraction and scoring in isolation (zero network/disk).
- Clear-cut classifications for Projects, Areas, Resources, and Archives.
- Ambiguous input handling and low-confidence / NEEDS_REVIEW triggering.
- Deterministic execution (identical input yields identical output across runs).
- LLM-assisted classification with structured JSON validation.
- Graceful degradation and rule fallback upon LLM or network failure.
- Hybrid fusion combining heuristic and LLM scores with configured weights.
- Compatibility with ExtractedContent and RecordMetadata data contracts.
- CLI execution modes (text, file, and batch dry-run).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from adapters.models import ExtractedContent
from classify import (
    CATEGORY_ALIASES,
    TAXONOMY_CRITERIA,
    CategoryCriteria,
    PARACategory,
    PARAClassification,
    PARAClassifier,
    Signal,
    build_user_prompt,
    classify,
    extract_signals,
    get_criteria,
    parse_and_validate_llm_response,
    score_rules,
)
from classify.llm_prompt import LLMValidationError
from config.base import ClassifierConfig
from storage.models import RecordMetadata


# ==============================================================================
# 1. Taxonomy & Criteria Tests
# ==============================================================================

def test_para_category_enum_values() -> None:
    assert PARACategory.PROJECT.value == "Projects"
    assert PARACategory.AREA.value == "Areas"
    assert PARACategory.RESOURCE.value == "Resources"
    assert PARACategory.ARCHIVE.value == "Archives"


def test_para_category_from_string_aliases() -> None:
    # Plural and singular
    assert PARACategory.from_string("project") == PARACategory.PROJECT
    assert PARACategory.from_string("Projects") == PARACategory.PROJECT
    assert PARACategory.from_string("area") == PARACategory.AREA
    assert PARACategory.from_string("Areas") == PARACategory.AREA
    assert PARACategory.from_string("resource") == PARACategory.RESOURCE
    assert PARACategory.from_string("archive") == PARACategory.ARCHIVE

    # Synonyms and domain aliases
    assert PARACategory.from_string("task") == PARACategory.PROJECT
    assert PARACategory.from_string("habit") == PARACategory.AREA
    assert PARACategory.from_string("cheatsheet") == PARACategory.RESOURCE
    assert PARACategory.from_string("done") == PARACategory.ARCHIVE
    assert PARACategory.from_string("deprecated") == PARACategory.ARCHIVE

    # Unrecognized falls back to Resource
    assert PARACategory.from_string("unrecognized_xyz_99") == PARACategory.RESOURCE


def test_taxonomy_criteria_definitions() -> None:
    for cat in PARACategory:
        crit = get_criteria(cat)
        assert isinstance(crit, CategoryCriteria)
        assert crit.category == cat
        assert crit.title == cat.value
        assert len(crit.primary_signals) > 0
        assert crit.target_folder == f"wiki/{cat.value}"


# ==============================================================================
# 2. Pure Rule-Based Signal Extraction & Scoring (Zero I/O Isolation)
# ==============================================================================

def test_rule_clearcut_project_with_deadline() -> None:
    text = (
        "# Q4 Infrastructure Migration\n"
        "We must complete the migration. The deadline: Dec 15, 2026.\n"
        "Action items:\n"
        "- Deploy to production by EOD Friday\n"
        "- Ship v1.0 release\n"
    )
    signals = extract_signals(text)
    assert any(s.category == PARACategory.PROJECT for s in signals)
    assert any(s.name == "explicit_deadline" for s in signals)

    winner, conf, scores, needs_review, rationale = score_rules(signals)
    assert winner == PARACategory.PROJECT
    assert conf >= 0.60
    assert not needs_review
    assert "Projects" in rationale


def test_rule_clearcut_area_recurring_responsibility() -> None:
    text = (
        "# Team Engineering Operations\n"
        "Standard operating procedure for continuous maintenance and team 1-on-1s.\n"
        "Weekly review cadence:\n"
        "- Check system uptime\n"
        "- Maintain standards to follow across services\n"
        "- Budget management and health tracking\n"
    )
    signals = extract_signals(text)
    assert any(s.category == PARACategory.AREA for s in signals)
    assert any(s.name in ("sop_standards_marker", "recurring_cadence_marker") for s in signals)

    winner, conf, scores, needs_review, rationale = score_rules(signals)
    assert winner == PARACategory.AREA
    assert conf >= 0.60
    assert not needs_review


def test_rule_clearcut_resource_reference_material() -> None:
    text = (
        "# Python 3.14 Concurrency Cheatsheet\n"
        "A comprehensive syntax reference, api reference, and tutorial on subinterpreters.\n"
        "Overview of memory models and deep dive into GIL removal benchmarks."
    )
    signals = extract_signals(text)
    assert any(s.category == PARACategory.RESOURCE for s in signals)
    assert any(s.name in ("cheatsheet_reference_marker", "documentation_guide_marker") for s in signals)

    winner, conf, scores, needs_review, rationale = score_rules(signals)
    assert winner == PARACategory.RESOURCE
    assert conf >= 0.60
    assert not needs_review


def test_rule_clearcut_archive_completed_and_deprecated() -> None:
    text = (
        "# Project Odyssey Wrap-Up\n"
        "Status: DONE. Successfully shipped.\n"
        "Retrospective and post-mortem review of legacy code.\n"
        "This repository is now deprecated and archived."
    )
    signals = extract_signals(text)
    assert any(s.category == PARACategory.ARCHIVE for s in signals)
    assert any(s.name in ("completed_status_marker", "deprecated_legacy_marker") for s in signals)

    winner, conf, scores, needs_review, rationale = score_rules(signals)
    assert winner == PARACategory.ARCHIVE
    assert conf >= 0.60
    assert not needs_review


def test_rule_stale_metadata_triggers_archive() -> None:
    text = "Initial draft for auth notes. No active deadlines."
    old_timestamp = (datetime.now(timezone.utc) - timedelta(days=220)).isoformat()
    metadata = {"created_at": old_timestamp, "source_type": "text"}

    signals = extract_signals(text, metadata=metadata, archive_staleness_days=180)
    assert any(s.name == "historical_staleness" for s in signals)

    winner, conf, scores, _, _ = score_rules(signals)
    assert winner == PARACategory.ARCHIVE


def test_rule_ambiguous_content_triggers_needs_review() -> None:
    text = "Just a general musing about life and ideas on a quiet afternoon."
    signals = extract_signals(text)
    # Very few or zero signals
    winner, conf, scores, needs_review, rationale = score_rules(
        signals, confidence_threshold=0.60, default_fallback=PARACategory.RESOURCE
    )
    assert needs_review is True
    assert conf < 0.60
    assert winner == PARACategory.RESOURCE
    assert "low confidence" in rationale or "marginal" in rationale


# ==============================================================================
# 3. Determinism & Stability Tests
# ==============================================================================

def test_classifier_determinism_identical_runs() -> None:
    classifier = PARAClassifier(config=ClassifierConfig(method="rules"))
    content = "Ship the security audit deliverables by next week. Target date: 2026-12-01."
    meta = {"source_type": "text", "title": "Security Audit"}

    run1 = classifier.classify(content, metadata=meta)
    run2 = classifier.classify(content, metadata=meta)
    run3 = classifier.classify(content, metadata=meta)

    assert run1.category == run2.category == run3.category == PARACategory.PROJECT
    assert run1.confidence == run2.confidence == run3.confidence
    assert run1.rationale == run2.rationale == run3.rationale
    assert run1.content_hash == run2.content_hash == run3.content_hash
    assert len(run1.signals) == len(run2.signals)


# ==============================================================================
# 4. LLM Prompting & Schema Validation
# ==============================================================================

def test_build_user_prompt_formatting() -> None:
    prompt = build_user_prompt(
        content="Short test note content",
        metadata={"source_type": "url", "source_uri": "https://example.com/spec", "title": "API Spec"},
    )
    assert "https://example.com/spec" in prompt
    assert "API Spec" in prompt
    assert "Short test note content" in prompt


def test_parse_and_validate_llm_response_valid() -> None:
    raw_json = json.dumps({
        "category": "Projects",
        "confidence": 0.88,
        "rationale": "Clear deadline and deliverable identified.",
        "signals": ["due by Friday", "deliver v1.0"],
        "title": "Deliver v1.0",
        "tags": ["release", "v1"],
        "summary": "Team must deliver v1 by Friday.",
    })
    validated = parse_and_validate_llm_response(raw_json)
    assert validated.category == PARACategory.PROJECT
    assert validated.confidence == 0.88
    assert validated.title == "Deliver v1.0"
    assert len(validated.signals) == 2


def test_parse_and_validate_llm_response_markdown_fences() -> None:
    raw = (
        "```json\n"
        "{\n"
        '  "category": "Areas",\n'
        '  "confidence": 0.75,\n'
        '  "rationale": "Weekly review routine.",\n'
        '  "signals": ["weekly sync"],\n'
        '  "title": "Weekly Review",\n'
        '  "tags": ["ops"],\n'
        '  "summary": "Weekly operations checklist."\n'
        "}\n"
        "```"
    )
    validated = parse_and_validate_llm_response(raw)
    assert validated.category == PARACategory.AREA
    assert validated.confidence == 0.75


def test_parse_and_validate_llm_response_malformed_raises() -> None:
    bad_json = "{category: 'Projects', invalid...}"
    with pytest.raises(LLMValidationError):
        parse_and_validate_llm_response(bad_json)


# ==============================================================================
# 5. Hybrid Resolution & Graceful Fallback
# ==============================================================================

def test_hybrid_resolution_fuses_rule_and_mock_llm() -> None:
    # Rule would favor Project
    content = "Deadline: tomorrow. Ship deliverable."

    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({
        "category": "Projects",
        "confidence": 0.90,
        "rationale": "Urgent deadline tomorrow with active deliverable.",
        "signals": ["deadline: tomorrow", "ship deliverable"],
        "title": "Ship Deliverable",
        "tags": ["launch", "project"],
        "summary": "Urgent tomorrow delivery.",
    })
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    classifier = PARAClassifier(
        config=ClassifierConfig(method="hybrid", rule_weight=0.4, llm_weight=0.6),
        groq_client=mock_client,
    )

    result = classifier.classify(content)
    assert result.category == PARACategory.PROJECT
    assert result.method == "hybrid"
    assert result.confidence >= 0.70
    assert not result.needs_review
    assert any(s.source == "llm" for s in result.signals)
    assert any(s.source == "rule_content" for s in result.signals)


def test_llm_failure_gracefully_falls_back_to_rules() -> None:
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = TimeoutError("Groq request timed out")

    classifier = PARAClassifier(
        config=ClassifierConfig(method="hybrid", llm_max_retries=1),
        groq_client=mock_client,
    )

    content = "Cheatsheet and API documentation for Python 3.14."
    result = classifier.classify(content)

    assert result.method == "rules_fallback"
    assert result.category == PARACategory.RESOURCE
    assert "Rules Fallback" in result.rationale


# ==============================================================================
# 6. Data Contract Compatibility & Markdown Frontmatter
# ==============================================================================

def test_classify_with_extracted_content_and_record_metadata() -> None:
    extracted = ExtractedContent(
        text="# Developer Guide\nDocumentation on setting up SecondSelf.",
        title="Developer Guide",
        source_type="markdown",
        source_uri="file:///docs/guide.md",
    )
    metadata = RecordMetadata(
        created_at=datetime.now(timezone.utc).isoformat(),
        source_type="markdown",
        content_hash="abc12345def67890",
    )

    res = classify(extracted, metadata=metadata)
    assert res.category == PARACategory.RESOURCE
    assert res.title == "Developer Guide"
    assert res.content_hash == extracted.content_hash


def test_markdown_frontmatter_rendering() -> None:
    cls = PARAClassification(
        category=PARACategory.PROJECT,
        confidence=0.85,
        rationale="Clear deadline.",
        title="Launch MVP",
        tags=["mvp", "launch"],
        summary="Launch v1.0 by Friday.",
        content_hash="hash123",
        classifier_version="1.0.0",
        method="rules",
        needs_review=False,
    )
    fm = cls.to_markdown_frontmatter()
    assert fm.startswith("---")
    assert fm.endswith("---")
    assert 'title: "Launch MVP"' in fm
    assert 'category: "Projects"' in fm
    assert "confidence: 0.85" in fm
    assert "needs_review: false" in fm
    assert "  - mvp" in fm

    wiki_path = cls.to_wiki_path("note_uuid_123")
    assert wiki_path.as_posix() == "wiki/Projects/note_uuid_123.md"


# ==============================================================================
# 7. CLI Execution Interface Tests
# ==============================================================================

def test_classify_cli_text_mode(capsys: pytest.CaptureFixture[str]) -> None:
    from classify import main

    exit_code = main(["-t", "Syntax reference and cheatsheet for regex", "--format", "json"])
    assert exit_code == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["category"] == "Resources"
    assert data["confidence"] > 0.50


def test_classify_cli_file_mode(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from classify import main

    sample_record = tmp_path / "sample_record.json"
    sample_record.write_text(
        json.dumps({
            "id": "rec-123",
            "raw_content": "Deliverable: ship new feature by tomorrow morning.",
            "metadata": {"title": "Ship Feature"},
        }),
        encoding="utf-8",
    )

    exit_code = main(["-f", str(sample_record), "--format", "json"])
    assert exit_code == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["category"] == "Projects"


def test_classify_cli_batch_dry_run(capsys: pytest.CaptureFixture[str]) -> None:
    from classify import main

    exit_code = main(["--batch", "--dry-run", "--method", "rules"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Batch Complete" in out
