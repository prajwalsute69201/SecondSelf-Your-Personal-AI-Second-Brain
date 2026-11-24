"""SecondSelf — Autonomous Classification CLI Implementation.

Provides the CLI commands and batch processing orchestrator for autonomous PARA classification.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from classify.classifier import PARAClassifier
from classify.models import PARAClassification
from config import get_settings

logger = logging.getLogger("secondself.classify")

EXIT_SUCCESS = 0
EXIT_GENERAL_ERROR = 1
EXIT_USAGE_ERROR = 2
EXIT_IO_ERROR = 3


def find_unprocessed_records(
    raw_dir: Path,
    wiki_dir: Path,
) -> list[tuple[str, Path, dict[str, Any]]]:
    """Scans raw storage for records that do not yet exist in wiki/*/*.md."""
    candidates: list[tuple[str, Path, dict[str, Any]]] = []

    # Build set of already existing record IDs across all wiki categories
    existing_wiki_ids: set[str] = set()
    if wiki_dir.exists():
        for cat_dir in wiki_dir.glob("*"):
            if cat_dir.is_dir():
                for note_file in cat_dir.glob("*.md"):
                    existing_wiki_ids.add(note_file.stem)

    if not raw_dir.exists():
        return candidates

    # Search both raw/ root and raw/objects/ subdirectories
    json_paths = list(raw_dir.glob("*.json")) + list(raw_dir.glob("objects/**/*.json"))

    for json_path in json_paths:
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            rec_id = data.get("id") or data.get("record_id") or json_path.stem
            if rec_id not in existing_wiki_ids:
                candidates.append((rec_id, json_path, data))
        except Exception as exc:
            logger.warning("Could not parse raw file '%s': %s", json_path, exc)

    return candidates


def serialize_note_to_wiki(
    classification: PARAClassification,
    raw_payload: str,
    record_id: str,
    wiki_dir: Path,
) -> Path:
    """Writes classified note to wiki/{category}/{record_id}.md with YAML frontmatter."""
    target_dir = wiki_dir / classification.category.value
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / f"{record_id}.md"

    frontmatter = classification.to_markdown_frontmatter()
    note_content = (
        f"{frontmatter}\n\n"
        f"# {classification.title}\n\n"
        f"> **Summary:** {classification.summary}\n\n"
        f"{raw_payload.strip()}\n"
    )

    target_file.write_text(note_content, encoding="utf-8")
    return target_file


def process_batch(
    raw_dir: Path,
    wiki_dir: Path,
    classifier: PARAClassifier,
    dry_run: bool = False,
) -> list[tuple[str, PARAClassification, Path | None]]:
    """Runs batch classification over all unprocessed raw captures."""
    unprocessed = find_unprocessed_records(raw_dir, wiki_dir)
    results: list[tuple[str, PARAClassification, Path | None]] = []

    for rec_id, path, data in unprocessed:
        raw_content = data.get("raw_content") or data.get("payload") or ""
        metadata = data.get("metadata") or {}
        metadata.setdefault("record_id", rec_id)

        classification = classifier.classify(content=raw_content, metadata=metadata)
        written_path: Path | None = None

        if not dry_run:
            written_path = serialize_note_to_wiki(
                classification=classification,
                raw_payload=raw_content,
                record_id=rec_id,
                wiki_dir=wiki_dir,
            )

        results.append((rec_id, classification, written_path))

    return results


def build_arg_parser() -> argparse.ArgumentParser:
    """Builds the CLI argument parser for classify.py."""
    parser = argparse.ArgumentParser(
        prog="secondself-classify",
        description="Autonomous PARA Knowledge Classification Engine.",
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "-t", "--text",
        dest="text",
        type=str,
        help="Direct text snippet to classify",
    )
    group.add_argument(
        "-f", "--file",
        dest="file_path",
        type=Path,
        help="Path to raw JSON capture record to classify",
    )
    group.add_argument(
        "-b", "--batch",
        dest="batch",
        action="store_true",
        help="Batch process all unclassified records in raw/ and write to wiki/",
    )
    parser.add_argument(
        "-m", "--method",
        dest="method",
        choices=["rules", "llm", "hybrid"],
        default=None,
        help="Classification method (overrides configuration setting)",
    )
    parser.add_argument(
        "--confidence-threshold",
        dest="confidence_threshold",
        type=float,
        default=None,
        help="Confidence threshold ceiling (0.0 to 1.0)",
    )
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Execute classification without modifying wiki files on disk",
    )
    parser.add_argument(
        "--format",
        dest="format",
        choices=["text", "json"],
        default="text",
        help="Output serialization format (default: text)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI execution entrypoint."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    settings = get_settings()
    cfg_kwargs: dict[str, Any] = {}
    if args.method:
        cfg_kwargs["method"] = args.method
    if args.confidence_threshold is not None:
        cfg_kwargs["confidence_threshold"] = args.confidence_threshold

    config = (
        settings.classifier.model_copy(update=cfg_kwargs)
        if cfg_kwargs
        else settings.classifier
    )
    classifier = PARAClassifier(config=config)

    # 1. Single text classification
    if args.text is not None:
        res = classifier.classify(content=args.text)
        if args.format == "json":
            sys.stdout.write(json.dumps(res.to_dict(), indent=2) + "\n")
        else:
            sys.stdout.write(
                f"Category:   {res.category.value}\n"
                f"Confidence: {res.confidence:.2%}\n"
                f"Needs Rev:  {res.needs_review}\n"
                f"Method:     {res.method}\n"
                f"Rationale:  {res.rationale}\n"
            )
        return EXIT_SUCCESS

    # 2. Single file classification
    if args.file_path is not None:
        if not args.file_path.exists():
            sys.stderr.write(f"Error: Target file does not exist: {args.file_path}\n")
            return EXIT_IO_ERROR
        try:
            data = json.loads(args.file_path.read_text(encoding="utf-8"))
            content = data.get("raw_content") or data.get("payload") or ""
            metadata = data.get("metadata") or {}
            res = classifier.classify(content=content, metadata=metadata)
            if args.format == "json":
                sys.stdout.write(json.dumps(res.to_dict(), indent=2) + "\n")
            else:
                sys.stdout.write(
                    f"File:       {args.file_path}\n"
                    f"Category:   {res.category.value}\n"
                    f"Confidence: {res.confidence:.2%}\n"
                    f"Needs Rev:  {res.needs_review}\n"
                    f"Method:     {res.method}\n"
                    f"Rationale:  {res.rationale}\n"
                )
            return EXIT_SUCCESS
        except Exception as exc:
            sys.stderr.write(f"Error processing file '{args.file_path}': {exc}\n")
            return EXIT_GENERAL_ERROR

    # 3. Batch processing or default behavior
    raw_dir = settings.paths.raw_dir
    wiki_dir = settings.paths.wiki_dir

    results = process_batch(
        raw_dir=raw_dir,
        wiki_dir=wiki_dir,
        classifier=classifier,
        dry_run=args.dry_run,
    )

    if args.format == "json":
        output_payload = [
            {"id": rec_id, "classification": c.to_dict(), "written_path": str(p) if p else None}
            for rec_id, c, p in results
        ]
        sys.stdout.write(json.dumps(output_payload, indent=2) + "\n")
    else:
        sys.stdout.write(
            f"Autonomous PARA Classification Batch Complete:\n"
            f"Processed: {len(results)} items (dry_run={args.dry_run}, method={config.method})\n"
        )
        for rec_id, c, p in results:
            status_tag = "[NEEDS_REVIEW]" if c.needs_review else "[OK]"
            sys.stdout.write(
                f" - {rec_id[:8]} -> {c.category.value} ({c.confidence:.0%}) {status_tag} {c.title}\n"
            )

    return EXIT_SUCCESS
