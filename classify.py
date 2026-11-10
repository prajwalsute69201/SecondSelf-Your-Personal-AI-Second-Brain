"""Classification pipeline and CLI for SecondSelf (Task 2.1.4).

Processes unprocessed raw captures from `raw/`, extracts readable content,
classifies them into PARA categories ('Projects', 'Areas', 'Resources', 'Archives')
with semantic tags and concise summaries using Groq LLM, writes structured markdown
notes into `wiki/{para}/{id}.md`, and updates `data/index.json`.

Usage:
    python classify.py                  # Classify all unprocessed captures
    python classify.py --id <capture_id> # Classify a single specific capture
    python classify.py --dry-run        # Preview classification without writing files
    python classify.py --help           # Show CLI options
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

# Ensure repository root is on sys.path so 'lib' is importable
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lib.extract import extract_text
from lib.llm import VALID_PARA_CATEGORIES, classify_content
from lib.models import CaptureMeta, WikiNote
import lib.storage as storage

__all__ = [
    "extract_wiki_id",
    "classify_capture",
    "classify_all",
    "main",
]


def extract_wiki_id(raw_id: str) -> str:
    """Extract short wiki ID from raw capture ID.

    Convention: YYYY-MM-DD_{uuid8} -> {uuid8}.
    If raw_id does not contain an underscore, returns raw_id unchanged.

    Args:
        raw_id: Raw capture identifier string.

    Returns:
        str: Short wiki identifier.
    """
    if "_" in raw_id:
        return raw_id.split("_", 1)[-1]
    return raw_id


def classify_capture(
    meta: CaptureMeta,
    root: Path | str | None = None,
    model: str | None = None,
    dry_run: bool = False,
) -> WikiNote:
    """Extract content, classify via LLM, and persist wiki note for a raw capture.

    Args:
        meta: CaptureMeta instance of the raw capture to classify.
        root: Optional project root path override.
        model: Optional Groq model identifier override.
        dry_run: If True, extracts and classifies but skips writing wiki files and index.

    Returns:
        WikiNote: Constructed (and persisted, unless dry_run) WikiNote instance.

    Raises:
        ValueError: If capture is already processed, content extraction fails,
            extracted text is empty, or LLM classification fails validation.
        RuntimeError: If Groq API request fails (network, rate limit, auth).
    """
    # 1. Idempotency check: verify not already processed
    index = storage.load_index(root=root)
    raw_processed = index.get("raw_processed", {})
    if isinstance(raw_processed, dict) and meta.id in raw_processed:
        raise ValueError(f"Capture {meta.id} has already been processed.")

    # 2. Extract readable text using lib.extract
    try:
        content_text = extract_text(meta, root=root)
    except Exception as e:
        raise ValueError(f"Failed to extract text from capture {meta.id}: {e}") from e

    if not content_text or not content_text.strip():
        raise ValueError(
            f"Extracted content is empty for capture {meta.id}. Cannot classify empty content."
        )

    # 3. Classify via LLM (lib.llm handles token truncation, retries, and schema validation)
    classification = classify_content(content_text, model=model)

    para = classification["para"]
    tags = classification["tags"]
    summary = classification["summary"]

    # 4. Construct WikiNote
    wiki_id = extract_wiki_id(meta.id)
    note = WikiNote(
        id=wiki_id,
        raw_id=meta.id,
        para=para,
        tags=tags,
        summary=summary,
        created=meta.timestamp,
        links=[],
        body=content_text,
    )

    if dry_run:
        return note

    # 5. Persist wiki note
    storage.write_wiki_note(note, root=root)

    # 6. Update index.json (only reached if write_wiki_note succeeded)
    now_iso = datetime.now(timezone.utc).isoformat()
    if "raw_processed" not in index or not isinstance(index["raw_processed"], dict):
        index["raw_processed"] = {}

    index["raw_processed"][meta.id] = {
        "wiki_id": note.id,
        "para": note.para,
        "classified_at": now_iso,
        "content_hash": meta.content_hash,
    }
    storage.save_index(index, root=root)

    return note


def classify_all(
    root: Path | str | None = None,
    model: str | None = None,
    capture_ids: list[str] | None = None,
    dry_run: bool = False,
) -> tuple[list[WikiNote], list[tuple[str, str]]]:
    """Discover and classify all unprocessed raw captures.

    Args:
        root: Optional project root path override.
        model: Optional Groq model identifier override.
        capture_ids: Optional list of capture IDs to restrict processing to.
        dry_run: If True, classifies without writing files or updating index.

    Returns:
        tuple: (list of successful WikiNote instances, list of (capture_id, error_message) tuples).
    """
    unprocessed = storage.read_raw_captures(root=root)

    if capture_ids:
        target_set = set(capture_ids)
        unprocessed = [m for m in unprocessed if m.id in target_set]

    successful: list[WikiNote] = []
    failed: list[tuple[str, str]] = []

    for meta in unprocessed:
        try:
            note = classify_capture(meta, root=root, model=model, dry_run=dry_run)
            successful.append(note)
        except Exception as e:
            failed.append((meta.id, str(e)))

    return successful, failed


def build_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser for classify.py."""
    parser = argparse.ArgumentParser(
        prog="classify.py",
        description="Classify unprocessed SecondSelf raw captures into PARA wiki notes.",
    )
    parser.add_argument(
        "--id",
        dest="capture_id",
        metavar="ID",
        help="Classify a specific raw capture by ID instead of all unprocessed items.",
    )
    parser.add_argument(
        "--model",
        dest="model",
        metavar="MODEL",
        help="Override the default Groq LLM model identifier.",
    )
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Preview extraction and LLM classification without saving notes or modifying index.",
    )
    parser.add_argument(
        "--root",
        dest="root",
        metavar="PATH",
        help=argparse.SUPPRESS,  # For internal / test usage
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for classify.py.

    Args:
        argv: Command-line argument list (defaults to sys.argv[1:]).

    Returns:
        int: Exit status code (0 for success, 1 for failure, 130 for interrupt).
    """
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 0

    root = Path(args.root).resolve() if args.root else None

    target_ids = [args.capture_id] if args.capture_id else None

    # Check for pending captures
    try:
        pending = storage.read_raw_captures(root=root)
        if target_ids:
            pending = [m for m in pending if m.id in set(target_ids)]
    except Exception as e:
        print(f"Error inspecting raw captures: {e}", file=sys.stderr)
        return 1

    if not pending:
        if args.capture_id:
            print(f"Capture {args.capture_id!r} not found or already processed.")
        else:
            print("No unprocessed captures found in raw/. Everything is up to date.")
        return 0

    if args.dry_run:
        print(f"DRY RUN: Classifying {len(pending)} pending capture(s) (no files will be written)...")
    else:
        print(f"Classifying {len(pending)} pending capture(s)...")

    try:
        successful, failed = classify_all(
            root=root,
            model=args.model,
            capture_ids=target_ids,
            dry_run=args.dry_run,
        )
    except KeyboardInterrupt:
        print("\nClassification cancelled by user.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Fatal classification error: {e}", file=sys.stderr)
        return 1

    # Output results
    for note in successful:
        tags_str = ", ".join(note.tags) if note.tags else "no tags"
        action = "[dry-run] Would classify" if args.dry_run else "Classified"
        print(f"{action} raw/{note.raw_id} → wiki/{note.para}/{note.id}.md [{tags_str}]")

    if failed:
        for cap_id, err in failed:
            print(f"Error classifying raw/{cap_id}: {err}", file=sys.stderr)

        print(
            f"\nClassification finished with errors: {len(successful)} processed, "
            f"{len(failed)} failed.",
            file=sys.stderr,
        )
        return 1

    prefix = "Dry-run preview complete" if args.dry_run else "Classification complete"
    print(f"\n{prefix}: {len(successful)} processed, 0 failed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
