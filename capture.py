#!/usr/bin/env python3
"""SecondSelf — Multi-Modal Raw Knowledge Capture CLI.

The Archivist (Milestone 1):
Ingests raw knowledge from multiple sources (direct text, web URLs, local PDFs,
markdown/text files, or piped stdin) and saves immutable, content-hashed JSON
records to the raw/ storage vault using Content Extraction Adapters and RawStore.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from adapters import (
    ExtractedContent,
    ExtractionError,
    MalformedContentError,
    MissingDependencyError,
    UnsupportedContentError,
    extract_content,
)
from config import get_settings
from storage import RawStore

logger = logging.getLogger("secondself.capture")

# Standard exit codes
EXIT_SUCCESS = 0
EXIT_GENERAL_ERROR = 1
EXIT_USAGE_ERROR = 2
EXIT_IO_ERROR = 3
EXIT_NETWORK_ERROR = 4


class RawCapturePayload(BaseModel):
    """Immutable data contract for raw captured knowledge items."""

    id: str = Field(..., description="Canonical UUIDv4 identifier")
    timestamp: str = Field(..., description="ISO-8601 UTC timestamp of ingestion")
    source_type: str = Field(..., description="Capture mode: 'text', 'url', 'file', or 'stdin'")
    source_uri: str | None = Field(default=None, description="Origin URL or source file path")
    raw_content: str = Field(..., description="Extracted unstructured content")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Diagnostic and source metadata")

    def to_json(self, indent: int = 2) -> str:
        """Serializes payload to formatted JSON."""
        return self.model_dump_json(indent=indent)


def compute_content_hash(content: str) -> str:
    """Computes SHA-256 digest of normalized UTF-8 string."""
    return hashlib.sha256(content.strip().encode("utf-8")).hexdigest()


# ------------------------------------------------------------------------------
# Adapter Delegate Functions (Backward Compatibility for Unit Tests & Callers)
# ------------------------------------------------------------------------------

def extract_text(content: str) -> tuple[str, dict[str, Any]]:
    """Extracts text content delegating to the TextExtractor adapter."""
    res = extract_content(content, adapter_name="text")
    return res.text, {"title": res.title, "char_count": res.char_count}


def extract_url(url: str, timeout: float = 15.0) -> tuple[str, dict[str, Any]]:
    """Fetches URL content delegating to the HTMLExtractor adapter."""
    try:
        res = extract_content(url, adapter_name="html", timeout=timeout)
        meta = dict(res.metadata)
        meta["title"] = res.title
        meta["char_count"] = res.char_count
        return res.text, meta
    except ExtractionError as exc:
        raise ConnectionError(str(exc)) from exc


def extract_file(file_path: Path) -> tuple[str, dict[str, Any]]:
    """Extracts file content delegating to the appropriate file adapter."""
    if not file_path.exists():
        raise FileNotFoundError(f"Source file does not exist: '{file_path}'")
    if not file_path.is_file():
        raise ValueError(f"Path is not a regular file: '{file_path}'")

    res = extract_content(file_path)
    meta = dict(res.metadata)
    meta["title"] = res.title
    meta["char_count"] = res.char_count
    return res.text, meta


# ------------------------------------------------------------------------------
# Core Business Function (Importable & Testable)
# ------------------------------------------------------------------------------

def capture(
    *,
    text: str | None = None,
    url: str | None = None,
    file_path: str | Path | None = None,
    stdin_content: str | None = None,
    output_dir: str | Path | None = None,
    dry_run: bool = False,
    timeout: float | None = None,
) -> tuple[RawCapturePayload, Path | None]:
    """Ingests unstructured content and saves an immutable JSON record to raw/.

    Args:
        text: Raw text or thought string.
        url: Web URL to fetch and parse.
        file_path: Local path to a text, markdown, or PDF file.
        stdin_content: Input piped from standard input.
        output_dir: Destination folder. Defaults to settings.paths.raw_dir.
        dry_run: If True, computes payload without writing to disk.
        timeout: Network timeout in seconds for URL requests.

    Returns:
        A tuple of (RawCapturePayload, saved_file_path_or_None).
    """
    settings = get_settings()
    out_path = Path(output_dir) if output_dir else settings.paths.raw_dir
    req_timeout = timeout if timeout is not None else settings.groq.timeout_seconds

    inputs = [
        ("text", text),
        ("url", url),
        ("file", file_path),
        ("stdin", stdin_content),
    ]
    active_inputs = [(kind, val) for kind, val in inputs if val is not None]

    if not active_inputs:
        raise ValueError("No input source provided. Supply --text, --url, --file, or --stdin.")
    if len(active_inputs) > 1:
        names = ", ".join(f"--{k}" for k, _ in active_inputs)
        raise ValueError(f"Conflicting inputs: Multiple input flags provided ({names}). Select only one.")

    source_type, source_val = active_inputs[0]
    now = datetime.now(timezone.utc)
    timestamp_iso = now.isoformat()
    capture_id = str(uuid.uuid4())
    short_id = capture_id[:8]

    # Delegate extraction to Adapter layer
    extracted: ExtractedContent
    source_uri: str | None

    if source_type == "text":
        source_uri = None
        extracted = extract_content(str(source_val), adapter_name="text")
    elif source_type == "url":
        source_uri = str(source_val)
        try:
            extracted = extract_content(source_uri, adapter_name="html", timeout=req_timeout)
        except ExtractionError as exc:
            raise ConnectionError(f"Failed to fetch URL '{source_uri}': {exc}") from exc
    elif source_type == "file":
        resolved_file = Path(source_val).resolve()
        if not resolved_file.exists():
            raise FileNotFoundError(f"Source file does not exist: '{resolved_file}'")
        source_uri = str(resolved_file)
        extracted = extract_content(resolved_file, source_uri=source_uri)
    elif source_type == "stdin":
        source_uri = "stdin"
        extracted = extract_content(str(source_val), adapter_name="text", source_uri="stdin")
    else:
        raise ValueError(f"Unknown source type: {source_type}")

    meta = dict(extracted.metadata)
    meta["title"] = extracted.title
    meta["char_count"] = extracted.char_count
    meta["word_count"] = extracted.word_count
    meta["content_hash"] = extracted.content_hash
    meta["captured_at"] = timestamp_iso
    if extracted.warnings:
        meta["warnings"] = extracted.warnings

    payload = RawCapturePayload(
        id=capture_id,
        timestamp=timestamp_iso,
        source_type=source_type,
        source_uri=source_uri,
        raw_content=extracted.text,
        metadata=meta,
    )

    if dry_run:
        logger.info("[Dry Run] Extracted %d chars (ID: %s). Skipping disk write.", extracted.char_count, capture_id)
        return payload, None

    # Persist via RawStore (Content-Addressed Storage + Manifest logging)
    try:
        out_path.mkdir(parents=True, exist_ok=True)
        store = RawStore(root_dir=out_path)
        store.write(
            content=extracted.text,
            source_type=source_type,
            source_uri=source_uri,
            content_type=extracted.metadata.get("format", "text/plain"),
            adapter_name=extracted.source_type,
            record_id=capture_id,
            extra=meta,
        )
    except Exception as exc:
        raise IOError(f"RawStore write error in '{out_path}': {exc}") from exc

    # File naming: YYYYMMDD_HHMMSS_<short_id>.json (flat timestamped copy)
    timestamp_prefix = now.strftime("%Y%m%d_%H%M%S")
    target_file = out_path / f"{timestamp_prefix}_{short_id}.json"

    try:
        target_file.write_text(payload.to_json(indent=2), encoding="utf-8")
        logger.info("Saved raw capture record to: %s", target_file)
    except Exception as exc:
        raise IOError(f"Failed to write capture file '{target_file}': {exc}") from exc

    return payload, target_file


# ------------------------------------------------------------------------------
# CLI Interface & Argument Parsing
# ------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Constructs the CLI argument parser with mutually exclusive groups and epilog."""
    settings = get_settings()

    parser = argparse.ArgumentParser(
        prog="capture.py",
        description="SecondSelf Multi-Modal Knowledge Ingestion CLI (The Archivist)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument("-t", "--text", type=str, help="Direct thought, idea, or text note to capture.")
    input_group.add_argument("-u", "--url", type=str, help="Web page URL to fetch, clean, and archive.")
    input_group.add_argument("-f", "--file", type=Path, help="Local PDF, Markdown, or text file path to extract.")
    input_group.add_argument("-s", "--stdin", action="store_true", help="Read raw content directly from standard input (pipe).")

    parser.add_argument("-o", "--output-dir", type=Path, default=settings.paths.raw_dir, help=f"Target directory for saved JSON records (default: {settings.paths.raw_dir}).")
    parser.add_argument("--dry-run", action="store_true", help="Extract and validate content without persisting to disk.")
    parser.add_argument("--timeout", type=float, default=settings.groq.timeout_seconds, help=f"HTTP network timeout in seconds (default: {settings.groq.timeout_seconds}).")
    parser.add_argument("--format", choices=["summary", "json", "path"], default="summary", help="Stdout output format.")

    log_group = parser.add_mutually_exclusive_group()
    log_group.add_argument("-v", "--verbose", action="store_true", help="Enable detailed diagnostic logging (DEBUG level).")
    log_group.add_argument("-q", "--quiet", action="store_true", help="Suppress informational messages.")

    parser.add_argument("--version", action="version", version=f"%(prog)s (SecondSelf v{settings.app.version})")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI execution entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        log_level = logging.DEBUG
    elif args.quiet:
        log_level = logging.ERROR
    else:
        log_level = logging.INFO

    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    stdin_text: str | None = None
    if args.stdin:
        try:
            stdin_text = sys.stdin.read()
        except OSError:
            stdin_text = None
    elif not (args.text or args.url or args.file):
        try:
            is_tty = sys.stdin.isatty() if hasattr(sys.stdin, "isatty") else True
            if not is_tty:
                stdin_text = sys.stdin.read()
        except (OSError, ValueError):
            stdin_text = None

    if not (args.text or args.url or args.file or stdin_text):
        parser.print_help(sys.stderr)
        sys.stderr.write("\nError: No input specified. Provide --text, --url, --file, or pipe via stdin.\n")
        return EXIT_USAGE_ERROR

    try:
        payload, saved_path = capture(
            text=args.text,
            url=args.url,
            file_path=args.file,
            stdin_content=stdin_text,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
            timeout=args.timeout,
        )
    except (ValueError, FileNotFoundError, MalformedContentError) as exc:
        logger.error("Input error: %s", exc)
        return EXIT_IO_ERROR if isinstance(exc, FileNotFoundError) else EXIT_USAGE_ERROR
    except (ConnectionError, ExtractionError) as exc:
        logger.error("Network / Extraction error: %s", exc)
        return EXIT_NETWORK_ERROR
    except IOError as exc:
        logger.error("File I/O error: %s", exc)
        return EXIT_IO_ERROR
    except Exception as exc:
        logger.error("Unexpected runtime error: %s", exc, exc_info=args.verbose)
        return EXIT_GENERAL_ERROR

    if args.format == "json":
        print(payload.to_json(indent=2))
    elif args.format == "path":
        print(str(saved_path) if saved_path else "[dry-run]")
    else:
        title = payload.metadata.get("title", "Untitled")
        char_count = payload.metadata.get("char_count", len(payload.raw_content))
        print("✓ Captured Successfully!")
        print(f"  • ID:       {payload.id}")
        print(f"  • Source:   {payload.source_type} ({payload.source_uri or 'inline'})")
        print(f"  • Title:    {title}")
        print(f"  • Size:     {char_count:,} characters")
        if saved_path:
            print(f"  • Location: {saved_path}")
        else:
            print("  • Location: [dry-run: not saved]")

    return EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
