"""Core capture functions and CLI for the SecondSelf personal knowledge system.

Provides functions and a command-line interface to capture notes, links, and files
into the raw/ storage with unique identifiers, timestamps, metadata, and content
deduplication hashing.

Interactive mode:
    Run `python capture.py` without arguments to capture a note from stdin.
    End input with Ctrl+Z then Enter on Windows, or Ctrl+D on macOS/Linux.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

# Ensure repository root is on sys.path so 'lib' is importable regardless of caller's cwd
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lib.models import CaptureMeta, CaptureResult
import lib.storage as storage
from lib.storage import content_hash, generate_capture_id, write_raw_capture

__all__ = [
    "capture_note",
    "capture_link",
    "capture_file",
]


def find_duplicate_capture(target_hash: str) -> str | None:
    """Check if content with target_hash has already been captured.

    Checks all captures in raw/ (including processed ones) and data/index.json,
    ensuring duplicates are detected even if earlier captures are marked processed.

    Args:
        target_hash: Hexadecimal SHA-256 digest to search for.

    Returns:
        The capture ID of the existing capture if found, else None.
    """
    raw_dir = storage.get_raw_dir()
    if raw_dir.exists():
        for item in sorted(raw_dir.iterdir()):
            if not item.is_dir() or item.name.startswith("."):
                continue
            meta_file = item / "meta.json"
            if not meta_file.exists():
                continue
            try:
                with meta_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                h = data.get("content_hash")
                if h:
                    clean_h = str(h).removeprefix("sha256:")
                    clean_target = target_hash.removeprefix("sha256:")
                    if clean_h == clean_target:
                        return str(data.get("id") or item.name)
            except Exception:
                continue

    try:
        index = storage.load_index()
        raw_processed = index.get("raw_processed", {})
        if isinstance(raw_processed, dict):
            for cap_id, val in raw_processed.items():
                h = None
                if isinstance(val, dict):
                    h = val.get("content_hash")
                elif isinstance(val, str):
                    h = val
                if h:
                    clean_h = str(h).removeprefix("sha256:")
                    clean_target = target_hash.removeprefix("sha256:")
                    if clean_h == clean_target:
                        return str(cap_id)
    except Exception:
        pass

    # Check SQLite database for existing content hash
    try:
        from lib import database as db
        clean_target = target_hash.removeprefix("sha256:")
        with db.db_session() as conn:
            row = conn.execute(
                "SELECT id FROM captures WHERE content_hash = ? OR content_hash = ? LIMIT 1;",
                (target_hash, clean_target),
            ).fetchone()
            if row:
                return str(row["id"])
    except Exception:
        pass

    return None


def capture_note(text: str) -> CaptureResult:
    """Capture a plain text or Markdown note to the raw capture store.

    Args:
        text: Plain text or Markdown content of the note.

    Returns:
        CaptureResult dataclass describing the capture.

    Raises:
        ValueError: If note text is empty or whitespace-only.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Note text cannot be empty")

    capture_id = generate_capture_id()
    now_iso = datetime.now(timezone.utc).isoformat()
    content = text
    c_hash = content_hash(content)

    existing_id = find_duplicate_capture(c_hash)
    if existing_id:
        print(
            f"Warning: content with hash {c_hash} was already captured at raw/{existing_id}",
            file=sys.stderr,
        )

    meta = CaptureMeta(
        id=capture_id,
        timestamp=now_iso,
        type="note",
        source="cli",
        original_filename=None,
        content_hash=c_hash,
    )

    try:
        saved_path = write_raw_capture(meta, content)
    except Exception:
        raw_dir = storage.get_raw_dir()
        partial_dir = raw_dir / capture_id
        if partial_dir.exists():
            shutil.rmtree(partial_dir, ignore_errors=True)
        raise

    print(f"Captured → raw/{capture_id}")

    return CaptureResult(
        id=capture_id,
        path=str(saved_path),
        type="note",
    )


def capture_link(url: str, notes: str = "") -> CaptureResult:
    """Capture a URL and optional notes to the raw capture store.

    Stores the URL in CaptureMeta.source for Phase 2 web extraction and
    preserves optional user notes in the raw content file (content.txt).

    Args:
        url: The web URL being captured.
        notes: Optional user notes or annotations accompanying the link.

    Returns:
        CaptureResult dataclass describing the capture.

    Raises:
        ValueError: If URL is empty or whitespace-only.
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError("URL cannot be empty")

    capture_id = generate_capture_id()
    now_iso = datetime.now(timezone.utc).isoformat()

    if notes:
        content = f"{url}\n\n{notes}"
    else:
        content = url

    c_hash = content_hash(content)

    existing_id = find_duplicate_capture(c_hash)
    if existing_id:
        print(
            f"Warning: content with hash {c_hash} was already captured at raw/{existing_id}",
            file=sys.stderr,
        )

    meta = CaptureMeta(
        id=capture_id,
        timestamp=now_iso,
        type="link",
        source=url,
        original_filename=None,
        content_hash=c_hash,
    )

    try:
        saved_path = write_raw_capture(meta, content)
    except Exception:
        raw_dir = storage.get_raw_dir()
        partial_dir = raw_dir / capture_id
        if partial_dir.exists():
            shutil.rmtree(partial_dir, ignore_errors=True)
        raise

    print(f"Captured → raw/{capture_id}")

    return CaptureResult(
        id=capture_id,
        path=str(saved_path),
        type="link",
    )


def capture_file(path: str | Path) -> CaptureResult:
    """Capture a local file to the raw capture store.

    Preserves original file bytes exactly without transformation,
    records the original basename in CaptureMeta.original_filename,
    and preserves the file extension in raw/{capture_id}/content.{ext}.

    Args:
        path: Filesystem path to the local file (str or Path).

    Returns:
        CaptureResult dataclass describing the capture.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the path is not a regular file.
        OSError: If reading the file fails.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not file_path.is_file():
        raise ValueError(f"Path is not a regular file: {path}")

    try:
        content = file_path.read_bytes()
    except Exception as e:
        raise OSError(f"Cannot read file {path}: {e}") from e

    capture_id = generate_capture_id()
    now_iso = datetime.now(timezone.utc).isoformat()
    c_hash = content_hash(content)
    orig_name = file_path.name

    existing_id = find_duplicate_capture(c_hash)
    if existing_id:
        print(
            f"Warning: content with hash {c_hash} was already captured at raw/{existing_id}",
            file=sys.stderr,
        )

    meta = CaptureMeta(
        id=capture_id,
        timestamp=now_iso,
        type="file",
        source=str(path),
        original_filename=orig_name,
        content_hash=c_hash,
    )

    try:
        saved_path = write_raw_capture(meta, content)
    except Exception:
        raw_dir = storage.get_raw_dir()
        partial_dir = raw_dir / capture_id
        if partial_dir.exists():
            shutil.rmtree(partial_dir, ignore_errors=True)
        raise

    print(f"Captured → raw/{capture_id}")

    return CaptureResult(
        id=capture_id,
        path=str(saved_path),
        type="file",
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser for SecondSelf capture CLI."""
    parser = argparse.ArgumentParser(
        prog="capture.py",
        description="SecondSelf capture CLI: capture notes, links, and files into raw storage.",
        epilog=(
            "Interactive mode:\n"
            "  Run without arguments to capture a note from stdin.\n"
            "  End interactive input with Ctrl+Z then Enter on Windows, or Ctrl+D on macOS/Linux."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(
        dest="command",
        title="subcommands",
        description="Available capture commands (run without arguments for stdin note capture)",
    )

    # Subcommand: note
    p_note = subparsers.add_parser(
        "note",
        help="Capture a text or Markdown note",
        description="Capture plain text or Markdown note into raw storage.",
    )
    p_note.add_argument(
        "text",
        help="Text or Markdown content to capture",
    )

    # Subcommand: link
    p_link = subparsers.add_parser(
        "link",
        help="Capture a web link / URL",
        description="Capture a web URL link into raw storage for Phase 2 web extraction.",
    )
    p_link.add_argument(
        "url",
        help="The URL to capture",
    )

    # Subcommand: file
    p_file = subparsers.add_parser(
        "file",
        help="Capture a local file",
        description="Capture a local file byte-for-byte into raw storage.",
    )
    p_file.add_argument(
        "path",
        help="Path to the local file to capture",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for SecondSelf capture.

    Args:
        argv: Command-line argument list (defaults to sys.argv[1:]).

    Returns:
        Integer exit code (0 for success, non-zero for error).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "note":
            capture_note(args.text)
        elif args.command == "link":
            capture_link(args.url)
        elif args.command == "file":
            capture_file(args.path)
        elif args.command is None:
            if sys.stdin.isatty():
                print(
                    "Enter note content (press Ctrl+Z then Enter on Windows, or Ctrl+D on macOS/Linux to finish):"
                )
            text = sys.stdin.read()
            capture_note(text)
        else:
            parser.print_help(sys.stderr)
            return 2
        return 0
    except KeyboardInterrupt:
        print("\nCapture cancelled by user.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
