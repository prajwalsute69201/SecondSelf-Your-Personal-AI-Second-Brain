"""Text extraction helpers for SecondSelf raw captures and files.

Provides dedicated and unified extraction helpers for:
- Markdown and plain text notes (direct decoding, UTF-8 and UTF-8 BOM).
- Web links (fetching HTML via requests and stripping via BeautifulSoup,
  falling back gracefully to the URL string on network/parsing failure).
- Local files, including PDFs via pypdf (ordered page extraction with
  fallback to filename) and safe binary file handling.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import sys
from typing import Any

from bs4 import BeautifulSoup
import pypdf
import requests

# Ensure repository root is on sys.path so 'lib' is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from lib.models import CaptureMeta
    import lib.storage as storage
except ImportError:
    from .models import CaptureMeta  # type: ignore[no-redef]
    from . import storage  # type: ignore[no-redef]

__all__ = [
    "extract_text",
    "extract_text_from_note",
    "extract_text_from_link",
    "extract_text_from_pdf",
    "extract_text_from_file",
]

# Standard text file extensions
_KNOWN_TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".rst",
    ".json",
    ".yaml",
    ".yml",
    ".csv",
    ".tsv",
    ".py",
    ".js",
    ".html",
    ".xml",
    ".css",
    ".sh",
    ".log",
    ".ini",
    ".toml",
}


def extract_text_from_note(path: Path | str) -> str:
    """Read and extract text from a note Markdown or plain text file.

    Handles UTF-8 and UTF-8 BOM encoding correctly.

    Args:
        path: Filesystem path to the note file (e.g. content.md).

    Returns:
        str: Extracted note text content.

    Raises:
        FileNotFoundError: If the note file does not exist.
        ValueError: If path points to a directory.
        OSError: If opening or reading the file fails.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Note file not found: {path}")
    if not file_path.is_file():
        raise ValueError(f"Path is not a regular file: {path}")

    raw_bytes = file_path.read_bytes()
    # Decode with utf-8-sig to automatically strip UTF-8 BOM if present
    try:
        text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        # Fallback to utf-8 with replacement for corrupted or non-standard characters
        text = raw_bytes.decode("utf-8", errors="replace")
    return text.replace("\r\n", "\n")


def extract_text_from_link(
    url_or_path: str | Path,
    timeout: float = 10.0,
    session: requests.Session | None = None,
) -> str:
    """Extract readable text from a URL or link content file, falling back to the URL string.

    Fetches the web page, strips HTML tags (scripts, styles, nav, etc.) with BeautifulSoup,
    and returns clean text. If network errors, timeouts, HTTP errors, or non-HTML content occur,
    safely falls back to the URL string (and preserves user notes if present).

    Args:
        url_or_path: Web URL string (e.g. 'https://...') or path to a content.txt file.
        timeout: Network request timeout in seconds (default: 10.0).
        session: Optional requests.Session for connection pooling or test mocking.

    Returns:
        str: Extracted web page text, or fallback URL string.
    """
    target_url = ""
    notes = ""
    url_fallback = ""

    # Check if url_or_path is an existing local file (e.g. raw/{id}/content.txt)
    is_file = False
    if isinstance(url_or_path, Path):
        is_file = url_or_path.is_file()
    elif isinstance(url_or_path, str) and not url_or_path.startswith(("http://", "https://")):
        try:
            is_file = Path(url_or_path).is_file()
        except OSError:
            is_file = False

    if is_file:
        file_path = Path(url_or_path)
        try:
            content_text = file_path.read_text(encoding="utf-8-sig")
            lines = content_text.splitlines()
            if lines:
                target_url = lines[0].strip()
                if len(lines) > 1:
                    notes = "\n".join(lines[1:]).strip()
            url_fallback = content_text.strip()
        except Exception:
            url_fallback = str(url_or_path)
            target_url = str(url_or_path).strip()
    else:
        target_url = str(url_or_path).strip()
        url_fallback = target_url

    if not target_url.startswith(("http://", "https://")):
        return url_fallback

    # Fetch webpage and strip HTML
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36 SecondSelf/1.0"
            )
        }
        getter = session.get if session else requests.get
        response = getter(target_url, timeout=timeout, headers=headers)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "").lower()
        if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
            # Non-HTML resource (e.g. raw PDF or image URL); fallback to URL string
            return url_fallback

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer", "noscript", "svg"]):
            tag.decompose()

        extracted_text = soup.get_text(separator="\n", strip=True)
        cleaned_text = re.sub(r"\n{3,}", "\n\n", extracted_text).strip()

        if not cleaned_text:
            return url_fallback

        if notes:
            return f"{cleaned_text}\n\nUser Notes:\n{notes}"
        return cleaned_text

    except Exception:
        # Fallback to URL string on network failure, timeout, or bad status
        return url_fallback


def extract_text_from_pdf(
    path: Path | str,
    fallback_filename: str | None = None,
) -> str:
    """Extract text from a PDF document page by page in original order using pypdf.

    If extraction fails (malformed PDF, unreadable, encrypted) or the PDF contains
    no extractable text (scanned document), falls back to the filename as specified
    in the implementation plan.

    Args:
        path: Filesystem path to the PDF file.
        fallback_filename: Optional filename override for fallback (defaults to path.name).

    Returns:
        str: Extracted text from all pages in order, or fallback filename.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        ValueError: If path points to a directory.
        OSError: If opening or reading the file from the filesystem fails.
    """
    file_path = Path(path)
    fallback = fallback_filename or file_path.name

    if not file_path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")
    if not file_path.is_file():
        raise ValueError(f"Path is not a regular file: {path}")

    try:
        reader = pypdf.PdfReader(str(file_path))
        page_texts: list[str] = []

        for page in reader.pages:
            text = page.extract_text()
            if text and text.strip():
                page_texts.append(text.strip())

        combined_text = "\n\n".join(page_texts).strip()
        if combined_text:
            return combined_text

        # If PDF contained 0 extractable text characters (e.g. scanned image PDF)
        return fallback

    except Exception:
        # Fallback to filename on malformed, corrupted, or unreadable PDF
        return fallback


def extract_text_from_file(
    path: Path | str,
    original_filename: str | None = None,
) -> str:
    """Extract readable text from a captured file (PDF, plain text, or binary).

    - For PDF files (.pdf): extracts text page-by-page via pypdf with filename fallback.
    - For text files (.txt, .md, code, etc.): decodes as UTF-8 / UTF-8 BOM.
    - For unsupported binary files: safely avoids garbage decoding and falls back to filename.

    Args:
        path: Filesystem path to the file.
        original_filename: Optional original filename from metadata (defaults to path.name).

    Returns:
        str: Extracted text, or original filename as fallback.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If path points to a directory.
        OSError: If opening or reading the file fails.
    """
    file_path = Path(path)
    fallback = original_filename or file_path.name

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not file_path.is_file():
        raise ValueError(f"Path is not a regular file: {path}")

    # Check extension
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path, fallback_filename=fallback)

    if ext in _KNOWN_TEXT_EXTENSIONS:
        try:
            return extract_text_from_note(file_path)
        except Exception:
            return fallback

    # For files with unknown/unsupported extensions, inspect content safely
    try:
        raw_bytes = file_path.read_bytes()
    except Exception as e:
        raise OSError(f"Cannot read file {path}: {e}") from e

    if not raw_bytes:
        return ""

    # Check for null byte in the first 1024 bytes (strong binary indicator)
    sample = raw_bytes[:1024]
    if b"\x00" in sample:
        # Binary file; avoid corrupted text decoding and fall back to filename
        return fallback

    # Try strict UTF-8 decoding on the entire file
    try:
        return raw_bytes.decode("utf-8-sig").replace("\r\n", "\n")
    except UnicodeDecodeError:
        # Binary or incompatible encoding; fall back to filename
        return fallback


def extract_text(
    source: Path | str | CaptureMeta,
    root: Path | str | None = None,
    timeout: float = 10.0,
) -> str:
    """Extract readable text from a raw capture directory, CaptureMeta, or file path.

    Central dispatcher for Phase 2 classification (Task 2.1.4):
    Inspects capture type ('note', 'link', 'file') and invokes the appropriate
    extraction helper, returning text ready for classification.

    Args:
        source: Path to raw/{capture_id} directory, CaptureMeta instance, or file path.
        root: Optional project root override for resolving paths.
        timeout: Network timeout for link extraction.

    Returns:
        str: Extracted text content.

    Raises:
        FileNotFoundError: If capture directory or content file is missing.
        ValueError: If capture type is unknown or invalid.
    """
    if isinstance(source, CaptureMeta):
        meta = source
        raw_dir = storage.get_raw_dir(root)
        capture_dir = raw_dir / meta.id
    elif isinstance(source, (str, Path)):
        src_path = Path(source)
        # Check if it's a directory containing meta.json
        if src_path.is_dir() and (src_path / "meta.json").is_file():
            capture_dir = src_path
            # Read meta.json
            import json

            with (capture_dir / "meta.json").open("r", encoding="utf-8") as f:
                meta_data = json.load(f)
            meta = CaptureMeta(
                id=str(meta_data["id"]),
                timestamp=str(meta_data["timestamp"]),
                type=str(meta_data["type"]),
                source=str(meta_data["source"]),
                original_filename=(
                    str(meta_data["original_filename"])
                    if meta_data.get("original_filename") is not None
                    else None
                ),
                content_hash=str(meta_data["content_hash"]),
            )
        else:
            # Standalone file or URL string
            if isinstance(source, str) and source.startswith(("http://", "https://")):
                return extract_text_from_link(source, timeout=timeout)
            if src_path.is_file():
                if src_path.suffix.lower() == ".pdf":
                    return extract_text_from_pdf(src_path)
                return extract_text_from_file(src_path)
            raise FileNotFoundError(f"Source path not found: {source}")
    else:
        raise TypeError(f"Unsupported source type for extraction: {type(source).__name__}")

    if not capture_dir.is_dir():
        raise FileNotFoundError(f"Capture directory not found: {capture_dir}")

    c_type = (meta.type or "").lower().strip()

    if c_type == "note":
        # Look for content.md or content.txt
        content_file = capture_dir / "content.md"
        if not content_file.is_file():
            # Check for alternative content file
            candidates = [f for f in capture_dir.iterdir() if f.name.startswith("content.") and f.is_file()]
            if candidates:
                content_file = candidates[0]
            else:
                raise FileNotFoundError(f"Missing content file in note capture: {capture_dir}")
        return extract_text_from_note(content_file)

    elif c_type == "link":
        content_file = capture_dir / "content.txt"
        if content_file.is_file():
            return extract_text_from_link(content_file, timeout=timeout)
        # Fallback to URL in meta.source
        return extract_text_from_link(meta.source, timeout=timeout)

    elif c_type == "file":
        # Find content.* in capture_dir (excluding meta.json)
        candidates = [
            f for f in capture_dir.iterdir()
            if f.is_file() and f.name != "meta.json" and f.name.startswith("content")
        ]
        if not candidates:
            # If no content file found, fall back to original filename
            return meta.original_filename or capture_dir.name
        content_file = candidates[0]
        return extract_text_from_file(content_file, original_filename=meta.original_filename)

    else:
        raise ValueError(f"Unknown capture type for extraction: {meta.type!r}")
