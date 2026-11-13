"""Filesystem storage helpers for the SecondSelf personal knowledge system.

Provides centralized persistence functions for raw captures, wiki notes,
and system index state, anchored to the project root.
"""

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
from typing import Any
import uuid

import yaml

# Ensure project root is in sys.path
_SYS_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_SYS_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SYS_PROJECT_ROOT))

logger = logging.getLogger(__name__)

try:
    from lib.constants import DEFAULT_USER_ID
except ImportError:
    try:
        from .constants import DEFAULT_USER_ID
    except ImportError:
        try:
            from lib.database import DEFAULT_USER_ID
        except ImportError:
            DEFAULT_USER_ID = "00000000-0000-0000-0000-000000000001"

try:
    from lib.models import CaptureMeta, WikiNote
except ImportError:
    from .models import CaptureMeta, WikiNote

__all__ = [
    "PROJECT_ROOT",
    "get_project_root",
    "set_project_root",
    "get_raw_dir",
    "get_wiki_dir",
    "get_data_dir",
    "get_index_path",
    "generate_capture_id",
    "write_raw_capture",
    "read_raw_captures",
    "read_all_raw_captures",
    "write_wiki_note",
    "read_wiki_notes",
    "delete_wiki_note",
    "load_index",
    "save_index",
    "content_hash",
]

def _detect_default_project_root() -> Path:
    """Detect the canonical SecondSelf project root.

    Supports SECONDSELF_ROOT / SECONDSELF_DATA_DIR environment variables,
    defaults to the repository directory (parent of lib/), and handles parent workspace runners.
    """
    env_root = os.getenv("SECONDSELF_ROOT")
    if env_root and Path(env_root).resolve().exists():
        return Path(env_root).resolve()

    env_data = os.getenv("SECONDSELF_DATA_DIR")
    if env_data and Path(env_data).resolve().exists():
        p = Path(env_data).resolve()
        if p.name == "data":
            return p.parent
        return p

    repo_dir = Path(__file__).resolve().parent.parent
    if (repo_dir / "wiki").exists() or (repo_dir / "data").exists():
        return repo_dir

    if (repo_dir / "secondself" / "wiki").exists():
        return (repo_dir / "secondself").resolve()

    return repo_dir


# Project root: defaults to the repository root directory (parent of lib/)
PROJECT_ROOT = _detect_default_project_root()
_project_root = PROJECT_ROOT

DEFAULT_INDEX: dict[str, Any] = {
    "raw_processed": {},
    "embeddings_version": "all-MiniLM-L6-v2",
    "last_graph_build": None,
}


def get_project_root() -> Path:
    """Return the active project root path."""
    return _project_root


def set_project_root(root: Path | str | None) -> None:
    """Override the project root directory (useful for testing in temp dirs)."""
    global _project_root
    if root is None:
        _project_root = _detect_default_project_root()
    else:
        _project_root = Path(root).resolve()


def get_raw_dir(root: Path | str | None = None) -> Path:
    """Return path to the raw/ captures directory."""
    base = Path(root).resolve() if root else _project_root
    return base / "raw"


def get_wiki_dir(root: Path | str | None = None) -> Path:
    """Return path to the wiki/ directory."""
    base = Path(root).resolve() if root else _project_root
    return base / "wiki"


def get_data_dir(root: Path | str | None = None) -> Path:
    """Return path to the data/ directory."""
    base = Path(root).resolve() if root else _project_root
    return base / "data"


def get_index_path(root: Path | str | None = None, user_id: str | None = None) -> Path:
    """Return path to data/index.json, or data/{user_id}/index.json for scoped users."""
    base_data = get_data_dir(root)
    if user_id and user_id != DEFAULT_USER_ID:
        scoped_dir = base_data / user_id
        scoped_dir.mkdir(parents=True, exist_ok=True)
        return scoped_dir / "index.json"
    return base_data / "index.json"


def generate_capture_id() -> str:
    """Generate a capture ID in the format YYYY-MM-DD_{uuid8}.

    Returns:
        String ID containing current UTC date and first 8 characters of UUID4.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    uid8 = uuid.uuid4().hex[:8]
    return f"{today}_{uid8}"


def _sanitize_path_segment(segment: str, field_name: str) -> str:
    """Validate and sanitize a single path segment against path traversal."""
    if not segment or not isinstance(segment, str):
        raise ValueError(f"{field_name} must be a non-empty string")
    clean = Path(segment).name
    if (
        clean != segment
        or ".." in segment
        or "/" in segment
        or "\\" in segment
        or not clean.strip()
    ):
        raise ValueError(
            f"Invalid or unsafe path component for {field_name}: {segment!r}"
        )
    return clean


def _resolve_content_filename(meta: CaptureMeta, content: str | bytes) -> str:
    """Determine a predictable, safe content filename based on type and extension."""
    c_type = (meta.type or "").lower().strip()
    if c_type == "note":
        return "content.md"
    elif c_type == "link":
        return "content.txt"
    elif c_type == "file":
        if meta.original_filename:
            orig_name = Path(meta.original_filename).name
            ext = Path(orig_name).suffix
            if ext:
                clean_ext = "".join(c for c in ext if c.isalnum() or c == ".").lower()
                if clean_ext and clean_ext != ".":
                    return f"content{clean_ext}"
        return "content.bin" if isinstance(content, bytes) else "content.txt"
    else:
        if meta.original_filename:
            orig_name = Path(meta.original_filename).name
            ext = Path(orig_name).suffix
            if ext:
                clean_ext = "".join(c for c in ext if c.isalnum() or c == ".").lower()
                if clean_ext and clean_ext != ".":
                    return f"content{clean_ext}"
        return "content.bin" if isinstance(content, bytes) else "content.txt"


def write_raw_capture(
    meta: CaptureMeta,
    content: str | bytes,
    root: Path | str | None = None,
) -> Path:
    """Save a raw capture folder containing meta.json and content file.

    Args:
        meta: CaptureMeta metadata describing the capture.
        content: Raw note/link/file content as a string or bytes.
        root: Optional root directory override (for testing).

    Returns:
        Path to the created raw capture directory (raw/{meta.id}).
    """
    if not isinstance(meta, CaptureMeta):
        raise TypeError(
            f"meta must be an instance of CaptureMeta, got {type(meta).__name__}"
        )
    if not isinstance(content, (str, bytes)):
        raise TypeError(
            f"content must be str or bytes, got {type(content).__name__}"
        )

    clean_id = _sanitize_path_segment(meta.id, "meta.id")
    raw_dir = get_raw_dir(root)
    capture_dir = (raw_dir / clean_id).resolve()

    if not capture_dir.is_relative_to(raw_dir.resolve()):
        raise ValueError(f"Path traversal detected in capture ID: {meta.id!r}")

    capture_dir.mkdir(parents=True, exist_ok=True)

    meta_file = capture_dir / "meta.json"
    meta_dict = asdict(meta)
    with meta_file.open("w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2, ensure_ascii=False)

    content_filename = _resolve_content_filename(meta, content)
    content_file = capture_dir / content_filename
    if isinstance(content, bytes):
        with content_file.open("wb") as f:
            f.write(content)
    else:
        with content_file.open("w", encoding="utf-8") as f:
            f.write(content)

    # Durable persistence: dual-write capture record into SQLite database
    try:
        from . import database as db
        db.upsert_capture(
            meta=meta,
            content=content,
            content_path=str(content_file),
            status="pending",
            root=root,
        )
    except Exception as e:
        logger.warning(f"Dual-write to SQLite failed for capture {meta.id}: {e}")

    return capture_dir


def read_all_raw_captures(
    root: Path | str | None = None,
    user_id: str | None = None,
) -> list[CaptureMeta]:
    """Read all raw captures (both processed and unprocessed) optionally filtered by user_id.

    Args:
        root: Optional root directory override (for testing).
        user_id: Optional user ID to scope captures to.

    Returns:
        List of all CaptureMeta instances found on filesystem or SQLite for the user.
    """
    raw_dir = get_raw_dir(root)
    captures: list[CaptureMeta] = []
    seen_ids: set[str] = set()

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
                cap_user = str(data.get("user_id") or DEFAULT_USER_ID)
                if user_id is not None and cap_user != user_id:
                    continue
                meta = CaptureMeta(
                    id=str(data["id"]),
                    timestamp=str(data["timestamp"]),
                    type=str(data["type"]),
                    source=str(data["source"]),
                    original_filename=(
                        str(data["original_filename"])
                        if data.get("original_filename") is not None
                        else None
                    ),
                    content_hash=str(data["content_hash"]),
                    user_id=cap_user,
                )
                captures.append(meta)
                seen_ids.add(meta.id)
            except Exception:
                continue

    # Augment with captures stored in SQLite if not already on filesystem
    try:
        from . import database as db
        db_captures = db.list_captures(user_id=user_id, root=root)
        for row in db_captures:
            cid = row["id"]
            if cid not in seen_ids:
                captures.append(
                    CaptureMeta(
                        id=cid,
                        timestamp=row["timestamp"],
                        type=row["type"],
                        source=row["source"],
                        original_filename=row["original_filename"],
                        content_hash=row["content_hash"],
                        user_id=row.get("user_id", user_id or DEFAULT_USER_ID),
                    )
                )
                seen_ids.add(cid)
    except Exception:
        pass

    return captures


def read_raw_captures(
    root: Path | str | None = None,
    user_id: str | None = None,
) -> list[CaptureMeta]:
    """Read unprocessed raw captures from raw/{id}/meta.json optionally filtered by user_id.

    Args:
        root: Optional root directory override (for testing).
        user_id: Optional user ID to scope captures to.

    Returns:
        List of CaptureMeta instances for items not present in raw_processed.
    """
    index = load_index(root=root, user_id=user_id)
    raw_processed = index.get("raw_processed", {})
    if not isinstance(raw_processed, dict):
        raw_processed = {}

    raw_dir = get_raw_dir(root)
    captures: list[CaptureMeta] = []
    seen_ids: set[str] = set()

    required_fields = {
        "id",
        "timestamp",
        "type",
        "source",
        "original_filename",
        "content_hash",
    }

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
            except Exception:
                continue

            if not isinstance(data, dict):
                continue

            missing = required_fields - set(data.keys())
            if missing:
                continue

            cap_user = str(data.get("user_id") or DEFAULT_USER_ID)
            if user_id is not None and cap_user != user_id:
                continue

            meta = CaptureMeta(
                id=str(data["id"]),
                timestamp=str(data["timestamp"]),
                type=str(data["type"]),
                source=str(data["source"]),
                original_filename=(
                    str(data["original_filename"])
                    if data["original_filename"] is not None
                    else None
                ),
                content_hash=str(data["content_hash"]),
                user_id=cap_user,
            )

            if meta.id not in raw_processed and item.name not in raw_processed:
                captures.append(meta)
                seen_ids.add(meta.id)

    # Augment with pending captures from SQLite for this user
    try:
        from . import database as db
        pending_db = db.list_captures(status="pending", user_id=user_id, root=root)
        for row in pending_db:
            cid = row["id"]
            if cid not in seen_ids and cid not in raw_processed:
                captures.append(
                    CaptureMeta(
                        id=cid,
                        timestamp=row["timestamp"],
                        type=row["type"],
                        source=row["source"],
                        original_filename=row["original_filename"],
                        content_hash=row["content_hash"],
                        user_id=row.get("user_id", user_id or DEFAULT_USER_ID),
                    )
                )
                seen_ids.add(cid)
    except Exception:
        pass

    return captures


def write_wiki_note(note: WikiNote, root: Path | str | None = None) -> Path:
    """Write a WikiNote as Markdown with YAML frontmatter to wiki/{para}/{id}.md.

    Args:
        note: WikiNote instance to persist.
        root: Optional root directory override (for testing).

    Returns:
        Path to the saved markdown file.
    """
    if not isinstance(note, WikiNote):
        raise TypeError(
            f"note must be an instance of WikiNote, got {type(note).__name__}"
        )

    clean_para = _sanitize_path_segment(note.para, "note.para")
    clean_id = _sanitize_path_segment(note.id, "note.id")

    wiki_dir = get_wiki_dir(root)
    target_dir = (wiki_dir / clean_para).resolve()
    if not target_dir.is_relative_to(wiki_dir.resolve()):
        raise ValueError(f"Path traversal detected in note.para: {note.para!r}")

    target_dir.mkdir(parents=True, exist_ok=True)
    note_file = target_dir / f"{clean_id}.md"

    note_user = note.user_id or DEFAULT_USER_ID
    frontmatter = {
        "id": note.id,
        "raw_id": note.raw_id,
        "para": note.para,
        "tags": list(note.tags) if note.tags is not None else [],
        "summary": note.summary,
        "created": note.created,
        "links": list(note.links) if note.links is not None else [],
        "user_id": note_user,
    }
    fm_yaml = yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True)

    body_text = note.body or ""
    if body_text.startswith("\r\n"):
        body_text = body_text[2:]
    elif body_text.startswith("\n"):
        body_text = body_text[1:]

    content = (
        f"---\n{fm_yaml}---\n\n{body_text}"
        if body_text
        else f"---\n{fm_yaml}---\n"
    )

    with note_file.open("w", encoding="utf-8") as f:
        f.write(content)

    # Durable persistence: dual-write wiki note into SQLite database
    try:
        from . import database as db
        db.upsert_note(note, root=root)
    except Exception as e:
        logger.warning(f"Dual-write to SQLite failed for note {note.id}: {e}")

    return note_file


def read_wiki_notes(
    root: Path | str | None = None,
    user_id: str | None = None,
) -> list[WikiNote]:
    """Discover, parse, and return all WikiNote markdown files under wiki/ scoped to user_id.

    Args:
        root: Optional root directory override (for testing).
        user_id: Optional user ID to filter notes by.

    Returns:
        List of parsed WikiNote instances.
    """
    wiki_dir = get_wiki_dir(root)
    notes: list[WikiNote] = []
    seen_ids: set[str] = set()
    required_fields = {
        "id",
        "raw_id",
        "para",
        "tags",
        "summary",
        "created",
        "links",
    }

    if wiki_dir.exists():
        for note_path in sorted(wiki_dir.rglob("*.md")):
            try:
                with note_path.open("r", encoding="utf-8") as f:
                    text = f.read()
            except Exception as e:
                raise ValueError(f"Failed to read wiki note at {note_path}: {e}") from e

            text = text.lstrip("\ufeff")
            lines = text.splitlines(keepends=True)
            if not lines or lines[0].strip() != "---":
                raise ValueError(
                    f"Malformed wiki note at {note_path}: missing opening '---' delimiter"
                )

            closing_idx = -1
            for i in range(1, len(lines)):
                if lines[i].strip() == "---":
                    closing_idx = i
                    break

            if closing_idx == -1:
                raise ValueError(
                    f"Malformed wiki note at {note_path}: missing closing '---' delimiter"
                )

            fm_text = "".join(lines[1:closing_idx])
            body_lines = lines[closing_idx + 1 :]
            body_text = "".join(body_lines)

            if body_text.startswith("\r\n"):
                body_text = body_text[2:]
            elif body_text.startswith("\n"):
                body_text = body_text[1:]

            try:
                fm_data = yaml.safe_load(fm_text)
            except Exception as e:
                raise ValueError(
                    f"Failed to parse YAML frontmatter in {note_path}: {e}"
                ) from e

            if not isinstance(fm_data, dict):
                raise ValueError(
                    f"Frontmatter in {note_path} must be a mapping, got {type(fm_data).__name__}"
                )

            missing = required_fields - set(fm_data.keys())
            if missing:
                raise ValueError(
                    f"Wiki note {note_path} missing required frontmatter fields: {sorted(missing)}"
                )

            note_owner = str(fm_data.get("user_id") or DEFAULT_USER_ID)
            if user_id is not None and note_owner != user_id:
                continue

            raw_tags = fm_data.get("tags")
            if raw_tags is None:
                tags = []
            elif isinstance(raw_tags, list):
                tags = [str(t) for t in raw_tags]
            else:
                raise ValueError(
                    f"Frontmatter 'tags' in {note_path} must be a list, got {type(raw_tags).__name__}"
                )

            raw_links = fm_data.get("links")
            if raw_links is None:
                links = []
            elif isinstance(raw_links, list):
                links = [str(l) for l in raw_links]
            else:
                raise ValueError(
                    f"Frontmatter 'links' in {note_path} must be a list, got {type(raw_links).__name__}"
                )

            nid = str(fm_data["id"])
            note = WikiNote(
                id=nid,
                raw_id=str(fm_data["raw_id"]),
                para=str(fm_data["para"]),
                tags=tags,
                summary=str(fm_data["summary"]),
                created=str(fm_data["created"]),
                links=links,
                body=body_text,
                user_id=note_owner,
            )
            notes.append(note)
            seen_ids.add(nid)

    # Augment with notes persisted in SQLite for this user
    try:
        from . import database as db
        db_notes = db.list_notes(user_id=user_id, root=root)
        for dn in db_notes:
            if dn.id not in seen_ids:
                notes.append(dn)
                seen_ids.add(dn.id)
    except Exception:
        pass

    return notes


def delete_wiki_note(
    note_id: str,
    user_id: str | None = None,
    root: Path | str | None = None,
) -> bool:
    """Delete a wiki note from disk and SQLite scoped to user_id."""
    effective_user_id = user_id or DEFAULT_USER_ID
    wiki_dir = get_wiki_dir(root)
    deleted_from_disk = False

    if wiki_dir.exists():
        for note_file in wiki_dir.rglob(f"{note_id}.md"):
            try:
                with note_file.open("r", encoding="utf-8") as f:
                    text = f.read()
                lines = text.splitlines()
                fm_lines = []
                in_fm = False
                for line in lines:
                    if line.strip() == "---":
                        if not in_fm:
                            in_fm = True
                            continue
                        else:
                            break
                    if in_fm:
                        fm_lines.append(line)
                fm_data = yaml.safe_load("\n".join(fm_lines)) or {}
                owner = fm_data.get("user_id", DEFAULT_USER_ID)
                if user_id is None or owner == effective_user_id:
                    note_file.unlink(missing_ok=True)
                    deleted_from_disk = True
            except Exception as e:
                logger.warning(f"Failed to delete disk note {note_file}: {e}")

    try:
        from . import database as db
        db_deleted = db.delete_note(note_id, user_id=effective_user_id, root=root)
        return deleted_from_disk or db_deleted
    except Exception as e:
        logger.warning(f"Failed to delete note {note_id} from SQLite: {e}")
        return deleted_from_disk


def load_index(
    root: Path | str | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    """Load data/index.json if present; otherwise return the default template.

    Args:
        root: Optional root directory override (for testing).
        user_id: Optional user ID for scoped index.

    Returns:
        Index dictionary with raw_processed, embeddings_version, and last_graph_build.
    """
    index_path = get_index_path(root, user_id=user_id)
    if not index_path.exists():
        return {
            "raw_processed": {},
            "embeddings_version": "all-MiniLM-L6-v2",
            "last_graph_build": None,
        }

    try:
        with index_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise ValueError(
            f"Failed to read JSON from index file {index_path}: {e}"
        ) from e

    if not isinstance(data, dict):
        raise ValueError(
            f"Index file at {index_path} must contain a JSON object, got {type(data).__name__}"
        )

    if "raw_processed" in data and not isinstance(data["raw_processed"], dict):
        raise ValueError(
            f"Index 'raw_processed' must be a mapping, got {type(data['raw_processed']).__name__}"
        )

    return data


def save_index(
    index: dict[str, Any],
    root: Path | str | None = None,
    user_id: str | None = None,
) -> Path:
    """Save the supplied index mapping atomically to data/index.json (or scoped user index).

    Args:
        index: Full index mapping to serialize.
        root: Optional root directory override (for testing).
        user_id: Optional user ID for scoped index.

    Returns:
        Path to the saved index.json file.
    """
    if not isinstance(index, dict):
        raise ValueError(
            f"index must be an instance of dict, got {type(index).__name__}"
        )

    index_path = get_index_path(root, user_id=user_id)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_file = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=index_path.parent,
            delete=False,
            suffix=".tmp",
        ) as f:
            tmp_file = Path(f.name)
            json.dump(index, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_file, index_path)

        # Synchronize processed state with SQLite database
        try:
            from . import database as db
            raw_proc = index.get("raw_processed", {})
            if isinstance(raw_proc, dict):
                for cap_id in raw_proc.keys():
                    db.update_capture_status(str(cap_id), "completed", user_id=user_id, root=root)
            if index.get("last_graph_build"):
                db.set_system_metadata("last_graph_build", str(index["last_graph_build"]), root=root)
        except Exception as e:
            logger.debug(f"Non-fatal error updating SQLite from index: {e}")

        return index_path
    except Exception as e:
        if tmp_file and tmp_file.exists():
            try:
                tmp_file.unlink()
            except OSError:
                pass
        raise IOError(f"Failed to save index to {index_path}: {e}") from e


def content_hash(data: str | bytes) -> str:
    """Compute the hexadecimal SHA-256 digest of string or raw byte data.

    Args:
        data: String (encoded as UTF-8) or raw bytes to hash.

    Returns:
        Hexadecimal SHA-256 digest string.
    """
    if isinstance(data, str):
        data_bytes = data.encode("utf-8")
    elif isinstance(data, bytes):
        data_bytes = data
    else:
        raise TypeError(
            f"Expected str or bytes for content_hash, got {type(data).__name__}"
        )

    return hashlib.sha256(data_bytes).hexdigest()

