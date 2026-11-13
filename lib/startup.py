"""Startup Data Recovery, Migration, and Restoration Engine for SecondSelf.

Ensures that all captured notes, uploaded files, processed knowledge,
graph nodes, relationships, and metadata persist permanently and are
automatically restored on application startup, system restart, or deployment.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
from typing import Any

from . import database as db
from . import storage
from .models import CaptureMeta, GraphEdge, GraphNode, WikiNote

logger = logging.getLogger(__name__)

__all__ = [
    "sync_filesystem_to_db",
    "sync_db_to_filesystem",
    "ensure_graph_populated",
    "ensure_data_restored",
]

_STARTUP_COMPLETED = False


def sync_filesystem_to_db(root: Path | str | None = None) -> dict[str, int]:
    """Scan filesystem stores and synchronize any records into SQLite database.

    Idempotent: preserves existing IDs, timestamps, and metadata without duplicates.
    """
    proj_root = Path(root).resolve() if root else storage.get_project_root()
    synced_captures = 0
    synced_notes = 0

    # 1. Read index to know which captures are processed
    index_data = storage.load_index(root=proj_root)
    raw_processed = index_data.get("raw_processed", {})
    if not isinstance(raw_processed, dict):
        raw_processed = {}

    # 2. Sync raw captures from filesystem -> DB
    raw_dir = storage.get_raw_dir(root=proj_root)
    if raw_dir.exists():
        for item in sorted(raw_dir.iterdir()):
            if not item.is_dir() or item.name.startswith("."):
                continue
            meta_path = item / "meta.json"
            if not meta_path.exists():
                continue

            try:
                with meta_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)

                cap_id = str(data.get("id") or item.name)
                # Check if already in DB
                existing = db.get_capture(cap_id, root=proj_root)

                # Determine content file path
                content_file = None
                for c_name in ("content.md", "content.txt", "content.bin", "content.pdf"):
                    p = item / c_name
                    if p.exists():
                        content_file = p
                        break
                if not content_file:
                    for p in item.iterdir():
                        if p.is_file() and p.name != "meta.json":
                            content_file = p
                            break

                raw_content_text = None
                if content_file and content_file.suffix.lower() in (".md", ".txt", ".json", ".py", ".csv"):
                    try:
                        raw_content_text = content_file.read_text(encoding="utf-8", errors="replace")
                    except Exception:
                        pass

                is_processed = (cap_id in raw_processed or item.name in raw_processed)
                status = "completed" if is_processed else "pending"

                meta = CaptureMeta(
                    id=cap_id,
                    timestamp=str(data.get("timestamp", datetime.now(timezone.utc).isoformat())),
                    type=str(data.get("type", "note")),
                    source=str(data.get("source", "cli")),
                    original_filename=data.get("original_filename"),
                    content_hash=str(data.get("content_hash", "")),
                )

                db.upsert_capture(
                    meta=meta,
                    content=raw_content_text,
                    content_path=str(content_file) if content_file else None,
                    status=status if not existing else existing["status"],
                    root=proj_root,
                )
                if not existing:
                    synced_captures += 1
            except Exception as e:
                logger.warning(f"Error syncing capture {item.name} to DB: {e}")

    # 3. Sync wiki notes from filesystem -> DB
    try:
        wiki_notes = storage.read_wiki_notes(root=proj_root)
        for note in wiki_notes:
            existing_note = db.get_note(note.id, root=proj_root)
            db.upsert_note(note, root=proj_root)
            if not existing_note:
                synced_notes += 1
    except Exception as e:
        logger.warning(f"Error syncing wiki notes to DB: {e}")

    # 4. Sync graph from graph.json -> DB if graph_nodes table is empty
    existing_graph = db.load_graph_records(root=proj_root)
    if not existing_graph or not existing_graph.get("nodes"):
        graph_path = proj_root / "data" / "graph.json"
        if graph_path.exists():
            try:
                with graph_path.open("r", encoding="utf-8") as f:
                    g_data = json.load(f)
                if isinstance(g_data, dict) and g_data.get("nodes"):
                    db.save_graph_records(
                        nodes=g_data.get("nodes", []),
                        edges=g_data.get("edges", []),
                        metadata=g_data.get("metadata", {}),
                        root=proj_root,
                    )
            except Exception as e:
                logger.warning(f"Error syncing graph.json to DB: {e}")

    return {"synced_captures": synced_captures, "synced_notes": synced_notes}


def sync_db_to_filesystem(root: Path | str | None = None) -> dict[str, int]:
    """Restore filesystem artifacts from SQLite database if missing.

    Ensures that disk loss (e.g. container recreation with persistent DB)
    does not leave the filesystem missing markdown wiki files or graph.json.
    """
    proj_root = Path(root).resolve() if root else storage.get_project_root()
    restored_notes = 0
    restored_graph = 0

    # 1. Restore wiki notes to markdown files if missing
    db_notes = db.list_notes(root=proj_root)
    wiki_dir = storage.get_wiki_dir(root=proj_root)
    for note in db_notes:
        note_file = wiki_dir / note.para / f"{note.id}.md"
        if not note_file.exists():
            try:
                storage.write_wiki_note(note, root=proj_root)
                restored_notes += 1
            except Exception as e:
                logger.warning(f"Failed to restore note {note.id} to markdown: {e}")

    # 2. Restore graph.json if missing on disk but exists in DB
    graph_path = proj_root / "data" / "graph.json"
    if not graph_path.exists():
        graph_records = db.load_graph_records(root=proj_root)
        if graph_records and graph_records.get("nodes"):
            try:
                graph_path.parent.mkdir(parents=True, exist_ok=True)
                with graph_path.open("w", encoding="utf-8") as f:
                    json.dump(graph_records, f, indent=2, ensure_ascii=False)
                restored_graph = len(graph_records["nodes"])
            except Exception as e:
                logger.warning(f"Failed to restore graph.json from DB: {e}")

    return {"restored_notes": restored_notes, "restored_graph": restored_graph}


def ensure_graph_populated(root: Path | str | None = None, user_id: str | None = None) -> bool:
    """Ensure that the knowledge graph is materialized and non-empty if notes exist.

    If graph.json is missing or has 0 nodes, but notes exist (in DB or wiki/),
    automatically rebuilds the graph from persisted records scoped to user_id.
    """
    proj_root = Path(root).resolve() if root else storage.get_project_root()
    import build_graph
    graph_path = build_graph.get_graph_path(root=proj_root, user_id=user_id)

    needs_rebuild = False

    if not graph_path.exists():
        needs_rebuild = True
    else:
        try:
            with graph_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            nodes = data.get("nodes", []) if isinstance(data, dict) else []
            if len(nodes) == 0:
                needs_rebuild = True
        except Exception:
            needs_rebuild = True

    if needs_rebuild:
        # Check if we have notes to build from for this user
        notes = storage.read_wiki_notes(root=proj_root, user_id=user_id)
        if not notes:
            notes = db.list_notes(user_id=user_id, root=proj_root)

        if notes:
            logger.info(f"Rebuilding knowledge graph from {len(notes)} persisted notes on startup...")
            try:
                graph_data, _ = build_graph.build_and_save_graph(root=proj_root, user_id=user_id)
                db.save_graph_records(
                    nodes=graph_data.get("nodes", []),
                    edges=graph_data.get("edges", []),
                    metadata=graph_data.get("metadata", {}),
                    user_id=user_id,
                    root=proj_root,
                )
                return True
            except Exception as e:
                logger.error(f"Failed to rebuild graph on startup: {e}")
                return False

    return False


def ensure_data_restored(root: Path | str | None = None, force: bool = False) -> dict[str, Any]:
    """Master startup initialization and restoration procedure.

    Idempotent: runs safely on every application boot, restart, or page load.
    1. Reconnects to and initializes the SQLite database.
    2. Performs bidirectional synchronization between filesystem and SQLite.
    3. Rebuilds the knowledge graph if notes exist but graph is empty or missing.
    4. Gathers and returns complete statistics of the restored knowledge brain.
    """
    global _STARTUP_COMPLETED
    if _STARTUP_COMPLETED and not force:
        # Return quick stats
        proj_root = Path(root).resolve() if root else storage.get_project_root()
        return {
            "status": "ready",
            "stats": db.get_stats_summary(root=proj_root),
            "rebuilt_graph": False,
        }

    proj_root = Path(root).resolve() if root else storage.get_project_root()
    messages: list[str] = []

    try:
        # 1. Reconnect and initialize database
        db.init_db(root=proj_root)
        messages.append("SQLite database initialized successfully.")

        # 2. Sync filesystem to DB
        fs_sync = sync_filesystem_to_db(root=proj_root)
        if fs_sync["synced_captures"] > 0 or fs_sync["synced_notes"] > 0:
            messages.append(
                f"Synchronized from filesystem: {fs_sync['synced_captures']} captures, "
                f"{fs_sync['synced_notes']} notes."
            )

        # 3. Sync DB to filesystem (restore any missing markdown files)
        db_sync = sync_db_to_filesystem(root=proj_root)
        if db_sync["restored_notes"] > 0 or db_sync["restored_graph"] > 0:
            messages.append(
                f"Restored to filesystem: {db_sync['restored_notes']} notes, "
                f"{db_sync['restored_graph']} graph nodes."
            )

        # 4. Auto-rebuild graph if notes exist but graph is missing/empty
        rebuilt = ensure_graph_populated(root=proj_root)
        if rebuilt:
            messages.append("Knowledge graph was automatically materialized from persisted records.")

        stats = db.get_stats_summary(root=proj_root)
        _STARTUP_COMPLETED = True

        return {
            "status": "ready",
            "db_path": str(db.get_db_path(root=proj_root)),
            "stats": stats,
            "rebuilt_graph": rebuilt,
            "messages": messages,
        }
    except Exception as e:
        logger.error(f"Startup data restoration encountered an error: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "stats": {},
            "rebuilt_graph": False,
            "messages": messages,
        }

