"""Persistent SQLite database manager for SecondSelf.

Provides durable, ACID-compliant storage for raw captures, wiki notes,
knowledge graph nodes, relationships, and system metadata.
Ensures data persists across application restarts, reboots, and service crashes.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sqlite3
from typing import Any, Generator, Sequence

try:
    from lib.models import CaptureMeta, GraphEdge, GraphNode, WikiNote
except ImportError:
    from .models import CaptureMeta, GraphEdge, GraphNode, WikiNote

logger = logging.getLogger(__name__)

__all__ = [
    "get_db_path",
    "get_connection",
    "init_db",
    "upsert_capture",
    "update_capture_status",
    "get_capture",
    "list_captures",
    "count_captures",
    "upsert_note",
    "delete_note",
    "get_note",
    "list_notes",
    "save_graph_records",
    "load_graph_records",
    "get_system_metadata",
    "set_system_metadata",
    "get_stats_summary",
]

_SCHEMA_VERSION = 1


def get_db_path(root: Path | str | None = None) -> Path:
    """Return the absolute path to the SecondSelf SQLite database file.

    Priority:
    1. Explicit root directory if supplied (e.g. for testing in temp dirs).
    2. SECONDSELF_DB_PATH environment variable if set.
    3. SECONDSELF_DATA_DIR / secondself.db if set.
    4. data/secondself.db under the resolved project root.
    """
    if root is not None:
        base = Path(root).resolve()
        return (base / "data" / "secondself.db").resolve()

    env_db = os.getenv("SECONDSELF_DB_PATH")
    if env_db and env_db.strip():
        return Path(env_db.strip()).resolve()

    env_data = os.getenv("SECONDSELF_DATA_DIR")
    if env_data and env_data.strip():
        return (Path(env_data.strip()).resolve() / "secondself.db")

    # Avoid circular import by resolving dynamically
    from . import storage
    base = storage.get_project_root()
    return (base / "data" / "secondself.db").resolve()


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Ensure database tables and indexes exist on connection."""
    row = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='captures';").fetchone()
    if row:
        return

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS captures (
            id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            type TEXT NOT NULL,
            source TEXT NOT NULL,
            original_filename TEXT,
            content_hash TEXT NOT NULL,
            content_path TEXT,
            raw_content TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            error_message TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_captures_hash ON captures(content_hash);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_captures_status ON captures(status);")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS notes (
            id TEXT PRIMARY KEY,
            raw_id TEXT,
            para TEXT NOT NULL,
            tags TEXT NOT NULL,
            summary TEXT NOT NULL,
            created TEXT NOT NULL,
            links TEXT NOT NULL,
            body TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_notes_para ON notes(para);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_notes_raw_id ON notes(raw_id);")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS graph_nodes (
            id TEXT PRIMARY KEY,
            label TEXT NOT NULL,
            para TEXT NOT NULL,
            tags TEXT NOT NULL,
            summary TEXT,
            content_preview TEXT,
            group_name TEXT
        );
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS graph_edges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            target TEXT NOT NULL,
            weight REAL NOT NULL DEFAULT 1.0,
            edge_type TEXT NOT NULL DEFAULT 'semantic',
            UNIQUE(source, target)
        );
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON graph_edges(source);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON graph_edges(target);")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS graph_metadata (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            generated_at TEXT,
            node_count INTEGER DEFAULT 0,
            edge_count INTEGER DEFAULT 0
        );
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS system_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
    )

    now_iso = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO system_metadata (key, value, updated_at)
        VALUES ('schema_version', ?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at;
        """,
        (str(_SCHEMA_VERSION), now_iso),
    )
    conn.commit()


def get_connection(root: Path | str | None = None) -> sqlite3.Connection:
    """Create and return a configured SQLite connection.

    Enables WAL mode, foreign keys, and busy timeout for safe concurrent operations.
    """
    db_path = get_db_path(root)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path), timeout=15.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA busy_timeout=10000;")
    _ensure_schema(conn)
    return conn


@contextmanager
def db_session(root: Path | str | None = None) -> Generator[sqlite3.Connection, None, None]:
    """Context manager for database transactions with automatic commit/rollback."""
    conn = get_connection(root)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(root: Path | str | None = None) -> None:
    """Initialize SQLite database tables and indexes idempotently.

    Never deletes or overwrites existing records or tables.
    """
    db_path = get_db_path(root)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with db_session(root) as conn:
        # 1. Raw captures table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS captures (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                type TEXT NOT NULL,
                source TEXT NOT NULL,
                original_filename TEXT,
                content_hash TEXT NOT NULL,
                content_path TEXT,
                raw_content TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                error_message TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_captures_hash ON captures(content_hash);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_captures_status ON captures(status);")

        # 2. Wiki notes table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS notes (
                id TEXT PRIMARY KEY,
                raw_id TEXT,
                para TEXT NOT NULL,
                tags TEXT NOT NULL,
                summary TEXT NOT NULL,
                created TEXT NOT NULL,
                links TEXT NOT NULL,
                body TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_notes_para ON notes(para);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_notes_raw_id ON notes(raw_id);")

        # 3. Graph nodes table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS graph_nodes (
                id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                para TEXT NOT NULL,
                tags TEXT NOT NULL,
                summary TEXT,
                content_preview TEXT,
                group_name TEXT
            );
            """
        )

        # 4. Graph edges table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS graph_edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                target TEXT NOT NULL,
                weight REAL NOT NULL DEFAULT 1.0,
                edge_type TEXT NOT NULL DEFAULT 'semantic',
                UNIQUE(source, target)
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON graph_edges(source);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON graph_edges(target);")

        # 5. Graph metadata table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS graph_metadata (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                generated_at TEXT,
                node_count INTEGER DEFAULT 0,
                edge_count INTEGER DEFAULT 0
            );
            """
        )

        # 6. System metadata table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS system_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )

        # Record schema version
        now_iso = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """
            INSERT INTO system_metadata (key, value, updated_at)
            VALUES ('schema_version', ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at;
            """,
            (str(_SCHEMA_VERSION), now_iso),
        )


def upsert_capture(
    meta: CaptureMeta,
    content: str | bytes | None = None,
    content_path: str | None = None,
    status: str = "pending",
    error_message: str | None = None,
    root: Path | str | None = None,
) -> None:
    """Insert or update a raw capture record in SQLite."""
    now_iso = datetime.now(timezone.utc).isoformat()
    raw_text = content.decode("utf-8", errors="replace") if isinstance(content, bytes) else content

    with db_session(root) as conn:
        conn.execute(
            """
            INSERT INTO captures (
                id, timestamp, type, source, original_filename,
                content_hash, content_path, raw_content, status,
                error_message, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                content_hash = excluded.content_hash,
                content_path = COALESCE(excluded.content_path, captures.content_path),
                raw_content = COALESCE(excluded.raw_content, captures.raw_content),
                status = excluded.status,
                error_message = excluded.error_message,
                updated_at = excluded.updated_at;
            """,
            (
                meta.id,
                meta.timestamp,
                meta.type,
                meta.source,
                meta.original_filename,
                meta.content_hash,
                str(content_path) if content_path else None,
                raw_text,
                status,
                error_message,
                now_iso,
                now_iso,
            ),
        )


def update_capture_status(
    capture_id: str,
    status: str,
    error_message: str | None = None,
    root: Path | str | None = None,
) -> None:
    """Update processing status ('pending', 'processing', 'completed', 'failed') of a capture."""
    now_iso = datetime.now(timezone.utc).isoformat()
    with db_session(root) as conn:
        conn.execute(
            """
            UPDATE captures
            SET status = ?, error_message = ?, updated_at = ?
            WHERE id = ?;
            """,
            (status, error_message, now_iso, capture_id),
        )


def get_capture(capture_id: str, root: Path | str | None = None) -> dict[str, Any] | None:
    """Retrieve capture record by ID."""
    with db_session(root) as conn:
        row = conn.execute("SELECT * FROM captures WHERE id = ?;", (capture_id,)).fetchone()
        return dict(row) if row else None


def list_captures(status: str | None = None, root: Path | str | None = None) -> list[dict[str, Any]]:
    """Return all captures, optionally filtered by status."""
    with db_session(root) as conn:
        if status:
            cursor = conn.execute(
                "SELECT * FROM captures WHERE status = ? ORDER BY timestamp ASC;",
                (status,),
            )
        else:
            cursor = conn.execute("SELECT * FROM captures ORDER BY timestamp ASC;")
        return [dict(r) for r in cursor.fetchall()]


def count_captures(root: Path | str | None = None) -> dict[str, int]:
    """Return counts of total, pending, completed, and failed captures."""
    with db_session(root) as conn:
        total = conn.execute("SELECT COUNT(*) FROM captures;").fetchone()[0]
        pending = conn.execute("SELECT COUNT(*) FROM captures WHERE status = 'pending';").fetchone()[0]
        completed = conn.execute("SELECT COUNT(*) FROM captures WHERE status = 'completed';").fetchone()[0]
        failed = conn.execute("SELECT COUNT(*) FROM captures WHERE status = 'failed';").fetchone()[0]
        return {
            "total": total,
            "pending": pending,
            "completed": completed,
            "failed": failed,
        }


def upsert_note(note: WikiNote, root: Path | str | None = None) -> None:
    """Insert or update a wiki note in SQLite."""
    now_iso = datetime.now(timezone.utc).isoformat()
    tags_json = json.dumps(list(note.tags) if note.tags else [])
    links_json = json.dumps(list(note.links) if note.links else [])

    with db_session(root) as conn:
        conn.execute(
            """
            INSERT INTO notes (
                id, raw_id, para, tags, summary, created, links, body, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                raw_id = excluded.raw_id,
                para = excluded.para,
                tags = excluded.tags,
                summary = excluded.summary,
                created = excluded.created,
                links = excluded.links,
                body = excluded.body,
                updated_at = excluded.updated_at;
            """,
            (
                note.id,
                note.raw_id,
                note.para,
                tags_json,
                note.summary,
                note.created,
                links_json,
                note.body or "",
                now_iso,
            ),
        )


def delete_note(note_id: str, root: Path | str | None = None) -> bool:
    """Delete a wiki note from SQLite by note ID."""
    with db_session(root) as conn:
        cursor = conn.execute("DELETE FROM notes WHERE id = ?;", (note_id,))
        # Also clean up graph nodes and edges involving this note
        conn.execute("DELETE FROM graph_nodes WHERE id = ?;", (note_id,))
        conn.execute("DELETE FROM graph_edges WHERE source = ? OR target = ?;", (note_id, note_id))
        return cursor.rowcount > 0


def get_note(note_id: str, root: Path | str | None = None) -> WikiNote | None:
    """Retrieve a single WikiNote by ID from SQLite."""
    with db_session(root) as conn:
        row = conn.execute("SELECT * FROM notes WHERE id = ?;", (note_id,)).fetchone()
        if not row:
            return None
        tags = json.loads(row["tags"]) if row["tags"] else []
        links = json.loads(row["links"]) if row["links"] else []
        return WikiNote(
            id=row["id"],
            raw_id=row["raw_id"] or "",
            para=row["para"],
            tags=tags,
            summary=row["summary"],
            created=row["created"],
            links=links,
            body=row["body"],
        )


def list_notes(root: Path | str | None = None) -> list[WikiNote]:
    """Return all WikiNotes saved in SQLite."""
    with db_session(root) as conn:
        cursor = conn.execute("SELECT * FROM notes ORDER BY para, summary ASC;")
        notes: list[WikiNote] = []
        for row in cursor.fetchall():
            tags = json.loads(row["tags"]) if row["tags"] else []
            links = json.loads(row["links"]) if row["links"] else []
            notes.append(
                WikiNote(
                    id=row["id"],
                    raw_id=row["raw_id"] or "",
                    para=row["para"],
                    tags=tags,
                    summary=row["summary"],
                    created=row["created"],
                    links=links,
                    body=row["body"],
                )
            )
        return notes


def save_graph_records(
    nodes: Sequence[GraphNode | dict[str, Any]],
    edges: Sequence[GraphEdge | dict[str, Any]],
    metadata: dict[str, Any] | None = None,
    root: Path | str | None = None,
) -> None:
    """Atomically persist graph nodes, edges, and metadata in SQLite."""
    meta = metadata or {}
    gen_at = meta.get("generated_at") or datetime.now(timezone.utc).isoformat()
    node_count = meta.get("node_count", len(nodes))
    edge_count = meta.get("edge_count", len(edges))

    with db_session(root) as conn:
        # Clear existing graph nodes and edges
        conn.execute("DELETE FROM graph_nodes;")
        conn.execute("DELETE FROM graph_edges;")

        # Insert nodes
        for n in nodes:
            if isinstance(n, GraphNode):
                nid = n.id
                lbl = n.label
                para = n.para
                tags_j = json.dumps(list(n.tags) if n.tags else [])
                sum_text = n.summary
                preview = n.content_preview
                grp = n.group
            else:
                nid = str(n.get("id", ""))
                lbl = str(n.get("label", nid))
                para = str(n.get("para", "Resources"))
                raw_tags = n.get("tags", [])
                tags_j = json.dumps(list(raw_tags) if isinstance(raw_tags, list) else [])
                sum_text = n.get("summary", "")
                preview = n.get("content_preview", "")
                grp = str(n.get("group", para))

            if not nid:
                continue

            conn.execute(
                """
                INSERT INTO graph_nodes (id, label, para, tags, summary, content_preview, group_name)
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (nid, lbl, para, tags_j, sum_text, preview, grp),
            )

        # Insert edges
        for e in edges:
            if isinstance(e, GraphEdge):
                src = e.source
                tgt = e.target
                weight = float(e.weight)
                e_type = e.type
            else:
                src = str(e.get("source", ""))
                tgt = str(e.get("target", ""))
                weight = float(e.get("weight", 1.0))
                e_type = str(e.get("type", "semantic"))

            if not src or not tgt:
                continue

            conn.execute(
                """
                INSERT OR IGNORE INTO graph_edges (source, target, weight, edge_type)
                VALUES (?, ?, ?, ?);
                """,
                (src, tgt, weight, e_type),
            )

        # Upsert graph metadata
        conn.execute(
            """
            INSERT INTO graph_metadata (id, generated_at, node_count, edge_count)
            VALUES (1, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                generated_at = excluded.generated_at,
                node_count = excluded.node_count,
                edge_count = excluded.edge_count;
            """,
            (gen_at, node_count, edge_count),
        )


def load_graph_records(root: Path | str | None = None) -> dict[str, Any] | None:
    """Retrieve full knowledge graph data (nodes, edges, metadata) from SQLite."""
    with db_session(root) as conn:
        meta_row = conn.execute("SELECT * FROM graph_metadata WHERE id = 1;").fetchone()
        node_rows = conn.execute("SELECT * FROM graph_nodes;").fetchall()
        edge_rows = conn.execute("SELECT * FROM graph_edges;").fetchall()

        if not node_rows and not meta_row:
            return None

        nodes: list[dict[str, Any]] = []
        for r in node_rows:
            tags = json.loads(r["tags"]) if r["tags"] else []
            nodes.append(
                {
                    "id": r["id"],
                    "label": r["label"],
                    "para": r["para"],
                    "tags": tags,
                    "summary": r["summary"],
                    "content_preview": r["content_preview"],
                    "group": r["group_name"],
                }
            )

        edges: list[dict[str, Any]] = []
        for r in edge_rows:
            edges.append(
                {
                    "source": r["source"],
                    "target": r["target"],
                    "weight": r["weight"],
                    "type": r["edge_type"],
                }
            )

        metadata = {
            "generated_at": meta_row["generated_at"] if meta_row else None,
            "node_count": meta_row["node_count"] if meta_row else len(nodes),
            "edge_count": meta_row["edge_count"] if meta_row else len(edges),
        }

        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": metadata,
        }


def get_system_metadata(key: str, default: Any = None, root: Path | str | None = None) -> Any:
    """Retrieve a value from system_metadata table."""
    with db_session(root) as conn:
        row = conn.execute("SELECT value FROM system_metadata WHERE key = ?;", (key,)).fetchone()
        return row["value"] if row else default


def set_system_metadata(key: str, value: Any, root: Path | str | None = None) -> None:
    """Set a value in system_metadata table."""
    now_iso = datetime.now(timezone.utc).isoformat()
    val_str = str(value)
    with db_session(root) as conn:
        conn.execute(
            """
            INSERT INTO system_metadata (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at;
            """,
            (key, val_str, now_iso),
        )


def get_stats_summary(root: Path | str | None = None) -> dict[str, Any]:
    """Retrieve comprehensive system statistics directly from SQLite."""
    with db_session(root) as conn:
        total_notes = conn.execute("SELECT COUNT(*) FROM notes;").fetchone()[0]
        cap_counts = count_captures(root)

        meta_row = conn.execute("SELECT * FROM graph_metadata WHERE id = 1;").fetchone()
        graph_nodes = meta_row["node_count"] if meta_row else conn.execute("SELECT COUNT(*) FROM graph_nodes;").fetchone()[0]
        graph_edges = meta_row["edge_count"] if meta_row else conn.execute("SELECT COUNT(*) FROM graph_edges;").fetchone()[0]
        last_build = meta_row["generated_at"] if meta_row else None

        para_rows = conn.execute("SELECT para, COUNT(*) as cnt FROM notes GROUP BY para;").fetchall()
        para_breakdown = {"Projects": 0, "Areas": 0, "Resources": 0, "Archives": 0}
        for r in para_rows:
            cat = r["para"]
            if cat in para_breakdown:
                para_breakdown[cat] = r["cnt"]
            else:
                para_breakdown[cat] = r["cnt"]

        return {
            "total_notes": total_notes,
            "raw_captures": cap_counts["total"],
            "pending_captures": cap_counts["pending"],
            "completed_captures": cap_counts["completed"],
            "failed_captures": cap_counts["failed"],
            "graph_nodes": graph_nodes,
            "graph_edges": graph_edges,
            "last_graph_build": last_build,
            "para_breakdown": para_breakdown,
        }
