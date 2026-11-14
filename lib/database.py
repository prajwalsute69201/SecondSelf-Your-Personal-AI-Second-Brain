"""Persistent multi-tenant database manager for SecondSelf.

Provides durable, ACID-compliant storage for users, sessions, captures, wiki notes,
knowledge graph nodes, relationships, and system metadata.
Ensures strict user data isolation: all queries and records are scoped to user_id.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sqlite3
import sys
from typing import Any, Generator, Sequence
import uuid

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from lib.constants import DEFAULT_ADMIN_EMAIL, DEFAULT_USER_ID, VALID_CATEGORIES
except ImportError:
    try:
        from .constants import DEFAULT_ADMIN_EMAIL, DEFAULT_USER_ID, VALID_CATEGORIES
    except ImportError:
        DEFAULT_USER_ID = "00000000-0000-0000-0000-000000000001"
        DEFAULT_ADMIN_EMAIL = "admin@secondself.ai"
        VALID_CATEGORIES = (
            "Projects", "Areas", "Resources", "Archives",
            "Work", "Payments", "Finance", "Personal", "Learning",
            "Health", "People", "Ideas", "Events", "Shopping",
        )

try:
    from lib.models import (
        CaptureMeta,
        GraphEdge,
        GraphNode,
        User,
        UserSession,
        WikiNote,
    )
except ImportError:
    from .models import (
        CaptureMeta,
        GraphEdge,
        GraphNode,
        User,
        UserSession,
        WikiNote,
    )

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_USER_ID",
    "DEFAULT_ADMIN_EMAIL",
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
    "get_note_by_id",
    "list_notes",
    "get_all_notes",
    "save_graph_records",
    "load_graph_records",
    "get_graph_nodes",
    "get_graph_edges",
    "get_system_metadata",
    "set_system_metadata",
    "get_stats_summary",
    "get_all_system_stats",
    "create_user",
    "get_user_by_id",
    "get_user_by_email",
    "list_all_users",
    "update_user",
    "create_session",
    "get_session",
    "delete_session",
    "create_password_reset",
    "get_password_reset",
    "mark_password_reset_used",
    "create_verification_token",
    "get_verification_token",
    "mark_email_verified",
]

_SCHEMA_VERSION = 2


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

    from . import storage
    base = storage.get_project_root()
    return (base / "data" / "secondself.db").resolve()


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Ensure database tables, multi-tenant columns, and indexes exist idempotently."""
    # 1. Users table
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT,
            role TEXT NOT NULL DEFAULT 'user',
            is_active INTEGER NOT NULL DEFAULT 1,
            email_verified INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            last_login_at TEXT
        );
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);")

    # 2. Sessions table
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            expires_at TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);")

    # 3. Password Resets table
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS password_resets (
            token TEXT PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            expires_at TEXT NOT NULL,
            used INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        );
        """
    )

    # 4. Email Verification Tokens table
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS verification_tokens (
            token TEXT PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            expires_at TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )

    # Helper to migrate v1 tables with single-column primary keys to compound (user_id, id)
    def _migrate_pk_if_needed(tbl_name: str, create_sql: str, req_pks: tuple[str, ...]) -> None:
        exists = conn.execute(
            "SELECT count(*) FROM sqlite_master WHERE type='table' AND name=?;",
            (tbl_name,),
        ).fetchone()[0]
        if not exists:
            conn.execute(create_sql)
            return

        t_info = conn.execute(f"PRAGMA table_info({tbl_name});").fetchall()
        pks = tuple(c[1] for c in sorted(t_info, key=lambda item: item[5]) if c[5] > 0)
        if set(pks) != set(req_pks):
            conn.execute(f"ALTER TABLE {tbl_name} RENAME TO {tbl_name}_old_mig;")
            conn.execute(create_sql)
            old_cols = [c[1] for c in t_info]
            new_cols = [c[1] for c in conn.execute(f"PRAGMA table_info({tbl_name});").fetchall()]
            common = [c for c in old_cols if c in new_cols]
            if "user_id" not in old_cols:
                ins = ", ".join(common) + ", user_id"
                sel = ", ".join(common) + f", '{DEFAULT_USER_ID}'"
            else:
                ins = ", ".join(common)
                sel = ", ".join(common)
            conn.execute(f"INSERT OR IGNORE INTO {tbl_name} ({ins}) SELECT {sel} FROM {tbl_name}_old_mig;")
            conn.execute(f"DROP TABLE {tbl_name}_old_mig;")

    # 5. Raw captures table (multi-tenant)
    captures_create_sql = """
        CREATE TABLE IF NOT EXISTS captures (
            id TEXT NOT NULL,
            user_id TEXT NOT NULL DEFAULT '00000000-0000-0000-0000-000000000001',
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
            updated_at TEXT NOT NULL,
            PRIMARY KEY (user_id, id)
        );
    """
    _migrate_pk_if_needed("captures", captures_create_sql, ("user_id", "id"))
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_captures_user_id_id ON captures(user_id, id);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_captures_user_id ON captures(user_id);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_captures_user_hash ON captures(user_id, content_hash);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_captures_user_status ON captures(user_id, status);")

    # 6. Wiki notes table (multi-tenant)
    notes_create_sql = """
        CREATE TABLE IF NOT EXISTS notes (
            id TEXT NOT NULL,
            user_id TEXT NOT NULL DEFAULT '00000000-0000-0000-0000-000000000001',
            raw_id TEXT,
            para TEXT NOT NULL,
            tags TEXT NOT NULL,
            summary TEXT NOT NULL,
            created TEXT NOT NULL,
            links TEXT NOT NULL,
            body TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (user_id, id)
        );
    """
    _migrate_pk_if_needed("notes", notes_create_sql, ("user_id", "id"))
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_notes_user_id_id ON notes(user_id, id);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_notes_user_id ON notes(user_id);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_notes_user_para ON notes(user_id, para);")

    # 7. Graph nodes table (multi-tenant)
    graph_nodes_create_sql = """
        CREATE TABLE IF NOT EXISTS graph_nodes (
            id TEXT NOT NULL,
            user_id TEXT NOT NULL DEFAULT '00000000-0000-0000-0000-000000000001',
            label TEXT NOT NULL,
            para TEXT NOT NULL,
            tags TEXT NOT NULL,
            summary TEXT,
            content_preview TEXT,
            group_name TEXT,
            PRIMARY KEY (user_id, id)
        );
    """
    _migrate_pk_if_needed("graph_nodes", graph_nodes_create_sql, ("user_id", "id"))
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_graph_nodes_user_id_id ON graph_nodes(user_id, id);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_graph_nodes_user ON graph_nodes(user_id);")

    # 8. Graph edges table (multi-tenant)
    graph_edges_create_sql = """
        CREATE TABLE IF NOT EXISTS graph_edges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL DEFAULT '00000000-0000-0000-0000-000000000001',
            source TEXT NOT NULL,
            target TEXT NOT NULL,
            weight REAL NOT NULL DEFAULT 1.0,
            edge_type TEXT NOT NULL DEFAULT 'semantic',
            UNIQUE(user_id, source, target)
        );
    """
    ge_info = conn.execute("PRAGMA table_info(graph_edges);").fetchall()
    ge_cols = [c[1] for c in ge_info]
    if ge_info and "user_id" not in ge_cols:
        conn.execute("ALTER TABLE graph_edges RENAME TO graph_edges_old_mig;")
        conn.execute(graph_edges_create_sql)
        conn.execute(
            f"INSERT OR IGNORE INTO graph_edges (source, target, weight, edge_type, user_id) "
            f"SELECT source, target, weight, edge_type, '{DEFAULT_USER_ID}' FROM graph_edges_old_mig;"
        )
        conn.execute("DROP TABLE graph_edges_old_mig;")
    else:
        conn.execute(graph_edges_create_sql)

    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_graph_edges_user_id_source_target ON graph_edges(user_id, source, target);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_graph_edges_user ON graph_edges(user_id);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON graph_edges(user_id, source);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON graph_edges(user_id, target);")

    # 9. Graph metadata table (multi-tenant)
    graph_meta_create_sql = """
        CREATE TABLE IF NOT EXISTS graph_metadata (
            user_id TEXT PRIMARY KEY,
            generated_at TEXT,
            node_count INTEGER DEFAULT 0,
            edge_count INTEGER DEFAULT 0
        );
    """
    gm_info = conn.execute("PRAGMA table_info(graph_metadata);").fetchall()
    gm_cols = [c[1] for c in gm_info]
    if gm_info and "user_id" not in gm_cols:
        conn.execute("ALTER TABLE graph_metadata RENAME TO graph_metadata_old_mig;")
        conn.execute(graph_meta_create_sql)
        conn.execute(
            f"INSERT OR IGNORE INTO graph_metadata (user_id, generated_at, node_count, edge_count) "
            f"SELECT '{DEFAULT_USER_ID}', generated_at, node_count, edge_count FROM graph_metadata_old_mig;"
        )
        conn.execute("DROP TABLE graph_metadata_old_mig;")
    else:
        conn.execute(graph_meta_create_sql)

    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_graph_metadata_user_id ON graph_metadata(user_id);")

    # 10. System metadata table
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS system_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
    )

    # 11. Ensure default initial admin user exists
    now_iso = datetime.now(timezone.utc).isoformat()
    admin_row = conn.execute("SELECT id FROM users WHERE id = ?;", (DEFAULT_USER_ID,)).fetchone()
    if not admin_row:
        # Default admin account: admin@secondself.ai / AdminPassword123!
        # PBKDF2 salt and hash precomputed for instant startup
        default_hash = "pbkdf2_sha256$260000$44454641554c5453414c543230323621$4b34479cbcf7696495be0d421255e4bf5b72e505299ad072782b5f7b0f69a6fc"
        conn.execute(
            """
            INSERT OR IGNORE INTO users (
                id, email, password_hash, full_name, role, is_active, email_verified, created_at, updated_at
            ) VALUES (?, ?, ?, ?, 'admin', 1, 1, ?, ?);
            """,
            (
                DEFAULT_USER_ID,
                DEFAULT_ADMIN_EMAIL,
                default_hash,
                "SecondSelf Administrator",
                now_iso,
                now_iso,
            ),
        )

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
    """Create and return a configured SQLite connection."""
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
    """Initialize database and schema."""
    with db_session(root) as conn:
        _ensure_schema(conn)


# =============================================================================
# User Account & Authentication Database Operations
# =============================================================================

def create_user(
    email: str,
    password_hash: str,
    full_name: str | None = None,
    role: str = "user",
    is_active: bool = True,
    email_verified: bool = False,
    user_id: str | None = None,
    root: Path | str | None = None,
) -> User:
    """Insert a new user record into the database."""
    now_iso = datetime.now(timezone.utc).isoformat()
    clean_email = email.strip().lower()
    uid = user_id or str(uuid.uuid4())

    with db_session(root) as conn:
        conn.execute(
            """
            INSERT INTO users (
                id, email, password_hash, full_name, role, is_active, email_verified, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                uid,
                clean_email,
                password_hash,
                full_name.strip() if full_name else None,
                role,
                1 if is_active else 0,
                1 if email_verified else 0,
                now_iso,
                now_iso,
            ),
        )

    return User(
        id=uid,
        email=clean_email,
        full_name=full_name,
        role=role,
        is_active=is_active,
        email_verified=email_verified,
        created_at=now_iso,
        last_login_at=None,
    )


def get_user_by_id(user_id: str, root: Path | str | None = None) -> User | None:
    """Retrieve user entity by unique user ID."""
    with db_session(root) as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?;", (user_id,)).fetchone()
        if not row:
            return None
        return User(
            id=row["id"],
            email=row["email"],
            full_name=row["full_name"],
            role=row["role"],
            is_active=bool(row["is_active"]),
            email_verified=bool(row["email_verified"]),
            created_at=row["created_at"],
            last_login_at=row["last_login_at"],
        )


def get_user_by_email(email: str, root: Path | str | None = None) -> tuple[User, str] | None:
    """Retrieve user entity and password_hash by email address."""
    clean_email = email.strip().lower()
    with db_session(root) as conn:
        row = conn.execute("SELECT * FROM users WHERE LOWER(email) = ?;", (clean_email,)).fetchone()
        if not row:
            return None
        user = User(
            id=row["id"],
            email=row["email"],
            full_name=row["full_name"],
            role=row["role"],
            is_active=bool(row["is_active"]),
            email_verified=bool(row["email_verified"]),
            created_at=row["created_at"],
            last_login_at=row["last_login_at"],
        )
        return user, row["password_hash"]


def list_all_users(root: Path | str | None = None) -> list[dict[str, Any]]:
    """Retrieve all users with aggregated statistics for Admin portal."""
    with db_session(root) as conn:
        rows = conn.execute(
            """
            SELECT 
                u.id, u.email, u.full_name, u.role, u.is_active, u.email_verified,
                u.created_at, u.last_login_at,
                COUNT(DISTINCT n.id) as note_count,
                COUNT(DISTINCT c.id) as capture_count
            FROM users u
            LEFT JOIN notes n ON n.user_id = u.id
            LEFT JOIN captures c ON c.user_id = u.id
            GROUP BY u.id
            ORDER BY u.created_at DESC;
            """
        ).fetchall()
        return [dict(r) for r in rows]


def update_user(user_id: str, updates: dict[str, Any], root: Path | str | None = None) -> bool:
    """Update user fields (e.g. role, is_active, password_hash, email_verified, last_login_at)."""
    now_iso = datetime.now(timezone.utc).isoformat()
    allowed_fields = {
        "full_name",
        "role",
        "is_active",
        "email_verified",
        "password_hash",
        "last_login_at",
    }
    set_clauses = []
    params = []

    for k, v in updates.items():
        if k in allowed_fields:
            set_clauses.append(f"{k} = ?")
            if isinstance(v, bool):
                params.append(1 if v else 0)
            else:
                params.append(v)

    if not set_clauses:
        return False

    set_clauses.append("updated_at = ?")
    params.append(now_iso)
    params.append(user_id)

    with db_session(root) as conn:
        cursor = conn.execute(
            f"UPDATE users SET {', '.join(set_clauses)} WHERE id = ?;",
            tuple(params),
        )
        return cursor.rowcount > 0


def create_session(token: str, user_id: str, expires_at: str, root: Path | str | None = None) -> None:
    """Store an active session token."""
    now_iso = datetime.now(timezone.utc).isoformat()
    with db_session(root) as conn:
        conn.execute(
            "INSERT INTO sessions (token, user_id, expires_at, created_at) VALUES (?, ?, ?, ?);",
            (token, user_id, expires_at, now_iso),
        )


def get_session(token: str, root: Path | str | None = None) -> User | None:
    """Verify session token and return authenticated User if valid and active."""
    now_iso = datetime.now(timezone.utc).isoformat()
    with db_session(root) as conn:
        row = conn.execute(
            """
            SELECT u.* FROM sessions s
            JOIN users u ON s.user_id = u.id
            WHERE s.token = ? AND s.expires_at > ? AND u.is_active = 1;
            """,
            (token, now_iso),
        ).fetchone()
        if not row:
            return None
        return User(
            id=row["id"],
            email=row["email"],
            full_name=row["full_name"],
            role=row["role"],
            is_active=bool(row["is_active"]),
            email_verified=bool(row["email_verified"]),
            created_at=row["created_at"],
            last_login_at=row["last_login_at"],
        )


def delete_session(token: str, root: Path | str | None = None) -> None:
    """Invalidate and remove session token."""
    with db_session(root) as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?;", (token,))


def create_password_reset(token: str, user_id: str, expires_at: str, root: Path | str | None = None) -> None:
    """Record a password reset token."""
    now_iso = datetime.now(timezone.utc).isoformat()
    with db_session(root) as conn:
        conn.execute(
            "INSERT INTO password_resets (token, user_id, expires_at, used, created_at) VALUES (?, ?, ?, 0, ?);",
            (token, user_id, expires_at, now_iso),
        )


def get_password_reset(token: str, root: Path | str | None = None) -> dict[str, Any] | None:
    """Retrieve valid, unused password reset record."""
    now_iso = datetime.now(timezone.utc).isoformat()
    with db_session(root) as conn:
        row = conn.execute(
            "SELECT * FROM password_resets WHERE token = ? AND used = 0 AND expires_at > ?;",
            (token, now_iso),
        ).fetchone()
        return dict(row) if row else None


def mark_password_reset_used(token: str, root: Path | str | None = None) -> None:
    """Mark a password reset token as used."""
    with db_session(root) as conn:
        conn.execute("UPDATE password_resets SET used = 1 WHERE token = ?;", (token,))


def create_verification_token(token: str, user_id: str, expires_at: str, root: Path | str | None = None) -> None:
    """Record an email verification token."""
    now_iso = datetime.now(timezone.utc).isoformat()
    with db_session(root) as conn:
        conn.execute(
            "INSERT INTO verification_tokens (token, user_id, expires_at, created_at) VALUES (?, ?, ?, ?);",
            (token, user_id, expires_at, now_iso),
        )


def get_verification_token(token: str, root: Path | str | None = None) -> dict[str, Any] | None:
    """Retrieve valid email verification record."""
    now_iso = datetime.now(timezone.utc).isoformat()
    with db_session(root) as conn:
        row = conn.execute(
            "SELECT * FROM verification_tokens WHERE token = ? AND expires_at > ?;",
            (token, now_iso),
        ).fetchone()
        return dict(row) if row else None


def mark_email_verified(user_id: str, root: Path | str | None = None) -> None:
    """Mark user account as email verified."""
    update_user(user_id, {"email_verified": True}, root=root)


# =============================================================================
# Scoped Multi-Tenant Capture Operations
# =============================================================================

def upsert_capture(
    meta: CaptureMeta,
    content: str | bytes | None = None,
    content_path: str | None = None,
    status: str = "pending",
    error_message: str | None = None,
    user_id: str | None = None,
    root: Path | str | None = None,
) -> None:
    """Insert or update a raw capture record in SQLite scoped to user_id."""
    effective_user_id = user_id or meta.user_id or DEFAULT_USER_ID
    now_iso = datetime.now(timezone.utc).isoformat()
    raw_text = content.decode("utf-8", errors="replace") if isinstance(content, bytes) else content

    with db_session(root) as conn:
        conn.execute(
            """
            INSERT INTO captures (
                id, user_id, timestamp, type, source, original_filename,
                content_hash, content_path, raw_content, status,
                error_message, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, id) DO UPDATE SET
                content_hash = excluded.content_hash,
                content_path = COALESCE(excluded.content_path, captures.content_path),
                raw_content = COALESCE(excluded.raw_content, captures.raw_content),
                status = excluded.status,
                error_message = excluded.error_message,
                updated_at = excluded.updated_at;
            """,
            (
                meta.id,
                effective_user_id,
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
    user_id: str | None = None,
    root: Path | str | None = None,
) -> None:
    """Update processing status of a capture scoped to user_id."""
    effective_user_id = user_id or DEFAULT_USER_ID
    now_iso = datetime.now(timezone.utc).isoformat()
    with db_session(root) as conn:
        conn.execute(
            """
            UPDATE captures
            SET status = ?, error_message = ?, updated_at = ?
            WHERE id = ? AND user_id = ?;
            """,
            (status, error_message, now_iso, capture_id, effective_user_id),
        )


def get_capture(
    capture_id: str,
    user_id: str | None = None,
    root: Path | str | None = None,
) -> dict[str, Any] | None:
    """Retrieve capture record by ID scoped to user_id."""
    effective_user_id = user_id or DEFAULT_USER_ID
    with db_session(root) as conn:
        row = conn.execute(
            "SELECT * FROM captures WHERE id = ? AND user_id = ?;",
            (capture_id, effective_user_id),
        ).fetchone()
        return dict(row) if row else None


def list_captures(
    status: str | None = None,
    user_id: str | None = None,
    root: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Return all captures for user_id, optionally filtered by status."""
    effective_user_id = user_id or DEFAULT_USER_ID
    with db_session(root) as conn:
        if status:
            cursor = conn.execute(
                "SELECT * FROM captures WHERE user_id = ? AND status = ? ORDER BY timestamp ASC;",
                (effective_user_id, status),
            )
        else:
            cursor = conn.execute(
                "SELECT * FROM captures WHERE user_id = ? ORDER BY timestamp ASC;",
                (effective_user_id,),
            )
        return [dict(r) for r in cursor.fetchall()]


def count_captures(user_id: str | None = None, root: Path | str | None = None) -> dict[str, int]:
    """Return capture counts scoped to user_id."""
    effective_user_id = user_id or DEFAULT_USER_ID
    with db_session(root) as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM captures WHERE user_id = ?;",
            (effective_user_id,),
        ).fetchone()[0]
        pending = conn.execute(
            "SELECT COUNT(*) FROM captures WHERE user_id = ? AND status = 'pending';",
            (effective_user_id,),
        ).fetchone()[0]
        completed = conn.execute(
            "SELECT COUNT(*) FROM captures WHERE user_id = ? AND status = 'completed';",
            (effective_user_id,),
        ).fetchone()[0]
        failed = conn.execute(
            "SELECT COUNT(*) FROM captures WHERE user_id = ? AND status = 'failed';",
            (effective_user_id,),
        ).fetchone()[0]
        return {
            "total": total,
            "pending": pending,
            "completed": completed,
            "failed": failed,
        }


# =============================================================================
# Scoped Multi-Tenant Wiki Note Operations
# =============================================================================

def upsert_note(
    note: WikiNote,
    user_id: str | None = None,
    root: Path | str | None = None,
) -> None:
    """Insert or update a wiki note in SQLite scoped to user_id."""
    effective_user_id = user_id or note.user_id or DEFAULT_USER_ID
    now_iso = datetime.now(timezone.utc).isoformat()
    tags_json = json.dumps(list(note.tags) if note.tags else [])
    links_json = json.dumps(list(note.links) if note.links else [])

    with db_session(root) as conn:
        conn.execute(
            """
            INSERT INTO notes (
                id, user_id, raw_id, para, tags, summary, created, links, body, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, id) DO UPDATE SET
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
                effective_user_id,
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


def delete_note(
    note_id: str,
    user_id: str | None = None,
    root: Path | str | None = None,
) -> bool:
    """Delete a wiki note from SQLite by note ID scoped to user_id."""
    effective_user_id = user_id or DEFAULT_USER_ID
    with db_session(root) as conn:
        cursor = conn.execute(
            "DELETE FROM notes WHERE id = ? AND user_id = ?;",
            (note_id, effective_user_id),
        )
        conn.execute(
            "DELETE FROM graph_nodes WHERE id = ? AND user_id = ?;",
            (note_id, effective_user_id),
        )
        conn.execute(
            "DELETE FROM graph_edges WHERE user_id = ? AND (source = ? OR target = ?);",
            (effective_user_id, note_id, note_id),
        )
        return cursor.rowcount > 0


def get_note(
    note_id: str,
    user_id: str | None = None,
    root: Path | str | None = None,
) -> WikiNote | None:
    """Retrieve a single WikiNote by ID scoped to user_id."""
    effective_user_id = user_id or DEFAULT_USER_ID
    with db_session(root) as conn:
        row = conn.execute(
            "SELECT * FROM notes WHERE id = ? AND user_id = ?;",
            (note_id, effective_user_id),
        ).fetchone()
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
            user_id=row["user_id"],
        )


get_note_by_id = get_note


def get_graph_nodes(user_id: str | None = None, root: Path | str | None = None) -> list[dict[str, Any]]:
    """Retrieve all graph nodes scoped to user_id."""
    effective_user_id = user_id or DEFAULT_USER_ID
    with db_session(root) as conn:
        rows = conn.execute(
            "SELECT * FROM graph_nodes WHERE user_id = ? ORDER BY id ASC;",
            (effective_user_id,),
        ).fetchall()
        result = []
        for r in rows:
            tags = json.loads(r["tags"]) if r["tags"] else []
            result.append(
                {
                    "id": r["id"],
                    "label": r["label"],
                    "para": r["para"],
                    "tags": tags,
                    "summary": r["summary"],
                    "content_preview": r["content_preview"],
                    "group": r["group_name"],
                    "user_id": r["user_id"],
                }
            )
        return result


def get_graph_edges(user_id: str | None = None, root: Path | str | None = None) -> list[dict[str, Any]]:
    """Retrieve all graph edges scoped to user_id."""
    effective_user_id = user_id or DEFAULT_USER_ID
    with db_session(root) as conn:
        rows = conn.execute(
            "SELECT * FROM graph_edges WHERE user_id = ? ORDER BY source, target ASC;",
            (effective_user_id,),
        ).fetchall()
        return [
            {
                "source": r["source"],
                "target": r["target"],
                "weight": r["weight"],
                "type": r["edge_type"],
                "user_id": r["user_id"],
            }
            for r in rows
        ]


def list_notes(
    user_id: str | None = None,
    root: Path | str | None = None,
) -> list[WikiNote]:
    """Return all WikiNotes saved in SQLite scoped to user_id."""
    effective_user_id = user_id or DEFAULT_USER_ID
    with db_session(root) as conn:
        cursor = conn.execute(
            "SELECT * FROM notes WHERE user_id = ? ORDER BY para, summary ASC;",
            (effective_user_id,),
        )
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
                    user_id=row["user_id"],
                )
            )
        return notes


get_all_notes = list_notes


# =============================================================================
# Scoped Multi-Tenant Knowledge Graph Operations
# =============================================================================

def save_graph_records(
    nodes: Sequence[GraphNode | dict[str, Any]],
    edges: Sequence[GraphEdge | dict[str, Any]],
    metadata: dict[str, Any] | None = None,
    user_id: str | None = None,
    root: Path | str | None = None,
) -> None:
    """Atomically persist graph nodes, edges, and metadata scoped to user_id."""
    effective_user_id = user_id or DEFAULT_USER_ID
    meta = metadata or {}
    gen_at = meta.get("generated_at") or datetime.now(timezone.utc).isoformat()
    node_count = meta.get("node_count", len(nodes))
    edge_count = meta.get("edge_count", len(edges))

    with db_session(root) as conn:
        # Clear existing graph nodes and edges for this specific user
        conn.execute("DELETE FROM graph_nodes WHERE user_id = ?;", (effective_user_id,))
        conn.execute("DELETE FROM graph_edges WHERE user_id = ?;", (effective_user_id,))

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
                INSERT INTO graph_nodes (id, user_id, label, para, tags, summary, content_preview, group_name)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (nid, effective_user_id, lbl, para, tags_j, sum_text, preview, grp),
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
                INSERT OR IGNORE INTO graph_edges (user_id, source, target, weight, edge_type)
                VALUES (?, ?, ?, ?, ?);
                """,
                (effective_user_id, src, tgt, weight, e_type),
            )

        # Upsert graph metadata
        conn.execute(
            """
            INSERT INTO graph_metadata (user_id, generated_at, node_count, edge_count)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                generated_at = excluded.generated_at,
                node_count = excluded.node_count,
                edge_count = excluded.edge_count;
            """,
            (effective_user_id, gen_at, node_count, edge_count),
        )


def load_graph_records(
    user_id: str | None = None,
    root: Path | str | None = None,
) -> dict[str, Any] | None:
    """Retrieve knowledge graph data scoped to user_id."""
    effective_user_id = user_id or DEFAULT_USER_ID
    with db_session(root) as conn:
        meta_row = conn.execute(
            "SELECT * FROM graph_metadata WHERE user_id = ?;",
            (effective_user_id,),
        ).fetchone()
        node_rows = conn.execute(
            "SELECT * FROM graph_nodes WHERE user_id = ?;",
            (effective_user_id,),
        ).fetchall()
        edge_rows = conn.execute(
            "SELECT * FROM graph_edges WHERE user_id = ?;",
            (effective_user_id,),
        ).fetchall()

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


# =============================================================================
# System Metadata & Statistics Operations
# =============================================================================

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


def get_stats_summary(user_id: str | None = None, root: Path | str | None = None) -> dict[str, Any]:
    """Retrieve comprehensive knowledge statistics scoped to a single user."""
    effective_user_id = user_id or DEFAULT_USER_ID
    with db_session(root) as conn:
        total_notes = conn.execute(
            "SELECT COUNT(*) FROM notes WHERE user_id = ?;",
            (effective_user_id,),
        ).fetchone()[0]
        cap_counts = count_captures(effective_user_id, root)

        meta_row = conn.execute(
            "SELECT * FROM graph_metadata WHERE user_id = ?;",
            (effective_user_id,),
        ).fetchone()
        graph_nodes = (
            meta_row["node_count"]
            if meta_row
            else conn.execute(
                "SELECT COUNT(*) FROM graph_nodes WHERE user_id = ?;",
                (effective_user_id,),
            ).fetchone()[0]
        )
        graph_edges = (
            meta_row["edge_count"]
            if meta_row
            else conn.execute(
                "SELECT COUNT(*) FROM graph_edges WHERE user_id = ?;",
                (effective_user_id,),
            ).fetchone()[0]
        )
        last_build = meta_row["generated_at"] if meta_row else None

        para_rows = conn.execute(
            "SELECT para, COUNT(*) as cnt FROM notes WHERE user_id = ? GROUP BY para;",
            (effective_user_id,),
        ).fetchall()
        para_breakdown = {cat: 0 for cat in VALID_CATEGORIES}
        for r in para_rows:
            cat = r["para"]
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


def get_all_system_stats(root: Path | str | None = None) -> dict[str, Any]:
    """Platform-level aggregates for the Administrator portal."""
    with db_session(root) as conn:
        total_users = conn.execute("SELECT COUNT(*) FROM users;").fetchone()[0]
        active_users = conn.execute("SELECT COUNT(*) FROM users WHERE is_active = 1;").fetchone()[0]
        verified_users = conn.execute("SELECT COUNT(*) FROM users WHERE email_verified = 1;").fetchone()[0]
        total_notes = conn.execute("SELECT COUNT(*) FROM notes;").fetchone()[0]
        total_captures = conn.execute("SELECT COUNT(*) FROM captures;").fetchone()[0]
        total_edges = conn.execute("SELECT COUNT(*) FROM graph_edges;").fetchone()[0]

        return {
            "total_users": total_users,
            "active_users": active_users,
            "verified_users": verified_users,
            "total_notes": total_notes,
            "total_captures": total_captures,
            "total_edges": total_edges,
        }
