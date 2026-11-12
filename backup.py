"""Database and Knowledge Base Backup & Restoration Utility for SecondSelf.

Provides safe, atomic, idempotent backup and recovery for:
- SQLite database (secondself.db)
- Processed wiki notes (wiki/)
- Raw captures store (raw/)
- Knowledge graph and index files (data/)

Usage:
    python backup.py create                   # Create timestamped backup archive
    python backup.py create --dest /path/to   # Create backup in custom directory
    python backup.py restore <backup_path>    # Restore knowledge base from backup
    python backup.py list                     # List available backups
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import tarfile
from typing import Any

# Ensure repository root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lib import database as db
from lib import startup
from lib import storage

logger = logging.getLogger(__name__)

DEFAULT_BACKUP_DIR = _PROJECT_ROOT / "backups"


def create_backup(
    dest_dir: Path | str | None = None,
    root: Path | str | None = None,
    as_archive: bool = True,
) -> Path:
    """Create a complete, atomic backup of the SecondSelf persistent data.

    Args:
        dest_dir: Directory to save backup into (defaults to backups/).
        root: Optional project root path override.
        as_archive: If True, packages backup as .tar.gz; else as directory.

    Returns:
        Path: Path to created backup file or directory.
    """
    proj_root = Path(root).resolve() if root else storage.get_project_root()
    target_dir = Path(dest_dir).resolve() if dest_dir else (proj_root / "backups")
    target_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_name = f"secondself_backup_{timestamp_str}"
    temp_stage = target_dir / f".tmp_{backup_name}"
    temp_stage.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Hot backup of SQLite database using SQLite Online Backup API
        db_path = db.get_db_path(root=proj_root)
        if db_path.exists():
            stage_db = temp_stage / "secondself.db"
            src_conn = db.get_connection(root=proj_root)
            dst_conn = sqlite3.connect(str(stage_db))
            with dst_conn:
                src_conn.backup(dst_conn)
            dst_conn.close()
            src_conn.close()

        # 2. Copy wiki/ directory
        wiki_dir = storage.get_wiki_dir(root=proj_root)
        if wiki_dir.exists():
            shutil.copytree(wiki_dir, temp_stage / "wiki", dirs_exist_ok=True)

        # 3. Copy raw/ directory
        raw_dir = storage.get_raw_dir(root=proj_root)
        if raw_dir.exists():
            shutil.copytree(raw_dir, temp_stage / "raw", dirs_exist_ok=True)

        # 4. Copy data/ artifacts
        data_dir = storage.get_data_dir(root=proj_root)
        stage_data = temp_stage / "data"
        stage_data.mkdir(parents=True, exist_ok=True)
        for artifact in ("graph.json", "index.json", "embeddings.pkl"):
            src_art = data_dir / artifact
            if src_art.exists():
                shutil.copy2(src_art, stage_data / artifact)

        # 5. Write manifest metadata
        manifest = {
            "backup_name": backup_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_root": str(proj_root),
            "stats": db.get_stats_summary(root=proj_root),
        }
        with (temp_stage / "manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        if as_archive:
            archive_path = target_dir / f"{backup_name}.tar.gz"
            with tarfile.open(archive_path, "w:gz") as tar:
                tar.add(temp_stage, arcname=backup_name)
            shutil.rmtree(temp_stage, ignore_errors=True)
            return archive_path
        else:
            final_dir = target_dir / backup_name
            temp_stage.rename(final_dir)
            return final_dir

    except Exception:
        shutil.rmtree(temp_stage, ignore_errors=True)
        raise


def restore_backup(
    backup_path: Path | str,
    root: Path | str | None = None,
) -> dict[str, Any]:
    """Restore persistent SecondSelf data from a backup directory or .tar.gz archive.

    Args:
        backup_path: Path to backup file (.tar.gz) or directory.
        root: Target project root directory to restore into.

    Returns:
        dict: Restoration summary and verified stats.
    """
    src = Path(backup_path).resolve()
    if not src.exists():
        raise FileNotFoundError(f"Backup path does not exist: {src}")

    proj_root = Path(root).resolve() if root else storage.get_project_root()
    temp_extract: Path | None = None

    if src.is_file() and src.name.endswith(".tar.gz"):
        temp_extract = proj_root / ".tmp_restore_extract"
        temp_extract.mkdir(parents=True, exist_ok=True)
        with tarfile.open(src, "r:gz") as tar:
            tar.extractall(path=temp_extract)
        # Find inner directory
        entries = [e for e in temp_extract.iterdir() if e.is_dir()]
        restore_source = entries[0] if entries else temp_extract
    else:
        restore_source = src

    try:
        # 1. Restore SQLite database
        src_db = restore_source / "secondself.db"
        if src_db.exists():
            target_db = db.get_db_path(root=proj_root)
            target_db.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_db, target_db)

        # 2. Restore wiki/ notes
        src_wiki = restore_source / "wiki"
        if src_wiki.exists():
            target_wiki = storage.get_wiki_dir(root=proj_root)
            target_wiki.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src_wiki, target_wiki, dirs_exist_ok=True)

        # 3. Restore raw/ captures
        src_raw = restore_source / "raw"
        if src_raw.exists():
            target_raw = storage.get_raw_dir(root=proj_root)
            target_raw.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src_raw, target_raw, dirs_exist_ok=True)

        # 4. Restore data/ files
        src_data = restore_source / "data"
        if src_data.exists():
            target_data = storage.get_data_dir(root=proj_root)
            target_data.mkdir(parents=True, exist_ok=True)
            for item in src_data.iterdir():
                if item.is_file():
                    shutil.copy2(item, target_data / item.name)

        # 5. Trigger startup recovery to verify integrity
        recovery_result = startup.ensure_data_restored(root=proj_root, force=True)

        return {
            "status": "restored",
            "source_backup": str(src),
            "restored_root": str(proj_root),
            "recovery": recovery_result,
        }
    finally:
        if temp_extract and temp_extract.exists():
            shutil.rmtree(temp_extract, ignore_errors=True)


def list_backups(dest_dir: Path | str | None = None) -> list[dict[str, Any]]:
    """List available backups in the backup directory."""
    target_dir = Path(dest_dir).resolve() if dest_dir else (storage.get_project_root() / "backups")
    if not target_dir.exists():
        return []

    results = []
    for item in sorted(target_dir.iterdir(), reverse=True):
        if item.name.startswith("."):
            continue
        size_bytes = item.stat().st_size if item.is_file() else sum(f.stat().st_size for f in item.rglob("*"))
        results.append({
            "name": item.name,
            "path": str(item),
            "size_bytes": size_bytes,
            "modified": datetime.fromtimestamp(item.stat().st_mtime, tz=timezone.utc).isoformat(),
        })
    return results


def main() -> None:
    """CLI entry point for backup operations."""
    parser = argparse.ArgumentParser(description="SecondSelf Backup and Recovery Tool")
    subparsers = parser.add_subparsers(dest="action", required=True)

    # create
    sub_create = subparsers.add_parser("create", help="Create a persistent data backup")
    sub_create.add_argument("--dest", help="Destination folder for backup file")
    sub_create.add_argument("--no-archive", action="store_true", help="Store as folder instead of .tar.gz")

    # restore
    sub_restore = subparsers.add_parser("restore", help="Restore knowledge base from backup")
    sub_restore.add_argument("backup_path", help="Path to .tar.gz or backup directory")

    # list
    sub_list = subparsers.add_parser("list", help="List available backups")
    sub_list.add_argument("--dest", help="Backup directory to inspect")

    args = parser.parse_args()

    if args.action == "create":
        out = create_backup(dest_dir=args.dest, as_archive=not args.no_archive)
        print(f"✅ Backup created successfully at:\n   {out}")
    elif args.action == "restore":
        res = restore_backup(args.backup_path)
        print(f"✅ Backup restored successfully from {args.backup_path}")
        print(f"   Notes: {res['recovery']['stats'].get('total_notes', 0)}")
        print(f"   Captures: {res['recovery']['stats'].get('raw_captures', 0)}")
        print(f"   Graph Nodes: {res['recovery']['stats'].get('graph_nodes', 0)}")
    elif args.action == "list":
        backups = list_backups(dest_dir=args.dest)
        if not backups:
            print("No backups found.")
        else:
            print(f"Found {len(backups)} backup(s):")
            for b in backups:
                mb = b["size_bytes"] / (1024 * 1024)
                print(f" - {b['name']} ({mb:.2f} MB) - {b['modified']}")


if __name__ == "__main__":
    main()

