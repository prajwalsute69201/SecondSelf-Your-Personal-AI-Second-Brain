"""SecondSelf — Append-Only Manifest Audit Log."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Generator

from storage.models import ManifestEntry


class Manifest:
    """Append-only audit log recording every written raw record."""

    def __init__(self, manifest_file: Path) -> None:
        self.file_path = manifest_file

    def append(self, entry: ManifestEntry) -> None:
        """Appends a new record entry atomically to the JSONL log file.

        Args:
            entry: ManifestEntry describing the written record.
        """
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        # Canonical JSON string on a single line
        line = json.dumps(entry.model_dump(), sort_keys=True, ensure_ascii=False) + "\n"
        with self.file_path.open("a", encoding="utf-8") as f:
            f.write(line)

    def iter_entries(self) -> Generator[ManifestEntry, None, None]:
        """Yields entries one-by-one from the manifest file without buffering all in memory."""
        if not self.file_path.exists():
            return

        with self.file_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                clean_line = line.strip()
                if not clean_line:
                    continue
                try:
                    data = json.loads(clean_line)
                    yield ManifestEntry.model_validate(data)
                except Exception:
                    # Ignore malformed trailing lines if any, or continue
                    continue

    def read_all(self) -> list[ManifestEntry]:
        """Returns all logged manifest entries."""
        return list(self.iter_entries())

    def find_by_hash(self, content_hash: str) -> ManifestEntry | None:
        """Looks up a manifest entry by content SHA-256 hash."""
        target = content_hash.strip().lower()
        for entry in self.iter_entries():
            if entry.content_hash.lower() == target:
                return entry
        return None

    def find_by_id(self, record_id: str) -> ManifestEntry | None:
        """Looks up a manifest entry by unique record ID."""
        target = record_id.strip()
        for entry in self.iter_entries():
            if entry.record_id == target:
                return entry
        return None

    def count(self) -> int:
        """Returns the total number of entries logged in the manifest."""
        return sum(1 for _ in self.iter_entries())
