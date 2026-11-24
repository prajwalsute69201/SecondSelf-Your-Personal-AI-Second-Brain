"""SecondSelf — Artifact Staging Manager (Task 3.3).

Safely stages precomputed Task 3.1 artifacts (graph.json, index.json) into the static
frontend assets directory without mutating the source corpus or original artifacts.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger("secondself.preview.staging")


def compute_sha256(file_path: Path) -> str:
    """Computes hexadecimal SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class StagingError(Exception):
    """Raised when artifact staging fails due to missing source or permissions."""


class ArtifactStagingManager:
    """Manages read-only staging of Task 3.1 artifacts into public/data/."""

    def __init__(
        self,
        source_graph: Path,
        public_dir: Path,
        source_index: Path | None = None,
    ) -> None:
        self.source_graph = Path(source_graph).resolve()
        self.public_dir = Path(public_dir).resolve()
        self.source_index = Path(source_index).resolve() if source_index else None
        self.dest_data_dir = self.public_dir / "data"

    def stage(self) -> dict[str, dict[str, Any]]:
        """Copies source artifacts to public/data/ defensively.

        Returns:
            Dictionary containing metadata for each staged artifact (path, sha256, size_bytes).
        """
        if not self.source_graph.exists() or not self.source_graph.is_file():
            raise StagingError(
                f"Source graph artifact not found at: {self.source_graph}. "
                "Ensure Task 3.1 graph projection has been generated first."
            )

        self.dest_data_dir.mkdir(parents=True, exist_ok=True)
        staged_manifest: dict[str, dict[str, Any]] = {}

        # 1. Stage graph.json into public/data/graph.json
        dest_graph = self.dest_data_dir / "graph.json"
        try:
            if not (dest_graph.exists() and dest_graph.samefile(self.source_graph)):
                shutil.copy2(self.source_graph, dest_graph)
                logger.info("Staged %s -> %s", self.source_graph, dest_graph)
        except (shutil.SameFileError, OSError) as err:
            logger.debug("Artifact already in place or samefile: %s", err)

        graph_hash = compute_sha256(dest_graph)
        staged_manifest["graph.json"] = {
            "source_path": str(self.source_graph),
            "staged_path": str(dest_graph),
            "sha256": graph_hash,
            "size_bytes": dest_graph.stat().st_size,
        }

        # 2. Also ensure root public/graph.json exists as fallback path
        root_graph = self.public_dir / "graph.json"
        if not root_graph.exists() or root_graph.stat().st_size != dest_graph.stat().st_size:
            try:
                shutil.copy2(self.source_graph, root_graph)
            except (shutil.SameFileError, OSError):
                pass

        # 3. Stage index.json if provided or found in source vicinity
        candidate_index = self.source_index or (self.public_dir / "index.json")
        if candidate_index and candidate_index.exists() and candidate_index.is_file():
            dest_index = self.dest_data_dir / "index.json"
            try:
                if not (dest_index.exists() and dest_index.samefile(candidate_index)):
                    shutil.copy2(candidate_index, dest_index)
            except (shutil.SameFileError, OSError):
                pass

            index_hash = compute_sha256(candidate_index)
            staged_manifest["index.json"] = {
                "source_path": str(candidate_index),
                "staged_path": str(dest_index if dest_index.exists() else candidate_index),
                "sha256": index_hash,
                "size_bytes": candidate_index.stat().st_size,
            }

        return staged_manifest
