"""SecondSelf — Semantic Vectorization and Auto-Linking Data Contracts.

Defines the core data models for vector embeddings, relational links,
auto-linking results, and note representations.
"""

from __future__ import annotations

import datetime
import math
import re
from pathlib import Path
from typing import Any, Literal
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

EmbeddingStatus = Literal["success", "failed"]
LinkMethod = Literal["item_to_item", "item_to_centroid", "item_to_tag"]


class Embedding(BaseModel):
    """Normalized dense semantic vector representation of content."""

    vector: list[float] = Field(
        default_factory=list,
        description="Normalized high-dimensional float array",
    )
    model_name: str = Field(
        ...,
        description="Canonical identifier of embedding model (e.g. sentence-transformers/all-MiniLM-L6-v2)",
    )
    dimensions: int = Field(
        ...,
        ge=1,
        description="Dimensionality of embedding vector (e.g. 384)",
    )
    content_hash: str = Field(
        ...,
        description="SHA-256 digest of normalized source text for idempotency",
    )
    status: EmbeddingStatus = Field(
        default="success",
        description="Generation status ('success' or 'failed' on graceful degradation)",
    )
    error_message: str | None = Field(
        default=None,
        description="Diagnostic error message if generation failed",
    )
    created_at: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat(),
        description="ISO-8601 UTC timestamp of vector calculation",
    )

    model_config = ConfigDict(populate_by_name=True)

    def to_numpy(self) -> np.ndarray:
        """Converts vector into a 1D NumPy float32 array."""
        if not self.vector:
            return np.zeros(self.dimensions, dtype=np.float32)
        return np.asarray(self.vector, dtype=np.float32)

    def norm(self) -> float:
        """Computes Euclidean (L2) norm of the vector."""
        if not self.vector:
            return 0.0
        return float(np.linalg.norm(self.vector))

    def is_normalized(self, tol: float = 1e-3) -> bool:
        """Returns True if the vector norm is approximately 1.0."""
        return abs(self.norm() - 1.0) <= tol

    def dot(self, other: Embedding | list[float] | np.ndarray) -> float:
        """Computes inner product with another vector."""
        if isinstance(other, Embedding):
            arr = other.to_numpy()
        else:
            arr = np.asarray(other, dtype=np.float32)
        return float(np.dot(self.to_numpy(), arr))

    def cosine_similarity(self, other: Embedding | list[float] | np.ndarray) -> float:
        """Computes cosine similarity between two vectors."""
        vec_a = self.to_numpy()
        if isinstance(other, Embedding):
            vec_b = other.to_numpy()
        else:
            vec_b = np.asarray(other, dtype=np.float32)

        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


class Link(BaseModel):
    """Directed or reciprocal relationship between two knowledge items or categories."""

    source_id: str = Field(..., description="Unique ID of source item")
    target_id: str = Field(..., description="Unique ID of linked target item or category")
    score: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Cosine similarity or affinity score",
    )
    reason: str = Field(
        ...,
        description="Auditable human-readable explanation for this relationship",
    )
    method: LinkMethod = Field(
        default="item_to_item",
        description="Classification of relationship mechanism",
    )
    target_title: str | None = Field(
        default=None,
        description="Descriptive title of target item if available",
    )
    target_category: str | None = Field(
        default=None,
        description="PARA category of target item if available",
    )

    model_config = ConfigDict(populate_by_name=True)

    def to_wikilink(self) -> str:
        """Formats link as a Markdown Wikilink with percentage affinity."""
        display = self.target_title or self.target_id
        pct = max(0, min(100, int(round(self.score * 100))))
        return f"- [[{self.target_id}|{display}]] — Match: {pct}%"

    def to_dict(self) -> dict[str, Any]:
        """Serializes link into a plain dictionary."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "score": round(self.score, 4),
            "reason": self.reason,
            "method": self.method,
            "target_title": self.target_title,
            "target_category": self.target_category,
            "wikilink": self.to_wikilink(),
        }


class LinkResult(BaseModel):
    """Complete relationship discovery output for a single knowledge item."""

    item_id: str = Field(..., description="Identifier of the evaluated item")
    links: list[Link] = Field(
        default_factory=list,
        description="Discovered item-to-item semantic relationships",
    )
    embedding_ref: str = Field(
        default="",
        description="SHA-256 content hash of the underlying vector",
    )
    nearest_centroid: str | None = Field(
        default=None,
        description="Nearest PARA category centroid name",
    )
    centroid_score: float | None = Field(
        default=None,
        description="Similarity score to the nearest PARA centroid",
    )
    category_alignment: bool | None = Field(
        default=None,
        description="True if item's assigned category matches its nearest centroid",
    )
    centroid_link: Link | None = Field(
        default=None,
        description="Secondary link to nearest PARA category centroid",
    )
    status: Literal["success", "partial", "failed"] = Field(
        default="success",
        description="Overall linking status",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Diagnostic and note context metadata",
    )

    model_config = ConfigDict(populate_by_name=True)

    def to_dict(self) -> dict[str, Any]:
        """Serializes link result into a plain JSON-compatible dictionary."""
        return {
            "item_id": self.item_id,
            "links": [link.to_dict() for link in self.links],
            "embedding_ref": self.embedding_ref,
            "nearest_centroid": self.nearest_centroid,
            "centroid_score": round(self.centroid_score, 4) if self.centroid_score is not None else None,
            "category_alignment": self.category_alignment,
            "centroid_link": self.centroid_link.to_dict() if self.centroid_link else None,
            "status": self.status,
            "metadata": self.metadata,
        }


class NoteItem(BaseModel):
    """Standardized representation of an ingested or curated Markdown note."""

    id: str = Field(..., description="Unique canonical note identifier")
    title: str = Field(default="Untitled", description="Derived or frontmatter title")
    category: str = Field(default="Resources", description="Assigned PARA category")
    summary: str = Field(default="", description="1-2 sentence core takeaway")
    tags: list[str] = Field(default_factory=list, description="Topical tags")
    content: str = Field(default="", description="Raw note body content")
    file_path: Path | None = Field(default=None, description="Absolute filesystem path to note")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Custom frontmatter attributes")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def composite_text(self) -> str:
        """Constructs dense semantic payload for vector embedding as per architecture spec:
        text_payload = f"{title}\\n{summary}\\n{' '.join(tags)}\\n{content[:1500]}"
        """
        tags_str = " ".join(self.tags)
        body_snippet = self.content[:1500].strip()
        lines = [self.title, self.summary, tags_str, body_snippet]
        return "\n".join(part for part in lines if part.strip()).strip()

    @classmethod
    def from_markdown_file(cls, path: Path) -> NoteItem:
        """Parses a markdown file with optional YAML frontmatter into a NoteItem."""
        text = path.read_text(encoding="utf-8")
        note_id = path.stem
        title = "Untitled"
        category = path.parent.name if path.parent.name in ("Projects", "Areas", "Resources", "Archives") else "Resources"
        summary = ""
        tags: list[str] = []
        metadata: dict[str, Any] = {}

        # Parse YAML frontmatter if present (between --- markers)
        fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", text, flags=re.DOTALL)
        body = text
        if fm_match:
            frontmatter_raw = fm_match.group(1)
            body = fm_match.group(2)
            for line in frontmatter_raw.splitlines():
                if ":" in line:
                    key, val = line.split(":", 1)
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    if key == "title" and val:
                        title = val
                    elif key == "category" and val:
                        category = val
                    elif key == "summary" and val:
                        summary = val
                    elif key == "tags":
                        pass  # handled or populated
                    metadata[key] = val

            # Check list items for tags
            tag_matches = re.findall(r"^\s*-\s+([a-zA-Z0-9_\-]+)\s*$", frontmatter_raw, flags=re.MULTILINE)
            if tag_matches:
                tags = tag_matches

        # Check for first H1 header if title still default
        if title == "Untitled":
            h1_match = re.search(r"^#\s+(.+)$", body, flags=re.MULTILINE)
            if h1_match:
                title = h1_match.group(1).strip()

        # Clean related notes section out of body if already present
        clean_body = re.sub(r"##\s+Related Notes.*?(?:\n##|\Z)", "", body, flags=re.DOTALL).strip()

        return cls(
            id=note_id,
            title=title,
            category=category,
            summary=summary,
            tags=tags,
            content=clean_body,
            file_path=path.resolve(),
            metadata=metadata,
        )
