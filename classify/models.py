"""SecondSelf — Autonomous Classification Data Contracts and Models.

Defines the core data contracts for classification signals and the structured
PARAClassification result produced by the classifier module.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field

from classify.taxonomy import PARACategory, get_criteria

SignalSource = Literal["rule_content", "rule_metadata", "llm", "heuristic"]
ClassificationMethod = Literal["rules", "llm", "hybrid", "rules_fallback"]


class Signal(BaseModel):
    """An individual piece of auditable evidence extracted from content or metadata."""

    name: str = Field(..., description="Machine-readable signal identifier")
    category: PARACategory = Field(..., description="PARA category supported by this signal")
    weight: float = Field(..., ge=0.0, description="Additive confidence weight")
    matched_text: str = Field(default="", description="Relevant textual pattern or value matched")
    source: SignalSource = Field(default="rule_content", description="Origin of signal extraction")
    description: str = Field(default="", description="Human-readable explanation of the signal")

    model_config = ConfigDict(populate_by_name=True)

    def to_dict(self) -> dict[str, Any]:
        """Serializes signal to plain dictionary."""
        return {
            "name": self.name,
            "category": self.category.value,
            "weight": round(self.weight, 4),
            "matched_text": self.matched_text,
            "source": self.source,
            "description": self.description,
        }


class PARAClassification(BaseModel):
    """Deterministic, auditable classification result for an ingested knowledge item."""

    category: PARACategory = Field(..., description="Classified PARA category")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score bounded between 0.0 and 1.0",
    )
    rationale: str = Field(..., description="Auditable human-readable justification for the decision")
    signals: list[Signal] = Field(
        default_factory=list,
        description="Ordered list of evidence signals used in this classification",
    )
    category_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Score distribution across all candidate PARA categories",
    )
    needs_review: bool = Field(
        default=False,
        description="True if confidence is below threshold or classification is ambiguous",
    )
    method: ClassificationMethod = Field(
        default="hybrid",
        description="Execution mode: 'rules', 'llm', 'hybrid', or 'rules_fallback'",
    )
    classifier_version: str = Field(default="1.0.0", description="Version of classifier logic/prompt")
    content_hash: str = Field(default="", description="Deterministic SHA-256 digest of input content")
    title: str = Field(default="Untitled", description="Derived descriptive note title")
    tags: list[str] = Field(default_factory=list, description="Topical or thematic tags")
    summary: str = Field(default="", description="Concise 1-2 sentence core takeaway")
    suggested_folder: str = Field(default="", description="Target destination directory in wiki vault")
    timestamp: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat(),
        description="ISO-8601 UTC timestamp of classification event",
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Diagnostic and source metadata")

    model_config = ConfigDict(populate_by_name=True)

    def model_post_init(self, __context: Any) -> None:
        """Derives default target directory if not explicitly populated."""
        if not self.suggested_folder:
            criteria = get_criteria(self.category)
            self.suggested_folder = criteria.target_folder

    def to_dict(self) -> dict[str, Any]:
        """Serializes result into a plain JSON-compatible dictionary."""
        return {
            "category": self.category.value,
            "confidence": round(self.confidence, 4),
            "rationale": self.rationale,
            "needs_review": self.needs_review,
            "method": self.method,
            "classifier_version": self.classifier_version,
            "content_hash": self.content_hash,
            "title": self.title,
            "tags": self.tags,
            "summary": self.summary,
            "suggested_folder": self.suggested_folder,
            "timestamp": self.timestamp,
            "category_scores": {k: round(v, 4) for k, v in self.category_scores.items()},
            "signals": [s.to_dict() for s in self.signals],
            "metadata": self.metadata,
        }

    def to_frontmatter_dict(self) -> dict[str, Any]:
        """Generates structured metadata dictionary suitable for YAML note frontmatter."""
        return {
            "title": self.title,
            "category": self.category.value,
            "confidence": round(self.confidence, 4),
            "needs_review": self.needs_review,
            "classifier_version": self.classifier_version,
            "method": self.method,
            "content_hash": self.content_hash,
            "tags": self.tags,
            "summary": self.summary,
            "classified_at": self.timestamp,
        }

    def to_markdown_frontmatter(self) -> str:
        """Renders YAML frontmatter block for markdown note serialization."""
        fm = self.to_frontmatter_dict()
        lines = ["---"]
        for key, value in fm.items():
            if isinstance(value, list):
                if not value:
                    lines.append(f"{key}: []")
                else:
                    lines.append(f"{key}:")
                    for item in value:
                        lines.append(f"  - {item}")
            elif isinstance(value, bool):
                lines.append(f"{key}: {'true' if value else 'false'}")
            elif isinstance(value, (int, float)):
                lines.append(f"{key}: {value}")
            else:
                escaped = str(value).replace('"', '\\"')
                lines.append(f'{key}: "{escaped}"')
        lines.append("---")
        return "\n".join(lines)

    def to_wiki_path(self, record_id: str, base_dir: Path | str = "wiki") -> Path:
        """Computes the target file path: wiki/{category}/{record_id}.md."""
        return Path(base_dir) / self.category.value / f"{record_id}.md"
