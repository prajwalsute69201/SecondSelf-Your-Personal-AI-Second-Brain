"""SecondSelf — Data Models for Retrieval-Augmented Generation (Task 4.1).

Defines typed data structures for retrieved corpus items, citation mappings,
bounded context bundles, and final synthesized answers.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RetrievedItem:
    """Represents an individual corpus item retrieved for query context."""
    id: str
    title: str
    snippet: str
    content: str
    score: float
    category: str = "Resources"
    source_type: str = "markdown"
    retrieval_method: str = "vector"  # "vector" or "graph_neighbor"
    tags: list[str] = field(default_factory=list)
    file_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Citation:
    """Explicit citation mapping an answer claim back to a source item."""
    item_id: str
    title: str
    snippet: str
    score: float
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ContextBundle:
    """Bounded context bundle assembled from top retrieved items."""
    formatted_context: str
    items: list[RetrievedItem]
    total_chars: int
    estimated_tokens: int
    dropped_items_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "formatted_context": self.formatted_context,
            "items": [it.to_dict() for it in self.items],
            "total_chars": self.total_chars,
            "estimated_tokens": self.estimated_tokens,
            "dropped_items_count": self.dropped_items_count,
        }


@dataclass
class Answer:
    """Complete answer produced by the retrieval & synthesis engine."""
    question: str
    text: str
    citations: list[Citation] = field(default_factory=list)
    retrieved: list[RetrievedItem] = field(default_factory=list)
    mode: str = "llm_synthesis"  # "llm_synthesis" or "retrieval_only"
    model: str | None = None
    prompt_version: str | None = None
    latency_ms: float = 0.0
    is_grounded: bool = True
    uncited_claims_flag: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "text": self.text,
            "citations": [c.to_dict() for c in self.citations],
            "retrieved": [r.to_dict() for r in self.retrieved],
            "mode": self.mode,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "latency_ms": round(self.latency_ms, 2),
            "is_grounded": self.is_grounded,
            "uncited_claims_flag": self.uncited_claims_flag,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def render_console(self) -> str:
        """Renders an attractive human-readable console representation."""
        lines = [
            "",
            "=" * 74,
            f"  SECONDSELF ANSWER [{self.mode.upper()}]",
            "=" * 74,
            f"  Question : {self.question}",
            f"  Model    : {self.model or 'None (pure retrieval)'}",
            f"  Latency  : {self.latency_ms:.1f} ms",
            f"  Grounded : {'YES' if self.is_grounded else 'NO (Unverified citations detected)'}",
        ]
        if self.uncited_claims_flag:
            lines.append("  Warning  : [!] Uncited claims detected in response text")

        lines.extend([
            "-" * 74,
            "  SYNTHESIZED ANSWER:",
            "",
        ])

        # Indent answer text slightly for readability
        for line in self.text.splitlines():
            lines.append(f"    {line}")

        lines.extend([
            "",
            "-" * 74,
            f"  CITATIONS ({len(self.citations)}):",
        ])

        if not self.citations:
            lines.append("    (No specific citations provided)")
        else:
            for idx, c in enumerate(self.citations, 1):
                clean_id = c.item_id.replace("item:", "")
                lines.append(f"    [{idx}] {c.title} (ID: {clean_id[:8]}... • Score: {c.score:.2f})")
                if c.snippet:
                    lines.append(f"        \"{c.snippet[:120]}...\"")

        lines.extend([
            "-" * 74,
            f"  TOP RETRIEVED SOURCES ({len(self.retrieved)}):",
        ])
        for idx, r in enumerate(self.retrieved, 1):
            clean_id = r.id.replace("item:", "")
            lines.append(
                f"    {idx}. [{r.category}] {r.title} "
                f"(Score: {r.score:.3f} • Method: {r.retrieval_method} • ID: {clean_id[:8]})"
            )

        lines.extend(["=" * 74, ""])
        return "\n".join(lines)
