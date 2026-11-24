"""SecondSelf — Versioned LLM Prompting and Strict JSON Output Validation.

Provides versioned system and user prompts, strict JSON schema definitions,
and fail-safe parsing and validation for LLM-driven PARA classification.
"""

from __future__ import annotations

import json
import re
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, field_validator

from classify.taxonomy import PARACategory

PROMPT_VERSION = "1.0.0"

# ------------------------------------------------------------------------------
# JSON Schema for LLM Response Format
# ------------------------------------------------------------------------------

LLM_CLASSIFICATION_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "PARAClassificationResult",
    "description": "Structured JSON response required for autonomous PARA classification",
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": ["Projects", "Areas", "Resources", "Archives"],
            "description": "The classified P.A.R.A category",
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Confidence score from 0.0 (uncertain) to 1.0 (certain)",
        },
        "rationale": {
            "type": "string",
            "description": "Explicit, auditable justification explaining why this item belongs to this category",
        },
        "signals": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of textual or contextual evidence markers discovered in the content",
        },
        "title": {
            "type": "string",
            "description": "Concise, descriptive title for the note (maximum 8 words)",
        },
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "3 to 6 lowercase hyphenated topical tags (e.g. 'vector-search', 'ai-rag')",
        },
        "summary": {
            "type": "string",
            "description": "High-signal 1-2 sentence core takeaway summarizing key insights",
        },
    },
    "required": ["category", "confidence", "rationale", "signals", "title", "tags", "summary"],
    "additionalProperties": False,
}


# ------------------------------------------------------------------------------
# System & User Prompt Templates (Version 1.0.0)
# ------------------------------------------------------------------------------

SYSTEM_PROMPT_V1 = """You are SecondSelf's Autonomous PARA Knowledge Classifier (Engine v1.0.0).
Your task is to analyze captured knowledge items and classify them into Tiago Forte's P.A.R.A. framework with complete objectivity, strict determinism, and high signal-to-noise.

The P.A.R.A. Taxonomy:
1. Projects: Active, time-bound efforts with a defined objective and explicit deadline or finish line (e.g., 'Deliver Q4 MVP by Dec 15', 'Prepare slides for Friday conference').
2. Areas: Ongoing responsibilities, standards to maintain, habits, routines, or continuous domains with no fixed end date (e.g., 'Weekly System Review Checklist', 'Personal Health Routine', 'Team 1-on-1 Notes').
3. Resources: Reference materials, tutorials, guides, documentation, cheatsheets, articles, or topics of ongoing interest for future lookup without immediate action (e.g., 'Python 3.14 Concurrency Guide', 'Cosine Similarity Math').
4. Archives: Inactive, completed, deprecated, retrospective, or historical items from Projects, Areas, or Resources (e.g., '2025 Retrospective', 'Deprecated Auth Architecture', completed milestones).

Decision Rules:
- If content has explicit deadlines, deliverables, or sprint goals -> Projects.
- If content describes continuous maintenance, habits, protocols, or recurring cadences -> Areas.
- If content is informational, reference, cheatsheet, tutorial, or external article -> Resources.
- If content is marked DONE, completed, deprecated, or historical post-mortem -> Archives.
- If ambiguous, assign lower confidence (< 0.60) and note the ambiguity in rationale.

Output Constraint:
Return ONLY a valid JSON object matching the required schema. Do NOT include markdown code blocks, explanation outside JSON, or conversational filler."""

USER_PROMPT_TEMPLATE_V1 = """Classify the following ingested content item:

[Source Metadata]
Source Type: {source_type}
Source URI: {source_uri}
Capture Date: {timestamp}
Given Title: {title}

[Content Payload]
{content}

Respond strictly in valid JSON format matching the schema."""


# ------------------------------------------------------------------------------
# Pydantic Schema for LLM Response Validation
# ------------------------------------------------------------------------------

class LLMResponseSchema(BaseModel):
    """Pydantic validator for raw JSON responses received from LLM providers."""

    category: PARACategory = Field(..., description="Target PARA category")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    rationale: str = Field(..., description="Auditable reasoning")
    signals: list[str] = Field(default_factory=list, description="Extracted signal markers")
    title: str = Field(default="Untitled", description="Derived concise title")
    tags: list[str] = Field(default_factory=list, description="Topical tags")
    summary: str = Field(default="", description="1-2 sentence core takeaway")

    model_config = ConfigDict(populate_by_name=True)

    @field_validator("category", mode="before")
    @classmethod
    def validate_category(cls, value: Any) -> PARACategory:
        """Coerces string category or alias to canonical PARACategory enum."""
        if isinstance(value, PARACategory):
            return value
        return PARACategory.from_string(str(value))

    @field_validator("confidence", mode="before")
    @classmethod
    def validate_confidence(cls, value: Any) -> float:
        """Clamps confidence score to valid [0.0, 1.0] interval."""
        try:
            val = float(value)
            return max(0.0, min(1.0, val))
        except (ValueError, TypeError):
            return 0.50

    @field_validator("title", mode="before")
    @classmethod
    def clean_title(cls, value: Any) -> str:
        """Ensures title is non-empty and stripped."""
        text = str(value or "Untitled").strip()
        return text if text else "Untitled"


class LLMValidationError(Exception):
    """Raised when an LLM response cannot be parsed or validated against the schema."""
    pass


def build_user_prompt(
    content: str,
    metadata: dict[str, Any] | None = None,
    max_content_chars: int = 12000,
) -> str:
    """Formats the standardized user prompt, safely truncating oversized text.

    Args:
        content: Unstructured or markdown content.
        metadata: Sidecar metadata dict.
        max_content_chars: Maximum characters included to prevent token explosion.

    Returns:
        Formatted user prompt string.
    """
    meta = metadata or {}
    text = content.strip()

    if len(text) > max_content_chars:
        # Keep head and tail to preserve context and conclusions
        head_len = max_content_chars // 2
        tail_len = max_content_chars // 2
        text = (
            f"{text[:head_len]}\n\n[... content truncated for LLM classification ...]\n\n"
            f"{text[-tail_len:]}"
        )

    return USER_PROMPT_TEMPLATE_V1.format(
        source_type=meta.get("source_type", "text"),
        source_uri=meta.get("source_uri") or "N/A",
        timestamp=meta.get("timestamp") or meta.get("created_at") or "N/A",
        title=meta.get("title", "Untitled"),
        content=text,
    )


def parse_and_validate_llm_response(raw_response_text: str) -> LLMResponseSchema:
    """Parses and strictly validates raw text returned by the LLM into LLMResponseSchema.

    Args:
        raw_response_text: Raw string returned by LLM completion API.

    Returns:
        Validated LLMResponseSchema instance.

    Raises:
        LLMValidationError: If the response is not valid JSON or violates schema constraints.
    """
    cleaned = raw_response_text.strip()

    # Strip markdown code fences if present (e.g. ```json ... ```)
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

    # Locate JSON boundaries if extra leading/trailing prose exists
    json_start = cleaned.find("{")
    json_end = cleaned.rfind("}")
    if json_start != -1 and json_end != -1 and json_end > json_start:
        cleaned = cleaned[json_start : json_end + 1]

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise LLMValidationError(f"Malformed JSON in LLM response: {exc}") from exc

    if not isinstance(data, dict):
        raise LLMValidationError(f"Expected JSON object, got {type(data).__name__}")

    try:
        return LLMResponseSchema.model_validate(data)
    except Exception as exc:
        raise LLMValidationError(f"Schema validation failed for LLM output: {exc}") from exc
