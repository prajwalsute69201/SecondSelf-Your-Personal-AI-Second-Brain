"""SecondSelf — Autonomous PARA Classification Module.

Exposes the unified API for autonomous knowledge classification, deterministic
heuristic signal extraction, and Groq LLM-assisted hybrid inference.
"""

from __future__ import annotations

from classify.classifier import PARAClassifier, classify
from classify.cli import main, process_batch
from classify.llm_prompt import (
    PROMPT_VERSION,
    SYSTEM_PROMPT_V1,
    LLMResponseSchema,
    LLMValidationError,
    build_user_prompt,
    parse_and_validate_llm_response,
)
from classify.models import PARAClassification, Signal
from classify.rules import extract_signals, score_rules
from classify.taxonomy import (
    CATEGORY_ALIASES,
    TAXONOMY_CRITERIA,
    CategoryCriteria,
    PARACategory,
    get_criteria,
)

__all__ = [
    "PARACategory",
    "CategoryCriteria",
    "TAXONOMY_CRITERIA",
    "CATEGORY_ALIASES",
    "get_criteria",
    "Signal",
    "PARAClassification",
    "extract_signals",
    "score_rules",
    "SYSTEM_PROMPT_V1",
    "PROMPT_VERSION",
    "LLMResponseSchema",
    "LLMValidationError",
    "build_user_prompt",
    "parse_and_validate_llm_response",
    "PARAClassifier",
    "classify",
    "main",
    "process_batch",
]
