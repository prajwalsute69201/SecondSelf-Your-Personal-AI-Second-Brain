"""SecondSelf — Validation & Quality Gate Public API."""

from __future__ import annotations

from validation.acceptance import AcceptanceCriteriaValidator
from validation.base import BaseValidator, IntegrityFailure, ValidationError, VerificationError
from validation.completeness import CompletenessValidator
from validation.integrity import IntegrityValidator
from validation.models import ValidationResult, VerificationReport, VerificationSummary
from validation.orchestrator import VerificationOrchestrator
from validation.report import format_terminal_report
from validation.schema import SchemaValidator

__all__ = [
    "AcceptanceCriteriaValidator",
    "BaseValidator",
    "CompletenessValidator",
    "IntegrityFailure",
    "IntegrityValidator",
    "SchemaValidator",
    "ValidationError",
    "ValidationResult",
    "VerificationError",
    "VerificationOrchestrator",
    "VerificationReport",
    "VerificationSummary",
    "format_terminal_report",
]
