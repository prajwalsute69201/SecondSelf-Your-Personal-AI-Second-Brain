"""SecondSelf — Base Validator Interface & Exceptions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from validation.models import ValidationResult


class VerificationError(Exception):
    """Base exception for pipeline quality gate and deliverable failures."""


class ValidationError(VerificationError):
    """Raised when a deliverable fails schema, completeness, or acceptance criteria."""


class IntegrityFailure(VerificationError):
    """Raised when cryptographic data tampering or hash mismatch is detected."""


class BaseValidator(ABC):
    """Abstract base class for all quality gate validators."""

    name: str = "base_validator"

    @abstractmethod
    def validate(self, target: Any, context: dict[str, Any] | None = None) -> ValidationResult:
        """Executes verification rules against the target deliverable.

        Args:
            target: The deliverable to verify (RawRecord, RawStore, ExtractedContent).
            context: Optional contextual parameters (min_records, environment).

        Returns:
            ValidationResult with passed status, checks run, and error details.
        """
