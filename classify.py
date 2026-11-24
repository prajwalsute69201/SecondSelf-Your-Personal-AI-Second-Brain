#!/usr/bin/env python3
"""SecondSelf — Autonomous PARA Classification CLI & Service.

The Librarian (Milestone 2 - Task 2.1):
Autonomous PARA classification module mapping verified, extracted content into
the P.A.R.A. taxonomy (Projects, Areas, Resources, Archives) with clear decision
logic, confidence scoring, auditable rationale, and graceful fallback.

Usage:
    python classify.py --text "Ship v1.0 by Friday"
    python classify.py --file raw/sample_capture.json
    python classify.py --batch --dry-run
    python classify.py --batch
"""

from __future__ import annotations

import sys

from classify import (
    CATEGORY_ALIASES,
    TAXONOMY_CRITERIA,
    CategoryCriteria,
    PARACategory,
    PARAClassification,
    PARAClassifier,
    Signal,
    classify,
    extract_signals,
    get_criteria,
    main,
    process_batch,
    score_rules,
)
from classify.cli import (
    EXIT_GENERAL_ERROR,
    EXIT_IO_ERROR,
    EXIT_SUCCESS,
    EXIT_USAGE_ERROR,
    build_arg_parser,
    find_unprocessed_records,
    serialize_note_to_wiki,
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
    "PARAClassifier",
    "classify",
    "main",
    "process_batch",
    "serialize_note_to_wiki",
    "find_unprocessed_records",
    "build_arg_parser",
    "EXIT_SUCCESS",
    "EXIT_GENERAL_ERROR",
    "EXIT_USAGE_ERROR",
    "EXIT_IO_ERROR",
]


if __name__ == "__main__":
    sys.exit(main())
