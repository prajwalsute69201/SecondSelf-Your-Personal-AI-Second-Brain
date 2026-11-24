"""SecondSelf — Fixed PARA Knowledge Taxonomy and Decision Criteria.

Defines the core PARA taxonomy (Projects, Areas, Resources, Archives) formulated
by Tiago Forte, enriched with formal decision criteria, time horizons, signal definitions,
and alias normalizers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PARACategory(str, Enum):
    """The four core categories of the P.A.R.A. knowledge organization framework."""

    PROJECT = "Projects"
    AREA = "Areas"
    RESOURCE = "Resources"
    ARCHIVE = "Archives"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_string(cls, value: str) -> PARACategory:
        """Parses arbitrary string input or alias into a canonical PARACategory.

        Handles case-insensitivity, singular/plural variants, and domain synonyms.
        Falls back to RESOURCE if unparseable.

        Args:
            value: Raw category string (e.g. 'project', 'Areas', 'reference', 'done').

        Returns:
            Canonical PARACategory enum member.
        """
        cleaned = value.strip().lower()
        if cleaned in CATEGORY_ALIASES:
            return CATEGORY_ALIASES[cleaned]

        # Exact match attempt
        for member in cls:
            if member.value.lower() == cleaned:
                return member

        # Default fallback
        return cls.RESOURCE


@dataclass(frozen=True)
class CategoryCriteria:
    """Formal decision criteria, boundary conditions, and signal profiles for a PARA category."""

    category: PARACategory
    title: str
    description: str
    core_question: str
    time_horizon: str
    target_folder: str
    primary_signals: tuple[str, ...] = field(default_factory=tuple)
    counter_signals: tuple[str, ...] = field(default_factory=tuple)
    examples: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Serializes criteria into a plain dictionary."""
        return {
            "category": self.category.value,
            "title": self.title,
            "description": self.description,
            "core_question": self.core_question,
            "time_horizon": self.time_horizon,
            "target_folder": self.target_folder,
            "primary_signals": list(self.primary_signals),
            "counter_signals": list(self.counter_signals),
            "examples": list(self.examples),
        }


# Canonical per-category definitions and decision rubrics
TAXONOMY_CRITERIA: dict[PARACategory, CategoryCriteria] = {
    PARACategory.PROJECT: CategoryCriteria(
        category=PARACategory.PROJECT,
        title="Projects",
        description="Active, time-bound efforts with a specific objective, deliverable, and explicit deadline.",
        core_question="Does this have a clear end state and an active deadline or target completion date?",
        time_horizon="Finite (days, weeks, months with defined completion)",
        target_folder="wiki/Projects",
        primary_signals=(
            "due dates, deadlines, target dates",
            "sprint goals, milestones, release targets (e.g. v1.0, MVP)",
            "explicit deliverables (launch, ship, deliver, implement)",
            "action items, tasks, checklists with immediate completion intent",
        ),
        counter_signals=(
            "no deadline or indefinite ongoing scope (points to Area)",
            "pure general reference without action items (points to Resource)",
            "already completed, shipped, or superseded efforts (points to Archive)",
        ),
        examples=(
            "Deliver Q4 Marketing Campaign by Dec 15",
            "Ship SecondSelf v0.2 Graph Projection",
            "Organize Conference Workshop slides by Friday",
        ),
    ),
    PARACategory.AREA: CategoryCriteria(
        category=PARACategory.AREA,
        title="Areas",
        description="Ongoing responsibilities, standards, routines, and domains of life/work requiring continuous maintenance with no defined end date.",
        core_question="Is this a standard of performance or domain of responsibility to maintain over time indefinitely?",
        time_horizon="Indefinite (ongoing, recurring cadence, never 'finished')",
        target_folder="wiki/Areas",
        primary_signals=(
            "recurring cadences (weekly review, monthly budget, daily routine)",
            "responsibilities (health, finances, team management, operations)",
            "standard operating procedures (SOP), standards to maintain, protocols",
            "KPIs, habit tracking, personal governance, continuous upkeep",
        ),
        counter_signals=(
            "discrete one-off deadline or single finish line (points to Project)",
            "static technical documentation without personal responsibility (points to Resource)",
            "retired roles or discontinued habits (points to Archive)",
        ),
        examples=(
            "Weekly System Architecture Review Checklist",
            "Personal Health & Workout Routine Maintenance",
            "Direct Reports 1-on-1 Notes & Career Development",
        ),
    ),
    PARACategory.RESOURCE: CategoryCriteria(
        category=PARACategory.RESOURCE,
        title="Resources",
        description="Topics of ongoing or future interest, reference material, cheatsheets, tutorials, guides, and documentation for future retrieval.",
        core_question="Is this reference material or conceptual knowledge that could be useful in the future?",
        time_horizon="Timeless (stored for future lookup, research, or utility)",
        target_folder="wiki/Resources",
        primary_signals=(
            "cheatsheets, syntax guides, API references, manuals, handbooks",
            "articles, tutorials, deep-dives, book summaries, literature notes",
            "curated lists, frameworks, definitions, design patterns, whitepapers",
            "neutral or third-party knowledge without direct personal commitments",
        ),
        counter_signals=(
            "contains active personal deadlines or sprint deliverables (points to Project)",
            "personal recurring checklists or maintenance duties (points to Area)",
            "deprecated frameworks or obsolete historical tech notes (points to Archive)",
        ),
        examples=(
            "Python 3.14 Concurrency & Subinterpreters Cheatsheet",
            "Cosine Similarity & Vector Embedding Mathematical Foundations",
            "Guide to Vercel Serverless Function Optimization",
        ),
    ),
    PARACategory.ARCHIVE: CategoryCriteria(
        category=PARACategory.ARCHIVE,
        title="Archives",
        description="Inactive, completed, deprecated, or superseded items from Projects, Areas, or Resources preserved for historical auditability.",
        core_question="Is this effort finished, inactive, deprecated, or no longer relevant for day-to-day operations?",
        time_horizon="Past (historical, frozen, inactive)",
        target_folder="wiki/Archives",
        primary_signals=(
            "completed project post-mortems, retrospective summaries",
            "deprecated software architectures, legacy API specs, obsolete libraries",
            "past routines, discontinued responsibilities, old job notes",
            "explicit status markers: 'DONE', 'SHIPPED', 'DEPRECATED', 'ARCHIVED'",
            "staleness: items untouched/completed for over 180 days",
        ),
        counter_signals=(
            "active pending deadline or upcoming milestone (points to Project)",
            "currently active recurring habit or ongoing maintenance (points to Area)",
            "currently valid reference documentation (points to Resource)",
        ),
        examples=(
            "2025 Q3 Legacy Auth Migration (Completed & Retired)",
            "Deprecated Python 2.7 Migration Checklist",
            "Post-Mortem: Serverless Cold Start Latency Spike",
        ),
    ),
}

# Domain and common linguistic aliases mapping directly to PARACategory
CATEGORY_ALIASES: dict[str, PARACategory] = {
    # Projects
    "project": PARACategory.PROJECT,
    "projects": PARACategory.PROJECT,
    "task": PARACategory.PROJECT,
    "tasks": PARACategory.PROJECT,
    "deliverable": PARACategory.PROJECT,
    "milestone": PARACategory.PROJECT,
    "sprint": PARACategory.PROJECT,
    "work": PARACategory.PROJECT,
    # Areas
    "area": PARACategory.AREA,
    "areas": PARACategory.AREA,
    "habit": PARACategory.AREA,
    "habits": PARACategory.AREA,
    "routine": PARACategory.AREA,
    "routines": PARACategory.AREA,
    "maintenance": PARACategory.AREA,
    "responsibility": PARACategory.AREA,
    "responsibilities": PARACategory.AREA,
    "standard": PARACategory.AREA,
    "standards": PARACategory.AREA,
    "sop": PARACategory.AREA,
    # Resources
    "resource": PARACategory.RESOURCE,
    "resources": PARACategory.RESOURCE,
    "reference": PARACategory.RESOURCE,
    "references": PARACategory.RESOURCE,
    "cheatsheet": PARACategory.RESOURCE,
    "guide": PARACategory.RESOURCE,
    "guides": PARACategory.RESOURCE,
    "article": PARACategory.RESOURCE,
    "articles": PARACategory.RESOURCE,
    "doc": PARACategory.RESOURCE,
    "docs": PARACategory.RESOURCE,
    "documentation": PARACategory.RESOURCE,
    "reading": PARACategory.RESOURCE,
    "notes": PARACategory.RESOURCE,
    "tutorial": PARACategory.RESOURCE,
    # Archives
    "archive": PARACategory.ARCHIVE,
    "archives": PARACategory.ARCHIVE,
    "archived": PARACategory.ARCHIVE,
    "done": PARACategory.ARCHIVE,
    "completed": PARACategory.ARCHIVE,
    "deprecated": PARACategory.ARCHIVE,
    "inactive": PARACategory.ARCHIVE,
    "legacy": PARACategory.ARCHIVE,
    "historical": PARACategory.ARCHIVE,
    "trash": PARACategory.ARCHIVE,
    "old": PARACategory.ARCHIVE,
}


def get_criteria(category: PARACategory | str) -> CategoryCriteria:
    """Retrieves the CategoryCriteria descriptor for a category or alias string."""
    cat_enum = category if isinstance(category, PARACategory) else PARACategory.from_string(category)
    return TAXONOMY_CRITERIA[cat_enum]
