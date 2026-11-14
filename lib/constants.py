"""System-wide constants and centralized category configuration for SecondSelf."""

from __future__ import annotations

from typing import Any

DEFAULT_USER_ID: str = "00000000-0000-0000-0000-000000000001"
DEFAULT_ADMIN_EMAIL: str = "admin@secondself.ai"

# Centralized category definitions for personal and professional knowledge management
CATEGORY_DEFINITIONS: dict[str, dict[str, Any]] = {
    # Original PARA categories
    "Projects": {
        "icon": "🚀",
        "description": "Active work with a specific goal, outcome, and deadline",
        "color": "#3b82f6",
        "border": "#1d4ed8",
        "highlight": "#60a5fa",
        "badge_class": "badge-projects",
    },
    "Areas": {
        "icon": "🌱",
        "description": "Ongoing responsibilities and life domains without a fixed end date",
        "color": "#10b981",
        "border": "#047857",
        "highlight": "#34d399",
        "badge_class": "badge-areas",
    },
    "Resources": {
        "icon": "📚",
        "description": "Reference material, topics of interest, guides, and bookmarks",
        "color": "#f59e0b",
        "border": "#b45309",
        "highlight": "#fbbf24",
        "badge_class": "badge-resources",
    },
    "Archives": {
        "icon": "📦",
        "description": "Completed, inactive, or historical items",
        "color": "#64748b",
        "border": "#334155",
        "highlight": "#94a3b8",
        "badge_class": "badge-archives",
    },
    # Expanded personal and professional categories
    "Work": {
        "icon": "💼",
        "description": "Work-related tasks, activities, and responsibilities",
        "color": "#6366f1",
        "border": "#4338ca",
        "highlight": "#818cf8",
        "badge_class": "badge-work",
    },
    "Payments": {
        "icon": "💳",
        "description": "Bills, transactions, subscriptions, and payments",
        "color": "#ec4899",
        "border": "#be185d",
        "highlight": "#f472b6",
        "badge_class": "badge-payments",
    },
    "Finance": {
        "icon": "💰",
        "description": "Financial planning, budgets, and expenses",
        "color": "#14b8a6",
        "border": "#0f766e",
        "highlight": "#2dd4bf",
        "badge_class": "badge-finance",
    },
    "Personal": {
        "icon": "👤",
        "description": "Personal notes, goals, and activities",
        "color": "#8b5cf6",
        "border": "#6d28d9",
        "highlight": "#a78bfa",
        "badge_class": "badge-personal",
    },
    "Learning": {
        "icon": "🎓",
        "description": "Courses, skills, and learning materials",
        "color": "#06b6d4",
        "border": "#0e7490",
        "highlight": "#22d3ee",
        "badge_class": "badge-learning",
    },
    "Health": {
        "icon": "❤️",
        "description": "Fitness, wellness, and health-related records",
        "color": "#ef4444",
        "border": "#b91c1c",
        "highlight": "#f87171",
        "badge_class": "badge-health",
    },
    "People": {
        "icon": "👥",
        "description": "Contacts, colleagues, and important relationships",
        "color": "#f97316",
        "border": "#c2410c",
        "highlight": "#fb923c",
        "badge_class": "badge-people",
    },
    "Ideas": {
        "icon": "💡",
        "description": "Thoughts, concepts, and future possibilities",
        "color": "#eab308",
        "border": "#a16207",
        "highlight": "#facc15",
        "badge_class": "badge-ideas",
    },
    "Events": {
        "icon": "📅",
        "description": "Meetings, appointments, and important dates",
        "color": "#d946ef",
        "border": "#a21caf",
        "highlight": "#e879f9",
        "badge_class": "badge-events",
    },
    "Shopping": {
        "icon": "🛒",
        "description": "Purchases, products, and shopping lists",
        "color": "#84cc16",
        "border": "#4d7c0f",
        "highlight": "#a3e635",
        "badge_class": "badge-shopping",
    },
}

# Ordered tuple of valid category names
VALID_CATEGORIES: tuple[str, ...] = tuple(CATEGORY_DEFINITIONS.keys())

# Alias for backwards compatibility with PARA-specific code
VALID_PARA_CATEGORIES: tuple[str, ...] = VALID_CATEGORIES

DEFAULT_CATEGORY: str = "Resources"

__all__ = [
    "DEFAULT_USER_ID",
    "DEFAULT_ADMIN_EMAIL",
    "CATEGORY_DEFINITIONS",
    "VALID_CATEGORIES",
    "VALID_PARA_CATEGORIES",
    "DEFAULT_CATEGORY",
]
