"""SecondSelf — Knowledge Graph Schema & Semantic Contract.

Defines versioned node types, edge types, property contracts, and deterministic ID generators
for transforming classified and linked corpus artifacts into a canonical knowledge graph.
"""

from __future__ import annotations

import enum
import re
import uuid
from typing import Any
from pydantic import BaseModel, ConfigDict, Field

# Schema and Projection Versioning
GRAPH_SCHEMA_VERSION = "1.0.0"
PROJECTION_VERSION = "1.0.0"

# Fixed UUIDv5 namespace for SecondSelf graph entities (deterministic generation)
SECONDSELF_NAMESPACE = uuid.UUID("7a2e8c14-5b8d-4e9f-a63b-1d7e5f4c9a2b")


class NodeType(str, enum.Enum):
    """Categorical classification of entities in the knowledge graph."""

    ITEM = "Item"
    PARA_CATEGORY = "PARA_Category"
    TAG = "Tag"
    SOURCE = "Source"


class EdgeType(str, enum.Enum):
    """Typed semantic relationships between graph entities."""

    BELONGS_TO = "BELONGS_TO"          # Item -> PARA_Category
    SEMANTIC_LINK = "SEMANTIC_LINK"    # Item -> Item
    EXTRACTED_FROM = "EXTRACTED_FROM"  # Item -> Source
    TAGGED_WITH = "TAGGED_WITH"        # Item -> Tag


class PropertySchema(BaseModel):
    """Specification of required and optional properties for a node or edge type."""

    name: str
    required_properties: list[str] = Field(default_factory=list)
    optional_properties: list[str] = Field(default_factory=list)
    description: str = ""

    model_config = ConfigDict(populate_by_name=True)


# ==============================================================================
# Canonical Schema Contracts
# ==============================================================================

NODE_SCHEMAS: dict[NodeType, PropertySchema] = {
    NodeType.ITEM: PropertySchema(
        name="Item",
        required_properties=["record_id", "title", "category"],
        optional_properties=[
            "confidence",
            "summary",
            "tags",
            "source_type",
            "created_at",
            "content_hash",
            "char_count",
            "word_count",
            "needs_review",
            "file_path",
        ],
        description="Core ingested note or document representing an atom of knowledge.",
    ),
    NodeType.PARA_CATEGORY: PropertySchema(
        name="PARA_Category",
        required_properties=["category", "display_name"],
        optional_properties=["target_folder", "description", "color_hex"],
        description="Fixed structural quadrant of the PARA organizational taxonomy.",
    ),
    NodeType.TAG: PropertySchema(
        name="Tag",
        required_properties=["name", "slug"],
        optional_properties=["frequency"],
        description="Topical, thematic, or keyword classification label.",
    ),
    NodeType.SOURCE: PropertySchema(
        name="Source",
        required_properties=["source_type", "display_name"],
        optional_properties=["source_uri", "adapter_name"],
        description="Provenance origin or ingestion adapter channel (e.g. url, pdf, text).",
    ),
}

EDGE_SCHEMAS: dict[EdgeType, PropertySchema] = {
    EdgeType.BELONGS_TO: PropertySchema(
        name="BELONGS_TO",
        required_properties=["category"],
        optional_properties=["confidence", "method", "rationale", "assigned_at"],
        description="Connects an Item to its assigned PARA_Category quadrant.",
    ),
    EdgeType.SEMANTIC_LINK: PropertySchema(
        name="SEMANTIC_LINK",
        required_properties=["score"],
        optional_properties=["similarity_pct", "method", "reason", "target_title"],
        description="Reciprocal or thresholded semantic association between two Items.",
    ),
    EdgeType.EXTRACTED_FROM: PropertySchema(
        name="EXTRACTED_FROM",
        required_properties=["source_type"],
        optional_properties=["source_uri", "char_count", "captured_at"],
        description="Provenance edge linking an Item to its original ingestion Source.",
    ),
    EdgeType.TAGGED_WITH: PropertySchema(
        name="TAGGED_WITH",
        required_properties=["tag_slug"],
        optional_properties=["tag_name"],
        description="Topical tagging association linking an Item to a Tag entity.",
    ),
}


# ==============================================================================
# Deterministic Identifier Generators
# ==============================================================================

def slugify(text: str) -> str:
    """Normalizes string into a clean alphanumeric slug with hyphen delimiters."""
    cleaned = re.sub(r"[^\w\s-]", "", text.strip().lower())
    return re.sub(r"[-\s]+", "-", cleaned).strip("-_")


def make_item_id(record_id: str) -> str:
    """Computes stable, canonical node ID for an Item."""
    clean_id = record_id.strip()
    return f"item:{clean_id}"


def make_para_id(category: str) -> str:
    """Computes stable, canonical node ID for a PARA Category."""
    clean_cat = category.strip().lower()
    return f"para:{clean_cat}"


def make_tag_id(tag_text: str) -> str:
    """Computes stable, canonical node ID for a Tag."""
    slug = slugify(tag_text)
    return f"tag:{slug}"


def make_source_id(source_type: str) -> str:
    """Computes stable, canonical node ID for an ingestion Source."""
    clean_src = source_type.strip().lower()
    return f"source:{clean_src}"


def make_uuid5(canonical_key: str) -> str:
    """Generates deterministic UUIDv5 from a canonical string key."""
    return str(uuid.uuid5(SECONDSELF_NAMESPACE, canonical_key))
