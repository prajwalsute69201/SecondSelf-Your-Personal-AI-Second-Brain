"""SecondSelf — Typed Domain Configuration Schemas.

Defines modular Pydantic models for individual configuration concerns:
- App & runtime execution environment
- File layout & storage directories
- Groq LLM API connectivity
- Vector embeddings & graph similarity math
- Serverless API & CORS settings
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field, field_validator


# Compute project root (parent of config directory)
DEFAULT_BASE_DIR = Path(__file__).resolve().parent.parent

EnvironmentType = Literal["local", "test", "staging", "production"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class AppConfig(BaseModel):
    """General application metadata and runtime mode."""
    name: str = Field(default="SecondSelf", description="Application identifier")
    environment: EnvironmentType = Field(
        default="local",
        description="Deployment target environment: local, test, staging, or production",
    )
    debug: bool = Field(default=False, description="Enable verbose diagnostic logs and tracing")
    log_level: LogLevel = Field(default="INFO", description="Standard logging threshold")
    version: str = Field(default="0.1.0", description="SemVer application version")

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def is_local(self) -> bool:
        return self.environment == "local"

    @property
    def is_test(self) -> bool:
        return self.environment == "test"


class PathsConfig(BaseModel):
    """Filesystem directory and path resolution."""
    base_dir: Path = Field(default=DEFAULT_BASE_DIR, description="Absolute project root path")
    raw_dir: Path = Field(default=DEFAULT_BASE_DIR / "raw", description="Raw capture JSON storage")
    wiki_dir: Path = Field(default=DEFAULT_BASE_DIR / "wiki", description="Curated markdown vault")
    public_dir: Path = Field(default=DEFAULT_BASE_DIR / "public", description="Static web assets")
    index_file: Path = Field(
        default=DEFAULT_BASE_DIR / "public" / "index.json",
        description="Precomputed vector embeddings index",
    )
    graph_file: Path = Field(
        default=DEFAULT_BASE_DIR / "public" / "graph.json",
        description="Precomputed force-directed graph model",
    )


class GroqConfig(BaseModel):
    """Groq LPU LLM inference client configuration."""
    api_key: str = Field(
        default="",
        description="Groq API key (starts with 'gsk_')",
    )
    classify_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Model used for autonomous PARA categorization",
    )
    synthesis_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Model used for conversational RAG question answering",
    )
    timeout_seconds: float = Field(
        default=15.0,
        ge=1.0,
        le=60.0,
        description="HTTP request timeout ceiling for Groq client in seconds",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum exponential backoff retries for rate-limited requests",
    )

    @field_validator("api_key")
    @classmethod
    def validate_api_key_format(cls, v: str) -> str:
        v = v.strip()
        # Allow empty or dummy keys only in local/test; validated at Settings level
        if v and not v.startswith("gsk_") and not v.startswith("mock_"):
            raise ValueError("Groq API key must start with prefix 'gsk_' (or 'mock_' in tests).")
        return v


class VectorConfig(BaseModel):
    """Vector embeddings, cosine similarity thresholds, and RAG retrieval parameters."""
    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace model ID for dense semantic vectorization",
    )
    dimensions: int = Field(
        default=384,
        description="Embedding vector dimensionality",
    )
    similarity_threshold: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity required to inject automatic [[wikilinks]]",
    )
    top_k_links: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum reciprocal links injected per markdown note",
    )
    rag_min_score: float = Field(
        default=0.40,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for RAG context inclusion",
    )
    rag_top_k: int = Field(
        default=5,
        ge=1,
        le=15,
        description="Number of candidate notes passed into Groq context window",
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        le=256,
        description="Batch size for vectorizer embedding generation",
    )
    cache_enabled: bool = Field(
        default=True,
        description="Enable content-hash caching to avoid re-embedding unchanged notes",
    )
    centroid_linking_enabled: bool = Field(
        default=True,
        description="Enable per-PARA category centroid calculation and secondary alignment linking",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum exponential backoff retries on transient embedding provider failure",
    )
    timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=120.0,
        description="Timeout in seconds for remote embedding API calls",
    )


class ServerlessConfig(BaseModel):
    """Vercel Serverless Function (api/ask.py) runtime options."""
    cors_origins: list[str] = Field(
        default_factory=lambda: ["*"],
        description="Permitted Cross-Origin Resource Sharing (CORS) origins",
    )
    cache_enabled: bool = Field(
        default=True,
        description="Enable in-memory index caching across warm serverless invocations",
    )


ClassifierMethod = Literal["rules", "llm", "hybrid"]


class ClassifierConfig(BaseModel):
    """Autonomous PARA classification heuristics, scoring weights, and LLM hyperparameters."""

    method: ClassifierMethod = Field(
        default="hybrid",
        description="Autonomous classification strategy: 'rules', 'llm', or 'hybrid'",
    )
    rule_weight: float = Field(
        default=0.40,
        ge=0.0,
        le=1.0,
        description="Weight assigned to heuristic rule score in hybrid decision mode",
    )
    llm_weight: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="Weight assigned to LLM confidence score in hybrid decision mode",
    )
    confidence_threshold: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold (0.0-1.0); below this, item is flagged needs_review",
    )
    archive_staleness_days: int = Field(
        default=180,
        ge=1,
        description="Inactivity age threshold in days for qualifying completed/stale notes as Archive",
    )
    classifier_version: str = Field(
        default="1.0.0",
        description="Semantic version of the classification engine and prompts",
    )
    cache_enabled: bool = Field(
        default=True,
        description="Enable deterministic in-memory result caching by content_hash",
    )
    cache_size: int = Field(
        default=512,
        ge=16,
        description="Maximum entries for the in-memory classification cache",
    )
    default_fallback_category: str = Field(
        default="Resources",
        description="Default safe category assignment when zero definitive signals are found",
    )
    llm_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for LLM classification (0.0 for strict determinism)",
    )
    llm_timeout_seconds: float = Field(
        default=15.0,
        ge=1.0,
        le=60.0,
        description="HTTP request timeout for LLM classification calls in seconds",
    )
    llm_max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts for transient LLM/network errors",
    )


class PreviewConfig(BaseModel):
    """Static preview server and frontend verification configuration (Task 3.3)."""
    host: str = Field(
        default="127.0.0.1",
        description="Hostname or IP interface for static preview server",
    )
    port: int = Field(
        default=8000,
        ge=1024,
        le=65535,
        description="TCP port for local static preview server",
    )
    readiness_timeout_seconds: float = Field(
        default=5.0,
        ge=0.5,
        le=30.0,
        description="Maximum timeout ceiling in seconds for server readiness polling",
    )
    poll_interval_seconds: float = Field(
        default=0.1,
        ge=0.01,
        le=1.0,
        description="Interval in seconds between server readiness polling attempts",
    )
    artifacts_dir: Path = Field(
        default=DEFAULT_BASE_DIR / "verify-artifacts",
        description="Target directory for verification reports, logs, and screenshots",
    )
    report_filename: str = Field(
        default="verify-report.json",
        description="Default machine-readable verification report filename",
    )
    staging_enabled: bool = Field(
        default=True,
        description="Ensure safe read-only staging copy of artifacts prior to serving",
    )


class AskConfig(BaseModel):
    """Configuration for retrieval-augmented generation & question answering (Task 4.1)."""
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Default number of top items to retrieve for synthesis",
    )
    min_score: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity threshold for retrieved items",
    )
    max_context_chars: int = Field(
        default=8000,
        ge=500,
        le=32000,
        description="Maximum character budget for assembled LLM context",
    )
    expand_graph: bool = Field(
        default=False,
        description="Enable hybrid 1st-degree graph neighbor expansion via build_graph output",
    )
    graph_discount_factor: float = Field(
        default=0.85,
        ge=0.1,
        le=1.0,
        description="Discount multiplier applied to graph-expanded neighbor similarity scores",
    )
    prompt_version: str = Field(
        default="1.0.0",
        description="Semantic version of RAG prompt templates",
    )



