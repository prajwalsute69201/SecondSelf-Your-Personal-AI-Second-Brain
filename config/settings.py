"""SecondSelf — Central Configuration Loader with Fail-Fast Validation.

Implements Twelve-Factor App principles:
- Reads configuration from OS environment variables and environment files (.env).
- Cascades overrides: Defaults -> .env files -> Runtime environment variables.
- Validates types, ranges, and required secrets at application initialization.
- Exposes typed, structured configuration via cached `get_settings()`.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from config.base import (
    AppConfig,
    AskConfig,
    ClassifierConfig,
    ClassifierMethod,
    DEFAULT_BASE_DIR,
    EnvironmentType,
    GroqConfig,
    LogLevel,
    PathsConfig,
    PreviewConfig,
    ServerlessConfig,
    VectorConfig,
)


def resolve_env_files(target_env: str) -> tuple[Path, ...]:
    """Resolves ordered list of .env files based on active environment."""
    candidates: list[Path] = []
    if target_env == "test":
        candidates.append(DEFAULT_BASE_DIR / ".env.test")
    elif target_env == "local":
        candidates.append(DEFAULT_BASE_DIR / ".env.local")
    elif target_env in ("staging", "preview"):
        candidates.append(DEFAULT_BASE_DIR / ".env.staging")
    elif target_env == "production":
        candidates.append(DEFAULT_BASE_DIR / ".env.production")

    candidates.append(DEFAULT_BASE_DIR / ".env")
    return tuple(f for f in candidates if f.exists())


class Settings(BaseSettings):
    """Consolidated application settings model with fail-fast validation."""

    model_config = SettingsConfigDict(
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # 1. Environment & Logging
    environment: EnvironmentType = Field(default="local")
    debug: bool = Field(default=False)
    log_level: LogLevel = Field(default="INFO")

    # 2. Groq LLM
    groq_api_key: str = Field(default="")
    groq_classify_model: str = Field(default="llama-3.3-70b-versatile")
    groq_synthesis_model: str = Field(default="llama-3.3-70b-versatile")
    groq_timeout_seconds: float = Field(default=15.0)
    groq_max_retries: int = Field(default=3)

    # 3. Vector & Graph
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    embedding_dimensions: int = Field(default=384)
    similarity_threshold: float = Field(default=0.65)
    top_k_links: int = Field(default=5)
    rag_min_score: float = Field(default=0.40)
    rag_top_k: int = Field(default=5)

    # 4. Serverless & CORS
    cors_origins: str = Field(default="*")
    serverless_cache_enabled: bool = Field(default=True)

    # 5. Classifier Config
    classifier_method: ClassifierMethod = Field(default="hybrid")
    classifier_rule_weight: float = Field(default=0.40)
    classifier_llm_weight: float = Field(default=0.60)
    classifier_confidence_threshold: float = Field(default=0.60)
    classifier_archive_staleness_days: int = Field(default=180)
    classifier_version: str = Field(default="1.0.0")
    classifier_cache_enabled: bool = Field(default=True)
    classifier_cache_size: int = Field(default=512)
    classifier_default_fallback_category: str = Field(default="Resources")
    classifier_llm_temperature: float = Field(default=0.0)
    classifier_llm_timeout_seconds: float = Field(default=15.0)
    classifier_llm_max_retries: int = Field(default=3)

    # 6. Static Preview & Verification
    preview_host: str = Field(default="127.0.0.1")
    preview_port: int = Field(default=8000)

    # Structured Domain Sub-configs (Initialized after validation)
    app: AppConfig = Field(default_factory=AppConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    groq: GroqConfig = Field(default_factory=GroqConfig)
    vector: VectorConfig = Field(default_factory=VectorConfig)
    serverless: ServerlessConfig = Field(default_factory=ServerlessConfig)
    classifier: ClassifierConfig = Field(default_factory=ClassifierConfig)
    preview: PreviewConfig = Field(default_factory=PreviewConfig)
    ask: AskConfig = Field(default_factory=AskConfig)

    @model_validator(mode="after")
    def validate_and_compose(self) -> Settings:
        """Fail-fast validation for critical secrets and environment constraints."""
        # 1. Validate Secret Presence in Production and Staging
        is_live = self.environment in ("production", "staging")
        is_placeholder = (
            not self.groq_api_key
            or "your_groq_api_key" in self.groq_api_key
            or self.groq_api_key == "gsk_placeholder"
        )

        if is_live and is_placeholder:
            raise ValueError(
                f"FATAL: Missing or invalid GROQ_API_KEY for environment '{self.environment}'. "
                "Production and Staging deployments require a valid Groq API key (starts with 'gsk_'). "
                "Configure this in your Vercel Project Environment Variables."
            )

        # 2. Synchronize App Sub-model
        self.app = AppConfig(
            environment=self.environment,
            debug=self.debug,
            log_level=self.log_level,
        )

        # 3. Synchronize Groq Sub-model
        self.groq = GroqConfig(
            api_key=self.groq_api_key,
            classify_model=self.groq_classify_model,
            synthesis_model=self.groq_synthesis_model,
            timeout_seconds=self.groq_timeout_seconds,
            max_retries=self.groq_max_retries,
        )

        # 4. Synchronize Vector Sub-model
        self.vector = VectorConfig(
            model_name=self.embedding_model,
            dimensions=self.embedding_dimensions,
            similarity_threshold=self.similarity_threshold,
            top_k_links=self.top_k_links,
            rag_min_score=self.rag_min_score,
            rag_top_k=self.rag_top_k,
        )

        # 5. Synchronize Serverless Sub-model
        origins = [o.strip() for o in self.cors_origins.split(",") if o.strip()]
        self.serverless = ServerlessConfig(
            cors_origins=origins if origins else ["*"],
            cache_enabled=self.serverless_cache_enabled,
        )

        # 6. Synchronize Classifier Sub-model
        self.classifier = ClassifierConfig(
            method=self.classifier_method,
            rule_weight=self.classifier_rule_weight,
            llm_weight=self.classifier_llm_weight,
            confidence_threshold=self.classifier_confidence_threshold,
            archive_staleness_days=self.classifier_archive_staleness_days,
            classifier_version=self.classifier_version,
            cache_enabled=self.classifier_cache_enabled,
            cache_size=self.classifier_cache_size,
            default_fallback_category=self.classifier_default_fallback_category,
            llm_temperature=self.classifier_llm_temperature,
            llm_timeout_seconds=self.classifier_llm_timeout_seconds,
            llm_max_retries=self.classifier_llm_max_retries,
        )

        # 7. Ensure Base Paths Exist
        self.paths = PathsConfig()

        # 8. Synchronize Static Preview Config
        self.preview = PreviewConfig(
            host=self.preview_host,
            port=self.preview_port,
        )

        # 9. Synchronize Ask/RAG Config
        self.ask = AskConfig()
        return self


@lru_cache(maxsize=4)
def _load_settings_cached(target_env: str, env_files: tuple[Path, ...]) -> Settings:
    """Internal cached loader for specific target environments and file sets."""
    kwargs: dict[str, Any] = {"environment": target_env}
    if env_files:
        kwargs["_env_file"] = env_files
    return Settings(**kwargs)


def get_settings(env: str | None = None, _reload: bool = False) -> Settings:
    """Returns the singleton validated application settings instance.

    Args:
        env: Optional target environment name override (local, test, staging, production).
        _reload: If True, clears the LRU cache and re-parses settings.

    Returns:
        Structured, validated Settings instance.
    """
    target_env = (
        env
        or os.getenv("ENVIRONMENT")
        or os.getenv("APP_ENV")
        or "local"
    ).strip().lower()

    if _reload:
        _load_settings_cached.cache_clear()

    env_files = resolve_env_files(target_env)
    return _load_settings_cached(target_env, env_files)
