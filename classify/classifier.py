"""SecondSelf — Autonomous PARA Classification Orchestrator.

Combines deterministic heuristic signal scoring with Groq LLM-assisted inference
into an auditable, deterministic hybrid classification pipeline with graceful degradation.
"""

from __future__ import annotations

import collections
import hashlib
import json
import logging
import time
from typing import Any

from adapters.models import ExtractedContent
from config import get_settings
from config.base import ClassifierConfig
from storage.models import RecordMetadata
from classify.llm_prompt import (
    PROMPT_VERSION,
    SYSTEM_PROMPT_V1,
    LLMValidationError,
    build_user_prompt,
    parse_and_validate_llm_response,
)
from classify.models import PARAClassification, Signal
from classify.rules import extract_signals, score_rules
from classify.taxonomy import PARACategory, get_criteria

logger = logging.getLogger("secondself.classify")


class PARAClassifier:
    """Autonomous classifier mapping arbitrary content into the fixed PARA taxonomy.

    Features:
    - Pure heuristic rules engine with zero network dependencies.
    - LLM-assisted classification with strict JSON schema enforcement (Groq LPU).
    - Hybrid resolution fusing rule and LLM probabilities with configurable weights.
    - Deterministic caching by SHA-256 content hash.
    - Graceful fallback to heuristic rules upon LLM timeouts, rate limits, or errors.
    - Full explainability and auditable rationale per decision.
    """

    def __init__(
        self,
        config: ClassifierConfig | None = None,
        groq_client: Any | None = None,
    ) -> None:
        """Initializes classifier with configuration and optional injected Groq client.

        Args:
            config: Optional ClassifierConfig instance. Defaults to app settings.
            groq_client: Optional Groq client (allows mocking in tests).
        """
        self.config = config or get_settings().classifier
        self._groq_client = groq_client
        self._cache: collections.OrderedDict[str, PARAClassification] = collections.OrderedDict()

    def _get_groq_client(self) -> Any | None:
        """Lazily initializes and caches the Groq API client."""
        if self._groq_client is not None:
            return self._groq_client

        groq_cfg = get_settings().groq
        api_key = groq_cfg.api_key.strip()

        # In local or test environments without an active Groq key, return None
        if not api_key or "your_groq_api_key" in api_key or api_key == "gsk_placeholder":
            return None

        try:
            from groq import Groq

            self._groq_client = Groq(
                api_key=api_key,
                timeout=self.config.llm_timeout_seconds,
            )
            return self._groq_client
        except Exception as exc:
            logger.warning("Failed to initialize Groq client: %s. Using heuristic fallback.", exc)
            return None

    def _compute_hash(self, text: str) -> str:
        """Computes deterministic SHA-256 digest of normalized text."""
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

    def _derive_fallback_title_and_summary(
        self, content: str, metadata: dict[str, Any]
    ) -> tuple[str, str, list[str]]:
        """Derives a title, summary, and tags heuristically when LLM is unavailable."""
        # Check metadata first
        title = metadata.get("title")
        lines = [line.strip() for line in content.splitlines() if line.strip()]

        if not title or title.lower() in ("untitled", "n/a"):
            for line in lines:
                if line.startswith("# "):
                    title = line.lstrip("# ").strip()
                    break
            if not title and lines:
                words = lines[0].split()
                title = " ".join(words[:8])
            if not title:
                title = "Untitled Capture"

        # Summary heuristic: first informative paragraph
        summary = ""
        for line in lines:
            if not line.startswith("#") and len(line) > 20:
                summary = line[:200]
                break
        if not summary and lines:
            summary = lines[0][:200]

        # Basic tags heuristic
        tags = [str(metadata.get("source_type", "capture")).lower()]
        return title, summary, tags

    def classify_rules_only(
        self,
        content: str,
        metadata: dict[str, Any],
        content_hash: str,
    ) -> PARAClassification:
        """Executes purely deterministic, rule-based heuristic classification."""
        signals = extract_signals(
            content=content,
            metadata=metadata,
            archive_staleness_days=self.config.archive_staleness_days,
        )
        winner, confidence, scores, needs_review, rationale = score_rules(
            signals=signals,
            confidence_threshold=self.config.confidence_threshold,
            default_fallback=PARACategory.from_string(self.config.default_fallback_category),
        )

        title, summary, tags = self._derive_fallback_title_and_summary(content, metadata)

        return PARAClassification(
            category=winner,
            confidence=confidence,
            rationale=rationale,
            signals=signals,
            category_scores=scores,
            needs_review=needs_review,
            method="rules",
            classifier_version=self.config.classifier_version,
            content_hash=content_hash,
            title=title,
            tags=tags,
            summary=summary,
            suggested_folder=get_criteria(winner).target_folder,
            metadata=metadata,
        )

    def _call_llm(
        self,
        content: str,
        metadata: dict[str, Any],
    ) -> tuple[PARACategory, float, str, list[str], str, list[str], str]:
        """Invokes Groq LLM API with exponential backoff retries and strict JSON validation.

        Returns:
            Tuple of (category, confidence, rationale, signals, title, tags, summary).
        """
        client = self._get_groq_client()
        if client is None:
            raise RuntimeError("Groq client is unavailable or GROQ_API_KEY is not configured.")

        groq_cfg = get_settings().groq
        model = groq_cfg.classify_model or "llama-3.3-70b-versatile"
        prompt = build_user_prompt(content=content, metadata=metadata)

        last_error: Exception | None = None
        for attempt in range(self.config.llm_max_retries):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT_V1},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=self.config.llm_temperature,
                    seed=42,
                )
                raw_text = response.choices[0].message.content or ""
                validated = parse_and_validate_llm_response(raw_text)

                return (
                    validated.category,
                    validated.confidence,
                    validated.rationale,
                    validated.signals,
                    validated.title,
                    validated.tags,
                    validated.summary,
                )
            except (LLMValidationError, Exception) as exc:
                last_error = exc
                wait_seconds = (2 ** attempt) * 0.5
                logger.warning(
                    "LLM classification attempt %d failed: %s. Retrying in %.1fs...",
                    attempt + 1,
                    exc,
                    wait_seconds,
                )
                time.sleep(wait_seconds)

        raise RuntimeError(f"LLM classification failed after {self.config.llm_max_retries} retries: {last_error}")

    def classify(
        self,
        content: str | ExtractedContent,
        metadata: dict[str, Any] | RecordMetadata | None = None,
    ) -> PARAClassification:
        """Classifies arbitrary content into the PARA taxonomy.

        Args:
            content: Raw string or normalized ExtractedContent model.
            metadata: Optional diagnostic or storage metadata dictionary.

        Returns:
            Deterministic, auditable PARAClassification dataclass instance.
        """
        # Normalize content and metadata
        if isinstance(content, ExtractedContent):
            raw_text = content.text
            meta_dict = dict(content.metadata)
            meta_dict.setdefault("title", content.title)
            meta_dict.setdefault("source_type", content.source_type)
            meta_dict.setdefault("source_uri", content.source_uri)
            content_hash = content.content_hash or self._compute_hash(raw_text)
        else:
            raw_text = str(content)
            content_hash = self._compute_hash(raw_text)
            if isinstance(metadata, RecordMetadata):
                meta_dict = metadata.model_dump()
            elif isinstance(metadata, dict):
                meta_dict = dict(metadata)
            else:
                meta_dict = {}

        # 1. Deterministic In-Memory Cache Lookup
        if self.config.cache_enabled and content_hash in self._cache:
            cached_result = self._cache[content_hash]
            # Move to end for LRU behavior
            self._cache.move_to_end(content_hash)
            return cached_result

        method = self.config.method

        # 2. Heuristic Rules-Only Execution
        if method == "rules":
            result = self.classify_rules_only(raw_text, meta_dict, content_hash)
            self._cache_result(content_hash, result)
            return result

        # 3. LLM-Assisted & Hybrid Execution
        rule_signals = extract_signals(
            content=raw_text,
            metadata=meta_dict,
            archive_staleness_days=self.config.archive_staleness_days,
        )
        (
            rule_winner,
            rule_conf,
            rule_scores,
            rule_needs_review,
            rule_rationale,
        ) = score_rules(
            signals=rule_signals,
            confidence_threshold=self.config.confidence_threshold,
            default_fallback=PARACategory.from_string(self.config.default_fallback_category),
        )

        try:
            (
                llm_winner,
                llm_conf,
                llm_rationale,
                llm_signal_texts,
                llm_title,
                llm_tags,
                llm_summary,
            ) = self._call_llm(raw_text, meta_dict)

            # Convert LLM signal strings to typed Signal objects
            combined_signals: list[Signal] = list(rule_signals)
            for sig_text in llm_signal_texts:
                combined_signals.append(
                    Signal(
                        name="llm_signal_marker",
                        category=llm_winner,
                        weight=round(llm_conf, 2),
                        matched_text=sig_text[:80],
                        source="llm",
                        description=f"LLM evidence: '{sig_text}'",
                    )
                )

            if method == "llm":
                # Pure LLM resolution
                needs_review = llm_conf < self.config.confidence_threshold
                result = PARAClassification(
                    category=llm_winner,
                    confidence=llm_conf,
                    rationale=f"[LLM Resolution] {llm_rationale}",
                    signals=combined_signals,
                    category_scores={llm_winner.value: llm_conf},
                    needs_review=needs_review,
                    method="llm",
                    classifier_version=self.config.classifier_version,
                    content_hash=content_hash,
                    title=llm_title,
                    tags=llm_tags,
                    summary=llm_summary,
                    suggested_folder=get_criteria(llm_winner).target_folder,
                    metadata=meta_dict,
                )
            else:
                # Hybrid Resolution: Combine Rule Distribution with LLM Distribution
                # LLM probability distribution: winner gets llm_conf, remainder split evenly
                rem = (1.0 - llm_conf) / 3.0
                llm_dist = {
                    cat: (llm_conf if cat == llm_winner else rem)
                    for cat in PARACategory
                }

                w_rule = self.config.rule_weight
                w_llm = self.config.llm_weight

                # Weighted linear combination
                combined_scores: dict[str, float] = {}
                for cat in PARACategory:
                    score = (w_rule * rule_scores.get(cat.value, 0.25)) + (w_llm * llm_dist[cat])
                    combined_scores[cat.value] = score

                # Select winning category deterministically
                winner = max(
                    PARACategory,
                    key=lambda c: (combined_scores[c.value], -list(PARACategory).index(c)),
                )
                final_conf = combined_scores[winner.value]
                needs_review = final_conf < self.config.confidence_threshold

                hybrid_rationale = (
                    f"[Hybrid Decision] Classified as '{winner.value}' with {final_conf:.2f} score "
                    f"(Rule Weight: {w_rule:.2f}, LLM Weight: {w_llm:.2f}). "
                    f"Rules supported '{rule_winner.value}' ({rule_conf:.2f}); "
                    f"LLM supported '{llm_winner.value}' ({llm_conf:.2f}). "
                    f"LLM Rationale: {llm_rationale}"
                )

                result = PARAClassification(
                    category=winner,
                    confidence=final_conf,
                    rationale=hybrid_rationale,
                    signals=combined_signals,
                    category_scores=combined_scores,
                    needs_review=needs_review,
                    method="hybrid",
                    classifier_version=self.config.classifier_version,
                    content_hash=content_hash,
                    title=llm_title,
                    tags=llm_tags,
                    summary=llm_summary,
                    suggested_folder=get_criteria(winner).target_folder,
                    metadata=meta_dict,
                )

        except Exception as exc:
            # Graceful Degradation: Fallback to Heuristic Rules on LLM / Network Failure
            logger.warning(
                "LLM inference encountered an error (%s); gracefully degrading to heuristic rules.",
                exc,
            )
            fallback_res = self.classify_rules_only(raw_text, meta_dict, content_hash)
            result = PARAClassification(
                category=fallback_res.category,
                confidence=fallback_res.confidence,
                rationale=f"[Rules Fallback — LLM Degraded] {fallback_res.rationale}",
                signals=fallback_res.signals,
                category_scores=fallback_res.category_scores,
                needs_review=fallback_res.needs_review,
                method="rules_fallback",
                classifier_version=self.config.classifier_version,
                content_hash=content_hash,
                title=fallback_res.title,
                tags=fallback_res.tags,
                summary=fallback_res.summary,
                suggested_folder=fallback_res.suggested_folder,
                metadata=meta_dict,
            )

        self._cache_result(content_hash, result)
        return result

    def _cache_result(self, content_hash: str, result: PARAClassification) -> None:
        """Stores result in LRU in-memory cache if caching is enabled."""
        if not self.config.cache_enabled:
            return
        if len(self._cache) >= self.config.cache_size:
            self._cache.popitem(last=False)  # Evict oldest entry
        self._cache[content_hash] = result


# ------------------------------------------------------------------------------
# Module-Level Convenience Function
# ------------------------------------------------------------------------------

_GLOBAL_CLASSIFIER: PARAClassifier | None = None


def classify(
    content: str | ExtractedContent,
    metadata: dict[str, Any] | RecordMetadata | None = None,
    config: ClassifierConfig | None = None,
) -> PARAClassification:
    """Classifies content into the PARA taxonomy using the configured classifier.

    Args:
        content: Clean text string or normalized ExtractedContent instance.
        metadata: Optional metadata dictionary or RecordMetadata sidecar.
        config: Optional ClassifierConfig override.

    Returns:
        Structured, explainable PARAClassification result.
    """
    global _GLOBAL_CLASSIFIER
    if config is not None:
        return PARAClassifier(config=config).classify(content=content, metadata=metadata)

    if _GLOBAL_CLASSIFIER is None:
        _GLOBAL_CLASSIFIER = PARAClassifier()

    return _GLOBAL_CLASSIFIER.classify(content=content, metadata=metadata)
