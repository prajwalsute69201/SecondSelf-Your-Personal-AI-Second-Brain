"""SecondSelf — Grounded Answer Synthesis Engine (Task 4.1).

Executes grounded synthesis using Groq LPU LLM inference with strict citation parsing,
hallucination validation, and guaranteed zero-crash retrieval-only fallback.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Sequence

from config import get_settings
from config.base import GroqConfig
from ask.context import ContextBuilder
from ask.models import Answer, Citation, ContextBundle, RetrievedItem
from ask.prompts import (
    NOT_FOUND_TOKEN,
    PROMPT_VERSION,
    SYSTEM_PROMPT_V1,
    build_synthesis_user_prompt,
)

logger = logging.getLogger("secondself.ask.synthesizer")

# Regex to detect citations in answer text:
# Matches [Note: item:...], [item:...], [Note: <uuid>], or [<uuid>]
CITATION_REGEX = re.compile(
    r"\[(?:Note:\s*)?(item:[a-zA-Z0-9_\-]+|[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}|[a-zA-Z0-9_\-]{6,})\]",
    re.IGNORECASE,
)


class Synthesizer:
    """Orchestrates grounded answer generation with citation extraction and fallback."""

    def __init__(
        self,
        groq_client: Any | None = None,
        groq_config: GroqConfig | None = None,
        context_builder: ContextBuilder | None = None,
    ) -> None:
        settings = get_settings()
        self.groq_config = groq_config or settings.groq
        self._groq_client = groq_client
        self.context_builder = context_builder or ContextBuilder()

    def _get_groq_client(self) -> Any | None:
        """Lazily instantiates the Groq API client."""
        if self._groq_client is not None:
            return self._groq_client

        api_key = self.groq_config.api_key.strip()
        if not api_key or "your_groq_api_key" in api_key or api_key == "gsk_placeholder":
            return None

        try:
            from groq import Groq

            self._groq_client = Groq(
                api_key=api_key,
                timeout=self.groq_config.timeout_seconds,
            )
            return self._groq_client
        except Exception as exc:
            logger.warning("Could not initialize Groq client: %s; falling back to retrieval-only mode", exc)
            return None

    def synthesize(
        self,
        question: str,
        retrieved_items: Sequence[RetrievedItem],
        use_llm: bool = True,
        max_context_chars: int | None = None,
    ) -> Answer:
        """Synthesizes an answer from retrieved items or degrades to retrieval-only mode.

        Args:
            question: Natural language query.
            retrieved_items: Ranked list of retrieved corpus items.
            use_llm: Set False to force pure retrieval-only output.
            max_context_chars: Optional override for context character budget.

        Returns:
            Grounded Answer dataclass instance with citations and latency.
        """
        start_time = time.monotonic()

        # Handle empty retrieval immediately
        if not retrieved_items:
            elapsed = (time.monotonic() - start_time) * 1000
            return Answer(
                question=question,
                text=NOT_FOUND_TOKEN,
                citations=[],
                retrieved=[],
                mode="retrieval_only",
                model=None,
                prompt_version=PROMPT_VERSION,
                latency_ms=elapsed,
                is_grounded=True,
                uncited_claims_flag=False,
            )

        # Assemble bounded context
        context_bundle: ContextBundle = self.context_builder.build(retrieved_items, max_chars=max_context_chars)

        client = self._get_groq_client() if use_llm else None

        # If LLM disabled or unconfigured, execute retrieval-only mode
        if not use_llm or client is None:
            return self._build_retrieval_only_answer(question, context_bundle.items, start_time)

        # Execute LLM generation with retry backoff
        user_prompt = build_synthesis_user_prompt(question, context_bundle.formatted_context)
        model_name = self.groq_config.synthesis_model

        for attempt in range(self.groq_config.max_retries):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT_V1},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,  # Zero temperature for deterministic grounded answers
                    max_tokens=1024,
                )

                answer_text = response.choices[0].message.content.strip()

                # Extract and validate citations
                citations, is_grounded, uncited_flag = self._extract_and_validate_citations(
                    answer_text, context_bundle.items
                )

                elapsed = (time.monotonic() - start_time) * 1000
                return Answer(
                    question=question,
                    text=answer_text,
                    citations=citations,
                    retrieved=list(retrieved_items),
                    mode="llm_synthesis",
                    model=model_name,
                    prompt_version=PROMPT_VERSION,
                    latency_ms=elapsed,
                    is_grounded=is_grounded,
                    uncited_claims_flag=uncited_flag,
                )
            except Exception as exc:
                wait_time = (2 ** attempt) * 0.5
                logger.warning(
                    "LLM synthesis attempt %d failed: %s. Retrying in %.1fs...",
                    attempt + 1,
                    exc,
                    wait_time,
                )
                time.sleep(wait_time)

        # Graceful fallback on permanent LLM failure
        logger.error("LLM synthesis failed after %d retries; falling back to retrieval-only answer", self.groq_config.max_retries)
        return self._build_retrieval_only_answer(question, context_bundle.items, start_time)

    def _extract_and_validate_citations(
        self,
        answer_text: str,
        retrieved_items: Sequence[RetrievedItem],
    ) -> tuple[list[Citation], bool, bool]:
        """Extracts cited item IDs, validates they exist in retrieved items, and checks groundedness."""
        raw_matches = CITATION_REGEX.findall(answer_text)

        # Map canonical ID and short UUID to item
        item_lookup: dict[str, RetrievedItem] = {}
        for it in retrieved_items:
            item_lookup[it.id.lower()] = it
            bare_id = it.id.replace("item:", "").lower()
            item_lookup[bare_id] = it

        valid_citations: list[Citation] = []
        seen_ids: set[str] = set()
        is_grounded = True

        for match in raw_matches:
            clean_match = match.strip().lower()
            target_item = item_lookup.get(clean_match)

            if target_item is not None:
                if target_item.id not in seen_ids:
                    seen_ids.add(target_item.id)
                    valid_citations.append(
                        Citation(
                            item_id=target_item.id,
                            title=target_item.title,
                            snippet=target_item.snippet or target_item.content[:200],
                            score=target_item.score,
                            source=target_item.file_path or target_item.title,
                        )
                    )
            else:
                # LLM cited an ID not in the retrieved context
                logger.warning("Unverified/hallucinated citation detected in answer: '%s'", match)
                is_grounded = False

        # Check for uncited claims: if answer does not say "not found", but has 0 citations
        is_not_found = NOT_FOUND_TOKEN.lower() in answer_text.lower()
        uncited_claims_flag = (len(valid_citations) == 0) and (not is_not_found) and (len(retrieved_items) > 0)

        return valid_citations, is_grounded, uncited_claims_flag

    def _build_retrieval_only_answer(
        self,
        question: str,
        retrieved_items: Sequence[RetrievedItem],
        start_time: float,
    ) -> Answer:
        """Constructs an explainable, cited answer from top matches when LLM is unavailable."""
        elapsed = (time.monotonic() - start_time) * 1000

        if not retrieved_items:
            return Answer(
                question=question,
                text=NOT_FOUND_TOKEN,
                citations=[],
                retrieved=[],
                mode="retrieval_only",
                model=None,
                prompt_version=PROMPT_VERSION,
                latency_ms=elapsed,
                is_grounded=True,
                uncited_claims_flag=False,
            )

        # Build clean bulleted response
        lines = [
            f"Here are the most relevant notes found in your SecondSelf corpus for: \"{question}\":\n"
        ]
        citations: list[Citation] = []

        for idx, item in enumerate(retrieved_items[:5], 1):
            snippet = item.snippet or item.content[:200]
            clean_snippet = snippet.replace("\n", " ").strip()
            lines.append(
                f"- **{item.title}** ([{item.category}]) [Note: {item.id}]:\n"
                f"  \"{clean_snippet}\""
            )
            citations.append(
                Citation(
                    item_id=item.id,
                    title=item.title,
                    snippet=clean_snippet,
                    score=item.score,
                    source=item.file_path or item.title,
                )
            )

        text = "\n".join(lines)
        return Answer(
            question=question,
            text=text,
            citations=citations,
            retrieved=list(retrieved_items),
            mode="retrieval_only",
            model=None,
            prompt_version=PROMPT_VERSION,
            latency_ms=elapsed,
            is_grounded=True,
            uncited_claims_flag=False,
        )
