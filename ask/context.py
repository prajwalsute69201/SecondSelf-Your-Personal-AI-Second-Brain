"""SecondSelf — Bounded Context Assembly Engine (Task 4.1).

Assembles retrieved corpus items into a deterministic, bounded text block
with explicit ID delimiters and strict character/token budget enforcement.
"""

from __future__ import annotations

import logging
from typing import Sequence

from config import get_settings
from ask.models import ContextBundle, RetrievedItem

logger = logging.getLogger("secondself.ask.context")


class ContextBuilder:
    """Assembles and bounds textual context for LLM prompt generation."""

    def __init__(self, max_chars: int | None = None) -> None:
        self.max_chars = max_chars or get_settings().ask.max_context_chars

    def build(
        self,
        items: Sequence[RetrievedItem],
        max_chars: int | None = None,
    ) -> ContextBundle:
        """Constructs bounded context block from ranked retrieved items.

        Args:
            items: Ranked list of retrieved corpus items.
            max_chars: Optional override for maximum character budget.

        Returns:
            ContextBundle containing formatted text, included items, and truncation metrics.
        """
        budget = max_chars or self.max_chars
        if not items:
            return ContextBundle(
                formatted_context="(No relevant items found in SecondSelf corpus)",
                items=[],
                total_chars=0,
                estimated_tokens=0,
                dropped_items_count=0,
            )

        formatted_blocks: list[str] = []
        included_items: list[RetrievedItem] = []
        current_chars = 0
        dropped_count = 0

        for idx, item in enumerate(items):
            content_text = item.content.strip() or item.snippet.strip() or item.title

            block = (
                f"--- START SOURCE ITEM [{item.id}] ---\n"
                f"Item ID: {item.id}\n"
                f"Title: {item.title}\n"
                f"Category: {item.category}\n"
                f"Relevance Score: {item.score:.3f}\n"
                f"Content:\n{content_text}\n"
                f"--- END SOURCE ITEM [{item.id}] ---"
            )

            block_len = len(block) + 2  # account for double newline separator

            if current_chars + block_len <= budget:
                formatted_blocks.append(block)
                included_items.append(item)
                current_chars += block_len
            else:
                # If first item exceeds budget, cleanly truncate it
                if not formatted_blocks:
                    available = budget - 200
                    if available > 100:
                        truncated_content = content_text[:available] + "... [truncated]"
                        truncated_block = (
                            f"--- START SOURCE ITEM [{item.id}] ---\n"
                            f"Item ID: {item.id}\n"
                            f"Title: {item.title}\n"
                            f"Category: {item.category}\n"
                            f"Relevance Score: {item.score:.3f}\n"
                            f"Content:\n{truncated_content}\n"
                            f"--- END SOURCE ITEM [{item.id}] ---"
                        )
                        formatted_blocks.append(truncated_block)
                        included_items.append(item)
                        current_chars += len(truncated_block)
                    dropped_count += len(items) - 1
                    break
                else:
                    dropped_count += 1

        final_context = "\n\n".join(formatted_blocks)
        estimated_tokens = max(1, len(final_context) // 4)

        logger.debug(
            "Assembled context: %d items included, %d dropped, %d chars (~%d tokens)",
            len(included_items),
            dropped_count,
            len(final_context),
            estimated_tokens,
        )

        return ContextBundle(
            formatted_context=final_context,
            items=included_items,
            total_chars=len(final_context),
            estimated_tokens=estimated_tokens,
            dropped_items_count=dropped_count,
        )
