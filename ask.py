#!/usr/bin/env python3
"""SecondSelf — Question Answering CLI & Library Entrypoint (Task 4.1).

Answers natural-language questions over the SecondSelf personal knowledge corpus:
- Retrieves semantically closest notes via dense vector similarity.
- Optionally expands 1st-degree neighbors from the knowledge graph.
- Binds context and synthesizes grounded answers with strict source citations.
- Gracefully degrades to an explainable retrieval-only summary when LLM is unavailable.

Usage (CLI):
  python ask.py "What are the core ideas of SecondSelf?"
  python ask.py "How does vector linking work?" --no-llm
  python ask.py "Explain force layout" --expand-graph --json

Usage (Python):
  from ask import ask
  answer = ask("How does PARA classification work?")
  print(answer.text)
  for citation in answer.citations:
      print(f"- {citation.title} ({citation.item_id})")
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure workspace root is in python path
WORKSPACE_ROOT = Path(__file__).resolve().parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from ask.cli import ask, main
from ask.models import Answer, Citation, ContextBundle, RetrievedItem

__all__ = [
    "Answer",
    "Citation",
    "ContextBundle",
    "RetrievedItem",
    "ask",
    "main",
]

if __name__ == "__main__":
    sys.exit(main())
