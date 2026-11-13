"""Question answering engine for the SecondSelf personal knowledge base (Task 4.1.1).

Retrieves top-K relevant wiki notes using semantic embedding cosine similarity,
assembles a grounded RAG prompt adhering to strict context size limits (~6000 tokens),
invokes the Groq LLM (llama-3.1-8b-instant, temperature=0.3) via `lib/llm.py`,
and returns an `AskResult` with source citations.

Usage:
    python ask.py "What are my career goals?"
    python ask.py "What ML resources have I saved?"
    python ask.py "Summarize my active projects"
    python ask.py --top-k 3 --json "What books did I read?"
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import logging
from pathlib import Path
import re
import sys
from typing import Any, Sequence

# Ensure repository root is on sys.path so 'lib' is importable
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import lib.embeddings as emb
from lib.llm import DEFAULT_MODEL, RAG_PROMPT_TEMPLATE, build_rag_prompt, synthesize_answer
from lib.models import AskResult, WikiNote
import lib.storage as storage

logger = logging.getLogger(__name__)

# RAG Guardrail Constants (Task 4.1.3)
DEFAULT_TOP_K: int = 5
DEFAULT_TEMPERATURE: float = 0.3
DEFAULT_SIMILARITY_THRESHOLD: float = 0.25
MAX_CONTEXT_TOKENS: int = 6000
CHARS_PER_TOKEN: int = 4
MAX_CONTEXT_CHARS: int = MAX_CONTEXT_TOKENS * CHARS_PER_TOKEN  # ~24,000 characters
MAX_SINGLE_NOTE_CHARS: int = 12000
MAX_QUESTION_CHARS: int = 2000

NO_NOTES_ANSWER: str = "I don't have notes about that."
EMPTY_QUESTION_ANSWER: str = "Please provide a valid question."

__all__ = [
    "DEFAULT_TOP_K",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_SIMILARITY_THRESHOLD",
    "MAX_CONTEXT_TOKENS",
    "MAX_CONTEXT_CHARS",
    "MAX_SINGLE_NOTE_CHARS",
    "MAX_QUESTION_CHARS",
    "NO_NOTES_ANSWER",
    "EMPTY_QUESTION_ANSWER",
    "RAG_PROMPT_TEMPLATE",
    "build_rag_prompt",
    "format_note_context",
    "build_rag_context",
    "retrieve_relevant_notes",
    "validate_citations",
    "ask",
    "build_parser",
    "main",
]


def format_note_context(note: WikiNote, max_chars: int = MAX_SINGLE_NOTE_CHARS) -> str:
    """Format a single WikiNote into a context block with distinct boundaries for RAG.

    Includes note ID, title/summary, PARA category, topic tags, and body content.
    Handles missing optional metadata safely and isolates content to prevent cross-note
    confusion or prompt injection.

    Args:
        note: WikiNote instance to format.
        max_chars: Maximum character limit for the note body.

    Returns:
        str: Formatted context block with metadata and bounded content.
    """
    body = (note.body or "").strip()
    if not body:
        body = "(Empty content)"
    elif len(body) > max_chars:
        body = body[:max_chars].rstrip() + "\n...[content truncated]..."

    note_id = note.id if note.id and str(note.id).strip() else "unknown-id"
    para = note.para if note.para and str(note.para).strip() else "Uncategorized"
    summary = note.summary if note.summary and str(note.summary).strip() else "(Untitled)"

    if note.tags and isinstance(note.tags, (list, tuple)):
        clean_tags = [str(t).strip() for t in note.tags if str(t).strip()]
        tags_str = ", ".join(clean_tags) if clean_tags else "none"
    elif isinstance(note.tags, str) and note.tags.strip():
        tags_str = note.tags.strip()
    else:
        tags_str = "none"

    return (
        f"--- Note [{note_id}] ---\n"
        f"PARA: {para}\n"
        f"Tags: {tags_str}\n"
        f"Summary: {summary}\n"
        f"Content:\n"
        f"{body}\n"
        f"--- End Note [{note_id}] ---"
    )


def build_rag_context(
    retrieved_notes: list[tuple[WikiNote, float]],
    max_total_chars: int = MAX_CONTEXT_CHARS,
) -> str:
    """Assemble formatted note blocks into a combined RAG context string under token limit.

    Args:
        retrieved_notes: List of (WikiNote, similarity_score) tuples, sorted by score descending.
        max_total_chars: Maximum allowed characters for the combined context block.

    Returns:
        str: Joined note context blocks within the budget.
    """
    if not retrieved_notes:
        return ""

    blocks: list[str] = []
    current_chars = 0

    for note, _score in retrieved_notes:
        block = format_note_context(note)
        block_len = len(block) + 2  # account for delimiter newlines

        if current_chars + block_len <= max_total_chars:
            blocks.append(block)
            current_chars += block_len
        else:
            remaining_chars = max_total_chars - current_chars
            # If at least 300 characters remain, provide a truncated version of this note
            if remaining_chars > 300:
                truncated_block = format_note_context(note, max_chars=remaining_chars - 150)
                blocks.append(truncated_block)
            break

    return "\n\n".join(blocks)


def retrieve_relevant_notes(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    threshold: float | None = DEFAULT_SIMILARITY_THRESHOLD,
    root: Path | str | None = None,
    model: Any | None = None,
    user_id: str | None = None,
) -> list[tuple[WikiNote, float]]:
    """Retrieve top-K most semantically relevant WikiNotes for a question scoped to user_id.

    Loads embeddings and full note bodies scoped to user_id,
    computes cosine similarity between the embedded query and each note vector,
    filters by similarity threshold, and returns the top-K matches sorted descending.

    Args:
        question: User query string.
        top_k: Maximum number of notes to retrieve.
        threshold: Minimum similarity score threshold. If None, no score filtering is applied.
        root: Optional project root directory override.
        model: Optional pre-loaded sentence-transformers model instance.
        user_id: Optional user ID to isolate notes search to.

    Returns:
        list[tuple[WikiNote, float]]: Retrieved notes paired with their cosine similarity scores.
    """
    if not question or not isinstance(question, str) or not question.strip():
        return []

    if top_k <= 0:
        return []

    # Load stored note embeddings
    try:
        embeddings = emb.load_embeddings(root=root)
    except Exception as e:
        logger.warning(f"Failed to load embeddings: {e}")
        return []

    if not embeddings:
        return []

    # Load existing wiki notes scoped to user_id
    try:
        notes_list = storage.read_wiki_notes(root=root, user_id=user_id)
    except Exception as e:
        logger.warning(f"Failed to read wiki notes: {e}")
        return []

    if not notes_list:
        return []

    notes_by_id = {note.id: note for note in notes_list}

    # Embed the user question
    try:
        q_vec = emb.embed_text(question.strip(), model=model, root=root)
    except Exception as e:
        logger.error(f"Failed to embed question: {e}")
        return []

    # Score each note against the question vector
    scored_notes: list[tuple[WikiNote, float]] = []
    for note_id, note_vec in embeddings.items():
        note = notes_by_id.get(note_id)
        if note is None:
            continue

        try:
            score = emb.cosine_similarity(q_vec, note_vec)
        except Exception:
            score = 0.0

        if threshold is not None and score < threshold:
            continue

        scored_notes.append((note, float(score)))

    # Sort descending by similarity score
    scored_notes.sort(key=lambda item: item[1], reverse=True)

    return scored_notes[:top_k]


def validate_citations(
    answer: str,
    valid_ids: set[str],
) -> tuple[str, list[str]]:
    """Validate cited note IDs in the LLM answer against retrieved note IDs (Task 4.1.3).

    Extracts all citation patterns [id] from the answer. If a citation refers to
    a note ID not present in valid_ids, it is marked as unverified and annotated as
    [id (unverified)] to prevent presenting false attribution to the user.

    Args:
        answer: Raw synthesized answer text from the LLM.
        valid_ids: Set of valid note IDs that were included in the prompt context.

    Returns:
        tuple[str, list[str]]: (sanitized_answer, list_of_invalid_citations)
    """
    if not answer or not isinstance(answer, str):
        return "", []

    invalid_citations: list[str] = []

    def _replace_citation(match: re.Match[str]) -> str:
        citation_id = match.group(1)
        # Check if the citation matches any valid retrieved note ID
        if citation_id in valid_ids:
            return match.group(0)

        # Flag and annotate unverified citation
        invalid_citations.append(citation_id)
        return f"[{citation_id} (unverified)]"

    # Pattern matches bracketed citations like [note001], [proj-01], etc.
    # Excludes markdown links [text](url) using negative lookahead
    sanitized = re.sub(r"\[([a-zA-Z0-9_-]+)\](?!\()", _replace_citation, answer)
    return sanitized, invalid_citations


def ask(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    root: Path | str | None = None,
    model: Any | None = None,
    llm_model: str | None = None,
    client: Any | None = None,
    similarity_threshold: float | None = DEFAULT_SIMILARITY_THRESHOLD,
    user_id: str | None = None,
) -> AskResult:
    """Ask a question against the SecondSelf personal knowledge base (Task 4.1.1).

    Executes the 6-stage RAG pipeline:
    1. Validate input query.
    2. Retrieve top-K relevant notes by embedding similarity.
    3. Check relevance guardrails (returns "I don't have notes about that" if empty).
    4. Assemble full wiki bodies into context under the ~6000 token limit.
    5. Synthesize answer via Groq LLM (`synthesize_answer`).
    6. Return `AskResult` with answer and structured source attributions.

    Args:
        question: User query string.
        top_k: Maximum number of notes to retrieve as context (default: 5).
        root: Optional project root path override.
        model: Optional pre-loaded sentence-transformers model.
        llm_model: Optional Groq model override (defaults to DEFAULT_MODEL).
        client: Optional Groq client instance.
        similarity_threshold: Minimum cosine similarity to consider a note relevant.
        user_id: Optional user ID to isolate notes search to.

    Returns:
        AskResult: Dataclass containing the synthesized answer and source citations.
    """
    if not question or not isinstance(question, str) or not question.strip():
        return AskResult(answer=EMPTY_QUESTION_ANSWER, sources=[])

    clean_question = question.strip()
    if len(clean_question) > MAX_QUESTION_CHARS:
        clean_question = clean_question[:MAX_QUESTION_CHARS].rstrip()
        logger.info(f"Question exceeded {MAX_QUESTION_CHARS} chars; truncated.")

    # Step 1-3: Retrieve top-K relevant notes
    retrieved = retrieve_relevant_notes(
        question=clean_question,
        top_k=top_k,
        threshold=similarity_threshold,
        root=root,
        model=model,
        user_id=user_id,
    )

    if not retrieved:
        return AskResult(answer=NO_NOTES_ANSWER, sources=[])

    # Build application-controlled source citations (only retrieved notes)
    sources: list[dict[str, Any]] = [
        {
            "id": note.id,
            "summary": note.summary,
            "relevance_score": round(score, 4),
            "para": note.para,
        }
        for note, score in retrieved
    ]

    # Step 4: Build RAG context within token budget
    context = build_rag_context(retrieved, max_total_chars=MAX_CONTEXT_CHARS)

    # Step 5: Synthesize answer via Groq LLM
    answer = synthesize_answer(
        context=context,
        question=clean_question,
        model=llm_model,
        client=client,
        temperature=DEFAULT_TEMPERATURE,
    )

    if not answer or not answer.strip():
        raise RuntimeError("LLM synthesis returned an empty answer.")

    # Guardrail: Programmatic citation validation against retrieved notes
    valid_ids = {note.id for note, _ in retrieved}
    sanitized_answer, invalid_citations = validate_citations(
        answer=answer.strip(),
        valid_ids=valid_ids,
    )

    if invalid_citations:
        logger.warning(
            f"Answer cited unverified note IDs not present in retrieved context: {invalid_citations}"
        )

    return AskResult(answer=sanitized_answer, sources=sources)


def build_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser for ask.py."""
    parser = argparse.ArgumentParser(
        description="SecondSelf knowledge base question answering engine (Task 4.1.1)."
    )
    parser.add_argument(
        "question",
        nargs="?",
        default="",
        help="Question to ask your knowledge brain.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of relevant notes to retrieve (default: {DEFAULT_TOP_K}).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help=f"Minimum cosine similarity threshold (default: {DEFAULT_SIMILARITY_THRESHOLD}).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Groq LLM model override (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output structured AskResult as JSON.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Project root directory override (for testing).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for ask.py."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.question or not args.question.strip():
        print("Error: No question provided. Please specify a question to ask.", file=sys.stderr)
        parser.print_usage(file=sys.stderr)
        return 1

    try:
        result = ask(
            question=args.question,
            top_k=args.top_k,
            root=args.root,
            llm_model=args.model,
            similarity_threshold=args.threshold,
        )
    except Exception as e:
        print(f"Error executing ask: {e}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(asdict(result), indent=2))
        return 0

    print(f"\nQuestion: {args.question.strip()}\n")
    print("Answer:")
    print(result.answer)
    print()

    if result.sources:
        print("Sources:")
        for src in result.sources:
            score_str = f"score: {src['relevance_score']:.4f}"
            print(f"  - [{src['id']}] ({src['para']}) {src['summary']} ({score_str})")
    else:
        print("Sources: (none)")

    return 0


if __name__ == "__main__":
    sys.exit(main())

