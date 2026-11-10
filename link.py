"""Semantic linking pipeline and CLI for SecondSelf (Task 2.2.2).

Discovers semantically related SecondSelf knowledge notes using sentence embeddings
generated in Task 2.2.1 (all-MiniLM-L6-v2), computes cosine similarity against a
defined threshold (default 0.75), updates YAML frontmatter `links: [...]`, and
appends deduplicated `[[target-id]]` wikilinks into note bodies.

Usage:
    python link.py                      # Link all wiki notes (threshold 0.75)
    python link.py --threshold 0.80     # Use custom similarity threshold
    python link.py --id <note_id>       # Link a specific wiki note
    python link.py --dry-run            # Preview links without modifying files
    python link.py --force-reembed      # Recompute embeddings for all notes
    python link.py --help               # Show CLI options
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import sys
from typing import Any, Sequence

import numpy as np
import yaml

# Ensure repository root is on sys.path so 'lib' is importable
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import lib.embeddings as emb
from lib.models import WikiNote
import lib.storage as storage

logger = logging.getLogger(__name__)

# Default similarity threshold specified in Implementationplan.md (Tuned in Task 2.2.3)
DEFAULT_SIMILARITY_THRESHOLD: float = 0.65

__all__ = [
    "DEFAULT_SIMILARITY_THRESHOLD",
    "LinkReport",
    "extract_note_title",
    "strip_appended_wikilinks",
    "build_note_text",
    "append_wikilink_to_body",
    "append_wikilinks_to_body",
    "link_pair",
    "safe_read_wiki_notes",
    "compute_note_embeddings",
    "find_similar_notes",
    "link_notes",
    "link_all",
    "build_parser",
    "main",
]


@dataclass
class LinkReport:
    """Detailed summary report of a linking execution.

    Attributes:
        notes_examined: Total count of wiki notes processed or inspected.
        comparisons_performed: Number of pairwise cosine similarity calculations evaluated.
        links_created: Number of new directed link connections added across notes.
        links_preserved: Number of pre-existing links preserved without modification.
        notes_skipped: Number of notes skipped due to empty content or embedding failures.
        errors: List of error or warning message strings encountered.
        new_links: List of tuples (source_id, target_id, similarity_score) for links created.
    """

    notes_examined: int = 0
    comparisons_performed: int = 0
    links_created: int = 0
    links_preserved: int = 0
    notes_skipped: int = 0
    errors: list[str] = field(default_factory=list)
    new_links: list[tuple[str, str, float]] = field(default_factory=list)


def extract_note_title(note: WikiNote) -> str:
    """Extract or infer a title for a note if present.

    Checks:
    1. An explicit 'title' attribute on the note object if present.
    2. The first non-empty line of the note body if it is a markdown heading (# Heading).

    Args:
        note: WikiNote instance.

    Returns:
        str: Extracted title or empty string.
    """
    if hasattr(note, "title") and getattr(note, "title"):
        title_val = str(getattr(note, "title")).strip()
        if title_val:
            return title_val

    if note.body:
        for line in note.body.splitlines():
            s = line.strip()
            if s:
                if s.startswith("#") and not s.startswith("###"):
                    return s.lstrip("#").strip()
                break

    return ""


def strip_appended_wikilinks(body: str) -> str:
    """Strip trailing auto-generated wikilinks to get core content for hashing.

    Prevents self-modifying hash cycles where adding a wikilink would make a note
    appear modified on subsequent runs.

    Args:
        body: Markdown body string.

    Returns:
        str: Body string without trailing standalone wikilinks.
    """
    if not body:
        return ""

    lines = body.rstrip().splitlines()
    while lines:
        line_clean = lines[-1].strip()
        # Matches lines containing only [[...]] links
        if line_clean and re.fullmatch(r"(\[\[[a-zA-Z0-9_\-]+\]\]\s*)+", line_clean):
            lines.pop()
        else:
            break

    return "\n".join(lines).rstrip()


def build_note_text(note: WikiNote) -> str:
    """Construct text representation of a note for semantic embedding.

    Per Implementationplan.md Task 2.2.2: embed(title + summary + body).
    Extracts title if available, combined with summary and clean core body.

    Args:
        note: WikiNote instance.

    Returns:
        str: Consolidated text ready for embedding.
    """
    title = extract_note_title(note)
    core_body = strip_appended_wikilinks(note.body or "")

    parts: list[str] = []
    if title:
        parts.append(title)
    if note.summary and note.summary.strip():
        parts.append(note.summary.strip())
    if core_body and core_body.strip():
        parts.append(core_body.strip())

    return "\n\n".join(parts)


def append_wikilink_to_body(body: str, target_id: str) -> str:
    """Append [[target_id]] wikilink to note body safely and idempotently.

    If [[target_id]] already appears anywhere in the body, returns body unchanged.
    Ensures clean markdown spacing without duplicate newlines or whitespace drift.

    Args:
        body: Current note markdown body string.
        target_id: Wiki note ID to link to.

    Returns:
        str: Updated body text containing the wikilink.
    """
    wikilink = f"[[{target_id}]]"
    if wikilink in body:
        return body

    body_trimmed = body.rstrip() if body else ""
    if not body_trimmed:
        return f"{wikilink}\n"

    # If body already ends with a wikilink, stack the new one on the next line
    if re.search(r"\[\[[a-zA-Z0-9_\-]+\]\]$", body_trimmed):
        return f"{body_trimmed}\n{wikilink}\n"

    return f"{body_trimmed}\n\n{wikilink}\n"


def append_wikilinks_to_body(body: str, target_ids: Sequence[str]) -> str:
    """Append multiple [[target_id]] wikilinks to note body, deduplicated.

    Args:
        body: Current note markdown body string.
        target_ids: Sequence of wiki note IDs to link to.

    Returns:
        str: Updated body text with new wikilinks appended.
    """
    new_targets = [tid for tid in target_ids if f"[[{tid}]]" not in body]
    if not new_targets:
        return body

    body_trimmed = body.rstrip() if body else ""
    links_block = "\n".join(f"[[{tid}]]" for tid in new_targets)

    if not body_trimmed:
        return f"{links_block}\n"

    if re.search(r"\[\[[a-zA-Z0-9_\-]+\]\]$", body_trimmed):
        return f"{body_trimmed}\n{links_block}\n"

    return f"{body_trimmed}\n\n{links_block}\n"


def link_pair(source: WikiNote, target_id: str) -> bool:
    """Add a relationship from source note to target_id in both frontmatter and body.

    Args:
        source: WikiNote being linked.
        target_id: Target note identifier to link to.

    Returns:
        bool: True if a new link was created, False if it was already present.
    """
    if source.id == target_id:
        return False

    already_in_links = target_id in source.links
    already_in_body = f"[[{target_id}]]" in (source.body or "")

    if not already_in_links:
        source.links.append(target_id)

    if not already_in_body:
        source.body = append_wikilink_to_body(source.body or "", target_id)

    return not (already_in_links and already_in_body)


def safe_read_wiki_notes(
    root: Path | str | None = None,
) -> tuple[list[WikiNote], list[str]]:
    """Discover and parse all wiki markdown notes, isolating per-file read errors.

    Args:
        root: Optional root directory override.

    Returns:
        tuple: (list of valid WikiNote instances, list of error strings for bad files).
    """
    wiki_dir = storage.get_wiki_dir(root)
    if not wiki_dir.exists():
        return [], []

    valid_notes: list[WikiNote] = []
    errors: list[str] = []

    required_fields = {
        "id",
        "raw_id",
        "para",
        "tags",
        "summary",
        "created",
        "links",
    }

    for note_path in sorted(wiki_dir.rglob("*.md")):
        try:
            with note_path.open("r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            errors.append(f"Failed to read wiki note at {note_path}: {e}")
            continue

        text = text.lstrip("\ufeff")
        lines = text.splitlines(keepends=True)
        if not lines or lines[0].strip() != "---":
            errors.append(f"Malformed note at {note_path}: missing opening '---' delimiter")
            continue

        closing_idx = -1
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                closing_idx = i
                break

        if closing_idx == -1:
            errors.append(f"Malformed note at {note_path}: missing closing '---' delimiter")
            continue

        fm_text = "".join(lines[1:closing_idx])
        body_lines = lines[closing_idx + 1 :]
        body_text = "".join(body_lines)

        if body_text.startswith("\r\n"):
            body_text = body_text[2:]
        elif body_text.startswith("\n"):
            body_text = body_text[1:]

        try:
            fm_data = yaml.safe_load(fm_text)
        except Exception as e:
            errors.append(f"YAML parsing error in {note_path}: {e}")
            continue

        if not isinstance(fm_data, dict):
            errors.append(f"Frontmatter in {note_path} is not a YAML dictionary")
            continue

        missing = required_fields - set(fm_data.keys())
        if missing:
            errors.append(
                f"Note {note_path} missing required frontmatter fields: {sorted(missing)}"
            )
            continue

        raw_tags = fm_data.get("tags") or []
        tags = [str(t) for t in raw_tags] if isinstance(raw_tags, list) else []

        raw_links = fm_data.get("links") or []
        links = [str(l) for l in raw_links] if isinstance(raw_links, list) else []

        note = WikiNote(
            id=str(fm_data["id"]),
            raw_id=str(fm_data["raw_id"]),
            para=str(fm_data["para"]),
            tags=tags,
            summary=str(fm_data["summary"]),
            created=str(fm_data["created"]),
            links=links,
            body=body_text,
        )
        valid_notes.append(note)

    return valid_notes, errors


def compute_note_embeddings(
    notes: Sequence[WikiNote],
    embeddings: dict[str, np.ndarray],
    model: Any | None = None,
    root: Path | str | None = None,
    force_reembed: bool = False,
    note_hashes: dict[str, str] | None = None,
) -> tuple[dict[str, np.ndarray], list[str], list[str], list[str]]:
    """Compute embeddings for new or changed notes, reusing valid existing ones.

    Args:
        notes: Sequence of WikiNote instances to embed.
        embeddings: Existing mapping of note ID to embedding vector.
        model: Optional pre-loaded SentenceTransformer instance.
        root: Optional project root path override.
        force_reembed: If True, recomputes vectors even if valid existing ones exist.
        note_hashes: Optional mapping of note ID to previous content hash.

    Returns:
        tuple: (updated_embeddings, newly_embedded_ids, skipped_ids, error_messages).
    """
    updated: dict[str, np.ndarray] = dict(embeddings)
    newly_embedded: list[str] = []
    skipped: list[str] = []
    errors: list[str] = []

    hashes_map = note_hashes if note_hashes is not None else {}

    for note in notes:
        text = build_note_text(note)
        if not text or not text.strip():
            skipped.append(note.id)
            errors.append(f"Note '{note.id}' has empty content; cannot compute embedding.")
            continue

        curr_hash = storage.content_hash(text)
        prev_hash = hashes_map.get(note.id)

        # Check if existing embedding is present and valid
        existing_vec = updated.get(note.id)
        has_valid_existing = (
            existing_vec is not None
            and isinstance(existing_vec, np.ndarray)
            and existing_vec.shape == (emb.EMBEDDING_DIMENSION,)
            and np.all(np.isfinite(existing_vec))
        )

        # Reuse existing valid embedding if content is unchanged and not forced
        if not force_reembed and has_valid_existing:
            if prev_hash is None or prev_hash == curr_hash:
                hashes_map[note.id] = curr_hash
                continue

        # Generate new embedding
        try:
            vec = emb.embed_text(text, model=model, root=root)
        except Exception as e:
            skipped.append(note.id)
            errors.append(f"Failed to generate embedding for note '{note.id}': {e}")
            continue

        # Validate generated vector
        if vec.shape != (emb.EMBEDDING_DIMENSION,):
            skipped.append(note.id)
            errors.append(
                f"Generated embedding for note '{note.id}' has invalid shape {vec.shape}, "
                f"expected ({emb.EMBEDDING_DIMENSION},)"
            )
            continue

        if not np.all(np.isfinite(vec)):
            skipped.append(note.id)
            errors.append(
                f"Generated embedding for note '{note.id}' contains non-finite values (NaN/Inf)."
            )
            continue

        updated[note.id] = vec
        hashes_map[note.id] = curr_hash
        newly_embedded.append(note.id)

    return updated, newly_embedded, skipped, errors


def find_similar_notes(
    source_id: str,
    source_vec: np.ndarray,
    all_embeddings: dict[str, np.ndarray],
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> list[tuple[str, float]]:
    """Find other notes whose embedding similarity meets or exceeds threshold.

    Never compares a note to itself. Returns matches sorted descending by similarity.

    Args:
        source_id: ID of the note being compared.
        source_vec: 1D embedding vector of the source note.
        all_embeddings: Dictionary of note ID to embedding vector.
        threshold: Minimum similarity threshold (inclusive).

    Returns:
        list of (target_id, similarity_score) sorted descending by score.
    """
    matches: list[tuple[str, float]] = []

    for target_id, target_vec in all_embeddings.items():
        if target_id == source_id:
            continue

        try:
            score = emb.cosine_similarity(source_vec, target_vec)
        except Exception as e:
            logger.warning(
                f"Failed to compute similarity between {source_id} and {target_id}: {e}"
            )
            continue

        if score >= threshold:
            matches.append((target_id, score))

    matches.sort(key=lambda item: item[1], reverse=True)
    return matches


def link_notes(
    notes: list[WikiNote] | None = None,
    root: Path | str | None = None,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    model: Any | None = None,
    dry_run: bool = False,
    force_reembed: bool = False,
    note_id: str | None = None,
    bidirectional: bool = True,
) -> LinkReport:
    """Identify semantically related notes and create bidirectional relationships.

    Workflow:
    1. Discovers notes in wiki/ or uses provided list.
    2. Loads existing embeddings from data/embeddings.pkl (recovering safely from corruption).
    3. Computes or reuses embeddings for notes.
    4. Evaluates pairwise cosine similarity (skipping self-comparison).
    5. Links notes when similarity >= threshold (updating frontmatter and body).
    6. Preserves existing metadata, body content, and unrelated links.
    7. Atomically saves updated notes and data/embeddings.pkl.

    Args:
        notes: Optional pre-loaded list of WikiNote instances.
        root: Optional project root path override.
        threshold: Similarity score threshold (default: 0.75).
        model: Optional SentenceTransformer model instance.
        dry_run: If True, previews links without writing files or updating embeddings.pkl.
        force_reembed: If True, re-embeds notes even if existing embeddings are present.
        note_id: Optional ID to restrict linking focus to a single note.
        bidirectional: If True, reciprocal links are updated on both notes.

    Returns:
        LinkReport: Full report detailing execution metrics, created links, and errors.
    """
    report = LinkReport()

    # 1. Load notes
    if notes is None:
        all_notes, read_errors = safe_read_wiki_notes(root=root)
        report.errors.extend(read_errors)
    else:
        all_notes = list(notes)

    if not all_notes:
        report.notes_examined = 0
        return report

    notes_by_id: dict[str, WikiNote] = {n.id: n for n in all_notes}

    # If specific note_id requested, verify existence
    if note_id is not None:
        if note_id not in notes_by_id:
            report.errors.append(f"Wiki note with ID '{note_id}' not found.")
            return report
        target_notes = [notes_by_id[note_id]]
    else:
        target_notes = all_notes

    report.notes_examined = len(target_notes)

    # 2. Load existing embeddings safely (Edge Cases 3.5 & 3.13)
    existing_embeddings: dict[str, np.ndarray] = {}
    try:
        existing_embeddings = emb.load_embeddings(root=root)
    except Exception as e:
        report.errors.append(
            f"Existing embeddings corrupted or incompatible ({e}); resetting embeddings cache."
        )
        existing_embeddings = {}

    # Load index note hashes if present
    index_data = storage.load_index(root=root)
    note_hashes: dict[str, str] = index_data.get("note_hashes", {})
    if not isinstance(note_hashes, dict):
        note_hashes = {}

    # 3. Compute or reuse embeddings
    embeddings, newly_embedded, skipped_ids, emb_errors = compute_note_embeddings(
        notes=all_notes,
        embeddings=existing_embeddings,
        model=model,
        root=root,
        force_reembed=force_reembed,
        note_hashes=note_hashes,
    )
    report.notes_skipped = len(skipped_ids)
    report.errors.extend(emb_errors)

    # Pre-calculate count of existing links across target notes
    initial_links: set[tuple[str, str]] = set()
    for n in all_notes:
        for tgt in n.links:
            initial_links.add((n.id, tgt))

    # Notes that are modified during this run
    modified_notes: set[str] = set()

    # 4. Pairwise similarity evaluation and linking
    # To avoid duplicate pairwise calculations, track evaluated pairs
    evaluated_pairs: set[tuple[str, str]] = set()

    for source_note in target_notes:
        src_id = source_note.id
        if src_id not in embeddings:
            continue

        src_vec = embeddings[src_id]

        for other_note in all_notes:
            tgt_id = other_note.id
            if tgt_id == src_id:
                # Never compare a note with itself (Edge Case 3.6)
                continue

            if tgt_id not in embeddings:
                continue

            pair_key = (min(src_id, tgt_id), max(src_id, tgt_id))
            if pair_key in evaluated_pairs:
                continue
            evaluated_pairs.add(pair_key)

            tgt_vec = embeddings[tgt_id]

            try:
                score = emb.cosine_similarity(src_vec, tgt_vec)
                report.comparisons_performed += 1
            except Exception as e:
                report.errors.append(
                    f"Similarity calculation failed between '{src_id}' and '{tgt_id}': {e}"
                )
                continue

            if score >= threshold:
                # Link source -> target
                src_link_new = link_pair(source_note, tgt_id)
                if src_link_new:
                    report.links_created += 1
                    report.new_links.append((src_id, tgt_id, score))
                    modified_notes.add(src_id)
                else:
                    report.links_preserved += 1

                # Link target -> source if bidirectional
                if bidirectional:
                    tgt_link_new = link_pair(other_note, src_id)
                    if tgt_link_new:
                        report.links_created += 1
                        report.new_links.append((tgt_id, src_id, score))
                        modified_notes.add(tgt_id)
                    else:
                        report.links_preserved += 1

    # Count pre-existing links on target notes that were not part of similar pairs
    for n in target_notes:
        for tgt in n.links:
            if (n.id, tgt) in initial_links:
                pair_key = (min(n.id, tgt), max(n.id, tgt))
                # If not already counted in preserved links above
                if pair_key not in evaluated_pairs:
                    report.links_preserved += 1

    if dry_run:
        return report

    # 5. Persist modified wiki notes
    for note_id_to_save in modified_notes:
        note_to_save = notes_by_id[note_id_to_save]
        try:
            storage.write_wiki_note(note_to_save, root=root)
        except Exception as e:
            report.errors.append(f"Failed to persist wiki note '{note_id_to_save}': {e}")

    # 6. Persist embeddings atomically
    try:
        emb.save_embeddings(embeddings, root=root)
    except Exception as e:
        report.errors.append(f"Failed to save embeddings to data/embeddings.pkl: {e}")

    # 7. Update index.json note hashes
    try:
        index_data["note_hashes"] = note_hashes
        storage.save_index(index_data, root=root)
    except Exception as e:
        report.errors.append(f"Failed to update data/index.json: {e}")

    return report


def link_all(
    root: Path | str | None = None,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    model: Any | None = None,
    dry_run: bool = False,
    force_reembed: bool = False,
) -> LinkReport:
    """Convenience entry point to discover and link all wiki notes.

    Args:
        root: Optional project root path override.
        threshold: Minimum cosine similarity threshold (default 0.75).
        model: Optional SentenceTransformer model instance.
        dry_run: If True, previews links without writing files.
        force_reembed: If True, re-embeds all notes.

    Returns:
        LinkReport: Execution summary.
    """
    return link_notes(
        notes=None,
        root=root,
        threshold=threshold,
        model=model,
        dry_run=dry_run,
        force_reembed=force_reembed,
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser for link.py."""
    parser = argparse.ArgumentParser(
        prog="link.py",
        description="Auto-link semantically related SecondSelf knowledge notes via embeddings.",
    )
    parser.add_argument(
        "--threshold",
        dest="threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        metavar="FLOAT",
        help=f"Cosine similarity threshold (default: {DEFAULT_SIMILARITY_THRESHOLD}).",
    )
    parser.add_argument(
        "--id",
        dest="note_id",
        metavar="NOTE_ID",
        help="Link a specific wiki note by ID rather than all notes.",
    )
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Preview semantic relationships without writing files or modifying embeddings.pkl.",
    )
    parser.add_argument(
        "--force-reembed",
        dest="force_reembed",
        action="store_true",
        help="Force recomputation of embeddings even if cached in data/embeddings.pkl.",
    )
    parser.add_argument(
        "--root",
        dest="root",
        metavar="PATH",
        help=argparse.SUPPRESS,  # For internal test use
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for link.py.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        int: Exit status code (0 for success, 1 for failure, 130 for user interrupt).
    """
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 0

    root = Path(args.root).resolve() if args.root else None

    # Load model failure handling (Edge Case 3.2)
    try:
        model = emb.load_model(root=root)
    except Exception as e:
        print(f"Error: Failed to load embedding model: {e}", file=sys.stderr)
        return 1

    try:
        report = link_notes(
            root=root,
            threshold=args.threshold,
            model=model,
            dry_run=args.dry_run,
            force_reembed=args.force_reembed,
            note_id=args.note_id,
        )
    except KeyboardInterrupt:
        print("\nLinking cancelled by user.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Fatal error during linking: {e}", file=sys.stderr)
        return 1

    # Report results
    prefix = "[dry-run] " if args.dry_run else ""

    if report.notes_examined == 0:
        print("No wiki notes found in wiki/. Nothing to link.")
        return 0

    print(
        f"{prefix}Examined {report.notes_examined} note(s), performed "
        f"{report.comparisons_performed} similarity comparisons (threshold: {args.threshold:.2f})."
    )

    if report.new_links:
        print(f"\n{prefix}Created relationships:")
        # Show unique undirected pairs for cleaner display
        seen_pairs: set[tuple[str, str]] = set()
        for src, tgt, score in report.new_links:
            key = (min(src, tgt), max(src, tgt))
            if key not in seen_pairs:
                seen_pairs.add(key)
                print(f"  {src} <-> {tgt} (similarity: {score:.4f})")
    else:
        print(f"\n{prefix}No new semantic links created (all below threshold or already linked).")

    print("\nSummary:")
    print(f"  Notes examined:        {report.notes_examined}")
    print(f"  Comparisons performed: {report.comparisons_performed}")
    print(f"  Links created:         {report.links_created}")
    print(f"  Existing preserved:    {report.links_preserved}")
    print(f"  Notes skipped:         {report.notes_skipped}")
    print(f"  Errors encountered:    {len(report.errors)}")

    if report.errors:
        print("\nErrors / Warnings:", file=sys.stderr)
        for err in report.errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

