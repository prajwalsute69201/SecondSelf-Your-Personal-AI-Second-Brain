"""Central pipeline orchestrator for SecondSelf (Task 2.2.4).

Coordinates the Librarian workflow by orchestrating:
1. Discovery and classification of pending raw captures into PARA wiki notes (classify.py).
2. Semantic embedding generation and relationship linking across wiki notes (link.py).
3. End-to-end processing combining classification and linking (pipeline.py process).

Usage:
    python pipeline.py process      # Classify pending captures + auto-link all notes
    python pipeline.py classify     # Classify pending raw captures only
    python pipeline.py link         # Auto-link existing wiki notes only
    python pipeline.py --dry-run    # Preview operations without writing files
    python pipeline.py --help       # Show CLI options
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import logging
from pathlib import Path
import sys
from typing import Any, Sequence

# Ensure repository root is on sys.path so 'lib' is importable
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import build_graph
import classify
import lib.embeddings as emb
from lib.models import WikiNote
import lib.storage as storage
import link
from link import DEFAULT_SIMILARITY_THRESHOLD, LinkReport

logger = logging.getLogger(__name__)

__all__ = [
    "StageFailure",
    "PipelineReport",
    "run_classify_stage",
    "run_link_stage",
    "run_graph_stage",
    "run_pipeline",
    "process",
    "classify_only",
    "link_only",
    "build_parser",
    "main",
]


@dataclass
class StageFailure:
    """Structured failure record for a specific item in a pipeline stage.

    Attributes:
        item_id: Unique capture ID or wiki note ID affected by the failure.
        stage: Pipeline stage identifier ('classify' or 'link').
        error_category: Categorization of the error (e.g. 'ExtractionError', 'ClassificationError').
        error_message: Human-readable error description without sensitive credentials.
        can_continue: Whether other items in the batch continued processing.
        retryable: Whether repeating the execution may succeed (e.g. rate limits or network issues).
    """

    item_id: str
    stage: str
    error_category: str
    error_message: str
    can_continue: bool = True
    retryable: bool = False


@dataclass
class PipelineReport:
    """Consolidated summary report returned from a pipeline execution.

    Attributes:
        command: Pipeline command executed ('process', 'classify', or 'link').
        captures_discovered: Total count of unprocessed captures found in raw/.
        captures_classified: Count of captures successfully converted into wiki notes.
        captures_failed: Count of captures that failed classification.
        notes_examined: Count of wiki notes examined during linking.
        comparisons_performed: Number of pairwise similarity evaluations performed.
        links_created: Number of new directed link connections established.
        links_preserved: Count of pre-existing links preserved intact.
        notes_skipped: Number of notes skipped during linking (e.g. empty content).
        failures: List of StageFailure records describing errors encountered.
        classified_notes: List of WikiNote instances produced by classification.
        link_report: LinkReport instance from the linking stage, if executed.
    """

    command: str = "process"
    captures_discovered: int = 0
    captures_classified: int = 0
    captures_failed: int = 0
    notes_examined: int = 0
    comparisons_performed: int = 0
    links_created: int = 0
    links_preserved: int = 0
    notes_skipped: int = 0
    failures: list[StageFailure] = field(default_factory=list)
    classified_notes: list[WikiNote] = field(default_factory=list)
    link_report: LinkReport | None = None
    graph_rebuilt: bool = False
    graph_node_count: int = 0
    graph_edge_count: int = 0
    graph_messages: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """True if the pipeline completed without any stage failures."""
        return len(self.failures) == 0


def _categorize_classify_error(err_str: str) -> tuple[str, bool]:
    """Categorize a classification error string into category and retryability.

    Args:
        err_str: Error message string.

    Returns:
        tuple: (error_category: str, is_retryable: bool).
    """
    s = err_str.lower()
    if "extract" in s or "pdf" in s or "empty content" in s:
        return "ExtractionError", False
    if "rate limit" in s or "429" in s or "timeout" in s or "connection" in s or "network" in s:
        return "LLMNetworkError", True
    if "auth" in s or "api_key" in s or "api key" in s:
        return "AuthenticationError", False
    if "json" in s or "para" in s or "schema" in s:
        return "ClassificationError", True
    return "ClassifyError", False


def _categorize_link_error(err_str: str) -> tuple[str, bool]:
    """Categorize a linking error string into category and retryability.

    Args:
        err_str: Error message string.

    Returns:
        tuple: (error_category: str, is_retryable: bool).
    """
    s = err_str.lower()
    if "model" in s or "sentence-transformers" in s:
        return "ModelLoadError", True
    if "empty content" in s or "whitespace" in s:
        return "EmptyContentError", False
    if "non-finite" in s or "nan" in s or "dimension" in s:
        return "EmbeddingValidationError", False
    if "similarity" in s or "cosine" in s:
        return "SimilarityCalculationError", False
    if "storage" in s or "persist" in s or "file" in s:
        return "StorageError", False
    return "LinkingError", False


def run_classify_stage(
    root: Path | str | None = None,
    model: str | None = None,
    dry_run: bool = False,
    capture_ids: list[str] | None = None,
) -> tuple[list[WikiNote], list[StageFailure], int]:
    """Execute classification stage on unprocessed raw captures.

    Discovers unprocessed captures, extracts readable text, classifies via LLM,
    persists wiki notes, and updates index.json atomically.

    Args:
        root: Optional project root path override.
        model: Optional Groq LLM model override.
        dry_run: If True, previews classification without writing files.
        capture_ids: Optional list of specific capture IDs to classify.

    Returns:
        tuple: (successful_notes, stage_failures, discovered_count).
    """
    # 1. Discover pending captures
    try:
        pending = storage.read_raw_captures(root=root)
        if capture_ids:
            target_set = set(capture_ids)
            pending = [m for m in pending if m.id in target_set]
    except Exception as e:
        failure = StageFailure(
            item_id="discovery",
            stage="classify",
            error_category="StorageReadError",
            error_message=f"Failed to inspect raw captures: {e}",
            can_continue=False,
            retryable=False,
        )
        return [], [failure], 0

    discovered_count = len(pending)
    if discovered_count == 0:
        return [], [], 0

    # 2. Run classify_all
    try:
        successful_notes, failed_tuples = classify.classify_all(
            root=root,
            model=model,
            capture_ids=capture_ids,
            dry_run=dry_run,
        )
    except Exception as e:
        failure = StageFailure(
            item_id="batch",
            stage="classify",
            error_category="ClassifyFatalError",
            error_message=f"Fatal classification failure: {e}",
            can_continue=False,
            retryable=True,
        )
        return [], [failure], discovered_count

    # 3. Process failures into StageFailure objects
    failures: list[StageFailure] = []
    for cap_id, err_msg in failed_tuples:
        cat, retryable = _categorize_classify_error(err_msg)
        failures.append(
            StageFailure(
                item_id=cap_id,
                stage="classify",
                error_category=cat,
                error_message=err_msg,
                can_continue=True,
                retryable=retryable,
            )
        )

    return successful_notes, failures, discovered_count


def run_link_stage(
    notes: list[WikiNote] | None = None,
    root: Path | str | None = None,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    model: Any | None = None,
    dry_run: bool = False,
    force_reembed: bool = False,
    note_id: str | None = None,
) -> tuple[LinkReport, list[StageFailure]]:
    """Execute linking stage across wiki notes using semantic embeddings.

    Computes/reuses embeddings, evaluates cosine similarity, updates YAML
    frontmatter links and body wikilinks, and saves embeddings atomically.

    Args:
        notes: Optional pre-loaded WikiNote list.
        root: Optional project root path override.
        threshold: Minimum cosine similarity threshold (default: 0.65).
        model: Optional SentenceTransformer model instance.
        dry_run: If True, previews linking without writing files.
        force_reembed: If True, recomputes embeddings even if cached.
        note_id: Optional ID to restrict linking focus to a single note.

    Returns:
        tuple: (LinkReport, list of StageFailure objects).
    """
    try:
        report = link.link_notes(
            notes=notes,
            root=root,
            threshold=threshold,
            model=model,
            dry_run=dry_run,
            force_reembed=force_reembed,
            note_id=note_id,
        )
    except Exception as e:
        fatal_report = LinkReport()
        cat, retryable = _categorize_link_error(str(e))
        failure = StageFailure(
            item_id=note_id or "all_notes",
            stage="link",
            error_category=cat,
            error_message=f"Fatal linking failure: {e}",
            can_continue=False,
            retryable=retryable,
        )
        return fatal_report, [failure]

    failures: list[StageFailure] = []
    for err_msg in report.errors:
        cat, retryable = _categorize_link_error(err_msg)
        failures.append(
            StageFailure(
                item_id="link_stage",
                stage="link",
                error_category=cat,
                error_message=err_msg,
                can_continue=True,
                retryable=retryable,
            )
        )

    return report, failures


def _categorize_graph_error(err_str: str) -> tuple[str, bool]:
    """Categorize a graph building error string into category and retryability.

    Args:
        err_str: Error message string.

    Returns:
        tuple: (error_category: str, is_retryable: bool).
    """
    s = err_str.lower()
    if "json" in s or "schema" in s:
        return "GraphSchemaError", False
    if "storage" in s or "oserror" in s or "permission" in s or "write" in s:
        return "StorageWriteError", False
    if "invariant" in s or "para" in s or "duplicate" in s:
        return "GraphInvariantError", False
    return "GraphBuildError", False


def run_graph_stage(
    root: Path | str | None = None,
    dry_run: bool = False,
) -> tuple[dict[str, Any] | None, list[StageFailure], list[str]]:
    """Execute knowledge graph construction stage (Task 3.2.5).

    Materializes wiki notes and semantic/explicit relationships into
    data/graph.json and synchronizes data/index.json last_graph_build timestamp.

    Args:
        root: Optional project root path override.
        dry_run: If True, previews graph build without writing files.

    Returns:
        tuple: (graph_data_or_None, list of StageFailure objects, list of diagnostic messages).
    """
    try:
        graph_data, messages = build_graph.build_and_save_graph(root=root, dry_run=dry_run)
    except Exception as e:
        cat, retryable = _categorize_graph_error(str(e))
        failure = StageFailure(
            item_id="graph.json",
            stage="graph",
            error_category=cat,
            error_message=f"Failed to build or save knowledge graph: {e}",
            can_continue=False,
            retryable=retryable,
        )
        return None, [failure], []

    failures: list[StageFailure] = []
    for msg in messages:
        if "invalid PARA" in msg or "Duplicate node ID" in msg or "references non-existent" in msg:
            cat, retryable = _categorize_graph_error(msg)
            failures.append(
                StageFailure(
                    item_id="graph_invariant",
                    stage="graph",
                    error_category=cat,
                    error_message=msg,
                    can_continue=True,
                    retryable=False,
                )
            )

    return graph_data, failures, messages


def run_pipeline(
    command: str = "process",
    root: Path | str | None = None,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    model: Any | None = None,
    llm_model: str | None = None,
    dry_run: bool = False,
    force_reembed: bool = False,
    capture_ids: list[str] | None = None,
    note_id: str | None = None,
) -> PipelineReport:
    """Coordinate SecondSelf pipeline execution for classify, link, or process.

    Processing sequence:
    1. If command is 'classify' or 'process':
       - Discover unprocessed raw captures.
       - Extract text and classify into PARA categories.
       - Write wiki notes and update index.json.
    2. If command is 'link' or 'process':
       - Discover all wiki notes (including newly classified ones).
       - Compute/reuse embeddings.
       - Create semantic links where similarity >= threshold.
       - Persist updated notes and embeddings.pkl.

    Args:
        command: Pipeline command ('process', 'classify', or 'link').
        root: Optional project root path override.
        threshold: Cosine similarity threshold for linking (default: 0.65).
        model: Optional pre-loaded SentenceTransformer instance.
        llm_model: Optional Groq LLM model override.
        dry_run: If True, previews changes without writing files.
        force_reembed: If True, forces re-embedding of notes.
        capture_ids: Optional list of capture IDs to restrict classification to.
        note_id: Optional wiki note ID to restrict linking to.

    Returns:
        PipelineReport: Comprehensive report of execution counts and failures.

    Raises:
        ValueError: If command is not one of ('process', 'classify', 'link').
    """
    valid_commands = {"process", "classify", "link"}
    clean_command = (command or "process").lower().strip()
    if clean_command not in valid_commands:
        raise ValueError(
            f"Invalid pipeline command {command!r}. Must be one of: {sorted(valid_commands)}"
        )

    report = PipelineReport(command=clean_command)

    # -------------------------------------------------------------------------
    # Stage 1: Classify (for 'classify' and 'process')
    # -------------------------------------------------------------------------
    if clean_command in ("classify", "process"):
        notes, classify_failures, discovered_count = run_classify_stage(
            root=root,
            model=llm_model,
            dry_run=dry_run,
            capture_ids=capture_ids,
        )
        report.captures_discovered = discovered_count
        report.captures_classified = len(notes)
        report.captures_failed = len(classify_failures)
        report.classified_notes = notes
        report.failures.extend(classify_failures)

    # -------------------------------------------------------------------------
    # Stage 2: Link (for 'link' and 'process')
    # -------------------------------------------------------------------------
    if clean_command in ("link", "process"):
        # When running 'process', pass None so run_link_stage re-reads all notes
        # from disk to ensure newly written notes and existing notes are linked
        link_report, link_failures = run_link_stage(
            notes=None,
            root=root,
            threshold=threshold,
            model=model,
            dry_run=dry_run,
            force_reembed=force_reembed,
            note_id=note_id,
        )
        report.link_report = link_report
        report.notes_examined = link_report.notes_examined
        report.comparisons_performed = link_report.comparisons_performed
        report.links_created = link_report.links_created
        report.links_preserved = link_report.links_preserved
        report.notes_skipped = link_report.notes_skipped
        report.failures.extend(link_failures)

    # -------------------------------------------------------------------------
    # Stage 3: Graph Rebuilding (for 'process' - Task 3.2.5)
    # -------------------------------------------------------------------------
    if clean_command == "process":
        # Only rebuild graph if linking stage completed without failures
        link_failed = any(f.stage == "link" for f in report.failures)
        if not link_failed:
            graph_data, graph_failures, graph_messages = run_graph_stage(
                root=root,
                dry_run=dry_run,
            )
            report.failures.extend(graph_failures)
            if graph_data:
                report.graph_rebuilt = True
                report.graph_node_count = graph_data.get("metadata", {}).get("node_count", 0)
                report.graph_edge_count = graph_data.get("metadata", {}).get("edge_count", 0)
                report.graph_messages = graph_messages

    return report


def process(
    root: Path | str | None = None,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    model: Any | None = None,
    llm_model: str | None = None,
    dry_run: bool = False,
    force_reembed: bool = False,
) -> PipelineReport:
    """Convenience function: classify all pending captures and link all wiki notes."""
    return run_pipeline(
        command="process",
        root=root,
        threshold=threshold,
        model=model,
        llm_model=llm_model,
        dry_run=dry_run,
        force_reembed=force_reembed,
    )


def classify_only(
    root: Path | str | None = None,
    llm_model: str | None = None,
    dry_run: bool = False,
    capture_ids: list[str] | None = None,
) -> PipelineReport:
    """Convenience function: classify pending raw captures only."""
    return run_pipeline(
        command="classify",
        root=root,
        llm_model=llm_model,
        dry_run=dry_run,
        capture_ids=capture_ids,
    )


def link_only(
    root: Path | str | None = None,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    model: Any | None = None,
    dry_run: bool = False,
    force_reembed: bool = False,
    note_id: str | None = None,
) -> PipelineReport:
    """Convenience function: link wiki notes only."""
    return run_pipeline(
        command="link",
        root=root,
        threshold=threshold,
        model=model,
        dry_run=dry_run,
        force_reembed=force_reembed,
        note_id=note_id,
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser for pipeline.py."""
    parser = argparse.ArgumentParser(
        prog="pipeline.py",
        description="SecondSelf Librarian pipeline orchestrator: classify captures and auto-link wiki notes.",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        description="Valid pipeline commands (defaults to 'process' if omitted)",
    )

    # Subcommand: process
    proc_parser = subparsers.add_parser(
        "process",
        help="Classify all pending raw captures then auto-link all wiki notes (default).",
    )
    _add_shared_arguments(proc_parser)

    # Subcommand: classify
    class_parser = subparsers.add_parser(
        "classify",
        help="Classify pending raw captures into PARA wiki notes only.",
    )
    _add_shared_arguments(class_parser)

    # Subcommand: link
    link_parser = subparsers.add_parser(
        "link",
        help="Discover semantic relationships and auto-link wiki notes only.",
    )
    _add_shared_arguments(link_parser)

    # Also add shared arguments to root parser for `python pipeline.py --dry-run` etc.
    _add_shared_arguments(parser)

    return parser


def _add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    """Add shared pipeline command-line arguments to a parser."""
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Preview pipeline stages without writing files or updating index state.",
    )
    parser.add_argument(
        "--threshold",
        dest="threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        metavar="FLOAT",
        help=f"Cosine similarity threshold for linking (default: {DEFAULT_SIMILARITY_THRESHOLD}).",
    )
    parser.add_argument(
        "--model",
        dest="llm_model",
        metavar="MODEL",
        help="Override default Groq LLM model identifier for classification.",
    )
    parser.add_argument(
        "--force-reembed",
        dest="force_reembed",
        action="store_true",
        help="Force recomputation of embeddings even if cached in data/embeddings.pkl.",
    )
    parser.add_argument(
        "--id",
        dest="item_id",
        metavar="ID",
        help="Target a specific capture ID (for classify) or note ID (for link).",
    )
    parser.add_argument(
        "--root",
        dest="root",
        metavar="PATH",
        help=argparse.SUPPRESS,  # For internal test usage
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for pipeline.py.

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

    command = args.command or "process"
    root = Path(args.root).resolve() if getattr(args, "root", None) else None
    dry_run = getattr(args, "dry_run", False)
    threshold = getattr(args, "threshold", DEFAULT_SIMILARITY_THRESHOLD)
    llm_model = getattr(args, "llm_model", None)
    force_reembed = getattr(args, "force_reembed", False)
    item_id = getattr(args, "item_id", None)

    prefix = "[dry-run] " if dry_run else ""
    print(f"{prefix}Starting SecondSelf pipeline: {command.upper()}")

    target_caps = [item_id] if item_id and command == "classify" else None
    target_note = item_id if item_id and command == "link" else None

    # Load embedding model if linking stage will run
    model_instance = None
    if command in ("link", "process"):
        try:
            model_instance = emb.load_model(root=root)
        except Exception as e:
            print(f"Error: Failed to load embedding model: {e}", file=sys.stderr)
            return 1

    try:
        report = run_pipeline(
            command=command,
            root=root,
            threshold=threshold,
            model=model_instance,
            llm_model=llm_model,
            dry_run=dry_run,
            force_reembed=force_reembed,
            capture_ids=target_caps,
            note_id=target_note,
        )
    except KeyboardInterrupt:
        print("\nPipeline execution cancelled by user.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Fatal pipeline error: {e}", file=sys.stderr)
        return 1

    # Print Stage Progress
    if command in ("classify", "process"):
        print(f"\n{prefix}Classification Stage:")
        print(f"  Pending discovered:    {report.captures_discovered}")
        print(f"  Successfully processed: {report.captures_classified}")
        print(f"  Failed captures:        {report.captures_failed}")
        for note in report.classified_notes:
            action = "[dry-run] Would write" if dry_run else "Wrote"
            print(f"    - {action} raw/{note.raw_id} → wiki/{note.para}/{note.id}.md")

    if command in ("link", "process"):
        print(f"\n{prefix}Linking Stage:")
        print(f"  Notes examined:        {report.notes_examined}")
        print(f"  Comparisons performed: {report.comparisons_performed} (threshold: {threshold:.2f})")
        print(f"  Links created:         {report.links_created}")
        print(f"  Existing preserved:    {report.links_preserved}")
        print(f"  Notes skipped:         {report.notes_skipped}")
        if report.link_report and report.link_report.new_links:
            seen_pairs: set[tuple[str, str]] = set()
            for src, tgt, score in report.link_report.new_links:
                pair = (min(src, tgt), max(src, tgt))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    print(f"    - Linked {src} <-> {tgt} (similarity: {score:.4f})")

    if command == "process" and report.graph_rebuilt:
        action = "[dry-run] Would build" if dry_run else "Built"
        print(f"\n{prefix}Graph Stage:")
        print(f"  {action} knowledge graph: {report.graph_node_count} nodes, {report.graph_edge_count} edges")
        if report.graph_messages:
            for msg in report.graph_messages:
                print(f"    - {msg}")

    # Print Summary & Failures
    print("\n" + "=" * 60)
    print(f"SecondSelf Pipeline Summary ({command})")
    print("=" * 60)
    if command in ("classify", "process"):
        print(f"Captures Processed:    {report.captures_classified} / {report.captures_discovered}")
    if command in ("link", "process"):
        print(f"Semantic Links Added:  {report.links_created}")
    if command == "process":
        print(f"Graph Rebuilt:         {'Yes' if report.graph_rebuilt else 'No'} ({report.graph_node_count} nodes, {report.graph_edge_count} edges)")
    print(f"Stage Failures:        {len(report.failures)}")
    print("=" * 60)

    if report.failures:
        print("\nFailures Encountered:", file=sys.stderr)
        for f in report.failures:
            retry_str = "Yes" if f.retryable else "No"
            cont_str = "Yes" if f.can_continue else "No"
            print(
                f"  [{f.stage}] Item: {f.item_id} | Category: {f.error_category}\n"
                f"    Message: {f.error_message}\n"
                f"    Can continue: {cont_str} | Retryable: {retry_str}",
                file=sys.stderr,
            )
        return 1

    print(f"\n{prefix}Pipeline completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

