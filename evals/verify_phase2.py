"""SecondSelf — CLI Entrypoint for Phase 2 Real Data Verification.

Runs the end-to-end verification harness against the real golden benchmark dataset,
computes quality metrics, validates pipeline sanity checks, and outputs auditable
JSON and Markdown reports.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from evals.harness import VerificationHarness

logger = logging.getLogger("secondself.verify")


def build_arg_parser() -> argparse.ArgumentParser:
    """Builds CLI argument parser for Phase 2 verification."""
    parser = argparse.ArgumentParser(
        prog="secondself-verify-phase2",
        description="Verify SecondSelf Phase 2 (Classification & Auto-Linking) on real data.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to verification dataset JSON (defaults to evals/datasets/para_golden.json)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional limit on the number of dataset items to evaluate",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory to save generated JSON and Markdown reports (default: reports/)",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "json", "all", "summary"],
        default="markdown",
        help="Console output format (default: markdown)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed for sampling (default: 42)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero exit code if quality gate fails",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output (useful in automated CI scripts)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI execution entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    harness = VerificationHarness()

    try:
        report = harness.run_verification(
            dataset_path=args.dataset,
            sample_size=args.sample_size,
            seed=args.seed,
        )
    except Exception as exc:
        logger.error("Verification execution failed: %s", exc, exc_info=True)
        return 1

    # Ensure output directory exists
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "verification_report.json"
    md_path = out_dir / "verification_report.md"

    json_str = report.to_json(indent=2)
    md_str = report.to_markdown()

    json_path.write_text(json_str, encoding="utf-8")
    md_path.write_text(md_str, encoding="utf-8")
    logger.info("Saved JSON report to '%s'", json_path)
    logger.info("Saved Markdown summary to '%s'", md_path)

    # Output to stdout according to format
    if not args.quiet:
        if args.format == "json":
            print(json_str)
        elif args.format == "summary":
            print(f"Status: {'PASS' if report.overall_pass else 'FAIL'}")
            print(f"Accuracy: {report.classification_metrics.accuracy * 100:.1f}%")
            print(f"Macro-F1: {report.classification_metrics.macro_f1:.4f}")
            print(f"Links: {report.graph_topology_metrics.total_edges} (Avg deg {report.graph_topology_metrics.avg_degree:.2f})")
            print(f"Summary: {report.summary}")
        else:  # markdown or all
            print(md_str)

    if args.strict and not report.overall_pass:
        logger.error("Strict mode: Quality gate breached.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
