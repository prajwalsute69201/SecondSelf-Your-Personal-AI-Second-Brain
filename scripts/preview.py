#!/usr/bin/env python3
"""SecondSelf — Local Static Frontend Preview Script (Task 3.3).

Stages Task 3.1 precomputed graph artifacts, starts a robust static HTTP server,
polls readiness, and optionally executes the automated verification test harness.

Exit Codes:
  0: Success (or all verification checks passed)
  1: Verification failure (data contract, missing assets, broken routes)
  2: Server or setup failure (port unavailable, source artifacts missing)
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_settings
from preview.report import VerificationReport
from preview.server import ServerReadinessTimeout, StaticPreviewServer
from preview.staging import ArtifactStagingManager, StagingError
from preview.verifier import FrontendVerifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("secondself.scripts.preview")


def build_arg_parser() -> argparse.ArgumentParser:
    """Builds CLI argument parser for local preview and verification."""
    settings = get_settings()
    parser = argparse.ArgumentParser(
        prog="preview",
        description="Stage data artifacts and serve SecondSelf static explorer locally.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.preview.port,
        help=f"Port to bind static HTTP server (default: {settings.preview.port})",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=settings.preview.host,
        help=f"Host address to bind server (default: {settings.preview.host})",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=settings.paths.public_dir,
        help="Directory of static frontend assets to serve (default: public/)",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=settings.paths.graph_file,
        help="Source path to generated graph.json artifact from Task 3.1 (default: public/graph.json)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run automated frontend verification suite and exit with CI status code",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=settings.preview.artifacts_dir / settings.preview.report_filename,
        help="Target path for machine-readable verification report JSON",
    )
    parser.add_argument(
        "--no-staging",
        action="store_true",
        help="Skip artifact staging step",
    )
    return parser


def main() -> int:
    """CLI execution entrypoint."""
    parser = build_arg_parser()
    args = parser.parse_args()

    # 1. Stage Data Artifacts (Read-only copy)
    if not args.no_staging:
        logger.info("Staging Task 3.1 artifacts from %s...", args.source)
        try:
            staging_mgr = ArtifactStagingManager(
                source_graph=args.source,
                public_dir=args.dir,
            )
            manifest = staging_mgr.stage()
            logger.info("Staged %d artifact(s) successfully.", len(manifest))
        except StagingError as stage_err:
            logger.error("Artifact staging error: %s", stage_err)
            return 2
        except Exception as exc:
            logger.error("Unexpected staging failure: %s", exc)
            return 2

    # 2. Start Preview Server with Clean Lifecycle Management
    server = StaticPreviewServer(
        directory=args.dir,
        host=args.host,
        port=args.port,
    )

    try:
        server_url = server.start()
        print("\n" + "=" * 65)
        print(f"  SecondSelf Knowledge Graph Preview is Live!")
        print(f"  URL: {server_url}")
        print(f"  Directory: {args.dir.resolve()}")
        print("=" * 65 + "\n")
    except ServerReadinessTimeout as timeout_err:
        logger.error("Server readiness timeout: %s", timeout_err)
        return 2
    except OSError as os_err:
        logger.error("Socket error binding preview server: %s", os_err)
        return 2

    # 3. If --verify flag passed, execute verification suite & exit
    if args.verify:
        logger.info("Running automated verification suite against %s...", server_url)
        try:
            verifier = FrontendVerifier(
                public_dir=args.dir,
                server_url=server_url,
                artifacts_dir=args.report.parent,
                source_graph=args.source,
            )
            report: VerificationReport = verifier.run_all()
            report.save(args.report)
            print(report.render_console_summary())
            return report.exit_code
        finally:
            server.stop()

    # 4. Interactive Serving Mode until Ctrl+C
    print("Serving preview interactively. Press Ctrl+C to stop...\n")

    def handle_signal(sig: int, frame: Any) -> None:
        print("\nShutting down preview server cleanly...")
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        while server.is_running:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
