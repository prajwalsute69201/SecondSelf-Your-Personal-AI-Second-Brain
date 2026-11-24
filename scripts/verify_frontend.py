#!/usr/bin/env python3
"""SecondSelf — Frontend Quality Gate & Verification Runner (Task 3.3).

Executes automated static frontend verification against real Task 3.1 artifacts:
- Artifact integrity & topology checks
- Data contract schema compliance
- Static HTTP route availability (HTML, CSS, JS, data)
- Page load smoke test
- Graceful failure simulation

Exit Codes:
  0: All checks passed (Clean Quality Gate)
  1: Verification failures (Broken contract, missing assets, 404s)
  2: Setup/Server error (Missing artifacts, port binding error)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

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
logger = logging.getLogger("secondself.scripts.verify_frontend")


def build_arg_parser() -> argparse.ArgumentParser:
    """Builds CLI argument parser for standalone frontend verification."""
    settings = get_settings()
    parser = argparse.ArgumentParser(
        prog="verify_frontend",
        description="Run automated frontend verification and generate audit evidence report.",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Optional base URL of an already-running static preview server",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=settings.paths.public_dir,
        help="Directory of static frontend assets to verify (default: public/)",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=settings.paths.graph_file,
        help="Source path to precomputed graph.json artifact from Task 3.1 (default: public/graph.json)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=settings.preview.artifacts_dir / settings.preview.report_filename,
        help="Target output path for machine-readable verification report JSON",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.preview.port,
        help=f"Port to bind if spinning up an automated preview server (default: {settings.preview.port})",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=settings.preview.host,
        help=f"Host interface to bind automated preview server (default: {settings.preview.host})",
    )
    return parser


def main() -> int:
    """CLI execution entrypoint."""
    parser = build_arg_parser()
    args = parser.parse_args()

    # 1. Stage Artifacts (Read-only copy)
    try:
        staging_mgr = ArtifactStagingManager(
            source_graph=args.source,
            public_dir=args.dir,
        )
        staging_mgr.stage()
    except StagingError as stage_err:
        logger.error("Artifact staging error: %s", stage_err)
        return 2
    except Exception as exc:
        logger.error("Unexpected staging failure: %s", exc)
        return 2

    # 2. Run Verification (with existing server URL or managed ephemeral server)
    if args.url:
        logger.info("Verifying against external running server at %s...", args.url)
        verifier = FrontendVerifier(
            public_dir=args.dir,
            server_url=args.url,
            artifacts_dir=args.report.parent,
            source_graph=args.source,
        )
        report: VerificationReport = verifier.run_all()
        report.save(args.report)
        print(report.render_console_summary())
        return report.exit_code

    # Managed ephemeral server
    logger.info("Spinning up managed preview server on %s:%d...", args.host, args.port)
    try:
        with StaticPreviewServer(
            directory=args.dir,
            host=args.host,
            port=args.port,
        ) as server:
            verifier = FrontendVerifier(
                public_dir=args.dir,
                server_url=server.url,
                artifacts_dir=args.report.parent,
                source_graph=args.source,
            )
            report = verifier.run_all()
            report.save(args.report)
            print(report.render_console_summary())
            return report.exit_code
    except ServerReadinessTimeout as timeout_err:
        logger.error("Server readiness timeout: %s", timeout_err)
        return 2
    except OSError as os_err:
        logger.error("Socket error binding preview server: %s", os_err)
        return 2


if __name__ == "__main__":
    sys.exit(main())
