"""SecondSelf — Local Preview & Verification Package (Task 3.3).

Provides automated local preview server, artifact staging, schema contract verification,
and machine-readable evidence reporting for the static knowledge graph explorer.
"""

from preview.report import CheckResult, VerificationReport
from preview.server import ServerReadinessTimeout, StaticPreviewServer
from preview.staging import ArtifactStagingManager, StagingError, compute_sha256
from preview.verifier import FrontendVerifier

__all__ = [
    "ArtifactStagingManager",
    "CheckResult",
    "FrontendVerifier",
    "ServerReadinessTimeout",
    "StaticPreviewServer",
    "StagingError",
    "VerificationReport",
    "compute_sha256",
]
