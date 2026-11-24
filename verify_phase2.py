"""SecondSelf — Root Script Entrypoint for Phase 2 Verification.

Convenience wrapper delegating to evals.verify_phase2.main().
Usage:
    python verify_phase2.py --help
    python verify_phase2.py --format markdown
    python verify_phase2.py --sample-size 10
"""

import sys
from evals.verify_phase2 import main

if __name__ == "__main__":
    sys.exit(main())
