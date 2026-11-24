"""SecondSelf — Root Script Entrypoint for Graph Projection.

Convenience CLI wrapper delegating to graph.cli.main().
Usage:
    python build_graph.py --help
    python build_graph.py --format all --stats
    python build_graph.py --dry-run
"""

import sys
from graph.cli import main

if __name__ == "__main__":
    sys.exit(main())
