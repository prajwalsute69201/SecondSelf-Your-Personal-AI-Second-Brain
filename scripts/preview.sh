#!/usr/bin/env bash
# SecondSelf — Local Preview & Verification Shell Wrapper (Task 3.3)
set -euo pipefail

PORT="${1:-8000}"
echo "=========================================================="
echo " Starting SecondSelf Static Knowledge Graph Local Preview "
echo " Port: ${PORT}"
echo "=========================================================="

python3 scripts/preview.py --port "${PORT}"
