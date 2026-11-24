#!/usr/bin/env bash
set -euo pipefail

echo "================================================================"
echo "          SecondSelf Dependency & Security Audit"
echo "================================================================"

python3 scripts/audit_deps.py

if command -v pip-audit &> /dev/null; then
    echo "Running deep pip-audit on all tiers..."
    pip-audit -r requirements/runtime.txt
else
    echo "Note: Install 'pip-audit' via 'pip install -r requirements/security.txt' for deep CVE checks."
fi
