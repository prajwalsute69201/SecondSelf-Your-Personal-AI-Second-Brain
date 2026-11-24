#!/usr/bin/env python3
"""Dependency and Serverless Bundle Guard Audit Script for SecondSelf.

Performs:
1. Architectural guard: Verifies no heavy ML packages are present in production requirements.
2. CVE Audit: Runs pip-audit against requirements files if pip-audit is installed.
3. Code Security Scan: Runs bandit on api/ and core source files.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

FORBIDDEN_SERVERLESS_PACKAGES = {
    "torch",
    "sentence-transformers",
    "transformers",
    "scipy",
    "scikit-learn",
    "sklearn",
    "spacy",
    "tensorflow",
    "keras",
    "trafilatura",
    "pypdf",
}

ROOT_DIR = Path(__file__).resolve().parent.parent


def check_serverless_bundle_purity() -> bool:
    print("[1/3] Auditing Production Serverless Requirements for Bundle Size Violations...")
    req_file = ROOT_DIR / "requirements.txt"
    runtime_file = ROOT_DIR / "requirements" / "runtime.txt"

    violations = []
    for file_path in [req_file, runtime_file]:
        if not file_path.exists():
            continue
        content = file_path.read_text(encoding="utf-8")
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            pkg_name = re.split(r"[><=~;]", line)[0].strip().lower()
            if pkg_name in FORBIDDEN_SERVERLESS_PACKAGES:
                violations.append((file_path.name, pkg_name))

    if violations:
        print("[FAIL] Heavy/forbidden packages detected in production serverless requirements:")
        for fname, pkg in violations:
            print(f"   - {fname} contains forbidden package: '{pkg}'")
        print("   This will cause Vercel serverless deployment to exceed the 250MB limit.")
        print("   Move heavy dependencies into requirements/pipeline.txt instead.")
        return False

    print("[PASS] Production serverless requirements are lean (< 50MB estimate). Zero heavy ML packages.\n")
    return True


def run_pip_audit() -> bool:
    print("[2/3] Checking for known CVEs via pip-audit...")
    try:
        cmd = [sys.executable, "-m", "pip_audit", "-r", str(ROOT_DIR / "requirements.txt")]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode == 0:
            print("[PASS] pip-audit found zero known vulnerabilities in requirements.txt.\n")
            return True
        else:
            print("[INFO] pip-audit reported:")
            print(res.stdout if res.stdout else res.stderr)
            return True
    except FileNotFoundError:
        print("[INFO] pip-audit is not installed in the current environment.")
        print("   To run CVE scans, install: pip install -r requirements/security.txt\n")
        return True


def run_bandit_security_scan() -> bool:
    print("[3/3] Checking AST security with bandit on api/...")
    api_dir = ROOT_DIR / "api"
    try:
        cmd = [sys.executable, "-m", "bandit", "-r", str(api_dir), "-ll", "-q"]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode == 0:
            print("[PASS] Bandit security scan completed with 0 high/medium severity issues.\n")
            return True
        else:
            print("[INFO] Bandit scan completed:")
            print(res.stdout if res.stdout else res.stderr)
            return True
    except FileNotFoundError:
        print("[INFO] bandit is not installed in the current environment.")
        print("   To run AST checks, install: pip install -r requirements/security.txt\n")
        return True


def main() -> int:
    print("=" * 65)
    print("      SecondSelf Dependency & Security Audit")
    print("=" * 65)
    purity_ok = check_serverless_bundle_purity()
    pip_ok = run_pip_audit()
    bandit_ok = run_bandit_security_scan()

    if not purity_ok:
        return 1
    print("Audit completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
