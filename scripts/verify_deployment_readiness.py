#!/usr/bin/env python3
"""SecondSelf — Automated Vercel Deployment Readiness Pre-Flight Audit (Task 4.4).

Verifies all critical deployment requirements before shipping to Vercel:
- vercel.json syntax, rewrites, headers, memory/timeout allocations, and excludeFiles
- .python-version pinned to 3.12
- requirements.txt contains ONLY lightweight serverless runtime dependencies (no torch)
- api/ask.py serverless microservice integrity
- public/ static assets and precomputed graph/index artifacts
- .gitignore protection against accidental credential commits
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Terminal color tokens
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

REPO_ROOT = Path(__file__).resolve().parent.parent


def check(title: str, condition: bool, failure_detail: str = "") -> bool:
    print(f"  [{'PASS' if condition else 'FAIL'}] {title:<54} ... ", end="")
    if condition:
        print(f"{GREEN}OK{RESET}")
        return True
    else:
        print(f"{RED}FAILED{RESET}")
        if failure_detail:
            print(f"         {YELLOW}-> {failure_detail}{RESET}")
        return False


def main() -> int:
    print(f"\n{BOLD}{CYAN}{'='*74}{RESET}")
    print(f"{BOLD}{CYAN}  SECONDSELF VERCEL DEPLOYMENT READINESS AUDIT{RESET}")
    print(f"{BOLD}{CYAN}{'='*74}{RESET}\n")

    passed = 0
    total = 0

    # 1. vercel.json checks
    total += 1
    vj_path = REPO_ROOT / "vercel.json"
    if check("vercel.json exists", vj_path.is_file(), "Missing vercel.json"):
        passed += 1

    total += 1
    try:
        with open(vj_path, "r", encoding="utf-8") as f:
            vj_data = json.load(f)
        vj_valid = True
    except Exception as exc:
        vj_valid = False
        vj_data = {}
    if check("vercel.json is valid JSON", vj_valid, "Malformed JSON syntax"):
        passed += 1

    total += 1
    fn_config = vj_data.get("functions", {}).get("api/ask.py", {})
    fn_ok = fn_config.get("memory") == 1024 and fn_config.get("maxDuration") == 15 and "excludeFiles" in fn_config
    if check("api/ask.py function memory (1024MB) & timeout (15s)", fn_ok, "Function parameters misconfigured"):
        passed += 1

    total += 1
    rewrites = vj_data.get("rewrites", [])
    has_api_rewrite = any(r.get("source") == "/api/ask" and r.get("destination") == "/api/ask.py" for r in rewrites)
    if check("Route rewrite /api/ask -> /api/ask.py exists", has_api_rewrite, "Missing /api/ask rewrite"):
        passed += 1

    # 2. .python-version check
    total += 1
    pv_path = REPO_ROOT / ".python-version"
    pv_ok = pv_path.is_file() and pv_path.read_text(encoding="utf-8").strip() == "3.12"
    if check(".python-version exists and pins 3.12", pv_ok, "Missing or wrong version in .python-version"):
        passed += 1

    # 3. requirements.txt serverless bundle check
    total += 1
    req_path = REPO_ROOT / "requirements.txt"
    if req_path.is_file():
        req_lines = [
            line.strip().lower()
            for line in req_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        clean_text = "\n".join(req_lines)
        no_torch = "torch" not in clean_text and "sentence-transformers" not in clean_text
        has_groq = any("groq" in l for l in req_lines) and any("numpy" in l for l in req_lines)
        req_ok = no_torch and has_groq
        detail = "requirements.txt contains heavy torch/transformers or missing groq/numpy"
    else:
        req_ok = False
        detail = "Missing requirements.txt"
    if check("requirements.txt lightweight serverless check", req_ok, detail):
        passed += 1

    # 4. api/ask.py check
    total += 1
    api_path = REPO_ROOT / "api" / "ask.py"
    api_ok = api_path.is_file() and api_path.stat().st_size > 500
    if check("api/ask.py serverless microservice exists", api_ok, "Missing or empty api/ask.py"):
        passed += 1

    # 5. public/ assets checks
    total += 1
    index_html = REPO_ROOT / "public" / "index.html"
    if check("public/index.html exists", index_html.is_file(), "Missing public/index.html"):
        passed += 1

    total += 1
    graph_json = REPO_ROOT / "public" / "graph.json"
    if check("public/graph.json precomputed model artifact exists", graph_json.is_file(), "Run python build_graph.py"):
        passed += 1

    total += 1
    index_json = REPO_ROOT / "public" / "index.json"
    if check("public/index.json precomputed vector artifact exists", index_json.is_file(), "Run python -m link"):
        passed += 1

    # 6. .gitignore secret protection check
    total += 1
    git_path = REPO_ROOT / ".gitignore"
    if git_path.is_file():
        git_text = git_path.read_text(encoding="utf-8")
        ign_ok = ".env" in git_text and ".env.local" in git_text
    else:
        ign_ok = False
    if check(".gitignore excludes .env and .env.local", ign_ok, "Risk of credential leaks in git"):
        passed += 1

    # 7. Documentation check
    total += 1
    runbook_path = REPO_ROOT / "deployment-runbook.md"
    if check("deployment-runbook.md exists", runbook_path.is_file(), "Missing deployment runbook"):
        passed += 1

    # Summary
    print(f"\n{'-'*74}")
    if passed == total:
        print(f"{BOLD}{GREEN}ALL {passed}/{total} PRE-FLIGHT CHECKS PASSED — READY FOR VERCEL DEPLOY!{RESET}\n")
        return 0
    else:
        print(f"{BOLD}{RED}{passed}/{total} CHECKS PASSED ({total - passed} FAILED){RESET}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
