# SecondSelf — Frontend Manual QA Checklist & CI Gating Guide (Task 3.3)

This document provides a repeatable, human-verifiable QA checklist and automated CI gating guide for the **SecondSelf Static Knowledge Graph Explorer** (`public/`).

When browser automation (e.g. Playwright) is not installed or when performing release qualification before production deployment, execute the following manual test scenarios.

---

## 1. Quickstart: Launch Preview Server

Run the preview script from the workspace root:

```bash
# Starts the server on http://127.0.0.1:8000 and stages artifacts into public/data/
python scripts/preview.py --port 8000

# Or execute automated verification immediately:
python scripts/preview.py --verify
```

---

## 2. Manual QA Verification Matrix

| # | Feature / Test Flow | Step-by-Step Instructions | Expected Behavior | Verdict |
|---|---------------------|---------------------------|-------------------|---------|
| **1** | **Initial Page Load** | 1. Open `http://localhost:8000` in browser.<br>2. Observe initial load state. | • Loading spinner appears briefly.<br>• Header displays "SecondSelf Graph v1.0".<br>• Stats strip shows total counts: Notes $\ge 24$, Links $\ge 70$.<br>• Canvas displays force-directed network graph without console errors. | `[ ] PASS` |
| **2** | **Interactive Graph Canvas** | 1. Click and drag background.<br>2. Scroll mouse wheel / pinch.<br>3. Click `+` and `−` floating buttons.<br>4. Click Fit button (`⛶`).<br>5. Click Freeze Physics button (`⚡`). | • Canvas pans smoothly.<br>• Graph zooms in and out cleanly.<br>• Fit button re-centers and scales entire graph to viewport.<br>• Physics freeze stops node drift. | `[ ] PASS` |
| **3** | **Neighborhood Highlighting** | 1. Click any circular Note node (e.g., `Idea for SecondSelf...`). | • Selected node and 1st-degree neighbors remain vibrant.<br>• All non-connected nodes and edges dim to $\approx 10\%$ opacity.<br>• Right-hand Detail Drawer slides open (`440px`). | `[ ] PASS` |
| **4** | **Detail Drawer Inspection** | 1. Inspect metadata inside open drawer.<br>2. Check PARA badge, confidence score, rationale blockquote, tags, and provenance. | • PARA Category pill displays correct color (Projects: Amber, Areas: Blue, Resources: Green, Archives: Slate).<br>• Confidence meter displays percentage bar.<br>• Rationale explains deterministic classification logic.<br>• SHA-256 hash and file path are visible. | `[ ] PASS` |
| **5** | **Backlink Navigation** | 1. In the open drawer, scroll to "Connected Relationships".<br>2. Click any connected node (e.g., `Resources` or `text`). | • Drawer instantly updates with selected backlink's metadata.<br>• Canvas camera smoothly pans and centers on the target node. | `[ ] PASS` |
| **6** | **Search Filtering** | 1. Press `/` keyboard shortcut.<br>2. Type `force-directed`.<br>3. Click the `×` clear button. | • Search input receives focus immediately.<br>• Graph and list real-time filter to matching notes.<br>• Live counter announces `Showing 1 of N nodes`.<br>• Clear button restores all nodes. | `[ ] PASS` |
| **7** | **PARA Category Toggle** | 1. Click `Resources` pill in filter bar.<br>2. Observe graph and counter. | • `Resources` pill toggles `aria-pressed="false"`.<br>• Resource nodes and connected edges fade from view.<br>• Counter updates accordingly. | `[ ] PASS` |
| **8** | **Confidence Threshold Slider**| 1. Drag `Min Conf` slider to `0.50`. | • Notes with confidence $< 0.50$ are hidden.<br>• Value badge displays `0.50`. | `[ ] PASS` |
| **9** | **Reset Filters Action** | 1. With search active and categories deselected, click "Reset Filters". | • Search input clears.<br>• All four category pills activate (`aria-pressed="true"`).<br>• Confidence slider resets to `0.00`.<br>• Full corpus returns to view. | `[ ] PASS` |
| **10**| **View Switcher (Card List)** | 1. Click "List" tab in top header.<br>2. Navigate cards with `Tab` and press `Enter`. | • Canvas smoothly hides; card grid appears.<br>• Cards show title, category, confidence progress bar, tags, and snippet.<br>• Pressing `Enter` on a card opens its detail drawer. | `[ ] PASS` |
| **11**| **Keyboard Accessibility** | 1. Open detail drawer.<br>2. Press `Escape`.<br>3. Inspect focus state. | • Detail drawer slides closed.<br>• Focus restores to the previously active element.<br>• Visible high-contrast focus rings appear on all focused elements. | `[ ] PASS` |
| **12**| **Responsive Mobile Layout** | 1. Open Chrome/Firefox DevTools responsive mode ($375\text{px}$ width).<br>2. Open drawer. | • Top header stacks cleanly.<br>• Graph legend auto-collapses to save screen real estate.<br>• Detail drawer expands to $100\text{vw}$ full-width with large touch-friendly close button. | `[ ] PASS` |
| **13**| **Graceful Failure State** | 1. Stop server.<br>2. In a temporary test, remove `public/data/graph.json`.<br>3. Reload page. | • UI does NOT show a blank page.<br>• User-facing Error Overlay displays warning icon, error description, and working "Retry Loading" button. | `[ ] PASS` |

---

## 3. Automated CI Quality Gate Integration

In continuous integration (GitHub Actions, GitLab CI, or pre-commit hooks), run `scripts/verify_frontend.py` before releasing to GitHub Pages, Netlify, or Vercel:

```bash
# Automated run: starts managed server, runs all 5 checks, saves report, returns CI exit code
python scripts/verify_frontend.py --report verify-artifacts/verify-report.json
```

### Exit Codes:
- **`0`**: **All checks passed.** Clean quality gate; safe to deploy static site.
- **`1`**: **Verification failure.** Schema violation, missing data fields, broken 404 assets, or smoke test failure. CI pipeline halts.
- **`2`**: **Setup error.** Missing Task 3.1 source artifacts or port binding conflict.

### Example GitHub Actions Workflow Step:
```yaml
- name: Verify Static Frontend Quality Gate
  run: |
    python scripts/verify_frontend.py --report verify-artifacts/verify-report.json
  
- name: Upload Verification Evidence
  if: always()
  uses: actions/upload-artifact@v4
  with:
    name: frontend-verification-evidence
    path: verify-artifacts/
```

---

## 4. How to Add a New Verification Check

1. Open [`preview/verifier.py`](file:///d:/Second-Self/preview/verifier.py).
2. Define a new check method:
   ```python
   def verify_custom_rule(self) -> CheckResult:
       start = time.monotonic()
       errors: list[str] = []
       # Your custom verification logic here...
       return CheckResult(
           name="custom_rule",
           status="FAIL" if errors else "PASS",
           description="Verifies custom business logic or SEO tags",
           duration_ms=(time.monotonic() - start) * 1000,
           errors=errors,
       )
   ```
3. Call `self.verify_custom_rule()` in `FrontendVerifier.run_all()` and register the result into `report.add_check()`.
4. The result will automatically appear in `verify-report.json` and the formatted console summary.
