"""SecondSelf — Verification Report Formatter."""

from __future__ import annotations

from validation.models import VerificationReport


def format_terminal_report(report: VerificationReport) -> str:
    """Formats a VerificationReport for human-readable terminal display."""
    lines: list[str] = [
        "=" * 68,
        f"        SecondSelf Quality Gate — {report.phase}",
        "=" * 68,
        f"Run ID:        {report.run_id}",
        f"Timestamp:     {report.timestamp}",
        f"Overall Status: {report.overall_status} (Exit Code: {report.exit_code})",
        f"Summary:       {report.summary.passed_checks}/{report.summary.total_checks} checks passed "
        f"({report.summary.failed_checks} failed, {report.summary.warning_count} warnings)",
        "-" * 68,
        "Validator Breakdown:",
    ]

    for name, res in report.results.items():
        symbol = "✓" if res.passed else "✗"
        status_text = "PASS" if res.passed else "FAIL"
        lines.append(f"  [{symbol}] {name.ljust(32)} {status_text} ({res.checks_run} checks)")
        for err in res.errors:
            lines.append(f"      • ERROR: {err}")
        for warn in res.warnings:
            lines.append(f"      • WARN:  {warn}")

    lines.append("=" * 68)
    return "\n".join(lines)
