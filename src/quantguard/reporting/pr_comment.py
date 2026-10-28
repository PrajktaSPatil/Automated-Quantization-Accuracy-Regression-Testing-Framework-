"""Compact Markdown for a pull-request comment.

The marker lets CI find and update its own comment instead of adding a new one
on every push.
"""

from __future__ import annotations

from typing import Any

from quantguard.reporting.engine import _score, comparison_lookup

MARKER = "<!-- quantguard-report -->"
_LABELS = {
    "PASS": "PASS",
    "FAIL": "FAIL",
    "ERROR": "ERROR (infrastructure, not a quality verdict)",
    "WARNING": "WARNING (passes, needs a look)",
}


def render_pr_comment(payload: dict[str, Any], *, report_url: str | None = None) -> str:
    decision = payload["decision"]
    candidate = payload["candidate"]
    baseline = payload["baseline"]
    lines = [
        MARKER,
        f"## QuantGuard: {_LABELS.get(decision['status'], decision['status'])}",
        "",
        f"Candidate `{candidate['model_id']}` {candidate['version']} "
        f"({candidate.get('quantization_method') or candidate['precision']}) "
        f"vs baseline {baseline['version']} ({baseline['precision']}).",
        "",
    ]
    for reason in decision["reasons"][:8]:
        lines.append(f"- {reason}")
    if len(decision["reasons"]) > 8:
        lines.append(f"- ... and {len(decision['reasons']) - 8} more in the full report")

    lookup = comparison_lookup(payload)
    quality = [check for check in decision["checks"] if check["scope"] == "quality"]
    if quality:
        lines.extend(
            [
                "",
                "| Ref | Task | Metric | Reference | Candidate | Regression | Limit | Status |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for check in quality:
            reference = check.get("reference", "baseline")
            item = lookup.get((reference, check["task"], check["metric"]), {})
            lines.append(
                f"| {'prev' if reference == 'previous_release' else 'base'} | {check['task']} | {check['metric']} | "
                f"{_score(item.get('baseline_value'), item.get('baseline_stderr'))} | "
                f"{_score(item.get('candidate_value'), item.get('candidate_stderr'))} | "
                f"{check['observed_display']} | {check['threshold_display']} | {check['status'].upper()} |"
            )
    evaluation = payload.get("evaluation") or {}
    source = (payload.get("baseline_source") or {}).get("source", "live")
    lines.extend(
        [
            "",
            "<sub>"
            f"limit={evaluation.get('limit')} seed={evaluation.get('seed')} "
            f"batch={evaluation.get('batch_size')} device={evaluation.get('device')} "
            f"baseline={source} experiment={payload['experiment_id']}"
            "</sub>",
        ]
    )
    if report_url:
        lines.append(f"\n[Full report]({report_url})")
    return "\n".join(lines) + "\n"
