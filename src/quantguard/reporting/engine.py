"""JSON, Markdown, and HTML reports from one experiment payload."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

from quantguard.logging import get_logger, log_event

logger = get_logger("reporting")


def write_reports(payload: dict[str, Any], directory: Path) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "report.json"
    md_path = directory / "report.md"
    html_path = directory / "report.html"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    markdown = render_markdown(payload)
    md_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(_render_html(markdown, payload), encoding="utf-8")
    log_event(logger, "REPORT_GENERATED", "wrote release reports", directory=str(directory))
    return {"json": str(json_path), "markdown": str(md_path), "html": str(html_path)}


def render_markdown(payload: dict[str, Any]) -> str:
    baseline = payload["baseline"]
    candidate = payload["candidate"]
    decision = payload["decision"]
    lines = [
        "# QUANTGUARD RELEASE REPORT",
        "",
        f"Experiment: `{payload['experiment_id']}`",
        f"Created: {payload['created_at']}",
        f"Config hash: `{payload['config_hash']}`",
        "",
        "## Baseline",
        "",
        f"{baseline['model_id']} / {baseline['version']} / {baseline['precision']}",
        f"Revision: `{baseline.get('revision')}`",
        f"Quantization: {baseline.get('quantization_method')}",
        f"Artifact: `{baseline['artifact_uri']}`",
        f"Status: {baseline['status']}",
        "",
        "## Candidate",
        "",
        f"{candidate['model_id']} / {candidate['version']} / {candidate['precision']}",
        f"Revision: `{candidate.get('revision')}`",
        f"Quantization: {candidate.get('quantization_method')}",
        f"Source: {candidate.get('source_model_id')} / {candidate.get('source_version')}",
        f"Artifact: `{candidate['artifact_uri']}`",
        "",
        "## Quality",
        "",
        "| Task | Metric | Baseline | Candidate | Regression | Policy | Status |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    comparisons = {
        (item["task"], item["metric"]): item for item in payload.get("comparisons", [])
    }
    for check in payload["decision"]["checks"]:
        if check["scope"] != "quality":
            continue
        comparison = comparisons.get((check["task"], check["metric"]), {})
        lines.append(
            "| {task} | {metric} | {base} | {cand} | {observed} | {threshold} | {status} |".format(
                task=check["task"],
                metric=check["metric"],
                base=_fmt(comparison.get("baseline_value")),
                cand=_fmt(comparison.get("candidate_value")),
                observed=check["observed_display"],
                threshold=check["threshold_display"],
                status=check["status"].upper(),
            )
        )

    lines.extend(["", "## Performance", ""])
    base_perf = payload.get("baseline_performance") or {}
    cand_perf = payload.get("candidate_performance") or {}
    for key in ("model_size_bytes", "latency_ms_median", "throughput_seq_per_s", "gpu_memory_mib"):
        if key in base_perf and key in cand_perf:
            lines.append(
                f"- {key}: baseline {_fmt(base_perf[key])}, candidate {_fmt(cand_perf[key])}, "
                f"change {_percent_change(base_perf[key], cand_perf[key])}"
            )
    if payload.get("baseline_performance", {}).get("methodology"):
        lines.extend(["", "### Measurement methodology", "", "```json"])
        lines.append(json.dumps(payload["baseline_performance"]["methodology"], indent=2, sort_keys=True))
        lines.append("```")

    lines.extend(
        [
            "",
            "## Policies",
            "",
        ]
    )
    for check in decision["checks"]:
        lines.append(
            f"- [{check['status'].upper()}] {check['scope']} {check['metric']}: {check['reason']}"
        )

    lines.extend(
        [
            "",
            "## Final release gate",
            "",
            decision["status"],
            "",
            "Reasons:",
            "",
        ]
    )
    for reason in decision["reasons"]:
        lines.append(f"- {reason}")

    env = payload.get("environment") or {}
    lines.extend(
        [
            "",
            "## Environment",
            "",
            f"- Python: {env.get('python_version')}",
            f"- Platform: {env.get('platform')}",
            f"- Git SHA: `{env.get('git_sha')}`",
            f"- Torch: {env.get('torch_version')}",
            f"- CUDA: {env.get('cuda_version')}",
            f"- GPU: {env.get('gpu_name')} ({env.get('gpu_total_mib')} MiB, cc {env.get('compute_capability')})",
            f"- Packages: `{json.dumps(env.get('packages') or {}, sort_keys=True)}`",
            "",
        ]
    )
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _percent_change(baseline: float, candidate: float) -> str:
    if baseline == 0:
        return "undefined"
    change = (candidate - baseline) / abs(baseline) * 100.0
    return f"{change:+.2f}%"


def _render_html(markdown: str, payload: dict[str, Any]) -> str:
    status = escape(str(payload["decision"]["status"]))
    body = escape(markdown)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>QuantGuard {status}</title>
  <style>
    body {{ font-family: Georgia, serif; margin: 2rem auto; max-width: 60rem; line-height: 1.45; }}
    pre {{ white-space: pre-wrap; background: #f6f6f6; padding: 1rem; }}
    .status {{ font-weight: 700; }}
  </style>
</head>
<body>
  <p class="status">Gate: {status}</p>
  <pre>{body}</pre>
</body>
</html>
"""
