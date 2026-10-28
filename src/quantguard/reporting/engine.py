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
        "## Quality (candidate vs approved baseline)",
        "",
    ]
    lines.extend(_quality_table(payload["decision"]["checks"], payload.get("comparisons", []), "baseline"))
    previous = payload.get("previous_release")
    if previous:
        lines.extend(
            [
                "",
                f"## Quality (candidate vs previous release {previous.get('version')})",
                "",
            ]
        )
        lines.extend(
            _quality_table(
                payload["decision"]["checks"], payload.get("previous_release_comparisons", []), "previous_release"
            )
        )
    if payload.get("notes"):
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {note}" for note in payload["notes"])
    lines.append(f"\nBaseline source: {payload.get('baseline_source', 'evaluated')}")

    lines.extend(["", "## Performance", ""])
    base_perf = payload.get("baseline_performance") or {}
    cand_perf = payload.get("candidate_performance") or {}
    for key in ("model_size_bytes", "latency_ms_median", "throughput_seq_per_s", "gpu_memory_mib"):
        if key in base_perf and key in cand_perf:
            lines.append(
                f"- {key}: baseline {_fmt(base_perf[key])}, candidate {_fmt(cand_perf[key])}, "
                f"change {_percent_change(base_perf[key], cand_perf[key])}"
            )
    if base_perf.get("methodology"):
        lines.extend(["", "### Measurement methodology", "", "```json"])
        lines.append(json.dumps(base_perf["methodology"], indent=2, sort_keys=True))
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


def _quality_table(checks: list[dict[str, Any]], comparisons: list[dict[str, Any]], reference: str) -> list[str]:
    lines = [
        "| Task | Metric | Reference | Candidate | Regression | Noise | Policy | Status |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    lookup = {(item["task"], item["metric"]): item for item in comparisons}
    for check in checks:
        if check["scope"] != "quality" or check.get("reference", "baseline") != reference:
            continue
        comparison = lookup.get((check["task"], check["metric"]), {})
        lines.append(
            "| {task} | {metric} | {base} | {cand} | {observed} | {noise} | {threshold} | {status} |".format(
                task=check["task"],
                metric=check["metric"],
                base=_fmt(comparison.get("baseline_value")),
                cand=_fmt(comparison.get("candidate_value")),
                observed=check["observed_display"],
                noise=check.get("noise_display") or "n/a",
                threshold=check["threshold_display"],
                status=check["status"].upper(),
            )
        )
    return lines


def render_pr_comment(payload: dict[str, Any], *, run_url: str | None = None) -> str:
    """Compact Markdown for a pull-request comment. Starts with a marker so it can be updated in place."""

    decision = payload["decision"]
    candidate = payload["candidate"]
    headline = {
        "PASS": "PASS",
        "WARNING": "WARNING",
        "FAIL": "FAIL",
        "ERROR": "ERROR",
    }[decision["status"]]
    lines = [
        COMMENT_MARKER,
        f"## QuantGuard: {headline}",
        "",
        f"`{candidate['model_id']}` {candidate['version']} ({candidate.get('quantization_method') or candidate['precision']})"
        f" vs baseline {payload['baseline']['version']}",
        "",
    ]
    lines.extend(f"- {reason}" for reason in decision["reasons"][:8])
    quality = [c for c in decision["checks"] if c["scope"] == "quality"]
    if quality:
        lines.extend(["", "<details><summary>Quality checks</summary>", ""])
        lines.extend(_quality_table(decision["checks"], payload.get("comparisons", []), "baseline"))
        if payload.get("previous_release"):
            lines.extend(["", f"Previous release {payload['previous_release'].get('version')}:", ""])
            lines.extend(
                _quality_table(decision["checks"], payload.get("previous_release_comparisons", []), "previous_release")
            )
        lines.extend(["", "</details>"])
    perf = [c for c in decision["checks"] if c["scope"] == "performance"]
    if perf:
        lines.extend(["", "<details><summary>Performance checks</summary>", ""])
        lines.extend(f"- [{c['status'].upper()}] {c['metric']}: {c['reason']}" for c in perf)
        lines.extend(["", "</details>"])
    for note in payload.get("notes", []):
        lines.append(f"\n> {note}")
    evaluation = payload.get("evaluation", {})
    lines.extend(
        [
            "",
            f"Tasks: {', '.join(evaluation.get('tasks', []))}; limit: {evaluation.get('limit') or 'full'}; "
            f"seed: {evaluation.get('seed')}; config `{payload['config_hash'][:12]}`",
        ]
    )
    if run_url:
        lines.append(f"[Full report and artifacts]({run_url})")
    return "\n".join(lines) + "\n"


COMMENT_MARKER = "<!-- quantguard-report -->"


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


def _score(value: float | None, stderr: float | None = None) -> str:
    if value is None:
        return "missing"
    return f"{value:.6g}" + (f" ± {stderr:.3g}" if stderr is not None else "")

def comparison_lookup(payload: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    refs = {"baseline": "baseline", "previous_release": "previous_release"}
    result = {}
    for row in payload.get("comparisons", []):
        ref = refs.get(row.get("reference", "baseline"), "baseline")
        result[(ref, row.get("task"), row.get("metric"))] = row
    return result
