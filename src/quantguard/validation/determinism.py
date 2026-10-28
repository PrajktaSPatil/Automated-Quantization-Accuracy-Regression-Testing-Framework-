"""Run-to-run variance: evaluate the same model repeatedly and compare scores."""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

from quantguard.config.loader import config_hash
from quantguard.config.schema import ModelSpec, QuantGuardConfig
from quantguard.evaluators.base import EvaluationBackend
from quantguard.metadata.environment import collect_environment
from quantguard.results.normalizer import normalize_lm_eval


def measure_determinism(
    evaluator: EvaluationBackend,
    config: QuantGuardConfig,
    spec: ModelSpec,
    *,
    runs: int,
    work_dir: Path,
    repo_root: Path,
) -> dict[str, Any]:
    if runs < 2:
        raise ValueError("determinism needs at least 2 runs")
    environment = collect_environment(repo_root)
    digest = config_hash(config)
    values: dict[tuple[str, str], list[float]] = {}
    for index in range(runs):
        raw = evaluator.evaluate(spec, config.evaluation, environment, work_dir / f"run_{index}.json")
        normalized = normalize_lm_eval(
            raw,
            model_id=spec.model_id,
            model_version=spec.version,
            evaluation_config_hash=digest,
            backend_version=str((raw.get("quantguard") or {}).get("backend_version")),
        )
        for metric in normalized.metrics:
            values.setdefault((metric.task, metric.name), []).append(metric.value)
    report = []
    for (task, name), series in sorted(values.items()):
        report.append(
            {
                "task": task,
                "metric": name,
                "runs": len(series),
                "min": min(series),
                "max": max(series),
                "max_abs_difference": max(series) - min(series),
                "stdev": statistics.pstdev(series),
                "identical": len(set(series)) == 1,
            }
        )
    return {
        "runs": runs,
        "all_identical": all(item["identical"] for item in report),
        "metrics": report,
    }
