
"""Convert lm-eval output into NormalizedEvaluation."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from quantguard.exceptions import EvaluationError
from quantguard.results.schema import NormalizedEvaluation, NormalizedMetric

_SKIP_NAMES = {"alias", "name", "sample_len"}


def _utc_now() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _split_metric_key(key: str) -> str | None:
    """Extract the base metric name from an lm-eval result key."""
    if key in _SKIP_NAMES:
        return None

    metric_name, _, metric_filter = key.partition(",")

    if metric_name.endswith("_stderr"):
        return None

    if metric_filter and metric_filter != "none":
        return f"{metric_name},{metric_filter}"
    return metric_name


def normalize_lm_eval(
    raw: dict[str, Any],
    *,
    model_id: str,
    model_version: str,
    evaluation_config_hash: str,
    backend_version: str,
    timestamp: str | None = None,
) -> NormalizedEvaluation:
    """Normalize lm-eval results and preserve available standard errors."""
    results = raw.get("results")

    if not isinstance(results, dict):
        raise EvaluationError(
            "lm-eval output is missing a results object"
        )

    higher_map = raw.get("higher_is_better") or {}

    if not isinstance(higher_map, dict):
        raise EvaluationError(
            "lm-eval higher_is_better must be an object when present"
        )

    stamp = timestamp or _utc_now()
    metrics: list[NormalizedMetric] = []

    for task, task_results in results.items():
        if not isinstance(task_results, dict):
            raise EvaluationError(
                f"lm-eval results for task {task} are not an object"
            )

        task_directions = higher_map.get(task) or {}

        if not isinstance(task_directions, dict):
            raise EvaluationError(
                f"higher_is_better for task {task} is not an object"
            )

        # Collect standard errors, e.g. acc_stderr,none -> acc.
        standard_errors: dict[str, float] = {}
        raw_counts = raw.get("n-samples") or raw.get("n_samples") or {}
        counts = raw_counts.get(task) if isinstance(raw_counts, dict) else None
        if isinstance(counts, dict):
            counts = counts.get("effective", counts.get("original"))
        n_samples = counts if isinstance(counts, int) and not isinstance(counts, bool) else None
        raw_task_samples = ((raw.get("per_sample") or {}).get(task) or {})
        task_samples: dict[str, dict[str, float]] = {}
        for sample_name, sample_values in raw_task_samples.items():
            if isinstance(sample_values, dict):
                try:
                    task_samples[str(sample_name)] = {str(k): float(v) for k, v in sample_values.items()}
                except (TypeError, ValueError):
                    continue

        for raw_key, raw_value in task_results.items():
            key = str(raw_key)
            base_name = key.split(",", 1)[0]

            if not base_name.endswith("_stderr"):
                continue

            metric_name = base_name.removesuffix("_stderr")
            metric_filter = key.split(",", 1)[1] if "," in key else None
            if metric_filter is not None and metric_filter != "none":
                metric_name = f"{metric_name},{metric_filter}"

            if (
                isinstance(raw_value, (int, float))
                and not isinstance(raw_value, bool)
            ):
                error_value = float(raw_value)

                if error_value >= 0:
                    standard_errors[metric_name] = error_value

        # Normalize regular metrics; standard errors are attached to them.
        for key, value in task_results.items():
            metric_name = _split_metric_key(str(key))

            if metric_name is None:
                continue

            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue

            direction_name = metric_name.split(",", 1)[0]
            if direction_name not in task_directions:
                raise EvaluationError(
                    f"lm-eval did not report higher_is_better "
                    f"for {task}.{direction_name}"
                )

            metrics.append(
                NormalizedMetric(
                    name=metric_name,
                    value=float(value),
                    stderr=standard_errors.get(metric_name),
                    n_samples=n_samples,
                    samples=task_samples.get(metric_name, task_samples.get(metric_name.split(",", 1)[0])),
                    higher_is_better=bool(task_directions[direction_name]),
                    task=str(task),
                    model_id=model_id,
                    model_version=model_version,
                    evaluation_config_hash=evaluation_config_hash,
                    timestamp=stamp,
                )
            )

    if not metrics:
        raise EvaluationError(
            "lm-eval output contained no numeric task metrics"
        )

    return NormalizedEvaluation(
        backend="lm_eval",
        backend_version=backend_version,
        model_id=model_id,
        model_version=model_version,
        evaluation_config_hash=evaluation_config_hash,
        metrics=metrics,
        per_sample={str(task): {str(metric): values for metric, values in (raw.get("per_sample", {}).get(task, {}) or {}).items() if isinstance(values, dict)} for task in results},
        metadata={
            "versions": raw.get("versions"),
            "n_shot": raw.get("n-shot") or raw.get("n_shot"),
            "config": raw.get("config"),
        },
    )
