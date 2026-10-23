"""Convert lm-eval output into NormalizedEvaluation."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from quantguard.exceptions import EvaluationError
from quantguard.results.schema import NormalizedEvaluation, NormalizedMetric

_SKIP_SUFFIXES = ("_stderr",)
_SKIP_NAMES = {"alias"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _split_metric_key(key: str) -> str | None:
    if key in _SKIP_NAMES:
        return None
    metric_name = key.split(",", 1)[0]
    if metric_name.endswith(_SKIP_SUFFIXES) or not isinstance(metric_name, str):
        return None
    if metric_name.endswith("_stderr"):
        return None
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
    if "results" not in raw or not isinstance(raw["results"], dict):
        raise EvaluationError("lm-eval output is missing a results object")

    higher_map = raw.get("higher_is_better") or {}
    if not isinstance(higher_map, dict):
        raise EvaluationError("lm-eval higher_is_better must be an object when present")

    stamp = timestamp or _utc_now()
    metrics: list[NormalizedMetric] = []
    for task, task_results in raw["results"].items():
        if not isinstance(task_results, dict):
            raise EvaluationError(f"lm-eval results for task {task} are not an object")
        task_directions = higher_map.get(task) or {}
        if task_directions and not isinstance(task_directions, dict):
            raise EvaluationError(f"higher_is_better for task {task} is not an object")
        for key, value in task_results.items():
            metric_name = _split_metric_key(str(key))
            if metric_name is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if metric_name not in task_directions:
                raise EvaluationError(
                    f"lm-eval did not report higher_is_better for {task}.{metric_name}"
                )
            metrics.append(
                NormalizedMetric(
                    name=metric_name,
                    value=float(value),
                    higher_is_better=bool(task_directions[metric_name]),
                    task=str(task),
                    model_id=model_id,
                    model_version=model_version,
                    evaluation_config_hash=evaluation_config_hash,
                    timestamp=stamp,
                )
            )

    if not metrics:
        raise EvaluationError("lm-eval output contained no numeric task metrics")

    return NormalizedEvaluation(
        backend="lm_eval",
        backend_version=backend_version,
        model_id=model_id,
        model_version=model_version,
        evaluation_config_hash=evaluation_config_hash,
        metrics=metrics,
        metadata={
            "versions": raw.get("versions"),
            "n_shot": raw.get("n-shot") or raw.get("n_shot"),
            "config": raw.get("config"),
        },
    )
