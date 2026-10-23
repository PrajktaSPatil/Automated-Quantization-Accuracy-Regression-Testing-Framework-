"""Apply configured thresholds. Task and metric policies override the default."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from quantguard.config.schema import MetricPolicy, PerformancePolicy, QualityPolicy, TaskPolicy
from quantguard.exceptions import MetricMissingError, PolicyConfigError
from quantguard.regression.engine import MetricComparison


class PolicyCheck(BaseModel):
    scope: Literal["quality", "performance"]
    task: str | None
    metric: str
    required: bool
    threshold_display: str
    observed_display: str
    status: Literal["pass", "fail", "missing"]
    reason: str


def resolve_metric_policy(
    quality: QualityPolicy,
    task: str,
    metric: str,
) -> MetricPolicy:
    """Precedence: task+metric, task, global metric, default."""

    task_policy: TaskPolicy | None = quality.tasks.get(task)
    if task_policy and metric in task_policy.metrics:
        return task_policy.metrics[metric]
    if task_policy is not None:
        return MetricPolicy(
            max_regression_pp=task_policy.max_regression_pp
            if task_policy.max_regression_pp is not None
            else quality.default.max_regression_pp,
            max_regression=task_policy.max_regression
            if task_policy.max_regression is not None
            else quality.default.max_regression,
            scale=task_policy.scale or quality.default.scale,
            higher_is_better=quality.default.higher_is_better,
            required=task_policy.required,
            allow_missing=task_policy.allow_missing,
        )
    if metric in quality.metrics:
        return quality.metrics[metric]
    return quality.default


def _quality_check(comparison: MetricComparison, policy: MetricPolicy) -> PolicyCheck:
    if comparison.status == "missing" or comparison.regression_amount is None:
        if policy.allow_missing:
            return PolicyCheck(
                scope="quality",
                task=comparison.task,
                metric=comparison.metric,
                required=False,
                threshold_display="missing allowed",
                observed_display="missing",
                status="missing",
                reason="metric missing and policy allows it",
            )
        raise MetricMissingError(
            f"Required metric {comparison.task}.{comparison.metric} is missing"
        )

    if policy.scale == "proportion":
        if comparison.regression_pp is None:
            raise PolicyConfigError(
                f"{comparison.task}.{comparison.metric} is not marked as a proportion metric"
            )
        observed = comparison.regression_pp
        threshold = policy.max_regression_pp
        if threshold is None:
            raise PolicyConfigError(
                f"{comparison.task}.{comparison.metric} has no max_regression_pp"
            )
        threshold_display = f"maximum {threshold:.2f} pp"
        observed_display = f"{observed:.2f} pp"
    else:
        observed = comparison.regression_amount
        threshold = policy.max_regression
        if threshold is None:
            raise PolicyConfigError(
                f"{comparison.task}.{comparison.metric} has no max_regression"
            )
        threshold_display = f"maximum {threshold:.6g}"
        observed_display = f"{observed:.6g}"

    failed = observed > threshold
    return PolicyCheck(
        scope="quality",
        task=comparison.task,
        metric=comparison.metric,
        required=policy.required,
        threshold_display=threshold_display,
        observed_display=observed_display,
        status="fail" if failed else "pass",
        reason=(
            f"regression {observed_display} exceeded {threshold_display}"
            if failed
            else f"regression {observed_display} within {threshold_display}"
        ),
    )


def evaluate_quality_policy(
    comparisons: list[MetricComparison],
    quality: QualityPolicy,
) -> list[PolicyCheck]:
    checks: list[PolicyCheck] = []
    for comparison in comparisons:
        policy = resolve_metric_policy(quality, comparison.task, comparison.metric)
        checks.append(_quality_check(comparison, policy))
    return checks


def evaluate_performance_policy(
    *,
    baseline: dict[str, float],
    candidate: dict[str, float],
    policy: PerformancePolicy,
    higher_is_better: dict[str, bool],
) -> list[PolicyCheck]:
    checks: list[PolicyCheck] = []
    for field_name in (
        "latency_ms",
        "throughput_seq_per_s",
        "gpu_memory_mib",
        "model_size_bytes",
    ):
        limit = getattr(policy, field_name)
        if limit is None:
            continue
        if field_name not in baseline or field_name not in candidate:
            raise MetricMissingError(f"Required performance metric {field_name} is missing")
        base = baseline[field_name]
        cand = candidate[field_name]
        if base == 0:
            raise PolicyConfigError(f"Baseline {field_name} is zero; relative change is undefined")
        if higher_is_better[field_name]:
            regression_percent = (base - cand) / abs(base) * 100.0
        else:
            regression_percent = (cand - base) / abs(base) * 100.0
        failed = regression_percent > limit.max_regression_percent
        checks.append(
            PolicyCheck(
                scope="performance",
                task=None,
                metric=field_name,
                required=limit.required,
                threshold_display=f"maximum {limit.max_regression_percent:.2f}% regression",
                observed_display=f"{regression_percent:.2f}% regression",
                status="fail" if failed else "pass",
                reason=(
                    f"{field_name} regressed {regression_percent:.2f}%"
                    if failed
                    else f"{field_name} regression {regression_percent:.2f}% is within policy"
                ),
            )
        )
    return checks
