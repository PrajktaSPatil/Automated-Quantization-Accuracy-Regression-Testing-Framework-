"""Direction-aware comparison of baseline and candidate metrics."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from quantguard.results.schema import NormalizedEvaluation, NormalizedMetric


class MetricComparison(BaseModel):
    task: str
    metric: str
    higher_is_better: bool
    baseline_value: float | None
    candidate_value: float | None
    absolute_difference: float | None
    regression_amount: float | None
    regression_pp: float | None
    relative_change: float | None
    status: Literal["improved", "unchanged", "regressed", "missing"]


def _index(evaluation: NormalizedEvaluation) -> dict[tuple[str, str], NormalizedMetric]:
    indexed: dict[tuple[str, str], NormalizedMetric] = {}
    for metric in evaluation.metrics:
        indexed[(metric.task, metric.name)] = metric
    return indexed


def _regression_amount(baseline: float, candidate: float, higher_is_better: bool) -> float:
    if higher_is_better:
        return baseline - candidate
    return candidate - baseline


def compare_evaluations(
    baseline: NormalizedEvaluation,
    candidate: NormalizedEvaluation,
    *,
    proportion_metrics: set[tuple[str, str]] | None = None,
) -> list[MetricComparison]:
    """Compare every metric present on either side.

    `regression_amount` is positive when the candidate is worse.
    `regression_pp` is set for proportion metrics and is percentage points.
    """

    proportion_metrics = proportion_metrics or set()
    left = _index(baseline)
    right = _index(candidate)
    comparisons: list[MetricComparison] = []
    for key in sorted(set(left) | set(right)):
        base = left.get(key)
        cand = right.get(key)
        if base is None or cand is None:
            present = base or cand
            assert present is not None
            comparisons.append(
                MetricComparison(
                    task=key[0],
                    metric=key[1],
                    higher_is_better=present.higher_is_better,
                    baseline_value=None if base is None else base.value,
                    candidate_value=None if cand is None else cand.value,
                    absolute_difference=None,
                    regression_amount=None,
                    regression_pp=None,
                    relative_change=None,
                    status="missing",
                )
            )
            continue
        if base.higher_is_better != cand.higher_is_better:
            comparisons.append(
                MetricComparison(
                    task=key[0],
                    metric=key[1],
                    higher_is_better=base.higher_is_better,
                    baseline_value=base.value,
                    candidate_value=cand.value,
                    absolute_difference=None,
                    regression_amount=None,
                    regression_pp=None,
                    relative_change=None,
                    status="missing",
                )
            )
            continue
        absolute = cand.value - base.value
        regression = _regression_amount(base.value, cand.value, base.higher_is_better)
        relative = None if base.value == 0 else absolute / abs(base.value)
        regression_pp = regression * 100.0 if key in proportion_metrics else None
        if regression > 0:
            status: Literal["improved", "unchanged", "regressed", "missing"] = "regressed"
        elif regression < 0:
            status = "improved"
        else:
            status = "unchanged"
        comparisons.append(
            MetricComparison(
                task=key[0],
                metric=key[1],
                higher_is_better=base.higher_is_better,
                baseline_value=base.value,
                candidate_value=cand.value,
                absolute_difference=absolute,
                regression_amount=regression,
                regression_pp=regression_pp,
                relative_change=relative,
                status=status,
            )
        )
    return comparisons
