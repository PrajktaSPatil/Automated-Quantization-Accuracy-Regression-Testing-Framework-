"""Direction-aware comparison of baseline and candidate metrics."""

from __future__ import annotations

import math
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
    regression_relative_pct: float | None = None
    baseline_stderr: float | None = None
    candidate_stderr: float | None = None
    stderr_diff: float | None = None
    difference_stderr: float | None = None
    stderr_method: Literal["paired", "independent"] | None = None
    paired: bool = False
    status: Literal["improved", "unchanged", "regressed", "missing"]


def _index(evaluation: NormalizedEvaluation) -> dict[tuple[str, str], NormalizedMetric]:
    indexed: dict[tuple[str, str], NormalizedMetric] = {}
    for metric in evaluation.metrics:
        indexed[(metric.task, metric.name)] = metric
    return indexed


def combined_stderr(base: float | None, cand: float | None) -> float | None:
    """Standard error of (candidate - baseline) assuming independent estimates.

    Both models are scored on the same documents, so their errors are positively
    correlated and the true standard error of the difference is smaller. Treating
    them as independent is the conservative choice: it can only hide small
    regressions, never invent one.
    """

    if base is None or cand is None:
        return None
    return math.sqrt(base * base + cand * cand)


def paired_stderr(
    baseline: NormalizedEvaluation,
    candidate: NormalizedEvaluation,
    task: str,
    metric: str,
    base_value: float,
    cand_value: float,
) -> float | None:
    """Standard error of the mean per-document difference, or None when pairing is not valid.

    Valid only when both runs scored exactly the same documents and the per-document scores
    average to the reported metric (so the metric really is a mean of those scores).
    """

    left = baseline.per_sample.get(task, {}).get(metric)
    right = candidate.per_sample.get(task, {}).get(metric)
    if not left or not right or left.keys() != right.keys() or len(left) < 2:
        return None
    ids = list(left)
    a = [left[i] for i in ids]
    b = [right[i] for i in ids]
    if abs(sum(a) / len(a) - base_value) > 1e-6 or abs(sum(b) / len(b) - cand_value) > 1e-6:
        return None
    diffs = [x - y for x, y in zip(a, b)]
    mean = sum(diffs) / len(diffs)
    variance = sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1)
    return math.sqrt(variance / len(diffs))


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
        regression_relative_pct = None if base.value == 0 else regression / abs(base.value) * 100.0
        paired_se = paired_stderr(baseline, candidate, key[0], key[1], base.value, cand.value)
        stderr_diff = paired_se if paired_se is not None else combined_stderr(base.stderr, cand.stderr)
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
                regression_relative_pct=regression_relative_pct,
                baseline_stderr=base.stderr,
                candidate_stderr=cand.stderr,
                stderr_diff=stderr_diff,
                paired=paired_se is not None,
                status=status,
            )
        )
    return comparisons


def paired_difference_stderr(base: NormalizedMetric, cand: NormalizedMetric) -> float | None:
    """Compatibility helper: paired SE for two aligned per-example metric vectors."""
    if not base.samples or not cand.samples or base.samples.keys() != cand.samples.keys() or len(base.samples) < 30:
        return None
    if abs(sum(base.samples.values()) / len(base.samples) - base.value) > 1e-6 or abs(sum(cand.samples.values()) / len(cand.samples) - cand.value) > 1e-6:
        return None
    diffs = [base.samples[k] - cand.samples[k] for k in base.samples]
    mean = sum(diffs) / len(diffs)
    return math.sqrt(sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1) / len(diffs))
