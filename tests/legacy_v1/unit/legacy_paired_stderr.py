import math

import pytest

from quantguard.regression.engine import compare_evaluations, paired_difference_stderr
from tests.unit.fixtures import evaluation, metric


def _metric(values: dict[str, float], stderr: float):
    item = metric("t", "acc", sum(values.values()) / len(values))
    item.samples = values
    item.stderr = stderr
    return item


def test_paired_stderr_is_smaller_than_independent_for_correlated_models() -> None:
    n = 400
    base = {str(i): float(i % 4 != 0) for i in range(n)}
    cand = dict(base)
    for i in range(0, n, 40):  # flip 10 answers
        cand[str(i + 1)] = 1.0 - cand[str(i + 1)]
    left, right = _metric(base, 0.022), _metric(cand, 0.022)
    paired = paired_difference_stderr(left, right)
    independent = math.sqrt(0.022**2 + 0.022**2)
    assert paired is not None and paired < independent / 3
    comparison = compare_evaluations(evaluation(left), evaluation(right))[0]
    assert comparison.stderr_method == "paired"
    assert comparison.difference_stderr == paired


def test_falls_back_to_independent_without_samples() -> None:
    left, right = metric("t", "acc", 0.8), metric("t", "acc", 0.7)
    left.stderr = right.stderr = 0.01
    comparison = compare_evaluations(evaluation(left), evaluation(right))[0]
    assert comparison.stderr_method == "independent"
    assert comparison.difference_stderr == pytest.approx(math.sqrt(2) * 0.01)


def test_paired_refused_when_reported_score_is_not_the_sample_mean() -> None:
    base = {str(i): float(i % 2) for i in range(100)}
    left, right = _metric(base, 0.05), _metric(dict(base), 0.05)
    left.value = 0.9  # e.g. a metric that is not a plain mean (perplexity, F1)
    assert paired_difference_stderr(left, right) is None


def test_paired_refused_with_too_few_or_mismatched_examples() -> None:
    small = {str(i): 1.0 for i in range(10)}
    assert paired_difference_stderr(_metric(small, 0.1), _metric(dict(small), 0.1)) is None
    a = {str(i): 1.0 for i in range(100)}
    b = {str(i): 1.0 for i in range(1, 101)}
    assert paired_difference_stderr(_metric(a, 0.1), _metric(b, 0.1)) is None
