from tests.unit.fixtures import evaluation, metric
from quantguard.regression.engine import compare_evaluations


def test_higher_is_better_drop_is_regression() -> None:
    comparisons = compare_evaluations(
        evaluation(metric("mmlu", "acc", 0.847)),
        evaluation(metric("mmlu", "acc", 0.841)),
        proportion_metrics={("mmlu", "acc")},
    )
    assert comparisons[0].status == "regressed"
    assert comparisons[0].regression_amount == pytest_approx(0.006)
    assert comparisons[0].regression_pp == pytest_approx(0.6)


def test_lower_is_better_increase_is_regression() -> None:
    comparisons = compare_evaluations(
        evaluation(metric("latency", "seconds", 1.0, higher_is_better=False)),
        evaluation(metric("latency", "seconds", 1.2, higher_is_better=False)),
    )
    assert comparisons[0].status == "regressed"
    assert comparisons[0].regression_amount == pytest_approx(0.2)
    assert comparisons[0].regression_pp is None


def test_lower_is_better_decrease_is_improvement() -> None:
    comparisons = compare_evaluations(
        evaluation(metric("latency", "seconds", 1.0, higher_is_better=False)),
        evaluation(metric("latency", "seconds", 0.8, higher_is_better=False)),
    )
    assert comparisons[0].status == "improved"
    assert comparisons[0].absolute_difference == pytest_approx(-0.2)


def test_missing_metric_is_reported() -> None:
    comparisons = compare_evaluations(
        evaluation(metric("mmlu", "acc", 0.5), metric("gsm8k", "exact_match", 0.4)),
        evaluation(metric("mmlu", "acc", 0.5)),
    )
    missing = [item for item in comparisons if item.status == "missing"]
    assert [item.metric for item in missing] == ["exact_match"]


def pytest_approx(value: float):
    import pytest

    return pytest.approx(value)
