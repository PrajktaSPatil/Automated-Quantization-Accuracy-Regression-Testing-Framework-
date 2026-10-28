import pytest

from quantguard.config.schema import MetricPolicy, PerformanceLimit, PerformancePolicy, QualityPolicy, TaskPolicy
from quantguard.exceptions import MetricMissingError
from quantguard.policy.engine import evaluate_performance_policy, evaluate_quality_policy, resolve_metric_policy
from quantguard.regression.engine import compare_evaluations
from tests.unit.fixtures import evaluation, metric


def _quality() -> QualityPolicy:
    return QualityPolicy(
        default=MetricPolicy(scale="proportion", max_regression_pp=0.5),
        tasks={
            "gsm8k": TaskPolicy(scale="proportion", max_regression_pp=1.0),
            "arc_easy": TaskPolicy(
                metrics={
                    "acc_norm": MetricPolicy(scale="proportion", max_regression_pp=0.2),
                }
            ),
        },
        metrics={"exact_match": MetricPolicy(scale="proportion", max_regression_pp=1.5)},
    )


def test_task_metric_overrides_task_and_default() -> None:
    policy = resolve_metric_policy(_quality(), "arc_easy", "acc_norm")
    assert policy.max_regression_pp == 0.2


def test_task_override_beats_default() -> None:
    policy = resolve_metric_policy(_quality(), "gsm8k", "exact_match")
    assert policy.max_regression_pp == 1.0


def test_global_metric_used_when_task_has_no_override() -> None:
    policy = resolve_metric_policy(_quality(), "mmlu", "exact_match")
    assert policy.max_regression_pp == 1.5


def test_threshold_fail_and_pass() -> None:
    quality = QualityPolicy(default=MetricPolicy(scale="proportion", max_regression_pp=0.5))
    fail_case = compare_evaluations(
        evaluation(metric("mmlu", "acc", 0.847)),
        evaluation(metric("mmlu", "acc", 0.841)),
        proportion_metrics={("mmlu", "acc")},
    )
    pass_case = compare_evaluations(
        evaluation(metric("mmlu", "acc", 0.847)),
        evaluation(metric("mmlu", "acc", 0.844)),
        proportion_metrics={("mmlu", "acc")},
    )
    assert evaluate_quality_policy(fail_case, quality)[0].status == "fail"
    assert evaluate_quality_policy(pass_case, quality)[0].status == "pass"


def test_missing_required_metric_raises() -> None:
    quality = QualityPolicy(default=MetricPolicy(scale="proportion", max_regression_pp=0.5))
    comparisons = compare_evaluations(
        evaluation(metric("mmlu", "acc", 0.5)),
        evaluation(metric("gsm8k", "acc", 0.5)),
    )
    with pytest.raises(MetricMissingError):
        evaluate_quality_policy(comparisons, quality)


def test_missing_metric_can_be_allowed() -> None:
    quality = QualityPolicy(
        default=MetricPolicy(scale="proportion", max_regression_pp=0.5, required=False, allow_missing=True)
    )
    comparisons = compare_evaluations(
        evaluation(metric("mmlu", "acc", 0.5)),
        evaluation(metric("gsm8k", "acc", 0.5)),
    )
    checks = evaluate_quality_policy(comparisons, quality)
    assert all(check.status == "missing" for check in checks)


def test_performance_direction() -> None:
    policy = PerformancePolicy(
        latency_ms=PerformanceLimit(max_regression_percent=10, required=True),
        throughput_seq_per_s=PerformanceLimit(max_regression_percent=10, required=True),
    )
    checks = evaluate_performance_policy(
        baseline={"latency_ms": 100.0, "throughput_seq_per_s": 10.0},
        candidate={"latency_ms": 120.0, "throughput_seq_per_s": 8.0},
        policy=policy,
        higher_is_better={"latency_ms": False, "throughput_seq_per_s": True},
    )
    assert {check.metric: check.status for check in checks} == {
        "latency_ms": "fail",
        "throughput_seq_per_s": "fail",
    }
