import pytest

from quantguard.config.schema import MetricPolicy, QualityPolicy
from quantguard.gate.release import decide
from quantguard.policy.engine import evaluate_quality_policy
from quantguard.regression.engine import compare_evaluations
from tests.unit.fixtures import evaluation, metric


def _run(base, cand, policy, *, base_se=None, cand_se=None):
    left = metric("t", "acc", base)
    right = metric("t", "acc", cand)
    left.stderr, right.stderr = base_se, cand_se
    comparisons = compare_evaluations(evaluation(left), evaluation(right), proportion_metrics={("t", "acc")})
    return evaluate_quality_policy(comparisons, QualityPolicy(default=policy))[0]


POLICY = MetricPolicy(scale="proportion", max_regression_pp=1.0)


def test_without_stderr_behaves_like_a_fixed_threshold() -> None:
    assert _run(0.80, 0.78, POLICY).status == "fail"
    assert _run(0.80, 0.795, POLICY).status == "pass"


def test_drop_over_limit_but_inside_noise_is_inconclusive() -> None:
    check = _run(0.80, 0.78, POLICY, base_se=0.02, cand_se=0.02)  # 2pp drop, ~5.5pp noise band
    assert check.status == "inconclusive"
    assert check.noise_display is not None
    assert decide([check]).status == "WARNING"


def test_drop_over_limit_and_outside_noise_fails() -> None:
    check = _run(0.80, 0.70, POLICY, base_se=0.01, cand_se=0.01)  # 10pp drop, ~2.8pp band
    assert check.status == "fail"
    assert decide([check]).status == "FAIL"


def test_on_inconclusive_fail_makes_it_a_failure() -> None:
    strict = POLICY.model_copy(update={"on_inconclusive": "fail"})
    assert _run(0.80, 0.78, strict, base_se=0.02, cand_se=0.02).status == "fail"


def test_use_stderr_false_restores_fixed_threshold() -> None:
    fixed = POLICY.model_copy(update={"use_stderr": False})
    assert _run(0.80, 0.78, fixed, base_se=0.02, cand_se=0.02).status == "fail"


def test_drop_under_limit_passes_even_with_tiny_noise() -> None:
    assert _run(0.80, 0.795, POLICY, base_se=0.0001, cand_se=0.0001).status == "pass"


def test_relative_threshold_on_raw_metric() -> None:
    policy = MetricPolicy(scale="raw", max_regression_relative_percent=3.0)
    left = metric("wikitext", "word_perplexity", 10.0, higher_is_better=False)
    right = metric("wikitext", "word_perplexity", 10.5, higher_is_better=False)
    check = evaluate_quality_policy(
        compare_evaluations(evaluation(left), evaluation(right)), QualityPolicy(default=policy)
    )[0]
    assert check.status == "fail"
    assert "relative" in check.threshold_display
    ok = evaluate_quality_policy(
        compare_evaluations(
            evaluation(left), evaluation(metric("wikitext", "word_perplexity", 10.2, higher_is_better=False))
        ),
        QualityPolicy(default=policy),
    )[0]
    assert ok.status == "pass"


def test_policy_needs_some_threshold() -> None:
    with pytest.raises(ValueError):
        MetricPolicy(scale="proportion")


def test_absolute_and_relative_limits_either_can_trigger() -> None:
    policy = MetricPolicy(scale="proportion", max_regression_pp=5.0, max_regression_relative_percent=1.0)
    assert _run(0.80, 0.79, policy).status == "fail"  # 1.25% relative, 1pp absolute
