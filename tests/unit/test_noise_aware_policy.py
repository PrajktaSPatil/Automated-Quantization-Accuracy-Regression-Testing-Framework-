import pytest

from quantguard.config.schema import MetricPolicy, QualityPolicy, TaskPolicy
from quantguard.gate.release import decide
from quantguard.policy.engine import evaluate_quality_policy
from quantguard.regression.engine import compare_evaluations, paired_stderr
from quantguard.results.schema import NormalizedEvaluation
from tests.unit.fixtures import evaluation, metric


def _eval(value: float, stderr: float | None, *, name: str = "acc", hib: bool = True, per_sample=None):
    m = metric("t", name, value, higher_is_better=hib).model_copy(update={"stderr": stderr})
    ev = evaluation(m)
    if per_sample is not None:
        ev = ev.model_copy(update={"per_sample": {"t": {name: per_sample}}})
    return ev


def _policy(**kwargs) -> QualityPolicy:
    return QualityPolicy(
        default=MetricPolicy(scale="proportion", max_regression_pp=1.0),
        **kwargs,
    )


def _gate(base, cand, policy, proportion=True):
    comparisons = compare_evaluations(base, cand, proportion_metrics={("t", "acc")} if proportion else set())
    return decide(evaluate_quality_policy(comparisons, policy))


def test_regression_beyond_noise_fails() -> None:
    # 5 pp drop, se_diff ~ 1.4 pp -> lower bound ~ 2.7 pp > 1 pp
    decision = _gate(_eval(0.80, 0.01), _eval(0.75, 0.01), _policy())
    assert decision.status == "FAIL"
    assert "beyond noise" in decision.reasons[0]


def test_regression_within_noise_warns_instead_of_failing() -> None:
    # 2 pp drop, se_diff ~ 4.2 pp -> cannot be separated from noise
    decision = _gate(_eval(0.80, 0.03), _eval(0.78, 0.03), _policy())
    assert decision.status == "WARNING"
    assert "sampling noise" in decision.reasons[0]
    assert decision.exit_code == 0


def test_on_uncertain_fail_is_strict() -> None:
    decision = _gate(_eval(0.80, 0.03), _eval(0.78, 0.03), _policy(on_uncertain="fail"))
    assert decision.status == "FAIL"


def test_missing_stderr_falls_back_to_fixed_threshold() -> None:
    decision = _gate(_eval(0.80, None), _eval(0.78, None), _policy())
    assert decision.status == "FAIL"


def test_noise_aware_can_be_disabled_per_metric() -> None:
    policy = _policy(
        tasks={"t": TaskPolicy(scale="proportion", max_regression_pp=1.0, noise_aware=False)}
    )
    assert _gate(_eval(0.80, 0.03), _eval(0.78, 0.03), policy).status == "FAIL"


def test_within_threshold_passes_regardless_of_noise() -> None:
    assert _gate(_eval(0.80, 0.05), _eval(0.795, 0.05), _policy()).status == "PASS"


def test_relative_scale_for_perplexity() -> None:
    policy = QualityPolicy(
        default=MetricPolicy(scale="relative", max_regression_percent=3.0),
    )
    base = _eval(10.0, None, name="word_perplexity", hib=False)
    ok = _eval(10.2, None, name="word_perplexity", hib=False)
    bad = _eval(10.5, None, name="word_perplexity", hib=False)
    assert _gate(base, ok, policy, proportion=False).status == "PASS"
    assert _gate(base, bad, policy, proportion=False).status == "FAIL"


def test_only_listed_metrics_skips_the_rest() -> None:
    policy = QualityPolicy(
        default=MetricPolicy(scale="relative", max_regression_percent=1.0),
        tasks={
            "t": TaskPolicy(
                only_listed_metrics=True,
                metrics={"word_perplexity": MetricPolicy(scale="relative", max_regression_percent=3.0)},
            )
        },
    )
    base = evaluation(
        metric("t", "word_perplexity", 10.0, higher_is_better=False),
        metric("t", "byte_perplexity", 2.0, higher_is_better=False),
    )
    cand = evaluation(
        metric("t", "word_perplexity", 10.1, higher_is_better=False),
        metric("t", "byte_perplexity", 9.0, higher_is_better=False),
    )
    checks = evaluate_quality_policy(compare_evaluations(base, cand), policy)
    assert [c.metric for c in checks] == ["word_perplexity"]


def test_relative_policy_requires_threshold() -> None:
    with pytest.raises(ValueError):
        MetricPolicy(scale="relative")


def _ps(values: list[float]) -> dict[str, float]:
    return {str(i): v for i, v in enumerate(values)}


def test_paired_stderr_is_smaller_when_errors_are_correlated() -> None:
    base_scores = [1.0] * 70 + [0.0] * 30
    cand_scores = [1.0] * 69 + [0.0] * 31  # one flipped document
    base = _eval(0.70, 0.0458, per_sample=_ps(base_scores))
    cand = _eval(0.69, 0.0462, per_sample=_ps(cand_scores))
    paired = paired_stderr(base, cand, "t", "acc", 0.70, 0.69)
    assert paired is not None and paired < 0.02  # independent would be ~0.065
    comparison = compare_evaluations(base, cand, proportion_metrics={("t", "acc")})[0]
    assert comparison.paired is True
    assert comparison.stderr_diff == pytest.approx(paired)


def test_paired_stderr_rejects_different_documents() -> None:
    base = _eval(0.5, 0.05, per_sample=_ps([1.0, 0.0] * 5))
    cand = _eval(0.5, 0.05, per_sample={str(i + 100): v for i, v in enumerate([1.0, 0.0] * 5)})
    assert paired_stderr(base, cand, "t", "acc", 0.5, 0.5) is None
    assert compare_evaluations(base, cand)[0].paired is False


def test_paired_stderr_rejects_scores_that_do_not_match_the_metric() -> None:
    base = _eval(0.9, 0.05, per_sample=_ps([1.0, 0.0] * 5))  # mean is 0.5, not 0.9
    cand = _eval(0.5, 0.05, per_sample=_ps([1.0, 0.0] * 5))
    assert paired_stderr(base, cand, "t", "acc", 0.9, 0.5) is None
