from quantguard.config.schema import MetricPolicy, QualityPolicy
from quantguard.gate.release import decide
from quantguard.policy.engine import PolicyCheck, evaluate_quality_policy
from quantguard.regression.engine import compare_evaluations
from tests.unit.fixtures import evaluation, metric


def _check(status: str, required: bool = True) -> PolicyCheck:
    return PolicyCheck(
        scope="quality",
        task="mmlu",
        metric="acc",
        required=required,
        threshold_display="maximum 0.50 pp",
        observed_display="0.10 pp",
        status=status,  # type: ignore[arg-type]
        reason="fixture",
    )


def test_pass_when_all_required_checks_pass() -> None:
    decision = decide([_check("pass")])
    assert decision.status == "PASS"
    assert decision.exit_code == 0


def test_fail_when_required_policy_fails() -> None:
    quality = QualityPolicy(default=MetricPolicy(scale="proportion", max_regression_pp=0.5))
    comparisons = compare_evaluations(
        evaluation(metric("gsm8k", "acc", 0.812)),
        evaluation(metric("gsm8k", "acc", 0.801)),
        proportion_metrics={("gsm8k", "acc")},
    )
    decision = decide(evaluate_quality_policy(comparisons, quality))
    assert decision.status == "FAIL"
    assert decision.exit_code == 1


def test_warning_does_not_fail_the_process() -> None:
    decision = decide([_check("fail", required=False)])
    assert decision.status == "WARNING"
    assert decision.exit_code == 0


def test_infrastructure_error_beats_a_passing_check() -> None:
    decision = decide([_check("pass")], infrastructure_error="benchmark crashed")
    assert decision.status == "ERROR"
    assert decision.exit_code == 2
    assert decision.reasons == ["benchmark crashed"]
