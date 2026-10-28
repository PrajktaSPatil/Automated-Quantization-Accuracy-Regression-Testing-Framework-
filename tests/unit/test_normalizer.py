import pytest

from quantguard.exceptions import EvaluationError
from quantguard.results.normalizer import normalize_lm_eval


def test_normalizes_accuracy_and_ignores_stderr() -> None:
    raw = {
        "results": {
            "arc_easy": {
                "alias": "arc_easy",
                "acc,none": 0.847,
                "acc_stderr,none": 0.01,
            }
        },
        "higher_is_better": {"arc_easy": {"acc": True}},
        "versions": {"arc_easy": 1},
    }
    normalized = normalize_lm_eval(
        raw,
        model_id="org/model",
        model_version="1",
        evaluation_config_hash="abc",
        backend_version="0.4",
    )
    assert len(normalized.metrics) == 1
    metric = normalized.metrics[0]
    assert metric.name == "acc"
    assert metric.value == pytest.approx(0.847)
    assert metric.higher_is_better is True
    assert metric.task == "arc_easy"


def test_missing_direction_is_an_error() -> None:
    raw = {"results": {"arc_easy": {"acc,none": 0.5}}, "higher_is_better": {}}
    with pytest.raises(EvaluationError):
        normalize_lm_eval(
            raw,
            model_id="org/model",
            model_version="1",
            evaluation_config_hash="abc",
            backend_version="0.4",
        )


def test_empty_results_are_an_error() -> None:
    with pytest.raises(EvaluationError):
        normalize_lm_eval(
            {"results": {"arc_easy": {"alias": "arc_easy"}}},
            model_id="org/model",
            model_version="1",
            evaluation_config_hash="abc",
            backend_version="0.4",
        )
