from pathlib import Path

from quantguard.config.schema import QuantGuardConfig
from quantguard.pipeline import ReleasePipeline
from tests.unit.fakes import FakeEvaluator, bits_accuracy, config_dict


def _config(tmp_path: Path, bits: int = 8, **extra) -> QuantGuardConfig:
    raw = config_dict(report_dir=str(tmp_path / "reports"), database_path=str(tmp_path / "db.sqlite"), **extra)
    raw["candidate"]["quantization_config"] = {"bits": bits}
    return QuantGuardConfig.model_validate(raw)


def _run(tmp_path, config, evaluator=None):
    evaluator = evaluator or FakeEvaluator(bits_accuracy)
    payload, decision = ReleasePipeline(evaluator, tmp_path).run(config, profile=False)
    assert payload["runtime_seconds"] >= 0
    return payload, decision, evaluator


def test_equivalent_candidate_passes(tmp_path) -> None:
    _, decision, _ = _run(tmp_path, _config(tmp_path, bits=8))
    assert decision.status == "PASS"


def test_two_bit_candidate_fails(tmp_path) -> None:
    payload, decision, _ = _run(tmp_path, _config(tmp_path, bits=2))
    assert decision.status == "FAIL"
    assert payload["comparisons"][0]["stderr_method"] == "paired"


def test_reports_written_including_pr_comment(tmp_path) -> None:
    payload, _, _ = _run(tmp_path, _config(tmp_path, bits=2))
    directory = tmp_path / "reports" / payload["experiment_id"]
    comment = (directory / "pr_comment.md").read_text(encoding="utf-8")
    assert "<!-- quantguard-report -->" in comment and "FAIL" in comment
    assert "<table>" in (directory / "report.html").read_text(encoding="utf-8")
    assert "samples" not in payload["candidate_evaluation"]["metrics"][0] or payload["candidate_evaluation"]["metrics"][0].get("samples") is None


def test_stored_baseline_is_used_so_fp16_is_not_reevaluated(tmp_path) -> None:
    store = {"baseline_store": {"directory": str(tmp_path / "baselines"), "baseline_label": "fp16", "save_if_missing": True}}
    first = FakeEvaluator(bits_accuracy)
    payload, _, _ = _run(tmp_path, _config(tmp_path, **store), first)
    assert first.calls == ["1", "1-q"]
    assert payload["baseline_source"]["source"] == "live" and "saved_to" in payload["baseline_source"]

    second = FakeEvaluator(bits_accuracy)
    payload, decision, _ = _run(tmp_path, _config(tmp_path, **store), second)
    assert second.calls == ["1-q"]  # baseline came from the JSON store
    assert payload["baseline_source"]["source"] == "stored"
    assert decision.status == "PASS"


def test_stale_baseline_is_an_error_not_a_verdict(tmp_path) -> None:
    store = {"baseline_store": {"directory": str(tmp_path / "baselines"), "baseline_label": "fp16", "save_if_missing": True}}
    _run(tmp_path, _config(tmp_path, **store))
    changed = _config(tmp_path, **store)
    changed.evaluation.limit = 50
    _, decision, _ = _run(tmp_path, changed)
    assert decision.status == "ERROR"
    assert "limit" in decision.reasons[0]


def test_stale_baseline_can_fall_back_to_live(tmp_path) -> None:
    store = {"baseline_store": {"directory": str(tmp_path / "baselines"), "baseline_label": "fp16", "save_if_missing": True}}
    _run(tmp_path, _config(tmp_path, **store))
    store["baseline_store"]["on_stale"] = "evaluate"
    changed = _config(tmp_path, **store)
    changed.evaluation.limit = 50
    evaluator = FakeEvaluator(bits_accuracy)
    _, decision, _ = _run(tmp_path, changed, evaluator)
    assert decision.status in {"PASS", "WARNING"}
    assert evaluator.calls == ["1", "1-q"]


def _record_release(tmp_path, config, label: str, accuracy: float) -> None:
    from quantguard.baselines.store import BaselineStore, fingerprint_inputs
    from quantguard.metadata.environment import collect_environment
    from quantguard.results.normalizer import normalize_lm_eval

    spec = config.candidate.model_copy(update={"version": "rel"})
    raw = FakeEvaluator(lambda _spec: accuracy).evaluate(spec, config.evaluation, None, tmp_path / "x.json")
    evaluation = normalize_lm_eval(
        raw, model_id=spec.model_id, model_version="rel", evaluation_config_hash="h", backend_version="fake-0"
    )
    environment = collect_environment(tmp_path)
    BaselineStore(tmp_path / "baselines").save(
        label=label,
        spec=spec,
        evaluation=evaluation,
        inputs=fingerprint_inputs(spec, config.evaluation, environment.packages),
        kind="release",
    )


def _release_config(tmp_path, required: bool) -> QuantGuardConfig:
    return _config(
        tmp_path,
        baseline_store={
            "directory": str(tmp_path / "baselines"),
            "baseline_label": "fp16",
            "save_if_missing": True,
            "previous_release_label": "rel",
            "previous_release_required": required,
        },
    )


def test_previous_release_regression_is_advisory_by_default(tmp_path) -> None:
    _record_release(tmp_path, _config(tmp_path), "rel", accuracy=0.76)  # last release scored higher
    payload, decision, _ = _run(tmp_path, _release_config(tmp_path, required=False))
    refs = {check.reference for check in decision.checks}
    assert refs == {"baseline", "previous_release"}
    assert decision.status == "WARNING"  # fine vs FP16, worse than last release
    assert payload["release_comparisons"]


def test_previous_release_regression_can_be_required(tmp_path) -> None:
    _record_release(tmp_path, _config(tmp_path), "rel", accuracy=0.76)
    _, decision, _ = _run(tmp_path, _release_config(tmp_path, required=True))
    assert decision.status == "FAIL"


def test_missing_previous_release_is_an_error(tmp_path) -> None:
    _, decision, _ = _run(tmp_path, _release_config(tmp_path, required=False))
    assert decision.status == "ERROR"
