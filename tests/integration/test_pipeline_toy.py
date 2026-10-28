"""End-to-end gate behaviour on the numpy toy backend (no GPU, no network)."""

import json
from pathlib import Path

import pytest

from quantguard.config.schema import QuantGuardConfig
from quantguard.pipeline import ReleasePipeline, create_baseline
from quantguard.validation.injection import Scenario, default_scenarios, run_suite
from quantguard.validation.toy import ToyEvaluator


class CountingEvaluator(ToyEvaluator):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    def evaluate(self, spec, evaluation, environment, raw_output_path):
        self.calls.append(spec.version)
        return super().evaluate(spec, evaluation, environment, raw_output_path)


def _config(tmp_path: Path, *, candidate_version="1.0-q", quant=None, limit=3000) -> QuantGuardConfig:
    quant = quant if quant is not None else {"bits": 8, "group_size": 128}
    return QuantGuardConfig.model_validate(
        {
            "name": "toy",
            "baseline": {"model_id": "toy/mlp", "version": "fp", "precision": "fp32", "artifact_uri": "toy://mlp", "status": "approved"},
            "candidate": {
                "model_id": "toy/mlp", "version": candidate_version, "precision": "int8", "artifact_uri": "toy://mlp",
                "quantization_method": "fake-rtn", "quantization_config": quant, "source_model_id": "toy/mlp",
            },
            "evaluation": {"backend": "lm_eval", "tasks": ["toy_cls"], "limit": limit, "device": "cpu", "dtype": "float32"},
            "quality_policy": {
                "default": {"scale": "proportion", "max_regression_pp": 1.0},
                "tasks": {
                    "toy_cls": {
                        "scale": "proportion", "max_regression_pp": 1.0, "only_listed_metrics": True,
                        "metrics": {"acc": {"scale": "proportion", "max_regression_pp": 1.0}},
                    }
                },
            },
            "baseline_store": {"directory": str(tmp_path / "baselines")},
            "database_path": str(tmp_path / "q.sqlite"),
            "report_dir": str(tmp_path / "reports"),
        }
    )


def test_equivalent_candidate_passes_and_reports_noise(tmp_path: Path) -> None:
    payload, decision = ReleasePipeline(ToyEvaluator(), tmp_path).run(_config(tmp_path), profile=False)
    assert decision.status == "PASS", decision.reasons
    assert payload["comparisons"][0]["paired"] is True
    report = (Path(_config(tmp_path).report_dir) / payload["experiment_id"] / "report.md").read_text()
    assert "paired stderr" in report or "Noise" in report


def test_two_bit_candidate_fails_the_gate(tmp_path: Path) -> None:
    _, decision = ReleasePipeline(ToyEvaluator(), tmp_path).run(
        _config(tmp_path, quant={"bits": 2, "group_size": 128}), profile=False
    )
    assert decision.status == "FAIL"
    assert decision.exit_code == 1


def test_stored_baseline_is_used_instead_of_re_evaluating(tmp_path: Path) -> None:
    config = _config(tmp_path)
    create_baseline(config, ToyEvaluator(), tmp_path)
    counting = CountingEvaluator()
    payload, decision = ReleasePipeline(counting, tmp_path).run(config, profile=False)
    assert counting.calls == ["1.0-q"]  # baseline was not evaluated again
    assert payload["baseline_source"].startswith("stored:")
    assert decision.status == "PASS"


def test_stored_baseline_with_different_settings_is_an_error(tmp_path: Path) -> None:
    create_baseline(_config(tmp_path, limit=3000), ToyEvaluator(), tmp_path)
    _, decision = ReleasePipeline(ToyEvaluator(), tmp_path).run(_config(tmp_path, limit=1000), profile=False)
    assert decision.status == "ERROR"
    assert "not comparable" in decision.reasons[0]


def test_release_is_recorded_and_used_as_previous_release(tmp_path: Path) -> None:
    config = _config(tmp_path)
    create_baseline(config, ToyEvaluator(), tmp_path)
    _, first = ReleasePipeline(ToyEvaluator(), tmp_path).run(config, profile=False, record_release=True)
    assert first.status == "PASS"
    assert list((tmp_path / "baselines" / "toy" / "releases").glob("*.json"))

    nxt = _config(tmp_path, candidate_version="1.1-q", quant={"bits": 8, "group_size": 64})
    payload, decision = ReleasePipeline(ToyEvaluator(), tmp_path).run(nxt, profile=False)
    assert payload["previous_release"]["version"] == "1.0-q"
    assert any(c["reference"] == "previous_release" for c in payload["decision"]["checks"])
    assert decision.status == "PASS"


def test_failed_gate_does_not_record_a_release(tmp_path: Path) -> None:
    config = _config(tmp_path, quant={"bits": 2, "group_size": 128})
    create_baseline(config, ToyEvaluator(), tmp_path)
    ReleasePipeline(ToyEvaluator(), tmp_path).run(config, profile=False, record_release=True)
    assert not (tmp_path / "baselines" / "toy" / "releases").exists()


def test_regression_against_previous_release_is_caught(tmp_path: Path) -> None:
    good = _config(tmp_path)
    create_baseline(good, ToyEvaluator(), tmp_path)
    ReleasePipeline(ToyEvaluator(), tmp_path).run(good, profile=False, record_release=True)
    # Loosen the baseline policy so only the release-to-release policy can catch 2-bit damage.
    loose = _config(tmp_path, candidate_version="2.0-q", quant={"bits": 2, "group_size": 128})
    loose.quality_policy.tasks["toy_cls"].metrics["acc"].max_regression_pp = 90.0
    loose.baseline_store.previous_release_policy = good.quality_policy
    _, decision = ReleasePipeline(ToyEvaluator(), tmp_path).run(loose, profile=False)
    assert decision.status == "FAIL"
    assert any(c.reference == "previous_release" and c.status == "fail" for c in decision.checks)


def test_injection_suite_detects_bad_and_spares_equivalent(tmp_path: Path) -> None:
    config = _config(tmp_path)
    scenarios = [
        Scenario("w2", "bad", {"bits": 2, "group_size": 128}),
        *default_scenarios(equivalent_seeds=3)[4:],
    ]
    result = run_suite(config, ToyEvaluator(), tmp_path, scenarios)
    assert result.detection == (1, 1)
    assert result.false_alarms[0] == 0
    assert result.errors == 0
    assert "Regression detection rate" in result.to_markdown()


def test_pr_comment_and_trend_cli(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from quantguard.cli.main import app

    config = _config(tmp_path)
    create_baseline(config, ToyEvaluator(), tmp_path)
    payload, _ = ReleasePipeline(ToyEvaluator(), tmp_path).run(config, profile=False, record_release=True)
    report = Path(config.report_dir) / payload["experiment_id"] / "report.json"
    runner = CliRunner()
    out = tmp_path / "c.md"
    result = runner.invoke(app, ["pr-comment", "--report", str(report), "--output", str(out), "--run-url", "http://x"])
    assert result.exit_code == 0, result.output
    text = out.read_text()
    assert text.startswith("<!-- quantguard-report -->") and "QuantGuard: PASS" in text and "http://x" in text

    cfg_path = tmp_path / "cfg.json"  # YAML is a superset of JSON
    cfg_path.write_text(json.dumps(config.model_dump(mode="json")))
    chart = tmp_path / "trend.html"
    import os

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = runner.invoke(app, ["trend", "--config", str(cfg_path), "--output", str(chart)])
    finally:
        os.chdir(cwd)
    assert result.exit_code == 0, result.output
    assert "<svg" in chart.read_text()
