from pathlib import Path

import pytest

from quantguard.config.schema import QuantGuardConfig
from quantguard.exceptions import ConfigurationError
from quantguard.validation.determinism import measure_determinism
from quantguard.validation.injection import (
    Scenario,
    ScenarioOutcome,
    load_scenarios,
    run_scenarios,
    scenario_config,
    summarize,
    write_summary,
)
from tests.unit.fakes import FakeEvaluator, bits_accuracy, config_dict

ROOT = Path(__file__).resolve().parents[2]


def _base(tmp_path) -> QuantGuardConfig:
    return QuantGuardConfig.model_validate(
        config_dict(report_dir=str(tmp_path / "r"), database_path=str(tmp_path / "db.sqlite"))
    )


def test_real_scenario_file_loads() -> None:
    scenarios = load_scenarios(ROOT / "configs" / "injection-scenarios.yaml")
    assert {s.expect for s in scenarios} == {"detect", "clean"}
    assert any("2bit" in s.name for s in scenarios)


def test_scenario_config_overrides_candidate_only(tmp_path) -> None:
    base = _base(tmp_path)
    scenario = Scenario(name="x", expect="detect", candidate={"quantization_config": {"bits": 2}})
    config = scenario_config(base, scenario, str(tmp_path / "out"))
    assert config.candidate.quantization_config == {"bits": 2}
    assert config.candidate.version.endswith("-x")
    assert config.baseline == base.baseline


def test_detection_and_false_alarm_rates_with_fake_models(tmp_path) -> None:
    scenarios = [
        Scenario(name="eight", expect="clean", candidate={"quantization_config": {"bits": 8}}),
        Scenario(name="six", expect="clean", candidate={"quantization_config": {"bits": 6}}),
        Scenario(name="two", expect="detect", candidate={"quantization_config": {"bits": 2}}),
        Scenario(name="three", expect="detect", candidate={"quantization_config": {"bits": 3}}),
    ]
    summary = run_scenarios(FakeEvaluator(bits_accuracy), _base(tmp_path), scenarios, tmp_path / "v", tmp_path)
    assert summary.detection_rate == 1.0
    assert summary.false_alarm_rate == 0.0
    paths = write_summary(summary, tmp_path / "v")
    assert "Regression detection rate" in Path(paths["markdown"]).read_text(encoding="utf-8")


def test_summary_arithmetic_ignores_errors() -> None:
    def outcome(name, expect, status):
        return ScenarioOutcome(name=name, expect=expect, status=status, seconds=1.0)

    summary = summarize(
        [
            outcome("a", "detect", "FAIL"),
            outcome("b", "detect", "WARNING"),
            outcome("c", "clean", "FAIL"),
            outcome("d", "clean", "PASS"),
            outcome("e", "clean", "ERROR"),
        ]
    )
    assert summary.detection_rate == 0.5
    assert summary.warn_on_regression_rate == 0.5
    assert summary.false_alarm_rate == 0.5
    assert summary.error_count == 1


def test_duplicate_scenario_names_rejected(tmp_path) -> None:
    path = tmp_path / "s.yaml"
    path.write_text("scenarios:\n  - {name: a, expect: clean}\n  - {name: a, expect: detect}\n", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_scenarios(path)


def test_determinism_report(tmp_path) -> None:
    config = _base(tmp_path)
    report = measure_determinism(
        FakeEvaluator(bits_accuracy), config, config.candidate, runs=3, work_dir=tmp_path / "d", repo_root=tmp_path
    )
    assert report["all_identical"] is True
    assert report["metrics"][0]["max_abs_difference"] == 0.0
