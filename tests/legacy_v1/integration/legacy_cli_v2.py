import json
from pathlib import Path

from typer.testing import CliRunner

from quantguard.baselines.store import BaselineStore, fingerprint_inputs
from quantguard.cli.main import app
from quantguard.config.loader import load_config
from tests.unit.fixtures import evaluation, metric
from tests.unit.test_reporting_and_storage import _payload

ROOT = Path(__file__).resolve().parents[2]
runner = CliRunner()


def test_every_shipped_config_validates() -> None:
    for path in sorted((ROOT / "configs").glob("*.yaml")):
        if path.name == "injection-scenarios.yaml":
            continue
        assert load_config(path).baseline_store is not None or "qwen2.5-0.5b" in path.name, path.name
        result = runner.invoke(app, ["validate", "--config", str(path)])
        assert result.exit_code == 0, (path.name, result.output)


def test_simulate_noise_command(tmp_path) -> None:
    out = tmp_path / "sim.md"
    result = runner.invoke(
        app, ["simulate-noise", "--sample-sizes", "100", "--drops-pp", "0,4", "--trials", "300", "--output", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert "stderr-paired" in out.read_text(encoding="utf-8")
    raw = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert raw["study"].startswith("synthetic")
    assert raw["results"][0]["fail_rate_95pct_wilson_ci"][0] <= raw["results"][0]["fail_rate"]


def test_pr_comment_command(tmp_path) -> None:
    report = tmp_path / "report.json"
    report.write_text(json.dumps(_payload()), encoding="utf-8")
    out = tmp_path / "c.md"
    result = runner.invoke(app, ["pr-comment", "--report-json", str(report), "--output", str(out)])
    assert result.exit_code == 0, result.output
    assert "<!-- quantguard-report -->" in out.read_text(encoding="utf-8")


def test_baseline_list_and_trend_commands(tmp_path) -> None:
    config = load_config(ROOT / "configs" / "nightly-gpu.yaml")
    store = BaselineStore(tmp_path / "b")
    for label, value in (("release-1", 0.70), ("release-2", 0.69)):
        store.save(
            label=label,
            spec=config.candidate,
            evaluation=evaluation(metric("arc_easy", "acc", value)),
            inputs=fingerprint_inputs(config.candidate, config.evaluation, {}),
            kind="release",
        )
    listed = runner.invoke(app, ["baseline", "list", "--directory", str(tmp_path / "b")])
    assert listed.exit_code == 0 and "release-2" in listed.output
    svg = tmp_path / "t.svg"
    plotted = runner.invoke(
        app, ["trend", "--baselines", str(tmp_path / "b"), "--task", "arc_easy", "--metric", "acc", "--output", str(svg)]
    )
    assert plotted.exit_code == 0, plotted.output
    assert svg.read_text(encoding="utf-8").count("<circle") == 2


def test_promote_a_passing_run_then_use_it_as_previous_release(tmp_path, monkeypatch) -> None:
    import yaml

    from quantguard.pipeline import ReleasePipeline
    from quantguard.config.schema import QuantGuardConfig
    from tests.unit.fakes import FakeEvaluator, bits_accuracy, config_dict

    raw = config_dict(
        report_dir=str(tmp_path / "reports"),
        database_path=str(tmp_path / "db.sqlite"),
        baseline_store={"directory": str(tmp_path / "baselines"), "baseline_label": "fp16", "save_if_missing": True},
    )
    config = QuantGuardConfig.model_validate(raw)
    payload, decision = ReleasePipeline(FakeEvaluator(bits_accuracy), tmp_path).run(config, profile=False)
    assert decision.status == "PASS"

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    experiment_dir = tmp_path / "reports" / payload["experiment_id"]
    result = runner.invoke(
        app,
        ["baseline", "promote", "--experiment-dir", str(experiment_dir), "--config", str(cfg_path), "--label", "release-1"],
    )
    assert result.exit_code == 0, result.output
    record = BaselineStore(tmp_path / "baselines").load("release-1")
    assert record.kind == "release"
    assert record.evaluation.metrics[0].samples  # per-example scores kept for paired comparisons

    # The next run can compare against it.
    raw["baseline_store"]["previous_release_label"] = "release-1"
    again = QuantGuardConfig.model_validate(raw)
    _, decision2 = ReleasePipeline(FakeEvaluator(bits_accuracy), tmp_path).run(again, profile=False)
    assert decision2.status == "PASS"
    assert {check.reference for check in decision2.checks} == {"baseline", "previous_release"}


def test_promote_refuses_a_failing_run(tmp_path, monkeypatch) -> None:
    import yaml

    from quantguard.pipeline import ReleasePipeline
    from quantguard.config.schema import QuantGuardConfig
    from tests.unit.fakes import FakeEvaluator, bits_accuracy, config_dict

    raw = config_dict(
        report_dir=str(tmp_path / "reports"),
        database_path=str(tmp_path / "db.sqlite"),
        baseline_store={"directory": str(tmp_path / "baselines"), "baseline_label": "fp16"},
    )
    raw["candidate"]["quantization_config"] = {"bits": 2}
    payload, decision = ReleasePipeline(FakeEvaluator(bits_accuracy), tmp_path).run(
        QuantGuardConfig.model_validate(raw), profile=False
    )
    assert decision.status == "FAIL"
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app,
        [
            "baseline", "promote", "--experiment-dir", str(tmp_path / "reports" / payload["experiment_id"]),
            "--config", str(cfg_path), "--label", "release-bad",
        ],
    )
    assert result.exit_code == 2
    assert not BaselineStore(tmp_path / "baselines").exists("release-bad")
