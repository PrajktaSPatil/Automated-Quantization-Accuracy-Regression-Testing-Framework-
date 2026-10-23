import json
from pathlib import Path

from typer.testing import CliRunner

from quantguard.cli.main import app

ROOT = Path(__file__).resolve().parents[2]
runner = CliRunner()


def test_validate_real_config() -> None:
    result = runner.invoke(app, ["validate", "--config", str(ROOT / "configs" / "qwen2.5-0.5b.yaml")])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["name"] == "qwen2.5-0.5b-int8"
    assert len(payload["config_hash"]) == 64


def test_validate_missing_file_exits_error() -> None:
    result = runner.invoke(app, ["validate", "--config", str(ROOT / "configs" / "missing.yaml")])
    assert result.exit_code != 0


def test_gate_command_fails_on_regression(tmp_path: Path) -> None:
    from quantguard.results.schema import NormalizedEvaluation

    def dump(version: str, value: float) -> Path:
        evaluation = NormalizedEvaluation(
            backend="lm_eval",
            backend_version="test",
            model_id="Qwen/Qwen2.5-0.5B",
            model_version=version,
            evaluation_config_hash="abc",
            metrics=[
                {
                    "name": "acc",
                    "value": value,
                    "higher_is_better": True,
                    "task": "arc_easy",
                    "model_id": "Qwen/Qwen2.5-0.5B",
                    "model_version": version,
                    "evaluation_config_hash": "abc",
                    "timestamp": "2026-01-01T00:00:00+00:00",
                }
            ],
        )
        path = tmp_path / f"{version}.json"
        path.write_text(evaluation.model_dump_json(), encoding="utf-8")
        return path

    result = runner.invoke(
        app,
        [
            "gate",
            "--baseline",
            str(dump("base", 0.80)),
            "--candidate",
            str(dump("cand", 0.70)),
            "--config",
            str(ROOT / "configs" / "qwen2.5-0.5b.yaml"),
        ],
    )
    assert result.exit_code == 1, result.output
    assert "FAIL" in result.output
