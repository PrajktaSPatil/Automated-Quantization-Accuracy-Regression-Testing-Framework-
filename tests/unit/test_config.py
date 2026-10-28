from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from quantguard.config.loader import canonical_json, config_hash, load_config
from quantguard.config.schema import QuantGuardConfig
from quantguard.exceptions import ConfigurationError

ROOT = Path(__file__).resolve().parents[2]


def _document(**overrides: object) -> dict:
    document = {
        "name": "unit",
        "baseline": {
            "model_id": "org/model",
            "version": "1",
            "precision": "fp16",
            "artifact_uri": "hf://org/model",
            "status": "approved",
        },
        "candidate": {
            "model_id": "org/model",
            "version": "2",
            "precision": "int8",
            "quantization_method": "torchao-int8-weight-only",
            "source_model_id": "org/model",
            "source_version": "1",
            "artifact_uri": "hf://org/model",
            "status": "candidate",
        },
        "evaluation": {"backend": "lm_eval", "tasks": ["arc_easy"]},
        "quality_policy": {
            "default": {"scale": "proportion", "max_regression_pp": 0.5},
        },
    }
    document.update(overrides)
    return document


def test_real_config_loads_and_hashes_stably() -> None:
    loaded = load_config(ROOT / "configs" / "qwen2.5-0.5b.yaml")
    assert loaded.baseline.status == "approved"
    assert loaded.candidate.quantization_method == "torchao-int8-weight-only"
    assert config_hash(loaded) == config_hash(loaded)
    assert len(config_hash(loaded)) == 64


def test_hash_changes_when_threshold_changes() -> None:
    first = QuantGuardConfig.model_validate(_document())
    second_raw = _document()
    second_raw["quality_policy"]["default"]["max_regression_pp"] = 1.0
    second = QuantGuardConfig.model_validate(second_raw)
    assert config_hash(first) != config_hash(second)
    assert canonical_json(first) != canonical_json(second)


def test_baseline_must_be_approved() -> None:
    raw = _document()
    raw["baseline"]["status"] = "candidate"
    with pytest.raises(ValidationError):
        QuantGuardConfig.model_validate(raw)


def test_quantized_model_requires_source() -> None:
    raw = _document()
    raw["candidate"]["source_model_id"] = None
    with pytest.raises(ValidationError):
        QuantGuardConfig.model_validate(raw)


def test_malformed_yaml(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("tasks: [\n", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_config(path)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError):
        load_config(tmp_path / "missing.yaml")


def test_unsupported_backend_rejected(tmp_path: Path) -> None:
    raw = _document()
    raw["evaluation"]["backend"] = "shell"
    path = tmp_path / "bad.yml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_config(path)
