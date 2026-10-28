from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from quantguard.dashboard import _DashboardState
from quantguard.storage.sqlite import ExperimentStore
from tests.unit.fakes import config_dict


def _config(path: Path) -> None:
    path.write_text(json.dumps(config_dict()), encoding="utf-8")


def test_dashboard_reads_actual_history_without_model_loading(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    _config(tmp_path / "configs" / "smoke.yaml")
    database = ExperimentStore(tmp_path / "quantguard.sqlite")
    database.save("exp_one", {
        "experiment_id": "exp_one", "created_at": "2026-01-01T00:00:00Z", "name": "smoke",
        "config_hash": "abc", "decision": {"status": "PASS"}, "comparisons": [],
    })

    data = _DashboardState(tmp_path).data()

    assert data["latest"]["experiment_id"] == "exp_one"
    assert data["runs"][0]["gate_status"] == "PASS"
    assert data["configs"] == ["configs/smoke.yaml"]


def test_dashboard_run_accepts_only_project_configs(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    _config(tmp_path / "configs" / "smoke.yaml")
    state = _DashboardState(tmp_path)

    with pytest.raises(ValueError, match="inside the project's configs"):
        state.start("../outside.yaml")


def test_dashboard_reads_validation_summary_records(tmp_path: Path) -> None:
    output = tmp_path / "reports" / "validation"
    output.mkdir(parents=True)
    (output / "validation_results.json").write_text(json.dumps({"detection_rate": 1.0, "outcomes": []}), encoding="utf-8")

    data = _DashboardState(tmp_path).data()

    assert data["injections"][0]["detection_rate"] == 1.0


def test_dashboard_reads_measured_determinism_when_present(tmp_path: Path) -> None:
    output = tmp_path / "reports"
    output.mkdir()
    (output / "determinism.json").write_text(json.dumps({
        "runs": 3, "all_identical": False,
        "metrics": [{"task": "arc", "metric": "acc", "stdev": 0.01, "identical": False}],
    }), encoding="utf-8")

    data = _DashboardState(tmp_path).data()

    assert data["determinism"][0]["metrics"][0]["stdev"] == 0.01


def test_dashboard_skips_malformed_latest_history_payload(tmp_path: Path) -> None:
    database = ExperimentStore(tmp_path / "quantguard.sqlite")
    database.save("good", {
        "experiment_id": "good", "created_at": "2026-01-01T00:00:00Z", "name": "good",
        "config_hash": "abc", "decision": {"status": "PASS"},
    })
    database.save("bad", {
        "experiment_id": "bad", "created_at": "2026-01-02T00:00:00Z", "name": "bad",
        "config_hash": "abc", "decision": {"status": "ERROR"},
    })
    with sqlite3.connect(tmp_path / "quantguard.sqlite") as connection:
        connection.execute("UPDATE experiments SET payload_json = '{broken' WHERE experiment_id = 'bad'")

    data = _DashboardState(tmp_path).data()

    assert data["latest"]["experiment_id"] == "good"
    assert data["errors"] and "bad" in data["errors"][0]
