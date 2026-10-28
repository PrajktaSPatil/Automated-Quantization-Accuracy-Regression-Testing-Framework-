import json
from pathlib import Path

from quantguard.reporting.engine import render_markdown, write_reports
from quantguard.storage.sqlite import ExperimentStore


def _payload() -> dict:
    return {
        "experiment_id": "exp_test",
        "name": "unit",
        "created_at": "2026-01-01T00:00:00+00:00",
        "config_hash": "abc",
        "baseline": {
            "model_id": "org/model",
            "version": "1",
            "precision": "fp16",
            "revision": "rev",
            "quantization_method": None,
            "artifact_uri": "hf://org/model",
            "status": "approved",
        },
        "candidate": {
            "model_id": "org/model",
            "version": "2",
            "precision": "int8",
            "revision": "rev",
            "quantization_method": "torchao-int8-weight-only",
            "source_model_id": "org/model",
            "source_version": "1",
            "artifact_uri": "hf://org/model",
        },
        "comparisons": [
            {
                "task": "arc_easy",
                "metric": "acc",
                "baseline_value": 0.5,
                "candidate_value": 0.49,
            }
        ],
        "baseline_performance": {"model_size_bytes": 100, "methodology": {"warmup_iterations": 1}},
        "candidate_performance": {"model_size_bytes": 50},
        "decision": {
            "status": "FAIL",
            "exit_code": 1,
            "reasons": ["arc_easy acc exceeded policy"],
            "checks": [
                {
                    "scope": "quality",
                    "task": "arc_easy",
                    "metric": "acc",
                    "required": True,
                    "threshold_display": "maximum 0.50 pp",
                    "observed_display": "1.00 pp",
                    "status": "fail",
                    "reason": "arc_easy acc exceeded policy",
                }
            ],
        },
        "environment": {
            "python_version": "3.11.15",
            "platform": "Windows",
            "git_sha": "deadbeef",
            "torch_version": "2.12.1+cu130",
            "cuda_version": "13.0",
            "gpu_name": "NVIDIA GeForce RTX 2050",
            "gpu_total_mib": 4095,
            "compute_capability": "8.6",
            "packages": {"torch": "2.12.1"},
        },
    }


def test_reports_include_gate_reason(tmp_path: Path) -> None:
    paths = write_reports(_payload(), tmp_path)
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    html = Path(paths["html"]).read_text(encoding="utf-8")
    body = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert "FAIL" in markdown
    assert "arc_easy acc exceeded policy" in markdown
    assert "Gate: FAIL" in html
    assert body["decision"]["status"] == "FAIL"
    assert "QUANTGUARD RELEASE REPORT" in render_markdown(body)


def test_sqlite_round_trip(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path / "history.sqlite")
    payload = _payload()
    store.save(payload["experiment_id"], payload)
    loaded = store.get("exp_test")
    assert loaded is not None
    assert loaded["decision"]["status"] == "FAIL"
    assert store.list_recent()[0]["gate_status"] == "FAIL"
