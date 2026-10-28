from pathlib import Path

from quantguard.baselines.store import BaselineStore, comparability_problems
from quantguard.config.schema import EvaluationConfig, ModelSpec
from tests.unit.fixtures import evaluation, metric


def _spec(version: str) -> ModelSpec:
    return ModelSpec(model_id="m", version=version, precision="fp16", artifact_uri="hf://m", revision="r1")


def _eval_cfg(**kw) -> EvaluationConfig:
    return EvaluationConfig(backend="lm_eval", tasks=["arc_easy"], seed=0, **kw)


def _save(store: BaselineStore, version: str, value: float, **kw) -> None:
    store.save_release(
        spec=_spec(version),
        evaluation_config=_eval_cfg(**kw),
        evaluation=evaluation(metric("arc_easy", "acc", value), version=version),
        packages={"torch": "2.1"},
        git_sha="abc",
        performance=None,
        gate_status="PASS",
    )


def test_baseline_roundtrip(tmp_path: Path) -> None:
    store = BaselineStore(tmp_path, "my model/v1")
    assert not store.has_baseline()
    store.save_baseline(
        spec=_spec("1.0"),
        evaluation_config=_eval_cfg(),
        evaluation=evaluation(metric("arc_easy", "acc", 0.8)),
        packages={},
        git_sha=None,
        performance={"latency_ms_median": 5.0},
    )
    record = store.load_baseline()
    assert record.kind == "baseline"
    assert record.evaluation.metrics[0].value == 0.8
    assert record.performance == {"latency_ms_median": 5.0}
    assert "/" not in store.dir.name


def test_latest_release_skips_the_candidate_version(tmp_path: Path) -> None:
    store = BaselineStore(tmp_path, "m")
    _save(store, "1.0", 0.70)
    _save(store, "1.1", 0.72)
    assert store.latest_release().model["version"] == "1.1"
    assert store.latest_release(exclude_version="1.1").model["version"] == "1.0"
    assert len(store.all_releases()) == 2
    assert BaselineStore(tmp_path, "other").latest_release() is None


def test_changed_settings_make_a_record_incomparable(tmp_path: Path) -> None:
    store = BaselineStore(tmp_path, "m")
    _save(store, "1.0", 0.7, limit=100)
    record = store.latest_release()
    blocking, _ = comparability_problems(record, _eval_cfg(limit=200), {}, None)
    assert any("limit" in item for item in blocking)
    blocking, _ = comparability_problems(record, _eval_cfg(limit=100), {}, None)
    assert blocking == []


def test_package_drift_is_advisory_not_blocking(tmp_path: Path) -> None:
    store = BaselineStore(tmp_path, "m")
    _save(store, "1.0", 0.7)
    blocking, advisory = comparability_problems(store.latest_release(), _eval_cfg(), {"torch": "2.9"}, None)
    assert blocking == [] and any("torch" in item for item in advisory)
