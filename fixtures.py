"""Shared builders for unit tests. Values are fixtures, not model measurements."""

from quantguard.results.schema import NormalizedEvaluation, NormalizedMetric


def metric(
    task: str,
    name: str,
    value: float,
    *,
    higher_is_better: bool = True,
    model_version: str = "v1",
) -> NormalizedMetric:
    return NormalizedMetric(
        name=name,
        value=value,
        higher_is_better=higher_is_better,
        task=task,
        model_id="fixture-model",
        model_version=model_version,
        evaluation_config_hash="abc",
        timestamp="2026-01-01T00:00:00+00:00",
    )


def evaluation(*metrics: NormalizedMetric, version: str = "v1") -> NormalizedEvaluation:
    return NormalizedEvaluation(
        backend="lm_eval",
        backend_version="test",
        model_id="fixture-model",
        model_version=version,
        evaluation_config_hash="abc",
        metrics=list(metrics),
    )
