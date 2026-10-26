"""Internal evaluation schema. Regression does not read raw lm-eval documents."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field



class NormalizedMetric(BaseModel):
    name: str
    value: float
    standard_error: float | None = None
    higher_is_better: bool
    task: str
    model_id: str
    model_version: str
    evaluation_config_hash: str
    timestamp: str
class NormalizedEvaluation(BaseModel):
    backend: str
    backend_version: str
    model_id: str
    model_version: str
    evaluation_config_hash: str
    metrics: list[NormalizedMetric]
    raw_result_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
