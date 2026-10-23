"""Validated configuration. Thresholds live here, not in Python conditionals."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class ModelSpec(BaseModel):
    model_id: str
    version: str
    revision: str | None = None
    precision: str
    quantization_method: str | None = None
    quantization_config: dict[str, Any] = Field(default_factory=dict)
    source_model_id: str | None = None
    source_version: str | None = None
    artifact_uri: str
    status: Literal["approved", "candidate", "draft"] = "candidate"

    @field_validator("model_id", "version", "precision", "artifact_uri")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must not be empty")
        return value.strip()

    @model_validator(mode="after")
    def _quantized_candidate_has_source(self) -> ModelSpec:
        if self.quantization_method and not self.source_model_id:
            raise ValueError("quantized models require source_model_id")
        return self


class EvaluationConfig(BaseModel):
    backend: Literal["lm_eval"]
    tasks: list[str] = Field(min_length=1)
    batch_size: int = Field(default=1, ge=1)
    limit: int | None = Field(default=None, ge=1)
    num_fewshot: int | None = Field(default=None, ge=0)
    device: str = "cuda"
    dtype: Literal["float16", "bfloat16", "float32"] = "float16"
    seed: int = 0

    @field_validator("tasks")
    @classmethod
    def _tasks_non_empty(cls, tasks: list[str]) -> list[str]:
        cleaned = [task.strip() for task in tasks]
        if any(not task for task in cleaned):
            raise ValueError("task names must not be empty")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("duplicate tasks are not allowed")
        return cleaned


class MetricPolicy(BaseModel):
    """Thresholds for one metric. Scale `proportion` means lm-eval fractions (0-1)."""

    max_regression_pp: float | None = Field(default=None, ge=0)
    max_regression: float | None = Field(default=None, ge=0)
    scale: Literal["proportion", "raw"] = "raw"
    higher_is_better: bool | None = None
    required: bool = True
    allow_missing: bool = False

    @model_validator(mode="after")
    def _threshold_matches_scale(self) -> MetricPolicy:
        if self.scale == "proportion" and self.max_regression_pp is None and self.required:
            raise ValueError("proportion metrics require max_regression_pp")
        if self.scale == "raw" and self.max_regression is None and self.required:
            raise ValueError("raw metrics require max_regression")
        if self.allow_missing and self.required:
            raise ValueError("allow_missing cannot be combined with required")
        return self


class TaskPolicy(BaseModel):
    max_regression_pp: float | None = Field(default=None, ge=0)
    max_regression: float | None = Field(default=None, ge=0)
    scale: Literal["proportion", "raw"] | None = None
    required: bool = True
    allow_missing: bool = False
    metrics: dict[str, MetricPolicy] = Field(default_factory=dict)


class QualityPolicy(BaseModel):
    default: MetricPolicy
    tasks: dict[str, TaskPolicy] = Field(default_factory=dict)
    metrics: dict[str, MetricPolicy] = Field(default_factory=dict)


class PerformanceLimit(BaseModel):
    """Maximum allowed degradation relative to the baseline, in percent."""

    max_regression_percent: float = Field(ge=0)
    required: bool = True


class PerformancePolicy(BaseModel):
    latency_ms: PerformanceLimit | None = None
    throughput_seq_per_s: PerformanceLimit | None = None
    gpu_memory_mib: PerformanceLimit | None = None
    model_size_bytes: PerformanceLimit | None = None


class PerformanceMeasureConfig(BaseModel):
    warmup_iterations: int = Field(default=3, ge=0)
    measured_iterations: int = Field(default=10, ge=1)
    max_new_tokens: int = Field(default=8, ge=1)
    prompts: list[str] = Field(default_factory=lambda: ["The capital of France is"])

    @field_validator("prompts")
    @classmethod
    def _prompts(cls, prompts: list[str]) -> list[str]:
        cleaned = [prompt.strip() for prompt in prompts if prompt.strip()]
        if not cleaned:
            raise ValueError("at least one prompt is required")
        return cleaned


class QuantGuardConfig(BaseModel):
    name: str
    baseline: ModelSpec
    candidate: ModelSpec
    evaluation: EvaluationConfig
    quality_policy: QualityPolicy
    performance_policy: PerformancePolicy = Field(default_factory=PerformancePolicy)
    performance: PerformanceMeasureConfig = Field(default_factory=PerformanceMeasureConfig)
    database_path: str = "quantguard.sqlite"
    report_dir: str = "reports"

    @model_validator(mode="after")
    def _baseline_is_approved(self) -> QuantGuardConfig:
        if self.baseline.status != "approved":
            raise ValueError("baseline.status must be 'approved'")
        if self.candidate.status == "approved":
            raise ValueError("candidate.status cannot be 'approved'")
        return self
