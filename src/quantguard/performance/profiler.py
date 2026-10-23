"""Real model-size, latency, throughput, and GPU-memory measurements."""

from __future__ import annotations

import time
from typing import Any

import torch
from pydantic import BaseModel, Field

from quantguard.config.schema import ModelSpec, PerformanceMeasureConfig
from quantguard.exceptions import EvaluationError
from quantguard.models.loader import load_causal_lm, parameter_storage_bytes


class PerformanceSample(BaseModel):
    model_id: str
    model_version: str
    precision: str
    quantization_method: str | None
    model_size_bytes: int
    gpu_memory_mib: float
    latency_ms_median: float
    latency_ms_p95: float
    latency_ms_mean: float
    throughput_seq_per_s: float
    methodology: dict[str, Any] = Field(default_factory=dict)


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        raise EvaluationError("latency sample is empty")
    index = min(len(sorted_values) - 1, max(0, round(fraction * (len(sorted_values) - 1))))
    return sorted_values[index]


def profile_model(
    spec: ModelSpec,
    measure: PerformanceMeasureConfig,
    *,
    device: str,
    dtype: str,
) -> PerformanceSample:
    """Measure one loaded model.

    Latency is the wall time of `generate` after CUDA synchronization.
    Warmup iterations are excluded. Throughput is completed sequences per second
    over the measured iterations only.
    """

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise EvaluationError("performance profiling requested CUDA but it is not available")

    model = None
    try:
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        model, tokenizer = load_causal_lm(spec, device=device, dtype=dtype)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        encoded = tokenizer(measure.prompts, return_tensors="pt", padding=True)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        size_bytes = parameter_storage_bytes(model)

        def _generate() -> None:
            with torch.no_grad():
                model.generate(
                    **encoded,
                    max_new_tokens=measure.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

        for _ in range(measure.warmup_iterations):
            _generate()
        if device.startswith("cuda"):
            torch.cuda.synchronize()

        samples_ms: list[float] = []
        measured_start = time.perf_counter()
        for _ in range(measure.measured_iterations):
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            started = time.perf_counter()
            _generate()
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            samples_ms.append((time.perf_counter() - started) * 1000.0)
        elapsed = time.perf_counter() - measured_start

        peak_mib = 0.0
        if device.startswith("cuda"):
            peak_mib = torch.cuda.max_memory_allocated() / (1024 * 1024)

        samples_ms.sort()
        sequence_count = measure.measured_iterations * len(measure.prompts)
        return PerformanceSample(
            model_id=spec.model_id,
            model_version=spec.version,
            precision=spec.precision,
            quantization_method=spec.quantization_method,
            model_size_bytes=size_bytes,
            gpu_memory_mib=peak_mib,
            latency_ms_median=samples_ms[len(samples_ms) // 2],
            latency_ms_p95=_percentile(samples_ms, 0.95),
            latency_ms_mean=sum(samples_ms) / len(samples_ms),
            throughput_seq_per_s=sequence_count / elapsed,
            methodology={
                "warmup_iterations": measure.warmup_iterations,
                "measured_iterations": measure.measured_iterations,
                "batch_size": len(measure.prompts),
                "prompts": measure.prompts,
                "max_new_tokens": measure.max_new_tokens,
                "do_sample": False,
                "device": device,
                "dtype": dtype,
                "gpu_name": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
                "torch_version": torch.__version__,
                "clock": "torch.cuda.synchronize then time.perf_counter around generate",
            },
        )
    except EvaluationError:
        raise
    except Exception as exc:
        raise EvaluationError(f"Performance profiling failed for {spec.model_id}: {exc}") from exc
    finally:
        if model is not None:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
