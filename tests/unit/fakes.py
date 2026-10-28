"""A fake evaluation backend so the pipeline can be tested without models or a GPU."""

from __future__ import annotations

import json
import zlib
from pathlib import Path
from typing import Any, Callable

import numpy as np

from quantguard.config.schema import EvaluationConfig, ModelSpec


class FakeEvaluator:
    """Both models answer the same n questions; accuracy comes from `accuracy_fn(spec)`.

    Shared per-question difficulty makes the models correlated, like a real baseline and
    its quantized copy, with a little independent flip noise per model version.
    """

    name = "fake"

    def __init__(self, accuracy_fn: Callable[[ModelSpec], float], n: int = 600, flip: float = 0.01) -> None:
        self.accuracy_fn = accuracy_fn
        self.n = n
        self.flip = flip
        self.calls: list[str] = []

    def evaluate(
        self,
        spec: ModelSpec,
        evaluation: EvaluationConfig,
        environment: Any,
        raw_output_path: Path,
    ) -> dict[str, Any]:
        self.calls.append(spec.version)
        difficulty = np.random.default_rng(1234).random(self.n)
        noise = np.random.default_rng(zlib.crc32(spec.version.encode())).random(self.n)
        correct = (difficulty < self.accuracy_fn(spec)) ^ (noise < self.flip)
        p = float(correct.mean())
        raw: dict[str, Any] = {
            "results": {
                "arc_easy": {
                    "alias": "arc_easy",
                    "acc,none": p,
                    "acc_stderr,none": float(np.sqrt(p * (1 - p) / (self.n - 1))),
                }
            },
            "higher_is_better": {"arc_easy": {"acc": True}},
            "versions": {"arc_easy": 1},
            "n-shot": {"arc_easy": 0},
            "n-samples": {"arc_easy": {"original": self.n, "effective": self.n}},
            "per_sample": (
                {"arc_easy": {"acc": {str(i): float(v) for i, v in enumerate(correct)}}}
                if evaluation.log_samples
                else None
            ),
            "quantguard": {"backend_version": "fake-0"},
        }
        raw_output_path.parent.mkdir(parents=True, exist_ok=True)
        raw_output_path.write_text(json.dumps(raw), encoding="utf-8")
        return raw


def config_dict(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "name": "unit",
        "baseline": {
            "model_id": "org/m", "version": "1", "precision": "fp16",
            "artifact_uri": "hf://org/m", "status": "approved",
        },
        "candidate": {
            "model_id": "org/m", "version": "1-q", "precision": "int8",
            "quantization_method": "rtn-fakequant", "quantization_config": {"bits": 8},
            "source_model_id": "org/m", "source_version": "1", "artifact_uri": "hf://org/m",
        },
        "evaluation": {"backend": "lm_eval", "tasks": ["arc_easy"], "log_samples": True},
        "quality_policy": {"default": {"scale": "proportion", "max_regression_pp": 2.0}},
    }
    base.update(overrides)
    return base


def bits_accuracy(spec: ModelSpec) -> float:
    """Fixture accuracy: baseline 0.70, 8-bit ~lossless, fewer bits progressively worse."""

    if spec.quantization_method is None:
        return 0.70
    bits = int(spec.quantization_config.get("bits", 8))
    return {8: 0.70, 6: 0.695, 4: 0.64, 3: 0.55, 2: 0.30}[bits]
