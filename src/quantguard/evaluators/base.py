"""Evaluator interface. lm-eval is one backend, not the framework."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from quantguard.config.schema import EvaluationConfig, ModelSpec
from quantguard.metadata.environment import EnvironmentInfo


class EvaluationBackend(Protocol):
    name: str

    def evaluate(
        self,
        spec: ModelSpec,
        evaluation: EvaluationConfig,
        environment: EnvironmentInfo,
        raw_output_path: Path,
    ) -> dict[str, Any]:
        """Run the evaluation and write the raw backend document."""
