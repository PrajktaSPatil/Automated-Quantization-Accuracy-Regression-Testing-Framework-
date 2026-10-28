"""Versioned JSON baselines: scores plus the settings that produced them.

Layout (committed to git; scores only, never weights):

    baselines/<name>/baseline.json            approved FP16 reference
    baselines/<name>/releases/<version>.json  one file per released candidate

A stored score is only comparable to a new one when the evaluation settings match,
so every record carries them and `comparability_problems` checks before use.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from quantguard.config.schema import EvaluationConfig, ModelSpec
from quantguard.exceptions import BaselineNotFoundError, ConfigurationError
from quantguard.results.schema import NormalizedEvaluation

SCHEMA_VERSION = 1

# Settings that change the numbers. A mismatch makes a stored score incomparable.
_COMPARABLE_FIELDS = ("tasks", "limit", "num_fewshot", "seed", "dtype", "batch_size", "max_length")


class BaselineRecord(BaseModel):
    schema_version: int = SCHEMA_VERSION
    kind: Literal["baseline", "release"]
    created_at: str
    model: dict[str, Any]
    evaluation_settings: dict[str, Any]
    backend_version: str
    packages: dict[str, str | None] = Field(default_factory=dict)
    git_sha: str | None = None
    evaluation: NormalizedEvaluation
    performance: dict[str, Any] | None = None
    gate_status: str | None = None


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    if not cleaned:
        raise ConfigurationError(f"cannot derive a file name from {value!r}")
    return cleaned


def _settings(evaluation: EvaluationConfig) -> dict[str, Any]:
    dumped = evaluation.model_dump(mode="json")
    return {name: dumped[name] for name in _COMPARABLE_FIELDS}


def comparability_problems(
    record: BaselineRecord,
    evaluation: EvaluationConfig,
    packages: dict[str, str | None],
    backend_version: str | None,
) -> tuple[list[str], list[str]]:
    """Return (blocking, advisory). Blocking problems make the stored score unusable."""

    blocking: list[str] = []
    current = _settings(evaluation)
    for name in _COMPARABLE_FIELDS:
        if record.evaluation_settings.get(name) != current[name]:
            blocking.append(
                f"{name}: stored {record.evaluation_settings.get(name)!r} vs config {current[name]!r}"
            )
    advisory: list[str] = []
    if backend_version and record.backend_version not in ("unknown", "None") and record.backend_version != backend_version:
        advisory.append(f"lm-eval: stored {record.backend_version} vs current {backend_version}")
    for name in ("torch", "transformers", "torchao"):
        old, new = record.packages.get(name), packages.get(name)
        if old and new and old != new:
            advisory.append(f"{name}: stored {old} vs current {new}")
    return blocking, advisory


class BaselineStore:
    def __init__(self, root: Path, name: str = "default") -> None:
        self.dir = root / _slug(name)

    @property
    def baseline_path(self) -> Path:
        return self.dir / "baseline.json"

    @property
    def releases_dir(self) -> Path:
        return self.dir / "releases"

    def save_baseline(
        self,
        *,
        spec: ModelSpec,
        evaluation_config: EvaluationConfig,
        evaluation: NormalizedEvaluation,
        packages: dict[str, str | None],
        git_sha: str | None,
        performance: dict[str, Any] | None,
    ) -> Path:
        record = self._record("baseline", spec, evaluation_config, evaluation, packages, git_sha, performance, None)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._write(self.baseline_path, record)
        return self.baseline_path

    def save_release(
        self,
        *,
        spec: ModelSpec,
        evaluation_config: EvaluationConfig,
        evaluation: NormalizedEvaluation,
        packages: dict[str, str | None],
        git_sha: str | None,
        performance: dict[str, Any] | None,
        gate_status: str,
    ) -> Path:
        record = self._record("release", spec, evaluation_config, evaluation, packages, git_sha, performance, gate_status)
        self.releases_dir.mkdir(parents=True, exist_ok=True)
        path = self.releases_dir / f"{_slug(spec.version)}.json"
        self._write(path, record)
        return path

    def load_baseline(self) -> BaselineRecord:
        if not self.baseline_path.is_file():
            raise BaselineNotFoundError(
                f"No stored baseline at {self.baseline_path}. Run `quantguard baseline create` first."
            )
        return self._read(self.baseline_path)

    def has_baseline(self) -> bool:
        return self.baseline_path.is_file()

    def latest_release(self, *, exclude_version: str | None = None) -> BaselineRecord | None:
        if not self.releases_dir.is_dir():
            return None
        records = [self._read(path) for path in sorted(self.releases_dir.glob("*.json"))]
        records = [r for r in records if r.model.get("version") != exclude_version]
        if not records:
            return None
        return max(records, key=lambda r: r.created_at)

    def all_releases(self) -> list[BaselineRecord]:
        if not self.releases_dir.is_dir():
            return []
        records = [self._read(path) for path in self.releases_dir.glob("*.json")]
        return sorted(records, key=lambda r: r.created_at)

    @staticmethod
    def _record(kind, spec, evaluation_config, evaluation, packages, git_sha, performance, gate_status) -> BaselineRecord:
        return BaselineRecord(
            kind=kind,
            created_at=datetime.now(timezone.utc).isoformat(),
            model=spec.model_dump(mode="json"),
            evaluation_settings=_settings(evaluation_config),
            backend_version=evaluation.backend_version,
            packages=packages,
            git_sha=git_sha,
            evaluation=evaluation,
            performance=performance,
            gate_status=gate_status,
        )

    @staticmethod
    def _write(path: Path, record: BaselineRecord) -> None:
        path.write_text(json.dumps(record.model_dump(mode="json"), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @staticmethod
    def _read(path: Path) -> BaselineRecord:
        try:
            return BaselineRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise ConfigurationError(f"Unreadable baseline file {path}: {exc}") from exc


def fingerprint_inputs(spec: ModelSpec, evaluation: EvaluationConfig, packages: dict[str, str | None]) -> dict[str, Any]:
    """Return the model/evaluation/library identity used by compatibility callers."""
    return {"model": spec.model_dump(mode="json"), "evaluation": _settings(evaluation), "packages": dict(packages)}
