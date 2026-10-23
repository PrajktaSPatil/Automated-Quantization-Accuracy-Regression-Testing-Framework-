"""End-to-end release run. Evaluation backends are injected."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quantguard.config.loader import config_hash
from quantguard.config.schema import QuantGuardConfig
from quantguard.evaluators.base import EvaluationBackend
from quantguard.exceptions import EvaluationError, MetricMissingError, PolicyConfigError, QuantGuardError
from quantguard.gate.release import ReleaseDecision, decide
from quantguard.logging import get_logger, log_event
from quantguard.metadata.environment import EnvironmentInfo, collect_environment
from quantguard.policy.engine import (
    PolicyCheck,
    evaluate_performance_policy,
    evaluate_quality_policy,
    resolve_metric_policy,
)
from quantguard.regression.engine import compare_evaluations
from quantguard.reporting.engine import write_reports
from quantguard.results.normalizer import normalize_lm_eval
from quantguard.results.schema import NormalizedEvaluation
from quantguard.storage.sqlite import ExperimentStore

logger = get_logger("pipeline")

_PERFORMANCE_DIRECTION = {
    "latency_ms": False,
    "throughput_seq_per_s": True,
    "gpu_memory_mib": False,
    "model_size_bytes": False,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _performance_summary(sample_dump: dict[str, Any]) -> dict[str, float]:
    return {
        "latency_ms": float(sample_dump["latency_ms_median"]),
        "throughput_seq_per_s": float(sample_dump["throughput_seq_per_s"]),
        "gpu_memory_mib": float(sample_dump["gpu_memory_mib"]),
        "model_size_bytes": float(sample_dump["model_size_bytes"]),
    }


class ReleasePipeline:
    def __init__(self, evaluator: EvaluationBackend, repo_root: Path) -> None:
        self.evaluator = evaluator
        self.repo_root = repo_root

    def run(self, config: QuantGuardConfig, *, profile: bool = True) -> tuple[dict[str, Any], ReleaseDecision]:
        created_at = _utc_now()
        experiment_id = f"exp_{uuid.uuid4().hex}"
        digest = config_hash(config)
        environment = collect_environment(self.repo_root)
        report_dir = Path(config.report_dir) / experiment_id
        if not report_dir.is_absolute():
            report_dir = self.repo_root / report_dir
        report_dir.mkdir(parents=True, exist_ok=True)

        try:
            baseline_raw = self.evaluator.evaluate(
                config.baseline,
                config.evaluation,
                environment,
                report_dir / "baseline_raw.json",
            )
            candidate_raw = self.evaluator.evaluate(
                config.candidate,
                config.evaluation,
                environment,
                report_dir / "candidate_raw.json",
            )
            baseline_eval = self._normalize(baseline_raw, config.baseline.model_id, config.baseline.version, digest)
            candidate_eval = self._normalize(
                candidate_raw,
                config.candidate.model_id,
                config.candidate.version,
                digest,
            )
            proportion = self._proportion_keys(baseline_eval, candidate_eval, config)
            comparisons = compare_evaluations(
                baseline_eval,
                candidate_eval,
                proportion_metrics=proportion,
            )
            for comparison in comparisons:
                if comparison.status == "regressed":
                    log_event(
                        logger,
                        "REGRESSION_DETECTED",
                        "candidate regressed on a metric",
                        task=comparison.task,
                        metric=comparison.metric,
                        regression_amount=comparison.regression_amount,
                    )
            checks = evaluate_quality_policy(comparisons, config.quality_policy)
            baseline_performance = None
            candidate_performance = None
            if profile and self._performance_requested(config):
                from quantguard.performance.profiler import profile_model

                baseline_performance = profile_model(
                    config.baseline,
                    config.performance,
                    device=config.evaluation.device,
                    dtype=config.evaluation.dtype,
                )
                candidate_performance = profile_model(
                    config.candidate,
                    config.performance,
                    device=config.evaluation.device,
                    dtype=config.evaluation.dtype,
                )
                checks.extend(
                    evaluate_performance_policy(
                        baseline=_performance_summary(baseline_performance.model_dump()),
                        candidate=_performance_summary(candidate_performance.model_dump()),
                        policy=config.performance_policy,
                        higher_is_better=_PERFORMANCE_DIRECTION,
                    )
                )
            decision = decide(checks)
        except (MetricMissingError, PolicyConfigError, EvaluationError, QuantGuardError) as exc:
            decision = decide([], infrastructure_error=str(exc))
            payload = self._payload(
                experiment_id=experiment_id,
                created_at=created_at,
                config=config,
                digest=digest,
                environment=environment,
                decision=decision,
                comparisons=[],
                checks=decision.checks,
                baseline_eval=None,
                candidate_eval=None,
                baseline_performance=None,
                candidate_performance=None,
            )
            self._finish(config, payload, decision)
            return payload, decision

        for check in checks:
            if check.status == "fail":
                log_event(logger, "POLICY_FAILED", check.reason, metric=check.metric, required=check.required)

        payload = self._payload(
            experiment_id=experiment_id,
            created_at=created_at,
            config=config,
            digest=digest,
            environment=environment,
            decision=decision,
            comparisons=[item.model_dump() for item in comparisons],
            checks=[item.model_dump() for item in checks],
            baseline_eval=baseline_eval,
            candidate_eval=candidate_eval,
            baseline_performance=None if baseline_performance is None else baseline_performance.model_dump(),
            candidate_performance=None if candidate_performance is None else candidate_performance.model_dump(),
        )
        self._finish(config, payload, decision)
        return payload, decision

    def _normalize(
        self,
        raw: dict[str, Any],
        model_id: str,
        model_version: str,
        digest: str,
    ) -> NormalizedEvaluation:
        quantguard_meta = raw.get("quantguard") or {}
        return normalize_lm_eval(
            raw,
            model_id=model_id,
            model_version=model_version,
            evaluation_config_hash=digest,
            backend_version=str(quantguard_meta.get("backend_version") or "unknown"),
        )

    def _proportion_keys(
        self,
        baseline: NormalizedEvaluation,
        candidate: NormalizedEvaluation,
        config: QuantGuardConfig,
    ) -> set[tuple[str, str]]:
        keys = {(metric.task, metric.name) for metric in baseline.metrics}
        keys.update((metric.task, metric.name) for metric in candidate.metrics)
        selected: set[tuple[str, str]] = set()
        for task, metric in keys:
            policy = resolve_metric_policy(config.quality_policy, task, metric)
            if policy.scale == "proportion":
                selected.add((task, metric))
        return selected

    def _performance_requested(self, config: QuantGuardConfig) -> bool:
        policy = config.performance_policy
        return any(
            getattr(policy, name) is not None
            for name in ("latency_ms", "throughput_seq_per_s", "gpu_memory_mib", "model_size_bytes")
        )

    def _payload(
        self,
        *,
        experiment_id: str,
        created_at: str,
        config: QuantGuardConfig,
        digest: str,
        environment: EnvironmentInfo,
        decision: ReleaseDecision,
        comparisons: list[dict[str, Any]],
        checks: list[dict[str, Any]],
        baseline_eval: NormalizedEvaluation | None,
        candidate_eval: NormalizedEvaluation | None,
        baseline_performance: dict[str, Any] | None,
        candidate_performance: dict[str, Any] | None,
    ) -> dict[str, Any]:
        decision_dump = decision.model_dump()
        decision_dump["checks"] = checks or decision_dump["checks"]
        return {
            "experiment_id": experiment_id,
            "name": config.name,
            "created_at": created_at,
            "config_hash": digest,
            "evaluation": config.evaluation.model_dump(mode="json"),
            "baseline": config.baseline.model_dump(mode="json"),
            "candidate": config.candidate.model_dump(mode="json"),
            "environment": environment.model_dump(mode="json"),
            "baseline_evaluation": None if baseline_eval is None else baseline_eval.model_dump(mode="json"),
            "candidate_evaluation": None if candidate_eval is None else candidate_eval.model_dump(mode="json"),
            "comparisons": comparisons,
            "baseline_performance": baseline_performance,
            "candidate_performance": candidate_performance,
            "decision": decision_dump,
        }

    def _finish(self, config: QuantGuardConfig, payload: dict[str, Any], decision: ReleaseDecision) -> None:
        report_dir = Path(config.report_dir) / payload["experiment_id"]
        if not report_dir.is_absolute():
            report_dir = self.repo_root / report_dir
        write_reports(payload, report_dir)
        database = Path(config.database_path)
        if not database.is_absolute():
            database = self.repo_root / database
        ExperimentStore(database).save(payload["experiment_id"], payload)
        event = {
            "PASS": "GATE_PASSED",
            "FAIL": "GATE_FAILED",
            "ERROR": "GATE_ERROR",
            "WARNING": "GATE_WARNING",
        }[decision.status]
        log_event(logger, event, decision.reasons[0] if decision.reasons else decision.status, status=decision.status)
