"""End-to-end release run. Evaluation backends are injected."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quantguard.baselines.store import BaselineRecord, BaselineStore, comparability_problems
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


class _StoredPerformance:
    """Adapts a stored performance dict to the `.model_dump()` interface of a fresh profile."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def model_dump(self) -> dict[str, Any]:
        return dict(self._data)


class ReleasePipeline:
    def __init__(self, evaluator: EvaluationBackend, repo_root: Path) -> None:
        self.evaluator = evaluator
        self.repo_root = repo_root

    def run(
        self,
        config: QuantGuardConfig,
        *,
        profile: bool = True,
        record_release: bool = False,
    ) -> tuple[dict[str, Any], ReleaseDecision]:
        created_at = _utc_now()
        extra: dict[str, Any] = {"notes": [], "baseline_source": "evaluated", "previous_release": None,
                                 "previous_release_comparisons": []}
        store = self._store(config)
        experiment_id = f"exp_{uuid.uuid4().hex}"
        digest = config_hash(config)
        environment = collect_environment(self.repo_root)
        report_dir = Path(config.report_dir) / experiment_id
        if not report_dir.is_absolute():
            report_dir = self.repo_root / report_dir
        report_dir.mkdir(parents=True, exist_ok=True)

        try:
            stored_baseline = self._stored_baseline(config, store, environment, extra)
            baseline_raw = None
            if stored_baseline is None:
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
            if stored_baseline is not None:
                baseline_eval = stored_baseline.evaluation
            else:
                baseline_eval = self._normalize(
                    baseline_raw, config.baseline.model_id, config.baseline.version, digest
                )
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
            checks.extend(
                self._previous_release_checks(config, store, candidate_eval, environment, proportion, extra)
            )
            baseline_performance = None
            candidate_performance = None
            if profile and self._performance_requested(config):
                from quantguard.performance.profiler import profile_model

                if stored_baseline is not None and stored_baseline.performance:
                    baseline_performance = _StoredPerformance(stored_baseline.performance)
                else:
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
                extra=extra,
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
            extra=extra,
        )
        self._finish(config, payload, decision)
        if record_release and store is not None and decision.status in ("PASS", "WARNING"):
            path = store.save_release(
                spec=config.candidate,
                evaluation_config=config.evaluation,
                evaluation=candidate_eval,
                packages=dict(environment.packages),
                git_sha=environment.git_sha,
                performance=payload["candidate_performance"],
                gate_status=decision.status,
            )
            log_event(logger, "RELEASE_RECORDED", "stored release scores", path=str(path))
        return payload, decision

    def _store(self, config: QuantGuardConfig) -> BaselineStore | None:
        settings = config.baseline_store
        if settings is None:
            return None
        root = Path(settings.directory)
        if not root.is_absolute():
            root = self.repo_root / root
        return BaselineStore(root, settings.name or config.name)

    def _stored_baseline(
        self,
        config: QuantGuardConfig,
        store: BaselineStore | None,
        environment: EnvironmentInfo,
        extra: dict[str, Any],
    ) -> BaselineRecord | None:
        settings = config.baseline_store
        if store is None or settings is None or not settings.use_stored_baseline or not store.has_baseline():
            return None
        record = store.load_baseline()
        blocking, advisory = comparability_problems(
            record, config.evaluation, environment.packages, environment.packages.get("lm-eval")
        )
        if blocking:
            raise EvaluationError(
                "Stored baseline is not comparable with this config ("
                + "; ".join(blocking)
                + "). Re-create it with `quantguard baseline create`."
            )
        if advisory and settings.version_mismatch == "error":
            raise EvaluationError("Stored baseline was produced by different library versions: " + "; ".join(advisory))
        extra["notes"].extend(f"Version drift vs stored baseline: {item}" for item in advisory)
        if record.model.get("revision") != config.baseline.revision or record.model.get("version") != config.baseline.version:
            raise EvaluationError(
                "Stored baseline model/version differs from config.baseline; re-create the stored baseline."
            )
        extra["baseline_source"] = f"stored:{record.created_at}"
        return record

    def _previous_release_checks(
        self,
        config: QuantGuardConfig,
        store: BaselineStore | None,
        candidate_eval: NormalizedEvaluation,
        environment: EnvironmentInfo,
        proportion: set[tuple[str, str]],
        extra: dict[str, Any],
    ) -> list[PolicyCheck]:
        settings = config.baseline_store
        if store is None or settings is None or not settings.compare_previous_release:
            return []
        previous = store.latest_release(exclude_version=config.candidate.version)
        if previous is None:
            extra["notes"].append("No previous release stored; release-to-release comparison skipped.")
            return []
        blocking, advisory = comparability_problems(
            previous, config.evaluation, environment.packages, environment.packages.get("lm-eval")
        )
        if blocking:
            extra["notes"].append(
                "Previous release skipped because its evaluation settings differ: " + "; ".join(blocking)
            )
            return []
        extra["notes"].extend(f"Version drift vs previous release: {item}" for item in advisory)
        policy = settings.previous_release_policy or config.quality_policy
        keys = set(proportion)
        for metric in previous.evaluation.metrics + candidate_eval.metrics:
            if resolve_metric_policy(policy, metric.task, metric.name).scale == "proportion":
                keys.add((metric.task, metric.name))
        comparisons = compare_evaluations(previous.evaluation, candidate_eval, proportion_metrics=keys)
        extra["previous_release"] = {
            "version": previous.model.get("version"),
            "created_at": previous.created_at,
            "git_sha": previous.git_sha,
        }
        extra["previous_release_comparisons"] = [item.model_dump() for item in comparisons]
        return evaluate_quality_policy(comparisons, policy, reference="previous_release")

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
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        decision_dump = decision.model_dump()
        decision_dump["checks"] = checks or decision_dump["checks"]
        return {
            **(extra or {}),
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


def create_baseline(
    config: QuantGuardConfig,
    evaluator: EvaluationBackend,
    repo_root: Path,
    *,
    profile: bool = False,
) -> Path:
    """Evaluate the approved baseline once and store its scores as versioned JSON."""

    if config.baseline_store is None:
        raise PolicyConfigError("config has no baseline_store section; nothing to write to")
    environment = collect_environment(repo_root)
    pipeline = ReleasePipeline(evaluator, repo_root)
    store = pipeline._store(config)
    assert store is not None
    work = Path(config.report_dir)
    if not work.is_absolute():
        work = repo_root / work
    raw = evaluator.evaluate(config.baseline, config.evaluation, environment, work / "baseline_raw.json")
    evaluation = pipeline._normalize(raw, config.baseline.model_id, config.baseline.version, config_hash(config))
    performance = None
    if profile and pipeline._performance_requested(config):
        from quantguard.performance.profiler import profile_model

        performance = profile_model(
            config.baseline, config.performance, device=config.evaluation.device, dtype=config.evaluation.dtype
        ).model_dump()
    return store.save_baseline(
        spec=config.baseline,
        evaluation_config=config.evaluation,
        evaluation=evaluation,
        packages=dict(environment.packages),
        git_sha=environment.git_sha,
        performance=performance,
    )
