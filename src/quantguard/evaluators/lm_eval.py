"""lm-evaluation-harness adapter."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Protocol

from quantguard.config.schema import EvaluationConfig, ModelSpec
from quantguard.exceptions import EvaluationError
from quantguard.logging import get_logger, log_event
from quantguard.metadata.environment import EnvironmentInfo
from quantguard.models.loader import load_causal_lm

logger = get_logger("evaluators.lm_eval")


class RawEvaluation(Protocol):
    raw: dict[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def lm_eval_version() -> str:
    try:
        return metadata.version("lm-eval")
    except metadata.PackageNotFoundError as exc:
        raise EvaluationError("lm-eval is not installed") from exc


class LMEvalEvaluator:
    """Runs lm-eval against a model already loaded by QuantGuard."""

    name = "lm_eval"

    def evaluate(
        self,
        spec: ModelSpec,
        evaluation: EvaluationConfig,
        environment: EnvironmentInfo,
        raw_output_path: Path,
    ) -> dict[str, Any]:
        del environment  # recorded by the caller; the backend must not invent it
        log_event(
            logger,
            "EVALUATION_STARTED",
            "starting lm-eval",
            model_id=spec.model_id,
            version=spec.version,
            tasks=evaluation.tasks,
        )
        started = _utc_now()
        model = None
        try:
            import torch
            from lm_eval import simple_evaluate
            from lm_eval.models.huggingface import HFLM

            model, tokenizer = load_causal_lm(
                spec,
                device=evaluation.device,
                dtype=evaluation.dtype,
            )
            lm = HFLM(
                pretrained=model,
                tokenizer=tokenizer,
                batch_size=evaluation.batch_size,
                dtype=evaluation.dtype,
                device=evaluation.device,
            )
            results = simple_evaluate(
                model=lm,
                tasks=evaluation.tasks,
                batch_size=evaluation.batch_size,
                limit=evaluation.limit,
                device=evaluation.device,
                random_seed=evaluation.seed,
                numpy_random_seed=evaluation.seed,
                torch_random_seed=evaluation.seed,
                fewshot_random_seed=evaluation.seed,
                num_fewshot=evaluation.num_fewshot,
                log_samples=False,
            )
        except EvaluationError:
            raise
        except Exception as exc:
            raise EvaluationError(f"lm-eval failed for {spec.model_id}@{spec.version}: {exc}") from exc
        finally:
            if model is not None:
                del model
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

        if not isinstance(results, dict) or "results" not in results:
            raise EvaluationError("lm-eval returned no results object")

        payload = {
            "results": results.get("results"),
            "higher_is_better": results.get("higher_is_better"),
            "versions": results.get("versions"),
            "n-shot": results.get("n-shot"),
            "config": _json_safe(results.get("config")),
            "quantguard": {
                "started_at": started,
                "finished_at": _utc_now(),
                "model_id": spec.model_id,
                "model_version": spec.version,
                "revision": spec.revision,
                "precision": spec.precision,
                "quantization_method": spec.quantization_method,
                "tasks": evaluation.tasks,
                "batch_size": evaluation.batch_size,
                "limit": evaluation.limit,
                "num_fewshot": evaluation.num_fewshot,
                "seed": evaluation.seed,
                "dtype": evaluation.dtype,
                "device": evaluation.device,
                "backend_version": lm_eval_version(),
            },
        }
        raw_output_path.parent.mkdir(parents=True, exist_ok=True)
        raw_output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        log_event(
            logger,
            "EVALUATION_COMPLETED",
            "lm-eval finished",
            model_id=spec.model_id,
            version=spec.version,
            raw_output=str(raw_output_path),
        )
        return payload


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
    except TypeError:
        return json.loads(json.dumps(value, default=str))
    return value
