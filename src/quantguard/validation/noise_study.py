"""Compare a fixed-threshold gate with the noise-aware gate on the toy backend.

Answers the project's research question with measured rates: how often does each gate
flag equivalent models (false alarms) and how often does it catch damaged ones
(detection), as a function of the evaluation sample size (`--limit`).

Two evaluation modes are reported because they behave differently:
  paired    baseline and candidate are scored on the same documents (what lm-eval does
            when both are evaluated in one run).
  unpaired  they are scored on independent draws (a stored baseline from another
            sample, run, or library version).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from quantguard.config.schema import EvaluationConfig, MetricPolicy, ModelSpec, QualityPolicy, TaskPolicy
from quantguard.gate.release import decide
from quantguard.policy.engine import evaluate_quality_policy
from quantguard.regression.engine import compare_evaluations
from quantguard.results.normalizer import normalize_lm_eval
from quantguard.validation.toy import ToyEvaluator

SCENARIOS: list[tuple[str, str, dict]] = [
    ("equivalent (8-bit, jitter)", "equivalent", {"bits": 8, "group_size": 128, "scale_jitter": 0.001}),
    ("w4-g128 (mild)", "mild", {"bits": 4, "group_size": 128}),
    ("w4 bad calibration", "bad", {"bits": 4, "group_size": 128, "clip_ratio": 0.5, "scale_jitter": 0.3}),
    ("w3 per-channel", "bad", {"bits": 3, "group_size": -1}),
    ("w2-g128", "bad", {"bits": 2, "group_size": 128}),
]


def _policy(noise_aware: bool, max_pp: float) -> QualityPolicy:
    metric = MetricPolicy(scale="proportion", max_regression_pp=max_pp, noise_aware=noise_aware)
    return QualityPolicy(
        default=metric,
        tasks={"toy_cls": TaskPolicy(scale="proportion", max_regression_pp=max_pp, only_listed_metrics=True,
                                     noise_aware=noise_aware, metrics={"acc": metric})},
    )


def _spec(quant: dict | None, version: str) -> ModelSpec:
    return ModelSpec(
        model_id="toy/mlp", version=version, precision="toy", artifact_uri="toy://mlp",
        quantization_method="fake-rtn" if quant else None, quantization_config=quant or {},
        source_model_id="toy/mlp" if quant else None,
    )


def _evaluation(raw: dict, version: str):
    return normalize_lm_eval(raw, model_id="toy/mlp", model_version=version, evaluation_config_hash="x", backend_version="toy")


def run_noise_study(limits: list[int], trials: int, max_pp: float = 1.0) -> str:
    evaluator = ToyEvaluator()
    out = Path("/tmp/noise_study_raw.json")
    naive, aware = _policy(False, max_pp), _policy(True, max_pp)
    lines = [
        "# Noise study (toy backend)",
        "",
        f"Threshold: {max_pp} pp accuracy regression. {trials} trials per cell. "
        "Cells show FAIL rate for the fixed-threshold gate / noise-aware gate "
        "(noise-aware also raised WARNING in brackets).",
        "",
    ]
    for mode in ("paired", "unpaired"):
        lines += [f"## {mode.capitalize()} evaluation", "", "| Scenario | Kind | " + " | ".join(f"limit={n}" for n in limits) + " |",
                  "| --- | --- | " + " | ".join("---" for _ in limits) + " |"]
        for name, kind, quant in SCENARIOS:
            cells = []
            for n in limits:
                fail_naive = fail_aware = warn_aware = 0
                for t in range(trials):
                    base_cfg = EvaluationConfig(backend="lm_eval", tasks=["toy_cls"], limit=n, device="cpu", dtype="float32", seed=t)
                    cand_seed = t if mode == "paired" else t + 100_000
                    cand_cfg = base_cfg.model_copy(update={"seed": cand_seed})
                    base = _evaluation(evaluator.evaluate(_spec(None, "fp"), base_cfg, None, out), "fp")  # type: ignore[arg-type]
                    cand = _evaluation(evaluator.evaluate(_spec(dict(quant, seed=t), "q"), cand_cfg, None, out), "q")  # type: ignore[arg-type]
                    prop = {("toy_cls", "acc")}
                    comparisons = compare_evaluations(base, cand, proportion_metrics=prop)
                    fail_naive += decide(evaluate_quality_policy(comparisons, naive)).status == "FAIL"
                    status = decide(evaluate_quality_policy(comparisons, aware)).status
                    fail_aware += status == "FAIL"
                    warn_aware += status == "WARNING"
                cells.append(f"{fail_naive / trials:.0%} / {fail_aware / trials:.0%} [{warn_aware / trials:.0%}]")
            lines.append(f"| {name} | {kind} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limits", type=int, nargs="+", default=[100, 300, 1000, 3000])
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--max-pp", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("noise_study.md"))
    args = parser.parse_args()
    text = run_noise_study(args.limits, args.trials, args.max_pp)
    args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
