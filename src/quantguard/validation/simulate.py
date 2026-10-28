"""Monte Carlo study of the comparator on synthetic multiple-choice evaluations.

Question answered: at a given sample size, how often does each policy raise a
false alarm on an equivalent model, and how often does it catch a real drop?

Model of one CI run. Both models answer the same `n` questions. The baseline gets
each question right with probability `base_acc`. The candidate disagrees with the
baseline on a fraction of questions (`flip_rate`, quantization noise); a true
accuracy drop `drop_pp` is built in by making flips asymmetric. Standard errors
use the usual binomial formula per model combined as independent ("independent"),
or the standard error of the per-question difference ("paired", used when
per-example results are logged). FAIL means the drop exceeds the limit and is
larger than z standard errors; over the limit but inside the noise is WARN.

This is a simulation of the statistics, not a measurement of any model.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class SimulationResult:
    policy: str
    n: int
    drop_pp: float
    threshold_pp: float
    trials: int
    fail_rate: float
    warn_rate: float
    pass_rate: float


def _one_sided_flip_rates(base_acc: float, flip_rate: float, drop_pp: float) -> tuple[float, float]:
    """P(correct->wrong), P(wrong->correct) giving a net drop of `drop_pp` points."""

    down = flip_rate + drop_pp / 100.0 / (2.0 * base_acc)
    up = (base_acc * (down) - drop_pp / 100.0) / (1.0 - base_acc)
    return min(max(down, 0.0), 1.0), min(max(up, 0.0), 1.0)


def simulate(
    *,
    n: int,
    drop_pp: float,
    threshold_pp: float,
    trials: int = 4000,
    base_acc: float = 0.7,
    flip_rate: float = 0.03,
    z: float = 1.96,
    seed: int = 0,
) -> list[SimulationResult]:
    """Outcomes for the fixed-threshold policy and the two stderr-tolerant policies."""

    rng = np.random.default_rng(seed)
    down, up = _one_sided_flip_rates(base_acc, flip_rate, drop_pp)
    base = rng.random((trials, n)) < base_acc
    flips = rng.random((trials, n))
    cand = np.where(base, flips >= down, flips < up)

    p_base = base.mean(axis=1)
    p_cand = cand.mean(axis=1)
    regression_pp = (p_base - p_cand) * 100.0
    observed = regression_pp > threshold_pp

    # lm-eval reports the sample standard error (n-1 denominator) per model.
    se_base = np.sqrt(p_base * (1 - p_base) / max(n - 1, 1))
    se_cand = np.sqrt(p_cand * (1 - p_cand) / max(n - 1, 1))
    se_independent_pp = np.sqrt(se_base**2 + se_cand**2) * 100.0
    # Paired: standard error of the per-question difference.
    diff = base.astype(np.float64) - cand.astype(np.float64)
    se_paired_pp = diff.std(axis=1, ddof=1) / np.sqrt(n) * 100.0

    def result(policy: str, fail: np.ndarray, warn: np.ndarray) -> SimulationResult:
        return SimulationResult(
            policy, n, drop_pp, threshold_pp, trials,
            float(fail.mean()), float(warn.mean()), float(1 - fail.mean() - warn.mean()),
        )

    zeros = np.zeros(trials, dtype=bool)
    results = [result("fixed-threshold", observed, zeros)]
    for name, se in (("stderr-independent", se_independent_pp), ("stderr-paired", se_paired_pp)):
        significant = regression_pp > z * se
        results.append(result(name, observed & significant, observed & ~significant))
    return results


def sweep(
    sample_sizes: list[int],
    drops_pp: list[float],
    threshold_pp: float,
    *,
    trials: int = 4000,
    base_acc: float = 0.7,
    flip_rate: float = 0.03,
    seed: int = 0,
) -> list[SimulationResult]:
    results: list[SimulationResult] = []
    for n in sample_sizes:
        for drop in drops_pp:
            results.extend(
                simulate(
                    n=n, drop_pp=drop, threshold_pp=threshold_pp, trials=trials,
                    base_acc=base_acc, flip_rate=flip_rate, seed=seed,
                )
            )
    return results


def to_markdown(results: list[SimulationResult]) -> str:
    lines = [
        "| n | true drop (pp) | policy | FAIL rate | WARN rate |",
        "| ---: | ---: | --- | ---: | ---: |",
    ]
    for item in results:
        lines.append(
            f"| {item.n} | {item.drop_pp:g} | {item.policy} | {item.fail_rate:.1%} | {item.warn_rate:.1%} |"
        )
    return "\n".join(lines)


def as_dicts(results: list[SimulationResult]) -> list[dict]:
    return [asdict(item) for item in results]


def wilson_interval(rate: float, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a simulated event rate."""
    if trials <= 0:
        raise ValueError("trials must be positive")
    denominator = 1 + z * z / trials
    center = (rate + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / trials + z * z / (4 * trials * trials)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def required_sample_size(threshold_pp: float, base_acc: float = 0.7, z: float = 1.96) -> int:
    """Rough n at which z standard errors of the difference equal the threshold.

    Below this n the stderr band is wider than the limit, so a drop equal to the
    limit cannot be confirmed as a FAIL.
    """

    variance = 2 * base_acc * (1 - base_acc)
    return math.ceil(variance * (z * 100.0 / threshold_pp) ** 2)  # independent estimate
