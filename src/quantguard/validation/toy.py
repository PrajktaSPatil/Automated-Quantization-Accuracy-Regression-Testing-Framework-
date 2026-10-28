"""A small numpy classifier that stands in for an LLM benchmark.

It exists so the gate's statistics can be exercised, measured and unit-tested on any
machine in seconds. It is NOT evidence about LLM quantization. It produces
lm-eval-shaped output (accuracy with standard error, plus a perplexity-like metric
with no standard error) and is damaged by the same `fake-rtn` quantizer the real
loader uses.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from quantguard.config.schema import EvaluationConfig, ModelSpec
from quantguard.metadata.environment import EnvironmentInfo
from quantguard.models.fakequant import FakeQuantConfig, fake_quantize_matrix

_CLASSES, _DIM, _HIDDEN = 10, 64, 256


def _make_data(seed: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = np.random.default_rng(1234).normal(0, 1.0, (_CLASSES, _DIM))
    y = rng.integers(0, _CLASSES, n)
    x = centers[y] + rng.normal(0, 3.4, (n, _DIM))
    return x.astype(np.float32), y


@lru_cache(maxsize=2)
def _trained_weights(seed: int = 0) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    x, y = _make_data(seed + 1, 20000)
    dims = [(_HIDDEN, _DIM), (_HIDDEN, _HIDDEN), (_CLASSES, _HIDDEN)]
    w = [rng.normal(0, np.sqrt(2.0 / i), (o, i)).astype(np.float32) for o, i in dims]
    b = [np.zeros(o, np.float32) for o, _ in dims]
    m = [np.zeros_like(a) for a in w + b]
    v = [np.zeros_like(a) for a in w + b]
    step = 0
    for _ in range(25):
        order = rng.permutation(len(x))
        for start in range(0, len(x), 128):
            idx = order[start : start + 128]
            xb, yb = x[idx], y[idx]
            h1 = np.maximum(xb @ w[0].T + b[0], 0)
            h2 = np.maximum(h1 @ w[1].T + b[1], 0)
            logits = h2 @ w[2].T + b[2]
            p = np.exp(logits - logits.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            g = p
            g[np.arange(len(yb)), yb] -= 1
            g /= len(yb)
            gw2, gb2 = g.T @ h2, g.sum(0)
            d2 = (g @ w[2]) * (h2 > 0)
            gw1, gb1 = d2.T @ h1, d2.sum(0)
            d1 = (d2 @ w[1]) * (h1 > 0)
            gw0, gb0 = d1.T @ xb, d1.sum(0)
            grads = [gw0, gw1, gw2, gb0, gb1, gb2]
            step += 1
            params = w + b
            for k, (pa, gr) in enumerate(zip(params, grads)):
                m[k] = 0.9 * m[k] + 0.1 * gr
                v[k] = 0.999 * v[k] + 0.001 * gr * gr
                pa -= 0.002 * (m[k] / (1 - 0.9**step)) / (np.sqrt(v[k] / (1 - 0.999**step)) + 1e-8)
    return tuple(w + b)


def _forward(weights: tuple[np.ndarray, ...], x: np.ndarray) -> np.ndarray:
    w0, w1, w2, b0, b1, b2 = weights
    h = np.maximum(x @ w0.T + b0, 0)
    h = np.maximum(h @ w1.T + b1, 0)
    return h @ w2.T + b2


class ToyEvaluator:
    """Implements the EvaluationBackend protocol on the toy classifier.

    `evaluation.seed` selects which test documents are scored, so repeated runs with
    different seeds behave like independent draws of an evaluation set of size `limit`.
    The real harness scores a fixed prefix instead, which makes it deterministic.
    """

    name = "toy"

    def __init__(self, pool_size: int = 20000) -> None:
        self.x, self.y = _make_data(99, pool_size)

    def evaluate(
        self,
        spec: ModelSpec,
        evaluation: EvaluationConfig,
        environment: EnvironmentInfo,
        raw_output_path: Path,
    ) -> dict[str, Any]:
        del environment
        weights = _trained_weights(0)
        if spec.quantization_method == "fake-rtn":
            cfg = FakeQuantConfig.from_dict(spec.quantization_config)
            rng = np.random.default_rng(cfg.seed)
            quantized = [fake_quantize_matrix(w, cfg, rng) for w in weights[:3]]
            weights = tuple(quantized) + weights[3:]
        elif spec.quantization_method is not None:
            raise ValueError(f"toy backend cannot apply {spec.quantization_method}")

        n = evaluation.limit or len(self.y)
        pick = np.random.default_rng(evaluation.seed).permutation(len(self.y))[:n]
        logits = _forward(weights, self.x[pick])
        logits = logits - logits.max(1, keepdims=True)
        logp = logits - np.log(np.exp(logits).sum(1, keepdims=True))
        correct = (logits.argmax(1) == self.y[pick]).astype(np.float64)
        nll = -logp[np.arange(n), self.y[pick]]
        stderr = float(correct.std(ddof=1) / np.sqrt(n)) if n > 1 else None
        payload: dict[str, Any] = {
            "results": {
                "toy_cls": {
                    "alias": "toy_cls",
                    "acc,none": float(correct.mean()),
                    "acc_stderr,none": stderr if stderr is not None else "N/A",
                    "perplexity,none": float(np.exp(nll.mean())),
                    "perplexity_stderr,none": "N/A",
                }
            },
            "higher_is_better": {"toy_cls": {"acc": True, "perplexity": False}},
            "versions": {"toy_cls": 1},
            "n-shot": {"toy_cls": 0},
            "config": {},
            "per_sample": {"toy_cls": {"acc": {str(int(d)): float(c) for d, c in zip(pick, correct)}}}
            if evaluation.paired
            else {},
            "quantguard": {"backend_version": "toy-1", "model_version": spec.version},
        }
        raw_output_path.parent.mkdir(parents=True, exist_ok=True)
        raw_output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload
