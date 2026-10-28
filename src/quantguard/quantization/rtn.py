"""Round-to-nearest weight fake-quantization.

This is a degradation injector, not a deployment format: weights are quantized
and immediately dequantized, so accuracy reflects the chosen bit width, group
size and scale estimate while memory and latency do not change. It exists so the
regression gate can be checked against regressions whose cause is known
(2-bit weights, a wrong group size, a badly estimated scale).

group_size semantics (along the input dimension of each Linear weight):
  positive int  one scale per `group_size` input channels
  None          one scale per output channel
  -1            one scale for the whole tensor (a typical "wrong group size" bug)

clip_ratio scales the min/max range used to compute each scale. 1.0 is plain
min/max. Values far from 1.0 model a badly calibrated range: too small clips
outliers hard, too large wastes resolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from quantguard.exceptions import ConfigurationError


@dataclass(frozen=True)
class RTNConfig:
    bits: int = 8
    group_size: int | None = None
    symmetric: bool = True
    clip_ratio: float = 1.0
    skip_modules: tuple[str, ...] = ("lm_head",)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> RTNConfig:
        allowed = {"bits", "group_size", "symmetric", "clip_ratio", "skip_modules"}
        unknown = set(raw) - allowed
        if unknown:
            raise ConfigurationError(f"Unknown rtn-fakequant options: {sorted(unknown)}")
        config = cls(
            bits=int(raw.get("bits", 8)),
            group_size=raw.get("group_size"),
            symmetric=bool(raw.get("symmetric", True)),
            clip_ratio=float(raw.get("clip_ratio", 1.0)),
            skip_modules=tuple(raw.get("skip_modules", ("lm_head",))),
        )
        if not 2 <= config.bits <= 16:
            raise ConfigurationError("rtn-fakequant bits must be between 2 and 16")
        if config.group_size is not None and config.group_size != -1 and config.group_size < 1:
            raise ConfigurationError("rtn-fakequant group_size must be null, -1, or a positive integer")
        if config.clip_ratio <= 0:
            raise ConfigurationError("rtn-fakequant clip_ratio must be positive")
        return config


def _grouped(weight: np.ndarray, group_size: int | None) -> tuple[np.ndarray, int]:
    """Return weights shaped (groups, members) and the padding added to the last axis."""

    rows, cols = weight.shape
    if group_size == -1:
        return weight.reshape(1, -1), 0
    size = cols if group_size is None else min(group_size, cols)
    pad = (-cols) % size
    padded = np.pad(weight, ((0, 0), (0, pad)), mode="edge") if pad else weight
    return padded.reshape(rows * (padded.shape[1] // size), size), pad


def fake_quantize_array(weight: np.ndarray, config: RTNConfig) -> np.ndarray:
    """Quantize then dequantize a 2-D weight matrix. Returns float32."""

    if weight.ndim != 2:
        raise ValueError("fake_quantize_array expects a 2-D weight matrix")
    original_shape = weight.shape
    work = weight.astype(np.float32)
    groups, pad = _grouped(work, config.group_size)

    if config.symmetric:
        qmax = 2 ** (config.bits - 1) - 1
        amax = np.abs(groups).max(axis=1, keepdims=True) * config.clip_ratio
        scale = np.where(amax == 0, 1.0, amax / qmax)
        quantized = np.clip(np.round(groups / scale), -qmax - 1, qmax)
        restored = quantized * scale
    else:
        levels = 2**config.bits - 1
        low = groups.min(axis=1, keepdims=True) * config.clip_ratio
        high = groups.max(axis=1, keepdims=True) * config.clip_ratio
        span = high - low
        scale = np.where(span == 0, 1.0, span / levels)
        zero = np.round(-low / scale)
        quantized = np.clip(np.round(groups / scale) + zero, 0, levels)
        restored = (quantized - zero) * scale

    if config.group_size == -1:
        return restored.reshape(original_shape).astype(np.float32)
    rows, cols = original_shape
    restored = restored.reshape(rows, cols + pad)
    return restored[:, :cols].astype(np.float32)
