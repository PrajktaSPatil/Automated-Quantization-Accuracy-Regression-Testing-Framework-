"""Simulated weight-only quantization used to inject known-bad quantization.

This is *fake* quantization: weights are rounded to a low-bit grid and written back in
floating point. Accuracy is realistic, size and speed are not. Its purpose is to give
the regression gate a controllable source of damage (few bits, wrong group size,
badly calibrated ranges) without needing a GPU quantization library.

The numeric core is plain numpy so it can be unit-tested without torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from quantguard.exceptions import CandidateUnavailableError

_ALLOWED_KEYS = {"bits", "group_size", "symmetric", "clip_ratio", "scale_jitter", "seed", "skip"}


@dataclass(frozen=True)
class FakeQuantConfig:
    bits: int = 8
    group_size: int = 128  # -1 means one scale per output row (per-channel)
    symmetric: bool = True
    # Fraction of the observed range used for the scale. 1.0 is a well-calibrated max range;
    # values below 1 clip outliers, modelling a calibration set that under-estimates the range.
    clip_ratio: float = 1.0
    # Relative log-normal noise on every scale, modelling miscalibrated (noisy) scales.
    scale_jitter: float = 0.0
    seed: int = 0
    skip: tuple[str, ...] = ("lm_head", "embed")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> FakeQuantConfig:
        unknown = set(raw) - _ALLOWED_KEYS
        if unknown:
            raise CandidateUnavailableError(f"unknown fake-rtn options: {sorted(unknown)}")
        cfg = cls(**{k: (tuple(v) if k == "skip" else v) for k, v in raw.items()})
        if not 2 <= cfg.bits <= 8:
            raise CandidateUnavailableError("fake-rtn bits must be between 2 and 8")
        if cfg.group_size != -1 and cfg.group_size < 2:
            raise CandidateUnavailableError("fake-rtn group_size must be -1 or >= 2")
        if not 0 < cfg.clip_ratio <= 1:
            raise CandidateUnavailableError("fake-rtn clip_ratio must be in (0, 1]")
        if cfg.scale_jitter < 0:
            raise CandidateUnavailableError("fake-rtn scale_jitter must be >= 0")
        return cfg


def fake_quantize_matrix(weight: np.ndarray, cfg: FakeQuantConfig, rng: np.random.Generator) -> np.ndarray:
    """Round-to-nearest quantize a 2-D (out, in) matrix per group along the input axis."""

    if weight.ndim != 2:
        raise ValueError("expected a 2-D weight matrix")
    out_features, in_features = weight.shape
    group = in_features if cfg.group_size == -1 or cfg.group_size > in_features else cfg.group_size
    pad = (-in_features) % group
    work = weight.astype(np.float32)
    if pad:  # repeat the last column so the padded group's range is not distorted
        work = np.concatenate([work, np.repeat(work[:, -1:], pad, axis=1)], axis=1)
    grouped = work.reshape(out_features, -1, group)

    if cfg.symmetric:
        qmax = 2 ** (cfg.bits - 1) - 1
        scale = np.abs(grouped).max(axis=2, keepdims=True) * cfg.clip_ratio / qmax
        scale = np.where(scale == 0, 1.0, scale)
        if cfg.scale_jitter:
            scale = scale * np.exp(rng.normal(0.0, cfg.scale_jitter, scale.shape)).astype(np.float32)
        quant = np.clip(np.round(grouped / scale), -qmax - 1, qmax)
        restored = quant * scale
    else:
        qmax = 2**cfg.bits - 1
        low = grouped.min(axis=2, keepdims=True) * cfg.clip_ratio
        high = grouped.max(axis=2, keepdims=True) * cfg.clip_ratio
        scale = (high - low) / qmax
        scale = np.where(scale == 0, 1.0, scale)
        if cfg.scale_jitter:
            scale = scale * np.exp(rng.normal(0.0, cfg.scale_jitter, scale.shape)).astype(np.float32)
        zero = np.round(-low / scale)
        quant = np.clip(np.round(grouped / scale) + zero, 0, qmax)
        restored = (quant - zero) * scale

    restored = restored.reshape(out_features, -1)[:, :in_features]
    return restored.astype(weight.dtype)


def apply_fake_rtn(model: Any, raw_config: dict[str, Any]) -> Any:
    """Fake-quantize every nn.Linear weight of a torch model in place."""

    import torch

    cfg = FakeQuantConfig.from_dict(raw_config)
    rng = np.random.default_rng(cfg.seed)
    touched = 0
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear) or any(token in name for token in cfg.skip):
            continue
        weight = module.weight.detach().to("cpu", dtype=torch.float32).numpy()
        restored = fake_quantize_matrix(weight, cfg, rng)
        with torch.no_grad():
            module.weight.copy_(torch.from_numpy(restored).to(module.weight.device, module.weight.dtype))
        touched += 1
    if touched == 0:
        raise CandidateUnavailableError("fake-rtn found no nn.Linear layers to quantize")
    return model
