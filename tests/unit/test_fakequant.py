import numpy as np
import pytest

from quantguard.exceptions import CandidateUnavailableError
from quantguard.models.fakequant import FakeQuantConfig, fake_quantize_matrix


def _weights(seed: int = 0, shape=(64, 300)) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=shape).astype(np.float32)


def _rmse(cfg: FakeQuantConfig, w: np.ndarray) -> float:
    out = fake_quantize_matrix(w, cfg, np.random.default_rng(0))
    return float(np.sqrt(((out - w) ** 2).mean()))


def test_error_grows_as_bits_shrink() -> None:
    w = _weights()
    errors = [_rmse(FakeQuantConfig(bits=b, group_size=128), w) for b in (8, 4, 3, 2)]
    assert errors == sorted(errors)
    assert errors[-1] > 10 * errors[0]


def test_smaller_groups_are_more_accurate() -> None:
    w = _weights()
    w[:, ::50] *= 8  # outliers hurt wide groups most
    assert _rmse(FakeQuantConfig(bits=4, group_size=32), w) < _rmse(FakeQuantConfig(bits=4, group_size=-1), w)


def test_clip_ratio_and_jitter_add_error() -> None:
    w = _weights()
    base = _rmse(FakeQuantConfig(bits=8, group_size=128), w)
    assert _rmse(FakeQuantConfig(bits=8, group_size=128, clip_ratio=0.4), w) > base
    assert _rmse(FakeQuantConfig(bits=8, group_size=128, scale_jitter=0.3), w) > base


def test_shape_dtype_and_non_divisible_group_size_are_preserved() -> None:
    w = _weights(shape=(8, 130)).astype(np.float16)
    out = fake_quantize_matrix(w, FakeQuantConfig(bits=4, group_size=64), np.random.default_rng(0))
    assert out.shape == w.shape and out.dtype == w.dtype


def test_asymmetric_mode_runs_and_is_close_at_8_bits() -> None:
    w = _weights()
    assert _rmse(FakeQuantConfig(bits=8, group_size=128, symmetric=False), w) < 0.05


def test_levels_are_limited_by_bit_width() -> None:
    w = _weights(shape=(4, 128))
    out = fake_quantize_matrix(w, FakeQuantConfig(bits=2, group_size=128), np.random.default_rng(0))
    assert len(np.unique(out[0])) <= 4


def test_same_seed_gives_same_result() -> None:
    w = _weights()
    cfg = FakeQuantConfig(bits=4, group_size=64, scale_jitter=0.2)
    a = fake_quantize_matrix(w, cfg, np.random.default_rng(3))
    b = fake_quantize_matrix(w, cfg, np.random.default_rng(3))
    assert np.array_equal(a, b)


@pytest.mark.parametrize("raw", [{"bits": 1}, {"bits": 9}, {"group_size": 1}, {"clip_ratio": 0}, {"nope": 1}])
def test_invalid_options_are_rejected(raw) -> None:
    with pytest.raises(CandidateUnavailableError):
        FakeQuantConfig.from_dict(raw)
