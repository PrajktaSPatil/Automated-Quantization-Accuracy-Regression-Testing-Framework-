import numpy as np
import pytest

from quantguard.exceptions import ConfigurationError
from quantguard.quantization.rtn import RTNConfig, fake_quantize_array


def _weights() -> np.ndarray:
    rng = np.random.default_rng(0)
    w = rng.normal(size=(32, 130)).astype(np.float32)  # 130 is not a multiple of 64
    w[:, 5] *= 25  # outlier channel
    return w


def _rmse(config: RTNConfig) -> float:
    w = _weights()
    q = fake_quantize_array(w, config)
    assert q.shape == w.shape
    return float(np.sqrt(((q - w) ** 2).mean()))


def test_more_bits_means_less_error() -> None:
    assert _rmse(RTNConfig(8, 64)) < _rmse(RTNConfig(4, 64)) < _rmse(RTNConfig(2, 64))


def test_smaller_groups_beat_one_scale_per_tensor_with_outliers() -> None:
    assert _rmse(RTNConfig(4, 32)) < _rmse(RTNConfig(4, -1))


def test_bad_clip_ratio_hurts() -> None:
    assert _rmse(RTNConfig(4, 64, clip_ratio=0.2)) > _rmse(RTNConfig(4, 64, clip_ratio=1.0))


def test_asymmetric_mode_and_zero_matrix() -> None:
    assert _rmse(RTNConfig(8, 64, symmetric=False)) < 0.05
    zeros = fake_quantize_array(np.zeros((4, 8), dtype=np.float32), RTNConfig(4, 4))
    assert not zeros.any()


def test_values_land_on_a_small_grid() -> None:
    q = fake_quantize_array(_weights(), RTNConfig(2, None))
    assert max(len(set(np.round(row, 5))) for row in q) <= 4  # 2 bits -> at most 4 levels per row


def test_config_validation() -> None:
    with pytest.raises(ConfigurationError):
        RTNConfig.from_mapping({"bits": 1})
    with pytest.raises(ConfigurationError):
        RTNConfig.from_mapping({"typo": 3})
    with pytest.raises(ConfigurationError):
        RTNConfig.from_mapping({"group_size": 0})
