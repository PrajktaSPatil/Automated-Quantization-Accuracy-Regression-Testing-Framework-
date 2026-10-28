"""Quantization helpers used to inject known degradations for validation."""

from quantguard.quantization.rtn import RTNConfig, fake_quantize_array

__all__ = ["RTNConfig", "fake_quantize_array"]
