"""Verify that the real TorchAO backend transforms weights and runs inference."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
torchao = pytest.importorskip("torchao")

from quantguard.models.loader import apply_torchao_int8


def test_torchao_int8_transforms_linear_weight_and_runs_forward() -> None:
    int8_tensor_type = torchao.quantization.Int8Tensor
    model = torch.nn.Sequential(torch.nn.Linear(8, 4, bias=False))
    sample = torch.randn(2, 8)

    quantized_model = apply_torchao_int8(model, {})

    assert quantized_model is model
    assert isinstance(model[0].weight, int8_tensor_type)
    output = model(sample)
    assert output.shape == (2, 4)
    assert torch.isfinite(output).all()


def test_torchao_int8_rejects_unsupported_group_size() -> None:
    from quantguard.exceptions import CandidateUnavailableError

    model = torch.nn.Sequential(torch.nn.Linear(8, 4, bias=False))
    with pytest.raises(CandidateUnavailableError, match="group_size"):
        apply_torchao_int8(model, {"group_size": 64})
