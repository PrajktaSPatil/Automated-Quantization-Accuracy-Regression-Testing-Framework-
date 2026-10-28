"""Load a Hugging Face causal LM and optionally apply torchao INT8 weights."""

from __future__ import annotations

from typing import Any

from typing import TYPE_CHECKING

from quantguard.config.schema import ModelSpec
from quantguard.exceptions import CandidateUnavailableError

if TYPE_CHECKING:  # heavy imports stay lazy so the gate logic runs without torch installed
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

SUPPORTED_QUANTIZATION = {None, "torchao-int8-weight-only", "fake-rtn"}


def load_causal_lm(
    spec: ModelSpec,
    *,
    device: str,
    dtype: str,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    if spec.quantization_method not in SUPPORTED_QUANTIZATION:
        raise CandidateUnavailableError(
            f"Unsupported quantization method: {spec.quantization_method}"
        )
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
    try:
        tokenizer = AutoTokenizer.from_pretrained(spec.model_id, revision=spec.revision)
        model = AutoModelForCausalLM.from_pretrained(
            spec.model_id,
            revision=spec.revision,
            dtype=torch_dtype,
        )
    except Exception as exc:
        raise CandidateUnavailableError(
            f"Could not load {spec.model_id} revision {spec.revision}: {exc}"
        ) from exc

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise CandidateUnavailableError("CUDA was requested but torch.cuda.is_available() is false")

    model = model.to(device)
    if spec.quantization_method == "torchao-int8-weight-only":
        model = apply_torchao_int8(model, spec.quantization_config)
    elif spec.quantization_method == "fake-rtn":
        from quantguard.models.fakequant import apply_fake_rtn

        model = apply_fake_rtn(model, spec.quantization_config)
    model.eval()
    return model, tokenizer


def apply_torchao_int8(model: PreTrainedModel, quantization_config: dict[str, Any]) -> PreTrainedModel:
    if quantization_config.get("group_size") not in (None,):
        raise CandidateUnavailableError(
            "torchao-int8-weight-only currently accepts only the default Int8WeightOnlyConfig"
        )
    try:
        from torchao.quantization import Int8WeightOnlyConfig, quantize_
    except ImportError as exc:
        raise CandidateUnavailableError(
            "torchao is not installed; cannot apply torchao-int8-weight-only"
        ) from exc
    quantize_(model, Int8WeightOnlyConfig())
    return model


def parameter_storage_bytes(model: PreTrainedModel) -> int:
    total = 0
    for parameter in model.parameters():
        storage = parameter.untyped_storage()
        total += storage.nbytes()
    return total
