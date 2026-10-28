# QUANTGUARD RELEASE REPORT

Experiment: `exp_0f7dce1a34a2418ba101ef17ca22993d`
Created: 2026-10-28T17:56:45.609006+00:00
Config hash: `89611cbd37f749faafc63c4cfdf02434afa05a8b9177e4aa5957ea51543ef208`

## Baseline

HuggingFaceTB/SmolLM2-135M / 1.0.0 / fp32
Revision: `main`
Quantization: None
Artifact: `hf://HuggingFaceTB/SmolLM2-135M@main`
Status: approved

## Candidate

HuggingFaceTB/SmolLM2-135M / 1.0.0-w8 / int8-simulated
Revision: `main`
Quantization: fake-rtn
Source: HuggingFaceTB/SmolLM2-135M / 1.0.0
Artifact: `hf://HuggingFaceTB/SmolLM2-135M@main`

## Quality (candidate vs approved baseline)

| Task | Metric | Reference | Candidate | Regression | Noise | Policy | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| arc_easy | acc | 0.416667 | 0.416667 | 0.00 pp | independent stderr 21.02 pp | maximum 3.00 pp | PASS |

Baseline source: evaluated

## Performance


## Policies

- [PASS] quality acc: regression 0.00 pp within maximum 3.00 pp

## Final release gate

PASS

Reasons:

- All required policies passed.

## Environment

- Python: 3.12.14
- Platform: Windows-11-10.0.26200-SP0
- Git SHA: `None`
- Torch: 2.14.1+cpu
- CUDA: None
- GPU: None (None MiB, cc None)
- Packages: `{"accelerate": "1.15.0", "lm-eval": "0.4.13", "numpy": "2.5.4", "pydantic": "2.14.1", "torch": "2.14.1+cpu", "transformers": "5.21.0", "typer": "0.27.3"}`
