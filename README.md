# QuantGuard

QuantGuard compares an approved baseline model with a quantized candidate, then returns a release decision: `PASS`, `FAIL`, `ERROR`, or `WARNING`.

Quality numbers come from [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). Thresholds come from YAML. The regression engine does not assume every metric is accuracy.

## What is already verified on this machine

- PyTorch `2.12.1+cu130` sees the NVIDIA GeForce RTX 2050 (4095 MiB).
- `Qwen/Qwen2.5-0.5B` revision `060db6499f32faf8b98477b0a26969ef7d8b9987` loads in FP16 and generates text. Peak allocated memory on that run was 958 MiB.

No benchmark score is checked in. A release report is produced only by `quantguard run`.

## Layout

- `src/quantguard/config` loads and hashes YAML.
- `src/quantguard/evaluators` adapts lm-eval.
- `src/quantguard/results` normalizes backend output.
- `src/quantguard/regression` compares metrics using each metric's direction.
- `src/quantguard/policy` applies task and metric thresholds.
- `src/quantguard/gate` maps checks to PASS, FAIL, ERROR, or WARNING.
- `src/quantguard/performance` measures size, latency, throughput, and GPU memory.
- `src/quantguard/reporting` writes JSON, Markdown, and HTML.
- `src/quantguard/storage` stores experiments in SQLite.

## Setup

The project virtualenv and Hugging Face cache on this laptop are:

- `E:\quantguard-venv`
- `HF_HOME=E:\hf-cache`

```powershell
$env:HF_HOME = "E:\hf-cache"
E:\quantguard-venv\Scripts\python.exe -m pip install -e ".[gpu,dev]"
E:\quantguard-venv\Scripts\quantguard.exe doctor
E:\quantguard-venv\Scripts\quantguard.exe validate --config configs/qwen2.5-0.5b.yaml
```

## Run

Full ARC-Easy evaluation of the FP16 baseline and the torchao INT8 candidate:

```powershell
$env:HF_HOME = "E:\hf-cache"
E:\quantguard-venv\Scripts\quantguard.exe run --config configs/qwen2.5-0.5b.yaml
```

`configs/qwen2.5-0.5b-smoke.yaml` sets `limit: 8` so a short GPU path can be checked. A smoke run is not a release decision. The release config does not set a limit.

Exit codes: `PASS` and `WARNING` are 0, `FAIL` is 1, `ERROR` is 2. GitHub Actions on a self-hosted Windows GPU runner uses that code. The workflow does not reimplement the comparison.

## First quantization path

The candidate uses `torchao-int8-weight-only`. QuantGuard loads the source revision, moves it to CUDA, and applies `Int8WeightOnlyConfig`. Other methods are rejected instead of being reported as measured.

## Tests

```powershell
E:\quantguard-venv\Scripts\python.exe -m pytest
```
