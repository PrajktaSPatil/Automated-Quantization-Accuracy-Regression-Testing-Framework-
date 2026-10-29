# Real TorchAO CPU smoke evaluation

Date: 2026-10-29

This is a real model evaluation using TorchAO INT8 weight-only quantization, not the
`rtn-fakequant` injector. The candidate was configured as
`torchao-int8-weight-only`; both baseline and candidate loaded
`HuggingFaceTB/SmolLM2-135M` at revision
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`.

## Settings

- Task: `arc_easy`, 200 examples, batch size 8, seed 0
- Device and dtype: CPU, float32
- Packages: PyTorch `2.15.0+cpu`, TorchAO `0.18.0+cpu`, lm-eval `0.4.13`,
  Transformers `5.21.1`
- Performance profiling: skipped; this run provides no latency or memory results

## Results

| Metric | FP32 baseline | TorchAO INT8 candidate | Change |
| --- | ---: | ---: | ---: |
| `acc` | 0.570 | 0.565 | -0.5 percentage points |
| `acc_norm` | 0.555 | 0.560 | +0.5 percentage points |

The gate status was **PASS** with a configured maximum regression of 3 percentage
points. For `acc`, the report gives a paired standard error of 0.87 percentage
points; the observed 0.5-point drop is within the configured limit. This is a
single 200-example smoke run, not a broad detection-rate or deployment-performance
study.

Experiment report: `reports/exp_ea9363c8963f40f6910e20aa48a6305a/` (generated
reports are git-ignored). Git could not read the checkout's commit SHA during this
run because Windows reported dubious repository ownership; the report's Git SHA is
therefore unavailable.
