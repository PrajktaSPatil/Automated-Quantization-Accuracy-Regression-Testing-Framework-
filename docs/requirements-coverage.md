# Requirement coverage and evidence

This is an implementation and verification ledger. Passing unit/fake-backend tests do not establish
real model accuracy, CUDA behavior, or CI behavior. Status applies to code in this repository as of
the latest local test run; anything not explicitly listed as experimentally validated remains
unverified on a real model.

| Requirement | Status | Evidence / limitation |
| --- | --- | --- |
| YAML configuration, model/task/sample settings, seed, thresholds, device | UNIT-TESTED | `tests/unit/test_config.py`; `configs/` templates. Revisions in smoke configs may still be unset. |
| lm-eval integration and comparable baseline/candidate pipeline | IMPLEMENTED; integration-tested with fakes | `evaluators/lm_eval.py`, `pipeline.py`, CLI integration tests. No model downloaded or evaluated in this run. |
| Standard-error normalization and direction-aware comparison | UNIT-TESTED | Normalizer, paired/independent error and policy tests; 95 active tests pass (1 skipped). Legacy v1 tests are retained under `tests/legacy_v1/` because they target an incompatible older API. |
| Missing metrics, thresholds, baseline compatibility and gate exits | UNIT-TESTED | baseline, policy, CLI, pipeline tests. |
| Paired uncertainty | UNIT-TESTED | Requires aligned, complete per-example means; at least 30 samples. Otherwise independent SE or no uncertainty estimate. |
| Versioned JSON baselines | IMPLEMENTED; unit-tested | Save/compatibility/overwrite behavior covered. No trusted baseline is included. |
| RTN fake quantization stress injection | UNIT-TESTED; synthetic only | Quantization and injection tests. Does not represent real INT8/2-bit backend results. |
| Real INT8 TorchAO path | IMPLEMENTED; not experimentally validated here | Backend checks exist; this machine's test environment has no Torch/CUDA/lm-eval install. |
| HTML/Markdown/JSON reports and stored trend inputs | UNIT-TESTED | Report/storage tests; trend uses actual stored release records. No experiment data shipped. |
| Local dashboard | UNIT-TESTED; CLI smoke-tested | `quantguard dashboard`; reads actual SQLite and validation result files, safe config allowlist. HTTP browser session not opened in this run. |
| CPU smoke evaluation | BLOCKED | Bundled Python 3.12.14 has no Torch. `quantguard run --config configs/ci-smoke-cpu.yaml --skip-performance` produced ERROR and exit 2: `No module named 'torch'`. The configured SmolLM2 model is not cached. |
| Local GPU evaluation | BLOCKED | RTX 2050 4 GiB and CUDA 13.0 driver are visible, but active Python has no Torch. `E:\quantguard-venv` is stale: its configured Python 3.11 base executable no longer exists. Cached Qwen revision exists, but its `model.safetensors` blob is 0 bytes. |
| GitHub PR and self-hosted GPU workflows | IMPLEMENTED; remote-unverified | Workflow/action YAML exists. No GitHub connection or runner execution was available. |
| Real regression detection / false-alarm experiment | NOT EXPERIMENTALLY-VALIDATED | Existing simulation is synthetic; real model injection and unchanged-control scenarios have not been run. |
| Runtime/variance/sample-size/task/caching experiments | NOT EXPERIMENTALLY-VALIDATED | Tools/docs are present; this run did not produce experiment outputs. Runtime is now recorded per pipeline run. |
| Dependency/revision pinning | PARTIAL | Model revision fields and baseline library fingerprints exist. `constraints.txt` is absent; do not describe all dependencies as pinned. |
| Complete research claims | NOT ESTABLISHED | No real baseline/candidate scores, detection rate, false-alarm rate, or GPU performance measurements are included. |

## Statistical limits

The independent uncertainty estimate is `sqrt(SE_baseline^2 + SE_candidate^2)`. Paired estimates
use the sample standard deviation of aligned per-example score differences divided by `sqrt(n)`;
this requires the per-example mean to match the aggregate metric. Both rely on the evaluation sample
being representative and do not account for task/dataset shift. Multiple task/metric checks increase
the chance of at least one false alarm; there is no family-wise correction in the current policy.
Perplexity and non-mean metrics generally cannot use paired per-example accuracy uncertainty.

## Current locally verified evidence

- `python -m pytest -q --tb=short --basetemp work/pytest-tmp/active`: 95 passed, 1 skipped on Python 3.12.14. The 34 tests under `tests/legacy_v1/` target a superseded API and are excluded from active collection.
- `quantguard validate --config configs/ci-smoke-cpu.yaml`: succeeds and prints the parsed configuration hash.
- No real CPU/GPU evaluation or regression-injection experiment was run. See `docs/experiments.md`
  for the experiment definitions and commands.
