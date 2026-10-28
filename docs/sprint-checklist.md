# Sprint audit and completion checklist

## Initial audit

- [x] Read the project source, configs, tests, workflows, dependencies and docs before editing.
- [x] Check archive contents and preserve the ZIP. The provided archive contains no `.git` directory, so no Git history or pre-existing worktree changes were available to preserve.
- [x] Run the starting test suite. Initial attempt: 58 passed, 29 errored because Windows denied pytest's default temp root under `AppData\Local\Temp`.
- [x] Inspect available execution environments and model cache. Bundled Python 3.12.14 was usable; Python 3.11 was present by file path but execution was denied. `E:\quantguard-venv` points to a removed Python 3.11 base interpreter. RTX 2050/4 GiB and CUDA 13.0 driver are present. Cached ARC/Wikitext datasets exist; the cached pinned Qwen model's safetensors blob is 0 bytes.

## Highest-priority work

- [x] Make existing tests runnable with temp files redirected into the workspace.
- [x] Add a localhost dashboard backed by stored run reports, with safe configuration selection and run initiation.
- [x] Record full pipeline runtime in each result payload.
- [x] Add raw JSON and Wilson intervals to the synthetic Monte Carlo output.
- [x] Update the evidence ledger so implemented code is not confused with real-model validation.
- [x] Run CLI config validation and dashboard HTTP checks.
- [ ] Real CPU model evaluation — blocked by missing Torch in the usable Python environment; smoke command recorded ERROR/exit 2.
- [ ] Real GPU quantization — blocked by unusable Python 3.11 environment and incomplete cached model weights.
- [ ] Real-model injection/control and determinism experiments — blocked by the same model/runtime issues.
- [ ] GitHub Actions execution — unverified; no remote GitHub workflow was run.

## Final local validation

- `python -m pytest -q --tb=short --basetemp work/pytest-tmp/active`: 95 passed, 1 skipped on Python 3.12.14. The 34 tests under `tests/legacy_v1/` target a superseded API and are excluded from active collection.
- `python -m compileall -q src tests`: succeeded.
- `quantguard validate --config configs/ci-smoke-cpu.yaml`: succeeded.
- `quantguard dashboard --port 8765`: HTTP 200 for `/` and `/api/data`; path traversal rejected with HTTP 400.
- Synthetic study: 1,000 trials per cell, fixed seed 0. Outputs and assumptions are in `docs/results/sim-1000-trials.*`; these are not model measurements.
- Real CPU smoke: ERROR, process exit code 2, reason `No module named 'torch'`.
- `quantguard doctor`: reports Python 3.12.14, no Torch in active Python, no CUDA visibility in that Python. `nvidia-smi` separately confirms RTX 2050/4 GiB and CUDA 13.0 driver.

## Five-minute review demo

1. Show `docs/requirements-coverage.md` and explain which claims are tested versus blocked.
2. Run `quantguard validate --config configs/ci-smoke-cpu.yaml` to show config validation.
3. Run `quantguard simulate-noise --trials 1000 --seed 0 --output docs/results/demo.md`, then inspect both output files. State clearly that it is synthetic.
4. Start `quantguard dashboard`; show the stored ERROR run and that unsupported measurements are marked unavailable.
5. Show the actual error report in `reports/<experiment-id>/report.html` and the smoke log; explain that the real model evaluation stopped because the usable Python lacks Torch.
