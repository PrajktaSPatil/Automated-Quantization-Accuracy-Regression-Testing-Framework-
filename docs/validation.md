# Validation, experiments and spec coverage

## What was measured, and on what

Everything numeric below comes from the **toy backend**: a numpy classifier damaged by the same
`fake-rtn` quantizer used for real models. It validates the gate's statistics and plumbing. It is not
evidence about LLM accuracy. LLM results need the real runs listed under "To run on the GPU machine".

### Injected regressions through the full pipeline (toy, limit 3000)
Reproduce: `quantguard inject-suite --config configs/toy-validation.yaml --backend toy`
Full table: [injection_results_toy.md](injection_results_toy.md).

* Regression detection (FAIL on bad candidates: 2-bit, 3-bit per-channel, 4-bit bad calibration): 3/3
* False alarms (FAIL on 5 equivalent candidates): 0/5
* The ordinary 4-bit candidate (about 0.5 pp real loss) passed a 1 pp threshold, as intended.

### Fixed threshold vs noise-aware gate ([noise_study.md](noise_study.md))
100 trials per cell, 1 pp accuracy threshold, cells are `fixed / noise-aware FAIL rate`.

* **Unpaired** (baseline from a different sample or run): the fixed-threshold gate falsely failed
  equivalent models 40% of the time at limit 100-1000. The noise-aware gate: 0-5%.
* **Paired** (same documents, as in a single lm-eval run): fixed-threshold false alarms are 11% at
  limit 100 and about 0% from limit 1000, so the benefit there is mostly at small sample sizes.
* Cost of noise handling: it needs evidence. A real 3 pp loss (3-bit per-channel) is a FAIL in 87% of
  paired trials at limit 3000 but 11% at limit 100; the rest are `WARNING` (flagged, not confirmed).
  Paired standard errors matter: with the independent formula that same case was caught only 53% of the
  time at limit 3000.
* Knob: `on_uncertain: fail` gives the strict behaviour (fail on any over-threshold regression) when a
  team prefers false alarms to missed regressions.

## Spec coverage

| Requirement | Where |
| --- | --- |
| YAML config with absolute/relative thresholds per metric | `config/schema.py`, `configs/` (`max_regression_pp`, `max_regression_percent`, `max_regression`) |
| Runner with fixed seeds, batch size, prompts; perplexity | `evaluators/lm_eval.py`; `wikitext` task + `scale: relative` policy |
| Baseline store (FP16 + previous release, version metadata) | `baselines/store.py`, `quantguard baseline create`, `--record-release` |
| Comparator with statistical tolerance | `regression/engine.py` (paired / independent stderr), `policy/engine.py` |
| GitHub Actions, reusable, caching | `action.yml`, `.github/workflows/` |
| CPU smoke per PR, full GPU nightly | `pr-cpu-smoke.yml`, `nightly-gpu.yml` |
| PR comment, HTML report, trend chart | `reporting/engine.py`, `reporting/trend.py`, `pr-comment`, `trend` |
| Injected 2-bit / group size / calibration regressions | `models/fakequant.py`, `validation/injection.py`, `inject-suite` |
| Baselines: manual / fixed threshold / statistical tolerance | `noise_aware: false` vs default; `noise_study.py` |
| Pin versions and seeds | seeds in config; `scripts/freeze_constraints.sh`; library drift check in the store |
| No models in the repo | stores hold scores only |

## To run on the GPU machine (not yet executed)

1. `quantguard doctor`, then `quantguard baseline create --config configs/qwen2.5-0.5b.yaml`, commit `baselines/`.
2. `quantguard inject-suite --config configs/qwen2.5-0.5b.yaml --output reports/injection.md`
   (real-model detection and false-alarm rates).
3. Sample size vs noise: `quantguard run --config ... --limit {100,300,1000}`; read the "Noise" column.
4. Determinism: run the same config 5 times, compare scores (they should be identical on one machine);
   repeat with a different batch size to measure kernel-level variance.
5. Task selection: add tasks (for example `hellaswag`, `piqa`) and compare the paired stderr per task.
6. Caching impact: time a cold and a warm run of `pr-cpu-smoke`.
7. Pin the CPU config's `revision` to a commit hash, then `quantguard baseline create --config configs/cpu-smoke.yaml`
   and commit it so PR runs skip the baseline evaluation.

## Known limits
* The paired standard error needs per-document scores; it applies to metrics that are a mean of them
  (acc, acc_norm, exact_match). Perplexity has no standard error, so it uses a relative threshold.
* Independent-stderr fallback is conservative and under-detects small regressions.
* `fake-rtn` simulates quantization damage; size and speed are not affected.
* Multiple metrics and tasks are tested independently; with many gated metrics, expect an occasional
  false alarm at 95% confidence per metric (raise `confidence_z` if that matters).
