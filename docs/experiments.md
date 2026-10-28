# Experiments

## Locally executed synthetic run (2026-10-28)

Command: `quantguard simulate-noise --trials 1000 --seed 0 --output docs/results/sim-1000-trials.md`.
The command also writes a machine-readable JSON companion with all parameters and 95% Wilson
intervals for simulated PASS/FAIL/WARN rates. Each cell has 1,000 Monte Carlo trials; the scenario
assumes 70% baseline accuracy, 3% per-example disagreement, and a 2 pp fixed threshold. This is
synthetic evidence about comparator behavior only, not a real model experiment.

For the paired policy at a zero-pp drop, the simulated FAIL rate was 2.3% at `n=100` (Wilson 95%
interval 1.5–3.4%), 1.8% at `n=500` (1.1–2.8%), and 0% at `n=2,000` (0–0.38%). For a true simulated
4 pp drop, paired detection was 53.9% at `n=100` (50.8–57.0%) and 99.0% at `n=500` (98.2–99.5%).
The independent-uncertainty policy is more conservative: it detected 3.9% of 4 pp drops at `n=500`,
with most other runs WARN. These results depend on the assumed 3% disagreement process. Do not cite
them as measured QuantGuard false-alarm or detection rates on quantized models.

Raw output: [`sim-1000-trials.json`](results/sim-1000-trials.json). Human-readable table:
[`sim-1000-trials.md`](results/sim-1000-trials.md).

Research question: can a cheap, low-noise CI suite reliably catch real accuracy regressions without
too many false alarms?

Status legend: **Simulated** = measured on synthetic data, shows the statistics but says nothing
about a specific model. **Tool ready** = the command exists and is tested against a fake backend, but
has not been run on a real model yet. Nothing below was measured on a real model.

| # | Experiment | Status | How to run |
| --- | --- | --- | --- |
| 1 | Sample size (`--limit`) vs noise | Simulated | `quantguard simulate-noise`; then rerun real runs at several `limit` values |
| 2 | Threshold choices | Simulated | `quantguard simulate-noise --threshold-pp X` |
| 3 | Task selection | Not run | vary `evaluation.tasks`, compare detection on `validate-injection` |
| 4 | Determinism across runs | Tool ready | `quantguard determinism --config <cfg> --runs 3` |
| 5 | Caching impact on runtime | Not run | run the workflow twice; the job summary prints runtime and `cache-hit` |
| 6 | Detection of injected regressions | Tool ready | `quantguard validate-injection --config <cfg> --scenarios configs/injection-scenarios.yaml` |
| 7 | False-alarm rate on equivalent models | Simulated, real tool ready | `control-*` scenarios plus `determinism` |

## Simulation setup (experiments 1, 2, 7)

Both models answer the same `n` multiple-choice questions. Baseline accuracy is 70%. The candidate
disagrees with the baseline on 3% of questions (quantization noise); a true drop is built in by
making those disagreements asymmetric. 5,000 trials per cell, seed 0. Policies compared:

- **fixed**: FAIL when the drop exceeds the limit (the "no noise handling" baseline of the project)
- **independent**: FAIL when the drop exceeds the limit and is larger than 1.96 x sqrt(se_b^2 + se_c^2)
- **paired**: same, using the standard error of the per-question difference (`log_samples: true`)

The 3% disagreement rate is an assumption. The paired result depends on it directly; measure the real
FP16-vs-quantized disagreement rate and rerun with `--flip-rate`.

### Sample size (limit 2 pp, equivalent model and a true 4 pp drop)

| n | false alarm: fixed | independent | paired | detect 4 pp: fixed | independent | paired |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 31.4% | 0.0% | 0.8% | 82.0% | 0.0% | 14.1% (+67.9% WARN) |
| 100 | 20.9% | 0.0% | 2.4% | 88.0% | 0.0% | 55.5% (+32.6% WARN) |
| 250 | 6.8% | 0.0% | 1.7% | 95.8% | 0.2% | 94.7% |
| 500 | 1.8% | 0.0% | 1.7% | 99.2% | 3.0% | 99.2% |
| 2500 | 0.0% | 0.0% | 0.0% | 100% | 100% | 100% |

Full table: `docs/results/sim-sample-size.md`.

Reading it:

- A plain fixed threshold false-alarms on 21% of equivalent models at n=100 and 31% at n=50.
- Independent standard errors remove the false alarms but also almost never FAIL a real drop until
  n is in the thousands, because they ignore that both models see the same questions. They turn real
  regressions into WARNINGs. This is why paired standard errors are the recommended setting.
- Paired standard errors cut false alarms to roughly 2% at small n while still failing most
  genuine 4 pp drops from n=100 and nearly all from n=250. The remaining 1-2% is the expected floor
  of a one-sided z=1.96 test (about 2.5%); raise `confidence_z` to push it lower at the cost of power.
- Where the drop is over the limit but inside the noise band the gate says WARNING, not PASS, so
  nothing is silently waved through.

### Threshold choice (n = 500)

| limit (pp) | false alarm: fixed | paired | detect 1 pp: fixed | paired | detect 3 pp: fixed | paired |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 29.6% | 2.7% | 71.0% | 19.5% (+51.5% WARN) | 99.8% | 95.0% |
| 1.0 | 15.5% | 2.7% | 52.8% | 19.5% (+33.3% WARN) | 99.3% | 95.0% |
| 2.0 | 1.8% | 1.7% | 14.5% | 14.2% | 87.8% | 87.8% |
| 3.0 | 0.1% | 0.1% | 1.8% | 1.8% | 49.4% | 49.4% |

At n=500 a limit at or below about 1 pp is under the noise floor: a fixed threshold there is mostly
firing on noise. To enforce a tight limit you need a larger `limit` (the sample count). A rough
guide, from the independent estimate: confirming a drop equal to the limit needs about
`2 p (1-p) (1.96 x 100 / limit_pp)^2` examples, e.g. about 4,000 for a 2 pp limit and 16,000 for 1 pp.
Paired errors need far fewer, in proportion to the disagreement rate.

## Experiments to run on real hardware

### 4. Determinism

`quantguard determinism --config configs/nightly-gpu.yaml --runs 3` evaluates the candidate three
times and writes max difference per metric. With fixed seeds and the same batch size the expected
result is bit-identical scores; any difference is the run-to-run variance to budget for. Also compare
across batch sizes and across devices (CPU vs GPU), since fp16 numerics change with both.

### 6 and 7. Injected regressions and equivalent models

`configs/injection-scenarios.yaml` defines three equivalent controls (8-bit per channel, 8-bit group
128, 6-bit group 64) and four known-bad candidates (2-bit, 3-bit per tensor, 4-bit per tensor as a
wrong group size, 4-bit with a badly estimated range). The runner reports regression detection rate,
false-alarm rate, warn-only rate and seconds per scenario. The labels are expectations: if a "clean"
control regresses on your model, or a "bad" one does not, relabel it and say so in the write-up.
The injector simulates the effect of each mistake; it is not GPTQ or AWQ, so also test one real
GPTQ/AWQ artifact if the release pipeline uses them.

### 3. Task selection

Run the scenario file with one task at a time (`arc_easy`, `hellaswag`, `wikitext`, ...) and compare
detection rate against cost. Perplexity (`wikitext`) is cheap and tends to move first under heavy
quantization; accuracy tasks catch behaviour changes perplexity can miss. Record which subset gives the
best detection per second.

### 5. Caching

Run the workflow twice on the same config. First run: cold cache. Second run: the job summary line
shows runtime and `cache-hit`. Record both, plus the same with the stored baseline removed
(`save_if_missing`) to quantify what the JSON baseline store saves.

## Reporting results

Add real measurements to `docs/results/` with the command, commit SHA, hardware, library versions
(they are in `report.json` under `environment`) and the raw JSON, and keep simulated and measured
numbers in separate tables.
