# QuantGuard

A CI gate that stops quantization changes from silently hurting accuracy. It evaluates a quantized
candidate with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), compares it
with the approved FP16 baseline and the previous release, and returns `PASS`, `WARNING`, `FAIL` or
`ERROR` (exit codes 0, 0, 1, 2). Thresholds live in YAML.

The point of the design is that evaluation noise is real. A regression only fails the build when it is
both over its threshold and larger than sampling noise (see "How the gate decides").

## Pipeline

```
PR / new model -> CI runner -> load model -> lm-eval (fixed tasks, seed, limit) incl. wikitext perplexity
  -> compare with stored FP16 baseline + last release (JSON) using noise-aware thresholds
  -> PASS / WARNING / FAIL / ERROR  +  PR comment, HTML report, trend chart
```

## CI tiers

| Tier | Where | When | What |
| --- | --- | --- | --- |
| Unit + injected regressions | GitHub-hosted | every PR | pytest, plus `inject-suite` on a numpy toy model (no GPU, no download) |
| CPU smoke | GitHub-hosted | every PR touching the code/configs | small model, `--limit 60`, cached downloads (`configs/cpu-smoke.yaml`) |
| Full GPU | self-hosted | nightly / on demand / `workflow_call` | full tasks, tight thresholds, optional release recording (`configs/qwen2.5-0.5b.yaml`) |

Details: [docs/ci.md](docs/ci.md). The reusable piece is the composite action in `action.yml`.

## How the gate decides

For each metric the regression is `baseline - candidate` (flipped for lower-is-better metrics).

1. Within the threshold: `PASS`.
2. Over the threshold: compute the standard error of the difference and a one-sided lower bound
   `regression - z * stderr` (`z = 1.645` by default).
   * lower bound above the threshold: the regression is real, `FAIL`;
   * otherwise it cannot be separated from noise: `WARNING` (configurable with `on_uncertain: fail|pass`).
3. The standard error is **paired** when both runs scored the same documents (per-document scores are
   logged and checked against the reported mean), which is far more powerful than the independent
   formula. It falls back to the harness' standard errors, then to a plain fixed threshold for metrics
   with no standard error (perplexity).

Perplexity uses a relative threshold (`scale: relative`, `max_regression_percent`).
Measured behaviour of this rule vs a fixed threshold: [docs/noise_study.md](docs/noise_study.md).

## Commands

```bash
quantguard validate --config configs/qwen2.5-0.5b.yaml
quantguard baseline create --config configs/qwen2.5-0.5b.yaml      # once: store FP16 scores in baselines/
quantguard run --config configs/qwen2.5-0.5b.yaml                  # gate; add --limit N, --skip-performance
quantguard run --config ... --record-release                       # store candidate scores if the gate passes
quantguard inject-suite --config configs/cpu-smoke.yaml            # inject 2-bit / bad group size / bad calibration
quantguard inject-suite --config configs/toy-validation.yaml --backend toy   # same, no GPU, ~1 minute
quantguard pr-comment --report reports/<exp>/report.json           # Markdown for the PR
quantguard trend --config configs/qwen2.5-0.5b.yaml                # HTML trend chart over releases
quantguard history                                                 # past runs (SQLite)
```

Reports for every run (`reports/<experiment>/`): `report.json`, `report.md`, `report.html`.

## Stored baselines

`baselines/<name>/baseline.json` and `baselines/<name>/releases/<version>.json` hold scores, per-document
results, settings and library versions, never weights. A stored score is reused only when tasks, limit,
few-shot, seed, dtype and batch size match; otherwise the run is an `ERROR` that says how to recreate it.
Library drift (lm-eval, torch, transformers, torchao) is reported, or made an error with
`baseline_store.version_mismatch: error`. See [baselines/README.md](baselines/README.md).

## Injecting regressions

`quantization_method: fake-rtn` rounds every `nn.Linear` weight to a low-bit grid (options: `bits`,
`group_size`, `symmetric`, `clip_ratio`, `scale_jitter`, `seed`). It simulates damage; it does not shrink
the model. Real compression uses `torchao-int8-weight-only`. The injection suite runs 2-bit, wrong group
size and poor calibration scenarios plus equivalent (should-pass) candidates, and reports detection rate
and false-alarm rate.

## Setup

```bash
pip install -e ".[dev]"              # gate logic, toy backend, tests (no torch needed)
pip install -e ".[eval]"             # + lm-eval/transformers (install a CPU torch yourself)
pip install -e ".[gpu]"              # + torch, torchao
bash scripts/freeze_constraints.sh   # pin versions once, commit requirements/constraints.txt
pytest -q
```

Windows GPU laptop used during development: venv `E:\quantguard-venv`, `HF_HOME=E:\hf-cache`. PyTorch sees
the RTX 2050 (4 GiB); `Qwen/Qwen2.5-0.5B` loads in FP16 (958 MiB peak).

## Layout

`src/quantguard/`: `config` (schema, hash), `evaluators` (lm-eval adapter), `results` (normalizer),
`regression` (paired/independent stderr), `policy` (thresholds), `gate`, `baselines`, `models`
(loader, `fakequant`), `performance`, `reporting` (md/html/PR comment/trend), `storage` (SQLite),
`validation` (injection suite, toy backend, noise study), `pipeline`, `cli`.

More: [docs/configuration.md](docs/configuration.md), [docs/ci.md](docs/ci.md), [docs/validation.md](docs/validation.md).
