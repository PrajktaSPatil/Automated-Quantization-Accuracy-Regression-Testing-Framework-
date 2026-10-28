# CI integration

## Tiers
* `unit-tests.yml`: pytest and the toy injection suite on every PR (hosted, no GPU).
* `pr-cpu-smoke.yml`: `configs/cpu-smoke.yaml` on every PR touching code/configs. CPU torch wheel,
  small model, `--limit 60`, `--skip-performance`. Posts the PR comment.
* `nightly-gpu.yml`: full run on a `[self-hosted, gpu]` runner, nightly and on demand; also a
  `workflow_call` so another repository's release pipeline can reuse it. `record_release: true` stores
  the scores and commits `baselines/`, and the run uploads `reports/trend.html`.

## The reusable action
```yaml
- uses: actions/checkout@v4
- uses: <org>/<repo>@v1        # or ./ inside this repo
  with:
    config: configs/my-model.yaml
    install-extras: eval        # gpu on CUDA runners
    torch-index-url: https://download.pytorch.org/whl/cpu
    extra-args: --skip-performance --limit 200
```
Outputs: `status`, `report-dir`. The job fails with exit code 1 (regression) or 2 (error). The action
runs the gate, writes `pr_comment.md` and the step summary, creates or updates one PR comment (found by
a hidden marker), and uploads `reports/`.

## Caching
* pip: `actions/setup-python` cache keyed on `pyproject.toml`.
* Models and datasets: `actions/cache` on `HF_HOME`, keyed on the config file hash (which pins model
  revisions and task lists) with an OS-level restore key. Measure the effect by comparing the
  "Restore model and dataset cache" step and total runtime on a cold vs warm run.
* Stored baselines avoid re-evaluating FP16 on every run, which roughly halves evaluation time.
* Models are never committed; `baselines/` holds scores only.

## Permissions and forks
The comment step needs `pull-requests: write`. Fork PRs get a read-only token, so the comment step is
`continue-on-error`; the step summary and artifacts still carry the report.

## Pinning
Run `scripts/freeze_constraints.sh` on the trusted environment and commit
`requirements/constraints.txt`; the action installs with `-c requirements/constraints.txt` when present.
Stored baselines record library versions and the gate reports drift.
