# Configuration reference

All thresholds are YAML; see `configs/qwen2.5-0.5b.yaml` (GPU), `configs/cpu-smoke.yaml` (CPU), `configs/toy-validation.yaml`.

## `baseline` / `candidate` (ModelSpec)
`model_id`, `version`, `revision` (pin a commit hash), `precision`, `quantization_method`
(`null`, `torchao-int8-weight-only`, `fake-rtn`), `quantization_config`, `source_model_id` (required for
quantized models), `artifact_uri`, `status` (baseline must be `approved`, candidate must not).

`fake-rtn` options: `bits` 2-8, `group_size` (-1 = per row), `symmetric`, `clip_ratio` (0-1, under-estimated
range), `scale_jitter` (noisy scales), `seed`, `skip` (name fragments, default `lm_head`, `embed`).

## `evaluation`
`tasks`, `batch_size`, `limit` (documents per task; omit for full), `num_fewshot`, `device`, `dtype`,
`seed`, `paired` (default true: log per-document scores for the paired standard error).
Seeds, batch size and prompts are fixed by the config; the harness scores a fixed prefix of documents
when `limit` is set, so results are deterministic for a given environment.

## `quality_policy`
| Key | Meaning |
| --- | --- |
| `confidence_z` | one-sided z for the noise test (1.645 = 95%) |
| `on_uncertain` | `warn` (default), `fail`, `pass` when over threshold but inside noise |
| `default` | metric policy used when nothing more specific matches |
| `tasks.<task>` | task policy; `only_listed_metrics: true` gates only the metrics under `metrics` |
| `metrics.<name>` | policy for a metric name across tasks |

Metric/task policy fields: `scale` (`proportion` -> `max_regression_pp`; `relative` ->
`max_regression_percent`; `raw` -> `max_regression`), `required` (false makes a failure a WARNING),
`allow_missing`, `noise_aware` (false = fixed threshold).
Precedence: task+metric, task, global metric, default. Filtered lm-eval metrics are named
`metric,filter` (for example `exact_match,strict-match`).

## `baseline_store`
`directory`, `name` (defaults to the config name), `use_stored_baseline`, `compare_previous_release`,
`previous_release_policy` (own thresholds for release-to-release; defaults to `quality_policy`),
`version_mismatch` (`warn` or `error` on library drift). Omit the section to evaluate the baseline
every run and skip release comparison.

## `performance_policy` / `performance`
Maximum regression percent for `latency_ms`, `throughput_seq_per_s`, `gpu_memory_mib`,
`model_size_bytes`; measurement settings under `performance`. Needs CUDA for memory numbers.

## Choosing thresholds
Noise sets the smallest regression you can detect: roughly `1.645 * stderr_of_difference`. With a paired
standard error of 0.6 pp a 1 pp threshold is meaningful; with 5 pp (a `--limit 60` smoke run) it is not.
Use the CPU tier for catastrophic breakage and the GPU tier for tight thresholds.
