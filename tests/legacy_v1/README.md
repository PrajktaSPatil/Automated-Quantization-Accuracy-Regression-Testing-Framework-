# Legacy API tests

These test modules target the earlier QuantGuard API (label-based baseline store,
`log_samples`, and the `ScenarioOutcome` injection API). The active source in this
project uses named baseline stores, `EvaluationConfig.paired`, and `run_suite`.
The tests are retained here for reference and are excluded from default pytest
collection so the active suite only tests the current API. Equivalent current
coverage is in `test_baseline_store.py`, `test_pipeline_toy.py`,
`test_noise_aware_policy.py`, and `test_pipeline_toy.py` injection tests.
