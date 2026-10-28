# Stored baselines

`quantguard baseline create --config <config>` writes `baselines/<name>/baseline.json` (the approved
FP16 scores). `quantguard run --record-release` adds `baselines/<name>/releases/<version>.json` when
the gate passes. Commit these files; they hold scores and settings only, never model weights.

A stored score is used only if tasks, limit, few-shot, seed, dtype and batch size match the current
config. Otherwise the run reports ERROR and tells you to re-create the baseline.
