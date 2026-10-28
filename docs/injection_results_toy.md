# Injected-regression validation

- Regression detection rate (FAIL on bad candidates): 3/3
- Flagged (FAIL or WARNING) on bad candidates: 3/3
- False-alarm rate (FAIL on equivalent candidates): 0/5
- Infrastructure errors: 0

| Scenario | Kind | Gate | First reason |
| --- | --- | --- | --- |
| w2-g128 | bad | FAIL | regression 41.43 pp exceeded maximum 1.00 pp and is beyond noise (paired stderr 1.08 pp, 1.645-sigma lower bound 39.65 pp) |
| w3-per-channel | bad | FAIL | regression 3.27 pp exceeded maximum 1.00 pp and is beyond noise (paired stderr 0.66 pp, 1.645-sigma lower bound 2.17 pp) |
| w4-bad-calibration | bad | FAIL | regression 2.73 pp exceeded maximum 1.00 pp and is beyond noise (paired stderr 0.62 pp, 1.645-sigma lower bound 1.71 pp) |
| w4-g128 | borderline | PASS | All required policies passed. |
| w8-g128-jitter-seed0 | equivalent | PASS | All required policies passed. |
| w8-g128-jitter-seed1 | equivalent | PASS | All required policies passed. |
| w8-g128-jitter-seed2 | equivalent | PASS | All required policies passed. |
| w8-g128-jitter-seed3 | equivalent | PASS | All required policies passed. |
| w8-g128-jitter-seed4 | equivalent | PASS | All required policies passed. |
