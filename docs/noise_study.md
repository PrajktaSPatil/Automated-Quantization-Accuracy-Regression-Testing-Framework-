# Noise study (toy backend)

Threshold: 1.0 pp accuracy regression. 100 trials per cell. Cells show FAIL rate for the fixed-threshold gate / noise-aware gate (noise-aware also raised WARNING in brackets).

## Paired evaluation

| Scenario | Kind | limit=100 | limit=300 | limit=1000 | limit=3000 |
| --- | --- | --- | --- | --- | --- |
| equivalent (8-bit, jitter) | equivalent | 11% / 0% [11%] | 2% / 0% [2%] | 0% / 0% [0%] | 0% / 0% [0%] |
| w4-g128 (mild) | mild | 51% / 2% [49%] | 45% / 3% [42%] | 32% / 2% [30%] | 12% / 2% [10%] |
| w4 bad calibration | bad | 61% / 6% [55%] | 69% / 13% [56%] | 72% / 18% [54%] | 91% / 32% [59%] |
| w3 per-channel | bad | 73% / 11% [62%] | 79% / 22% [57%] | 94% / 47% [47%] | 100% / 87% [13%] |
| w2-g128 | bad | 100% / 100% [0%] | 100% / 100% [0%] | 100% / 100% [0%] | 100% / 100% [0%] |

## Unpaired evaluation

| Scenario | Kind | limit=100 | limit=300 | limit=1000 | limit=3000 |
| --- | --- | --- | --- | --- | --- |
| equivalent (8-bit, jitter) | equivalent | 40% / 5% [35%] | 42% / 3% [39%] | 41% / 2% [39%] | 23% / 0% [23%] |
| w4-g128 (mild) | mild | 46% / 5% [41%] | 43% / 3% [40%] | 43% / 3% [40%] | 36% / 0% [36%] |
| w4 bad calibration | bad | 52% / 7% [45%] | 55% / 8% [47%] | 72% / 7% [65%] | 77% / 18% [59%] |
| w3 per-channel | bad | 56% / 8% [48%] | 65% / 15% [50%] | 85% / 28% [57%] | 96% / 51% [45%] |
| w2-g128 | bad | 100% / 100% [0%] | 100% / 100% [0%] | 100% / 100% [0%] | 100% / 100% [0%] |
