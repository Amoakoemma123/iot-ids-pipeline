# Paper 2 supplementary material (double-brace placeholders are filled from values.json by build.py --supp)

## Title

**Supplementary material for: What Can an Attacker Actually Change? Evasion and Hardening of Compact Flow-Based Intrusion Detectors Under a Realistic Threat Model**

## Contents

This supplement gives the tables referred to in the main text as Tables S1–S14. Every number is computed from the released result files by the same script as the main text (https://github.com/Amoakoemma123/iot-ids-pipeline). ASR is the share of detected attack flows of the attack set classified as benign after the attack; all values are means over the seeds stated in each caption.

## S1. Saturation of the black-box search

| Dataset | Model | Grid of 24 points | Standard grid | Grid + 1,000 points | Grid + adaptive (TTL kept) | Grid + adaptive (any TTL) | Strongest |
|---|---|---|---|---|---|---|---|
| TON-IoT | CNN | 45.9% | 47.9% | 49.3% | 51.0% | n/a | 51.4% |
| TON-IoT | MLP | 60.0% | 61.5% | 63.1% | 64.7% | n/a | 64.8% |
| TON-IoT | RF | 23.8% | 24.5% | 24.9% | 25.1% | n/a | 25.2% |
| TON-IoT | GBDT | 23.5% | 25.0% | 29.8% | 34.5% | n/a | 35.3% |
| UNSW-NB15 | CNN | 58.7% | 58.6% | 59.9% | 61.3% | 99.3% | 99.3% |
| UNSW-NB15 | MLP | 57.8% | 58.3% | 62.6% | 65.8% | 95.3% | 95.4% |
| UNSW-NB15 | RF | 11.3% | 12.0% | 16.1% | 18.5% | 87.2% | 89.7% |
| UNSW-NB15 | GBDT | 17.8% | 18.7% | 28.5% | 32.2% | 71.4% | 75.0% |

## S2. Per-action ablation and budget floors

The ablation and floor grids are scaled copies of one unit grid, so the candidates of a larger budget do not contain those of a smaller one; this is why a few entries are not monotone (for example bytes only above bytes and packets for the UNSW-NB15 random forest).

| Dataset | Model | Bytes only | Packets only | Delays only | Bytes + packets | All (volume only) | Floors × 0 | Floors × 0.5 | Floors × 2 |
|---|---|---|---|---|---|---|---|---|---|
| TON-IoT | CNN | 12.7% | 11.1% | 0.9% | 26.1% | 47.9% | 4.5% | 37.1% | 56.2% |
| TON-IoT | MLP | 40.4% | 23.5% | 6.5% | 56.1% | 61.5% | 8.5% | 55.1% | 65.4% |
| TON-IoT | RF | 7.9% | 8.1% | 0.1% | 11.7% | 24.5% | 3.5% | 21.3% | 25.3% |
| TON-IoT | GBDT | 11.8% | 6.9% | 0.5% | 18.6% | 25.0% | 3.0% | 19.0% | 44.9% |
| UNSW-NB15 | CNN | 33.3% | 22.8% | 30.8% | 42.5% | 58.6% | 35.7% | 54.3% | 62.2% |
| UNSW-NB15 | MLP | 39.0% | 25.7% | 24.8% | 41.4% | 58.3% | 42.8% | 53.4% | 61.7% |
| UNSW-NB15 | RF | 5.4% | 4.7% | 1.0% | 4.9% | 12.0% | 5.9% | 11.8% | 11.7% |
| UNSW-NB15 | GBDT | 5.7% | 12.0% | 2.0% | 15.3% | 18.7% | 9.5% | 17.1% | 18.9% |

## S3. End-to-end evasion and prevalence-weighted ASR

| Dataset | Model | Missed before attack | Overall evasion, volume only | Overall evasion, all actions | Prevalence-weighted ASR, all actions |
|---|---|---|---|---|---|
| TON-IoT | CNN | 1.6% | 52.2% | 52.2% | 52.1% |
| TON-IoT | MLP | 7.4% | 67.5% | 67.5% | 65.5% |
| TON-IoT | RF | 0.1% | 25.2% | 25.2% | 24.3% |
| TON-IoT | GBDT | 0.5% | 35.6% | 35.6% | 35.3% |
| UNSW-NB15 | CNN | 28.8% | 72.4% | 99.5% | 99.1% |
| UNSW-NB15 | MLP | 38.2% | 78.7% | 97.1% | 94.5% |
| UNSW-NB15 | RF | 0.5% | 19.8% | 89.7% | 87.9% |
| UNSW-NB15 | GBDT | 0.6% | 34.7% | 75.2% | 74.0% |

## S4. Cheapest successful action

For each flow that the grid search evades without a TTL change, the smallest share of the budget used by any successful candidate, and the absolute amounts of that candidate (medians over flows, means over seeds).

| Dataset | Model | Class | Seeds | Share of budget | Bytes | Packets | Delay (s) |
|---|---|---|---|---|---|---|---|
| TON-IoT | CNN | backdoor | 5 | 0.56 | 148 | 2.6 | 0.17 |
| TON-IoT | CNN | ddos | 5 | 0.77 | 82 | 4.2 | 0.24 |
| TON-IoT | CNN | dos | 5 | 0.43 | 79 | 5.8 | 0.16 |
| TON-IoT | CNN | injection | 5 | 0.68 | 321 | 3.3 | 0.18 |
| TON-IoT | CNN | mitm | 5 | 0.69 | 318 | 2.9 | 0.13 |
| TON-IoT | CNN | password | 5 | 0.59 | 253 | 2.8 | 0.09 |
| TON-IoT | CNN | ransomware | 5 | 0.29 | 78 | 1.4 | 0.11 |
| TON-IoT | CNN | scanning | 5 | 0.27 | 78 | 1.2 | 0.11 |
| TON-IoT | CNN | xss | 5 | 0.56 | 187 | 2.7 | 0.36 |
| TON-IoT | GBDT | backdoor | 5 | 0.96 | 365 | 4.4 | 0.16 |
| TON-IoT | GBDT | ddos | 5 | 0.69 | 237 | 2.4 | 0.24 |
| TON-IoT | GBDT | dos | 5 | 0.82 | 422 | 1.7 | 0.25 |
| TON-IoT | GBDT | injection | 5 | 0.80 | 431 | 4.8 | 0.22 |
| TON-IoT | GBDT | mitm | 5 | 0.82 | 481 | 3.5 | 0.26 |
| TON-IoT | GBDT | password | 5 | 0.74 | 360 | 2.7 | 0.02 |
| TON-IoT | GBDT | ransomware | 5 | 0.54 | 133 | 2.2 | 0.14 |
| TON-IoT | GBDT | scanning | 5 | 0.34 | 130 | 1.4 | 0.11 |
| TON-IoT | GBDT | xss | 5 | 0.78 | 391 | 3.6 | 0.72 |
| UNSW-NB15 | CNN | analysis | 5 | 0.49 | 156 | 1.9 | 0.24 |
| UNSW-NB15 | CNN | backdoor | 5 | 0.56 | 168 | 1.6 | 0.24 |
| UNSW-NB15 | CNN | dos | 5 | 0.55 | 219 | 2.4 | 0.24 |
| UNSW-NB15 | CNN | exploits | 5 | 0.55 | 329 | 3.1 | 0.25 |
| UNSW-NB15 | CNN | fuzzers | 5 | 0.56 | 268 | 2.0 | 0.28 |
| UNSW-NB15 | CNN | generic | 5 | 0.42 | 74 | 1.6 | 0.10 |
| UNSW-NB15 | CNN | reconnaissance | 5 | 0.63 | 173 | 2.0 | 0.32 |
| UNSW-NB15 | CNN | shellcode | 5 | 0.70 | 141 | 3.0 | 0.32 |
| UNSW-NB15 | CNN | worms | 5 | 0.71 | 664 | 3.3 | 0.41 |
| UNSW-NB15 | GBDT | analysis | 5 | 0.71 | 183 | 3.6 | 0.18 |
| UNSW-NB15 | GBDT | backdoor | 5 | 0.76 | 178 | 3.4 | 0.13 |
| UNSW-NB15 | GBDT | dos | 5 | 0.76 | 167 | 3.7 | 0.17 |
| UNSW-NB15 | GBDT | exploits | 5 | 0.76 | 191 | 4.9 | 0.32 |
| UNSW-NB15 | GBDT | fuzzers | 5 | 0.67 | 102 | 4.0 | 0.18 |
| UNSW-NB15 | GBDT | generic | 5 | 0.52 | 86 | 2.7 | 0.16 |
| UNSW-NB15 | GBDT | reconnaissance | 5 | 0.79 | 297 | 3.8 | 0.12 |
| UNSW-NB15 | GBDT | shellcode | 5 | 0.87 | 192 | 4.0 | 0.07 |
| UNSW-NB15 | GBDT | worms | 2 | 1.00 | 0 | 24.0 | 0.00 |

## S5. Further data splits

| Dataset | Model | Split | Clean macro-F1 | Grid, volume only | Grid, both | Strongest, all actions |
|---|---|---|---|---|---|---|
| TON-IoT | CNN | main | 0.948 | 42.1% | 42.1% | 45.8% |
| TON-IoT | CNN | 1 | 0.952 | 46.3% | 46.3% | 51.5% |
| TON-IoT | CNN | 2 | 0.947 | 43.5% | 43.5% | 44.8% |
| TON-IoT | MLP | main | 0.948 | 53.8% | 53.8% | 57.7% |
| TON-IoT | MLP | 1 | 0.950 | 76.1% | 76.1% | 80.7% |
| TON-IoT | MLP | 2 | 0.949 | 66.7% | 66.7% | 69.2% |
| TON-IoT | RF | main | 0.955 | 23.5% | 23.5% | 24.1% |
| TON-IoT | RF | 1 | 0.957 | 19.4% | 19.4% | 20.5% |
| TON-IoT | RF | 2 | 0.955 | 22.5% | 22.5% | 23.7% |
| TON-IoT | GBDT | main | 0.961 | 25.1% | 25.1% | 34.2% |
| TON-IoT | GBDT | 1 | 0.964 | 31.6% | 31.6% | 36.7% |
| TON-IoT | GBDT | 2 | 0.962 | 32.5% | 32.5% | 36.9% |
| TON-IoT | CNN, AT-threat | main | 0.902 | 3.9% | 3.9% | 5.6% |
| TON-IoT | CNN, AT-threat | 1 | 0.890 | 0.0% | 0.0% | 0.0% |
| TON-IoT | CNN, AT-threat | 2 | 0.916 | 0.0% | 0.0% | 0.0% |
| TON-IoT | RF, augmented | main | 0.954 | 9.1% | 9.1% | 9.4% |
| TON-IoT | RF, augmented | 1 | 0.957 | 9.0% | 9.0% | 9.7% |
| TON-IoT | RF, augmented | 2 | 0.954 | 9.4% | 9.4% | 10.1% |
| TON-IoT | GBDT, augmented | main | 0.963 | 0.3% | 0.3% | 2.7% |
| TON-IoT | GBDT, augmented | 1 | 0.961 | 2.6% | 2.6% | 4.8% |
| TON-IoT | GBDT, augmented | 2 | 0.962 | 2.2% | 2.2% | 3.3% |
| UNSW-NB15 | CNN | main | 0.557 | 62.9% | 99.3% | 99.5% |
| UNSW-NB15 | CNN | 1 | 0.552 | 54.9% | 99.7% | 99.8% |
| UNSW-NB15 | CNN | 2 | 0.535 | 68.0% | 99.8% | 99.8% |
| UNSW-NB15 | MLP | main | 0.561 | 65.2% | 86.6% | 88.0% |
| UNSW-NB15 | MLP | 1 | 0.552 | 77.4% | 99.6% | 99.7% |
| UNSW-NB15 | MLP | 2 | 0.539 | 58.0% | 98.5% | 98.7% |
| UNSW-NB15 | RF | main | 0.603 | 12.6% | 79.9% | 85.0% |
| UNSW-NB15 | RF | 1 | 0.605 | 11.1% | 80.2% | 82.5% |
| UNSW-NB15 | RF | 2 | 0.593 | 13.2% | 83.1% | 86.7% |
| UNSW-NB15 | GBDT | main | 0.610 | 17.5% | 54.5% | 71.1% |
| UNSW-NB15 | GBDT | 1 | 0.603 | 22.3% | 55.3% | 73.0% |
| UNSW-NB15 | GBDT | 2 | 0.576 | 24.6% | 51.5% | 69.4% |
| UNSW-NB15 | CNN, AT-threat | main | 0.495 | 9.2% | 48.3% | 53.1% |
| UNSW-NB15 | CNN, AT-threat | 1 | 0.494 | 5.1% | 21.1% | 25.2% |
| UNSW-NB15 | CNN, AT-threat | 2 | 0.504 | 4.2% | 17.4% | 20.1% |
| UNSW-NB15 | RF, augmented | main | 0.601 | 0.1% | 0.2% | 31.5% |
| UNSW-NB15 | RF, augmented | 1 | 0.598 | 0.1% | 2.3% | 32.8% |
| UNSW-NB15 | RF, augmented | 2 | 0.588 | 0.1% | 0.2% | 30.5% |
| UNSW-NB15 | GBDT, augmented | main | 0.605 | 0.0% | 0.3% | 40.2% |
| UNSW-NB15 | GBDT, augmented | 1 | 0.611 | 0.0% | 0.8% | 41.5% |
| UNSW-NB15 | GBDT, augmented | 2 | 0.580 | 0.0% | 0.2% | 41.3% |

## S6. Strongest-attack ASR per attack class

| Dataset | Class | Test share of attack flows | CNN | MLP | RF | GBDT |
|---|---|---|---|---|---|---|
| TON-IoT | backdoor | 13.2% | 33.4% | 73.3% | 0.2% | 19.4% |
| TON-IoT | ddos | 14.1% | 43.1% | 54.5% | 21.5% | 11.7% |
| TON-IoT | dos | 10.4% | 11.4% | 15.9% | 9.8% | 1.3% |
| TON-IoT | injection | 14.1% | 38.9% | 52.6% | 0.9% | 10.1% |
| TON-IoT | mitm | 0.7% | 10.4% | 46.4% | 1.9% | 14.9% |
| TON-IoT | password | 14.0% | 61.5% | 74.2% | 1.7% | 57.4% |
| TON-IoT | ransomware | 10.1% | 95.4% | 98.4% | 96.3% | 75.5% |
| TON-IoT | scanning | 13.1% | 88.7% | 94.8% | 76.8% | 77.1% |
| TON-IoT | xss | 10.3% | 48.6% | 57.7% | 0.6% | 35.3% |
| UNSW-NB15 | analysis | 2.4% | 97.3% | 99.9% | 99.8% | 99.8% |
| UNSW-NB15 | backdoor | 2.4% | 98.1% | 99.2% | 98.4% | 95.1% |
| UNSW-NB15 | dos | 6.5% | 98.6% | 93.0% | 88.3% | 77.5% |
| UNSW-NB15 | exploits | 36.2% | 98.9% | 90.4% | 74.0% | 63.3% |
| UNSW-NB15 | fuzzers | 29.3% | 99.3% | 96.3% | 94.6% | 90.5% |
| UNSW-NB15 | generic | 7.6% | 99.8% | 99.0% | 96.5% | 91.9% |
| UNSW-NB15 | reconnaissance | 13.4% | 99.7% | 97.4% | 99.9% | 47.0% |
| UNSW-NB15 | shellcode | 2.0% | 99.9% | 97.0% | 99.9% | 74.9% |
| UNSW-NB15 | worms | 0.2% | 100.0% | 91.6% | 93.7% | 48.4% |

## S7. Defended CNNs under the feature-space reference attacks

| Dataset | Model | White-box ε-PGD (as in training) | BPDA PGD, full ball | Black-box corners, full ball | Black-box, mutable features ± | Black-box, mutable, increases only | Realistic strongest |
|---|---|---|---|---|---|---|---|
| TON-IoT | CNN | 75.1% | 90.6% | 100.0% | 95.6% | 73.5% | 51.4% |
| TON-IoT | CNN, clean loop | 64.1% | 71.1% | 100.0% | 77.3% | 38.7% | 26.2% |
| TON-IoT | CNN, AT-all | 27.5% | 91.4% | 100.0% | 64.2% | 33.7% | 20.9% |
| TON-IoT | CNN, AT-all (BPDA) | 55.6% | 46.7% | 100.0% | 49.4% | 21.1% | 16.9% |
| TON-IoT | CNN, CCAT | 27.0% | 84.1% | 100.0% | 62.7% | 25.8% | 19.1% |
| TON-IoT | CNN, AT-threat | 77.9% | 88.4% | 100.0% | 67.9% | 36.4% | 5.5% |
| TON-IoT | CNN, TRADES | 81.2% | 98.1% | 100.0% | 72.2% | 28.4% | 17.1% |
| UNSW-NB15 | CNN | 63.2% | 96.5% | 100.0% | 100.0% | 95.2% | 99.3% |
| UNSW-NB15 | CNN, clean loop | 46.5% | 84.8% | 98.2% | 96.8% | 65.9% | 81.9% |
| UNSW-NB15 | CNN, AT-all | 19.5% | 54.0% | 63.7% | 58.4% | 28.6% | 99.0% |
| UNSW-NB15 | CNN, AT-all (BPDA) | 21.2% | 26.7% | 82.5% | 68.2% | 29.1% | 99.9% |
| UNSW-NB15 | CNN, CCAT | 14.0% | 49.3% | 61.4% | 51.3% | 29.7% | 98.4% |
| UNSW-NB15 | CNN, AT-threat | 27.3% | 88.5% | 97.1% | 84.9% | 66.9% | 35.6% |
| UNSW-NB15 | CNN, TRADES | 18.5% | 52.0% | 91.9% | 81.8% | 66.3% | 59.5% |

## S8. Saturation of the attacks on the trees

| Dataset | Model | Strongest (v3 union) | Regular grids (3 resolutions) | Long adaptive search, new flows | Union of all |
|---|---|---|---|---|---|
| TON-IoT | RF | 24.7% | 125: 24.0%, 729: 24.2%, 4,913: 24.7% | 0.3% | 25.3% |
| TON-IoT | RF, augmented | 10.7% | 125: 9.2%, 729: 9.3%, 4,913: 10.5% | 0.2% | 11.2% |
| TON-IoT | RF, augmented w/o bytes | 12.9% | 125: 11.6%, 729: 12.2%, 4,913: 13.1% | 0.0% | 13.5% |
| TON-IoT | GBDT | 35.3% | 125: 24.5%, 729: 26.9%, 4,913: 27.4% | 3.5% | 39.1% |
| TON-IoT | GBDT, augmented | 3.5% | 125: 1.6%, 729: 1.9%, 4,913: 2.3% | 0.3% | 4.1% |
| TON-IoT | GBDT, augmented w/o bytes | 17.1% | 125: 13.3%, 729: 14.7%, 4,913: 15.5% | 0.2% | 17.3% |
| UNSW-NB15 | RF | 88.7% | 81: 85.6%, 625: 88.3%, 2,401: 89.2% | 0.3% | 90.1% |
| UNSW-NB15 | RF, augmented | 29.3% | 81: 0.4%, 625: 0.7%, 2,401: 0.9% | 0.3% | 29.7% |
| UNSW-NB15 | RF, augmented w/o bytes | 28.1% | 81: 0.1%, 625: 0.3%, 2,401: 0.4% | 1.1% | 29.3% |
| UNSW-NB15 | GBDT | 73.5% | 81: 56.5%, 625: 62.7%, 2,401: 67.5% | 4.8% | 78.8% |
| UNSW-NB15 | GBDT, augmented | 39.0% | 81: 0.9%, 625: 4.5%, 2,401: 3.1% | 10.2% | 49.3% |
| UNSW-NB15 | GBDT, augmented w/o bytes | 38.5% | 81: 0.7%, 625: 1.1%, 2,401: 1.9% | 5.5% | 44.1% |

## S9. Physical feasibility of the grid search

| Dataset | Model | Grid, all actions | Grid, integer packets and MTU bound | Evasions lost |
|---|---|---|---|---|
| TON-IoT | CNN | 47.9% | 46.8% | 1.3% |
| TON-IoT | CNN, clean loop | 23.3% | 23.1% | 0.4% |
| TON-IoT | CNN, AT-all | 17.5% | 17.7% | 0.1% |
| TON-IoT | CNN, CCAT | 17.1% | 17.2% | 0.2% |
| TON-IoT | CNN, AT-threat | 4.4% | 4.2% | 0.3% |
| TON-IoT | CNN, TRADES | 15.0% | 14.7% | 0.3% |
| TON-IoT | MLP | 61.5% | 61.7% | 0.3% |
| TON-IoT | MLP, AT-threat | 5.6% | 5.6% | 0.1% |
| TON-IoT | RF | 24.2% | 24.4% | 0.0% |
| TON-IoT | RF, augmented | 9.1% | 9.5% | 0.1% |
| TON-IoT | GBDT | 24.0% | 23.7% | 0.9% |
| TON-IoT | GBDT, augmented | 1.6% | 1.9% | 0.0% |
| UNSW-NB15 | CNN | 98.9% | 98.9% | 0.0% |
| UNSW-NB15 | CNN, clean loop | 79.5% | 79.6% | 0.0% |
| UNSW-NB15 | CNN, AT-all | 98.9% | 98.9% | 0.0% |
| UNSW-NB15 | CNN, CCAT | 98.2% | 98.2% | 0.0% |
| UNSW-NB15 | CNN, AT-threat | 30.3% | 30.4% | 0.0% |
| UNSW-NB15 | CNN, TRADES | 53.2% | 53.3% | 0.0% |
| UNSW-NB15 | MLP | 94.8% | 94.8% | 0.0% |
| UNSW-NB15 | MLP, AT-threat | 47.4% | 47.4% | 0.0% |
| UNSW-NB15 | RF | 85.6% | 85.5% | 0.3% |
| UNSW-NB15 | RF, augmented | 0.5% | 0.5% | 0.0% |
| UNSW-NB15 | GBDT | 55.7% | 55.8% | 0.8% |
| UNSW-NB15 | GBDT, augmented | 1.8% | 2.1% | 0.2% |

## S10. One further configuration per defence

| Dataset | Model | Clean macro-F1 | Benign FPR | Strongest ASR | ASR at matched FPR | Missed or evaded, matched FPR |
|---|---|---|---|---|---|---|
| TON-IoT | CNN, AT-all, ε = 0.02 | 0.841 | 0.094 | 23.8% | 83.0% | 83.6% |
| TON-IoT | CNN, AT-all, ε = 0.05 | 0.848 | 0.084 | 21.2% | 67.4% | 70.0% |
| TON-IoT | CNN, AT-all, ε = 0.1 | 0.863 | 0.054 | 18.7% | 41.1% | 45.2% |
| TON-IoT | CNN, AT-threat, β = 0.25 | 0.922 | 0.035 | 0.1% | 18.1% | 23.3% |
| TON-IoT | CNN, AT-threat, β = 0.5 | 0.902 | 0.052 | 5.6% | 49.0% | 55.7% |
| TON-IoT | CNN, AT-threat, β = 1 | 0.912 | 0.054 | 0.1% | 20.5% | 24.9% |
| UNSW-NB15 | CNN, AT-all, ε = 0.02 | 0.446 | 0.221 | 91.9% | 94.8% | 95.1% |
| UNSW-NB15 | CNN, AT-all, ε = 0.05 | 0.446 | 0.212 | 100.0% | 100.0% | 100.0% |
| UNSW-NB15 | CNN, AT-all, ε = 0.1 | 0.427 | 0.220 | 99.8% | 100.0% | 100.0% |
| UNSW-NB15 | CNN, AT-threat, β = 0.25 | 0.467 | 0.236 | 43.6% | 81.1% | 83.0% |
| UNSW-NB15 | CNN, AT-threat, β = 0.5 | 0.495 | 0.233 | 53.1% | 60.4% | 62.4% |
| UNSW-NB15 | CNN, AT-threat, β = 1 | 0.499 | 0.230 | 52.1% | 59.1% | 61.7% |

## S11. Operating point with all actions on UNSW-NB15

| Dataset | Model | Seeds | Benign FPR, own | ASR, own | Benign FPR, matched | Detected, matched | ASR, matched | Missed or evaded, matched | Δ missed or evaded (p) | Missed or evaded, 1% FPR |
|---|---|---|---|---|---|---|---|---|---|---|
| TON-IoT | CNN | 5 | 0.015 | 51.0% | 0.015 | 98.6% | 46.2% | 46.9% | – | 51.3% |
| TON-IoT | CNN, clean loop | 5 | 0.035 | 26.1% | 0.014 | 98.4% | 60.3% | 60.9% | +14.0 pp (0.292) | 62.8% |
| TON-IoT | CNN, AT-all | 5 | 0.084 | 20.8% | 0.013 | 92.5% | 62.3% | 65.2% | +18.3 pp (0.107) | 67.3% |
| TON-IoT | CNN, CCAT | 5 | 0.082 | 18.9% | 0.014 | 89.8% | 70.7% | 73.0% | +26.1 pp (0.091) | 75.6% |
| TON-IoT | CNN, AT-threat | 5 | 0.044 | 5.5% | 0.013 | 94.1% | 35.0% | 38.3% | −8.6 pp (0.408) | 40.0% |
| TON-IoT | CNN, TRADES | 5 | 0.050 | 16.7% | 0.014 | 60.9% | 87.9% | 92.7% | +45.8 pp (< 0.001) | 93.4% |
| TON-IoT | MLP | 5 | 0.015 | 64.7% | 0.015 | 92.9% | 62.5% | 65.2% | – | 68.1% |
| TON-IoT | MLP, AT-threat | 3 | 0.063 | 6.9% | 0.014 | 97.2% | 33.5% | 35.3% | −31.5 pp (0.020) | 40.6% |
| TON-IoT | RF | 3 | 0.010 | 24.2% | 0.009 | 99.9% | 11.8% | 11.9% | – | 10.5% |
| TON-IoT | RF, augmented | 3 | 0.011 | 9.1% | 0.009 | 99.9% | 8.5% | 8.5% | −3.3 pp (0.011) | 6.2% |
| TON-IoT | GBDT | 3 | 0.008 | 24.0% | 0.008 | 99.8% | 16.7% | 16.9% | – | 9.8% |
| TON-IoT | GBDT, augmented | 3 | 0.008 | 1.6% | 0.008 | 99.9% | 1.5% | 1.6% | −15.3 pp (0.019) | 0.6% |
| UNSW-NB15 | CNN | 5 | 0.198 | 99.2% | 0.198 | 74.0% | 99.1% | 99.3% | – | 99.9% |
| UNSW-NB15 | CNN, clean loop | 5 | 0.209 | 81.5% | 0.199 | 92.7% | 83.6% | 84.9% | −14.5 pp (< 0.001) | 98.7% |
| UNSW-NB15 | CNN, AT-all | 5 | 0.217 | 99.0% | 0.196 | 83.8% | 99.4% | 99.4% | +0.1 pp (0.448) | 100.0% |
| UNSW-NB15 | CNN, CCAT | 5 | 0.212 | 98.4% | 0.198 | 77.6% | 99.5% | 99.6% | +0.3 pp (0.639) | 100.0% |
| UNSW-NB15 | CNN, AT-threat | 5 | 0.241 | 34.9% | 0.199 | 95.4% | 43.6% | 45.9% | −53.5 pp (0.002) | 98.3% |
| UNSW-NB15 | CNN, TRADES | 5 | 0.241 | 57.1% | 0.198 | 94.6% | 52.4% | 54.7% | −44.7 pp (0.003) | 100.0% |
| UNSW-NB15 | MLP | 5 | 0.191 | 95.3% | 0.191 | 65.1% | 95.2% | 96.9% | – | 99.5% |
| UNSW-NB15 | MLP, AT-threat | 3 | 0.218 | 48.3% | 0.195 | 85.1% | 61.8% | 68.3% | −28.4 pp (0.035) | 99.9% |
| UNSW-NB15 | RF | 3 | 0.177 | 85.6% | 0.178 | 99.6% | 35.6% | 35.9% | – | 100.0% |
| UNSW-NB15 | RF, augmented | 3 | 0.181 | 0.5% | 0.178 | 100.0% | 0.0% | 0.0% | −35.8 pp (< 0.001) | 55.3% |
| UNSW-NB15 | GBDT | 3 | 0.130 | 55.7% | 0.130 | 99.4% | 52.0% | 52.2% | – | 89.1% |
| UNSW-NB15 | GBDT, augmented | 3 | 0.126 | 1.8% | 0.130 | 100.0% | 1.2% | 1.2% | −51.0 pp (0.001) | 10.0% |

## S12. BPDA action-space PGD against the networks

| Dataset | Model | Seeds | Strongest attack | BPDA action PGD | New evasions | Strongest with BPDA |
|---|---|---|---|---|---|---|
| TON-IoT | CNN | 5 | 51.4% | 43.1% | 0.0% | 51.5% |
| TON-IoT | CNN, clean loop | 5 | 26.2% | 22.3% | 0.0% | 26.2% |
| TON-IoT | CNN, AT-all | 5 | 20.9% | 16.2% | 0.1% | 21.0% |
| TON-IoT | CNN, CCAT | 5 | 19.1% | 14.8% | 0.1% | 19.2% |
| TON-IoT | CNN, AT-threat | 5 | 5.5% | 3.2% | 0.0% | 5.5% |
| TON-IoT | CNN, TRADES | 5 | 17.1% | 12.4% | 0.0% | 17.1% |
| TON-IoT | CNN, AT-all (BPDA) | 3 | 16.9% | 14.2% | 0.0% | 16.9% |
| TON-IoT | MLP | 5 | 64.8% | 57.9% | 0.0% | 64.8% |
| TON-IoT | MLP, AT-threat | 3 | 6.9% | 3.6% | 0.0% | 6.9% |
| UNSW-NB15 | CNN | 5 | 99.3% | 98.6% | 0.0% | 99.3% |
| UNSW-NB15 | CNN, clean loop | 5 | 81.9% | 76.6% | 0.0% | 81.9% |
| UNSW-NB15 | CNN, AT-all | 5 | 99.0% | 98.2% | 0.0% | 99.0% |
| UNSW-NB15 | CNN, CCAT | 5 | 98.4% | 97.7% | 0.0% | 98.4% |
| UNSW-NB15 | CNN, AT-threat | 5 | 35.6% | 24.4% | 0.0% | 35.6% |
| UNSW-NB15 | CNN, TRADES | 5 | 59.5% | 39.7% | 0.0% | 59.5% |
| UNSW-NB15 | CNN, AT-all (BPDA) | 3 | 99.9% | 99.9% | 0.0% | 99.9% |
| UNSW-NB15 | MLP | 5 | 95.4% | 90.7% | 0.0% | 95.4% |
| UNSW-NB15 | MLP, AT-threat | 3 | 48.5% | 42.4% | 0.0% | 48.6% |

## S13. Further defence variants

| Dataset | Model | Seeds | Clean macro-F1 | Δ clean vs undefended | Rare-class F1 | Benign FPR | Strongest ASR, all actions | Undefended, same seeds | Δ vs undefended | Δ vs clean loop | Strongest ASR, volume only |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TON-IoT | CNN, AT-all (BPDA) | 3 | 0.585 ± 0.139 | −0.362 | 0.161 | 0.280 | 16.9% ± 23.3 | 49.6% | −32.6 pp | −8.2 pp | = all actions |
| TON-IoT | RF, augmented ×2 | 3 | 0.954 ± 0.001 | −0.001 | 0.650 | 0.011 | 8.6% ± 0.1 | 25.0% | −16.5 pp | – | = all actions |
| TON-IoT | GBDT, augmented ×2 | 3 | 0.957 ± 0.004 | −0.003 | 0.679 | 0.009 | 5.5% ± 8.3 | 35.5% | −30.1 pp | – | = all actions |
| TON-IoT | RF, augmented w/o delay | 3 | 0.954 ± 0.000 | −0.001 | 0.650 | 0.011 | 19.5% ± 1.5 | 25.0% | −5.5 pp | – | = all actions |
| TON-IoT | GBDT, augmented w/o delay | 3 | 0.958 ± 0.006 | −0.001 | 0.702 | 0.011 | 9.9% ± 2.8 | 35.5% | −25.7 pp | – | = all actions |
| TON-IoT | RF, augmented w/o bytes | 3 | 0.954 ± 0.000 | −0.001 | 0.649 | 0.011 | 13.0% ± 0.5 | 25.0% | −12.1 pp | – | = all actions |
| TON-IoT | GBDT, augmented w/o bytes | 3 | 0.961 ± 0.001 | +0.001 | 0.712 | 0.007 | 16.0% ± 2.9 | 35.5% | −19.5 pp | – | = all actions |
| UNSW-NB15 | CNN, AT-all (BPDA) | 3 | 0.440 ± 0.006 | −0.114 | 0.197 | 0.218 | 99.9% ± 0.1 | 99.0% | +0.9 pp | +18.1 pp | 46.4% ± 11.4 |
| UNSW-NB15 | RF, augmented ×2 | 3 | 0.602 ± 0.002 | 0.000 | 0.391 | 0.183 | 24.9% ± 3.5 | 89.2% | −64.4 pp | – | 0.3% ± 0.2 |
| UNSW-NB15 | GBDT, augmented ×2 | 3 | 0.608 ± 0.005 | +0.004 | 0.396 | 0.146 | 37.7% ± 2.4 | 73.4% | −35.7 pp | – | 2.5% ± 2.3 |
| UNSW-NB15 | RF, augmented w/o delay | 3 | 0.601 ± 0.003 | −0.002 | 0.388 | 0.181 | 31.0% ± 0.8 | 89.2% | −58.2 pp | – | 10.4% ± 0.2 |
| UNSW-NB15 | GBDT, augmented w/o delay | 3 | 0.611 ± 0.001 | +0.007 | 0.399 | 0.126 | 40.8% ± 1.2 | 73.4% | −32.6 pp | – | 5.2% ± 2.1 |
| UNSW-NB15 | RF, augmented w/o bytes | 3 | 0.601 ± 0.005 | −0.002 | 0.386 | 0.181 | 30.9% ± 0.3 | 89.2% | −58.4 pp | – | 0.9% ± 0.1 |
| UNSW-NB15 | GBDT, augmented w/o bytes | 3 | 0.606 ± 0.002 | +0.001 | 0.384 | 0.126 | 39.3% ± 1.3 | 73.4% | −34.0 pp | – | 1.2% ± 0.6 |

## S14. Networks against trees, grid search

| Dataset | Attack | CNN − RF | CNN − GBDT | MLP − RF | MLP − GBDT |
|---|---|---|---|---|---|
| TON-IoT | Grid, β = 0.5 | +23.4 pp (p 0.003) | +22.9 pp (p 0.003) | +37.0 pp (p 0.003) | +36.4 pp (p 0.001) |
| TON-IoT | Grid, β = 2 | +36.0 pp (p < 0.001) | +13.0 pp (p 0.014) | +43.8 pp (p < 0.001) | +20.8 pp (p 0.004) |
| UNSW-NB15 | Grid, volume only, β = 0.5 | +46.6 pp (p 0.003) | +39.9 pp (p 0.003) | +46.3 pp (p 0.003) | +39.6 pp (p 0.003) |
| UNSW-NB15 | Grid, both, β = 0.5 | +12.6 pp (p 0.025) | +42.0 pp (p < 0.001) | +8.5 pp (p 0.059) | +37.9 pp (p < 0.001) |
| UNSW-NB15 | Grid, volume only, β = 2 | +62.0 pp (p < 0.001) | +53.2 pp (p < 0.001) | +58.8 pp (p < 0.001) | +50.1 pp (p < 0.001) |
