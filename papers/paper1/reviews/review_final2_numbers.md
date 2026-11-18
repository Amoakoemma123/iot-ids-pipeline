# Paper 1: independent numbers audit (final round 2), 2026-11-14

**Scope.** I checked every number in `Manuscript_filled.md` (abstract, Sections 3 to 7, Tables 1 to 5, A.1 and A.2) and the five highlights (`build.py` HIGHLIGHTS, filled from `values.json`).

**Method.** I recomputed each value from the raw result files with my own scripts. `compute_values.py` was not run. The scripts are in the session scratchpad (`load.py`, `a1.py` to `a13.py`) and use numpy and scipy (`ttest_ind`, `equal_var=False`). Data sources:

- Forests, LR, networks, int8, port ablation and min–max control: `results_v2/<ds>/<cfg>/seed*.json`.
- Rolled-back GBDTs, capped GBDTs, small trees, the `_ev` runs, GBDT port ablation, validation scores and `val_seed0.json`: `results_p1final`.
- Latency: `results_p1final/latency.json`.

**Conventions.** KB means bytes / 1024 (mean over seeds). SD uses ddof = 1. Welch p is taken over 5 seeds. Validation macro-F1 is seed 0 unless stated.

**Consistency checks.**
- The v2 RF and LR test scores are identical to the p1final refits.
- The v2 GBDT scores equal the p1final `raw_macro_f1` values.
- The seed-0 network F1 values in `latency.json` equal the v2 seed files.

## Summary

- **About 175 distinct checks**, plus a cell-by-cell comparison of all 147 rows of Table A.1 (7 numeric columns each, 1,029 cells) and all 18 rows × 3 datasets of Table 2.
- **Outright numerical errors: 0.** Every table cell and every stated number matches my recomputation to the precision printed.
- **Inconsistent or misleading statements: 10**, all moderate or minor. None is critical. They are listed by severity at the end.

## Checks (all PASS unless marked)

### Abstract and highlights
| # | Claim | Recomputed | Result |
|---|---|---|---|
| 1 | ≤ ~100 KB TON: int8 CNN and size-budgeted tree tied, both 0.951 | int8 CNN w2 0.95148 (106.0 KB) vs GBDT 15 leaves lr 0.1 0.95055 (109.8 KB); Welch p 0.666 | PASS |
| 2 | ≤ ~100 KB UNSW: tree better by 0.012 | GBDT 15 leaves ≤ 50 iterations 0.57446 − int8 CNN w2 0.56275 = 0.0117; p 0.014 | PASS |
| 3 | From 201 KB (TON) and 97 KB (UNSW) a tree is significantly better than every network | smallest significant: gbc50_l31 201.2 KB (p 0.0004); gbc50_l15 97.3 KB (p 0.014) | PASS |
| 4 | Best tree beats best network by 0.010 / 0.046 | 0.00974 (p 5e-7); 0.04574 (p 3.5e-5) | PASS |
| 5 | An unbounded forest is never more accurate than smaller trees | RF 200 trees, unlimited depth vs best smaller tree: TON 0.9603 < 0.9619; UNSW 0.6056 < 0.6085; N-BaIoT 0.99970 < 0.99987 | PASS |
| 6 | Int8 cost: CNNs ≤ 0.010, MLPs up to 0.021 (seed means) | max CNN mean drop 0.0104 (UNSW w0.5); max MLP 0.0207 (UNSW w1) | PASS |
| 7 | TON tied-best tree models 9.8–82.1 ms; best network 0.42 ms | tied set {gb_l63 9.76, gb05_l63 19.37, gb05_l7 23.80, gb05_l31 35.82, gb05_l15 82.07}; CNN w2 0.4226 | PASS |
| 8 | Compiled tree library 59–108× cheaper per tree | 59.5 (TON), 61.8 (UNSW), 107.7 (N-BaIoT) | PASS |
| 9 | Rollback removed every collapse | after rollback, 0 of 40 combinations are > 0.05 below the lr 0.05 mean; minimum seeds 0.942 / 0.951 / 0.574 / 0.597 | PASS |
| 10 | Port removal costs about 0.015 | −0.013 to −0.015 | PASS |
| 11 | Highlight 2 (201 / 97 KB), Highlight 3 (0.010 / 0.021), Highlight 4 (9.8–82.1, 0.42) | same values as above | PASS |
| 12 | Highlight 5 "keep 20 extra … iterations" | 20 in 191 of 197 non-capped runs; 1 or 17 in 6 N-BaIoT runs | minor, see D7 |
| 13 | Abstract length | 249 words (limit 250) | PASS |

### Section 3 / Table 1
| # | Claim | Recomputed | Result |
|---|---|---|---|
| 14 | Test sizes 27,533 / 15,894 / 22,453 | `len(yte.npy)` | PASS |
| 15 | 25 / 10 / 115 selected features; feature list in Appendix A | `features.json` | PASS |
| 16 | 10 / 10 / 9 classes | `unique(yte)` | PASS |
| 17 | MITM 156 test records, worms 19 | yte class counts | PASS |
| 18 | 0.1% (TON) and 16.3% (UNSW) of test records matched on selected features | 1 − clean_n/n = 0.127% and 16.27% | PASS |
| 19 | Oversampled sizes 293,550 and 363,220 | only checked to be divisible by 10 (equal class sizes); not derivable from the result files | not verifiable |

### Section 4
| # | Claim | Recomputed | Result |
|---|---|---|---|
| 20 | CNN 23,946 parameters, MLP 17,162 (TON) | `params` field | PASS |
| 21 | TON GBDT 7 leaves, seed 0: log-loss 0.042 at iteration 248, 361,323 at 268, test macro-F1 0.496 | 0.04243, 248; 361,322.8, 268; 0.4960 | PASS |
| 22 | TFLite float32 files are −2.9 to 7.6 KB larger than 4 B/param | −2.91 to +7.55 KB | PASS (wording, D9) |
| 23 | Pickled/compact ratio 9.6–9.9 for RFs | 9.60–9.92 (main-grid RFs) | PASS (scope, D6) |
| 24 | Pickled/compact ratio 8–16 for GBDTs | 8.15–15.68 | PASS |
| 25 | Unbounded UNSW forest has trees with > 65,535 nodes | mean 67,578 nodes per tree | PASS |
| 26 | 49 points per dataset (37 tree/LR + 12 networks) | 8 + 1 + 8 + 12 + 8 = 37 | PASS |

### Section 5.1
| # | Claim | Recomputed | Result |
|---|---|---|---|
| 27 | Best mean: GBDT 63 leaves lr 0.1, 0.962 (TON); GBDT 31 leaves lr 0.05, 0.608 (UNSW) | 0.96194; 0.60848 | PASS |
| 28 | "5 and 9 tree configurations … tied with it" | 4 and 8 besides the best (5 and 9 including it) | wording, D2 |
| 29 | Best network CNN w2 0.952, p < 0.001 (TON) | 0.95220, p 5e-7 | PASS |
| 30 | Best network int8 CNN w2 0.563, p < 0.001 (UNSW) | 0.56275, p 3.5e-5 | PASS |
| 31 | UNSW int8 +0.002 vs its float32 version | +0.00185 | PASS |
| 32 | N-BaIoT: 22 of 29 main-grid configurations ≥ 0.998 | 22 / 29 | PASS |
| 33 | LR 0.874 / 0.405 | 0.87381 / 0.40515 | PASS |
| 34 | UNSW benign FPR 11–23% for every model except LR | non-LR 10.96–22.57%; LR 23.4% | wording, D5 |
| 35 | Unbounded-RF lowest per-class F1: backdoor 0.18, analysis 0.22, dos 0.36 | 0.183, 0.216, 0.364 | PASS |
| 36 | CNN w1 0.947 on TON | 0.94713 | PASS |
| 37 | Before rollback: 6/20 (TON) and 2/20 (UNSW) more than 0.05 below the lr 0.05 mean; minimum 0.334 | 6, 2; 0.33405 | PASS |
| 38 | lr 0.05, 7 leaves, TON as run: one seed 0.798 | 0.798 | PASS |
| 39 | 17 of 120 runs with validation loss at the stop > 10× its best | 17 / 120 | PASS |
| 40 | After rollback: lowest seed 0.942 / 0.951 (TON), 0.574 / 0.597 (UNSW); 0/20 and 0/20 below the threshold | 0.9417 / 0.9505 / 0.5737 / 0.5970; 0 and 0 | PASS |
| 41 | Validation-split early stopping: −0.001 / +0.003, p 0.118 / 0.062; lowest seed 0.942 / 0.565 | −0.00112 / +0.00282; 0.1178 / 0.0620; 0.9416 / 0.5655 | PASS |
| 42 | 1,000-iteration cap reached in 14/20 (UNSW) and 4/20 (TON) lr 0.05 runs | 14, 4 | PASS |
| 43 | Clean records: lead of best tree over best network 0.010 (TON) | 0.00974 | PASS |
| 44 | Clean records: lead 0.086 (UNSW) | 0.0855 using gb05_l31 (best on the full test set); 0.107 using the best clean-record tree | ambiguity, D4 |
| 45 | Best clean-record tree: GBDT 63 leaves 0.962 (TON); RF 200 trees, unlimited depth 0.667 (UNSW) | 0.9619; 0.6666 | PASS |
| 46 | Table 2: 18 rows × 3 datasets (mean ± SD, KB) | all 108 cells match | PASS |

### Section 5.2
| # | Claim | Recomputed | Result |
|---|---|---|---|
| 47 | ≤ 110 KB, TON: int8 CNN w2 106 KB 0.951, +0.001 over GBDT 15 leaves lr 0.1 (110 KB, 0.951), p 0.666 | +0.00093, p 0.6658 | PASS |
| 48 | ≤ 110 KB, UNSW: gbc50_l15 97 KB 0.574, +0.012, p 0.014 | PASS | PASS |
| 49 | Smallest tree above the best network: gbc50_l31 201 KB 0.955 p < 0.001 (TON); gbc100_l7 86 KB 0.567 p 0.282 (UNSW) | 201.2, 0.9552, 0.0004; 85.9, 0.5666, 0.282 | PASS |
| 50 | Quantised encoding: "GBDTs shrink by a factor of 0.4–1.7" | compact/quantised = 0.40–1.75; on N-BaIoT 12 of 20 GBDT configurations grow | misleading, D3 |
| 51 | 16-bit leaf rounding changes macro-F1 by at most 0.000 | max 0.00031 (UNSW) | PASS (rounding) |
| 52 | Best GBDT ≤ 110 KB quantised: GBDT 15 leaves lr 0.1, 75 KB, 0.951 (TON); gbc200_l7 99 KB 0.580 (UNSW) | 74.9 KB; 99.5 KB, 0.5801 | PASS |
| 53 | Validation selection, best: gb_l63 891 KB, validation 0.961, test 0.962 (TON); gb05_l31 4,112 KB, 0.620 / 0.608 (UNSW) | seed-0 validation 0.9612 and 0.6198 | PASS |
| 54 | Smallest within 0.01: gbc200_l7 96 KB 0.953 / 0.946 (TON); gb05_l15 1,580 KB 0.611 / 0.606 (UNSW); DT depth 12 1.9 KB 0.996 / 0.996 (N-BaIoT) | PASS on seed-0 validation | PASS (but see D1) |
| 55 | Threshold margins 0.002 / 0.001; seeds within 0.01: 1 of 5 / 5 of 5 | 0.0018 / 0.0011; 1 / 5 | PASS |
| 56 | RF 200 trees, unlimited depth 0.960 at 22,486 KB vs 0.962 at 891 KB | PASS | PASS |

### Section 5.3
| # | Claim | Recomputed | Result |
|---|---|---|---|
| 57 | Size reduction 1.4–3.8× | 1.44–3.81 | PASS |
| 58 | CNN mean drop ≤ 0.002 (TON) / 0.010 (UNSW) / 0.001 (N-BaIoT); single seed ≤ 0.020 | 0.0023 / 0.0104 / 0.0007; 0.02049 | PASS (rounding) |
| 59 | MLP mean drop up to 0.021 (UNSW), 0.016 (N-BaIoT); single seeds 0.051 / 0.073 | 0.0207 / 0.0163; 0.0510 / 0.0733 | PASS |
| 60 | Agreement ≥ 99.4% (TON), 89.7% (UNSW); single seeds 98.9% / 85.3% | 99.36 / 89.68; 98.86 / 85.31 | PASS |
| 61 | Full-integer vs int8: seed-0 change ≤ 0.000 | exactly 0 | PASS |

### Section 5.4, Table 3, Table 4
| # | Claim | Recomputed | Result |
|---|---|---|---|
| 62 | Table 3: 29 rows × 6 cells | all 174 cells match `latency.json` | PASS |
| 63 | Table 4: 18 rows × 4 latencies, 3 sizes, 3 F1 values | all match (sizes in KB/1024) | PASS |
| 64 | Network latency: TF float32 0.35–0.47; TFLite float32 0.004–0.076; int8 0.05–10.00; CNN w1/w2 int8 0.4–10.0; full-integer 0.06–10.07 ms | 0.353–0.472; 0.0038–0.0757; 0.0486–9.998; 0.420–9.998; 0.0604–10.075 | PASS |
| 65 | XNNPACK changes int8 times by ≤ 4% | 4.08% | PASS (rounding) |
| 66 | Full-integer faster in 1 of 18; loses ≤ 0.020 vs float32 | 1 / 18; 0.02049 | PASS |
| 67 | Tree latency: RF 25 trees 1.2–1.4 ms; RF 100 trees 4.1–4.3 ms; GBDT main 4.1–83.8; capped 3.4–19.7; 7–10 µs/tree | 1.222–1.352; 4.103–4.257; 4.063–83.772; 3.358–19.668; 7.44–10.12 | PASS |
| 68 | Best TON GBDT: 1,210 trees (580–1,210); 9.8 ms (4.7–10.2); 23× best network | PASS; 9.759 / 0.4226 = 23.1 | PASS |
| 69 | UNSW: 84 ms vs 1.59 ms | 83.77 vs 1.589 | PASS |
| 70 | Tied models: 9.8–82.1 ms (TON), 6.7–83.8 ms (UNSW) | 9.76–82.07; 6.68–83.77 | PASS |
| 71 | Fastest within 0.005 of best: gbc100_l31 0.958, 4.02 ms (TON); gbc200_l31 0.608, 6.68 ms (UNSW) | PASS | PASS |
| 72 | 100 KB budget, TON: best network 2.85 ms, best tree 5.13 ms | 2.849 / 5.127 | PASS |
| 73 | Batch of 1,000: GBDT 46–782 ms, 65–155× cheaper per record; float32 networks 0.8–88.2; int8 26.2–9630.3 | 46.35–782.39, 64.8–154.8; 0.78–88.16; 26.22–9630.31 | PASS |
| 74 | LightGBM 0.07–0.14 µs/tree; best TON GBDT would take about 0.16 ms | 0.0695–0.1356; 1,210 × 0.1356 µs = 0.164 ms | PASS |
| 75 | "Most accurate tree models are slower than every network" | TON 9.76 > 2.85 (slowest network); UNSW 83.8 > 1.59; N-BaIoT 10.51 > 10.00 | PASS |
| 76 | "int8 therefore saves memory, not time" | true against float32 TFLite; false against float32 TensorFlow for 7 of 18 configurations (Table 3) | nuance, D10 |

### Sections 5.5 and 5.6, Table 5, Sections 6 and 7
| # | Claim | Recomputed | Result |
|---|---|---|---|
| 77 | Min–max CNN 0.900 ± 0.005; quantile transform +0.047, p < 0.001 (TON); +0.020, p 0.021 (UNSW) | 0.89995 ± 0.0051; +0.0471, p 2.8e-5; +0.0202, p 0.0206 | PASS |
| 78 | rf_v1 pickled 97,704 KB (95 MB), compact 10,171 KB | 97,703.9 KB = 95.4 MiB; 10,170.9 | PASS |
| 79 | 350 KB ≈ 3.7 × 94 KB | 3.74 | PASS |
| 80 | 0.962 > 0.957 (p < 0.001), 11× smaller | p 9.4e-5; 11.4× | PASS |
| 81 | RF 25 trees depth 16 within 0.004 at 1,028 KB | 0.0036; 1,028.0 | PASS |
| 82 | Table 5: 6 rows, with / without / difference / p | −0.0149, −0.0147, −0.0154, −0.0133, −0.0151, −0.0139; all p < 0.001; all ablated GBDTs rolled back | PASS |
| 83 | Ransomware loss 0.06 (rf_v1) and 0.05 (GBDT lr 0.05) | 0.0552 and 0.0546; ransomware is the largest loss for all six models | PASS (rounding) |
| 84 | 6.1: same budget claims, 23×, unbounded-forest claim | same as checks 1–5 and 68 | PASS |
| 85 | 7: every number repeats the abstract or 5.4 (0.951, 0.012, 201 / 97, 0.010 / 0.046, 0.010 / 0.021, 9.8 vs 0.42, 0.958 / 4.02 ms, 0.015) | consistent with abstract, highlights and body | PASS |
| 86 | Table A.1: 147 rows × 7 numeric columns | all 1,029 cells match (validation column = seed mean for trees, seed 0 for networks, as the caption says) | PASS (see D1) |
| 87 | Table A.2: 12 rows, all as-run / rolled-back means and minimum seeds, iteration ranges | all match | PASS |

**Internal consistency.** Abstract, highlights, Sections 5 and 6, the conclusion and the tables agree on every repeated quantity:

- 0.951 / 0.012
- 201 / 97 KB
- 0.010 / 0.046
- 0.010 / 0.021
- 9.8–82.1 ms / 0.42 ms
- 59–108×
- 0.015
- 23×
- 0.958 / 4.02 ms

The one exception is the validation scores (D1).

## Discrepancies, ranked by severity

No critical errors were found. Each item gives the location, the stated value, the recomputed value, how it was computed, and a suggested fix.

### Moderate

**D1. Two different "validation macro-F1" values for the same configuration (5.2 vs Table A.1).**
- *Stated.* Section 5.2: "GBDT with 7 leaves and at most 200 iterations (96 KB; validation 0.953, test 0.946)", "GBDT with 31 leaves at lr 0.05 (4,112 KB; validation 0.620 …)" and "GBDT with 15 leaves at lr 0.05 (1,580 KB; validation 0.611 …)".
- *Table A.1.* The same configurations show 0.947, 0.614 and 0.609. Other mismatches include TON gb_l7 (0.954 seed 0 vs 0.947 mean) and rf_t10_d4 (0.834 vs 0.791).
- *Computed.* Seed-0 `val_macro_f1`, as used by the selection rule in 4.4, vs the mean over seeds. The Table A.1 caption says "mean over seeds; * seed 0 only", so each number is right for its own definition.
- *Why it matters.* A reader who applies the 4.4 rule to Table A.1 gets a different answer on TON. The selected small configuration (mean validation 0.947) falls below the 0.951 threshold, and the smallest configuration qualifying on mean validation would be gbc50_l31 (201 KB).
- *Fix.* Either show seed-0 validation in Table A.1 (or add it as a column), or state in 5.2 that the quoted validation scores are seed-0 values while Table A.1 shows seed means.

**D3. "In the quantised encoding … the GBDTs shrink by a factor of 0.4–1.7" (5.2).**
- *Recomputed.* compact / quantised bytes range from 0.40 to 1.75. A factor below 1 means the model grows.
- *Detail.* On N-BaIoT the quantised encoding is larger than the compact one for 12 of 20 GBDT configurations, up to 2.49× (gbc50_l7), because the bin edges of 115 features are stored. TON and UNSW shrink by 1.17–1.75×.
- *Fix.* Write something like "shrink by 1.2–1.7× on TON-IoT and UNSW-NB15; on N-BaIoT the stored bin edges make small GBDTs up to 2.5× larger".

### Minor

**D2. "5 and 9 tree configurations, respectively, are statistically tied with it" (5.1).**
- *Recomputed.* 4 and 8 configurations other than the best have Welch p ≥ 0.05. The stated counts include the best itself (`compute_values.py` line 365, `p is bt or …`).
- *Fix.* "5 and 9 configurations, including the best, are statistically tied", or quote 4 and 8.

**D4. Clean-record lead "0.086 on UNSW-NB15" (5.1).**
- *Recomputed.* 0.0855 is the clean-record score of gb05_l31 (the best tree on the full test set, 0.645) minus CNN w2 float32 (0.5595). The next sentence names RF 200 trees, unlimited depth (0.667) as the most accurate clean-record tree, which leads by 0.107.
- *Assessment.* Conservative, not wrong.
- *Fix.* Say "the tree that is best on the full test set leads by …".

**D5. "every model except LR classifies 11–23% of the benign test records as attacks" (5.1).**
- *Recomputed.* Non-LR models: 10.96–22.57%. LR: 23.4%, which also rounds to 23%.
- *Fix.* Drop "except LR" ("every model, LR included, …"), or use one decimal (11.0–22.6% vs 23.4%).

**D6. "9.6–9.9 times the compact size for the random forests" (4.2).**
- *Recomputed.* True for the 8 main-grid forests (9.60–9.92). The shallow forests of the small-tree group (10 and 25 trees, depth 4 and 6) reach 9.84–10.88.
- *Fix.* "9.6–10.9", or "for the forests of the main grid". The "about ten times" in 1 and 6.2 stays correct.

**D7. "keeps the iterations after its best validation score" / Highlight 5 "keep 20 extra … iterations".**
- *Recomputed.* n_iter_raw − best_iter = 20 in 191 of 197 runs that did not hit the cap. Six N-BaIoT runs kept 1 or 17 iterations after the best one; capped runs keep 0–20.
- *Assessment.* Essentially right.
- *Fix.* "up to 20" would be exact.

**D8. Latency protocol (4.2).**
- *Stated.* "median over five repeats of 200 calls after 20 warm-up calls".
- *Data.* `latency.json` records "GBDT seeds 1-4: 3 x 100 after 10". The seed ranges quoted in 5.4 (4.7–10.2 ms, 580–1,210 trees) come from this shorter protocol.
- *Fix.* Add one clause to 4.2.

**D9. "are -2.9–7.6 KB larger than four bytes per parameter" (4.2).**
- *Recomputed.* −2.91 to +7.55 KB is correct. The MLP w2 files are smaller than 4 B/param, so "−2.9 KB larger" reads oddly.
- *Fix.* "differ from four bytes per parameter by −2.9 to +7.6 KB".

**D10. "On this CPU and runtime, int8 therefore saves memory, not time" (5.4).**
- *Data.* True against float32 TFLite (int8 is slower in all 18 cases). Against float32 TensorFlow, the runtime of Table 3 and of the "best network 0.42 ms" claims, int8 is faster in 7 of 18 cases. Examples: MLP w1 TON 0.174 vs 0.356 ms; MLP w0.5 0.059 vs 0.360 ms; CNN w0.5 TON 0.216 vs 0.400 ms.
- *Fix.* "int8 saves memory, not time, relative to float32 TensorFlow Lite".

### Rounding only (no action needed)

| Location | Stated | Exact |
|---|---|---|
| CNN max single-seed int8 drop; full-integer loss | 0.020 | 0.02049 |
| XNNPACK effect | 4% | 4.08% |
| 16-bit leaf rounding | 0.000 | 0.00031 |
| Ransomware loss, rf_v1 / GBDT lr 0.05 | 0.06 / 0.05 | 0.0552 / 0.0546 |
| Pickled rf_v1 | "95 MB" | 95.4 MiB |
| µs per tree | "about 7–10" | 7.44–10.12 |
| GBDT batch speed-up | 65–155× | 64.8–154.8× |

### Not verifiable from the result files

Table 1 training, validation and oversampled counts come from `Paper6/data_stats.json`, which is derived from the raw data and was not loaded. Only the test counts were verified, from `yte.npy`.
