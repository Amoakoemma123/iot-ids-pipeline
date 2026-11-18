# Paper 6: independent numbers audit (final round 2, 2026-11-14)

**Scope.** I recomputed the values from the raw seed JSON and prediction files. For the networks, RF, random splits, capacity controls, resplits, column orders and attention these come from `results_v4`. For every GBDT I used `results_p6gb`, as the paper does. Table 7 comes from `results_v2`/`results_v3`, the compact forests from `experiments/baseline/results_v2`, and costs from `latency.json`. I wrote my own scripts for this: `C:\Users\G15165~1\AppData\Local\Temp\claude\p6audit\audit.py`, `audit2.py` and `audit3.py`. `compute_values.py` was read only to learn definitions: "better single stream" is the stream with the higher seed mean; the within-seed semantic − random difference uses random splits r and r+5 for seed r; resplits are 6 observations (seed mean plus 5 splits); Holm runs over 8 tests. Method: paired t on per-seed differences (t(4) 95% CI, ties |d| ≤ 1e-6), Welch t for trees vs networks, Holm step-down, Spearman with 5,000-permutation p, and macro-F1/accuracy recomputed from `*_pred.npy` with `clean_split0.npy`/`yte_split0.npy`.

**Counts.** About 620 numeric values checked: all of Tables 2, 3, 4, 5, 6, 7, 8, B.1, B.2 and B.3, plus about 90 numbers in the abstract, highlights, Sections 3–6 and the conclusion.
- **Wrong numbers: 0.**
- **Inconsistent / misattributed: 1.** The GBDT cost in Table 8 and Section 5.6 is for a different model (see A1).
- **Minor wording notes: 4.**

The recomputed GBDT predictions reproduce the stored macro-F1 exactly (difference 0.000000 for every TON-IoT/UNSW-NB15 seed).

## A. Discrepancies

**A1 (real, methodological). Table 8 and Section 5.6: the GBDT size and latency are not for the GBDT whose accuracy is reported.**
- `bench_latency.py` (run 2026-11-03) builds the GBDT from `results_v4/<ds>/tune_trees.json`, with no rollback.
- On TON-IoT that configuration is learning rate 0.05 with 127 leaves. In results_v4 it ran to 134–162 iterations.
- The reported TON-IoT GBDT (`results_p6gb`, re-tuned with rollback) uses learning rate 0.1 with 127 leaves and is evaluated at 46–70 iterations (raw 66–90).
- So "23 MB", "19.10 ms/record" and "78.0 ms/1,000" on TON-IoT describe a model with roughly 2–3 times as many trees as the one scored at 0.962. The true footprint is probably about 8–11 MB.
- On UNSW-NB15 and N-BaIoT the hyperparameters are the same, but the timed model keeps the 20 post-best iterations. This is negligible on UNSW-NB15 (966–1,000 of at most 1,000). On N-BaIoT it adds about 10–18% (best 108–230 iterations).
- Fix: re-run `bench_latency.py` with the `results_p6gb` tuned parameters and rollback, or state the caveat.
- None of the qualitative claims change: the GBDT is still far larger than late fusion and slower per record.

**A2 (definitional, OK). Section 4.1: "1 of the 90 network runs of the main and capacity experiments used the full budget".**
- There are 105 such runs (3×5×5 main + 2×3×5 capacity).
- 90 excludes the 15 reused N-BaIoT runs (s1, early, late), which had a 60-epoch cap. The single capped run is confirmed: N-BaIoT s2, seed 4, best epoch 118.
- Suggest saying "of the 90 new network runs".

**A3 (wording). Section 5.1: "attack detection rate … at least 0.996 for every model on TON-IoT".**
- The minimum is RF stream 2 at 0.99653, which Table B.3 prints as 0.997. The claim is true but loose; "at least 0.997" would match the table.

**A4 (wording). Section 5.2: "No CNN beats the tuned RF on all features in macro-F1".**
- True on all three datasets, but on N-BaIoT only by 7×10⁻⁷ (stream-1 CNN 0.9996854 against RF 0.9996861). It is fine as worded together with "on N-BaIoT both are at the ceiling".

**A5 (wording). Abstract: "Tuned tree ensembles beat every untuned fusion network in macro-F1 except on N-BaIoT".**
- On N-BaIoT the RF and GBDT means are also above every fusion network, but not significantly (Welch p = 0.404 and 0.180; differences about 1e-4). "Except on N-BaIoT, where all models are at the ceiling" would be more exact.

**A6 (note). Section 4.2: "Their hyperparameters … are chosen on the validation macro-F1".**
- The GBDT grid was re-run with rollback, and on TON-IoT it changed the selected learning rate from 0.05 to 0.1. The text is consistent with this but does not say it.
- The paper's GBDT numbers moved little from results_v4:
  - TON-IoT: all features 0.9621 → 0.9622; stream 1 0.9258 → 0.9249.
  - UNSW-NB15: all features 0.6051 → 0.6085, driven entirely by seed 1 (0.5938 → 0.6105, rolled back from 986 to 966 iterations).

## B. Verified values (selection; all match after rounding to the printed digits)

| Location | Stated | Recomputed |
|---|---|---|
| Abstract/T3: early − better stream | +0.044 / +0.060 (5/5) | +0.0440 [+0.0433,+0.0446] / +0.0597 [+0.0420,+0.0774], 5/5 |
| Abstract: RF gain from all features | +0.032 / +0.061 | +0.03245 / +0.0606 |
| Abstract/Highlight 2/T3: late − early | +0.005 [+0.004,+0.007]; +0.016 [+0.001,+0.031] | +0.0052 [+0.0037,+0.0068]; +0.0161 [+0.0015,+0.0307] |
| B.1 unadjusted p (8 tests) | 0.001, 0.056, 0.001, 0.032, 0.038, 0.128, 0.027, 0.013 | 0.0007, 0.0557, 0.0013, 0.0323, 0.0379, 0.1280, 0.0271, 0.0126 |
| B.1 Holm p | 0.006, 0.135, 0.009, 0.135, 0.135, 0.135, 0.135, 0.076 | 0.0056, 0.1354, 0.0088, 0.1354, 0.1354, 0.1354, 0.1354, 0.0755 |
| B.1 margin readings | 4 negligible (TON-IoT), 4 inconclusive (UNSW-NB15) | same |
| Section 5.2 shares (TON-IoT) | 41% / 71% / 73% | 40.6% / 71.4% / 73.1% |
| Section 5.2 median CI half-width | ≈0.010 / 0.001 | 0.0096 / 0.0014 |
| T3 RF − late (Welch) | +0.008, +0.034, 0.000 (p = 0.404) | +0.0078 (p = 1e-8), +0.0341 (p = 4e-7), +0.0001 (p = 0.404) |
| T3 GBDT − late (Welch) | +0.009, +0.036, 0.000 (p = 0.180) | +0.0094 (p = 3e-6), +0.0358 (p = 1e-7), +0.0001 (p = 0.180) |
| T2 GBDT all/s1/s2 (TON-IoT) | 0.962±0.001 / 0.925±0.001 / 0.836±0.001; FPR 0.008/0.045/0.036 | 0.9622±0.0011 / 0.9249±0.0014 / 0.8357±0.0009; 0.0076/0.0454/0.0364 |
| T2 GBDT (UNSW-NB15) | 0.608±0.002 / 0.540±0.002 / 0.365±0.001; FPR 0.127/0.155/0.196 | 0.6085±0.0023 / 0.5402±0.0020 / 0.3649±0.0014; 0.1266/0.15455/0.1960 |
| T2 every other row (all 3 datasets) | – | all match |
| Section 4.2 GBDT iterations (UNSW-NB15) | 966–1,000 | best_iter 966–1,000 |
| Section 5.1 UNSW-NB15 FPR range | 0.127–0.205 | 0.1266–0.2053 |
| Section 5.1 benign→fuzzers share | 82% / 82% | 82.3% / 82.1% |
| Section 5.1/B.3 PR-AUC | 0.974/0.975/0.974 vs 0.978/0.981; 0.621/0.635/0.632 vs 0.651/0.657 | 0.9735/0.9751/0.9735 vs 0.9776/0.9813; 0.6209/0.6352/0.6318 vs 0.6506/0.6567 |
| Section 5.2 resplits | late−early 6/6 +0.005 [+0.004,+0.006]; 5/6 +0.011 [+0.001,+0.020]; RF−late +0.006 [+0.005,+0.008], +0.041 [+0.035,+0.046]; attention per split 0.000…+0.004, −0.010…+0.017; early−best stream +0.045, +0.050 | identical |
| Section 5.2 duplicate records | 2,586 (16.3%); 35; 81/80/71/10% | 2,586 (16.27%); 35; 80.6/80.5/70.6/9.9% |
| B.2 (all 14 rows × 4 columns) | e.g. RF UNSW-NB15 0.607/0.652/0.636/0.831 | 0.6067/0.6519/0.6361/0.8315 |
| Section 5.2 RF−late on matched / clean records | −0.007 accuracy / +0.036; F1 lead +0.034 → +0.073 | −0.0070 / +0.0358; +0.0726 |
| Section 5.2 late−early on clean records (UNSW-NB15) | +0.025 [+0.001,+0.048] | +0.0249 [+0.0014,+0.0484] |
| Section 5.2 earlier run | +0.007 (4/5), −0.003 | +0.0073 (4/5), −0.0030 |
| Section 4.2 reruns | 51 runs, 9 same epoch, mean 0.003 (max 0.018); TON-IoT 0.001/0.002 (21), UNSW-NB15 0.006/0.018 (25), N-BaIoT 0.000 (5) | 51, 9, 0.0033/0.0183; 21: 0.0009/0.0021; 25: 0.0060/0.0183; 5: 0.0001 |
| Section 4.2 reused runs | 104 of 120 random-split runs, all 30 late, 17 of 20 attention (TON-IoT+UNSW-NB15), 15 N-BaIoT main | 104/120; late 10+10+10; attention 7+10; 15 |
| Table 6 (all 32 rows, including the 6 GBDT rows) | e.g. TON-IoT GBDT s1 0.925/0.849/0.600–0.927/+0.076 [−0.006,+0.157] | 0.9249/0.8490/0.6005–0.9271/+0.0759 [−0.0057,+0.1575] |
| Section 5.3 TON-IoT late, semantic − random | +0.001 [0.000,+0.002], p = 0.027, above 8/10 | +0.00137 [+0.00025,+0.00250], p = 0.0274, 8 |
| Section 5.3 UNSW-NB15 late, semantic − random | +0.009 [−0.001,+0.019], above 9/10 | +0.0093 [−0.0006,+0.0192], 9 |
| Section 5.3 fusion gain (semantic vs random) | +0.049 vs +0.012; +0.076 vs +0.029 | +0.0492/+0.0123; +0.0758/+0.0287 |
| Section 5.3 random-split late vs early | +0.004, all ten | +0.0038, 10/10 |
| Section 5.3 column orders | 0.946–0.950; 0.527–0.565; SD 0.015 vs 0.011 | 0.9464–0.9499; 0.5266–0.5650; 0.0149/0.0113 |
| Section 5.4 attention weights | 0.59 (0.55–0.63), 0.46, 0.73 (0.66–0.81); dos 0.30, backdoor 0.88; within-class SD 0.14 | 0.590 (0.551–0.627), 0.458, 0.733 (0.662–0.812); 0.300/0.879; 0.135 |
| Section 5.4 ρ (weight vs stream advantage) | 0.16 (−0.21…+0.32), 0.44 (+0.38…+0.50); permutation p 0.56/0.22 | 0.164 (−0.212…0.321), 0.438 (0.382…0.503); 0.565/0.218 |
| Section 5.4 ablation drops | 0.459/0.248; 0.288/0.142; 0.678/0.079 | 0.4588/0.2485; 0.2881/0.1421; 0.6783/0.0785 |
| Section 5.4 ρ (reliance vs weight / advantage) | 0.94/0.93/0.93; 0.03 (p = 0.95), 0.54 (p = 0.11) | 0.939/0.927/0.933; 0.030 (0.948), 0.539 (0.119) |
| Section 5.4 between-seed ρ; DoS | 0.74 (0.47), 0.55 (−0.17), 0.89; DoS lowest in 4/5 seeds, 0.97 vs 0.94 | 0.739 (0.467), 0.547 (−0.167), 0.888; 4/5; 0.973/0.938 |
| Table 7 (36 cells), Section 5.5 | 0.028, FPR 0.753, best epoch ≤ 4, majority baseline 0.037 | 0.0277, 0.7527, [3,1,4,1,1], 0.0372 |
| Section 5.6 compact forests | 4.4/7.3 MB, 0.946/0.572, ×13; 0.955 at 41.1 MB, 0.603 at 125.7 MB | 4.43/7.32, 0.9462/0.5723, 12.9; 0.9547/41.05, 0.6030/125.73 |
| Section 5.6/abstract footprints | 221 and 636 MB, 4,416,482 nodes, 343 kB, ×14.5 | 221.1/636.1, 4,416,482, 343.1, 14.50 |
| Table 8 (all 72 cells) | – | match latency.json (but see A1 for the GBDT row) |
| Tables 4/5 (140 cells) and test n | e.g. late MITM 0.66, GBDT worms 0.61 | 0.6555, 0.608; all class counts match yte_split0 |
| Table 1 / Section 3.1 | sizes; RFECV 0.554 vs 0.526 | data_stats.json match; rfecv 0.5541 at 10 features, 0.5260 at 40 |
| Highlights | +0.005; 4.4 MB | match |

## C. Qualitative claims checked

- "in every seed" (early − better stream, Section 5.2): 5/5 on TON-IoT and UNSW-NB15. True.
- "not significant after Holm" on UNSW-NB15: all four family Holm p ≥ 0.076. True.
- Attention fusion is not significant in the family (Holm 0.135 on both datasets). True.
- MLP attention − early on TON-IoT: +0.0033 [+0.0011,+0.0055], 5/5. True.
- "No CNN beats the tuned RF": true on all datasets, but on N-BaIoT only by 7e-7 (A4).
- "RF exceeded late fusion on every split of both datasets": 6/6 and 6/6. True.
- "semantic protocol stream weaker than every random stream" (TON-IoT): 0/10 random below, for both the CNN and the RF. True.
- "UNSW-NB15 semantic streams below nearly all random streams": 9/10 for each stream. True.
- "semantic attention mean above all ten random-split values" (N-BaIoT): 10/10. True.
- The conclusion's "within 0.002 on TON-IoT" is consistent: point estimate +0.0014, CI upper bound +0.0025.
