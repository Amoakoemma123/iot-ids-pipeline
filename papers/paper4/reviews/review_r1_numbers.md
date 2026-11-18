# Paper 4 (Federated CNN-IDS) — Round-1 numbers audit

Audited text: `Revision_v2/Paper4/Paper4_v1_text.md` (extracted from `Finished_Papers/Paper4/IDS_IoT_Federated_Learning_Paper.docx`).
Ground truth: `Papers pipeline/experiments/paper4_federated/results/*.json` (TON-IoT, seed 42), `results/seed_43|seed_44/*.json`, `results_unsw/*.json`, the three `global_summary.json` files, `run_fl.log`, code `run_federated.py` + `run_extra_experiments.py`. Centralised baseline: `_v1_backup_2026-10-24/experiments/baseline/results/<ds>/metrics.json` (the `cnn` block). Dataset counts: `Papers pipeline/dataset2/UNSW_NB15.csv` (label column read only). No files modified, no models loaded.

How the code defines its fields (needed to read the paper's numbers):
- `best_f1` = max macro-F1 over the test-set checkpoints (every 2 rounds); `best_round` = round of that max; `final_metrics` = last round. **"Best F1" in every table is the best checkpoint chosen on the test set.** There is no validation set, because val is merged into the FL training pool.
- `convergence_round` = first checkpoint with F1 ≥ 0.90 × best_f1 (as §3.8 says).
- Communication = 95,784 B (model) × 2 × N × rounds; MB = 1e6 B.
- **Multi-seed runs change only the TF seed.** `iid_partition`/`dirichlet_partition` default to `seed=SEED`(42), `load_dataset(seed=SEED)`, `shuffle(seed=SEED)`. The client sizes in the JSONs are the same for all three seeds, so the data split and the client partition are the same in every seed.
- SCAFFOLD is called without an `lr` argument, so it runs with `lr=LR=1e-3`. The value 0.01 appears nowhere in the code.
- DP: the batch-mean gradient is clipped **per tensor** to C=1, then N(0,(σC/B)²) noise is added and Adam applies the update. This is not per-sample DP-SGD. ε comes from an RDP accountant that assumes Poisson-subsampled, per-sample-clipped Gaussian noise.

## Summary counts

| Category | Count |
|---|---|
| Numeric mismatches (N1–N17) | 17 |
| False / overstated / cherry-picked / ambiguous claims (C1–C27) | 27 |
| Unfinished text / internal-consistency / housekeeping (U1–U9) | 9 |
| Citation-content advisories (not results-checkable) (R1–R4) | 4 |
| **Total problems** | **57** |
| Verified numbers/items (itemised in §F) | ≈120 |

Severity: H = changes a headline or conclusion; M = a wrong number or claim a reviewer would catch; L = cosmetic or rounding.

## A. Numeric mismatches

| # | Sev | Location | Text says | Data says (file/field) | Fix |
|---|---|---|---|---|---|
| N1 | M | §5 para 1 | "best F1=0.8449 at round 4" | `fedavg_iid.json` best_round=16. Round 4 is convergence_round (F1 0.7643) | "best F1=0.8449 at round 16 (90%-of-peak reached at round 4)". The abstract's "in 16 rounds" is correct, so §5 contradicts the abstract |
| N2 | M | Table 2, UNSW centralised Accuracy | 0.7564 | v1 backup `UNSW_NB15/metrics.json` cnn.accuracy = 0.7712 (live re-run: 0.7351). 0.7564 is not found in any result file | 0.7712 |
| N3 | H | Table 5, row "DP (σ=1.0, B=1024)", ε column | 4.008 | `dp_fedavg_lb.json` dp.epsilon = 9.045 (larger B raises sampling rate q) | 9.045. Also say in text that B=1024 has a weaker guarantee |
| N4 | M | §8 "UNSW-NB15 ... best F1=0.0120" | 0.0120 | `results_unsw/dp_fedavg_eps13.json` best_f1 = 0.0138 (R2). 0.0120 is final F1 | 0.0138 (or say "final 0.0120"). UNSW ε=4.087 is not reported |
| N5 | L | §7, Table 4, §10.4, §11: FedAdam UNSW | 0.1102 | `results_unsw/fedadam_alpha0.5.json` best_f1 = 0.110251 → 0.1103 | 0.1103 |
| N6 | M | §6 para 1 | "a 33.2% relative decline from the IID baseline" | (0.844946−0.534811)/0.844946 = **36.7%**. 33.2% matches the stale 20-round α=0.1 run in `run_fl.log` (best 0.5648@R18) | 36.7% |
| N7 | L | §6 para 1 | "oscillates between 0.29 and 0.53 for the remaining rounds" | `fedavg_alpha0.1.json` history after R14: min 0.2711 (R30), max 0.5139 (R20) | "between 0.27 and 0.51" |
| N8 | M | §7 ("underperforms FedAvg by 0.1306 pp"); §11 fragment ("-0.1306 pp") | 0.1306 pp | 0.784577−0.653987 = 0.1306 F1 = **13.1 pp** | "by 13.1 pp" |
| N9 | M | §8 | "only a +0.0174 pp change vs. the standard B=256" | 0.064217−0.046846 = 0.0174 F1 = **1.7 pp** | "+1.7 pp" |
| N10 | L | §3.1 | "23,946 trainable parameters (TON-IoT)" | 95,784 B / 4 = 23,946 **total** weights, of which 192 are BN moving mean/var (non-trainable), so 23,754 trainable. With GAP the parameter count does not depend on n_features (UNSW model is also 95,784 B) | "23,946 parameters (23,754 trainable), identical for both datasets" |
| N11 | H | §3.7 ("learning rate η=0.01"); Table 4 SCAFFOLD note ("SGD lr=0.01") | 0.01 | `run_federated.py` L701-710 call `run_fl(... algorithm='scaffold', **kwargs)` with no lr, so lr=LR=1e-3. No 0.01 in either script | Say lr=0.001. This also makes Table 4's "SCAFFOLD optimiser baseline" (FedAvg-SGD lr=0.001) really lr-matched; state it correctly |
| N12 | M | §6 para 2 and §9 | "worms: 44 total, shellcode: 1,160" | `dataset2/UNSW_NB15.csv` attack_cat: worms = 130, shellcode = 1,133 (175,341 rows). 44 is the worms count of the official UNSW *test* CSV | "worms: 130, shellcode: 1,133 in total" (about 85% of each in the FL pool) |
| N13 | L | Table 7 Std column | ±0.020 (α=0.5), ±0.010 (FedProx) | Population std (ddof=0): 0.0194, 0.0108, so ±0.019 and ±0.011. IID 0.0180 ✓, α=0.1 0.0395 ✓. With n=3 the sample std (ddof=1) is the usual choice: 0.022 / 0.024 / 0.048 / 0.013 | Report sample std and say which one is used |
| N14 | L | Table 2, Tables 4/6 TON N=4 total, text "15.32/15.3 MB" | 15.32 MB | total_bytes 15,325,440 → 15.325 MB, which rounds to 15.33 | 15.33 MB (or 15.3) |
| N15 | L | Abstract "best F1=0.050" for DP | 0.050 | Max over σ∈{1,1.5,4} = 0.0497 ✓. But the B=1024 configuration in Table 5 reached 0.0642 | "best F1 ≤ 0.064 across all DP configurations" |
| N16 | L | Table 4 caption / FedAvg-SGD rows | "(α=0.5, N=4, 20 Rounds)" | The table has an α=0.1, 50-round row. FedAvg-SGD times (253 s / 733 s) come from Keras `model.fit` (graph mode), while all other rows use an eager custom loop, so the times are not comparable | Fix the caption. Drop or footnote the FedAvg-SGD times |
| N17 | L | Table 2 "Accuracy" column for FL rows | 0.9154 / 0.8267 | These are accuracy **at the best-F1 checkpoint** (TON R16; UNSW R18). Final-round accuracy is 0.8883 / 0.8290 | Label the column "accuracy @ best round" or report final |

## B. Qualitative claims: false, overstated, cherry-picked or ambiguous

| # | Sev | Location | Claim | Problem (evidence) | Fix |
|---|---|---|---|---|---|
| C1 | H | Abstract; contribution (6); §10.6; §11 | "Key results are validated across three random seeds" | Only 4 TON-IoT configs have 3 seeds (fedavg_iid, fedavg_α0.5, fedavg_α0.1, fedprox_α0.5). None of the UNSW, SCAFFOLD, FedAvg-SGD, FedAdam, DP, α=1.0 or scaling runs is multi-seed. The seeds change only TF init/dropout, not the data split or client partition (identical `client_sizes` across seeds) | "Four TON-IoT FedAvg/FedProx configurations were repeated over 3 model-initialisation seeds (fixed partition)" |
| C2 | H | §10.6 | "the final model is sensitive to the random client partition drawn at initialisation" | False. The partition is identical across seeds 42/43/44 (`dirichlet_partition(..., seed=SEED)` default). Only initialisation and dropout vary | Attribute the variance to initialisation/training stochasticity, or re-run with seed-dependent partitions |
| C3 | H | §10.6 | "FedProx ... consistently underperforms FedAvg α=0.5 ... across all three seeds" | False. Seed 43: FedProx 0.7786 > FedAvg 0.7597 (`seed_43/*.json` best_f1). On **final-round** F1, FedProx's mean (0.7428±0.043) is above FedAvg's (0.6997±0.122) | "FedProx is lower in 2 of 3 seeds. The mean difference (2.0 pp best-round) is within seed variance" |
| C4 | M | §10.6 | "confirming that the qualitative ordering of algorithms is stable across seeds" | The FedAvg/FedProx ordering flips at seed 43 (C3). Only two algorithms are in the sweep | Remove or qualify |
| C5 | H | Abstract, §5, §10.1, §11 | "only 3.5 pp below centralised" and "consistent ~3.5 pp gap ... on both datasets" | Uses the **best seed** (seed 42: 0.8449. Seeds 43/44: 0.8011/0.8272, mean 0.8244, so the gap is **5.6 pp**) and the **best test checkpoint** (final 0.8193, gap 6.1 pp; 3-seed final mean 0.7881, gap 9.2 pp). The UNSW 3.5 pp is seed 42 only. Relative gaps differ (4.0% TON vs 6.7% UNSW) | Report the mean±std gap, or best and final; drop "consistent" |
| C6 | H | All "Best F1" columns/claims | Best F1 used as the performance measure | `best_f1` is the max over 10–25 **test-set** evaluations, which is model selection on test data (val is merged into training, so no held-out val exists). All best-F1 figures are optimistic | Select the round on a held-out validation split, or report final-round F1 as primary |
| C7 | M | Abstract, §6, §10.2, §11 | α=0.1 "fails to converge", "fundamentally cannot converge ... regardless of training duration" | By the paper's own definition the convergence round is 12 (Table 3a). Seed 44 peaks at R44 (0.6316), so it is still improving late. Only 50 rounds tested, so "regardless of duration" is untested. UNSW α=0.1 rises smoothly (best 0.3565@R24, final 0.3530) | "unstable/oscillating within 50 rounds on TON-IoT" |
| C8 | M | Abstract | "performance degrades to 0.5348 even after 50 rounds" | 0.5348 is the **R14 peak**. After 50 rounds F1 = 0.3021 (`final_metrics.f1`) | "peaks at 0.5348 (R14) and ends at 0.3021 after 50 rounds" |
| C9 | M | §10.2 | "This finding [oscillation, no convergence at 50 rounds] is consistent across both TON-IoT and UNSW-NB15" | False for UNSW: α=0.1 is monotone-ish and plateaus (`results_unsw/fedavg_alpha0.1.json`), 30 rounds only | Limit the oscillation claim to TON-IoT |
| C10 | H | Abstract, §7, §10.4, §11 | SCAFFOLD α=0.1 "+6.6 pp over FedAvg"; "preferred when heterogeneity is severe" | Single seed. FedAvg α=0.1 across seeds is 0.5348/0.5828/**0.6316** (mean 0.5831). Seed 44 FedAvg beats SCAFFOLD (0.6011). The gain over the mean is 1.8 pp, below 1 sample-std (0.048). **UNSW SCAFFOLD α=0.1 = 0.3534 < FedAvg 0.3565**: the result exists (§3.7 says both datasets) but is never reported, so this is selective reporting. The TON SCAFFOLD α=0.1 trace collapses to 0.0728 at R32 (instability not mentioned) | Report UNSW α=0.1. Run SCAFFOLD over seeds. Downgrade to "comparable/inconclusive" |
| C11 | M | §7, §10.4, §11, Table 4 note | FedAvg-SGD ≈ FedAvg-Adam at α=0.1 (Δ<0.002) means the "optimiser confound fully eliminated" / "optimiser irrelevant" | Single seed, best-checkpoint comparison. Δ=0.0013 is far inside FedAvg's seed spread (std 0.04). At α=0.5 the optimiser effect is 23.8 pp | "no measurable optimiser effect at α=0.1 in one run" |
| C12 | M | Abstract, §7, Table 4, §10.4, §11 | FedAdam "diverges after round 8", second-moment growth makes the adaptive lr unstable | F1 never exceeds 0.113 and accuracy never exceeds 0.213 (TON) or 0.397 (UNSW). It never learned. With server lr=1e-3 and Adam's normalised step, each weight moves about ≤1e-3 per round, so the global model stays near initialisation. "Diverges" and the v-growth mechanism are not shown by any logged quantity | "fails to train with server lr=1e-3 (untuned)". Drop the mechanism, or tune η_s (Reddi et al. use far larger server lr) |
| C13 | M | §3.5 | "FedAdam is evaluated on TON-IoT only ... renders UNSW-NB15 evaluation uninformative" | Contradicted by `results_unsw/fedadam_alpha0.5.json` and by §7/Table 4/§10.4/§11, which report it | Delete the sentence |
| C14 | H | §3.6, §8, §10.5 ("slightly overestimating the DP budget") | (ε,δ)-DP with ε≈4.0/2.1/0.6 | The code clips the **batch-mean** gradient **per tensor** to C (`local_train_dp`), not per sample. The accountant assumes per-sample clipping with noise σ relative to sensitivity C. Actual per-example sensitivity of a clipped batch mean is ~2C (×√L across tensors), against noise σC/B, so the effective noise multiplier is about B/(2√L) times smaller. **The reported ε is not a valid guarantee, and the real ε is far larger, not "slightly over-estimated"** | Implement per-sample clipping (e.g. TF-Privacy DP-Keras optimiser) or remove the ε claims. Fix the limitation's direction |
| C15 | H | Abstract, §8, §10.3 | DP collapse is "a fundamental property of the noise-clipping interaction", "at any reasonable epsilon", "near-random (random F1≈0.10)" | F1 values (0.006–0.064) are **below** chance and become constant to 4 dp after R4–R6 (σ=1: final accuracy 0.0148, a constant minority-class predictor). That points to training breakdown or an implementation artefact (one of the DP runs crashed earlier on None gradients, see `run_fl.log`), not a measured privacy-utility curve. All are single-seed and TON plus 1 UNSW run | Present as "our DP implementation failed to train". Remove "fundamental" and "any reasonable ε" |
| C16 | M | §8 para 1 and last para | B=1024 "confirms that gradient clipping — not noise magnitude — is the primary cause" | Non-sequitur. B=1024 changes noise scale, step count (4× fewer) and ε (9.0); clipping (C) was never varied. Under Adam the gradient scale set by clipping is largely normalised away | Remove the causal claim, or add a C sweep / no-clip+noise / clip-only ablation |
| C17 | L | §8 | UNSW collapse is worse "consistent with its smaller per-client dataset size" | UNSW pool 149,039 vs TON 156,017 (−4%). Min client 25,419 vs 26,232. Difference is negligible and the explanation speculative | Drop or soften |
| C18 | M | §5 para 2 | 15.3 MB "is negligible compared to transmitting the full training dataset" | TON pool at 25 float32 features = 156,017×25×4 = 15.6 MB (about the same as 15.3 MB). UNSW at 16 features = 9.5 MB, which is **less** than FL's 15.3 MB. At 64 pre-selection features TON = 39.9 MB. Not "negligible" | State the comparison numerically. The benefit is privacy, not bandwidth |
| C19 | M | §5, Table 2, §10.1, §11 | Centralised-vs-federated gap | Not like-for-like: the v1 centralised CNN was trained on the train split (128,484) **with SMOTE** and val early-stopping. FL trains on train+val (156,017), no SMOTE, test-set best round | State the differences, or re-run centralised on the same data/no SMOTE |
| C20 | M | §6, §10.1, §11 | IID gap "validated across datasets" | UNSW α=1.0 (0.5194) **beats the centralised baseline** (0.5182). UNSW FL accuracy (0.8267) beats centralised accuracy (0.7712). Both are unmentioned and weaken the "consistent gap" story | Discuss |
| C21 | L | §11 | "Non-IID partitioning degrades performance proportionally to heterogeneity severity" | UNSW α=1.0 > IID (+3.6 pp), so not monotone, let alone proportional | "generally degrades, with an α=1.0 exception on UNSW" |
| C22 | M | §9, §11 | TON scaling "decreases strictly monotonically" | Holds only for best-round, seed-42. Final-round: N=2 0.7655 < N=4 0.7726. Each N uses a different Dirichlet draw (partition confound). Single seed | Qualify. Report final-round too |
| C23 | L | §9 | "N=8 marginally outperforms N=4 (0.4403 vs 0.4010)"; explanation via rare-class coverage | The gap is 3.9 pp, not marginal (larger than the IID gap the paper calls meaningful). N=16 is only 0.4 pp below N=4. The rare-class explanation has no per-class evidence and uses wrong counts (N12) | Rephrase. Mark the explanation as a hypothesis |
| C24 | L | §9 | "N=2 achieves the best federated F1 on both datasets" | On UNSW the α=1.0 run (0.5194) is higher than N=2 (0.4930). True only within the α=0.5 scaling set | "best among the α=0.5 scaling runs" |
| C25 | M | Abstract, §7, §10.4, §11 | "FedProx does not outperform FedAvg on either dataset" / "without accuracy benefit" | UNSW difference 0.39 pp, single seed (noise level). TON: seed 43 FedProx wins. On final-round F1 FedProx is better on average (C3) | "no significant difference" |
| C26 | M | §4.2 | "All experiments are seeded (seed=42) for reproducibility" | Runs are not deterministic. The seed-42 TON α=0.1 run in `run_fl.log` (20 rounds) gives R2 0.3882 / R18 0.5648, while the cached seed-42 50-round JSON gives R2 0.3033 / best 0.5348@R14 over the same first rounds | State GPU non-determinism. The seed does not guarantee bit-reproducibility |
| C27 | L | §5 | Faster TON convergence "reflecting the higher signal-to-noise ratio in the more class-balanced TON-IoT partitions" | Speculative. No SNR or balance metric reported | Mark as hypothesis or drop |

## C. Unfinished text, internal inconsistency, housekeeping

| # | Location | Problem | Fix |
|---|---|---|---|
| U1 | Figure 3 caption | "Pending results shown as 0." Unfinished, and all α results exist for both datasets | Delete. Regenerate the figure |
| U2 | Figure 6 caption | "Pending UNSW results shown as empty." All UNSW N=2/8/16 results exist | Delete. Regenerate the figure |
| U3 | §11 para 2 end | Corrupted fragment "5 (-0.1306 pp vs. FedAvg), with a meaningful gain at α=0.1 confirming variance reduction benefits under severe heterogeneity. Multi-seed validation across three seeds (42, 43, 44) has been completed; results are reported in Table 7." It is a broken sentence plus an internal status note | Delete |
| U4 | §7 / §10.4 / §11 | The SCAFFOLD vs FedAvg-SGD argument (10.8 pp / 6.8 pp / Δ<0.002) is repeated almost word-for-word three times. Double spaces after "benefits." and "limitation." in §7 | Keep once (§7). Summarise elsewhere |
| U5 | §10.5 fifth limitation | "a fair comparison would require a SGD-based FedAvg baseline". Stale: that baseline is already in Table 4/§7 | Rewrite: the remaining confound is the lr/optimiser sweep and the single seed |
| U6 | §3.2 vs §4.1 | Duplicate sentence "No SMOTE is applied — clients train on the original class-imbalanced data, reflecting realistic FL conditions" | Keep one |
| U7 | §4.2 vs §10.6 | "All experiments are seeded (seed=42)" contradicts the multi-seed section | Reconcile |
| U8 | §10.6 prose | Code identifier "fedavg_alpha0.1" in running text | "FedAvg at α=0.1" |
| U9 | Header / references | "Paper 4 of 4" (the project series has 7 papers). [19] Flower is never cited in the text. [24] listed before [23]. [22] placed after Data Availability. §3.3 calls FedAvg local training "SGD" although it uses Adam. §3.3 uses N both for client count and Σn_k | Fix the series label, order the refs, fix the notation |

## D. Citation-content advisories (not results-checkable; verify against the sources)

| # | Location | Concern |
|---|---|---|
| R1 | §2.3 [15] Rey et al. 2022 | That paper is FL malware detection on N-BaIoT, not FedAvg on CICIDS-2017 "within 5% of centralised" |
| R2 | §2.3 [16] Nguyen et al. DIoT | DIoT uses a GRU-based model, not an LSTM. Check the "significant degradation under non-IID" attribution |
| R3 | §2.3 [17] Mothukuri et al. | A general FL security/privacy survey, not a survey of FL-based IDS |
| R4 | §2.2/§3.8 [14] Hsieh et al. 2020 | Dirichlet label partitioning is usually credited to Hsu et al. 2019. Check the attribution |

## E. Results that exist for one seed only, and the claims that rest on them

Multi-seed (seeds 42/43/44, TON-IoT only, same partition): `fedavg_iid`, `fedavg_alpha0.5`, `fedavg_alpha0.1`, `fedprox_alpha0.5`.

Single seed (42) only:
- **TON-IoT:** `fedavg_alpha1.0`, `fedadam_alpha0.5`, `dp_fedavg_eps13/eps7/eps2`, `dp_fedavg_lb`, `scale_N2/N8/N16`, `scaffold_alpha0.5`, `scaffold_alpha0.1`, `fedavg_sgd_alpha0.5`, `fedavg_sgd_alpha0.1`.
- **UNSW-NB15:** every run (`fedavg_iid/alpha1.0/alpha0.5/alpha0.1`, `fedprox`, `fedadam`, `dp_fedavg_eps13`, `scaffold_alpha0.5/0.1`, `scale_N2/N8/N16`). The UNSW `global_summary.json` also omits fedadam and dp (it was not regenerated after `run_extra_experiments.py`).

Claims resting on single-seed results:
- SCAFFOLD +6.6 pp over FedAvg at α=0.1, and "preferred under severe heterogeneity". FedAvg is multi-seed but SCAFFOLD is not, and the gain vanishes against seed 44.
- SCAFFOLD +10.8 pp / +6.8 pp over FedAvg-SGD. FedAvg-SGD Δ<0.002 "confound eliminated".
- All UNSW claims: "consistent 3.5 pp gap", α=1.0 anomaly, FedProx < FedAvg on UNSW, FedAdam "dataset-independent divergence", UNSW DP collapse, UNSW N=8 non-monotonicity.
- FedAdam divergence (both datasets).
- The whole DP section (ε values, the 94%/92% drops, the B=1024 "clipping is the cause" conclusion).
- Client scalability: TON "strictly monotonic", N=2 "best", the comm-vs-accuracy guidance.
- α=1.0 "gap below 6 pp" (§10.1).
- §6 "oscillates ... regardless of training duration" is based on the seed-42 trajectory. Seeds 43/44 exist but are not used for this claim, and seed 44 contradicts it (late peak R44).

## F. Verified items (text/table value = data)

Dataset/setup: TON 25 features, 10 classes; UNSW 16 features, 10 classes (`global_summary.json` n_features/n_classes). TON total 183,550, pool 156,017, test 27,533. UNSW total 175,341, pool 149,039, test 26,302 (n_train/n_test; `run_fl.log` and UNSW log: 0 duplicates removed). 9 attack + Normal for both. Conv1D 32/64, k=3, BN, max-pool, GAP, dense 128→64, dropout 0.3, 10-way softmax (`build_model`). 93.5 KB = 95,784 B/1024. N=4, 2 local epochs, Adam lr 1e-3, batch 256. TON 20 rounds (50 for α=0.1); UNSW 20 (30 for α=0.1). Eval every 2 rounds. Convergence = 90% of peak (code L453-456). Comm formula as in code. μ=0.01. FedAdam η_s=1e-3, β1=0.9, β2=0.99. DP C=1.0, σ∈{1.0,1.5,4.0}, δ=1e-5, noise std σC/B. σC/B=0.0039 (B=256) and 0.00098 (B=1024). SCAFFOLD control-variate update formula matches code. SCAFFOLD uses SGD. Centralised CNN TON F1 0.8803, acc 0.9372, UNSW F1 0.5182 (v1 backup). Sub-ms CNN latency (0.19/0.16 ms mean, v1 `latency.json`).

Abstract/§5/Table 2: TON FedAvg IID best 0.8449 (R16 = "in 16 rounds"), gap 3.54 pp, conv 4, acc@best 0.9154. UNSW IID 0.4836, gap 3.46 pp, conv 10, acc@best 0.8267. Per-round 0.766 MB (≈0.77). SCAFFOLD 0.6540 (α=0.5) and 0.6011 (α=0.1).

Table 3a: IID 0.8449/0.8193/4/20/+0.0000. α=1.0 0.8225/0.8225/6/20/−0.0225. α=0.5 0.7846/0.7726/6/20/−0.0604. α=0.1 0.5348/0.3021/12/50/−0.3101. Text: 0.8449→0.8225; −3.8 pp to 0.7846; α=0.1 peak at R14.

Table 3b: IID 0.4836/0.4793/10/20/+0.0000. α=1.0 0.5194/0.5194/8/20/+0.0358. α=0.5 0.4010/0.3856/16/20/−0.0826. α=0.1 0.3565/0.3530/6/30/−0.1271.

§7/Table 4: FedAvg 0.7846/0.4010/conv 6/961 s. FedProx 0.7577/0.3971/6/1223 s, +27% time (1223.4/961.0 = 1.273). SCAFFOLD 0.6540/0.3414/12/1323 s. FedAvg-SGD α=0.5 0.5464/conv 4/253 s. FedAvg-SGD α=0.1 0.5335/conv 18/733 s/50 rounds. FedAdam TON 0.1128/conv 4/920 s, best at R8, final 0.0145. FedAvg-SGD 23.8 pp below FedAvg-Adam. SCAFFOLD − FedAvg-SGD = 10.8 pp (α=0.5) and 6.8 pp (α=0.1). |FedAvg − FedAvg-SGD| at α=0.1 = 0.0013 < 0.002. SCAFFOLD − FedAvg α=0.1 = 6.6 pp (seed 42). UNSW FedAdam peaks at R8. UNSW SCAFFOLD α=0.5 0.3414. Multi-seed means FedProx 0.7635, FedAvg 0.7839.

§8/Table 5: ε 4.008/2.064/0.622 (σ=1/1.5/4, B=256). Best F1 0.0468/0.0198/0.0497/0.0642. Drops −0.7377/−0.7648/−0.7348/−0.7204. 94% and 92% relative drops. B=1024.

§9/Table 6: TON 0.8390/0.7846/0.7501/0.7127. UNSW 0.4930/0.4010/0.4403/0.3966. Comm/round 0.383/0.766/1.533/3.065 MB. Totals 7.66/30.65/61.30 MB (15.32 → 15.33, see N14). N=2 TON close to IID (0.8390 vs 0.8449).

§10/§11: α=1.0 vs centralised gap 5.8 pp (< 6 pp). Rel. drop > 33% (actually 36.7%). Table 7 per-seed values all match (0.8449/0.8011/0.8272; 0.7846/0.7597/0.8073; 0.5348/0.5828/0.6316; 0.7577/0.7786/0.7541). Means 0.8244/0.7839/0.5831/0.7635. Std ±0.018 (IID) and ±0.040 (α=0.1) as population std. Max std 0.040 is α=0.1. Conclusion TON scaling values match.

## G. Extra context the revision may need (derived, not in the paper)

- Final-round 3-seed means (TON): IID 0.7881±0.060, α=0.5 0.6997±0.122, α=0.1 0.3454±0.119, FedProx 0.7428±0.043 (sample std).
- Best-round 3-seed sample std: IID 0.022, α=0.5 0.024, α=0.1 0.048, FedProx 0.013.
- α=0.1 runs (TON 50 rounds) cost 38.31 MB. UNSW α=0.1 (30 rounds) costs 22.99 MB. Not stated in the paper.
