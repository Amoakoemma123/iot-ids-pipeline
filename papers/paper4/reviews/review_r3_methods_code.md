# Paper 4 v2 (Federated CNN-IDS): methods and code audit, round 3

Date: 2026-10-25.

Scope:
- `Papers pipeline/experiments/paper4_federated/run_fl_v2.py` (RF2) at HEAD 8843fb7, and its diff against 857a346 (round 2).
- `src/data/loader.py` (LD), including the new `return_meta` path.
- Queue scripts `logs/p4r3_*.sh` and `logs/p4r4_*.sh`.
- All result JSONs in `results_v2/` (plus `_superseded_r2/` where the text quotes it).
- `Revision_v2/Paper4/compute_values.py` (CV), `values.json`, and the manuscript `Manuscript_filled.md` (P).

How it was checked:
- I read the code and the full code diff since round 2. Only RF2 and LD changed.
- I imported `experiments()` from RF2 (CPU, no training) and compared the config the code generates today with `cfg` in every stored JSON. I also checked seeds, history length and non-finite flags.
- I recomputed every stored ε with `dp_accounting` (RDP, plus PLD for comparison).
- I reloaded TON-IoT with `return_meta=True`, rebuilt the natural partition for all five repetition seeds, and compared them.
- Scripts are in my scratch directory. No file other than this report was written, and no model was trained.

Severity: **critical** = a reported result or headline claim is invalid. **major** = a result is biased, or a method or claim is misdescribed in a way a referee would reject. **minor** = needs a fix or a disclosure.

**Summary.** The round-2 critical bug (C1) is fixed:
- FedProx, FedAdam and FedLC were tuned and run at the tuned client lr, 0.003.
- All 1,000+ result files match the configs the current code generates.
- Every ε reproduces exactly.

No new critical issue. There are four major issues, all in how results are described or tested, none in the training code:
1. The "natural" partition is effectively one fixed partition.
2. "Own traffic" is not what was measured.
3. Several headline claims are not significant after the Holm correction the paper says it relies on.
4. FedLC has no Methods description.

---

## Critical

None.

---

## Major

### M1. The natural device-level partition is effectively the same partition in all five repetitions, and the text describes it wrongly
- **Where:** RF2:114-124. P:53 says "taking hosts in random order and assigning each to the currently smallest client". P:51 says "Each repetition draws a new partition".
- **What the code does:** it sorts hosts by size, largest first (`order[np.argsort(-sizes[order], kind='stable')]`, RF2:122). Randomness only breaks ties between hosts of equal size. TON-IoT has 682 destination hosts in training, and 235 of them hold a single record. So randomness only moves the tiny hosts.
- **Evidence:**
  - `client_sizes` is [33,550, 31,645, 31,645, 31,644] in all 5 repetitions of every `*_nat` experiment.
  - Rebuilding the partitions puts only 1.18–1.32% of training records in a different client than repetition 0.
  - Client 0 is always one single host, 192.168.1.190: 26% of the training data and 7 classes.
- **Impact:**
  - The ± and p-values in §V-C (Table III; FedAvg −0.129 against central, Holm p = 0.017; +0.266 against local, Holm p < 0.001) reflect only initialisation and batch-order noise, not partition variability.
  - The Dirichlet comparisons do include partition variability. So the claim "Federation costs more than under Dirichlet skew" compares intervals of different kinds. It is also not tested directly.
  - "0 of 20 site–repetition pairs" is really 4 sites × 5 near-identical runs.
  - Three of the seven Holm-significant TON results are natural-partition comparisons. Their low p-values partly come from this missing variance source.
- **Fix:**
  - Text: describe the assignment as largest-first greedy balancing, and say the partition is essentially fixed across repetitions. Then the repetitions vary only initialisation and batch order. Say that client 0 is one host.
  - Optionally, randomise the host order truly (drop the size sort; balance will be worse), or draw several balanced partitions by random restarts. Then re-run.

### M2. "Every site gains, also on its own traffic" is not what was measured
- **Where:** abstract (P:9), P:145, P:223 and the conclusion (P:231). The computation is CV:184-198.
- **What is measured:** the per-site gain is the mean per-class F1 on the **pooled** test set, restricted to the classes that make up ≥1% of the site's training data. That test set mixes traffic to all 682 hosts. It is "own classes", not "own traffic", meaning the site's own devices.
- **Why it matters:** for a device partition, the obvious reading of "own traffic" is test records whose destination host belongs to the site. Those can be identified: the loader already returns `meta_test['dst_ip']`, and 99.8% of test records have a training host. Pooled per-class F1 also includes precision on other sites' devices.
- **Impact:** the device-level headline is overstated. The §V-C mechanism ("a site's own model generalises poorly even to the classes it has seen") is measured on other sites' devices, which supports a different claim.
- **Fix:** either reword to "on the classes it holds", or evaluate on each site's own test hosts. That second option needs predictions, and the models are not saved, so it means a re-run of `fedavg_nat` and `local_nat` with per-host predictions stored. It is cheap on the GPU.

### M3. Headline effects are not significant after the Holm correction the paper says its conclusions rest on, and one Discussion claim is false
- **Where:** P:133 says "we base conclusions on effects that are significant after Holm correction". In `values.json`, only 7 comparisons are Holm-significant: on TON, FedAvg vs local at α = 0.1, three natural-partition tests, and SCAFFOLD vs FedAvg at α = 0.5; on UNSW, FedAvg vs central at IID and at α = 0.5.
- **Not Holm-significant but stated as findings:**
  - Abstract: TON FedAvg vs central −0.059 (Holm 0.348), vs local +0.080 (0.083), α = 0.1 gap −0.185 (0.147).
  - P:85: "the more heterogeneous the clients, the more federation costs". No test of the trend is given, and none of the TON FedAvg-vs-central gaps is Holm-significant.
  - Conclusion: the gap "grows from −0.042 with IID clients to −0.185" (Holm 0.059 and 0.147).
- **False claim in §VI-A (P:219):** "in a preliminary tuning … FedProx and FedAdam … both appeared worse than FedAvg …; neither difference survived matched tuning". For FedAdam on TON at α = 0.5 the effect is almost unchanged:
  - Round 2, mismatched lr: −0.089 [−0.114, −0.063].
  - Now, matched lr: −0.083 [−0.118, −0.049], raw p = 0.003.
  - Only the Holm correction, new in this round, moved it above 0.05. Matched tuning did not remove the FedAdam deficit; it confirmed it. The tuning runs agree: FedAdam's best val F1 is 0.791 against FedAvg's 0.858 on the same tuning seeds.
- **Fix:**
  - Phrase the non-significant effects as estimates ("−0.059 [−0.101, −0.016], not significant after Holm correction").
  - Drop or qualify "the more heterogeneous … the more it costs", or test the trend directly.
  - Rewrite the §VI-A sentence: the FedProx/UNSW gain disappeared; the FedAdam deficit on TON stayed at about −0.08 but is not significant after correction.

### M4. FedLC is reported in the abstract, Tables I–III and §V-B/C, but has no Methods description
- **Where:** §III-E and §IV-A (P:65, P:71) do not mention FedLC. The code is RF2:224, 232-234 and 400-402, with the τ grid at RF2:76.
- **What the code does:**
  - The loss uses calibrated logits log softmax − τ·n_{k,c}^{−1/4}. That is equivalent to FedLC's z_c − τ·n_{k,c}^{−1/4}, because log-softmax shifts the logits by a constant.
  - Counts come from the client's own training labels. Absent classes are clamped to n = 1, so they get the largest offset, τ.
  - Evaluation uses uncalibrated outputs.
  - τ is tuned over {0.1, 0.3, 1} on the two tuning repetitions. The selected values are 0.3 (TON) and 1.0 (UNSW). UNSW's value is the **upper grid edge** (`fedlc_edge: true` in `tuned_v3.json`).
  - The TON scores span only 0.0009 (0.8604–0.8613), far below run-to-run noise (see m2).
- The implementation is correct, but none of this is in the paper.
- **Fix:** add one sentence to §IV-A (form of the calibration, per-client counts, the clamp for absent classes) and the τ grid, selected values and edge flag to §III-E.

---

## Minor

1. **Run-to-run noise is now directly measurable and larger than several tuning margins (RF2:267 `jit_compile=True`, no op determinism).**
   - On TON, `scaffold_a0.1` and `scaffold_a0.1_lr0.1` have **identical** configs (TON's tuned SCAFFOLD lr is 0.1). Per repetition, their test F1 differs by 0.000, 0.023, **0.041**, 0.015 and 0.009.
   - FedProx at μ = 1e-4, which is almost FedAvg, differs from FedAvg by up to 0.036 val F1 on the same UNSW tuning seed.
   - So the choices of μ, τ and the TON FedAvg-SGD lr (0.3 vs 0.01–0.1, within 0.01) are within noise.
   - P:61 discloses non-determinism. Fix: add the measured spread (e.g. "identical configurations differ by up to 0.04 macro-F1").
2. **The Holm family is padded with redundant and ablation tests (CV:136-147).**
   - The 29 TON tests include `scaffold_a0.1_lr0.1` vs `scaffold_a0.1`, which on TON compares a configuration with itself.
   - They also include `scaffold_a0.1_lr0.1` vs `fedavgsgd_a0.1`, which duplicates `scaffold_a0.1` vs `fedavgsgd_a0.1`, plus the BN and SCAFFOLD-lr ablations.
   - This makes Holm needlessly conservative for the primary questions.
   - The 100-round and port-ablation p-values (CV:153-158) are unadjusted, and P:135-137 does not say so.
   - Fix: define a primary family (federation-vs-references and algorithm-vs-FedAvg), drop the TON self-comparison, and label the other p-values as unadjusted.
3. **"Partitions disjoint from the five reported repetitions" (P:65) is the wrong word.** Tuning seeds 100 and 101 draw independent partitions of the same training records and use the same validation set. Fix: "independently drawn partitions (seeds not used for reported runs)".
4. **"The same code without noise" in the full-budget and 40-round DP checks (P:191) also has no clipping.** `dp_full_nonoise` (RF2:611) and `dpt_r40_nonoise` (RF2:628) set `clip=0`. Fix: "without clipping or noise".
5. **The centralised reference is still not exactly "the same optimiser" (open from r2 minor 8).** `run_centralised` (RF2:486-515) never resets Adam, while FedAvg resets it every round (RF2:396). Its lr is FedAvg's, not tuned for centralised training. P:61 discloses pooled-validation selection for local-only, but not these two points. Fix: one clause in §III-D.
6. **DP implementation details (open from r2 minor 11):**
   - An empty Poisson draw skips the step (RF2:311). The probability is negligible, and the accountant still counts the step, so ε stays conservative.
   - The sampling and noise PRNGs are seeded per repetition, not cryptographically secure.
   - Optionally: the unit of privacy is one flow; a device's many correlated flows are not protected as a whole.
   - Fix: one sentence in §IV-B.
7. **The BN swing statistic covers rounds 11–50 only (CV:160-163).** P:161 says "mean change … between consecutive rounds" without that range. Fix: add the range.
8. **Release hygiene.**
   - The round-2 `tune_*` directories (single seed, mismatched lr for FedProx/FedAdam) still sit next to `tune3_*` in `results_v2/<ds>/`.
   - The `tune` group (RF2:558-568), `TUNE_GRID` and `finalize_tuning` still contain the round-2 single-stage logic. Re-running `--group tune` would reproduce the C1 bug.
   - CV:99 reads the stale `tuned.json` before CV:336-343 overwrites the values from `tuned_v3.json`. FedLC is not in that list.
   - The UNSW SCAFFOLD lr 0.3 runs quoted in P:141 exist only in `_superseded_r2/`.
   - Fix: move the round-2 tune results into `_superseded_r2`, delete or guard the `tune` group, and say in the README which folders back which text.
9. **Still open from rounds 1/2:**
   - There is no schema assertion when `pipeline.pkl` is loaded (RF2:93-101).
   - No worst-client metric is reported; the own-class per-client gain is now reported, which partly addresses this.
   - In the TON DP JSONs, `cfg.norm = 'bn'` is cosmetic; the top-level `norm = 'gn'` is what ran, because RF2:378 forces GN.

---

## Verified correct (no action)

- **Round-2 C1 fix:**
  - `tune3b` and `tunelc` ran in a separate process after `finalize3a` (p4r3_gpu.sh, p4r4_gpu.sh).
  - Every `tune3_fedprox_*`, `tune3_fedadam_*` and `tune3_fedlc_*` JSON, and every core, natural or FedLC FedProx/FedAdam/FedLC JSON, has `lr = 0.003` = `tuned_v3.fedavg`.
  - μ, η_s and τ in the core runs equal `tuned_v3.json` (TON μ 1e-4, η_s 0.03, τ 0.3; UNSW μ 1e-4, η_s 0.03, τ 1.0).
  - SCAFFOLD is 0.1 / 0.03 and FedAvg-SGD 0.3 / 0.03, as stated in P:65.
  - Selection is the mean over the two tuning seeds (RF2:661-679).
- **Config conformance:** I compared every result directory in `results_v2/{TON_IoT,UNSW_NB15}` with what `experiments()` generates now. They match exactly in algo, α, lr, μ, server_lr, τ, rounds, local epochs, norm, partition, clip, σ and N; the only difference is the cosmetic TON DP `norm` field. Seeds and lengths:
  - core 0–4 (50 rounds)
  - extra, BN-recal and DP 0–2
  - long 0–2 (100 rounds)
  - natural, FedLC, ports, followup and followup2 0–4
  - dptune and dpfull 0–1 (20, 40 and 50 rounds as stated)
  - tune3 seeds 100 and 101
- **Selective rerun (`prepare_rerun`, RF2:682-701):**
  - TON moved FedProx, FedAdam and FedAvg-SGD; UNSW moved FedProx, FedAdam and SCAFFOLD (including lr 0.1, which was then re-run).
  - The FedAvg lr was unchanged on both datasets, so central, local, FedAvg, scaling and BN runs were kept from c306262.
  - The code diff since 857a346 leaves the non-FedLC `step`, `dp_step`, aggregation, SCAFFOLD and FedAdam code unchanged. The kept runs are therefore valid for the current code.
- **Natural partition alignment (LD:88, 150-151, 234-237):**
  - `meta` is copied before any row drop, re-indexed by `df.index` after `dropna` and deduplication, and sliced with the same `idx_train/val/test` as X, after the cross-split guard.
  - The CSV has a unique RangeIndex. `meta_train` length equals `X_train` (128,484) and `meta_test` equals `X_test` (27,533).
  - IPs never become features (they stay in `_DROP_COLS`).
  - Local-only and federated runs get the same natural partition (same `prng`, same calls).
- **Dirichlet and IID partitions:** these come from `default_rng(1000+rep)`, and sizes differ across repetitions (e.g. α = 0.5: [25,576, 23,961, 26,463, 52,484] for rep 0 against [20,920, 36,633, 41,331, 29,600] for rep 1).
- **FedAvg, FedProx, FedAdam and SCAFFOLD:**
  - Unchanged from round 2 and correct.
  - SCAFFOLD's uniform averaging of the control variate is now disclosed (P:71).
  - FedProx is described as using local Adam (P:71).
- **FedLC math:** see M4. It is correct.
- **Validation selection and no test leakage:**
  - `select` (RF2:362-365) uses validation F1 only, earliest maximum.
  - `keep_best` and `run_centralised` keep the weights of that round.
  - Test scores are logged but never read by any choice, including tuning (`finalize_tuning3` reads `val_f1_selected`).
- **GroupNorm and BN:**
  - GroupNorm is unchanged and correct.
  - BN running statistics are averaged with all weights.
  - The BN recalibration diagnostic (RF2:344-359) sets momentum 0, does one training-mode pass on 8,192 pooled training records, then restores the statistics. It is correctly labelled diagnostic and non-deployable (P:161).
- **DP-SGD:**
  - Per-example clipping uses the global norm, and Gaussian noise σC goes on the sum.
  - The sum is divided by the expected batch B = q·n_k.
  - Each epoch has ceil(1/q) Poisson draws, and the accountant counts R·E·ceil(1/q) Poisson-subsampled Gaussian steps with RDP at δ = 1e-5. Sampler and accountant agree.
  - The model is GN with dropout 0.
  - The assumptions that are public are now stated: preprocessing, n_k and hyperparameters (P:75; resolves r2 M1).
- **ε:** all 37 DP JSONs with noise reproduce exactly. Examples:
  - TON σ 0.6 / 1 / 1.5 / 2.5: mean 12.15 / 3.12 / 1.58 / 0.81.
  - TON full budget σ 1.8 at 50×2: 2.81 and 3.03, mean 2.9.
  - TON 40-round σ 1: 4.25 and 4.60, mean 4.4.
  - UNSW: 18.52 / 5.27 / 2.62 / 1.33.
  - PLD gives values about 8–12% tighter; optionally report them.
- **DP robustness numbers in P:191:**
  - C 2 / 0.5 gives 0.667 / 0.649.
  - lr 0.001 / 0.006 gives 0.604 / 0.652.
  - 40 rounds gives 0.659 against 0.772.
  - Full budget gives 0.670 against 0.847.
  - All match the JSON means over two repetitions.
- **100-round runs:** central at 100 rounds × 2 epochs = 200 passes. FedAvg IID, α = 0.5 and α = 0.1 at 100 rounds, seeds 0–2, paired by seed.
- **Holm implementation (CV:168-174):** the step-down is correct (running max of (m−i)·p_(i), capped at 1). The family has 29 tests on TON and 22 on UNSW, as stated in P:133.
- **Disclosures added since round 2:** dropped last batch, GPU/XLA non-determinism, pooled validation for local-only, the deployable-selection argument, the dropout and batching difference in the DP comparator, FedProx and SCAFFOLD grid edges, the ± definition and the SD column.

---

## Round-2 findings: status

| R2 | Finding | Status |
|---|---|---|
| C1 | FedProx μ / FedAdam η_s tuned at client lr 1e-3, run at 3e-3 | **Fixed.** Staged tune3a → finalize3a → tune3b in separate processes; all FedProx/FedAdam/FedLC tuning and core runs at lr 0.003; re-run complete. The `tune` group still contains the old logic (minor 8). The FedAdam deficit on TON survived the fix (M3). |
| M1 | DP guarantee overclaimed (preprocessing, n_k, hyperparameters) | **Fixed** (P:75 states the three public assumptions). Optional point on flow-level unit still open (minor 6). |
| M2 | Not reproducible; tuning margins below noise | **Partly fixed.** Disclosed (P:61); tuning now averages 2 seeds. Still no op determinism; measured spread not given; several choices still within noise (minor 1). |
| m1 | DP comparator differs in dropout and batching | **Fixed** (Table VI label, P:191). |
| m2 | DP lr/C/B not tuned under DP | **Fixed** (dptune grid: C ∈ {0.5, 1, 2}, lr ∈ {1e-3, 3e-3, 6e-3}, 40 rounds; full-budget run). |
| m3 | Rep 0 shared the tuning partition | **Fixed** (tuning seeds 100/101). |
| m4 | No multiple-comparison correction | **Fixed in mechanics** (Holm, 29/22 tests). **New:** claims not aligned with it (M3) and family composition (minor 2). |
| m5 | SCAFFOLD weighted vs uniform averaging | **Disclosed** (P:71). |
| m6 | UNSW SCAFFOLD α = 0.1 NaN runs | **Fixed/disclosed.** Re-tuned to lr 0.03 (no NaN); lr 0.01 / 0.1 / 0.3 reported, "some with non-finite weights". The lr 0.3 runs live in `_superseded_r2` (minor 8). |
| m7 | Last partial batch dropped | **Disclosed** (P:61). |
| m8 | Centralised / local references not exactly the same optimiser | **Partly fixed.** Pooled-validation selection is disclosed; no Adam reset and untuned lr are not (minor 5). |
| m9 | FedProx with local Adam | **Disclosed** (P:71). |
| m10 | Tuned values on grid edge | **Fixed** (wider grids; FedAvg lr 0.003 now interior; FedProx μ and UNSW SCAFFOLD edges disclosed, with SCAFFOLD lr 0.01 probed). **New:** FedLC τ on UNSW at the upper edge, undisclosed (M4). |
| m11 | DP empty-draw skip; PRNG | **Open** (minor 6). |
| m12 | Captions (± meaning, ε statistic) | **Fixed** (P:65 defines ±; SD column; ε "max over clients" with the range). |
| m13 | Per-client metrics; `pipeline.pkl` assertion | **Partly fixed** (own-class per-client gain reported); assertion **open** (minor 9). |

## Re-runs needed

- **None needed** for the correctness of the current numbers.
- **Optional, M1:** a truly randomised natural partition.
- **Optional, M2:** `fedavg_nat` and `local_nat` with per-host predictions stored, if "own traffic" is kept.
- Everything else is text: M1–M4 and minors 1–7.
