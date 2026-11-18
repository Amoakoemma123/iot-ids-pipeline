# Paper 4 (Federated CNN-IDS): methods and code audit, round 1

Date: 2026-10-25. Scope: `Papers pipeline/experiments/paper4_federated/run_federated.py` (RF), `run_extra_experiments.py` (RX), shared `src/data/loader.py`, `src/data/preprocessing.py`, `src/models/cnn.py`, `src/training/trainer.py`, `experiments/baseline/config.py`. Results: `results/`, `results/seed_43|44/`, `results_unsw/`, and `run_fl.log`. Paper: `Revision_v2/Paper4/Paper4_v1_text.md` (P).

How it was checked: I read all the code. Result JSONs and logs were parsed with small Python scripts. ε was recomputed with the `dp_accounting` 0.4.3 package from `ids_env`, which is pure Python. The client partitions were rebuilt from the class totals in `run_fl.log` using the same RNG stream. The rebuilt client sizes match the JSONs exactly. No file was modified, no training was run, and no model was loaded. `run_federated.py` is byte-identical to the v1 snapshot commit 7183c0e, so every result was produced by this code. The one exception is noted in C1.

Severity: **critical** = a reported result or headline claim is invalid. **major** = a result is biased or a method is misdescribed in a way a referee would reject. **minor** = needs a fix or a disclosure.

---

## Critical

### C1. DP-FedAvg clients never update the model weights, so every DP result comes from an untrained network
- **Where:** RF:187-189 and RF:201-202 (loss reduced outside the `GradientTape` context), RF:204-209 (the "skip None" filter).
- **What is wrong:** `per_sample_loss` is computed inside `with tf.GradientTape() as tape:`, but `tf.reduce_mean(per_sample_loss)` is evaluated *after* the block (RF:201). The tape never recorded the target op, so `tape.gradient` returns `None` for **every** trainable variable. The first DP run in `run_fl.log` crashed for exactly this reason: `tf.norm(g)` received `None` for the first trainable tensor, the conv1 kernel. That tensor sits on the loss path, so the crash cannot be explained by BatchNorm. The "fix" at RF:203-209 (comment: "skip None gradients from BN non-trainable vars") misdiagnoses the problem. `model.trainable_variables` contains no non-trainable variables. The filter therefore drops *all* gradients, and `opt.apply_gradients(zip([], []))` is a no-op. Only the BatchNorm moving mean and variance change, because `model(Xb, training=True)` updates them in the forward pass.
- **Evidence in the results:**
  - (a) After round 2 to 4 the DP histories are frozen. Every checkpoint has identical F1 and accuracy (eps7: F1 0.020 and acc 0.109 at all 10 checkpoints). That is a constant predictor: macro-F1 0.0197 is exactly what predicting the single class "injection", 10.9% of the data, gives.
  - (b) DP runs take **22 s/round** against **48 s/round** for FedAvg over the same number of batches, consistent with forward-only steps. B=1024 takes 7.1 s/round.
  - (c) The collapse does not depend on σ (1.0, 1.5, 4.0) or on B (256, 1024). The injected per-coordinate noise (σC/B = 0.001 to 0.016) is far too small to do that.
  - (d) UNSW DP shows the same frozen pattern.
- **Impact:** P Table 5, all of §8, the DP sentences in the abstract, §10.3 and the conclusion, and the UNSW DP number (0.0120/0.0138) are invalid. The "privacy-utility collapse" and "gradient clipping, not noise, is the cause" claims (P:216, P:238) have no support. §8's explanation that "BatchNormalization layers contribute None gradients" (P:238) repeats the bug.
- **Fix:** compute the loss inside the tape and **assert** that no gradient is `None` (fail loudly instead of filtering). Before any DP sweep, run two sanity ablations: (i) σ=0 with clipping disabled must reproduce `fedavg_alpha0.5` within seed noise; (ii) σ=0 with clipping only. Then apply C2.
- **Optional 5-line confirmation (no training):** `v=tf.Variable(1.); with tf.GradientTape() as t: l=v*2.` then `print(t.gradient(tf.reduce_mean(l), v))` prints `None`.

### C2. The DP mechanism provides no meaningful guarantee even once C1 is fixed, so the reported ε values are invalid
- **Where:** RF:199-216 (clipping and noise) and RF:309-333 (accountant). The paper is wrong at P:52, P:78, P:238 and P:284.
- **What is wrong:**
  1. **The clip is per batch, not per sample.** The clip is applied to the *mean* mini-batch gradient. Changing one record can change the clipped mean by up to 2C. The noise std is σC/B, so the effective noise multiplier is σ/(2B) ≈ 0.002 for σ=1, B=256. Even with the optimistic Poisson credit, RDP gives ε ≈ 6×10⁸ (σ=1) and ≈ 4×10⁷ (σ=4). In other words, there is no privacy.
  2. **The clip is per tensor, not global.** Each of the 14 trainable tensors is clipped to C separately (RF:207-211), so the global L2 norm can reach √14·C ≈ 3.74C. With true per-sample clipping, this alone would change the effective multiplier to σ/√14. The real ε at δ=1e-5 (q=0.00976, T=4080) would then be **256 / 61 / 3.5** for σ = 1.0 / 1.5 / 4.0, not the reported 4.0 / 2.1 / 0.6.
  3. **BatchNorm breaks per-example DP.** Batch statistics couple the samples in a batch. The BN moving mean and variance are part of `model.weights` (RF:89-90) and are sent to the server **with no noise**. That releases raw-data feature statistics outside any accounting.
  4. **The accountant's sampling assumption does not match the training loop.** The accountant assumes Poisson subsampling (RF:317-321), but the loop uses fixed-size shuffled batches that visit every record once per epoch (RF:180-184). Shuffling with Poisson-amplified accounting can underestimate ε considerably (Chua et al., ICML 2024, "How private are DP-SGD implementations?"). The steps are also counted as `n//B` (RF:315), while the loop runs `ceil(n/B)` batches, including a smaller final batch. That is a minor undercount: 4.009 becomes 4.03.
  5. **The level of privacy is mismatched with what the paper describes.** The code is *record-level* local DP-SGD, accounted per client using the smallest client (RF:461). That choice gives the worst case among clients: per-client ε is 4.01 / 3.64 / 3.93 / 2.24. The paper instead cites McMahan et al. 2018 [18], which is *client-level* DP-FedAvg with server-side noise on clipped client updates. It describes the method as "clipping each client's gradient" (P:78) and says "per-sample gradient clipping" (P:52). None of these is what the code does.
  6. **The ε for the B=1024 run is misreported.** `dp_fedavg_lb.json` stores ε = **9.045** (I reproduced this exactly with dp_accounting: q=1024/26232, T=1000). P Table 5 prints 4.008.
  7. P:284 says batch-level clipping is "slightly overestimating the DP budget". The opposite is true: it voids the guarantee. The experiment names `dp_fedavg_eps13/eps7/eps2` and the comments at RF:19-21 and RF:666-669 refer to an older calibration and contradict the stored ε.
- **Impact:** every ε in Table 5, §3.6 and the abstract is invalid, and the "formal (ε,δ) guarantee" contribution (RF:8, P:28) does not stand.
- **Fix (recommended: record-level, which suits 4 cross-silo clients):**
  - Remove BN, or replace it with `LayerNormalization`. TF 2.10 in `ids_env` has no Keras `GroupNormalization`.
  - Compute per-example gradients with `tf.vectorized_map` inside a `@tf.function`, or use TF-Privacy 0.9.0, which is **already installed** in `ids_env`: `VectorizedDPKerasAdamOptimizer(l2_norm_clip=C, noise_multiplier=σ, num_microbatches=B)`. Do not use the non-vectorised microbatch loop, which is about B times slower.
  - Clip globally across all tensors. Sum the clipped per-example gradients, add N(0, σ²C²), and divide by the expected batch size.
  - Use Poisson sampling (each record included with probability q_k = B/n_k), or keep shuffling and state the shuffling caveat.
  - Account per client with dp_accounting RDP or PLD, using T_k = R·E·n_k/B, and report max_k ε_k.
  - State the threat model: record-level, per silo, against the server and the other clients; aggregation is post-processing.
- **Why not client-level DP:** with N=4 and full participation, q=1. I computed ε(20 rounds, δ=1e-5) = 30.1 / 12.3 / 4.2 for σ = 1 / 2 / 5, with noise σS/4 on the averaged update. Client-level DP needs hundreds of clients. Mention it only in the discussion.
- **Cost on the GTX 1650 (4 GB):**
  - The model has 23,754 trainable parameters. Per-example gradients for B=256 take 256 × 23,754 × 4 B ≈ **24 MB** of GPU memory, which is trivial. RAM is not a concern (the TON training matrix is about 16 MB).
  - Compute cost is about 2 to 5 times a normal step under `tf.function` + `vectorized_map`. One 20-round DP run would take roughly **6 to 12 min** with graph-compiled steps, or **40 to 60 min** with the current eager style.

### C3. The "FedAdam diverges" result is a configuration and implementation artefact
- **Where:** RF:269-290, RF:369 (`global_weights` = `model.weights`, which includes BN moving statistics), RF:432-433, RF:661-663 and RX:142-143.
- **What is wrong:**
  1. Server Adam with `server_lr=1e-3` and `eps=1e-8` moves every parameter by at most about 1e-3 per round, since |m̂/√v̂| ≲ 1. Over 20 rounds the global model therefore cannot move more than about 0.02 from its random initialisation. Reddi et al. tune η_s over roughly 1e-2 to 1, with τ=1e-3.
  2. Adam is also applied to the **BN moving mean and variance**, so the global BN statistics stay near (0, 1) while the clients learn real statistics.
- **Result:** the global model barely leaves initialisation (TON best 0.113, then a drift down to 0.0145). That is under-stepping, not the "second-moment instability" the paper describes (P:212).
- **Impact:** the FedAdam rows in P Table 4, the abstract, §7, §10.4 ("warrants caution for practitioners") and the conclusion are unsupported on both datasets. P:74 also says FedAdam is "evaluated on TON-IoT only" but then reports a UNSW result.
- **Fix:**
  - Aggregate the non-trainable BN statistics by plain FedAvg, or keep them local (FedBN).
  - Apply Adam only to the trainable parameters, with τ=1e-3.
  - Tune η_s over {1e-2, 3e-2, 1e-1} on the **validation** split (see M1).
  - Report the tuned result.

### C4. All UNSW-NB15 Paper 4 results are stale (v1 leaky data)
- **Where:** RF:545-589 loads `experiments/baseline/results/UNSW_NB15/pipeline.pkl`, which was regenerated on 2026-10-24 07:05 under loader v2.1. All `results_unsw/*.json` files date from 2–3 Aug and were produced with the v1 loader and pipeline.
- **What changed:**
  - `id` is now dropped (loader.py:26). Under v1 it was kept, and the v1 RFECV selected it; SHAP ranked it first.
  - Deduplication now runs after all column drops (loader.py:137-140): 175,341 rows become **105,955** (−39.6%).
  - Missing values, median, p99 and zero-variance statistics are now fitted on training rows only.
  - A cross-split guard was added (loader.py:200-221). It removes 0/0 rows on v2.1.
- **New UNSW figures:**

  | Quantity | Old (v1) | New (v2.1) |
  |---|---|---|
  | FL pool (train+val) | 149,039 | 74,167 + 15,894 = **90,061** |
  | Test set | 26,302 | **15,894** |
  | Features after MI+RFECV | 16 (including the leaky `id`) | 59 → 40 → **10** |
  | Centralised CNN F1 (Paper 1) | 0.5182 | **0.5139** (accuracy 0.7351; `baseline/results/UNSW_NB15/metrics.json`) |

- **Impact:** every UNSW number in P is stale: abstract, Table 1, Table 2, Table 3b, the UNSW columns of Tables 4 and 6, §6, §7, §9, the "consistent 3.5 pp gap across datasets" argument, and the conclusion.
- **Code compatibility:** the code needs no change. It will pick up the new 10-feature pipeline automatically. A mismatched pickle would raise an IndexError on `mi_mask_` rather than misalign silently. Still, add an assertion that `pipeline.selected_features_` is a subset of `data['feature_names']` and that `mi_mask_` has length `X.shape[1]` (RF:510-516, RF:558-563).
- **TON-IoT is not stale with respect to the loader.** The v2.1 logs (`logs/v31_ton.log`, `logs/v3_ton.log`) show the same 211,043 → 183,550 rows, the same 64 features, the same 128,484 / 27,533 / 27,533 split, and a cross-split guard that removed 0/0 rows. The TON `pipeline.pkl` (27 Jul) therefore still matches its data, and the TON numbers do not need re-running because of the pipeline change. They do need re-running because of C1–C3 and M1–M3.

---

## Major

### M1. Model selection and headline metrics use the test set; there is no validation set
- **Where:** RF:440-452 (the global model is evaluated on `X_test` every 2 rounds; `best_f1 = max(test F1)`), RF:535-537 and RF:582-583 (validation is merged into training), RX:85-101. Paper: P:118, P:122, and the "Best F1" in every table.
- **What is wrong:** every headline number is the maximum test F1 over 10 to 25 checkpoints. The convergence round is defined relative to that test peak as well.
- **Impact:** the bias is large and optimistic. From the existing JSONs (TON, 3 seeds):

  | Configuration | Best-on-test (mean ± sd) | Final-round (mean ± sd) |
  |---|---|---|
  | FedAvg α=0.5 | 0.784 ± 0.024 | **0.700 ± 0.122** |
  | FedAvg α=0.1 | 0.583 | **0.345** |
  | FedAvg IID | 0.824 | **0.788** |

  FedProx's final-round F1 for seed 42 is 0.696 against a best of 0.758. The ranking FedAvg > FedProx is not stable under final-round reporting: seed 43 FedAvg final is 0.559, while FedProx final is 0.779.
- **Fix:** keep the loader's validation split as a global validation set, or give each client a 10% local validation split and aggregate. Select the round on validation macro-F1, and report test F1 at that round **and** at the final round, or the mean of the last 3 checkpoints. As an interim step, final-round numbers can be reported straight from the existing TON JSONs (`final_metrics`) without re-running.

### M2. Seeds 43/44 do not change the partition; most comparisons are single-seed
- **Where:** RF:593-616. `iid_partition` and `dirichlet_partition` are called without `seed=` and default to `SEED=42` (RF:100, RF:106). The `tf.data` shuffle uses `seed=SEED` (RF:143, 159, 182, 225). `load_dataset(..., seed=SEED)` is at RF:503 and RF:551. RF:700 is `if seed == SEED:`, which gates SCAFFOLD and the DP large-batch run. RX has no `--seed` argument.
- **What the seed actually changes:** only the global TF seed (model initialisation, dropout, and through the op+global seed pair the shuffle order). The client partitions are identical across seeds 42/43/44. The data split is also fixed, which is acceptable but should be stated.
- **Which experiments have seeds 43/44:** only TON `fedavg_iid`, `fedavg_alpha0.5`, `fedavg_alpha0.1` and `fedprox_alpha0.5`. `run_multi_seed.py` also planned `scale_N2`, `scale_N8` and UNSW `fedavg_iid` and `fedavg_alpha0.5`; no results for those exist.
- **Impact:**
  - P:306 attributes the α=0.1 variance to "the random client partition drawn at initialisation". That is false, because the partition is fixed.
  - SCAFFOLD, FedAdam, DP, scalability, FedAvg-SGD and all of UNSW are single-seed.
  - The claim "SCAFFOLD beats FedAvg at α=0.1 by 6.6 pp" (0.6011 vs 0.5348) falls inside the FedAvg seed spread: best F1 0.535 / 0.583 / 0.632, and seed 44 beats SCAFFOLD. The "FedAvg-SGD ≈ FedAvg-Adam, Δ<0.002" claim compares one seed against a quantity with sd ≈ 0.05.
  - Table 7 reports the population sd (ddof=0). The sample sd values are 0.022 / 0.024 / 0.048 / 0.013.
- **Fix:**
  - Pass `seed` to every partition function and to `.shuffle()`.
  - Call `tf.config.experimental.enable_op_determinism()` (available in TF 2.10).
  - Use ≥3 seeds (5 for the headline α sweep and algorithm table) for **every** compared configuration, and report mean ± sample sd with paired differences.

### M3. SCAFFOLD learning rate does not match the paper; SCAFFOLD's extra communication is not counted
- **Where:** RF:701-710 does not pass `lr` to SCAFFOLD, so it uses `LR=1e-3` (RF:63, RF:222). P:82 says η=0.01. Table 4's notes column says "SGD lr=0.01" for SCAFFOLD and "SGD lr=0.001 (SCAFFOLD optimiser baseline)" for FedAvg-SGD. These contradict each other.
- **What is wrong:**
  - The control-variate maths are correct. The local correction is g − c_i + c (RF:243-245). The client update follows option II, c_i⁺ = c_i − c + (x − y_i)/(Kη) (RF:253-255). The server update is c += (1/N)·ΣΔc_i (RF:300-305), which is correct for full participation.
  - Plain SGD at 1e-3 with no momentum under-trains badly in 40 local epochs. The comparison with FedAvg-Adam is therefore uninformative, although FedAvg-SGD at the same 1e-3 is a fair matched baseline.
  - The model weights are averaged weighted by n_k (RF:296-299), while c is averaged uniformly. That is a minor variant; the paper uses one weighting for both.
  - SCAFFOLD sends control variates in both directions, which doubles the communication (RF:391 ignores this).
- **Impact:** the SCAFFOLD description is wrong, and Table 4 shows 15.32 MB for SCAFFOLD when it should be about 30.6 MB (20 rounds) or about 76.6 MB (50 rounds).
- **Fix:** choose one SGD learning rate by validation grid {1e-3, 1e-2, 1e-1} and use it for both SCAFFOLD and FedAvg-SGD. Fix the text. Count the control-variate bytes.

### M4. The centralised baseline is not comparable to the FL runs
- **Where:** P:110, P:122, P:130-136 use the Paper 1 CNN. That model was trained with **SMOTE** (config.py:35 `apply_smote=True`; UNSW is capped at 50K per class, config.py:58-60), up to 100 epochs with EarlyStopping on val_loss and ReduceLROnPlateau (trainer.py:20-26, config.py:78), on **train only**. FL uses train+val, no SMOTE, a fixed 20×2 local epochs, and test-set peak selection.
- **Impact:** the "only 3.5 pp below centralised" claim and the "consistent 3.5 pp across datasets" claim (abstract, §5, §10.1, conclusion) are confounded. The UNSW baseline has also moved to 0.5139 (C4).
- **Fix:** add a matched pooled baseline: the same CNN on the same pooled train data, no SMOTE, the same epoch budget (40 epochs, i.e. FedAvg with N=1), and the same validation-based selection. Paper 1's SMOTE number can be quoted as a secondary reference.

### M5. Preprocessing is fitted centrally on pooled data, contradicting the FL premise
- **Where:** RF:510-533 and RF:558-580. The MI filter, RFECV (with an RF estimator) and the MinMax scaler were fitted on the pooled training set (preprocessing.py:74-133). The fit uses train rows only, so the test set does not leak.
- **Impact:** in a real federation no party has the pooled raw data. P:16 and P:140 ("no raw data leaves the client") overstate the privacy story. The MinMax scaler also encodes global per-feature min and max values.
- **Fix:** at minimum, disclose that feature selection and scaling are assumed to be agreed beforehand from a public or reference sample. Better, use a client-local scaler or federated min/max, and fix the feature set on public data or on one client.

### M6. Communication cost claims are wrong
- **Where:** RF:96-97 and RF:391. Paper: P:58, P:118, P:140, P:250-256.
- **What is correct:** the computation (Σ model.weights × 4 B × 2 × N × R) is right. 95,784 B per model gives 0.766 MB/round for N=4.
- **What is wrong:**
  1. The total covers 23,946 parameters, of which 23,754 are trainable and 192 are BN statistics. P:58 calls all 23,946 "trainable".
  2. P:140 says 15.3 MB is "negligible compared to transmitting the full training dataset". The preprocessed TON training matrix is 156,017 × 25 × 4 B ≈ **15.6 MB**, and the full raw CSV is 29.9 MB. The 50-round α=0.1 runs cost 38.3 MB, which exceeds the raw CSV.
  3. SCAFFOLD's cost is not doubled (M3).
  4. Table 3a/4 rows run for 50 rounds but are shown with the 20-round cost.
- **Fix:** correct the claim, add control-variate bytes, and report cost per round and to reach a target validation F1.

### M7. DP accounting and reporting details (to fix together with C2)
- RF:461 uses `min(client sizes)`. That is acceptable as the worst case, but report it as max_k ε_k and say so.
- RF:328-333 has a fallback formula that silently returns a non-RDP bound if `dp_accounting` fails to import. It should raise instead. The fallback did not fire for the stored values: I reproduced 4.009 / 2.064 / 0.622 / 9.045 and UNSW 4.087 with RDP.
- With the v2.1 UNSW pool, the smallest α=0.5 client will have about 15k records. Same σ=1, B=256 gives ε ≈ 5.6 (about 4.1 before), so the privacy settings must be re-derived per dataset.

---

## Minor

1. **Dirichlet partition (RF:106-128).** I rebuilt it exactly from the class totals. For TON at α=0.5, every client has every class (client 2 has only 2 MITM and 16 scanning). At α=0.1 the split is close to label-disjoint: clients have 4 / 4 / 3 / 0 absent classes, client 3 holds all 15,823 scanning records, and client 2 holds 99.8% of "normal". At N=16, six clients each miss one class. No client is empty, but the empty-client fallback at RF:124-125 would draw 10 records **with replacement from the global pool**, duplicating other clients' data. With the smaller UNSW pool (worms ≈ 100 records) at N=16 it could trigger. Raise an error instead, or use a minimum-size re-draw. Publish the per-client class-count tables. The P:86 sentence about α→0 is fine.
2. **Fresh Adam each round (RF:140, 157, 179).** The optimiser state and bias correction restart every round, which is standard but undisclosed. Each round also rebuilds the `tf.data` pipeline with the same seed, so each client sees the same batch order in every round.
3. **Model rebuilt per client per round (RF:400).** `build_model` runs for every client in every round, in eager mode with no `tf.function`. This causes the memory growth behind the "cap at 30 to avoid TF OOM" comment (RF:61) and makes training about 3.8 times slower than graph mode. Keras `fit` FedAvg-SGD takes 12.6 s/round against 48 s/round for the eager loop at the same step count. Reuse one client model and wrap the train step in `@tf.function`.
4. **Paper text and number errors:**
   - Abstract: "0.8449 in 16 rounds", but §5 says "at round 4". The best is at round 16; round 4 is the convergence round.
   - "0.1306 pp" should be 13.06 pp (P:188, P:312), and "+0.0174 pp" should be 1.74 pp (P:216).
   - The UNSW DP figure 0.0120 is the final value; the best is 0.0138.
   - The UNSW FedAdam best is 0.1103, not 0.1102.
   - P:74 and P:78 say "TON-only", yet UNSW FedAdam and DP results are reported.
   - P:312 contains a stray sentence fragment.
   - The FedAdam UNSW rows say "diverges R8"; the best is at round 8 (fine), but see C3.
5. **Convergence round.** The RF:393 comment says "90% of final F1", but the code uses the peak (RF:454-456), consistent with P:86. Either way it is anchored to test F1, so it should move to validation (M1).
6. **Evaluation is only on the pooled global test set (RF:443).** For a non-IID FL paper, add per-client test metrics (on each client's local test distribution) or at least the worst-client F1.
7. **No schema assertion when loading `pipeline.pkl` (RF:510-516, 558-563).** See C4.
8. **RX lacks `--seed` (RX:126-129).** Its partitions and seeds are hard-coded to 42. The FedAvg-SGD loop uses Keras `fit`, so the batch order differs from SCAFFOLD's `tf.data` pipeline. That is a small confound.
9. **FedProx (RF:154-173) is implemented correctly.** The proximal term covers trainable variables against the round's global weights, with μ/2·‖w−w_t‖². μ=0.01 is untuned. Add μ ∈ {0.001, 0.01, 0.1} on validation, or state that μ was not tuned.
10. **Client weighting n_k/Σn (RF:260-267) matches P:66.** SMOTE is not applied in FL (`transform` with the default `apply_smote=False`), which matches P:62 and P:92. Partitioning is applied to the transformed pooled train+val set, so no cross-client SMOTE leak is possible.

---

## Re-run plan

### What must be re-run, and why

| Group | Stale because | Re-run? |
|---|---|---|
| All UNSW (13 JSONs) | C4 (v1 leaky data, 16 → 10 features, −40% rows) | Yes, all |
| All DP (TON 4 + UNSW 1) | C1 (no training), C2 (invalid mechanism and ε) | Yes, after the code fix |
| FedAdam (TON + UNSW) | C3 | Yes, after the fix and an η_s grid |
| TON FedAvg / FedProx / scale / SCAFFOLD / FedAvg-SGD | Not stale with respect to data; stale with respect to protocol (M1 test selection, M2 partition seeds, M3 SCAFFOLD lr) | Yes, under the corrected protocol with ≥3 seeds. Interim: final-round F1 can be quoted from the existing JSONs |
| Centralised baseline | M4 | New matched N=1 run per dataset and seed |

### Protocol changes to make before re-running
1. **Validation:** keep the validation split. Select the round on validation F1; report test F1 at the selected round and at the final round.
2. **Seeds:** the seed must control the partition, initialisation and shuffle. Enable op determinism. Use seeds {42, 43, 44}, with 45 and 46 added for the α sweep and the algorithm table if time allows.
3. **DP:** fix C1 and C2 (no BN or LayerNorm, vectorised per-example clipping, global clip, Poisson sampling, per-client RDP). Add the σ=0 sanity ablations.
4. **FedAdam:** FedAvg for the BN statistics, Adam on trainable parameters only, τ=1e-3, η_s grid {1e-2, 3e-2, 1e-1} on validation (seed 42), then 3 seeds of the chosen η_s.
5. **SGD learning rate:** one grid {1e-3, 1e-2, 1e-1} on validation, shared by SCAFFOLD and FedAvg-SGD. Count the SCAFFOLD control-variate bytes.
6. **Baseline:** add the matched centralised baseline (FedAvg N=1, 40 epochs, no SMOTE).
7. **UNSW α=0.1:** with the model-reuse fix, run 50 rounds as on TON instead of 30.

### Compute estimate
Timings are measured from the JSON `elapsed_sec` values and `run_fl.log` on this machine (GTX 1650, current eager code, evaluation included):

| Run type | Time per round |
|---|---|
| FedAvg | 47–59 s |
| FedProx | 61–68 s |
| SCAFFOLD | 64–66 s |
| FedAdam | 46 s |
| FedAvg-SGD (Keras `fit`) | 12.6–14.7 s |

These figures are for the 156k TON pool. UNSW v2.1 has 90k records, about 0.6 times as many, so expect about 28 s/round for FedAvg. Fixed DP is estimated at 2 to 5 times a FedAvg step.

**A. Current eager code (only the correctness fixes):**

| Block (per seed) | TON | UNSW v2.1 |
|---|---|---|
| FedAvg IID, α=1.0, α=0.5 (20 r) | 50 min | 30 min |
| FedAvg α=0.1 (50 r) | 42 min | 25 min |
| FedProx α=0.5 | 22 min | 13 min |
| FedAdam (chosen η_s) | 15 min | 9 min |
| SCAFFOLD α=0.5 (20 r) + α=0.1 (50 r) | 76 min | 45 min |
| FedAvg-SGD α=0.5 + α=0.1 | 16 min | 10 min |
| Scale N=2/8/16 | 49 min | 30 min |
| Matched centralised (N=1, 40 epochs) | 16 min | 10 min |
| DP: σ ∈ {1, 1.5, 4} + B=1024 (fixed, ~50 min each) | 200 min | 50 min (σ=1 only) |
| **Per seed** | **≈ 8.1 h** | **≈ 3.7 h** |
| **3 seeds** | **≈ 24 h** | **≈ 11 h** |

One-off runs add about 3 h: the FedAdam η_s grid (2 extra runs per dataset), the SGD-lr grid (4 runs), and the DP sanity ablations (2 runs). **Total ≈ 38 h of GPU wall-clock** (about 1.6 days, sequential).

**B. Recommended: add `@tf.function` train steps and reuse one client model (about 3.5–3.8 times faster on the evidence above; DP runs 6–12 min each):** about **10–12 h for 3 seeds on both datasets**. Moving the α sweep and the algorithm table to 5 seeds adds about 4 h. Memory is not a constraint: the data is under 20 MB, the model about 0.1 MB, and per-example gradients about 24 MB at B=256. Two processes can share the 4 GB GPU with memory growth enabled, which would roughly halve wall-clock if the CPU keeps up. Run each seed and experiment in its own process to avoid the TF memory growth seen before.

**Do first (about 1 h):**
1. Fix C1 and run the σ=0/no-clip sanity check against `fedavg_alpha0.5`.
2. Fix C3 and run a single FedAdam η_s probe.
3. Re-run TON `fedavg_iid` seed 42 under the new protocol to confirm continuity with the old histories.
