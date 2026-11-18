# Paper 4 v2 (Federated CNN-IDS): methods and code audit, round 2

Date: 2026-10-25. Scope: `Papers pipeline/experiments/paper4_federated/run_fl_v2.py` (RF2, HEAD 857a346), shared `src/data/loader.py`, `src/data/preprocessing.py`, `experiments/baseline/config.py`, results in `results_v2/` (excluding `_bn_superseded`), `Revision_v2/Paper4/compute_values.py` (CV), and the manuscript `Revision_v2/Paper4/Manuscript_filled.md` (P, Sections III and IV).

How it was checked:
- I read all of the code, plus the git history of RF2 (bbbfca9, then 401f938, then 857a346). The TON DP runs were produced under bbbfca9. The only change since then is the default normalisation, and DP forces GroupNorm in both versions, so those runs are valid for the current code.
- I parsed all 259 result JSONs with small Python scripts.
- I recomputed ε with `dp_accounting` (RDP and PLD) from `ids_env`. I read the Keras 2.10 `OptimizerV2` source to check what `reset_optimizer` zeros.
- No file was modified, no training was run, and no model was loaded.

Severity: **critical** = a reported result or headline claim is invalid. **major** = a result is biased, or a method or guarantee is misdescribed in a way a referee would reject. **minor** = needs a fix or a disclosure.

---

## Critical

### C1. FedProx μ and FedAdam η_s were tuned at client lr 1e-3, but every reported FedProx and FedAdam run uses client lr 3e-3
- **Where:** RF2:469-483 (the `tune` group reads `tuned.json` once, at start-up, through `T = tuned(ds)`). At that point the file did not exist, so `T.get('fedavg', LR)` returned `LR = 1e-3` for FedProx (RF2:479-480) and FedAdam (RF2:481-482). The core group (RF2:485, 494-495) then uses `lr_adam = T['fedavg'] = 0.003` with the μ and η_s chosen at 1e-3.
- **Evidence:** every `tune_fedprox_*` and `tune_fedadam_*` JSON on both datasets has `cfg.lr = 0.001`. Every `fedprox_a*` and `fedadam_a*` core JSON has `cfg.lr = 0.003`. On TON, repetition 0 (same partition and initialisation) gives these validation F1 values:

  | Run | Client lr | η_s / μ | Val F1 (selected) |
  |---|---|---|---|
  | FedAvg core (tuned) | 0.003 | – | 0.806 |
  | FedAdam, tuning run | 0.001 | η_s 0.03 | **0.795** |
  | FedAdam, core run (reported) | 0.003 | η_s 0.03 | **0.727** |
  | FedProx, tuning run | 0.001 | μ 0.01 | 0.766 |
  | FedProx, core run (reported) | 0.003 | μ 0.01 | 0.802 |

  The reported FedAdam configuration is therefore 0.07 F1 worse on validation than the configuration its η_s was chosen for. Under the tuned combination, FedAdam is only about 0.01 below FedAvg on repetition 0, against the reported −0.089 [−0.114, −0.063], p < 0.001. On UNSW, FedAdam rep 0 gives 0.413 (tuning, lr 1e-3) against 0.386 (core, lr 3e-3); FedAvg gives 0.411.
- **Impact:** the FedAdam and FedProx rows in Tables I and II are not "tuned" as P:63 states.
  - The size and significance of every FedAdam-vs-FedAvg difference in §V-B are configuration artefacts. TON α=0.5 is the clearest case.
  - The same applies to FedProx: on TON, "FedProx differs by −0.049, p = 0.032", and on UNSW, "FedProx gives a small, significant gain under severe skew". μ was chosen for another client learning rate.
  - The affected claims are the abstract's "FedAdam … do[es] not improve on tuned FedAvg" and "FedProx helps only under severe skew on UNSW-NB15", the §VI-A sentence "tuning … removed … the apparent failure of FedAdam", and the conclusion. The direction (FedAdam is not better than FedAvg) may survive, but the reported magnitude does not.
- **Fix:**
  - Re-run the FedProx μ grid and the FedAdam η_s grid at client lr 3e-3, seed 0, α=0.5. That is 9 runs per dataset. Better, tune FedAdam jointly over client lr ∈ {1e-3, 3e-3} × η_s, as Reddi et al. tune both.
  - Then re-run the 4 core FedProx/FedAdam experiments × 5 repetitions per dataset.
  - In code, make the tune stage two-phase: tune FedAvg first, run `finalize_tuning`, then tune the rest. Alternatively, assert `c['lr'] == T['fedavg']` when the core group is built.
  - State the client lr used for FedProx and FedAdam in §III-E.

---

## Major

### M1. The DP threat-model statement overclaims: training records influence the released model outside the DP mechanism
- **Where:** P:73 ("The guarantee covers the training records of each client against anyone who sees the client's model updates or the global models, including the server"). In code:
  - RF2:77-81 loads `pipeline.pkl`, whose MinMax scaler, MI filter and RFECV were fitted on the pooled training split (preprocessing.py:74-133), which includes every client's records.
  - RF2:345-347 sends client sizes n_k to the server in clear, as FedAvg weights.
  - RF2:339 and RF2:406 make q = B/n_k depend on n_k.
  - RF2:509 sets the DP learning rate from the non-private FedAvg tuning on the same training data.
- **What is wrong:**
  - The per-feature min and max are literally values of individual training records. They are fixed into every released model through the input scaling.
  - Feature selection was fitted on all records.
  - Under the add/remove adjacency the accountant uses, n_k differs between neighbouring datasets, and the server sees it exactly.
  - Hyperparameters chosen on the same data also leak, though only a little (Papernot & Steinke, ICLR 2022).
  - The DP-SGD steps themselves are correct (see "Verified correct"). The guarantee holds only *conditional on* these quantities being treated as public.
- **Impact:** the guarantee as stated in §IV-B and the abstract is stronger than what the code gives. §III-A already says preprocessing is fitted centrally, but it does not connect this to DP.
- **Fix:** this is text only. State that the guarantee is conditional on the preprocessing parameters, the client sizes and the hyperparameters being public, that is, fixed independently of the protected records. In a deployment, scaling bounds would come from protocol and data-independent ranges or a public reference sample. Add one sentence to Limitations. Optionally, also state that the unit of privacy is one flow record. A host or attack campaign contributes many correlated flows, so record-level DP does not protect a device's traffic as a whole.

### M2. Runs are not reproducible from their seed, and several tuning decisions are smaller than the run-to-run noise
- **Where:** RF2:220 (`jit_compile=True` on the non-DP loop, with Dropout inside). There is no `tf.config.experimental.enable_op_determinism()` anywhere, although round 1 recommended it (r1 M2).
- **Evidence:** the tuning run with the chosen value has the same cfg, partition, initialisation and numpy batch order as core repetition 0, yet the two differ:

  | Pair (rep 0) | Round-1 val F1 | Selected val F1 | Selected test F1 |
  |---|---|---|---|
  | TON `tune_fedavg_lr0.003` vs `fedavg_a0.5` | 0.503 vs 0.453 | 0.8065 vs 0.8058 | 0.7985 vs 0.7917 |
  | TON `tune_scaffold_lr0.1` vs `scaffold_a0.5` | 0.366 vs 0.305 | 0.720 vs 0.722 | 0.716 vs 0.718 |
  | UNSW `tune_scaffold_lr0.3` vs `scaffold_a0.5` | 0.066 vs 0.066 | **0.439 vs 0.402** | 0.443 vs 0.412 |
  | UNSW `tune_fedavgsgd_lr0.03` vs `fedavgsgd_a0.5` | 0.100 vs 0.088 | 0.409 vs 0.377 | 0.408 vs 0.384 |

  The most likely cause is that stateful random ops (Dropout) under XLA do not honour TF op seeds, combined with GPU nondeterminism. I did not confirm this without running.
- **Impact:**
  1. "We release the code and results" cannot be reproduced bit-for-bit.
  2. At a fixed partition, run-to-run noise is 0.001–0.04 validation F1. That is larger than the margins behind several single-run tuning choices:
     - UNSW FedAvg lr: the whole grid spans 0.009.
     - UNSW SCAFFOLD: 0.1 vs 0.3 differ by 0.006. Lr 0.3 was chosen, and it then diverged at α=0.1 (m6).
     - UNSW FedAvg-SGD: 0.03 vs 0.3 differ by 0.0003.
     - TON FedProx: μ 0.001 / 0.01 / 0.1 lie within 0.007.
  3. The paired design stays valid, because pairs share partition, initialisation and batch order.
- **Fix:**
  - Disclose that runs are seeded but not bit-reproducible on GPU, and give the measured run-to-run spread.
  - For the tuning, average 2–3 seeds per grid value, or report which choices are within noise.
  - To make runs reproducible, call `enable_op_determinism()` and either drop `jit_compile` or use a stateless dropout seeded per step.

---

## Minor

1. **DP sanity comparator differs in more than the DP machinery (RF2:510, 516; P:73, P:152).**
   - `fedavg_gn_short` uses dropout 0.3 (`build_model` default) and fixed shuffled batches averaged by the actual batch size.
   - `dp_noclip_nonoise` uses dropout 0, Poisson batches, and the sum divided by the expected B.
   - "Matches non-private FedAvg on the same schedule (0.760 vs 0.735)" is therefore not like-for-like. On UNSW, the DP code without noise is better than FedAvg (0.437 vs 0.408).
   - The abstract's "costs about 13 macro-F1 points" is measured against the no-clip/no-noise DP run (0.760 → 0.633). Against `fedavg_gn_short` the cost is about 10.
   - Fix: state the reference and the dropout difference, or add a dropout-0 FedAvg short run.
2. **DP configuration not tuned for DP (RF2:509).** The DP runs reuse the lr tuned for non-private FedAvg (3e-3), with C=1 and B=256 fixed. The near-flat cost across ε 0.8–12 may be specific to this configuration, for example Adam amplifying noise in low-gradient coordinates. Fix: say that lr, C and B were not tuned under DP. Optionally add a small lr × C probe at one σ.
3. **Repetition 0 shares the tuning partition (RF2:306-311, 520-531).** Hyperparameters were chosen on the validation F1 of the rep-0 partition and initialisation, and rep 0 is then reported. Because of M2 the core rep-0 run is not an exact copy of the tuning run, but the selection still favours rep 0. I re-computed every paired comparison of §V-A/B without rep 0: no sign or significance conclusion changes.
   - TON: FedAvg−central −0.047 (p 0.034); FedAvg−local α 0.5 +0.094 (p < 0.001); FedProx−FedAvg α 0.5 −0.060 (p 0.019); FedAdam−FedAvg α 0.5 −0.094 (p 0.002).
   - UNSW: FedProx−FedAvg α 0.1 +0.041 (p 0.020); FedAvg−local α 0.5 +0.047 (p 0.152).
   - Fix: one sentence of disclosure, with this robustness check.
4. **Multiple comparisons (CV:126-141).** There are 18 paired t-tests per dataset with n = 5 and no correction. TON FedProx α=0.5 (p = 0.032) and UNSW FedProx α=0.1 (p = 0.027) would not survive Holm. The abstract's "FedProx helps only under severe skew on UNSW-NB15" rests on the latter, and C1 also affects it. Fix: report Holm-adjusted p, or soften the claim to "a small gain (p = 0.027, uncorrected)".
5. **SCAFFOLD averaging is inconsistent (RF2:347 vs RF2:362; open from r1 M3).** The model is averaged with weights n_k, while the control variate c is averaged uniformly. Karimireddy et al. use the same (uniform) weights for both. Fix: use one weighting for both, or disclose it.
6. **UNSW SCAFFOLD α=0.1, lr 0.3: 2 of 5 repetitions reach NaN (RF2:366-367).**
   - Rep 1 is non-finite from round 2, and its selected round is round 1 (val 0.066).
   - Rep 2 is non-finite from round 18.
   - Their final test F1 is 0.002. The mean 0.185 ± 0.116 mixes untrained, diverged models.
   - P:129 says "fails". Fix: state "2 of 5 repetitions diverged to non-finite weights".
7. **The last partial batch of each epoch is dropped (RF2:246-251).** At most 255 records per client-epoch are dropped, re-drawn each epoch. That is about 0.6% for TON N=4 clients, but 17% for the smallest UNSW α=0.1 local-only client (923 → 768) and 22% for the smallest UNSW N=16 client (655 → 512). It is undisclosed. SCAFFOLD's K counts the executed steps, so the control-variate maths is unaffected. Fix: disclose, or pass the partial batch through `run_batches`.
8. **The centralised and local-only references are not exactly "the same optimiser" (RF2:416-432; P:59).**
   - `run_centralised` never calls `reset_optimizer`, so Adam state persists across "rounds", whereas FedAvg resets it every round.
   - Its lr is FedAvg's tuned lr and is not tuned for centralised training.
   - Local-only clients select their round on the **pooled** validation set, which no single client holds. FL selection uses the same pooled set, so a server would need a central validation set.
   - Fix: disclose all three in §III-D. The comparisons stay fair across methods.
9. **FedProx runs with local Adam (RF2:177, 190-191).** The proximal gradient μ(w − w_g) is rescaled by Adam, so μ does not have its SGD meaning. Fix: write "FedProx (local Adam)" in §IV-A.
10. **Tuned values on the grid edge (`tuned.json`).**
    - FedAvg lr 0.003 is the upper edge of {3e-4, 1e-3, 3e-3} on both datasets.
    - UNSW SCAFFOLD lr 0.3 is the upper edge of its grid.
    - The centralised model and DP inherit lr 0.003.
    - Fix: extend the grid by one value (1e-2), or disclose.
11. **DP implementation details (RF2:262-265; RF2:306-308).**
    - A Poisson draw that comes out empty skips the step instead of taking a noise-only step. That is a (negligible) departure from the analysed mechanism: P(empty) = (1−q)^n ≈ e^−256. The accountant still counts the step, so ε stays conservative.
    - Sampling and noise come from PRNGs seeded with the repetition index. That is fine for experiments, but a deployment needs a cryptographically secure RNG.
    - Fix: one sentence in §IV-B. Optionally take the noise-only step.
12. **Table captions (CV:34-43, 189-191).** "±" is a t-based 95% CI half-width, not a standard deviation, and the ε column is the mean (range) over repetitions of the per-run maximum over clients. Fix: say both in the captions. P:133 already mixes "± 0.380" with "standard deviation 0.153", which confuses readers.
13. **Still open from round 1:**
    - No per-client test or worst-client metrics (r1 minor 6). Per-class F1 is stored; per-client is not.
    - No schema assertion when loading `pipeline.pkl` (r1 minor 7, RF2:77-86).
    - The empty-client fix in `dirichlet_partition` (RF2:102-107) is correct and never triggered: the minimum client size over all stored runs is 655. It is undisclosed, which is fine.

---

## Verified correct (no action)

- **DP-SGD (RF2:198-216):**
  - Per-example gradients come from `tf.vectorized_map` with a tape inside.
  - The clip is on the global L2 norm over all trainable tensors.
  - Noise N(0, σ²C²) is added to the **sum**, and the sum is divided by the expected batch size B = q·n_k.
  - Poisson inclusion uses rate q = B/n_k (RF2:263). Every client has n_k ≥ 7,720 in the DP runs, so q < 1 always.
  - The loop runs ceil(1/q) draws per epoch (RF2:262), and the accountant composes exactly R·E·ceil(1/q) `PoissonSampledDpEvent(q, GaussianDpEvent(σ))` steps with the RDP accountant at δ = 1e-5 (RF2:405-411). The sampler and the accountant agree.
  - I re-derived every stored ε exactly (24 runs). Examples: TON σ=1 rep 0 gives 2.987, and UNSW σ=2.5 rep 1 gives 1.489. A PLD accountant gives about 10% tighter values (e.g. 2.69 and 1.36); optionally report those.
  - ε is computed per client, and the maximum is reported.
- **Model in DP:** DP forces GroupNorm and dropout 0 (RF2:314-316). The TON DP JSONs say `cfg.norm='bn'` (the v2 default) but `norm='gn'` at top level. GroupNorm has no trainable/inference difference and no non-trainable weights, so `training=False` in the per-example pass has no effect. Adam on the noisy gradient is post-processing.
- **Threat model:** apart from M1, the statement is correct. By adaptive composition it holds against the server and other clients, and it does not hide participation.
- **`reset_optimizer` (RF2:238-240):** in Keras 2.10 `OptimizerV2`, `variables()` returns `_weights`, which holds only `iterations` and the slots. The hyperparameters are created through `add_weight` without being appended (optimizer_v2.py:1112-1133, 1232-1234). The reset therefore zeros the Adam moments and the bias-correction counter but not lr or β. Adam state is not shared across clients.
- **FedAvg:** aggregation is weighted by n_k over all weights.
- **FedProx:** the proximal term covers trainable variables against the round's global weights, and the anchor is set after `assign` (RF2:332-334).
- **FedAdam (RF2:348-358):**
  - The pseudo-gradient is Δ = avg − w (Reddi's sign).
  - The moments are m = 0.9m + 0.1Δ and v = 0.99v + 0.01Δ², with no bias correction, τ = 1e-3, applied to trainable weights only. Non-trainable weights are averaged (relevant only to BN).
- **SCAFFOLD:**
  - The local step is g − c_i + c.
  - Option II is c_i⁺ = c_i − c + (x − y)/(K·η), with K equal to the executed steps of each client.
  - The global update is c += mean Δc_i.
  - Local training is SGD without momentum.
  - Communication is doubled: 1,520,256 B against 760,128 B per round.
- **XLA loop (`run_fixed`):** the `input_signature` prevents retracing. XLA recompiles once per distinct (number of batches) shape, which costs performance only.
- **Protocol:**
  - The partition comes from `default_rng(1000+rep)`, the batch order from `default_rng(2000+rep)`, and the initialisation from `set_random_seed(rep)`.
  - Client sizes are identical across all algorithms and the local-only runs for a given repetition and α, and the partition changes across repetitions. The pairing is therefore valid.
  - The centralised model shares FedAvg's initialisation.
  - Selection (`select`) uses validation F1 only, taking the earliest maximum. In all 259 JSONs, the recomputed selected round equals the stored one, and `test_selected_full.f1` equals `test_f1_selected`.
- **GroupNorm (RF2:113-130):** it computes per-sample moments over (length, channels-in-group) with 8 groups (32 and 64 channels are divisible by 8) and per-channel γ/β. This is correct.
- **Sizes:** 23,754 parameters × 4 B = 95.0 kB, and 0.76 MB per round. The TON matrix is 128,484 × 25 × 4 B = 12.8 MB. All of these match P.
- **Statistics (CV):**
  - Means use t-based CIs with ddof = 1.
  - Paired t-tests run on shared repetitions; the BN rows (3 repetitions) pair only with repetitions 0–2.
  - Local-only is the unweighted mean over clients per repetition, paired by partition.
  - I spot-checked the reported paired values (−0.059, +0.080, +0.286, −0.089) and they reproduce.

---

## Round-1 findings: status

| R1 | Finding | Status in run_fl_v2.py |
|---|---|---|
| C1 | DP gradients never reached the model | **Fixed.** The loss is inside the tape, per example, and no gradients are filtered. The sanity runs train (0.760 TON). |
| C2.1 | Per-batch clip | **Fixed** (per-example). |
| C2.2 | Per-tensor clip | **Fixed** (global norm). |
| C2.3 | BN breaks per-example DP; BN statistics sent unnoised | **Fixed** (GroupNorm, no non-trainable weights). |
| C2.4 | Accountant assumes Poisson; loop shuffled | **Fixed** (Poisson sampler; steps match). |
| C2.5 | Privacy level misdescribed (client vs record) | **Fixed** in text and code (record-level, per client, max reported). See new M1 for the remaining caveats. |
| C2.6 | B=1024 ε misreported | **Superseded** (no such run; all ε re-derived exactly). |
| C2.7 | "Overestimating budget" text | **Superseded.** |
| C3 | FedAdam under-stepping; Adam on BN statistics | **Fixed** in implementation (trainable only, τ = 1e-3, η_s grid on validation). **New C1:** η_s tuned at the wrong client lr. |
| C4 | UNSW results stale | **Fixed.** All UNSW runs are new, on the v2.1 pipeline (74,167/15,894/15,894, 10 features). |
| M1 | Test-set selection | **Fixed** (validation selection; test at selected and final round). |
| M2 | Seeds did not change the partition; single-seed comparisons | **Fixed** (new partition, initialisation and batch order per repetition; 5 repetitions core, 3 extra; ddof = 1; paired tests). **Partly open:** no op determinism, runs not reproducible (new M2). |
| M3 | SCAFFOLD lr; communication not doubled | **Fixed** (lr grid; doubled bytes). Weighted-vs-uniform averaging still **open** (minor 5). |
| M4 | Centralised baseline not comparable | **Fixed** (same model, no SMOTE, 100 epochs, validation selection). Small residual differences in minor 8. |
| M5 | Preprocessing fitted centrally | **Disclosed** (P §III-A, Limitations). Not connected to the DP claim (new M1). |
| M6 | Communication claims | **Fixed** (95.0 kB, 12.8 MB comparison, SCAFFOLD ×2, per-round MB). |
| M7 | DP accounting details | **Fixed** (max_k ε_k, no silent fallback, per-dataset ε). |
| m1 | Empty-client fallback duplicated data | **Fixed** (moves 1% of the largest client; never triggered). Class counts stored per client. |
| m2 | Fresh Adam each round; same batch order every round | **Fixed and disclosed** (P §IV-A); the batch order now changes every round. |
| m3 | Model rebuilt per client, eager | **Fixed** (one reused client model, compiled loops). |
| m4 | Text and number errors | **Superseded** (new manuscript). |
| m5 | Convergence round anchored to test | **Fixed** (selected round on validation). |
| m6 | No per-client or worst-client metrics | **Open.** |
| m7 | No `pipeline.pkl` schema assertion | **Open.** |
| m8 | RX had no seed; Keras-fit FedAvg-SGD | **Fixed** (single script; FedAvg-SGD shares SCAFFOLD's loop). |
| m9 | FedProx μ untuned | **Fixed** in principle (grid), but tuned at client lr 1e-3 (**new C1**). |
| m10 | Client weighting / SMOTE | **Still correct.** |

## Re-run needed

- **C1:** the FedProx μ and FedAdam η_s grids at client lr 3e-3 (9 runs per dataset, seed 0). Then `fedprox_a{0.5,0.1}` and `fedadam_a{0.5,0.1}` × 5 repetitions per dataset (40 runs). Then rebuild `values.json`.
- Everything else is text or disclosure only, plus the optional determinism and seed-averaged tuning (M2).
