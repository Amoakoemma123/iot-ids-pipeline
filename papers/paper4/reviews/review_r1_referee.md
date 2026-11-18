# Referee Report — Round 1

**Manuscript:** "Federated Lightweight CNN-Based Intrusion Detection for Privacy-Preserving Collaborative IoT Security" (Paper 4 of the IDS/IoT series)
**Venue type assumed:** mid-tier Q1 security / IoT journal (e.g., *Computers & Security*, *IEEE Internet of Things Journal*)
**Reviewer expertise:** federated learning, differential privacy, ML-based intrusion detection
**Date:** 2026-10-25
**Material examined:** manuscript text (`Paper4_v1_text.md`), released code (`run_federated.py`, `run_extra_experiments.py`, `src/data/loader.py`, `src/data/preprocessing.py`), cached results (`results/`, `results/seed_43`, `results/seed_44`, `results_unsw/`) and `run_fl.log`. I did not re-run anything. Where I say "the code does X", I mean the released code as it stands. If the reported runs came from a different revision, the authors should say so.

**Recommendation:** Reject in current form (resubmission encouraged after the core experiments are redone).
**Score:** 3 / 10

---

## 1. Summary of the submission

The paper trains a ~24k-parameter 1-D CNN (taken from the authors' Paper 1) with simulated federated learning on TON-IoT (25 features) and UNSW-NB15 (16 features), both treated as 10-class problems. It uses N = 4 clients with full participation, 2 local epochs, Adam with lr = 1e-3, B = 256, and 20 rounds (50 or 30 for α = 0.1). The experiments are:

1. a centralised-vs-FedAvg-IID comparison;
2. a Dirichlet label-skew sweep (IID, α = 1.0, 0.5, 0.1);
3. a comparison of FedAvg, FedProx (μ = 0.01), FedAdam (server lr = 1e-3), SCAFFOLD and a FedAvg-SGD control;
4. DP-FedAvg at three noise multipliers plus a B = 1024 variant;
5. client scaling (N = 2, 4, 8, 16);
6. a three-seed repeat of four TON-IoT configurations.

The headline claims are:
- FedAvg-IID is ~3.5 pp macro-F1 below centralised training on both datasets;
- FedAvg "fundamentally cannot converge" at α = 0.1;
- SCAFFOLD gives "genuine variance-reduction benefit" beyond the optimiser change;
- FedAdam diverges;
- DP-FedAvg collapses utility at every ε tested;
- F1 falls as N grows;
- communication is "negligible".

## 2. Strengths

- The questions are relevant to practitioners: how much non-IID label skew costs a small IDS model, whether standard drift remedies help, and whether DP-SGD is usable at all for imbalanced multi-class traffic.
- The authors correctly saw that SCAFFOLD (SGD) vs FedAvg (Adam) is confounded, and added a FedAvg-SGD control. Few applied FL-IDS papers do this.
- Code, cached per-round histories and client sizes are released. This made it possible to check the claims below, and the authors deserve credit for it.
- Macro-F1 is the right primary metric for a 10-class imbalanced IDS task. Reporting both best and final F1 in Table 3 is a good habit, although it is not applied consistently.
- The model is small enough that the cost and latency arguments for edge deployment are plausible.

## 3. Major concerns

### M1. All UNSW-NB15 results come from a leaky, non-deduplicated pipeline and must be withdrawn or redone

The shared data loader has since been corrected (see the comment in `src/data/loader.py`, `_DROP_COLS`). It confirms that the UNSW-NB15 sequential row index `id` was kept as a feature. That column correlates with the label through the file's ordering: RFECV kept it, and SHAP in a sibling paper ranked it first. Because `id` was unique per row, it also disabled deduplication: about 38.6% of records were duplicates.

Table 1 reflects this. It reports UNSW "Total samples (after dedup) = 175,341", which is exactly the size of the official UNSW-NB15 training partition, so no deduplication took place.

Every UNSW number in the paper is therefore unreliable: Tables 2, 3b, 4, 6, the UNSW DP and FedAdam runs, and every "confirmed across datasets" sentence. Duplicates shared between training and test inflate scores. The `id` feature lets the model shortcut through file order. Under Dirichlet partitioning it also interacts with the partition, because class blocks map to `id` ranges.

The cross-dataset "consistency" arguments (the "near-identical ~3.5 pp gap", "confirms cross-dataset generality", "FedAdam dataset-independent", the "DP UNSW collapse") cannot stand until the UNSW runs are repeated on the corrected pipeline.

### M2. Model selection on the test set and best-of-run reporting inflate every headline number

- `run_fl` evaluates the global model on the **test set** every 2 rounds and reports `best_f1 = max(history)`.
- Train and validation are merged for FL (`X_train = vstack([X_tr, X_val])`), so no validation data is left for round selection.
- The best checkpoint is therefore chosen on test data.

The inflation is large. Final-round and best-round test F1 from the authors' own multi-seed files:

| Config (TON-IoT) | Best F1, 3 seeds (paper) | Final-round F1, 3 seeds |
|---|---|---|
| FedAvg IID | 0.8244 ± 0.022 | **0.788 ± 0.060** |
| FedAvg α=0.5 | 0.7839 ± 0.024 | **0.700 ± 0.122** (seed 43: 0.559) |
| FedAvg α=0.1 | 0.5831 ± 0.048 | **0.345 ± 0.119** |
| FedProx α=0.5 | 0.7635 ± 0.013 | **0.743 ± 0.043** |

(Sample standard deviations. The paper's ±0.018 / 0.020 / 0.040 / 0.010 are population standard deviations, which understate spread at n = 3.)

Two qualitative claims fail once selection is removed:

- **"FedProx consistently underperforms FedAvg across all three seeds" is false.** On seed 43, FedProx (0.7786) beats FedAvg (0.7597) even on best-F1. On final-round F1, FedProx has the higher mean (0.743 vs 0.700) and much lower variance.
- **The "~3.5 pp IID gap" depends on a single seed and test-set selection.** Across the three seeds the TON gap is 3.5 to 7.9 pp on best-F1 and up to 16 pp on final-round F1.

The text also misreports the IID best round. §5 says "best F1 = 0.8449 at round 4", but the JSON gives round 16. Round 4 is the "convergence round", which is itself defined relative to the test-selected peak.

### M3. "Three seeds" do not vary the partition, and the runs are not reproducible

- `iid_partition` and `dirichlet_partition` take `seed=SEED` (= 42) as a default argument, and `main(seed=...)` never passes the seed through. `load_dataset(..., seed=SEED)` and the tf.data shuffle seed are also fixed.
- The released `client_sizes` are identical across seeds 42, 43 and 44 (for example α=0.5: [26232, 31041, 27183, 71561] in all three).
- The "multi-seed validation" therefore varies only weight initialisation and GPU nondeterminism. With N = 4 clients and a single Dirichlet draw, the partition is the dominant source of variance, and it is never resampled.
- §10.6 says the α=0.1 variance "is sensitive to the random client partition drawn at initialisation". That is factually wrong, because the partition was the same in all three runs.

Runs are not reproducible even at a fixed seed:
- `run_fl.log` contains a 20-round FedAvg α=0.1 run (same seed, same partition) that peaked at F1 = 0.5648 at round 18.
- The 50-round run reported in the paper peaks at 0.5348 at round 14, and its first 20 rounds differ from the logged run (round 18: 0.453 vs 0.565).
- The statement "All experiments are seeded (seed=42) for reproducibility" (§4.2) is therefore not accurate.

Single-seed configurations make up most of the paper's claims: all of SCAFFOLD, FedAvg-SGD, FedAdam, DP, scaling, α = 1.0, and every UNSW result. The FedAvg α=0.1 best-F1 seed spread alone (0.535 to 0.632) is larger than the claimed SCAFFOLD gain (+6.6 pp, single seed: 0.601). Seed 44 of plain FedAvg (0.632) beats SCAFFOLD.

### M4. The DP-FedAvg experiment provides no formal guarantee, and its utility results look like an implementation failure

**(a) The mechanism is not DP-SGD.** `local_train_dp` computes the gradient of the **batch-mean** loss, clips it, and adds N(0, (σC/B)²) noise. A per-sample gradient function (`sample_grad`) is defined but never called.
- With batch-level clipping, one record can change the clipped batch gradient by up to the full clip norm, so its sensitivity is about C, not C/B. The effective noise multiplier is σ/B (≈ 0.004 at σ = 1, B = 256), and the true ε is unbounded for practical purposes.
- Clipping is applied **per tensor** (14 trainable tensors, each clipped to C = 1), so the total L2 sensitivity is up to √14·C ≈ 3.7C, not C.
- The accountant assumes Poisson subsampling with q = B/n_min, but the sampler is shuffle-and-batch with fixed batch size.
- **BatchNorm breaks the per-example sensitivity argument**, because each example's output depends on the rest of the batch. Its running mean and variance, which are deterministic functions of the raw client data, sit in `model.weights` and are transmitted and averaged **without any noise**.

The limitation in §10.5 ("batch-level clipping … slightly overestimating the DP budget") is wrong both in direction and in size: there is no valid (ε, δ) guarantee. The ε values in Table 5 (4.0 / 2.1 / 0.6) should be removed. In addition, the B = 1024 row reuses ε = 4.008, but the cached JSON gives **ε = 9.04** for that run.

**(b) The utility "collapse" cannot be a privacy–utility tradeoff, given how little noise is actually added.**
- Per-coordinate noise of 0.0039 (σ = 1) or 0.0156 (σ = 4) on per-tensor-clipped batch gradients, followed by Adam, should behave much like FedAvg with mild gradient noise.
- Instead, the σ = 1 run has F1 = 0.0063 and accuracy = 1.5%, **identical at every evaluation from round 4 to round 20**. That is far below a majority-class predictor (about 23% accuracy for "normal") and below uniform random guessing.
- This pattern points to a numerical failure: NaN or Inf weights, saturated BatchNorm statistics, or Adam amplifying noise on near-zero-gradient coordinates. It does not point to an information-theoretic tradeoff.
- The paper's conclusion that "gradient clipping — not noise magnitude — is the primary cause" was never tested: no clip-only (σ = 0) run exists. The B = 1024 run changes q, the number of steps and ε together, so it isolates nothing.
- The broad claim that DP collapse is "a fundamental property of the noise-clipping interaction with multi-class IoT traffic" is unsupported. It also contradicts a large DP-SGD literature showing usable accuracy at ε ≈ 1–8 for small tabular models, given proper per-example clipping, GroupNorm instead of BN, and tuned clip norm, batch size and learning rate (see Ponomareva et al., "How to DP-fy ML", JAIR 2023).

**(c) The threat model is not stated.** The paper never says who the adversary is or what the privacy unit is: a record, a flow, a device, or a client. With N = 4 silos, client-level DP (McMahan et al., ICLR 2018, cited as [18]) is meaningless, and record-level local DP-SGD is the only sensible notion. The paper also equates "raw data never leaves the client" with a privacy guarantee (§1, §5), even though gradient-inversion and membership-inference attacks on shared updates are well documented. The unnoised BatchNorm statistics are an additional direct leak.

### M5. FedAdam is misconfigured, so "divergence" is a mis-diagnosis

- `fedadam_aggregate` uses server lr η = 1e-3 and τ (eps) = 1e-8, and applies Adam to **all** `model.weights`, including BatchNorm moving mean and variance.
- With the Adam step normalised to about η per coordinate, the global model can move at most about 0.02 per parameter over 20 rounds. It essentially stays at initialisation (round-2 F1 = 0.013, accuracy = 4%).
- Reddi et al. [12] tune η over a log grid that typically ends up around 1e-2 to 1, with τ around 1e-3. Running Adam on the BN buffers is also not meaningful.
- The curve (peak 0.11, then drift down) shows an optimiser that never trained, not one that diverged.
- The conclusion "warrants caution for practitioners considering server-side adaptive optimisers" (§10.4, abstract, conclusion) is not supported. A minimal fix is to apply FedAdam to trainable weights only (average the BN buffers or use FedBN), tune η ∈ {1e-2, 3e-2, 1e-1, 3e-1, 1} with τ = 1e-3 on a validation split, and report the tuned curve.

### M6. The SCAFFOLD / FedAvg-SGD analysis does not support "genuine variance-reduction benefit"

- §3.7 and Table 4 state SCAFFOLD uses SGD lr = 0.01. In the released code, `run_fl(..., algorithm='scaffold', **kwargs)` passes no `lr`, so SCAFFOLD ran with the default lr = 1e-3. FedAvg-SGD (`run_extra_experiments.py`) also uses lr = 1e-3. Either the description is wrong, or the reported runs did not come from the released code. The authors must clarify which.
- If both used plain SGD at lr = 1e-3 with no momentum, both are badly under-tuned. FedAvg-SGD at 0.546 is 24 pp below FedAvg-Adam. A gain of SCAFFOLD over an under-tuned SGD baseline says little about variance reduction in a well-tuned regime. Each method needs its own learning-rate sweep, selected on validation data.
- **Δ < 0.002 does not mean "the confound is fully eliminated".** The statement that FedAvg-SGD ≈ FedAvg-Adam at α = 0.1 rests on one seed, while FedAvg-Adam's α=0.1 best-F1 seed spread is about ±0.05. Two single-seed numbers agreeing within 0.002 is coincidence-level evidence.
- **SCAFFOLD's own α = 0.1 run is unstable.** F1 drops to 0.073 at round 32 and then recovers, and the reported 0.601 is a single test-selected peak at round 26.
- **The UNSW SCAFFOLD α = 0.1 result is left out of the text.** It is 0.3534 vs FedAvg 0.3565, so SCAFFOLD does not help there. Only the α = 0.5 UNSW number appears in §7.
- **The implementation departs from the algorithm.** Weights are aggregated with data-size weights, but control variates use a uniform 1/N average. Karimireddy et al. use the same averaging for both, so the authors should justify this or fix it.

### M7. The centralised "upper bound" is not comparable, and the local-only lower bound is missing

- The centralised reference (0.8803 / 0.5182) comes from Paper 1 [7]. That model was trained **with SMOTE**, on the training split only, presumably with validation-based early stopping, and with a different epoch budget.
- The FL runs use no SMOTE, train+val merged, and test-selected rounds.
- The UNSW accuracy pattern shows the mismatch: FedAvg-IID has higher accuracy than "centralised" (0.829 vs 0.756) but lower macro-F1. That is the signature of SMOTE moving the decision boundary, not of a federation penalty.
- The ~3.5 pp "gap" is therefore not a measured cost of federation.

Requested baselines:
- **Pooled centralised** training with the identical recipe: no SMOTE, same optimiser, same total number of sample-passes (20 rounds × 2 epochs = 40 epochs), same selection rule.
- **Local-only** models, each client training on its own shard only, evaluated on the global test set and, ideally, on a held-out local test split per client.

Without the local-only bound the paper cannot show that federation helps any participant, which is the core question for collaborative IDS.

### M8. Partitioning, protocol and scaling design are too narrow for the claims

- **Partitioning.** The only partitions are a single Dirichlet label-skew draw per configuration, which also induces quantity skew (α=0.5: one client holds 46% of data), plus IID. There is no feature or covariate shift, no natural partition, and no repeated draws. "Hospitals vs factories" heterogeneity (§1, §10.2) is mostly feature and concept shift, which Dirichlet label skew does not model. A natural partition is feasible: by subnet or host in TON-IoT before IP removal, by device in N-BaIoT, or cross-dataset FL on a common NetFlow feature set (the NF-* datasets of Sarhan et al.; I believe NF-UQ-NIDS-v2 is the usual choice, but please verify).
- **Setting.** N = 4 with full participation is a cross-silo setting. The paper should say so and drop the cross-device "IoT devices" framing, or add client sampling (for example 10–20% of 50–100 clients) and straggler simulation.
- **BatchNorm under non-IID.** Averaging BN statistics across label-skewed clients is a known cause of FedAvg degradation (FedBN, Li et al., ICLR 2021). Part of the α=0.1 failure, and of the late-round decline (FedAvg α=0.1 falls from 0.53 to about 0.30 after round 24), may be a BN artefact rather than "fundamental" non-convergence. A FedBN or GroupNorm ablation is needed before calling the failure fundamental. No lr decay, no change to local epochs and no server learning rate were tried either.
- **Scaling.** The N-scaling study changes the per-client data size and the effective heterogeneity together: with α fixed, more clients each see fewer classes. It uses one draw per N and one seed. "Strictly monotonic" on four single-seed points is not evidence. The UNSW non-monotonicity explanation is post-hoc and rests on the leaky pipeline. The rare-class counts quoted (worms 44, shellcode 1,160) also need checking: from memory, the official UNSW-NB15 training partition has 130 Worms and 1,133 Shellcode records, so please verify against the data.
- **Centrally fitted preprocessing.** MI feature selection, RFECV and the StandardScaler were all fitted centrally on pooled data in Paper 1 and then reused by every client. In a real federation no party can do this without the data sharing FL is meant to avoid. The authors should either fit preprocessing in a federated way (federated mean and variance via secure aggregation, client-local feature selection) or state this as a deviation that favours FL.
- **Port features.** The TON-IoT feature set includes `src_port` and `dst_port`. The authors' own loader flags these as likely label proxies on testbed data. A port-ablation run is needed to show the FL conclusions do not rest on a shortcut.

### M9. The communication-cost claims are wrong

§5 says 15.3 MB is "negligible compared to transmitting the full training dataset". At float32:
- **TON-IoT:** the 25-feature training pool is 156,017 × 25 × 4 B ≈ **15.6 MB**, essentially equal to the FL traffic. Per client, the raw shard is ≈ 3.9 MB and the FL traffic is ≈ 3.8 MB.
- **UNSW-NB15:** the 16-feature pool is ≈ **9.5 MB**, so FL uses about **1.6× more** bandwidth than shipping the preprocessed data. The α=0.1 runs cost 38 MB.

At this model and data scale, communication is not an argument for FL; privacy is the only argument. In addition:
- "Scales linearly with N" (§9) is true by construction of the formula, not a finding.
- The model size counts non-trainable BN buffers; 23,946 is the total, the trainable count is 23,754.
- No compression (quantisation, sparsification) is evaluated, so "communication efficiency" cannot be a listed contribution.
- The wall-clock comparison (FedProx +27%) was measured on a shared single-machine simulation with GPU nondeterminism. FedAvg α=0.5 alone ranges 961–1,045 s across seeds, so it should be dropped or replaced by FLOP counts.

### M10. Novelty and positioning are weak relative to the 2022–2026 FL-IDS literature

Benchmarking FedAvg, FedProx and SCAFFOLD under Dirichlet label skew on TON-IoT, UNSW-NB15 or CIC datasets is now routine. Lines of work the paper should engage with (citations I am confident of unless marked):

- **FL-IDS evaluation on ToN-IoT with non-IID splits and several aggregators:** Campos et al., "Evaluating Federated Learning for intrusion detection in Internet of Things: Review and challenges", *Computer Networks* 203 (2022). This is the closest prior work and is not cited.
- **Centralised vs on-device vs federated IDS:** Rahman et al., *IEEE Network* (2020).
- **Federated IDS with DP in industrial / agricultural IoT:** Friha et al., FELIDS (*J. Parallel Distrib. Comput.*, 2022) and 2DF-IDS (*Computers & Security*, 2023). The latter is directly comparable to the DP part and appears in a target venue.
- **Federated zero-day botnet detection:** Popoola et al., *IEEE IoT Journal* (2022).
- **FL benchmark datasets for IIoT IDS:** Ferrag et al., Edge-IIoTset (*IEEE Access*, 2022), which includes an FL evaluation.
- **Heterogeneous-network FL-IDS:** de Carvalho Bertoli et al., stacked-unsupervised federated IDS, *Computers & Security* (2023). I am fairly but not fully sure of the venue and year.
- **Poisoning of FL-IDS:** Nguyen T.D. et al., "Poisoning attacks on federated learning-based IoT intrusion detection system" (NDSS DISS workshop, 2020). Robust aggregation: FLAME (USENIX Security 2022), FLTrust (NDSS 2021). An IDS federation whose participants may themselves be compromised is a central threat, and the paper does not discuss it.
- **Label-skew-specific FL methods,** which are more relevant than FedProx for this setting: FedLC / logit calibration (Zhang et al., ICML 2022), FedRS (Li & Zhan, KDD 2021), MOON (Li et al., CVPR 2021), FedDyn (Acar et al., ICLR 2021), FedNova (Wang et al., NeurIPS 2020).
- **Personalised and clustered FL,** which the paper defers to future work: Ditto (Li et al., ICML 2021), FedRep (Collins et al., ICML 2021), IFCA (Ghosh et al., NeurIPS 2020).
- **Dirichlet partitioning** should be credited to Hsu, Qi & Brown (arXiv:1909.06335, 2019). [14] (Hsieh et al.) uses skewed label partitions and is not the Dirichlet reference.

Citation errors in related work:
- [15] Rey et al. (2022) studies **malware detection on N-BaIoT**, not CICIDS-2017.
- [16] DIoT is a self-learning anomaly detector. I believe it uses a GRU language model, not an LSTM; please check.
- [17] Mothukuri et al. is a general FL security and privacy survey, not an FL-IDS survey.

As written, the novelty claim ("systematic non-IID severity analysis, algorithm comparison, DP evaluation, and scalability study") is incremental. A rigorous, well-controlled negative-results paper could still be publishable, for example: "label-skew FL-IDS: what survives proper selection, seeds and BN handling". That requires the methodological fixes above; at present the negative results (FedAdam, DP) are artefacts.

## 4. Minor concerns

1. **The draft is unfinished.**
   - Figure 3 and Figure 6 captions say "Pending results shown as 0 / empty".
   - The Future-Work paragraph ends with a corrupted fragment: "5 (-0.1306 pp vs. FedAvg)…".
   - The header says "Paper 4 of 4", but the series has more papers.
   - Reference [7] is "[Authors] (2026)… [Paper 1 of this series]", an unpublished self-citation on which the whole baseline depends.
2. **Unit errors.** "0.1306 pp" should be 13.06 pp, and "+0.0174 pp" (§8) should be 1.74 pp. Use either F1 points (0.xx) or pp consistently.
3. **Internal contradictions.**
   - §3.5 says FedAdam was run on TON-IoT only, and §3.6 says the same for DP-FedAvg, but both are reported for UNSW.
   - Limitation 5 says a SGD-based FedAvg baseline "would be required", but one is in Table 4.
   - The code header lists ε ≈ 13 / 7 / 2, while the paper gives ε ≈ 4.0 / 2.1 / 0.6.
   - The abstract says DP best F1 = 0.050; the table gives 0.0497.
4. **"Expected random F1 ≈ 0.10" is incorrect** for an imbalanced 10-class macro-F1. Give the actual majority-class and stratified-random macro-F1 on this test set as reference lines.
5. **The convergence-round definition is degenerate.** "First round reaching 90% of peak" measured on the test set lets a collapsed DP run "converge at round 2". Use rounds to reach a fixed fraction of the centralised score, or AUC of the learning curve, on validation data, and evaluate every round rather than every two.
6. **Only macro-F1, precision, recall and accuracy are reported.** For an IDS, add per-class recall and F1 (which attack classes vanish under α=0.1?), confusion matrices, and binary detection rate and false-positive rate at a fixed operating point.
7. **"Under IID … each client receives very few samples of UNSW's rarest attack classes, while Dirichlet α = 1.0 concentrates these classes"** is a hypothesis presented as an explanation, and it rests on the leaky data (M1). Test it with per-class results or remove it.
8. **The FedProx μ = 0.01 "commonly used default" needs a sweep** (μ ∈ {1e-3, 1e-2, 1e-1, 1}). Note also that FedProx's main advantage is with partial participation and variable local work, neither of which is present here.
9. **References are out of order** ([24] before [23]; [22] after the Data Availability section), and [19] (Flower) is not cited in the text.
10. **Table 1** should report class counts per split after the corrected deduplication, and the dedup statement must be accurate for UNSW.
11. **"23,946 trainable parameters"** should read 23,946 total / 23,754 trainable (192 BN buffers).
12. **Abstract.** "Key results are validated across three random seeds" overstates the evidence: only 4 TON-IoT configurations, with the partition unchanged (M3).
13. **§5 attributes faster IID convergence on TON-IoT** to "more class-balanced partitions", but IID partitions are equally balanced by construction. Remove this.
14. **Data availability.** The Zenodo DOI should point to the exact code revision that produced the reported numbers (see the SCAFFOLD lr discrepancy, M6).

## 5. Requested changes

### 5.1 Minimal new experiments required for the paper to be publishable

These are the minimum. Without them, the core conclusions do not follow from the evidence.

1. **Redo UNSW-NB15 on the corrected pipeline** (no `id`, deduplication applied), or replace it with a dataset free of known leakage (for example CIC-IoT2023 or N-BaIoT). Remove all current UNSW numbers.
2. **Fix the evaluation protocol.**
   - Hold out a validation split, global or per-client local, for round and checkpoint selection.
   - Report test F1 at the validation-selected round **and** at the final round.
   - Never take the maximum over test evaluations.
3. **Fix the seeding.**
   - Pass the seed to the partitioner so that each repetition draws a new partition **and** a new initialisation.
   - Run ≥ 5 repetitions for **every** configuration that supports a claim: FedAvg IID, α ∈ {1.0, 0.5, 0.1}, every algorithm at α = 0.5 and 0.1, and every N.
   - Report mean ± 95% CI (t-interval or bootstrap) and use paired tests across repetitions for algorithm comparisons.
   - Enable deterministic ops, or document the residual nondeterminism.
4. **Add matched baselines.**
   - Pooled centralised training with the identical recipe (no SMOTE, same optimiser, same sample-pass budget, same selection rule).
   - Local-only training per client.
   - Report both alongside FL.
5. **Tune every algorithm fairly** with small validation-selected grids.
   - FedProx: μ ∈ {1e-3, 1e-2, 1e-1, 1}.
   - FedAdam: server η ∈ {1e-2, 3e-2, 1e-1, 3e-1, 1}, τ = 1e-3, trainable weights only.
   - SCAFFOLD and FedAvg-SGD: the same local-lr grid {0.01, 0.03, 0.1} (optionally with momentum), with the lr actually used stated correctly.
   - FedAvg-Adam: local lr ∈ {3e-4, 1e-3, 3e-3}.
6. **Run a normalisation ablation**: FedBN (local BN) and/or GroupNorm or LayerNorm in place of BN, at α = 0.5 and 0.1, before calling α = 0.1 non-convergence "fundamental".
7. **DP: rebuild correctly or remove.** If it is kept:
   - Use a vetted implementation with true per-example clipping (TF-Privacy's DP Keras optimizers with `num_microbatches = batch_size`, or a JAX/PyTorch equivalent).
   - Replace BatchNorm with GroupNorm.
   - Make the accountant match the sampler (Poisson sampling, or a shuffling-aware bound).
   - State the privacy unit (record-level, local to each silo) and the threat model.
   - Include sanity runs: C = ∞ with σ = 0 through the DP code path must reproduce FedAvg, and a clip-only run (σ = 0) separates clipping from noise.
   - Report the utility-ε curve at ε ∈ {1, 3, 8, ∞} with tuned C, B and lr.

   If this is out of scope, remove DP and the "privacy-utility" contribution entirely, and do not claim that DP is infeasible for IoT IDS.

### 5.2 Strongly recommended (would substantially raise the paper's value)

8. At least one **non-Dirichlet partition**: a natural per-host or per-subnet split, or cross-dataset FL on a common NetFlow feature set. Also 3–5 Dirichlet draws per α.
9. **Decouple the scaling study.** Hold per-client size fixed while varying N (subsample), and/or vary N under IID, so data-per-client and heterogeneity effects are separated. Add a partial-participation setting (for example 50 clients, 10–20% sampled per round).
10. **Port-ablation** (drop `src_port` and `dst_port`) on TON-IoT, to rule out shortcut learning.
11. **Per-class results** under each α, plus detection rate and false-positive rate.
12. One **robustness experiment** relevant to the security venue: a label-flipping or backdoor client against FedAvg vs a robust aggregator (coordinate-wise median, trimmed mean, or FLTrust). If omitted, discuss it explicitly as a limitation.
13. At least one **label-skew-specific method** (FedLC / logit adjustment, or FedRS) and one **personalised baseline** (FedBN, Ditto or FedRep), because these are the natural remedies for the regime where the paper reports failure.

### 5.3 Text changes required regardless

- **Delete or correct:**
  - all DP ε claims and the "slightly overestimating" limitation;
  - the "FedAdam diverges / warrants caution" statements;
  - "the optimiser confound is fully eliminated";
  - "FedProx consistently underperforms across all seeds";
  - "communication negligible compared to transmitting the dataset";
  - "the final model is sensitive to the random client partition drawn at initialisation";
  - "FedAvg fundamentally cannot converge regardless of training duration".
- **Reframe the setting as cross-silo** (N ≤ 16, full participation). Add a threat-model paragraph that covers update leakage and BN-statistic leakage.
- **State that preprocessing (MI, RFECV, scaling) was fitted centrally,** and discuss the implication or federate it.
- **Repair the draft:** pending-figure captions, the corrupted conclusion fragment, pp units, reference ordering and accuracy ([14], [15], [16], [17]), the placeholder [7], the parameter count, the B = 1024 ε, and the IID best round.
- **Expand related work** with the 2020–2025 FL-IDS and label-skew FL lines in M10, and position the contribution explicitly against Campos et al. (2022) and Friha et al. (2023).

## 6. Do the conclusions follow from the evidence?

| Claim | Supported? | Why |
|---|---|---|
| FL-IDS ~3.5 pp below centralised on both datasets | No | UNSW leaky; non-matched centralised baseline; single seed; test-selected; seed spread 3.5–7.9 pp |
| Non-IID degrades F1, severely at α=0.1 | Qualitatively plausible (TON) | Direction is credible, but magnitudes are test-selected and come from one partition draw; BN confound untested |
| FedAvg "fundamentally" cannot converge at α=0.1 | No | No lr decay, no local-epoch/BN ablation; late-round decline suggests a fixable instability |
| FedProx no better than FedAvg | Not established | Reverses on seed 43 and on final-round means; μ untuned |
| SCAFFOLD gives genuine variance-reduction benefit | No | Single seed, inside FedAvg's seed spread; lr description inconsistent with code; under-tuned SGD controls; UNSW α=0.1 counter-result omitted |
| FedAdam diverges | No | Server lr 1e-3 with τ=1e-8 on all weights including BN buffers: the model never trains |
| DP-FedAvg collapses at all ε | No | Mechanism not DP; ε invalid; output pattern indicates numerical failure |
| F1 declines with N | Weak | One seed and one draw per N; confounded design |
| Communication efficient | No | FL traffic ≈ raw data (TON) or > raw data (UNSW) |

## 7. Overall assessment

The topic is relevant and the code release is commendable, but the paper is not yet a scientific contribution. Every cross-dataset claim rests on a UNSW-NB15 pipeline now known to leak. The TON-IoT results are selected on the test set, and the "multi-seed" validation never resamples the partition. Several headline findings are artefacts of implementation or configuration: FedAdam "divergence" (server lr 1e-3 applied to BN buffers), DP "collapse" (a non-DP mechanism whose output is frozen far below chance), and SCAFFOLD superiority (single seed, lr mismatch with the text). The communication argument is numerically wrong. Novelty over existing FL-IDS benchmarks (Campos et al. 2022; Friha et al. 2023) is modest even if everything were correct.

The minimal experiment list in §5.1 is achievable with the existing small model; each run takes about 15–50 minutes on the authors' hardware. Completing it would turn the paper into a credible, if incremental, controlled study of label-skew FL for IoT IDS. I would be willing to review a resubmission.

**Recommendation: Reject (encourage resubmission). Score: 3/10.**
