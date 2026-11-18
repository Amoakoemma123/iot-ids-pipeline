# Referee Report, Round 2 (revised manuscript)

**Manuscript:** "Federated Intrusion Detection for IoT Networks Revisited: Data Heterogeneity, Tuned Baselines and Record-Level Differential Privacy"
**Venue:** *IEEE Internet of Things Journal*
**Reviewer expertise:** federated learning, differential privacy, ML-based intrusion detection
**Date:** 2026-10-25
**Material examined:** `Manuscript_filled.md`, `Manuscript_v2.pdf` (captions and layout), the three figures in `figures/`, the four round-1 reports, `Paper4_v1_text.md`, the code `run_fl_v2.py`, every result JSON in `results_v2/` (TON_IoT and UNSW_NB15, including `tuned.json` and the `tune_*` runs), and `src/data/preprocessing.py`. I did not run any training. I recomputed every mean, 95% t-interval and paired t-test in Sections V-A to V-C from the per-seed JSONs (script in my scratch space). **All the numbers I checked match the manuscript to three decimals.** The problems below are about design, interpretation and disclosure, not arithmetic.

**Recommendation:** Major revision
**Score:** 5 / 10 (round 1: 3 / 10)

---

## Part 1. Status of the round-1 major points

| # | Round-1 point | Status | One-line justification |
|---|---|---|---|
| M1 | UNSW-NB15 leaky `id` column and no deduplication | **Resolved** | Rebuilt on the corrected pipeline: 74,167/15,894/15,894 records (about 106k in total against 175k before), and `id` is dropped. |
| M2 | Round selected on the test set; best-of-run reporting | **Resolved** (one caveat) | Validation-selected and final-round test F1 are both reported, and `select()` reads only `val_f1`. Caveat: the validation set is a pooled, server-held split (see major concern 5). |
| M3 | "Seeds" did not resample the partition; single-seed claims | **Mostly resolved** | `prng = default_rng(1000+rep)` redraws the partition, and client sizes differ across repetitions (checked). Core results use 5 repetitions; scaling, BN and DP use only 3. GPU/XLA nondeterminism is not documented. |
| M4 | DP mechanism was not DP-SGD; ε invalid; no threat model | **Largely resolved** | The code now does per-example global-norm clipping, Poisson sampling with q = B/n_k, a fixed expected-batch denominator, an RDP accountant matched to the sampler, a GroupNorm model without dropout, two sanity runs and a stated threat model. Remaining gaps: the preprocessing (min-max scaler, MI/RFE) was fitted on the private records, DP hyperparameters were tuned without privacy, and the schedule is short (major concern 4). |
| M5 | FedAdam misconfigured | **Partly resolved** | τ = 10⁻³, trainable weights only, and the server learning rate is tuned. **But the server learning rate (and FedProx's μ) were tuned with client lr = 0.001, while the reported runs use client lr = 0.003** (major concern 1). |
| M6 | SCAFFOLD lr mismatch; "genuine variance reduction" | **Mostly resolved** | The lr is tuned, the FedAvg-SGD control is tuned and the claim is withdrawn. Still open: control variates are averaged uniformly while weights are averaged by client size (undisclosed); non-finite weights appear in 2 of 5 UNSW α = 0.1 runs and are not reported; the text still talks of SCAFFOLD's "advantage" (major concern 7). |
| M7 | Non-matched centralised baseline; no local-only bound | **Resolved** (one caveat) | The centralised model now uses the same recipe, and local-only models are added. Caveat: local models are evaluated only on the global test set, and per-client results are not shown (major concern 6). |
| M8 | Partitions, protocol and scaling too narrow; BN; preprocessing; ports | **Partly resolved** | Resolved: the GroupNorm/BatchNorm ablation, the cross-silo limitation is stated, centralised preprocessing is acknowledged. Unresolved: no natural partition, no partial participation, the scaling study is still confounded, no port ablation (`src_port`/`dst_port` are still features and rank first in the authors' own SHAP file), and no FedBN. |
| M9 | Communication claims wrong | **Resolved** | 95.0 kB model, 0.76 MB/round, 38.0 MB over 50 rounds against 12.8 MB of data (all verified). The paper now says correctly that FL does not save bandwidth. |
| M10 | Novelty and positioning weak | **Partly resolved** | Campos et al., Friha et al. (2DF-IDS), Hsu et al. and NIID-Bench are now cited. Label-skew-specific methods, personalised FL, FedBN, DP disparate impact and poisoning work are still not engaged. Novelty remains modest (Part 2, §4.1). |
| Minor | Units, unfinished text, majority baseline, per-class results, μ sweep, references, DOI | **Mostly resolved** | Units are fixed and the text is clean. Per-class results and benign FPR are added (per-class only for DP). Still missing: majority/stratified-random macro-F1 reference lines, and the repository DOI ("[repository DOI to be added]"). |
| Code C1–C3 | DP never updated the weights; invalid ε; FedAdam artefact | **C1, C2 resolved; C3 partly** | See M4 and M5. SCAFFOLD's double communication is now counted. |

---

## Part 2. Fresh report on the revised manuscript

### 1. Summary

The paper re-evaluates FL for a 23.8k-parameter 1-D CNN IDS on TON-IoT (25 features) and deduplicated UNSW-NB15 (10 features). The setup is 4 Dirichlet-partitioned clients (α ∈ {∞, 1.0, 0.5, 0.1}) with full participation, 50 rounds × 2 local epochs, and GroupNorm. The protocol redraws the partition in each of 5 repetitions, selects the round on validation data, tunes each algorithm once at α = 0.5, and adds a same-recipe centralised model and local-only models. Algorithms are compared with paired t-tests. Secondary studies cover N ∈ {2, 8, 16}, BatchNorm vs GroupNorm, and record-level DP-SGD at four noise levels (20 rounds × 1 epoch, 3 repetitions).

Headline claims:
- FedAvg costs 4–19 macro-F1 points against pooled training and gains 8–29 points over local training on TON-IoT.
- FedProx, FedAdam and SCAFFOLD do not beat tuned FedAvg.
- BatchNorm makes the evaluated model unstable.
- DP-SGD at ε ≈ 0.8–12 costs about 10–13 points, concentrated in minority classes.

### 2. Strengths

1. **This is a genuine rebuild, not a patch.** Every round-1 critical defect was addressed in code, and I could reproduce every reported statistic from the released JSONs. The Introduction states openly that the earlier version was flawed.
2. **Protocol hygiene is now above the FL-IDS norm:**
   - selected-round and final-round scores reported side by side;
   - partitions redrawn per repetition;
   - paired comparisons with CIs rather than bare means;
   - the FedAvg-SGD control separating SCAFFOLD's drift correction from its optimiser.
3. **The DP implementation is correct in its core mechanics:**
   - `tf.vectorized_map` per-example gradients;
   - `tf.linalg.global_norm` clipping;
   - noise σC on the sum, divided by the expected batch size, which is the right estimator under Poisson sampling;
   - accountant steps `rounds × local_epochs × ceil(1/q)` matching the sampler loop;
   - no batch-dependent statistics.

   The two sanity runs (no clip/no noise; clip only) are exactly the checks I asked for.
4. **The bounding references (centralised and local-only) turn "FL works" into a measurable trade-off.** The finding that the gap to pooling and the gain over isolation both grow with skew (Fig. 1) is the clearest result in the paper.
5. **The per-class DP analysis (Fig. 3)** is informative for practitioners: MITM 0.24 → 0.13 → 0.00 and password 0.54 → 0.15 on TON-IoT.
6. **The communication paragraph is now correct and candid.**

### 3. Major concerns

**MC1. The FedProx and FedAdam tuning was done at a different client learning rate from the reported runs. The headline "drift-reducing algorithms do not improve on tuned FedAvg" is therefore confounded.**

- In `experiments(..., 'tune')`, FedProx and FedAdam use `c['lr'] = T.get('fedavg', LR)`. Because all tuning runs were launched before `tuned.json` existed, they ran at **lr = 0.001**. This is confirmed by the stored `cfg` of `tune_fedprox_mu0.01/seed0.json` and `tune_fedadam_server_lr0.03/seed0.json` on both datasets.
- The core runs (`fedprox_a0.5`, `fedadam_a0.5`) use **lr = 0.003**, FedAvg's own tuned value.
- So FedAvg is evaluated at its jointly tuned optimum, while μ and the server lr are tuned for a different local optimiser setting.
- This matters for the conclusion. At the matched client lr of the tuning run (0.001, α = 0.5, seed 0), **FedAdam (val F1 0.795) beat FedAvg (0.781)** on TON-IoT, and FedProx (0.766) trailed it only slightly. In the core table FedAdam is −0.089 [−0.114, −0.063].
- Section III-E presents the grid as if it were a fair per-algorithm tuning ("FedAdam's server learning rate ({0.01, …, 1}; 0.03 and 0.03)") and never discloses the mismatch.
- **Required:** tune jointly (client lr × μ, client lr × server lr) or at least re-tune μ and η_s at client lr 0.003. Then re-run the 5-repetition core comparison and rewrite §V-B, the abstract and contribution 3 accordingly.

**MC2. The tuning itself is too thin to support "tuned" claims.**

- *Edge optima.* FedAvg's selected lr is the **largest grid value (0.003) on both datasets**. SCAFFOLD's selected lr on UNSW-NB15 (0.3) is also the grid maximum, and FedProx's μ on UNSW (0.001) is the grid minimum, which effectively means "turn FedProx off". An optimum on the grid edge is not a tuned value; the grids must be extended.
- *Single repetition.* Selection uses one repetition, and the differences being chosen between are below the repetition noise:
  - UNSW FedAvg: lr 3e-4/1e-3/3e-3 → 0.410/0.415/0.419;
  - UNSW FedAvg-SGD: 0.366/0.409/0.364/0.408, non-monotone and evidently noise;
  - UNSW FedProx: 0.420/0.406/0.419/0.374.

  Choices like these are effectively random. Tuning over ≥3 repetitions (different partitions from the evaluation repetitions) would fix this.
- *α = 0.5 only.* Tuning only at α = 0.5 systematically disadvantages the algorithms built for severe heterogeneity, which is exactly where FedProx and SCAFFOLD are expected to help. The paper concedes this ("hyperparameters tuned in one heterogeneity setting need not transfer"), and the SCAFFOLD collapse on UNSW at α = 0.1 is a direct example. Either tune per α or state the comparison at α = 0.1 as untuned.
- *Leakage into the evaluation.* The tuning repetition uses the same partition seed (`1000+0`) as core repetition 0, and for FedAvg the tuning run is the core run. This is minor, but the evaluation repetitions should use partitions disjoint from tuning.

**MC3. Multiple comparisons and small n: several "significant" claims would not survive correction.**

- Sections V-A to V-C report about 26 paired t-tests on n = 5 (or n = 3) with no correction.
- Under Holm–Bonferroni, the claims at p = 0.018–0.032 would not survive:
  - "FedProx gives a small, significant gain under severe skew" on UNSW (+0.034, p = 0.027), which appears in the **abstract** ("FedProx helps only under severe skew on UNSW-NB15");
  - FedProx < FedAvg on TON at α = 0.5 (p = 0.032);
  - FedAvg − central at α = 0.5 (p = 0.018).
- With 5 partitions, the CI half-widths at α = 0.1 are ±0.10, so the study is underpowered to detect the 2–3 point differences it discusses.
- **Required:** pre-declare the primary comparisons, apply Holm (or report q-values), use exact Wilcoxon or permutation tests as a check, and soften the abstract.
- Increasing the number of partitions to 10 for α = 0.1 would cost little with this model.

**MC4. The DP evaluation: the mechanism is now right, but the scope of the guarantee and the utility conclusions need tightening.**

- **(a) Preprocessing is outside the guarantee.**
  - The min-max scaler (`MinMaxScaler().fit(X_sel)` in `preprocessing.py`) and the MI → RFECV feature selection were fitted on the pooled training records, which are the very records DP claims to protect.
  - Min-max scaling is the textbook non-private operation: a single outlier record determines the released scale parameters.
  - Calling the result "(ε, δ) for the training records of each client against anyone who sees the client's model updates" is therefore only true **conditional on the preprocessing being public or derived from public data**. This must be stated explicitly in §IV-B. Better: fix the scaler to data-independent feature ranges (or DP quantiles), and use a feature set chosen without the private data.
- **(b) Hyperparameters are tuned non-privately.** The DP runs inherit lr = 0.003 from non-private tuning on the same training data, and C = 1 is fixed. Either account for tuning (e.g., Papernot & Steinke, ICLR 2022) or state that ε excludes hyperparameter selection, as is common.
- **(c) The DP mechanism is under-tuned.** C, lr and B are not tuned for DP (Ponomareva et al., JAIR 2023, recommend tuning C and B at fixed ε). The flat utility curve (0.658 → 0.648 → 0.660 → 0.633 for ε ≈ 12 → 0.8) is consistent with the loss being dominated by something other than the noise level. One candidate is the interaction of Adam with clipped and noisy gradients (Adam normalises update magnitudes, so noise-dominated coordinates move at the full lr). Another is the 20-round budget. The text interprets the flatness as a property of the problem ("most of it is incurred at the weakest noise level"). Without a σ = 0.3 point, or a DP-SGD-with-momentum comparison, the paper cannot say which explanation holds.
- **(d) The 20-round schedule.** The non-private reference on the same schedule reaches only 0.735 (TON), against 0.840 at 50 rounds, and the sanity runs select rounds 16–20, so they are still improving at the end of the budget. The DP cost is thus measured in an under-trained regime, and the abstract's "costs about 13 macro-F1 points" is relative to that regime. I would like at least one σ at 50 × 2 (with σ raised to hold ε fixed), or an explicit statement that the cost at convergence is unknown.
- **(e) Baseline choice.** "About 13 points" uses the DP code without clipping or noise (0.760) as the baseline. Against the GroupNorm FedAvg reference on the same schedule (0.735) it is about 10 points. State the baseline in the abstract.
- **(f) Scope.** DP is studied only at α = 0.5 with N = 4. Under α = 0.1 the small clients (down to 923 records on UNSW) would have q > 0.25 and much larger ε, and the code would break for n_k < 256 (q > 1). This should be noted.
- **(g) Minor accounting points.**
  - Empty Poisson batches are skipped (`if len(b)`) but still counted as steps. With expected batch 256 this is negligible, but it should be documented.
  - The stored `cfg` of the TON DP runs records `norm: 'bn'` while the model actually built was GN (the code forces GN for `dpsgd`). Fix the metadata so readers of the released JSONs are not misled.
- **(h) Missing literature.** The per-class finding is the known *disparate impact* of DP-SGD on under-represented classes (Bagdasaryan, Poursaeed & Shmatikov, NeurIPS 2019). It should be cited and the result framed as an IDS-specific confirmation, not a new observation.
- **(i) Missing validation of the guarantee.** No empirical check is reported (e.g., a membership-inference audit on updates at σ = 0 vs σ = 2.5). This is optional for this venue, but it would substantiate the threat-model paragraph.

**MC5. The validation set is held centrally, which is at odds with the federated premise.**

- Round selection uses a stratified validation split drawn from the pooled data and evaluated by the server ("After every round the global model is evaluated on the validation and test sets").
- A real server holds no labelled pooled traffic. The realistic equivalent is federated validation: each client evaluates on its own local validation shard, and confusion matrices are aggregated, ideally with secure aggregation.
- If the clients' validation shards together equal the pooled split, the global macro-F1 is identical, so the fix may be purely descriptive. But the paper must say how selection would be done in deployment.
- The same pooled split is used to select the rounds of the **local-only** models. That gives local clients information about classes they never see, which favours them and is conservative for the "gain over local" claim, but it should be stated.

**MC6. "Each site gains from joining" is not shown; local-only results are averaged away.**

- §VI-B says "the more skewed the sites, the more each gains from joining". The tables show only the mean over clients.
- The per-client JSONs contradict "each" on several repetitions:
  - TON α = 0.5, repetition 2: the largest client alone reaches 0.865 against FedAvg's 0.863;
  - UNSW α = 0.5, repetition 0: client 0 alone reaches 0.490 against FedAvg 0.413;
  - UNSW α = 0.5, repetition 2: one client reaches 0.503 against 0.400.
- At α = 0.5 on UNSW the mean gain is not significant (+0.032, p = 0.271).
- Local models are also evaluated only on the **global** test set, which mechanically penalises a client for classes it never sees. The operator-relevant question is performance on that client's own traffic distribution.
- **Required:**
  - report min, max and size-weighted local-only results, plus the fraction of clients that gain;
  - evaluate on per-client test sets drawn with the client's label mix;
  - qualify the "each gains" sentence.

**MC7. The algorithm narrative partly describes the old draft, not this one.**

- Contribution 3 says "SCAFFOLD's apparent advantage reflects its optimiser rather than its drift correction". The conclusion says "SCAFFOLD's reported advantage is an optimiser effect". The abstract says "SCAFFOLD's advantage disappears once both use the same optimiser".
- In this study SCAFFOLD has **no** advantage. It trails Adam-FedAvg by −0.125 [−0.165, −0.085] and ties FedAvg-SGD. The "advantage" exists only in the rejected v1, which readers have never seen.
- The correct statement is: "SCAFFOLD's drift correction gives no measurable benefit over FedAvg with the same SGD optimiser, and both trail Adam-based FedAvg." Similarly, "removed the apparent advantage of SCAFFOLD and the apparent failure of FedAdam" (§VI-A) refers to the previous draft.
- More generally, the Introduction's account of "our own earlier version" ("a DP implementation in which no gradient reached the model") describes an unpublished, rejected draft. It does not belong in a journal paper; keep it in the response letter.
- **SCAFFOLD on UNSW at α = 0.1 is misreported and under-reported:**
  - With lr 0.3, 2 of 5 runs produce **non-finite weights** (`any_nonfinite = True`, seeds 1 and 2), and three runs select round 1, 2 or 10. That is numerical divergence and should be named as such.
  - With lr 0.1 the text says "one of five repetitions still collapses". In fact **two** do: seed 1 (val F1 stuck at 0.066, a constant predictor) and seed 4 (selected round 2, then stuck at about 0.12). Final-round mean is 0.187.
  - The paired test against FedAvg-SGD at lr 0.3 (−0.176 [−0.302, −0.050], p = 0.018) is not reported.
  - The explanation "control variates are unreliable … beyond any learning-rate effect" is not established. Only two learning rates were tried, both at the top of the grid. The smallest client holds 923 records, i.e., 6 local steps per round, which makes the option-II control-variate estimate (x − y)/(Kη) very noisy. Try lr ∈ {0.01, 0.03} and a lower server step before attributing the failure to the method.
- **Undisclosed SCAFFOLD deviation.** The implementation averages weights by client size but control variates uniformly (`sum(d)/len(clients)`). This was raised in round 1 and is neither changed nor disclosed.

**MC8. The BatchNorm/GroupNorm claim is overstated, and part of it is unsupported by released evidence.**

- *Unsupported mechanism claim.* "Evaluating the same weights with batch statistics gives a smooth curve: the running averages … lag behind the weights." No experiment in `run_fl_v2.py` or `results_v2` evaluates with batch statistics. The claim appears only in the code docstring. Either add the evidence (a curve of train-mode evaluation, or BN statistics recomputed on a calibration set before each evaluation) or drop the mechanistic claim.
- *Not federation-specific.* The instability is shown in **centralised** training: TON validation F1 swings 0.12–0.88 between consecutive epochs in repetitions 0–2. It is a property of Keras BatchNorm (momentum 0.99) with Adam at lr 0.003 on this model, not of federation.
  - The abstract ("Batch normalisation makes federated evaluation unstable") and the conclusion ("Batch normalisation should be avoided in federated IDS models") misattribute it.
  - The standard remedies were not tried: lower BN momentum, recomputing statistics before evaluation, FedBN (Li et al., ICLR 2021), which the round-1 report requested.
  - On UNSW the effect is much weaker: FedAvg-BN 0.423 vs GN 0.433 selected; central-BN final 0.470 vs 0.518.
- *Selected-round result favours BN.* At the selected round BN is *higher* on TON at α = 0.5 (0.865 ± 0.010 vs 0.840 ± 0.043). "Statistically indistinguishable" rests on n = 3, and the paired difference uses only repetitions 0–2. The honest conclusion is: "BN gives similar or better selected-round accuracy but an unstable final model in this setup; with validation-based checkpointing the instability is harmless". That is a weaker and more useful statement.
- *Contribution 4 is not new.* Hsieh et al. [21] already show BN failure under skew and recommend GN. Contribution 4 should be reframed as a confirmation, not a "finding".
- *Fig. 2 is a single repetition (repetition 0).* Add the per-round standard deviation over repetitions, or a band.

**MC9. The training budget is not shown to be sufficient, which affects the "cost of federation".**

- Selected rounds cluster at the end of the 50-round budget:
  - FedAvg α = 0.5 TON: rounds 44, 45, 49, 47, 50;
  - α = 1.0: 47, 50, 46, 50, 46;
  - IID: 50, 41, 38, 45, 47.

  These models are still improving.
- "Same number of passes" (100 epochs) gives the centralised model about 50k sequential steps. The FedAvg global model makes only 50 averaged moves of about 1k local steps each.
- The reported gaps (−0.042 IID, −0.059 at α = 0.5) are therefore gaps at a fixed communication budget, not at convergence.
- **Required:** run FedAvg to convergence (e.g., 150–200 rounds) for at least IID and α = 0.5 to show the gap is not mostly budget. Alternatively, state that the gap is defined at 50 rounds and plot validation F1 against rounds with CIs.

**MC10. Novelty and positioning for IEEE IoT-J.**

A controlled re-evaluation can be publishable. As it stands, however, most findings confirm known results:

| Finding | Already known from |
|---|---|
| Tuned FedAvg is hard to beat | NIID-Bench [11]; Reddi et al. [19] |
| BN fails under label skew | Hsieh et al. [21] |
| DP harms minority classes | Bagdasaryan et al. 2019 |
| FL bandwidth exceeds the data size for small tabular models | Simple arithmetic |

The IoT-specific value is the IDS numbers, but the evidence base is narrow:
- two datasets, one explicitly "non-IoT", with UNSW's centralised ceiling at 0.53 macro-F1 and a benign FPR of 12.8%, which is unusable as an IDS;
- only simulated Dirichlet partitions, N = 4, full participation;
- one small CNN.

For a re-evaluation paper to be a contribution at IoT-J, I would expect at least:
1. **One natural partition.** For example, TON-IoT split by source host or subnet before IP removal; N-BaIoT by device; or cross-dataset FL on a shared NetFlow schema such as the NF-* datasets.
2. **One genuinely IoT-centric dataset with device structure,** such as CIC-IoT2023 or Edge-IIoTset (Ferrag et al., 2022).
3. **One label-skew-specific or personalised baseline,** since the paper's own result is that the generic drift fixes fail precisely there. Candidates: FedLC/logit adjustment, FedRS, FedBN, Ditto.
4. **A port-feature ablation.** `dst_port`/`src_port` are still among the 25 TON features and are the top-ranked SHAP features in the authors' baseline (`shap_importance.json`). If the federated conclusions rest on a testbed port shortcut, the heterogeneity story changes.

Without (1)–(2), the paper reads as a careful methodology note better suited to a venue such as *Computers & Security* or an IEEE Access-type outlet. With them, it would be a solid benchmark contribution.

### 4. Minor concerns

1. **Benign FPR claim.** "Benign false-positive rates stay at a few percent (Table I)" (§V-A) holds for TON only. UNSW FPRs are 6–16%, with CIs wider than the means (6.31 ± 7.03). Qualify the sentence.
2. **p-value formatting.** "p = <0.001" appears several times; write "p < 0.001".
3. **Table and figure naming.** The PDF captions say "Table 1…5" and "Figure 1…3", but the text says "Table I…V" and "Fig. 1". Use IEEE style consistently.
4. **Table III caption.** It says "three repetitions", but its first row (N = 4) is the 5-repetition core result. Also, the UNSW scaling and BN results are discussed in the text ("0.474 ± 0.124 to 0.372 ± 0.101") but appear in no table. Add a UNSW column or table.
5. **Fig. 3 rows.** The caption says "three noise levels"; four were run, and σ = 1.5 is omitted without comment. The "no noise" row is the run with no clipping either; say so, or show the clip-only row. In the PDF the left panel's "no noise" label overlaps the right panel.
6. **Fig. 1 coverage.** Fig. 1 omits FedAdam and SCAFFOLD. The centralised band on UNSW is nearly invisible. Consider a paired-difference forest plot (FL − central, FL − local, algorithm − FedAvg), which conveys the paired design far better than overlapping marginal CIs.
7. **"± denotes a 95% CI".** With n = 3 the t multiplier is 4.30. Give the SD alongside in the tables, as the text already does once (0.153 vs 0.009). Readers misread these as SDs.
8. **UNSW DP wording.** "Backdoor, DoS and shellcode classes, already weak without noise, disappear": shellcode is 0.39 without noise, which is not "weak". Analysis and worms are ≈0 in every non-central configuration; say so.
9. **Majority-class reference.** Give majority-class and stratified-random macro-F1 as reference lines (requested in round 1, minor 4). This matters for interpreting UNSW SCAFFOLD values of 0.066–0.09, which are constant predictors.
10. **Selection-rule claim.** "The conclusions do not depend on the selection rule" holds for FedAvg, but SCAFFOLD α = 0.1 (0.657 vs 0.597 TON; 0.185 vs 0.088 UNSW) and the BN runs are clear exceptions. Qualify the sentence.
11. **Parameter size.** "95.0 kB of parameters" should read "23,754 parameters (95.0 kB in float32)". Note that the size is the same for UNSW because of global average pooling.
12. **Determinism.** Document GPU/XLA nondeterminism (`jit_compile=True`) or enable deterministic ops.
13. **Reproducibility.** The repository DOI is still a placeholder. Archive the exact commit and the results JSON, including `tuned.json` and the `tune_*` runs, which are essential to judge MC1–MC2.
14. **Related-work positioning.**
    - "Most of these works report results for one partition and select models on the evaluation data" is a strong generalisation. Support it with a small tabulated audit of the cited works, or soften it.
    - Add FedBN, MOON/FedDyn/FedLC, Ditto, Bagdasaryan et al. 2019, Ponomareva et al. 2023, and at least one FL-IDS poisoning reference (Nguyen et al. 2020; FLAME, USENIX Sec 2022). The limitations paragraph mentions poisoning, but the related work does not.
15. **Generative-AI declaration.** The declaration states that Claude wrote the analysis code and drafted the paper. Check the IEEE policy on disclosure placement (acknowledgments section) and wording.
16. **DP-table reference row.** The row "FedAvg, BatchNorm, minibatch Adam (no DP machinery)" adds a BN model to a GN-based DP comparison. It is not needed and confuses the reference. Keep only the GN short-schedule row as the non-private reference.

### 5. Do the conclusions follow from the tables?

| Claim (abstract/conclusion) | Verdict |
|---|---|
| FedAvg costs −0.042 / −0.059 / −0.185 vs centralised (TON) | Numbers correct; **conditional on the 50-round budget** (MC9). |
| Federation gains +0.080 / +0.286 over local training | Correct as mean over clients on the global test set; "each site gains" is not supported (MC6). |
| FedAdam and SCAFFOLD do not improve on tuned FedAvg | SCAFFOLD: supported. FedAdam/FedProx: **confounded by the tuning lr mismatch** (MC1). |
| "SCAFFOLD's advantage disappears once both use the same optimiser" | Mis-stated: there is no advantage in this study (MC7). |
| FedProx helps only under severe skew on UNSW | p = 0.027 uncorrected; does not survive multiplicity correction (MC3). |
| BN makes federated evaluation unstable; avoid BN | Instability real but not federation-specific; mechanism claim unsupported; remedies untested (MC8). |
| DP at ε ≈ 0.8 costs ~13 points, flat across ε, concentrated in minority classes | Directionally supported; guarantee excludes preprocessing and tuning; short schedule; DP hyperparameters untuned (MC4). |
| FL does not save bandwidth | Supported. |

### 6. Requested changes (prioritised)

**Required:**
1. Re-tune FedProx μ and FedAdam η_s at the client lr actually used (or jointly), extend the grids past their edges (FedAvg lr ≥ 0.01; SCAFFOLD lr ≥ 1; FedProx μ ≤ 10⁻⁴), tune over ≥3 repetitions disjoint from the evaluation partitions, and re-run the core algorithm comparison (MC1, MC2).
2. Apply a multiplicity correction to the paired tests, pre-declare the primary comparisons, and soften the FedProx/UNSW abstract claim (MC3).
3. State the conditional scope of the DP guarantee: preprocessing fitted on private records, hyperparameters tuned non-privately. Preferably, refit preprocessing on data-independent ranges. Add one DP configuration at the full training budget, with ε held fixed, and a small C/lr sweep at one ε (MC4).
4. Report per-client local-only results, the fraction of clients that gain, and per-client-distribution test results; fix the "each gains" sentence (MC6).
5. Rewrite the SCAFFOLD narrative:
   - drop "advantage";
   - report non-finite runs and the correct collapse count (2/5 at lr 0.1);
   - test lower learning rates before attributing failure to control variates;
   - disclose or fix the uniform control-variate averaging.

   Remove the references to the unpublished earlier draft (MC7).
6. Provide evidence for the BN mechanism or remove it, test BN recalibration or FedBN, and reframe the BN result as confirming [21], not as federation-specific (MC8).
7. Show convergence (longer FedAvg runs) or define the gaps as fixed-budget gaps (MC9).
8. Describe a deployable validation/selection procedure (federated validation) (MC5).

**Strongly recommended for IoT-J (MC10):**
9. One natural-partition experiment and one device-structured IoT dataset.
10. A port-feature ablation on TON-IoT.
11. One label-skew-specific or personalised baseline (FedLC/FedRS, or FedBN/Ditto).

### 7. Overall assessment

The revision is a substantial and honest improvement. The round-1 defects that made the paper unpublishable are gone:
- UNSW leakage;
- test-set selection;
- a fixed partition;
- a non-DP "DP" mechanism;
- an untrained FedAdam;
- wrong bandwidth arithmetic.

The released numbers are internally consistent with the text. What remains is a set of design and interpretation problems that still bear on the headline conclusions:
- the FedProx/FedAdam comparison is confounded by tuning at a different client learning rate;
- several significance claims do not survive correction;
- the DP guarantee needs explicit scoping and the DP utility curve is measured in an under-trained regime;
- the BN and SCAFFOLD narratives over-interpret or mis-describe the evidence;
- the 50-round budget may inflate the federation gap.

On novelty, the paper is a careful confirmation of known FL phenomena on two small-feature IDS benchmarks. For IoT-J it needs at least one realistic (natural, device-level) partition or IoT dataset to justify the venue. I recommend **major revision**, and I would be glad to see it again.

**Recommendation: Major revision. Score: 5/10.**
