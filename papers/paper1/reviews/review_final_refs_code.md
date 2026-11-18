# Paper 1: final check of references and methods against code

Date: 2026-11-07. Scope: `Manuscript_filled.md` (and `_anon`, whose reference list is identical), the code in `Papers pipeline/experiments/baseline/run_p1_v2.py`, `bench_p1_latency.py`, `paper6_multimodal/run_p6_v2.py` (loader, metrics, BestWeights), `src/data/loader.py`, `src/data/preprocessing.py`, `paper4_federated/prepare_nbaiot.py`, `Revision_v2/Paper1/compute_values.py`, and the result JSONs and saved seed-0 models in `baseline/results_v2/`. No training was run and no dataset was loaded. I read the JSON results and opened four small saved GBDT pickles (TON-IoT `gb_l7`, `gb_l15`, `gb05_l7`, `gb05_l15`, seed 0) plus one RF pickle.

Severity: **critical** means a reported result or headline claim is wrong or unsupported; **major** means a stated method differs from what was done in a way that affects reported numbers or their interpretation; **minor** means a wording, completeness or precision issue.

Summary: 34 references. All exist, and the authors, year, title, venue, volume, issue, pages and DOI match Crossref. There are no missing or uncited references. One claim in Related Work contradicts a cited abstract (major), and there are four minor citation-wording points. On the code side, one critical bug was found: the GBDT "collapse" is caused by evaluating diverged models, because scikit-learn's early stopping does not roll back to the best iteration. It changes a headline claim and may change the 100 KB comparison. There is one major timing mismatch (GBDT latency is not single-threaded), and several minor method-description gaps.

---

## 1. References

### 1.1 check_citations.py

- `python check_citations.py` (reads `Manuscript.md`): `cited but not in references: []`, `in references but not cited: []`.
- The same check run on `Manuscript_filled.md` gives the same result, and its reference list is byte-identical to that of `Manuscript.md` and `Manuscript_filled_anon.md`.
- **minor (tooling).** `check_citations.py` is hard-wired to `Manuscript.md`, not to the filled or anonymised files that are submitted. It passes today only because the reference sections are identical. Point it at the file that is built into the .docx, or loop over all three.

### 1.2 Existence and metadata (Crossref, fetched 2026-11-07)

All 32 DOIs were resolved on `api.crossref.org/works/<doi>`. For each one, the author list (names and order), year, title, container, volume, issue and pages or article number were compared with the reference entry. **All 32 match.** The two entries without a DOI (Arp et al. 2022, USENIX Security, pp. 3971–3988; Pedregosa et al. 2011, JMLR 12(85), 2825–2830) were confirmed by URL in the earlier audit (`reference_audit/refcheck.json`, HTTP 200), and their metadata is correct.

Small points, none of which needs action unless the journal style requires it:

| Reference | Observation | Severity |
|---|---|---|
| Ullah et al. (2026), CMC 88(3) | No pages or article number. Crossref lists "1-10", which is a placeholder. The publisher page (techscience.com/cmc/v88n3/68157, online 2026-07-23) gives no first or last page either. Leaving them out is correct for now; add them when assigned. | minor |
| Ngo et al. (2023), Future Internet 15(1), 9 | Crossref "issued" is 2022-12-26 (online). It sits in the 2023 volume, so 2023 is correct under APA. | none |
| D'hooge et al. (2022), DIMVA | Correct. The LNCS volume (13358) could be added for completeness. | none |

### 1.3 Claims compared with the cited sources (abstracts from Crossref, OpenAlex, Semantic Scholar or the publisher page)

| # | Location | Claim in text | Evidence | Severity |
|---|---|---|---|---|
| R1 | §2.1, para 3 ("Two gaps remain…") | "The studies that include trees use one dataset, without a compact CNN of the kind proposed in the neural studies **and without int8 versions of the networks on the same data**." | De Bernardis et al. (2026), abstract: "TensorFlow Lite Micro full-integer INT8 for the MLP (9.34× size reduction to 13.03 KB)", trained on TON_IoT. So the closest prior study **does** have an int8 network on the same data. It also runs a cross-domain evaluation on CIC-IoT-Dataset2023, so "one dataset" holds only for training. Suggested wording: "…use one training dataset and no compact CNN; De Bernardis et al. include an int8 MLP, but not at matched size budgets or over several seeds." | **major** |
| R2 | §2.1, para 2 | De Bernardis et al.: "a quantised LightGBM model of 74 KB **and a depth-5 decision tree** gave the best accuracy–size trade-off." | Abstract: "LightGBM INT8 (F1=0.9992) provides the best accuracy–size trade-off **among ensemble models**". The depth-5 DT is credited with the fastest inference (5.6 µs on the ESP32-C3), not with the best trade-off. Reword to "a quantised 74 KB LightGBM model gave the best accuracy–size trade-off among the ensembles, and a depth-5 decision tree ran in 5.6 µs". | minor |
| R3 | §2.2 | "tree ensembles match or beat deep networks … particularly on medium-sized data with uninformative or heavy-tailed features (Grinsztajn; Shwartz-Ziv & Armon; McElfresh)" | Grinsztajn: medium-sized data, uninformative features, irregular targets. The heavy-tailed finding is McElfresh's ("GBDTs are much better than NNs at handling skewed or heavy-tailed feature distributions"). McElfresh's main conclusion is that the gap is often negligible, and TabPFN is best on average. As a grouped citation the claim is supported. If the authors want precision, attribute "heavy-tailed" to McElfresh. | minor |
| R4 | §2.3, last sentence | "none of the lightweight-IDS studies in Section 2.1 tests whether its accuracy survives without such fields" | For most of these papers this can be checked only from the abstract (`literature_verified.md` says the Table B figures are abstract-only). §2.1 hedges the parallel claim ("as far as their texts (or … their abstracts) show"); this sentence does not. Add the same hedge. | minor |
| R5 | §2.1 | Tekin et al. (2023): "compare the energy and latency of several learners on cloud, edge and IoT devices and find a decision tree cheapest on-device" | No abstract could be retrieved in this pass (Crossref, OpenAlex and S2 all failed or were rate-limited). The claim matches the 2026-11-01 entry in `literature_verified.md` (A3). It was not re-verified today. | minor (unverified today) |
| — | all other citations | Zhao 2022, Lahasan & Samma 2022, Zakariyya 2023, Musthafa 2025, Ngo 2023, Mazinani 2026 (six DL models, int8 >90% size reduction, MLP and TCN best), Sharmila & Nagapadma 2023 (uint8, −92.23% size), Ullah 2026 (FP32/FP16/INT8 1D-CNN, 58.02→25.72 KB, *estimated* latency, Cortex-M4 reference), Soe 2020, Javed 2024 (273–276 µs < 0.3 ms), Jacob 2018, Apruzzese 2023 (hardware as a deployment factor, four platforms), Engelen 2021, Flood 2024, Booij 2022, Layeghy 2023, Geirhos 2020, D'hooge 2022, Borisov 2024, Shwartz-Ziv 2022, Sommer & Paxson 2010, Arp 2022, the dataset papers, Breiman, Chawla, Pedregosa | Each is supported by its abstract or title, or is a standard method citation. | OK |

---

## 2. Methods compared with code

### 2.1 Critical

**C1. The GBDT "collapse" is an evaluation bug: scikit-learn's HistGradientBoosting keeps the diverged iterations after early stopping.**
- *Location:* Abstract ("At the library's default learning rate, gradient boosting collapsed on some seeds"); §4.1 GBDT bullet; §5.1 para 2; Table 2 (lr 0.1 rows); Table A.2; §5.6 (GBDT lr 0.1 row); §6.2 rec. 4; §6.3 ("we did not diagnose why…").
- *Evidence:* `make_tree` uses `HistGradientBoostingClassifier(max_iter=1000, early_stopping=True, validation_fraction=0.1, n_iter_no_change=20)`. In scikit-learn 1.7.2, `fit` simply `break`s when early stopping fires (`gradient_boosting.py` ≈ line 1028). It does **not** truncate to the best iteration, so the stored model contains the 20 iterations after the best validation loss. The saved seed-0 models show that during those iterations the boosting diverged numerically:

  | TON-IoT seed 0 | test macro-F1 | best val. log-loss (iter) | val. log-loss at stop (iter) | train log-loss at stop |
  |---|---|---|---|---|
  | gb_l7 (lr 0.1) | 0.496 | 0.042 (249) | **361,323** (268) | 367,154 |
  | gb05_l7 (lr 0.05) | 0.798 | 0.049 (294) | **65,256** (313) | 65,123 |
  | gb_l15 (lr 0.1) | 0.945 | 0.050 (66) | 7.8 (85) | 9.8 |
  | gb05_l15 (lr 0.05) | 0.960 | 0.028 (1000) | 0.028 (1000, cap) | 0.0075 |

  So the "collapsed" seeds are models evaluated *after* the loss exploded (no NaN, just huge leaf values; the leaf value is −G/(H+λ) with `l2_regularization=0`). They are not a property of gradient boosting at lr 0.1. Every early-stopped GBDT run is also evaluated 20 iterations past its best point.
- *Consequences:*
  1. The abstract claim, the 6/20 and 2/20 collapse counts, Table A.2, recommendation 4 and the GBDT rows of Table 2 and Table 4 describe a code artefact.
  2. **The ≤100 KB comparison may change.** The non-diverged lr-0.1 seeds are small and accurate. TON-IoT `gb_l7` seeds 1–3 reach 0.935–0.944 at 67–76 KB, and `gb_l15` seeds 1–3 reach 0.949–0.952 at 90–121 KB. The configuration *means* (0.750 at 121 KB; 0.827 at 149 KB) are dragged down, and pushed above the 110 KB budget, by the diverged seeds, which also have more iterations (229 and 245 KB). With a correct GBDT, a tree model may sit at ≈0.95 below 100 KB, against 0.951 for the int8 CNN of width 2 and 0.923 for the RF currently quoted. The "+0.028 / +0.046 for networks up to 100 KB" claim (Abstract, §5.2, §6.1, §7) therefore needs re-running before it can stand.
  3. GBDT sizes, n_iter, per-tree latency and "10,000 trees" all include the 20 extra iterations. That is small, but it adds to the bias.
- *Fix (trees only, cheap):* re-fit each GBDT with `max_iter = argmax(validation_score_)` taken from a first fit, or evaluate with `staged_predict_proba` at the best iteration. Add `l2_regularization` (e.g. 1.0) as a sensitivity run. Then recompute values.json and re-state §5.1, §5.2, §5.6, Table A.2, the abstract and the conclusion. Only seed-0 models are saved, so the other seeds need re-fitting. No networks need to be re-run.
- *Severity:* **critical**.

### 2.2 Major

**M1. GBDT latency is not single-thread.**
- *Location:* §4.2 ("Latency is measured … on one thread"), §5.4, Table 3, Abstract (97 and 87 ms), §6.2 rec. 3.
- *Evidence:* `bench_p1_latency.py` limits TF to 1 intra- and 1 inter-op thread, sets TFLite `num_threads=1`, and calls `t.set_params(n_jobs=1)` only `if hasattr(t, 'n_jobs')`. `HistGradientBoostingClassifier` has no `n_jobs`. Its `predict` → `_raw_predict(X, n_threads=None)` → `_openmp_effective_n_threads()` uses all logical cores (i5-10200H: 4 cores and 8 threads). Neither `OMP_NUM_THREADS` nor `threadpoolctl` is set anywhere in the pipeline (grep finds none). The RF (n_jobs=1) and TF/TFLite timings are single-threaded; the GBDT timings are not. For batch-1000 calls, OpenMP parallelises over rows, so the GBDT "28–385 ms per 1,000" and "214–371× batching gain" are multi-thread numbers. For single records only one row exists, but each of the up to 10,000 per-tree calls opens an OpenMP parallel region. The "9–11 µs per tree" therefore includes OpenMP team overhead and could be either inflated or deflated relative to a true single thread.
- *Fix:* re-run the benchmark with `OMP_NUM_THREADS=1` set before importing scikit-learn, or wrap the calls in `threadpool_limits(1)`. Record the setting in latency.json.
- *Severity:* **major**. The batch numbers are affected directly; the single-record GBDT numbers are uncertain.

### 2.3 Minor

| # | Location | Text | Code | Severity |
|---|---|---|---|---|
| m1 | §4.1 GBDT | "up to 1,000 iterations with early stopping" | The 1,000-iteration cap was binding in 19 of 20 UNSW-NB15 lr-0.05 runs for 31 and 63 leaves (n_iter = 1000 in 4/5 and 5/5), in 3 of 5 `gb05_l7` runs on TON-IoT, and in the seed-0 `gb05_l15` model on TON-IoT, which is the one timed (val loss still improving at 1000). The best lr-0.05 GBDTs are therefore cap-limited, and their size and latency ("10,000 trees", 97 ms) are set by the cap. State this. | minor |
| m2 | §4.1 Quantisation | "post-training **full-integer** quantisation (int8 weights and activations…)" | `tflite_int8` sets `Optimize.DEFAULT` and a representative dataset, but neither `target_spec.supported_ops=[TFLITE_BUILTINS_INT8]` nor int8 input/output types. That is "integer quantisation with float fallback" and float I/O (the docstring says "float input/output"). Unsupported ops would silently stay float32. Either say "int8 weights and activations with float input and output (float fallback allowed)", or force INT8 ops and confirm the graph. Op coverage was not inspected here. | minor |
| m3 | §4.1 Quantisation | "calibrated on 500 training records" | 500 records drawn from `Qfit`, the SMOTE-oversampled set, so they include synthetic records. Say "500 records of the oversampled training set". | minor |
| m4 | §4.1 networks | quantile transform "fitted on a 200,000-record subsample of the oversampled training set" | True for TON-IoT and UNSW-NB15. For N-BaIoT, `Xfit` = the 110,227-record training set, which is not oversampled and smaller than 200,000, so the transform is fitted on all of it. Add "(all training records for N-BaIoT)". | minor |
| m5 | §3.1 | "Feature selection is fitted on the training split: a mutual-information filter … followed by RFECV…" | MI uses the full training split, but RFECV is fitted on a random **50,000-record subsample** of it (`rfecv_subsample=50_000`, seed 42). It uses 3-fold `StratifiedKFold` without shuffling and step 1. State the subsample. | minor |
| m6 | §3.1 | "exact duplicates are removed after all column drops" | Deduplication (loader step 5) precedes the train-fitted column drops: >40%-missing columns (step 6) and zero-variance columns (step 9). Median imputation and the ±inf→p99 replacement (steps 7–8) are also not mentioned. The cross-split guard (step 10) runs after these steps, so the val and test guarantee holds, but the sentence is inaccurate. Within-training duplicates can reappear. Mention steps 6–9 briefly. | minor |
| m7 | §3.1 | "oversampled with SMOTE until every class has as many records as the largest" | Matches the data: TON 10×29,355 and UNSW 10×36,322. Note that `config.py` applies `smote_max_per_class=50,000` to UNSW-NB15. It is non-binding because it exceeds the largest class (36,322), as compute_values.py notes. The one-hot encoding uses `drop_first=True`, and SMOTE uses k=5 with seed 42, fixed across model seeds (consistent with §6.3). | none (optional detail) |
| m8 | §3.1 N-BaIoT | pre-processing description | Matches `prepare_nbaiot.py`: Bernoulli 3% sample per device-class file, float32 deduplication, BASHLITE tcp/udp dropped (40 and 32 distinct vectors), 70/15/15 split stratified by device×class, guard, all 115 features kept, min–max fitted on train. Not mentioned: 17 rows with identical vectors but conflicting labels were removed. Trivial. | none |
| m9 | §4.2 | compact encoding: "two 16-bit child indices" | The UNSW-NB15 RF with unlimited depth has ≈67,600 nodes per tree on average (max 68,813 in seed 0), which exceeds the 65,535 limit of a uint16 index. The encoding as described cannot address those trees. The effect on the size (≈1 byte more per node) is small, and only the 198 MB configuration is affected. Add "(32-bit indices for trees with more than 65,535 nodes)", or note the exception. | minor |
| m10 | §4.2 | pickled trees are larger "since scikit-learn stores a 64-bit value per class for every node" | True for RF (`tree_.value` is float64, n_nodes × n_classes). For HistGradientBoosting, each node is a ≈50-byte record with one float64 value, not one per class. Reword to "stores 64-bit values and bookkeeping fields for every node". | minor |
| m11 | §4.2 / bench docstring | per 1,000 records | Code: `timed(..., [X]*10, repeats=1, warm=1)` gives the **mean** of 10 calls after 1 warm-up. The script docstring says "median of 10 calls". The paper states no aggregation, which is fine; fix the docstring or report "mean of 10 calls". | minor |
| m12 | §4.4 | "macro-averaged area under the precision–recall curve" | Computed as one-vs-rest **average precision** (`average_precision_score`), the standard step-wise estimator, not a trapezoidal AUC. Say "average precision (area under the PR curve)". | minor |
| m13 | §5.5 / §4.1 | input-scaling control | Matches the code (`minmax` group, CNN width 1, TON and UNSW, five seeds). One TON run (seed 4) reached the 100-epoch cap (best epoch 92), so the min–max CNN may be slightly under-trained in that seed. | minor |

### 2.4 Checked and consistent

- **Split and seeds.** Split 0 of the loader (seed 42, stratified 70/15/15), five model seeds 0–4 on the same split, SMOTE seed fixed. Table 1 counts match `Paper6/data_stats.json` and `nbaiot_prepared.json` (128,484/293,550/27,533/27,533; 74,167/363,220/15,894/15,894; 110,227/22,503/22,453). The duplicate-on-selected-features shares are 0.13% (TON) and 16.27% (UNSW), and every class is present in the clean subsets.
- **Feature counts.** 25 on TON-IoT (including `src_port`, `dst_port`), 10 on UNSW-NB15 (no `id`), 115 on N-BaIoT.
- **Models.** LR uses lbfgs multinomial, L2, C=1, max_iter 2000, on the quantile features. RF uses `max_features='sqrt'`, 25 or 100 trees at depths 8/12/16, 200 trees unlimited, and the earlier configuration (200 trees, depth 20, min leaf 2). GBDT uses 7/15/31/63 leaves at lr 0.1 and lr 0.05, max_iter 1000, 10% internal validation, n_iter_no_change 20.
- **Networks.** The CNN architecture and the MLP architecture match. Parameter counts were recomputed by hand: CNN 23,946 and MLP 17,162 on TON-IoT, as stated. Widths use `max(4, round(n·w))`. Training: Adam 1e-3, ReduceLROnPlateau (patience 5, factor 0.5, min 1e-5), batch 256, ≤100 epochs, EarlyStopping patience 15 on val_loss, and BestWeights always restores the best epoch (asserted). `jit_compile=True` is recorded in every network JSON, and no network run hit the epoch cap apart from m13.
- **Port ablation.** The six configurations match §4.3. Both ports are removed after selection; MLP width 1 without ports has 16,906 parameters (= 17,162 − 2×128).
- **Metrics.** Macro-F1 on the test set. Benign FPR = share of benign records predicted as any attack; attack detection = share of attack records not predicted benign. The int8 metrics come from the TFLite outputs, and agreement is with the float32 predictions.
- **Statistics.** `scipy.stats.ttest_ind(equal_var=False)` (Welch), no multiplicity correction, as stated. "Smallest within 0.01 of the best mean" is implemented as described, over all points including int8.
- **Sizes.** Float32 networks = 4 bytes × `count_params()` (including the BN moving statistics). Int8 = TFLite flatbuffer length. Tree compact size = mean over seeds of the documented encoding. Pickled size = `len(pickle.dumps)`.
- **Latency.** Seed-0 models, CPU recorded as Intel Core i5-10200H, TF 2.10.0, `CUDA_VISIBLE_DEVICES=-1`. Single-record timing is the median of 5 repeats of 200 calls after 20 warm-up calls, as stated. Trees are timed with `predict`, float32 networks with `tf.function`, and int8 networks with the TFLite interpreter (1 thread). The exception is GBDT threading (M1).
