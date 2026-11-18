# Paper 6: review of the Methods against the code (round 1)

Scope: `Manuscript_filled.md` §3 and §4 checked against `Papers pipeline/experiments/paper6_multimodal/run_p6_v2.py`, `bench_latency.py`, `src/data/loader.py`, `src/data/preprocessing.py`, `experiments/baseline/config.py`, the baseline `pipeline.pkl` files, `experiments/paper4_federated/prepare_nbaiot.py`, and `Revision_v2/Paper6/compute_values.py`. This was a read-only review. Nothing was trained. The data checks used the project loader and pipeline on CPU (scratch script `chk_data.py`), and the saved JSON in `results_v3` was read directly.

## What checks out

- **Fit set of the TON-IoT `pipeline.pkl`.** The file dates from 27 Jul, which is older than loader v2.1. It is still consistent with the current split: the scaler saw 128,484 rows, its min and max equal those of the current `X_train` exactly, and the selected-feature names line up with the loader's columns. UNSW-NB15 matches in the same way. MI, RFECV and min–max were therefore fitted on the current training split only.
- **SMOTE.** It runs after the split and on the training rows only (`random_state=42`). Two calls gave bit-identical output. The sizes match Table 1: TON 128,484 → 293,550 and UNSW 74,167 → 363,220. Validation and test are never resampled.
- **QuantileTransformer.** It is fitted on `Xfit` only (`run_p6_v2.py:70`), so there is no val/test leakage.
- **Model selection.** Early stopping and LR reduction monitor `val_loss`, and the test set is used only for the final prediction.
- **Architectures** match §4.1: early fusion = `_encoder` on `np.r_[s1,s2]`, late fusion has a Dense(128) layer, and attention fusion is a 2-way softmax followed by a weighted sum and Dense(64).
- **Parameter counts in the JSON:** early 34,634, late 85,770, attention 73,036, and early_wide 86,613 (TON).
- **Attention extraction** (`:165-169`) reads the softmax layer on the same test inputs, and column 0 is the weight that multiplies e1. This is correct.
- **Metrics:** macro-F1 uses sklearn macro over all classes, which are all present in the test set. Benign FPR = share of `normal` test records predicted as non-normal. Both are correct.
- **Random splits.** Each split r has the semantic sizes, uses rng(1000+r) and runs with seed r mod 5. I checked this against `cfg` in every `late_rand*` file. No random split reproduces the semantic one.
- **Test counts:** MITM 156 and worms 19 are confirmed.
- **N-BaIoT:** 0 val/test rows are identical to a training row.
- **Epoch caps.** `epochs == 60` in compute_values counts true caps correctly: any run that reaches 60 epochs has best_epoch ≥ 51, so early stopping never fired.

## Bugs

**B1. Best weights are NOT restored when a run hits the epoch cap** (`run_p6_v2.py:154`, Keras 2.10).
- In `keras/callbacks.py:2030-2040` of the installed `tf.keras` 2.10, `restore_best_weights` only acts when early stopping fires. If `fit` ends at the epoch budget, the final-epoch weights are evaluated instead.
- This affects the 13 capped main runs (best epoch → evaluated epoch):
  - TON s1 seeds 0 and 4 (59 and 55 → 60)
  - TON s2 seed 3 (60, no effect)
  - TON early seed 1 (59)
  - TON late seed 3 (57)
  - TON attention seeds 0, 2 and 4 (57, 60, 54)
  - N-BaIoT s2, all five seeds (60, 58, 58, 56, 51)
- It also affects any capped `early_wide` or `_e120` control run (TON early_wide seed 4: best 51, epochs 60).
- §4.1 says "best weights restored", which is false for these runs.
- Impact on results: probably small, because the LR has already been reduced by then. However, it falls unevenly (3 of 5 attention seeds on TON against 1 of 5 for early and late), and it feeds into the TON late-vs-early test, the only one significant after Holm.
- Fix: save a `ModelCheckpoint(save_best_only=True)` or keep the best weights in a callback and restore them in `on_train_end`. Alternatively, report the limitation.

**B2. The cross-split guard runs before feature selection, so many UNSW-NB15 test records still duplicate training records on the features the models see** (`loader.py:206-227`, which runs before `pipe.transform`).
- Measured on the selected features:

  | Dataset | Test identical to a train row (features + label) | Feature vector seen in training (any label) | Val identical to a train row |
  |---|---|---|---|
  | UNSW-NB15 (10 features) | 2,174 / 15,894 (13.7%) | 16.3% | 13.7% |
  | TON-IoT | 35 / 27,533 (0.1%) | – | – |

- Per UNSW class, the share of test records identical to a training record is: generic 0.78, analysis 0.53, backdoor 0.49, reconnaissance 0.22, dos 0.17. There are also 735 duplicates within the UNSW test set.
- §3.1 lists the guard after feature selection and scaling ("…fitted on the training split, test and validation records identical to a training record are removed"), so the reader will assume the guard applies to the model inputs.
- Impact: the absolute UNSW scores are partly measured on memorisable records. This is also likely to favour the RF, which can memorise exact vectors, over the CNNs, so it bears on the "RF beats every CNN by +0.036 on UNSW" result. The architecture ranking is probably unchanged, because all models share the same test set.
- Fix: either re-apply the guard on the selected features (and report the counts), or state the order and report the 13.7% honestly. A cheap robustness check is to re-score the saved models on the deduplicated test subset (this needs the predictions, which are not saved; see L7).

**B3. compute_values never defines `TON_IoT_early_wide_params`** (`compute_values.py:139-143` only define `_params_single`, `_late` and `_attn`).
- The placeholder in §4.2 will stay as ⟦…⟧ even after the controls finish.
- The value in the JSON is 86,613.

**B4. The size of the Holm family depends on which result files exist** (`compute_values.py:184`). The text contradicts the code.
- §4.3: "The fusion-architecture comparisons (late against early, attention against early, late against attention, **and the two capacity comparisons, on each dataset**) form one family of **9** paired t-tests".
- Five comparisons on three datasets make 15 (or 11 if early_wide runs on TON only). The current 9 is the family before the controls exist.
- As control files arrive, the family grows and every Holm p-value changes, including the one significant result (TON late − early, p = 0.022). The build will silently change the reported adjusted p-values.
- Fix: decide the family in advance and hard-code it, and fail the build if any member is missing.

## Mismatches between the Methods text and the code

1. **Early fusion is not invariant to the split.**
   - Text: "Early fusion is not repeated, because a random split only reorders its inputs" (§4.2). Also `compute_values.py:285`: "early fusion does not depend on the split".
   - A 1-D CNN with kernel 3 and max-pooling is not invariant to feature order. The semantic early fusion feeds `np.r_[s1,s2]`, so the semantic grouping also fixes which features are neighbours.
   - "early fusion = one CNN on all features" is therefore true only for one particular order. The §5.3 comparison "late fusion on random splits vs early fusion" mixes a semantic-ordered early model with randomly grouped late models.
   - Fix: run early fusion on the 10 random orders (cheap), or remove the claim.

2. **The quantile-transform fit set is described inaccurately.**
   - Text: "quantile transform … (1,000 quantiles, fitted on the training rows)".
   - Code: it is fitted on `Xfit`. For TON and UNSW that is the SMOTE-oversampled set (56% and 80% synthetic rows), subsampled to 200,000 rows (`subsample=200000`).
   - This is not leakage, but the text should say "fitted on the oversampled training set".

3. **N-BaIoT has no feature selection.**
   - Text: "Feature selection leaves the streams unequal in size: … 30 and 85 on N-BaIoT" (§3.2), and "Each dataset's selected features are split…".
   - `prepare_nbaiot.py` keeps all 115 features (only a zero-variance drop, which removed none).
   - It also min–max scales N-BaIoT with the lower bound clipped to 0 (`:84`). §3.1 does not mention the scaling, while §4.2 says the RF runs on "the min–max scaled features".
   - Also unmentioned: 17 conflicting-label rows are dropped (`:60-62`).

4. **Order of the guard in the pipeline.** See B2. "test and validation records identical to a training record are removed" is placed after feature selection and scaling in the text; in the code it happens before.

5. **"Best weights restored".** See B1.

6. **Seeds.** Text: "the seeds vary initialisation and batch order only".
   - The seeds also change the dropout masks.
   - Op determinism is not enabled (`tf.config.experimental.enable_op_determinism` is absent), so cuDNN nondeterminism adds run-to-run noise and the runs are not bit-reproducible.
   - For the RF, the seed changes the bootstrap and the feature sampling.

7. **"10 (6 + 4)" on UNSW-NB15 is the RFECV floor.** `min_features_to_select=10` (`preprocessing.py:118`), and UNSW selected exactly 10. Either say so or note it.

8. **Stream 1 on UNSW-NB15.** The text says it holds "mean size per direction", but only `smean` (source) was selected; `dmean` was not.

9. **RF invariance to the quantile transform.** Text: "Its predictions do not depend on monotone transforms of single features, so the quantile transform would not change it."
   - This is only approximately true. Thresholds are midpoints and change under a nonlinear map, and the quantile transform clips values outside the training range, so it is not injective there.
   - Suggest: "would change it at most marginally".

10. **Latency docstring.** `bench_latency.py` says "batch1000_ms: … median of 20 calls". With `repeats=1`, `time_calls` returns the mean of the 20 calls, not the median. The timing itself is fine.

## Lesser concerns (fairness and inference)

- **L1. Random-split comparison mixes a 5-seed mean with single-seed values** (`compute_values.py:262`, §4.3 "position of the semantic mean among the ten random splits"). The semantic value is a mean over 5 seeds, while each random split is one seed. The two have different variances, so "below n/10" and the claim "semantic above all ten" partly reflect averaging. Compare per-seed semantic values, or run each random split with all 5 seeds.
- **L2. The confidence intervals cover seed variance only.** The test set is one fixed split, so the CIs and Holm p-values ignore test-sampling variance. The +0.005 TON late − early gap is about 140 records out of 27.5k. A paired bootstrap over test records would show whether it holds beyond this split, but it needs saved predictions (see L7).
- **L3. The "better single stream" is chosen on the test means** (`compute_values.py:165,167`). This is descriptive only, but it is a selection on test.
- **L4. RF against CNN.** Neither model family is tuned. The CNNs get validation-based early stopping and an LR schedule. The RF has fixed settings (max_depth 20 on 293k–363k rows, which may if anything limit it). Any unfairness therefore works against the RF, and "no CNN beats the RF" is robust. Both train on SMOTE data with no class weights.
- **L5. Prior shift between training and validation.** Training is class-balanced by SMOTE, but `val_loss` for early stopping uses the imbalanced validation set. UNSW stream-2 CNNs stop at best epochs 2–12, which may reflect this rather than convergence.
- **L6. Attention weights.** The weights multiply un-normalised ReLU embeddings, so a weight alone does not measure a stream's contribution: the norms of e1 and e2 matter too. This supports the paper's caution, but should be stated as a reason.
- **L7. Predictions and models are not saved** (`run_cnn` and `run_rf` save metrics only). Post-hoc checks such as B2 re-scoring, a bootstrap or McNemar are therefore impossible without retraining. Save `pred` (an int8 vector per run).
- **L8. Table 2 parameter column.** The "Params" column uses the first dataset's count for each row (`compute_values.py:121`). N-BaIoT differs slightly (34,569 / 85,641 / 72,971). Label it "(TON-IoT)".
- **L9. Provenance of `data_stats.json`.** compute_values and verify read `data_stats.json`, but no script in Paper6 or paper6_multimodal writes it, so its provenance is undocumented. Its counts do match my recomputation (Table 1 and the test counts).
- **L10. Latency measured during training.** `run_cnn` still times 200 GPU calls in every run (`:170-179`). This is harmless, but those numbers must never be reported (the paper correctly uses `bench_latency.py`).
