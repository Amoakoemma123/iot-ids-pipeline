# Paper 6: review of the Methods against the code (round 2)

**Scope.** I checked `Manuscript_filled.md` §3–§6 against these files:

- `Papers pipeline/experiments/paper6_multimodal/run_p6_v2.py` (working copy = commit `d2e74ca`, no uncommitted diff)
- `bench_latency.py`
- `src/data/loader.py` and `src/data/preprocessing.py`
- `experiments/baseline/config.py` and the baseline `pipeline.pkl` files
- `experiments/paper4_federated/prepare_nbaiot.py`
- `Revision_v2/Paper6/compute_values.py` and `values.json`
- every JSON and `*_pred.npy` file in `results_v4/`, `results_v3/` and `results_v2/`
- git history `55350cc` → `f5b08e9` → `d2e74ca`
- the queue logs in `Papers pipeline/logs/`

**Method.** This was a read-only review. Nothing was trained and no full dataset was loaded. I used small scripts to read the result JSONs, the saved predictions, the clean masks, `streams.json` and `tune_trees.json`, and the two pickled baseline pipelines.

## Summary

- **No leakage of test data into training or model selection.** The quantile transform, the min–max scaling, SMOTE, feature selection (split 0), tree tuning, early stopping and the LR schedule are all fitted on training or validation data. The test set is used only for final scoring and for the post-hoc attention ablation.
- **The restore-best-weights fix (R1-B1) is correct, and it covers every number in Tables 2–6 and 8.** The only reported numbers it does not cover are the "quantile" rows of Table 7 (finding M1).
- **The most serious problem is reproducibility.** The manuscript's justification for reusing v3 runs does not hold empirically (M2). Two equally valid replications of the UNSW-NB15 main experiment disagree on the descriptive claims (M3).

Severity counts: critical 0, major 3, minor 14.

---

## 1. Verification of the round-1 restore-best-weights fix (R1-B1)

**The fix is correct.**

- `BestWeights` (`run_p6_v2.py:221-241`) copies `model.get_weights()` whenever `val_loss` strictly improves.
- It restores those weights in `on_train_end`, so this happens whether or not early stopping fired.
- `EarlyStopping` no longer sets `restore_best_weights` (`:285`), so the two callbacks cannot conflict.
- Ties are handled consistently. Both `<` in the callback and `np.argmin` pick the first minimum, and `:293` asserts `restored_epoch == best_epoch`.

**It is applied to every network run in `results_v4`.**

- All 546 new network JSONs (main, capacity, mlp, colorder, resplit, random) carry `restored_epoch`, and it equals `best_epoch` in every one.
- No run is missing the field.
- All 546 `*_pred.npy` files reproduce their JSON `macro_f1` exactly.

**The reused v3 runs are also valid on this point.**

- 104 random-split runs and 15 N-BaIoT main runs are reused. All have `epochs < 60`, so early stopping fired.
- In the installed Keras 2.10.0 (`keras/callbacks.py:2030-2040`), `EarlyStopping(restore_best_weights=True)` restores the best weights when it fires. These runs were therefore evaluated at their best epoch.
- `reuse_v3` (`:384-402`) excludes every v3 run that reached 60 epochs. This is conservative: it also excludes runs where early stopping fired exactly at epoch 60.
- All 119 reused files match the v4 stream definitions (`cfg.s1` against `streams.json`), the seed and the experiment name.
- The data inputs are unchanged since v3:
  - `loader.py` was last changed before `55350cc`.
  - `pipeline.pkl` dates from 27 Jul (TON-IoT) and 24 Oct (UNSW-NB15).
  - `nbaiot_prepared.npz` dates from 26 Oct.
- The v3 and v4 models are the same architecture (identical parameter counts).

**Size of the bug in v3 (for the record).** The 30 v3 runs that hit the cap were re-run in v4. Macro-F1 changed by −0.0033 to +0.0054. This confirms round 1's expectation that the bug's effect was small.

**Text against code.**

- §4.1 (l.77): "1 of the 90 network runs of the main and capacity experiments used the full budget (latest best epoch 118)". This is correct. The run is N-BaIoT stream 2, seed 4: 120 epochs, best epoch 118.
- No run in any other v4 group (random, mlp, colorder, resplit) reached 120 epochs.

---

## 2. Major findings

### M1. Table 7's "quantile" rows come from the superseded v3 run, not the final run, and still include last-epoch-weight runs (major)

- **Where:** `compute_values.py:27` (`RES_Q = results_v3`) and `:352-364`; Manuscript l.231 and Table 7 (l.233-238).
- **What the text says:** "Table 7 compares that run with the final one; nothing else differed."
- **What the code does:** the "quantile" row is computed from `results_v3`, not `results_v4`.
- **Consequences:**
  - **Budget and bug.** These runs had the 60-epoch cap and the Keras restore bug. On TON-IoT, eight of the averaged runs (s1 seeds 0 and 4, s2 seed 3, early seed 1, late seed 3, attention seeds 0, 2 and 4) were evaluated with last-epoch weights. Table 7 is the only reported table still affected by B1.
  - **Numbers disagree with Table 2.** On UNSW-NB15 the rows read 0.498 / 0.362 / 0.562 / 0.569 / 0.559, against Table 2's 0.497 / 0.361 / 0.557 / 0.573 / 0.565.
  - **RF column is the old, untuned RF.** The "RF, stream 1" column (0.927) comes from v3, i.e. the v2 RF with `max_depth=20, min_samples_leaf=2`. The prose in the same section quotes the tuned v4 RF ("scored 0.928").
  - **What is true:** v2 → v3 differed only in input scaling. I confirmed this with `git diff 55350cc f5b08e9`; the only other changes are an added control group and the encoder width argument.
- **Impact:** the scaling lesson is unchanged, because the effect is about 0.9 macro-F1. However, the claim "the final one" is false, and readers will see numbers that disagree with Table 2.
- **Fix, either of:**
  - Relabel the row as "quantile (v3 run, 60-epoch budget)" and note the 8 capped runs.
  - Better: show v4 values in the quantile row and say that the v2 and v4 runs also differ in budget and weight restoration.

### M2. The justification for reusing v3 runs is contradicted by the project's own result files: same-seed re-runs do not stop at the same epoch (major)

- **Where:** Manuscript l.85: "These runs stopped early before the cap, so they restored the best weights and would have stopped at the same epoch under the present budget, up to GPU non-determinism." The code is `run_p6_v2.py:7-9` and `:384-402`.
- **Evidence from v3 against v4.** 51 v3 runs stopped early and were re-run in v4 with the same seed, data and architecture (the v4 main group re-ran everything except the N-BaIoT reuse set).
  - Only 9 of the 51 stopped at the same epoch. Examples:
    - UNSW-NB15 attention seed 1: 16 → 65 epochs
    - TON-IoT early seed 3: 38 → 58
    - UNSW-NB15 early seed 4: 36 → 16
  - Macro-F1 differences: mean |Δ| = 0.0033, maximum 0.018 (UNSW-NB15 attention seed 1).
- **Evidence within v3 itself.** The same code was run with a 60- and a 120-epoch budget (`results_v3/TON_IoT/early_e120`, `late_e120`):
  - early seed 0 stopped at 47 under the 60 cap and at 71 under the 120 cap;
  - late seed 1 stopped at 52 and 42.
  - So even without any code change, "same seed ⇒ same stopping epoch" does not hold on this GPU.
- **What this means:**
  - The reused runs are not reproductions of what v4 would have produced. They are independent draws under an equivalent protocol: same data, architecture, LR schedule and patience, with best weights restored. In that sense reuse is still legitimate, and the restoration argument above is sound.
  - The phrase "up to GPU non-determinism" hides that this non-determinism is about as large as the effects reported. TON-IoT late − early is +0.005, and semantic − random late fusion is +0.001 / +0.009.
- **Secondary consequence (minor): selection bias.** The final sample is a mixture: "v3 runs that stopped before 60" plus "fresh v4 runs replacing those that did not". That over-represents runs that stop early relative to a clean v4 sample. The bias is probably tiny (it affects 27 of 40 TON-IoT random-split networks, and their F1 does not depend strongly on stopping epoch), but it is a design asymmetry.
- **Secondary consequence: pairing by seed is nominal.** The "within-seed" pairing of semantic runs (v4) with random runs (mostly v3), and the claim "the same seed is used for every architecture", give little real pairing when one seed does not reproduce. The paired t-tests remain valid as tests on differences of independent runs, but the pairing gains nothing.
- **Fix:**
  - Replace the sentence with an accurate one, for example: "these runs are independent replicates under an equivalent protocol; with GPU non-determinism, same-seed re-runs differ (51 re-run pairs: median stopping-epoch change N, mean |Δ macro-F1| 0.003, max 0.018)."
  - Alternatively, re-run the 104 + 15 runs. The 15 N-BaIoT runs are cheap.

### M3. Two valid replications of the UNSW-NB15 main experiment disagree on the descriptive claims (major, interpretation)

- **Why v3 counts as a replication:** every v3 UNSW-NB15 main CNN run (25 runs) stopped early (best epoch ≤ 30), so all are valid under the final protocol. v3 and v4 are therefore two independent 5-seed replications of the identical protocol.

| UNSW-NB15 | v3 (valid replicate) | v4 (reported) |
|---|---|---|
| Late − early | +0.007 [−0.002, +0.016], 4/5, p = 0.090 | +0.016 [+0.002, +0.031], 5/5, p = 0.038 |
| Attention − early | −0.003 [−0.012, +0.006], 2/5 | +0.008 [−0.004, +0.021], 3/5 |
| Late − attention | +0.010, 5/5 | +0.008, 5/5 |
| Semantic − random, late fusion | +0.006 (4/5) | +0.009 (5/5) |

- **Pooled result:** over all 10 runs, late − early = +0.012 (p = 0.006) and attention − early = +0.003 (p = 0.42).
- **Impact:** the confirmatory conclusion ("no UNSW-NB15 comparison significant after Holm") is unaffected. But several descriptive statements in §5.2 and the Abstract rest on one draw that the project's own replicate does not reproduce:
  - "+0.016 … (5/5)"
  - "late minus early fusion becomes +0.025 [+0.001, +0.048]" on the clean subset (l.144)
  - the sign of attention − early
- **Fix:** report the replication. Either pool the 10 runs, or add one sentence to §6.3 that run-to-run variation at a fixed seed is about 0.01 on UNSW-NB15. Avoid leaning on "5/5" for UNSW-NB15.

---

## 3. Minor findings

### m1. Table 1 still says UNSW-NB15 stream 1 contains "mean size per direction" (minor; R1 item 8 not fixed)

- **Where:** Manuscript l.57; `compute_values.py:153`.
- **Code:** `streams.json` gives `[sbytes, dbytes, sload, dload, smean, ackdat]`. Only `smean` (source) is selected; `dmean` is not.
- **Fix:** write "source mean size". §3.2's "mean packet size" is fine.

### m2. The quantile-transform description does not fit N-BaIoT (minor)

- **Where:** Manuscript l.77: "fitted on a 200,000-record subsample of the oversampled training set".
- **Code:** `_finish` (`run_p6_v2.py:61`) fits on `Xfit`.
  - On TON-IoT (293,550 rows) and UNSW-NB15 (363,220 rows), `Xfit` is the SMOTE set and is subsampled to 200,000. The text is correct there.
  - On N-BaIoT, `Xfit` is the un-oversampled training set (110,227 rows), so it is neither oversampled nor subsampled.
- **Fix:** add "(all 110,227 training records on N-BaIoT)".

### m3. Tree invariance to the quantile transform is still overstated (minor; R1 item 9 not fixed)

- **Where:** Manuscript l.81 and l.240: "Tree predictions do not depend on monotone transforms…" and "which is monotone and therefore leaves the RF unchanged".
- **Why it is overstated:**
  - Split thresholds are midpoints, which move under a nonlinear map.
  - `QuantileTransformer` clips values outside the training range and maps ties and plateaus non-injectively.
  - The trees were in fact never trained on quantile inputs.
- **Fix:** "would change them at most marginally".

### m4. PR-AUC and attack detection rate are promised but never reported (minor)

- **Where:** §4.3 (l.93): "we also report … the attack detection rate and the macro-averaged area under the precision–recall curve".
- **Code:** both are computed (`metrics`, `run_p6_v2.py:202-218`), and `values.json` holds `*_prauc` and `*_det`. No table or sentence in §5 reports either.
- **Fix:** add a PR-AUC column to Table 2 (values exist for every non-reused run), or remove the promise.

### m5. Table 2's "Params" column is TON-IoT-only but not labelled (minor; R1-L8 not fixed)

- **Where:** `compute_values.py:180`; Manuscript Table 2 (l.101).
- **Detail:** parameter counts differ by dataset:
  - MLP early fusion: TON-IoT 29,770 vs UNSW-NB15 27,850
  - MLP attention fusion: 60,108 vs 58,188
  - CNN late fusion: TON-IoT 85,770 vs N-BaIoT 85,641
- **Fix:** label the header "Params (TON-IoT)".

### m6. GBDT on UNSW-NB15 runs to its iteration cap rather than stopping early (minor)

- **Where:** `run_p6_v2.py:309` (`max_iter=1000`); Manuscript l.81 ("up to 1,000 iterations with early stopping").
- **Results:** `n_iter` for UNSW-NB15 GBDT on all features is 986–1000 over 5 seeds. `gb_s1` and `gb_s2` also reach 1000 on some seeds.
- **Why:** the internal 10% early-stopping split is drawn from the SMOTE set, whose synthetic neighbours keep improving the internal loss.
- **Impact:** any bias works against the trees, so "trees beat every network" is unaffected. The cap also explains the 1,000 × 10-class model behind the 115 ms per-record latency.
- **Fix:** state that the UNSW-NB15 GBDT hit the cap.

### m7. Latency benchmark: thread settings differ between model families and are not reported (minor)

- **Where:** `bench_latency.py:50`, `:64-67` and `:76-80`.
- **Thread settings:**
  - TensorFlow uses its default intra- and inter-op pools (all 8 logical cores).
  - The RF is timed per record with `n_jobs=1` and for batches with `n_jobs=-1`.
  - The HistGradientBoosting classifier uses OpenMP on all cores for both modes, which probably inflates its single-record time (19–115 ms).
- **Reporting:** `latency.json` records `"threads": 8`, but this is `os.cpu_count()`, not a setting.
- **Docstring error (R1 item 10, not fixed):** the docstring (`:8`) still says "median of 20 calls". With `repeats=1`, `time_calls` returns the mean of 20 calls.
- **Isolation, verified:**
  - Warm-up is 5 calls of each input shape before timing, which covers `tf.function` tracing.
  - `p6lat_queue.log` shows START 06:44:29 and DONE 06:55:09 on 3 Nov. The last other job (P3 budget) ended at 06:44:12. No other result or log file was written in that window, and the next job (Paper 1 baselines) began writing at 06:56:43.
  - So "no other jobs running" is supported.
- **Model sizes:**
  - The CNN size is `params × 4 bytes`, which includes BN moving statistics. That is fine, but the unit is KiB, labelled "kB".
  - The RF and GBDT size is pickle length in MiB, labelled "MB".
- **Fix:** state the thread settings and unify the size units.

### m8. Resplits reuse features selected on split 0, whose training rows overlap the resplit test sets (minor)

- **Where:** `run_p6_v2.py:96-104`; Manuscript l.89.
- **Detail:** MI and RFECV were fitted, with labels, on split-0 training rows. Roughly 70% of each new split's test records are among them.
- **Impact:** this is mild label leakage into feature selection for the resplit replicates. It applies to all models alike and is unlikely to change rankings.
- **Fix:** the text says the features were kept, but should name this consequence in a clause.

### m9. Resplit SMOTE is built without the UNSW-NB15 `smote_max_per_class` (minor; no effect)

- **Where:** `run_p6_v2.py:103`.
- **Detail:** `pipe.__class__(seed=split, smote_k=pipe.smote_k)` drops `smote_max_per_class=50000`, which is set in the UNSW-NB15 `pipeline.pkl`.
- **Why there is no effect:** the UNSW-NB15 majority class (36,322 training rows) is below the cap, so capped and uncapped SMOTE both balance to the majority (10 × 36,322 = 363,220). The resplits are equivalent.
- **Risk:** this would silently diverge on a dataset whose majority class exceeds 50k.

### m10. Random-split and resplit summaries mix 5-seed means with single-seed values (minor; R1-L1 still applies)

- **Random splits:** `compute_values.py:380` counts "semantic above n of the ten" using the 5-seed semantic mean against single-run random values. This affects l.182: "semantic above 8 of the ten", "9 of the ten", and "semantic attention fusion is above all ten random splits" on N-BaIoT.
- **Resplits:** `split_vals` (`:275-280`) puts the split-0 seed mean beside single-seed resplit values. These "six split-level observations" (l.150) have unequal variances.
- **Text:** both are labelled descriptive, which is acceptable. The "above all ten" phrasing should be softened, or compared per seed.

### m11. The Holm family can still shrink silently (minor; residual of R1-B4)

- **Where:** `compute_values.py:231-271`.
- **Detail:** the family is now hard-coded (4 comparisons × 2 datasets), and `values.json` reports `holm_family_size = 8`, matching §4.3. But if a member's result files are missing, `paired()` returns None, the member drops out, and Holm adjusts over fewer tests without an error.
- **Fix:** `assert m_ == 8`.

### m12. The "better single stream" baseline is chosen on test means (minor; R1-L3 unchanged)

- **Where:** `compute_values.py:123-124` and `:260`.
- **Detail:** this is descriptive and has little effect (on TON-IoT and UNSW-NB15 stream 1 is better by a wide margin). It remains a selection on test.

### m13. Undisclosed pre-processing details (minor)

- **One-hot encoding (`loader.py:117-128`):**
  - Category sets and the 30-value cardinality rule are decided on the full dataset before the split. This is label-free but not train-only.
  - `drop_first=True` removes one level per field.
- **N-BaIoT (`prepare_nbaiot.py`):**
  - Scaling clips only the lower bound, so test values can exceed 1.
  - 17 conflicting-label rows are dropped.
  - Deduplication keys include the device, but the cross-split guard does not. This is harmless: the guard is the stricter of the two.
- **Statistics provenance (R1-L9, unchanged):** `data_stats.json`, the source of Table 1 counts and the TON-IoT stream-1 medians and SDs, has no generating script in the repository. I re-checked its counts against the clean masks and `yte_split0.npy`; they agree.

### m14. Provenance of v3: runs predate the commit that is supposed to contain their code (minor)

- **Timing:** `results_v3` runs were written from 31 Oct 17:00 to 1 Nov 19:17. Commit `f5b08e9` (the v3 code) is dated 1 Nov 05:20. So the main runs and most random-split runs were produced by an uncommitted working copy.
- **Consistency evidence:**
  - parameter counts are identical;
  - JSON keys match the `f5b08e9` `run_cnn` (including `latency_ms`);
  - the git diff from v2 adds only the quantile transform and a control group.
- **What cannot be proven:** that the working copy at launch equalled `f5b08e9`.
- **Also:** the v3 RF files are all time-stamped 31 Oct 17:00:00, which suggests they were copied from v2. That is acceptable, because RF inputs did not change.
- **Fix:** state in the release notes that v3 RF results were copied from v2.

---

## 4. Checks that pass

### Data and splits

- **Loader (`loader.py`):**
  - Identifier columns are dropped.
  - Deduplication happens after all column drops.
  - The stratified 70/15/15 split uses seed 42 (+k for resplits).
  - Train-only statistics: >40% missing, median, p99, zero variance.
  - The cross-split guard compares all columns plus the label, before feature selection. §3.1 now states this order correctly (R1-B2 fixed in the text).
- **Feature selection and SMOTE:** MI > 0.05, then RFECV (50-tree RF, 3-fold, macro-F1, 50k subsample, minimum 10 features). The scaler is fitted on the full training split. SMOTE is applied to training only.
- **UNSW-NB15 RFECV:** the cross-validated score is 0.554 at 10 features and 0.526 at all 40, with the argmax at the floor. This matches §3.1.
- **Duplicate shares:** the clean mask (`_finish`, `:69-70`) compares selected, scaled float32 vectors of test records with pre-SMOTE training records, ignoring the label.
  - TON-IoT 0.13% (text: 0.1%); UNSW-NB15 16.27% (text: 16.3%); N-BaIoT 0%.
  - Every class keeps at least 16 clean test records, so `clean_macro_f1` with `labels=range(K)` never scores an empty class as 0.
  - Setting clean = macro for the reused N-BaIoT runs is valid.

### Streams and models

- **Streams match the text.** TON-IoT stream 2 contains ports, DNS, protocol, service and connection-state fields. N-BaIoT is split 30 + 85 (MI_dir + H against HH, HH_jit and HpHp).
- **Random splits:** `rng(1000+r)` with the semantic sizes and seed r mod 5. Column orders use `rng(2000+r)` with seed r.
- **Architectures and parameter counts match §4.1, §4.2 and Table 2:**
  - early 34,634; late 85,770; attention 73,036; wide 86,613; early with head 43,594; late on all features 85,770 (the CNN encoder size does not depend on input length)
  - MLP 27,466 / 28,874 / 29,770 / 72,842 / 60,108 (TON-IoT)
  - I recomputed these by hand from the layer definitions.

### Training

- Adam with LR 1e-3, ReduceLROnPlateau (patience 4, ×0.5, minimum 1e-5), batch 512, early stopping on `val_loss` with patience 10, budget 120.
- No class weights; training uses `Qfit` and `yfit`.
- Seeds come from `tf.keras.utils.set_random_seed` (initialisation, shuffle and dropout). Op determinism is off, as the text admits.

### Trees

- The grids match §4.2. Tuning is on validation macro-F1 with all features and seed 0, then fixed for each stream and resplit.
- Tuned settings:
  - TON-IoT: RF unlimited depth, leaf size 1; GBDT lr 0.05, 127 leaves.
  - UNSW-NB15: RF depth 20, leaf 1; GBDT lr 0.05, 31 leaves.
- Per-run predictions are mapped through `classes_`.

### Metrics and statistics

- Macro-F1 covers all classes, all of which are present in the test set. FPR, detection rate and one-vs-rest macro AP are standard.
- Paired t-tests, t-based 95% CIs, the 1e-6 tie rule, Welch's test for tree-vs-network comparisons, Holm step-down over 8 tests, within-seed semantic − random differences, Spearman correlations and the two-sided permutation test (20,000 permutations) are all implemented as §4.3 describes.
- Holm-adjusted p-values quoted in the text match `values.json`:
  - TON-IoT late − early 0.006; late − wide 0.009; late − late-all 0.135
  - UNSW-NB15: 0.135, and 0.076 for late − late-all

### Attention analysis

- The weights are read from the softmax layer, and column 0 multiplies e1.
- The ablation replaces the test-mean embedding and keeps the weights.
- `attn_base_check` = 1.0000 in all 15 attention runs, i.e. the re-assembled head reproduces the model's predictions exactly.

### Latency and size

- The CPU-only benchmark runs untrained CNNs in inference mode, and the trees are fitted with their tuned settings.
- The ratio of RF to late fusion per record is 14 on TON-IoT.
- RF sizes are 211 and 607 MB; UNSW-NB15 has 4,416,482 nodes.

---

## 5. Recommended actions (in order)

1. **M1:** label Table 7 correctly, or rebuild its quantile row from v4.
2. **M2:** rewrite the l.85 justification for reuse using the measured same-seed variability. Ideally re-run the 15 N-BaIoT runs and the 104 random-split runs so that every reported run comes from the v4 code.
3. **M3:** disclose the v3 replication for UNSW-NB15, or pool it. Tone down "5/5" and the clean-subset CI for UNSW-NB15.
4. **Minor wording fixes:** m1–m4, m6 and m7.
5. **Code hygiene:** m9 and m11. Add a generator for `data_stats.json` (m13).
