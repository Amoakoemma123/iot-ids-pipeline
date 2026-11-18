# Paper 5 v2: methods-vs-code audit

Date: 2026-10-24. Read-only audit. No files were edited and nothing was retrained.

**Files checked**
- `Revision_v2/Paper5/Manuscript_filled.md`, plus `Manuscript.md` for which numbers are placeholders and which are typed in by hand
- `Papers pipeline/experiments/paper5_xai/run_xai_v2.py`
- `src/data/loader.py`, `src/data/preprocessing.py`
- `experiments/baseline/run_experiment.py`, `config.py`
- `src/models/cnn.py`, `random_forest.py`, `src/training/trainer.py`
- `results_v2/<ds>/*.json`
- `Revision_v2/Paper5/compute_values.py` and `values.json`

**Checks I ran** (scripts are in the session scratchpad):
- recomputed every summary statistic from the per-row JSON
- paired SHAP-vs-LIME deletion differences
- the chance level of the Jaccard index
- conversion of Jaccard to the number of shared features
- the UNSW conflict count from the raw CSV
- residual duplicates after the v2 loader and the saved `pipeline.pkl`, which is under 100 MB
- whether single-field counterfactual copies give feature combinations that occur in real data (TON and UNSW)

Line numbers below refer to `run_xai_v2.py` unless another file is named. "MS" means Manuscript_filled.md.

---

## HIGH severity

### H1. The counterfactuals are not "records that can occur". The search copies fields one at a time, so dependent fields become inconsistent.
- **Paper.**
  - §2.3: "each counterfactual is a record that can occur".
  - §4.6: restricting changes to observed values means "the reported evasions correspond to traffic that can exist" (also in contribution 3).
  - §5.3: the TTL story ("simply give the attack the TTL of benign traffic").
- **Code** (`cf_search`, L314–331; `field_groups`, L127–135):
  - Only one-hot dummies of one source column move together.
  - Every other field is copied on its own into the attack record. Fields that are functions of each other are not kept consistent.
  - **UNSW examples:**
    - `ct_state_ttl` is derived from `sttl`, `dttl` and `state`.
    - `smean` = `sbytes`/`spkts`.
    - `sload` and `dload` are derived from bytes and duration.
  - **TON examples:**
    - `proto`, `service` and `conn_state` must be compatible.
    - `src_ip_bytes` must be at least `src_bytes`, and at least 20 bytes times `src_pkts`.
- **What I measured.** I took random training attacks and random benign donors and copied one field, exactly as `cf_search` does with k = 1:

  | Dataset | Field copied | Result |
  |---|---|---|
  | TON | `proto` only | Of the copies that change the value, **67%** give a `proto`×`service`×`conn_state` combination never seen in training |
  | TON | `service` only | 63% unseen combinations |
  | TON | `src_ip_bytes` only | **20%** have fewer than 20 IP bytes per packet (0.00% in real data); 18% have `src_ip_bytes` < `src_bytes` (0.56% in real data) |
  | UNSW | `sttl` (benign value onto an attack record) | The resulting (`sttl`, `ct_state_ttl`) pair occurs anywhere in training only **23.8%** of the time |

- **Why it matters.** These field patterns are exactly the most common routes in Tables 6 and 7:
  - The TON CNN changes only `proto` in many k = 1 counterfactuals (`proto` is its top field for 5 of the 9 classes).
  - For UNSW, `sttl` appears in 85% of RF and 46% of CNN counterfactuals, mostly with k = 1.
  - The RF *generic* class needs `sttl` and `ct_state_ttl` together (49 of 50 with k = 2). That fits the model having learned the joint pattern, while the CNN is "evaded" by inconsistent records.
- **Effect.**
  - The 100% evasion rates and the fields-changed numbers (k) are overstated as "feasible" evasions.
  - The claim that "an attacker could evade them by changing one header field" (§5.3) is not supported. Changing the TTL on the wire would also change `ct_state_ttl`.
  - The hardening result in §5.4 inherits the problem.
- **Fix.**
  1. Copy dependency-closed blocks:
     - UNSW: {`sttl`, `ct_state_ttl`}, {`sbytes`, `smean`, `sload`}, {`dbytes`, `dload`}
     - TON: {`proto`, `service`, `conn_state`}, {`src_bytes`, `src_pkts`, `src_ip_bytes`}, and likewise for the dst side
  2. Alternatively, reject a candidate whose categorical combination is unseen, or that breaks the byte and packet inequalities.
  3. Report a plausibility rate for each counterfactual, for example the distance to the nearest real record compared with the distances between real records.
  4. Rerun `counterfactual` and `hardening`.
  5. If no rerun is possible, remove "records that can occur" and "traffic that can exist", and describe the constraint as per-field marginal feasibility only.

### H2. Jaccard is misread as "fewer than half of the top-5 features shared", and no chance level is given
- **Paper.**
  - §5.2: "SHAP and LIME share on average fewer than half of their top-5 features".
  - §6.2: they "usually disagree on most of the top-5 features".
  - §7: they "agreed on fewer than half of the top-5 features".
- **What the numbers mean.** Jaccard 0.39 is *not* fewer than half of 5. For two 5-sets, |A∩B| = 10J/(1+J). Converted row by row from `local.json`:

  | Dataset | Model | Mean features shared (of 5) | Alerts sharing at least 3 of 5 |
  |---|---|---|---|
  | TON | RF | **2.70** | **65%** |
  | TON | CNN | 2.13 | 34% |
  | UNSW | RF | 2.82 | 62% |
  | UNSW | CNN | 2.60 | 56% |

  So for 3 of the 4 model/dataset pairs the explainers share more than half of their top-5 on average. The statements in §5.2, §6.2 and §7 are wrong except for the TON CNN.
- **Chance level.** The expected Jaccard@5 for two random 5-subsets is **0.12 for TON** (25 features) and **0.31 for UNSW** (11 features).
  - UNSW agreement (0.42 and 0.37) is therefore only modestly above chance.
  - TON agreement (0.39 and 0.29) is far above chance.
  - The paper compares the two datasets in raw Jaccard, which is not a like-for-like comparison.
  - The same applies to LIME self-agreement (UNSW 0.80 against TON 0.70) and the ≥0.75 shares, and to the deletion test at k = 5, which removes 45% of the UNSW features but 20% of the TON features.
- **Effect.** This changes the wording of a headline conclusion. The numbers themselves are correct.
- **Fix.**
  - Report overlap@5 (|A∩B|/5) or the mean number of shared features alongside Jaccard.
  - Add the random-set baseline, or report (J − J_chance)/(1 − J_chance).
  - Rewrite §5.2, §6.2 and §7 accordingly, for example "SHAP and LIME share about 2.7 of 5 top features for the RF, well above chance on TON but close to chance on UNSW".

### H3. Hardening is not evaluated "on the same held-out test attacks" and favours the hardened model in more ways than the paper admits
- **Paper (§4.7).** "We then repeat the counterfactual search against the original and the retrained model on the same held-out test attacks."
- **Code** (`evasion`, L394–407):
  1. The query pool is the attacks correctly detected by *each* model (`p == ci`, L401), so the two pools differ.
  2. Both runs use a generator seeded with `SEED+11`, but `cf_search` consumes a number of draws that depends on when the search stops. After the first class (backdoor) the random streams diverge. Even with identical pools, the two models are queried on **different records** for 8 of the 9 classes.
  3. The hardened model is evaluated with the **same benign donor pool** (`D['X_tr']` normals), the same fields, the same 400-try random search and the same k ≤ 4 that produced its training augmentations. §6.3 admits only "the same search".
  4. Single seed and no confidence intervals: n = 50 per class, so ±0.14 at p = 0.5.
- **Rounding.** `mean_evasion_before` for TON is 0.9978 (backdoor 0.98), but `compute_values.py` L197 rounds it to "100%". The abstract and §5.4 say "from 100% to 64%"; it should read 99.8%.
- **Effect.** The direction is probably robust, since evasion falls in 8 of 9 classes. The 64% figure is not a paired comparison and is optimistic for the hardened model.
- **Fix.**
  - Draw the query set once from attacks detected by **both** models.
  - Seed a separate generator for each record (e.g. `default_rng([SEED, ti])`) so both models see identical candidate streams.
  - Evaluate with a **held-out donor pool** (benign validation records) and with a larger or different budget.
  - Report per-class CIs and McNemar on the paired outcomes.
  - Print one decimal, or say "99.8%".

---

## MEDIUM severity

### M1. CNN SHAP is stochastic and unseeded, but the paper calls SHAP deterministic
- **Paper.**
  - §4.4: "SHAP for these models is deterministic given its background, so its stability is not at issue."
  - §6.2: "(it is deterministic …)".
  - §4.9: "All runs use seed 42."
- **Code** (`shap_cnn`, L158–166): it calls `GradientExplainer.shap_values` with its defaults, `nsamples=200, rseed=None`.
  - In SHAP 0.49.1 (`shap/explainers/_gradient.py`, L317–318 and L562–563), `rseed=None` becomes `np.random.randint(...)` from the unseeded global NumPy generator. Each value is then a Monte-Carlo estimate over random background points and interpolation points.
  - CNN SHAP is therefore neither deterministic nor reproducible run to run.
  - TreeSHAP for the RF is deterministic.
- **Effect.**
  - The recommendation to use SHAP because it is deterministic holds only for the RF.
  - The CNN SHAP top-k sets used for Jaccard, Spearman and deletion carry unmeasured sampling noise, which slightly *understates* the CNN's SHAP–LIME agreement and SHAP faithfulness.
- **Fix.**
  - Pass `rseed=SEED` and a larger `nsamples`.
  - Measure CNN SHAP self-agreement exactly as for LIME (20 repeats with different `rseed`).
  - Correct §4.4, §4.9 and §6.2.

### M2. The UNSW conflict count (4,845) is wrong
- **Paper (§3.3).** "A further 4,845 records share identical features with a record of a different label."
- **compute_values.py L57–59.** It computes `feat_only.duplicated(keep=False).sum() − nid.duplicated(keep=False).sum()` on the *raw* file. That difference is not the number of conflicting records: in a group {A, A, B} it gives 1 where the true count is 3 raw rows, or 2 after deduplication.
- **Recomputed.**
  - In the deduplicated data, **8,472** records share their feature vector with a record of a different `attack_cat`.
  - Only **458** of these conflict on the binary benign/attack label.
  - In the raw file, 30,528 rows fall in such groups.
- **Effect.** A reported number is wrong by 43%.
- **Fix.** Use `dd.drop(columns=['label','attack_cat']).duplicated(keep=False).sum()` on the deduplicated frame. Consider also quoting the 458 binary conflicts.

### M3. Deduplication is still incomplete, and there is notable duplication in the model's feature space
- **Paper.** §3.3 and §3.4 present the v2 data as corrected ("real dedup"), and the 43.2% overlap as the problem that was fixed.
- **Code (loader.py).**
  - Duplicates are dropped at step 4 (L98–101).
  - High-cardinality categoricals, notably UNSW `proto` with more than 30 values, are dropped only afterwards (L121–124).
  - Records that differ only in `proto` therefore survive as exact duplicates.
- **Measured** (v2 loader plus the saved UNSW `pipeline.pkl`):
  - In the 59-column cleaned space, **1.9%** of UNSW test records have an exact (features and label) copy in train.
  - In the **11 selected features the models actually see**, **12.6%** do.
  - By class: generic **74.6%**, backdoor 55.2%, analysis 52.7%, dos 25.0%, reconnaissance 23.8%.
  - TON: 0.1%.
- **Effect.** UNSW per-class performance for generic, backdoor and analysis still partly measures recall of seen inputs. The baseline figures and the claim of corrected data should state this. It probably shifts UNSW macro-F1 a little; I did not retrain to check.
- **Fix.**
  - Deduplicate after the final column set is fixed (move step 4 after L124), or deduplicate again on the selected features before splitting.
  - Report the model-space overlap.
  - Optionally use a group-aware split on the feature hash.

### M4. The deletion test is aligned with SHAP's reference distribution, so "SHAP more faithful than LIME" is partly built in
- **Paper (§4.5 and §5.2).** "SHAP names features that matter … more precisely than LIME. SHAP leads LIME at every k."
- **Code.**
  - Deletion replaces features with the training mean (L248, L271).
  - Interventional TreeSHAP and GradientExplainer both define attributions relative to a 200-record training background. Their attribution approximates exactly the effect of moving features towards the background.
  - LIME (L169–172, with `discretize_continuous=False` and the default `sample_around_instance=False`) samples **N(train mean, train sd) for every feature, ignoring the instance**. It also gives one-hot columns continuous Gaussian values: the same "fractional one-hot" infeasibility the paper criticises in DiCE (§4.6).
  - Top-k is chosen by |attribution|. Features pushing against the predicted class are also removed; this happens for 15% of CNN TON records at k = 1.
- **Paired check (positive).** SHAP − LIME paired bootstrap differences are > 0 with the 95% CI excluding 0 at every k, for both models and both datasets. For example, TON RF k = 3: +0.041 [0.020, 0.061]. The ordering claim holds as stated. The unpaired CIs in Tables 4 and 5 overlap, so the paper should quote the paired ones.
- **Fix.**
  - Add a deletion variant that does not share SHAP's reference, for example replacement with values from a random benign donor, or the marginal-sampling average.
  - Report the paired differences.
  - State the LIME sampling settings: perturbations centred on the training mean, no categorical handling.
  - Consider passing `categorical_features` for the one-hot columns.

### M5. On UNSW a third of the explained records are misclassified, and per-class means are by true class
- **Code.**
  - `stage_local` stratifies by true class (L244), explains the *predicted* class (L260, L262) and groups per-class results by **true** class (L282).
  - In `local.json` the misclassified share is TON RF 1%, TON CNN 6%, **UNSW RF 33%** and **UNSW CNN 36%**.
  - The summary means are class-balanced, not weighted by traffic.
- **Effect.**
  - The UNSW agreement and deletion numbers largely describe explanations of wrong predictions.
  - The TON class range "0.19 (injection) to 0.57 (password)" is fine as by-true-class, but should be labelled as such.
- **Fix.** Disclose the misclassified share, and report the measures for correctly classified records only (or separately). State that the means are class-balanced.

### M6. Shortcut retrains are single runs, with one class tested on 19 records
- **Paper (§5.5).** "the largest per-class changes are worms 0.49 → 0.34 and dos 0.35 → 0.30".
- **Facts.**
  - UNSW worms has **19** test records, so its F1 is very noisy.
  - Each retrain is one run with no CI.
  - Feature selection and SMOTE are not redone without the removed features (`_shortcut_check`, L442). That is defensible as "otherwise identical", but it biases towards a larger drop.
- **Fix.** Bootstrap the test set for F1 CIs, or use several seeds. Avoid citing the worms change, or report its n.

---

## LOW severity / disclosure

1. **Rank intervals are truncated** (L220–221). `int(np.percentile(...))` floors interpolated percentiles, so `rank_hi` (and `rank_lo`) can be biased one rank downward. Use `np.percentile(..., method='lower')` for lo and `'higher'` for hi. Per-instance SHAP values are not saved, so I could not recompute the size of the effect. It is likely zero or one rank for some rows of Tables 2 and 3.
2. **Interaction values use a different SHAP setting** (L231). They use `tree_path_dependent`, not the interventional, background-based setting of the main analysis. §4.2 does not say so.
3. **Pre-processing details not stated in §3.2.**
   - UNSW SMOTE is capped at 50,000 per class (`config.py` L58–60).
   - One-hot encoding uses `drop_first=True`.
   - CNN training also uses ReduceLROnPlateau.
   - The CNN class weights are computed on SMOTE-balanced labels, so they are about 1 on TON, which makes "class weights" nearly a no-op.
4. **Provenance of TON results.**
   - The TON baseline artefacts (`pipeline.pkl`, models, `metrics.json`) are the v1 files dated 27 July. They were not regenerated.
   - The claim that the pipeline gives "exactly the same data" is supported: the v2 RF and CNN macro-F1 recomputed in `performance.json` match v1 to every digit.
   - The paper should say that the TON models were reused rather than retrained.
   - TON `ports.json` was written by an older version of `_shortcut_check`: its per-class key is `with_ports`, while the current code writes `with_`. Rerun it so the code and the file match.
5. **What "changed" and k mean** (L322–330, L355–356).
   - k counts the fields *selected*. A donor value may equal the original value, and co-selected fields that were not needed are counted in `top_fields`.
   - So "at least 85% of counterfactuals change `sttl`" really means "select `sttl`". This is acknowledged as an upper bound for k, but not for the field shares.
   - Fix: prune each successful counterfactual to a minimal subset, and count only fields whose value actually changes.
6. **RF and CNN counterfactuals use different records.** They use different query records (separate pools and one shared generator, L337–353). The claim "the smaller model is the easier one to evade" (§5.3) is therefore an unpaired comparison.
7. **Latent bug in `field_groups`** (L133). It groups any 0/1 feature by `rsplit('_',1)[0]`, so unrelated binary numerics that share a prefix would be merged (e.g. UNSW `is_ftp_login` becomes `is_ftp`). This does not affect the current feature sets: TON gives exactly the 18 intended fields and UNSW 11 singletons.
8. **Seeds.** Stability uses SEED+1 and hardening SEED+7/+11. CNN SHAP is unseeded (see M1). "All runs use seed 42" should be qualified.
9. **Different SHAP backgrounds.** The global and local stages each draw their own 200-record background (L201, L247). This is harmless but should be noted.
10. **Hand-typed claims in `Manuscript.md`** that are not placeholders, so check them by hand on any rerun:
    - "single-rank" intervals
    - "SHAP leads LIME at every k"
    - "backdoor evasion almost disappears, ransomware remains fully evadable"
    - "none of them attacks"
    - 0.67 (verified: v1 `consistency_results.json` jaccard@5 = 0.6667)

---

## Checked and correct

**Data and models**
- **Split and loader.**
  - `id` is dropped (loader L26).
  - Dedup comes before the split.
  - Steps 6–9 are fitted on training rows only.
  - The stratified 70/15/15 split with seed 42 is as described.
  - The MI → RFECV → MinMax → SMOTE order is fitted on train only. Validation and test are never resampled.
- **Recreated v1 split for the 43.2% figure.**
  - It reproduces the loader's split: same `train_test_split` calls, and string labels stratify the same way as label-encoded ones.
  - Hashing the raw rather than the normalised strings does not matter: the duplicate count is 67,601 either way.
- **UNSW audit numbers** (recomputed): 175,341 rows; 67,601 duplicates (38.6%); 107,740 after dedup; generic 40,000 → 4,181; 47,911 leading benign rows; generic first appears at 67% of the file; 0 duplicates while `id` was present; `sttl` values 254/31/62 cover 99.2%; `sttl` = 31 in 39,123 rows with 0% attacks; 88.8%, 62.9% and 48.2% as stated.
- **Feature and test counts.** TON 25 features and 27,533 test records; UNSW 11 and 16,161.
- **Models.**
  - RF hyperparameters match §3.4: 200 trees, depth 20, minimum leaf 2, sqrt features.
  - CNN architecture matches; it has about 23.9k parameters.
  - CNN early stopping on validation loss with patience 15.
- **Baseline F1.** RF and CNN macro-F1 in `performance.json` equal `metrics.json`. The v1 and v2 F1 values in Table 1 match the metrics files.

**Explanations**
- **Explainers.**
  - TreeSHAP is interventional, with probability output and a 200-record training background.
  - The CNN uses GradientExplainer.
  - LIME uses 1,000 samples, no discretisation and all features.
  - Local and stability explanations target the predicted class. Class indices line up between SHAP output, `predict_proba` and softmax columns.
- **Global importance.**
  - Mean |SHAP| is taken over classes on 1,000 random test records.
  - 500 bootstrap resamples; the double argsort rank computation is correct.
  - Cross-model Jaccard recomputed: TON 0.25 and 0.25, UNSW 0.43.
- **Agreement and stability.**
  - Jaccard is computed on the correct sets: top-k by |attribution| of the same record and the same class.
  - Spearman is computed on |attribution|.
  - Sample sizes: 300 per model on TON and 289 on UNSW (worms has 19); stability uses 100 records × 20 runs; the mean pairwise Jaccard and the ≥0.75 share are recomputed and match.
- **Deletion test.**
  - It replaces top-k features with the training mean and measures the drop in predicted-class probability.
  - The random baseline is 10 draws per record.
  - All values in Tables 4 and 5 recompute exactly from the rows.
- **Bootstrap.** Percentile intervals with 2,000 resamples over records, NaNs removed. The shared seed across calls does not bias the intervals.

**Counterfactuals, hardening and shortcuts**
- **Counterfactual protocol.** As described:
  - only correctly detected test attacks, up to 50 per class
  - k = 1–4 with 400 candidates per k
  - each candidate copies k fields from one random benign *training* record (not SMOTE output)
  - success means the argmax is the benign class
  - TON has 18 fields and UNSW 11
  - evadable % and k values match Tables 6 and 7
  - top-3 fields recomputed: TON RF `dst_port` > `conn_state` > `proto`; TON CNN `proto` > `dst_port` > `conn_state`
  - the 85% and 46% `sttl` shares are correct lower bounds as computed
- **Hardening construction.** Up to 200 detected training attacks per class, labelled with their true class, appended to the SMOTE-resampled training set and fitted with `clone(rf)` using the same settings and seed. No test data leaks into the augmentation. Augmented counts: 1,796 and 1,687.
- **Shortcut checks.**
  - The removed features are correct: TON `src_port` and `dst_port`; UNSW `sttl` and `ct_state_ttl`. UNSW `is_sm_ips_ports` would match "port" but is not selected.
  - Macro-F1 is the mean of per-class F1, which equals sklearn's macro.
  - 0.9565 → 0.9420 and 0.5913 → 0.5661 match.
  - Worms and dos are the two largest drops.

**compute_values.py**
- Apart from M2 (conflict count) and the rounding in H3, every placeholder maps to the right JSON field and is formatted correctly.
- `cf_fields` and `cf_top_share` rely on per-class top-5 lists. That makes them lower bounds, which the text says ("at least").
