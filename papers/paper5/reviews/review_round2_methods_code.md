# Paper 5 v3: methods-vs-code audit, round 2

Date: 2026-10-24. This was a read-only audit: no files were changed, no models were trained, and no `.pkl` or `.h5` file was loaded. The only things I ran were checks on the result JSONs and one pandas check, which ran `src.data.loader.load_dataset` on both CSVs to look for duplicates in the space of the selected features.

**What I compared**
- `Manuscript_filled.md` (cited as "MS", with line numbers) against:
  - `run_xai_v3.py` ("v3"), `run_xai_v2.py` ("v2") and `leakage_decomposition.py` ("LD")
  - `src/data/loader.py` and `src/data/preprocessing.py`
  - `experiments/baseline/config.py` and `run_experiment.py`
  - `compute_values.py`
  - `results_v3/**` and `results_v2/**`
- Git history (`f816628..b35de9e`): the TON all-groups `hardening.json` files come from the commit before `groups` was added to the output. That commit has the same logic, so those files are valid.

**Severity scale**
- **critical:** the reported result is wrong.
- **major:** the result or its interpretation would change, or a reviewer would require a fix.
- **minor:** disclosure, wording or precision.

**Overall.** I found no critical bug. The main pipeline matches the text, with these exceptions:
- one real coding bug in the UNSW dependency groups (MAJ-1)
- one claim about the data that the corrected pipeline does not meet in the models' own feature space (MAJ-2)
- several design choices that reviewers would challenge (MAJ-3 to MAJ-9)

I found no leakage from test into train through the donors, backgrounds, support model, scaler or LIME statistics: all of them are built from `X_tr`, the training rows without SMOTE, and the scaler is fitted on train only. Section 5 lists what I checked and found correct.

---

## 1. MAJOR findings

### MAJ-1. `ct_state_ttl` belongs to two UNSW dependency groups, so the groups are not disjoint (coding bug)
- **Where:** `run_xai_v3.py:71-75` and `115-124`.
  - `dependency_groups()` uses `taken` only for the one-hot or singleton pass (L125-128).
  - The pattern pass (L120-124) never excludes indices that an earlier pattern group has already claimed.
  - `'conn_history': ('ct_',)` therefore also matches `ct_state_ttl`, which `'ttl'` already holds.
- **Evidence:** `results_v3/UNSW_NB15/seed42/counterfactual.json` contains `"ttl": ["sttl","ct_state_ttl"]` and `"conn_history": ["ct_state_ttl","ct_dst_src_ltm","ct_srv_dst"]`. The same overlap appears in all-groups `hardening.json` (groups list of 5).
- **Impact on results:**
  1. MS L109 says "UNSW-NB15 has 5 groups" whose values must change together. In fact the groups overlap: choosing `conn_history` also rewrites the model's #2 feature, which is derived from TTL.
  2. **SHAP-guided mode (UNSW).** `ga` (v3 L469) sums |SHAP| of `ct_state_ttl` into both groups. This inflates the rank of `conn_history`, and SHAP-guided candidates often pick `ttl` and `conn_history` together, which changes fewer *distinct* features than k suggests. This is a plausible cause of the result that "SHAP guidance does not help on this dataset" (MS L231):
     - CNN SHAP-mode k = 2.56, against 1.79 for random
     - median queries 332 against 205
  3. **Group shares for UNSW.** "ttl (76%), tcp timing (36%), src volume (33%)" (MS L231) and the "conn history" share are blurred, because a `conn_history` copy also changes a TTL feature.
  4. The all-groups UNSW hardening rows of Table 9 are affected in the same way. The attacker-only results (`ttl`, `src_volume`) are **not** affected, because `conn_history` is not in `ATTACKER_GROUPS`.
- **Fix:**
  - Exclude already-taken indices in the pattern loop (`idx = [i for i ... if i not in taken and ...]`), or match `ct_` only with an explicit list of `ct_*` features that excludes `ct_state_ttl`.
  - Rerun `counterfactual` and the all-groups `hardening` for UNSW, three seeds.
  - Recheck the SHAP-guided sentence in MS L231.

### MAJ-2. Train/test duplicates remain in the model's feature space (13.7% of UNSW test records)
- **Paper:**
  - MS L54: "duplicates are judged on the final feature columns … any validation or test record identical to a training record is removed".
  - MS L60: "All results that follow use the corrected data".
- **Code:**
  - Deduplication (loader L137-140) and the cross-split guard (L200-221) both operate on the 59 cleaned columns of UNSW.
  - Feature selection then keeps 10 of them, so records that differ only in a discarded column collapse into exact duplicates in the space the models see.
- **Measured** (the current loader, the selected raw columns from `performance.json`, keys over features plus label; MinMax scaling preserves equality):

  | Dataset | Test records with a copy in train (features + label) | Same features, any label | Copies on all cleaned columns |
  |---|---|---|---|
  | UNSW | **13.7%** | 16.3% | 0.00% (the guard works at that level) |
  | TON | 0.1% | 0.1% | 0.00% |

  UNSW by class: generic **77.6%**, analysis **53.1%**, backdoor **48.7%**, reconnaissance 21.7%, dos 16.7%, fuzzers 12.5%, shellcode 11.0%, normal 9.7%, exploits 3.4%, worms 0%.
  - Round 1 measured 12.6% on the v2.0 data. The v2.1 change did not reduce this; it is slightly higher.
  - 10.1% of benign UNSW training records are duplicated within the benign set in the selected space.
- **Impact:**
  - The UNSW macro-F1 in Table 2 and the per-class scores for generic, analysis and backdoor still partly measure recall of inputs seen in training.
  - The UNSW counterfactual and agreement samples for those classes contain such records.
  - The statement in MS L54 is true only at the level of the cleaned columns.
  - Table 1 shows that copies were classified *less* accurately in condition A, so the bias need not be upward. It is undisclosed all the same.
- **Fix:**
  - Either rerun the guard after feature selection, on the selected features (this requires refitting the pipeline), or report the model-space overlap and the UNSW scores on the "no copy" subset.
  - Change MS L54 to "identical on all cleaned columns".

### MAJ-3. Table 1 (A vs B, C vs D) measures "id available to feature selection", not "id removed alone"
- **Where:** `leakage_decomposition.py:53-64` (`train_eval` refits MI and RFECV for each condition) and L71-76 (A is the legacy v1 model and pipeline).
- **Evidence** (`leakage_decomposition.json`):
  - A has 16 features: `id`, `dur`, `rate`, `dload`, `dmean`, `ct_srv_src`, `ct_src_dport_ltm`, `ct_dst_sport_ltm`, `service_dns` and others.
  - B has 10 features. It drops 7 of A's features besides `id` and adds `ct_state_ttl`.
  - C and D differ in more than `id`: C has `sinpkt`, D has `sload` and `ackdat`.
- **What is fine:**
  - Test-set sharing is correct. A and B use the same rows and split (`deduplicate=False`, same y). For C and D, dedup ignores `id`, and the guard removed 0 records in both (n = 15,894 each; I confirmed that D removes 0).
  - A's scores reproduce the v1 `metrics.json` to every digit, so the old split and feature alignment are sound.
- **Problems:**
  - **Wording.** The abstract (MS L9: "removing it alone lowers RF macro-F1 by 0.034") and MS L60 ("measured on identical data", "otherwise identical settings") overstate the comparison. The *data* are identical, but the *feature sets* are not. The 0.034 and 0.015 effects include the consequence of re-running selection without `id`.
  - **A is not a re-training.** It is the July v1 model, while B–D are trained now. The settings are the same (v1 `config.py`, `preprocessing.py` and `random_forest.py` are identical to the current files), but this is one draw of RF, MI and RFECV against another.
  - **One seed.** Every condition uses a single RF seed and a single selection run. The C–D gap of 0.015 is larger than the spread over RF seeds (D: 0.6050–0.6089), but selection variability is not assessed.
  - **"Most of the inflated score."** A − D = 0.029, while A − B = 0.034, which is larger than the total. That phrase in the abstract should be worded more cautiously.
- **Fix:**
  - Add a fixed-feature arm: A's 16 features minus `id`, i.e. A's feature set with `id` deleted and the RF retrained with no new selection. C's features minus `id` would do the same job.
  - Alternatively, describe the comparison as "with or without `id` available to the pipeline".
  - Optionally use three RF seeds per condition.

### MAJ-4. Neither deletion strategy is neutral between SHAP and LIME, and LIME's configuration is not disclosed
- **Where:**
  - `run_xai_v3.py:254-263`: mean replacement, plus replacement by a random training record drawn from 5,000.
  - L279-281: SHAP background of 200 random training records.
  - `run_xai_v2.py:169-172`: LIME with `discretize_continuous=False` and the default `sample_around_instance=False`.
- **The alignment problem.** Interventional TreeSHAP with a background of random training records *is* the average effect of replacing a coalition's features with values from random training records. The new "random training record" strategy (MS L105) is therefore SHAP's own reference distribution. It strengthens the alignment criticised in round 1 (M4) rather than removing it.
- **LIME's sampling.** LIME, as configured, samples every feature from N(training mean, training sd) independently of the instance, including continuous values for one-hot columns. Its surrogate is therefore close to a global linear fit around the training mean. That explains the low R² (CNN 0.06), and it handicaps LIME in any instance-centred deletion test.
- **Disclosure.** MS L89 says "without discretisation", which is not LIME's default (`True`), but does not state that the perturbations are centred on the training mean rather than on the instance.
- **Impact:**
  - "SHAP's top features are the more faithful under two deletion strategies" (MS L211 and L292) is partly built into the design.
  - The LIME recommendations in §6.2 rest on an instance-agnostic, non-default configuration.
  - The paired numbers themselves are correct (recomputed; see §5).
- **Fix:**
  - Add a strategy independent of either explainer: ROAR-style retraining, class-conditional or nearest-benign-neighbour replacement, or LIME's own sampling distribution as a symmetric counterpart.
  - Or rerun LIME with `sample_around_instance=True` and default discretisation as a sensitivity check.
  - Disclose the LIME settings in §4.1.

### MAJ-5. CNN SHAP instability depends on GradientExplainer's default `nsamples=200`; the SHAP and LIME stability comparisons are not like for like
- **Where:** `run_xai_v3.py:136-144`. `shap_values(..., rseed=rseed)` uses SHAP 0.49's default `nsamples=200`, and the paper never states this number.
- **Stability stage (L355-360):**
  - RF: 3 backgrounds, giving 3 pairs per record.
  - CNN: 5 runs, each with a new background *and* a new `rseed`, giving 10 pairs per record.
- **Impact:**
  1. "SHAP for the CNN is itself unstable across backgrounds" (abstract) and "about as unstable as LIME" (MS L209) confound:
     - background sampling
     - Monte-Carlo sampling noise, which falls with `nsamples`
  2. LIME is given a higher budget (5,000 samples) but SHAP is not. The recommendation in §6.2 ("averaging SHAP over several backgrounds and seeds") is one remedy; raising `nsamples` is the obvious other.
  3. The CNN SHAP used for agreement and deletion is one 200-sample run, so CNN SHAP–LIME agreement (0.28) is capped by SHAP's own self-agreement (0.48). This should be stated where the CNN agreement is interpreted.
  4. The RF value (0.90) captures only background variation (TreeSHAP is exact given its background), so "SHAP stable for RF, not for CNN" compares two different sources of variance.
- **Fix:**
  - Separate the two sources: (a) fixed background with varying `rseed`; (b) fixed `rseed` with varying background.
  - Repeat one CNN arm at `nsamples` = 1,000–2,000.
  - Report `nsamples` in §4.1 and §4.4.

### MAJ-6. The hardening control is close to a no-op by construction
- **Where:** `run_xai_v3.py:530-537`. The control is convex combinations of two training attacks of the same class.
- **Why it is weak.** This is the SMOTE generator, applied on top of a training set that is already SMOTE-oversampled. The added points lie inside the convex hull of each class, which the RF already covers. The control therefore tests "more in-distribution attack data", not the obvious alternative: attacks whose attacker-controllable groups are copied from benign donors, *without* the selection that they evade.
- **Impact:**
  - "The gain comes from the counterfactuals and not from more attack data" (MS L251 and L280) is supported only against this weak control.
  - With a benign-donor perturbation control, the effect could come from noise augmentation on the controllable features rather than from the counterfactual search.
- **Also:**
  - Generation and evaluation use the same `cf_search`, the same groups and the same k ≤ 4 budget. The donor pools are disjoint but i.i.d. halves of the same benign set. This is disclosed in MS L288.
  - Hardening covers the RF only.
- **Fix:**
  - Add a control arm that copies the same groups from random poolA donors with no success filter, labelled with the attack class, with matched per-class counts.
  - Optionally evaluate against a different search, for example SHAP-guided or a larger budget.

### MAJ-7. The "queries to first evasion" comparison leaves out the cost of the explanation and counts candidates the support check rejected
- **Where:** `run_xai_v3.py:425-441`. `queries = queries + t + 1` counts every candidate in batch order, including candidates rejected by the offline support check. SHAP itself is computed at L465-466 and is not counted.
- **Impact:**
  - MS L217 ("explanations are useful to an attacker mainly as a way to save queries"; CNN 22 → 4) ignores three things:
    - SHAP needs white-box access (TreeSHAP, gradients) or many model calls (KernelSHAP), so the query saving is not free.
    - An attacker would never submit support-rejected candidates. The infeasible share differs between modes (TON CNN: 0.89 random against 0.68 SHAP), which alone shifts the count.
    - Medians are computed over successes only, and success rates differ between modes (TON RF: 100% random, 87% SHAP).
  - The direction of the effect may survive, but the size of the claim needs qualifying.
- **Fix:**
  - Count only feasible candidates as queries.
  - Report SHAP's own cost in model calls, or state that it assumes white-box access.
  - Compare queries with failures counted at the budget, e.g. as a survival curve.

### MAJ-8. The attacker-controllable `ttl` group includes `ct_state_ttl`, which depends on the responder
- **Where:** `run_xai_v3.py:78-82`. The code comment says: "ct_state_ttl also depends on the reply TTL and state, so this is a generous reading".
- **What the paper says:** MS L111 ("ttl and src volume"), L231 ("The TTL is a particularly weak point, because the sender sets it") and L251 ("a benign TTL value is available to any sender").
- **Impact:** every UNSW attacker-only number rests on this generous reading. The paper presents it as sender-controlled, not as an upper bound.
  - 64–76% evadable (RF) and 68–82% (CNN)
  - hardening 75% → 46%
- **Fix:**
  - State in §4.6 that `ct_state_ttl` is also copied, and that it depends on `dttl` and the connection state.
  - Better, add an arm with `sttl` only (keeping `ct_state_ttl` fixed, or recomputing it), or restrict the donors to those with the attack's `dttl` and state.

### MAJ-9. One third of the UNSW records that are explained are misclassified, and this is still not disclosed (round-1 M5 not addressed in the text)
- **Where:** `run_xai_v3.py:276` (stratified by true class) and L294-298 (the explained class is the *predicted* one).
- **Evidence** (`local.json`, seed 42): misclassified share UNSW RF **35%**, UNSW CNN **33%**, TON RF 1%, TON CNN 6%.
  - `summary_correct_only` is computed (L328) but not reported.
  - Correct-only j5: UNSW RF 0.447, CNN 0.379, close to the pooled values.
- **Impact:**
  - The UNSW agreement and deletion results in Table 6 are largely explanations of wrong predictions.
  - The means are class-balanced, not weighted by traffic, and the paper does not say so.
- **Fix:**
  - Add one sentence giving the misclassified share and the correct-only values. These numbers already exist.
  - State that the summaries are class-balanced over true classes.

---

## 2. MINOR findings

### Method statements that do not match the code
- **min-1. The §3.3 duplicate statistics use a different dedup definition from the corrected pipeline.**
  - `compute_values.py:89-100` counts duplicates on the *raw* columns minus `id`, which includes the high-cardinality `proto`: 67,601 (38.6%), 107,740 left, generic 4,181, conflicts 8,472.
  - The pipeline (loader L137-140, after dropping `proto`) removes **69,386**, leaving **105,955** rows (test 15,894) with **generic 4,137**.
  - The 44.0% "copy in training" figure (Table 1) uses the pipeline definition, so the same paragraph (MS L58) mixes the two.
  - The `sttl` statistics (254/31/62, 99.2%, 39,123 rows, no attacks) happen to be identical under both definitions.
  - Fix: quote the pipeline numbers, or say "on all raw columns".
- **min-2. The support check is not exactly as described, and its metric is weak for volume features.**
  - `Support` (v3 L398-412) uses the nearest neighbour within a random **20,000-record subsample** of benign training records, and τ comes from 5,000 of them.
  - MS L109 says "nearest benign training record".
  - The distance is Euclidean on MinMax-scaled features. For heavy-tailed byte, load and packet features, large real changes are tiny in scaled units, so the check mainly enforces the categorical and flag profile.
  - On TON, τ = 0.0025 is very strict for flags but almost blind to volume. Within-set duplicates (UNSW benign 10.1%) push τ down.
  - Fix: state the subsample, and consider log-scaling or rank-scaling before the nearest-neighbour step.
- **min-3. The k range depends on the mode.** In attacker mode, `range(1, min(max_k, len(G))+1)` (v3 L425) caps k at 3 on TON and 2 on UNSW, not at 4 as MS L109 implies. With 2 groups at k = 2 on UNSW, both groups are always copied.
- **min-4. "Any group" mode is near-tautological on UNSW.** The 5 groups cover all 10 features, so k = 4 leaves at most one or two attack features in place and the candidate is almost the donor. 100% evasion is expected; say so next to the 100% figures (MS L215 and L231).
- **min-5. The port check retrains without re-selecting features** (`run_xai_v2.py:428-449`), whereas the TTL check re-selects. MS L119 describes both, but not this asymmetry.
  - TON `ports.json` was written by the pre-refactor `stage_ports` (key `with_ports`). That function is equivalent to the current one, so the values are valid, but the file does not match the current code.
- **min-6. The global SHAP stage (Tables 3–4) was not upgraded.**
  - `run_xai_v2.py:158-166`: the CNN `GradientExplainer` call still uses `rseed=None` (unseeded, cannot be reproduced) with `nsamples=200`.
  - The bootstrap rank intervals resample records only, so they leave out the explainer variance that matters for the CNN (self-agreement 0.48).
  - L220-221: `int(np.percentile(...))` still truncates, which can make "single-rank intervals" (MS L129) look narrower than they are. Round-1 L1 is not fixed.
  - L231: interaction values use `tree_path_dependent`, which MS L93 does not disclose. Round-1 L2 is not fixed.
- **min-7. Pipeline and training details are still not disclosed** (round-1 L3):
  - `drop_first=True` one-hot encoding
  - the UNSW SMOTE cap of 50,000 per class
  - ReduceLROnPlateau
  - CNN class weights computed on SMOTE-balanced labels, which makes them about 1 on TON
- **min-8. Seeds and provenance.**
  - The seed-42 CNN was trained without any TensorFlow seed: there is no `set_random_seed` anywhere in `experiments/baseline` or `src`. Seeds 43 and 44 are seeded (v3 L213). "Seed 42" is a nominal label for the CNN.
  - The TON seed-42 RF and CNN are the July v1 files, reused. That is valid because the TON data are byte-identical, and I confirmed that the v1 and current TON `metrics.json` match to every digit, but the paper should say the models were reused.
  - Seeds 43 and 44 vary only the model seed. Selection and SMOTE stay at 42, so "range over seeds" does not cover pipeline variability (MS L74). The text says "on the same data", which is correct.

### Correctness and precision
- **min-9. Hardening evaluation is still not paired at the candidate level** (round-1 H3.2 not fixed).
  - `erng` is re-created for each model (v3 L549), but `cf_search` consumes a number of draws that depends on when it stops, and records not detected by a model are skipped (L552). The candidate streams therefore diverge after the first record.
  - The query set is identical, but the comment "same evaluation randomness for every model" is not achieved.
  - The `hardening` and `hardening_attacker` runs use **different query sets** (n = 482 and 481 on UNSW): `sup_idx` is drawn after generation has consumed `rng`, so the two "original data" rows of Table 9 are not on the same records.
  - There is no paired test such as McNemar. Per-seed `evasion_ci` values are computed but not reported.
  - Fix: use `default_rng([seed, ti])` for each record and draw `sup_idx` before generation.
- **min-10. RF and CNN counterfactuals use different records** (v3 L456-464: separate detection pools, shared `rng`; UNSW n = 254 for the RF, 257 for the CNN). Comparisons between the two models in §5.3 are unpaired (round-1 L6).
- **min-11. Group shares count groups that were selected, not groups that were necessary.** There is no pruning to a minimal subset, so co-selected groups are counted (round-1 L5). The 88%, 76% and similar shares are therefore upper bounds.
- **min-12. Small per-class samples are reported without n.**
  - Counterfactuals: UNSW worms n = 14 (RF) and 17 (CNN) in Table 8.
  - Hardening: worms n = 13 in the query set.
  - TTL: worms has 19 test records, yet the paper cites "worms 0.53 → 0.30" (MS L272; round-1 M6).
  - Fix: add an n column, or drop worms from the narrative.
- **min-13. The stability comparisons are not like for like.**
  - LIME uses 20 runs at 1,000 samples, 5 runs at 5,000 (MS L101), and SHAP 3 runs for the RF.
  - Mean pairwise Jaccard is unbiased whatever the number of runs. The share of records at or above 0.75 (32% against 82%) and the CI widths, however, depend on the number of pairs per record (190, 10, 3 and 10).
  - Fix: use 5 runs at 1,000 samples for the comparison, or state the caveat.
- **min-14. The chance correction is a uniform-random null.**
  - The hypergeometric formula is correct: `expected_jaccard(25,5)` = 0.1209, `(10,5)` = 0.3504, `(18,5)` = 0.1740.
  - It assumes both explainers pick features at random. A more informative null pairs each record's SHAP top-5 with the *global* top-5, or uses LIME and SHAP from different records of the same class.
  - On UNSW (p = 10, k = 5) Jaccard is coarse and close to saturation. The top-3 values are computed (`chance_j3`, `j3`) but not reported.
  - MS L207 compares the field-level 0.44 with the feature-level 0.39 without the field-level chance level (0.174). Chance-corrected, both TON models are at 0.32.
- **min-15. One sentence about LIME against random features is imprecise.** MS L211 says "LIME's [features beat random] too, except for the UNSW-NB15 CNN, where under mean replacement … three of nine". Recomputed over 3 seeds × 3 k:
  - TON CNN: one mean-replacement cell is not significant (17/18).
  - UNSW CNN, mean replacement: 5/9 significant, 3 with a negative mean.
  - UNSW CNN, random record: 7/9 significant.
  - "12 of 18" is correct, but "except the UNSW CNN" is not.
- **min-16. Hardening control sampling.** `rng.choice(pool, 2)` (v3 L534) samples with replacement and can pair a record with itself. This is negligible but trivial to fix with `replace=False`.
- **min-17. Latency is a single timed call**, with the RF using `n_jobs=-1` (v3 L240 and L248). Say "single run, all cores", or take the median of 5 runs.

---

## 3. Round-1 findings: status

| Round 1 | Status in v3 | Notes |
|---|---|---|
| H1 inconsistent field-by-field counterfactuals | **Mostly fixed** | Dependency groups and the nearest-neighbour support check exist. New overlap bug (MAJ-1); the generous `ttl` group (MAJ-8); the weakness of the support metric (min-2) |
| H2 Jaccard misread, no chance level | **Fixed** | Shared counts, chance-corrected values and field level are all reported. Caveats on the null in min-14 |
| H3 hardening not paired, same donors, one seed | **Largely fixed** | Fixed query set, disjoint pools, 3 seeds, a control, benign FPR. Candidate streams still diverge and no paired test (min-9); weak control (MAJ-6) |
| M1 CNN SHAP unseeded, "deterministic" | **Fixed in v3 stages and text** | `rseed` passed; instability measured and reported. Still unseeded in global SHAP (min-6); `nsamples` not considered (MAJ-5) |
| M2 conflict count | **Fixed** | 8,472, on the raw-dedup definition (min-1) |
| M3 incomplete dedup | **Partly fixed** | Dedup now runs after the column drops, and the guard is in place. Model-space overlap 13.7%, not disclosed (MAJ-2) |
| M4 deletion aligned with SHAP | **Not resolved** | The second strategy is SHAP's own background distribution; LIME settings not disclosed (MAJ-4). Paired CIs added |
| M5 UNSW misclassified share | **Not addressed in the text** | Computed (`summary_correct_only`) but not reported (MAJ-9) |
| M6 single-run shortcut retrains, worms n = 19 | **Partly** | TTL now re-selected and the CNN added, but still one seed and worms still cited (min-12) |
| L1 rank truncation | Not fixed | min-6 |
| L2 interactions path-dependent | Not disclosed | min-6 |
| L3 pipeline details | Not disclosed | min-7 |
| L4 TON model reuse, `ports.json` provenance | Not stated / file not regenerated | min-5, min-8 |
| L5 selected vs necessary groups | Not fixed | min-11 |
| L6 RF and CNN on different counterfactual records | Not fixed | min-10 |
| L7 latent `field_groups` prefix bug | Harmless for the current features | It has a sibling: `dependency_groups` does not exclude taken indices (MAJ-1) |
| L8 seeds | Mostly fixed | The seed-42 CNN is unseeded (min-8) |

---

## 4. Method statements checked one by one (§3–4)

| MS statement | Code | Match? |
|---|---|---|
| 70/15/15 stratified split, seed 42; train-only missing filter, median, inf→p99, zero-variance; guard on val/test | loader L148-221 | Yes. The guard is at the cleaned-column level only (MAJ-2) |
| Dedup "judged on the final feature columns" | loader L137-140 | On the cleaned columns, before the train-fitted drops and before selection (MAJ-2) |
| MI → RFECV → MinMax → SMOTE; val and test never resampled | preprocessing.py | Yes. SMOTE cap not stated (min-7) |
| RF: 200 trees, depth 20, leaf 2, √p | `random_forest.py` | Yes |
| CNN architecture, 23,946 parameters, Adam, class weights, patience 15 | `cnn.py`, `trainer.py`, `performance.json` | Yes. Seed-42 CNN unseeded (min-8) |
| Three seeds on the same data | v3 L196-224 | Yes. Only the model seed varies |
| TreeSHAP interventional, probability output, 200 random training records; GradientExplainer with the same background | v2 L149-155, v3 L136-144, L279 | Yes. `nsamples=200` not stated (MAJ-5) |
| LIME without discretisation, ridge surrogate, all features, 1,000 samples | v2 L169-180 | Yes. Mean-centred sampling not stated (MAJ-4) |
| Explanations target the predicted class | v3 L294-297, L353 | Yes |
| Global: 1,000 test records, mean over classes, 500 bootstrap; interactions on 100 records | v2 L198-239 | Yes. Rank truncation and path-dependent interactions (min-6) |
| Agreement: up to 30 per class, same records for every seed; shared count, J, Spearman on \|attr\|; J₀ hypergeometric (0.12 TON, 0.35 UNSW); field level with one-hot summed | v3 L276, L285-304, L88-94, L104-112 | Yes (J₀ recomputed). n = 300 TON, 289 UNSW |
| Stability: 10 per class, seed-42 models; LIME 20×1,000 and 5×5,000; SHAP 3 RF backgrounds, 5 CNN background+seed runs; consensus = mean \|W\| of 20 runs; 0.75 share | v3 L342-385 | Yes (n = 100). Comparability caveats in min-13 and MAJ-5 |
| Deletion: k = 1, 3, 5; training mean; random training record (10 drawn from 5,000); random-k baseline with 10 draws; paired bootstrap | v3 L254-272, L305-323, L97-101 | Yes |
| Dependency groups: TON 6 (protocol state incl. DNS, src/dst volume, src_port, dst_port, duration); UNSW 5 | v3 L64-77, L115-129, `counterfactual.json` | TON yes. UNSW groups **overlap** (MAJ-1) |
| Support: nearest benign record within the 95th percentile of benign NN distances | v3 L398-412 | Approximately: 20k subsample, τ from 5k (min-2) |
| k = 1..4, 200 candidates each, stop at the first accepted | v3 L425-441 | Yes, except attacker mode caps k at the number of groups (min-3) |
| Three modes (random, SHAP top-k groups with only the donor varying, attacker groups TON src_port/src_volume/duration, UNSW ttl/src_volume) | v3 L81-82, L467-474 | Yes. `ttl` includes `ct_state_ttl` (MAJ-8) |
| Up to 30 detected test attacks per class, per model and seed; queries to the first evasion | v3 L456-464, L438 | Yes. Query counting in MAJ-7 |
| Hardening: up to 200 detected training attacks per class, labelled with the true class, appended to SMOTE data, same RF settings and seed | v3 L520-529, L562-566 | Yes (`clone` keeps `random_state`) |
| Control of the same size, convex combinations of two training attacks of the class | v3 L530-537 | Yes. Weak control (MAJ-6), with replacement (min-16) |
| Disjoint donor pools, separate support models | v3 L515-517 | Yes |
| One query set: up to 60 per class, detected by all three models | v3 L538-545, L567 | Yes. Differs between the two variants (min-9) |
| Attacker-only in both generation and evaluation | v3 L512-513 | Yes |
| Ports: TON RF retrained without `src_port` and `dst_port` | v2 L428-457 | Yes, without re-selection (min-5) |
| TTL: `sttl`, `dttl`, `ct_state_ttl` removed before selection, RF and CNN retrained; `sttl` majority rule trained on train, applied to test | v3 L585-633 | Yes |
| Bootstrap: 2,000 resamples, percentile, seed 42 | v2 L69-77 | Yes |

---

## 5. Checked and correct

- **No train/test leakage in any stage.** Backgrounds, deletion means and donors, LIME training statistics, counterfactual donors and support models, and hardening pools all come from `X_tr` (not SMOTE). The scaler, MI and RFECV are fitted on train. The test set is used only for queries and evaluation.
- **Class indices.**
  - `normal` is taken from `classes.index` (TON 5, UNSW 6).
  - The RF `predict_proba` columns are 0..C−1 for all seeds, because every class is present in `y_tr_sm`.
  - The SHAP output is `(n, f, c)`, indexed by the predicted class.
- **Denominators.**
  - Success = successes / n.
  - Group shares = counts / successes, with groups drawn without replacement, so each share is at most 100%.
  - `infeasible_share` = rejected / tried.
  - No count can exceed its denominator.
- **Seeds and models.**
  - Every stage loads the model of its own seed (`model_paths`).
  - The pipeline is always the seed-42 `pipeline.pkl`, as the text says.
  - Agreement records are the same for every seed (`stratified_idx(seed=42)`).
- **Table 1.**
  - A reproduces the v1 RF exactly (0.637520…).
  - D equals the seed-42 baseline (0.608949…).
  - A and B share 26,302 test records, with an identical unseen mask of 14,732; C and D share 15,894, with the guard removing 0 in both.
  - The UNSW loader: 105,955 rows, test 15,894.
- **TON data unchanged under loader v2.1.** The v1 and current `metrics.json` agree to every digit (RF 0.9565302995966034), and the guard removed 0 records.
- **Recomputed from the JSON:**
  - the paired SHAP-minus-LIME significance counts: RF 18/18 on both datasets; TON CNN 15/18 with all 18 positive; UNSW CNN 18/18
  - LIME beating random for the UNSW CNN in 12/18 cells
  - the attacker-only success ranges
  - the hardening numbers in Table 9: n_aug 687–757; evasion 36–46 → 13–16, control 39–43; UNSW 73–77 → 46, control 72–77
  - the median queries (TON CNN 22.5 → 4; UNSW CNN 205 → 332)
