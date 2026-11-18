# Paper 5: final check of references and of methods against code (2026-11-07)

**Scope**
- Manuscript: `Revision_v2/Paper5/Manuscript_filled.md` (38 references).
- Earlier checks I read: `literature_verified.md`, `reference_audit/refcheck.json` and `review_round2_methods_code.md`.
- Code: `Papers pipeline/experiments/paper5_xai/run_xai_v3.py`, the `run_xai_v2.py` helpers, `leakage_decomposition*.py`, `feature_space_copies.py`, `src/data/{loader,preprocessing}.py`, `src/models/{random_forest,cnn}.py`, `src/training/trainer.py` and `experiments/baseline/config.py`.
- Result provenance: `Revision_v2/Paper5/compute_values.py`, plus the `results_v3` and `results_v2` JSON files, their timestamps and the git log.
- I did not train any model or load any dataset.

**Severity scale**
- **High:** could change a reported result or headline claim.
- **Medium:** a methods statement that is wrong or incomplete in a way a referee would flag, or a bias that has not been quantified.
- **Low:** wording, precision or provenance.

**Overall**
- **References:** all 38 exist, and their metadata are correct. I found no fabricated reference and no claim that its source contradicts.
- **Methods against code:** the settings match the text in nearly every respect. I found no code bug that changes a reported number.
- **Worth acting on:**
  - one number reported under two conventions (M1);
  - one undisclosed choice in the deletion test that could favour one explainer (M2);
  - several settings that are not disclosed (M3–M5);
  - some issues from round 2 that are still open (M14–M17).

---

## References

### R0. Method and overall result
- I took every DOI in the list (32 of 38) to `api.crossref.org/works/<doi>` and compared title, authors, container, volume, issue, pages and year with the entry. All 32 match. Only the small items listed below differ.
- The six entries without a DOI are correct as cited:
  - Abadi et al. (2016), OSDI '16, pp. 265–283;
  - Arp et al. (2022), USENIX Security '22, pp. 3971–3988, confirmed on the USENIX page;
  - Krishna et al. (2024), TMLR, OpenReview id jESY2WTZCe (confirmed in the earlier audit; OpenReview blocked automated access today);
  - Lundberg & Lee (2017), NeurIPS 30, pp. 4765–4774;
  - Pedregosa et al. (2011), JMLR 12, 2825–2830;
  - Wachter et al. (2017), Harvard JOLT 31(2), 841–887.
- **Caveat on the earlier audit.** `refcheck.json` matched four non-DOI entries (Abadi, Arp, Lundberg & Lee, Pedregosa) to the *wrong* Crossref records, for example "Stochastic Amortization…" for Lundberg & Lee, and still marked them `ok` (title overlap 0.25–0.67). Those four are correct, but that audit does not verify them. I checked them by hand.
- `check_citations.py` reads `Manuscript.md`, not `Manuscript_filled.md`. I ran it on the filled manuscript too. Every in-text citation is in the reference list and every reference is cited. No `{{placeholders}}` are left.

### R1. Claim support
I checked each citing sentence against the abstract or full text, or against the content notes in `literature_verified.md`.

| Location | Claim | Verdict |
|---|---|---|
| §2.4, §5.5 | Beltiukov et al. (2023) report detectors that separate benign from malicious flows almost entirely by the TTL | **Supported.** I checked the full text (arXiv 2306.08853, §5.2): "all four models use almost exclusively the TTL (time-to-live) feature to discriminate between benign and malicious flows, which is an obvious shortcut" (rule TTL ≤ 63). |
| §2.1 | Arp et al. (2022): explanations as a check for spurious correlations, one of ten pitfalls | Supported |
| §2.2, §4.5 | Warnecke et al. (2020): descriptive accuracy = remove the top-ranked features and measure the effect | Supported |
| §2.2 | Arreche et al. (2024): stability and robustness, among other metrics | Supported (six metrics, three datasets, seven models) |
| §2.2 | Kalakoti et al. (2025) "do the same" as Kalakoti et al. (2024) (faithfulness, monotonicity, complexity, sensitivity) | **Low:** loosely worded. The 2025 paper scores faithfulness, complexity, robustness and reliability for LIME, SHAP, IG and DeepLIFT. Suggest "evaluate several explainers for faithfulness, complexity and robustness for SOC alert classification". |
| §2.3 | Kacem et al. (2025) combine an RF IDS with SHAP, LIME and DiCE | Supported per the content notes, although the title names only LIME and DiCE. Low risk. |
| §2.4 | Engelen et al. (2021) and Lanvin et al. (2023) "show that correcting them changes detection performance" | Lanvin: supported. Engelen 2021 mainly documents the errors and releases a corrected dataset. **Low:** consider attributing the performance part to Lanvin only. |
| all others | Zebin, Gaspar, Mohale & Obagbuwa, Layeghy & Portmann, Krishna, Nauta, Bodria, Kumar & Thing, Maseno, Pawlicki, Okada, Galwaduge, Flood, Cantone, Jacobs, Erion, Lundberg 2020, Chawla, Breiman, Samek | Supported |

### R2. Small metadata items (all Low)

| Ref | Item |
|---|---|
| Kacem et al. (2025) | Crossref gives the first author as "Kacem, Mohamed Aziz Ben". The surname may be "Ben Kacem". Check the IEEE Xplore author page and write "Ben Kacem, M. A." if so. |
| Lanvin et al. (2023) | The LNCS volume (13857) is missing; most Elsevier styles want it. |
| Kumar & Thing (2026) | The LNCS volume number is missing (Crossref has none yet); acceptable. |
| Moustafa & Slay (2015) | Cited for the training file of 175,341 records. That train/test partition is described in Moustafa & Slay (2016, *Information Security Journal: A Global Perspective* 25, 18–31). Optional: add it. |
| LIME | No version is given. The installed version is `lime` 0.2.0.1; add it to §4.1. SHAP 0.49 is correct (0.49.1 installed). |

---

## Methods-vs-code

### Confirmed matches (no action)
- **RF:** 200 trees, `max_depth=20`, `min_samples_leaf=2`, `max_features='sqrt'`.
- **CNN:**
  - Conv 32/64 filters, kernel 3, batch normalisation and max-pooling, then global average pooling, Dense 128/64 and dropout 0.3.
  - 23,946 parameters (recomputed by hand for 10 classes).
  - Adam; ReduceLROnPlateau with factor 0.5 and patience 7; early stopping on validation loss with patience 15; class weights from the SMOTE labels.
- **Pre-processing:**
  - Identifier and free-text columns are dropped, and categoricals with more than 30 values are dropped.
  - One-hot encoding uses `drop_first`.
  - Duplicates are removed after all column drops, and the split is stratified 70/15/15 with seed 42.
  - Missing-value filtering (>40%), median imputation, p99 replacement of infinities and zero-variance removal are fitted on training rows only.
  - The cross-split guard keys on features plus label.
  - Feature selection is MI (0.05) followed by RFECV, then MinMax scaling. SMOTE is capped at 50,000 per class on UNSW.
- **TreeSHAP:** interventional, `model_output='probability'`, background of 200 random (non-SMOTE) training records.
- **GradientExplainer:** default `nsamples=200`. Global importance: mean |SHAP| over classes, 1,000 test records, 500 bootstrap rank resamples, truncated percentiles ("rounded down" is disclosed). Interactions: path-dependent, 100 records.
- **LIME:**
  - `discretize_continuous=False`, ridge surrogate, all features, 1,000 samples.
  - The contribution w·(x−μ)/σ uses LIME's own `scaler.mean_` and `scale_`. I confirmed in the installed `lime_tabular.py` (L257, L348) that the surrogate is fitted on (x−mean_)/scale_, so the contribution formula and the "slope per standard deviation" statement are correct.
- **Agreement:** top-k by |attribution|. Jaccard, shared count and Spearman are computed on |attributions|. The hypergeometric chance level is correct: 0.1209 for (25, 5) and 0.3504 for (10, 5).
- **Stability:**
  - 10 records per class.
  - LIME: 20 runs at 1,000 samples and 5 runs at 5,000.
  - SHAP: 3 backgrounds for the RF; 5 background–seed pairs for the CNN.
  - Budget stage: CNN at 200 and 1,000 samples with a fixed background and with new backgrounds; RF backgrounds of 200 and 1,000.
- **Deletion:** four strategies, 10 records from 5,000 donors, 10 Gaussian draws, 10 nearest neighbours on the features that are not replaced, random baseline of 10 draws, paired bootstrap. Tables use `condx` (exact copies excluded).
- **Counterfactuals:**
  - TON-IoT has 6 groups and UNSW-NB15 has 5; this matches the JSON outputs. On UNSW the `ct_state_ttl` overlap bug from round 2 is fixed and the UNSW counterfactuals were re-run.
  - Attacker groups: TON = `src_port`, `src_volume`, `duration`; UNSW = `sttl` plus `src_volume` (`sbytes`, `sload`, `smean`).
  - k = 1..4 with 200 candidates each; k is capped by the number of allowed groups.
  - Support check: 95th-percentile nearest-neighbour radius.
  - SHAP-guided search takes the top-k groups ranked by summed |SHAP|.
- **Hardening:**
  - Up to 200 correctly detected training attacks per class; disjoint donor pools A and B.
  - Interpolation control and copy control: same per-class counts and same k distribution, 50 tries, last candidate kept.
  - Query set = up to 60 detected test attacks per class, intersected over the four models.
  - The `hardening_attacker` and `hardening_attacker_r3` outputs are identical, so the run is reproducible and the 25–30 and 0–2 unsupported-copy counts belong to the same runs as Table 9.
- **Leakage decomposition (fixed features):** one load of the data per pair, the same 10 features with and without `id`, a paired record-level bootstrap and three RF seeds. The corrected pair reproduces the baseline seed-42 RF (0.6089, n_test 15,894).
- **Feature-space copies:** exact float32 equality on the scaled selected features, with label (13.7%) and without label (16.3%). F1 without copies excludes the features-only copies, as the text says.

### Findings

| # | Location | Evidence | Severity |
|---|---|---|---|
| **M1** | Abstract ("raised RF accuracy by 0.053 with duplicates and 0.051 without them"); §3.3 | `leakage_decomposition_fixed.json`, accuracy effect per seed: duplicates kept 0.0528, 0.0528, 0.0526; duplicates removed **0.0497 (seed 42)**, 0.0512, 0.0517. Text and abstract quote the seed means (`decf_*_acc_effect_mean`), but the macro-F1 effects in the same sentence are seed-42 values. Table 1 is also seed 42, and gives 0.8499 − 0.8002 = **0.050**. A reader who checks against Table 1 finds 0.050, not 0.051. | **Medium** |
| **M2** | §4.5 deletion test | `topk()` (`run_xai_v2.py` L85) ranks by **absolute** attribution, so the "top-k features" that are replaced include features with *negative* attribution for the predicted class. Replacing those should *raise* confidence. The text does not say this. If one explainer puts more negative-signed features in its top-k, its measured drop shrinks. That is plausible for LIME contributions, whose surrogate fits poorly (R² 0.06–0.30). The SHAP > LIME gap may therefore be partly an artefact of signed versus unsigned ranking. The rows store only `shap_abs`, so this cannot be checked from saved results. **Fix:** at minimum, state "top-k by absolute attribution". Better, add a robustness row that replaces the top-k *positive* attributions only, on seed 42 (cheap: deletion stage only). | **Medium** (could change the size of the SHAP–LIME gap; unlikely to reverse the direction) |
| **M3** | §4.1 LIME settings | The kernel width is the LIME default, 0.75·√p (3.75 on TON-IoT, 2.37 on UNSW-NB15), with an exponential kernel on Euclidean distance in standardised space and `feature_selection='auto'` (all features here). None of this is reported. Continuous-mode LIME is known to be sensitive to kernel width, and the parallel referee report raises the same point. **Fix:** report the kernel width and kernel; ideally add one alternative width as a robustness check. | Medium |
| **M4** | §3.2, §3.4 ("After feature selection … UNSW-NB15 10") and §5.5 (TTL re-selection "selects 10 … features") | `preprocessing.py` L112–119: `RFECV(..., min_features_to_select=10)` with a 50-tree RF scored on f1_macro and 5-fold CV. Both UNSW selections end **exactly at the floor of 10**, so the floor, not RFECV's optimum, may set the feature count. This is not disclosed. Further settings missing from §3.2: MI threshold 0.05, RFECV estimator (RF, 50 trees), 50,000-row subsample. | Medium |
| **M5** | §3.4 ("the seeds vary model training only"); "range over seeds" in Tables 5–9 | For the explanation stages, `seed` also changes the SHAP background (`default_rng(seed)`), the deletion donor pool, LIME's `random_state`, the GradientExplainer `rseed`, and the counterfactual and hardening random streams. Only the test sample (`stratified_idx(seed=42)`) is fixed. The ranges for agreement, deletion, counterfactuals and hardening therefore mix retraining with explainer and search randomness. **Fix:** "the seeds vary model training and the random draws of the explanation and search procedures; data, split and feature selection are fixed". | Low–Medium |
| M6 | §5.2, "LIME ranked by its raw weights" (36, 30, 36 and 35 of 36) | `compute_values.py` L477–481 counts the raw-weight deletion cells over `('mean','record','gauss','cond')`. The `deletion.json` raw runs (24 Oct) predate `condx`, so the conditional strategy here *includes* exact copies of the record, while the primary analysis and §4.5 exclude them. The effect is small (differences ≤ 0.008 in the primary runs). Disclose it, or re-run `deletion` for raw weights. | Low |
| M7 | §3.3 ("Letting feature selection run again in each condition … gives similar macro-F1 effects (0.034 and 0.015)") | `dec_reselect_kept` = A_original − B. A is the **v1 model and v1 pipeline** (16 features, pre-fix loader, v1 scaler and selection), evaluated on the v1 split. B is retrained with the current code. The 0.034 therefore mixes the `id` effect with all v1→v2 pipeline differences. The 0.015 (C − D) uses the current code for both arms and is clean. **Fix:** say "the original model against a re-selected model without `id`", or re-run A with the current code and `drop_id=False, deduplicate=False`. | Low–Medium |
| M8 | §4.3 ("summing attributions within each one-hot group") | `field_attr` sums **absolute** attributions (`np.abs(att[idx]).sum()`). Fields are formed by `rsplit('_',1)` of binary 0/1 columns. Say "summing absolute attributions". | Low |
| M9 | §4.7 copy control ("the same groups, number of groups and support check") | The code draws the groups at random from the same group *set*, with the same k as the matched counterfactual (`crng.choice(len(groups), k)`). It does not reuse the counterfactual's own groups. Suggest "the same set of groups, the same number of groups and the same support check". | Low |
| M10 | §4.6 support check ("the 95th percentile of their nearest-neighbour distances") | τ is computed from 5,000 of the 20,000 benign records (`Support`, L567). τ ≈ 0.0025 on TON-IoT and 0.018 on UNSW-NB15 in scaled units, which is almost blind to heavy-tailed volume features (round-2 min-2; only partly disclosed as "this radius is small"). | Low |
| M11 | §4.6 and §5.3, queries to first evasion | `cf_search` counts every candidate up to the first hit, including candidates the support check rejects. An attacker with the support check would not need to query those. The text discloses that SHAP computation is not counted, but not this (round-2 MAJ-7, partly open). | Low |
| M12 | §5.2 ("excluding misclassified records … leaves it unchanged (UNSW-NB15 RF: 0.41 [0.39, 0.43])") | All records: 0.39 [0.37, 0.40]. The value rises by 0.02 and the intervals barely overlap. Say "changes it little". | Low |
| M13 | §5.5 (CNN macro-F1 0.5139 → 0.5106 without TTL) | `stage_ttl_reselect` is a single run with a fixed TF seed, while the seed-42 baseline CNN is unseeded. The drop of 0.003 is far below the CNN's seed range (0.5139–0.5305). The text's conclusion ("can largely replace it") is consistent with this, but the CNN comparison is within noise; say so. | Low |
| M14 | Table 8, §5.5 ("worms 0.53 → 0.30") | Still reported without n: worms n = 14 (RF) and 17 (CNN) counterfactual queries, 13 in the hardening query set, 19 test records (round-2 min-12, open). Add n or drop worms from the narrative. | Low |
| M15 | §5.4 hardening comparisons | Interval separation is per arm (`boot_ci` of each arm). The evaluation streams are not paired per record: `erng` is re-created per model but consumed differently (round-2 min-9, open). A paired test (McNemar on the common query set) would be the right test. The text's "interval lies below" wording is accurate but conservative. | Low |
| M16 | Tables 3–4 (CNN global SHAP) | `run_xai_v2.shap_cnn` is called without `rseed`, so the results cannot be reproduced exactly. The rank intervals reflect record resampling only, not explainer noise, which is large for the CNN (round-2 min-6, open). One clause in §6.3 would cover it. | Low |
| M17 | Provenance: TON-IoT counterfactuals (Table 7) | `results_v3/TON_IoT/seed*/counterfactual.json` were written by the **v3.0** code (24 Oct, 12:02–13:07, before commit 5c819da; `attacker_groups` is saved as a list). For TON-IoT the v3.0 and v3.1 group definitions are identical: there was no overlap bug, and the attacker groups were the same three dependency groups. The results are therefore valid, but they are not byte-reproducible with the current script. Note it in the repository README, or re-run. Likewise, the TON-IoT seed-42 models and `pipeline.pkl` are the July v1 files (the text says the data are identical, and the round-2 check confirmed that the metrics match). | Low |
| M18 | §3.3 duplicate statistics | The generic count of 40,000 → 4,181 and the 8,472 / 458 conflicts are computed on the raw CSV columns minus `id`. The 105,955 records and the 44.0% use the pipeline definition. The paragraph now flags the 105,955, but the generic and conflict counts are still on the raw definition (pipeline: generic 4,137). Say "on all raw columns" for those (round-2 min-1, partly open). | Low |

### Code bugs that could change results
- **None found.** The one earlier bug (the overlapping UNSW dependency groups) is fixed, and the affected UNSW counterfactual and hardening runs were redone.
- Minor code issues with no material effect:
  - `rng.choice(pool, 2)` in the interpolation control samples with replacement (round-2 min-16).
  - Per-record evaluation streams in hardening are not paired (M15).
  - The run order means `hardening` and `hardening_attacker` use different query sets.
- The only open issue that could move a reported comparison is **M2** (absolute versus signed top-k in deletion). It needs a short deletion-stage re-run to settle.
