# Referee report: Computer Networks

**Manuscript:** "How Light Is Lightweight? Detection Accuracy Against Model Footprint in IoT Intrusion Detection"
**Reviewed version:** Manuscript_filled.md (final numbers), Highlights from build.py/values.json
**Date:** 2026-11-14

**Score: 5.5 / 10**
**Recommendation: Major revision**

---

## Summary

The paper compares compact neural detectors (a 1D-CNN and an MLP at three widths, float32 and post-training int8) with tree models (LR, single trees, random forests, histogram GBDTs, including GBDTs with capped iterations) across a grid of model sizes on TON-IoT, UNSW-NB15 and N-BaIoT, using five training seeds on one data split. It reports macro-F1, a "compact" deployable size, and single-thread CPU latency, and adds a port-number ablation and three "pitfalls": scikit-learn's early-stopped HistGradientBoosting keeps the 20 non-improving iterations, in which boosting sometimes diverged; min-max scaling handicaps the CNN; and pickled forest size is about 10 times a compact encoding. The headline claims are that up to about 100 KB int8 CNNs and size-budgeted trees tie on TON-IoT and trees are better on UNSW-NB15; that from 201 KB (TON-IoT) and 97 KB (UNSW-NB15) a tree is "significantly more accurate than every network"; and that latency rankings depend on the runtime.

The work is careful and unusually candid. It reports seeds, validation-based selection, clean-record re-analysis, runtime controls, and a code release with an automated number check. The main message, that lightweight-network claims need size-matched tree baselines and deployable-unit accounting, is useful to practitioners. However, (a) the headline crossover claims rest on differences of 0.003 to 0.012 macro-F1. Their significance comes only from training-seed variance on one split. When I resample the test records the authors released, the two crossover claims no longer hold. (b) The size accounting leaves out the networks' input transform, which is about as large as the int8 networks themselves. (c) The tuning budget strongly favours trees. (d) The latency section mostly measures framework overhead. (e) The novelty over Grinsztajn et al. (2022) and especially De Bernardis et al. (2026) is incremental, and the networking side of the contribution is thin for *Computer Networks*. All of these can be fixed, mostly with the existing infrastructure.

---

## Spot checks against code and result files

- **GBDT rollback** (gbdt_rollback.py): this matches the text. `_predictors` is truncated to `argmax(validation_score_)`, and a unit test checks equivalence with a refit at `max_iter=k`. The docstring says the divergence comes from leaf values -G/(H+lambda) with `l2_regularization=0`. That cause is not mentioned in the manuscript, and it matters for Major 6.
- **Compact size** (run_p1_v2.compact_bytes, run_p1_final.compact_bytes): this matches Section 4.2. Networks are counted as `4*params` in float32 and as the TFLite blob in int8.
- **Quantile transform** (run_p6_v2._finish): `QuantileTransformer(n_quantiles=1000)` is fitted on all selected features. It is a required deployment component for the networks and LR, and it is not counted in their size (Major 2).
- **Thread control** (bench_p1_final.py): the environment variables are set before import, and every call runs under `threadpool_limits(1)`. latency.json confirms one thread for OpenBLAS and OpenMP. This is correct.
- **LightGBM proxy** (bench_p1_final.lgbm_proxy): a different model, trained without early stopping, with the iteration count copied from the seed-0 sklearn model. The manuscript describes it correctly.
- **Resplits:** `run_p6_v2.load(ds, split=k)` already supports new stratified splits (`RESPLITS = (1..5)`). The single-split limitation is therefore cheap to remove (Major 1).
- **int8 drops** (results_v2): the largest single-seed drop for the CNNs is 0.0205 (CNN w0.5, UNSW-NB15), which the text reports as "at most 0.020". Table 4 shows 0.529 to 0.508 (0.021). Minor rounding inconsistency.
- **Quantised GBDT encoding:** the size ratio compact/quantised ranges from 0.40 (N-BaIoT gbc50_l7: 38 KB becomes 96 KB because 19,050 float32 bin edges are stored) to 1.75. "Shrink by a factor of 0.4" therefore means the model grows 2.5-fold (Minor 6).

### Test-record bootstrap (my computation, from released `seed*_pred.npy` and `yte.npy`)

Method: paired bootstrap over test records, with one seed drawn at random per replicate (300 replicates). Difference = model A minus model B in macro-F1.

| Dataset | A vs B | Observed diff | 95% CI | P(diff <= 0) |
|---|---|---|---|---|
| TON-IoT | GBDT 31 leaves <=50 it (201 KB) vs CNN w2 f32 | +0.0030 | [-0.0005, +0.0064] | 0.05 |
| TON-IoT | GBDT 63 leaves lr 0.1 (best) vs CNN w2 | +0.0097 | [+0.0052, +0.0145] | 0.00 |
| TON-IoT | GBDT 63 lr 0.1 vs GBDT 31 <=100 it | +0.0044 | [-0.0014, +0.0092] | 0.10 |
| TON-IoT | GBDT 63 lr 0.1 vs earlier RF | +0.0049 | [+0.0012, +0.0085] | 0.00 |
| UNSW-NB15 | GBDT 15 leaves <=50 it (97 KB) vs CNN w2 f32 | +0.0136 | [-0.0033, +0.0287] | 0.08 |
| UNSW-NB15 | GBDT 31 lr 0.05 (best) vs CNN w2 | +0.0476 | [+0.031, +0.066] | 0.00 |

The best-tree-versus-best-network gaps (0.010 and 0.046) hold up. The "from 201 KB / from 97 KB, significantly more accurate than every network" thresholds do not hold up once test-sample variance is included. The UNSW-NB15 comparison was made against the float32 CNN w2 because int8 predictions were not stored. The int8 model is 0.002 better, so the int8 comparison would be weaker still.

---

## Major issues

### M1. The crossover claims ("significantly more accurate from 201 KB / 97 KB"; "trees won on UNSW-NB15 under 100 KB") are not supported once test-sample variance is considered
**Location:** Abstract (sentences 4–5), Highlights 1–2 ("trees won", "a tree significantly beats every network"), §5.2 paragraph 1, §6.1, §7.
**Why it matters:** The Welch tests use five seeds on one split, so they measure training randomness only, as §4.4 and §6.3 admit. Several configurations are deterministic or nearly so (SD 0.000), which makes Welch's t-test degenerate. The claims are also selected from 37 tree configurations against 6 networks with no multiplicity correction, and the comparison pairs are chosen on the test set (M3). My test-record bootstrap above puts the 95% CIs of the 201 KB (TON-IoT) and 97 KB (UNSW-NB15) comparisons across zero. "Significantly", "won" and "beats" in the abstract and highlights are the strongest claims in the paper, and the evidence for them is weakest. The editor asked specifically about claims of "winning" or "beating"; these qualify.
**Fix:** (i) Use the existing resplit loader (`run_p6_v2.load(split=k)`) to add at least 5 data splits × several seeds on TON-IoT and UNSW-NB15, and test across splits. Alternatively, report a paired bootstrap over test records combined with seeds (or a corrected resampled t-test) for every claimed difference. (ii) Correct for multiplicity, or pre-specify a small number of budget comparisons. (iii) Rewrite the abstract, highlights and conclusion so that only the robust result is stated as a win: the best tree beats the best network by 0.010 / 0.046. Report the per-budget crossovers as point estimates with CIs ("no detectable difference below about 200 KB on TON-IoT; trees ahead by 0.01 at about 100 KB on UNSW-NB15, CI including 0"). Change Highlight 1 "trees won" and Highlight 2 "significantly beats" accordingly.

### M2. The size accounting is asymmetric and omits the networks' preprocessing
**Location:** §4.1 (quantile transform), §4.2, §5.2, Table 2, Fig. 1, Highlights 1–2.
**Why it matters:** The networks (and LR) need the fitted QuantileTransformer at inference: 1,000 quantiles × d features. In float32 that is about 100 KB on TON-IoT (25 features), about 40 KB on UNSW-NB15 and about 460 KB on N-BaIoT. On TON-IoT this is roughly the size of the int8 CNN w2 (106 KB) that "ties" at the 100 KB budget. The tree models use min-max scaling, which can be folded into the thresholds at zero cost. Adding the transform moves every network to the right in Fig. 1 and changes the under-100 KB conclusion. Further asymmetries: the trees are counted in a hypothetical encoding that has not been implemented, while int8 networks are counted as real TFLite files with graph overhead. RF leaves are already float16, while float32 networks are counted at 4 B/param with no file overhead. The "deployable" tree encoding has never been executed, so its byte count is not verified.
**Fix:** Count the input transform in every model's footprint. Report it separately, and also with a reduced transform (e.g. 64–256 quantiles, or a piecewise-linear table) together with its accuracy effect. Use one consistent rule per family: either nominal bytes for all, or real artefacts for all. The best option is to export the trees with a real compiler (Treelite, m2cgen or emlearn, or ONNX-ML) and report the generated artefact's size. That also fixes M5. State that the tree encoding is nominal if it remains so.

### M3. Per-budget "best model" comparisons are selected on the test set
**Location:** §5.2 paragraph 1, §5.4 ("tied with the best", "fastest within 0.005 of the best"), Abstract, Highlights.
**Why it matters:** §4.4 defines validation-based selection only for "overall best" and "smallest within 0.01". All per-budget winners and the "smallest tree significantly above the best network" are chosen by test-set mean among many configurations. That is a winner's-curse selection, made worse by M1. The validation score also comes from seed 0 only ("validation scores of the other network seeds were not recorded"). The authors' own check shows the fragility: on TON-IoT the validation-selected small GBDT is within 0.01 of the best on only 1 of 5 test seeds.
**Fix:** For every budget, select each family's representative on validation macro-F1, averaged over all seeds, then report its test score. Recompute the network validation scores for seeds 1–4. The `nets_val` group in run_p1_final.py already does this. It is GPU time, but this is the core claim. Label every test-selected statement as descriptive, and keep it out of the abstract and highlights.

### M4. The tuning budget is strongly asymmetric, so the neural side is a weak baseline
**Location:** §4.1, §6.3 (last sentence).
**Why it matters:** The tree families get 37 configurations: depth, number of trees, leaves, learning rate, iteration caps and small trees. The networks get one architecture each at three widths, with fixed learning rate, dropout and batch size, and the CNN is inherited from the authors' earlier draft. A 1D convolution over an unordered vector of 10–25 heterogeneous tabular features has no meaningful locality prior. Feature selection uses a random-forest wrapper, which the paper admits may favour trees. Under these conditions "trees beat networks from X KB" partly reflects how much each family was tuned. For a paper whose stated purpose is a fair comparison, this is the main threat to validity.
**Fix:** Give the networks a comparable search: learning rate, depth and width, dropout and weight decay, and a feature-selection-independent input (e.g. all post-cleaning features). Matching the number of tried configurations or the compute would be the cleanest design. Add at least one modern tabular network, or quantisation-aware training, which is the standard route for int8 deployment. Alternatively, restrict the conclusions explicitly to "the compact CNN/MLP architectures typical of the IoT-IDS literature" and remove the general family-level statements in §6.1.

### M5. The latency section mostly measures framework overhead, and the int8 timings reflect an outdated runtime path
**Location:** §5.4, Tables 3–4, Abstract ("9.8–82.1 ms ... best network 0.42 ms"), Highlight 4, §6.1, §7.
**Why it matters:** scikit-learn HistGradientBoosting predicts one record through a per-iteration Python loop (about 7–10 µs per tree). TensorFlow-eager per-call overhead is about 0.35 ms. The authors themselves conclude that the ranking "is a property of scikit-learn and TensorFlow ... not of the model families". Yet the abstract and Highlight 4 headline exactly these confounded numbers: trees at 9.8–82.1 ms against the "best CNN 0.42 ms". The float32 TFLite number for that network is 0.026 ms, and LightGBM would take about 0.16 ms. The int8 results (up to 10 ms, slower than float32, no XNNPACK effect) indicate that TFLite 2.10 on x86 ran reference integer kernels. Newer LiteRT/XNNPACK versions have QS8 kernels, and ARM builds use optimised int8 paths. "Int8 saves memory, not time" is therefore a statement about one obsolete build. The compiled-tree evidence is a proxy LightGBM model, not the models evaluated. The 100 KB comparison ("best network 2.85 ms, best tree 5.13 ms") pits two unrepresentative runtimes against each other.
**Fix:** Time the actual trained trees through a compiled runtime (Treelite, m2cgen-generated C, or ONNX Runtime) and the networks through a current TFLite/LiteRT with int8 XNNPACK. Ideally also measure on one real ARM gateway class device (e.g. a Raspberry Pi 4 or 5), which is the hardware the paper's motivation names. Then drop the scikit-learn/TF-eager numbers from the abstract and highlights, or present them explicitly as the naive Python baseline. Rewrite Highlight 4. Report peak RAM as well as model bytes.

### M6. The GBDT "pitfall" needs a cause, and probably a better remedy than rollback
**Location:** §4.1 (GBDT bullet), §5.1 paragraph 2, Table A.2, Contribution 4, Recommendation 4, Highlight 5.
**Why it matters:** Keeping the n_iter_no_change iterations is documented scikit-learn behaviour (`n_iter_` includes them). The genuinely surprising finding is that multiclass boosting diverges: the internal validation log-loss goes from 0.042 to 361,323 within 20 iterations. The authors' code comment attributes this to leaf values -G/(H+lambda) with `l2_regularization=0` (near-zero Hessians on SMOTE-generated, nearly pure regions). If so, the proper fix is regularisation (l2_regularization > 0 or min_hessian), and rollback only hides the symptom. Rollback cannot help a user whose divergence happens before the best iteration. Without this analysis, the "pitfall" reads as a hyperparameter default interacting with SMOTE rather than a general evaluation lesson. It should not be presented as a library defect.
**Fix:** Add a small controlled experiment: the same configurations with l2_regularization ∈ {0, 0.1, 1}, with and without SMOTE, reporting divergence counts and accuracy. State the mechanism. Provide a minimal reproducible example (ideally reported upstream). Phrase Recommendation 4 as "regularise and evaluate at the best iteration".

### M7. Novelty and fit for *Computer Networks*
**Location:** §1 contributions, §2.1 "Two gaps remain".
**Why it matters:** "Trees match or beat networks on tabular data" is established (Grinsztajn et al., 2022; Shwartz-Ziv & Armon, 2022; McElfresh et al., 2023). De Bernardis et al. (2026) already compare LR/DT/RF/GBDT/MLP on TON-IoT with quantisation and real microcontroller deployment, which this paper lacks. The added value is the size-budget grid, three datasets, seeds, and the pitfalls. That is a solid benchmarking note, but its networking content is thin: no flow-level or online setting, feature extraction excluded (§6.3 says it "can dominate the cost"), no throughput in flows/s at line rate, no device measurements. The gap claim ("none of the neural studies above reports a comparison with a size-limited tree") rests on about 8 hand-picked papers.
**Fix:** Sharpen the contribution statement around what is new relative to De Bernardis et al. (2026). Support the gap claim with a systematic literature count (search string, number of papers, share reporting size-matched trees, units used). Add a networking-relevant cost dimension: feature-extraction cost per flow, or sustained flows/s per core against realistic gateway traffic rates, or at least one embedded-device measurement. Otherwise a security/ML venue may be a better fit than *Computer Networks*.

### M8. UNSW-NB15 results are operationally meaningless as configured, and SMOTE to full balance distorts the operating point
**Location:** §3.1 (SMOTE "until every class has as many records as the largest"), §5.1 (benign FPR 11–23%), Appendix A.
**Why it matters:** Oversampling every attack class to the majority size moves the classifier's priors toward attacks. That is the likely reason that nearly every model flags 11–23% of benign UNSW-NB15 traffic. On UNSW-NB15, half of the paper's evidence comes from models the authors themselves call "far too many for operational use". Model comparisons made at a distorted operating point may not carry over to a calibrated one, and SMOTE on one-hot columns creates fractional categorical values. Using only the official training partition with 10 RFECV-floor features is a further non-standard choice that makes comparison with the literature hard.
**Fix:** Add a no-oversampling (or class-weighted / prior-corrected) variant for at least the representative models. Report macro-F1 together with a benign-FPR-constrained operating point (e.g. detection rate at 1% FPR). Discuss whether the tree-network ranking survives. Consider also using the official UNSW-NB15 train/test split.

### M9. Clarity, density and the self-referential framing
**Location:** throughout; in particular §5.1–5.4 and §5.5.
**Why it matters:** The body is about 8.6k words, plus a very long appendix, and the results read as a stream of numbers. §5.4 is a single paragraph containing more than 40 numeric values. Sentences generated from templates read awkwardly, e.g. "Ransomware loses most for every forest and network ...; for the GBDT at learning rate 0.05 the class that loses most is ransomware", "-2.9–7.6 KB larger", and "shrink by a factor of 0.4–1.7". Repeatedly anchoring the paper on "our own earlier, unpublished draft" (§1, §5.1, §5.5, §6.1, Fig. 1 ringed point) is unusual. Readers cannot inspect that draft, and the 0.880 figure cannot be verified. The LSTM paragraph in §5.5 has no experiment, while Contribution 4 promises that each pitfall is "shown with a controlled experiment".
**Fix:** Restructure the results around the three research questions, each with one figure or table and a few sentences of interpretation. Move the spreads and secondary numbers to tables. Present §5.5 as controlled ablations (min-max vs quantile; pickled vs compact) without relying on an unpublished draft, or cite it as a preprint. Drop or shorten the LSTM paragraph. Aim for at most about 7k words in the body.

---

## Minor issues

1. **Abstract, "an unbounded forest was never more accurate than smaller trees":** this is false as written. The unbounded RF (0.606 on UNSW-NB15) beats most smaller trees. Write "than the best smaller tree model". The claim also relies on non-significant differences (0.960 vs 0.962; 0.606 vs 0.608). State it as "no more accurate than". Note too that on clean UNSW-NB15 records the unbounded RF is the best tree (0.667, §5.1).
2. **§5.2, "up to about 100 KB (at most 110 KB, so that the largest int8 network ... is included)":** the budget edge is chosen after looking at the results, and it also happens to include the 110 KB GBDT. Fix the budgets beforehand (e.g. 16/32/64/128/256 KB) or show the full Pareto front with uncertainty bands.
3. **Highlight 1, "int8 CNNs ... tied":** only one int8 CNN (width 2) ties. Highlight 1 also calls GBDT 15 leaves lr 0.1 a "size-budgeted tree", but it comes from the uncapped main grid (it equals the ≤200-iteration model).
4. **§5.3 vs Table 4:** "at most 0.020 for a single seed" is 0.0205 in the results, and Table 4 shows 0.021. Round consistently.
5. **§5.3:** int8 calibration uses 500 SMOTE-oversampled records. Synthetic records are a questionable calibration set. Calibrate on real training records, or test the sensitivity.
6. **§4.2/§5.2, quantised GBDT encoding:** a "factor of 0.4" means a 2.5-fold increase (stored bin edges dominate on N-BaIoT). Rephrase. Also, the 8-bit bin index requires binning each input at inference, which adds cost and is not timed.
7. **§4.2:** the compact encoding ignores GBDT missing-value routing bits and the per-class tree layout. State that the encoding is nominal and has not been implemented (see M2).
8. **Table 3:** it omits the single decision trees and capped GBDTs that were benchmarked (bench_p1_final has them). A depth-12 tree (12 KB, 0.942) is probably the fastest model of all and belongs in the latency discussion.
9. **§5.4:** "the best network" switches between the TF float32 CNN w2 (TON-IoT, 0.42 ms) and the TFLite int8 CNN w2 (UNSW-NB15, 1.59 ms). Name the runtime every time a latency is compared.
10. **§7, "the fastest tree model within 0.005 ... and a compiled tree library were much faster":** 4.02 ms is still about 10 times the TF network and about 150 times the TFLite network. Quantify "much faster" and say what it is faster than.
11. **§6.1:** "a compiled GBDT" is recommended, but no compiled version of the evaluated GBDTs was run (proxy only).
12. **§4.4:** Welch's t-test with SD = 0 on one side (LR, DT, near-deterministic RFs) is degenerate. Say how such cases were handled.
13. **§4.1:** say whether seeds affect SMOTE or the quantile subsample. The text says not; confirm this explicitly in §4.4.
14. **§3.2/§5.1:** the 16.3% train–test overlap on UNSW-NB15 after feature selection is large. Consider de-duplicating after feature selection as the primary protocol rather than as a secondary analysis.
15. **§5.6:** ports are removed, but service_dns/service_http and DNS fields remain, and these encode the same testbed services. Discuss this, or ablate them too. The GBDT lr 0.1 per-class loss is not reported.
16. **Appendix A:** explain the asterisk (seed-0 validation, networks only) in the table itself, and add the clean-record macro-F1 column.
17. **§4.1:** the GPU non-determinism under XLA is acknowledged, but report the observed spread in rerun-with-same-seed results, or disable XLA for one check.
18. **Data availability:** give a tagged release or commit hash and archive it (e.g. Zenodo DOI). A bare GitHub URL can change.
19. **Keywords/title:** "How Light Is Lightweight?" is fine. Consider adding "Pareto front" or "size budget" to the keywords.
20. **Typography:** use "−2.9 to 7.6 KB" rather than "-2.9–7.6 KB", and use consistent minus signs (Table 5 uses "−", §4.2 uses "-").

---

## Strengths

- The research question is relevant to practice: size-matched baselines and deployable-unit accounting for lightweight-IDS claims.
- The protocol is transparent: five seeds, held-out validation, released code, result files and predictions, and an automated number-to-text check. My spot checks of the size computation, the rollback and the thread control matched the text.
- Size-budgeted GBDTs, single trees and shallow forests fill the under-100 KB range that is usually missing.
- The runtime controls (TFLite float32, XNNPACK on/off, full-integer int8, compiled-tree proxy) correctly warn readers against reading framework latency as model cost.
- The clean-record re-analysis, port ablation and input-scaling ablation are useful robustness checks. The pickled-versus-compact size finding (about 10×) is a concrete, reusable warning.
- Limitations are stated honestly (single split, laptop CPU, N-BaIoT leakage, RF-wrapper feature selection).
