# Referee report: "How Light Is Lightweight? Detection Accuracy Against Model Footprint in IoT Intrusion Detection"

Journal: Elsevier *Computer Networks*
Material reviewed: Manuscript_filled.md (full text, Tables 1–4, Appendix A incl. the learning-rate table), figures/fig1_accuracy_size.png, and the result summary points.json / values.json (read only, to check the numbers the text quotes).
Date: 2026-11-07

## 1. Summary

The paper compares logistic regression, random forests (RF) at six size settings, histogram GBDTs (4 leaf counts x 2 learning rates), and a 1D-CNN and an MLP at three widths in float32 and int8 TFLite. It uses TON-IoT, UNSW-NB15 and N-BaIoT with five seeds on one fixed split. It reports macro-F1, a "compact" model size computed analytically, the pickled size, and single-thread CPU latency for the seed-0 model. The headline claims are:
(a) up to about 100 KB, int8 CNNs are the most accurate (+0.028 and +0.046 macro-F1 over the best tree of similar size);
(b) at 1–2 MB, GBDT beats every network (+0.009, +0.043), and unbounded forests are never needed;
(c) int8 costs CNNs at most 0.010 but MLPs up to 0.073 on single seeds;
(d) the most accurate GBDTs are 1–2 orders of magnitude slower per record (97 vs 0.41 ms);
(e) GBDT at the default learning rate collapses on some seeds;
(f) removing ports costs about 0.015 macro-F1 on TON-IoT.
The paper ends with six recommendations for lightweight-IDS evaluation.

The question is worth asking, and the paper is unusually honest about its own limitations. But the two headline comparisons, the size crossover and the latency ranking, depend on design choices that favour one family. The tree size grid has no GBDT below about 120 KB and no tree quantisation, while the networks get int8. The latency compares an interpreted per-tree Python loop with compiled framework kernels. Model selection and the "smallest within 0.01" rule are applied on the test set. One headline result is within 3x10^-5 of its threshold. The GBDT collapse at the default settings is left undiagnosed, although GBDT is the reference model the whole argument rests on. Novelty over De Bernardis et al. (2026), Tekin et al. (2023) and the tabular-ML literature is modest. The networking content (flow rates, feature-extraction cost, gateway hardware) is thin for *Computer Networks*.

**Score: 5/10**
**Recommendation: Major revision**

## 2. Major issues

### M1. The "networks win below about 100 KB" result comes from the tree grid, not from the families
*Location:* Abstract ("Up to about 100 KB, int8 CNNs were the most accurate models"); Section 4.1 (GBDT grid); Section 5.2; Section 6.1; Appendix A.

*Why it matters:*
- The GBDT grid bounds only the leaves per tree. The number of iterations is set by early stopping up to 1,000. As a result, the smallest GBDT is 121 KB on TON-IoT and 216 KB on UNSW-NB15 (Appendix A), and no GBDT exists in the at-most-100 KB region the claim is about.
- The only trees at about 100 KB are 25-tree RFs of depth 8. These are the weakest trees in the grid (0.923 TON-IoT, 0.517 UNSW-NB15).
- GBDT "size budgets" are not budgets at all. Section 5.4 says the best TON-IoT GBDT used the full 1,000 iterations (10,000 trees). That means early stopping did not stop, and the size was set by the cap.
- The networks are given int8 quantisation, while trees are encoded with 32-bit thresholds and float16/float32 leaves (Section 4.2). Thresholds and leaves of a tree can be quantised just as easily: 8-bit feature indices are enough for at most 115 features, and 16-bit or 8-bit thresholds/leaves are routine. De Bernardis et al. (2026), cited in Section 2.1, report a quantised LightGBM of 74 KB as the best trade-off on TON-IoT, which directly contradicts the framing.
- On UNSW-NB15 the crossover is already at 216 KB (GBDT 7 leaves, lr 0.1, 0.589 vs the best network's 0.563). That is far below the "one to two megabytes" in the Abstract.

*What would fix it:*
- Add a real size-budgeted GBDT: cap the iterations (e.g. 25/50/100/200) and depth/leaves so that several GBDTs fall at 10–100 KB.
- Add single decision trees and shallow RFs (depth 4–6).
- Report trees with a quantised encoding (e.g. 8-bit feature index, 16-bit threshold, 8/16-bit leaf) beside int8 networks.
- Then restate the crossover per dataset. The Abstract's "1–2 MB" should describe where trees overtake, not where the best GBDT sits.

### M2. The latency comparison measures runtimes, not models, and the conclusions overreach
*Location:* Section 4.2 ("the ranking of models by cost per decision is what we compare"); Section 5.4; Section 6.1 ("better served by a compact network or a small forest, unless the trees are compiled"); Abstract; Conclusion.

*Why it matters:*
- About 9–11 µs per tree (Section 5.4) is the overhead of scikit-learn's per-predictor Python dispatch, not the cost of evaluating a tree. A compiled 15-leaf tree is on the order of tens of nanoseconds. The 236x gap is therefore mostly an implementation artefact. The authors acknowledge this ("a compiled tree implementation ... would be faster per tree, which we did not measure") but still put the ranking in the Abstract and Conclusion.
- The network side has artefacts too. The int8 CNNs are slower than float32 (CNN width 2 int8: 10.955 ms on N-BaIoT vs 0.461 ms float32; Table 3). On a modern x86 CPU this usually means TFLite fell back to reference kernels, XNNPACK was not used, or interpreter set-up was inside the timed loop. Reporting such numbers without diagnosis weakens the claim that "the ranking ... is what we compare". The ranking is exactly what changes when the runtime changes.
- Latency is measured for one seed-0 model per configuration, with no dispersion. GBDT latency depends on the number of trees, which varies with early stopping across seeds.

*What would fix it:*
- Time trees through at least one compiled path (Treelite, lleaves, m2cgen/emlearn C code, or ONNX Runtime with TreeEnsemble ops).
- Check the TFLite int8 configuration (XNNPACK on, `allocate_tensors` outside the loop, number of threads fixed).
- Report the time per record across the five seeds.
- Ideally, add one ARM measurement (Raspberry Pi class), since the paper's audience is gateway deployment.
- Until then, remove the cross-family latency ranking from the Abstract and Conclusion, or state it as a property of the scikit-learn vs TensorFlow runtimes.

### M3. Model selection, the "smallest within 0.01" rule, and an adaptive grid all use the test set
*Location:* Section 4.4 ("For each dataset we identify the configuration with the highest mean macro-F1 ..."); Section 4.1 ("We first ran the library's default learning rate ... so we added a second grid at learning rate 0.05"); Section 5.2.

*Why it matters:*
- The best configuration and the smallest one within 0.01 of it are both chosen on test macro-F1. A validation split exists (Table 1) but is not used for these choices.
- The lr 0.05 grid was added after seeing results. That result is reported as the overall winner on both hard datasets.
- The headline "smallest configuration within 0.01 of the best is ... a network on TON-IoT, CNN width 2 (int8) (106 KB, macro-F1 0.951)" is decided by a margin far below the noise. From points.json, the best is 0.96145, so the threshold is 0.95145. CNN width 2 int8 scores 0.95148, a margin of 0.00003 against a seed SD of 0.0009. The next candidate, MLP width 2 int8 (0.9498), misses by 0.0017. A different seed or a different test sample could flip the conclusion to a tree at about 1 MB.

*What would fix it:*
- Select on validation macro-F1 and report test numbers for the selected configurations.
- State plainly that the lr 0.05 grid was added post hoc.
- Give the "within 0.01" statement with uncertainty, e.g. the proportion of bootstrap test resamples or seeds in which each candidate qualifies, or drop the knife-edge framing.

### M4. The statistical inference ignores test-set uncertainty and is unsuitable for some comparisons
*Location:* Section 4.4 (Welch's t-test over seeds); Section 5.1 ("Welch p < 0.001"); Table 4; Section 3.2 (MITM 156 test records, worms 19).

*Why it matters:*
- All five seeds share one split and one SMOTE-oversampled training set (acknowledged in Section 6.3). The t-tests therefore measure only optimisation randomness, not test-sample variance or split variance.
- Macro-F1 averages over ten classes, two of them tiny. On UNSW-NB15 one class has 19 test records, so a few records can move macro-F1 by about 0.01. That is the size of several claimed effects (+0.009 on TON-IoT, the port effect of −0.013 to −0.015).
- Welch's t on n=5 per arm is fragile, and with SD 0.000 (LR) or bimodal collapses (GBDT lr 0.1, SD 0.275) its assumptions fail.
- There is no multiplicity control, although the paper makes dozens of pairwise statements.

*What would fix it:*
- Add a paired bootstrap over test records (or McNemar-style tests for paired predictions) for the key pairs.
- Better, use repeated splits (e.g. 5 splits x 3 seeds) for at least the headline comparisons: best GBDT vs best network, the 100 KB comparison, and the port ablation.
- Report per-class F1 for the small classes, or a weighted-F1 or balanced-accuracy check showing that the conclusions do not rest on 19 worms records.

### M5. The GBDT collapse is left undiagnosed, which leaves the reference model untrustworthy
*Location:* Section 4.1; Section 5.1 ("We did not diagnose the cause"); Appendix A learning-rate table (lowest seed 0.334 TON-IoT, 0.365 UNSW-NB15; and lr 0.05, 7 leaves, TON-IoT: 0.927 ± 0.072, lowest seed 0.798); Section 6.2 item 4.

*Why it matters:*
- A histogram GBDT at lr 0.1 falling to 0.334 macro-F1 on 25 well-behaved features is not ordinary seed variance. It suggests a pipeline problem.
- One plausible cause comes from the design itself. Early stopping uses an internal random 10% of the SMOTE-oversampled training set. SMOTE puts synthetic neighbours of the same minority records in both parts, so the stopping signal is unreliable. The networks, by contrast, early-stop on the real validation split (Section 4.1), so the two families are not tuned on equal terms.
- The lr 0.05 runs are still unstable at 7 leaves on TON-IoT.
- The paper's main message is "compare against a strong tree reference". A reference that silently fails in 6 of 20 default runs needs an explanation, not just a second grid.

*What would fix it:*
- Report `n_iter_` and the training/validation loss for the collapsed seeds.
- Rerun with early stopping on the provided validation split (`validation_fraction=None` with an explicit eval set, or LightGBM/XGBoost with an eval set), and/or without SMOTE for the trees, which handle imbalance via class weights.
- Show whether the collapse persists.
- If it disappears, the recommendation in Section 6.2 item 4 and the Abstract sentence about collapse must change.

### M6. Size accounting differs between families and between float32 and int8
*Location:* Section 4.2; Section 5.3 ("shrinks the networks by a factor of 1.4–3.8").

*Why it matters:*
- Float32 network size is 4 bytes x parameters with no graph overhead, while int8 size is the real TFLite file with graph and quantisation metadata. Tree size is an analytical encoding that was never serialised or checked.
- So three different kinds of number are compared on one axis, and the int8/float32 ratio in Section 5.3 mixes an analytical size with a file size.
- This matters exactly at the 100 KB crossover (M1), where 106 KB vs 101 KB decides the comparison.

*What would fix it:*
- Either export all models to one deployable format and measure files (e.g. float32 and int8 TFLite for networks, and generated C arrays or Treelite/ONNX files for trees), or compute every family analytically and report file overhead separately.
- At minimum, report float32 TFLite file sizes.

### M7. Novelty and framing: a comparison of known families, organised around correcting an unpublished study
*Location:* Section 1, paragraph 3 and contribution 4; Section 5.5; Section 6.1; Conclusion ("The earlier version of this study recommended a CNN over a 95 MB forest ...").

*Why it matters:*
- Trees vs networks on tabular data (Grinsztajn et al., 2022; McElfresh et al., 2023) and trees vs MLP with quantisation on TON-IoT with MCU deployment (De Bernardis et al., 2026) are already published.
- The added value here is mainly the multi-dataset, multi-seed, unit-consistent framing and the int8 CNN. That is useful but incremental.
- A substantial part of the narrative (Section 5.5, the Conclusion, contribution 4) is about errors in an *unpublished* earlier version. Readers cannot check or cite that version, and correcting it is not a scientific contribution of a journal article.
- The general lessons are worth keeping: min–max scaling of heavy-tailed counts hurts batch-normalised networks; pickled scikit-learn size is about 10x a deployable encoding; shuffled-record "sequences" are meaningless. They should be framed as pitfalls found in the published literature, ideally with examples, not as corrections to the authors' own draft.

*What would fix it:*
- Remove "earlier version" framing from the Abstract, Introduction, Conclusion and contributions.
- Recast Section 5.5 as "Input scaling and size-accounting pitfalls", supported by the min–max control experiment and by citing published lightweight-IDS papers that compare against unbounded or pickled forests, or use min–max-scaled CNN inputs. Without such citations the claim that the pitfall is common in the literature is unsupported (Section 2.1 says only "as far as their texts (or ... their abstracts) show").

### M8. Weak fit with *Computer Networks*: the networking side of "lightweight" is missing
*Location:* Section 4.2; Section 6.3 ("Feature extraction, which can dominate the cost of a flow-based detector, is not included").

*Why it matters:*
- For a networks venue, cost per decision means something only next to flow arrival rates and the cost of computing the features. Building 115 Kitsune statistics or TON-IoT flow features per flow may cost more than any classifier here.
- No throughput (flows/s per core), memory at run time (RSS/peak RAM, as distinct from file size), or energy is reported, and nothing runs on gateway-class hardware.

*What would fix it:*
- Add per-flow feature-extraction cost for at least one dataset (e.g. CICFlowMeter/Zeek or Kitsune feature computation time per packet or flow).
- Express classifier latency as a share of a realistic per-flow budget at gateway traffic rates, and report peak inference RAM.
- One ARM board measurement would greatly strengthen relevance.

### M9. Feature selection uses a random forest, which may favour trees
*Location:* Section 3.1 (MI filter, then RFECV with a 50-tree random forest scored by macro-F1; floor of ten features; 10 features kept on UNSW-NB15).

*Why it matters:*
- Selecting features with an RF wrapper optimises the feature set for tree-style splits, and then trees and networks are compared on that set.
- On UNSW-NB15 the procedure stops at the floor of 10 features. That suggests the floor, not the data, set the dimension, and networks may lose more from the aggressive reduction.
- The comparison between families is therefore conditioned on a tree-chosen feature set.

*What would fix it:*
- Repeat the headline comparison on all post-filter features (or on a model-agnostic selection such as the MI filter only) for TON-IoT and UNSW-NB15.
- Report how many features RFECV would keep without the floor.

## 3. Minor issues

1. **Dataset versions are not specified (Section 3.1).** State which TON-IoT file (e.g. Train_Test_Network) and which UNSW-NB15 release (the 2.5M-record CSVs or the 257k training/testing partition) were used. Table 1's sizes cannot be reconciled with either without this.
2. **The quantisation claims compare unlike statistics (Abstract, Conclusion).** "Int8 quantisation cost the CNNs at most 0.010 macro-F1 but the MLPs up to 0.073 on single seeds" sets a seed mean against a single-seed maximum. Give the single-seed maximum for CNNs too, or means for both. Also, on UNSW-NB15 the best network is the int8 CNN (0.563), which beats its own float32 version (0.561; values.json drop −0.002). Say so explicitly, because it shows that the differences at this level are noise.
3. **Agreement vs accuracy (Section 5.3).** MLP width 1 int8 agrees with float32 on only 89.7% of UNSW-NB15 test records but loses only 0.021 macro-F1. Discuss what the 10% disagreement means for operational alerting (it may be FPR churn).
4. **Figure 1 omits the lr 0.1 GBDTs.** Section 5.2 cites "GBDT, 7 leaves, lr 0.1 (216 KB, 0.589)" as the crossover on UNSW-NB15, but the legend shows only "Gradient-boosted trees (lr 0.05)". Plot both, or adjust the text. The very long error bar of TON-IoT GBDT 7 leaves (lr 0.05) needs a caption note. The figure caption itself is missing from the text.
5. **Table and figure captions, and table numbering.** In the source, Tables 2–4 and the learning-rate table ("Table A.2") have no captions, and Table A.1 is not labelled. Check the Word build.
6. **Sign conventions.** "The best network is +0.009 below" (Section 5.1) and "falls by −0.015" (Section 5.6) are double-signed. Write "0.009 below" and "falls by 0.015".
7. **Templating artefacts.** "the largest int8 network, CNN width 2 (int8) (106 KB, 0.951)" and "(RF, 25 trees, depth 8 (101 KB, 0.923))" have nested parentheses (Section 5.2). "Ransomware loses most in every model (for the earlier forest ransomware (−0.06))" (Section 5.6) is ungrammatical. Give per-model ransomware F1 changes in Table 4.
8. **"Up to about 100 KB" vs 106 KB (Section 5.2, Abstract).** The winning network is 106 KB (108,592 bytes) and its tree comparator 101 KB. Define the budget explicitly (e.g. at most 110 KB) or phrase it as "at about 100 KB".
9. **The test overlap check works against the trees (Section 3.2, Section 5.1).** 16.3% of UNSW-NB15 test records match a training record on the selected features. Removing them *increases* the tree lead (+0.071). That is consistent with trees not relying on memorisation, but it also shows that the 10-feature space collapses many records. Report how many classes are affected and the macro-F1 of each family on that subset.
10. **N-BaIoT adds little.** It is saturated (21/29 configurations at 0.998 or above) and acknowledged to leak through overlapping windows (Section 3.2). Consider moving it to a supplement or replacing it with a harder IoT set (e.g. CICIoT2023, used by Mazinani et al., 2026, or the newer Edge-IIoTset), which would also strengthen the IoT claim, given that UNSW-NB15 is not IoT traffic.
11. **SMOTE before quantile fitting (Section 4.1).** The quantile transform is fitted on a subsample of the *oversampled* set, so synthetic points shape the network input scaling. This is minor, but state it, or fit on the original training records.
12. **"Unbounded forest was never needed" (Abstract, Section 6.1).** On UNSW-NB15 the unbounded RF ties the best (0.606 vs 0.606), and it has the lowest benign FPR (0.110 vs 0.148 for the best GBDT; Appendix A). Because the paper reports FPR, mention that the operational ranking by FPR differs from the macro-F1 ranking.
13. **Benign FPR on UNSW-NB15 is about 0.11–0.23 for every model (Appendix A).** That is operationally unusable and should be discussed. The low macro-F1 values (about 0.6) also need a sentence on which classes drive them (likely DoS/exploits/analysis/backdoor confusion).
14. **Port ablation (Section 4.3, Section 5.6).** Only ports are removed. Other testbed-specific fields that are often flagged in TON-IoT (e.g. DNS/SSL/HTTP string-derived fields, `service`, connection state) are not discussed. State which of the 25 selected features remain and whether any of them is an identifier proxy. List the 25 and 10 selected features in the appendix.
15. **Batch latency for int8 is missing (Table 3, "–").** Explain why: the TFLite interpreter was not resized for batches. Fill it in or justify the omission.
16. **The quantisation calibration set is fixed at 500 records (Section 4.1)** and was not varied. A quick sensitivity check (e.g. 100/500/2,000) would support the conclusion that "MLP int8 is unpredictable".
17. **The Section 2.1 sentence on hedged evidence ("or, where these were not available to us, their abstracts")** is unusually weak for a related-work claim on which the novelty rests. Obtain the full texts or soften the claim.
18. **Reproducibility (Section 4.1).** GPU non-determinism with XLA is acknowledged. Please also state the library versions (scikit-learn, TensorFlow/TFLite) in the paper, since HistGradientBoosting defaults and TFLite kernels change between releases. Both affect M2 and M5.

## 4. Strengths

- It asks a clear and useful question (accuracy at a given *size*, not accuracy against an unbounded baseline) and answers it with an explicit accuracy–footprint plane (Figure 1, Appendix A).
- It is transparent: full 29-configuration grids per dataset, seed SDs, the default-learning-rate collapse reported rather than hidden (Appendix A table), compact vs pickled size reported side by side, and code and result files released.
- The min–max vs quantile input-scaling control (Section 5.5) is a clean, single-factor experiment with a practically relevant result (+0.047 macro-F1 for the CNN on TON-IoT).
- It handles data hygiene carefully: duplicates removed after column drops, a cross-split identity guard, a re-analysis on test records without feature-space matches, and an honest statement of N-BaIoT window leakage and UNSW-NB15's non-IoT nature.
- The limitations section (Section 6.3) is candid and specific, including the fact that the seeds do not vary the split.
- The recommendations in Section 6.2 are concrete and, once M1–M5 are addressed, would be a worthwhile methodological checklist for the lightweight-IDS community.
