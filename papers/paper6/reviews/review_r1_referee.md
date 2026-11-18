# Referee report, Round 1

**Manuscript:** "Feature Groups Are Not Modalities: A Controlled Evaluation of Multi-Stream Fusion for IoT Intrusion Detection"
**Journal:** Elsevier *Internet of Things*
**Date:** 2026-11-01

**Score:** 5.5 / 10
**Recommendation:** Major revision

As instructed, I have not counted the unfinished parts against the paper. These are the capacity-matched control (early_wide, 120-epoch runs) and the latency section §5.6, which appear as ⟦…⟧ markers. I do comment on whether the planned controls are adequate (Major 2).

---

## Summary

The authors ask whether splitting flow features into semantic "streams", encoding each stream with its own 1D-CNN and fusing them (early, late or attention fusion) improves IoT intrusion detection once the obvious confounds are controlled. The study uses TON-IoT, UNSW-NB15 and N-BaIoT, five seeds, paired differences with a Holm-corrected family, ten random feature splits of matched sizes, a random forest (RF) on the same features and a planned capacity-matched single network. The main findings are:

- The gain of fused models over single branches comes from the extra features: early fusion (one CNN on all features) and the RF capture it.
- Late fusion adds a small gain (+0.005 on TON-IoT, Holm p = 0.022). Attention fusion adds nothing.
- The semantic grouping matters little once the streams are fused.
- Attention weights track per-class stream usefulness only weakly and are unstable across seeds.
- Min–max inputs silently broke the volume-stream CNN on TON-IoT, which reversed the conclusion of the authors' earlier version.

The paper is honest, unusually self-critical and clearly written. The question is worth asking. The fusion-IDS literature does compare fused models almost exclusively against their own branches, and a paper that states the right controls is useful.

My concerns are about how far the evidence goes:

1. Every variance estimate comes from one data split per dataset.
2. The 1D-CNN encoder introduces a feature-order confound, and the paper explicitly denies it in one place.
3. The random-split statistics are under-specified and partly confounded with seed.
4. The headline "not modalities" claim is wider than two small CNN encoders on two non-saturated datasets can support, only one of which is IoT.
5. Novelty is methodological, and its main support (the claim that "none" of the prior papers ran these controls) rests on a sample that was largely read from abstracts.

---

## MAJOR issues

### M1. All uncertainty is conditional on one train/validation/test split, so the CIs and "5/5 seeds" overstate reliability.

§4.3 says: "Data, split and features are identical across seeds, so the seeds vary initialisation and batch order only." §6.3 repeats: "on one data split per dataset."

Every conclusion in Table 3, and the Holm-adjusted p = 0.022 for late against early fusion, is therefore a statement about this particular test set. With RF standard deviations of 0.000–0.002 (Table 2), the "+0.004 (5/5)" RF advantage over late fusion on TON-IoT is effectively a single-split point estimate. Differences of a few thousandths of macro-F1 are well within typical split-to-split variation on TON-IoT, where the MITM class has 156 test records and moves macro-F1 by itself (§5.2).

The issue is sharpest for the one positive result in the paper (late > early, +0.005). That result is driven largely by MITM (0.63 → 0.66, Table 4), and a 0.03 F1 change on 156 records is roughly 2–3 records per class-side.

**Fix:** For TON-IoT and UNSW-NB15, repeat the core comparison over at least 5 independent stratified splits. The full pipeline (feature selection, SMOTE, scaling) must be refitted inside each split. Repeated 5×2 CV with the corrected resampled t-test (Nadeau & Bengio) or a hierarchical/mixed model with split and seed as factors would also work. Report split-level CIs for early, late, attention, early_wide and RF. If compute is limited, do this at least for early, late, early_wide and RF on TON-IoT and UNSW. Also report the late-minus-early difference with MITM excluded.

### M2. The capacity control as planned is necessary but not sufficient. The 120-epoch check is too narrow.

§4.2 plans one variant: "the single network on all features with every layer widened by a factor of 1.6". Matching the parameter count separates parameters from grouping. But late fusion differs from early fusion in three other ways:

- it has an extra 128-unit dense layer after the embeddings (a deeper head);
- it keeps two independent sets of BatchNorm statistics;
- it behaves like a two-member ensemble whose members are combined before the softmax.

The random-split result in §5.3 already suggests that the gain is architectural rather than semantic: "Late fusion on random splits also stays above early fusion on TON-IoT (by +0.003 on average, all ten splits)". So the control has to identify *which* architectural factor is responsible.

**Fix:** Add the following.

- (a) A **duplicated-input late fusion**: two encoders, each on *all* features, with the same head. This is the cleanest test of whether splitting the features adds anything beyond having two encoders.
- (b) **Early fusion with the same 128-unit dense head** as late fusion, to match depth as well as width.
- (c) Optionally, an **ensemble of two independently trained early-fusion CNNs** (averaged logits).

Run early_wide and (a) on TON-IoT *and* UNSW-NB15. §4.2 gives the parameter count only "on TON-IoT", which leaves the scope unclear. Use 5 seeds and include them in the Holm family.

For the epoch budget, §4.2 says 13/75 runs hit the 60-epoch cap, 8 of them on TON-IoT, yet only early and late fusion on TON-IoT are rerun. Do the following:

- rerun every configuration that hit the cap, including the 5 non-TON runs and attention fusion;
- give early_wide the same 120-epoch budget;
- report the best-epoch distribution per architecture in a supplementary table.

Without this, a capacity or budget artefact cannot be ruled out as the source of the only significant fusion effect.

### M3. Feature-order confound in the 1D-CNN, and an incorrect statement that depends on it.

§4.1: the encoder "treats the feature vector as a one-channel sequence", with kernel-3 convolutions. Tabular features have no natural order, so the encoder's inductive bias depends on how the columns happen to be ordered.

§4.2 then states: "Early fusion is not repeated, because a random split only reorders its inputs." For a convolutional encoder this is false. Reordering changes which features share a receptive field, so early fusion on a random split is a *different model*. The same confound affects every random-split CNN in Table 6: a random split changes both the grouping and the within-stream order, and the two effects cannot be separated.

The CNN choice also weakens generality. A reader can reasonably object that the conclusion holds only for an encoder whose convolutions mix arbitrary neighbouring features.

**Fix:**

- (i) Correct the sentence in §4.2.
- (ii) Run early fusion under the ten random column orders (or at least five) and report its spread. This is the order-only baseline that the random-split fusion results should be compared against.
- (iii) Repeat the main comparison (single streams, early, late, attention, RF) with an **MLP encoder**, which is order-invariant, on TON-IoT and UNSW-NB15.

If the conclusions survive an MLP encoder, the paper becomes much stronger. A gradient-boosted tree baseline (LightGBM or XGBoost) would also make the "strong tabular baseline" claim more convincing than RF alone.

### M4. The random-split analysis is statistically under-specified and confounded with seed. The text also partly contradicts its own test.

These are the specific problems:

- **Seed confound.** §5.3: "split r uses seed r mod 5". Each random-split value mixes split variance with a single seed, while the semantic value is a mean over five seeds. The two distributions are not comparable: the semantic mean has had seed noise averaged out, and each random point has not.
- **The Mann–Whitney test is undefined.** §5.3 reports "semantic above all ten, Mann–Whitney p = 0.01" but never says what the two samples are. If it is 5 semantic seeds against 10 random splits, and all five semantic seeds exceeded all ten random values, the exact two-sided p would be about 0.0007. If one sample is a single mean, Mann–Whitney is the wrong test. The reported p = 0.01 cannot be reconstructed from the text.
- **"Above all ten" with a tied maximum.** Table 6 gives the random range as 0.949–0.953 and the semantic value as 0.953, so "above all ten" depends on unreported decimals.
- **Text against test.** §5.3 says "Once the streams are fused the differences almost vanish". On TON-IoT, though, the semantic split *does* help late fusion (+0.002, the authors' own p = 0.01). The conclusions (⟦random-split sentence⟧, §6.1 "Third, ⟦…⟧", and §6.3 "the random-split control bounds what any grouping of these features can add") need to state this honestly: the effect is small and significant on TON-IoT, and not detectable on UNSW-NB15.
- **The bound is overstated.** Ten random splits do not bound "what any grouping can add". They sample ten groupings out of a combinatorially large space.

**Fix:**

- Train every random split with all five seeds, or at least three. If compute forbids this, compare the semantic split's *single-seed* values (seed r matched to split r) with the random splits.
- State the test exactly: which samples, one- or two-sided, and the exact or asymptotic method. A permutation or rank test on split-level means (semantic seed-mean against random split-level seed-means) is the natural choice.
- Report a size-matched effect estimate with a CI.
- Soften the "bounds what any grouping can add" sentence.

### M5. The scope of the claims exceeds the evidence, starting with the title.

The title and §6.1 make a general claim: "Flow features of one record are not modalities … do not need different encoders". The §7 conclusion says fusion "did not improve intrusion detection beyond what one network, or a random forest, achieves". The evidence behind these claims has several limits:

- (a) Only one non-saturated IoT dataset. N-BaIoT is at the ceiling (§5.1: "this dataset cannot separate the architectures"), and UNSW-NB15 is enterprise traffic (§3.1: "it is not an IoT dataset").
- (b) Very small streams: UNSW-NB15 has 10 features in total, split 6 + 4. Published fusion designs typically split 40–80 features.
- (c) Small CNN encoders only.
- (d) A significant positive late-fusion effect on TON-IoT, which contradicts the unqualified wording in §7.

The abstract's "Late fusion added +0.005 and +0.007" treats the UNSW-NB15 value as if it were established. It is not: CI [−0.002, +0.016], Holm p = 0.538.

**Fix:**

- Qualify the title and conclusions, for example by scoping them to "within-record flow-feature groups, small encoders, these benchmarks".
- In the abstract and §7, separate the significant TON-IoT effect from the non-significant UNSW-NB15 one.
- Add a second non-saturated IoT benchmark on which fusion papers actually report results. CICIoT2023 or Edge-IIoTset would be good choices; Edge-IIoTset also offers a richer feature set. This would answer both the "only one IoT dataset" and the "UNSW is not IoT" objections and improves fit to the journal.
- For UNSW-NB15, explain why RFECV kept only 10 features, and show that the conclusions hold with a larger feature set (e.g. the top 30 by MI). Six-plus-four is barely a "multi-stream" setting.

### M6. The pre-processing choices may disadvantage stream 2 and favour the RF. They need to be documented and checked.

These choices raise concerns:

- **SMOTE on categorical fields.** SMOTE runs on the training split (§3.1). On TON-IoT, stream 2 consists largely of categorical fields: "ports, protocol, service, connection state, DNS fields". The paper never says how these are encoded (label, ordinal or one-hot). SMOTE interpolation produces non-existent category values, which affects the CNN and RF differently, and stream 2 most of all.
- **Unclear order of fitting.** It is unclear whether the quantile transform (§4.1) is fitted on pre- or post-SMOTE rows. It is also unclear whether the RF is trained on the SMOTE'd set.
- **Selection biased towards the RF.** Feature selection uses "mutual information followed by recursive feature elimination with cross-validation" with an estimator the paper does not name. If that estimator is tree-based, selection is tuned to the model family the paper then declares the winner.
- **Unequal tuning.** No hyperparameter search is reported for either model. The RF uses a fixed configuration and the CNN uses fixed settings, so the claim "No CNN beats the RF" (§5.2) is a comparison of untuned models.
- **High benign FPR on UNSW-NB15.** A benign FPR of 0.15–0.21 for every model (§5.1) is far above what the literature reports. This is probably a result of deduplication and the cross-split guard, but readers will suspect a pipeline problem unless it is explained.

**Fix:**

- State the categorical encoding, the RFECV estimator, the order in which SMOTE, the quantile transform and the RF are fitted, and whether the RF sees SMOTE'd data.
- Add a no-SMOTE (class-weighted) run of the main TON-IoT and UNSW-NB15 comparison.
- Report a small, equal-budget hyperparameter search for both families, or at least a sensitivity check (CNN learning rate/width; RF depth).
- Explain the UNSW-NB15 FPR, for example by showing the normal-class confusion row and comparing it with a no-deduplication run.

### M7. The attention-weight analysis is weaker than its conclusions.

These are the problems:

- **Low power.** The Spearman correlations in §5.4 are computed over 10 classes per seed (9 on N-BaIoT). With n = 10, ρ = 0.55 has a 95% CI of roughly [−0.1, 0.9], so "moderate on UNSW-NB15" is not distinguishable from "weak on TON-IoT".
- **Wrong reference measure.** "Stream advantage" is measured with *separately trained* single-stream CNNs, not with the attention model itself. A weight can be a faithful description of the fused model's routing and still disagree with what independent models find.
- **Weights not identifiable.** In α₁e₁ + α₂e₂ (Fig. 1), the scale of the embeddings is not constrained. The network can compensate a small α with a large ‖e‖, so α by itself is not a measure of reliance.

**Fix:**

- Measure stream reliance *inside the attention model*: the per-class drop in F1 when e₁ or e₂ is zeroed or mean-replaced, or when one stream's inputs are permuted. Correlate that with α.
- Report α‖e‖, or normalise the embeddings.
- Give CIs or permutation p-values for the per-seed ρ values, and pool over seeds where that is appropriate.
- Keep the seed-instability result. It is the most convincing part of the section.

### M8. The novelty claim and its literature basis need to be made systematic.

§2.1: "In a sample of 14 recent fusion and multi-view IDS papers that we checked (five in full text, nine from their abstracts), none reported a random grouping of features, none a single network with matched parameters, and at most two a tree ensemble". An abstract cannot show that a control is *absent*. This sentence carries the novelty claim (Contribution 2, "have not been combined in fusion-IDS work"), so it has to be verifiable.

Related work also already makes the paper's central point in part. Li et al. (2025) conclude that "multi-view learning is not the best choice in all scenarios" on the same two datasets, and Ayantayo et al. (2023) compare early and late fusion on UNSW-NB15. The incremental contribution over these needs to be stated explicitly.

**Fix:**

- Read all 14 papers in full, or drop the abstract-only ones from the "none" claim.
- Add a table (supplementary is fine) listing each paper, its datasets, its baselines, whether it used a random grouping, a parameter-matched control, a tree ensemble and multiple seeds, and its reported gain.
- Add one paragraph that positions this work against Li et al. (2025) and Ayantayo et al. (2023).

---

## MINOR issues

1. **The Holm family is inconsistent (§4.3).** The paper lists "late against early, attention against early, late against attention, and the two capacity comparisons, on each dataset" and calls this "one family of 9 paired t-tests". Three comparisons on three datasets already make 9. Adding two capacity comparisons per dataset makes 15, or 11 if the capacity comparisons are on TON-IoT only. Fix the count and list the tests in a table. Consider leaving N-BaIoT (at the ceiling) out of the confirmatory family, or report it as pre-specified anyway.
2. **The RF−CNN paired comparison pairs on arbitrary seeds** (Table 3, "RF (all features) − late fusion … (5/5)"). RF seed k has no natural link to CNN seed k. Use an unpaired comparison (Welch) or say explicitly that the pairing is nominal.
3. **The "silent failure" claim in §5.5 is overstated.** "only the random-forest reference reveals it" (§5.5 and Contribution 4) is too strong. The authors themselves note that "the best validation epoch … was at most 4" and that "the validation loss diverges". A macro-F1 of 0.028 on 10 classes is also far below what a constant majority-class predictor would score. Training curves and a trivial baseline would have exposed the failure too. Rephrase as "a tree reference is a cheap and reliable detector".
4. **The BatchNorm explanation in §5.5 is asserted, not shown.** The text says BN "estimates statistics during training that do not match evaluation". Support it with one ablation: no BN, BN with a log transform, or BN evaluated with batch statistics. Otherwise present it as a hypothesis.
5. **The earlier version (§1, §5.5).** If the earlier dual-stream version was *published*, the authors should cite it and state whether a correction or erratum is being issued. If it was not published, say so.
6. **N-BaIoT design.** The split is random by device and class, and Kitsune features are time-windowed statistics, so temporally adjacent windows can fall in both training and test sets. Either use a device-held-out evaluation, which might move N-BaIoT off the ceiling and make it informative, or move N-BaIoT to the supplement as a ceiling check.
7. **Fig. 2 labels.** The x-tick labels overlap and are unreadable in all three panels ("Late fusion"/"Attention fusion", "stream 1"/"stream 2"). The y-axes use different ranges per panel (0.994–1.000 on N-BaIoT), which visually exaggerates differences of a few thousandths. Rotate or abbreviate the labels, and say in the caption that the axes differ.
8. **Fig. 3 labels.** Class labels overlap (ransomware/normal on TON-IoT; mirai udp/normal on N-BaIoT) or are truncated ("reconnaissan", "mirai udppla"). The caption should say what the error bars are: the range or the SD over seeds. The N-BaIoT x-axis spans only 0–0.015, which makes "stream advantage" there essentially noise. Say this explicitly.
9. **Unlabelled tables.** Tables 4 and 5 (per-class F1) have no captions in the manuscript. The text refers to them, but the markdown has no "Table 4" or "Table 5" labels. State whether the per-class values are seed means.
10. **Number formatting.** Table 3 shows "−0.000" (and "1/5" with it). Report three or four decimals consistently, or write "0.000". In §5.4, "-0.20" uses a hyphen instead of a minus sign.
11. **Abstract wording.** "The RF on all features matched or beat every fusion model (0.957 and 0.605 against 0.953 and 0.569)" is true at the macro level. Table 5 shows the CNNs ahead of the RF on analysis and backdoor in UNSW-NB15. Add "in macro-F1".
12. **Missing threshold-free metrics.** For an IDS venue, the paper should report at least one threshold-free metric (e.g. macro PR-AUC) and attack-class recall alongside macro-F1 and benign FPR.
13. **Inference latency (§4.3/§5.6, pending).** When this section is filled in, report per-record latency with batch size 1 *and* batched, on a constrained device or with a CPU-thread cap, and give model size in KB. Laptop CPU latency alone does not say much about IoT deployment.
14. **Stream definition provenance.** §3.2 says the split was "decided before any model was trained", but it is also "the split used in the earlier version of this work", which had seen results. Say so plainly.
15. **Generative-AI declaration.** It states that Claude was used "to write the analysis code and to draft the paper". This is acceptable under Elsevier policy, but the released code should make clear that the authors checked it. A short statement on code verification, such as unit tests or independent reruns, would help.

---

## What would raise the score

- **To about 7:**
  - Multi-split uncertainty (M1).
  - The full capacity/ensemble control set, including duplicated-input late fusion, on two datasets (M2).
  - An MLP-encoder replication and the corrected early-fusion order statement (M3).
  - A properly specified random-split test with seeds and splits not confounded (M4).
  - Claims rescoped in the title, abstract and §7 (M5).
- **To about 8:** in addition, a second non-saturated IoT dataset such as CICIoT2023 or Edge-IIoTset (M5); a systematic literature table (M8); and an ablation-based check of the attention weights (M7).

With these changes, the paper would be a solid, citable methods paper for *Internet of Things*. Its main value would be the checklist of controls that fusion-IDS papers should report, and that is a contribution the field needs.
