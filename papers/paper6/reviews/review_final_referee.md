# Referee report, final round

**Manuscript:** "Fusing Feature Groups of One Flow Record: A Controlled Evaluation of Multi-Stream Intrusion Detection on IoT and Network Benchmarks"
**Journal:** Elsevier *Internet of Things*
**Date:** 2026-11-07
**Materials read:** Manuscript_filled.md (full), captions in Manuscript_v2.docx, Highlights.docx, figures/fig1–fig3, review_r2_referee.md (to check whether earlier points were fixed). No experiments were run.

**Score:** 7.5 / 10 (round 2: 7.0)
**Recommendation:** Minor revision. No new experiments are strictly needed. The remaining issues are about how the bounds are stated, one misattribution, and making the IoT trade-off visible in the abstract. One cheap CNN tuning check (M3) would make the tree-versus-network claim much harder to attack, and I recommend it.

---

## Summary

The paper asks whether splitting the features of one flow record into two "semantic" streams, encoding each with its own small 1D-CNN and fusing them (early, late or attention fusion) improves intrusion detection. It tests this once the obvious confounds are controlled: more features, more parameters, the choice of grouping, and a strong tabular baseline. Datasets are TON-IoT, UNSW-NB15 (not IoT, 10 selected features) and N-BaIoT (saturated, used as a ceiling check).

The design is thorough for a paper of this kind:
- five seeds with paired differences and an 8-test Holm family;
- ten random splits of the same sizes, compared within seeds;
- three capacity and architecture controls: wide early fusion, early fusion with the late-fusion head, and late fusion on all features;
- random column orders and a full MLP-encoder replication;
- five further data splits;
- tuned RF and GBDT on all features and on each stream;
- an in-model ablation of attention weights;
- a documented failure caused by input scaling.

The findings are consistent:
- The gain of the fused model over its branches is the gain from using all features. An RF gets the same gain (+0.032 and +0.061 against +0.044 and +0.060).
- Late fusion adds +0.005 on TON-IoT (Holm p = 0.006). Most of this is recovered by early fusion with the late-fusion head, or by late fusion on all features.
- Random groupings serve the fused models almost as well as the semantic one.
- Tuned tree ensembles beat every network in macro-F1 on both non-saturated datasets.
- Attention weights track how the model routes information (ρ ≈ 0.93 with in-model reliance), but not which stream is more useful.

The paper is honest and self-critical. It discloses that the stream rule came from an earlier version whose results the authors had seen, that the equivalence margin was set after the results, that two Holm tests were added after review, and that GPU non-determinism is of the order of the small effects. Its contribution is methodological: a checklist of controls, and evidence that the usual branch-versus-fused comparison cannot support claims about fusion.

## Status of the main round-2 points

| R2 point | Status now |
|---|---|
| N1 general title / one IoT dataset | **Partly.** The title is rescoped ("on IoT and Network Benchmarks") and §6.1 now says "on these benchmarks". There is still no second non-saturated IoT dataset (see M1). |
| N2 absence of evidence read as no effect | **Mostly.** A ±0.01 margin is introduced and labelled post hoc, and the UNSW-NB15 grouping effect is called "inconclusive" in §5.3 and §6.1. But the abstract and parts of §5.3 still state the point estimate as a bound (see M2). |
| N3 Table 7 does not match Table 2 | **Fixed.** There are now three rows per dataset with the protocol stated, and the caption explains why the RF values differ. |
| N4 PR-AUC and detection rate promised but missing | **Fixed** in the §5.1 text. A column would still be better (minor m3). |
| N5 tuned trees vs untuned CNNs; UNSW-NB15 FPR | **Partly.** The FPR is now explained: 82% of the misclassified benign records go to fuzzers. The RF lead on the duplicate-free subset is reported (+0.073). The CNNs are still untuned; this is acknowledged in §6.3 but not tested (see M3). |
| n4 head attribution | **Partly.** The wording now says "mostly from the wider head". The direct difference is still not reported, and "wider" is the wrong word (see M4). |
| n1 caption fields | **Fixed.** Captions in the .docx now carry the correct numbers. |
| n10 compact tree | **Partly.** A compact forest from the companion study is quoted in §5.6, but the abstract and highlights do not reflect what it implies (see M5). |
| n5, n11, n12, n14, n15, n17 | **Mixed.** n5, n12, n14 and n15 are still open (minors m1, m2, m6, m7). n11 is mostly fixed ("p < 0.001", minus signs), except "1 ties". n17 is partly done: the "none" count is now restricted to the eight full-text papers. |

---

## Major issues

### M1. The IoT evidence rests on one non-saturated dataset, and this should be clearer in the framing for an IoT journal.

**Location:** Title; Abstract; §6.1, second paragraph; §6.3.

**Why it matters:** TON-IoT is the only IoT dataset that can separate the architectures. UNSW-NB15 is enterprise traffic reduced to 10 features (a 6 + 4 split, which makes "two streams" nearly degenerate). N-BaIoT is saturated, and its train and test records share time windows. The rescoped title is acceptable. But the abstract opens with "Intrusion detection studies increasingly…" and closes with "On these benchmarks…", and a reader of *Internet of Things* will still ask why Edge-IIoTset or CICIoT2023 is missing. Three papers in Table A.1 use those datasets. §6.3 concedes the point ("the IoT evidence rests mainly on one dataset, TON-IoT"), but only in the limitations.

**Fix:** Preferred: add Edge-IIoTset or CICIoT2023 for the core comparisons only (early, late, late on all features, random splits, RF). This is CPU-cheap for the trees and modest for the CNNs. If that is not possible, add one sentence to the abstract saying that TON-IoT is the only non-saturated IoT benchmark in the study. Also say in §1 (contribution 1) that UNSW-NB15 is included as a non-IoT replication.

### M2. The abstract and §5.3 still state the point estimate as an upper bound on the grouping effect, which contradicts §5.3's own CI reading.

**Location:**
- Abstract, last sentence: "semantic grouping of one record's features added at most about 0.01 macro-F1."
- §5.3: "The semantic split therefore gives late fusion at most about 0.01 of macro-F1 over the random-split mean".
- §7: "within about 0.01 on UNSW-NB15".

**Why it matters:** On UNSW-NB15 the within-seed difference is +0.009 [−0.001, +0.019]. Late fusion minus late fusion on all features is +0.011 [+0.004, +0.018] (5/5 seeds, unadjusted p = 0.013). §5.3 and §6.1 correctly say that "a grouping effect of up to about 0.02 cannot be excluded". "At most about 0.01" is therefore a statement about the point estimate, not a bound, and the abstract is the sentence most readers will quote.

The UNSW-NB15 picture also leans the same way across tests: the semantic split beats 9 of 10 random splits, late beats late-on-all in 5/5 seeds, and on the duplicate-free subset late minus early rises to +0.025. That is a consistent, if unresolved, hint that grouping helps a little on this dataset. It deserves an even-handed sentence rather than a ceiling.

A related problem is "(not significant)" for late minus early on UNSW-NB15 (+0.016 [+0.001, +0.031]). The unadjusted CI excludes zero, so the test is non-significant only after Holm correction. The abstract should say "not significant after Holm correction" for both datasets.

**Fix:** Rewrite the abstract's closing sentence along these lines: "semantic grouping changed fused-model macro-F1 by +0.001 on TON-IoT (negligible) and +0.009 on UNSW-NB15, where effects up to about 0.02 could not be excluded." Make the same change in §5.3 ("at most about 0.01" → "point estimates of at most about 0.01") and §7. Also state that the confirmatory family does not contain the semantic-versus-random test, which is the paper's central grouping question. Late versus late-on-all is the confirmatory proxy, and the random-split comparison is descriptive.

### M3. The tree-versus-network comparison is tuned against untuned, and on TON-IoT the margin is small.

**Location:** §4.2 ("The networks, in contrast, use one fixed configuration and were not tuned per dataset"); §5.2 ("No CNN beats the RF on all features in macro-F1"); Abstract ("Tree ensembles beat every fusion network"); Highlight 3; §6.3.

**Why it matters:**
- The RF and GBDT got a validation grid search. The CNNs got none.
- On TON-IoT the RF beats late fusion by only +0.008. That is about the size of the effects the paper treats as meaningful, and smaller than typical gains from tuning learning rate, width and dropout on tabular networks.
- On UNSW-NB15, the column-order spread of early fusion alone (0.527–0.565) shows that the CNNs are sensitive to configuration choices.
- The +0.034 UNSW-NB15 margin and the +0.073 duplicate-free margin are probably robust. The TON-IoT one is not demonstrated.
- The paper's own wording, "might narrow the gap to the trees" (§6.3), concedes the point.

**Fix:** Run a small validation grid for early and late fusion on TON-IoT and UNSW-NB15 (learning rate {3×10⁻⁴, 10⁻³}, width ×{1, 2}, dropout {0.1, 0.3}), select on one seed and report five seeds. This is the cheapest experiment that would close the most obvious rebuttal. If it cannot be run, qualify the tree claim in the abstract and Highlight 3: "tuned tree ensembles beat every untuned fusion network in macro-F1". Also add "in macro-F1" to Highlight 3 in any case. The CNNs are ahead per class on UNSW-NB15 analysis and backdoor (Table 5). This is round-1 minor 11, still open.

### M4. The late-fusion advantage on TON-IoT is credited to a "wider head", but the control adds a layer, and the decisive difference is never reported or tested.

**Location:** §5.2: "Giving early fusion the late-fusion head recovered most of the TON-IoT advantage (+0.004 of +0.005) … On TON-IoT the advantage therefore comes mostly from the wider head, not from the number of parameters or from the semantic grouping." §6.1: "comes mostly from its wider head". Abstract: "a gain that late fusion with both encoders on all features reproduced".

**Why it matters:**
- In §4.1, early fusion is encoder → 64-unit embedding → dropout → softmax. "Early fusion with head" adds a 128-unit dense layer, so the control adds depth (an extra non-linear layer), not width. Wide early fusion, which does add width, gained only +0.002.
- So the evidence points to the extra hidden layer, not to "wider".
- The attribution also rests on subtracting two separate differences (+0.005 − +0.004). The direct comparison, late fusion minus early fusion with head, is not in Table 3 and was asked for in round 2 (n4). By subtraction it is about +0.001–0.002, and its CI is unknown.
- "Reproduced" in the abstract is too strong. Late-on-all is 0.951 against 0.953 for late fusion and 0.948 for early fusion, so it recovers about 60% of the gain. The difference to late fusion is +0.001 [0.000, +0.003], which is not an equivalence result.

**Fix:**
- Add "Late fusion − early fusion with head" to Table 3 with its CI.
- Replace "wider head" with "the extra 128-unit dense layer (head)".
- In the abstract, write "a gain that late fusion with both encoders on all features largely reproduced".

### M5. The tree advantage depends on 221–636 MB forests, and at an IoT-relevant size it disappears. The abstract and highlights omit this.

**Location:** §5.6, last paragraph; Abstract; Highlight 3; §6.1.

**Why it matters:** For this journal, footprint is part of the result. The tuned RF that "beats every fusion network" occupies 221 MB on TON-IoT and 636 MB on UNSW-NB15, against 343 kB for late fusion. The companion-study forest of 4.4 MB scores 0.946 on TON-IoT. That is *below* late fusion (0.953) by 0.007, more than the late-minus-early effect the paper calls significant, and it is still about 13 times larger than late fusion. So at a deployable size the ranking on TON-IoT reverses or ties.

§5.6 says this carefully, but the abstract and Highlight 3 present the trees as winning without qualification. Many readers will take away "use an RF instead". The paper's real message is subtler: the trees are the right *diagnostic reference*, while the CNNs remain competitive per kilobyte.

There is also a provenance problem. The compact-forest numbers come from an uncited companion study (see m7). The reader cannot check them, and they are not in this paper's result files.

**Fix:**
- Add one clause to the abstract, for example: "…except on the saturated N-BaIoT, at a memory cost of hundreds of megabytes; a 4.4 MB forest scored below late fusion on TON-IoT."
- Qualify Highlight 3 in the same way.
- Either cite the companion study properly or reproduce the compact-forest point here. The latter takes minutes on CPU and is covered by this paper's own verify script.

### M6. Test-set duplicates on the selected features change the UNSW-NB15 conclusions materially, but they appear in only two sentences.

**Location:** §3.1, last paragraph (16.3% of UNSW-NB15 test records coincide with a training record on the selected features); §5.2 (late minus early on duplicate-free records becomes +0.025 [+0.001, +0.048]; the RF lead becomes +0.073).

**Why it matters:**
- Removing these records changes late minus early by +0.009 and more than doubles the RF lead (+0.034 → +0.073). These are among the largest swings in the paper, larger than most effects in Table 3.
- The reader is not told whether these coinciding records carry the *same* label as their training match or a *different* one. That is projection-induced label noise from reducing 40 features to 10, and it is probably linked to the benign/fuzzers overlap behind the 13–21% FPR.
- If many duplicates carry conflicting labels, the full-test UNSW-NB15 numbers are partly measuring irreducible error.
- The doubling of the RF advantage on the duplicate-free subset also needs a sentence of explanation. It implies the CNNs do relatively *better* on duplicated records, which is the opposite of the usual memorisation concern.

**Fix:**
- Add a short table, or rows in Table 3, with the duplicate-free results for the main comparisons and the RF on UNSW-NB15. This is computable from the saved predictions; the random-split runs are excluded because their predictions were not kept, as the paper already states.
- Report the share of coinciding test records whose training match has a different label.
- Add one sentence interpreting the RF swing.

---

## Minor issues

**m1. "Late fusion on random splits also stays above early fusion on TON-IoT (by +0.004 on average, all ten splits)" (§5.3, end).** Early fusion was not trained on random splits, because it has no streams. The comparator must be the seed-matched semantic early-fusion run, or the column-order runs. Say which. This was round-2 n5 and is still open.

**m2. Figure 3 label overlap.** This is still present: "exploits" overlaps "backdoor", "shellcode" overlaps "dos" (UNSW-NB15); "mirai udp" overlaps "mirai udpplain"; "gafgyt junk" and "gafgyt scan" collide (N-BaIoT); "ransomware" and "normal" touch (TON-IoT). The caption also does not say that the N-BaIoT x-axis spans only about 0.013, which makes the vertical spread look more meaningful than it is. Use adjustText or leader lines, and note the x-range.

**m3. PR-AUC and detection rate appear only in §5.1 prose.** Add them as columns in Table 2, at least for the eight semantic-split CNNs and the two trees, so that the claim "the other metrics rank the models as macro-F1 does" can be checked.

**m4. Holm-adjusted p-values.** Only five of the eight are given in the text, and the two attention-versus-early tests have none. Add a small table of the full family with raw and adjusted p, effect, CI and margin verdict. This was asked for in round 2 (n11).

**m5. Figure 2 and Table 6 compare unlike quantities.** The semantic value is a mean of five seeds, but each random split is a single seed, so the grey spread includes seed noise that the red diamond does not. Statements such as "semantic above 9 of the ten" are therefore slightly biased in favour of the semantic split when seed variance is large, as on UNSW-NB15 (SD up to 0.011). Note this in the caption, or plot the semantic seeds individually.

**m6. Reference: Sanjalawe et al.** The citation is dated 2025, vol. 5, but the DOI is 10.1016/j.iotcps.2026.03.002. Verify the year, volume and pages. This was round-2 n15 and is unchanged.

**m7. Companion studies uncited.** §3.1 ("the corrected pre-processing pipeline of our companion studies", "the preparation of our federated-learning study") and §5.6 ("Our companion study of the accuracy–footprint trade-off") rely on uncited work. Cite them as under review or as preprints, or state that the relevant code and results are in the released repository. In the anonymised version, handle this through the journal's mechanism for self-citations.

**m8. GPU non-determinism per dataset.** §4.2 gives a mean change of 0.003 and a maximum of 0.018 for same-seed reruns, pooled across datasets. Report it per dataset. On TON-IoT, where seed SDs are ≤ 0.001, a pooled 0.003 would be alarming. It presumably comes mostly from UNSW-NB15, and the reader should be able to see that.

**m9. Column-order sensitivity on UNSW-NB15.** The SD over orders is 0.015, larger than the seed SD (0.011) and comparable to late minus early (+0.016). State explicitly that the MLP results are the cleaner evidence on UNSW-NB15 for differences below about 0.02 (round-2 n8).

**m10. Silent-failure evidence.** §5.5 says that "the validation loss diverges while training accuracy rises", but no curve is shown. Add a supplementary figure of the training and validation loss of the min–max stream-1 CNN on TON-IoT against the quantile run. It is the most persuasive evidence for contribution 4 and costs nothing. Also consider one sentence noting that standardisation or a log transform are the usual remedies, which makes the failure a known pitfall rather than a new one. That is fine, but frame it as such.

**m11. Quantile transform fitted on the oversampled set.** §4.1: the transform is fitted on a subsample of the SMOTE-augmented training set, so synthetic records shape the quantiles. This is harmless but should be stated as a choice. Fitting on real records only would be cleaner.

**m12. Table A.1.** Add a "multiple seeds/runs reported" column, as requested in rounds 1 and 2. It supports contribution 2 directly.

**m13. Table 6, "(3/5, 1 ties)".** Write "1 tie".

**m14. Figure 1.** It omits the dropout layers and the 64-unit embedding inside each encoder. It also does not show the three capacity controls, which carry the key argument of §5.2. A compact fifth panel or an inset for "early fusion with head" and "late fusion on all features" would help.

**m15. Abstract density.** The abstract is 235 words with about 15 numerical values. Consider dropping the attention clause or the N-BaIoT clause to make room for the footprint qualifier (M5).

**m16. Related work.** The point that tuned tabular deep models narrow the gap to trees (§6.3) would benefit from citing Gorishniy et al. (2021, "Revisiting deep learning models for tabular data", NeurIPS 34), which is directly about tuned MLPs and ResNets against GBDT. Verify the citation before adding it.

**m17. Wording.** §5.2: "A fusion paper that reports only branch-versus-fused comparisons is reporting this effect" is slightly polemical. "…cannot distinguish fusion from this effect" is safer and is what the evidence shows.

---

## Things done well

- **Control design.** Random splits of the same sizes, capacity-matched networks, late fusion on all features, an order-independent encoder and per-stream trees together form a genuinely useful template for the fusion-IDS literature. Table 3 is a model of how to report such a study.
- **Statistical hygiene.** Paired within-seed differences with t-CIs and counts, Welch for trees against networks, an explicit and small Holm family with N-BaIoT excluded, and data-split replicates kept descriptive.
- **Honesty.** The paper discloses that the stream rule came from a version whose results were seen, that the margin was chosen post hoc, that two family members were added after review, the GPU non-determinism, the reuse of earlier runs and the RFECV–RF coupling. Very few IDS papers are this candid.
- **Section 5.5.** It turns the authors' own earlier error into a transferable lesson, and Table 7 now isolates the scaling change cleanly from the protocol change.
- **Section 5.4.** It separates "weights describe routing" (ρ ≈ 0.93 with in-model reliance) from "weights explain usefulness" (ρ = 0.03 and 0.54), a distinction that is often blurred.
- **Reporting of round-2 fixes.** Table 7 is fixed, PR-AUC and detection rate are now reported, the UNSW-NB15 FPR is explained, captions are correct in the .docx, the abstract is no longer circular, and the "none" count in §2.1 is restricted to the full-text papers.
- **Reproducibility.** The code and result files are released, and an automated check of every number in the text is in place.

## Assessment

- **Novelty:** Modest but real and methodological. No single control is new; the combination, applied to a popular design pattern, is.
- **Soundness:** Good. The main gap is the tuning asymmetry (M3).
- **Conclusions versus evidence:** They follow on TON-IoT. On UNSW-NB15 a few sentences still turn a point estimate into a bound (M2), and the deployment qualifier on the tree result is missing from the abstract (M5).
- **Clarity:** High, although the paper is dense.
- **Fit:** Adequate for *Internet of Things*. A second IoT benchmark would make it strong (M1).

**Path to accept:** M2, M4 and M5 are text changes. M6 can be computed from saved predictions. M3 needs either a small CNN grid or qualified wording. M1 needs either a second IoT dataset or a clearer statement in the abstract.
