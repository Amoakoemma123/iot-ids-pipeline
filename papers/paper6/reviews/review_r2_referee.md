# Referee report, Round 2

**Manuscript:** "Feature Groups Are Not Modalities: A Controlled Evaluation of Multi-Stream Fusion for Flow-Based IoT Intrusion Detection" (revised)
**Journal:** Elsevier *Internet of Things*
**Date:** 2026-11-03
**Materials read:** Manuscript_filled.md (full), Manuscript_v2.docx (captions and fields), Highlights.docx, figures/fig1–fig3, review_r1_referee.md

**Score:** 7.0 / 10 (round 1: 5.5)
**Recommendation:** Minor revision. This is at the major/minor boundary. It is minor only if the authors rescope the title and claims as asked in N1/N2. If they want to keep the general claim in the title, they need a second non-saturated IoT dataset, and that makes it a major revision.

---

## Summary

The revision asks the same question as before: once the obvious confounds are controlled, does splitting one flow record's features into semantic streams and fusing per-stream 1D-CNN encoders (early, late or attention fusion) help intrusion detection? The authors have added most of the controls requested in round 1:

- five further stratified data splits;
- three capacity/architecture controls on both non-saturated datasets: wide early fusion, early fusion with the late-fusion head, and late fusion with both encoders on all features;
- five random column orders and a full MLP-encoder replication;
- a tuned GBDT next to a tuned RF;
- a within-seed paired design for the random-split comparison;
- an in-model ablation of the attention streams;
- an 8-test Holm family that is now stated explicitly;
- an appendix table of the 14 surveyed fusion papers.

The results tell a coherent story:

- The gain of fusion over single streams is the gain from using all features.
- Late fusion adds about +0.005 macro-F1 on TON-IoT (Holm p = 0.006). This is largely reproduced by giving early fusion the late-fusion head, and by late fusion on all features.
- Random groupings serve the fused models almost as well as the semantic one.
- Tree ensembles beat every network in macro-F1 on both non-saturated datasets.
- Attention weights describe the model's routing (ρ ≈ 0.93 with in-model reliance) but not stream usefulness.

The paper is careful, self-critical and much more convincing than round 1. Its value is methodological: a set of controls the fusion-IDS literature should adopt, and a clear demonstration that the usual branch-versus-fused comparison measures the effect of having more features.

Three things keep it from a clean accept:

1. The IoT evidence still rests on one non-saturated dataset, TON-IoT. UNSW-NB15 is enterprise traffic with only 10 features, and N-BaIoT is a ceiling check. The categorical title outruns this.
2. Several "no effect" statements are absence-of-evidence inferences drawn from 5 seeds. The power is far lower on UNSW-NB15 than the limitations section admits.
3. There are reporting defects: metrics that are promised but never reported, a table that does not match the main results, stale caption numbering in the Word file, and one uncorrected overclaim from round 1.

---

## Status of round-1 points

### Major points

| # | Round-1 point | Status | Reason (checked against the text) |
|---|---|---|---|
| M1 | Uncertainty conditional on one data split | **Partly** | Five further splits were added (§4.2 "Data splits", §5.2 "Other data splits"), with six split-level observations. However, "The selected features are kept from the original split": feature selection, which uses the labels, was not refitted, so the new test sets overlap the records the selector saw. There is one seed per split. The late-minus-early difference without MITM, which was asked for, is not reported. |
| M2 | Capacity control insufficient; epoch budget | **Addressed** (mostly) | Wide early fusion, early fusion with head and late fusion on all features were run on TON-IoT and UNSW-NB15, and two of them are in the Holm family. A 120-epoch budget is used with best-weight restore, and 1/90 runs hit the cap. The optional two-CNN ensemble was not run. No best-epoch distribution table was given, and the cap statement covers only the "main and capacity" runs, not the MLP, order and split runs. |
| M3 | Feature-order confound; false sentence in §4.2 | **Addressed** | The sentence is corrected. Five random orders were run (§5.3). The MLP replication covers all five architectures and the random splits. GBDT was added. |
| M4 | Random-split test under-specified, seed-confounded, bound overstated | **Partly** | The Mann–Whitney test is replaced by a within-seed paired difference with a CI. Not every split was trained with all seeds, but each seed is paired with two random splits, which is the fallback I offered. However, §6.3 still says "the random-split control bounds what any grouping of these features can add". This is the exact sentence I asked to be softened. |
| M5 | Claims exceed evidence; second IoT dataset; UNSW 10 features | **Partly** | The title now says "Flow-Based IoT", the abstract separates TON-IoT (significant) from UNSW-NB15 (not significant), and §7 says "On the flow-feature benchmarks we studied". The RFECV floor on UNSW-NB15 is explained (0.554 at 10 features against 0.526 with all 40). No second IoT dataset was added; it is only mentioned in §6.3. There is no larger-feature-set check on UNSW-NB15. "Feature Groups Are Not Modalities" and §6.1 remain general (see N1). |
| M6 | Pre-processing may favour RF / hurt stream 2 | **Partly** | The categorical encoding (one-hot, at most 30 levels), the RFECV estimator (50-tree RF), the fitting order and SMOTE for the trees are all now stated. The trees got a small grid search, but the CNNs got none, so the tuning asymmetry is now reversed rather than removed. There is no no-SMOTE/class-weighted run. The UNSW benign FPR of 0.13–0.21 is still not explained: no confusion row and no no-deduplication comparison. |
| M7 | Attention analysis weaker than conclusions | **Addressed** | In-model mean-replacement ablation, reliance–weight ρ with permutation p, the weight-scaled contribution and seed stability are all reported (§5.4). Per-seed CIs are not given, but the permutation p on seed-averaged weights is adequate. |
| M8 | Novelty claim rests on abstracts | **Partly** | Table A.1 was added (8 papers in full text, 6 abstract-only), along with a positioning paragraph on Ayantayo et al. and Li et al. (§2.1). The "none … none" sentence in §2.1 still pools the abstract-only papers. The requested columns for multiple seeds and reported gain are missing. |

### Minor points

| # | Round-1 point | Status | Reason |
|---|---|---|---|
| 1 | Holm family inconsistent | **Mostly** | The 8 tests are listed in §4.3. Not all 8 adjusted p-values are reported anywhere; attention-versus-early has none. There is no table of the family. |
| 2 | Nominal RF–CNN pairing | **Addressed** | Welch is used. |
| 3 | "Only the RF reveals it" overstated | **Addressed** | §5.5: "Training curves and a trivial baseline would have revealed it"; Contribution 4 is rephrased. |
| 4 | BN mechanism asserted | **Addressed** (as hypothesis) | "The most likely mechanism … We did not isolate this mechanism." |
| 5 | Earlier version | **Addressed** | Stated as unpublished (§1). |
| 6 | N-BaIoT temporal leakage | **Partly** | Leakage is acknowledged and N-BaIoT is labelled a ceiling check (§3.1). There is no device-held-out run, which is acceptable. |
| 7 | Fig. 2 labels and axes | **Addressed** | Labels are rotated, and the caption notes that the axes differ. |
| 8 | Fig. 3 labels and error bars | **Partly** | The caption now defines the bars as the range over seeds. Labels still overlap ("ransomware"/"normal" on TON-IoT, "exploits"/"dos" and "fuzzers"/"shellcode" on UNSW-NB15, "mirai udp"/"normal" on N-BaIoT) or are truncated ("reconnaissan", "mirai udppla"). "gafgyt junk" sits outside the plot area. The caption still does not say that the N-BaIoT x-axis spans only about 0.015. |
| 9 | Tables 4/5 unlabelled | **Addressed** | Captions are present and state "mean over five seeds". Caption numbering in the .docx is stale; see n1. |
| 10 | Number formatting | **Partly** | §5.4 still has a hyphen in "(lowest -0.17)". Table 3 and §5.2 report "p = 0.000"; this should be "p < 0.001". |
| 11 | "in macro-F1" in abstract | **Not addressed** | The abstract says "Tree ensembles on all features beat every fusion model except on the saturated N-BaIoT", and Highlight 3 says "matched or beat every fusion network". The CNNs are still ahead per class on UNSW-NB15 analysis and backdoor (Table 5). |
| 12 | Threshold-free metric and attack recall | **Not addressed** (claimed only) | §4.3: "we also report … the attack detection rate and the macro-averaged area under the precision–recall curve". Neither number appears anywhere in §5 or in the tables. See N4. |
| 13 | Latency: batch 1 and batched, size, constrained device | **Partly** | Batch 1 and 1,000-record timings and model sizes are given. There is no constrained device and no thread cap. Table 8's caption promises "and model size", but the table has no size column. |
| 14 | Stream provenance | **Addressed** | §3.2 states it plainly. |
| 15 | Gen-AI declaration and code verification | **Addressed** | An automated number-checking statement was added. |

---

## New MAJOR issues

### N1. The title and §6.1 still state a general claim that rests on one non-saturated IoT dataset and small encoders.

**Title:** "Feature Groups Are Not Modalities".

**§6.1:** "Flow features of one record are not modalities in the multimodal-learning sense … and do not need different encoders", followed by "they show that the evidence offered for fusion within a flow record does not support it."

The evidence comes from three sources:

- **TON-IoT**, the only non-saturated IoT dataset, with 25 features.
- **UNSW-NB15**, which is not IoT, has 10 features and has wide CIs.
- **N-BaIoT**, which is saturated.

All of it uses about 35–87k-parameter encoders. §6.3 concedes that "the IoT evidence rests mainly on one dataset". A reader at this journal will ask why Edge-IIoTset or CICIoT2023 is missing. Fusion papers actually report on these datasets (Table A.1 lists three such papers), and they have richer feature sets on which a two-stream split is less degenerate than 6 + 4.

The "does not support it" sentence also generalises from the authors' streams to the literature's evidence. The literature's evidence was not re-run. It was shown to lack controls, which is a different statement.

**Fix:** Either:

- (a) add one non-saturated IoT benchmark (Edge-IIoTset preferred) for at least the main comparison, the late-on-all control, the random splits and the RF; or
- (b) scope the title, for example "Are Flow-Feature Groups Modalities? A Controlled Evaluation of Multi-Stream Fusion for IoT Intrusion Detection", or keep the current title with the subtitle carrying "on TON-IoT and UNSW-NB15".

In either case, rewrite the §6.1 sentence to: "the comparisons usually offered for fusion within a flow record cannot distinguish it from the effect of using all features, and on our benchmarks the controls remove most of the apparent gain."

### N2. Absence of significance is repeatedly read as absence of effect, and the stated power is wrong for UNSW-NB15.

Examples:

- **Abstract:** "on TON-IoT, late fusion with both encoders on all features matched it, so the gain comes from the architecture, not the grouping". The supporting result is late minus late-on-all = +0.001 [0.000, +0.003], 4/5 seeds, Holm p = 0.135, which is not a test of equivalence.
- **§6.1:** "the semantic grouping itself contributes almost nothing … by about 0.01 at most, and not significantly".
- **§5.4 and abstract:** attention weights are "not tied" to stream usefulness, based on ρ = 0.54, p = 0.11 with ten classes on UNSW-NB15.

On UNSW-NB15 the grouping effect estimates are of the same size as the effects the paper elsewhere treats as a gain:

- late minus late-on-all is +0.011 [+0.004, +0.018], 5/5 seeds;
- semantic minus random late fusion is +0.009 [−0.001, +0.019], 5/5 seeds;
- Highlight 2 advertises late fusion's "+0.016".

So "the grouping contributes almost nothing" holds on TON-IoT. On UNSW-NB15 the honest summary is "a possible gain of about 0.01 that this design cannot resolve".

§6.3 says "Five seeds and six data splits give limited power for differences below about 0.005 macro-F1". This is true for TON-IoT only. The UNSW-NB15 paired CIs have half-widths of 0.010–0.015 (Table 3), so the minimum detectable difference there is about 0.02.

**Fix:**

- Pre-state a practical-equivalence margin (for example ±0.005 or ±0.01 macro-F1) and report TOST or CI-within-margin conclusions for the key nulls (late versus late-on-all, semantic versus random, attention versus early).
- Reword "matched" as "came within +0.001 [0.000, +0.003]".
- Give dataset-specific detectable effects in §6.3.
- Rewrite the UNSW-NB15 grouping sentences in §5.2, §5.3, §6.1 and §7 to say "not resolved", not "almost nothing".

### N3. Table 7's "quantile" rows do not match Table 2, although the caption says they are the final run.

Table 7 caption: "the final run a quantile transform. Architecture, data, seeds and training were otherwise identical."

- **UNSW-NB15**, Table 7 quantile row: 0.498 / 0.362 / 0.562 / 0.569 / 0.559. Table 2: 0.497 / 0.361 / **0.557** / **0.573** / **0.565**.
- **RF, stream 1:** 0.927 in Table 7 against 0.928 in Table 2 (TON-IoT); 0.545 against 0.546 (UNSW-NB15).

The late-fusion and attention differences on UNSW-NB15 (0.004–0.006) are of the size of the paper's effects. The values look like they come from the earlier 60-epoch protocol. If so, the min–max row was also trained under that protocol, and the comparison is protocol-matched but is not "the final one".

**Fix:** State which protocol each row comes from, and change the caption accordingly. If both rows are from the 60-epoch protocol, say so and note that Table 2 supersedes them. The verify script should also cover cross-table consistency.

### N4. Metrics promised in the methods are never reported.

§4.3: "we also report the benign false-positive rate …, the attack detection rate and the macro-averaged area under the precision–recall curve (one-vs-rest)". §4.2 also notes that the reused runs "have no PR-AUC". Neither PR-AUC nor detection rate appears in any table or paragraph of §5. For an IDS venue this was a round-1 request (minor 12), and the text now claims it was done.

**Fix:** Add PR-AUC and attack detection rate columns to Table 2, or a supplementary table, for at least the semantic-split models. If they are left out, delete the sentence.

### N5. The tree–network comparison is now tuned against untuned, and the UNSW-NB15 FPR remains unexplained.

§4.2: the RF and GBDT hyperparameters "are chosen on the validation macro-F1". The CNNs use one fixed configuration (learning rate 10⁻³, fixed widths, dropout 0.3).

The trees' advantage is large on UNSW-NB15 (+0.034), so it is probably robust, but it is small on TON-IoT (+0.008). The conclusion "No CNN beats the RF" therefore needs a matched-budget check, at least on TON-IoT.

Separately, every model labels 13–21% of benign UNSW-NB15 records as attacks. This is still not explained, and readers will attribute it to the pipeline. This was round-1 M6.

**Fix:**

- Run a small validation grid for late and early fusion on TON-IoT and UNSW-NB15: learning rate {3×10⁻⁴, 10⁻³}, width ×{1, 2}, dropout {0.1, 0.3}, one seed for selection and five for the final model. Report whether the RF margin survives.
- For the FPR, show the normal-class confusion row (which attack classes absorb the benign records) and one sentence relating it to deduplication and to the 16.3% of test records that coincide with a training record on the selected features.
- Also report the RF-minus-late difference on the duplicate-free UNSW-NB15 test subset. Only the late-minus-early difference is given there (§5.2).

---

## New MINOR issues

**n1. Caption numbering in the Word file is stale.** Every caption in Manuscript_v2.docx reads "Table 1." or "Figure 1." (12 SEQ fields with cached value 1), and settings.xml has no `updateFields` flag. The text cross-references are plain text ("Table 4"), so the PDF the editor builds will show captions that disagree with the text.

- **Fix:** Update the fields in Word (Ctrl+A, F9), or run update_word_fields.py, before submission and check the PDF proof.

**n2. The abstract is exactly at the limit.** By whitespace count it is 250 words, against a 250-word limit. Editorial Manager counts tokens such as "min–max", "+0.044" and "(RF)" differently and may reject it.

- **Fix:** Cut 20–30 words. It is also number-dense: 14 numerical values, which is hard to read.
- One sentence is circular: "Combining the streams raised macro-F1 … by +0.044 … and one CNN on all features already captured this gain". Early fusion *is* one CNN on all features, so this says the model captured its own gain. Rewrite it as, for example: "Combining the streams in one CNN raised macro-F1 over the better single stream by +0.044 and +0.060, but an RF gained similar amounts (+0.032, +0.061) from the same features."

**n3. Highlights.** There are 5 highlights, of 72, 73, 68, 73 and 76 characters, so all are within the 85-character limit.

- Highlight 2, "Late fusion added at most +0.016 macro-F1", quotes the non-significant UNSW-NB15 point estimate as an upper bound; the CI goes to +0.031. Suggested replacement (≤ 85 characters): "Late fusion added +0.005 macro-F1 on TON-IoT and no significant gain on UNSW-NB15".
- Highlight 3 should say "in macro-F1".

**n4. The §5.2 attribution of the late-fusion advantage is imprecise.** The text says: "On TON-IoT the advantage therefore comes from the late-fusion architecture, two encoders and a wider head". By Table 3, early fusion with the head already recovers +0.004 of the +0.005. That leaves late minus early-with-head ≈ +0.002, which is the most the second encoder can contribute.

- **Fix:** Say that the head accounts for most of the advantage, and report late minus early-with-head directly.

**n5. §5.3: "Late fusion on random splits also stays above early fusion on TON-IoT (by +0.004 on average, all ten splits)".** Early fusion was not trained on random splits; it was trained on random column orders.

- **Fix:** State what the comparator is: the seed-matched semantic early-fusion run, or the order runs.

**n6. The data-split replication reuses the original feature selection.** §4.2: "The selected features are kept from the original split". Selection used labels on a training split that now partly overlaps the new test sets.

- **Fix:** State this as a limitation. Better, refit RFECV per split; this needs CPU only.

**n7. The random-split comparisons mix training protocols.** §4.2 says 104/120 random-split runs come from the 60-epoch protocol, which restored the best weights only when early stopping fired. The semantic runs use the 120-epoch protocol. The argument that they are equivalent is reasonable, but the comparison relies on it.

- **Fix:** Report the within-seed difference using only the 16 re-run splits, or rerun the 104 when the GPU is free, or at least state the best-epoch range of the reused runs.

**n8. The UNSW-NB15 CNN results depend on column order.** Early fusion on five random column orders spans 0.527–0.565 (SD 0.015), which is larger than the seed SD and larger than late minus early (+0.016).

- **Fix:** Say explicitly that UNSW-NB15 CNN comparisons below about 0.02 are within order variation, and that the MLP results are the cleaner evidence there.

**n9. Effect of the tree/feature-selection coupling.** §6.3 notes that the RFECV selector uses an RF and so may favour the trees, but the effect is not measured.

- **Fix:** Use a cheap CPU-only check. Retrain the RF and GBDT on a selection made by mutual information alone, or on all 40 UNSW-NB15 features, and report whether the RF-minus-CNN margin keeps its sign. This also partly answers the round-1 request for a larger UNSW-NB15 feature set.

**n10. No compact tree was tested for IoT deployment.** §5.6 gives RF sizes of 211 MB and 607 MB and says "smaller forests trade some accuracy for size, which we do not explore here". For this journal, the RF-beats-CNN finding needs a size-comparable point.

- **Fix:** Add a depth-limited RF or a small GBDT (for example 50 trees, depth 10, and a size near 1 MB) with its macro-F1. This needs CPU only, a few minutes.
- Also add the size column that Table 8's caption promises.

**n11. Formatting and statistics notation.**

- Report "p < 0.001", not "p = 0.000" (Table 3, §5.2).
- §5.4 "(lowest -0.17)" needs a minus sign.
- Table 6 has "(3/5, 1 ties)"; write "1 tie".
- Give the eight Holm-adjusted p-values in one small table, including attention versus early.

**n12. Fig. 3 label overlaps and truncations remain** (round-1 minor 8).

- **Fix:** Use adjustText or abbreviations, lengthen the truncated labels, keep "gafgyt junk" inside the axes, and note the N-BaIoT x-range in the caption.

**n13. Fig. 1 does not match the text.** It omits the attention head's 64-unit dense layer and the dropout layers, and draws the single-stream and early-fusion models identically apart from the inputs.

- **Fix:** Add the dense-64 layer to the attention branch, so that the parameter difference from late fusion (73,036 against 85,770) is visible.

**n14. Companion studies.** §3.1 and §3.1-N-BaIoT rely on "the corrected pre-processing pipeline of our companion studies" and "the preparation of our federated-learning study", but neither is cited.

- **Fix:** Cite them, or say that they are under review and that the pipeline is in the released repository.

**n15. Reference check.** Sanjalawe et al. is dated 2025 in vol. 5 of *Internet of Things and Cyber-Physical Systems*, but its DOI is 10.1016/j.iotcps.**2026**.03.002.

- **Fix:** Verify the year, volume and pages.

**n16. Confirmatory family.** §4.3 says "The confirmatory family was fixed before the experiments reported here." Two of its four tests (late versus wide, late versus late-on-all) were added in response to the round-1 review, after the main results had been seen.

- **Fix:** Say so. This is not a problem, but a referee will ask.

**n17. §2.1 "none … none" sentence.** Restrict the "none" counts to the eight full-text papers ("none of the eight read in full …"), and give the abstract-only count separately. Add a multiple-seeds column to Table A.1, as requested in round 1.

---

## Assessment

**Novelty.** The novelty is methodological and modest but real. No individual control is new; random subspaces, capacity matching and tree baselines are all standard. Their combination applied to fusion-IDS, the demonstration that branch-versus-fused gains are feature-count effects, and the documented silent failure give the field a useful checklist. The paper is citable as a negative-results/methods paper.

**Claims versus evidence.** Mostly scoped now. The exceptions are the title, §6.1 and the UNSW-NB15 "no effect" wording (N1, N2).

**Statistics.** The design is now sound: paired within-seed tests, Welch for trees against networks, an explicit 8-test Holm family with N-BaIoT excluded, and split-level replicates kept descriptive. The weaknesses are absence-of-evidence reading, not stating the per-dataset power, and incomplete reporting of the family (N2, n11).

**Clarity.** The paper is clear and economical. The abstract is too dense and has one circular sentence (n2).

**Fit to the journal.** Adequate. It would be stronger with a second IoT dataset and a compact-tree deployment point (N1, n10).

**What would bring this to accept (about 8):**

- N1: a second IoT dataset, or a rescoped title and §6.1.
- N2: equivalence-style reporting.
- N3–N4: fix Table 7 and report the promised metrics.
- N5: a small CNN tuning check, and the FPR explanation.
- n1: updated caption fields.
