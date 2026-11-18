# Referee report: Journal of Network and Computer Applications

**Manuscript:** "Learning New Attack Families After Deployment: A Controlled Evaluation of Class-Incremental Learning for Compact IoT Intrusion Detectors"
**Material reviewed:** Manuscript_filled.md (full text, Tables 1–7, Figure 1), the figure file, values.json, and the released experiment script `experiments/paper3_continual/run_cl_v2.py`. The Data Availability statement says this code is public, so I read it to check how the methods described in the text were implemented. I did not re-run anything.
**Date:** 2026-11-07

---

## 1. Summary of the submission

The paper tests six class-incremental strategies on a small 1D-CNN intrusion detector. The strategies are fine-tuning, online EWC, LwF, A-GEM, and experience replay with 500 or 2,000 stored records. Joint retraining serves as the upper bound and a random forest as a reference. The data are TON-IoT, UNSW-NB15 and N-BaIoT, each split into three tasks. The protocol is careful:

- benign traffic appears in every task;
- pre-processing is fitted on Task 0 only;
- all strategies share a bit-identical Task-0 model;
- results cover five seeds on the semantic task order and one seed on each of four random orders;
- the main comparisons use paired t-tests with Holm correction.

The main findings are as follows:

- ER-2000 closes 81–91% of the gap between fine-tuning and joint retraining (64–91% over random orders).
- EWC and A-GEM are no better than fine-tuning, and LwF helps only a little.
- Replay's remaining shortfall is on the newest task.
- At the alert level, forgetting is mild: fine-tuning still flags 0.70–0.88 of old attack records.
- Replay raises the benign FPR on TON-IoT, and a memory that is half benign records reduces this.
- Fitting the pre-processing on all data (the "oracle" setting) changes macro-F1 by at most 0.029.
- A random-forest reference exposed a training failure in an earlier, unpublished version of the study (Section 5.6).

The writing is clear and unusually honest about its limitations. Every number is generated from result files. However, the main empirical claims are weaker than the text says, for three reasons:

1. **A-GEM implementation flaw.** The released A-GEM code gives zero weight to the old attack classes in the reference gradient. The negative A-GEM result, which the paper uses as its cleanest mechanistic argument, is therefore not a valid test of A-GEM.
2. **Replay loss design.** The replay loss applies class re-weighting on top of 1:1 memory mixing. This plausibly creates the "newest-task shortfall" that the paper presents as a finding about replay.
3. **Narrow scope.** The contribution consists of confirming known results (van de Ven et al., 2022; Rahman et al., 2022) plus a few evaluation views, on three short sequences. No systems-level evidence (model size, update cost, latency) backs the "compact / edge" framing, which matters for JNCA.

## 2. Score and recommendation

**Score: 5/10.**
**Recommendation: Major revision.**

The protocol and the reporting are above the average for continual-learning IDS papers. The A-GEM flaw and the replay-weighting confound affect two of the headline conclusions. Both must be fixed and re-run before the conclusions can stand. Novelty is modest, so after the fixes the paper's value rests on being a trustworthy, well-controlled benchmark. That makes the correctness issues decisive.

---

## 3. Major issues

### M1. The A-GEM reference gradient ignores old attack classes (implementation flaw)

**Location:** Section 4.2 ("A-GEM … gradient projection with a 2,000-record memory"); Section 5.2 (A-GEM bullet); Section 5.3 (A-GEM on UNSW-NB15); Abstract and Conclusion ("EWC and A-GEM were no better than fine-tuning").

**What the code does:** In `run_cl_v2.py`, class weights for A-GEM come from the current task's labels only:

```python
pool_y = yt if buf is None or buf.y is None or strategy == 'agem' else np.concatenate([yt, buf.y])
w = tf.constant(class_weights(pool_y, n_cls))
```

`class_weights` returns 0 for any class with no records in `pool_y`. The A-GEM reference loss then reuses the same weighted loss on the memory, `lref, _ = ce(xm, ym)`, with `ce = sum(l * w[y]) / sum(w[y])`. Every old attack class has weight 0 in this loss. Only benign records, which are present in every task, contribute. The projection therefore protects the loss on benign memory records and leaves the loss on old attack classes unconstrained.

**Why it matters:**
- Section 5.2 interprets the result mechanistically: "Used as training data, the memory prevents forgetting. Used only to constrain the gradient direction, it does not." That contrast is not supported if the constraint never saw the old attack classes.
- The pattern in the results fits the flaw. A-GEM has the lowest final benign FPR on TON-IoT (0.005) and UNSW-NB15 (0.017). On UNSW-NB15 it "moves old attacks into the benign class" (Section 5.3), with detection-level retention of 0.604 in Table 2. This is what a benign-only constraint would produce.
- The paper's central qualitative claim, that gradient projection fails just as regularisation does, partly rests on this result.

**What would fix it:**
- Compute the reference loss on the memory unweighted, or with weights computed over the memory's own label distribution.
- Re-run A-GEM on all three datasets, on both the semantic order (five seeds) and the four random orders.
- Revise Sections 5.2, 5.3 and 6.1, the Abstract and the Conclusion to match. A-GEM may still fail in class-incremental learning, as van de Ven et al. (2022) report, but the paper must show it with a correct implementation.

### M2. The replay loss re-weights old classes twice, which confounds the "newest-task shortfall" finding

**Location:**
- Section 4.1: "class-weighted cross-entropy on the current training pool (weights inversely proportional to class frequency, capped at 50)".
- Section 4.2: the memory "is added to each new task's training data".
- Section 5.2 and Table 3: "The remaining cost of replay therefore lies in learning the newest task, not in forgetting".
- This claim is repeated in the Abstract and the Conclusion.

**What the code does:** The text says the memory is concatenated to the new task's data. In the code, each minibatch of 512 current records is paired with 512 records drawn with replacement from the memory, so the memory supplies half of every gradient step. On top of that, class weights are computed from the pooled counts (current task plus memory). Each memory class has only 2,000 / (number of seen classes) records, about 200, so its weight is very high, up to the cap of 50. The new, large attack classes get low weights. Old classes are therefore up-weighted twice, once by oversampling and once by the loss weights. This pushes the model strongly towards stability.

**Why it matters:**
- The finding that replay "keeps the earlier tasks at the joint level but under-learns the newest task" is the opposite of the usual recency bias in class-incremental replay (cf. the bias-correction literature that the authors themselves cite in Section 6.2). Given the loss design, it may be an artefact of this weighting rather than a property of replay.
- Plasticity values in values.json support this reading. ER-2000 reaches 0.541 on UNSW-NB15 against 0.835 for fine-tuning and 0.774 for joint retraining, and 0.823 on N-BaIoT against 0.975 and 0.991.
- The budget check in Section 5.2 also fits it. Longer training raised newest-task retention but lowered retention of the earlier tasks, which is the behaviour expected when the trade-off is set by the loss.

**What would fix it:**
1. Describe the replay mixing accurately in Section 4.2: 1:1 per batch, sampled with replacement.
2. Run at least two variants of ER-2000: (a) no class weights, or weights computed from the sampling distribution actually used; and (b) the memory concatenated to the data, as the text currently says.
3. Report the newest-task and earlier-task retention for each variant.
4. If the shortfall shrinks, reframe Section 5.2 and the Abstract claim that "its remaining shortfall lies in the newest family, not in forgetting".

### M3. "No better than fine-tuning" is read from non-significant tests with n = 5

**Location:**
- Section 5.2: "Online EWC did not differ significantly from fine-tuning on any dataset (differences −0.003, +0.042, +0.054; Holm-adjusted p 0.11, 0.19, 0.08)".
- The Abstract and the Conclusion state this more strongly: "EWC and A-GEM were no better than fine-tuning".
- Section 4.5 defines the statistics.

**Why it matters:**
- A paired t-test with five seeds has little power. Gains of +0.042 and +0.054 macro-F1 on UNSW-NB15 and N-BaIoT are not negligible, and the EWC standard deviation on UNSW-NB15 is 0.043. "Not significant" is not evidence of equivalence.
- The five seeds vary only network initialisation and memory sampling. The released code fixes the data split (`seed=42`), the benign shares ("fixed per order, independent of the training seed") and the feature selection. The reported uncertainty therefore covers training randomness only, not data or split variability. Section 4.5 does not say so.

**What would fix it:**
- Report paired differences with 95% CIs for every confirmatory comparison, as Table 6 already does for the oracle ablation.
- Reword the claim to something like "small, non-significant gains (+0.04 to +0.05) on two datasets, far below replay".
- State in Section 4.5 what the seed controls, and that the split is fixed.
- If feasible, vary the split as well, or add seeds for the EWC and LwF comparisons. Also state the size of the Holm family (six comparisons per dataset?).

### M4. The main practical alternative, retraining a tree model, is missing, yet the deployment advice rests on the CNN

**Location:** Section 4.3 (random forest only at the final stage); Section 6.1 ("Keep a small replay memory", "Joint retraining remained the best option", "Use a non-neural reference"); Section 6.2 ("An incremental tree ensemble … was not evaluated").

**Why it matters:**
- The random forest trained on all data beats every network configuration on every dataset (0.965 / 0.575 / 1.000 against 0.922 / 0.520 / 0.987 for joint CNN retraining).
- Section 6.1 states that retraining takes "five to ten minutes … on a laptop" and that "the case for continual learning … rests on data retention limits more than on compute."
- Under that framing, a reader at JNCA will ask how a random forest retrained on the replay memory plus the new task's data compares with CNN + ER-2000. That baseline uses the same memory, costs seconds, and decides whether the paper's recommendation (a CNN with replay) is the right engineering choice.

**What would fix it:**
- Add "RF + ER-2000", that is, a random forest retrained after each task on the memory plus the current task, under the same protocol. If possible, also report the random forest after every stage, not only the last.
- If the random forest with memory matches or beats CNN + replay, the deployment advice should say so.

### M5. Novelty relative to existing work is limited and should be stated more modestly

**Location:** Section 1 (contributions 1–4); Section 2.2, last paragraph.

**Why it matters:**
- "Regularisation fails and replay works in class-incremental learning" is already established in general continual learning (van de Ven et al., 2022, cited) and in security (Rahman et al., 2022, cited).
- For network intrusion detection, Talpini et al. (2024) already use class-incremental tasks with benign traffic in every task, a full-data bound and 15 task orders, which the authors acknowledge ("come closest to our protocol").
- The remaining contributions are evaluation views:
  - **Detection-level retention.** This is binary attack-versus-benign recall on old classes.
  - **Zero-day flagging.** This is attack-versus-benign recall on classes not yet trained.
  - **Oracle pre-processing ablation.** Its effect is null on two datasets and inconclusive on the third, and it is partly a property of the authors' compression map and clipping (see M6).
  - **Contribution 4 (a random-forest reference).** This is good practice, not a research contribution.
- Several of the strongest statements in the paper concern the authors' own unpublished earlier version, which a journal reader cannot inspect (see M7).

**What would fix it:**
- Position the paper explicitly as a controlled replication and extension of Talpini et al. (2024) and van de Ven et al. (2022) for IoT intrusion detection.
- Claim the detection-level and not-yet-learned views as recommended reporting practice rather than as new findings.
- Check the 2025–2026 continual-learning intrusion-detection literature again; the most recent works cited are 2025 conference and journal papers.
- Corrected results for M1 and M2, together with M4, would strengthen the case considerably.

### M6. The pre-processing description does not match the implementation, and the "oracle" ablation is pipeline-specific

**Location:** Section 3.3; Section 5.5, "Pre-processing scope"; Table 6.

**Why it matters, point by point:**

1. **N-BaIoT has no feature selection.** Section 3.3 says feature selection and min–max scaling are fitted on Task 0. In the released code, N-BaIoT (`D['raw']` false) gets min–max scaling only, and values.json lists `NB_n_features = 115`, all features. The oracle ablation on N-BaIoT therefore changes only the min–max range. This should be stated, because it changes what "inconclusive on N-BaIoT" means.

2. **The clipping window is not [−1, 2] in min–max units.** The text says "Later-task values outside the Task-0 range are clipped to [−1, 2]" and, in Section 5.5, "Only 0.002% of TON-IoT test values fall outside the Task-0 range". In the code, the clip to [−1, 2] is applied after the asinh compression. On the min–max scale, the upper limit of 2 corresponds to about 2,000 times the Task-0 range. The reported 0.002% is therefore the share of values beyond that extreme limit, not the share outside [0, 1]. The share of later-task values outside the Task-0 range, which is the relevant quantity for the leakage argument, is not reported and may be much larger.

3. **The null result is a property of this pipeline.** The asinh map was introduced precisely to extrapolate smoothly. The authors partly say this ("under our pipeline"), but the Abstract states the result without that qualifier. The number of selected features (13 for TON-IoT and 39 for UNSW-NB15, according to values.json) and their overlap between the Task-0 and all-data fits are not reported, so the reader cannot judge how much the ablation actually changed.

4. **One result is unexplained.** Oracle pre-processing *lowers* not-yet-learned flagging on TON-IoT and N-BaIoT (Table 6, about −0.04 to −0.06). If future-data leakage gave the first model knowledge of later tasks, the opposite would be expected. The text reports this without comment.

**What would fix it:**
- Correct the description of the clipping and report the true share of later-task values outside [0, 1] for each dataset.
- State that N-BaIoT uses all 115 features.
- Report the number of selected features and the overlap between the two fits.
- Discuss the sign of the not-yet-learned flagging differences.
- Qualify the Abstract sentence accordingly.

### M7. Too much of the paper argues with an unpublished earlier version

**Location:**
- Abstract, last sentence.
- Section 1, paragraph 3 and contribution 4.
- Section 5.2: "It contradicts the earlier version of this study, which presented EWC as a competitive storage-free alternative".
- Section 5.6 (whole section, Table 7).
- Section 7, last paragraph.

**Why it matters:**
- A journal paper should stand alone. Reviewers and readers cannot inspect the "earlier version". Statements about what it "presented" or "reported" are unverifiable and read like an internal changelog.
- The "silent failure" framing overstates the problem. Under the present training budget, the min–max-only network reaches 0.871 on TON-IoT (Table 7). That is under-training on badly scaled inputs, not a silent collapse.
- The earlier configuration is not reproduced: 0.805 against 0.768 (Section 5.6), and "the present code also computes the loss differently". The causal story in the Conclusion ("caused mainly by uncompressed heavy-tailed inputs") therefore rests on the present-code factorial rather than on the earlier failure itself.
- The suggested mechanism, batch normalisation with heavy-tailed min–max features (Section 1: "the batch-normalised CNN failed silently"), is never measured.

**What would fix it:**
- Remove references to the earlier version from the Abstract, the contributions and the Conclusion.
- Recast Section 5.6 as a methodological finding about input scaling for neural intrusion detectors on flow features, supported by the Table 7 factorial (ideally on more than TON-IoT and joint retraining).
- If the batch-normalisation mechanism is claimed, support it with a diagnostic, for example the distribution of batch-normalisation running statistics or activations with and without the compression map. Otherwise drop it.
- Move the history to the cover letter or a short footnote.

### M8. JNCA fit: no systems or deployment evidence for a "compact IoT detector"

**Location:** Title ("Compact IoT Intrusion Detectors"); Section 1 ("costly on edge infrastructure"); Section 6.1 ("five to ten minutes … on a laptop"); Section 6.2 ("Training times are indicative only").

**Why it matters:**
- JNCA readers expect the network-applications side to be quantified. The paper reports no parameter count, model size, per-update training time, peak memory, inference latency or energy for any strategy. The only figures are the replay memory sizes (102–898 kB) and a qualitative laptop timing.
- Since the authors argue that the case for continual learning rests on data-retention rather than compute, the paper should at least quantify both sides:
  - the stored data volume for joint retraining compared with the memory;
  - the update cost of each strategy compared with retraining.

**What would fix it:**
- Add a short table with, for each strategy: parameters, update time per task (median over seeds, on a dedicated machine), peak memory and stored data.
- Ideally include one measurement on a representative edge device, or remove "edge" and "compact" from the framing.

---

## 4. Minor issues

1. **Early stopping (Section 4.1).** The phrase "validation records available at that stage" is ambiguous. In the code, incremental strategies stop on the current task's validation records only ("buffer-free"), and joint retraining stops on all tasks' validation records. Say so explicitly; it affects how replay balances the old and new tasks.
2. **Online EWC.** The code sums the Fisher matrices over tasks with no decay, which is online EWC with γ = 1. State γ, and state the grid steps for λ (10⁰–10⁷) and for the LwF weight.
3. **Tuning on all tasks' validation data (Section 4.2).** EWC and LwF are tuned with validation data from all tasks after the last one. The authors acknowledge that this is optimistic (Section 6.2). Note also that ER and A-GEM were not tuned at all (memory sampling ratio, learning rate), which slightly tilts the comparison in the other direction.
4. **Table 2.**
   - FM (detection) takes negative values (for example ER-2000 on TON-IoT: −0.006). Under the definition "max over t < T of R[t, j] − R[T, j]", this means the final value exceeds the earlier maximum. Add a sentence explaining it.
   - BWT duplicates −FM for almost every row (Section 4.4 says so), so consider dropping the column and keeping plasticity, which is currently not reported in any table even though it is central to M2.
   - The random-forest rows show "±" for macro-F1 but leave every other cell blank. Add at least the random forest's benign FPR (it is quoted in the text: 0.005 and 0.081) and its detection-level A.
5. **Per-class test counts.** UNSW-NB15 Task 2 has only 568 attack test records over four classes (Table 1). Recall for backdoor (0.01 against 0.65) and especially for worms is therefore estimated from very few records. Give per-class test counts in an appendix and add binomial CIs for the per-class recalls quoted in Section 5.2.
6. **Data preparation (Section 3.1).**
   - Specify which TON-IoT and UNSW-NB15 files were used (for example the TON-IoT train/test network subset and the UNSW-NB15 partitioned or full CSVs).
   - Explain why the two BASHLITE flood classes were dropped from N-BaIoT.
   - Explain why a 3% sample was used.
   - The random record-level split mixes devices and time windows. The N-BaIoT features are sliding-window statistics, so near-duplicates across the split are likely even after exact de-duplication. Acknowledge this.
7. **Number of selected features.** Report it (13 for TON-IoT, 39 for UNSW-NB15 and 115 for N-BaIoT according to values.json). Readers otherwise cannot interpret the memory sizes in Section 5.2 or the 1D-CNN input length.
8. **Model choice.** A 1D convolution with kernel 3 over a tabular feature vector assumes that the feature order carries meaning. Justify this, or add an MLP of similar size as a check (Section 4.1).
9. **Terminology: "zero-day flagging".** The term is used in Section 4.4 and Table 6 but suits the benchmark poorly, since these are known attack families withheld from training. The paper uses "not-yet-learned" in Section 5.4; use it throughout. Table 4 also does not report the summary "mean over stages" defined in Section 4.4, but Table 6 does. Report it consistently.
10. **Detection-level retention depends on the operating point.** It is defined through the argmax over benign and attack classes. Section 5.3 rightly pairs it with the benign FPR, but a threshold-free measure would let strategies be compared at a fixed FPR, for example attack-versus-benign AUROC or recall at a fixed benign FPR of 1%. This would make the "forgetting the class, keeping the alert" message more robust.
11. **Operational relevance of the benign FPR.** A benign FPR of about 0.2 on UNSW-NB15 for every network, including joint retraining (against 0.081 for the random forest), and 0.13 for replay on TON-IoT, would be unusable in practice. Say so plainly in Section 6.1, and connect it to M4.
12. **Section 2.2.** "We checked four studies in full text" is unusual wording for a journal paper, because it implies that the other cited works were not read. Rephrase, for example "Four studies are closest to ours; their protocols differ as follows".
13. **Abstract.** It is dense, with more than 15 numbers. Cut it to the three or four key results and remove the sentence about the earlier version (M7).
14. **Figure 1.**
   - The detection-level row uses a 0–1 y-axis while every value lies between 0.55 and 1.0. Rescale it so the differences are visible.
   - ER-2000 and joint retraining overlap almost completely in the top row; consider markers or an inset.
   - The paper has only one figure. A protocol schematic (tasks, benign shares, the evaluation after each task) and a gap-closure plot across orders would help the JNCA readership.
15. **Holm correction.** Section 4.5 says Holm is applied "within each dataset". State the family (six strategy-versus-fine-tuning comparisons?), and say whether ER-2000b and the budget check are excluded. They are reported with uncorrected p-values.
16. **Section 5.2, LwF grid.** The values 0.226 and 0.214 against 0.266 are development-seed validation scores. Label them as such so they are not confused with test macro-F1.
17. **Section 5.3, ER-2000b.** "Removes 57% of replay's excess over joint retraining" comes from a single benign share (0.5). The limitation is acknowledged in Section 6.2, but a two- or three-point sweep would be cheap and would turn a single-point observation into a usable recommendation.
18. **Declarations.** The data-availability URL should point to a frozen release (tag or DOI, for example via Zenodo) so that the code reviewed matches the paper.

---

## 5. Things done well

- **Protocol hygiene:**
  - benign traffic in every task;
  - pre-processing fitted on Task 0;
  - test-set de-duplication against training records;
  - a bit-identical shared Task-0 model, verified;
  - the test set is never divided between tasks.

  Together these are better than most continual-learning intrusion-detection papers (Sections 3.2–3.3, 4.1).
- **Upper bound and external reference.** Both a joint-retraining bound and a random-forest reference are reported, and expressing replay as "% of gap closed" makes the results easy to interpret (Section 5.2).
- **Evaluation views.** Reporting class-level retention, detection-level retention and the benign FPR side by side (Section 5.3) is useful practice. The authors rightly warn that detection-level retention must be read together with the FPR.
- **Robustness checks.** Five seeds plus four random orders (Table 5), with the honest observation that absolute levels and the value of a larger memory depend on the order.
- **Clear, falsifiable follow-ups.** The budget check (Section 5.2), the benign-share memory (Section 5.3) and the input-map × budget factorial (Table 7) each target one specific question.
- **Candid limitations section (6.2).** It includes the admission that the compression map was fixed after the failure was observed, and that Table 7 is scored on the test set and is diagnostic rather than a selection.
- **Reproducibility.** Every number is generated from result files and the code is released. This made it possible to find the issues in M1, M2 and M6, which is to the authors' credit.
