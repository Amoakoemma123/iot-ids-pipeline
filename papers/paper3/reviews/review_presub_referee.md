# Pre-submission referee report (JNCA): Paper 3, class-incremental learning for a compact IoT IDS

Reviewed: `Manuscript_filled.md` (2026-11-09 21:53), `values.json`, `build.py` (highlights and captions), `run_cl_v2.py`, `analyze_cl_v2.py`, raw JSONs in `results_v3` and `results_p3final`. I formed this view from the manuscript first and only then read the three `review_final_*.md` files.

## Score and recommendation

**6/10, major revision** as I would rate it at JNCA. This is a borderline case close to minor revision.

What holds up:
- The protocol is careful and the numbers are clean. Every number I checked reproduces, from values.json and from the raw JSONs.
- The A-GEM fix and the three new groups are implemented as the text describes.

Why it is still major:
- The wording around the new results overstates them in several places.
- The new random-forest-on-memory result undermines the paper's own framing, and the text does not deal with that. A CNN "for compact detectors" that loses to an RF on the same 2,000 records, and on two datasets even loses to that RF when the CNN is jointly retrained, needs a direct discussion.
- At JNCA this would draw a major-revision request on framing, but not on soundness.

## Verification performed

**Checked against values.json (all match the text):**
- Table 2, all A-GEM rows.
- Gap-closed percentages for ER-500, ER-2000, LwF and A-GEM. I recomputed them from the Table 2 means: 91/85/81, 88/62/74, 11/16, 31/9/57.
- Random-order gap closed: 91/64/84.
- RF−joint gaps: 0.042/0.054/0.013.
- ER-2000b: −0.057, 57% of the excess.
- Storage ratios: 64/37/55.
- ER-2000-pool per-task retention and shortfall verdict.
- RF+memory figures.
- Range-share values.
- Holm p-values.

**Recomputed from the raw JSONs with my own script (confusion matrices and `macro_f1_seen`, paired scipy t-tests, my own Holm):**

*Final macro-F1, five seeds, TON / UNSW / N-BaIoT*

| Run | TON | UNSW | N-BaIoT |
|---|---|---|---|
| ER-2000 | 0.872 | 0.461 | 0.851 |
| ER-2000-pool | 0.883 | 0.497 | 0.852 |
| RF+memory | 0.939 | 0.473 | 0.991 |
| A-GEM (agem2) | 0.531 | 0.167 | 0.670 |
| Fine-tuning | 0.358 | 0.130 | 0.253 |
| Joint retraining | 0.922 | 0.520 | 0.987 |

*Benign FPR*
- ER-2000: 0.132 / 0.205 / 0.002.
- ER-2000-pool: 0.030 / 0.201 / 0.002.
- RF+memory: 0.005 / 0.015 / 0.000.

*Per-task retention and old-task detection*
- ER-2000-pool per-task R: TON [0.900, 0.932, 0.946]; UNSW [0.671, 0.611, 0.679]; N-BaIoT [0.948, 0.755, 0.829].
- Fine-tuning and A-GEM old-task detection reproduce §5.3: 0.875/0.698/0.875 and 0.892/0.729/0.995.

*Paired tests*

| Comparison | TON | UNSW | N-BaIoT |
|---|---|---|---|
| Pool − ER, macro-F1 (d, p) | +0.011, p = 0.048 | +0.037, p = 0.0005 | +0.001, p = 0.94 |
| Pool − ER, FPR (d, p) | −0.101, p = 0.0011 | not checked | not checked |
| RF+memory − ER (d, p) | +0.067, p = 0.0001 | +0.013, p = 0.023 | +0.141, p = 0.0009 |
| RF+memory − pool (d, p) | +0.056, p < 0.001 | **−0.024, p = 0.0007** | +0.139, p < 0.001 |
| A-GEM vs fine-tuning, Holm-adjusted p | 0.003 | 0.19 | 0.024 |
| Online EWC vs fine-tuning, Holm-adjusted p | 0.056 | 0.19 | 0.053 |

All Holm values match the text.

**Code checks in `run_cl_v2.py`:**
- **agem2.** The reference loss uses `class_weights(buf.y)` and `training=False`. The memory is 2,000 records, and the reference batch is 512 records drawn with replacement. Matches §4.2.
- **er2000p.** The memory is concatenated once into `Xfit`, with no per-step memory draw. The class weights are computed over task + memory, as in ER. Matches §4.3.
- **rf_er2000.** It uses `Buffer(2000, default_rng(seed+1))` with the same `add` sequence, so the memory is identical to that of the CNN ER-2000 run. Matches §4.3.

One mismatch in wording: §4.3 says "every seen class receives the same total weight per epoch". That holds only below the weight cap of 50 (`WMAX`); see minor 6.

## Major issues

1. **The claim that "RF on the same memory beat the CNN with replay on all three" contradicts the authors' own ablation.**
   - **Where:**
     - Abstract: "A random forest retrained on the same memory was better than the CNN with replay on TON-IoT, UNSW-NB15 and N-BaIoT."
     - §6.1: "beat the CNN with replay on all three datasets, so where a tree model fits the deployment it is the simpler and stronger choice."
     - Highlight 2.
   - **What is wrong:**
     - The claim holds only against ER-2000.
     - ER-2000-pool is also CNN replay, and the paper promotes it. It beats RF+memory on UNSW-NB15: 0.497 against 0.473, d = −0.024, p = 0.0007 (my test).
     - §5.2 says only "numerically ahead … (not tested)", although the test is trivial and clear.
   - **Fix:**
     - Say "better than the CNN with ER-2000 on all three datasets, and better than ER-2000-pool on TON-IoT and N-BaIoT but not on UNSW-NB15", and report the test.
     - Reword Highlight 2. For example: "A random forest on the same replay memory beat CNN replay on two of three datasets".

2. **"Better" and "stronger" rest on macro-F1 alone; on the paper's own alert-level view, RF+memory is worse on UNSW-NB15.**
   - **Where:** §5.2, "A tree model on the same memory"; §6.1, "simpler and stronger".
   - **What is wrong:**
     - On UNSW-NB15, RF+memory flags only 0.702 of earlier-task attack records (values.json `P3F_UNSW_rfmem_old_det`). Task 1 alone is 0.499.
     - Every CNN replay variant flags at least 0.990, and ER-2000-pool flags 0.999.
     - Its earlier-class retention is also lower than ER-2000's on TON-IoT (0.933 against 0.956) and UNSW-NB15 (0.526 against 0.702).
     - The forest buys its low FPR (0.015) by missing old attacks. The paper's central recommendation is to read class retention, detection retention and FPR together, but that recommendation is not applied to this baseline.
   - **Fix:** Report RF+memory old-task detection retention in §5.2, and qualify "stronger" to "higher macro-F1 and lower FPR; on UNSW-NB15 it misses half of the Task-1 attacks".

3. **The RF+memory result is not confronted in the framing, and "compact" is not substantiated against the RF.**
   - **Where:** Title; §1 contributions; §6.1, "Joint retraining remained the best option"; Conclusion.
   - **What is wrong:**
     - RF+memory (2,000 stored records) exceeds CNN *joint retraining* on TON-IoT (0.939 against 0.922) and N-BaIoT (0.991 against 0.987).
     - So "Joint retraining remained the best option" is true only within the CNN family.
     - The paper calls the RF "simpler", yet reports no RF model size or inference cost. A 200-tree, depth-20 forest is probably orders of magnitude larger than the 24k-parameter CNN (under 100 KiB). That is the one axis on which the "compact" CNN could win.
   - **Fix:**
     - Change §6.1 to "Joint retraining remained the best option for the network".
     - Report the RF size (nodes, or serialised KiB) and per-record prediction time next to the CNN's 24k parameters.
     - State plainly in the Conclusion what the CNN offers over the RF, or narrow the claims.

4. **The attribution "Most of the false-positive cost therefore comes from the double weighting … under-represented benign traffic explains a smaller part" is not supported.**
   - **Where:** §5.3, final paragraph.
   - **What is wrong:**
     - The two interventions were run separately, each against ER-2000. Pool alone gave 0.030, which removes about 100% of the excess. ER-2000b alone gave 0.075, which removes 57%. Together they "explain" 157%, so they are not additive, and there is no pool + benign-half cell.
     - Both act on the same quantity, the effective loss weight of benign relative to old attack classes. Oversampling class-balanced memory under-weights benign precisely because benign holds only 1/k of the memory. The two "causes" are therefore not separable as worded.
     - Pool also changes the replay ratio and memory exposure: each memory record is seen once per epoch, against about |task|/2,000 times under 1:1 mixing (≈19× on TON-IoT Task 1). It is not purely a weighting change.
   - **Fix:** "Removing the per-minibatch oversampling alone brought the TON-IoT FPR to the joint level (0.030 against 0.032). A half-benign memory under the original scheme removed 57% of the excess. The two act on the same effective benign weight and were not combined, so we do not apportion the cost between them."

5. **The Conclusion is stronger than the Results on the weighting effect.**
   - **Where:** §7, paragraph 2, "Both depend largely on the double weighting …", against §5.2, "the location of the shortfall depends partly on the double weighting".
   - **What is wrong:**
     - "Largely" and "partly" disagree.
     - The *gap* to joint retraining hardly moved: macro-F1 changed by +0.001 on N-BaIoT (CI [−0.048, +0.050]) and by +0.011 on TON-IoT (uncorrected p = 0.048). Only its *location* moved. On N-BaIoT, Task 1 fell to 0.755 against 0.997 for joint retraining.
   - **Fix:** Use "partly" in both places. Add that pool changed final macro-F1 only on UNSW-NB15 (+0.037), so the stability–plasticity balance moved but the overall gap did not close.

6. **The highlight and abstract describe the pool ablation wrongly.**
   - **Where:**
     - Abstract: "replaying the memory once per task instead of per minibatch".
     - Highlight 4: "Replaying memory once per task, not per batch, cut TON-IoT false alarms 0.132→0.030".
   - **What is wrong:** The code adds the memory to the training pool, so each record is used once *per epoch* over up to 50 epochs. "Once per task" reads as a single pass.
   - **Fix:**
     - Abstract: "adding the memory once to each task's training pool instead of to every minibatch".
     - Highlight 4: "Adding memory to the training pool, not every batch, cut TON-IoT false alarms 0.132→0.030". This is 79 characters.

7. **Some text still reflects the conclusions from before the pool and A-GEM results.**
   - §4.2: "ER and A-GEM were not tuned (memory sampling ratio and learning rate fixed), which slightly favours the tuned strategies." The pool ablation shows that the sampling ratio moves the TON FPR fourfold and UNSW macro-F1 by +0.037. Change "slightly" to "can matter; Section 5.2".
   - §6.1, FPR bullet: "replay on TON-IoT raised the benign FPR to 0.116–0.132, partly because the class-balanced memory holds few benign records (a half-benign memory lowered it to 0.075)". This omits the pool result, which the same paper calls the main cause. The next sentence, "above 0.1 for replay on TON-IoT would be unusable", should note that ER-2000-pool reaches 0.030.
   - §6.2: "Because plain replay already recovers most of the gap, they would mainly target the newest-task shortfall". The pool ablation shows that this shortfall is largely a weighting setting. Reword.
   - §5.2: "Replay was the only family that substantially reduced forgetting". A-GEM cut class-level FM from 0.956 to 0.402 on N-BaIoT and from 0.970 to 0.694 on TON-IoT, using the same memory. Say "Strategies that use the memory were the only ones …", or "replay reduced forgetting most".
   - §5.2: "the memory recovered a sizeable part of what replay recovers". This is true on N-BaIoT (57% against 81%) but not on UNSW-NB15 (9% against 85%). On N-BaIoT the seeds range from 0.319 to 0.809 (SD 0.204). Say "on TON-IoT and N-BaIoT, with large seed variance on N-BaIoT".

8. **JNCA systems fit (M8 of the earlier round) is still only partly met.** The paper now gives parameters, storage and median update times. It still gives no inference latency, peak memory or edge measurement, and none at all for the RF, the strongest method. Given major issue 3, this is now the main JNCA-fit risk. **Fix:** add a small table per strategy (parameters or model size, update time, prediction time per 10k records, stored data), on one dedicated machine, including RF and RF+memory.

## Minor issues

1. **§5.2 RF paragraph, wrong cross-reference.** "The CNN with the memory added once to the training pool (ER-2000-pool, below)". ER-2000-pool is described *above* ("Is the shortfall a weighting effect?"). Change to "above".
2. **§5.2 RF paragraph, ambiguous pronoun.** "It retained 0.933, 0.526 and 0.991 of the earlier classes and ended at benign FPRs of 0.005, 0.015 and 0.000; one retraining sequence took 5, 7 and 16 s." After the interposed ER-2000-pool sentence, "It" reads as ER-2000-pool. Write "RF + memory retained …".
3. **§5.2 "Is the shortfall a weighting effect?", templated wording.** "…by more than half on TON-IoT, UNSW-NB15 and N-BaIoT; at least one earlier task now fell … on TON-IoT, UNSW-NB15 and N-BaIoT". This reads as a generated string. Use "on all three datasets" and give the worst earlier task: TON T0 0.900 against 0.967; UNSW T1 0.611 against 0.635; N-BaIoT T1 0.755 against 0.997.
4. **§5.2, missing cost of the pool.** Pool lowered earlier-task class retention (0.956→0.916, 0.702→0.641, 0.976→0.852). On TON-IoT it also lowered earlier-task detection retention, to 0.980. Mention this in §5.3, next to the FPR gain.
5. **§5.3 UNSW bullet.** A-GEM is given at Task 1 (0.184) but not at the end (0.048), where it falls back like fine-tuning because it forgets Task 1 (R = 0.000). Add it to the "falls back" sentence.
6. **§4.3, replay-weighting check.** "every seen class receives the same total weight per epoch". This holds only below the weight cap of 50; rare UNSW classes can hit the cap. Add "(up to the weight cap)". Also say that pool changes the replay exposure (major issue 4).
7. **§4.2 ER.** "The replay-weighting check of Section 4.3 removes this double weighting" should read "removes the oversampling and keeps the loss weights".
8. **§1 contributions.** The pool finding is now in the abstract, the highlights and the Conclusion but not in the contribution list. Add it to contribution 4, or as a fifth contribution, phrased as a design sensitivity.
9. **§5.5 range share.** "0.00% of UNSW-NB15 test values". The raw value is 0.0037%, next to 0.1% of records. Write "<0.01%" or "0.004%". Also report the share for later-task training data, which is what matters for leakage (values.json: 0.02%, 0.00%, 0.86%).
10. **Highlight 1.** "64–91%" merges the semantic order (81–91) with the random orders (64–91). Write "64–91% of the gap to full retraining across task orders", or quote 81–91% for the semantic order.
11. **§5.2, ER-2000-pool reporting.** The pool's final macro-F1 on TON-IoT rests on uncorrected p = 0.048. Label it "marginal".
12. **§5.2, memory 500 → 2,000 sentence.** Fine as written, but "Raising … improved" still reads as a finding on all three datasets. Lead with "clearly only on UNSW-NB15".
13. **Line 1 of `Manuscript_filled.md`.** It still carries the draft header. It does not reach the docx (build.py skips it), but remove it from the files you distribute.
14. **Abstract length.** 247 words against the 250 limit, so any added qualifier needs a cut elsewhere. Candidate: the not-yet-learned sentence.

## Numbers or claims found WRONG

1. **"A random forest retrained on the same memory beat the CNN with replay on all three datasets"** (§6.1; also Highlight 2 and the abstract's "the CNN with replay").
   - **Status:** False for ER-2000-pool on UNSW-NB15.
   - **Correct:** RF+memory 0.473 against ER-2000-pool 0.497, paired d = −0.024, p = 0.0007.
   - **How checked:** `results_p3final/UNSW_NB15/{rfmem,replayabl}`, five paired seeds, scipy `ttest_rel`.
2. **"…on UNSW-NB15 it was numerically ahead of the forest (not tested)"** (§5.2).
   - **Status:** Misleading.
   - **Correct:** The difference is significant (same test as item 1).
3. **"Replaying the memory once per task"** (abstract, Highlight 4).
   - **Status:** Describes the method wrongly.
   - **Correct:** The memory is part of the training pool, once per epoch (`run_cl_v2.py` l.283–286, where `Xfit` is rebuilt per task and iterated every epoch).
4. **"Both depend largely on the double weighting"** (Conclusion).
   - **Status:** Not supported.
   - **Correct:** N-BaIoT final macro-F1 was unchanged (+0.001, p = 0.94), and §5.2 itself says "partly".
5. **"Most of the false-positive cost therefore comes from the double weighting …, and under-represented benign traffic explains a smaller part"** (§5.3).
   - **Status:** Not a valid decomposition.
   - **Why:** The two separate interventions remove 100% and 57% of the excess and were never combined (major issue 4).
6. **"0.00% of UNSW-NB15 test values"** (§5.5).
   - **Status:** Rounds a non-zero value to zero.
   - **Correct:** 0.0037% (`results_p3final/UNSW_NB15/rangeshare.json` `te.out01` = 3.7e-5).

Not wrong: every other number I checked, about 60 values in total, matches values.json, and the subset I recomputed matches the raw JSONs. This includes all Holm-adjusted p-values for A-GEM, EWC and LwF.

## Residual items from the earlier reviews

**Unaddressed or partly addressed:**
- **M8, systems evidence.** Partly met: parameters, storage and update times are given. Latency, peak memory, any edge device and any RF size or cost are still missing.
- **M7.** §5.6 still argues with "a first run of the present protocol". It is acceptable as recast, but the 0.805 against 0.768 non-reproduction remains, and the batch-normalisation mechanism is still unmeasured (it is disclaimed).
- **Minor 5.** Per-class test counts and binomial CIs for the rare UNSW-NB15 recalls (backdoor 0.01 against 0.65) are not given; there is only a pointer to Table 1.
- **Minor 8.** MLP check of the kernel-3 assumption: not done, only acknowledged.
- **Minor 10.** Threshold-free detection measure (AUROC, or recall at a fixed FPR): not done, only acknowledged. It matters more now that the RF+memory and pool comparisons trade FPR against detection.
- **Minor 17.** Benign-share sweep: not done, only acknowledged.
- **Minor 18.** Frozen release (tag or DOI): the data-availability statement still gives a bare GitHub URL.
- **M5, literature.** The 2025–2026 literature re-check: the newest citations are still from 2025.

**Addressed:** M1, the A-GEM fix (verified in code and results); M2, the weighting ablation (run, though over-interpreted, see majors 4 and 5); M3; M4, the RF baseline (run, though over-claimed, see majors 1–3); M6; the numbers review's W1–W4 and O1–O7; and C3–C15 from the code review.
