# Referee report (independent, second final round)

**Manuscript:** "Fusing Feature Groups of One Flow Record: A Controlled Evaluation of Multi-Stream Intrusion Detection on IoT and Network Benchmarks"
**Journal:** Elsevier *Internet of Things*
**Date:** 2026-11-14
**Materials read:** Manuscript_filled.md (all of it, appendices included); HIGHLIGHTS in build.py; review_r1/r2/final_referee.md and response_final.md, for context. Code checked: run_p6_v2.py, run_p6_gb_rollback.py, baseline/gbdt_rollback.py, bench_latency.py, and the parts of compute_values.py that produce the Holm family, the Welch tests and the compact forests. Results checked: results_v4/, results_p6gb/, results_v4/latency.json. I recomputed the four confirmatory paired tests, the Holm adjustment and the semantic-minus-random within-seed differences from the JSON files, and they match the text. I also ran three new checks from the saved predictions and JSONs: a per-class breakdown of late minus early fusion, a test-record bootstrap for one seed, and a provenance check of the GBDT in the cost table. No raw data or models were loaded.

**Score:** 6.5 / 10
**Recommendation:** Major revision. This is close to the minor/major line. Two of the issues below are factual and can be fixed without training: M1 (the cost table describes a different GBDT from the one evaluated) and M2 (the one confirmed gain is concentrated in a single 156-record class). Two need some new runs, a few GPU-hours on the authors' own setup: M3 (a small symmetric tuning check) and M4 (re-running the reused random-split runs). M5 (a second non-saturated IoT dataset) is what an IoT journal will press on hardest. It can be handled by stronger rescoping, but a new dataset is the better fix.

---

## Summary

The paper asks whether splitting one flow record's features into two "semantic" streams, encoding each with its own compact 1D-CNN and fusing them (early, late or attention fusion) improves multi-class intrusion detection once the obvious confounds are controlled. Those confounds are more features, more parameters, the particular grouping, and the lack of a strong tabular baseline. The experiments use TON-IoT, UNSW-NB15 (an enterprise dataset reduced to 10 features) and N-BaIoT (saturated, a ceiling check). Each configuration has five seeds, with:

- ten random splits of the same sizes;
- three capacity controls;
- an MLP encoder and random column orders;
- five further data splits;
- RF and GBDT on all features and on each stream;
- an in-model ablation of the attention weights;
- an account of a scaling failure that produced the earlier version's headline result.

The main findings:

- The fused model's gain over its branches is the gain from using all features. An RF gets the same gain.
- Late fusion adds +0.005 macro-F1 on TON-IoT (Holm p = 0.006). Mostly this is reproduced by early fusion with the late-fusion head, or by late fusion with both encoders on all features.
- Random groupings serve the fused models about as well as the semantic one.
- The trees lead every network in macro-F1.
- Attention weights track routing, not stream usefulness.

The study is careful and unusually candid. Post-hoc decisions, reused runs, GPU non-determinism and the post-hoc margin are all disclosed. Every number I recomputed matched. My concerns are about what the evidence can carry:

- One cost comparison is internally inconsistent.
- The one confirmed fusion gain is driven by a single tiny class.
- Tuning is asymmetric.
- The central grouping comparison mixes old and new runs.
- The IoT evidence rests on one dataset.

The manuscript is also far longer and denser than its message needs.

**Novelty and fit.** The finding that grouping and fusing columns of one table adds little beyond one network on all columns will not surprise ML readers. Its value is as an audit of a growing practice in IoT-IDS papers, with a reusable checklist of controls (§6.2). That is a legitimate contribution for *Internet of Things*, but a modest one. It depends on the IoT side being convincing, and at present that side is one non-saturated dataset (M5) plus a footprint argument that M1 undermines.

---

## Major issues

### M1. The cost table (Table 8) and the footprint claims describe a different GBDT from the one whose accuracy is reported. The rolled-back GBDT is probably small enough to undo the abstract's trade-off.

**Location:** §5.6 ("The GBDT is smaller (23 and 35 MB)"; GBDT row of Table 8); Abstract ("the tuned forests occupy 221–636 MB against 343 kB for late fusion, and a 4.4 MB forest scored below late fusion"); Highlight 3; §6.1 (fourth conclusion); §7.

**What the code shows:**
- `bench_latency.py` (2026-11-03) fits the GBDT with `R.tuned(ds)` from `results_v4/<ds>/tune_trees.json`, without rollback.
- On TON-IoT that is learning rate 0.05 with 127 leaves. The tuning log shows this configuration fitted 160 iterations; the pickle is 22.1 MiB.
- The GBDT in Tables 2–3 comes from `results_p6gb`. It was re-tuned after rollback (learning rate 0.1, 127 leaves) and keeps only 46–70 iterations per seed (`best_iter` 70, 59, 64, 46, 60).
- So Table 8 times and sizes a model that was never scored.

**Why it matters:** Pickle size scales roughly linearly with the number of stored trees. At about 0.14 MiB per iteration, the evaluated TON-IoT GBDT should be roughly 8 MB. This is an estimate and needs re-measuring. Its single-record latency would also fall. That GBDT is the best model in the paper on TON-IoT (0.962, +0.009 over late fusion). The abstract's IoT argument, that the trees' lead "comes with a memory cost" and that "a 4.4 MB forest scored below late fusion", picks the RF as the representative tree. The GBDT shows the accuracy lead is available at single-digit megabytes. On UNSW-NB15, v4 and p6gb chose the same GBDT parameters, but the models still differ in rollback, so they should be re-benched too.

**Further measurement problems:**
- The GBDT was timed with all cores for single records. The 114.95 ms per record on UNSW-NB15 is very likely thread start-up overhead, not model cost. The text half-concedes this ("probably inflates").
- The sizes are Python pickles holding float64 node arrays. The network size is raw float32 weights (343 kB). The two are not comparable.

**Fix (no training needed):**
1. Re-run `bench_latency.py` on the rolled-back, re-tuned GBDT from `run_p6_gb_rollback`, with `OMP_NUM_THREADS=1` (or an equivalent setting) for single-record timing.
2. Report both kinds of model in comparable units: parameter or node counts plus serialised size, and preferably an ONNX or treelite export for the trees.
3. Add one compact GBDT point, for example 31 leaves at the rolled-back iteration count.
4. Rewrite the abstract sentence, Highlight 3, §6.1 and §7 to match. If an ~8 MB GBDT beats every network, say so: it is the practically relevant result for IoT gateways.
5. Add a check to `verify.py` that the benchmarked model's hyperparameters equal those of the scored model.

### M2. The one confirmed fusion gain (TON-IoT, late minus early, +0.005) comes mostly from one class with 156 test records, and the seed-only intervals leave out test-sampling uncertainty.

**Location:** §5.2; Table 3; Table B.1; Highlight 2; Abstract ("Late fusion added +0.005 on TON-IoT (significant after Holm correction)").

**What I computed:** I split the late-minus-early macro-F1 difference into per-class contributions (ΔF1_c / K, averaged over seeds, from the JSON per-class F1).
- TON-IoT: MITM contributes +0.0033 of the +0.0052, which is 63%. Every other class contributes at most +0.0005.
- UNSW-NB15: worms (19 test records) contributes +0.0072 of the +0.0161, which is 45%.
- A test-record bootstrap of seed 0 on TON-IoT gives late minus early = +0.0042 with a 95% interval of [+0.0018, +0.0067]. That half-width (≈0.0025) is about twice the seed-based half-width the paper reports (≈0.001).

**Why it matters:**
- The Holm-significant result is mostly a statement about MITM F1 (0.62 → 0.66 on 156 records).
- The t-intervals over five seeds on one fixed test set describe training noise, not how well the result generalises to new traffic.
- The six split-level replicates (§5.2, "Other data splits") do include test sampling, and they support the TON-IoT sign (6/6). They are the better inferential unit, but they appear only as a descriptive aside.

**Fix:**
1. Report the per-class decomposition in one sentence and one column of Table B.1.
2. Add a paired test-record bootstrap (or a seeds × records hierarchical bootstrap) for the confirmatory family, or make the six split-level replicates the confirmatory unit.
3. Reword the abstract and Highlight 2 so that "significant" is tied to what was tested. For example: "+0.005 on TON-IoT, concentrated in the rare MITM class".

### M3. The tree-versus-network comparison is still tuned against untuned. A small symmetric grid would settle it, and the authors' own response lists it as open.

**Location:** §4.2; §5.2 ("No CNN beats the tuned RF…"); Abstract ("Tuned tree ensembles beat every untuned fusion network"); Highlight 3; §6.3.

**Why it matters:**
- The qualifier "untuned" is now honest, but it leaves the paper's most quotable practical comparison unresolved on its only non-saturated IoT dataset. On TON-IoT the gap is +0.008 to +0.009, and §6.3 admits that tuning "could plausibly change the ranking".
- "Tuned" for the trees is itself a 4-point grid on one seed. The UNSW-NB15 GBDT stopped at the 1,000-iteration cap on 4 of 5 seeds, so it is budget-limited, not tuned.
- The networks select their epoch on validation loss under a SMOTE-balanced training set and an imbalanced validation set. On UNSW-NB15, early fusion's best epoch is 6 on two seeds and 15–34 on the others (results_v4), and its seed SD (0.011) is ten times late fusion's (0.001). Part of the UNSW-NB15 late-minus-early difference may come from unstable stopping of early fusion rather than from architecture.

**Fix:**
1. Give early and late fusion a comparable small grid on validation macro-F1, for example learning rate {1e-3, 3e-4} × dropout {0.1, 0.3} × width {1, 2}. With runs of about 5 minutes on this hardware (train_s in results_v4), this is roughly 2–3 GPU-hours for two datasets.
2. Raise the GBDT `max_iter` on UNSW-NB15.
3. Report whether stopping on validation macro-F1 instead of validation loss changes the early-fusion instability.

If the authors decline, the tree claim should leave the abstract and highlights and stay in §5.2 as descriptive.

### M4. The central grouping comparison compares new semantic-split runs with reused older random-split runs. Re-running them is cheap and would make it like-for-like.

**Location:** §4.2 ("Random feature splits", the reuse paragraph); §5.3; Table 6; §6.3.

**Why it matters:**
- All 30 late-fusion random-split runs, and 17 of the 20 attention-fusion random-split runs on TON-IoT and UNSW-NB15, come from the earlier run. Every semantic-split run is new.
- The authors measure same-seed rerun differences of up to 0.002 on TON-IoT and 0.018 on UNSW-NB15, mean 0.006. Those are of the order of the effects being read: +0.001 [0.000, +0.002] on TON-IoT and +0.009 [−0.001, +0.019] on UNSW-NB15.
- The reused runs saved no predictions. So the grouping question, which is the paper's main question, has no duplicate-free score, no PR-AUC and no bootstrap.
- About 300 words of §4.2 and several caveats in §5.3 and §6.3 exist only to explain this reuse.

**Fix:** Re-run the 40 late and attention random-split runs on TON-IoT and UNSW-NB15 under the current code, saving predictions. That is about 40 × 5 min ≈ 3.5 GPU-hours. Then:
- report the duplicate-free semantic-minus-random difference;
- delete the reuse paragraph and the related caveats;
- consider making the within-seed semantic-minus-random test a pre-declared secondary test, since late versus late-on-all is only a proxy for it.

### M5. For an IoT journal, the IoT evidence rests on one non-saturated dataset. A second one remains the most valuable addition.

**Location:** Title; Abstract; §1 (contribution 1); §3.1; §6.1–6.3.

**Why it matters:**
- UNSW-NB15 is not IoT. RFECV reduced it to 10 features, a 6 + 4 split in which a 4-feature "stream" barely tests grouping, and 16.3% of its test records match a training record on those features.
- N-BaIoT is saturated, and its windows leak across the split.
- So every claim of the form "for IoT IDS" rests on TON-IoT, whose confirmed fusion effect is largely one class (M2).
- The abstract now says this openly ("the only non-saturated IoT benchmark in the study"), but an editor will read that as an admission.
- Three of the 14 audited papers use Edge-IIoTset or CICIoT2023, so the gap is visible from the paper's own Table A.1.

**Fix:** Preferred: add CICIoT2023 or Edge-IIoTset (multi-class) for the core set only: stream CNNs, early fusion, late fusion, late fusion on all features, ten random splits, RF/GBDT, five seeds. The pipeline already handles new splits. If it is genuinely not feasible, reframe the paper around TON-IoT throughout, from the title through the conclusion, and present UNSW-NB15 as a non-IoT robustness check. The §6.2 recommendations are then supported by a case study, not a general result.

### M6. The paper is about 13,300 words and very dense. The message fits in about 8,500–9,000, and the density hides the findings.

**Location:** Throughout; worst in §4.2, §5.2 (capacity paragraph), §5.3 and §6.3.

**Why it matters:**
- Almost every sentence carries two or three numbers with intervals, seed counts and caveats. A typical *Internet of Things* reader will not get through §5.2–5.3.
- Many caveats appear three times (method, result, limitation). Examples are GPU non-determinism, the reused runs and the RF-based selector.
- The 248-word abstract has 20 numbers.

**Concrete cuts:**
- **§4.2 reuse paragraph (~300 words):** delete after M4. Otherwise move it to an appendix.
- **§5.2 capacity paragraph (~450 words):** keep the TON-IoT attribution in 3–4 sentences. Move the UNSW-NB15 run-instability and duplicate-free details to Table B.2's caption.
- **Table 6 (32 rows):** keep CNN late and attention fusion plus the two single-stream rows per dataset in the main text (about 12 rows). Move the tree, MLP and N-BaIoT rows to the appendix.
- **Table 3:** move the MLP rows and "late − attention" to the appendix.
- **Table B.3:** merge its PR-AUC and detection columns into Table 2, and drop the duplicate macro-F1/FPR columns.
- **§5.1:** delete the PR-AUC sentence that lists six values already in Table B.3.
- **§5.5 (~550 words):** cut to about 250. It is a known pitfall (heavy-tailed inputs with batch normalisation), and a branch scoring below a constant predictor is not really "silent". Keep the Table 7 TON-IoT rows and the recommendation. Contribution 4 should be worded more modestly.
- **§5.6:** shorten Table 8 to the five main CNNs, the MLP pair and the trees. Drop the per-1,000 columns for N-BaIoT.
- **§6.3:** about 450 words, half repeating §4. Keep it to the limitations not stated elsewhere.
- **§2.1:** the paragraph counting controls duplicates Table A.1. Shorten it to the counts.
- **Abstract:** aim for about 180 words with about 8 numbers. Lead with the three findings (features, not fusion; grouping does not matter; trees are the reference) and the one practical recommendation.

---

## Minor issues

1. **Highlights omit the paper's distinctive control.** None of the five highlights mentions random feature groupings, the result that most clearly separates this paper from earlier ones. Replace Highlight 4 or 5 with, for example, "Random feature groupings served fused models as well as semantic ones". Highlight 3 must change after M1.

2. **Label conflicts on UNSW-NB15 are speculation, but they are checkable.** §5.2 says "the label of the matching training record was not saved, so we cannot report the share of matches with a different label." The prepared splits and selected features exist, and `_finish()` already hashes the selected-feature vectors of the training split. Storing the labels alongside the hashes is a few lines of code and needs no training. Report the share. It decides whether part of the UNSW-NB15 error is irreducible, and that bears on reading the tree lead.

3. **The ±0.01 margin is not justified in IDS terms.** On TON-IoT a 0.01 change in macro-F1 is a 0.1 change in F1 for one class, for example MITM. That is hardly negligible for an operator. Tie the margin to something operational, such as the change in benign FPR expressed as false alarms per 10⁴ flows, or present results without the "negligible" label.

4. **The GBDT's internal early-stopping split is drawn from the SMOTE-oversampled training set** (`validation_fraction=0.1` on `Xfit`). So the "best internal-validation iteration" is picked partly on synthetic records. §4.2 should say so. Using the real validation split (`X_val` via a custom loop or `n_iter` selection) would be cleaner.

5. **The weights-versus-reliance correlation is close to tautological.** ρ = 0.94 between mean stream-1 weight and reliance in §5.4 is largely built in: the weight multiplies the embedding that is ablated, so a heavily weighted stream will cost more when replaced. Say this, and give the stream-advantage result (ρ = 0.03 on TON-IoT) as the substantive one. The permutation p-values come from 10 classes and should be called exploratory.

6. **The companion studies are unpublished and uncited.** The pre-processing pipeline ("our companion studies"), the N-BaIoT preparation ("our federated-learning study") and the compact forests ("a companion study … not yet published") all rely on unpublished work. Cite preprints or manuscripts under review where they exist (anonymised in the blind version), or describe the parts used in an appendix. Results from another study should not appear in the abstract (the 4.4 MB forest) unless they are reproduced in this paper's code. They are in fact read by compute_values.py, so say that they are re-scored here.

7. **Feature selection is fixed from the original split in the resplits.** Its labels leak into the new test sets. This is acknowledged, but if the split-level replicates become the confirmatory unit (M2), refit RFECV per split for at least TON-IoT.

8. **"Early fusion" for one CNN on all features invites confusion.** Readers expect "early fusion" to mean feature concatenation before stream-specific processing, which here is identical to one CNN. A footnote is enough. Calling it "single network (= early fusion)" in tables would help.

9. **The table header "Params (TON-IoT)" sits in a three-dataset table.** Add the UNSW-NB15 and N-BaIoT counts, or say in the caption that they differ.

10. **"4/5 seeds" for differences with intervals touching zero** (e.g. +0.001 [0.000, +0.003]): state the convention for an interval bound that rounds to 0.000, as Table B.1 does at full precision.

11. **The GBDT single-record latency** (19.10 and 114.95 ms) should either be measured single-threaded or dropped (see M1).

12. **Sanjalawe et al. (2025) has a DOI with "2026" in it.** That is correct per Crossref, but expect a copy-editor query. Add "(published 2025; DOI as registered)" in the response, not the paper.

13. **AI declaration.** The declaration that Claude was used "to write and edit the manuscript text" is compliant with Elsevier policy. Some editors will still look harder at a long, very dense manuscript with this declaration. The length cuts in M6 also help here.

14. **Figure 1 and §4.1** both describe the architectures in full. One of them can be shortened.

---

## Strengths

- **The controls are the right ones, and they are combined.** Random groupings of matched sizes, capacity-matched and head-matched single networks, late fusion on all features, an order-independent encoder, resplits and per-stream trees. I know of no fusion-IDS paper with this set.
- **The statistics are disclosed honestly.** A pre-declared Holm family, with the post-hoc additions labelled. Unadjusted and adjusted p-values side by side. A margin explicitly flagged as post hoc. Repeated warnings that non-significance is not equivalence. My recomputation of the family from the JSON results reproduced every reported difference and adjusted p-value (e.g. TON-IoT late − early +0.0052, p = 0.0007, Holm 0.006; UNSW-NB15 late − late-on-all +0.0108, Holm 0.076).
- **The code is clean and auditable.** The BestWeights callback fixes a real Keras restore bug. The GBDT rollback is verified against `staged_predict_proba`. Results are saved per seed with configurations. An automated number-checker ties the text to the files.
- **The scaling failure is reported with candour.** It reversed the earlier version's conclusions, and the paper says so. The "tree reference as a sanity check" recommendation is practical and cheap.
- **The audit of 14 fusion papers** (Table A.1) separates full-text from abstract-only evidence, which is rare and welcome.
- **The duplicate analysis on UNSW-NB15** (Table B.2) is a useful finding in its own right. Matched records are equally hard for every model, and the RF's lead lies in the unmatched records.

## Assessment

The science is sound and honestly reported. The remaining weaknesses are of three kinds:

1. One factual inconsistency that changes a headline trade-off (M1).
2. Inferential claims that need sharper wording or a little more computation (M2, M4, minor 2).
3. Scope and length (M5, M6).

M1, M2, minor 2 and the length cuts need no training. M3 and M4 together need a few GPU-hours. With M1–M4 and M6 done and the framing tightened (M5 by rescoping at minimum), I would expect to move to accept with minor revisions, at about 7.5–8. Without M1 fixed, the abstract contains a claim the paper's own code does not support.
