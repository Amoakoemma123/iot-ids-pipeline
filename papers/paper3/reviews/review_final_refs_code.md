# Paper 3: final check of the references and of the methods against the code (2026-11-07)

Manuscript checked: `Manuscript_filled.md`. Its reference section is identical to the one in `Manuscript.md`.
Code checked: `Papers pipeline/experiments/paper3_continual/run_cl_v2.py` (git 96b2fbc; the training logic is unchanged since be87c9c, the v3 commit that produced results_v3), `analyze_cl_v2.py`, `src/data/loader.py`, `src/data/preprocessing.py`, `experiments/paper4_federated/prepare_nbaiot.py` and `Revision_v2/Paper3/compute_values.py`. I also read the stored run files in `results_v3`. Nothing was trained and no dataset was loaded.

Severity: **critical** means a reported result or conclusion is invalid or confounded. **major** means the text misdescribes the protocol in a way that matters, or a claim is unsupported. **minor** means a description gap or a small inaccuracy.

---

## References

### Checks run

- **check_citations.py.** It passes on `Manuscript.md`: 0 citations missing from the list and 0 uncited entries. I also ran a copy pointed at `Manuscript_filled.md`, and it passes too. Limitation: the script does not tell the a/b suffixes apart. I checked those by hand. Chaudhry 2019a (A-GEM) is cited in §2.1 and §4.2, and Chaudhry 2019b (ER) is cited in §2.1 and §4.2, so both are used.
- **Existence and metadata.** All 44 entries are in the 2026-11-05 audit (`reference_audit/refcheck.json`, P3), and the entry text matches exactly. That audit only checked title overlap. I therefore re-queried Crossref for all 35 Crossref DOIs and compared the authors, year, title, container, volume, issue and pages. All 35 match. Notes on that comparison:
  - **De Lange et al. (2022).** Crossref still holds the early-access record: year 2021, pages "1-1", first author spelled "Delange". The cited final issue, TPAMI 44(7) 3366–3385 (2022), is correct, as confirmed in literature_verified.md. No change needed.
  - **Amalapuram et al. (2024).** Crossref gives the family name as "Reddy Tamma". The citation "Tamma, B. R." is acceptable.
  - **Chaudhry 2019b and Farquhar & Gal 2018.** Both are arXiv DataCite DOIs. Crossref returns 404, but both resolve at doi.org. Authors and titles are correct.
  - **The 9 entries without a DOI.** These are Chaudhry 2019a, Ioffe & Szegedy 2015, Jordaney 2017, Kingma & Ba 2015, Lopez-Paz & Ranzato 2017, Pendlebury 2019, Rahman 2022, Schwarz 2018 and Yang 2021. All were confirmed earlier, by DBLP/URL in the audit and against the proceedings pages in literature_verified.md. I re-checked Rahman et al. on PMLR: authors Rahman, Coull & Wright, PMLR 199, pp. 564–582. Correct.
  - **Not in literature_verified.md.** Guyon et al. (2002), Ioffe & Szegedy (2015) and Kingma & Ba (2015) are not listed there. Guyon matches Crossref in full (Machine Learning 46(1–3) 389–422). Ioffe and Kingma were matched by title in the audit, and their metadata are standard and correct.
- **Result: no entry is nonexistent, and no metadata error was found.**

### Findings

**R1. Rahman et al. (2022) is misrepresented (major).** Location: §2.2, para. 1, "Rahman et al. (2022) found that class-incremental methods such as EWC, LwF and iCaRL largely fail for malware classification, while replay and joint retraining dominate". The paper is echoed in §5.2: "This pattern matches … the results for malware classification (Rahman et al., 2022)".
- Evidence from the full text (PMLR PDF):
  - Contributions list: "For Class-IL, 10 of the 11 methods performed poorly, with only iCaRL … performing marginally better against the Joint replay baseline."
  - §5, Class-IL: "All the regularization-based techniques and LwF perform poorly and very close to None … iCaRL outperforms all the other CL techniques with 96.2% mean accuracy" on Drebin.
  - Conclusion: "only iCaRL … performed near Joint replay baselines."
  - Plain experience replay (ER, Rolnick et al.) is among the methods that performed poorly. The "replay 20–50% of the data" result is partial joint replay in the **Domain-IL** setting.
- iCaRL is therefore the counter-example, not one of the methods that fail.
- Suggested text: "Rahman et al. (2022) found that, in class-incremental malware classification, regularisation methods and LwF performed close to no adaptation and only the exemplar-based iCaRL approached joint retraining; in the domain-incremental setting, replaying 20–50% of past data outperformed all continual-learning methods." The §5.2 sentence can stay once §2.2 says this, because regularisation fails and exemplar replay works in both studies.

**R2. Alsaedi et al. (2020) is cited for the network flows (minor).** Location: §3.1, "TON-IoT contains network flows … (Alsaedi et al., 2020; Moustafa, 2021)". Alsaedi et al. describe the TON_IoT *telemetry* (IoT/IIoT sensor) datasets. The network flows used here (train_test_network.csv) are described by Moustafa (2021), as literature_verified.md d2 notes. Put Moustafa (2021) first, or write "(Moustafa, 2021; see also Alsaedi et al., 2020)".

**R3. Mai et al. (2022) are paraphrased loosely (minor).** Location: §2.1, "controlled comparisons show that protocol details drive rankings (Mai et al., 2022)". Mai et al. is an empirical survey of *online* CL in image classification. Its main point is that implementation tricks and settings change the ranking of methods. Suggested wording: "controlled comparisons show that implementation tricks and evaluation settings change the ranking of methods (Mai et al., 2022)".

**R4. Protocol statements rest on abstracts only (minor; already flagged in literature_verified.md).** §2.2 attributes specific scenarios or methods to studies verified from the abstract only: Amalapuram et al. 2021, 2022, 2024 and 2025, Prasath et al., Oikonomou et al., Faber et al., Mao et al., Guo et al. and A H et al. The current wording is generic and matches those abstracts ("studied class imbalance", "compared EWC and GEM", "shares replay", and so on). The negative statements ("none of these separates …") are correctly limited to the four studies read in full. No change is needed, but do not add protocol claims about the abstract-only studies.

**R5. Lopez-Paz & Ranzato (2017) page numbers come from OpenAlex only (minor).** The pages (6467–6476) are not on the NeurIPS page. This is acceptable for NeurIPS; you can drop the pages if the journal's style allows.

---

## Methods vs code

### Confirmed matches

The following were checked against the code and stored results and are correct as written:
- **Network:** architecture (Conv32/Conv64, kernel 3, BN momentum 0.9, max-pool 2, GAP, Dense 128/64, dropout 0.3, single linear head).
- **Training:** masking of unseen logits in training and prediction; class weights inverse to frequency, capped at 50; Adam 1e-3 with 100/15 (Task 0 and joint) and 5e-4 with 50/10 (incremental); batch size 512; best weights restored.
- **Data splits and tasks:** benign shares disjoint and fixed per order (rng 700+order); random orders keep the task sizes (rng 500+order); the test set is never split; the Table 1 counts equal the stored `n_train` values (TON 33,085 / 38,395 / 57,004).
- **Regularisation grids and tuning:** EWC grid 10⁰–10⁷; LwF grid 0.1–32, temperature 2 (with T² scaling); tuning on development seed 100 by validation macro-F1 over all seen classes after the last task. The tuned values (EWC 100 / 1e5 / 100; LwF 4 / 1 / 0.5) and the N-BaIoT development scores (0.266 at 0.5, 0.226 at 0.25, 0.214 at 0.1) match the text. The EWC choices are interior.
- **Fisher information:** 1,000 records, per-sample, with true labels.
- **Random forest:** 200 trees, depth 20, minimum leaf 2, five seeds, Task-0 pre-processing.
- **Ablations:**
  - Factorial: v2 budget 40/20 epochs with patience 5. The Task-0 confusion matrices of the min-max/v2 cell equal the archived results_v2 Task-0 matrices for all 5 seeds.
  - Budget check: CFGI = CFG0.
  - Benign-share memory: 1,000 benign records, with the rest split equally among the attack classes.
- **Metrics:** R, Rd, FM, BWT, plasticity, zero-day flagging (record-weighted per task, unweighted mean over the 3 cells) and benign FPR all follow §4.4.
- **Statistics:** paired t-test vs fine-tuning on final macro-F1 with Holm correction over 6 comparisons per dataset; uncorrected p-values for the exploratory tests; t-based 95% CIs for the oracle ablation.

### Findings

**C1. A-GEM bug: the reference gradient ignores every earlier attack class (critical).**
- Location: §4.2 (A-GEM), §5.2 ("A-GEM did not differ either … Used only to constrain the gradient direction, it does not"), §5.3 ("A-GEM on UNSW-NB15 … moves old attacks into the benign class"), Table 2, Table 4, Table 5, the Abstract ("EWC and A-GEM were no better than fine-tuning") and §6.1/§7.
- Evidence, `run_cl_v2.py`:
  - l. 252: `pool_y = yt if … strategy == 'agem' …`. For A-GEM the class weights come from the current task's labels only.
  - l. 184–189: `class_weights` gives weight **0** to every class absent from `pool_y`, which means every earlier attack class.
  - l. 264–267: `ce` returns `sum(l * w[y]) / sum(w[y])`.
  - l. 283–284: the A-GEM reference loss is `ce(xm, ym)` on the memory batch.
  - Result: in the reference loss, every memory record of an earlier attack class has weight 0. Only the benign memory records, whose class is present in the current task, contribute. The projection therefore only stops the benign loss on the memory from rising. It never protects the old attack classes.
- This explains the observed A-GEM behaviour: no gain over fine-tuning, and on UNSW-NB15 the lowest benign FPR (0.017) with old attacks moved into the benign class (detection retention 0.496). It is an artefact of the bug, not a property of A-GEM.
- Fix:
  - Compute the reference loss with weights from the memory's own labels (`class_weights(ym)` or `class_weights(buf.y)`), or unweighted.
  - Rerun A-GEM: main, 5 seeds × 3 datasets, and orders, 4 × 3. The Task-0 cache makes this cheap.
  - Then re-derive every A-GEM statement, including "A-GEM gives the lowest rate" in §5.4 and the BN explanation in §5.2.
  - Until then, the A-GEM conclusions cannot be reported.

**C2. Replay: the training mix and the class weights do not match the text and over-weight earlier classes (major; may be critical for the interpretive claims).**
- Location: §4.2, "a class-balanced episodic memory … is added to each new task's training data"; §4.1, "batch size 512"; and the interpretations in §5.2 ("its remaining shortfall lies in the newest family, not in forgetting"), §5.3 (replay's benign FPR "comes from under-representing benign traffic") and the Abstract.
- Evidence:
  - l. 305–310 and 272–273: each step draws **512 memory records with replacement** and concatenates them with the 512 current records. Each batch is therefore 1,024 rows, half of them from memory. The memory is not simply added to the training data.
  - l. 252–253: the class weights are computed on the union pool, current task plus memory, as if each memory record appeared once per epoch.
  - The two effects multiply. Example, TON Task 2 with ER-2000. The memory holds 2,000/6 ≈ 333 records for each of benign and the 5 earlier attack classes, in a pool of about 59,000 records with 10 classes.
    - Each earlier class gets weight ≈ 59,000/(10·333) ≈ 17.7 and about 512/6 ≈ 85 rows per batch, a weighted contribution of ≈ 1,500.
    - A new class with n_c records gets 512·n_c/57,004 rows × 59,004/(10·n_c) ≈ 53, whatever n_c is.
    - So each earlier attack class weighs about **28 times** as much as a newest-task class in every update. Benign traffic, which is mostly in the current-task part, also weighs far less than each old attack class.
- Consequence: the result that replay keeps the old tasks but under-fits the newest one (Table 3, newest-task recall 0.218 on UNSW-NB15 and 0.484 on N-BaIoT), and replay's high benign FPR on TON-IoT, may be produced by this weighting rather than by replay as such. The benign-share ablation (ER-2000b) is consistent with this reading. The headline ranking (replay ≫ the rest) is probably robust.
- Fix:
  - Describe the scheme exactly (1:1 interleaving, effective batch 1,024).
  - Either compute the weights on the sampled mixture, or run one sensitivity check (unweighted, or mixture-matched weights, ER-2000, 5 seeds) before claiming that the shortfall "lies in the newest family, not in forgetting" and the benign-FPR mechanism.
  - At minimum, list the weighting as an alternative explanation in §5.2, §5.3 and the Limitations.

**C3. Early stopping and checkpoint selection use current-task validation only (major).**
- Location: §4.1, "Early stopping monitors macro-F1 on the validation records available at that stage".
- Evidence:
  - l. 247, 251 and 312: for every incremental strategy except joint, `va = task['va']` (this task's benign share plus its attack classes), scored with `labels=sorted(set(yv))`. The code docstring says "current task; buffer-free".
  - Old-class validation data are never used to choose the epoch.
- Effect: the restored "best" epoch is the one that maximises plasticity on the new task. This works against retention for EWC, LwF and A-GEM (A-GEM stopped after 11 and 16 epochs on TON) and probably adds to the near-total class-level forgetting.
- Fix:
  - State it explicitly: "validation records of the current task (benign share and new classes); validation data of earlier tasks are not retained."
  - Note in the Limitations that this choice favours plasticity. A replay strategy could fairly use its memory for validation.

**C4. The clipping order is described wrongly, and the "0.002% outside the Task-0 range" statistic is mislabelled (major).**
- Location: §3.3, "Later-task values outside the Task-0 range are clipped to [−1, 2]. After scaling, every feature is compressed …"; §5.5, "Only 0.002% of TON-IoT test values fall outside the Task-0 range".
- Evidence:
  - l. 161–165: the code applies min–max, **then** asinh, **then** clips to [−1, 2].
  - In min–max units, the effective clip is therefore [−1, ≈ 2,000], because asinh(1000x)/asinh(1000) = 2 at x ≈ 2.0×10³.
  - The 0.002% figure (`clip_share_te` = 1.96e-5, compute_values l. 145) is the share of compressed values outside [−1, 2]. It is not the share outside the Task-0 range [0, 1], which is not computed anywhere and is presumably larger.
- Fix:
  - Say "after compression, values are clipped to [−1, 2] (min–max values below −1 or above about 2,000)".
  - Either compute the true share of test values outside [0, 1], which needs no training (only `preprocess`), or delete the sentence.
  - The argument that the shortcut is harmless should not rest on this number.

**C5. "Pre-processing fitted on the first task only" is not fully true (major for the claim; the effect is probably small).**
- Location: Abstract, §1 contribution 1, §3.3 and §7.
- Evidence:
  - `src/data/loader.py` steps 6–9 use statistics of the **full training partition** (all tasks): dropping numeric columns with >40% missing, median imputation, ±inf replaced by the train 99th percentile, and dropping zero-variance columns.
  - Step 4 decides the one-hot cardinality (≤30) and the dummy columns on the full dataset before the split.
  - For N-BaIoT, `prepare_nbaiot.py` drops zero-variance features and fits min–max (with a clip at 0) on the full training split. The Task-0 refit is affine, so this does not matter for the scaling.
  - Only MI, RFECV and MinMax are refitted on Task 0, so the oracle ablation varies only these.
- Fix: in §3.3, state that the dataset-level cleaning (imputation, infinity replacement, column drops, one-hot schema) uses the whole training split for both protocols, and that the Task-0-only fit covers feature selection and scaling.

**C6. N-BaIoT has no feature selection and its preparation is under-described (minor).**
- Evidence:
  - `preprocess` l. 154–158: there is no MI/RFECV for N-BaIoT, and all 115 features are used (stored runs: 115 features).
  - §3.3 implies MI + RFECV for every dataset.
  - `prepare_nbaiot.py`/`nbaiot_prepared.json` show further steps the text omits:
    - removal of conflicting-label vectors (17 rows);
    - the two BASHLITE floods dropped because only 40 and 32 distinct float32 vectors remain (MIN_DISTINCT), which is the actual reason;
    - a stratified 70/15/15 split by device × class;
    - a cross-split guard.
- Fix: one sentence in §3.1 and one in §3.3.

**C7. Feature-selection settings are not reported (minor).** Location: §3.3. The code uses an MI threshold of 0.05; RFECV with a 50-tree random forest, 3-fold stratified CV, macro-F1 scoring, step 1, at least 10 features, on a 50,000-row subsample; and seed 42. The resulting feature counts are 13 for TON-IoT and 39 for UNSW-NB15. Add these, for example in a table footnote or the supplement.

**C8. "Counts only after five epochs" does not describe the code (minor).**
- Location: §4.1.
- Evidence: l. 313–318. The patience counter starts at epoch 1. `MIN_EPOCHS = 5` only blocks a stop before epoch 5, which can never happen because patience is at least 5 (also under the v2 budget). Best weights can come from epoch 1.
- Fix: delete the clause, or implement it.

**C9. A-GEM and batch normalisation: the stated mechanism is partly wrong (minor; superseded by C1).**
- Location: §5.2, "Neither EWC nor A-GEM protects the batch-normalisation statistics, which are re-estimated on each new task's data".
- Evidence: l. 284. The A-GEM reference loss runs `model(xm, training=True)`, so the BN running statistics (and dropout) are also updated on the memory batches.
- Also: the A-GEM reference batch is 512 records, which the text does not say.

**C10. The online-EWC details should be stated (minor).** Location: §4.2. The code sums the Fisher over tasks without decay (γ = 1), estimates it on 1,000 training records of the task just learned using the true labels, and anchors the penalty λ/2·ΣF(θ−θ*)² at the most recent parameters. "Accumulated over tasks" is correct, but say "no decay (γ = 1)".

**C11. Memory sizes and storage figures (minor).**
- Evidence (stored runs):
  - On UNSW-NB15 the final ER-2000 memory holds 1,889 records, not 2,000, because some rare classes have fewer records than their quota.
  - ER-500 ends with 50 records per class on TON-IoT.
  - The storage figures (102 / 305 / 898 "kB") are computed with /1024 (compute_values l. 146), so they are KiB.
- Fix: say "at most 2,000"; write KiB or recompute in kB (104 / 312 / 920 kB).

**C12. "Verified bit-identical" overstates the check (minor).**
- Location: §4.1 and §5.1.
- Evidence:
  - l. 297–331: Task 0 is trained once per (dataset, order, seed, pre-processing) and the weights are reused from `cache_v2/task0`, so identity holds by construction.
  - The check (`compute_values` l. 69) compares the Task-0 macro-F1 to 6 decimals.
- Suggested wording: "Task 0 is trained once per seed and the same weights are reused by all strategies."

**C13. Seeds and sources of variation (minor).**
- Location: §4.5.
- Evidence:
  - The random orders use training seed = order index (1–4).
  - The data split (seed 42), the benign shares (per order) and the pre-processing fit are fixed across seeds.
  - The five-seed SDs and paired tests therefore cover initialisation and batch order only, not the data split.
- Fix: one sentence in §4.5 or the Limitations.

**C14. LwF distils benign as well as the old attack classes (minor).** Location: §4.2. In l. 255 and 279–282, `old_seen` includes benign, so the distillation covers benign plus the old attack classes and is computed on the current-task data. Say "old classes, including benign".

**C15. The random-forest equivalence argument is slightly loose (minor).** Location: §5.6, "A tree is unaffected by monotone rescaling … so this reference applies to both versions". v2 clipped the min–max values at [−1, 2], while v3 effectively clips at [−1, ≈2,000] (see C4). The clipping differs, so the inputs are not exactly monotone transforms of each other on clipped values. The effect is negligible (≈0.002% of values), but "applies to both versions up to the clipping of a few extreme values" is more accurate.

### No other code bugs found

The following were checked and found correct:
- **Task-0 cache:** the keys include the order, seed, oracle flag and factorial variant. The cache was created after the v2 runs had finished, and the v3 Task-0 matrices differ from the v2 ones, so no stale v2 weights were reused.
- **Overrides:** the global overrides in the factorial and budget groups are restored afterwards.
- **Buffer and LwF:** the buffer size is parsed correctly for er500, er2000 and er2000b; the LwF loss is masked correctly.
- **Metrics:** the formulas in `analyze_cl_v2.metrics` match §4.4.
- **compute_values.py:** reads results_v3 for the main, orders, oracle, budget, benignbuf, factorial and rf groups, and results_v2 only for §5.6.
- **Reproducibility of the stored runs:** the training logic did not change between the v3 commits (be87c9c → 96b2fbc only added groups, the grid extension and the tuning criterion), so the released code should reproduce the stored runs.

### Priority

1. **C1:** fix and rerun A-GEM. Until then the A-GEM results are invalid.
2. **C2:** describe the replay mixing and either test the double weighting or caveat the newest-task and benign-FPR interpretations.
3. **R1, C3, C4, C5:** text corrections.
4. The remaining items are minor wording and reporting fixes.
