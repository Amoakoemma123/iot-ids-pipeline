# Paper 3 — final numbers and consistency audit

Date: 2026-11-07. Scope: `Manuscript_filled.md` (built 2026-11-07 00:33), tables and captions (`build.py` TABLE_CAPTIONS), Figure 1 caption, Highlights (`build.py` HIGHLIGHTS / `Highlights.docx`), checked against `values.json` (2026-11-03 06:55), and spot-checked against the raw run JSONs under `Papers pipeline/experiments/paper3_continual/results_v3` and `results_v2`.

## What was done

- `verify.py` run: **0 failures, 0 pending**.
- `compute_values.py` was **not** run. `values.json` is newer than every result JSON (newest result written about 12 minutes before values.json), so the values are not stale.
- Every `{{key}}` in `Manuscript.md` was traced to `values.json`, and every filled value in the text and tables matches. There are no `⟦…⟧` / `{{` leftovers in the filled text, the docx files or the highlights.
- I checked these directly against the raw JSONs, using my own code (my own macro-F1 from the confusion matrices, not `analyze_cl_v2.metrics`). All of them match `values.json` to the stated precision:
  - final macro-F1 mean ± SD for all 7 strategies × 3 datasets (the stored `macro_f1_seen` equals macro-F1 recomputed from the last-stage confusion matrix over all classes, with a maximum difference of 0.0)
  - earlier-task detection and class retention (fine-tuning: 0.875/0.698/0.875 and 0.000/0.000/0.037)
  - final benign FPR (TON ER-2000 0.132, ER-500 0.116, joint 0.032)
  - zero-day flagging at stage 0 (0.738/0.779, 0.523/0.895, 0.461/0.430)
  - Holm-adjusted p-values (e.g. EWC 0.11/0.19/0.08; LwF TON 0.003)
  - RF macro-F1 (5 runs per dataset: 0.965/0.575/1.000)
  - oracle differences and p-values (TON joint −0.016, p 0.0084; NB fine-tuning −0.029, p 0.25)
  - factorial cells (0.805/0.871/0.908)
  - earlier-protocol means (TON joint 0.768, ER-2000 0.507, fine-tuning 0.348)
  - Table 1 training records (n_train)
  - Task-0 confusion matrices, which are identical across all 7 strategies for every seed on all three datasets. This is stronger than the F1-equality check in compute_values, so "verified bit-identical" holds.

The problems found are therefore not transcription errors. They are method descriptions that do not match the code, wording that is stronger than the evidence, and inconsistent use of "earlier version".

---

## WRONG

### W1. Share of TON-IoT test values "outside the Task-0 range"
- **Quote (§5.5, Pre-processing scope):** "Only 0.002% of TON-IoT test values fall outside the Task-0 range, and the compression map handles larger values smoothly."
- **Problem:** `TON_clip_share` comes from `clip_share_te`. In `run_cl_v2.py` (lines 161–164) this is computed **after** the asinh compression, as the share of values outside [−1, 2] in compressed units. In compressed units, 2 corresponds to x ≈ sinh(2·asinh 1000)/1000 ≈ 2,000 in min–max units. So the 0.002% counts values more than about 2,000 Task-0 ranges above the maximum, not values outside the Task-0 range.
- **Correct value / source:** The share of values outside [0, 1] (the Task-0 range) is not stored, and it cannot be computed without loading the data. The closest stored figure: in the min–max-only runs (`results_v3/TON_IoT/factorial/*minmax*`, and the `results_v2` runs), `clip_share_te` = 5.03 × 10⁻⁵, i.e. **0.005%** of test values lie outside [−1, 2] in min–max units. That is a lower bound on "outside the Task-0 range".
- **Fix:** Either recompute the share outside [0, 1] in a data-loading run, or reword, e.g. "only 0.005% of TON-IoT test values lie more than one Task-0 range outside it".

### W2. Order of clipping and compression in §3.3
- **Quote (§3.3):** "Later-task values outside the Task-0 range are clipped to [−1, 2]. After scaling, every feature is compressed by the fixed monotone map x ↦ asinh(1000x)/asinh(1000) … and extrapolates smoothly to later-task values".
- **Problem:** The text reads as min–max → clip to [−1, 2] → compress. The code does min–max → asinh → `np.clip(X, -1, 2)` (`run_cl_v2.py` 161–165; `PreprocessingPipeline` uses sklearn `MinMaxScaler` without clipping). In min–max units the effective window is about [−1, ≈2,000], not [−1, 2]. The "extrapolates smoothly" claim is only true because the clip comes after the compression.
- **Fix:** Say "min–max scaling, then the compression map, then clipping to [−1, 2] in compressed units (about −1 to 2,000 in min–max units)".

### W3. Table 7 caption, §4.3 and §5.6: training budget of joint retraining
- **Quote (Table 7 caption, build.py):** "The earlier budget is 40 epochs on Task 0 and 20 on later tasks (patience 5); the present one is 100 and 50 (patience 15 and 10)." Also §4.3 ("the earlier 40/20 epochs with patience 5, or the present budget") and §5.6, first paragraph.
- **Problem:** Table 7 is for joint retraining only. In `run_cl_v2.py` (line 243) joint retraining uses `cfg = CFG0` at **every** stage. `BUDGETS` (line 75) is v3 = 100 epochs/patience 15/lr 1e-3 and v2 = 40/patience 5/lr 1e-3 for CFG0. The 50- and 20-epoch budgets never apply to joint retraining. §4.1 and §5.2 ("Incremental tasks are trained with a smaller budget than joint retraining") already say this correctly, so the caption contradicts the paper's own Section 4.1.
- **Correct:** For joint retraining, the earlier budget is at most 40 epochs (patience 5) and the present budget at most 100 epochs (patience 15), at every stage. §5.6 paragraph 1 can keep 40/20 for the incremental strategies, but should say so.

### W4. "Rises" for zero-day flagging after Task 1
- **Quote (§5.4):** "After Task 1, flagging of the still-unseen Task-2 families rises to 0.754–0.908 on TON-IoT, 0.886–0.981 on UNSW-NB15 and 0.984–1.000 on N-BaIoT."
- **Problem:** The lower ends of the ranges are below the stage-0 rates for the same tasks: TON Task 2 was 0.779 after Task 0 and A-GEM is 0.754 after Task 1; UNSW was 0.895, A-GEM 0.886. For A-GEM the rate falls.
- **Source:** values.json `TON_zd_stage0_t2` 0.779, `TON_agem_zd_stage1` 0.754, `UNSW_zd_stage0_t2` 0.895, `UNSW_agem_zd_stage1` 0.886.
- **Fix:** "…is 0.754–0.908 … (it rises for every strategy except A-GEM on TON-IoT and UNSW-NB15)".

---

## OVERSTATED

### O1. Highlight 1 leaves out the task-order qualifier
- **Quote (Highlights):** "Replay of 2,000 records recovered 81–91% of the gap to full retraining"
- **Problem:** 81–91% holds for the semantic order only. Over the random orders it is 64–91% (UNSW 64%). The abstract, §6.1 and §7 all give both ranges.
- **Source:** values.json `*_er2000_gap_closed` 91/85/81; `*_er2000_orders_gap_closed` 91/64/84.
- **Fix:** e.g. "Replay of 2,000 records recovered 64–91% of the gap to full retraining" (65 characters).

### O2. "EWC and A-GEM were no better than fine-tuning"
- **Quotes:** Abstract: "EWC and A-GEM were no better than fine-tuning". Highlight 2: "Online EWC and A-GEM were no better than plain fine-tuning on three datasets". §7: "Online EWC and A-GEM were no better than fine-tuning".
- **Problem:** This rests on Holm-corrected non-significance with n = 5 paired seeds. That is absence of evidence, not equivalence. EWC gained +0.054 on N-BaIoT (raw p 0.027, Holm 0.08) and +0.042 on UNSW-NB15 (Holm 0.19). It also cut class-level forgetting noticeably on UNSW (FM 0.662 vs 0.872) and N-BaIoT (0.854 vs 0.956).
- **Source:** values.json `*_ewc_vs_naive_d/pholm`, `*_ewc_FM`; raw p recomputed from `results_v3/N_BaIoT/main`.
- **Fix:** "did not significantly improve final macro-F1 over fine-tuning". §5.2 already uses this wording correctly.

### O3. "Its remaining shortfall lies in the newest family, not in forgetting"
- **Quotes:** Abstract ("its remaining shortfall lies in the newest family, not in forgetting"), §5.2 ("The remaining cost of replay therefore lies in learning the newest task, not in forgetting"), §7 ("The remaining gap to retraining lies in learning the newest family").
- **Problem:** Table 3 only covers attack-class recall. On TON-IoT a visible part of the 0.050 macro-F1 gap is the **benign** class: ER-2000 benign recall is 0.87 against 0.97 for joint (FPR 0.132 vs 0.032), and that also lowers attack-class precision. The claim holds well on UNSW-NB15 and N-BaIoT, where benign recall is equal (0.80/0.80 and 1.00/1.00).
- **Source:** values.json `TON_er2000_rec_normal` 0.87, `TON_joint_rec_normal` 0.97.
- **Fix:** Add "…and, on TON-IoT, in benign false positives", or limit the claim to forgetting versus plasticity for the attack classes.

### O4. §5.6: the earlier failure "lowered … replay"
- **Quote (§5.6):** "The failure lowered the upper bound and replay much more than fine-tuning, which shrank the apparent benefit of replay."
- **Problem:** Placed after the three-dataset summary, but it does not hold on N-BaIoT. There ER-2000 was **higher** in the earlier run (0.879) than now (0.851), and joint hardly changed (0.983 vs 0.987).
- **Source:** values.json `NB_v2_er2000_f1` 0.879, `NB_er2000_f1` 0.851. Confirmed from `results_v2/N_BaIoT/main`.
- **Fix:** "On TON-IoT and UNSW-NB15, the failure lowered…".

### O5. Abstract on the pre-processing shortcut (no N-BaIoT caveat), and "did not inflate"
- **Quote (Abstract):** "Fitting the pre-processing on future data changed final macro-F1 by at most 0.029 on average."
- **Problem:** The statement is literally true for the mean changes, but it reads as a bound. On N-BaIoT the 95% CI for fine-tuning is [−0.089, +0.031], and §1 contribution 3, §5.5, §6.2 and §7 all call that dataset inconclusive. The abstract is the only summary place without the caveat. Separately, §5.5 says "the shortcut did not inflate final results on TON-IoT and UNSW-NB15", but the TON ER-2000 CI reaches +0.022.
- **Source:** values.json `NB_naive_oracle_ci`, `TON_er2000_oracle_ci` [−0.007, +0.022].
- **Fix:** Abstract: add "(inconclusive on N-BaIoT)". §5.5: "did not inflate final results by more than about 0.02".

### O6. "Five seeds and five task orders"
- **Quotes:** Abstract ("uses five seeds and five task orders"), §1 contribution 1, §7 ("five seeds and five task orders").
- **Problem:** This reads as 5 × 5 runs. In fact there are five seeds on the semantic order only, and one run on each of the four random orders. Those runs also use seed = order index (run keys (1,1), (2,2), (3,3), (4,4)), so order effects are confounded with seed effects. §4.5 and the Table 5 caption state this correctly.
- **Fix:** "five seeds on the semantic order and four further random task orders".

### O7. "The input map was the larger factor"
- **Quote (§5.6):** "The input map was the larger factor. Adding the compression map raised final macro-F1 by +0.103 under the earlier budget and by +0.051 under the present one. The longer budget added +0.066 with min–max inputs…"
- **Problem:** Under the present budget, adding the map (+0.051) is **smaller** than adding the budget to min–max inputs (+0.066). The map is larger only under the earlier budget and on average (two-way average: map ≈ +0.077, budget ≈ +0.041). `verify.py` passes this check only because of a −0.02 tolerance (`map_effect > max(budget effects) − 0.02`). No test was run; the seed SD of the min–max cells is up to 0.027.
- **Fix:** "Averaged over the two budgets, the input map contributed more (+0.077 vs +0.041)…".

---

## AMBIGUOUS

### A1. "Earlier version" refers to two different studies
- **Quotes:** §1: "The earlier, unpublished version … one seed … benign traffic only in the first task and no upper bound. Re-running it with a corrected protocol exposed a further problem … joint-retraining macro-F1 of 0.77". Abstract: "exposed a silent failure of the network in an earlier version of this study, which had depressed the retraining bound on TON-IoT from 0.922 to 0.768". Highlight 5: "…silent CNN failure in an earlier version". §5.6 title and text ("The previous version of this protocol (Section 1)"), §6.1 ("exposed the earlier version's error") and §7.
- **Problem:** The 0.768 joint result and the silent failure come from `results_v2`: 5 seeds per strategy and benign traffic in every task (checked in the run files). That is the **corrected-protocol re-run**, not the one-seed original that §1 describes as having "no upper bound". Other sentences, by contrast, do mean the original: §5.2 "presented EWC as a competitive storage-free alternative" and §5.6 "reported different Task-0 scores". A reader cannot tell which version had a 0.768 retraining bound when §1 says it had none. Also, "depressed … from 0.922 to 0.768" puts the whole drop on the failure, while §5.6 admits the reconstruction reaches 0.805 and the remaining 0.037 involves code and loss changes. §4.3 also uses the internal label "**v2 factorial**".
- **Fix:** Name the two consistently, e.g. "original study" (v1) and "first re-run" (v2). Rename "v2 factorial" to "input-map × budget factorial". In the abstract and highlight, say "in a re-run of an earlier version".

### A2. Table 2 caption: "mean ± standard deviation over all classes"
- **Quote (build.py, Table 2 caption):** "Final macro-F1: mean ± standard deviation over all classes."
- **Problem:** The SD is over the five seeds; the macro average is over all classes (including benign). As written, it suggests an SD across classes.
- **Fix:** "Final macro-F1 over all classes including benign: mean ± SD over five seeds."

### A3. "Over the Task-0 classes" includes benign
- **Quote (§5.1):** "That model reaches a macro-F1 of 0.987 on TON-IoT, 0.921 on UNSW-NB15 and 0.987 on N-BaIoT over the Task-0 classes."
- **Problem:** `macro_f1_seen` at stage 0 is computed over the seen classes, which include `normal` (stage-0 seen = dos, normal, scanning on TON).
- **Fix:** "over benign traffic and the Task-0 attack classes".

### A4. Uncorrected p-values used for "significant" outside the confirmatory family
- **Quote (§5.5):** "the only significant change was a small loss for joint retraining on TON-IoT (−0.016, p = 0.0084)". Similar unlabelled p-values appear for ER-2000b (p = 0.0032, p = 0.009) and the factorial reconstruction (p = 0.3).
- **Problem:** §4.5 says "Other comparisons are descriptive", yet "significant" is used without saying the p is uncorrected. With within-dataset Holm over the 3 oracle tests, p = 0.025 (still < 0.05). Over all 9 oracle tests, p = 0.076 (not significant). The budget check and the memory comparison are correctly labelled "uncorrected".
- **Fix:** Label as "uncorrected p = 0.0084 (Holm over the three TON tests: 0.025)". Optionally drop "significant" for non-confirmatory comparisons.

---

## OK-but-note

1. **All filled numbers verified.** See "What was done". verify.py reports 0/0, and the independent raw recomputation agrees.
2. **Draft header in Manuscript_filled.md.** Line 1 reads "# Paper 3 v2 manuscript (double-brace placeholders are filled from values.json by build.py)". build.py skips line 0, so it does not reach the docx, but the filled .md files still carry it.
3. **Training time.** §6.1 says "a full retraining sequence took five to ten minutes". The raw joint `train_s` sums are 275–604 s, i.e. 4.6–10.1 min (verify.py accepts 240–660 s). Fine, but "about five to ten minutes" is more exact.
4. **p-value formatting.** "0.00015" (§5.2, memory comparison) sits next to "< 0.001" elsewhere. In §5.2 LwF "p 0.003" and "p < 0.001" are Holm-adjusted but not labelled as such; EWC is labelled.
5. **Abstract length.** 248 words against the 250 limit (build.py). No room for the O5 caveat without trimming elsewhere.
6. **N-BaIoT benign FPR.** §5.3 says it "stays at or below 0.003 for every strategy at every stage". That is true of the seed means; individual seeds reach 0.0048 (stage 0). Add "on average" or quote 0.005.
7. **§4.3 count of ablations.** "Two further ablations on TON-IoT address…" is followed by three ablations; the third (budget check) runs on all three datasets. Reads oddly, but is not wrong.
8. **Memory sizes are KiB.** "102 kB / 305 kB / 898 kB" are computed as bytes/1024. Either "KiB", or kB = 104/312/920.
9. **"Every regularisation strategy".** §5.5 says "replay beat fine-tuning and every regularisation strategy in every order". The computation also includes A-GEM, which is gradient projection; "every non-replay strategy" (as in the next sentence and the abstract) is more accurate.
10. **"Placing the rare classes last thus hurts…".** §5.5 makes a causal reading from one semantic order against four single-seed random orders. Reasonable, but "suggests" would be safer.
11. **Gap closed over random orders.** It is computed as a ratio of means, from the 3-decimal rounded means in values.json. The values (91/64/84) reproduce, but it is not a mean of per-order ratios.
12. **Negative detection-level FM.** Table 2 shows −0.006, −0.017, −0.007 and −0.001 in a column headed "forgetting". These are correct (final retention above the earlier maximum), but the caption could say "negative = backward transfer".
13. **Memory 500 → 2,000 "improved".** §5.2: +0.017/+0.087/+0.053, with p 0.29/0.00015/0.14. It is labelled exploratory, but only UNSW is clearly different.
14. **"Replay raised the benign FPR on TON-IoT to 0.132".** Abstract and Highlight 4: this is ER-2000; ER-500 is 0.116. §6.1 gives the range 0.116–0.132 correctly. Acceptable as written ("Replay with 2,000 records" was just named in the abstract), but the highlight does not name the variant.

---

## Counts

| Class | Count |
|---|---|
| WRONG | 4 |
| OVERSTATED | 7 |
| AMBIGUOUS | 4 |
| OK-but-note | 14 |
