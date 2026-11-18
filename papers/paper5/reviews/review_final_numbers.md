# Paper 5: final audit of numbers and internal consistency

Date: 2026-11-07. Files audited: Manuscript.md, Manuscript_filled.md (and _anon), build.py (HIGHLIGHTS, figure and table captions), values.json, verify.py, make_figures.py, and the raw result files under `Papers pipeline/experiments/paper5_xai/results_v2` and `results_v3`. compute_values.py was **not** run. No models were trained and no datasets were loaded. The two raw CSV files were only line-counted with `wc -l`.

## What was checked and found correct

- `verify.py`: **0 failures**, about 10,272 words before the references. The abstract is 248 words (limit 250). No `{{…}}` or `⟦…⟧` placeholders are left in Manuscript_filled.md, Manuscript_filled_anon.md or Manuscript_v2.docx. Manuscript_filled.md (12:34) is newer than Manuscript.md (12:33), and the docx files (12:36) are newer than both. The anonymised file differs from the named one only in the repository link and the CRediT block.
- Hardcoded numbers in Manuscript.md (everything else is a placeholder):
  - 211,043 TON-IoT records: matches `wc -l` (211,044 lines including the header). UNSW-NB15 has 175,341 rows.
  - 0.67 for the earlier pooled Jaccard: matches results/TON_IoT/consistency_results.json, rf jaccard@5 = 0.6667.
  - 50,000 for the SMOTE cap and the seeds 42–44: consistent with the rest of the text.
- Values checked directly against the raw files, all of which agree with values.json and the text:
  - **Table 1 / decomposition** (leakage_decomposition_fixed.json):
    - Macro-F1 for seed 42: 0.6345, 0.6038, 0.6160 and 0.6089.
    - id effect: 0.0307 [0.0159, 0.0463] with duplicates kept; 0.0070 [−0.0063, 0.0195] with duplicates removed.
    - Seed ranges: 0.030–0.031 and 0.006–0.011. The lower bound is below 0 in all three seeds when duplicates are removed.
    - Accuracy differences per seed: 0.0528, 0.0528 and 0.0526 (duplicates kept); 0.0497, 0.0512 and 0.0517 (duplicates removed).
    - Test records: 26,302 and 15,894.
  - **Table 2** (performance.json for seeds 42–44):
    - Macro-F1 values and ranges, FPRs (0.83%, 3.39%, 14.93% and 19.93%) and model sizes (96.8 MB, 512.3 MB and 0.34 MB) all match.
    - Node counts (704,184 and 3,729,438), 23,946 parameters and times (57, 65, 59 and 48 ms) all match.
  - **Tables 3 and 4** (results_v2 global_shap.json): mean |SHAP| values and rank intervals for the top features match. Cross-model Jaccard: TON-IoT 0.25 / 0.25, UNSW-NB15 0.43. I recomputed both from the tables by hand.
  - **Tables 5 and 6** (local_c.json and stability_c.json for seed 42):
    - j5, shared5, chance-corrected J, fj5, ρ and R² all match, as do self-agreement at 1,000 and 5,000 samples, SHAP self-agreement and the consensus value.
    - Share of records at or above 0.75: 29%, 66% and 9%. Chance J₀: 0.1209 (TON-IoT), 0.1740 (TON-IoT fields), 0.3504 (UNSW-NB15).
    - Three-seed j5 ranges: TON-IoT 0.22–0.24 and 0.30–0.32; UNSW-NB15 0.39–0.41 and 0.48–0.51.
    - Share of the UNSW-NB15 RF sample misclassified: 1 − 188/289 = 35%.
  - **Deletion significance counts** (deletion_c.json for seeds 42–44), recounted independently:
    - SHAP − LIME significant: 36 / 28 / 36 / 31. SHAP − LIME positive: 36 / 35 / 36 / 35. No combination where LIME is significantly better.
    - SHAP better than random: 36 in all four cases. LIME better than random: 36 / 30 / 18 / 30.
    - Top-3 drops for each strategy in Tables 5 and 6 match.
  - **Tables 7 and 8 and Section 5.3** (counterfactual.json for seeds 42–44):
    - Success rates: TON-IoT attacker-controllable 30.7 / 40.7 / 44.4% (RF) and 15.9 / 17.8 / 4.4% (CNN). UNSW-NB15: 42.5 / 40.6 / 41.7% (RF) and 29.6 / 21.2 / 20.9% (CNN).
    - k values: 1.56 [1.49, 1.63] and 1.30 [1.25, 1.36]. Share evadable with one group: 47% and 70%.
    - Median queries: 22.5 → 4 and 206 → 401. SHAP-guided RF success: 87%.
    - Group definitions: 6 groups on TON-IoT covering 25 features; 5 groups on UNSW-NB15 covering 10.
  - **Table 9 and Section 5.4** (hardening*.json for seeds 42–44):
    - All evasion means and ranges, F1 values and FPRs match when recomputed. Example: the TON-IoT counterfactual arm has FPR 0.795%, shown as 0.79.
    - Number of augmented records: 697–755, 663–728, 1799–1800 and 1685.
    - Seeds whose CIs separate: TON-IoT 3/3 and 0/3; UNSW-NB15 2/3 and 1/3, recomputed by hand from evasion_ci.
    - Per-class values: injection 53→2, password 51→2, MITM 98→21, ransomware 86→72.
  - **Section 5.5**:
    - Ports (ports.json): 0.9565 → 0.9420.
    - TTL reselection (ttl_reselect.json): 0.6089 → 0.5794 for the RF and 0.5139 → 0.5106 for the CNN. RF per-class drops: worms 0.528 → 0.295 and shellcode 0.575 → 0.532. Worms accounts for 0.233 of the 0.295 total drop in per-class F1 (79%).
    - sttl rule: 88.7%, 63.7%, macro-F1 0.2016, majority-class baseline 49.0%.
  - **Feature-space copies** (feature_space_copies.json): UNSW-NB15 13.7% / 16.3%, generic 78%, analysis 53%. RF 0.6089 → 0.6472 and CNN 0.5139 → 0.4920, with the CNN pattern holding in every seed. TON-IoT 0.13%.
- Arithmetic checks:
  - 67,601 / 175,341 = 38.55%; 175,341 − 67,601 = 107,740.
  - 11,570 / 26,302 = 44.0%. The weighted accuracy of rows 2 and 3 of Table 1 gives 0.8212, which matches row 1.
  - TON-IoT attacker evasion reduced by 61% (41 → 16, "more than half"). UNSW-NB15 reduced by 24% (42.5 → 32.1, "about a quarter").
- Captions against figure code (make_figures.py):
  - Fig. 1 uses 2,500-record blocks and deduplicates on all columns except `id`.
  - Fig. 2 shows the chance line and the 0.75 line.
  - Fig. 3 uses mean replacement in the top row and `condx` (nearest records, copies excluded) in the bottom row.
  - Fig. 4 shows the mean over seeds with whiskers for the range.
  - Table 9 caption: the means over seeds (0.9569 and 0.6071) correctly differ from the seed-42 values in Table 2 (0.9565 and 0.6089).
  - Tables 7 and 8 captions say "seed 42", which is correct.

---

## WRONG

**W1. Jaccard index read as "fraction of the top-5 set reproduced".** Section 5.2, Stability:
> "recomputed with new backgrounds and random seeds it reproduces only 0.48 [0.46, 0.50] of its CNN top-5 set on TON-IoT and 0.63 [0.61, 0.65] on UNSW-NB15"
> "Even with five times the default budget, the TON-IoT CNN attributions reproduce little more than half of their top-5 set."

- **What the numbers mean:** a Jaccard index of J between two top-5 sets means s = 10J/(1+J) shared features.
  - J = 0.48 → about 3.2 of 5 shared (65%).
  - J = 0.63 → about 3.9 of 5 shared.
  - J = 0.57–0.60 (the 1,000-sample results) → about 3.6–3.75 of 5 shared (73–75%), not "little more than half".
- **Effect:** the wording overstates how unstable the explanations are. The comparisons with LIME are still valid because they use the same metric.
- **Fix:** say "self-agreement (top-5 Jaccard) of only 0.48". For the 1,000-sample case, use "a top-5 Jaccard of only 0.57–0.60" or "about 3.7 of 5 features".
- **Source:** stability_c.json (cnn shap_self 0.477) and values.json `TON_cnn_sb_*`.

**W2. Averaging LIME runs does not bring LIME "closer" to SHAP.** Section 5.2, Stability:
> "Averaging 20 LIME runs barely brings LIME closer to SHAP (RF on TON-IoT: 0.22 [0.20, 0.24] against 0.22 [0.20, 0.25] for a single run) … (CNN on TON-IoT: 0.31 [0.28, 0.34] against 0.33 [0.30, 0.36])"

- **Direction is wrong:** the consensus explanation is no closer to SHAP for the RF and slightly further from it for the CNN.
  - Paired consensus − single: RF −0.002 [−0.011, 0.007]; CNN −0.020 [−0.047, 0.006]. Neither is significant.
  - The conclusion that the disagreement is systematic still holds.
- **Unexplained difference:** the "single run" value here (0.22) differs from the 0.24 in Table 5. The reason is that this value comes from the 100-record stability subsample (10 per class), while Table 5 uses the 300-record agreement sample. The text does not say so.
- **Fix:** "Averaging 20 LIME runs does not bring LIME closer to SHAP (… paired difference −0.002 [−0.011, 0.007] for the RF and −0.020 [−0.047, 0.006] for the CNN, on the stability subsample)".
- **Source:** stability_c.json, `paired_consensus_minus_single`.

## OVERSTATED

**O1. UNSW-NB15 attacker-controllable evasion "more than on TON-IoT for both models".** Section 5.3:
> "43% (RF; range over seeds 41–43%) and 30% (CNN; 21–30%) of detected attacks remain evadable, more than on TON-IoT for both models."

- **RF holds in seed 42 only:**

  | Seed | TON-IoT RF | UNSW-NB15 RF |
  |---|---|---|
  | 42 | 30.7% | 42.5% |
  | 43 | 40.7% | 40.6% (lower) |
  | 44 | 44.4% | 41.7% (lower) |

  The TON-IoT range over seeds (31–44%) contains the whole UNSW-NB15 range.
- **CNN holds in every seed:** UNSW-NB15 21–30% against TON-IoT 4–18%.
- verify.py itself tests the RF claim as "UNSW RF attacker success about as on TON" (line 160), which contradicts the text.
- **Fix:** "…more than on TON-IoT for the CNN in every seed; for the RF only in seed 42 (similar over seeds: 41–43% against 31–44%)".
- **Source:** results_v3/*/seed4x/counterfactual.json, `rf.attacker.success`.

**O2. UNSW-NB15 hardening gain credited "only" to the evasion criterion.** This appears in three places:
- Section 5.4: "Here only the evasion criterion, which selects the few copies that cross the decision boundary, makes the augmentation effective."
- Section 1, contribution 4: "…and from the evasion criterion itself on UNSW-NB15."
- Section 6.1: "on UNSW-NB15 only the evasions themselves helped."

What the data show:
- The means are consistent: 32% (counterfactuals, range 30–35) against 41% (benign-donor copies, range 39–42), and the ranges do not overlap.
- Per seed, however, the counterfactual model's 95% CI lies below the benign-donor control's in **1 of 3** seeds, and below the original model and interpolation control in 2 of 3. The manuscript reports both counts.
- In seed 42 the counterfactual arm is 35 [30, 39] and the benign-donor arm 42 [37, 46], so the intervals overlap.
- The direction is supported; "only" is not.

**Fix:** use "mainly" or "most of the gain", or add "(consistent in the means over seeds, but separated from the benign-donor control in only 1 of 3 seeds)". Source: hardening_attacker.json for seeds 42–44, values.json `UNSW_hard_att_sep_copy` = 1 and `UNSW_hard_att_counterfactual_ev42` = "35 [30, 39]".

**O3. "SHAP beats LIME in deletion tests under all four replacement strategies".** This is Highlight 4 (build.py). The abstract has a similar sentence: "SHAP's top features lower model confidence more than LIME's."

- **Holds for the RF:** SHAP − LIME is significant in 36/36 combinations on both datasets.
- **Weaker for the CNN:** significant in 28/36 (TON-IoT) and 31/36 (UNSW-NB15); the point estimate is positive in 35/36.
- **Example:** UNSW-NB15 CNN, mean replacement, k = 3, seed 42: 0.02 [−0.01, 0.05], not significant.
- Section 5.2 states this correctly. The highlight and abstract read as unconditional.
- **Fix for the highlight:** "SHAP's top features beat LIME's in deletion tests under four replacement strategies" (drop "all"). Alternatively: "…in every RF test and most CNN tests".
- **Source:** deletion_c.json, recounted.

**O4. "With features fixed, the identifier still raised accuracy without any duplicates".** This is Highlight 2.

- **Duplicates are not fully gone:** only exact duplicates on the cleaned columns are removed. 16.3% of UNSW-NB15 test records still match a training record's features in the 10-feature space, and 13.7% match it with the label too (Section 3.3).
- **No interval for accuracy:** no confidence interval is reported for the accuracy effect, only the seed range 0.050–0.052. Macro-F1 has a bootstrap interval.
- **Fix:** "…without exact duplicates" or "…after deduplication". Consider adding a paired bootstrap interval for the accuracy effect.

## AMBIGUOUS

**A1. The headline accuracy effect mixes conventions.** Abstract and Section 3.3:
> "the identifier raised RF accuracy by 0.053 with duplicates and 0.051 without them"
> "`id` raises it by 0.053 with duplicates and by 0.051 without them (0.053 and 0.050–0.052 over three RF seeds)"

- The 0.053 and 0.051 values are **means over the three seeds** (values.json `decf_*_acc_effect_mean`; compute_values.py line 395).
- The macro-F1 effects in the same sentences (0.031 and 0.007) are seed-42 values.
- A reader who works it out from Table 1 (seed-42 accuracy) gets 0.8499 − 0.8002 = **0.0497 ≈ 0.050**, not 0.051.
- **Fix:** label 0.051 as the mean over seeds, or quote the seed-42 value of 0.050 to match Table 1.

**A2. Section 5.4 mixes the two datasets in one sentence.**
> "This arm is degenerate by construction, since on UNSW-NB15 four of the five groups cover almost the whole record: copying a whole protocol state from a benign flow makes a record benign in every feature the model sees"

- "Protocol state" is a TON-IoT group. The UNSW-NB15 groups are ttl, src_volume, dst_volume, tcp_timing and conn_history.
- The sentence also explains TON-IoT with a UNSW-NB15 argument.
- **Fix:** split it. On UNSW-NB15, four of the five groups cover 8–9 of the 10 features. On TON-IoT, the protocol-state group alone holds 16 of the 25 features.
- **Source:** counterfactual.json `groups`.

**A3. The ratio for conditional replacement covers SHAP only.** Section 5.2, Faithfulness:
> "Conditional replacement produces much smaller drops (0.22–0.55 times the mean-replacement drop at k = 3, depending on model and dataset)"

- 0.22–0.55 is the SHAP top-3 ratio only (`d_cond_ratio` uses `condx_shap_3 / mean_shap_3`).
- The ratio is 0.18–0.49 for LIME and 0.09–0.43 for random features.
- **Fix:** add "for SHAP's top features".

**A4. Reselecting features gives "similar macro-F1 effects (0.034 and 0.015)".** Section 3.3.

- With features fixed, the effects are 0.031 and 0.007, so the duplicates-removed effect roughly doubles under reselection (0.015 against 0.007).
- No interval is given for 0.015, so the reader cannot tell whether it still includes zero.
- **Fix:** "similar with duplicates kept (0.034) and larger without them (0.015), where reselection also changes other features". Alternatively, report the interval.

**A5. Excluding misclassified records "leaves it unchanged".** Section 5.2:
> "excluding misclassified records (35% of the UNSW-NB15 RF sample) leaves it unchanged (UNSW-NB15 RF: 0.41 [0.39, 0.43])"

- All records: 0.386 [0.369, 0.403]. Correctly classified only: 0.407 [0.386, 0.429].
- That is a +0.02 shift with barely overlapping intervals.
- **Fix:** "changes it little" or "slightly raises it".
- **Source:** local_c.json, `summary_correct_only`.

**A6. How firmly the TTL shortcut is stated differs between sections.**
- Abstract: "The corrected models expose a second shortcut, the source TTL."
- Section 5.1: "Section 5.5 shows that this is a second shortcut."
- Section 5.5 concedes: "the evidence that the TTL is a shortcut rests on its SHAP rank and on the majority rule rather than on a dependence of performance on it."
  - Removing the TTL fields costs the RF 0.030 macro-F1, 79% of it from worms, and the CNN 0.003.
- The evidence (value 31 occurs only in benign records; the sttl rule reaches 88.7% binary accuracy) supports calling it a shortcut the data offer. It is weaker evidence that the models depend on it.
- **Fix:** soften Section 5.1 to "Section 5.5 examines whether this is a second shortcut", or add "available" in the abstract.

## OK-but-note

**N1. The PDFs are stale.** Manuscript_v2.pdf and Manuscript_v2_anon.pdf date from 2026-10-31; the docx files from 2026-11-07. A sentence-by-sentence comparison finds one real content difference: the PDF lacks "The analysis code was written with the assistance of Claude (Anthropic) and checked by the authors." in the data-availability statement. All numbers checked in the abstract and Section 5.4 are the same. Regenerate the PDFs if they will be uploaded.

**N2. Highlight 5 is missing "from".** The filled text reads "Counterfactual retraining cut constrained evasion 41% to 16% on TON-IoT, at equal F1" (84 characters). Adding "from" pushes it past the 85-character limit. Suggestion: "Counterfactual retraining cut constrained evasion from 41% to 16% on TON-IoT" (76 characters). If the F1 point should stay, change "at equal F1" to "F1 unchanged" and shorten elsewhere.

**N3. The 95% rank intervals bootstrap records only, not explainer randomness.** This affects Tables 3 and 4 and Section 4.2. All UNSW-NB15 CNN rank intervals are single ranks, even though GradientExplainer's local self-agreement is only 0.63. Averaging over 1,000 records likely smooths out sampling noise, but "how firm the ranking is" refers to resampling records only. One clause would make this explicit.

**N4. The TON-IoT "exact coincidences are rare (0.1%)" figure depends on the definition.** feature_space_copies.json also reports 17.1% at 8-decimal rounding and 24.9% at 6-decimal rounding. Rounding merges distinct small scaled byte counts, so exact float32 equality is the right definition. A reviewer browsing the repository may still ask, so a footnote or repository note could help.

**N5. The original-model evasion for TON-IoT is 31% in Section 5.3 and 41% in Section 5.4.** Section 4.7 explains the difference (different query sets and donor pools). This is fine, but readers comparing the abstract with Table 7 may notice it.

**N6. The abstract and Section 5.2 say LIME at default settings "is unstable" in general.** On UNSW-NB15 the RF's LIME self-agreement is 0.82 and 75% of records reach 0.75. The claim holds for TON-IoT and for the CNN, less so for the UNSW-NB15 RF.

**N7. "Both explainers name features that matter to the model" (Section 5.2) is qualified a few sentences later, but not where it is made.** For the UNSW-NB15 RF, LIME's top features beat random ones in only 18/36 combinations, and under mean replacement they score 0.40 against 0.40.

**N8. The Fig. 4 caption says "e.g. worms", but worms is the only class with fewer than 20 query records per seed** (13 per seed; every other class has 57–60). "(worms)" would be more precise.

**N9. "On TON-IoT, explanations are therefore useful to an attacker mainly as a way to save queries" holds for the CNN only.** For the RF, the SHAP-guided median number of queries is 203 against 202 for random search, and success is lower (87%). Consider "for the CNN" in that sentence.

---

## Counts

| Class | Count |
|---|---|
| WRONG | 2 (W1, W2) |
| OVERSTATED | 4 (O1–O4) |
| AMBIGUOUS | 6 (A1–A6) |
| OK-but-note | 9 (N1–N9) |
| **Total findings** | **21** |

No number in the text, tables, abstract or highlights disagrees with values.json. Every raw-file spot check reproduced values.json. The problems are interpretation, wording and generalisation, not data errors.
