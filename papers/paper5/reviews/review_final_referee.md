# Referee report (final round, fresh read): Computers & Security

**Manuscript:** "When SHAP Finds the Leak: Auditing Agreement, Faithfulness and Feasible Counterfactuals in Network Intrusion Detection"
**Version reviewed:** Manuscript_filled.md (build of 2026-11-07), figures/fig1–fig4, captions in build.py
**Reviewer stance:** demanding referee, fresh read; earlier round-2 points checked at the end (Section 7). No experiments were re-run.

---

## 1. Summary

The paper audits post-hoc explanations of two IDS models (200-tree RF; 24k-parameter 1D-CNN) on TON-IoT and UNSW-NB15, with three training seeds. It has four parts:

1. **Audit case study (§3.3, Table 1).** In an earlier version of the authors' pipeline, global SHAP ranked the UNSW-NB15 row `id` first. The identifier encodes block-ordered file position. Because it made every row unique, it also silently disabled duplicate removal (38.6% of records are redundant copies). With the feature set fixed, `id` raises RF accuracy by about 0.05 with or without duplicates, but raises macro-F1 significantly only when duplicates are present. The corrected models then lean on the source TTL (§5.5).
2. **Local explanation quality (§5.2, Tables 5–6).** SHAP and LIME share about 2–3 of their top-5 features for the same alert. Chance-corrected Jaccard is 0.05–0.22, so the UNSW RF is close to chance. LIME at 1,000 samples is unstable, and so is CNN GradientExplainer at its default 200 samples. In deletion tests with four replacement strategies, including a conditional one, SHAP's top features lower model confidence more than LIME's.
3. **Feasible counterfactuals (§5.3, Tables 7–8).** Benign-donor group copying with a support check evades every detected attack when any group may change. When only attacker-controllable groups may change, 16–43% of detected attacks are evadable.
4. **Hardening (§5.4, Table 9).** Under the attacker-controllable model, counterfactual retraining cuts RF evasion from 41% to 16% (TON-IoT) and from 43% to 32% (UNSW-NB15). An interpolation control does nothing. A benign-donor control without the evasion criterion is almost as effective on TON-IoT but not on UNSW-NB15.

The paper is carefully written, unusually self-critical, and much more rigorous than the typical IDS-XAI paper. My remaining concerns are about (i) whether some headline claims follow from the statistics as reported, (ii) the interpretation of counterfactual "evasion" under a very tight support radius, (iii) residual leakage in the "corrected" data, and (iv) the size of the novelty claim.

## 2. Score and recommendation

**Score: 6.5 / 10.**
**Recommendation: Major revision (narrow scope).**

Most of the work needed is re-analysis of saved outputs and recalibration of claims. Only issues M5 and M6 need small new runs. If these are addressed, I would expect to recommend acceptance.

---

## 3. Major issues

### M1. The UNSW-NB15 hardening claim and contribution 4 over-interpret overlapping intervals

- **Location:** §1 contribution 4 ("...and from the evasion criterion itself on UNSW-NB15"); §5.4 ("Here only the evasion criterion ... makes the augmentation effective"); §6.1 ("on UNSW-NB15 only the evasions themselves helped"); Table 9.
- **Problem:** §5.4 itself reports that the counterfactual model's 95% interval lies below the benign-donor control's interval in only 1 of 3 seeds on UNSW-NB15, and in 0 of 3 on TON-IoT. A mean difference of 32% vs 41%, separated in one seed of three, does not show that "only the evasion criterion" matters.
- **Why it matters:** This is the only result that separates counterfactual search from plain benign-value augmentation. It is the security-relevant core of contribution 4.
- **CI overlap is also the wrong test.** All models are evaluated on the same fixed query set per seed (§4.7), so the outcomes are paired. Non-overlapping CIs is a conservative and non-standard criterion.
- **Fix:**
  - Report paired tests per seed on the shared query records (McNemar, or a paired bootstrap of the evasion difference), for counterfactuals vs benign-donor and counterfactuals vs original.
  - Pool over seeds, e.g. with a stratified Cochran–Mantel–Haenszel test or a seed×record bootstrap.
  - Rephrase contribution 4, §5.4 and §6.1 according to the result. If the difference is not robust, say "on UNSW-NB15 the counterfactuals helped more than either control in the mean, but the difference from benign-donor copies was significant in only one seed".

### M2. Counterfactual "evasion" under a tight support radius may partly measure class overlap, and its local cost is unmeasured

- **Location:** §4.6 (support radius = 95th percentile of benign nearest-neighbour distances; "This radius is small, so accepted counterfactuals lie close to real benign flows"); §5.3; §5.4; Table 9 FPR column.
- **How the support check constrains evasion.** An accepted candidate must lie within the benign-to-benign nearest-neighbour radius of some benign record in all features.
  - In the attacker-controllable arm, the features the attacker cannot change are left as they were. So an attack can only evade if its uncontrolled features already coincide, within that radius, with some benign flow.
  - "Evadable" therefore partly identifies attacks that are already nearly indistinguishable from benign traffic in the non-controllable part of the record. That is class overlap, not a model weakness in the usual sense.
- **Same argument for the all-groups arm.** The 100% figure there is close to tautological. Copying up to 4 of 5–6 groups and then requiring proximity to a benign record yields a near-benign record. The paper concedes this for UNSW-NB15 (§4.6, §5.4), but the abstract still headlines it ("evade every model when any field may change").
- **The local cost of hardening is not measured.** Hardening adds hundreds of records labelled *attack* that lie within the benign support radius. The natural cost is false positives on the benign records next to them. The aggregate benign FPR over roughly 15k benign test flows (0.79% vs 0.82%) cannot reveal a local cost. This was requested in round 2 (M3a) and has not been done.
- **Fix:**
  - Report the benign FPR of each retrained model on benign test records that lie within τ (or within the k nearest neighbours) of the augmented counterfactuals and of the benign-donor copies.
  - Report, per class, the share of evadable attacks whose *non-controllable* features are already within τ of a benign training record before any change.
  - Discuss what "evasion" means under this constraint.
  - In the abstract, drop the all-groups 100% headline or label it explicitly as an expected-by-construction upper bound.

### M3. The "only with duplicates" macro-F1 conclusion compares significance across conditions and test populations

- **Location:** Abstract ("macro-F1 significantly only while duplicates were present"); §1 contribution 1; §3.3; §6.1; §7; Table 1.
- **Problem 1: no interaction test.** The `id` effect on macro-F1 is 0.031 [0.016, 0.046] with duplicates and 0.007 [−0.006, 0.020] without. The two intervals overlap. "Significant in one condition, not in the other" is not a test of the interaction; this is the difference-in-significance fallacy.
- **Problem 2: different test populations.** The two pairs are evaluated on different test sets: split 1 has 26,302 records and split 2 has 15,894. Their class compositions also differ sharply (for example, *generic* shrinks from 40,000 to 4,181 records, §3.3). Because macro-F1 weights classes equally, the change in the `id` effect between pairs mixes "duplicates present" with "different class mix and class sizes in the test set".
- **Problem 3: accuracy has no interval.** Accuracy effects (0.053 / 0.051) are given without CIs, although they carry the abstract's first quantitative claim.
- **Fix:**
  - Give a bootstrap CI for the interaction, i.e. the difference of the two paired differences. Resampling is independent across the two splits, which is straightforward.
  - Add paired CIs for the accuracy effects.
  - Either evaluate both pairs on a common test population (e.g. split-2 test records that also occur in split 1's test set), or state explicitly that test composition differs.
  - Soften "only while duplicates were present" to "clearly while duplicates were present; without them the effect was small and its interval included zero".

### M4. Uncertainty is still reported at the wrong level for several headline numbers

- **Location:** §4.9; Tables 2, 5–9; §5.3; the abstract.
- **The CIs cover record-sampling variance for seed 42 only.** The text acknowledges this (§6.3), but the headline values are chosen inconsistently.
  - §5.3 headlines seed 42: RF attacker-controllable 31% (range 31–44%), at the bottom of its seed range; CNN 16% (range 4–18%), near the top.
  - §5.4 headlines means over seeds.
  - TON-IoT CNN macro-F1 varies 0.830–0.895 across seeds, far beyond any record-level interval.
  - The seed-42 CNN "was trained without a fixed TensorFlow seed" (§3.4), so the model behind every reported CNN CI cannot be reproduced exactly.
- **Bootstrap design.** It is not stated whether the record bootstrap respects the class-stratified sampling (30 per class). The means are class-balanced, not traffic-weighted, and this should be said in Tables 5–6.
- **Tables 7–8.** These are seed-42 per-class shares on at most 30 records per class, so each cell carries roughly ±17 points, and no interval is given.
- **Fix:**
  - Use one convention throughout: the mean over seeds as the headline, with a two-level (seed × record) bootstrap CI. This needs no new model runs.
  - State the stratified resampling.
  - Add Wilson intervals, or at least n per class, to Tables 7–8.
  - Either retrain the seed-42 CNN with a fixed seed or say plainly that its exact values are not reproducible.

### M5. The LIME results are specific to one configuration that is known to be fragile

- **Location:** §4.1; §5.2; §6.2; §7 ("SHAP's top features were the more faithful ... whether LIME was ranked by contribution or by raw weight").
- **Problem 1: one LIME configuration.** LIME is run in its continuous mode (no discretisation) with the default kernel width, which is not reported. In that mode LIME is notoriously sensitive to kernel width. The surrogate R² of 0.06 (TON CNN) and 0.12 (UNSW CNN) shows that the local linear model barely fits.
- **The ranking robustness check is a good addition, but configuration matters more than ranking.** The general conclusion that SHAP names more faithful features than LIME, and the practical advice in §6.2, rest on one LIME configuration whose fit is poor.
- **Problem 2: mean replacement is not neutral.** With contribution ranking, w·(x−μ)/σ is exactly the change in the surrogate when the feature is set to μ. Mean replacement is therefore arguably *LIME-favourable*, which strengthens the authors' case. The paper should say so instead of listing only Gaussian replacement as LIME's own distribution.
- **Fix:**
  - On the stability/agreement subsample, run LIME with at least two other kernel widths, and with its default quartile discretisation (which is LIME's actual default representation).
  - Report agreement, R² and the k = 3 deletion difference for each.
  - If SHAP still leads, the claim generalises. If not, restrict it to "LIME in continuous mode at the default kernel width".
  - Add the mean-replacement observation to §4.5.

### M6. The "corrected" UNSW-NB15 data still contain substantial feature-space duplicates, and all downstream results use them

- **Location:** §3.2 (deduplication before feature selection); §3.3, final paragraph.
- **Size of the residue.** 13.7% of UNSW-NB15 test records are identical, label included, to a training record in the ten selected features. That includes 78% of *generic* and 53% of *analysis* test records. A further share match a training record with a different label. Excluding them moves RF macro-F1 by 0.04.
- **Why it matters.** The paper's message is that duplicates distort evaluation, yet the explanation-quality samples, the counterfactual query sets and the hardening query sets all draw from this test set. For *generic*, most explained records have an exact feature-space twin in training.
- **Fix:**
  - Repeat the key UNSW-NB15 explanation measures, excluding test records with a feature-space twin in training: agreement, the deletion difference at k = 3, and attacker-controllable evasion.
  - Alternatively, justify why deduplicating before feature selection is the right design, and show that the conclusions do not depend on it.
  - At minimum, flag in Tables 6 and 8 which classes are dominated by such records.

### M7. Claims about what explanations achieved are broader than the evidence

- **Location:** Abstract, last sentence ("Explanations are effective for auditing data and hardening detectors"); §7, last sentence; Highlights.
- **Auditing rests on one retrospective case.** §6.1 says so ("it shows that the audit can work, not how often it would"). Section 5.5 also concedes that, once suspected, the TTL shortcut is visible from simple statistics, and a schema check would have caught `id`. The `id` column was in the authors' own earlier pipeline, so the "audit" is a post-mortem of the authors' own bug. That is useful, but it is weak evidence that explanations are an *effective* audit tool.
- **Hardening used no attribution method.** The counterfactuals come from random group search (§6.1: "the gain should be credited to counterfactual analysis rather than to feature attributions"). On TON-IoT, the benign-donor control, which uses no explanation at all, achieved most of the gain.
- **Fix:** Change the abstract and conclusion to wording such as: "A global explanation exposed a leak in one case study, and counterfactual analysis identified which sender-controlled fields to augment." Do the same in the Highlights.

### M8. Novelty and positioning

- **Location:** §2.2 ("Each of the measures we use therefore has precedent"); §2.4; §3.4 ("the best model in our baseline runs of this pipeline ... (results in the released repository)"; "the seed-42 models are those of the baseline study").
- **What is new is the combination.** The contribution is the protocol, the `id`×deduplication interaction and the threat-model contrast. That can suffice for C&S, but the individual pieces are known:
  - `sttl` dominance in UNSW-NB15 is widely observed.
  - Disagreement between explainers is established (Krishna et al.; Kumar & Thing).
  - Evasion guided by explanations is established (Okada et al.; Pawlicki et al.).
- **TTL prior work.** The TTL finding is positioned against Beltiukov et al. (other network data). The authors should look for prior reports of `sttl`/TTL dominance specifically in UNSW-NB15, including the dataset authors' own statistical analyses, and present the finding as a confirmation.
- **Related submissions.** The paper refers to a "baseline study" whose models it reuses but which it does not cite. If this is a companion submission in the same series, the editor should be told, and the manuscript should cite it (or the repository release) and state its overlap with that work. Otherwise reviewers may question salami slicing.
- **Fix:**
  - Add a short "What is new" paragraph at the end of §2.
  - Cite or disclose the baseline study.
  - Position the TTL result against prior UNSW-NB15 observations.

---

## 4. Minor issues

1. **Abstract, "with four replacement strategies, including one matching neither explainer's sampling".** Fine, but for UNSW CNN under mean replacement the difference is 0.02 [−0.01, 0.05] (Table 6), which is not significant. §5.2 "SHAP's are the more influential under every replacement strategy" therefore needs "in the mean" or "with one non-significant exception per model".
2. **§5.2 Faithfulness: two different counts.** "for the CNN in 35 (TON-IoT) and 35 (UNSW-NB15) of 36 combinations" counts point estimates. "significant in ... 28 for the CNN on TON-IoT ... 31 on UNSW-NB15" counts significant results. Readers will conflate them; label each count explicitly.
3. **Templated prose.** Several sentences read like auto-filled placeholders: "in 36, 36, 36 and 36 of these combinations"; "100% for the RF and 100% for the CNN"; "100% and 100% for the two controls"; "both controls stay at 41% (39–42% and 39–42%)"; "protocol state (88%), dst port (27%) and src volume (18%)". Rewrite them as natural sentences (e.g. "in all 36 combinations for every model–dataset pair").
4. **§5.2 Stability, "changing the background as well lowers it only to 0.57 ... and 0.73".** These are 1,000-sample values, to be compared with 0.60 and 0.76, not with the 200-sample 0.50 in the previous clause. State the budget.
5. **§5.3, "Guiding the search by SHAP does not make evasion easier for the RF (87% success...)".** The threat model is unclear. If this is the all-groups arm, 87% is *lower* than random's 100%. Say so, and give the attacker-controllable SHAP-guided result too.
6. **§5.3, median queries.** Medians are over successful searches only, so SHAP-guided and random medians have different denominators (survivorship). Report the success-weighted expected queries, or the queries at a fixed success rate.
7. **§4.6, UNSW attacker-controllable TTL.** Changing `sttl` while leaving `ct_state_ttl` unchanged produces records inconsistent with the dataset's own derivation of `ct_state_ttl` from `sttl`, `dttl` and `state`. This conflicts with the feasibility premise. Either recompute `ct_state_ttl` with the dataset rule, or acknowledge the inconsistency (the support check only partly guards against it).
8. **Monotone volume constraint.** The TON-IoT attacker-controllable arm copies source volume and duration in both directions. A monotone (increase-only) variant was requested in round 2. §6.3 now acknowledges this as an upper bound, which is acceptable, but a one-line result for the monotone variant would make the 16–43% figures much more credible.
9. **§5.5, sttl majority rule.** "88.7% ... correctly as benign or attack" needs the *binary* majority baseline next to it. Only the ten-class baseline (49.0%) is given.
10. **§5.5, worms.** The worms drop (0.53 → 0.30) rests on very few test records. Give n.
11. **Table 6, "Top-5 Jaccard on fields".** The row is identical to the feature-level row, presumably because none of the ten UNSW features is one-hot. Say so in a note or drop the row.
12. **Table 9 caption vs text.** The caption says the macro-F1/FPR are means over seeds, while the text quotes "seed 42: 39 [35, 44]". Add seed-42 CIs to the table, or move them to a note, so readers can find them.
13. **§4.1, LIME settings.** State LIME's version, kernel width and kernel; state SHAP's `nsamples` for GradientExplainer explicitly.
14. **§4.3, chance baseline.** Uniform random top-5 sets are a floor. Two explainers of the same model will both favour globally important features. Add the agreement of each local top-5 with the model's global top-5 as a stronger baseline (requested in round 2, M4; still missing). Also report agreement normalised by the self-agreement ceiling, J / √(J_SHAP,self · J_LIME,self). §5.2 now argues this qualitatively for the CNN but gives no number.
15. **§3.4, "CNN ... class weights computed on the oversampled training labels".** After SMOTE these weights are near-uniform except for classes capped at 50,000. Clarify whether weighting has any effect.
16. **§3.4, model choice.** The 1D-CNN applies convolutions over an arbitrary feature order. One sentence acknowledging that locality is not meaningful here (or that the feature order is fixed) would pre-empt a common reviewer objection.
17. **§4.2, global importance.** It is averaged over classes for multiclass models. Say whether this is the mean over class-specific |SHAP| or |SHAP| of the predicted class, since this changes rankings.
18. **§6.2, recommendations.** "at least 5,000 samples" for LIME rests on TON-IoT RF reaching 0.81. For the CNN, 5,000 samples gave 0.62. Make the recommendation model-specific.
19. **Figure 4.** The TON-IoT ransomware bar shows benign-donor copies (about 61%) beating counterfactuals (about 72%), and MITM shows the opposite (41% vs 21%). One sentence in §5.4 noting this class-level heterogeneity would be honest and informative.
20. **Figure 3 caption.** "favours neither explainer" is slightly too strong. Conditional replacement is still a distribution choice; "matches neither explainer's sampling" (the wording of the abstract) is more accurate.
21. **Declarations.** The generative-AI statement says Claude was used "to write and edit the manuscript text" and the code. This is permitted by Elsevier, but the editor may ask how the numbers were verified. A sentence pointing to the automated value-filling and verification scripts in the repository would help.
22. **Title.** It is now accurate ("Network Intrusion Detection"). The keywords still list "Internet of Things"; keep it only if the IoT angle is defended in §1.

---

## 5. Things done well

1. **The audit narrative is genuinely instructive.** The `id`→no-deduplication interaction, the block-ordered file (Fig. 1A) and the counter-intuitive finding that copied test records were *harder* (0.7527 vs 0.8750 accuracy) are a valuable cautionary tale for the UNSW-NB15 community. The feature-fixed decomposition (Table 1) now isolates `id` properly.
2. **The local evaluation is unusually rigorous for this literature.** It compares the same record and the same predicted class. It reports chance-corrected Jaccard (the J₀ values of 0.12 and 0.35 check out under the hypergeometric distribution), agreement on fields, self-agreement for *both* explainers, a consensus-LIME check, separation of GradientExplainer background and Monte Carlo variance, and paired bootstrap differences.
3. **The deletion design is fair.** It uses four replacement strategies, explicitly argues which one each explainer would favour, includes a conditional strategy that matches neither explainer's sampling, uses a random-feature baseline, and checks robustness to LIME ranking. This is how faithfulness comparisons should be done.
4. **The threat-model contrast is the most useful practical result.** "Any field" gives degenerate, unhardenable evasion; "attacker-controllable" gives a smaller, partly hardenable surface. Together with the benign-donor control, this clarifies where the hardening gain comes from, and the authors report the deflating result honestly.
5. **Candour.** The limitations (§6.3), the admission that the TTL is visible from simple statistics, and the statement that the audit is n = 1 are all appropriate. The authors repeatedly decline to over-sell on points where many papers would.
6. **Reproducibility.** The public code and result files, and a manuscript whose numbers are filled from result JSONs, are exemplary.

---

## 6. Requested changes (checklist)

- [ ] Paired tests (McNemar or paired bootstrap) for hardening; rephrase the UNSW-NB15 "only the evasion criterion" claim (M1).
- [ ] Local FPR near augmented records; pre-existing benign proximity of evadable attacks; abstract wording for the all-groups arm (M2).
- [ ] Interaction CI for Table 1; accuracy CIs; common-test-population check or caveat (M3).
- [ ] Mean-over-seeds headline with a seed×record bootstrap; stratified resampling stated; per-class n/CIs in Tables 7–8; seed-42 CNN reproducibility (M4).
- [ ] LIME kernel-width and discretised variants on a subsample; mean-replacement note (M5).
- [ ] UNSW-NB15 key measures excluding feature-space twins (M6).
- [ ] Abstract, conclusion and Highlights recalibrated (M7).
- [ ] "What is new" paragraph; disclose or cite the baseline study; UNSW-specific TTL prior work (M8).

## 7. Status of round-2 major points

| Round-2 point | Status now |
|---|---|
| M1 deletion test built in SHAP's value function | **Fixed.** Gaussian and conditional strategies added; SHAP still leads. |
| M2 generous attacker-controllable model | **Partly fixed.** UNSW now uses `sttl` only, but `ct_state_ttl` is left inconsistent (minor 7). TON monotone variant not done, only acknowledged in §6.3. |
| M3a local cost / meaning of support-constrained evasion | **Not addressed** (now M2). |
| M3b weak control | **Fixed.** Benign-donor control added; it is informative and reported honestly. |
| M3c hardening uses no explanation | **Fixed in §6.1, not in abstract/conclusion** (M7). |
| M3d CIs and paired test for evasion | **Partly.** Seed-42 CIs given; CI-overlap counting instead of a paired test (M1). |
| M3e inconsistent evasion figures across §5.3/§5.4 | **Fixed** (explained in §4.7). |
| M4 weak chance baseline | **Partly.** Self-agreement ceiling discussed qualitatively; global-ranking baseline and normalised agreement missing (minor 14). |
| M5 CNN SHAP budget | **Fixed.** Budget and background separated. |
| M6 statistics level | **Not fixed** (now M4). Table 1 paired CIs added. |
| M7 decomposition did not isolate `id` | **Fixed.** Feature-fixed conditions. |
| M8 novelty / title / TTL prior work | **Partly.** Title fixed and "only explanations" claim removed; UNSW-specific TTL prior work and companion-study disclosure still needed (M8). |
