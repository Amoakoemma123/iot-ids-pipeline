# Referee report, round 2: Computers & Security

**Manuscript:** "When SHAP Finds the Leak: Auditing Agreement, Faithfulness and Feasible Counterfactuals in IoT Intrusion Detection" (revised)

**Date:** 2026-10-24

**What I read:**
- `Manuscript_filled.md` and figures 1–4 (PNG)
- round-1 reports `review_methods_code.md` and `review_numbers.md`
- `run_xai_v3.py` and `src/data/loader.py`
- the result JSONs in `results_v3/` (per-seed `local`, `hardening`, `hardening_attacker`, and `leakage_decomposition`) and `values.json`

**Not done:** I trained no models and loaded no models.

**Round-1 verdict:** major revision, 5/10.

---

## Part A. Status of the round-1 points

| # | Round-1 point | Status | Reason |
|---|---|---|---|
| 1 | Per-instance agreement misread ("fewer than half") | **Resolved** | The paper now reports shared features (2.70 / 2.07 / 2.99 / 2.63 of 5) and uses "two or three of their top five" in §5.2, §6.2, §7 and the abstract. |
| 2 | No chance correction | **Resolved, with a caveat** | The hypergeometric J₀ (0.12 TON, 0.35 UNSW) and (J−J₀)/(1−J₀) are now reported, and the UNSW CNN is correctly described as "barely above" chance. The random-set null is a weak baseline (see major concern M4). |
| 3 | CNN SHAP wrongly called deterministic | **Resolved, partly** | `rseed` is now passed. CNN SHAP self-agreement is measured (0.48 TON, 0.63 UNSW) and the text says plainly that it is unstable. However, GradientExplainer still runs at its default `nsamples=200`, so the instability is partly an estimator-budget artefact that the paper presents as a property of SHAP (M5). |
| 4 | LIME sample-size sensitivity untested | **Resolved** | The paper adds 1,000 vs 5,000 samples, consensus LIME over 20 runs, and the share of records at 0.75 or above. The LIME sampling distribution (centred on the training mean, `sample_around_instance=False`) is still not stated (m3). |
| 5 | Single deletion strategy | **Partly resolved** | A second strategy (values from random training records) is added, with paired CIs. But this strategy is exactly the value function of interventional SHAP with a training background, so it favours SHAP even more than mean replacement does (M1). |
| 6 | Infeasible counterfactuals | **Largely resolved** | Dependency groups are now copied jointly from one donor, and a nearest-neighbour support check is added. New issues: the attacker-controllable UNSW "ttl" group includes `dttl` (the responder's TTL), and the support radius is so tight that accepted counterfactuals are almost copies of benign records (M2, M3). |
| 7 | Hardening: no control, inconsistent query sets, no benign FPR | **Partly resolved** | There is now a control, one fixed jointly-detected query set, disjoint donor pools, benign FPR and three seeds. Remaining problems: the control (convex interpolation) is weak, the evaluation RNG streams still diverge between models, and the bootstrap CIs on evasion that are computed are not reported (M3). |
| 8 | Single seed | **Partly resolved** | There are three training seeds. However, all CIs come from seed 42 only, and the decomposition (Table 1), global importance, stability and the TTL/port retrains are single-seed (M6). |
| 9 | No decomposition of id vs duplicate effects | **Resolved, with a confound** | Table 1 is a good addition. But the pipeline re-runs feature selection in each condition, so A and B differ in six features besides `id`. Condition A has no `ct_state_ttl` but has `dur`, `rate`, `dmean`, `ct_srv_src` and others (`leakage_decomposition.json`). "Removing it alone" (abstract) is therefore not accurate (M7). |
| 10 | TTL removed after feature selection | **Resolved** | TTL fields are now removed before selection. The ten replacement features are listed, and the RF and CNN are retrained. |
| 11 | "Lightweight" unsupported | **Resolved** | Table 2 reports size, node and parameter counts and latency, and the text concedes that the RF (97 / 512 MB) is not suitable for constrained devices. |
| 12 | Audit not framed as a case study | **Resolved** | §3.3 is titled as a case study, and §6.1 says "This is one case study; it shows that the audit can work, not how often it would." |
| 13 | Uncited "original analysis" | **Resolved, partly** | It is now an "earlier, unpublished version of our pipeline", with the v1 pooled 0.67 explained. The paper still compares against unpublished numbers in several places (§4.3, §4.6, §5.2), which a reader cannot check. |
| 14 | Figures not cited | **Resolved** | Figures 1A/B, 2, 3 and 4 are all cited in the text. |
| 15 | (number audit) Conflict count of 4,845 was wrong | **Resolved** | Now 8,472, which matches the recomputation. |
| 16 | (code audit) Residual duplicates in model space | **Partly resolved** | Deduplication now runs after the column drops, and a cross-split guard has been added. The guard compares rows on the post-OHE columns, not on the 10 selected features, so test records whose 10-feature vector has a copy in training remain. Round 1 measured 12.6% of such test records on the earlier 11-feature set. Please report this overlap (m8). |
| 17 | (code audit) Rank intervals truncated by `int(np.percentile)` | **Unresolved** | The code is unchanged (`run_xai_v2.py` L220–221). |
| 18 | (code audit) Interaction values use path-dependent TreeSHAP | **Unresolved (disclosure)** | §4.2 still does not say so. |
| 19 | (code audit) A third of explained UNSW records are misclassified | **Unresolved (disclosure)** | 188/289 RF and 193/289 CNN records are correct, so about 35% are wrong. The code computes `summary_correct_only`, but the paper does not report it. |

**Overall.** This revision is substantive and responsive. Most of the round-1 headline problems have been fixed, and the paper is now much more honest about what explanations can and cannot do. The remaining problems are narrower but still matter for the central claims about faithfulness and hardening.

---

## Part B. Fresh referee report

### B1. Summary

The paper audits SHAP and LIME explanations for a Random Forest and a 23.9k-parameter 1D-CNN trained on TON-IoT and UNSW-NB15, with three training seeds. It makes four claims:

1. **Data audit.** In an earlier version of the authors' pipeline, global SHAP exposed the UNSW-NB15 row `id` as a leak through file order. Because `id` made every row unique, it had also silently disabled deduplication (38.6% duplicates). A four-condition decomposition (Table 1) attributes the inflation to `id` rather than to the duplicates. The corrected models then rely on a second shortcut, the source TTL.
2. **Local explanations.** For the same alert, SHAP and LIME share about 2–3 of their top-5 features. This is above chance on TON-IoT but near chance for the UNSW CNN. LIME at 1,000 samples is unstable, and so is CNN GradientExplainer. In deletion tests, SHAP's top features are more influential than LIME's.
3. **Counterfactuals.** Dependency-closed, support-constrained counterfactuals evade all models when any group may change. When only attacker-controllable groups may change, far fewer attacks evade on TON-IoT (RF 31–44%).
4. **Hardening.** Under the attacker-controllable threat model, retraining on counterfactuals cuts RF evasion from 41% to 15% (TON) and from 75% to 46% (UNSW), at unchanged macro-F1 and FPR. An equal-size interpolation control has no effect.

### B2. Strengths

1. **The audit narrative is valuable and well documented.** The `id` / deduplication interaction is a realistic, instructive failure:
   - "With `id` present every record was unique, so the deduplication step removed nothing".
   - Figure 1A shows the block ordering convincingly.
   - The counter-intuitive finding that test records with a training copy were classified *less* accurately (0.7527 vs 0.8750 accuracy) is interesting and correctly explained by label conflicts.
2. **The local evaluation is much more careful than is typical in IDS-XAI papers.** It uses same-record comparison, chance correction, field-level merging, self-agreement for *both* explainers, a consensus-LIME check, paired bootstrap differences, and two replacement strategies.
3. **The finding about consensus LIME is useful.** "Averaging 20 LIME runs barely brings LIME closer to SHAP" (consensus gain 0.02 [−0.01, 0.04]) shows the disagreement is systematic rather than sampling noise.
4. **The threat-model contrast is the paper's most important practical result.** "Any group may change" gives 100% evasion that no retraining closes. "Attacker-controllable" gives a smaller, hardenable surface. The hardening experiment now has a control, a fixed jointly-detected query set (538–540 TON, 481–488 UNSW records per seed), disjoint donor pools, and benign FPR.
5. **The limitations section is candid and specific.** It covers testbed data, sampled k as an upper bound, judgement-based groups, and the same search for generation and evaluation.
6. **The work is reproducible.** Every placeholder is generated from result JSONs, and I could verify the numbers I checked (see B5).

### B3. Major concerns

**M1. The deletion test is constructed in SHAP's own value function, so "SHAP is more faithful than LIME" is close to built in.**

- §4.5 introduces the second strategy to escape mean replacement: "the values of a random training record (averaged over 10 records drawn from 5,000)".
- Interventional TreeSHAP with a background of 200 random training records *defines* a feature's contribution by replacing the "absent" features with values from random background records. The random-record deletion test therefore measures, almost exactly, the quantity SHAP's attributions estimate. Mean replacement is close to the same reference for a model that is locally near-linear.
- LIME, by contrast, fits a surrogate to perturbations drawn from N(train mean, train sd) for every feature (`LimeTabularExplainer(..., discretize_continuous=False)`, with the default `sample_around_instance=False`). It even gives one-hot columns continuous values.
- The statement "For the RF, the paired SHAP − LIME difference is positive with an interval excluding zero in all 18 combinations" is correct. But it shows that SHAP is more faithful *to a SHAP-shaped removal operator*. It does not show that SHAP is more faithful in general.

**Requested:**
- (a) Evaluate each explainer also under its own perturbation distribution, i.e. replace with draws from LIME's Gaussian. If SHAP still wins there, the claim becomes much stronger.
- (b) Add a removal operator that neither explainer uses, for example a class-conditional donor (benign donors for attack records) or a k-NN on-manifold imputation.
- (c) Optionally, add ROAR-style remove-and-retrain on a subset (Hooker et al., 2019).
- At minimum, state the alignment explicitly in §4.5 and §6.2, and soften "it was both stable and the most faithful in the deletion test".

**M2. The attacker-controllable threat model is more generous than the text states.**

- §4.6 says the attacker-controllable groups are "ttl and src volume" on UNSW-NB15, fields "an attacker can set without changing the attack itself".
- In the code, `DEPENDENCY_GROUPS['UNSW_NB15']['ttl'] = ('sttl', 'dttl', 'ct_state_ttl')`. So the attacker also sets the **responder's** TTL `dttl` and the derived `ct_state_ttl`. The code comment itself calls this "a generous reading"; the paper does not.
- An attacker cannot choose the victim's reply TTL. Since `ct_state_ttl` is a function of `sttl`, `dttl` and `state`, copying the whole block from a benign donor gives the attacker control over the very shortcut the paper identifies.
- The headline "64% (RF) and 68% (CNN) of detected attacks evadable" and the hardening floor of 46% are therefore overestimates of what a sender can do.
- Similarly, "src volume" on TON (`src_bytes`, `src_pkts`, `src_ip_bytes`) and "duration" are copied from a benign donor in *both directions*. Padding (increasing bytes and duration) is plausible. *Reducing* the source volume of a DDoS, DoS or password brute-force to a benign flow's volume generally defeats the attack.

**Requested:**
- (a) Restrict the UNSW controllable TTL to `sttl`, with `ct_state_ttl` recomputed from the dataset's rule using the record's own `dttl` and `state`. If that is not possible, report results with and without `dttl`.
- (b) Add a monotone variant (volume and duration may only increase) for TON.
- (c) Report how the 31–44% and 41% → 15% figures change.
- Until then, the sentence "The TTL is a particularly weak point, because the sender sets it" needs a qualification.

**M3. What a successful counterfactual means under the support check, and what the hardening control rules out.**

*(a) Accepted counterfactuals are near-copies of benign records.*
- The support radius is the 95th percentile of benign-to-benign nearest-neighbour distances: τ = 0.0025 on TON and 0.0187 on UNSW in [0,1]-scaled space (`values.json`).
- 89–94% of TON candidates are rejected as infeasible.
- An accepted counterfactual therefore lies essentially *on top of* an existing benign training record in all features, including those the attacker did not change.
- In that case "evasion" means that the attack, after the change, is feature-indistinguishable from real benign traffic. Classifying it as benign is not a model weakness that more training can fix without false positives on its benign neighbours. This also explains why "any group" hardening stays at 100%.
- Please discuss this, and report the FPR **on the benign records that lie within τ of the augmented counterfactuals** (or on benign test records near them), not just the aggregate benign FPR. An aggregate FPR of 0.81% vs 0.82% over about 15k benign test flows cannot show a local cost.

*(b) The control is weak.*
- Convex combinations of two same-class training attacks add almost no new information to a model already trained on SMOTE output (which *is* convex interpolation). That such a control does nothing is expected.
- The informative ablation is the same augmentation without the model-in-the-loop selection: attack records with their attacker-controllable groups copied from random benign donors, filtered by the support check but *not* required to be classified benign.
- If that ablation hardens as well, the gain comes from data augmentation over the controllable fields, not from counterfactual search. Then the claims "explanations can also support hardening" (§6.1) and "Explanations are effective … for hardening detectors" (abstract) are unsupported.

*(c) The hardening pipeline uses no explanation.*
- The generator in `stage_hardening` is the *random* group search. SHAP is not used to generate the hardening data.
- The abstract and conclusion attribute hardening to "explanations". This is a framing problem regardless of (b). Call it counterfactual (adversarial) augmentation, or show that SHAP-guided generation is needed.

*(d) Reporting.*
- `evasion_ci` is computed for each seed but not reported. For example, TON seed 42 gives original 0.363 [0.322, 0.402] vs counterfactual 0.133 [0.106, 0.165]. Add these CIs and a McNemar test on paired outcomes to Table 9.
- `evaluate()` reseeds `erng` per model, but the candidate streams diverge after the first record whose search stops at a different k. The paired comparison is therefore only approximately paired, and the comment "same evaluation randomness for every model" overstates it. Use a per-record generator, e.g. `default_rng([seed, ti])`.

*(e) Inconsistent evasion figures for the same threat model.*
- Original-RF evasion under the attacker model is 31% (31–44%) in §5.3 but 41% (36–46%) in §5.4.
- Likewise UNSW: 64% vs 75%.
- The cause is presumably the different query sets (30 vs 60 per class, jointly detected) and the donor pool and support radius built on half the benign data. Please explain this in the text; at present a reader sees two different numbers for the same model and attack.

**M4. The chance baseline for agreement is too weak to support "agreement is well above chance".**

- Two explainers of the same model will both favour globally important features. A uniform random 5-subset is therefore a floor, not a meaningful null.
- Two informative baselines are cheap to add from saved data:
  - (i) the Jaccard between each explainer's local top-5 and the model's *global* top-5 (a constant ranking);
  - (ii) the ceiling set by self-agreement. Cross-method agreement cannot exceed what each method reproduces on its own.
- For the TON CNN, SHAP–LIME 0.28 sits against SHAP self-agreement 0.48 and LIME self-agreement 0.46. A large part of the "disagreement" is then noise within each method rather than conflict between them. That reading differs from the §6.2 message "SHAP and LIME typically share two or three of their top five features".
- Please report agreement normalised by the ceiling (e.g. J / √(J_SHAP,self · J_LIME,self)) and the global-ranking baseline. Revise the conclusion that the CNN disagreement is "systematic" if most of it is within-method noise. The consensus-LIME check addresses only LIME's noise, not SHAP's, which matters for the CNN.

**M5. The CNN SHAP instability depends on the estimator budget, and this is not tested.**

- `shap_cnn` calls `GradientExplainer.shap_values(X, rseed=…)` with the default `nsamples=200`. Expected-gradients estimates are Monte Carlo averages whose variance falls with `nsamples`.
- The self-agreement experiment varies the background and `rseed` together, so it cannot separate (a) sensitivity to the choice of background, which is a property of the explanation, from (b) Monte Carlo error at 200 samples, which is a property of the budget.
- "SHAP for the CNN is itself unstable across backgrounds" (abstract) and "SHAP for the CNN is no more stable than LIME" (§6.2) may therefore be artefacts of the default budget, exactly as LIME at 1,000 samples was.
- **Requested:**
  - Repeat on the 100-record stability sample with `nsamples` ∈ {200, 1000, 5000}, varying the background and `rseed` separately.
  - Consider DeepExplainer or IntegratedGradients with many steps as a deterministic comparison.
  - The recommendation in §6.2 ("averaging SHAP over several backgrounds and seeds") should follow from this experiment.

**M6. The statistical design understates uncertainty, and several claims rest on one run.**

- "Means are reported with percentile bootstrap 95% confidence intervals over records … from the seed-42 models; ranges over seeds give the means for seeds 42, 43 and 44."
  - For several measures the seed range exceeds the CI width. Examples: TON CNN Jaccard 0.28 [0.26, 0.30] vs seeds 0.25–0.30; TON CNN shared features 2.07 [1.96, 2.19] vs 1.79–2.23; TON CNN ρ 0.44 [0.42, 0.47] vs 0.33–0.49.
  - The reported CIs therefore do not cover model-retraining variance.
  - Use a two-level (seed × record) bootstrap, or report the mean over seeds with a CI that pools both sources. The paper has the per-seed rows, so this needs no new computation.
- The record bootstrap ignores the class-stratified design. Resample within class, and state that the means are class-balanced (30 per class), not traffic-weighted.
- Single-run claims:
  - Table 1: the identifier effects of 0.034 and 0.015 and the duplicate effect of 0.006 have no CI. A and B, and C and D, share test records, so a paired bootstrap of the macro-F1 difference is straightforward and should be given.
  - The TTL and port retrains: "worms 0.53 → 0.30" rests on about 19 test records.
  - Global importance and stability.
  - Either add seeds or CIs, or soften the wording.
- Counterfactual evasion shares in Tables 7–8 have no intervals; with n = 30 per class each cell has a CI of about ±17 points. Figure 4 shows bars on per-class values; the caption should say whether these are seed ranges, and should note that UNSW worms has n = 13.

**M7. The decomposition does not isolate `id`.**

- §3.3 says "we trained the RF under four conditions with otherwise identical settings", and the abstract says "removing it alone lowers RF macro-F1 by 0.034".
- In `leakage_decomposition.json`, MI and RFECV feature selection is re-run in each condition:

  | Condition | Features |
  |---|---|
  | A | 16: `id`, `dur`, `rate`, `dmean`, `ct_srv_src`, `ct_src_dport_ltm`, `service_dns`, … (no `ct_state_ttl`) |
  | B | 10: includes `ct_state_ttl` and `rate`, not `dur` |
  | C | 10: includes `id` and `sinpkt` |
  | D | 10 |

- The A-vs-B effect is therefore "`id` + a different feature subset". It is not implausible that most of it is `id`, given its rank-1 SHAP, but "alone" is not established.
- **Requested:** a fifth condition that drops `id` from A's own selected feature set without re-selecting (A minus `id`), and likewise C minus `id`, both with paired CIs. Alternatively, rephrase as "removing `id` and re-running feature selection".

**M8. Novelty and fit for C&S.**

- The paper itself states that "each of the measures we use therefore has precedent" (§2.2). The contribution is the combination, the audit narrative and the threat-model contrast. That can be enough for C&S, but only if the hardening and faithfulness claims (M1–M3) survive the stronger controls. As written, the security-relevant contribution (hardening) leans on a weak control and a generous threat model.
- The TTL shortcut in UNSW-NB15 is, to my knowledge, not new: `sttl` dominating UNSW-NB15 importance rankings is widely observed, and TTL-based host artefacts are discussed in the benchmark-critique literature.
  - §2.4 says "host-identifying TTL values in UNSW-NB15 are part of this picture" without citing who showed it. Please cite prior reports (check Flood et al., 2024, and earlier UNSW-NB15 feature studies) and position the TTL finding as confirmation.
  - "only the explanations made either visible" (§5.5) should be removed. The authors' own univariate `sttl` rule (88.7% binary accuracy) exposes the TTL shortcut without any explainer, and a schema check would have exposed `id`.
- Only one of the two datasets is IoT, yet the title says "IoT Intrusion Detection". Adding one IoT-native dataset (CICIoT2023 or Edge-IIoTset) would considerably strengthen the external validity of the explanation-quality findings. Otherwise, change the title to "network intrusion detection".

### B4. Minor concerns

- **m1. Factual inconsistency in §5.2 Faithfulness.**
  - The text says: "for the CNN LIME's top features beat random ones significantly in only 12 of 18 … LIME's do too, except for the UNSW-NB15 CNN, where under mean replacement they are no more influential than random features in three of nine combinations."
  - From `results_v3/*/local.json`:
    - The UNSW CNN failures are 4 of 9 under mean replacement (seeds 43 and 44, k = 3 and 5) and 2 of 9 under random-record replacement (seeds 43 and 44, k = 5). That is 6 of 18 in total, matching "12 of 18" but not "three of nine under mean replacement".
    - The TON CNN also has one non-significant LIME-vs-random cell (seed 43, k = 5, mean replacement; `lime_gt_rand_sig = 17`). That contradicts "LIME's do too, except for the UNSW-NB15 CNN".
  - Please correct both statements.
- **m2.** "Averaging 20 LIME runs … (RF on TON-IoT: 0.39 [0.36, 0.42] against 0.38 [0.34, 0.41] for a single run)". The single-run value comes from the stability sample (seed 43, 10 per class), not from Table 5 (0.39 [0.37, 0.41]). Say so, or a reader will think the numbers conflict.
- **m3.** State the LIME settings in full:
  - samples are drawn around the training mean (`sample_around_instance=False`), so the surrogate is not local to the instance;
  - no categorical handling for one-hot columns;
  - kernel width is the default.
  This bears on both agreement and the low CNN R² (0.06).
- **m4.** Raw LIME self-agreement is compared across datasets ("On UNSW-NB15 LIME self-agreement at 1,000 samples is 0.79 … and SHAP self-agreement 0.88"). Chance self-agreement differs (0.12 vs 0.35), so give chance-corrected values or a caveat. The same applies to the 0.75 threshold.
- **m5.** The 44.0% overlap: `share_test_seen = 0.4399`. Round 1 measured 43.2% against the training split and 44.1% against train+validation. Define it precisely ("features and label identical to a record in the training or validation split").
- **m6.** Rank intervals still use `int(np.percentile(...))` (`run_xai_v2.py` L220–221), which floors interpolated percentiles. Use `method='lower'` / `'higher'`. Also state that the interaction values use path-dependent TreeSHAP, unlike the interventional setting of the main analysis.
- **m7.** Disclose that about 35% of the explained UNSW records are misclassified (RF 101/289, CNN 96/289). Report the correct-only agreement and deletion values, which the code already computes; for example, correct-only Jaccard is 0.447 for the RF at seed 42. Readers should know that UNSW explanations are largely explanations of errors.
- **m8.** Report the share of corrected UNSW test records whose 10-feature vector (and label) matches a training record. The cross-split guard (loader step 10) compares post-OHE columns, not the selected features. Round 1 found 12.6% on the earlier 11-feature set (generic 74.6%).
- **m9.** Table 4: the UNSW CNN ranks `sbytes` 9th (0.0056) and `dbytes` 10th (0.0035), near zero, while `smean` is 1st. Because `smean` is roughly `sbytes`/`spkts`, comment on how correlated features split attribution in the interventional setting.
- **m10.** §5.3: "Guiding the search by SHAP does not make evasion easier for the RF (87% success…)". SHAP-guided success is *lower* than random (87% vs 100%) because the group set is fixed. This compares search strategies, not "easier evasion". Report queries at equal success, or the success rate at a fixed query budget.
- **m11.** Hardening on UNSW: "macro-F1 is 0.6047 against 0.6071" is a drop of 0.0024, larger than the 0.0004 seed range. That is still small, but "without loss of accuracy" (§7) should say "with a macro-F1 change of −0.002".
- **m12.** Table 9 caption and §5.4: say that "original data" in the hardening table is the seed-specific original model (macro-F1 0.9569 is the mean over seeds), which differs from Table 2's seed-42 value of 0.9565.
- **m13.** Figure 4 shows only the attacker-controllable condition; say so in the caption. Its error bars need a definition. Figure 1B: worms bars are invisible at this scale; add counts.
- **m14.** Data availability: "(v2 release to be added)" must be completed before acceptance. Remove the build header line "# Paper 5 v3 manuscript (double-brace placeholders …)" from the submission file.
- **m15.** Related work on realistic adversarial attacks against NIDS (problem-space constraints, e.g. Pierazzi et al., 2020; Apruzzese et al.'s work on realistic adversarial attacks against NIDS) and on adversarial training for NIDS should be cited in §2.3. That literature is the natural comparison point for the attacker-controllable threat model and for "counterfactual hardening".
- **m16.** The phrase "SHAP's top features beat random ones in every combination" is correct (I checked all 36 cells). Nothing to change; noted for the authors.

### B5. Spot-check of numbers (values.json and results JSONs against the text)

The following match:
- Table 1, all cells
- Table 5 agreement, stability and paired rows
- Table 9 means and seed ranges
- TON hardening per-seed evasion: 0.363 / 0.393 / 0.461 → 0.133 / 0.156 / 0.160; control 0.428 / 0.393 / 0.387
- per-class means in Tables 7 and 8, except the UNSW CNN attacker-controllable mean: the per-class average is 67.1%, the text says 68%, presumably because the per-class n is unequal (worms)
- J₀ values
- field-level Jaccard, which is genuinely identical for the TON RF and CNN (0.44 [0.42, 0.46]) and not a copy error

The only mismatches are m1 and m2.

### B6. Specific requested changes (checklist)

1. Faithfulness: add a removal operator that SHAP does not share (LIME's own perturbation distribution and a class-conditional or on-manifold donor), and temper the "more faithful" conclusion accordingly (M1).
2. Threat model: remove `dttl` from the attacker-controllable UNSW group (recompute `ct_state_ttl`), add monotone (increase-only) volume and duration for TON, and rerun §5.3 and §5.4 (M2).
3. Hardening:
   - add the "controllable-field augmentation without model-in-the-loop selection" ablation;
   - report local FPR near the counterfactuals;
   - report per-seed evasion CIs and McNemar;
   - fix the per-record RNG;
   - reconcile 31% vs 41% and 64% vs 75%;
   - stop attributing hardening to "explanations" unless SHAP-guided generation is shown to matter (M3).
4. Agreement: add the global-ranking baseline and ceiling-normalised agreement (M4).
5. CNN SHAP: run an `nsamples` sensitivity test and separate background variation from Monte Carlo variation (M5).
6. Statistics: use a two-level bootstrap over seed and record, resample within class, give paired CIs for the Table 1 differences, and give CIs for the evasion shares (M6).
7. Decomposition: add "A minus `id`" without re-selection, or reword "alone" (M7).
8. Positioning: cite prior TTL / `sttl` observations, remove "only the explanations made either visible", and either retitle away from "IoT" or add an IoT-native dataset (M8).
9. Fix m1 and m2 and the disclosure items m3, m5, m6, m7 and m8.

### B7. Recommendation

**Major revision (a smaller one than round 1). Score: 6/10.**

The authors addressed nearly every round-1 point, and the paper is now a careful, honest study. The audit case study and the threat-model contrast would interest C&S readers. I do not recommend minor revision because two of the four claimed contributions are not yet supported against the obvious alternative explanations:
- faithfulness, where the test is aligned with SHAP's value function;
- hardening, where the control is weak, the threat model is generous and no explanation is used.

Fixing them needs new, but inexpensive, experiments on the existing models: a few hours of CPU, and no new datasets except under M8. If the stronger controls confirm the results, and the threat model is tightened, I would expect to recommend acceptance after minor revision.
