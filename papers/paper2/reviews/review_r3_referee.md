# Referee report (round 3): JISA

**Manuscript:** "What Can an Attacker Actually Change? Evasion and Hardening of Compact Flow-Based Intrusion Detectors Under a Realistic Threat Model" (v4)

**Material reviewed:**
- Manuscript_filled.md (v4): about 13,260 words before the references, of which about 3,800 are in the nine tables. Abstract 249 words.
- Supplement_filled.md (Tables S1–S10).
- build.py: TABLE_CAPTIONS, SUPP_CAPTIONS, HIGHLIGHTS.
- review_r2_referee.md, review_r2_numbers.md and literature_verified.md.
- run_adv_v3.py: `Space` (feature map, mutable sets, BPDA transform), `_pgd_action`/`_pgd_action_strong`, `search`, `adaptive`, `train_robust`.
- run_adv_r2.py, all of it: `Thresh`, `tau_for`, `part_op`, `part_fs`, `part_union`, `part_feas`, `part_sat`.
- compute_values.py: Holm, two-level bootstrap, `paired`, `op_paired`, PRIMARY, crossover, and the operating-point, sweep and saturation tables.
- results_r2 JSONs (the `op` block for every seed of the undefended and main defended models) and values.json for the primary family.

As instructed, I loaded no models or data and ran no attacks. The recomputations below use only the stored JSON values.

**Score: 6.5/10. Recommendation: major revision.** This is borderline with minor revision, and the work needed is cheap. The round-2 revision is substantial and mostly well done:
- the strengthened feature-space reference;
- an evaluation at a matched operating point;
- a pre-declared primary family;
- five seeds for every CNN defence;
- feasibility and saturation checks;
- the promised supplementary analyses.

However, the new operating-point analysis has a construction flaw, and the paper reports it selectively. As a result, two of the abstract's defence statements do not follow from the authors' own stored results. The comparison with the feature-space reference is also still not on a common scale, so "overstated" remains close to a tautology. None of this needs retraining.

---

## Summary

**The good news first.**
- The conventional reference is now strong. It uses black-box corners, exhaustive patterns on the mutable features, and PGD through a straight-through (BPDA) quantile transform. Table S7 makes the AT-all/CCAT gradient-masking story clear: white-box ε-PGD gives 27.5%/27.0%, BPDA PGD gives 91.4%/84.1%, and the black-box corners give 100%.
- The matched-FPR idea is the right one. The TON-IoT result that "most of the gain was a shifted threshold" is an important, publishable observation. So is the finding that the clean-loop control alone halves the ASR at its own threshold.
- The tree-augmentation results are now reported honestly. They include the held-out bytes action, the β = 2 adaptive numbers and a saturation subsample, and the UNSW GBDT is called a loose lower bound.

**Remaining problems, in order of importance.**
1. **The decision rules differ (W1).** At the "matched FPR", defended models use a benign-probability threshold, while the undefended reference keeps the argmax rule. The two rules are not equivalent at equal FPR. The stored 1%-FPR results show that the threshold rule alone cuts the undefended TON-IoT RF's ASR from 24% to 10% at almost the same FPR.
2. **The UNSW-NB15 defence conclusions use all-actions ASR, which is dominated by the TTL (W2).** The paper itself declares volume and timing to be its main UNSW measure. The volume-only ASR at the matched FPR is already stored (`op.fpr_match.asr_volume`) but is not reported. On that measure every network defence keeps a large gain, AT-all included, and AT-threat is not distinguishable from AT-all or TRADES.
3. **The feature-space comparison is not budget-matched (W3).** The "increases only" variant is also not a relaxation of the realistic action set on UNSW-NB15. Lesson 1 contradicts the paper's own numbers.
4. **The masking diagnosis is applied asymmetrically (W4).** It was applied to AT-all, but the action-space PGD used to train and evaluate AT-threat and TRADES goes through the same non-BPDA quantile gradient.
5. **The tree claims still rest on unsaturated searches.** On TON-IoT an exact attack is easy, because the feature map is affine in the actions (W5).

---

## Status of the round-2 points

| R2 point | Status | Comment |
|---|---|---|
| W1 "grew with the budget" | **Fixed** | The sentence is corrected, and the reason (grids that are not nested) is explained. Neither nested grids nor a running maximum was adopted, but the text no longer claims monotonicity. |
| W2 ranking by the masked ε-PGD | **Mostly fixed** | The new "Is AT-all a fair baseline?" paragraph withdraws the TON-IoT ranking. However, the new UNSW-NB15 ranking statement ("under the realistic attack the most evaded") holds only with the TTL action (see W2 below). |
| W3 definition of the strongest attack | **Fixed** | §4.3 now lists the components exactly, and `part_union` shows that the omitted attacks add at most 0.8 pp. One inconsistency remains: Table 4 still gives TON-IoT CNN 51.3% (volume only) against 51.4% (all actions), while Table 8 says "= all actions" (small edit 6). |
| W4 promised analyses | **Fixed** | Prevalence-weighted ASR is in Table S3 and the cheapest action in Table S4. |
| W5 "unseen" TTL wording | **Fixed** | |
| W6 off-by-one in the query counts | **Fixed** | |
| W7 header-byte rule, two-packet rule | **Mostly fixed** | §3 is correct now. Table 2 ("flows with ≥ 2 packets") and Contribution 1 ("delays only on flows with at least two packets") still contradict the code (`p + ap >= 2`) and §3 (small edit 2). |
| W8 Holm families | **Fixed** | The primary and secondary families are declared (12 + 120/238 tests). See M3 for the wording "pre-specified". |
| W9 Table 8 splits | **Fixed** | Moved to Table S5, with a caption noting that the strongest attack differs between the splits. |
| M1 weak feature-space reference | **Largely addressed** | The reference is now strong, and AT-all with a BPDA inner maximiser was trained. Still open: the comparison is not on a common scale, and the "increases only" variant is mis-framed (W3). |
| M2 statistics and operating point | **Partly addressed** | Matched-FPR evaluation, a primary family and five seeds were added, which is good. The operating-point construction compares two different decision rules (W1). The UNSW-NB15 primary endpoint contradicts the stated main measure (W2). |
| M3 conditional headline claims | **Mostly addressed** | The abstract now names the floors and the TTL dependence. The crossover is in ε only and not in natural units (W3). The "overstated" claim is still binary. |
| M4 tree search not saturated / exact attack | **Partly addressed** | Added: the saturation subsample, a held-out bytes action, the stronger β = 2 numbers and an explicit statement that candidates are shared. No exact attack was run, and the UNSW GBDT search is still 10.3 pp short of saturation (W5). |
| M5 TTL-free detectors | **Addressed** | The MLP, the FPRs and end-to-end evasion for the CNN and GBDT were added, and the abstract and conclusion are qualified. Give end-to-end evasion for all four models (minor). |
| M6 fractional packets / MTU | **Addressed for the grid** | Grid ASR changes by ≤ 1.1 pp (Table S9). The strongest attack and the augmentation and training candidates are still fractional (M6). |
| M7 uncertainty reporting | **Mostly addressed** | Two-level bootstrap, undefended values on the same seeds, and a configuration sweep were added. The sweep has a single seed, and seed noise dominates it (M2). |
| R1 M5.3 held-out action for the network defences | **Not addressed** | It is not mentioned among the limitations either. |
| Minor 1 (Vitorino 2023 datasets) | **Changed, unverified** | Table 1 now says "IoT-23, Bot-IoT". literature_verified.md (a18) records "dataset not extracted". See R1 below. |
| Minor 5 (duplicate TON-IoT rows) | **Not fixed** | Table 6 still has "Grid, volume only" and "Grid, both" rows with identical TON-IoT values. They are excluded from the tests now, but they waste space. |
| Minor 13 (surrogate on disjoint data) | **Not done** | It is acknowledged as a limitation. |
| Minor 15 (AI declaration) | **Not fixed** | It still says "to write and edit the manuscript text" (small edit 9). |
| Minor 19 (length) | **Partly** | Three tables moved to the supplement, but the paper grew from about 12k to about 13.3k words (see M1). |
| Other minors (2–4, 6–12, 14, 16–18) | **Fixed** | |

---

## New weaknesses

### W1. The matched-FPR comparison uses a different decision rule for the defended and undefended models

**Location:** §4.5 "Operating point"; Table 7; §5.4 "Networks: most of the gain was a shifted threshold"; Abstract; Highlights 3–4; §6.1 second lesson; §7.

**What the code does.** `run_adv_r2.part_op` evaluates defended models with `Thresh`: a flow is benign iff p_benign ≥ τ, with τ set so that the validation FPR equals the undefended model's validation FPR. The undefended reference is evaluated at its "native" argmax rule. `compute_values.op_paired` states this explicitly: "the undefended reference is taken at its own operating point", and Table 7 prints "= own".

**Why it matters.** "Benign iff p_benign ≥ τ" is a one-vs-rest rule. "Benign iff p_benign is the largest of the class probabilities" is not. At equal FPR the two rules flag different attack flows, and an attack that pushes p_benign just above the largest attack-class probability (often well below 0.5 with ten classes) succeeds under argmax but not under a threshold. The stored results show the size of this effect for the undefended models at almost the same FPR (`op.native` against `op.fpr1`, seeds 0–2):

| TON-IoT | Native rule: test FPR, ASR | Threshold at 1% validation FPR: test FPR, ASR |
|---|---|---|
| RF | 0.009–0.010, 23.5–24.7% | 0.011–0.012, **9.8–11.3%** |
| GBDT | 0.007–0.010, 17.8–29.2% | 0.013–0.015, **8.3–12.4%** |
| CNN (5 seeds) | 0.014–0.016, 45.0–61.6% | 0.014–0.016, 37.1–61.7% (per-seed changes from −9 to +9 pp) |

For the trees, the rule change alone more than halves the ASR at a slightly *higher* FPR. Rule and threshold are therefore confounded in every "Δ vs undefended" of Table 7. The tree rows are the clearest case:
- "RF, augmented: −15.8 pp" and "GBDT, augmented: −22.5 pp" compare a thresholded augmented tree with an argmax undefended tree.
- An undefended RF under the threshold rule is likely already near 10%, so the matched-rule gain may be only a few points.

For the networks the bias has no fixed sign and is of the order of the effects tested. On TON-IoT the AT-threat difference is −16.1 pp with per-seed differences of +4, −28, −33, −25 and +1 pp, so ±9 pp of rule noise matters. The comparisons between defences (AT-threat against AT-all, TRADES and the clean loop) are fair, because both sides use `fpr_match`. Only the comparisons with the undefended model are affected.

**Fix (cheap, no retraining):**
- Add a `fpr_own` operating point for every undefended model, with `Thresh` and τ set to its *own* validation FPR, and use it as the reference in `op_paired` and in Table 7.
- Report how far the rule change alone moves the undefended ASR. Readers deploying a detector will want to know that a one-vs-rest benign threshold cuts tree evasion by half at no FPR cost (TON-IoT RF 24% → 10%, GBDT 24% → 10%, from Table 7's own "ASR at 1% FPR" column). This is a finding in its own right, and the paper is currently silent on it.
- Recompute the primary family and restate the abstract and §5.4 accordingly.

### W2. The UNSW-NB15 defence conclusions are stated on the TTL-dominated measure; the volume-only results at the matched FPR are computed but not reported, and they change the conclusions

**Location:** Abstract ("while on UNSW-NB15 it cut evasion from 99.2% to 43.6%"); Highlight 4; §5.4 ("AT-all and CCAT, which do not train on the TTL, stayed at 99.4% and 99.5%. AT-threat was significantly better than AT-all, CCAT and the clean loop on UNSW-NB15 (p ≤ 0.024)"; "Is AT-all a fair baseline?": "under the realistic attack the most evaded"; "Feature-space hardening against realistic attacks": "On our data, the protection that feature-space hardening seemed to give came from the training loop and the shifted threshold"); §6.1 second lesson; §7.

**The contradiction.** §3 says: "we … treat the former [volume and timing] as the main result on UNSW-NB15". §5.5 and the fifth lesson then argue that the TTL is a testbed artefact. Yet the primary UNSW-NB15 endpoint, and every defence statement about UNSW-NB15 in the abstract and conclusion, use all actions. The undefended CNN's 99.2% is almost entirely TTL: 61.3% with volume and timing.

**What the results already contain.** `part_op` stores `asr_volume` and `e2e_volume` at the matched FPR. From results_r2 (CNN, five seeds, matched FPR, grid + adaptive, β = 0.5, paired with the undefended model at its own operating point; raw p, unadjusted):

| UNSW-NB15 CNN, volume and timing | ASR at matched FPR | Δ vs undefended (61.3%) | raw p | Missed or evaded (undefended 72.3%) |
|---|---|---|---|---|
| Clean loop | 43.5% | −17.9 pp | 0.010 | 47.6% |
| AT-all | 31.6% | −29.8 pp | 0.014 | 42.3% |
| CCAT | 35.6% | −25.8 pp | < 0.001 | 50.6% |
| AT-threat | 18.8% | −42.5 pp | < 0.001 | 22.2% |
| TRADES | 16.0% | −45.3 pp | < 0.001 | 20.4% |
| MLP, AT-threat (3 seeds) | 31.5% | against 65.8% | | |

AT-threat compared with the other CNN defences on the same measure:
- AT-all: −12.7 pp (p = 0.27);
- CCAT: −16.7 pp (p = 0.064);
- TRADES: +2.8 pp (p = 0.48);
- clean loop: −24.6 pp (p < 0.001).

**What follows.**
- On the paper's own main UNSW-NB15 measure, every network defence keeps a large gain at the matched FPR, the ε-ball defences included.
- AT-threat is not distinguishable from AT-all, CCAT or TRADES.
- "AT-all and CCAT … stayed at 99.4% and 99.5%" and "the ε-ball defences were … under the realistic attack the most evaded" describe only the TTL.
- The Dyrmishi-style conclusion ("protection … came from the training loop and the shifted threshold") holds on TON-IoT but not on UNSW-NB15. There, AT-all beats the undefended model by 30 pp at the matched FPR, and the loop by about 12 pp (not significant).

**Fix:**
- Make the volume-only ASR (and end-to-end evasion) at the matched FPR the primary UNSW-NB15 endpoint, or include both.
- Report both in Table 7.
- Restate the abstract, Highlight 4, §5.4, the second lesson and §7. One possible wording: "On UNSW-NB15 every network defence kept part of its gain against volume and timing changes at the matched FPR, and the action-space defences were the only ones that also resisted the TTL choice, because they were trained on it."
- W1 applies here as well.

### W3. The feature-space comparison is not on a common budget, "increases only" is not a relaxation of the realistic actions on UNSW-NB15, and the first lesson contradicts the evidence

**Location:** Abstract (sentence 4); Highlight 2; §4.3 "Conventional reference"; §5.1 "The feature-space reference" and Table 5; §6.1 first lesson; §7.

1. **Magnitude.** The paper concedes the scale problem itself: "0.005 of the min–max range of a heavy-tailed byte count can exceed a flow's whole volume". The realistic budget at β = 0.5 is about 0.5(b + 1,000) bytes. An ε-ball that allows changes of a different order of magnitude will evade more, and that says nothing about consistency or realism. The crossover column of Table 5 is in ε, which has no physical meaning. R2 M3 asked for a common, physically interpretable scale, and it is still missing.

   **Fix (cheap):** add a "consistency-relaxed realistic" attack. Take the per-feature range that the realistic actions can reach for each flow at β (for example [x_j, x_j + Δ_j^max] for src_bytes, and the corresponding signed ranges for smean, sload and dload), and search the corners of that box independently per feature, as `part_fs` already does for the ε-ball. This separates the two effects that "overstated" currently mixes:
   - (a) a larger magnitude;
   - (b) breaking the feature dependencies.

   Alternatively, report the per-flow natural-unit size of ε on each mutable feature (median bytes, packets and seconds at ε = 0.005) next to the realistic budget.
2. **Direction on UNSW-NB15.** The realistic actions *decrease* dload (any delay), can decrease sload (delay) and can decrease smean (packets carrying only headers). `Space.mutable` on UNSW-NB15 is {sbytes, sload, smean, ackdat, dload}, and `part_fs` defines "increases only" as {0, +ε} on all five. The increase-only ball therefore excludes the directions the realistic attack uses. This is not a "restriction to what the attacker can change" (§5.1: "and so it did even when only increases were allowed"). Table 5 shows it directly: on the UNSW-NB15 RF, increases only reach 2.1%/2.3% at ε = 0.01/0.05 against 20.4% for the realistic attack. Use per-feature sign constraints that match the action map (sbytes and ackdat ↑, dload ↓, sload and smean ±), or drop the "increases only" framing on UNSW-NB15. On TON-IoT all four mutable features only increase, so the framing is correct there, but it still ignores ipb = b + h·p.
3. **Lesson 1 contradicts the evidence.** "the conventional ε-ball is not a conservative substitute for it. Made strong, the feature-space attack evaded more detected flows than the realistic one for every model and ε we tested…". An attack that evades *more* is conservative from the defender's point of view. The only exceptions are the UNSW-NB15 trees at ε ≤ 0.01, which is a magnitude effect. What the data do support is narrower:
   - feature-space ASRs are not interpretable as realistic vulnerability;
   - a *weak* feature-space attack mis-ranked the defences.

   Rewrite the lesson, Highlight 2 and the corresponding sentences of the abstract and conclusion in those terms.

### W4. The gradient-masking diagnosis is applied to AT-all but not to the action-space defences

**Location:** §4.3 "White-box action PGD"; §4.4 AT-threat and TRADES; §5.1 "How strong are the realistic attacks?"; §5.4.

**The asymmetry.** `_pgd_action` and `_pgd_action_strong` both call `_grad_fns`, which goes through `Space.tf_quantile`, the same non-BPDA, piecewise-linear transform that the paper blames for AT-all's apparent robustness. Two things follow:
- **Training.** The AT-threat and TRADES inner maximisers use these gradients (7 steps).
- **Evaluation.** The strong white-box action PGD is "0.4–5.8 percentage points below the grid search", and §5.1 itself reads that as weak gradients.

AT-threat's 5.5% on TON-IoT therefore rests entirely on black-box searches. Those searches are nearly saturated on TON-IoT but not on UNSW-NB15 (+9.8 pp from 1,000 points, +13.5 pp from the adaptive search on the undefended models). If BPDA can lift ε-PGD on AT-all from 27.5% to 91.4%, readers will ask what it does to action-space PGD on AT-threat.

**Fix:** run `_pgd_action_strong` with `tf_quantile_bpda` (the margin version already exists in `_bpda_fns`) against AT-threat, TRADES and the undefended networks at β = 0.5, add it to the strongest attack, and report the change. If AT-threat is not affected, say so. That would strengthen the paper's best network result. If it is affected, the "AT-threat versus AT-all" comparisons need the same caveat as AT-all.

### W5. The tree-augmentation result is still a lower bound, although an exact attack is straightforward on TON-IoT

**Location:** Abstract ("from 35.3% to 3.0%"); §5.4 "Trees"; §6.2; Table S8.

**The search is not saturated.**
- The saturation check shows the searches are not saturated on the undefended TON-IoT GBDT (+3.8 pp: the long adaptive search finds 3.5% new flows) or on the augmented UNSW-NB15 GBDT (+10.3 pp).
- Regular grids up to 17³ = 4,913 points find only 27.4% on the undefended TON-IoT GBDT, against 35.3% for the strongest attack. The evading regions are therefore thin, which is exactly where black-box search under-reports.

**An exact attack is straightforward on TON-IoT.** In `Space.selected` the TON-IoT map is affine in the actions:
- src_bytes = b + a_b;
- src_pkts = p + a_p;
- duration = d + a_d (with one binary condition, p + a_p ≥ 2);
- src_ip_bytes = ipb + a_b + h·a_p, with h fixed per flow.

Min–max scaling is affine as well. Every split of every tree is therefore a half-space in a 3-D action box. The exact minimal-evasion problem is a standard MILP (Kantchelian et al., 2016, which the paper cites), or, for the RF, an enumeration of the cells of the arrangement on a few hundred flows. The limitation in §6.2, "feasible in principle … and was not run", is not a sufficient answer for the paper's most important positive claim.

**Fix:** run the exact attack for the RF and GBDT, undefended and augmented, on TON-IoT at β = 0.5 on the 500-flow saturation subsample. If 3.0% holds, the paper gains a certified number. Also tone down the abstract: "cut … to 3.0%" should read "to at least 3.0% under our strongest search; 10–16% when an action was held out of the augmentation". The held-out results (9.9% without delays, 16.0% without bytes) are the honest generalisation figures and belong next to the 3.0%.

---

## Major/minor issues

### M1. Length and structure

At about 13.3k words with 9 main tables (two of them 24 and 40 rows), this is long for JISA. Most JISA research articles are 8–11k words. The prose (about 9.4k words) is acceptable. The bulk is in the tables and in §5.4 (3,250 words). Suggestions:
- **Table 8 (40 rows):** keep the 18 main rows. Move the ×2, w/o-delay, w/o-bytes and BPDA rows to a new supplementary table, and cite them in one sentence.
- **Table 6 (40 cells):** keep "Strongest, all actions" and "Strongest, volume only" (4 rows). Move the grid rows to the supplement. This also removes the duplicate TON-IoT rows.
- **§5.3:** fold its five sentences into §5.1 ("Which actions matter?"). The table is already Table S6.
- **§5.4 "Is AT-all a fair baseline?":** shorten to the three numbers that matter (27.5% → 91.4%, BPDA training collapses, ranking reverses only through the TTL), and move the sweep sentence to the limitations.
- **§2.3** repeats much of §1's contributions list. Cut one of the two.
- **§4.3:** the strongest-attack query arithmetic and the list of excluded attacks can go to the supplement.

A target of about 10.5–11k words including tables is realistic.

### M2. The configuration sweep cannot support ranking statements, and it undercuts the AT-threat > AT-all claim on TON-IoT

**Location:** §5.4 ("One further configuration of each defence … showed that the configuration matters"); Table S10; §6.2.

On seed 0 at the matched FPR, the TON-IoT results are:

| Defence (seed 0) | ASR at matched FPR |
|---|---|
| AT-all, ε = 0.02 | 83.0% |
| AT-all, ε = 0.05 | 67.4% |
| AT-all, ε = 0.1 | 41.1% |
| AT-threat, β = 0.5 (main configuration) | 49.0% |
| AT-threat, β = 0.25 | 18.1% |
| AT-threat, β = 1 | 20.5% |

- **AT-all.** It improves monotonically with ε, and at ε = 0.1 it beats the main AT-threat on the same seed. The ε-ball baseline was therefore not run at its best setting, and the primary result "AT-threat < AT-all (−27.3 pp, Holm p = 0.028)" holds for ε = 0.05 only.
- **AT-threat.** The main AT-threat's five matched-FPR seeds are 49.0%, 18.3%, 23.8%, 36.3% and 47.5%, so seed 0 is its worst seed. One seed per configuration cannot separate configuration from seed noise (SD about 14 pp).

**Fix:** run the sweep on seeds 0–2, including AT-all ε = 0.2 if training is stable, before making any between-defence ranking on TON-IoT. Otherwise, limit the claim to "than AT-all at ε = 0.05". The abstract's TON-IoT statement does not rank against AT-all, but §5.4, the second lesson and §7 do.

### M3. Primary family: the wording and the choice of endpoint

**Location:** §4.5 "Tests"; §5.4.
- **"Pre-specified."** The family was specified in response to the round-2 review (compute_values.py: "pre-specified in review round 2, M2"), after the main results were known. Say "specified before the operating-point experiments were run", not "pre-specified", which implies preregistration.
- **Endpoint: ASR or end-to-end evasion.** At the matched FPR the detection sets differ greatly. TRADES on TON-IoT detects only 41–73% of the sampled attacks before modification (`op.fpr_match.detected`, mean 0.61), against 97–99% for the undefended CNN. An ASR conditional on detection then compares different flow populations. End-to-end evasion (missed or evaded) is the operationally relevant endpoint and does not have this problem. Use it as the primary endpoint, or at least report the detection rate at the matched FPR in Table 7.
- **Mixed endpoints in one family.** The family mixes endpoints: networks use the grid + adaptive search at the matched FPR, and trees use the strongest attack at their own threshold. This is acceptable, but the caption should state it clearly. Also apply W1 to the tree rows of Table 7, which currently print a matched-FPR Δ next to an own-threshold p-value.

### M4. Table 7: wrong seed count for the trees, and a meaningless "ASR at 1% FPR" for some models

**Location:** Table 7 caption (build.py '5.4'); the "ASR at 1% FPR" column; §5.4 ("every UNSW-NB15 network was evaded on at least 96.0% of its detections").
- **Seed count.** The caption says "Means over five seeds (MLP with AT-threat: three)". The operating-point runs for all tree rows have seeds 0–2 only (`r2seeds`: `rf_`/`gb_` → SEEDS3; results_r2 contain op for seeds 0–2). Correct the caption.
- **Near-empty detection sets.** At 1% FPR several UNSW-NB15 models detect almost nothing. The TRADES CNN detects 0.0–5.5% of sampled attacks per seed (seed 1: 0.0%), the AT-all CNN 11–20%, and the GBDT on seed 1 11% (τ ≈ 0). Its "100.0%" is an ASR over a handful of flows. Replace the column with end-to-end evasion (or add the detection rate), and rephrase the sentence as "at 1% FPR the UNSW-NB15 networks detected only 0–50% of the attacks before modification, and almost every detected flow could be evaded".
- **Ties at the threshold.** For tree probabilities with many ties (RF votes are multiples of 0.01), `tau_for` with `np.quantile` plus "≥ τ" does not hit the target FPR. State the realised test FPR, which the table already does for the matched column but not for the 1% column.

### M5. The network–tree comparison at a deployable operating point

The paper's network–tree comparison is made at the argmax rule. Table 7 shows that at a 1% FPR, under a one-vs-rest threshold, the TON-IoT trees are evaded on about 10% of detected flows (RF 10.4%, GBDT 9.8%) against about 50% for the CNN (50.3%) and the MLP (65.3%). The network–tree gap is therefore larger at a deployable operating point than in Tables 4 and 6, and the floors-dependence caveat ("the gap almost vanished" at floors × 0) may not hold there. One sentence and one Table 6 row at 1% FPR would show whether the "trees are harder to evade" conclusion depends on the decision rule.

### M6. The feasibility check covers only the grid

**Location:** §3 (last sentence of "Feature consistency"); §5.1; §6.2.

`part_feas` re-runs only the β = 0.5 grid with integer packets and the MTU bound. Three things remain fractional:
- the strongest attack, which includes the dense grid, the adaptive search and PGD;
- the AT-threat and TRADES inner maximisers;
- the tree-augmentation candidates.

The 1.1 pp bound is reassuring for the grid but does not cover the headline numbers. At minimum, apply `feasible()` to the evading candidates of the adaptive search (cheap: re-check only the successful actions). The TON-IoT and UNSW-NB15 MTU rules also differ (1,460 payload bytes against 1,500 bytes including headers), while the Table S9 caption says "a 1,500-byte MTU". Align the caption.

### M7. Related work and citations

- **R1.** Vitorino, Praça & Maia (2023), Table 1: "IoT-23, Bot-IoT". The authors' own verification log (literature_verified.md, a18) records "dataset not extracted"; the cell was filled after R2 minor 1 without a recorded check. My recollection is that this Annals of Telecommunications paper evaluates on IoT-23 only, but I am not certain. Check the full text and correct, or write "–" as for the other unextracted cells. The models (RF, XGB, LGBM, isolation forest) match the abstract.
- **R2.** BPDA is used throughout (§2 contribution 2, §4.3, §4.4, Table S7), but Athalye, Carlini & Wagner (2018), "Obfuscated gradients give a false sense of security", ICML, PMLR 80:274–283, the paper that introduced BPDA, is not cited. literature_verified.md lists it as relevant. Add it where "straight-through (BPDA)" first appears.
- **R3.** The score-based adaptive search is close to the Square Attack (Andriushchenko et al., ECCV 2020) and to the random-search components of AutoAttack (Croce & Hein, ICML 2020). One sentence that positions it, or a run of one of them, would let readers calibrate "strong".
- **R4.** On benchmark artefacts, Arp et al. (2022), "Dos and don'ts of machine learning in computer security", USENIX Security, is the standard reference for spurious correlations and would support the fifth lesson alongside Flood et al.
- **R5.** All other references check out against literature_verified.md and my own knowledge (venues, volumes and pages): Apruzzese 2019 NCA (datasets and models per FT), Sheatsley 2022 JCS, Han 2021 JSAC, Dyrmishi 2023 S&P, Flood 2024 EuroS&P, Debicha 2023 C&S, Chernikova & Oprea 2022 TOPS, Alhajjar 2021 ESWA, Simonetto 2022 IJCAI, Teuffenbach 2020 CD-MAKE, and the ML foundations. I found no reference I suspect does not exist. The descriptions of Dyrmishi et al. and Flood et al. are accurate.

### M8. Reproducibility

The released scripts, the per-seed JSONs (with packed per-flow evasion masks), the fixed attack-set indices and the automated text-versus-results check are good practice. To complete them:
1. Release the trained models, or their hashes and the exact training commands. All round-2 results depend on the saved models, and retraining on a different GPU or TensorFlow version will not reproduce them bit for bit.
2. Add an environment file with pinned versions of TensorFlow, scikit-learn (HistGradientBoosting early stopping depends on the version) and NumPy.
3. List the run order of run_adv_v3.py and run_adv_r2.py, with the approximate compute time per part (`eval_hours_total` is computed but not reported).
4. Document in the README that the matched-FPR analysis uses the threshold rule and not argmax (see W1).

---

## Small edits

1. **§3, "to within 3e-04 (rounding)".** `map_err` is 3.4e-04 on UNSW-NB15 (R2 numbers B6). Write "about 3e-04" or "within 4e-04".
2. **Table 2 and Contribution 1.** "flows with ≥ 2 packets" and "delays only on flows with at least two packets" should read "flows with at least two packets after any added packets", as in §3 and the code.
3. **§3, TON-IoT header bytes.** The per-packet header estimate is (ipb − b)/p clipped to [20, 60] bytes (40 if p = 0). Say so.
4. **§5.4 Trees.** "the saturation check on a subsample left the augmented trees' ASR within 3.8 pp of it". For the augmented TON-IoT trees the gain was 0.5–0.6 pp (Table S8: 10.7 → 11.2, 3.5 → 4.1). The 3.8 pp belongs to the undefended GBDT. Use the smaller number; it helps the claim.
5. **Table 9 and its caption.** The "Volume only, with/without TTL features" columns are *grid* ASRs (58.6/29.8 for the CNN), while the text of §5.5 quotes the *strongest* ASRs (54.8/36.3). Say "grid" in the column header.
6. **Tables 4 and 8.** Table 4 gives TON-IoT CNN 51.3% (strongest, volume only) and 51.4% (all actions), while Table 8 says "= all actions" and Table 5 uses 51.3%. Use the `strongest_plus_volume` union, which adds the white-box attack on TON-IoT, or footnote the difference.
7. **Table 6.** Drop the TON-IoT "Grid, both" rows (identical to "Grid, volume only").
8. **Abstract, "(CNN 35.0% against 51.0%, not significant after correction)".** Add "with five seeds". After W1 the numbers may change in any case.
9. **Declaration of generative AI.** "used Claude (Anthropic) to write and edit the manuscript text". Elsevier's policy permits AI tools "to improve readability and language", not authorship of the text. Reword it to match what was done, and keep the disclosure that analysis code was AI-assisted under Data availability.
10. **Figure 2** shows the CNN defences at their own thresholds only. Since §5.4's message is that this view is misleading, add the matched-FPR curve (at least at β = 0.5) or say in the caption that the curves are at the own threshold.
11. **§5.1 "Queries and missed attacks".** "the strongest attack assumes thousands of queries per flow". Give the numbers (1,761 and 7,163) here rather than only in §4.3.
12. **§5.5.** Give end-to-end evasion for all four TTL-free models, not only the CNN and GBDT.
13. **Table S8 caption.** "Strongest: the union of the main evaluation on the same flows" is also the starting point of the "Union of all" column. Say so, so that readers do not add the columns.

---

## What would most raise the score

1. **Fix the operating-point construction (W1).** Evaluate the undefended models under the same threshold rule at their own FPR, recompute Table 7 and the primary family, and report the effect of the rule change itself. This needs no retraining.
2. **Report the UNSW-NB15 defences on volume and timing at the matched FPR (W2).** The numbers are already in results_r2. Restate the abstract, Highlights 3–4, §5.4, the second lesson and §7.
3. **Put the feature-space comparison on a common scale (W3).** Add a consistency-relaxed realistic attack or natural-unit ε, fix the UNSW-NB15 "increases only" signs, and rewrite Lesson 1 and Highlight 2.
4. **Run BPDA action-space PGD against AT-threat and TRADES (W4),** and an exact MILP/cell attack on the TON-IoT trees (W5).
5. **Cut about 2.5k words** by moving the variant rows and grid rows to the supplement (M1), and fix the Table 7 caption and the 1%-FPR column (M4).
