# Referee report (round 2): JISA

**Manuscript:** "What Can an Attacker Actually Change? Evasion and Hardening of Compact Flow-Based Intrusion Detectors Under a Realistic Threat Model" (v3)
**Material reviewed:** Manuscript_filled.md (v3, about 12,000 words including tables and references), review_r1_referee.md, run_adv_v3.py (893 lines; I read the feature map, both PGD variants, the grid, adaptive search, strongest-attack union, training loops, augmentation and driver), compute_values.py (Holm families, bootstrap, paired tests, the table builders), values.json (about 60 values checked against the text), result JSONs (seed 0 for spot checks), Figures 1 and 3, DESIGN.md. As instructed, I ignored the unfilled split placeholders in §5.2 and Table 8.

**Score: 6/10. Recommendation: major revision.** This is a clear improvement on round 1 (5/10). The remaining problems are concentrated in the defence claims and the conventional-attack reference, not in the threat model.

---

## Summary

The revision takes round 1 seriously, and much of the new work is good:
- **Feature map.** dload is recomputed, ACK delay is restricted to flows with ackdat > 0, a delay needs at least two packets, header bytes are added on UNSW-NB15, and TTL 0 is dropped.
- **Fresh evaluation candidates.** Evaluation draws its random points with seeds disjoint from the augmentation.
- **Stronger and broader attacks.** There is a saturation table, a score-based adaptive search, a TTL sweep over 1–255, a strong margin-loss PGD and a black-box ε-ball search that also covers the trees.
- **Controls and new detectors.** A clean-loop control has been added, plus a second augmentation round, held-out-action augmentation and UNSW detectors trained without the TTL features.
- **Statistics and presentation.** Holm correction is applied, the main defences have five seeds, related work is repositioned, and the material on the unpublished earlier version has been removed.

The UNSW-NB15 story is now honest. Tree augmentation that looked almost perfect against its own TTL candidates (0.6%/1.7%) falls to 30.8%/40.3% once unseen TTLs are allowed. That is a useful, publishable finding.

Four problems remain.

1. **The conventional-attack reference is itself a weak, masked attack, and the defence ranking rests on it.** The authors' own black-box ε-ball search evades 100.0% of detected flows of *every* TON-IoT CNN defence, AT-all and CCAT included, at every ε. Under white-box ε-PGD the same models look robust (23.9%/30.2%). The claim that "each defence was most robust to the perturbations it was trained on" therefore contradicts the paper's own numbers.
2. **The network-defence conclusions do not survive the authors' own correction, and they are confounded by the operating point.** Several claims fail after correction:
   - AT-threat vs undefended has Holm p = 0.106 (TON-IoT) and 0.168 (UNSW-NB15).
   - AT-threat vs the clean loop has Holm p = 0.515 (TON-IoT) and 1.000 (UNSW-NB15).
   - The clean loop vs undefended, on which the "half of that gain" sentence rests, has Holm p = 1.000.

   Meanwhile every defence raises the benign FPR and the detection rate (UNSW-NB15 CNN: detection 71.2% → 97.7%, FPR 0.198 → 0.241). Part of the apparent robustness is a decision boundary moved towards "attack".
3. **Several headline statements are conditional on choices the abstract does not mention.** On TON-IoT, the additive floors carry most of the result (floors × 0 gives 3.0–8.5% ASR, and the network–tree gap almost vanishes). Without the TTL features on UNSW-NB15, the network–tree gap closes. "Overstated on both datasets" holds only for particular ε values and attack types.
4. **The tree-augmentation result is still a lower bound from a search that is not saturated.** It is not checked by an exact attack, even though one is cheap in a 3–4-dimensional action space.

I also found about nine factual errors or text/code mismatches (below). None is fatal, but together they suggest that the automated text-versus-results check verifies numbers but not the sentences built around them.

---

## Status of the round-1 points

| R1 point | Status | Comment |
|---|---|---|
| M1 tree augmentation non-adaptive | **Largely addressed** | Fresh random directions (EVAL_OFF/AUG_OFF), +1,000 points, adaptive search, TTL sweep, second round and held-out action are all added. Still open: the 2^k corners and the 4 TTL candidates are still shared with the augmentation, and §4.3 does not say so. The held-out action on TON-IoT is the weakest one (see M4). There is no exact attack. |
| M2 feature-consistency gaps | **Mostly addressed** | dload, ACK delay, the ≥ 2-packet rule, UNSW header bytes, no TTL 0 and the floor sensitivity are all done. New gaps: fractional packets, no MTU bound, and a header estimate that is a clipped mean packet size (W7, M6). The realised overhead per class was computed (`min_action_per_class`) but not reported (W4). |
| M3 "misjudged in both directions" depends on budgets/TTL | **Partly addressed** | The abstract now names the TTL artefact, white-box vs white-box is reported, and "reversed" is softened. Still missing: the common-budget plot or crossover points. The claim remains binary and depends on ε and on which feature-space attack is used (M3 below). |
| M4 black-box beats white-box | **Addressed for action PGD** | The strong PGD, saturation table and adaptive search are in, and the "masking" sentence is honest. However, the same diagnosis now applies to the feature-space ε-PGD, which the paper still uses as a reference (M1 below). |
| M5.1 training-loop confound | **Addressed** | The clean-loop control was added and is informative. |
| M5.2 one configuration per defence | **Not addressed** | It is listed as a limitation only. The ε = 0.05 AT-all is probably trained against a masked attack (M1 below), so the comparison with it is weak. |
| M5.3 same action family (networks) | **Partly** | The unseen-TTL sweep for AT-threat is good (8.9%). There is no held-out-action test for the network defences. |
| M5.4 TRADES description | **Fixed** | |
| M5.5 direct defence-vs-defence tests | **Added** | None of them is significant after Holm correction. The text says so for AT-threat but still concludes "most effective defence" (M2 below). |
| M5.6 end-to-end metric as a column | **Partly** | The end-to-end figure is given in the text only (two places). `T['overall']` is computed in compute_values.py but not printed. |
| M6 mixed 5-seed/3-seed baselines | **Not fixed** | Table 10 still shows 5-seed undefended means next to Δ values paired over 3 seeds (for example AT-all on TON-IoT: 18.9% against 51.4%, but Δ = −30.6 pp). |
| M6 bootstrap CIs, splits, Holm, per-pair Welch | **Partly** | The CIs cover flow resampling only (M7 below). There is one seed per split (pending). Holm is applied, but the families are inconsistent (W8). Per-pair Welch tests are done. |
| M7 novelty/related work | **Addressed** | Table 1 and the new citations are good. Engagement with Dyrmishi et al. is thin (minor 16). |
| M8 unpublished earlier version | **Fixed** | |
| M9 weak UNSW models, TTL-free detector, queries | **Mostly addressed** | TTL-free detectors were added (3 seeds, no MLP), and the FPR caveat and query counts are in. The disjoint-data surrogate is not done. |
| Minors 1–16 | Mostly fixed | Still open: 3 (prevalence-weighted ASR promised but absent), 6 (cheapest action computed but absent), 8 and 11 (TTL-only and β = 2 / ε = 0.01 values in the text only), 12 (Figure 3 has no seed spread) and 16 (AI statement wording). |

---

## Wrong items (factual errors, or text contradicting the code or results)

**W1. "Success grew with the budget for every model (Figure 1)" (§5.1, "Which actions matter?").** This is false.
- UNSW-NB15 RF, grid volume-only: 8.4% (β = 0.1), 12.9% (0.25), 12.0% (0.5), 11.2% (1), 10.6% (2). Figure 1 shows the decline.
- UNSW-NB15 MLP, grid "both": 94.8% (β = 1) and then 94.7% (β = 2).

The cause is structural. `search` scales one fixed unit grid to each budget box, so the candidates at β = 2 do not contain those at β = 0.5, and a worst-case ASR that falls with the budget is a symptom of non-saturation. The same mechanism explains why Table 6 is not monotone (floors × 2: RF 11.7% < floors × 1: 12.0%; RF bytes only 5.4% > bytes + packets 4.9%). Correct the sentence, and either make the grids nested (include the smaller-β candidates) or report the running maximum over β as a lower bound.

**W2. "AT-all and CCAT, which were trained in that ball, were the most robust CNNs on TON-IoT (23.9% and 30.2%) … Each defence was most robust to the perturbations it was trained on, and a defence chosen with the conventional attack would have been the wrong one" (§5.4, "The two threat models ranked the defences differently"; repeated in §6.1 and §7 "ranked the defences differently").** The paper's own black-box ε-ball corner search (`bb_unconstrained_e*`) evades **100.0%** of detected flows of every TON-IoT CNN at ε = 0.01, 0.05 and 0.1: undefended, clean loop, AT-all, CCAT, AT-threat and TRADES alike. Under the strongest conventional attack the authors ran, AT-all is not robust in its own ball, and there is no ranking at all. The 23.9%/30.2% come from a white-box ε-PGD that this black-box search beats by about 75 pp. By the paper's own Carlini et al. criterion (§5.1), that points to uninformative gradients, not robustness. On UNSW-NB15 the black-box search does separate the defences (AT-all 72.6%, AT-threat 95.6%, TRADES 82.2% at ε = 0.05), but the ε-PGD values (21.4%, 27.3%, 16.9%) are again far below it. The ranking statement must be withdrawn or restated in terms of the white-box attack and its weakness (see M1).

**W3. Definition of the "strongest attack" (§4.3: "a flow counts as evaded if any of the attacks above evades it (with and without the TTL choice)").** In the code, `E05` holds only `grid`, `grid_vol`, `dense1000(_vol)`, `ad_none/ad_cands/ad_free` and `wbs`. The union therefore *excludes*:
- the TTL sweep over 1–255;
- the basic white-box PGD;
- transfer;
- the 24-point grid;
- the per-action grids.

For the augmented UNSW-NB15 trees, the TTL sweep alone evades 17.8%/20.8%. Those flows are not guaranteed to be in the union, because `ad_free` changes the TTL in only 25% of proposals and starts from a random non-zero action. The strongest-attack ASR can therefore be understated, and the text does not describe what is computed. A related point: on TON-IoT, `vol_keys` drops `wbs` although the white-box attack is volume-only there. That is the only reason "Strongest, volume only" (51.3%) differs from "all actions" (51.4%) for the TON-IoT CNN in Tables 4 and 10. Either fix the union or state exactly which attacks it contains.

**W4. Promised analyses are absent.**
- §3 says "we also report it [the ASR] weighted by the prevalence of each class in the test set".
- §4.3 says "for each flow that evades without a TTL change, the cheapest successful action".

Neither appears anywhere in the results. Both are computed (`*_strongest_b0.5_prev`, `T['overall']`, `T['minaction']`) but never inserted. Either report them (they are cheap and answer R1 M2.6 and minor 3) or delete the promises. For reference, the prevalence-weighted ASR is close to the stratified one (TON-IoT CNN 52.1% against 51.4%; RF 24.3% against 25.2%).

**W5. "TTL values never seen in benign training flows evaded the undefended models about as often as the candidates (CNN 84.5% against 83.8%)" (§5.5).** Two problems.
1. In the code, "unseen" means "not among the four candidates" (`if float(t) not in set(sp.ttl_cands)`), not "never seen in benign training flows". Many of those values do occur in benign training data.
2. "About as often" holds for the CNN, the MLP and the GBDT, but not for the RF (unseen 36.6% against candidates 53.5%; Table 11).

The wording in §5.4 ("TTL values outside the candidates") is the correct one; use it here too.

**W6. Query counts are off by one (§5.1, "Queries and missed attacks": "the median evading flow needed 3 queries … and 19 (CNN) to 64 (GBDT)").** `first` is the zero-based candidate index (`hit.argmax(1) + ti * K`). The medians are therefore the 4th and the 20th to 65th queries. Index 64 on UNSW-NB15 is exactly the first candidate with a changed TTL (zero action, first TTL candidate). So the "64" for both trees means that the median evading tree flow is evaded by the first TTL change, which deserves saying. Note also that query 1 (all actions zero) is the unmodified, already-detected flow and can never succeed.

**W7. The UNSW-NB15 header-byte rule is misdescribed (§3 "Feature consistency": "each added packet also adds the flow's mean header size (between 28 and 40 bytes)").** The code computes `h = clip(sbytes/spkts, 28, 40)`. That is the flow's **mean packet size** clipped to [28, 40], not a header estimate, and it equals 40 for every flow whose mean packet exceeds 40 B. Describe it as "40 bytes per added packet, or the flow's mean packet size if smaller (at least 28)". Similarly, Table 2 says delays apply to "flows with ≥ 2 packets", but the code allows a delay whenever p + a_p ≥ 2, that is, after the added packets. §1 contribution 1 says "delays only on flows of at least two packets". Both are acceptable designs, but the text must match the code.

**W8. The Holm families are misdescribed (§4.5: "Within each family of tests (threat-model comparisons, network–tree comparisons, defence comparisons) p-values are adjusted").**
- **Threat-model comparisons.** compute_values.py contains no threat-model tests, and §5.1 reports no p-values.
- **Network–tree comparisons.** The family is each dataset × attack combination (4 tests), not the 40 network–tree tests.
- **Defence comparisons.** The family is all 154 or 186 tests per dataset. These mix clean-F1 tests, ten attack variants and defence-versus-defence tests, including exact duplicates on TON-IoT (`bb_both` = `bb_volume`).

The network–tree claims are thus judged by a lenient family and the defence claims by a very strict one, and §4.5 discloses neither. State the families exactly (see M7).

**W9. Table 8 rows are not comparable (ignoring the pending placeholders).** The "main" rows are seed 0 of the full evaluation: TON-IoT CNN 45.8%, while Tables 4 and 10 give 51.4% as the mean over five seeds. The split runs use `light=True`, so they have no dense grid and adaptive search at β = 0.5 only, and their "Strongest" column is a different, weaker union (the seed-0 result for `cnn_none_split1` lists `strongest_components` without `dense1000`). Either run the splits with the full evaluation or compare only the grid columns, and mark "seed 0" in the table.

---

## Major issues

### M1. The feature-space reference attacks are weak, and the conclusions drawn from them are unsafe (§4.3, §5.1, §5.4, §6.1, §7)

The paper accepts in §5.1 that a black-box attack beating a white-box one signals uninformative gradients. It applies this to action-space PGD but not to its own feature-space ε-PGD, where the evidence is stronger:

- On TON-IoT, black-box ε-ball corners evade 100% of every CNN, defended or not, while white-box ε-PGD gives 22.3%–83.8% (W2).
- ε-PGD restricted to the mutable features is **non-monotone in ε**: TON-IoT CNN 42.2% (ε = 0.01), 39.8% (0.05), 39.1% (0.1); UNSW-NB15 CNN without TTL features 42.2% (0.05), 39.8% (0.1). A worst-case attack over nested balls cannot decrease. The cause is the same piecewise-linear quantile transform, whose gradient is zero outside the reference range and on flat segments.
- AT-all and CCAT are **trained** with this same 7-step ε-PGD through the quantile transform. They are therefore probably trained against a masked inner maximiser, which would explain why they are robust to white-box ε-PGD and not at all to random corners. This weakens the comparison "action-space adversarial training did better than training in the ε-ball" (§6.1, §7): the ε-ball baseline may be a poorly trained one.

Three conclusions depend on this attack:
- the defence-ranking claim (W2);
- "Restricted to the mutable features, ε-PGD evaded the CNN on only 39.8%, less than the realistic attack, so the excess of the full ball comes from features the attacker cannot change" (§5.1);
- "Comparing white-box with white-box … 42.1% … against 75.1%" (§5.1).

**Fix:**
- Run the black-box corner and random search on the mutable features only. Constrain it to increases if the point is "what the attacker could change".
- Strengthen ε-PGD: margin loss, restarts including corners, and attack in quantile space or with a straight-through or BPDA approximation of the quantile transform.
- Report AT-all and CCAT robustness under the strongest feature-space attack.
- Restate or drop the ranking claim. Consider training AT-all with an inner maximiser that is not masked (for example PGD in quantile space) before concluding that ε-ball training is inferior.

### M2. The network-defence claims are not supported after correction and are confounded by the operating point (Abstract, §5.4, §6.1, §7)

**Statistics.** With the authors' Holm families (values.json):

| Comparison | Δ | Holm p | raw p |
|---|---|---|---|
| CNN AT-threat vs undefended, TON-IoT | −45.9 pp | 0.106 | < 0.001 |
| CNN AT-threat vs undefended, UNSW-NB15 | −63.7 pp | 0.168 | 0.001 |
| Clean loop vs undefended, TON-IoT ("half of that gain") | −24.5 pp | 1.000 | 0.018 |
| AT-threat vs clean loop, TON-IoT | −22.2 pp | 0.515 | 0.004 |
| AT-threat vs clean loop, UNSW-NB15 | −44.4 pp | 1.000 | 0.042 |
| AT-threat vs AT-all / CCAT / TRADES, TON-IoT | −16.1 / −23.1 / −12.6 pp | 1.000 each | 0.024–0.033 |
| AT-threat vs AT-all, UNSW-NB15 | −62.0 pp | 1.000 | 0.016 |

Only the tree-augmentation effects (Holm 0.002–0.031) and the AT-threat TTL-only effect on UNSW-NB15 (0.022) survive. Yet the abstract and conclusion state without qualification that "adversarial training on the attacker's actions was the most effective defence we tested" and that the control "accounted for half of its gain". The authors cannot both adopt a strict correction and then state the uncorrected rankings as findings. Effect sizes this large with p < 0.05 unadjusted are suggestive, and the problem is mostly too few seeds (3 for the clean loop, AT-all, CCAT and TRADES).

**Operating point.** Every network defence moves the decision boundary towards "attack":
- TON-IoT FPR: 0.015 → 0.037 (clean loop), 0.044 (AT-threat), 0.101 (AT-all).
- UNSW-NB15 FPR: 0.198 → 0.241 (AT-threat).
- UNSW-NB15 CNN detection before attack: 71.2% → 97.7%.

A detector that flags more traffic is trivially harder to "evade into benign". The clean-loop control in particular adds 128 *real attack* records to every step (`train_robust`, `j = rng.choice(att, …)`), so it is also a class-prior shift towards attacks, not only a "training loop" control. "Part of the networks' gain came from the training loop" should read "from the loop and the extra weight on attack records".

**Fix:**
- Compare defences at a matched operating point: threshold the benign probability of each defended model so that its benign FPR equals the undefended model's (and 1%), then recompute ASR and the end-to-end evasion.
- Pre-specify a small primary family (for example strongest-attack ASR of AT-threat, AT-all, TRADES and the clean loop vs undefended and vs each other, per dataset) and apply Holm to that.
- Raise the clean loop, AT-all and TRADES to five seeds. On the authors' timings this is about 10–15 GPU hours.
- Restate the abstract and conclusion to match whatever survives.

### M3. The headline comparisons are conditional on the budget floors, the ε value and the attack type (Abstract, §5.1, §6.1, §7)

- **Floors.** §5.1 itself shows that with proportional budgets (floors × 0) the TON-IoT volume-only ASR is 4.5% (CNN), 8.5% (MLP), 3.5% (RF) and 3.0% (GBDT). The abstract's "the strongest attack evaded 51.4% of the CNN's detections" is therefore mainly a statement about padding small flows by up to 500 bytes, 5 packets and 0.5 s. At floors × 0 the network–tree gap also nearly disappears (4.5/8.5 against 3.5/3.0 pp), so "trees … harder to evade" on TON-IoT holds only at floors × 0.5 or more. The abstract and conclusion must say this. §6.1 "third lesson" does, but the abstract does not.
- **"Overstated on both datasets."**
  - On UNSW-NB15 with volume and timing only, the realistic attack is *not* below white-box ε-PGD for the networks: CNN 61.5% against 63.2% (ε = 0.05); MLP 66.2% against 63.8% (0.05) and 64.7% (0.1). The MLP result is higher.
  - The black-box ε-ball search evades 99.6%–100% of the UNSW-NB15 networks, above even the all-actions realistic attack (99.3%, 95.4%). So "the realistic attack exceeded it only through the TTL" is true against white-box ε-PGD only.
  - For the UNSW-NB15 RF, black-box ε = 0.01 gives 35.4%, which is above the realistic volume-only 19.4%. So "overstated" for the trees also depends on ε.

  R1 M3 asked for ASR on a common, physically interpretable budget (for example the natural-unit size of the ε-ball on the mutable features against β) or crossover points. This is still missing, and the claim is still binary.

**Fix:** report the crossover (the ε at which the feature-space attack equals the realistic attack at each β, per model and attack type). Rewrite the abstract sentence as "at ε ≥ 0.01 and β = 0.5, the feature-space attack overstated …, except for the UNSW-NB15 networks with volume and timing only, where they were similar". State the floor dependence in the abstract.

### M4. The tree-augmentation robustness is a lower bound from a search that is not saturated (§4.3, §5.4, Abstract, §7)

The tree result ("cut evasion of gradient-boosted trees on TON-IoT to 3.0% at no accuracy cost") is now the paper's most important claim, and it rests entirely on black-box search.
- **The search is still climbing on the GBDT.** For the undefended GBDT on TON-IoT, Table 5 runs 23.5% (24 points), 25.0% (standard), 29.8% (+1,000), 34.5% (+adaptive) and 35.3% (strongest). Each stronger search still adds several points. For the augmented GBDT, seed 0 rises from 0.3% (grid) to 2.7% (strongest). Nothing shows that 3.0% is close to the true worst case.
- **An exact attack is cheap here.** The action space is 3-D (TON-IoT) or 4-D (UNSW-NB15), the feature map is known, and tree ensembles are piecewise constant. Each split threshold on a recomputed feature defines a level set in action space. Enumerating the cells, or solving a MILP over the actions with the tree constraints (Kantchelian et al., 2016, already cited), would give the exact minimal evasion for each flow and budget, at least for the RF with depth 16 on a subsample of a few hundred flows. That would turn a lower bound into an exact number.
- **Shared candidates.** The 2^k corners and the four TTL candidates are shared between augmentation and evaluation. §4.3 says only that random seeds are disjoint. Say so explicitly.
- **The held-out action on TON-IoT is the least effective one.** Delays alone evade the undefended trees on only 0.1% (RF) and 0.5% (GBDT) of flows (Table 6), so "augmentation without delays, attacked with delays" barely tests generalisation on TON-IoT. Hold out bytes or packets, which alone evade 7–12% of the undefended trees.
- **β = 2.** The text quotes only the grid (21.5%); the adaptive search at β = 2 gives 23.9% for the augmented GBDT and 19.1% for the RF. Use the stronger number.

### M5. "Trees were more accurate and, apart from the TTL, harder to evade" is contradicted by the paper's own TTL-free detectors (Abstract, §5.2, §7)

§5.5 shows that when the UNSW-NB15 detectors are trained without sttl and ct_state_ttl, the strongest attack evades the CNN on 36.3%, the RF on 28.4% and the GBDT on 36.7%, so the gap "largely closed". So "apart from the TTL" works only in a narrow sense: the attacker does not touch the TTL but the model still uses it. Once the artefact is removed from the model, the GBDT is as evadable as the CNN.

Three further problems:
- The detection sets differ greatly. The TTL-free CNN detects 99.6% of sampled attacks before modification, against 71.2% with the TTL features, so the ASR denominators are not comparable. Report end-to-end evasion too.
- Only 3 seeds and no MLP are reported.
- The network union includes a white-box attack while the trees get no gradient-based or exact attack. This asymmetry favours the trees in the comparison (small here, but M4 applies).

**Fix:** state in the abstract and conclusion that on UNSW-NB15 the network–tree gap depends on the TTL features being in the model. Add the TTL-free MLP and the FPR of the TTL-free models (0.203, 0.176 and 0.138 in values.json), and report end-to-end evasion for these models.

### M6. Remaining physical-feasibility gaps in the action map (§3, Table 2, `Space.selected`)

The revision fixed the round-1 gaps but claims more than the code delivers ("every action-dependent feature is recomputed under physical constraints", §1).
1. **Fractional packets.** a_p = u · β(p + 10) is never rounded, so modified records carry non-integer packet counts (for example 2.7 packets). This feeds into smean, src_ip_bytes and the two-packet rule. Trees with thresholds between integers, and networks after the quantile transform, may react to off-manifold values. Use ceil(a_p), or round and re-check.
2. **No MTU bound.** Payload can be added without packets (a_p = 0, a_b up to β(b + 1000)), so smean′ = b′/p′ can exceed about 1,500 bytes per packet on long flows with large packets, which Ethernet does not allow. Enforce p′ ≥ ⌈b′/MSS⌉ (adding the necessary packets), or reject such candidates.
3. **Header estimate on UNSW-NB15.** See W7.

**Fix:** report the share of successful candidates that violate integrality or the MTU bound, and re-run the β = 0.5 grid with rounding and the MSS constraint. This is cheap, because it uses only the grid search on the saved models.

### M7. Uncertainty reporting is still misleading in places (§4.5, Tables 4 and 10)

- **Bootstrap CIs.** `boot_ci` resamples flows *within* each seed and averages over seeds. The interval ignores seed variance, which the paper itself identifies as the dominant source ("can be off by tens of points"). Table 4 shows "51.4% ± 7.4 [50.8–52.0]": the CI is about one twelfth of the seed SD and invites misreading. Use a two-level bootstrap (seeds, then flows), or label the CI explicitly as "flow-sampling only, conditional on the trained models".
- **Mixed baselines (R1 M6, not fixed).** Table 10 still pairs 3-seed Δ values with 5-seed undefended means. compute_values.py already computes `*_undef_matched`, so add it as a column.
- **Holm families.** See W8. With families of 4 per attack for the network–tree tests and 154/186 for the defences, readers cannot calibrate the evidence. Declare the families, remove the duplicate tests (TON-IoT `bb_both` = `bb_volume`) and consider the primary-family approach of M2.
- **One configuration per defence (R1 M5.2).** This is still untested. With ε = 0.05 in min–max space AT-all loses 0.074 macro-F1 on TON-IoT. A smaller ε, or a correctly maximised one (M1), may give a different trade-off. At least one AT-all ε and one AT-threat β_train sweep on seed 0 would show whether the ranking is robust.

---

## Minor issues

1. Table 1, Vitorino et al. (2023): "IoT intrusion data" is vague; name the dataset(s) (authors: verify against the paper). Vitorino et al. (2022) should probably also have a row, since the text describes it.
2. Figure 1 shows the grid with volume only (it should say so in the caption), and Figure 3 has no seed spread (R1 minor 12). Add error bars.
3. Table 5: the "Strongest" column is all-actions while the other UNSW-NB15 columns are volume-only (58.6% → 99.3%). Label it.
4. Table 6: the ablation and floor grids are not nested (W1), so non-monotone entries appear (UNSW-NB15 RF bytes only 5.4% > bytes + packets 4.9%; floors × 2 11.7% < floors × 1 12.0%). Explain this in a note or use nested grids.
5. Tables 4 and 7 repeat identical TON-IoT rows ("Grid, both" = "Grid, volume only"). Drop them; they also inflate the test families.
6. Queries: the adaptive search costs 1 + 40 × 16 = 641 queries per flow, and the dense grid 1,000 × 5 = 5,000 on UNSW-NB15. The "strongest attack", which is the main measure, therefore assumes several thousand queries per flow, each of which raises an alert. Report its query budget alongside the grid's median.
7. §4.2: the GBDT uses scikit-learn's internal early stopping (`validation_fraction = 0.1`); state it.
8. §4.4 AT-threat/TRADES inner maximiser: 7 sign steps of 0.05 of the box from a uniform random start, so at most 0.35 of the box from the start. State it. A weak inner maximiser may partly explain why AT-threat is only about twice the clean loop on TON-IoT.
9. §4.4 clean loop: say that the extra records are real attack flows, so the control also shifts the class prior (M2).
10. §4.4 second augmentation round: "another sample of training attack flows" is a different random draw from the same pool (`row_seed = seed + 8`), so it may overlap with the first. Say "a second random sample".
11. §5.4: "The augmented GBDT was both the most robust and the most accurate model on TON-IoT". 0.961 ± 0.002 against 0.960 ± 0.003 is a tie; say "as accurate as".
12. Abstract: "on UNSW-NB15 TTL values not used in training still evaded 30.8%–40.3% of detected flows". This figure is the strongest attack, which combines unseen TTLs with volume and timing actions. The TTL alone evades 17.8%–20.8%. Rephrase.
13. The transfer surrogate shares data and seed with the target (labelled correctly now). A surrogate trained on a disjoint subset (R1 minor 2) is still missing and would make §5.2's "best case" statement concrete.
14. Abstract: "(five seeds)" is attached to the models, but most defences have three; say "five seeds for the undefended models and main defences".
15. Declarations: "used Claude … to write and edit the manuscript text". Elsevier's policy allows AI tools to improve readability and language, not to replace authoring. Check the wording with the journal. The code disclosure belongs in the Methods or Data availability (it is in the latter, which is fine). DESIGN.md in the repository is stale (it still lists "dbytes, dload unchanged", K = 256 and TTL candidates including 0) and should be updated or marked superseded before release.
16. Dyrmishi et al. (2023): the clean-loop result bears directly on their finding that unrealistic hardening helps against realistic attacks. On TON-IoT, AT-all's −30.6 pp against the undefended model shrinks to −6.2 pp against the loop control. Discuss this in §6.1. It is one of the paper's more interesting observations.
17. §5.2: "each network was evaded more than each tree under every attack" refers to the five attacks in Table 7, not to every attack (for example floors × 0, or the β = 2 adaptive search where GBDT 60.0% is close to CNN 66.2%). Qualify.
18. Table 10 should include TTL-only ASR for the UNSW-NB15 defences (R1 minor 8). The text relies on +15.6 and +19.7 pp for AT-all and CCAT, and on 8.7% for AT-threat.
19. Length: 11 tables in about 12,000 words is dense for JISA. Tables 5, 6 and 9 could go to a supplement, with one sentence each in the text.

---

## What would most raise the score

1. **Fix the conventional reference (M1, W2).** Use a strong feature-space attack, including a mutable-only black-box search, and drop or restate the defence-ranking claim. Check whether AT-all is trained against a masked inner maximiser.
2. **Make the network-defence claims statistically and operationally sound (M2).** Compare at matched FPR, use a small pre-specified Holm family, give five seeds to the clean loop, AT-all and TRADES, and make the abstract and conclusion follow what survives.
3. **Run an exact attack on the trees (M4)** for at least one dataset, budget and subsample of flows, and use a stronger held-out action on TON-IoT. If augmentation survives an exact attack, that is the paper's lasting contribution.
4. **State the conditions in the abstract (M3, M5):** the budget floors on TON-IoT, the TTL features in the model on UNSW-NB15, and the ε and attack type behind "overstated".
5. **Correct the wrong items (W1–W9),** especially the strongest-attack definition, the "grew with the budget" sentence, and the promised prevalence-weighted ASR and cheapest-action results.
6. **Close the remaining feasibility gaps (M6):** integer packets and the MTU bound. This is cheap with the saved models.
