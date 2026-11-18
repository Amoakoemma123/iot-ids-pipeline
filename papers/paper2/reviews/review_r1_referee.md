# Referee report (round 1): JISA

**Manuscript:** "What Can an IoT Attacker Actually Change? Evasion of Compact Intrusion Detectors Under a Realistic Threat Model"
**Material reviewed:** Manuscript_filled.md (about 7,970 words), run_adv_v2.py (571 lines; I read the feature map, success definition, search, PGD, defences, augmentation and driver), DESIGN.md, values.json (spot checks), Paper2_v1_text.md (spot checks).

**Score: 5/10. Recommendation: major revision.**

---

## Summary

The paper evaluates four compact multi-class flow detectors (1D-CNN, MLP, RF, HistGBDT) on TON-IoT (25 features) and UNSW-NB15 (10 features) under a problem-space threat model. In that model the attacker can add payload bytes, packets, flow delay and ACK delay, each capped at β times the flow's own value plus a floor, and on UNSW-NB15 it can also pick one of four "benign" TTLs. Derived features are recomputed, and an attack counts as a success only on attack flows that were detected before modification. Attacks are a fixed-candidate black-box search (128 or 320 queries per flow), action-space PGD, transfer from the CNN, and a conventional L∞ PGD reference in min–max space. Defences are AT-all (ε-ball), CCAT, AT-threat (action-space PGD on attack flows), TRADES (action-space) and one round of augmentation for the trees. The main claims are:
- the ε-ball misjudges vulnerability in both directions and reverses the ranking of defences;
- trees are more accurate and harder to evade than networks;
- AT-threat is the best network defence on TON-IoT;
- one round of augmentation nearly eliminates evasion of the trees;
- the authors' earlier CCAT does not protect rare classes.

The writing is clear and unusually candid. The success definition is the right one. Releasing code with a check from text to results is commendable, and the white-box versus black-box comparison is a useful sanity check. However, I see four problems:
- **Novelty is incremental.** Most elements already appear in Apruzzese et al. (2019), Debicha et al. (2023), Alhajjar et al. (2021), Flood et al. (2024) and Dyrmishi et al. (2023), plus uncited work on realistic adversarial training for tree-based NIDS.
- **The feature map has physical-consistency gaps** that the paper says do not exist.
- **The strongest result is evaluated non-adaptively.** Tree augmentation is evaluated with the same action family, the same four TTLs and partly the same random candidate directions it was trained on.
- **Several headline claims rest on arbitrary budgets, on the TTL artefact that the paper itself discounts, or on n = 3 paired tests.**

---

## Major issues

### M1. The tree-augmentation result is evaluated non-adaptively and partly in-sample with respect to the candidate grid (§4.4, §5.4, Abstract, §6.1, §7)

The abstract and conclusion headline that "retraining the trees once … reduced attack success to 1.5% … and below 1% on UNSW-NB15", and §6.1 concludes that "a tree ensemble with adversarial augmentation was the best option we tested". In the code, `augmentation()` builds its candidates with `candidates(sp, Rj, BETA_TRAIN, 24, seed)`. The evaluation calls `candidates(sp, R, beta, 120|48, seed)` with the **same seed**. Because both use `np.random.default_rng(seed).uniform(0, 1, (n_rand, k))`, the first 24 random unit directions in the evaluation are identical to the augmentation's, and the 2^k corners are shared by construction. At β = 0.5, therefore, 32 of the 128 TON-IoT candidates (and 40 of the 64 volume candidates per TTL on UNSW-NB15) are exactly the relative actions the trees were retrained on. They differ only in being applied to test rather than training flows. The TTL options are also the same four values (31, 254, 62, 0) in both phases.

The paper does acknowledge "augmentation and evaluation use the same action family and the same TTL candidates", but it does not mention the shared candidate directions. More importantly, it still draws unqualified deployment conclusions. A fixed, non-query-adaptive candidate list is not an adaptive attack in the sense of Tramèr et al. (2020).

**Fix:**
- Re-evaluate the augmented trees with fresh candidate directions (different seed, 10^3 to 10^4 points per flow) and a finer grid.
- Add a genuinely adaptive search on the augmented model, for example coordinate or bisection search along each action, a genetic or hill-climbing search, or HopSkipJump restricted to the action box.
- Use TTL values not seen in augmentation (all observed sttl values, or 1–255).
- Run held-out-action tests: augment with bytes and packets, attack with delay, and the reverse.
- Run a second augmentation round (DESIGN.md planned "one and two rounds") and report whether the attack recovers after re-searching.
- Until then, restate the abstract and §6.1/§7 claims as "against the same fixed action family and candidate set".

### M2. Feature-consistency gaps: some recomputed features are physically impossible (§3 "Feature consistency", Table 1, `Space.selected`)

§3 states: "Every selected feature that depends on the actions is recomputed." In the code that is not the case, and some actions produce unrealisable records:

1. **UNSW `dload` is left unchanged although `dur` changes.** dload is destination bits per second over the same flow duration. The flow-delay and ACK-delay actions lengthen `dur` (`d2 = d + ad + ak`), which must lower dload. DESIGN.md lists "dbytes, dload … unchanged", so this is a design choice, but it contradicts the stated principle and hands the model an inconsistent record.
2. **ACK delay on non-TCP flows.** The ACK-delay floor (β·(ackdat + 0.1 s)) gives UDP, ICMP and other flows with `ackdat = 0` a positive ackdat, which cannot occur. The action must be masked to TCP flows that complete the handshake.
3. **Delay on single-packet flows.** Lengthening a flow needs at least two packets. Candidates with a_p = 0 and a_d > 0 on flows with p ≤ 1 (common for scans, which evade at 75–94% in Table 3) are infeasible.
4. **Packets without bytes on UNSW.** If UNSW `sbytes` counts header bytes (Argus `sbytes` normally does; please check, for example min(sbytes/spkts) on the data), then each added packet must add at least one header to sbytes. As written, `smean = (b + ab)/(p + ap)` can be lowered with zero-cost empty packets. The TON-IoT map handles header bytes for `src_ip_bytes`; the UNSW map does not.
5. **The TTL candidate 0 is infeasible.** values.json gives the "benign TTL candidates" as **31, 254, 62, 0**. A TTL of 0 is not a valid transmitted TTL (the packet is discarded at the first hop). sttl = 0 in UNSW reflects flows without observed source TTL, not a choice an attacker can make. In addition, 254 and 62 are the TTLs typical of the *attack* traffic in UNSW-NB15 (Flood et al., 2024), so calling them "benign TTL candidates" is misleading. The TTL-only and "both" figures, the defence results on UNSW and the augmentation all depend on this set.
6. **The floors dominate small flows.** For a 1-packet, 0-byte scan flow at β = 0.5, the attacker may add 500 bytes, 5 packets and 0.5 s, which is a 6-fold packet increase. "Relative budget" is then a misnomer. The very high scanning and ransomware evasion (Table 3) may be produced by the floors.

**Fix:**
- Recompute dload.
- Restrict ACK delay to TCP and delay to flows with at least two packets (or force a_p ≥ 1 when a_d > 0).
- Add the per-packet header cost to sbytes if applicable.
- Drop TTL 0 and report results per TTL value.
- Report the realised relative overhead per class and add a sensitivity analysis on the floors (for example 0, ½× and 2×).
- Correct the "every selected feature" sentence.

### M3. "Misjudged in both directions" depends on the budgets chosen and on the TTL artefact the paper itself discounts (Abstract, §5.1, §6.1, §7)

The headline is that the ε-ball "overstated vulnerability on TON-IoT … and understated it on UNSW-NB15". Three problems:

- **TON-IoT.** The comparison pits ε = 0.05 (white-box) against β = 0.5 (black-box). Both budgets are arbitrary and in different units. At β = 2 the CNN reaches 62.0%, and ε = 0.01 already gives 73.7%. Because the networks see quantile-transformed inputs, a 0.01 change in min–max space of heavy-tailed features is a very large change in both natural units and quantile space (0.01 × the range of src_bytes can be millions of bytes). The ε-ball is therefore not a "small" ball, and its saturation at ε = 0.01 is evidence of this. "Overstated" is a statement about two particular budget choices, not about the threat models.
- **UNSW-NB15.** With volume and timing only, which §3 declares "the main result on UNSW-NB15", the two are equal (63.0% vs 64.5% CNN; 65.8% vs 72.4% MLP). The "understated" half rests entirely on the TTL choice, which §3 says "exploits a dataset artefact rather than a property of the detectors". The paper cannot discount the TTL in §3 and use it as a headline in the abstract. The abstract's "choosing the TTL alone evaded 77.5%" figure also includes the infeasible TTL 0 and the attack-typical 254 (M2.5).
- **Ranking.** "the ranking reversed" / "ranked in the opposite order" is overstated. On TON-IoT the realistic ranking is AT-threat < AT-all < TRADES < CCAT, and the conventional ranking is CCAT < AT-all < … < AT-threat. The best and worst swap, but AT-all is second in both. TRADES's unconstrained ASR is not reported, and the UNSW ranking is not given. It is also unsurprising that each defence is most robust to the perturbation set it was trained on.

**Fix:**
- Plot ASR against a common, physically interpretable budget, for example the natural-unit size of the ε-ball versus β, or the ε-ball restricted to mutable features.
- State the crossover points instead of a binary "overstated/understated".
- Report the comparison with white-box versus white-box (action PGD 32.7% vs ε-PGD 74.6%) as well as black-box versus white-box.
- Rewrite the abstract so that the UNSW claim is explicitly "via the TTL artefact".
- Give the full ranking with TRADES and UNSW, and soften "reversed".

### M4. Black-box search outperforming white-box PGD is a gradient-masking symptom, not evidence against it (§4.3, §5.1)

§4.3 claims "the black-box search … checks that the piecewise-linear quantile transform does not mask gradients". In fact §5.1 reports that the white-box attack found "fewer evasions than the black-box search for both networks at every budget, by 4.5–19.7 percentage points". For AT-threat, white-box gives 0.4% against black-box 3.4%. Under Carlini et al. (2019), a black-box attack beating a white-box one indicates that the gradient attack is weak or masked. The PGD used is weak: one random start, a log-probability (not margin) loss, 50 sign steps of 5% of the box, no restarts from corners, and a differentiable quantile transform with zero gradient outside the reference range and flat segments.

The paper rightly falls back on the search as the main measure. But "close to exhaustive" is asserted without evidence: 8 corners plus 120 fixed random points in 3-D, the same relative points for every flow, no adaptivity. For axis-aligned tree regions, a thin evasive slab can be missed.

**Fix:**
- Strengthen PGD (margin/CW loss, 10 or more restarts including corner starts, step-size schedule, more steps).
- Show a saturation curve of ASR against the number of candidates (up to about 10^4) for each model, including the defended ones.
- Add one adaptive black-box search (see M1).
- Replace "checks that … does not mask gradients" with an honest statement.

### M5. The defence comparison is not budget- or training-matched (§4.4, §5.4)

1. **Training-loop confound.** The undefended networks are trained with Keras `fit` (≤100 epochs, ReduceLROnPlateau, patience 15). All defended networks are trained with a different custom loop (≤30×200 steps, no LR schedule, patience 5). Part of the clean-F1 drop (−0.028 to −0.127) and part of the ASR change may come from the loop rather than the defence. **Fix:** add a "custom loop, clean only" control (same loop, adversarial half replaced by clean attack records) and pair the defences against it.
2. **One configuration per defence.** AT-all and CCAT use ε = 0.05 in min–max space, which is very large after the quantile transform (see M3). AT-threat uses β = 0.5, and TRADES uses a weight of 6 only. A conclusion of the form "AT-threat … ahead of ε-ball adversarial training" needs trade-off curves (clean macro-F1 against ASR over ε_train ∈ {0.005, 0.01, 0.02, 0.05}, β_train ∈ {0.25, 0.5, 1} and the TRADES weight).
3. **AT-threat and TRADES are trained and evaluated on the same action family.** This is the same issue as M1 for networks. The β = 2 result is helpful, but it uses the same directions. **Fix:** held-out-action evaluation, and unseen TTL values.
4. **Text error.** §5.4 says "The three defences trained against feature-space perturbations (AT-all, CCAT and TRADES)". In §4.4 and in the code (`train_robust`, `defence == 'trades'` uses `_pgd_action`), TRADES is trained on **action-space** perturbations. This misdescribes the comparison and must be corrected, together with the interpretation that follows from it.
5. **Defences are never compared with each other directly.** "Adversarial training matched to the threat model did so most" is supported only by each defence's paired difference against the undefended model, not by a paired AT-threat vs AT-all or AT-threat vs TRADES test.
6. **AT-threat changes the denominator.** On UNSW, AT-threat lowered the pre-attack miss rate from 22.4% to 2.1%, so its ASR is computed over a different and larger set of flows. Paired ΔASR across models with different detection sets is confounded. **Fix:** make the end-to-end metric (the share of all sampled attack flows classified benign after the attack) a column in Tables 2 and 4, not one sentence in §5.1/§5.4.

### M6. Statistical support is weak, and the reporting mixes 5-seed and 3-seed baselines (§4.5, §5.2, §5.4, Tables 2 and 4)

- **Mixed baselines.** Table 4 shows undefended rows as 5-seed means (CNN TON 50.7%), while the Δ column and the text use 3-seed baselines (48.7%; also 71.1 vs 67.2, 24.7 vs 24.9, 99.1 vs 98.6, 87.0 vs 87.3). Readers cannot reconcile "14.1%, −34.6 pp" with 50.7%. **Fix:** show the 3-seed undefended means in Table 4, or add a column.
- **Seed variance is narrow.** Seeds vary only initialisation, on a single split and a fixed 5,000-flow sample. No data-sampling uncertainty is reported, and Tables 2 and 3 give no SDs or CIs for ASR. **Fix:** bootstrap CIs over flows, plus at least 3 data splits for the main comparisons.
- **The paired tests have n = 3** (df = 2). The paper's own UNSW conclusions are non-significant (p = 0.052–0.109), yet the text draws rankings from them. With about 40 tests and no correction, several of the "p = 0.05x" statements carry no weight. **Fix:** Holm correction within each family, or report effect sizes with CIs and drop the p-value language. Raise the defended seeds to 5 at least for CNN AT-threat, AT-all and the trees; tree augmentation costs very little.
- **Pooled Welch test.** The network vs tree test pools CNN and MLP versus RF and GBDT (10 vs 10 runs) and treats different models as replicates of one population. Report per-pair tests (CNN vs RF, and so on) or a mixed model with model as a factor.
- **Claims stated as "held" without significance.** §5.5 says "TRADES gives lower clean macro-F1 than CCAT: this held (0.886 against 0.893 …)". That difference is 0.007 with SDs of 0.012 and 0.003 over 3 seeds, and should not be called "held".

### M7. Novelty relative to prior work is overstated, and key related work is missing (§1, §2, contributions)

- **Restricted flow attacks with recomputed derived features and malicious-only success counting** are in Apruzzese et al. (2019) and Debicha et al. (2023).
- **Black-box attacks on tree ensembles over mutable features** are in Alhajjar et al. (2021).
- **TCP/IP-derived constraints** are in Sheatsley et al. (2022).
- **The UNSW TTL artefact** is in Flood et al. (2024).
- **Realistic versus unrealistic hardening** is in Dyrmishi et al. (2023).

The genuinely new parts are narrower: TON-IoT, a side-by-side comparison of action-space and ε-ball adversarial training for networks under one evaluation, and a cheap augmentation result for trees.

Directly relevant work is not cited:
- realistic adversarial training and hardening of random-forest NIDS (Apruzzese, Andreolini, Marchetti, Venturi & Colajanni, "Hardening random forest cyber detectors against adversarial attacks", IEEE TETCI, 2020);
- realistic constrained perturbations with adversarial training of tree ensembles for IoT intrusion detection (Vitorino, Praça & Maia, "Towards adversarial realism and robust learning for IoT intrusion detection and classification", Annals of Telecommunications, 2023; and the A2PM work in Future Internet, 2022);
- constrained adversarial attack and training frameworks (Simonetto et al., IJCAI 2022);
- the practitioner critique of gradient-based threat models (Apruzzese et al., "Real attackers don't compute gradients", SaTML 2023).

(Authors: verify each before citing.)

The paper also does not engage with the fact that its own AT-all result partly *confirms* Dyrmishi et al.: ε-ball training cut the realistic black-box ASR from 48.7% to 14.1% on TON-IoT.

**Fix:** reposition the contributions accordingly, add a comparison table against prior work (dataset, action space, recomputation, success definition, models, defences), and discuss Dyrmishi et al. explicitly.

### M8. The narrative is built around an unpublished earlier version (§1, contribution 3, §5.5, Abstract, §7)

The abstract's last sentence, §1's third paragraph, all of §5.5 and the conclusion ("the earlier vulnerability figures … are withdrawn") address a version that readers cannot access and that was never published. A journal article cannot "withdraw" unpublished results. CCAT is also not an established baseline in the literature. **Fix:**
- Remove §5.5 and the references to the earlier version from the abstract, introduction and conclusion; at most, put a short note in the cover letter or supplement.
- Motivate CCAT on its own terms (Wu et al., 2021; Xu et al., 2021) as a simple imbalance-aware baseline.
- Use the freed space for the analyses requested above.

### M9. Practical meaning of the UNSW-NB15 models and of black-box queries (§4.1, §5, §6)

- **The UNSW detectors are weak.** Macro-F1 is 0.55–0.61, and the benign FPR is 0.13–0.20 (0.198 for the CNN, 0.261 for CNN+TRADES). A detector that flags one in five benign flows would not be deployed. Robustness percentages for such models are hard to interpret, and "harder to evade" is partly a statement about which flows are detected in the first place (networks miss 22–36% of attacks before any attack).
- **The 10-feature set includes six attacker-touched features**, including both TTL features. The single most informative control is missing: retrain without `sttl` and `ct_state_ttl`, then report clean performance and volume-only robustness. That would separate the artefact from the detector far better than the current TTL ablation.
- **The query-access assumption is strong.** Each failed query is itself a detected attack flow that raises an alert, and IDS verdicts are rarely observable to the attacker. 128–320 queries per flow is therefore a strong assumption, while transfer (the realistic setting) gets little attention. Discuss this, and report the query count to first success.

---

## Minor issues

1. §1: "this model is unrealistic in two directions" is fine, but cite Apruzzese et al. (2019) and Sheatsley et al. (2022) for the "impossible changes" side.
2. §3 "Knowledge": the transfer surrogate is a CNN trained on the *same* data, features, pre-processing and seed. That is a near-white-box surrogate and should be labelled as such, ideally alongside a surrogate trained on a disjoint subset.
3. §3 "Success": the 5,000-flow sample is class-stratified (about 555 per class on TON-IoT), so the aggregate ASR is close to a class-macro average and is dominated by easy classes such as scanning and ransomware. Say so, and add a prevalence-weighted ASR.
4. §4.4 AT-threat: "keeps the TTL unchanged with probability 0.5". In the code, `rng.choice(list(sp.ttl_cands) + [None]) if rng.random() < 0.5 else None` keeps it unchanged with probability 0.6, and the TTL is drawn per batch, not per flow. Correct the text or the code.
5. §5.3: "the trees are almost never evaded on backdoor, injection, password and XSS flows (RF 0.2%–0.8%)". This is true of the RF only. The GBDT is evaded on 16.8%, 6.6%, 38.1% and 23.5% of those classes, so change "the trees" to "the random forest".
6. §5.3: "Our search records whether a flow evades, not which candidate succeeded". This is a code limitation that is trivial to remove. Report the cheapest successful action per flow, a per-action ablation (bytes only, packets only, delay only) and the distribution of the minimal cost of evasion. Reviewers will ask for this.
7. §5.1: "the realistic threat model removes a quarter to a third of the evasions" compares white-box ε-PGD with a black-box search (see M3).
8. §5.4 UNSW: the TTL-only paired reductions (−45.4 pp, p = 0.012; −51.4 pp, p = 0.009) appear only in the text. Add volume-only, TTL-only and both columns for UNSW in Table 4.
9. §4.2: the GBDT learning-rate justification refers to "a companion size study". Either cite it or remove the reference.
10. §4.3: the unconstrained reference is run only on networks. State why the trees get no feature-space reference; a decision-based feature-space attack would do.
11. Table 2: add SDs for ASR, and β = 2 and ε = 0.01 columns, since the text relies on them.
12. Figures 1 and 2 are referenced but not included in the reviewed file. Please make sure they show seed spread.
13. §6.2 should also list: the infeasible TTL 0; the dload inconsistency; ACK delay on non-TCP flows; query alerts; a single data split; and a fixed feature set selected on clean data (feature selection decides the attack surface).
14. The keywords and title promise "IoT", but UNSW-NB15 is not an IoT dataset. Qualify this in the title or abstract.
15. The abstract is dense with numbers (13 figures). Keep about 5 and state the caveats (fixed action family, TTL artefact).
16. Declarations: the AI-use statement says Claude was used "to write the analysis code and to draft the paper". Check that this wording meets JISA/Elsevier policy, which covers writing assistance; the code part may need separate disclosure in the methods.

---

## What would most raise the score

1. **Adaptive, out-of-family evaluation of the defences** (M1, M5.3): fresh and denser candidates, an adaptive query search, unseen TTLs, held-out actions and a second augmentation round. If tree augmentation survives this, it becomes the paper's real contribution.
2. **A corrected feature map** (M2): dload, TCP-only ACK delay, delay only on flows with at least two packets, header cost per packet, no TTL 0, and a sensitivity analysis on the floors.
3. **A TTL-free UNSW detector** and an honest UNSW framing (M3, M9).
4. **Matched defence comparison** (M5): a custom-loop clean control, trade-off curves over training budgets, direct defence-vs-defence tests, end-to-end miss-rate columns, and the TRADES description fixed.
5. **Better statistics** (M6): consistent 3-seed baselines, bootstrap CIs, 5 seeds for the key defences, more than one split, and Holm correction or CIs instead of p-values.
6. **Repositioning against prior work** (M7) and removal of the material about the unpublished earlier version (M8).
