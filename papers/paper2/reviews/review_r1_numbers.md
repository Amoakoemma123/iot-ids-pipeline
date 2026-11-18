# Paper 2 v2: number audit, round 1

Audited file: `Manuscript_filled.md` (2026-11-06).
Method: an independent script (`scratchpad/p2audit/audit.py`, which does not use `compute_values.py`) read every `results_v2/<dataset>/<model>_<defence>/seed<k>.json`. It computes:
- Means over seeds 0–4 for undefended models and seeds 0–2 for defended models.
- SDs with ddof = 1. This matches the manuscript; ddof = 0 would give 0.002, not 0.003, for TON GBDT.
- Paired differences over seeds 0–2, with a one-sample t-test on the differences.
- Welch's t-test for networks against trees (10 runs against 10).
- Rare-class F1: the mean of `per_class_f1` over `rare_classes`.
- Miss rate: 1 − n_detected/n_attacked.
- 'Overall' evasion: ((n_attacked − n_detected) + ASR·n_detected)/n_attacked, averaged over seeds.
- Per-class means of `bb_both_b0.5_per_class`, seed ranges, monotonicity over β, and white-box against black-box gaps.

Raw-file inventory: all 22 model/defence folders are present. Undefended folders hold seeds 0–4 and defended folders seeds 0–2, as the text states.

**Overall result.** Every number I recomputed reproduces at the stated rounding: all cells of Tables 2, 3 and 4 and every number in §5.1–5.5, the abstract and the conclusion. There are no wrong numbers in the sense of a miscomputed value. The problems are:
- one factual misdescription (E1);
- several places where a 3-seed defended value sits next to a 5-seed undefended value (M1–M4);
- several verbal claims that are stronger than the numbers support (O1–O5).

---

## ERRORS

**E1. §5.4 TON-IoT paragraph: TRADES is misclassified as a feature-space defence.**
Quoted: "The three defences trained against feature-space perturbations (AT-all, CCAT and TRADES) left the CNN evaded on 14.1–18.4% …"
Other parts of the manuscript say TRADES is trained on action-space perturbations:
- §4.4: "KL divergence between predictions on clean and action-perturbed attack flows, the perturbation found by 7 steps of action-space ascent".
- §5.4 UNSW: "The defences that train on the actions, the TTL included, … (AT-threat …, TRADES …)".
- §5.5: "TRADES computed on the attacker's actions".

Correct statement: only AT-all and CCAT are trained in the feature-space ε-ball. All the numbers in the sentence (14.1–18.4%; p = 0.001, 0.054, 0.053; ΔF1 −0.055 to −0.089) are correct.
Suggested wording: "AT-all and CCAT, trained against feature-space perturbations, and TRADES …".

No other numerical errors were found.

---

## MIXED 5-SEED / 3-SEED COMPARISONS (task 5)

**M1. Abstract: the same undefended CNN is quoted with two different baselines.**
Quoted: "(CNN: 74.6% against 50.7% of detected attacks)" and, two sentences later, "cut the CNN's attack success on TON-IoT from 48.7% to 3.4%".
- 50.7% is the 5-seed mean of `bb_volume_b0.5` for TON `cnn_none`.
- 48.7% is the mean over seeds 0–2 of the same quantity.

Both are correct, but a reader sees two different baselines for the same model. Fix: add "(seeds 0–2)" to the second figure, or say "paired over three seeds".

**M2. Table 4: the undefended rows are 5-seed means, but Δ ASR is paired against 3-seed baselines.**
So "defended value − table baseline" does not equal the printed Δ:

| Row | Defended − 5-seed baseline | Printed Δ (vs 3-seed baseline) |
|---|---|---|
| TON CNN AT-threat | 3.4 − 50.7 = −47.3 | −45.4 (baseline 48.7) |
| TON MLP AT-threat | 7.7 − 67.2 = −59.5 | −63.4 (baseline 71.1) |
| UNSW CNN AT-threat | 74.8 − 98.6 = −23.8 | −24.3 (baseline 99.1) |
| UNSW MLP AT-threat | 67.3 − 95.8 = −28.5 | −29.5 (baseline 96.8) |
| UNSW GBDT aug | 0.7 − 79.4 = −78.7 | −78.3 (baseline 79.0) |

Δ clean and rare-class F1 have the same issue, but the gaps there are small (≤0.005). The Δ values are correct. The table needs a caption or footnote saying that undefended rows are seeds 0–4 and Δ is paired over seeds 0–2, or it should also give the 3-seed baselines.

**M3. §5.4 "Robustness did not carry over": a 3-seed value is compared with a 5-seed baseline.**
Quoted: "remained nearly as vulnerable as the undefended CNN to the unconstrained attack (67.6% against 74.6% at ε = 0.05)".
- 67.6% is the 3-seed mean for `cnn_at_threat`.
- 74.6% is the 5-seed mean for `cnn_none`.
- The paired 3-seed baseline is **75.4%**: paired Δ = −7.8 pp, p = 0.57.

The conclusion does not change, and the mixing actually understates the gap. Still, it should be "67.6% against 75.4%", or say which seeds each figure uses.

**M4. §5.4 UNSW: 3-seed defended values are compared with 5-seed undefended values.**
Quoted: "AT-threat also made the CNN miss fewer attacks before any modification (2.1% against 22.4%) … over all sampled attack flows, 75.3% end up classified benign, against 98.9% without defence."
- 2.1% and 75.3% are seeds 0–2 of `cnn_at_threat`.
- 22.4% and 98.9% are seeds 0–4 of `cnn_none`.
- The paired seeds 0–2 baselines are a **23.4%** miss rate and **99.4%** overall evasion.

For comparison, the same paragraph quotes the undefended full-attack ASR as the 3-seed 99.1%. So within one paragraph the undefended CNN appears both as a 3-seed value (99.1%) and as 5-seed values (22.4%, 98.9%). Suggested correction: "(2.1% against 23.4%) … 75.3% … against 99.4% without defence".

The following comparisons use 3-seed baselines consistently and are fine:
- 48.7 → 3.4 and 71.1 → 7.7 (TON);
- 62.0% at β = 2 (TON);
- 24.7 → 9.4 (TON RF);
- 87.0 / 79.0 → 0.5 / 0.7 (UNSW trees);
- 99.1% (UNSW CNN).

---

## OVERSTATEMENTS

**O1. "Opposite order" / "ranking reversed"** (abstract, §5.4, §6.1, conclusion).
Quoted:
- Abstract: "under the conventional attack the ranking reversed".
- §5.4: "Evaluated with the conventional attack alone, the defences would have been ranked in the opposite order".
- §6.1: "it ranked the defences in the opposite order from the realistic attack".
- Conclusion: "ranked defences in the opposite order".

TON-IoT CNN, seeds 0–2:

| Defence | Black-box ASR, β = 0.5 | Unconstrained PGD ASR, ε = 0.05 |
|---|---|---|
| AT-threat | 3.4% | 67.6% |
| AT-all | 14.1% | 25.1% |
| TRADES | 14.5% | **79.3%** |
| CCAT | 18.4% | 21.6% |

- Order under the realistic attack (best first): AT-threat, AT-all, TRADES, CCAT.
- Order under the unconstrained attack: CCAT, AT-all, AT-threat, TRADES.

This is not the reverse. Only 4 of the 6 pairs flip, and AT-threat still beats TRADES under the conventional attack. The claim holds for AT-threat against the two ε-ball defences (AT-all, CCAT).

Also unreported: TRADES *raises* the unconstrained ASR above the undefended CNN (79.3% against 75.4%, paired +3.9 pp, p = 0.60).

Suggested wording: "the ε-ball-trained defences (AT-all, CCAT) looked best under the conventional attack and AT-threat looked weak, the reverse of their ranking under the realistic attack".

**O2. §5.2: "The random forest was the most robust model on both datasets (24.9% and 18.5%)."**
This is true only for volume and timing at β = 0.5. It is not true:
- at TON β = 0.1, where GBDT has 17.9% and RF 21.0%;
- for UNSW with all actions, where GBDT has 79.4% and RF 87.3% (the full attack is the one the paper's headline numbers use);
- for TTL-only, where GBDT has 8.8% and RF 33.0%.

TON β = 0.25 is essentially a tie (RF 23.8%, GBDT 24.4%). Add "under volume-and-timing actions at β ≥ 0.25", or say that GBDT is more robust under the full UNSW attack.

**O3. §5.4: "the augmented GBDT was both the most robust and the most accurate model on TON-IoT"; §6.1: "removed most of the remaining evasions at no clean cost … a tree ensemble with adversarial augmentation was the best option we tested."**
These hold at β = 0.5 (the augmentation budget). At β = 2 on TON-IoT, seeds 0–2:
- augmented GBDT: 18.4% (from 49.8%);
- augmented RF: **20.6%** (from 27.8%, a removal of only −7.2 pp, about a quarter);
- CNN with AT-threat: **10.0%**.

So at β = 2 the AT-threat CNN is more robust than either augmented tree. The text reports the "four times the budget" result for the CNN (§5.4, TON) and for the trees on UNSW, but not for the trees on TON, where it is weaker. Report the TON tree values at β = 2, or limit the "most robust" and "best option" claims to β = 0.5.

**O4. Abstract and conclusion: "below 1% on UNSW-NB15".**
This is true at β = 0.5 (RF 0.5%, GBDT 0.7%). At β = 2, GBDT is 1.6% and RF 0.9% (§5.4 reports this). The conclusion does not state the budget. Add "at β = 0.5" or "at the training budget".

**O5. §5.1: "Volume and timing actions alone evade the CNN on 63.0% and the MLP on 65.8% … close to the unconstrained attack (64.5% and 72.4%)".**
This holds for the CNN (1.5 pp). For the MLP the gap is 6.6 pp. Not wrong, but "close" fits only the CNN well.

---

## AMBIGUITIES

**A1. §5.1: "The white-box action PGD found fewer evasions than the black-box search for both networks at every budget, by 4.5–19.7 percentage points."**
The range is right only if, on UNSW, white-box PGD (which loops over the TTL choices) is compared with the black-box search with *all actions*. Gaps:
- TON: 8.2–19.7 pp.
- UNSW, against all actions: CNN 4.5–7.8 pp, MLP 14.5–18.2 pp.
- UNSW, against volume-only: white-box PGD is *higher* by 5.3–45.1 pp.

Say "than the black-box search with the same actions (TTL included on UNSW-NB15)".

**A2. §5.1: "Success grows with the budget for every model".**
This is true for the seed-mean volume-only curves (Figure 1). It is not true for:
- every seed: RF and GBDT seeds are non-monotone on both datasets;
- the MLP's all-action ASR on UNSW (98.4% at β = 1, 98.1% at β = 2);
- the CNN's white-box ASR on UNSW (94.7% at β = 1, 94.5% at β = 2).

Add "(volume-only, seed means)".

**A3. §5.2: "the networks' advantage in ASR shrinks to +13.8 pp".**
The number is correct: 97.2 − 83.4 = 13.8 pp; Welch p < 0.001. But a higher ASR is a disadvantage for the defender. Say "the networks' excess ASR" or "the gap".

**A4. Abstract: "at a clean macro-F1 cost of −0.036".**
A "cost of −0.036" reads as a gain. Use "a clean macro-F1 drop of 0.036" or "a change of −0.036".

**A5. §5.1: "On TON-IoT all four models detect at least 94.1% of the attacks and the two measures agree".**
The MLP's miss rate is 5.94%, so 94.06% detected, which rounds to 94.1%. Its ASR is 67.2% against an overall evasion of 69.1%: they agree to within about 2 pp, not exactly.

**A6. §5.5: the earlier version's ASR definition.**
The description "counted every correctly classified test record, benign records included" matches `Paper2_v1_text.md` line 125 ("fraction of correctly-classified clean samples that are misclassified"). It also matches the arithmetic: 1 − 0.0409/0.9372 = 95.6%, close to the reported 95.7%. However, v1's own Figure 3 caption (line 157) defines ASR as "the proportion of correctly classified attack samples that are misclassified". v1 is internally inconsistent here. The manuscript's reading is the one the numbers support. A footnote noting the conflict would make the text defensible against a reviewer who reads the caption.

**A7. §5.5: "Third, that the random forest is less vulnerable than the CNN to transferred attacks: this held".**
In both versions the CNN is the *source* of the transferred attack. The comparison is the RF under transfer (18.5% / 60.6%) against the CNN under the white-box attack (32.7% / 93.3%). That is fair, as in v1's Table 10, but "the CNN to transferred attacks" is loose. Say "the random forest is evaded less by attacks transferred from the CNN than the CNN itself is by the same white-box attack".

**A8. Macro-F1 field.**
The result files hold both `macro_f1` (the one used throughout) and `clean_macro_f1`, computed on a "clean" test subset (`clean_n` = 13,308 on UNSW). On TON-IoT they agree. On UNSW they differ by up to 0.04 for the trees: RF 0.603 against 0.636, augmented GBDT 0.609 against 0.648. The manuscript's "full test set" matches `macro_f1`, so there is no error. Note that the alternative would widen the tree-network accuracy gap, and a sentence in §4.5 could pre-empt a reviewer question.

**A9. §5.4: "Every defence lowered the black-box ASR" (TON).**
This is true of the means. Two of the seven reductions are borderline over three seeds: CCAT (p = 0.054) and TRADES (p = 0.053). The paragraph reports this, so it is acceptable.

---

## VERIFIED VERBAL CLAIMS (true as stated)

- "realistic threat model removes a quarter to a third of the evasions": CNN 1 − 50.7/74.6 = 32%; MLP 1 − 67.2/88.1 = 24%.
- "98.6%, more than any ε-ball we tried": the CNN's highest unconstrained ASR is 67.1%, at ε = 0.1.
- "least for the random forest" (budget growth): RF +6.5 pp on TON and +7.8 pp on UNSW; other models +22 to +36 pp.
- "Trees were more accurate and harder to evade": macro-F1 gap of −0.009 (TON) and −0.051 (UNSW); volume ASR gap of +30.9 pp and +40.3 pp; all-actions gap on UNSW +13.8 pp; all Welch p < 0.001.
- "a single-seed robustness result for a network can be off by tens of percentage points": UNSW CNN range 44.1–82.2%; MLP 48.5–84.2%.
- Transfer "fewer than with queries": true for every model and dataset.
- §5.3: "at least 93.7% and 75.2%"; "RF 0.2%–0.8%"; CNN 30.2–56.8%; MLP 51.4–88.6%; "at least 62.0% … except worms on the GBDT (41.1%)". All exact.
- AT-threat "did so most, and at the smallest clean cost among the network defences": −0.036 (CNN) and −0.037 (MLP), against −0.055 to −0.089 for the others.
- "at β = 2, four times that budget … 10.0% against 62.0%": paired, 3-seed, p = 0.001.
- UNSW: "not significant over three seeds (p = 0.109, 0.052 and 0.083)": exact.
- "no network defence prevented evasion through the TTL": the lowest all-action ASR for a network is 67.3%.
- "nearly as vulnerable": 67.6% against 75.4% (paired), p = 0.57.
- "virtually every sampled attack": augmented trees miss 0.03% (RF) and 0.01% (GBDT).
- "four times the augmentation budget": β = 2 against β = 0.5.
- CCAT rare-class F1 claims: 0.518 / 0.383 and 0.239 / 0.194; paired −0.100 and −0.070; "no more robust than AT-all": 18.4% against 14.1%, and 97.9% against 97.9%.
- "Every network defence lowered rare-class F1 (−0.090 to −0.235; −0.025 to −0.116) and raised the benign FPR (+0.032 to +0.064; +0.018 to +0.063)": exact.
- "tree augmentation changed … by at most 0.021": the largest change is +0.021, the GBDT's rare-class F1 on TON.
- §5.5 "TRADES gives lower clean macro-F1 than CCAT: this held": 0.886 against 0.893, and 0.426 against 0.475.
- §5.5 "TRADES … more robust of the two on UNSW": 69.8% against 97.9%.

## EARLIER-VERSION NUMBERS (task 4)

Checked against `Paper2_v1_text.md`:

| Manuscript | v1 source | Status |
|---|---|---|
| accuracy 93.7% → 4.1% at PGD ε = 0.01 (TON) | line 149; Table 3, line 420 (0.0409); baseline 0.9372 at line 453 | correct; described correctly as accuracy |
| ASR 95.7% | line 149; Table 3, line 420 | correct |
| ASR 91.5% on UNSW at ε = 0.3 | lines 155 and 169; Table 3, line 429 | correct; v1 attack was PGD-40 |
| F1 0.880 (TON) and 0.518 (UNSW) | lines 211, 398–399 (0.8803, 0.5182), 500 | correct; these are clean macro-F1 of the baseline CNN |
| present CNN 0.948 / 0.554; unconstrained 73.7% / 50.6% at ε = 0.01 | raw files: 0.9476 / 0.5535; 73.71% / 50.55% | correct |

The only caveat is A6, v1's two definitions of ASR.

---

## COUNT OF CLAIMS CHECKED

- Table 2: 56 cells (all match).
- Table 3: 36 cells (all match).
- Table 4: 22 rows, 136 numeric cells, including ± SDs with ddof = 1 (all match).
- Numbers in the text of the abstract, §5.1–5.5, §6.1 and the conclusion: 118 (all match their stated source quantity).
- Earlier-version numbers: 6 (all match).
- Verbal / qualitative claims: 31 (O1–O5, A1–A3, A5, A7, A9 and the verified list).

**Total: about 383 claims checked.**
- 1 error (E1, descriptive);
- 4 mixed-seed comparisons (M1–M4);
- 5 overstatements (O1–O5);
- 9 ambiguities (A1–A9);
- 0 miscomputed numbers.
