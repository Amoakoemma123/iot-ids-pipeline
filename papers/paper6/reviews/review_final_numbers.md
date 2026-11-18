# Paper 6: final audit of numbers and internal consistency

**Audited:** `Manuscript_filled.md` (built 2026-11-07 00:34 from `Manuscript.md` + `values.json` of 2026-11-03 16:06), the `Highlights.docx` text, the table and figure captions in `build.py`, and `figures/fig2_random_splits.png` and `fig3_attention.png`.

**How it was checked:**

- **verify.py.** Run as is: **0 failures, 0 pending**.
- **Independent recomputation from the raw result files.** My own scripts read `results_v4/<ds>/<exp>/seed<k>.json`, `*_rand<r>`, `*_split<k>`, `early_perm<k>`, `latency.json`, `results_v2` and `results_v3`, and `data_stats.json`. They do not use `values.json` or `compute_values.py`. The following were recomputed:
  - all of Table 2 (means, SDs, FPR, params);
  - all of Table 3, the Holm family and the Welch tests;
  - PR-AUC and the attack detection rate;
  - the duplicate-free subset;
  - Table 6 rows for the CNN, RF and MLP-late models;
  - the data-split replicates and column-order ranges;
  - the attention analysis: per-seed ρ, extreme classes and seed stability;
  - Table 7 (v2 and v3);
  - latency, model sizes, epoch counts and reuse counts.
- **compute_values.py** was not run. No model was trained and no dataset was loaded.

**Freshness:**

- No file under `results_v4` is newer than `values.json`.
- The docx files and `Highlights.docx` were rebuilt after `Manuscript.md`.
- `make_figures.py` now reads `results_v4`, and the figures were regenerated on 11-03 16:01. Round-2 E1 is fixed.
- Table 7 now labels its v3 row as "quantile, first-run protocol", and its caption explains that the RF is untuned there. Round-2 E2 is fixed.
- There are no `{{…}}` or `⟦…⟧` placeholders left in the filled md, the anon md or either docx.

Overall, every number I recomputed matches the printed value at the printed precision. There is one minor exception (W1). The remaining findings are about wording: claims in the Highlights, abstract and conclusion that are stronger than the results section supports.

---

## WRONG

| # | Location | Exact quote | Correct value | Source |
|---|---|---|---|---|
| W1 (minor) | §5.1, last sentence | "The attack detection rate … is **at least 0.997** for every model on TON-IoT" | The minimum is **0.99653** (RF, stream 2). It rounds to 0.997 but is below 0.997, so "at least 0.997" is false when read literally. All CNNs are ≥ 0.998. Fix: "at least 0.996", or "about 0.997 or higher". (The UNSW-NB15 minimum of 0.968, for the GBDT, is fine at 0.96814.) | mean `attack_detection`, `results_v4/TON_IoT/rf_s2/seed0-4.json`; `TON_IoT_det_min` = f3(0.99653) |

## OVERSTATED

| # | Location | Exact quote | Problem / correct statement | Source |
|---|---|---|---|---|
| O1 | Abstract, last sentence | "On these benchmarks, semantic grouping of one record's features **added at most about 0.01** macro-F1." | Point estimates are ≤ 0.009, but the body says the upper bound is not established. §5.3 says "a grouping effect of **up to about 0.02 cannot be excluded**" (UNSW within-seed +0.009 [−0.001, +0.019]), and the Conclusion and §6.1 repeat this. §5.2 also reports late on semantic streams minus late on all features = **+0.011 [+0.004, +0.018], 5/5** on UNSW-NB15, "a hint … that the grouping itself may help". The abstract is stronger than the body. Suggest: "added about 0.01 macro-F1 or less (point estimates), with effects up to about 0.02 not excluded on UNSW-NB15". | `late` vs `late_rand*`, `late_all`, results_v4 |
| O2 | Highlight 2; Conclusion ("and attention fusion gained nothing") | "attention fusion **gained nothing**" | The abstract says "added nothing **significant**", which is correct. The data: CNN attention − early on TON-IoT is +0.0014 [−0.0001, +0.0029], 4/5, unadjusted p = 0.056; on UNSW-NB15 +0.008 [−0.004, +0.021]. With the MLP encoder, attention − early on TON-IoT is **+0.003 [+0.001, +0.005], 5/5**, an interval that excludes zero (Table 3). "Gained nothing" contradicts Table 3. Use "no significant gain" in the Highlight and the Conclusion. | Table 3; `attn`, `early`, `m_attn`, `m_early` |
| O3 | Abstract; §6.1 ("that late fusion on all features reproduced") | "a gain that late fusion with both encoders on all features **reproduced**" | Late on all features − early = +0.0038 against late − early = +0.0052, so it reproduced about 73% of the gain. Late − late on all features on TON-IoT is +0.0014 [+0.0002, +0.0026], 4/5, unadjusted p = 0.032 (Holm 0.135). The interval excludes zero at full precision, as noted for the semantic-vs-random case. Use "largely reproduced". | `late`, `late_all`, `early` TON |
| O4 | Highlight 1 | "Fusing two flow-feature streams helps **only as much as** using all features" | Late fusion beat one CNN on all features by +0.005 on TON-IoT (Holm p = 0.006, the paper's one confirmed effect) and by +0.016 on UNSW-NB15 (n.s.). The body says the gain is small and does not depend on the grouping, which does not make it zero. Suggest: "…helps little more than using all features". | Table 3 |
| O5 | §5.3, para 3 | "Two further controls **rule out** the encoder as an explanation." | On UNSW-NB15 the column-order SD (0.015) is larger than the seed SD (0.011). The order range 0.527–0.565 (0.038) exceeds every fusion effect, and order is confounded with seed (§4.2). The MLP semantic − random interval on UNSW is +0.007 [−0.004, +0.018], which is as wide as the CNN one. These controls make the encoder an unlikely explanation; they do not rule it out. Use "make … unlikely" or "do not support". | `early_perm0-4`, `m_late_rand*` |
| O6 | §5.3, para 3 | "The MLPs scored **within a few thousandths of the corresponding CNNs** (late fusion 0.953 and 0.569)." | True for late fusion (0.000 and 0.004). Read generally, it is false: **MLP stream 2 on UNSW-NB15 is 0.374 against 0.361 (+0.013)**, and MLP early is 0.552 against 0.557 (0.005). Restrict it to "The MLP fusion models scored…" or "Late MLP fusion scored within…". | Table 2 |
| O7 | §5.2, para 3; §6.1 | "On TON-IoT the advantage therefore comes mostly from the wider head, **not from the number of parameters** or from the semantic grouping." | Widening to the late-fusion parameter count recovered +0.0021 of +0.0052 (≈40%; [+0.001, +0.003], 5/5). The head recovered +0.0037 (≈71%; [+0.001, +0.006]). The two intervals overlap, and head against wide was not tested. Parameters account for a measurable part, so "mainly from the head, less from the parameter count" is supportable but "not from" is not. | `early_wide`, `early_head`, `early` TON |
| O8 | §5.3, para 4 (TON effect "small but consistent … excluding zero at full precision, unadjusted p = 0.027"); §4.2 / §6.3 ("**most** random-split networks come from the earlier run") | see quotes | For late fusion, **all 30 random-split runs (10 per dataset) are reused earlier-run results**. Every semantic late-fusion run on TON-IoT and UNSW-NB15 is new (on N-BaIoT the semantic late runs are reused too). The TON +0.0014 "consistent" difference therefore compares new runs with old runs. Same-seed reruns differ by 0.003 on average, and TON late/attention reruns by up to 0.002, which is larger than the effect. I checked for a systematic shift: the signed mean (v4 − v3) over the 51 reruns is +0.0002, and −0.0001 on TON, so no bias is evident. The claim of consistency nevertheless rests on a run mix that the text describes only as "most". State "all late-fusion random-split runs" explicitly and soften "consistent". | `reused_from` field in `late_rand*/seed*.json`; rerun pairs v3/v4 |
| O9 | §5.1 | "The other metrics **rank the models as macro-F1 does**." | This holds roughly for PR-AUC, with ties: TON early and attention PR-AUC are both 0.974 although attention has the higher F1. It does not hold for FPR. On UNSW-NB15 the GBDT has the lowest FPR (0.128 against 0.147 for the RF) but a lower F1; attention has a higher F1 than early fusion but a higher FPR (0.195 against 0.188); on TON-IoT, early and attention have the same FPR (0.016). Suggest "PR-AUC ranks the models broadly as macro-F1 does". | Table 2; `macro_pr_auc` |

## AMBIGUOUS

| # | Location | Exact quote | Issue / suggestion | Source |
|---|---|---|---|---|
| A1 | Highlight 2 | "Late fusion gained +0.005 macro-F1 on TON-IoT" | Gained over what? The baseline (early fusion, i.e. one CNN on all features) is not named. Read alone, the highlight suggests a gain over the single streams. Suggest: "Late fusion beat one CNN on all features by only +0.005 macro-F1 on TON-IoT". | Table 3 |
| A2 | §5.2 ("by +0.016 on UNSW-NB15 (5/5; p = 0.135)"; "came within +0.001 … (p = 0.135)"; "UNSW-NB15 +0.008, p = 0.135"); Table 3 caption | "p = 0.135" | These are Holm-adjusted p values but are not labelled as such each time. Meanwhile Table 3 prints unadjusted CIs that exclude zero for UNSW late − early (+0.001, +0.031), late − wide (+0.001, +0.014) and late − late on all features (+0.004, +0.018), next to "not significant" in the text. Write "Holm-adjusted p" at each mention, and add "CIs are unadjusted" to the Table 3 caption. | Holm recomputed: 0.006, 0.009, 0.076, 0.135 ×5 |
| A3 | §5.2, "Other data splits" | "Attention fusion **stayed within** +0.002 and +0.006 of early fusion." | +0.002 and +0.006 are **mean** differences over six splits. Per split, UNSW attention − early ranges from **−0.010 to +0.017**, so "stayed within" reads as a bound it does not have. Suggest: "differed from early fusion by +0.002 and +0.006 on average". | `attn_split1-5`, `early_split1-5` |
| A4 | §5.5, para 2 | "the stream-1 CNN reaches 0.904, **close to the RF**" | The RF on stream 1 scores 0.927–0.928, a gap of 0.023, which is larger than every fusion effect in the paper. Suggest "within 0.03 of the RF". | Table 7 |
| A5 | Table 6, UNSW-NB15 attention row | "(3/5, 1 ties)" | Grammar: "1 tie". The tie is a coincidental difference of 1.6 × 10⁻⁷ (seed 2: semantic 0.56025 against the mean of two random runs, 0.56025). It is correct under the 10⁻⁶ rule, but the caption could mention ties. | `attn`, `attn_rand2`, `attn_rand7` seed 2 |
| A6 | §5.2; §6.1 | "the **wider** head" | Early fusion has no hidden head layer (embedding → dropout → softmax); "early fusion with head" adds a 128-unit dense layer. "Wider" suggests a wider layer. "The extra 128-unit dense head" is clearer. | §4.1 / §4.2 architecture description |

## OK, with notes

1. **verify.py passes.** All checks pass (0 failures, 0 pending), including no unfilled placeholders, anonymity of the anon docx, and the round-2 checks.
2. **Table 2 reproduces in full.** All 19 rows × 3 datasets (mean, SD, FPR, params) reproduce from raw files. Examples: TON late 0.9528 ± 0.0005, FPR 0.0110; UNSW early 0.5566 ± 0.0113; MLP UNSW stream 2 0.3738 ± 0.0193; RF TON 0.9606; GBDT UNSW FPR 0.1284. The caption's "differ by up to 6%" is correct (MLP stream 2: 27,082 against 28,874).
3. **Table 3 and the Holm family reproduce.** All rows reproduce, as do the Holm-adjusted values (0.0056, 0.0088, 0.0755, then 0.1354 for the remaining five) and the Welch p values (N-BaIoT 0.404 and 0.157). The "half-width ≈ 0.010 against 0.001" statement is correct: the UNSW median is 0.0096.
4. **The abstract numbers reproduce.** +0.044 / +0.060; RF +0.032 (0.0325) / +0.061 (0.0606); late +0.005 / +0.016; semantic − random +0.001 / +0.009; min–max stream-1 F1 0.028; FPR 0.753; best epoch ≤ 4 (epochs [3, 1, 4, 1, 1]); majority-class macro-F1 0.037. The abstract has 235 words (limit 250). All Highlights are ≤ 85 characters (longest 78).
5. **The random-split, column-order, data-split and duplicate-free results reproduce.**
   - Table 6: the sampled rows reproduce (CNN s1/s2/late/attention, RF s1/s2 and MLP late on all datasets), as do the counts "semantic above 8 / 9 of the ten" and "N-BaIoT attention above all ten".
   - Fusion gain: +0.049 / +0.012 and +0.076 / +0.029.
   - Late on random splits > early in all ten TON splits (minimum +0.0002).
   - Column order: 0.946–0.950 and 0.527–0.565, SD 0.015 against 0.011.
   - Data splits: late − early 6/6 and 5/6; RF − late +0.006 and +0.041; early − stream 1 +0.045 and +0.050.
   - Duplicate-free: RF lead +0.008 and +0.073; UNSW late − early +0.025 [+0.001, +0.048]; 16.3% and 0.1% removed.
6. **The attention results in §5.4 reproduce.**
   - Mean ρ with stream advantage: TON 0.16 (per seed −0.21 to +0.32); UNSW 0.44 (+0.38 to +0.50).
   - Weight extremes: dos 0.30 and backdoor 0.88 (TON); mirai_udpplain 0.38 and gafgyt_junk 0.99 (N-BaIoT).
   - Seed stability: 0.74 (minimum 0.47), 0.89, and 0.55 (minimum −0.17).
   - Fig. 3 matches these values. Cosmetic: the "gafgyt scan" and "gafgyt junk" labels in Fig. 3 nearly overlap.
7. **Table 7, §4.1 and §4.2 reproduce.**
   - Table 7: v2 and v3 means match the min–max and "first-run protocol" rows exactly; the final rows match Table 2.
   - §4.1: 1 of 90 runs hit the budget (my count is 105 including the 15 reused N-BaIoT runs, which the text excludes); the latest best epoch is 118.
   - §4.2: 104 of 120 random-split runs are reused (TON 27, UNSW 40, N-BaIoT 37), as are 15 N-BaIoT main runs. Reruns: 51, 9 at the same epoch, mean |Δ| 0.003, maximum 0.018. The signed mean shift is +0.0002, so the reuse introduces no systematic bias.
8. **Latency and sizes (§5.6, Table 8) reproduce from `latency.json`.**
   - RF size: 221 MB and 636 MB (stored as MiB and converted to 10⁶ bytes, consistently).
   - Late-fusion weights: 343 kB = 85,770 × 4 bytes.
   - Ratio of RF to late-fusion time per record: 14.5.
   - GBDT: 23 / 35 MB; 114.95 ms per record on UNSW-NB15.
   - The compact-forest numbers come from the companion study (`experiments/baseline/results_v2`, pickle size). The size method is the same, so the comparison is fair. Keep them identical to what the companion paper prints.
9. **Table 1 and §3 reproduce from `data_stats.json`.** Counts, 156 MITM and 19 worms, stream-1 medians ≤ 0.0002 and SDs 0.0027–0.014 all match. One oddity, not an error: the TON-IoT validation and test sets have identical per-class counts (27,533 each). This means the cross-split guard removed no TON validation or test record.
10. **Section 2 and Appendix A are internally consistent.** 14 studies: 8 full text, 6 abstract-only, 3 with a tree comparison (Kiflay, Dinh 2024b, Singh). On N-BaIoT, both within-seed semantic − random CIs exclude zero (late +0.0005, attention +0.0011). The text calls them negligible, which is consistent with its ±0.01 margin.

---

## Counts

| Class | Count |
|---|---|
| WRONG | 1 (minor: W1) |
| OVERSTATED | 9 (O1–O9) |
| AMBIGUOUS | 6 (A1–A6) |
| OK-but-note | 10 |

**Most important before submission:**
- O1: abstract "at most about 0.01" against "up to 0.02 not excluded" in the body.
- O2: "attention fusion gained nothing" in Highlight 2 and the Conclusion, while the abstract says "nothing significant" and the MLP interval excludes zero.
- O8: all late-fusion random-split runs are reused runs, but the text says "most".
- O3 and O4: "reproduced" and "only as much as" in the abstract and Highlight 1.

Each is a one-phrase edit in `Manuscript.md` or `build.py` (HIGHLIGHTS). None needs new values.
