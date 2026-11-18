# Paper 6 — independent numbers audit, round 1

Audited file: `Manuscript_filled.md` (as of 2026-11-01 18:41).
Method: every quantitative statement was recomputed directly from `results_v3/<ds>/<exp>/seed<k>.json` (random splits `<arch>_rand<r>/seed<r mod 5>.json`), `results_v2` (Table 7 only) and `data_stats.json`, using independent scripts (scratchpad `audit.py`, `audit2.py`). `values.json` was not used. `compute_values.py` was read and audited (section D). `⟦…⟧` placeholders and the in-progress runs (`early_wide`, `*_e120`) were ignored, apart from one forward-looking note on the Holm family (C1).

Integrity checks passed: every random-split file has `cfg.seed == r mod 5` and a stream-1 list identical to `streams.json["random"][r]`. The 10 random partitions per dataset are distinct, have the semantic sizes (7/6/30) and none equals the semantic split. Semantic runs have matching seeds and streams. RF results are identical in results_v2 and results_v3, which agrees with "nothing else differed". The v2 and v3 CNNs have the same parameter counts and the same 60-epoch budget.

## A. Count

About **590 numerical values** were checked: the abstract (14), Table 1 and §3 (≈30), §4 parameter counts, ratios and epoch counts (≈10), Table 2 (24 means, 24 SDs, 24 FPRs, 3 parameter counts), §5.1 text (12), Table 3 (18 cells × 4 numbers = 72), §5.2 text (≈20), Tables 4–5 (140 per-class F1 values and 20 test-n values), §5.3 text (≈25), Table 6 (18 rows × 5 = 90), §5.4 (≈25), §5.5 and Table 7 (≈35), and the qualitative claims below. Training hyperparameters (filters, learning rate, batch size, quantiles) cannot be checked against result files and were not counted.

**Result:** 3 wrong values or statements, 1 mislabelled column, and every other number reproduces exactly at the printed precision. Borderline roundings were checked at full precision, e.g. TON s2 backdoor 0.99499 → 0.99, UNSW s2 dos 0.07499 → 0.07, TON RF s1 random max 0.93050 → 0.931, UNSW RF s2 random min 0.36785 vs semantic 0.36803 (1/10 below is correct), N-BaIoT late FPR 0.00048 → 0.000.

## B. Wrong values

| # | Quoted text | Correct value | Source |
|---|---|---|---|
| W1 | Abstract: "On N-BaIoT **either stream alone** already scores 1.000." | Only **stream 1** scores 1.000 (CNN 0.9997, RF 0.9997). Stream 2 scores **0.994** (CNN 0.9939) and 0.998 (RF). §5.1 itself says "between 0.994 and 1.000". Suggest: "On N-BaIoT stream 1 alone already scores 1.000 and stream 2 0.994." | `N_BaIoT/s2/seed*.json`, `rf_s2` |
| W2 | §4.3: "(late against early, attention against early, late against attention, **and the two capacity comparisons, on each dataset**) form **one family of 9** paired t-tests" | The arithmetic does not work: 5 comparisons × 3 datasets = 15. Capacity runs exist only for TON-IoT, which gives 3×3 + 2 = **11**. The Holm p values printed in §5.2 (0.022, 0.538) are those of a **9-test family with no capacity tests** (3 architecture comparisons × 3 datasets). See C1 for the values under the declared family. | recomputed; `compute_values.py` l.184 |
| W3 | Table 3, N-BaIoT, "RF all features − RF better stream … **(5/5)**" | **2/5 positive, 3 ties.** Three of the five seed differences are 1.8×10⁻⁷, a floating-point residue far below one test record (≈10⁻⁵ of macro-F1). Raw differences: [4.0e-5, 1.8e-7, 1.8e-7, 8.4e-5, 1.8e-7]. Count ties with a tolerance (e.g. \|d\| < 1e-6). | `N_BaIoT/rf`, `rf_s1` |
| W4 (labelling) | Table 2 "Params" column: 34,634 / 85,770 / 73,036 | These are the **TON-IoT** counts, and UNSW has the same counts. N-BaIoT has 34,569 / 85,641 / 72,971. Label the column "Params (TON-IoT)" or add a footnote. | `params` field |

Everything else reproduces: Table 1 and the §3 counts, all of Table 2 (incl. "SD at most 0.009" = 0.0094; FPR range 0.150–0.206; fusion 0.948–0.953), all Table 3 means, CIs and positive counts except W3, Tables 4–5 (all 140 cells), Table 6 (all 90 cells), the §5.3 MWU p values (0.01, 0.25, 0.77, 0.37; exact test), +0.002/+0.006, +0.049/+0.012, +0.072/+0.029, +0.003 (10/10), and all §5.4 attention numbers (0.60 [0.58–0.62], 0.44, 0.74 [0.64–0.81], dos 0.32, backdoor 0.89, within-class SD 0.14, ρ 0.17 [−0.20, 0.33], 0.55 [0.44, 0.71], N-BaIoT 0.39/0.99, between-seed 0.67/0.25, 0.55/0.08, 0.85). The §5.5 and Table 7 numbers also reproduce (0.028, 0.753, 0.176, best epoch ≤ 4 [3,1,4,1,1], medians ≤ 0.0002, SD 0.0027–0.014). So do "13 of 75 runs capped, 8 on TON-IoT" and "about 2.5 times" (2.48).

## C. Claim problems, ranked by severity

**C1 (high): the Holm family as declared changes the reported significance.** Holm-adjusted p values recomputed from raw files:

| Comparison | raw p | family of 9 (what the text reports) | family of 11 (adds TON capacity tests, current early_wide files) |
|---|---|---|---|
| TON late − early | 0.0024 | **0.022** | 0.026 |
| TON late − attention | 0.0047 | 0.038 | 0.047 |
| UNSW late − attention | 0.0097 | 0.068 | 0.087 |
| TON late − early_wide | 0.020 | – | **0.162** |
| TON early_wide − early | 0.075 | – | 0.528 |
| UNSW late − early | 0.090 | **0.538** | 0.538 |

(The early_wide runs may still be changing; the 11-test column is only a preview.) The point matters for the pending capacity sentence: once the capacity tests join the family, **late fusion does not significantly beat the widened single network** (Holm p ≈ 0.16, raw mean +0.003, 5/5). Fix §4.3 so the stated family size matches the family actually used, and fix it in advance rather than letting it follow whichever files exist (see D1).

**C2 (high): the semantic vs random comparison claims more than the design supports, and its statistic is not declared.** §4.3 says the semantic and random splits "are compared by the position of the semantic mean among the ten random splits", but §5.3 reports Mann–Whitney p values (method concerns in M1). Reporting is also selective. The text cites TON late fusion "semantic above all ten, p = 0.01" but describes N-BaIoT only as "differ … by less than 0.001". On N-BaIoT, attention fusion on the semantic split is also **above all ten** random splits (MWU p = **0.005**), and late fusion is above 9/10 (p = 0.055). Either report all six p values or none.

**C3 (medium): the abstract presents the UNSW late-fusion gain as real.** "Late fusion added +0.005 and +0.007 over that network": the UNSW +0.007 has CI [−0.002, +0.016], 4/5 seeds and Holm p = 0.54. Add "(not significant on UNSW-NB15)" or quote only TON-IoT.

**C4 (medium): §5.2, "the fusion models differ only on the small classes".** This is true for TON-IoT, where MITM spreads 0.034 across early/late/attention fusion and every other class ≤ 0.005. It is false for UNSW-NB15: **dos** (527 test records) 0.343/0.347/0.326 (spread 0.022), analysis 0.015, fuzzers (2,378 records) 0.012, shellcode 0.011. Rephrase, e.g. "the largest differences are on the small classes".

**C5 (medium): §5.2, "falls below late fusion, although it adds a weighting module to it".** Attention fusion is not late fusion plus a module. It replaces the 128-unit fusion layer with a weighting layer and a 64-unit layer, and it has *fewer* parameters (73,036 vs 85,770, §4.1). Rephrase.

**C6 (medium): §5.4, the attention–usefulness correlation.** The numbers reproduce, but:
- each seed's class weights are correlated with the **seed-averaged** stream advantage. Pairing each seed with its own s1/s2 runs gives TON mean ρ = **0.13** (−0.30 to 0.33) and UNSW **0.49** (0.33 to 0.59).
- with 10 classes none of these is significant. For the seed-averaged weights, TON ρ = 0.20 (p = 0.58) and UNSW ρ = 0.49 (p = 0.15). "Moderate" for UNSW should carry this caveat.
- the N-BaIoT value (mean ρ = 0.37, range 0.17–0.68) is computed by the pipeline but not reported.
- "the DoS class gets the lowest stream-1 weight" holds for the seed mean and in 4/5 seeds. In seed 4 the lowest is scanning.

**C7 (low-medium): §5.5, "the validation loss diverges while training accuracy rises".** The result files store no training histories, so this cannot be verified from them. Cite a log or figure, or soften it. Best epoch ≤ 4 is verified.

**C8 (low): §3.1, "Two classes are small: MITM 156 and worms 19".** UNSW shellcode (164), backdoor (191) and analysis (194) are as small as MITM. Say "the smallest classes are …".

**C9 (low): §5.3/Table 6 display.** "semantic above all ten" for TON late fusion, yet the table shows semantic 0.953 and random max 0.953 (0.9533 vs 0.9531). The same happens for several N-BaIoT rows (e.g. attention: semantic 0.999 vs max 0.999, 10/10). Print 4 decimals in Table 6 or add a note.

**C10 (low): Table 3, N-BaIoT CIs printed as "[−0.000, +0.000]".** These are uninformative and look like zero variance. Print them ×10⁻³ or with 4 decimals, or grey the column as a ceiling case.

**C11 (low): §5.2, "the RF gains a similar amount (+0.030 and +0.060)".** On TON-IoT this is two-thirds of the CNN gain (+0.044), from a stronger single-stream baseline (RF stream 1 0.927 vs CNN 0.904). "A similar amount" is acceptable for UNSW but generous for TON.

**C12 (low): §6.3, "Five seeds give limited power for differences below about 0.005".** This depends on the dataset. The SD of the paired late−early differences is 0.0016 on TON (resolution ≈ 0.002) but 0.0073 on UNSW (≈ 0.009 at 95%). Quote both or say "on UNSW-NB15 below about 0.01".

**C13 (low): abstract and §7, "on two datasets [the weights] changed between seeds".** This rests on an arbitrary cut-off. UNSW also has a seed pair at ρ = 0.68. Consider quoting the numbers instead.

## D. Statistical-method concerns

**M1. Mann–Whitney of 5 semantic seeds against 10 random splits is pseudo-replicated and not independent.** The question is whether the semantic *grouping* beats random groupings. The unit of replication is the split, and the semantic arm has n = 1 split; its five seeds only replicate initialisation noise. The random arm mixes split variance with seed variance, because each split has one seed. On TON late fusion the seed SD is 0.0007 and the random-split SD 0.0014, so half the random spread is seed noise. The samples also share seeds: semantic seed s and random splits s and s+5 use the same seed, so the two "independent" samples are linked. MWU p = 0.01 therefore overstates the evidence. The appropriate test is the rank of the semantic split among the random splits: empirical one-sided p = (k+1)/(n+1). That gives **0.09** for TON late (and N-BaIoT attention) and 0.36 for UNSW late, and **cannot reach 0.05 with 10 splits**. Better: train every random split with all 5 seeds (or ≥ 19 splits) and compare split means, or fit a mixed model with split as a random effect. A seed-matched comparison (semantic seed s vs mean of random splits s and s+5) gives TON late p = 0.038 and UNSW 0.16, but it is still one semantic split.

**M2. The Holm family is defined by whatever result files exist.** `compute_values.py` l.184 selects the family by key substring over all paired comparisons that ran, so its size changes silently: 9 now, 11 when early_wide is included, 15 if capacity runs are added for UNSW/N-BaIoT. The family should be fixed in §4.3 and hard-coded. Including the three N-BaIoT ceiling comparisons dilutes power. That is defensible, but state it: without N-BaIoT (6 tests), UNSW late−attention becomes significant (Holm 0.039). "Late vs attention" is also logically implied by the other two comparisons; consider dropping it.

**M3. n = 5 paired t-tests.** With five pairs, normality cannot be checked, and a sign or Wilcoxon test cannot fall below p = 0.0625 (two-sided) even at 5/5. Say so explicitly, so readers do not read the 5/5 counts as significance. The seeds vary initialisation only, so the CIs ignore test-set sampling error. On TON-IoT a +0.004 difference over 27,533 test records should also get a test-set bootstrap or a McNemar test on paired predictions, to show it is beyond test-set noise.

**M4. Ties in positive-seed counts (W3).** `paired()` counts `d > 0` with no tolerance. Floating-point residues count as wins.

**M5. Rank correlation over 9–10 classes.** Several stream advantages are pure noise near the ceiling (TON backdoor +0.004, xss +0.003; N-BaIoT all ≤ 0.014), so their ranks are arbitrary and ρ is unstable. Consider excluding classes where both streams score F1 > 0.98, or use a weighted measure. Pair weights and advantages by seed (C6).

**M6. Table 6 "random splits below semantic" compares a 5-seed mean with single-seed runs.** It is not biased in expectation, but it compares quantities with different noise levels. Report the semantic per-seed range alongside, or use M1's design.

**M7. Notes on `compute_values.py`.** Its numbers agree with this audit, apart from the issues above.
- the Holm filter is a substring match (l.184). It works now (`_late_early` also catches `_late_early_wide`, as intended) but is fragile.
- `pos` counts with no tie tolerance (l.71; W3).
- the Table 2 Params column takes the first dataset that has runs (l.121; W4).
- the MWU calls (l.284, l.289) use `scipy` 'auto', which is exact here (no ties, small n) and matches the exact p values.
- `n_capped` counts `epochs == 60`, which is correct for the main runs.
- the scaling table reuses v3 RF stream-1 for both rows; verified identical to v2.
- the attention ρ uses seed-averaged advantage (l.207; C6).
- no other computational errors were found.
