# Paper 6: independent audit of numbers and references, round 2

**Audited file:** `Manuscript_filled.md` (timestamp 2026-11-03 07:07). Figure and table captions were taken from `build.py`, and the figure data source from `make_figures.py`.

**How the numbers were checked:** every number was recomputed from the raw files. The sources were:
- `results_v4/<ds>/<exp>/seed<k>.json`
- random splits: `<arch>_rand<r>/seed<r mod 5>.json`
- data splits: `<arch>_split<k>/seed<k>.json`
- `results_v4/latency.json`, `clean_split0.npy` and `data_stats.json`
- `results_v2` and `results_v3`, for Table 7 and to check where the figures' data comes from

The scripts are my own and live in the scratchpad (`audit/a1.py` to `a7.py`). They do not use `values.json` or the logic in `compute_values.py`. Values close to a rounding boundary were checked at 4 to 6 decimals.

**How the references were checked:** all 38 DOIs were looked up in Crossref. The 9 references without a DOI were checked on PMLR, NeurIPS, USENIX, JMLR, arXiv or OpenAlex. Claims were checked against abstracts, the round-1 full-text notes, and one new full-text read (Dinh et al., arXiv 2403.15511v2).

## Summary

- **Numbers.** About 600 values were checked: abstract, Tables 1 to 8, all in-text statistics, the Holm family, counts, latency and model sizes. Almost all of them reproduce at the printed precision. I found **5 errors**:
  - E1 and E2: **the figures and Table 7 come from the superseded results_v3 run, not from the final run.** This is the most serious finding.
  - E3: wrong parameter-count caption on Table 2.
  - E4: the "SD at most" statement is contradicted by Table 2.
  - E5: PR-AUC and the attack detection rate are promised in the methods but never reported.
  
  There are also 10 warnings.
- **Numbers that are confirmed:**
  - Holm correction over 8 tests: 0.006, 0.009, 0.076 and 0.135, and nothing on UNSW-NB15 is significant.
  - All of Table 3, Table 6, Tables 4–5 and Table 8.
  - The data-split replicates, the duplicate-free subset and the attention analysis in §5.4.
  - The run counts: 104/120 reused runs, 15 N-BaIoT reused runs, and 1 of 90 runs at the epoch budget (best epoch 118).
- **References.** All 47 are real and their metadata is correct. Citations and list match one to one. There is **1 citation problem**: the "extended version of Dinh et al. (2024)" is quoted with numbers but is not in the reference list. There are also 5 low-severity concerns about whether a source supports the claim made.

---

## A. Number errors

| # | Location | Text says | Correct / actual | How computed |
|---|---|---|---|---|
| **E1** (high) | Figures 2 and 3 (`make_figures.py` line 12: `RES = … 'results_v3'`; PNGs dated 2026-11-01 19:39, before the v4 text of 11-03) | Fig. 2 caption: "red diamonds: semantic split (mean of five seeds)". Fig. 3 shows class attention weights. The text and tables use v4. | The figures plot **results_v3**, which had the 60-epoch cap. Their values differ from the text. Fig. 2: UNSW semantic late fusion is 0.569 (text and Table 6: 0.573); UNSW attention 0.559 (text 0.565). Fig. 3: TON dos weight 0.32 (text 0.30); TON mean 0.597 (text 0.59); UNSW mean 0.441 (text 0.46); UNSW normal 0.19 (v4 0.21). The v3 attention values are the round-1 numbers (0.60, 0.44, dos 0.32, backdoor 0.89). | Recomputed class weights and semantic and random means from v3 and v4 (`a7.py`). Fix: point `make_figures.py` at results_v4 and regenerate. |
| **E2** (high) | Table 7, "quantile" rows; Table 7 caption ("the final run a quantile transform … otherwise identical"); §5.5 ("Table 7 compares that run with the final one; nothing else differed") | UNSW quantile: 0.498 / 0.362 / 0.562 / 0.569 / 0.559. TON RF stream 1: 0.927. | The quantile rows are **results_v3**, not the final v4 run. v3 also had the 60-epoch cap, so it is not "the final one". The final (v4) UNSW values are 0.497 / 0.361 / **0.557 / 0.573 / 0.565**, which is what Table 2 says, so Table 7 contradicts Table 2 by up to 0.006. The RF stream-1 column is the untuned v2/v3 RF (0.9271). Table 2 and the §5.5 text ("the RF on the same seven features scored **0.928**") use the tuned v4 RF (0.9281), so **the same paragraph quotes 0.928 while its table shows 0.927**. The v2-against-v3 comparison itself is clean: both used the 60-epoch cap, and only the scaling differed. | `S(ds, e, results_v2/v3/v4)` means (`a5.py`). Fix: either label the row "quantile (earlier 60-epoch run)" and drop the word "final", or show v4 and say the budget also differed. In both cases make the RF number consistent. |
| **E3** (medium) | Table 2 caption (build.py): "Params = trainable parameters of the CNN on TON-IoT and UNSW-NB15 (N-BaIoT differs by less than 0.1%)" | — | (a) The column also holds the **MLP** rows, and their UNSW-NB15 counts differ from TON-IoT: MLP stream 1 27,338 against 27,466; stream 2 **27,082 against 28,874**; early 27,850 against 29,770; late 70,922 against 72,842; attention 58,188 against 60,108. (b) The N-BaIoT CNN counts differ by up to **0.19%**, not "less than 0.1%": 34,569 against 34,634 (0.19%), late 85,641 against 85,770 (0.15%), wide 86,510 against 86,613 (0.12%). | `params` field in the result files and `latency.json`. Fix: "Params (TON-IoT)" plus "UNSW-NB15 MLPs have 2–6% fewer; N-BaIoT CNNs differ by <0.2%". |
| **E4** (low-medium) | §5.1: "standard deviations over seeds are small on TON-IoT (at most 0.001) and larger on UNSW-NB15 (up to 0.011, for early fusion)" | at most 0.001 / up to 0.011 | Table 2 itself shows a TON **MLP stream 1 SD of 0.002** (0.0023; early fusion with head is 0.0015). On UNSW, **MLP stream 2 has 0.019** and MLP attention 0.014. The sentence is true only for the five main CNNs (TON max 0.0012, UNSW max 0.0113). | Table 2 recomputation (`a1.py`). Fix: "for the five CNN architectures". |
| **E5** (medium) | §4.3: "we also report the benign FPR, **the attack detection rate and the macro-averaged area under the precision–recall curve**"; §4.2 explains why reused runs have no PR-AUC | Promised metrics | Neither the attack detection rate nor PR-AUC appears in any table or sentence of the results. Both are in the result files: for example, UNSW late-fusion PR-AUC is 0.635 and TON late-fusion detection rate is 0.999. | grep of Manuscript_filled.md. Fix: report them (a supplementary table) or remove the promise. |

## B. Warnings (numbers right, but wording or method open to challenge)

| # | Location | Issue |
|---|---|---|
| W1 | §6.1 "random groupings … change … the fused models by about 0.01 at most, **and not significantly**"; Table 6 and §5.3 TON late fusion "+0.001 [0.000, +0.002] (5/5)" | At full precision the TON within-seed CI is **[+0.0002, +0.0025]**, which excludes zero (one-sample t p = 0.027). The N-BaIoT CIs also exclude zero: late [+0.0001, +0.0008], p = 0.020; attention [+0.0004, +0.0018], p = 0.013. Rounding to "0.000" hides this. These tests are descriptive and outside the family, so "not significantly" is defensible only if you say "uncorrected tests outside the confirmatory family". §5.3 words it more carefully ("not significant on UNSW-NB15"). Either reword §6.1 or say that the TON CI excludes 0 but the effect is 0.001. |
| W2 | Abstract: "Replacing the semantic streams by random ones **changed late fusion by +0.001 and +0.009**" | The sign is reversed. The reported numbers are semantic minus random, so replacing semantic streams by random ones *lowered* late fusion by 0.001 and 0.009. Suggest "lowered … by 0.001 and 0.009" or "the semantic streams beat random ones by …". |
| W3 | §5.6: "RF … about **14 times** as long as late fusion" | 8.035 / 0.554 = **14.5**. "About 15" is closer; "about 14" is acceptable but borderline. |
| W4 | §5.6: "**335 kB** of 32-bit weights for late fusion" | 85,770 × 4 B = 343,080 B, which is 343 kB or 335 KiB. Use KiB, or 343 kB. The RF and GBDT "MB" figures in `latency.json` also look binary (MiB), so state the unit convention once. |
| W5 | §3.1: "RFECV … 0.554 with ten features against 0.526 with all 40" | Hard-coded and not a placeholder. I found no result file in `Papers pipeline` that contains these CV scores, so they could not be verified. Add them to `data_stats.json` or `values.json`. |
| W6 | §6.3: "Five seeds and six data splits give limited power for differences below about **0.005** macro-F1" (hard-coded) | This holds only for TON-IoT, where the late − early CI half-width is ≈ 0.0015. On UNSW-NB15 the half-width is ≈ 0.015, and the Holm-corrected tests miss +0.008 to +0.016 effects. This was flagged in round 1 (C12) and is still unchanged. Suggest "about 0.005 on TON-IoT and 0.015 on UNSW-NB15". |
| W7 | Table 1, UNSW stream 1: "bytes, load **and mean size per direction**, ACK delay" | The stream holds `sbytes, dbytes, sload, dload, smean, ackdat`. The mean size is source-only (`smean`; there is no `dmean`). Suggest "mean source packet size". |
| W8 | §5.4: "the DoS class gets the lowest stream-1 weight" | True for the seed mean and in 4 of 5 seeds; in seed 4 the lowest is xss. §5.4 also writes "lowest -0.17" with a hyphen; it should be a minus sign. |
| W9 | §4.3 and §5.2: Holm family and tests | This now matches what was computed: 8 paired t-tests, 4 comparisons × TON-IoT and UNSW-NB15. Recomputed Holm p values: TON late−early **0.0056**, TON late−wide **0.0088**, UNSW late−late_all **0.0755**, and 0.1354 for the other five. All printed values are correct. The round-1 M3 caveat still applies: with n = 5, a t-test is the only test that can reach p < 0.0625, and test-set sampling error is not included. |
| W10 | Early fusion on random column orders (§5.3) | `early_perm<k>` uses seed k, so order and seed are confounded. The reported "SD 0.015 over orders against 0.011 over seeds" therefore includes seed noise. Say so. |

## C. Values verified (selection; all reproduce)

- **Abstract:** +0.044 / +0.060 (early − better stream); RF +0.032 / +0.061 (0.0325, 0.0606); late +0.005 (Holm 0.0056), +0.016 (Holm 0.135); random +0.001 / +0.009; 0.028.
- **Table 1 and §3:** all counts; 0.1% (0.127%) and 16.3% duplicate test records; clean_n 27,498 and 13,308; N-BaIoT test set fully clean; MITM 156; worms 19.
- **§4:** parameter counts 34,634 / 85,770 / 73,036 / 86,613 / 43,594; ratio 2.48 ("about 2.5"); 1 of 90 runs at budget, best epoch 118 (N-BaIoT s2, seed 4); 104 of 120 random-split runs and 15 N-BaIoT semantic runs reused from v3 (119 files with `reused_from`). Every reused run stopped before epoch 60 with best = epochs − 10. Data splits use seed = split index with the original features.
- **Table 2:** all 57 means, SDs and FPRs, including borderline values: early_head SD 0.00149, RF N-BaIoT s2 0.99850, GBDT N-BaIoT s2 0.99853, GBDT UNSW s1 FPR 0.15455.
- **Table 3:** all 30 cells (mean, CI, positive count) and the Welch p values (0.404, 0.157); "better single stream" is stream 1 on all three datasets.
- **§5.2 text:** MITM / worms / DoS per-class values; RF − late +0.008 / +0.034. Duplicate-free subset: TON changes < 1×10⁻⁵; UNSW late − early **+0.025 [+0.001, +0.048]** (note: 4/5 seeds, not stated). Data splits: 6/6 +0.005 [+0.004, +0.006] (upper limit 0.00646); 5/6 +0.011 [+0.001, +0.020]; attention +0.002 / +0.006; RF − late +0.006 [+0.005, +0.008] and +0.041 [+0.035, +0.046], 6/6 on both; early − best +0.045 / +0.050.
- **Tables 4 and 5:** all 140 per-class cells.
- **Table 6:** all 30 rows (mean, min–max, within-seed CI, positive and tie counts). §5.3: 8/10 and 9/10 below the semantic split; +0.049 / +0.012 and +0.076 / +0.029; late on random splits > early in 10/10 (+0.004); column orders 0.946–0.950 and 0.527–0.565; MLP +0.001 / +0.007.
- **§5.4:** 0.59 (0.55–0.63), 0.46, 0.73 (0.66–0.81); dos 0.30 and backdoor 0.88; within-class SD 0.14; ρ 0.16 (−0.21 to +0.32) and 0.44 (+0.38 to +0.50); permutation p 0.56 / 0.22; ablation 0.459 / 0.248 / 0.288 / 0.142 / 0.678 / 0.079; reliance–weight ρ 0.94 / 0.93 / 0.93 (p < 0.001); reliance–advantage ρ 0.03 (p 0.95) and 0.54 (p 0.12); N-BaIoT 0.38 (mirai_udpplain) to 0.99 (gafgyt_junk); between-seed ρ 0.74 (min 0.47), 0.55 (min −0.17), 0.89.
- **§5.5:** 0.028; 0.753 FPR; 0.176; best epoch ≤ 4 ([3,1,4,1,1]); majority-class macro-F1 0.0372; medians ≤ 0.00016; SD 0.0027–0.0142 (but see E2 for 0.927 against 0.928).
- **Table 8 and §5.6:** all 72 latency cells; 211 MB and 607 MB; 4,416,482 nodes; GBDT 22 / 34 MB; 114.95 ms.

---

## D. References

The sub-audit covered all 47 references. The ones new since round 1 are Apruzzese 2023, Du 2023, Ho 1998, Liang 2024, McElfresh 2023, Wiegreffe & Pinter 2019 and Sanjalawe 2025; all seven checked directly.

### D.1 Problems

| # | Severity | Issue |
|---|---|---|
| R1 | medium | **A source is quoted but missing from the list.** §2.1 quotes "the extended version of Dinh et al. (2024)" (RF 0.727 against MIAE+RF 0.728 on UNSW-NB15), and Table A.1 marks the study "FT (extended version)". That version is a different work: Dinh et al. (2024), *Multiple-input auto-encoder guided feature selection for IoT intrusion detection systems*, arXiv:2403.15511. The numbers are confirmed in its Table V: UNSW F-scores 0.614 / 0.606 / 0.720 / 0.727 for the standalone methods against 0.728 for MIAE+RF. Add it to the list and cite it there. The ICC abstract mentions only "conventional classifiers". |
| R2 | low-medium | **Attota 2021** is marked "Abs" in Table A.1, yet the row gives details that are not in the abstract: the MQTT dataset; bi-flow, uni-flow and packet views; the RF only as a stacking ensembler. Its tree-ensemble entry is "no", where the other abstract-only rows say "not in abstract". Either obtain the full text and mark it FT, or make the row follow the abstract-only convention. |
| R3 | low | **Li 2025**: "with host and network data as views". The abstract speaks of host and network *scenarios*. "Generally, but not always, better" is fair, but the view description goes slightly beyond what was read. Soften it or confirm against the full text. |
| R4 | low | **Afraji 2025** in "accuracies often near 100%": the 100% figure is binary only; multiclass is 97%. Flagged in round 1 and still unqualified. |
| R5 | low | **Sanjalawe 2025**: "selected IIoT **flow** features". The abstract says only "selected features"; drop "flow". |
| R6 | low (framing) | Al Qathrady 2024 and Ishtiaq 2025 are single-network attention models, yet they count among the "14 fusion and multi-view" papers behind the "none reported" claims. A referee may say they pad the denominator. |

### D.2 Reference table

All 47 references are OK (metadata correct; claim plausibly supported unless noted).

| Reference | Status | Note |
|---|---|---|
| AboulEla & Kashef 2026, Sensors 26(13) 4196 | OK | "modest, dataset-dependent" is in the abstract |
| Afraji et al. 2025, Computation 13(9) 222 | OK | see R4 |
| Al Qathrady et al. 2024, CAAI TIT 9(6) 1398–1411 | OK | see R6 |
| Alabbadi & Bajaber 2025, Sensors 25(12) 3693 | OK | claim rests on the round-1 full-text read |
| Alsaedi et al. 2020, IEEE Access 8 | OK | |
| Apruzzese et al. 2023, EuroS&P 592–614 (new) | OK | |
| Arp et al. 2022, USENIX Sec 3971–3988 | OK | |
| Attota et al. 2021, IEEE Access 9 | OK metadata | see R2 |
| Ayantayo et al. 2023, J Big Data 10, 167 | OK | |
| Bahdanau et al. 2015, ICLR | OK | |
| Baltrušaitis et al. 2019, TPAMI 41(2) | OK | |
| Chaabouni et al. 2019, COMST 21(3) | OK | |
| Chawla et al. 2002, JAIR 16 | OK | |
| Dinh et al. 2024, ICC 2707–2712 | OK | see R1 (extended version missing) |
| Diro & Chilamkurti 2018, FGCS 82 | OK | |
| Du et al. 2023, ICML PMLR 202, 8632–8656 (new) | OK | "uni-modal ensembles can match" is supported; optionally add the URL |
| Engelen et al. 2021, SPW 7–12 | OK | |
| Grinsztajn et al. 2022, NeurIPS 35 (DOI 10.52202/068431-0037) | OK | |
| Ho 1998, TPAMI 20(8) (new) | OK | |
| Huang et al. 2022, ICML PMLR 162 | OK | |
| Imrana et al. 2021, ESWA 185 | OK | |
| Ioffe & Szegedy 2015, PMLR 37 | OK | |
| Ishtiaq et al. 2025, Array 27 | OK | see R6 |
| Jain & Wallace 2019, NAACL | OK | |
| Kiflay et al. 2024, Array 22 | OK | abstract confirms two RFs, six flow features and 32 payload bytes |
| Lansky et al. 2021, IEEE Access 9 | OK | |
| Li et al. 2025, IEEE Access 13 | OK metadata | see R3 |
| Liang et al. 2024, ACM CSUR 56(10) (new) | OK | optionally add the article number |
| McElfresh et al. 2023, NeurIPS 36 (DOI 10.52202/075280-3337) (new) | OK | |
| Meidan et al. 2018, IEEE Pervasive 17(3) | OK | |
| Mirsky et al. 2018, NDSS | OK | |
| Moustafa 2021, SCS 72 | OK | |
| Moustafa & Slay 2015, MilCIS | OK | |
| Nagrani et al. 2021, NeurIPS 34, 14200–14213 | OK | |
| Pedregosa et al. 2011, JMLR 12 | OK | |
| Peng et al. 2022, CVPR | OK | |
| Ramachandram & Taylor 2017, IEEE SPM 34(6) | OK | |
| Sanjalawe et al. 2025, IoTCPS 5, 143–164 | OK | the 2026 DOI is correct for the 2025 volume (published online March 2026); see R5 |
| Shwartz-Ziv & Armon 2022, Inf Fusion 81 | OK | |
| Singh et al. 2026, ICISSP 396–405 | OK | abstract: XGBoost and RF reach "ideal accuracy" against 99.32% for the CNN, which supports the claim |
| Sommer & Paxson 2010, IEEE S&P | OK | |
| Ullah et al. 2024, Sensors 24(13) 4152 | OK | |
| Vaswani et al. 2017, NIPS 30 | OK | optionally add pp. 5998–6008 |
| Wang et al. 2020, CVPR | OK | |
| Wiegreffe & Pinter 2019, EMNLP-IJCNLP (new) | OK | |
| Wu et al. 2022, ICML PMLR 162 | OK | format with "(Vol. 162, pp. …). PMLR." for consistency |
| Xu et al. 2025, Sci Rep 15, 21986 | OK | |

**Citations against the list:** 47 cited works map to 47 entries, with no orphans in either direction. The one exception is the uncited Dinh extended version (R1). Table A.1's counts (8 full text, 6 abstract-only, 3 with a tree ensemble) agree with §2.1.
