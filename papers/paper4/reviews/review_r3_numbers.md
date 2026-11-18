# Paper 4 v2 — round-3 numbers and claims audit

Audited: `Revision_v2/Paper4/Manuscript_filled.md`, with table/figure captions from `build.py` (TABLE_CAPTIONS, FIGURES) and `make_figures.py`. `Manuscript_v2.pdf` was text-extracted and compared token by token: all 419 distinct decimal numbers in the markdown appear in the PDF and the PDF contains none that the markdown lacks, so the PDF is in sync.
Ground truth: `Papers pipeline/experiments/paper4_federated/results_v2/<ds>/<exp>/seed<r>.json`, `tuned_v3.json` and `tuned.json`. `_superseded_r2` was used only to check claims about the earlier round. `_bn_superseded` was ignored.

**Method.** I used my own scripts (scratch dir), not `compute_values.py` or `values.json`:
- I re-derived every validation-selected and final test macro-F1 from `history` (earliest argmax of `val_f1`), and every local-only score from `per_client[*].history`.
- Means use a t-based 95% CI. Paired t-tests are run over shared repetitions.
- Holm step-down is applied to the family of paired tests per dataset, built independently.
- The per-client own-class gain uses the ≥ 1% rule.
- Partition statistics come from `client_class_counts`. Validation and test sizes were checked by integrality of `acc × N`.
- ε was recomputed per client from `client_sizes`, with my own step count `rounds × local_epochs × ⌈n_k/256⌉`, in two ways: (a) dp_accounting's RDP accountant; (b) my own integer-order RDP bound for the Poisson-subsampled Gaussian with the Balle et al. conversion.
- No model was loaded or trained, and no file other than this report was written.

## Summary counts

| Category | Count |
|---|---|
| Numeric values checked (7 tables ≈ 290 cells, ≈ 55 paired-test statements × mean/CI/p/Holm, prose/abstract/conclusion/captions/setup numbers) | **≈ 610** (419 distinct decimal tokens) |
| Claim-wording / consistency checks | ≈ 45 |
| Stored JSON fields vs values re-derived from history | 0 mismatches (all runs, all histories have the configured length) |
| **WRONG numbers** (stated value ≠ data) | **0** |
| **Wrong or overstated claims** (new in round 3) | **14** (3 high, 6 medium, 5 low) |
| Round-2 items (P1–P28) | 21 fixed, 4 partly fixed, 3 not fixed |

Every table cell, mean, CI, SD, p-value, Holm-adjusted p and family size (29 TON-IoT / 22 UNSW-NB15) reproduces exactly. The same holds for every ε (mean, min and max over repetitions; dp_accounting with my step count reproduces every stored per-client ε to 3 decimals, and my integer-order bound is ≥ the stored value everywhere, with equality for σ ≥ 1.5), every per-client count, partition statistic, size, round, FPR and MB figure, and every rounded "points" figure.

## WRONG numbers

None. Each number spot-checked as at risk was correct:
- 0.193→0.009, 0.151→0.017, GN 0.012/0.016, UNSW 0.024→0.009 (swing over rounds 11–50, from the `*_recal` runs).
- 78 and 52 of 100.
- 0.279 ± 0.087 and final 0.183 ± 0.081.
- 0.659 at ε = 4.4, 0.772, 11 points.
- 0.670 vs 0.847 at ε = 2.9, 18 points.
- C = 2 0.667, C = 0.5 0.649, lr 0.604/0.652.
- MITM 0.24→0.13, and 0 in all 3 repetitions from σ = 1.5.
- Shellcode 0.39.
- Majority-class F1 0.037/0.066.
- 23,754 parameters.
- 27,533/15,894 validation and test records (integrality is exact at N, off by ≥ 0.32 at N ± 1).

## Wrong / overstated / unsupported claims (round 3)

| # | Sev. | Location | Text says | Data says | Suggested fix |
|---|---|---|---|---|---|
| N1 | **High** | Abstract ("Dirichlet and device-level client partitions redrawn in each of five repetitions"); Table III caption ("five repetitions with redrawn assignments"); III-B ("taking hosts in random order and assigning each to the currently smallest client") | The device-level partition is redrawn per repetition | `natural_partition()` assigns hosts **largest first** (`argsort(-sizes)`), so only equal-size hosts are shuffled. In all five repetitions of `fedavg_nat` and `local_nat`, the client sizes are identical (33,550 / 31,645 / 31,645 / 31,644). Client 0 has identical class counts in every repetition, and the other clients differ by a few dozen records. **The natural partition is effectively fixed.** So the Table III CIs, the paired tests in V-C and "0 of 20 site–repetition pairs" reflect only initialisation and batch-order variance over **4 sites**, not partition variance. | Describe the assignment correctly (largest host first, greedy to the smallest client; effectively deterministic). Drop "redrawn" for the natural partition. Say "4 sites × 5 training seeds" and note that the CIs exclude partition variance. Alternatively, randomise the host order and rerun. |
| N2 | **High** | V-A ("we base conclusions on effects that are significant after Holm correction") vs Abstract, V-A, Conclusion | The headline findings rest on Holm-significant effects | Of the abstract's federation numbers, only +0.286 (Holm 0.026) and the natural-partition results (−0.129, Holm 0.017; +0.266, Holm < 0.001) survive Holm. The other abstract claims do not: −0.059 (Holm 0.348), −0.185 (0.147) and +0.080 (0.083). "The gap … widens" (−0.059 → −0.185) is a comparison of two non-significant gaps. In V-B, "FedAdam trails FedAvg on TON-IoT" rests on Holm 0.060 / 0.198. The paper states a rule it does not follow. | Either drop the sentence, or mark the abstract and conclusion effects as uncorrected (for example, "−0.059, 5/5 repetitions; Holm p = 0.35"). Raw sign counts help: all TON FedAvg-vs-central and FedAvg-vs-local differences are 5/5 in the same direction. |
| N3 | **High** | VI-A | "in a preliminary tuning in which FedProx and FedAdam inherited a different client learning rate, both appeared worse than FedAvg … neither difference survived matched tuning" | FedAdam's deficit on TON-IoT survived: round-2 FedAdam α = 0.5 against the (unchanged) FedAvg was −0.089, p = 0.001; it is now −0.083 [−0.118, −0.049], p = 0.003, 0/5 positive. At α = 0.1 it grew from −0.027 to −0.046. Section V-B itself says "FedAdam trails FedAvg on TON-IoT". Only the FedProx differences (TON −0.049 → −0.007; UNSW +0.034 → +0.007) disappeared. | "FedProx appeared worse on TON-IoT and better under severe skew on UNSW-NB15; neither difference survived matched tuning, while FedAdam's deficit on TON-IoT remained." |
| N4 | Med | V-C ("although each client holds more classes than at α = 0.5") | Natural clients hold more classes than α = 0.5 clients | Classes ≥ 1% per client: natural 6/7/8/8 (mean 7.25); α = 0.5 repetition 0: 9/8/7/6 (7.5), and over all five repetitions a mean of 7.85. Classes present (> 0): natural 8/9/9/10; α = 0.5 10/10/10/10 in every repetition. **False.** III-B correctly compares with α = 0.1. | Replace "α = 0.5" with "α = 0.1" (5/3/6/6 in repetition 0). |
| N5 | Med | III-E (tuning), IV-A (algorithms) | III-E: "Hyperparameters are tuned … for every algorithm" and lists grids. The abstract says "tuned hyperparameters for every algorithm". | FedLC is tuned (`tune3_fedlc_tau{0.1,0.3,1}`, seeds 100–101; selected τ = 0.3 on TON-IoT and **τ = 1.0 on UNSW-NB15, the upper edge of its grid**, `fedlc_edge: true`), but neither the τ grid, the selected values nor the edge are reported. FedLC's method (logit calibration τ·n_{k,c}^(−1/4)) is not described in IV-A at all. Grid-edge cases for FedProx and SCAFFOLD are disclosed, but this one is not. | Add FedLC to IV-A and its τ grid and choices to III-E, with the UNSW edge noted. |
| N6 | Med | VI-B and Conclusion ("Record-level DP at ε ≈ 1 is affordable for benign traffic and the most common attacks") | DP spares the most common attacks | On TON-IoT the attack classes that hold up are DDoS (13,990 training records) and DoS (10,269). Password (13,893, the second-largest attack class) falls from 0.539 to 0.178 at σ = 2.5. Backdoor (13,096) falls 0.970 → 0.815 and scanning (13,031) falls 0.923 → 0.784. V-F itself says "These include well-represented classes". The conclusion contradicts the body. | "affordable for benign, DoS and DDoS traffic but not for several attack classes, including well-represented ones". |
| N7 | Med | Abstract ("Batch normalisation's running statistics make the evaluated model swing between rounds, unlike group normalisation"); Contribution 4; Conclusion ("should be avoided") | General BN instability | Mean round-to-round swing in validation F1 is large only on TON-IoT: BN 0.193 central / 0.151 FedAvg, against GN 0.012 / 0.016. On UNSW-NB15 it is BN 0.024–0.029 central against GN 0.012, and FedAvg BN 0.009–0.014 against GN 0.006. Section V-D says the UNSW swing is "much smaller", but the abstract and conclusion do not qualify it. (Round-2 P10 residue.) | "On TON-IoT, …", or "markedly on TON-IoT, slightly on UNSW-NB15". |
| N8 | Med | V-D ("For the federated models and on UNSW-NB15 the final-round spread is similar for both normalisations") | UNSW final-round spread similar for BN and GN | UNSW centralised: final SD with BN 0.048 (n = 3) against **0.012** with GN (n = 5), four times larger. TON FedAvg α = 0.5: 0.035 vs 0.032, and α = 0.1: 0.066 vs 0.116, so similar or smaller. UNSW FedAvg α = 0.5: 0.042 vs 0.037. The claim holds for the federated models but not for the UNSW centralised model. | "For the federated models the final-round spread is similar; the UNSW-NB15 centralised BN model also varies more (SD 0.048 vs 0.012)." |
| N9 | Med | V-A ("Federation therefore improves …") / Conclusion ("Federation is clearly worthwhile for clients with skewed attack mixes") | Clear benefit under skew | On UNSW-NB15 at α = 0.5 the gain is +0.032 [−0.037, +0.100], p = 0.271, 3/5 positive. On their own classes, UNSW clients do *not* gain (−0.015 and −0.002; 12/20 pairs better alone at both α). On TON-IoT at α = 0.5 the gain has Holm p = 0.083. "Clearly" holds for severe skew on both datasets (5/5 positive) and for TON-IoT, not in general. | "worthwhile under severe skew on both datasets and on TON-IoT; on UNSW-NB15 at α = 0.5 the gain is not significant and clients gain nothing on their own classes". |
| N10 | Low | V-F ("the analysis and worms classes are near zero in every federated configuration") | Near zero everywhere | Near zero in the DP and short-schedule runs (≤ 0.06). In the 50-round runs worms is not near zero: FedAvg α = 1.0 0.200, α = 0.1 0.199, FedProx α = 0.1 0.207, N = 2 0.417 (means over repetitions), and analysis is 0.138 at α = 1.0. | "in every DP configuration" (or "on this schedule"). |
| N11 | Low | V-B ("with larger learning rates (0.1 and 0.3, …) it collapses in several repetitions, some with non-finite weights") | Evidence in released results | In `results_v2` only lr 0.1 exists. There, one repetition collapses (rep 1: selected 0.088, final 0.002, non-finite) and rep 4 ends at 0.126. The lr 0.3 runs (rep 1 and 2 non-finite, rep 3 and 4 ending at 0.066) exist only in `_superseded_r2/UNSW_NB15/scaffold_a0.1`. The sentence is true across both learning rates, but half the evidence sits in a folder labelled superseded. | Say so ("lr 0.3, from the earlier tuning round"), or keep that folder documented in the release. |
| N12 | Low | V-B ("These differences are not significant after Holm correction") | Scope unclear | The same paragraph reports SCAFFOLD − FedAvg(Adam) at TON α = 0.5, which *is* Holm-significant (0.026). Also not reported: on the natural partition SCAFFOLD trails Adam-FedAvg, −0.094 [−0.128, −0.060], Holm 0.040. | "The SCAFFOLD-versus-FedAvg-SGD differences are not significant …". |
| N13 | Low | V-A / V-C, per-client "± " values (0.049 ± 0.021, 0.068 ± 0.045, 0.136 ± 0.027, −0.015 ± 0.033, −0.002 ± 0.053) | CI over 20 client–repetition pairs | The four clients of a repetition share one FedAvg model and one partition, so the 20 pairs are not independent and the CIs are too narrow. With a fixed natural partition (N1) they are 4 sites. | Average per repetition first (n = 5), or call it the spread over pairs. |
| N14 | Low | Holm family (V-A: "29 (TON-IoT) … paired comparisons") | Family of distinct comparisons | On TON-IoT, `scaffold_a0.1_lr0.1` uses lr 0.1, which is already the tuned SCAFFOLD lr, so it replicates `scaffold_a0.1`. Two of the 29 tests (`scaffold_a0.1_lr0.1` vs `fedavgsgd_a0.1` and vs `scaffold_a0.1`) are a rerun of the same configuration. The family also includes several tests never reported (the lr 0.01/0.1 follow-ups). This makes Holm conservative, not anti-conservative. | Drop the duplicate on TON-IoT, or state the family composition. |

Minor wording, not counted:
- V-A: "the SGD-based methods reach 4–6%". The values are 4.19–6.29% on TON-IoT; FedAdam α = 0.1 (4.22%, Adam-based) is also above 4%.
- III-E: "partitions disjoint from the five reported repetitions". These are different random draws (seeds 100–101) of the same training records, not disjoint record sets. "Different" is more accurate.
- V-F: the per-class account uses the no-noise DP code as the reference (MITM 0.24), but the abstract's "10 points" uses non-private FedAvg. Against that reference MITM loses only 2.4 points at σ = 0.6 (0.158 → 0.134), so "five classes lose ≥ 10 points" holds only against the DP-code reference. Name the reference.

## Round-2 items: status

| Item | Status | Note |
|---|---|---|
| P1 tuning at the wrong client lr / FedAvg lr at grid edge | **Fixed** | `tune3_fedprox*` and `tune3_fedadam*` now have cfg.lr = 0.003, the core lr. The FedAvg grid {1e-3 … 1e-2} selects 0.003, an interior value, on both datasets. Edges for FedProx μ and UNSW SCAFFOLD are disclosed; the UNSW FedLC edge is not (N5). |
| P2 FedProx UNSW "significant gain" | Fixed | Now +0.007, p = 0.391, and the claim is gone. |
| P3 "none improves" vs FedProx gain | Fixed | — |
| P4 "conclusions do not depend on selection rule" | Fixed | Now qualified. Note: at the final round, FedProx > FedAvg on UNSW α = 0.5 (+0.015 [+0.005, +0.026], p = 0.015, 5/5) and FedAdam < FedAvg on UNSW α = 0.1 (−0.049, p = 0.032). The generic caveat covers these. |
| P5 SCAFFOLD "advantage" | Fixed | Wording now matches the data. |
| P6 "control variates add little" | Fixed | Now "do not significantly improve". |
| P7 lr effect not ruled out | Fixed | lr 0.01, 0.03 and 0.1 (plus 0.3 superseded) were run. Rep 1 is the weakest at every lr, while FedAvg-SGD scores 0.365 on rep 1, so "this partition is hard for SCAFFOLD" holds. |
| P8 "one of five collapses" | Fixed | Now "several" (see N11 on provenance). |
| P9 BN "statistically indistinguishable" (n = 3) | **Not fixed** | Still says "statistically indistinguishable". FedAvg α = 0.5 BN − GN is +0.040 [−0.039, +0.119], with BN higher in 3/3 repetitions. |
| P10 BN instability stated generally | Partly | V-D now restricts the predictability claim. The abstract, contribution 4 and conclusion are still unqualified (N7), and V-D has a new overstatement (N8). |
| P11 batch-statistics evaluation not in results | Fixed | `*_recal` runs, `val_f1_recal`. |
| P12 UNSW N-scaling not in table, "steadily/similarly" | Fixed | Table V added; UNSW is hedged. |
| P13 Table III caption repetition count | Fixed | Now the Table IV caption: "N = 4 row … five; the other rows three". |
| P14 DP cost "a few minority classes" | Partly | The body is fixed ("five of nine … well-represented"). VI-B and the conclusion still say DP is "affordable for … the most common attacks" (N6). |
| P15 MITM at weakest noise | Fixed | — |
| P16 UNSW generic/recon "almost as well" | Fixed | — |
| P17 "matches" non-private | Fixed | "close to", with the dropout and sampling differences noted. |
| P18 "13 points", reference unnamed | Fixed | "about 10 points … against non-private training on the same schedule" (10.2). |
| P19 ε ≈ 1 vs 0.8 | Fixed | — |
| P20 heterogeneity monotonic incl. IID | Fixed | "Under label skew … α = 1.0 costs no more than IID". |
| P21 test-set selection inflation / "within the spread" | Fixed | The inflation claim was removed. |
| P22 multiple testing | Partly | Holm is applied and reported throughout, but the paper's own rule "conclusions on Holm-significant effects" is not followed (N2). |
| P23 "p = <0.001" | Fixed | — |
| P24 local models selected on pooled validation | Fixed | Now stated in III-D. |
| P25 centralised "round" units | Fixed | In the Table I caption. |
| P26 TON DP JSON `cfg.norm = 'bn'` | **Not fixed** | All 18 TON `dp_*` files still record `cfg.norm: 'bn'` with run-level `norm: 'gn'` (the UNSW files are consistent). Release metadata only. |
| P27 FedAvg-SGD grid / near-ties | Partly | The grid is now stated. Near-ties are not mentioned: TON FedAvg-SGD 0.714 (0.3) vs 0.705 (0.03); UNSW FedLC 0.451 (1.0) vs 0.442 (0.3). |
| P28 abstract "five repetitions" though DP/BN use three | **Not fixed** | The abstract still says only "five repetitions". It now also wrongly includes the device-level partition among the redrawn partitions (N1). |

## Verified (selected)

- **Tables I–III**: 17 + 17 + 8 rows. Selected and final mean ± h, SD, mean selected round (rounded: 40.8→41, 45.6→46, 39.8→40, 41.2→41; UNSW 36.8→37, 31.2→31, 38.6→39, …) and benign FPR all match.
- **Tables IV–V**: all 14 rows, including 0.38/0.76/1.52/3.04 MB (380,064 to 3,040,512 B) and BN 0.77 MB (766,272 B).
- **Tables VI–VII**: σ, ε mean (min–max), selected and final all match. ε is independently reproduced.
- **III-A–C, IV-C**: 128,484 / 74,167 (the client sizes sum to these); 27,533 / 15,894; 25 / 10 features (raw MB / n / 4 B); 23,754 parameters = 95.0 kB, the same on both datasets; 0.76 MB / 38.0 MB / 12.8 MB; SCAFFOLD ×2.
- **Partitions**: α = 0.1 repetition 0 has 93/98/72/57% and 5/3/6/6 classes. Natural: 33,550 / 31,645 / 31,645 / 31,644, 64/51/31/37%, and 6/7/8/8 classes.
- **III-E**: tuned values match `tuned_v3.json` (FedAvg 0.003/0.003; μ 1e-4/1e-4; server lr 0.03/0.03; SCAFFOLD 0.1/0.03; FedAvg-SGD 0.3/0.03). The grids match TUNE3_A/B. Every core run's cfg uses these values. "Ten times larger" matches the single-repetition SCAFFOLD 0.3 against 0.03.
- **V-A to V-F paired statements** (≈ 55): every mean, CI, p and Holm p matches. So do the r100 and no-ports ablations, which sit outside the Holm family as the code intends.
