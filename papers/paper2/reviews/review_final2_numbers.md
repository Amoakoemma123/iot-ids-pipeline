# Paper 2: final independent numbers audit (round "final2"), 2026-11-14

**Scope.** I checked Manuscript_filled.md, Supplement_filled.md and the HIGHLIGHTS in build.py against the raw result JSONs in `Papers pipeline/experiments/paper2_adv_v2/results_v3/` and `results_r2/`. I read compute_values.py only to learn the JSON layout and definitions. All values were recomputed with my own scripts:
- `C:\Users\G15165~1\AppData\Local\Temp\claude\audit_p2\audit.py`
- `C:\Users\G15165~1\AppData\Local\Temp\claude\audit_p2\audit2.py`
- `C:\Users\G15165~1\AppData\Local\Temp\claude\audit_p2\audit3.py`

The scripts use their own seed handling, their own paired and Welch t-tests (scipy), their own Holm step-down, and their own two-level bootstrap with a different RNG seed (4,000 reps). They also check each stored ASR against its evasion bitmask, and check that the stored e2e equals 1 − detected·(1 − ASR).

**Bottom line.**
- About 650 individual values were checked: about 400 in the main text, Tables 3–9 and the abstract, about 250 in Tables S1–S14.
- **0 numeric errors.** Every value I recomputed matches at the printed precision.
- **1 labelling error:** the p-values in Table 7's tree rows.
- **4 qualitative claims are stated more broadly than the data support.**
- **4 small inconsistencies** of definition or seed set, each ≤ 0.2 pp or a matter of presentation.
- The newest additions reproduce exactly: AT-all ε = 0.1 on five seeds, the matched-FPR primary family, and the relaxed attack.

---

## A. Checks (pass/fail)

### Abstract and highlights
| # | Claim | Stated | Recomputed | |
|---|---|---|---|---|
| 1 | TON strongest, CNN / RF | 51.4 / 25.2% | 51.42 / 25.15 | PASS |
| 2 | ε = 0.005 mutable ball beats realistic on every model except UNSW trees | – | crossover ≤ 0.005 for all except UNSW RF (0.02) and GBDT (0.01) | PASS |
| 3 | Every network defence raised the benign FPR | – | paired ΔFPR from +0.011 to +0.264, all > 0 | PASS |
| 4 | TON AT-threat vs undefended, missed or evaded at matched FPR | 38.3 vs 46.9% | 38.34 vs 46.91 | PASS |
| 5 | "none significantly reduced" on TON | – | Holm p: AT-threat 0.584, MLP AT-threat 0.122; other defences go up | PASS |
| 6 | TRADES increased it | +45.8 pp | +45.81 pp, Holm p 0.00147 | PASS |
| 7 | UNSW 70.0% → 20.4–22.2% | – | 69.97 → TRADES 20.38 / AT-threat 22.24 (CNN only; see D9) | PASS |
| 8 | TON GBDT augmented 35.3 → 3.0%, at no accuracy cost | – | 35.26 → 2.97; ΔF1 +0.0008 | PASS |
| 9 | UNSW augmented trees 30.8–40.3% | – | 30.79 / 40.33 | PASS |
| 10 | Highlights 3–5 (qualitative) | – | consistent with rows 5, 6–7 and 8–9 | PASS |

### §5.1 and Tables 3–5
| # | Item | Result |
|---|---|---|
| 11–50 | Table 3, all 8 rows: macro-F1 ± SD, rare-class F1, FPR, detected | PASS (all 40 cells) |
| 51 | "detected at least 92.6%"; UNSW miss 28.8 / 38.2%; trees < 1% | PASS (92.63; 28.84 / 38.20; 0.46 / 0.60) |
| 52–115 | Table 4, all cells with SD | PASS. TON "volume only" = all actions by the stated convention; the raw volume-only key for the TON CNN is 51.28% |
| 116–123 | Table 4 95% bootstrap CIs | PASS within Monte Carlo error (largest difference 0.1 pp) |
| 124 | 1,000 further points add ≤ 4.7 / 9.8 pp | PASS (4.72 / 9.76) |
| 125 | Adaptive adds 0.6–9.5 / 2.7–13.5 pp | PASS |
| 126 | Strong PGD vs basic, +12.6 pp (TON CNN) | PASS (12.61) |
| 127 | Strong PGD 0.4–5.8 pp below the grid | PASS (−0.40 … −5.78) |
| 128 | BPDA action PGD new evasions ≤ 0.1% | PASS (maximum 0.056%, TON CCAT) |
| 129 | Tree saturation ≤ 3.8 pp TON, ≤ 10.3 pp UNSW | PASS (3.80; 10.27) |
| 130 | Integer packets / MTU change ≤ 1.1 pp | PASS (1.09, TON CNN) |
| 131 | Extended union adds ≤ 0.8 pp | PASS (0.81, UNSW GBDT) |
| 132 | UNSW RF peaks at β = 0.25 (12.9 vs 10.6 at β = 2); every other curve grows | PASS (all other curves monotone) |
| 133 | Bytes / packets / delays alone, TON CNN: 12.7 / 11.1 / 0.9 | PASS |
| 134 | Delays alone, UNSW: CNN 30.8, trees 1.0 / 2.0 | PASS |
| 135 | Floors × 0: 3.0–8.5% (TON), 5.9–42.8% (UNSW) | PASS |
| 136 | Per class: ransomware ≥ 75.5, scanning ≥ 76.8; RF 0.2–1.7; UNSW ≥ 74.0; GBDT recon 47.0, worms 48.4 | PASS |
| 137 | Median query 4th / 9th / 65th | PASS |
| 138 | Query cap 1,769 / 7,243 | PASS (arithmetic from the stated grid sizes; not in the JSON) |
| 139 | UNSW CNN 72.4% overall vs 61.5% ASR | PASS |
| 140–235 | Table 5, all 12 rows: realistic, width, relaxed, ±/increase-only at ε = 0.01/0.05, full ball, crossovers | PASS |
| 236 | Corner search 100% at ε = 0.01 on every TON model; TON CNN ε-PGD 73.8% | PASS (exactly 1.0 on every seed; 73.81) |
| 237 | Realistic interval width ≤ 4e-04 TON, 1e-01 UNSW | PASS (4.3e-04; 1.2e-01) |
| 238 | Relaxed gain: TON CNN +10.0, networks up to 16.1, RF +0.5 / +1.3 | PASS (10.04; 16.12; 0.53 / 1.34; see D4) |

### §5.2 and Table 6
| # | Item | Result |
|---|---|---|
| 239–274 | Table 6: 12 differences and 12 Holm-adjusted Welch p-values | PASS |
| 275 | Seed SD 12.7 / 10.2 / 1.4 | PASS |
| 276 | Splits: 3 of 3 on both datasets | PASS |
| 277 | Transfer 33.9 / 67.8 / 16.6 / 59.2 / 9.5 / 37.5 | PASS |
| 278 | Thresholded gap widens on TON (46.2, 62.5 vs 11.8, 16.7) | PASS |
| 279–298 | S14: 20 values | PASS |

### §5.3 and Tables 7–8 (matched FPR, primary family)
| # | Item | Result |
|---|---|---|
| 299–514 | Table 7: all 24 rows × 9 numeric cells | PASS (values). Table 7 tree-row p-values: FAIL, see D1 |
| 515 | Primary family TON: 12 differences, raw and Holm p | PASS. Holm p: 0.584, 0.363, 0.363, 0.584, 0.0015, 0.122, 0.053, 0.137, 0.020, 0.020, < 0.001, 0.002 |
| 516 | Primary family UNSW: 12 differences and Holm p | PASS (0.002, 0.055, 0.048, 0.003, < 0.001, 0.017, 0.235, 0.055, 0.717, 0.015, < 0.001, 0.002) |
| 517 | 5 of 12 and 8 of 12 significant | PASS |
| 518 | Rule effect alone: −4.9 / −12.4 / −7.7 pp | PASS (RF and GBDT on 3 seeds) |
| 519 | ≥ 93.4% missed or evaded at 1% FPR, every UNSW network | PASS (minimum 93.37) |
| 520 | TON FPR 0.015 / 0.035 / 0.044 / 0.084; AT-threat 5.5%; clean loop 26.2% | PASS |
| 521 | TRADES detects 60.9% at the matched threshold | PASS |
| 522 | Table S11: 24 rows, all differences and unadjusted p | PASS |
| 523 | "With TTL, AT-all / CCAT lost advantage; AT-threat / TRADES kept" | PASS (+0.1, +0.3 vs −53.5, −44.7) |
| 524–700 | Table 8 and S13: means ± SD, ΔF1, rare-class F1, FPR, strongest, undefended on the same seeds, Δ vs undefended, Δ vs loop, volume only | PASS |
| 701 | AT-all / CCAT ε-PGD 27.5 / 27.0; BPDA 91.4 / 84.1; UNSW corners 63.7 / 61.4 / 97.1 | PASS |
| 702 | AT-all (BPDA): F1 0.585, FPR 0.280, matched ASR 71.9% | PASS |
| 703 | S10 sweep: all 12 rows | PASS |
| 704 | **AT-all ε = 0.1, five seeds:** missed or evaded 55.2%; +8.3 pp (p 0.234); AT-threat − AT-all −16.8 pp (p 0.100); F1 0.889; FPR 0.065 | PASS (55.17; +8.26, p 0.2342; −16.83, p 0.0998; 0.8887; 0.0646). Five v3 files and five r2 files present; per seed 45.2 / 57.4 / 47.5 / 60.6 / 65.2 |
| 705 | Feature-space hardening: −30.5 vs undefended, −5.2 vs loop | PASS |
| 706 | Tree augmentation text (TON): 3.0 / 10.3; ΔF1 +0.001 / −0.001; ΔFPR 0 / 0; 1.5 vs 16.7; ×2 5.5 / 8.6; w/o delay 9.9 / 19.5; w/o bytes 16.0 / 13.0; bytes alone 11.8 / 7.9; β = 2 23.9 / 19.1 vs 60.0 / 28.7; 0.961 vs 0.960 | PASS |
| 707 | UNSW trees: 0.6 / 1.7; 1.1 / 2.2; 17.8 / 20.8; 30.6 / 39.9; 30.8 / 40.3; ΔF1 −0.003 / +0.002; ΔFPR +0.005 / 0; +10.2; ×2 24.9 / 37.7; w/o delay 10.4 / 5.2, "more than with delays" | PASS |
| 708 | Costs: rare-class ΔF1 −0.459…−0.061 / −0.120…−0.040; FPR +0.020…+0.264 / +0.011…+0.043; CCAT vs AT-all rare F1; AT-threat ΔF1 −0.029 / −0.040; trees ≤ 0.025 | PASS |

### §5.4, Table 9, Discussion and Conclusion
| # | Item | Result |
|---|---|---|
| 709–736 | Table 9: all 28 cells | PASS |
| 737 | No-TTL detectors: F1, ΔF1 (−0.021 / −0.020 / −0.021 / −0.017), FPR, detected 99.6 vs 71.2% | PASS |
| 738 | Volume-only with TTL features, seeds 0–2: 54.8 / 70.7 / 20.4 / 33.9 | PASS |
| 739 | No-TTL strongest 36.3 / 42.1 / 28.4 / 36.7; overall 36.6 / 42.7 / 28.8 / 37.1 | PASS as computed, but inconsistent with Table 9 (D3) |
| 740 | Lessons 1–5 and Conclusion numbers (3.0–8.5%, 0.128–0.198, 3.0%, 30.8–40.3%) | PASS. Qualitative wording: see D5, D6 |
| 741 | Supplement S1, S2, S3, S6, S8, S9, S12 (spot and full rows) | PASS |
| – | Secondary family sizes 118 / 236 | NOT independently recomputed. They match values.json; my scripts do not rebuild the list of secondary tests |

---

## B. Discrepancies by severity

### Medium: fix before submission

**D1. Table 7: the tree-row p-values are labelled "Holm p" but are unadjusted secondary p-values.**
- **Where:** Table 7, column "Δ missed or evaded (Holm p)".
- **Stated:**
  - TON RF augmented −3.3 pp (0.011); TON GBDT augmented −15.3 pp (0.019).
  - UNSW RF augmented −7.9 pp (0.070); UNSW GBDT augmented −17.2 pp (0.026).
- **Recomputed:** these are raw paired p-values, 0.0105 / 0.0193 / 0.0701 / 0.0265, from op_paired with no adjustment. The tree tests that belong to the primary family use the strongest-attack ASR, not this endpoint: TON Holm p < 0.001 and 0.002, UNSW < 0.001 and 0.002.
- **Effect:** a reader takes 0.011 / 0.019 / 0.026 as Holm-adjusted primary results.
- **Fix:** mark the tree-row p-values as unadjusted (for example with a dagger), or rename the column.

**D2. §5.3, "on UNSW-NB15 every AT-threat configuration was evaded less than every AT-all configuration" holds only with all actions.**
- **Stated:** the sentence above (Table S10, seed 0).
- **Recomputed:**
  - All actions: true. Matched ASR: AT-threat 81.1 / 60.4 / 59.1% vs AT-all 94.8 / 100 / 100%. Strongest attack: also true.
  - Volume and timing, which the paper uses as its main UNSW measure: false. Matched ASR at β = 0.25 / 0.5 / 1 for AT-threat is 43.6 / 21.5 / 28.1%. For AT-all at ε = 0.02 / 0.05 / 0.1 it is 32.2 / 26.2 / 21.9%.
- **Fix:** add "with all actions, including the TTL".

### Low: wording that is broader than the data

**D5. Lesson 1 (§6.1) and the Conclusion drop the RF increases-only exception.**
- **Lesson 1 says:** "…except the UNSW-NB15 trees at the smallest balls."
- **§5.1 says:** also "for the RF with increases only, at every ε up to 0.1". This is correct: UNSW RF increases-only reaches 2.1–2.8% at every ε, against a realistic 20.4%.
- **Also:** "smallest ε" covers two ε values for RF ± (0.005 and 0.01) and for GBDT increases-only.
- **Fix:** mirror the §5.1 qualifier.

**D6. Lesson 2, "Thresholding … lowered evasion of the undefended models, the trees most", is true on TON-IoT only.**
- **TON:** RF −12.4, GBDT −7.7, CNN −4.9, MLP −2.1 pp. The trees move most.
- **UNSW, volume only (the main measure):** CNN −2.4, MLP −3.2, RF −5.0, GBDT −2.0 pp. The GBDT moves least.
- **Fix:** say "on TON-IoT".

**D7. §5.3, "under the realistic attack the most evaded [CNNs]" (UNSW AT-all / CCAT) holds only among the defended CNNs.**
- The undefended CNN (99.3%) and AT-all (BPDA) (99.9%) are evaded more than AT-all (99.0%) and CCAT (98.4%).
- **Fix:** say "the most evaded of the defences".

**D9. Abstract: "the action-space defences cut it from 70.0% to 20.4%–22.2%" describes the CNN only.**
- The MLP with AT-threat is also an action-space defence, and it goes from 75.3% to 42.6%.
- **Fix (optional):** say "for the CNN".

### Low: inconsistencies of definition or seed set (≤ 0.2 pp or presentation)

**D3. §5.4 text vs Tables 9 and 5, for the same described quantity.**
- **Text:** the no-TTL strongest attack, "now volume and timing only", evaded 36.3% (CNN) and 42.1% (MLP). The text uses the key `strongest_b0.5`.
- **Tables 9 and 5:** "Strongest, volume only, without TTL features" is 36.1% and 41.9%. The tables use `strongest_volume_b0.5`.
- The overall figures 36.6% and 42.7% also come from the all-actions key; with the volume key they are 36.4% and 42.4%.
- RF and GBDT are identical under both keys.
- **Fix:** use one key throughout. The Manuscript.md placeholders `UNSW_NB15_*_nottl_strongest_b0.5` are the ones to change.

**D4. §5.1 relaxed gain "+10.0 pp for the TON-IoT CNN" vs Table 5, where 61.3% − 51.4% = 9.9 pp.**
- The gain is measured against the relax-file baseline, the volume-only union without the white-box PGD (51.28%). Table 5 shows the all-components strongest attack (51.42%).
- This is rounding-level. Either quote +9.9 pp or note the baseline.

**D8. §5.3 and Table 7: the MLP Δ cannot be derived from the displayed means.**
- **Stated:** "35.3% against 65.2%" with Δ −31.5 pp. The 65.2% is the five-seed undefended MLP; the paired Δ uses seeds 0–2, where the undefended MLP is 66.9%.
- **UNSW:** "42.6% against 75.3%" with Δ −36.4 pp; the undefended MLP on seeds 0–2 is 79.0%.
- The same applies to the tree rows of Table 7: five-seed versus three-seed numbers elsewhere.
- **Fix:** add a footnote, or show the undefended mean on the same seeds as Table 8 does.

**D10. §5.3 sweep: seed-0 ASR is set against the five-seed missed-or-evaded share.**
- "ε = 0.1 looked best (41.1%, an ASR)" is followed by the five-seed result of "55.2% missed or evaded".
- Like for like: seed 0 has 45.2% missed or evaded; the five-seed ASR is 53.1%.
- The conclusion ("did not hold across seeds") stands either way.

### Rounding only (no action needed)
- Table 6 and §5.2, "Holm-adjusted p ≤ 0.005": CNN − GBDT is 0.0053, printed as 0.005.
- Values sitting at exact halves: GBDT ± ε = 0.05 86.55 → 86.6; UNSW MLP relaxed 82.35 → 82.3; UNSW CNN SD 12.75 → 12.7. The printed values are within half a unit.
- "Within 3.8 pp" for the augmented trees' saturation: the augmented trees themselves gained only 0.5 pp; 3.8 pp is the undefended GBDT. The statement is true but loose.

### Provenance note (not a results error)
- Every AT-all ε = 0.1 result JSON stores `cfg.eps_train = 0.05`. This is a global constant. The ε actually used is parsed from the defence name: run_adv_v3.py lines 325–326 set `eps = float('0.1')`. So the models were trained at ε = 0.1, but anyone reading the JSON alone would be misled.
- The r2 file for seed 0 (dated 12 Nov, no `bpa` block) predates seeds 1–4 (14 Nov). The matched-threshold procedure is the same: tau is set from each seed's undefended validation FPR, and the mean matched test FPR is 0.012 against 0.0155 for the undefended model.
- Suggestion: record `eps_train_used` in the result cfg, or mention this in REPRODUCE.md.
