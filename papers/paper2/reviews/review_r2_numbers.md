# Paper 2 v3: number audit, round 2

Audited file: `Manuscript_filled.md` (10 Nov 2026, 01:34), plus `HIGHLIGHTS` in `build.py` and the table captions in `build.py`.
Raw data: `Papers pipeline/experiments/paper2_adv_v2/results_v3/<dataset>/<model>_<defence>/seed<k>.json` and `attack_set*.json`.
Skipped as pending: the §5.2 sentence with ⟦…_splits_nets_above_trees⟧ and the split-1/split-2 rows of Table 8.

## Method

I wrote my own scripts and did not import `compute_values.py`. They are in
`C:\Users\G15165~1\AppData\Local\Temp\claude\bf2fe0286de83ad4\6eeb657b-c718-4aa2-9a76-d31c9b1894d8\scratchpad\p2audit_r2\`:
- `audit2.py` rebuilds Tables 3–11 and diffs them cell by cell against the manuscript.
- `text_check.py` prints every quantity quoted in the text.
- `t9check.py` checks the source of Table 9 and the bootstrap noise.
- Outputs are in `out\`.

I read `compute_values.py` once, and only to confirm what the Holm family contains, because §5.4 does not list it.

**Conventions**
- **Seeds.** Seeds 0–4 for `*_none`, `cnn_at_threat`, `rf_augment` and `gb_augment`. Seeds 0–2 for all other defences and for `*_nottl`. The inventory of the result folders matches this exactly (`cnn_at_threat`, `rf_augment` and `gb_augment` each have seed0–4).
- **SD.** ddof = 1.
- **Rare-class F1.** The mean of `per_class_f1` over `rare_classes`.
- **ASR.** As defined in `run_adv_v3.py`: the share of detected attack flows (`n_detected`) classified benign.
- **'Overall' evasion.** 1 − (n_detected/n_attacked)·(1 − ASR), averaged over seeds.
- **Paired tests.** One-sample t-tests on the differences, over the shared seeds.
- **Holm family (defences).** Per dataset, every paired defence test:
  - each defended model against its undefended model, for macro-F1 and 10 ASR keys where present;
  - CNN defences against the clean loop, for 3 ASR keys;
  - the 8 defence-against-defence pairs, for 6 ASR keys.

  This gives **154 tests (TON-IoT) and 186 (UNSW-NB15)**, the counts the text states.
- **Table 7.** Welch's t-test with Holm correction within dataset × attack (4 tests), as the caption says.
- **Bootstrap CIs.** Resampling flows within each seed, from `ev_strongest_b0.5`.

## Overall result

- Tables 3, 4 (point values and SDs), 5, 6, 7, 8 (main-split rows) and 10 reproduce **exactly**, cell for cell.
- Table 11 reproduces exactly.
- Table 9 reproduces exactly, **but from a different quantity from the one its caption and §5.3 name** (M1 below).
- The Table 4 CIs agree to within bootstrap Monte-Carlo noise (B1).
- Every number in the abstract, §5.1–5.5, §6 and §7 reproduces, except the verbal claim in M2.
- The p-values are correct as labelled, both unadjusted and Holm-adjusted.

---

## MISMATCHES

### M1. Table 9 and §5.3 are labelled grid search but use the strongest attack
**Manuscript text.** The Table 9 caption (build.py, '5.3') reads: "Grid-search ASR per attack class at β = 0.5 (volume, timing and, on UNSW-NB15, TTL)". §5.3 reads: "Table 9 gives the grid-search ASR per attack class at β = 0.5 with all actions."

**What I found.** All 72 data cells of Table 9 equal the seed means of `strongest_b0.5_per_class`, the union of all attacks. They do not match `bb_both_b0.5_per_class`, the grid search.

**Effect.** Every number in the §5.3 text inherits this:

| Claim in §5.3 | Manuscript (= strongest) | Grid search, all actions (`bb_both_b0.5_per_class`) |
|---|---|---|
| ransomware evaded every model on at least | 75.5% | 45.9% |
| scanning, at least | 76.8% | 73.1% |
| RF on backdoor/injection/password/XSS | 0.2%–1.7% | 0.2%–0.5% |
| GBDT on the same classes | 10.1%–57.4% | 5.8%–38.0% |
| UNSW: networks and RF, every class at least | 74.0% | 65.6% (RF exploits) |
| UNSW GBDT reconnaissance / worms | 47.0% / 48.4% | 31.8% / 29.5% |

The largest per-cell gaps are:
- TON GBDT ransomware: 75.5 against 45.9.
- UNSW GBDT exploits: 63.3 against 36.1.
- UNSW GBDT fuzzers: 90.5 against 67.7.

Source: `results_v3/*/{cnn,mlp,rf,gb}_none/seed0-4.json`, keys `strongest_b0.5_per_class` and `bb_both_b0.5_per_class`.

**Fix.** Change the caption and the §5.3 sentence to "ASR of the strongest attack per attack class". The alternative is to replace the table and the text numbers with the grid-search values above. The rest of the paper uses "strongest" as its main measure, so relabelling is the simpler fix.

### M2. §5.1: "Success grew with the budget for every model (Figure 1)" is false for the UNSW-NB15 random forest
Figure 1 shows the volume-and-timing grid, five-seed means. The UNSW RF falls after β = 0.25:

| β | 0.1 | 0.25 | 0.5 | 1 | 2 |
|---|---|---|---|---|---|
| UNSW RF `bb_volume_b*` | 8.4% | 12.9% | 12.0% | 11.2% | 10.6% |

The other seven model/dataset curves are increasing. In round 1 (v2 results) this claim held for the seed means; with the v3 results it no longer does. Table 7 already shows the effect: the β = 2 RF volume-only value is 10.6%, below its 12.0% at β = 0.5.

Source: `results_v3/UNSW_NB15/rf_none/seed0-4.json`.

**Fix.** Say "for every model except the UNSW-NB15 random forest, whose volume-only ASR peaked at β = 0.25", or "for every model on TON-IoT and the networks and GBDT on UNSW-NB15".

---

## MINOR / NOT ERRORS (for the authors' judgement)

### B1. Table 4: 95% CIs
Four of the eight CIs differ from my bootstrap by 0.1 in one bound:

| Row | Manuscript | Mine |
|---|---|---|
| TON CNN | 50.8–52.0 | 50.8–52.1 |
| TON MLP | 64.2–65.4 | 64.2–65.5 |
| TON GBDT | 34.7–35.9 | 34.6–35.8 |
| UNSW RF | 89.3–90.0 | 89.3–90.1 |

Re-running with 4 RNG seeds and 1,000 resamples moves these bounds by 0.02–0.07, across the rounding boundary. A normal approximation gives 50.80–52.04, 64.22–65.43, 34.67–35.86 and 89.31–90.06. The printed values are consistent with these: this is Monte-Carlo noise, not an error.

### B2. Mixed 3-seed and 5-seed comparisons that remain
They do not change any conclusion.
- **§5.4, ranking paragraph.** "AT-threat (77.9%) and TRADES (83.8%) … no better than the undefended CNN (75.1%)."
  - 75.1% is the five-seed undefended value. TRADES (and AT-all and CCAT, 23.9% and 30.2%) are seeds 0–2, whose matched undefended mean is **77.1%**.
  - Paired: TRADES +6.7 pp (p = 0.54); AT-threat (five seeds) +2.8 pp (p = 0.42).
- **§5.5.** "Against the volume-only attack on the detectors with TTL features (61.5%, 19.4% and 34.3%)…"
  - These are five-seed values. The no-TTL detectors are seeds 0–2.
  - The matched seeds 0–2 with-TTL values are **54.8% (CNN), 20.4% (RF), 33.9% (GBDT)**.
  - The direction of every comparison is unchanged: CNN harder (36.3 < 54.8), RF easier (28.4 > 20.4), GBDT about as easy (36.7 against 33.9).

### B3. §5.4: "The augmented GBDT was both the most robust and the most accurate model on TON-IoT"
This holds at β = 0.5. At β = 2:
- augmented GBDT: 21.5%;
- augmented RF: **18.6%**;
- CNN with AT-threat: **18.9%**.

So the GBDT is not the most robust there. This is the round-1 O3 issue again, with v3 numbers.

### B4. §5.1: "the median evading flow needed 3 queries … on TON-IoT and 19 (CNN) to 64 (GBDT) on UNSW-NB15"
These are means over seeds of the per-seed medians.
- UNSW CNN seed medians are 6, 12, 64, 6 and 8: the mean is 19.2, but the median over seeds is 8.
- RF is also 64 on every seed, as is GBDT.

The numbers are correct as means of medians. Consider "8 (median over seeds)", or naming RF too.

### B5. §5.2: "On TON-IoT each network was evaded more than each tree under every attack"
This is true for every attack in Table 7. It is not true for the black-box feature-space search: every model is evaded on 100% at every ε on TON-IoT, a tie. Suggested wording: "under every realistic attack".

### B6. §3: "this mapping reproduces the pipeline's features to within 3e-04 (rounding)"
`attack_set.json` gives `map_err` = 3.40e-04 for UNSW-NB15 and 1.0e-07 for TON-IoT. 3.4e-04 is not within 3e-04, although it rounds to it. Say "to within 4e-04" or "about 3e-04". This is outside the requested §5–7 scope; noted for completeness.

### B7. Table 8: comparability for the pending split rows (no number is wrong)
The main-split rows (seed 0) reproduce exactly. Their "Strongest, all actions" includes the dense 1,000-point grid (`strongest_components` = grid, grid_vol, dense1000, dense1000_vol, ad_none, ad_cands, ad_free, wbs). The split runs use `light=True`: their strongest attack has no `dense1000` component and no β = 2 adaptive search (components = grid, grid_vol, ad_none, ad_cands, ad_free, wbs).

When the split rows are filled, either:
- recompute the main row with the same components, or
- footnote that the main-split "strongest" is a stronger attack than the split "strongest".

---

## VERIFIED (selection; all reproduce at the stated rounding)

**Abstract**
- 51.4% / 25.2% (TON strongest, CNN / RF).
- 5.5% (AT-threat CNN).
- "half of that gain": the clean loop gives −24.48 pp against AT-threat's −45.93 pp, or 53.3%. Against AT-threat paired over the loop's seeds 0–2 (−46.69 pp) it is 52.4%.
- 3.0% (GBDT augmented, TON).
- 30.8%–40.3%.
- "five seeds".

**Highlights:** all are verbal and consistent with the numbers. The "half" in the highlights is 53%.

**§5.1**
- Macro-F1 0.960 / 0.955 / 0.947 / 0.948.
- Detection "at least 92.6%" (MLP 92.63%).
- UNSW macro-F1 0.550–0.605; FPR 0.128–0.198; miss rates 28.8% / 38.2%; trees under 1% (0.46% and 0.60%).
- White-box ε = 0.01: 73.8% / 86.5%.
- Black-box ε-ball, TON: 100% for every model and seed.
- Strongest 51.4% / 64.8%; β = 2 grid 63.4% / 71.2%; strong PGD 42.1% against ε-PGD 75.1%; mutable-only 39.8%.
- UNSW: 61.5% / 66.2% against 63.2% / 63.8%; trees 19.4% / 34.3% against 62.0% / 98.7%; TTL alone 83.8%.
- All actions 99.3% / 95.4% / 89.7% / 75.0%, above the networks' maximum white-box ε-PGD (66.2% / 64.7%).
- Saturation:
  - +1,000 points: at most 4.72 pp (TON) and 9.76 pp (UNSW);
  - adaptive: 0.57–9.45 pp and 2.72–13.50 pp;
  - strong PGD over basic: +12.61 pp;
  - strong PGD below the grid: 0.40–5.78 pp (on UNSW against the grid with both actions).
- Ablation 12.7 / 11.1 / 0.9 / 47.9; UNSW delays 30.8 / 1.0 / 2.0.
- Floors × 0: 3.0–8.5% and 5.9–42.8%. The floor amounts 500 B / 5 packets / 0.5 s are β × floors.
- Overall evasion 72.4% against ASR 61.5% (UNSW CNN, strongest volume-only).

**§5.2**
- All 40 cells of Table 7: differences and Holm-adjusted Welch p-values.
- Text ranges +16.2 to +39.7, p ≤ 0.005; +27.2 to +46.8; +9.6 (p 0.026), +5.7 (p 0.103), +24.3, +20.4.
- SDs 12.7 / 10.2 / 1.4.
- "tens of points": UNSW CNN seeds range 45.0–78.9%.
- Transfer 33.9 / 67.8, 16.6 / 59.2, 9.5 / 37.5, all below the grid search, on UNSW with both actions.

**§5.4**
- Number of tests: 154 / 186.
- Clean loop: 25.1%, −24.5 pp, 53%.
- Against the loop: −22.2, −9.6, −6.2, +0.9.
- AT-threat:
  - −45.9 pp over 5 seeds, p = 0.0007, Holm 0.106;
  - MLP 6.9%;
  - ΔF1 −0.029 / −0.042, the smallest of the adversarial defences;
  - against other defences: −12.6 (TRADES) to −23.1 (CCAT), Holm 1.0, raw p 0.024–0.034, three common seeds;
  - β = 2: 18.9% against 63.4%.
- ε = 0.05 ranking: 23.9 / 30.2 / 77.9 / 83.8 / 75.1.
- TON trees:
  - 35.3 → 3.0 and 25.2 → 10.3, Holm 0.031 and 0.002;
  - ΔF1 +0.001 / −0.001;
  - second round 5.5 / 8.6; without delays 9.9 / 19.5;
  - β = 2: 21.5 against 50.4 (50.45 exact; it rounds down from 50.446).
- UNSW nets:
  - −38.9 / −35.1 volume-only; +15.6 / +19.7 TTL-only; 99.5 / 99.6 all actions; loop 81.8;
  - AT-threat 35.6%, −63.7 pp, p 0.0011, Holm 0.168; TTL 8.7 / 8.9;
  - TRADES 51.6; MLP AT-threat 48.5;
  - ΔF1 −0.040 / −0.125; FPR 0.241;
  - detection 97.7 against 71.2; overall 36.9 against 99.5.
- UNSW trees:
  - grid 0.6 / 1.7; strongest volume-only 1.1 / 2.2;
  - sweep 17.8 / 20.8; adaptive with any TTL 30.6 / 39.9;
  - strongest 30.8 / 40.3, Holm 0.004 / 0.030;
  - ΔF1 −0.003 / +0.002 (+0.0015); ΔFPR +0.005 / 0.000 (−0.0004);
  - second round 24.9 / 37.7; without delays 10.4 / 5.2.
- Costs:
  - rare-class F1 −0.256 to −0.063 and −0.121 to −0.040;
  - FPR +0.022 to +0.086 and +0.012 to +0.043;
  - CCAT 0.453 / 0.364 and 0.236 / 0.194; undefended 0.620 / 0.307; AT-threat 0.521 / 0.267;
  - tree changes at most 0.025 (TON GBDT ×2, rare-class F1 −0.0252).

**§5.5**
- 84.5 against 83.8.
- No-TTL macro-F1 0.533 / 0.581 / 0.588, paired Δ −0.021 / −0.021 / −0.017.
- Strongest 36.3 / 28.4 / 36.7.
- Table 11 (28 cells).

**§6 and §7:** 3.0–8.5%; 0.128–0.198; "half"; 3.0%; 30.8%–40.3%.

**Seed counts:** the Table 10 "Seeds" column, §4.4, §5.4 and the captions all match the files.

---

## COUNT

- Table cells compared automatically: about 960 numeric tokens in Tables 3–11. Mismatches: 72 cells of Table 9 (one cause: M1) and 4 CI bounds within noise (B1).
- Numbers in the text of the abstract, §5–7 and the highlights: about 210, including p-values, ranges, percentages and ratios. Mismatches: the §5.3 numbers (part of M1) and one verbal claim (M2).
- **Total: about 1,170 claims checked.**
- **2 real mismatches:** M1 (Table 9 and §5.3 label: strongest attack, not grid search) and M2 (budget monotonicity is false for the UNSW RF).
- 7 minor notes (B1–B7).
- 0 miscomputed values, seed counts or p-values.
