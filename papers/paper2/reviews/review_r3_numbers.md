# Paper 2 v4: number audit, round 3

Audited files:
- `Manuscript_filled.md` (13 Nov 2026, 01:19);
- `Supplement_filled.md` (Tables S1–S10);
- `HIGHLIGHTS`, `TABLE_CAPTIONS` and `SUPP_CAPTIONS` in `build.py`.

Raw data:
- `Papers pipeline/experiments/paper2_adv_v2/results_v3/<dataset>/<tag>/seed<k>.json`, written by `run_adv_v3.py`;
- `results_r2/<dataset>/<tag>/seed<k>.json`, written by `run_adv_r2.py`;
- `attack_set*.json`.

## Method

I used the same approach as round 2. I wrote my own scripts, which read only the JSON files. They do not import `compute_values.py` or reuse any of its functions. I read `compute_values.py` only for its definitions, and I report below where those definitions differ from the paper. The scripts are in `C:\Users\G15165~1\AppData\Local\Temp\claude\bf2fe0286de83ad4\777750fb-fd31-4dd1-9617-244756f9a74e\scratchpad\p2audit_r3\`:

- `lib.py`: loading, my own Holm step-down, paired and Welch tests, formatting, and a markdown-table parser.
- `tables.py`: rebuilds main Tables 3–9 and Tables S1–S10 from the JSONs, then diffs them cell by cell against the parsed manuscript and supplement tables. Output: `out_tables.txt`.
- `text.py`: recomputes every quantity and verbal claim in the text, plus the highlight lengths and the abstract word count. Output: `out_text.txt`.
- `cicheck.py`: checks the bootstrap CIs for Monte-Carlo noise. Output: `out_ci.txt`.

**Conventions**

- **Seeds.** I took the seeds from the folder inventory, not from the code:
  - `results_v3`: five seeds for the undefended models, the five CNN defences, and RF/GBDT augmented; three seeds for the rest.
  - `results_r2`: five seeds for the networks and the CNN defences; three seeds (0–2) for every tree tag, `mlp_at_threat`, `cnn_at_all_bpda` and `*_nottl`; seed 0 only for the sweep tags.
- **SD.** ddof = 1.
- **Rare-class F1.** The mean of `per_class_f1` over `rare_classes`.
- **ASR keys.**
  - Strongest: `strongest_b0.5`. Its union in `run_adv_v3.py` is grid, grid_vol, dense1000, dense1000_vol, ad_none, ad_cands, ad_free and wbs.
  - Volume-only strongest: `strongest_volume_b0.5`, the union of the keys ending in `_vol` plus ad_none. It therefore excludes the strong white-box PGD, even on TON-IoT (see E4).
- **Matched operating point.** `op/fpr_match/*` for the defended model against `op/native/*` for the undefended model. The paired t-test runs over the shared seeds.
- **Primary family (12 tests per dataset).** Holm-adjusted with my own implementation:
  - 6 network defences against the undefended model at the matched FPR;
  - CNN AT-threat against 4 other CNN defences, both at the matched FPR;
  - 2 tree augmentations against the undefended tree, on `strongest_b0.5` over 5 seeds.
- **Secondary family.** Rebuilt from the text's description: clean macro-F1 and the ASR keys against the undefended model, CNN defences against the clean loop, and defence-against-defence pairs. On TON-IoT the duplicate keys are dropped.
- **Table 6.** Welch's t-test, Holm within dataset × attack (4 tests).
- **Two-level bootstrap.** Resample seeds, then flows within each drawn seed, from `ev_strongest_b0.5`. 1,000 resamples, run with several RNG seeds.

## Overall result

- **Tables.** 1,582 numeric cells in main Tables 3–9 and Tables S1–S10 were compared. **1,576 reproduce exactly.** The other 6 are bootstrap CI bounds of Table 4, and these are within Monte-Carlo noise (N1).
- **Holm p-values.**
  - All 24 primary tests reproduce, as do the counts "6 of 12" (TON-IoT) and "8 of 12" (UNSW-NB15).
  - All 40 Welch/Holm cells of Table 6 reproduce.
  - The secondary-family sizes (120 and 238) reproduce.
- **Text.** Every number quoted in the abstract, §5.1–5.5, §6 and §7 reproduces.
- **Round-2 findings.** M1 (Table 9 label) and M2 (budget monotonicity) are fixed. B2, B3, B4, B5 and B7 are fixed. B6 is only partly fixed (E7). B1 remains, as noise (N1).
- **Remaining problems.** One caption states the wrong seed count (E1). One methods sentence claims a check that the code did not fully run (E2). The rest are small inconsistencies.

---

## ERRORS, by severity

### Moderate

#### E1. Table 7 caption: "Means over five seeds (MLP with AT-threat: three)". The tree rows are three seeds
- **Location.** `build.py` `TABLE_CAPTIONS['5.4'][0]`, which is the Table 7 caption.
- **What the files contain.** `results_r2` has only seeds 0–2 for `rf_none`, `rf_augment`, `gb_none` and `gb_augment` on both datasets. So every tree row of Table 7 is a three-seed mean. This covers the own and matched FPR, the own and matched ASR, the grid-only Δ, the missed-or-evaded share and the ASR at 1% FPR.
  - Examples: TON RF 24.2%, TON GBDT 24.0%, UNSW RF 85.6%, UNSW GBDT 55.7%; the Δ values −15.8, −22.5, −85.6 and −54.5 pp.
  - Only the bracketed "strongest" Δ and its Holm p use five seeds (from `results_v3`).
- **Consequences in the text.**
  - §5.4 (TON trees) says "with the threshold matched, the grid search evaded the augmented GBDT on 1.5% of detected flows, against 24.0% without augmentation". This sits in a sentence that begins "(five seeds, …)", so the reader takes 1.5% and 24.0% as five-seed values. They are three-seed values.
  - The five-seed grid value for the undefended TON GBDT is 25.0% (Table 4), so Table 4 and Table 7 show two different "grid, all actions" values for the same model.
- **Source.** `results_r2/*/{rf,gb}_{none,augment}/seed0-2.json`, key `op`.
- **Fix.** Use "Means over five seeds (MLP with AT-threat and all tree rows: three; the bracketed tree Δ: five)". Alternatively, run `run_adv_r2.py --parts op` for tree seeds 3–4.

#### E2. §4.3 says the 0.8 pp check covered the basic PGD and transfer. It did not
- **Location.** §4.3, "Strongest attack" paragraph.
- **Text.** "We checked that the attacks it leaves out (the TTL sweep, the basic PGD, transfer, a 24-point grid and the per-action grids) add little: across all models, adding them raised the strongest ASR by at most 0.8 pp."
- **What the code does.** `run_adv_r2.part_union` extends the union with only three of these: the 24-point grid, the four per-action grids and, on UNSW-NB15, the TTL sweep. Separately, on TON-IoT it puts the white-box attack into the volume-only union. The basic white-box PGD (`wb_action_b0.5`) and transfer (`transfer_action_b0.5`) are **not** in the extended union. `run_adv_v3.py` stores only their rates, not per-flow outcomes, so their contribution cannot be computed from the files.
- **The value is right for what was computed.** The largest gain over all r2 tags is 0.81 pp, on UNSW GBDT volume-only (0.17 pp with all actions). The largest all-actions gain is 0.61 pp (UNSW CNN AT-threat, β = 1).
- **Source.** `run_adv_r2.py`, function `part_union`; `results_r2/*/*/seed*.json`, keys `union/strongest_plus*`.
- **Fix.** Remove "the basic PGD, transfer" from the list. Or say that these two were not included, and that they evade fewer flows than the strong PGD and the grid on every model (Table 4 shows this for transfer).

### Minor

#### E3. §4.3: the worst-case query count is 1,769 / 7,243, not 1,761 / 7,163
- **Text.** "the strongest attack uses up to 1,761 queries on TON-IoT and 7,163 on UNSW-NB15."
- **What the paper's numbers assume.** `compute_values.py` counts the 1,000-point grid as `1000 − 8` (TON-IoT) and `(1000 − 16) × 5` (UNSW-NB15).
- **What the code runs.** In `run_adv_v3.evaluate_attacks` the dense grid is `unit_grid(k, 1000, …)[2 ** k:]`. `unit_grid` returns the 2^k corners **plus** 1,000 random points, and the slice removes only the corners. So the grid has 1,000 points.
- **Correct totals.**
  - TON-IoT: 128 + 1,000 + (1 + 40 × 16) = **1,769**.
  - UNSW-NB15: 64 × 5 + 1,000 × 5 + 3 × 641 = **7,243**.
  - If the strong PGD's success checks are counted as queries, add 16 (TON-IoT) and 80 (UNSW-NB15): 4 restarts × 4 checks, times the TTL options.
- **Source.** `run_adv_v3.py`, functions `unit_grid` and `evaluate_attacks`; `compute_values.py`, keys `TON_IoT_strongest_queries` and `UNSW_NB15_strongest_queries`.
- **Note.** This is a bug in `compute_values.py`, not a different definition.

#### E4. Table 8 caption: "On TON-IoT the volume-only strongest attack equals the all-actions one". Not as the data are stored
- **The data.** In `results_v3` the volume-only union leaves out the strong white-box PGD (`wbs`), even on TON-IoT, where that PGD only uses volume and timing actions. The two keys therefore differ for every TON-IoT network:
  - undefended CNN: by up to 0.53 pp per seed; means 51.3% against 51.4%;
  - MLP: up to 0.15 pp per seed;
  - AT-all: up to 0.24 pp per seed; CCAT up to 0.12; TRADES up to 0.14.
- **Where the inconsistency shows.**
  - Table 4 (TON CNN), Table 5 ("Realistic", TON CNN 51.3%) and Table 6 ("Strongest, volume only" against "all actions", e.g. +26.1 against +26.3 pp) report the two as different numbers.
  - Table 8 prints "= all actions".
  - `run_adv_r2.part_union` explicitly treats them as equal on TON-IoT ("the white-box attack counts as volume and timing").
- **Effect.** At most 0.15 pp on any mean. No conclusion changes.
- **Fix.** Either OR `wbs` into the TON-IoT volume-only union everywhere (then Table 4 TON CNN becomes 51.4%, and Table 5 and the TON rows of Table 6 move by ≤ 0.2 pp), or drop the sentence from the Table 8 caption.

#### E5. §5.5 and Table 9 against Table 5: the same "volume and timing only" quantity appears as 36.3% and as 36.1%
- **The two versions.**
  - §5.5 says: "Without the TTL the attacker loses its free action, and the strongest attack, now volume and timing only, evaded 36.3% (CNN), 42.1% (MLP), 28.4% (RF) and 36.7% (GBDT)". The same values appear in Table 9, "Strongest, without TTL features". These are `strongest_b0.5` of `*_nottl`, which includes the white-box PGD.
  - Table 5 gives the same detectors as "Realistic, volume and timing": CNN 36.1%, MLP 41.9%. These are `strongest_volume_b0.5`.
- **Mixed comparison.** §5.5 compares 36.3% and 42.1% with the with-TTL values 54.8% and 70.7%. Those are `strongest_volume_b0.5`, which excludes the white-box attack. The comparison mixes the two definitions.
- **Effect.** 0.2 pp. The directions are unchanged (36.1 < 54.8, 41.9 < 70.7, 28.4 > 20.4, 36.7 ≈ 33.9).
- **Source.** `results_v3/UNSW_NB15/{cnn,mlp}_none_nottl/seed0-2.json`.
- **Fix.** Use one key in both places.

#### E6. §5.1: the summary sentence understates where the feature-space attack did not overstate
- **Text.** "In short, the feature-space attack overstated what volume and timing changes achieve for every model and every ε we tested, except the UNSW-NB15 trees at the smallest ε (and, for the RF with increases only, at every ε up to 0.1)."
- **Where it did not overstate (feature-space ≤ realistic):**
  - UNSW RF, ± patterns: ε = 0.005 **and 0.01** (8.4% and 15.8% against 20.4%).
  - UNSW GBDT, ± patterns: ε = 0.005.
  - UNSW GBDT, increases only: ε = 0.005 **and 0.01** (19.8% and 30.8% against 33.9%).
  - UNSW RF, increases only: every ε.
- **The rest of the paper is consistent.** The crossover column of Table 5 and the preceding sentence ("0.02 for the RF and 0.01 for the GBDT; increases only: > 0.1 and 0.02") are correct. Only "at the smallest ε" is too narrow.
- **Fix.** "except the UNSW-NB15 trees at the smallest one or two ε".
- **Elsewhere.** The abstract (ε = 0.005), §6 ("at the smallest balls") and §7 ("at small ε") are correct.

#### E7. §3: "to within 3e-04 (rounding)". Round-2 B6, only partly fixed
- `attack_set.json` gives `map_err` = **3.397e-04** for UNSW-NB15 and 1.0e-07 for TON-IoT.
- 3.4e-04 is not within 3e-04. The added "(rounding)" does not make the sentence true.
- **Fix.** "to within 3.4e-04" or "about 3e-04".

#### E8. Table S4 caption: "means over five seeds". The UNSW-NB15 GBDT worms row uses two seeds
- For `gb_none` on UNSW-NB15, `min_action_per_class['worms']` is null on seeds 0, 1 and 4: no worms flow was evaded without a TTL change.
- The row (share 1.00, 0 bytes, 24.0 packets, 0.00 s) is therefore the mean of seeds 2 and 3 only (10 and 38 packets).
- All other S4 rows have five seeds.
- **Fix.** Footnote the row, or drop it.

#### E9. Table 5 caption: "five for the networks with TTL features, three otherwise"
- The TON-IoT networks use five seeds, but TON-IoT has no TTL features.
- **Correct wording.** "five for the undefended networks, three for the trees and for the detectors without TTL features". The cell values themselves are right.

#### E10. Table S9 caption: "a 1,500-byte MTU"
- The code (`run_adv_r2.feasible`) and §4.3 use 1,460 **payload** bytes per packet on TON-IoT and 1,500 bytes including headers on UNSW-NB15.
- The caption gives only the UNSW-NB15 bound.

---

## NOTES (not errors; for the authors' judgement)

### N1. Table 4 CIs: Monte-Carlo noise remains (round-2 B1)
- With 1,000 resamples, 6 of the 8 printed CIs differ from my first bootstrap run in one bound.
- Across 4 RNG seeds the bounds move by up to 0.7 pp. For example, the TON CNN upper bound ranges 57.30–57.96; the paper has 57.4.
- Every printed bound lies inside the range I get across RNG seeds, so none is wrong.
- **Recommendation.** 10,000 resamples would make the first decimal stable.

### N2. §5.4, TON-IoT trees: "the saturation check … left the augmented trees' ASR within 3.8 pp of it"
- True, but loose.
- For the augmented trees the gain was 0.5 pp (RF and GBDT). The 3.8 pp is the **undefended** TON GBDT.

### N3. §5.4: "under the realistic attack the most evaded" (UNSW-NB15 ε-ball defences)
- At the model's own threshold (Table 8 and S7, "Realistic strongest"):
  - AT-all 99.0% and CCAT 98.4%;
  - undefended CNN 99.3%;
  - AT-all (BPDA) 99.9%.
- So at that threshold AT-all and CCAT are not the most evaded CNNs.
- At the matched FPR (Table 7), they are: 99.4% and 99.5%, against 99.2%.
- **Fix.** Say "at the matched FPR", or "among the most evaded".

### N4. The secondary family overlaps the primary family
- The secondary family (120 / 238 tests, both reproduced) contains the two tree tests "strongest-attack ASR against the undefended tree". These are also in the primary family.
- "All other defence comparisons" is therefore not literally true. The effect on the secondary adjusted p-values is negligible.

### N5. Table 9 mixes seed sets across columns
- "Volume only, with TTL features" is the five-seed grid value: CNN 58.6%. On seeds 0–2 it would be 52.0%.
- "Volume only, without TTL features" uses seeds 0–2.
- §5.5 does use seed-matched values (54.8% etc.), so the text is correct.
- Also in §5.5, "99.6% against 71.2%" (detection) compares three seeds with five. Seeds 0–2 with TTL features give 73.7%.

---

## VERIFIED (all reproduce at the stated rounding)

**Abstract**
- ε = 0.005 mutable-feature attack above the realistic attack, strictly, on every model except the UNSW trees (also the four no-TTL detectors).
- 51.4% / 25.2%.
- Floors claim.
- Every network defence raised the FPR: all 7 defences × 2 datasets, positive on **every** seed.
- 35.0% against 51.0%, not significant (Holm 0.429).
- 99.2% → 43.6%.
- 35.3% → 3.0%, ΔF1 +0.001.
- 30.8%–40.3%.
- 249 words.

**Highlights**
- Lengths 82, 77, 81, 76 and 85 characters, all ≤ 85.
- All are verbal and consistent with the data.

**§3–§4**
- 5,000 flows per dataset.
- TTL candidates 31, 254, 62, 1.
- Rare classes: mitm; analysis, backdoor, shellcode, worms.
- 128 / 320 queries per flow and budget.
- 17³ / 7⁴ grids, 500 flows, seeds 0–2.
- Seed counts in §4.4 match the folder inventory.
- Attack and defence hyper-parameters match the code: adaptive 40 × 16 with σ 0.3 → 0.02; strong PGD restarts and steps; BPDA step 0.25ε → 0.025ε; AT 7 steps of ε/4; TRADES 6×; augmentation with 24 random points and corners, 20,000 flows.

**§5.1**
- 0.960 / 0.955 / 0.947 / 0.948; ≥ 92.6%.
- UNSW: macro-F1 0.550–0.605; FPR 0.128–0.198; missed 28.8 / 38.2; trees 0.5 / 0.6.
- Strongest attack: 51.4 / 64.8 / 25.2 / 35.3; 61.5 / 66.2 / 19.4 / 34.3; TTL alone 83.8; 99.3 / 95.4 / 89.7 / 75.0.
- Black-box corners at ε = 0.01: exactly 1.0 on every TON model and seed (v3 five seeds, r2 seeds).
- White-box ε-PGD on the TON CNN: 73.8%.
- Crossovers: as in Table 5.
- Saturation:
  - +1,000 points: 4.72 / 9.76 pp;
  - adaptive: 0.57–9.45 and 2.72–13.50 pp;
  - strong over basic PGD: +12.61 pp;
  - strong PGD below the grid: 0.40–5.78 pp;
  - tree saturation: 3.80 / 10.27 pp.
- Feasibility: largest change −1.09 pp (TON CNN, and TON GBDT w/o bytes), so "at most 1.1 pp" holds.
- Budget monotonicity: the 7 other curves are increasing; UNSW RF peaks at 12.9% (β = 0.25) and falls to 10.6% (β = 2). M2 is fixed.
- Ablation 12.7 / 11.1 / 0.9 / 47.9; delays 30.8 / 1.0 / 2.0.
- Floors × 0: 3.0–8.5% and 5.9–42.8%.
- Queries, as the median over seeds of the per-seed median + 1: 4th (TON CNN), 9th (UNSW CNN), 65th (RF and GBDT, every seed). The 65th query is the first candidate with TTL 31 and zero action. B4 is fixed.
- 72.4% against 61.5%.

**§5.2**
- All 40 cells of Table 6.
- +16.2 to +39.7 pp, p ≤ 0.005.
- Floors × 0: 3.0–8.5%.
- +27.2 to +46.8 pp.
- +9.6 (0.026), +5.7 (0.103), +24.3 / +20.4.
- SDs 12.7 / 10.2 / 1.4.
- "tens of points": UNSW CNN seeds range 45.0–78.9%.
- Splits: nets above both trees in 3/3 splits on each dataset.
- Transfer 33.9 / 67.8, 16.6 / 59.2, 9.5 / 37.5, below the grid search on every model.

**§5.3 (Table S6).** M1 is fixed (the table is now labelled as the strongest attack):
- ransomware ≥ 75.5, scanning ≥ 76.8;
- RF 0.2–1.7, GBDT 10.1–57.4, networks higher on all four classes;
- UNSW min over networks and RF 74.0 (RF exploits);
- GBDT 47.0 / 48.4.

**§5.4**
- FPRs 0.015 / 0.035 / 0.044 / 0.084.
- 5.5 against 51.4; 26.2.
- Matched FPR: 60.3 / 62.3 / 70.7 / 87.9 against 51.0.
- Primary family, TON-IoT:
  - TRADES +36.8 (Holm 0.0028);
  - AT-threat −16.1 (0.429);
  - MLP 33.5 against 64.7 (0.104);
  - AT-threat against AT-all −27.3 (0.028), TRADES −52.9 (0.0135), clean loop −25.3 (0.028), CCAT (0.116).
- 38.3 against 51.8.
- Primary family, UNSW-NB15:
  - AT-threat −55.6 (0.0136); TRADES 52.4 (0.0136); clean loop 83.6 (0.0034);
  - AT-all 99.4, CCAT 99.5;
  - AT-threat better than AT-all, CCAT and the loop with p ≤ 0.0241; not better than TRADES (0.678);
  - MLP 61.8 against 95.3 (0.128).
- 6/12 and 8/12 significant.
- ≥ 96.0% at 1% FPR.
- Table S7: 27.5 / 27.0, 91.4 / 84.1. The TON corner search is exactly 1.0 for every CNN and seed, BPDA variant included.
- BPDA variant: 0.585 / 0.280 / 71.9%; UNSW corners 63.7 / 61.4 / 97.1.
- Table S10: 18.1 / 20.5 / 49.0; 83.0 / 67.4 / 41.1. Every UNSW AT-threat configuration is below every AT-all configuration, at the own threshold and at the matched FPR.
- −30.5 / −5.2 pp.
- TON trees:
  - 35.3 → 3.0 and 25.2 → 10.3, Holm 0.0023 and 0.0001;
  - ΔF1 +0.0008 / −0.0006;
  - ΔFPR −0.0001 / +0.0005, both printed as 0.000, correctly;
  - matched 1.5 against 24.0 (three seeds; see E1);
  - second round 5.5 / 8.6; w/o delay 9.9 / 19.5; bytes only 11.8 / 7.9; w/o bytes 16.0 / 13.0, which lie between the undefended and the augmented values;
  - β = 2: 23.9 / 19.1 against 60.0 / 28.7;
  - 0.961 against 0.960.
- UNSW trees:
  - grid 0.6 / 1.7; strongest volume-only 1.1 / 2.2; sweep 17.8 / 20.8; adaptive with any TTL 30.6 / 39.9; strongest 30.8 / 40.3;
  - Holm 0.0002 / 0.0019;
  - ΔF1 −0.0034 / +0.0015; ΔFPR +0.0047 / −0.0004;
  - long adaptive search +10.2; second round 24.9 / 37.7; w/o delays 10.4 / 5.2.
- Costs:
  - rare-class F1 −0.4587 to −0.0613 and −0.1196 to −0.0400, every one negative;
  - FPR +0.0200 to +0.2644 and +0.0106 to +0.0426; both TON extremes at the large end are the BPDA variant;
  - CCAT 0.459 / 0.349, 0.211 / 0.192; 0.620 / 0.307; 0.521 / 0.267;
  - −0.029 / −0.040;
  - tree changes at most 0.0252 (TON GBDT ×2, rare-class F1), so "at most 0.025" holds.

**§5.5**
- 84.5 / 83.8; 36.6 / 53.5.
- No-TTL detectors: macro-F1 0.533 / 0.532 / 0.581 / 0.588; Δ −0.021 / −0.020 / −0.021 / −0.017; FPR 0.203 / 0.192 / 0.176 / 0.138.
- Strongest 36.3 / 42.1 / 28.4 / 36.7 (see E5).
- With TTL features, seeds 0–2: 54.8 / 70.7 / 20.4 / 33.9.
- Detection 99.6 against 71.2; overall evasion 36.6 / 37.1.

**§6 and §7:** 3.0–8.5%; 0.128–0.198; "three of the four other CNN defences" (AT-all, TRADES and the loop significant, CCAT not); 3.0%; 30.8%–40.3%.

**Captions with seed counts:**
- Tables 3, 4, 6 and 9, S1–S3, S5–S8 and S10: match the files.
- Table 7 (E1), S4 (E8) and the Table 5 wording (E9): do not.

---

## COUNT

| Severity | Count | Findings |
|---|---|---|
| Critical (a reported result or conclusion wrong) | 0 | – |
| Moderate | 2 | E1 Table 7 caption seed count; E2 the left-out-attacks check did not include basic PGD or transfer |
| Minor | 8 | E3 query counts 1,761 / 7,163 → 1,769 / 7,243; E4 TON volume-only "= all actions"; E5 36.3 against 36.1; E6 "at the smallest ε"; E7 3e-04; E8 S4 worms row from 2 seeds; E9 Table 5 caption wording; E10 S9 MTU wording |
| Notes | 5 | N1–N5 |

- **Claims checked:** 1,582 table cells, about 230 numbers and verbal claims in the text, 15 captions and 5 highlights.
- **Mismatched numbers:** 0 miscomputed table values, 0 wrong p-values, and 0 wrong significance calls.
- **One wrong number in the text:** the query count (E3).
