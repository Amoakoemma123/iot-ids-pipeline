# Paper 5: response to the final review round (2026-11-07)

Reviews: `review_final_referee.md` (6.5/10, major revision, narrow scope), `review_final_numbers.md`, `review_final_refs_code.md`.
Status: text and re-analysis done; the new runs requested by the referee are prepared as `Papers pipeline/experiments/paper5_xai/run_xai_final.py` and queued by `Papers pipeline/logs/p5final_queue.sh` (**not launched**). Their numbers are placeholders in the manuscript (shown as ⟦key⟧ in the build) and fill in automatically once the queue has finished and `compute_values.py` → `build.py` (+ `--anon`) → `scrub.py` → `verify.py` are re-run. Result-dependent conclusion sentences (`ls_conclusion`, `pos_conclusion`, `UNSW_mcn_conclusion`) are generated from the results so the wording follows the evidence.

Legend: **Done** = changed now; **Run** = text written, numbers pending the queue.

## Referee (review_final_referee.md)

| Item | Change |
|---|---|
| M1 hardening, CI overlap vs paired test | **Run + Done.** §4.7 describes exact McNemar tests per seed and pooled (stratified, summed discordant pairs) plus a seed × record bootstrap of the paired evasion difference; `hardening_paired` re-runs the attacker-controllable hardening with identical random streams (verified to reproduce Table 9, checked by verify.py) and saves per-record outcomes. "Only the evasion criterion" removed from §1 (contribution 4), §5.4, §6.1; now "on average ... most of the gain", with the per-seed CI-separation counts and the McNemar p-values; the closing sentence is generated from the test outcome. |
| M2 meaning of support-constrained evasion; local cost | **Done (text)** §4.6, §5.3 (all-groups arm called an upper bound by construction), abstract (100% headline removed; "all when any field may, by construction"), §6.1. **Run:** share of evaded vs non-evaded attacks whose non-controllable features already lie within τ of a benign training record; benign FPR on the 5 nearest benign test records of each added counterfactual / benign-donor copy (and within τ), per retrained model (§5.4, new paragraph). |
| M3 "only with duplicates" | **Done:** softened everywhere to "clearly only with duplicates" (abstract, §1, §2.5, §6.1, §7); test sizes/composition difference stated. **Run (`decomp_ci`):** interaction CI (difference of the two paired macro-F1 effects, independent bootstraps), paired accuracy CIs, and both effects on the common test population (records in both test sets, matched by raw `id`; 2,370 records in the smoke run). |
| M4 statistics level | **Done:** §4.9 states the convention (tables: seed-42 record CIs + range over seeds; text headlines for agreement, counterfactuals and hardening: mean over seeds with a two-level seed × record bootstrap), non-stratified resampling of a class-stratified sample, class-balanced means (also in Table 5 caption). New row in Tables 5–6 (mean over seeds, seed × record CI); §5.3 headlines now means over seeds with seed × record CIs and per-seed values; Table 9 has a seed-42 CI column; Tables 7–8 have n per class and Wilson 95% intervals; seed-42 CNN non-reproducibility stated plainly (§3.4, §6.3). |
| M5 LIME configuration | **Run (`lime_sens`):** 0.5×, 1×, 2× default kernel width (continuous) and the default discretised mode, on 15 records/class of the agreement sample, both datasets, RF and CNN: agreement, R², deletion SHAP − LIME for 4 strategies × 3 k. Text in §5.2 ("LIME configuration") and §4.5; conclusion sentence generated from the result (generalises vs restricted to continuous/default width). **Done:** LIME version, kernel, kernel width (0.75√p = 3.75 / 2.37) in §4.1; mean replacement noted as LIME-favourable in §4.5. |
| M6 feature-space twins | **Done (from saved per-record outputs + `twin_mask.py`, data load only):** UNSW agreement without twins (RF 0.39, CNN 0.48, unchanged), SHAP − LIME significant in 36/36 (RF) and 31/36 (CNN) cells without twins; Table 8 marks twin-dominated classes (†). **Run:** attacker-controllable evasion without twins (from `hardening_paired`). |
| M7 claims about auditing/hardening | **Done:** abstract last sentence, §6.1 (one retrospective case of our own error; a schema check would have caught it), §7 last sentence, Highlights, cover letter. |
| M8 novelty, companion study, TTL prior work | **Done:** new §2.5 "What is new"; companion study disclosed in §3.4 and in the cover letter (no citation invented); UNSW-specific `sttl` dominance positioned against Kasongo & Sun (2020, Crossref-verified, new reference) and Mohale & Obagbuwa (2025, already cited; full text confirms `sttl` as top feature), our result presented as a confirmation with a different reading. |
| m1 UNSW CNN mean-replacement not significant | Done: "in the mean", example 0.02 [−0.01, 0.05] given. |
| m2 two counts (35/36 vs 28/31) | Done: positive and significant counts labelled separately. |
| m3 templated prose | Done: "36, 36, 36 and 36", "100% ... 100%", "both controls ... (39–42% and 39–42%)", group lists rewritten (phrases generated when values coincide). |
| m4 budget of 0.57/0.73 | Done ("still at 1,000 samples"). |
| m5 SHAP-guided threat model | Done: stated it is the all-groups arm and lower than random (87% vs 100%). An attacker-controllable SHAP-guided arm was not run (deferred; noted here). |
| m6 medians over successes | Done: success rates given next to medians and the survivorship caveat stated. |
| m7 `ct_state_ttl` inconsistency | Done: acknowledged in §4.6. |
| m8 monotone variant | **Run (`cf_monotone`)**: TON-IoT, RF and CNN, source volume and duration increase-only; text in §5.3 and §6.3. |
| m9 binary majority baseline | Done (51.0%). |
| m10 worms n | Done (19 test records; Fig. 4 caption "worms, the only class with fewer than 20 query records"). |
| m11 Table 6 field row | Done: row dropped for UNSW-NB15, caption note. |
| m12 Table 9 seed-42 CIs | Done (new column). |
| m13 LIME settings, GradientExplainer nsamples | Done (§4.1). |
| m14 global-ranking baseline, normalised agreement | Done for SHAP (local SHAP vs global top-5, new Table 5–6 row; normalised agreement J/√(J_SHAP,self·J_LIME,self) in §5.2). LIME vs global top-5: **Run** (from `lime_sens`). |
| m15 class weights after SMOTE | Done: SMOTE equalises all classes (the 50,000 cap does not bind), so weights are uniform (§3.2, §3.4). This also corrected the earlier statement that UNSW minority classes are "raised to at most 50,000". |
| m16 CNN locality | Done (§3.4). |
| m17 global importance averaging | Done (mean of class-specific |SHAP| over all classes, §4.2). |
| m18 model-specific LIME recommendation | Done (§6.2). |
| m19 Fig. 4 class heterogeneity | Done (MITM 21% vs 41%, ransomware 72% vs 61%, §5.4). |
| m20 Fig. 3 caption | Done ("matches neither explainer's sampling"). |
| m21 how numbers were verified | Done (data availability statement). |
| m22 IoT keyword | Kept: TON-IoT is an IoT testbed and §1 frames the IoT setting; UNSW-NB15 is explicitly not IoT (§3.1). |

## Numbers review (review_final_numbers.md)

| Item | Change |
|---|---|
| W1 Jaccard read as fraction reproduced | Done: "top-5 self-agreement (Jaccard) of only 0.48 ... about 3.2 of 5 features"; 1,000-sample case "Jaccard of only 0.60 (about 3.8 of 5)"; shared counts computed from J. |
| W2 consensus LIME "closer" | Done: "does not bring LIME closer", paired differences −0.002 [−0.011, 0.007] and −0.020 [−0.047, 0.006], stability subsample explained. verify.py now checks the CI includes zero. |
| O1 UNSW vs TON RF evasion | Done: per-seed values; "for the CNN more in every seed; for the RF similar (more in seed 42 only)"; verify.py checks it. |
| O2 "only" the evasion criterion | Done (see M1). |
| O3 Highlight 4 / abstract | Done: "in all RF and most CNN deletion tests". |
| O4 Highlight 2 "without any duplicates" | Done: "without exact duplicates" (Highlight, abstract, §3.3 note on feature-space coincidences). |
| A1 0.051 vs 0.050 | Done: abstract and §3.3 quote seed-42 values (0.053 / 0.050), matching Table 1, with seed ranges. |
| A2 §5.4 mixed datasets | Done: split per dataset with computed group sizes (TON protocol state 16 of 25 features; any four of the five UNSW groups 7–9 of 10). |
| A3 conditional ratio for SHAP only | Done: SHAP 0.22–0.55, LIME 0.18–0.49, random 0.09–0.43. |
| A4 re-selection "similar" | Done: described as different comparisons (also review M7 of refs/code: the first one involves the earlier pipeline), no "similar". No CI is available for 0.015 without new runs. |
| A5 misclassified "unchanged" | Done: "slightly raises it (0.41 against 0.39)". |
| A6 TTL shortcut firmness | Done: abstract "lean on", §5.1 "examines whether", §5.5 "a shortcut the data offer". |
| N1 stale PDFs | Not regenerated (Word output only this round). **The PDFs in Paper5 are stale and must be regenerated (or deleted) before upload.** |
| N2 Highlight 5 | Done: "Counterfactual retraining cut constrained evasion from 41% to 16% on TON-IoT" (76 chars). All highlights ≤ 82 chars. |
| N3 rank intervals record-only | Done (§4.2, §6.3). |
| N4 TON exact-equality definition | Done (§3.3 parenthesis). |
| N5 31% vs 41% | Already explained in §4.7; §5.3 headline is now the mean over seeds (39%). |
| N6 LIME unstable "in general" | Done: "on TON-IoT and for the CNN"; UNSW RF reasonably stable (75% ≥ 0.75). |
| N7 "both explainers name features that matter" | Done: sentence removed; LIME vs random counts per model given. |
| N8 Fig. 4 caption worms | Done. |
| N9 queries saving "on TON-IoT" | Done: "for the CNN on TON-IoT ...; for the RF they are not useful at all". |

## References and methods vs code (review_final_refs_code.md)

| Item | Change |
|---|---|
| R1 Kalakoti 2025 wording | Done ("evaluate several explainers for faithfulness, complexity and robustness"). |
| R1 Engelen vs Lanvin | Done (Engelen: documents errors, releases corrected data; Lanvin: performance changes). |
| R2 Kacem surname | Done: OpenAlex/IEEE full name "Mohamed Aziz Ben Kacem" → "Ben Kacem, M. A." (Crossref splits it as given "Mohamed Aziz Ben", family "Kacem"; Tunisian compound surname). Moved alphabetically; check_citations.py handles "Ben". **Authors may want to confirm on the IEEE Xplore author page.** |
| R2 Lanvin LNCS volume | Done: Vol. 13857 (OpenAlex biblio volume; Crossref has none). |
| R2 Moustafa & Slay 2016 | Not added (optional). |
| R2 LIME version | Done (0.2.0.1). |
| M1 two conventions | Done (A1). |
| M2 absolute top-k in deletion | Done: §4.5 states "ranked by absolute attribution"; **Run:** positive-only (signed) ranking for SHAP and LIME, 4 strategies × 3 k, plus share of top-5 features with positive sign (§5.2 "Positive attributions only"). |
| M3 kernel width/kernel | Done (§4.1) + **Run** (M5). |
| M4 RFECV floor, MI threshold, estimator | Done (§3.2, filled from pipeline.pkl): MI < 0.05 dropped; RFECV with a 50-tree RF, 3-fold stratified CV (the review's "5-fold" is not what the code does: `rfecv_cv=3`), macro-F1, 50,000-record subsample, floor 10; TON 25 of 27 (interior), UNSW 10 of 40 = floor (CV score highest there and lower for every larger set). verify.py checks the floor statement. |
| M5 what the seed varies | Done (§3.4). |
| M6 raw-weight deletion predates condx | Done: disclosed in §5.2. |
| M7 re-selection mixes v1 pipeline | Done (§3.3). |
| M8 field sums absolute | Done (§4.3). |
| M9 copy control groups | Done ("the same set of groups ..."). |
| M10 τ from 5,000 of 20,000 | Done (§4.6); heavy tails in §6.3. |
| M11 queries include rejected candidates | Done (§4.6). |
| M12 = A5 | Done. |
| M13 CNN TTL drop within noise | Done (§5.5). |
| M14 worms n | Done. |
| M15 paired test for hardening | Run (M1). |
| M16 CNN global SHAP rseed / rank intervals | Done (§4.2, §6.3). |
| M17 TON counterfactual provenance (v3.0 code) | Not changed in the paper (results valid; group definitions identical). Suggest a README note in the repository. |
| M18 raw-column definitions for generic/conflict counts | Not changed this round (low). |

## Prepared runs (not launched)

`Papers pipeline/logs/p5final_queue.sh` (resumable; `export OMP_NUM_THREADS=2`; writes "P5FINAL DONE" to `logs/p5final_queue.log`), stages of `experiments.paper5_xai.run_xai_final`, outputs in `experiments/paper5_xai/results_final/`:

| Stage | Data | Purpose | Est. time |
|---|---|---|---|
| decomp_ci | UNSW, seed 42 | M3 (interaction CI, accuracy CIs, common population) | ~5 min |
| lime_sens | TON, UNSW, seed 42 | M5, refs-M2, m14 (LIME vs global) | ~15 min per dataset |
| cf_monotone | TON, seed 42 | m8 | ~10 min |
| hardening_paired | TON, UNSW, seeds 42–44 | M1, M2, M6 (evasion without twins) | ~12 min per TON seed, ~7.5 min per UNSW seed |

Total ≈ 1 h 45 min (allow 2–2.5 h with OMP_NUM_THREADS=2 and another job running). Peak RAM ≈ 3 GB (UNSW hardening: 0.5 GB RF pickle plus a retrained RF on ~363k SMOTE rows); the other stages stay below ~2 GB. Smoke tests (`--smoke`, tiny subsamples, 5-tree RFs for retraining) of decomp_ci, lime_sens, cf_monotone and hardening_paired all completed in < 90 s each; smoke outputs were deleted.

After the queue: `compute_values.py` → `build.py --anon` → `build.py` → `scrub.py` on every docx → `verify.py` (must reach 0 failures; it checks that `hardening_paired` reproduces Table 9 exactly) → `check_citations.py`. Then read the generated conclusion sentences in §5.2 and §5.4 and adjust abstract/§6.1/§7 if the paired tests or LIME sensitivity contradict the current hedged wording.

## Other

- Word count before references rose from ~10.3k to ~13.4k (including ~96 pending placeholders); abstract 249/250.
- References: 39 (Kasongo & Sun 2020 added), all cited (check_citations.py).
