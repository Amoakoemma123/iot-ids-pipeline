# Paper 6: response to the final-round reviews (2026-11-07)

Reviews: `review_final_referee.md` (7.5/10, minor), `review_final_numbers.md`, `review_final_refs_code.md`.
No model was trained and no dataset was loaded. All new numbers come from saved result files through `compute_values.py`. `verify.py` gives 0 failures (about 40 new checks under "round-3 review fixes"). The fixed `check_citations.py` gives 51 references, matched one to one in both `Manuscript_filled.md` and `Manuscript.md`. The abstract is 248 words and the longest highlight is 84 characters.

## Referee: major issues

| Item | What was changed (or why not) |
|---|---|
| M1 one non-saturated IoT dataset | No dataset was added. The abstract now says that TON-IoT is "the only non-saturated IoT benchmark in the study". Contribution 1 names UNSW-NB15 as a non-IoT replication and N-BaIoT as a ceiling check. §6.3 says so plainly and cites CICIoT2023 (Neto et al., 2023) and Edge-IIoTset (Ferrag et al., 2022), with both DOIs checked on Crossref. |
| M2 point estimate stated as a bound | Abstract: "changed late-fusion macro-F1 by +0.001 on TON-IoT and +0.009 on UNSW-NB15, where an effect up to about 0.02 cannot be excluded". The old closing "at most about 0.01" sentence is removed. "Not significant after Holm correction" is now used for UNSW late − early. §5.3 and §6.1 now say "point estimates of at most about 0.01", and §7 says "by a point estimate of about 0.01". §4.3 now says the semantic-versus-random test is not in the confirmatory family and that late versus late-on-all is its confirmatory proxy. §5.3 adds an even-handed sentence on the UNSW-NB15 hints: semantic above 9 of 10, late above late-on-all in 5/5 seeds, a larger effect on duplicate-free records. |
| M3 tuned trees vs untuned CNNs | No grid was run, because no training was allowed. The claim is qualified instead. Abstract: "Tuned tree ensembles beat every untuned fusion network in macro-F1". Highlight 3: "Tuned trees beat untuned CNNs in macro-F1". §5.2: "a comparison of tuned trees with untuned networks … a margin of this size might be closed by tuning", and it notes that the CNNs lead on analysis and backdoor. §6.3 adds Gorishniy et al. (2021), checked on the NeurIPS proceedings page (pp. 18932–18943; there is no Crossref DOI), and says that tuning could plausibly change the TON-IoT ranking. **The small CNN grid remains open and would need new runs.** |
| M4 "wider head", missing direct test | The text now says "the extra 128-unit dense layer" in §5.2, §6.1 and Fig. 1. Table 3 gains the row "Late fusion − early fusion with head": TON +0.001 [−0.001, +0.004] (4/5), UNSW +0.010 [+0.004, +0.015] (5/5). §5.2 gives the shares of the late-minus-early gain: wide early fusion ≈41%, head ≈71%, late on all features ≈73%. The attribution now reads "mainly from the extra dense layer and less from the parameter count (the intervals overlap …)". The abstract and §6.1 say "largely reproduced". §5.2 adds that on UNSW-NB15 the head does not explain the gain. |
| M5 forest footprint missing from abstract and highlights | Abstract: "the tuned forests occupy 221–636 MB against 343 kB for late fusion, and a 4.4 MB forest scored below late fusion on TON-IoT". Highlight 3 now mentions the 4.4 MB forest. §5.6 adds that this forest is below late fusion by more than late − early while being about 13 times larger. §6.1 adds a fourth conclusion and §7 one sentence. The compact-forest numbers come from the companion study's result files (`experiments/baseline/results_v2`), which `compute_values.py` already reads and `verify.py` checks. The companion study is now called "not yet published" rather than left implicit; see m7. |
| M6 test-set duplicates | Added Appendix B, Table B.2, computed from the saved per-record predictions and the clean mask. For the main models it gives macro-F1 on all records, macro-F1 without a match, and accuracy on matched and on unmatched records. New §5.2 paragraph: on UNSW-NB15, 81% of backdoor, 80% of generic and 71% of analysis test records have a match, against 10% of benign. Every model is about equally accurate on matched records (0.630–0.644, against 0.786–0.833 on the rest), and the RF is no better than late fusion there (−0.007 accuracy, against +0.036 on unmatched records). This explains why the RF lead rises from +0.034 to +0.073. **Not done: the share of matches whose training record has a different label.** The label of the matching training record was not saved, and computing it would mean reloading the dataset, which was not allowed. The text says so, and it frames label conflict as a likely reading of the low matched-record accuracy, not as a measured fact. |

## Referee: minor issues

| Item | Change |
|---|---|
| m1 random-split late vs early | The comparator is now stated: "the five-seed mean of early fusion on the semantic order". It also says early fusion has no streams, so it was not trained on random splits. |
| m2 Fig. 3 label overlap | Labels were hand-placed with leader lines, and the panels were widened (`FIG3_LABELS` in `make_figures.py`). Checked visually: no label overlaps another. The caption now notes the N-BaIoT x-range of about 0.013. |
| m3 PR-AUC and detection columns | Added as Table B.3 (macro-F1, PR-AUC, detection rate and FPR for every model on TON-IoT and UNSW-NB15). Table 2 was left as it is, because 10+ columns would be unreadable. §5.1 points to Table B.3. |
| m4 full Holm table | Table B.1 lists all 8 tests with effect, unadjusted CI, seed count, unadjusted p, Holm p and margin reading. It includes both attention-versus-early tests (Holm p = 0.135 on each). The attention Holm p-values are also given in §5.2. |
| m5 Fig. 2 / Table 6 unlike quantities | The Fig. 2 and Table 6 captions now say that random splits are single runs and the semantic value is a 5-seed mean. §5.3 now reads "the semantic five-seed mean exceeds N of the ten single-run values" and says that the within-seed differences are the like-for-like statistic. |
| m6 Sanjalawe year/DOI | Checked again on Crossref. Print, issued and published dates are all 2025, vol. 5, pp. 143–164; the DOI suffix just says 2026. The entry is correct and was left unchanged. |
| m7 companion studies uncited | They cannot be cited, because they are not yet published. §5.6 now calls the footprint study "not yet published" and says its values are read from its result files. The §3.1 "released with this paper" wording was left as it is (see repository note below). **Open: the main session should decide on citing these as preprints, or on the self-citation route in the anonymised copy.** |
| m8 GPU non-determinism per dataset | §4.2 now gives the change per dataset: TON-IoT 0.001 mean, 0.002 max (21 runs); UNSW-NB15 0.006 mean, 0.018 max (25 runs); N-BaIoT 0.000 (5 runs). |
| m9 column order on UNSW | Added: the order SD exceeds the seed SD and is comparable to late − early, so for differences below about 0.02 the MLP results are the cleaner evidence. |
| m10 silent-failure curves | **Not possible from saved outputs.** The min–max run saved only summary results, with no training history. The unsupported phrase "validation loss diverges while training accuracy rises" was removed, and the text says the curves were not saved. A sentence was added framing heavy-tailed inputs as a known pitfall (standardisation or a log transform), with the point being that the failure was silent. |
| m11 quantile fit on the SMOTE set | Now stated in §4.1 as a choice, with fitting on real records only named as an equally valid alternative. |
| m12 Table A.1 seeds column | Column "Several seeds or runs" added. It is based on a fresh full-text check of the 8 FT papers: none reports several seeds; Kiflay gives a 10-fold CV mean without spread; AboulEla state the absence of repeated runs as a limitation. The 6 abstracts do not mention it. §2.1 is updated to match. |
| m13 "1 ties" | Fixed in `put()`. Table 6 now reads "(3/5, 1 tie)". |
| m14 Fig. 1 | Encoder embeddings (64) and the dropout → softmax heads are now shown. A fifth panel shows "Early fusion with head (control)". The caption describes the two controls that are not drawn. |
| m15 abstract density | The attention sentence was shortened and the closing sentence dropped to make room for the footprint and the IoT-scope statements (248 words). |
| m16 Gorishniy et al. 2021 | Added in §6.3 and in the references (NeurIPS 34, pp. 18932–18943, proceedings URL). |
| m17 polemical wording | Now reads "…cannot distinguish fusion from this effect". |

## Numbers audit

| Item | Change |
|---|---|
| W1 "at least 0.997" | `det_min` is now rounded down: "at least 0.996" (RF stream 2 is 0.99653). The text adds "at least 0.997 for the five main CNNs". verify checks the literal bound. |
| O1 abstract bound | See M2. |
| O2 "attention gained nothing" | Highlight 2 was replaced, and Highlight 4 now says "no significant gain". §7 and §6.1 say "no significant gain". §5.2 now reports the MLP attention − early interval on TON-IoT: +0.003 [+0.001, +0.005], 5/5. |
| O3 "reproduced" | "Largely reproduced" in the abstract and §6.1. §5.2 gives ≈73% and late − late-on-all +0.001 [0.000, +0.003], unadjusted p = 0.032, Holm p = 0.135. |
| O4 Highlight 1 | Now "…helps little more than using all features". |
| O5 "rule out" | Now "make the encoder an unlikely explanation, although on UNSW-NB15 they cannot rule it out". |
| O6 MLP "within a few thousandths" | Now restricted to late MLP fusion, with a note that some other MLP models differ more. |
| O7 "not from the number of parameters" | Replaced. See M4. |
| O8 reuse "most" | §4.2 now says that all 30 late-fusion random-split runs and 17 of 20 attention random-split runs (TON-IoT and UNSW-NB15) are reused, while every semantic late and attention run there is new. §5.3 softens "consistent" and notes that TON reruns differ by up to 0.002. §6.3 gives 104 of 120 and says "including all late-fusion random-split runs". |
| O9 "other metrics rank as macro-F1" | Now "PR-AUC ranks the models broadly as macro-F1 does … the benign FPR does not always", with the GBDT/RF and attention/early examples. |
| A1 Highlight 2 baseline | Now "Late fusion beat one CNN on all features by only +0.005 macro-F1 on TON-IoT". |
| A2 Holm-adjusted p unlabelled | Every Holm p in §5.2 is now labelled. Unadjusted p and CI are added for UNSW late − early. The Table 3 caption says that CIs are unadjusted and points to Table B.1. §4.3 says the same. |
| A3 "stayed within" | Now "differed … by +0.002 and +0.006 on average (per split 0.000 to +0.004 and −0.010 to +0.017)". |
| A4 "close to the RF" | Now "within 0.03 of the RF". |
| A5 "1 ties" | Fixed (m13). The Table 6 caption now explains the tie rule. |
| A6 "wider head" | Fixed (M4). |

## References and code

| Item | Change |
|---|---|
| R1 Ayantayo / Dinh 2024a | §3.2 now reads: "Grouping the features of one flow table into separately encoded subsets is common in the fusion literature, although with other rules (Ayantayo et al., 2023, group by data type; Dinh et al., 2024a, give each feature group its own sub-encoder); we know of no study that uses exactly this volume-versus-behaviour split." |
| R2 Attota | The abstract (checked verbatim) mentions neither MQTT nor the three views. §2.1 now says only "an ensemble of networks on multiple views of IoT network data in a federated multi-view IDS". Table A.1 now reads "IoT network data (not named in the abstract)" and "multiple views of the network data". |
| R3 `check_citations.py` | Rewritten: year suffixes, "(2024a, 2024b)" lists, Appendix A/B included, Declarations excluded, both md files checked, CIC-IoT token excluded. Result: 0/0. |
| R4 Dinh 2024b version | The citation now points to v1 (22 Mar 2024). Its Table VIII contains the same 0.727/0.728 comparison (confirmed by the full-text check), and the title and author order match v1. The label 2024b stays. |
| R5 datasets in §6.3 | CICIoT2023 and Edge-IIoTset are now cited, with DOIs checked on Crossref. |
| R6 Wang 2020 | Moved to the clause "joint training is not automatically better than the best single modality". |
| R7 | No change (info). |
| M1 (code) repository | Left as it is, per instructions. The Paper 6 code and results are still not in the public repo, and the main session handles this at submission. No new claim was made. |
| M2 capped run | Now "(N-BaIoT stream 2, seed 4, best epoch 118, so it was still improving at the cap …)". verify checks which run it is. |
| M3 Params column | The header is now "Params (TON-IoT)". |
| M4 thread settings | Clause added in §5.6. The Table 8 caption already stated them. |
| M5 5-seed mean vs single runs | See m5. |
| M6 resplit trees | Now "the RF on all features and on each stream, and the GBDT on all features". |
| M7 "original order" | Now "semantic order, stream 1 then stream 2". |
| M8 "recorded" weights | Now "computed for every test record; class means and within-class spreads are saved". |
| M9 undisclosed pre-processing | §3.1 now lists the >40% missing-column drop, median imputation, ±inf → 99th percentile, constant-feature removal, and label-free category sets fixed before the split with drop-first. This follows the reviewer's reading of `loader.py`. |
| M10 GBDT early stopping | "After 20 iterations without improvement" added. |
| M11 Holm family can shrink | `assert m_ == 8` added in `compute_values.py`. |
| M12 per-seed ρ | Now "computed for each seed's weights against the seed-averaged stream advantage". |
| M13 | No change (info; release notes). |

## Open items needing new runs or a decision by the main session
1. The small CNN tuning grid (referee M3). It would need training.
2. The share of duplicate matches with a different label (M6). It would need the training data reloaded, or the matching labels saved.
3. Training curves for the min–max failure (m10). They would need a rerun.
4. Citation route for the unpublished companion studies (m7).
5. Pushing the code and results to the public repository (refs/code M1).
