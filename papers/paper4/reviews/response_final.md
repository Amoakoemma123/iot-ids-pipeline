# Paper 4: response to the final review round (2026-11-07)

Reviews: review_final_referee.md (6.5/10), review_final_numbers.md, review_final_refs_code.md.
Part A (text, no training) is done. Part B (diagnostic runs) is prepared but not run: code in
`Papers pipeline/experiments/paper4_federated/run_p4final.py`, queue `Papers pipeline/logs/p4final_queue.sh`, results go
to `results_p4final/`. compute_values.py reads them when present. Until then the text shows 33 pending `⟦P4F_…⟧`
placeholders (10 in the main text, 33 in the supplement). verify.py reports them as pending, not as failures.

Build: compute_values → build (main, --anon, --ieee, --supp, --supp --anon) → scrub on every docx → submission_docs
(cover letter) → verify: **0 failures**. IEEE version: **8 pages** (Word page count via page_count.ps1; References end on
page 8 with about two lines to spare). Abstract: 219 words.

## Numbers audit (review_final_numbers.md)

| Item | Change |
|---|---|
| W1 device vs Dirichlet "similar or stronger" | V-C now says "tended to cost more than Dirichlet partitions of similar label skew (α = 0.5), but not more than α = 0.1". S4 adds the Welch differences against α = 0.1 (TON −0.179 vs −0.185 is about equal; N-BaIoT costs about half). verify checks this. |
| W2 S7 "4 to 7" | The DP costs are now computed from unrounded means (compute_values), so UNSW reads "5 to 7". TON costs are unchanged (5/8/8/9/10). |
| O1 "never improve significantly" | Abstract: "do not improve on tuned FedAvg after Holm correction". Conclusion: "after Holm correction". Cover letter finding (3): "after Holm correction". |
| O2 "buys nothing" | Removed. V-B now says no gain survives correction, but only 12 of 24 algorithm comparisons are equivalent at ±0.05 (TOST, 90% CI); the others, including FedLC on UNSW (+0.063 [+0.015, +0.110]), are inconclusive. |
| O3 "same conclusions hold under device partitions" | Removed. V-B gives FedAdam on N-BaIoT (best point estimate, +0.028 [−0.012, +0.068]) and the uncorrected FedAdam deficits at α = 0.1 and on the host partition. |
| O4 FedAdam worse unqualified | Contribution 4 and the abstract no longer claim it. V-B says "the only algorithm difference significant after correction" at α = 0.5 and adds the tuning caveat (server lr only, at FedAvg's client lr; referee M4). |
| O5 "several repetitions" | The SCAFFOLD lr-0.3 sentence is removed (see M2 of the code review). |
| A1 primary family on N-BaIoT | III-D: "on N-BaIoT, comparisons with the local-only reference and between algorithms use the device partition only". |
| A2 repetition counts | III-D adds the 100-round check (3) and "contrasts between run sets of different sizes are unpaired". Table IV caption: non-private 50 × 2 has five repetitions. S4 (100 rounds) and S6 (scaling, Fig. 2) are labelled unpaired. The full-budget DP gap is now paired: 0.173 [0.144, 0.202]. The contrast with non-private FedAvg is labelled unpaired. |
| A3 personalisation rows and ± units | Table S6 rows renamed "… (alert columns: with the federated model)". The S5 text says which columns score the pair. The main text gives the units ("± over devices" on N-BaIoT). |
| A4 "usually better on own traffic" | Conclusion: "Under device-level partitions and random record splits, …". |
| A5 "most of the cost with the first noise" | V-E now quotes σ = 0.2 (5 points, ε ≈ 402) and σ = 0.3 (8 points) and says the curve is flat from σ = 0.3. |
| Note 1 stale PDFs | Not regenerated here (Word output only). Regenerate the PDFs from the new .docx before upload. |
| Note 2 DDoS borderline | V-E: "(DDoS 9.8 points)". |
| Note 3 BN finding TON-specific | Abstract and Conclusion now say "On TON-IoT"; V-D keeps "On UNSW-NB15 the swing is small either way". |
| Note 5 MITM not monotone | S7 lists MITM F1 at every σ (0.24 / 0.05 / 0.05 / 0.13 / 0.07 / 0) and calls it erratic. verify checks the non-monotonicity. |
| Note 6 proximal "does neither" only at one epoch | Main text: "after one epoch, the proximal term does not". The low-lr and proximal details moved to S5 / Table S6. |
| Note 7 detection from different run sets | S5 now says that Table S6 comes from the personalisation re-run and gives both run sets' detection rates. |
| Note 8 "still trails" with a CI including 0 | S4: "differences … are" (no longer "trails"), plus "some intervals include zero". |
| Note 9 cover letter O1 | Fixed (see O1). |
| Note 10 Highlights | None needed for IoT-J. Not added. |

## References and methods vs code (review_final_refs_code.md)

| Item | Change |
|---|---|
| R1 check_citations.py crashes | Rewritten as a numeric IEEE check: 42 references, all cited, first cited in order. |
| R2 [22] evidence | Re-verified on the arXiv API (literature_verified.md). |
| R3 [1] Diro | Sec. I: "[1]" now supports "need traffic from many sites"; "unable or unwilling to pool" cites [2] only. |
| R4 [11]/[12] | III-A: "network flows [12] from an IoT/IIoT testbed [11]". |
| R5 Rey et al. [23] | Reworded to what the paper's abstract states: it compares federated, local and centralised malware detection on N-BaIoT. |
| R7 literature_verified.md | Created: all 42 references re-verified on Crossref (30 DOIs), arXiv (3) and OpenAlex plus proceedings pages (10). Recent [5], [30] and [35] confirmed; [30]'s title grammar is as published. Raw records are in reference_check_final*.json. |
| M1 S1D "diverged" | "values that diverged on both other datasets (SGD learning rate 1) or scored lowest in their grids there (μ = 10⁻¹, server learning rate 0.003)". |
| M2 SCAFFOLD lr 0.3 rests on superseded runs | Claim removed from S3. The verify check that read `_superseded_r2` is removed. |
| M3 commit 08673aa | Not changed. The local repo has no remote; confirm that the public mirror's run_fl_v2.py equals local HEAD before submission (open item). |
| M4 BN re-estimation at scoring only | V-D: "we score each round's model with these statistics, while training continues with the server-averaged ones". |
| M5 SCAFFOLD averaging | IV: "(option II; model updates size-weighted, control variates averaged uniformly)". |
| M6 centralised round = 2 epochs | III-C: "checkpointed and selected every two epochs"; V-D: "two-epoch checkpoints"; S1C explains the selected "round". The Table I caption already said so. |
| M7 DP reference | V-E: "against the same code without clipping or noise" (also S7). |
| M8 guard definition | III-A and S1-A: "identical in features and label to a training record". |
| M10 ε range | Table IV caption: "for each repetition the maximum over clients …, shown as mean (minimum–maximum) over repetitions". |

## Referee (review_final_referee.md)

| Item | Change / status |
|---|---|
| M1 own-traffic reversal vs random split | **Text:** the abstract, Contribution 2, V-C and the Conclusion now qualify the reversal with "under random record splits, which favour [the local model]". V-C adds two observations that point to site-specific fitting: on other sites' traffic of a site's classes FedAvg is better (+0.138); N-BaIoT local models reach 0.97–0.99 on own traffic. Discussion and Limitations call for blocked or temporal splits. **Runs (pending):** a blocked per-site split (contiguous blocks in file order per host/device and class: 65% train, 5% gap, 12.5% val, 5% gap, 12.5% test) plus a random control with identical group sizes. FedAvg vs local, 3 repetitions, TON-IoT hosts and N-BaIoT devices. Text with placeholders is in V-C and S5. TON-IoT has no timestamp column in train_test_network.csv, so its split is blocked in file order and is described as "not necessarily temporal". N-BaIoT file order is capture order. **After the runs:** reread the abstract and conclusion wording against the result. |
| M2 label-prior correction | Not computable from saved results: only per-site scores were saved, not predictions. Added to the runs at no extra training cost: every FedAvg run in run_p4final scores the selected model with prior correction ('prior') and with masking only ('mask'). FedAvg re-runs on the original split (seeds 0–2) are paired with the saved local models. Text with placeholders is in V-C and S5. |
| M3 DP optimiser / batch | **Text:** V-E says the cost appears at noise levels with no meaningful privacy and may reflect Adam's rescaling. The full-budget gap is now a paired CI. S7 notes that no clipping-only full-budget run exists (minor 24). **Runs (pending):** at the ε of the σ = 1 runs (≈3, short schedule, seeds 0–2, so ε is identical): SGD with momentum 0.9 (lr 0.03/0.1, chosen on validation), and Adam with B = 1024 with σ recalibrated per repetition (≈1.64). Optional full budget (P4_DP_FULL=1): SGD-momentum at the chosen lr, and B = 1024 (σ ≈ 3.43). Text with placeholders is in V-E and S7. |
| M4 FedAdam tuning, "buys nothing" | Tuning caveat added (V-B, IV). "Buys nothing" replaced by the TOST count (12/24 equivalent at ±0.05). A 2-D FedAdam grid was not run; the claim is qualified instead, as the referee allowed. |
| M5 uncited premise | No audit of published studies was done, and no examples could be verified without full-text reading. The Introduction (and the cover letter) no longer attribute these practices to the literature: "How much FL helps an IDS depends on evaluation choices … Rather than audit published studies for these choices, we show how much each matters." The abstract is reworded the same way. |
| M6 GPU nondeterminism | Distribution reported: mean absolute re-run difference 0.036, SD 0.049, max 0.109 (15 pairs; V-C, S1C). S1C says the deterministic mode was not tried. The 96.5% vs 91.1% detection gap is no longer interpreted ("a re-run … detected 91.1%, so we do not interpret the detection gap"). |
| minor 1 BN at selected round | V-D: no loss at the selected round (+0.022 vs GN), so the swing matters for deploying an arbitrary round. Conclusion: "should not be deployed at an arbitrary round". |
| minor 2 ranking only under GN | Limitations: "algorithms are compared with group normalisation only". |
| minor 3 DDoS | V-E: "DDoS 9.8 points". |
| minor 4 injection / longer schedule | V-E: "Part of this cost falls on classes that only the longer non-private schedule learns." |
| minor 5 validation not protected | IV: "the validation records used for round selection are not protected". |
| minor 6 slope coding | V-A: "skew levels coded 0–3". |
| minor 7 notation | "FedAvg − local", "FedAvg is worse in … pairs", "worse on 9 of 9 devices". |
| minor 8 "closes" | Abstract and V-C: "parity within our resolution". |
| minor 9 late selected rounds | V-A: selected rounds cluster near the end (round 47 of 50); the 100-round check is referenced. |
| minor 10 selected-round column in Table III | Not added (space). |
| minor 11 feature skew | V-C names feature skew between devices as the likely mechanism and cites FedBN [31]. |
| minor 12 FedBN not run | Limitations names FedBN and Ditto as not evaluated. |
| minor 13 GBDT anchor | Not added (space, and needs training). |
| minor 14 feature selection fitted centrally | Limitations: "Preprocessing, including feature selection, is fitted centrally". |
| minor 15 t-test sentence | Rephrased as the referee suggested. |
| minor 16 abstract length | 249 → 219 words. |
| minor 17 91.1% placement | Now in the detection sentence as a re-run caveat (M6). |
| minor 18 grids | Folded into the recommendation list ("tune baselines and widen grids with edge optima"). |
| minor 19 union rows in Table II | Not added (8-page limit). The union numbers stay in the text and Table S6. |
| minor 20 Zenodo DOI | Open (needs the user's account). |
| minor 21 recent refs | Verified (literature_verified.md). |
| minor 22–23 figure cosmetics | Not changed. |
| minor 24 clipping-only full budget | Absence noted in S7. |
| IoT-J fit | Deployment: the 23,754-parameter model is small enough for an IoT gateway; device-side cost was not measured. |
| Cover letter | States that rebuilding the N-BaIoT pipeline changed the own-traffic result from −0.032 (p = 0.147, 6/9 devices) to −0.081 (p = 0.006, 9/9). The old value was re-computed from `_superseded_r5`. |

## Space (8-page IEEE limit)
To make room: Table III keeps the references and partitions (6 rows); all N-BaIoT rows, including the algorithms, are
in the new Supplementary Table S3 (supplement tables after S2 renumbered by one). Table labels were shortened. Related
Work, Discussion and Limitations were condensed. The pooled-validation deployment note, the low-lr/proximal
fine-tuning details and the UNSW duplicate-vector disclosure moved to the supplement (S1C, S5, S3).

## Pending after the queue finishes
1. Run compute_values.py, then all builds, scrub and verify (the P4F placeholders fill automatically).
2. Reread V-C, V-E, the abstract and the Conclusion against the outcomes:
   - If the blocked split removes or reverses the own-traffic advantage, say so in the abstract and the Conclusion.
   - If the prior correction closes the gap, recommend "federate, then apply the local label prior".
   - If SGD-momentum or B = 1024 shrinks the DP cost, report the better configuration.
3. Check the IEEE page count again. The margin is about two lines.
4. Regenerate the PDFs and confirm the public commit (M3).
