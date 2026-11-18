# Paper 4 final numbers and consistency audit (2026-11-07)

Scope: Manuscript_filled.md, Supplement_filled.md (built 2026-11-07 00:35), the source Manuscript.md / Supplement.md, values.json (2026-10-31 04:52), build.py captions, and raw result files under `Papers pipeline/experiments/paper4_federated/results_v2/`. compute_values.py was not run, and no models were trained or datasets loaded.

## What was checked and how

- `verify.py`: **0 failures** (about 5,883 words before references).
- Leftover placeholders: none. There are no `{{`/`⟦` in the filled .md files or in any of the 7 .docx files (main, anon, ieee, supplement, supplement anon). The working header "# Paper 4 v2 manuscript (double-brace placeholders …)" is in the filled .md only and does not reach the .docx.
- Hard-coded (non-placeholder) numbers in Manuscript.md and Supplement.md: almost every result number is a placeholder. The few literal numbers (8,192 records, 192 values, rounds 11–50, 100 epochs, the grids) agree with run_fl_v2.py and compute_values.py: `fed_recal(per_client=8192)`, 96 channels × 2 statistics, and `history[10:]`.
- No stale values. No non-archived seed*.json under results_v2 is newer than values.json.
- Independent recomputation from the raw seed files (my own scripts, separate from verify.py and compute_values.py) **matches the manuscript to the last digit** for:
  - Table I: every TON-IoT row tested (central, local α 0.5/0.1, FedAvg IID/1.0/0.5/0.1, FedAdam 0.5, SCAFFOLD 0.5, FedAvg-SGD 0.5, FedLC 0.1), for mean ± CI, SD and the final round.
  - Table II: FedAvg host 0.720 ± 0.099 (SD 0.080) and local 0.530 ± 0.040. Table III: central, FedAvg-dev, local-dev, IID9, α 0.5/0.1 N9, FedAdam-dev.
  - Table IV: DP sanity runs, σ = 0.6, σ = 2.5, full budget σ = 1.8 and no-noise. The 20×1 non-private row comes from `fedavg_gn_short` (0.722, 0.726, 0.759 → 0.735).
  - **All 41 rows of Table S1**: mean differences, uncorrected p and Holm-adjusted p, with family sizes 19/13/9. All † marks are correct. Edge cases: UNSW IID p = 0.00057 (shown as "< 0.001", correct); TON α = 0.1 vs local p = 0.00101 (shown as 0.001, correct); SCAFFOLD vs Adam-FedAvg p = 0.00096 (shown as "p < 0.001", correct).
  - Own-traffic reversal on TON-IoT hosts: 17/20 pairs, −0.121 ± 0.083, p = 0.015; detection 99.7% vs 96.5%; FPR 2.6% vs 3.4%.
  - Own-traffic reversal on N-BaIoT: 43/45, −0.081 ± 0.014, p = 0.0001; device as unit p = 0.006, 9/9 devices worse; detection 99.9% vs 99.9%; FPR 0.8% vs 1.6%.
  - Fixed vs redrawn host order: 0.770 ± 0.031 vs 0.720 ± 0.099.
  - TON training counts: password 13,893, backdoor 13,096, ransomware 9,986, scanning 13,031, MITM 727. Total 128,484.
  - Host-partition own-test sizes 5,175–11,147. Client sizes 23,741–52,783.
  - N-BaIoT device training records: the Table S3 column sums to 110,227. Own-test sizes 584–3,949.
  - DP per class against the same code without noise: 5 classes lose ≥ 10 points at σ = 0.6; 7 of 9 attack classes at the full budget; password 0.85 → 0.31.
  - DP per class against non-private FedAvg (S7): 4 classes at σ = 0.6, 5 at σ = 2.5.
  - Repetition counts: core and fed-recal runs 5; scaling, central BN, 100-round and DP runs 3; DP tuning 2; N-BaIoT tuning 2; TON tuning 3.
  - 95.0 kB per model, 0.76 MB per round, 38.0 MB per run, 12.8 MB raw training matrix.
- "10 of the 17 primary comparisons of FedAvg with a reference are significant": correct (TON 3/7, UNSW 2/5, N-BaIoT 5/5).
- Repository commit 08673aa exists in `release_v2/gh/IREAP-IoT-IDS-v2` (origin = github.com/Amoakoemma123/iot-ids-pipeline). It is on origin/main and is the "Paper 4 round 6 … results" commit. The local `fedavg_natr/seed0.json` there matches the local file.
- Grid-edge claims match the tuning directories, including the N-BaIoT grid exclusions.

---

## WRONG

**W1. The device-level vs Dirichlet comparison is wrong for "stronger" skew.**
- Location: main text, V-C, end of "Cost against pooled training".
- Quote: "Device-level partitions thus tend to cost more than Dirichlet partitions of similar or stronger label skew".
- Correct values:
  - Against the *stronger* skew (α = 0.1), device-level partitions do not cost more. TON-IoT host partition −0.179 vs α = 0.1 −0.185, about equal. N-BaIoT device partition −0.105 vs α = 0.1 −0.214, so it costs about **half as much**.
  - The claim holds only against α = 0.5. TON −0.179 vs −0.059 (Welch −0.120, p = 0.024). N-BaIoT −0.105 vs −0.057 (Welch −0.048, p = 0.003).
- Source: Table S1 rows; S4 "Device-level against Dirichlet partitions".
- Fix: say "…than Dirichlet partitions of similar (α = 0.5) label skew", as the Conclusion and the cover letter already do ("with a similar label mix").

**W2. Smallest UNSW DP cost misrounded (minor).**
- Location: Supplement S7, first paragraph.
- Quote: "On UNSW-NB15 … noise costs 4 to 7 points against same-schedule non-private training".
- Correct value: the smallest cost is 0.4080 − 0.3627 = 0.0453 (σ = 0.6), which rounds to **5** points. The largest is 0.4080 − 0.3360 = 0.072 (7 points). The range is "5 to 7" (or "4.5 to 7").
- Source: raw `UNSW_NB15/fedavg_gn_short` mean 0.40795 and `dp_sigma0.6` mean 0.36267; Table S7.

## OVERSTATED

**O1. "Never improve significantly" is missing "after correction".**
- Locations:
  - Abstract: "With matched tuning, FedProx, FedAdam, SCAFFOLD and FedLC never improve significantly on FedAvg".
  - Conclusion: "FedProx, FedAdam, SCAFFOLD and FedLC did not improve significantly on it".
  - The cover letter's finding (3) has the same wording.
- Why it is overstated: before correction two primary comparisons *are* significant improvements.
  - UNSW FedProx α = 0.1: +0.045 [+0.011, +0.080], p = 0.021.
  - UNSW FedLC α = 0.5: +0.063 [+0.015, +0.110], p = 0.021.
  - Holm-adjusted p = 0.187 for both.
  - Contribution 4 and V-B correctly say "after correction".
- In addition, SCAFFOLD is compared with FedAvg-SGD, not with FedAvg.
- Source: Table S1.
- Fix: "…do not improve significantly on FedAvg after Holm correction", and drop "never".

**O2. "The extra machinery buys nothing".**
- Location: V-B, last sentence.
- Quote: "Within the resolution of five repetitions, the extra machinery buys nothing over tuned FedAvg."
- Why it is overstated: this is absence of evidence, not a shown equivalence.
  - UNSW FedLC α = 0.5 is +0.063 [+0.015, +0.110]. That is larger than the stated detectable difference (0.04 on UNSW) and significant before correction.
  - N-BaIoT FedAdam is +0.028 [−0.012, +0.068].
- Source: Table S1; values `UNSW_mde` = 0.04.
- Fix: "…none of the extra machinery gives a gain that survives correction".

**O3. "The same conclusions hold … under the device-level partitions".**
- Location: V-B.
- Quote: "The same conclusions hold without port features and under the device-level partitions, although the order of the weaker methods changes."
- Why it is overstated: the FedAdam-worse finding does not carry over.
  - On N-BaIoT FedAdam has the *best* point estimate (0.920 ± 0.024 vs FedAvg 0.893; +0.028, p = 0.125).
  - On the TON-IoT host partition FedAdam is −0.076, p = 0.020 uncorrected, Holm 0.202 (not significant).
  - "The order of the weaker methods changes" understates a worst-to-best flip.
  - The "no improvement after correction" part does hold.
- Source: Table III; Table S1.

**O4. "FedAdam is significantly worse on TON-IoT" is unqualified.**
- Locations: Abstract; Contribution 4.
- Why it is overstated: the only significant result is at α = 0.5 (−0.083, Holm p = 0.042). At α = 0.1 it is −0.046, Holm 0.119. On the host partition it is −0.076, Holm 0.202.
- V-B states this correctly ("the only algorithm difference significant after correction").
- Source: Table S1.
- Fix: add "at α = 0.5".

**O5. "Several repetitions" for SCAFFOLD collapse (minor).**
- Location: Supplement S3, last paragraph.
- Quote: "At learning rate 0.3 SCAFFOLD collapses in several repetitions with non-finite weights".
- Correct value: **2** repetitions (verify.py check: "SCAFFOLD lr 0.3 at alpha 0.1 produced non-finite weights in 2 repetitions").
- Fix: "in two of the repetitions".

## AMBIGUOUS

**A1. The primary-family description does not fit N-BaIoT.**
- Location: III-D.
- Quote: "…against the local-only reference at α = 0.5, α = 0.1 and the device-level partition, and FedProx, FedAdam and FedLC against FedAvg … at α = 0.5, α = 0.1 and the device-level partition (19, 13 and 9 comparisons)".
- The counts are correct.
- The N-BaIoT family of 9 has no local-only and no algorithm comparisons at α = 0.5 or 0.1. It contains dev−central, dev−local, the 4 algorithms on dev, and IID/α 0.5/α 0.1 − central. The sentence qualifies only the centralised comparisons with "(where one exists)".
- Fix: "(on N-BaIoT, local-only and algorithm comparisons on the device partition only)".

**A2. Mixed repetition counts are not stated everywhere.**
- III-D: "Core results use five repetitions; the scaling study, the batch-normalisation runs with server-averaged statistics only, and the DP study use three." This omits:
  - the 100-round check (3 repetitions; stated only in S4);
  - the DP tuning runs (2; stated in S7);
  - the tuning repetitions (3/2; stated in S1-D).
- Table IV caption says "three repetitions", but its "Non-private FedAvg 50 × 2" row has five (the caption's "repeats Table I" implies this).
- Comparisons between 3-run and 5-run sets are unpaired:
  - V-E "16 against non-private FedAvg" (0.840, 5 runs, vs 0.683, 3 runs);
  - S4 "narrows the TON-IoT gap at α = 0.5 by about a third" (−0.059 from 5 runs vs −0.042 from 3 runs);
  - S6 scaling (N = 4 from 5 runs vs N = 2/8/16 from 3);
  - Fig. 2, which mixes the five FedAvg runs with the 3-run centralised BN model.
- Fix: say once that these cross-set contrasts are unpaired, and add "100-round check" to the III-D sentence.

**A3. Personalisation: run sets and ± units are mixed.**
- Locations: Table S4 labels and V-C "Personalisation".
- The rows "Fine-tuning, same lr; with the federated model" read as if all columns score the pair. Per the caption, only the three alert columns do; the own-traffic difference and common-test macro-F1 are for the adapted model alone.
- The ± unit differs: repetitions (n = 5) on TON-IoT, devices (n = 9) on N-BaIoT. The main text places "−0.130 ± 0.052" (TON) next to "−0.073 ± 0.045" (N-BaIoT) without saying so.
- Fix: rename the rows, for example "Fine-tuning, same lr (alert columns: with the federated model)", and give the unit in the text.

**A4. Conclusion scope of "own model is usually better".**
- Location: Conclusion.
- Quote: "…in an exploratory analysis a site's own model is usually better on its own traffic, also without port features."
- This holds for the device-level partitions (17/20, 43/45, 16/20 without ports).
- Under Dirichlet partitions on TON-IoT (own classes, common test set) only 3/20 and 4/20 clients do better alone, and FedAvg is better by 0.049 and 0.068. On UNSW-NB15 it is 12/20 and 12/20.
- Source: S3, S4.
- Fix: "under device-level partitions, a site's own model is usually better…".

**A5. "Most of the cost appears with the first noise".**
- Location: V-E.
- Quote: "On the short schedule, noise costs 8 points … at σ = 0.3 (ε ≈ 117), 8 at ε = 12.15 … and 10 at ε = 0.81: most of the cost appears with the first noise".
- Table IV's first noise level is σ = 0.2 (ε ≈ 402). It costs 5 points (0.735 → 0.681; S7 says so), about half of the eventual 10.
- The main text skips σ = 0.2, so "first noise" silently means σ = 0.3.
- Fix: "most of the cost appears by σ = 0.3 (5 points already at σ = 0.2)".

## OK but note

1. **The PDFs are stale.** Manuscript_v2.pdf, _anon.pdf and _ieee.pdf (2026-10-31) lack 4 sentences that are in the 2026-11-07 .docx:
   - "A client's own test traffic is the test traffic to its hosts (5,175 to 11,147 records with 7 to 10 classes per site)."
   - the AI-assistance sentence in Data/code availability;
   - both Acknowledgment sentences.

   The Supplement PDF matches its .docx. Regenerate the PDFs before upload.
2. **DDoS at the full DP budget is borderline.** "only DDoS and DoS and benign traffic lose less than ten points" (V-E) and "7 of 9 attack classes lose ten points or more" (Abstract) are true, but DDoS loses 9.8 points (0.9313 → 0.8337). With two-decimal numbers (0.93 → 0.83) a reader will read it as 10. Consider "(DDoS: 9.8 points)".
3. **The BN finding is stated generally but is TON-specific.** "federated re-estimation of the statistics removes the swing" (Abstract) and "Batch normalisation should not be used with server-averaged running statistics" (Conclusion) rest on TON-IoT. On UNSW-NB15 the swing is small either way (0.010 / 0.007; S6). The text says so in V-D but not in the Abstract or Conclusion.
4. "so our group-normalised results are, if anything, slightly conservative" (V-D) rests on exploratory, uncorrected p = 0.018 / 0.007. That is acceptable as worded, but it is not a primary result.
5. **The MITM trajectory is not monotone.** S7: "the rare MITM class … falls from 0.24 to 0.13 at σ = 0.6 and is not detected at all from σ = 1.5". At σ = 0.2 and 0.3 MITM F1 is 0.051, *lower* than at σ = 0.6 (0.134), and at σ = 1 it is 0.068. The sentence implies a steady decline; MITM is erratic.
6. **N-BaIoT proximal fine-tuning "does neither" holds at one epoch only.** V-C: "whereas the proximal term does neither (0.845 ± 0.026; −0.041 ± 0.028)" is true at one epoch. At 5 epochs proximal (−0.025, 0.850) beats same-lr fine-tuning (−0.032, 0.832) on both counts (Table S4). Add "after one epoch".
7. **The detection figures come from different runs.** "both models detect 99.9% of a device's own attacks" (V-C, main run) sits beside Table S4's "Federated model alone 100.0" (personalisation re-run). Both are correct for their run sets. The TON 96.5% vs 91.1% split is explained in the text; the N-BaIoT one is not.
8. **"Still trails" at α = 0.1 rests on a CI that includes zero.** S4 100 rounds: "FedAvg trails the centralised model by … −0.227 [−0.498, +0.045] (α = 0.1)". The UNSW IID/α = 0.1 intervals also include 0. "Trails" is a point-estimate statement; the main text's wording ("leaves the gaps within their intervals") is fine.
9. **The cover letter repeats O1** ("do not improve significantly", with no "after correction").
10. **There is no Highlights section in the package.** Index Terms are consistent with the content. If the target journal asks for highlights, none exist to audit.

Also verified OK, with no action needed:
- Abstract numbers: +0.190, +0.187, 17/20, 16/20, 43/45, ε ≈ 2.9, 17 points, 7 of 9.
- "halves it on N-BaIoT": −0.073 → −0.034, 53% closed.
- 89% and 28% shares of the gain given up: rough check 88% and 29% from the means; the per-repetition shares are in values.
- Swing 0.110 / 0.016 / 0.018.
- BN−GN differences +0.032 and +0.039.
- Final round 0.855 ± 0.037 vs 0.634 ± 0.339.
- Two-model alerting 99.9% / 4.1% / 16.4%.
- Host vs α = 0.1 cost: −0.179 vs −0.185.
- SD 0.080 vs 0.034.
- The 80% power detectable differences: 0.05 / 0.04.
- Exact signed-rank tests with n = 5 cannot reach p < 0.05: true, minimum two-sided p = 0.0625.
- "eight-page limit" was not checked.

---

## Counts

| Class | Count |
|---|---|
| WRONG | 2 (W1, W2) |
| OVERSTATED | 5 (O1–O5) |
| AMBIGUOUS | 5 (A1–A5) |
| OK but note | 10 |

Most serious:
- **W1**: "similar or stronger label skew" is contradicted by α = 0.1 on both datasets.
- **O1**: Abstract and Conclusion say the algorithms "never" / "did not" improve significantly without "after correction", while two UNSW improvements are significant uncorrected.
- **O3**: "same conclusions hold under device-level partitions", while FedAdam flips to the best point estimate on N-BaIoT.
- **Stale PDFs** (note 1).
