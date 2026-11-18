# Paper 4 v2 — Round 4 number audit (Manuscript_filled.md, Supplement_filled.md)

Date: 2026-10-26. I recomputed every number from the raw `results_v2/<dataset>/<experiment>/seed<r>.json` files with my own script. I did not import compute_values.py or use values.json, except to see which placeholder a sentence uses. Script and full output:
`C:\Users\G15165~1\AppData\Local\Temp\claude\C--Users-G15-1650\a140992e-6b2a-45f9-a1f9-a63388cafb8a\scratchpad\audit\audit.py` and `audit_out.tsv` (787 rows).

Methods: mean ± t-based 95% half-width (ddof=1). Paired differences use the shared seed index, with a paired t-test. Holm adjustment (step-down, monotone) is applied within each dataset's Table S1 family. Own-traffic statistics use `site_selected` and `per_client[j].site_selected`. Fine-tuning uses `finetune[j][E]`. Label skew uses `client_class_counts`. DP uses `dp.epsilon_max` and `per_class_f1`.

## 1. Coverage

| Block | Numbers checked |
|---|---|
| Table I (TON main, 17 rows × 5 columns) | 85 |
| Table S2 (UNSW main) | 85 |
| Table II (TON hosts, incl. own-host column) | 40 |
| Table III (N-BaIoT) | 44 |
| Table IV / Table S6 (DP: ε, selected, final) | 40 |
| Table S1 (43 comparisons: diff+CI, p, Holm; plus 3 family sizes) | 132 |
| Table S3 (per device, 9 × 6) | 54 |
| Tables S4/S5 (scaling, BN; plus MB/round) | 38 + 10 |
| Text: exploratory diffs, 100-round, ports, trend, per-client, own traffic, fine-tuning, skew, sizes, comms, swing, BN, DP costs/tuning/per-class, tuning selections, rerun noise, majority F1 | ~150 |
| **Total concrete stated values compared** | **≈ 660** (plus 18 tuning-grid selections, 8 partition-pairing integrity checks and about 40 context recomputations) |

Integrity checks:
- Local-only and FedAvg runs share client sizes in every repetition (α = 0.5 and 0.1 on both datasets, the TON host partition, the N-BaIoT devices). The fine-tuning runs share them too.
- Every tuned value in S1 D is the argmax of the mean validation macro-F1 over the tune3 runs. The edge claim (only FedProx μ on TON-IoT) is correct.
- The main runs use the tuned hyperparameters (checked from `cfg`).

## 2. Numerical discrepancies

None of the ≈660 values differs from its recomputed value at the printed precision. Every value in Tables I–IV and S1–S6 matches, as do the Holm-adjusted p-values and † marks, the family sizes (19/13/11) and all prose restatements. The counts also match: 17 of 20, 6 of 9, 1/20, 0/20, 3 and 4 of 20, 12 and 12 of 20.

There is one ambiguity that a reader is likely to misread:

| # | Location | Stated | Recomputed | Severity |
|---|---|---|---|---|
| N1 | S6, last sentence ("On UNSW-NB15 … batch normalisation differs from group normalisation by …") | +0.003 [-0.006, +0.011], p = 0.457 | This is **server-averaged** BN vs GN (fedrecal runs, `run_vs_gn`). The sentence right before it is about BN *with federated re-estimation*, so a reader will take it as that comparison. With federated re-estimation the value is **-0.002 [-0.013, +0.009], p = 0.657**. | Minor (wording). The conclusion is the same either way. |

## 3. Unsupported, overstated or internally inconsistent claims

| # | Location | Claim | What the data show | Severity |
|---|---|---|---|---|
| C1 | Abstract, last sentence | "Record-level DP-SGD at ε ≈ 2.9 costs 17 macro-F1 points on TON-IoT, **concentrated in a few attack classes**." | In the full-budget runs (σ = 1.8, ε = 2.93, no-noise reference `dp_full_nonoise`), **8 of 9 attack classes lose ≥ 10 points**: password -54, ransomware -23, injection -22, scanning -15, xss -15, MITM -14, backdoor -11, DDoS -10. Only DoS (-5) and benign (-3) are spared. "Concentrated in a few classes" is true only of the 20-round short-schedule runs, and V.E draws it from those. The abstract attaches it to the full-budget number, where it does not hold. | **Major** |
| C2 | Conclusion | "Record-level DP-SGD at single-digit ε **erases several attack classes**." | No class is erased in the full-budget run (password 0.85→0.31, MITM 0.28→0.14). In the short-schedule runs only MITM reaches 0, from σ = 1.5, and it scores only 0.24 even without noise. Password drops to 0.15–0.18. Backdoor, ransomware and scanning lose 10–20 points but stay at 0.70–0.82. "Erases" fits at most one or two classes. | Moderate |
| C3 | VI Discussion | "most federation gaps are large but, with five repetitions, **not individually significant** after correction" | Of the 17 primary federation-vs-reference comparisons (vs centralised or local-only), **9 are significant after Holm**: TON 3/7, UNSW 2/5, N-BaIoT 4/5. Some gaps are not large either (UNSW local α = 0.5: +0.032; N-BaIoT IID: -0.020). | Moderate |
| C4 | V.C, first paragraph | Host-partition FedAvg "varies **much more** between repetitions **than under Dirichlet partitions** (standard deviation 0.080)" | The SD is 0.080 on the host partition against **0.084 at α = 0.1** (and 0.034 at α = 0.5, 0.009 at α = 1). It varies more than at α = 0.5, not more than under Dirichlet partitions in general. | Moderate |
| C5 | Abstract; Contribution 3; Conclusion | "At matched label skew, device partitions cost more against pooled training than Dirichlet partitions." / "Device-level partitions cost more than Dirichlet partitions with a similar label mix." | Neither supporting comparison is significant. TON: -0.120 [-0.257, +0.017], p = 0.071 (exploratory). N-BaIoT: -0.024 [-0.059, +0.012], Holm p = 0.553. V.C hedges this properly ("Neither … significant … both point the same way"). The abstract, Contribution 3 and the Conclusion state it as an established result. | Moderate |
| C6 | V.B, end | "The **ranking** is the same without port features and under the device-level partitions." | The strict rankings differ. TON α = 0.5: FedLC > FedProx > FedAvg > FedAdam > SCAFFOLD > FedAvg-SGD. No ports: FedLC 0.650 > **FedAvg 0.646 > FedProx 0.618** > FedAdam 0.522 > SCAFFOLD 0.514 > FedAvg-SGD 0.502. TON hosts: FedLC > FedAvg > FedProx > **FedAvg-SGD 0.674 > SCAFFOLD 0.670 > FedAdam 0.644**. N-BaIoT: FedAvg > FedProx > FedLC > FedAdam > SCAFFOLD > FedAvg-SGD. What holds is the *conclusion*: no algorithm beats FedAvg, and FedAdam/SGD trail. Reword. | Minor–moderate |
| C7 | V.A and S3 (trend) | "exploratory linear trend -0.046 per level, p < 0.001; UNSW-NB15: -0.036, p = 0.001" | Slopes reproduce, but the p-value comes from OLS on 20 pooled points. That treats four correlated measurements per repetition (same centralised run, same seed) as independent. Repetition-level slopes tested over the 5 repetitions give p = 0.019 (TON) and p = 0.030 (UNSW). The stated p-values overstate the precision; the direction is unchanged. | Minor–moderate |
| C8 | III.D, family definition vs Table S1 | "FedAvg against the centralised and local-only references at every partition, and FedProx … at α = 0.5, 0.1 and the device partition (19, 13 and 11)" | The counts match Table S1, but the description does not. (a) There is no local-only comparison at IID (any dataset) or at N-BaIoT IID/α = 0.5/α = 0.1. (b) N-BaIoT has algorithms only on the device partition. (c) The N-BaIoT family contains two partition-vs-partition comparisons (device − α = 0.1, device − α = 0.5) that the definition does not mention. The analogous TON comparison (hosts − α = 0.5) is treated as exploratory, so the two datasets are handled differently. | Minor |
| C9 | V.C own traffic (TON vs N-BaIoT) | N-BaIoT "points the same way, more weakly … p = 0.147" | TON uses the repetition as unit (n = 5) and N-BaIoT the device (n = 9). With the repetition as unit, N-BaIoT gives **-0.032 ± 0.025, p = 0.025**, and local is better in 32 of 45 device–repetition pairs. The device unit is defensible for a fixed partition, but "more weakly" partly reflects the choice of unit. State the unit, or report both. | Minor |
| C10 | V.C fine-tuning | The fine-tuned models score 0.551 / 0.779 on the common test set, "against 0.720 ± 0.099 and 0.820 ± 0.024 for the federated" | The fine-tuning comes from separate re-runs (`fedavg_natr_ft`, `fedavg_dev_ft`). Their own federated models scored **0.700 ± 0.059** (TON) and **0.809 ± 0.021** (N-BaIoT), and their pre-fine-tuning own-traffic gaps were -0.138 ± 0.063 and -0.043 ± 0.048. The like-for-like loss is somewhat smaller than implied (TON 0.700→0.551). The conclusion is unchanged; quote the same-run reference or note the rerun noise (up to 0.084, S1 C). | Minor |
| C11 | V.D vs S6 vs Table S4 (server-averaged BN) | Main V.D: swing "0.110 for FedAvg at α = 0.5". S6: "from 0.151 to 0.017". Table S4: BN α = 0.5 0.865 ± 0.010 / final 0.739 ± 0.087. V.D: final "0.634 ± 0.339 with server-averaged statistics". | Three run sets are used for the same setting. 0.110 and 0.634 ± 0.339 come from the 5-repetition `fedavg_bn_a0.5_fedrecal` runs. 0.151 comes from the 3-repetition `fedavg_bn_a0.5_recal` runs. Table S4 uses the 3-repetition `fedavg_bn_a0.5` runs (swing 0.145; final 0.739 vs 0.634). The paper therefore reports two different swings and two different final scores for "FedAvg, α = 0.5, server-averaged BN". III.D also says the server-averaged BN study uses three repetitions, yet V.D quotes five-repetition runs. | Minor–moderate (consistency) |
| C12 | V.E per-class vs DP cost | "Noise costs 8 points … 10 …" vs "five of the nine attack classes lose at least ten points" | The point costs (7.7, 10.2) use the non-private same-schedule run as reference. The per-class count (and Fig. 3) uses the DP code without noise. Against the non-private reference only **4** classes lose ≥ 10 points at both noise levels; MITM goes 0.16→0.13 at σ = 0.6. Against the DP-code reference, the costs would be 10.1 and 12.6 points. Use one reference, or say which. The five classes are backdoor, MITM, password, ransomware and scanning; naming MITM as the fifth would help. | Minor |
| C13 | S3, SCAFFOLD | Non-finite weights are attributed to the earlier learning rate 0.3 only | `UNSW_NB15/scaffold_a0.1_lr0.1/seed1.json` also has `any_nonfinite = True`; this is the repetition that collapses to 0.088. | Minor |
| C14 | S3, first sentence | "the centralised model reaches 0.532 ± 0.005, with benign false-positive rates above 10%" | If this means every model, it is false: the α = 0.1 FedAvg, FedProx and FedAdam runs have 5.5–6.3% (Table S2). If it means the centralised model only, it should be singular. | Minor (wording) |
| C15 | V.A | "With group normalisation, selected and final rounds are close." | This holds for most Dirichlet rows. But SCAFFOLD α = 0.1 differs by 0.060 and FedProx α = 0.1 by 0.052. On the host partition FedAdam differs by 0.095 and SCAFFOLD by 0.101 (Table II). | Minor |
| C16 | III.C / Table I / V.D | Centralised "for 100 epochs"; "Selected round 41"; "a centralised model between epochs" | Central runs log 50 evaluations of 2 epochs each (`rounds = 50, local_epochs = 2`). The "selected round" 41 is evaluation 41, i.e. epoch 82, and the swing is measured between 2-epoch evaluations. Say so in the table note. | Minor |
| C17 | V.C | Host partition "costs as much as α = 0.1" | The point estimates agree (-0.179 vs -0.185). The direct paired difference is +0.006 [-0.195, +0.206], which is uninformative. Acceptable as a point statement only. | Info |
| C18 | VI | Fixed host order gives an interval "several times narrower" | The ratio of CI half-widths is 3.16× (0.099 vs 0.031). Acceptable. Note that the fixed partition also uses a different assignment rule (large hosts first). | Info |

## 4. Cross-references and numbering

- Main tables appear in the order I (TON main), II (TON hosts), III (N-BaIoT), IV (TON DP), and every reference points to the right one.
- Supplementary tables are S1 (paired), S2 (UNSW main), S3 (N-BaIoT per device), S4/S5 (TON/UNSW scaling and BN), S6 (UNSW DP). All references in the main text and supplement resolve correctly.
- Figs. 1–3 are referenced and exist (fig1_paired_differences, fig2_normalisation, fig3_dp_per_class). `figures/fig1_heterogeneity.png` is a leftover file that build.py does not use.
- Supplement section references from the main text (S1, S3, S4, S5, S6, S7) all point to the right content.

## 5. Verified and correct (selection)

Everything below matches the raw files:
- Headline values: abstract +0.190 / +0.174; 17 of 20 and -0.121 ± 0.083 (p = 0.015); 6 of 9 and -0.032 ± 0.046 (p = 0.147); +0.286; ε 2.93 and the 17-point gap (0.683 ± 0.057 vs 0.856 ± 0.046).
- "FedAdam the only algorithm difference significant after correction" (true across all three families).
- Fine-tuning values (-0.011 ± 0.049, -0.010 ± 0.017, 2- and 5-epoch values, 0.551, 0.779, 0.537, 0.773).
- Own-class restriction 0.138 ± 0.072; skew statistics (56%/77%/52%/38%/54%/84%, ranges, class counts, client sizes); dataset sizes; majority-class F1 0.037/0.066.
- Model size and communication (23,754 parameters, 95.0 kB, 0.76/38.0/12.8 MB, SCAFFOLD 76.0 MB, N = 2/8/16 MB per round).
- All 100-round, no-ports and BN comparisons; swings 0.196/0.009/0.151/0.017/0.012/0.016/0.018/0.010/0.007.
- DP tuning values (0.649/0.667/0.648/0.604/0.652/0.659 at ε 4.4/0.772); MITM 0.24→0.13 and 0 from σ = 1.5; the UNSW per-class DP claims in S7; the "4 to 7 points" range (4.5–7.2).
- Tuning selections and the centralised learning-rate check; FedLC grid widened twice (tuned_v3_round4 → pre_lcext → v3); SCAFFOLD lr 0.3 in the single-repetition round; "the same repetition is weakest at every learning rate" (rep 1).
- Rerun noise up to 0.084.
