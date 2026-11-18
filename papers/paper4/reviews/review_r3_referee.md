# Referee Report, Round 3 (second revision)

**Manuscript:** "Federated Intrusion Detection for IoT Networks Revisited: Data Heterogeneity, Tuned Baselines and Record-Level Differential Privacy"
**Venue:** *IEEE Internet of Things Journal*
**Reviewer expertise:** federated learning, differential privacy, ML-based intrusion detection
**Date:** 2026-10-25

**Material examined:**
- `Manuscript_filled.md` and `Manuscript_v2.pdf` (all 21 pages: captions, tables, figures, layout);
- the four figures in `figures/`;
- my round-1 and round-2 reports;
- `run_fl_v2.py` (current version) and `compute_values.py`;
- every non-superseded JSON in `results_v2/TON_IoT` and `results_v2/UNSW_NB15`, including `tuned_v3.json` and all `tune3_*` runs;
- `_superseded_r2/`, which I used only to check the history of the FedProx/FedAdam runs.

I trained nothing. I recomputed means, t-intervals, paired tests and Wilcoxon checks with my own script, outside the project.

**Recommendation:** Major revision. It is closer to acceptance than in round 2, and most of what remains is interpretation and disclosure rather than new compute.
**Score:** 6 / 10 (round 1: 3/10; round 2: 5/10)

---

## Part 1. Status of the round-2 points

### Major concerns

| # | Round-2 point | Status | One-line justification |
|---|---|---|---|
| MC1 | FedProx μ and FedAdam η_s tuned at client lr 0.001, reported at 0.003 | **Resolved** | The `tune3_fedprox_*` and `tune3_fedadam_*` cfgs record lr = 0.003, the tuned FedAvg value, and the core runs use the same value. Note: FedAdam's selected η_s is still 0.03, so its core runs were simply re-run with the same settings (see N5). |
| MC2 | Tuning too thin: grid edges, one repetition, α = 0.5 only, overlap with the evaluation partitions | **Partly resolved** | Fixed: the grids were widened, FedAvg's lr is now interior on both datasets, and the tuning seeds (100, 101) are disjoint from the evaluation seeds. Still open: (a) two tuning repetitions, not ≥3; (b) FedProx μ (lowest value) and UNSW SCAFFOLD lr (0.03, lowest) are still at grid edges; both are disclosed, and the μ edge does no harm; (c) FedLC τ is at the upper edge on UNSW (τ = 1.0), and neither the FedLC grid nor the selected values appear anywhere in §III-E; (d) tuning is still done only at α = 0.5, which is stated as a limitation. |
| MC3 | Multiple comparisons and small n | **Partly resolved** | Holm adjustment is now applied over 29 (TON) and 22 (UNSW) comparisons. But: no primary comparisons were declared, so the adjustment runs over one large family that mixes headline tests with diagnostics such as SCAFFOLD lr variants. No Wilcoxon or permutation check is given; with n = 5 the exact two-sided Wilcoxon cannot go below p = 0.0625, which shows how little power there is. And the abstract and conclusion still lead with numbers that are not significant after Holm (see N3). |
| MC4 | DP scope, tuning, schedule, baseline | **Mostly resolved** | (a), (b): preprocessing, n_k and hyperparameters are now explicitly assumed public. (c): a small C/lr grid was added (2 repetitions); its best C = 2 is at the grid edge. (d): a 40-round run and a full-budget σ = 1.8 run were added. (e): the abstract now names its baseline. (h): Bagdasaryan et al. is cited. Not done: (f) α = 0.1 and the q > 1 failure for n_k < 256 are not mentioned; (g) empty Poisson batches are not documented, and the TON DP JSONs still record `norm: 'bn'` although the model is GN. The full-budget result (an 18-point cost) is not carried into the abstract (see N4). |
| MC5 | Server-held validation set | **Resolved** | §III-D now describes federated confusion-matrix aggregation, which gives exactly the pooled macro-F1. It also states that local-only models are selected on the pooled validation set. |
| MC6 | "Each site gains"; local-only results averaged away | **Partly resolved** | Win counts (1/20, 0/20) and an "own classes" gain are now reported, and the "each gains" wording under Dirichlet skew is withdrawn. But the "own classes" metric is still computed on the common test set, so it is not the per-client-distribution evaluation I asked for, and it is biased toward FedAvg (N2). Min, max and size-weighted local results are computed in `compute_values.py` but are not shown in the paper. |
| MC7 | SCAFFOLD narrative; earlier-draft references; uniform control-variate averaging | **Mostly resolved** | Done: "advantage" is gone; non-finite runs and collapses are reported; lr 0.01 is tested; the uniform control-variate mean is disclosed; the Introduction no longer describes the earlier draft. Remaining: §VI-A still narrates a "preliminary tuning", and its claim that FedAdam's deficit did not survive matched tuning is factually wrong (N5). |
| MC8 | BN mechanism unsupported; not federation-specific; FedBN; Fig. 2 single repetition | **Partly resolved** | Done: recalibration evidence was added (swing 0.193 → 0.009 centralised TON, 0.151 → 0.017 FedAvg), the effect is called non-federation-specific, and the result is framed as confirming Hsieh et al. Not done: FedBN is not tried, and Fig. 2 is still repetition 0 only, with a legend that overlaps the data. The conclusion "BN should be avoided" is contradicted by the authors' own recalibration runs (N6). |
| MC9 | 50-round budget | **Partly resolved** | 100-round runs (n = 3) were added. The TON α = 0.5 gap shrinks from −0.059 to −0.042, about 30%, and the selected round is still late (100, 67, 68). "Change little" and "not an artefact" are too strong. Fair wording would be: "the gap persists at 100 rounds but partly closes; convergence is not established". |
| MC10 | Novelty and IoT-J fit | **Partly resolved** | Added: a device-level partition, a port ablation and FedLC. These are real improvements. But the "natural" partition is effectively one fixed partition and is described incorrectly (N1). No device-structured IoT dataset was added (CICIoT2023, Edge-IIoTset, N-BaIoT). Personalised FL and FedBN remain untested. |

### Minor concerns

| # | Round-2 point | Status | One-line justification |
|---|---|---|---|
| m1 | Benign FPR claim | **Resolved** | Now limited to TON (<4%), with the SGD methods at 4–6%. |
| m2 | "p = <0.001" | **Resolved** | Now "p < 0.001" throughout. |
| m3 | Table/figure naming | **Resolved** | The PDF uses TABLE I–VII and Fig. 1–3, as IEEE requires. |
| m4 | Table III caption; UNSW scaling not in a table | **Resolved** | Table V added; the N = 4 row is labelled as five repetitions. |
| m5 | Fig. 3 rows, σ = 1.5 missing, label overlap | **Resolved** | All four σ values and both sanity rows are shown; no overlap. |
| m6 | Fig. 1 coverage | **Resolved** | Replaced by a paired-difference forest plot covering every algorithm and the natural partition. |
| m7 | SD alongside CI | **Resolved** | SD column in Tables I–III. Tables IV–VII still show CI only. |
| m8 | UNSW DP wording | **Resolved** | Shellcode is now quoted at 0.39; analysis and worms are flagged as ≈0 everywhere. |
| m9 | Majority / stratified-random reference | **Resolved** | Given as 0.037 / 0.066 and ≈0.10. The 0.10 is exact: a frequency-matched random guesser's macro-F1 is 1/K. |
| m10 | Selection-rule claim | **Resolved** | Now qualified (SCAFFOLD, BN). |
| m11 | Parameter size | **Resolved** | "23,754 parameters (95.0 kB …)". |
| m12 | Determinism | **Resolved** | XLA nondeterminism is stated. |
| m13 | Repository DOI / archive | **Not resolved** | A GitHub URL only. No archived DOI and no commit hash. |
| m14 | Related-work positioning | **Mostly resolved** | FedDyn, MOON, FedBN, FedLC, Ditto, Bagdasaryan, Ponomareva and the poisoning work are cited. "Few of these works report…" is still unsupported, though it is softened. |
| m15 | Generative-AI declaration | **Resolved** | Placed in the Acknowledgment, consistent with IEEE practice. |
| m16 | BN row in the DP table | **Resolved** | The reference row is now the GN short-schedule run (`fedavg_gn_short`, 0.735; verified). |

---

## Part 2. New problems in this revision

### N1. The "natural device-level partition" is not redrawn, and the Methods text describes a different algorithm from the code (major)

- The paper says hosts are assigned "taking hosts in random order and assigning each to the currently smallest client". Table III's caption says "five repetitions with redrawn assignments". The abstract says "Dirichlet and device-level client partitions redrawn in each of five repetitions".
- The code does something else. `natural_partition` sorts hosts by **size, largest first** (`order[np.argsort(-sizes[order], kind='stable')]`) and uses the random order only to break ties between hosts of equal size.
- The result is almost deterministic:
  - client sizes are [33,550, 31,645, 31,645, 31,644] in all five repetitions;
  - client 0's class-count vector is identical in all five;
  - the other clients differ by a few dozen records.
- So the five "repetitions" of Table III vary only initialisation and batch order. The partition is fixed. That is exactly the single-partition design the paper criticises in its Introduction.

**Consequences:**
- the CIs in Table III understate the uncertainty (local-only 0.504 ± 0.009);
- "Holm-adjusted p < 0.001" and "0 of 20 site–repetition pairs" are really four sites in one partition;
- "Dirichlet label partitions understate the cost of federating device traffic" rests on n = 1 partition.

**Required:**
- correct the description in the abstract, Contribution 1, §III-B and the Table III caption;
- preferably, generate genuinely different device-level partitions (random host order without size sorting, with a size-balance tolerance, or random 4-way groupings of hosts) and re-run Table III;
- report how many distinct destination hosts exist, and how many carry most of the traffic.

### N2. "Every site gains, also on its own traffic" is not what was measured, and the metric favours FedAvg (major)

- What was computed is per-class F1 on the **common** test set, averaged over the classes a client holds at ≥1%. That is not the client's own traffic.
- Under the device partition, the client's own traffic would be the test records whose destination host belongs to that client. These can be identified, because `dst_ip` is available for the test split.
- Per-class F1 on the full test set also counts, against a held class's precision, every record of an unseen class that the local model maps into that class. The local model cannot do anything else, so the "own classes" gain still partly measures the unseen-class effect it is meant to remove.
- The abstract's "every site gains, also on its own traffic" and the conclusion's "under a natural device-level partition every site gains also on its own traffic" are therefore overclaims.

**Required:**
- evaluate each site's local and federated models on that site's own test traffic (its hosts' test records), or on a test subset resampled to the site's label mix with unseen classes excluded;
- otherwise, rephrase as "on the client's classes, in the common test set".

### N3. The abstract leads with effects the paper's own rule declares unsupported (major)

- §V-A says "we base conclusions on effects that are significant after Holm correction". Yet the abstract and conclusion quote numbers that are not:

  | Quoted effect | Holm-adjusted p |
  |---|---|
  | −0.059 vs centralised (TON, α = 0.5) | 0.348 |
  | +0.080 over local (TON, α = 0.5) | 0.083 |
  | −0.185 vs centralised (TON, α = 0.1) | 0.147 |
  | "cost … grows from −0.042 … to −0.185" | 0.059 and 0.147 |

- The monotone-trend claim ("the more heterogeneous, the more it costs") is never tested as a trend. A single pre-declared trend test, or a mixed model with α as a covariate, would be more powerful than 29 pairwise tests.
- **Required:** declare 4–6 primary comparisons and correct over those, with the rest labelled exploratory. Alternatively, keep the large family and rewrite the abstract and conclusion so that each quoted effect carries its adjusted status. Either is acceptable; the current mixture is not.

### N4. The DP headline uses the most favourable schedule (moderate)

- The abstract says DP "costs about 10 macro-F1 points … against non-private training on the same schedule". The authors' own full-budget run (50 × 2, σ = 1.8, ε ≈ 2.9) shows **0.670 vs 0.847, an 18-point cost**. The 40-round run shows 11 points.
- So the cost grows with training budget. The 10-point figure applies to an under-trained regime that a practitioner would not deploy.
- The full-budget evidence is thin: n = 2, and the non-private reference is 0.847 ± 0.140 (selected rounds 23 and 44).
- **Required:**
  - state the full-budget cost in the abstract (e.g., "about 10 points on a short schedule and about 18 at the full training budget, ε ≈ 3");
  - add a third repetition to the full-budget pair;
  - note that the DP tuning grid's best C (= 2) is at its edge, so "not an artefact of the DP configuration" is only shown for C ≤ 2 and lr ∈ [0.001, 0.006].

### N5. A factually wrong statement about FedAdam in §VI-A (moderate)

- "in a preliminary tuning in which FedProx and FedAdam inherited a different client learning rate, both appeared worse than FedAvg … neither difference survived matched tuning."
- For FedAdam this is false. Matched tuning selected **the same** η_s = 0.03, and the core FedAdam runs have identical configuration before and after:
  - round 2 (`_superseded_r2`): TON α = 0.5, 0.751;
  - now: 0.757, −0.083 [−0.118, −0.049] vs FedAvg, p = 0.003 (Holm 0.060).
- FedAdam is still about 8 points worse on TON at α = 0.5. Only FedProx changed: its μ went from 0.01 to 10⁻⁴, which is effectively FedAvg.
- The paragraph also keeps the response-letter narrative I asked to remove.
- **Required:** delete or correct the sentence. Report FedAdam as worse than FedAvg on TON (uncorrected), rather than "does not improve".

### N6. The BN conclusion contradicts the authors' recalibration data (moderate)

- The recalibration runs (`*_bn_*_recal`) store `test_f1_recal`. Selecting the round on `val_f1_recal` gives:

  | Model (TON, α = 0.5, reps 0–2) | Test macro-F1 |
  |---|---|
  | FedAvg-BN + recalibration | 0.867, 0.871, 0.883 (mean 0.874) |
  | GN FedAvg, same repetitions | 0.792, 0.822, 0.863 (mean 0.826) |
  | Centralised BN + recalibration | 0.892 |
  | Centralised GN | 0.893 |

- The final-round recalibrated FedAvg-BN is also 0.863–0.883.
- So with recalibrated statistics, BN is at least as good as GN, and in the federated case about 5 points better, though only on n = 3 and not formally tested.
- The paper calls recalibration "not a deployable remedy" because it "needs pooled data". That is too strong. BN statistics can be re-estimated federatedly: clients return per-layer sums and sums of squares on local data for the current global model, layer by layer if exactness is required, and the server aggregates them by size. This costs a handful of extra messages and is standard practice.
- The conclusion "Batch normalisation should be avoided in federated IDS models" is therefore not supported.
- **Required:**
  - report the recalibrated test scores;
  - either test federated recalibration or FedBN, or soften the conclusion to "BN with server-averaged running statistics should be avoided; GN or re-estimated BN statistics avoid the problem";
  - add repetition bands to Fig. 2 and move its legend off the data.

### N7. The port shortcut undercuts the TON headline numbers (moderate)

- The ablation is welcome and candid. It shows that the two port features carry 14 points of the centralised TON score and 19–22 points of FedAvg's, and that the federation gap roughly doubles without them (−0.110 vs −0.059 at α = 0.5).
- Yet every TON headline in the abstract is the with-ports number, and the algorithm comparison is not repeated without ports.
- Since the paper's subject is the cost of federation, and that cost changes materially without the shortcut, the abstract should give the no-port gap, and at least FedAvg vs FedProx/FedLC should be checked without ports.

### N8. FedLC is under-specified and possibly under-tuned (minor/moderate)

- FedLC's τ grid ({0.1, 0.3, 1}), the selected values (0.3 TON, 1.0 UNSW, the latter at the edge) and the calibration formula (δ_c = τ·n_{k,c}^{-1/4}, with absent classes set to count 1) are not in the paper. TON tuning scores are flat across τ (0.8607 / 0.8613 / 0.8604).
- With the client counts here (10³–10⁴ per class), the per-class offsets differ by less than about 1 logit even at τ = 1. A wider grid (τ up to 5–10) is needed before concluding that "the label-skew method FedLC behaves the same".
- Add the grid to §III-E.

### N9. Other wording and detail

- **"Federation costs more than under Dirichlet skew"** (§V-C, abstract). It costs more than at α = 0.5 (−0.129 vs −0.059) but less than at α = 0.1 (−0.185). Say so.
- **The SCAFFOLD α = 0.1 UNSW explanation** ("this partition is hard for SCAFFOLD rather than the learning rate being wrong"). Plausible, since seed 1 is weakest at lr 0.01, 0.03 and 0.1 (0.157 / 0.153 / 0.088) while FedAvg-SGD reaches 0.365 on that seed. But it is one partition; phrase it as an observation.
- **Own-class gain intervals.** The "± …" on the own-class gains (0.049 ± 0.021 etc.) treats 20 client–repetition pairs as independent. The four clients in a repetition share a partition, so use the repetition as the unit.
- **The 100-round comparison** uses `central_r100` (200 epochs) on n = 3, and its CIs at α = 0.1 span zero. Present it as a robustness check, not as proof.

### N10. Presentation and IoT-J fit

- **Format.** The PDF is single-column, about 1.5-spaced, 21 pages, with no author affiliations and no biographies. IoT-J requires the IEEE two-column template with affiliations. In that template the paper would run to roughly 11–13 pages, which is acceptable, but check the current IoT-J length policy and overlength charges.
- **Length is reasonable.** About 8k words including tables and references.
- **Density.** §V-B is one 450-word paragraph packing about 18 CIs with raw and adjusted p-values into running text. Move the paired comparisons into a table (difference, CI, p, Holm p), and keep the prose to the conclusions. The same applies to §V-A and §V-C.
- **IoT fit.** This remains the weakest aspect for this journal:
  - one of the two datasets is explicitly non-IoT;
  - the IoT dataset's signal depends heavily on testbed ports;
  - the "device-level" partition is a single deterministic split by destination IP of a testbed.

  A device-structured IoT dataset (CICIoT2023, Edge-IIoTset or N-BaIoT, partitioned by device) would make the venue case much stronger. Without one, the contribution is a carefully executed methodology note on two benchmarks.

---

## Part 3. Spot-check of headline numbers against the JSONs

All values were recomputed from the per-seed JSONs with my own script (mean, t-based 95% CI half-width, paired t-test).

| # | Claim in manuscript | Recomputed | Match |
|---|---|---|---|
| 1 | TON centralised 0.899 ± 0.012; FedAvg α = 0.5 0.840 ± 0.043 | 0.899 ± 0.012; 0.840 ± 0.043 | Yes |
| 2 | TON FedAvg − central: −0.042 [−0.059, −0.025] IID; −0.059 [−0.101, −0.016], p = 0.018; −0.185 [−0.286, −0.084] | identical (p = 0.0025, 0.0183, 0.0070) | Yes |
| 3 | TON FedAvg − local: +0.080 [+0.043, +0.117]; +0.286 [+0.194, +0.379] | identical | Yes |
| 4 | TON FedAdam − FedAvg α = 0.5: −0.083 [−0.118, −0.049], p = 0.003 | identical | Yes |
| 5 | TON SCAFFOLD − FedAvg: −0.125 [−0.165, −0.085]; SCAFFOLD − FedAvg-SGD: +0.020 [−0.001, +0.042], p = 0.058 | identical | Yes |
| 6 | Natural: FedAvg 0.770 ± 0.031; −0.129 [−0.165, −0.092]; +0.266 [+0.236, +0.297]; local 0.504 ± 0.009 | identical | Yes (but see N1: one effective partition) |
| 7 | UNSW FedAvg − central −0.065 / −0.099 / −0.171; gain over local +0.032 (p = 0.271) / +0.136 | identical | Yes |
| 8 | 100 rounds TON: −0.036 / −0.042 / −0.227; UNSW −0.045 / −0.119 / −0.197 | identical; selected rounds TON α = 0.5: 100, 67, 68 (mean 78) | Yes |
| 9 | BN swing 0.193 → 0.009 (central TON), 0.151 → 0.017 (FedAvg); UNSW 0.024 → 0.009 | 0.193 / 0.009; 0.151 / 0.017; 0.024 / 0.009 | Yes (recalibrated test scores not reported; N6) |
| 10 | DP TON: 0.658 ± 0.053 at ε 12.15 (11.72–12.50); 0.633 ± 0.022 at ε 0.81 (0.78–0.84); GN short 0.735 ± 0.051 | identical (per-rep ε 11.72/12.50/12.23; 0.78/0.84/0.82) | Yes |
| 11 | DP full budget 0.670 vs 0.847 at ε 2.9; 40 rounds 0.659 vs 0.772; C = 2 0.667, C = 0.5 0.649; lr 0.001/0.006 0.604/0.652 | identical; full-budget ε 2.81/3.03; n = 2, non-private CI ± 0.140 | Yes |
| 12 | Per-class DP (Fig. 3): MITM 0.24 → 0.13 → 0 from σ = 1.5; password 0.54 → 0.15; five attack classes lose ≥10 points at both extremes | MITM 0.24/0.13/0.00; password 0.54/0.15; backdoor, ransomware, scanning, password and MITM lose ≥10; injection, XSS and DDoS do not | Yes |
| 13 | Tuning: FedAvg lr 0.003 both; μ 10⁻⁴ both; η_s 0.03 both; SCAFFOLD 0.1 / 0.03; FedAvg-SGD 0.3 / 0.03 | matches `tuned_v3.json`; tune3b cfgs use lr 0.003 | Yes (FedLC τ omitted from the text; N8) |

**Arithmetic is not the problem.** Every number I checked matches to three decimals. The problems are in the descriptions (N1, N2, N5), the selection of what goes in the abstract (N3, N4, N7), and one omitted result (N6).

---

## Part 4. Overall assessment

### What improved

This is again a serious, good-faith revision:
- the FedProx/FedAdam tuning confound is gone;
- tuning partitions are disjoint from the evaluation partitions;
- multiplicity is addressed;
- the BN mechanism now has evidence behind it;
- a natural partition, a port ablation, a label-skew baseline, a longer schedule and a DP tuning/full-budget check have all been added.

The code is clean and the released JSONs reproduce every reported statistic.

### What still holds the paper back

Several of the new additions are oversold in the text:
- the "natural, redrawn" partition is a single deterministic split;
- "own traffic" is the common test set;
- the abstract quotes effects the paper itself treats as unsupported after Holm, and the DP cost from the most favourable schedule;
- the BN recommendation ignores the authors' own recalibration result;
- §VI-A misstates the FedAdam history.

None of these needs heavy compute. The partition fix and the per-site evaluation need a few dozen small CNN runs.

### Venue fit

The IoT-J fit is still marginal without a device-structured IoT dataset.

### Recommendation

**Major revision. Score: 6/10.**

### What would most raise the score

In order of impact:
1. **Fix the natural partition.** Make it genuinely redrawn, describe it correctly, and evaluate each site on its own hosts' test traffic. This turns the paper's most IoT-relevant result from n = 1 into evidence (N1, N2).
2. **Add one device-structured IoT dataset** (Edge-IIoTset, CICIoT2023 or N-BaIoT) with a per-device partition. This is the single biggest step for IoT-J fit (MC10).
3. **Rewrite the abstract and conclusion against a declared primary family.** Include the no-port federation gap and the full-budget DP cost (about 18 points). Correct the FedAdam sentence and the "costs more than Dirichlet" phrasing (N3, N4, N5, N7, N9).
4. **Report BN with recalibrated statistics,** and either test federated recalibration or FedBN, or soften "avoid BN" (N6).
5. **Specify and widen the FedLC τ grid; add a third tuning repetition.** Move the paired comparisons into a table and convert the paper to the IEEE two-column template with affiliations (N8, N10, MC2).

With items 1, 3 and 4, I would expect to recommend minor revision. Adding item 2 would make this a solid IoT-J benchmark paper.
