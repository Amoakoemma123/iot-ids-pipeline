# Referee Report: Final Round (fresh read)

**Manuscript:** "Federated Intrusion Detection for IoT Networks Revisited: Device-Level Partitions, Tuned Baselines and Record-Level Differential Privacy"
**Journal:** IEEE Internet of Things Journal
**Material reviewed:** `Manuscript_filled.md` (main text, Tables I–IV), `Supplement_filled.md` (S1–S7, Tables S1–S7), `figures/fig1_paired_differences.png`, `fig2_normalisation.png`, `fig3_dp_per_class.png`. No experiments were run. I spot-checked numbers by hand against the tables (Section 6 below).
**Date:** 2026-11-07

**Score:** 6.5 / 10
**Recommendation:** Major revision, limited in scope. The protocol is sound and honestly reported. But the paper's most quotable new finding, the own-traffic reversal, rests on a random record split that the authors themselves say favours the local model. Two other headline results (the DP cost and "FedAdam is worse") may come from the optimiser or tuning design rather than from federation or privacy. Each needs one cheap diagnostic, not a new study. If those diagnostics support the current wording, I would expect to move to accept.

---

## 1. Summary

The paper re-evaluates federated learning for a compact 1D-CNN IDS (23,754 parameters, group normalisation) on TON-IoT, UNSW-NB15 and N-BaIoT. The protocol has these parts:
- validation-based round selection;
- partitions redrawn in each of five repetitions;
- tuning of every algorithm at a matched client learning rate;
- same-recipe centralised and local-only references;
- a declared family of primary comparisons with Holm correction (19/13/9 comparisons);
- record-level DP-SGD with an RDP accountant.

Besides Dirichlet label skew, it uses two device-level partitions: TON-IoT by destination host (4 sites) and N-BaIoT by physical device (9 sites).

Main findings:
1. FedAvg beats local-only training on the common test set (+0.190† on TON-IoT hosts, +0.187† on N-BaIoT devices). On each site's own test traffic the local model is better (17/20 and 43/45 site–repetition pairs; exploratory). This holds without port features.
2. Local fine-tuning recovers own-traffic accuracy at a cost in breadth that depends on the dataset (89% of the federated gain lost on TON-IoT, 28% on N-BaIoT). Running both models raises the benign FPR on unfamiliar traffic to 16.4%.
3. None of FedProx, FedAdam, SCAFFOLD or FedLC improves significantly on tuned FedAvg. FedAdam is significantly worse on TON-IoT at α = 0.5.
4. BN with server-averaged statistics makes the model swing between rounds; federated re-estimation removes the swing.
5. DP-SGD at ε ≈ 2.9 (full budget) costs 17 macro-F1 points on TON-IoT, and 7 of 9 attack classes lose at least ten points.

The paper is unusually candid about what is exploratory, and its arithmetic is clean. Its weaknesses are now mostly about interpretation. In three places a cheap diagnostic separates "a property of federated IDS" from "a property of this experimental design".

---

## 2. Novelty and fit

- **Novelty is moderate, and it is a methodology contribution.** Several individual findings are known from the general FL and DP literature:
  - no FL algorithm dominates tuned FedAvg [8];
  - BN with averaged statistics breaks under non-IID data [20], [30];
  - DP hurts some classes disproportionately [38];
  - a client's own model fits its own distribution best, which is the premise of personalised FL [31]–[34].
- What is new is the combination: a protocol that would pass a methodological audit, plus the quantification of these effects for IDS under device-level partitions. The own-traffic/common-test trade-off with measured operational metrics (detection rate, FPR, two-model union) is the most original part.
- **The motivation is asserted, not shown.** The Introduction says FL-IDS studies "frequently" pick the best test round, use one partition, leave baselines untuned and report unsatisfied DP budgets. Not one example is cited for these practices (§I, para. 2). For a paper whose main contribution is "fix the protocol", this is the weakest link (Major 5).
- **IoT-J fit is acceptable, not strong.** The IoT angle is the device-level partitions and the IoT datasets. There is no device-side cost analysis: compute, memory or energy of local training on constrained hardware, or straggler and partial participation. S1-E's honest communication analysis shows that FL does not save bandwidth here. One paragraph on where the 23.7k-parameter model would train (gateway versus device) would help the fit.

---

## 3. Major issues

### M1. The own-traffic reversal is what a random record split predicts; its headline status needs a split that can falsify it

**Location:** Abstract (sentences 4–5), Contribution 2, §V-C "Common test set versus own traffic", §VI "Deployment", §VII.

**Why it matters:** This is the paper's main new result, and the paper's own numbers point to memorisation of in-split neighbours more than to "a site's own traffic is different":
- On TON-IoT, restricted to a site's own classes on the common test set (other sites' traffic of the same classes), FedAvg is better by 0.138 ± 0.072 (§V-C). On the site's own test traffic, the local model is better by 0.121. So the local model is good at class *c* on its own hosts' held-out records but poor at class *c* elsewhere. This is the signature of host-specific fitting, which a random record split rewards. TON-IoT attack flows to a host come in bursts and sessions, so many test flows are near-duplicates of training flows.
- On N-BaIoT, S1-A states that "the stream statistics of neighbouring packets overlap, [so] a random split still makes absolute N-BaIoT scores optimistic". The Kitsune features are damped-window statistics, so consecutive records are strongly autocorrelated. A device-local model trained on a random 70% of a device's capture is effectively tested on interpolations of its training trajectory. The local model reaches 0.97–0.99 on own traffic for every device (Table S3).
- Local own-traffic attack detection is 99.7% (TON-IoT) and 99.9% (N-BaIoT), which again points to in-distribution memorisation.

The Limitations paragraph concedes this ("a temporal split would test the own-traffic results for future traffic"). But the abstract and Conclusion still present the reversal as a property of federation ("a site's own model is usually better on its own traffic"). The operational claim, that a site should keep or adapt a local model, is about *future* traffic. The current design cannot support that claim.

**Fix (in order of preference):**
1. Run the own-traffic comparison with a per-site temporal or blocked split:
   - TON-IoT: by the flow timestamp, if it was retained before the identifier drop, within each host and class;
   - N-BaIoT: by record order within each device–class file, which is capture order (e.g., first 70% train, a gap, last 15% test).
   If classes then vanish from some sites' test traffic, a blocked split (contiguous blocks per host and class) is enough. Only FedAvg and the local model are needed, five repetitions, no tuning. This is the single analysis that would most raise the paper's value.
2. If that is infeasible, demote the reversal in the abstract and Conclusion to a conditional statement ("under random record splits, which favour a site's own model, …"). Present the restricted-class contrast (0.138 for FedAvg against 0.121 for local) as evidence that the advantage is site-specific fitting.

### M2. An untested simpler explanation for the reversal: the site's label prior

**Location:** §V-C, §V-C "Personalisation", Table II, Table S4.

**Why it matters:**
- On TON-IoT the federated model predicts over all 10 classes. Many sites hold only 7–9 of them (§III-B: "7 to 10 classes per site"), and their class proportions differ strongly from the pooled mix. Own-traffic macro-F1 is computed over the classes present at the site. Any FedAvg prediction of a class the site never sees is a pure loss, and the local model avoids this almost by construction.
- Part of the "reversal", and part of what one epoch of fine-tuning recovers, may simply be the label prior and label space, not site-specific feature patterns. On N-BaIoT, most devices hold all 9 classes, but the class proportions still differ by device.

**Fix:** Score the selected federated model with a zero-training-cost prior correction at each site: add log(π_site,c) − log(π_pooled,c) to the logits, and mask classes absent from the site's training data. Report it as an extra row in Table II and Table S4.
- If the correction closes much of the gap, the honest message is "federate, then apply the local label prior". That is cheaper and keeps breadth, unlike fine-tuning.
- If it does not, the claim that the local model captures site-specific feature patterns becomes much stronger.

FedLC [18] already uses n_{k,c} in training, so the machinery exists.

### M3. The DP cost may be partly an optimiser artefact; batch size, the main DP hyperparameter, was never varied

**Location:** §IV para. 2, §V-E, Table IV, Fig. 3, S7, Abstract (last sentence), §VII.

**Why it matters:**
- Table IV shows that the first, nearly negligible noise causes most of the damage. Without noise or clipping the DP code reaches 0.760. At σ = 0.2 (ε ≈ 402, no meaningful privacy) it drops to 0.681, and from there to σ = 2.5 (ε = 0.81) it falls only to 0.633. §V-E calls this curve "nearly flat across a fifteen-fold range of ε" but does not explain it.
- A rough calculation: at B = 256 and C = 1, σ = 0.2 adds noise with a per-coordinate std of about 8·10⁻⁴ to the averaged gradient. Over 23,754 parameters, that is a noise norm of about 0.12, comparable to a clipped mean gradient. Adam (the step in §IV) divides each coordinate by its running RMS, so for weakly informative coordinates it turns this noise into full-size updates. Bias in Adam's second moment under DP noise is a known problem. In short, the 8-point "cost" at ε = 402 is more likely an interaction between Adam and DP noise than a cost of privacy.
- S7 tunes C, the learning rate and the schedule, but not the batch size (expected Poisson batch), which the DP literature treats as the main lever. Larger batches reduce noise relative to signal roughly in proportion to B, at a modest cost in ε. Round 1 already raised this with reference to Ponomareva et al.; it is still not addressed.
- The per-class "disparate impact on well-represented classes" conclusion (§V-E) inherits this uncertainty. Password, for example, loses 14 points from clipping alone (0.54 → 0.40, Fig. 3).

**Fix:** At the full budget (σ = 1.8 or matched ε ≈ 3), add two short diagnostic configurations with three repetitions:
- (a) DP-SGD with SGD plus momentum, or a DP-bias-corrected Adam, instead of Adam;
- (b) a 4–8× larger expected batch with σ recalibrated to the same ε.

If the gap stays at ~17 points, the conclusion stands and is much stronger. If it shrinks, report the better configuration and reword the abstract and conclusion. Also report the paired difference and CI for the abstract's 17-point gap (0.683 ± 0.057 against 0.856 ± 0.046 is given only as two unpaired means).

### M4. "FedAdam is significantly worse" and "the extra machinery buys nothing" overreach the tuning design and the power

**Location:** Abstract ("FedAdam is significantly worse on TON-IoT"), Contribution 4, §V-B last sentence ("the extra machinery buys nothing over tuned FedAvg"), §VII.

**Why it matters:**
- **FedAdam tuning:** §III-D tunes FedAdam's server learning rate only, at a client learning rate fixed at FedAvg's optimum, with τ = 10⁻³ and β₂ = 0.99 fixed (§IV). Reddi et al. [16] tune the client and server learning rates jointly and note that τ matters. The local Adam state is also reset every round, so FedAdam stacks two adaptive optimisers. Matching the client learning rate was a reasonable fix for the earlier mismatched tuning. But it builds in an advantage for FedAvg, whose client learning rate is optimal by construction. The only algorithm difference that survives correction (−0.083†) is exactly the comparison most exposed to this.
- **"Buys nothing":** §III-D says that only differences of about 0.05 (TON-IoT) and 0.04 (UNSW-NB15) are detectable. Several uncorrected effects are of that size and favour the alternatives: FedLC on UNSW α = 0.5 (+0.063 [+0.015, +0.110]), FedProx on UNSW α = 0.1 (+0.045 [+0.011, +0.080]), and FedAdam on N-BaIoT (+0.028 [−0.012, +0.068]). With CIs this wide, "no significant improvement" is defensible; "buys nothing" is not. Note also that FedLC's UNSW gain comes with a higher benign FPR (17.4% against 11.8%, Table S2), which is worth one sentence.

**Fix:**
- Run a small 2-D grid for FedAdam on TON-IoT (client learning rate {10⁻³, 3·10⁻³} × server learning rate, ideally τ ∈ {10⁻³, 10⁻²}) in the tuning repetitions. Otherwise, qualify the claim as "FedAdam is worse at FedAvg's client learning rate".
- Replace "buys nothing" with a bounded statement ("no algorithm improves on FedAvg by more than the ~5 points our design can detect"). Better still, report an equivalence check (TOST at ±0.05) for the algorithm comparisons; with these CIs most will be inconclusive, and the paper should say so.

### M5. The paper's premise, that FL-IDS evaluations commonly fail these checks, is uncited

**Location:** §I para. 2 ("Reported scores are frequently the best test score over training rounds; results come from one random client partition; … DP budgets are reported that the implemented mechanism does not satisfy"), §II.

**Why it matters:** The contribution is a protocol. Its importance depends on how common these failures are. Round 4 (minor 20) flagged the uncited "few of these works …" sentence; it was removed from §II, but the same claim now stands, uncited, in the Introduction.

**Fix:** Add a short, systematic audit, e.g., a table in S2 coding 15–25 FL-IDS papers from the cited surveys [2], [5], [6] for the following:
- round selection on test data or not;
- number of partitions or repetitions;
- tuned baselines;
- same-recipe centralised reference;
- per-site evaluation;
- whether the DP accounting matches the implementation.

Cite the counts in §I. This is desk work, would sharpen the novelty claim, and makes the recommendations in §VI "Evaluation" evidence-based.

### M6. GPU nondeterminism is as large as the effects, and its consequences are under-reported

**Location:** §V-C "Personalisation" ("identical configurations differed by up to 0.109 in test macro-F1"), S1-C, Table II against Table S4.

**Why it matters:**
- A difference of 0.109 between nominally identical runs exceeds most effects the paper interprets. FedAvg's own-traffic attack detection is 96.5% in Table II and 91.1% in the re-run (Table S4). That is a 5.4-point swing in the operational metric the paper uses to argue that "part of this is missed attacks".
- The paired design assumes that a repetition's two arms share partition, initialisation and batch order. With nondeterministic training they share the first two only approximately in effect, so the pairing gains less than the reader assumes.
- Reproducibility from the released commit is also weaker than "a repetition reproduces its partition and initialisation" suggests.

**Fix:**
- Report the distribution of re-run differences (mean absolute difference and SD across the five re-run pairs), not only the maximum.
- State whether `TF_DETERMINISTIC_OPS` and disabling XLA was tried and what it costs.
- In §V-C, avoid quoting the detection-rate split ("part of this is missed attacks") as a finding when two runs of the same configuration disagree by more than the effect. Either average the two run sets or present the binary metrics with CIs.

---

## 4. Minor issues

| # | Location | Issue | Fix |
|---|---|---|---|
| 1 | §V-D, §VII ("Batch normalisation should not be used with server-averaged running statistics") | Table S5 shows that BN with server-averaged statistics, *at the validation-selected round*, scores 0.862 ± 0.020 (5 reps), above GN's 0.840 ± 0.043. The swing is a final-round and deployment-stability problem, not an accuracy problem once the paper's own selection protocol is used. | Say this explicitly in §V-D. Phrase the conclusion as "should not be deployed at an arbitrary round". |
| 2 | §V-D, Contribution 5 | BN with re-estimation beats GN by +0.032/+0.039 (uncorrected). All algorithm comparisons use the weaker GN, so it is unknown whether the algorithm ranking holds under BN with re-estimation. | One sentence in Limitations. |
| 3 | §V-E, Fig. 3 | "7 of 9 attack classes lose ten points or more … only DDoS and DoS … lose less". DDoS reads 0.93 → 0.83 in Fig. 3, i.e. exactly 10 points after rounding. The count depends on unrounded values. | State the DDoS loss to three decimals, or say "7–8 of 9". |
| 4 | §V-E | Injection is unaffected on the short schedule (0.57 → 0.58 at σ = 0.6) but loses 22 points at the full budget (0.80 → 0.58). The "spread" at the full budget partly reflects that the non-private full-budget model learns classes the short schedule never learned. | One clause: the full-budget cost includes classes that only the longer non-private schedule learns. |
| 5 | §III-D | The DP study uses validation-based selection. Validation records are not protected by the record-level guarantee. | Say so in §IV (the guarantee covers training records only). |
| 6 | §V-A | The "slope over the four skew levels" codes IID, α = 1.0, 0.5 and 0.1 as equally spaced, although FedAvg is non-monotone (α = 1.0: 0.872 > IID 0.857). The model is still not stated (round-4 minor 5). | State the coding and model in S2, or drop the slope and keep the three paired gaps. |
| 7 | §V-C | Sign and notation: "the local model is better … by −0.121 ± 0.083" uses a negative number for a local advantage and ± for a paired difference, whereas elsewhere paired differences use [CI]. "Worse for 9 of 9 devices" is ambiguous about who is worse. | Use "FedAvg − local = −0.121 [−0.204, −0.038]" and "FedAvg is worse on 9 of 9 devices". |
| 8 | §V-C "Personalisation", abstract ("closes this gap on TON-IoT") | 0.008 ± 0.059 has a CI of ±6 points; "closes" asserts equivalence the design cannot show. | "Closes the gap within ±6 points" or "brings it to parity within our resolution". |
| 9 | Table I, S4 | FedAvg's selected rounds cluster at 41–49 of 50. The 100-round check narrows the TON-IoT α = 0.5 gap by a third. The core federation gaps are therefore measured at a budget where FedAvg has not converged. | State this in §V-A in one clause (currently only in S4). |
| 10 | Table III | Still no "Selected round" column, unlike Table I (round-4 minor 10). | Add it, or drop it from Table I. |
| 11 | §III-B, S1-B | The device-versus-Dirichlet contrast is confounded by quantity skew on both datasets (TON hosts 23.7k–52.8k; N-BaIoT devices 3.1k–19.0k). On N-BaIoT the device partition has *milder* label skew yet costs more. This points to feature (covariate) skew by device, which the paper never names. | Name feature skew as the likely mechanism. Cite FedBN's feature-shift setting [31]. Ideally add a Dirichlet control matched on client sizes. |
| 12 | §II, §V-C | The personalisation baselines are fine-tuning variants (same learning rate, lr/10, proximal). Ditto [32] and FedBN [31] are cited but not run. FedBN is especially natural because the BN code already exists and M2/minor 11 suggest feature skew. | Run FedBN on N-BaIoT, or state in Limitations that personalised FL proper was not evaluated (the current text says "personalisation only through local adaptation"; fine, but name FedBN). |
| 13 | §III-C | Choice of a 1D convolution over an arbitrary feature order on tabular flow features is not justified, and there is no strong non-neural centralised reference (e.g., gradient-boosted trees). Without one, the absolute level of the "centralised" anchor (0.899 TON, 0.532 UNSW) cannot be judged. | Add one centralised GBDT row to Table I/S2, or justify the architecture in one sentence and acknowledge that the centralised anchor is model-specific. |
| 14 | §III-A | Feature selection (not just scaling) is fitted on pooled training data, so local models use features chosen with other sites' data. This is mentioned for scaling only. | Add "and feature selection" to the Limitations sentence. |
| 15 | §III-D | "Exact non-parametric tests cannot reach p < 0.05, so we use t-tests" reads as choosing the test that can reject. | Rephrase: "n = 5 does not allow exact non-parametric tests at α = 0.05; t-tests rely on approximate normality of the paired differences, which we cannot check". |
| 16 | Abstract | 249 words, at the limit, with 14 numbers. | Trim to ~220. For example, drop "(16 of 20 without port features)" and keep "also without port features". |
| 17 | §V-C | "(91.1% in the re-run used for personalisation below)" in the middle of the main argument is confusing (see M6). | Move it to the Personalisation paragraph. |
| 18 | §VI "Evaluation" | "Grids must be checked for edge optima and widened where they occur" is a lesson from the authors' own history. | Fine as a recommendation; fold it into the list in the same paragraph. |
| 19 | §V-C "Personalisation" | The two-model union's 16.4% common-test FPR is the most useful operational number in the paper but appears only in prose and Table S4. | Add the union rows (own-traffic detection/FPR, common-test FPR) to Table II. |
| 20 | Declarations | Commit hash given (good), but there is no archived DOI (open since round 2). | Archive a Zenodo release and cite the DOI. |
| 21 | Refs. | [35] (ACM CSUR vol. 58, no. 11, 2026, doi 10.1145/3801079) and [5] (vol. 57, no. 12, 2025) are recent. [30]'s title reads "Why batch normalization damage…" (grammar as published?). | Verify all three against the publisher pages, per the only-existing-references rule. |
| 22 | Fig. 1 | Legible and † marks are now on the plot (good). Rows with two datasets leave uneven vertical gaps, and the x-axis has no shading for the ±0.05 detectable-difference band. | Optional: shade ±0.05 to visualise the power statement. |
| 23 | Fig. 2 | The centralised BN grey band still dominates the TON-IoT panel. | Lighten it or move it to its own panel; put the swing numbers (0.110 against 0.016/0.018) in the caption. |
| 24 | Fig. 3 | Cell values are readable now. The 50 × 2 rows lack a "clipping only" sanity row, so the full-budget clipping effect is unknown (it matters for Password). | Add the row or note its absence. |

---

## 5. Check of the earlier major issues (round 4)

| Round-4 issue | Status | Comment |
|---|---|---|
| M1 own traffic: ports, binary metrics, random split | **Partly resolved** | No-port re-run done (16/20, −0.110); detection and FPR reported in Table II; limitation added. The random-split confound remains and is now the main issue (M1 above). |
| M2 claims against the primary family | **Resolved** | "Exploratory" in the abstract; "tended to cost more"; family timing honestly stated ("after the first analysis of TON-IoT and UNSW-NB15 but before …"); family definition matches Table S1 (19/13/9 verified). |
| M3 fine-tuning conclusion, personalised baselines | **Mostly resolved** | Dataset dependence stated (89% against 28%); lr/10 and proximal variants added; two-model union evaluated with FPR. FedBN/Ditto not run (minor 12). |
| M4 BN numbers and framing | **Resolved** | Two run sets labelled; both are in Table S5; Contribution 5 reframed; communication and privacy cost of re-estimation stated; Wang et al. [30] cited. New nuance: minor 1. |
| M5 DP thinness | **Partly resolved** | Full-budget per-class row added; "erases" softened; σ = 0.2/0.3 added, which *exposed* the jump at negligible noise but did not explain it (M3 above). |

The N-BaIoT pipeline was rebuilt since round 4: the two BASHLITE flood classes were dropped, deduplication is done at 32-bit precision and a train/test identity guard was added. The N-BaIoT reversal went from non-significant (−0.032, p = 0.147, 6/9 devices) to −0.081, p = 0.006, 9/9 devices. The changes are well justified in S1-A. But because the headline result changed with preprocessing, the cover letter should state this explicitly.

---

## 6. Spot-check of numbers (hand-computed from the tables)

| Claim | Check | Result |
|---|---|---|
| N-BaIoT local, common test 0.705 (Table III) | Mean of the Table S3 last column | 0.705 ✓ |
| Device-unit own-traffic difference −0.081 ± 0.050, p = 0.006, 9/9 | Table S3 differences, n = 9 | mean −0.0812, SD 0.065, half-width 0.050, t = 3.73, p ≈ 0.006 ✓ |
| Host − α = 0.5 Welch −0.120 [−0.217, −0.022], p = 0.024 | From Table I/II means and SDs | df ≈ 5.4, CI [−0.218, −0.022], p ≈ 0.025 ✓ |
| "10 of the 17 primary comparisons of FedAvg with a reference are significant" | Table S1 | 3 (TON) + 2 (UNSW) + 5 (N-BaIoT) of 7 + 5 + 5 ✓ |
| Holm adjustments, TON-IoT | Order and multipliers | Consistent (e.g., 0.001 × 19 = 0.019) ✓ |
| Fine-tuning gives up 89% / 28% of the gain | (0.710 − 0.551)/(0.710 − 0.530); (0.900 − 0.844)/(0.900 − 0.705) | 0.88 / 0.29 ✓ |
| Short-schedule DP: 5 attack classes lose ≥ 10 points at σ = 0.6 | Fig. 3 | Backdoor, MITM, Password, Ransomware, Scanning ✓ |
| Full-budget DP: 7 of 9 classes lose ≥ 10 points | Fig. 3 | 7 if DDoS (0.93 → 0.83) is just under 10 points unrounded; see minor 3 |
| Communication: 0.76 MB per round, 38 MB per 50 rounds, 12.8 MB data matrix | 23,754 × 4 B; 128,484 × 25 × 4 B | ✓ |

The arithmetic is clean.

---

## 7. Things done well

- **Honest statistics.** A declared primary family with Holm correction, the timing of its declaration stated plainly, exploratory analyses labelled throughout, a power statement, and Welch tests where partitions are not shared. This is rare in the FL-IDS literature and the strongest asset of the paper.
- **Validation-selected and final-round scores side by side** (Tables I, III, IV, S2). The BN analysis shows why this matters (0.862 selected against 0.634 final).
- **Own-traffic scoring with operational metrics** (attack detection, benign FPR), and the two-model union reported together with its 16.4% FPR on unfamiliar traffic. Reporting the downside of one's own suggested remedy is good practice.
- **Same-recipe centralised and local references**, and the centralised learning rate checked over the FedAvg grid (S1-D).
- **DP done properly.** Poisson sampling matched to the accountant, sanity runs without noise and with clipping only, ε ranges over repetitions, the unit of privacy stated (a flow record, not a device), and the untracked privacy cost of tuning disclosed.
- **Careful data hygiene on N-BaIoT.** Deduplication at the model's precision, dropping the degenerate flood classes with a stated reason, and a train/test identity guard.
- **Robustness checks that could have hurt the story** were run and reported: no-port re-runs, fixed against redrawn host partitions (0.770 ± 0.031 against 0.720 ± 0.099), 100 rounds, and the SCAFFOLD collapse.
- **Concise, readable main text** with the detail correctly pushed to the supplement. Fig. 1 is an effective one-glance summary of the primary family.

---

## 8. Requests in priority order

1. A temporal or blocked per-site split for the own-traffic comparison (FedAvg against local only), or a conditional rewording of the reversal everywhere it appears (M1).
2. Prior and label-space correction of the federated model at each site, as an extra row in Tables II and S4 (M2).
3. DP diagnostics at the full budget: an SGD-with-momentum or bias-corrected optimiser, and a larger batch at matched ε; a paired CI for the 17-point gap (M3).
4. Joint client and server learning-rate tuning for FedAdam, or a qualified claim; replace "buys nothing" with a bounded or equivalence statement (M4).
5. A literature audit table supporting the premise in §I (M5).
6. The distribution of re-run differences from nondeterminism, and binary own-traffic metrics with CIs (M6).
