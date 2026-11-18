# Referee Report, Round 4 (third revision)

**Manuscript:** "Federated Intrusion Detection for IoT Networks Revisited: Device-Level Partitions, Tuned Baselines and Record-Level Differential Privacy"
**Venue:** *IEEE Internet of Things Journal*
**Reviewer expertise:** federated learning, differential privacy, ML-based intrusion detection
**Date:** 2026-10-26

**Material examined:**
- `Manuscript_filled.md` and `Manuscript_v2_ieee.pdf` (7 pages, two-column);
- `Supplement_filled.md`;
- the three figures in `figures/`;
- my round-3 report.

I also spot-checked the new own-traffic numbers against the per-seed JSONs (`results_v2/TON_IoT/{local,fedavg,fedprox,fedadam}_natr`) and the N-BaIoT per-device table by hand. All of them reproduce (details in Part 4). I trained nothing.

**Recommendation:** Minor revision. Some of the remaining points need a few cheap runs (own traffic without port features; per-class DP at the full budget). Most are about interpretation and wording.
**Score:** 7 / 10 (round 1: 3; round 2: 5; round 3: 6)

---

## 1. Summary

The paper re-evaluates federated learning for a compact 1D-CNN IDS on three datasets: TON-IoT, UNSW-NB15 and, new in this round, N-BaIoT. The protocol now has:
- partitions redrawn in every repetition;
- validation-based round selection;
- tuned baselines at a matched client learning rate;
- same-recipe centralised and local-only references;
- a declared family of primary comparisons, Holm-corrected within each dataset.

Two device-level partitions are studied: TON-IoT by destination host (now genuinely redrawn) and N-BaIoT by physical device. The main new finding concerns per-site evaluation:
- On the common test set, FedAvg beats local-only training by large, significant margins (+0.190 on TON-IoT hosts, +0.174 on N-BaIoT devices).
- On each site's own test traffic the ordering reverses. On TON-IoT the local model wins in 17 of 20 site–repetition pairs (−0.121, p = 0.015, exploratory). On N-BaIoT the same direction appears, but weakly and not significantly (−0.032, p = 0.147).
- One epoch of local fine-tuning recovers own-traffic accuracy at a cost in breadth.

Other findings:
- FedProx, FedAdam, SCAFFOLD and FedLC never significantly beat tuned FedAvg. FedAdam is significantly worse on TON-IoT.
- BN with server-averaged running statistics makes the model swing between rounds; federated re-estimation fixes this.
- Record-level DP-SGD costs about 8–10 points on a short schedule, almost independent of ε over 0.8–12, and 17 points at the full budget (ε ≈ 2.9).

This is a much better paper than the round-3 version. The round-3 points with the largest impact have been dealt with:
- the "natural" partition is now genuinely redrawn;
- own-traffic evaluation is real;
- a device-structured IoT dataset has been added;
- a primary family has been declared;
- the full-budget DP cost is in the abstract;
- federated BN re-estimation has been tested.

The paper now fits in 7 two-column pages. What remains is mainly about how strongly the new own-traffic and fine-tuning findings are stated, and one plausible confound behind them.

**Novelty and IoT-J fit.** The contribution is still methodological (protocol, references, honest baselines) rather than algorithmic. With N-BaIoT per-device partitions and per-site evaluation, it now fits IoT-J. The own-traffic versus common-test-set trade-off is the most interesting result for an IoT audience. However:
- it is the classic motivation for personalised FL, so it needs to be placed in that literature (M3);
- it rests on a record-level random split, which may inflate the local model's advantage (M1).

---

## 2. Major issues

### M1. The own-traffic reversal may be driven by host-specific shortcuts and the random record split, not by "a site's own attack traffic differing"

**Location:** §V-C ("Common test set versus own traffic"), abstract, Contribution 2, Conclusion.

**The concern:**
- TON-IoT is split 70/15/15 at random by record, so a host's test flows come from the same capture period, and often the same sessions, as its training flows.
- The feature set includes source and destination ports. The paper itself calls these testbed shortcuts, and they are closely tied to the destination host.
- A local model trained only on one site's hosts can memorise host- and port-specific patterns that its own test traffic shares. A model averaged over four sites cannot do this as well.
- The authors' explanation ("a site's own attack traffic differs from the same classes elsewhere, and its local model fits it better") is one reading. "The local model overfits host-specific artefacts that a random split rewards" is another. The paper cannot tell them apart at present.

**What the JSONs show** (`site_selected` fields, which also store binary metrics; my computation):

| Own-traffic metric (TON-IoT host partition, mean over 20 site–repetitions) | Local only | FedAvg |
|---|---|---|
| Multi-class macro-F1 | 0.791 | 0.670 |
| Attack detection rate (binary) | 99.7% | 96.5% |
| Benign FPR | 2.6% | 3.4% |

- So most of the 12-point macro-F1 gap is confusion between attack types. Only about 3 points come from attacks missed outright.
- A 3-point loss in detection is still operationally relevant. But the binary view is what an operator acts on, and it puts the "reversal" in proportion.
- The local model's 99.7% own-traffic detection rate is itself a hint of in-distribution memorisation.

**Required:**
1. Repeat the host-partition own-traffic comparison without the two port features. This is a handful of runs; the no-port pipeline already exists (Supplement S4). If the reversal survives, the claim becomes much stronger. If it does not, it must be reframed.
2. Report own-traffic attack detection rate and benign FPR alongside macro-F1 (Table II and Table S3). The data are already stored.
3. Add a sentence in Limitations: a random record split favours a site's local model on its own traffic, and a temporal split (train on early traffic, test on later traffic per host) would test whether the advantage holds for future traffic. If the TON-IoT timestamps allow it, a temporal per-host split would be the ideal check. I do not insist on it for this round.
4. Until 1–3 are done, soften the abstract's general statement "federation buys detection of attacks seen elsewhere, not better detection of a site's own traffic". It rests on one exploratory test on one dataset; N-BaIoT is not significant (p = 0.147) and is driven by two cameras.

### M2. Claims in the abstract and conclusion that the primary family does not support

**Location:** abstract sentences 5–6, Contribution 3, §V-C, §VI "Deployment", Conclusion.

The paper declares a primary family, and the abstract now leads with † effects. But three abstract-level claims are not primary-significant, and the abstract states them without qualification.

**(a) "At matched label skew, device partitions cost more against pooled training than Dirichlet partitions."**
- The evidence: TON-IoT host vs α = 0.5, −0.120 [−0.257, +0.017], p = 0.071, exploratory; N-BaIoT device vs α = 0.5, −0.024 [−0.059, +0.012], Holm p = 0.553.
- The text in §V-C is honest ("Neither difference … is significant on its own"). The abstract, Contribution 3, the Discussion ("should expect device-level partitions to cost more") and the Conclusion are not.
- "Matched label skew" is not matched either. On N-BaIoT the device partition is markedly milder (38% vs 54% top-two share).
- The TON-IoT host partition also has strong quantity skew: the largest host holds 26% of the records and client sizes range from 23.7k to 52.8k. So the extra cost may come from size imbalance rather than from the partition being by device.
- **Fix:** write "tended to cost more (not significant on either dataset)". Mention the quantity-skew confound. Ideally, add a Dirichlet control matched on both label skew and client sizes.

**(b) The own-traffic reversal.** It is exploratory, yet it is the paper's headline (see M1). Either add it to the primary family in a clearly labelled post-hoc way, or mark it as exploratory in the abstract (e.g., "in an exploratory analysis").

**(c) "Fixed before the analyses reported here."** The primary family was evidently declared during the revision, after many of these results had been seen in rounds 1–3. The claim of pre-specification is therefore not credible as worded. Say "defined in this revision, before the device-partition and N-BaIoT analyses were run" (if true), or simply "declared post hoc to limit multiplicity".

**Family definition.** The definition in §III-D does not match Table S1:
- the N-BaIoT family (11) includes "device − α = 0.1" and "device − α = 0.5", which the text does not mention;
- the equivalent TON-IoT comparison (host − α = 0.5) is treated as exploratory.

Make the two datasets consistent and describe the family exactly.

### M3. Local fine-tuning: the "trade-off, not escape" conclusion is TON-specific and untested against personalised FL

**Location:** §V-C ("Local fine-tuning"), §VI "Deployment", Conclusion.

**What the numbers show:**
- On TON-IoT, fine-tuning gives up about 89% of FedAvg's common-test-set gain over local training: (0.720 − 0.551) / (0.720 − 0.530). The fine-tuned model is essentially the local model again.
- On N-BaIoT it gives up only about 24% ((0.820 − 0.779) / (0.820 − 0.646)), and own-traffic parity with the local model is restored (−0.010 ± 0.017).
- So on N-BaIoT fine-tuning comes close to dominating both references. "Fine-tuning moves a site along the trade-off rather than escaping it" holds for TON-IoT, not for N-BaIoT.

**Problems with the fine-tuning setup and the recommendation:**
- Only one fine-tuning recipe is tried: full-model Adam, the same learning rate, no regularisation.
- Standard personalised FL methods are designed precisely for this trade-off: Ditto (already cited), FedBN (cited), head-only fine-tuning, proximal fine-tuning, or a reduced learning rate. None is tested.
- "A site gains most by keeping both the federated model and a locally adapted copy" is not evaluated: no rule for combining the two models (union of alerts, max-confidence, gating), and no measure of the combined FPR.

**Fix:**
- State the dataset dependence.
- Either add one personalised-FL baseline (Ditto, or proximal/head-only fine-tuning; cheap at this model size) and one simple two-model combination with its benign FPR, or remove "keeping both" from the Conclusion and Deployment paragraph and present it as a hypothesis.
- Position the own-traffic finding against the personalised FL literature. That local models win on their own distribution is the standard motivation there; what is new here is quantifying it for IDS with device partitions.

### M4. Normalisation section: inconsistent numbers and a framing that contradicts the data

**Location:** §V-D, Contribution 5, §III-D, Fig. 2, Supplement S6/Table S4.

**Inconsistent numbers:**
- The FedAvg BN swing is 0.110 in §V-D and 0.151 in Supplement S6.
- The final-round FedAvg-BN score is 0.634 ± 0.339 in §V-D ("same runs", 5 repetitions) and 0.739 ± 0.087 in Table S4 (3 repetitions).
- §III-D says the server-averaged BN study uses three repetitions; Fig. 2's legend says five.
- These are presumably two run sets (the original 3-repetition BN runs and the 5-repetition federated-re-estimation runs, which also log server-averaged scores). The reader cannot tell this. Label the run sets, keep one set of numbers in the main text, and add rows for BN with federated re-estimation to Table S4.

**Framing:**
- BN with federated re-estimation beats GN at the selected round: +0.032 [+0.009, +0.055] at α = 0.5 and +0.039 [+0.018, +0.060] at α = 0.1, both uncorrected.
- Contribution 5 still says the finding is "in line with Hsieh et al.'s recommendation of group normalisation". With the authors' own data, the accurate statement is that GN avoids the swing but BN with re-estimated statistics is at least as good and possibly better.
- All core results use GN, the weaker of the two. That does not invalidate the comparisons between algorithms, but it should be said.

**Missing details:**
- Federated re-estimation costs one extra round-trip per normalisation layer per round, and it releases activation statistics. Neither the communication nor the privacy cost is stated; the latter matters for the DP section.
- Related work on BN in FL is missing; cite at least one of the works on BN statistics mismatch in non-IID FL (e.g., FixBN- or FedTAN-type methods) besides Hsieh et al. and FedBN.

### M5. The DP results are thinner than the abstract suggests

**Location:** abstract last sentence, §V-E, Table IV, Fig. 3, Conclusion.

**(a) The per-class evidence comes from a different run than the abstract's headline.**
- The abstract couples "ε ≈ 2.9 costs 17 points" with "concentrated in a few attack classes".
- Fig. 3, the only per-class evidence, is from the short 20 × 1 schedule. The full-budget σ = 1.8 run has no per-class breakdown.
- **Fix:** add the per-class F1 for the full-budget run (one row in Fig. 3 or a table in S7), or detach the two clauses.

**(b) The cost is flat in ε.**
- Noise costs about 8 points at ε = 12.15 and 10 at ε = 0.81, and the four σ levels lie within ±0.05 of each other.
- A fifteen-fold flat region with the jump between σ = 0 and σ = 0.6 is unusual. It suggests that either small amounts of noise interact badly with something in the training (Adam with noisy gradients, Poisson batches with GN, or selection noise), or the 3-repetition CIs simply cannot resolve the curve.
- **Fix:** add one or two smaller noise levels (σ = 0.2–0.3) to show where the cost appears, or state the flatness as unexplained.
- The comparison is not uniform either. The short-schedule costs are quoted against non-private FedAvg (0.735), while the full-budget cost is quoted against "the same code without noise" (0.856). Use one reference consistently, or report both.

**(c) "Erases several attack classes" (Conclusion) is too strong.**
- Only MITM reaches zero, and it starts at 0.24.
- Password drops from 0.54 to 0.15–0.34; backdoor, ransomware and scanning lose 10–20 points.
- Write "sharply degrades several attack classes".

---

## 3. Minor issues

| # | Location | Issue | Concrete fix |
|---|---|---|---|
| 1 | Abstract | 249 words, at the IEEE limit, and full of figures. "−0.121" uses a hyphen-minus throughout the paper. | Trim to about 220 words (e.g., drop the "+0.174" detail or the FedLC list). Use a typographic minus (−). |
| 2 | Abstract, §V-B | "FedProx, FedAdam, SCAFFOLD and FedLC never improve significantly on FedAvg" hides that FedAdam is significantly worse on TON-IoT (†). | Add "and FedAdam is significantly worse on TON-IoT". |
| 3 | §V-B last sentence | "detects differences of a few points" is vague. | Give the approximate minimum detectable difference (≈ CI half-width × 1.4 at 80% power, i.e. ~5–7 points for the α = 0.5 comparisons). |
| 4 | §III-D | No non-parametric check (open since round 2). With n = 5 the exact two-sided Wilcoxon cannot go below p = 0.0625. | One sentence acknowledging this, or a sign-flip permutation test for the † comparisons. |
| 5 | §V-A | "Exploratory linear trend −0.046 per level, p < 0.001": the model is unspecified (equally spaced ordinal levels? pooled points ignoring repetition pairing?). | State the model (e.g., a regression on level index with a repetition random effect) in one clause or in S2. |
| 6 | §V-C, TON host vs α = 0.5 | "Paired" test between two different partition schemes; the pairing by seed has no shared partition. | Use Welch's test, or say what is shared across the pair. |
| 7 | §V-C own traffic | The unit of analysis differs between datasets without saying so: TON-IoT uses repetitions (n = 5), N-BaIoT uses devices averaged over repetitions (n = 9). | State the unit for each, e.g., "(n = 9 devices, means over five repetitions)". |
| 8 | §V-C own traffic | Own-traffic test sizes (~5–11k records per site) and the number of classes present (7–10) are not given. A macro-F1 over classes with very few own-test records is noisy. | Give the ranges in S1-B and the minimum class count per site. |
| 9 | Table II | The own-host column has no CI. Its values for FedAvg (0.670) and FedProx (0.646) coincide with the final-round common-test column. I verified that this is a genuine coincidence in the JSONs, but readers will suspect a fill error. | Add ± CI to the own-host column, which makes the distinction obvious, and add own-traffic attack detection/FPR (M1). |
| 10 | Table III | No "Selected round" column, unlike Table I. | Add it, or remove it from Table I and give the selected rounds in S4. |
| 11 | Table I (PDF p. 3–4) | The table breaks across pages: the last row (FedLC, α = 0.1) is split, with "0.669 ±" on p. 3 and "0.094 … 1.22" on p. 4 under a repeated header. | Keep the table on one page (float to the top of p. 4, or move the SGD rows to S3). |
| 12 | Fig. 1 | The y-axis labels and legend are very small at column width. Rows with only two datasets leave uneven gaps. | Make it a two-column figure (`figure*`) or enlarge the fonts to ≥ 8 pt at print size. Mark † significance on the plot. |
| 13 | Fig. 2 | The grey min–max band of the centralised BN model (3 repetitions) dominates the TON-IoT panel and hides the others. | Use a lighter band or a separate panel for the centralised model. Report the swing numbers in the caption. |
| 14 | Fig. 3 | The cell annotations are unreadable at print size. Class names are lower-case codes. | Larger font or fewer decimals; capitalise class names to match the text (e.g., "MITM", "XSS"). |
| 15 | §IV FedLC | n_{k,c}^{−1/4} is undefined for absent classes. | State the convention (count set to 1, as in the code). |
| 16 | §III-C | "the selection is deployable because clients can report validation confusion counts": this implies a validation split held by each client, but §III-A describes one pooled validation split, and the per-client validation sets under the host partition are not described. | One sentence: validation records are partitioned with the training records (same host or device assignment). |
| 17 | §III-A | "Preprocessing is fitted centrally": fine as a simplification. But for the own-traffic analysis, the min–max scaling fitted on pooled data leaks cross-site ranges into local models. | Mention it in Limitations (it slightly helps local models generalise, so it is conservative for the common test set). |
| 18 | §VI Evaluation | "FedLC's best τ on UNSW-NB15 lay at the edge of two successive grids" reads as narrative. | Keep it as a methodological lesson but in the past tense, without the history. The S3 sentence "In an earlier tuning round with a single tuning repetition, learning rate 0.3 was selected …" is also history; delete it. |
| 19 | Supplement S4 | "A longer budget therefore narrows the gaps at most slightly": the TON-IoT α = 0.5 gap narrows by ~30% (−0.059 → −0.042) on non-paired run sets. | Write "narrows the TON-IoT α = 0.5 gap by about a third and leaves the others within their wide intervals". |
| 20 | §II | "Few of these works report results over several partitions or state how the reported round was selected" is still unsupported (open since round 2). | Cite two or three examples, or drop the sentence. |
| 21 | §II | Personalised FL for IDS is not discussed, although the paper's central new finding is about personalisation. | Add 2–3 references on personalised or clustered FL-IDS. |
| 22 | Declarations | GitHub URL only; no archived DOI or commit hash (open since round 2). | Archive a release on Zenodo and cite DOI + tag. |
| 23 | Title block | "Affiliations: to be completed by the authors before submission." | Complete the affiliations; IoT-J also needs author biographies at acceptance. |
| 24 | Table IV | CIs but no SD column, unlike Tables I–III (round-2 m7). | Add SD or state that the ± values are CI half-widths in the caption (they are, but say so). |
| 25 | Ref. [32] | A 2026 ACM CSUR article; please verify volume, issue and DOI. | Check against the publisher page. |
| 26 | §V-E | "including well-represented ones" without the class counts. | Give the training counts for password, backdoor, ransomware and scanning (one parenthesis or S7). |

---

## 4. Spot-check of new numbers

| Claim | Check | Result |
|---|---|---|
| TON host own traffic: local 0.791, FedAvg 0.670, 17/20, −0.121 ± 0.083, p = 0.015 | Recomputed from `local_natr`/`fedavg_natr` `site_selected` | Matches (0.7913 / 0.6700 / 17 / −0.1213; t-test on 5 repetition means gives p ≈ 0.016) |
| Table II own-host FedAvg 0.670 = final 0.670; FedProx 0.646 = 0.646 | Per-seed values | Genuine coincidence (own 0.67003 vs final 0.66977; 0.64578 vs 0.64589) |
| N-BaIoT Table S3: mean difference −0.032 ± 0.046, p = 0.147, 6/9 worse | Hand-computed from the table, n = 9 devices | Matches (SD 0.060, t = 1.61, p ≈ 0.146; half-width 0.046) |
| N-BaIoT local, common test set 0.646 | Mean of the last column of S3 | 0.646 |
| Holm values in Table S1 (TON) | Monotonicity and order with rounded p | Consistent |
| DP per-class: five attack classes lose ≥ 10 points at both extremes | Fig. 3 cells | Backdoor, MITM, password, ransomware and scanning: yes; injection, XSS and DDoS: no |

The arithmetic remains clean.

---

## 5. Status of the round-3 points

### Round-3 status of the earlier points

| # | Point | Status | Justification |
|---|---|---|---|
| MC1 | Tuning at a mismatched client lr | **Resolved** | Unchanged; matched lr. |
| MC2 | Thin tuning, grid edges | **Mostly resolved** | Three tuning repetitions on TON-IoT and UNSW-NB15 (two on N-BaIoT); the FedLC grid is widened to 100 and the values are listed (S1-D); only the harmless μ edge remains; tuning at α = 0.5 only is listed as a limitation. |
| MC3 | Multiplicity, small n | **Mostly resolved** | A primary family is declared (19/13/11) and Holm is applied within each dataset. Still open: the "fixed before" wording (M2c), the family description mismatch (M2), no non-parametric check (minor 4). |
| MC4 | DP scope, tuning, schedule | **Resolved** | Full budget with 3 repetitions in the abstract; best C now interior; empty Poisson batches documented; α = 0.1 / q → 1 caveat added. |
| MC5 | Server-held validation | **Resolved** | Unchanged (minor 16 asks for one clarifying sentence). |
| MC6 | Per-site evaluation | **Resolved** | Real own-traffic evaluation (new caveats in M1). |
| MC7 | SCAFFOLD narrative | **Mostly resolved** | One history sentence remains in S3 (minor 18). |
| MC8 | BN mechanism, FedBN, Fig. 2 | **Mostly resolved** | Federated re-estimation tested; Fig. 2 has 5-repetition bands and the legend is off the data. FedBN is not tested; the numbers are inconsistent (M4). |
| MC9 | 50-round budget | **Mostly resolved** | Labelled exploratory; "narrows at most slightly" is slightly understated (minor 19). |
| MC10 | Novelty / IoT fit | **Mostly resolved** | N-BaIoT per device added; host partition redrawn. Personalised FL is still only naive fine-tuning (M3). |
| m1–m12, m15, m16 | — | **Resolved** | As in round 3. |
| m7 | SD columns | **Partly** | Table IV still lacks SD (minor 24). |
| m13 | Archived DOI / commit hash | **Not resolved** | Minor 22. |
| m14 | "Few of these works…" | **Not resolved** | Minor 20. |

### Round-3 new points

| # | Point | Status | Justification |
|---|---|---|---|
| N1 | Natural partition not redrawn; text/code mismatch | **Resolved** | Hosts are taken in a fresh random order per repetition and assigned to the currently smallest client; 682 hosts, with the largest-host and top-ten shares reported; the fixed and redrawn designs are contrasted (0.770 ± 0.031 vs 0.720 ± 0.099). |
| N2 | "Own traffic" was the common test set | **Resolved** | Each site is scored on its own hosts' test records, and on N-BaIoT on the device's test split. New interpretation risk: M1. |
| N3 | Abstract leads with non-significant effects | **Partly resolved** | The federation-gap headlines are now †. But the device-vs-Dirichlet and own-traffic claims in the abstract are not primary-significant (M2). The no-port gap is still not in the abstract. |
| N4 | DP headline on the favourable schedule | **Resolved** | The 17-point full-budget cost is in the abstract, with 3 repetitions; the C grid is interior. New issue: the per-class clause belongs to a different run (M5a). |
| N5 | Wrong FedAdam statement | **Resolved** | FedAdam reported as significantly worse on TON-IoT; the history paragraph is removed. |
| N6 | BN conclusion vs recalibration | **Resolved (with new issues)** | Federated re-estimation tested and the conclusion softened. The numbers are inconsistent and Contribution 5 is framed in favour of GN (M4). |
| N7 | Port shortcut | **Partly resolved** | Algorithm ranking checked without ports (S4). The no-port gap is not in the abstract, and the new host/own-traffic analyses are not checked without ports (M1). |
| N8 | FedLC grid / formula | **Resolved** | Grid {0.1 … 100}, selected 3/10/30 (interior), formula in §IV (absent-class convention missing, minor 15). |
| N9 | Wording details | **Mostly resolved** | "Costs as much as α = 0.1" now stated; SCAFFOLD is an observation; own-class CIs over repetition means; the 100-round study is labelled exploratory. The "matched label skew" phrasing is still loose (M2a). |
| N10 | Format, density, IoT fit | **Mostly resolved** | Two-column IEEE layout, 7 pages (under the 8-page limit); paired comparisons moved to Table S1; N-BaIoT added. Affiliations still a placeholder; figure legibility (minor 11–14). |

---

## 6. Clarity after condensing

The condensation mostly works; the main text reads well. Places where the move to the supplement now leaves the reader short:
- **Definition of "own traffic" and its sizes** (minor 8). Without them the 17/20 result cannot be judged.
- **The two BN run sets** (M4).
- **The per-client validation data under device partitions** (minor 16).
- **The "restricted to a site's classes" metric** (0.138 ± 0.072). Its ≥1% threshold and its common-test-set basis are defined only in S4. Give one clause in §V-C.
- **The DP full-budget configuration** (σ = 1.8, 50 × 2, 3 repetitions). It is only in prose. Add it as a row in Table IV so that the abstract's headline is in a table.

---

## 7. Overall assessment

The authors have done the work asked for in round 3, and in several places more: the N-BaIoT device partition, real own-traffic scoring, fine-tuning, and federated BN re-estimation. The protocol is now a good template for FL-IDS evaluation, and the paper is concise.

What keeps it from acceptance is that the newest and most quotable findings are stated more firmly than the evidence allows:
- the own-traffic reversal is exploratory, plausibly inflated by ports and the random split, and weak on N-BaIoT;
- "device partitions cost more" is not significant on either dataset;
- "fine-tuning trades breadth for own-traffic accuracy" holds for TON-IoT but not for N-BaIoT;
- "keep both models" is untested.

These need wording changes plus two cheap checks (own traffic without ports; per-class DP at the full budget).

**Recommendation: Minor revision. Score: 7/10.**

### Top five requests
1. Re-run the host-partition own-traffic comparison without port features, and report own-traffic attack detection and FPR (M1).
2. Bring the abstract, contributions and conclusion in line with the primary family: mark own traffic as exploratory, and the device-vs-Dirichlet difference as non-significant and confounded by quantity skew; describe the family exactly; drop "fixed before" (M2).
3. Qualify the fine-tuning conclusion by dataset; test one personalised-FL method or a two-model combination, or drop "keep both" (M3).
4. Reconcile the BN numbers and run sets, and reframe Contribution 5 given that re-estimated BN beats GN (M4).
5. Give per-class DP results for the full-budget run, explain or probe the flat ε curve, and soften "erases" (M5).
