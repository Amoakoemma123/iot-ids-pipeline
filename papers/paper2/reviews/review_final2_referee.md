# Referee report: Journal of Information Security and Applications

**Manuscript:** "What Can an Attacker Actually Change? Evasion and Hardening of Compact Flow-Based Intrusion Detectors Under a Realistic Threat Model" (v5, Manuscript_filled.md, Supplement_filled.md, highlights from build.py)

**What I checked:** the full manuscript, the supplement (S1–S14), the highlights, the code (`run_adv_v3.py`: `Space.selected`, `train_robust`, `_pgd_action`, `search`, `adaptive`, `augmentation`, `evaluate_attacks`; `run_adv_r2.py`: `Thresh`, `tau_for`, `part_op`, `part_opu`, `part_fs`, `feasible`), `compute_values.py` (primary family, Table 7 builder), and per-seed `op`/`opu` results in `results_r2/TON_IoT`. I did not load data or models. I read the round-1 to round-3 reports and do not repeat points that were fixed.

**Score: 6/10. Recommendation: major revision.**

---

## Summary

The paper attacks four compact flow classifiers (1D-CNN, MLP, RF, histogram GBDT) on TON-IoT and UNSW-NB15. The attacker can only pad, add packets, delay and, on UNSW-NB15, choose the TTL. Derived features are recomputed, and success counts only detected attack flows turned benign. Attacks include a dense black-box grid, a score-based adaptive search, white-box action PGD (including BPDA) and transfer. The paper compares them with a strengthened feature-space reference. Defences are feature-space AT, CCAT, action-space AT, TRADES, a clean-loop control and tree augmentation. They are compared at each model's own operating point and at a matched benign FPR under a common threshold rule.

Main claims:
1. A strong feature-space attack overstates what volume and timing changes achieve, partly because it breaks feature dependencies.
2. Trees are harder to evade, but the gap rests on the budget floors (TON-IoT) and on the TTL features (UNSW-NB15).
3. At a matched rule and FPR, no network defence significantly lowers TON-IoT end-to-end evasion; TRADES raises it. On UNSW-NB15 the action-space defences help.
4. One round of tree augmentation cuts GBDT evasion on TON-IoT from 35.3% to 3.0%, but unseen TTL values still evade 30.8–40.3% on UNSW-NB15.

This is careful, unusually self-critical work. The threat model is well specified, the code matches the text in every place I checked, and the earlier rounds fixed most of the methodological problems: the decision-rule confound, BPDA, the relaxed attack and the volume-only UNSW endpoint. Four problems remain, and all of them bear on the defence conclusions, which are now the paper's main message:
- the network and tree defences are trained with inner searches of very different strength;
- the single-point matched-FPR comparison is fragile and underpowered, so "no significant gain" carries little information;
- an unmodelled side effect of the main action is described as "conservative" when its direction is unknown;
- the paper is too long and too dense for the journal.

---

## Major issues

### 1. The network and tree defences are trained against different inner searches, and this confounds "trees can be hardened cheaply, networks cannot"

**Location:** §4.4 (AT-threat, TRADES, tree augmentation); §5.3; §6.1 (third lesson); §7; Abstract; Highlights 3 and 5.

**Why it matters.**
- **Tree augmentation** (`augmentation()`) evaluates all 2^k corners of the β = 0.5 box plus 24 random points for every training flow and keeps the most benign-looking candidate.
- **AT-threat and TRADES** (`train_robust` → `_pgd_action(..., steps=7)`) start from a uniform random point and take 7 sign steps of 0.05 of the box. They can therefore move at most 0.35 of the box from the start and never start at a corner.
- **Where evasions lie.** §5.1 and Table S2 show that TON-IoT evasions come mainly from padding small flows up to the floors, that is, from at or near the upper corners. AT-threat reaches that region only when the random start is already ≥ 0.65 in every relevant coordinate.
- **Consequence.** The trees were trained on the region where evasions occur, and the networks mostly were not. The BPDA check (Table S12) shows the gradients are informative enough, but it does not address how far the inner maximiser can travel. Round 2 noted the 0.35 reach as a minor point. The text now states it, but the consequence has not been tested.

**The paper's own data point the same way.** Table S10: on seed 0, AT-threat trained at β = 1 left 24.9% missed or evaded at the matched FPR, and at β = 0.25 23.3%, against 55.7% for the main β = 0.5 configuration. At β = 1 the 0.35-of-box reach covers 0.7 of the β = 0.5 box in absolute terms. Seed 0 is the main configuration's worst seed (per-seed matched e2e: 0.557, 0.195, 0.252, 0.398, 0.516), so this is not conclusive. It is, however, the same kind of seed-0 signal that led the authors to rerun AT-all at ε = 0.1 on all five seeds. AT-threat did not get the same treatment. That is an asymmetry in tuning effort between the two families being compared.

**Fix (cheap, no new attack code needed):**
- (a) Train AT-threat and TRADES with a stronger inner maximiser: restarts from the zero and full corners (as `_pgd_action_strong` already does), more steps, or simply the augmentation's corner + random grid as a black-box inner maximiser. The last is the most direct like-for-like comparison with tree augmentation.
- (b) Rerun AT-threat at β = 1 (and/or 0.25) on all five seeds, as was done for AT-all ε = 0.1.
- (c) If the network results do not change, the claim is much stronger. If they do, the third lesson and the network part of the conclusion must be rewritten.

At minimum, state the asymmetry in §4.4 and §6.2 and remove the implied comparison "the only defence in our study that did so" (§6.1).

### 2. The matched-FPR comparison rests on one tail threshold, is underpowered, and its null result is worded as a finding

**Location:** §4.5 "Operating point" and "Tests"; Table 7; §5.3 heading "Networks: no gain on TON-IoT…"; §6.1 second lesson; §7 ("Defences for networks mostly moved the decision threshold"); Abstract; Highlight 3.

**Why it matters.**
- **Seed variance is very large at the operating point.** The TON-IoT threshold sits at about 1.5% of validation benign flows, in the tail of p_benign. Per-seed end-to-end evasion at the matched FPR ranges 36.1%–87.8% for the clean loop and 19.5%–55.7% for AT-threat, and the undefended CNN ranges 37.5%–57.0%.
- **The paired test is very wide.** The paired differences for AT-threat have SD ≈ 21 pp over five seeds. The −8.6 pp effect therefore has a 95% CI of roughly −34 to +17 pp (my computation from the per-seed JSON). The study can only detect differences of about ±25 pp. "None significantly reduced" is true, but the heading "no gain" and the conclusion "defences for networks mostly moved the decision threshold" read as evidence of absence, which the data cannot give.
- **The pairing assumption does not hold.** Pairing by seed assumes a shared initialisation makes the runs comparable. The defended networks are trained with a different loop (30 × 200 steps, different batches), and the per-seed undefended and AT-threat values are not positively correlated (seed 1: 0.43 vs 0.19; seed 0: 0.42 vs 0.56). Paired t-tests buy nothing here and may mislead.
- **The match is systematically tighter for defended models.** Their test FPRs at the "matched" threshold are lower than the reference's (TON-IoT: 0.010–0.0145, mean ≈ 0.013 for AT-threat, against a mean of 0.0155 for the undefended CNN under the same rule). In a steep tail this biases the comparison against the defences.
- **TRADES' penalty is mostly clean detection loss under this rule.** At the matched threshold TRADES detects only 60.9% of attack flows before any attack (Table 7). The "+45.8 pp, TRADES increased it" in the abstract is therefore largely a calibration and detection effect of one threshold rule on p_benign. It may not survive another reasonable rule (for example, thresholding 1 − max attack probability).

**Fix:**
- (a) Replace or complement the single operating point with an attack-aware ROC: end-to-end "missed or evaded" against benign FPR over a range (for example 0.5%–5% on TON-IoT), with seed bands. Optionally add a partial AUC over that range as the primary endpoint. This removes both the threshold-tail fragility and the val/test FPR mismatch.
- (b) Report the difference with a CI (and unpaired Welch as a check). Word the TON-IoT network result as "not distinguishable from zero; the 95% CI spans −34 to +17 pp", and change Highlight 3, the §5.3 heading and §7 accordingly.
- (c) Decompose "missed or evaded" into clean misses and evasions in Table 7. This is especially needed for UNSW-NB15, where most of the gain of the clean loop and AT-threat comes from detecting 93–95% instead of 74% of attacks before modification.
- (d) Add a second matched rule as a sensitivity check for the TRADES claim.

### 3. The unmodelled destination response is called "conservative", but its direction is unknown

**Location:** §3 "Attacker" ("These restrictions are conservative: extra packets may in practice draw extra acknowledgements from the destination, which we do not model"); §6.2.

**Why it matters.**
- The flagship TON-IoT result is driven by adding packets and padding to small flows: scans, ransomware traffic and similar (§5.1, Table S6).
- Against a closed or filtered port, each added packet draws a RST or ICMP reply, or nothing. Against an open port it draws ACKs. This changes destination-side counters (dst_pkts, dst_bytes, dst_ip_bytes) and possibly the Zeek conn_state of the flow.
- If any of these are among the 25 selected TON-IoT features, the realistic attack may produce feature vectors that were never evaluated. Whether that helps or hurts the attacker is unknown, so it is not "conservative". It is an unquantified error in either direction on the main result.
- The manuscript does not list the selected features anywhere, in the main text or the supplement, so the reader cannot judge the exposure.

**Fix:**
- (a) Add a table (supplement is fine) of the 25 TON-IoT and 10 UNSW-NB15 selected features, each marked as attacker-set, recomputed, fixed by the attack, or destination-side and potentially reactive.
- (b) Run a simple sensitivity analysis: for each added packet, assume one destination reply of header size (and, for single-packet flows, the conn_state change implied by a reply), recompute those features, and rerun the β = 0.5 grid. The feasibility check (Table S9) shows this costs minutes.
- (c) Replace "conservative" with an accurate statement.

### 4. Headline numbers mix decision rules and metrics, and the tree claim is still a lower bound

**Location:** Abstract; Highlight 5; §7.

**Why it matters.**
- **Two rules and two metrics.** After round 3 the paper argues that defences must be compared under the same rule. The abstract applies that standard to the networks ("share missed or evaded at the undefended rate and rule: 38.3% against 46.9%") but reports the trees at their own argmax rule and on a different metric ("ASR from 35.3% to 3.0%"). The matched-rule tree result (Table 7: 16.9% → 1.6% missed or evaded, three seeds) is still a large gain. It should be the one quoted, or both rules should be quoted for both families.
- **Lower bound.** The 3.0% is a search-based lower bound. On the saturation subsample the union of all searches gave 4.1% against 3.5% for the strongest attack (Table S8). Round 3's request for an exact MILP attack on TON-IoT, where the action map is affine, is still open. The abstract and conclusion should say "at least 3.0% under our strongest search".
- **Generalisation figures.** The held-out-action results (9.9%–19.5%) belong next to the 3.0% in the abstract or conclusion, not only in §5.3.

**Fix:**
- Use one rule and one metric for both families in the abstract and highlights.
- Add "lower bound" wording.
- Run the exact attack on the 500-flow subsample for the TON-IoT RF and GBDT, undefended and augmented, or explain concretely why it is infeasible.

### 5. Length and density: the paper should be cut by about 25–30%

**Location:** whole manuscript (≈13.2k words before references; 9 main tables, two of them 24 rows; the abstract is 246 words with 17 numbers).

**Why it matters.** JISA papers are usually 8–11k words. The bigger problem is density: most result sentences carry four to eight numbers with confidence qualifiers, so the argument is hard to follow. Examples are the §5.1 feature-space paragraph (about 420 words, 20+ numbers) and the §5.3 network paragraph (about 480 words, 30+ numbers). An editor may return the paper on readability alone.

**Concrete cuts:**
- **Table 8** (24 rows): move to the supplement. Table 7 already carries the defence argument. Keep three sentences on clean cost in the text.
- **Table 4:** drop the TON-IoT "Grid, TTL only" and "Grid, both" columns, which are n/a or duplicates. Keep grid, grid + adaptive, strongest (volume only and all actions) and transfer.
- **§5.1 "The feature-space reference":** state the result in three sentences (magnitude, consistency, the UNSW-tree exception) and leave the crossovers to Table 5.
- **§5.3 "Is AT-all a fair baseline?":** keep the BPDA result (27.5% → 91.4%) and the sweep outcome in two sentences; the rest can go to the supplement.
- **§5.3 network paragraph:** give the primary results in a sentence or two and let Table 7 carry the pairwise p-values.
- **§2.3** largely restates the contributions list; merge it into Table 1's lead-in.
- **§6.1:** the five lessons restate §5 with the numbers again; cut them to one or two sentences each without numbers.
- **§6.2:** convert to a compact list of limitations.

The target is about 9.5–10k words including tables.

### 6. Lesson 1 still contradicts the evidence

**Location:** §6.1 first lesson ("not a conservative substitute"); §7 second sentence.

Round 3 raised this and the wording was only partly changed. An attack that evades more is an upper bound, which is conservative for the defender, for every model except the UNSW-NB15 trees at small ε. The evidence supports two narrower claims:
- (i) the ε-ball misstates the size of the threat, mostly upward, and for networks partly through broken dependencies (the relaxed attack adds 7.7–16.1 pp);
- (ii) it can rank defences differently: on UNSW-NB15 the ε-ball defences are the least evaded CNNs under the corner search and the most evaded under the realistic attack, and a weak ε-PGD made them look robust.

Lead with (ii), which is the practically important and genuinely non-conservative failure. Drop "not a conservative substitute", or limit it explicitly to the UNSW-NB15 trees and to defence rankings.

---

## Minor issues

1. **Table 7, column "Δ missed or evaded (Holm p)".** For the tree rows the p-values (0.011, 0.019, 0.070, 0.026) are unadjusted secondary tests. In `compute_values.op_rows` only PRIM_NET rows get Holm values; the tree primary tests use the strongest-attack ASR at the argmax rule. Mark the tree cells "(p, unadjusted, 3 seeds)" or give them their own column. Note in the text that the UNSW-NB15 RF augmentation gain at the matched rule (−7.9 pp, p = 0.070) is not significant.
2. **§5.4 vs Table 9.** The text says the TTL-free detectors were evaded on "36.3% (CNN), 42.1% (MLP)", while Table 9 gives 36.1% and 41.9%. The text uses `strongest_b0.5` (which includes the white-box PGD) and the table uses `strongest_volume_b0.5`. Without TTL features the two should be one measure. Use the same union in both places.
3. **Table 5, "Realistic, volume and timing".** The TON-IoT RF/GBDT values are 25.0%/35.5% and the UNSW-NB15 RF/GBDT 20.4%/33.9%, against 25.2%/35.3% and 19.4%/34.3% in Table 4. They come from different seed sets. State the seeds in the caption.
4. **Pre-registration of the primary family.** The family was revised after the operating-point results; this is disclosed. For transparency, report in the supplement what the original family (argmax reference, ASR endpoint) gave. Also align the code comment in `compute_values.py` ("revised in round 3 before the new runs were read") with the manuscript's statement ("after those results").
5. **§4.5, bootstrap CIs.** CIs that resample five seeds understate between-seed uncertainty. Say so, or use t-based CIs over seeds for the headline numbers.
6. **Supplement S11 title.** "Operating point with all actions on UNSW-NB15" also repeats all TON-IoT rows, with unadjusted p-values that differ from the Holm values in Table 7 (for example MLP AT-threat 0.020 vs 0.122). Drop the TON-IoT rows or label them.
7. **"Seeds vary initialisation… but not the data".** All p-values are therefore conditional on one training set. Say explicitly that the tests address training randomness only, and that the resplits (one seed, Table S5) are the only check on data variability.
8. **§3, budget floors.** The floors (1,000 bytes, 10 packets, 1 s) drive the TON-IoT result. Give a short operational justification (for example, what 500 bytes and 5 extra packets mean for a SYN scan, and whether payload on SYN packets is itself anomalous), or present floors × 0.5 alongside × 1 in the main text.
9. **Abstract, "the trees were harder to evade only while the TTL features were in the model".** Without TTL features the RF is still below both networks (28.4% vs 36.1%/41.9%), and no test is reported. Write "the gap largely closed".
10. **Abstract.** It has 17 numbers. Cut it to the four or five that carry the conclusions, and give one sentence of practical recommendation.
11. **§4.2.** Report the GBDT's number of iterations after early stopping and the RF's out-of-bag score, or similar, so readers can judge model capacity, which matters for the network-tree comparison.
12. **Figure 2.** State in the caption that the curves are at each model's own threshold, since §5.3 argues that this view is misleading (raised in round 3; still unclear in the build captions).
13. **Highlight 5** is exactly 85 characters. Fine, but any edit will exceed the limit.
14. **Data availability.** The models are "available on request". JISA encourages deposit; a Zenodo deposit of 1.5 GB is feasible and would make the SHA-256 list meaningful.
15. **§5.1, "Queries".** The 1,769 and 7,243 queries are upper bounds for the union. Also give the median number of queries to first success for the adaptive search, which is the realistic score-based attacker.

---

## Strengths

- The threat model is precise and implemented faithfully. `Space.selected` matches Table 2 and §3 in every rule I checked: header bytes per added packet, the ≥ 2-packet delay rule after added packets, ACK delay only with a handshake, the sload/dload rescaling and the ct_state_ttl lookup. The zero-action mapping is checked against the pipeline.
- Success is counted correctly (detected attacks turned benign), and end-to-end evasion is also reported, which avoids the usual ASR denominator trap.
- The attacks are strong and checked: saturation grids, adaptive search, strong and BPDA PGD (with new evasions ≤ 0.1%), feasibility with integer packets and the MTU bound, and seed-disjoint evaluation grids.
- The feature-space reference was strengthened honestly. The consistency-relaxed attack cleanly separates magnitude from dependency effects, which is a useful methodological contribution.
- The decision-rule confound was removed, a clean-loop control is included, and the paper reports results that go against its earlier claims (TRADES worse; AT-all ε = 0.1 not holding across seeds).
- The TTL artefact is handled thoroughly: a sweep, unseen values, TTL-free detectors and separate reporting.
- Tests against held-out actions and values show the limits of augmentation instead of hiding them.
- Reproducibility is strong: every number is computed from result files, and the code and run order are released.

## What would most raise the score

1. Retrain AT-threat with a corner-reaching inner search, or the augmentation's grid, on five seeds (Major 1).
2. Replace the single matched point with an attack-aware ROC or partial AUC, and report the network null result as a CI (Major 2).
3. Run the destination-response sensitivity check and publish the selected-feature table (Major 3).
4. Cut to about 10k words, with one rule and one metric for all headline numbers (Majors 4 and 5).
