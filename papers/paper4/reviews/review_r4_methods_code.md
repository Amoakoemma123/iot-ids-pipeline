# Paper 4 v2 (Federated CNN-IDS): methods and code audit, round 4

Date: 2026-10-26.

## Scope

- `Papers pipeline/experiments/paper4_federated/run_fl_v2.py` (RF) at HEAD 52bbca3, and its full diff since 8843fb7 (round 3).
- `prepare_nbaiot.py` (NB) and `src/data/loader.py` (LD).
- The queue scripts `logs/p4r5_*.sh`.
- Every result JSON in `results_v2/{TON_IoT,UNSW_NB15,N_BaIoT}`.
- `Revision_v2/Paper4/compute_values.py` (CV) and `verify.py`.
- `Manuscript_filled.md` (P) and `Supplement_filled.md` (S).

## How it was checked

- I read the code and the diff. Since ab89b53, the training, aggregation, DP and evaluation code is unchanged. The changes are new groups, grids, N-BaIoT loading, the device partition, `local_finetune` and the stale-cache guard.
- I imported `experiments()` from RF (CPU, no training) and compared its generated config with `cfg` in every stored JSON:
  - TON 414 files, UNSW 274, N-BaIoT 110.
  - **All match.**
- I recomputed every tuning selection from the `tune3_*` JSONs.
- I recomputed ε for the new DP runs (`dp_full_sigma1.8` seed 2, `dpt_C4`).
- I recomputed the own-traffic, per-device, fine-tuning, BN re-estimation and trend statistics from the JSONs.
- I hashed the model inputs of all three datasets to look for duplicates across the split.
- I opened one raw N-BaIoT file.
- Scripts are in the scratchpad (`audit/c1`–`c9`). No project file other than this report was written. No model was trained.

Severity: **critical** = a reported result or headline claim is invalid. **major** = a result is biased, or a method or claim is misdescribed in a way a referee would reject. **minor** = needs a fix or a disclosure.

## Summary

All four round-3 major issues are resolved in substance:
- The host partition is now truly redrawn (`randomized=True`).
- "Own traffic" is now measured on each site's own test hosts or devices.
- FedLC is described.
- Non-significant effects are phrased as estimates.

The training algorithms, DP-SGD, the RDP accounting and federated BN re-estimation are implemented correctly.

There is no critical bug, but there are four new major issues:
1. N-BaIoT's model input is heavily duplicated across the split.
2. N-BaIoT tuning did not run on the partition the paper states.
3. The "device partitions cost more" headline rests on two non-significant tests.
4. The own-traffic reversal is open to a memorisation explanation that the random record split cannot rule out.

---

## Critical

None.

---

## Major

### M1. N-BaIoT: deduplication is done in float64 before scaling, but the model sees float32, so 29% of test records are exact copies of training records

**Where:** NB:55-59 deduplicates on the raw float64 values. NB:75 then min–max scales and casts to float32. There is no cross-split guard (LD:206-227 has one, NB does not).

**Evidence (`audit/c5_nb.py`, `c6_nbdup.py`, `c7_raw.py`):**
- Only 107,574 of the 148,024 training inputs are unique.
- 9,280 of 31,720 test records (29%) have a training record with an identical input, always with the same label:
  - 8,094 of them have the twin on the **same device**;
  - 1,186 have it only on other devices.
  - The validation set is affected the same way: 9,224 of 31,719 records.
- By class, the share of test records with a training twin is:
  - gafgyt_tcp 99.9% and gafgyt_udp 99.9%;
  - gafgyt combo, junk and scan about 15%;
  - Mirai classes about 3%;
  - benign 0%.
- Cause: in `Danmini_Doorbell/gafgyt_attacks/tcp.csv`, 85,227 distinct float64 rows collapse to **94** distinct rows in float32.
- So the S1-A statement "remove 185 exact duplicates" is true only in a representation the model never sees.

**Effect:**
- Absolute N-BaIoT scores are inflated: two of the 11 classes are pure lookup.
- Macro-F1 differences between methods are compressed, because two classes sit at the ceiling for every method that saw that device's data.
- Validation selection is also partly on copies of training records.
- Same-device twins help the local model and FedAvg equally. The 1,186 cross-device twins favour FedAvg, so the own-traffic reversal on N-BaIoT is, if anything, understated.
- Direction of the main comparisons: probably unchanged. Magnitudes: not trustworthy as stated.

**Fix:**
- In `prepare_nbaiot.py`, cast to float32 (after scaling) before deduplicating, and add the LD step-10 cross-split guard.
- Re-run the `device`, `finetune` and N-BaIoT `central` groups. This is GPU-cheap; tuning could be kept and disclosed.
- At minimum, disclose the duplicate share by class and correct S1-A.

### M2. N-BaIoT hyperparameters were tuned on a Dirichlet α = 0.5 partition with 4 clients, not on the device partition as stated

**Where:** P:56 says "(N-BaIoT: on the device partition)". The code is RF:669-675: `tune3a/b/lc` call `base_cfg(algo=..)` without `partition='device'`, so they get the defaults alpha = 0.5 and n_clients = 4.

**Evidence:** every `results_v2/N_BaIoT/tune3_*/seed10x.json` has `cfg.alpha = 0.5`, `n_clients = 4` and no partition field (`audit/c2_tune.py`).

**Effect:**
- The N-BaIoT algorithm comparison (Table III, S1 family) uses hyperparameters tuned on 4 Dirichlet clients but is evaluated on 9 device clients. The Methods text is wrong.
- The N-BaIoT margins are within noise anyway: FedProx 1e-4 vs 1e-3 is 0.8704 vs 0.8703, and FedLC τ = 30 vs 3 is 0.8710 vs 0.8701. So the choices are probably immaterial, but the description must match.

**Fix:** correct P:56 (and S1-D) to "at α = 0.5 with four clients on every dataset". Or re-tune on `partition='device'` and re-run `device`.

### M3. The "device partitions cost more than Dirichlet at matched skew" headline rests on two non-significant, inconsistently classified tests; the trend p-value is overstated

**Where:**
- The claim appears in the abstract (P:9), contribution 3 (P:21), §V-C (P:97) and the conclusion (P:159).
- The tests are CV:115 (TON `fedavg_natr` vs `fedavg_a0.5`, EXPLORATORY) and CV:546 (N-BaIoT `fedavg_dev` vs `fedavg_a0.5_N9`, in the PRIMARY family).

**Evidence:**
- TON: −0.120 [−0.257, +0.017], uncorrected p = 0.071.
- N-BaIoT: −0.024 [−0.059, +0.012], p = 0.138, Holm p = 0.553.
- §V-C concedes "Neither difference … is significant on its own". The abstract and conclusion still state it as a finding.
- The same kind of comparison is primary on one dataset and exploratory on the other.

**Further statistical issues:**
- **Own-traffic reversal.** The abstract headline is exploratory (TON p = 0.015 uncorrected; N-BaIoT p = 0.147). The abstract says neither.
- **Linear trend (CV:321-328).** `linregress` runs over 20 pooled points: 4 skew levels × 5 repetitions, where all four points of a repetition share one central run and one seed. That gives df = 18 and "p < 0.001". At the repetition level (one slope per repetition, one-sample t over 5 slopes) the evidence is weaker but still present:
  - TON: slopes −0.066, −0.084, −0.026, −0.026, −0.029, p = 0.019.
  - UNSW: p = 0.030.
  - The equal spacing of IID, α = 1, 0.5 and 0.1 as 0–3 is also arbitrary.

**Fix:**
- Phrase the partition-cost claim as a consistent but non-significant direction everywhere, including the abstract and conclusion. Put both partition comparisons in the same class (both primary or both exploratory).
- Mark the own-traffic results as exploratory in the abstract.
- Report the repetition-level trend test.

### M4. The own-traffic reversal may be memorisation within a random record split, which the paper's mechanism sentence does not consider

**Where:** P:126 says "a site's own attack traffic differs from the same classes elsewhere, and its local model fits it better than a model averaged over all sites". Also P:9 and P:153.

**Issue:**
- The 70/15/15 split is random at the record level (LD:161-173). A host's test records are therefore drawn from the same captures, sessions and port/service combinations as its training records.
- On TON-IoT the features include `src_port` and `dst_port`. The paper itself says these act as testbed shortcuts.
- A local model trained 100 epochs on a quarter of the data can memorise host-specific port and flow signatures. Those signatures recur in that host's test records. In a temporally separated deployment they would recur much less.
- The same holds for N-BaIoT, plus M1.

**What is not measured:**
- The reversal may be an artefact of evaluating on interleaved records from the same capture, not a property of "a site's own traffic".
- "Federation buys detection of attacks seen elsewhere" is inferred, not measured. The code can score each model on *other* sites' test hosts: the complement of `site_te[k]`. It does not.

**Fix (cheap, no new training):**
- Remove the sentence explaining the mechanism, or qualify it as "under a random record split".
- Name the threat in the Limitations.
- Optional re-run: `local_natr` and `fedavg_natr` without ports (the `portsalg` machinery already exists). Add an "other sites' traffic" score to `site_scores`.

---

## Minor

1. **Fine-tuning is compared with FedAvg from a different run.**
   - P:128 compares the fine-tuned common-test score (0.551 TON, 0.779 N-BaIoT) with `fedavg_natr`/`fedavg_dev` (0.720 / 0.820). Those are separate runs of the same config, and their run-to-run differences reach 0.084 (S1-C).
   - `fedavg_natr_ft` stores its own pre-fine-tuning model: common test 0.700 / 0.809, and own-traffic gap −0.138 (19 of 20 pairs) / −0.043.
   - Within-run, one epoch costs −0.149 ± 0.037 (TON) and −0.031 ± 0.013 (N-BaIoT) on the common test set. Its own-traffic gain over the same-run FedAvg is +0.127 and +0.033.
   - The conclusions hold. Report the within-run numbers (CV:604-626).
2. **The Holm family as defined in the text (P:56) is not the family in the code (CV:105-111, 544-546).**
   - "FedAvg against the centralised … at every partition" omits α = 1.0 (CV:112 makes it exploratory).
   - On N-BaIoT the family also contains the two device-vs-Dirichlet tests (11 = 9 + 2), which the definition does not mention.
   - "Fixed before the analyses reported here" should be softened: the family changed from 29/22 tests in round 3 to 19/13/11 now.
3. **The full-budget DP comparator is "without clipping or noise", not "without noise" (P:136).** `dp_full_nonoise` has `clip = 0`. This is round-3 minor 4, still open in the main text.
4. **The BN swing is reported twice with different values.**
   - Main text: 0.110, server-averaged, from the five `fedavg_bn_a0.5_fedrecal` runs.
   - S6: 0.151, from the three `fedavg_bn_a0.5_recal` runs.
   - Both are correct for their runs. Say which run set each comes from.
5. **The FedLC description (P:60) omits details of the code:**
   - classes absent from a client are clamped to n = 1, which gives the largest offset τ (up to 30 on N-BaIoT);
   - evaluation uses the uncalibrated softmax;
   - the logits are log(p + 1e-8), so gradients vanish for p ≲ 1e-8.
   - Add one clause.
6. **UNSW-NB15 model-input duplicates (pipeline level).** After the companion pipeline's feature selection to 10 features:
   - 2,586 of 15,894 UNSW test records (16%) have an identical training input, 412 of them with a different label;
   - only 64,546 of 74,167 training inputs are unique;
   - TON-IoT has only 35 such records.
   - P:38 ("exact duplicates are removed after all column drops") is true before feature selection only. This affects every method equally; disclose it.
7. **Release hygiene (round-3 minor 8, partly open).**
   - The `tune` group, `TUNE_GRID`, `finalize_tuning` and the docstring usage `--group tune` are still in RF. Re-running them would reproduce the round-2 C1 bug.
   - `tune_*` directories still sit in `results_v2/<ds>/`.
   - `prepare_rerun` and CV:168 still read `tuned.json`. CV:478-485 overwrites all six values from `tuned_v3.json`, so the numbers are correct.
8. **The stale-cache guard compares `cfg` only (RF:905-912), not the code version.**
   - It is valid for the current results: the training, evaluation and DP code is unchanged since ab89b53, and the natural/site code predates every natr run.
   - It would silently keep stale results after a code change. Store the git hash in each JSON.
9. **Still open from earlier rounds:**
   - there is no schema assertion when `pipeline.pkl` is loaded (RF:122);
   - in the TON DP JSONs, `cfg.norm = 'bn'` is cosmetic, because RF:471 forces GN.
10. **The per-device statistic is fine, but its unit should be stated in the main text.**
    - On N-BaIoT the unit is the device: n = 9, 5-repetition means, p = 0.147. On TON it is the repetition (n = 5, p = 0.015).
    - Both choices are defensible: the N-BaIoT partition is fixed, and the TON sites are redrawn. At the repetition level N-BaIoT would give p = 0.025, so the paper chose the conservative unit.
    - P:126 gives the N-BaIoT p-value without saying the unit.
11. **Wording (P:9, P:128): "gives up much of the gain on other sites' attacks"** is measured on the common test set, which also contains the site's own traffic. Say "on the common test set", or measure other sites' traffic (see M4).
12. **`verify.py:205` compares p-values as strings** (`'0.128' > '0.05'`). It is correct for the current values but fails (IndexError) if the p-value is printed as "p < 0.001".

---

## Verified correct (no action)

- **Config conformance.** All 798 stored configs equal what `experiments()` generates now.
  - FedProx, FedAdam and FedLC tuning and core runs use the tuned client lr, 0.003, on all three datasets.
  - Tuned values: TON μ 1e-5, η_s 0.03, τ 3, SCAFFOLD 0.1, FedAvg-SGD 0.3; UNSW 1e-3, 0.03, 10, 0.03, 0.03; N-BaIoT 1e-4, 0.03, 30, 0.1, 0.1.
  - Every value reproduces from the `tune3_*` means: three seeds on TON/UNSW, two on N-BaIoT, as S1-D states.
  - The edge flags match the S1-D sentence: only TON μ is at an edge.
  - The FedLC τ extension ran before the dependent groups were re-run (`p4r5_lcext.sh`), and the stale guard moved the superseded FedProx/FedLC files to `_stale/`.
- **No test leakage into any choice.**
  - `select` uses validation F1 only.
  - `finalize_tuning3` reads `val_f1_selected`.
  - The centralised lr check reads validation only.
  - Fine-tuning epochs are fixed ({1, 2, 5}) and nothing is selected on them.
  - The BN re-estimation round is selected on `val_f1_fedrecal` (CV:303).
- **Redrawn host partition (RF:144-160).**
  - Random order with greedy smallest-client assignment, as S1-B describes.
  - Sizes differ across repetitions, from 23,741 to 52,783.
- **Own-traffic comparison is fair in construction.**
  - `local_natr`, `fedavg_natr` and `fedavg_natr_ft` have identical `client_sizes`, class counts and per-site test sizes in every repetition. They share `prng = default_rng(1000 + rep)` and the same calls.
  - The same holds for the N-BaIoT device runs.
  - The site test set is the test records to the site's hosts or device; the 46 TON test records to unseen hosts belong to no site.
  - The local and federated models are scored with the same `site_scores`, at their pooled-validation-selected round. This is disclosed, and it disadvantages the local model on its own traffic.
  - The client index in `per_client[j]` matches `site_selected[j]`.
- **Own-traffic statistics reproduce.**
  - TON: −0.121 ± 0.083, p = 0.015, 17 of 20 pairs.
  - N-BaIoT, per device: −0.032 ± 0.046, p = 0.147, 6 of 9 devices.
  - Fine-tuning: TON −0.011 ± 0.049, N-BaIoT −0.010 ± 0.017.
  - Common-test scores after fine-tuning: 0.551 and 0.779.
  - The "Own-host" column of Table II coincides with the final-round column for FedAvg and FedProx to three decimals. I checked per repetition: it is a coincidence, and `site_selected` is computed on the selected model.
- **Local fine-tuning (RF:568-583).**
  - Each site starts from the validation-selected global weights.
  - Adam is reset, with the same lr and fixed batches; dropout is active.
  - Epochs are cumulative.
  - The global model is untouched, so the final-round scores are unaffected.
- **Federated BN re-estimation (RF:406-434).**
  - Layer by layer: the probe uses the already re-estimated statistics of earlier layers.
  - Per-channel mean and mean square are taken over up to 8,192 training records per client, with n_k weights.
  - var = E[x²] − E[x]², which is the correct pooled (law of total variance) formula.
  - It uses only training records, scores, then restores the statistics. Training-mode BN does not use the running statistics, so training is unaffected.
  - Numbers reproduce: TON α = 0.5 swing 0.110 → 0.018, final 0.634 → 0.855; UNSW 0.010 / 0.007.
- **Algorithms (unchanged since round 3, still correct):**
  - FedProx: the anchor is the global model, with local Adam.
  - FedAdam: Reddi et al. without bias correction, τ = 1e-3, trainable weights only.
  - SCAFFOLD: option II, K = actual local steps, uniform c averaging (disclosed).
  - FedLC: calibrated logits.
  - On N-BaIoT, SCAFFOLD divides by N = 9.
- **DP-SGD.**
  - Per-example global-norm clipping.
  - Noise σC on the sum, divided by B = q·n_k.
  - Poisson sampling with q = B/n_k and ceil(1/q) draws per epoch; the accountant counts R·E·ceil(1/q) steps (RDP, δ = 1e-5).
  - GN model with dropout 0; ε is the maximum over clients.
  - New ε values reproduce exactly: `dp_full_sigma1.8` seeds 0–2 give 2.806, 3.029 and 2.950 (mean 2.9); `dpt_C4` gives 2.987 and 3.235.
  - The three full-budget repetitions give 0.683 (DP) against 0.856 (no clip, no noise).
- **Statistics.**
  - `paired` is a paired t-test over shared repetitions with a t-based CI.
  - The Holm step-down is correct (running max, capped at 1).
  - The family sizes are 19 / 13 / 11, as stated.
- **N-BaIoT preparation, apart from M1.**
  - Bernoulli 3% sample per device–class file.
  - Split stratified by device × class.
  - Zero-variance drop and min–max fitted on training only (no feature was dropped).
  - Test values above the training max are kept (max 1.13).
  - Counts match S1-A.

---

## Round-3 findings: status

| R3 | Finding | Status |
|---|---|---|
| M1 | Natural partition effectively fixed; text wrong | **Fixed.** `natural_r` (random host order) is the reported partition. The fixed-order run is kept as a contrast (S4: 0.770 ± 0.031 against 0.720 ± 0.099). The text describes the procedure correctly. |
| M2 | "Own traffic" was the pooled test set restricted to the site's classes | **Fixed.** Scored on the site's own test hosts or devices. The restricted-class number is kept as a contrast. **New:** interpretation threat (M4). |
| M3 | Headline claims not Holm-significant; false §VI-A sentence | **Mostly fixed.** Estimates are phrased with CIs, and the false sentence is gone. **New:** partition-cost headline, abstract own-traffic and trend p-value (M3 here). |
| M4 | FedLC undocumented | **Fixed** (P:60, S1-D grids and selected values; UNSW edge resolved by widening the grid to 100). Clamp detail missing (minor 5). |
| m1 | Run-to-run noise | **Fixed** (S1-C: up to 0.084). Not yet used for the fine-tuning comparison (minor 1). |
| m2 | Holm family padded | **Fixed** (primary family defined). New text–code mismatch (minor 2). |
| m3 | "Disjoint partitions" wording | **Fixed** (S1-D). |
| m4 | "Without clipping or noise" | **Partly fixed.** S7 is correct; P:136 is not (minor 3). |
| m5 | Centralised Adam state and lr | **Fixed** (S1-C; lr check in S1-D: 0.003 best on both datasets). |
| m6 | DP empty draw, PRNG, flow-level unit | **Fixed** (S1-E, P:62). |
| m7 | BN swing range | **Fixed** (S6: rounds 11–50). Two different values remain (minor 4). |
| m8 | Release hygiene | **Partly open** (minor 7). |
| m9 | `pipeline.pkl` assertion; DP `norm` field | **Open** (minor 9). Worst-client view: addressed by the per-site and per-device tables. |

## Re-runs

- **Recommended, M1:** rebuild N-BaIoT with float32 deduplication and a cross-split guard, then re-run `central`, `device` and `finetune` for N-BaIoT (GPU, 5 repetitions).
- **Optional:**
  - M2: re-tune N-BaIoT on the device partition. Or just fix the text.
  - M4: `local_natr` and `fedavg_natr` without ports, with an "other sites' traffic" score.
- **Text only:** M2 (if no re-tune), M3, M4 (wording) and minors 1–5, 10, 11.
