# Paper 5 v2: independent number check

Checked on 2026-10-24 against `Manuscript_filled.md` and `Highlights.docx`. I did not use `compute_values.py` or `values.json` for any recomputation. I opened `compute_values.py` once, after my own count of conflicting-label records did not reproduce 4,845, to see what the paper's number actually counts (item M2).

Scripts are in `E:\Research_Works\IDS_IOT_Research\_scratch\review\`:
- `data_checks.py`: raw UNSW-NB15 and TON-IoT CSVs, plus a rebuild of the original split
- `conflict_variants.py`: alternative definitions of the conflicting-label count
- `json_checks.py`: every results_v2 JSON and the baseline metrics, with bootstrap CIs recomputed from the per-instance rows
- Outputs: `json_out.txt`, `data_out.json`

**Method**
- **Data.** I applied the original loader's cleaning steps: drop the binary `label`, strip and lowercase strings, then drop exact duplicates. The duplicate check excludes `id`.
- **Original split.** I rebuilt it from the git snapshot `7183c0e:src/data/loader.py`. With `id` present the deduplication removes 0 rows and no labels are NaN, so the rows keep their file order. I then reproduced the split as `train_test_split` with test_size 0.15 and stratification on `y`, followed by 0.15/0.85 for validation, seed 42, using LabelEncoder on lowercase `attack_cat`. The resulting sizes are 122,738 / 26,301 / 26,302.
- **Bootstrap CIs.** Percentile bootstrap, 2,000 resamples, seed 42, rebuilt from `rows` in local.json and stability.json and from `k_hist` in counterfactual.json. I also ran a second seed to gauge Monte Carlo noise.

**Totals:** about 410 numbers or claims checked: 304 table cells, 90 text items, and repeated occurrences in the abstract, highlights and conclusion, with 6 mismatches or unsupported claims (3 important, 3 minor) and 3 minor wording notes.

---

## 1. Data audit (Sections 1, 3.1, 3.3, 5.5, abstract, highlights)

| # | Item | Paper | Recomputed | Status |
|---|---|---|---|---|
| 1 | TON-IoT records | 211,043 | 211,043 | OK |
| 2 | UNSW-NB15 training-file records | 175,341 | 175,341 | OK |
| 3 | Duplicates removed with `id` present | "removed nothing" | 0 | OK |
| 4 | `id` equals row position | sequential row number | id == position+1 for all rows | OK |
| 5 | Exact duplicates excluding `id` | 67,601 | 67,601 | OK |
| 6 | Duplicate share | 38.6% | 38.554% | OK |
| 7 | Rows after deduplication | 107,740 | 107,740 | OK |
| 8 | *generic* before → after | 40,000 → 4,181 | 40,000 → 4,181 | OK |
| 9 | Leading records that are all benign | first 47,911 | first non-normal row is index 47,911, so 47,911 benign rows | OK |
| 10 | *generic* only after X% of file | "after the first 67%" | first generic at row 117,204 = **66.84%** | MINOR (see m1) |
| 11 | Records with identical features and a different label | 4,845 | **8,472** in the deduplicated data (1,772 feature vectors); 30,528 before deduplication | **MISMATCH** (M2) |
| 12 | Original test records with an exact copy (features + label) in training | 43.2% | 43.17% (44.09% counting train+val; 46.13% on features only) | OK |
| 13 | Corrected UNSW test size | 16,161 | 0.15 × 107,740 = 16,161; performance.json gives 16,161 | OK |
| 14 | `sttl` top values | 254, 31, 62 | 254 (52,038), 31 (39,123), 62 (15,753) | OK |
| 15 | Share of the top three values | 99.2% | 99.23% | OK |
| 16 | `sttl`=31 count | 39,123 | 39,123 | OK |
| 17 | `sttl`=31 attacks | "none of them attacks" | 0 (all normal) | OK |
| 18 | Binary accuracy from `sttl` majority label | 88.8% | 88.76% (in-sample, deduplicated data) | OK |
| 19 | 10-class accuracy from `sttl` | 62.9% | 62.86% | OK |
| 20 | Majority-class baseline | 48.2% | 48.16% (normal). The binary majority baseline, not quoted in the paper, is 51.8% | OK |
| 21 | "UNSW-NB15 retains no port feature after selection" | no port feature | true for the 11 selected features. However, the training CSV has **no port-number columns at all** (no `sport`/`dsport`); only the count features `ct_src_dport_ltm`, `ct_dst_sport_ltm` and `is_sm_ips_ports` exist | MINOR wording (m3) |
| 22 | The file "is ordered in blocks" | qualitative | normal at rows 0–66.1%, attacks from 27.3% onward, generic from 66.8% onward | OK |
| 23 | Intro: `sttl` alone separates benign/attack for 88.8% | 88.8% | 88.76% | OK |
| 24 | Abstract/intro/6.1/highlight: 38.6%, 43.2% | as above | as above | OK (4 occurrences) |

## 2. Table 1 and Section 3.3 text (baseline metrics, original backup vs current)

| Model | Orig F1 | Corr F1 | Orig acc | Corr acc | Status |
|---|---|---|---|---|---|
| RF | 0.6375 | 0.5913 | 0.8212 | 0.7907 | OK (4/4) |
| 1D-CNN | 0.5182 | 0.5111 | 0.7712 | 0.7326 | OK (4/4) |
| AE+RF | 0.5354 | 0.5040 | 0.8034 | 0.7392 | OK (4/4) |
| LSTM | 0.4600 | 0.4509 | 0.7372 | 0.7083 | OK (4/4) |
| LR | 0.3644 | 0.3691 | 0.6297 | 0.6302 | OK (4/4) |

| # | Claim | Recomputed | Status |
|---|---|---|---|
| 25 | "accuracy falls for every model except logistic regression, which barely changes" | the four others fall by 2.9–6.4 points; LR +0.05 points | OK |
| 26 | TON-IoT results unchanged | all five TON metrics.json entries are identical between backup and current. I did not rebuild the TON data itself, so "exactly the same data" is supported only through identical metrics | OK |
| 27 | RF 0.6375 → 0.5913 (abstract, contribution 1, conclusion) | matches | OK (3 occurrences) |

## 3. Section 3.4, methods settings, original-analysis facts

| # | Item | Paper | Recomputed / source | Status |
|---|---|---|---|---|
| 28 | RF: 200 trees, depth 20, leaf 2, √p | as stated | `src/models/random_forest.py` | OK |
| 29 | CNN architecture, about 24k parameters | as stated | `cnn.py`; my parameter count is 23,946 for 10 classes, independent of input length | OK |
| 30 | Early-stopping patience 15 | 15 | `CNN_TRAIN patience=15` | OK |
| 31 | Features TON / UNSW | 25 / 11 | 25 / 11 | OK |
| 32 | Original UNSW set had 16 features including `id` | 16 | 16 in v1 shap_results.json | OK |
| 33 | v1: SHAP ranked `id` first for both models | first | RF 0.0484 (next 0.0258); CNN 0.0614 (next 0.0466) | OK |
| 34 | Test macro-F1 RF/CNN, TON | 0.9565 / 0.8803 | 0.9565 / 0.8803 | OK |
| 35 | Test macro-F1 RF/CNN, UNSW | 0.5913 / 0.5111 | 0.5913 / 0.5111 | OK |
| 36 | Test records | 27,533 / 16,161 | 27,533 / 16,161 | OK |
| 37 | Nine attack classes plus benign (both datasets) | 10 classes | 10 classes in performance.json | OK |
| 38 | CF fields TON / UNSW | 18 / 11 | 18 / 11 | OK |
| 39 | Global SHAP: 1,000 records, 500 rank bootstraps, 100 interaction records, 200-record background | as stated | run_xai_v2.py; `n` = 1000 in JSON | OK |
| 40 | Agreement sample of up to 30 per class | as stated | 300 TON rows (30 × 10); 289 UNSW rows (worms 19) | OK |
| 41 | Stability: 10 per class, 20 runs | as stated | 100 rows per model | OK |
| 42 | CF: 400 candidates, k ≤ 4, up to 50 per class | as stated | n = 50, except UNSW worms (RF 14, CNN 12) | OK |
| 43 | Bootstrap 2,000 resamples, seed 42 | as stated | N_BOOT=2000; my recomputation reproduces the stored CIs exactly | OK |
| 44 | SHAP 0.49 | 0.49 | shap-0.49.1 installed | OK |
| 45 | v1 pooled comparison over 12 records | 12 | 12 LIME records (4 classes × 3) | OK |
| 46 | v1 pooled RF TON Jaccard | 0.67 | 0.6667 | OK (abstract and 5.2) |
| 47 | "SHAP for these models is deterministic given its background" (4.4); "it is deterministic" (6.2) | deterministic | true for interventional TreeSHAP. **False for the CNN**: GradientExplainer draws a random reference and interpolation point per sample (`rseed=None` leads to `np.random.randint`), and run_xai_v2.py does not seed it | **UNSUPPORTED** (M3) |
| 48 | Hardening evaluated "on the same held-out test attacks" (4.7) | same attacks | `evasion()` draws from the attacks *each model itself* detects correctly (same RNG seed), so the two query sets can differ | MINOR (m2) |

## 4. Table 2 (TON-IoT global SHAP)

All 60 cells (feature, mean |SHAP|, rank interval × 10 ranks × 2 models) match global_shap.json exactly. RF: dst_port 0.0351 1–1 … conn_state_s0 0.0107 10–10. CNN: dns_RA_f 0.0853 1–2 … proto_tcp 0.0377 9–10. **OK (60/60).**

| # | Section 5.1 text | Paper | Recomputed | Status |
|---|---|---|---|---|
| 49 | RF top 3 with single-rank intervals | dst_port, src_ip_bytes, conn_state_rej; 1–1, 2–2, 3–3 | same | OK |
| 50 | CNN top 3 | dns_RA_f, dst_port, conn_state_rej | same | OK |
| 51 | Cross-model Jaccard top-5 / top-10 | 0.25 / 0.25 | 0.25 / 0.25 (recomputed from rankings) | OK |
| 52 | Strongest RF interactions | dst_port×duration, src_ip_bytes×conn_state_rej, dst_port×conn_state_rej | same order from the matrix (0.00668, 0.00553, 0.00473) | OK |
| 53 | "Port features prominent for both" | qualitative | dst_port is RF #1 and CNN #2; src_port is RF #5 and CNN #8 | OK |

The interaction values use `tree_path_dependent` TreeSHAP, not the interventional variant named in 4.1. The sentence makes no claim about which, so this is only a note.

## 5. Table 3 (UNSW-NB15 global SHAP)

All 60 cells match. **OK (60/60).**

| # | Text | Paper | Recomputed | Status |
|---|---|---|---|---|
| 54 | RF top 3 / CNN top 3 | sttl, ct_state_ttl, sbytes / sttl, ct_state_ttl, smean | same | OK |
| 55 | Cross-model top-5 Jaccard | 0.43 | 0.4286 (top-10 0.82) | OK |
| 56 | Both models rank `sttl` first | yes | yes; RF interval 1–1, CNN 1–1 | OK |

## 6. Tables 4 and 5 (per-instance measures)

Every mean and CI was recomputed from `rows`. With seed 42 my values equal the stored ones to four decimals. A second seed moves the CI bounds by at most 0.003.

| Measure | TON RF | TON CNN | UNSW RF | UNSW CNN |
|---|---|---|---|---|
| Jaccard@5 | 0.39 [0.37,0.41] OK | 0.29 [0.27,0.31] OK | 0.42 [0.39,0.44] OK | 0.37 [0.35,0.39] OK |
| Spearman ρ | 0.58 [0.56,0.59] OK | 0.45 [0.42,0.47] OK | 0.26 [0.22,0.30] OK | 0.17 [0.13,0.21] OK |
| LIME R² | 0.29 [0.28,0.31] OK | 0.06 [0.05,0.07] OK | 0.31 [0.29,0.33] OK | 0.08 [0.07,0.10] OK |
| Self-agreement | 0.70 [0.67,0.72] OK | 0.46 [0.43,0.50] OK | 0.80 [0.78,0.83] OK | 0.58 [0.56,0.60] OK |
| % ≥ 0.75 | 32 OK | 5 OK | 68 OK | 7 OK |
| del SHAP k=1 | 0.53 [0.49,0.56] OK | 0.54 [0.49,0.58] OK | 0.42 [0.39,0.44] OK | 0.48 [0.44,0.52] OK |
| del LIME k=1 | 0.40 [0.36,0.43] OK | 0.39 [0.34,0.44] OK | 0.26 [0.24,0.29] OK | 0.27 [0.23,0.30] OK |
| del rand k=1 | 0.11 [0.10,0.12] OK | 0.15 [0.13,0.17] OK | 0.17 [0.16,0.18] OK | 0.21 [0.19,0.23] OK |
| del SHAP k=3 | 0.76 [0.73,0.78] OK | 0.74 [0.70,0.78] OK | 0.59 [0.55,0.62] OK | 0.60 [0.56,0.63] OK |
| del LIME k=3 | 0.72 [0.69,0.74] OK | 0.65 [0.60,0.69] OK | 0.44 [0.41,0.48] OK | 0.43 [0.39,0.47] OK |
| del rand k=3 | 0.28 [0.27,0.29] OK | 0.42 [0.39,0.44] OK | 0.38 [0.36,0.41] OK | 0.41 [0.38,0.43] OK |
| del SHAP k=5 | 0.84 [0.82,0.86] OK | 0.85 [0.83,0.88] OK | 0.60 [0.57,0.64] OK | 0.62 [0.59,0.66] OK |
| del LIME k=5 | 0.79 [0.77,0.82] OK | 0.79 [0.75,0.82] OK | 0.55 [0.51,0.58] OK | 0.53 [0.49,0.57] OK |
| del rand k=5 | 0.42 [0.40,0.44] OK | 0.59 [0.56,0.62] OK | 0.49 [0.46,0.51] OK | 0.49 [0.46,0.52] OK |

**56/56 cells OK.**

### Section 5.2 text, abstract, highlights, conclusion

| # | Claim | Paper | Recomputed | Status |
|---|---|---|---|---|
| 57 | "SHAP and LIME share on average **fewer than half of their top-5 features**" (5.2). Also in the conclusion ("agreed on fewer than half of the top-5 features") and Highlight 3 ("share under half of their top-5 features") | < 2.5 of 5 | A top-5 Jaccard J means 10J/(1+J) shared features. Mean shared features per instance: **TON RF 2.70, TON CNN 2.13, UNSW RF 2.82, UNSW CNN 2.60**. Instances sharing ≥3 of 5: 65%, 34%, 62%, 56%. Only TON CNN shares fewer than half | **MISMATCH** (M1, 3 occurrences) |
| 58 | "SHAP and LIME usually disagree on most of the top-5 features" (6.2) | most instances share ≤2 | only TON CNN (66% share ≤2); for the other three, most instances share ≥3 | **MISMATCH** (M1) |
| 59 | Jaccard per class, TON RF: min injection 0.19, max password 0.57 | 0.19 / 0.57 | 0.19 / 0.57 | OK |
| 60 | In-text Jaccard and stability values and CIs (all repeated from the tables) | as tables | match | OK (14 values) |
| 61 | "SHAP leads LIME at every k for both models and both datasets" | 12/12 | 12/12. Paired bootstrap of SHAP−LIME is >0 in all 12; the smallest gap is TON RF k=3, 0.041 [0.019, 0.063] | OK |
| 62 | LIME barely above random at k=3, UNSW CNN | 0.43 vs 0.41 | paired LIME−random 0.023 [−0.012, 0.056], not significant | OK |
| 63 | LIME weakest for the UNSW CNN | qualitative | LIME−random gap at k=3: UNSW CNN 0.023, below UNSW RF 0.060 and TON RF/CNN 0.44/0.23 | OK |
| 64 | CNN random baseline is high | 0.42 at k=3 | 0.416 vs RF 0.281 | OK |
| 65 | Abstract: 0.76 vs 0.28; 0.39 vs 0.67 | as stated | match | OK |
| 66 | Highlight 4 / "LIME often unstable" | qualitative | 32%/5% (TON) and 68%/7% (UNSW) of records reach 0.75 | OK |

## 7. Tables 6 and 7 (counterfactuals)

I rebuilt the k distribution from `k_hist` and re-bootstrapped it. The paper's CIs equal the stored CIs. My re-bootstrap from the histogram differs by at most 0.02 because the draws come in a different order (Monte Carlo noise). I checked the "most-changed fields" against `top_fields`.

| Table | Cells | Result |
|---|---|---|
| Table 6 (TON) evadable %, 9 classes × 2 models | 18 | all 100: OK |
| Table 6 k mean [CI] | 18 | all OK (e.g. backdoor RF 2.62 [2.40, 2.84]; ransomware CNN 1.04 [1.00, 1.10]) |
| Table 6 most-changed fields | 18 | all OK |
| Table 7 (UNSW) evadable % | 18 | all 100: OK |
| Table 7 k mean [CI] | 18 | all OK (worms RF n=14: 1.86 [1.64, 2.00]; worms CNN n=12: 1.17 [1.00, 1.42]) |
| Table 7 most-changed fields | 18 | all OK. Second place is a tie in 5 cases (RF backdoor, shellcode, worms; CNN exploits, fuzzers), and the paper picks one tied field; also OK for CNN generic, where the top two are tied |

| # | Section 5.3 text | Paper | Recomputed | Status |
|---|---|---|---|---|
| 67 | TON evadable RF/CNN | 100 / 100% | overall_success 1.0 / 1.0 | OK |
| 68 | TON mean k | 1.86 [1.78, 1.94]; 1.30 [1.25, 1.34] | 1.858 [1.778, 1.938]; 1.296 [1.251, 1.340] | OK |
| 69 | "the smaller model is the easier one to evade" | CNN needs fewer fields | 1.30 vs 1.86 (UNSW 1.05 vs 1.60), both at 100% | OK |
| 70 | TON most frequent fields | RF dst_port, conn_state, proto; CNN proto, dst_port, conn_state | from summed per-class top-5 lists: RF 246/145/102; CNN 172/143/79 | OK |
| 71 | UNSW mean k | 1.60 [1.55, 1.65]; 1.05 [1.03, 1.07] | 1.599 [1.553, 1.647]; 1.046 [1.027, 1.066] | OK |
| 72 | UNSW most frequent fields | RF sttl, ct_state_ttl, dbytes; CNN sttl, dload, ct_state_ttl | RF 352/110/38; CNN 191/110/98 | OK |
| 73 | "at least 85%" of RF counterfactuals change `sttl` | ≥85% | exactly 352/414 = 85.0% (sttl is in every class's top-5, so this is exact) | OK |
| 74 | CNN: `sttl` in "at least 46%" | ≥46% | 191/412 = 46.4% | OK |
| 75 | "evade both models for almost every detected attack" (abstract, conclusion) | almost every | every queried detected attack (100%, 4/4 settings) | OK (conservative) |
| 76 | "usually by changing one or two fields" | qualitative | share with k ≤ 2: TON RF 79.6%, TON CNN 99.8%, UNSW RF 100%, UNSW CNN 100% | OK |

## 8. Section 5.4 (hardening)

| # | Item | Paper | Recomputed | Status |
|---|---|---|---|---|
| 77 | TON augmented variants | 1,796 | 1,796 | OK |
| 78 | TON evasion before → after | 100% → 64% | 99.8% → 63.6% (backdoor 0.98 before) | OK, rounding (note n1) |
| 79 | TON macro-F1 before / after | 0.9565 / 0.9561 | 0.95653 / 0.95606 | OK |
| 80 | "backdoor evasion almost disappears" | qualitative | 0.98 → 0.02 | OK |
| 81 | "ransomware remains fully evadable" | qualitative | 1.00 → 1.00 | OK |
| 82 | UNSW variants | 1,687 | 1,687 | OK |
| 83 | UNSW evasion before / after | 100 / 100% | 1.0 / 1.0 in all 9 classes | OK |
| 84 | UNSW macro-F1 before / after | 0.5913 / 0.5901 | 0.59125 / 0.59011 | OK |
| 85 | Highlight 5 / abstract "100% to 64% … at unchanged accuracy" | as stated | 99.8 → 63.6%; macro-F1 −0.0005 | OK |

## 9. Section 5.5 (shortcuts)

| # | Item | Paper | Recomputed | Status |
|---|---|---|---|---|
| 86 | TON without ports | 0.9565 → 0.9420 | 0.95653 → 0.94199 | OK |
| 87 | "does not depend on them for most of its accuracy" | qualitative | −0.015 macro-F1; the largest per-class drops are ransomware −0.055 and xss −0.049 | OK |
| 88 | UNSW without sttl and ct_state_ttl | 0.5913 → 0.5661 | 0.59125 → 0.56613 | OK |
| 89 | Largest per-class changes | worms 0.49 → 0.34, dos 0.35 → 0.30 | worms 0.491 → 0.340 (−0.152), dos 0.350 → 0.304 (−0.046); next is shellcode −0.018 | OK |
| 90 | sttl facts | see items 14–20 | | OK |

---

## Mismatches and unsupported claims, ranked by importance

**M1 (important): "fewer than half of their top-5 features" is wrong.** The claim appears in Section 5.2, the Conclusion and Highlight 3; Section 6.2 has the related "usually disagree on most of the top-5 features".
- A mean top-5 Jaccard below 0.5 does not mean fewer than half the features are shared. With two 5-sets, J = 0.43 already means 3 of 5 shared.
- Per instance, SHAP and LIME share on average 2.70 (TON RF), 2.13 (TON CNN), 2.82 (UNSW RF) and 2.60 (UNSW CNN) of their top-5 features.
- In three of the four settings, most alerts share at least 3 of 5 (65%, 62% and 56%). Only the TON CNN fits the sentence (34% share ≥3).
- The headline finding ("agree only moderately"; 0.39 vs 0.67) still stands.
- **Suggested rewording:** "share on average about 2–3 of their top-5 features (top-5 Jaccard 0.29–0.42)", and in 6.2, "often disagree on two or more of the top-5 features".

**M2 (important): 4,845 does not count what the sentence says.** The sentence is "A further 4,845 records share identical features with a record of a different label".
- Under that definition, the deduplicated data has **8,472 such records** (1,772 distinct feature vectors carrying two or more labels). The raw file has 30,528.
- 4,845 equals (raw records with any feature-twin, 84,049) minus (raw records with any exact twin, 79,204). That is the number of raw records whose only twins carry a different label. It leaves out records that have both an exact copy and a conflicting twin, and it is taken before deduplication, although the sentence ("a further") refers to the deduplicated data.
- **Suggested fix:** report 8,472 records in 1,772 conflicting feature groups, after deduplication. Alternatively, keep 4,845 and define it precisely.

**M3 (important, qualitative): CNN SHAP is not deterministic.** Section 4.4 says "SHAP for these models is deterministic given its background, so its stability is not at issue here", and Section 6.2 recommends SHAP because "it is deterministic".
- This holds for interventional TreeSHAP (RF).
- SHAP's GradientExplainer (expected gradients) samples a random background row and a random interpolation point for each of `nsamples=200` draws. With `rseed=None` it seeds from `np.random.randint`, and run_xai_v2.py sets no seed. So the CNN SHAP values change between runs, and their stability was not measured.
- **Suggested fix:** restrict the claim to the RF, or state that CNN SHAP is stochastic and that its run-to-run variability was not measured.

**m1 (minor): generic position.** *Generic* attacks first appear at row 117,204, which is 66.8% of the file, so "only after the first 67% of the file" is slightly too strong. Say "only in the last third of the file (from 66.8%)".

**m2 (minor, methods wording): hardening query sets.** Section 4.7 says the search is repeated "on the same held-out test attacks". The code draws, with the same seed, from the test attacks that each model itself classifies correctly, so the two query sets can differ slightly. Either rephrase to "on held-out test attacks correctly detected by each model", or fix the query set.

**m3 (minor): ports in UNSW-NB15.** "UNSW-NB15 retains no port feature after selection" implies that selection removed port features. The training CSV contains no port-number columns at all; only the port-derived count features `ct_src_dport_ltm`, `ct_dst_sport_ltm` and `is_sm_ips_ports` exist, and none was selected. Say "the UNSW-NB15 training file has no port-number fields, and none of its port-count features is selected".

**Notes, not errors**
- **n1.** TON evasion before hardening is 99.8%, not exactly 100%, because backdoor is 0.98. "100%" is an acceptable rounding, but "99.8%" would be exact.
- **n2.** The 43.2% overlap counts matches on features plus label, against the training split only. It rises to 44.1% if validation counts as training, and to 46.1% on features alone. It may be worth stating "(features and label identical)".
- **n3.** The interaction values use path-dependent TreeSHAP, while 4.1 says interventional for the RF. This is harmless, but could be stated.
