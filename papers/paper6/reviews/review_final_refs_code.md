# Paper 6: final check of references and of the Methods against the code (2026-11-07)

Scope: `Manuscript_filled.md` (417 lines, 48 references), `literature_verified.md`, `check_citations.py`, `reference_audit/` (Crossref audit of 2026-11-05), and the code in `Papers pipeline/experiments/paper6_multimodal/` (`run_p6_v2.py`, `bench_latency.py`) together with `compute_values.py`, `src/data/loader.py` and `src/data/preprocessing.py`. Nothing was trained and no dataset was loaded. Only result JSONs, `streams.json`, `tune_trees.json`, `latency.json` and the small `clean_split0.npy` / `yte_split0.npy` arrays were read. Git history (`55350cc` v2, `f5b08e9` v3, `d2e74ca`/`b37e9c3` v4) was diffed.

Severity scale: **high** = would mislead a reader or fail a check at submission; **medium** = needs a text or code fix before submission; **low** = wording, disclosure or robustness; **info** = verified, no action.

**Result in brief.** All 48 references exist. Their metadata match Crossref, PMLR, NeurIPS or arXiv. Citations and the reference list match one to one. There are no critical problems. One citation is unsupported (R1). The data-availability statement is currently untrue for Paper 6 (M1). No code bug was found that changes a reported number. The remaining items are wording or disclosure fixes, several of them left open from round 2.

---

## References

### How the references were checked

- **Audit coverage.** All 48 entries of the current list are in `reference_audit/refcheck.json` with their exact text, so there are no new or changed entries since the audit.
  - The audit checks **existence only**: DOI title overlap ≥ 0.6. It does not check authors, volume or pages.
  - For four entries the audit's first-pass search hit a wrong record. These are Wu 2022 ("Neural Networks and Deep Learning"), Vaswani 2017 (DOI 10.65215/…), Nagrani 2021 and Pedregosa 2011. Nagrani and Pedregosa were rescued by an arXiv title match.
  - Wu was **not** really confirmed by the audit. I confirmed it directly (see below).
- **Full metadata re-check, this round.**
  - For all 38 DOIs I fetched the Crossref record and compared every author surname, year, volume, issue, first page and article number with the entry. **All 38 match.**
  - Sanjalawe 2025: the DOI suffix says 2026, but Crossref registers volume 5 as 2025, so the entry is correct as written.
- **References without a Crossref DOI.**
  - Checked on PMLR (citation_* meta tags): Du 2023 (v202, 8632–8656, 8 authors), Wu 2022 (v162, 24043–24055, 4 authors), Huang 2022 (v162, 9226–9259) and Ioffe & Szegedy 2015 (v37, 448–456). **All match.**
  - Checked on the NeurIPS proceedings page: Nagrani 2021 (pp. 14200–14213, 6 authors). **Matches.**
  - Checked on the arXiv API and abstract page: Dinh 2024b (arXiv:2403.15511; v1 22 Mar 2024, v2 25 Nov 2025, no journal-ref). See R4.
  - Arp 2022, Bahdanau 2015, Vaswani 2017 and Pedregosa 2011 were confirmed in earlier rounds (USENIX, arXiv and JMLR pages), and their metadata are unchanged.
- **Citation cross-check.** `check_citations.py` crashes (see R3). A fixed copy (same regex, plus year suffixes, plus Appendix A) run on both `Manuscript_filled.md` and `Manuscript.md` gives:
  - cited but not listed: none (the one hit, "CIC-IoT 2022", is a dataset name, a false positive);
  - listed but not cited: none;
  - 48 references, one to one.

### Findings

| ID | Location | Finding and evidence | Severity |
|---|---|---|---|
| R1 | §3.2, l.64: "This flow-versus-behaviour split is used, in similar forms, in the fusion literature (Ayantayo et al., 2023; Dinh et al., 2024a)." | **The citation does not support the claim.** Neither paper splits features into volume/timing versus protocol/behaviour. Ayantayo et al. group UNSW-NB15/NSL-KDD features **by data type** (float / integer / string), as recorded in `literature_verified.md` B1 [FT] and review_r1. Dinh et al. 2024a use one sub-encoder per input source or feature group of different dimension. In the extended version (2024b) the groups come from **KL-divergence clustering** of feature distributions (review_r1, l.91). Both are examples of grouping a flow table's features, not of this particular split. **Fix:** "Grouping the features of one flow table into separately encoded subsets is common in the fusion literature (Ayantayo et al., 2023, by data type; Dinh et al., 2024a, by feature group)". Alternatively, drop the claim that the *same* split is used. | medium |
| R2 | §2.1, l.32 ("Attota et al. (2021) train one network per feature level (bi-flow, uni-flow and packet features) of an MQTT dataset") and Table A.1, Attota row (Source = Abs) | **The description goes beyond the source the table says was read.** The row is marked abstract-only, but the dataset (MQTT) and the three views are not in the abstract (review_r1 l.89, review_r2 R2). `literature_verified.md` B12 marks it [FT] from an earlier session, while the round-1 reviewer could not retrieve the PDF. Its "not in abstract" cells are now consistent. The §2.1 sentence and the "Branch inputs" cell still rest on a full-text read that is not documented. **Fix:** obtain the PDF and mark the row FT, or reword to what the abstract says ("an ensemble of per-view networks in a federated multi-view IDS"). | low |
| R3 | `check_citations.py` l.6 and l.16 | **The citation checker is broken and checks the wrong file.** (a) `re.search(r'\((\d{4})\)', r).group(1)` returns None for "(2024a)" and "(2024b)", so the script crashes (AttributeError) before printing anything. (b) It reads `Manuscript.md`, not `Manuscript_filled.md`. (c) It ignores Appendix A, whose table cites the 14 studies. **Fix:** use `\((\d{4}[a-z]?)\)` and capture the suffix in the body regex too (`((?:19\|20)\d\d[a-z]?)`). Add a pattern for "Dinh et al. (2024a, 2024b)". Exclude the 'CIC-IoT' token. A working version (scratchpad `cc.py`) gives 0/0 on both files. | low (tooling) |
| R4 | Reference "Dinh … (2024b) … (arXiv:2403.15511v2)" | **The year does not match the version cited.** v2, the version whose Table V gives the quoted 0.727/0.728, was posted on **25 Nov 2025**. APA dates a preprint version by its posting date, so either cite "(2025)" with v2, or cite v1 (2024) if v1 contains the same table. The label "2024b" would then change to "Dinh et al., 2025" in §2.1 and Table A.1. Existence and content are fine: the numbers were confirmed in round 2 (R1 there). | low |
| R5 | §6.3, l.281 ("CICIoT2023 or Edge-IIoTset"); Table A.1 dataset names | Several datasets are named without a reference: CICIoT2023, Edge-IIoTset, CIC-IoT 2022, X-IIoTID and others. In the table this is acceptable, because the studies are cited. In §6.3, where they are proposed as future benchmarks, a reader may expect a citation. If one is added, it must be verified first (the user's only-existing-references rule). | low (optional) |
| R6 | §2.2, l.40: "one modality can dominate optimisation and leave the other under-used (Wang et al., 2020; Peng et al., 2022; Wu et al., 2022; Huang et al., 2022)" | **Wang 2020 supports the first half of the sentence, not this clause.** Wang 2020 attributes the failure of joint training to overfitting and to different generalisation rates of the modalities, not to dominance. It does support the preceding clause ("joint training is not automatically better than the best single modality"). The other three support the dominance clause. **Fix:** move the Wang citation to the preceding clause. | low |
| R7 | §3.1, l.50: TON-IoT network flows "(Alsaedi et al., 2020; Moustafa, 2021)" | Alsaedi 2020 describes the *telemetry* (IoT/IIoT sensor) part of TON_IoT. The network flows are described by Moustafa 2021. The pairing is acceptable as an umbrella citation. | info |
| R8 | §2.1 counts (l.34) and Table A.1 | **Verified.** Eight studies are FT and six are Abs. Of the FT studies, two (Kiflay, Dinh 2024b) ran a tree ensemble, and one Abs study (Singh) did. Random-grouping and matched-parameter controls are 0/14. All of this matches the round-1/2 notes. The Kiflay (decision-level fusion beats one RF on the concatenated inputs), AboulEla ("modest, dataset-dependent"), Singh (trees reach "ideal accuracy" against 99.32% for the CNN), Li ("generally, but not always"), Afraji (binary only) and Sanjalawe ("selected features") wordings are now supported by their sources. | info |
| R9 | Remaining claim–source pairs | **Verified and supported.** Baltrušaitis 2019 and Ramachandram 2017 (multimodal surveys); Grinsztajn 2022, Shwartz-Ziv 2022 and McElfresh 2023 (trees strong on tabular data); Arp 2022, Sommer 2010 and Apruzzese 2023 (security ML pitfalls); Engelen 2021 (dataset errors); Du 2023 (weaker uni-modal features; uni-modal ensembles); Jain & Wallace 2019 and Wiegreffe & Pinter 2019; Ho 1998 (random subspace); Ioffe & Szegedy 2015 (BN, cited as the "most likely mechanism"); Chawla 2002; Meidan 2018 and Mirsky 2018 (N-BaIoT, Kitsune, nine devices, 115 features); Moustafa & Slay 2015. | info |

---

## Methods-vs-code

Code of record is `run_p6_v2.py` at `b37e9c3`, which writes `results_v4`. `compute_values.py` reads:

- `results_v4` for everything except Table 7;
- `results_v2` (min–max) and `results_v3` (quantile, 60-epoch protocol) for Table 7 and the re-run statistics;
- `experiments/baseline/results_v2` for the compact forests;
- `latency.json` (from `bench_latency.py`) for Table 8.

`run_multimodal_experiments.py`, `run_multiseed.py` and `extract_attention_perclass.py` are v1 scripts and are **not** used by any reported number.

### Verified, matches the text

| Item | Text | Code / evidence |
|---|---|---|
| Streams | TON 7 + 18, UNSW 6 + 4, N-BaIoT 30 + 85; stream contents as in Table 1 | `GROUPS`; `streams.json`. TON s2 = ports, DNS fields, proto/service/conn_state one-hots. UNSW s2 = sttl, ct_state_ttl, ct_dst_src_ltm, ct_srv_dst. N-BaIoT regex `MI_dir_\|H_L` gives MI_dir 15 + H 15 against HH 35, HH_jit 15, HpHp 35. |
| Encoder | Conv64(k3)+BN, max-pool, Conv128+BN, GAP, Dense64 | `_encoder` (cnn) |
| Heads | single/early: dropout 0.3 + softmax; late: concat→Dense128→dropout→softmax; attention: Dense(2, softmax) on concat, weighted sum, Dense64, dropout, softmax | `build` |
| Parameter counts | 34,634 / 85,770 / 73,036 / 86,613 / 43,594; MLP 27,466 / 28,874 / 29,770 / 72,842 / 60,108 | Recomputed by hand from the layer shapes. All match, and they agree with `params` in the JSONs. Wide = every width × 1.6 (102/205/102). |
| Capacity controls | wide early fusion, early fusion with 128 head, late fusion with both encoders on all features | `early_wide`, `early_head`, `late_all` (`inputs` → `[X, X]`) |
| Quantile transform | uniform, 1,000 quantiles, 200,000-record subsample of the SMOTE set; whole training set on N-BaIoT | `_finish`: `QuantileTransformer(n_quantiles=1000, subsample=200000, random_state=0).fit(Xfit)`. On N-BaIoT, `Xfit = Xtr` (110,227 < 200k). |
| Training | Adam 1e-3, ReduceLROnPlateau(0.5, patience 4, min 1e-5), batch 512, ≤120 epochs, EarlyStopping(val_loss, patience 10), no class weights | `run_cnn` |
| Best-weight restore | restored in every run, including capped runs | `BestWeights` restores in `on_train_end`. An assertion checks that the restored epoch equals argmin(val_loss). Every non-reused JSON has `restored_epoch == best_epoch`. |
| Budget statistics | 1 of 90 runs capped, latest best epoch 118 | 90 = 60 main (75 minus 15 reused N-BaIoT) + 30 capacity. The only capped run is N-BaIoT `s2` seed 4 (epochs 120, best 118). No MLP, column-order or resplit run is capped. |
| Reuse | 104 of 120 random-split network runs + 15 N-BaIoT main runs from v3; v3 differed only in the 60-epoch cap and conditional restore | Counted in results_v4: TON 27 + UNSW 40 + N-BaIoT 52 = 119 marked `reused_from`. Every reused run has epochs < 60. Every reused random run's `cfg.s1` equals the v4 `random_split` features. `git diff 55350cc f5b08e9` and the v3 `run_cnn` confirm the identical data, model and schedule. |
| Re-run variation | 51 runs, 9 same best epoch, ΔF1 mean 0.003, max 0.018 | `values.json`, computed by `compute_values` (a) |
| Table 7 provenance | min–max first run / quantile same protocol / final run | v2 → v3 diff adds only the quantile transform (plus a separate `controls` group and latency code). v2 TON s1 best epochs 3, 1, 4, 1, 1 ("at most 4"). Every v3 UNSW main run stopped early (epochs 12–40), so the §5.2 claim that "every UNSW-NB15 network stopped early" is true. |
| Trees | RF 200 trees; HistGB ≤1,000 iterations with early stopping on an internal 10% split; grid as stated; tuned on validation macro-F1, all features, then fixed; trained on min–max SMOTE data | `make_tree`, `tune_trees`, `run_tree`. Tuned: TON RF depth None, leaf 1, GB lr 0.05 / 127 leaves; UNSW RF depth 20, leaf 1, GB 0.05 / 31; N-BaIoT RF 20/1, GB 0.05/127. UNSW GB n_iter 986–1,000. |
| Random splits | 10 per dataset, semantic sizes, split r uses seed r mod 5; within-seed difference = semantic minus the mean of the two random splits with that seed | `random_split` (rng 1000+r); `jobs('random')`; `compute_values` l.398 |
| Column order | 5 random orders of all features, order k with seed k | `column_order` (rng 2000+r); `jobs('colorder')` |
| MLP | two 128-unit layers with BN, 64 embedding; TON/UNSW only | `_encoder(kind='mlp')`; no `m_*` results for N-BaIoT |
| Resplits | 5 new stratified 70/15/15 splits (loader seed 42+k), selected features kept, min–max/SMOTE/quantile refitted, one seed per split | `load(split=k)`; `jobs('resplit')` |
| Clean subset | test records whose selected-feature vector matches a training record removed: TON 0.1%, UNSW 16.3%, N-BaIoT 0% | `_finish`; `clean_split0.npy` gives 0.0013 / 0.1627 / 0. No class empties in the clean subset (UNSW worms 18 of 19), so clean macro-F1 has no zero-division artefact. |
| Statistics | paired t CI over seeds, ties at <1e-6, Welch for trees vs networks, Holm over 8 | `tci`, `paired`, Welch at l.270, Holm step-down with running max at l.278–284. Family size 8. Significant: TON late−early (0.006) and TON late−wide (0.009). |
| Attention analysis | per-record weights, class means, mean-replacement ablation with weights kept, reliance, permutation p | `attention_analysis`; `compute_values` l.318–352. `attn_base_check` = 1 in all runs. |
| Latency | CPU, untrained networks, tuned trees, one record per call and 1,000 per call, i5-10200H | `bench_latency.py` (`CUDA_VISIBLE_DEVICES=-1`); `latency.json`. RF 210.8 MiB = 221 MB. RF/late = 8.035/0.554 = 14.5. Late fusion 85,770 × 4 B = 343 kB. |
| Majority-class baseline | 0.037 on TON | 2p/(1+p)/K with p = 6290/27533 gives 0.037 |
| Pre-processing (§3.1) | ≤30-value one-hot, dedup after drops, 70/15/15 stratified, cross-split guard on all columns plus label, MI > 0.05, RFECV 3-fold, 50-tree RF, macro-F1, 50k subsample, ≥10 features, min–max, SMOTE | `loader.py` steps 1–10; `preprocessing.py` l.47–131 |

### Findings

| ID | Location | Finding and evidence | Severity |
|---|---|---|---|
| M1 | Declarations, l.289 ("The code, stream definitions and all result files are available at https://github.com/Amoakoemma123/iot-ids-pipeline") and l.285 ("We release the code, the result files and the stream definitions") | **The Paper 6 code and results are not in the public repository.** The local clone of the public repo (`release_v2/gh/IREAP-IoT-IDS-v2`, branch main, up to date with origin, last P6-relevant commit none) contains only the **v1** scripts `run_multimodal_experiments.py`, `run_multiseed.py` and `extract_attention_perclass.py`. `run_multimodal_experiments.py` is the v1 code with the leaky UNSW `id` that the paper says it replaces. Missing from the repo are `run_p6_v2.py`, `bench_latency.py`, `results_v2/`, `results_v3/` and `results_v4/` (all untracked even in the private `Papers pipeline` repo, which has no remote), and `compute_values.py`, which lives in `Revision_v2/Paper6`. **Fix before submission:** push `run_p6_v2.py`, `bench_latency.py`, `compute_values.py`, the three result directories (JSON + pred/clean npy) and `streams.json`. Mark or remove the v1 scripts. Then record the commit hash in the manuscript, as was done for P4/P5. | **high** (submission blocker) |
| M2 | §4.1, l.77 ("1 of the 90 network runs … used the full budget (latest best epoch 118)") | **The one capped run was still improving, so the budget was likely binding for it.** It is N-BaIoT stream 2, seed 4: epochs 120, best epoch 118. Its validation loss improved two epochs before the cap. N-BaIoT is not in the confirmatory family and is saturated (0.994), so no conclusion changes. **Fix:** add "(N-BaIoT stream 2, seed 4; still improving at the cap)". | low |
| M3 | Table 2 header "Params" (l.101) | **Still unlabelled (round-2 m5 open).** Counts are TON-IoT values and differ on the other datasets: MLP early fusion 29,770 vs 27,850 on UNSW; late fusion 85,770 vs 85,641 on N-BaIoT; maximum deviation 6% (`params_dev_max_pct`). **Fix:** "Params (TON-IoT)". | low |
| M4 | §5.6 / Table 8 | **Thread settings are not reported (round-2 m7 open).** The RF is timed per record with `n_jobs=1` and per batch with `n_jobs=-1`. HistGB uses OpenMP on all cores in both modes, which probably inflates its 19–115 ms single-record times. TensorFlow uses its default pools on 8 logical cores. **Fix:** add a clause to §5.6 or the Table 8 caption. | low |
| M5 | §5.3, l.182 ("semantic above 8 of the ten", "9 of the ten", "semantic attention fusion is above all ten random splits there") | **A 5-seed mean is compared with single-run values (round-2 m10 open).** `compute_values` l.397 counts random runs below the semantic *5-seed mean*, so the counts are not like-for-like. On N-BaIoT, 37 of the 40 random networks are also v3 runs. The paired within-seed differences are the correct statistic and are already reported. **Fix:** soften the wording, e.g. "the semantic seed mean exceeds 8 of the ten random-split values". | low |
| M6 | §4.2 "Data splits", l.89 ("single-stream, early, late and attention networks and the trees") | `jobs('resplit')` trains `rf`, `gb`, `rf_s1`, `rf_s2`, but **no `gb_s1`/`gb_s2`**. Nothing in the text uses per-stream GBDT on resplits, so this is wording only. **Fix:** "and the RF (all features and per stream) and GBDT (all features)". | low |
| M7 | §5.3, l.184 ("early fusion on five random column orders … (original order 0.948)") | **"Original order" is not the dataset's column order.** The reference early-fusion input is `X[:, np.r_[s1, s2]]`: the stream-1 features in `GROUPS` order, then stream 2. That is the semantic order. **Fix:** "(semantic order, stream 1 then stream 2: 0.948)". | low |
| M8 | §4.1, l.75 ("The weight of stream 1 is recorded for every test record") | Weights are computed per record, but only aggregates are saved: overall mean, per-class means, within-class SD and norm-share per class. They are not stored per record. **Fix:** "is computed for every test record". | low |
| M9 | §3.1 pre-processing description | **Undisclosed steps (round-2 m13 partly open).** `loader.py` steps 6–9 are not mentioned: drop numeric columns with >40% missing on train; train-median imputation; ±inf replaced by the train 99th percentile; drop zero-variance features. The one-hot category sets and the 30-value rule are decided on the full data before the split (label-free), and `drop_first=True` is used. None of this changes results. A sentence or a pointer to the companion study suffices. | low |
| M10 | §4.2 Trees: "early stopping on an internal 10% split" | `n_iter_no_change=20` is not stated. The internal split is drawn from the **SMOTE-augmented** set, which is why the UNSW GBDT runs to the cap (now disclosed). Optional: add "(20 iterations without improvement)". | info |
| M11 | `compute_values.py` l.278–284 (Holm) | **The Holm family can shrink without an error (round-2 m11 open).** If a family member's files were missing, `paired()` would return None and Holm would silently adjust over fewer tests. It is 8 now (`holm_family_size`). **Fix:** `assert m_ == 8`. | low (robustness; no current effect) |
| M12 | `compute_values.py` l.327–329 (attention vs stream advantage) | **"Per seed" is half true.** The per-seed ρ correlates each seed's class weights with the **seed-averaged** stream advantage, not with that seed's advantage. The text (l.227, "Spearman's ρ over classes, per seed") is defensible, but a reader may assume both sides are per seed. Optional: "per seed, against the seed-averaged advantage". | info |
| M13 | Provenance of v3 (round-2 m14) | v3 main and most random runs were written before commit `f5b08e9`. The diff, parameter counts and JSON keys are consistent with that commit, but this cannot be proven. The v3 RF files are copies of v2 (inputs unchanged). Mention both in the release notes when M1 is fixed. | info |

### Code bugs that could change results

None found. Specific checks:

- **Restore logic.** `BestWeights` and the strict `<` comparison agree with `np.argmin`, enforced by an assertion. EarlyStopping fires before `on_train_end`, so the restore always happens.
- **Reuse.** The `epochs >= 60` test is conservative. Every reused random split's features equal the v4 split.
- **Index handling.**
  - `_perm` sets `n1 = len(order)` and `n2 = 0`, which gives a correct early-fusion input.
  - `late_all` uses `n1 = n2 = all` with inputs `[X, X]`.
  - `run_tree` picks columns by its suffix.
  - Tree probabilities are mapped through `classes_`.
- **Metrics.**
  - Macro-F1 averages over all classes.
  - The clean subset keeps all 10 classes on every dataset, so there is no zero-division artefact.
  - PR-AUC is the macro average over the classes present.
- **Pairing.** Paired comparisons align seeds by index. The within-seed random comparison uses r ∈ {s, s+5}.
- **Resplit SMOTE.** Built without UNSW's `smote_max_per_class`. As round 2 showed (m9), this has no effect, because the majority class (36,322) is below the 50k cap.

The only residual risk to the numbers is the documented, unavoidable GPU nondeterminism. The text discloses it, with run-to-run ΔF1 0.003 on average and 0.018 at most.

---

## Priority list

1. **M1 (high):** push the P6 code, `compute_values.py` and `results_v2/v3/v4` to the public repo. Mark the v1 scripts as superseded. Cite the commit.
2. **R1 (medium):** reword the §3.2 "flow-versus-behaviour split … in the literature" sentence to what Ayantayo and Dinh actually do.
3. **Low items:** R2–R6 and M2–M9, M11. Each is a one-line text change or a small assert.
