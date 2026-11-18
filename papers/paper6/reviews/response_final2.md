# Paper 6: response to the final-round-2 reviews (2026-11-15)

Referee report: review_final2_referee.md (6.5, major revision). Numbers audit: review_final2_numbers.md (0 wrong numbers; A1).
All numbers below are filled from values.json; verify.py: 0 failures.

## Referee: major issues

| Item | What changed |
|---|---|
| M1 / audit A1: cost table described another GBDT | Table 8 now times the scored, rolled-back GBDT (refitted with seed 0, test score reproduced): TON-IoT 700 trees, 10 MB, 5.56 ms per record on one thread. Single-record times for both tree ensembles are single-threaded (minor 11). §5.6, §6.1, §7 and the abstract now say the most accurate model needs 10 MB; the "hundreds of megabytes" claim is kept only for the RF. |
| M2: late − early gain concentrated in MITM; seed-only intervals | Seeds × test-records bootstrap (B = 2,000) added to Table B.1 and §5.2: TON-IoT [+0.003, +0.007]. §5.2 states that 63% of the gain comes from MITM (156 test records) and gives the UNSW-NB15 worms share; the split-level replicates are reported next to it. |
| M3: tuned trees vs untuned networks | New runs (run_p6_nettune.py): symmetric grid for early and late CNN fusion (learning rate 10⁻³/3 × 10⁻⁴ × dropout 0.1/0.3 × width ×1/×2), chosen on validation macro-F1 with seed 0 as for the trees, then five seeds (Table B.6, §4.2, §5.2). Results: on TON-IoT tuning raised early fusion by +0.006 [+0.004, +0.008] (5/5) and late fusion by +0.001; **tuned late minus tuned early fusion is 0.000 [−0.003, +0.003]**, so the fixed-configuration late-fusion gain is matched by tuning early fusion. The trees stay ahead of the best tuned network: RF +0.007, GBDT +0.009 on TON-IoT; +0.034 and +0.036 on UNSW-NB15 (Welch p < 0.001). On UNSW-NB15 the grid kept the fixed late-fusion configuration and its retraining scored −0.009 [−0.017, −0.001] against the original runs; the paper reports this as run-to-run GPU variation and no longer treats the UNSW-NB15 architecture difference as reproducible. "Untuned" is removed from the abstract, highlights, §5.2 and §6.3; §6.3 states that only the two fusion CNNs were tuned. |
| M4: reused random-split runs | All TON-IoT and UNSW-NB15 random-split and semantic-split networks were re-run with the present code; reuse remains only on N-BaIoT (§4.2). Semantic − random, late fusion: TON-IoT +0.001 [0.000, +0.002], UNSW-NB15 +0.006 [+0.001, +0.012]. The abstract's "an effect up to about 0.02 cannot be excluded" is removed because the new interval excludes it. |
| M5: one non-saturated IoT dataset | Rescoped, no new dataset: title "…: A Controlled Study on TON-IoT"; contribution 1 names UNSW-NB15 a non-IoT robustness check and N-BaIoT a ceiling check; §6.3 names CICIoT2023 and Edge-IIoTset as the missing second case. |
| M6: length | Partly done: §2.1, §5.2 (capacity, other splits), §5.5 (about 250 words) cut; several Table 3/6 rows moved to Tables B.4/B.5. The new grid adds a paragraph and Table B.6, so the main text is not down to 8.5–9k words. **Open if the editor asks.** |

## Referee: minor issues

1. Highlights now include the random-grouping result; all five rewritten (≤ 85 characters).
2. Label conflicts measured (dup_labels.py): for 39% of the matched UNSW-NB15 test records the majority label of the matching training records differs; none of the 35 TON-IoT matches conflicts (§5.2).
3. The ±0.01 margin is tied to a 0.1 change in one class's F1 on TON-IoT and called a reading aid (§4.3).
4. §4.2 states that the GBDT's internal early-stopping split is drawn from the oversampled set.
5. §5.4 says the weight–reliance correlation is largely built in.
6. Companion studies: §5.6 says their result files are read by our analysis scripts and released with the code; the cover letter declares them.
7. Resplit feature-selection leak: acknowledged in §4.2; the replicates are descriptive, not confirmatory.
8. §4.1 explains the name "early fusion" for one CNN on all features.
9. Params column: caption says counts differ on other datasets.
10. Convention for interval bounds that round to 0.000 stated in §4.3.
11. Single-record GBDT latency now single-threaded (see M1).
12. Sanjalawe et al. (2025): published 2025; the DOI contains 2026 as registered with Crossref. No change to the paper.
13. AI declaration kept as is (Elsevier-compliant).
14. Not changed: Figure 1 and §4.1 both describe the architectures.

## Numbers audit

A1 is M1 above. No other discrepancy.
