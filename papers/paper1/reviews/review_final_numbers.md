# Paper 1 v2: final audit of numbers and internal consistency (2026-11-07)

Scope: Manuscript_filled.md (and Manuscript_v2.docx, whose paragraph text matches it), build.py HIGHLIGHTS, values.json, points.json, the raw files in `Papers pipeline/experiments/baseline/results_v2/` (seed*.json, latency.json) and the figure.

## How it was checked

- **verify.py** was run (it writes nothing): **0 failures, 0 pending**.
- **compute_values.py was not run**, because it overwrites values.json and points.json. Instead I checked that no result file is newer than values.json (none is; values.json, points.json and fig1 are all from 2026-11-04 22:30).
- **Re-filled in memory.** I filled Manuscript.md from values.json + build.EXTRA in memory. The output is byte-identical to Manuscript_filled.md, and no placeholders are missing. The highlights fill to 62–76 characters and match Highlights.docx.
- **Docx check.** Every non-table paragraph of Manuscript_filled.md appears in Manuscript_v2.docx. The docx has 6 tables and captions Table 1–4, A.1, A.2 and Figure 1.
- **Anonymised version.** The anon md differs from the full one only in the repository link and the CRediT paragraph, as intended.
- **Independent recomputation from the raw seed files** (scratch script, no datasets loaded, nothing trained):
  - all 29×3 configuration means, SDs and minima
  - int8 drops per seed, int8 agreement and n_iter
  - clean-record F1 and the GBDT collapse counts
  - the port deltas and the worst class per model
  - latency entries, plus extra Welch tests.
- **Sample of key numbers confirmed directly against the raw files:**
  - **Best models and Welch tests:**
    - TON best GBDT 0.96145 (gb05_l15) and best net 0.95220 (CNN w2 float32), Welch p = 2.4e-7
    - UNSW best GBDT 0.60560 and best net 0.56275 (CNN w2 int8), p = 7.8e-6
  - **Budget comparisons:** +0.028 (0.95148 vs RF 25/d8 0.92338) and +0.046 (0.56275 vs 0.51653)
  - **GBDT collapses:** 6 of 20 / 2 of 20, minimum 0.334; lowest lr-0.05 seeds 0.798 and 0.594
  - **Clean-record leads:** +0.009 / +0.071
  - **Min–max controls:** 0.900 ± 0.005, +0.047, +0.020
  - **Port deltas:** −0.0149, −0.0147, −0.0151, −0.0139, −0.0132, +0.0067; ransomware is worst in all six models, including GBDT lr 0.1
  - **Latency:** 97.16 / 0.412 ms gives 236×; UNSW 87.23 vs 1.573 ms
  - **Parameter counts:** 23,946 / 17,162 (also re-derived by hand from the layer list, as were all float32 and LR sizes)
  - **Other:** 21 of 29 N-BaIoT configurations ≥ 0.998; 0.1% / 16.3% duplicate share (clean_n 27,498 / 27,533 and 13,308 / 15,894); earlier CNN 0.880 (Paper1_v1_text.md, 0.8803); earlier forest 97,704 KB = 95 MB.

**Result: every number printed in the text, tables, abstract and highlights equals its values.json entry, and every sampled number equals the raw files.** The findings below are therefore about interpretation, wording and consistency, plus two outright errors.

---

## WRONG

### W1. The attack detection rate is announced but never reported
- **Quote (§4.4):** "we also report the benign false-positive rate (FPR, …), the attack detection rate and the macro-averaged area under the precision–recall curve"
- **Problem:** No table or sentence reports the detection rate. Table A.1 has macro-F1, benign FPR, PR-AUC and the two sizes only.
- **Source:** The field exists in the raw files (`attack_detection` in every seed*.json, e.g. TON gb05_l15 0.9994, UNSW gb05_l15 0.9785). compute_values.py never reads it.
- **Fix:** Either drop "the attack detection rate" from §4.4, or add a column to Table A.1.

### W2. "A few hundred small trees" contradicts the paper's own results
- **Quote (§1, para 2):** "gradient-boosted trees reach high accuracy with a few hundred small trees"
- **Contradiction:**
  - §5.4 says "The best GBDT on TON-IoT evaluates 10,000 trees per record (1,000 boosting iterations, one tree per class)".
  - The abstract and conclusion say "their accuracy came from thousands of trees".
- **Correct value:** best TON GBDT = 10,000 trees (seed 0; n_iter 788–1,000 across seeds × 10 classes). Best UNSW GBDT = 9,020 trees (`UNSW_NB15_best_tree_ntrees`). Even the equally accurate 63-leaf TON GBDT uses 217–273 iterations × 10 classes ≈ 2,200–2,700 trees.
- **Fix:** For example, "gradient-boosted trees reach high accuracy with a few hundred boosting rounds of small trees", or simply "with small trees".

---

## OVERSTATED

### O1. "236 times slower" rests on a tie-break in the fourth decimal
- **Quotes:**
  - **Abstract and §7:** "the most accurate GBDTs were the slowest models, 97 and 87 ms per record against 0.41 and 1.57 ms"
  - **§5.4:** "The most accurate models are therefore the slowest … 97 ms, 236 times the best network"
  - **§6.1:** "cost 236 times the per-record latency"
  - **Highlight 4:** "The best GBDTs were 236 times slower per record than the best CNN on TON-IoT"
- **Problem on TON-IoT:** The "best" GBDT (15 leaves, lr 0.05, 0.96145) and the 63-leaf GBDT at lr 0.05 (0.96141) are indistinguishable (Welch p = 0.95, my computation from the seed files). The 63-leaf model runs in **25.1 ms**, i.e. **61×** the best network, not 236× (latency.json, TON gb05_l63 single_ms 25.12).
- **Problem on UNSW-NB15:** The unbounded forest ties the best GBDT (0.60558 vs 0.60560, p = 0.99) at **8.4 ms**. The 25-tree depth-16 forest (0.601, within 0.005, and above the best network with p < 0.001) takes **1.29 ms**.
- **What survives:** "GBDTs were the slowest" is true. Equal accuracy is available from tree models at a fraction of that latency, so the latency cost belongs to the GBDT choice, not to accuracy itself.
- **Fix:** Report the latency range over the statistically tied top configurations (TON: 25–97 ms). Say that RFs of near-best accuracy take 1.3–8 ms, and soften "236 times" in Highlight 4 accordingly.

### O2. "An unbounded forest was never needed" does not hold on duplicate-free UNSW records
- **Quotes:**
  - **Abstract:** "an unbounded forest was never needed"
  - **§6.1:** "an unbounded forest is never needed"
  - **§5.1:** "the ranking holds on the test records that have no exact match"
- **Problem:** On the 83.7% of UNSW-NB15 test records without a training match, the unbounded forest (200 trees, unlimited depth) scores **0.667** clean macro-F1 against **0.630** for the best GBDT (gb05_l15), Welch p = 0.005 (`clean_macro_f1` in the seed files). The earlier forest (0.644) and the 63-leaf GBDT (0.645) also beat the "best" GBDT there.
- **What holds:** The tree-above-network ranking does hold.
- **What does not hold:** The within-tree ranking, and with it the "never needed" claim. On all records, the unbounded forest ties the GBDT on UNSW (p = 0.99), so "never needed" there means "not more accurate", not "less accurate".
- **Fix:** Restrict the §5.1 sentence to "the tree-over-network ranking holds". Qualify "never needed" ("on all test records"), or report the clean-record result.

### O3. The int8 comparison sets a seed mean against a single-seed maximum
- **Quotes:**
  - **Abstract:** "Int8 quantisation cost the CNNs at most 0.010 macro-F1 but the MLPs up to 0.073 on single seeds"
  - **Highlight 3:** "Int8 cost CNNs at most 0.010 macro-F1, MLPs up to 0.073 on single seeds"
  - **§7:** the same sentence
- **Problem:** 0.010 is the largest *mean-over-seeds* CNN drop (UNSW CNN w0.5, 0.0104). 0.073 is the largest *single-seed* MLP drop (N-BaIoT MLP w2, seed-level 0.0733).
- **Like-for-like numbers:**
  - Largest single-seed CNN drop: **0.021** (UNSW CNN w0.5, 0.0205).
  - Largest mean MLP drop: **0.021** (UNSW MLP w1, 0.0207); 0.016 on N-BaIoT.
- **Fix:** For example, "at most 0.010 for the CNNs and 0.021 for the MLPs on average, and up to 0.021 vs 0.073 on single seeds". The §5.3 text states both correctly; only the summary sentences mix them.

### O4. Lr 0.05 is not seed-stable either
- **Quotes:**
  - **Abstract and Highlight 5:** "At the library's default learning rate, gradient boosting collapsed on some seeds"
  - **§6.2 item 4:** "a default GBDT collapsed on some seeds here"
- **Implication:** These sentences suggest lr 0.05 is seed-stable.
- **Problem:** On TON-IoT the 7-leaf GBDT at lr 0.05 has one seed at **0.798** against 0.957–0.960 for the other four (mean 0.927 ± 0.072; Table A.2; raw gb05_l7). By the paper's own criterion ("more than 0.05 below the mean"), that seed would also count as a collapse. §5.1 does disclose the 0.798.
- **Fix:** For example, "collapsed far more often at the default learning rate (6 of 20 vs 1 of 20 seed–leaf combinations on TON-IoT)".

### O5. Highlight 1 is false on N-BaIoT
- **Quote (Highlight 1):** "Up to about 100 KB, int8 CNNs beat size-matched tree ensembles"
- **Problem:** The highlight is unqualified, but it is false on N-BaIoT:
  - RF 25 trees, depth 16 (93 KB): **0.9996**
  - int8 CNN of width 2 (106 KB): **0.9982**
  - Difference +0.001 in favour of the forest, Welch p = 0.010 (`N_BaIoT_tree_le_cnn`, `N_BaIoT_tree_le_cnn_welch_p` in values.json)
  - RF 25 trees, depth 12 (78 KB) also scores 0.999.
- **Consistency:** The abstract, §6.1 and §7 restrict the claim to TON-IoT and UNSW-NB15. The highlight does not.
- **Fix:** Add "on TON-IoT and UNSW-NB15" (fits within 85 characters if shortened, e.g. "Up to ~100 KB, int8 CNNs beat size-matched trees on TON-IoT and UNSW-NB15").

### O6. "Smallest within 0.01 is a network on TON-IoT" passes the threshold by 0.00003
- **Quote (§5.2):** "The smallest configuration within 0.01 of the best is therefore a network on TON-IoT, CNN width 2 (int8) (106 KB, macro-F1 0.951)"
- **Problem:** The threshold is 0.96145 − 0.01 = 0.95145, and the CNN w2 int8 mean is 0.95148. The margin of **3 × 10⁻⁵** is 30 times smaller than its seed SD (0.0009). A different best model (gb05_l63, 0.96141, statistically tied) would move the threshold by 4 × 10⁻⁵. verify.py asserts this result as if it were robust.
- **Neighbouring points:** MLP w2 int8 (61 KB, 0.94984) misses by 0.0016. RF 25 trees, depth 16 (1,028 KB, 0.953) is in.
- **Fix:** Say the CNN "sits exactly at the 0.01 boundary", or report the rule's result with that caveat.

### O7. "Timed on GPUs" generalises beyond the cited literature
- **Quote (Abstract, first sentence):** "Lightweight intrusion detectors for the Internet of Things are usually compact neural networks, justified by comparison with tree ensembles of unbounded size and timed on GPUs."
- **Problem:** §2.1 supports the first half (no neural study compared against size-limited trees). It contradicts "timed on GPUs" as the usual practice: Musthafa et al. use a Raspberry Pi, Ngo et al. an MCU and FPGA, Mazinani et al. an STM32, and Ullah et al. a Cortex-M4 estimate. The only GPU timing documented is the authors' own earlier version.
- **Fix:** For example, "…and often timed on GPUs or workstations", or drop that clause.

---

## AMBIGUOUS

### A1. The GBDT iteration cap was reached but this is not disclosed
- **Quotes:**
  - **§4.1:** "up to 1,000 iterations with early stopping after 20 iterations without improvement"
  - **§5.4:** "(1,000 boosting iterations, one tree per class)"
- **Problem:** Early stopping often did not trigger (`n_iter` in the seed files):
  - **UNSW-NB15, lr 0.05:** 14 of 20 runs hit the 1,000 cap (63 leaves 5/5, 31 leaves 4/5, 7 leaves 3/5, 15 leaves 2/5).
  - **TON-IoT, lr 0.05:** 4 of 20 (7 leaves 3/5, 15 leaves 1/5).
  - **TON best GBDT seed 0:** n_iter = 1,000.
- **Effect:** GBDT sizes, latencies and possibly accuracy are set by the cap, not by convergence.
- **Fix:** Add one sentence in §4.1 or §6.3.

### A2. The CNN at width 0.5 is left out of the int8 speed statement
- **Quote (§5.4):** "Int8 makes the MLPs of width 0.5 and 1 faster (0.05–0.29 ms) but not the wider models"
- **Problem:** The sentence says nothing about the CNN at width 0.5. In int8 it is faster on TON-IoT and UNSW-NB15 (0.213 / 0.138 ms vs 0.401 / 0.416 ms) and slower on N-BaIoT (0.740 vs 0.409 ms) (Table 3). A reader cannot tell where the CNN at width 0.5 belongs.

### A3. Agreement rates are seed means, but this is not stated
- **Quote (§5.3):** "agree with their float32 versions on at least 99.4% of TON-IoT test records and 89.7% of UNSW-NB15's"
- **Problem:** These are minima over configurations of the seed-mean agreement. The per-seed minima are **98.9%** (TON CNN w0.5) and **85.3%** (UNSW MLP w1).
- **Fix:** Add "(mean over seeds)".

### A4. The 350 KB earlier CNN vs the 94 KB present CNN is unexplained
- **Quotes:**
  - **§1:** "It recommended a 350 KB CNN over a 95 MB random forest"
  - **§5.5:** "the 95 MB that the earlier version compared with a 350 KB CNN"
  - **§4.1:** "the compact network of the earlier version … (23,946 parameters on TON-IoT)"
- **Problem:** 23,946 parameters × 4 bytes = 94 KB, as in Table 2. The 350 KB was the earlier on-disk Keras file (Paper1_v1_text.md, lines 221 and 269: "350 KB on disk"). The text never says so. The §5.5 conclusion ("came mostly from comparing it with an oversized forest in its pickled form") should note that the CNN side was also a serialised size (3.7× its parameter bytes).

### A5. The 21-of-29 count does not match what the appendix displays
- **Quote (§5.1):** "On N-BaIoT 21 of the 29 configurations reach at least 0.998"
- **Problem:** Table A.1 shows MLP width 1 (float32) on N-BaIoT as "0.998 ± 0.001" (unrounded 0.99768), which is not counted. A reader counting rows ≥ 0.998 in the table gets **22**. Either say "21 … reach a mean of at least 0.998 before rounding" or use a threshold that survives rounding.

### A6. A plus sign is used with "below", and a minus sign with "falls by"
- **Quotes:**
  - **§5.1:** "The best network is +0.009 below on TON-IoT … and +0.043 below on UNSW-NB15"
  - **§5.6:** "macro-F1 on TON-IoT falls by −0.015 for the medium forest, −0.015 … −0.015 … −0.014"
- **Problem:** "+x below" and "falls by −x" are double signs.
- **Fix:** Use unsigned values (e.g. "0.009 below", "falls by 0.015"), or change the verbs.

### A7. Template artefacts left in the filled text
- **§5.6:** "Ransomware loses most in every model (for the earlier forest ransomware (−0.06))". This comes from the `{{TON_noports_rf_v1_worst}}` format `'ransomware (−0.06)'` and reads as a draft remnant. Suggested: "(by 0.06 F1 for the earlier forest)".
- **§5.2:** "the largest int8 network, CNN width 2 (int8) (106 KB, 0.951) on TON-IoT" and "the int8 CNN (CNN width 2 (int8) (106 KB, 0.563))". The nested and doubled parentheses come from the label and value formats.
- **§5.1:** "GBDT, 15 leaves, lr 0.05, 0.961" inside parentheses. This reads as a list of four items.

### A8. Table headers and captions do not describe their columns
- **Table 2:** The second column is headed "Size" but holds the encoding ("compact", "float32", "int8"). The actual sizes are in the "… size (KB)" columns. Suggested header: "Encoding".
- **Table A.1:** The caption says "compact size and pickled scikit-learn size", but for networks the "Compact size (KB)" column is 4 B/parameter (float32) or the TFLite file (int8). The Table 2 caption explains this; A.1 does not. The A.1 caption also does not say that the lr-0.1 GBDTs are included, although Figure 1 excludes them.

### A9. The UNSW clean-record lead compares against the float32 network
- **Quote (§5.1):** "on those records the best tree leads the best network by … +0.071 on UNSW-NB15"
- **Problem:** On UNSW-NB15 the best network is the **int8** CNN w2 (0.563). The raw files hold no clean-record F1 for int8 models, so compute_values.py used the **float32** CNN w2 clean F1 (0.5595; `UNSW_NB15_clean_net_f1`).
- **Effect:** The conclusion is unaffected, but the comparison is not like-for-like.
- **Fix:** State that the float32 version was used, or compute the int8 clean F1.

### A10. The 8–12× pickle ratio is attributed to the unbounded forest
- **Quote (Abstract):** "an unbounded forest was never needed, and its pickled file was 8–12 times larger than a compact encoding"
- **Problem:** 8–12 is the range over *all* tree configurations (points.json). Both extremes are GBDTs (8.1 for UNSW gb05, 12.2 for N-BaIoT gb_l7). For random forests, including the unbounded forest, the ratio is **9.6–9.9**.
- **§4.2:** The explanation ("since scikit-learn stores a 64-bit value per class for every node") applies to random forests, not to the histogram GBDT, yet §4.2 attaches it to the 8–12 range for all "trees".
- **Fix:** In the abstract, say "about ten times".

### A11. Figure 1 does not show every configuration
- **Quote (§5.2):** "Figure 1 places every configuration on the accuracy–footprint plane."
- **Problem:** The figure omits the 8 GBDT configurations at lr 0.1 per dataset. The caption says so, and make_figures.py filters `gb_`.
- **Effect:** The UNSW overtaking point named in the same paragraph ("GBDT, 7 leaves, lr 0.1 (216 KB, 0.589)") is not visible in the figure.
- **Fix:** "every configuration except the lr-0.1 GBDTs", or name an lr-0.05 point instead.

---

## OK-but-note

### N1. The "highest mean" is a statistical tie
- **Quote (§5.1):** "Gradient-boosted trees at learning rate 0.05 have the highest mean macro-F1 on TON-IoT … and on UNSW-NB15"
- **Note:** Literally true, but the lead is within noise:
  - **TON-IoT:** gb05_l63 0.96141 (p = 0.95); unbounded forest 0.96028 (p = 0.12).
  - **UNSW-NB15:** unbounded forest 0.60558 (p = 0.99); earlier forest 0.60517 (p = 0.86).
- **Suggestion:** "Tree ensembles of 1.5 MB or more tie at the top; the best mean is a GBDT".

### N2. Welch tests are reported for only two kinds of comparison
- **§4.4 says:** "Configurations are compared with Welch's t-test over seeds".
- **Actually reported:** p-values only for best tree vs best network (§5.1) and in Table 4.
- **Comparisons without a reported test, with my p-values from the seed files:**
  - budget comparisons: TON p = 2.5e-9, UNSW p = 2.0e-4
  - TON overtaking forest vs best network: p = 0.008
  - min–max controls: TON p = 2.8e-5, UNSW p = 0.021
  - best GBDT vs earlier forest: p = 7.7e-5
- **Note:** All support the text. Consider adding them, or wording §4.4 as "selected comparisons".

### N3. "One to two orders of magnitude"
- **Quote (§5.4):** "costs one to two orders of magnitude in per-record latency"
- **Note:** The ratios are 55× (UNSW-NB15) and 236× (TON-IoT); the latter is above two orders. After O1 the TON tie-range is 61–236×. Acceptable as written.

### N4. "With one to two megabytes" understates UNSW-NB15
- **Quotes:** "With one to two megabytes, gradient-boosted trees beat every network" (abstract, §6.1, §7, Highlight 2).
- **Note:**
  - On UNSW-NB15, a GBDT already beats every network at **216 KB** (7 leaves, lr 0.1, 0.589 vs 0.563), and at 818 KB with lr 0.05 (0.601).
  - On TON-IoT the first tree above the best network is a forest at 1,028 KB (0.953 vs 0.952, p = 0.008).
- **Effect:** The statement is conservative, not wrong.

### N5. "Almost independent of width"
- **Quote (§5.4):** "The networks take 0.35–0.60 ms per record in float32, almost independent of width"
- **Note:** True on TON-IoT and UNSW-NB15. On N-BaIoT the MLP rises from 0.388 ms (width 0.5) to 0.602 ms (width 2), +55%.

### N6. "About 0.015" for the port ablation
- **Quote (abstract, §7):** "Removing port numbers cost about 0.015 macro-F1 on TON-IoT"
- **Note:** The range is 0.013–0.015 over five models. The lr-0.1 GBDT changes by +0.007 (p = 0.77, not significant), which is disclosed in §5.6. Fine.

### N7. Displayed differences vs differences of displayed means
Each case below is the unrounded difference, which is correct but does not equal the difference of the rounded means:
- **Table 4, MLP width 1:** shows 0.948 → 0.935 with difference −0.014 (true value −0.0139).
- **§5.3:** "0.016 on N-BaIoT on average" while Table 2 / A.1 show 0.999 → 0.982 (true value 0.0163).
- **Table A.2, TON 15 leaves:** shows 0.827 / 0.961 with difference +0.135 (true value 0.1349).

Readers sometimes flag this; a footnote "differences computed before rounding" would pre-empt it.

### N8. Part of the earlier CNN's gap is unattributed
- **§5.5:** The earlier CNN scored 0.880. The same architecture on min–max inputs in the present pipeline scores 0.900, so 0.020 of the 0.067 gap comes from other pipeline changes. These are not attributed.
- **Note:** "The earlier comparison therefore understated the CNN" remains supported.

### N9. Internal header in the filled md files
- Line 1 of Manuscript_filled.md and Manuscript_filled_anon.md is an internal header: "# Paper 1 v2 manuscript (double-brace placeholders are filled from values.json by build.py)".
- build.py skips it, so it is **not** in either docx (checked). It matters only if the md files themselves are shared.

### Checks with no issue found (for the record)
- **Placeholders and markers:** no ⟦…⟧, {{…}}, TODO or TBD anywhere in the filled md files or the docx.
- **Highlights:** within the length limit and identical to Highlights.docx.
- **Abstract:** 237 words (limit 250).
- **Table captions** match their contents for Tables 1, 3, 4 and A.2. Table 1 classes, features and record counts match data_stats via values.json. Table 3 has all 29 configurations and per-record-only int8 rows, as its caption says.
- **Figure 1** was regenerated with the current points.json (same timestamp). The LR-on-lower-edge rule applies only on UNSW-NB15, as the caption says.
- **Sizes:** float32 network and LR compact sizes re-derived by hand: TON CNN 94 KB, MLP 67 KB, LR 1.0 KB; UNSW MLP 60 KB, LR 0.4 KB; N-BaIoT CNN 93 KB, MLP 112 KB, LR 4.1 KB. All correct.
- **Counts:** "29 points (17 tree and LR configurations, and six networks in float32 and int8)" is correct.
- **verify.py's blind spots:** it does not check W1, W2, O2, O5 or O6. It passes O6 only by the 3 × 10⁻⁵ margin.

---

## Counts

| Class | Count | Items |
|---|---|---|
| WRONG | 2 | W1, W2 |
| OVERSTATED | 7 | O1–O7 |
| AMBIGUOUS | 11 | A1–A11 |
| OK-but-note | 9 | N1–N9 |
| **Total** | **29** | |
