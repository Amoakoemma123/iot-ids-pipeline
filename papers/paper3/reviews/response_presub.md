# Paper 3: response to the pre-submission referee report (review_presub_referee.md, 6/10)

## Wrong items
1. **"RF beat CNN replay on all three".** Now compared with both replay variants. RF + memory is above ER-2000 on all three datasets and above ER-2000-pool on TON-IoT and N-BaIoT. It is below ER-2000-pool on UNSW-NB15 (−0.024, p < 0.001). Updated in the abstract (generated verdict), §5.2, §6.1, the conclusion, highlight 2 ("two of three datasets") and the cover letter.
2. **"Numerically ahead (not tested)".** Removed. The paired test is now reported in §5.2.
3. **"Once per task".** Now worded "added to each task's training pool instead of every minibatch" in the abstract, and "Memory in the training pool, not in every batch" in highlight 4. §4.3 now also explains how this changes how often the memory is revisited.
4. **Conclusion "largely".** Changed to "partly". Added that final macro-F1 rose clearly only on UNSW-NB15.
5. **§5.3 apportioning of the false-positive cost.** Rewritten: pooling alone reaches the joint level, and the half-benign memory removes 57%. Both act on the same effective benign weight and were not combined, so the cost is no longer apportioned. The cost in earlier-task retention is now stated. §6.2 says the two were not combined.
6. **"0.00%".** Small shares now keep two significant digits (UNSW 0.0037%). The later-task training share has been added.

## Majors
1. **RF compared with only one replay variant.** Addressed under wrong item 1.
2. **RF detection level.** RF + memory's earlier-task detection is now reported in §5.2: UNSW 0.702, and only 0.499 on Task 1, against 0.998 for ER-2000. The low false-positive rate is tied to the missed attacks there.
3. **Framing.**
   - §6.1 now reads "best option for the network".
   - §5.2 and the conclusion state that RF + memory is above CNN joint retraining on TON-IoT and N-BaIoT.
   - The RF size and single-thread prediction time are measured against the network (new script bench_p3_size.py, results_p3final/size_bench.json, sentence in §6.1). This runs gated on RAM; verify keeps it pending until the run is done.
   - The contribution list now includes the replay-weighting finding.
4. **Weighting attribution.** Addressed under wrong item 5.
5. **Conclusion stronger than the results.** Addressed under wrong item 4.
6. **Method described wrongly.** Addressed under wrong item 3.
7. **Old conclusions still in the text.**
   - §4.2 "slightly favours" now points to the replay-weighting check.
   - §5.2 now reads "Replay reduced forgetting most".
   - The A-GEM mechanism sentence is now per dataset, with the N-BaIoT seed range.
   - The §6.1 false-positive bullet and the "unusable" sentence now include the pooled result.
   - The §6.2 iCaRL sentence has been reworded.
8. **Systems evidence.** Partly addressed: RF and CNN size and prediction time are now given (major 3). There is still no edge-device measurement; §6.1 and §6.2 already say so.

## Minors
- Fixed: 1–12 (minor 13 does not reach the docx; minor 14: the abstract is now 243 words after the not-yet-learned sentence was cut).
- Not done, as before: the residual items (MLP check of the kernel-3 assumption, AUROC, benign-share sweep, DOI tag, a 2026 literature re-check), which are acknowledged as limitations.
