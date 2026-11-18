# Paper 7: failure modes of the re-run series

"What Re-running a Machine-Learning IoT Intrusion Detection Series Taught Us: Failure Modes and the Checks That Catch Them"
(submitted to the International Journal of Information Security).

* `Manuscript.md` - the text with `{{PX:key}}` placeholders; `Manuscript_filled.md` - filled by `build.py` from `values.json`.
* `compute_values.py` - copies every value of Papers 1-6 (`papers/paper1..6/values.json`) as `PX:key` and computes the
  `P7:` values, mostly from the original (v1) result files in `original_v1/experiments/`. The original model files are
  not included (the forest pickle is ~100 MB); their sizes and SHA-256 are in
  `original_v1/experiments/baseline/results/models/model_file_sizes.json`.
* `verify.py` - checks every qualitative statement of the paper against `values.json` (0 failures at release).
* `checks/ids_checks.py` - the check library of Section 6 (plain arrays, no dependency on this pipeline).
* `checks/reproduce_headline.py` - reproduces the evidence for F4, F5, F7 and F17 in seconds on a CPU:
  `python papers/paper7/checks/reproduce_headline.py` from the repository root.
* `review_rounds_table.md` - per-report outcomes behind Table 4; the review reports of Studies 1-6 are in
  `papers/paper<k>/reviews/`.

**Not released:** the external audit of Section 5.3 (coding matrices, per-repository evidence, cloned repositories) is
reported anonymously and is available to the editor and reviewers on request. Without it, `compute_values.py` keeps the
released `P7:ext_*` values. Paper 7's own review reports are withheld for the same reason, because rounds written before
the audit was anonymised name the audited repositories.

Run from this folder: `python compute_values.py && python build.py && python verify.py`.
