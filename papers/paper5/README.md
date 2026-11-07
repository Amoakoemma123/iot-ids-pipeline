# Paper 5 manuscript scripts

* `Manuscript.md`: the manuscript text with `{{placeholders}}`.
* `Supplement.md`: the Supplementary Material, with placeholders from the same `values.json`.
* `compute_values.py`: computes every placeholder from the result files in `experiments/` and writes `values.json`.
  A few values (dataset sizes, the earlier-pipeline comparison) need the datasets or the earlier results, which are not
  in this repository; `values.json` holds all values as used.
* `Manuscript_filled.md`, `Supplement_filled.md`: the texts with the values inserted.
* `verify.py`: checks each qualitative claim of the text against `values.json` and the result files. Run
  `python papers/paper5/verify.py` from the repository root; it exits with the number of failed checks.
