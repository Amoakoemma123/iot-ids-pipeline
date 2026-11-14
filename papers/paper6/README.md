# Paper 6 manuscript scripts

* `Manuscript.md`: the manuscript text with `{{placeholders}}`; `Manuscript_filled.md`: with the values inserted.
* `compute_values.py`: computes every placeholder from the result files in `experiments/paper6_multimodal/`
  (`results_v4/` = final runs; `results_p6gb/` = the GBDT runs re-evaluated at their best internal-validation
  iteration by `run_p6_gb_rollback.py`, used in place of the `results_v4` GBDT runs; `results_v2/` and `results_v3/` =
  the min-max vs quantile input-scaling comparison) and from the Paper 1 random-forest results in
  `experiments/baseline/results_v2/`. Writes `values.json`.
* `data_stats.json`: dataset statistics used in the text; `rfecv_scores.json`: cross-validated RFECV scores of the
  fitted Paper 1 pre-processing pipelines (the pipelines themselves, 13 MB each, are not in the repository;
  `compute_values.py` refreshes this file when they are present).
* `make_figures.py`, `build.py`, `docx_helpers.py`: figures and the Word manuscript (`build.py [--anon]`).
* `verify.py`: checks every quantitative and qualitative claim against `values.json` and the result files;
  `check_citations.py` matches citations and references.

Run from the repository root: `python papers/paper6/compute_values.py`, `python papers/paper6/make_figures.py`,
`python papers/paper6/build.py`, `python papers/paper6/build.py --anon`, `python papers/paper6/verify.py`
(exit code = number of failed checks).

Experiment code: `experiments/paper6_multimodal/run_p6_v2.py` (training and evaluation),
`run_p6_gb_rollback.py` (GBDT rollback, uses `experiments/baseline/gbdt_rollback.py`), `bench_latency.py` (CPU latency).
