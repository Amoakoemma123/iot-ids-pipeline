# Paper 1 manuscript scripts

* `Manuscript.md`: the manuscript text with `{{placeholders}}`; `Manuscript_filled.md`: with the values inserted.
* `compute_values.py`: computes every placeholder and the accuracy-size points (`points.json`) from the result files
  in `experiments/baseline/`: `results_v2/` (trees, networks, int8 models, port ablation; JSON per seed) and
  `results_p1final/` (final review round: GBDTs re-evaluated at their best validation iteration, capped GBDTs,
  small trees, validation-based selection, early stopping on the provided validation split, port ablation of the
  rolled-back GBDTs, and `latency.json`, the single-thread latency benchmark). Every GBDT, budget, selection and
  latency value is withheld until all final groups exist. Writes `values.json`.
  `results_v2/TON_IoT/gb_l7/model_seed0.pkl` is the one diverged GBDT shown in Section 4.1; `clean_split0.npy` marks
  the test records without a training match on the selected features.
* `make_figures.py`, `build.py`, `docx_helpers.py`: Figure 1 and the Word manuscript (`build.py [--anon]`).
* `verify.py`: checks every quantitative and qualitative claim against `values.json`, the result files and the code
  (`run_p1_v2.py`, `run_p1_final.py`, `bench_p1_final.py`, `gbdt_rollback.py` with its test
  `test_gbdt_rollback.py`); `check_citations.py` matches citations and references.

Run from the repository root: `python papers/paper1/compute_values.py`, `python papers/paper1/make_figures.py`,
`python papers/paper1/build.py`, `python papers/paper1/build.py --anon`, `python papers/paper1/verify.py`
(exit code = number of failed checks).

Experiments (run from the repository root): `python -m experiments.baseline.run_p1_v2 --dataset D --group trees|nets|ports`,
then `python -m experiments.baseline.run_p1_final --dataset D --group G` for the groups gb_rollback, gb_ports,
nets_val0, trees_val, gb_capped, small_trees, gb_extval, and `python -m experiments.baseline.bench_p1_final
--dataset D --lgbm-proxy` on an otherwise idle machine. The trained models (about 4.8 GB) are not in the repository.
