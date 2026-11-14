# Paper 2 manuscript scripts

* `Manuscript.md`, `Supplement.md`: the manuscript and supplement text with `{{placeholders}}`.
* `compute_values.py`: computes every placeholder from the result files in `experiments/paper2_adv_v2/`
  (`results_v3/` = trained models and the realistic, feature-space and white-box attacks; `results_r2/` = the
  evaluations of the saved models: matched-FPR operating point, BPDA and action-space PGD, consistency-relaxed attack,
  feasibility, tree saturation, union of attacks) and writes `values.json` and `asr.json`.
* `Manuscript_filled.md`, `Supplement_filled.md`: the text with the values inserted.
* `make_figures.py`, `build.py`, `docx_helpers.py`: Figures 1-4 and the Word files
  (`build.py [--anon]` = manuscript, `build.py --supp [--anon]` = supplement).
* `verify.py`: checks every quantitative and qualitative claim against `values.json`, the result files and the
  experiment code; `check_citations.py` matches citations and references.

Run from the repository root:
`python papers/paper2/compute_values.py`, `python papers/paper2/make_figures.py`, `python papers/paper2/build.py`,
`python papers/paper2/build.py --anon`, `python papers/paper2/build.py --supp`, `python papers/paper2/build.py --supp --anon`,
`python papers/paper2/verify.py` (exit code = number of failed checks).

How the experiments were run (order of the groups, compute time) and the SHA-256 hashes of the 210 trained models:
`experiments/paper2_adv_v2/REPRODUCE.md` and `MODELS_SHA256.txt`. The models (1.5 GB) are available from the
authors on request. `run_adv_v3.py` imports the Paper 1 and Paper 6 training code
(`experiments/baseline/run_p1_v2.py`, `experiments/paper6_multimodal/run_p6_v2.py`).
