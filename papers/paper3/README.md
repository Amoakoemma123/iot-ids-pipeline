# Paper 3 manuscript scripts

* `Manuscript.md`: the manuscript text with `{{placeholders}}`.
* `compute_values.py`: computes every placeholder from the result files in `experiments/paper3_continual/`
  (`results_v3/` and the final round `results_p3final/`) and writes `values.json`.
* `Manuscript_filled.md`: the text with the values inserted.
* `make_figures.py`, `build.py`, `docx_helpers.py`: Figure 1 and the Word manuscript (`build.py` writes
  `Manuscript_v2.docx`, `Manuscript_v2_anon.docx` and `Highlights.docx`).
* `verify.py`: checks every quantitative and qualitative claim of the built Word files against `values.json` and the
  result files. Run from the repository root, after `build.py` and `build.py --anon`:
  `python papers/paper3/compute_values.py`, `python papers/paper3/build.py`, `python papers/paper3/build.py --anon`,
  `python papers/paper3/verify.py` (exit code = number of failed checks).

The final review round (`experiments/paper3_continual/results_p3final/<dataset>/<group>/`) holds the corrected
A-GEM (group `agemfix`), the replay-weighting ablation ER-2000-pool (`replayabl`), the random forest retrained on the
replay memory and the all-data forest per stage (`rfmem`) and the range-share diagnostic (`rangeshare`);
queue: `experiments/paper3_continual/p3final_queue.sh` (run from the repository root; written for the authors' tree, where it lives in `logs/`).
