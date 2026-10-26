# IREAP IoT IDS — Papers Pipeline, version 2 (corrected pipeline and revised experiments)

This is version 2 of the code-and-results archive for the IoT intrusion-detection paper series by
E. K. Amoako et al. It supersedes version 1 (2026-08-07), whose UNSW-NB15 results were affected by a
data-pipeline error described below. Version 1 is archived on Zenodo:
https://doi.org/10.5281/zenodo.21844113

## What changed since version 1

* **Data pipeline (`src/data/loader.py`).** The UNSW-NB15 row index `id` was used as a feature and,
  because every row was unique, it also disabled duplicate removal. Version 2 drops identifier columns,
  removes duplicates after all column drops, fits every cleaning statistic on the training split only,
  and removes validation/test records identical to a training record. TON-IoT data are unchanged.
* **Paper 5 (explainability)** — `experiments/paper5_xai/run_xai_v3.py`, `leakage_decomposition*.py`,
  `feature_space_copies.py`; results in `experiments/paper5_xai/results_v3/` (and `results_v2/` for global
  SHAP and the port check).
* **Paper 4 (federated learning)** — rebuilt in `experiments/paper4_federated/run_fl_v2.py`: validation-based
  round selection, client partitions redrawn per repetition, tuned baselines, matched centralised and
  local-only references, group normalisation, record-level DP-SGD with an RDP accountant, natural
  device-level partitions (TON-IoT by destination host, redrawn per repetition; N-BaIoT one client per device),
  per-site evaluation on each site's own traffic, local fine-tuning, FedLC, and batch normalisation with federated
  re-estimation of its statistics. Results in `experiments/paper4_federated/results_v2/<dataset>/<experiment>/seed<r>.json`
  (superseded runs kept in `_bn_superseded/`, `_superseded_r2/` and `_stale/` for transparency). N-BaIoT is prepared by
  `experiments/paper4_federated/prepare_nbaiot.py` (3% sample per device–class file, seed 42; counts in `dataset4/`).
* Other papers of the series (1–3, 6, 7) are being revised; their scripts in this archive are the
  version-1 code and have not yet been re-run on the corrected pipeline.

## Not included

The datasets are not redistributed; obtain them from their providers and place them as configured in
`experiments/baseline/config.py`:
* TON_IoT network data — N. Moustafa, UNSW Canberra (`dataset/train_test_network.csv`)
* UNSW-NB15 training set — N. Moustafa and J. Slay (`dataset2/UNSW_NB15.csv`)
* CICIDS2017 — Canadian Institute for Cybersecurity
* N-BaIoT — Y. Meidan et al., UCI Machine Learning Repository dataset 442 (extract to `dataset4/raw/<device>/`)

Trained model files (`*.pkl`, `*.h5`) and run logs are not included; every result can be regenerated.

## Environment

Python 3.10, TensorFlow 2.10, scikit-learn, SHAP 0.49, LIME, dp_accounting, imbalanced-learn.
Experiments were run on a laptop (Intel i5-10200H, 16 GB RAM, NVIDIA GTX 1650 4 GB).

## Reproducing

From the archive root:

    python -m experiments.baseline.run_experiment --dataset UNSW_NB15        # corrected baseline + pipeline.pkl
    python -m experiments.paper5_xai.run_xai_v3 --dataset all                # Paper 5
    python -m experiments.paper4_federated.run_fl_v2 --dataset TON_IoT --group core --seeds 0,1,2,3,4   # Paper 4
    python -m experiments.paper4_federated.prepare_nbaiot                                           # N-BaIoT sample
    python -m experiments.paper4_federated.run_fl_v2 --dataset N_BaIoT --group device --seeds 0,1,2,3,4

See the docstrings of each script for the experiment groups.

## License

CC BY 4.0 (as version 1).
