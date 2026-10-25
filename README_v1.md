# IREAP IoT IDS — Papers Pipeline: Code, Models & Experiment Results

**Author:** Amoako Kwame Emmanuel
**Affiliation:** Independent Researcher, Sunyani, Ghana
**ORCID:** [0009-0004-3477-6832](https://orcid.org/0009-0004-3477-6832)
**Date:** August 2026
**Zenodo DOI:** https://doi.org/10.5281/zenodo.21843194

---

## Overview

This deposit contains the complete source code, trained models, experiment pipelines,
run logs, and numerical results for a seven-paper research programme on intelligent
intrusion detection for resource-constrained IoT environments.

The programme is built around the **IREAP** framework:
**I**ntelligent, **R**obust, **E**xplainable, **A**daptive, and
**P**rivacy-preserving IoT Intrusion Detection.

---

## Folder Structure

```
Papers pipeline/
├── src/                          # Shared library (all papers)
│   ├── data/
│   │   ├── loader.py             # Dataset loader with OHE fix + stratified split
│   │   └── preprocessing.py     # MI selection, RFECV, SMOTE pipeline
│   ├── models/
│   │   ├── cnn.py                # 1D-CNN model definition
│   │   ├── lstm.py               # LSTM model definition
│   │   ├── random_forest.py      # RF wrapper
│   │   ├── logistic_regression.py
│   │   ├── autoencoder_rf.py     # AE-RF hybrid
│   │   └── continual_cnn.py      # CNN with continual learning support
│   ├── evaluation/
│   │   ├── metrics.py            # F1, accuracy, per-class metrics
│   │   ├── latency.py            # Inference latency benchmarking
│   │   └── cl_metrics.py         # Continual learning metrics (FM, BWT)
│   ├── training/
│   │   └── trainer.py            # Generic training loop
│   └── explainability/
│       └── shap_analysis.py      # SHAP global + interaction analysis
│
├── experiments/
│   ├── baseline/                 # Paper 1 — Lightweight IDS Comparison
│   │   ├── config.py             # Per-dataset hyperparameter overrides
│   │   ├── run_experiment.py     # Main runner (all 4 datasets)
│   │   ├── eval_only.py          # Evaluation + SHAP + figures (no retraining)
│   │   └── results/              # Trained models (.h5, .pkl) + metrics JSON
│   │
│   ├── ablation/                 # Paper 1 — Ablation Study (A/B/C/D conditions)
│   │   ├── config.py
│   │   ├── run_ablation.py
│   │   └── results/
│   │       ├── condition_A/      # Full pipeline (MI+RFECV+SMOTE)
│   │       ├── condition_B/      # MI+RFECV, no SMOTE
│   │       ├── condition_C/      # MI+SMOTE, no RFECV
│   │       ├── condition_D/      # All features, no SMOTE
│   │       └── figures/          # Ablation bar charts
│   │
│   ├── adversarial_robustness/   # Paper 2 — Adversarial Robustness
│   │   ├── attacks/
│   │   │   ├── pgd.py            # PGD attack
│   │   │   ├── fgsm.py           # FGSM attack
│   │   │   └── cw.py             # Carlini-Wagner attack
│   │   ├── defenses/
│   │   │   ├── ccat.py           # Class-Conditional Adversarial Training
│   │   │   ├── adversarial_training.py
│   │   │   └── feature_squeezing.py
│   │   ├── run_robustness.py     # Main robustness evaluation
│   │   ├── run_ccat.py           # CCAT training (all datasets)
│   │   ├── run_ccat_sensitivity.py
│   │   ├── run_confidence_intervals.py
│   │   └── results/
│   │       ├── TON_IoT/          # CCAT, TRADES, robustness JSON + model .h5
│   │       ├── UNSW_NB15/
│   │       └── CICIDS2017/
│   │
│   ├── transfer_attack/          # Paper 2 — Transfer Attack Supplement
│   │   ├── run_transfer.py
│   │   └── results/transfer_results.json
│   │
│   ├── continual_learning/       # Paper 3 — Continual Learning
│   │   ├── config.py
│   │   ├── run_continual.py      # Replay, EWC, LwF, GEM evaluation
│   │   └── results/
│   │       ├── TON_IoT/
│   │       ├── UNSW_NB15/
│   │       └── CICIDS2017/
│   │
│   ├── paper4_federated/         # Paper 4 — Federated Learning
│   │   ├── run_federated.py      # FedAvg, SCAFFOLD, DP evaluation
│   │   ├── run_extra_experiments.py
│   │   └── results/
│   │
│   ├── paper5_xai/               # Paper 5 — Multi-Level XAI
│   │   ├── run_xai_experiments.py  # SHAP, LIME, DiCE, Jaccard consistency
│   │   └── results/
│   │
│   ├── paper6_multimodal/        # Paper 6 — Dual-Stream Multi-Modal IDS
│   │   ├── run_multimodal_experiments.py
│   │   ├── run_multiseed.py
│   │   ├── extract_attention_perclass.py
│   │   └── results/
│   │
│   └── paper7_integrated/        # Paper 7 — Full IREAP Framework
│       ├── unified_pipeline.py   # IREAP modules (CCAT, FedAvg, Replay, SHAP)
│       ├── run_integrated.py     # C0–C4 ablation (Baseline → Full IREAP)
│       ├── run_multiseed.py      # Multi-seed CIs (C2/C3, seeds 42/123/456)
│       ├── run_multiseed_c4.py   # Multi-seed CIs (C4, seeds 42/123/456)
│       └── results/
│           ├── TON_IoT/
│           └── UNSW_NB15/
│
├── logs/                         # Full training logs for all papers
├── IDS_IoT_Training.ipynb        # Interactive training notebook
├── run_hparam_sweep.py           # Hyperparameter sweep (Paper 1)
├── run_multi_seed.py             # Multi-seed runner (Papers 2, 4)
├── run_rf_adversarial.py         # RF adversarial evaluation (Paper 2)
├── run_trades.py                 # TRADES adversarial training (Paper 2)
└── generate_training_figures.py  # Training curve figure generator
```

---

## Paper Mapping

| Experiment Folder | Paper | Key Method |
|-------------------|-------|------------|
| `experiments/baseline/` | Paper 1 | CNN / LSTM / RF / AE-RF comparison |
| `experiments/ablation/` | Paper 1 | MI, RFECV, SMOTE ablation (A/B/C/D) |
| `experiments/adversarial_robustness/` | Paper 2 | CCAT, TRADES, PGD, FGSM, C&W |
| `experiments/transfer_attack/` | Paper 2 | Cross-architecture transfer attacks |
| `experiments/continual_learning/` | Paper 3 | Replay, EWC, LwF, GEM |
| `experiments/paper4_federated/` | Paper 4 | FedAvg, SCAFFOLD, Differential Privacy |
| `experiments/paper5_xai/` | Paper 5 | SHAP, LIME, DiCE, Jaccard consistency |
| `experiments/paper6_multimodal/` | Paper 6 | Dual-stream flow+behavioural fusion |
| `experiments/paper7_integrated/` | Paper 7 | Full IREAP (C0→C4 ablation) |

---

## Datasets (not included — publicly available)

| Dataset | Rows | Source |
|---------|------|--------|
| TON_IoT | 211,043 | https://research.unsw.edu.au/projects/toniot-datasets |
| UNSW-NB15 | 175,341 | https://research.unsw.edu.au/projects/unsw-nb15-dataset |
| CICIDS-2017 | 2,830,743 | https://www.unb.ca/cic/datasets/ids-2017.html |
| NSL-KDD | 125,973 | https://www.unb.ca/cic/datasets/nsl.html |

Place datasets as:
- `dataset/train_test_network.csv` — TON_IoT
- `dataset2/UNSW_NB15.csv` — UNSW-NB15
- `dataset3/IDS_datasets/dataset_cicids2017.parquet` — CICIDS-2017
- `dataset3/NSL_KDD.csv` — NSL-KDD

---

## Environment Setup

```bash
python -m venv ids_env
ids_env/Scripts/activate      # Windows
# ids_env/bin/activate        # Linux/macOS
pip install -r requirements.txt
```

**Key dependencies:** TensorFlow 2.x, scikit-learn, imbalanced-learn,
pandas, numpy, shap, lime, dice-ml, python-docx, matplotlib, pyarrow

---

## Reproducing Results

All results can be reproduced from scratch. Pre-trained models are also
included in each `results/models/` subdirectory — no retraining needed
for evaluation.

```bash
# Paper 1 — Baseline IDS (CNN/LSTM/RF/AE-RF, 4 datasets)
python -m experiments.baseline.run_experiment

# Paper 1 — Ablation Study
python -m experiments.ablation.run_ablation

# Paper 2 — Adversarial Robustness (CCAT + TRADES)
python -m experiments.adversarial_robustness.run_robustness
python -m experiments.adversarial_robustness.run_ccat         # CCAT training only
python -m experiments.adversarial_robustness.run_confidence_intervals

# Paper 3 — Continual Learning (Replay, EWC, LwF, GEM)
python -m experiments.continual_learning.run_continual

# Paper 4 — Federated Learning (FedAvg, SCAFFOLD, DP)
python -m experiments.paper4_federated.run_federated

# Paper 5 — Multi-Level XAI (SHAP, LIME, DiCE, Jaccard)
python -m experiments.paper5_xai.run_xai_experiments

# Paper 6 — Multi-Modal IDS (dual-stream, multi-seed)
python -m experiments.paper6_multimodal.run_multimodal_experiments
python -m experiments.paper6_multimodal.run_multiseed

# Paper 7 — Full IREAP (C0→C4, multi-seed)
python -m experiments.paper7_integrated.run_integrated
python -m experiments.paper7_integrated.run_multiseed          # C2/C3 CIs
python -m experiments.paper7_integrated.run_multiseed_c4       # C4 CIs
```

---

## Key Results Summary

### Paper 1 — Baseline IDS (TON_IoT, macro F1)
| Model | F1 | Params | Latency |
|-------|----|--------|---------|
| CNN | 0.8803 | 23,946 | 0.19 ms |
| LSTM | 0.7898 | 74,890 | 0.31 ms |
| RF | 0.9565 | — | 0.064 ms |
| AE-RF | 0.8216 | — | 0.21 ms |

### Paper 2 — Adversarial Robustness (CCAT, macro F1)
| Dataset | Standard F1 | CCAT F1 | Gain |
|---------|------------|---------|------|
| TON_IoT | 0.6800 | 0.8196 | +0.140 |
| UNSW-NB15 | 0.4940 | 0.4870 | −0.007 |
| CICIDS-2017 | 0.1800 | 0.5314 | +0.351 |

### Paper 7 — Full IREAP (C0→C4 ablation, seed=42)
| Condition | TON_IoT F1 | UNSW-NB15 F1 |
|-----------|-----------|-------------|
| C0: Baseline CNN | 0.8662 | 0.5504 |
| C1: +Multi-Modal | 0.7087 | 0.5702 |
| C2: +Adversarial (CCAT) | 0.7511 | 0.4961 |
| C3: +Federated (FedAvg) | 0.7114 | 0.4427 |
| C4: Full IREAP | 0.7455 | 0.4968 |

Multi-seed C4 (n=3): TON_IoT 0.5251 ± 0.2559 | UNSW-NB15 0.4838 ± 0.0197

---

## Pre-trained Models

Model files are included in each experiment's `results/models/` subdirectory:
- `.h5` files — TensorFlow/Keras neural network weights
- `.pkl` files — scikit-learn Random Forest and AE-RF models

No retraining is required to reproduce evaluation metrics and figures.

---

## Hardware

All experiments were run on:
- CPU: Intel Core i7 (11th Gen)
- GPU: NVIDIA GTX 1650 (4 GB VRAM)
- RAM: 16 GB
- OS: Windows 11

GPU memory growth is enabled automatically in all scripts via
`tf.config.experimental.set_memory_growth`.

---

## Citation

If you use this code or the associated papers in your research, please cite:

> Emmanuel, A. K. (2026). *IREAP IoT IDS Research Programme: Code, Trained Models,
> and Pipelines for Seven-Paper Series on Lightweight Deep Learning, Adversarial
> Robustness, Continual Learning, Federated Privacy, Explainability, Multi-Modal
> Fusion, and Integrated Framework Design*. Zenodo.
> https://doi.org/10.5281/zenodo.21843194

---

## License

This code is released under the **MIT License**.
The pre-trained model weights are released under **CC BY 4.0**.

---

## AI Disclosure

Portions of this codebase and documentation were prepared with assistance
from Claude (Anthropic). All scientific decisions, experimental designs,
and conclusions are solely those of the author.
