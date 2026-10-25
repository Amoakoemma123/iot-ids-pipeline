"""
experiments/adversarial_robustness/config.py
Configuration for adversarial robustness experiments (Paper 2).
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[2]
BASELINE_DIR = ROOT / 'experiments' / 'baseline'
RESULTS_DIR  = Path(__file__).resolve().parent / 'results'

# Datasets for Paper 2 (TON_IoT, UNSW_NB15, CICIDS2017 — CNN only)
DATASETS = ['TON_IoT', 'UNSW_NB15', 'CICIDS2017']

# Dataset file paths and label columns (mirror baseline config)
from experiments.baseline.config import DATASETS as BASELINE_DATASETS
DATASET_PATHS = {ds: BASELINE_DATASETS[ds] for ds in DATASETS}

# ── Attack parameters ──────────────────────────────────────────────────────────

# Epsilon values for FGSM and PGD sweep (features scaled to [0,1])
EPSILONS = [0.01, 0.05, 0.1, 0.2, 0.3]

# PGD evaluation: step size = epsilon * ALPHA_RATIO, clipped to ε-ball each iter
PGD_ALPHA_RATIO = 0.1    # alpha = epsilon * 0.1
PGD_N_ITER      = 40     # number of PGD steps for evaluation

# C&W L2 attack parameters
CW_CONFIDENCE     = 0      # κ: confidence margin (0 = just flip prediction)
CW_LEARNING_RATE  = 0.01   # Adam learning rate for the inner optimisation
CW_MAX_ITER       = 100    # inner optimisation steps per binary search round
CW_BINARY_STEPS   = 9     # binary search steps for constant c
CW_N_SAMPLES      = 1000  # number of test samples to attack (C&W is slow)
CW_INITIAL_CONST  = 1e-2  # initial c value

# ── Defense 1: PGD adversarial training ────────────────────────────────────────

ADV_TRAIN_EPSILON    = 0.1   # ε used to generate adversarial examples
ADV_TRAIN_RATIO      = 0.5   # fraction of each batch that is adversarial
ADV_TRAIN_EPOCHS     = 50    # max training epochs (EarlyStopping applies)
ADV_TRAIN_PATIENCE   = 10    # EarlyStopping patience
ADV_TRAIN_BATCH      = 256
ADV_TRAIN_PGD_STEPS  = 7     # PGD steps for inner maximisation (Madry et al.)
ADV_TRAIN_PGD_ALPHA_RATIO = 0.25   # alpha = epsilon * 0.25 (= epsilon/4)

# Per-dataset training subsample (None = use all SMOTE-balanced data)
# Cap all datasets at 100K SMOTE-balanced samples to keep PGD training epochs
# fast (eager PGD loop has high Python overhead; 100K ≈ 20 min/epoch on GTX 1650)
ADV_TRAIN_SUBSAMPLE = {
    'TON_IoT':    100_000,
    'UNSW_NB15':  100_000,
    'CICIDS2017': 100_000,
}

# ── Defense 2: Feature squeezing ───────────────────────────────────────────────

FS_N_BITS       = 4   # bit-depth reduction (2^4 = 16 quantisation levels)
FS_MEDIAN_WIN   = 3   # median-filter window size across feature vector

# ── Evaluation ─────────────────────────────────────────────────────────────────
EVAL_BATCH_SIZE = 512    # batch size for attack generation and inference
SEED = 42
