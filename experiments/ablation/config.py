"""
experiments/ablation/config.py
Configuration for the 2x2 factorial ablation study.

Conditions (matching Baseline_LWIDS/results_ablation.csv):
  A: MI filter + RFECV + SMOTE          (full pipeline)
  B: MI filter + RFECV, no SMOTE        (feature selection only)
  C: MI filter + SMOTE, no RFECV        (SMOTE only, no RFECV)
  D: All features, no SMOTE             (raw features, no oversampling)
"""
from pathlib import Path

BASE_DIR    = Path(__file__).resolve().parents[2]
DATA_DIR    = BASE_DIR / 'dataset'
RESULTS_DIR = Path(__file__).resolve().parent / 'results'
MODELS_DIR  = RESULTS_DIR / 'models'

# Ablation runs only on TON_IoT (paper's primary dataset)
DATASET     = (DATA_DIR / 'train_test_network.csv', 'type')

SEED = 42

SPLIT = dict(test_size=0.15, val_size=0.15, seed=SEED)

# ── Condition definitions ─────────────────────────────────────────────────────
# Each condition is a dict of kwargs forwarded to PreprocessingPipeline
# plus 'apply_smote' and 'skip_rfecv' flags.

CONDITIONS = {
    'A': dict(
        description     = 'MI+RFECV+SMOTE (Full pipeline)',
        mi_threshold    = 0.05,
        rfecv_subsample = 50_000,
        apply_smote     = True,
        skip_rfecv      = False,
        smote_k         = 5,
        smote_max_per_class = None,
        seed            = SEED,
    ),
    'B': dict(
        description     = 'MI+RFECV, no SMOTE',
        mi_threshold    = 0.05,
        rfecv_subsample = 50_000,
        apply_smote     = False,
        skip_rfecv      = False,
        smote_k         = 5,
        smote_max_per_class = None,
        seed            = SEED,
    ),
    'C': dict(
        description     = 'MI+SMOTE, no RFECV',
        mi_threshold    = 0.05,
        rfecv_subsample = 50_000,
        apply_smote     = True,
        skip_rfecv      = True,
        smote_k         = 5,
        smote_max_per_class = None,
        seed            = SEED,
    ),
    'D': dict(
        description     = 'All features, no SMOTE',
        mi_threshold    = 0.0,   # keep all features (effectively no MI filter)
        rfecv_subsample = 50_000,
        apply_smote     = False,
        skip_rfecv      = True,
        smote_k         = 5,
        smote_max_per_class = None,
        seed            = SEED,
    ),
}

# ── Model training params (same as baseline) ──────────────────────────────────
LR_PARAMS  = dict(seed=SEED)
RF_PARAMS  = dict(seed=SEED)
CNN_TRAIN  = dict(epochs=100, batch_size=256, patience=15)
LSTM_TRAIN = dict(window=10, epochs=100, batch_size=128, patience=15,
                  subsample=80_000, device='/CPU:0')
