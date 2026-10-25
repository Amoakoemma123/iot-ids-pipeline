"""
experiments/baseline/config.py
Central configuration for the baseline experiment.
"""
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR     = Path(__file__).resolve().parents[2]   # repo root
DATA_DIR     = BASE_DIR / 'dataset'
RESULTS_DIR  = Path(__file__).resolve().parent / 'results'
MODELS_DIR   = RESULTS_DIR / 'models'

# Dataset CSV paths and their label columns: (csv_path, label_col)
DATASETS = {
    'TON_IoT':    (DATA_DIR / 'train_test_network.csv',      'type'),
    'CICIDS2017': (BASE_DIR / 'dataset3' / 'IDS_datasets' / 'dataset_cicids2017.parquet', 'Label'),
    'UNSW_NB15':  (BASE_DIR / 'dataset2' / 'UNSW_NB15.csv',  'attack_cat'),
    'NSL_KDD':    (BASE_DIR / 'dataset3' / 'NSL_KDD.csv',    'attack_type'),
}

# ── Reproducibility ───────────────────────────────────────────────────────────

SEED = 42

# ── Data split (70 / 15 / 15) ─────────────────────────────────────────────────

SPLIT = dict(test_size=0.15, val_size=0.15, seed=SEED)

# ── Preprocessing defaults ────────────────────────────────────────────────────

PREPROCESSING = dict(
    mi_threshold        = 0.05,
    rfecv_subsample     = 50_000,
    apply_smote         = True,
    smote_k             = 5,
    smote_max_per_class = None,   # None = no cap (standard SMOTE)
    seed                = SEED,
)

# Per-dataset overrides merged at runtime
PREPROCESSING_OVERRIDES = {
    # CICIDS2017: 2.83M rows (full dataset), 15 classes, label col='Label'.
    # mi_threshold=0.0: keep all features; mi_subsample=100K for speed on large data.
    # skip_rfecv=True: RFECV too slow/unstable on this scale.
    # smote_max_per_class=50K: oversample ultra-rare classes (Heartbleed=11,
    # SQL Injection=21, Infiltration=36) and mid-rare classes to 50K.
    # Large classes (BENIGN=2.27M, DoS Hulk=231K, PortScan=159K, DDoS=128K)
    # stay at their original sizes (already above 50K cap).
    'CICIDS2017': dict(
        mi_threshold        = 0.0,
        skip_rfecv          = True,
        mi_subsample        = 100_000,
        smote_max_per_class = 50_000,
    ),
    # UNSW_NB15: re-enable SMOTE with per-class cap=50K to match paper's approach.
    # This gives ~300K training samples and a manageable (~300 MB) RF model.
    'UNSW_NB15': dict(
        smote_max_per_class = 50_000,
    ),
    # NSL_KDD: 5 classes (Normal, DoS, Probe, R2L, U2R); 38 pre-normalised features.
    # mi_threshold=0.0 keeps all features (no zero-MI features expected in pre-processed data).
    # skip_rfecv=True: data is already curated; RFECV adds time without meaningful gain.
    # smote_max_per_class=50_000: oversample rare R2L (~995) and U2R (~52) classes
    # while capping the large Normal/DoS classes to prevent RF model explosion.
    'NSL_KDD': dict(
        mi_threshold        = 0.0,
        skip_rfecv          = True,
        smote_max_per_class = 50_000,
    ),
}

# ── Model hyperparameters ─────────────────────────────────────────────────────

LR_PARAMS = dict(seed=SEED)
RF_PARAMS = dict(seed=SEED)

CNN_TRAIN = dict(epochs=100, batch_size=256, patience=15)

LSTM_TRAIN = dict(
    window=10, epochs=100, batch_size=128, patience=15,
    subsample=80_000, device='/CPU:0',
)

AERF_TRAIN = dict(ae_epochs=50, ae_batch=256)

# ── Latency ───────────────────────────────────────────────────────────────────

LATENCY = dict(n_samples=1000, n_runs=5, seed=SEED)

# ── SHAP ──────────────────────────────────────────────────────────────────────

SHAP = dict(n_background=100, n_explain=1000, seed=SEED)
