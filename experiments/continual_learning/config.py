"""
experiments/continual_learning/config.py
Configuration for the continual learning experiment.
"""
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR    = Path(__file__).resolve().parents[2]
DATA_DIR    = BASE_DIR / 'dataset'
RESULTS_DIR = Path(__file__).resolve().parent / 'results'

DATASETS = {
    'TON_IoT':    (DATA_DIR / 'train_test_network.csv',                               'type'),
    'UNSW_NB15':  (BASE_DIR / 'dataset2' / 'UNSW_NB15.csv',                          'attack_cat'),
    'CICIDS2017': (BASE_DIR / 'dataset3' / 'IDS_datasets' / 'dataset_cicids2017.parquet', 'Label'),
}

# ── Reproducibility ───────────────────────────────────────────────────────────

SEED = 42

# ── Preprocessing (reuse baseline pipeline, no SMOTE during CL tasks) ────────

PREPROCESSING = dict(
    mi_threshold        = 0.05,
    rfecv_subsample     = 50_000,
    apply_smote         = False,   # SMOTE applied only on Task 0 initial train
    smote_k             = 5,
    smote_max_per_class = None,
    seed                = SEED,
)

PREPROCESSING_OVERRIDES = {
    'CICIDS2017': dict(
        mi_threshold        = 0.0,
        skip_rfecv          = True,
        mi_subsample        = 100_000,
        smote_max_per_class = 50_000,
    ),
    'UNSW_NB15': dict(
        smote_max_per_class = 50_000,
    ),
}

# ── Task splits (class groups per dataset) ────────────────────────────────────
#
# Each dataset is split into sequential tasks. Task 0 = initial training
# (all samples for the first group). Tasks 1+ = incremental sessions.
# Classes within a group are indices into the LabelEncoder output.
#
# TON_IoT  (10 classes): Normal, backdoor, ddos, dos, injection,
#                         mitm, password, ransomware, scanning, xss
# UNSW_NB15 (10 classes): Normal, Analysis, Backdoor, DoS, Exploits,
#                          Fuzzers, Generic, Reconnaissance, Shellcode, Worms
# CICIDS2017 (15 classes): BENIGN + 14 attack types

TASK_SPLITS = {
    'TON_IoT': [
        ['normal', 'scanning', 'dos'],        # Task 0 — benign + common flood attacks
        ['ddos', 'password', 'mitm'],          # Task 1 — network-level attacks
        ['backdoor', 'ransomware', 'injection', 'xss'],  # Task 2 — endpoint attacks
    ],
    'UNSW_NB15': [
        ['normal', 'generic', 'exploits'],     # Task 0
        ['fuzzers', 'reconnaissance', 'dos'],  # Task 1
        ['analysis', 'backdoor', 'shellcode', 'worms'],  # Task 2
    ],
    'CICIDS2017': [
        ['benign', 'bot', 'ddos', 'dos hulk'],                  # Task 0
        ['dos goldeneye', 'dos slowhttptest', 'dos slowloris',
         'ftp-patator', 'ssh-patator'],                          # Task 1
        ['infiltration', 'portscan',
         'web attack \ufffd brute force',
         'web attack \ufffd sql injection',
         'web attack \ufffd xss'],                               # Task 2
        ['heartbleed'],                                           # Task 3 (rare)
    ],
}

# ── CNN training hyperparameters ──────────────────────────────────────────────

CNN_INIT = dict(epochs=50, batch_size=256, patience=10, lr=1e-3)
CNN_INCREMENTAL = dict(epochs=20, batch_size=256, patience=5, lr=5e-4)

# ── Strategy hyperparameters ──────────────────────────────────────────────────

EWC_PARAMS = dict(
    lambda_ewc   = 5000,    # regularisation strength (Fisher penalty weight)
    fisher_samples = 500,   # samples used to estimate Fisher Information Matrix
)

REPLAY_PARAMS = dict(
    memory_size  = 2000,    # total replay buffer capacity (across all seen tasks)
    replay_ratio = 0.3,     # fraction of each minibatch drawn from replay memory
)

LWF_PARAMS = dict(
    temperature = 2.0,      # softening temperature for distillation
    alpha       = 0.5,      # weight of distillation loss (1-alpha = CE loss)
)

# ── Evaluation ────────────────────────────────────────────────────────────────

EVAL = dict(
    metrics = ['macro_f1', 'accuracy', 'forgetting_measure', 'bwt', 'plasticity'],
)
