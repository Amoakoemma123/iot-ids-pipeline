"""
experiments/adversarial_robustness/run_transfer_attack.py
Black-box transfer attack experiment for Paper 2 (top-tier fix).

Threat model
------------
The adversary has NO access to the target model weights.  Instead they train
a local surrogate CNN (same architecture, different random seed) on publicly
available data of the same type, generate adversarial examples from the
surrogate, and transfer them to fool the target CNN.

This evaluates the real-world black-box threat level of IoT IDS deployments.

Metrics
-------
- Transfer ASR  : fraction of surrogate-adversarial examples that also fool
                  the target model (subset: samples correctly classified clean)
- Transfer F1   : macro-F1 of target CNN on surrogate-adversarial examples
- White-box ASR : same metric but with target-model-generated attacks (upper bound)
- Transferability ratio = Transfer ASR / White-box ASR  (0..1)

Usage
-----
    python -m experiments.adversarial_robustness.run_transfer_attack
    python -m experiments.adversarial_robustness.run_transfer_attack --dataset TON_IoT
"""
import argparse
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
try:
    import tensorflow as tf
    for _gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(_gpu, True)
    _gpus = tf.config.list_physical_devices('GPU')
    print(f"[GPU] {len(_gpus)} device(s): {[g.name for g in _gpus]}")
except Exception as e:
    print(f"[GPU] Warning: {e}")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

from src.data   import load_dataset
from src.models import build_cnn

from experiments.adversarial_robustness.config import (
    DATASETS, DATASET_PATHS, EPSILONS, PGD_ALPHA_RATIO, PGD_N_ITER,
    EVAL_BATCH_SIZE, SEED, BASELINE_DIR, RESULTS_DIR,
)
from experiments.adversarial_robustness.attacks.fgsm import fgsm_attack
from experiments.adversarial_robustness.attacks.pgd  import pgd_attack
from experiments.baseline.config import PREPROCESSING, PREPROCESSING_OVERRIDES, SPLIT

# Epsilons to sweep for transfer experiment (reduced for speed)
TRANSFER_EPSILONS = [0.01, 0.05, 0.1, 0.2, 0.3]

# Surrogate training hyperparameters (fast training — we only need a decent model)
SURROGATE_EPOCHS    = 30
SURROGATE_PATIENCE  = 7
SURROGATE_BATCH     = 256
SURROGATE_SEED      = 123   # different from SEED=42 → different init


def _save_json(obj: dict, path: Path):
    def _default(o):
        if isinstance(o, np.integer):  return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray):  return o.tolist()
        raise TypeError(f"Not serialisable: {type(o)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=_default)
    print(f"  Saved → {path}")


def _cnn_predict(model, X: np.ndarray, n_features: int,
                 batch_size: int = 512) -> np.ndarray:
    preds = []
    for s in range(0, len(X), batch_size):
        batch = tf.constant(
            X[s:s + batch_size].reshape(-1, n_features, 1).astype(np.float32))
        preds.append(np.argmax(model(batch, training=False).numpy(), axis=1))
    return np.concatenate(preds)


def _metrics(y_true, y_pred) -> dict:
    return {
        'accuracy':  float(accuracy_score(y_true, y_pred)),
        'f1':        float(f1_score(y_true, y_pred, average='macro',
                                    zero_division=0)),
    }


def _asr(y_true, y_clean, y_adv) -> float:
    correct = y_clean == y_true
    if correct.sum() == 0:
        return 0.0
    return float((correct & (y_adv != y_true)).sum() / correct.sum())


def train_surrogate(X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray,
                    n_features: int, n_classes: int,
                    out_path: Path) -> tf.keras.Model:
    """Train a surrogate CNN with a different random seed."""
    if out_path.exists():
        print(f"  Surrogate already trained — loading {out_path}")
        return tf.keras.models.load_model(str(out_path))

    tf.random.set_seed(SURROGATE_SEED)
    np.random.seed(SURROGATE_SEED)
    surrogate = build_cnn(n_features, n_classes)

    # Class weights for imbalanced datasets
    classes = np.unique(y_train)
    cw = compute_class_weight('balanced', classes=classes, y=y_train)
    cw_dict = dict(enumerate(cw))

    cb = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=SURROGATE_PATIENCE,
            restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            str(out_path), save_best_only=True, monitor='val_loss', verbose=0),
    ]

    # Cap training data at 100K to keep surrogate training fast
    if len(X_train) > 100_000:
        rng = np.random.default_rng(SURROGATE_SEED)
        idx = rng.choice(len(X_train), 100_000, replace=False)
        X_fit, y_fit = X_train[idx], y_train[idx]
        print(f"  Surrogate training subsample: {len(X_train):,} → 100,000")
    else:
        X_fit, y_fit = X_train, y_train

    X_fit_3d = X_fit.reshape(-1, n_features, 1).astype(np.float32)
    X_val_3d = X_val.reshape(-1, n_features, 1).astype(np.float32)

    surrogate.fit(
        X_fit_3d, y_fit,
        validation_data=(X_val_3d, y_val),
        epochs=SURROGATE_EPOCHS,
        batch_size=SURROGATE_BATCH,
        class_weight=cw_dict,
        callbacks=cb,
        verbose=1,
    )
    print(f"  Surrogate saved → {out_path}")
    return surrogate


def run_one(dataset_name: str):
    print(f"\n{'='*60}")
    print(f" Transfer Attack — Dataset: {dataset_name}")
    print(f"{'='*60}")

    csv_path, label_col = DATASET_PATHS[dataset_name]
    bl_models_dir  = BASELINE_DIR / 'results' / 'models'  / dataset_name
    bl_results_dir = BASELINE_DIR / 'results'             / dataset_name
    out_dir        = RESULTS_DIR  / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n[1] Loading data and pipeline…")
    # CICIDS2017 (2.83M rows) causes OOM when fully loaded alongside TF.
    # For transfer attack we only need ~400K rows: use a stratified subsample
    # loaded directly from parquet. The baseline CNN and pipeline are unchanged.
    _load_path = csv_path
    if dataset_name == 'CICIDS2017':
        import pandas as pd
        print("  CICIDS2017: loading 400K-row subsample to avoid OOM…")
        _df_full = pd.read_parquet(str(csv_path))
        _df_full = _df_full.dropna(subset=[label_col])
        _df_full = _df_full.drop_duplicates()
        # Stratified sample per class to preserve class distribution
        _df_sample = _df_full.groupby(label_col, group_keys=False).apply(
            lambda g: g.sample(min(len(g), max(1, int(400_000 * len(g) / len(_df_full)))),
                               random_state=SEED))
        _sample_path = Path('/tmp/cicids_transfer_sample.parquet')
        _df_sample.to_parquet(str(_sample_path))
        del _df_full, _df_sample
        import gc; gc.collect()
        _load_path = str(_sample_path)

    data = load_dataset(_load_path, label_col=label_col, **SPLIT)
    X_train_raw, y_train = data['X_train'], data['y_train']
    X_val_raw,   y_val   = data['X_val'],   data['y_val']
    X_test_raw,  y_test  = data['X_test'],  data['y_test']
    n_classes = len(data['class_names'])

    pipeline = joblib.load(bl_results_dir / 'pipeline.pkl')

    X_train_sc, _ = pipeline.transform(X_train_raw, y=y_train)
    X_val_sc,   _ = pipeline.transform(X_val_raw,   y=y_val)
    X_test,     _ = pipeline.transform(X_test_raw,  y=y_test)
    n_features = X_test.shape[1]

    print(f"  n_features={n_features}  n_classes={n_classes}  "
          f"train={len(X_train_sc):,}  test={len(X_test):,}")

    # ── Load target CNN (baseline) ─────────────────────────────────────────
    print("\n[2] Loading target CNN (baseline)…")
    target_cnn = tf.keras.models.load_model(
        str(bl_models_dir / 'model_cnn.h5'))
    y_target_clean = _cnn_predict(target_cnn, X_test, n_features, EVAL_BATCH_SIZE)
    target_clean_m = _metrics(y_test, y_target_clean)
    print(f"  Target clean: acc={target_clean_m['accuracy']:.4f}  "
          f"f1={target_clean_m['f1']:.4f}")

    # ── Train surrogate CNN ────────────────────────────────────────────────
    print("\n[3] Training surrogate CNN (seed=123, independent init)…")
    # SMOTE-balanced training for surrogate (same as target was trained on)
    X_train_sm, y_train_sm = pipeline.transform(
        X_train_raw, y=y_train, apply_smote=True)

    surrogate_path = out_dir / 'model_cnn_surrogate.h5'
    surrogate_cnn = train_surrogate(
        X_train_sm, y_train_sm,
        X_val_sc, y_val,
        n_features, n_classes,
        surrogate_path,
    )
    y_surrogate_clean = _cnn_predict(surrogate_cnn, X_test, n_features, EVAL_BATCH_SIZE)
    surr_clean_m = _metrics(y_test, y_surrogate_clean)
    print(f"  Surrogate clean: acc={surr_clean_m['accuracy']:.4f}  "
          f"f1={surr_clean_m['f1']:.4f}")

    # ── Transfer attacks ──────────────────────────────────────────────────
    print("\n[4] Generating transfer attacks (surrogate → target)…")
    results = {
        'target_clean':    {**target_clean_m, 'n_params': int(target_cnn.count_params())},
        'surrogate_clean': {**surr_clean_m,   'n_params': int(surrogate_cnn.count_params())},
        'fgsm': {},
        'pgd':  {},
    }

    for eps in TRANSFER_EPSILONS:
        print(f"\n  ε = {eps}")

        # FGSM from surrogate, evaluate on target
        X_fgsm_surr = fgsm_attack(surrogate_cnn, X_test, y_test,
                                   epsilon=eps, batch_size=EVAL_BATCH_SIZE)
        y_target_on_fgsm = _cnn_predict(target_cnn, X_fgsm_surr, n_features,
                                         EVAL_BATCH_SIZE)
        m_fgsm = _metrics(y_test, y_target_on_fgsm)
        m_fgsm['transfer_asr'] = _asr(y_test, y_target_clean, y_target_on_fgsm)
        # White-box FGSM ASR from the existing results for comparison
        results['fgsm'][str(eps)] = m_fgsm
        print(f"    FGSM transfer: acc={m_fgsm['accuracy']:.4f}  "
              f"f1={m_fgsm['f1']:.4f}  transfer_asr={m_fgsm['transfer_asr']:.4f}")

        # PGD from surrogate, evaluate on target
        alpha = eps * PGD_ALPHA_RATIO
        X_pgd_surr = pgd_attack(surrogate_cnn, X_test, y_test,
                                 epsilon=eps, alpha=alpha,
                                 n_iter=PGD_N_ITER, batch_size=EVAL_BATCH_SIZE,
                                 seed=SEED)
        y_target_on_pgd = _cnn_predict(target_cnn, X_pgd_surr, n_features,
                                        EVAL_BATCH_SIZE)
        m_pgd = _metrics(y_test, y_target_on_pgd)
        m_pgd['transfer_asr'] = _asr(y_test, y_target_clean, y_target_on_pgd)
        results['pgd'][str(eps)] = m_pgd
        print(f"    PGD  transfer: acc={m_pgd['accuracy']:.4f}  "
              f"f1={m_pgd['f1']:.4f}  transfer_asr={m_pgd['transfer_asr']:.4f}")

    # ── Load white-box ASR from existing results for comparison ───────────
    wb_path = out_dir / 'robustness_results.json'
    if wb_path.exists():
        with open(wb_path) as f:
            wb = json.load(f)
        results['whitebox_fgsm_asr'] = {
            eps: wb['fgsm'][eps]['asr'] for eps in wb['fgsm']}
        results['whitebox_pgd_asr'] = {
            eps: wb['pgd'][eps]['asr'] for eps in wb['pgd']}
        # Transferability ratio = transfer_asr / whitebox_asr
        results['transferability_ratio_fgsm'] = {
            eps: (results['fgsm'][eps]['transfer_asr'] /
                  wb['fgsm'][eps]['asr'])
            if wb['fgsm'][eps]['asr'] > 0 else 0.0
            for eps in results['fgsm']
            if eps in wb['fgsm']
        }
        results['transferability_ratio_pgd'] = {
            eps: (results['pgd'][eps]['transfer_asr'] /
                  wb['pgd'][eps]['asr'])
            if wb['pgd'][eps]['asr'] > 0 else 0.0
            for eps in results['pgd']
            if eps in wb['pgd']
        }

    _save_json(results, out_dir / 'transfer_attack_results.json')
    print(f"\nDone — transfer attack results saved → {out_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Black-box transfer attack experiment (Paper 2 fix)')
    parser.add_argument('--dataset', default='all',
                        help=f'Dataset name or "all". Choices: {DATASETS + ["all"]}')
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == 'all' else [args.dataset]
    for ds in datasets:
        if ds not in DATASETS:
            print(f"Unknown dataset '{ds}'. Valid: {DATASETS}")
            continue
        run_one(ds)


if __name__ == '__main__':
    main()
