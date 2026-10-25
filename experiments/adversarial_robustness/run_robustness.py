"""
experiments/adversarial_robustness/run_robustness.py
End-to-end adversarial robustness experiment for Paper 2.

Experiments
-----------
1. Baseline CNN performance (clean)
2. FGSM attack sweep (5 epsilons)
3. PGD attack sweep (5 epsilons)
4. C&W L2 attack (1,000 samples)
5. Defense A — PGD adversarial training (SMOTE-balanced data)
6. Defense B — Feature squeezing (bit-depth + median, no retraining)

Usage
-----
    cd IDS_IOT_Research
    python -m experiments.adversarial_robustness.run_robustness --dataset TON_IoT
    python -m experiments.adversarial_robustness.run_robustness --dataset all
    python -m experiments.adversarial_robustness.run_robustness --dataset TON_IoT --skip-attacks
"""
import argparse
import json
import os
import sys
import time
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

from src.data   import load_dataset
from src.models import build_cnn

from experiments.adversarial_robustness.config import (
    DATASETS, DATASET_PATHS,
    EPSILONS, PGD_ALPHA_RATIO, PGD_N_ITER,
    CW_CONFIDENCE, CW_LEARNING_RATE, CW_MAX_ITER,
    CW_BINARY_STEPS, CW_N_SAMPLES, CW_INITIAL_CONST,
    ADV_TRAIN_EPSILON, ADV_TRAIN_RATIO, ADV_TRAIN_EPOCHS,
    ADV_TRAIN_PATIENCE, ADV_TRAIN_BATCH,
    ADV_TRAIN_PGD_STEPS, ADV_TRAIN_PGD_ALPHA_RATIO, ADV_TRAIN_SUBSAMPLE,
    FS_N_BITS, FS_MEDIAN_WIN,
    EVAL_BATCH_SIZE, SEED,
    BASELINE_DIR, RESULTS_DIR,
)
from experiments.adversarial_robustness.attacks.fgsm import fgsm_attack
from experiments.adversarial_robustness.attacks.pgd  import pgd_attack
from experiments.adversarial_robustness.attacks.cw   import cw_l2_attack
from experiments.adversarial_robustness.defenses.adversarial_training import adversarial_train_cnn
from experiments.adversarial_robustness.defenses.feature_squeezing import (
    bit_depth_squeeze, median_squeeze, evaluate_squeeze)
from experiments.baseline.config import PREPROCESSING, PREPROCESSING_OVERRIDES, SPLIT


# ── Helpers ───────────────────────────────────────────────────────────────────

# Config fingerprint — changing any of these invalidates the attack cache
_CACHE_CONFIG = {
    'epsilons':       EPSILONS,
    'pgd_n_iter':     PGD_N_ITER,
    'pgd_alpha_ratio': PGD_ALPHA_RATIO,
    'cw_n_samples':   CW_N_SAMPLES,
    'cw_binary_steps': CW_BINARY_STEPS,
    'cw_max_iter':    CW_MAX_ITER,
}


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


def _metrics(y_true, y_pred, label: str = '') -> dict:
    m = {
        'accuracy':  float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average='macro',
                                           zero_division=0)),
        'recall':    float(recall_score(y_true, y_pred, average='macro',
                                        zero_division=0)),
        'f1':        float(f1_score(y_true, y_pred, average='macro',
                                    zero_division=0)),
    }
    if label:
        print(f"    {label:45s}  acc={m['accuracy']:.4f}  f1={m['f1']:.4f}")
    return m


def _asr(y_true, y_clean, y_adv) -> float:
    correct = y_clean == y_true
    if correct.sum() == 0:
        return 0.0
    return float((correct & (y_adv != y_true)).sum() / correct.sum())


def _cnn_predict(model, X: np.ndarray, n_features: int,
                 batch_size: int = 512) -> np.ndarray:
    preds = []
    for s in range(0, len(X), batch_size):
        batch = tf.constant(
            X[s:s + batch_size].reshape(-1, n_features, 1).astype(np.float32))
        preds.append(np.argmax(model(batch, training=False).numpy(), axis=1))
    return np.concatenate(preds)


def _latency_ms(model, X: np.ndarray, n_features: int,
                n_runs: int = 5, n_samples: int = 1000) -> float:
    idx = np.random.default_rng(SEED).choice(len(X), size=min(n_samples, len(X)),
                                              replace=False)
    X_bench = tf.constant(X[idx].reshape(-1, n_features, 1).astype(np.float32))
    model(X_bench, training=False)   # warm-up
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model(X_bench, training=False)
        times.append(time.perf_counter() - t0)
    return float(np.mean(times) / len(idx) * 1000)


def _cache_valid(cache_path: Path) -> bool:
    """Return True if the existing cache matches the current config fingerprint."""
    if not cache_path.exists():
        return False
    try:
        with open(cache_path) as f:
            cached = json.load(f)
        return cached.get('_config') == _CACHE_CONFIG
    except Exception:
        return False


# ── Core experiment ───────────────────────────────────────────────────────────

def run_one(dataset_name: str, skip_attacks: bool = False,
            skip_adv_training: bool = False):
    print(f"\n{'='*60}")
    print(f" Dataset: {dataset_name}")
    print(f"{'='*60}")

    csv_path, label_col = DATASET_PATHS[dataset_name]
    bl_models_dir  = BASELINE_DIR / 'results' / 'models'  / dataset_name
    bl_results_dir = BASELINE_DIR / 'results'             / dataset_name
    out_dir        = RESULTS_DIR  / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    attacks_cache = out_dir / 'attacks_cache.json'

    # ── 1. Load & preprocess ──────────────────────────────────────────────────
    print("\n[1/6] Loading data…")
    data = load_dataset(csv_path, label_col=label_col, **SPLIT)
    X_train_raw, y_train = data['X_train'], data['y_train']
    X_val_raw,   y_val   = data['X_val'],   data['y_val']
    X_test_raw,  y_test  = data['X_test'],  data['y_test']
    feature_names        = data['feature_names']
    class_names          = data['class_names']
    n_classes            = len(class_names)

    print("  Loading baseline pipeline…")
    pipeline = joblib.load(bl_results_dir / 'pipeline.pkl')

    # Scaled data (no SMOTE) — used for attack generation and validation
    X_train_sc, _ = pipeline.transform(X_train_raw, y=y_train)
    X_val_sc,   _ = pipeline.transform(X_val_raw,   y=y_val)
    X_test,     _ = pipeline.transform(X_test_raw,  y=y_test)
    n_features     = X_test.shape[1]

    print(f"  n_features={n_features}  n_classes={n_classes}  "
          f"train={len(X_train_sc):,}  test={len(X_test):,}")

    # ── 2. Load baseline CNN ──────────────────────────────────────────────────
    print("\n[2/6] Loading baseline CNN…")
    cnn_path = bl_models_dir / 'model_cnn.h5'
    if not cnn_path.exists():
        raise FileNotFoundError(
            f"Baseline CNN not found at {cnn_path}. "
            "Run `python -m experiments.baseline.run_experiment` first.")
    cnn = tf.keras.models.load_model(str(cnn_path))
    print(f"  Parameters: {cnn.count_params():,}")

    y_clean = _cnn_predict(cnn, X_test, n_features, EVAL_BATCH_SIZE)
    base_m  = _metrics(y_test, y_clean, 'Baseline (clean)')
    lat_ms  = _latency_ms(cnn, X_test, n_features)
    print(f"    Latency: {lat_ms:.4f} ms/sample")

    results = {
        'baseline':              {**base_m, 'latency_ms': lat_ms,
                                  'n_params': int(cnn.count_params())},
        'fgsm':                  {},
        'pgd':                   {},
        'cw':                    {},
        'adversarial_training':  {},
        'feature_squeezing':     {},
    }

    # ── 3 & 4. Attacks (FGSM / PGD / C&W) ────────────────────────────────────
    if skip_attacks and _cache_valid(attacks_cache):
        print("\n[3/6] Loading cached attack results (--skip-attacks)…")
        with open(attacks_cache) as f:
            cached = json.load(f)
        results['fgsm'] = cached['fgsm']
        results['pgd']  = cached['pgd']
        results['cw']   = cached['cw']
        print("  Cache loaded — skipping attack generation.")
    elif _cache_valid(attacks_cache):
        print("\n[3/6] Loading cached attack results…")
        with open(attacks_cache) as f:
            cached = json.load(f)
        results['fgsm'] = cached['fgsm']
        results['pgd']  = cached['pgd']
        results['cw']   = cached['cw']
        print("  Cache loaded — skipping attack generation.")
    else:
        if attacks_cache.exists():
            print("\n[3/6] Attack cache is stale (config changed) — regenerating…")
        else:
            print("\n[3/6] FGSM and PGD attacks…")

        for eps in EPSILONS:
            print(f"\n  ε = {eps}")

            X_fgsm = fgsm_attack(cnn, X_test, y_test, epsilon=eps,
                                  batch_size=EVAL_BATCH_SIZE)
            y_fgsm = _cnn_predict(cnn, X_fgsm, n_features, EVAL_BATCH_SIZE)
            fgsm_m = _metrics(y_test, y_fgsm, f'FGSM  ε={eps}')
            fgsm_m['asr'] = _asr(y_test, y_clean, y_fgsm)
            results['fgsm'][str(eps)] = fgsm_m

            alpha  = eps * PGD_ALPHA_RATIO
            X_pgd  = pgd_attack(cnn, X_test, y_test, epsilon=eps, alpha=alpha,
                                 n_iter=PGD_N_ITER, batch_size=EVAL_BATCH_SIZE,
                                 seed=SEED)
            y_pgd  = _cnn_predict(cnn, X_pgd, n_features, EVAL_BATCH_SIZE)
            pgd_m  = _metrics(y_test, y_pgd, f'PGD   ε={eps}')
            pgd_m['asr'] = _asr(y_test, y_clean, y_pgd)
            results['pgd'][str(eps)] = pgd_m

        print(f"\n[4/6] C&W L2 attack (n_samples={CW_N_SAMPLES})…")
        rng    = np.random.default_rng(SEED)
        idx_cw = rng.choice(len(X_test), size=min(CW_N_SAMPLES, len(X_test)),
                            replace=False)
        X_cw_src, y_cw_src = X_test[idx_cw], y_test[idx_cw]

        X_cw = cw_l2_attack(
            cnn, X_cw_src, y_cw_src,
            confidence=CW_CONFIDENCE,
            learning_rate=CW_LEARNING_RATE,
            max_iter=CW_MAX_ITER,
            binary_search_steps=CW_BINARY_STEPS,
            initial_const=CW_INITIAL_CONST,
            batch_size=32,
            seed=SEED,
        )
        y_cw_clean = _cnn_predict(cnn, X_cw_src, n_features, EVAL_BATCH_SIZE)
        y_cw_adv   = _cnn_predict(cnn, X_cw,     n_features, EVAL_BATCH_SIZE)
        cw_m       = _metrics(y_cw_src, y_cw_adv, 'C&W L2')
        cw_m['asr']       = _asr(y_cw_src, y_cw_clean, y_cw_adv)
        # Extended L2 perturbation statistics (full distribution)
        l2_norms = np.sqrt(np.sum((X_cw - X_cw_src) ** 2, axis=1))
        float32_eps = np.finfo(np.float32).eps
        n_zero = int((l2_norms < float32_eps).sum())
        cw_m['median_l2'] = float(np.median(l2_norms))   # backward compat
        cw_m['l2_stats'] = {
            'mean':               float(np.mean(l2_norms)),
            'std':                float(np.std(l2_norms)),
            'min':                float(np.min(l2_norms)),
            'p5':                 float(np.percentile(l2_norms, 5)),
            'p25':                float(np.percentile(l2_norms, 25)),
            'p50':                float(np.median(l2_norms)),
            'p75':                float(np.percentile(l2_norms, 75)),
            'p95':                float(np.percentile(l2_norms, 95)),
            'max':                float(np.max(l2_norms)),
            'n_zero_float32':     n_zero,
            'pct_zero_float32':   float(100 * n_zero / len(l2_norms)),
        }
        results['cw'] = cw_m

        _save_json({'_config': _CACHE_CONFIG,
                    'fgsm': results['fgsm'],
                    'pgd':  results['pgd'],
                    'cw':   results['cw']}, attacks_cache)
        print(f"  Attack results cached → {attacks_cache}")

    # ── 5. Defense A: PGD Adversarial Training ────────────────────────────────
    print("\n[5/6] Defense A — PGD Adversarial Training…")

    adv_cnn_path = out_dir / 'model_cnn_adv.h5'

    if skip_adv_training and adv_cnn_path.exists():
        print(f"  --skip-adv-training: loading saved model from {adv_cnn_path}")
        adv_cnn = tf.keras.models.load_model(str(adv_cnn_path))
    else:
        # Get SMOTE-balanced training data (critical for imbalanced datasets)
        print("  Applying SMOTE to training data…")
        X_train_sm, y_train_sm = pipeline.transform(
            X_train_raw, y=y_train, apply_smote=True)
        print(f"  SMOTE training set: {len(X_train_sm):,} samples")

        cw_train = compute_class_weight(
            'balanced', classes=np.unique(y_train_sm), y=y_train_sm)
        cw_dict  = dict(enumerate(cw_train))

        pgd_alpha = ADV_TRAIN_EPSILON * ADV_TRAIN_PGD_ALPHA_RATIO
        subsample = ADV_TRAIN_SUBSAMPLE.get(dataset_name)
        adv_cnn   = build_cnn(n_features, n_classes)

        adv_cnn = adversarial_train_cnn(
            model=adv_cnn,
            X_train=X_train_sm,
            y_train=y_train_sm,
            X_val=X_val_sc,
            y_val=y_val,
            epsilon=ADV_TRAIN_EPSILON,
            adv_ratio=ADV_TRAIN_RATIO,
            epochs=ADV_TRAIN_EPOCHS,
            patience=ADV_TRAIN_PATIENCE,
            batch_size=ADV_TRAIN_BATCH,
            pgd_steps=ADV_TRAIN_PGD_STEPS,
            pgd_alpha=pgd_alpha,
            train_subsample=subsample,
            class_weight=cw_dict,
            out_path=adv_cnn_path,
        )
        adv_cnn.save(str(adv_cnn_path))
        print(f"  PGD-adversarially trained model saved → {adv_cnn_path}")

    y_adv_clean = _cnn_predict(adv_cnn, X_test, n_features, EVAL_BATCH_SIZE)
    adv_lat_ms  = _latency_ms(adv_cnn, X_test, n_features)
    adv_clean_m = _metrics(y_test, y_adv_clean, 'AdvTrain-PGD (clean)')
    adv_clean_m['latency_ms'] = adv_lat_ms
    adv_clean_m['n_params']   = int(adv_cnn.count_params())

    adv_attacks = {}
    for eps in EPSILONS:
        print(f"\n  AdvTrain ε = {eps}")
        X_f = fgsm_attack(adv_cnn, X_test, y_test, epsilon=eps,
                          batch_size=EVAL_BATCH_SIZE)
        y_f = _cnn_predict(adv_cnn, X_f, n_features, EVAL_BATCH_SIZE)
        fm  = _metrics(y_test, y_f, f'AdvTrain-PGD + FGSM ε={eps}')
        fm['asr'] = _asr(y_test, y_adv_clean, y_f)

        X_p = pgd_attack(adv_cnn, X_test, y_test, epsilon=eps,
                         alpha=eps * PGD_ALPHA_RATIO, n_iter=PGD_N_ITER,
                         batch_size=EVAL_BATCH_SIZE, seed=SEED)
        y_p = _cnn_predict(adv_cnn, X_p, n_features, EVAL_BATCH_SIZE)
        pm  = _metrics(y_test, y_p, f'AdvTrain-PGD + PGD  ε={eps}')
        pm['asr'] = _asr(y_test, y_adv_clean, y_p)

        adv_attacks[str(eps)] = {'fgsm': fm, 'pgd': pm}

    results['adversarial_training'] = {
        'clean':   adv_clean_m,
        'attacks': adv_attacks,
    }

    # ── 6. Defense B: Feature Squeezing ───────────────────────────────────────
    print("\n[6/6] Defense B — Feature Squeezing…")

    bit_fn    = lambda X: bit_depth_squeeze(X, n_bits=FS_N_BITS)
    median_fn = lambda X: median_squeeze(X, window=FS_MEDIAN_WIN)

    def _fs_eval_attacks(squeeze_fn, label_prefix):
        """Evaluate feature squeezing on clean + all FGSM/PGD epsilons."""
        out = {}
        # Clean
        m = evaluate_squeeze(cnn, X_test, y_test, squeeze_fn,
                             n_features, EVAL_BATCH_SIZE)
        _metrics(y_test,
                 _cnn_predict(cnn,
                              squeeze_fn(X_test), n_features, EVAL_BATCH_SIZE),
                 f'{label_prefix} (clean)')
        out['clean'] = m

        # FGSM & PGD at each epsilon
        out['fgsm'] = {}
        out['pgd']  = {}
        for eps in EPSILONS:
            X_fgsm = fgsm_attack(cnn, X_test, y_test, epsilon=eps,
                                  batch_size=EVAL_BATCH_SIZE)
            m_fgsm = evaluate_squeeze(cnn, X_fgsm, y_test, squeeze_fn,
                                      n_features, EVAL_BATCH_SIZE)
            print(f"    {label_prefix} + FGSM ε={eps}: "
                  f"acc={m_fgsm['accuracy']:.4f}  f1={m_fgsm['f1']:.4f}")
            out['fgsm'][str(eps)] = m_fgsm

            X_pgd = pgd_attack(cnn, X_test, y_test, epsilon=eps,
                               alpha=eps * PGD_ALPHA_RATIO, n_iter=PGD_N_ITER,
                               batch_size=EVAL_BATCH_SIZE, seed=SEED)
            m_pgd = evaluate_squeeze(cnn, X_pgd, y_test, squeeze_fn,
                                     n_features, EVAL_BATCH_SIZE)
            print(f"    {label_prefix} + PGD  ε={eps}: "
                  f"acc={m_pgd['accuracy']:.4f}  f1={m_pgd['f1']:.4f}")
            out['pgd'][str(eps)] = m_pgd
        return out

    print(f"\n  Bit-depth squeezing ({FS_N_BITS}-bit, {2**FS_N_BITS} levels)…")
    results['feature_squeezing']['bit_depth'] = _fs_eval_attacks(
        bit_fn, f'BitSqueeze-{FS_N_BITS}bit')

    print(f"\n  Median squeezing (window={FS_MEDIAN_WIN})…")
    results['feature_squeezing']['median'] = _fs_eval_attacks(
        median_fn, f'MedianSqueeze-w{FS_MEDIAN_WIN}')

    # ── Save ──────────────────────────────────────────────────────────────────
    _save_json(results, out_dir / 'robustness_results.json')
    print(f"\nDone — results saved → {out_dir}")
    return results


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Adversarial robustness experiment (Paper 2)')
    parser.add_argument('--dataset', default='TON_IoT',
                        help=f'Dataset name or "all". Choices: {DATASETS + ["all"]}')
    parser.add_argument('--skip-attacks', action='store_true',
                        help='Load cached attack results even if cache is stale')
    parser.add_argument('--skip-adv-training', action='store_true',
                        help='Load saved model_cnn_adv.h5 instead of retraining')
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == 'all' else [args.dataset]
    for ds in datasets:
        if ds not in DATASETS:
            print(f"Unknown dataset '{ds}'. Valid: {DATASETS}")
            continue
        run_one(ds, skip_attacks=args.skip_attacks,
                skip_adv_training=args.skip_adv_training)


if __name__ == '__main__':
    main()
