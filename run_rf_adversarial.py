"""
run_rf_adversarial.py
Evaluates RF models under PGD adversarial examples crafted targeting the CNN.
Runs all 3 datasets sequentially. Fully non-interactive.

Output: experiments/adversarial_robustness/results/rf_adversarial_results.json
"""
import os, sys, json, time, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import tensorflow as tf
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
print(f"[GPU] {tf.config.list_physical_devices('GPU')}")

from src.data.loader import load_dataset
from src.data.preprocessing import PreprocessingPipeline
from experiments.baseline.config import (
    DATASETS, SPLIT, PREPROCESSING, PREPROCESSING_OVERRIDES, SEED
)

DATASETS_TO_RUN = ['TON_IoT', 'UNSW_NB15', 'CICIDS2017']
EPSILONS        = [0.01, 0.1, 0.3]
N_EVAL          = 3000
PGD_STEPS       = 40
MODELS_DIR      = ROOT / 'experiments' / 'baseline' / 'results' / 'models'
OUT_PATH        = ROOT / 'experiments' / 'adversarial_robustness' / 'results' / 'rf_adversarial_results.json'

np.random.seed(SEED)
tf.random.set_seed(SEED)


def pgd_attack_cnn(cnn, X, eps, steps=40, batch_size=512):
    """PGD-40 targeting CNN, returns adversarial X."""
    alpha = eps / 10.0
    X_adv = X.copy().astype(np.float32)
    n_feat = X.shape[1]
    for start in range(0, len(X), batch_size):
        X_b = tf.constant(X[start:start+batch_size].reshape(-1, n_feat, 1))
        x_t = X_b + tf.random.uniform(tf.shape(X_b), -eps, eps)
        x_t = tf.clip_by_value(x_t, 0., 1.)
        for _ in range(steps):
            with tf.GradientTape() as tape:
                tape.watch(x_t)
                preds = cnn(x_t, training=False)
                loss  = tf.reduce_mean(
                    tf.keras.losses.sparse_categorical_crossentropy(
                        tf.argmax(cnn(X_b, training=False), axis=1), preds))
            grad  = tape.gradient(loss, x_t)
            x_t   = x_t + alpha * tf.sign(grad)
            x_t   = tf.clip_by_value(x_t, X_b - eps, X_b + eps)
            x_t   = tf.clip_by_value(x_t, 0., 1.)
            x_t   = tf.stop_gradient(x_t)
        X_adv[start:start+batch_size] = x_t.numpy().reshape(-1, n_feat)
    return X_adv


def cnn_predict(cnn, X, batch_size=512):
    n_feat = X.shape[1]
    preds = []
    for s in range(0, len(X), batch_size):
        b = tf.constant(X[s:s+batch_size].reshape(-1, n_feat, 1).astype(np.float32))
        preds.append(np.argmax(cnn(b, training=False).numpy(), axis=1))
    return np.concatenate(preds)


all_results = {}

for ds in DATASETS_TO_RUN:
    print(f"\n{'='*60}")
    print(f"  RF Adversarial Evaluation — {ds}")
    print(f"{'='*60}")
    t_ds = time.time()

    # ── Load data ─────────────────────────────────────────────────────────────
    csv_path, label_col = DATASETS[ds]
    data = load_dataset(csv_path, label_col=label_col, **SPLIT)

    pp_kw = {**PREPROCESSING}
    pp_kw.update(PREPROCESSING_OVERRIDES.get(ds, {}))
    valid = {'mi_threshold','smote_k','smote_max_per_class',
             'skip_rfecv','rfecv_subsample','mi_subsample','seed'}
    pp_kw = {k: v for k, v in pp_kw.items() if k in valid}

    pipeline = PreprocessingPipeline(**pp_kw)
    print("[1] Fitting preprocessing pipeline...")
    _, _ = pipeline.fit_transform(
        data['X_train'], data['y_train'], feature_names=data['feature_names'])
    X_test, _ = pipeline.transform(data['X_test'], y=data['y_test'])
    y_test = data['y_test']
    n_feat = X_test.shape[1]
    print(f"    Features={n_feat}  Test={len(X_test):,}")

    # ── Load models ───────────────────────────────────────────────────────────
    print("[2] Loading CNN and RF models...")
    cnn = tf.keras.models.load_model(str(MODELS_DIR / ds / 'model_cnn.h5'), compile=False)
    rf  = joblib.load(MODELS_DIR / ds / 'model_rf.pkl')
    print(f"    CNN loaded | RF: {type(rf).__name__}")

    # ── Sample eval subset ────────────────────────────────────────────────────
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X_test), size=min(N_EVAL, len(X_test)), replace=False)
    X_eval = X_test[idx]
    y_eval = y_test[idx]

    # ── Clean performance ─────────────────────────────────────────────────────
    print("[3] Clean performance...")
    cnn_clean = cnn_predict(cnn, X_eval)
    rf_clean  = rf.predict(X_eval)
    cnn_f1_clean = float(f1_score(y_eval, cnn_clean, average='macro', zero_division=0))
    rf_f1_clean  = float(f1_score(y_eval, rf_clean,  average='macro', zero_division=0))
    print(f"    CNN clean F1={cnn_f1_clean:.4f}  RF clean F1={rf_f1_clean:.4f}")

    # ── PGD attack + RF evaluation ────────────────────────────────────────────
    ds_results = {
        'clean': {'cnn_f1': cnn_f1_clean, 'rf_f1': rf_f1_clean},
        'pgd': {}
    }

    for eps in EPSILONS:
        print(f"[4] PGD ε={eps} (targeting CNN)...")
        t0 = time.time()
        X_adv = pgd_attack_cnn(cnn, X_eval, eps, steps=PGD_STEPS)

        cnn_adv = cnn_predict(cnn, X_adv)
        rf_adv  = rf.predict(X_adv)

        cnn_f1_adv = float(f1_score(y_eval, cnn_adv, average='macro', zero_division=0))
        rf_f1_adv  = float(f1_score(y_eval, rf_adv,  average='macro', zero_division=0))

        # Attack success rate (fraction of correctly classified samples that get flipped)
        cnn_correct = cnn_clean == y_eval
        rf_correct  = rf_clean  == y_eval
        cnn_asr = float((cnn_correct & (cnn_adv != y_eval)).sum() / max(cnn_correct.sum(), 1))
        rf_asr  = float((rf_correct  & (rf_adv  != y_eval)).sum() / max(rf_correct.sum(), 1))

        print(f"    CNN: F1={cnn_f1_adv:.4f}  ASR={cnn_asr:.4f}  "
              f"RF: F1={rf_f1_adv:.4f}  ASR={rf_asr:.4f}  ({time.time()-t0:.1f}s)")

        ds_results['pgd'][str(eps)] = {
            'cnn': {'f1': cnn_f1_adv, 'asr': cnn_asr,
                    'acc': float(accuracy_score(y_eval, cnn_adv))},
            'rf':  {'f1': rf_f1_adv,  'asr': rf_asr,
                    'acc': float(accuracy_score(y_eval, rf_adv))},
        }

    all_results[ds] = ds_results
    print(f"  {ds} done in {(time.time()-t_ds)/60:.1f} min")

# ── Save ──────────────────────────────────────────────────────────────────────
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\n[DONE] Results saved to {OUT_PATH}")

# Summary
print("\n=== SUMMARY ===")
for ds, res in all_results.items():
    print(f"\n{ds}:")
    print(f"  Clean:    CNN F1={res['clean']['cnn_f1']:.4f}  RF F1={res['clean']['rf_f1']:.4f}")
    for eps, r in res['pgd'].items():
        print(f"  PGD ε={eps}: CNN F1={r['cnn']['f1']:.4f} (ASR={r['cnn']['asr']:.3f})  "
              f"RF F1={r['rf']['f1']:.4f} (ASR={r['rf']['asr']:.3f})")
