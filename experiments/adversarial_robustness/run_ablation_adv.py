"""
experiments/adversarial_robustness/run_ablation_adv.py
Comprehensive defense ablation study for Paper 2 (top-tier submission fix).

Ablation conditions
-------------------
A. Feature squeezing — bit-depth:
   2-bit (4 levels), 4-bit (16 levels, current), 8-bit (256 levels)
   → Evaluate clean accuracy + ASR reduction under FGSM/PGD on all 3 datasets

B. Feature squeezing — median window size:
   window=3 (current), 5, 7
   → Same evaluation protocol

C. Adversarial training — mix ratio:
   adv_ratio = 0.25, 0.50 (current), 0.75
   → Train fresh CNNs on TON_IoT; compare clean F1 and robustness at ε=0.1

D. Adversarial training — PGD epsilon during training:
   train_eps = 0.05, 0.10 (current), 0.20
   → Same protocol on TON_IoT

All results saved to:
    experiments/adversarial_robustness/results/ablation_defense_results.json

Usage
-----
    python -m experiments.adversarial_robustness.run_ablation_adv
    python -m experiments.adversarial_robustness.run_ablation_adv --section A
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
    print(f"[GPU] {len(tf.config.list_physical_devices('GPU'))} GPU(s)")
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
    ADV_TRAIN_EPSILON, ADV_TRAIN_EPOCHS, ADV_TRAIN_PATIENCE, ADV_TRAIN_BATCH,
    ADV_TRAIN_PGD_STEPS, ADV_TRAIN_PGD_ALPHA_RATIO, ADV_TRAIN_SUBSAMPLE,
    EVAL_BATCH_SIZE, SEED, BASELINE_DIR, RESULTS_DIR,
)
from experiments.adversarial_robustness.attacks.fgsm import fgsm_attack
from experiments.adversarial_robustness.attacks.pgd  import pgd_attack
from experiments.adversarial_robustness.defenses.feature_squeezing import (
    bit_depth_squeeze, median_squeeze, evaluate_squeeze)
from experiments.adversarial_robustness.defenses.adversarial_training import (
    adversarial_train_cnn)
from experiments.baseline.config import SPLIT

ABLATION_RESULTS_PATH = RESULTS_DIR / 'ablation_defense_results.json'

# Ablation values
FS_BITS_OPTIONS   = [2, 4, 8]          # Section A
FS_WINDOW_OPTIONS = [3, 5, 7]          # Section B
ADV_RATIO_OPTIONS = [0.25, 0.50, 0.75] # Section C
ADV_EPS_OPTIONS   = [0.05, 0.10, 0.20] # Section D

# For adv training ablations, only run on TON_IoT (fastest dataset)
ADV_ABLATION_DATASET = 'TON_IoT'

# Evaluation epsilons for attack-resistance check
EVAL_EPSILONS = [0.01, 0.1, 0.3]


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


def _asr(y_true, y_clean, y_adv) -> float:
    correct = y_clean == y_true
    if correct.sum() == 0:
        return 0.0
    return float((correct & (y_adv != y_true)).sum() / correct.sum())


def _load_dataset_and_cnn(dataset_name: str):
    """Load preprocessed data + baseline CNN for a dataset."""
    csv_path, label_col = DATASET_PATHS[dataset_name]
    bl_models_dir  = BASELINE_DIR / 'results' / 'models'  / dataset_name
    bl_results_dir = BASELINE_DIR / 'results'             / dataset_name

    data = load_dataset(csv_path, label_col=label_col, **SPLIT)
    pipeline = joblib.load(bl_results_dir / 'pipeline.pkl')

    X_train_sc, _ = pipeline.transform(data['X_train'], y=data['y_train'])
    X_val_sc,   _ = pipeline.transform(data['X_val'],   y=data['y_val'])
    X_test,     _ = pipeline.transform(data['X_test'],  y=data['y_test'])
    n_features = X_test.shape[1]
    n_classes  = len(data['class_names'])

    cnn = tf.keras.models.load_model(str(bl_models_dir / 'model_cnn.h5'))
    y_clean = _cnn_predict(cnn, X_test, n_features, EVAL_BATCH_SIZE)

    return {
        'cnn':         cnn,
        'pipeline':    pipeline,
        'X_train_raw': data['X_train'],
        'y_train':     data['y_train'],
        'X_val_sc':    X_val_sc,
        'y_val':       data['y_val'],
        'X_train_sc':  X_train_sc,
        'X_test':      X_test,
        'y_test':      data['y_test'],
        'y_clean':     y_clean,
        'n_features':  n_features,
        'n_classes':   n_classes,
    }


# ── Section A: Feature squeezing — bit-depth ─────────────────────────────────

def section_a_bit_depth():
    """Ablate feature squeezing bit-depth across all 3 datasets."""
    print("\n" + "="*60)
    print(" SECTION A: Feature Squeezing — Bit-Depth Ablation")
    print("="*60)

    results = {}
    for ds in DATASETS:
        print(f"\n  Dataset: {ds}")
        d = _load_dataset_and_cnn(ds)
        cnn, X_test, y_test, y_clean = d['cnn'], d['X_test'], d['y_test'], d['y_clean']
        nf = d['n_features']
        ds_results = {}

        for n_bits in FS_BITS_OPTIONS:
            print(f"    Bit-depth = {n_bits} ({2**n_bits} levels)")
            sq_fn = lambda X, nb=n_bits: bit_depth_squeeze(X, n_bits=nb)
            cond  = {}

            # Clean accuracy
            cond['clean'] = evaluate_squeeze(cnn, X_test, y_test, sq_fn, nf,
                                             EVAL_BATCH_SIZE)

            # Under attack at each eval epsilon
            cond['fgsm'] = {}
            cond['pgd']  = {}
            for eps in EVAL_EPSILONS:
                X_f = fgsm_attack(cnn, X_test, y_test, epsilon=eps,
                                  batch_size=EVAL_BATCH_SIZE)
                m_f = evaluate_squeeze(cnn, X_f, y_test, sq_fn, nf, EVAL_BATCH_SIZE)
                cond['fgsm'][str(eps)] = m_f

                X_p = pgd_attack(cnn, X_test, y_test, epsilon=eps,
                                 alpha=eps * PGD_ALPHA_RATIO, n_iter=PGD_N_ITER,
                                 batch_size=EVAL_BATCH_SIZE, seed=SEED)
                m_p = evaluate_squeeze(cnn, X_p, y_test, sq_fn, nf, EVAL_BATCH_SIZE)
                cond['pgd'][str(eps)] = m_p

                print(f"      ε={eps}: FGSM acc={m_f['accuracy']:.4f}  "
                      f"PGD acc={m_p['accuracy']:.4f}")

            ds_results[str(n_bits)] = cond

        results[ds] = ds_results

    return results


# ── Section B: Feature squeezing — window size ───────────────────────────────

def section_b_window():
    """Ablate median-filter window size across all 3 datasets."""
    print("\n" + "="*60)
    print(" SECTION B: Feature Squeezing — Median Window Ablation")
    print("="*60)

    results = {}
    for ds in DATASETS:
        print(f"\n  Dataset: {ds}")
        d = _load_dataset_and_cnn(ds)
        cnn, X_test, y_test = d['cnn'], d['X_test'], d['y_test']
        nf = d['n_features']
        ds_results = {}

        for win in FS_WINDOW_OPTIONS:
            print(f"    Window = {win}")
            sq_fn = lambda X, w=win: median_squeeze(X, window=w)
            cond  = {}

            cond['clean'] = evaluate_squeeze(cnn, X_test, y_test, sq_fn, nf,
                                             EVAL_BATCH_SIZE)
            cond['fgsm'] = {}
            cond['pgd']  = {}
            for eps in EVAL_EPSILONS:
                X_f = fgsm_attack(cnn, X_test, y_test, epsilon=eps,
                                  batch_size=EVAL_BATCH_SIZE)
                m_f = evaluate_squeeze(cnn, X_f, y_test, sq_fn, nf, EVAL_BATCH_SIZE)
                cond['fgsm'][str(eps)] = m_f

                X_p = pgd_attack(cnn, X_test, y_test, epsilon=eps,
                                 alpha=eps * PGD_ALPHA_RATIO, n_iter=PGD_N_ITER,
                                 batch_size=EVAL_BATCH_SIZE, seed=SEED)
                m_p = evaluate_squeeze(cnn, X_p, y_test, sq_fn, nf, EVAL_BATCH_SIZE)
                cond['pgd'][str(eps)] = m_p

                print(f"      ε={eps}: FGSM acc={m_f['accuracy']:.4f}  "
                      f"PGD acc={m_p['accuracy']:.4f}")

            ds_results[str(win)] = cond

        results[ds] = ds_results

    return results


# ── Section C: Adversarial training — mix ratio ──────────────────────────────

def section_c_adv_ratio():
    """Ablate adversarial training mix ratio on TON_IoT."""
    print("\n" + "="*60)
    print(" SECTION C: Adversarial Training — Mix Ratio Ablation (TON_IoT)")
    print("="*60)

    ds = ADV_ABLATION_DATASET
    d  = _load_dataset_and_cnn(ds)
    X_train_sm, y_train_sm = d['pipeline'].transform(
        d['X_train_raw'], y=d['y_train'], apply_smote=True)
    cnn, X_test, y_test = d['cnn'], d['X_test'], d['y_test']
    y_clean = d['y_clean']
    nf = d['n_features']
    n_classes = d['n_classes']

    cw = compute_class_weight('balanced', classes=np.unique(y_train_sm), y=y_train_sm)
    cw_dict = dict(enumerate(cw))
    subsample = ADV_TRAIN_SUBSAMPLE.get(ds)
    pgd_alpha = ADV_TRAIN_EPSILON * ADV_TRAIN_PGD_ALPHA_RATIO

    out_dir = RESULTS_DIR / ds
    results = {}

    for ratio in ADV_RATIO_OPTIONS:
        ratio_key = str(ratio)
        print(f"\n  adv_ratio = {ratio}")
        model_path = out_dir / f'model_cnn_adv_ratio{int(ratio*100)}.h5'

        adv_cnn = build_cnn(nf, n_classes)
        adv_cnn = adversarial_train_cnn(
            model=adv_cnn,
            X_train=X_train_sm,
            y_train=y_train_sm,
            X_val=d['X_val_sc'],
            y_val=d['y_val'],
            epsilon=ADV_TRAIN_EPSILON,
            adv_ratio=ratio,
            epochs=ADV_TRAIN_EPOCHS,
            patience=ADV_TRAIN_PATIENCE,
            batch_size=ADV_TRAIN_BATCH,
            pgd_steps=ADV_TRAIN_PGD_STEPS,
            pgd_alpha=pgd_alpha,
            train_subsample=subsample,
            class_weight=cw_dict,
            out_path=model_path,
        )

        y_adv_clean = _cnn_predict(adv_cnn, X_test, nf, EVAL_BATCH_SIZE)
        clean_f1 = float(f1_score(y_test, y_adv_clean, average='macro', zero_division=0))
        clean_acc = float(accuracy_score(y_test, y_adv_clean))
        print(f"    Clean: acc={clean_acc:.4f}  f1={clean_f1:.4f}")

        cond = {'clean': {'accuracy': clean_acc, 'f1': clean_f1}, 'attacks': {}}
        for eps in EVAL_EPSILONS:
            X_f = fgsm_attack(adv_cnn, X_test, y_test, epsilon=eps,
                              batch_size=EVAL_BATCH_SIZE)
            y_f = _cnn_predict(adv_cnn, X_f, nf, EVAL_BATCH_SIZE)
            asr_f = _asr(y_test, y_adv_clean, y_f)

            X_p = pgd_attack(adv_cnn, X_test, y_test, epsilon=eps,
                             alpha=eps * PGD_ALPHA_RATIO, n_iter=PGD_N_ITER,
                             batch_size=EVAL_BATCH_SIZE, seed=SEED)
            y_p = _cnn_predict(adv_cnn, X_p, nf, EVAL_BATCH_SIZE)
            asr_p = _asr(y_test, y_adv_clean, y_p)

            cond['attacks'][str(eps)] = {
                'fgsm_asr': asr_f,
                'pgd_asr':  asr_p,
                'fgsm_f1':  float(f1_score(y_test, y_f, average='macro', zero_division=0)),
                'pgd_f1':   float(f1_score(y_test, y_p, average='macro', zero_division=0)),
            }
            print(f"    ε={eps}: FGSM_ASR={asr_f:.4f}  PGD_ASR={asr_p:.4f}")

        results[ratio_key] = cond

    return results


# ── Section D: Adversarial training — training epsilon ───────────────────────

def section_d_adv_eps():
    """Ablate adversarial training epsilon on TON_IoT."""
    print("\n" + "="*60)
    print(" SECTION D: Adversarial Training — Training Epsilon Ablation (TON_IoT)")
    print("="*60)

    ds = ADV_ABLATION_DATASET
    d  = _load_dataset_and_cnn(ds)
    X_train_sm, y_train_sm = d['pipeline'].transform(
        d['X_train_raw'], y=d['y_train'], apply_smote=True)
    X_test, y_test = d['X_test'], d['y_test']
    y_clean = d['y_clean']
    nf = d['n_features']
    n_classes = d['n_classes']

    cw = compute_class_weight('balanced', classes=np.unique(y_train_sm), y=y_train_sm)
    cw_dict = dict(enumerate(cw))
    subsample = ADV_TRAIN_SUBSAMPLE.get(ds)

    out_dir = RESULTS_DIR / ds
    results = {}

    for train_eps in ADV_EPS_OPTIONS:
        eps_key = str(train_eps)
        print(f"\n  train_eps = {train_eps}")
        model_path = out_dir / f'model_cnn_adv_traineps{int(train_eps*100)}.h5'
        pgd_alpha  = train_eps * ADV_TRAIN_PGD_ALPHA_RATIO

        adv_cnn = build_cnn(nf, n_classes)
        adv_cnn = adversarial_train_cnn(
            model=adv_cnn,
            X_train=X_train_sm,
            y_train=y_train_sm,
            X_val=d['X_val_sc'],
            y_val=d['y_val'],
            epsilon=train_eps,
            adv_ratio=0.5,   # fixed at current default
            epochs=ADV_TRAIN_EPOCHS,
            patience=ADV_TRAIN_PATIENCE,
            batch_size=ADV_TRAIN_BATCH,
            pgd_steps=ADV_TRAIN_PGD_STEPS,
            pgd_alpha=pgd_alpha,
            train_subsample=subsample,
            class_weight=cw_dict,
            out_path=model_path,
        )

        y_adv_clean = _cnn_predict(adv_cnn, X_test, nf, EVAL_BATCH_SIZE)
        clean_f1  = float(f1_score(y_test, y_adv_clean, average='macro', zero_division=0))
        clean_acc = float(accuracy_score(y_test, y_adv_clean))
        print(f"    Clean: acc={clean_acc:.4f}  f1={clean_f1:.4f}")

        cond = {'clean': {'accuracy': clean_acc, 'f1': clean_f1}, 'attacks': {}}
        for eps in EVAL_EPSILONS:
            X_f = fgsm_attack(adv_cnn, X_test, y_test, epsilon=eps,
                              batch_size=EVAL_BATCH_SIZE)
            y_f = _cnn_predict(adv_cnn, X_f, nf, EVAL_BATCH_SIZE)
            asr_f = _asr(y_test, y_adv_clean, y_f)

            X_p = pgd_attack(adv_cnn, X_test, y_test, epsilon=eps,
                             alpha=eps * PGD_ALPHA_RATIO, n_iter=PGD_N_ITER,
                             batch_size=EVAL_BATCH_SIZE, seed=SEED)
            y_p = _cnn_predict(adv_cnn, X_p, nf, EVAL_BATCH_SIZE)
            asr_p = _asr(y_test, y_adv_clean, y_p)

            cond['attacks'][str(eps)] = {
                'fgsm_asr': asr_f, 'pgd_asr': asr_p,
                'fgsm_f1':  float(f1_score(y_test, y_f, average='macro', zero_division=0)),
                'pgd_f1':   float(f1_score(y_test, y_p, average='macro', zero_division=0)),
            }
            print(f"    ε={eps}: FGSM_ASR={asr_f:.4f}  PGD_ASR={asr_p:.4f}")

        results[eps_key] = cond

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Defense ablation study for adversarial robustness (Paper 2)')
    parser.add_argument('--section', default='all',
                        choices=['A', 'B', 'C', 'D', 'all'],
                        help='Which ablation section to run (default: all)')
    args = parser.parse_args()

    # Load existing results if any (to allow incremental runs)
    if ABLATION_RESULTS_PATH.exists():
        with open(ABLATION_RESULTS_PATH) as f:
            results = json.load(f)
    else:
        results = {}

    if args.section in ('A', 'all'):
        results['bit_depth'] = section_a_bit_depth()
        _save_json(results, ABLATION_RESULTS_PATH)

    if args.section in ('B', 'all'):
        results['median_window'] = section_b_window()
        _save_json(results, ABLATION_RESULTS_PATH)

    if args.section in ('C', 'all'):
        results['adv_ratio'] = section_c_adv_ratio()
        _save_json(results, ABLATION_RESULTS_PATH)

    if args.section in ('D', 'all'):
        results['adv_train_eps'] = section_d_adv_eps()
        _save_json(results, ABLATION_RESULTS_PATH)

    print(f"\nAll ablation results saved → {ABLATION_RESULTS_PATH}")


if __name__ == '__main__':
    main()
