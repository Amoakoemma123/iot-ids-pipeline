"""
experiments/adversarial_robustness/run_binary_adv.py
Binary adversarial training ablation on CICIDS-2017 (Paper 2 — top-tier fix).

Motivation
----------
The CICIDS-2017 adversarially trained CNN achieved F1=0.1799 (vs baseline 0.7168).
This experiment isolates the cause: is the collapse due to:
  (a) the 15-class extreme imbalance (e.g., Heartbleed=11 samples), OR
  (b) the adversarial training procedure itself disrupting gradient flow?

Method
------
1. Collapse CICIDS-2017 labels → binary (0=BENIGN, 1=ATTACK)
2. Build fresh CNN for binary classification (n_classes=2)
3. Apply same PGD adversarial training (ε=0.1, ratio=0.5, 50 epochs)
4. Compare binary-adv-trained clean F1 vs multi-class adv-trained F1=0.1799
5. Also evaluate binary model under attacks for robustness comparison

Hypothesis: if binary model achieves much higher clean F1 with adversarial training,
it confirms the collapse in the 15-class case is caused by class imbalance + gradient
dilution, not by the adversarial training method itself.

Usage
-----
    python -m experiments.adversarial_robustness.run_binary_adv
"""
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
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

from src.data   import load_dataset
from src.models import build_cnn

from experiments.adversarial_robustness.config import (
    DATASET_PATHS, EPSILONS, PGD_ALPHA_RATIO, PGD_N_ITER,
    ADV_TRAIN_EPSILON, ADV_TRAIN_RATIO, ADV_TRAIN_EPOCHS, ADV_TRAIN_PATIENCE,
    ADV_TRAIN_BATCH, ADV_TRAIN_PGD_STEPS, ADV_TRAIN_PGD_ALPHA_RATIO,
    ADV_TRAIN_SUBSAMPLE, EVAL_BATCH_SIZE, SEED, BASELINE_DIR, RESULTS_DIR,
)
from experiments.adversarial_robustness.attacks.fgsm import fgsm_attack
from experiments.adversarial_robustness.attacks.pgd  import pgd_attack
from experiments.adversarial_robustness.defenses.adversarial_training import (
    adversarial_train_cnn)
from experiments.baseline.config import SPLIT

DATASET = 'CICIDS2017'
OUT_DIR = RESULTS_DIR / DATASET


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


def to_binary(y: np.ndarray, benign_class_id: int) -> np.ndarray:
    """Map benign_class_id → 0, all others → 1."""
    return (y != benign_class_id).astype(np.int64)


def main():
    print(f"\n{'='*60}")
    print(f" Binary Adversarial Training Ablation — {DATASET}")
    print(f"{'='*60}")

    # ── Load data ─────────────────────────────────────────────────────────
    print("\n[1] Loading data…")
    csv_path, label_col = DATASET_PATHS[DATASET]

    # CICIDS2017 (2.83M rows) causes OOM on dropna copy with pandas.
    # Use a pre-sampled 600K-row parquet created by prep_cicids_sample.py.
    import tempfile
    _sample_path = Path(tempfile.gettempdir()) / 'cicids_600k_sample.parquet'
    if not _sample_path.exists():
        raise FileNotFoundError(
            f"Pre-sampled file not found: {_sample_path}\n"
            "Run first:  python -m experiments.adversarial_robustness.prep_cicids_sample")
    print(f"  Using pre-sampled CICIDS2017: {_sample_path}")
    data = load_dataset(str(_sample_path), label_col=label_col, **SPLIT)

    pipeline = joblib.load(BASELINE_DIR / 'results' / DATASET / 'pipeline.pkl')

    X_train_sc, _ = pipeline.transform(data['X_train'], y=data['y_train'])
    X_val_sc,   _ = pipeline.transform(data['X_val'],   y=data['y_val'])
    X_test,     _ = pipeline.transform(data['X_test'],  y=data['y_test'])
    n_features = X_test.shape[1]
    n_classes_multi = len(data['class_names'])

    print(f"  n_features={n_features}  n_classes (multi)={n_classes_multi}")
    print(f"  Class names: {data['class_names']}")

    y_train_multi = data['y_train']
    y_val_multi   = data['y_val']
    y_test_multi  = data['y_test']

    # Find benign class ID (encoded label 0 corresponds to 'BENIGN' in CICIDS2017)
    class_names = data['class_names']
    benign_id = next((i for i, n in enumerate(class_names)
                      if 'benign' in n.lower() or 'normal' in n.lower()), 0)
    print(f"  Benign class ID = {benign_id} ({class_names[benign_id]})")

    # Convert to binary
    y_train_bin = to_binary(y_train_multi, benign_id)
    y_val_bin   = to_binary(y_val_multi,   benign_id)
    y_test_bin  = to_binary(y_test_multi,  benign_id)

    n_benign_test  = int((y_test_bin == 0).sum())
    n_attack_test  = int((y_test_bin == 1).sum())
    print(f"  Binary test set: BENIGN={n_benign_test:,}  ATTACK={n_attack_test:,}")

    # ── Baseline binary CNN (standard training, no adversarial) ───────────
    print("\n[2] Training baseline binary CNN (standard, no adversarial)…")
    binary_baseline_path = OUT_DIR / 'model_cnn_binary_baseline.h5'

    if binary_baseline_path.exists():
        print(f"  Loading saved baseline binary model…")
        binary_baseline_cnn = tf.keras.models.load_model(str(binary_baseline_path))
    else:
        # Apply SMOTE on binary labels for balanced training
        X_train_sm_bin, y_train_sm_bin = pipeline.transform(
            data['X_train'], y=y_train_bin, apply_smote=True)
        print(f"  SMOTE: {len(X_train_sc):,} → {len(X_train_sm_bin):,} samples")

        binary_baseline_cnn = build_cnn(n_features, n_classes=2)
        cb = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=ADV_TRAIN_PATIENCE,
                restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(
                str(binary_baseline_path), save_best_only=True, verbose=0),
        ]
        cw = compute_class_weight('balanced',
                                  classes=np.unique(y_train_sm_bin),
                                  y=y_train_sm_bin)
        binary_baseline_cnn.fit(
            X_train_sm_bin.reshape(-1, n_features, 1),
            y_train_sm_bin,
            validation_data=(X_val_sc.reshape(-1, n_features, 1), y_val_bin),
            epochs=ADV_TRAIN_EPOCHS,
            batch_size=ADV_TRAIN_BATCH,
            class_weight=dict(enumerate(cw)),
            callbacks=cb,
            verbose=1,
        )
        print(f"  Saved → {binary_baseline_path}")

    y_bin_base_pred = _cnn_predict(binary_baseline_cnn, X_test, n_features,
                                   EVAL_BATCH_SIZE)
    bin_base_f1  = float(f1_score(y_test_bin, y_bin_base_pred,
                                  average='macro', zero_division=0))
    bin_base_acc = float(accuracy_score(y_test_bin, y_bin_base_pred))
    print(f"  Binary baseline: acc={bin_base_acc:.4f}  f1={bin_base_f1:.4f}")

    # ── Adversarially trained binary CNN ──────────────────────────────────
    print("\n[3] PGD adversarial training — binary CNN…")
    binary_adv_path = OUT_DIR / 'model_cnn_binary_adv.h5'

    if binary_adv_path.exists():
        print(f"  Loading saved binary adversarial model…")
        binary_adv_cnn = tf.keras.models.load_model(str(binary_adv_path))
    else:
        # SMOTE on binary training labels
        X_train_sm_bin, y_train_sm_bin = pipeline.transform(
            data['X_train'], y=y_train_bin, apply_smote=True)

        cw = compute_class_weight('balanced',
                                  classes=np.unique(y_train_sm_bin),
                                  y=y_train_sm_bin)
        cw_dict = dict(enumerate(cw))

        pgd_alpha = ADV_TRAIN_EPSILON * ADV_TRAIN_PGD_ALPHA_RATIO
        subsample = ADV_TRAIN_SUBSAMPLE.get(DATASET)

        binary_adv_cnn = build_cnn(n_features, n_classes=2)
        binary_adv_cnn = adversarial_train_cnn(
            model=binary_adv_cnn,
            X_train=X_train_sm_bin,
            y_train=y_train_sm_bin,
            X_val=X_val_sc,
            y_val=y_val_bin,
            epsilon=ADV_TRAIN_EPSILON,
            adv_ratio=ADV_TRAIN_RATIO,
            epochs=ADV_TRAIN_EPOCHS,
            patience=ADV_TRAIN_PATIENCE,
            batch_size=ADV_TRAIN_BATCH,
            pgd_steps=ADV_TRAIN_PGD_STEPS,
            pgd_alpha=pgd_alpha,
            train_subsample=subsample,
            class_weight=cw_dict,
            out_path=binary_adv_path,
        )
        binary_adv_cnn.save(str(binary_adv_path))
        print(f"  Binary adv model saved → {binary_adv_path}")

    y_bin_adv_clean = _cnn_predict(binary_adv_cnn, X_test, n_features,
                                   EVAL_BATCH_SIZE)
    bin_adv_f1  = float(f1_score(y_test_bin, y_bin_adv_clean,
                                 average='macro', zero_division=0))
    bin_adv_acc = float(accuracy_score(y_test_bin, y_bin_adv_clean))
    print(f"  Binary adv-trained: acc={bin_adv_acc:.4f}  f1={bin_adv_f1:.4f}")

    # ── Attack evaluation on adversarially trained binary model ───────────
    print("\n[4] Evaluating binary adv model under attacks…")
    attack_results = {}
    for eps in EPSILONS:
        X_f = fgsm_attack(binary_adv_cnn, X_test, y_test_bin, epsilon=eps,
                          batch_size=EVAL_BATCH_SIZE)
        y_f = _cnn_predict(binary_adv_cnn, X_f, n_features, EVAL_BATCH_SIZE)
        asr_f = _asr(y_test_bin, y_bin_adv_clean, y_f)

        X_p = pgd_attack(binary_adv_cnn, X_test, y_test_bin, epsilon=eps,
                         alpha=eps * PGD_ALPHA_RATIO, n_iter=PGD_N_ITER,
                         batch_size=EVAL_BATCH_SIZE, seed=SEED)
        y_p = _cnn_predict(binary_adv_cnn, X_p, n_features, EVAL_BATCH_SIZE)
        asr_p = _asr(y_test_bin, y_bin_adv_clean, y_p)

        attack_results[str(eps)] = {
            'fgsm_asr': asr_f,
            'pgd_asr':  asr_p,
            'fgsm_f1':  float(f1_score(y_test_bin, y_f, average='macro', zero_division=0)),
            'pgd_f1':   float(f1_score(y_test_bin, y_p, average='macro', zero_division=0)),
        }
        print(f"  ε={eps}: FGSM_ASR={asr_f:.4f}  PGD_ASR={asr_p:.4f}")

    # ── Comparison summary ─────────────────────────────────────────────────
    results = {
        'dataset':  DATASET,
        'purpose':  'Isolate cause of adv_training collapse in 15-class setting',
        'binary_baseline': {
            'clean_accuracy': bin_base_acc,
            'clean_f1':       bin_base_f1,
        },
        'binary_adversarial_training': {
            'clean_accuracy': bin_adv_acc,
            'clean_f1':       bin_adv_f1,
            'attacks':        attack_results,
        },
        'multiclass_reference': {
            'baseline_f1':      0.7168,   # from full robustness results
            'adv_trained_f1':   0.1799,   # collapse result
            'n_classes':        n_classes_multi,
        },
        'interpretation': (
            f"Binary adv-trained F1={bin_adv_f1:.4f} vs "
            f"multi-class adv-trained F1=0.1799. "
            f"{'Confirmed: collapse is due to class imbalance, not adv training method.' if bin_adv_f1 > 0.5 else 'Collapse persists even in binary case — architecture-level issue.'}"
        ),
    }

    _save_json(results, OUT_DIR / 'binary_adv_results.json')
    print(f"\nDone — results saved → {OUT_DIR / 'binary_adv_results.json'}")

    # Print comparison
    print("\n" + "="*50)
    print(" COMPARISON SUMMARY")
    print("="*50)
    print(f"  Binary baseline clean F1:         {bin_base_f1:.4f}")
    print(f"  Binary adv-trained clean F1:       {bin_adv_f1:.4f}")
    print(f"  Multi-class baseline clean F1:     0.7168")
    print(f"  Multi-class adv-trained clean F1:  0.1799  ← collapse")
    delta = bin_adv_f1 - 0.1799
    print(f"  Delta (binary vs multi-class adv): {delta:+.4f}")
    if bin_adv_f1 > 0.5:
        print("  => Class imbalance is the PRIMARY cause of adv training collapse.")
    else:
        print("  => Architecture/training issue persists even in binary case.")


if __name__ == '__main__':
    main()
