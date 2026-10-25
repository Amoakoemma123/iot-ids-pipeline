"""
experiments/adversarial_robustness/run_ccat.py

Run Class-Conditional Adversarial Training (CCAT) on CICIDS-2017 and compare
results to standard adversarial training (already in robustness_results.json).

Outputs
-------
  experiments/adversarial_robustness/results/CICIDS2017/ccat_results.json
"""
import sys, json, time
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score

# GPU memory growth
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.baseline.config import DATASETS, SPLIT, PREPROCESSING, PREPROCESSING_OVERRIDES, SEED
from src.data.loader import load_dataset
from src.data.preprocessing import PreprocessingPipeline
from src.models.cnn import build_cnn
from experiments.adversarial_robustness.defenses.ccat import ccat_train_cnn

np.random.seed(SEED)
tf.random.set_seed(SEED)

DS         = 'CICIDS2017'
OUT_DIR    = Path(__file__).resolve().parent / 'results' / DS
MODELS_DIR = ROOT / 'experiments' / 'baseline' / 'results' / 'models' / DS

# ── CICIDS-2017 original class counts (before SMOTE) ─────────────────────────
# Rare = original count < 1000 samples
# Heartbleed=11, SQL Injection=21, Infiltration=36, Web XSS=652
# These are labels AFTER LabelEncoder (0-based alphabetic order):
# ['benign','bot','ddos','dos goldeneye','dos hulk','dos slowhttptest',
#  'dos slowloris','ftp-patator','heartbleed','infiltration','portscan',
#  'ssh-patator','web attack brute force','web attack sql injection',
#  'web attack xss']
#   idx:    0      1     2        3             4          5
#           6          7           8          9         10
#           11          12                    13                 14
RARE_CLASS_LABELS = {8, 9, 13, 14}   # heartbleed, infiltration, sql, xss
RARE_THRESHOLD    = 1000             # original sample count threshold

# CCAT hyperparameters (same ε as standard adversarial training)
EPSILON    = 0.1
ADV_RATIO  = 0.5
EPOCHS     = 50
PATIENCE   = 10
BATCH_SIZE = 256
PGD_STEPS  = 7
SUBSAMPLE  = 100_000
EPSILONS   = [0.01, 0.05, 0.1, 0.2, 0.3]


def compute_class_weights(y_train, n_classes):
    """Inverse-frequency class weights, capped at 100× majority."""
    counts = np.bincount(y_train, minlength=n_classes).astype(float)
    counts = np.where(counts == 0, 1, counts)
    weights = counts.sum() / (n_classes * counts)
    weights = np.clip(weights, 1.0, 100.0)
    return {i: float(w) for i, w in enumerate(weights)}


def pgd_attack(model, X, y_ohe, eps, steps=40, batch_size=512):
    """PGD attack for evaluation (same as transfer script)."""
    alpha  = eps / 10.0
    X_adv  = X.copy().astype(np.float32)
    n      = len(X)
    for start in range(0, n, batch_size):
        X_b = tf.constant(X[start:start+batch_size])
        y_b = tf.constant(y_ohe[start:start+batch_size])
        x_t = X_b + tf.random.uniform(tf.shape(X_b), -eps, eps)
        x_t = tf.clip_by_value(x_t, 0., 1.)
        for _ in range(steps):
            with tf.GradientTape() as tape:
                tape.watch(x_t)
                loss = tf.keras.losses.categorical_crossentropy(y_b, model(x_t, training=False))
            grad  = tape.gradient(loss, x_t)
            x_t   = x_t + alpha * tf.sign(grad)
            x_t   = tf.clip_by_value(x_t, X_b - eps, X_b + eps)
            x_t   = tf.clip_by_value(x_t, 0., 1.)
        X_adv[start:start+batch_size] = x_t.numpy()
    return X_adv


def main():
    print("=" * 60)
    print("  CCAT — CICIDS-2017")
    print("=" * 60)

    # ── Load and preprocess ───────────────────────────────────────────────
    print("\n[1] Loading data...")
    csv_path, label_col = DATASETS[DS]
    data = load_dataset(csv_path, label_col=label_col, **SPLIT)

    pp_kwargs = {**PREPROCESSING}
    pp_kwargs.update(PREPROCESSING_OVERRIDES.get(DS, {}))
    valid_keys = {'mi_threshold','smote_k','smote_max_per_class',
                  'skip_rfecv','rfecv_subsample','mi_subsample','seed'}
    pp_kwargs  = {k: v for k, v in pp_kwargs.items() if k in valid_keys}

    pipeline = PreprocessingPipeline(**pp_kwargs)
    print("  Fitting pipeline...")
    X_train_sm, y_train_sm = pipeline.fit_transform(
        data['X_train'], data['y_train'], feature_names=data['feature_names'])
    X_val_sc,  _           = pipeline.transform(data['X_val'],  y=data['y_val'])
    X_test,    _           = pipeline.transform(data['X_test'], y=data['y_test'])
    y_val  = data['y_val']
    y_test = data['y_test']

    n_features = X_test.shape[1]
    n_classes  = len(data['class_names'])
    print(f"  Features={n_features}  Classes={n_classes}  "
          f"Train(SMOTE)={len(X_train_sm):,}  Test={len(X_test):,}")
    print(f"  Class names: {list(enumerate(data['class_names']))}")

    # ── Verify rare class indices ──────────────────────────────────────────
    print(f"\n  Rare class indices: {RARE_CLASS_LABELS}")
    for idx in sorted(RARE_CLASS_LABELS):
        count_orig = int((data['y_train'] == idx).sum())
        count_sm   = int((y_train_sm == idx).sum())
        print(f"    [{idx}] {data['class_names'][idx]}: "
              f"orig={count_orig}  post-SMOTE={count_sm}")

    # ── Class weights ──────────────────────────────────────────────────────
    class_weights = compute_class_weights(y_train_sm, n_classes)
    print(f"\n  Class weights (sample):")
    for i, w in list(class_weights.items())[:5]:
        print(f"    class {i} ({data['class_names'][i]}): {w:.2f}")

    # ── Build CNN (fresh weights, same architecture as target) ─────────────
    print("\n[2] Building fresh CNN for CCAT...")
    ccat_model = build_cnn(n_features, n_classes)
    ccat_model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    ccat_path = OUT_DIR / 'model_cnn_ccat.h5'

    # ── Run CCAT ───────────────────────────────────────────────────────────
    print(f"\n[3] Running CCAT (ε={EPSILON}, adv_ratio={ADV_RATIO}, "
          f"pgd_steps={PGD_STEPS}, subsample={SUBSAMPLE:,})...")
    t0 = time.time()
    ccat_train_cnn(
        model           = ccat_model,
        X_train         = X_train_sm,
        y_train         = y_train_sm,
        X_val           = X_val_sc,
        y_val           = y_val,
        epsilon         = EPSILON,
        adv_ratio       = ADV_RATIO,
        epochs          = EPOCHS,
        patience        = PATIENCE,
        batch_size      = BATCH_SIZE,
        rare_classes    = RARE_CLASS_LABELS,
        pgd_steps       = PGD_STEPS,
        train_subsample = SUBSAMPLE,
        class_weight    = class_weights,
        out_path        = ccat_path,
    )
    print(f"  CCAT training done in {(time.time()-t0)/60:.1f} min")

    # ── Evaluate on clean test set ─────────────────────────────────────────
    print("\n[4] Evaluating CCAT model...")
    X_test_3d = X_test.reshape(-1, n_features, 1)
    pred_clean = np.argmax(ccat_model.predict(X_test_3d, batch_size=512, verbose=0), axis=1)
    acc_clean  = float(accuracy_score(y_test, pred_clean))
    f1_clean   = float(f1_score(y_test, pred_clean, average='macro', zero_division=0))
    print(f"  CCAT clean  — Acc={acc_clean:.4f}  Macro F1={f1_clean:.4f}")

    # Per-class F1 (rare vs common)
    f1_per_class = f1_score(y_test, pred_clean, average=None, zero_division=0)
    rare_f1   = float(np.mean([f1_per_class[i] for i in RARE_CLASS_LABELS
                               if i < len(f1_per_class)]))
    common_f1 = float(np.mean([f1_per_class[i] for i in range(n_classes)
                                if i not in RARE_CLASS_LABELS]))
    print(f"  Rare-class F1={rare_f1:.4f}  Common-class F1={common_f1:.4f}")

    # ── Evaluate under adversarial attacks ─────────────────────────────────
    print("\n[5] Adversarial evaluation...")
    y_ohe = tf.keras.utils.to_categorical(y_test, n_classes)
    # Use subsample for speed
    rng   = np.random.RandomState(SEED)
    idx_e = rng.choice(len(X_test), min(3000, len(X_test)), replace=False)
    X_ev  = X_test[idx_e]
    y_ev  = y_test[idx_e]
    y_ev_ohe = tf.keras.utils.to_categorical(y_ev, n_classes)
    X_ev_3d  = X_ev.reshape(-1, n_features, 1)

    attack_results = {}
    for eps in EPSILONS:
        print(f"  ε={eps}  PGD-40...", end=' ', flush=True)
        t0 = time.time()
        X_adv = pgd_attack(ccat_model, X_ev_3d, y_ev_ohe, eps, steps=40)
        pred_adv = np.argmax(ccat_model.predict(X_adv, batch_size=512, verbose=0), axis=1)
        acc_adv  = float(accuracy_score(y_ev, pred_adv))
        f1_adv   = float(f1_score(y_ev, pred_adv, average='macro', zero_division=0))
        correct  = pred_clean[idx_e] == y_ev
        asr      = float(np.mean(pred_adv[correct] != y_ev[correct])) if correct.any() else 0.
        print(f"Acc={acc_adv:.4f}  F1={f1_adv:.4f}  ASR={asr:.4f}  [{time.time()-t0:.1f}s]")
        attack_results[str(eps)] = {
            'accuracy': acc_adv, 'f1': f1_adv, 'asr': asr
        }

    # ── Save results ───────────────────────────────────────────────────────
    # Load standard adversarial training results for comparison
    std_res = json.load(open(OUT_DIR / 'robustness_results.json'))
    std_adv = std_res['adversarial_training']

    ccat_results = {
        'method': 'CCAT',
        'hyperparameters': {
            'epsilon': EPSILON, 'adv_ratio': ADV_RATIO,
            'pgd_steps': PGD_STEPS, 'subsample': SUBSAMPLE,
            'rare_class_threshold': RARE_THRESHOLD,
            'rare_classes': sorted(RARE_CLASS_LABELS),
        },
        'clean': {
            'accuracy': acc_clean, 'f1': f1_clean,
            'rare_class_f1': rare_f1, 'common_class_f1': common_f1,
            'per_class_f1': f1_per_class.tolist(),
        },
        'attacks': attack_results,
        'comparison': {
            'standard_adv_training_clean_f1':  std_adv['clean']['f1'],
            'standard_adv_training_clean_acc': std_adv['clean']['accuracy'],
            'ccat_clean_f1':  f1_clean,
            'ccat_clean_acc': acc_clean,
            'f1_improvement': round(f1_clean - std_adv['clean']['f1'], 4),
            'acc_improvement': round(acc_clean - std_adv['clean']['accuracy'], 4),
        }
    }

    out_path = OUT_DIR / 'ccat_results.json'
    with open(out_path, 'w') as f:
        json.dump(ccat_results, f, indent=2)
    print(f"\nResults saved → {out_path}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("CCAT vs Standard Adversarial Training — CICIDS-2017")
    print("=" * 60)
    comp = ccat_results['comparison']
    print(f"  Standard AT clean F1 : {comp['standard_adv_training_clean_f1']:.4f}")
    print(f"  CCAT       clean F1  : {comp['ccat_clean_f1']:.4f}  "
          f"(+{comp['f1_improvement']:.4f})")
    print(f"  Standard AT clean Acc: {comp['standard_adv_training_clean_acc']:.4f}")
    print(f"  CCAT       clean Acc : {comp['ccat_clean_acc']:.4f}  "
          f"(+{comp['acc_improvement']:.4f})")
    print(f"\n  CCAT robust F1 under PGD:")
    for eps_str, v in attack_results.items():
        std_f1 = std_adv['attacks'].get(eps_str, {}).get('pgd', {}).get('f1', 'N/A')
        print(f"    ε={eps_str}:  CCAT F1={v['f1']:.4f}  "
              f"Std-AT F1={std_f1 if isinstance(std_f1, str) else f'{std_f1:.4f}'}")


if __name__ == '__main__':
    main()
