"""
run_ccat_sensitivity.py
=======================
CCAT hyperparameter sensitivity sweep on CICIDS-2017.

Varies:
  - theta (rare-class threshold): {500, 1000, 2000}
  - adv_ratio: {0.3, 0.5, 0.7}
= 9 configurations total.

For each config:
  - Trains a fresh CCAT CNN (same architecture as Paper 2 Table 8)
  - Evaluates clean macro F1 on the full held-out test set
  - Evaluates robust macro F1 under PGD-40 at ε ∈ {0.01, 0.1, 0.3}
  - Saves config results incrementally so progress is not lost

Output: experiments/adversarial_robustness/results/CICIDS2017/ccat_sensitivity.json
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

DS       = 'CICIDS2017'
OUT_DIR  = Path(__file__).resolve().parent / 'results' / DS
OUT_FILE = OUT_DIR / 'ccat_sensitivity.json'

# CICIDS-2017 TOTAL dataset counts per class (alphabetical label order):
# idx  class                     total
#  0   BENIGN                    2,273,097
#  1   Bot                       1,966
#  2   DDoS                      128,027
#  3   DoS GoldenEye             10,293
#  4   DoS Hulk                  231,073
#  5   DoS Slowhttptest          5,499
#  6   DoS slowloris             5,796
#  7   FTP-Patator               7,938
#  8   Heartbleed                11
#  9   Infiltration              36
# 10   PortScan                  158,930
# 11   SSH-Patator               5,897
# 12   Web Attack-Brute Force    1,507
# 13   Web Attack-SQL Injection  21
# 14   Web Attack-XSS            652
CLASS_COUNTS = {
    0: 2273097, 1: 1966,  2: 128027, 3: 10293,  4: 231073,
    5: 5499,    6: 5796,  7: 7938,   8: 11,     9: 36,
    10: 158930, 11: 5897, 12: 1507,  13: 21,    14: 652,
}

# Fixed training params (same as Paper 2)
EPSILON    = 0.1
EPOCHS     = 50
PATIENCE   = 10
BATCH_SIZE = 256
PGD_STEPS  = 7
SUBSAMPLE  = 100_000

# Sweep grid
THETAS     = [500, 1000, 2000]
ADV_RATIOS = [0.3, 0.5, 0.7]
EVAL_EPS   = [0.01, 0.1, 0.3]   # PGD ε values for robustness eval


def get_rare_classes(theta):
    """Return set of class indices with total count < theta."""
    return {idx for idx, cnt in CLASS_COUNTS.items() if cnt < theta}


def compute_class_weights(y_train, n_classes):
    counts  = np.bincount(y_train, minlength=n_classes).astype(float)
    counts  = np.where(counts == 0, 1, counts)
    weights = counts.sum() / (n_classes * counts)
    weights = np.clip(weights, 1.0, 100.0)
    return {i: float(w) for i, w in enumerate(weights)}


def pgd_attack(model, X_3d, y_ohe, eps, steps=40, batch_size=512):
    alpha = eps / 10.0
    X_adv = X_3d.copy().astype(np.float32)
    n     = len(X_3d)
    for start in range(0, n, batch_size):
        X_b = tf.constant(X_3d[start:start+batch_size])
        y_b = tf.constant(y_ohe[start:start+batch_size])
        x_t = X_b + tf.random.uniform(tf.shape(X_b), -eps, eps)
        x_t = tf.clip_by_value(x_t, 0., 1.)
        for _ in range(steps):
            with tf.GradientTape() as tape:
                tape.watch(x_t)
                loss = tf.keras.losses.categorical_crossentropy(
                    y_b, model(x_t, training=False))
            grad = tape.gradient(loss, x_t)
            x_t  = x_t + alpha * tf.sign(grad)
            x_t  = tf.clip_by_value(x_t, X_b - eps, X_b + eps)
            x_t  = tf.clip_by_value(x_t, 0., 1.)
        X_adv[start:start+batch_size] = x_t.numpy()
    return X_adv


def main():
    # ── Load and preprocess ONCE ──────────────────────────────────────────────
    print("=" * 65)
    print("  CCAT Sensitivity Sweep — CICIDS-2017")
    print(f"  Grid: θ={THETAS} × adv_ratio={ADV_RATIOS}")
    print(f"  = {len(THETAS)*len(ADV_RATIOS)} configurations")
    print("=" * 65)

    print("\n[1/3] Loading & preprocessing data (once)...")
    csv_path, label_col = DATASETS[DS]
    data = load_dataset(csv_path, label_col=label_col, **SPLIT)

    pp_kwargs = {**PREPROCESSING}
    pp_kwargs.update(PREPROCESSING_OVERRIDES.get(DS, {}))
    valid_keys = {'mi_threshold', 'smote_k', 'smote_max_per_class',
                  'skip_rfecv', 'rfecv_subsample', 'mi_subsample', 'seed'}
    pp_kwargs  = {k: v for k, v in pp_kwargs.items() if k in valid_keys}

    pipeline = PreprocessingPipeline(**pp_kwargs)
    print("  Fitting pipeline (MI + SMOTE, RFECV skipped)...")
    X_train_sm, y_train_sm = pipeline.fit_transform(
        data['X_train'], data['y_train'], feature_names=data['feature_names'])
    X_val_sc, _  = pipeline.transform(data['X_val'],  y=data['y_val'])
    X_test, _    = pipeline.transform(data['X_test'], y=data['y_test'])
    y_val  = data['y_val']
    y_test = data['y_test']

    n_features = X_test.shape[1]
    n_classes  = len(data['class_names'])
    print(f"  Features={n_features}  Classes={n_classes}  "
          f"Train(SMOTE)={len(X_train_sm):,}  Test={len(X_test):,}")

    X_test_3d = X_test.reshape(-1, n_features, 1)

    # Fixed eval subsample for PGD (3000 samples, same seed as Paper 2)
    rng     = np.random.RandomState(SEED)
    idx_ev  = rng.choice(len(X_test), min(3000, len(X_test)), replace=False)
    X_ev_3d = X_test_3d[idx_ev]
    y_ev    = y_test[idx_ev]
    y_ev_ohe = tf.keras.utils.to_categorical(y_ev, n_classes)

    # ── Load existing results (resume if interrupted) ──────────────────────────
    if OUT_FILE.exists():
        with open(OUT_FILE) as f:
            all_results = json.load(f)
        print(f"\n  Resuming — {len(all_results)} configs already done.")
    else:
        all_results = []

    done_keys = {(r['theta'], r['adv_ratio']) for r in all_results}

    # ── Sweep ─────────────────────────────────────────────────────────────────
    configs = [(t, a) for t in THETAS for a in ADV_RATIOS]
    total   = len(configs)

    for run_idx, (theta, adv_ratio) in enumerate(configs, 1):
        if (theta, adv_ratio) in done_keys:
            print(f"\n[Skip {run_idx}/{total}] θ={theta}, adv_ratio={adv_ratio} (already done)")
            continue

        rare_classes = get_rare_classes(theta)
        print(f"\n{'='*65}")
        print(f"[Run {run_idx}/{total}] θ={theta}, adv_ratio={adv_ratio}")
        print(f"  Rare classes ({len(rare_classes)}): {sorted(rare_classes)}")
        print(f"  → {[data['class_names'][i] for i in sorted(rare_classes)]}")
        print(f"{'='*65}")

        class_weights = compute_class_weights(y_train_sm, n_classes)

        # Build fresh CNN
        model = build_cnn(n_features, n_classes)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Train CCAT
        t0 = time.time()
        ccat_train_cnn(
            model           = model,
            X_train         = X_train_sm,
            y_train         = y_train_sm,
            X_val           = X_val_sc,
            y_val           = y_val,
            epsilon         = EPSILON,
            adv_ratio       = adv_ratio,
            epochs          = EPOCHS,
            patience        = PATIENCE,
            batch_size      = BATCH_SIZE,
            rare_classes    = rare_classes,
            pgd_steps       = PGD_STEPS,
            train_subsample = SUBSAMPLE,
            class_weight    = class_weights,
            out_path        = None,          # don't save individual models
        )
        train_time = time.time() - t0
        print(f"  Training time: {train_time/60:.1f} min")

        # Clean evaluation
        pred_clean = np.argmax(
            model.predict(X_test_3d, batch_size=512, verbose=0), axis=1)
        f1_clean  = float(f1_score(y_test, pred_clean, average='macro', zero_division=0))
        acc_clean = float(accuracy_score(y_test, pred_clean))

        f1_per_class = f1_score(y_test, pred_clean, average=None, zero_division=0)
        rare_f1   = float(np.mean([f1_per_class[i] for i in rare_classes
                                   if i < len(f1_per_class)]))
        common_f1 = float(np.mean([f1_per_class[i] for i in range(n_classes)
                                    if i not in rare_classes]))

        print(f"  Clean — Macro F1={f1_clean:.4f}  Acc={acc_clean:.4f}  "
              f"Rare F1={rare_f1:.4f}  Common F1={common_f1:.4f}")

        # Robust evaluation (PGD-40) at 3 ε values
        robust = {}
        for eps in EVAL_EPS:
            print(f"  PGD ε={eps}...", end=' ', flush=True)
            t1  = time.time()
            X_adv = pgd_attack(model, X_ev_3d, y_ev_ohe, eps, steps=40)
            pred_adv = np.argmax(
                model.predict(X_adv, batch_size=512, verbose=0), axis=1)
            f1_adv  = float(f1_score(y_ev, pred_adv, average='macro', zero_division=0))
            acc_adv = float(accuracy_score(y_ev, pred_adv))
            correct = pred_clean[idx_ev] == y_ev
            asr     = float(np.mean(pred_adv[correct] != y_ev[correct])) \
                      if correct.any() else 0.
            robust[str(eps)] = {'f1': f1_adv, 'accuracy': acc_adv, 'asr': asr}
            print(f"F1={f1_adv:.4f}  ASR={asr:.4f}  [{time.time()-t1:.1f}s]")

        # Save this config's result
        result = {
            'theta':       theta,
            'adv_ratio':   adv_ratio,
            'rare_classes': sorted(rare_classes),
            'n_rare':      len(rare_classes),
            'clean': {
                'macro_f1':    f1_clean,
                'accuracy':    acc_clean,
                'rare_f1':     rare_f1,
                'common_f1':   common_f1,
                'per_class_f1': f1_per_class.tolist(),
            },
            'robust_pgd': robust,
            'train_time_min': round(train_time / 60, 1),
        }
        all_results.append(result)

        # Incremental save after each config
        with open(OUT_FILE, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"  Saved ({len(all_results)}/{total} done) → {OUT_FILE}")

        # Free model memory
        del model
        tf.keras.backend.clear_session()

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("CCAT SENSITIVITY RESULTS — CICIDS-2017")
    print(f"  Standard AT baseline clean F1 = 0.1799")
    print(f"  Reference CCAT (θ=1000, adv_ratio=0.5) clean F1 = 0.5314")
    print(f"{'='*65}")
    print(f"{'θ':>6}  {'adv_ratio':>9}  {'n_rare':>6}  {'F1_clean':>8}  "
          f"{'F1@ε0.01':>9}  {'F1@ε0.1':>8}  {'F1@ε0.3':>8}")
    print("-" * 65)
    for r in sorted(all_results, key=lambda x: (x['theta'], x['adv_ratio'])):
        rob = r['robust_pgd']
        print(f"{r['theta']:>6}  {r['adv_ratio']:>9.1f}  {r['n_rare']:>6}  "
              f"{r['clean']['macro_f1']:>8.4f}  "
              f"{rob.get('0.01', {}).get('f1', 0):>9.4f}  "
              f"{rob.get('0.1',  {}).get('f1', 0):>8.4f}  "
              f"{rob.get('0.3',  {}).get('f1', 0):>8.4f}")

    print(f"\n[DONE] All {total} configurations complete.")
    print(f"Results: {OUT_FILE}")


if __name__ == '__main__':
    main()
