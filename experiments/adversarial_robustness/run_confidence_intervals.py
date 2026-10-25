"""
experiments/adversarial_robustness/run_confidence_intervals.py

Compute mean ± std for key attack metrics by running PGD-40 with 5 different
random seeds on the ALREADY-TRAINED baseline CNN models (no retraining).

This captures variance from PGD's random initialisation — the dominant source
of stochasticity in adversarial evaluation when the model weights are fixed.

Outputs: experiments/adversarial_robustness/results/confidence_intervals.json
"""
import sys, json, time
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score

for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.baseline.config import DATASETS, SPLIT, PREPROCESSING, PREPROCESSING_OVERRIDES, SEED
from src.data.loader import load_dataset
from src.data.preprocessing import PreprocessingPipeline

DATASETS_LIST = ['TON_IoT', 'UNSW_NB15', 'CICIDS2017']
MODELS_DIR    = ROOT / 'experiments' / 'baseline' / 'results' / 'models'
OUT_PATH      = Path(__file__).resolve().parent / 'results' / 'confidence_intervals.json'
N_SEEDS       = 5
SEEDS         = [42, 43, 44, 45, 46]
N_EVAL        = 3000
TARGET_EPS    = [0.05, 0.1, 0.2]   # representative ε values for CI table
PGD_STEPS     = 40


def pgd_attack(model, X, y_ohe, eps, steps, seed, batch_size=512):
    tf.random.set_seed(seed)
    alpha = eps / 10.0
    X_adv = X.copy().astype(np.float32)
    for start in range(0, len(X), batch_size):
        X_b = tf.constant(X[start:start+batch_size])
        y_b = tf.constant(y_ohe[start:start+batch_size])
        x_t = X_b + tf.random.uniform(tf.shape(X_b), -eps, eps)
        x_t = tf.clip_by_value(x_t, 0., 1.)
        for _ in range(steps):
            with tf.GradientTape() as tape:
                tape.watch(x_t)
                loss = tf.keras.losses.categorical_crossentropy(y_b, model(x_t, training=False))
            grad = tape.gradient(loss, x_t)
            x_t  = x_t + alpha * tf.sign(grad)
            x_t  = tf.clip_by_value(x_t, X_b - eps, X_b + eps)
            x_t  = tf.clip_by_value(x_t, 0., 1.)
        X_adv[start:start+batch_size] = x_t.numpy()
    return X_adv


def run_dataset(ds_name):
    print(f"\n{'='*55}\n  CI Evaluation — {ds_name}\n{'='*55}")

    csv_path, label_col = DATASETS[ds_name]
    data = load_dataset(csv_path, label_col=label_col, **SPLIT)

    pp_kwargs = {**PREPROCESSING}
    pp_kwargs.update(PREPROCESSING_OVERRIDES.get(ds_name, {}))
    valid_keys = {'mi_threshold','smote_k','smote_max_per_class',
                  'skip_rfecv','rfecv_subsample','mi_subsample','seed'}
    pp_kwargs = {k: v for k, v in pp_kwargs.items() if k in valid_keys}

    pipeline = PreprocessingPipeline(**pp_kwargs)
    print("  Fitting pipeline...")
    pipeline.fit_transform(data['X_train'], data['y_train'],
                           feature_names=data['feature_names'])
    X_test, _ = pipeline.transform(data['X_test'], y=data['y_test'])
    y_test     = data['y_test']
    n_features = X_test.shape[1]
    n_classes  = len(data['class_names'])

    # Fixed eval subset (same seed for all runs)
    rng   = np.random.RandomState(SEED)
    idx_e = rng.choice(len(X_test), min(N_EVAL, len(X_test)), replace=False)
    X_ev  = X_test[idx_e].reshape(-1, n_features, 1).astype(np.float32)
    y_ev  = y_test[idx_e]
    y_ohe = tf.keras.utils.to_categorical(y_ev, n_classes).astype(np.float32)

    # Load target CNN
    cnn_path = MODELS_DIR / ds_name / 'model_cnn.h5'
    model    = tf.keras.models.load_model(str(cnn_path))
    model.trainable = False

    # Clean accuracy (deterministic — one run)
    pred_clean = np.argmax(model.predict(X_ev, batch_size=512, verbose=0), axis=1)
    acc_clean  = float(accuracy_score(y_ev, pred_clean))
    f1_clean   = float(f1_score(y_ev, pred_clean, average='macro', zero_division=0))
    print(f"  Clean: Acc={acc_clean:.4f}  F1={f1_clean:.4f}")

    results = {
        'clean': {'accuracy': acc_clean, 'f1': f1_clean},
        'pgd': {}
    }

    for eps in TARGET_EPS:
        accs, f1s, asrs = [], [], []
        print(f"\n  PGD ε={eps}  ({N_SEEDS} seeds):")
        for seed in SEEDS:
            t0 = time.time()
            X_adv    = pgd_attack(model, X_ev, y_ohe, eps, PGD_STEPS, seed)
            pred_adv = np.argmax(model.predict(X_adv, batch_size=512, verbose=0), axis=1)
            acc_adv  = float(accuracy_score(y_ev, pred_adv))
            f1_adv   = float(f1_score(y_ev, pred_adv, average='macro', zero_division=0))
            correct  = pred_clean == y_ev
            asr      = float(np.mean(pred_adv[correct] != y_ev[correct])) if correct.any() else 0.
            accs.append(acc_adv); f1s.append(f1_adv); asrs.append(asr)
            print(f"    seed={seed}  Acc={acc_adv:.4f}  F1={f1_adv:.4f}  ASR={asr:.4f}  [{time.time()-t0:.1f}s]")

        results['pgd'][str(eps)] = {
            'accuracy_mean': float(np.mean(accs)),
            'accuracy_std':  float(np.std(accs)),
            'f1_mean':       float(np.mean(f1s)),
            'f1_std':        float(np.std(f1s)),
            'asr_mean':      float(np.mean(asrs)),
            'asr_std':       float(np.std(asrs)),
            'seeds':         SEEDS,
            'raw_accuracy':  accs,
            'raw_f1':        f1s,
            'raw_asr':       asrs,
        }
        print(f"    → Acc={np.mean(accs):.4f}±{np.std(accs):.4f}  "
              f"F1={np.mean(f1s):.4f}±{np.std(f1s):.4f}  "
              f"ASR={np.mean(asrs):.4f}±{np.std(asrs):.4f}")

    return results


def main():
    all_results = {}
    for ds in DATASETS_LIST:
        try:
            all_results[ds] = run_dataset(ds)
        except Exception as e:
            import traceback
            print(f"\n  ERROR on {ds}: {e}")
            traceback.print_exc()
            all_results[ds] = {'error': str(e)}

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nConfidence intervals saved → {OUT_PATH}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("CONFIDENCE INTERVALS SUMMARY (PGD-40, 5 seeds)")
    print("=" * 65)
    print(f"{'Dataset':<14} {'ε':<6} {'Accuracy':>16} {'Macro F1':>16} {'ASR':>14}")
    print("-" * 65)
    for ds, res in all_results.items():
        if 'error' in res:
            print(f"{ds:<14} ERROR: {res['error']}")
            continue
        for eps_str, v in res['pgd'].items():
            print(f"{ds:<14} {eps_str:<6} "
                  f"{v['accuracy_mean']:.4f}±{v['accuracy_std']:.4f}   "
                  f"{v['f1_mean']:.4f}±{v['f1_std']:.4f}   "
                  f"{v['asr_mean']:.4f}±{v['asr_std']:.4f}")


if __name__ == '__main__':
    main()
