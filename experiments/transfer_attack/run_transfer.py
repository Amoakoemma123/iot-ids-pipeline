"""
experiments/transfer_attack/run_transfer.py

Black-box transfer attack evaluation for Paper 2.

Strategy
--------
  1. For each dataset, reload X_test / y_test using the saved pipeline.
  2. Train a substitute MLP (architecturally distinct from the target CNN).
  3. Generate FGSM and PGD adversarial examples on the substitute.
  4. Feed those adversarial examples to the original CNN (target).
  5. Compute Transfer Success Rate (TSR) and Transfer F1-drop.

Transfer Success Rate (TSR)
    Fraction of correctly-classified clean samples that are misclassified
    by the target CNN when presented with adversarial examples crafted on
    the substitute model.

Results saved to:  experiments/transfer_attack/results/transfer_results.json
"""

import os, sys, json, time
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import f1_score

# ── GPU memory growth (prevents OOM on GTX 1650) ─────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
RESULTS_DIR = ROOT / 'experiments' / 'baseline' / 'results'
MODELS_DIR  = RESULTS_DIR / 'models'
OUT_DIR     = Path(__file__).resolve().parent / 'results'
OUT_DIR.mkdir(exist_ok=True)

from experiments.baseline.config import (
    DATASETS, SPLIT, PREPROCESSING, PREPROCESSING_OVERRIDES, SEED
)
from src.data.loader import load_dataset
from src.data.preprocessing import PreprocessingPipeline

# ── Config ────────────────────────────────────────────────────────────────────
PAPER2_DATASETS = ['TON_IoT', 'UNSW_NB15', 'CICIDS2017']
EPSILONS        = [0.01, 0.05, 0.1, 0.2, 0.3]
PGD_STEPS       = 40
N_EVAL          = 3000   # cap test samples for speed on CICIDS2017
SUB_EPOCHS      = 50
SUB_BATCH       = 256
SUB_PATIENCE    = 10

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# Substitute MLP (architecturally distinct from target 1-D CNN)
# ─────────────────────────────────────────────────────────────────────────────

def build_substitute_mlp(n_features, n_classes):
    """3-hidden-layer MLP: Dense(256) → Dense(128) → Dense(64) → softmax.
    No convolutions, no pooling — clearly different from the target CNN."""
    inp = tf.keras.layers.Input(shape=(n_features,), name='mlp_input')
    x   = tf.keras.layers.Dense(256, activation='relu', name='d1')(inp)
    x   = tf.keras.layers.BatchNormalization(name='bn1')(x)
    x   = tf.keras.layers.Dropout(0.3, name='drop1')(x)
    x   = tf.keras.layers.Dense(128, activation='relu', name='d2')(x)
    x   = tf.keras.layers.BatchNormalization(name='bn2')(x)
    x   = tf.keras.layers.Dropout(0.3, name='drop2')(x)
    x   = tf.keras.layers.Dense(64, activation='relu', name='d3')(x)
    x   = tf.keras.layers.Dropout(0.2, name='drop3')(x)
    out = tf.keras.layers.Dense(n_classes, activation='softmax', name='output')(x)
    model = tf.keras.Model(inputs=inp, outputs=out, name='SubstituteMLP')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ─────────────────────────────────────────────────────────────────────────────
# Attack helpers (applied to substitute model, 2-D input: N x F)
# ─────────────────────────────────────────────────────────────────────────────

def fgsm(model, x, y_onehot, eps):
    x_t = tf.cast(x, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x_t)
        loss = tf.keras.losses.categorical_crossentropy(
            y_onehot, model(x_t, training=False))
    grad  = tape.gradient(loss, x_t)
    x_adv = tf.clip_by_value(x_t + eps * tf.sign(grad), 0.0, 1.0)
    return x_adv.numpy()


def pgd(model, x, y_onehot, eps, steps=40):
    alpha = eps / 10.0
    x_t   = tf.cast(x, tf.float32)
    x_adv = tf.clip_by_value(
        x_t + tf.random.uniform(tf.shape(x_t), -eps, eps), 0.0, 1.0)
    for _ in range(steps):
        with tf.GradientTape() as tape:
            tape.watch(x_adv)
            loss = tf.keras.losses.categorical_crossentropy(
                y_onehot, model(x_adv, training=False))
        grad  = tape.gradient(loss, x_adv)
        x_adv = x_adv + alpha * tf.sign(grad)
        x_adv = tf.clip_by_value(x_adv, x_t - eps, x_t + eps)
        x_adv = tf.clip_by_value(x_adv, 0.0, 1.0)
    return x_adv.numpy()

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation on target CNN (3-D input: N x F x 1)
# ─────────────────────────────────────────────────────────────────────────────

def eval_transfer(target_cnn, X_clean_2d, y_true, X_adv_2d):
    """Return (TSR, f1_clean, f1_adv). Target CNN expects (N, F, 1)."""
    X_clean = X_clean_2d.reshape(X_clean_2d.shape[0], X_clean_2d.shape[1], 1)
    X_adv   = X_adv_2d.reshape(X_adv_2d.shape[0],   X_adv_2d.shape[1],   1)

    pred_clean = np.argmax(
        target_cnn.predict(X_clean, batch_size=512, verbose=0), axis=1)
    pred_adv   = np.argmax(
        target_cnn.predict(X_adv,   batch_size=512, verbose=0), axis=1)

    correct = (pred_clean == y_true)
    tsr     = float(np.mean(pred_adv[correct] != y_true[correct])) if correct.any() else 0.0
    f1_clean = float(f1_score(y_true, pred_clean, average='macro', zero_division=0))
    f1_adv   = float(f1_score(y_true, pred_adv,   average='macro', zero_division=0))
    return tsr, f1_clean, f1_adv

# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset runner
# ─────────────────────────────────────────────────────────────────────────────

def run_dataset(ds_name):
    print(f"\n{'='*60}\n  TRANSFER ATTACK — {ds_name}\n{'='*60}")

    csv_path, label_col = DATASETS[ds_name]

    # ── Load & preprocess ──────────────────────────────────────────────────
    print("  Loading data...")
    data = load_dataset(csv_path, label_col=label_col, **SPLIT)

    pp_kwargs = {**PREPROCESSING}
    pp_kwargs.update(PREPROCESSING_OVERRIDES.get(ds_name, {}))
    # Only pass params that PreprocessingPipeline.__init__ accepts
    valid_keys = {
        'mi_threshold', 'smote_k', 'smote_max_per_class',
        'skip_rfecv', 'rfecv_subsample', 'mi_subsample', 'seed'
    }
    pp_kwargs = {k: v for k, v in pp_kwargs.items() if k in valid_keys}

    pipeline = PreprocessingPipeline(**pp_kwargs)

    print("  Fitting pipeline...")
    X_train, y_train_sm = pipeline.fit_transform(
        data['X_train'], data['y_train'],
        feature_names=data['feature_names']
    )
    X_test, _ = pipeline.transform(data['X_test'], y=data['y_test'])
    y_test     = data['y_test']

    n_features = X_test.shape[1]
    n_classes  = len(data['class_names'])
    print(f"  Features={n_features}  Classes={n_classes}  "
          f"Train={len(X_train):,}  Test={len(X_test):,}")

    # ── Subsample eval set ─────────────────────────────────────────────────
    rng = np.random.RandomState(SEED)
    if len(X_test) > N_EVAL:
        idx    = rng.choice(len(X_test), N_EVAL, replace=False)
        X_eval = X_test[idx]
        y_eval = y_test[idx]
    else:
        X_eval = X_test
        y_eval = y_test
    print(f"  Evaluating on {len(X_eval):,} samples")

    # ── Load target CNN ────────────────────────────────────────────────────
    cnn_path   = MODELS_DIR / ds_name / 'model_cnn.h5'
    print(f"  Loading target CNN → {cnn_path.name}")
    target_cnn = tf.keras.models.load_model(str(cnn_path))
    target_cnn.trainable = False

    # ── Train substitute MLP ───────────────────────────────────────────────
    print("  Training substitute MLP...")
    sub = build_substitute_mlp(n_features, n_classes)

    y_ohe_train = tf.keras.utils.to_categorical(y_train_sm, n_classes)
    if len(X_train) > 200_000:
        idx_s       = rng.choice(len(X_train), 200_000, replace=False)
        X_tr, y_tr  = X_train[idx_s], y_ohe_train[idx_s]
    else:
        X_tr, y_tr  = X_train, y_ohe_train

    sub.fit(
        X_tr, y_tr,
        epochs=SUB_EPOCHS,
        batch_size=SUB_BATCH,
        validation_split=0.1,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=SUB_PATIENCE,
            restore_best_weights=True, verbose=0
        )],
        verbose=0
    )
    sub_pred = np.argmax(sub.predict(X_eval, batch_size=512, verbose=0), axis=1)
    sub_acc  = float(np.mean(sub_pred == y_eval))
    sub_f1   = float(f1_score(y_eval, sub_pred, average='macro', zero_division=0))
    print(f"  Substitute MLP — Acc={sub_acc:.4f}  F1={sub_f1:.4f}")

    # ── Run transfer attacks ───────────────────────────────────────────────
    y_ohe_eval = tf.keras.utils.to_categorical(y_eval, n_classes)
    results    = {
        'substitute_accuracy': sub_acc,
        'substitute_f1':       sub_f1,
        'n_eval':              int(len(X_eval)),
        'epsilons':            {}
    }

    for eps in EPSILONS:
        print(f"\n  ε = {eps}")
        eps_res = {}

        # FGSM
        t0 = time.time()
        print(f"    FGSM  ...", end=' ', flush=True)
        X_adv_f = fgsm(sub, X_eval, y_ohe_eval, eps)
        tsr_f, f1c, f1a_f = eval_transfer(target_cnn, X_eval, y_eval, X_adv_f)
        print(f"TSR={tsr_f*100:5.1f}%  F1: {f1c:.4f} → {f1a_f:.4f}  "
              f"(drop={f1c-f1a_f:.4f})  [{time.time()-t0:.1f}s]")
        eps_res['fgsm'] = {'tsr': tsr_f, 'f1_clean': f1c,
                           'f1_adv': f1a_f, 'f1_drop': round(f1c - f1a_f, 4)}

        # PGD
        t0 = time.time()
        print(f"    PGD-{PGD_STEPS} ...", end=' ', flush=True)
        X_adv_p = pgd(sub, X_eval, y_ohe_eval, eps, steps=PGD_STEPS)
        tsr_p, _, f1a_p = eval_transfer(target_cnn, X_eval, y_eval, X_adv_p)
        print(f"TSR={tsr_p*100:5.1f}%  F1: {f1c:.4f} → {f1a_p:.4f}  "
              f"(drop={f1c-f1a_p:.4f})  [{time.time()-t0:.1f}s]")
        eps_res['pgd'] = {'tsr': tsr_p, 'f1_clean': f1c,
                          'f1_adv': f1a_p, 'f1_drop': round(f1c - f1a_p, 4)}

        results['epsilons'][str(eps)] = eps_res

    return results

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    all_results = {}
    for ds in PAPER2_DATASETS:
        try:
            all_results[ds] = run_dataset(ds)
        except Exception as e:
            import traceback
            print(f"\n  ERROR on {ds}: {e}")
            traceback.print_exc()
            all_results[ds] = {'error': str(e)}

    out_path = OUT_DIR / 'transfer_results.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nResults saved → {out_path}")

    # ── Summary table ──────────────────────────────────────────────────────
    print("\n" + "="*72)
    print("TRANSFER ATTACK SUMMARY")
    print("="*72)
    print(f"{'Dataset':<14} {'ε':<6} {'FGSM TSR':>10} {'PGD TSR':>9} "
          f"{'FGSM F1↓':>10} {'PGD F1↓':>9}")
    print("-"*72)
    for ds, res in all_results.items():
        if 'error' in res:
            print(f"{ds:<14} ERROR: {res['error']}")
            continue
        for eps_str, er in res['epsilons'].items():
            print(f"{ds:<14} {eps_str:<6} "
                  f"{er['fgsm']['tsr']*100:>9.1f}% "
                  f"{er['pgd']['tsr']*100:>8.1f}% "
                  f"{er['fgsm']['f1_drop']:>10.4f} "
                  f"{er['pgd']['f1_drop']:>9.4f}")


if __name__ == '__main__':
    main()
