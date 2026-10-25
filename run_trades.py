"""
run_trades.py
TRADES adversarial training on TON_IoT, UNSW_NB15, CICIDS2017.
Implements: L = CE(f(x), y) + beta * KL(softmax(f(x)) || softmax(f(x_adv)))
where x_adv = argmax KL(f(x) || f(x+delta)) subject to ||delta||_inf <= eps.
Fully non-interactive. Runs all 3 datasets sequentially.

Output: experiments/adversarial_robustness/results/trades_results.json
"""
import os, sys, json, time, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
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
from src.models.cnn import build_cnn
from experiments.baseline.config import (
    DATASETS, SPLIT, PREPROCESSING, PREPROCESSING_OVERRIDES, SEED
)

DATASETS_TO_RUN = ['TON_IoT', 'UNSW_NB15', 'CICIDS2017']
EPSILON         = 0.1
BETA            = 6.0     # TRADES regularization strength (Zhang et al., 2019)
PGD_STEPS_TRAIN = 7       # inner maximization steps during training
PGD_STEPS_EVAL  = 40      # evaluation PGD steps
EPOCHS          = 50
PATIENCE        = 10
BATCH_SIZE      = 256
SUBSAMPLE       = 100_000
EPSILONS_EVAL   = [0.01, 0.1, 0.3]
N_EVAL          = 3000

OUT_PATH  = ROOT / 'experiments' / 'adversarial_robustness' / 'results' / 'trades_results.json'
MODEL_DIR = ROOT / 'experiments' / 'adversarial_robustness' / 'results'

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ── TRADES inner maximization (maximize KL, not CE) ────────────────────────
def _trades_inner(model, X_3d_orig, epsilon, alpha, n_steps):
    """Find x_adv that maximizes KL(softmax(f(x)) || softmax(f(x+delta)))."""
    delta = tf.random.uniform(tf.shape(X_3d_orig), -epsilon, epsilon)
    X_adv = tf.clip_by_value(X_3d_orig + delta, 0.0, 1.0)

    # Get clean softmax (fixed — computed once)
    p_clean = tf.nn.softmax(model(X_3d_orig, training=False))

    for _ in range(n_steps):
        with tf.GradientTape() as tape:
            tape.watch(X_adv)
            log_p_adv = tf.nn.log_softmax(model(X_adv, training=False))
            # KL(p_clean || p_adv) = sum(p_clean * (log p_clean - log p_adv))
            kl = tf.reduce_mean(
                tf.reduce_sum(
                    p_clean * (tf.math.log(p_clean + 1e-8) - log_p_adv),
                    axis=1))
        grad  = tape.gradient(kl, X_adv)
        X_adv = X_adv + alpha * tf.sign(grad)
        X_adv = tf.clip_by_value(X_adv, X_3d_orig - epsilon, X_3d_orig + epsilon)
        X_adv = tf.clip_by_value(X_adv, 0.0, 1.0)
        X_adv = tf.stop_gradient(X_adv)

    return X_adv


def trades_train_cnn(model, X_train, y_train, X_val, y_val,
                     epsilon, beta, epochs, patience, batch_size,
                     pgd_steps=7, train_subsample=None,
                     class_weight=None, out_path=None):
    """Train CNN with TRADES loss."""
    alpha = epsilon / 4.0
    n_features = X_train.shape[1]
    optimizer  = tf.keras.optimizers.Adam(1e-3)
    lr_sched   = tf.keras.callbacks.ReduceLROnPlateau(
        factor=0.5, patience=7, min_lr=1e-6, verbose=0)

    if train_subsample and len(X_train) > train_subsample:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_train), size=train_subsample, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]
        print(f"  Subsampled to {len(X_train):,}")

    sw_all = None
    if class_weight:
        sw_all = np.array([class_weight.get(int(c), 1.0) for c in y_train],
                          dtype=np.float32)

    X_val_gpu = tf.constant(X_val.reshape(-1, n_features, 1).astype(np.float32))
    y_val_gpu = tf.constant(y_val, dtype=tf.int64)

    ce_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    best_val_loss = np.inf
    best_weights  = model.get_weights()
    patience_cnt  = 0
    indices       = np.arange(len(X_train))
    current_lr    = 1e-3

    for epoch in range(1, epochs + 1):
        np.random.shuffle(indices)
        ep_losses = []

        for start in range(0, len(X_train), batch_size):
            idx = indices[start:start + batch_size]
            if len(idx) < 4:
                continue
            X_b = tf.constant(
                X_train[idx].reshape(-1, n_features, 1).astype(np.float32))
            y_b = tf.constant(y_train[idx], dtype=tf.int64)
            sw  = (tf.constant(sw_all[idx]) if sw_all is not None else None)

            # TRADES inner: find adversarial examples via KL maximization
            X_adv = _trades_inner(model, X_b, epsilon, alpha, pgd_steps)

            # TRADES outer: CE(f(x), y) + beta * KL(f(x) || f(x_adv))
            with tf.GradientTape() as tape:
                logits_clean = model(X_b,    training=True)
                logits_adv   = model(X_adv,  training=True)

                p_clean  = tf.nn.softmax(logits_clean)
                log_p_adv = tf.nn.log_softmax(logits_adv)

                ce   = ce_loss_fn(y_b, p_clean, sample_weight=sw)
                kl   = tf.reduce_mean(
                    tf.reduce_sum(
                        p_clean * (tf.math.log(p_clean + 1e-8) - log_p_adv),
                        axis=1))
                loss = ce + beta * kl

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            ep_losses.append(float(loss))

        # Validation loss (clean)
        val_loss_sum = 0.0
        for vs in range(0, len(y_val), batch_size):
            vX = X_val_gpu[vs:vs+batch_size]
            vy = y_val_gpu[vs:vs+batch_size]
            vp = tf.nn.softmax(model(vX, training=False))
            val_loss_sum += float(ce_loss_fn(vy, vp)) * int(vX.shape[0])
        val_loss   = val_loss_sum / len(y_val)
        train_loss = float(np.mean(ep_losses)) if ep_losses else float('nan')
        print(f"  Epoch {epoch:3d}/{epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        # LR reduction (manual)
        if epoch % 8 == 0 and current_lr > 1e-6:
            current_lr = max(current_lr * 0.5, 1e-6)
            optimizer.learning_rate.assign(current_lr)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_weights  = model.get_weights()
            patience_cnt  = 0
            if out_path:
                model.save(str(out_path))
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"  EarlyStopping at epoch {epoch}")
                break

    model.set_weights(best_weights)
    return model


def pgd_eval(model, X, y, eps, steps=40, batch_size=512):
    """PGD-40 evaluation returning (f1, acc, asr)."""
    alpha  = eps / 10.0
    n_feat = X.shape[1]
    X_adv  = X.copy().astype(np.float32)

    for start in range(0, len(X), batch_size):
        X_b = tf.constant(X[start:start+batch_size].reshape(-1, n_feat, 1))
        y_b = tf.constant(y[start:start+batch_size], dtype=tf.int64)
        x_t = X_b + tf.random.uniform(tf.shape(X_b), -eps, eps)
        x_t = tf.clip_by_value(x_t, 0., 1.)
        for _ in range(steps):
            with tf.GradientTape() as tape:
                tape.watch(x_t)
                preds = model(x_t, training=False)
                loss  = tf.keras.losses.sparse_categorical_crossentropy(y_b, preds)
            grad = tape.gradient(loss, x_t)
            x_t  = x_t + alpha * tf.sign(grad)
            x_t  = tf.clip_by_value(x_t, X_b - eps, X_b + eps)
            x_t  = tf.clip_by_value(x_t, 0., 1.)
            x_t  = tf.stop_gradient(x_t)
        X_adv[start:start+batch_size] = x_t.numpy().reshape(-1, n_feat)

    y_clean = np.concatenate([
        np.argmax(model(
            tf.constant(X[s:s+batch_size].reshape(-1, n_feat, 1).astype(np.float32)),
            training=False).numpy(), axis=1)
        for s in range(0, len(X), batch_size)])
    y_adv = np.concatenate([
        np.argmax(model(
            tf.constant(X_adv[s:s+batch_size].reshape(-1, n_feat, 1).astype(np.float32)),
            training=False).numpy(), axis=1)
        for s in range(0, len(X_adv), batch_size)])

    correct = y_clean == y
    asr = float((correct & (y_adv != y)).sum() / max(correct.sum(), 1))
    return (float(f1_score(y, y_adv, average='macro', zero_division=0)),
            float(accuracy_score(y, y_adv)),
            asr)


def compute_class_weights(y, n_classes):
    counts  = np.bincount(y, minlength=n_classes).astype(float)
    counts  = np.where(counts == 0, 1, counts)
    weights = counts.sum() / (n_classes * counts)
    weights = np.clip(weights, 1.0, 100.0)
    return {i: float(w) for i, w in enumerate(weights)}


all_results = {}

for ds in DATASETS_TO_RUN:
    print(f"\n{'='*60}")
    print(f"  TRADES — {ds}  (ε={EPSILON}, β={BETA})")
    print(f"{'='*60}")
    t_ds = time.time()

    # ── Load + preprocess ─────────────────────────────────────────────────────
    csv_path, label_col = DATASETS[ds]
    data = load_dataset(csv_path, label_col=label_col, **SPLIT)

    pp_kw = {**PREPROCESSING}
    pp_kw.update(PREPROCESSING_OVERRIDES.get(ds, {}))
    valid = {'mi_threshold','smote_k','smote_max_per_class',
             'skip_rfecv','rfecv_subsample','mi_subsample','seed'}
    pp_kw = {k: v for k, v in pp_kw.items() if k in valid}

    pipeline = PreprocessingPipeline(**pp_kw)
    print("[1] Fitting pipeline...")
    X_train_sm, y_train_sm = pipeline.fit_transform(
        data['X_train'], data['y_train'], feature_names=data['feature_names'])
    X_val_sc, _ = pipeline.transform(data['X_val'],  y=data['y_val'])
    X_test,   _ = pipeline.transform(data['X_test'], y=data['y_test'])
    y_val  = data['y_val']
    y_test = data['y_test']

    n_feat    = X_test.shape[1]
    n_classes = len(data['class_names'])
    print(f"  Features={n_feat}  Classes={n_classes}  "
          f"Train(SMOTE)={len(X_train_sm):,}  Test={len(X_test):,}")

    class_weights = compute_class_weights(y_train_sm, n_classes)

    # ── Build + train ─────────────────────────────────────────────────────────
    print(f"\n[2] Training TRADES model (β={BETA})...")
    model = build_cnn(n_feat, n_classes)
    out_path = MODEL_DIR / ds / 'model_cnn_trades.h5'
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    trades_train_cnn(
        model, X_train_sm, y_train_sm, X_val_sc, y_val,
        epsilon=EPSILON, beta=BETA, epochs=EPOCHS, patience=PATIENCE,
        batch_size=BATCH_SIZE, pgd_steps=PGD_STEPS_TRAIN,
        train_subsample=SUBSAMPLE, class_weight=class_weights,
        out_path=out_path,
    )
    train_min = (time.time() - t0) / 60
    print(f"  Training done in {train_min:.1f} min")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    # Sample evaluation subset
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X_test), size=min(N_EVAL, len(X_test)), replace=False)
    X_ev = X_test[idx]
    y_ev = y_test[idx]

    print("\n[3] Evaluating...")
    # Clean
    clean_preds = np.concatenate([
        np.argmax(model(
            tf.constant(X_ev[s:s+512].reshape(-1, n_feat, 1).astype(np.float32)),
            training=False).numpy(), axis=1)
        for s in range(0, len(X_ev), 512)])
    f1_clean  = float(f1_score(y_ev, clean_preds, average='macro', zero_division=0))
    acc_clean = float(accuracy_score(y_ev, clean_preds))
    print(f"  Clean — F1={f1_clean:.4f}  Acc={acc_clean:.4f}")

    pgd_results = {}
    for eps in EPSILONS_EVAL:
        print(f"  PGD ε={eps}...")
        f1_adv, acc_adv, asr = pgd_eval(model, X_ev, y_ev, eps, PGD_STEPS_EVAL)
        pgd_results[str(eps)] = {'f1': f1_adv, 'acc': acc_adv, 'asr': asr}
        print(f"    F1={f1_adv:.4f}  Acc={acc_adv:.4f}  ASR={asr:.4f}")

    all_results[ds] = {
        'epsilon': EPSILON, 'beta': BETA, 'train_time_min': round(train_min, 1),
        'clean': {'f1': f1_clean, 'acc': acc_clean},
        'pgd':   pgd_results,
    }
    print(f"\n  {ds} total: {(time.time()-t_ds)/60:.1f} min")

    # Save incrementally
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved → {OUT_PATH}")

print(f"\n{'='*60}")
print("  TRADES COMPLETE")
print(f"{'='*60}")
print(f"\nResults saved to: {OUT_PATH}")

print("\n=== SUMMARY ===")
for ds, r in all_results.items():
    print(f"\n{ds} (β={r['beta']}, ε_train={r['epsilon']}, "
          f"train={r['train_time_min']}min):")
    print(f"  Clean F1={r['clean']['f1']:.4f}")
    for eps, pr in r['pgd'].items():
        print(f"  PGD ε={eps}: F1={pr['f1']:.4f}  ASR={pr['asr']:.3f}")
