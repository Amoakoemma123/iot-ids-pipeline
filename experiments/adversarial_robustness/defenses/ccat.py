"""
experiments/adversarial_robustness/defenses/ccat.py

Class-Conditional Adversarial Training (CCAT) — novel contribution for Paper 2.

Root cause of standard adversarial training failure on CICIDS-2017:
  PGD is applied uniformly to ALL classes, including ultra-rare classes whose
  SMOTE-oversampled examples (Heartbleed=11→50K synthetic, SQL=21→50K synthetic,
  Infiltration=36→50K synthetic) have fragile synthetic decision boundaries.
  PGD perturbations push these synthetic rare-class examples into the dominant
  benign class, destroying the training signal for rare classes entirely.

CCAT Fix:
  - Rare classes (original count < rare_threshold): train on CLEAN examples only.
    Their SMOTE-synthetic examples are preserved unperturbed.
  - Common classes (original count >= rare_threshold): standard PGD adversarial
    training at the specified epsilon budget.
  - Class-weighted loss: amplifies the gradient signal from rare classes to
    counteract the statistical dominance of the common classes.

This is the first class-conditional adversarial training approach evaluated in
the IoT IDS context with severe class imbalance.
"""
import numpy as np
import tensorflow as tf
from pathlib import Path


def _pgd_batch(model, X_3d: tf.Tensor, y: tf.Tensor,
               epsilon: float, alpha: float, n_steps: int,
               loss_fn) -> tf.Tensor:
    """PGD inner loop on GPU (same as base adversarial_training.py)."""
    delta = tf.random.uniform(tf.shape(X_3d), -epsilon, epsilon, dtype=tf.float32)
    X_adv = tf.clip_by_value(X_3d + delta, 0.0, 1.0)
    for _ in range(n_steps):
        with tf.GradientTape() as tape:
            tape.watch(X_adv)
            loss = loss_fn(y, model(X_adv, training=False))
        grad  = tape.gradient(loss, X_adv)
        X_adv = X_adv + alpha * tf.sign(grad)
        X_adv = tf.clip_by_value(X_adv, X_3d - epsilon, X_3d + epsilon)
        X_adv = tf.clip_by_value(X_adv, 0.0, 1.0)
        X_adv = tf.stop_gradient(X_adv)
    return X_adv


def ccat_train_cnn(
        model: tf.keras.Model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epsilon: float,
        adv_ratio: float,
        epochs: int,
        patience: int,
        batch_size: int,
        rare_classes: set,
        pgd_steps: int = 7,
        pgd_alpha: float | None = None,
        train_subsample: int | None = None,
        class_weight: dict | None = None,
        out_path: Path | None = None,
) -> tf.keras.Model:
    """
    Class-Conditional Adversarial Training (CCAT).

    Parameters
    ----------
    rare_classes    : set of class integer labels treated as rare.
                      These classes receive ONLY clean training examples.
    class_weight    : {class_id: weight} — amplifies rare-class gradient signal.
    All other params: same as adversarial_train_cnn in adversarial_training.py.

    Returns
    -------
    model : trained Keras model with best weights restored
    """
    if pgd_alpha is None:
        pgd_alpha = epsilon / 4.0

    n_features  = X_train.shape[1]
    loss_fn     = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer   = model.optimizer

    # Identify rare / common sample masks for the full training set
    is_rare_all  = np.isin(y_train, list(rare_classes))
    is_common_all = ~is_rare_all

    n_rare   = is_rare_all.sum()
    n_common = is_common_all.sum()
    print(f"  CCAT: rare-class samples={n_rare:,}  common-class samples={n_common:,}")
    print(f"  Rare classes: {sorted(rare_classes)}")

    # Optional subsample (stratified within common + rare)
    if train_subsample is not None and len(X_train) > train_subsample:
        rng   = np.random.default_rng(42)
        n_r   = int(train_subsample * (n_rare / len(X_train)))
        n_c   = train_subsample - n_r
        idx_r = rng.choice(np.where(is_rare_all)[0],  size=min(n_r, n_rare),   replace=False)
        idx_c = rng.choice(np.where(is_common_all)[0], size=min(n_c, n_common), replace=False)
        idx   = np.concatenate([idx_r, idx_c])
        X_train = X_train[idx]
        y_train = y_train[idx]
        is_rare_all  = np.isin(y_train, list(rare_classes))
        is_common_all = ~is_rare_all
        print(f"  Subsampled to {len(X_train):,} samples "
              f"(rare={is_rare_all.sum():,}, common={is_common_all.sum():,})")

    # Pre-compute sample weights
    if class_weight is not None:
        sw_all = np.array([class_weight.get(int(c), 1.0) for c in y_train],
                          dtype=np.float32)
    else:
        sw_all = None

    # Validation tensors
    X_val_gpu = tf.constant(X_val.reshape(-1, n_features, 1).astype(np.float32))
    y_val_gpu = tf.constant(y_val, dtype=tf.int64)

    best_val_loss  = np.inf
    best_weights   = model.get_weights()
    patience_count = 0
    indices        = np.arange(len(X_train))

    for epoch in range(1, epochs + 1):
        np.random.shuffle(indices)
        epoch_losses = []

        for start in range(0, len(X_train), batch_size):
            idx = indices[start:start + batch_size]
            if len(idx) < 4:
                continue

            X_b = X_train[idx].astype(np.float32)
            y_b = y_train[idx]

            # ── Split batch into rare and common samples ──────────────────
            rare_mask   = np.isin(y_b, list(rare_classes))
            common_mask = ~rare_mask

            X_rare   = X_b[rare_mask]
            y_rare   = y_b[rare_mask]
            X_common = X_b[common_mask]
            y_common = y_b[common_mask]

            # ── Adversarial examples from common-class samples only ───────
            parts_X = []
            parts_y = []

            if len(X_rare) > 0:
                parts_X.append(X_rare.reshape(-1, n_features, 1))
                parts_y.append(y_rare)

            if len(X_common) > 0:
                n_adv_c = min(int(len(X_common) * adv_ratio),
                              max(1, len(X_common) - 1))
                n_cln_c = len(X_common) - n_adv_c

                if n_cln_c > 0:
                    parts_X.append(X_common[:n_cln_c].reshape(-1, n_features, 1))
                    parts_y.append(y_common[:n_cln_c])

                if n_adv_c > 0:
                    X_adv_orig = tf.constant(
                        X_common[n_cln_c:].reshape(-1, n_features, 1))
                    y_adv_gpu_b = tf.constant(y_common[n_cln_c:], dtype=tf.int64)
                    X_adv = _pgd_batch(model, X_adv_orig, y_adv_gpu_b,
                                       epsilon, pgd_alpha, pgd_steps, loss_fn)
                    parts_X.append(X_adv.numpy())
                    parts_y.append(y_common[n_cln_c:])

            if not parts_X:
                continue

            X_mixed = tf.constant(np.concatenate(parts_X, axis=0))
            y_mixed = tf.constant(np.concatenate(parts_y, axis=0), dtype=tf.int64)

            sw = None
            if sw_all is not None:
                sw_batch = sw_all[idx]
                # Expand to match mixed batch (rare first, then clean common, then adv common)
                # For adversarial common samples: keep same weight as original
                # Reconstruct weights in same order as parts
                sw_parts = []
                if len(X_rare) > 0:
                    sw_parts.append(sw_batch[rare_mask])
                if len(X_common) > 0:
                    sw_common_b = sw_batch[common_mask]
                    if n_cln_c > 0:
                        sw_parts.append(sw_common_b[:n_cln_c])
                    if n_adv_c > 0:
                        sw_parts.append(sw_common_b[n_cln_c:])
                if sw_parts:
                    sw = tf.constant(np.concatenate(sw_parts, axis=0), dtype=tf.float32)

            with tf.GradientTape() as tape:
                preds = model(X_mixed, training=True)
                loss  = loss_fn(y_mixed, preds, sample_weight=sw)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_losses.append(float(loss))

        # Validation loss
        val_loss_sum = 0.0
        for vs in range(0, len(y_val), batch_size):
            vX = X_val_gpu[vs:vs + batch_size]
            vy = y_val_gpu[vs:vs + batch_size]
            vp = model(vX, training=False)
            val_loss_sum += float(loss_fn(vy, vp)) * int(vX.shape[0])
        val_loss   = val_loss_sum / len(y_val)
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float('nan')
        print(f"  Epoch {epoch:3d}/{epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss  = val_loss
            best_weights   = model.get_weights()
            patience_count = 0
            if out_path:
                model.save(str(out_path))
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"  EarlyStopping at epoch {epoch} (patience={patience})")
                break

    model.set_weights(best_weights)
    return model
