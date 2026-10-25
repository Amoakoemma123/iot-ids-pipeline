"""
experiments/adversarial_robustness/defenses/adversarial_training.py
PGD adversarial training defense for the lightweight 1-D CNN IDS.

At each training step a fraction (adv_ratio) of the batch is replaced with
multi-step PGD adversarial examples generated on-the-fly on the GPU
(Madry et al., 2018 — "Towards Deep Learning Models Resistant to Adversarial
Attacks").  EarlyStopping monitors clean validation loss.
"""
import numpy as np
import tensorflow as tf
from pathlib import Path


def _pgd_batch_gpu(model, X_3d_orig: tf.Tensor, y_batch: tf.Tensor,
                   epsilon: float, alpha: float, n_steps: int,
                   loss_fn) -> tf.Tensor:
    """
    Multi-step PGD inner maximisation on GPU.

    Parameters
    ----------
    model     : Keras CNN (called with training=False)
    X_3d_orig : clean batch tensor (B, n_features, 1), constant
    y_batch   : true labels (B,), int64
    epsilon   : L-inf perturbation budget
    alpha     : PGD step size
    n_steps   : number of PGD iterations
    loss_fn   : SparseCategoricalCrossentropy instance

    Returns
    -------
    Adversarial examples tensor (B, n_features, 1), clipped to [0, 1]
    """
    # Random start within ε-ball
    delta  = tf.random.uniform(tf.shape(X_3d_orig), -epsilon, epsilon,
                                dtype=tf.float32)
    X_adv  = tf.clip_by_value(X_3d_orig + delta, 0.0, 1.0)

    for _ in range(n_steps):
        with tf.GradientTape() as tape:
            tape.watch(X_adv)
            preds = model(X_adv, training=False)
            loss  = loss_fn(y_batch, preds)
        grad  = tape.gradient(loss, X_adv)
        X_adv = X_adv + alpha * tf.sign(grad)
        # Project back onto ε-ball around original sample
        X_adv = tf.clip_by_value(X_adv, X_3d_orig - epsilon,
                                  X_3d_orig + epsilon)
        X_adv = tf.clip_by_value(X_adv, 0.0, 1.0)
        X_adv = tf.stop_gradient(X_adv)   # detach for next iteration

    return X_adv


def adversarial_train_cnn(
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
        pgd_steps: int = 7,
        pgd_alpha: float | None = None,
        train_subsample: int | None = None,
        class_weight: dict | None = None,
        out_path: Path | None = None,
) -> tf.keras.Model:
    """
    Train the CNN with mixed clean + PGD-adversarial batches (GPU-accelerated).

    Parameters
    ----------
    model          : compiled Keras CNN (will be trained in-place)
    X_train        : SMOTE-balanced, scaled training data (N, n_features) in [0,1]
    y_train        : integer labels (N,)
    X_val          : scaled validation data
    y_val          : validation labels
    epsilon        : PGD ε for on-the-fly adversarial example generation
    adv_ratio      : fraction of each batch replaced by adversarial examples
    epochs         : maximum training epochs
    patience       : EarlyStopping patience on val_loss
    batch_size     : total batch size (clean + adversarial combined)
    pgd_steps      : number of PGD inner steps (default 7, Madry et al.)
    pgd_alpha      : PGD step size; defaults to epsilon / 4 if None
    train_subsample: cap training samples per epoch (stratified); None = use all
    class_weight   : optional {class_id: weight} dict for imbalanced classes
    out_path       : if given, best model weights are saved here (HDF5)

    Returns
    -------
    model : trained Keras model with best weights restored
    """
    if pgd_alpha is None:
        pgd_alpha = epsilon / 4.0

    n_features = X_train.shape[1]
    loss_fn    = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer  = model.optimizer
    n_adv      = int(batch_size * adv_ratio)

    # Optional stratified subsample (used for CICIDS2017 to keep epochs fast)
    if train_subsample is not None and len(X_train) > train_subsample:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_train), size=train_subsample, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]
        print(f"  Training subsampled to {len(X_train):,} samples")

    # Pre-compute sample weights from class_weight dict
    if class_weight is not None:
        sample_weights_all = np.array(
            [class_weight.get(int(c), 1.0) for c in y_train], dtype=np.float32)
    else:
        sample_weights_all = None

    # Validation tensors on GPU
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

            actual_adv  = min(n_adv, max(1, len(X_b) // 2))
            actual_cln  = len(X_b) - actual_adv

            X_clean_gpu  = tf.constant(X_b[:actual_cln].reshape(-1, n_features, 1))
            X_adv_orig   = tf.constant(X_b[actual_cln:].reshape(-1, n_features, 1))
            y_adv_gpu    = tf.constant(y_b[actual_cln:], dtype=tf.int64)

            X_adv_gpu = _pgd_batch_gpu(model, X_adv_orig, y_adv_gpu,
                                        epsilon, pgd_alpha, pgd_steps, loss_fn)

            X_mixed = tf.concat([X_clean_gpu, X_adv_gpu], axis=0)
            y_mixed = tf.constant(y_b, dtype=tf.int64)

            sw = (tf.constant(sample_weights_all[idx], dtype=tf.float32)
                  if sample_weights_all is not None else None)

            with tf.GradientTape() as tape:
                preds = model(X_mixed, training=True)
                loss  = loss_fn(y_mixed, preds, sample_weight=sw)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_losses.append(float(loss))

        # Validation — batched to avoid OOM on large datasets
        val_loss_sum = 0.0
        val_n = len(y_val)
        for vs in range(0, val_n, batch_size):
            vX = X_val_gpu[vs:vs + batch_size]
            vy = y_val_gpu[vs:vs + batch_size]
            vp = model(vX, training=False)
            val_loss_sum += float(loss_fn(vy, vp)) * int(vX.shape[0])
        val_loss   = val_loss_sum / val_n
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float('nan')
        print(f"  Epoch {epoch:3d}/{epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss  = val_loss
            best_weights   = model.get_weights()
            patience_count = 0
            if out_path is not None:
                model.save(str(out_path))
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"  EarlyStopping triggered at epoch {epoch} "
                      f"(patience={patience})")
                break

    model.set_weights(best_weights)
    return model
