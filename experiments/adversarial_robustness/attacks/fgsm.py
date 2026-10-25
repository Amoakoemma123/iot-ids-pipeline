"""
experiments/adversarial_robustness/attacks/fgsm.py
Fast Gradient Sign Method (FGSM) attack for the 1-D CNN IDS.

Feature space is assumed to be MinMax-scaled to [0, 1].
"""
import numpy as np
import tensorflow as tf


def fgsm_attack(
        model: tf.keras.Model,
        X: np.ndarray,
        y: np.ndarray,
        epsilon: float,
        batch_size: int = 512,
) -> np.ndarray:
    """
    Generate FGSM adversarial examples for the 1-D CNN.

    Parameters
    ----------
    model      : compiled Keras CNN (input shape (None, n_features, 1))
    X          : clean samples, shape (N, n_features), float32 in [0, 1]
    y          : true integer labels, shape (N,)
    epsilon    : perturbation magnitude (L-inf norm)
    batch_size : samples to process per batch

    Returns
    -------
    X_adv : adversarial examples, shape (N, n_features), clipped to [0, 1]
    """
    X = X.astype(np.float32)
    n_features = X.shape[1]
    X_adv = np.empty_like(X)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        X_batch = tf.constant(X[start:end].reshape(-1, n_features, 1))
        y_batch = tf.constant(y[start:end], dtype=tf.int64)

        with tf.GradientTape() as tape:
            tape.watch(X_batch)
            preds = model(X_batch, training=False)
            loss  = loss_fn(y_batch, preds)

        grad = tape.gradient(loss, X_batch)
        perturbation = epsilon * tf.sign(grad)
        X_adv_batch  = X_batch + perturbation
        X_adv[start:end] = np.clip(
            X_adv_batch.numpy().reshape(-1, n_features), 0.0, 1.0)

    return X_adv
