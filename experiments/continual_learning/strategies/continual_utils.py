"""
experiments/continual_learning/strategies/continual_utils.py
Shared utilities for continual learning strategies.
"""
import numpy as np
import tensorflow as tf


def flatten_weights(model: tf.keras.Model) -> np.ndarray:
    """Flatten all trainable weights into a single 1-D array."""
    return np.concatenate([w.numpy().flatten() for w in model.trainable_variables])


def unflatten_weights(model: tf.keras.Model, flat: np.ndarray) -> None:
    """Write a flat weight vector back into the model's trainable variables."""
    ptr = 0
    for w in model.trainable_variables:
        size = w.numpy().size
        w.assign(flat[ptr:ptr + size].reshape(w.shape))
        ptr += size


def clone_weights(model: tf.keras.Model) -> list:
    """Return a deep copy of all trainable weights as a list of numpy arrays."""
    return [w.numpy().copy() for w in model.trainable_variables]


def restore_weights(model: tf.keras.Model, weights: list) -> None:
    """Restore weights from a list of numpy arrays."""
    for w, saved in zip(model.trainable_variables, weights):
        w.assign(saved)


def random_subset(X: np.ndarray, y: np.ndarray,
                  n: int, seed: int = 42) -> tuple:
    """Return a stratified random subset of (X, y) of size n."""
    rng = np.random.default_rng(seed)
    n = min(n, len(X))
    idx = rng.choice(len(X), n, replace=False)
    return X[idx], y[idx]


def compute_task_gradients(model: tf.keras.Model,
                            X: np.ndarray,
                            y: np.ndarray,
                            batch_size: int = 64) -> np.ndarray:
    """
    Compute the mean gradient of CE loss w.r.t. all trainable weights
    over a dataset, returned as a flat numpy array.
    """
    X_exp = np.expand_dims(X.astype(np.float32), -1)
    y_int = y.astype(np.int32)

    n_params = sum(w.numpy().size for w in model.trainable_variables)
    grad_accum = np.zeros(n_params, dtype=np.float64)
    n_batches = 0

    dataset = tf.data.Dataset.from_tensor_slices((X_exp, y_int)).batch(batch_size)
    for X_b, y_b in dataset:
        with tf.GradientTape() as tape:
            logits = model(X_b, training=False)
            loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(y_b, logits)
            )
        grads = tape.gradient(loss, model.trainable_variables)
        flat = np.concatenate([g.numpy().flatten() for g in grads if g is not None])
        grad_accum += flat
        n_batches += 1

    return (grad_accum / max(n_batches, 1)).astype(np.float32)


def save_checkpoint(model: tf.keras.Model, path: str) -> None:
    """Save model weights to a .weights.h5 checkpoint file."""
    model.save_weights(path)


def load_checkpoint(model: tf.keras.Model, path: str) -> None:
    """Load model weights from a checkpoint file."""
    model.load_weights(path)
