"""
experiments/adversarial_robustness/attacks/pgd.py
Projected Gradient Descent (PGD) attack for the 1-D CNN IDS.

PGD is iterative FGSM with a random start inside the ε-ball, projected back
after each step (Madry et al., 2018).
"""
import numpy as np
import tensorflow as tf


def pgd_attack(
        model: tf.keras.Model,
        X: np.ndarray,
        y: np.ndarray,
        epsilon: float,
        alpha: float,
        n_iter: int,
        batch_size: int = 512,
        seed: int = 42,
) -> np.ndarray:
    """
    Generate PGD adversarial examples for the 1-D CNN.

    Parameters
    ----------
    model      : compiled Keras CNN (input shape (None, n_features, 1))
    X          : clean samples (N, n_features), float32 in [0, 1]
    y          : true integer labels (N,)
    epsilon    : L-inf perturbation budget
    alpha      : step size per PGD iteration
    n_iter     : number of PGD steps
    batch_size : samples to process per batch
    seed       : random seed for the initial random perturbation

    Returns
    -------
    X_adv : adversarial examples (N, n_features), clipped to [0, 1]
    """
    rng = np.random.default_rng(seed)
    X = X.astype(np.float32)
    n_features = X.shape[1]
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    # Random start within ε-ball
    delta_init = rng.uniform(-epsilon, epsilon, size=X.shape).astype(np.float32)
    X_adv = np.clip(X + delta_init, 0.0, 1.0)

    for _ in range(n_iter):
        X_adv_next = np.empty_like(X_adv)

        for start in range(0, len(X), batch_size):
            end   = min(start + batch_size, len(X))
            X_var = tf.Variable(
                X_adv[start:end].reshape(-1, n_features, 1), trainable=True,
                dtype=tf.float32)
            y_batch = tf.constant(y[start:end], dtype=tf.int64)

            with tf.GradientTape() as tape:
                preds = model(X_var, training=False)
                loss  = loss_fn(y_batch, preds)

            grad    = tape.gradient(loss, X_var)
            X_step  = (X_var + alpha * tf.sign(grad)).numpy().reshape(-1, n_features)
            X_orig  = X[start:end]

            # Project back onto ε-ball around original sample
            X_proj  = np.clip(X_step, X_orig - epsilon, X_orig + epsilon)
            X_adv_next[start:end] = np.clip(X_proj, 0.0, 1.0)

        X_adv = X_adv_next

    return X_adv
