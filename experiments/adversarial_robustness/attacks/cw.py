"""
experiments/adversarial_robustness/attacks/cw.py
Carlini & Wagner L2 attack for the 1-D CNN IDS.

Implements the C&W L2 attack (Carlini & Wagner, 2017) with binary search
over the regularisation constant c. Works in tanh-space to enforce the
[0, 1] box constraint naturally.
"""
import numpy as np
import tensorflow as tf


def cw_l2_attack(
        model: tf.keras.Model,
        X: np.ndarray,
        y: np.ndarray,
        confidence: float = 0.0,
        learning_rate: float = 0.01,
        max_iter: int = 100,
        binary_search_steps: int = 9,
        initial_const: float = 1e-2,
        batch_size: int = 32,
        seed: int = 42,
) -> np.ndarray:
    """
    Generate C&W L2 adversarial examples for the 1-D CNN.

    Parameters
    ----------
    model               : compiled Keras CNN
    X                   : clean samples (N, n_features), float32 in [0, 1]
    y                   : true integer labels (N,)
    confidence          : kappa — margin above the best non-target logit (0 = just flip)
    learning_rate       : Adam LR for the inner optimisation
    max_iter            : inner optimisation steps per binary-search round
    binary_search_steps : rounds of binary search for constant c
    initial_const       : starting value of c
    batch_size          : samples per batch (keep small — C&W is slow)
    seed                : unused (reserved for API consistency)

    Returns
    -------
    X_adv : best adversarial examples found (N, n_features), clipped to [0, 1].
            Samples that could not be attacked are returned unchanged.
    """
    X = X.astype(np.float32)
    n_features = X.shape[1]
    X_adv = X.copy()

    # Map clean inputs into tanh-space: x = 0.5*(tanh(w)+1)  ⟹  w = atanh(2x-1)
    # Clip to avoid atanh(±1) = ±inf
    x_tanh_all = np.arctanh(np.clip(2.0 * X - 1.0, -0.9999, 0.9999))

    for start in range(0, len(X), batch_size):
        end     = min(start + batch_size, len(X))
        X_batch = X[start:end]           # (B, n_features)
        y_batch = y[start:end]           # (B,)
        B       = len(X_batch)

        x_tanh  = x_tanh_all[start:end].copy()

        best_adv = X_batch.copy()
        best_l2  = np.full(B, np.inf)

        c_lo  = np.zeros(B, dtype=np.float32)
        c_hi  = np.full(B, 1e10, dtype=np.float32)
        c_cur = np.full(B, initial_const, dtype=np.float32)

        for _bs in range(binary_search_steps):
            w         = tf.Variable(x_tanh.copy(), dtype=tf.float32)
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

            for _it in range(max_iter):
                with tf.GradientTape() as tape:
                    x_adv_2d = 0.5 * (tf.tanh(w) + 1.0)           # (B, n_features)
                    x_adv_3d = tf.reshape(x_adv_2d, (-1, n_features, 1))
                    logits   = model(x_adv_3d, training=False)      # (B, n_classes)

                    # L2 loss
                    l2 = tf.reduce_sum(
                        tf.square(x_adv_2d - tf.constant(X_batch)), axis=1)  # (B,)

                    # Adversarial objective: f = max(Z[y] - max_{i≠y} Z[i], -κ)
                    n_classes   = tf.shape(logits)[1]
                    one_hot     = tf.one_hot(y_batch, n_classes)
                    logit_true  = tf.reduce_sum(logits * one_hot, axis=1)
                    logit_other = tf.reduce_max(
                        logits * (1.0 - one_hot) + one_hot * (-1e9), axis=1)
                    f_val = tf.maximum(logit_true - logit_other + confidence,
                                       tf.zeros_like(logit_true))

                    c_tf = tf.constant(c_cur, dtype=tf.float32)
                    loss = tf.reduce_sum(l2 + c_tf * f_val)

                grad = tape.gradient(loss, w)
                optimizer.apply_gradients([(grad, w)])

            # Evaluate candidates after inner loop
            x_np  = (0.5 * (np.tanh(w.numpy()) + 1.0))   # (B, n_features)
            x_3d  = x_np.reshape(-1, n_features, 1)
            preds = np.argmax(
                model(tf.constant(x_3d.astype(np.float32)),
                      training=False).numpy(), axis=1)

            for i in range(B):
                l2_i = float(np.sum((x_np[i] - X_batch[i]) ** 2))
                if preds[i] != y_batch[i] and l2_i < best_l2[i]:
                    best_l2[i]  = l2_i
                    best_adv[i] = np.clip(x_np[i], 0.0, 1.0)
                    c_hi[i]     = c_cur[i]       # success → tighten
                else:
                    c_lo[i]     = c_cur[i]       # failure → loosen

            for i in range(B):
                if c_hi[i] < 1e9:
                    c_cur[i] = (c_lo[i] + c_hi[i]) / 2.0
                else:
                    c_cur[i] = min(c_cur[i] * 10.0, 1e10)

        X_adv[start:end] = best_adv
        found = np.sum(best_l2 < np.inf)
        print(f"    C&W batch [{start}:{end}]: {found}/{B} samples attacked")

    return X_adv
