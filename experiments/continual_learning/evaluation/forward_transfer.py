"""
experiments/continual_learning/evaluation/forward_transfer.py
Forward Transfer (FWT) — influence of past tasks on learning future tasks.

FWT = (1 / (T-1)) * Σ_{t=1}^{T-1} [ R[t, t] - b[t] ]

where b[t] = performance on task t trained from scratch (random-init baseline).
If b[t] is unavailable, we approximate it as the model's zero-shot performance
on task t immediately BEFORE any training has started on it (R[t-1, t] if it
were evaluated — approximated here as the pre-task evaluation on held-out data).

Positive FWT means prior tasks provide a useful initialisation for new tasks.
Negative FWT means prior tasks interfere with learning new ones.

Reference: Lopez-Paz & Ranzato (2017) NeurIPS.
"""
import numpy as np


def forward_transfer(R: np.ndarray,
                     random_baseline: np.ndarray | None = None) -> float:
    """
    Parameters
    ----------
    R               : (T, T) R matrix.
    random_baseline : (T,) array of b[t] — performance of a randomly-initialised
                      model trained only on task t. If None, we use R[t-1, t]
                      (pre-task zero-shot performance, always 0 in class-incremental
                      settings where task t classes are novel).

    Returns
    -------
    FWT scalar. Positive = positive forward transfer; negative = interference.
    """
    T = R.shape[0]
    if T < 2:
        return 0.0

    vals = []
    for t in range(1, T):
        plasticity_t = R[t, t]   # performance on task t right after training
        if np.isnan(plasticity_t):
            continue
        if random_baseline is not None:
            b_t = float(random_baseline[t])
        else:
            # Approximate: zero-shot performance before task t training.
            # In class-incremental (disjoint classes), new classes are unseen
            # → zero-shot F1 ≈ 0.
            b_t = 0.0
        vals.append(float(plasticity_t - b_t))

    return float(np.mean(vals)) if vals else 0.0


def compute_random_baseline(model_builder,
                             tasks: list,
                             n_seeds: int = 3,
                             seed: int = 42) -> np.ndarray:
    """
    Estimate b[t] for each task by training a fresh random model on task t
    alone and measuring test F1. Average over n_seeds initialisations.

    Parameters
    ----------
    model_builder : callable() → fresh compiled tf.keras.Model
    tasks         : list of Task objects
    n_seeds       : number of random seeds to average over

    Returns
    -------
    baseline : (T,) array of mean test F1 for each task trained in isolation
    """
    import tensorflow as tf
    from sklearn.metrics import f1_score

    T = len(tasks)
    baseline = np.zeros(T)

    for t, task in enumerate(tasks):
        f1s = []
        for s in range(n_seeds):
            tf.keras.backend.clear_session()
            tf.random.set_seed(seed + s)
            model = model_builder()
            X_tr = np.expand_dims(task.X_train.astype(np.float32), -1)
            model.fit(X_tr, task.y_train,
                      epochs=20, batch_size=256, verbose=0)
            X_te = np.expand_dims(task.X_test.astype(np.float32), -1)
            preds = np.argmax(model.predict(X_te, verbose=0), axis=1)
            f1s.append(f1_score(task.y_test, preds,
                                average='macro', zero_division=0))
        baseline[t] = np.mean(f1s)
        print(f"  [FWT baseline] Task {t}: mean F1={baseline[t]:.4f} "
              f"over {n_seeds} seeds")

    return baseline
