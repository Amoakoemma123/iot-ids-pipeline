"""
experiments/continual_learning/evaluation/forgetting_measure.py
Forgetting Measure (FM) — average drop in performance on previously learned tasks.

FM = (1 / (T-1)) * Σ_{j=0}^{T-2} [ max_{t=j..T-1} R[t,j] - R[T-1,j] ]

Positive FM means the model forgot old tasks after learning new ones.
FM = 0 means perfect retention. FM < 0 is theoretically possible (positive transfer).

Reference: Chaudhry et al. (2018) "Efficient Lifelong Learning with A-GEM." ICLR.
"""
import numpy as np


def forgetting_measure(R: np.ndarray) -> float:
    """
    Parameters
    ----------
    R : (T, T) matrix where R[t, j] = macro F1 on task j after training on task t.
        Upper triangle (j > t) should be NaN.

    Returns
    -------
    FM scalar (higher = more forgetting).
    """
    T = R.shape[0]
    if T < 2:
        return 0.0
    vals = []
    for j in range(T - 1):
        col = R[j:, j]
        col = col[~np.isnan(col)]
        if len(col) == 0:
            continue
        vals.append(float(np.max(col) - col[-1]))
    return float(np.mean(vals)) if vals else 0.0


def per_task_forgetting(R: np.ndarray) -> np.ndarray:
    """
    Return per-task forgetting: array of length T-1.
    forgetting[j] = max_{t>=j} R[t,j] - R[T-1,j]
    """
    T = R.shape[0]
    result = np.full(T - 1, np.nan)
    for j in range(T - 1):
        col = R[j:, j]
        col = col[~np.isnan(col)]
        if len(col) > 0:
            result[j] = float(np.max(col) - col[-1])
    return result
