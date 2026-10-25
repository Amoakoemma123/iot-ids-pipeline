"""
experiments/continual_learning/evaluation/backward_transfer.py
Backward Transfer (BWT) — influence of learning new tasks on past tasks.

BWT = (1 / (T-1)) * Σ_{j=0}^{T-2} [ R[T-1, j] - R[j, j] ]

Negative BWT → future tasks hurt past task performance (forgetting).
Positive BWT → future tasks improve past task performance (beneficial transfer).
BWT is more informative than FM because it is signed.

Reference: Lopez-Paz & Ranzato (2017) "Gradient Episodic Memory." NeurIPS.
"""
import numpy as np


def backward_transfer(R: np.ndarray) -> float:
    """
    Parameters
    ----------
    R : (T, T) R matrix (upper triangle NaN).

    Returns
    -------
    BWT scalar. Negative = forgetting; positive = improvement on past tasks.
    """
    T = R.shape[0]
    if T < 2:
        return 0.0
    vals = []
    for j in range(T - 1):
        if not np.isnan(R[T - 1, j]) and not np.isnan(R[j, j]):
            vals.append(float(R[T - 1, j] - R[j, j]))
    return float(np.mean(vals)) if vals else 0.0


def per_task_bwt(R: np.ndarray) -> np.ndarray:
    """
    Per-task BWT: array of length T-1.
    bwt[j] = R[T-1, j] - R[j, j]
    """
    T = R.shape[0]
    result = np.full(T - 1, np.nan)
    for j in range(T - 1):
        if not np.isnan(R[T - 1, j]) and not np.isnan(R[j, j]):
            result[j] = float(R[T - 1, j] - R[j, j])
    return result
