"""
src/evaluation/cl_metrics.py
Continual learning evaluation metrics.

Metrics
-------
- Macro F1 per task, per training session   (R matrix: R[t][j] = F1 after task t on task j)
- Forgetting Measure (FM)                   : average drop in performance on old tasks
- Backward Transfer (BWT)                   : signed average transfer to past tasks
- Plasticity (FWT)                          : performance on new task right after training
- Adaptation Rate                           : epochs needed to reach 90% of peak F1

References
----------
Lopez-Paz & Ranzato (2017) "Gradient Episodic Memory for Continual Learning."
Chaudhry et al. (2018) "Efficient Lifelong Learning with A-GEM."
"""
import numpy as np


def compute_r_matrix(results: list) -> np.ndarray:
    """
    Build the R matrix from experiment results.

    Parameters
    ----------
    results : list of dicts, one per training session (task 0, 1, ..., T-1).
              Each dict has key 'task_f1' : list of F1 scores evaluated on all
              tasks seen so far after training on the current task.
              Shape: results[t]['task_f1'][j] = F1 on task j after training on task t.

    Returns
    -------
    R : (T, T) array where R[t, j] = F1 on task j after training on task t.
        Upper triangle (j > t) is NaN (task j not yet introduced).
    """
    T = len(results)
    R = np.full((T, T), np.nan)
    for t, res in enumerate(results):
        for j, f1 in enumerate(res.get('task_f1', [])):
            if j <= t:
                R[t, j] = f1
    return R


def forgetting_measure(R: np.ndarray) -> float:
    """
    FM = (1/(T-1)) * Σ_{j=0}^{T-2} [ max_{t=j..T-1} R[t,j] - R[T-1,j] ]

    Average maximum-minus-final performance drop across all tasks except the last.
    Higher FM → more catastrophic forgetting.
    """
    T = R.shape[0]
    if T < 2:
        return 0.0
    fm_vals = []
    for j in range(T - 1):
        col = R[j:, j]   # F1 on task j from session j onwards
        col = col[~np.isnan(col)]
        if len(col) == 0:
            continue
        fm_vals.append(float(np.max(col) - col[-1]))
    return float(np.mean(fm_vals)) if fm_vals else 0.0


def backward_transfer(R: np.ndarray) -> float:
    """
    BWT = (1/(T-1)) * Σ_{j=0}^{T-2} [ R[T-1,j] - R[j,j] ]

    Signed measure: negative BWT → forgetting; positive BWT → improvement
    (e.g. future tasks provide complementary features helping past tasks).
    """
    T = R.shape[0]
    if T < 2:
        return 0.0
    vals = []
    for j in range(T - 1):
        if not np.isnan(R[T-1, j]) and not np.isnan(R[j, j]):
            vals.append(float(R[T-1, j] - R[j, j]))
    return float(np.mean(vals)) if vals else 0.0


def plasticity(R: np.ndarray) -> float:
    """
    Plasticity = (1/T) * Σ_{t=0}^{T-1} R[t, t]

    Average performance on each new task immediately after training on it.
    Measures how well the model adapts to new information.
    """
    T = R.shape[0]
    vals = [R[t, t] for t in range(T) if not np.isnan(R[t, t])]
    return float(np.mean(vals)) if vals else 0.0


def average_accuracy(R: np.ndarray) -> float:
    """
    A_T = (1/T) * Σ_{j=0}^{T-1} R[T-1, j]

    Average accuracy on all tasks after the final training session.
    Primary summary metric for comparing strategies.
    """
    T = R.shape[0]
    final_row = R[T-1, :]
    vals = final_row[~np.isnan(final_row)]
    return float(np.mean(vals)) if len(vals) > 0 else 0.0


def summarise(R: np.ndarray, strategy_name: str = '') -> dict:
    """
    Compute all CL metrics from the R matrix and return as a dict.
    """
    metrics = {
        'strategy':         strategy_name,
        'average_accuracy': average_accuracy(R),
        'plasticity':       plasticity(R),
        'forgetting':       forgetting_measure(R),
        'bwt':              backward_transfer(R),
        'R_matrix':         R.tolist(),
    }
    print(f"\n{'='*55}")
    print(f"  Strategy: {strategy_name}")
    print(f"  Average Accuracy (A_T):  {metrics['average_accuracy']:.4f}")
    print(f"  Plasticity:              {metrics['plasticity']:.4f}")
    print(f"  Forgetting Measure (FM): {metrics['forgetting']:.4f}")
    print(f"  Backward Transfer (BWT): {metrics['bwt']:.4f}")
    print(f"{'='*55}")
    return metrics
