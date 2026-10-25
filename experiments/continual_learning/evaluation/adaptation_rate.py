"""
experiments/continual_learning/evaluation/adaptation_rate.py
Adaptation Rate — how quickly the model learns each new task.

Defined as the number of training epochs required to reach a target
fraction of the peak validation F1 achieved on that task.
Default target = 90% of peak val F1 (threshold=0.9).

A lower adaptation rate means faster learning of new attack patterns,
which is important in dynamic IoT environments where threats emerge rapidly.
"""
import numpy as np


def adaptation_rate(history: list,
                    threshold: float = 0.9) -> int | None:
    """
    Parameters
    ----------
    history   : list of dicts with key 'val_f1', one entry per epoch.
                e.g. [{'epoch': 1, 'val_f1': 0.42}, {'epoch': 2, ...}, ...]
    threshold : fraction of peak val_f1 that counts as "adapted" (default 0.9)

    Returns
    -------
    epoch (1-indexed) at which val_f1 first reaches threshold * peak_val_f1,
    or None if the threshold was never reached.
    """
    if not history:
        return None

    f1_values = [h['val_f1'] for h in history]
    peak = max(f1_values)
    target = threshold * peak

    for h in history:
        if h['val_f1'] >= target:
            return h['epoch']

    return None   # threshold never reached (e.g. very short training)


def mean_adaptation_rate(per_task_histories: list,
                          threshold: float = 0.9) -> float:
    """
    Average adaptation rate across all tasks.

    Parameters
    ----------
    per_task_histories : list of history lists, one per task.
    threshold          : target fraction of peak F1.

    Returns
    -------
    Mean epochs-to-adapt across tasks (None tasks excluded from mean).
    """
    rates = []
    for t, hist in enumerate(per_task_histories):
        r = adaptation_rate(hist, threshold)
        if r is not None:
            rates.append(r)
            print(f"  [AdaptRate] Task {t}: reached {threshold*100:.0f}% "
                  f"peak F1 at epoch {r}")
        else:
            print(f"  [AdaptRate] Task {t}: threshold not reached in "
                  f"{len(hist)} epochs")
    return float(np.mean(rates)) if rates else float('nan')
