"""
experiments/adversarial_robustness/defenses/feature_squeezing.py
Feature squeezing defense for the lightweight 1-D CNN IDS.

Feature squeezing (Xu et al., 2018) reduces the precision of the input space
to remove small adversarial perturbations.  Two squeezers are implemented:

  1. Bit-depth reduction  — quantise each feature to 2^n_bits levels
  2. Median smoothing     — replace each feature with the median of its
                            neighbouring features (1-D median filter)

Both are applied at inference time with no model retraining required.
"""
import numpy as np
from scipy.ndimage import median_filter


# ── Squeezers ─────────────────────────────────────────────────────────────────

def bit_depth_squeeze(X: np.ndarray, n_bits: int = 4) -> np.ndarray:
    """
    Reduce feature precision to n_bits (2^n_bits uniform levels in [0,1]).

    Parameters
    ----------
    X      : (N, n_features) float array, values in [0, 1]
    n_bits : bit depth (4 → 16 levels, 8 → 256 levels)

    Returns
    -------
    X_sq : quantised array, same shape as X
    """
    n_levels = (2 ** n_bits) - 1
    return (np.round(X * n_levels) / n_levels).astype(np.float32)


def median_squeeze(X: np.ndarray, window: int = 3) -> np.ndarray:
    """
    Apply a 1-D median filter across the feature vector of each sample.

    Parameters
    ----------
    X      : (N, n_features) float array
    window : median filter window size (must be odd)

    Returns
    -------
    X_sq : smoothed array, same shape as X
    """
    return np.apply_along_axis(
        lambda row: median_filter(row, size=window, mode='nearest'),
        axis=1, arr=X,
    ).astype(np.float32)


# ── Evaluation helper ─────────────────────────────────────────────────────────

def evaluate_squeeze(model, X: np.ndarray, y_true: np.ndarray,
                     squeeze_fn, n_features: int,
                     batch_size: int = 512) -> dict:
    """
    Run the CNN on squeezed inputs and return classification metrics.

    Parameters
    ----------
    model      : Keras CNN (input shape (None, n_features, 1))
    X          : input samples (N, n_features), adversarial or clean
    y_true     : true integer labels (N,)
    squeeze_fn : callable X → X_squeezed (same shape)
    n_features : number of features
    batch_size : inference batch size

    Returns
    -------
    dict with accuracy, precision, recall, f1
    """
    import tensorflow as tf
    from sklearn.metrics import (accuracy_score, precision_score,
                                  recall_score, f1_score)

    X_sq   = squeeze_fn(X)
    preds  = []
    for s in range(0, len(X_sq), batch_size):
        batch = tf.constant(
            X_sq[s:s + batch_size].reshape(-1, n_features, 1).astype(np.float32))
        preds.append(np.argmax(model(batch, training=False).numpy(), axis=1))
    y_pred = np.concatenate(preds)

    return {
        'accuracy':  float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average='macro',
                                           zero_division=0)),
        'recall':    float(recall_score(y_true, y_pred, average='macro',
                                        zero_division=0)),
        'f1':        float(f1_score(y_true, y_pred, average='macro',
                                    zero_division=0)),
    }
