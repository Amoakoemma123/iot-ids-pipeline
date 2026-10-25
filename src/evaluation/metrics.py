"""
src/evaluation/metrics.py
Standardised evaluation metrics for all five baseline models.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from scipy.stats import chi2


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray,
                   y_proba: np.ndarray | None = None,
                   class_names: list | None = None) -> dict:
    """
    Compute all standard evaluation metrics.

    Returns
    -------
    dict with: accuracy, precision, recall, macro_f1, roc_auc (if y_proba given)
    """
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1   = f1_score(y_true, y_pred, average='macro', zero_division=0)

    result = dict(accuracy=acc, precision=prec, recall=rec, macro_f1=f1)

    if y_proba is not None:
        try:
            n_cls = y_proba.shape[1]
            if n_cls == 2:
                auc = roc_auc_score(y_true, y_proba[:, 1])
            else:
                auc = roc_auc_score(y_true, y_proba, multi_class='ovr',
                                    average='macro')
            result['roc_auc'] = auc
        except Exception:
            result['roc_auc'] = float('nan')

    return result


def classification_report_dict(y_true: np.ndarray, y_pred: np.ndarray,
                                class_names: list | None = None) -> dict:
    """Per-class precision, recall, F1 as a dictionary."""
    return classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )


def mcnemar_test(y_true: np.ndarray,
                 y_pred_a: np.ndarray,
                 y_pred_b: np.ndarray) -> dict:
    """
    McNemar's test to compare two classifiers (§4.1).

    Tests whether the difference in per-sample errors between model A
    and model B is statistically significant.

    Returns
    -------
    dict with: chi2, p_value, n01, n10
        n01 = samples A correct, B wrong
        n10 = samples A wrong,   B correct
    """
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    n01 = int(np.sum( correct_a & ~correct_b))  # A right, B wrong
    n10 = int(np.sum(~correct_a &  correct_b))  # A wrong, B right

    # Continuity-corrected McNemar statistic
    denom = n01 + n10
    if denom == 0:
        return dict(chi2=0.0, p_value=1.0, n01=0, n10=0)

    chi2_stat = (abs(n01 - n10) - 1) ** 2 / denom
    p_value   = 1 - chi2.cdf(chi2_stat, df=1)

    return dict(chi2=float(chi2_stat), p_value=float(p_value), n01=n01, n10=n10)


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                         normalize: str | None = 'true') -> np.ndarray:
    """Return confusion matrix, optionally normalised."""
    cm = confusion_matrix(y_true, y_pred)
    if normalize == 'true':
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    return cm
