"""Roll a fitted scikit-learn HistGradientBoosting model back to its best internal-validation iteration.

Why: with early_stopping=True, HistGradientBoosting{Classifier,Regressor}.fit stops when the last
n_iter_no_change validation scores do not improve on the score n_iter_no_change+1 iterations earlier, but it
does NOT truncate the model: the stored ensemble contains the n_iter_no_change non-improving iterations
(scikit-learn 1.7.2, gradient_boosting.py, fit loop `break`). When boosting diverges numerically inside those
iterations (leaf values -G/(H+lambda) with l2_regularization=0 can explode), the returned model is evaluated
after the explosion (e.g. Paper 1, TON-IoT, 7 leaves, lr 0.1, seed 0: internal validation log-loss 0.042 at
iteration 249, 361,323 at iteration 268; test macro-F1 0.496).

rollback_best_iter(model) truncates the model in place to the iteration with the highest validation_score_
(validation_score_[0] is the baseline before any tree; validation_score_[k] the score after k iterations, so the
best k is argmax and the model keeps k iterations). Verified in test_gbdt_rollback.py: predictions of the
rolled-back model equal those of a model refitted with max_iter=k on the same data and random_state (same
internal split), and equal staged_predict_proba at iteration k of the unrolled model.

This module is NEW and is not used by any existing code path (run_p1_v2.make_tree, run_p6_v2.make_tree and the
Paper 2 scripts are unchanged); it is imported only by run_p1_final.py, bench_p1_final.py and run_p6_gb_rollback.py.
"""
import numpy as np


def best_iter(model):
    """number of iterations at the best internal-validation score (>= 1), or None without validation scores"""
    vs = np.asarray(getattr(model, 'validation_score_', []), dtype=float)
    if vs.size == 0:
        return None
    vs = np.where(np.isfinite(vs), vs, -np.inf)
    return max(1, int(np.argmax(vs)))       # first maximum; keep at least one iteration


def rollback_best_iter(model):
    """truncate a fitted HistGradientBoosting model in place to its best internal-validation iteration.

    Returns a dict with n_iter_raw (iterations fitted), best_iter (iterations kept), and the internal
    validation/training scores (negative loss for scoring='loss') at the best and at the last fitted iteration.
    Without validation scores (early_stopping off) the model is left unchanged.
    """
    n_raw = len(model._predictors)
    vs = np.asarray(getattr(model, 'validation_score_', []), dtype=float)
    ts = np.asarray(getattr(model, 'train_score_', []), dtype=float)
    k = best_iter(model)
    info = dict(n_iter_raw=int(n_raw), best_iter=int(k if k is not None else n_raw),
                val_score_best=float(vs[k]) if k is not None else None,
                val_score_last=float(vs[-1]) if vs.size else None,
                train_score_best=float(ts[k]) if (k is not None and ts.size > k) else None,
                train_score_last=float(ts[-1]) if ts.size else None,
                rolled_back=False)
    if k is None or k >= n_raw:
        return info
    model._predictors = model._predictors[:k]          # n_iter_ is len(self._predictors) (a property)
    if vs.size:
        model.validation_score_ = vs[:k + 1]
    if ts.size:
        model.train_score_ = ts[:k + 1]
    info['rolled_back'] = True
    assert model.n_iter_ == k
    return info
